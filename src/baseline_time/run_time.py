"""
Runtime analysis: per-text latency and GPU memory of the zero-shot detectors, all at batch size 1.

Models (--model): lastde, detectllm, binoculars, fdgpt, irm (copies of src/baseline with score(text) for one text) and
joint (our score, src.baseline_time.joint). Each is loaded once; loading is timed but not part of the latency.

Per text, on the --split texts of --dataset (all, or the first --n_texts):
    latency_ms      wall-clock time of model.score(text): tokenisation, forward passes and the score, with all GPUs
                    synchronised before the start and after the end (GPU work is asynchronous)
    peak_memory_mb  peak GPU memory allocated by PyTorch while scoring that text (summed over GPUs; the peak counter
                    is reset before each text). It includes the model weights; extra_memory_mb = peak − the memory
                    allocated right after loading (activations, logits and other temporary tensors)
N_WARMUP texts (the first ones) are scored once before the timed loop and not recorded (the first passes are slow).
The RNGs are seeded with --seed before the timed loop (lastde samples).

Output: output/time_space/{model}_{dataset}_s{seed}.json with the settings, the environment (GPU, versions), every
model component's dtype and parameter count, the load time, the memory after loading, a summary (mean, std, median,
p95 of latency; mean, std, max of peak and extra memory), the detection metrics of the scores as a sanity check
(src.utils.evaluation, NaN scores dropped) and the per-text values.

    PYTHONPATH="${BASE_ZMGT}" uv run -m src.baseline_time.run_time --model joint \
        --dataset drlXDomainLen_wiki_300_tokens
"""

import json
import os
import random
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.utils import evaluation, load_data

cfg = Config()
OUTPUT_DIR = os.path.join(cfg.base_dir, "output", "time_space")
N_WARMUP = 5
MODELS = ("lastde", "detectllm", "binoculars", "fdgpt", "irm", "joint")
MB = 1024 ** 2


def load_model(name: str):
    """The detector with a score(text) -> float method (higher = machine); imported lazily (different venvs)."""
    if name == "lastde":
        from src.baseline_time.lastde_doubleplus import LastdeDoublePlus
        return LastdeDoublePlus()
    if name == "detectllm":
        from src.baseline_time.DNADetectLLM import DetectLLM
        return DetectLLM()
    if name == "binoculars":
        from src.baseline_time.binoculars import Binoculars
        return Binoculars()
    if name == "fdgpt":
        from src.baseline_time.fdgpt import FastDetectGPT
        return FastDetectGPT()
    if name == "irm":
        from src.baseline_time.irm import IRM
        return IRM()
    if name == "joint":
        from src.baseline_time.joint import JointScore
        return JointScore()
    raise ValueError(f"Invalid model: {name}")


def gpus() -> range:
    return range(torch.cuda.device_count())


def synchronize() -> None:
    for d in gpus():
        torch.cuda.synchronize(d)


def reset_peak() -> None:
    for d in gpus():
        torch.cuda.reset_peak_memory_stats(d)


def allocated_mb() -> float:
    return sum(torch.cuda.memory_allocated(d) for d in gpus()) / MB


def peak_mb() -> float:
    return sum(torch.cuda.max_memory_allocated(d) for d in gpus()) / MB


def components(model) -> dict[str, dict]:
    """dtype and parameter count of every torch module held by the detector (e.g. scoring and reference model)."""
    out = {}
    for name, value in vars(model).items():
        if isinstance(value, torch.nn.Module):
            params = list(value.parameters())
            out[name] = {"dtype": str(params[0].dtype) if params else None,
                         "n_params": int(sum(p.numel() for p in params))}
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "min": float(values.min()), "max": float(values.max())}


def run(args: Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The runtime analysis needs a GPU.")
    data = load_data(args=args)[args.split]
    if args.n_texts:
        data = data[:args.n_texts]
    texts = [item["text"] for item in data]
    labels = np.asarray([int(item["label"]) for item in data])

    synchronize()
    start = time.perf_counter()
    model = load_model(args.model)
    synchronize()
    load_time_s = time.perf_counter() - start
    static_mb = allocated_mb()
    print(f"{args.model}: loaded in {load_time_s:.1f} s, {static_mb:.0f} MB allocated on {torch.cuda.device_count()} GPU(s)")

    for text in texts[:N_WARMUP]:
        model.score(text)

    seed_everything(args.seed)
    scores, latency_ms, peak_memory_mb = [], [], []
    for text in tqdm(texts, desc=f"Timing {args.model}"):
        synchronize()
        reset_peak()
        start = time.perf_counter()
        value = model.score(text)
        synchronize()
        latency_ms.append((time.perf_counter() - start) * 1000)
        peak_memory_mb.append(peak_mb())
        scores.append(value)

    latency_ms, peak_memory_mb, scores = map(np.asarray, (latency_ms, peak_memory_mb, scores))
    extra_memory_mb = peak_memory_mb - static_mb
    valid = np.isfinite(scores)
    metrics = evaluation(labels[valid], scores[valid]) if len(np.unique(labels[valid])) == 2 else None

    import transformers
    return {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "environment": {"gpu": torch.cuda.get_device_name(0), "n_gpus": torch.cuda.device_count(),
                        "torch": torch.__version__, "transformers": transformers.__version__},
        "components": components(model),
        "settings": getattr(model, "settings", None),
        "batch_size": 1,
        "n_texts": len(texts),
        "n_warmup": N_WARMUP,
        "n_nan_scores": int((~valid).sum()),
        "load_time_s": load_time_s,
        "memory_after_load_mb": static_mb,
        "summary": {"latency_ms": stats(latency_ms), "peak_memory_mb": stats(peak_memory_mb),
                    "extra_memory_mb": stats(extra_memory_mb)},
        "metrics": metrics,
        "per_text": {"latency_ms": latency_ms.tolist(), "peak_memory_mb": peak_memory_mb.tolist(),
                     "score": [None if not np.isfinite(s) else float(s) for s in scores],
                     "label": labels.tolist()},
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_texts", type=int, default=0, help="only the first n texts (0 = all)")
    args, _ = parser.parse_known_args()  # lastde parses its own flags from sys.argv

    output = run(args)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{args.model}_{args.dataset}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    s = output["summary"]
    print(f"{args.model}: latency {s['latency_ms']['mean']:.1f} ± {s['latency_ms']['std']:.1f} ms, "
          f"peak memory {s['peak_memory_mb']['mean']:.0f} ± {s['peak_memory_mb']['std']:.0f} MB "
          f"(extra {s['extra_memory_mb']['mean']:.0f} MB) over {output['n_texts']} texts")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
