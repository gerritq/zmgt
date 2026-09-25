import argparse
from argparse import Namespace
import json
import os
from datetime import datetime

from src.utils import load_data, evaluation

from src.config import Config
cfg = Config()

def run(args: Namespace) -> None:
    # data 
    data = load_data(args=args)

    if args.model == "binoculars":
        from src.baseline.binoculars import Binoculars
        model = Binoculars()

    elif args.model == "nts":
        from src.baseline.nts import TeSeN
        model = TeSeN()

    elif args.model == "fdgpt":
        from src.baseline.fdgpt import FastDetectGPT
        model = FastDetectGPT()

    elif args.model == "rank":
        from src.baseline.rank import Rank
        model = Rank()

    elif args.model == "entropy":
        from src.baseline.entropy import Entropy
        model = Entropy()

    elif args.model == "llr":
        from src.baseline.llr import LLR
        model = LLR(method="llr")

    elif args.model == "likelihood":
        from src.baseline.llr import LLR
        model = LLR(method="likelihood")

    elif args.model == "lastde":
        from src.baseline.lastde_doubleplus import LastdeDoublePlus
        model = LastdeDoublePlus()

    elif args.model == "gecscore":
        from src.baseline.gecscore import GECScore
        model = GECScore()

    elif args.model == "detectllm":
        from src.baseline.DNADetectLLM import DetectLLM
        model = DetectLLM()

    elif args.model == "revise":
        from src.baseline.revise import ReviseDetect
        model = ReviseDetect()

    elif args.model == "irm":
        from src.baseline.irm import IRM
        model = IRM()

    elif args.model == "curvature":
        from src.baseline.geometric import Curvature
        model = Curvature()

    elif args.model == "repreguard":
        from src.baseline.repreguard import RepreGuard
        model = RepreGuard()

    elif args.model == "id":
        from src.baseline.id import IDEstimator
        model = IDEstimator()

    elif args.model == "text_fluoroscopy":
        from src.baseline.text_fluoroscopy import TextFluoroscopy
        model = TextFluoroscopy()
    
    elif args.model == "radar":
        from src.baseline.supervised_model import SupervisedModel
        model = SupervisedModel(model_name="TrustSafeAI/RADAR-Vicuna-7B")
    
    elif args.model == "openai_roberta":
        from src.baseline.supervised_model import SupervisedModel
        model = SupervisedModel(model_name="openai-community/roberta-base-openai-detector")
    
    elif args.model == "editlens":
        from src.baseline.supervised_model import SupervisedModel
        model = SupervisedModel(model_name="pangram/editlens_roberta-large")

    else:
        raise ValueError(f"Invalid model: {args.model}")

    # run model
    y_scores, y_true = model.run(data=data)

    # evaluation
    # y_true = [item["label"] for item in data['test']]
    metrics = evaluation(y_true=y_true, y_score=y_scores)

    # save results
    output = {
        **vars(args),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    if hasattr(model, "settings"):
        output["settings"] = model.settings

    with open(os.path.join(cfg.baseline_output_dir, f"{args.model}_{args.dataset}_s{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(args=args)

if __name__ == "__main__":
    main()
