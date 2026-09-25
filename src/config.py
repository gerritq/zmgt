import os
from dataclasses import dataclass, field

@dataclass
class Config:
    train_size: int = 600
    test_size: int = 300
    val_size: int = 300
    seed: int = 42
    base_dir: str = os.getenv("BASE_ZMGT")
    zero_dir: str = os.getenv("BASE_ZERO")
    data_raw_dir: str = os.path.join(zero_dir, "data", "raw")
    # special dir for raid bc it's so large
    raid_raw_dir: str = "/scratch/prj/inf_nlg_ai_detection/coe/data/raw/raid"
    
    
    data_set_dir: str = os.path.join(zero_dir, "data", "sets")

    baseline_output_dir: str = os.path.join(base_dir, "output", "baselines", "sandbox")
    zero_output_dir: str = os.path.join(base_dir, "output", "zero")
    item_output_dir: str = os.path.join(base_dir, "output", "items")

    
    model_dict: dict[str, str] = field(default_factory=lambda: {
        "q1.7b": "Qwen/Qwen3-1.7B",
        "q4b": "Qwen/Qwen3-4B",
        "q8b": "Qwen/Qwen3-8B",
        "l1b": "meta-llama/Llama-3.2-1B-Instruct",
        "l3b": "meta-llama/Llama-3.2-3B-Instruct",
        "l8b": "meta-llama/Llama-3.1-8B-Instruct",
        "g1b": "google/gemma-3-1b-it",
        "g4b": "google/gemma-3-4b-it",
        "g12b": "google/gemma-3-12b-it"

    })

    os.makedirs(data_set_dir, exist_ok=True)
    os.makedirs(baseline_output_dir, exist_ok=True)
    os.makedirs(zero_output_dir, exist_ok=True)
    os.makedirs(item_output_dir, exist_ok=True)


