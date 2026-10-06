import os 
import random
import json
from dataclasses import dataclass
import re
import warnings
import argparse
import pandas as pd
from collections import defaultdict
from itertools import product

from src.config import Config

"""
- current implementation considers only test sets for zero-shot methods
"""


def balanced_pair_splits(pairs, cfg, split_pair, rng, description):
    """Build balanced, disjoint train/validation/test splits from source pairs."""
    split_sizes = {
        "train": cfg.train_size,
        "val": cfg.val_size,
        "test": cfg.test_size,
    }

    pair_counts = {name: size // 2 for name, size in split_sizes.items()}
    total_pairs = sum(pair_counts.values())
    if len(pairs) < total_pairs:
        raise ValueError(
            f"Not enough pairs for {description}: need {total_pairs}, found {len(pairs)}."
        )

    selected_pairs = rng.sample(pairs, total_pairs)
    splits = {}
    start = 0
    for name, count in pair_counts.items():
        split_pairs = selected_pairs[start:start + count]
        start += count
        split = [sample for pair in split_pairs for sample in split_pair(pair)]
        rng.shuffle(split)
        splits[name] = split
    return splits


def detectRLX_attacks(cfg):
    """Create English-only, class-balanced detectRLX attack subsets."""
    raw_dir = os.path.join(cfg.data_raw_dir, "drlX", "Binary")
    numbered_general_attacks = sorted(
        match.group(1)
        for filename in os.listdir(raw_dir)
        if (match := re.fullmatch(r"binary_(general_\d+)_open\.json", filename))
    )
    attack_groups = {
        "perturbation": (
            "character_deletion", "character_insertion",
            "character_substitution", "zero_width_insertion",
        ),
        "paraphrasing": (
            "backtranslation", "encoder_paraphrasing",
            "decoder_paraphrasing", "seq2seq_paraphrasing",
        ),
        "structure": ("condensing", "expanding", "polishing"),
        "length": tuple(numbered_general_attacks),
    }
    output_prefix = "drlXAttacks"
    output_dir = os.path.join(cfg.data_set_dir, output_prefix)
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(cfg.seed)

    def split_pair(item, attack):
        metadata = {
            "attack": attack,
            "domain": item["domain"],
            "lang": item["lang"],
            "model": item["model"],
        }
        return [
            {**metadata, "text": item["human_written_text"], "label": 0},
            {**metadata, "text": item["llm_generated_text"], "label": 1},
        ]

    for category, attacks in attack_groups.items():
        print(f"Processing detectRLX {category} attacks: {list(attacks)}")
        for attack in attacks:
            source_path = os.path.join(raw_dir, f"binary_{attack}_open.json")
            with open(source_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            pairs = [
                item for item in data
                if item["lang"].lower() == "english"
                and item["human_written_text"].strip()
                and item["llm_generated_text"].strip()
            ]
            splits = balanced_pair_splits(
                pairs,
                cfg,
                lambda pair: split_pair(pair, attack),
                rng,
                f"attack={attack!r}",
            )
            output_path = os.path.join(
                output_dir, f"{output_prefix}_{attack}_s{cfg.seed}.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(splits, f, ensure_ascii=False, indent=4)


def detectRLX(cfg):
    """Create English-only, class-balanced detectRLX domain and model subsets."""
    source_path = os.path.join(
        cfg.data_raw_dir, "drlX", "Binary", "binary_general_open.json"
    )
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [
        item for item in data
        if item["lang"].lower() == "english"
        and item["human_written_text"].strip()
        and item["llm_generated_text"].strip()
    ]
    if not data:
        raise ValueError("No valid English human-machine pairs found in detectRLX.")

    rng = random.Random(cfg.seed)
    dimensions = ("domain", "model")

    def output_name(value):
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

    def split_pair(item):
        metadata = {
            "domain": item["domain"],
            "lang": item["lang"],
            "model": item["model"],
        }
        return [
            {**metadata, "text": item["human_written_text"], "label": 0},
            {**metadata, "text": item["llm_generated_text"], "label": 1},
        ]

    for fixed_dimension in dimensions:
        print(f"Processing detectRLX subsets for {fixed_dimension} ...")
        prefix = f"drlX{fixed_dimension.capitalize()}"
        output_dir = os.path.join(cfg.data_set_dir, prefix)
        os.makedirs(output_dir, exist_ok=True)

        elements = sorted({item[fixed_dimension] for item in data})
        for value in elements:
            pairs = [item for item in data if item[fixed_dimension] == value]
            splits = balanced_pair_splits(
                pairs,
                cfg,
                split_pair,
                rng,
                f"{fixed_dimension}={value!r}",
            )
            output_path = os.path.join(
                output_dir, f"{prefix}_{output_name(value)}_s{cfg.seed}.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(splits, f, ensure_ascii=False, indent=4)


def detectRLX_languages(cfg):
    """Create various language splits."""
    source_path = os.path.join(
        cfg.data_raw_dir, "drlX", "Binary", "binary_general_open.json"
    )
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [
        item for item in data
        if item["lang"].lower() != "english"
        and item["human_written_text"].strip()
        and item["llm_generated_text"].strip()
    ]

    rng = random.Random(cfg.seed)

    def output_name(value):
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

    def split_pair(item):
        metadata = {
            "domain": item["domain"],
            "lang": item["lang"],
            "model": item["model"],
        }
        return [
            {**metadata, "text": item["human_written_text"], "label": 0},
            {**metadata, "text": item["llm_generated_text"], "label": 1},
        ]

    print(f"Processing detectRLX subsets for languages ...")
    prefix = f"drlXLang"
    output_dir = os.path.join(cfg.data_set_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)

    elements = sorted({item["lang"] for item in data})
    for value in elements:
        pairs = [item for item in data if item["lang"] == value]
        splits = balanced_pair_splits(
            pairs,
            cfg,
            split_pair,
            rng,
            f"lang={value!r}",
        )
        output_path = os.path.join(
            output_dir, f"{prefix}_{output_name(value)}_s{cfg.seed}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, ensure_ascii=False, indent=4)


def detectRLX_mix(cfg):
    """Create English-only, class-balanced detectRLX mix subset."""
    source_path = os.path.join(
        cfg.data_raw_dir, "drlX", "Binary", "binary_general_open.json"
    )
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [
        item for item in data
        if item["lang"].lower() == "english"
        and item["human_written_text"].strip()
        and item["llm_generated_text"].strip()
    ]
    if not data:
        raise ValueError("No valid English human-machine pairs found in detectRLX.")

    rng = random.Random(cfg.seed)

    def output_name(value):
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

    def split_pair(item):
        metadata = {
            "domain": item["domain"],
            "lang": item["lang"],
            "model": item["model"],
        }
        return [
            {**metadata, "text": item["human_written_text"], "label": 0},
            {**metadata, "text": item["llm_generated_text"], "label": 1},
        ]


    print(f"Processing detectRLX subsets for mix ...")
    prefix = f"drlXMix"
    output_dir = os.path.join(cfg.data_set_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)

    splits = balanced_pair_splits(
        data,
        cfg,
        split_pair,
        rng,
        "mixed domains and generators",
    )
    output_path = os.path.join(
        output_dir, f"{prefix}_mixed_s{cfg.seed}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=4)

def detectRLX_wikipedia_en(cfg):
    """Sample 500 English Wikipedia pairs."""
    source_path = os.path.join(
        cfg.data_raw_dir, "drlX", "Binary", "binary_general_open.json"
    )
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [
        item for item in data
        if item["lang"].lower() == "english" and "wiki" in item["domain"].lower()
        and item["human_written_text"].strip()
        and item["llm_generated_text"].strip()
    ]
    print("Domains:", {item["domain"] for item in data})
    if not data:
        raise ValueError("No valid English human-machine pairs found in detectRLX.")

    rng = random.Random(cfg.seed)

    def split_pair(item, idx):
        metadata = {
            "index": idx,
            "domain": item["domain"],
            "lang": item["lang"],
            "model": item["model"],
        }
        return [
            {**metadata, "text": item["human_written_text"], "label": 0},
            {**metadata, "text": item["llm_generated_text"], "label": 1},
        ]

    selected_pairs = rng.sample(data, 250)
    all_samples = [sample for idx, pair in enumerate(selected_pairs) for sample in split_pair(pair, idx)]
    out= {'train': all_samples, 'val': all_samples, 'test': all_samples}

    prefix = f"drlXWikipedia"
    output_dir = os.path.join(cfg.data_set_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}_en_s{cfg.seed}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)

def m4(cfg):
    random.seed(cfg.seed)

    input_path = os.path.join(cfg.data_raw_dir, "m4", "wikipedia_chatgpt.jsonl")
    with open(input_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    pairs = [pair for pair in pairs if pair["human_text"].strip() and pair["machine_text"].strip()]
    n_pairs = cfg.test_size // 2
    assert len(pairs) >= n_pairs, "Not enough human-machine pairs"

    out_data = []
    for pair in random.sample(pairs, n_pairs):
        out_data.extend([
            {"text": pair["human_text"], "label": 0},
            {"text": pair["machine_text"], "label": 1},
        ])
    random.shuffle(out_data)

    folder_path = os.path.join(cfg.data_set_dir, "m4Domains")
    os.makedirs(folder_path, exist_ok=True)
    out_path = os.path.join(
        folder_path, f"m4Domains_wikipedia_s{cfg.seed}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=4)
        
def raid_paired(cfg):
    """Create 50 clean/attacked machine-text pairs for each RAID attack type."""
    input_path = os.path.join(cfg.raid_raw_dir, "train.csv")
    df = pd.read_csv(input_path, low_memory=False)

    required_columns = {
        "id", "adv_source_id", "source_id", "model", "decoding",
        "repetition_penalty", "attack", "domain", "title", "prompt",
        "generation",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"RAID train split is missing columns: {sorted(missing_columns)}")

    df = df[df["domain"].eq("wiki")].copy()
    # Keep both sides of every pair on the same non-human, greedy-decoded
    # generation setting.
    df = df[
        df["model"].astype(str).str.lower().ne("human")
        & df["decoding"].astype(str).str.lower().eq("greedy")
    ].copy()
    clean = df[df["attack"] == "none"].copy()
    attacked = df[df["attack"] != "none"].copy()
    pairs = attacked.merge(
        clean[["id", "generation", "model"]],
        left_on="adv_source_id",
        right_on="id",
        suffixes=("_perturbed", "_original"),
        validate="many_to_one",
    )

    assert (pairs["adv_source_id"] == pairs["id_original"]).all()
    assert pairs["domain"].eq("wiki").all()
    assert pairs["attack"].ne("none").all()

    pairs = pairs.dropna(subset=["generation_perturbed", "generation_original"])
    pairs = pairs[
        pairs["generation_perturbed"].str.strip().ne("")
        & pairs["generation_original"].str.strip().ne("")
    ]
    pairs = pairs[
        pairs["model_perturbed"].ne("human")
        & pairs["model_original"].ne("human")
    ].copy()
    assert pairs["model_perturbed"].ne("human").all()
    assert pairs["model_original"].ne("human").all()

    def json_value(value):
        return None if pd.isna(value) else value

    output_dir = os.path.join(cfg.data_set_dir, "raidAttack")
    os.makedirs(output_dir, exist_ok=True)
    for attack, attack_pairs in pairs.groupby("attack", sort=True):
        if len(attack_pairs) < 50:
            raise ValueError(
                f"Not enough valid RAID Wikipedia pairs for {attack!r}: "
                f"need 50, found {len(attack_pairs)}."
            )
        selected_pairs = attack_pairs.sample(n=50, random_state=cfg.seed)
        all_samples = []
        for _, pair in selected_pairs.iterrows():
            shared_metadata = {
                "pair_id": json_value(pair["id_perturbed"]),
                "original_id": json_value(pair["id_original"]),
                "attacked_id": json_value(pair["id_perturbed"]),
                "adv_source_id": json_value(pair["adv_source_id"]),
                "source_id": json_value(pair["source_id"]),
                "pair_attack": json_value(pair["attack"]),
                "domain": json_value(pair["domain"]),
                "title": json_value(pair["title"]),
                "prompt": json_value(pair["prompt"]),
                "decoding": json_value(pair["decoding"]),
                "repetition_penalty": json_value(pair["repetition_penalty"]),
            }
            all_samples.extend([
                {
                    **shared_metadata,
                    "id": json_value(pair["id_original"]),
                    "model": json_value(pair["model_original"]),
                    "attack": "none",
                    "text": pair["generation_original"],
                },
                {
                    **shared_metadata,
                    "id": json_value(pair["id_perturbed"]),
                    "model": json_value(pair["model_perturbed"]),
                    "attack": json_value(pair["attack"]),
                    "text": pair["generation_perturbed"],
                },
            ])

        assert len(all_samples) == 100
        assert sum(sample["attack"] == "none" for sample in all_samples) == 50
        assert sum(sample["attack"] == attack for sample in all_samples) == 50
        out = {"train": all_samples, "val": all_samples, "test": all_samples}
        output_path = os.path.join(output_dir, f"raidAttack_{attack}_s{cfg.seed}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

def raid_domains(cfg):
    """Create clean, balanced, disjoint RAID subsets for every domain."""
    input_path = os.path.join(cfg.raid_raw_dir, "train.csv")
    df = pd.read_csv(input_path, low_memory=False)

    required_columns = {"domain", "attack", "model", "decoding", "generation"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"RAID train split is missing columns: {sorted(missing_columns)}")

    split_sizes = {
        "train": cfg.train_size,
        "val": cfg.val_size,
        "test": cfg.test_size,
    }
    if any(size < 2 or size % 2 for size in split_sizes.values()):
        raise ValueError("RAID domain split sizes must be positive, even integers.")
    class_counts = {name: size // 2 for name, size in split_sizes.items()}
    total_per_class = sum(class_counts.values())

    clean = df[df["attack"].eq("none")].dropna(
        subset=["domain", "model", "generation"]
    ).copy()
    clean = clean[clean["generation"].astype(str).str.strip().ne("")]
    # Human controls have no decoding value; restrict only machine texts to
    # genuine greedy-decoded rows before drawing balanced samples.
    clean = clean[
        clean["model"].astype(str).str.lower().eq("human")
        | clean["decoding"].astype(str).str.lower().eq("greedy")
    ]
    if clean.empty:
        raise ValueError("No clean RAID rows with non-empty generations were found.")

    output_dir = os.path.join(cfg.data_set_dir, "raidDomain")
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(cfg.seed)
    domains = sorted(clean["domain"].unique())

    for domain in domains:
        domain_rows = clean[clean["domain"].eq(domain)]
        human_rows = domain_rows[
            domain_rows["model"].astype(str).str.lower().eq("human")
        ]
        machine_rows = domain_rows[
            domain_rows["model"].astype(str).str.lower().ne("human")
        ]
        if len(human_rows) < total_per_class or len(machine_rows) < total_per_class:
            raise ValueError(
                f"Not enough clean RAID rows for domain={domain!r}: need "
                f"{total_per_class} human and machine rows, found "
                f"{len(human_rows)} human and {len(machine_rows)} machine."
            )

        human_indices = rng.sample(human_rows.index.tolist(), total_per_class)
        machine_indices = rng.sample(machine_rows.index.tolist(), total_per_class)
        splits = {}
        start = 0
        for split_name, count in class_counts.items():
            stop = start + count
            split = []
            for index in human_indices[start:stop]:
                row = clean.loc[index]
                split.append({
                    "text": str(row["generation"]),
                    "label": 0,
                    "domain": str(domain),
                    "model": "human",
                    "attack": "none",
                })
            for index in machine_indices[start:stop]:
                row = clean.loc[index]
                split.append({
                    "text": str(row["generation"]),
                    "label": 1,
                    "domain": str(domain),
                    "model": str(row["model"]),
                    "sampling": "greedy",
                    "attack": "none",
                })
            rng.shuffle(split)
            splits[split_name] = split
            start = stop

        output_path = os.path.join(
            output_dir, f"raidDomain_{str(domain).lower()}_s{cfg.seed}.json"
        )
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(splits, file, ensure_ascii=False, indent=4)


def raid_models(cfg):
    """Create clean, balanced, disjoint RAID subsets for every model."""
    input_path = os.path.join(cfg.raid_raw_dir, "train.csv")
    df = pd.read_csv(input_path, low_memory=False)

    required_columns = {"domain", "attack", "model", "decoding", "generation"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"RAID train split is missing columns: {sorted(missing_columns)}")

    split_sizes = {
        "train": cfg.train_size,
        "val": cfg.val_size,
        "test": cfg.test_size,
    }
    if any(size < 2 or size % 2 for size in split_sizes.values()):
        raise ValueError("RAID model split sizes must be positive, even integers.")
    class_counts = {name: size // 2 for name, size in split_sizes.items()}
    total_per_class = sum(class_counts.values())

    clean = df[df["attack"].eq("none")].dropna(
        subset=["domain", "model", "generation"]
    ).copy()
    clean = clean[clean["generation"].astype(str).str.strip().ne("")]
    # Human controls have no decoding value; restrict only machine texts to
    # genuine greedy-decoded rows before drawing balanced samples.
    clean = clean[
        clean["model"].astype(str).str.lower().eq("human")
        | clean["decoding"].astype(str).str.lower().eq("greedy")
    ]
    human_rows = clean[clean["model"].astype(str).str.lower().eq("human")]
    if len(human_rows) < total_per_class:
        raise ValueError(
            f"Not enough clean human RAID rows: need {total_per_class}, "
            f"found {len(human_rows)}."
        )

    output_dir = os.path.join(cfg.data_set_dir, "raidModel")
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(cfg.seed)
    models = sorted(
        model for model in clean["model"].unique()
        if str(model).lower() != "human"
    )

    for model in models:
        machine_rows = clean[clean["model"].eq(model)]
        if len(machine_rows) < total_per_class:
            raise ValueError(
                f"Not enough clean RAID rows for model={model!r}: need "
                f"{total_per_class}, found {len(machine_rows)}."
            )

        human_indices = rng.sample(human_rows.index.tolist(), total_per_class)
        machine_indices = rng.sample(machine_rows.index.tolist(), total_per_class)
        splits = {}
        start = 0
        for split_name, count in class_counts.items():
            stop = start + count
            split = []
            for index in human_indices[start:stop]:
                row = clean.loc[index]
                split.append({
                    "text": str(row["generation"]),
                    "label": 0,
                    "domain": str(row["domain"]),
                    "model": "human",
                    "attack": "none",
                })
            for index in machine_indices[start:stop]:
                row = clean.loc[index]
                split.append({
                    "text": str(row["generation"]),
                    "label": 1,
                    "domain": str(row["domain"]),
                    "model": str(model),
                    "sampling": "greedy",
                    "attack": "none",
                })
            rng.shuffle(split)
            splits[split_name] = split
            start = stop

        output_name = re.sub(r"[^a-z0-9]+", "_", str(model).lower()).strip("_")
        output_path = os.path.join(
            output_dir, f"raidModel_{output_name}_s{cfg.seed}.json"
        )
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(splits, file, ensure_ascii=False, indent=4)


def ntsraid(cfg):
    """Create per-domain NTS-RAID datasets from every valid test record."""
    input_path = os.path.join(cfg.data_raw_dir, "nts_bench", "raid_test.json")
    with open(input_path, encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {input_path}.")

    records_by_domain = defaultdict(list)
    for record in records:
        domain = record.get("domain")
        text = record.get("generation")
        source = record.get("content_source")
        if not isinstance(domain, str) or not domain:
            raise ValueError("NTS-RAID contains a record without a valid domain.")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"NTS-RAID record in domain {domain!r} lacks content_source."
            )

        is_human = source == "human"
        model = "human" if is_human else source.removeprefix("machine:")
        records_by_domain[domain].append({
            "id": record.get("id"),
            "text": text,
            "label": 0 if is_human else 1,
            "domain": domain,
            "model": model,
            "content_source": source,
        })

    output_dir = os.path.join(cfg.data_set_dir, "ntsraidDomain")
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for domain, samples in sorted(records_by_domain.items()):
        human_count = sum(sample["label"] == 0 for sample in samples)
        non_human_count = len(samples) - human_count
        if not human_count or not non_human_count:
            raise ValueError(
                f"NTS-RAID domain {domain!r} needs both human and non-human "
                f"records; found {human_count} human and {non_human_count} non-human."
            )

        splits = {"train": samples, "val": samples, "test": samples}
        output_name = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
        output_path = os.path.join(
            output_dir, f"ntsraidDomain_{output_name}_s{cfg.seed}.json"
        )
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(splits, file, ensure_ascii=False, indent=4)
        output_paths.append(output_path)

    return output_paths


def sampling_data(cfg):
    """Create matched human/LLaMA sampling-parameter evaluation datasets."""
    sampling_dir = os.path.join(cfg.data_raw_dir, "sampling")
    group_specs = {
        "temperature": "samplingTemperature",
        "top-k": "samplingTopK",
        "top-p": "samplingTopP",
        "eta-sampling": "samplingEta",
        "repetition-penalty": "samplingRepetitionPenalty",
    }
    file_pattern = re.compile(
        r"Llama-3\.2-3B-Instruct_(temperature|top-k|top-p|eta-sampling|repetition-penalty)_(.+)\.json"
    )

    def load_records(path):
        with open(path, encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {path}.")
        return records

    def records_by_id(records, description, skip_empty_text=False):
        indexed = {}
        for record in records:
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{description} contains a row without a valid id.")
            if identifier in indexed:
                raise ValueError(f"{description} contains duplicate id {identifier!r}.")
            if not isinstance(record.get("text"), str) or not record["text"].strip():
                if skip_empty_text:
                    continue
                raise ValueError(f"{description} contains an empty text for id {identifier!r}.")
            indexed[identifier] = record
        return indexed

    mistral_by_id = records_by_id(
        load_records(os.path.join(sampling_dir, "mistral-chat.json")),
        "mistral-chat.json",
    )
    human_by_source_id = records_by_id(
        load_records(os.path.join(sampling_dir, "human.json")),
        "human.json",
    )

    grouped_files = defaultdict(list)
    for filename in sorted(os.listdir(sampling_dir)):
        match = file_pattern.fullmatch(filename)
        if match is not None:
            grouped_files[match.group(1)].append((match.group(2), filename))
    missing_groups = set(group_specs).difference(grouped_files)
    if missing_groups:
        raise ValueError(f"Missing sampling files for groups: {sorted(missing_groups)}")

    machine_by_file = {}
    for files in grouped_files.values():
        for value, filename in files:
            machine_by_file[filename] = records_by_id(
                load_records(os.path.join(sampling_dir, filename)), filename,
                skip_empty_text=True,
            )

    common_ids = set(mistral_by_id)
    for records in machine_by_file.values():
        common_ids.intersection_update(records)

    # Map each LLaMA-file ID to its human source via mistral-chat.json.
    human_by_llama_id = {}
    for identifier in common_ids:
        source_id = mistral_by_id[identifier].get("source_id")
        human_record = human_by_source_id.get(source_id)
        if human_record is not None:
            human_by_llama_id[identifier] = {
                "id": identifier,
                "source_id": source_id,
                "text": human_record["text"],
            }

    if cfg.test_size < 2 or cfg.test_size % 2:
        raise ValueError("cfg.test_size must be a positive, even integer.")
    sample_size = cfg.test_size // 2
    if len(human_by_llama_id) < sample_size:
        raise ValueError(
            f"Only {len(human_by_llama_id)} LLaMA IDs map to human texts; "
            f"need {sample_size}."
        )
    selected_ids = random.Random(cfg.seed).sample(
        sorted(human_by_llama_id), sample_size
    )

    for group, output_folder in group_specs.items():
        output_dir = os.path.join(cfg.data_set_dir, output_folder)
        os.makedirs(output_dir, exist_ok=True)
        for value, filename in grouped_files[group]:
            machine_records = machine_by_file[filename]
            samples = []
            for identifier in selected_ids:
                human_record = human_by_llama_id[identifier]
                machine_record = machine_records[identifier]
                samples.extend([
                    {
                        "id": identifier,
                        "source_id": human_record["source_id"],
                        "text": human_record["text"],
                        "label": 0,
                        "model": "human",
                    },
                    {
                        "id": identifier,
                        "source_id": human_record["source_id"],
                        "text": machine_record["text"],
                        "label": 1,
                        "model": "Llama-3.2-3B-Instruct",
                        "sampling_group": group,
                        "sampling_value": value,
                    },
                ])

            output_name = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
            output_path = os.path.join(
                output_dir, f"{output_folder}_{output_name}_s{cfg.seed}.json"
            )
            with open(output_path, "w", encoding="utf-8") as file:
                out = {"train": samples, "val": samples, "test": samples}
                json.dump(out, file, ensure_ascii=False, indent=4)

def editlens(cfg):
    """Create balanced, disjoint EditLens splits for every source."""
    input_path = os.path.join(
        cfg.data_raw_dir, "editlens", "data", "train-00000-of-00001.parquet"
    )
    required_columns = {
        "text_id", "text", "text_type", "model", "source", "source_id",
        "cosine_score", "soft_ngrams_score", "split",
    }
    df = pd.read_parquet(input_path)
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"EditLens data is missing columns: {sorted(missing_columns)}")

    split_sizes = {"train": cfg.train_size, "val": cfg.val_size, "test": cfg.test_size}
    if any(size < 2 or size % 2 for size in split_sizes.values()):
        raise ValueError("EditLens split sizes must be positive, even integers.")
    class_counts = {name: size // 2 for name, size in split_sizes.items()}
    total_per_class = sum(class_counts.values())

    human_type = "human_written"
    machine_types = {"ai_generated"} # , "ai_edited"
    df = df.dropna(subset=["text", "text_type", "source"]).copy()
    df = df[df["text"].astype(str).str.strip().ne("")]
    df = df[df["text_type"].isin(machine_types | {human_type})]
    if df.empty:
        raise ValueError("No valid human or machine EditLens rows were found.")

    output_dir = os.path.join(cfg.data_set_dir, "editlens")
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(cfg.seed)

    def to_sample(row, label):
        return {
            "id": str(row["text_id"]), "source_id": str(row["source_id"]),
            "source": str(row["source"]), "text": str(row["text"]),
            "label": label, "text_type": str(row["text_type"]),
            "model": str(row["model"]),
            "cosine_score": None if pd.isna(row["cosine_score"]) else float(row["cosine_score"]),
            "soft_ngrams_score": None if pd.isna(row["soft_ngrams_score"]) else float(row["soft_ngrams_score"]),
            "original_split": str(row["split"]),
        }

    for source in sorted(df["source"].unique()):
        source_rows = df[df["source"].eq(source)]
        human_rows = source_rows[source_rows["text_type"].eq(human_type)]
        machine_rows = source_rows[source_rows["text_type"].isin(machine_types)]
        if len(human_rows) < total_per_class or len(machine_rows) < total_per_class:
            raise ValueError(
                f"Not enough EditLens rows for source={source!r}: need "
                f"{total_per_class} human and machine rows, found "
                f"{len(human_rows)} human and {len(machine_rows)} machine."
            )

        human_indices = rng.sample(human_rows.index.tolist(), total_per_class)
        machine_indices = rng.sample(machine_rows.index.tolist(), total_per_class)
        splits = {}
        start = 0
        for split_name, count in class_counts.items():
            stop = start + count
            samples = [to_sample(df.loc[index], 0) for index in human_indices[start:stop]]
            samples += [to_sample(df.loc[index], 1) for index in machine_indices[start:stop]]
            rng.shuffle(samples)
            splits[split_name] = samples
            start = stop

        output_name = re.sub(r"[^a-z0-9]+", "_", str(source).lower()).strip("_")
        output_path = os.path.join(output_dir, f"editlens_{output_name}_s{cfg.seed}.json")
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(splits, file, ensure_ascii=False, indent=4)

def detectRLX_domain_length_balanced(cfg, length=500, unit="tokens", n_per_class=100, model="l8b"):
    """
    English-only detectRLX domain subsets of n_per_class human and n_per_class machine texts, each with at least
    length units and truncated to exactly length units; unit is "tokens" (model's tokenizer, no special tokens) or
    "chars". Human and machine texts are drawn independently (not as pairs); only a test split is written, with the
    unit as file name suffix (e.g. drlXDomainLen_wiki_500_tokens_s42.json).
    """
    if unit not in ("tokens", "chars"):
        raise ValueError(f"unit must be 'tokens' or 'chars', not {unit!r}.")
    from transformers import AutoTokenizer

    source_path = os.path.join(
        cfg.data_raw_dir, "drlX", "Binary", "binary_general_open.json"
    )
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = [
        item for item in data
        if item["lang"].lower() == "english"
        and item["human_written_text"].strip()
        and item["llm_generated_text"].strip()
    ]
    if not data:
        raise ValueError("No valid English human-machine pairs found in detectRLX.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dict[model])
    rng = random.Random(cfg.seed)
    prefix = "drlXDomainLen"
    output_dir = os.path.join(cfg.data_set_dir, prefix)
    os.makedirs(output_dir, exist_ok=True)

    def output_name(value):
        return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")

    def truncated(text):
        """
        The text cut to exactly length units, or None if it is shorter (or, for tokens, does not re-tokenize
        exactly).
        """
        if unit == "chars":
            return text[:length] if len(text) >= length else None
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < length:
            return None
        text = tokenizer.decode(ids[:length])
        # Decoding can merge tokens at the cut, so keep only texts that re-tokenize to exactly length tokens.
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) != length:
            return None
        return text

    def draw(items, text_key, label, domain):
        """n_per_class random texts of at least length units, truncated to length; None if too few."""
        # The same human text can occur in several pairs (one per generator), so deduplicate by text.
        unique = list({item[text_key]: item for item in items}.values())
        rng.shuffle(unique)
        samples = []
        for item in unique:
            text = truncated(item[text_key])
            if text is None:
                continue
            samples.append({
                "domain": item["domain"],
                "lang": item["lang"],
                "model": item["model"],
                "text": text,
                "label": label,
            })
            if len(samples) == n_per_class:
                return samples
        warnings.warn(
            f"Not enough {'human' if label == 0 else 'machine'} texts with >= {length} {unit} for "
            f"domain={domain!r}: need {n_per_class}, found {len(samples)} of {len(unique)}; skipping the domain."
        )
        return None

    for domain in sorted({item["domain"] for item in data}):
        print(f"Processing detectRLX domain {domain!r} at {length} {unit} ...")
        pairs = [item for item in data if item["domain"] == domain]
        human = draw(pairs, "human_written_text", 0, domain)
        machine = draw(pairs, "llm_generated_text", 1, domain)
        if human is None or machine is None:
            continue
        samples = human + machine
        rng.shuffle(samples)
        output_path = os.path.join(
            output_dir, f"{prefix}_{output_name(domain)}_{length}_{unit}_s{cfg.seed}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"test": samples}, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    
    if args.name == "detectRLX":
        detectRLX(cfg=cfg)

    elif args.name == "detectRLX_attacks":
        detectRLX_attacks(cfg=cfg)

    elif args.name == "detectRLX_languages":
        detectRLX_languages(cfg=cfg)

    elif args.name == "detectRLX_mix":
        detectRLX_mix(cfg=cfg)

    elif args.name == "detectRLX_wikipedia_en":
        detectRLX_wikipedia_en(cfg=cfg)

    elif args.name == "m4":
        m4(cfg=cfg)

    elif args.name == "raid_paired":
        raid_paired(cfg=cfg)

    elif args.name == "raid_domains":
        raid_domains(cfg=cfg)

    elif args.name == "raid_models":
        raid_models(cfg=cfg)

    elif args.name == "ntsraid":
        ntsraid(cfg=cfg)

    elif args.name == "sampling_data":
        sampling_data(cfg=cfg)

    elif args.name == "editlens":
        editlens(cfg=cfg)

    elif args.name == "detectRLX_domain_length_balanced":
        detectRLX_domain_length_balanced(cfg=cfg, length=500, unit="tokens")
        detectRLX_domain_length_balanced(cfg=cfg, length=2000, unit="chars")

    elif args.name == "all":
        detectRLX(cfg=cfg)
        detectRLX_attacks(cfg=cfg)
        detectRLX_languages(cfg=cfg)
        detectRLX_mix(cfg=cfg)
        # m4(cfg=cfg)

    else:
        raise ValueError(f"Unknown name: {args.name}")
