import os
import json
import numpy as np
import torch
import re
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, f1_score
from argparse import Namespace

from src.config import Config
cfg = Config()

def load_data(args: Namespace):
    
    folder_name = args.dataset.split("_")[0]
    data_dir = os.path.join(cfg.data_set_dir, folder_name)

    with open(os.path.join(data_dir, f"{args.dataset}_s{args.seed}.json"), "r") as f:
        data = json.load(f)
    
    return data

def evaluation(y_true: list[float],
               y_score: list[float]):

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n_valid = int(np.sum(~np.isnan(y_score) & (y_score != 0)))

    auroc = float(roc_auc_score(y_true, y_score))
    aupr = float(average_precision_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    mask_fpr_0_05 = fpr <= 0.05
    tpr_at_fpr_0_05 = float(np.max(tpr[mask_fpr_0_05])) if np.any(mask_fpr_0_05) else 0.0
    mask_fpr_0_01 = fpr <= 0.01
    tpr_at_fpr_0_01 = float(np.max(tpr[mask_fpr_0_01])) if np.any(mask_fpr_0_01) else 0.0

    # Detector scores are not calibrated to a common operating threshold.
    # Report F1 and accuracy at the threshold that maximizes F1 on this split.
    f1_by_threshold = [
        f1_score(y_true, y_score >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    best_idx = int(np.argmax(f1_by_threshold))
    y_pred = (y_score >= thresholds[best_idx]).astype(int)
    f1 = float(f1_by_threshold[best_idx])
    accuracy = float(np.mean(y_pred == y_true))

    return {
        "n_total": len(y_true),
        "n_valid": n_valid,
        "auroc": auroc,
        "tpr_at_fpr_0_05": tpr_at_fpr_0_05,
        "tpr_at_fpr_0_01": tpr_at_fpr_0_01,
        "aupr": aupr,
        "f1": f1,
        "accuracy": accuracy,
    }


def return_device():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    return DEVICE
