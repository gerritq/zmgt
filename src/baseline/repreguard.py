import argparse
import logging
from src.baseline.repreguard_detector import AIHumanFunctionModel
import numpy as np
import logging
import json
# from src.baseline.repreguard.metrics import get_roc_by_threshold,get_roc_metrics
import os
from datetime import datetime
from collections import defaultdict

# my packages
from argparse import Namespace
from datasets import Dataset
import sys

# def process_eval(args,train_json_data, test_json_data,test_data_path):
#     print(f"Eval in {args.train_data_path}")
#     # with open(train_filepath, 'r') as json_file:
#     #     train_dataset_result = json.load(json_file)
#     real_preds = []
#     sample_preds = []

#     for item in train_json_data:
#         if item["train_input_label"] == 0:
#             real_preds.append(np.mean(item['rep_reader_scores_dict']))
#         elif item["train_input_label"] == 1:
#             sample_preds.append(np.mean((item['rep_reader_scores_dict'])))
        
#     roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy,tpr_at_fpr_0_01 = get_roc_metrics(real_preds,sample_preds)

#     train_result = {
#             "roc_auc": roc_auc,
#             "optimal_threshold": optimal_threshold,
#             "conf_matrix": conf_matrix,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "accuracy": accuracy,
#             "tpr_at_fpr_0_01": tpr_at_fpr_0_01
#         }
#     print(f"Train result: {train_result}")

#     print(f"Eval in {test_data_path}")
#     real_preds = []
#     sample_preds = []

#     for item in test_json_data:
#         if item["test_input_label"] == 0:
#             real_preds.append(np.mean(item['rep_reader_scores_dict']))
#         elif item["test_input_label"] == 1:
#             sample_preds.append(np.mean((item['rep_reader_scores_dict'])))
        
#     roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy,tpr_at_fpr_0_01 = get_roc_by_threshold(real_preds,
#                                                                                             sample_preds,threshold=optimal_threshold)
#     test_result = {
#             "auroc": roc_auc,
#             "optimal_threshold": optimal_threshold,
#             "conf_matrix": conf_matrix,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "accuracy": accuracy,
#             "tpr_at_fpr_0_01": tpr_at_fpr_0_01
#         }

#     return train_result,test_result

class RepreGuard:
    def __init__(self,
                 base_model: str = "meta-llama/Llama-3.1-8B"):
        self.model = AIHumanFunctionModel(
            model_name_or_path=base_model,
            ntrain=0,  # Only needed for bootstrapping, which we do not use.
            rep_token=0.1,
            batch_size=8,
            random_seed=42,
        )


    def run(self, data: list[dict]) -> list[float]:
        """
        Their code is a bit confusing but the pipeline is
        1) fit pca on train via process_train_data
        2) score test via process_test_data; this returns a list of dicts with the scores
        """
        # we us val rather than train as method works well for small samples
        train_json_data = self.model.process_train_data(train_data=data['val'])

        test_json_data = self.model.process_test_data(test_data=data["test"])

        scores: list[float | None] = [None] * len(data['test'])
        for item in test_json_data:
            scores[item["source_index"]] = item["rep_reader_scores_dict"]

        if any(score is None for score in scores):
            raise RuntimeError("RepReGuard did not produce a score for every input item.")

        return [float(score) for score in scores], [item["label"] for item in data['test']]

    # def run(self,
    #         args: Namespace,
    #         training_data: Dataset,
    #         ood_data: list[Dataset]
    #         ) -> None:
        
    #     # process
    #     train_json_data = self.model.process_train_data(train_data=training_data['train'])
    #     scores = [x['rep_reader_scores_dict'] for x in train_json_data]

    #     return scores
        
    #     train_result, test_result = process_eval(args,train_json_data, test_json_data,test_data_path)

    #     for ood_dict in ood_data:
    #         ood_ds = ood_dict['data']
    #         ood_name = ood_dict['name']

    #         print("Processing OOD dataset: ", ood_name)

    #         test_data_path = "skip"
    #         args.train_data_path = "skip"

    #         test_json_data = self.model.process_test_data(test_data=ood_ds['test'])

    #         train_result, test_result = process_eval(args,train_json_data, test_json_data,test_data_path)
    #         result = {"train_result": train_result, "test_result": test_result}
            
    #         # save here
    #         args_copy = Namespace(**vars(args))  
    #         out_args = return_args(args_copy)
    #         out_args['target_dataset'] = ood_name
    #         out_args['datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
    #         out = {"args": out_args, 
    #                "train_results": result,
    #                "metrics": test_result}
            
    #         length_suffix = f"_L{args.max_chars}" if getattr(args, "max_chars", -1) > 0 else ""
    #         file_name = f"{args.model}_{args.dataset}_2_{ood_name}_N{args.training_size}_S{args.seed}{length_suffix}.json"
    #         self.out_dir = os.path.join(BASE_DIR, "output", "baseline", args.folder)
    #         os.makedirs(self.out_dir, exist_ok=True)
    #         with open(os.path.join(self.out_dir, file_name), "w") as f:
    #             json.dump(out, f, indent=2)
