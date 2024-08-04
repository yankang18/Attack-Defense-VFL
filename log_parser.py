import json

import pandas as pd

from store_utils import get_path_slash, ExperimentResultStructure, ExperimentTaskCategoryDirName

if __name__ == '__main__':

    # === using specific dataset_dir_name (e.g., default_credit) or '*' to create Excel result.
    dataset_dir_name = "*"

    # === the relative location of the exp_result directory.
    loc = "."
    # task_category = ExperimentTaskCategoryDirName.MODEL_COMPLETE_DIR_NAME
    task_category = ExperimentTaskCategoryDirName.MAIN_TASK_DIR_NAME
    dataset_dirs = ExperimentResultStructure.search_subdirs_of_task_category_dir(task_category, dataset_dir_name, loc)

    splitter = get_path_slash()
    print("dataset_dirs:", dataset_dirs)
    for data_dir in dataset_dirs:
        dataset_name = data_dir.split(splitter)[-1]
        exp_list = ExperimentResultStructure.search_exp_results_of_attacks_by_dir(data_dir)
        record_list = list()
        ordered_cols = list()
        print("exp_list:", exp_list)
        for exp in exp_list:
            print(exp)
            with open(exp, "r") as f:
                exp_result = json.loads(f.read())
                print(exp_result)
            result_to_save = dict()
            result_to_save["dataset"] = exp_result["dataset"]
            # data_name = result_to_save["dataset"]
            result_to_save["arch_config_name"] = exp_result["arch_config_name"]
            result_to_save["task_model_type"] = exp_result["task_model_type"]
            result_to_save["optimizer_name"] = exp_result["optimizer_name"]
            result_to_save["lr"] = exp_result["lr"]
            result_to_save["wd"] = exp_result["wd"]
            result_to_save["has_active_bottom"] = str(exp_result["has_active_bottom"])
            result_to_save["has_interactive_layer"] = str(exp_result.get("has_interactive_layer"))
            result_to_save["Imbalanced"] = str(exp_result["imbal"])
            apply_protection_name = result_to_save["apply_protection_name"] = exp_result["defense_args"][
                "apply_protection_name"]
            result_to_save["apply_encoder"] = str(exp_result["defense_args"]["apply_encoder"])
            result_to_save["apply_negative_loss"] = str(exp_result["defense_args"]["apply_negative_loss"])
            if result_to_save["apply_negative_loss"] == "True":
                result_to_save["lambda_nl"] = exp_result["defense_args"]["lambda_nl"]
            else:
                result_to_save["lambda_nl"] = 0

            value = 0
            if exp_result["defense_args"]["apply_protection_name"] == "D_SGD":
                value = exp_result["defense_args"][apply_protection_name]["grad_bins"]
            elif exp_result["defense_args"]["apply_protection_name"] == "DP_LAPLACE":
                value = exp_result["defense_args"][apply_protection_name]["noise_scale"]
            elif exp_result["defense_args"]["apply_protection_name"] == "GC":
                value = exp_result["defense_args"][apply_protection_name]["gc_percent"]
            elif exp_result["defense_args"]["apply_protection_name"] == "PPDL":
                value = exp_result["defense_args"][apply_protection_name]["ppdl_theta_u"]
            elif exp_result["defense_args"]["apply_protection_name"] == "ISO":
                value = exp_result["defense_args"][apply_protection_name]["ratio"]
            elif exp_result["defense_args"]["apply_protection_name"] == "MARVELL":
                value = exp_result["defense_args"][apply_protection_name]["init_scale"]
            result_to_save["defense_value"] = value

            ordered_cols.extend(["dataset", "arch_config_name", "task_model_type", "optimizer_name", "lr", "wd",
                                 "has_interactive_layer", "has_active_bottom", "Imbalanced", "apply_encoder",
                                 "apply_protection_name", "defense_value", "apply_negative_loss", "lambda_nl"])

            other_metrics = exp_result.get('other_metrics')
            mc_result = exp_result.get('mc_result')
            mc_args = exp_result.get('mc_args')

            if other_metrics:
                if exp_result["num_classes"] == 2:
                    eval_metric = other_metrics["NBS"].split(":")[0]
                    result_to_save[f"Norm-based_{eval_metric}"] = other_metrics["NBS"].split(":")[-1]
                    result_to_save[f"Direction-based_{eval_metric}"] = other_metrics["DBS"].split(":")[-1]
                    DLI_metric = other_metrics.get("DLI")
                    if DLI_metric:
                        result_to_save[f"DLI-based_{eval_metric}"] = DLI_metric.split(":")[-1]

                    result_to_save["best_val_auc"] = other_metrics["best_val_auc"]
                    result_to_save["fscore"] = other_metrics["val_fscore"]
                else:
                    result_to_save["best_val_acc"] = other_metrics["best_val_acc"]

            if mc_result:
                # arguments and results for model completion attack
                if exp_result["num_classes"] == 2:
                    result_to_save["mc_auc"] = f"{exp_result['mc_result']['val_auc'] * 100:.2f}%"
                else:
                    result_to_save["mc_acc"] = f"{exp_result['mc_result']['acc'] * 100:.2f}%"

                result_to_save["mc_best_epoch"] = exp_result["mc_result"]["best_epoch"]
                result_to_save["train_all"] = str(exp_result["mc_args"]["train_all"])
                result_to_save["num_samples_ft"] = str(exp_result["mc_args"]["num_samples_ft"])
                result_to_save["load_pretrained_model"] = str(exp_result["mc_args"]["load_pretrained_model"])

                ordered_cols.extend(["train_all", "load_pretrained_model", "num_samples_ft"])

            print(result_to_save)
            record_list.append(result_to_save)
        df = pd.DataFrame.from_records(record_list)
        columns = df.columns

        ordered_cols = ordered_cols + [col for col in columns if col not in ordered_cols]
        df = df[ordered_cols]
        if dataset_name is not None:
            output_file = ExperimentResultStructure.create_root_file_path(f"{dataset_name}_{task_category}.csv", loc)
            df.to_csv(output_file, index=None)
            print("[INFO] saved excel exp result file to {}.".format(output_file))
        else:
            raise Exception("no dataset name in result record.")
