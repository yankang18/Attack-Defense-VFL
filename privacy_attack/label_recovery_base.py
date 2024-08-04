# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score

from arch_utils import build_models, prepare_optimizers, OptimName
from privacy_defense.defense_methods import DefenseName
from store_utils import get_path_slash
from svfl.data_utils import fetch_batch_data, label_to_one_hot
from utils import get_timestamp, cross_entropy_for_one_hot, calculate_entropy


class LabelRecoveryMethodName(object):
    NBS = "NBS_NORM_BASED_SCORING"
    DBS = "DBS_DIRECTION_BASED_SCORING"
    DLI = "DLI_DIRECT_LABEL_INFERENCE"
    RR = "RRA_RESIDUE_RECONSTRUCTION"


BINARY_INSTANT_LABEL_RECOVERY_METHODS = [LabelRecoveryMethodName.NBS, LabelRecoveryMethodName.DBS,
                                         LabelRecoveryMethodName.DLI]
BINARY_OPTIM_LABEL_RECOVERY_METHODS = [LabelRecoveryMethodName.RR]
BINARY_LABEL_RECOVERY_METHODS = BINARY_INSTANT_LABEL_RECOVERY_METHODS + BINARY_OPTIM_LABEL_RECOVERY_METHODS


class LabelRecoveryBase(object):

    def __init__(self,
                 arch_config,
                 machine_dict,
                 datasets_dict,
                 defense_dict,
                 label_rec_hyperparam_dict):

        self.arch_config = arch_config

        self.machine_dict = machine_dict
        self.device = machine_dict["device"]

        print("[INFO] machine_dict: \n {}".format(machine_dict))

        self.defense_dict = defense_dict
        self.encoder = defense_dict["encoder"]
        self.apply_protection_fn = defense_dict["apply_protection"]
        self.apply_defense_name = defense_dict['args']['apply_protection_name']

        print("[INFO] defense_dict: \n {}".format(defense_dict))

        self.dataset_name = datasets_dict["dataset_name"]
        self.train_dataset = datasets_dict['train_dataset']
        self.num_classes = datasets_dict["num_classes"]
        self.is_imbal = datasets_dict["is_imbal"]

        print("[INFO] datasets_dict: \n {}".format(datasets_dict))

        self.exp_result_dir = label_rec_hyperparam_dict["exp_result_dir"]
        self.has_active_bottom = label_rec_hyperparam_dict["has_active_bottom"]
        self.arch_config_name = label_rec_hyperparam_dict["arch_config_name"]
        self.batch_size_list = label_rec_hyperparam_dict["batch_size_list"]
        self.num_exp = label_rec_hyperparam_dict["num_experiments"]
        self.model_lr = label_rec_hyperparam_dict['model_lr']

        print("[INFO] label_rec_hyperparam_dict: \n {}".format(label_rec_hyperparam_dict))

        self.active_party_has_bottom_model = None

    def save_exp_res(self, exp_record_list):
        timestamp = get_timestamp()
        exp_df = pd.DataFrame.from_records(exp_record_list)
        first_value = 0
        second_value = 0
        if self.defense_dict["args"]["apply_protection_name"] == DefenseName.D_SGD:
            first_value = self.defense_dict["args"][DefenseName.D_SGD]["grad_bins"]
            second_value = self.defense_dict["args"][DefenseName.D_SGD]["bound_abs"]
        elif self.defense_dict["args"]["apply_protection_name"] == DefenseName.DP_LAPLACE:
            first_value = self.defense_dict["args"][DefenseName.DP_LAPLACE]["noise_scale"]
        elif self.defense_dict["args"]["apply_protection_name"] == DefenseName.GRAD_COMPRESSION:
            first_value = self.defense_dict["args"][DefenseName.GRAD_COMPRESSION]["gc_percent"]
        elif self.defense_dict["args"]["apply_protection_name"] == DefenseName.PPDL:
            first_value = self.defense_dict["args"][DefenseName.PPDL]["ppdl_theta_u"]
        elif self.defense_dict["args"]["apply_protection_name"] == DefenseName.ISO:
            first_value = self.defense_dict["args"][DefenseName.ISO]["ratio"]
        elif self.defense_dict["args"]["apply_protection_name"] == DefenseName.MARVELL:
            first_value = self.defense_dict["args"][DefenseName.MARVELL]["init_scale"]
        slash = get_path_slash()
        exp_file_full_name = f"{self.exp_result_dir}{slash}exp_{self.dataset_name}_{self.arch_config_name}" \
                             f"_ab{self.has_active_bottom}_imbal{self.is_imbal}_{self.apply_defense_name}_" \
                             f"fv{first_value}_sv{second_value}_{timestamp}.csv"
        exp_df.to_csv(exp_file_full_name)
        print(f"[INFO] Save result to {exp_file_full_name}")

    def grad_based_label_attack(self, batch_data_a, gt_label, batch_gt_one_hot_label, model_dict, mu_a, mu_a_grad, mu_b,
                                **args):
        return 0.0

    def run(self, vfl_arch_train_class):

        exp_record_list = list()
        for batch_size in self.batch_size_list:

            vfl_arch_train = vfl_arch_train_class(machine_dict=self.machine_dict, defense_dict=self.defense_dict)

            recovery_rate = []
            global_result_matrix = np.zeros((self.num_classes, self.num_classes))
            gt_sample_label_list = []
            pr_sample_label_list = []
            pr_sample_prob_list = []
            for exp_id in range(self.num_exp):
                # ====== prepare models ======
                model_dict = build_models(self.arch_config)

                # ====== prepare optimizers ======
                optimizer_dict = prepare_optimizers(model_dict=model_dict,
                                                    learning_rate=self.model_lr,
                                                    weight_decay=0.0,
                                                    optimizer_args_dict=None,
                                                    optim_name=OptimName.ADAM)

                # ====== get data of parties ======
                gt_data_a, gt_data_b, gt_label = fetch_batch_data(self.dataset_name, self.train_dataset,
                                                                  batch_size, self.device)

                gt_one_hot_label = label_to_one_hot(gt_label, self.num_classes).to(self.device)

                # ====== one communication round training ======
                gt_label = gt_label.reshape(-1, 1)
                loss, (mu_a, mu_a_grad), (mu_b, _) = vfl_arch_train.forward_and_backward_to_cutlayer(
                    gt_data_a,
                    gt_data_b,
                    gt_label,
                    gt_one_hot_label,
                    model_dict=model_dict,
                    optimizer_dict=optimizer_dict,
                    encoder=self.encoder,
                    criterion=cross_entropy_for_one_hot)

                # ====== perform gradient-based attack ======
                args = {"global_result_matrix": global_result_matrix,
                        "gt_sample_label_list": gt_sample_label_list,
                        "pr_sample_prob_list": pr_sample_prob_list,
                        "pr_sample_label_list": pr_sample_label_list,
                        "arch_config_name": self.arch_config_name}

                suc_cnt = self.grad_based_label_attack(gt_data_a, gt_label, gt_one_hot_label, model_dict,
                                                       mu_a, mu_a_grad, mu_b, **args)

                rate = suc_cnt / batch_size
                print(f"exp_id={exp_id}, # of suc:{suc_cnt}; bs:{batch_size}; n_classes:{self.num_classes}; "
                      f"rec_rate={rate};")
                recovery_rate.append(rate)
                exp_record_list.append({"exp_id": exp_id, "num_class": self.num_classes, "batch_size": batch_size,
                                        "rec_rate": rate})

            mean_rate = np.mean(recovery_rate)
            std_rate = np.std(recovery_rate)

            pearson = pearsonr(gt_sample_label_list, pr_sample_label_list)
            print(f"[INFO] Pearson: {pearson}")

            auc = 0.0
            if self.num_classes == 2:
                auc = roc_auc_score(y_true=gt_sample_label_list, y_score=pr_sample_prob_list)
                print(f"[INFO] AUC: {auc}")
            else:
                print(f"[INFO] AUC cannot apply because the class number is larger than 2.")
            acc = accuracy_score(y_true=gt_sample_label_list, y_pred=pr_sample_label_list)
            print(f"[INFO] global_result_matrix: \n {global_result_matrix}, {np.sum(global_result_matrix)}")
            entropy_value = calculate_entropy(global_result_matrix, N=self.num_classes)
            print(f"[INFO] entropy_value:{entropy_value}")
            exp_result = {"exp_id": -1, "num_class": self.num_classes, "batch_size": batch_size,
                          "rec_rate_mean": mean_rate, "rec_rate_std": std_rate, "entropy_value": entropy_value,
                          "rec_auc": auc, "rec_acc": acc}
            print(exp_result)
            exp_record_list.append(exp_result)

        self.save_exp_res(exp_record_list)
