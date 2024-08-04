# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from arch_utils import evaluate_result, f_beta_score
from privacy_attack_during_training.label_recovery_base import BINARY_INSTANT_LABEL_RECOVERY_METHODS, \
    BINARY_OPTIM_LABEL_RECOVERY_METHODS
from privacy_defense.defense_methods import DefenseName
from store_utils import save_model_checkpoints, save_exp_result
from svfl.data_utils import fetch_parties_data
from utils import printf


def label_to_one_hot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    one_hot_target = torch.zeros(target.size(0), num_classes)
    one_hot_target.scatter_(1, target, 1)
    return one_hot_target


def validate(vfl_arch_train, model_dict, val_loader, criterion, ds_name, num_classes, device="cpu"):
    """
    Validate the trained vfl.

    :param vfl_arch_train: the specific vfl to conduct the training.
    :param model_dict: the dictionary of models in the vfl.
    :param val_loader: data loader for validation.
    :param criterion: the criterion to compute the loss.
    :param ds_name: name of the dataset.
    :param num_classes: the number of classes.
    :param encoder: the encoder for protection privacy of labels. default:None
    :param device: the device to run the training. default:'cpu'.
    :return: the results of validation.
    """

    vfl_type = vfl_arch_train.get_vfl_type()
    defense_args = vfl_arch_train.get_defense_args_info()
    encoder = vfl_arch_train.get_encoder()

    apply_negative_loss = defense_args["apply_negative_loss"]

    is_binary_classification_task = True if num_classes == 2 else False

    lambda_nl = 0
    if apply_negative_loss:
        lambda_nl = defense_args['lambda_nl']

    for _, m in model_dict.items():
        m.eval() if m is not None else _

    suc_cnt = 0
    sample_cnt = 0
    val_loss_list = []
    negative_loss_list = []
    with torch.no_grad():
        all_predict_label_list = list()
        all_binary_predict_prob_list = list()
        all_actual_label_list = list()
        for gt_val_data, gt_val_label in val_loader:
            gt_val_one_hot_label = label_to_one_hot(gt_val_label, num_classes).to(device)
            val_data_a, val_data_b = fetch_parties_data(dataset_name=ds_name, data=gt_val_data, device=device)

            val_logit, nl_logit = vfl_arch_train.forward(val_data_a, val_data_b, model_dict)
            if vfl_type == "VNN":
                val_loss = criterion(val_logit, gt_val_one_hot_label)
            elif vfl_type == "VLR":
                bce_criterion = nn.BCEWithLogitsLoss()
                gt_val_label = gt_val_label.reshape(-1, 1).type_as(val_logit)
                val_loss = bce_criterion(val_logit, gt_val_label)
            else:
                raise Exception("dose not support vfl type : {}".format(vfl_type))

            val_loss_list.append(val_loss.item())
            if apply_negative_loss:
                if nl_logit is not None:
                    negative_loss = 1 / criterion(nl_logit, gt_val_one_hot_label)
                    negative_loss_list.append(lambda_nl * negative_loss.item())
                else:
                    raise Exception()

            if vfl_type == "VNN":
                predict_prob = F.softmax(val_logit, dim=-1)
                if encoder is not None:
                    predict_prob = encoder.decoder(predict_prob)
                    predict_labels = torch.argmax(predict_prob, dim=-1)
                else:
                    predict_labels = torch.argmax(predict_prob, dim=-1)
            elif vfl_type == "VLR":
                predict_prob = torch.sigmoid(val_logit.flatten())
                predict_labels = torch.round(predict_prob).long()
            else:
                raise Exception("dose not support vfl type : {}".format(vfl_type))

            sample_cnt += predict_labels.shape[0]
            actual_labels = torch.argmax(gt_val_one_hot_label, dim=-1)
            for predict, pred_prob, actual in zip(predict_labels, predict_prob, actual_labels):
                # enc_result_matrix[actual, enc_predict] += 1
                # result_matrix[actual, predict] += 1
                if is_binary_classification_task:
                    if vfl_type == "VNN":
                        pred_prob = pred_prob[1]
                    all_binary_predict_prob_list.append(pred_prob.cpu())
                all_predict_label_list.append(predict.cpu())
                all_actual_label_list.append(actual.cpu())
                if predict == actual:
                    suc_cnt += 1
        val_auc = -1
        val_f1 = -1
        if is_binary_classification_task:
            # val_auc = roc_auc_score(y_true=all_actual_label_list, y_score=all_predict_label_list)
            val_auc = roc_auc_score(y_true=all_actual_label_list, y_score=all_binary_predict_prob_list)
            val_f1 = f1_score(y_true=all_actual_label_list, y_pred=all_predict_label_list)
        val_acc = suc_cnt / float(sample_cnt)
        val_nl_loss = 0.0 if len(negative_loss_list) == 0 else np.mean(negative_loss_list)
        val_result = {"val_loss": np.mean(val_loss_list), "val_nl_loss": val_nl_loss,
                      "val_acc": val_acc, "val_auc": val_auc, "val_f1": val_f1}
    return val_result


def save_exp_results(curr_epoch, best_value, metric, exp_dict, defense_args_dict, postfix, suffix=""):
    exp_name = exp_dict["exp_name"]
    full_dir_name = exp_dict["full_dir_name"]
    arch_config_name = exp_dict["arch_config_name"]
    dataset_name = exp_dict["dataset_name"]
    num_classes = exp_dict["num_classes"]
    vfl_type = exp_dict["vfl_type"]
    task_model_type = exp_dict["task_model_type"]
    is_imbal = exp_dict["is_imbal"]
    has_active_bottom = exp_dict["has_active_bottom"]
    has_interactive_layer = exp_dict["has_interactive_layer"]
    optimizer_name = exp_dict["optimizer_name"]
    lr = exp_dict["lr"]
    wd = exp_dict["wd"]
    if defense_args_dict["apply_protection_name"] == DefenseName.MARVELL and "y" in defense_args_dict[
        DefenseName.MARVELL].keys():
        defense_args_dict[DefenseName.MARVELL].pop("y")
    save_exp_result({
        'dataset': dataset_name,
        'num_classes': num_classes,
        'epoch': curr_epoch + 1,
        'arch_config_name': arch_config_name,
        'task_model_type': task_model_type,
        'imbal': is_imbal,
        'has_active_bottom': has_active_bottom,
        'has_interactive_layer': has_interactive_layer,
        'vfl_type': vfl_type,
        'optimizer_name': optimizer_name,
        'lr': lr,
        'wd': wd,
        'defense_args': defense_args_dict,
        'best_value': best_value,
        'val_metric': metric,
        'other_metrics': postfix,
    }, dir=full_dir_name, filename=exp_name + "_" + suffix)
    return dataset_name, full_dir_name


class VFLLearner(object):

    def __init__(self, machine_dict, hyperparameter_dict):

        self.verbose = hyperparameter_dict["verbose"]
        printf("[INFO] machine_dict: \n {}".format(machine_dict), verbose=self.verbose)
        self.device = machine_dict['device']

        printf("[INFO] hyperparameter_dict: \n {}".format(hyperparameter_dict), verbose=self.verbose)
        self.epochs = hyperparameter_dict['epochs']
        self.print_interval = hyperparameter_dict.get('print_interval')

        self.label_recovery_attack_method_fn_dict = dict()
        self.eval_label_recovery_metric_dict = dict()

    def register_label_recovery_attack(self, attack_name, attack_method_fn, eval_metric="AUC"):
        self.label_recovery_attack_method_fn_dict[attack_name] = attack_method_fn
        self.eval_label_recovery_metric_dict[attack_name] = eval_metric

    def run(self, vfl_arch_train, model_dict, optimizer_dict, data_dict, criterion, exp_dict):

        train_loader = data_dict["data_loader_dict"]["train_loader"]
        val_loader = data_dict["data_loader_dict"]["val_loader"]
        test_loader = data_dict["data_loader_dict"]["test_loader"]
        num_classes = data_dict["num_classes"]
        dataset_name = data_dict["dataset_name"]

        has_label_recovery_attack = True if len(self.label_recovery_attack_method_fn_dict) > 0 else False
        is_binary_classification_task = True if num_classes == 2 else False

        attack_name_list = list(self.eval_label_recovery_metric_dict.keys())

        # record the epoch-averaged label recovery results of the whole experimental run.
        whole_run_bilabel_recovery_result_dict = {attack_name: list() for attack_name in attack_name_list}

        for k, m in model_dict.items():
            model_dict[k] = m.to(self.device) if m is not None else m

        val_metric = "val_auc" if num_classes == 2 else "val_acc"
        best_value = 0.0  # trace the best validation value.
        for i_epoch in range(self.epochs):
            tqdm_train = tqdm(train_loader, desc='Training (epoch #{})'.format(i_epoch + 1), disable=not self.verbose)

            # record the batch-level label recovery results of current epoch.
            epoch_bilabel_recovery_result_dict = {attack_name: list() for attack_name in attack_name_list}

            postfix = {'train_loss': 0.0, 'test_acc': 0.0}
            for i, (gt_data, gt_label) in enumerate(tqdm_train):

                for _, m in model_dict.items():
                    m.train() if m is not None else _

                gt_data_a, gt_data_b = fetch_parties_data(dataset_name=dataset_name, data=gt_data, device=self.device)
                gt_one_hot_label = label_to_one_hot(gt_label, num_classes).to(self.device)

                # ============ forward to active party and backward to cut layer of active party ============
                gt_label = gt_label.reshape(-1, 1)
                train_loss, (mu_a, mu_a_grad), (mu_b, mu_b_grad) = vfl_arch_train.forward_and_backward_to_cutlayer(
                    gt_data_a,
                    gt_data_b,
                    gt_label,
                    gt_one_hot_label,
                    model_dict,
                    optimizer_dict=optimizer_dict,
                    criterion=criterion)

                # ============ label recovery attacks leveraging backward intermediate gradient ============
                if has_label_recovery_attack:
                    # === If all labels in the batch is 0 (negative), we will not perform label recovery.
                    # === Note that we consider integer 1 as the positive label for 2 label-classification.
                    if torch.sum(gt_label) != 0:
                        for attack_name, attack_method in self.label_recovery_attack_method_fn_dict.items():
                            eval_metric = self.eval_label_recovery_metric_dict[attack_name]
                            if (attack_name in BINARY_INSTANT_LABEL_RECOVERY_METHODS) and is_binary_classification_task:
                                # print("I attach name:", i, attack_name, attack_method)
                                attack_assist_args = {"y": gt_label.detach().clone(),
                                                      "passive_data": gt_data_a.detach().clone()}
                                # eval_metric = self.eval_label_recovery_metric_dict[attack_name]
                                pred = attack_method(mu_a_grad.detach().clone(), attack_assist_args)
                                eval_value = evaluate_result(gt_label.numpy(), pred, eval_metric=eval_metric)
                                epoch_bilabel_recovery_result_dict[attack_name].append(eval_value)
                            elif (attack_name in BINARY_OPTIM_LABEL_RECOVERY_METHODS) and is_binary_classification_task:
                                if i_epoch == 0 or (i_epoch + 1) % 2 == 0:
                                    # print("O attach name:", i, attack_name, attack_method)
                                    attack_assist_args = {"passive_data": gt_data_a.detach().numpy()}
                                    eval_metric = self.eval_label_recovery_metric_dict[attack_name]
                                    pred = attack_method(mu_a_grad.detach().numpy(), attack_assist_args)
                                    eval_value = evaluate_result(gt_label.numpy(), pred, eval_metric=eval_metric)
                                    epoch_bilabel_recovery_result_dict[attack_name].append(eval_value)

                # ============ update local bottom models of all parties ============
                vfl_arch_train.update_local_models(mu_a, mu_b, mu_a_grad, mu_b_grad, model_dict,
                                                   optimizer_dict=optimizer_dict)

                # ============ validation ========================
                if i == len(train_loader) - 1:
                    # if i % self.print_interval == 0:
                    #     val_result = self.validate(vfl_arch_train, model_dict, val_loader, criterion)
                    val_result = validate(vfl_arch_train, model_dict, val_loader, criterion, dataset_name, num_classes,
                                          device=self.device)
                    current_val_value = val_result[val_metric]

                    postfix['train_loss'] = train_loss.item()
                    postfix['val_loss'] = val_result["val_loss"]
                    postfix['val_nl_loss'] = val_result["val_nl_loss"]
                    postfix['val_acc'] = '{:.2f}%'.format(val_result["val_acc"] * 100)
                    postfix['val_auc'] = '{:.2f}%'.format(val_result["val_auc"] * 100)
                    postfix['val_f1'] = '{:.2f}%'.format(val_result["val_f1"] * 100)

                    # ========== compute label recovery attack scores of current epoch ========================
                    if has_label_recovery_attack:
                        for attack_name, eval_metric in self.eval_label_recovery_metric_dict.items():
                            if (attack_name in BINARY_INSTANT_LABEL_RECOVERY_METHODS) and is_binary_classification_task:
                                result_list = epoch_bilabel_recovery_result_dict[attack_name]
                                curr_result = self.eval_epochwise_label_recovery_result(
                                    attack_name, result_list, whole_run_bilabel_recovery_result_dict)
                                postfix[attack_name[:3]] = f"{eval_metric}:{curr_result * 100:.2f}%"
                            elif (attack_name in BINARY_OPTIM_LABEL_RECOVERY_METHODS) and is_binary_classification_task:
                                # === For now, if the attack requires optimization process (e.g., RR and DLG),
                                # === we perform the attack every N epochs to save time.
                                if i_epoch == 0 or (i_epoch + 1) % 2 == 0:
                                    # if i_epoch == 0:
                                    result_list = epoch_bilabel_recovery_result_dict[attack_name]
                                    curr_result = self.eval_epochwise_label_recovery_result(
                                        attack_name, result_list, whole_run_bilabel_recovery_result_dict)
                                    postfix[attack_name[:3]] = f"{eval_metric}:{curr_result * 100:.2f}%"

                    # ========== record best validation value ========================
                    is_best = current_val_value > best_value
                    best_value = max(current_val_value, best_value)
                    postfix["best_" + val_metric] = f"{best_value * 100:.2f}%"

                    # ========== compute the f-score of best main task score and defense score ==========
                    num_bilabel_recovery = len(whole_run_bilabel_recovery_result_dict)
                    if num_bilabel_recovery > 0 and is_binary_classification_task:
                        bilbl_rec_score_sum = 0.0
                        for eal in whole_run_bilabel_recovery_result_dict.values():
                            bilbl_rec_score_sum += (0.5 - np.abs(np.quantile(eal, 0.95) - 0.5))

                        bilbl_rec_score_avg = bilbl_rec_score_sum / num_bilabel_recovery
                        bilbl_fscore = f_beta_score(bilbl_rec_score_avg, best_value, beta=2.0)

                        postfix["val_fscore"] = f"{bilbl_fscore * 100:.2f}%"

                    tqdm_train.set_postfix(postfix)

                    # ========== save model ========================
                    full_dir_name = exp_dict["full_dir_name"]
                    defense_args_dict = vfl_arch_train.get_defense_args_info()
                    if is_best:
                        test_result = validate(vfl_arch_train, model_dict, test_loader, criterion, dataset_name,
                                               num_classes, device=self.device)
                        postfix['test_acc'] = '{:.2f}%'.format(test_result["val_acc"] * 100)
                        postfix['test_auc'] = '{:.2f}%'.format(test_result["val_auc"] * 100)
                        postfix['test_f1'] = '{:.2f}%'.format(test_result["val_f1"] * 100)
                        # save experimental result and model when achieving the current best validation score.
                        save_exp_results(i_epoch, best_value, val_metric, exp_dict, defense_args_dict, postfix, "best")
                        save_model_checkpoints(model_dict, is_best, dir=full_dir_name)
                    if i_epoch == self.epochs - 1:
                        test_result = validate(vfl_arch_train, model_dict, test_loader, criterion, dataset_name,
                                               num_classes, device=self.device)
                        postfix['test_acc'] = '{:.2f}%'.format(test_result["val_acc"] * 100)
                        postfix['test_auc'] = '{:.2f}%'.format(test_result["val_auc"] * 100)
                        postfix['test_f1'] = '{:.2f}%'.format(test_result["val_f1"] * 100)
                        # save experimental result when reaching the last epoch.
                        save_exp_results(i_epoch, best_value, val_metric, exp_dict, defense_args_dict, postfix, "last")

    @staticmethod
    def eval_epochwise_label_recovery_result(attack_name, result_list, whole_run_bilabel_recovery_result_dict):
        epoch_avg = np.mean(result_list)
        whole_run_bilabel_recovery_result_dict[attack_name].append(epoch_avg)
        epoch_avg_list = whole_run_bilabel_recovery_result_dict[attack_name]
        curr_result = np.quantile(epoch_avg_list, 0.95)
        # curr_result = np.mean(epoch_avg_list)
        return curr_result
