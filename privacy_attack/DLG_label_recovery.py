# -*- coding: utf-8 -*-
import sys

sys.path.append("../svfl/")
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn

from arch_utils import build_layer_model, ModelType
from privacy_attack.label_recovery_base import LabelRecoveryBase
from utils import cross_entropy_for_one_hot


def forward_top_models(top_model_list, x):
    for m in top_model_list:
        x = m(x)
    return x


def aggregate(top_model_list, mu_a, mu_b):
    if mu_b is not None:
        if len(top_model_list) > 0:
            x = torch.cat((mu_a, mu_b), dim=-1)
            return forward_top_models(top_model_list, x)
        else:
            return mu_a + mu_b
    else:
        return forward_top_models(top_model_list, mu_a) if len(top_model_list) > 0 else mu_a


class DLGLabelRecovery(LabelRecoveryBase):

    def __init__(self,
                 arch_config,
                 machine_dict,
                 datasets_dict,
                 defense_dict,
                 dlg_hyperparam_dict,
                 label_rec_hyperparam_dict):
        super(DLGLabelRecovery, self).__init__(arch_config,
                                               machine_dict,
                                               datasets_dict,
                                               defense_dict,
                                               label_rec_hyperparam_dict)
        self.device = machine_dict["device"]
        self.dlg_iter = dlg_hyperparam_dict["dlg_iter"]
        self.dlg_lr = dlg_hyperparam_dict["dlg_lr"]

        print("[INFO] dlg_hyperparam_dict: \n {}".format(dlg_hyperparam_dict))

    def grad_based_label_attack(self, batch_data_a, batch_gt_label, batch_gt_one_hot_label, model_dict,
                                mu_a, mu_a_grad, mu_b, **args):

        global_result_matrix = args["global_result_matrix"]
        gt_sample_label_list = args["gt_sample_label_list"]
        pr_sample_label_list = args["pr_sample_label_list"]
        pr_sample_prob_list = args["pr_sample_prob_list"]
        arch_config_name = args["arch_config_name"]

        interactive_layer = model_dict.get(ModelType.INTERACTIVE_LAYER)
        task_model = model_dict.get(ModelType.TASK_MODEL)
        passive_bottom_model = model_dict.get(ModelType.PASSIVE_BOTTOM)
        active_bottom_model = model_dict.get(ModelType.ACTIVE_BOTTOM)
        self.active_party_has_bottom_model = True if active_bottom_model is not None else False

        # ============ build fake active party's models at passive party  ============
        active_fake_top_models = list()
        if interactive_layer is not None:
            fake_interactive_layer = build_layer_model(self.arch_config["interactive_layer_config"])
            active_fake_top_models.append(fake_interactive_layer)
        if task_model is not None:
            fake_task_model = build_layer_model(self.arch_config["task_model_config"])
            active_fake_top_models.append(fake_task_model)

        # ============ compute ground truth gradient of passive bottom model ============
        pbm_grad = torch.autograd.grad(outputs=mu_a, inputs=passive_bottom_model.parameters(), grad_outputs=mu_a_grad)
        original_pbm_grad = list((_.detach().clone() for _ in pbm_grad))

        # ============ prepare active party's label (and data), which to be recovered ============
        criterion = cross_entropy_for_one_hot
        truth_label = batch_gt_one_hot_label
        if arch_config_name == "VLR":
            criterion = nn.BCEWithLogitsLoss()
            truth_label = batch_gt_label
        dummy_label = torch.randn(truth_label.size()).to(self.device).requires_grad_(True)

        if self.active_party_has_bottom_model:
            print("[DEBUG] Create dummy output of active party's bottom model.")
            dummy_mu_b = torch.randn(mu_b.size()).to(self.device).requires_grad_(True)
            recover_variables = [dummy_mu_b, dummy_label]
        else:
            print("[DEBUG] There is no active party's bottom model.")
            dummy_mu_b = None
            recover_variables = [dummy_label]

        # ============ prepare optimizer for label recovery ============
        if len(active_fake_top_models) > 0:
            param = list(active_fake_top_models[0].parameters())
            for i in range(1, len(active_fake_top_models)):
                param += list(active_fake_top_models[i].parameters())
            optimizer = torch.optim.Adam(recover_variables + param, lr=self.dlg_lr)
        else:
            optimizer = torch.optim.Adam(recover_variables, lr=self.dlg_lr)
            # optimizer = torch.optim.SGD(recover_variables, lr=self.label_learning_rate)

        # ============ perform dlg attack for recovering active's labels  ============
        for iteration in range(self.dlg_iter):
            def closure():
                optimizer.zero_grad()

                mu_a_prime = passive_bottom_model(batch_data_a)
                dummy_pred = aggregate(active_fake_top_models, mu_a_prime, dummy_mu_b)

                dummy_soft_label = torch.softmax(dummy_label, dim=-1)
                if arch_config_name == "VLR":
                    dummy_soft_label = torch.sigmoid(dummy_label)

                dummy_loss = criterion(dummy_pred, dummy_soft_label)
                dummy_pbm = torch.autograd.grad(dummy_loss, passive_bottom_model.parameters(), create_graph=True)

                grad_diff = 0
                for (gx, gy) in zip(dummy_pbm, original_pbm_grad):
                    grad_diff += ((gx - gy) ** 2).sum()
                    # grad_diff += torch.sqrt((gx - gy) ** 2).sum()
                grad_diff.backward()

                return grad_diff

            optimizer.step(closure)
            iter = iteration + 1
            if iter % 1000 == 0:
                if dummy_mu_b is not None:
                    numpy_dummy_logits_b = dummy_mu_b.cpu().detach().numpy()
                    numpy_logits_b = mu_b.cpu().detach().numpy()
                    logit_b_mse = (np.square(numpy_logits_b - numpy_dummy_logits_b)).mean()
                    print(f"[{iter}] iters, logit_b_mse={logit_b_mse}")

                loss = closure()
                print(f"[{iter}] iters, loss:{loss}")

                # # numpy_dummy_label = dummy_label.detach().numpy()
                # for index, (pr, gt) in enumerate(zip(dummy_label, batch_gt_one_hot_label)):
                #     pr_lbl = torch.argmax(pr, dim=-1)
                #     gt_lbl = torch.argmax(gt, dim=-1)
                #     correct = True if pr_lbl == gt_lbl else False
                #     if correct is False:
                #         print(f"[{iter}]-[{index}] pr: {pr}, {pr_lbl}")
                #         print(f"[{iter}]-[{index}] gt: {gt}, {gt_lbl}")
                #         print(f"[{iter}]-[{index}]: {correct}")
                #         # if index >= 20:
                #         #     break

                # original_pred_final = F.softmax(logits_b + logits_a, dim=-1)
                # dummy_pred_final = F.softmax(dummy_data + logits_a, dim=-1)
                # pred_b_diff = 0
                # for b1, b2 in zip(dummy_pred_final, original_pred_final):
                #     pred_b_diff += ((b1 - b2) ** 2).sum().item()
                # print(f"{iteration} {pred_b_diff}", pred_b_diff)

        # ============ compute results of recovering active's labels  ============
        suc_cnt = 0
        if arch_config_name == "VNN":
            dummy_label_prob = torch.softmax(dummy_label, dim=-1)
            dummy_pred_label = torch.argmax(dummy_label_prob, dim=-1)
            truth_label = torch.argmax(truth_label, dim=-1)
        elif arch_config_name == "VLR":
            dummy_label_prob = torch.sigmoid(dummy_label)
            dummy_pred_label = torch.round(dummy_label_prob).long()
        else:
            raise Exception("Dose not support arch:[{}] for now.".format(arch_config_name))
        # result_matrix = np.zeros((num_classes, num_classes))
        for index, (pr_prob, pr, gt) in enumerate(zip(dummy_label_prob, dummy_pred_label, truth_label)):
            pr_sample_prob_list.append(pr_prob.cpu().item())
            pr_sample_label_list.append(pr.cpu().item())
            gt_sample_label_list.append(gt.cpu().item())
            # result_matrix[gt, pr] += 1
            global_result_matrix[gt, pr] += 1
            if pr == gt:
                suc_cnt += 1
        return suc_cnt
