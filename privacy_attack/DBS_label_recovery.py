# -*- coding: utf-8 -*-
import numpy as np
import torch


def standardize(value):
    value = value.view(-1)
    val_max = torch.max(value)
    val_min = torch.min(value)
    return (value - val_min + 1e-16) / (val_max - val_min + 1e-16)


def cosine_similarity_torch(value_1, value_2):
    cosine_matrix = torch.cosine_similarity(value_1, value_2)
    return cosine_matrix


def direction_based_scoring_attack(gradient, attack_assist_args):
    y = attack_assist_args['y']
    batch_size = gradient.shape[0]

    pos_idx = np.random.choice(np.where(y.numpy() == 1)[0], size=1)[0]
    pos_grad = gradient[pos_idx: pos_idx + 1]
    grad = torch.reshape(gradient, shape=(batch_size, -1))
    cos_grad = cosine_similarity_torch(grad, pos_grad)

    pred = standardize(cos_grad)
    return pred.cpu().numpy()


# class DBSLabelRecoveryBase(LabelRecoveryBase):
#
#     def __init__(self,
#                  arch_config,
#                  machine_dict,
#                  datasets_dict,
#                  defense_dict,
#                  dlg_hyperparam_dict,
#                  label_rec_hyperparam_dict):
#         super(DBSLabelRecoveryBase, self).__init__(arch_config,
#                                                    machine_dict,
#                                                    datasets_dict,
#                                                    defense_dict,
#                                                    label_rec_hyperparam_dict)
#         self.device = machine_dict["device"]
#         self.num_iterations = dlg_hyperparam_dict["num_iterations"]
#         self.label_recovery_learning_rate = dlg_hyperparam_dict["label_recovery_learning_rate"]
#
#     def grad_based_label_attack(self, batch_data_a, batch_gt_one_hot_label, model_dict, mu_a, mu_a_grad, mu_b, **args):
#         pass
