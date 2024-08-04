# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from arch_utils import ModelType
from utils import cross_entropy_for_one_hot
from utils import keep_predict_loss


def label_to_one_hot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


class VFLLRTrain(object):

    def __init__(self, machine_dict, defense_dict):

        print("[DEBUG] ===> Using VFL Logistic Regression Train.")

        self.device = machine_dict['device']
        self.cuda_id = machine_dict['cuda_id']
        # self.random_seed = machine_dict['random_seed']

        self.defense_dict = defense_dict
        self.defense_dict["device"] = self.device
        self.protection_name = defense_dict['args']['apply_protection_name']
        self.apply_protection_fn = defense_dict["apply_protection"]
        self.encoder = defense_dict['encoder']

        self.criterion = nn.BCEWithLogitsLoss()

    def get_defense_args_info(self):
        return self.defense_dict['args']

    def get_encoder(self):
        return self.encoder

    @staticmethod
    def get_vfl_type():
        return "VLR"

    @staticmethod
    def update_model(model_optimizer, loss):
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    def forward(self, batch_data_a, batch_data_b, model_dict):

        logit, (_, _,) = self._forward_consider_cutlayer(batch_data_a, batch_data_b, model_dict)

        return logit, None

    def forward_and_backward_to_cutlayer(self, bt_data_a, bt_data_b, gt_label, bt_one_hot_label,
                                         model_dict, optimizer_dict=None, encoder=None,
                                         criterion=cross_entropy_for_one_hot):

        # ====== forward to loss ======
        logit, (mu_a, mu_b) = self._forward_consider_cutlayer(bt_data_a, bt_data_b, model_dict)

        # # ====== compute loss ======
        # # = Note: probably encoder is NOT suitable for VLR =
        # if encoder is not None:
        #     _, _, gt_one_hot_label = encoder(bt_one_hot_label)
        #     # print("[DEBUG] svfl_nn_train 62 batch_label:", batch_label[:10])
        #     # print("[DEBUG] svfl_nn_train 64 gt_one_hot_label:", gt_one_hot_label[:10])
        # else:
        #     gt_one_hot_label = bt_one_hot_label
        #
        # # loss = criterion(logit, gt_one_hot_label)
        # # y = torch.topk(bt_one_hot_label, 1)[1].squeeze(1)

        # print(logit.shape)
        # print(gt_label.shape)

        gt_label = gt_label.type_as(logit)
        loss = self.criterion(logit, gt_label)
        y = gt_label

        # ====== backward to cut-layer ======
        mu_a_grad, mu_b_grad = self._backward_to_cutlayer(loss, logit, y)
        return loss, (mu_a, mu_a_grad), (mu_b, mu_b_grad)

    def _forward_consider_cutlayer(self, batch_data_a, batch_data_b, model_dict):

        bottom_model_a = model_dict[ModelType.PASSIVE_BOTTOM]
        bottom_model_b = model_dict[ModelType.ACTIVE_BOTTOM]

        mu_a = bottom_model_a(batch_data_a)
        mu_b = bottom_model_b(batch_data_b)
        logit = mu_a + mu_b

        return logit, (mu_a, mu_b)

    def _backward_to_cutlayer(self, loss, z, y=None):

        z_gradients = torch.autograd.grad(loss, z, retain_graph=True)
        z_grad = z_gradients[0].detach().clone()

        # Note that: we should treat these gradients as HE encrypted.
        mu_a_grad = z_grad
        mu_b_grad = z_grad

        if "marvell" in self.apply_protection_fn.__str__():
            self.defense_dict['args'][self.protection_name]['y'] = y
        mu_a_grad = self.apply_protection_fn(mu_a_grad, **self.defense_dict['args'][self.protection_name])
        return mu_a_grad, mu_b_grad

    def update_local_models(self,
                            mu_a, mu_b,
                            mu_a_grad, mu_b_grad,
                            model_dict,
                            optimizer_dict):

        bottom_model_active = model_dict[ModelType.ACTIVE_BOTTOM]

        bottom_model_passive_optimizer = optimizer_dict[ModelType.PASSIVE_BOTTOM]
        bottom_model_active_optimizer = optimizer_dict[ModelType.ACTIVE_BOTTOM]

        active_party_has_bottom_model = True if bottom_model_active is not None else False

        passive_model_loss = keep_predict_loss(mu_a, mu_a_grad)
        self.update_model(model_optimizer=bottom_model_passive_optimizer, loss=passive_model_loss)

        if active_party_has_bottom_model:
            active_model_loss = keep_predict_loss(mu_b, mu_b_grad)
            self.update_model(model_optimizer=bottom_model_active_optimizer, loss=active_model_loss)
