# -*- coding: utf-8 -*-

import torch

from arch_utils import ModelType
from utils import cross_entropy_for_one_hot
from utils import keep_predict_loss


class VFLNNTrain(object):

    def __init__(self, machine_dict, defense_dict):

        print("[DEBUG] ===> Using VFL Neural Network Train [Version 2].")

        self.device = machine_dict['device']
        self.cuda_id = machine_dict['cuda_id']
        # self.random_seed = machine_dict['random_seed']

        self.defense_dict = defense_dict
        self.defense_dict["device"] = self.device
        self.protection_name = defense_dict['args']['apply_protection_name']
        self.apply_protection_fn = defense_dict["apply_protection"]
        self.apply_protection_to_passive_party = defense_dict["apply_protection_to_passive_party"]
        self.apply_negative_loss = defense_dict["apply_negative_loss"]
        self.encoder = defense_dict['encoder']

        self.lambda_nl = 0
        if self.apply_negative_loss:
            self.lambda_nl = defense_dict['args']['lambda_nl']

    def get_defense_args_info(self):
        return self.defense_dict['args']

    def get_encoder(self):
        return self.encoder

    @staticmethod
    def get_vfl_type():
        return "VNN"

    @staticmethod
    def update_model(model_optimizer, loss):
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

    def forward(self, batch_data_a, batch_data_b, model_dict):

        active_task_model = model_dict[ModelType.TASK_MODEL]
        # interactive_layer = model_dict[ModelType.INTERACTIVE_LAYER]
        active_interactive_layer = model_dict[ModelType.ACTIVE_INTERACTIVE_LAYER]
        passive_interactive_layer = model_dict[ModelType.PASSIVE_INTERACTIVE_LAYER]

        bottom_model_passive = model_dict[ModelType.PASSIVE_BOTTOM]
        bottom_model_active = model_dict[ModelType.ACTIVE_BOTTOM]

        # ============ vertical federated learning forward ====== #

        active_party_has_bottom_model = True if bottom_model_active is not None else False
        has_interactive_layer = True if passive_interactive_layer is not None else False

        mu_a = bottom_model_passive(batch_data_a)

        mu_b = None
        adversarial_logit = None
        if active_party_has_bottom_model:
            mu_b = bottom_model_active(batch_data_b)

        if has_interactive_layer:
            z_a = passive_interactive_layer(mu_a)
            if active_party_has_bottom_model:
                z_b = active_interactive_layer(mu_b)
                z = z_a + z_b
            else:
                z = z_a

            # apply negative loss
            if self.apply_negative_loss:
                adversarial_model = model_dict[ModelType.NEGATIVE_LOSS_MODEL]
                adversarial_logit = adversarial_model(z_a)
        else:
            z = mu_a
            if active_party_has_bottom_model:
                z = torch.cat((mu_a, mu_b), dim=-1)

            # apply negative loss
            if self.apply_negative_loss:
                adversarial_model = model_dict[ModelType.NEGATIVE_LOSS_MODEL]
                adversarial_logit = adversarial_model(mu_a)

        logit = active_task_model(z) if active_task_model is not None else z

        return logit, adversarial_logit

    def forward_and_backward_to_cutlayer(self, bt_data_a, bt_data_b, gt_label, bt_one_hot_label,
                                         model_dict, optimizer_dict=None, criterion=cross_entropy_for_one_hot):

        # ====== forward to loss ======
        (logit, nl_logit), (mu_a, mu_a_ph), (mu_b, mu_b_ph), (z_a, z_a_ph), (z_b, z_b_ph) = self._forward_consider_cutlayer(
            bt_data_a,
            bt_data_b,
            model_dict)

        # ====== compute loss ======
        if self.encoder is not None:
            # _, _, gt_one_hot_label = self.encoder(bt_one_hot_label)
            _, _, gt_one_hot_label = self.encoder(bt_one_hot_label)
            # print("[DEBUG] svfl_nn_train 100 batch_label:", bt_one_hot_label[:10])
            # print("[DEBUG] svfl_nn_train 101 gt_one_hot_label:", gt_one_hot_label[:10])
        else:
            gt_one_hot_label = bt_one_hot_label

        loss = criterion(logit, gt_one_hot_label)
        if self.apply_negative_loss:
            negative_loss = 1 / criterion(nl_logit, bt_one_hot_label)
            conf = self.lambda_nl * negative_loss
            # print("here:", self.lambda_nl)
            loss += conf
            # confusion_model_optimizer = optimizer_dict[ModelType.PASSIVE_CONFUSION_MODEL]
            # confusion_model_optimizer.zero_grad()
            # conf.backward()
            # confusion_model_optimizer.step()

        # ====== backward to cut-layer ======
        y = torch.topk(bt_one_hot_label, 1)[1].squeeze(1)
        # loss, mu_a_ph, mu_b_ph, z, z_ph, model_dict, optimizer_dict):
        mu_a_grad, mu_b_grad = self._backward_to_cutlayer(loss=loss,
                                                          mu_a_ph=mu_a_ph, mu_b_ph=mu_b_ph,
                                                          z_a=z_a, z_a_ph=z_a_ph,
                                                          z_b=z_b, z_b_ph=z_b_ph,
                                                          model_dict=model_dict, optimizer_dict=optimizer_dict, y=y)
        return loss, (mu_a, mu_a_grad), (mu_b, mu_b_grad)

    def _forward_consider_cutlayer(self, batch_data_a, batch_data_b, model_dict):

        active_task_model = model_dict[ModelType.TASK_MODEL]
        # interactive_layer = model_dict[ModelType.INTERACTIVE_LAYER]
        active_interactive_layer = model_dict[ModelType.ACTIVE_INTERACTIVE_LAYER]
        passive_interactive_layer = model_dict[ModelType.PASSIVE_INTERACTIVE_LAYER]

        bottom_model_passive = model_dict[ModelType.PASSIVE_BOTTOM]
        bottom_model_active = model_dict[ModelType.ACTIVE_BOTTOM]

        # ============ vertical federated learning forward ====== #

        active_party_has_bottom_model = True if bottom_model_active is not None else False
        has_interactive_layer = True if passive_interactive_layer is not None else False

        # mu_a_ph and mu_b_ph are the input to the latter layers such as the [interactive layer] or the [task model].
        mu_a_ph = torch.tensor([], requires_grad=True)
        mu_a = bottom_model_passive(batch_data_a)
        mu_a_ph.data = mu_a.data

        mu_b, mu_b_ph = None, None
        if active_party_has_bottom_model:
            mu_b = bottom_model_active(batch_data_b)
            mu_b_ph = torch.tensor([], requires_grad=True)
            mu_b_ph.data = mu_b.data

        # z_a_ph and z_b_ph are the input to the later layers, i.e., [task model].
        # They are used to store gradients.
        z_a, z_b = None, None
        z_a_ph, z_b_ph = None, None
        adversarial_logit = None
        if has_interactive_layer:

            # z_a = passive_interactive_layer(mu_a_ph)
            # z_ph = torch.tensor([], requires_grad=True)
            # if active_party_has_bottom_model:
            #     z_b = active_interactive_layer(mu_b_ph)
            #     z_ph.data = z_a.data + z_b.data
            # else:
            #     z_ph.data = z_a.data

            z_a = passive_interactive_layer(mu_a_ph)
            z_a_ph = torch.tensor([], requires_grad=True)
            z_a_ph.data = z_a.data
            if active_party_has_bottom_model:
                z_b = active_interactive_layer(mu_b_ph)
                z_b_ph = torch.tensor([], requires_grad=True)
                z_b_ph.data = z_b.data
                z_ph = z_a_ph + z_b_ph
            else:
                z_ph = z_a_ph

            # apply negative loss
            if self.apply_negative_loss:
                adversarial_model = model_dict[ModelType.NEGATIVE_LOSS_MODEL]
                adversarial_logit = adversarial_model(z_a_ph)

        else:
            z_ph = mu_a_ph
            if active_party_has_bottom_model:
                z_ph = torch.cat((mu_a_ph, mu_b_ph), dim=-1)

            # apply negative loss
            if self.apply_negative_loss:
                adversarial_model = model_dict[ModelType.NEGATIVE_LOSS_MODEL]
                adversarial_logit = adversarial_model(mu_a_ph)

        logit = active_task_model(z_ph) if active_task_model is not None else z_ph

        return (logit, adversarial_logit), (mu_a, mu_a_ph), (mu_b, mu_b_ph), (z_a, z_a_ph), (z_b, z_b_ph)

    def _backward_to_cutlayer(self, loss, mu_a_ph, mu_b_ph, z_a, z_a_ph, z_b, z_b_ph, model_dict, optimizer_dict,
                              y=None):

        passive_interactive_layer = model_dict[ModelType.PASSIVE_INTERACTIVE_LAYER]
        bottom_model_active = model_dict[ModelType.ACTIVE_BOTTOM]
        task_model_active = model_dict[ModelType.TASK_MODEL]

        active_party_has_bottom_model = True if bottom_model_active is not None else False
        has_interactive_layer = True if passive_interactive_layer is not None else False
        has_task_model = True if task_model_active is not None else False

        if has_task_model:
            task_model_optimizer = optimizer_dict[ModelType.TASK_MODEL]
            self.update_model(model_optimizer=task_model_optimizer, loss=loss)
        else:
            loss.backward()

        if has_interactive_layer:
            passive_interactive_layer_optimizer = optimizer_dict[ModelType.PASSIVE_INTERACTIVE_LAYER]

            mu_b_grad = None
            if active_party_has_bottom_model:
                z_a_grad = z_a_ph.grad
                z_b_grad = z_b_ph.grad

                if self.apply_protection_to_passive_party:
                    protected_z_grad = self.apply_protection(y, z_a_grad)
                    z_a_grad = protected_z_grad
                else:
                    z_grad = torch.cat((z_a_grad, z_b_grad), dim=-1)
                    protected_z_grad = self.apply_protection(y, z_grad)
                    z_a_grad = protected_z_grad[:, :z_a_grad.shape[-1]]
                    z_b_grad = protected_z_grad[:, z_a_grad.shape[-1]:]
                z_a_grad = z_a_grad.to(self.device)
                z_b_grad = z_b_grad.to(self.device)

                passive_intr_layer_loss = keep_predict_loss(z_a, z_a_grad)
                self.update_model(model_optimizer=passive_interactive_layer_optimizer, loss=passive_intr_layer_loss)
                mu_a_grad = mu_a_ph.grad

                active_interactive_layer_optimizer = optimizer_dict[ModelType.ACTIVE_INTERACTIVE_LAYER]
                active_intr_layer_loss = keep_predict_loss(z_b, z_b_grad)
                self.update_model(model_optimizer=active_interactive_layer_optimizer, loss=active_intr_layer_loss)
                mu_b_grad = mu_b_ph.grad
            else:
                z_a_grad = z_a_ph.grad
                protected_z_a_grad = self.apply_protection(y, z_a_grad).to(self.device)
                passive_intr_layer_loss = keep_predict_loss(z_a, protected_z_a_grad)
                self.update_model(model_optimizer=passive_interactive_layer_optimizer, loss=passive_intr_layer_loss)
                mu_a_grad = mu_a_ph.grad

        else:
            # if has no interactive layer, z is the concatenation of mu_a and mu_b
            mu_b_grad = None
            if active_party_has_bottom_model:
                mu_b_grad = mu_b_ph.grad
                mu_a_grad = mu_a_ph.grad

                if self.apply_protection_to_passive_party:
                    protected_mu_a_grad = self.apply_protection(y, mu_a_grad)
                    mu_a_grad = protected_mu_a_grad
                else:
                    mu_grad = torch.cat((mu_a_grad, mu_b_grad), dim=-1)
                    protected_mu_grad = self.apply_protection(y, mu_grad)
                    mu_a_grad = protected_mu_grad[:, :mu_a_grad.shape[-1]]
                    mu_b_grad = protected_mu_grad[:, mu_b_grad.shape[-1]:]

                mu_a_grad = mu_a_grad.to(self.device)
                mu_b_grad = mu_b_grad.to(self.device)
            else:
                mu_a_grad = mu_a_ph.grad
                mu_a_grad = self.apply_protection(y, mu_a_grad).to(self.device)

        return mu_a_grad, mu_b_grad

    def apply_protection(self, y, grad):
        if "marvell" in self.apply_protection_fn.__str__():
            self.defense_dict['args'][self.protection_name]['y'] = y
            protected_grad = self.apply_protection_fn(grad, **self.defense_dict['args'][self.protection_name])
        else:
            protected_grad = self.apply_protection_fn(grad, **self.defense_dict['args'][self.protection_name])
        return protected_grad

    def update_local_models(self,
                            mu_a, mu_b,
                            mu_a_grad, mu_b_grad,
                            model_dict,
                            optimizer_dict):

        bottom_model_active = model_dict[ModelType.ACTIVE_BOTTOM]
        active_party_has_bottom_model = True if bottom_model_active is not None else False

        bottom_model_passive_optimizer = optimizer_dict[ModelType.PASSIVE_BOTTOM]
        passive_model_loss = keep_predict_loss(mu_a, mu_a_grad)
        self.update_model(model_optimizer=bottom_model_passive_optimizer, loss=passive_model_loss)

        if active_party_has_bottom_model:
            bottom_model_active_optimizer = optimizer_dict[ModelType.ACTIVE_BOTTOM]
            active_model_loss = keep_predict_loss(mu_b, mu_b_grad)
            self.update_model(model_optimizer=bottom_model_active_optimizer, loss=active_model_loss)
