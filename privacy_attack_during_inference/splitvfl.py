# -*- coding: utf-8 -*-
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from privacy_attack_during_inference.splitvfl_models.layers.passportconv2d_nonsignloss import PassportBlock
from splitvfl_model_config import get_models
from splitvfl_models.lenet_classifer import Classifier
from splitvfl_utils import save_exp_result
from utils import cross_entropy_for_one_hot, label_to_one_hot


class SplitVFL(object):

    def __init__(self, hyperparameter_dict, encoder=None, device="cpu"):
        # self.exp_df = pd.DataFrame(
        #     columns=['exp_id', 'dataset', 'num_class', "batch_size", "rec_rate", "rec_rate_mean", "rec_rate_std"])
        # self.dataset_name = dataset_name
        self.device = device
        self.args = None

        self.image_half_dim = hyperparameter_dict.get("image_half_dim")
        self.num_classes = hyperparameter_dict["num_classes"]
        self.optim_name = hyperparameter_dict["optimizer_name"]
        self.max_epochs = hyperparameter_dict["max_epochs"]
        self.model_name = hyperparameter_dict["model_name"]
        # self.dataset_name = hyperparameter_dict["dataset_name"]
        self.exp_dir = hyperparameter_dict["exp_dir"]
        self.seed = hyperparameter_dict["seed"]
        self.lambda_cf = hyperparameter_dict["lambda_cf"]
        self.apply_negative_loss = hyperparameter_dict["apply_negative_loss"]

        self.lr_inner = hyperparameter_dict["learning_rate"]
        self.wd = hyperparameter_dict["weight_decay"]
        self.neg_lr = hyperparameter_dict["neg_lr"]
        self.neg_wd = hyperparameter_dict["neg_wd"]

        self.aggregate_mode = hyperparameter_dict["aggregate_mode"]
        print("hyperparameter_dict:", json.dumps(hyperparameter_dict, indent=4))

        self.active_party_has_model = True
        self.net_passive = None
        self.net_active = None
        self.intr_passive = None
        self.intr_active = None
        self.top_model = None
        self.classifier = None
        self.pp_block = None

        # self.main_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # self.passive_criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.main_criterion = cross_entropy_for_one_hot
        self.negative_criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # self.negative_criterion = torch.nn.KLDivLoss().to(self.device)

        self.logs = {'train_acc': [],
                     'test_acc': [],
                     'test_loss': [],
                     'train_loss': [],
                     'best_test_acc': -np.inf,
                     'best_test_loss': None,
                     'best_model': [],
                     'sigma': [],
                     'local_loss': [],
                     'avg_norm': [],
                     'all_model': [],
                     # 'model_keys': list(self.net_t.state_dict().keys()),
                     'scale': [],
                     'bias': []
                     }

        self.encoder = encoder

    def save_models(self, model_dir):

        if self.net_active is not None:
            n_pp_net_active = self.net_active.get_number_passports()
            model_active_name = f"{self.model_name}_active" + f"_pp{n_pp_net_active}" + f"_{self.aggregate_mode}" + f"_seed{self.seed}" + ".pkl"
            torch.save(self.net_active.state_dict(), model_dir + model_active_name)
            print(f"saved models:{model_dir}/{model_active_name}")

        n_pp_net_passive = self.net_passive.get_number_passports()
        n_pp_top_model = self.top_model.get_number_passports()

        model_passive_name = f"{self.model_name}_passive" + f"_pp{n_pp_net_passive}" + f"_{self.aggregate_mode}" + f"_seed{self.seed}" + ".pkl"
        top_model_name = "top_model" + f"_pp{n_pp_top_model}" + f"_{self.aggregate_mode}" + f"_seed{self.seed}" + ".pkl"

        torch.save(self.net_passive.state_dict(), model_dir + model_passive_name)
        torch.save(self.top_model.state_dict(), model_dir + top_model_name)
        print(f"saved models:{model_dir}/{model_passive_name}")
        print(f"saved models:{model_dir}/{top_model_name}")

    def aggregate(self, logits_a, logits_b, scale_list, bias_list):
        if logits_b is not None:
            z_a = self.intr_passive(logits_a)
            z_b = self.intr_active(logits_b)
            z = z_a + z_b
            if self.top_model:
                # if self.aggregate_mode == "cat":
                #     # print("logits_a.shape:", logits_a.shape)
                #     # print("logits_b.shape:", logits_b.shape)
                #     z_a = self.intr_passive(logits_a)
                #     z_b = self.intr_active(logits_b)
                #     logits = z_a + z_b
                # elif self.aggregate_mode == "add":
                #     logits = logits_a + logits_b
                # else:
                #     raise Exception(f"does not support : {self.aggregate_mode}")
                # # print("agg scale_list:", len(scale_list))
                # # print("agg bias_list:", len(bias_list), bias_list[0].shape)
                return self.top_model(z, scale_list, bias_list)
            else:
                return z
        else:
            z_a = self.intr_passive(logits_a)
            return self.top_model(z_a) if self.top_model else z_a

    def validate(self, test_ldr, split_data, apply_negative_loss=False):
        self.model_to_eval()

        with torch.no_grad():

            all_predict_label_list = list()
            all_actual_label_list = list()
            suc_cnt = 0
            sample_cnt = 0

            # total = 0
            # correct = 0
            val_loss = 0
            val_passive_loss = 0
            for batch_idx, (x, y) in enumerate(test_ldr):
                x, y = x.to(self.device), y.to(self.device)

                if split_data:
                    batch_data_a, batch_data_b = self.split_data(x)
                else:
                    batch_data_a = x
                    batch_data_b = None

                gt_val_one_hot_label = label_to_one_hot(y, self.num_classes).to(self.device)

                logits, passive_logits = self.forward(batch_data_a, batch_data_b, apply_negative_loss)
                loss = self.main_criterion(logits, gt_val_one_hot_label)
                val_loss += loss.item()

                predict_prob = F.softmax(logits, dim=-1)
                if self.encoder is not None:
                    dec_predict_prob = self.encoder.decoder(predict_prob)
                    # print("[DEBUG] predict_prob:", predict_prob)
                    # print("[DEBUG] dec_predict_prob:", dec_predict_prob)
                    predict_label = torch.argmax(dec_predict_prob, dim=-1)
                else:
                    predict_label = torch.argmax(predict_prob, dim=-1)

                sample_cnt += predict_label.shape[0]
                actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                for predict, actual in zip(predict_label, actual_label):
                    # enc_result_matrix[actual, enc_predict] += 1
                    # result_matrix[actual, predict] += 1
                    all_predict_label_list.append(predict)
                    all_actual_label_list.append(actual)
                    if predict == actual:
                        suc_cnt += 1

                if apply_negative_loss:
                    # negative_loss_1 = 1 / entropy(F.softmax(passive_logits, dim=1))
                    # negative_loss_2 = 1 / self.negative_criterion(passive_logits, y)
                    # negative_loss = negative_loss_1 + negative_loss_2
                    negative_loss = 1 / self.negative_criterion(passive_logits, y)
                    val_passive_loss += negative_loss.item()

                # _, predicted = logits.max(1)
                # total += y.size(0)
                # correct += predicted.eq(y).sum().item()

            # test_acc = 1. * correct / total
            val_loss = val_loss / (batch_idx + 1)
            val_passive_loss = val_passive_loss / (batch_idx + 1)

            val_auc = -1
            if self.num_classes == 2:
                val_auc = roc_auc_score(y_true=all_actual_label_list, y_score=all_predict_label_list)
            val_acc = suc_cnt / float(sample_cnt)
            val_result = {"val_loss": val_loss, "val_passive_loss": val_passive_loss,
                          "val_acc": val_acc, "val_auc": val_auc}

        return val_result

    def get_model_parameter(self):
        if self.net_active is not None:
            return list(self.net_passive.parameters()) + list(self.net_active.parameters()) \
                   + list(self.intr_active.parameters()) + list(self.intr_passive.parameters()) \
                   + list(self.top_model.parameters())
        else:
            return list(self.net_passive.parameters()) + list(self.intr_active.parameters()) \
                   + list(self.intr_passive.parameters()) + list(self.top_model.parameters())

    # def get_neg_model_parameter(self):
    #     if apply_negative_loss:
    #         if self.net_active is not None:
    #             return list(self.net_passive.parameters()) + list(self.net_active.parameters()) + list(
    #                 self.top_model.parameters()) + list(self.classifier.parameters())
    #         else:
    #             return list(self.net_passive.parameters()) + list(
    #                 self.top_model.parameters()) + list(self.classifier.parameters())
    #     else:
    #         if self.net_active is not None:
    #             return list(self.net_passive.parameters()) + list(self.net_active.parameters()) + list(
    #                 self.top_model.parameters())
    #         else:
    #             return list(self.net_passive.parameters()) + list(self.top_model.parameters())

    def model_to_train(self):
        self.net_passive.train()
        if self.net_active is not None:
            self.net_active.train()
        self.intr_passive.train()
        self.intr_active.train()
        self.top_model.train()
        self.classifier.train()
        self.pp_block.train()

    def model_to_eval(self):
        self.net_passive.eval()
        if self.net_active is not None:
            self.net_active.eval()
        self.intr_passive.eval()
        self.intr_active.eval()
        self.top_model.eval()
        self.classifier.eval()
        self.pp_block.eval()

    def create_optimizer(self, apply_negative_loss):
        # net_a is passive party, net_b is active party
        parameters = self.get_model_parameter()
        if self.optim_name == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=self.lr_inner, momentum=0.9, weight_decay=self.wd)
            neg_optim = torch.optim.Adam(list(self.classifier.parameters()), lr=self.neg_lr,
                                         weight_decay=self.neg_wd) if apply_negative_loss else None
        elif self.optim_name == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=self.lr_inner, weight_decay=self.wd)
            neg_optim = torch.optim.Adam(list(self.classifier.parameters()), lr=self.neg_lr,
                                         weight_decay=self.neg_wd) if apply_negative_loss else None
        else:
            raise ValueError('Optimizer should be in [sgd, adam]')
        return optimizer, neg_optim

    def split_data(self, x):
        batch_data_a = x[:, :, :self.image_half_dim]
        batch_data_b = x[:, :, self.image_half_dim:]
        return batch_data_a, batch_data_b

    def forward(self, data_a, data_b, apply_negative_loss=False):

        mu_a = self.net_passive(data_a)
        mu_b = None
        if self.net_active is not None:
            mu_b = self.net_active(data_b)
        # scale_list = self.net_active.scales
        # bias_list = self.net_active.biases
        scale_list = None
        bias_list = None
        # logits_a, _, _ = self.pp_block(logits_a)

        if mu_b is not None:
            z_a = self.intr_passive(mu_a)
            z_b = self.intr_active(mu_b)
            z = z_a + z_b
            logit = self.top_model(z, scale_list, bias_list) if self.top_model else z
        else:
            z_a = self.intr_passive(mu_a)
            logit = self.top_model(z_a) if self.top_model else z_a

        if apply_negative_loss:
            return logit, self.classifier(z_a)
        else:
            return logit, None

    def init_models(self, passport_args):
        self.net_passive, self.net_active, self.intr_passive, self.intr_active, self.top_model = get_models(passport_args)
        self.net_passive.to(self.device)
        if self.net_active is not None:
            self.net_active.to(self.device)

        self.intr_passive.to(self.device)
        self.intr_active.to(self.device)
        self.top_model.to(self.device)
        self.classifier = Classifier(120, 10).to(self.device)
        # self.classifier = Classifier(128, 10).to(self.device)
        # TODO: test
        self.pp_block = PassportBlock(12, 12, 3, 1, 1, "bn").to(self.device)

    def train_batch(self, batch_data_a, batch_data_b, batch_label, optimizer, neg_optim, apply_negative_loss=False):
        one_hot_batch_label = label_to_one_hot(batch_label, self.num_classes).to(self.device)

        if self.encoder is not None:
            _, _, gt_one_hot_label = self.encoder(one_hot_batch_label)
        else:
            gt_one_hot_label = one_hot_batch_label

        self.model_to_train()

        logits, passive_logits = self.forward(batch_data_a, batch_data_b, apply_negative_loss)

        # print("top model output logits shape:", logits.shape)
        # print("top model output label shape:", label.shape)
        loss = self.main_criterion(logits, gt_one_hot_label)
        main_loss_val = loss.item()
        negative_loss_val = 0
        if apply_negative_loss:
            # negative_loss_1 = 1 / entropy(F.softmax(passive_logits, dim=1))
            # negative_loss_2 = 1 / self.negative_criterion(passive_logits, batch_label)
            # negative_loss = negative_loss_1 + negative_loss_2
            negative_loss = 1 / self.negative_criterion(passive_logits, batch_label)
            loss = loss + self.lambda_cf * negative_loss
            negative_loss_val = negative_loss.item()

        # if neg_optim is not None:
        #     neg_optim.zero_grad()
        optimizer.zero_grad()

        loss.backward()

        # if neg_optim is not None:
        #     neg_optim.step()
        optimizer.step()

        return loss.item(), main_loss_val, negative_loss_val

    def train(self, train_loader, test_loader, split_data=True):

        optimizer, neg_optim = self.create_optimizer(self.apply_negative_loss)

        epoch_loss = []
        epoch_main_loss = []
        epoch_negative_loss = []
        for epoch in range(self.max_epochs):
            total_loss = 0
            total_main_loss = 0
            total_negative_loss = 0

            batch_idx = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)

                if split_data:
                    batch_data_a, batch_data_b = self.split_data(x)
                else:
                    batch_data_a = x
                    batch_data_b = None
                loss_val, main_loss_val, negative_loss_val = self.train_batch(batch_data_a,
                                                                              batch_data_b,
                                                                              y,
                                                                              optimizer,
                                                                              neg_optim,
                                                                              self.apply_negative_loss)

                total_loss += loss_val
                total_main_loss += main_loss_val
                total_negative_loss += negative_loss_val

            epoch_loss.append(total_loss / (batch_idx + 1))
            epoch_main_loss.append(total_main_loss / (batch_idx + 1))
            epoch_negative_loss.append(total_negative_loss / (batch_idx + 1))

            if (epoch + 1) == self.max_epochs or (epoch + 1) % 1 == 0:
                # val_result = {"val_loss": test_loss, "passive_test_loss": passive_test_loss,
                #               "val_acc": val_acc, "val_auc": val_auc}
                train_result_dict = self.validate(train_loader, split_data=split_data,
                                                  apply_negative_loss=self.apply_negative_loss)
                loss_train = train_result_dict["val_loss"]
                acc_train = train_result_dict["val_acc"]
                auc_train = train_result_dict["val_auc"]
                # loss_train, acc_train, passive_loss_train = self.validate(train_loader,
                #                                                           split_data=split_data,
                #                                                           apply_passive_loss=self.apply_passive_loss)
                test_result_dict = self.validate(test_loader, split_data=split_data,
                                                 apply_negative_loss=self.apply_negative_loss)
                loss_test = test_result_dict["val_loss"]
                acc_test = test_result_dict["val_acc"]
                auc_test = test_result_dict["val_auc"]
                passive_loss = test_result_dict["val_passive_loss"]

                self.logs['test_loss'].append(loss_test)
                self.logs['test_acc'].append(acc_test)

                if self.logs['best_test_acc'] < acc_test:
                    self.logs['best_test_acc'] = acc_test
                    self.logs['best_test_loss'] = loss_test
                    # self.logs['best_model'] = []
                    self.save_models(self.exp_dir)
                    save_exp_result({
                        'epoch': epoch + 1,
                        'log': self.logs
                    }, dir=self.exp_dir, filename="exp_result")

                epoch_str = 'Epoch {}/{}'.format(epoch, self.max_epochs)
                loss_str = "Train Loss {:.4f} -- Test Loss {:.4f} -- Passive Loss {:.4f}".format(loss_train, loss_test,
                                                                                                 passive_loss)
                acc_str = "Train acc {:.4f} -- Test acc {:.4f} -- Best acc {:.4f}".format(acc_train, acc_test,
                                                                                          self.logs['best_test_acc'])
                print(epoch_str + ": " + loss_str + "; " + acc_str)
                # print('Epoch {}/{}'.format(epoch, self.max_epochs))
                # print("Train Loss {:.4f} -- Test Loss {:.4f}".format(loss_train, loss_test))
                # print("Train acc {:.4f} -- Test acc {:.4f} -- Best acc {:.4f}".format(acc_train, acc_test,
                #                                                                       self.logs['best_test_acc']))
