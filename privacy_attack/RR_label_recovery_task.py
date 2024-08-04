import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.simple_dataset import SimpleDataset
from models.mlp_torch import LinearModel, weights_init
from privacy_attack.RR_util import estimated_labels_precision, compute_estimated_labels
from privacy_defense.defense_methods import DefenseName, DEFENSE_FUNCTIONS_DICT
from store_utils import ExperimentResultStructure, save_exp_result
from svfl.data_utils import get_default_credit_data, get_vehicle_data
from utils import set_random_seed, get_timestamp


def compute_true_residue(passive_data, active_data, active_label):
    criterion = nn.BCEWithLogitsLoss()
    act_func = nn.Sigmoid()

    passive_feat_dim = passive_data.shape[-1]
    active_feat_dim = active_data.shape[-1]

    passive_local_model = LinearModel(input_dim=passive_feat_dim, output_dim=1, bias=False)
    active_local_model = LinearModel(input_dim=active_feat_dim, output_dim=1, bias=True)

    passive_z = passive_local_model(passive_data)
    active_z = active_local_model(active_data)
    z = torch.add(passive_z, active_z)

    passive_z.retain_grad()
    active_z.retain_grad()
    z.retain_grad()

    pred = act_func(z)
    loss = criterion(pred, active_label)
    loss.backward()

    # print("gradients of model A:")
    # for param in passive_local_model.parameters():
    #     print(param, param.grad)
    #
    # print("gradients of model B:")
    # for param in active_local_model.parameters():
    #     print(param, param.grad)
    #
    # print("residue d:")
    # print(z.grad)

    residue_d = z.grad.detach()
    residue_passive = passive_z.grad.detach()
    residue_active = active_z.grad.detach()

    # print("[DEBUG] residue_d 0:", residue_d)
    # print("[DEBUG] residue_passive:", residue_passive)
    # print("[DEBUG] residue_active:", residue_active)

    passive_grad = list(passive_local_model.parameters())[-1].grad.detach()
    active_grad = list(active_local_model.parameters())[0].grad.detach()
    return residue_d, passive_grad, active_grad


def optimize_residue(data_loader, inpupt_dim, learning_rate=0.01, epoch=5):
    mse_criterion = nn.MSELoss()
    residue_model = LinearModel(input_dim=inpupt_dim, output_dim=1, bias=False)
    residue_model.apply(weights_init)

    optimizer = torch.optim.Adam(residue_model.parameters(), lr=learning_rate)

    # tqdm_train = tqdm(range(epoch), desc='Training')
    # for ep in tqdm_train:
    for ep in range(epoch):
        loss_list = list()
        for data, target in data_loader:
            optimizer.zero_grad()

            logit = residue_model(data)
            mse_loss = mse_criterion(logit, target)
            mse_loss.backward()

            optimizer.step()
            loss_list.append(mse_loss.item())
        # tqdm_train.set_postfix({"loss": np.mean(loss_list)})
        if ep % 100 == 0:
            print("ep: {}, loss: {}".format(ep, np.mean(loss_list)))

    return list(residue_model.parameters())[-1].detach().numpy()


def compute_model_grad(residue_d, data):
    grad = 0
    for d, x in zip(residue_d, data):
        grad += d * x
    return grad


def perform_label_leakage(comp_passive_data, comp_active_data, comp_active_label,
                          mask_passive_data, mask_active_data, mask_active_label,
                          batch_size, C, optimizer_dict, compute_true_residue_fn,
                          apply_protection_fn=None, defense_args=None):
    residue_d, passive_grad_true, active_grad_true = compute_true_residue_fn(comp_passive_data, comp_active_data,
                                                                             comp_active_label)
    # print("residue_d:", residue_d, residue_d.shape)

    if "marvell" in apply_protection_fn.__str__():
        raise Exception("Does not support Marvel against residue reconstruction attack for now.")
    residue_d = apply_protection_fn(residue_d, **defense_args)

    # === mask residue_d with zeros ===
    if C > batch_size:
        residue_zeros = np.zeros((C - batch_size, 1))
        # residue_d = np.concatenate((residue_d[0:fl_batch_size], residue_zeros), axis=0)
        residue_d = np.concatenate((residue_d, residue_zeros), axis=0)
    residue_d = torch.tensor(residue_d).float()
    # print("[DEBUG] residue_d:", residue_d, residue_d.shape)

    # if apply_protection_fn is not None:
    residue_d = apply_protection_fn(residue_d, **defense_args)

    # === each party computes gradient of local model ===
    passive_grad = compute_model_grad(residue_d, mask_passive_data)
    active_grad = compute_model_grad(residue_d, mask_active_data)

    # print("[DEBUG] passive_grad_true:", passive_grad_true)
    # print("[DEBUG] passive_grad:", passive_grad)
    # print("[DEBUG] active_grad_true:", active_grad_true)
    # print("[DEBUG] active_grad:", active_grad)

    # === passive party perform residual reconstruction attack         ===
    # === assuming passive party only knows the batch-sum passive grad ===
    passive_data_T = torch.transpose(mask_passive_data, 0, 1)
    input_dim = passive_data_T.shape[-1]
    # print("[DEBUG] data_B_T:", data_B_T, data_B_T.shape)
    # print("[DEBUG] data_B_T input_dim:", input_dim)

    passive_feat_dim = mask_passive_data.shape[-1]
    residue_label = passive_grad.reshape(passive_feat_dim, -1)
    # print("[DEBUG] residue_label:", residue_label)

    # === prepare dataset for residue reconstruction ===
    rr_dataset = SimpleDataset(passive_data_T.detach().numpy(), residue_label.detach().numpy(), label_type=torch.float)

    bs = optimizer_dict["batch_size"]
    lr = optimizer_dict["learning_rate"]
    ep = optimizer_dict["epoch"]
    data_loader = DataLoader(rr_dataset, batch_size=bs, shuffle=True)
    estimated_residue = optimize_residue(data_loader=data_loader, inpupt_dim=input_dim, learning_rate=lr, epoch=ep)

    ################################
    # Compute label recovery result
    ################################
    true_residue = residue_d.detach().numpy().flatten()
    estimated_residue = estimated_residue.flatten()
    estimated_labels = compute_estimated_labels(estimated_residue)

    return estimated_labels_precision(estimated_labels, mask_active_label.flatten())


if __name__ == '__main__':
    criterion = nn.BCEWithLogitsLoss()
    mse_criterion = nn.MSELoss()
    act_func = nn.Sigmoid()

    result_list = list()
    residue_optimizer_dict = {"batch_size": 32, "learning_rate": 0.002, "epoch": 500}

    # fl_batch_size_list = [8, 16, 32, 64, 128, 256, 512]
    # # num_attempts_list = [64, 32, 16, 8, 4, 2, 1]
    # num_attempts_list = [20, 20, 20, 20, 20, 20, 20]
    # # fl_batch_size_list = [8, 16, 48, 128, 256, 512]
    # # num_attempts_list = [64, 32, 11, 8, 4, 2, 1]
    # # multiple_list = [0, 2, 4, 6]
    # multiple_list = [0, 1, 3, 5, 7]
    # # num_attempts = 10

    # fl_batch_size_list = [128, 256, 2048, 4096, 8192]
    # num_attempts_list = [10, 10, 10, 10, 10]
    # multiple_list = [0]

    fl_batch_size_list = [64]
    num_attempts_list = [20]
    multiple_list = [0]

    ################
    # Prepare data
    ################
    # data_dir = "/Users/yankang/Documents/Data/"
    data_dir = "E:\\dataset\\"
    data_dir_dict = {"default_credit": [get_default_credit_data],
                     "vehicle": [get_vehicle_data]}

    dataset_list = ["default_credit"]
    # dataset_list = ["vehicle"]
    # dataset_list = ["default_credit", "vehicle"]

    defense_name = DefenseName.NONE
    apply_protection = DEFENSE_FUNCTIONS_DICT[defense_name]
    # defense_args = {"noise_scale": 0.001}
    # defense_args = {}
    # defense_args = {"gc_percent": 0.5}
    defense_args = {"ratio": 5}

    run_ts = get_timestamp()

    #######################################
    # Start residue reconstruction attack #
    #######################################
    for dataset in dataset_list:

        data_A, data_B, label_B = data_dir_dict[dataset][0](data_dir, shuffle=False)

        result_dict = dict()
        for attempts, batch_size in zip(num_attempts_list, fl_batch_size_list):

            result_dict[str(batch_size)] = dict()
            for multiple in multiple_list:

                C = (multiple + 1) * batch_size
                print("C: {}, fl_batch_size: {}".format(C, batch_size))
                # print("residue_optimizer_dict: ", residue_optimizer_dict)

                acc_list = []
                auc_list = []
                for seed in range(attempts):
                    set_random_seed(seed)
                    print("===> seed:", seed)
                    # randomly select batch_size number of data
                    indices = np.random.choice(data_A.shape[0], C)

                    tensor_data_A = torch.tensor(data_A[indices[:batch_size]]).float()
                    tensor_data_B = torch.tensor(data_B[indices[:batch_size]]).float()
                    tensor_label_B = torch.tensor(label_B[indices[:batch_size]].reshape(-1, 1)).float()

                    m_tensor_data_A = torch.tensor(data_A[indices]).float()
                    m_tensor_data_B = torch.tensor(data_B[indices]).float()
                    m_tensor_label_B = torch.tensor(label_B[indices].reshape(-1, 1)).float()

                    #########################
                    # Perform label leakage #
                    #########################
                    acc, auc = perform_label_leakage(comp_passive_data=tensor_data_A,
                                                     comp_active_data=tensor_data_B,
                                                     comp_active_label=tensor_label_B,
                                                     mask_passive_data=m_tensor_data_A,
                                                     mask_active_data=m_tensor_data_B,
                                                     mask_active_label=m_tensor_label_B,
                                                     batch_size=batch_size,
                                                     C=C,
                                                     optimizer_dict=residue_optimizer_dict,
                                                     compute_true_residue_fn=compute_true_residue,
                                                     apply_protection_fn=apply_protection,
                                                     defense_args=defense_args)
                    print("[INFO] ACC:{}, AUC:{}".format(acc, auc))

                    acc_list.append(acc)
                    auc_list.append(auc)

                mean_acc, mean_auc = np.mean(acc_list), np.mean(auc_list)
                result_dict[str(batch_size)][str(multiple)] = {"mean_acc": mean_acc, "mean_auc": mean_auc}
                result = dataset + "_fbs" + str(batch_size) + "_C" + str(C) + "_acc" + str(mean_acc) + "_auc" + str(
                    mean_auc)
                result_list.append(result)

            print("results: \n", result_list)

        rr_exp_dict = {"exp_result": result_dict,
                       "batch_size_list": fl_batch_size_list,
                       "multiple_list": multiple_list}

        save_exp_result_dir = ExperimentResultStructure.create_residue_recons_task_subdir_path(dataset)
        os.makedirs(save_exp_result_dir, exist_ok=True)
        print("[INFO] save experimental result dir:", save_exp_result_dir)

        args_str = "_".join([str(key) + str(value) for key, value in defense_args.items()])
        if defense_name != DefenseName.NONE:
            save_exp_result_filename = "rr_attack_{}_{}_{}".format(str(defense_name), args_str, run_ts)
        else:
            save_exp_result_filename = "rr_attack_{}_{}".format(str(defense_name), run_ts)
        save_exp_result(rr_exp_dict, dir=save_exp_result_dir, filename=save_exp_result_filename)
        print("saved experimental results to {}".format(save_exp_result_filename))
