import sys

sys.path.append("./")
sys.path.append("../")
import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from arch_utils import ModelType, get_architecture_config, build_layer_model
from passport_config import get_passport_config
from splitvfl_data import select_samples_by_labels, select_samples_by_labels_from_2party_dataset
from splitvfl_models.lenet_classifer import Classifier
from splitvfl_utils import get_number_passports, set_seed, load_model, save_models, save_exp_result
from store_utils import get_path_slash, ExperimentResultStructure
from svfl.data_utils import fetch_parties_data, get_dataset
from utils import str2bool


def frozen_net(model_list, frozen):
    for model in model_list:
        for param in model.parameters():
            param.requires_grad = not frozen


def prepare_models(attack_mode, arch_config, vfl_type, vfl_model_dir, passport_args, train_all, load_pretrained_model,
                   num_classes, seed, lr, wd):
    top_model_pp_args = passport_args["top_model_force_passport"]
    active_bottom_pp_args = passport_args["active_model_force_passport"]
    passive_bottom_pp_args = passport_args["passive_model_force_passport"]

    model_dict = dict()
    train_model_dict = dict()
    if attack_mode == "model_complete":
        print("[INFO] Prepare models for [Model Completion].")
        # === For model complete attack, we must have a task model (Not necessarily true, need further investigation).
        # === Model complete attack (i.e., the passive party) does not involve a active bottom model.
        # === In model complete attack, the attack trains the task model and the (pre-trained) passive bottom model,
        # === and then leverages the two models to perform label prediction on samples.

        passive_bottom_model = build_layer_model(arch_config["passive_bottom_config"])
        model_name = f"passive_bottom_checkpoint.pth"

        # passive_num_pp = get_number_passports(passive_bottom_pp_args)
        # passive_bottom_model = LeNetWithAct(1, passport_pos=passive_bottom_pp_args)
        # passive_bottom_model = get_lenet(struct_config=None)
        # model_name = f"lenet_passive_pp" + passive_num_pp + "_add" + f"_seed{seed}.pkl"

        if load_pretrained_model:
            load_model(passive_bottom_model, model_name, vfl_model_dir)

        # top_num_pp = get_number_passports(top_model_pp_args)
        # active_task_model = LeNetClassifier(num_classes=10, passport_pos=top_model_pp_args, act="relu")
        # model_name = f"top_model_pp" + top_num_pp + "_cat" + f"_seed{seed}.pkl"
        # load_model(active_task_model, model_name, vfl_model_dir)
        task_model_input_dim = arch_config['task_model_config']['struct_config']['layer_input_dim_list'][0]
        if arch_config['active_bottom_config'] is not None:
            task_model_input_dim = int(task_model_input_dim / 2)
        active_task_model = Classifier(task_model_input_dim, num_classes)
        # active_task_model = Classifier(120, 10)
        # active_task_model = Classifier(336, 10)

        if train_all:
            print("[INFO] train task model and passive bottom model.")
            parameters = list(passive_bottom_model.parameters()) + list(active_task_model.parameters())
            optimizer = optim.Adam(parameters, lr=lr, weight_decay=wd)
            train_model_dict[ModelType.PASSIVE_BOTTOM] = passive_bottom_model

            # === train only passive bottom model
            # parameters = list(passive_bottom_model.parameters())
            # optimizer = optim.Adam(parameters, lr=0.01, weight_decay=1e-6)
            # frozen_net([active_task_model], True)
        else:
            print("[INFO] train only task model.")
            parameters = list(active_task_model.parameters())
            optimizer = optim.Adam(parameters, lr=lr, weight_decay=wd)
            frozen_net([passive_bottom_model], True)
        train_model_dict[ModelType.TASK_MODEL] = active_task_model
        active_bottom_model = None

    elif attack_mode == "model_inversion":
        print("[INFO] Prepare models for [Model Inversion].")
        # For model inversion attack, we must have a task model and passive bottom model.
        # For VHNN scenario, we have a active bottom model, while for VSNN scenario, we have no active bottom model.
        # The task model and the active bottom model will be frozen when conduct the model inversion attack training.
        # Thus, only the passive bottom model will be training for model inversion.

        active_task_model = build_layer_model(arch_config["task_model_config"])
        model_name = f"task_model_checkpoint.pth"
        load_model(active_task_model, model_name, vfl_model_dir)
        if vfl_type == "VHNN":
            active_bottom_model = build_layer_model(arch_config["active_bottom_config"])
            model_name = f"active_bottom_checkpoint.pth"

            # active_bottom_num_pp = get_number_passports(active_bottom_pp_args)
            # active_bottom_model_name = f"lenet_active_pp" + active_bottom_num_pp + "_cat" + f"_seed{seed}"
            # active_bottom_model = build_layer_model(arch_config["active_bottom_config"])
            load_model(active_bottom_model, model_name, vfl_model_dir)

            frozen_net([active_task_model, active_bottom_model], True)
        else:
            # VSNN
            active_bottom_model = None
            frozen_net([active_task_model], True)

        # passive bottom model is to be trained in model inversion attack
        # passive_bottom_model = LeNet(1, passport_pos={'0': False, '1': False, '2': False})
        passive_bottom_model = build_layer_model(arch_config["passive_bottom_config"])
        parameters = list(passive_bottom_model.parameters())
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=wd)
        train_model_dict[ModelType.PASSIVE_BOTTOM] = passive_bottom_model

        # model_dict["passive_bottom_model"] = passive_bottom_model
    else:
        raise Exception("Does not support attack mode:{}".format(attack_mode))

    model_dict[ModelType.PASSIVE_BOTTOM] = passive_bottom_model
    model_dict[ModelType.ACTIVE_BOTTOM] = active_bottom_model
    model_dict[ModelType.TASK_MODEL] = active_task_model

    return model_dict, train_model_dict, optimizer


def test_shadow(model_dict, dataloader, data_dict, device):
    num_classes = data_dict["num_classes"]
    dataset_name = data_dict["dataset_name"]

    for m in model_dict.values():
        if m is not None:
            m.eval()

    predict_y = list()
    predict_prob_y = list()
    actual_y = list()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            y = y.to(device)
            x_passive, x_active = fetch_parties_data(dataset_name=dataset_name,
                                                     data=x,
                                                     device=device)

            logit = forward(model_dict, x_passive, x_active)
            correct += torch.sum((torch.argmax(logit, dim=1) == y).float()).item()
            total += y.size(0)

            predict_prob = F.softmax(logit, dim=-1)
            predict_labels = torch.argmax(predict_prob, dim=1)
            for predict, pred_prob, actual in zip(predict_labels, predict_prob, y):
                predict_y.append(predict.cpu().item())
                if num_classes == 2:
                    predict_prob_y.append(pred_prob[1].cpu().item())
                actual_y.append(actual.cpu().item())

    acc = accuracy_score(y_true=actual_y, y_pred=predict_y)
    if num_classes == 2:
        auc = roc_auc_score(y_true=actual_y, y_score=predict_prob_y)
    else:
        auc = 0.0
    return {"auc": auc, "acc": acc}


def forward(model_dict, x_passive, x_active):
    feat_passive = model_dict[ModelType.PASSIVE_BOTTOM](x_passive)
    if model_dict[ModelType.ACTIVE_BOTTOM] is not None:
        feat_active = model_dict[ModelType.ACTIVE_BOTTOM](x_active)
        feat = torch.cat([feat_passive, feat_active], dim=-1)
        logit = model_dict[ModelType.TASK_MODEL](feat)
    else:
        logit = model_dict[ModelType.TASK_MODEL](feat_passive)
    return logit


def train_shadow(attack_mode, data_dict, train_model_dict, model_dict, optimizer, save_model_dir, max_epoch, metric,
                 device="cpu"):
    data_loader_dict = data_dict["data_loader_dict"]
    dataset_name = data_dict["dataset_name"]
    train_data_loader = data_loader_dict["train_data_loader"]
    val_data_loader = data_loader_dict["val_data_loader"]
    test_data_loader = data_loader_dict["test_data_loader"]

    criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0.
    test_acc_list = []
    best_metric = {metric: 0, 'best_epoch': 0}
    # train_models_dict = {"model_inversion": [ModelType.PASSIVE_BOTTOM],
    #                      "model_complete": [ModelType.PASSIVE_BOTTOM, ModelType.TASK_MODEL]}
    # train_models_list = train_models_dict[attack_mode]
    print("[DEBUG] train models:{}".format(train_model_dict.keys()))
    # tqdm_train = tqdm(train_loader, desc='Training (epoch #{})'.format(i_epoch + 1))
    for epoch in tqdm(range(max_epoch)):

        for n, m in train_model_dict.items():
            if m is not None:
                m.train()

        losses, batch = 0., 0
        for x, y in train_data_loader:
            y = y.to(device)
            x_passive, x_active = fetch_parties_data(dataset_name=dataset_name, data=x, device=device)

            logit = forward(model_dict, x_passive, x_active)
            loss = criterion(logit, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()
            batch += 1
        avg_loss = losses / batch

        # ==== finished current epoch of training  ====
        # ==== and starting validation             ====
        val_result = test_shadow(model_dict, val_data_loader, data_dict, device)
        val_acc = val_result["acc"]
        val_auc = val_result["auc"]
        val_metric = val_result[metric]
        if val_metric > best_metric[metric]:
            best_metric[metric] = val_metric
            best_metric["best_epoch"] = epoch

            print(f"find current best model with val {metric}:[{best_metric[metric]}] at epoch:[{epoch}].")

            save_models(model_dict, save_model_dir, attack_mode)

        if (epoch + 1) % 10 == 0:  # early stop round = 20
            test_result = test_shadow(model_dict, test_data_loader, data_dict, device)
            test_metric = test_result[metric]
            print(f"test {metric}:[{test_metric}], at epoch:[{epoch}].")

        # print("epoch:{}, loss:{:.4f}, val acc:{:.4f}, val auc:{:.4f}, best {}:{:.4f}".format(epoch, avg_loss, val_acc,
        #                                                                                      val_auc, metric,
        #                                                                                      best_metric[metric]))
    return best_metric


def get_data_dict(attack_mode, dataset_name, **args):
    if attack_mode == "model_complete":
        print("[INFO] get data for model completion.")
        return get_mc_data_dict(dataset_name, **args)
    elif attack_mode == "model_inversion":
        print("[INFO] get data for model inversion.")
        return get_mi_data_dict(dataset_name, **args)
    else:
        raise Exception("Does not support attack mode:{}".format(attack_mode))


def get_mi_data_dict(dataset_name, **args):
    batch_size = args["batch_size"]
    attack_data_type = args["attack_data_type"]
    num_workers = args["num_workers"]

    train_set, val_set, test_set, input_dims, num_classes, pos_ratio, col_names = get_dataset(dataset_name=dataset_name,
                                                                                              **args)

    if num_classes == 10:
        label_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_sample_per_label = int(num_samples_ft / len(label_set))
        label_sample_dict = {str(lbl): num_sample_per_label for lbl in label_set}
    elif num_classes == 2:
        label_set = [0, 1]
        pos_num_samples = int(num_samples_ft * pos_ratio)
        neg_num_samples = num_samples_ft - pos_num_samples
        label_sample_dict = {"1": pos_num_samples, "0": neg_num_samples}
    else:
        raise Exception("does not support {} classes for now.")

    print("[INFO]: samples of labels dict:{}".format(label_sample_dict))

    if dataset_name in ['nuswide2', 'nuswide10', 'bhi', 'vehicle', 'default_credit', 'criteo']:
        train_set = select_samples_by_labels_from_2party_dataset(train_set, label_set, label_sample_dict)
    elif dataset_name in ['cifar2', 'cifar10', 'cifar100', 'mnist', 'fmnist']:
        train_set = select_samples_by_labels(train_set, label_set, label_sample_dict)
    else:
        raise Exception("Does not support dataset:{}".format(dataset_name))

    print("[INFO] train_set len:", len(train_set))
    print("[INFO] val_set len:", len(val_set))
    print("[INFO] test_set len:", len(test_set))

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)

    data_loader_dict = {"train_data_loader": train_dataloader,
                        "val_data_loader": val_dataloader,
                        "test_data_loader": test_dataloader}

    data_dict = dict()
    data_dict['dataset_name'] = dataset_name
    data_dict['num_classes'] = num_classes
    data_dict['input_dims'] = input_dims
    data_dict['data_loader_dict'] = data_loader_dict
    data_dict['col_names'] = col_names

    return data_dict


def get_mc_data_dict(dataset_name, **args):
    batch_size = args["batch_size"]
    # attack_data_type = args["attack_data_type"]
    num_samples_ft = args["num_samples_ft"]
    # pos_ratio = args["pos_ratio"]
    num_workers = args["num_workers"]
    train_set, val_set, test_set, input_dims, num_classes, pos_ratio, col_names = get_dataset(dataset_name=dataset_name,
                                                                                              **args)

    if num_classes == 10:
        label_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_sample_per_label = int(num_samples_ft / len(label_set))
        label_sample_dict = {str(lbl): num_sample_per_label for lbl in label_set}
    elif num_classes == 2:
        label_set = [0, 1]
        pos_num_samples = int(num_samples_ft * pos_ratio)
        neg_num_samples = num_samples_ft - pos_num_samples
        label_sample_dict = {"1": pos_num_samples, "0": neg_num_samples}
    else:
        raise Exception("does not support {} classes for now.")

    print("[INFO]: samples of labels dict:{}".format(label_sample_dict))

    if dataset_name in ['nuswide2', 'nuswide10', 'bhi', 'vehicle', 'default_credit', 'criteo']:
        train_set = select_samples_by_labels_from_2party_dataset(train_set, label_set, label_sample_dict)
    elif dataset_name in ['cifar2', 'cifar10', 'cifar100', 'mnist', 'fmnist']:
        train_set = select_samples_by_labels(train_set, label_set, label_sample_dict)
    else:
        raise Exception("Does not support dataset:{}".format(dataset_name))

    print("[INFO] train_set len:", len(train_set))
    print("[INFO] val_set len:", len(val_set))
    print("[INFO] test_set len:", len(test_set))

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=1024, shuffle=False, num_workers=num_workers, pin_memory=True)

    data_loader_dict = {"train_data_loader": train_dataloader,
                        "val_data_loader": val_dataloader,
                        "test_data_loader": test_dataloader}

    data_dict = dict()
    data_dict['dataset_name'] = dataset_name
    data_dict['num_classes'] = num_classes
    data_dict['input_dims'] = input_dims
    data_dict['data_loader_dict'] = data_loader_dict
    data_dict['col_names'] = col_names

    return data_dict


def main_task(exp_args, vfl_exp_data, load_vfl_model_dir, save_exp_result_dir, attack_data_type, device):
    train_all = exp_args.train_all
    load_pretrained_model = exp_args.load_pretrained_model
    if not load_pretrained_model:
        assert train_all is True

    attack_mode = exp_args.attack_mode.lower()

    vfl_train_seed = exp_args.seed
    vfl_type = vfl_exp_data["vfl_type"]
    arch_config_name = vfl_exp_data["arch_config_name"]

    dataset_name = vfl_exp_data["dataset"]
    num_classes = vfl_exp_data["num_classes"]
    is_imbal = vfl_exp_data["imbal"]
    has_active_bottom = vfl_exp_data["has_active_bottom"]
    has_interactive_layer = vfl_exp_data["has_interactive_layer"]
    task_model_type = vfl_exp_data["task_model_type"]
    eval_metric = "auc" if num_classes == 2 else "acc"
    args = {"data_dir": exp_args.data_dir, "num_classes": num_classes, "imbal": is_imbal,
            "batch_size": exp_args.batch_size, "num_workers": exp_args.num_workers,
            "attack_data_type": attack_data_type, "num_samples_ft": exp_args.num_samples_ft}

    passport_args = get_passport_config()

    set_seed(exp_args.seed)

    data_dict = get_data_dict(attack_mode, dataset_name, **args)
    input_dims = data_dict['input_dims']
    models_args = {'dnnfm': {'col_names': data_dict['col_names']}}

    arch_config = get_architecture_config(arch_config_name=arch_config_name,
                                          input_dims=input_dims,
                                          num_classes=num_classes,
                                          has_active_bottom=has_active_bottom,
                                          has_interactive_layer=has_interactive_layer,
                                          task_model_type=task_model_type,
                                          models_args=models_args)

    model_dict, train_model_dict, optimizer = prepare_models(attack_mode, arch_config, vfl_type, load_vfl_model_dir,
                                                             passport_args, train_all, load_pretrained_model,
                                                             num_classes,
                                                             vfl_train_seed, exp_args.lr, exp_args.wd)
    for k, v in model_dict.items():
        if v is not None:
            model_dict[k] = v.to(device)
    best_metric = train_shadow(attack_mode=attack_mode,
                               data_dict=data_dict,
                               model_dict=model_dict,
                               train_model_dict=train_model_dict,
                               optimizer=optimizer,
                               save_model_dir=save_exp_result_dir,
                               max_epoch=exp_args.max_epoch,
                               metric=eval_metric,
                               device=device)

    vfl_exp_data = dict(vfl_exp_data, **{"mc_result": best_metric})
    vfl_exp_data = dict(vfl_exp_data, **{"mc_args": vars(exp_args)})
    if train_all:
        save_exp_result(vfl_exp_data, save_exp_result_dir, "train_all_result")
    else:
        save_exp_result(vfl_exp_data, save_exp_result_dir, "result")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples_ft", type=int, default=40, help="total number of samples for model completion")
    parser.add_argument("--lr", type=float, default=0.005, help='learning rate')
    parser.add_argument("--wd", type=float, default=1e-6, help='weight decay')
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--attack_mode", type=str, default="model_inversion")
    parser.add_argument("--train_all", type=str2bool, default=False, help='whether to train all model')
    parser.add_argument("--load_pretrained_model", type=str2bool, default=True,
                        help='whether to load pretrained model')

    exp_args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splitter = get_path_slash()

    print("[DEBUG] exp_args.exp_path:{}".format(exp_args.exp_path))
    print("[DEBUG] path:", exp_args.exp_path.split(splitter))

    vfl_exp_data = json.load(open(exp_args.exp_path))
    print("[INFO] vfl_exp_data:{}".format(vfl_exp_data))
    vfl_model_dir = splitter.join(exp_args.exp_path.split(splitter)[:-1])
    print("[INFO] vfl_model_dir:{}".format(vfl_model_dir))

    dataset_name = vfl_exp_data["dataset"]
    attack_data_type = "iid"
    num_samples_ft = exp_args.num_samples_ft

    # save_exp_result_dir = os.path.join("..", "exp_result", "model_complete",
    #                                    "{}_{}_{}".format(dataset_name, attack_data_type, num_samples),
    #                                    "{}".format(vfl_model_dir.split(splitter)[-1]))
    if exp_args.attack_mode == "model_complete":
        save_exp_result_dir = ExperimentResultStructure.create_model_complete_task_subdir_path(
            os.path.join("{}_{}_{}".format(dataset_name, attack_data_type, num_samples_ft),
                         "{}".format(vfl_model_dir.split(splitter)[-1])))
    elif exp_args.attack_mode == "model_inversion":
        save_exp_result_dir = ExperimentResultStructure.create_model_inversion_task_subdir_path(
            os.path.join("{}_{}_{}".format(dataset_name, attack_data_type, num_samples_ft),
                         "{}".format(vfl_model_dir.split(splitter)[-1])))
    else:
        raise ValueError(f"unsupport attack model{exp_args.attack_mode}")
    os.makedirs(save_exp_result_dir, exist_ok=True)
    print("[INFO] save mc/mi experimental result dir:", save_exp_result_dir)

    main_task(exp_args, vfl_exp_data, vfl_model_dir, save_exp_result_dir, attack_data_type, device)
