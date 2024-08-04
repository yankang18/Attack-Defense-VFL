import sys

sys.path.append("./")
sys.path.append("../")
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from loss_utils import TVloss, l2loss, calculate_psnr, calculate_ssim, select_image

from arch_utils import ModelType, get_architecture_config, build_layer_model
from passport_config import get_passport_config
from splitvfl_data import select_samples_by_labels, select_samples_by_labels_from_2party_dataset
import numpy as np
from splitvfl_utils import get_number_passports, set_seed, load_model, save_models, save_exp_result
from store_utils import get_path_slash, ExperimentResultStructure
from svfl.data_utils import fetch_parties_data, get_dataset
from utils import str2bool
import copy
from torchvision.utils import save_image


def frozen_net(model_list, frozen):
    for model in model_list:
        for param in model.parameters():
            param.requires_grad = not frozen


def prepare_models(arch_config, vfl_model_dir, shadow_model_dir, passport_args):
    top_model_pp_args = passport_args["top_model_force_passport"]
    active_bottom_pp_args = passport_args["active_model_force_passport"]
    passive_bottom_pp_args = passport_args["passive_model_force_passport"]

    model_dict = dict()
    # For model inversion attack, we must have a task model and passive bottom model.
    # For VHNN scenario, we have a active bottom model, while for VSNN scenario, we have no active bottom model.
    # The task model and the active bottom model will be frozen when conduct the model inversion attack training.
    # Thus, only the passive bottom model will be training for model inversion.
    real_passive_bottom_model = build_layer_model(arch_config["passive_bottom_config"])
    model_name = f"passive_bottom_checkpoint.pth"
    load_model(real_passive_bottom_model, model_name, vfl_model_dir)
    shadow_passive_bottom_model = build_layer_model(arch_config["passive_bottom_config"])
    model_name = f"passive_bottom_model_inversion.pkl"
    load_model(shadow_passive_bottom_model, model_name, shadow_model_dir)

    return real_passive_bottom_model, shadow_passive_bottom_model


def get_data_dict(attack_mode, dataset_name, **args):
    if attack_mode == "model_inversion":
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


def model_inversion_attack(x_gen, extractor, x_optimizer, real_feature, real_image, total_epoch, lambda_tv):
    feature_mse_list, tv_loss_list, image_mses = [], [], []

    best_x = None
    best_psnr = 0.
    ssim_score = 0.
    best_epoch = -1
    converge_flag = False
    save_image(real_image, "./real_image.jpg")
    # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for epoch in range(total_epoch):
        extractor.eval()

        x_optimizer.zero_grad()

        pred_feature = extractor(x_gen)

        feature_mse = ((real_feature - pred_feature) ** 2).mean()
        # feature_mse = ((real_feature - pred_feature) ** 2).sum()
        # feature_mse = 1 - cos(real_feature, pred_feature).mean()
        tv_loss = TVloss(x_gen, pow=2)
        loss = feature_mse + lambda_tv * tv_loss
        # norm_loss = l2loss(x_gen)
        # loss = feature_mse + lambda_tv * tv_loss + lambda_l2 * norm_loss

        loss.backward()
        x_optimizer.step()

        with torch.no_grad():
            image_mse = ((real_image - x_gen) ** 2).mean()

        feature_mse_list.append(feature_mse.item())
        tv_loss_list.append(tv_loss.item())
        image_mses.append(image_mse.item())

        if epoch % 100 == 0 and epoch > 0:
            print("epoch:%4d, feature_mse:%2.6f, tv_loss:%2.6f, image_mse:%2.6f"
                  % (epoch, np.mean(feature_mse_list), np.mean(tv_loss_list), np.mean(image_mses)))
            feature_mse_list, tv_loss_list, image_mses = [], [], []

            x_gen_copy = copy.deepcopy(x_gen)
            psnr_score_2 = calculate_psnr(real_image, x_gen_copy, max_val=2)
            print(f"PSNR2:{psnr_score_2}.")

            if psnr_score_2 > best_psnr:
                best_psnr = psnr_score_2
                best_epoch = epoch
                best_x = x_gen_copy
                save_image(x_gen, f"./{epoch}.jpg")

                psnr_score_1 = calculate_psnr(real_image, best_x, max_val=1)
                ssim_score = calculate_ssim(real_image, best_x).cpu().detach().item()
                print(
                    f"[INFO] save best recovered image with SSIM:{ssim_score}, PSNR1:{psnr_score_1}, PSNR2:{psnr_score_2}.")

            elif epoch >= best_epoch + 8000:
                converge_flag = True

        if converge_flag:
            break

    psnr = calculate_psnr(real_image, best_x, max_val=2)
    # file_name = get_exp_result_file_name(arg_dict, psnr, ssim_score)
    # plot_pair_images([real_image[0], best_x[0]], show_fig=True)
    return {"recovered_image": best_x, "PSNR": psnr, "SSIM": ssim_score}


def main_task(exp_args, vfl_exp_data, load_vfl_model_dir, shadow_model_dir, save_exp_result_dir, attack_data_type,
              device):
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
    data_loader_dict = data_dict["data_loader_dict"]
    dataset_name = data_dict["dataset_name"]
    train_data_loader = data_loader_dict["train_data_loader"]
    val_data_loader = data_loader_dict["val_data_loader"]
    test_data_loader = data_loader_dict["test_data_loader"]
    arch_config = get_architecture_config(arch_config_name=arch_config_name,
                                          input_dims=input_dims,
                                          num_classes=num_classes,
                                          has_active_bottom=has_active_bottom,
                                          has_interactive_layer=has_interactive_layer,
                                          task_model_type=task_model_type,
                                          models_args=models_args)

    real_passive_bottom_model, shadow_passive_bottom_model = prepare_models(arch_config, load_vfl_model_dir,
                                                                            shadow_model_dir, passport_args)
    real_passive_bottom_model = real_passive_bottom_model.to(device)
    shadow_passive_bottom_model = shadow_passive_bottom_model.to(device)
    for x, y in train_data_loader:
        y = y.to(device)
        x_passive, x_active = fetch_parties_data(dataset_name=dataset_name, data=x, device=device)
        real_feature = real_passive_bottom_model(x_passive).detach()
        dummy_x = torch.randn(x_passive.size()).to(device)
        dummy_x.requires_grad = True
        learning_rate = 1e-2
        eps = 1e-4
        AMSGrad = True
        epoch = 400000
        lambda_tv = 2e0
        optimizer = optim.Adam(params=[dummy_x], lr=learning_rate, eps=eps, amsgrad=AMSGrad)
        return model_inversion_attack(x_gen=dummy_x,
                                      extractor=shadow_passive_bottom_model,
                                      x_optimizer=optimizer,
                                      real_feature=real_feature,
                                      real_image=x_passive,
                                      total_epoch=epoch,
                                      lambda_tv=lambda_tv)
    # vfl_exp_data = dict(vfl_exp_data, **{"mc_result": best_metric})
    # vfl_exp_data = dict(vfl_exp_data, **{"mc_args": vars(exp_args)})
    # if train_all:
    #     save_exp_result(vfl_exp_data, save_exp_result_dir, "train_all_result")
    # else:
    #     save_exp_result(vfl_exp_data, save_exp_result_dir, "result")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples_ft", type=int, default=40, help="total number of samples for model completion")
    parser.add_argument("--lr", type=float, default=0.005, help='learning rate')
    parser.add_argument("--wd", type=float, default=1e-6, help='weight decay')
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--attack_mode", type=str, default="model_complete")
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
    shadow_model_dir = splitter.join(exp_args.exp_path.split(splitter)[:-1])
    print("[INFO] shadow_model_dir:{}".format(shadow_model_dir))
    vfl_model_path = exp_args.exp_path.split(splitter)[:-1]
    vfl_model_path[2] = "main_task"
    vfl_model_path[3] = vfl_model_path[3].split("_")[0]
    vfl_model_dir = splitter.join(vfl_model_path)
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
    # os.makedirs(save_exp_result_dir, exist_ok=True)
    print("[INFO] save mc/mi experimental result dir:", save_exp_result_dir)

    main_task(exp_args, vfl_exp_data, vfl_model_dir, shadow_model_dir, save_exp_result_dir, attack_data_type, device)
