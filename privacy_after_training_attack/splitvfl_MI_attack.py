import sys

sys.path.append("./")
sys.path.append("../")
import copy
import numpy as np
import torch
import torch.optim as optim
import argparse
from utils import str2bool
import json
from loss_utils import TVloss, l2loss, calculate_psnr, calculate_ssim, select_image
from splitvfl_data import get_mnist, DatasetSplit
from splitvfl_models.lenet import LeNetConvBottom
from splitvfl_utils import load_model, get_number_passports, set_seed


def prepare_shadow_model(model_dir, device):
    # shadow_passive_bottom_model = LeNet(1, passport_pos={'0': False, '1': False})
    shadow_passive_bottom_model = LeNetConvBottom(1, passport_pos={'0': False, '1': False, '2': False})
    model_name = "mi_passive_bottom_model"
    load_model(shadow_passive_bottom_model, model_name, model_dir)
    return shadow_passive_bottom_model.to(device)


def prepare_real_model(model_dir, pp_args, seed, device):
    real_passive_bottom_model = LeNetConvBottom(1, passport_pos=pp_args)
    num_pp = get_number_passports(pp_args)
    model_name = f"lenet_passive_pp" + num_pp + "_cat" + f"_seed{seed}"
    load_model(real_passive_bottom_model, model_name, model_dir)
    return real_passive_bottom_model.to(device)


def prepare_real_feature(real_passive_bottom_model, data_set, label, index, image_half_dim=None, split_data=True):
    image = select_image(data_set, label=label, index=index)
    image = image.expand(1, image.shape[0], image.shape[1], image.shape[2])

    # iteration = iter(data_loader)
    # image, _ = next(iteration)
    if split_data:
        if image_half_dim is not None:
            image = image[:, :, :image_half_dim, :]
            print("[DEBUG] image shape:{}".format(image.shape))

    real_feature = real_passive_bottom_model(image)
    return real_feature.detach(), image


def model_inversion_attack(x_gen, extractor, x_optimizer, real_feature, real_image, total_epoch, lambda_tv):
    feature_mse_list, tv_loss_list, image_mses = [], [], []

    best_x = None
    best_psnr = 0.
    ssim_score = 0.
    best_epoch = -1
    converge_flag = False
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

        if epoch % 1000 == 0:
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


def run(vfl_setting_dict, dataset_dict, hyperparameter_dict, exp_dict, device):
    vfl_type = vfl_setting_dict["vfl_type"]
    vfl_apply_pp = vfl_setting_dict["vfl_apply_pp"]
    vfl_train_seed = vfl_setting_dict["vfl_train_seed"]
    real_model_pp_args = vfl_setting_dict["real_model_pp_args"]
    mi_train_data = vfl_setting_dict["mi_train_data"]
    ts = vfl_setting_dict["timestamp"]

    dataset_name = dataset_dict["dataset_name"]
    num_classes = dataset_dict["num_classes"]
    train_set = dataset_dict["train_set"]
    image_half_dim = dataset_dict["image_half_dim"]
    channel = dataset_dict["channel"]
    split_data = dataset_dict["split_data"]

    learning_rate = hyperparameter_dict["learning_rate"]
    epoch = hyperparameter_dict["epoch"]
    lambda_tv = hyperparameter_dict["lambda_tv"]

    label = exp_dict["label"]
    index = exp_dict["index"]

    # ========== prepare models ==========
    vfl_setting = "{}_{}_{}_pp{}".format(vfl_type, dataset_name, num_classes, vfl_apply_pp)

    real_model_dir = "./split_vfl/exp_result/{}_{}/".format(vfl_setting, ts)
    real_passive_bottom_model = prepare_real_model(real_model_dir, real_model_pp_args, vfl_train_seed, device)

    shadow_model_dir = "./split_vfl/exp_result/{}_mi_{}_{}/".format(vfl_setting, mi_train_data, ts)
    shadow_passive_bottom_model = prepare_shadow_model(shadow_model_dir, device)

    # ========== prepare reference data ===========
    # real_feature, real_image = prepare_real_feature(shadow_passive_bottom_model, train_set, label, index, image_half_dim)
    real_feature, real_image = prepare_real_feature(real_passive_bottom_model, train_set, label, index, image_half_dim,
                                                    split_data)
    print("[DEBUG] real_feature shape:", real_feature.shape)
    print("[DEBUG] real_image shape:", real_image.shape)

    # ========== prepare dummy x to be optimized ===========

    if split_data:
        # x = torch.zeros((1, channel, image_half_dim, image_half_dim * 2)).to(device)
        x = torch.randn((1, channel, image_half_dim, image_half_dim * 2)).to(device)
        # x = torch.ones((1, channel, image_half_dim, image_half_dim * 2)).to(device) * 0.5
    else:
        # x = torch.zeros((1, channel, image_half_dim * 2, image_half_dim * 2)).to(device)
        x = torch.randn((1, channel, image_half_dim * 2, image_half_dim * 2)).to(device)
        # x = torch.ones((1, channel, image_half_dim*2, image_half_dim * 2)).to(device) * 0.5

    x.requires_grad = True

    eps = 1e-3
    AMSGrad = True
    optimizer = optim.Adam(params=[x], lr=learning_rate, eps=eps, amsgrad=AMSGrad)

    return model_inversion_attack(x_gen=x,
                                  extractor=shadow_passive_bottom_model,
                                  x_optimizer=optimizer,
                                  real_feature=real_feature,
                                  real_image=real_image,
                                  total_epoch=epoch,
                                  lambda_tv=lambda_tv)


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
    parser.add_argument("--attack_mode", type=str, default="model_complete")
    parser.add_argument("--train_all", type=str2bool, default=False, help='whether to train all model')
    parser.add_argument("--load_pretrained_model", type=str2bool, default=True,
                        help='whether to load pretrained model')
    exp_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vfl_exp_data = json.load(open(exp_args.exp_path))
    print("[INFO] vfl_exp_data:{}".format(vfl_exp_data))
    seed = 1
    set_seed(seed)
    print("[INFO] seed:", seed)

    # ====== no passport in VHNN =====
    vfl_type = "VHNN"
    mi_train_data = "iid"
    real_model_pp_args = {'0': False, '1': False}
    vfl_apply_pp = False
    split_data = True

    ts = "1647580131"
    vfl_train_seed = 0

    # ====== config VFL setting ======

    vfl_setting_dict = dict()
    vfl_setting_dict["vfl_type"] = vfl_type
    vfl_setting_dict["vfl_apply_pp"] = vfl_apply_pp
    vfl_setting_dict["vfl_train_seed"] = vfl_train_seed
    vfl_setting_dict["real_model_pp_args"] = real_model_pp_args
    vfl_setting_dict["mi_train_data"] = mi_train_data
    vfl_setting_dict["timestamp"] = ts

    # ====== config dataset ======

    train_set, test_set, image_half_dim = get_mnist()
    train_indices = np.random.choice(a=60000, size=30000, replace=False)
    train_set = DatasetSplit(train_set, train_indices)

    dataset_name = "mnist"
    num_classes = 10

    dataset_dict = dict()
    dataset_dict["dataset_name"] = dataset_name
    dataset_dict["num_classes"] = num_classes
    dataset_dict["train_set"] = train_set
    dataset_dict["image_half_dim"] = image_half_dim
    dataset_dict["channel"] = 1
    dataset_dict["split_data"] = split_data

    # ====== config hyperparameter for training ======

    # learning_rate = 0.003
    # lambda_TV = 0.0001
    # lambda_TV = 50.0
    # learning_rate = 0.004
    hyperparameter_dict = dict()
    hyperparameter_dict["epoch"] = 40000
    hyperparameter_dict["learning_rate"] = 1e-1
    hyperparameter_dict["lambda_tv"] = 2e0

    # label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_list = [4]
    index_list = [1]
    psnr_list, ssim_list = [], []
    for label in label_list:
        for index in index_list:
            exp_dict = {"label": label, "index": index}
            print("[INFO] Experiment for:{}".format(exp_dict))
            print("[INFO] hyperparameter_dict for:{}".format(hyperparameter_dict))
            result_dict = run(vfl_setting_dict, dataset_dict, hyperparameter_dict, exp_dict, device)
            psnr_list.append(result_dict["PSNR"])
            ssim_list.append(result_dict["SSIM"])

    print("[INFO] vfl setting:\n {}".format(vfl_setting_dict))
    print("[INFO] MI PSNR mean:{}; std:{}, all:{}".format(np.mean(psnr_list), np.std(psnr_list), psnr_list))
    print("[INFO] MI SSIM mean:{}; std:{}, all:{}".format(np.mean(ssim_list), np.std(ssim_list), ssim_list))
