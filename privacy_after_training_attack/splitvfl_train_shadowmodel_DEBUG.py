import argparse
import os

import torch

from privacy_defense.defense_methods import DefenseName
from privacy_after_training_attack.splitvfl_train_shadowmodel import main_task
from store_utils import get_experiment_name
from store_utils import get_path_slash, ExperimentResultStructure
from utils import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples_mc", type=int, default=40, help="total number of samples for model completion")
    parser.add_argument("--lr", type=float, default=0.003, help='learning rate')
    parser.add_argument("--wd", type=float, default=1e-5, help='weight decay')
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--train_all", type=str2bool, default=False, help='whether to train all model')
    parser.add_argument("--load_pretrained_model", type=str2bool, default=True,
                        help='whether to load pretrained model')
    exp_args = parser.parse_args()

    exp_args.data_dir = "/Users/yankang/Documents/dataset/"

    # exp_args.lr = 0.01  # for nuswide and vehicle
    exp_args.lr = 0.005  # for mnist, fmnist
    exp_args.wd = 1e-6
    # exp_args.wd = 0
    exp_args.max_epoch = 300
    exp_args.batch_size = 64
    # exp_args.batch_size = 256  # for ResNet
    # exp_args.optim_name = "ADAGRAD".upper()
    # exp_args.optim_name = "ADAM".upper()
    exp_args.load_pretrained_model = True
    exp_args.train_all = False

    # ts = '1650933130' # lr=0.01 no nl, 89.21, 0.624
    # ts = '1650932517' # lr=0.01 has nl, 88.30, 0.5891

    # ts = '1650956397' # lr=0.002 no nl, 88.10, 0.5792
    ts = '1650956784'  # lr=0.002 has nl, 85.32, 0.6002

    arch_config_name = 'VNN_LENET'
    sub_vfl_type = 'VHNN'
    has_active_bottom = True

    dataset_name = 'fmnist'
    num_classes = 10
    is_imbal = False

    apply_passport = False
    apply_encoder = False
    apply_negative_loss = True
    apply_defense_name = DefenseName.ISO

    exp_name = get_experiment_name(dataset_name, num_classes, apply_defense_name,
                                   apply_encoder, apply_passport, apply_negative_loss,
                                   is_imbal, sub_vfl_type, has_active_bottom)

    vfl_exp_data = dict()
    vfl_exp_data["vfl_type"] = sub_vfl_type
    vfl_exp_data["imbal"] = is_imbal
    vfl_exp_data["arch_config_name"] = 'VNN_LENET'
    vfl_exp_data["dataset"] = dataset_name
    vfl_exp_data["num_classes"] = num_classes

    vfl_model_dir = "../exp_result/main_task/" + dataset_name + "/" + exp_name + "_" + ts
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attack_data_type = "iid"
    num_samples_ft = exp_args.num_samples_ft

    splitter = get_path_slash()
    save_exp_result_dir = ExperimentResultStructure.create_model_completion_task_subdir_path(
        os.path.join("{}_{}_{}".format(dataset_name, attack_data_type, num_samples_ft),
                     "{}".format(vfl_model_dir.split(splitter)[-1])))
    os.makedirs(save_exp_result_dir, exist_ok=True)
    print("[INFO] save mc/mi experimental result dir:", save_exp_result_dir)

    main_task(exp_args, vfl_exp_data, vfl_model_dir, save_exp_result_dir, attack_data_type, device)
