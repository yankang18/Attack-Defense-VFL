# -*- coding: utf-8 -*-

import argparse
import os
import pprint

import torch

from DLG_label_recovery import DLGLabelRecovery
from ae_models.vae_models import VAE
from arch_utils import get_architecture_config
from privacy_defense.defense_methods import DefenseName, DEFENSE_FUNCTIONS_DICT, init_defense_args_dict
from store_utils import ExperimentResultStructure
from svfl.data_utils import get_dataset
from svfl.svfl_lr_train import VFLLRTrain
from svfl.svfl_nn_train import VFLNNTrain
from utils import set_random_seed, str2bool


def main_task(exp_args, data_dir, save_exp_result_dir):
    # ================== machine info ===================
    machine_dict = dict()
    machine_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    machine_dict['cuda_id'] = exp_args.device_id

    seed = exp_args.seed
    machine_dict['random_seed'] = seed
    set_random_seed(seed)
    print("[INFO] Using seed {}".format(seed))

    # ================== datasets ==================
    dataset_name = exp_args.dataset_name
    is_imbal = exp_args.is_imbal  # this parameter only works for 2-label NUSWIDE dataset.

    datasets_dict = dict()
    args = {"imbal": is_imbal, "data_dir": data_dir}
    train_dst, val_dst, test_dst, input_dims, num_classes, pos_ratio, col_names = get_dataset(dataset_name, **args)
    datasets_dict['dataset_name'] = dataset_name
    datasets_dict['train_dataset'] = train_dst
    datasets_dict['num_classes'] = num_classes
    datasets_dict['is_imbal'] = is_imbal

    # ================ defense ======================
    apply_negative_loss = exp_args.apply_nl
    apply_encoder = exp_args.apply_encoder
    apply_defense_name = exp_args.defense_name

    # arguments for different privacy defense methods
    defense_args_dict = init_defense_args_dict(apply_defense_name)
    # for apply_dp_laplace in the range of (0.0001，0.001，0.01，0.1)
    if apply_defense_name == DefenseName.DP_LAPLACE:
        defense_args_dict[DefenseName.DP_LAPLACE]['noise_scale'] = exp_args.noise_scale
    # for apply_grad_compression in the range of (0.10，0.25，0.50，0.75)
    elif apply_defense_name == DefenseName.GRAD_COMPRESSION:
        defense_args_dict[DefenseName.GRAD_COMPRESSION]['gc_percent'] = exp_args.gc_percent
    # for apply_d_sgd in the range of (24, 18, 12, 6, 4)
    elif apply_defense_name == DefenseName.D_SGD:
        defense_args_dict[DefenseName.D_SGD]['grad_bins'] = exp_args.grad_bins
        # for apply_d_sgd, default is 3e-2
        defense_args_dict[DefenseName.D_SGD]['bound_abs'] = exp_args.bound_abs
    # for iso in the range of (1, 2.75, 25)
    elif apply_defense_name == DefenseName.ISO:
        defense_args_dict[DefenseName.ISO]['ratio'] = exp_args.ratio
    # for marvell in the range of (0.1, 0.25, 1, 4)
    elif apply_defense_name == DefenseName.MARVELL:
        defense_args_dict[DefenseName.MARVELL]['init_scale'] = exp_args.init_scale
    # for negative loss in the range of (10, 20)
    # elif apply_defense_name == DefenseName.NEGATIVE_LOSS:
    # # for apply_dp_gc_ppdl in the range of (0.75，0.50，0.25，0.10)
    elif apply_defense_name == DefenseName.PPDL:
        defense_args_dict[DefenseName.PPDL]['ppdl_theta_u'] = exp_args.ppdl_theta_u
    elif apply_defense_name == DefenseName.MAX_NORM:
        pass
    elif apply_defense_name == DefenseName.NONE:
        pass
    else:
        print(apply_defense_name)
        raise ValueError("invalid defense_name")
    defense_args_dict['lambda_nl'] = exp_args.lambda_nl
    defense_args_dict['apply_protection_name'] = apply_defense_name
    defense_args_dict['apply_encoder'] = apply_encoder  # for apply CoAE
    defense_args_dict["apply_negative_loss"] = apply_negative_loss

    defense_dict = dict()
    defense_dict['args'] = defense_args_dict
    defense_dict['apply_protection'] = DEFENSE_FUNCTIONS_DICT[apply_defense_name]
    if apply_encoder:
        print("[INFO] Apply encoder for defense")
        entropy_lbda = 0.0
        if num_classes == 2:
            model_timestamp = '1646890978'  # n_classes 2
        elif num_classes == 10:
            model_timestamp = '1647156930'  # n_classes 10
        else:
            raise Exception("[INFO] Does not support {} for now.".format(num_classes))

        dim = datasets_dict['num_classes']
        encoder = VAE(z_dim=num_classes, input_dim=num_classes, hidden_dim=(num_classes * 6) ** 2).to(
            machine_dict['device'])
        model_name = f"vae_{dim}_{str(entropy_lbda)}_{model_timestamp}"
        encoder.load_model(f"../ae_models/vae_trained_models/{model_name}")
    else:
        print("[INFO] Does not apply encoder for defense")
        encoder = None
    defense_dict['encoder'] = encoder

    # ==================== prepare models of the VFL architecture ======================
    arch_config_name = exp_args.arch_config
    has_active_bottom = exp_args.has_ab
    has_interactive_layer = exp_args.has_intr_layer
    arch_config = get_architecture_config(arch_config_name=arch_config_name,
                                          input_dims=input_dims,
                                          num_classes=datasets_dict['num_classes'],
                                          has_active_bottom=has_active_bottom,
                                          has_interactive_layer=has_interactive_layer)
    VFL_TRAIN = {"VLR": VFLLRTrain, "VNN": VFLNNTrain}
    vfl_arch_train_class = VFL_TRAIN[arch_config["vfl_type"]]
    pp = pprint.PrettyPrinter(indent=4)
    if exp_args.verbose:
        pp.pprint(arch_config)

    # ================== hyper-parameter info ===================
    batch_size = exp_args.batch_size
    dlg_lr = exp_args.dlg_lr
    dlg_iter = exp_args.dlg_iter

    total_sample_examined = 2048 if batch_size <= 2048 else batch_size
    label_rec_hyperparam_dict = dict()
    label_rec_hyperparam_dict["arch_config_name"] = arch_config_name
    label_rec_hyperparam_dict["has_active_bottom"] = has_active_bottom
    label_rec_hyperparam_dict["exp_result_dir"] = save_exp_result_dir
    label_rec_hyperparam_dict["batch_size_list"] = [batch_size]
    label_rec_hyperparam_dict["num_experiments"] = int(total_sample_examined / batch_size)
    label_rec_hyperparam_dict['model_lr'] = 1e-3

    dlg_hyperparam_dict = dict()
    dlg_hyperparam_dict["dlg_iter"] = dlg_iter
    dlg_hyperparam_dict["dlg_lr"] = dlg_lr

    exp = DLGLabelRecovery(arch_config,
                           machine_dict,
                           datasets_dict,
                           defense_dict,
                           dlg_hyperparam_dict,
                           label_rec_hyperparam_dict)
    exp.run(vfl_arch_train_class=vfl_arch_train_class)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--verbose", type=str2bool, default=False, help='whether to print log, set True to print.')

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--dlg_lr", type=float, default=0.001, help='learning rate')
    parser.add_argument("--dlg_wd", type=float, default=0, help='weight decay')
    parser.add_argument("--dlg_iter", type=int, default=2000)

    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--optim_name", type=str, default="ADAM", choices=["ADAM", "SGD", "ADAGRAD"])
    parser.add_argument("--dataset_name", type=str, default="vehicle",
                        choices=["vehicle", "nuswide2", "nuswide10", "cifar10", "cifar2", "bhi", "mnist", "fmnist",
                                 "ctr_avazu", "criteo", "default_credit"])
    parser.add_argument("--arch_config", type=str, default="VLR",
                        choices=["VLR", "VNN_MLP", "VNN_MLP_V2", "VNN_RESNET", "VNN_LENET", "VNN_DNNFM",
                                 "VNN_DNNFM_V2"])
    parser.add_argument("--defense_name", type=str, default="NONE",
                        choices=["D_SGD", "DP_LAPLACE", "GC", "PPDL", "ISO", "MAX_NORM", "MARVELL", "NONE"])
    parser.add_argument("--is_imbal", type=str2bool, default=False, help='for imbal')
    parser.add_argument("--has_ab", type=str2bool, default=True, help='whether to use active bottom')
    parser.add_argument("--has_intr_layer", type=str2bool, default=False, help='whether to use interactive layer')
    parser.add_argument("--task_model_type", type=str, default="MLP_1_LAYER",
                        choices=["MLP_0_LAYER", "MLP_1_LAYER", "MLP_2_LAYER", "MLP_3_LAYER"])

    parser.add_argument("--apply_encoder", type=str2bool, default=False, help='whether to apply (V)AE encoder')
    parser.add_argument("--apply_nl", type=str2bool, default=False, help='whether to apply negative loss')
    parser.add_argument("--lambda_nl", type=float, default=10, help='lambda for negative loss')
    parser.add_argument("--noise_scale", type=float, default=0.0001, help='[0.0001, 0.001, 0.01, 0.1]')
    parser.add_argument("--gc_percent", type=float, default=0.1, help='[0.1, 0.25, 0.5, 0.75]')
    parser.add_argument("--ppdl_theta_u", type=float, default=0.75, help='[0.75, 0.50, 0.25, 0.10]')
    parser.add_argument("--grad_bins", type=float, default=24, help='[24, 18, 12, 6, 4, 2]')
    parser.add_argument("--bound_abs", type=float, default=2e-3, help='for apply_d_sgd')
    parser.add_argument("--ratio", type=float, default=3e-2, help='[1, 2.75, 25]')
    parser.add_argument("--init_scale", type=float, default=1.0, help='[0.1, 0.25, 1.0, 4.0]')

    exp_args = parser.parse_args()

    dataset_name = exp_args.dataset_name
    save_exp_result_dir = ExperimentResultStructure.create_gradient_inversion_task_subdir_path(dataset_name)
    os.makedirs(save_exp_result_dir, exist_ok=True)
    print("[INFO] save DLG experimental result dir:", save_exp_result_dir)

    main_task(exp_args, data_dir=exp_args.data_dir, save_exp_result_dir=save_exp_result_dir)
