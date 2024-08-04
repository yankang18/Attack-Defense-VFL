import logging
import sys

sys.path.append("./")
sys.path.append("../")
import argparse
import os
import pprint

import torch.utils

from ae_models.vae_models import VAE
from arch_utils import get_architecture_config, build_models, prepare_optimizers, \
    determine_sub_vfl_type, ModelType
from privacy_attack_during_training.DBS_label_recovery import direction_based_scoring_attack
from privacy_attack_during_training.DLI_label_recovery import direct_label_inference_attack
from privacy_attack_during_training.NBS_label_recovery import norm_based_scoring_attack
from privacy_attack_during_training.RR_label_recovery import residue_reconstruction_attack
from privacy_attack_during_training.label_recovery_base import LabelRecoveryMethodName
from privacy_defense.defense_methods import DefenseName, DEFENSE_FUNCTIONS_DICT, init_defense_args_dict

from store_utils import get_timestamp, get_experiment_name, ExperimentResultStructure
from svfl.data_utils import get_print_interval, get_data_dict
from svfl.svfl_learner import VFLLearner
from svfl.svfl_lr_train import VFLLRTrain
from svfl.svfl_nn_train import VFLNNTrain
from utils import set_random_seed, str2bool, cross_entropy_for_one_hot, printf


def main_task(exp_args, data_dir, VFLNNTrainClass=VFLNNTrain):
    # ================== machine ===================
    machine_dict = dict()
    machine_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    machine_dict['cuda_id'] = exp_args.device_id
    machine_dict['random_seed'] = exp_args.seed
    set_random_seed(exp_args.seed)
    printf(f"[INFO] Using seed {exp_args}", verbose=exp_args.verbose)

    saved_model_dir = ExperimentResultStructure.create_main_task_subdir_path(exp_args.dataset_name)
    printf(f"[INFO] saved main task experimental result dir:{saved_model_dir}", verbose=exp_args.verbose)

    optimizer_name = exp_args.optim_name
    learning_rate = exp_args.lr
    weight_decay = exp_args.wd
    epochs = exp_args.num_epochs
    batch_size = exp_args.batch_size
    # batch_size = 4096  # large batch size

    arch_config_name = exp_args.arch_config
    has_active_bottom = exp_args.has_ab
    has_interactive_layer = exp_args.has_intr_layer
    task_model_type = exp_args.task_model_type

    apply_encoder = exp_args.apply_encoder
    apply_negative_loss = exp_args.apply_nl
    apply_passport = False
    apply_defense_name = exp_args.defense_name

    # ================== prepare data ==================
    dataset_name = exp_args.dataset_name
    is_imbal = exp_args.is_imbal  # this parameter only works for NUSWIDE dataset.   # ----------------> check
    args = {"imbal": is_imbal, "batch_size": batch_size, "data_dir": data_dir, "num_workers": exp_args.num_workers}
    data_dict = get_data_dict(dataset_name, **args)
    input_dims = data_dict['input_dims']
    num_classes = data_dict['num_classes']

    # ================ defense methods ======================
    # arguments for different privacy defense methods;
    # each argument's values are listed in a range for experiments.
    # refer to paper: (1) Label Inference Attacks Against Vertical Federated Learning
    #                 (2) Label Leakage and Protection in Two-party Split Learning
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
        printf(apply_defense_name, verbose=exp_args.verbose)
        raise ValueError("invalid defense name:[{}]".format(apply_defense_name))
    defense_args_dict['lambda_nl'] = exp_args.lambda_nl
    defense_args_dict['apply_protection_name'] = apply_defense_name
    defense_args_dict['apply_encoder'] = apply_encoder  # for apply CoAE
    defense_args_dict["apply_negative_loss"] = apply_negative_loss
    defense_dict = dict()
    defense_dict['args'] = defense_args_dict
    defense_dict['apply_protection'] = DEFENSE_FUNCTIONS_DICT[apply_defense_name]
    defense_dict['apply_protection_to_passive_party'] = True
    defense_dict["apply_negative_loss"] = apply_negative_loss
    if apply_encoder:
        printf("[INFO] Apply encoder for defense", verbose=exp_args.verbose)
        entropy_lbda = 0.0
        if num_classes == 2:
            dim = 2
            model_timestamp = '1646890978'  # n_classes 2
        elif num_classes == 10:
            dim = 10
            model_timestamp = '1647156930'  # n_classes 10
        else:
            raise Exception("[INFO] Does not support {} for now.".format(num_classes))

        encoder = VAE(z_dim=num_classes, input_dim=num_classes, hidden_dim=(num_classes * 6) ** 2).to(
            machine_dict['device'])
        model_name = f"vae_{dim}_{str(entropy_lbda)}_{model_timestamp}"
        encoder.load_model(f"../ae_models/vae_trained_models/{model_name}")
    else:
        encoder = None
    defense_dict['encoder'] = encoder
    pp = pprint.PrettyPrinter(indent=4)
    printf("[INFO] defense_dict: \n ", verbose=exp_args.verbose)
    if exp_args.verbose:
        pp.pprint(defense_dict)

    # ============ train hyper-parameters ==========
    print_interval = get_print_interval(dataset_name)
    hyperparameter_dict = dict()
    hyperparameter_dict["epochs"] = epochs
    hyperparameter_dict["print_interval"] = print_interval
    hyperparameter_dict["verbose"] = exp_args.verbose

    # ==================== prepare models for parties of the VFL ======================
    models_args = {'dnnfm': {'col_names': data_dict['col_names']}}  # only used for 'dnnfm' model
    arch_config = get_architecture_config(arch_config_name=arch_config_name,
                                          input_dims=input_dims,
                                          num_classes=num_classes,
                                          has_active_bottom=has_active_bottom,
                                          has_interactive_layer=has_interactive_layer,
                                          task_model_type=task_model_type,
                                          models_args=models_args)

    vfl_type = arch_config["vfl_type"]
    if exp_args.verbose:
        pp.pprint(arch_config)
    VFL_TRAIN = {"VLR": VFLLRTrain, "VNN": VFLNNTrainClass}
    vfl_arch_train_class = VFL_TRAIN[vfl_type]
    vfl_arch_train = vfl_arch_train_class(machine_dict=machine_dict, defense_dict=defense_dict)
    model_dict = build_models(arch_config, verbose=exp_args.verbose)

    sub_vfl_type = determine_sub_vfl_type(vfl_type, has_active_bottom)
    printf("[DEBUG] {} Architecture: \n".format(sub_vfl_type), verbose=exp_args.verbose)

    # ==================== prepare optimizers of the VFL architecture ======================
    optim_args_dict = {ModelType.NEGATIVE_LOSS_MODEL: {"learning_rate": 0.0001, "weight_decay": 1e-5}}
    optimizer_dict = prepare_optimizers(model_dict, learning_rate, weight_decay,
                                        optimizer_args_dict=optim_args_dict, optim_name=optimizer_name)
    criterion = cross_entropy_for_one_hot

    # ==================== register label recovery attacks ======================
    svfl = VFLLearner(machine_dict, hyperparameter_dict)
    svfl.register_label_recovery_attack(LabelRecoveryMethodName.NBS, norm_based_scoring_attack)
    svfl.register_label_recovery_attack(LabelRecoveryMethodName.DBS, direction_based_scoring_attack)
    if vfl_type == "VLR":
        svfl.register_label_recovery_attack(LabelRecoveryMethodName.DLI, direct_label_inference_attack)
        if exp_args.apply_rr_attack:
            svfl.register_label_recovery_attack(LabelRecoveryMethodName.RR, residue_reconstruction_attack)

    # ==================== start training ======================

    # prepare experimental results directory
    run_ts = get_timestamp()
    printf("[INFO] Start running experiment using [product] vfl framework with timestamp:[{}].".format(run_ts),
           verbose=exp_args.verbose)
    exp_name = get_experiment_name(dataset_name, num_classes, apply_defense_name,
                                   apply_encoder, apply_passport, apply_negative_loss,
                                   is_imbal, sub_vfl_type, has_active_bottom)
    exp_name_ts = exp_name + "_" + str(run_ts)
    full_dir_name = os.path.join(saved_model_dir, exp_name_ts)
    os.makedirs(full_dir_name, exist_ok=True)
    exp_dict = {"exp_name": exp_name, "full_dir_name": full_dir_name, "arch_config_name": arch_config_name,
                "dataset_name": dataset_name, "num_classes": num_classes, "vfl_type": sub_vfl_type,
                "task_model_type": task_model_type, "has_interactive_layer": has_interactive_layer,
                "has_active_bottom": has_active_bottom, "is_imbal": is_imbal, "optimizer_name": optimizer_name,
                "lr": learning_rate, "wd": weight_decay}

    svfl.run(vfl_arch_train=vfl_arch_train,
             model_dict=model_dict,
             optimizer_dict=optimizer_dict,
             data_dict=data_dict,
             criterion=criterion,
             exp_dict=exp_dict)

    # ==================== start testing ======================
    test_loader = data_dict["data_loader_dict"].get("test_loader")
    # if test_loader is not None:
    #     printf("[INFO] Start testing.", verbose=exp_args.verbose)
    #     validate(vfl_arch_train, model_dict, test_loader, criterion, dataset_name, num_classes, device="cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--verbose", type=str2bool, default=False, help='whether to print log, set True to print.')

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="nuswide2",
                        choices=["vehicle", "nuswide2", "nuswide10", "cifar10", "cifar2", "bhi", "mnist", "fmnist",
                                 "ctr_avazu", "criteo", "default_credit"])
    parser.add_argument("--is_imbal", type=str2bool, default=False, help='for imbal')

    parser.add_argument("--arch_config", type=str, default="VNN_MLP_V2",
                        choices=["VLR", "VNN_MLP", "VNN_MLP_V2", "VNN_RESNET", "VNN_LENET", "VNN_DNNFM",
                                 "VNN_DNNFM_V2"])
    parser.add_argument("--has_ab", type=str2bool, default=True, help='whether to use active bottom')
    parser.add_argument("--has_intr_layer", type=str2bool, default=False, help='whether to use interactive layer')
    parser.add_argument("--task_model_type", type=str, default="MLP_1_LAYER",
                        choices=["MLP_0_LAYER", "MLP_1_LAYER", "MLP_2_LAYER", "MLP_3_LAYER"])

    parser.add_argument("--apply_rr_attack", type=str2bool, default=False, help='whether to apply negative loss')

    parser.add_argument("--defense_name", type=str, default="NONE",
                        choices=["D_SGD", "DP_LAPLACE", "GC", "PPDL", "ISO", "MAX_NORM", "MARVELL", "NONE"])
    parser.add_argument("--apply_encoder", type=str2bool, default=False, help='whether to apply (V)AE encoder')
    parser.add_argument("--apply_nl", type=str2bool, default=False, help='whether to apply negative loss')
    parser.add_argument("--lambda_nl", type=float, default=20, help='lambda for negative loss')
    parser.add_argument("--noise_scale", type=float, default=0.0001, help='[0.0001, 0.001, 0.01, 0.1]')
    parser.add_argument("--gc_percent", type=float, default=0.1, help='[0.1, 0.25, 0.5, 0.75]')
    parser.add_argument("--ppdl_theta_u", type=float, default=0.75, help='[0.75, 0.50, 0.25, 0.10]')
    parser.add_argument("--grad_bins", type=float, default=24, help='[24, 18, 12, 6, 4, 2]')
    parser.add_argument("--bound_abs", type=float, default=1e-3, help='for apply_d_sgd')
    parser.add_argument("--ratio", type=float, default=3e-2, help='[1, 2.75, 25]')
    parser.add_argument("--init_scale", type=float, default=1.0, help='[0.1, 0.25, 1.0, 4.0]')

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
    parser.add_argument("--wd", type=float, default=0, help='weight decay')
    parser.add_argument("--optim_name", type=str, default="ADAM", choices=["ADAM", "SGD", "ADAGRAD"])

    exp_args = parser.parse_args()

    main_task(exp_args, data_dir=exp_args.data_dir)
