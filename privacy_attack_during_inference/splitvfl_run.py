import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from ae_models.vae_models import VAE
from passport_config import get_passport_config
from splitvfl import SplitVFL
from splitvfl_data import DatasetSplit, get_mnist
from splitvfl_utils import apply_passport, set_seed, get_timestamp

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def run(data_dict, hyperparameter_dict, passport_args_dict, device, encoder=None, split_data=True):
    data_loader_dict = data_dict["data_loader_dict"]
    train_ldr = data_loader_dict["train_loader"]
    test_ldr = data_loader_dict["val_loader"]

    exp = SplitVFL(hyperparameter_dict, encoder=encoder, device=device)
    exp.init_models(passport_args_dict)
    exp.train(train_ldr, test_ldr, split_data=split_data)


def get_data_dict(dataset_name, **args):
    batch_size = args["batch_size"]
    # train_dst, test_dst, input_dims, num_classes = get_dataset(dataset_name, **args)

    num_classes = 10
    train_set, test_set, image_half_dim = get_mnist()
    # train_set, test_set, image_half_dim = get_cifar()

    print("[INFO] train_set length:{}".format(len(train_set)))
    print("[INFO] test_set length:{}".format(len(test_set)))

    train_indices = np.random.choice(a=60000, size=30000, replace=False)
    # train_indices = np.random.choice(a=50000, size=30000, replace=False)
    print("[DEBUG] train_indices: {}".format(train_indices[:50]))
    # train_indices = np.arange(0, 30000)
    train_set = DatasetSplit(train_set, train_indices)

    print("[INFO] after sample selection - train_set length:{}".format(len(train_set)))
    print("[INFO] after sample selection - test_set length:{}".format(len(test_set)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    data_loader_dict = {"train_loader": train_loader, "val_loader": val_loader, "test_loader": None}

    data_dict = dict()
    data_dict['dataset_name'] = dataset_name
    data_dict['num_classes'] = num_classes
    data_dict['image_half_dim'] = image_half_dim
    data_dict['data_loader_dict'] = data_loader_dict

    return data_dict


def determine_apply_passport(passport_args):
    has_active_bottom_model = True if passport_args["active_model_force_passport"] is not None else False
    if has_active_bottom_model:
        top_pp = apply_passport(passport_args["top_model_force_passport"])
        active_pp = apply_passport(passport_args["active_model_force_passport"])
        passive_pp = apply_passport(passport_args["passive_model_force_passport"])
        apply_pp = top_pp or active_pp or passive_pp
    else:
        top_pp = apply_passport(passport_args["top_model_force_passport"])
        passive_pp = apply_passport(passport_args["passive_model_force_passport"])
        apply_pp = top_pp or passive_pp

    return apply_pp


if __name__ == '__main__':

    # ====== config machine ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    gpu = 6
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        logger.info(f"Using GPU : {gpu}")
    else:
        logger.info("GPU is not available.")

    # ====== config dataset ======
    dataset_name = "MNIST".lower()
    # dataset_name = "CIFAR".lower()
    batch_size = 64
    num_classes = 10
    args = {"num_classes": num_classes, "batch_size": batch_size}

    # ====== config VFL setting ======
    vfl_type = "VHNN".upper()
    split_data = True

    passport_args = get_passport_config()

    vfl_apply_pp = determine_apply_passport(passport_args)

    apply_encoder = True
    apply_negative_loss = True

    # =======================
    ts = get_timestamp()
    vfl_setting = "{}_{}_{}_defNONE_enc{}_pp{}".format(vfl_type, dataset_name, num_classes, apply_encoder, vfl_apply_pp)
    vfl_model_dir = "./split_vfl/exp_result/{}_{}/".format(vfl_setting, ts)
    os.makedirs(vfl_model_dir, exist_ok=True)

    # ========================
    hyperparameter_dict = dict()
    hyperparameter_dict["num_classes"] = num_classes
    hyperparameter_dict["learning_rate"] = 0.001
    hyperparameter_dict["neg_lr"] = 0.0001
    hyperparameter_dict["neg_wd"] = 1e-5
    hyperparameter_dict["weight_decay"] = 1e-6
    hyperparameter_dict["optimizer_name"] = "adam"
    hyperparameter_dict["max_epochs"] = 25
    hyperparameter_dict["model_name"] = "lenet"
    hyperparameter_dict["aggregate_mode"] = "add"
    hyperparameter_dict["apply_negative_loss"] = apply_negative_loss
    hyperparameter_dict["apply_encoder"] = apply_encoder
    hyperparameter_dict["lambda_cf"] = 10  # 1, 5, 10, 20
    hyperparameter_dict["exp_dir"] = vfl_model_dir

    if apply_encoder:
        print("[INFO] Apply encoder for defense")
        # entropy_lbda = 0.0
        entropy_lbda = 1.0
        if num_classes == 2:
            dim = 2
            model_timestamp = '1646890978'
        elif num_classes == 10:
            dim = 10
            # model_timestamp = '1647156930'
            model_timestamp = '1648763417'
        else:
            raise Exception("[INFO] Does not support {} for now.".format(num_classes))

        encoder = VAE(z_dim=num_classes, input_dim=num_classes, hidden_dim=(num_classes * 6) ** 2).to(device)
        model_name = f"vae_{dim}_{str(entropy_lbda)}_{model_timestamp}"
        encoder.load_model(f"../ae_models/vae_trained_models/{model_name}")
    else:
        encoder = None

    # ====== start run ======
    # seed_list = [0, 1, 2]
    seed_list = [2]
    for seed in seed_list:
        set_seed(seed)

        data_dict = get_data_dict(dataset_name, **args)
        hyperparameter_dict["seed"] = seed
        hyperparameter_dict["image_half_dim"] = data_dict['image_half_dim']
        run(data_dict, hyperparameter_dict, passport_args, device=device, encoder=encoder, split_data=split_data)
