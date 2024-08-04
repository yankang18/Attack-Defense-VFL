import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def apply_passport(pp_pos):
    """
    Check whether passport protection is applied according to 'pp_pos'

    :param pp_pos:
    :return: True if passport protection is applied. Otherwise, return False.
    """
    for key, val in pp_pos.items():
        if val:
            return True
    return False


def get_number_passports(passport_pos):
    # count = 0
    # for key, val in passport_pos.items():
    #     if val:
    #         count += 1
    # return count
    return "".join(["1" if val else "0" for _, val in passport_pos.items()])


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def load_model(model, model_name, model_dir):
    full_path = os.path.join(model_dir, f"{model_name}")
    checkpoint = torch.load(full_path)
    model.load_state_dict(checkpoint)
    print(f"loaded model:{full_path}")


def save_models(model_dict, model_dir, attack_mode):
    for model_name, model in model_dict.items():
        if model is not None:
            full_file_name = os.path.join(model_dir, "{}_{}.pkl".format(model_name, attack_mode))
            torch.save(model.state_dict(), full_file_name)
            print("[INFO] saved models:{}".format(full_file_name))


def save_exp_result(result_json, dir, filename):
    file_name = filename + '.json'
    file_full_name = os.path.join(dir, file_name)
    with open(file_full_name, 'w') as outfile:
        json.dump(result_json, outfile)
