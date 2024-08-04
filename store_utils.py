from __future__ import print_function

import json
import os
import shutil
import sys
from datetime import datetime
from glob import glob

import torch
import torch.nn.parallel

from arch_utils import MODEL_TYPE_LIST


def save_checkpoint(state, is_best, dir, filename):
    filepath = os.path.join(dir, f'{filename}_checkpoint.pth')
    torch.save(state, filepath)
    # print("[INFO] save model checkpoint result to {}.".format(filepath))
    if is_best:
        file_path = os.path.join(dir, f'{filename}_best.pth')
        shutil.copyfile(filepath, file_path)
        # print("[INFO] save best model checkpoint result to {}.".format(filepath))


def save_model_checkpoints(model_dict, is_best, dir):
    model_state_dict = dict()
    for name, model in model_dict.items():
        if model is not None:
            filepath = os.path.join(dir, f'{name}_checkpoint.pth')
            torch.save(model.state_dict(), filepath)

    # print("[INFO] save model checkpoint result to {}.".format(filepath))
    # if is_best:
    #     file_path = os.path.join(dir, f'{filename}_best.pth')
    #     shutil.copyfile(filepath, file_path)
    return model_state_dict


def load_model_checkpoint(dir, model_dict):
    for model_type in MODEL_TYPE_LIST:
        file_path = os.path.join(dir, f'{model_type}_checkpoint.pth')
        if not os.path.exists(file_path):
            print("[INFO] {} does not exist.".format(file_path))
        else:
            model_dict[model_type].load_state_dict(torch.load(file_path))
            print("[INFO] Load [{}] checkpoint from:{}".format(model_type, file_path))


def save_exp_result(result_json, dir, filename):
    file_name = filename + '.json'
    file_full_name = os.path.join(dir, file_name)
    with open(file_full_name, 'w') as outfile:
        json.dump(result_json, outfile)
    # print("[INFO] save experimental result to {}.".format(file_full_name))


def record_model_state(model_dict):
    model_state_dict = dict()
    for name, model in model_dict.items():
        model_state_dict[name] = model.state_dict() if model is not None else None
    return model_state_dict


def get_experiment_name(dataset_name, num_classes, defense_name, apply_encoder, apply_passport, apply_nl, is_imbal,
                        arch_type, has_active_bottom):
    exp_name = f"{dataset_name}+{num_classes}+{arch_type}+ab{has_active_bottom}+def{defense_name}+enc{apply_encoder}+pp{apply_passport}+nl{apply_nl}+imbal{is_imbal}"
    exp_name = exp_name.replace("_", "")
    exp_name = exp_name.replace("+", "_")
    return exp_name
    # return dataset_name + "_" + str(num_classes) + "_def" + defense_name + "_enc" + str(apply_encoder)


def get_timestamp():
    return int(datetime.utcnow().timestamp())


def get_path_slash():
    return "\\" if sys.platform == "win32" else "/"


class ExperimentTaskCategoryDirName(object):
    MAIN_TASK_DIR_NAME = "main_task"
    MODEL_COMPLETE_DIR_NAME = "model_complete"
    MODEL_INVERSION_DIR_NAME = "model_inversion"
    RESIDUE_RECONSTRUCTION_DIR_NAME = "residue_recons"
    GRADIENT_INVERSION_DIR_NAME = "gradient_inversion"


def get_exp_root_dir_name(loc=".."):
    return os.path.join(loc, "exp_result")


class ExperimentResultStructure(object):

    @staticmethod
    def create_main_task_subdir_path(dataset_name, loc=".."):
        return os.path.join(get_exp_root_dir_name(loc), ExperimentTaskCategoryDirName.MAIN_TASK_DIR_NAME, dataset_name)

    @staticmethod
    def create_model_complete_task_subdir_path(dataset_name, loc=".."):
        return os.path.join(get_exp_root_dir_name(loc), ExperimentTaskCategoryDirName.MODEL_COMPLETE_DIR_NAME,
                            dataset_name)

    @staticmethod
    def create_model_inversion_task_subdir_path(dataset_name, loc=".."):
        return os.path.join(get_exp_root_dir_name(loc), ExperimentTaskCategoryDirName.MODEL_INVERSION_DIR_NAME,
                            dataset_name)

    @staticmethod
    def create_residue_recons_task_subdir_path(dataset_name, loc=".."):
        return os.path.join(get_exp_root_dir_name(loc), ExperimentTaskCategoryDirName.RESIDUE_RECONSTRUCTION_DIR_NAME,
                            dataset_name)

    @staticmethod
    def create_gradient_inversion_task_subdir_path(dataset_name, loc=".."):
        return os.path.join(get_exp_root_dir_name(loc), ExperimentTaskCategoryDirName.GRADIENT_INVERSION_DIR_NAME,
                            dataset_name)

    @staticmethod
    def create_root_file_path(file_name, loc=".."):
        return os.path.join(get_exp_root_dir_name(loc), file_name)

    @staticmethod
    def search_exp_results_of_main_task_by_dataset(dataset_name="*", loc=".."):
        return glob(
            os.path.join(get_exp_root_dir_name(loc), ExperimentTaskCategoryDirName.MAIN_TASK_DIR_NAME, dataset_name,
                         "*", "*_last.json"))

    @staticmethod
    def search_subdirs_of_task_category_dir(task_category_name, subdir_names="*", loc=".."):
        return glob(os.path.join(get_exp_root_dir_name(loc), task_category_name, subdir_names))

    @staticmethod
    def search_subdirs_of_model_complete(subdir_names="*", loc=".."):
        return ExperimentResultStructure.search_subdirs_of_task_category_dir(
            ExperimentTaskCategoryDirName.MODEL_COMPLETE_DIR_NAME, subdir_names, loc)

    @staticmethod
    def search_exp_results_of_attacks_by_dir(full_dir_name):
        return glob(os.path.join(full_dir_name, "*", "*.json"))
