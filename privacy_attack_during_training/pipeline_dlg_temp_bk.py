#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---
# @File: pipeline_dlg.py
# @Author: Jiahuan Luo
# @Institution: Webank, Shenzhen, China
# @E-mail: jiahuanluo@webank.com
# @Time: 2022/3/23 14:13
# ---

import os

dataset_list = ["vehicle", "default_credit"]
defense_dict = {
    # "D_SGD": {"grad_bins": [24, 18, 12, 6, 4, 2]},
    # "DP_LAPLACE": {"noise_scale": [0.0001, 0.001, 0.01, 0.1]},
    # "GRAD_COMPRESSION": {"gc_percent": [0.10, 0.25, 0.50, 0.75]},
    # "PPDL": {"ppdl_theta_u": [0.75, 0.50, 0.25, 0.10]},
    "ISO": {"ratio": [1, 2.75, 25]},
    "MAX_NORM": {},
    "MARVELL": {"init_scale": [0.1, 0.25, 1.0, 4.0]},
    "NONE": {}
}
arch_type = ["VLR", "VNN_MLP", "VNN_RESNET"]
base_cmd = "python DLG_label_recovery_task.py --num_iterations 2000 "
for dataset in dataset_list:
    for arch in arch_type:
        if arch == "VLR":
            has_active_bottom_choices = [True]
        else:
            has_active_bottom_choices = [True, False]
        for has_active_bottom in has_active_bottom_choices:
            if dataset == "vehicle":
                is_balanced_choices = [False]
            else:
                is_balanced_choices = [True, False]
            for is_balanced in is_balanced_choices:
                for defense_name, ars_dict in defense_dict.items():
                    if ars_dict:
                        for arg_name, arg_values in ars_dict.items():
                            for arg_value in arg_values:
                                cmd = base_cmd + f"--dataset_name {dataset} --arch_type {arch} " \
                                                 f"--is_imbal {is_balanced} --has_ab {has_active_bottom} " \
                                                 f"--defense_name {defense_name} --{arg_name} {arg_value}"
                                print(cmd)
                                os.system(cmd)
                    else:
                        cmd = base_cmd + f"--dataset_name {dataset} --arch_type {arch} --is_imbal {is_balanced} " \
                                         f"--has_ab {has_active_bottom} --defense_name {defense_name}"
                        print(cmd)
                        os.system(cmd)
