#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ---
# @File: pipeline_dlg.py
# @Author: Jiahuan Luo
# @Institution: Webank, Shenzhen, China
# @E-mail: jiahuanluo@webank.com
# @Time: 2022/3/23 14:13
# ---
import logging

import torch

from pipeline_utils import create_pipeline_tasks, run_tasks

logging.basicConfig()
logger = logging.getLogger('pipeline_dlg')
fh = logging.FileHandler('pipeline_dlg.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)

num_gpus = torch.cuda.device_count()

data_dir = "E:/dataset/"
dataset_list = ["default_credit"]
# dataset_list = ["vehicle"]
# dataset_list = ["vehicle", "default_credit"]

arch_type = ["VLR"]

defense_dict = {
    "ISO": {"ratio": [1, 2.75, 5, 10, 25]},
    "D_SGD": {"grad_bins": [24, 18, 12, 6, 4]},
    "DP_LAPLACE": {"noise_scale": [0.0001, 0.001, 0.01, 0.1]},
    "GC": {"gc_percent": [0.10, 0.25, 0.50, 0.75]},
    # "PPDL": {"ppdl_theta_u": [0.75, 0.50, 0.25, 0.10]},
    "MAX_NORM": {},
    # "MARVELL": {"init_scale": [0.1, 0.25, 1.0, 2.0, 4.0]},
    "NONE": {}
}

# === apply_nl_choices, apply_encoder_choices
apply_privacy_enhance_module = [(False, False)]


verbose = True
seed = 0
num_workers = 0
num_task_per_device = 2
dlg_lr = 0.01
wd = 0
dlg_iter = 10000
batch_size = 256

base_cmd = f"python DLG_label_recovery_task.py --seed {seed} --data_dir {data_dir} --dlg_lr {dlg_lr} --dlg_wd {wd} " \
           f"--dlg_iter {dlg_iter} --batch_size {batch_size} --num_workers {num_workers} --verbose {verbose}"

task_list = create_pipeline_tasks(base_cmd=base_cmd,
                                  dataset_list=dataset_list,
                                  arch_type_list=arch_type,
                                  defense_dict=defense_dict,
                                  apply_privacy_enhance_module=apply_privacy_enhance_module)

run_tasks(task_list, num_gpus, num_task_per_device, logger)
