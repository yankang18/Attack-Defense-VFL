import logging
import torch
import sys

sys.path.append("./")
sys.path.append("../")

from pipeline_utils import create_pipeline_tasks, run_tasks

logging.basicConfig()
logger = logging.getLogger('pipeline_main')
fh = logging.FileHandler('pipeline_main.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)

data_dir = "../data"

# set one or more datasets from "vehicle", "nuswide2", "nuswide10", "cifar10", "cifar2", "bhi", "mnist", "fmnist",
# "ctr_avazu", "criteo", "default_credit"

# dataset_list = ["fmnist"]
# dataset_list = ["nuswide2"]
dataset_list = ["default_credit"]

# set one model architecture from "VLR", "VNN_MLP", "VNN_MLP_V2", "VNN_RESNET", "VNN_LENET", "VNN_DNNFM",
#                                  "VNN_DNNFM_V2"

# arch_type = ["VNN_MLP_V2"]
# arch_type = ["VNN_LENET"]
arch_type = ["VLR"]

# defense_dict = {
#     "ISO": {"ratio": [1, 2.75, 5, 10, 25]},
#     "D_SGD": {"grad_bins": [24, 18, 12, 6, 4]},
#     "DP_LAPLACE": {"noise_scale": [0.0001, 0.001, 0.01, 0.1]},
#     "GC": {"gc_percent": [0.10, 0.25, 0.50, 0.75]},
#     "PPDL": {"ppdl_theta_u": [0.75, 0.50, 0.25, 0.10]},
#     "MAX_NORM": {},
#     "MARVELL": {"init_scale": [0.1, 0.25, 1.0, 2.0, 4.0]}
# }

defense_dict = {
    "NONE": {}
}

# the three combination of apply_nl_choices and apply_encoder_choices
# (False, False) : does not apply both negative loss and label encoder.
# (True, False) : applies negative loss but does not apply label encoder.
# (False, True) : does not apply negative loss but applies label encoder.
# apply_privacy_enhance_module = [(False, False), (True, False), (False, True)]
apply_privacy_enhance_module = [(False, False)]

verbose = True

seed = 0
num_workers = 2
num_task_per_device = 2
lr = 0.01
wd = 1e-5
num_epochs = 10
optim_name = "ADAM"

base_cmd = f"python svfl_main_task.py --seed {seed} --data_dir {data_dir} --lr {lr} --wd {wd} " \
           f"--num_epochs {num_epochs} --num_workers {num_workers} --verbose {verbose} --optim_name {optim_name} "

task_list = create_pipeline_tasks(base_cmd=base_cmd,
                                  dataset_list=dataset_list,
                                  arch_type_list=arch_type,
                                  defense_dict=defense_dict,
                                  apply_privacy_enhance_module=apply_privacy_enhance_module)

print("tasks:")
for task in task_list:
    print(task)

num_gpus = torch.cuda.device_count()
run_tasks(task_list, num_gpus, num_task_per_device, logger)
