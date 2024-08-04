import logging
import math
import os
import sys
import time
from multiprocessing import Process, Queue

import torch

sys.path.append("./")
sys.path.append("../")

from store_utils import ExperimentResultStructure

logging.basicConfig()
logger = logging.getLogger('pipeline_mc')
fh = logging.FileHandler('pipeline_mc.log')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)
#####
# num_gpus = torch.cuda.device_count()
num_gpus = torch.cuda.device_count()
num_task_per_device = 1
logger.info(f"num_gpus{num_gpus}")


#####

class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get_current_time(self):
        return (time.time() - self.start_time) / 3600


class GPUManager(object):
    def __init__(self, num_gpus=4):
        self.gpu_queue = Queue()
        for _ in range(num_task_per_device):
            for device_id in range(num_gpus):
                self.gpu_queue.put(device_id)

    def require(self):
        try:
            return self.gpu_queue.get()
        except:
            return None

    def add_gpu(self, gpu_id):
        self.gpu_queue.put(gpu_id)


timer = Timer()
gpu_manager = GPUManager(num_gpus=num_gpus)


def chunks(L, m):
    n = int(math.ceil(len(L) / float(m)))
    return [L[i:i + n] for i in range(0, len(L), n)]


def run_gpu_model(cmd, log_file=None):
    if log_file:
        cmd = f"nohup {cmd} > {log_file}"
    while True:
        gpu_id = gpu_manager.require()
        if gpu_id is not None:
            try:
                run_cmd = f"export CUDA_VISIBLE_DEVICES={gpu_id} && {cmd}"
                logger.info(f"{run_cmd} start time: {timer.get_current_time()}")
                os.system(run_cmd)
                logger.info(f"{run_cmd} finished time: {timer.get_current_time()}")
            except:
                logger.warning(f'{cmd} failed')
            gpu_manager.add_gpu(gpu_id)
            break


data_dir = "../data"

num_samples = 60
# pos_ratio = 0.2

num_workers = 4
base_cmd = f"python splitvfl_train_shadowmodel.py "

# === using specific dataset_dir_name (e.g., nuswide2) or '*' to perform model completion attack.
dataset_dir_name = "bhi"
saved_exps = ExperimentResultStructure.search_exp_results_of_main_task_by_dataset(dataset_dir_name)
task_list = []
for exp_path in saved_exps:
    cmd = base_cmd + f"--exp_path {exp_path} --data_dir {data_dir} --num_workers {num_workers} --num_samples_ft {num_samples} "
    task_list.append(cmd)
for exp_path in saved_exps:
    if "defNONE" in exp_path:
        cmd = base_cmd + f"--exp_path {exp_path} --num_workers {num_workers} --train_all True " \
                         f"--load_pretrained_model False --num_samples_ft {num_samples} "
        task_list.append(cmd)

if sys.platform == "win32":
    for cmd in task_list:
        print("[INFO] execute: {}".format(cmd))
        os.system(cmd)
else:
    task_chunks = chunks(task_list, 1)
    for tasks in task_chunks:
        model_processes = []
        for cmd in tasks:
            p = Process(target=run_gpu_model, args=(cmd,))
            p.start()
            time.sleep(10)
            logger.info(f'create training task: {cmd}...')
            model_processes.append(p)
        for p in model_processes:
            p.join()

    logger.info(f'finished: {timer.get_current_time()} hours')
