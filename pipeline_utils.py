import math
import sys
import subprocess
import time
import os
import logging
from multiprocessing import Process, Queue
import torch


def get_dataset_cmd(dataset_name):
    return "--dataset_name {}".format(dataset_name)


def get_arch_cmd(arch, task_model, is_balanced, has_ab, has_intr_layer):
    return "--arch_config {} --task_model_type {} --is_imbal {} --has_ab {} --has_intr_layer {}".format(arch,
                                                                                                        task_model,
                                                                                                        is_balanced,
                                                                                                        has_ab,
                                                                                                        has_intr_layer)


def get_defense_cmd(defense_name, apply_encoder, apply_nl):
    return "--defense_name {} --apply_encoder {} --apply_nl {}".format(defense_name, apply_encoder, apply_nl)


def get_defense_arg_cmd(arg_name, arg_value):
    return "--{} {}".format(arg_name, arg_value)


def create_pipeline_tasks(*,
                          base_cmd,
                          dataset_list,
                          arch_type_list,
                          defense_dict,
                          apply_privacy_enhance_module,
                          imb_choice_on=True,
                          has_ab_choice_on=False,
                          has_intr_choice_on=False):
    has_intr_layer_choices = [True, False] if has_intr_choice_on else [False]
    task_list = []
    for dataset in dataset_list:
        if "nuswide2" in dataset:
            is_imbalanced_choices = [True, False] if imb_choice_on else [False]
        else:
            is_imbalanced_choices = [False]

        for is_balanced in is_imbalanced_choices:
            # === iterate all architecture choices === #
            for arch in arch_type_list:
                if arch == "VLR":
                    has_active_bottom_choices = [True]
                else:
                    # when arch is "VNN"
                    has_active_bottom_choices = [True, False] if has_ab_choice_on else [True]

                for has_active_bottom in has_active_bottom_choices:
                    for has_intr_layer in has_intr_layer_choices:

                        # TODO: need more elegant way to deal with task model choices
                        if has_intr_layer:
                            task_model = "MLP_1_LAYER"
                        else:
                            task_model = "MLP_2_LAYER"

                        # === iterate all defense choices === #
                        for apply_nl, apply_encoder in apply_privacy_enhance_module:
                            for defense_name, args_dict in defense_dict.items():
                                if args_dict:
                                    # === generate cmd for defense with defense arguments === #
                                    for arg_name, arg_values in args_dict.items():
                                        for arg_value in arg_values:
                                            dataset_cmd = get_dataset_cmd(dataset)
                                            arch_cmd = get_arch_cmd(arch,
                                                                    task_model,
                                                                    is_balanced,
                                                                    has_active_bottom,
                                                                    has_intr_layer)
                                            defense_cmd = get_defense_cmd(defense_name, apply_encoder, apply_nl)
                                            defense_arg_cmd = get_defense_arg_cmd(arg_name, arg_value)
                                            cmd = " ".join(
                                                [base_cmd, dataset_cmd, arch_cmd, defense_cmd, defense_arg_cmd])
                                            task_list.append(cmd)
                                else:
                                    # === generate cmd for defense without defense arguments === #
                                    dataset_cmd = get_dataset_cmd(dataset)
                                    arch_cmd = get_arch_cmd(arch,
                                                            task_model,
                                                            is_balanced,
                                                            has_active_bottom,
                                                            has_intr_layer)
                                    defense_cmd = get_defense_cmd(defense_name, apply_encoder, apply_nl)
                                    cmd = " ".join([base_cmd, dataset_cmd, arch_cmd, defense_cmd])
                                    task_list.append(cmd)
    return task_list


class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def get_current_time(self):
        return (time.time() - self.start_time) / 3600


class GPUManager(object):
    def __init__(self, num_gpus, num_task_per_device):
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


def chunks(L, m):
    n = int(math.ceil(len(L) / float(m)))
    return [L[i:i + n] for i in range(0, len(L), n)]


def run_tasks(task_list, num_gpus, num_task_per_device, logger, syn=True):
    timer = Timer()
    gpu_manager = GPUManager(num_gpus=num_gpus, num_task_per_device=num_task_per_device)

    def run_gpu_model(cmd, log_file=None):
        if log_file:
            cmd = f"nohup {cmd} > {log_file}"
        while True:
            gpu_id = gpu_manager.require()
            if gpu_id is not None:
                try:
                    run_cmd = f"export CUDA_VISIBLE_DEVICES={gpu_id} && {cmd}"
                    execute_task(run_cmd)
                except:
                    logger.warning(f'{cmd} failed')
                gpu_manager.add_gpu(gpu_id)
                break

    def execute_task(run_cmd):
        logger.info(f"{run_cmd} start time: {timer.get_current_time()}")
        os.system(run_cmd)
        logger.info(f"{run_cmd} finished time: {timer.get_current_time()}")

    if syn:
        # execute tasks synchronously.
        logger.info("execute tasks synchronously.")
        for cmd in task_list:
            execute_task(cmd)
    else:
        # performance tasks asynchronously.
        logger.info("execute tasks asynchronously.")
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
