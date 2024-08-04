# 
This is the official repo for the paper [A Framework for Evaluating Privacy-Utility Trade-off in Vertical Federated Learning](https://arxiv.org/abs/2209.03885). 

Note that:
This code base is designed to investigate the efficacy of current VFL defense mechanisms against VFL attacks. Additionally, it aims to provide a resilient framework that can adapt to future attacks and defenses. While this code base is not yet comprehensive and may not cover all possible VFL scenarios, it is freely available for adoption and modification. It can serve as a starting point for further development and refinement.

## 1. Methodology



## 2. Dataset

We use the following datasets for experiments. 

| Dataset | # of classes | VFL algo.  | |
|--------------|------------------------|------------------------|------------------------|
|Default Credit  | 2 | VLR | |
|Vehicle  | 2 | VLR | |
|NUSWIDE10     | 10  | VNN | can be downloaded at [here](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html) or [here](https://opendatalab.com/OpenDataLab/NUS-WIDE) |
|NUSWIDE2-imb  | 2 | VNN | the same as above|
|NUSWIDE2-bal  | 2 | VNN | the same as above|
|Criteo  | 2 | VNN | |
|BHI  | 2 | VNN | |
|FMNIST  | 10 | VNN | can be downloaded using Pytorch.|
|CIFAR10  | 10 | VNN | can be downloaded using Pytorch. |

You can adopt any dataset to run the code. 

## 3. Module description

### Modules in `svfl/` directory:

The `data_utils.py` module provides functions for fetching and preprocessing data. Notably, the `get_dataset` function retrieves data based on the specified dataset name.

The `pipeline_main.py` creates batches of tasks, each of which executes a defense against a specific attack. To configure the pipeline, users must specify the data directory, selected datasets, model architectures, and defenses in this file. The available defenses include ISO, D_SGD, DP_LAPLACE, GC, PPDL, MAX_NORM, and MARVELL, while the possible attacks are NBS, DBS, DLI, and RR.

Each task in `pipeline_main.py` invokes `svfl_main_task.py`, which serves as the primary entry point. The `main_task` function in `svfl_main_task.py` wires up all necessary components to execute a task involving a defense against an attack in the Vertical Federated Learning (VFL) setting.

The `svfl_lr_train.py` and `svfl_nn_train.py` modules contain code for training Vertical Linear Regression (VLR) and Vertical Neural Networks (VNN), respectively.

## 4. Run the code

You can run the code by executing `run.sh`

## 5. Citation

If you think our work is helpful and used our code in your work, please cite our paper:
```
@article{kang2022vfl,
  title={A framework for evaluating privacy-utility trade-off in vertical federated learning},
  author={Kang, Yan and Luo, Jiahuan and He, Yuanqin and Zhang, Xiaojin and Fan, Lixin and Yang, Qiang},
  journal={arXiv preprint arXiv:2209.03885},
  year={2022}
}
```
