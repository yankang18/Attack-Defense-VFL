import argparse

from svfl.svfl_main_task import main_task
from svfl.svfl_nn_train import VFLNNTrain
from utils import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--verbose", type=str2bool, default=True, help='whether to print log, set True to print.')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
    parser.add_argument("--wd", type=float, default=1e-6, help='weight decay')
    parser.add_argument("--optim_name", type=str, default="ADAM", choices=["ADAM", "SGD", "ADAGRAD"])
    parser.add_argument("--dataset_name", type=str, default="nuswide",
                        choices=["vehicle", "nuswide", "cifar10", "cifar2", "bhi", "mnist", "fmnist"])
    parser.add_argument("--arch_config", type=str, default="VNN_RESNET",
                        choices=["VLR", "VNN_MLP", "VNN_MLP_V2", "VNN_RESNET", "VNN_LENET"])
    parser.add_argument("--defense_name", type=str, default="MARVELL",
                        choices=["D_SGD", "DP_LAPLACE", "GC", "PPDL", "ISO", "MAX_NORM", "MARVELL", "NONE"])
    parser.add_argument("--is_imbal", type=str2bool, default=False, help='for imbal')
    parser.add_argument("--has_ab", type=str2bool, default=False, help='whether to use active bottom')
    parser.add_argument("--has_intr_layer", type=str2bool, default=True, help='whether to use interactive layer')
    parser.add_argument("--task_model_type", type=str, default="MLP_0_LAYER",
                        choices=["MLP_0_LAYER", "MLP_1_LAYER", "MLP_2_LAYER", "MLP_3_LAYER"])
    parser.add_argument("--apply_encoder", type=str2bool, default=False, help='whether to apply (V)AE encoder')
    parser.add_argument("--apply_nl", type=str2bool, default=False, help='whether to apply negative loss')
    parser.add_argument("--lambda_nl", type=float, default=False, help='lambda for negative loss')
    parser.add_argument("--noise_scale", type=float, default=0.0001, choices=[0.0001, 0.001, 0.01, 0.1],
                        help='for apply_dp_laplace')
    parser.add_argument("--gc_percent", type=float, default=0.1, choices=[0.1, 0.25, 0.5, 0.75],
                        help='for apply_grad_compression')
    parser.add_argument("--ppdl_theta_u", type=float, default=0.75, choices=[0.75, 0.50, 0.25, 0.10],
                        help='for apply_dp_gc_ppdl')
    parser.add_argument("--grad_bins", type=float, default=24, choices=[24, 18, 12, 6, 4, 2], help='for apply_d_sgd')
    parser.add_argument("--bound_abs", type=float, default=3e-2, help='for apply_d_sgd')
    parser.add_argument("--ratio", type=float, default=3e-2, choices=[1, 2.75, 25], help='for apply_iso')
    parser.add_argument("--init_scale", type=float, default=1.0, choices=[0.1, 0.25, 1.0, 4.0],
                        help='for marvell the initial value of P is scale * g')
    exp_args = parser.parse_args()

    exp_args.seed = 0
    # ===============================================
    exp_args.dataset_name = 'vehicle'
    # exp_args.dataset_name = 'default_credit'
    # exp_args.dataset_name = 'nuswide2'
    # exp_args.dataset_name = 'fmnist'
    # exp_args.dataset_name = 'ctr_avazu'
    # exp_args.dataset_name = 'criteo'
    # exp_args.dataset_name = 'bhi'
    exp_args.is_imbal = False

    # exp_args.lr = 0.05  # for default_credit
    exp_args.lr = 0.01  # for vehicle
    # exp_args.lr = 0.002  # for nuswide
    # exp_args.lr = 0.002  # for mnist
    # exp_args.lr = 0.05  # for criteo
    # exp_args.wd = 1e-5  # for criteo
    # exp_args.wd = 0
    # exp_args.num_epochs = 10  # for criteo
    # exp_args.num_epochs = 100  # for nuswide
    # exp_args.num_epochs = 5  # for vehicle
    exp_args.num_epochs = 50  # for vehicle
    # exp_args.num_epochs = 50  # for default_credit

    # exp_args.batch_size = 64  # for default_credit
    exp_args.batch_size = 256  # for vehicle
    # exp_args.batch_size = 2048  # for LBS of default_credit and vehicle
    # exp_args.batch_size = 256  # for nuswide
    # exp_args.batch_size = 256  # for resnet
    # exp_args.batch_size = 512  # for criteo
    # batch_size = 4096  # large batch size
    exp_args.optim_name = "ADAM".upper()
    # exp_args.optim_name = "SGD".upper()  # for criteo

    exp_args.arch_config = "VLR"
    # exp_args.arch_config = "VNN_MLP"
    # exp_args.arch_config = "VNN_MLP_V2"
    # exp_args.arch_config = "VNN_LENET"
    # exp_args.arch_config = "VNN_DNNFM_V2"
    # exp_args.arch_config = "VNN_DNNFM"
    exp_args.has_ab = True
    exp_args.has_intr_layer = False
    exp_args.task_model_type = "MLP_2_LAYER"
    # exp_args.task_model_type = "MLP_1_LAYER"
    # exp_args.task_model_type = "MLP_0_LAYER"

    exp_args.apply_rr_attack = True

    exp_args.apply_encoder = False
    exp_args.apply_nl = False
    exp_args.apply_passport = False

    # defense_list = ["MAX_NORM", "ISO", "DP_LAPLACE", "GC", "D_SGD"]
    # exp_args.defense_name = "NONE"
    # exp_args.defense_name = "MAX_NORM"
    # exp_args.defense_name = "ISO"
    # exp_args.defense_name = "DP_LAPLACE"
    # exp_args.defense_name = "GC"
    # exp_args.defense_name = "D_SGD"
    # exp_args.defense_name = "MARVELL"

    # ================ defense methods ======================

    # for apply_dp_laplace in the range of (0.0001, 0.001, 0.01, 0.1)
    exp_args.noise_scale = 0.01

    # for apply_grad_compression in the range of (0.10, 0.25, 0.50, 0.75)
    exp_args.gc_percent = 0.1

    # for apply_d_sgd in the range of (24, 18, 12, 6, 4)
    exp_args.grad_bins = 6
    exp_args.bound_abs = 3e-2
    # exp_args.bound_abs = 6e-2

    # for iso in the range of (1, 2.75, 5, 10, 25)
    exp_args.ratio = 5

    # for marvell in the range of (0.1, 0.25, 1, 4, 8, 12)
    exp_args.init_scale = 4

    # for negative loss in the range of (10, 20)
    exp_args.lambda_nl = 10

    # # for apply_dp_gc_ppdl in the range of (0.75，0.50，0.25，0.10)
    exp_args.ppdl_theta_u = 0.76

    # data_dir = "/Users/yankang/Documents/dataset/"
    data_dir = "E:/dataset/"

    defense_list = ["NONE", "MAX_NORM", "ISO", "DP_LAPLACE", "GC", "D_SGD"]
    for dn in defense_list:
        exp_args.defense_name = dn
        main_task(exp_args, data_dir, VFLNNTrainClass=VFLNNTrain)
