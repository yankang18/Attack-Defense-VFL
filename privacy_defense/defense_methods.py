import math
from collections import OrderedDict

import torch
from torch import nn

from privacy_defense.defense_tools import DPLaplacianNoiseApplyer, TensorPruner, dp_gc_ppdl, multistep_gradient
from privacy_defense.solver import solve_isotropic_covariance


# from privacy_defense.solver import solve_isotropic_covariance
class DefenseName(object):
    D_SGD = "D_SGD"
    DP_LAPLACE = "DP_LAPLACE"
    GRAD_COMPRESSION = "GC"
    PPDL = "PPDL"
    ISO = "ISO"
    MAX_NORM = "MAX_NORM"
    MARVELL = "MARVELL"
    ADVERSARIAL_LOSS = "ADVERS_LOSS"
    NONE = "NONE"


def raise_error(param, param_name, defense_name):
    if param is None:
        raise Exception("{} cannot be None for {}".format(param_name, defense_name))


def init_defense_args_dict(apply_defense_name):
    defense_args_dict = dict()

    # for initializing defense args dictionary
    defense_args_dict[apply_defense_name] = OrderedDict()

    # for apply CoAE
    defense_args_dict['apply_encoder'] = False  # for apply CoAE

    # for recording defense method name
    defense_args_dict['apply_protection_name'] = apply_defense_name

    return defense_args_dict


def apply_identity(gradients, **args):
    return gradients


def apply_d_sgd(gradients, **args):
    grad_bins_num = args.get("grad_bins")
    raise_error(grad_bins_num, "grad_bins", "apply_d_sgd")
    bound_abs = args.get("bound_abs")
    # bound_abs = 3e-2 if bound_abs is None else bound_abs
    d_grad = multistep_gradient(gradients, bins_num=grad_bins_num, bound_abs=bound_abs)
    return d_grad


def apply_dp_laplace(gradients, **args):
    noise_scale = args.get("noise_scale")
    raise_error(noise_scale, "noise_scale", "apply_dp_laplace")
    dp = DPLaplacianNoiseApplyer(beta=noise_scale)
    noisy_grad = dp.laplace_mech(gradients)
    return noisy_grad


def apply_grad_compression(gradients, **args):
    gc_percent = args["gc_percent"]
    raise_error(gc_percent, "gc_percent", "apply_grad_compression")
    tensor_pruner = TensorPruner(zip_percent=gc_percent)
    tensor_pruner.update_thresh_hold(gradients)
    pruned_grad = tensor_pruner.prune_tensor(gradients)
    return pruned_grad


def apply_dp_gc_ppdl(gradients, **args):
    ppdl_theta_u = args["ppdl_theta_u"]
    raise_error(ppdl_theta_u, "ppdl_theta_u", "apply_dp_gc_ppdl")
    dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[gradients], theta_u=ppdl_theta_u, gamma=0.001, tau=0.0001)
    return gradients


# def apply_dp_gaussian(gradients, **args):
#     location = 0.0
#     threshold = 0.2  # 1e9
#     dp_strength = args["dp_strength"]
#     device = args["device"]
#     with torch.no_grad():
#         scale = dp_strength
#         norm_factor_a = torch.div(torch.max(torch.norm(gradients, dim=1)), threshold + 1e-6).clamp(min=1.0)
#         protected_gradients = torch.div(gradients, norm_factor_a) + torch.normal(location, scale, gradients.shape).to(
#             device)
#         # # double noise
#         # norm_factor_b = torch.div(torch.max(torch.norm(pred_b_gradients_clone, dim=1)),
#         #                           threshold + 1e-6).clamp(min=1.0)
#         # pred_b_gradients_clone = torch.div(pred_b_gradients_clone, norm_factor_b) + \
#         #                          torch.normal(location, scale, pred_b_gradients_clone.shape).to(self.device)
#     return protected_gradients

# expectation (max norm)
def apply_max_norm_gradient_masking(g, **args):
    # add scalar noise with aligning the maximum norm
    g = g.cpu()
    g_original_shape = g.shape  # (1024, 128)
    g = g.view(g_original_shape[0], -1)  # (1024, 128)
    g_norm = torch.norm(g, dim=1, keepdim=True)
    g_norm = g_norm.view(-1, 1)
    max_norm = torch.max(g_norm)
    stds = torch.sqrt(torch.maximum(max_norm ** 2 / (g_norm ** 2 + 1e-32) - 1.0, torch.tensor(0.0)))  # (1024, 1)
    # standard_gaussian_noise = tf.random.normal(shape=(tf.shape(g)[0], 1), mean=0.0, stddev=1.0)  # (1024, 1)
    standard_gaussian_noise = torch.normal(size=(g.size(0), 1), mean=0.0, std=1.0)  # (1024, 1)
    gaussian_noise = standard_gaussian_noise * stds  # (1024, 128)
    res = g * (1 + gaussian_noise)  # (1024, 128)
    res = res.view(g_original_shape)
    return res


# white_gaussian (iso)
def apply_ios_gaussian_noise_gradient_masking(g, **args):
    # add scalar noise with aligning the maximum norm
    g = g.cpu()
    ratio = args.get("ratio")
    raise_error(ratio, "ratio", "apply_iso")

    g_original_shape = g.shape
    g = g.view(g_original_shape[0], -1)  # (1024, 128)

    g_norm = torch.norm(g, dim=1, keepdim=False)
    g_norm = g_norm.view(-1, 1)
    max_norm = torch.max(g_norm)
    gaussian_noise = torch.normal(size=g.shape, mean=0.0,
                                  std=ratio * max_norm / torch.sqrt(torch.tensor(g.shape[1], dtype=torch.float32)))
    res = g + gaussian_noise
    res = res.view(g_original_shape)
    return res


# gradient_perp_masking
def apply_gradient_perp_masking(g, lower=1.0, upper=10.0, **args):
    # add scalar noise with aligning the maximum norm
    g_original_shape = g.shape
    g = g.view(g_original_shape[0], -1)  # (1024, 128)

    g_norm = torch.norm(g, dim=1, keepdim=False)
    g_norm = g_norm.view(-1, 1)
    max_norm = torch.max(g_norm)

    std_gaussian_noise = torch.normal(size=g.shape, mean=0.0, std=1.0)
    inner_product = torch.sum(torch.multiply(std_gaussian_noise, g), dim=1, keepdim=True)
    init_perp = std_gaussian_noise - torch.divide(inner_product, torch.square(g_norm) + 1e-16) * g
    unit_perp = nn.functional.normalize(init_perp, p=2, dim=1)
    norm_to_align = torch.FloatTensor(g_norm.shape).uniform_(lower * max_norm, upper * max_norm)
    perp = torch.sqrt(torch.square(norm_to_align) - torch.square(g_norm) + 1e-8) * unit_perp
    res = g + perp
    res = res.view(g_original_shape)
    return res


def apply_marvell_gradient_perturb(g, **args):
    """construct a noise perturbation layer that uses sumKL to determine the type of perturbation

    Args:
        p_frac (str, optional): The weight for the trace of the covariance of the
                                positive example's gradient noise. Defaults to 'pos_frac'.
                                The default value will be computed from the fraction of
                                positive examples in the batch of gradients.
        dynamic (bool, optional): whether to adjust the power constraint hyperparameter P
                                to satisfy the error_prob_lower_bound or sumKL_threshold.
                                Defaults to False.
        error_prob_lower_bound (float, optional): The lower bound of the passive party's
                                detection error (the average of FPR and FNR). Given this
                                value, we will convert this into a corresponding sumKL_threshold,
                                where if the sumKL of the solution is lower than sumKL_threshold,
                                then the passive party detection error will be lower bounded
                                by error_prob_lower_bound.
                                Defaults to None.
        sumKL_threshold ([type], optional): Give a perturbation that have the sum of
                                KL divergences between the positive perturbed distribution
                                and the negative perturbed distribution upper bounded by
                                sumKL_threshold. Defaults to None.
        init_scale (float, optional): Determines the first value of the power constraint P.
                                P = init_scale * g_diff_norm**2. If dynamic, then this
                                init_scale could be increased geometrically if the solution
                                does not satisfy the requirement. Defaults to 1.0.
        uv_choice (str, optional): A string of three choices.
                                'uv' model the distribution of positive gradient and negative
                                gradient with individual isotropic gaussian distribution.
                                'same': average the computed empirical isotropic gaussian distribution.
                                'zero': assume positive gradient distribution and negative
                                gradient distribution to be dirac distributions. Defaults to 'uv'.

    Returns:
        [type]: [description]
    """
    y = args.get("y")
    raise_error(y, "y", "apply_marvell")

    g = g.cpu()
    y = y.cpu()

    if torch.sum(y) == 0 or len(y) == 0:
        return g

    init_scale = args.get("init_scale")
    raise_error(init_scale, "init_scale", "apply_marvell")

    p_frac = 'pos_frac'
    dynamic = False
    error_prob_lower_bound = None,
    sumKL_threshold = None
    uv_choice = 'uv'

    # print('p_frac', p_frac)
    # print('dynamic', dynamic)
    if dynamic and (error_prob_lower_bound is not None):
        '''
        if using dynamic and error_prob_lower_bound is specified, we use it to 
        determine the sumKL_threshold and overwrite what is stored in it before.
        '''
        sumKL_threshold = (2 - 4 * error_prob_lower_bound) ** 2
        # print('error_prob_lower_bound', error_prob_lower_bound)
        # print('implied sumKL_threshold', sumKL_threshold)
    # elif dynamic:
    # print('using sumKL_threshold', sumKL_threshold)

    # print('init_scale', init_scale)
    # print('uv_choice', uv_choice)

    # the batch label was stored in shared_var.batch_y in train_and_test
    # print('start')
    # start = time.time()
    g_original_shape = g.shape  # (1024, 128)
    g = g.view(g_original_shape[0], -1)  # (1024, 128)

    # y = shared_var.batch_y  # (1024)
    pos_g = g[y == 1]  # (n, 128)
    pos_g_mean = torch.mean(pos_g, dim=0, keepdim=True)  # shape [1, 128]
    pos_coordinate_var = torch.mean(torch.square(pos_g - pos_g_mean), dim=0)  # use broadcast (128)
    neg_g = g[y == 0]  # (n, 128)
    neg_g_mean = torch.mean(neg_g, dim=0, keepdim=True)  # shape [1, 128]
    neg_coordinate_var = torch.mean(torch.square(neg_g - neg_g_mean), dim=0)  # (128)

    avg_pos_coordinate_var = torch.mean(pos_coordinate_var)  # (1)
    avg_neg_coordinate_var = torch.mean(neg_coordinate_var)  # (1)
    # print('pos', avg_pos_coordinate_var)
    # print('neg', avg_neg_coordinate_var)

    g_diff = pos_g_mean - neg_g_mean  # (1, 128)
    g_diff_norm = float(torch.norm(g_diff))  # (1)
    # if g_diff_norm ** 2 > 1:
    #     print('pos_g_mean', pos_g_mean.shape)
    #     print('neg_g_mean', neg_g_mean.shape)
    #     assert g_diff_norm

    if uv_choice == 'uv':
        u = float(avg_neg_coordinate_var)
        v = float(avg_pos_coordinate_var)
        # if u == 0.0:
        #     print('neg_g')
        #     # print(neg_g)
        # if v == 0.0:
        #     print('pos_g')
        #     print(pos_g)

    elif uv_choice == 'same':
        u = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
        v = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
    elif uv_choice == 'zero':
        u, v = 0.0, 0.0

    d = float(g.shape[1])

    if p_frac == 'pos_frac':
        p = float(torch.sum(y) / len(y))  # p is set as the fraction of positive in the batch
    else:
        p = float(p_frac)

    # print('u={0},v={1},d={2},g={3},p={4},P={5}'.format(u,v,d,g_diff_norm**2,p,P))

    scale = init_scale

    # print('compute problem instance', time.time() - start)
    # start = time.time()

    lam10, lam20, lam11, lam21 = None, None, None, None
    while True:
        P = scale * g_diff_norm ** 2
        # print('g_diff_norm ** 2', g_diff_norm ** 2)
        # print('P', P)
        # print('u, v, d, p', u, v, d, p)
        lam10, lam20, lam11, lam21, sumKL = solve_isotropic_covariance(u=u, v=v, d=d, g=g_diff_norm ** 2, p=p,
                                                                       P=P, lam10_init=lam10, lam20_init=lam20,
                                                                       lam11_init=lam11, lam21_init=lam21)
        # print('sumKL', sumKL)
        # print()

        # print(scale)
        if not dynamic or sumKL <= sumKL_threshold:
            break

        scale *= 1.5  # loosen the power constraint

    # print('solving time', time.time() - start)
    # start = time.time()

    perturbed_g = g  # (1024, 128)
    y_float = y.clone().detach().float()  # (1024)

    # positive examples add noise in g1 - g0
    perturbed_g += torch.multiply(torch.normal(mean=0, std=1, size=y.shape), y_float).view(-1, 1) * g_diff * (
            math.sqrt(lam11 - lam21) / g_diff_norm)  # (1024, 128)

    # add spherical noise to positive examples
    if lam21 > 0.0:
        perturbed_g += torch.normal(mean=0, std=1, size=g.shape) * y_float.view(-1, 1) * math.sqrt(lam21)

    # negative examples add noise in g1 - g0
    perturbed_g += torch.multiply(torch.normal(mean=0, std=1, size=y.shape), 1 - y_float).view(-1, 1) * g_diff * (
            math.sqrt(lam10 - lam20) / g_diff_norm)

    # add spherical noise to negative examples
    if lam20 > 0.0:
        perturbed_g += torch.normal(mean=0, std=1, size=g.shape) * (1 - y_float).view(-1, 1) * math.sqrt(
            lam20)  # (1024, 128)

    # print('noise adding', time.time() - start)

    # print('a')
    # print(perturbed_g)
    # print('b')
    # print(perturbed_g[y==1])
    # print('c')
    # print(perturbed_g[y==0])

    '''
    pos_cov = tf.linalg.matmul(a=g[y==1] - pos_g_mean, b=g[y==1] - pos_g_mean, transpose_a=True) / g[y==1].shape[0]
    print('pos_var', pos_coordinate_var)
    print('pos_cov', pos_cov)
    print('raw svd', tf.linalg.svd(pos_cov, compute_uv=False))
    print('diff svd', tf.linalg.svd(pos_cov - tf.linalg.tensor_diag(pos_coordinate_var), compute_uv=False))
    # assert False
    '''
    # if shared_var.counter < 2000:
    #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_unperturbed',
    #             arr=g.numpy())
    #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_cut_layer_perturbed',
    #             arr=perturbed_g.numpy())
    #     np.save(file=os.path.join(shared_var.logdir, str(shared_var.counter)) + '_label',
    #             arr=shared_var.batch_y.numpy())
    perturbed_g = perturbed_g.view(g_original_shape)
    return perturbed_g


#
# def apply_marvell(gradients, **args):
#     marvell_s = args["marvell_s"]
#     raise_error(marvell_s, "marvell_s", "apply_marvell")
#     num_classes = args["num_classes"]
#     assert num_classes == 2
#     return KL_gradient_perturb_tf(gradients, marvell_s)


DEFENSE_FUNCTIONS_DICT = {DefenseName.D_SGD: apply_d_sgd,
                          DefenseName.DP_LAPLACE: apply_dp_laplace,
                          DefenseName.GRAD_COMPRESSION: apply_grad_compression,
                          DefenseName.PPDL: apply_dp_gc_ppdl,
                          DefenseName.ISO: apply_ios_gaussian_noise_gradient_masking,
                          DefenseName.MAX_NORM: apply_max_norm_gradient_masking,
                          DefenseName.MARVELL: apply_marvell_gradient_perturb,
                          DefenseName.NONE: apply_identity}

# if __name__ == '__main__':
#     result = apply_KL_gradient_perturb(torch.tensor([[0.2, 0.1],
#                                                      [0.1, 0.21]]), 0.2)
#     print("result:{}".format(result))
