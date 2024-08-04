import torch
from sklearn.metrics import roc_auc_score

from models.ctrdnn_torch import get_dnnfm
from models.lenet_torch import get_lenet
from models.mlp_torch import get_active_linear_model, get_passive_linear_model
from models.mlp_torch import get_mlp, get_passive_interactive_model, get_active_interactive_model
from models.resnet_torch import get_resnet18
from utils import printf


class ArchConfigName(object):
    VLR = "VLR"
    VNN_MLP = "VNN_MLP"
    VNN_MLP_V2 = "VNN_MLP_V2"
    VNN_RESNET = "VNN_RESNET"
    VNN_LENET = "VNN_LENET"
    VNN_DNNFM = "VNN_DNNFM"
    VNN_DNNFM_V2 = "VNN_DNNFM_V2"


class ModelType(object):
    PASSIVE_BOTTOM = "passive_bottom"
    ACTIVE_BOTTOM = "active_bottom"
    INTERACTIVE_LAYER = "interactive_layer"
    ACTIVE_INTERACTIVE_LAYER = "active_interactive_layer"
    PASSIVE_INTERACTIVE_LAYER = "passive_interactive_layer"
    NEGATIVE_LOSS_MODEL = "negative_loss_model"
    TASK_MODEL = "task_model"


class OptimName(object):
    SGD = "SGD"
    ADAM = "ADAM"
    ADAGRAD = "ADAGRAD"


MODEL_TYPE_LIST = [ModelType.PASSIVE_BOTTOM, ModelType.ACTIVE_BOTTOM, ModelType.INTERACTIVE_LAYER, ModelType.TASK_MODEL]


class TaskModelType(object):
    MLP_0_LAYER = "MLP_0_LAYER"
    MLP_1_LAYER = "MLP_1_LAYER"
    MLP_2_LAYER = "MLP_2_LAYER"
    MLP_3_LAYER = "MLP_3_LAYER"


def determine_sub_vfl_type(vfl_type, has_active_bottom):
    """
        Determine sub vfl type: 'VLR', 'VSNN' or 'VHNN'.
    :return: sub_vfl_type
    """

    if vfl_type == "VNN":
        sub_vfl_type = "VHNN" if has_active_bottom else "VSNN"
    elif vfl_type == "VLR":
        sub_vfl_type = "VLR"
    else:
        raise Exception("Does not support vfl type:{}".format(vfl_type))
    return sub_vfl_type


def evaluate_result(true_value, pred_score, eval_metric="AUC"):
    auc = roc_auc_score(y_true=true_value, y_score=pred_score)
    return auc


def f_beta_score(value1, value2, beta=1):
    """
        A more general F score, that uses a positive real factor β, where β is chosen such that
        'value2' is considered β times as important as 'value1'
    """

    beta_square = beta ** 2
    return (1 + beta_square) * value1 * value2 / (beta_square * value1 + value2)


def build_interactive_layer(arch_dict):
    interactive_layer_config = arch_dict["interactive_layer_config"]
    if interactive_layer_config is not None:
        return interactive_layer_config["model"](dims=interactive_layer_config["dims"])
    else:
        return None


def build_layer_model(model_config):
    if model_config is not None:
        model = model_config["model_fn"](struct_config=model_config["struct_config"])
        return model
    else:
        return None


def build_task_model(arch_dict):
    task_model_config = arch_dict["task_model_config"]
    if task_model_config is not None:
        return task_model_config["model"](dims=task_model_config["dims"])
    else:
        return None


def build_models(arch_config: dict, verbose=False):
    passive_bottom_model = build_layer_model(arch_config["passive_bottom_config"])
    printf("[DEBUG] passive_bottom_model struct: \n {}".format(passive_bottom_model), verbose=verbose)

    active_bottom_model = build_layer_model(arch_config["active_bottom_config"])
    printf("[DEBUG] active_bottom_model struct: \n {}".format(active_bottom_model), verbose=verbose)

    interactive_layer = build_layer_model(arch_config["interactive_layer_config"])
    printf("[DEBUG] interactive_layer struct: \n {}".format(interactive_layer), verbose=verbose)

    negative_loss_model = build_layer_model(arch_config["negative_loss_model_config"])
    printf("[DEBUG] negative_loss_model struct: \n {}".format(negative_loss_model), verbose=verbose)

    passive_interactive_layer = build_layer_model(arch_config["passive_interactive_layer_config"])
    printf("[DEBUG] passive_interactive_layer struct: \n {}".format(passive_interactive_layer), verbose=verbose)

    active_interactive_layer = build_layer_model(arch_config["active_interactive_layer_config"])
    printf("[DEBUG] active_interactive_layer struct: \n {}".format(active_interactive_layer), verbose=verbose)

    task_model = build_layer_model(arch_config["task_model_config"])
    printf("[DEBUG] task_model struct: \n {}".format(task_model), verbose=verbose)

    return {ModelType.PASSIVE_BOTTOM: passive_bottom_model,
            ModelType.ACTIVE_BOTTOM: active_bottom_model,
            ModelType.INTERACTIVE_LAYER: interactive_layer,
            ModelType.PASSIVE_INTERACTIVE_LAYER: passive_interactive_layer,
            ModelType.NEGATIVE_LOSS_MODEL: negative_loss_model,
            ModelType.ACTIVE_INTERACTIVE_LAYER: active_interactive_layer,
            ModelType.TASK_MODEL: task_model}


def prepare_optimizers(model_dict, learning_rate=0.01, weight_decay=0.0, optimizer_args_dict=None,
                       optim_name=OptimName.ADAM, momentum=0.95):
    """
        Create optimizers for models in 'model_dict'.

    :param model_dict: dictionary for models.
    :param learning_rate: default learning rate for all models.
    :param weight_decay: default weight decay for all models.
    :param optimizer_args_dict: optimizer arguments (e.g., learning rate) for specified models.
    :param optim_name: the name of the optimizer.
    :param momentum: the momentum for SGD optimizer.
    :return: optimizer dictionary corresponding to models in the 'model_dict'.
    """

    optimizer_dict = dict()
    for model_name, model in model_dict.items():
        if model is not None:
            lr = learning_rate
            wd = weight_decay
            optim_args = optimizer_args_dict.get(model_name) if optimizer_args_dict else None
            if optim_args is not None:
                _lr = optim_args.get("learning_rate")
                _wd = optim_args.get("weight_decay")

                lr = _lr if _lr is not None else lr
                wd = _wd if _wd is not None else wd

            optim_name = optim_name.upper()
            if optim_name == OptimName.ADAM:
                optimizer_dict[model_name] = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=wd)
            elif optim_name == OptimName.ADAGRAD:
                optimizer_dict[model_name] = torch.optim.Adagrad(list(model.parameters()), lr=lr, weight_decay=wd)
            elif optim_name == OptimName.SGD:
                optimizer_dict[model_name] = torch.optim.SGD(list(model.parameters()), lr=lr, weight_decay=wd,
                                                             momentum=momentum)
            else:
                raise Exception("[INFO] Does not support optimizer {}".format(optim_name))

            printf(
                "[INFO] create [{}] optimizer for [{}] with lr-[{}], wd-[{}].".format(optim_name, model_name, lr, wd),
                verbose=False)
        else:
            printf("[INFO] [{}] is None".format(model_name), verbose=False)
    return optimizer_dict


def get_architecture_config(arch_config_name,
                            input_dims,
                            num_classes=2,
                            has_active_bottom=True,
                            has_interactive_layer=True,
                            task_model_type=TaskModelType.MLP_1_LAYER,
                            models_args=None):
    """
    Get the model configuration of the VFL architecture.

    :param arch_config_name: the type of vertical federated learning architecture, should be one of the ArchType.
    :param input_dims: the list of input dimensions for parties' bottom models.
    :param num_classes: the number of classes for the classification problem.
    :param has_active_bottom: whether the active party has bottom model (default True).
    :param has_interactive_layer: whether the active party has interactive layer (default True).
    :param task_model_type: the type of task model, should be one of the TaskModelType.
    :param models_args: the outside arguments for models.
    :return: the dictionary describing the VFL architecture.
    """

    # The last element in input_dims is for active party.
    passive_input_dim = input_dims[0]
    active_input_dim = input_dims[1]

    arch_config_dict = {ArchConfigName.VLR: [get_vlr, 0],
                        ArchConfigName.VNN_MLP: [get_vnn_mlp, 0],
                        ArchConfigName.VNN_MLP_V2: [get_vnn_mlp, 1],
                        ArchConfigName.VNN_LENET: [get_vnn_lenet, 0],
                        ArchConfigName.VNN_RESNET: [get_vnn_resnet, 0],
                        ArchConfigName.VNN_DNNFM: [get_vnn_dnnfm, 0],
                        ArchConfigName.VNN_DNNFM_V2: [get_vnn_dnnfm, 1]}
    get_arch_config_fn, arch_idx = arch_config_dict.get(arch_config_name)
    arch_config_list = get_arch_config_fn(has_active_bottom,
                                          has_interactive_layer,
                                          task_model_type,
                                          active_input_dim,
                                          passive_input_dim,
                                          num_classes,
                                          models_args)

    sel_arch_config = arch_config_list[arch_idx]
    if sel_arch_config is None:
        raise Exception("Does not support architecture type : [{}].".format(arch_config_name))

    return sel_arch_config


def get_model_shape(has_active_bottom,
                    has_interactive_layer,
                    interactive_layer_output,
                    two_party_bottom_model_output_dim):
    interactive_layer_input = None
    active_intr_model_input, active_intr_model_output = None, None
    passive_intr_model_input, passive_intr_model_output = None, None
    if has_interactive_layer:
        interactive_layer_input = two_party_bottom_model_output_dim
        passive_intr_model_output = interactive_layer_output
        active_intr_model_output = interactive_layer_output
        task_model_input = interactive_layer_output
        adversarial_model_input = active_intr_model_output

        passive_intr_model_input = int(two_party_bottom_model_output_dim / 2)
        active_intr_model_input = int(two_party_bottom_model_output_dim / 2)
        if has_active_bottom is False:
            # has only passive bottom model
            interactive_layer_input = int(two_party_bottom_model_output_dim / 2)
    else:
        task_model_input = two_party_bottom_model_output_dim
        adversarial_model_input = int(two_party_bottom_model_output_dim / 2)
        if has_active_bottom is False:
            # has only passive bottom model
            task_model_input = int(two_party_bottom_model_output_dim / 2)

    model_shape_dict = dict()
    model_shape_dict["active_intr_model_input"] = active_intr_model_input
    model_shape_dict["active_intr_model_output"] = active_intr_model_output
    model_shape_dict["passive_intr_model_input"] = passive_intr_model_input
    model_shape_dict["passive_intr_model_output"] = passive_intr_model_output
    model_shape_dict["interactive_layer_input"] = interactive_layer_input
    model_shape_dict["task_model_input"] = task_model_input
    model_shape_dict["adversarial_model_input"] = adversarial_model_input

    return model_shape_dict


# ============================= model configuration for VFL architecture ====================================
# If certain model does not appear in the architecture, it configuration would be None.
# for example, there is no interactive layer and top model in VLR. Therefore, the configuration for
# interactive layer and top model is None.
# ===========================================================================================================


def get_vlr(has_active_bottom,
            has_interactive_layer,
            task_model_type,
            active_input_dim,
            passive_input_dim,
            num_classes,
            model_args=None):
    vlr = {"passive_bottom_config": {"model_fn": get_passive_linear_model,
                                     "struct_config": {"layer_input_dim_list": [passive_input_dim, 1]}},
           "active_bottom_config": {"model_fn": get_active_linear_model,
                                    "struct_config": {"layer_input_dim_list": [active_input_dim, 1]}},
           "interactive_layer_config": None,
           "passive_interactive_layer_config": None,
           "active_interactive_layer_config": None,
           "negative_loss_model_config": None,
           "task_model_config": None,
           "vfl_type": "VLR"}
    return vlr, None


# ==============================
# = 2-layer MLP neural network =
# ==============================

def get_vnn_mlp(has_active_bottom,
                has_interactive_layer,
                task_model_type,
                active_input_dim,
                passive_input_dim,
                num_classes,
                model_args=None):
    two_party_bottom_model_output_dim = 64
    interactive_layer_output = 32

    model_shape_dict = get_model_shape(has_active_bottom,
                                       has_interactive_layer,
                                       interactive_layer_output,
                                       two_party_bottom_model_output_dim)
    active_intr_model_input = model_shape_dict["active_intr_model_input"]
    active_intr_model_output = model_shape_dict["active_intr_model_output"]
    passive_intr_model_input = model_shape_dict["passive_intr_model_input"]
    passive_intr_model_output = model_shape_dict["passive_intr_model_output"]
    interactive_layer_input = model_shape_dict["interactive_layer_input"]
    task_model_input = model_shape_dict["task_model_input"]
    adversarial_model_input = model_shape_dict["adversarial_model_input"]

    # task_act = "relu"
    task_act = "sigmoid"
    task_model_dict = {TaskModelType.MLP_0_LAYER: None,
                       TaskModelType.MLP_1_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, num_classes],
                                                       "act_type": task_act
                                                   }},
                       TaskModelType.MLP_2_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, 32, num_classes],
                                                       "act_type": task_act}},
                       TaskModelType.MLP_3_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, 32, 16,
                                                                                num_classes],
                                                       "act_type": task_act}}}

    task_model = task_model_dict[task_model_type]
    if task_model is None:
        interactive_layer_output = num_classes

    act = "leakyrelu"
    vnn_mlp = {"passive_bottom_config": {"model_fn": get_mlp,
                                         "struct_config": {
                                             "layer_input_dim_list": [passive_input_dim, 64, 32], "act_type": act,
                                             "final_act_type": None}},
               "active_bottom_config": {"model_fn": get_mlp,
                                        "struct_config": {
                                            "layer_input_dim_list": [active_input_dim, 64, 32],
                                            "act_type": act,
                                            "final_act_type": None}} if has_active_bottom else None,
               "interactive_layer_config": {"model_fn": get_mlp,
                                            "struct_config": {
                                                "layer_input_dim_list": [interactive_layer_input,
                                                                         interactive_layer_output],
                                                "act_type": act,
                                                "final_act_type": None}} if has_interactive_layer else None,
               "passive_interactive_layer_config": None,
               "active_interactive_layer_config": None,
               "negative_loss_model_config": {"model_fn": get_mlp,
                                              "struct_config": {
                                                  "layer_input_dim_list": [adversarial_model_input,
                                                                           num_classes],
                                                  "act_type": act}
                                              },
               "task_model_config": task_model,
               "vfl_type": "VNN"}

    vnn_mlp_v2 = {"passive_bottom_config": {"model_fn": get_mlp,
                                            "struct_config": {
                                                "layer_input_dim_list": [passive_input_dim, 64, 32],
                                                "act_type": act,
                                                "final_act_type": act}},
                  "active_bottom_config": {"model_fn": get_mlp,
                                           "struct_config": {
                                               "layer_input_dim_list": [active_input_dim, 64, 32],
                                               "act_type": act,
                                               "final_act_type": act}
                                           } if has_active_bottom else None,
                  "interactive_layer_config": {"model_fn": get_mlp,
                                               "struct_config": {
                                                   "layer_input_dim_list": [interactive_layer_input,
                                                                            interactive_layer_output],
                                                   "act_type": act,
                                                   "final_act_type": act}
                                               } if has_interactive_layer else None,
                  "passive_interactive_layer_config": {"model_fn": get_passive_interactive_model,
                                                       "struct_config": {
                                                           "layer_input_dim_list": [passive_intr_model_input,
                                                                                    passive_intr_model_output],
                                                           "final_act_type": act}
                                                       } if has_interactive_layer else None,
                  "active_interactive_layer_config": {"model_fn": get_active_interactive_model,
                                                      "struct_config": {
                                                          "layer_input_dim_list": [active_intr_model_input,
                                                                                   active_intr_model_output],
                                                          "final_act_type": act}
                                                      } if (has_interactive_layer and has_active_bottom) else None,
                  "negative_loss_model_config": {"model_fn": get_mlp,
                                                 "struct_config": {
                                                     "layer_input_dim_list": [adversarial_model_input,
                                                                              num_classes],
                                                     "act_type": act}
                                                 },
                  "task_model_config": task_model,
                  "vfl_type": "VNN"}
    return vnn_mlp, vnn_mlp_v2


# =========================
# = LetNet neural network =
# =========================

def get_vnn_lenet(has_active_bottom,
                  has_interactive_layer,
                  task_model_type,
                  active_input_dim,
                  passive_input_dim,
                  num_classes,
                  model_args=None):
    two_party_bottom_model_output_dim = 256
    interactive_layer_output = 128

    model_shape_dict = get_model_shape(has_active_bottom,
                                       has_interactive_layer,
                                       interactive_layer_output,
                                       two_party_bottom_model_output_dim)
    active_intr_model_input = model_shape_dict["active_intr_model_input"]
    active_intr_model_output = model_shape_dict["active_intr_model_output"]
    passive_intr_model_input = model_shape_dict["passive_intr_model_input"]
    passive_intr_model_output = model_shape_dict["passive_intr_model_output"]
    interactive_layer_input = model_shape_dict["interactive_layer_input"]
    task_model_input = model_shape_dict["task_model_input"]
    adversarial_model_input = model_shape_dict["adversarial_model_input"]

    task_model_dict = {TaskModelType.MLP_0_LAYER: None,
                       TaskModelType.MLP_1_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, num_classes],
                                                       "act_type": "relu"}
                                                   },
                       TaskModelType.MLP_2_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list":
                                                           [task_model_input, 84, num_classes],
                                                       "act_type": "relu"}
                                                   },
                       TaskModelType.MLP_3_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list":
                                                           [task_model_input, 84, 32, num_classes],
                                                       "act_type": "relu"}
                                                   }
                       }

    task_model = task_model_dict[task_model_type]
    if task_model is None:
        interactive_layer_output = num_classes

    vnn_lenet = {"passive_bottom_config": {"model_fn": get_lenet,
                                           "struct_config": None},
                 "active_bottom_config": {"model_fn": get_lenet,
                                          "struct_config": None} if has_active_bottom else None,
                 "interactive_layer_config": {"model_fn": get_mlp,
                                              "struct_config": {
                                                  "layer_input_dim_list": [interactive_layer_input,
                                                                           interactive_layer_output]}
                                              } if has_interactive_layer else None,
                 "passive_interactive_layer_config": {"model_fn": get_passive_interactive_model,
                                                      "struct_config": {
                                                          "layer_input_dim_list": [passive_intr_model_input,
                                                                                   passive_intr_model_output]}
                                                      } if has_interactive_layer else None,
                 "active_interactive_layer_config": {"model_fn": get_active_interactive_model,
                                                     "struct_config": {
                                                         "layer_input_dim_list": [active_intr_model_input,
                                                                                  active_intr_model_output]}
                                                     } if has_interactive_layer else None,
                 "negative_loss_model_config": {"model_fn": get_mlp,
                                                "struct_config": {
                                                    "layer_input_dim_list": [adversarial_model_input,
                                                                             num_classes]}
                                                },
                 "task_model_config": task_model,
                 "vfl_type": "VNN"}

    return vnn_lenet, None


# =========================
# = ResNet neural network =
# =========================

def get_vnn_resnet(has_active_bottom,
                   has_interactive_layer,
                   task_model_type,
                   active_input_dim,
                   passive_input_dim,
                   num_classes,
                   model_args=None):
    two_party_bottom_model_output_dim = 128
    interactive_layer_output = 64

    model_shape_dict = get_model_shape(has_active_bottom,
                                       has_interactive_layer,
                                       interactive_layer_output,
                                       two_party_bottom_model_output_dim)
    active_intr_model_input = model_shape_dict["active_intr_model_input"]
    active_intr_model_output = model_shape_dict["active_intr_model_output"]
    passive_intr_model_input = model_shape_dict["passive_intr_model_input"]
    passive_intr_model_output = model_shape_dict["passive_intr_model_output"]
    interactive_layer_input = model_shape_dict["interactive_layer_input"]
    task_model_input = model_shape_dict["task_model_input"]
    adversarial_model_input = model_shape_dict["adversarial_model_input"]

    task_model_dict = {TaskModelType.MLP_0_LAYER: None,
                       TaskModelType.MLP_1_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, num_classes]}
                                                   },
                       TaskModelType.MLP_2_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, 64, num_classes]}
                                                   },
                       TaskModelType.MLP_3_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, 64, 32, num_classes]}
                                                   }
                       }

    task_model = task_model_dict[task_model_type]
    if task_model is None:
        interactive_layer_output = num_classes

    vnn_resnet20 = {"passive_bottom_config": {"model_fn": get_resnet18,
                                              "struct_config": None},
                    "active_bottom_config": {"model_fn": get_resnet18,
                                             "struct_config": None} if has_active_bottom else None,
                    "interactive_layer_config": {"model_fn": get_mlp,
                                                 "struct_config": {
                                                     "layer_input_dim_list": [interactive_layer_input,
                                                                              interactive_layer_output]}
                                                 } if has_interactive_layer else None,
                    "passive_interactive_layer_config": {"model_fn": get_passive_interactive_model,
                                                         "struct_config": {
                                                             "layer_input_dim_list": [passive_intr_model_input,
                                                                                      passive_intr_model_output]}
                                                         } if has_interactive_layer else None,
                    "active_interactive_layer_config": {"model_fn": get_active_interactive_model,
                                                        "struct_config": {
                                                            "layer_input_dim_list": [active_intr_model_input,
                                                                                     active_intr_model_output]}
                                                        } if has_interactive_layer else None,
                    "negative_loss_model_config": {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [adversarial_model_input,
                                                                                num_classes]}
                                                   },
                    "task_model_config": task_model,
                    "vfl_type": "VNN"}

    return vnn_resnet20, None


# =========================
# = DNN FM neural network =
# =========================

def get_vnn_dnnfm(has_active_bottom,
                  has_interactive_layer,
                  task_model_type,
                  active_input_dim,
                  passive_input_dim,
                  num_classes,
                  model_args):
    two_party_bottom_model_output_dim = 128
    interactive_layer_output = 64

    model_shape_dict = get_model_shape(has_active_bottom,
                                       has_interactive_layer,
                                       interactive_layer_output,
                                       two_party_bottom_model_output_dim)
    active_intr_model_input = model_shape_dict["active_intr_model_input"]
    active_intr_model_output = model_shape_dict["active_intr_model_output"]
    passive_intr_model_input = model_shape_dict["passive_intr_model_input"]
    passive_intr_model_output = model_shape_dict["passive_intr_model_output"]
    interactive_layer_input = model_shape_dict["interactive_layer_input"]
    task_model_input = model_shape_dict["task_model_input"]
    adversarial_model_input = model_shape_dict["adversarial_model_input"]

    # task_model_act = "relu"
    task_model_act = "leakyrelu"
    task_model_dict = {TaskModelType.MLP_0_LAYER: None,
                       TaskModelType.MLP_1_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, num_classes],
                                                       "act_type": task_model_act
                                                   }},
                       TaskModelType.MLP_2_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, 64, num_classes],
                                                       "act_type": task_model_act}},
                       TaskModelType.MLP_3_LAYER: {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [task_model_input, 64, 32,
                                                                                num_classes],
                                                       "act_type": task_model_act}}}

    task_model = task_model_dict[task_model_type]
    if task_model is None:
        interactive_layer_output = num_classes
        passive_intr_model_output = num_classes
        active_intr_model_output = num_classes

    act = 'relu'
    dnnfm_col_names = model_args["dnnfm"]["col_names"]
    vnn_dnnfm = {"passive_bottom_config": {"model_fn": get_dnnfm,
                                           "struct_config": {
                                               "col_names": dnnfm_col_names[0]
                                           }},
                 "active_bottom_config": {"model_fn": get_dnnfm,
                                          "struct_config": {
                                              "col_names": dnnfm_col_names[1]
                                          }} if has_active_bottom else None,
                 "interactive_layer_config": {"model_fn": get_mlp,
                                              "struct_config": {
                                                  "layer_input_dim_list": [interactive_layer_input,
                                                                           interactive_layer_output],
                                                  "act_type": act,
                                                  "final_act_type": None}} if has_interactive_layer else None,
                 "passive_interactive_layer_config": None,
                 "active_interactive_layer_config": None,
                 "negative_loss_model_config": {"model_fn": get_mlp,
                                                "struct_config": {
                                                    "layer_input_dim_list": [interactive_layer_output,
                                                                             num_classes],
                                                    "act_type": act}
                                                },
                 "task_model_config": task_model,
                 "vfl_type": "VNN"}

    vnn_dnnfm_v2 = {"passive_bottom_config": {"model_fn": get_dnnfm,
                                              "struct_config": {
                                                  "col_names": dnnfm_col_names[0]
                                              }},
                    "active_bottom_config": {"model_fn": get_dnnfm,
                                             "struct_config": {
                                                 "col_names": dnnfm_col_names[1]
                                             }} if has_active_bottom else None,
                    "interactive_layer_config": {"model_fn": get_mlp,
                                                 "struct_config": {
                                                     "layer_input_dim_list": [interactive_layer_input,
                                                                              interactive_layer_output],
                                                     "act_type": act,
                                                     "final_act_type": act}
                                                 } if has_interactive_layer else None,
                    "passive_interactive_layer_config": {"model_fn": get_passive_interactive_model,
                                                         "struct_config": {
                                                             "layer_input_dim_list": [passive_intr_model_input,
                                                                                      passive_intr_model_output],
                                                             "final_act_type": act}
                                                         } if has_interactive_layer else None,
                    "active_interactive_layer_config": {"model_fn": get_active_interactive_model,
                                                        "struct_config": {
                                                            "layer_input_dim_list": [active_intr_model_input,
                                                                                     active_intr_model_output],
                                                            "final_act_type": act}
                                                        } if (has_interactive_layer and has_active_bottom) else None,
                    "negative_loss_model_config": {"model_fn": get_mlp,
                                                   "struct_config": {
                                                       "layer_input_dim_list": [adversarial_model_input,
                                                                                num_classes],
                                                       "act_type": act}
                                                   },
                    "task_model_config": task_model,
                    "vfl_type": "VNN"}
    return vnn_dnnfm, vnn_dnnfm_v2

# if __name__ == '__main__':
#     has_interactive_layer_list = [True, False]
#     has_active_bottom_list = [True, False]
#     interactive_layer_output = 64
#     two_party_bottom_model_output_dim = 128
#     for has_interactive_layer in has_interactive_layer_list:
#         for has_active_bottom in has_active_bottom_list:
#             result_dict = get_model_shape(has_active_bottom,
#                                           has_interactive_layer,
#                                           interactive_layer_output,
#                                           two_party_bottom_model_output_dim)
#             print("has_interactive_layer:{}, has_active_bottom:{}".format(has_interactive_layer, has_active_bottom))
#             print(result_dict)
