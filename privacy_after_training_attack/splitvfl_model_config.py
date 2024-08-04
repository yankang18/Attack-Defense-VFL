# from models.alexnet_normal import *
# from models.alexnet_passport import *
# from models.resnet_passport import *

from splitvfl_models.convnet import get_convnet5
from splitvfl_models.fullyconnet_normal import LeNetClassifier, InteractiveModel
from splitvfl_models.lenet import LeNetWithAct, get_lenet


# from splitvfl_models.alexnet import AlexNet


# def construct_model(force_passport):
#     # model_name = args.model_name
#     # passportlayer = args.passportlayer
#     # normlayer = args.normlayer
#     # # pretrain = args.pretrain
#     # if model_name == 'alex_mnist':
#     #     model = AlexNetNormal(1, 10, 3)
#     # elif model_name == 'LeNet_mnist':
#     #     passport_pos = construct_passport_pos(model_name, passportlayer)
#     #     norm_pos = construct_norm_pos(model_name, normlayer)
#     #     model = LeNetPassport(1, 10, passport_pos, norm_pos)
#     # elif model_name == 'LeNet_cifar10':
#     #     passport_pos = construct_passport_pos(model_name, passportlayer)
#     #     norm_pos = construct_norm_pos(model_name, normlayer)
#     #     model = LeNetPassport(3, 10, passport_pos, norm_pos)
#     # elif model_name == 'alex_cifar10':
#     #     passport_pos = construct_passport_pos(model_name, passportlayer)
#     #     norm_pos = construct_norm_pos(model_name, normlayer)
#     #     model = AlexNetPassport(3, 10, passport_pos, norm_pos)
#     # elif model_name == 'resnet_cifar100':
#     #     passport_kwargs = load_passport_config()
#     #     model = ResNet18Passport(passport_kwargs=passport_kwargs)
#     # else:
#     #     raise NameError('Model {} is not implemented yet!'.format(model_name))
#     # return model
#     # return get_convnet5()
#     return LeNetWithAct(1, passport_pos=force_passport)


def get_models_2(args):
    print("[DEBUG] Construct top model:")
    task_model = LeNetClassifier(dims=(1024, 256), num_classes=10, passport_pos=args["top_model_force_passport"], act="relu")
    interactive_passive = InteractiveModel(dims=(4096, 1024))
    interactive_active = InteractiveModel(dims=(4096, 1024))

    if args["active_model_force_passport"] is not None:
        print("[DEBUG] Construct active bottom model:")
        model_active = get_convnet5()
    else:
        print("[DEBUG] No active model is constructed:")
        model_active = None

    print("[DEBUG] Construct passive bottom model:")

    model_passive = get_convnet5()
    print("[DEBUG] model_passive : \n {}".format(model_passive))
    print("[DEBUG] model_active : \n {}".format(model_active))
    print("[DEBUG] interactive_passive : \n {}".format(interactive_passive))
    print("[DEBUG] interactive_active : \n {}".format(interactive_active))
    print("[DEBUG] task_model : \n {}".format(task_model))
    return model_passive, model_active, interactive_passive, interactive_active, task_model


def get_models(args):
    print("[DEBUG] Construct top model:")
    # task_model = FullyConnNet(in_channels=12, num_classes=10, passport_pos=args["top_model_force_passport"])
    task_model = LeNetClassifier(dims=(120, 84), num_classes=10, passport_pos=args["top_model_force_passport"], act="relu")
    interactive_passive = InteractiveModel(dims=(128, 120))
    interactive_active = InteractiveModel(dims=(128, 120))

    if args["active_model_force_passport"] is not None:
        print("[DEBUG] Construct active bottom model:")
        model_active = get_lenet(struct_config=None)
    else:
        print("[DEBUG] No active model is constructed:")
        model_active = None

    print("[DEBUG] Construct passive bottom model:")
    model_passive = get_lenet(struct_config=None)
    print("[DEBUG] model_passive : \n {}".format(model_passive))
    print("[DEBUG] model_active : \n {}".format(model_active))
    print("[DEBUG] interactive_passive : \n {}".format(interactive_passive))
    print("[DEBUG] interactive_active : \n {}".format(interactive_active))
    print("[DEBUG] task_model : \n {}".format(task_model))
    return model_passive, model_active, interactive_passive, interactive_active, task_model
