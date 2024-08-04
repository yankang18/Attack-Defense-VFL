import torch


def direct_label_inference_attack(gradient, attack_assist_args):
    output_dim = gradient.shape[-1]
    if output_dim == 1:
        pred = torch.lt(gradient, 0).long()
    elif output_dim == 2:
        pred = torch.lt(gradient[:, 1], 0).long()
    else:
        raise Exception("DLI only works for binary classification task for now.")
    return pred.cpu().numpy()
