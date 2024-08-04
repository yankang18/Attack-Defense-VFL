from collections import OrderedDict

import torch
import torch.nn as nn

from privacy_after_training_attack.splitvfl_models.layers.conv2d import ConvBlock
from privacy_after_training_attack.splitvfl_models.layers.passportconv2d_nonsignloss import PassportBlock


class ConvNetBottom(nn.Module):

    def __init__(self, structure_config):
        super(ConvNetBottom, self).__init__()

        inp = structure_config['in_channels']
        maxpool_idx_list = structure_config['maxpool_idx_list']
        oups = structure_config['layer_out_channels_dict']
        kp = structure_config['layer_kernel_padding_dict']
        norm_idx_dict = structure_config['layer_norm_dict']
        self.passport_idx_dict = structure_config['layer_passport_dict']
        self.fc_dim_list = structure_config['fc_dim_list']

        num_layer = len(kp) + len(maxpool_idx_list)

        layers = []
        for layer_idx in range(num_layer):
            if layer_idx in maxpool_idx_list:
                print(f"[DEBUG] Convnet layer:{layer_idx} max pool .")
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layer_idx][0]
                p = kp[layer_idx][1]
                if self.passport_idx_dict[layer_idx]:
                    print(f"[DEBUG] Convnet layer:{layer_idx} applies passport.")
                    layers.append(
                        PassportBlock(inp, oups[layer_idx], ks=k, s=1, pd=p, norm_type=norm_idx_dict[layer_idx]))
                else:
                    print(f"[DEBUG] Convnet layer:{layer_idx} does not apply passport.")
                    layers.append(ConvBlock(inp, oups[layer_idx], ks=k, s=1, pd=p, norm_type=norm_idx_dict[layer_idx]))

                inp = oups[layer_idx]

        self.features = nn.Sequential(*layers)

        if self.fc_dim_list is not None:
            self.fc = nn.Linear(self.fc_dim_list[0], self.fc_dim_list[1])

        self.scale = None
        self.bias = None

    def get_number_passports(self):
        dict1 = OrderedDict(sorted(self.passport_idx_dict.items()))
        return "".join(["1" if val else "0" for _, val in dict1.items()])

    def forward(self, x, force_passport=True):
        for m in self.features:
            if isinstance(m, PassportBlock):
                x, self.scale, self.bias = m(x, force_passport)
            else:
                x = m(x)

        x = x.view(x.size(0), -1)

        if self.fc_dim_list is not None:
            x = self.fc(x)
        return x


def get_convnet5(with_fc=False):
    structure_config = dict()
    structure_config['in_channels'] = 3
    structure_config['maxpool_idx_list'] = [2, 5]
    structure_config['layer_out_channels_dict'] = {0: 64, 1: 64, 3: 128, 4: 128}
    structure_config['layer_kernel_padding_dict'] = {0: (3, 1), 1: (3, 1), 3: (3, 1), 4: (3, 1)}
    structure_config['layer_norm_dict'] = {0: 'bn', 1: 'bn', 3: None, 4: None}
    structure_config['layer_passport_dict'] = {0: False, 1: False, 3: False, 4: False}
    structure_config['fc_dim_list'] = None
    if with_fc:
        structure_config['fc_dim_list'] = [32768, 1024]
    return ConvNetBottom(structure_config=structure_config)


def get_convnet_7(with_fc=False):
    structure_config = dict()
    structure_config['in_channels'] = 3
    structure_config['maxpool_idx_list'] = [2]
    structure_config['layer_out_channels_dict'] = {0: 64, 1: 64, 3: 128, 4: 128, 5: 128, 6: 128}
    structure_config['layer_kernel_padding_dict'] = {0: (3, 1), 1: (3, 1), 3: (3, 1), 4: (3, 1), 5: (3, 1), 6: (3, 1)}
    structure_config['layer_norm_dict'] = {0: 'bn', 1: 'bn', 3: None, 4: None, 5: None, 6: None}
    structure_config['layer_passport_dict'] = {0: False, 1: False, 3: False, 4: False, 5: False, 6: False}
    structure_config['fc_dim_list'] = None
    if with_fc:
        structure_config['fc_dim_list'] = [32768, 1024]
    return ConvNetBottom(structure_config=structure_config)


if __name__ == '__main__':
    model = get_convnet5(with_fc=False)
    x = torch.randn(10, 3, 32, 32)
    y = model(x)
    print("y shape", y.shape)
    for param in model.named_parameters():
        print("param:", param[0])
    for k in model.state_dict().keys():
        print("k:", k)
