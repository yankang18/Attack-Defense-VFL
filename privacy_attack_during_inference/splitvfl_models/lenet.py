from collections import OrderedDict

import torch
import torch.nn as nn

from privacy_attack_during_inference.splitvfl_models.layers.conv2d import ConvBlock
from privacy_attack_during_inference.splitvfl_models.layers.passportconv2d_nonsignloss import PassportBlock


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class LeNet(nn.Module):

    def __init__(self, in_channels, passport_pos):
        super(LeNet, self).__init__()

        inp = in_channels
        norm_type_dict = {0: 'bn', 1: None, 2: None}
        # norm_type = None

        layers = []
        # output channel
        oups = {
            0: 12,
            1: 12,
            # 2: 12,
        }

        # kernel size, stride
        ks = {
            0: (5, 2),
            1: (5, 2),
            # 2: (5, 1),
        }

        self.passport_pos = passport_pos
        for layeridx in range(2):
            k = ks[layeridx][0]
            s = ks[layeridx][1]
            if self.passport_pos[layeridx]:
                print(f"[DEBUG] LeNet layer:{layeridx} applies passport.")
                layers.append(PassportBlock(inp, oups[layeridx], k, s, 2, norm_type_dict[layeridx]))
            else:
                print(f"[DEBUG] LeNet layer:{layeridx} does not apply passport.")
                layers.append(ConvBlock(inp, oups[layeridx], k, s, 2, norm_type_dict[layeridx]))

            inp = oups[layeridx]

        self.features = nn.Sequential(*layers)

        self.scales = None
        self.biases = None

    def get_number_passports(self):
        return "".join(["1" if val else "0" for _, val in self.passport_pos.items()])

    def forward(self, x, force_passport=True):
        self.scales = list()
        self.biases = list()
        for m in self.features:
            if isinstance(m, PassportBlock):
                x, scale, bias = m(x, force_passport)
                # print("scale:", scale.shape)
                # print("bias:", bias.shape)
                self.scales.append(scale)
                self.biases.append(bias)
            else:
                x = m(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetWithAct(nn.Module):

    def __init__(self, in_channels, passport_pos):
        super(LeNetWithAct, self).__init__()

        inp = in_channels
        norm_type_dict = {0: 'bn', 1: None, 2: None}

        # output channel
        oups = {
            0: 12,
            1: 12,
            # 2: 12,
        }

        # kernel size, stride
        ks = {
            0: (5, 2),
            1: (5, 2),
            # 2: (5, 1),
        }

        self.passport_pos = passport_pos

        layers = []
        for layeridx in range(2):
            k = ks[layeridx][0]
            s = ks[layeridx][1]
            if self.passport_pos[layeridx]:
                print(f"[DEBUG] LeNet layer:{layeridx} applies passport.")
                layers.append(PassportBlock(inp, oups[layeridx], k, s, 2, norm_type_dict[layeridx]))
            else:
                print(f"[DEBUG] LeNet layer:{layeridx} does not apply passport.")
                layers.append(ConvBlock(inp, oups[layeridx], k, s, 2, norm_type_dict[layeridx]))

            inp = oups[layeridx]

        self.features = nn.Sequential(*layers)
        self.features.append(nn.Flatten())

        self.scales = None
        self.biases = None

    def get_number_passports(self):
        return "".join(["1" if val else "0" for _, val in self.passport_pos.items()])

    def forward(self, x, force_passport=True):
        self.scales = list()
        self.biases = list()
        for m in self.features:
            if isinstance(m, PassportBlock):
                x, scale, bias = m(x, force_passport)
                # print("scale:", scale.shape)
                # print("bias:", bias.shape)
                self.scales.append(scale)
                self.biases.append(bias)
            else:
                x = m(x)
        return x

# def forward(self, x):
#     # print("lenet input shape:", x.shape)
#
#     for m in self.features:
#         x = m(x)
#     # x = x.view(x.size(0), -1)
#     # x = self.fc(x)
#
#     # print("lenet output shape:", x.shape)
#     return x


class LeNetConvBottom(nn.Module):

    def __init__(self, structure_config):
        super(LeNetConvBottom, self).__init__()

        inp = structure_config['in_channels']
        maxpool_idx_list = structure_config['layer_maxpool_list']
        oups = structure_config['layer_out_channels_dict']
        ks = structure_config['layer_kernel_stride_dict']
        norm_idx_dict = structure_config['layer_norm_dict']
        self.passport_idx_dict = structure_config['layer_passport_dict']
        self.fc_dim_list = structure_config['fc_dim_list']

        num_layer = len(ks) + len(maxpool_idx_list)

        layers = []
        for layer_idx in range(num_layer):
            if layer_idx in maxpool_idx_list:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = ks[layer_idx][0]
                s = ks[layer_idx][1]
                if self.passport_idx_dict[layer_idx]:
                    print(f"[DEBUG] LeNet layer:{layer_idx} applies passport.")
                    layers.append(
                        PassportBlock(inp, oups[layer_idx], ks=k, s=s, pd=0, norm_type=norm_idx_dict[layer_idx]))
                else:
                    print(f"[DEBUG] LeNet layer:{layer_idx} does not apply passport.")
                    layers.append(ConvBlock(inp, oups[layer_idx], ks=k, s=s, pd=0, norm_type=norm_idx_dict[layer_idx]))

                inp = oups[layer_idx]

        # print("[DEBUG] network architecture: \n {}".format(layers))

        self.features = nn.Sequential(*layers)

        self.scales = None
        self.biases = None

        if self.fc_dim_list is not None:
            self.fc = nn.Linear(1024, 256)

    def get_number_passports(self):
        dict1 = OrderedDict(sorted(self.passport_idx_dict.items()))
        return "".join(["1" if val else "0" for _, val in dict1.items()])

    def forward(self, x, force_passport=True):
        self.scales = list()
        self.biases = list()
        for m in self.features:
            if isinstance(m, PassportBlock):
                x, scale, bias = m(x, force_passport)
                # print("scale:", scale.shape)
                # print("bias:", bias.shape)
                self.scales.append(scale)
                self.biases.append(bias)
            else:
                x = m(x)

        x = x.view(x.size(0), -1)
        if self.fc_dim_list is not None:
            x = self.fc(x)
        return x


def get_lenet(struct_config=None):
    struct_config = {} if struct_config is None else struct_config

    # with_fc = struct_config.get("with_fc")
    # with_fc = False if with_fc is None else True

    layer_maxpool_list = struct_config.get("layer_maxpool_list")

    # === default value for structure config ===
    structure_config = dict()
    structure_config['in_channels'] = 1
    structure_config['layer_maxpool_list'] = [1] if layer_maxpool_list is None else layer_maxpool_list
    structure_config['layer_out_channels_dict'] = {0: 8, 2: 16}
    structure_config['layer_kernel_stride_dict'] = {0: (5, 1), 2: (5, 1)}
    structure_config['layer_norm_dict'] = {0: 'bn', 2: None}
    structure_config['layer_passport_dict'] = {0: False, 2: False}
    structure_config['fc_dim_list'] = None
    # if with_fc:
    #     structure_config['fc_dim_list']
    #     structure_config['fc_dim_list'] = [1024, 256]

    model = LeNetConvBottom(structure_config=structure_config)
    print("[DEBUG] LeNet model struct: \n {}".format(model))
    return model


if __name__ == '__main__':

    model = LeNet(in_channels=1, passport_pos={0: False, 1: False})
    # model = get_lenet()
    print(model)
    x = torch.randn(10, 1, 14, 28)
    y = model(x)
    print("y shape", y.shape)

    print("-" * 100)

    model_2 = get_lenet(struct_config=None)
    # print(model_2)
    y = model_2(x)
    print("y shape", y.shape)

    # for param in model.named_parameters():
    #     print("param:", param[0])
    # for k in model.state_dict().keys():
    #     print("k:", k)
