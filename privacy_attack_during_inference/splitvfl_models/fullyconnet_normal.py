import torch.nn as nn

from privacy_attack_during_inference.splitvfl_models.layers.conv2d import ConvBlock
from privacy_attack_during_inference.splitvfl_models.layers.passportconv2d_nonsignloss import PassportBlock


class FullyConnNet(nn.Module):

    def __init__(self, in_channels, num_classes, passport_pos):
        super(FullyConnNet, self).__init__()

        layers = []

        norm_type = 'bn'
        inp = in_channels

        # output channel
        oups = {
            0: 12,
            1: 12,
            2: 12,
        }
        # oups = {
        #     0: 64,
        #     1: 32,
        #     2: 16,
        # }

        # kernel size, stride
        ks = {
            0: (3, 2),
            1: (3, 2),
            2: (3, 1),
        }

        self.passport_pos = passport_pos
        for layeridx in range(2):
            k = ks[layeridx][0]
            s = ks[layeridx][1]
            if self.passport_pos[str(layeridx)]:
                print(f"[DEBUG] FullyConnNet layer:{layeridx} applies passport.")
                layers.append(PassportBlock(inp, oups[layeridx], k, s, 2, norm_type))
            else:
                print(f"[DEBUG] FullyConnNet layer:{layeridx} does not apply passport.")
                layers.append(ConvBlock(inp, oups[layeridx], k, s, 2, norm_type))

            inp = oups[layeridx]

        layers.append(nn.Flatten())
        # 180 for two party bottom model, while 144 for passive bottom model only.
        layers.append(nn.Linear(144, num_classes))
        self.features = nn.Sequential(*layers)

        self.scale = None
        self.bias = None

    def get_number_passports(self):
        # count = 0
        # for key, val in self.passport_pos.items():
        #     if val:
        #         count += 1
        # return count
        return "".join(["1" if val else "0" for _, val in self.passport_pos.items()])

    def forward(self, x, scales=None, biases=None, force_passport=True):
        if scales is not None and len(scales) == 0:
            scales = None
        if biases is not None and len(biases) == 0:
            biases = None

        # print("scales len:", len(scales))
        # print("biases len:", len(biases))

        for m in self.features:
            if isinstance(m, PassportBlock):
                x, self.scale, self.bias = m(x, force_passport)
            elif isinstance(m, ConvBlock):
                x = m(x, scales=scales, biases=biases)
                scales = None
                biases = None
            else:
                x = m(x)

        return x


class LeNetClassifier(nn.Module):

    def __init__(self, dims, num_classes=10, act="sigmoid", passport_pos=None):
        super(LeNetClassifier, self).__init__()

        self.passport_pos = passport_pos

        if act == "sigmoid":
            act_func = nn.Sigmoid()
        elif act == "relu":
            act_func = nn.ReLU(True)
        else:
            raise Exception(f"does not support activation {act}")

        self.classifier = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(672, 120),
            # act_func,
            # nn.Linear(120, 84),
            # act_func,
            # nn.Linear(84, num_classes),
            nn.Linear(dims[0], dims[1]),
            act_func,
            nn.Linear(dims[1], num_classes),
        )

    def get_number_passports(self):
        return "".join(["1" if val else "0" for _, val in self.passport_pos.items()])

    def forward(self, x, scales=None, biases=None, force_passport=True):
        x = self.classifier(x)
        return x


class InteractiveModel(nn.Module):

    def __init__(self, dims):
        super(InteractiveModel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(dims[0], dims[1])
            # nn.Linear(336, 120)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

# if __name__ == '__main__':
#     passport_pos = {'0': True, '1': False, '2': False}
#     # norm_pos = {0: 'bn', 2: 'bn', 4: 'bn', 5: None, 6: None}
#     model = LeNetPassport(3, 10, passport_pos)
#     x = torch.randn(10, 3, 32, 32)
#     y = model(x)
#     for param in model.named_parameters():
#         print(param[0])
#     for k in model.state_dict().keys():
#         print(k)
#         print(torch.mean)
