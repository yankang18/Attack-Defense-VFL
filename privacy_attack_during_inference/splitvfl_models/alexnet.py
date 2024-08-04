import torch
import torch.nn as nn

from privacy_attack_during_inference.splitvfl_models.layers.conv2d import ConvBlock
from privacy_attack_during_inference.splitvfl_models.layers.passportconv2d_nonsignloss import PassportBlock


class AlexNet(nn.Module):

    def __init__(self, in_channels, passport_pos):
        super(AlexNet, self).__init__()

        inp = in_channels

        maxpoolidx = [1, 3, 7]

        layers = []

        # oups = {
        #     0: 64,
        #     2: 192,
        #     4: 384,
        #     5: 256,
        #     6: 256
        # }
        # kp = {
        #     0: (5, 2),
        #     2: (5, 2),
        #     4: (3, 1),
        #     5: (3, 1),
        #     6: (3, 1)
        # }
        oups = {
            0: 32,
            2: 64,
            4: 128,
            5: 256,
            6: 128
        }
        kp = {
            0: (3, 2),
            2: (3, 2),
            4: (3, 1),
            5: (2, 1),
            6: (2, 1)
        }

        norm_pos = {0: 'bn', 2: 'bn', 4: 'bn', 5: None, 6: None}
        for layeridx in range(8):
            if layeridx in maxpoolidx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layeridx][0]
                p = kp[layeridx][1]
                if passport_pos[str(layeridx)]:
                    print(f"[DEBUG] Alexnet layer:{layeridx} applies passport.")
                    layers.append(PassportBlock(inp, oups[layeridx], k, 1, p, norm_pos[layeridx]))
                else:
                    print(f"[DEBUG] Alexnet layer:{layeridx} does not apply passport.")
                    layers.append(ConvBlock(inp, oups[layeridx], k, 1, p, norm_pos[layeridx]))

                inp = oups[layeridx]

        self.features = nn.Sequential(*layers)

        # self.fc = nn.Linear(4 * 4 * 256, num_classes)

        self.scale = None
        self.bias = None

    # def set_intermediate_keys(self, pretrained_model, x, y=None):
    #     with torch.no_grad():
    #         for pretrained_layer, self_layer in zip(pretrained_model.features, self.features):
    #             if isinstance(self_layer, PassportBlock):
    #                 self_layer.set_key(x, y)
    #
    #             x = pretrained_layer(x)
    #             if y is not None:
    #                 y = pretrained_layer(y)

    def forward(self, x, force_passport=True):
        for m in self.features:
            if isinstance(m, PassportBlock):
                x, self.scale, self.bias = m(x, force_passport)
            else:
                x = m(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x


if __name__ == '__main__':
    passport_pos = {'0': False, '2': False, '4': False, '5': False, '6': False}
    model = AlexNet(3, passport_pos=passport_pos)
    x = torch.randn(10, 3, 32, 32)
    y = model(x)
    print("y shape", y.shape)
    for param in model.named_parameters():
        print("param:", param[0])
    for k in model.state_dict().keys():
        print("k:", k)
