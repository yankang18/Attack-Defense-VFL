import torch.nn as nn
import torch.nn.init as init


class ConvBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, norm_type='bn', relu=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=i, out_channels=o, kernel_size=(ks, ks), stride=(s, s), padding=pd, bias=True)
        self.norm_type = norm_type

        if self.norm_type == 'bn':
            self.bn = nn.BatchNorm2d(o)
        elif self.norm_type == 'gn':
            self.bn = nn.GroupNorm(o // 16, o)
        elif self.norm_type == 'in':
            self.bn = nn.InstanceNorm2d(o)
        else:
            self.bn = None

        if relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, scales=None, biases=None):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.bn(x)
        # print("scales:", scales)
        # print("biases:", biases)
        if scales is not None and biases is not None:
            # print("convent forward")
            x = scales[-1] * x + biases[-1]

        if self.relu is not None:
            x = self.relu(x)
        return x


# class ConvBlock(nn.Module):
#     def __init__(self, i, o, ks=3, s=1, pd=1, norm_type='bn', relu=True):
#         super().__init__()
#
#         self.conv = nn.Conv2d(i, o, ks, s, pd, bias=True)
#         self.norm_type = norm_type
#
#         if self.norm_type == 'bn':
#             self.bn = nn.BatchNorm2d(o)
#         elif self.norm_type == 'gn':
#             self.bn = nn.GroupNorm(o // 16, o)
#         elif self.norm_type == 'in':
#             self.bn = nn.InstanceNorm2d(o)
#         else:
#             self.bn = None
#
#         if relu:
#             self.relu = nn.ReLU(inplace=True)
#             # self.relu = nn.Sigmoid()
#             # self.pool = nn.AvgPool2d(2, 2)
#         else:
#             self.relu = None
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
#
#     def forward(self, x, scales=None, biases=None):
#         x = self.conv(x)
#         if self.norm_type is not None:
#             x = self.bn(x)
#         # print("scales:", scales)
#         # print("biases:", biases)
#         if scales is not None and biases is not None:
#             # print("convent forward")
#             x = scales[-1] * x + biases[-1]
#
#         if self.relu is not None:
#             x = self.relu(x)
#             # x = self.pool(x)
#         return x