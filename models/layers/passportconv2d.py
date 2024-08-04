import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class PassportBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, norm_type=None, FC_type=True):
        super(PassportBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=i, out_channels=o, kernel_size=(ks, ks), stride=(s, s), padding=pd, bias=False)
        # self.conv = nn.Conv2d(i, o, ks, s, pd, bias=False)
        self.FC_type = FC_type
        # print("self.FC_type:", self.FC_type)
        if self.FC_type:
            self.encode = nn.Linear(o, o // 4, bias=False)
            print("PP encoder:{}".format(self.encode))
            self.leakrelu = nn.LeakyReLU(inplace=True)
            self.decode = nn.Linear(o // 4, o, bias=False)
            print("PP decoder:{}".format(self.decode))
        self.key_type = 'random'
        self.weight = self.conv.weight
        self.norm_type = norm_type

        self.init_scale()
        self.init_bias()
        self.register_buffer('key', None)
        self.register_buffer('skey', None)

        if self.norm_type == 'bn':
            self.bn = nn.BatchNorm2d(o, affine=False)
        elif self.norm_type == 'gn':
            self.bn = nn.GroupNorm(o // 16, o, affine=False)
        elif self.norm_type == 'in':
            self.bn = nn.InstanceNorm2d(o, affine=False)
        else:
            self.bn = None

        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.ones_(self.scale)
        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def get_scale(self, force_passport=False):
        if self.scale is not None and not force_passport:
            return self.scale.view(1, -1, 1, 1)
        else:
            skey = self.skey
            # print("skey shape:", skey.shape)
            scalekey = self.conv(skey)
            b = scalekey.size(0)
            c = scalekey.size(1)
            scale = scalekey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
            scale = scale.mean(dim=0).view(1, c, 1, 1)
            # print("scale shape:", scale.shape)
            scale = scale.view(-1, c, 1, 1)
            if self.FC_type:
                scale = scale.view(1, c)
                scale = self.decode(self.leakrelu(self.encode(scale))).view(1, c, 1, 1)

            # print("scale shape:", scale.shape)
            return scale

    def get_bias(self, force_passport=False):
        if self.bias is not None and not force_passport:
            return self.bias.view(1, -1, 1, 1).cuda()
        else:
            key = self.key

            biaskey = self.conv(key)  # key batch always 1
            b = biaskey.size(0)
            c = biaskey.size(1)
            bias = biaskey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
            bias = bias.mean(dim=0).view(1, c, 1, 1)

            if self.FC_type:
                bias = bias.view(1, c)
                bias = self.decode(self.leakrelu(self.encode(bias))).view(1, c, 1, 1)

            return bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keyname = prefix + 'key'
        skeyname = prefix + 'skey'

        if keyname in state_dict:
            self.register_buffer('key', torch.randn(*state_dict[keyname].size()))
        if skeyname in state_dict:
            self.register_buffer('skey', torch.randn(*state_dict[skeyname].size()))

        scalename = prefix + 'scale'
        biasname = prefix + 'bias'
        if scalename in state_dict:
            self.scale = nn.Parameter(torch.randn(*state_dict[scalename].size()))

        if biasname in state_dict:
            self.bias = nn.Parameter(torch.randn(*state_dict[biasname].size()))

        super(PassportBlock, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                         missing_keys, unexpected_keys, error_msgs)

    def generate_key(self, *shape):
        global key_type

        newshape = list(shape)
        # print("shape:", shape)
        # print("newshape1:", newshape)

        newshape[0] = 1

        min = -1.0
        max = 1.0
        key = np.random.uniform(min, max, newshape)
        # print("key:", key.shape)
        return key

    def passport_selection(self, passport_candidates):
        b, c, h, w = passport_candidates.size()

        if c == 3:  # input channel
            randb = random.randint(0, b - 1)
            return passport_candidates[randb].unsqueeze(0)

        passport_candidates = passport_candidates.view(b * c, h, w)
        full = False
        flag = [False for _ in range(b * c)]
        channel = c
        passportcount = 0
        bcount = 0
        passport = []

        while not full:
            if bcount >= b:
                bcount = 0

            randc = bcount * channel + random.randint(0, channel - 1)
            while flag[randc]:
                randc = bcount * channel + random.randint(0, channel - 1)
            flag[randc] = True

            passport.append(passport_candidates[randc].unsqueeze(0).unsqueeze(0))

            passportcount += 1
            bcount += 1

            if passportcount >= channel:
                full = True

        passport = torch.cat(passport, dim=1)
        return passport

    def set_key(self, x, y=None):
        # print("x shape:", x.shape)
        n = int(x.size(0))

        if n != 1:
            x = self.passport_selection(x)
            if y is not None:
                y = self.passport_selection(y)

        # assert x.size(0) == 1, 'only batch size of 1 for key'

        # print("x shape:", x.shape)
        self.register_buffer('key', x)

        # assert y is not None and y.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('skey', y)

    def forward(self, x, force_passport=True):
        # print("x.shape:", x.shape)
        if self.key is None:
            self.set_key(torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device),
                         torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device))
            print('set key is done.')
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.bn(x)

        print("[DEBUG] PassportBlock.forward force_passport:{}".format(force_passport))
        scale = self.get_scale(force_passport)
        bias = self.get_bias(force_passport)
        print("[DEBUG] scale.shape:", scale.shape)
        print("[DEBUG] bias.shape:", bias.shape)
        x = scale * x + bias
        x = self.relu(x)
        return x, scale, bias
