import math
from collections import OrderedDict

import torch.nn.functional
from torch import nn


HASH = '1d3f7974'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=3, skip=True, **kwargs):
        super().__init__()
        self.skip_enable = skip
        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


        if self.skip_enable:
            self.skip = nn.Conv2d(out_channels, out_channels,
                                  kernel_size=1, stride=2, bias=False)
            self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))


        self.outputs = OrderedDict()
    def forward(self, inp):
        if isinstance(inp, dict):
            inp = next(reversed(inp.values()))

        self.outputs = OrderedDict()
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                if self.skip_enable:
                    skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                if self.skip_enable:
                    skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            if self.skip_enable:
                x += skip
            x = self.nonlin3(x)
            output = self.output(x)
            self.outputs[f"out{t+1}"] = torch.clone(output)
        return self.outputs


class CORblock_S_unfold(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, skip=True, **kwargs):
        super().__init__()
        self.skip_enable = skip
        self.times = 3
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


        if self.skip_enable:
            self.skip = nn.Conv2d(out_channels, out_channels,kernel_size=1, stride=2, bias=False)
            self.norm_skip = nn.BatchNorm2d(out_channels)


        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,kernel_size=1, bias=False)
        self.conv11 = nn.Conv2d(out_channels, out_channels * self.scale,kernel_size=1, bias=False)
        self.conv12 = nn.Conv2d(out_channels, out_channels * self.scale,kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin11 = nn.ReLU(inplace=True)
        self.nonlin12 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,kernel_size=3, stride=2, padding=1, bias=False)
        self.conv21 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,kernel_size=3, stride=1, padding=1, bias=False)
        self.conv22 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,kernel_size=3, stride=1, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.nonlin21 = nn.ReLU(inplace=True)
        self.nonlin22 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,kernel_size=1, bias=False)
        self.conv31 = nn.Conv2d(out_channels * self.scale, out_channels,kernel_size=1, bias=False)
        self.conv32 = nn.Conv2d(out_channels * self.scale, out_channels,kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)
        self.nonlin31 = nn.ReLU(inplace=True)
        self.nonlin32 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(3):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))


        self.outputs = OrderedDict()
    def forward(self, inp):
        if isinstance(inp, dict):
            inp = next(reversed(inp.values()))

        self.outputs = OrderedDict()
        x = self.conv_input(inp)


        if self.skip_enable:
            skip = self.norm_skip(self.skip(x))

        x = self.conv1(x)
        x = getattr(self, f'norm1_0')(x)
        x = self.nonlin1(x)

        x = self.conv2(x)
        x = getattr(self, f'norm2_0')(x)
        x = self.nonlin2(x)

        x = self.conv3(x)
        x = getattr(self, f'norm3_0')(x)

        if self.skip_enable:
            x += skip
        x = self.nonlin3(x)
        output = self.output(x)
        self.outputs[f"out1"] = torch.clone(output)
        #######################################################"""
        if self.skip_enable:
            skip = x

        x = self.conv11(x)
        x = getattr(self, f'norm1_1')(x)
        x = self.nonlin11(x)

        x = self.conv21(x)
        x = getattr(self, f'norm2_1')(x)
        x = self.nonlin21(x)

        x = self.conv31(x)
        x = getattr(self, f'norm3_1')(x)

        if self.skip_enable:
            x += skip
        x = self.nonlin31(x)
        output = self.output(x)
        self.outputs[f"out2"] = torch.clone(output)
        #############################################
        if self.skip_enable:
            skip = x

        x = self.conv12(x)
        x = getattr(self, f'norm1_2')(x)
        x = self.nonlin12(x)

        x = self.conv22(x)
        x = getattr(self, f'norm2_2')(x)
        x = self.nonlin22(x)

        x = self.conv32(x)
        x = getattr(self, f'norm3_2')(x)

        if self.skip_enable:
            x += skip
        x = self.nonlin32(x)

        output = self.output(x)
        self.outputs[f"out3"] = torch.clone(output)
        return self.outputs

def distill(model):
    #v4_outs = model["V4"].outputs
    it_outs = model.IT.outputs
    features =[0]*len(it_outs)
    for k, it_out in it_outs.items():
        out = nn.functional.adaptive_avg_pool2d(it_out, 1)
        # features[it_out] = out.view(out.size(0), -1)
        features[int(k[-1])-1] = out.view(out.size(0), -1)
    distances = 0
    for i in range(len(it_outs)-1):
        fi = torch.nn.functional.cosine_similarity(features[i].unsqueeze(1), features[i].unsqueeze(0), 2)
        fi1 = torch.nn.functional.cosine_similarity(features[i+1].unsqueeze(1), features[i+1].unsqueeze(0), 2)
        distances += torch.norm(fi-fi1.detach()).mean()
    return distances / (len(it_outs)-1)

def distill_first(model):
    #v4_outs = model["V4"].outputs
    it_outs = model.IT.outputs
    features =[0]*len(it_outs)
    for k, it_out in it_outs.items():
        out = nn.functional.adaptive_avg_pool2d(it_out, 1)
        # features[it_out] = out.view(out.size(0), -1)
        features[int(k[-1])-1] = out.view(out.size(0), -1)
    distances = 0
    i = 0

    return distances / (len(it_outs)-1)

def distill_cos(model):
    #v4_outs = model["V4"].outputs
    it_outs = model.IT.outputs
    features =[0]*len(it_outs)
    for k, it_out in it_outs.items():
        out = nn.functional.adaptive_avg_pool2d(it_out, 1)
        # features[it_out] = out.view(out.size(0), -1)
        features[int(k[-1])-1] = out.view(out.size(0), -1)
    distances = 0
    for i in range(len(it_outs)-1):
        distances += - torch.nn.functional.cosine_similarity(features[i], features[i+1], 1)
    return distances / (len(it_outs)-1)


def distill_full(model):
    #v4_outs = model["V4"].outputs
    it_outs = model.IT.outputs
    features =[0]*len(it_outs)
    for k, it_out in it_outs.items():
        assert it_out.shape[1] == 512
        features[int(k[-1])-1] = it_out.flatten(-2)
    distances = 0
    for i in range(len(it_outs)-1):
        for k in range(features[i].shape[-1]):
            fi = torch.nn.functional.cosine_similarity(features[i][:,:,k].unsqueeze(1), features[i][:,:,k].unsqueeze(0), 2)
            fi1 = torch.nn.functional.cosine_similarity(features[i+1][:,:,k].unsqueeze(1), features[i+1][:,:,k].unsqueeze(0), 2)
            distances += torch.norm(fi-fi1.detach()).mean()
    return distances / ((len(it_outs)-1) * features[0].shape[-1])


def CORnet_S(times_first=False, **kwargs):
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=1 if times_first else 2)),
        ('V4', CORblock_S(128, 256, times=1 if times_first else 3)), #4
        ('IT', CORblock_S(256, 512, **kwargs)), #2
        # ('decoder', nn.Sequential(OrderedDict([
        #     ('avgpool', nn.AdaptiveAvgPool2d(1)),
        #     ('flatten', Flatten()),
        # ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    model.num_features=512
    return model

def CORnet_S_unfold(**kwargs):
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=3)), #4
        ('IT', CORblock_S_unfold(256, 512, **kwargs)), #2
        # ('decoder', nn.Sequential(OrderedDict([
        #     ('avgpool', nn.AdaptiveAvgPool2d(1)),
        #     ('flatten', Flatten()),
        # ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    model.num_features=512
    return model