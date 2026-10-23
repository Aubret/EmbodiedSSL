import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class FastUpProjectionBlock(nn.Module):
    '''
    Fast Up Projection Block as proposed in "Deeper Depth Prediction with
    Fully Convolutional Residual Networks" by Laina I. et al.
    '''
    def __init__(self, in_channels, out_channels):
        super(FastUpProjectionBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First Branch
        self.convA11 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 3))
        self.convA12 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 3))
        self.convA13 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 2))
        self.convA14 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 2))

        self.reluA2 = nn.ReLU(inplace=True)
        self.batchnormA2 = nn.BatchNorm2d(out_channels)

        self.convA3 = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=(3, 3),
                                padding=(1, 1))
        self.batchnormA3 = nn.BatchNorm2d(out_channels)

        # Second Branch
        self.convB11 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 3))
        self.convB12 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 3))
        self.convB13 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(3, 2))
        self.convB14 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(2, 2))
        self.batchnormB2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W)
        out: Tensor of shape (batch_size, out_channels, H', W')
        '''

        inpA11 = F.pad(self.convA11(inp), pad=(1, 1, 1, 1))
        inpA12 = F.pad(self.convA12(inp), pad=(1, 1, 1, 0))
        inpA13 = F.pad(self.convA13(inp), pad=(1, 0, 1, 1))
        inpA14 = F.pad(self.convA14(inp), pad=(1, 0, 1, 0))
        outA1 = torch.cat((inpA11, inpA12), dim=2)
        outA2 = torch.cat((inpA13, inpA14), dim=2)
        outA = torch.cat((outA1, outA2), dim=3)
        outA = self.reluA2(self.batchnormA2(outA))
        outA = self.batchnormA3(self.convA3(outA))

        inpB11 = F.pad(self.convB11(inp), pad=(1, 1, 1, 1))
        inpB12 = F.pad(self.convB12(inp), pad=(1, 1, 1, 0))
        inpB13 = F.pad(self.convB13(inp), pad=(1, 0, 1, 1))
        inpB14 = F.pad(self.convB14(inp), pad=(1, 0, 1, 0))

        outB1 = torch.cat((inpB11, inpB12), dim=2)
        outB2 = torch.cat((inpB13, inpB14), dim=2)

        outB = torch.cat((outB1, outB2), dim=3)
        outB = self.batchnormB2(outB)

        out = self.relu(outA + outB)

        return out

class SkipConv(nn.Module):
    '''
    Residual Block.
        Output: H(x) = f(x) + g(x),
    where
        f = 1x1, 3x3, 1x1 convolutions
    and
        g(x) = x, if projection = False
        g(x) = 1x1 convolution, otherwise.
    '''
    def __init__(self, in_channels, d1, d2=None, stride=1, projection=False):
        super(SkipConv, self).__init__()

        self.in_channels = in_channels
        d2 = d2 if d2 != None else d1 * 4
        self.d1 = d1
        self.d2 = d2
        self.projection = projection

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=d1,
                              kernel_size=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(d1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=d1,
                              out_channels=d1,
                              kernel_size=(3, 3),
                              stride=(stride, stride),
                              padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(d1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=d1,
                               out_channels=d2,
                               kernel_size=(1, 1))
        self.batchnorm3 = nn.BatchNorm2d(d2)

        if projection:
            self.proj = nn.Conv2d(in_channels=in_channels,
                                  out_channels=d2,
                                  kernel_size=(1, 1),
                                  stride=(stride, stride))
            self.batchnorm_proj = nn.BatchNorm2d(d2)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W)
        out: Tensor of shape (batch_size, d2, H', W')
        '''

        if self.projection:
            res = self.batchnorm_proj(self.proj(inp))

        out = self.relu1(self.batchnorm1(self.conv1(inp)))
        out = self.relu2(self.batchnorm2(self.conv2(out)))
        out = self.batchnorm3(self.conv3(out))

        if self.projection:
            out += res
        return out

class SkipConvBlock(nn.Module):
    '''
    Repetition * num_blocks of SkipConv.
    '''

    def __init__(self, num_blocks, in_channels, d1, d2=None, stride=1):
        super(SkipConvBlock, self).__init__()

        d2 = d2 if d2 is not None else d1 * 4
        modules = OrderedDict()
        modules['Skip0'] = (SkipConv(in_channels=in_channels,
                                        d1=d1,
                                        d2=d2,
                                        stride=stride,
                                        projection=True))
        for i in range(1, num_blocks):
            modules['Proj'+str(i)] = (SkipConv(in_channels=d2,
                                      d1=d1,
                                      d2=d2))

        self.net = nn.Sequential(modules)

    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W)
        out: Tensor of shape (batch_size, d2, H', W')
        '''

        return self.net(inp)
