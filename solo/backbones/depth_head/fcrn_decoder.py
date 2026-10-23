import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from solo.backbones.depth_head.blocks import *


class FCRN(nn.Module):
    '''
    FCRN: Fully Convolutional Residual Network based on a ResNet-50 architecture.

    Implementation follows architecture of FCRN in "Deeper Depth Prediction with
    Fully Convolutional Residual Networks" by Laina I. et al.
    https://github.com/georgeyiasemis/FCRN-PyTorch/tree/main
    https://chercheurs.lille.inria.fr/~valko/hp/publications/grill2020bootstrap.pdf
    '''
    def __init__(self, out_channel=1):
        super(FCRN, self).__init__()

        decoder = OrderedDict()
        # Convolution Layer
        conv2 = nn.Conv2d(in_channels=512 * 4,
                          out_channels=512 * 2,
                          kernel_size=(1, 1))
        batchnorm2 = nn.BatchNorm2d(512 * 2)

        decoder['Conv2'] = nn.Sequential(conv2, batchnorm2)

        # Decoder
        # Fast Up Projection Blocks
        decoder['UpProj1'] = FastUpProjectionBlock(in_channels=512 * 2, out_channels=512)
        decoder['UpProj2'] = FastUpProjectionBlock(in_channels=256 * 2, out_channels=256)
        decoder['UpProj3'] = FastUpProjectionBlock(in_channels=128 * 2, out_channels=128)
        decoder['UpProj4'] = FastUpProjectionBlock(in_channels=64 * 2, out_channels=64)

        # Out Convolution Layer
        conv3 = nn.Conv2d(in_channels=64,
                          out_channels=out_channel,
                          kernel_size=(3, 3),
                          padding=(1, 1))
        # decoder['Conv3'] = conv3 #nn.Sequential(conv3, nn.ReLU())
        # decoder['Conv3'] = nn.Sequential(conv3, nn.Softplus())
        decoder['Conv3'] = nn.Sequential(conv3)
        self.net = nn.Sequential(decoder)


    def forward(self, inp):
        '''
        inp: Tensor of shape (batch_size, in_channels, H, W)
        out: Tensor of shape (batch_size, 1, H', W')
        '''

        return self.net(inp)
