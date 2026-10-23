import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        if down_x.shape[2] <= x.shape[2]-8:
            down_x = torch.nn.functional.pad(down_x, pad=(0, 0, 4, 4))
        if down_x.shape[2] <= x.shape[2]-4:
            down_x = torch.nn.functional.pad(down_x, pad=(0, 0, 2, 2))
        if down_x.shape[2] < x.shape[2]:
            down_x = torch.nn.functional.pad(down_x, pad=(0, 0, 1, 0))
        if down_x.shape[2] < x.shape[2]:
            down_x = torch.nn.functional.pad(down_x, pad=(0, 0, 0, 1))
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x

#https://github.com/rawmarshmellows/pytorch-unet-resnet-50-encoder/blob/master/u_net_resnet_50_encoder.py
class UNetResnet50(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        up_blocks = []
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=64,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        # up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
        #                                             up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, layers):


        layers = [l.chunk(2) for l in layers]
        layers= [l[0] - l[1] for l in layers]
        # layers= [torch.cat(l, dim=1) for l in layers]
        layer0, layer1, layer2, layer3, layer4 = layers


        x = self.bridge(layer4)

        # x = self.up_blocks[0](x, layer4)
        x = self.up_blocks[0](x, layer3)
        x = self.up_blocks[1](x, layer2)
        x = self.up_blocks[2](x, layer1)
        x = self.up_blocks[3](x, layer0)
        output_feature_map = x
        x = self.out(x)
        return x
