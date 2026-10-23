import torch
import torch.nn as nn
import torch.nn.functional as F

def make_dilated_layer4(layer4):
    """
    Modify ResNet layer4 to have:
    - stride = 1
    - dilation = 2
    """
    for name, module in layer4.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (3, 3):
                module.dilation = (2, 2)
                module.padding = (2, 2)
                module.stride = (1, 1)
        if "downsample.0" in name:
            module.stride = (1, 1)
    return layer4

def segmentation_head(num_classes, in_channels=2048):
    return nn.Sequential(
        nn.Conv2d(in_channels, 256, kernel_size=3, padding=6, dilation=6, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=6, dilation=6, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, num_classes, kernel_size=1)
    )