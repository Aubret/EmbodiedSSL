from torch import nn

from .hmax import HMAX
from pathlib import Path

from .prednet import PredNet, load_weight
from .scattering import Scattering


def load_hmax(**kwargs):
    # p = Path.cwd() / 'solo/backbones/bioinspired/universal_patch_set.mat'
    p = Path.cwd().parent / 'solo-learn/solo/backbones/bioinspired/universal_patch_set.mat'
    return HMAX(str(p))

def load_scattering(**kwargs):
    return Scattering(**kwargs)

def load_prednet(average=True, **kwargs):
    n_channels = 3

    stack_sizes = (n_channels, 48, 96, 192)
    # stack_sizes = (n_channels, 48, 224, 224)
    R_stack_sizes = stack_sizes
    A_filter_sizes = (3, 3, 3)
    Ahat_filter_sizes = (3, 3, 3, 3)
    R_filter_sizes = (3, 3, 3, 3)
    prednet = PredNet(stack_sizes, R_stack_sizes, A_filter_sizes, Ahat_filter_sizes, R_filter_sizes,
                      output_mode='R3', return_sequences=True, average=average, **kwargs)
    prednet.num_features = 192
    prednet.load_state_dict(load_weight(Path.cwd().parent / 'solo-learn/solo/backbones/bioinspired/prednet_kitti_weights.hdf5'))
    conv_layers = {k: v for k, v in prednet.conv_layers.items()}
    del prednet.conv_layers
    prednet.add_module("conv_layers", nn.ModuleDict(conv_layers))

    return prednet