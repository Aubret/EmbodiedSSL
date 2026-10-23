from functools import partial

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


from .registry import register_model


def model_pytorch(model_name, *args):
    import torchvision.models as zoomodels
    model = zoomodels.__dict__[model_name](pretrained=True)
    return model

@register_model()
def dinov2_small(**kwargs):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.head = nn.Identity()
    return model, model.embed_dim

@register_model()
def dinov2_base(**kwargs):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model.head = nn.Identity()
    return model, model.embed_dim


@register_model()
def sup_rn50(**kwargs):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    return model, 2048

@register_model()
def byol_rn50_proj():
    from .byol import ByolProjector
    model = ByolProjector("rn50")
    model.backbone.fc = nn.Identity()
    return model, 2048

@register_model()
def mocov3_rn50_proj():
    from .mocov3 import MoCoV3Projector
    model = MoCoV3Projector()
    return model, 2048

@register_model()
def mocov3_vit_proj_base():
    from .mocov3 import MoCoV3ProjectorVit
    model = MoCoV3ProjectorVit(300, "base")
    return model, 768

@register_model()
def mocov3_vit_proj_small():
    from .mocov3 import MoCoV3ProjectorVit
    model = MoCoV3ProjectorVit(300, "small")
    return model, 384

@register_model()
def vicreg_rn50x1():
    from solo.backbones.zoo.vicreg import VICReg_proj
    return VICReg_proj(), 2048

@register_model()
def dino_base_8():
    from solo.backbones.zoo import dino
    return dino.dino_base_8(), 768

@register_model()
def dino_rn50():
    from solo.backbones.zoo import dino
    return dino.dino_rn50(), 2048

@register_model()
def dino_small_16():
    from solo.backbones.zoo import dino
    return dino.dino_small_16(), 384

@register_model()
def dino_small_8():
    from solo.backbones.zoo import dino
    return dino.dino_small_8(), 384

@register_model()
def dino_base_16():
    from solo.backbones.zoo import dino
    return dino.dino_base_16(), 768