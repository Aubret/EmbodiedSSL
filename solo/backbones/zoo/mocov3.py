import math
from collections import OrderedDict

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from torch.utils import model_zoo
from torchvision.models import resnet50
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import PatchEmbed

__all__ = [
    'vit_small',
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
]

# Extracted from https://github.com/bethgelab/model-vs-human/

CLASSIFIER_WEIGHTS = {
    "InsDis": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/InsDis_classifier.pth",
    "CMC": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/CMC_classifier.pth",
    "MoCo": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/MoCo_classifier.pth",
    "PIRL": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/PIRL_classifier.pth",
    "MoCoV2": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/MoCov2_classifier.pth",
    "InfoMin": "https://github.com/bethgelab/model-vs-human/releases/download/v0.2/InfoMin_classifier.pth"
}


PYTORCH_HUB = "kantharajucn/PyContrast:push_pretrained_models_to_pytorch_hub"




class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_prefix_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, **kwargs):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model



def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def build_classifier(model_name, classes=1000):
    """Building a linear layer"""

    n_class = classes

    n_feat = 2048

    classifier = nn.Linear(n_feat, n_class)
    checkpoint = torch.hub.load_state_dict_from_url(CLASSIFIER_WEIGHTS.get(model_name), map_location='cpu')

    state_dict = OrderedDict()
    for k, v in checkpoint["classifier"].items():
        k = k.replace('module.', '')
        state_dict[k] = v
    classifier.load_state_dict(state_dict)

    return classifier


def InsDis(pretrained=False, **kwargs):
    """
    Unsupervised Feature Learning via Non-parameteric Instance Discrimination
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "InsDis"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)

    return model, classifier


def CMC(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "CMC"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def MoCo(pretrained=False, **kwargs):
    """
    Contrastive Multiview Coding
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "MoCo"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier

def MoCoV3(**kwargs):
    """
    Improved Baselines with Momentum Contrastive Learning
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = resnet50()
    model.fc = nn.Identity()
    checkpoint = model_zoo.load_url("https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar")

    print(checkpoint["state_dict"].keys())
    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if 'module.base_encoder' in k and ".fc." not in k:
            new_state_dict[k.replace("module.base_encoder.","")] = v
    model.load_state_dict(new_state_dict)
    return model

def MoCoV3Projector(epochs=1000):
    """
    Improved Baselines with Momentum Contrastive Learning
    :param pretrained:
    :param kwargs:
    :return:
    """
    model = resnet50()
    #avgpool, fc.0, fc.3
    def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU())
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)



    model.fc = nn.Identity()
    checkpoint = model_zoo.load_url(f"https://dl.fbaipublicfiles.com/moco-v3/r-50-{epochs}ep/r-50-{epochs}ep.pth.tar")

    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if 'module.base_encoder' in k :
            new_state_dict[k.replace("module.base_encoder.","")] = v
    model.fc = _build_mlp(2, 2048, 4096, 256)
    # model. = _build_mlp(2, 2048, 4096, 256)
    model.load_state_dict(new_state_dict)
    return model

def MoCoV3ProjectorVit(epochs=300, size="base"):
    if size == "base":
        model = vit_base()
    else:
        model = vit_small()
    #avgpool, fc.0, fc.3
    def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU())
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)



    model.fc = nn.Identity()
    print(f"https://dl.fbaipublicfiles.com/moco-v3/vit-{size[0]}-{epochs}ep/vit-{size[0]}-{epochs}ep.pth.tar")
    checkpoint = model_zoo.load_url(f"https://dl.fbaipublicfiles.com/moco-v3/vit-{size[0]}-{epochs}ep/vit-{size[0]}-{epochs}ep.pth.tar")

    new_state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if 'module.base_encoder' in k :
            new_state_dict[k.replace("module.base_encoder.","")] = v
    model.head = _build_mlp(3, 768 if size =="base" else 384, 4096, 256)
    # model. = _build_mlp(2, 2048, 4096, 256)
    model.load_state_dict(new_state_dict)
    return model


def MoCoV3All(**kwargs):
    def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        layers = []

        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            layers.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                layers.append(nn.BatchNorm1d(dim2))
                layers.append(nn.ReLU(inplace=True))
            elif last_bn:
                layers.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*layers)

    class MoCoV3Model(nn.Module):
        def __init__(self):
            super().__init__()

            # backbone
            backbone = resnet50()
            backbone.fc = nn.Identity()
            self.backbone = backbone

            # projector
            self.projector = _build_mlp(
                num_layers=2,
                input_dim=2048,
                mlp_dim=4096,
                output_dim=256
            )

            # predictor
            self.predictor = _build_mlp(
                num_layers=2,
                input_dim=256,
                mlp_dim=4096,
                output_dim=256,
                last_bn=False
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.projector(x)
            x = self.predictor(x)
            return x

    model = MoCoV3Model()

    checkpoint = model_zoo.load_url(
        "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
    )

    new_state_dict = {}

    for k, v in checkpoint["state_dict"].items():

        if k.startswith("module.base_encoder."):
            new_k = k.replace("module.base_encoder.", "")

            if new_k.startswith("fc."):
                new_k = new_k.replace("fc.", "projector.")
            else:
                new_k = "backbone." + new_k

            new_state_dict[new_k] = v

        elif k.startswith("module.predictor."):
            new_k = k.replace(
                "module.predictor.",
                "predictor."
            )
            new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(
        new_state_dict,
        strict=False
    )

    print("Missing:", missing)
    print("Unexpected:", unexpected)

    return model

def MoCoV2(pretrained=False, **kwargs):
    """
    Improved Baselines with Momentum Contrastive Learning
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "MoCoV2"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def PIRL(pretrained=False, **kwargs):
    """
    Self-Supervised Learning of Pretext-Invariant Representations
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "PIRL"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier


def InfoMin(pretrained=False, **kwargs):
    """
    What Makes for Good Views for Contrastive Learning?
    :param pretrained:
    :param kwargs:
    :return:
    """
    name = "InfoMin"
    model = torch.hub.load(PYTORCH_HUB,
                           name,
                           pretrained=pretrained)
    classifier = build_classifier(name)
    return model, classifier
