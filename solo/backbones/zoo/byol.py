import os.path
import re
import urllib.request

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50

model_urls = {
    'rn50': 'https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl',
    # 'rn200x4': 'https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res200x2.pkl',
}

def load_checkpoint(checkpoint_path):
    import dill
    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint_data = dill.load(checkpoint_file)
        print('=> loading checkpoint from {}, saved at step {}'.format(
            checkpoint_path, checkpoint_data['step']
        ))
        return checkpoint_data


def convert_mlp_weights(weights, bn_states):
    projector_sd = {}
    predictor_sd = {}

    for k, v in weights.items():

        if "projector" not in k and "predictor" not in k:
            continue
        target_dict = projector_sd if "projector" in k else predictor_sd

        if k == "projector/linear":
            prefix = "net.0"
        elif "linear_1" in k:
            prefix = "net.3"
        else:
            continue

        for p_k, p_v in v.items():
            tensor = torch.from_numpy(p_v)

            if p_k == "w":
                target_dict[f"{prefix}.weight"] = tensor.permute(1, 0)
            elif p_k == "b":
                target_dict[f"{prefix}.bias"] = tensor

    # batchnorm stats + params
    for k, v in weights.items():

        if "batch_norm" not in k:
            continue

        target_dict = projector_sd if "projector" in k else predictor_sd

        for p_k, p_v in v.items():
            tensor = torch.from_numpy(p_v).squeeze()

            if p_k == "scale":
                target_dict["net.1.weight"] = tensor
            elif p_k == "offset":
                target_dict["net.1.bias"] = tensor

    for k, v in bn_states.items():

        if "batch_norm" not in k:
            continue

        target_dict = projector_sd if "projector" in k else predictor_sd

        target_dict["net.1.running_mean"] = torch.from_numpy(
            v["average"]
        ).squeeze()

        target_dict["net.1.running_var"] = torch.from_numpy(
            v["hidden"]
        ).squeeze()

    return projector_sd, predictor_sd

def ByolProjector(backbone):
    model = resnet50()
    #backbone.avgpool,projector.net.0,projector.net.3
    class BYOLProjector(nn.Module):
        def __init__(
                self,
                in_dim=2048,
                hidden_dim=4096,
                out_dim=256
        ):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=True),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_dim, out_dim, bias=False),
            )

        def forward(self, x):
            return self.net(x)

    class full_model(nn.Module):
        def __init__(self, backbone, projector):
            super().__init__()
            self.backbone = backbone
            self.projector = projector
        def forward(self, x):
            return self.projector(self.backbone(x))

    model = full_model(model, BYOLProjector())
    home = os.environ['HOME']


    if os.path.exists(f"{home}/.cache/torch/hub/checkpoints/byol_r50_full.pth.tar"):
        ckpt = torch.load(f"{home}/.cache/torch/hub/checkpoints/byol_r50_full.pth.tar")
        model.backbone.load_state_dict(ckpt["backbone"])
        model.projector.load_state_dict(ckpt["projector"])
        return model

    # ckpt = model_zoo.load_url(model_urls[backbone])
    # gdown.download(model_urls[backbone])
    if not os.path.exists(f"{home}/.cache/torch/hub/checkpoints/"):
        os.makedirs(f"{home}/.cache/torch/hub/checkpoints/")
    if not os.path.exists(f"{home}/.cache/torch/hub/checkpoints/pretrain_res50x1.pkl"):
        urllib.request.urlretrieve(model_urls[backbone], f"{home}/.cache/torch/hub/checkpoints/pretrain_res50x1.pkl")
    ckpt = load_checkpoint(f"{home}/.cache/torch/hub/checkpoints/pretrain_res50x1.pkl")

    weights = ckpt['experiment_state'].online_params
    bn_states = ckpt['experiment_state'].online_state

    state_dict = {}
    print('==> process weights')
    for k, v in zip(weights.keys(), weights.values()):
        if 'projector' in k or 'predictor' in k:
            continue
        f_k = k
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/(conv|batchnorm)_([0-9])',
            lambda m: 'layer{}.{}.{}{}'.format(int(m[1]) + 1, int(m[2]), m[3], int(m[4]) + 1),
            f_k
        )
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/shortcut_(conv|batchnorm)',
            lambda m: 'layer{}.{}.{}'.format(int(m[1]) + 1, int(m[2]), 'downsample.' + m[3]) \
                .replace('conv', '0').replace('batchnorm', '1'),
            f_k
        )
        f_k = re.sub(
            '.*initial_(conv|batchnorm)(_1)?',
            lambda m: '{}'.format(m[1] + '1'),
            f_k
        )
        f_k = f_k.replace('batchnorm', 'bn')
        f_k = f_k.replace('classifier', 'fc')
        for p_k, p_v in zip(v.keys(), v.values()):
            p_k = p_k.replace('w', '.weight')
            p_k = p_k.replace('b', '.bias')
            p_k = p_k.replace('offset', '.bias')
            p_k = p_k.replace('scale', '.weight')
            ff_k = f_k + p_k
            p_v = torch.from_numpy(p_v)
            # print(k, ff_k, p_v.shape)
            if 'conv' in ff_k or 'downsample.0' in ff_k:
                state_dict[ff_k] = p_v.permute(3, 2, 0, 1)
            elif 'bn' in ff_k or 'downsample.1' in ff_k:
                state_dict[ff_k] = p_v.squeeze()
            elif 'fc.weight' in ff_k:
                state_dict[ff_k] = p_v.permute(1, 0)
            else:
                state_dict[ff_k] = p_v

    print('==> process bn_states')
    for k, v in zip(bn_states.keys(), bn_states.values()):
        if 'projector' in k or 'predictor' in k:
            continue
        f_k = k
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/(conv|batchnorm)_([0-9])',
            lambda m: 'layer{}.{}.{}{}'.format(int(m[1]) + 1, int(m[2]), m[3], int(m[4]) + 1),
            f_k
        )
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9]*)/~/shortcut_(conv|batchnorm)',
            lambda m: 'layer{}.{}.{}'.format(int(m[1]) + 1, int(m[2]), 'downsample.' + m[3]) \
                .replace('conv', '0').replace('batchnorm', '1'),
            f_k
        )
        f_k = re.sub(
            '.*initial_(conv|batchnorm)',
            lambda m: '{}'.format(m[1] + '1'),
            f_k
        )
        f_k = f_k.replace('batchnorm', 'bn')
        f_k = f_k.replace('/~/mean_ema', '.running_mean')
        f_k = f_k.replace('/~/var_ema', '.running_var')
        assert np.abs(v['average'] - v['hidden']).sum() == 0
        state_dict[f_k] = torch.from_numpy(v['average']).squeeze()

    # if 'res200' in args.byol_wts_path:
    #     pt_state_dict = resnet200().state_dict()
    # else:
    pt_state_dict = resnet50().state_dict()

    pt_state_dict = {k: v for k, v in pt_state_dict.items() if 'tracked' not in k}

    assert len(pt_state_dict) == len(state_dict)
    for (k, v), (pk, pv) in zip(sorted(list(state_dict.items())), sorted(list(pt_state_dict.items()))):
        assert k == pk
        assert v.shape == pv.shape

    projector_state_dict, _ = convert_mlp_weights(weights, bn_states)
    full_ckpt = {
        "backbone": state_dict,
        "projector": projector_state_dict
    }

    torch.save(
        full_ckpt,
        f"{home}/.cache/torch/hub/checkpoints/byol_r50_full.pth.tar"
    )
    ckpt = torch.load(f"{home}/.cache/torch/hub/checkpoints/byol_r50_full.pth.tar")


    model.backbone.load_state_dict(ckpt["backbone"])
    model.projector.load_state_dict(ckpt["projector"])
    return model