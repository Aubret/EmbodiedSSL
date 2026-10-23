# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities import grad_norm
from torchvision.models.feature_extraction import create_feature_extractor

from solo.losses.byol import byol_loss_func
from solo.methods import BYOL
from solo.methods.base import BaseMomentumMethod
from solo.methods.byol import MultiLayerProj
from solo.utils.actions import prepare_aa_input, get_crop_diffparams, get_action_dims
from solo.utils.misc import omegaconf_select, get_rank
from solo.utils.momentum import initialize_momentum_params
from solo.utils.visual_streams import init_cfg_streams, create_stream


class AABYOL(BYOL):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        cfg.method_kwargs.action_layers = omegaconf_select(cfg, "method_kwargs.action_layers", 2)
        cfg.method_kwargs.action_hidden_dim = omegaconf_select(cfg, "method_kwargs.action_hidden_dim", 4096)
        cfg.method_kwargs.action_weight = omegaconf_select(cfg, "method_kwargs.action_weight", 0.1)
        cfg.method_kwargs.use_crop_params = omegaconf_select(cfg, "method_kwargs.use_crop_params", 2)
        cfg = init_cfg_streams(cfg)

        self.backbone = create_feature_extractor(self.backbone, return_nodes=list(cfg.method_kwargs.layer_names))
        self.momentum_backbone = create_feature_extractor(self.momentum_backbone, return_nodes=list(cfg.method_kwargs.layer_names))

        self.ventral = create_stream(self.cfg.method_kwargs.ventral)
        self.momentum_ventral = create_stream(self.cfg.method_kwargs.ventral)

        self.dorsal = create_stream(cfg.method_kwargs.dorsal)
        # self.momentum_dorsal = create_stream(cfg.method_kwargs.dorsal)
        initialize_momentum_params(self.ventral, self.momentum_ventral)
        aa_input_dim = get_action_dims(cfg)

        self.action_predictor = MultiLayerProj(cfg.method_kwargs.action_layers, 2*self.features_dim, cfg.method_kwargs.action_hidden_dim, aa_input_dim, bias=False)
        self.action_bn = torch.nn.BatchNorm1d(aa_input_dim, affine=False)
        if get_rank() == 0:
            print(self)

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "action_predictor", "params": self.action_predictor.parameters()},
            {"name": "action_bn", "params": self.action_bn.parameters()},
        ]

        if self.cfg.method_kwargs.dorsal.enabled:
            extra_learnable_params.append({"name": "dorsal", "params": self.dorsal.parameters()})
        if self.cfg.method_kwargs.ventral.enabled:
            extra_learnable_params.append({"name": "ventral", "params": self.ventral.parameters()})

        return super().learnable_params + extra_learnable_params


    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:

        extra_momentum_pairs = []
        # if self.cfg.method_kwargs.dorsal.enabled:
        #     extra_momentum_pairs.append((self.dorsal, self.momentum_dorsal))
        # if self.cfg.method_kwargs.ventral.enabled:
        #     extra_momentum_pairs.append((self.ventral, self.momentum_ventral))
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = BaseMomentumMethod.forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.ventral(o)

        z = self.projector(o)
        # out.update({"q": q, "z": z})
        out.update({"z": z, "p": self.predictor(z)})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = BaseMomentumMethod.momentum_forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.momentum_ventral(o)

        z = self.momentum_projector(o)
        out.update({"z": z})
        return out





    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """

        out = BaseMomentumMethod.training_step(self, batch, batch_idx)

        #BYOL stuff
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative cosine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        #AA stuff
        v1 = self.dorsal(out[self.cfg.method_kwargs.dorsal.layer_name][0])
        v2 = self.dorsal(out[self.cfg.method_kwargs.dorsal.layer_name][1])

        action_prediction = self.action_predictor(torch.cat((v1, v2), dim=1))

        _, X, targets = batch

        action = X[-1] if self.cfg.data.dataset == "nymeria" else None
        # Add crop params if required
        if self.cfg.method_kwargs.use_crop_params == 2:
            crop_diff = get_crop_diffparams(X[0][1], X[1][1])
            action = crop_diff if action is None else torch.cat((action, crop_diff), dim=1)

        action = self.action_bn(action)
        action_loss = torch.nn.functional.mse_loss(action_prediction, action)
        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()
        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "action_loss": action_loss.detach()
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        loss = neg_cos_sim + class_loss + self.cfg.method_kwargs.action_weight*action_loss
        # loss.mean().backward()  # keep graph if needed for debugging

        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(f"[UNUSED] {name}")

        return loss

def get_action(src1, src2):
    cy1 = src1[:, 0] #+ 0.5 * src1[:, 2]
    cx1 = src1[:, 1] #+ 0.5 * src1[:, 3]
    h1, w1 = src1[:, 2], src1[:, 3]

    cy2 = src2[:, 0] #+ 0.5 * src2[:, 2]
    cx2 = src2[:, 1] #+ 0.5 * src2[:, 3]
    h2, w2 = src2[:, 2], src2[:, 3]


    d_cx = cx1 - cx2
    d_cy = cy1 - cy2
    d_sx = torch.sqrt(w1 / w2)
    d_sy = torch.sqrt(h1 / h2)

    # if len(src1) > 4:
    # return torch.stack((d_cx, d_cy, d_sx, d_sy, torch.abs(src1[:, 4] - src2[:, 4])), dim=1)
    return torch.stack((d_cx, d_cy, d_sx, d_sy), dim=1)

    # return torch.stack((d_cx, d_cy, d_sx, d_sy), dim=1)