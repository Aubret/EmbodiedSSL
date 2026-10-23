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

from typing import Any, List, Sequence, Dict

import numpy as np
import omegaconf
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

from solo.losses import simclr_loss_func
from solo.methods import MoCoV3, SimCLR, BaseMethod
from solo.utils.actions import get_crop_diffparams, get_action_dims
from solo.utils.misc import omegaconf_select
from solo.utils.visual_streams import create_stream, init_cfg_streams


class AASimCLR(SimCLR):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        cfg.method_kwargs.aa_layers= omegaconf_select(cfg, "method_kwargs.aa_layers", 3)
        cfg.method_kwargs.aa_hidden_dim = omegaconf_select(cfg, "method_kwargs.aa_hidden_dim", 4096)
        cfg.method_kwargs.aa_weight = omegaconf_select(cfg, "method_kwargs.aa_weight", 1)
        cfg.method_kwargs.tt_weight = omegaconf_select(cfg, "method_kwargs.tt_weight", 1)
        cfg.method_kwargs.first_bn = omegaconf_select(cfg, "method_kwargs.first_bn", True)
        cfg.method_kwargs.affine = omegaconf_select(cfg, "method_kwargs.affine", False)
        cfg.method_kwargs.aa_temperature = omegaconf_select(cfg, "method_kwargs.aa_temperature", 0.1)
        cfg.method_kwargs.use_crop_params = omegaconf_select(cfg, "method_kwargs.use_crop_params", 0)
        cfg = init_cfg_streams(cfg)

        aa_input_dim = get_action_dims(cfg)


        self.action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                aa_input_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                first_bn=cfg.method_kwargs.first_bn,
                                                affine=cfg.method_kwargs.affine
                                                )

        aa_vis_input_dim = self.features_dim * 2
        if "resnet50_dorsal" in self.cfg.backbone.name:
            aa_vis_input_dim = self.features_dim

        self.vis_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                aa_vis_input_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                )

        self.ventral = create_stream(self.cfg.method_kwargs.ventral)
        self.dorsal = create_stream(cfg.method_kwargs.dorsal)
        self.backbone = create_feature_extractor(self.backbone, return_nodes=list(cfg.method_kwargs.layer_names))

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "action_projector", "params": self.action_projector.parameters()},
            {"name": "vis_action_projector", "params": self.vis_action_projector.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    def compute_action_loss(self, batch, out, v1, v2):
        indexes = batch[0]
        indexes = indexes.repeat(2)

        out1 = self.dorsal(out[self.cfg.method_kwargs.dorsal.layer_name][v1])
        out2 = self.dorsal(out[self.cfg.method_kwargs.dorsal.layer_name][v2])

        vis_action_proj = self.vis_action_projector(torch.cat((out1, out2), dim=1))

        _, X, targets = batch
        action = X[-1] if len(X) > self.num_crops else None
        # Add crop params if required
        if self.cfg.method_kwargs.use_crop_params in [2, 3, 4]:
            crop_diff = get_crop_diffparams(X[v1][1], X[v2][1])
            if self.cfg.method_kwargs.use_crop_params == 4:
                crop_diff = crop_diff[:, :2]
            action = crop_diff if action is None else torch.cat((action, crop_diff), dim=1)

        action_proj = self.action_projector(action)

        return simclr_loss_func(
            torch.cat((vis_action_proj, action_proj)),
            indexes=indexes,
            temperature=self.cfg.method_kwargs.aa_temperature
        )

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = BaseMethod.forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.ventral(o)
        z = self.projector(o)
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = BaseMethod.multicrop_forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.ventral(o)
        z = self.projector(o)
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
        # SimCLR stuff
        indexes = batch[0]

        out = BaseMethod.training_step(self, batch, batch_idx)
        class_loss = out["loss"]

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(2)
        nce_loss = 0
        for v1 in range(self.num_crops):
            for v2 in range(v1+1, self.num_crops):
                nce_loss += simclr_loss_func(
                    torch.cat((out["z"][v1], out["z"][v2])),
                    indexes=indexes,
                    temperature=self.temperature
                )

        # AA stuff
        if n_augs == 2:
            aa_nce_loss = self.compute_action_loss(batch, out, 0, 1)
        else:
            aa_nce_loss = 0
            for v1 in range(self.num_crops):
                for v2 in range(v1+1, self.num_crops):
                    aa_nce_loss += self.compute_action_loss(batch , out, v1, v2)

        metrics = {
            "train_constrastive_loss": nce_loss,
            "train_aa_constrastive_loss": aa_nce_loss
        }

        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)
        return self.cfg.method_kwargs.tt_weight * nce_loss + class_loss + self.cfg.method_kwargs.aa_weight * aa_nce_loss

