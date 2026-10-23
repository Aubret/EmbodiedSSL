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

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.vicreg import vicreg_loss_func
from solo.methods import VICReg
from solo.methods.base import BaseMethod
from solo.utils.actions import get_action_dims
from solo.utils.misc import omegaconf_select


class AAVICReg(VICReg):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                sim_loss_weight (float): weight of the invariance term.
                var_loss_weight (float): weight of the variance term.
                cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        cfg.method_kwargs.aa_weight = omegaconf_select(cfg, "method_kwargs.aa_weight", 1)
        cfg.method_kwargs.tt_weight = omegaconf_select(cfg, "method_kwargs.tt_weight", 1)
        cfg.method_kwargs.aa_layers= omegaconf_select(cfg, "method_kwargs.aa_layers", 3)
        cfg.method_kwargs.aa_last_bn = omegaconf_select(cfg, "method_kwargs.aa_last_bn", False)
        cfg.method_kwargs.aa_bn = omegaconf_select(cfg, "method_kwargs.aa_bn", True)
        cfg.method_kwargs.use_crop_params = omegaconf_select(cfg, "method_kwargs.use_crop_params", 0)


        # projector
        self.vis_action_projector = self._build_mlp(cfg.method_kwargs.layers, 2*self.features_dim, proj_hidden_dim, proj_output_dim, last_bn=cfg.method_kwargs.aa_last_bn)


        self.action_projector = self._build_mlp(cfg.method_kwargs.layers,get_action_dims(cfg), proj_hidden_dim, proj_output_dim, last_bn=cfg.method_kwargs.aa_last_bn, first_bn=cfg.method_kwargs.aa_bn)

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "vis_action_projector", "params": self.vis_action_projector.parameters()},
                                  {"name" : "action_projector", "params": self.action_projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        out = BaseMethod.training_step(self, batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- vicreg loss -------
        vicreg_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )



        # AA stuff
        v1 = out["feats"][0]
        v2 = out["feats"][1]
        vis_action_proj = self.vis_action_projector(torch.cat((v1, v2), dim=1))

        _, X, targets = batch
        action = X[-1] if len(X) > 2 else None
        action_proj = self.action_projector(action)

        aa_vicreg_loss = vicreg_loss_func(
            vis_action_proj,
            action_proj,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        metrics = {
            "train_vicreg_los": vicreg_loss,
            "train_aa_vicreg_loss": aa_vicreg_loss
        }

        self.log_dict(metrics, on_epoch=True, on_step=True, sync_dist=True)

        return self.cfg.method_kwargs.tt_weight*vicreg_loss + class_loss + self.cfg.method_kwargs.aa_weight * aa_vicreg_loss
