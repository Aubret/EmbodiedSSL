# Copyright 2023 solo-learn development team.
import os
import time
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

from typing import Any, Dict, List, Sequence, Tuple, Type, Union

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

from solo.losses.mocov3 import mocov3_loss_func
from solo.methods import MoCoV3
from solo.methods.base import BaseMomentumMethod
from solo.utils.actions import get_crop_diffparams, prepare_aa_input, get_action_dims
from solo.utils.misc import omegaconf_select, get_rank
from solo.utils.momentum import initialize_momentum_params
from solo.utils.visual_streams import create_stream, init_cfg_streams


class AAMoCoV3(MoCoV3):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        ### AAMOCOV3
        cfg.method_kwargs.pre_aa_layers= omegaconf_select(cfg, "method_kwargs.pre_aa_layers", 0)
        cfg.method_kwargs.aa_layers= omegaconf_select(cfg, "method_kwargs.aa_layers", 3)
        cfg.method_kwargs.aa_layers_pred= omegaconf_select(cfg, "method_kwargs.aa_layers_pred", 3   )
        cfg.method_kwargs.aa_hidden_dim = omegaconf_select(cfg, "method_kwargs.aa_hidden_dim", 4096)
        cfg.method_kwargs.aa_weight = omegaconf_select(cfg, "method_kwargs.aa_weight", 1)
        cfg.method_kwargs.tt_weight = omegaconf_select(cfg, "method_kwargs.tt_weight", 1)
        cfg.method_kwargs.aa_temperature = omegaconf_select(cfg, "method_kwargs.aa_temperature", 0.2)
        cfg.method_kwargs.equivariant = omegaconf_select(cfg, "method_kwargs.equivariant", False)
        cfg.method_kwargs.aa_bn = omegaconf_select(cfg, "method_kwargs.aa_bn", True)
        cfg.method_kwargs.aa_last_bn = omegaconf_select(cfg, "method_kwargs.aa_last_bn", True)
        cfg.method_kwargs.use_crop_params = omegaconf_select(cfg, "method_kwargs.use_crop_params", 0)
        cfg = init_cfg_streams(cfg)

        self.backbone = create_feature_extractor(self.backbone, return_nodes=list(cfg.method_kwargs.layer_names))
        self.momentum_backbone = create_feature_extractor(self.momentum_backbone, return_nodes=list(cfg.method_kwargs.layer_names))

        self.ventral = create_stream(self.cfg.method_kwargs.ventral)
        self.momentum_ventral = create_stream(self.cfg.method_kwargs.ventral)

        self.dorsal = create_stream(cfg.method_kwargs.dorsal)
        self.momentum_dorsal = create_stream(cfg.method_kwargs.dorsal)
        aa_input_dim = get_action_dims(cfg)

        self.action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                aa_input_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=cfg.method_kwargs.aa_last_bn,
                                                first_bn=cfg.method_kwargs.aa_bn
                                                )

        self.momentum_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                aa_input_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=cfg.method_kwargs.aa_last_bn,
                                                first_bn=cfg.method_kwargs.aa_bn
                                                )

        self.action_predictor = self._build_mlp(cfg.method_kwargs.aa_layers_pred,
                                                cfg.method_kwargs.proj_output_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=False
                                                )

        aa_input_dim = self.features_dim * 2
        if cfg.method_kwargs.pre_aa_layers:
            aa_input_dim = self.features_dim
        if "resnet50_dorsal" in self.cfg.backbone.name and cfg.method_kwargs.dorsal.enabled is True:
            aa_input_dim = aa_input_dim // 2

        if self.cfg.method_kwargs.use_crop_params == 1:
            aa_input_dim += 4


        self.vis_pre_action_projector = self._build_mlp(cfg.method_kwargs.pre_aa_layers,
                                                aa_input_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=False
                                                )

        self.momentum_vis_pre_action_projector = self._build_mlp(cfg.method_kwargs.pre_aa_layers,
                                                aa_input_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=False
                                                )
        aa_input_dim = self.features_dim * 2
        if "resnet50_dorsal" in self.cfg.backbone.name and cfg.method_kwargs.dorsal.enabled is True:
            aa_input_dim = aa_input_dim // 2
        if cfg.method_kwargs.pre_aa_layers:
            aa_input_dim = cfg.method_kwargs.proj_output_dim * 2

        # aa_input_dim = self.features_dim
        self.vis_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                aa_input_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=cfg.method_kwargs.aa_last_bn
                                                )

        self.momentum_vis_action_projector = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                         aa_input_dim,
                                                         cfg.method_kwargs.aa_hidden_dim,
                                                         cfg.method_kwargs.proj_output_dim,
                                                         last_bn=cfg.method_kwargs.aa_last_bn
                                                         )

        self.vis_action_predictor = self._build_mlp(cfg.method_kwargs.aa_layers,
                                                cfg.method_kwargs.proj_output_dim,
                                                cfg.method_kwargs.aa_hidden_dim,
                                                cfg.method_kwargs.proj_output_dim,
                                                last_bn=False
                                                )

        if cfg.method_kwargs.equivariant:
            self.predictor = self._build_mlp(
                cfg.method_kwargs.layers_pred ,
                cfg.method_kwargs.proj_output_dim + + cfg.method_kwargs.proj_output_dim,
                cfg.method_kwargs.pred_hidden_dim,
                cfg.method_kwargs.proj_output_dim,
                last_bn=False,
            )
        initialize_momentum_params(self.action_projector, self.momentum_action_projector)
        initialize_momentum_params(self.vis_pre_action_projector, self.momentum_vis_pre_action_projector)
        initialize_momentum_params(self.vis_action_projector, self.momentum_vis_action_projector)
        initialize_momentum_params(self.dorsal, self.momentum_dorsal)
        initialize_momentum_params(self.ventral, self.momentum_ventral)
        if get_rank() == 0:
            print(self)


    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "action_projector", "params": self.action_projector.parameters()},
            {"name": "action_predictor", "params": self.action_predictor.parameters()},
            {"name": "vis_action_projector", "params": self.vis_action_projector.parameters()},
            {"name": "vis_pre_action_projector", "params": self.vis_pre_action_projector.parameters()},
            {"name": "vis_action_predictor", "params": self.vis_action_predictor.parameters()},
        ]

        if self.cfg.method_kwargs.dorsal.enabled:
            extra_learnable_params.append({"name": "dorsal", "params": self.dorsal.parameters()})
        if self.cfg.method_kwargs.ventral.enabled:
            extra_learnable_params.append({"name": "ventral", "params": self.ventral.parameters()})
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [
            (self.action_projector, self.momentum_action_projector),
            (self.vis_action_projector, self.momentum_vis_action_projector),
            (self.vis_pre_action_projector, self.momentum_vis_pre_action_projector)
        ]
        if self.cfg.method_kwargs.dorsal.enabled:
            extra_momentum_pairs.append((self.dorsal, self.momentum_dorsal))
        if self.cfg.method_kwargs.ventral.enabled:
            extra_momentum_pairs.append((self.ventral, self.momentum_ventral))
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = BaseMomentumMethod.forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.ventral(o)

        z = self.projector(o)
        # out.update({"q": q, "z": z})
        out.update({"z": z})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = BaseMomentumMethod.momentum_forward(self, X)
        o = out[self.cfg.method_kwargs.ventral.layer_name]
        o = self.momentum_ventral(o)

        k = self.momentum_projector(o)
        # out.update({"q": q, "z": z})
        out.update({"k": k})
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
        class_loss = out["loss"]

        # AA stuff
        v1 = self.dorsal(out[self.cfg.method_kwargs.dorsal.layer_name][0])
        v2 = self.dorsal(out[self.cfg.method_kwargs.dorsal.layer_name][1])


        av1 = self.vis_pre_action_projector(v1)
        av2 = self.vis_pre_action_projector(v2)

        vis_action_proj = self.vis_action_projector(prepare_aa_input(self.cfg, av1, av2, batch))
        vis_action_pred = self.vis_action_predictor(vis_action_proj)

        mom_v1 = self.momentum_dorsal(out["momentum_"+self.cfg.method_kwargs.dorsal.layer_name][0])
        mom_v2 = self.momentum_dorsal(out["momentum_"+self.cfg.method_kwargs.dorsal.layer_name][1])


        step , X, targets = batch
        # action = X[-1]
        action = X[-1] if len(X) > 2 else None
        # Add crop params if required
        if self.cfg.method_kwargs.use_crop_params in [2,3,4]:
            crop_diff = get_crop_diffparams(X[0][1], X[1][1])
            if self.cfg.method_kwargs.use_crop_params == 4:
                crop_diff = crop_diff[:,:2]
            action = crop_diff if action is None else torch.cat((action, crop_diff), dim=1)

        action_proj = self.action_projector(action)
        action_pred = self.action_predictor(action_proj)

        with torch.no_grad():
            amom_v1 = self.momentum_vis_pre_action_projector(mom_v1)
            amom_v2 = self.momentum_vis_pre_action_projector(mom_v2)
            mom_vis_action_proj = self.momentum_vis_action_projector(prepare_aa_input(self.cfg, amom_v1, amom_v2, batch))
            mom_action_proj = self.momentum_action_projector(action)

        aa_contrastive_loss =  mocov3_loss_func(
            vis_action_pred, mom_action_proj, temperature=self.cfg.method_kwargs.aa_temperature
        ) + mocov3_loss_func(action_pred, mom_vis_action_proj, temperature=self.cfg.method_kwargs.aa_temperature)

        # MoCoV3 stuff
        if not self.cfg.method_kwargs.equivariant:
            Q = ( self.predictor(out["z"][0]), self.predictor(out["z"][1]))
            K = out["momentum_k"]

            contrastive_loss = mocov3_loss_func(
                Q[0], K[1], temperature=self.temperature
            ) + mocov3_loss_func(Q[1], K[0], temperature=self.temperature)
        else:
            Q = self.predictor(torch.cat((out["z"][0], action_proj), dim=1))
            contrastive_loss = mocov3_loss_func(Q, out["momentum_k"][1], temperature=self.temperature)



        metrics = {
            "train_contrastive_loss": contrastive_loss,
            "train_aa_constrastive_loss": aa_contrastive_loss
        }
        for i in range(action.shape[1]):
            metrics[f"a{i}"] = torch.abs(action[0,i])


        # if batch_idx == 0:
        #     img = X[0]
        #     img2 = X[1]
        #     mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        #     unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        #     os.makedirs(f"/home/aubret/test_images/diff_sameaug{self.cfg.data.dataset_kwargs.gaze_size}", exist_ok=True)
        #     for i in range(32):
        #         imgi = unnormalize(img[i:i+1].cpu())
        #         imgi2 = unnormalize(img2[i:i+1].cpu())
        #         torchvision.utils.save_image(imgi, f"/home/aubret/test_images/diff_sameaug{self.cfg.data.dataset_kwargs.gaze_size}/{step[i].item()}_1.png")
        #         torchvision.utils.save_image(imgi2, f"/home/aubret/test_images/diff_sameaug{self.cfg.data.dataset_kwargs.gaze_size}/{step[i].item()}_2.png")
        #     print("finish saving")

        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)
        loss = self.cfg.method_kwargs.tt_weight * contrastive_loss + class_loss + self.cfg.method_kwargs.aa_weight * aa_contrastive_loss


        # loss.mean().backward()  # keep graph if needed for debugging
        #
        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         print(f"[UNUSED] {name}")

        return loss

