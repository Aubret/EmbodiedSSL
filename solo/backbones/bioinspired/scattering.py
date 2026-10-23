import torch
from torch import nn


class Scattering(nn.Module):

    def __init__(self, J=3, L=6, shape=224, device="cuda", **kwargs):
        super().__init__()

        from kymatio.torch import Scattering2D
        self.scattering = Scattering2D(J=J, L=L, shape=(shape, shape), backend='torch')
        self.channels = 1
        #self.num_features = 217 * 28 * 28
        if J == 5 and L == 6:
            self.num_features = 391 * 7 * 7
        if J == 4 and L == 6:
            self.num_features = 241 * 14 * 14
        if J == 3 and L == 6:
            self.num_features = 127 * 28 * 28
        if J == 4 and L == 6 and shape == 32:
            self.num_features = 241 * 2 * 2
        self.device = "cpu"

    @torch.no_grad()
    def forward(self, img):
        """Run through everything and concatenate the output of the C2s."""
        if self.device == "cpu" and img.device.type != "cpu":
            self.scattering = self.scattering.to(img.device)
            self.device = "gpu"

        # print(img.shape)
        # s1 = self.scattering(img[:,0].contiguous())
        # s2 = self.scattering(img[:,1].contiguous())
        # s3 = self.scattering(img[:,2].contiguous())
        # assert img.shape[1] == 3
        # return torch.cat((s1, s2, s3), dim=1).flatten(1).detach()
        out = self.scattering(img.contiguous().squeeze()).detach().flatten(1)
        return out