import torch
from torch import nn


class FlowNetDecoder(nn.Module):
    # modified from https://github.com/MarkoMilenovic01/OpticalFlowFlownetPytorch/blob/main/src/model.py
    def __init__(self, in_channels=6):
        super().__init__()

        # ---- Decoder ----
        # self.up3 = nn.ConvTranspose2d(2*2048, 512, 2, stride=2)   # 512 → 256, H/8 → H/4
        # self.up2 = nn.ConvTranspose2d(2*1024+512, 256, 2, stride=2)   # 512 (256+256) → 128, H/4 → H/2
        # self.up1 = nn.ConvTranspose2d(2*512+256, 128, 2, stride=2)    # 256 (128+128) → 64, H/2 → H

        # self.up3 = nn.ConvTranspose2d(2*2048, 512, 3, stride=2)   # 512 → 256, H/8 → H/4
        # self.up2 = nn.ConvTranspose2d(2*1024+512, 256, 3, stride=2)   # 512 (256+256) → 128, H/4 → H/2
        # self.up1 = nn.ConvTranspose2d(2*512+256, 128, 3, stride=2)    # 256 (128+128) → 64, H/2 → H

        self.up3= nn.ConvTranspose2d( 2*2048, 512,kernel_size=4, stride=2, padding=1)  # H/32 → H/16
        self.up2 = nn.ConvTranspose2d(2*1024+512, 256,kernel_size=4, stride=2, padding=1)  # H/16 → H/8
        self.up1 = nn.ConvTranspose2d( 2*512+256, 128,kernel_size=4, stride=2, padding=1)  # H/8 → H
        self.up0 = nn.ConvTranspose2d(2*256+128, 64, kernel_size=4, stride=2, padding=1)  # H/8 → H

        # ---- Final flow prediction (only 1!) ----
        # self.flow_final = nn.Conv2d(2*256+128, 2, 1)             # 128 (64+64) → 2, H×W
        self.flow_final = nn.Conv2d(64, 2, 1)             # 128 (64+64) → 2, H×W

    def forward(self, layers):
        # Asssumes that the batch axis concatenates the two image samples (x,x')
        layer1, layer2, layer3, layer4 = layers
        layer1 = torch.cat(layer1.split(layer1.shape[0]//2), dim=1)
        layer2 = torch.cat(layer2.split(layer2.shape[0]//2), dim=1)
        layer3 = torch.cat(layer3.split(layer3.shape[0]//2), dim=1)
        layer4 = torch.cat(layer4.split(layer4.shape[0]//2), dim=1)
        # layer4 = layer4[:layer4.shape[0]//2]

        # Decoder
        x = self.up3(layer4)
        x = torch.cat([x, torch.nn.functional.pad(layer3, pad=(0, 0, 1, 0))], dim=1)
        # x = torch.cat([x, layer3], dim=1)

        x = self.up2(x)
        # print(x.shape, layer2.shape)
        x = torch.cat([x, torch.nn.functional.pad(layer2, pad=(0, 0, 1, 1))], dim=1)
        # x = torch.cat([x, layer2], dim=1)

        x = self.up1(x)            # x: H×W, 64 channels
        x = torch.cat([x, torch.nn.functional.pad(layer1, pad=(0, 0, 2, 2))], dim=1)  # x: H×W, 128 channels (64+64)
        # x = torch.cat([x, layer1], dim=1)  # x: H×W, 128 channels (64+64)

        x = self.flow_final(self.up0(x))
        return x
