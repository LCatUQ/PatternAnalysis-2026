"""2D U-Net and a U-Net with dilated context aggregation for prostate MRI slices."""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    """Two 3x3 convolutions with batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ContextBlock(nn.Module):
    """Dilated context aggregation (Yu & Koltun, 2016) with a residual skip."""

    def __init__(self, channels, rates=(1, 2, 4, 8)):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for r in rates
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return x + out


class UNet(nn.Module):
    """Four-level U-Net, optionally with a context block at 1/8 resolution."""

    def __init__(self, in_ch=1, n_classes=6, base=32, context=False, rates=(1, 2, 4, 8)):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.enc1 = conv_block(in_ch, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.enc4 = conv_block(c3, c4)
        self.context = ContextBlock(c3, rates) if context else nn.Identity()
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(c4, c4 * 2)
        self.up4 = nn.ConvTranspose2d(c4 * 2, c4, 2, stride=2)
        self.dec4 = conv_block(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.context(self.enc3(self.pool(e2)))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_model(name="unet", **kwargs):
    """Return 'unet' (baseline) or 'can' (dilated context)."""
    name = name.lower()
    if name == "unet":
        return UNet(context=False, **kwargs)
    if name == "can":
        return UNet(context=True, **kwargs)
    raise ValueError(f"unknown model: {name}")
