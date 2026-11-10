"""Convolutional autoencoder, built from scratch.

Baseline for phase 2. Later variants (VAE, adversarial loss) go in this file too.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """image (C x S x S) -> latent vector -> image.

    The encoder halves the resolution down to 4x4; the decoder mirrors it with
    nearest-neighbour upsampling + conv (avoids checkerboard artifacts).
    """

    def __init__(self, in_channels: int = 1, image_size: int = 64, base_channels: int = 64,
                 latent_dim: int = 256):
        super().__init__()
        if image_size < 8 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two >= 8")
        n_stages = int(math.log2(image_size // 4))
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        enc: list[nn.Module] = []
        ch, chans = in_channels, []
        for i in range(n_stages):
            out = min(base_channels * 2**i, 512)
            enc += [
                nn.Conv2d(ch, out, 4, stride=2, padding=1),
                nn.BatchNorm2d(out) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            chans.append(out)
            ch = out
        self.encoder = nn.Sequential(*enc)
        self.feat_channels = ch
        self.to_latent = nn.Linear(ch * 16, latent_dim)
        self.from_latent = nn.Linear(latent_dim, ch * 16)

        dec: list[nn.Module] = []
        for out in reversed(chans[:-1]):
            dec += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, out, 3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
            ]
            ch = out
        dec += [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, in_channels, 3, padding=1),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z).view(-1, self.feat_channels, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def build_model(cfg) -> ConvAutoencoder:
    return ConvAutoencoder(
        in_channels=1 if cfg.grayscale else 3,
        image_size=cfg.image_size,
        base_channels=cfg.base_channels,
        latent_dim=cfg.latent_dim,
    )
