"""Residual convolutional beta-VAE, built from scratch (BASELINE.md §2).

image (1 x 64 x 64, values in [0, 1]) -> mu, logvar (latent_dim) -> logits (1 x 64 x 64)

No encoder->decoder skip connections and no spatial latent: everything the
decoder draws has to pass through the single latent vector, which is what
forces a car to come out as a face.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def norm(c: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(groups, c), c)


class ResBlock(nn.Module):
    """GN-SiLU-Conv3x3 twice, plus a skip (identity, or Conv1x1 when channels change)."""

    def __init__(self, c_in: int, c_out: int, groups: int):
        super().__init__()
        self.body = nn.Sequential(
            norm(c_in, groups), nn.SiLU(), nn.Conv2d(c_in, c_out, 3, padding=1),
            norm(c_out, groups), nn.SiLU(), nn.Conv2d(c_out, c_out, 3, padding=1),
        )
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class Down(nn.Module):
    """Stride-2 Conv 4x4."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Nearest x2 then Conv 3x3."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class BetaVAE(nn.Module):
    def __init__(self, in_channels: int = 1, image_size: int = 64, channels=(64, 128, 256, 512),
                 res_blocks: int = 2, groups: int = 32, latent_dim: int = 128, logvar_clamp: float = 10.0):
        super().__init__()
        channels = list(channels)
        if image_size // 2 ** len(channels) != 4:
            raise ValueError(f"{len(channels)} stages take {image_size}px to "
                             f"{image_size // 2 ** len(channels)}px; expected 4px")
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp
        top = channels[-1]
        self.top = top

        enc: list[nn.Module] = [nn.Conv2d(in_channels, channels[0], 3, padding=1)]
        c = channels[0]
        for c_out in channels:
            for _ in range(res_blocks):
                enc.append(ResBlock(c, c_out, groups))
                c = c_out
            enc.append(Down(c))
        enc += [ResBlock(c, c, groups), norm(c, groups), nn.SiLU(), nn.Flatten()]
        self.encoder = nn.Sequential(*enc)
        self.to_stats = nn.Linear(c * 16, 2 * latent_dim)

        self.from_latent = nn.Linear(latent_dim, top * 16)
        dec: list[nn.Module] = [ResBlock(top, top, groups)]
        c = top
        for c_out in reversed(channels):
            dec.append(Up(c))
            for _ in range(res_blocks):
                dec.append(ResBlock(c, c_out, groups))
                c = c_out
        dec += [norm(c, groups), nn.SiLU(), nn.Conv2d(c, in_channels, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x in [0, 1] -> (mu, logvar)."""
        mu, logvar = self.to_stats(self.encoder(x)).chunk(2, dim=1)
        return mu, logvar.clamp(-self.logvar_clamp, self.logvar_clamp)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z -> logits (apply sigmoid for an image)."""
        return self.decoder(self.from_latent(z).view(-1, self.top, 4, 4))

    def forward(self, x: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic: decode mu, return an image in [0, 1]. A car always gives the same face."""
        mu, _ = self.encode(x)
        return torch.sigmoid(self.decode(mu).float())


def kl_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) summed over latent dims, in fp32. Shape (B,)."""
    mu, logvar = mu.float(), logvar.float()
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1)


def recon_per_sample(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits summed over pixels, in fp32. Shape (B,)."""
    return F.binary_cross_entropy_with_logits(logits.float(), x.float(), reduction="none").flatten(1).sum(1)


def build_model(cfg) -> BetaVAE:
    return BetaVAE(
        in_channels=cfg.in_channels,
        image_size=cfg.image_size,
        channels=cfg.channels,
        res_blocks=cfg.res_blocks_per_stage,
        groups=cfg.gn_groups,
        latent_dim=cfg.latent_dim,
        logvar_clamp=cfg.logvar_clamp,
    )
