"""Data: preprocessing helpers, array loading and GPU augmentation.

Faces live in RAM / on the GPU as uint8 arrays (N x 80 x 80); cars as N x 64 x 64.
Reading 160k small JPGs from Google Drive every epoch is far too slow, so the
JPGs are decoded once by the scripts in ``scripts/``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------- preprocessing (CPU, PIL)

def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image folder not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_gray(src, background: int = 255) -> Image.Image:
    """Open an image (path or file object) as 8-bit grayscale, compositing transparency onto white."""
    img = Image.open(src)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (background,) * 3 + (255,))
        img = Image.alpha_composite(bg, img)
    return img.convert("L")


def celeba_crop(img: Image.Image, crop: int = 148, size: int = 80) -> np.ndarray:
    """CelebA aligned 178x218 -> center crop 148x148 (whole head) -> size x size uint8."""
    w, h = img.size
    left, top = (w - crop) // 2, (h - crop) // 2
    img = img.crop((left, top, left + crop, top + crop))
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def car_crop(img: Image.Image, size: int = 64, margin: float = 0.05, white_thresh: int = 240) -> np.ndarray:
    """Car on a white background -> crop to the car + margin -> pad to a white square -> size x size uint8."""
    a = np.asarray(img, dtype=np.uint8)
    ys, xs = np.nonzero(a < white_thresh)
    if len(xs) == 0:  # blank image: just square it
        x0, y0, x1, y1 = 0, 0, a.shape[1], a.shape[0]
    else:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    car = img.crop((x0, y0, x1, y1))
    w, h = car.size
    side = max(w, h) + 2 * int(round(margin * max(w, h)))
    canvas = Image.new("L", (side, side), 255)       # white square, car centered
    canvas.paste(car, ((side - w) // 2, (side - h) // 2))
    return np.asarray(canvas.resize((size, size), Image.LANCZOS), dtype=np.uint8)


# ---------------------------------------------------------------- arrays

def load_array(path: str | Path, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)


def read_names(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]


def select(names: list[str], wanted: list[str]) -> list[int]:
    idx = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in idx]
    if missing:
        raise KeyError(f"{len(missing)} selected images not found, e.g. {missing[:3]}")
    return [idx[w] for w in wanted]


def to_float(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (N, H, W) or (N, 1, H, W) -> float (N, 1, H, W) in [0, 1]."""
    if x_u8.dim() == 3:
        x_u8 = x_u8.unsqueeze(1)
    return x_u8.float().div_(255.0)


def center_view(x_u8: torch.Tensor, size: int = 64) -> torch.Tensor:
    """Eval view of faces: the whole 80px image resized to 64px."""
    x = to_float(x_u8)
    if x.shape[-1] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)


def augment(x_u8: torch.Tensor, size: int, area_min: float, rot_deg: float, flip: bool,
            gen: torch.Generator) -> torch.Tensor:
    """Random crop (area area_min..1, random position) + rotation + flip, resized to size x size.

    One affine resample per image on the GPU. ``gen`` makes it reproducible (seeded per step).
    """
    x = to_float(x_u8)
    b, dev = x.shape[0], x.device

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(b, device=dev, generator=gen)

    s = u(area_min, 1.0).sqrt()                      # side fraction of the crop
    a = u(-rot_deg, rot_deg) * (math.pi / 180.0)
    fx = torch.where(torch.rand(b, device=dev, generator=gen) < 0.5, -1.0, 1.0) if flip \
        else torch.ones(b, device=dev)
    tx, ty = u(-1.0, 1.0) * (1 - s), u(-1.0, 1.0) * (1 - s)
    cos, sin = a.cos(), a.sin()
    theta = torch.stack([
        torch.stack([s * cos * fx, -s * sin, tx], dim=1),
        torch.stack([s * sin * fx, s * cos, ty], dim=1),
    ], dim=1)
    # downsample to ~the output resolution first, so the bilinear resample doesn't alias
    pre = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    grid = F.affine_grid(theta, (b, 1, size, size), align_corners=False)
    return F.grid_sample(pre, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
