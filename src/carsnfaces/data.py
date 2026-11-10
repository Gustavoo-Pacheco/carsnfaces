"""Image datasets: a flat folder of images (searched recursively)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image folder not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


def load_rgb(path: str | Path, background=(255, 255, 255)) -> Image.Image:
    """Open an image as RGB, compositing transparency (Cartoon Set PNGs) onto a background."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, background)
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def build_transform(size: int, grayscale: bool, augment: bool) -> T.Compose:
    steps: list = []
    if augment:
        # Faces in CelebA/Cartoon Set are pre-aligned. Random crops and shifts force the
        # model to locate features instead of always drawing eyes at the same pixels.
        steps += [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.RandomHorizontalFlip(),
        ]
    else:
        steps += [T.Resize(size), T.CenterCrop(size)]
    if grayscale:
        steps.append(T.Grayscale(num_output_channels=1))
    steps.append(T.ToTensor())  # values in [0, 1]
    return T.Compose(steps)


class ImageFolderFlat(Dataset):
    """All images under a folder, no labels. Returns (tensor, index)."""

    def __init__(self, root: str | Path, transform, paths: list[Path] | None = None):
        self.paths = paths if paths is not None else list_images(root)
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        return self.transform(load_rgb(self.paths[i])), i
