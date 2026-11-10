"""Turn car images into faces with a trained autoencoder.

    python -m carsnfaces.infer --ckpt runs/faces_64/final.pt \
        --input data/processed/car_fronts_64 --out outputs/faces_64/car_fronts

Writes, for each input image:
    faces/<name>.png   the generated face (used in the survey)
    pairs/<name>.png   car | face side by side (for inspection)
plus latents.pt (names + latent vectors, for later analysis).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .config import Config
from .data import ImageFolderFlat, build_transform
from .model import build_model


@torch.no_grad()
def run(ckpt_path: str, input_dir: str, out_dir: str, batch_size: int = 64) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ImageFolderFlat(input_dir, build_transform(cfg.image_size, cfg.grayscale, augment=False))
    out = Path(out_dir)
    (out / "faces").mkdir(parents=True, exist_ok=True)
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    names, latents = [], []
    for start in range(0, len(ds), batch_size):
        idx = range(start, min(start + batch_size, len(ds)))
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        z = model.encode(x)
        y = model.decode(z)
        for j, i in enumerate(idx):
            name = ds.paths[i].stem
            save_image(y[j], out / "faces" / f"{name}.png")
            save_image(torch.stack([x[j], y[j]]), out / "pairs" / f"{name}.png", nrow=2, padding=2, pad_value=1.0)
            names.append(name)
        latents.append(z.cpu())

    torch.save({"names": names, "latents": torch.cat(latents), "ckpt": str(ckpt_path)}, out / "latents.pt")
    print(f"[infer] {len(names)} images -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    run(args.ckpt, args.input, args.out, args.batch_size)


if __name__ == "__main__":
    main()
