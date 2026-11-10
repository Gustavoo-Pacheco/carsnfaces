"""Turn cars into faces with a trained beta-VAE (decodes mu: same car -> same face).

    # a folder of raw car photos (white background), cropped/padded like DVM-CAR
    python -m carsnfaces.infer --ckpt runs/faces_64_baseline/final.pt --input my_cars/ --out outputs/x

    # a preprocessed array (+ names file), optionally restricted to a list
    python -m carsnfaces.infer --ckpt .../final.pt --input $CNF_DATA/processed/dvm_fronts64.npy \
        --list data/selections/car_fronts.txt --out outputs/faces_64_baseline/study_fronts

Writes faces/<name>.png (the generated face), pairs/<name>.png (car | face), and
latents.pt (names, mu, logvar, per-image KL to the prior).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from .config import config_from_dict
from .data import car_crop, center_view, list_images, load_array, load_gray, read_names, select
from .model import build_model, kl_per_sample


def load_inputs(src: str, size: int, list_file: str | None):
    p = Path(src)
    if p.suffix == ".npy":
        arr, names = load_array(p, mmap=True), read_names(p.with_suffix(".txt"))
        idx = select(names, read_names(list_file)) if list_file else list(range(len(names)))
        return torch.from_numpy(np.ascontiguousarray(arr[idx])), [names[i] for i in idx]
    paths = list_images(p)
    if list_file:
        keep = set(read_names(list_file))
        paths = [q for q in paths if q.name in keep]
    return torch.from_numpy(np.stack([car_crop(load_gray(q), size) for q in paths])), [q.name for q in paths]


@torch.no_grad()
def run(ckpt_path: str, src: str, out_dir: str, list_file: str | None = None, batch_size: int = 256) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    x_u8, names = load_inputs(src, cfg.image_size, list_file)
    out = Path(out_dir)
    (out / "faces").mkdir(parents=True, exist_ok=True)
    (out / "pairs").mkdir(parents=True, exist_ok=True)

    mus, lvs = [], []
    for i in range(0, len(names), batch_size):
        x = center_view(x_u8[i:i + batch_size].to(device), cfg.image_size)
        mu, lv = model.encode(x)
        y = torch.sigmoid(model.decode(mu).float())
        for j in range(x.shape[0]):
            stem = Path(names[i + j]).stem
            save_image(y[j], out / "faces" / f"{stem}.png")
            save_image(torch.stack([x[j], y[j]]), out / "pairs" / f"{stem}.png", nrow=2, padding=2, pad_value=1.0)
        mus.append(mu.float().cpu()); lvs.append(lv.float().cpu())

    mu, lv = torch.cat(mus), torch.cat(lvs)
    torch.save({"names": names, "mu": mu, "logvar": lv, "kl": kl_per_sample(mu, lv), "ckpt": str(ckpt_path)},
               out / "latents.pt")
    print(f"[infer] {len(names)} images -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="folder of car images, or a .npy array with a .txt of names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", default=None, help="optional file of image names to keep")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    run(args.ckpt, args.input, args.out, args.list, args.batch_size)


if __name__ == "__main__":
    main()
