"""CelebA (aligned & cropped JPGs) -> 80x80 grayscale uint8 arrays, official split (BASELINE.md §1).

    python scripts/prepare_celeba.py --src <folder with list_eval_partition.csv + img_align_celeba> \
        --out $CNF_DATA/processed

Writes <out>/celeba80_{train,val,test}.npy (+ .txt with file names, same order) and the fixed
face probe lists <out>/selections/{probe_faces,eval_faces}.txt (from val, seed 0).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carsnfaces.data import celeba_crop, load_gray  # noqa: E402

SPLITS = {0: "train", 1: "val", 2: "test"}


def _load(path: str) -> np.ndarray:
    return celeba_crop(load_gray(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="celeba80")
    ap.add_argument("--n-probe", type=int, default=16)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    part = next(src.rglob("list_eval_partition.csv"))
    img_dir = next(p.parent for p in src.rglob("000001.jpg"))
    with open(part) as f:
        rows = list(csv.reader(f))
    if not rows[0][1].strip().isdigit():
        rows = rows[1:]  # header
    out.mkdir(parents=True, exist_ok=True)

    for code, split in SPLITS.items():
        names = [r[0] for r in rows if int(r[1]) == code]
        t = time.time()
        with Pool(os.cpu_count()) as pool:
            arr = np.stack(pool.map(_load, [str(img_dir / n) for n in names], chunksize=512))
        np.save(out / f"{args.prefix}_{split}.npy", arr)
        (out / f"{args.prefix}_{split}.txt").write_text("\n".join(names) + "\n")
        print(f"[celeba] {split}: {arr.shape} in {time.time() - t:.0f}s")
        if split == "val":
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(len(names))
            sel = out / "selections"
            sel.mkdir(exist_ok=True)
            probe = [names[i] for i in order[: args.n_probe]]
            ev = [names[i] for i in order[args.n_probe: args.n_probe + args.n_eval]]
            (sel / "probe_faces.txt").write_text("\n".join(probe) + "\n")
            (sel / "eval_faces.txt").write_text("\n".join(ev) + "\n")
            print(f"[celeba] selections: {len(probe)} probe, {len(ev)} eval faces")


if __name__ == "__main__":
    main()
