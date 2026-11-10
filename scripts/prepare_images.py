"""Make a folder of square, fixed-size images (e.g. the hand-picked car fronts).

    python scripts/prepare_images.py --input data/raw/stanford_cars \
        --list data/selections/car_fronts.txt --out data/processed/car_fronts_64 --size 64

--list   optional file of image paths to keep, relative to --input
         (one per line, '#' comments allowed). Paths, not bare names: Stanford Cars
         reuses filenames like 00001.jpg across its train and test folders.
--mode   crop: center-crop to a square (default); pad: pad to a square with --pad-color
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carsnfaces.data import list_images, load_rgb  # noqa: E402


def read_list(path: str) -> set[str]:
    names = set()
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(Path(line).as_posix())
    return names


def to_square(img: Image.Image, size: int, mode: str, pad_color) -> Image.Image:
    if mode == "pad":
        return ImageOps.pad(img, (size, size), color=pad_color, method=Image.LANCZOS)
    return ImageOps.fit(img, (size, size), method=Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--list", default=None)
    ap.add_argument("--mode", choices=["crop", "pad"], default="crop")
    ap.add_argument("--pad-color", default="white")
    args = ap.parse_args()

    paths = list_images(args.input)
    if args.list:
        keep = read_list(args.list)
        root = Path(args.input)
        paths = [p for p in paths if p.relative_to(root).as_posix() in keep]
        missing = keep - {p.relative_to(root).as_posix() for p in paths}
        if missing:
            print(f"[warn] {len(missing)} listed files not found, e.g. {sorted(missing)[:5]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in paths:
        # flatten the relative path into the name so train/00001 and test/00001 don't collide
        name = "__".join(p.relative_to(args.input).with_suffix("").parts)
        to_square(load_rgb(p), args.size, args.mode, args.pad_color).save(out / f"{name}.png")
    print(f"[prepare] {len(paths)} images -> {out}")


if __name__ == "__main__":
    main()
