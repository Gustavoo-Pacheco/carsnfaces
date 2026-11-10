"""DVM-CAR -> 64x64 grayscale car arrays + fixed probe/eval lists (BASELINE.md §1).

    # 1. look at the image table (viewpoint codes) and exit
    python scripts/prepare_dvm.py --tables-zip tables_V2.0.zip --inspect

    # 2. everything
    python scripts/prepare_dvm.py --fronts-zip Confirmed_fronts.zip --tables-zip tables_V2.0.zip \
        --full-zip resized_DVM_v2.zip --rear-view 180 --out $CNF_DATA/processed

Fronts: all quality-checked fronts -> <out>/dvm_fronts64.npy (+ .txt names).
Rears:  only the selected ones (probe + eval) are read from the full zip -> <out>/dvm_rears64.npy.
Lists:  <out>/selections/{probe,eval}_{fronts,rears}.txt (seed 0). Probe fronts use distinct makers;
        probe rears match the probe fronts' model (Genmodel ID) where possible.

DVM file names: Maker$$Model$$Year$$Color$$GenmodelID$$AdvID$$image_N.jpg
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carsnfaces.data import car_crop, load_gray  # noqa: E402

_ZIP: zipfile.ZipFile | None = None


def _init(zip_path: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(zip_path)


def _load(member: str) -> np.ndarray:
    return car_crop(load_gray(io.BytesIO(_ZIP.read(member))))


def image_members(zip_path: str) -> dict[str, str]:
    """basename -> member path, for all JPG/PNG files in a zip (skips macOS junk)."""
    with zipfile.ZipFile(zip_path) as z:
        return {PurePosixPath(m).name: m for m in z.namelist()
                if m.lower().endswith((".jpg", ".jpeg", ".png")) and "__MACOSX" not in m
                and not PurePosixPath(m).name.startswith("._")}


def process(zip_path: str, members: list[str]) -> np.ndarray:
    with Pool(os.cpu_count(), initializer=_init, initargs=(zip_path,)) as pool:
        return np.stack(pool.map(_load, members, chunksize=256))


def parts(name: str) -> dict:
    p = name.split("$$")
    return {"maker": p[0], "model": p[1], "genmodel": p[4] if len(p) > 4 else p[1]}


def read_image_table(tables_zip: str) -> pd.DataFrame:
    with zipfile.ZipFile(tables_zip) as z:
        member = next(m for m in z.namelist() if PurePosixPath(m).name.lower() == "image_table.csv")
        return pd.read_csv(z.open(member))


def find_col(df: pd.DataFrame, *keys: str) -> str:
    for c in df.columns:
        if all(k in c.lower() for k in keys):
            return c
    raise KeyError(f"no column with {keys} in {list(df.columns)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fronts-zip")
    ap.add_argument("--tables-zip", required=True)
    ap.add_argument("--full-zip")
    ap.add_argument("--out")
    ap.add_argument("--rear-view", type=int)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    table = read_image_table(args.tables_zip)
    view_col = find_col(table, "viewpoint")
    name_col = find_col(table, "name")
    if args.inspect:
        print(table.head(10).to_string())
        print(table[view_col].value_counts().sort_index().to_string())
        for v in sorted(table[view_col].dropna().unique())[:12]:
            print(v, table.loc[table[view_col] == v, name_col].head(2).tolist())
        return

    out, sel = Path(args.out), Path(args.out) / "selections"
    sel.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- fronts: all of them
    t = time.time()
    fm = image_members(args.fronts_zip)
    front_names = sorted(fm)
    fronts = process(args.fronts_zip, [fm[n] for n in front_names])
    np.save(out / "dvm_fronts64.npy", fronts)
    (out / "dvm_fronts64.txt").write_text("\n".join(front_names) + "\n")
    print(f"[dvm] fronts {fronts.shape} in {time.time() - t:.0f}s")

    order = [front_names[i] for i in rng.permutation(len(front_names))]
    probe_f, makers = [], set()
    for n in order:
        if parts(n)["maker"] not in makers:
            probe_f.append(n); makers.add(parts(n)["maker"])
        if len(probe_f) == args.n_probe:
            break
    for n in order:  # fewer makers than probes: fill up
        if len(probe_f) == args.n_probe:
            break
        if n not in probe_f:
            probe_f.append(n)
    taken = set(probe_f)
    eval_f = [n for n in order if n not in taken][: args.n_eval]

    # ---- rears: pick from the image table, read only those from the full zip
    rm = image_members(args.full_zip)
    rear_pool = [n for n in table.loc[table[view_col] == args.rear_view, name_col].astype(str) if n in rm]
    print(f"[dvm] {len(rear_pool)} rear images (viewpoint {args.rear_view}) found in the full zip")
    rear_pool = [rear_pool[i] for i in rng.permutation(len(rear_pool))]
    by_gen, by_maker = {}, {}
    for n in rear_pool:
        by_gen.setdefault(parts(n)["genmodel"], []).append(n)
        by_maker.setdefault(parts(n)["maker"], []).append(n)
    probe_r, used = [], set()
    for f in probe_f:
        p = parts(f)
        cands = [n for n in by_gen.get(p["genmodel"], []) + by_maker.get(p["maker"], []) + rear_pool
                 if n not in used]
        probe_r.append(cands[0]); used.add(cands[0])
    eval_r = [n for n in rear_pool if n not in used][: args.n_eval]

    rear_names = probe_r + eval_r
    t = time.time()
    rears = process(args.full_zip, [rm[n] for n in rear_names])
    np.save(out / "dvm_rears64.npy", rears)
    (out / "dvm_rears64.txt").write_text("\n".join(rear_names) + "\n")
    print(f"[dvm] rears {rears.shape} in {time.time() - t:.0f}s")

    for fname, names in (("probe_fronts", probe_f), ("eval_fronts", eval_f),
                         ("probe_rears", probe_r), ("eval_rears", eval_r)):
        (sel / f"{fname}.txt").write_text("\n".join(names) + "\n")
        print(f"[dvm] {fname}: {len(names)}")
    same = sum(parts(a)["genmodel"] == parts(b)["genmodel"] for a, b in zip(probe_f, probe_r))
    print(f"[dvm] probe rears with the same model as their front: {same}/{len(probe_f)}")


if __name__ == "__main__":
    main()
