#!/usr/bin/env python3
"""Build the 80 m class-map labels of the final programme (Ravi, 2026-10-04).

For every patch, from the 10 m flood label F (`patch_*_flood_mask.tif`, 1 = flood) and the 10 m
permanent water W (`input_10m.tif` band 5, 1 = water), count the 64 10 m cells of each 80 m cell:

    nF = flooded cells outside permanent water      nW = permanent-water cells
    nU = cells that are flood or permanent water

and assign ONE class per 80 m cell (32-of-64 majority rule, D4 of METHOD_PLAN_RERUN.md):

    2  permanent water   nW >= 32                     (wins a 32/32 tie with flood)
    1  flood             nF >= 32
    3  mixed             nU >= 32, but nF < 32 and nW < 32 (flood + water together)
    0  dry               otherwise

Output: <out>/<product>.npz, keys = patch numbers as strings, each a (32, 32) uint8 class map,
plus <out>/manifest.csv (product, patches, cells per class, md5 of the npz).

`--check16 N` additionally rebuilds the OLD 16-of-64 layers (union, flood, pw) for N random
products and compares them with the old `labels_80m_global` files, to prove that the pipeline is
the same apart from the threshold. Exits nonzero on any missing input or mismatch.

    python build_labels_80m.py --patches <patches_tif> --out <dir> --workers 40 [--check16 30 --old <labels_80m_global>]
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import rasterio

NODATA = 9999
PATCH_RE = re.compile(r"patch_(\d+)_flood_mask\.tif$")


def counts(flood_path: Path, in10_path: Path):
    y = rasterio.open(flood_path).read(1)
    w = rasterio.open(in10_path).read(5)
    if y.shape != (256, 256) or w.shape != (256, 256):
        raise ValueError(f"unexpected shape {y.shape} / {w.shape} at {flood_path}")
    W = (w == 1)
    F = (y == 1) & ~W
    U = (y == 1) | W
    pool = lambda a: a.reshape(32, 8, 32, 8).sum(axis=(1, 3))
    return pool(F), pool(W), pool(U)


def classify(nF, nW, nU, thr=32):
    c = np.zeros(nF.shape, np.uint8)
    c[(nU >= thr)] = 3
    c[(nF >= thr)] = 1
    c[(nW >= thr)] = 2
    return c


def one_product(args):
    product_dir, out_dir = args
    files = sorted(p for p in product_dir.iterdir() if PATCH_RE.search(p.name))
    if not files:
        return product_dir.name, None, f"no flood_mask files in {product_dir}"
    maps, tot = {}, np.zeros(4, np.int64)
    for f in files:
        idx = int(PATCH_RE.search(f.name).group(1))
        in10 = f.with_name(f"patch_{idx:04d}_input_10m.tif")
        if not in10.exists():
            return product_dir.name, None, f"missing {in10}"
        nF, nW, nU = counts(f, in10)
        c = classify(nF, nW, nU)
        maps[str(idx)] = c
        tot += np.bincount(c.ravel(), minlength=4)[:4]
    out = out_dir / f"{product_dir.name}.npz"
    part = out.with_suffix(".part.npz")
    np.savez_compressed(part, **maps)
    part.replace(out)
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    return product_dir.name, (len(files), *tot.tolist(), md5), None


def check16(product_dir: Path, old_dir: Path):
    """Rebuild the old 16/64 union/flood/pw layers and compare with the old npz."""
    old = np.load(old_dir / f"{product_dir.name}.npz")
    bad = 0
    for k in old.files:
        idx = int(k)
        f = product_dir / f"patch_{idx:04d}_flood_mask.tif"
        nF, nW, nU = counts(f, f.with_name(f"patch_{idx:04d}_input_10m.tif"))
        mine = np.stack([nU >= 16, nF >= 16, nW >= 16], axis=-1).astype(np.uint8)
        if not np.array_equal(mine, old[k].astype(np.uint8)):
            bad += 1
    return product_dir.name, len(old.files), bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--check16", type=int, default=0)
    ap.add_argument("--old", type=Path)
    a = ap.parse_args()

    products = sorted(d for d in a.patches.glob("*/*") if d.is_dir())
    if len(products) == 0:
        sys.exit("no product directories found")

    if a.check16:
        if not a.old:
            sys.exit("--check16 needs --old")
        random.seed(0)
        sample = random.sample(products, min(a.check16, len(products)))
        with Pool(a.workers) as p:
            res = p.starmap(check16, [(d, a.old) for d in sample])
        tot = sum(r[1] for r in res); bad = sum(r[2] for r in res)
        for name, n, b in res:
            if b:
                print(f"  16/64 MISMATCH {name}: {b} of {n} patches")
        print(f"check16: {len(res)} products, {tot} patches, {bad} patches differ from the old labels")
        return 1 if bad else 0

    a.out.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    with Pool(a.workers) as p:
        for i, (name, row, err) in enumerate(p.imap_unordered(
                one_product, [(d, a.out) for d in products], chunksize=2), 1):
            if err:
                errors.append(f"{name}: {err}")
            else:
                rows.append((name, *row))
            if i % 200 == 0:
                print(f"  {i}/{len(products)} products", flush=True)
    import pandas as pd
    m = pd.DataFrame(rows, columns=["product", "patches", "dry", "flood", "pw", "mixed", "md5"])
    m.sort_values("product").to_csv(a.out / "manifest.csv", index=False)
    tot = m[["dry", "flood", "pw", "mixed"]].sum()
    print(f"products {len(m)}/{len(products)}, patches {m.patches.sum():,}; cells "
          f"dry {tot.dry:,} flood {tot.flood:,} pw {tot.pw:,} mixed {tot.mixed:,}")
    for e in errors:
        print("ERROR", e)
    if errors:
        (a.out / "INCOMPLETE").write_text("\n".join(errors))
        return 1
    (a.out / "COMPLETE").write_text(f"{len(m)} products\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
