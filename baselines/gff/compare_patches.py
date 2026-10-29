#!/usr/bin/env python3
"""Check 2: do patches rebuilt by our pipeline match the released patches of the same product?

For every product folder under --new that also exists under --ref, and every patch number in both:
grid (CRS, transform, shape) and pixel values of input_10m (NDVI, NDBI, PW), input_80m (5 bands),
input_160m (2 bands) and flood_mask. Prints, per product and file: patches compared, grid
mismatches, share of identical pixels and max |difference| per band (nodata 9999 compared as values).

    python compare_patches.py --new <batch>/patches/patches_tif --ref <reference patches_tif>
"""
import argparse
import re
from pathlib import Path

import numpy as np
import rasterio

FILES = {"input_10m": [3, 4, 5], "input_80m": [1, 2, 3, 4, 5], "input_160m": [1, 2], "flood_mask": [1]}
PAT = re.compile(r"patch_(\d+)_flood_mask\.tif$")


def patches(d: Path) -> set[int]:
    return {int(PAT.search(p.name).group(1)) for p in d.iterdir() if PAT.search(p.name)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    a = ap.parse_args()
    for nd in sorted(p for p in a.new.glob("*/*") if p.is_dir()):
        rd = a.ref / nd.parent.name / nd.name
        if not rd.exists():
            print(f"{nd.name}: no reference"); continue
        pn, pr = patches(nd), patches(rd)
        both = sorted(pn & pr)
        print(f"\n{nd.name}: new {len(pn)} patches, ref {len(pr)}, both {len(both)}, "
              f"only new {len(pn - pr)}, only ref {len(pr - pn)}")
        for f, bands in FILES.items():
            grid_bad, same, tot, maxd = 0, np.zeros(len(bands)), 0, np.zeros(len(bands))
            for i in both:
                with rasterio.open(nd / f"patch_{i:04d}_{f}.tif") as x, rasterio.open(rd / f"patch_{i:04d}_{f}.tif") as y:
                    if x.crs != y.crs or x.transform != y.transform or x.shape != y.shape:
                        grid_bad += 1; continue
                    for j, b in enumerate(bands):
                        u = x.read(b).astype(np.float64); v = y.read(b).astype(np.float64)
                        same[j] += np.sum(u == v); maxd[j] = max(maxd[j], float(np.nanmax(np.abs(u - v))))
                    tot += x.shape[0] * x.shape[1]
            txt = "  ".join(f"b{b}: {100 * same[j] / max(tot, 1):6.2f}% same, max|d| {maxd[j]:.4g}"
                            for j, b in enumerate(bands))
            print(f"  {f:11s} grid mismatches {grid_bad:3d} | {txt}")


if __name__ == "__main__":
    main()
