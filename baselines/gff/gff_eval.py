#!/usr/bin/env python3
"""Reproduction check: score GFF predictions with GFF's own rule (evaluate.py / gff.evaluation).

Per fold, pooled over every visit tile of every test region (torchmetrics multiclass F1, average none):
  all labels   WorldCover permanent water (class 80) set to ignore in prediction and label, then
               classes > 2 ignored and flood (2) merged into water (1)       -> F1-B, F1-W
  Kuro Siwo    only the hand-labelled regions; Kuro Siwo permanent water (label 1) ignored instead
Prints per fold and the mean +- sd over folds, the form of the paper's Table 2
(U-TAE F1-W 0.57 +- 0.07, Kuro Siwo F1-W 0.48 +- 0.05; MaxViT U-Net 0.53 +- 0.06 / 0.49 +- 0.09).

    python gff_eval.py --data <gff data> --pred <out>/<model> [--folds 0 1 2 3 4]
"""
import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds


def counts(p, t):
    m = (p >= 0) & (t >= 0)
    p, t = p[m], t[m]
    return np.array([np.sum((p == 1) & (t == 1)), np.sum((p == 1) & (t == 0)), np.sum((p == 0) & (t == 1)),
                     np.sum((p == 0) & (t == 0))], np.int64)


def f1s(c):
    tp, fp, fn, tn = c
    return 2 * tn / max(2 * tn + fp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)   # F1-B, F1-W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    rows = []
    for k in a.folds:
        names = [l.strip() for l in open(a.data / "partitions" / f"floodmap_partition_{k}.txt") if l.strip()]
        call, cks, nroi = np.zeros(4, np.int64), np.zeros(4, np.int64), 0
        for mn in names:
            mp = a.data / "rois" / mn
            if not mp.exists():
                continue
            meta = json.load(open(mp))
            stem = Path(meta["floodmap"]).stem
            pf = a.pred / f"fold{k}" / f"{stem}-pred.tif"
            if not pf.exists():
                continue
            nroi += 1
            ks = re.match(r"\d{3}-\d{1,2}-", mn) is not None
            tiles = gpd.read_file(a.data / "rois" / meta["visit_tiles"])
            with rasterio.open(a.data / "rois" / meta["floodmap"]) as L, rasterio.open(pf) as P, \
                    rasterio.open(a.data / "worldcover" / f"{stem}-worldcover.tif") as W:
                for g in tiles.geometry:
                    w = from_bounds(*g.bounds, L.transform).round_offsets().round_lengths()
                    t = L.read(1, window=w).astype(np.int16); p = P.read(1, window=w).astype(np.int16)
                    cov = W.read(1, window=w)
                    for mode in (("wc",) + (("ks",) if ks else ())):
                        tt, pp = t.copy(), p.copy()
                        if mode == "wc":
                            blk = cov == 80
                        else:
                            blk = t == 1
                        tt[blk] = -100; pp[blk] = -100
                        for x in (tt, pp):
                            x[x > 2] = -100; x[x == 2] = 1
                        c = counts(pp, tt)
                        if mode == "wc":
                            call += c
                        else:
                            cks += c
        fb, fw = f1s(call); kb, kw = f1s(cks)
        rows.append(dict(fold=k, regions=nroi, f1_b=fb, f1_w=fw, f1=(fb + fw) / 2, ks_f1_b=kb, ks_f1_w=kw, ks_f1=(kb + kw) / 2))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    print("mean +- sd over folds:", {c: f"{df[c].mean():.3f} +- {df[c].std():.3f}" for c in ["f1", "f1_b", "f1_w", "ks_f1", "ks_f1_w"]})


if __name__ == "__main__":
    main()
