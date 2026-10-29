#!/usr/bin/env python3
"""Score our fits and GFF's models against GFF's label on the same 80 m cells of each test site.

Per site and per patch of OUR grid (the batch's patches, 10 m UTM), nearest-neighbour onto that grid:
GFF label, GFF visit tiles (the cells GFF scores), GFF WorldCover, and each GFF model's class and
water probability. Then, at 10 m:
  valid    inside a visit tile and a usable label (Kuro Siwo class 3 and nodata 255 are ignored)
  PW       our WorldCover permanent water (input_10m band 5) OR GFF's own permanent-water rule
           (Kuro Siwo label class 1 on hand-labelled sites, WorldCover class 80 on generated ones)
  flood    Kuro Siwo: label 2; generated: label 1 or 2 (GFF's 'water'), in both cases outside PW
and per 80 m cell (64 cells of 10 m) our 32-of-64 rule (build_labels_80m.py) on the valid cells:
cells with < 32 valid are dropped; PW if nW >= 32, flood if nF >= 32, mixed if nU >= 32, else dry.
Only dry and flood cells are scored, for every model alike.
  ours     probability >= the fit's stored threshold (threshold.json, best_selected)
  GFF      flood if >= 32 of the cell's valid 10 m cells are predicted water (the label's rule);
           probability = mean water probability over the valid cells (for AUPRC)
GFF's own measure, at 10 m with GFF's own masking (their evaluate.py): pixels inside visit tiles with a
usable label, GFF's permanent water ignored (Kuro Siwo class 1 / WorldCover 80), target = flood (KS) or
water (generated); our 80 m yes/no is given to its 64 pixels. Pooled F1-W and F1-B -> <out>/gff_measure.csv.
Writes <out>/cells/<site>.npz (per cell: label, our probabilities per fit, GFF probability and yes/no
per model) and <out>/scores.csv (per site and pooled over sites: TP FP FN TN, IoU, F1, precision,
POD, POFD, FAR, miss rate, area bias, AUPRC), and prints the pooled table.

    python score_sites.py --batch <batch DATA_DIR> --ours <predict_ours out> --gff-data <gff data>
                          --gff-pred <gff_infer out> --models utae metnet --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio import features
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds


def window_read(src, bounds_src, pad=8):
    w = from_bounds(*bounds_src, src.transform).round_offsets().round_lengths()
    w = rasterio.windows.Window(w.col_off - pad, w.row_off - pad, w.width + 2 * pad, w.height + 2 * pad)
    w = w.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
    return src.read(1, window=w), src.window_transform(w)


def to_grid(arr, tr, crs, dst_tr, dst_crs, fill):
    out = np.full((256, 256), fill, arr.dtype)
    reproject(arr, out, src_transform=tr, src_crs=crs, dst_transform=dst_tr, dst_crs=dst_crs,
              resampling=Resampling.nearest, src_nodata=None, dst_nodata=fill)
    return out


def blocks(x):
    return x.reshape(32, 8, 32, 8).sum(axis=(1, 3))


def metrics(y, yes, p=None):
    tp = int(np.sum(yes & (y == 1))); fp = int(np.sum(yes & (y == 0)))
    fn = int(np.sum(~yes & (y == 1))); tn = int(np.sum(~yes & (y == 0)))
    d = lambda a, b: a / b if b else np.nan
    r = dict(cells=len(y), flood_cells=tp + fn, TP=tp, FP=fp, FN=fn, TN=tn, IoU=d(tp, tp + fp + fn),
             F1=d(2 * tp, 2 * tp + fp + fn), precision=d(tp, tp + fp), POD=d(tp, tp + fn), POFD=d(fp, fp + tn),
             FAR=d(fp, tp + fp), miss_rate=d(fn, tp + fn), area_bias=d(tp + fp, tp + fn))
    if p is not None and (tp + fn) > 0:
        o = np.argsort(-p, kind="stable"); yy = (y[o] == 1)
        prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
        r["AUPRC"] = float(np.sum(prec[yy]) / yy.sum())
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--ours", required=True, type=Path)
    ap.add_argument("--gff-data", required=True, type=Path)
    ap.add_argument("--gff-pred", required=True, type=Path)
    ap.add_argument("--models", nargs="*", default=["utae", "metnet"])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    (a.out / "cells").mkdir(parents=True, exist_ok=True)
    smap = pd.read_csv(a.batch / "gff_site_map.csv")
    fits = sorted(p.name for p in a.ours.iterdir() if (p / "prob.f16.npy").exists())
    P = {f: np.load(a.ours / f / "prob.f16.npy", mmap_mode="r") for f in fits}
    thr = {f: json.loads((a.ours / f / "threshold.json").read_text())["threshold"] for f in fits}
    idx = pd.read_csv(a.ours / fits[0] / "index.csv")
    for f in fits[1:]:
        if not pd.read_csv(a.ours / f / "index.csv")[["folder_name", "patch_number"]].equals(idx[["folder_name", "patch_number"]]):
            raise SystemExit(f"{f}: row order differs from {fits[0]}")
    rows, pixrows = [], []
    for s in smap.itertuples():
        ks = len(s.site.split("-")) == 5
        stem = Path(s.label_file).stem
        R = a.gff_data / "rois"
        tiles = gpd.read_file(R / s.visit_tiles)
        src = {"label": rasterio.open(R / s.label_file),
               "wc": rasterio.open(a.gff_data / "worldcover" / f"{stem}-worldcover.tif")}
        for m in a.models:
            for tag in ("pred", "prob"):
                src[f"{m}_{tag}"] = rasterio.open(a.gff_pred / m / f"fold{s.gff_fold}" / f"{stem}-{tag}.tif")
        lab_crs = src["label"].crs
        sub = idx[idx.folder_name == s.folder_name]
        pix = {k: np.zeros(4, np.int64) for k in [f"ours:{f}" for f in fits] + [f"gff:{m}" for m in a.models]}
        cell = {k: [] for k in ["label", "nvalid"] + [f"ours_{f}" for f in fits] +
                [f"{m}_{t}" for m in a.models for t in ("yes", "prob")]}
        pdir = a.batch / "patches" / s.folder_name.split("_")[0] / s.folder_name
        for r in sub.itertuples():
            with rasterio.open(pdir / f"patch_{int(r.patch_number):04d}_input_10m.tif") as pt:
                dtr, dcrs, pw_ours = pt.transform, pt.crs, pt.read(5) == 1
                pb = pt.bounds
            bsrc = transform_bounds(dcrs, lab_crs, *pb, densify_pts=21)
            g = {}
            for k, ds in src.items():
                arr, tr = window_read(ds, transform_bounds(dcrs, ds.crs, *pb, densify_pts=21) if ds.crs != lab_crs else bsrc)
                g[k] = to_grid(arr, tr, ds.crs, dtr, dcrs, 255)
            # visit tiles rasterised straight onto the patch grid
            vt = tiles.to_crs(dcrs)
            vt = vt[vt.intersects(box(*pb))]
            visit = features.geometry_mask(list(vt.geometry), (256, 256), dtr, invert=True) if len(vt) else np.zeros((256, 256), bool)
            lab = g["label"]
            valid = visit & (lab != 255) & (lab != 3)
            pw = pw_ours | ((lab == 1) if ks else (g["wc"] == 80))
            fl = ((lab == 2) if ks else ((lab == 1) | (lab == 2))) & ~pw
            nV, nW = blocks(valid), blocks(pw & valid)
            nF, nU = blocks(fl & valid), blocks((fl | pw) & valid)
            c = np.zeros((32, 32), np.int8)
            c[nU >= 32] = 3; c[nF >= 32] = 1; c[nW >= 32] = 2
            keep = (nV >= 32) & ((c == 0) | (c == 1))
            cell["label"].append(c[keep]); cell["nvalid"].append(nV[keep])
            for f in fits:
                cell[f"ours_{f}"].append(np.asarray(P[f][r.row_index], np.float32)[keep])
            for m in a.models:
                yes = blocks((g[f"{m}_pred"] == 1) & valid) >= 32
                pr = np.where(valid, g[f"{m}_prob"].astype(np.float32) / 250.0, 0.0)
                cell[f"{m}_yes"].append(yes[keep])
                cell[f"{m}_prob"].append((blocks(pr) / np.maximum(nV, 1))[keep])
            gpw = (lab == 1) if ks else (g["wc"] == 80)
            pv = visit & (lab != 255) & (lab != 3) & ~gpw
            tw = ((lab == 2) if ks else ((lab == 1) | (lab == 2)))[pv]
            def acc(key, yes10):
                y = yes10[pv]
                pix[key] += [np.sum(y & tw), np.sum(y & ~tw), np.sum(~y & tw), np.sum(~y & ~tw)]
            for f in fits:
                acc(f"ours:{f}", np.kron(np.asarray(P[f][r.row_index], np.float32) >= thr[f], np.ones((8, 8), bool)))
            for m in a.models:
                acc(f"gff:{m}", g[f"{m}_pred"] == 1)
        for v in src.values():
            v.close()
        cell = {k: np.concatenate(v) if v else np.zeros(0) for k, v in cell.items()}
        np.savez_compressed(a.out / "cells" / f"{s.site}.npz", **cell)
        for k, c in pix.items():
            pixrows.append(dict(site=s.site, model=k, TP=int(c[0]), FP=int(c[1]), FN=int(c[2]), TN=int(c[3])))
        y = cell["label"]
        for f in fits:
            rows.append(dict(site=s.site, model=f"ours:{f}", **metrics(y, cell[f"ours_{f}"] >= thr[f], cell[f"ours_{f}"])))
        for m in a.models:
            rows.append(dict(site=s.site, model=f"gff:{m}", **metrics(y, cell[f"{m}_yes"].astype(bool), cell[f"{m}_prob"])))
        print(f"{s.site}: {len(y):,} scored cells, {int((y == 1).sum()):,} flood", flush=True)
    df = pd.DataFrame(rows)
    pooled = []
    for m, g in df.groupby("model"):
        ys, yes, ps = [], [], []
        for site in g.site:
            z = np.load(a.out / "cells" / f"{site}.npz")
            k = m.split(":", 1)[1]
            ys.append(z["label"])
            if m.startswith("ours:"):
                ps.append(z[f"ours_{k}"]); yes.append(z[f"ours_{k}"] >= thr[k])
            else:
                ps.append(z[f"{k}_prob"]); yes.append(z[f"{k}_yes"].astype(bool))
        pooled.append(dict(site="ALL", model=m, **metrics(np.concatenate(ys), np.concatenate(yes), np.concatenate(ps))))
    df = pd.concat([df, pd.DataFrame(pooled)], ignore_index=True)
    df.to_csv(a.out / "scores.csv", index=False)
    px = pd.DataFrame(pixrows)
    px = pd.concat([px, px.groupby("model")[["TP", "FP", "FN", "TN"]].sum().reset_index().assign(site="ALL")], ignore_index=True)
    px["F1_W"] = 2 * px.TP / (2 * px.TP + px.FP + px.FN)
    px["F1_B"] = 2 * px.TN / (2 * px.TN + px.FP + px.FN)
    px["F1"] = (px.F1_W + px.F1_B) / 2
    px.to_csv(a.out / "gff_measure.csv", index=False)
    print(px[px.site == "ALL"][["model", "F1_W", "F1_B", "F1"]].round(4).to_string(index=False))
    pd.set_option("display.width", 250)
    print(df[df.site == "ALL"][["model", "cells", "flood_cells", "IoU", "F1", "precision", "POD", "POFD", "FAR",
                                 "area_bias", "AUPRC"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
