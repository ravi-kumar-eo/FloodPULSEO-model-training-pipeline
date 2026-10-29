#!/usr/bin/env python3
"""Independent verification of the final data (terrain, soil grid, labels) on a random sample.

Written from the specification, NOT from the build scripts, so a shared bug cannot pass:
  terrain   for every 80 m cell centre of the patch, the MERIT pixel that contains it (looked up
            directly in the GEE MERIT.tif) must give the stored UDA, HAND and flow sin/cos
            (nearest); elevation is compared with a bilinear sample (cubic expected, so close
            but not equal); geometry of input_80m / input_160m equals input_10m's footprint
  merit     the flow-direction value set of every sampled MERIT file (integer D8 codes only
            means GEE exported it with nearest-neighbour)
  labels    class map recomputed from the 10 m flood mask and permanent-water band
            (2 if nW>=32, else 1 if nF>=32, else 3 if nU>=32, else 0) must equal the npz

    python verify_final_data.py --n-products 250 --per-product 3 --workers 40
"""
from __future__ import annotations

import argparse, math, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_xy

R = Path('/home/ravi/shared_data/global_flood_dataset')
PT = R / 'pipeline/data/patches/patches_tif'
GEE = R / 'pipeline/data/GEE_exports'
LAB = R / 'ravi/experiments/metadata/final/labels_80m_cls'
ND = 9999.0
D8 = {1: 0, 2: 45, 4: 90, 8: 135, 16: 180, 32: 225, 64: 270, 128: 315}


def bilinear(arr, fr, fc):
    r0 = np.floor(fr - 0.5).astype(int); c0 = np.floor(fc - 0.5).astype(int)
    dr = fr - 0.5 - r0; dc = fc - 0.5 - c0
    H, W = arr.shape
    r0 = np.clip(r0, 0, H - 2); c0 = np.clip(c0, 0, W - 2)
    a, b = arr[r0, c0], arr[r0, c0 + 1]; c, d = arr[r0 + 1, c0], arr[r0 + 1, c0 + 1]
    return a * (1 - dr) * (1 - dc) + b * (1 - dr) * dc + c * dr * (1 - dc) + d * dr * dc


def check_product(args):
    folder, patches = args
    out = Counter(); notes = []
    d = PT / folder.split('_')[0] / folder
    mp = GEE / folder.split('_')[0] / folder / 'MERIT.tif'
    try:
        with rasterio.open(mp) as m:
            M = m.read().astype(np.float64); mcrs, mtr, mnd = m.crs, m.transform, m.nodata
        dirs = np.unique(M[1][np.isfinite(M[1])])
        nonint = dirs[(dirs != np.round(dirs))]
        out['merit_files'] += 1
        out['merit_nonint_dir_files'] += int(len(nonint) > 0)
        out['merit_4326'] += int(str(mcrs) == 'EPSG:4326')
        for pn in patches:
            stem = d / f'patch_{int(pn):04d}'
            with rasterio.open(f'{stem}_input_10m.tif') as s:
                b10, crs10 = s.bounds, s.crs; pw = s.read(5)
            with rasterio.open(f'{stem}_input_80m.tif') as s:
                b80, crs80, t80 = s.bounds, s.crs, s.transform; T = s.read().astype(np.float64)
            with rasterio.open(f'{stem}_input_160m.tif') as s:
                b160, crs160 = s.bounds, s.crs
            with rasterio.open(f'{stem}_flood_mask.tif') as s:
                fm = s.read(1)
            out['patches'] += 1
            geo_ok = (crs80 == crs10 == crs160 and
                      max(abs(b80[i] - b10[i]) for i in range(4)) < 0.01 and
                      max(abs(b160[i] - b10[i]) for i in range(4)) < 0.01 and abs(t80.a - 80) < 1e-6)
            out['geometry_bad'] += int(not geo_ok)
            if not geo_ok:
                notes.append(f'{folder}/{pn}: geometry 10m {tuple(b10)} 80m {tuple(b80)} 160m {tuple(b160)}')
            # 80 m cell centres -> MERIT pixel
            cc, rr = np.meshgrid(np.arange(32) + 0.5, np.arange(32) + 0.5)
            xs, ys = t80 * (cc.ravel(), rr.ravel())
            if mcrs != crs80:
                xs, ys = warp_xy(crs80, mcrs, list(xs), list(ys))
            inv = ~mtr
            fc, fr = inv * (np.asarray(xs), np.asarray(ys))
            fc, fr = np.asarray(fc), np.asarray(fr)
            ci, ri = np.floor(fc).astype(int), np.floor(fr).astype(int)
            inside = (ri >= 0) & (ri < M.shape[1]) & (ci >= 0) & (ci < M.shape[2])
            ri_, ci_ = np.clip(ri, 0, M.shape[1] - 1), np.clip(ci, 0, M.shape[2] - 1)
            src = M[:, ri_, ci_]                                       # (4, 1024)
            srcnd = ~inside | ((src[0] == mnd) if mnd is not None else False)
            stored = T.reshape(5, -1)
            for name, si, ti in (('uda', 2, 3), ('hand', 3, 4)):
                ok = ~srcnd & (stored[ti] != ND)
                eq = np.isclose(src[si][ok], stored[ti][ok], rtol=0, atol=1e-4)
                out[f'{name}_cells'] += int(ok.sum()); out[f'{name}_mismatch'] += int((~eq).sum())
                out[f'{name}_nd_where_src_valid'] += int((~srcnd & (stored[ti] == ND) & (src[si] != (mnd if mnd is not None else -1e30))).sum())
                # tie tolerance: a cell centre within 1e-3 MERIT pixel of a pixel edge
                edge = (np.minimum(fc - np.floor(fc), np.ceil(fc) - fc) < 1e-3) | (np.minimum(fr - np.floor(fr), np.ceil(fr) - fr) < 1e-3)
                out[f'{name}_mismatch_not_edge'] += int((~eq & ~edge[ok]).sum())
                # diagnose: distance of the mismatched centres to the nearest MERIT pixel edge, and
                # whether the stored value is one of the 8 neighbouring MERIT pixels
                bad = np.flatnonzero(ok)[~eq]
                if len(bad):
                    tag = 'geo' if str(mcrs) == 'EPSG:4326' else 'utm'
                    out[f'{name}_mismatch_{tag}'] += len(bad)
                    de = np.minimum.reduce([fc[bad] - np.floor(fc[bad]), np.ceil(fc[bad]) - fc[bad],
                                            fr[bad] - np.floor(fr[bad]), np.ceil(fr[bad]) - fr[bad]])
                    out[f'{name}_mismatch_edge_lt0.1px'] += int((de < 0.1).sum())
                    out[f'{name}_mismatch_edge_lt0.25px'] += int((de < 0.25).sum())
                    for k in bad:
                        nb = M[si, max(ri_[k] - 1, 0):ri_[k] + 2, max(ci_[k] - 1, 0):ci_[k] + 2]
                        out[f'{name}_mismatch_is_neighbour'] += int(np.isclose(nb, stored[ti][k], atol=1e-4).any())
            # flow direction
            code = src[1]
            exp_s = np.array([math.sin(math.radians(D8[int(c)])) if c in D8 else ND for c in code])
            exp_c = np.array([math.cos(math.radians(D8[int(c)])) if c in D8 else ND for c in code])
            ok = ~srcnd
            out['flow_cells'] += int(ok.sum())
            out['flow_mismatch'] += int((~(np.isclose(exp_s[ok], stored[1][ok], atol=1e-5) &
                                           np.isclose(exp_c[ok], stored[2][ok], atol=1e-5))).sum())
            # elevation vs bilinear
            ok = ~srcnd & (stored[0] != ND)
            bl = bilinear(M[0], fr, fc)
            dz = np.abs(bl[ok] - stored[0][ok])
            out['elev_cells'] += int(ok.sum())
            out['elev_absdiff_gt5m'] += int((dz > 5).sum()); out['elev_absdiff_gt20m'] += int((dz > 20).sum())
            out['_elev_sum_absdiff'] += float(dz.sum())
            # labels
            W = (pw == 1); F = (fm == 1) & ~W; U = (fm == 1) | W
            blk = lambda a: a.reshape(32, 8, 32, 8).sum((1, 3))
            nF, nW, nU = blk(F), blk(W), blk(U)
            cls = np.where(nW >= 32, 2, np.where(nF >= 32, 1, np.where(nU >= 32, 3, 0)))
            with np.load(LAB / f'{folder}.npz') as z:
                got = z[str(int(pn))]
            out['label_patches'] += 1
            out['label_cells_differ'] += int((got != cls).sum())
            out['label_values_seen_' + ''.join(map(str, sorted(np.unique(fm).astype(int).tolist())))[:12]] += 1
    except Exception as e:                                             # noqa: BLE001
        out['errors'] += 1; notes.append(f'{folder}: {type(e).__name__}: {e}')
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-products', type=int, default=250)
    ap.add_argument('--per-product', type=int, default=3)
    ap.add_argument('--workers', type=int, default=40)
    a = ap.parse_args()
    df = pd.read_csv(R / 'ravi/experiments/metadata/final/split/patches.csv',
                     usecols=['folder_name', 'patch_number', 'crs'], keep_default_na=False)
    rng = np.random.default_rng(7)
    prods = df.folder_name.unique()
    geo = [p for p in prods if (GEE / p.split('_')[0] / p / 'MERIT.tif').exists()]
    with rasterio.Env():
        pass
    pick = list(rng.choice(prods, size=min(a.n_products, len(prods)), replace=False))
    # always include every product whose MERIT export is geographic (EPSG:4326)
    for p in prods:
        if p.startswith('EMSR85') and p not in pick:
            pick.append(p)
    jobs = []
    for p in pick:
        pn = df.patch_number[df.folder_name == p].to_numpy()
        jobs.append((p, rng.choice(pn, size=min(a.per_product, len(pn)), replace=False)))
    tot = Counter(); notes = []
    with Pool(a.workers) as pool:
        for c, n in pool.imap_unordered(check_product, jobs):
            tot.update(c); notes += n
    for k in sorted(tot):
        print(f'{k:40s} {tot[k]}')
    if tot['elev_cells']:
        print(f'{"elev_mean_absdiff_vs_bilinear_m":40s} {tot["_elev_sum_absdiff"] / tot["elev_cells"]:.3f}')
    for n in notes[:30]:
        print('NOTE', n)


if __name__ == '__main__':
    sys.exit(main())
