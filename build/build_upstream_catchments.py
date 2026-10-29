#!/usr/bin/env python3
"""Upstream catchment of every patch of the final split: all land whose water flows into the patch.

Ravi, 2026-10-07 (option B): the upstream contribution of a patch is the rain that falls on ALL land
draining into the patch anywhere (the main river's basin, side streams, the slopes draining in),
summed per day. This script fixes that land once, from the drainage network alone: no flow speed,
storage or loss parameter. build_upstream_rain.py then sums the daily IMERG rain over it.

Drainage network: MERIT Hydro IHU, 30 arcsec (~1 km) D8 flow directions and basin ids (Eilander et
al., 2021, HESS 25, 5287; Zenodo record 5166932), read with pyflwdir (Deltares).

Per patch:
  S  the 30-arcsec cells whose centres lie inside the patch footprint (split bounds, patch CRS ->
     EPSG:4326); cells without a flow direction (sea) are dropped. If no centre falls inside, the
     cell containing the footprint centre is used.
  M  the cells of S whose downstream path does not pass through another cell of S. Their upstream
     sets are disjoint and together equal the union of the upstream sets of all of S, so
     catchment area = sum over M of the accumulated upstream area (km2, the cells themselves
     included), and later rain(catchment, day) = sum over M of the accumulated rain volume.
Window: every cell draining into a patch cell carries that cell's MERIT basin id (ids are per river
mouth), so the catchment lies inside the bounding box of the basins of the product's patch cells and
only that window of the global grid is read.

Output, per product, <out>/<product>.npz:
  patch_number (n,), n_cells (n,), n_outlets (n,), upstream_area_km2 (n,),
  outlets (sum n_outlets,) window-flat indices of M, offsets (n+1,),
  window (row0, col0, nrows, ncols) in the global 30-arcsec grid,
  upstream_bbox (W, S, E, N) of all cells draining into the product's patches (the rain region)
plus <out>/catchments.csv (one row per patch), <out>/basin_bbox.csv (cache), COMPLETE or INCOMPLETE.

    python build_upstream_catchments.py --split <final/split/patches.csv> --merit <merit_ihu_30s> \
        --out <final/upstream/catchments> --workers 40 [--products P1 P2 ...]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyflwdir
import rasterio
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.windows import Window

RES = 1.0 / 120.0                 # 30 arcsec
LON0, LAT0 = -180.0, 85.0         # origin of the MERIT IHU 30-arcsec grid (17,400 x 43,200)
NODATA_FD = 247                   # pyflwdir d8 nodata (sea)
REENTRY_STEPS, REENTRY_MARGIN = 30, 5   # path-following limits for M (cells)

_T: dict = {}


def basin_bboxes(basids: Path, cache: Path) -> dict:
    """Bounding box (row0, row1, col0, col1) of every MERIT basin id, computed once and cached."""
    if cache.exists():
        d = pd.read_csv(cache)
        return {int(r.basid): (int(r.r0), int(r.r1), int(r.c0), int(r.c1)) for r in d.itertuples()}
    parts = []
    with rasterio.open(basids) as s:
        H, W = s.shape
        for r0 in range(0, H, 200):
            a = s.read(1, window=Window(0, r0, W, min(200, H - r0)))
            rr, cc = np.nonzero(a)
            if len(rr):
                parts.append(pd.DataFrame({'basid': a[rr, cc], 'r': rr + r0, 'c': cc})
                             .groupby('basid').agg(r0=('r', 'min'), r1=('r', 'max'), c0=('c', 'min'), c1=('c', 'max')))
    d = pd.concat(parts).groupby(level=0).agg(r0=('r0', 'min'), r1=('r1', 'max'), c0=('c0', 'min'), c1=('c1', 'max'))
    d.index.name = 'basid'
    d.reset_index().to_csv(cache, index=False)
    return {int(i): (int(x.r0), int(x.r1), int(x.c0), int(x.c1)) for i, x in d.iterrows()}


def patch_cells(crs: str, b) -> tuple[np.ndarray, np.ndarray]:
    """Global (row, col) of the 30-arcsec cells whose centres lie inside the patch footprint."""
    t = _T.get(crs) or _T.setdefault(crs, Transformer.from_crs(crs, 'EPSG:4326', always_xy=True))
    lon, lat = t.transform([b[0], b[2], b[2], b[0]], [b[1], b[1], b[3], b[3]])
    poly = shapely.Polygon(zip(lon, lat))
    W, S, E, N = poly.bounds
    r0, r1 = int(np.floor((LAT0 - N) / RES)), int(np.floor((LAT0 - S) / RES))
    c0, c1 = int(np.floor((W - LON0) / RES)), int(np.floor((E - LON0) / RES))
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    inside = shapely.contains_xy(poly, LON0 + (cc.ravel() + 0.5) * RES, LAT0 - (rr.ravel() + 0.5) * RES)
    if not inside.any():
        c = poly.centroid
        return (np.array([int(np.floor((LAT0 - c.y) / RES))]), np.array([int(np.floor((c.x - LON0) / RES))]))
    return rr.ravel()[inside], cc.ravel()[inside]


def outlets_of(S: np.ndarray, idxs_ds: np.ndarray, ncols: int) -> np.ndarray:
    """M: cells of S whose downstream path does not pass through another cell of S.

    The path is followed until it is REENTRY_MARGIN cells outside the bounding box of S (or for
    REENTRY_STEPS steps, or to a pit), which is far beyond any loop back into a 2.56 km patch.
    """
    Sset = set(int(i) for i in S)
    r, c = S // ncols, S % ncols
    rlo, rhi, clo, chi = r.min() - REENTRY_MARGIN, r.max() + REENTRY_MARGIN, c.min() - REENTRY_MARGIN, c.max() + REENTRY_MARGIN
    keep = []
    for s in Sset:
        cur, hit = s, False
        for _ in range(REENTRY_STEPS):
            nxt = int(idxs_ds[cur])
            if nxt == cur:                      # pit / river mouth
                break
            if nxt in Sset:
                hit = True
                break
            rr, cc = divmod(nxt, ncols)
            if not (rlo <= rr <= rhi and clo <= cc <= chi):
                break
            cur = nxt
        if not hit:
            keep.append(s)
    return np.array(sorted(keep), dtype=np.int64)


def build_one(args):
    product, g, merit, out, bb = args
    try:
        t0 = time.time()
        cells = [patch_cells(crs, b) for crs, b in zip(g.crs, g[['bounds_minx', 'bounds_miny', 'bounds_maxx', 'bounds_maxy']].to_numpy())]
        allr = np.concatenate([c[0] for c in cells]); allc = np.concatenate([c[1] for c in cells])
        with rasterio.open(merit / '30sec_basids.tif') as s:
            w = Window(int(allc.min()), int(allr.min()), int(allc.max() - allc.min() + 1), int(allr.max() - allr.min() + 1))
            bas = s.read(1, window=w)[allr - allr.min(), allc - allc.min()]
        ids = np.unique(bas[bas > 0])
        n = len(g)
        res = dict(patch_number=g.patch_number.to_numpy(np.int32), n_cells=np.zeros(n, np.int32),
                   n_outlets=np.zeros(n, np.int32), upstream_area_km2=np.zeros(n, np.float64))
        if not len(ids):                                     # every patch cell at sea
            np.savez_compressed(out / f'{product}.npz', **res, outlets=np.zeros(0, np.int64),
                                offsets=np.zeros(n + 1, np.int64), window=np.zeros(4, np.int64),
                                upstream_bbox=np.full(4, np.nan))
            return product, 'ok-sea', time.time() - t0
        boxes = np.array([bb[int(i)] for i in ids])
        r0 = max(int(min(boxes[:, 0].min(), allr.min())) - 1, 0); r1 = min(int(max(boxes[:, 1].max(), allr.max())) + 1, 17399)
        c0 = max(int(min(boxes[:, 2].min(), allc.min())) - 1, 0); c1 = min(int(max(boxes[:, 3].max(), allc.max())) + 1, 43199)
        nr, nc = r1 - r0 + 1, c1 - c0 + 1
        with rasterio.open(merit / '30sec_flwdir.tif') as s:
            fd = s.read(1, window=Window(c0, r0, nc, nr))
        tr = Affine(RES, 0.0, LON0 + c0 * RES, 0.0, -RES, LAT0 - r0 * RES)
        flw = pyflwdir.from_array(fd, ftype='d8', transform=tr, latlon=True)
        upa = flw.upstream_area(unit='km2').ravel()
        ids_ds = flw.idxs_ds
        outl, offs = [], [0]
        for k, (rr, cc) in enumerate(cells):
            S = (rr - r0) * nc + (cc - c0)
            S = S[fd.ravel()[S] != NODATA_FD]
            res['n_cells'][k] = len(S)
            M = outlets_of(S, ids_ds, nc) if len(S) else np.zeros(0, np.int64)
            res['n_outlets'][k] = len(M)
            res['upstream_area_km2'][k] = float(upa[M].sum()) if len(M) else 0.0
            outl.append(M); offs.append(offs[-1] + len(M))
        outl = np.concatenate(outl) if outl else np.zeros(0, np.int64)
        if len(outl):
            lab = flw.basins(idxs=np.unique(outl))
            rr, cc = np.nonzero(lab)
            bbox = np.array([LON0 + (c0 + cc.min()) * RES, LAT0 - (r0 + rr.max() + 1) * RES,
                             LON0 + (c0 + cc.max() + 1) * RES, LAT0 - (r0 + rr.min()) * RES])
        else:
            bbox = np.full(4, np.nan)
        np.savez_compressed(out / f'{product}.npz', **res, outlets=outl, offsets=np.array(offs, np.int64),
                            window=np.array([r0, c0, nr, nc], np.int64), upstream_bbox=bbox)
        return product, 'ok', time.time() - t0
    except Exception:
        return product, 'FAILED ' + traceback.format_exc()[-600:], 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, type=Path)
    ap.add_argument('--merit', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--products', nargs='*', default=None, help='only these products (test runs)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.split, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_miny',
                                       'bounds_maxx', 'bounds_maxy'])
    if a.products:
        df = df[df.folder_name.isin(a.products)]
    t0 = time.time()
    bb = basin_bboxes(a.merit / '30sec_basids.tif', a.out.parent / 'basin_bbox.csv')
    print(f'basin boxes: {len(bb):,} basins ({time.time() - t0:.0f}s)', flush=True)
    jobs = [(p, g.sort_values('patch_number').reset_index(drop=True), a.merit, a.out, bb)
            for p, g in df.groupby('folder_name')]
    fails, done = [], 0
    with Pool(a.workers) as pool:
        for p, status, sec in pool.imap_unordered(build_one, jobs):
            done += 1
            if not status.startswith('ok'):
                fails.append((p, status)); print(f'FAILED {p}: {status}', flush=True)
            if done % 100 == 0 or done == len(jobs):
                print(f'{done:,}/{len(jobs):,} products, {len(fails)} failed, {time.time() - t0:.0f}s', flush=True)
    rows = []
    for p, _ in df.groupby('folder_name'):
        f = a.out / f'{p}.npz'
        if f.exists():
            z = np.load(f)
            rows.append(pd.DataFrame({'folder_name': p, 'patch_number': z['patch_number'], 'n_cells': z['n_cells'],
                                      'n_outlets': z['n_outlets'], 'upstream_area_km2': z['upstream_area_km2']}))
    if rows and not a.products:
        pd.concat(rows).to_csv(a.out / 'catchments.csv', index=False)
    if fails:
        (a.out / 'INCOMPLETE').write_text('\n'.join(f'{p}\t{s}' for p, s in fails))
        return 1
    if not a.products:
        (a.out / 'COMPLETE').write_text(f'{len(jobs)} products, {len(df)} patches\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
