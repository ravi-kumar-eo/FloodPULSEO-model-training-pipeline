#!/usr/bin/env python3
"""Per-patch daily weather of the final programme: local rain + soil moisture and UPSTREAM rain.

Ravi, 2026-10-07. For every patch of the final split and each of the 30 days before the mapping date
(day -30 ... day -1, the days of the existing weather):
  local_precip   mm/day, IMERG V07 cell containing the patch centre
  local_sm       m3/m3,  SMAP L4 cell containing the patch centre
  local_valid    1 where both are valid
  upstream_precip  mm/day, mean daily rain over all land whose water flows into the patch
                   (build_upstream_catchments.py, MERIT Hydro IHU 30 arcsec): the rain on every
                   30-arcsec cell of the catchment, weighted by cell area, summed with pyflwdir's
                   accuflux. No routing, speed, storage or loss parameter.
  upstream_area_km2  catchment area (the full MERIT catchment)
  upstream_coverage  share of the catchment area that lies inside the product's rain grid

RAIN SOURCE: the existing per-product exports (metadata/final/aoi_weather/<product>.npz, the
GEE_exports_31day grids: the AOI box + 0.6 deg on every side, native IMERG 0.1 deg / SMAP grids).
Where a catchment reaches beyond that box (large rivers), upstream_precip is the mean over the part
inside it and upstream_coverage records the share; upstream_area_km2 stays the full catchment.
Missing rain cells on a day are left out of that day's mean (area-weighted over valid cells).

Output <out>/<product>.npz with the arrays above (n patches x 30 days), patch_number, dates;
<out>/weather_summary.csv (one row per patch), COMPLETE or INCOMPLETE.

    python build_upstream_weather.py --final <metadata/final> --merit <merit_ihu_30s> --out <final/upstream/weather>
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyflwdir
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.windows import Window

RES, LON0, LAT0 = 1.0 / 120.0, -180.0, 85.0      # MERIT IHU 30-arcsec grid
R_EARTH = 6371.0072                              # km, authalic radius
_T: dict = {}


def centres(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Patch-centre lon/lat from the split bounds (patch CRS -> EPSG:4326), as the AOI loader does."""
    lon = np.empty(len(g)); lat = np.empty(len(g))
    cx = (g.bounds_minx.to_numpy() + g.bounds_maxx.to_numpy()) / 2
    cy = (g.bounds_miny.to_numpy() + g.bounds_maxy.to_numpy()) / 2
    for crs, idx in g.groupby('crs').indices.items():
        t = _T.get(crs) or _T.setdefault(crs, Transformer.from_crs(crs, 'EPSG:4326', always_xy=True))
        lon[idx], lat[idx] = t.transform(cx[idx], cy[idx])
    return lon, lat


def cell_of(arr, tr, lon, lat):
    """Values (30, n) of the grid cells containing the points, NaN outside the grid."""
    dx, x0, dy, y0 = tr[0], tr[2], tr[4], tr[5]
    col = np.floor((lon - x0) / dx).astype(int); row = np.floor((lat - y0) / dy).astype(int)
    H, W = arr.shape[1:]
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    out = np.full((arr.shape[0], len(lon)), np.nan, np.float32)
    out[:, ok] = arr[:, row[ok], col[ok]]
    return out


def build_one(args):
    product, g, final, merit, out = args
    try:
        t0 = time.time()
        w = np.load(final / 'aoi_weather' / f'{product}.npz')
        P, ptr, S, str_ = w['precip'].astype(np.float32), w['precip_transform'], w['sm'].astype(np.float32), w['sm_transform']
        P[~np.isfinite(P) | (P < 0)] = np.nan
        S[~np.isfinite(S) | (S < 0) | (S > 1)] = np.nan
        nd = P.shape[0]
        lon, lat = centres(g)
        lp, ls = cell_of(P, ptr, lon, lat), cell_of(S, str_, lon, lat)
        lvalid = np.isfinite(lp) & np.isfinite(ls)
        c = np.load(final / 'upstream' / 'catchments' / f'{product}.npz')
        if not np.array_equal(c['patch_number'], g.patch_number.to_numpy()):
            raise ValueError('catchment patch order differs from the split')
        n = len(g)
        up = np.full((n, nd), np.nan, np.float32)
        cov = np.zeros(n, np.float32)
        area = c['upstream_area_km2'].astype(np.float64)
        r0, c0, nr, nc = (int(v) for v in c['window'])
        M, off = c['outlets'], c['offsets']
        if nr and len(M):
            with rasterio.open(merit / '30sec_flwdir.tif') as s:
                fd = s.read(1, window=Window(c0, r0, nc, nr))
            flw = pyflwdir.from_array(fd, ftype='d8', transform=Affine(RES, 0, LON0 + c0 * RES, 0, -RES, LAT0 - r0 * RES),
                                      latlon=True)
            lat_top = LAT0 - (r0 + np.arange(nr)) * RES
            arow = R_EARTH ** 2 * math.radians(RES) * (np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_top - RES)))
            A = np.repeat(arow[:, None], nc, axis=1).astype(np.float64)
            # every 30-arcsec cell -> the rain cell containing its centre (-1 outside the rain grid)
            lon_c = LON0 + (c0 + np.arange(nc) + 0.5) * RES; lat_c = lat_top - RES / 2
            pc = np.floor((lon_c - ptr[2]) / ptr[0]).astype(int); pr = np.floor((lat_c - ptr[5]) / ptr[4]).astype(int)
            H, W = P.shape[1:]
            okc, okr = (pc >= 0) & (pc < W), (pr >= 0) & (pr < H)
            inside = okr[:, None] & okc[None, :]
            pcc, prr = np.clip(pc, 0, W - 1), np.clip(pr, 0, H - 1)
            per_patch = [M[off[k]:off[k + 1]] for k in range(n)]
            a_in = flw.accuflux(A * inside).ravel()
            for k in range(n):
                if len(per_patch[k]) and area[k] > 0:
                    cov[k] = a_in[per_patch[k]].sum() / area[k]
            for d in range(nd):
                day = P[d][prr[:, None], pcc[None, :]]            # (nr, nc) rain of the containing cell
                good = inside & np.isfinite(day)
                vol = flw.accuflux(np.where(good, day, 0.0) * A).ravel()     # mm * km2
                aw = flw.accuflux(A * good).ravel()                          # km2 with valid rain
                for k in range(n):
                    m = per_patch[k]
                    if len(m):
                        den = aw[m].sum()
                        up[k, d] = vol[m].sum() / den if den > 0 else np.nan
        # a patch with no land cell (sea) or no valid upstream rain: its own cell stands in
        miss = ~np.isfinite(up)
        up[miss] = lp.T[miss]
        np.savez_compressed(out / f'{product}.npz', patch_number=g.patch_number.to_numpy(np.int32),
                            local_precip=lp.T.astype(np.float32), local_sm=ls.T.astype(np.float32),
                            local_valid=lvalid.T.astype(np.uint8), upstream_precip=up.astype(np.float32),
                            upstream_area_km2=area, upstream_coverage=cov, upstream_filled=miss.astype(np.uint8),
                            dates=w['precip_dates'])
        return product, 'ok', time.time() - t0
    except Exception:
        return product, 'FAILED ' + traceback.format_exc()[-800:], 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', required=True, type=Path, help='metadata/final')
    ap.add_argument('--merit', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--products', nargs='*', default=None)
    a = ap.parse_args()
    if not (a.final / 'upstream' / 'catchments' / 'COMPLETE').exists():
        raise SystemExit('catchments are not COMPLETE')
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.final / 'split' / 'patches.csv', usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx',
                                                                  'bounds_miny', 'bounds_maxx', 'bounds_maxy'])
    if a.products:
        df = df[df.folder_name.isin(a.products)]
    jobs = [(p, g.sort_values('patch_number').reset_index(drop=True), a.final, a.merit, a.out)
            for p, g in df.groupby('folder_name')]
    t0, fails, done = time.time(), [], 0
    with Pool(a.workers) as pool:
        for p, status, sec in pool.imap_unordered(build_one, jobs):
            done += 1
            if status != 'ok':
                fails.append((p, status)); print(f'FAILED {p}: {status}', flush=True)
            if done % 100 == 0 or done == len(jobs):
                print(f'{done:,}/{len(jobs):,} products, {len(fails)} failed, {time.time() - t0:.0f}s', flush=True)
    if fails:
        (a.out / 'INCOMPLETE').write_text('\n'.join(f'{p}\t{s}' for p, s in fails))
        return 1
    if not a.products:
        rows = []
        for p, _ in df.groupby('folder_name'):
            z = np.load(a.out / f'{p}.npz')
            rows.append(pd.DataFrame({'folder_name': p, 'patch_number': z['patch_number'],
                                      'upstream_area_km2': z['upstream_area_km2'], 'upstream_coverage': z['upstream_coverage'],
                                      'local_precip_30d': np.nansum(z['local_precip'], 1),
                                      'upstream_precip_30d': np.nansum(z['upstream_precip'], 1),
                                      'local_valid_days': z['local_valid'].sum(1), 'upstream_filled_days': z['upstream_filled'].sum(1)}))
        pd.concat(rows).to_csv(a.out / 'weather_summary.csv', index=False)
        (a.out / 'COMPLETE').write_text(f'{len(jobs)} products, {len(df)} patches\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
