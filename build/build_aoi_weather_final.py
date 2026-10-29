#!/usr/bin/env python3
"""Build the AOI-weather files of the final programme (METHOD_PLAN_RERUN.md, D9, §4.6).

For every mapping product, from `pipeline/data/GEE_exports_31day/<EMSR>/<product>/`:
  Precipitation_<start>_<end>.tif   31 daily bands, IMERG, mm/day, EPSG:4326, 0.1 deg
  SoilMoisture_<start>_<end>.tif    31 daily bands, SMAP L4 daily mean, m3/m3, ~0.095 deg
each band described by its date. The 30 bands dated mapping_date-30 ... mapping_date-1 are
selected BY DATE (day 0, the mapping date, is dropped), checked to be consecutive, and stored at
native resolution with their geotransforms. Missing values become NaN.

Output <out>/<product>.npz:
  precip (30, Hp, Wp) float32, precip_transform (6,), precip_dates (30,)
  sm     (30, Hs, Ws) float32, sm_transform (6,),     sm_dates (30,)
  mapping_date, footprint_bounds_4326 (W,S,E,N of the product's patch footprint)
plus <out>/manifest.csv and COMPLETE (or INCOMPLETE listing every failure; exit 1).

    python build_aoi_weather_final.py --data <pipeline/data> --out <dir> --workers 40
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

NODATA = 9999.0


def read_var(folder: Path, key: str, mapping_date: dt.date):
    files = sorted(glob.glob(str(folder / f'{key}_*.tif')))
    if len(files) != 1:
        raise ValueError(f'{key}: expected exactly one file, found {len(files)}')
    with rasterio.open(files[0]) as s:
        dates = [dt.date.fromisoformat(d) for d in s.descriptions]
        want = [mapping_date - dt.timedelta(days=k) for k in range(30, 0, -1)]  # -30 ... -1
        idx = [dates.index(d) for d in want]                                    # raises if missing
        arr = s.read([i + 1 for i in idx]).astype(np.float32)
        if s.nodata is not None:
            arr[arr == s.nodata] = np.nan
        arr[arr >= NODATA] = np.nan
        tr = np.asarray(list(s.transform)[:6], dtype=np.float64)
        if str(s.crs) != 'EPSG:4326' or tr[4] >= 0:
            raise ValueError(f'{key}: expected north-up EPSG:4326, got {s.crs} {tr}')
    return arr, tr, np.asarray([d.isoformat() for d in want])


def build_one(args):
    folder, mapping_date, bounds, out = args
    name = Path(folder).name
    try:
        p, ptr, pd_ = read_var(Path(folder), 'Precipitation', mapping_date)
        s, st, sd = read_var(Path(folder), 'SoilMoisture', mapping_date)
        if np.isnan(bounds).any():
            raise ValueError('missing footprint bounds')
        dst = Path(out) / f'{name}.npz'
        part = dst.with_suffix('.part.npz')
        np.savez_compressed(part, precip=p, precip_transform=ptr, precip_dates=pd_,
                            sm=s, sm_transform=st, sm_dates=sd,
                            mapping_date=mapping_date.isoformat(), footprint_bounds_4326=bounds)
        part.replace(dst)
        return name, (p.shape[1], p.shape[2], s.shape[1], s.shape[2],
                      float(np.nanmax(p)) if np.isfinite(p).any() else np.nan,
                      float(np.isnan(p).mean()), float(np.isnan(s).mean()),
                      hashlib.md5(dst.read_bytes()).hexdigest()), None
    except Exception as e:                                    # noqa: BLE001
        return name, None, f'{type(e).__name__}: {e}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path, help='pipeline/data directory')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=40)
    a = ap.parse_args()

    ev = pd.read_csv(a.data / 'metadata' / 'released_events_metadata.csv', usecols=['folder_name', 'mapping_date'])
    pm = pd.read_csv(a.data / 'metadata' / 'released_patches_metadata.csv',
                     usecols=['folder_name', 'crs', 'bounds_minx', 'bounds_miny', 'bounds_maxx', 'bounds_maxy'])
    fp = pm.groupby('folder_name').agg(crs=('crs', 'first'), x0=('bounds_minx', 'min'), y0=('bounds_miny', 'min'),
                                       x1=('bounds_maxx', 'max'), y1=('bounds_maxy', 'max'))
    items = []
    for r in ev.itertuples():
        folder = a.data / 'GEE_exports_31day' / r.folder_name.split('_')[0] / r.folder_name
        if r.folder_name in fp.index:
            f = fp.loc[r.folder_name]
            b = np.asarray(transform_bounds(f.crs, 'EPSG:4326', f.x0, f.y0, f.x1, f.y1), dtype=np.float64)
        else:
            b = np.asarray([np.nan] * 4)
        items.append((str(folder), dt.date.fromisoformat(r.mapping_date), b, str(a.out)))
    a.out.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    with Pool(a.workers) as pool:
        for i, (name, row, err) in enumerate(pool.imap_unordered(build_one, items, chunksize=4), 1):
            (errors.append(f'{name}: {err}') if err else rows.append((name, *row)))
            if i % 300 == 0:
                print(f'  {i}/{len(items)}', flush=True)
    m = pd.DataFrame(rows, columns=['product', 'precip_h', 'precip_w', 'sm_h', 'sm_w', 'precip_max',
                                    'precip_nan_share', 'sm_nan_share', 'md5'])
    m.sort_values('product').to_csv(a.out / 'manifest.csv', index=False)
    print(f'products {len(m)}/{len(items)}; precip grid H x W median {m.precip_h.median():.0f} x '
          f'{m.precip_w.median():.0f} (min {m.precip_h.min()} x {m.precip_w.min()}); sm grid median '
          f'{m.sm_h.median():.0f} x {m.sm_w.median():.0f}; precip max {m.precip_max.max():.1f} mm/day')
    for e in errors:
        print('ERROR', e)
    if errors:
        (a.out / 'INCOMPLETE').write_text('\n'.join(errors))
        return 1
    (a.out / 'COMPLETE').write_text(f'{len(m)} products\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
