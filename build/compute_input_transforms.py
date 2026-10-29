#!/usr/bin/env python3
"""Input transforms of the final programme, measured on the TRAINING rows only (METHOD_PLAN_RERUN.md, C2).

Reads every training patch of `metadata/final/split/patches.csv` (all acquisitions) and the AOI
weather of the training products, then writes:

  metadata/final/input_transforms.json
      elev_min, elev_max       min / max of valid elevation cells (m)
      clay_median, sand_median SoilGrids medians of valid cells (%), the NoData fill
      uda_div = 7              log10(1 + UDA) / 7; asserted >= the training maximum
      hand_div = 300           HAND / 300; the training maximum is reported, never clipped
      precip_div               next multiple of 100 >= training maximum (mm/day); no clipping
      + the measured statistics behind each value
  metadata/final/data_version.json
      md5 of the split, flags, label manifest, AOI-weather manifest and transforms, and the
      terrain rebuild flag; every final run records the md5 of this file.

Fails (exit 1, nothing written) on a negative UDA or HAND value, a soil value outside 0-100,
an empty soil distribution, or a log10(1+UDA) maximum above uda_div.

    python compute_input_transforms.py --experiments <E> --patches <patches_tif> --workers 40
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ND = 9999.0
SOIL_BINS = np.linspace(0, 100, 10001)          # 0.01 % resolution for the medians


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def valid(a):
    # exact sentinel: UDA (km2) is legitimately above 9999 on large rivers
    return np.isfinite(a) & (a != ND)


def one(stem):
    out = {'emin': np.inf, 'emax': -np.inf, 'umax': -np.inf, 'umin': np.inf, 'hmax': -np.inf,
           'hmin': np.inf, 'hgt300': 0, 'u_gt9999': 0, 'n80': 0, 'u_neg': 0, 'h_neg': 0,
           'clay': np.zeros(len(SOIL_BINS) - 1, np.int64), 'sand': np.zeros(len(SOIL_BINS) - 1, np.int64),
           'soil_bad': 0, 'soil_nd': 0, 'n160': 0, 'err': None}
    try:
        with rasterio.open(f'{stem}_input_80m.tif') as s:
            e, _, _, u, h = s.read().astype(np.float64)
        ve, vu, vh = valid(e), valid(u), valid(h)
        if ve.any():
            out['emin'], out['emax'] = float(e[ve].min()), float(e[ve].max())
        if vu.any():
            out['umin'], out['umax'] = float(u[vu].min()), float(u[vu].max())
            out['u_neg'] = int((u[vu] < 0).sum())
        if vh.any():
            out['hmin'], out['hmax'] = float(h[vh].min()), float(h[vh].max())
            out['h_neg'] = int((h[vh] < 0).sum()); out['hgt300'] = int((h[vh] > 300).sum())
            out['u_gt9999'] = int((u[vu] > ND).sum())
        out['n80'] = int(ve.sum())
        with rasterio.open(f'{stem}_input_160m.tif') as s:
            if tuple(s.descriptions) != ('Clay', 'Sand'):
                raise ValueError(f'soil bands {s.descriptions}')
            c, sd = s.read().astype(np.float64)
        for k, a in (('clay', c), ('sand', sd)):
            v = valid(a)
            out['soil_nd'] += int((~v).sum())
            bad = v & ((a < 0) | (a > 100))
            out['soil_bad'] += int(bad.sum())
            out[k] += np.histogram(a[v & ~bad], SOIL_BINS)[0]
        out['n160'] = c.size
    except Exception as ex:                                      # noqa: BLE001
        out['err'] = f'{stem}: {type(ex).__name__}: {ex}'
    return out


def hist_median(h):
    c = np.cumsum(h)
    i = int(np.searchsorted(c, c[-1] / 2.0))
    return float((SOIL_BINS[i] + SOIL_BINS[i + 1]) / 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', required=True, type=Path)
    ap.add_argument('--patches', required=True, type=Path)
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--uda-div', type=float, default=7.0)
    ap.add_argument('--hand-div', type=float, default=300.0)
    a = ap.parse_args()

    fin = a.experiments / 'metadata' / 'final'
    split, lab, aoi = fin / 'split', fin / 'labels_80m_cls', fin / 'aoi_weather'
    for d in (split, lab, aoi):
        if not (d / 'COMPLETE').exists():
            sys.exit(f'{d} is not COMPLETE')
    df = pd.read_csv(split / 'patches.csv', usecols=['folder_name', 'patch_number', 'cohort'],
                     keep_default_na=False)
    tr = df[df.cohort == 'train']
    stems = [str(a.patches / f.split('_')[0] / f / f'patch_{int(p):04d}')
             for f, p in zip(tr.folder_name, tr.patch_number)]
    print(f'training rows {len(stems):,} in {tr.folder_name.nunique()} products', flush=True)

    tot = None
    errs = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, stems, chunksize=64), 1):
            if r['err']:
                errs.append(r['err']); continue
            if tot is None:
                tot = r; continue
            for k in ('emin', 'umin', 'hmin'):
                tot[k] = min(tot[k], r[k])
            for k in ('emax', 'umax', 'hmax'):
                tot[k] = max(tot[k], r[k])
            for k in ('hgt300', 'u_gt9999', 'n80', 'u_neg', 'h_neg', 'soil_bad', 'soil_nd', 'n160', 'clay', 'sand'):
                tot[k] = tot[k] + r[k]
            if i % 20000 == 0:
                print(f'  {i:,}/{len(stems):,}', flush=True)
    for e in errs[:20]:
        print('ERROR', e)
    if errs:
        print(f'{len(errs)} unreadable patches; nothing written'); return 1

    pmax, pnan = -np.inf, []
    for p in sorted(tr.folder_name.unique()):
        with np.load(aoi / f'{p}.npz') as z:
            x = z['precip']
            pnan.append(float(np.isnan(x).mean()))
            if np.isfinite(x).any():
                pmax = max(pmax, float(np.nanmax(x)))
    precip_div = float(math.ceil(pmax / 100.0) * 100.0)

    problems = []
    if tot['u_neg'] or tot['h_neg']:
        problems.append(f'negative UDA cells {tot["u_neg"]}, negative HAND cells {tot["h_neg"]}')
    if tot['soil_bad']:
        problems.append(f'{tot["soil_bad"]} soil cells outside 0-100 %')
    if tot['clay'].sum() == 0 or tot['sand'].sum() == 0:
        problems.append('empty soil distribution')
    if math.log10(1 + tot['umax']) > a.uda_div:
        problems.append(f'log10(1+UDA max {tot["umax"]:.4g}) > uda_div {a.uda_div}')
    for p_ in problems:
        print('PROBLEM', p_)
    if problems:
        return 1

    t = {'elev_min': tot['emin'], 'elev_max': tot['emax'],
         'clay_median': hist_median(tot['clay']), 'sand_median': hist_median(tot['sand']),
         'uda_div': a.uda_div, 'hand_div': a.hand_div, 'precip_div': precip_div,
         'measured': {'training_rows': len(stems), 'training_products': int(tr.folder_name.nunique()),
                      'valid_elevation_cells': int(tot['n80']),
                      'uda_min': tot['umin'], 'uda_max': tot['umax'],
                      'uda_cells_above_9999': int(tot['u_gt9999']),
                      'log10_1p_uda_max': math.log10(1 + tot['umax']),
                      'hand_min': tot['hmin'], 'hand_max': tot['hmax'],
                      'hand_cells_above_300': int(tot['hgt300']),
                      'soil_nodata_share': tot['soil_nd'] / max(2 * tot['n160'], 1),
                      'precip_max_mm_day': pmax, 'precip_nan_share_mean': float(np.mean(pnan))},
         'formulas': {'ndvi_ndbi': '(x+1)/2, invalid -> physical 0',
                      'elevation': '(x-elev_min)/(elev_max-elev_min), invalid -> 0',
                      'flow_sin_cos': '(x+1)/2, invalid -> physical 0',
                      'uda': 'log10(1+x)/uda_div', 'hand': 'x/hand_div',
                      'clay_sand': '%/100, invalid -> training median',
                      'precip': 'mm/day / precip_div, no clipping', 'sm': 'm3/m3 unchanged'},
         'created': dt.datetime.now().isoformat(timespec='seconds')}
    tf_path = fin / 'input_transforms.json'
    tf_path.write_text(json.dumps(t, indent=2))
    terrain_flag = Path.home() / 'final_build' / 'logs' / 'terrain_done.flag'
    v = {'name': 'final_v1',
         'split_patches_md5': md5(split / 'patches.csv'), 'split_flags_md5': md5(split / 'flood80_flags.csv'),
         'labels_manifest_md5': md5(lab / 'manifest.csv'), 'aoi_weather_manifest_md5': md5(aoi / 'manifest.csv'),
         'input_transforms_md5': md5(tf_path),
         'terrain': 'input_80m.tif rebuilt in place, nearest-neighbour UDA/HAND/flow',
         'terrain_flag': terrain_flag.read_text().strip() if terrain_flag.exists() else None,
         'soil': 'input_160m.tif (SoilGrids v2.0)',
         'created': dt.datetime.now().isoformat(timespec='seconds')}
    (fin / 'data_version.json').write_text(json.dumps(v, indent=2))
    print(json.dumps(t, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
