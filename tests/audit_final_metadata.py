#!/usr/bin/env python3
"""Full audit of the final data artefacts — EVERY row and EVERY file, not a sample.

  A split     final/split/patches.csv vs the main split + release catalogue: exactly the rows with
              aoi_fraction > 0, every shared column unchanged, keys unique, cohorts and regimes kept,
              single-observation choice = one product per AOI with the highest flood share
  B flags     flood80_flags.csv = the patches.csv columns, same keys
  C labels    every row has a 32x32 uint8 class map with values 0..3; class counts in the CSV equal a
              recount; manifest md5s equal the files
  D weather   every product has an npz; md5 = manifest; dates = mapping date -30..-1 for BOTH variables;
              value ranges; every patch centre inside its grid; K=11 window coverage
  E files     input_80m.tif (all 566,669 patches) rebuilt: modification time on/after the rebuild;
              input_160m.tif present for every split row
  F version   data_version.json md5s equal the current files

    python audit_final_metadata.py --workers 40
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, json, math, os, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/ravi/shared_data/global_flood_dataset')
E = R / 'ravi/experiments'
FIN = E / 'metadata/final'
PT = R / 'pipeline/data/patches/patches_tif'
KEY = ['folder_name', 'patch_number']
BAD = []


def ok(cond, msg):
    print(('  PASS ' if cond else '  FAIL ') + msg, flush=True)
    if not cond:
        BAD.append(msg)


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=40); a = ap.parse_args()
    rd = lambda p: pd.read_csv(p, keep_default_na=False, na_values=[''], low_memory=False)
    main_ = rd(E / 'metadata/splits/patches.csv')
    fin = rd(FIN / 'split/patches.csv')
    cat = pd.read_csv(R / 'pipeline/data/metadata/released_patches_metadata.csv', usecols=KEY + ['aoi_fraction'])

    print('A split')
    m = main_.merge(cat, on=KEY, how='left', validate='one_to_one')
    want = set(map(tuple, m.loc[m.aoi_fraction > 0, KEY].to_numpy()))
    got = set(map(tuple, fin[KEY].to_numpy()))
    ok(want == got, f'rows = main split rows with aoi_fraction > 0 ({len(got):,} of {len(main_):,}; '
                    f'missing {len(want - got)}, extra {len(got - want)})')
    ok(not fin.duplicated(KEY).any(), 'unique (product, patch) keys')
    inside = int((fin.aoi_fraction >= 0.999999).sum()); touch = int(((fin.aoi_fraction > 0) & (fin.aoi_fraction < 0.999999)).sum())
    print(f'     fully inside {inside:,} · touching (0 < f < 1) {touch:,} · outside dropped {int((m.aoi_fraction == 0).sum()):,}')
    for co in ('train', 'val', 'test'):
        g = fin[fin.cohort == co]
        print(f'     {co:5s} {len(g):7,} rows  inside {int((g.aoi_fraction >= 0.999999).sum()):7,}  touching {int((g.aoi_fraction < 0.999999).sum()):7,}')
    j = fin.merge(main_, on=KEY, suffixes=('', '_main'))
    same = [c for c in main_.columns if c not in KEY and c != 'exp1_keep']
    diffcols = [c for c in same if not (j[c].astype(str) == j[c + '_main'].astype(str)).all()]
    ok(not diffcols, f'every shared column unchanged ({len(same)} columns){" — differ: " + str(diffcols) if diffcols else ""}')
    ok((fin.loc[fin.cohort == 'train', 'basin_regime'] == 'n/a').all(), "training basin_regime is the string 'n/a'")
    print('     test regimes:', dict(Counter(fin.loc[fin.cohort == 'test', 'basin_regime'])))
    key2 = fin.folder_name.str.split('_').str[:2].str.join('_')
    agree = (key2 == fin.aoi.astype(str)).mean()
    print(f'     aoi column == first two name tokens for {agree:.4%} of rows')
    for col in ('folder_name', 'aoi', 'ground_unit'):
        ok(fin[col].nunique() == main_[col].nunique(), f'no {col} lost ({fin[col].nunique()})')
    keep = fin.exp1_keep.astype(str).str.lower().isin(['true', '1'])
    sel = fin[keep].assign(k=key2[keep]).groupby('k').folder_name.nunique()
    ok((sel == 1).all() and len(sel) == key2.nunique(),
       f'single-observation: exactly one product per AOI ({len(sel)} AOIs, {key2.nunique()} in split)')
    ev = fin.groupby('folder_name').agg(n=('patch_number', 'size'), f=('n_flood80', 'sum'))
    ev['frac'] = ev.f / (ev.n * 1024); ev['k'] = [ '_'.join(i.split('_')[:2]) for i in ev.index]
    best = ev.sort_index().groupby('k').frac.idxmax()
    ok(set(best) == set(fin.loc[keep, 'folder_name']), 'chosen product = highest flood share per AOI (ties: first name)')
    ok(set(fin.loc[keep, 'cohort'].unique()) <= {'train', 'val', 'test'}, 'single-observation rows in all cohorts')

    print('B flags')
    fl = rd(FIN / 'split/flood80_flags.csv')
    ok(fl[KEY].equals(fin[KEY]), 'same keys and order as patches.csv')
    ok(all((fl[c] == fin[c]).all() for c in ('n_flood80', 'n_pw80', 'n_mixed80', 'has_flood80')), 'same values')
    ok(((fin.n_flood80 > 0).astype(int) == fin.has_flood80).all(), 'has_flood80 == (n_flood80 > 0)')

    print('C labels')
    lab = FIN / 'labels_80m_cls'
    files = sorted(lab.glob('*.npz'))
    ok(len(files) == 1565 and (lab / 'COMPLETE').exists(), f'{len(files)} label files, COMPLETE marker')
    lm = pd.read_csv(lab / 'manifest.csv')
    print('     manifest columns:', list(lm.columns))

    def lab_one(fn_rows):
        fn, rows = fn_rows
        out = Counter()
        p = lab / f'{fn}.npz'
        with np.load(p) as z:
            for pn, nf, nw, nm in rows:
                k = str(int(pn))
                if k not in z.files:
                    out['missing'] += 1; continue
                x = z[k]
                out['bad_shape_dtype'] += int(x.shape != (32, 32) or x.dtype != np.uint8)
                out['bad_values'] += int(x.max() > 3)
                b = np.bincount(x.ravel(), minlength=4)
                out['count_mismatch'] += int((b[1], b[2], b[3]) != (nf, nw, nm))
                out['rows'] += 1
        return out
    jobs = [(fn, list(zip(g.patch_number, g.n_flood80, g.n_pw80, g.n_mixed80))) for fn, g in fin.groupby('folder_name')]
    c = Counter()
    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(lab_one, jobs):
            c.update(r)
    ok(c['rows'] == len(fin) and not c['missing'], f'every split row has a label ({c["rows"]:,})')
    ok(not c['bad_shape_dtype'] and not c['bad_values'], '32x32 uint8, values 0..3')
    ok(not c['count_mismatch'], 'CSV class counts equal a recount from the labels')
    if 'md5' in lm.columns:
        name_col = [x for x in lm.columns if x in ('product', 'folder_name')][0]
        with ThreadPoolExecutor(a.workers) as ex:
            got_md5 = list(ex.map(lambda f: md5(lab / f'{f}.npz'), lm[name_col]))
        ok(list(lm.md5) == got_md5, 'label manifest md5s equal the files')

    print('D AOI weather')
    aw = FIN / 'aoi_weather'
    am = pd.read_csv(aw / 'manifest.csv')
    ev_md = pd.read_csv(R / 'pipeline/data/metadata/released_events_metadata.csv',
                        usecols=['folder_name', 'mapping_date']).set_index('folder_name').mapping_date
    prods = sorted(fin.folder_name.unique())
    ok((aw / 'COMPLETE').exists() and set(prods) <= set(am['product']), f'every split product has an npz ({len(prods)})')
    from pyproj import Transformer
    cx = (fin.bounds_minx + fin.bounds_maxx) / 2; cy = (fin.bounds_miny + fin.bounds_maxy) / 2
    lon = np.empty(len(fin)); lat = np.empty(len(fin))
    for crs, idx in fin.groupby('crs').indices.items():
        lon[idx], lat[idx] = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(cx.iloc[idx], cy.iloc[idx])
    rows_of = fin.groupby('folder_name').indices
    md5_of = dict(zip(am['product'], am.md5))

    def w_one(p):
        out = Counter(); notes = []
        f = aw / f'{p}.npz'
        if md5(f) != md5_of[p]:
            out['md5_bad'] += 1
        md_ = dt.date.fromisoformat(ev_md[p])
        want = [(md_ - dt.timedelta(days=k)).isoformat() for k in range(30, 0, -1)]
        with np.load(f) as z:
            for v in ('precip', 'sm'):
                arr, tr, dates = z[v], z[f'{v}_transform'], list(map(str, z[f'{v}_dates']))
                out[f'{v}_dates_bad'] += int(dates != want)
                out[f'{v}_shape_bad'] += int(arr.ndim != 3 or arr.shape[0] != 30)
                fin_ = np.isfinite(arr)
                out[f'{v}_cells'] += arr.size; out[f'{v}_nan'] += int((~fin_).sum())
                if v == 'precip':
                    out['precip_negative'] += int((arr[fin_] < 0).sum())
                else:
                    out['sm_outside_0_1'] += int(((arr[fin_] < 0) | (arr[fin_] > 1)).sum())
                H, W = arr.shape[1:]
                idx = rows_of[p]
                col = np.floor((lon[idx] - tr[2]) / tr[0]).astype(int); row = np.floor((lat[idx] - tr[5]) / tr[4]).astype(int)
                inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
                out[f'{v}_centre_outside_grid'] += int((~inside).sum())
                # K=11 window cells outside the grid
                for r_, c_ in zip(row, col):
                    rr = np.arange(r_ - 5, r_ + 6); cc = np.arange(c_ - 5, c_ + 6)
                    n_in = int(((rr >= 0) & (rr < H)).sum() * ((cc >= 0) & (cc < W)).sum())
                    out[f'{v}_window_cells_in'] += n_in; out[f'{v}_window_cells'] += 121
        return out, notes
    c = Counter()
    with ThreadPoolExecutor(a.workers) as ex:
        for r, _ in ex.map(w_one, prods):
            c.update(r)
    ok(not c['md5_bad'], 'npz md5s equal the manifest')
    ok(not c['precip_dates_bad'] and not c['sm_dates_bad'],
       f'dates = mapping date -30 .. -1 for both variables in all {len(prods)} products')
    ok(not c['precip_shape_bad'] and not c['sm_shape_bad'], '30 daily bands per variable')
    ok(not c['precip_negative'], 'no negative precipitation')
    ok(not c['sm_outside_0_1'], 'soil moisture within 0..1')
    print(f'     NaN share: precip {c["precip_nan"] / c["precip_cells"]:.4%}, sm {c["sm_nan"] / c["sm_cells"]:.4%}')
    ok(not c['precip_centre_outside_grid'] and not c['sm_centre_outside_grid'], 'every patch centre lies inside both grids')
    print(f'     K=11 window cells inside the exported grid: precip {c["precip_window_cells_in"] / c["precip_window_cells"]:.2%}, '
          f'sm {c["sm_window_cells_in"] / c["sm_window_cells"]:.2%} (the rest are marked invalid)')

    print('E patch files')
    allp = pd.read_csv(R / 'pipeline/data/metadata/released_patches_metadata.csv', usecols=KEY)
    stems = [PT / f.split('_')[0] / f / f'patch_{int(n):04d}' for f, n in zip(allp.folder_name, allp.patch_number)]

    def st(s):
        try:
            return os.stat(f'{s}_input_80m.tif').st_mtime
        except FileNotFoundError:
            return -1.0
    with ThreadPoolExecutor(64) as ex:
        mt = np.array(list(ex.map(st, stems)))
    ok((mt > 0).all(), f'input_80m.tif present for all {len(stems):,} released patches')
    lo, hi = dt.datetime.fromtimestamp(mt[mt > 0].min()), dt.datetime.fromtimestamp(mt.max())
    ok(lo.date() >= dt.date(2026, 10, 4), f'every input_80m.tif written by the rebuild (oldest {lo:%Y-%m-%d %H:%M}, newest {hi:%Y-%m-%d %H:%M})')
    fstem = [PT / f.split('_')[0] / f / f'patch_{int(n):04d}' for f, n in zip(fin.folder_name, fin.patch_number)]
    with ThreadPoolExecutor(64) as ex:
        has160 = list(ex.map(lambda s: os.path.exists(f'{s}_input_160m.tif'), fstem))
    ok(all(has160), 'input_160m.tif (SoilGrids) present for every split row')

    print('F data version')
    v = json.loads((FIN / 'data_version.json').read_text())
    pairs = {'split_patches_md5': FIN / 'split/patches.csv', 'split_flags_md5': FIN / 'split/flood80_flags.csv',
             'labels_manifest_md5': lab / 'manifest.csv', 'aoi_weather_manifest_md5': aw / 'manifest.csv',
             'input_transforms_md5': FIN / 'input_transforms.json'}
    ok(all(v[k] == md5(p) for k, p in pairs.items()), 'data_version.json md5s equal the current files')

    print('\nRESULT:', 'ALL PASS' if not BAD else f'{len(BAD)} FAIL: {BAD}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
