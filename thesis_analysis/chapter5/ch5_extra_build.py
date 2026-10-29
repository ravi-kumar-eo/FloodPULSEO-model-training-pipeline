"""Per-patch source table of the added Chapter 5 analyses (12 October 2026), all 38 fits, test subset.

Runs locally on one core (nice), reading only saved outputs:
  inference_final/patches.csv           TP, FP, FN per patch at the fit's validation-selected global threshold
  inference_final/preds/test__prob/labels  probabilities and labels, for the probability sums per patch
  experiments/metadata/splits/patches.csv  patch footprints (centre of the patch for the weather cell)
  experiments/metadata/weather/<product>.npz  exported IMERG precipitation and SMAP soil moisture grids, 30 days
                                        ending on the day before the mapping date (dates checked per product)
  released_events_metadata.csv          mapping date and resolution class of each product

Output: plot/data/tables/extra_patches_test.parquet, one row per test patch (reference row order) with
  covariates  folder_name, patch_number, aoi, ground_unit, basin_regime, continent, climate, resolution_class,
              mapping_date, n_scored, n_flood, P30 (30-day local rain, mm), P7, Pmax_lag (days before the mapping
              date of the wettest day), SM1 (soil moisture of day -1), SM30 (30-day mean soil moisture)
  per fit     <fit>_tp, _fp, _fn, _spd (sum of p over scored dry cells), _spf (sum of p over flood cells)
Fit keys: <family>_<config>_s<seed>, config in t0, t3, t7, t15, t30, noup, nosum, sumonly, scratch.
Every fit's TP/FP/FN totals are checked against its metrics.csv (test, full rows, global rule).

    nice -n 19 python3 chapters/chapter5/plot/scripts/ch5_extra_build.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402

META = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata')
CATALOGUE = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/metadata/released_events_metadata.csv')
OUT = Path(__file__).resolve().parents[1] / 'data/tables'
CONFIGS = [('t0', 't0', None, (1, 2, 3)), ('t3', 't3', None, (1, 2, 3)), ('t7', 't7', None, (1, 2, 3)),
           ('t15', 't15', None, (1, 2, 3)), ('t30', 't30', None, (1, 2, 3)), ('noup', 't30', 'noup', (1,)),
           ('nosum', 't30', 'nosum', (1,)), ('sumonly', 't30', 'sumonly', (1,)), ('scratch', 't30', 'scratch', (1,))]


def fits():
    for f in cd.FAMILIES:
        for key, w, v, seeds in CONFIGS:
            for s in seeds:
                yield f'{f}_{key}_s{s}', f, w, v, s


def weather(ix):
    """Local precipitation and soil moisture of the cell containing each patch centre."""
    pt = pd.read_csv(META / 'splits/patches.csv', usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx',
                                                          'bounds_miny', 'bounds_maxx', 'bounds_maxy'])
    g = ix[['folder_name', 'patch_number']].merge(pt, on=['folder_name', 'patch_number'], how='left', validate='1:1')
    mdate = pd.read_csv(CATALOGUE, usecols=['folder_name', 'mapping_date']).set_index('folder_name').mapping_date
    out = {k: np.full(len(g), np.nan) for k in ('P30', 'P7', 'Pmax_lag', 'SM1', 'SM30')}
    for prod, ii in g.groupby('folder_name').indices.items():
        f = META / 'weather' / f'{prod}.npz'
        if not f.exists():
            continue
        z = np.load(f)
        m = pd.to_datetime(mdate[prod])
        for key in ('precip_dates', 'sm_dates'):
            d = pd.to_datetime([str(x) for x in z[key]])
            if not (len(d) == 30 and d.max() == m - pd.Timedelta(days=1) and d.min() == m - pd.Timedelta(days=30)):
                raise ValueError(f'{prod}: {key} do not end on the day before the mapping date')
        h = g.iloc[ii]
        tr = Transformer.from_crs(h.crs.iloc[0], 'EPSG:4326', always_xy=True)
        lon, lat = tr.transform(((h.bounds_minx + h.bounds_maxx) / 2).to_numpy(), ((h.bounds_miny + h.bounds_maxy) / 2).to_numpy())
        for var, tkey in (('precip', 'precip_transform'), ('sm', 'sm_transform')):
            a, _, c, _, e, fy = z[tkey]
            col = np.floor((lon - c) / a).astype(int)
            row = np.floor((lat - fy) / e).astype(int)
            G = z[var].astype(np.float64)
            ok = (row >= 0) & (row < G.shape[1]) & (col >= 0) & (col < G.shape[2])
            S = np.full((30, len(ii)), np.nan)
            S[:, ok] = G[:, row[ok], col[ok]]
            if var == 'precip':
                S[~np.isfinite(S) | (S < 0)] = np.nan
                out['P30'][ii] = np.nansum(S, 0)
                out['P7'][ii] = np.nansum(S[-7:], 0)
                has = np.isfinite(S).any(0) & (np.nanmax(np.nan_to_num(S, nan=-1), 0) > 0)
                lag = 30 - np.nanargmax(np.nan_to_num(S, nan=-1), 0)
                out['Pmax_lag'][ii] = np.where(has, lag, np.nan)
            else:
                S[~np.isfinite(S) | (S < 0) | (S > 1)] = np.nan
                out['SM1'][ii] = S[-1]
                with np.errstate(all='ignore'):
                    out['SM30'][ii] = np.nanmean(S, 0)
    return out


def main():
    ts = cd.test_set()
    ix = ts.index.copy()
    cat = pd.read_csv(CATALOGUE, usecols=['folder_name', 'resolution_class', 'mapping_date'])
    df = ix.merge(cat, on='folder_name', how='left', validate='m:1')
    Y = np.load(cd.run_dir('unet', 't0', 1) / 'inference_final/preds/test__labels.i8.npy')
    keep, truth = Y >= 0, Y == 1
    df['n_scored'] = keep.sum((1, 2))
    df['n_flood'] = truth.sum((1, 2))
    for k, v in weather(ix).items():
        df[k] = v
    print('weather done', flush=True)
    for key, f, w, v, s in fits():
        d = cd.run_dir(f, w, s, v)
        c = cd.counts(f, w, s, v)
        m = pd.read_csv(d / 'inference_final/metrics.csv')
        m = m[(m.cohort == 'test') & (m.rows == 'full') & (m.rule == 'global')].iloc[0]
        if (c.tp.sum(), c.fp.sum(), c.fn.sum()) != (m.tp, m.fp, m.fn):
            raise SystemExit(f'{d}: counts differ from metrics.csv')
        pdir = d / 'inference_final/preds'
        if not np.array_equal(np.load(pdir / 'test__labels.i8.npy', mmap_mode='r')[::53], Y[::53]):
            raise SystemExit(f'{d}: labels differ')
        P = np.load(pdir / 'test__prob.f16.npy').astype(np.float32)
        df[f'{key}_tp'], df[f'{key}_fp'], df[f'{key}_fn'] = c.tp.to_numpy(), c.fp.to_numpy(), c.fn.to_numpy()
        df[f'{key}_spd'] = np.where(keep & ~truth, P, 0).sum((1, 2))
        df[f'{key}_spf'] = np.where(truth, P, 0).sum((1, 2))
        del P
        print(key, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / 'extra_patches_test.parquet')
    print(f'{len(df):,} patches written', flush=True)


if __name__ == '__main__':
    main()
