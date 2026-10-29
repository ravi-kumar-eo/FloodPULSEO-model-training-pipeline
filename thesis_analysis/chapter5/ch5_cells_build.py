"""Per-cell and per-patch condition tables of the test subset, for the Chapter 5 analyses by input condition.
Runs ON BOLZANO from experiments/src (the loader's own TIF reader and the released patches live there):

    cd /home/ravi/shared_data/global_flood_dataset/ravi/experiments/src
    source /home/ravi/flood_envs/flood-ravi/activate_flood.sh
    nohup python3 -u /home/ravi/shared_data/global_flood_dataset/ravi/ch5_conditions/ch5_cells_build.py \
        > /home/ravi/shared_data/global_flood_dataset/ravi/ch5_conditions/build.log 2>&1 &

Rows: the 51,606 test patches in the evaluator's order (runs_final/*/inference_final/preds/test__index.csv, checked
identical for every fit). Cells: the 32 x 32 cells of 80 m of each patch, aligned with test__labels.i8.npy
(1 flood, 0 dry, -1 permanent water or mixed; scored cells are label >= 0).

Values are the physical values of the files the final loader reads (dataset_final.py, configs/_common.yaml):
  input_10m.tif  bands VV, VH, NDVI, NDBI, PW -> NDVI and NDBI, mean of the valid 10 m pixels of each 80 m cell
                 (valid: finite, not the 9999 sentinel, within [-1, 1])
  input_80m.tif  elevation (m), sin and cos of the flow direction, UDA (km2), HAND (m)
  input_160m.tif clay and sand (%, SoilGrids v2.0), each 160 m cell repeated over its 2 x 2 cells of 80 m
Invalid values are NaN here (the loader fills them for the network). Cells without a D8 flow direction (sin and cos
both 0 or invalid: the sea fill of the MERIT export, where elevation, UDA and HAND are 0) get NaN terrain.
Derived per cell: slope (degrees, centred differences of the 80 m elevation within the patch, one-sided at the
patch edge), flow-direction class 0-7 (angle of (cos, sin) in steps of 45 degrees; 255 = none) and the distance to
the nearest permanent-water or mixed cell of the same patch (in 80 m cells, Euclidean; 255 = none in the patch).

Predictions: for every finished fit of the 38, prob >= its validation-selected global threshold, the evaluator's
rule; its TP, FP and FN on the scored cells are checked against its metrics.csv (test, full rows, global rule).

Outputs in /home/ravi/shared_data/global_flood_dataset/ravi/ch5_conditions/:
  cells_test.npz        label (int8), ndvi, ndbi, elevation_m, slope_deg, uda_log10 (log10(1 + UDA km2)), hand_m,
                        clay_pct, sand_pct (float16), flowdir, dist_pw_cells (uint8); each (51606, 32, 32)
  preds_test_bits.npz   one packed boolean array per fit (np.packbits of the (51606, 32, 32) prediction)
  fits.json             per fit: run directory, threshold, verified TP, FP, FN
  patches_test.csv.gz   one row per test patch: identifiers, block (ground_unit), regime, continent, climate,
                        scored, flood and permanent-water/mixed cells, per-feature mean, median, min and max over
                        the scored cells, and TP, FP, FN of every fit
"""
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, '.')
from flood.data.dataset import _read, NODATA_SENTINEL  # noqa: E402  the loader's own TIF reader

E = Path('/home/ravi/shared_data/global_flood_dataset/ravi/experiments')
R = E / 'runs_final'
PT = Path('/home/ravi/shared_data/global_flood_dataset/pipeline/data/patches/patches_tif')
OUT = Path('/home/ravi/shared_data/global_flood_dataset/ravi/ch5_conditions')
FEATURES = ('ndvi', 'ndbi', 'elevation_m', 'slope_deg', 'uda_log10', 'hand_m', 'clay_pct', 'sand_pct')
WORKERS = 40


def fit_dirs():
    out = {}
    for f in ('unet', 'vit'):
        for s in (1, 2, 3):
            out[f'{f}_t0_s{s}'] = R / f'final_t0/seed{s}/{f}'
            out[f'{f}_t30_s{s}'] = R / f'final_t30/seed{s}/{f}'
            for t in (3, 7, 15):
                out[f'{f}_t{t}_s{s}'] = R / f'final_window/seed{s}/{f}_t{t}'
        for v in ('noup', 'nosum', 'sumonly'):
            out[f'{f}_{v}_s1'] = R / f'final_ablation/seed1/{f}_{v}'
        out[f'{f}_scratch_s1'] = R / f'final_scratch/seed1/{f}'
    return out


def bad(a):
    return ~np.isfinite(a) | (a == NODATA_SENTINEL)


def product_cells(args):
    folder, pnums = args
    n = len(pnums)
    out = {k: np.full((n, 32, 32), np.nan, np.float32) for k in FEATURES}
    fd = np.full((n, 32, 32), 255, np.uint8)
    act = folder.split('_')[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN 80 m cells in the 10 m means
        for k, p in enumerate(pnums):
            stem = PT / act / folder / f'patch_{int(p):04d}'
            d10, d80, d160 = (_read(f'{stem}_{s}.tif') for s in ('input_10m', 'input_80m', 'input_160m'))
            for name, i in (('ndvi', 2), ('ndbi', 3)):
                x = d10[..., i].astype(np.float32)
                x[bad(x) | (x < -1) | (x > 1)] = np.nan
                out[name][k] = np.nanmean(x.reshape(32, 8, 32, 8), axis=(1, 3))
            e, s, c, u, h = (d80[..., i].astype(np.float32) for i in range(5))
            d8 = ~(bad(s) | bad(c)) & ((s != 0) | (c != 0))
            e[bad(e) | ~d8] = np.nan
            u[bad(u) | (u < 0) | ~d8] = np.nan
            h[bad(h) | (h < 0) | ~d8] = np.nan
            gy, gx = np.gradient(e, 80.0)
            out['slope_deg'][k] = np.degrees(np.arctan(np.hypot(gx, gy)))
            out['elevation_m'][k], out['uda_log10'][k], out['hand_m'][k] = e, np.log10(1.0 + u), h
            ang = np.degrees(np.arctan2(np.where(d8, s, 0), np.where(d8, c, 1))) % 360
            fd[k] = np.where(d8, (np.round(ang / 45.0) % 8).astype(np.uint8), 255)
            for name, i in (('clay_pct', 0), ('sand_pct', 1)):
                x = d160[..., i].astype(np.float32)
                x[bad(x) | (x < 0) | (x > 100)] = np.nan
                out[name][k] = np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)
    return folder, out, fd


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    fits = {k: d for k, d in fit_dirs().items() if (d / 'inference_final/preds/test__prob.f16.npy').exists()}
    print(f'{len(fits)} finished fits; missing: {sorted(set(fit_dirs()) - set(fits))}', flush=True)
    idx = pd.read_csv(fits['unet_t0_s1'] / 'inference_final/preds/test__index.csv')
    for k, d in fits.items():
        j = pd.read_csv(d / 'inference_final/preds/test__index.csv', usecols=['folder_name', 'patch_number'])
        assert (j.folder_name.values == idx.folder_name.values).all() and \
               (j.patch_number.values == idx.patch_number.values).all(), k
    Y = np.load(fits['unet_t0_s1'] / 'inference_final/preds/test__labels.i8.npy')
    n = len(idx)
    assert Y.shape == (n, 32, 32)
    scored, flood = Y >= 0, Y == 1

    # features
    F = {k: np.full((n, 32, 32), np.nan, np.float32) for k in FEATURES}
    FD = np.full((n, 32, 32), 255, np.uint8)
    rows_of = idx.groupby('folder_name').indices
    tasks = [(f, idx.patch_number.values[r]) for f, r in rows_of.items()]
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, (folder, out, fd) in enumerate(ex.map(product_cells, tasks, chunksize=2)):
            r = rows_of[folder]
            for k in FEATURES:
                F[k][r] = out[k]
            FD[r] = fd
            if (i + 1) % 50 == 0:
                print(f'  features: {i + 1}/{len(tasks)} products, {time.time() - t0:.0f} s', flush=True)
    dist = np.full((n, 32, 32), 255, np.uint8)
    for i in range(n):
        pw = Y[i] == -1
        if pw.any():
            dist[i] = np.minimum(ndimage.distance_transform_edt(~pw), 254).round().astype(np.uint8)
    for k in FEATURES:
        print(f'  {k}: scored cells with a value {np.isfinite(F[k][scored]).mean():.4f}; '
              f'range {np.nanmin(F[k]):.3f} to {np.nanmax(F[k]):.3f}', flush=True)
    np.savez_compressed(OUT / 'cells_test.npz', label=Y, flowdir=FD, dist_pw_cells=dist,
                        **{k: F[k].astype(np.float16) for k in FEATURES})
    print(f'cells_test.npz written, {time.time() - t0:.0f} s', flush=True)

    # predictions of every fit, verified against the evaluator
    bits, meta = {}, {}
    pat = idx[['row_index', 'folder_name', 'patch_number', 'ground_unit', 'basin_regime', 'continent', 'climate']].copy()
    pat['scored'] = scored.sum((1, 2))
    pat['flood'] = flood.sum((1, 2))
    pat['pw_mixed'] = (Y == -1).sum((1, 2))
    for k, d in fits.items():
        thr = np.float32(json.loads((d / 'inference_final/threshold.json').read_text())['threshold'])
        P = np.load(d / 'inference_final/preds/test__prob.f16.npy').astype(np.float32) >= thr
        tp, fp, fn = (P & flood), (P & scored & ~flood), (~P & flood)
        m = pd.read_csv(d / 'inference_final/metrics.csv')
        m = m[(m.cohort == 'test') & (m.rows == 'full') & (m.rule == 'global')].iloc[0]
        got = (int(tp.sum()), int(fp.sum()), int(fn.sum()))
        assert got == (int(m.tp), int(m.fp), int(m.fn)), (k, got, (m.tp, m.fp, m.fn))
        bits[k] = np.packbits(P.ravel())
        meta[k] = {'run': str(d), 'threshold': float(thr), 'tp': got[0], 'fp': got[1], 'fn': got[2]}
        pat[f'{k}_tp'], pat[f'{k}_fp'], pat[f'{k}_fn'] = tp.sum((1, 2)), fp.sum((1, 2)), fn.sum((1, 2))
        print(f'  {k}: threshold {thr:.2f}, TP {got[0]:,} FP {got[1]:,} FN {got[2]:,} = metrics.csv', flush=True)
    np.savez_compressed(OUT / 'preds_test_bits.npz', shape=np.array([n, 32, 32]), **bits)
    (OUT / 'fits.json').write_text(json.dumps(meta, indent=1) + '\n')

    # per-patch summaries over the scored cells
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k in FEATURES:
            x = np.where(scored, F[k], np.nan).reshape(n, -1)
            pat[f'{k}_mean'], pat[f'{k}_median'] = np.nanmean(x, 1), np.nanmedian(x, 1)
            pat[f'{k}_min'], pat[f'{k}_max'] = np.nanmin(x, 1), np.nanmax(x, 1)
    pat.to_csv(OUT / 'patches_test.csv.gz', index=False)
    print(f'done, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
