#!/usr/bin/env python3
"""Gate tests of the final programme (METHOD_PLAN_RERUN.md v3, G3-G5). No training.

G3 loader    shapes, finiteness and value ranges of fine/med/coarse/weather/label on real rows;
             label values in {-1, 0, 1}; mixed_as respected; weather window dated to day -1.
G4 cache     eval-cache rows are bit-identical to the uncached path.
G5 identity  a T30 model warm-started from a T0 checkpoint reproduces the T0 output exactly at
             initialisation (float32, TF32 off), and two different weather records give the same
             output; for both families.

    python gate_final.py --family unet --n 64
"""
from __future__ import annotations

import argparse, json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

tf.config.experimental.enable_tensor_float_32_execution(False)

from flood.paths import METADATA                                     # noqa: E402
from flood.data.dataset_final import FinalFloodPatches, load_transforms  # noqa: E402
from flood.training.core_final import build_model, resolve_patches_dir  # noqa: E402
from flood.training import phases                                    # noqa: E402

FAIL = []


def check(ok, msg):
    print(('  PASS ' if ok else '  FAIL ') + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def cfg_of(name):
    d = Path(__file__).resolve().parents[1] / 'configs'
    c = yaml.safe_load(open(d / '_common.yaml')); c.update(yaml.safe_load(open(d / name)))
    c['patches_dir'] = resolve_patches_dir(c)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='unet', choices=['unet', 'vit'])
    ap.add_argument('--n', type=int, default=64)
    a = ap.parse_args()
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA) / 'final'
    t = load_transforms(md / 'input_transforms.json')
    c0, c30 = cfg_of(f't0_{a.family}_s1.yaml'), cfg_of(f't30_{a.family}_s1.yaml')
    kw = dict(patches_csv=md / 'split' / 'patches.csv', patches_dir=c0['patches_dir'],
              labels_dir=md / 'labels_80m_cls', flood_flags_csv=md / 'split' / 'flood80_flags.csv',
              transforms=t, aoi_weather_dir=md / 'aoi_weather')

    print('G3 loader')
    fp = FinalFloodPatches(cohort='train', temporal_days=30, **kw)
    fp.preload_labels()
    rng = np.random.default_rng(0)
    fl = np.flatnonzero(fp.has_flood); idx = np.r_[rng.choice(fl, a.n // 2, replace=False),
                                                    rng.choice(len(fp), a.n // 2, replace=False)]
    rows = [fp.load(int(i)) for i in idx]
    fine, med, coarse, w, lab = (np.stack([r[k] for r in rows]) for k in range(5))
    check(fine.shape[1:] == (256, 256, 2) and med.shape[1:] == (32, 32, 5) and coarse.shape[1:] == (16, 16, 2)
          and w.shape[1:] == (30, 11, 11, 8) and lab.shape[1:] == (32, 32, 1), 'shapes')
    check(all(np.isfinite(x).all() for x in (fine, med, coarse, w)), 'all finite')
    check(fine.min() >= 0 and fine.max() <= 1, f'NDVI/NDBI in [0,1] ({fine.min():.3f}..{fine.max():.3f})')
    for k, nm in enumerate(['elev', 'flow_sin', 'flow_cos', 'uda', 'hand']):
        print(f'     {nm:9s} min {med[..., k].min():.4f} max {med[..., k].max():.4f} mean {med[..., k].mean():.4f}')
    check(med[..., :4].min() >= 0 and med[..., :4].max() <= 1, 'elev/flow/uda in [0,1]')
    check(med[..., 4].min() >= 0, f'HAND >= 0 (max {med[..., 4].max():.3f}, >1 allowed: not clipped)')
    check(coarse.min() >= 0 and coarse.max() <= 1, 'clay/sand in [0,1]')
    check(set(np.unique(lab)) <= {-1, 0, 1}, f'labels {sorted(np.unique(lab))}')
    check((lab[: a.n // 2] == 1).any(axis=(1, 2, 3)).all(), 'flood-bearing rows carry class 1')
    pv, sv = w[..., 1], w[..., 5]
    check(set(np.unique(pv)) <= {0, 1} and set(np.unique(sv)) <= {0, 1}, 'validity is binary')
    print(f'     valid share precip {pv.mean():.3f} sm {sv.mean():.3f}; precip max {w[..., 0].max():.3f} '
          f'(x{t["precip_div"]:.0f} mm/day); sm range {w[..., 4][sv > 0].min():.3f}..{w[..., 4][sv > 0].max():.3f}')
    check(np.all(w[..., 0][pv == 0] == 0) and np.all(w[..., 4][sv == 0] == 0), 'invalid cells are 0')
    # centre cell offsets small (< one cell, ~0.1 deg ~ 11 km)
    check(np.abs(w[:, 0, 5, 5, 2]).max() < 0.12 and np.abs(w[:, 0, 5, 5, 3]).max() < 0.12,
          'centre cell is the patch cell (offset < 12 km)')
    evf = Path(os.environ.get('FLOOD_EVENTS_CSV') or
               Path(c0['patches_dir']).parents[1] / 'metadata' / 'released_events_metadata.csv')
    ev = pd.read_csv(evf, usecols=['folder_name', 'mapping_date']).set_index('folder_name').mapping_date
    p = fp.df.folder_name.iat[int(idx[0])]
    with np.load(md / 'aoi_weather' / f'{p}.npz') as z:
        last = str(z['precip_dates'][-1]); first = str(z['precip_dates'][0]); smd = str(z['sm_dates'][-1])
    md_ = pd.Timestamp(ev[p]); check(pd.Timestamp(last) == md_ - pd.Timedelta(days=1) and smd == last
                                     and pd.Timestamp(first) == md_ - pd.Timedelta(days=30),
                                     f'{p}: window {first}..{last}, mapping date {ev[p]}')
    fp3 = FinalFloodPatches(cohort='train', temporal_days=3, **kw)
    r3 = fp3.load(int(idx[0])); check(np.array_equal(r3[3], rows[0][3][-3:]), 'T3 = last 3 days of T30')
    for mode, want in (('flood', 1), ('dry', 0)):
        fpm = FinalFloodPatches(cohort='train', temporal_days=0, mixed_as=mode, **kw); fpm.preload_labels(verbose=False)
        n3 = int((fpm._flat != fp._flat).sum())
        print(f'     mixed_as={mode}: {n3:,} cells differ from mixed_as=pw')
    fpm = FinalFloodPatches(cohort='train', temporal_days=0, mixed_as='pw', **kw)

    print('G4 cache')
    sub = FinalFloodPatches(cohort='val', temporal_days=30, **kw)
    sub.df = sub.df.iloc[:200].reset_index(drop=True); sub.has_flood = sub.has_flood[:200]; sub._flat = None
    sub._centres()
    ref = [sub._load_uncached(i) for i in range(200)]
    sub.enable_eval_cache(workers=8)
    got = [sub.load(i) for i in range(200)]
    check(all(all(np.array_equal(x, y) for x, y in zip(r, g)) for r, g in zip(ref, got)), 'cache == uncached (200 rows)')
    for s in sub._shm.values():
        s.close(); s.unlink()

    print(f'G5 identity ({a.family}, float32)')
    from scripts.train import FAMILIES
    bb = FAMILIES[a.family][0]
    c0 = dict(c0); c30 = dict(c30)
    m0 = build_model(bb, c0)
    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / 't0.weights.h5'
        # perturb the T0 weights so the check is not passed by an all-default network
        for v in m0.trainable_variables:
            v.assign(v + tf.random.normal(v.shape, stddev=1e-3, dtype=v.dtype))
        m0.save_weights(str(ck))
        m30 = build_model(bb, c30)
        twin = dict(c30); twin['temporal_days'] = 0
        phases.load_static_checkpoint(m30, lambda: build_model(bb, twin), ck)
    x3 = (tf.constant(fine[:8]), tf.constant(med[:8]), tf.constant(coarse[:8]))
    p0, t0 = m0(x3, training=False)
    pa, ta = m30(x3 + (tf.constant(w[:8]),), training=False)
    pb, tb = m30(x3 + (tf.constant(w[8:16]),), training=False)
    d0 = float(tf.reduce_max(tf.abs(p0 - pa))); dt_ = float(tf.reduce_max(tf.abs(t0 - ta)))
    # float32 noise floor: the SAME T0 network on the same 8 rows inside a batch of 8*T rows
    # (U-TAE runs the trunk on B*T rows, and cuDNN may pick another algorithm per batch size)
    T = int(c30['temporal_days'])
    xb = tuple(tf.repeat(x, T, axis=0) for x in x3)
    pbig = m0(xb, training=False)[0][::T]
    floor = float(tf.reduce_max(tf.abs(p0 - pbig)))
    print(f'     noise floor (T0, batch 8 vs {8 * T}): {floor:.2e}')
    dw = float(tf.reduce_max(tf.abs(pa - pb)))
    check(max(d0, dt_) <= max(1e-6, 2 * floor), f'T30 at init == T0 within float32 noise: max|dprob| {d0:.2e}, max|dtau| {dt_:.2e}')
    check(dw < 1e-6, f'two weather records identical at init: {dw:.2e}')
    nw = sum(int(np.prod(v.shape)) for v in m30.trainable_variables) - sum(int(np.prod(v.shape)) for v in m0.trainable_variables)
    print(f'     weather-path parameters: {nw:,}')

    print('\nRESULT:', 'ALL PASS' if not FAIL else f'{len(FAIL)} FAIL: {FAIL}')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
