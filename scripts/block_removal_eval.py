#!/usr/bin/env python3
"""Response of a fit with dynamic hydroclimatic variables to the removal of a whole block of days (pre-defence,
Ravi 17 Oct 2026: remove days -3 to -1 together, days -7 to -4 together, ... instead of one day at a time, on the full
test set instead of 2,000 patches).

    python block_removal_eval.py --run-dir <runs_final>/final_window/seed1/vit_t7 [--batch 64] [--workers 24]

Loads the fit exactly as scripts/perturb_eval.py does (same checkpoint, data version and split checks) and predicts
the TEST rows (full row set) once per block of days that the fit uses, with the local and upstream precipitation of all
days of the block set to zero (channels 0 and 3 of FinalFloodPatches._wx; the summary statistics are computed inside
the network from the daily inputs, so they follow). The fit's validation-selected global threshold is applied unchanged.

Blocks (day -1 is the day before the mapping date):  b3 = days -1..-3,  b7 = days -4..-7,  b15 = days -8..-15,
b30 = days -16..-30. A fit with T days gets every block that lies within its T days (T3: b3; T7: b3, b7; ...).

Baseline: a 'none' pass with unchanged inputs in the same run (TP, FP and FN must lie within 0.1% of
inference_final/metrics.csv, as in perturb_eval.py).

Writes <run-dir>/inference_block/ (nothing in inference_final is touched):
  patches.csv   one row per test row and pass: tp, fp, fn, scored_cells, predicted cells
  summary.csv   one row per pass: pooled TP, FP, FN, IoU, POD, POFD, predicted cells, change from 'none' in %
"""
from __future__ import annotations

import argparse, csv, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
import yaml

import evaluate as ev                                                    # noqa: E402
from flood.paths import METADATA                                         # noqa: E402
from flood.data.dataset_final import FinalFloodPatches                    # noqa: E402

PRECIP = (0, 3)                                                           # channels of local and upstream precipitation
BLOCKS = (('b3', 1, 3), ('b7', 4, 7), ('b15', 8, 15), ('b30', 16, 30))    # (name, first day back, last day back)


def zero_block(wx0, first, last):
    """Copy of the (N, 30, 5) daily inputs with the precipitation of days -first..-last set to zero (index 29 = day -1)."""
    wx = wx0.copy()
    n = wx.shape[1]
    wx[:, n - last:n - first + 1, PRECIP] = 0.0
    return wx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    run = a.run_dir
    man = json.loads((run / 'run_manifest.json').read_text())
    cfg = yaml.safe_load(open(run / 'config.yaml'))
    T = int(cfg.get('temporal_days', 0))
    if not T or cfg.get('weather_mode', 'aoi') != 'upstream':
        raise SystemExit('block removal needs a fit with upstream-mode dynamic hydroclimatic variables')
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA)
    dp = ev.data_paths(cfg, md)
    if ev.md5(dp['version']) != man['data_version_md5'] or ev.md5(dp['split'] / 'patches.csv') != man['split_md5']:
        raise SystemExit('REFUSED: data version or split differs from the one this fit trained on')
    if cfg.get('mixed_precision', 'float32') != 'float32':
        tf.keras.mixed_precision.set_global_policy(cfg['mixed_precision'])
    thr = float(json.loads((run / 'inference_final' / 'threshold.json').read_text())['threshold'])
    ref = [r for r in csv.DictReader(open(run / 'inference_final' / 'metrics.csv'))
           if r['cohort'] == 'test' and r['rows'] == 'full' and r['rule'] == 'global'][0]

    for g in tf.config.list_physical_devices('GPU'):     # take GPU memory as needed (the GPU is shared)
        tf.config.experimental.set_memory_growth(g, True)
    model = ev.build_model(ev.FAMILIES[man['family']][0], cfg)
    w = run / 'checkpoints' / man.get('selection', {}).get('checkpoint', 'best_loss.weights.h5')
    dummy = [tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2)),
             tf.zeros((1,) + ev.weather_shape(cfg))]
    model(tuple(dummy), training=False)
    model.load_weights(str(w))

    kw = ev.final_dataset_kwargs(cfg, dp)
    kw.pop('mono_only')
    if not Path(kw['patches_dir']).exists():           # fits trained on AWS record the AWS patch folder
        kw['patches_dir'] = os.environ.get('FLOOD_PATCHES_DIR',
                                           '/home/ravi/shared_data/global_flood_dataset/pipeline/data/patches/patches_tif')
        print(f"patches_dir of the config not found; using {kw['patches_dir']}", flush=True)
    fp = FinalFloodPatches(cohort='test', allow_test=True, **kw)
    df = fp.df.reset_index(drop=True)
    fp.enable_eval_cache(workers=a.workers)
    wx0 = fp._wx.copy()
    if wx0.shape[1] != 30:
        raise SystemExit(f'expected 30 stored days, got {wx0.shape[1]}')
    passes = [('none', None)] + [(name, (first, last)) for name, first, last in BLOCKS if last <= T]
    out = run / 'inference_block'
    out.mkdir(exist_ok=True)
    rows, summary, pred_none = [], [], None
    for name, blk in passes:
        fp._wx = wx0 if blk is None else zero_block(wx0, *blk)
        t0 = time.time()
        P, _, Y = ev.predict(model, fp, a.batch, a.workers)
        tp, fpos, fn, nk = ev.counts(P, Y, thr)
        keep = Y >= 0
        pred = ((P.astype(np.float32) >= thr) & keep).reshape(len(df), -1).sum(1)
        if name == 'none':                             # reproduction check against the stored evaluation
            got = (int(tp.sum()), int(fpos.sum()), int(fn.sum()))
            want = (int(ref['tp']), int(ref['fp']), int(ref['fn']))
            rel = max(abs(g - w) / max(w, 1) for g, w in zip(got, want))
            print(f'reproduction: TP/FP/FN {got} vs stored {want}, largest relative difference {rel:.2e}', flush=True)
            if rel > 1e-3:
                raise SystemExit('REFUSED: unperturbed counts differ from inference_final/metrics.csv by more than 0.1%')
            pred_none = int(pred.sum())
        Tp, F, N, S = int(tp.sum()), int(fpos.sum()), int(fn.sum()), int(nk.sum())
        summary.append({'block': name, 'days': '' if blk is None else f'-{blk[0]}..-{blk[1]}', 'T': T,
                        'tp': Tp, 'fp': F, 'fn': N, 'scored_cells': S, 'iou': Tp / (Tp + F + N),
                        'pod': Tp / (Tp + N), 'pofd': F / (S - Tp - N), 'predicted_cells': int(pred.sum()),
                        'pred_change_pct': 100 * (int(pred.sum()) / pred_none - 1), 'seconds': round(time.time() - t0)})
        print(summary[-1], flush=True)
        for i in range(len(df)):
            rows.append({'block': name, 'row_index': i, 'folder_name': df.folder_name.iat[i],
                         'patch_number': int(df.patch_number.iat[i]), 'tp': int(tp[i]), 'fp': int(fpos[i]),
                         'fn': int(fn[i]), 'scored_cells': int(nk[i]), 'predicted': int(pred[i])})
    fp._wx = wx0
    for s in fp._shm.values():
        s.close(); s.unlink()
    for fname, rr in (('summary.csv', summary), ('patches.csv', rows)):
        with open(out / fname, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rr[0])); wr.writeheader(); wr.writerows(rr)
    print(f'wrote {out}')
    print('=== DONE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
