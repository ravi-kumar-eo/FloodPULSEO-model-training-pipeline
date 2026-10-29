#!/usr/bin/env python3
"""Sensitivity of a fit with dynamic hydroclimatic variables to perturbed inputs (thesis Section 5.3.6).

    python perturb_eval.py --run-dir <runs_final>/final_t30/seed1/unet [--batch 64] [--workers 24]

Loads the fit exactly as scripts/evaluate.py does (same checkpoint, data version and split checks) and predicts
the TEST rows (full row set) once per perturbation of the in-memory daily inputs (FinalFloodPatches._wx, shape
(N, 30, 5): local precipitation / precip_div, local soil moisture, validity, upstream precipitation / precip_div,
log10(1 + catchment area) / uda_div). Only the inputs change; the fit's validation-selected global threshold
(inference_final/threshold.json) is applied unchanged.

Perturbations
  none      unchanged inputs; TP, FP and FN must lie within 0.1% of inference_final/metrics.csv (GPU arithmetic can flip
            cells at the threshold between runs); the exact differences are printed. Every perturbation is compared with
            this unperturbed pass of the same run, not with the stored evaluation.
  none2     unchanged inputs, predicted a second time: the run-to-run noise of the GPU arithmetic, against which the
            response to the perturbations is read
  p0        local and upstream precipitation set to zero on all 30 days
  p05       local and upstream precipitation multiplied by 0.5
  p2        local and upstream precipitation multiplied by 2
  swap      channels 0-3 (precipitation, soil moisture, validity) of every row replaced by those of a random test row
            from another CEMS activation (seed 0); the catchment area (channel 4) stays the patch's own

Writes <run-dir>/inference_perturb/ (nothing in inference_final is touched):
  patches.csv   one row per test row and perturbation: tp, fp, fn, scored_cells, predicted cells, mean probability
  summary.csv   one row per perturbation: pooled TP, FP, FN, IoU, predicted cells, mean |p - p_none| over scored cells
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

PERTURBATIONS = ('none', 'none2', 'p0', 'p05', 'p2', 'swap')
PRECIP = (0, 3)                                                           # channels of local and upstream precipitation


def perturbed(wx0, act, name, seed=0):
    wx = wx0.copy()
    if name == 'p0':
        wx[:, :, PRECIP] = 0.0
    elif name in ('p05', 'p2'):
        wx[:, :, PRECIP] *= 0.5 if name == 'p05' else 2.0
    elif name == 'swap':
        rng = np.random.default_rng(seed)
        donor = rng.permutation(len(wx))
        for _ in range(100):                                              # re-draw donors from the same activation
            same = act[donor] == act
            if not same.any():
                break
            donor[same] = rng.integers(0, len(wx), int(same.sum()))
        if (act[donor] == act).any():
            raise SystemExit('swap: could not find donors from other activations')
        wx[:, :, :4] = wx0[donor, :, :4]
    elif name not in ('none', 'none2'):
        raise ValueError(name)
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
    if not int(cfg.get('temporal_days', 0)) or cfg.get('weather_mode', 'aoi') != 'upstream':
        raise SystemExit('perturbations need a fit with upstream-mode dynamic hydroclimatic variables')
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
    act = df.folder_name.str.split('_').str[0].to_numpy()
    out = run / 'inference_perturb'
    out.mkdir(exist_ok=True)
    rows, summary, P_none = [], [], None
    for name in PERTURBATIONS:
        fp._wx = perturbed(wx0, act, name)
        t0 = time.time()
        P, _, Y = ev.predict(model, fp, a.batch, a.workers)
        tp, fpos, fn, nk = ev.counts(P, Y, thr)
        keep = Y >= 0
        pred = ((P.astype(np.float32) >= thr) & keep).reshape(len(df), -1).sum(1)
        pmean = np.where(keep, P.astype(np.float32), 0).reshape(len(df), -1).sum(1) / np.maximum(nk, 1)
        if name == 'none':                             # reproduction check against the stored evaluation
            got = (int(tp.sum()), int(fpos.sum()), int(fn.sum()))
            want = (int(ref['tp']), int(ref['fp']), int(ref['fn']))
            rel = max(abs(g - w) / max(w, 1) for g, w in zip(got, want))
            stored = run / 'inference_final' / 'preds' / 'test__prob.f16.npy'
            dmax = float(np.abs(np.load(stored).astype(np.float32) - P.astype(np.float32)).max()) if stored.exists() else float('nan')
            print(f'reproduction: TP/FP/FN {got} vs stored {want}, largest relative difference {rel:.2e}, '
                  f'largest probability difference {dmax:.2e}', flush=True)
            if rel > 1e-3:
                raise SystemExit('REFUSED: unperturbed counts differ from inference_final/metrics.csv by more than 0.1%')
            P_none = P.astype(np.float32)
        dabs = float(np.abs(P.astype(np.float32) - P_none)[keep].mean())
        T, F, N = int(tp.sum()), int(fpos.sum()), int(fn.sum())
        summary.append({'perturbation': name, 'tp': T, 'fp': F, 'fn': N, 'iou': T / (T + F + N),
                        'predicted_cells': int(pred.sum()), 'mean_abs_prob_change': dabs, 'seconds': round(time.time() - t0)})
        print(summary[-1], flush=True)
        for i in range(len(df)):
            rows.append({'perturbation': name, 'row_index': i, 'folder_name': df.folder_name.iat[i],
                         'patch_number': int(df.patch_number.iat[i]), 'tp': int(tp[i]), 'fp': int(fpos[i]),
                         'fn': int(fn[i]), 'scored_cells': int(nk[i]), 'predicted': int(pred[i]),
                         'mean_prob': float(pmean[i])})
    fp._wx = wx0
    for s in fp._shm.values():
        s.close(); s.unlink()
    for fname, rr in (('summary.csv', summary), ('patches.csv', rows)):
        with open(out / fname, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rr[0])); wr.writeheader(); wr.writerows(rr)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
