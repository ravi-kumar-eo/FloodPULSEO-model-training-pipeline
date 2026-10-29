#!/usr/bin/env python3
"""Evaluator of the FINAL programme (METHOD_PLAN_RERUN.md v3, §8 and C12).

    python evaluate.py --run-dir <runs_final>/final_t30/seed1/unet [--with-test]

Protocol (frozen before any test score is seen):
  checkpoint   the run's selected checkpoint (run_manifest selection.checkpoint): best validation
               pooled flood-only IoU at the best of 49 thresholds (Ravi, 2026-10-05)
  population   flood-only cells: permanent water and mixed cells (label -1) are excluded
  threshold    ONE global threshold per fit: the value of 0.02, 0.04, ..., 0.98 that maximises
               the POOLED flood-only IoU on the validation rows of the fit's own row set
               (T0: single-observation rows; weather fits: all rows). Applied unchanged to
               every test group. The threshold head's per-patch tau is the secondary rule.
  rows         test probabilities are stored for ALL test rows, with `exp1_keep`, so every
               score exists on both row sets: 'full' (all acquisitions) and 'single'
               (one acquisition per AOI).
  aggregation  pooled IoU / precision / recall; AUPRC pooled (400 thresholds) and per product (200); per-product mean IoU over the FIXED set of
               flood-positive products of that row set; per-ground-unit IoU (TP, FP, FN pooled
               within the ground unit, then IoU), mean over flood-positive ground units;
               false-alarm share on dry products (FP cells / scored cells), reported separately.

Writes <run-dir>/inference_final/:
  preds/<cohort>__prob.f16.npy (N,32,32) · <cohort>__tau.f32.npy (N,) · <cohort>__labels.i8.npy
  preds/<cohort>__index.csv     row_index, folder_name, patch_number, exp1_keep, aoi, ground_unit,
                                basin_regime, continent, climate
  threshold.json · metrics.csv (one row per cohort x row set x aggregation x rule) · patches.csv
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
import yaml

from flood.paths import METADATA                                       # noqa: E402
from flood.data.dataset_final import FinalFloodPatches, load_transforms  # noqa: E402
from flood.training.core_final import build_model, data_paths, md5, STAGES, weather_shape, final_dataset_kwargs  # noqa: E402
from scripts.train import FAMILIES                               # noqa: E402

GRID = np.round(np.linspace(0.02, 0.98, 49), 4)
META = ['folder_name', 'patch_number', 'exp1_keep', 'aoi', 'ground_unit', 'basin_regime',
        'continent', 'climate']


def predict(model, fp, batch, workers):
    n = len(fp.df)
    P, TAU = np.zeros((n, 32, 32), np.float16), np.zeros(n, np.float32)
    Y = np.zeros((n, 32, 32), np.int8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s in range(0, n, batch):
            rows = list(ex.map(fp.load, range(s, min(s + batch, n))))
            k = len(rows[0]) - 1
            x = tuple(tf.convert_to_tensor(np.stack([r[j] for r in rows])) for j in range(k))
            out = model(x, training=False)
            p, tau = out if isinstance(out, (tuple, list)) else (out, None)
            P[s:s + len(rows)] = np.asarray(tf.cast(p, tf.float32))[..., 0]
            if tau is not None:
                TAU[s:s + len(rows)] = np.asarray(tf.cast(tau, tf.float32)).reshape(-1)
            Y[s:s + len(rows)] = np.stack([r[-1] for r in rows])[..., 0].astype(np.int8)
            if (s // batch) % 200 == 0:
                print(f'    {s + len(rows):,}/{n:,}', flush=True)
    return P, TAU, Y


def counts(P, Y, thr):
    """Per-patch TP, FP, FN on flood-only cells. thr: scalar or (N,) per-patch."""
    t = np.asarray(thr, np.float32).reshape(-1, 1, 1) if np.ndim(thr) else np.float32(thr)
    keep = Y >= 0
    truth = (Y == 1)
    pred = (P.astype(np.float32) >= t) & keep
    ax = (1, 2)
    return ((pred & truth).sum(ax).astype(np.int64), (pred & ~truth).sum(ax).astype(np.int64),
            (~pred & truth).sum(ax).astype(np.int64), keep.sum(ax).astype(np.int64))


def iou(tp, fp, fn):
    d = tp + fp + fn
    return tp / d if d else float('nan')


def auprc(p, y, n):
    """Area under the precision-recall curve on flood-only cells (TensorFlow, n thresholds)."""
    keep = y >= 0
    if not (y[keep] == 1).any():
        return float('nan')
    m = tf.keras.metrics.AUC(curve='PR', num_thresholds=n)
    m.update_state((y[keep] == 1).astype(np.float32), p[keep].astype(np.float32))
    return float(m.result())


def auprc_scores(df, P, Y):
    """Pooled AUPRC (400 thresholds) and the mean per-product AUPRC (200) over flood-positive products."""
    per = []
    for fn, idx in df.groupby('folder_name').indices.items():
        if (Y[idx] == 1).any():
            per.append(auprc(P[idx], Y[idx], 200))
    return {'pooled_auprc': auprc(P, Y, 400), 'product_mean_auprc': float(np.mean(per)) if per else float('nan')}


def pick_threshold(P, Y):
    best = (-1.0, None)
    for t in GRID:
        tp, fp, fn, _ = counts(P, Y, t)
        v = iou(tp.sum(), fp.sum(), fn.sum())
        if v > best[0]:
            best = (v, float(t))
    return best[1], best[0]


def scores(df, tp, fp, fn, nk):
    """Pooled, per-product, per-ground-unit and dry-product numbers for the rows in df."""
    out = {'n_patches': int(len(df))}
    T, F, N = int(tp.sum()), int(fp.sum()), int(fn.sum())
    out.update(pooled_iou=iou(T, F, N), pooled_precision=T / (T + F) if T + F else float('nan'),
               pooled_recall=T / (T + N) if T + N else float('nan'), tp=T, fp=F, fn=N)
    g = df.assign(tp=tp, fp=fp, fn=fn, nk=nk)
    pr = g.groupby('folder_name')[['tp', 'fp', 'fn', 'nk']].sum()
    pos = pr[(pr.tp + pr.fn) > 0]
    out['products_flood_positive'] = int(len(pos))
    out['product_mean_iou'] = float((pos.tp / (pos.tp + pos.fp + pos.fn)).mean()) if len(pos) else float('nan')
    dry = pr[(pr.tp + pr.fn) == 0]
    out['products_dry'] = int(len(dry))
    out['dry_false_alarm_share'] = float(dry.fp.sum() / dry.nk.sum()) if dry.nk.sum() else float('nan')
    gu = g.groupby('ground_unit')[['tp', 'fp', 'fn']].sum()
    gu = gu[(gu.tp + gu.fn) > 0]
    out['ground_units_flood_positive'] = int(len(gu))
    out['ground_unit_mean_iou'] = float((gu.tp / (gu.tp + gu.fp + gu.fn)).mean()) if len(gu) else float('nan')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--with-test', action='store_true')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    run = a.run_dir
    man = json.loads((run / 'run_manifest.json').read_text())
    if not (run / 'COMPLETE').exists():
        raise SystemExit(f'{run} is not COMPLETE')
    cfg = yaml.safe_load(open(run / 'config.yaml'))
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA)
    dp = data_paths(cfg, md)
    if md5(dp['version']) != man['data_version_md5']:
        raise SystemExit('REFUSED: the data version differs from the one this fit trained on')
    if md5(dp['split'] / 'patches.csv') != man['split_md5']:
        raise SystemExit('REFUSED: split md5 differs')
    if cfg.get('mixed_precision', 'float32') != 'float32':
        tf.keras.mixed_precision.set_global_policy(cfg['mixed_precision'])
    T = int(cfg.get('temporal_days', 0))
    rows_single = STAGES[man['stage']][1] == 'single'
    out = run / 'inference_final'; (out / 'preds').mkdir(parents=True, exist_ok=True)

    model = build_model(FAMILIES[man['family']][0], cfg)
    w = run / 'checkpoints' / man.get('selection', {}).get('checkpoint', 'best_loss.weights.h5')
    dummy = [tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2))]
    if T:
        dummy.append(tf.zeros((1,) + weather_shape(cfg)))
    before = np.asarray(tf.cast(model(tuple(dummy), training=False)[0], tf.float32))
    model.load_weights(str(w))
    if np.array_equal(before, np.asarray(tf.cast(model(tuple(dummy), training=False)[0], tf.float32))):
        raise SystemExit(f'checkpoint restored nothing: {w}')

    kw = final_dataset_kwargs(cfg, dp)
    kw.pop('mono_only')
    metrics, patch_rows = [], []
    thr = None
    for cohort in ['val'] + (['test'] if a.with_test else []):
        fp = FinalFloodPatches(cohort=cohort, allow_test=(cohort == 'test'), **kw)
        df = fp.df.reset_index(drop=True)
        if df.duplicated(['folder_name', 'patch_number']).any():
            raise SystemExit(f'{cohort}: duplicate (product, patch) rows')
        fp.enable_eval_cache(workers=a.workers)
        t0 = time.time()
        P, TAU, Y = predict(model, fp, a.batch, a.workers)
        print(f'  [{cohort}] {len(df):,} rows in {time.time() - t0:.0f}s', flush=True)
        for s in fp._shm.values():
            s.close(); s.unlink()
        np.save(out / 'preds' / f'{cohort}__prob.f16.npy', P)
        np.save(out / 'preds' / f'{cohort}__tau.f32.npy', TAU)
        np.save(out / 'preds' / f'{cohort}__labels.i8.npy', Y)
        idx = df[[c for c in META if c in df.columns]].copy(); idx.index.name = 'row_index'
        idx.to_csv(out / 'preds' / f'{cohort}__index.csv')
        single = df.exp1_keep.astype(str).str.lower().isin(['true', '1']).to_numpy()

        if cohort == 'val':
            sel = single if rows_single else np.ones(len(df), bool)
            thr, best = pick_threshold(P[sel], Y[sel])
            (out / 'threshold.json').write_text(json.dumps({
                'threshold': thr, 'val_pooled_iou': best, 'rows': 'single' if rows_single else 'full',
                'grid': '0.02..0.98 step 0.02', 'checkpoint': str(w), 'checkpoint_md5': md5(w)}, indent=2))
            print(f'  threshold {thr} (validation pooled flood-only IoU {best:.4f})', flush=True)
        au = {rs: auprc_scores(df[m].reset_index(drop=True), P[m], Y[m])
              for rs, m in (('full', np.ones(len(df), bool)), ('single', single))}
        for rule, th in (('global', thr), ('head', TAU)):
            tp, fpos, fn, nk = counts(P, Y, th)
            for rs, m in (('full', np.ones(len(df), bool)), ('single', single)):
                r = {**scores(df[m], tp[m], fpos[m], fn[m], nk[m]), **au[rs]}
                metrics.append({'run': str(run), 'cohort': cohort, 'rows': rs, 'rule': rule,
                                'threshold': thr if rule == 'global' else 'per-patch tau', **r})
                if cohort == 'test' and rule == 'global' and rs == 'full' and 'basin_regime' in df:
                    for reg in sorted(df.basin_regime[m].unique()):
                        mm = m & (df.basin_regime == reg).to_numpy()
                        metrics.append({'run': str(run), 'cohort': f'test:{reg}', 'rows': rs, 'rule': rule,
                                        'threshold': thr, **scores(df[mm], tp[mm], fpos[mm], fn[mm], nk[mm])})
            if rule == 'global':
                for i in range(len(df)):
                    patch_rows.append({'cohort': cohort, 'row_index': i, 'folder_name': df.folder_name.iat[i],
                                       'patch_number': int(df.patch_number.iat[i]), 'tp': int(tp[i]),
                                       'fp': int(fpos[i]), 'fn': int(fn[i]), 'scored_cells': int(nk[i]),
                                       'tau_head': float(TAU[i])})

    for name, rows in (('metrics.csv', metrics), ('patches.csv', patch_rows)):
        with open(out / name, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f'wrote {out}/metrics.csv ({len(metrics)} rows), patches.csv ({len(patch_rows):,})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
