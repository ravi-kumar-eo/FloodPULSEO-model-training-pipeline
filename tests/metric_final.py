#!/usr/bin/env python3
"""Check the SELECTION metric: FloodIoUBest inside Keras evaluate == the evaluator's pick_threshold.

Loads real validation rows once into arrays, then (1) model.evaluate on them gives val
flood_iou_best, (2) a plain forward pass on the same arrays + scripts/evaluate.pick_threshold
gives the evaluator's best pooled flood-only IoU. The two must agree (the evaluator rounds stored
probabilities to float16, so agreement is to ~1e-4, not bit-exact).

    python metric_final.py --family unet --weights <checkpoint> --rows 1024
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
import yaml

from flood.paths import METADATA                                             # noqa: E402
from flood.data.dataset_final import FinalFloodPatches, load_transforms        # noqa: E402
from flood.training.core_final import build_model, data_paths, resolve_patches_dir  # noqa: E402
from scripts.train import FAMILIES                                       # noqa: E402
from scripts.evaluate import pick_threshold                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True); ap.add_argument('--weights', required=True)
    ap.add_argument('--rows', type=int, default=1024)
    a = ap.parse_args()
    D = Path(__file__).resolve().parents[1] / 'configs'
    cfg = yaml.safe_load(open(D / '_common.yaml')); cfg.update(yaml.safe_load(open(D / f't0_{a.family}_s1.yaml')))
    cfg['patches_dir'] = resolve_patches_dir(cfg)
    tf.keras.mixed_precision.set_global_policy(cfg['mixed_precision'])
    dp = data_paths(cfg, Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA))
    fp = FinalFloodPatches(cohort='val', patches_csv=dp['split'] / 'patches.csv', patches_dir=cfg['patches_dir'],
                           labels_dir=dp['labels'], flood_flags_csv=dp['split'] / 'flood80_flags.csv', temporal_days=0,
                           transforms=load_transforms(dp['transforms']), mono_only=True)
    fp.preload_labels(verbose=False)
    rng = np.random.default_rng(0)
    idx = np.r_[rng.choice(np.flatnonzero(fp.has_flood), a.rows // 2, replace=False),
                rng.choice(len(fp), a.rows // 2, replace=False)]
    rows = [fp.load(int(i)) for i in idx]
    X = tuple(np.stack([r[k] for r in rows]) for k in range(3)); Y = np.stack([r[3] for r in rows])
    m = build_model(FAMILIES[a.family][0], cfg); m.load_weights(a.weights)
    m.compile(optimizer=FAMILIES[a.family][1](cfg), jit_compile=False)
    ds = tf.data.Dataset.from_tensor_slices((X, Y)).batch(32)
    r = m.evaluate(ds, return_dict=True, verbose=0)
    P = np.concatenate([np.asarray(tf.cast(m(tuple(tf.constant(x[s:s + 32]) for x in X), training=False)[0], tf.float32))
                        for s in range(0, len(idx), 32)])
    thr32, iou32 = pick_threshold(P[..., 0], Y[..., 0].astype(np.int8))
    thr16, iou16 = pick_threshold(P[..., 0].astype(np.float16), Y[..., 0].astype(np.int8))
    print(f'keras flood_iou_best        {r["flood_iou_best"]:.6f}')
    print(f'evaluator, float32 probs    {iou32:.6f} at threshold {thr32}')
    print(f'evaluator, float16 probs    {iou16:.6f} at threshold {thr16} (as the evaluator stores them)')
    ok = abs(r['flood_iou_best'] - iou32) < 1e-5 and abs(iou32 - iou16) < 1e-3
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
