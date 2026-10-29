#!/usr/bin/env python3
"""Post-fit check of a finished weather fit: does its selected checkpoint USE the weather?

On 2,048 fixed validation rows (half flood-bearing, seed 0, the same rows for every fit), compares
  T0 parent                    pooled flood-only IoU at the best of 49 thresholds
  this fit, true weather       the same score
  this fit, swapped weather    each row gets the weather of another product
and the mean / max |p(true) - p(swapped)|. Writes <run-dir>/weather_check.json. Validation only;
the test subset is never read.

    python weather_check.py --run-dir <runs_final>/final_t30/seed1/unet
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf
import yaml

from flood.paths import METADATA                                             # noqa: E402
from flood.data.dataset_final import FinalFloodPatches, load_transforms        # noqa: E402
from flood.training.core_final import build_model, data_paths, final_dataset_kwargs  # noqa: E402
from scripts.final.train import FAMILIES                                       # noqa: E402
from scripts.final.evaluate import pick_threshold                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--probe', type=int, default=2048)
    a = ap.parse_args()
    run = a.run_dir
    cfg = yaml.safe_load(open(run / 'config.yaml'))
    man = json.loads((run / 'run_manifest.json').read_text())
    fam = FAMILIES[man['family']]
    if cfg.get('mixed_precision', 'float32') != 'float32':
        tf.keras.mixed_precision.set_global_policy(cfg['mixed_precision'])
    dp = data_paths(cfg, Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA))
    T = int(cfg['temporal_days'])
    kw = final_dataset_kwargs(cfg, dp); kw.pop('mono_only')
    fp = FinalFloodPatches(cohort='val', **kw)
    fp.preload_labels(verbose=False)
    rng = np.random.default_rng(0)
    idx = np.r_[rng.choice(np.flatnonzero(fp.has_flood), a.probe // 2, replace=False),
                rng.choice(len(fp), a.probe // 2, replace=False)]
    rows = [fp.load(int(i)) for i in idx]
    X = [np.stack([r[k] for r in rows]) for k in range(4)]
    Y = np.stack([r[4] for r in rows])[..., 0].astype(np.int8)
    prods = fp.df.folder_name.to_numpy()[idx]
    perm = rng.permutation(len(idx))
    for _ in range(20):
        same = prods[perm] == prods
        if not same.any():
            break
        perm[same] = rng.permutation(perm[same])
    Xs = X[:3] + [X[3][perm]]

    m = build_model(fam[0], cfg)
    m.load_weights(str(run / 'checkpoints' / man['selection']['checkpoint']))
    twin = dict(cfg); twin['temporal_days'] = 0
    # the T0 reference: the warm-start parent, or (from-scratch fits) the same seed's T0 fit
    pck = (man.get('parent') or {}).get('checkpoint')
    if not pck:
        tdir = run.parents[2] / 'final_t0' / f"seed{man['seed']}" / man['family']
        pck = str(tdir / 'checkpoints' / json.loads((tdir / 'run_manifest.json').read_text())['selection']['checkpoint'])
    t0 = build_model(fam[0], twin); t0.load_weights(pck)

    def predict(model, xs, bs=16):
        return np.concatenate([np.asarray(tf.cast(model(tuple(tf.constant(x[s:s + bs]) for x in xs),
                                                        training=False)[0], tf.float32))[..., 0]
                               for s in range(0, len(idx), bs)])

    pt, ps, p0 = predict(m, X), predict(m, Xs), predict(t0, X[:3])
    res = {'run': str(run), 'temporal_days': T, 'probe_rows': len(idx),
           'selected_epoch': man['selection'].get('selected_epoch'),
           't0_parent': dict(zip(('threshold', 'iou'), pick_threshold(p0, Y))),
           'true_weather': dict(zip(('threshold', 'iou'), pick_threshold(pt, Y))),
           'swapped_weather': dict(zip(('threshold', 'iou'), pick_threshold(ps, Y))),
           'mean_abs_p_true_minus_swapped': float(np.abs(pt - ps).mean()),
           'max_abs_p_true_minus_swapped': float(np.abs(pt - ps).max())}
    res['iou_gain_true_minus_swapped'] = res['true_weather']['iou'] - res['swapped_weather']['iou']
    res['iou_gain_true_minus_t0'] = res['true_weather']['iou'] - res['t0_parent']['iou']
    (run / 'weather_check.json').write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
