#!/usr/bin/env python3
"""G7 learning test of the final weather models: does the weather path start to be USED?

Warm start of the T30 config from a T0 checkpoint, the real optimizers (core_final.
configure_warm_optimizers: copied weights at fine_tune_lr, new weather parts at new_parts_lr), real
training batches, mixed precision as configured. On fixed VALIDATION rows (half flood-bearing) it
reports, at step 0 and every 100 steps:
  sens_max / sens_mean   max / mean |p(true weather) - p(weather of another product)|
  dloss                  segmentation loss with swapped weather minus with true weather
                         (> 0 means the true weather helps on held-out rows)
  gates                  ViT gates g, g2; U-Net norm of the zero-initialised conv of the FiLM block
At step 0 it also checks the identity (T30 == T0 on the probe rows) and lists the gradient norm of
every weather component, so a component that receives no gradient is visible.

    python learn_final.py --family unet --parent <T0 best_loss.weights.h5> --steps 400 --batch 32
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
from flood.training.core_final import build_model, configure_warm_optimizers, data_paths, resolve_patches_dir, final_dataset_kwargs  # noqa: E402
from flood.training import phases                                              # noqa: E402
from flood.losses import segmentation_loss                                     # noqa: E402
from scripts.final.train import FAMILIES                                       # noqa: E402
from scripts.final.evaluate import pick_threshold                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=['unet', 'vit'])
    ap.add_argument('--parent', required=True)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--probe', type=int, default=64)
    ap.add_argument('--config', default=None, help='default: configs/final/t30_<family>_s1.yaml')
    ap.add_argument('--every', type=int, default=100, help='training steps between probes')
    ap.add_argument('--out', default=None, help='also write the final JSON here')
    a = ap.parse_args()
    D = Path(__file__).resolve().parents[1] / 'configs' / 'final'
    cfg = yaml.safe_load(open(D / '_common.yaml'))
    cfg.update(yaml.safe_load(open(a.config or D / f't30_{a.family}_s1.yaml')))
    cfg['patches_dir'] = resolve_patches_dir(cfg); cfg['batch_size'] = a.batch
    if cfg.get('mixed_precision', 'float32') != 'float32':
        tf.keras.mixed_precision.set_global_policy(cfg['mixed_precision'])
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA)
    dp = data_paths(cfg, md)
    kw = final_dataset_kwargs(cfg, dp); kw.pop('mono_only')
    train = FinalFloodPatches(cohort='train', **kw)
    ds = train.to_tf_dataset(batch_size=a.batch, flood_ratio=0.5, seed=1, workers=12)

    val = FinalFloodPatches(cohort='val', **kw); val.preload_labels(verbose=False)
    rng = np.random.default_rng(0)
    idx = np.r_[rng.choice(np.flatnonzero(val.has_flood), a.probe // 2, replace=False),
                rng.choice(len(val), a.probe // 2, replace=False)]
    rows = [val.load(int(i)) for i in idx]
    X = [np.stack([r[k] for r in rows]) for k in range(4)]; Y = tf.constant(np.stack([r[4] for r in rows]))
    prods = val.df.folder_name.to_numpy()[idx]
    perm = rng.permutation(len(idx))
    for _ in range(20):
        same = prods[perm] == prods
        if not same.any():
            break
        perm[same] = rng.permutation(perm[same])
    Xs = X[:3] + [X[3][perm]]

    fam = FAMILIES[a.family]
    model = build_model(fam[0], cfg)
    twin = dict(cfg); twin['temporal_days'] = 0
    phases.load_static_checkpoint(model, lambda: build_model(fam[0], twin), a.parent)
    ft, nlr, n_old, n_newv, n_new = configure_warm_optimizers(model, cfg, fam[1])
    t0 = build_model(fam[0], twin); t0.load_weights(a.parent)
    bb = model.backbone
    n_t0 = int(sum(np.prod(v.shape) for v in t0.trainable_variables))
    parts = {l.name: int(sum(np.prod(v.shape) for v in l.trainable_variables)) for l in phases.weather_layers(bb)}

    def predict(m, xs, bs=16):
        out = []
        for s in range(0, len(idx), bs):
            out.append(tf.cast(m(tuple(tf.constant(x[s:s + bs]) for x in xs), training=False)[0], tf.float32))
        return tf.concat(out, 0)

    Yi = Y.numpy()[..., 0].astype(np.int8)

    def iou_best(p):
        return pick_threshold(p.numpy()[..., 0], Yi)       # (threshold, pooled flood-only IoU)

    def probe(step):
        pt, ps = predict(model, X), predict(model, Xs)
        d = tf.abs(pt - ps)
        (tt, it), (ts, is_) = iou_best(pt), iou_best(ps)
        r = {'step': step, 'sens_max': float(tf.reduce_max(d)), 'sens_mean': float(tf.reduce_mean(d)),
             'loss_true': float(segmentation_loss(Y, pt)), 'dloss_swapped_minus_true':
                 float(segmentation_loss(Y, ps) - segmentation_loss(Y, pt)),
             'iou_best_true': round(float(it), 5), 'thr_true': tt,
             'iou_best_swapped': round(float(is_), 5), 'thr_swapped': ts}
        if a.family == 'vit':
            r['gate_g'] = float(bb.interaction.g.numpy())
            if getattr(bb.interaction, 'g2', None) is not None:
                r['gate_g2'] = float(bb.interaction.g2.numpy())
        elif getattr(bb, 'cond_block', None) is not None:
            r['film_c2_norm'] = float(tf.norm(bb.cond_block.c2.kernel))
        return r

    pt0 = predict(t0, X[:3]); pt = predict(model, X)
    ident = float(tf.reduce_max(tf.abs(pt - pt0)))
    t0_thr, t0_iou = iou_best(pt0)
    print(json.dumps({'t0_iou_best': round(float(t0_iou), 5), 't0_thr': t0_thr}), flush=True)
    # gradient of every weather component on one real batch, at step 0
    xb, yb = next(iter(ds))
    with tf.GradientTape() as tape:
        out = model.backbone(xb, training=True)
        loss = segmentation_loss(yb, tf.cast(out[0], tf.float32))
    vs = model.trainable_variables
    gs = dict(zip([id(v) for v in vs], tape.gradient(loss, vs)))
    gnorm = {}
    for l in phases.weather_layers(bb):
        g = [gs[id(v)] for v in l.trainable_variables if gs.get(id(v)) is not None]
        gnorm[l.name] = float(tf.linalg.global_norm(g)) if g else 0.0
    log = [probe(0)]

    class Probe(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            r = probe(a.every * (epoch + 1)); r['train_seg_loss'] = float(logs.get('seg_loss', float('nan')))
            log.append(r); print(json.dumps(r), flush=True)

    model.fit(ds, steps_per_epoch=a.every, epochs=max(a.steps // a.every, 1), verbose=0, callbacks=[Probe()])
    res = {'family': a.family, 'precision': cfg.get('mixed_precision'), 'batch': a.batch,
           'lr_copied': ft, 'lr_new': nlr, 't0_params': n_t0, 'weather_params': n_new,
           'weather_share_of_t0': round(n_new / n_t0, 4), 'weather_parts': parts,
           'identity_max_abs_diff_step0': ident, 'grad_norm_step0': gnorm,
           't0_iou_best': round(float(t0_iou), 5), 'log': log}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    sys.exit(main())
