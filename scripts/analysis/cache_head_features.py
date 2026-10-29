#!/usr/bin/env python3
"""Cache EXACTLY what ThresholdHead sees, plus the probability map and the label.

WHY THIS EXISTS
---------------
`ThresholdHead` reads `tf.stop_gradient(tokens)`. It can never alter the backbone, so it is
a strictly post-hoc component: any head variant is a function of (trunk tokens, label) and
nothing else. That means the expensive part — one forward pass of an 8.2 M-parameter
backbone over 496 k patches — can be paid ONCE, and every head variant afterwards is fitted
offline on the cached arrays in minutes. `head_sweep.py` does that part.

WHAT IS SAVED, per cohort, under <run-dir>/head_cache/
    <ck>__<cohort>__hsum.f16.npy   (N, 256)     the head's OWN patch summary
    <ck>__<cohort>__tsum.f16.npy   (N, 3*D)     mean | std | max over the RAW token axis
    <ck>__<cohort>__prob.f16.npy   (N, 32, 32)  flood probability
    <ck>__<cohort>__tau.f32.npy    (N,)         the CURRENT head's threshold, for reference
    <cohort>__labels.i8.npy        (N, 32, 32)  80 m label, -1 = permanent water
    <cohort>__index.csv            row_index, folder_name, patch_number

The token MEAN is exactly the summary the current head reduces to; std and max are a cheap
superset so a richer head can be tested without a second forward pass. D is 256 for the ViT
(trunk tokens) and 336 for the U-Net (flattened bottleneck).

HOW THE TOKENS ARE CAPTURED
---------------------------
The backbone returns (prob, tau) and never exposes the head's input. Rather than duplicate
the trunk's forward pass here — which would silently drift from `models/` — the head's bound
`call` is replaced by one that records its argument and delegates. What is cached is
therefore the head's input BY CONSTRUCTION, not a reimplementation of it. (Wrapping the head
in a new layer is not possible: Keras 3 forbids attaching sub-layers to a built layer.)

SELF-CHECK
----------
For any cohort that already has `inference/preds/<ck>__<cohort>__tau.f32.npy`, the tau this
script computes is compared against it and the job aborts on a mismatch. That verifies the
whole path — checkpoint, loader order, index attribution — against a known-good artefact.

TEST COHORTS ARE NOT REACHABLE from here: `--cohorts` rejects anything starting with `test`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # src/ on the path
from flood.paths import SRC, ROOT, METADATA
sys.path.insert(0, str(SRC))

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf                                        # noqa: E402
from tensorflow import keras                                   # noqa: E402

from flood.data.dataset import FloodPatches                          # noqa: E402
from flood.models.static.unet import ResidualUNet                  # noqa: E402
from flood.models.static.vit import PlainViT                          # noqa: E402
from flood.training.core import (FloodModel, resolve_patches_dir,     # noqa: E402
                          resolve_split, slice_spec, DEFAULT_VAL_SLICES)

BACKBONES = {'vit': PlainViT, 'unet': ResidualUNet}


def install_capture(head):
    """Record the tensor handed to `head` without altering the model's structure.

    Keras 3 forbids attaching a new sub-layer to a layer that is already built, so the head
    cannot be wrapped. Its bound `call` is replaced instead: a plain function is not a
    tracked type, so `Layer.__setattr__` lets it through, and `Layer.__call__` still
    dispatches to `self.call(...)` with the same signature. What lands in `box['t']` is
    therefore the head's input BY CONSTRUCTION rather than a reimplementation of the trunk.
    """
    box = {}
    orig = head.call

    def capturing_call(tokens, training=False):
        box['t'] = tokens
        return orig(tokens, training=training)

    head.call = capturing_call
    return box


def summarise(tokens):
    """(B, N, D) -> (B, 3D) : mean | std | max over the RAW token axis."""
    return tf.concat([tf.reduce_mean(tokens, axis=1),
                      tf.math.reduce_std(tokens, axis=1),
                      tf.reduce_max(tokens, axis=1)], axis=-1)


def head_summary(head, tokens):
    """The head's OWN patch summary: mean(LayerNorm(proj(tokens))).

    This is not the same thing as the mean of the raw tokens. `ThresholdHead.call` projects
    and LayerNorms EACH TOKEN and only then averages, and LayerNorm is per-token and
    nonlinear, so mean(LN(proj(t))) != LN(proj(mean(t))). Caching the raw token mean instead
    of this was measured on 2026-09-03 to make an offline replica of the head 19-32% WORSE
    than the shipped head, which invalidated a whole sweep. Reuse the head's own layers so
    the cached vector is its summary by construction.
    """
    x = head.proj(tokens) if head.proj is not None else tokens
    return tf.reduce_mean(head.norm(x), axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--weights', default='best_loss')
    ap.add_argument('--cohorts', nargs='+', default=None,
                    help="default follows the run's split: v0 train val_seen val_unseen, "
                         "v1 train val")
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--limit', type=int, default=0,
                    help='cache only the first N patches per cohort (smoke test)')
    a = ap.parse_args()

    run = Path(a.run_dir).resolve()
    cfg = yaml.safe_load(open(run / 'config.yaml'))
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or (SRC.parent / 'metadata'))
    split_dir, style = resolve_split(cfg, md)      # the run's OWN split, not the current one
    cohorts = a.cohorts or (['train'] + DEFAULT_VAL_SLICES[style])
    bad = [c for c in cohorts if c.startswith('test')]
    if bad:
        print(f"REFUSED: {bad} — this script never opens the test lockbox.")
        return 2
    out = run / 'head_cache'
    out.mkdir(exist_ok=True)
    workers = a.workers or min(os.cpu_count() or 8, 24)
    if not cfg.get('threshold_head', True):
        print('REFUSED: this run has no threshold head.')
        return 2

    model = FloodModel(BACKBONES[cfg['backbone']](cfg), cfg)
    dummy = (tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2)))
    before = float(tf.reduce_mean(model(dummy, training=False)[0]))
    wpath = run / 'checkpoints' / f'{a.weights}.weights.h5'
    model.load_weights(str(wpath))
    after = float(tf.reduce_mean(model(dummy, training=False)[0]))
    if abs(before - after) < 1e-9:
        raise RuntimeError(f'checkpoint restored nothing — {wpath}')
    print(f'  loaded {wpath.name}  (probe {before:.4f} -> {after:.4f})', flush=True)

    # install AFTER loading so weight paths are untouched
    head = model.backbone.thr_head
    cap = install_capture(head)

    for cohort in cohorts:
        coh, reg, _ = slice_spec(cohort, style)
        ds_obj = FloodPatches(patches_csv=split_dir / 'patches.csv',
                              patches_dir=resolve_patches_dir(cfg),
                              labels_dir=md / 'labels_80m_global', cohort=coh, regime=reg,
                              flood_flags_csv=split_dir / 'flood80_flags.csv')
        df = ds_obj.df
        n = len(df) if not a.limit else min(a.limit, len(df))
        ds = ds_obj.to_tf_dataset(batch_size=a.batch, shuffle=False, flood_ratio=0.0,
                                  workers=workers, return_index=True,
                                  deterministic=True, drop_remainder=False)

        TS = HS = PR = None
        TAU = np.full(n, np.nan, np.float32)
        LAB = np.zeros((n, 32, 32), np.int8)
        seen = np.zeros(n, bool)
        t0 = time.perf_counter()
        done = 0
        for (x, y, idx) in ds:
            prob, tau = model(x, training=False)
            if 't' not in cap:
                raise RuntimeError('threshold head was never called — capture failed')
            ts = np.asarray(summarise(cap['t']), np.float16)
            hs = np.asarray(head_summary(head, cap['t']), np.float16)
            i = np.asarray(idx, np.int64)
            m = i < n
            if not m.any():
                if (i >= n).all() and a.limit:
                    break
                continue
            i, ts, hs = i[m], ts[m], hs[m]
            if TS is None:
                TS = np.zeros((n, ts.shape[1]), np.float16)
                HS = np.zeros((n, hs.shape[1]), np.float16)
                PR = np.zeros((n, 32, 32), np.float16)
            TS[i] = ts
            HS[i] = hs
            PR[i] = np.asarray(prob, np.float16)[m, ..., 0]
            TAU[i] = np.asarray(tau, np.float32)[m, 0]
            LAB[i] = np.asarray(y, np.float32)[m, ..., 0].astype(np.int8)
            seen[i] = True
            done += int(m.sum())
            if done % (a.batch * 200) < a.batch:
                el = time.perf_counter() - t0
                print(f'    [{cohort}] {done:,}/{n:,}  {el:.0f}s  '
                      f'eta {el / max(done, 1) * (n - done):.0f}s', flush=True)
            if done >= n:
                break
        if not seen.all():
            raise RuntimeError(f'{(~seen).sum():,} patches of {n:,} never arrived for {cohort}')

        # verify the cached head summary really is the head's: push it back through the
        # head's last two layers and it must reproduce tau to float precision.
        chk = np.random.default_rng(0).choice(n, min(4096, n), replace=False)
        rec = np.asarray(tf.nn.sigmoid(head.d2(head.d1(
            tf.convert_to_tensor(HS[chk].astype(np.float32))))))[:, 0]
        rd = float(np.abs(rec - TAU[chk]).max())
        print(f'    hsum -> tau reconstruction: max|diff| = {rd:.2e}', flush=True)
        if rd > 5e-3:
            raise RuntimeError(f'cached head summary does not reproduce tau ({rd:.3e})')

        np.save(out / f'{a.weights}__{cohort}__tsum.f16.npy', TS)
        np.save(out / f'{a.weights}__{cohort}__hsum.f16.npy', HS)
        np.save(out / f'{a.weights}__{cohort}__prob.f16.npy', PR)
        np.save(out / f'{a.weights}__{cohort}__tau.f32.npy', TAU)
        np.save(out / f'{cohort}__labels.i8.npy', LAB)
        df.iloc[:n][['folder_name', 'patch_number']].reset_index(drop=True) \
          .rename_axis('row_index').to_csv(out / f'{cohort}__index.csv')
        el = time.perf_counter() - t0
        print(f'  [{cohort}] {n:,} patches in {el / 60:.1f} min  '
              f'tsum {TS.shape}  ({(TS.nbytes + PR.nbytes + LAB.nbytes) / 1e9:.2f} GB)',
              flush=True)

        # self-check against the already-published tau, when one exists
        ref = run / 'inference' / 'preds' / f'{a.weights}__{cohort}__tau.f32.npy'
        if ref.exists() and not a.limit:
            r = np.load(ref).reshape(-1)
            if len(r) == n:
                d = float(np.abs(r - TAU).max())
                print(f'    self-check vs inference/preds tau: max|diff| = {d:.2e}',
                      flush=True)
                if d > 1e-3:
                    raise RuntimeError(
                        f'tau mismatch for {cohort} (max {d:.3e}) — the cached features do '
                        f'not correspond to the published predictions')
    print('done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
