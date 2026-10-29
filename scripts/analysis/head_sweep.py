#!/usr/bin/env python3
"""Ask whether the per-patch threshold head has any room left in it.

Fits head variants on the TRAIN cohort's cached features and scores them on val_seen AND
val_unseen. Nothing here touches a GPU-trained backbone: `cache_head_features.py` already
paid that cost, and `ThresholdHead` is `stop_gradient`-isolated, so a head variant is a
pure function of (cached trunk summary, cached probability map).

WHY FIT ON TRAIN
----------------
A decision rule fitted on ONE validation cohort and applied to the other was measured to
LOSE 2.8-17.2% pooled IoU on 2026-09-02: it inherits that cohort's calibration and breaks
when the deployment regime changes. Fitting on train and scoring on both validation cohorts
is the protocol the trained head itself obeys, and it is the only one whose result transfers.

THE LADDER
----------
  V0  token mean, soft_threshold_loss             reproduces the shipped head
  V1  + probability-map order statistics          is the head INFORMATION-starved?
  V2  dry_weight sweep                            is the OBJECTIVE mis-weighted?
  V3  sharpness 20 -> 50                          is the relaxation too soft? (DBNet uses 50)
  V4  per-PIXEL threshold map                     is one scalar per patch too restrictive?
  V5  supervised on the ORACLE threshold          is the information there AT ALL?

V5 is the diagnostic. It discards `soft_threshold_loss` and regresses straight onto the
oracle, so:
    helps on both val cohorts -> the OBJECTIVE is the limit; there is scope.
    fits train, fails on val  -> a generalisation limit; little scope.
    fails even on train       -> the information is not in the head's inputs. Hard blockage.

DECISION RULE, fixed before looking: adopt a variant only if it beats the shipped head by
>5% relative pooled flood IoU on BOTH val_seen and val_unseen.

THE ORACLE is computed by Dinkelbach's method, which is EXACT for maximising a ratio of
sums: pooled IoU = sum(TP) / sum(TP+FP+FN) is linear-fractional in the per-patch choice, so
alternating "pick argmax_j (TP - lam*(TP+FP+FN)) per patch" with "lam <- achieved IoU"
converges to the true optimum. Ties break to the HIGHEST threshold, because a patch whose
probabilities never reach tau is indifferent above its own maximum and the conservative
choice is the meaningful one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))   # src/ on the path
from flood.paths import SRC, ROOT
sys.path.insert(0, str(SRC))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf                                        # noqa: E402
from tensorflow import keras                                   # noqa: E402
from tensorflow.keras import layers                            # noqa: E402

from flood.models.threshold_head import soft_threshold_loss          # noqa: E402

GRID = np.linspace(0.01, 0.99, 99).astype(np.float32)
COHORTS = ['train', 'val_seen', 'val_unseen']


# ----------------------------------------------------------------- cached data
def load(cache: Path, ck: str, cohort: str):
    TS = np.load(cache / f'{ck}__{cohort}__tsum.f16.npy')
    HS = np.load(cache / f'{ck}__{cohort}__hsum.f16.npy')
    P = np.load(cache / f'{ck}__{cohort}__prob.f16.npy').astype(np.float32)
    Y = np.load(cache / f'{cohort}__labels.i8.npy')
    tau_p = cache / f'{ck}__{cohort}__tau.f32.npy'
    TAU = np.load(tau_p).reshape(-1) if tau_p.exists() else None
    return TS, P, Y, TAU, HS


def counts(P, Y):
    """Per-patch TP/FP/FN at every grid threshold, via one histogram pass.

    A cell fires at GRID[j] iff p >= GRID[j], i.e. for every j below its own bin, so the
    per-threshold counts are suffix sums of a per-patch histogram. That is O(N*1024) once
    instead of O(99*N*1024) comparisons.
    """
    N = len(P)
    F = P.reshape(N, -1)
    keep = (Y >= 0).reshape(N, -1)
    pos = (Y == 1).reshape(N, -1) & keep
    neg = (Y == 0).reshape(N, -1) & keep
    b = np.searchsorted(GRID, F, side='right').astype(np.int32)        # 0..99
    idx = np.arange(N, dtype=np.int64)[:, None] * 100 + b
    hp = np.bincount(idx[pos], minlength=N * 100).reshape(N, 100)
    hn = np.bincount(idx[neg], minlength=N * 100).reshape(N, 100)
    del idx, b
    suf = lambda h: np.cumsum(h[:, ::-1], axis=1)[:, ::-1]             # noqa: E731
    cp, cn = suf(hp), suf(hn)
    TP = np.ascontiguousarray(cp[:, 1:].T.astype(np.float64))          # (99, N)
    FP = np.ascontiguousarray(cn[:, 1:].T.astype(np.float64))
    FN = pos.sum(1)[None, :].astype(np.float64) - TP
    return TP, FP, FN


def pooled(TP, FP, FN, sel):
    i = np.arange(TP.shape[1])
    t, f, n = TP[sel, i].sum(), FP[sel, i].sum(), FN[sel, i].sum()
    return t / (t + f + n)


def oracle(TP, FP, FN, iters=25):
    """Exact per-patch thresholds maximising POOLED IoU (Dinkelbach)."""
    D = TP + FP + FN
    sel = np.full(TP.shape[1], len(GRID) // 2)
    lam = pooled(TP, FP, FN, sel)
    for _ in range(iters):
        s = TP - lam * D
        sel = s.shape[0] - 1 - np.argmax(s[::-1], axis=0)               # ties -> highest tau
        new = pooled(TP, FP, FN, sel)
        if abs(new - lam) < 1e-10:
            lam = new
            break
        lam = new
    return GRID[sel], lam


def iou_at(P, Y, tau):
    """Pooled flood IoU when each patch is cut at its own tau (scalar or 32x32 map)."""
    keep, y = Y >= 0, Y == 1
    tau = np.asarray(tau, np.float32)
    if tau.ndim == 1:
        tau = tau[:, None, None]
    pred = (P >= tau) & keep
    tp = (pred & y).sum(); fp = (pred & ~y & keep).sum(); fn = (~pred & y & keep).sum()
    return float(tp / (tp + fp + fn))


# ----------------------------------------------------------------- features
def prob_features(P):
    """Order statistics of the probability map — what the shipped head cannot see."""
    F = P.reshape(len(P), -1)
    q = np.percentile(F, [10, 25, 50, 75, 90, 95, 99, 100], axis=1).T
    fr = np.stack([(F >= t).mean(1) for t in (.1, .2, .3, .4, .5, .6, .7, .8, .9)], 1)
    ent = -(F * np.log(F + 1e-6) + (1 - F) * np.log(1 - F + 1e-6)).mean(1)
    return np.column_stack([F.mean(1), F.std(1), q, fr, ent]).astype(np.float32)


def build_X(TS, P, use_tokens, use_prob, tok_dim, tok3=False):
    parts = []
    if use_tokens:
        parts.append(TS.astype(np.float32) if tok3 else TS[:, :tok_dim].astype(np.float32))
    if use_prob:
        parts.append(prob_features(P))
    return np.column_stack(parts)


# ----------------------------------------------------------------- head variants
def mlp2(in_dim, width=64):
    """Two outputs: 'silence this patch' logit, and a tau logit for patches that fire.

    The oracle threshold is 0.99 for ~78% of patches (every dry one) and spread out for the
    rest. Regressing that with a single MSE head just predicts the bulk and silences almost
    everything, which is what sank V5 on 2026-09-03. Splitting the decision from the level
    matches the shape of the target.
    """
    return keras.Sequential([
        layers.Input((in_dim,)),
        layers.LayerNormalization(epsilon=1e-6),
        layers.Dense(width, activation='gelu'),
        layers.Dense(2, bias_initializer=keras.initializers.Constant(-0.4)),
    ])


def mlp(in_dim, width=64):
    return keras.Sequential([
        layers.Input((in_dim,)),
        layers.LayerNormalization(epsilon=1e-6),
        layers.Dense(width, activation='gelu'),
        layers.Dense(1, bias_initializer=keras.initializers.Constant(-0.4)),
    ])


def pixel_head(in_dim, width=32):
    """DBNet-style: a per-PIXEL threshold map from the probability map + a broadcast summary."""
    pm = layers.Input((32, 32, 1)); ctx = layers.Input((in_dim,))
    c = layers.Dense(width, activation='gelu')(layers.LayerNormalization(epsilon=1e-6)(ctx))
    # broadcast the patch summary over the 32x32 grid. UpSampling2D rather than tf.tile:
    # Keras 3 forbids raw TF ops on the symbolic tensors of a Functional model.
    c = layers.UpSampling2D(size=(32, 32))(layers.Reshape((1, 1, width))(c))
    h = layers.Concatenate()([pm, c])
    h = layers.Conv2D(width, 3, padding='same', activation='gelu')(h)
    h = layers.Conv2D(width, 3, padding='same', activation='gelu')(h)
    out = layers.Conv2D(1, 1, bias_initializer=keras.initializers.Constant(-0.4))(h)
    return keras.Model([pm, ctx], out)


def thr_loss(y, p, tau_logit, sharpness, dry_weight):
    """soft_threshold_loss, accepting either a per-patch scalar or a 32x32 map."""
    tau = tf.nn.sigmoid(tau_logit)
    if len(tau.shape) == 2:
        return soft_threshold_loss(y, p, tau, sharpness=sharpness,
                                   pw_weight=0.0, dry_weight=dry_weight)
    y = tf.cast(y, tf.float32); p = tf.cast(p, tf.float32)
    is_pw = tf.cast(y < 0.0, tf.float32)
    tgt = tf.where(y < 0.0, tf.ones_like(y), y)
    w = 1.0 - is_pw
    ps = tf.sigmoid(sharpness * (p - tau))
    inter = tf.reduce_sum(w * ps * tgt, axis=[1, 2, 3])
    union = tf.reduce_sum(w * (ps + tgt - ps * tgt), axis=[1, 2, 3])
    soft = 1.0 - inter / (union + 1e-6)
    mass = tf.reduce_sum(w * ps, axis=[1, 2, 3]) / (tf.reduce_sum(w, axis=[1, 2, 3]) + 1e-6)
    hf = tf.cast(tf.reduce_sum((1.0 - is_pw) * tgt, axis=[1, 2, 3]) > 0.0, tf.float32)
    return tf.reduce_mean(hf * soft + (1.0 - hf) * dry_weight * mass)


def fit_head(Xtr, Ptr, Ytr, evals, kind='scalar', target=None,
             sharpness=20.0, dry_weight=1.0, epochs=6, batch=512, seed=0):
    """Train one head variant on TRAIN; return its tau on each eval cohort."""
    keras.utils.set_random_seed(seed)
    net = {'scalar': mlp, 'oracle2': mlp2, 'map': pixel_head}[kind](Xtr.shape[1])
    opt = keras.optimizers.Adam(1e-3)
    n = len(Xtr)
    Ytr4 = Ytr[..., None].astype(np.float32)
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(n)
        for s in range(0, n - batch + 1, batch):
            b = order[s:s + batch]
            x = tf.convert_to_tensor(Xtr[b])
            with tf.GradientTape() as tape:
                if kind == 'map':
                    logit = net([tf.convert_to_tensor(Ptr[b][..., None]), x], training=True)
                else:
                    logit = net(x, training=True)
                if kind == 'oracle2':                        # V5: supervised on the oracle
                    tgt = target[b]
                    sil = tf.cast(tgt >= 0.985, tf.float32)
                    ps = tf.nn.sigmoid(logit[:, 0]); pt = tf.nn.sigmoid(logit[:, 1])
                    bce = -tf.reduce_mean(sil * tf.math.log(ps + 1e-6)
                                          + (1 - sil) * tf.math.log(1 - ps + 1e-6))
                    fire = 1.0 - sil
                    mse = tf.reduce_sum(fire * tf.square(pt - tgt)) / (
                        tf.reduce_sum(fire) + 1e-6)
                    loss = bce + mse
                else:
                    loss = thr_loss(tf.convert_to_tensor(Ytr4[b]),
                                    tf.convert_to_tensor(Ptr[b][..., None]),
                                    logit, sharpness, dry_weight)
            opt.apply_gradients(zip(tape.gradient(loss, net.trainable_variables),
                                    net.trainable_variables))
    out = []
    for X, P in evals:
        taus = []
        for s in range(0, len(X), 4096):
            x = tf.convert_to_tensor(X[s:s + 4096])
            lo = net([tf.convert_to_tensor(P[s:s + 4096][..., None]), x],
                     training=False) if kind == 'map' else net(x, training=False)
            taus.append(np.asarray(tf.nn.sigmoid(lo)))
        t = np.concatenate(taus)
        if kind == 'oracle2':
            out.append(np.where(t[:, 0] >= 0.5, 0.99, t[:, 1]))
        else:
            out.append(t.reshape(len(X)) if kind == 'scalar' else t[..., 0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--weights', default='best_loss')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--max-train', type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    cache = run / 'head_cache'

    D = {c: load(cache, a.weights, c) for c in COHORTS}
    tok_dim = D['train'][0].shape[1] // 3
    if a.max_train:
        k = min(a.max_train, len(D['train'][0]))
        r = np.random.default_rng(0).choice(len(D['train'][0]), k, replace=False)
        D['train'] = tuple(x[r] if x is not None else None for x in D['train'])
    print(f"  run {run.name}  tokens D={tok_dim}  "
          + '  '.join(f'{c}={len(D[c][0]):,}' for c in COHORTS), flush=True)

    ORA, CEIL, CNT = {}, {}, {}
    for c in COHORTS:
        P, Y = D[c][1], D[c][2]
        CNT[c] = counts(P, Y)
        ORA[c], CEIL[c] = oracle(*CNT[c])
        print(f'    [{c}] oracle ceiling {CEIL[c]:.4f}', flush=True)

    # global threshold chosen on TRAIN, applied everywhere — the honest baseline
    TPt, FPt, FNt = CNT['train']
    gi = int(np.argmax([TPt[j].sum() / (TPt[j].sum() + FPt[j].sum() + FNt[j].sum())
                        for j in range(len(GRID))]))
    gthr = float(GRID[gi])

    rows = []
    def guard(name, fn):
        """A variant that blows up must not take the whole unattended sweep with it."""
        try:
            return record(name, fn())
        except Exception as e:                                  # noqa: BLE001
            print(f'    {name:38s} FAILED: {type(e).__name__}: {e}', flush=True)
            rows.append({'variant': name, 'val_seen': float('nan'),
                         'val_unseen': float('nan'), 'error': f'{type(e).__name__}: {e}'})
            return None

    def record(name, taus):
        r = {'variant': name}
        for c, t in zip(['train', 'val_seen', 'val_unseen'], taus):
            r[c] = iou_at(D[c][1], D[c][2], t)
        rows.append(r)
        print(f"    {name:34s} train {r['train']:.4f}   "
              f"val_seen {r['val_seen']:.4f}   val_unseen {r['val_unseen']:.4f}", flush=True)
        return r

    print(f'\n  baselines (global threshold {gthr:.2f}, selected on TRAIN)', flush=True)
    record('global threshold (from train)',
           [np.full(len(D[c][1]), gthr) for c in COHORTS])
    record('SHIPPED head (as trained)', [D[c][3] for c in COHORTS])
    record('ORACLE (uses truth)', [ORA[c] for c in COHORTS])

    # percentiles over a (425k, 1024) array are not cheap — compute once per cohort
    PF = {c: prob_features(D[c][1]) for c in COHORTS}
    # 'head' is the head's OWN summary, mean(LayerNorm(proj(tokens))) — the only feature set
    # that can reproduce the shipped head. 'tok3' is the richer raw-token summary.
    SETS = {'head': lambda c: [D[c][4].astype(np.float32)],
            'head+prob': lambda c: [D[c][4].astype(np.float32), PF[c]],
            'tok3+prob': lambda c: [D[c][0].astype(np.float32), PF[c]],
            'prob': lambda c: [PF[c]]}
    Xtr = {k: {c: np.column_stack(f(c)) for c in COHORTS} for k, f in SETS.items()}

    Ptr, Ytr = D['train'][1], D['train'][2]
    ev = lambda k: [(Xtr[k][c], D[c][1]) for c in COHORTS]                        # noqa: E731

    print('\n  V0/V1 — what the head can see', flush=True)
    for k in ['head', 'head+prob', 'tok3+prob', 'prob']:
        guard(f'V0/V1 input={k}', lambda k=k: fit_head(
            Xtr[k]['train'], Ptr, Ytr, ev(k), epochs=a.epochs))
    print('\n  V2 — dry_weight sweep (input=head+prob)', flush=True)
    for dw in [1.0, 3.0, 10.0, 30.0]:
        guard(f'V2 dry_weight={dw:g}', lambda dw=dw: fit_head(
            Xtr['head+prob']['train'], Ptr, Ytr, ev('head+prob'),
            dry_weight=dw, epochs=a.epochs))
    print('\n  V3 — sharpness (input=head+prob)', flush=True)
    guard('V3 sharpness=50', lambda: fit_head(
        Xtr['head+prob']['train'], Ptr, Ytr, ev('head+prob'),
        sharpness=50.0, epochs=a.epochs))
    print('\n  V4 — per-pixel threshold map', flush=True)
    guard('V4 pixel map (head+prob ctx)', lambda: fit_head(
        Xtr['head+prob']['train'], Ptr, Ytr, ev('head+prob'),
        kind='map', epochs=a.epochs))
    print('\n  V5 — supervised on the ORACLE threshold', flush=True)
    for k in ['head', 'head+prob']:
        guard(f'V5 oracle-supervised input={k}', lambda k=k: fit_head(
            Xtr[k]['train'], Ptr, Ytr, ev(k), target=ORA['train'], kind='oracle2',
            epochs=a.epochs))

    ship = next(r for r in rows if r['variant'] == 'SHIPPED head (as trained)')
    print('\n' + '=' * 92)
    print(f"{'variant':36s}{'train':>9s}{'val_seen':>10s}{'val_unseen':>12s}"
          f"{'vs shipped':>13s}{'passes':>9s}")
    print('=' * 100)
    for r in rows:
        if not np.isfinite(r['val_seen']):
            print(f"{r['variant']:36s}{'FAILED':>9s}")
            continue
        d1 = (r['val_seen'] / ship['val_seen'] - 1) * 100
        d2 = (r['val_unseen'] / ship['val_unseen'] - 1) * 100
        ok = 'YES' if (d1 > 5 and d2 > 5) else ''
        if r['variant'].startswith('ORACLE'):
            ok = '(truth)'
        print(f"{r['variant']:36s}{r['train']:9.4f}{r['val_seen']:10.4f}"
              f"{r['val_unseen']:12.4f}{d1:+6.1f}%{d2:+6.1f}%{ok:>9s}")
    print('=' * 100)
    print('DECISION RULE: adopt only if >+5% on BOTH cohorts.')
    out = run / 'head_cache' / f'{a.weights}__head_sweep.json'
    out.write_text(json.dumps({'run': run.name, 'weights': a.weights,
                               'global_threshold_from_train': gthr,
                               'ceiling': CEIL, 'rows': rows}, indent=2, default=float))
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
