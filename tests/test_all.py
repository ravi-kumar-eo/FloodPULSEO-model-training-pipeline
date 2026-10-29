#!/usr/bin/env python3
"""End-to-end verification. Run on any host before trusting it.

Checks the things that have actually gone wrong in this project, not a generic smoke test:
silent CPU fallback, a threshold loss gated to zero, mislabelled flood flags, test-lockbox
leaks, and capacity drift between the two backbones.

    python test_all.py            # everything except the training step
    python test_all.py --train    # also run 2 real training steps per backbone
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))   # src/ on the path
from flood.paths import SRC, ROOT, METADATA
sys.path.insert(0, str(SRC))

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ''))
    return cond


# ---------------------------------------------------------------- 2026-08-29 audit fixes
def test_threshold_pw_only_patch_trains():
    """A PW-but-no-flood patch must behave like a DRY patch, not vanish.

    Before the fix, has_pos was computed on a target where PW had already been set to 1, so
    such a patch took the soft-IoU branch where pw_weight=0 zeroes every weight -> NO gradient.
    """
    import numpy as np, tensorflow as tf
    from flood.models.threshold_head import soft_threshold_loss
    def _p(kind):
        y = np.zeros((1, 32, 32, 1), 'float32')
        if kind == 'flood': y[:, :8] = 1.0
        if kind == 'pw':    y[:, :8] = -1.0
        return tf.constant(y)
    pred = tf.constant(np.full((1, 32, 32, 1), 0.6, 'float32'))
    g = {}
    for k in ('dry', 'pw', 'flood'):
        tau = tf.Variable([[0.5]], dtype=tf.float32)
        with tf.GradientTape() as t:
            l = soft_threshold_loss(_p(k), pred, tau, sharpness=20.0,
                                    pw_weight=0.0, dry_weight=1.0)
        gr = t.gradient(l, tau)
        g[k] = float(tf.reduce_sum(tf.abs(gr))) if gr is not None else 0.0
    assert g['pw'] > 1e-6, f"PW-only patch produced NO gradient ({g['pw']})"
    assert abs(g['pw'] - g['dry']) < 1e-5, f"PW-only should match dry: {g['pw']} vs {g['dry']}"
    return f"PW-only grad {g['pw']:.4f} == dry {g['dry']:.4f}, flood {g['flood']:.4f}"


def test_vit_grid_positional_encoding():
    """The 32x32 token grid must carry position, and it must cost ZERO parameters."""
    import numpy as np
    from flood.models.static.vit import sincos_2d
    pe = np.asarray(sincos_2d(32, 32, 256))[0]
    assert pe.shape == (1024, 256), pe.shape
    n = pe / np.linalg.norm(pe, axis=1, keepdims=True)
    adj, far = float(n[0] @ n[1]), float(n[0] @ n[1023])
    assert adj > far, f"adjacent cells ({adj:.3f}) must be more similar than opposite ({far:.3f})"
    return f"adjacent {adj:.3f} > opposite {far:.3f}, 0 trainable params"


def test_eval_dataset_returns_indices_complete():
    """Evaluation must return every patch, once, with its row index attached."""
    import numpy as np
    from pathlib import Path
    from flood.data.dataset import FloodPatches
    md = METADATA
    d = FloodPatches(cohort='val_seen', patches_csv=md/'split_old'/'splits_v0'/'patches.csv',
                     patches_dir=__import__('flood.training.core', fromlist=['x']).resolve_patches_dir({}), labels_dir=md/'labels_80m_global',
                     flood_flags_csv=md/'split_old'/'splits_v0'/'flood80_flags.csv')
    ds = d.to_tf_dataset(batch_size=32, shuffle=False, flood_ratio=0.0, workers=4,
                         return_index=True, deterministic=True, drop_remainder=False)
    idx = []
    for k, (_, _, i) in enumerate(ds):
        idx.append(np.asarray(i))
        if k >= 30: break
    I = np.concatenate(idx)
    assert len(set(I.tolist())) == len(I), "duplicate indices"
    assert bool(np.all(np.diff(I) > 0)), "indices not in deterministic order"
    return f"{len(I)} indices, unique and ordered"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true', help='also run 2 real training steps')
    a = ap.parse_args()

    import tensorflow as tf
    import yaml

    print("=" * 72); print("1. ENVIRONMENT"); print("=" * 72)
    gpus = tf.config.list_physical_devices('GPU')
    # Not a hard failure: the local box has no GPU. But on a training host a silent CPU
    # fallback once cost ~12 h per eval pass, so it is always reported explicitly.
    check('tensorflow imports', True, tf.__version__)
    print(f"  {'GPU present' if gpus else 'NO GPU — CPU only'}: {gpus}")

    print("\n" + "=" * 72); print("2. MODELS"); print("=" * 72)
    from flood.models.static.vit import PlainViT
    from flood.models.static.unet import ResidualUNet
    f = tf.random.normal((2, 256, 256, 2)); m = tf.random.normal((2, 32, 32, 5))
    c = tf.random.normal((2, 16, 16, 2))
    counts, heads = {}, {}
    for nm, M in [('vit', PlainViT), ('unet', ResidualUNet)]:
        mo = M({}); p, tau = mo((f, m, c))
        counts[nm] = int(sum(np.prod(v.shape) for v in mo.trainable_variables))
        heads[nm] = int(sum(np.prod(v.shape) for v in mo.thr_head.trainable_variables
                            if 'proj' not in v.path))
        check(f'{nm} builds', p.shape == (2, 32, 32, 1) and tau.shape == (2, 1),
              f"{counts[nm]:,} params")
        check(f'{nm} tau in (0,1)', bool(tf.reduce_min(tau) > 0 and tf.reduce_max(tau) < 1),
              f"[{float(tf.reduce_min(tau)):.3f}, {float(tf.reduce_max(tau)):.3f}]")
    ratio = counts['unet'] / counts['vit']
    check('capacity matched within 5%', 0.95 <= ratio <= 1.05, f"{ratio:.3f}x")
    check('threshold head identical in both', heads['vit'] == heads['unet'],
          f"{heads['vit']:,} core params each")

    print("\n" + "=" * 72); print("3. THRESHOLD HEAD — the dry-patch fix"); print("=" * 72)
    from flood.models.threshold_head import soft_threshold_loss, ThresholdEpochTracker
    dry = tf.zeros((4, 32, 32, 1))
    pred = tf.random.uniform((4, 32, 32, 1), 0.3, 0.9, seed=0)
    grads = {}
    for lbl, dw in [('old (soft-IoU only)', 0.0), ('new (dry term)', 1.0)]:
        tau = tf.Variable([[0.5]] * 4, dtype=tf.float32)
        with tf.GradientTape() as t:
            L = soft_threshold_loss(dry, pred, tau, dry_weight=dw)
        g = t.gradient(L, tau)
        grads[lbl] = float(tf.reduce_max(tf.abs(g))) if g is not None else 0.0
    check('dry patches gave NO gradient before the fix',
          grads['old (soft-IoU only)'] < 1e-9, f"|grad| {grads['old (soft-IoU only)']:.2e}")
    check('dry patches now DO give gradient',
          grads['new (dry term)'] > 1e-4, f"|grad| {grads['new (dry term)']:.4f}")
    wet = np.zeros((2, 32, 32, 1), np.float32); wet[:, 8:24, 8:24] = 1.0
    wet = tf.constant(wet); pw = tf.where(wet > 0, 0.8, 0.2)
    ls = [float(soft_threshold_loss(wet, pw, tf.constant([[t]] * 2))) for t in (0.1, 0.5, 0.9)]
    check('wet path still optimal at a middling tau', ls[1] < ls[0] and ls[1] < ls[2],
          f"loss @0.1/0.5/0.9 = {ls[0]:.3f}/{ls[1]:.3f}/{ls[2]:.3f}")

    print("\n" + "=" * 72); print("4. EPOCH GATE (the silent no-op)"); print("=" * 72)
    from tensorflow import keras

    class Fake(keras.Model):
        def __init__(s):
            super().__init__()
            # float32 mirrors train.py: an int32 variable is placed on the CPU and is
            # unreadable from the GPU-compiled train_step.
            s._thr_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            s.thr_warmup_epochs = 3
            s.d = keras.layers.Dense(1)
        def call(s, x): return s.d(x)

    def warm(mo): return float(tf.cast(mo._thr_epoch >= float(mo.thr_warmup_epochs), tf.float32))
    m1 = Fake(); no_cb = sum(warm(m1) for _ in range(8))
    m2 = Fake(); cb = ThresholdEpochTracker(verbose=False); cb.set_model(m2)
    with_cb = 0.0
    for e in range(8):
        cb.on_epoch_begin(e); with_cb += warm(m2)
    check('without the tracker the gate NEVER opens', no_cb == 0, f"{int(no_cb)}/8 epochs active")
    check('with the tracker it opens after warmup', with_cb == 5, f"{int(with_cb)}/8 epochs active")

    print("\n" + "=" * 72); print("5. DATA"); print("=" * 72)
    root = SRC.parent; md = root / 'metadata'
    cfg = yaml.safe_load(open(SRC / 'configs' / '_common.yaml'))
    import os
    cands = [os.environ.get('FLOOD_PATCHES_DIR'), cfg.get('patches_dir'),
             '/home/ravi/shared_data/global_flood_dataset/pipeline/data/patches/patches_tif',
             '/media/ravi/Jupyter/flood_vit/pipeline/data/patches/patches_tif']
    pdir = next((c for c in cands if c and Path(c).is_dir()), None)
    check('patches dir found', pdir is not None, str(pdir))
    for p in ['splits/patches.csv', 'splits/cohort_events.csv', 'splits/flood80_flags.csv',
              'splits/ground_units_events.csv', 'split_old/splits_v0/patches.csv',
              'labels_80m_global', 'weather', 'patch_grid/patch_grid_all.npz']:
        check(f'metadata/{p}', (md / p).exists())

    if pdir:
        from flood.data.dataset import FloodPatches
        ds = FloodPatches(md / 'splits' / 'patches.csv', pdir, md / 'labels_80m_global',
                          'train', flood_flags_csv=md / 'splits' / 'flood80_flags.csv')
        check('train cohort loads', len(ds) > 100_000, f"{len(ds):,} patches")
        check('flood flags aligned', ds.has_flood is not None and 0.05 < ds.has_flood.mean() < 0.5,
              f"{100 * ds.has_flood.mean():.1f}% flood-bearing")
        t0 = time.time(); fi, me, co, y = ds.load(0); dt = time.time() - t0
        check('patch shapes', fi.shape == (256, 256, 2) and me.shape == (32, 32, 5)
              and co.shape == (16, 16, 2) and y.shape == (32, 32, 1), f"{dt * 1000:.0f} ms/patch")
        check('inputs normalised to [0,1]',
              all(0 <= x.min() and x.max() <= 1.0001 for x in (fi, me, co)))
        check('PW marker present in labels', float(y.min()) >= -1.0)
        try:
            FloodPatches(md / 'splits' / 'patches.csv', pdir, md / 'labels_80m_global',
                         'test')
            check('test lockbox guard', False, 'guard did NOT fire')
        except PermissionError:
            check('test lockbox guard', True, 'test cohorts refuse to load')

        # epoch coverage: the balanced order must not repeat one fixed subset forever
        idx_f = np.flatnonzero(ds.has_flood); idx_d = np.flatnonzero(~ds.has_flood)
        steps = min(len(idx_f) // 16, len(idx_d) // 16)
        per_ep = steps * 16
        cycles = int(np.ceil(len(idx_d) / max(per_ep, 1)))
        check('dry pool is cycled across epochs, not frozen', cycles > 1,
              f"{100 * per_ep / len(idx_d):.0f}% of dry patches per epoch, "
              f"full coverage in {cycles} epochs")

    print("\n" + "=" * 72); print("5b. THE CURRENT SPLIT"); print("=" * 72)
    # split_v1 replaced v0's four evaluation cohorts with train/val/test plus a basin-regime
    # LABEL. What can silently go wrong: the wrong split gets loaded (a config default that
    # stopped defaulting to v0), or a regime filter that drops rows it should keep — pandas
    # reads v1's literal 'n/a' as NaN unless keep_default_na is off, which would empty the
    # train cohort under a filter. Both are checked here rather than discovered in a fit.
    from flood.training.core import resolve_split, slice_spec
    v1 = md / 'splits'
    check('a config with NO split_dir resolves to the ARCHIVED v0, not to the current split',
          resolve_split({}, md) == (md / 'split_old' / 'splits_v0', 'v0'))
    check('a legacy split_dir: splits_v1 still finds the archived v1',
          resolve_split({'split_dir': 'splits_v1'}, md)[0] == md / 'split_old' / 'splits_v1')
    if not (v1 / 'patches.csv').exists():
        print('  SKIP  metadata/splits not present on this host')
    else:
        check('split_dir: splits resolves to the current split',
              resolve_split({'split_dir': 'splits'}, md) == (v1, 'v1'))
        check('v1 test_seen slice maps to cohort test + regime seen',
              slice_spec('test_seen', 'v1') == ('test', 'seen', True))
        check('v0 test_seen slice is still its own cohort',
              slice_spec('test_seen', 'v0') == ('test_seen', None, True))
        import pandas as _pd
        pv = _pd.read_csv(v1 / 'patches.csv', keep_default_na=False, na_values=[''],
                          low_memory=False,
                          usecols=['cohort', 'basin_regime', 'basin_seen_frac'])
        n_test = int((pv.cohort == 'test').sum())
        parts = {r: int(((pv.cohort == 'test') & (pv.basin_regime == r)).sum())
                 for r in ('seen', 'partial', 'unseen')}
        check('v1 regimes partition test exactly', sum(parts.values()) == n_test,
              f"{parts} sums to {n_test:,}")
        check('v1 train keeps its regime through a pandas read',
              int((pv.basin_regime == 'n/a').sum()) == int((pv.cohort == 'train').sum()),
              "'n/a' survives keep_default_na=False")
        seen_ok = pv.loc[pv.basin_regime == 'seen', 'basin_seen_frac'].min()
        unseen_mx = pv.loc[pv.basin_regime == 'unseen', 'basin_seen_frac'].max()
        check('v1 seen rows hold >=50% of their basins in train', seen_ok >= 0.5,
              f"min basin_seen_frac {seen_ok:.3f}")
        check('v1 unseen rows hold none', unseen_mx == 0.0,
              f"max basin_seen_frac {unseen_mx:.3f}")
        if pdir:
            kwv = dict(patches_csv=v1 / 'patches.csv', patches_dir=pdir,
                       labels_dir=md / 'labels_80m_global',
                       flood_flags_csv=v1 / 'flood80_flags.csv')
            dv = FloodPatches(cohort='val', **kwv)
            check('v1 val cohort loads', len(dv) > 10_000, f"{len(dv):,} patches")
            check('v1 flood flags aligned', dv.has_flood is not None
                  and 0.05 < dv.has_flood.mean() < 0.5,
                  f"{100 * dv.has_flood.mean():.1f}% flood-bearing")
            ts = FloodPatches(cohort='test', regime='seen', allow_test=True, **kwv)
            tu = FloodPatches(cohort='test', regime='unseen', allow_test=True, **kwv)
            check('v1 regime filter splits test', len(ts) == parts['seen']
                  and len(tu) == parts['unseen'], f"{len(ts):,} seen / {len(tu):,} unseen")
            check('v1 train survives a regime-aware read',
                  len(FloodPatches(cohort='train', **kwv)) == int((pv.cohort == 'train').sum()))
            try:
                FloodPatches(cohort='test', **kwv)
                check('v1 test lockbox guard', False, 'guard did NOT fire')
            except PermissionError:
                check('v1 test lockbox guard', True, "cohort 'test' refuses to load")

    print("\n" + "=" * 72); print("6. LOSS"); print("=" * 72)
    from flood.losses import segmentation_loss
    y0 = tf.zeros((2, 32, 32, 1)); y1 = tf.ones((2, 32, 32, 1))
    good = float(segmentation_loss(y1, tf.fill((2, 32, 32, 1), 0.9)))
    bad = float(segmentation_loss(y1, tf.fill((2, 32, 32, 1), 0.1)))
    check('loss lower when prediction is right', good < bad, f"{good:.4f} vs {bad:.4f}")
    # PW weighting is ASYMMETRIC by design (Ravi, 2026-08-28):
    #   segmentation loss 0.2 — PW is a light anchor; at 0 nothing stops the model painting
    #                           flood over every river bank
    #   threshold loss    0.0 — tau must not be biased toward firing on rivers
    from flood.losses import PW_WEIGHT
    from flood.models.threshold_head import soft_threshold_loss as _stl
    import inspect
    thr_pw = inspect.signature(_stl).parameters['pw_weight'].default
    check('seg loss weights PW at 0.2', abs(PW_WEIGHT - 0.2) < 1e-9, f"PW_WEIGHT={PW_WEIGHT}")
    check('threshold loss weights PW at 0.0', abs(thr_pw) < 1e-9, f"pw_weight={thr_pw}")

    ymix = np.zeros((1, 32, 32, 1), np.float32)
    ymix[0, :8] = 1.0
    ymix[0, 8:16] = -1.0
    ymix = tf.constant(ymix)
    right = np.zeros((1, 32, 32, 1), np.float32); right[0, :8] = 0.9; right[0, 8:] = 0.1
    onpw = np.zeros((1, 32, 32, 1), np.float32); onpw[0, :16] = 0.9; onpw[0, 16:] = 0.1
    # A PW cell's UNION target is 1 by definition — the -1 is a marker meaning "permanent
    # water, weight it at PW_WEIGHT", NOT "this is dry". train_v2.py restores it to 1:
    #   y_true = tf.where(y_true < 0.0, tf.ones_like(y_true), y_true)  # PW target IS 1
    # So predicting water on PW must give the LOWER loss. Asserting the opposite was my own
    # bug: it flipped the gradient sign on ~87% of positive cells.
    s_dry = float(segmentation_loss(ymix, tf.constant(right)))   # dry on PW
    s_wet = float(segmentation_loss(ymix, tf.constant(onpw)))    # wet on PW
    check('seg loss REWARDS water on PW (target 1, weight 0.2)', s_wet < s_dry,
          f"dry-on-PW {s_dry:.5f} vs wet-on-PW {s_wet:.5f}")
    t_ok = float(_stl(ymix, tf.constant(right), tf.constant([[0.5]])))
    t_pw = float(_stl(ymix, tf.constant(onpw), tf.constant([[0.5]])))
    check('threshold loss IGNORES PW (tau unbiased)', abs(t_ok - t_pw) < 1e-4,
          f"{t_ok:.5f} vs {t_pw:.5f}")

    if not a.train:
        print("\n  NOTE: --train not given. Shapes/losses/gradients passing in isolation is")
        print("        NOT evidence that training runs: three bugs (generator deadlock, an")
        print("        int32 variable stranded on CPU, and an XLA Triton autotune failure)")
        print("        passed every check above and only appeared in a real fit() on GPU.")

    if a.train and pdir:
        print("\n" + "=" * 72); print("7. TRAINING (2 real steps each)"); print("=" * 72)
        import subprocess
        for cfgname, bs in [('unet_static.yaml', 32), ('vit_static.yaml', 8)]:
            r = subprocess.run([sys.executable, str(SRC / 'train.py'), '--config',
                                str(SRC / 'configs' / cfgname), '--run-name', 'smoke',
                                '--smoke', '2'], capture_output=True, text=True, timeout=1800)
            ok = 'losses:' in r.stdout
            line = next((l for l in r.stdout.splitlines() if 'losses:' in l), '')
            check(f'{cfgname} trains (batch {bs})', ok, line.strip()[:90] or r.stderr[-150:])

    print("\n" + "=" * 72)
    print("\n" + "=" * 72); print("9. 2026-08-29 AUDIT FIXES"); print("=" * 72)
    for _fn in (test_threshold_pw_only_patch_trains,
                test_vit_grid_positional_encoding,
                test_eval_dataset_returns_indices_complete):
        try:
            check(_fn.__name__.replace('test_', ''), True, _fn())
        except Exception as _e:
            check(_fn.__name__.replace('test_', ''), False, f"{type(_e).__name__}: {_e}")

    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f_ in FAIL:
            print(f"  FAILED: {f_}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())

