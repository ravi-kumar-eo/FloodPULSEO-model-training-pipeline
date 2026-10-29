"""Paired warm start for the weather experiments (Exp 2/3/4).

WHY PAIRED. The M0 seed spread is +-0.007 val_seg_loss. An unpaired 3-vs-3 test has SE ~0.006
and could never detect a weather gain of that size — which is the likely reason weather "never
helped" in earlier work. So each weather run starts from THIS SEED'S OWN Exp 1 checkpoint and
the gain is `this run - its own T0 checkpoint`, which cancels seed variance.

THE THREE PHASES (drawn in metadata/plots/arch_{vit,unet}_daily.png):
  1  load the Exp 1 checkpoint of the same seed into the static trunk — it fits exactly,
     because at temporal_days == 0 not one weather weight exists
  2  FREEZE the static trunk; train the temporal encoder + interaction block only
  3  unfreeze and fine-tune everything at a low LR

Phase 1 is only sound because both interaction blocks are the IDENTITY at initialisation
(ViT gate g = 0; U-Net gamma = 1, beta = 0, conv kernel = 0) and the weather->tau projection is
zero-initialised too. Verified: two very different weather inputs give max|dprob| and max|dtau|
of exactly 0.0 before any training. If that ever stops holding, the warm start is invalid and
the gain is no longer attributable to weather.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow import keras

# Every submodule that belongs to the WEATHER path. Anything not named here is static trunk and
# is frozen in phase 2. Listed explicitly rather than matched by prefix: a silent rename would
# otherwise quietly unfreeze the trunk and void the comparison.
WEATHER_ATTRS = ('day_embed', 'day_encoder', 'tcn', 'tpool', 'interaction', 'thr_weather',
                 'day_inject', 'ltae', 'aoi_enc',   # U-TAE path; AOI-weather encoder (final programme)
                 'day_tcn', 'cond_block',           # U-TAE 'film' conditioning (2026-10-05)
                 'patch_enc')                       # per-patch + upstream weather encoder (2026-10-07)


def weather_layers(backbone):
    return [getattr(backbone, a) for a in WEATHER_ATTRS if getattr(backbone, a, None) is not None]


def load_static_checkpoint(model, build_twin, weights_path, verbose=True):
    """Phase 1 — load THIS SEED's Exp 1 checkpoint into the static trunk.

    A straight `load_weights` cannot be used: the Exp 1 file contains no weather weights, so the
    layer trees differ and Keras either errors or, with `skip_mismatch=True`, SILENTLY skips —
    including trunk layers that should have loaded, which would leave the trunk at random init
    and make the measured "weather gain" meaningless.

    So: build a `temporal_days = 0` TWIN, whose tree is identical to the Exp 1 model, load the
    checkpoint into it EXACTLY (no skipping), then copy variable-by-variable into the real model
    matched on `Variable.path`. Anything the twin has must land; anything only the full model has
    is a weather weight and keeps its (zero) init.

    Raises unless every twin variable is matched and the trunk demonstrably moved.
    """
    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(
            f'warm start needs the Exp 1 checkpoint at {p}. The paired design REQUIRES this '
            f"seed's own T0 fit; substituting another seed's would reintroduce the seed variance "
            f'the pairing exists to cancel.')

    twin = build_twin()                       # temporal_days = 0 -> identical tree to Exp 1
    twin.load_weights(str(p))                 # exact load, no skip_mismatch

    # Match by CONSTRUCTION ORDER, not by `Variable.path`: Keras numbers layer instances from a
    # GLOBAL counter, so the same layer is `residual_u_net_1/...` in one model and
    # `residual_u_net_2/...` in the next and paths never line up between two instances. The
    # static sublayers are created in the same order in both models, so the static variables
    # appear in the same order; every pair is shape-checked, which catches any drift.
    wl = set()
    for layer in weather_layers(model.backbone):
        wl.update(id(v) for v in layer.weights)
    static_dst = [v for v in model.weights if id(v) not in wl]
    src = list(twin.weights)
    if len(static_dst) != len(src):
        raise RuntimeError(
            f'warm start: the weather model has {len(static_dst)} static variables but the Exp 1 '
            f'checkpoint has {len(src)}. The static path must be IDENTICAL between Exp 1 and the '
            f'weather models, and it is not.')
    bad = [(i, tuple(x.shape), tuple(y.shape)) for i, (x, y) in enumerate(zip(src, static_dst))
           if tuple(x.shape) != tuple(y.shape)]
    if bad:
        raise RuntimeError(f'warm start: shape mismatch at {len(bad)} positions, e.g. {bad[:3]}')

    moved = 0
    for x, y in zip(src, static_dst):
        before = y.numpy()
        y.assign(x)
        if not np.allclose(before, y.numpy()):
            moved += 1
    if moved == 0:
        raise RuntimeError('warm start: loading changed NOTHING. The checkpoint is not being '
                           'applied and the run would train from scratch while claiming a warm start.')
    if verbose:
        n_w = len(model.weights) - len(static_dst)
        print(f'  [warm start] {len(src)} static variables loaded from {p.name} '
              f'({moved} differed from init); {n_w} weather variables left at init')


def set_trunk_trainable(backbone, trainable: bool, verbose=True):
    """Phase 2/3 — freeze or unfreeze everything that is NOT the weather path."""
    wl = set(id(l) for l in weather_layers(backbone))
    n_frozen = n_free = 0
    for layer in backbone.layers:
        if id(layer) in wl:
            layer.trainable = True
            n_free += 1
        else:
            layer.trainable = bool(trainable)
            n_frozen += 0 if trainable else 1
    if verbose:
        state = 'UNFROZEN (phase 3)' if trainable else 'FROZEN (phase 2)'
        print(f'  [phases] static trunk {state}; {n_free} weather submodules always trainable')


class UnfreezeAt(keras.callbacks.Callback):
    """Flip from phase 2 to phase 3 at a fixed epoch, and drop the LR for fine-tuning.

    Recompilation is required for a trainable-flag change to take effect in Keras, and the
    optimiser state is rebuilt with it — which is intended: phase 3 is a different optimisation
    problem from phase 2, at a different learning rate.

    CONTRACT: `recompile(lr)` MUST compile with a **new** optimizer instance and build it
    eagerly. Keras binds an optimizer to the exact variable list it was built on, so reusing
    the phase-2 optimizer (built on the weather path alone) makes the first phase-3 step raise
    `ValueError: Unknown variable ... recreate a new optimizer instance`. Building its slots
    lazily is equally fatal: they would be created inside the retraced train function, and
    variable creation during tracing is illegal. See `core.py::_phase3_recompile`.
    """

    def __init__(self, backbone, epoch: int, fine_tune_lr: float, recompile, lr_cb=None):
        super().__init__()
        self.backbone, self.epoch = backbone, int(epoch)
        self.fine_tune_lr, self.recompile = float(fine_tune_lr), recompile
        # The LR SCHEDULE callback, so phase 3 can be handed its own decay. Without this the
        # schedule -- which runs FIRST every epoch and assigns unconditionally -- puts the
        # freshly unfrozen trunk straight back on the phase-2 learning rate, and
        # `fine_tune_lr` lasts exactly one epoch. See callbacks.enter_phase3.
        self.lr_cb = lr_cb

    def on_epoch_begin(self, epoch, logs=None):
        if epoch != self.epoch:
            return
        set_trunk_trainable(self.backbone, True)
        print(f'  [phases] epoch {epoch+1}: fine-tuning everything at lr={self.fine_tune_lr:g}')
        # Switch the SCHEDULE into phase 3 BEFORE recompiling, so every later epoch decays from
        # fine_tune_lr instead of snapping back to the phase-2 curve.
        if self.lr_cb is not None and hasattr(self.lr_cb, 'enter_phase3'):
            self.lr_cb.enter_phase3(self.epoch, self.fine_tune_lr)
        else:
            print('  [phases] WARNING: no LR schedule handed in — fine_tune_lr will survive '
                  'only this epoch and the trunk will be re-trained at the phase-2 rate')
        self.recompile(self.fine_tune_lr)
        # `fit()` traces the train function ONCE, before the epoch loop, and `compile()` sets
        # `train_function = None` (keras/src/trainers/trainer.py:223). Recompiling from a
        # callback therefore leaves fit() calling None on the very next batch:
        # `TypeError: 'NoneType' object is not callable`. Nothing rebuilds it -- only
        # evaluate() re-makes its own function, so validation hid the asymmetry. This killed
        # exp2 ViT s1 (epoch 12) and exp2 U-Net s1 (epoch 10) on 2026-09-10, ~4 h each.
        # Reproduced on Keras 3.15 with a 3-layer model; `force` is what makes it retrace
        # against the phase-3 trainable set.
        self.model.make_train_function(force=True)
