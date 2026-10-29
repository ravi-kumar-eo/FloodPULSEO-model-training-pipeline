"""Per-patch decision threshold — a STANDARD component of the baseline, in BOTH backbones.

This is not an experiment, a variant, or an ablation arm. Every confirmatory run carries it,
which is why nothing in the run naming mentions it.

WHY A PER-PATCH THRESHOLD
-------------------------
Scoring every 32x32 patch at one global threshold is wrong almost everywhere: measured on the
T=7 model, only 5.4% of patches have an optimum within +-0.05 of the global value, and the
oracle inter-quartile range is 0.05-0.775. Scoring each patch at its OWN oracle threshold
raised mean flood IoU 0.1966 -> 0.3076.

Two honest caveats about that number, because it is the motivation and it is inflated:
  * it was measured on 24,891 flood-containing tiles only, excluding ~28,000 dry tiles.
    Deployment cannot know in advance which tiles contain flood.
  * it is an ORACLE (it uses test truth to pick tau), so it is an upper bound, not a target.
    A gradient-boosted regressor on probability-map summary statistics recovered only 8.9% of
    it out of sample, though its predicted tau correlated +0.589 with the oracle.

This head sees the TRUNK TOKENS rather than summarised probabilities, so it is a strictly
richer hypothesis class than that regressor. Whether that is enough is an open question the
thesis answers empirically.

WHY IT IS TINY
--------------
Thresholding `p >= tau` is identical to `(p - tau + 0.5) >= 0.5`, i.e. a per-patch shift of
the sigmoid. It is a low-dimensional calibration route, decoupled from pixel segmentation.
Giving it real capacity would let it re-solve segmentation through a 1-D bottleneck, which is
the worst possible place to spend parameters. ~25k params against a ~9-21M trunk (~0.1-0.3%).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ThresholdHead(layers.Layer):
    """(B, N, dim) trunk tokens -> (B, 1) decision threshold in (0, 1), one per patch.

    Backbone-agnostic: it only needs a token sequence, so the ViT passes its trunk tokens and
    the U-Net passes its flattened bottleneck. Identical parameter count in both, which is
    what keeps M0 a comparison of backbones rather than of heads.
    """

    def __init__(self, width: int = 64, in_dim: int = 256, **kw):
        super().__init__(**kw)
        self.width = int(width)
        # Backbones expose different feature widths (ViT trunk 256, U-Net bottleneck 336), so
        # without a projection the head would have a DIFFERENT parameter count in each — and
        # M0 would silently compare two heads as well as two backbones. Projecting to a common
        # in_dim first makes the head provably identical on both sides.
        self.in_dim = int(in_dim)

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.proj = (layers.Dense(self.in_dim, use_bias=False, name='thr_proj')
                     if d != self.in_dim else None)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name='thr_norm')
        self.d1 = layers.Dense(self.width, activation='gelu', name='thr_dense1')
        # float32 output: tau is a decision boundary, and bf16 rounding here is visible
        # in the metric. bias -0.4 -> sigmoid ~0.40 at init, near the global optimum
        # measured on the baseline (flood-only tau 0.496), so training starts sane.
        self.d2 = layers.Dense(1, dtype='float32', name='thr_dense2',
                               bias_initializer=keras.initializers.Constant(-0.4))
        super().build(input_shape)

    def call(self, tokens, training=False):
        x = self.proj(tokens) if self.proj is not None else tokens
        x = self.norm(x)
        x = tf.reduce_mean(x, axis=1)          # (B, in_dim) — patch-level summary
        x = self.d1(x)
        return tf.nn.sigmoid(self.d2(x))       # (B, 1) in (0, 1)

    def get_config(self):
        return {**super().get_config(), 'width': self.width, 'in_dim': self.in_dim}


def soft_threshold_loss(y_true, y_pred, tau,
                        sharpness: float = 20.0,
                        pw_weight: float = 0.0,
                        dry_weight: float = 1.0):
    """Differentiable objective for tau. Soft-IoU on wet patches + a dry-patch term.

    THE DRY-PATCH PROBLEM (fixed here; the archived version had only the soft-IoU term).
    Soft IoU is  1 - inter/union  with  inter = sum(w * p_soft * tgt). For an ALL-DRY patch
    tgt is 0 everywhere, so inter = 0 for ANY tau and the gradient w.r.t. tau vanishes. The
    head could therefore never learn to raise tau to suppress false positives — which is
    exactly the failure mode of this model: flood precision 0.176, FAR 0.824 on the 2026-08-27
    baseline test. Roughly 84% of patches have no flood at 80 m, so soft-IoU alone leaves the
    head blind on the large majority of its input.

    The fix: on dry patches, penalise the soft-positive MASS directly. Any probability that
    survives the threshold is a false positive, so `mean(p_soft)` is a proper cost with a
    non-zero gradient in tau, pushing tau up until the patch goes quiet.

    PERMANENT WATER CARRIES WEIGHT 0, matching `losses.PW_WEIGHT`.

    BUG FIXED 2026-08-28 (caught by Ravi asking, not by any test): the default was 0.2 while
    the segmentation loss used 0.0, so the two objectives scored DIFFERENT pixel populations.
    Measured on a patch that is 1/4 flood and 1/4 permanent water, a model that wrongly fires
    on the PW cells scored a threshold loss of 0.00089 versus 0.16736 for one that correctly
    ignores them — i.e. tau was being trained to fire on rivers, the exact failure the head
    exists to prevent. The segmentation loss was unaffected, so only tau was biased.

    If these two weights ever diverge again, tau is calibrated for a population that is never
    scored. They must match.
    """
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    tau = tf.reshape(tf.cast(tau, tf.float32), [-1, 1, 1, 1])

    is_pw = tf.cast(y_true < 0.0, tf.float32)
    tgt = tf.where(y_true < 0.0, tf.ones_like(y_true), y_true)   # PW counts as positive
    w = 1.0 - is_pw * (1.0 - pw_weight)                          # 1.0 flood/dry, 0.2 PW

    # relaxed decision: -> the hard indicator as sharpness -> inf
    p_soft = tf.sigmoid(sharpness * (y_pred - tau))

    inter = tf.reduce_sum(w * p_soft * tgt, axis=[1, 2, 3])
    union = tf.reduce_sum(w * (p_soft + tgt - p_soft * tgt), axis=[1, 2, 3])
    soft_iou_loss = 1.0 - inter / (union + 1e-6)                 # (B,)

    # per-patch positive mass, used only where the patch has no positive target
    pos_mass = tf.reduce_sum(w * p_soft, axis=[1, 2, 3]) / (
        tf.reduce_sum(w, axis=[1, 2, 3]) + 1e-6)                 # (B,) in [0, 1]
    # BUG FIXED 2026-08-29: this tested `tgt`, which has already had permanent water set to
    # 1. A patch with PW but NO flood therefore took the soft-IoU branch -- where every
    # weight is pw_weight=0, so it produced NO gradient at all -- instead of the dry branch
    # that teaches tau to suppress false flood over its non-PW area. Test ACTUAL flood:
    # PW keeps zero influence, but a PW-only patch now trains the threshold like a dry one.
    has_flood = tf.cast(
        tf.reduce_sum((1.0 - is_pw) * tgt, axis=[1, 2, 3]) > 0.0, tf.float32)
    has_pos = has_flood

    per_patch = has_pos * soft_iou_loss + (1.0 - has_pos) * dry_weight * pos_mass
    return tf.reduce_mean(per_patch)


class ThresholdEpochTracker(keras.callbacks.Callback):
    """Push the current epoch into `model._thr_epoch` so the warmup gate can move.

    BUG THIS PREVENTS (found 2026-08-27 in train_thrhead.py): `_cur_epoch` was created as a
    tf.Variable and READ by the warm gate, but never assigned anywhere. With warmup 3 the gate
    evaluated `(0 >= 3) = 0` on every step of every epoch, so the threshold loss was multiplied
    by ZERO for the entire run. The head trained, emitted tau, and received no gradient from
    its own objective — a silent no-op that looks like "the head does not help" after a full
    training budget. Register this callback whenever the threshold loss is active.
    """

    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = bool(verbose)

    def on_epoch_begin(self, epoch, logs=None):
        var = getattr(self.model, '_thr_epoch', None)
        if var is None:
            return
        var.assign(float(epoch))   # float32: keeps the variable on the GPU (see train.py)
        if self.verbose:
            warm = int(getattr(self.model, 'thr_warmup_epochs', 0))
            state = 'ACTIVE' if epoch >= warm else f'held (starts epoch {warm + 1})'
            print(f"  [thr] epoch {epoch+1} | threshold loss {state}")
