"""Segmentation loss — COPIED VERBATIM from the tested train_v2.py.

    L = 0.4 * Focal(alpha=0.85, gamma=2.0)
      + 0.4 * Tversky(alpha=0.3, beta=0.7)
      + 0.2 * Boundary(boundary_weight=2.0)

Nothing here is rewritten. The only change from train_v2.py is that PW_W is set from the
config at import time instead of via a `global` statement in train(), and a module-level
`segmentation_loss()` wrapper is provided so train.py can call it as a function.

WHY VERBATIM (2026-08-28): a from-scratch reimplementation introduced three silent defects —
the permanent-water target was set to 0 instead of 1 (inverting the gradient on ~87% of all
positive cells), focal alpha was 0.75 instead of 0.85 and was applied as a class-balancing
term rather than a flat multiplier, and the boundary edge was a soft difference instead of a
binary threshold. None of these were visible in the loss curve.

PERMANENT WATER: the loader writes -1 at PW cells. That is a MARKER, not a target — a PW
cell's union target is 1 by definition. Each loss restores y=1 and applies weight PW_W, so
PW contributes at 0.2 rather than being inverted or dropped.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

PW_W = 0.2          # pw_loss_weight from every A/B experiment


class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.85, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # exp6 P1: y_true == -1 marks permanent water -> excluded. Mean is over KEPT cells only,
        # otherwise masking would silently shrink the loss by the PW fraction instead of
        # reweighting it.
        is_pw = tf.cast(tf.less(y_true, 0.0), tf.float32)
        w = (1.0 - is_pw) + is_pw * PW_W
        y_true = tf.where(y_true < 0.0, tf.ones_like(y_true), y_true)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - pt, self.gamma)
        return tf.reduce_sum(w * focal_weight * bce) / (tf.reduce_sum(w) + 1e-6)


class TverskyLoss(keras.losses.Loss):
    """
    Tversky loss with α=0.3 (FP penalty), β=0.7 (FN penalty).
    β > α explicitly penalizes missed flood pixels (FN) more than false alarms (FP),
    which pushes recall up — the main bottleneck at precision=0.98, recall=0.77.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-4, name='tversky_loss'):
        super().__init__(name=name)
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # exp6 P1: y_true == -1 marks a permanent-water cell that must not contribute. w=0 there.
        is_pw = tf.cast(tf.less(y_true, 0.0), tf.float32)      # -1 marks permanent water
        w = (1.0 - is_pw) + is_pw * PW_W                       # PW_W=0 -> masked (exp6 behaviour)
        y_true = tf.where(y_true < 0.0, tf.ones_like(y_true), y_true)   # PW target IS 1
        tp = tf.reduce_sum(w * y_true * y_pred, axis=[1, 2, 3])
        fp = tf.reduce_sum(w * (1 - y_true) * y_pred, axis=[1, 2, 3])
        fn = tf.reduce_sum(w * y_true * (1 - y_pred), axis=[1, 2, 3])
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tf.reduce_mean(tversky)


class BoundaryLoss(keras.losses.Loss):
    """
    Boundary-weighted loss. Computes a distance-weighted map from the flood mask
    boundary using morphological erosion/dilation approximated with max-pooling,
    then applies higher weight to predictions near the boundary where FN errors
    are concentrated (mixed pixels, partial inundation).
    Weight = 1 + boundary_weight * boundary_map, so non-boundary pixels still
    contribute but boundary pixels are penalized 3x more.
    """
    def __init__(self, boundary_weight=2.0, name='boundary_loss'):
        super().__init__(name=name)
        self.boundary_weight = boundary_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        # exp6 P1: strip the -1 permanent-water sentinel BEFORE the morphology. A raw -1 would
        # corrupt the erosion (-max_pool(-y)) and invent boundaries along every river bank.
        is_pw = tf.cast(tf.less(y_true, 0.0), tf.float32)
        w = (1.0 - is_pw) + is_pw * PW_W
        y_true = tf.where(y_true < 0.0, tf.ones_like(y_true), y_true)

        # Approximate dilation and erosion with max/min pooling (3×3 kernel)
        # dilation: max over 3×3 neighbourhood
        dilated = tf.nn.max_pool2d(y_true, ksize=3, strides=1, padding='SAME')
        # erosion: min over 3×3 = negative of max-pool of negative
        eroded  = -tf.nn.max_pool2d(-y_true, ksize=3, strides=1, padding='SAME')
        # boundary pixels = dilated XOR eroded (pixels in dilation but not erosion)
        boundary = tf.cast(dilated - eroded > 0.5, tf.float32)

        # Per-pixel weight map: boundary pixels get (1 + boundary_weight), rest get 1
        weight_map = 1.0 + self.boundary_weight * boundary

        # Weighted BCE
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_sum(w * weight_map * bce) / (tf.reduce_sum(w) + 1e-6)


class CombinedLoss(keras.losses.Loss):
    """
    L = 0.4 × Focal(α=0.85, γ=2.0)
      + 0.4 × Tversky(α, β)        # α/β configurable (recall-weighting)
      + 0.2 × BoundaryLoss
    Tversky defaults α=0.3/β=0.7; pass tversky_alpha/tversky_beta to favour recall
    (e.g. α=0.2/β=0.8 penalises false-negatives harder).
    """
    def __init__(self, tversky_alpha=0.3, tversky_beta=0.7, name='combined_loss'):
        super().__init__(name=name)
        self.focal    = FocalLoss(alpha=0.85, gamma=2.0)
        self.tversky  = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.boundary = BoundaryLoss(boundary_weight=2.0)

    def call(self, y_true, y_pred):
        return (0.4 * self.focal(y_true, y_pred)
              + 0.4 * self.tversky(y_true, y_pred)
              + 0.2 * self.boundary(y_true, y_pred))



# --- module-level handles so train.py can call the composite as a plain function ---
PW_WEIGHT = PW_W                       # name kept for the existing imports/tests
_COMBINED = CombinedLoss(tversky_alpha=0.3, tversky_beta=0.7)


def segmentation_loss(y_true, y_pred, **_ignored):
    """0.4*focal + 0.4*tversky + 0.2*boundary, exactly as CombinedLoss defines it."""
    return _COMBINED(y_true, y_pred)


def focal_loss(y_true, y_pred):
    return _COMBINED.focal(y_true, y_pred)


def tversky_loss(y_true, y_pred):
    return _COMBINED.tversky(y_true, y_pred)


def boundary_loss(y_true, y_pred):
    return _COMBINED.boundary(y_true, y_pred)
