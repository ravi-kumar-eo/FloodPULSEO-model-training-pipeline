"""Validation metrics — COPIED VERBATIM from the tested train_v2.py.

Shared by train_vit.py and train_unet.py so both backbones are scored identically.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


class PWMaskedAUC(keras.metrics.AUC):
    """keras AUC that honours the -1 PERMANENT-WATER MARKER used by the losses.

    PORTED VERBATIM from the tested train_v2.py (2026-08-19). keras.metrics.AUC casts -1 to a
    POSITIVE; permanent water is the easiest class in the dataset (PW AUPRC = 1.0000) and a
    large share of positives, so every PW cell arrived as a FREE TRUE POSITIVE and inflated
    val_auprc far above real flood skill (0.397 observed vs ~0.13-0.16 flood-only on test).

    NOTE (2026-08-28): the A/B runs set `label_union_permwater=True` on the 10 m path, which
    writes PW as +1 rather than -1 — so this masking never engaged there and their val_auprc
    of 0.87-0.91 DID include permanent water. Their flood-only test AUPRC was 0.13-0.21. Do
    not treat those val numbers as a target.
    """

    def __init__(self, *a, population='flood', pw_weight=0.2, **kw):
        """population: 'union' scores permanent water at `pw_weight` (matching the training
        loss); 'flood' drops it entirely. See the class docstring."""
        super().__init__(*a, **kw)
        self.population = str(population)
        self.pw_weight = float(pw_weight)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        is_pw = tf.cast(tf.less(y_true, 0.0), tf.float32)
        # PW is a MARKER, not a target: its union label is 1 by definition.
        y_true = tf.where(y_true < 0.0, tf.ones_like(y_true), y_true)
        if self.population == 'union':
            # Weight PW exactly as the segmentation loss does (PW_W), so the metric scores
            # the SAME population the model is optimised on. Selecting a checkpoint on a
            # metric that ignores 87% of positive cells is selecting on the wrong thing.
            keep = (1.0 - is_pw) + is_pw * self.pw_weight
        else:
            keep = 1.0 - is_pw                       # flood-only: PW dropped
        if sample_weight is not None:
            keep = keep * tf.cast(sample_weight, keep.dtype)
        return super().update_state(y_true, y_pred, sample_weight=keep)


class IoUAtBestThreshold(PWMaskedAUC):
    """Flood IoU maximised over the AUC threshold grid: max_t TP/(TP+FP+FN).

    PORTED VERBATIM from train_v2.py. Threshold-free, so it is immune to the calibration
    drift that makes a FIXED-threshold IoU measure the epoch's calibration rather than its
    skill — the reason the old code stopped reporting val_iou@0.35 as the decision metric.
    """

    def __init__(self, name='iou_best', num_thresholds=200, **kwargs):
        super().__init__(curve='PR', num_thresholds=num_thresholds, name=name, **kwargs)

    def _iou_per_threshold(self):
        tp = tf.cast(self.true_positives, tf.float32)
        fp = tf.cast(self.false_positives, tf.float32)
        fn = tf.cast(self.false_negatives, tf.float32)
        return tf.math.divide_no_nan(tp, tp + fp + fn)

    def result(self):
        return tf.reduce_max(self._iou_per_threshold())

    def best_threshold(self):
        iou = self._iou_per_threshold().numpy()
        th = np.asarray(getattr(self, '_thresholds', None))
        return float(th[int(np.argmax(iou))])


class BestThresholdLogger(keras.callbacks.Callback):
    """Inject the epoch's best-IoU threshold into `logs`. MUST precede CSVLogger."""

    def __init__(self, iou_best_metric):
        super().__init__()
        self._metric = iou_best_metric

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        if not any(k.startswith('val_') for k in logs):
            logs['best_threshold'] = float('nan')
            return
        try:
            logs['best_threshold'] = self._metric.best_threshold()
        except Exception:
            logs['best_threshold'] = float('nan')
