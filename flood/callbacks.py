"""Training callbacks — COPIED VERBATIM from the tested train_v2.py.

Nothing here is rewritten. These five callbacks were in every A/B experiment and were
missing from the from-scratch trainer:

  WarmupPolyDecay          5-epoch linear warmup then linear decay to min_lr. The docstring
                           notes it "matches Swin Transformer / SegFormer official training
                           schedule" — its absence is the most likely cause of the ViT
                           collapsing at epoch 7 with no warmup on a constant LR.
  WeightEMA                exponential moving average of the weights, momentum 0.9,
                           starting after warmup.
  PostWarmupBestCheckpoint best-model checkpoint that IGNORES pre-warmup epochs, so a lucky
                           epoch-1 value cannot win the run.
  PeriodicCheckpoint       rolling per-epoch state for resume.
  StepStatusCallback       writes train_status.json for the dashboard.

Shared by train_vit.py and train_unet.py so both backbones use the identical protocol —
which is what makes M0 a comparison of architectures rather than of training recipes.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


class OneBasedCSVLogger(keras.callbacks.CSVLogger):
    """CSVLogger that writes 1-BASED epochs, matching what the progress bar already prints.

    Keras prints `Epoch 1/N` but its CSVLogger writes `epoch,0` for that SAME epoch — a known
    upstream inconsistency (keras-team/keras#20212). Two records of one run that disagree by
    one is a reporting trap, and it produced exactly that: months of reports saying "best at
    epoch 0" for what `train.log` called `Epoch 1/100`. The field numbers epochs from 1; there
    is no such thing as an epoch 0.

    NOTE `last_epoch.json` deliberately stays 0-BASED — it feeds Keras `initial_epoch`, which
    is a 0-based index, not a human-facing number. See EpochStamp in training/core.py.
    """

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch + 1, logs)


class WarmupPolyDecay(keras.callbacks.Callback):
    """Linear warmup then polynomial (linear, power=1.0) decay to min_lr.

    Matches Swin Transformer / SegFormer official training schedule.
    Warmup: min_lr → peak_lr over warmup_epochs.
    Decay:  peak_lr → min_lr linearly over (total_epochs - warmup_epochs).
    """

    def __init__(self, warmup_epochs, total_epochs, peak_lr, min_lr=1e-7):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = int(total_epochs)
        self.decay_epochs  = total_epochs - warmup_epochs
        self.peak_lr       = peak_lr
        self.min_lr        = min_lr
        self._p3 = None                       # set by phases.UnfreezeAt; see enter_phase3

    def enter_phase3(self, start_epoch, peak_lr):
        """Hand phase 3 its OWN decay, from `peak_lr` down over the epochs that remain.

        WHY THIS EXISTS. `UnfreezeAt` used to set `fine_tune_lr` directly on the optimizer, and
        it survived exactly ONE epoch: this callback is FIRST in the list and assigns
        unconditionally every `on_epoch_begin`, so from the next epoch the trunk -- newly
        unfrozen -- was back on the full schedule. Measured on exp2 seed 1 (2026-09-11):
        intended 1.2e-5, actual 1.20e-4 from epoch 16, i.e. 10x. The converged Exp 1 trunk was
        therefore RE-TRAINED at near-full rate rather than fine-tuned, which is the shape of
        what both runs did (sharp dip at the unfreeze, then oscillation that never settles,
        and phase 3 never beating phase 2).
        """
        self._p3 = (int(start_epoch), float(peak_lr))

    def on_epoch_begin(self, epoch, logs=None):
        if self._p3 is not None and epoch >= self._p3[0]:
            start, peak = self._p3
            span = max(self.total_epochs - start, 1)
            progress = min((epoch - start) / span, 1.0)
            lr = (peak - self.min_lr) * (1.0 - progress) + self.min_lr    # linear, as phase 2
            if epoch == start:
                print(f"\n  [phase3-lr] decaying from {peak:.2e} over epochs "
                      f"{start+1}-{self.total_epochs}")
        elif epoch < self.warmup_epochs:
            lr = self.min_lr + (self.peak_lr - self.min_lr) * (epoch + 1) / self.warmup_epochs
            print(f"\n  [Warmup] epoch {epoch+1}/{self.warmup_epochs}: lr={lr:.2e}")
        else:
            progress = (epoch - self.warmup_epochs) / self.decay_epochs
            lr = (self.peak_lr - self.min_lr) * (1.0 - progress) + self.min_lr
        self.model.optimizer.learning_rate.assign(lr)


class WeightEMA(keras.callbacks.Callback):
    """Per-epoch exponential moving average of model weights (SWA-style).

    Implemented manually (numpy on get_weights/set_weights) so it is robust across
    Keras versions — unlike the optimizer's use_ema, which is a silent no-op under
    some keras/tf_keras builds. Averaging starts after `start_epoch` (skip the noisy
    warmup). Call apply_to_model() at the end to load the smoothed weights.
    """

    def __init__(self, momentum=0.9, start_epoch=0):
        super().__init__()
        self.momentum    = momentum
        self.start_epoch = start_epoch
        self.shadow      = None
        self.n           = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        w = self.model.get_weights()
        if self.shadow is None:
            self.shadow = [x.copy() for x in w]
        else:
            m = self.momentum
            self.shadow = [m * s + (1.0 - m) * x for s, x in zip(self.shadow, w)]
        self.n += 1

    def apply_to_model(self):
        """Swap the EMA-averaged weights into the model. Returns True on success."""
        if self.shadow is None:
            return False
        try:
            self.model.set_weights(self.shadow)
            return True
        except Exception as e:
            print(f"  [EMA] apply failed ({e}) — keeping raw weights")
            return False


class PostWarmupBestCheckpoint(keras.callbacks.Callback):
    """Save best_model on val-metric improvement, but ONLY after warmup completes.

    Early-warmup val metrics are very noisy (the LR is still ramping and the model
    is barely trained), so plain save_best_only often latches onto a meaningless
    early spike. Gating selection to post-warmup epochs avoids that.
    """

    def __init__(self, filepath, warmup_epochs, monitor='val_iou', mode='max'):
        super().__init__()
        self.filepath      = str(filepath)
        self.warmup_epochs = warmup_epochs
        self.monitor       = monitor
        self.sign          = 1.0 if mode == 'max' else -1.0
        # BUG FIXED 2026-08-20: `best` was hard-coded to -inf, which only works for mode='max'.
        # With mode='min' the test is  sign*v > sign*best  ->  -v > +inf , which is NEVER true,
        # so the callback silently never fired and best_model.weights.h5 was never written.
        # Seed it to the WORST value for the chosen direction instead.
        self.best          = -float('inf') if mode == 'max' else float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) <= self.warmup_epochs:
            return  # ignore warmup epochs entirely
        v = logs.get(self.monitor)
        if v is None:
            return
        if self.sign * v > self.sign * self.best:
            self.best = v
            self.model.save_weights(self.filepath)
            print(f"\n  [Best] epoch {epoch+1}: {self.monitor}={v:.4f} (post-warmup) → saved best_model")


class PeriodicCheckpoint(keras.callbacks.Callback):
    """Save weights every `interval` epochs to allow training resumption.

    Also writes `epoch_XXXX.state.npz` holding the optimizer moments and the EMA accumulator, so a
    resume is exact rather than approximate (B-fix 4).
    """

    def __init__(self, checkpoints_dir, interval, ema_cb=None):
        super().__init__()
        self.checkpoints_dir = Path(checkpoints_dir)
        self.interval        = interval
        self.ema_cb          = ema_cb

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            path = self.checkpoints_dir / f"epoch_{epoch+1:04d}.weights.h5"
            self.model.save_weights(str(path))
            # companion state file; np.savez appends .npz itself, so name it without the suffix
            state_path = self.checkpoints_dir / f"epoch_{epoch+1:04d}.state.npz"
            save_train_state(state_path, self.model.optimizer, self.ema_cb, epoch + 1)
            print(f"\n  [Checkpoint] epoch {epoch+1} saved → {path.name} (+ {state_path.name})")
            # keep only the latest periodic checkpoint (save new BEFORE deleting old →
            # crash-safe; best_model.weights.h5 is untouched by this glob).
            for old in list(self.checkpoints_dir.glob("epoch_*.weights.h5")) + \
                       list(self.checkpoints_dir.glob("epoch_*.state.npz")):
                if old not in (path, state_path):
                    try:
                        old.unlink()
                    except OSError:
                        pass


class StepStatusCallback(keras.callbacks.Callback):
    """Write a live train_status.json every `every` steps so the dashboard shows
    WITHIN-epoch progress. CSVLogger only writes one row per epoch — useless when an
    epoch takes ~90 min. Atomic (.tmp + os.replace) so the dashboard never reads a
    half-written file."""

    def __init__(self, path, total_epochs, every=50):
        super().__init__()
        self.path = str(path)
        self.total_epochs = int(total_epochs)
        self.every = int(every)
        self.epoch = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def _write(self, step, logs):
        spe = (self.params or {}).get('steps') or 0
        d = {"kind": "train", "epoch": self.epoch, "total_epochs": self.total_epochs,
             "step": int(step), "steps_per_epoch": int(spe), "ts": time.time()}
        for k in ("loss", "iou", "precision", "recall", "auprc", "iou_best",
                  "swap_loss", "swap_w",          # live swap readout (2026-08-01)
                  "val_auprc", "val_iou_best", "best_threshold"):
            if logs and k in logs:
                d[k] = float(logs[k])
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, self.path)
        except OSError:
            pass

    def on_train_batch_end(self, batch, logs=None):
        if (batch + 1) % self.every == 0:
            self._write(batch + 1, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._write((self.params or {}).get('steps') or 0, logs)

class PolyDecay(keras.callbacks.Callback):
    """nnU-Net's `poly` schedule: lr = peak * (1 - epoch/total)**0.9, no warmup.

    THE CNN STANDARD, deliberately different from WarmupPolyDecay above.

    nnU-Net (Isensee et al., Nature Methods 2021) decays by (1 - epoch/max_epochs)^0.9 from
    epoch 1 with NO warmup, and a systematic evaluation of LR policies for 2D U-Net brain
    tumour segmentation found poly decay significantly outperformed the alternatives.

    Warmup exists to stop a large ADAPTIVE learning-rate variance distorting early gradients
    — an Adam/AdamW pathology. Applying a 5-epoch transformer warmup to a CNN imports a fix
    for a problem it does not have, so M0 gives each backbone its own literature-standard
    schedule (thesis_plan §4.1 permits "architecture-appropriate learning rates").
    """

    def __init__(self, total_epochs, peak_lr, power=0.9, min_lr=1e-7):
        super().__init__()
        self.total_epochs = int(total_epochs)
        self.peak_lr = float(peak_lr)
        self.power = float(power)
        self.min_lr = float(min_lr)
        self._p3 = None                       # set by phases.UnfreezeAt; see enter_phase3

    def enter_phase3(self, start_epoch, peak_lr):
        """Phase 3 gets its own poly decay from `peak_lr`. See WarmupPolyDecay.enter_phase3
        for why -- the U-Net hit the same defect at 8x (intended 1e-3, actual 8.27e-3)."""
        self._p3 = (int(start_epoch), float(peak_lr))

    def on_epoch_begin(self, epoch, logs=None):
        if self._p3 is not None and epoch >= self._p3[0]:
            start, peak = self._p3
            span = max(self.total_epochs - start, 1)
            frac = 1.0 - min((epoch - start) / span, 1.0)
            lr = max(peak * (frac ** self.power), self.min_lr)     # same power as phase 2
            if epoch == start:
                print(f"\n  [phase3-lr] decaying from {peak:.2e} over epochs "
                      f"{start+1}-{self.total_epochs} (power {self.power})")
        else:
            frac = 1.0 - (epoch / max(self.total_epochs, 1))
            lr = max(self.peak_lr * (frac ** self.power), self.min_lr)
        self.model.optimizer.learning_rate.assign(lr)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"\n  [poly] epoch {epoch+1}/{self.total_epochs}: lr={lr:.2e}")
