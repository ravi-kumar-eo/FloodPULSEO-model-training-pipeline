"""Shared training core for M0. Both backbones import this, so the protocol is provably
identical and only the ARCHITECTURE and its LR SCHEDULE differ.

Everything here that came from train_v2.py was copied verbatim (losses.py, callbacks.py,
the metric classes). The from-scratch rewrite that preceded this introduced six silent
defects — a permanent-water target of 0 instead of 1 (inverting the gradient on ~87% of
positive cells), focal alpha 0.75 instead of 0.85, a soft boundary edge, a fixed-0.5 IoU in
place of the 200-threshold sweep, the flood label channel instead of union, and no warmup,
gradient clipping or EMA at all. None of them were visible in the loss curve.

WHAT IS DELIBERATELY ABSENT: window/CrossPatchMixer, sibling chains, swap loss, and the
weather branch. M0 is static-only. The weather path will be ported verbatim when RQ2 starts.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path

# MUST precede `import tensorflow`: without it the ViT cannot compile on Bolzano's Blackwell
# GPU (XLA Triton autotune fails on the ModalityEncoder's tall Fourier matrix).
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

from flood.paths import SRC, ROOT, METADATA
sys.path.insert(0, str(SRC))

from flood.data.dataset import FloodPatches                                    # noqa: E402
from flood.losses import segmentation_loss                                     # noqa: E402
from flood.metrics import PWMaskedAUC, IoUAtBestThreshold, BestThresholdLogger  # noqa: E402
from flood.callbacks import (WarmupPolyDecay, PolyDecay, OneBasedCSVLogger,    # noqa: E402
                       PostWarmupBestCheckpoint, StepStatusCallback)
from flood.models.threshold_head import soft_threshold_loss, ThresholdEpochTracker  # noqa: E402


def freeze_code(run_dir: Path) -> dict:
    """Copy the exact source about to run, with md5s, so a result is always reproducible."""
    dst = run_dir / 'code'
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {}
    # Paths are relative to src/. A missing entry used to `continue`, which wrote an EMPTY
    # code_manifest.json and reported success — a silent loss of provenance. It now raises:
    # a run with no recorded source is worse than a run that refused to start.
    for rel in ['flood/training/core.py', 'flood/losses.py', 'flood/metrics.py',
                'flood/callbacks.py', 'flood/data/dataset.py',
                'flood/models/threshold_head.py', 'flood/models/static/vit.py',
                'flood/models/static/unet.py']:
        src = SRC / rel
        if not src.exists():
            raise FileNotFoundError(
                f'freeze_code: {src} is missing — the source layout moved without updating '
                f'this list, and the run would have no provenance record.')
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        manifest[rel] = hashlib.md5(src.read_bytes()).hexdigest()
    (run_dir / 'code_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


class FloodModel(keras.Model):
    """Backbone + the tested composite loss + the threshold objective.

    THE THRESHOLD HEAD IS THE ONLY ADDITION to the tested protocol. It is a standard
    component of the baseline in BOTH backbones, not an experimental arm.
    """

    def __init__(self, backbone, cfg, **kw):
        super().__init__(**kw)
        self.backbone = backbone
        self.use_thr = bool(cfg.get('threshold_head', True))
        self.lambda_thr = float(cfg.get('lambda_thr', 0.2))
        self.thr_sharpness = float(cfg.get('thr_sharpness', 20.0))
        # Per-BACKBONE default. The U-Net has no LR warmup (nnU-Net poly schedule), so a
        # ViT-derived '5' is meaningless for it; only the ViT must wait for its LR warmup.
        _bb = str(cfg.get('backbone', 'vit')).lower()
        self.thr_warmup_epochs = int(cfg.get('thr_warmup_epochs',
                                             5 if _bb == 'vit' else 2))
        self.thr_ramp_epochs = max(int(cfg.get('thr_ramp_epochs', 2)), 1)
        self.dry_weight = float(cfg.get('thr_dry_weight', 1.0))
        # float32 so the variable lives on the GPU: an int32 variable is placed on the CPU
        # and reading it inside the compiled train_step raises a device error.
        self._thr_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32,
                                      name='thr_epoch')

        THRESHOLD = 0.35                      # train_v2.py's fixed-threshold convention
        self._thr_fixed = THRESHOLD
        self.m_loss = keras.metrics.Mean(name='loss')
        self.m_seg = keras.metrics.Mean(name='seg_loss')
        self.m_thr = keras.metrics.Mean(name='thr_loss')
        self.m_tau = keras.metrics.Mean(name='tau')
        # TWO populations, both logged (Ravi, 2026-08-29). The PRIMARY metric must score
        # what the LOSS optimises -- union with permanent water at PW_W -- otherwise
        # checkpoint selection is driven by a different population than training.
        # Flood-only is kept alongside because it is the number that means something
        # operationally, but it is NOT used for selection.
        from flood.losses import PW_W
        self.m_auprc = PWMaskedAUC(curve='PR', name='auprc', num_thresholds=200,
                                   population='union', pw_weight=PW_W)
        self.m_ioubest = IoUAtBestThreshold(name='iou_best', num_thresholds=200,
                                            population='union', pw_weight=PW_W)
        self.m_auprc_f = PWMaskedAUC(curve='PR', name='auprc_flood', num_thresholds=200,
                                     population='flood')
        self.m_ioubest_f = IoUAtBestThreshold(name='iou_best_flood', num_thresholds=200,
                                              population='flood')
        self.m_acc = keras.metrics.BinaryAccuracy(name='accuracy', threshold=THRESHOLD)
        self.m_prec = keras.metrics.Precision(name='precision', thresholds=THRESHOLD)
        self.m_rec = keras.metrics.Recall(name='recall', thresholds=THRESHOLD)
        self.m_iou = keras.metrics.Mean(name='iou')

    @property
    def metrics(self):
        return [self.m_loss, self.m_seg, self.m_thr, self.m_tau, self.m_auprc,
                self.m_ioubest, self.m_auprc_f, self.m_ioubest_f,
                self.m_iou, self.m_acc, self.m_prec, self.m_rec]

    def call(self, x, training=False):
        return self.backbone(x, training=training)

    def _compute(self, x, y, training):
        out = self.backbone(x, training=training)
        prob, tau = out if self.use_thr else (out, None)
        seg = segmentation_loss(y, prob)
        thr = tf.constant(0.0)
        total = seg
        if self.use_thr:
            # DECOUPLED 2026-08-29 (Ravi): stop_gradient on `prob` means the threshold loss
            # can reach ONLY the threshold head's own weights -- it is mathematically unable
            # to touch the backbone. Consequences:
            #   * the segmentation objective is pure; no auxiliary term perturbs it
            #   * lambda_thr is no longer a hedge against contamination -> 1.0
            #   * no warmup gate / ramp is needed, so the head trains from epoch 0 against a
            #     backbone it cannot chase, removing the moving-target problem that made its
            #     benefit inconsistent (+26% on one checkpoint, -9% on another)
            # The head still SEES the backbone output; it just cannot push back into it.
            thr = soft_threshold_loss(y, tf.stop_gradient(prob), tau,
                                      sharpness=self.thr_sharpness,
                                      dry_weight=self.dry_weight)
            total = seg + self.lambda_thr * thr
        return prob, tau, total, thr, seg

    def _update(self, y, prob, tau, total, thr, seg):
        self.m_loss.update_state(total)
        self.m_seg.update_state(seg)
        self.m_thr.update_state(thr)
        if tau is not None:
            self.m_tau.update_state(tf.reduce_mean(tau))
        # PWMaskedAUC / IoUAtBestThreshold honour the -1 marker themselves.
        self.m_auprc.update_state(y, prob)
        self.m_ioubest.update_state(y, prob)
        self.m_auprc_f.update_state(y, prob)
        self.m_ioubest_f.update_state(y, prob)
        mask = tf.cast(y >= 0.0, tf.float32)
        tgt = tf.maximum(y, 0.0)
        self.m_acc.update_state(tgt, prob, sample_weight=mask)
        self.m_prec.update_state(tgt, prob, sample_weight=mask)
        self.m_rec.update_state(tgt, prob, sample_weight=mask)
        pred = tf.cast(prob >= self._thr_fixed, tf.float32) * mask
        tgt_m = tgt * mask
        inter = tf.reduce_sum(pred * tgt_m)
        union = tf.reduce_sum(pred) + tf.reduce_sum(tgt_m) - inter
        self.m_iou.update_state(inter / (union + 1e-7))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            prob, tau, total, thr, seg = self._compute(x, y, True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(total, self.trainable_variables), self.trainable_variables))
        self._update(y, prob, tau, total, thr, seg)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        prob, tau, total, thr, seg = self._compute(x, y, False)
        self._update(y, prob, tau, total, thr, seg)
        return {m.name: m.result() for m in self.metrics}


class EpochStamp(keras.callbacks.Callback):
    """Record the last COMPLETED epoch so --resume knows where to restart."""

    def __init__(self, run_dir: Path):
        super().__init__()
        self.path = Path(run_dir) / 'last_epoch.json'

    def on_epoch_end(self, epoch, logs=None):
        self.path.write_text(json.dumps({
            # 0-BASED ON PURPOSE: `next_epoch` feeds Keras `initial_epoch`, which is an
            # index, not a human-facing epoch number. training_log.csv is 1-based.
            'last_completed_epoch': int(epoch), 'next_epoch': int(epoch) + 1,
            'logs': {k: float(v) for k, v in (logs or {}).items()}}, indent=2))


STAGE_OF = [('M0', ('unet_static', 'vit_static')),
            ('RQ2', ('unet_daily', 'vit_daily')),
            ('RQ3', ('unet_summary', 'vit_summary'))]


def resolve_patches_dir(cfg):
    """The one path that genuinely differs between hosts."""
    cand = [os.environ.get('FLOOD_PATCHES_DIR'), cfg.get('patches_dir'),
            '/home/ravi/shared_data/global_flood_dataset/pipeline/data/patches/patches_tif',
            '/media/ravi/Jupyter/flood_vit/pipeline/data/patches/patches_tif']
    found = next((c for c in cand if c and Path(c).is_dir()), None)
    if found is None:
        raise FileNotFoundError(f"no patches dir found; tried {[c for c in cand if c]}")
    return found


# ------------------------------------------------------------------ which split, and how
# its evaluation populations are named. v0 had four evaluation COHORTS; v1 has one val and
# one test, with basin familiarity as a LABEL (`basin_regime`) on the test rows. A SLICE is
# what an evaluation actually scores: a name, the cohort it filters on, and an optional
# regime. The v1 names deliberately match v0's where they mean the same thing, so every
# output filename, plot and downstream reader keeps working across both splits.
SLICES = {
    'v0': {'train':        ('train', None),
           'val_seen':     ('val_seen', None),
           'val_unseen':   ('val_unseen', None),
           'test_seen':    ('test_seen', None),
           'test_unseen':  ('test_unseen', None)},
    'v1': {'train':        ('train', None),
           'val':          ('val', None),
           'test':         ('test', None),          # HEADLINE test — keeps `partial` in
           'test_seen':    ('test', 'seen'),
           'test_unseen':  ('test', 'unseen'),
           'test_partial': ('test', 'partial')},
}
# Selection happens on the basin-UNSEEN validation population: the harder, more honest
# deployment regime. v1 has exactly one val and it is unseen by construction.
THRESHOLD_SLICE = {'v0': 'val_unseen', 'v1': 'val'}
DEFAULT_VAL_SLICES = {'v0': ['val_seen', 'val_unseen'], 'v1': ['val']}
DEFAULT_TEST_SLICES = {'v0': ['test_seen', 'test_unseen'],
                       'v1': ['test', 'test_seen', 'test_unseen']}


# The split directories were reorganised 2026-09-05: the current split became `splits`, and the
# two superseded ones moved under `split_old/`. Two consequences are handled here.
#   * A config that names `splits_v1` (the unet_static_v1_s1 run does) must still find it. Sealed
#     run configs are never edited, so the alias lives in the code.
#   * `splits` USED to mean v0 and now means v2. Every config written before 2026-09-05 has no
#     `split_dir` at all, so an absent key must resolve to the ARCHIVED v0 — never to whatever is
#     current. Silently scoring a v0 run against v2 would be a wrong number, not an error.
# `splits` is THE split. Everything superseded lives under split_old/ and is reachable by the name
# its runs recorded, so no sealed config ever needs editing. NOTE: development runs from before
# 2026-09-06 recorded `splits` when that name meant a different build — evaluate_fast verifies the
# split md5 against run_manifest.json and refuses rather than scoring against the wrong data.
SPLIT_ALIASES = {'splits_v3': 'splits',
                 'splits_v1': 'split_old/splits_v1',
                 'splits_v0': 'split_old/splits_v0',
                 'splits_bal': 'split_old/splits_bal_seed230',
                 'splits_swap': 'split_old/splits_swap_seed72'}
DEFAULT_SPLIT = 'split_old/splits_v0'


def resolve_split(cfg, md):
    """Locate the split a config names and work out which generation it is.

    A run's frozen `config.yaml` carries `split_dir`, so an evaluation always scores the split the
    model was actually trained on rather than whatever is current. New configs must set it
    explicitly; see DEFAULT_SPLIT above for why an absent key is NOT the current split.
    """
    name = str(cfg.get('split_dir', DEFAULT_SPLIT))
    if name in SPLIT_ALIASES and not (Path(md) / name).is_dir():
        name = SPLIT_ALIASES[name]
    d = Path(md) / name
    style = 'v0'
    mf = d / 'manifest.json'
    if mf.exists():
        # split_v1 and split_v2 share the cohort naming (train/val/test + basin_regime label);
        # only v0 has the four separate evaluation cohorts.
        if str(json.loads(mf.read_text()).get('version', '')).startswith('split_v'):
            style = 'v1'
    elif (d / 'patches.csv').exists():         # no manifest: fall back to the column itself
        import pandas as pd
        if 'basin_regime' in pd.read_csv(d / 'patches.csv', nrows=0).columns:
            style = 'v1'
    return d, style


def slice_spec(name, style):
    """A slice name -> (cohort, regime, is_test) for this split generation."""
    try:
        cohort, regime = SLICES[style][name]
    except KeyError:
        raise SystemExit(f"slice {name!r} does not exist in split {style}; "
                         f"have {sorted(SLICES[style])}") from None
    return cohort, regime, cohort.startswith('test')



def _set_lr(opt, lr):
    """Keras 3 keeps the LR on the optimizer's `learning_rate` variable."""
    try:
        opt.learning_rate.assign(float(lr))
    except Exception:
        opt.learning_rate = float(lr)


def _build_compiled(build_backbone, cfg):
    """A bare, built FloodModel for `cfg`.

    The weather sublayers are created in __init__ but only BUILD when the model is CALLED with a
    weather tensor. Calling with three inputs leaves them variable-less, which silently makes a
    weather model indistinguishable from a T0 one.
    """
    m = FloodModel(build_backbone(cfg), cfg)
    x = [tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2))]
    T = int(cfg.get('temporal_days', 0))
    if T:
        x.append(tf.zeros((1, T, 3)))
    m(tuple(x), training=False)
    return m


def run(build_backbone, build_optimizer, build_lr_callback, argv=None,
        experiment=None, backbone=None):
    """One training run. The three callables are what a backbone customises.

    build_backbone(cfg)      -> keras model returning (prob_map, tau)
    build_optimizer(cfg)     -> keras optimizer  (ViT: AdamW; U-Net: SGD+Nesterov)
    build_lr_callback(cfg)   -> LR schedule      (ViT: WarmupPolyDecay; U-Net: PolyDecay)

    `experiment` / `backbone` are declared by the ENTRY POINT, never by the config. When given,
    the config is checked against the experiment's definition and the run directory becomes
    runs/<experiment>/seed<N>/<backbone>/ — so an Exp 3 fit cannot be reconfigured into an Exp 2
    fit, and cannot write into Exp 2's folder. Omitting them keeps the old run-name behaviour
    for legacy runs.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--runs-dir', default=str(SRC.parent / 'runs'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', type=int, default=0)
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    common = SRC / 'configs' / '_common.yaml'
    if common.exists():
        base = yaml.safe_load(open(common)) or {}
        base.update(cfg)
        cfg = base
    cfg['patches_dir'] = resolve_patches_dir(cfg)
    print(f"  patches_dir: {cfg['patches_dir']}")

    seed = int(cfg.get('seed', 1))
    tf.random.set_seed(seed)
    np.random.seed(seed)

    if experiment is not None:
        from flood.training.guard import check
        check(experiment, backbone, cfg)          # refuses a config that is a DIFFERENT experiment
        a.run_name = a.run_name or f'{experiment}_{backbone}_s{seed}'
        # `stage` is still written into run_manifest.json. It MUST be assigned on both branches:
        # leaving it unset here raised UnboundLocalError at the very END of a completed fit
        # (ViT seed 2, 2026-09-08) — 41 epochs trained and checkpointed, then no manifest.
        stage = experiment
        run_dir = Path(a.runs_dir) / experiment / f"seed{seed}" / backbone
    else:
        if not a.run_name:
            ap.error('--run-name is required when the entry point declares no experiment')
        stage = next((s for s, pref in STAGE_OF
                      if any(a.run_name.startswith(p_) for p_ in pref)), 'other')
        run_dir = Path(a.runs_dir) / stage / f"seed{seed}" / a.run_name
    if not a.smoke:
        if (run_dir / 'run_manifest.json').exists() and not a.resume:
            print(f"REFUSED: {run_dir} holds a COMPLETED run. Use another --run-name.")
            return 2
        if (run_dir / 'last_epoch.json').exists() and not a.resume:
            print(f"REFUSED: {run_dir} holds an INTERRUPTED run. Pass --resume.")
            return 2
        run_dir.mkdir(parents=True, exist_ok=True)
        freeze_code(run_dir)
        yaml.safe_dump(cfg, open(run_dir / 'config.yaml', 'w'), sort_keys=False)

    # FLOOD_METADATA_DIR lets a container point at metadata outside the code tree
    # (AWS mounts code and data separately); defaults to the in-repo layout.
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or (SRC.parent / 'metadata'))
    # WHICH SPLIT (Ravi, 2026-09-04). `split_dir` names a directory under metadata/ and
    # `val_cohort` the cohort selection validates on, because v1 renamed the cohorts:
    # v0 had train / val_seen / val_unseen / test_seen / test_unseen, v1 has train / val /
    # test with basin familiarity as a LABEL (`basin_regime`) on the test rows. Defaults
    # reproduce v0 exactly, so every existing config and every sealed run is untouched.
    split_dir, _style = resolve_split(cfg, md)
    val_cohort = str(cfg.get('val_cohort', 'val_unseen'))
    if not (split_dir / 'patches.csv').exists():
        raise FileNotFoundError(f"split_dir {split_dir} has no patches.csv")
    print(f"  split_dir:   {split_dir}  (validating on '{val_cohort}')")
    # EXP 1 is terrain-only and therefore trains and validates on D_mono (`mono_temporal: true`).
    # EXP 2 onward leaves it false and uses every acquisition, because once weather is an input the
    # dates are no longer identical and the repeat acquisitions become real additional samples.
    mono = bool(cfg.get('mono_temporal', False))
    if mono:
        print("  rows:        D_mono — one acquisition per site (exp1_keep)")
    kw = dict(patches_csv=split_dir / 'patches.csv',
              patches_dir=cfg['patches_dir'],
              labels_dir=md / 'labels_80m_global',
              flood_flags_csv=split_dir / 'flood80_flags.csv',
              mono_only=mono,
              # WITHOUT THIS the loader defaults to temporal_days=0 and yields THREE inputs, so
              # `PlainViT.call` sets weather=None, the entire weather path is skipped, every
              # weather gradient is None and fit() dies with "No gradients provided for any
              # variable". A weather experiment would otherwise train with no weather at all.
              # 0 for Exp 1, so its behaviour is unchanged. Measured 2026-09-10.
              temporal_days=int(cfg.get('temporal_days', 0)))
    workers = int(cfg.get('workers', 12))
    _train_fp = FloodPatches(cohort='train', **kw)
    # FLOOD_RAM_CACHE=1 decodes the flood pool once into RAM (49 GB) -- the half the
    # sampler re-reads in FULL every epoch. BIT-IDENTICAL to the TIFF path (verified:
    # 40 flood rows cache-hit, 20 dry rows fall through, np.array_equal on all arrays),
    # so it is an I/O change only and seeds stay comparable. Set via ENV, not config,
    # so the six M0 configs still differ by exactly one line (the seed).
    # Filled BEFORE any GPU op: forking after CUDA init is unsafe.
    ram_cache = os.environ.get('FLOOD_RAM_CACHE') == '1'
    if ram_cache:
        _train_fp.enable_ram_cache(
            workers=int(os.environ.get('FLOOD_RAM_CACHE_WORKERS', '32')))
    train_ds = _train_fp.to_tf_dataset(
        batch_size=int(cfg['batch_size']), flood_ratio=float(cfg.get('flood_ratio', 0.5)),
        seed=seed, workers=workers)
    # selection happens on the basin-UNSEEN validation cohort: the harder, more honest
    # deployment regime. v0 spells it `val_unseen`; v1 has one val and it is unseen by
    # construction, so there it is just `val`.
    val_ds = FloodPatches(cohort=val_cohort, **kw).to_tf_dataset(
        batch_size=int(cfg['batch_size']), shuffle=False, flood_ratio=0.0,
        seed=seed, workers=workers)

    model = FloodModel(build_backbone(cfg), cfg)
    model.compile(optimizer=build_optimizer(cfg))

    # Build the WRAPPER before fit(): Keras builds the backbone from train_step but leaves
    # FloodModel "not built", and ModelCheckpoint then raises on the FIRST epoch end.
    # The weather sublayers are created in __init__ but only BUILD when the model is CALLED
    # with a weather tensor, so a THREE-input dummy leaves them variable-less -- exactly what
    # _build_compiled's docstring warns about. That is not cosmetic: the warm start then reports
    # "0 weather variables left at init", `set_trunk_trainable` frees 4 submodules that own no
    # variables, and `trainable_variables` comes back EMPTY, so fit() dies on the first step with
    # `apply_gradients` -> "not enough values to unpack (expected 2, got 0)". Measured 2026-09-10.
    _dummy = [tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2))]
    _T_dummy = int(cfg.get('temporal_days', 0))
    if _T_dummy:
        _dummy.append(tf.zeros((1, _T_dummy, 3)))
    dummy = tuple(_dummy)
    model(dummy, training=False)

    initial_epoch = 0
    if a.resume and not a.smoke:
        stamp = run_dir / 'last_epoch.json'
        _ck = run_dir / 'checkpoints'
        cand = [p for p in (_ck / 'best_loss.weights.h5',
                            _ck / 'best_iou.weights.h5') if p.exists()]
        last = max(cand, key=lambda p: p.stat().st_mtime) if cand else _ck / '_none'
        if not stamp.exists() or not last.exists():
            print("REFUSED --resume: need last_epoch.json and a best_*.weights.h5")
            return 2
        initial_epoch = int(json.loads(stamp.read_text())['next_epoch'])
        # AdamW/SGD slots are created lazily; one dummy step forces them into existence so
        # the optimizer state actually restores instead of silently starting cold.
        with tf.GradientTape() as tp:
            _, _, tot, _, _ = model._compute(dummy, tf.zeros((1, 32, 32, 1)), True)
        model.optimizer.apply_gradients(
            zip(tp.gradient(tot, model.trainable_variables), model.trainable_variables))
        before = float(tf.reduce_mean(model(dummy, training=False)[0]))
        model.load_weights(str(last))
        after = float(tf.reduce_mean(model(dummy, training=False)[0]))
        print(f"  RESUMING at epoch {initial_epoch} ({len(model.optimizer.variables)} "
              f"optimizer slots, output {before:.6f} -> {after:.6f})")

    if a.smoke:
        print(f"\n=== SMOKE {a.smoke} steps ===")
        h = model.fit(train_ds, steps_per_epoch=a.smoke, epochs=1, verbose=2)
        print("  losses:", {k: round(float(v[0]), 5) for k, v in h.history.items()})
        return 0

    total_epochs = int(cfg.get('epochs', 100))
    # Checkpoint gating follows the backbone: the U-Net has no LR warmup to wait out,
    # so gating its checkpoints on a ViT warmup would just discard good early epochs.
    warmup = int(cfg.get('warmup_epochs', 5)) if str(cfg.get('backbone','vit')).lower() == 'vit' else 0
    ckpt = run_dir / 'checkpoints'
    ckpt.mkdir(parents=True, exist_ok=True)
    # EMA REMOVED 2026-08-29: WeightEMA accumulated shadow weights but apply_to_model() was
    # never called and they were never saved, so it cost memory and affected NO reported
    # number. Do not claim a protocol component that does not run.

    lr_cb = build_lr_callback(cfg)          # kept by name: UnfreezeAt needs it (phase-3 LR)
    cbs = [lr_cb]
    if cfg.get('threshold_head', True):
        cbs.append(ThresholdEpochTracker())
    # EARLY STOPPING MUST IGNORE PHASE 2. The interaction gate is zero-initialised, so during the
    # frozen phase almost nothing can move and val_seg_loss is EXPECTED not to improve -- it is a
    # known non-improving stretch, not evidence of convergence. Counting it against `patience`
    # spends the entire window before phase 3 even begins.
    #
    # Measured on exp2 seed 1 (2026-09-11): neither backbone beat its epoch-0 val_seg_loss, so
    # `wait` would have reached 15 at exactly epoch 15 -- Keras increments `wait` BEFORE the
    # improvement test and fires at `wait >= patience` (early_stopping.py:110/123). The unfreeze
    # runs at the START of epoch 15 and the stop fires at its END, so the run would have sealed
    # after ONE phase-3 epoch, having spent ~6 h on a phase whose only purpose was to prepare
    # for phase 3.
    #
    # start_from_epoch makes the patience window measure phase 3 against phase 3. Exp 1 sets no
    # `warm_start`, so it resolves to 0 and its behaviour is unchanged.
    es_start = int(cfg.get('freeze_epochs', 10)) if (
        cfg.get('warm_start') and experiment is not None and experiment != 'exp1') else 0
    cbs += [
        PostWarmupBestCheckpoint(filepath=ckpt / 'best_loss.weights.h5',
                                 warmup_epochs=warmup, monitor='val_seg_loss', mode='min'),
        PostWarmupBestCheckpoint(filepath=ckpt / 'best_iou.weights.h5',
                                 warmup_epochs=warmup, monitor='val_iou_best', mode='max'),
        # NO periodic/last checkpoint (Ravi, 2026-08-29): only the two SELECTION weights are
        # kept -- best val_seg_loss and best val_iou_best -- because both are reported. A
        # resume restarts from whichever of the two is newest.
        EpochStamp(run_dir),
        # train_status.json every 50 steps -> the dashboard's live within-epoch view.
        # CSVLogger only writes once per epoch, so without this the dashboard sits on
        # "waiting for first epoch" for the whole of a ~1 h epoch.
        StepStatusCallback(run_dir / 'train_status.json', total_epochs, every=50),
        BestThresholdLogger(model.m_ioubest),          # MUST precede CSVLogger
        OneBasedCSVLogger(run_dir / 'training_log.csv', append=bool(initial_epoch)),
        keras.callbacks.EarlyStopping(monitor='val_seg_loss', mode='min',
                                      patience=int(cfg.get('patience', 15)), min_delta=0.0,
                                      start_from_epoch=es_start,
                                      restore_best_weights=False),
    ]
    if es_start:
        print(f'  [phases] early stopping ignores the frozen phase; it starts counting at '
              f'epoch {es_start + 1} (patience {int(cfg.get("patience", 15))})')

    t0 = time.time()
    # ---------------------------------------------------------------- PAIRED WARM START
    # Exp 2/3/4 start from THIS SEED's own Exp 1 checkpoint, so the +-0.007 M0 seed spread
    # cancels in `this run - its own T0`. The path is DERIVED from experiment/seed/backbone and
    # is never a free-text flag: pointing it at another seed would silently reintroduce exactly
    # the variance the pairing exists to remove.
    if cfg.get('warm_start') and experiment is not None and experiment != 'exp1':
        from flood.training import phases
        # NOT `t0`: that name already holds the run's start time (set above and read by the
        # manifest's `hours` at the end of run()). Binding a Path to it made every warm-started
        # fit train to completion and then die on its LAST statement with
        # `TypeError: unsupported operand type(s) for -: 'float' and 'PosixPath'` -- the same
        # shape as the `stage` UnboundLocalError. Exp 1 never entered this branch, so nothing
        # caught it. Keep the two names distinct.
        t0_ckpt = Path(a.runs_dir) / 'exp1' / f'seed{seed}' / backbone / 'checkpoints' / 'best_loss.weights.h5'
        twin_cfg = dict(cfg); twin_cfg['temporal_days'] = 0
        phases.load_static_checkpoint(
            model, lambda: _build_compiled(build_backbone, twin_cfg), t0_ckpt)
        phases.set_trunk_trainable(model.backbone, False)      # phase 2
        model.compile(optimizer=build_optimizer(cfg), jit_compile=False)
        fe = int(cfg.get('freeze_epochs', 10))

        def _phase3_recompile(lr):
            """Hand phase 3 a FRESH optimizer, with its slots already built.

            Reusing `model.optimizer` here is what the code used to do, and it cannot work:
            Keras builds an optimizer against a fixed variable list (`build()` fills
            `_trainable_variables_indices`), and `apply()` then runs `_check_variables_are_known`
            against it. The phase-2 optimizer was built on the weather path alone, so the first
            phase-3 step -- where the trunk is suddenly trainable -- raises
            `ValueError: Unknown variable: <...>. This optimizer can only be called for the
            variables it was originally built with`.

            `build()` is called HERE, eagerly, and not left to the first `apply_gradients`:
            by then we are inside the retraced `train_function`, and creating optimizer slot
            variables during tracing is illegal.

            Cold moments are the point, not a cost -- phase 3 is a different optimisation
            problem at a different LR, exactly as UnfreezeAt's docstring says.
            """
            opt = build_optimizer(cfg)
            _set_lr(opt, lr)
            model.compile(optimizer=opt, jit_compile=False)
            opt.build(model.trainable_variables)     # train_step uses self.trainable_variables

        cbs.append(phases.UnfreezeAt(
            model.backbone, fe, float(cfg.get('fine_tune_lr', 1e-4)), _phase3_recompile,
            lr_cb=lr_cb))
        print(f'  [phases] warm start done; trunk frozen for epochs 1-{fe}; phase 3 starts at epoch {fe + 1}')

    model.fit(train_ds, validation_data=val_ds,
              steps_per_epoch=getattr(train_ds, 'steps_per_epoch', None),
              epochs=total_epochs, initial_epoch=initial_epoch, callbacks=cbs, verbose=1)

    (run_dir / 'run_manifest.json').write_text(json.dumps({
        'run_name': a.run_name, 'backbone': cfg['backbone'], 'seed': seed, 'stage': stage,
        'compute': os.environ.get('FLOOD_COMPUTE', 'bolzano'),
        'split_dir': str(cfg.get('split_dir', DEFAULT_SPLIT)),
        'split_resolved': str(split_dir),
        'mono_temporal': mono,
        'rows': 'D_mono (one acquisition per site)' if mono else 'D_multi (all acquisitions)',
        'val_cohort': val_cohort,
        'split_md5': hashlib.md5((split_dir / 'patches.csv').read_bytes()).hexdigest(),
        'hours': round((time.time() - t0) / 3600, 3),
        'params': int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        'optimizer': type(model.optimizer).__name__,
        'lr_schedule': type(cbs[0]).__name__,
        'ram_cache_flood': bool(ram_cache),
        'status': 'confirmatory'}, indent=2))
    return 0
