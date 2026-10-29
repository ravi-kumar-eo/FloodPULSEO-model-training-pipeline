"""Training core of the FINAL experiment programme (METHOD_PLAN_RERUN.md, version 3).

A separate module so the old core (`core.py`), its runs and its evaluators stay untouched.
Reused unchanged from the tested core: the compound segmentation loss, the threshold head and
its stop-gradient objective, the optimiser recipes and LR schedules, the warm start
(`phases.load_static_checkpoint`) and the identity-at-initialisation weather paths.

Changed (each item names its plan entry):
  * loader `FinalFloodPatches`: final split, 32/64 class-map labels, SoilGrids, versioned
    transforms, AOI weather (C2, C4)
  * stages are explicit: final_t0 (static, random init), final_t30 and final_window
    (weather, warm start from runs_final/final_t0/seed<S>/<family>) — no `exp1` name checks (C11)
  * the parent T0 must carry the SAME data version; its weights' md5 is recorded (C11)
  * per-epoch validation computes the segmentation loss, the 49-threshold pooled flood-only IoU
    (`val_flood_iou_best`) and cheap metrics; the old 200-threshold metrics are gone (C6)
  * early stopping on `stop_metric` (validation loss), checkpoint by `select_metric`
    (validation IoU at the best threshold) -- Ravi, 2026-10-06, option C
  * mixed precision via `mixed_precision: mixed_bfloat16` (C7)
  * validation rows decoded once into RAM when `eval_cache: true` (C5)
  * run manifest with data version, parent, precision, GPU and environment (G11)

Run dir: <runs_dir>/<stage>/seed<S>/<family>[_t<T> for final_window].
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')

import numpy as np
import tensorflow as tf
import yaml
from tensorflow import keras

from flood.paths import SRC
sys.path.insert(0, str(SRC))

from flood.losses import segmentation_loss                                   # noqa: E402
from flood.callbacks import OneBasedCSVLogger, PostWarmupBestCheckpoint, StepStatusCallback  # noqa: E402
from flood.models.threshold_head import soft_threshold_loss, ThresholdEpochTracker  # noqa: E402
from flood.training.core import resolve_patches_dir, _set_lr, EpochStamp      # noqa: E402
from flood.training import phases                                           # noqa: E402

SEL_CKPT = 'best_selected.weights.h5'     # the checkpoint chosen by `select_metric`

STAGES = {
    # stage: (allowed temporal_days, rows, warm start)
    'final_t0':     ((0,),         'single', False),
    'final_t30':    ((30,),        'full',   True),
    'final_window': ((3, 7, 15),   'full',   True),
    # Ravi 2026-10-07: input ablations at T30 (config `variant`: noup / nosum / sumonly), warm-started,
    # and the full T30 input trained from scratch (no parent)
    'final_ablation': ((30,),      'full',   True),
    'final_scratch':  ((30,),      'full',   False),
}
FROZEN_FILES = ['flood/training/core_final.py', 'flood/training/core.py', 'flood/training/phases.py',
                'flood/losses.py', 'flood/callbacks.py', 'flood/data/dataset.py',
                'flood/data/dataset_final.py', 'flood/models/threshold_head.py',
                'flood/models/static/vit.py', 'flood/models/static/unet.py',
                'flood/models/temporal/encoders.py', 'flood/models/temporal/interaction.py',
                'flood/models/temporal/utae.py', 'flood/models/temporal/aoi_weather.py',
                # 2026-10-08 (review of Ch4): also the entry point (optimisers, schedules), the
                # evaluator, the swap test, the path module and the builders of the upstream inputs
                'scripts/train.py', 'scripts/evaluate.py', 'tests/weather_check.py',
                'flood/paths.py', 'build/build_upstream_catchments.py', 'build/build_upstream_weather.py']


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def freeze_code(run_dir: Path) -> dict:
    dst = run_dir / 'code'
    man = {}
    for rel in FROZEN_FILES:
        src = SRC / rel
        if not src.exists():
            raise FileNotFoundError(f'freeze_code: {src} missing')
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        man[rel] = md5(src)
    (run_dir / 'code_manifest.json').write_text(json.dumps(man, indent=2))
    return man


GRID49 = np.round(np.linspace(0.02, 0.98, 49), 4)


class FloodIoUBest(keras.metrics.Metric):
    """Pooled flood-only IoU at the best of the 49 thresholds 0.02, 0.04, ..., 0.98.

    The SELECTION metric of the final programme (Ravi, 2026-10-05): the same quantity with which
    scripts/evaluate.py chooses each fit's global threshold (pick_threshold), accumulated over
    the whole validation pass. TP/FP/FN are summed over every flood-only cell (label >= 0;
    permanent-water and mixed cells, label -1, excluded) for each threshold, and the result is
    max_t TP/(TP+FP+FN). Counts are float64, exact up to 2^53 cells. Re-choosing the threshold every
    epoch makes the score immune to the calibration drift that moves a fixed-threshold IoU.
    """

    def __init__(self, name='flood_iou_best', **kw):
        super().__init__(name=name, **kw)
        self.thr = tf.constant(GRID49, tf.float32)
        self.tp = self.add_variable(shape=(49,), initializer='zeros', dtype='float64', name='tp')
        self.fp = self.add_variable(shape=(49,), initializer='zeros', dtype='float64', name='fp')
        self.fn = self.add_variable(shape=(49,), initializer='zeros', dtype='float64', name='fn')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        p = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        keep = y >= 0.0
        p = tf.boolean_mask(p, keep)
        t = tf.cast(tf.boolean_mask(y, keep) > 0.5, tf.float64)[:, None]        # (N,1)
        pred = tf.cast(p[:, None] >= self.thr[None, :], tf.float64)               # (N,49)
        self.tp.assign_add(tf.reduce_sum(pred * t, axis=0))
        self.fp.assign_add(tf.reduce_sum(pred * (1.0 - t), axis=0))
        self.fn.assign_add(tf.reduce_sum((1.0 - pred) * t, axis=0))

    def result(self):
        iou = tf.math.divide_no_nan(self.tp, self.tp + self.fp + self.fn)
        return tf.cast(tf.reduce_max(iou), tf.float32)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))


class FloodModelFinal(keras.Model):
    """Backbone + compound segmentation loss + gradient-isolated threshold objective.

    Metrics are the cheap ones only: the 200-threshold AUPRC/IoU sweeps of the old core cost a
    large share of every validation pass and selected nothing that is reported.
    """

    def __init__(self, backbone, cfg, **kw):
        super().__init__(**kw)
        self.backbone = backbone
        self.use_thr = bool(cfg.get('threshold_head', True))
        self.lambda_thr = float(cfg.get('lambda_thr', 1.0))   # _common.yaml: 1.0
        self.thr_sharpness = float(cfg.get('thr_sharpness', 20.0))
        self.dry_weight = float(cfg.get('thr_dry_weight', 1.0))
        self._thr_epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='thr_epoch')
        # Two learning rates (2026-10-05, approved by Ravi): the weights copied from T0 train with
        # `self.optimizer` at fine_tune_lr; the NEW weather parts, which start from zero, train
        # with `opt_new` at `lr_mult` times that rate (new_parts_lr / fine_tune_lr), following the
        # same schedule. None = one optimizer for everything (T0 fits).
        self.opt_new = None
        self._new_ids = frozenset()
        self.lr_mult = 1.0
        self.m_loss = keras.metrics.Mean(name='loss')
        self.m_seg = keras.metrics.Mean(name='seg_loss')
        self.m_thr = keras.metrics.Mean(name='thr_loss')
        self.m_tau = keras.metrics.Mean(name='tau')
        self.m_prec = keras.metrics.Precision(name='precision', thresholds=0.35)
        self.m_rec = keras.metrics.Recall(name='recall', thresholds=0.35)
        self.m_acc = keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.35)
        self.m_iou = keras.metrics.Mean(name='iou')             # batch IoU at 0.35, flood-only
        self.m_iou_best = FloodIoUBest()                         # SELECTION metric, validation only

    @property
    def metrics(self):
        return [self.m_loss, self.m_seg, self.m_thr, self.m_tau, self.m_iou, self.m_acc,
                self.m_prec, self.m_rec, self.m_iou_best]

    def call(self, x, training=False):
        return self.backbone(x, training=training)

    def _compute(self, x, y, training):
        out = self.backbone(x, training=training)
        prob, tau = out if self.use_thr else (out, None)
        prob = tf.cast(prob, tf.float32)
        seg = segmentation_loss(y, prob)
        thr = tf.constant(0.0)
        total = seg
        if self.use_thr:
            tau = tf.cast(tau, tf.float32)
            thr = soft_threshold_loss(y, tf.stop_gradient(prob), tau, sharpness=self.thr_sharpness,
                                      dry_weight=self.dry_weight)
            total = seg + self.lambda_thr * thr
        return prob, tau, total, thr, seg

    def _update(self, y, prob, tau, total, thr, seg):
        self.m_loss.update_state(total); self.m_seg.update_state(seg); self.m_thr.update_state(thr)
        if tau is not None:
            self.m_tau.update_state(tf.reduce_mean(tau))
        mask = tf.cast(y >= 0.0, tf.float32)            # flood-only population (PW / mixed excluded)
        tgt = tf.maximum(y, 0.0)
        self.m_prec.update_state(tgt, prob, sample_weight=mask)
        self.m_rec.update_state(tgt, prob, sample_weight=mask)
        self.m_acc.update_state(tgt, prob, sample_weight=mask)
        pred = tf.cast(prob >= 0.35, tf.float32) * mask
        inter = tf.reduce_sum(pred * tgt * mask)
        self.m_iou.update_state(inter / (tf.reduce_sum(pred) + tf.reduce_sum(tgt * mask) - inter + 1e-7))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            prob, tau, total, thr, seg = self._compute(x, y, True)
        grads = tape.gradient(total, self.trainable_variables)
        if self.opt_new is None:
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        else:
            old, new = [], []
            for g, v in zip(grads, self.trainable_variables):
                (new if id(v) in self._new_ids else old).append((g, v))
            self.optimizer.apply_gradients(old)
            self.opt_new.learning_rate.assign(self.optimizer.learning_rate * self.lr_mult)
            self.opt_new.apply_gradients(new)
        self._update(y, prob, tau, total, thr, seg)
        return {m.name: m.result() for m in self.metrics if m is not self.m_iou_best}

    def test_step(self, data):
        x, y = data
        prob, tau, total, thr, seg = self._compute(x, y, False)
        self._update(y, prob, tau, total, thr, seg)
        self.m_iou_best.update_state(y, prob)
        return {m.name: m.result() for m in self.metrics}


def weather_shape(cfg):
    T = int(cfg.get('temporal_days', 0))
    if not T:
        return None
    if str(cfg.get('weather_mode', 'aoi')) == 'upstream':
        return (T, 5)                              # per-patch daily weather + upstream rain
    k = int(cfg.get('aoi_k', 11))
    return (T, k, k, 8)


def final_dataset_kwargs(cfg, dp, mono_only=False):
    """Keyword arguments of FinalFloodPatches for a config: one place for trainer, evaluator, checks."""
    from flood.data.dataset_final import load_transforms
    T = int(cfg.get('temporal_days', 0))
    mode = str(cfg.get('weather_mode', 'aoi'))
    return dict(patches_csv=dp['split'] / 'patches.csv', patches_dir=cfg['patches_dir'],
                labels_dir=dp['labels'], flood_flags_csv=dp['split'] / 'flood80_flags.csv',
                mono_only=mono_only, temporal_days=T, transforms=load_transforms(dp['transforms']),
                mixed_as=str(cfg.get('mixed_as', 'pw')), soil_file=str(cfg.get('soil_file', 'input_160m.tif')),
                aoi_weather_dir=dp['aoi_weather'] if (T and mode == 'aoi') else None,
                aoi_k=int(cfg.get('aoi_k', 11)), weather_mode=mode,
                upstream_dir=dp['upstream_weather'] if (T and mode == 'upstream') else None)


def build_model(build_backbone, cfg):
    """A built FloodModelFinal (weather sublayers are built by a call WITH a weather tensor)."""
    m = FloodModelFinal(build_backbone(cfg), cfg)
    x = [tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2))]
    ws = weather_shape(cfg)
    if ws:
        x.append(tf.zeros((1,) + ws))
    m(tuple(x), training=False)
    return m


def best_epochs(log_csv, sel_metric, sel_mode, warmup):
    """Epoch (1-based) and value of the selected checkpoint and of the lowest validation loss."""
    import csv
    rows = [r for r in csv.DictReader(open(log_csv)) if int(r['epoch']) > warmup]
    if not rows:
        return {}
    key = sel_metric                                     # the CSV column is the Keras log key
    pick = (max if sel_mode == 'max' else min)(rows, key=lambda r: float(r[key]))
    lo = min(rows, key=lambda r: float(r['val_seg_loss']))
    return {'selected_epoch': int(pick['epoch']), 'selected_value': float(pick[key]),
            'lowest_val_seg_loss_epoch': int(lo['epoch']), 'lowest_val_seg_loss': float(lo['val_seg_loss']),
            'epochs_run': len(rows) + warmup}


def configure_warm_optimizers(model, cfg, build_optimizer):
    """Compile a warm-started weather model with its learning rates.

    Copied T0 weights: `fine_tune_lr`. New weather parts (phases.weather_layers): `new_parts_lr`
    (default: the same rate, i.e. one optimizer). Both optimizers are built eagerly, because
    creating optimizer slots inside the traced train step is illegal.
    Returns (fine_tune_lr, new_parts_lr, n copied variables, n new variables, n new parameters).
    """
    ft = float(cfg['fine_tune_lr'])
    opt = build_optimizer(cfg); _set_lr(opt, ft)
    model.compile(optimizer=opt, jit_compile=False)
    new_vars = [v for layer in phases.weather_layers(model.backbone) for v in layer.trainable_variables]
    new_ids = {id(v) for v in new_vars}
    old_vars = [v for v in model.trainable_variables if id(v) not in new_ids]
    nlr = float(cfg.get('new_parts_lr', ft))
    model.opt_new, model._new_ids, model.lr_mult = None, frozenset(), 1.0
    if nlr != ft:
        model.opt_new = build_optimizer(cfg); _set_lr(model.opt_new, nlr)
        model._new_ids = frozenset(new_ids)
        model.lr_mult = nlr / ft
        model.optimizer.build(old_vars)
        model.opt_new.build(new_vars)
    return ft, nlr, len(old_vars), len(new_vars), int(sum(np.prod(v.shape) for v in new_vars))


def data_paths(cfg, md: Path):
    fin = md / str(cfg.get('final_dir', 'final'))
    return {'split': fin / 'split', 'labels': fin / 'labels_80m_cls',
            'aoi_weather': fin / 'aoi_weather', 'transforms': fin / 'input_transforms.json',
            'upstream_weather': fin / 'upstream' / 'weather',
            'version': fin / 'data_version.json'}


def environment():
    gpus = tf.config.list_physical_devices('GPU')
    name = ''
    try:
        name = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    return {'tensorflow': tf.__version__, 'keras': keras.__version__, 'python': platform.python_version(),
            'gpu': name, 'n_gpus': len(gpus), 'host': platform.node(),
            'compute': os.environ.get('FLOOD_COMPUTE', platform.node())}


def run(build_backbone, build_optimizer, build_lr_callback, stage, family, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--runs-dir', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--smoke', type=int, default=0, help='N training steps, no run directory')
    ap.add_argument('--smoke-parent', default=None, help='checkpoint for a smoke warm start (tests only)')
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    common = Path(a.config).parent / '_common.yaml'
    if common.exists():
        base = yaml.safe_load(open(common)) or {}
        base.update(cfg); cfg = base
    cfg['patches_dir'] = resolve_patches_dir(cfg)
    # Per-MACHINE input settings (2026-10-08). A box with less RAM or fewer cores than Bolzano (e.g.
    # the AWS g7e.2xlarge: 62 GB, 8 vCPU) sets these in its environment; unset = the config values.
    # They change only where rows are read from and how many threads read them: the sampled rows and
    # the model are identical (batch arrival order is non-deterministic on EVERY machine, the training
    # pipeline runs tf.data with deterministic=False), and config.yaml below records the values used.
    for env, key, cast in (('FLOOD_TRAIN_CACHE_GB', 'train_cache_gb', float), ('FLOOD_WORKERS', 'workers', int),
                           ('FLOOD_CACHE_WORKERS', 'cache_workers', int), ('FLOOD_EVAL_CACHE', 'eval_cache',
                                                                          lambda v: v.lower() in ('1', 'true', 'yes'))):
        if os.environ.get(env):
            cfg[key] = cast(os.environ[env])
    if stage not in STAGES:
        raise SystemExit(f'unknown stage {stage}')
    days_ok, rows, warm = STAGES[stage]
    T = int(cfg.get('temporal_days', 0))
    if T not in days_ok:
        raise SystemExit(f'{stage}: temporal_days={T} not in {days_ok}')
    if str(cfg.get('backbone')) != family:
        raise SystemExit(f'{stage}: config backbone {cfg.get("backbone")!r} != entry point {family!r}')
    if T and str(cfg.get('weather_mode')) not in ('aoi', 'upstream'):
        raise SystemExit('final weather fits need weather_mode: aoi or upstream')
    if family == 'unet' and T and str(cfg.get('temporal_mode')) != 'utae':
        raise SystemExit('final U-Net weather fits need temporal_mode: utae')

    # RULES (Ravi, 2026-10-06, option C). EARLY STOPPING on the validation segmentation loss,
    # patience from the config (15) counted on that loss. CHECKPOINT = the epoch with the best
    # validation pooled flood-only IoU at the best of 49 thresholds, over every epoch trained.
    sel_metric = str(cfg.get('select_metric', 'val_seg_loss'))
    sel_mode = str(cfg.get('select_mode', 'min'))
    stop_metric = str(cfg.get('stop_metric', 'val_seg_loss'))
    stop_mode = str(cfg.get('stop_mode', 'min'))
    prec = str(cfg.get('mixed_precision', 'float32'))
    if prec != 'float32':
        keras.mixed_precision.set_global_policy(prec)

    seed = int(cfg.get('seed', 1))
    tf.random.set_seed(seed); np.random.seed(seed)
    name = family + (f'_t{T}' if stage == 'final_window' else '') + (f"_{cfg['variant']}" if cfg.get('variant') else '')
    run_dir = Path(a.runs_dir) / stage / f'seed{seed}' / name

    from flood.paths import METADATA
    md = Path(os.environ.get('FLOOD_METADATA_DIR') or METADATA)
    dp = data_paths(cfg, md)
    for k, p in dp.items():
        if not p.exists():
            raise SystemExit(f'missing final data artefact {k}: {p}')
    need = ('split', 'labels') + (('upstream_weather',) if str(cfg.get('weather_mode')) == 'upstream' and T
                                  else ('aoi_weather',))
    for k in need:
        if not (dp[k] / 'COMPLETE').exists():
            raise SystemExit(f'{dp[k]} has no COMPLETE marker')
    version = json.loads(dp['version'].read_text())
    from flood.data.dataset_final import FinalFloodPatches, load_transforms
    transforms = load_transforms(dp['transforms'])

    if not a.smoke:
        if (run_dir / 'run_manifest.json').exists() and not a.resume:
            raise SystemExit(f'REFUSED: {run_dir} holds a completed run')
        if (run_dir / 'last_epoch.json').exists() and not a.resume:
            raise SystemExit(f'REFUSED: {run_dir} holds an interrupted run; pass --resume')
        run_dir.mkdir(parents=True, exist_ok=True)
        freeze_code(run_dir)
        yaml.safe_dump(cfg, open(run_dir / 'config.yaml', 'w'), sort_keys=False)

    kw = final_dataset_kwargs(cfg, dp, mono_only=(rows == 'single'))
    workers = int(cfg.get('workers', 12))
    train_fp = FinalFloodPatches(cohort='train', **kw)
    # TRAINING RAM CACHE (2026-10-05): decoded rows in shared memory, flood rows first, within
    # `train_cache_gb`; bit-identical to the disk path, so an I/O change only.
    if float(cfg.get('train_cache_gb', 0)) > 0 and not a.smoke:
        train_fp.preload_labels(verbose=False)
        train_fp.enable_eval_cache(workers=int(cfg.get('cache_workers', 24)), tag='traincache',
                                   rows=train_fp.train_cache_rows(float(cfg['train_cache_gb']), seed=seed))
    train_ds = train_fp.to_tf_dataset(batch_size=int(cfg['batch_size']),
                                      flood_ratio=float(cfg.get('flood_ratio', 0.5)), seed=seed, workers=workers)
    val_fp = FinalFloodPatches(cohort='val', **kw)
    if cfg.get('eval_cache', True) and not a.smoke:
        val_fp.enable_eval_cache(workers=int(cfg.get('cache_workers', 24)))
    val_ds = val_fp.to_tf_dataset(batch_size=int(cfg['batch_size']), shuffle=False, flood_ratio=0.0,
                                  seed=seed, workers=workers)

    model = build_model(build_backbone, cfg)
    model.compile(optimizer=build_optimizer(cfg), jit_compile=False)

    # ------------------------------------------------------------ warm start (weather stages)
    # No frozen phase (frozen 2026-09-25): every weight trains from epoch 1 at `fine_tune_lr`,
    # one tenth of the T0 rate, and the LR schedule decays from there (enter_phase3 at epoch 0).
    parent = None
    lr_cb = build_lr_callback(cfg)
    if warm:
        if a.smoke and a.smoke_parent:
            pck = Path(a.smoke_parent)
        else:
            pdir = Path(a.runs_dir) / 'final_t0' / f'seed{seed}' / family
            pman = pdir / 'run_manifest.json'
            if not (pdir / 'COMPLETE').exists() or not pman.exists():
                raise SystemExit(f'parent T0 not complete: {pdir}')
            pm = json.loads(pman.read_text())
            if pm.get('selection', {}).get('metric') != sel_metric:
                raise SystemExit(f'parent {pdir} was selected by {pm.get("selection")}, this fit by {sel_metric}')
            if pm.get('early_stopping', {}).get('metric') != stop_metric:
                raise SystemExit(f'parent {pdir} was stopped by {pm.get("early_stopping")}, this fit by {stop_metric}')
            pck = pdir / 'checkpoints' / pm['selection']['checkpoint']
            if pm.get('data_version_md5') != md5(dp['version']):
                raise SystemExit(f'parent {pdir} was trained on another data version')
            if int(pm.get('seed', -1)) != seed or pm.get('family') != family:
                raise SystemExit('parent seed or family differs')
        if not pck.exists():
            raise SystemExit(f'parent checkpoint missing: {pck}')
        twin = dict(cfg); twin['temporal_days'] = 0
        phases.load_static_checkpoint(model, lambda: build_model(build_backbone, twin), pck)
        parent = {'checkpoint': str(pck), 'md5': md5(pck)}
        ft, nlr, n_old, n_new_vars, n_new = configure_warm_optimizers(model, cfg, build_optimizer)
        lr_cb.enter_phase3(0, ft)
        print(f'  [warm start] {pck}  (md5 {parent["md5"]}); {n_old} copied variables at '
              f'lr={ft:g}, {n_new_vars} new weather variables ({n_new:,} parameters) at lr={nlr:g}',
              flush=True)

    x_dummy = [tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2))]
    if weather_shape(cfg):
        x_dummy.append(tf.zeros((1,) + weather_shape(cfg)))
    x_dummy = tuple(x_dummy)

    if a.smoke:
        print(f'=== SMOKE {a.smoke} steps ({stage}, {family}, T{T}, {prec}) ===', flush=True)
        class _Clock(keras.callbacks.Callback):
            def __init__(self):
                super().__init__(); self.t = []
            def on_train_batch_end(self, batch, logs=None):
                self.t.append(time.perf_counter())
        clk = _Clock()
        h = model.fit(train_ds, steps_per_epoch=a.smoke, epochs=1, verbose=2, callbacks=[lr_cb, clk])
        dt = np.diff(clk.t[min(20, len(clk.t) - 2):]) * 1000
        print('  losses:', {k: round(float(v[0]), 5) for k, v in h.history.items()}, flush=True)
        print(f'  steady state after step 20: median {np.median(dt):.0f} ms/step, p90 {np.percentile(dt, 90):.0f}'
              f' -> {np.median(dt) * train_ds.steps_per_epoch / 60000:.1f} min per training epoch'
              f' ({train_ds.steps_per_epoch} steps)', flush=True)
        return 0

    total_epochs = int(cfg.get('epochs', 100))
    # the ViT T0 gates checkpoints on its LR warm-up; a warm-started fit has no warm-up
    warmup = int(cfg.get('warmup_epochs', 5)) if (family == 'vit' and not warm) else 0
    ckpt = run_dir / 'checkpoints'; ckpt.mkdir(parents=True, exist_ok=True)
    initial_epoch = 0
    if a.resume:
        stamp, last = run_dir / 'last_epoch.json', ckpt / SEL_CKPT
        if not stamp.exists() or not last.exists():
            raise SystemExit(f'REFUSED --resume: need last_epoch.json and {SEL_CKPT}')
        initial_epoch = int(json.loads(stamp.read_text())['next_epoch'])
        # weights only: a .weights.h5 file holds no optimizer state, so the moments restart
        model.load_weights(str(last))
        print(f'  RESUMING at epoch {initial_epoch + 1} from {last}', flush=True)
    cbs = [lr_cb]
    if cfg.get('threshold_head', True):
        cbs.append(ThresholdEpochTracker())
    cbs += [PostWarmupBestCheckpoint(filepath=ckpt / SEL_CKPT, warmup_epochs=warmup,
                                     monitor=sel_metric, mode=sel_mode),
            PostWarmupBestCheckpoint(filepath=ckpt / 'best_loss.weights.h5', warmup_epochs=warmup,
                                     monitor='val_seg_loss', mode='min'),        # record only
            EpochStamp(run_dir),
            StepStatusCallback(run_dir / 'train_status.json', total_epochs, every=50),
            OneBasedCSVLogger(run_dir / 'training_log.csv', append=bool(initial_epoch)),
            keras.callbacks.EarlyStopping(monitor=stop_metric, mode=stop_mode,
                                          patience=int(cfg.get('patience', 15)), min_delta=0.0,
                                          restore_best_weights=False)]
    t0 = time.time()
    model.fit(train_ds, validation_data=val_ds, steps_per_epoch=train_ds.steps_per_epoch,
              epochs=total_epochs, initial_epoch=initial_epoch, callbacks=cbs, verbose=2)
    (run_dir / 'run_manifest.json').write_text(json.dumps({
        'stage': stage, 'family': family, 'seed': seed, 'temporal_days': T,
        'rows': rows, 'mixed_as': kw['mixed_as'], 'precision': prec,
        'data_version_md5': md5(dp['version']), 'data_version': version,
        'split_md5': md5(dp['split'] / 'patches.csv'), 'transforms_md5': md5(dp['transforms']),
        'parent': parent, 'hours': round((time.time() - t0) / 3600, 3),
        'selection': {'metric': sel_metric, 'mode': sel_mode, 'checkpoint': SEL_CKPT,
                      'record_checkpoint': 'best_loss.weights.h5',
                      **best_epochs(run_dir / 'training_log.csv', sel_metric, sel_mode, warmup)},
        'early_stopping': {'metric': stop_metric, 'mode': stop_mode,
                           'patience': int(cfg.get('patience', 15))},
        'params': int(sum(np.prod(v.shape) for v in model.trainable_variables)),
        'optimizer': type(model.optimizer).__name__, 'lr_schedule': type(lr_cb).__name__,
        'fine_tune_lr': float(cfg['fine_tune_lr']) if warm else None,
        'new_parts_lr': float(cfg.get('new_parts_lr', cfg['fine_tune_lr'])) if warm else None,
        'weather_params': int(sum(np.prod(v.shape) for layer in phases.weather_layers(model.backbone)
                                  for v in layer.trainable_variables)),
        'steps_per_epoch': int(train_ds.steps_per_epoch), 'environment': environment(),
        'status': 'final'}, indent=2))
    (run_dir / 'COMPLETE').write_text('ok\n')
    return 0
