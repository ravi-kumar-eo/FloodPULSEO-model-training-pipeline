"""Checks for the U-TAE weather path of the U-Net (flood/models/temporal/utae.py), 2026-10-04.

Run inside the training image, from experiments/src:
    python tests/test_utae.py --runs-dir <runs> [--seed 1] [--bench-batch 32]

1. parameter counts: T0, U-TAE, FiLM
2. paired warm start from this seed's Exp 1 checkpoint (phases.load_static_checkpoint)
3. identity at initialisation: U-TAE output = T0 output; two different weather records agree
4. after the injector moves: the output depends on the weather; masks differ between cells
5. one gradient step reaches the injector (L-TAE only after the injector has moved)
6. timing and peak memory of a training step at --bench-batch, T = 30
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from flood.models.static.unet import ResidualUNet
from flood.training import phases
from flood.training.core import _build_compiled

ap = argparse.ArgumentParser()
ap.add_argument('--runs-dir', required=True)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--bench-batch', type=int, default=32)
ap.add_argument('--bench-steps', type=int, default=20)
ap.add_argument('--precision', default='float32', choices=['float32', 'mixed_float16', 'mixed_bfloat16'])
ap.add_argument('--inject-level', type=int, default=0)
ap.add_argument('--no-tf32', action='store_true', help='full fp32 maths, to separate rounding from design')
a = ap.parse_args()
if a.no_tf32:
    tf.config.experimental.enable_tensor_float_32_execution(False)
if a.precision != 'float32':
    tf.keras.mixed_precision.set_global_policy(a.precision)   # benchmark only

tf.random.set_seed(0)
rng = np.random.default_rng(0)
base = {'backbone': 'unet', 'unet_base': 21, 'unet_levels': 4, 'threshold_head': True}
cu = dict(base, temporal_days=30, temporal_mode='utae', utae_heads=14,
          utae_inject_level=a.inject_level)
cf = dict(base, temporal_days=30)
ck = Path(a.runs_dir) / 'exp1' / f'seed{a.seed}' / 'unet' / 'checkpoints' / 'best_loss.weights.h5'


def inputs(B):
    return [tf.constant(rng.uniform(0, 1, (B, 256, 256, 2)), tf.float32),
            tf.constant(rng.uniform(0, 1, (B, 32, 32, 5)), tf.float32),
            tf.constant(rng.uniform(0, 1, (B, 16, 16, 2)), tf.float32)]


def real_batch(B):
    """B flood-bearing validation patches with their 30-day records (random inputs saturate T0)."""
    import os
    from flood.data.dataset import FloodPatches
    md = Path(os.environ['FLOOD_METADATA_DIR'])
    fp = FloodPatches(patches_csv=md / 'splits' / 'patches.csv',
                      patches_dir=os.environ['FLOOD_PATCHES_DIR'],
                      labels_dir=md / 'labels_80m_global', cohort='val',
                      flood_flags_csv=md / 'splits' / 'flood80_flags.csv', temporal_days=30)
    ds = fp.to_tf_dataset(batch_size=B, flood_ratio=0.5, seed=7, workers=4)
    xb, yb = next(iter(ds))
    return [tf.convert_to_tensor(t) for t in xb[:3]], tf.convert_to_tensor(xb[3]), yb


def weather(B, scale):
    return tf.constant(np.concatenate([rng.uniform(0, scale, (B, 30, 1)),
                                       rng.uniform(0, .5, (B, 30, 1)),
                                       np.ones((B, 30, 1))], -1), tf.float32)


def count(m):
    return sum(int(np.prod(v.shape)) for v in m.weights)


B = 8
x, wA, ylab = real_batch(B)
wB = tf.reverse(wA, axis=[0])          # the same patches with another patch's record

t0 = _build_compiled(ResidualUNet, dict(base))
t0.load_weights(str(ck))
mu = _build_compiled(ResidualUNet, cu)
mf = _build_compiled(ResidualUNet, cf)
print(f'[1] params  T0 {count(t0):,} | U-TAE {count(mu):,} (+{count(mu) - count(t0):,}) | '
      f'FiLM {count(mf):,} (+{count(mf) - count(t0):,})')

phases.load_static_checkpoint(mu, lambda: _build_compiled(ResidualUNet, dict(base)), ck)
m = mu.backbone
p0, tau0 = t0.backbone(x)
pA, tauA = m(x + [wA])
pB, tauB = m(x + [wB])
d0 = float(tf.reduce_max(tf.abs(pA - p0))); dt0 = float(tf.reduce_max(tf.abs(tauA - tau0)))
dAB = float(tf.reduce_max(tf.abs(pA - pB))); dtAB = float(tf.reduce_max(tf.abs(tauA - tauB)))
print(f'[3] identity vs T0       max|dprob| {d0:.2e}  max|dtau| {dt0:.2e}')
print(f'[3] weather A vs B, init max|dprob| {dAB:.2e}  max|dtau| {dtAB:.2e}')
ok3 = d0 < 1e-6 and dt0 < 1e-6 and dAB == 0.0 and dtAB == 0.0

bce = tf.keras.losses.BinaryCrossentropy()
ylab01 = tf.cast(ylab > 0.5, tf.float32)
with tf.GradientTape() as tp:
    pr, _ = m(x + [wA])
    loss = bce(ylab01, pr)
g = tp.gradient(loss, [m.day_inject.out.kernel, m.ltae.key.kernel])
gn = [float(tf.norm(v)) if v is not None else 0.0 for v in g]
print(f'[5] grad norm at init    injector {gn[0]:.3e}  L-TAE key {gn[1]:.3e} '
      f'(L-TAE is 0 until the days differ)')

k = m.day_inject.out.kernel
saved = k.numpy()
k.assign(tf.random.normal(k.shape, stddev=0.5))
pA2, _ = m(x + [wA])
pB2, _ = m(x + [wB])
dep = float(tf.reduce_max(tf.abs(pA2 - pB2)))
T = 30
xm = m.merge(tf.concat([m.stem_fine(x[0]), m.stem_med(x[1]), m.stem_coarse(x[2])], -1))
z = xm
for i, blocks in enumerate(m.enc_blocks):
    if i == m.utae_level:
        z = z[:, None] + m.day_inject(wA)[:, :, None, None, :]
        z = tf.reshape(z, [B * T] + list(z.shape[2:]))
    for blk in blocks:
        z = blk(z)
    if i < m.levels - 1:
        z = m.downs[i](z)
mk = m.ltae(tf.reshape(z, [B, T] + list(z.shape[1:]))).numpy()
spread = float(mk.reshape(B, m.utae_heads, T, 16).std(-1).mean())
print(f'[4] after injector moves max|dprob| A vs B {dep:.4f}  mean|dprob| {float(tf.reduce_mean(tf.abs(pA2 - pB2))):.4f}')
print(f'[4] masks {mk.shape}, sum over days = 1: {np.allclose(mk.sum(2), 1, atol=1e-5)}; '
      f'mean std across the 16 bottleneck cells {spread:.4f} (0 = one weighting per patch)')
with tf.GradientTape() as tp:
    pr, _ = m(x + [wA])
    loss = bce(ylab01, pr)
gk = tp.gradient(loss, m.ltae.key.kernel)
print(f'[5] grad norm L-TAE key after the injector moves {float(tf.norm(gk)):.3e}')
k.assign(saved)
ok4 = dep > 0 and spread > 0

# ------------------------------------------------------------------ timing
Bb = a.bench_batch
xb = inputs(Bb) + [weather(Bb, 3)]
yb = tf.constant((rng.uniform(0, 1, (Bb, 32, 32, 1)) > 0.9).astype(np.float32))
def make_step(model, opt):
  @tf.function
  def step(model=model):
    with tf.GradientTape() as tp:
        pr, tau = model(xb, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(yb, pr)) + tf.reduce_mean(tau)
    gr = tp.gradient(loss, model.trainable_variables)
    opt.apply_gradients([(g_, v) for g_, v in zip(gr, model.trainable_variables) if g_ is not None])
    return loss
  return step


for name, mm in (('U-TAE', m), ('FiLM', mf.backbone)):
    try:
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception:
        pass
    step = make_step(mm, tf.keras.optimizers.SGD(1e-3, momentum=0.99, nesterov=True))
    step(); step()
    t = time.perf_counter()
    for _ in range(a.bench_steps):
        step()
    _ = float(step())
    ms = (time.perf_counter() - t) / (a.bench_steps + 1) * 1000
    try:
        peak = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2**30
    except Exception:
        peak = float('nan')
    print(f'[6] {name:6s} batch {Bb}: {ms:.0f} ms/step (model only, no data loading); '
          f'peak GPU memory {peak:.1f} GiB')

print('RESULT', 'PASS' if (ok3 and ok4) else 'FAIL')
