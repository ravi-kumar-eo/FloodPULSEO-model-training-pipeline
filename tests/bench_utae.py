"""Time one training step of the U-Net weather paths (FiLM, U-TAE) at a given precision.

    python tests/bench_utae.py --precision mixed_bfloat16 --batch 32 --steps 20
Model only, synthetic inputs, no data loading. Prints ms/step and peak GPU memory.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument('--precision', default='float32', choices=['float32', 'mixed_float16', 'mixed_bfloat16'])
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--steps', type=int, default=20)
ap.add_argument('--days', type=int, default=30)
a = ap.parse_args()
if a.precision != 'float32':
    tf.keras.mixed_precision.set_global_policy(a.precision)
from flood.models.static.unet import ResidualUNet   # after the policy is set

rng = np.random.default_rng(0)
B, T = a.batch, a.days
x = [tf.constant(rng.uniform(0, 1, s), tf.float32) for s in
     ((B, 256, 256, 2), (B, 32, 32, 5), (B, 16, 16, 2), (B, T, 3))]
y = tf.constant((rng.uniform(0, 1, (B, 32, 32, 1)) > 0.9).astype(np.float32))
base = {'backbone': 'unet', 'unet_base': 21, 'unet_levels': 4, 'threshold_head': True, 'temporal_days': T}
for name, cfg in (('U-TAE', dict(base, temporal_mode='utae')), ('U-TAE@8x8', dict(base, temporal_mode='utae', utae_inject_level=2)), ('FiLM', dict(base))):
    m = ResidualUNet(cfg); m(x)
    opt = tf.keras.optimizers.SGD(1e-3, momentum=0.99, nesterov=True)
    if a.precision == 'mixed_float16':
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

    @tf.function
    def step():
        with tf.GradientTape() as tp:
            p, tau = m(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, tf.cast(p, tf.float32)))
            loss += tf.reduce_mean(tf.cast(tau, tf.float32))
            sl = opt.scale_loss(loss) if a.precision == 'mixed_float16' else loss
        g = tp.gradient(sl, m.trainable_variables)
        opt.apply_gradients([(gi, v) for gi, v in zip(g, m.trainable_variables) if gi is not None])
        return loss
    try:
        tf.config.experimental.reset_memory_stats('GPU:0')
    except Exception:
        pass
    step(); step()
    t = time.perf_counter()
    for _ in range(a.steps):
        out = step()
    float(out)
    ms = (time.perf_counter() - t) / a.steps * 1000
    peak = tf.config.experimental.get_memory_info('GPU:0')['peak'] / 2**30
    print(f'{a.precision:15s} T{T:<3d} {name:9s} batch {B}: {ms:6.0f} ms/step  peak {peak:5.1f} GiB  loss {float(out):.3f}')
