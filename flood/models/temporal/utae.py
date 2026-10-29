"""U-TAE temporal path for the U-Net (Sainte Fare Garnot & Landrieu, 2021), adapted to one
static image plus a daily weather record.

WHY (Ravi, 2026-10-04). The U-Net weather path of 2 Sep 2026 (TCN -> AttentionPool -> one FiLM)
weights the days ONCE PER PATCH. The design decision was U-TAE, where every pixel weights the
dates with its own attention mask, as every ViT token does through cross-attention.

WHY A LITERAL L-TAE ON THE WEATHER ALONE IS NOT ENOUGH. Weather is one value per patch and day,
so every cell would see the same sequence, the keys would be identical and every cell would get
the same mask -- the per-patch design again. U-TAE gets per-pixel masks because its SHARED
ENCODER runs on every date. Here the per-date input is [static stems output + day-t weather],
so the encoder output of day t differs between cells through their land surface, and the keys
of the L-TAE are cell-dependent.

STRUCTURE, as in U-TAE (their Fig. 2):
  1. stems run ONCE (static inputs do not change between days); day-t weather is injected into
     the merged 32x32 map by a small per-day MLP            -> T maps per patch
  2. the U-Net ENCODER (the T0 encoder, weights shared) runs on all T maps (batched as B*T)
  3. an L-TAE at the 4x4 bottleneck: learned master query per head, keys from the per-day
     features + a sinusoidal RECENCY encoding (day -1 = position 0 for every window length)
     -> one attention mask per head, per day, per bottleneck cell
  4. masks are bilinearly upsampled to 8x8, 16x16, 32x32 and collapse the T feature maps of
     every level (U-TAE 'att_group': channels split into n_head groups, group h weighted by
     head h), bottleneck included
  5. the decoder runs once on the collapsed maps, exactly as in T0

IDENTITY AT INITIALISATION. The last layer of the injector is zero-initialised, so all T maps
equal the T0 map, and the collapse is anchored on day -1 (see `collapse`), so the network computes
exactly the T0 prediction and the paired warm start stays valid.

DEVIATIONS FROM U-TAE, kept deliberately small:
  * the bottleneck is collapsed with the masks like the skips; U-TAE passes the L-TAE value
    output through an MLP, which would break the identity start;
  * n_head must divide every encoder width (42, 84, 168, 336) -> 14 heads (U-TAE: 16);
  * positions are recency indices (days before the mapping date), not days of the year.
"""
from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ZERO = keras.initializers.Zeros()


class DayInjector(layers.Layer):
    """(B, T, 3) weather -> (B, T, C) additive per-day offset for the merged 32x32 map.

    Zero-initialised output layer: at step 0 every day adds 0, so all T maps are the T0 map.
    """

    def __init__(self, channels: int, hidden: int = 64, **kw):
        super().__init__(**kw)
        self.channels, self.hidden = int(channels), int(hidden)

    def build(self, input_shape):
        self.h = layers.Dense(self.hidden, activation='gelu', name='inject_h')
        self.out = layers.Dense(self.channels, kernel_initializer=ZERO, bias_initializer=ZERO,
                                name='inject_out')
        super().build(input_shape)

    def call(self, w):
        return self.out(self.h(w))


def recency_encoding(T: int, d: int) -> tf.Tensor:
    """(T, d) sinusoidal encoding of the recency index; row T-1 (day -1) has index 0."""
    pos = tf.cast(tf.range(T - 1, -1, -1), tf.float32)[:, None]          # T-1 ... 0
    i = tf.cast(tf.range(d // 2), tf.float32)[None, :]
    ang = pos / tf.pow(1000.0, 2.0 * i / float(d))                       # U-TAE uses T=1000
    return tf.concat([tf.sin(ang), tf.cos(ang)], axis=-1)


class LTAEMasks(layers.Layer):
    """L-TAE (Sainte Fare Garnot & Landrieu, 2020) mask branch on a (B, T, h, w, C) sequence.

    Returns attention masks (B, n_head, T, h, w), softmax over T. Master queries are learned
    parameters, one per head; keys come from each cell's own per-day features, so cells with
    different land surface get different masks.
    """

    def __init__(self, n_head: int = 14, d_k: int = 4, d_model: int = 256, **kw):
        super().__init__(**kw)
        self.n_head, self.d_k, self.d_model = int(n_head), int(d_k), int(d_model)

    def build(self, input_shape):
        self.in_norm = layers.LayerNormalization(epsilon=1e-6, name='ltae_in_norm')
        self.in_proj = layers.Dense(self.d_model, name='ltae_in_proj')
        self.key = layers.Dense(self.n_head * self.d_k, name='ltae_key')
        self.q = self.add_weight(name='ltae_master_query', shape=(self.n_head, self.d_k),
                                 initializer=keras.initializers.RandomNormal(
                                     stddev=math.sqrt(2.0 / self.d_k)))
        super().build(input_shape)

    def call(self, x):
        T = tf.shape(x)[1]
        h = self.in_proj(self.in_norm(x))                                 # (B,T,h,w,dm)
        h = h + tf.cast(recency_encoding(T, self.d_model), h.dtype)[None, :, None, None, :]
        k = self.key(h)                                                    # (B,T,h,w,H*dk)
        s = tf.shape(k)
        k = tf.reshape(k, [s[0], s[1], s[2], s[3], self.n_head, self.d_k])
        logits = tf.einsum('bthwnd,nd->bnthw', tf.cast(k, tf.float32), tf.cast(self.q, tf.float32))
        return tf.nn.softmax(logits / math.sqrt(self.d_k), axis=2)      # over T, float32 for stability


def collapse(feats, masks, n_head: int):
    """U-TAE 'att_group' aggregation.

    feats (B, T, H, W, C), masks (B, n_head, T, h, w) with h <= H. Masks are bilinearly
    resized to (H, W); channels are split into n_head groups and group g is summed over T with
    the weights of head g. Returns (B, H, W, C).
    """
    fs = tf.shape(feats)
    B, T, H, W = fs[0], fs[1], fs[2], fs[3]
    C = feats.shape[-1]
    m = tf.cast(masks, feats.dtype)
    if masks.shape[-1] != feats.shape[3]:
        ms = tf.shape(masks)
        m = tf.reshape(m, [ms[0] * n_head * ms[2], ms[3], ms[4], 1])
        m = tf.cast(tf.image.resize(m, [H, W], method='bilinear'), feats.dtype)  # resize returns float32
        m = tf.reshape(m, [B, n_head, T, H, W])
    # Anchored on day -1: f[-1] + sum_t a_t (f_t - f[-1]) equals sum_t a_t f_t because the
    # weights sum to 1, and is EXACTLY f[-1] (= the T0 map) while all days are identical, so the
    # identity start holds without float rounding from a 30-term sum.
    anchor = feats[:, -1]                                               # (B,H,W,C)
    f = tf.reshape(feats - anchor[:, None], [B, T, H, W, n_head, C // n_head])
    out = tf.einsum('bthwgc,bgthw->bhwgc', f, m)
    return anchor + tf.reshape(out, [B, H, W, C])


class FiLMResBlock(layers.Layer):
    """Per-day weather conditioning of the U-Net features, at the level where the days enter.

    WHY (2026-10-05). The first final design ADDED a day offset (DayInjector) to the raw 8x8
    features. Measured on the seed-1 T0 network: those features have a std of ~29 (the residual
    stream is never normalised), the offset ~0.003, and an additive offset is the same at every
    position, which the GroupNorms downstream largely remove. A 1e-2 offset moved the probability
    by 5e-7, the injector's gradient was ~150x smaller per parameter than the trunk's, and the
    trained T30 ignored its weather exactly (bf16 rounded the offset away entirely).

    HERE the day vector modulates the NORMALISED features inside a residual branch (FiLM;
    Perez et al., 2018): x_t = x + conv0(GELU(GN(conv(GELU(GN(x) * (1 + gamma_t) + beta_t))))).
    The modulation acts on unit-scale features, so the effect of the weather varies with the land
    surface of each position and survives the normalisations downstream; the branch output has the
    scale of any residual branch of the trunk.

    IDENTITY AT INITIALISATION: the last convolution `c2` is zero-initialised (as the zero
    convolutions of ControlNet), so x_t = x for every day and the network computes the T0
    prediction exactly. `c2` receives gradient from the first step, because its input is not zero
    and already depends on the weather through gamma/beta (the FiLM projection is randomly
    initialised, std 0.02), so the weather path does not wait for a scalar gate to move.

    Parameters at 168 channels and a 256-wide day vector: 2 GN (672) + FiLM Dense 256 -> 336
    (86,352) + two 3x3 convolutions (2 x 254,016) = 595,056.
    """

    def __init__(self, channels: int, **kw):
        super().__init__(**kw)
        self.channels = int(channels)

    def build(self, input_shape):                       # input_shape = shape of x
        from ..static.unet import _norm                 # lazy: unet imports this module lazily
        C = self.channels
        self.n1, self.n2 = _norm(), _norm()
        self.film = layers.Dense(2 * C, kernel_initializer=keras.initializers.RandomNormal(stddev=0.02),
                                 bias_initializer=ZERO, name='film')
        self.c1 = layers.Conv2D(C, 3, padding='same', use_bias=False, name='cond_c1')
        self.c2 = layers.Conv2D(C, 3, padding='same', use_bias=False, kernel_initializer=ZERO,
                                name='cond_c2')
        super().build(input_shape)

    def call(self, x, cond):                            # x (N,H,W,C), cond (N,D)
        g, b = tf.split(tf.cast(self.film(cond), x.dtype), 2, axis=-1)
        h = self.n1(x) * (1.0 + g[:, None, None, :]) + b[:, None, None, :]
        h = self.c1(tf.nn.gelu(h))
        h = self.c2(tf.nn.gelu(self.n2(h)))
        return x + h
