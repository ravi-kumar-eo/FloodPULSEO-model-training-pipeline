"""Standalone temporal encoders — one per family, each native to its backbone.

Drawn in `metadata/plots/arch_{vit,unet}_daily.png`. Both consume the SAME tensor:

    weather (B, T, 3) = [log1p(precip), soil moisture, valid]
    RIGHT-ALIGNED: index -1 is the day BEFORE the event reference date ("yesterday"),
    so slicing to a shorter window T is `w[:, -T:]` and "yesterday" never moves.

Ravi's rule (2026-09-02): weather is a TIME SERIES, not a feature vector — day order and day
identity must survive the encoder, and flattening to a vector is rejected. Neither family is
handicapped to match the other; parameter counts are REPORTED, never equalised.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..static.vit import LearnableFourierFeatures, TransformerBlock

MAX_DAYS = 30


class DayTokenEmbed(layers.Layer):
    """(B, T, 3) -> (B, T, d). One token per DAY.

    Each of the three variables is lifted by Learnable Fourier Features (a raw scalar carries
    almost nothing for dot-product attention), then summed with a learned VARIABLE embedding and
    a learned DAY embedding. The day embedding is indexed by RECENCY — row 0 is always
    yesterday — so a model trained at T=30 and evaluated at T=7 sees the same embedding for
    "yesterday", which is what makes the window study a slice rather than a retrain.
    """

    def __init__(self, d_model: int = 128, atom_dim: int = 64, n_vars: int = 3, **kw):
        super().__init__(**kw)
        self.d_model, self.atom_dim, self.n_vars = int(d_model), int(atom_dim), int(n_vars)

    def build(self, input_shape):
        self.lff = LearnableFourierFeatures(self.atom_dim, n_freq=32)
        self.var_emb = self.add_weight(
            name='var_emb', shape=(self.n_vars, self.atom_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.day_emb = self.add_weight(
            name='day_emb', shape=(MAX_DAYS, self.atom_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.proj = layers.Dense(self.d_model, name='day_proj')
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, w):
        T = tf.shape(w)[1]
        x = self.lff(tf.expand_dims(w, -1))                 # (B,T,V,atom)
        x = x + self.var_emb[None, None, :, :]
        # recency index: row 0 = the LAST day present, whatever T is
        rec = tf.reverse(self.day_emb[:T], axis=[0])        # (T,atom)
        x = x + rec[None, :, None, :]
        x = tf.reduce_sum(x, axis=2)                        # collapse the variable axis
        return self.norm(self.proj(x))                      # (B,T,d_model)


class DayTransformer(layers.Layer):
    """ViT arm: self-attention over DAY tokens — every day attends to every other day,
    before weather ever meets an image token. Depth 2 at d 128, as drawn."""

    def __init__(self, d_model: int = 128, depth: int = 2, num_heads: int = 4, **kw):
        super().__init__(**kw)
        self.d_model, self.depth, self.num_heads = int(d_model), int(depth), int(num_heads)

    def build(self, input_shape):
        # mlp_mult 2, not the trunk's 4: 2 x (attention 66,048 + norms 512 + mlp 65,920)
        # = 264,960, the drawn count. The day sequence is 30 tokens; a 4x MLP is not warranted.
        self.blocks = [TransformerBlock(self.d_model, self.num_heads, mlp_mult=2)
                       for _ in range(self.depth)]
        super().build(input_shape)

    def call(self, x, training=False):
        for b in self.blocks:
            x = b(x, training=training)
        return x


class GatedDilatedTCN(layers.Layer):
    """U-Net arm: 4 gated dilated CAUSAL convolutions, width 64, dilation 1-2-4-8.

    Receptive field 1 + 2*(1+2+4+8) = 31 days, i.e. the whole window. Causal padding means a day
    never sees its future. The convolutional family gets a convolutional temporal mechanism —
    the trunk is a CNN and cannot read a time axis at all, so this is the only place temporal
    reasoning happens on this arm.
    """

    def __init__(self, width: int = 64, dilations=(1, 2, 4, 8), **kw):
        super().__init__(**kw)
        self.width, self.dilations = int(width), tuple(dilations)

    def build(self, input_shape):
        self.inp = layers.Conv1D(self.width, 1, name='tcn_in')
        # No 1x1 residual projection: the gated output is already `width` wide, and adding
        # one costs 4 x (64*64+64) = 16,640 params for nothing. Dropping it reproduces the
        # drawn 99,584 exactly.
        self.filt, self.gate, self.norm = [], [], []
        for i, d in enumerate(self.dilations):
            self.filt.append(layers.Conv1D(self.width, 3, padding='causal', dilation_rate=d,
                                           activation='tanh', name=f'tcn_f{i}'))
            self.gate.append(layers.Conv1D(self.width, 3, padding='causal', dilation_rate=d,
                                           activation='sigmoid', name=f'tcn_g{i}'))
            self.norm.append(layers.LayerNormalization(epsilon=1e-6))
        super().build(input_shape)

    def call(self, w, training=False):
        x = self.inp(w)
        for f, g, n in zip(self.filt, self.gate, self.norm):
            x = n(x + f(x) * g(x))                          # gated residual, WaveNet-style
        return x                                            # (B,T,width)


class AttentionPool(layers.Layer):
    """(B, T, C) -> (B, C). A learned query attends over days, so the summary uses EVERY day
    rather than only the last one — the drawn note "uses every day, not just -1"."""

    def __init__(self, **kw):
        super().__init__(**kw)

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.q = self.add_weight(name='pool_q', shape=(1, c),
                                 initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.k = layers.Dense(c, name='pool_k')
        self.v = layers.Dense(c, name='pool_v')
        self.scale = float(c) ** -0.5
        super().build(input_shape)

    def call(self, x):
        a = tf.nn.softmax(tf.einsum('btc,qc->btq', self.k(x), self.q) * self.scale, axis=1)
        return tf.reduce_sum(a * self.v(x), axis=1)         # (B,C)
