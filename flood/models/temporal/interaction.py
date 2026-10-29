"""ONE interaction block per family, placed AFTER the static trunk.

WHY ONE LATE BLOCK, NOT FUSION AT EVERY DEPTH. Measured ICC 0.93 (precipitation) and 0.80 (soil
moisture) across the patches of one acquisition: weather is essentially CONSTANT within an
event, so it carries no per-patch information to inject early. Its only useful act is an
event-wide transformation, which one late block expresses fully. Multi-depth fusion was tried in
the archive (micro-AUPRC 0.757 -> 0.791) and is not the change that matters.

BOTH BLOCKS START AS THE IDENTITY. The ViT gate `g` and the U-Net's `gamma`/`beta` are zero-
initialised, so at step 0 the weather model computes EXACTLY what its M0 static checkpoint
computes. That is what makes the paired warm start valid: phase 1 begins at the M0 result, and
any change from there is attributable to the weather path.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ZERO = keras.initializers.Zeros()


class GatedCrossAttention(layers.Layer):
    """ViT arm: x <- x + g * CrossAttn(q=image tokens, kv=weather tokens).

    `g` is a single scalar initialised to 0, so this is the identity at initialisation. The
    image tokens stay the query throughout: weather modulates the spatial representation, it
    never replaces it.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 4, gate_init: float = 0.0,
                 ffn_mult: int = 0, **kw):
        super().__init__(**kw)
        self.d_model, self.num_heads = int(d_model), int(num_heads)
        self.gate_init = float(gate_init)
        # ffn_mult > 0 adds the gated feed-forward layer of Flamingo's GATED XATTN-DENSE block
        # (Alayrac et al., 2022): x <- x + g2 * FFW(LN(x)), g2 zero-initialised like g, so the
        # block stays the identity at initialisation. 0 keeps the first design (no FFW).
        self.ffn_mult = int(ffn_mult)

    def build(self, input_shape):
        self.nq = layers.LayerNormalization(epsilon=1e-6)
        self.nk = layers.LayerNormalization(epsilon=1e-6)
        # MultiHeadAttention already carries an output projection; adding a second Dense
        # costs 65,792 for nothing. Omitting it reproduces the drawn 198,401 exactly.
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads,
                                             key_dim=self.d_model // self.num_heads)
        # `g` MULTIPLIES the whole weather branch (x + g*h), so g = 0 is not merely "weather
        # muted" -- it sets the gradient of EVERY upstream weather variable to exactly zero.
        # Measured 2026-09-12 on a from-scratch model: with g = 0 only **3 of 58** weather
        # variables receive gradient (1 on the segmentation path); with g = 0.1, **58 of 58**.
        # `thr_weather` is different: a zero-init DENSE still gets gradient on its own kernel
        # because its input is non-zero. Only the multiplicative gate deadlocks.
        #
        # ZERO IS STILL THE DEFAULT and must stay so: it is what makes the paired warm start
        # valid (identity at init, so phase 1 begins exactly at the M0 result) -- the Flamingo
        # pattern, which exists for adapting a FROZEN pretrained model. Training from scratch
        # removes that premise, and a zero gate then just lets the trunk win the race and
        # converge before weather can contribute at all. Set `gate_init` in the config for
        # from-scratch runs ONLY.
        init = (ZERO if self.gate_init == 0.0
                else keras.initializers.Constant(self.gate_init))
        self.g = self.add_weight(name='gate', shape=(), initializer=init, trainable=True)
        if self.ffn_mult:
            self.nf = layers.LayerNormalization(epsilon=1e-6)
            self.ff = keras.Sequential([layers.Dense(self.d_model * self.ffn_mult, activation='gelu'),
                                        layers.Dense(self.d_model)])
            self.g2 = self.add_weight(name='gate_ffn', shape=(), initializer=init, trainable=True)
        super().build(input_shape)

    def call(self, x, w, training=False):
        h = self.att(self.nq(x), self.nk(w), training=training)
        x = x + tf.cast(self.g, x.dtype) * h
        if self.ffn_mult:
            x = x + tf.cast(self.g2, x.dtype) * self.ff(self.nf(x))
        return x


class FiLMConv(layers.Layer):
    """U-Net arm: h <- Conv(gamma * GroupNorm(h) + beta), gamma = 1 + Dense0(w), beta = Dense0(w).

    Both Dense layers are zero-initialised, so gamma = 1 and beta = 0 at step 0 and the block is
    a plain convolution of the normalised features. The residual form keeps it the identity:
    with the conv's own kernel free to learn, phase 1 still starts from the M0 output because the
    modulation contributes nothing until the Dense weights move.
    """

    def __init__(self, channels: int, groups: int = 6, **kw):
        super().__init__(**kw)
        self.channels, self.groups = int(channels), int(groups)

    def build(self, input_shape):
        self.gn = layers.GroupNormalization(groups=self.groups, epsilon=1e-6)
        self.to_gamma = layers.Dense(self.channels, kernel_initializer=ZERO,
                                     bias_initializer=ZERO, name='film_gamma')
        self.to_beta = layers.Dense(self.channels, kernel_initializer=ZERO,
                                    bias_initializer=ZERO, name='film_beta')
        self.conv = layers.Conv2D(self.channels, 3, padding='same', kernel_initializer=ZERO,
                                  name='film_conv')
        # Normalising the modulated branch before the residual add keeps its scale independent
        # of gamma/beta once they leave zero. Accounts for the drawn 21,546 (= 21,462 + 2*42).
        self.gn_out = layers.GroupNormalization(groups=self.groups, epsilon=1e-6)
        super().build(input_shape)

    def call(self, h, w, training=False):
        gamma = 1.0 + self.to_gamma(w)[:, None, None, :]
        beta = self.to_beta(w)[:, None, None, :]
        return h + self.gn_out(self.conv(gamma * self.gn(h) + beta))
