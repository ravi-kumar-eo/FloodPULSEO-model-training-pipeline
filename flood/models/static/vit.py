"""Plain vision transformer — the transformer backbone for M0.

STATIC ONLY. No weather branch, no CrossPatchMixer, no window. M0 compares two spatial
backbones on the land surface alone; weather belongs to RQ2 and the mixer was measured to
contribute capacity rather than cross-patch reasoning (permuting which patches share a window
changed flood IoU by -0.4%, while ablating the mixer entirely cost -35%).

INPUT GEOMETRY — the same power-of-two alignment the U-Net uses
--------------------------------------------------------------
One patch is 2.56 km = a 32x32 grid of 80 m cells = 1024 tokens, which is also the output grid.

    fine   10 m   256x256x2  ->  8x8 = 64 px per token   (256/32 = 8)
    medium 80 m    32x32x5   ->  1 px per token          (native)
    coarse 160 m   16x16x2   ->  1/4 px per token        (repeat x2)

Each modality is encoded AT ITS NATIVE RESOLUTION into one token per 80 m cell, then the three
modality tokens are fused. Nothing is average-pooled on the way in — averaging destroys the
HAND / upstream-drainage / flow-direction structure that carries the terrain signal.

S1 VV/VH are dropped from `fine` before it arrives (the model never sees SAR), leaving NDVI,
NDBI and the zeroed permanent-water band.

CAPACITY
--------
d_model=256, depth=8, heads=8 -> ~8.8M parameters, matching every run to date. Note this has
NEVER been varied across ten runs, so it is an untested assumption rather than a validated
choice; `d_model` and `depth` are config parameters so a pilot remains possible.

The components below (LearnableFourierFeatures, AxialCrossAttention, ModalityEncoder) are
ported faithfully from the validated model_v2_upe.py rather than reimplemented, so M0 tests
the architecture that produced the existing results.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..threshold_head import ThresholdHead


class LearnableFourierFeatures(layers.Layer):
    """Lift a scalar to `out_dim` via learnable Fourier features (Li et al. 2021).

    F = [cos(xW), sin(xW)] / sqrt(n_freq) -> MLP. A raw scalar carries almost no signal for a
    dot-product attention layer; the Fourier lift gives it a high-frequency representation
    that attention can actually separate.
    """

    def __init__(self, out_dim: int, n_freq: int = 32, **kw):
        super().__init__(**kw)
        self.out_dim, self.n_freq = int(out_dim), int(n_freq)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.Wr = self.add_weight(
            name='Wr', shape=(in_dim, self.n_freq),
            initializer=keras.initializers.RandomNormal(stddev=1.0), trainable=True)
        self.mlp = keras.Sequential([
            layers.Dense(self.out_dim, activation='gelu'),
            layers.Dense(self.out_dim)])
        super().build(input_shape)

    def call(self, x):
        proj = tf.matmul(x, self.Wr)
        ff = tf.concat([tf.cos(proj), tf.sin(proj)], axis=-1)
        ff = ff / tf.sqrt(tf.cast(self.n_freq, ff.dtype))
        return self.mlp(ff)


class AxialCrossAttention(layers.Layer):
    """Collapse axis A of (B, L, A, D) to (B, L, out_dim) by pooled-query cross-attention.

    Query = mean over A (one per L); keys/values from all A. Linear in the number of atoms
    rather than quadratic, which is what makes 64 pixels x 3 channels per token affordable.

    `l_chunk` splits the L axis. Attention is independent per L position so chunking is EXACT,
    not an approximation; it keeps the (B, lc, A, out) projections under the int32 element
    limit (B*L*A*out < 2^31) that otherwise caps batch size.
    """

    def __init__(self, out_dim: int, num_heads: int = 4, ff_mult: int = 2,
                 l_chunk: int = 256, **kw):
        super().__init__(**kw)
        assert out_dim % num_heads == 0, 'out_dim must be divisible by num_heads'
        self.out_dim, self.num_heads = int(out_dim), int(num_heads)
        self.hd = self.out_dim // self.num_heads
        self.ff_mult, self.l_chunk = int(ff_mult), int(l_chunk)

    def build(self, input_shape):
        self.q_proj = layers.Dense(self.out_dim)
        self.k_proj = layers.Dense(self.out_dim)
        self.v_proj = layers.Dense(self.out_dim)
        self.o_proj = layers.Dense(self.out_dim)
        self.n1 = layers.LayerNormalization(epsilon=1e-6)
        self.n2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = keras.Sequential([
            layers.Dense(self.out_dim * self.ff_mult, activation='gelu'),
            layers.Dense(self.out_dim)])
        super().build(input_shape)

    def _attend(self, x):
        B, L, A = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        q = self.q_proj(tf.reduce_mean(x, axis=2))                 # (B,L,out)
        k, v = self.k_proj(x), self.v_proj(x)                      # (B,L,A,out)
        q = tf.reshape(q, [B, L, self.num_heads, 1, self.hd])
        k = tf.reshape(k, [B, L, A, self.num_heads, self.hd])
        v = tf.reshape(v, [B, L, A, self.num_heads, self.hd])
        k = tf.transpose(k, [0, 1, 3, 2, 4])
        v = tf.transpose(v, [0, 1, 3, 2, 4])
        # Python-float scale: a tf.sqrt(float) constant is float32 and breaks mixed bfloat16
        att = tf.matmul(q, k, transpose_b=True) * (float(self.hd) ** -0.5)
        att = tf.nn.softmax(att, axis=-1)
        out = tf.matmul(att, v)                                    # (B,L,H,1,hd)
        out = tf.reshape(out, [B, L, self.out_dim])
        return self.o_proj(out)

    def call(self, x):
        L = x.shape[1]
        if L is not None and L > self.l_chunk:
            parts = [self._attend(x[:, i:i + self.l_chunk])
                     for i in range(0, L, self.l_chunk)]
            tok = tf.concat(parts, axis=1)
        else:
            tok = self._attend(x)
        tok = self.n1(tok)
        return tok + self.ff(self.n2(tok))


class ModalityEncoder(layers.Layer):
    """(B, N, P, C) scalar atoms -> (B, N, out_dim), one token per patch cell.

    Each atom = LFF(value) + channel embedding + position embedding, then the P*C atoms are
    collapsed by AxialCrossAttention. Learned channel and position identity means the encoder
    can tell "NDVI at pixel 7" from "NDBI at pixel 30" rather than seeing an unordered bag.
    """

    def __init__(self, n_positions: int, n_channels: int, out_dim: int,
                 atom_dim: int = 64, num_heads: int = 4, **kw):
        super().__init__(**kw)
        self.P, self.C = int(n_positions), int(n_channels)
        self.out_dim, self.atom_dim, self.num_heads = int(out_dim), int(atom_dim), int(num_heads)

    def build(self, input_shape):
        self.lff = LearnableFourierFeatures(self.atom_dim, n_freq=32)
        self.chan_emb = self.add_weight(
            name='chan_emb', shape=(self.C, self.atom_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.pos_emb = self.add_weight(
            name='pos_emb', shape=(self.P, self.atom_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.aca = AxialCrossAttention(self.out_dim, num_heads=self.num_heads)
        super().build(input_shape)

    def call(self, x):
        N = x.shape[1]
        v = self.lff(x[..., None])
        v = v + self.chan_emb[None, None, None, :, :]
        v = v + self.pos_emb[None, None, :, None, :]
        v = tf.reshape(v, [-1, N, self.P * self.C, self.atom_dim])
        return self.aca(v)



def sincos_2d(h: int, w: int, dim: int) -> tf.Tensor:
    """Fixed 2-D sine/cosine positional encoding for an h x w token grid -> (1, h*w, dim).

    WHY (2026-08-29): the UniversalPatchEncoder's `pos_emb` encodes position of a sub-pixel
    WITHIN an 80 m cell. Nothing told the transformer WHERE each of the 1,024 cells sits on
    the 32x32 grid, so global self-attention saw an unordered SET of tokens: it could not
    know which cells are adjacent, which direction is which, or where a cell lies in the
    patch. For terrain-driven flood prediction (upstream/downstream) that is a real handicap.

    Fixed rather than learned, so it adds ZERO trainable parameters and the capacity match
    against the U-Net (8,253,794 vs 8,224,042) is preserved exactly.
    Half the channels encode the row, half the column (Dosovitskiy 2021; Wang & Liu 2021).
    """
    assert dim % 4 == 0, f"dim must be divisible by 4 for 2-D sincos, got {dim}"
    d = dim // 2
    omega = 1.0 / (10000.0 ** (tf.range(d // 2, dtype=tf.float32) / float(d // 2)))
    def _axis(pos):                                    # pos: (n,) -> (n, d)
        out = tf.cast(pos, tf.float32)[:, None] * omega[None, :]
        return tf.concat([tf.sin(out), tf.cos(out)], axis=1)
    yy, xx = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
    pe = tf.concat([_axis(tf.reshape(yy, [-1])), _axis(tf.reshape(xx, [-1]))], axis=1)
    return pe[None, ...]                               # (1, h*w, dim)


class UniversalPatchEncoder(layers.Layer):
    """fine + medium + coarse -> (B, 1024, d_model). One token per 80 m cell."""

    def __init__(self, d_model: int, atom_dim: int = 64, num_heads: int = 4, **kw):
        super().__init__(**kw)
        self.d_model, self.atom_dim, self.num_heads = int(d_model), int(atom_dim), int(num_heads)

    def build(self, input_shape):
        # Fine channel count is READ, never hardcoded: the static programme feeds NDVI+NDBI
        # (2), earlier versions fed a zeroed permanent-water band as a third. A literal here
        # raises InvalidArgumentError inside the reshape below the moment that changes.
        fs = input_shape[0] if isinstance(input_shape[0], (list, tuple)) else input_shape
        self.fine_ch = int(fs[-1])
        self.enc_fine = ModalityEncoder(64, self.fine_ch, self.d_model, self.atom_dim,
                                        self.num_heads)
        self.enc_med = ModalityEncoder(1, 5, self.d_model, self.atom_dim, self.num_heads)
        self.enc_coarse = ModalityEncoder(1, 2, self.d_model, self.atom_dim, self.num_heads)
        self.mod_emb = self.add_weight(
            name='mod_emb', shape=(3, self.d_model),
            initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.fuse = AxialCrossAttention(self.d_model, num_heads=self.num_heads)
        super().build(input_shape)

    def call(self, fine, medium, coarse):
        ch = self.fine_ch
        f = tf.reshape(fine, [-1, 32, 8, 32, 8, ch])
        f = tf.transpose(f, [0, 1, 3, 2, 4, 5])
        f = tf.reshape(f, [-1, 1024, 64, ch])                      # 8x8 px per cell
        m = tf.reshape(medium, [-1, 1024, 1, 5])                   # 1 px per cell
        c = tf.repeat(tf.repeat(coarse, 2, axis=1), 2, axis=2)     # 16 -> 32
        c = tf.reshape(c, [-1, 1024, 1, 2])
        toks = tf.stack([self.enc_fine(f), self.enc_med(m), self.enc_coarse(c)], axis=2)
        toks = toks + self.mod_emb[None, None, :, :]
        return self.fuse(toks)


class TransformerBlock(layers.Layer):
    """Pre-norm self-attention + FFN."""

    def __init__(self, d_model: int, num_heads: int, mlp_mult: int = 4,
                 dropout: float = 0.05, **kw):
        super().__init__(**kw)
        self.d_model, self.num_heads = int(d_model), int(num_heads)
        self.mlp_mult, self.dropout = int(mlp_mult), float(dropout)

    def build(self, input_shape):
        self.n1 = layers.LayerNormalization(epsilon=1e-6)
        self.n2 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads,
            dropout=self.dropout)
        self.ff = keras.Sequential([
            layers.Dense(self.d_model * self.mlp_mult, activation='gelu'),
            layers.Dropout(self.dropout),
            layers.Dense(self.d_model)])
        super().build(input_shape)

    def call(self, x, training=False):
        h = self.n1(x)
        x = x + self.att(h, h, training=training)
        return x + self.ff(self.n2(x), training=training)


class Head80m(layers.Layer):
    """1024 tokens -> 32x32x1 logits. Three 3x3 convs on the token grid, then 1x1.

    Deliberately light and native at 80 m: v3's 10 m output was fabricated, since only
    NDVI/NDBI arrive at 10 m and neither observes water.
    """

    def __init__(self, width: int = 64, **kw):
        super().__init__(**kw)
        self.width = int(width)

    def build(self, input_shape):
        w = self.width
        self.body = keras.Sequential([
            layers.Conv2D(w, 3, padding='same', activation='gelu'),
            layers.Conv2D(w, 3, padding='same', activation='gelu'),
            layers.Conv2D(w, 3, padding='same', activation='gelu')])
        self.out = layers.Conv2D(1, 1, dtype='float32')
        super().build(input_shape)

    def call(self, tokens):
        b = tf.shape(tokens)[0]
        d = tokens.shape[-1]
        x = tf.reshape(tokens, [b, 32, 32, d])
        return self.out(self.body(x))


class PlainViT(keras.Model):
    """Static plain ViT with the standard per-patch threshold head.

    inputs  : (fine 256x256x2, medium 32x32x5, coarse 16x16x2)
    returns : (prob_map 32x32x1, tau (B,1))

    Same signature as ResidualUNet, so `train.py` swaps them on one config word and M0
    compares backbones rather than plumbing.
    """

    def __init__(self, config: dict | None = None, **kw):
        super().__init__(**kw)
        c = dict(config or {})
        d_model = int(c.get('d_model', 256))
        depth = int(c.get('depth', 8))
        heads = int(c.get('num_heads', 8))
        atom_dim = int(c.get('atom_dim', 64))
        dropout = float(c.get('dropout_rate', 0.05))
        self.use_thr_head = bool(c.get('threshold_head', True))

        self.upe = UniversalPatchEncoder(d_model, atom_dim, heads)
        # 32x32 token grid; non-trainable, so the parameter match with the U-Net is unchanged
        self._grid_pe = sincos_2d(32, 32, d_model)
        self.blocks = [TransformerBlock(d_model, heads, dropout=dropout) for _ in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = Head80m(width=int(c.get('head_width', 64)))
        self.thr_head = (ThresholdHead(width=int(c.get('thr_head_width', 64)))
                         if self.use_thr_head else None)

        # WEATHER PATH — built only when a window is configured. At temporal_days == 0 not one
        # weight below exists, so the model is bit-identical to M0 and an Exp 1 checkpoint loads
        # into it exactly. See metadata/plots/arch_vit_daily.png.
        self.temporal_days = int(c.get('temporal_days', 0))
        if self.temporal_days:
            from ..temporal.encoders import DayTokenEmbed, DayTransformer
            from ..temporal.interaction import GatedCrossAttention
            w_dim = int(c.get('w_dim', 128))
            # 'patch' = the old per-patch (T,3) record; 'aoi' = native AOI grids (final programme)
            self.weather_mode = str(c.get('weather_mode', 'patch')).lower()
            if self.weather_mode == 'aoi':
                from ..temporal.aoi_weather import AOIWeatherEncoder, AOIDayTokens
                self.aoi_enc = AOIWeatherEncoder(int(c.get('aoi_day_dim', 64)),
                                                 widths=tuple(c.get('aoi_widths', (16, 32))),
                                                 proj_layers=int(c.get('aoi_proj_layers', 1)))
                self.day_embed = AOIDayTokens(w_dim)
            elif self.weather_mode == 'upstream':
                # per-patch daily weather + upstream catchment rain (Ravi, 2026-10-07): the same
                # encoder as the U-Net, then the same day tokens as the AOI design
                from ..temporal.aoi_weather import PatchWeatherEncoder, AOIDayTokens
                self.patch_enc = PatchWeatherEncoder(int(c.get('aoi_day_dim', 64)),
                                                     series=bool(c.get('weather_series', True)),
                                                     summary=bool(c.get('weather_summary', True)),
                                                     upstream=bool(c.get('weather_upstream', True)))
                self.day_embed = AOIDayTokens(w_dim)
            else:
                self.day_embed = DayTokenEmbed(w_dim, atom_dim)
            self.day_encoder = DayTransformer(w_dim, depth=int(c.get('w_depth', 2)))
            # gate_init defaults to 0.0 = identity at init = the paired warm start stays valid.
            # Non-zero ONLY for from-scratch runs, where a zero gate deadlocks the weather path.
            self.interaction = GatedCrossAttention(
                d_model, num_heads=4, gate_init=float(c.get('gate_init', 0.0)),
                ffn_mult=int(c.get('xattn_ffn_mult', 0)))
            # tau IS the water level, and the trunk sees only static terrain — weather is the
            # head's ONLY source of event magnitude, so it is fed there as well.
            # ZERO-INITIALISED like the interaction gates: without this, weather perturbs
            # tau at step 0 and phase 1 does NOT start from the M0 result. Caught by the
            # identity test — prob was identical but tau was not.
            self.thr_weather = layers.Dense(
                d_model, kernel_initializer=keras.initializers.Zeros(),
                bias_initializer=keras.initializers.Zeros(), name='thr_weather')

    def call(self, inputs, training=False):
        fine, medium, coarse = inputs[0], inputs[1], inputs[2]
        weather = inputs[3] if (self.temporal_days and len(inputs) > 3) else None
        tokens = self.upe(fine, medium, coarse)                    # (B,1024,d)
        # Global grid position, added ONCE before self-attention. Without it the transformer
        # treats the 1,024 tokens as an unordered set (see sincos_2d docstring).
        tokens = tokens + tf.cast(self._grid_pe, tokens.dtype)
        for blk in self.blocks:
            tokens = blk(tokens, training=training)
        tokens = self.norm(tokens)

        # ONE interaction block, AFTER the trunk. `g` starts at 0, so before any weather
        # gradient flows this line is the identity and the model reproduces M0 exactly.
        w_tok = None
        if weather is not None:
            if getattr(self, 'aoi_enc', None) is not None:
                wx = self.aoi_enc(weather)
            elif getattr(self, 'patch_enc', None) is not None:
                wx = self.patch_enc(weather)
            else:
                wx = weather
            w_tok = self.day_encoder(self.day_embed(wx), training=training)
            tokens = self.interaction(tokens, w_tok, training=training)

        prob = tf.nn.sigmoid(tf.cast(self.head(tokens), tf.float32))
        if self.thr_head is None:
            return prob
        # stop_gradient: the threshold head reads the trunk but must never train it.
        # Its loss is gradient-isolated so the segmentation objective stays pure
        # (Ravi, 2026-08-29) — the head learns to READ features, not to reshape them.
        thr_in = tf.stop_gradient(tokens)
        if w_tok is not None:
            thr_in = thr_in + self.thr_weather(
                tf.reduce_mean(tf.stop_gradient(w_tok), axis=1))[:, None, :]
        return prob, self.thr_head(thr_in, training=training)
