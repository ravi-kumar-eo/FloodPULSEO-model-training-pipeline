"""AOI-weather encoders of the final programme (METHOD_PLAN_RERUN.md, C3).

Input  (B, T, K, K, 8) from `dataset_final.AOIWeather.window`:
       [P, P_valid, P_dx, P_dy, SM, SM_valid, SM_dx, SM_dy] per native cell, value 0 where invalid.
Output (B, T, out_dim): ONE vector per day. The day axis is kept, so the temporal paths of both
families (ViT day tokens, U-TAE per-day conditioning) still see the ordered daily sequence.

Per variable and day: a stack of 3x3 convolutions with GELU (`widths`) -> attention pooling with
one learned query over VALID cells only. Invalid cells already carry value 0 and validity 0 before
the convolutions (filled by the loader), and are excluded from the pooling softmax. A day with no
valid cell gets a learned "missing" vector; the masked softmax never sees only -inf, so no NaN.
The two pooled vectors and the two valid shares are projected to `out_dim` (`proj_layers` Dense
layers). Precipitation and soil moisture keep separate encoders because their K x K windows come
from different native grids (IMERG 0.1 deg, SMAP ~9 km): cell (i, j) is not the same ground in both.

This is regional hydroclimatic context around the patch; the offsets locate the patch relative to
the weather cells and do not delineate upstream catchments.

SIZES (2026-10-05, Ravi: the weather path must hold at least 20% of the T0 parameters).
Defaults reproduce the first final design (widths 16-32, one projection to 64: 14,882 parameters);
the final configs set widths 32-64-64, two projections to 256 (212,802 parameters).
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

MAX_DAYS = 30


class _VarEncoder(layers.Layer):
    def __init__(self, widths=(16, 32), **kw):
        super().__init__(**kw)
        self.widths = tuple(int(w) for w in widths)
        self.width = self.widths[-1]

    def build(self, input_shape):
        self.convs = [layers.Conv2D(w, 3, padding='same', activation='gelu', name=f'c{i + 1}')
                      for i, w in enumerate(self.widths)]
        self.score = layers.Dense(1, name='score')
        self.missing = self.add_weight(name='missing', shape=(self.width,),
                                       initializer=keras.initializers.RandomNormal(stddev=0.02))
        super().build(input_shape)

    def call(self, x):                                   # x (N, K, K, 4): value, valid, dx, dy
        valid = tf.cast(x[..., 1:2] > 0.5, tf.float32)   # (N,K,K,1)
        h = x
        for c in self.convs:
            h = c(h)                                     # (N,K,K,W)
        s = tf.cast(self.score(h), tf.float32)           # float32 for a stable softmax
        s = tf.where(valid > 0, s, tf.fill(tf.shape(s), -1e9))
        n = tf.shape(h)[0]
        s = tf.reshape(s, [n, -1])
        a = tf.nn.softmax(s, axis=-1)                    # finite even when all cells invalid
        hv = tf.reshape(h, [n, -1, self.width])
        pooled = tf.einsum('nk,nkw->nw', tf.cast(a, hv.dtype), hv)
        any_valid = tf.reduce_max(tf.reshape(valid, [n, -1]), axis=-1, keepdims=True)
        pooled = tf.where(any_valid > 0, pooled, tf.cast(self.missing, pooled.dtype)[None, :])
        share = tf.reduce_mean(tf.reshape(valid, [n, -1]), axis=-1, keepdims=True)
        return pooled, share


class AOIWeatherEncoder(layers.Layer):
    """(B, T, K, K, 8) -> (B, T, out_dim)."""

    def __init__(self, out_dim=64, widths=(16, 32), proj_layers=1, **kw):
        super().__init__(**kw)
        self.out_dim, self.widths, self.proj_layers = int(out_dim), tuple(widths), int(proj_layers)

    def build(self, input_shape):
        self.p_enc = _VarEncoder(self.widths, name='precip_enc')
        self.s_enc = _VarEncoder(self.widths, name='sm_enc')
        self.proj = [layers.Dense(self.out_dim, activation='gelu', name='day_proj')]
        for i in range(1, self.proj_layers):
            self.proj.append(layers.Dense(self.out_dim, name=f'day_proj{i + 1}'))
        super().build(input_shape)

    def call(self, w):
        B, T = tf.shape(w)[0], tf.shape(w)[1]
        K = w.shape[2]
        x = tf.reshape(w, [B * T, K, K, 8])
        p, ps = self.p_enc(x[..., 0:4])
        s, ss = self.s_enc(x[..., 4:8])
        d = tf.concat([p, s, tf.cast(ps, p.dtype), tf.cast(ss, p.dtype)], axis=-1)
        for layer in self.proj:
            d = layer(d)
        return tf.reshape(d, [B, T, self.out_dim])


class PatchWeatherEncoder(layers.Layer):
    """(B, T, 5) per-patch daily weather -> (B, T', out_dim) day vectors (Ravi, 2026-10-07).

    Input channels (dataset_final, weather_mode 'upstream'): [local rain, local soil moisture,
    valid, upstream rain, catchment size], rain as mm/day / precip_div and size as
    log10(1 + km2) / uda_div (constant over the days).

    series   one vector per day from that day's channels (two Dense layers).
    summary  the window summarised by fixed statistics with no tuned constant -- local and
             upstream rain: total, wettest day, last day; soil moisture: mean over valid days and
             last day; catchment size -- through two Dense layers into ONE vector that is added to
             every day vector, so each day's conditioning also carries the whole window.
    upstream False drops the upstream channels from both parts (ablation).
    series False -> a single 'day' carrying the summary vector (T' = 1, ablation).
    """

    def __init__(self, out_dim=256, series=True, summary=True, upstream=True, **kw):
        super().__init__(**kw)
        if not (series or summary):
            raise ValueError('PatchWeatherEncoder needs the series, the summary or both')
        self.out_dim, self.series, self.summary, self.upstream = int(out_dim), bool(series), bool(summary), bool(upstream)

    def build(self, input_shape):
        if int(input_shape[-1]) != 5:
            raise ValueError(f'PatchWeatherEncoder expects 5 channels, got {input_shape[-1]}')
        if self.series:
            self.day1 = layers.Dense(self.out_dim, activation='gelu', name='day1')
            self.day2 = layers.Dense(self.out_dim, name='day2')
        if self.summary:
            self.sum1 = layers.Dense(self.out_dim, activation='gelu', name='sum1')
            self.sum2 = layers.Dense(self.out_dim, name='sum2')
        super().build(input_shape)

    def summaries(self, w):
        """(B, 5) without upstream or (B, 9) with it, float32."""
        w = tf.cast(w, tf.float32)
        p, s, v = w[..., 0], w[..., 1], w[..., 2]
        f = [tf.reduce_sum(p, 1), tf.reduce_max(p, 1), p[:, -1],
             tf.reduce_sum(s * v, 1) / tf.maximum(tf.reduce_sum(v, 1), 1.0), s[:, -1]]
        if self.upstream:
            u, a = w[..., 3], w[..., 4]
            f += [tf.reduce_sum(u, 1), tf.reduce_max(u, 1), u[:, -1], a[:, -1]]
        return tf.stack(f, axis=-1)

    def call(self, w):
        x = w if self.upstream else w[..., :3]
        out = None
        if self.series:
            out = self.day2(self.day1(x))                                   # (B, T, D)
        if self.summary:
            sv = self.sum2(self.sum1(self.summaries(w)))                    # (B, D)
            out = sv[:, None, :] if out is None else out + tf.cast(sv, out.dtype)[:, None, :]
        return out


class DayTCN(layers.Layer):
    """U-Net arm: (B, T, d) -> (B, T, d), the day sequence in temporal context.

    The convolutional counterpart of the ViT's DayTransformer: four pre-norm residual blocks of a
    CAUSAL dilated 1-D convolution (kernel 3, dilations 1-2-4-8, receptive field
    1 + 2*(1+2+4+8) = 31 days, the whole window) -> GELU -> 1x1 convolution, then a final
    LayerNorm. Each day vector then also describes the days before it, so the per-day
    conditioning of the U-TAE path sees antecedent wetness, not only the weather of that day.
    At d = 256: 4 x 263,168 + 512 = 1,053,184 parameters (DayTransformer at d = 256, depth 2:
    1,054,208), so the two families' day-sequence encoders are matched.
    """

    def __init__(self, d_model: int = 256, dilations=(1, 2, 4, 8), kernel: int = 3, **kw):
        super().__init__(**kw)
        self.d_model, self.dilations, self.kernel = int(d_model), tuple(dilations), int(kernel)

    def build(self, input_shape):
        if int(input_shape[-1]) != self.d_model:
            raise ValueError(f'DayTCN expects day vectors of width {self.d_model}, got {input_shape[-1]}')
        self.norms = [layers.LayerNormalization(epsilon=1e-6, name=f'tcn_n{i}')
                      for i in range(len(self.dilations))]
        self.convs = [layers.Conv1D(self.d_model, self.kernel, padding='causal', dilation_rate=d,
                                    name=f'tcn_c{i}') for i, d in enumerate(self.dilations)]
        self.outs = [layers.Conv1D(self.d_model, 1, name=f'tcn_o{i}')
                     for i in range(len(self.dilations))]
        self.final = layers.LayerNormalization(epsilon=1e-6, name='tcn_final')
        super().build(input_shape)

    def call(self, x):
        for n, c, o in zip(self.norms, self.convs, self.outs):
            x = x + o(tf.nn.gelu(c(n(x))))
        return self.final(x)


class AOIDayTokens(layers.Layer):
    """ViT arm: (B, T, day_dim) day vectors -> (B, T, d_model) day tokens with RECENCY.

    Replaces DayTokenEmbed's per-value Fourier lift. The recency embedding is kept: row 0 is
    always day -1, whatever T is, exactly as in DayTokenEmbed.
    """

    def __init__(self, d_model=128, **kw):
        super().__init__(**kw)
        self.d_model = int(d_model)

    def build(self, input_shape):
        self.proj = layers.Dense(self.d_model, name='aoi_day_proj')
        self.day_emb = self.add_weight(name='day_emb', shape=(MAX_DAYS, self.d_model),
                                       initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, d):
        T = tf.shape(d)[1]
        rec = tf.reverse(self.day_emb[:T], axis=[0])     # (T, d): last row = day -1 = index 0
        x = self.proj(d) + tf.cast(rec, self.proj.compute_dtype)[None]
        return self.norm(x)
