"""Multi-resolution residual U-Net — the convolutional backbone for M0.

M0 asks whether a convolutional or a transformer backbone is the better spatial model under
one controlled protocol. That question is only meaningful if the U-Net is a CREDIBLE baseline,
not a strawman, so this is a residual U-Net with per-stream input stems and skip-connected
decoding, capacity-matched to the ViT (~8.8M parameters).

INPUT GEOMETRY — why everything is a power of two
-------------------------------------------------
One patch is 2.56 km = a 32x32 grid of 80 m cells, which is also the output grid. Every
stream aligns to it exactly:

    fine   10 m   256x256x2   ->  8x8 = 64 px per 80 m cell   (256/32 = 8)
    medium 80 m    32x32x5    ->  1 px per cell               (native)
    coarse 160 m   16x16x2    ->  1/4 px per cell             (16*2 = 32)

Each stream is encoded AT ITS NATIVE RESOLUTION and only then brought to 32x32:
10 m by three stride-2 convs (learned, 8x), 80 m as-is, 160 m by a stride-1 conv then a
nearest x2 upsample. Nothing is average-pooled on the way in — averaging destroys the
HAND / upstream-drainage / flow-direction structure that carries the terrain signal.

This mirrors what the ViT's UniversalPatchEncoder does (patchify 10 m into 8x8 atoms, take
80 m natively, repeat 160 m by 2), so M0 compares BACKBONES rather than input handling.

S1 VV/VH are dropped from `fine` before it arrives (the model never sees SAR), leaving
NDVI, NDBI and the zeroed permanent-water band.

CAPACITY
--------
base=32 with residual blocks lands near 8.8M, matching the ViT without distorting either
architecture. For reference: U-Net base 32 ~7-8M, base 64 ~31M, ResNet-18+U-Net ~14M,
the original Ronneberger U-Net ~31M.

NO WEATHER. M0 is static-only; the weather branch belongs to RQ2 and is deliberately absent
here so the two backbones are compared on land surface alone.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..threshold_head import ThresholdHead


class AdaptiveGroupNorm(layers.Layer):
    """GroupNorm whose group count is chosen at build time from the channel count.

    GroupNorm requires channels % groups == 0. Stem widths are derived from `base`, so a
    fixed groups=8 crashes for any base whose intermediate widths are not multiples of 8
    (base=24 produces 12 channels). Picking the largest valid divisor <= 8 keeps every
    configuration buildable.

    GroupNorm rather than BatchNorm because the effective batch is small (8-64 patches) and,
    more importantly, batches are CLASS-BALANCED 50/50 flood/dry by the sampler. BatchNorm
    statistics would describe a synthetic mixture that never occurs at inference, where ~84%
    of patches are dry.
    """

    def __init__(self, max_groups: int = 8, **kw):
        super().__init__(**kw)
        self.max_groups = int(max_groups)

    def build(self, input_shape):
        ch = int(input_shape[-1])
        g = next((k for k in range(min(self.max_groups, ch), 0, -1) if ch % k == 0), 1)
        self.gn = layers.GroupNormalization(groups=g, epsilon=1e-5)
        super().build(input_shape)

    def call(self, x):
        return self.gn(x)


def _norm():
    return AdaptiveGroupNorm(max_groups=8)


class ResBlock(layers.Layer):
    """Pre-activation residual block: GN -> GELU -> conv -> GN -> GELU -> conv (+ skip)."""

    def __init__(self, filters: int, **kw):
        super().__init__(**kw)
        self.filters = int(filters)

    def build(self, input_shape):
        f = self.filters
        self.n1, self.n2 = _norm(), _norm()
        self.c1 = layers.Conv2D(f, 3, padding='same', use_bias=False)
        self.c2 = layers.Conv2D(f, 3, padding='same', use_bias=False)
        self.proj = (layers.Conv2D(f, 1, use_bias=False)
                     if input_shape[-1] != f else None)
        super().build(input_shape)

    def call(self, x):
        skip = self.proj(x) if self.proj is not None else x
        h = self.c1(tf.nn.gelu(self.n1(x)))
        h = self.c2(tf.nn.gelu(self.n2(h)))
        return h + skip


class StreamStem(layers.Layer):
    """Encode one input stream at its native resolution, emit a 32x32 feature map.

    `down` stride-2 convs for the 10 m stream (8x = 3 convs), identity for 80 m, and a
    nearest x2 upsample for 160 m. Learned strided convs rather than pooling, for the reason
    in the module docstring.
    """

    def __init__(self, filters: int, down: int = 1, up: int = 1, **kw):
        super().__init__(**kw)
        self.filters, self.down, self.up = int(filters), int(down), int(up)

    def build(self, input_shape):
        f = self.filters
        self.convs = []
        n_down = {1: 0, 2: 1, 4: 2, 8: 3}[self.down]
        for i in range(n_down):
            # widen gradually: f//4 -> f//2 -> f
            w = max(8, f // (2 ** (n_down - i - 1)))
            self.convs.append(layers.Conv2D(w, 3, strides=2, padding='same', use_bias=False))
            self.convs.append(_norm())
        self.out = layers.Conv2D(f, 3, padding='same', use_bias=False)
        super().build(input_shape)

    def call(self, x):
        for lyr in self.convs:
            x = tf.nn.gelu(lyr(x)) if isinstance(lyr, layers.Conv2D) else lyr(x)
        if self.up > 1:
            x = tf.repeat(tf.repeat(x, self.up, axis=1), self.up, axis=2)
        return self.out(x)


class ResidualUNet(keras.Model):
    """Static multi-resolution residual U-Net with the standard per-patch threshold head.

    inputs  : (fine 256x256x2, medium 32x32x5, coarse 16x16x2)
    returns : (prob_map 32x32x1, tau (B,1))

    The threshold head is a STANDARD component of the baseline in both backbones. It reads a
    token sequence, so the flattened 4x4 bottleneck is passed to it — the same layer with the
    same parameter count the ViT uses, which is what keeps M0 a comparison of backbones.
    """

    def __init__(self, config: dict | None = None, **kw):
        super().__init__(**kw)
        c = dict(config or {})
        # base=21 -> 8,143,306 params = 0.987x the STATIC ViT's 8,253,794 (measured 2026-08-27).
        # Matched against the static ViT, not the 8.80M figure from earlier runs: that number
        # included the weather encoder and AOI-terrain branch, both absent in M0.
        # Capacity matching is on PARAMETERS; FLOPs differ by construction and are reported
        # separately, per thesis_plan §4.2 ("reported rather than assumed equal").
        self.base = int(c.get('unet_base', 21))
        self.levels = int(c.get('unet_levels', 4))          # 32 -> 16 -> 8 -> 4
        self.use_thr_head = bool(c.get('threshold_head', True))
        self.thr_width = int(c.get('thr_head_width', 64))

        b = self.base
        self.stem_fine = StreamStem(b, down=8)              # 256 -> 32
        self.stem_med = StreamStem(b, down=1)               # 32  -> 32
        self.stem_coarse = StreamStem(b, down=1, up=2)      # 16  -> 32
        self.merge = layers.Conv2D(b * 2, 1, use_bias=False)

        widths = [b * 2 * (2 ** i) for i in range(self.levels)]   # 64,128,256,512
        self.enc_blocks, self.downs = [], []
        for i, w in enumerate(widths):
            self.enc_blocks.append([ResBlock(w), ResBlock(w)])
            if i < self.levels - 1:
                self.downs.append(layers.Conv2D(widths[i + 1], 3, strides=2,
                                                padding='same', use_bias=False))

        self.dec_blocks, self.ups = [], []
        for i in range(self.levels - 2, -1, -1):
            self.ups.append(layers.Conv2DTranspose(widths[i], 2, strides=2, use_bias=False))
            self.dec_blocks.append([ResBlock(widths[i]), ResBlock(widths[i])])

        self.head_norm = _norm()
        self.head = layers.Conv2D(1, 1, dtype='float32')    # logits at 32x32
        self.thr_head = ThresholdHead(width=self.thr_width) if self.use_thr_head else None

        # WEATHER PATH — built only when a window is configured. At temporal_days == 0 not one
        # weight below exists, so every ResBlock is bit-identical to M0 and an Exp 1 checkpoint
        # loads exactly. See metadata/plots/arch_unet_daily.png.
        self.temporal_days = int(c.get('temporal_days', 0))
        # 'film' = the 2 Sep 2026 path (TCN -> one pooled vector per patch -> FiLM after the
        # decoder). 'utae' = U-TAE (per-cell temporal attention masks; temporal/utae.py), the
        # design decision Ravi restated on 2026-10-04. Default 'film' keeps every sealed run
        # and config bit-identical.
        self.temporal_mode = str(c.get('temporal_mode', 'film')).lower()
        if self.temporal_days and self.temporal_mode == 'utae':
            from ..temporal.utae import DayInjector, LTAEMasks
            self.utae_heads = int(c.get('utae_heads', 14))
            if any(w % self.utae_heads for w in widths):
                raise ValueError(f'utae_heads={self.utae_heads} must divide every encoder '
                                 f'width {widths}')
            # Encoder level at which the day-t weather enters (Ravi 2026-10-04, option D):
            # 0 = the merged 32x32 map, as in U-TAE (encoder runs T times in full);
            # 2 = the 8x8 level, so only the two deepest levels run once per day (~half the cost).
            # Levels above it run once and their skips carry no temporal dimension.
            self.utae_level = int(c.get('utae_inject_level', 0))
            if not 0 <= self.utae_level < self.levels:
                raise ValueError(f'utae_inject_level must be in 0..{self.levels - 1}')
            # How the day enters (2026-10-05): 'offset' = the first final design, an additive
            # per-day offset on the raw features (measured to be ignored, see FiLMResBlock);
            # 'film' = a FiLM-conditioned residual block on the normalised features, with the day
            # sequence first put in temporal context by a DayTCN.
            self.utae_condition = str(c.get('utae_condition', 'offset')).lower()
            if str(c.get('weather_mode', 'patch')).lower() == 'aoi':
                from ..temporal.aoi_weather import AOIWeatherEncoder
                self.aoi_enc = AOIWeatherEncoder(int(c.get('aoi_day_dim', 64)),
                                                 widths=tuple(c.get('aoi_widths', (16, 32))),
                                                 proj_layers=int(c.get('aoi_proj_layers', 1)))
            elif str(c.get('weather_mode', 'patch')).lower() == 'upstream':
                # per-patch daily weather + upstream catchment rain (Ravi, 2026-10-07)
                from ..temporal.aoi_weather import PatchWeatherEncoder
                self.patch_enc = PatchWeatherEncoder(int(c.get('aoi_day_dim', 64)),
                                                     series=bool(c.get('weather_series', True)),
                                                     summary=bool(c.get('weather_summary', True)),
                                                     upstream=bool(c.get('weather_upstream', True)))
            if self.utae_condition == 'film':
                if getattr(self, 'aoi_enc', None) is None and getattr(self, 'patch_enc', None) is None:
                    raise ValueError("utae_condition 'film' needs weather_mode: aoi or upstream")
                from ..temporal.aoi_weather import DayTCN
                from ..temporal.utae import FiLMResBlock
                self.day_tcn = (DayTCN(int(c.get('aoi_day_dim', 64)))
                                if str(c.get('day_context', 'none')).lower() == 'tcn' else None)
                self.cond_block = FiLMResBlock(widths[self.utae_level])
            elif self.utae_condition == 'offset':
                self.day_inject = DayInjector(widths[self.utae_level],
                                              hidden=int(c.get('utae_inject_hidden', 64)))
            else:
                raise ValueError(f'unknown utae_condition {self.utae_condition!r}')
            self.ltae = LTAEMasks(n_head=self.utae_heads, d_k=int(c.get('utae_dk', 4)),
                                  d_model=int(c.get('utae_dmodel', 256)))
        elif self.temporal_days:
            from ..temporal.encoders import GatedDilatedTCN, AttentionPool
            from ..temporal.interaction import FiLMConv
            self.tcn = GatedDilatedTCN(int(c.get('w_width', 64)))
            self.tpool = AttentionPool()
            self.interaction = FiLMConv(widths[0])
            # tau IS the water level, and the trunk sees only static terrain — weather is the
            # head's ONLY source of event magnitude, so it is fed there as well.
            # ZERO-INITIALISED like the interaction gates: without this, weather perturbs
            # tau at step 0 and phase 1 does NOT start from the M0 result. Caught by the
            # identity test — prob was identical but tau was not.
            self.thr_weather = layers.Dense(
                widths[-1], kernel_initializer=keras.initializers.Zeros(),
                bias_initializer=keras.initializers.Zeros(), name='thr_weather')

    def call(self, inputs, training=False):
        fine, medium, coarse = inputs[0], inputs[1], inputs[2]
        weather = inputs[3] if (self.temporal_days and len(inputs) > 3) else None

        x = tf.concat([self.stem_fine(fine),
                       self.stem_med(medium),
                       self.stem_coarse(coarse)], axis=-1)
        x = self.merge(x)                                    # (B,32,32,2b)

        utae = weather is not None and self.temporal_mode == 'utae'
        if utae:
            B, T = tf.shape(x)[0], tf.shape(weather)[1]

        skips = []
        for i, blocks in enumerate(self.enc_blocks):
            if utae and i == self.utae_level:
                # one map per day from here on; the encoder levels below are shared over T
                # (batched as B*T). Both conditionings are the identity at initialisation.
                if getattr(self, 'aoi_enc', None) is not None:
                    wx = self.aoi_enc(weather)
                elif getattr(self, 'patch_enc', None) is not None:
                    wx = self.patch_enc(weather)
                    T = tf.shape(wx)[1]          # 1 when only the window summary is given
                else:
                    wx = weather
                if self.utae_condition == 'film':
                    if self.day_tcn is not None:
                        wx = self.day_tcn(wx)                                     # (B,T,D)
                    x = tf.tile(x[:, None], [1, T, 1, 1, 1])                      # (B,T,H,W,C)
                    x = tf.reshape(x, [B * T] + list(x.shape[2:]))
                    x = self.cond_block(x, tf.reshape(wx, [B * T, wx.shape[-1]]))
                else:
                    x = x[:, None] + self.day_inject(wx)[:, :, None, None, :]    # (B,T,H,W,C)
                    x = tf.reshape(x, [B * T] + list(x.shape[2:]))
            for blk in blocks:
                x = blk(x)
            if i < self.levels - 1:
                skips.append(x)
                x = self.downs[i](x)                          # 32->16->8->4

        if utae:
            from ..temporal.utae import collapse

            def unfold(z):
                return tf.reshape(z, [B, T] + list(z.shape[1:]))
            seq = unfold(x)                                   # (B,T,4,4,C)
            masks = self.ltae(seq)                            # (B,heads,T,4,4)
            x = collapse(seq, masks, self.utae_heads)
            skips = [collapse(unfold(s), masks, self.utae_heads) if i >= self.utae_level else s
                     for i, s in enumerate(skips)]

        bottleneck = x                                        # (B,4,4,512)

        for up, blocks, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            x = up(x)
            x = tf.concat([x, skip], axis=-1)
            for blk in blocks:
                x = blk(x)

        # ONE FiLM block, AFTER the decoder. gamma = 1 and beta = 0 at initialisation, so
        # before any weather gradient flows this line is the identity and the model
        # reproduces M0 exactly. Conditioning the ENCODER instead would break that guarantee.
        w_vec = None
        if weather is not None and not utae:
            w_vec = self.tpool(self.tcn(weather, training=training))     # (B, w_width)
            x = self.interaction(x, w_vec, training=training)

        logits = self.head(tf.nn.gelu(self.head_norm(x)))     # (B,32,32,1)
        prob = tf.nn.sigmoid(tf.cast(logits, tf.float32))

        if self.thr_head is None:
            return prob
        b = tf.shape(bottleneck)[0]
        d = bottleneck.shape[-1]
        tokens = tf.reshape(bottleneck, [b, -1, d])           # (B,16,512)
        # stop_gradient: the threshold head reads the trunk but must never train it.
        # Its loss is gradient-isolated so the segmentation objective stays pure
        # (Ravi, 2026-08-29) — the head learns to READ features, not to reshape them.
        thr_in = tf.stop_gradient(tokens)
        if w_vec is not None:
            thr_in = thr_in + self.thr_weather(tf.stop_gradient(w_vec))[:, None, :]
        return prob, self.thr_head(thr_in, training=training)
