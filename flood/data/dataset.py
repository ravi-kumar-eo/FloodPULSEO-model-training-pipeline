"""Patch loader for the confirmatory programme.

Reads `metadata/splits/patches.csv`, filters by `cohort`, and yields
    (fine 256x256x2, medium 32x32x5, coarse 16x16x2), label 32x32x1

WEATHER is loaded ONLY when `temporal_days > 0` (Exp 2/3/4). At `temporal_days = 0` (Exp 1)
`input_2560m` is not read at all and the loader is byte-identical to the static programme, which
is what lets an Exp 1 checkpoint load into an Exp 2 trunk exactly.

`patch_XXXX_input_2560m.tif` holds 60 bands at 1x1 px: bands 0-29 precipitation, 30-59 soil
moisture, each ordered OLDEST -> NEWEST with the last entry the day BEFORE the event reference
date. It is returned as (T, 3) = [log1p(precip), soil moisture, valid], RIGHT-ALIGNED so index
-1 is always yesterday and a shorter window is the slice `[-T:]` — the window study is then a
slice, not a retrain.

S1 VV/VH are dropped from the 10 m stream inside `load_patch`, so the model never sees SAR.
Permanent water is dropped there too, so what reaches the model is NDVI and NDBI alone.
PW is the very thing being predicted — an earlier version leaked it and the model simply
copied an input channel (correlation 0.9942). It was then ZEROED rather than dropped, which
is numerically identical (zero x weight = 0) but carried a third of every batch, and a
third of the RAM cache, as constant zeros. Dropped outright 2026-09-06.

LABELS are the 80 m p25 pooling: an 80 m cell is flood iff >= 16 of its 64 ten-metre
sub-pixels qualify. Permanent water is written as -1, a marker the loss uses to down-weight
or skip those cells rather than punish the model for correctly finding rivers.

CLASS BALANCE (training only). Each batch is half flood-bearing and half dry, using the
`has_flood80` flag from `flood80_flags.csv`. That flag is computed FROM THE 80 m LABELS, not
from `flood_fraction`: flood_fraction is measured at 10 m and marks 3,811 train patches as
flood whose 80 m target is all-zero after p25 pooling, so training on it teaches the model to
find flood that its target says is not there.

VAL AND TEST ARE NEVER BALANCED. They keep the natural distribution (~84% dry) so the metrics
describe the real split rather than a synthetic mixture.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf

NODATA_SENTINEL = 9999.0
WEATHER_DAYS = 30                # bands per variable in input_2560m.tif
PRECIP_CLIP = 100.0              # mm/day; log1p'd first, so this only bounds extremes
SOIL_MEDIAN = (21.0, 40.0)       # clay %, sand % — neutral fill, not "very sandy"


def _clip_normalize(arr, lo, hi):
    return (np.clip(arr, lo, hi) - lo) / (hi - lo + 1e-8)


def normalize(d10, d80, d160):
    """Per-band clip-to-[p01,p99] then scale to [0,1]. Constants from 50,000 sampled patches.

    Ported verbatim from the validated loader so results stay comparable. UDA is log10'd
    first (7 orders of magnitude); HAND's negative values are resampling artifacts.
    """
    o10 = np.empty_like(d10)
    o10[..., 0] = _clip_normalize(d10[..., 0], -22.1, -2.0)     # S1_VV  (dropped later)
    o10[..., 1] = _clip_normalize(d10[..., 1], -34.4, 0.0)      # S1_VH  (dropped later)
    o10[..., 2] = _clip_normalize(d10[..., 2], -1.0, 1.0)       # NDVI
    o10[..., 3] = _clip_normalize(d10[..., 3], -1.0, 1.0)       # NDBI
    o10[..., 4] = np.clip(d10[..., 4], 0.0, 1.0)                # PermWater (zeroed)

    o80 = np.empty_like(d80)
    o80[..., 0] = _clip_normalize(d80[..., 0], 0.0, 1511.0)     # elevation
    o80[..., 1] = _clip_normalize(d80[..., 1], -1.0, 1.0)       # flow sin
    o80[..., 2] = _clip_normalize(d80[..., 2], -1.0, 1.0)       # flow cos
    o80[..., 3] = _clip_normalize(np.log10(np.maximum(d80[..., 3], 0.0) + 1.0), 0.0, 2.193)
    o80[..., 4] = _clip_normalize(d80[..., 4], 0.0, 300.0)      # HAND

    o160 = np.empty_like(d160)
    o160[..., 0] = _clip_normalize(d160[..., 0], 4.2, 40.4)     # clay
    o160[..., 1] = _clip_normalize(d160[..., 1], 17.8, 82.3)    # sand
    return o10, o80, o160


def _read(path, retries=3):
    """Read a patch GeoTIFF, retrying transient failures.

    `/shared_data` is a shared network volume and a read can fail transiently under
    concurrent access. A single such failure killed a run at epoch 1 on 2026-08-27
    (`RasterioIOError: Read failed`) after ~1 hour of training — 4 failures out of roughly
    120,000 reads, i.e. a ~0.003% rate that is nonetheless fatal without a retry.

    Thread count is NOT the cause: 24 concurrent threads over 240 patches produced 0
    failures locally. Rasterio also swallows the underlying GDAL message, so the original
    error is re-raised with the file path attached to make the next occurrence diagnosable.
    """
    last = None
    for attempt in range(retries):
        try:
            with rasterio.open(path) as s:
                return np.transpose(s.read(), (1, 2, 0)).astype(np.float32)
        except Exception as e:            # noqa: BLE001 - rasterio raises several types
            last = e
            if attempt < retries - 1:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"failed to read {path} after {retries} attempts: "
                       f"{type(last).__name__}: {last}") from last


class FloodPatches:
    """Cohort-filtered patch dataset.

    cohort : split v0 — 'train' | 'val_seen' | 'val_unseen' | 'test_seen' | 'test_unseen'
             split v1 — 'train' | 'val' | 'test'
             or a list of those. Test cohorts require allow_test=True — a deliberate
             speed bump so an accidental read of the lockbox has to be explicit.
    regime : split v1 only. Filters on `basin_regime` ('seen' | 'partial' | 'unseen'), the
             LABEL that replaced v0's separate seen/unseen test cohorts. v1's RQ2 contrast
             is cohort='test' with regime='seen' against regime='unseen'; the headline test
             metric is cohort='test' with no regime filter, which keeps `partial` in.
    """

    def __init__(self, patches_csv, patches_dir, labels_dir, cohort,
                 flood_flags_csv=None, allow_test=False, regime=None, mono_only=False,
                 temporal_days=0):
        cohorts = [cohort] if isinstance(cohort, str) else list(cohort)
        if any(c.startswith('test') for c in cohorts) and not allow_test:
            raise PermissionError(
                f"cohort {cohorts} includes TEST. The final test cohorts are opened once, "
                f"after every selection decision is frozen. Pass allow_test=True only from "
                f"evaluate.py at that point.")

        # keep_default_na=False: v1 writes the literal string 'n/a' as train's basin_regime
        # and pandas' default na_values turns that into NaN (plus a mixed-dtype warning),
        # which would make a regime filter silently drop every train row.
        df = pd.read_csv(patches_csv, keep_default_na=False, na_values=[''],
                         low_memory=False)
        sel = df.cohort.isin(cohorts)
        regimes = None
        if regime is not None:
            regimes = [regime] if isinstance(regime, str) else list(regime)
            if 'basin_regime' not in df.columns:
                raise ValueError(f"{patches_csv} has no basin_regime column — regime="
                                 f"{regimes} needs a v1-style split")
            sel &= df.basin_regime.isin(regimes)
        # EXP 1 (terrain only) trains and scores on D_mono: ONE acquisition per site, the wettest.
        # A terrain-only model shown one input and graded against a site's several different masks
        # is being set an impossible task -- 57.3% of patches have a byte-identical twin.
        if mono_only:
            if 'exp1_keep' not in df.columns:
                raise ValueError(f"{patches_csv} has no exp1_keep column — mono_only needs the "
                                 f"current split")
            keep = df.exp1_keep
            if keep.dtype == object:                      # 'True'/'False' under keep_default_na
                keep = keep.astype(str).str.lower().isin(('true', '1', 'yes'))
            sel &= keep.astype(bool)
        self.mono_only = bool(mono_only)
        # 0 = Exp 1: input_2560m is never opened and the loader is byte-identical
        # to the static programme. >0 = Exp 2/3/4, right-aligned window of that length.
        self.temporal_days = int(temporal_days)
        self.df = df[sel].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"no patches for cohort {cohorts}"
                             + (f" regime {regimes}" if regimes else ""))
        self.patches_dir = Path(patches_dir)
        self.labels_dir = Path(labels_dir)
        self.patches_csv = patches_csv
        self.flood_flags_csv = flood_flags_csv
        self.cohorts = cohorts
        self.regimes = regimes
        self._flat = None          # all labels for this cohort, in RAM (see preload_labels)

        self.has_flood = None
        if flood_flags_csv and Path(flood_flags_csv).exists():
            fl = pd.read_csv(flood_flags_csv,
                             usecols=['folder_name', 'patch_number', 'has_flood80'])
            m = self.df[['folder_name', 'patch_number']].merge(
                fl, on=['folder_name', 'patch_number'], how='left')
            if int(m.has_flood80.isna().sum()):
                raise RuntimeError(
                    f"{int(m.has_flood80.isna().sum())} patches missing from "
                    f"{flood_flags_csv}; the flag file is stale for this split")
            self.has_flood = m.has_flood80.to_numpy().astype(bool)

    def __len__(self):
        return len(self.df)

    def preload_labels(self, verbose=True):
        """Load every label this cohort needs into RAM, once.

        THIS IS THE THROUGHPUT FIX (2026-08-27). Profiling a single patch:

            read input_10m   17.7 ms
            read input_80m    4.2 ms
            read input_160m   2.9 ms
            normalize         0.7 ms
            LABEL LOOKUP    329.4 ms     <-- 93% of the time

        The old per-event LRU held 64 entries and evicted in insertion order, but the
        class-balanced sampler hops between events on almost every patch, so nearly every
        single patch re-read and re-unpacked a whole ~428-patch .npz from disk. That alone
        put the U-Net at 4 s/step with the GPU idle at 0-4%.

        The entire label set is 127 MB on disk; as one int8 array it is ~0.6 GB. Holding it
        in RAM removes the dominant cost completely.
        """
        if self._flat is not None:
            return
        keys = list(zip(self.df.folder_name, self.df.patch_number))
        self._flat = np.zeros((len(keys), 32, 32, 1), np.int8)
        pos = {}
        for i, (fol, pn) in enumerate(keys):
            pos.setdefault(fol, []).append((i, str(int(pn))))
        missing = 0
        for fol, items in pos.items():
            p = self.labels_dir / f"{fol}.npz"
            if not p.exists():
                missing += len(items)
                continue
            with np.load(p) as z:
                for i, key in items:
                    if key not in z:
                        missing += 1
                        continue
                    a = z[key]                                  # (32,32,3) union/flood/pw
                    # UNION channel (0) with permanent water marked -1 — exactly what the
                    # tested train_v2.py._label80 does when pw_loss_weight < 1.0:
                    #   "keep the UNION target so the terrain scaffolding that makes training
                    #    stable is preserved, but mark permanent-water cells with -1 so every
                    #    loss term can drop them."
                    # On non-PW cells union == flood anyway (measured: 272 differing cells of
                    # 1.34M), so this changes no target — it matches the tested path.
                    self._flat[i, ..., 0] = np.where(a[..., 2] > 0,
                                                     np.int8(-1), a[..., 0].astype(np.int8))
        if verbose:
            print(f"  [labels] preloaded {len(keys):,} labels "
                  f"({self._flat.nbytes / 1e6:.0f} MB in RAM)"
                  + (f", {missing:,} missing" if missing else ""))
        # A confirmatory run must not silently train or score on fabricated DRY labels:
        # a missing label previously became all-zeros, which is indistinguishable from a
        # genuinely dry patch and quietly biases every metric. Fail loudly instead.
        if missing:
            raise RuntimeError(
                f"{missing:,} of {len(keys):,} labels missing for cohort {self.cohorts} — "
                f"they would silently become DRY. Rebuild {self.labels_dir} or fix the split.")

    def _label(self, i):
        """80 m label for row `i`, 32x32x1 float32. -1 marks permanent water."""
        if self._flat is None:
            self.preload_labels()
        return self._flat[i].astype(np.float32)

    # ---------------------------------------------------------------- RAM cache
    # OPTION C (Ravi, 2026-09-01): cache the FLOOD-BEARING patches decoded in RAM, and
    # nothing on disk -- /shared_data is shared with other users.
    #
    # Why the flood pool specifically: with flood_ratio 0.5 the sampler draws 16 flood +
    # 16 dry per batch, so at 3,793 steps it consumes 60,688 flood patches per epoch --
    # i.e. 100% of the 60,703-patch flood pool, EVERY epoch, for the whole run. The dry
    # pool is 364,320 and only 17% is touched per epoch, cycling over 7 epochs. So the
    # flood half is the part re-decoded ~100 times and the dry half is not.
    #
    # Cost 49 GB of 241 GB free. Removes ~50% of all LZW decode. BIT-IDENTICAL: the cache
    # stores exactly what load() returns, so a cache hit and a cache miss cannot differ.
    def enable_ram_cache(self, workers=None, verbose=True):
        import time
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import shared_memory
        if self.has_flood is None:
            raise ValueError("enable_ram_cache needs flood_flags_csv (which rows are flood)")
        self.preload_labels(verbose=False)
        rows = np.flatnonzero(self.has_flood)
        n = len(rows)
        shapes = {'fine': (256, 256, 2), 'med': (32, 32, 5),
                  'coarse': (16, 16, 2), 'lab': (32, 32, 1)}
        gb = n * sum(int(np.prod(v)) for v in shapes.values()) * 4 / 1e9
        workers = workers or min(os.cpu_count() or 8, 32)
        if verbose:
            print(f"  [ramcache] {n:,} flood patches -> {gb:.1f} GB, {workers} processes",
                  flush=True)
        self._cache_pos = {int(r): j for j, r in enumerate(rows)}
        self._shm, self._cache = {}, {}
        for k, shp in shapes.items():
            sh = shared_memory.SharedMemory(create=True, size=n * int(np.prod(shp)) * 4)
            self._shm[k] = sh
            self._cache[k] = np.ndarray((n,) + shp, dtype=np.float32, buffer=sh.buf)
        names = {k: v.name for k, v in self._shm.items()}
        t0 = time.perf_counter()
        chunks = np.array_split(np.arange(n), max(workers * 4, 1))
        args = (names, n, str(self.patches_csv), str(self.patches_dir),
                str(self.labels_dir), list(self.cohorts), str(self.flood_flags_csv), rows,
                self.regimes, self.mono_only)
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_rc_init, initargs=args) as ex:
            done = sum(ex.map(_rc_fill, chunks))
        el = time.perf_counter() - t0
        if verbose:
            print(f"  [ramcache] filled {done:,}/{n:,} in {el:.0f}s "
                  f"({el / max(n,1) * 1000:.2f} ms/patch)", flush=True)
        return n

    def close_ram_cache(self):
        for sh in getattr(self, '_shm', {}).values():
            try:
                sh.close(); sh.unlink()
            except FileNotFoundError:
                pass
        self._shm, self._cache, self._cache_pos = {}, {}, {}

    def load(self, i):
        j = getattr(self, '_cache_pos', {}).get(int(i))
        if j is not None:
            c = self._cache
            return (c['fine'][j].copy(), c['med'][j].copy(),
                    c['coarse'][j].copy(), c['lab'][j].copy())
        return self._load_uncached(i)

    def _load_uncached(self, i):
        r = self.df.iloc[i]
        stem = (self.patches_dir / r.folder_name.split('_')[0] / r.folder_name /
                f"patch_{int(r.patch_number):04d}")
        d10 = _read(f"{stem}_input_10m.tif")
        d80 = _read(f"{stem}_input_80m.tif")
        d160 = _read(f"{stem}_input_160m_olm.tif")

        d10[d10 >= NODATA_SENTINEL] = 0.0
        d80[d80 >= NODATA_SENTINEL] = 0.0
        for c, med in enumerate(SOIL_MEDIAN):          # neutral soil, not "very sandy"
            band = d160[..., c]
            band[band >= NODATA_SENTINEL] = med
        d10 = np.nan_to_num(d10); d80 = np.nan_to_num(d80); d160 = np.nan_to_num(d160)

        weather = self._weather(stem) if self.temporal_days else None
        o10, o80, o160 = normalize(d10, d80, d160)
        # 10 m bands are [VV, VH, NDVI, NDBI, PW]. Keep NDVI+NDBI: SAR is out of scope for
        # the static programme and PW is the prediction target, never an input.
        fine = o10[..., 2:4]                           # -> (256,256,2)
        out = (fine.astype(np.float32), o80.astype(np.float32), o160.astype(np.float32))
        if self.temporal_days:
            out = out + (weather,)
        return out + (self._label(i),)

    def _weather(self, stem) -> np.ndarray:
        """(T, 3) = [log1p(precip), soil moisture, valid], right-aligned on yesterday.

        The `valid` channel is not cosmetic: soil moisture carries the 9999 sentinel on days
        SMAP did not retrieve, and a model given 9999 as a value would learn that number as
        "very wet". Sentinel days are zeroed and flagged instead, so the encoder can tell a dry
        day from a missing one.
        """
        d = _read(f"{stem}_input_2560m.tif").reshape(-1).astype(np.float32)
        precip, sm = d[:WEATHER_DAYS], d[WEATHER_DAYS:2 * WEATHER_DAYS]
        valid = ((precip < NODATA_SENTINEL) & (sm < NODATA_SENTINEL)).astype(np.float32)
        precip = np.where(valid > 0, np.clip(precip, 0.0, PRECIP_CLIP), 0.0)
        sm = np.where(valid > 0, np.clip(sm, 0.0, 1.0), 0.0)
        w = np.stack([np.log1p(precip), sm, valid], axis=-1)      # (30, 3), oldest -> newest
        return w[-int(self.temporal_days):].astype(np.float32)    # right-aligned slice

    # ------------------------------------------------------------------ tf.data
    def to_tf_dataset(self, batch_size=32, shuffle=True, flood_ratio=0.0,
                      seed=42, prefetch=True, workers=8, epochs_of_order=110,
                      return_index=False, deterministic=None, drop_remainder=None):
        """Flat patch dataset. `flood_ratio > 0` enables class balancing (training only).

        The sample ORDER is materialised up front as an index array. For a balanced dataset
        that means interleaving the two pools so every consecutive `batch_size` slice holds
        the intended flood/dry mix — the batching is positional, so the order IS the balance.

        `epochs_of_order` epochs are materialised up front, each drawing a different slice
        of the dry pool, so the model eventually sees the whole training set rather than one
        fixed subset. Default 110 covers a 100-epoch budget with headroom.
        """
        # Training: speed first (non-deterministic, drop the ragged tail — the sampler
        # repeats anyway). Evaluation (flood_ratio == 0): exactness first.
        is_train = flood_ratio > 0
        if deterministic is None:
            deterministic = not is_train
        if drop_remainder is None:
            drop_remainder = is_train

        self.preload_labels()
        n = len(self.df)
        rng = np.random.default_rng(seed)

        if flood_ratio > 0:
            if self.has_flood is None:
                raise ValueError("flood_ratio > 0 needs flood_flags_csv")
            idx_f = np.flatnonzero(self.has_flood)
            idx_d = np.flatnonzero(~self.has_flood)
            n_f = max(1, int(round(batch_size * flood_ratio)))
            n_d = batch_size - n_f
            steps = min(len(idx_f) // n_f, len(idx_d) // max(n_d, 1))
            print(f"  [balance] batch {batch_size} = {n_f} flood + {n_d} dry | "
                  f"pools {len(idx_f):,} / {len(idx_d):,} | {steps:,} steps/epoch")
            # Build `epochs_of_order` epochs of order up front, each drawing a DIFFERENT
            # slice of the dry pool, then let tf.data repeat the whole thing.
            #
            # BUG FIXED 2026-08-27 (self-inflicted by the deadlock fix). The first version
            # materialised ONE epoch and `.repeat()`ed it, so every epoch saw the identical
            # 118,880 patches in the identical order and **308,567 dry patches (84% of the
            # dry pool) were never seen at all**. The old Python generator reshuffled both
            # pools each epoch; that property was silently lost when the generator was
            # replaced. Cycling the dry pool restores it: the flood pool recycles each epoch
            # (it is small) while the dry pool advances, so full coverage takes
            # ceil(len(idx_d) / (steps * n_d)) epochs.
            n_cycles = int(np.ceil(len(idx_d) / max(steps * n_d, 1)))
            n_epochs = max(int(epochs_of_order), 1)
            chunks = []
            # Dry pool as one stream of successive permutations: an epoch takes the next
            # steps*n_d dry rows, and the remainder of a permutation is consumed BEFORE the next
            # one starts (fixed 2026-10-04; the old code reshuffled and skipped the tail).
            dry_stream = rng.permutation(idx_d); pos = 0
            for ep in range(n_epochs):
                f = rng.permutation(idx_f)
                need = steps * n_d; parts = []
                while need:
                    take = dry_stream[pos:pos + need]; parts.append(take)
                    pos += len(take); need -= len(take)
                    if pos >= len(dry_stream):
                        dry_stream = rng.permutation(idx_d); pos = 0
                d = np.concatenate(parts)
                chunks.append(np.concatenate(
                    [np.concatenate([f[s * n_f:(s + 1) * n_f], d[s * n_d:(s + 1) * n_d]])
                     for s in range(steps)]))
            order_of_samples = np.concatenate(chunks)
            print(f"  [coverage] {n_epochs} epochs of order pre-built; dry pool needs "
                  f"{n_cycles} epochs for full coverage "
                  f"({100 * steps * n_d / len(idx_d):.0f}% of dry patches per epoch)")
            total_steps = steps
        else:
            order_of_samples = rng.permutation(n) if shuffle else np.arange(n)
            total_steps = n // batch_size

        # Build an INDEX dataset and load inside tf.numpy_function, rather than streaming
        # samples out of a Python generator.
        #
        # DEADLOCK (2026-08-27, Bolzano GPU): `from_generator` + `model.fit()` hung with all
        # 249 threads in futex_do_wait and zero CPU progress — the generator holds the GIL
        # while the training step compiles, and neither side can advance. Eager forward,
        # eager gradient and a compiled step all worked in isolation, so the model was never
        # at fault. `interleave` over an index dataset lets tf.data own the parallelism and
        # releases the GIL between reads.
        idx_ds = tf.data.Dataset.from_tensor_slices(np.asarray(order_of_samples, np.int64))

        T = self.temporal_days

        def _load_one(i):
            return self.load(int(i))                  # (f,m,c,y) or (f,m,c,w,y)

        def _wrap(i):
            types = [tf.float32] * (5 if T else 4)
            outs = tf.numpy_function(_load_one, [i], types, stateful=False)
            if T:
                f, m, c, w, y = outs
                w.set_shape(getattr(self, 'weather_shape', (T, 3)))   # final loader: (T,K,K,8)
            else:
                f, m, c, y = outs
                w = None
            f.set_shape((256, 256, 2)); m.set_shape((32, 32, 5))
            c.set_shape((16, 16, 2)); y.set_shape((32, 32, 1))
            # The weather tensor is the FOURTH input, appended — never inserted — so a static
            # checkpoint's (fine, medium, coarse) ordering is untouched.
            return ((f, m, c, w) if T else (f, m, c)), y

        def _wrap_id(i):
            x, y = _wrap(i)
            return x, y, i                  # the ROW INDEX travels with its own data

        # return_index: evaluation needs to know WHICH patch each prediction came from.
        # The pipeline is parallel and (for training) non-deterministic, so an external
        # array indexed by arrival order can attribute a prediction to the WRONG event.
        # Carrying the index through the graph makes attribution correct by construction.
        _fn = _wrap_id if return_index else _wrap
        ds = idx_ds.map(_fn, num_parallel_calls=int(workers),
                        deterministic=bool(deterministic))
        # drop_remainder=True silently discards the final partial batch — fine for training
        # (the sampler repeats), WRONG for evaluation, where it loses real held-out patches.
        ds = ds.batch(batch_size, drop_remainder=bool(drop_remainder))
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        # Training repeats so `steps_per_epoch` can exceed one pass; validation must NOT,
        # or model.fit() would never finish a validation round.
        if flood_ratio > 0:
            ds = ds.repeat()
        ds.steps_per_epoch = total_steps
        return ds


# ------------------------------------------------------------------ RAM-cache workers
_RC = {}


def _rc_init(names, n, csv_path, pdir, ldir, cohorts, flags, rows, regimes=None, mono=False):
    """Worker: its own loader + a view onto the shared blocks. Fork-safe.

    `regimes` must be passed through: the worker re-filters the CSV itself, and a different
    filter would give it a different row order, so `rows` would index the wrong patches.
    """
    from multiprocessing import shared_memory
    ds = FloodPatches(patches_csv=csv_path, patches_dir=pdir, labels_dir=ldir,
                      cohort=cohorts, flood_flags_csv=flags, regime=regimes, mono_only=mono)
    ds.preload_labels(verbose=False)
    _RC['ds'] = ds
    _RC['rows'] = rows
    shapes = {'fine': (256, 256, 2), 'med': (32, 32, 5),
              'coarse': (16, 16, 2), 'lab': (32, 32, 1)}
    _RC['shm'], _RC['arr'] = {}, {}
    for k, nm in names.items():
        sh = shared_memory.SharedMemory(name=nm)
        _RC['shm'][k] = sh
        _RC['arr'][k] = np.ndarray((n,) + shapes[k], dtype=np.float32, buffer=sh.buf)


def _rc_fill(chunk):
    ds, rows, a = _RC['ds'], _RC['rows'], _RC['arr']
    for j in chunk:
        fine, med, coarse, lab = ds._load_uncached(int(rows[j]))
        a['fine'][j] = fine; a['med'][j] = med
        a['coarse'][j] = coarse; a['lab'][j] = lab
    return len(chunk)
