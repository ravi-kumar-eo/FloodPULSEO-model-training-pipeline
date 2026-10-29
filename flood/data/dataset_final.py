"""Loader of the final experiment programme (METHOD_PLAN_RERUN.md, version 3).

Subclass of `FloodPatches`, so the old loader, its runs and its evaluators stay byte-identical.
What differs from the base class:

LABELS (D4)   `labels_80m_cls/<product>.npz`, one (32, 32) uint8 class map per patch:
              0 dry, 1 flood, 2 permanent water, 3 mixed (flood + water together >= 32/64).
              Training target = union (classes 1, 2, 3) with permanent water marked -1, which
              the unchanged losses weight 0.2 and the evaluator excludes. Class 3 follows
              `mixed_as`: 'pw' (default, -1 like permanent water), 'flood' (1) or 'dry' (0).
SOIL (D7)     `input_160m.tif` (SoilGrids v2.0); NoData -> training medians from the transforms.
TRANSFORMS    from `input_transforms.json` (C2): no clipping of valid values (D8):
              NDVI, NDBI (x+1)/2 | elevation (x-min)/(max-min) | flow (x+1)/2 | UDA log10(1+x)/7 |
              HAND x/300 | clay, sand %/100.
WEATHER (D9)  AOI weather: for every patch, a K x K window of native IMERG and SMAP cells
              centred on the patch, last T days to day -1. Channels per variable: value,
              validity, east-west and north-south offset of the cell centre from the patch
              centre (km / 100, local equirectangular approximation, < 1% error within 100 km).
              Weather tensor (T, K, K, 8): [P, P_valid, P_dx, P_dy, SM, SM_valid, SM_dx, SM_dy].
              Invalid cells carry value 0 and validity 0.
UPSTREAM      weather_mode 'upstream' (Ravi, 2026-10-07): per patch and day, from
              metadata/final/upstream/weather/<product>.npz (build/build_upstream_weather.py):
              (T, 5) = [local rain / precip_div, local soil moisture, valid, upstream rain /
              precip_div, log10(1 + upstream area km2) / uda_div]. Upstream rain = mean daily rain
              over all land draining into the patch (MERIT Hydro IHU catchment); held in RAM.
CACHE         `enable_eval_cache()` decodes the rows' images and labels once into shared memory
              (validation); weather windows are cut from the in-RAM product grids per patch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import FloodPatches, _read, NODATA_SENTINEL

MIXED = {'pw': -1, 'flood': 1, 'dry': 0}


def load_transforms(path) -> dict:
    t = json.loads(Path(path).read_text())
    for k in ('elev_min', 'elev_max', 'clay_median', 'sand_median', 'precip_div', 'uda_div', 'hand_div'):
        if k not in t:
            raise KeyError(f'{path} lacks {k}')
    if not t['elev_max'] > t['elev_min']:
        raise ValueError('elevation range is empty')
    return t


def transform_inputs(d10, d80, d160, t):
    """Physical units -> network values. d10 bands: VV, VH, NDVI, NDBI, PW (PW and S1 unused)."""
    def bad(a):
        # EXACT sentinel: UDA is in km2 and large rivers carry real values above 9999
        # (0.19% of training cells), so a ">= 9999" test would zero the biggest rivers.
        return ~np.isfinite(a) | (a == NODATA_SENTINEL)

    ndvi, ndbi = d10[..., 2].copy(), d10[..., 3].copy()
    for a in (ndvi, ndbi):
        a[bad(a) | (a < -1.0) | (a > 1.0)] = 0.0              # missing -> physical 0
    fine = np.stack([(ndvi + 1.0) / 2.0, (ndbi + 1.0) / 2.0], axis=-1)

    e, s, c, u, h = (d80[..., i].copy() for i in range(5))
    e[bad(e)] = 0.0
    s[bad(s)] = 0.0; c[bad(c)] = 0.0                           # missing direction -> (0,0)
    u[bad(u) | (u < 0)] = 0.0
    h[bad(h) | (h < 0)] = 0.0
    med = np.stack([(e - t['elev_min']) / (t['elev_max'] - t['elev_min']),
                    (s + 1.0) / 2.0, (c + 1.0) / 2.0,
                    np.log10(1.0 + u) / t['uda_div'],
                    h / t['hand_div']], axis=-1)

    clay, sand = d160[..., 0].copy(), d160[..., 1].copy()
    clay[bad(clay) | (clay < 0) | (clay > 100)] = t['clay_median']
    sand[bad(sand) | (sand < 0) | (sand > 100)] = t['sand_median']
    coarse = np.stack([clay / 100.0, sand / 100.0], axis=-1)
    return fine.astype(np.float32), med.astype(np.float32), coarse.astype(np.float32)


class AOIWeather:
    """In-RAM native weather grids of the products, and per-patch K x K window cutting."""

    def __init__(self, aoi_dir, products, k=11):
        self.k = int(k)
        self.g = {}
        for p in sorted(set(products)):
            f = Path(aoi_dir) / f'{p}.npz'
            if not f.exists():
                raise FileNotFoundError(f'AOI weather missing for {p}: {f}')
            z = np.load(f)
            self.g[p] = {v: (z[v].astype(np.float32), z[f'{v}_transform']) for v in ('precip', 'sm')}

    def window(self, product, lon, lat, T, precip_div):
        """(T, K, K, 8) for the last T days, centred on (lon, lat)."""
        out = np.zeros((T, self.k, self.k, 8), np.float32)
        r = self.k // 2
        coslat = math.cos(math.radians(lat))
        for vi, var in enumerate(('precip', 'sm')):
            arr, tr = self.g[product][var]                     # (30, H, W), (6,) affine
            dx, x0, dy, y0 = tr[0], tr[2], tr[4], tr[5]       # dy < 0 (north-up)
            col = int(math.floor((lon - x0) / dx)); row = int(math.floor((lat - y0) / dy))
            H, W = arr.shape[1:]
            rr = np.arange(row - r, row + r + 1); cc = np.arange(col - r, col + r + 1)
            inside = ((rr[:, None] >= 0) & (rr[:, None] < H) & (cc[None, :] >= 0) & (cc[None, :] < W))
            vals = np.zeros((30, self.k, self.k), np.float32)
            ri, ci = np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)
            vals[:] = arr[:, ri][:, :, ci]
            valid = np.isfinite(vals) & inside[None]
            if var == 'precip':
                valid &= vals >= 0
                v = np.where(valid, vals / precip_div, 0.0)
            else:
                valid &= (vals >= 0) & (vals <= 1)
                v = np.where(valid, vals, 0.0)
            lon_c = x0 + (cc + 0.5) * dx; lat_c = y0 + (rr + 0.5) * dy
            ew = (lon_c[None, :] - lon) * 111.32 * coslat / 100.0
            ns = (lat_c[:, None] - lat) * 110.57 / 100.0
            b = vi * 4
            out[..., b] = v[-T:]
            out[..., b + 1] = valid[-T:]
            out[..., b + 2] = np.broadcast_to(ew, (self.k, self.k))
            out[..., b + 3] = np.broadcast_to(ns, (self.k, self.k))
        return out


class FinalFloodPatches(FloodPatches):
    """FloodPatches for the final programme (see module docstring)."""

    def __init__(self, *args, transforms=None, mixed_as='pw', soil_file='input_160m.tif',
                 aoi_weather_dir=None, aoi_k=11, weather_mode='aoi', upstream_dir=None, **kw):
        super().__init__(*args, **kw)
        if transforms is None:
            raise ValueError('FinalFloodPatches needs the transforms dict (input_transforms.json)')
        if mixed_as not in MIXED:
            raise ValueError(f'mixed_as must be one of {sorted(MIXED)}')
        self.t = transforms
        self.mixed_as = mixed_as
        self.soil_file = soil_file
        self.aoi = None
        self.weather_mode = str(weather_mode)
        if self.temporal_days and self.weather_mode == 'upstream':
            if upstream_dir is None:
                raise ValueError("weather_mode 'upstream' needs upstream_dir")
            self._load_upstream(Path(upstream_dir))
            self.weather_shape = (self.temporal_days, 5)
        elif self.temporal_days:
            if aoi_weather_dir is None:
                raise ValueError('weather fits need aoi_weather_dir')
            self.aoi = AOIWeather(aoi_weather_dir, self.df.folder_name.unique(), aoi_k)
            self.weather_shape = (self.temporal_days, int(aoi_k), int(aoi_k), 8)
            self._centres()

    def _load_upstream(self, d: Path):
        """(N, 30, 5) network-ready daily weather of every row, from the per-product files."""
        n = len(self.df)
        wx = np.zeros((n, 30, 5), np.float32)
        found = np.zeros(n, bool)
        pdiv, udiv = float(self.t['precip_div']), float(self.t['uda_div'])
        for fol, idx in self.df.groupby('folder_name').indices.items():
            f = d / f'{fol}.npz'
            if not f.exists():
                raise FileNotFoundError(f'upstream weather missing for {fol}: {f}')
            with np.load(f) as z:
                pos = {int(p): j for j, p in enumerate(z['patch_number'])}
                rows = [pos[int(p)] for p in self.df.patch_number.to_numpy()[idx]]
                lp, ls = z['local_precip'][rows], z['local_sm'][rows]
                lv = z['local_valid'][rows].astype(bool)
                up = z['upstream_precip'][rows]
                ar = z['upstream_area_km2'][rows]
            lp = np.where(np.isfinite(lp) & (lp >= 0), lp, 0.0)
            ls = np.where(np.isfinite(ls) & (ls >= 0) & (ls <= 1), ls, 0.0)
            up = np.where(np.isfinite(up) & (up >= 0), up, 0.0)
            wx[idx, :, 0] = lp / pdiv
            wx[idx, :, 1] = ls
            wx[idx, :, 2] = lv
            wx[idx, :, 3] = up / pdiv
            wx[idx, :, 4] = (np.log10(1.0 + np.maximum(ar, 0.0)) / udiv)[:, None]
            found[idx] = True
        if not found.all():
            raise RuntimeError(f'{int((~found).sum())} rows without upstream weather')
        self._wx = wx

    def _centres(self):
        from pyproj import Transformer
        lon = np.empty(len(self.df)); lat = np.empty(len(self.df))
        cx = (self.df.bounds_minx.to_numpy() + self.df.bounds_maxx.to_numpy()) / 2
        cy = (self.df.bounds_miny.to_numpy() + self.df.bounds_maxy.to_numpy()) / 2
        for crs, idx in self.df.groupby('crs').indices.items():
            tr = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
            lon[idx], lat[idx] = tr.transform(cx[idx], cy[idx])
        self._lon, self._lat = lon, lat

    # ------------------------------------------------------------------ labels
    def preload_labels(self, verbose=True):
        if self._flat is not None:
            return
        keys = list(zip(self.df.folder_name, self.df.patch_number))
        self._flat = np.zeros((len(keys), 32, 32, 1), np.int8)
        lut = np.array([0, 1, -1, MIXED[self.mixed_as]], np.int8)   # class -> target
        pos = {}
        for i, (fol, pn) in enumerate(keys):
            pos.setdefault(fol, []).append((i, str(int(pn))))
        missing = 0
        for fol, items in pos.items():
            p = self.labels_dir / f'{fol}.npz'
            if not p.exists():
                missing += len(items); continue
            with np.load(p) as z:
                for i, key in items:
                    if key not in z:
                        missing += 1; continue
                    self._flat[i, ..., 0] = lut[z[key]]
        if verbose:
            print(f'  [labels] {len(keys):,} class maps (mixed_as={self.mixed_as})')
        if missing:
            raise RuntimeError(f'{missing:,} labels missing in {self.labels_dir}')

    # ------------------------------------------------------------------ inputs
    def _images(self, i):
        r = self.df.iloc[i]
        stem = (self.patches_dir / r.folder_name.split('_')[0] / r.folder_name /
                f'patch_{int(r.patch_number):04d}')
        return transform_inputs(_read(f'{stem}_input_10m.tif'), _read(f'{stem}_input_80m.tif'),
                                _read(f'{stem}_{self.soil_file}'), self.t)

    def _weather_of(self, i):
        if self.weather_mode == 'upstream':
            return self._wx[i, -self.temporal_days:].copy()
        return self.aoi.window(self.df.folder_name.iat[i], self._lon[i], self._lat[i],
                               self.temporal_days, self.t['precip_div'])

    def _load_uncached(self, i):
        fine, med, coarse = self._images(i)
        out = (fine, med, coarse)
        if self.temporal_days:
            out = out + (self._weather_of(i),)
        return out + (self._label(i),)

    def load(self, i):
        j = getattr(self, '_cache_pos', {}).get(int(i))
        if j is None:
            return self._load_uncached(i)
        c = self._cache
        out = (c['fine'][j].copy(), c['med'][j].copy(), c['coarse'][j].copy())
        if self.temporal_days:
            out = out + (self._weather_of(i),)
        return out + (c['lab'][j].copy(),)

    # ------------------------------------------------------------------ eval cache
    ROW_BYTES = (256 * 256 * 2 + 32 * 32 * 5 + 16 * 16 * 2 + 32 * 32 * 1) * 4     # 550,912 B per row

    def enable_eval_cache(self, workers=24, verbose=True, rows=None, tag='evalcache'):
        """Decode the images and labels of `rows` (default: every row) once into shared memory.

        Validation decodes every row. Training (2026-10-05) decodes the rows of `train_cache_rows`:
        a decoded row is then a memory copy instead of three GeoTIFF reads, which the tf.data
        threads cannot parallelise beyond ~2 cores (Python GIL; measured 92% of 32 vCPUs idle,
        GPU ~58% busy). Values are bit-identical to the disk path (same `_images`, `_label`).
        """
        import time
        from concurrent.futures import ThreadPoolExecutor
        from multiprocessing import shared_memory
        self.preload_labels(verbose=False)
        rows = np.arange(len(self.df)) if rows is None else np.asarray(rows, dtype=np.int64)
        n = len(rows)
        shapes = {'fine': (256, 256, 2), 'med': (32, 32, 5), 'coarse': (16, 16, 2), 'lab': (32, 32, 1)}
        gb = n * self.ROW_BYTES / 1e9
        self._shm, self._cache = {}, {}
        for k, shp in shapes.items():
            sh = shared_memory.SharedMemory(create=True, size=max(n, 1) * int(np.prod(shp)) * 4)
            self._shm[k] = sh
            self._cache[k] = np.ndarray((max(n, 1),) + shp, dtype=np.float32, buffer=sh.buf)
        t0 = time.perf_counter()

        def fill(j):
            i = int(rows[j])
            f, m, c = self._images(i)
            self._cache['fine'][j] = f; self._cache['med'][j] = m
            self._cache['coarse'][j] = c; self._cache['lab'][j] = self._label(i)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(fill, range(n), chunksize=256))
        self._cache_pos = {int(i): j for j, i in enumerate(rows)}
        if verbose:
            print(f'  [{tag}] {n:,} of {len(self.df):,} rows, {gb:.1f} GB, {time.perf_counter() - t0:.0f}s',
                  flush=True)
        return n

    def train_cache_rows(self, budget_gb, seed=0):
        """Rows to decode into RAM for training within `budget_gb`: every flood-bearing row first
        (the sampler reads the whole flood pool every epoch), then dry rows in a fixed random order."""
        cap = int(budget_gb * 1e9 // self.ROW_BYTES)
        f = np.flatnonzero(self.has_flood)
        d = np.random.default_rng(seed).permutation(np.flatnonzero(~self.has_flood))
        return np.concatenate([f, d])[:cap]
