"""Shared data access for the Chapter 5 figures (FINAL programme, experiments/runs_final).

Every figure reads the saved evaluation outputs of each fit, never a training-status file:
  inference_final/patches.csv     per-patch TP, FP, FN and scored cells at the fit's validation-selected
                                  global threshold (all test rows)
  inference_final/preds/test__*   probabilities, labels and the row index (ground unit, regime, ...)
  inference_final/metrics.csv     the evaluator's scores; every recomputed score is checked against it

Units (Chapter 4, Section 4.3.1): mapping product = folder_name; ground unit = ground_unit; dependence
cluster = connected component of activations (EMSR code) and ground units, 61 in the test subset.
Row set: 'full' (all acquisitions) or 'single' (one acquisition per AOI, exp1_keep). Decided by Ravi on
10 October 2026: full rows primary (OPEN 19); cluster bootstrap with 4,000 replicates and 95 % percentile
intervals (OPEN 15).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')
FAMILIES = ('unet', 'vit')
FAMILY_LABEL = {'unet': 'U-Net', 'vit': 'ViT'}
SEEDS = (1, 2, 3)
N_BOOT = 4000
BOOT_SEED = 20261010
KEY = ['folder_name', 'patch_number']

# Colours (validated with the dataviz palette check, 10 October 2026): families from the data-paper palette
# (blue vs orange: protan/deutan dE 34.9), regimes green vs purple (deutan dE 13.8).
FAMILY_COLOR = {'unet': '#E0942A', 'vit': '#0257CE'}
REGIME_COLOR = {'seen': '#009E73', 'unseen': '#A457BF'}
REGIME_LABEL = {'seen': 'basin-seen', 'unseen': 'basin-unseen'}


def run_dir(family: str, window: str = 't0', seed: int = 1, variant: str | None = None) -> Path:
    """Run directory of a fit, as scripts/final/train.py names it."""
    if window == 't0':
        d = RUNS / f'final_t0/seed{seed}/{family}'
    elif window == 't30' and variant is None:
        d = RUNS / f'final_t30/seed{seed}/{family}'
    elif window == 't30' and variant == 'scratch':
        d = RUNS / f'final_scratch/seed{seed}/{family}'
    elif window == 't30':
        d = RUNS / f'final_ablation/seed{seed}/{family}_{variant}'
    else:
        d = RUNS / f'final_window/seed{seed}/{family}_{window}'
    for f in ('COMPLETE', 'run_manifest.json', 'inference_final/metrics.csv', 'inference_final/patches.csv'):
        if not (d / f).exists():
            raise FileNotFoundError(f'{d}: no {f} (fit not finished or not copied)')
    return d


class TestSet:
    """Row index, units and dependence clusters of the test subset (one reference, checked for every fit)."""

    def __init__(self):
        ix = pd.read_csv(run_dir('unet', 't0', 1) / 'inference_final/preds/test__index.csv')
        # Ravi, 20 Oct 2026: no partial regime; a test block with at least one Level-5 basin in training is basin-seen
        ix['basin_regime'] = ix.basin_regime.replace({'partial': 'seen'})
        self.index = ix
        self.n = len(ix)
        self.single = ix.exp1_keep.astype(str).str.lower().isin(['true', '1']).to_numpy()
        act = ix.folder_name.str.split('_').str[0].to_numpy()
        gu = ix.ground_unit.to_numpy()
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, g in sorted(set(zip(act, gu))):
            parent[find('A' + a)] = find('G' + g)
        # canonical cluster label = the smallest member name of the component, so the numbering of the clusters
        # (and with it every bootstrap replicate) is the same in every process; a set's order depends on the
        # per-process string hash seed (fixed 10 Oct 2026, after intervals differed slightly between runs)
        label = {}
        for node in parent:
            r = find(node)
            label[r] = min(label.get(r, node), node)
        roots = np.array([label[find('A' + a)] for a in act])
        _, self.cluster = np.unique(roots, return_inverse=True)
        self.n_clusters = int(self.cluster.max() + 1)
        self.products, self.product = np.unique(ix.folder_name.to_numpy(), return_inverse=True)
        self.units, self.unit = np.unique(gu, return_inverse=True)
        self.regime = ix.basin_regime.to_numpy()
        # each product and each ground unit lies in exactly one cluster
        self.product_cluster = self._owner(self.product, len(self.products))
        self.unit_cluster = self._owner(self.unit, len(self.units))
        unit_regime = pd.Series(self.regime).groupby(self.unit).agg(lambda s: s.iloc[0] if s.nunique() == 1 else 'mixed')
        self.unit_regime = unit_regime.reindex(range(len(self.units))).to_numpy()

    def _owner(self, member, n):
        own = np.full(n, -1)
        for m, c in zip(member, self.cluster):
            if own[m] == -1:
                own[m] = c
            elif own[m] != c:
                raise ValueError('a unit spans two dependence clusters')
        return own

    def rows(self, rows: str = 'full') -> np.ndarray:
        return self.single if rows == 'single' else np.ones(self.n, bool)


@lru_cache(maxsize=None)
def test_set() -> TestSet:
    return TestSet()


@lru_cache(maxsize=None)
def counts(family: str, window: str = 't0', seed: int = 1, variant: str | None = None) -> pd.DataFrame:
    """Per-patch TP, FP, FN, scored cells of one fit on the test subset, in the reference row order."""
    ts = test_set()
    d = run_dir(family, window, seed, variant)
    p = pd.read_csv(d / 'inference_final/patches.csv')
    p = p[p.cohort == 'test'].sort_values('row_index').reset_index(drop=True)
    if len(p) != ts.n or not (p[KEY].to_numpy() == ts.index[KEY].to_numpy()).all():
        raise ValueError(f'{d}: test rows differ from the reference index')
    return p[['tp', 'fp', 'fn', 'scored_cells']]


def _iou(t, f, n):
    d = t + f + n
    return np.divide(t, d, out=np.full(np.shape(d), np.nan, float), where=d > 0)


def scores(c: pd.DataFrame, rows: str = 'full') -> dict:
    """Pooled IoU, precision, recall; mean IoU over flood-positive products and ground units."""
    ts = test_set()
    m = ts.rows(rows)
    t, f, n = (c[k].to_numpy(float) * m for k in ('tp', 'fp', 'fn'))
    out = {'pooled_iou': t.sum() / (t + f + n).sum(), 'pooled_precision': t.sum() / (t + f).sum(),
           'pooled_recall': t.sum() / (t + n).sum()}
    for name, idx, k in (('product', ts.product, len(ts.products)), ('unit', ts.unit, len(ts.units))):
        T, F, N = (np.bincount(idx, v, minlength=k) for v in (t, f, n))
        pos = (T + N) > 0
        out[f'{name}_mean_iou'] = float(_iou(T, F, N)[pos].mean())
    return out


def check_against_metrics(family, window='t0', seed=1, variant=None, rows='full', tol=1e-9):
    """Recompute pooled / product-mean / ground-unit-mean IoU and compare with the evaluator's metrics.csv."""
    d = run_dir(family, window, seed, variant)
    m = pd.read_csv(d / 'inference_final/metrics.csv')
    m = m[(m.cohort == 'test') & (m.rows == rows) & (m.rule == 'global')].iloc[0]
    s = scores(counts(family, window, seed, variant), rows)
    for a, b in (('pooled_iou', 'pooled_iou'), ('product_mean_iou', 'product_mean_iou'),
                 ('unit_mean_iou', 'ground_unit_mean_iou')):
        if abs(s[a] - m[b]) > tol:
            raise ValueError(f'{d}: {a} {s[a]:.6f} differs from metrics.csv {m[b]:.6f}')
    return s


def unit_values(c: pd.DataFrame, rows: str = 'full'):
    """Per ground unit: IoU, flood cells, scored cells (all units; NaN IoU where no flood and no prediction)."""
    ts = test_set()
    m = ts.rows(rows)
    t, f, n, k = (c[x].to_numpy(float) * m for x in ('tp', 'fp', 'fn', 'scored_cells'))
    T, F, N, K = (np.bincount(ts.unit, v, minlength=len(ts.units)) for v in (t, f, n, k))
    return pd.DataFrame({'iou': _iou(T, F, N), 'flood_cells': T + N, 'scored_cells': K,
                         'regime': ts.unit_regime, 'cluster': ts.unit_cluster})


def product_values(c: pd.DataFrame, rows: str = 'full'):
    ts = test_set()
    m = ts.rows(rows)
    t, f, n, k = (c[x].to_numpy(float) * m for x in ('tp', 'fp', 'fn', 'scored_cells'))
    T, F, N, K = (np.bincount(ts.product, v, minlength=len(ts.products)) for v in (t, f, n, k))
    return pd.DataFrame({'iou': _iou(T, F, N), 'flood_cells': T + N, 'scored_cells': K,
                         'precision': np.divide(T, T + F, out=np.full(len(T), np.nan), where=(T + F) > 0),
                         'cluster': ts.product_cluster}, index=ts.products)


# ------------------------------------------------------------------ cluster bootstrap (OPEN 15 working choice)
@lru_cache(maxsize=None)
def boot_weights(n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n_clusters) multiplicities: whole dependence clusters drawn with replacement, the same draw
    for every network, so paired contrasts are computed on the same resampled test sets."""
    ts = test_set()
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, ts.n_clusters, ts.n_clusters), minlength=ts.n_clusters)
                     for _ in range(n_boot)]).astype(float)


def boot_pooled(c: pd.DataFrame, rows: str = 'full') -> dict:
    """Bootstrap replicates of pooled IoU, precision and recall."""
    ts = test_set()
    W = boot_weights()
    m = ts.rows(rows)
    t, f, n = (np.bincount(ts.cluster, c[x].to_numpy(float) * m, minlength=ts.n_clusters) for x in ('tp', 'fp', 'fn'))
    T, F, N = W @ t, W @ f, W @ n
    return {'pooled_iou': T / (T + F + N), 'pooled_precision': T / (T + F), 'pooled_recall': T / (T + N)}


def boot_mean(values: np.ndarray, owner_cluster: np.ndarray) -> np.ndarray:
    """Bootstrap replicates of the mean of per-unit (or per-product) values; NaN values are left out."""
    ts = test_set()
    W = boot_weights()
    ok = ~np.isnan(values)
    s = np.bincount(owner_cluster[ok], values[ok], minlength=ts.n_clusters)
    k = np.bincount(owner_cluster[ok], minlength=ts.n_clusters).astype(float)
    return (W @ s) / (W @ k)


def interval(reps: np.ndarray, level: float = 0.95):
    lo, hi = np.nanpercentile(reps, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


# ------------------------------------------------------------------ within-product AP from the saved probabilities
def _pr_auc(p: np.ndarray, y: np.ndarray, num_thresholds: int = 200) -> float:
    """Keras AUC(curve='PR', summation_method='interpolation') as the evaluator uses it (Davis & Goadrich
    interpolation over num_thresholds thresholds, a cell is positive when p > threshold)."""
    eps = 1e-7
    thr = np.array([0.0 - eps] + [(i + 1) / (num_thresholds - 1) for i in range(num_thresholds - 2)] + [1.0 + eps],
                   np.float32)
    pos = np.sort(p[y])
    neg = np.sort(p[~y])
    tp = (len(pos) - np.searchsorted(pos, thr, side='right')).astype(float)
    fp = (len(neg) - np.searchsorted(neg, thr, side='right')).astype(float)
    fn = len(pos) - tp
    dtp = tp[:-1] - tp[1:]
    P = tp + fp
    dp = P[:-1] - P[1:]
    slope = np.divide(dtp, np.maximum(dp, 0), out=np.zeros_like(dtp), where=np.maximum(dp, 0) != 0)
    intercept = tp[1:] - slope * P[1:]
    ratio = np.ones_like(dtp)
    ok = (P[:-1] > 0) & (P[1:] > 0)
    ratio[ok] = P[:-1][ok] / np.maximum(P[1:][ok], 0)
    num = slope * (dtp + intercept * np.log(ratio))
    den = np.maximum(tp[1:] + fn[1:], 0)
    inc = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return float(inc.sum())


@lru_cache(maxsize=None)
def product_ap(family: str, window: str = 't0', seed: int = 1, variant: str | None = None,
               rows: str = 'full') -> pd.Series:
    """Within-product AP (200 thresholds, flood-only cells) of every flood-positive product."""
    ts = test_set()
    d = run_dir(family, window, seed, variant) / 'inference_final/preds'
    ix = pd.read_csv(d / 'test__index.csv')
    if not (ix[KEY].to_numpy() == ts.index[KEY].to_numpy()).all():
        raise ValueError(f'{d}: prediction rows differ from the reference index')
    P = np.load(d / 'test__prob.f16.npy').astype(np.float32)
    Y = np.load(d / 'test__labels.i8.npy')
    m = ts.rows(rows)
    out = {}
    order = np.argsort(ts.product, kind='stable')
    bounds = np.searchsorted(ts.product[order], np.arange(len(ts.products) + 1))
    for j, name in enumerate(ts.products):
        r = order[bounds[j]:bounds[j + 1]]
        r = r[m[r]]
        if not len(r):
            continue
        y = Y[r]
        keep = y >= 0
        yy = (y[keep] == 1)
        if yy.any():
            out[name] = _pr_auc(P[r][keep], yy)
    return pd.Series(out, name='ap')


def check_ap_against_metrics(family, window='t0', seed=1, variant=None, rows='full', tol=2e-4):
    d = run_dir(family, window, seed, variant)
    m = pd.read_csv(d / 'inference_final/metrics.csv')
    m = m[(m.cohort == 'test') & (m.rows == rows) & (m.rule == 'global')].iloc[0]
    ap = product_ap(family, window, seed, variant, rows).mean()
    if abs(ap - m.product_mean_auprc) > tol:
        raise ValueError(f'{d}: mean within-product AP {ap:.6f} differs from metrics.csv {m.product_mean_auprc:.6f}')
    return ap
