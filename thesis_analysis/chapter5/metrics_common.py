"""IoU, POD and POFD of the Chapter 5 figures (Ravi, 17 and 19 Oct 2026: the thesis reports these three evaluation
metrics; F1, AUPRC, false alarm ratio, miss rate and area bias only in the appendix table of seed means).

  POD   probability of detection = TP / (TP + FN)
  POFD  probability of false detection = FP / (FP + TN) = FP / dry scored cells

Fit values: data/metrics/fits.csv (ch5_pod_pofd_data.py). Dry cells of a product: its scored cells minus its flood cells
of the label, from data/conditions/patches_test.csv.gz.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / 'data'


@lru_cache(maxsize=None)
def fits():
    """One row per fit: family, config (t0, t3, t7, t15, t30, noup, nosum, sumonly, scratch), seed, micro and macro
    IoU, POD and POFD, and the other metrics of the appendix table."""
    return pd.read_csv(DATA / 'metrics' / 'fits.csv')


def fit_table():
    """fits() indexed by (family, config, seed), the index of rq4_common.fits() with 'config' for 'window'."""
    return fits().set_index(['family', 'config', 'seed'])


@lru_cache(maxsize=None)
def product_cells():
    """Scored and flood cells of every test mapping product."""
    pt = pd.read_csv(DATA / 'conditions' / 'patches_test.csv.gz', usecols=['folder_name', 'scored', 'flood'])
    return pt.groupby('folder_name')[['scored', 'flood']].sum()


DRY_TOTAL = None


def dry_total():
    s = product_cells()
    return int(s.scored.sum() - s.flood.sum())


def add_pod_pofd(c):
    """Adds IoU, POD and POFD to a table of per-product counts (folder_name, tp, fp, fn)."""
    s = product_cells()
    c = c.copy()
    dry = c.folder_name.map(s.scored - s.flood)
    c['iou'] = c.tp / (c.tp + c.fp + c.fn)
    c['pod'] = c.tp / (c.tp + c.fn)
    c['pofd'] = c.fp / dry
    return c


def macro(f, config, s, m):
    return float(fit_table().loc[(f, config, s), f'macro_{m}'])


def micro(f, config, s, m):
    return float(fit_table().loc[(f, config, s), f'micro_{m}'])


def three_panels(letters=True):
    """IoU, POD and POFD panels for rq4_common.length_bars: value(c, f, w, s), c = 0 micro, 1 macro."""
    out = []
    for i, (m, name) in enumerate((('iou', 'IoU'), ('pod', 'POD'), ('pofd', 'POFD'))):
        out.append((f'({"abc"[i]}) {name}' if letters else name, '{:.3f}',
                    (lambda m: lambda c, f, w, s: micro(f, w, s, m) if c == 0 else macro(f, w, s, m))(m)))
    return tuple(out)
