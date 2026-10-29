"""Source data of the RQ4 timing analysis (Section 5.4): scores of every input-history length within classes of the
timing of the antecedent precipitation of the test patches.

Timing class of a test patch: the shortest input history (3, 7, 15 or 30 days) whose days hold at least half of the
precipitation of the 30 days before the mapping date, from the daily inputs that the networks receive
(experiments/metadata/final/upstream/weather/<product>.npz; the last column is day -1). Class T3: half of the 30-day
precipitation fell on days -3 to -1; T7: not within 3 days, but within 7; T15: within 15; T30: more than half fell
on days -30 to -16. Patches with less than 1 mm in 30 days form the class 'dry'. Computed for the upstream
precipitation (main) and the local precipitation (robustness).

Within a class, TP, FP and FN are summed over the scored cells of its patches for every fit (T0, T3, T7, T15, T30;
U-Net and ViT; seeds 1-3), and the measures are averaged over the seeds. Changes against T0 and against T30 are paired
by seed and carry 95% cluster bootstrap intervals (61 clusters, 4,000 replicates, ch5_data.boot_weights; replicates
that draw none of a class's clusters are left out of that class).

Writes chapters/chapter5/plot/data/rq4/timing.csv, one row per source (upstream, local), class, family and window:
  patches, flood_cells, clusters      size of the class
  iou, far, miss, pred_cells          seed means of micro IoU, false alarm ratio, miss rate and predicted flood cells
  d_iou_t0, d_iou_t0_lo, d_iou_t0_hi      micro IoU minus that of T0 (seed mean, interval)
  d_pred_t0_pct (lo, hi)               change of the predicted flood cells from T0, in per cent
  d_iou_t30 (lo, hi)                   micro IoU minus that of T30

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_rq4_timing_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402

W = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/final/upstream/weather')
HERE = Path(__file__).resolve().parents[1] / 'data'
OUT = HERE / 'rq4' / 'timing.csv'
WINDOWS = ('t0', 't3', 't7', 't15', 't30')
CLASSES = ('t3', 't7', 't15', 't30', 'dry')
LENGTHS = (3, 7, 15, 30)


def timing_classes():
    """Timing class of every test patch, in the reference row order, for upstream and local precipitation."""
    ts = cd.test_set()
    key = ts.index[cd.KEY].reset_index(drop=True)
    out = {src: np.empty(len(key), dtype=object) for src in ('up', 'loc')}
    pos = {(f, int(n)): i for i, (f, n) in enumerate(zip(key.folder_name, key.patch_number))}
    for f, g in key.groupby('folder_name'):
        z = np.load(W / f'{f}.npz')
        if z['upstream_precip'].shape[1] != 30:
            raise SystemExit(f'{f}: expected 30 days')
        idx = {int(n): i for i, n in enumerate(z['patch_number'])}
        for src, arr in (('up', z['upstream_precip']), ('loc', z['local_precip'])):
            for n in g.patch_number:
                p = np.nan_to_num(arr[idx[int(n)]])
                total = p.sum()
                if total < 1.0:
                    c = 'dry'
                else:
                    c = next(f't{L}' for L in LENGTHS if p[-L:].sum() >= 0.5 * total)
                out[src][pos[(f, int(n))]] = c
    if any((v == None).any() for v in out.values()):  # noqa: E711
        raise SystemExit('patches without a timing class')
    return out


def main():
    ts = cd.test_set()
    Wb = cd.boot_weights()
    cls = timing_classes()
    C = {(f, w, s): cd.counts(f, w, s) for f in cd.FAMILIES for w in WINDOWS for s in cd.SEEDS}
    flood = (C[('unet', 't0', 1)].tp + C[('unet', 't0', 1)].fn).to_numpy()
    rows = []
    for src, name in (('up', 'upstream'), ('loc', 'local')):
        for k in CLASSES:
            m = cls[src] == k
            if not m.any():
                continue
            present = np.bincount(ts.cluster[m], minlength=ts.n_clusters) > 0
            ok = (Wb[:, present] > 0).any(1)                 # replicates that draw at least one of the class's clusters
            for f in cd.FAMILIES:
                point, reps = {}, {}
                for w in WINDOWS:
                    pt, rp = [], []
                    for s in cd.SEEDS:
                        c = C[(f, w, s)]
                        t, fp, fn = (np.bincount(ts.cluster[m], c[x].to_numpy(float)[m], minlength=ts.n_clusters)
                                     for x in ('tp', 'fp', 'fn'))
                        T, F, N = Wb @ t, Wb @ fp, Wb @ fn
                        pt.append({'iou': t.sum() / (t.sum() + fp.sum() + fn.sum()), 'far': fp.sum() / (t.sum() + fp.sum()),
                                   'miss': fn.sum() / (t.sum() + fn.sum()), 'pred': t.sum() + fp.sum()})
                        with np.errstate(invalid='ignore', divide='ignore'):
                            rp.append({'iou': T / (T + F + N), 'pred': T + F})
                    point[w], reps[w] = pt, rp
                for w in WINDOWS:
                    r = {'source': name, 'class': k, 'family': f, 'window': w, 'patches': int(m.sum()),
                         'flood_cells': int(flood[m].sum()), 'clusters': int(present.sum())}
                    for q in ('iou', 'far', 'miss', 'pred'):
                        r[q if q != 'pred' else 'pred_cells'] = float(np.mean([p[q] for p in point[w]]))
                    for ref, tag in (('t0', 't0'), ('t30', 't30')):
                        d = np.mean([point[w][i]['iou'] - point[ref][i]['iou'] for i in range(3)])
                        dr = np.mean([reps[w][i]['iou'] - reps[ref][i]['iou'] for i in range(3)], axis=0)[ok]
                        lo, hi = cd.interval(dr)
                        r.update({f'd_iou_{tag}': d, f'd_iou_{tag}_lo': lo, f'd_iou_{tag}_hi': hi})
                    d = np.mean([point[w][i]['pred'] / point['t0'][i]['pred'] - 1 for i in range(3)]) * 100
                    dr = np.mean([reps[w][i]['pred'] / reps['t0'][i]['pred'] - 1 for i in range(3)], axis=0)[ok] * 100
                    lo, hi = cd.interval(dr)
                    r.update({'d_pred_t0_pct': d, 'd_pred_t0_pct_lo': lo, 'd_pred_t0_pct_hi': hi})
                    rows.append(r)
            print(name, k, int(m.sum()), 'patches', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 220)
    print(df[df.source == 'upstream'][['class', 'family', 'window', 'patches', 'flood_cells', 'clusters', 'iou',
                                       'd_iou_t0', 'd_iou_t0_lo', 'd_iou_t0_hi', 'd_pred_t0_pct', 'd_iou_t30']]
          .round(4).to_string(index=False))


if __name__ == '__main__':
    main()
