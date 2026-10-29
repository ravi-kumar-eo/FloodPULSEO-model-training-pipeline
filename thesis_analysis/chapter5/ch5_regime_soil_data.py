"""Source data of Figure 5.18 (Section 5.2.3): basin-seen against basin-unseen test ground by the sand and clay content
of the 80 m cell (SoilGrids 2.0, 0-5 cm), U-Net-T0 and ViT-T0, seeds 1-3.

Ravi, 15 Oct 2026: the earlier quintile classes repeated one pattern ten times and hid that the two regimes do not
share the extremes of soil texture (sand below 15%: 7 basin-unseen flood cells; sand of 60% and more: 1,051
basin-seen flood cells). The regimes are therefore compared only in fixed classes of their common range, and the
distribution of the cells of each regime over the whole range is given beside the comparison.
Writes chapters/chapter5/plot/data/conditions/:
  rq2_soil_hist.csv  share of the scored cells and of the flood cells of each regime in 5% bins of sand and clay content
  rq2_soil.csv       per class of the common range (sand 25-35, 35-45, 45-60%; clay 10-20, 20-30, 30-40%): cells and
                     flood cells and clusters of each regime, and per family the POD, POFD and IoU of each regime and
                     their difference (unseen minus seen) with its 95% cluster bootstrap interval (61 clusters, 4,000
                     replicates, the draws of clusters_boot.npz), seed means
Scored test cells of the basin-seen and basin-unseen spatial blocks (the partial block excluded), each fit at its
validation-selected threshold (preds_test_bits.npz).

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_regime_soil_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
CLASSES = {'sand_pct': (25, 35, 45, 60), 'clay_pct': (10, 20, 30, 40)}
HIST = {'sand_pct': np.arange(0, 105, 5), 'clay_pct': np.arange(0, 65, 5)}
FAM = ('unet', 'vit')


def metrics(c):
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    with np.errstate(all='ignore'):
        return {'pod': tp / (tp + fn), 'pofd': fp / (fp + tn), 'iou': tp / (tp + fp + fn)}


def main():
    cells = np.load(D / 'cells_test.npz')
    bits = np.load(D / 'preds_test_bits.npz')
    fits = [k for k in json.loads((D / 'fits.json').read_text()) if '_t0_' in k]
    cb = np.load(D / 'clusters_boot.npz')
    W, clus = cb['weights'].astype(float), cb['cluster'].astype(np.int64)
    K = W.shape[1]
    pt = pd.read_csv(D / 'patches_test.csv.gz', usecols=['row_index', 'basin_regime']).sort_values('row_index')
    Y = cells['label']
    regc = np.where(pt.basin_regime.to_numpy() == 'seen', 0, np.where(pt.basin_regime.to_numpy() == 'unseen', 1, -1))
    rows = np.repeat(np.arange(len(pt)), Y.shape[1] * Y.shape[2])
    sc, fl = (Y >= 0).ravel(), (Y == 1).ravel()
    P = {k: np.unpackbits(bits[k])[:Y.size].astype(bool) for k in fits}
    hist, out = [], []
    for feat, edges in CLASSES.items():
        x = cells[feat].astype(np.float32).ravel()
        valid = sc & np.isfinite(x) & (regc[rows] >= 0)
        for ri, r in enumerate(('seen', 'unseen')):
            m = valid & (regc[rows] == ri)
            h = np.histogram(x[m], HIST[feat])[0]
            hf = np.histogram(x[m & fl], HIST[feat])[0]
            hist += [{'feature': feat, 'regime': r, 'lower': a, 'upper': b, 'cells_share': h[i] / h.sum(),
                      'flood_share': hf[i] / hf.sum(), 'cells': int(h[i]), 'flood_cells': int(hf[i])}
                     for i, (a, b) in enumerate(zip(HIST[feat][:-1], HIST[feat][1:]))]
            print(feat, r, 'cells outside the common range:',
                  f'{100 * ((x[m] < edges[0]) | (x[m] >= edges[-1])).mean():.1f}%')
        C = len(edges) - 1
        cls = np.digitize(x, edges) - 1                                   # -1 below, C at or above the last edge
        inside = valid & (cls >= 0) & (cls < C)
        key = np.where(inside, (regc[rows] * K + clus[rows]) * C + cls, 0)
        nall = np.bincount(key[inside], minlength=2 * K * C).reshape(2, K, C)
        nfl = np.bincount(key[inside & fl], minlength=2 * K * C).reshape(2, K, C)
        per_fit = {}
        for k in fits:
            p = P[k]
            tp = np.bincount(key[inside & p & fl], minlength=2 * K * C).reshape(2, K, C)
            fp = np.bincount(key[inside & p & ~fl], minlength=2 * K * C).reshape(2, K, C)
            per_fit[k] = np.stack([tp, fp, nfl - tp, nall - nfl - fp], -1).astype(float)
        for c in range(C):
            rec = {'feature': feat, 'lower': edges[c], 'upper': edges[c + 1]}
            for ri, r in enumerate(('seen', 'unseen')):
                rec[f'{r}_cells'] = int(nall[ri, :, c].sum())
                rec[f'{r}_flood_cells'] = int(nfl[ri, :, c].sum())
                rec[f'{r}_clusters'] = int((nfl[ri, :, c] > 0).sum())
                rec[f'{r}_cells_share'] = nall[ri, :, c].sum() / nall[ri].sum()
            for f in FAM:
                vals = {(r, m): [] for r in (0, 1) for m in ('pod', 'pofd', 'iou')}
                reps = {m: [] for m in ('pod', 'pofd', 'iou')}
                for k in [k for k in fits if k.startswith(f)]:
                    a = per_fit[k][:, :, c, :]
                    ms = {ri: metrics(a[ri].sum(0)) for ri in (0, 1)}
                    rp = {ri: metrics(W @ a[ri]) for ri in (0, 1)}
                    for m in reps:
                        vals[(0, m)].append(ms[0][m]); vals[(1, m)].append(ms[1][m])
                        reps[m].append(rp[1][m] - rp[0][m])
                for m in reps:
                    s, u = float(np.mean(vals[(0, m)])), float(np.mean(vals[(1, m)]))
                    lo, hi = np.nanpercentile(np.mean(reps[m], 0), [2.5, 97.5])
                    rec.update({f'{f}_{m}_seen': s, f'{f}_{m}_unseen': u, f'{f}_{m}_diff': u - s,
                                f'{f}_{m}_lo': lo, f'{f}_{m}_hi': hi})
            out.append(rec)
    pd.DataFrame(hist).to_csv(D / 'rq2_soil_hist.csv', index=False)
    o = pd.DataFrame(out)
    o.to_csv(D / 'rq2_soil.csv', index=False)
    cols = ['feature', 'lower', 'upper', 'seen_flood_cells', 'unseen_flood_cells', 'seen_clusters', 'unseen_clusters']
    for f in FAM:
        cols += [f'{f}_{m}_{q}' for m in ('pod', 'pofd', 'iou') for q in ('diff', 'lo', 'hi')]
    print(o[cols].round(3).T.to_string())


if __name__ == '__main__':
    main()
