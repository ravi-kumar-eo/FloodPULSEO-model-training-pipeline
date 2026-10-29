"""IoU, POD and POFD of every fit and their paired changes (Ravi, 17 and 19 Oct 2026: the thesis reports IoU, POD and
POFD; F1, AUPRC, false alarm ratio, miss rate and area bias only in one appendix table of seed means).

  POD   probability of detection = TP / (TP + FN) (= recall)
  POFD  probability of false detection = FP / (FP + TN), TN = dry scored cells predicted dry

Test subset, full rows, scored cells, each fit at its validation-selected global threshold, from the per-patch counts
of every run (ch5_data.counts: TP, FP, FN and scored cells; dry cells = scored - TP - FN). Micro: counts pooled over
all scored test cells. Macro: the metric within each of the 394 flood-positive mapping products, then the mean.
Intervals: 95% cluster bootstrap, the 61 dependence clusters and 4,000 draws of ch5_data (the same draws for every
fit), paired by seed; the seed-mean change is the mean of the paired changes (Section 4.6.3).

Writes chapters/chapter5/plot/data/metrics/:
  fits.csv     one row per fit (family, config, seed): TP, FP, FN, TN, micro IoU, POD, POFD, precision, F1, false
               alarm ratio, miss rate, area bias, AUPRC (from data/rq4/fits.csv and data/rq3/ablations.csv),
               macro IoU, POD, POFD
  changes.csv  paired changes b - a with intervals, micro and macro IoU, POD and POFD: every length against T0, the
               adjacent lengths, T3/T7/T15 against T30, each ablation against T30 of seed 1, ViT minus U-Net at T0,
               and the T0-to-T30 change of ViT minus that of U-Net
Checks: TP, FP, FN equal data/rq4/fits.csv (T0-T30) and data/rq3/ablations.csv (ablations).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_pod_pofd_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'data'
OUT = DATA / 'metrics'
WINDOWS = ('t0', 't3', 't7', 't15', 't30')
ABLATIONS = ('noup', 'nosum', 'sumonly', 'scratch')
MEASURES = ('iou', 'pod', 'pofd')


def configs():
    for f in cd.FAMILIES:
        for w in WINDOWS:
            for s in cd.SEEDS:
                yield f, w, s, None
        for a in ABLATIONS:
            yield f, a, 1, a


def measures(c):
    """Point values and bootstrap replicates of micro and macro IoU, POD and POFD of one fit."""
    ts = cd.test_set()
    W = cd.boot_weights()
    tp, fp, fn, k = (c[x].to_numpy(float) for x in ('tp', 'fp', 'fn', 'scored_cells'))
    dry = k - tp - fn
    t, f, n, d = (np.bincount(ts.cluster, v, minlength=ts.n_clusters) for v in (tp, fp, fn, dry))
    T, F, N, D = W @ t, W @ f, W @ n, W @ d
    point = {'micro_iou': t.sum() / (t.sum() + f.sum() + n.sum()), 'micro_pod': t.sum() / (t.sum() + n.sum()),
             'micro_pofd': f.sum() / d.sum()}
    reps = {'micro_iou': T / (T + F + N), 'micro_pod': T / (T + N), 'micro_pofd': F / D}
    Tp, Fp, Np, Dp = (np.bincount(ts.product, v, minlength=len(ts.products)) for v in (tp, fp, fn, dry))
    pos = (Tp + Np) > 0                                  # the 394 flood-positive products
    prod = {'iou': np.where(pos, Tp / np.maximum(Tp + Fp + Np, 1), np.nan),
            'pod': np.where(pos, Tp / np.maximum(Tp + Np, 1), np.nan),
            'pofd': np.where(pos & (Dp > 0), Fp / np.maximum(Dp, 1), np.nan)}
    for m in MEASURES:
        point[f'macro_{m}'] = float(np.nanmean(prod[m]))
        reps[f'macro_{m}'] = cd.boot_mean(prod[m], ts.product_cluster)
    counts = {'tp': int(t.sum()), 'fp': int(f.sum()), 'fn': int(n.sum()), 'tn': int(d.sum() - f.sum()),
              'products': int(pos.sum())}
    return point, reps, counts


def main():
    M = {}
    rows = []
    rq4 = pd.read_csv(DATA / 'rq4' / 'fits.csv').set_index(['family', 'window', 'seed'])
    abl = pd.read_csv(DATA / 'rq3' / 'ablations.csv')
    for f, w, s, var in configs():
        c = cd.counts(f, 't30' if var else w, s, var)
        point, reps, cnt = measures(c)
        M[(f, w, s)] = (point, reps)
        if var is None:
            ref = rq4.loc[(f, w, s)]
            auprc = float(ref.auprc)
        else:
            ref = abl[(abl.family == f) & (abl.config == var)].iloc[0]
            auprc = float(ref.auprc)
        if (cnt['tp'], cnt['fp'], cnt['fn']) != (int(ref.tp), int(ref.fp), int(ref.fn)):
            raise SystemExit(f'{f} {w} {s}: counts differ from the stored fit table')
        tp, fp, fn = cnt['tp'], cnt['fp'], cnt['fn']
        rows.append({'family': f, 'config': w, 'seed': s, **cnt, **point, 'micro_precision': tp / (tp + fp),
                     'micro_f1': 2 * tp / (2 * tp + fp + fn), 'micro_far': fp / (tp + fp),
                     'micro_miss': fn / (tp + fn), 'micro_area_bias': (tp + fp) / (tp + fn), 'micro_auprc': auprc})
        print(f, w, s, {k: round(v, 4) for k, v in point.items()}, flush=True)
    fits = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    fits.to_csv(OUT / 'fits.csv', index=False)

    out = []

    def add(kind, f, a, b, seeds, pairs):
        """b minus a for each seed (pairs(s) -> (key_a, key_b)) and for the seed mean."""
        per = {}
        for s in seeds:
            ka, kb = pairs(s)
            (pa, ra), (pb, rb) = M[ka], M[kb]
            per[s] = {m: (pb[m] - pa[m], rb[m] - ra[m]) for m in pa}
        keys = list(per[seeds[0]])
        per['mean'] = {m: (np.mean([per[s][m][0] for s in seeds]), np.mean([per[s][m][1] for s in seeds], axis=0))
                       for m in keys}
        for s, d in per.items():
            for m, (v, r) in d.items():
                lo, hi = cd.interval(r)
                out.append({'kind': kind, 'family': f, 'from': a, 'to': b, 'seed': s, 'measure': m,
                            'change': v, 'lo': lo, 'hi': hi})

    pairs = ([('t0', w) for w in WINDOWS[1:]] + [(w, 't30') for w in ('t3', 't7', 't15')]
             + [('t3', 't7'), ('t7', 't15')])
    for f in cd.FAMILIES:
        for a, b in pairs:
            add('length', f, a, b, cd.SEEDS, lambda s, f=f, a=a, b=b: ((f, a, s), (f, b, s)))
        for v in ABLATIONS + ('t0',):
            add('ablation', f, 't30', v, (1,), lambda s, f=f, v=v: ((f, 't30', 1), (f, v, 1)))
    add('architecture', 'vit-unet', 'unet', 'vit', cd.SEEDS,
        lambda s: (('unet', 't0', s), ('vit', 't0', s)))
    # T0-to-T30 change of ViT minus that of U-Net, paired by seed and draw
    per = {}
    for s in cd.SEEDS:
        per[s] = {m: ((M[('vit', 't30', s)][0][m] - M[('vit', 't0', s)][0][m])
                      - (M[('unet', 't30', s)][0][m] - M[('unet', 't0', s)][0][m]),
                      (M[('vit', 't30', s)][1][m] - M[('vit', 't0', s)][1][m])
                      - (M[('unet', 't30', s)][1][m] - M[('unet', 't0', s)][1][m]))
                  for m in M[('vit', 't0', s)][0]}
    for m in per[1]:
        v = np.mean([per[s][m][0] for s in cd.SEEDS])
        r = np.mean([per[s][m][1] for s in cd.SEEDS], axis=0)
        lo, hi = cd.interval(r)
        out.append({'kind': 'architecture_change', 'family': 'vit-unet', 'from': 't0', 'to': 't30', 'seed': 'mean',
                    'measure': m, 'change': v, 'lo': lo, 'hi': hi})
    ch = pd.DataFrame(out)
    ch.to_csv(OUT / 'changes.csv', index=False)
    pd.set_option('display.width', 200)
    print(ch[ch.seed.astype(str) == 'mean'].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
