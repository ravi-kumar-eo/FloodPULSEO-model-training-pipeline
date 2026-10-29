"""Source data of the RQ4 figures (Section 5.4): the networks with an input history of 3, 7, 15 and 30 days (T3, T7,
T15, T30) and without one (T0), U-Net and ViT, seeds 1-3, on the test subset, computed exactly as ch5_rq3_data.py.

Test subset, full rows, flood-only cells (label >= 0), each fit at its validation-selected global threshold, the
evaluator's rule (prob >= threshold). The T0 and T30 rows are copied from data/rq3 (same computation); the T3, T7 and
T15 rows are computed here. Writes chapters/chapter5/plot/data/rq4/:
  fits.csv      one row per fit (family, window t0/t3/t7/t15/t30, seed): threshold, TP, FP, FN, TN, pooled precision,
                recall, IoU, exact pooled AUPRC (rq1_data.pr_auc: 400 thresholds, Keras interpolation, integer counts),
                product-mean IoU
  products.csv  one row per flood-positive mapping product (394) and fit: flood cells, TP, FP, FN, precision,
                recall, IoU and AUPRC from the run's plots/per_product_test.csv
  changes.csv   paired changes from the shorter to the longer of two input-history lengths of the same family and
                seed (and their seed mean): every length against T0, T3, T7 and T15 against T30, and the adjacent
                extensions T3 to T7, T7 to T15 (T15 to T30 is among the former): micro IoU, false alarm
                ratio, miss rate and area bias, and macro IoU, false alarm ratio and miss rate, each with its 95%
                cluster bootstrap interval (61 clusters, 4,000 replicates, the same draws for every fit; macro values
                as the difference of the two macro means, each over the products where its fit defines the measure,
                so that the changes equal the differences of the plotted values)
Checks: TP/FP/FN from the probabilities equal metrics.csv; the per-product mean IoU equals metrics.csv.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_rq4_data.py [--changes-only]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402
from rq1_data import pr_auc  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq4'
RQ3 = Path(__file__).resolve().parents[1] / 'data' / 'rq3'
WINDOWS = ('t0', 't3', 't7', 't15', 't30')
PAIRS = ([('t0', w) for w in ('t3', 't7', 't15', 't30')] + [(w, 't30') for w in ('t3', 't7', 't15')]
         + [('t3', 't7'), ('t7', 't15')])          # (a, b): from the shorter to the longer history, b minus a


def fit_rows():
    rows, prods = [], []
    for w in ('t3', 't7', 't15'):
        for s in cd.SEEDS:
            for f in cd.FAMILIES:
                d = cd.run_dir(f, w, s)
                ev = d / 'inference_final'
                thr = np.float32(json.loads((ev / 'threshold.json').read_text())['threshold'])
                P = np.load(ev / 'preds/test__prob.f16.npy').astype(np.float32)
                Y = np.load(ev / 'preds/test__labels.i8.npy')
                keep, truth = Y >= 0, Y == 1
                pr = (P >= thr) & keep
                tp, fp, fn = int((pr & truth).sum()), int((pr & ~truth).sum()), int((~pr & truth).sum())
                tn = int(keep.sum()) - tp - fp - fn
                m = pd.read_csv(ev / 'metrics.csv')
                g = m[(m.cohort == 'test') & (m.rows == 'full') & (m.rule == 'global')].iloc[0]
                if (tp, fp, fn) != (int(g.tp), int(g.fp), int(g.fn)):
                    raise SystemExit(f'{d}: TP/FP/FN differ from metrics.csv')
                sc = cd.check_against_metrics(f, w, s)
                auprc = pr_auc(P[keep], truth[keep])
                del P
                rows.append({'family': f, 'window': w, 'seed': s, 'threshold': float(thr), 'tp': tp, 'fp': fp,
                             'fn': fn, 'tn': tn, 'precision': tp / (tp + fp), 'recall': tp / (tp + fn),
                             'iou': tp / (tp + fp + fn), 'auprc': auprc, 'product_iou': sc['product_mean_iou']})
                pp = pd.read_csv(d / 'plots/per_product_test.csv')
                pp = pp[pp.flood > 0]
                if len(pp) != 394 or abs(pp.iou.mean() - sc['product_mean_iou']) > 1e-6:   # the CSV holds 6 decimals
                    raise SystemExit(f'{d}: per_product_test.csv differs from metrics.csv')
                cols = ['folder_name', 'flood', 'tp', 'fp', 'fn', 'precision', 'recall', 'iou', 'auprc']
                prods.append(pp[cols].assign(family=f, window=w, seed=s))
                print(f'{f} {w} seed {s}: IoU {rows[-1]["iou"]:.4f}  AUPRC {auprc:.4f}', flush=True)
    fits = pd.concat([pd.read_csv(RQ3 / 'fits.csv'), pd.DataFrame(rows)], ignore_index=True)
    prods = pd.concat([pd.read_csv(RQ3 / 'products.csv'), *prods], ignore_index=True)
    order = {w: i for i, w in enumerate(WINDOWS)}
    fits = fits.sort_values(['family', 'seed', 'window'], key=lambda c: c.map(order) if c.name == 'window' else c)
    return fits.reset_index(drop=True), prods


def boot_measures(c):
    """Replicates (4,000) of the pooled measures and the per-product values of one fit."""
    ts = cd.test_set()
    W = cd.boot_weights()
    t, f, n = (np.bincount(ts.cluster, c[x].to_numpy(float), minlength=ts.n_clusters) for x in ('tp', 'fp', 'fn'))
    T, F, N = W @ t, W @ f, W @ n
    pv = cd.product_values(c)
    T_, F_, N_ = (np.bincount(ts.product, c[x].to_numpy(float), minlength=len(ts.products)) for x in ('tp', 'fp', 'fn'))
    pos = (T_ + N_) > 0
    prod = pd.DataFrame({'iou': np.where(pos, pv.iou.to_numpy(), np.nan),
                         'far': np.where(pos & (T_ + F_ > 0), F_ / np.maximum(T_ + F_, 1), np.nan),
                         'miss': np.where(pos, N_ / np.maximum(T_ + N_, 1), np.nan)})
    point = {'iou': t.sum() / (t.sum() + f.sum() + n.sum()), 'far': f.sum() / (t.sum() + f.sum()),
             'miss': n.sum() / (t.sum() + n.sum()), 'bias': (t.sum() + f.sum()) / (t.sum() + n.sum())}
    reps = {'iou': T / (T + F + N), 'far': F / (T + F), 'miss': N / (T + N), 'bias': (T + F) / (T + N)}
    return point, reps, prod


def changes():
    ts = cd.test_set()
    M = {(f, w, s): boot_measures(cd.counts(f, w, s)) for f in cd.FAMILIES for w in WINDOWS for s in cd.SEEDS}
    out = []
    for f in cd.FAMILIES:
        for a, b in PAIRS:
            seed_reps = {}
            for s in (*cd.SEEDS, 'mean'):
                if s == 'mean':
                    d_point = {k: np.mean([r[k][0] for r in seed_reps.values()]) for k in seed_reps[1]}
                    d_reps = {k: np.mean([r[k][1] for r in seed_reps.values()], axis=0) for k in seed_reps[1]}
                else:
                    (pa, ra, qa), (pb, rb, qb) = M[(f, a, s)], M[(f, b, s)]
                    d_point = {f'micro_{k}': pb[k] - pa[k] for k in pa}
                    d_reps = {f'micro_{k}': rb[k] - ra[k] for k in ra}
                    for k in ('iou', 'far', 'miss'):
                        # difference of the two macro means, each over the products where that fit defines the
                        # measure (FAR: the products in which it predicts flood), as plotted in the figures
                        va, vb = qa[k].to_numpy(), qb[k].to_numpy()
                        d_point[f'macro_{k}'] = float(np.nanmean(vb) - np.nanmean(va))
                        d_reps[f'macro_{k}'] = (cd.boot_mean(vb, ts.product_cluster)
                                                - cd.boot_mean(va, ts.product_cluster))
                    seed_reps[s] = {k: (d_point[k], d_reps[k]) for k in d_point}
                for k in d_point:
                    lo, hi = cd.interval(d_reps[k])
                    out.append({'family': f, 'from': a, 'to': b, 'seed': s, 'measure': k,
                                'change': d_point[k], 'lo': lo, 'hi': hi})
        print(f, 'changes done', flush=True)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--changes-only' in sys.argv:
        fits = pd.read_csv(OUT / 'fits.csv')
    else:
        fits, prods = fit_rows()
        fits.to_csv(OUT / 'fits.csv', index=False)
        prods.to_csv(OUT / 'products.csv', index=False)
    print(fits[['family', 'window', 'seed', 'precision', 'recall', 'iou', 'auprc', 'product_iou']].round(4)
          .to_string(index=False))
    ch = changes()
    ch.to_csv(OUT / 'changes.csv', index=False)
    print(ch[(ch.seed == 'mean') & ch.measure.isin(['micro_iou', 'macro_iou'])].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
