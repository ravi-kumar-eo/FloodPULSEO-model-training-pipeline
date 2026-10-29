"""Source data of the first RQ3 figures (Figures 5.16 and 5.34): the twelve T0 and T30 fits (U-Net and ViT, seeds 1-3)
on the test subset, computed exactly as the RQ1 data (rq1_data.py) so that the T0 values equal those of Section 5.1.

Test subset, full rows, flood-only cells (label >= 0), each fit at its validation-selected global threshold, the
evaluator's rule (prob >= threshold). Writes chapters/chapter5/plot/data/rq3/:
  fits.csv      one row per fit (family, window t0/t30, seed): threshold, TP, FP, FN, TN, pooled precision, recall,
                IoU, exact pooled AUPRC (rq1_data.pr_auc: 400 thresholds, Keras interpolation, integer counts)
  products.csv  one row per flood-positive mapping product (394) and fit: flood cells, TP, FP, FN, precision,
                recall, IoU and AUPRC from the run's plots/per_product_test.csv
Checks: TP/FP/FN from the probabilities equal metrics.csv; the per-product mean IoU equals metrics.csv; the T0 rows
equal data/rq1/fits.csv (IoU and exact AUPRC).

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_rq3_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402
from rq1_data import pr_auc  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq3'
RQ1 = Path(__file__).resolve().parents[1] / 'data' / 'rq1' / 'fits.csv'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, prods = [], []
    for w in ('t0', 't30'):
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
                print(f'{f} {w} seed {s}: IoU {rows[-1]["iou"]:.4f}  AUPRC {auprc:.4f}  products {len(pp)}', flush=True)
    fits = pd.DataFrame(rows)
    ref = pd.read_csv(RQ1).set_index(['family', 'seed'])
    t0 = fits[fits.window == 't0'].set_index(['family', 'seed'])
    for k in ('iou', 'auprc'):
        if (t0[k] - ref.loc[t0.index, k]).abs().max() > 1e-9:
            raise SystemExit(f'T0 {k} differs from data/rq1/fits.csv')
    fits.to_csv(OUT / 'fits.csv', index=False)
    pd.concat(prods).to_csv(OUT / 'products.csv', index=False)
    print(fits[['family', 'window', 'seed', 'precision', 'recall', 'iou', 'auprc', 'product_iou']].round(4)
          .to_string(index=False))


if __name__ == '__main__':
    main()
