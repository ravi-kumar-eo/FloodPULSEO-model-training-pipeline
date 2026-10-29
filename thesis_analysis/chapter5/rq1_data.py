"""Source data of the RQ1 figures (rq1_01 ... rq1_10): the six T0 fits (U-Net and ViT, seeds 1-3) on the test subset.

Test subset, full rows, flood-only cells (label >= 0), each fit at its validation-selected global threshold, the
evaluator's rule (prob >= threshold). Writes chapters/chapter5/plot/data/rq1/:
  fits.csv         one row per fit: threshold, pooled precision / recall / IoU (metrics.csv), exact pooled AUPRC,
                   mean IoU over flood-positive products and ground units, the same three pooled scores for the
                   basin-seen and basin-unseen regimes, selected epoch, epochs run, hours, minutes per epoch, machine
  hist.npz         per fit, counts of flood and dry cells in 1,000 probability bins of width 0.001 (PR curves and
                   calibration); bin k holds 0.001 k <= p < 0.001 (k + 1), the last bin includes p = 1
  products.csv     one row per flood-positive product and fit: precision, recall, IoU, AUPRC (from the run's
                   plots/per_product_test.csv, written and checked by scripts/plot_run.py)
  agreement.csv    per seed, cells counted by what the two families predict on them: flood cells found by both, by
                   one family only, by neither; dry cells flagged by both or one family
Checks: TP/FP/FN at the threshold from the probabilities and from the histogram equal metrics.csv; per-product means
equal metrics.csv. The pooled AUPRC is recounted exactly (400 thresholds, Keras interpolation): the value in
metrics.csv is slightly too high because the evaluator counts all ~49 M cells in one float32 Keras update.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/rq1_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq1'
EDGES = np.linspace(0.0, 1.0, 1001)


def pr_auc(p, y, n=400):
    """Keras AUC(curve='PR', summation_method='interpolation', num_thresholds=n) with exact integer counts."""
    eps = 1e-7
    thr = np.array([0.0 - eps] + [(i + 1) / (n - 1) for i in range(n - 2)] + [1.0 + eps], np.float32)
    pos, neg = np.sort(p[y]), np.sort(p[~y])
    tp = (len(pos) - np.searchsorted(pos, thr, side='right')).astype(float)
    fp = (len(neg) - np.searchsorted(neg, thr, side='right')).astype(float)
    fn = len(pos) - tp
    dtp = tp[:-1] - tp[1:]
    P = tp + fp
    dp = np.maximum(P[:-1] - P[1:], 0)
    slope = np.divide(dtp, dp, out=np.zeros_like(dtp), where=dp != 0)
    intercept = tp[1:] - slope * P[1:]
    ratio = np.ones_like(dtp)
    ok = (P[:-1] > 0) & (P[1:] > 0)
    ratio[ok] = P[:-1][ok] / P[1:][ok]
    num = slope * (dtp + intercept * np.log(ratio))
    den = np.maximum(tp[1:] + fn[1:], 0)
    return float(np.divide(num, den, out=np.zeros_like(num), where=den != 0).sum())


def load(f, s):
    d = cd.run_dir(f, 't0', s)
    ev = d / 'inference_final'
    thr = np.float32(json.loads((ev / 'threshold.json').read_text())['threshold'])
    P = np.load(ev / 'preds/test__prob.f16.npy').astype(np.float32)
    Y = np.load(ev / 'preds/test__labels.i8.npy')
    return d, thr, P, Y


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ts = cd.test_set()
    rows, hist, prods, agree = [], {}, [], []
    for s in cd.SEEDS:
        pred = {}
        for f in cd.FAMILIES:
            d, thr, P, Y = load(f, s)
            keep, truth = Y >= 0, Y == 1
            pr = (P >= thr) & keep
            tp, fp, fn = int((pr & truth).sum()), int((pr & ~truth).sum()), int((~pr & truth).sum())
            m = pd.read_csv(d / 'inference_final/metrics.csv')
            g = lambda c: m[(m.cohort == c) & (m.rows == 'full') & (m.rule == 'global')].iloc[0]
            if (tp, fp, fn) != (int(g('test').tp), int(g('test').fp), int(g('test').fn)):
                raise SystemExit(f'{d}: TP/FP/FN differ from metrics.csv')
            hf = np.histogram(P[truth], bins=EDGES)[0]
            hd = np.histogram(P[keep & ~truth], bins=EDGES)[0]
            k = int(round(float(thr) * 1000))                   # thresholds lie on the 0.02 grid: bin of the threshold
            if (int(hf[k:].sum()), int(hd[k:].sum())) != (tp, fp):
                raise SystemExit(f'{d}: histogram counts at the threshold differ from metrics.csv')
            hist[f'{f}_s{s}_flood'], hist[f'{f}_s{s}_dry'] = hf, hd
            sc = cd.check_against_metrics(f, 't0', s)
            man = json.loads((d / 'run_manifest.json').read_text())
            env = man['environment']
            r = {'family': f, 'seed': s, 'threshold': float(thr), 'precision': tp / (tp + fp), 'recall': tp / (tp + fn),
                 'iou': tp / (tp + fp + fn), 'auprc': pr_auc(P[keep], truth[keep]),
                 'auprc_metrics_csv': float(g('test').pooled_auprc), 'product_iou': sc['product_mean_iou'],
                 'unit_iou': sc['unit_mean_iou'], 'selected_epoch': man['selection']['selected_epoch'],
                 'epochs_run': man['selection']['epochs_run'], 'hours': man['hours'],
                 'minutes_per_epoch': 60 * man['hours'] / man['selection']['epochs_run'],
                 'machine': env.get('compute'), 'gpu': env.get('gpu', '').split(',')[0]}
            for reg in ('seen', 'unseen'):
                q = g(f'test:{reg}')
                r.update({f'{reg}_precision': q.pooled_precision, f'{reg}_recall': q.pooled_recall,
                          f'{reg}_iou': q.pooled_iou, f'{reg}_patches': int(q.n_patches)})
            rows.append(r)
            pp = pd.read_csv(d / 'plots/per_product_test.csv', index_col=0)
            if abs(pp.iou.mean() - sc['product_mean_iou']) > 1e-6:      # the CSV holds 6 decimals
                raise SystemExit(f'{d}: per_product_test.csv differs from metrics.csv')
            prods.append(pp[['regime', 'flood', 'precision', 'recall', 'iou', 'auprc']].assign(family=f, seed=s))
            pred[f] = pr
            truth_ref = truth
            print(f'{f} seed {s}: IoU {r["iou"]:.4f}, AUPRC exact {r["auprc"]:.4f} (metrics.csv {r["auprc_metrics_csv"]:.4f})',
                  flush=True)
            del P
        u, v, t = pred['unet'], pred['vit'], truth_ref
        agree.append({'seed': s, 'flood_both': int((u & v & t).sum()), 'flood_vit_only': int((~u & v & t).sum()),
                      'flood_unet_only': int((u & ~v & t).sum()), 'flood_neither': int((~u & ~v & t).sum()),
                      'fp_both': int((u & v & ~t).sum()), 'fp_vit_only': int((~u & v & ~t).sum()),
                      'fp_unet_only': int((u & ~v & ~t).sum())})
    pd.DataFrame(rows).to_csv(OUT / 'fits.csv', index=False)
    np.savez_compressed(OUT / 'hist.npz', edges=EDGES, **hist)
    pd.concat(prods).rename_axis('product').to_csv(OUT / 'products.csv')
    pd.DataFrame(agree).to_csv(OUT / 'agreement.csv', index=False)
    print(pd.DataFrame(rows)[['family', 'seed', 'precision', 'recall', 'iou', 'auprc', 'product_iou', 'unit_iou',
                              'seen_recall', 'unseen_recall', 'selected_epoch', 'epochs_run', 'minutes_per_epoch',
                              'machine']].round(4).to_string(index=False))
    print(pd.DataFrame(agree).to_string(index=False))


if __name__ == '__main__':
    main()
