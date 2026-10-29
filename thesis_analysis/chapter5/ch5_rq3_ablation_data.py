"""Source data of the ablation figure (Figure 5.31): T0 and T30 of seed 1, T30 of seeds 2-3 (seed range)
and the four ablations of seed 1, per family, on the test subset, computed exactly as ch5_rq3_data.py.

Writes chapters/chapter5/plot/data/rq3/ablations.csv, one row per fit: config (t0, t30, noup, nosum, sumonly,
scratch), seed, TP, FP, FN, precision, recall, IoU, exact pooled AUPRC (rq1_data.pr_auc). TP/FP/FN are checked against
each run's metrics.csv.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_rq3_ablation_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402
from rq1_data import pr_auc  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq3' / 'ablations.csv'
FITS = [('t0', 't0', None, 1)] + [('t30', 't30', None, s) for s in (1, 2, 3)] + \
       [(v, 't30', v, 1) for v in ('noup', 'nosum', 'sumonly', 'scratch')]


def main():
    rows = []
    for f in cd.FAMILIES:
        for config, window, variant, seed in FITS:
            d = cd.run_dir(f, window, seed, variant)
            ev = d / 'inference_final'
            thr = np.float32(json.loads((ev / 'threshold.json').read_text())['threshold'])
            P = np.load(ev / 'preds/test__prob.f16.npy').astype(np.float32)
            Y = np.load(ev / 'preds/test__labels.i8.npy')
            keep, truth = Y >= 0, Y == 1
            pr = (P >= thr) & keep
            tp, fp, fn = int((pr & truth).sum()), int((pr & ~truth).sum()), int((~pr & truth).sum())
            m = pd.read_csv(ev / 'metrics.csv')
            g = m[(m.cohort == 'test') & (m.rows == 'full') & (m.rule == 'global')].iloc[0]
            if (tp, fp, fn) != (int(g.tp), int(g.fp), int(g.fn)):
                raise SystemExit(f'{d}: TP/FP/FN differ from metrics.csv')
            rows.append({'family': f, 'config': config, 'seed': seed, 'tp': tp, 'fp': fp, 'fn': fn,
                         'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'iou': tp / (tp + fp + fn),
                         'auprc': pr_auc(P[keep], truth[keep])})
            print(f, config, seed, round(rows[-1]['iou'], 4), round(rows[-1]['auprc'], 4), flush=True)
            del P
    pd.DataFrame(rows).to_csv(OUT, index=False)


if __name__ == '__main__':
    main()
