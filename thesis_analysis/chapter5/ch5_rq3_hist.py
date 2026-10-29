"""Probability histograms of the T0 and T30 fits (U-Net and ViT, seeds 1-3) on the test subset, for the
precision-recall curves of Section 5.3.1, built as rq1_data.py builds them for T0.

Writes chapters/chapter5/plot/data/rq3/hist.npz: edges (1,001 bin edges from 0 to 1) and, per fit,
<family>_<window>_s<seed>_flood and _dry, the counts of reference flood and dry scored cells in each probability bin;
bin k holds 0.001 k <= p < 0.001 (k + 1), the last bin includes p = 1. Check: the counts at and above the bin of the
fit's threshold equal TP and FP of its metrics.csv.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_rq3_hist.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq3' / 'hist.npz'
EDGES = np.linspace(0.0, 1.0, 1001)


def main():
    hist = {}
    for f in cd.FAMILIES:
        for w in ('t0', 't30'):
            for s in cd.SEEDS:
                d = cd.run_dir(f, w, s)
                ev = d / 'inference_final'
                thr = float(json.loads((ev / 'threshold.json').read_text())['threshold'])
                P = np.load(ev / 'preds/test__prob.f16.npy').astype(np.float32)
                Y = np.load(ev / 'preds/test__labels.i8.npy')
                keep, truth = Y >= 0, Y == 1
                hf = np.histogram(P[truth], bins=EDGES)[0]
                hd = np.histogram(P[keep & ~truth], bins=EDGES)[0]
                m = pd.read_csv(ev / 'metrics.csv')
                g = m[(m.cohort == 'test') & (m.rows == 'full') & (m.rule == 'global')].iloc[0]
                k = int(round(thr * 1000))
                if (int(hf[k:].sum()), int(hd[k:].sum())) != (int(g.tp), int(g.fp)):
                    raise SystemExit(f'{d}: histogram counts at the threshold differ from metrics.csv')
                hist[f'{f}_{w}_s{s}_flood'], hist[f'{f}_{w}_s{s}_dry'] = hf, hd
                hist[f'{f}_{w}_s{s}_thr'] = np.array(thr)
                print(f, w, s, 'ok', flush=True)
                del P
    np.savez_compressed(OUT, edges=EDGES, **hist)


if __name__ == '__main__':
    main()
