"""Cell counts by HAND class for Figure 5.19 (error changes along height above drainage). Runs ON BOLZANO, where
the per-cell HAND table of the test subset and every fit's saved predictions live:

    python3 ch5_hand_counts.py > fig5_7_hand_counts.csv        (then copy to chapters/chapter5/plot/data/)

Inputs
  tmp_analysis_unet_s1/cells_test.npz   label and HAND (m) of every test cell, rows in the evaluator's order
                                        (built 8 October 2026 by review_unet_s1/scripts/build_tables.py with the
                                        loader's own TIF reader; nearest-neighbour HAND of input_80m)
  runs_final/<fit>/inference_final/preds/test__prob.f16.npy, test__labels.i8.npy, threshold.json, metrics.csv
Rule: a cell is predicted flooded when prob >= the fit's validation-selected global threshold, on flood-only
cells (label >= 0), exactly the evaluator's rule. Every fit's total TP, FP and FN are checked against its
metrics.csv (test, full rows, global rule); the labels of every fit are checked against the HAND table's.
Output (CSV to stdout): one row per family, configuration, seed and HAND class: scored and flood cells, the fit's
and its same-seed T0's TP, FP, FN and FP on flood-free patches, and the turnover against T0 (FP removed and new,
flood cells recovered and newly missed). Fits that are not finished are skipped and listed on stderr.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/home/ravi/shared_data/global_flood_dataset/ravi/experiments/runs_final')
CELLS = Path('/home/ravi/shared_data/global_flood_dataset/ravi/tmp_analysis_unet_s1/cells_test.npz')
EDGES = [1, 2, 5, 10, 20, 50]
CLASSES = ['<1', '1-2', '2-5', '5-10', '10-20', '20-50', '>=50', 'nodata']
CONFIGS = [  # (label, run dir pattern relative to runs_final, seeds)
    ('T30', 'final_t30/seed{s}/{f}', (1, 2, 3)),
    ('T3', 'final_window/seed{s}/{f}_t3', (1, 2, 3)),
    ('T7', 'final_window/seed{s}/{f}_t7', (1, 2, 3)),
    ('T15', 'final_window/seed{s}/{f}_t15', (1, 2, 3)),
    ('T30 local', 'final_ablation/seed{s}/{f}_noup', (1,)),
    ('T30 daily', 'final_ablation/seed{s}/{f}_nosum', (1,)),
    ('T30 statistics', 'final_ablation/seed{s}/{f}_sumonly', (1,)),
    ('T30 random init', 'final_scratch/seed{s}/{f}', (1,)),
]


def finished(d):
    return all((d / x).exists() for x in ('COMPLETE', 'inference_final/metrics.csv', 'inference_final/threshold.json',
                                         'inference_final/preds/test__prob.f16.npy'))


def masks(d, Y, keep, truth, dry):
    thr = np.float32(json.loads((d / 'inference_final/threshold.json').read_text())['threshold'])
    if not np.array_equal(np.load(d / 'inference_final/preds/test__labels.i8.npy'), Y):
        raise SystemExit(f'{d}: labels differ from the HAND table')
    pred = (np.load(d / 'inference_final/preds/test__prob.f16.npy').astype(np.float32) >= thr) & keep
    tp, fp, fn = pred & truth, pred & ~truth, ~pred & truth
    m = pd.read_csv(d / 'inference_final/metrics.csv')
    m = m[(m.cohort == 'test') & (m.rows == 'full') & (m.rule == 'global')].iloc[0]
    if (int(tp.sum()), int(fp.sum()), int(fn.sum())) != (int(m.tp), int(m.fp), int(m.fn)):
        raise SystemExit(f'{d}: TP/FP/FN differ from metrics.csv')
    return tp, fp, fn, fp & dry


def main():
    z = np.load(CELLS)
    Y, H = z['label'], z['hand'].astype(np.float32)
    keep, truth = Y >= 0, Y == 1
    dry = ~truth.any(axis=(1, 2))[:, None, None] & keep          # flood-free patches
    cls = np.digitize(np.nan_to_num(H, nan=-1.0), EDGES)          # 0..6
    cls[np.isnan(H)] = 7
    cls = cls.ravel()
    count = lambda b: np.bincount(cls, b.ravel(), minlength=len(CLASSES)).astype(np.int64)
    rows, missing = [], []
    base = {'scored_cells': count(keep), 'flood_cells': count(truth)}
    for f in ('unet', 'vit'):
        for label, pat, seeds in CONFIGS:
            for s in seeds:
                d, d0 = R / pat.format(s=s, f=f), R / f'final_t0/seed{s}/{f}'
                if not (finished(d) and finished(d0)):
                    missing.append(f'{f} {label} s{s}')
                    continue
                tp0, fp0, fn0, fd0 = masks(d0, Y, keep, truth, dry)
                tp1, fp1, fn1, fd1 = masks(d, Y, keep, truth, dry)
                c = {**base, 'tp': count(tp1), 'fp': count(fp1), 'fn': count(fn1), 'fp_dry': count(fd1),
                     't0_tp': count(tp0), 't0_fp': count(fp0), 't0_fn': count(fn0), 't0_fp_dry': count(fd0),
                     'fp_removed': count(fp0 & ~fp1), 'fp_new': count(~fp0 & fp1),
                     'fn_recovered': count(fn0 & tp1), 'fn_new': count(tp0 & fn1)}
                for k, name in enumerate(CLASSES):
                    rows.append({'family': f, 'config': label, 'seed': s, 'hand_class': name,
                                 **{key: int(v[k]) for key, v in c.items()}})
                print(f'done {f} {label} s{s}', file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_csv(sys.stdout, index=False)
    print('not finished (skipped): ' + (', '.join(missing) or 'none'), file=sys.stderr)


if __name__ == '__main__':
    main()
