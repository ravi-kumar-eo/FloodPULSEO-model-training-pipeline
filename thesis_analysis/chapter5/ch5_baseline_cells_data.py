"""Section 5.5 — cell-level data of the comparison with the Global Flood Forecasting (GFF) models.

Reads the scored 80 m cells of the 21 sites with Kuro Siwo expert labels (experiments/runs_baselines/gff/results/
<batch>/cells/<site>.npz, written by experiments/src/baselines/gff/score_sites.py) and writes plot/data/baseline/:
  events.csv      one row per Kuro Siwo flood event and model group: mean IoU over the event's sites (as Table 5.7),
                  number of sites, scored and labelled flood cells. Ours = mean of seeds 1-3 at each site.
  pooled.csv      one row per fit (ours, every seed) and GFF model, pooled over the 21 sites: TP FP FN TN, IoU,
                  precision, POD, POFD, area bias and AUPRC (average precision, the rule of score_sites.py).
  pr_curves.csv   precision at 101 recall levels (0 to 1) pooled over the 21 sites: GFF models from their water
                  probability, ours as the mean and range of the three seeds (interpolated precision, the highest
                  precision at a recall at least as large).
  agreement.csv   per seed, architecture, input-history length (T0 to T30) and GFF model: labelled flood cells detected
                  by both, by ours only, by GFF only and by neither, and false alarms of both, ours only and GFF only.
Site rule (Ravi, 16 Oct 2026): the hand-labelled sites, without 267-1, which touches our validation subset.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_baseline_cells_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_baselines/gff')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'baseline'
GFF = {'utae': 'gff_utae', 'metnet': 'gff_maxvit'}
WINDOWS = (0, 3, 7, 15, 30)
REC = np.linspace(0, 1, 101)


RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')


def threshold(fit):
    """The fit's global threshold selected on our validation subset (inference_final/threshold.json)."""
    stage, seed, arch = fit.split('__')
    w = int(stage.replace('final_t', ''))
    run = (RUNS / f'final_t{w}' / seed.replace('s', 'seed') / arch if w in (0, 30)
           else RUNS / 'final_window' / seed.replace('s', 'seed') / f'{arch}_t{w}')
    return json.loads((run / 'inference_final' / 'threshold.json').read_text())['threshold']


def counts(y, yes):
    tp = int(np.sum(yes & (y == 1))); fp = int(np.sum(yes & (y == 0)))
    fn = int(np.sum(~yes & (y == 1))); tn = int(np.sum(~yes & (y == 0)))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, IoU=tp / (tp + fp + fn), precision=tp / max(tp + fp, 1),
                POD=tp / (tp + fn), POFD=fp / (fp + tn), area_bias=(tp + fp) / (tp + fn))


def curve(y, p):
    """Average precision and interpolated precision at the recall levels REC."""
    o = np.argsort(-p, kind='stable')
    yy = y[o] == 1
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    rec = tp / yy.sum()
    ap = float(np.sum(prec[yy]) / yy.sum())
    best = np.maximum.accumulate(prec[::-1])[::-1]          # highest precision at this or a larger recall
    idx = np.searchsorted(rec, REC, side='left').clip(0, len(rec) - 1)
    return ap, best[idx]


def main():
    sites = pd.read_csv(ROOT / 'results' / 'all' / 'all_sites.csv')
    sites = sites[sites.label.str.startswith('hand') & ~sites.site.str.startswith('267-1-')]
    assert len(sites) == 21, len(sites)
    z = {s.site: np.load(ROOT / 'results' / s.batch / 'cells' / f'{s.site}.npz') for s in sites.itertuples()}
    y = np.concatenate([z[s]['label'] for s in z]).astype(np.int8)
    print(f'{len(z)} sites, {y.size:,} scored cells, {int(y.sum()):,} flood ({y.mean():.4f})')
    OUT.mkdir(parents=True, exist_ok=True)

    # events: mean of the site IoU of each group (seed means for ours), as Table 5.7
    d = pd.read_csv(OUT / 'sites.csv')
    d = d[d.label_type == 'hand']
    ev = (d.groupby(['event', 'group'], as_index=False)
          .agg(IoU=('IoU', 'mean'), sites=('site', 'nunique'), cells=('cells', 'sum'),
               flood_cells=('flood_cells', 'sum'), country=('country', 'first'), flood_date=('flood_date', 'first'),
               our_test_event=('our_test_event', 'max')))
    ev.to_csv(OUT / 'events.csv', index=False)

    pooled, curves = [], []
    for m, g in GFF.items():
        p = np.concatenate([z[s][f'{m}_prob'] for s in z]); yes = np.concatenate([z[s][f'{m}_yes'] for s in z])
        ap, pr = curve(y, p)
        pooled.append(dict(group=g, seed=0, AUPRC=ap, **counts(y, yes)))
        curves.append(pd.DataFrame({'group': g, 'recall': REC, 'precision': pr, 'lo': pr, 'hi': pr}))
    votes = {}
    for a in ('unet', 'vit'):
        for w in WINDOWS:
            prs = []
            for s in (1, 2, 3):
                fit = f'final_t{w}__s{s}__{a}'
                p = np.concatenate([z[x][f'ours_{fit}'] for x in z])
                yes = p >= threshold(fit)
                ap, pr = curve(y, p)
                prs.append(pr)
                pooled.append(dict(group=f'{a}_t{w}', seed=s, AUPRC=ap, **counts(y, yes)))
                votes[(a, w, s)] = yes
            prs = np.array(prs)
            curves.append(pd.DataFrame({'group': f'{a}_t{w}', 'recall': REC, 'precision': prs.mean(0),
                                        'lo': prs.min(0), 'hi': prs.max(0)}))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(OUT / 'pooled.csv', index=False)
    pd.concat(curves).to_csv(OUT / 'pr_curves.csv', index=False)

    rows = []
    F, D = y == 1, y == 0
    for m, g in GFF.items():
        gy = np.concatenate([z[s][f'{m}_yes'] for s in z])
        for (a, w, s), o in votes.items():
            rows.append(dict(group=f'{a}_t{w}', gff=g, seed=s, flood=int(F.sum()),
                             both=int((o & gy & F).sum()), ours_only=int((o & ~gy & F).sum()),
                             gff_only=int((~o & gy & F).sum()), neither=int((~o & ~gy & F).sum()),
                             fa_both=int((o & gy & D).sum()), fa_ours_only=int((o & ~gy & D).sum()),
                             fa_gff_only=int((~o & gy & D).sum())))
    ag = pd.DataFrame(rows)
    ag.to_csv(OUT / 'agreement.csv', index=False)

    s = pooled.groupby('group')[['IoU', 'precision', 'POD', 'POFD', 'area_bias', 'AUPRC']].mean()
    print(s.round(3).to_string())
    m = ag.groupby(['group', 'gff']).mean(numeric_only=True)
    for c in ('both', 'ours_only', 'gff_only', 'neither'):
        m[c + '_share'] = m[c] / m.flood
    fa = m.fa_both + m.fa_ours_only + m.fa_gff_only
    for c in ('fa_both', 'fa_ours_only', 'fa_gff_only'):
        m[c + '_share'] = m[c] / fa
    print(m.filter(like='share').round(3).to_string())


if __name__ == '__main__':
    main()
