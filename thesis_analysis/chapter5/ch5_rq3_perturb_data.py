"""Source data of the perturbation figure of Section 5.3.6 and of its appendix table: response of the six T30
networks to perturbed dynamic hydroclimatic variables on the test subset (scripts/perturb_eval.py
on Bolzano; its inference_perturb/ folders copied to experiments/runs_final/final_t30/seed<s>/<family>/inference_perturb/).

Every perturbation is predicted with the fit's validation-selected threshold unchanged and compared with the
unperturbed pass of the same run:
  p0    local and upstream precipitation set to zero on all 30 days
  p05   local and upstream precipitation multiplied by 0.5
  p2    local and upstream precipitation multiplied by 2
  swap  precipitation, soil moisture and validity flag of every patch replaced by those of a random test patch from
        another CEMS activation; the catchment area stays the patch's own
The second unperturbed pass (none2) measures the run-to-run variation of the GPU arithmetic.

Writes chapters/chapter5/plot/data/rq3/perturb.csv, one row per family, seed (1-3, and 'mean' when all three seeds
are present) and perturbation:
  pred_change_pct              change of the predicted flood cells (TP + FP) over all scored test cells, in per cent
  pod_change, pofd_change      change of the probability of detection TP / (TP + FN) and of false detection
                               FP / (FP + TN)
  iou_change, far_change, miss_change   change of micro IoU, false alarm ratio and miss rate
  each with its 95% cluster bootstrap interval (61 clusters, 4,000 replicates, the draws of ch5_data.boot_weights;
  for 'mean' the replicates of the three seeds are averaged)
  changed_flood_share          share of the added (or removed) predicted cells that are flood in the reference,
                               (TP - TP_none) / (pred - pred_none), left empty when TP and the predicted cells
                               change in opposite directions; for 'mean' pooled over the seeds (summed changes),
                               with a cluster interval; precision_none: TP / (TP + FP) and pred_none: TP + FP of the
                               unperturbed pass
  mean_abs_prob_change         mean |p - p_none| over the scored cells (summary.csv)
  none2_identical              True when the repeated unperturbed pass reproduced TP, FP and FN exactly
Checks: the rows of every perturbation follow the reference test index (ch5_data.test_set()).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_rq3_perturb_data.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch5_data as cd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq3' / 'perturb.csv'
PERTURBATIONS = ('p0', 'p05', 'p2', 'swap')
CHANGES = (('pred', 'pred_change_pct'), ('pod', 'pod_change'), ('pofd', 'pofd_change'), ('iou', 'iou_change'),
           ('far', 'far_change'), ('miss', 'miss_change'))


def measures(t, f, n, dry):
    return {'tp': t, 'pred': t + f, 'iou': t / (t + f + n), 'far': f / (t + f), 'miss': n / (t + n),
            'pod': t / (t + n), 'pofd': f / dry}


def change(k, v, b):
    return (v[k] / b[k] - 1) * 100 if k == 'pred' else v[k] - b[k]


def main():
    ts = cd.test_set()
    W = cd.boot_weights()
    rows, store = [], {}
    for fam in cd.FAMILIES:
        for s in cd.SEEDS:
            d = cd.RUNS / f'final_t30/seed{s}/{fam}/inference_perturb'
            if not (d / 'patches.csv').exists():
                print('missing', d)
                continue
            p = pd.read_csv(d / 'patches.csv')
            summ = pd.read_csv(d / 'summary.csv').set_index('perturbation')
            sums, reps = {}, {}
            for name in ('none', *PERTURBATIONS):
                q = p[p.perturbation == name].sort_values('row_index')
                if len(q) != ts.n or not (q[cd.KEY].to_numpy() == ts.index[cd.KEY].to_numpy()).all():
                    raise SystemExit(f'{d} {name}: rows differ from the reference test index')
                c = [np.bincount(ts.cluster, q[x].to_numpy(float), minlength=ts.n_clusters) for x in ('tp', 'fp', 'fn')]
                dry = np.bincount(ts.cluster, (q.scored_cells - q.tp - q.fn).to_numpy(float), minlength=ts.n_clusters)
                sums[name] = measures(*(v.sum() for v in c), dry.sum())
                reps[name] = measures(*(W @ v for v in c), W @ dry)
            same = bool('none2' in summ.index and (summ.loc['none2', ['tp', 'fp', 'fn']].to_numpy()
                                                   == summ.loc['none', ['tp', 'fp', 'fn']].to_numpy()).all())
            for name in PERTURBATIONS:
                v, b, rv, rb = sums[name], sums['none'], reps[name], reps['none']
                r = {'family': fam, 'seed': s, 'perturbation': name, 'none2_identical': same,
                     'mean_abs_prob_change': float(summ.loc[name, 'mean_abs_prob_change']),
                     'changed_flood_share': (v['tp'] - b['tp']) / (v['pred'] - b['pred'])
                     if (v['tp'] - b['tp']) * (v['pred'] - b['pred']) > 0 else np.nan,   # undefined when TP and the
                                                                                        # prediction move apart
                     'precision_none': b['tp'] / b['pred'], 'pred_none': b['pred'],
                     'tp_none': b['tp'], 'd_tp': v['tp'] - b['tp'], 'd_pred': v['pred'] - b['pred']}
                store[(fam, s, name, 'd_tp')] = (r['d_tp'], rv['tp'] - rb['tp'])
                store[(fam, s, name, 'd_pred')] = (r['d_pred'], rv['pred'] - rb['pred'])
                for k, key in CHANGES:
                    rr = change(k, rv, rb)
                    store[(fam, s, name, key)] = (change(k, v, b), rr)
                    lo, hi = cd.interval(rr)
                    r.update({key: change(k, v, b), f'{key}_lo': lo, f'{key}_hi': hi})
                rows.append(r)
            print(fam, s, 'repeat pass identical' if same else 'no identical repeat pass', flush=True)
        if all((fam, s, 'p0', 'iou_change') in store for s in cd.SEEDS):
            for name in PERTURBATIONS:
                r = {'family': fam, 'seed': 'mean', 'perturbation': name}
                sub = [x for x in rows if x['family'] == fam and x['perturbation'] == name]
                r['mean_abs_prob_change'] = float(np.mean([x['mean_abs_prob_change'] for x in sub]))
                r['precision_none'] = sum(x['tp_none'] for x in sub) / sum(x['pred_none'] for x in sub)
                r['pred_none'] = float(np.mean([x['pred_none'] for x in sub]))
                dt, dp = sum(x['d_tp'] for x in sub), sum(x['d_pred'] for x in sub)
                r['changed_flood_share'] = dt / dp if dt * dp > 0 else np.nan   # pooled over the seeds
                if dt * dp > 0:
                    rt = sum(store[(fam, s, name, 'd_tp')][1] for s in cd.SEEDS)
                    rp = sum(store[(fam, s, name, 'd_pred')][1] for s in cd.SEEDS)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        lo, hi = cd.interval(np.where(rt * rp > 0, rt / rp, np.nan))
                    r.update(changed_flood_share_lo=lo, changed_flood_share_hi=hi)
                r['none2_identical'] = all(x['none2_identical'] for x in sub)
                for _, key in CHANGES:
                    pt = np.mean([store[(fam, s, name, key)][0] for s in cd.SEEDS])
                    rr = np.mean([store[(fam, s, name, key)][1] for s in cd.SEEDS], axis=0)
                    lo, hi = cd.interval(rr)
                    r.update({key: pt, f'{key}_lo': lo, f'{key}_hi': hi})
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    print(df[['family', 'seed', 'perturbation', 'pred_change_pct', 'pod_change', 'pofd_change', 'iou_change',
              'changed_flood_share', 'precision_none', 'mean_abs_prob_change']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
