"""RQ3 — Inter-event variation within spatial blocks: do the T30 networks change their prediction of a mapping
product in the direction of its observed flood, relative to the other flood events mapped on the same block?

Test spatial blocks mapped in at least two CEMS activations (15 blocks, 135 mapping products, 29 activations).
x: observed flooded share of the product relative to the mean of its block, log10(share + 1e-4) minus the block mean.
y: change of the predicted flooded share from T0 to T30, log10(T30 share + 1e-4) - log10(T0 share + 1e-4); the two
   predictions cover the same patches, so the footprint of the product cancels.
Predicted share: TP + FP over the scored cells, mean over seeds 1-3. One point per product; products whose change
lies below -1 are drawn as triangles on the lower edge. Spearman correlation per family in the panel title. A network that takes event information from the dynamic
hydroclimatic variables would place the points along a rising line.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_23_t30_inter_event.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rq1_common as rc
from rq1_common import ps

C = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
EPS = 1e-4
YMIN = -1.0                                    # lower limit of the y-axis; products below it are drawn on the edge


def products():
    p = pd.read_csv(C / 'patches_test.csv.gz')
    p['act'] = p.folder_name.str.split('_').str[0]
    n_act = p.groupby('ground_unit').act.nunique()
    q = p[p.ground_unit.isin(n_act[n_act >= 2].index)].copy()
    agg = {'scored': ('scored', 'sum'), 'flood': ('flood', 'sum')}
    for f in rc.FAMILIES:
        for w in ('t0', 't30'):
            q[f'{f}_{w}'] = sum(q[f'{f}_{w}_s{s}_tp'] + q[f'{f}_{w}_s{s}_fp'] for s in rc.SEEDS) / len(rc.SEEDS)
            agg[f'{f}_{w}'] = (f'{f}_{w}', 'sum')
    d = q.groupby(['ground_unit', 'act', 'folder_name']).agg(**agg).reset_index()
    lg = lambda v: np.log10(v / d.scored + EPS)
    d['obs'] = lg(d.flood)
    d['obs_dev'] = d.obs - d.groupby('ground_unit').obs.transform('mean')
    for f in rc.FAMILIES:
        d[f'{f}_change'] = lg(d[f'{f}_t30']) - lg(d[f'{f}_t0'])
    return d


def main():
    rc.style()
    d = products()
    blocks = sorted(d.ground_unit.unique())
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 3.0), sharex=True, sharey=True)
    for k, (ax, f) in enumerate(zip(axes, rc.FAMILIES)):
        rho, pv = spearmanr(d.obs_dev, d[f'{f}_change'])
        y = d[f'{f}_change'].to_numpy()
        off = y < YMIN                                     # off-scale products: triangles on the lower edge
        ax.scatter(d.obs_dev[~off], y[~off], s=14, color=ps.alpha_on_white(rc.COLOR[f], 0.8), edgecolor=rc.COLOR[f],
                   lw=0.5, zorder=3)
        ax.scatter(d.obs_dev[off], np.full(off.sum(), YMIN + 0.03), s=22, marker='v', color=rc.COLOR[f], zorder=3)
        ax.set_ylim(YMIN, 0.45)
        ax.axhline(0, color='black', lw=0.6, zorder=1)
        ax.axvline(0, color='black', lw=0.6, zorder=1)
        ax.set_xlabel('Labelled flooded share relative\nto the block mean (log10)', fontsize=ps.FONT_SIZE - 1.5)
        if k == 0:
            ax.set_ylabel('Change in predicted flooded\nshare, T30 vs T0 (log10)', fontsize=ps.FONT_SIZE - 1.5)
        rc.ygrid(ax)
        ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
        rc.panel_title(ax, f'({"ab"[k]}) {rc.LABEL[f]}, Spearman {rho:+.2f}')
        print(f'{rc.LABEL[f]}: Spearman {rho:+.3f} (p {pv:.3f}), n {len(d)}; change median {d[f"{f}_change"].median():+.3f}, '
              f'IQR {d[f"{f}_change"].quantile(0.25):+.3f}..{d[f"{f}_change"].quantile(0.75):+.3f}')
    rc.yticks_everywhere(axes)
    fig.tight_layout(w_pad=1.5)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    print('blocks', len(blocks), 'products', len(d), 'activations', d.act.nunique())


if __name__ == '__main__':
    main()
