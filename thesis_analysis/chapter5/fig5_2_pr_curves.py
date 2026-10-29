"""RQ1 — Precision-recall curves of U-Net-T0 and ViT-T0 on the test subset (micro, all scored cells pooled).

Each fit's curve comes from its counts of flood and dry cells in 1,000 probability bins (plot/data/rq1/hist.npz,
built and checked by rq1_data.py). Line: precision averaged over seeds 1-3 at each recall; light band: lowest to
highest seed. Marker: mean recall and precision of the three seeds at their validation-selected thresholds.
Triangle: mean recall and precision of the three seeds with the per-patch thresholds of the threshold head (Section
4.5.4; metrics.csv, test, full rows, rule 'head').
Dotted line: the flooded share of the scored cells, the precision of a prediction without skill.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_2_pr_curves.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import rq1_common as rc
from rq1_common import ps

GRID = np.linspace(0.02, 0.98, 97)


def curve(h, f, s):
    fl, dr = h[f'{f}_s{s}_flood'].astype(float), h[f'{f}_s{s}_dry'].astype(float)
    tp, fp = np.cumsum(fl[::-1])[::-1], np.cumsum(dr[::-1])[::-1]   # cells with p >= lower bin edge
    rec, prec = tp / fl.sum(), tp / np.maximum(tp + fp, 1)
    o = np.argsort(rec)
    return np.interp(GRID, rec[o], prec[o]), fl.sum() / (fl.sum() + dr.sum())


def head_point(f):
    """Mean recall and precision of the three seeds with the per-patch thresholds of the threshold head."""
    rp = []
    for s in rc.SEEDS:
        m = pd.read_csv(rc.RUNS / f'seed{s}/{f}/inference_final/metrics.csv')
        g = m[(m.cohort == 'test') & (m.rows == 'full') & (m.rule == 'head')].iloc[0]
        rp.append((g.pooled_recall, g.pooled_precision))
    return np.mean(rp, axis=0)


def main():
    rc.style()
    h = np.load(rc.DATA / 'hist.npz')
    fi = rc.fits().set_index(['family', 'seed'])
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, 2.6))
    for f in rc.FAMILIES:
        P = np.array([curve(h, f, s)[0] for s in rc.SEEDS])
        ax.fill_between(GRID, P.min(0), P.max(0), color=ps.alpha_on_white(rc.COLOR[f], 0.22), lw=0, zorder=1)
        ax.plot(GRID, P.mean(0), color=rc.COLOR[f], lw=1.2, zorder=3)
        r, p = fi.loc[f, 'recall'].mean(), fi.loc[f, 'precision'].mean()
        ax.scatter([r], [p], s=22, color=rc.COLOR[f], edgecolor='black', lw=0.6, zorder=4)
        hr, hp = head_point(f)
        ax.scatter([hr], [hp], s=26, marker='^', color=rc.COLOR[f], edgecolor='black', lw=0.6, zorder=4)
        i = np.argmin(np.abs(GRID - hr))
        print(f'{rc.LABEL[f]:5s} threshold head: recall {hr:.3f} precision {hp:.3f}; seed-mean curve at that recall '
              f'{np.interp(hr, GRID, P.mean(0)):.3f} (seed range {P[:, i].min():.3f}-{P[:, i].max():.3f})')
        for R in (0.3, 0.5, 0.7):
            i = np.argmin(np.abs(GRID - R))
            print(f'{rc.LABEL[f]:5s} precision at recall {R}: seeds {np.round(P[:, i], 3)}  mean {P[:, i].mean():.3f}')
        print(f'{rc.LABEL[f]:5s} operating point: recall {r:.3f} precision {p:.3f}; max precision on grid {P.mean(0).max():.3f}')
    base = curve(h, 'unet', 1)[1]
    ax.axhline(base, color='black', lw=0.6, ls=(0, (1, 1.5)), zorder=2)
    print(f'flooded share of scored cells {base:.4f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    rc.ygrid(ax); ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
    h_ = [Line2D([], [], color=rc.COLOR[f], lw=1.2, label=rc.LABEL[f]) for f in rc.FAMILIES] + \
         [Line2D([], [], ls='', marker='o', mfc='white', mec='black', mew=0.6, label='Selected threshold'),
          Line2D([], [], ls='', marker='^', mfc='white', mec='black', mew=0.6, label='Threshold head'),
          Line2D([], [], color='black', lw=0.6, ls=(0, (1, 1.5)), label='Flooded share of cells')]
    fig.legend(handles=h_, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.4, columnspacing=1.6)
    fig.tight_layout()
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
