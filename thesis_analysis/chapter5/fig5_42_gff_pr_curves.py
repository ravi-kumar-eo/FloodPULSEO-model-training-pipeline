"""Section 5.5.1 — Precision-recall curves pooled over the 21 expert-labelled sites: (a) U-Net-T15 and (b) ViT-T15,
each beside GFF U-TAE and GFF MaxViT U-Net, with the operating point of each model.

Curves: interpolated precision (the highest precision at a recall at least as large) at each recall, all scored cells
of the 21 sites pooled; U-Net-T15 and ViT-T15 = mean precision of seeds 1-3 at each recall; GFF from its water
probability, the mean over the 10 m pixels of each 80 m cell. Markers: operating points, ours at the
validation-selected global threshold (mean recall and precision of the three seeds), GFF at its released class decision
(an 80 m cell is flood when at least 32 of its 64 pixels are predicted water). Dotted line: flooded share of the scored
cells, the precision of a prediction without skill. T15 is the input-history length of the models of this thesis in
Section 5.5 (Ravi, 27 Oct 2026). Data: data/baseline/{pr_curves,pooled}.csv (ch5_baseline_cells_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_42_gff_pr_curves.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import baseline_common as bc
import rq1_common as rc
from rq1_common import ps

OURS = ('unet_t15', 'vit_t15')


def crossings(a, b):
    """Recall levels (0.05 to 0.95) where curve a moves above or below curve b."""
    d = np.sign(a.values - b.values)
    r = a.index.values
    return [round(r[i], 2) for i in range(1, len(d)) if d[i] != d[i - 1] and d[i] != 0 and 0.05 <= r[i] <= 0.95]


def main():
    rc.style()
    cu = pd.read_csv(bc.D / 'pr_curves.csv')
    po = pd.read_csv(bc.D / 'pooled.csv').groupby('group')[['precision', 'POD', 'IoU', 'AUPRC']].mean()
    p = pd.read_csv(bc.D / 'pooled.csv').iloc[0]
    share = (p.TP + p.FN) / (p.TP + p.FP + p.FN + p.TN)
    curve = {g: cu[cu.group == g].set_index('recall').precision for g in list(bc.GFF_MODELS) + list(OURS)}
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 3.2), sharey=True)
    for k, (ax, o) in enumerate(zip(axes, OURS)):
        for g in list(bc.GFF_MODELS) + [o]:
            col = bc.colour(g)
            ax.plot(curve[g].index, curve[g].values, color=col, lw=1.4, zorder=3)
            ax.scatter([po.loc[g, 'POD']], [po.loc[g, 'precision']], s=40, marker=bc.marker(g),
                       facecolor=ps.alpha_on_white(col, 0.75), edgecolor='black', lw=0.6, zorder=5)
        for g in bc.GFF_MODELS:
            print(f'{o} vs {g}: curves cross at recall {crossings(curve[o], curve[g])}')
        ax.axhline(share, color='black', lw=0.6, ls=(0, (1, 1.5)), zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Recall')
        if k == 0:
            ax.set_ylabel('Precision')
        rc.ygrid(ax)
        ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
        rc.panel_title(ax, f'({"ab"[k]}) {bc.LABEL[o]} and the GFF models')
    rc.yticks_everywhere(axes)
    for g in list(bc.GFF_MODELS) + list(OURS):
        r = curve[g]
        print(f'{g}: operating point recall {po.loc[g, "POD"]:.3f} precision {po.loc[g, "precision"]:.3f} IoU '
              f'{po.loc[g, "IoU"]:.3f}; precision at recall 0.2/0.45/0.8 {r.iloc[20]:.3f} {r.iloc[45]:.3f} {r.iloc[80]:.3f}')
    print(f'flooded share {share:.4f}')
    hl = [Line2D([], [], color=bc.colour(g), lw=1.4, marker=bc.marker(g), ms=5.5,
                 mfc=ps.alpha_on_white(bc.colour(g), 0.75), mec='black', mew=0.6, label=bc.LABEL[g])
          for g in list(bc.GFF_MODELS) + list(OURS)]
    hl.append(Line2D([], [], color='black', lw=0.6, ls=(0, (1, 1.5)), label='Flooded share of the scored cells'))
    fig.legend(handles=hl, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1.5, handlelength=2.0, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    ps.save(fig, __file__, subdir='5_5')


if __name__ == '__main__':
    main()
