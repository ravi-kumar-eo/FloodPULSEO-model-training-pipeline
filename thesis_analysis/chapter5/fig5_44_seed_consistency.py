"""Section 5.5 — Training variation against test-sample variation for every comparison of Chapter 5.

Rows: ViT minus U-Net at T0 and T30 (seeds matched), and every input-history length minus T0 for U-Net and ViT (paired
by seed). Diamond: mean of the three seed differences; line: its 95% cluster bootstrap interval (61 clusters, 4,000
replicates, the same draws for every fit); open circles: the difference of each seed. (a) micro IoU, (b) macro IoU.
Right of each row: the number of seeds whose difference has the sign of the mean. Data: data/uncertainty/consistency.csv
(ch5_uncertainty_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_44_seed_consistency.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import rq1_common as rc
import unc_common as uc
from rq1_common import ps

ROWS = ([('vit-unet t0', 'both', 'ViT − U-Net, T0'), ('vit-unet t30', 'both', 'ViT − U-Net, T30')]
        + [(f'{w}-t0', f, f'{rc.LABEL[f]}, {w.upper()} − T0') for f in rc.FAMILIES for w in ('t3', 't7', 't15', 't30')])
NEUTRAL = '#4D4D4D'


def main():
    rc.style()
    c = uc.read('consistency')
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 3.5), sharey=True)
    y = np.arange(len(ROWS))[::-1].astype(float)
    y[:2] += 0.5                                       # small gap after the family comparisons
    for ax, (m, title) in zip(axes, (('micro_iou', '(a) Micro IoU difference'), ('macro_iou', '(b) Macro IoU difference'))):
        for yi, (comp, fam, _) in zip(y, ROWS):
            g = c[(c.comparison == comp) & (c.family == fam) & (c.measure == m)]
            mean = g[g.seed == 'mean'].iloc[0]
            seeds = g[g.seed != 'mean'].change.to_numpy()
            col = NEUTRAL if fam == 'both' else rc.COLOR[fam]
            ax.plot([mean.lo, mean.hi], [yi, yi], color=col, lw=1.2, solid_capstyle='butt', zorder=2)
            ax.scatter(seeds, [yi] * 3, s=14, facecolor='white', edgecolor=col, lw=0.8, zorder=3)
            ax.scatter([mean.change], [yi], s=30, marker='D', color=col, edgecolor='black', lw=0.5, zorder=4)
            ax.annotate(f'{int((np.sign(seeds) == np.sign(mean.change)).sum())}/3', (1.0, yi), xycoords=('axes fraction', 'data'), xytext=(4, 0),
                        textcoords='offset points', va='center', ha='left', fontsize=ps.FONT_SIZE - 2.5)
        ax.axvline(0, color='black', lw=0.6, zorder=1)
        ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
        ax.set_axisbelow(True)
        rc.panel_title(ax, title)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, [r[2] for r in ROWS], fontsize=ps.FONT_SIZE - 1.5)
    axes[0].set_ylim(y.min() - 0.7, y.max() + 0.7)
    h = [Line2D([], [], ls='', marker='D', mfc=NEUTRAL, mec='black', mew=0.5, label='Seed mean'),
         Line2D([], [], color=NEUTRAL, lw=1.2, label='95% cluster interval of the mean'),
         Line2D([], [], ls='', marker='o', mfc='white', mec=NEUTRAL, mew=0.8, label='Seed 1, 2, 3')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.55, 0.04), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.05, 0.97, 1), w_pad=2.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    for comp, fam, lab in ROWS:
        for m in ('micro_iou', 'macro_iou'):
            g = c[(c.comparison == comp) & (c.family == fam) & (c.measure == m)]
            mean = g[g.seed == 'mean'].iloc[0]
            print(f'{lab:22s} {m}: mean {mean.change:+.4f} [{mean.lo:+.4f}, {mean.hi:+.4f}]  seeds '
                  f'{np.round(g[g.seed != "mean"].change.to_numpy(), 4)}  pairs {mean.pairs_higher}')


if __name__ == '__main__':
    main()
