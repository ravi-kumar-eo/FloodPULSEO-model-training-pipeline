"""RQ3 — Change from T0 to T30 by the number of IMERG cells (0.1 degree, about 10 km) that the test patches of a
mapping product occupy (data: ch5_rq3_aoi_size_data.py -> data/rq3/aoi_size.csv).

Products are grouped into 1, 2-3, 4-10 and more than 10 IMERG cells (the number of products under each label). Per
group, pooled over the scored cells of its products and averaged over seeds 1-3: (a) change in predicted flood cells
(%), (b) change in POD, (c) in POFD and (d) in micro IoU, T30 minus T0, paired by seed; whiskers: 95% cluster bootstrap
interval. Every panel has its own x-axis (Ravi, 15 Oct 2026).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_22_t30_aoi_size.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data' / 'rq3'
PANELS = (('dpred', '(a) Change in predicted flood cells (%)', '{:+.0f}'), ('dpod', '(b) Change in POD', '{:+.3f}'),
          ('dpofd', '(c) Change in POFD', '{:+.3f}'), ('diou', '(d) Change in micro IoU', '{:+.3f}'))


def main():
    rc.style()
    a = pd.read_csv(D / 'aoi_size.csv')
    a = a[a.group == 'imerg_cells'].reset_index(drop=True)
    x = np.arange(len(a))
    labels = [f'{c}\n({n})' for c, n in zip(a['class'], a.products)]
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH_IN, 4.6))
    for ax, (k, title, fmt) in zip(axes.ravel(), PANELS):
        span = max(a[[f'{f}_{k}_hi' for f in ('unet', 'vit')]].abs().max().max(),
                   a[[f'{f}_{k}_lo' for f in ('unet', 'vit')]].abs().max().max())
        for f, dx in (('unet', -0.19), ('vit', 0.19)):
            v, lo, hi = a[f'{f}_{k}'], a[f'{f}_{k}_lo'], a[f'{f}_{k}_hi']
            ax.bar(x + dx, v, 0.36, color=ps.alpha_on_white(rc.COLOR[f], 0.8), edgecolor=rc.COLOR[f], lw=0.6, zorder=2)
            ax.errorbar(x + dx, v, yerr=[v - lo, hi - v], fmt='none', ecolor='black', elinewidth=0.7, capsize=1.5,
                        zorder=3)
            for xi, vi, l, h in zip(x + dx, v, lo, hi):
                y, va = (h + 0.04 * span, 'bottom') if vi >= 0 else (l - 0.04 * span, 'top')
                t = fmt.format(vi)
                ax.text(xi, y, t.replace('+0.', '+.').replace('-0.', '−.').replace('-', '−'), ha='center', va=va,
                        fontsize=ps.FONT_SIZE - 3.5, zorder=4)
        ax.axhline(0, color='black', lw=0.6, zorder=1)
        ax.set_ylim(-1.35 * span, 1.35 * span)
        ax.set_xticks(x, labels, fontsize=ps.FONT_SIZE - 2)
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_xlabel('IMERG cells of the mapping product (products)', fontsize=ps.FONT_SIZE - 2)
        rc.ygrid(ax)
        rc.panel_title(ax, title)
    h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], 0.8), edgecolor=rc.COLOR[f], lw=0.6, label=rc.LABEL[f])
         for f in ('unet', 'vit')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=2, frameon=False,
               fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2, w_pad=1.5)
    ps.save(fig, __file__, subdir='5_3')


if __name__ == '__main__':
    main()
