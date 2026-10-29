"""Section 5.5 — Calibration of the flood probabilities of U-Net and ViT without (T0) and with (T30) the 30-day input
history.

Scored test cells of seeds 1-3 pooled, in 20 probability bins of width 0.05. Top: observed flooded share of the cells of
each bin against their mean predicted probability (reliability curve), T0 dashed and T30 solid; the diagonal marks
probabilities equal to the observed frequency. Bottom: share of the scored cells in each bin, logarithmic.
Bins with fewer than 1,000 cells are not drawn. Data: data/uncertainty/calibration.csv (ch5_uncertainty_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_46_calibration.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import rq1_common as rc
import unc_common as uc
from rq1_common import ps

STYLE = {'t0': dict(ls=(0, (3, 2)), marker='o', mfc='white'), 't30': dict(ls='-', marker='o')}


def main():
    rc.style()
    cal = uc.read('calibration').groupby(['family', 'window', 'bin'])[['cells', 'flood', 'psum', 'lo', 'hi']].sum()
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH_IN, 4.0), gridspec_kw={'height_ratios': [3, 1.3]},
                             sharex=True)
    for j, f in enumerate(rc.FAMILIES):
        ax, hx = axes[0, j], axes[1, j]
        ax.plot([0, 1], [0, 1], color='black', lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        for w in ('t0', 't30'):
            g = cal.loc[(f, w)]
            ok = g.cells >= 1000
            x, yv = (g.psum / g.cells)[ok], (g.flood / g.cells)[ok]
            st = STYLE[w]
            ax.plot(x, yv, color=rc.COLOR[f], lw=1.1, ms=3, mec=rc.COLOR[f], mew=0.7,
                    **{k: v for k, v in st.items() if k != 'mfc'}, mfc=st.get('mfc', rc.COLOR[f]), zorder=3)
            hx.plot((np.arange(20) + 0.5) / 20, 100 * g.cells / g.cells.sum(), color=rc.COLOR[f], lw=1.0, ms=2.5,
                    mec=rc.COLOR[f], mew=0.6, **{k: v for k, v in st.items() if k != 'mfc'},
                    mfc=st.get('mfc', rc.COLOR[f]), zorder=3)
            print(f'{rc.LABEL[f]} {w}: observed {np.round(yv.to_numpy(), 3)}  mean p {np.round(x.to_numpy(), 3)}')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('auto')
        rc.ygrid(ax); ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
        rc.panel_title(ax, f'({"ab"[j]}) {rc.LABEL[f]}')
        hx.set_yscale('log')
        rc.ygrid(hx); hx.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
        hx.set_xlabel('Predicted flood probability')
    axes[0, 0].set_ylabel('Flooded share in the label')
    axes[1, 0].set_ylabel('Cells (%)')
    h = [Line2D([], [], color='black', lw=1.1, ls=(0, (3, 2)), marker='o', ms=3, mfc='white', label='T0'),
         Line2D([], [], color='black', lw=1.1, marker='o', ms=3, label='T30'),
         Line2D([], [], color='black', lw=0.6, ls=(0, (1, 1.5)), label='Perfect calibration')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6, w_pad=1.5)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
