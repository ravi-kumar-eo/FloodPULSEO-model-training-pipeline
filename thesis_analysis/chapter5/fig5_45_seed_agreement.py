"""Section 5.5 — Agreement between the three seeds of a configuration, cell by cell.

For every scored test cell, the number of seeds (0-3) of U-Net-T0, U-Net-T30, ViT-T0 and ViT-T30 that predict flood at
their selected thresholds. (a) Cells predicted by one, two or three seeds, as a share of the cells predicted by at least
one seed. (b) Share of these cells that are reference flood (precision), with the flooded share of the scored cells as
dotted line. (c) Reference flood cells by the number of seeds that detect them. T0 hatched.
Data: data/uncertainty/votes.csv (ch5_uncertainty_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_45_seed_agreement.py
"""
import matplotlib.pyplot as plt
import numpy as np

import rq1_common as rc
import unc_common as uc
from rq1_common import ps


def main():
    rc.style()
    v = uc.read('votes').set_index(['family', 'window', 'votes'])
    share, prec, ref = {}, {}, {}
    for cfg in uc.CONFIGS:
        g = v.loc[cfg]
        pred = g.loc[[1, 2, 3]]
        share[cfg] = 100 * pred.cells / pred.cells.sum()
        prec[cfg] = pred.flood / pred.cells
        ref[cfg] = 100 * g.flood / g.flood.sum()
    base = v.flood.sum() / v.cells.sum()
    fig, axes = plt.subplots(3, 1, figsize=(ps.TEXT_WIDTH_IN, 5.6))
    uc.grouped_bars(axes[0], ['1', '2', '3'], share)
    uc.grouped_bars(axes[1], ['1', '2', '3'], prec, fmt='{:.2f}')
    axes[1].axhline(base, color='black', lw=0.6, ls=(0, (1, 1.5)), zorder=3)
    uc.grouped_bars(axes[2], ['0', '1', '2', '3'], ref)
    for ax, t in zip(axes, ('(a) Cells predicted as flood by one, two or three seeds (% of the cells predicted by any seed)',
                            '(b) Share of these cells that is flood in the label (precision)',
                            '(c) Flood cells of the label by the number of seeds that detect them (%)')):
        rc.panel_title(ax, t)
        ax.set_xlabel('Seeds predicting flood', fontsize=ps.FONT_SIZE - 1.5)
    fig.legend(handles=uc.legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=False,
               fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.0)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    for cfg in uc.CONFIGS:
        print(uc.CLABEL[cfg], 'share', np.round(share[cfg].to_numpy(), 1), 'precision', np.round(prec[cfg].to_numpy(), 3),
              'reference', np.round(ref[cfg].to_numpy(), 1))
    print('flooded share', round(base, 4))


if __name__ == '__main__':
    main()
