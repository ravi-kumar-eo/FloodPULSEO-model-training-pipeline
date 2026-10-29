"""Section 5.6 — POD and precision within a spatial tolerance (Ravi, 19 Oct 2026: POD and precision replace the miss
rate and the false alarm ratio, which are one minus them; precision may be used where a figure needs it).

A predicted flood cell counts as correct when a reference flood cell lies within d cells (chessboard distance on the
mosaic of its mapping product, one cell = 80 m), and a reference flood cell counts as detected when a predicted flood
cell of the same fit lies within d cells; d = 0 gives the scores of the earlier sections. (a) precision,
(b) POD, against the tolerance in metres; mean of seeds 1-3; T0 dashed, T30 solid.
Data: data/uncertainty/tolerance.csv (ch5_uncertainty_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_48_tolerance.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import rq1_common as rc
import unc_common as uc
from rq1_common import ps


def main():
    rc.style()
    t = uc.read('tolerance')
    t['precision'] = t.pred_near / t.pred
    t['pod'] = t.ref_near / t.ref
    m = t.groupby(['family', 'window', 'd'])[['precision', 'pod']].mean()
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 2.7))
    for ax, (k, title) in zip(axes, (('precision', '(a) Precision'), ('pod', '(b) POD'))):
        for f, w in uc.CONFIGS:
            g = m.loc[(f, w)][k]
            ax.plot(80 * g.index.to_numpy(), g.to_numpy(), color=rc.COLOR[f], lw=1.1, marker='o', ms=3,
                    ls=(0, (3, 2)) if w == 't0' else '-', mfc='white' if w == 't0' else rc.COLOR[f],
                    mec=rc.COLOR[f], mew=0.7, zorder=3)
            print(uc.CLABEL[(f, w)], k, np.round(g.to_numpy(), 3))
        ax.set_xticks(80 * np.arange(5))
        ax.set_xlabel('Tolerance (m)')
        ax.set_ylim(0, 1)
        rc.ygrid(ax)
        rc.panel_title(ax, title)
    h = [Line2D([], [], color=rc.COLOR[f], lw=1.1, marker='o', ms=3, ls=(0, (3, 2)) if w == 't0' else '-',
                mfc='white' if w == 't0' else rc.COLOR[f], mec=rc.COLOR[f], label=uc.CLABEL[(f, w)])
         for f, w in uc.CONFIGS]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=4, frameon=False,
               fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2.0)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
