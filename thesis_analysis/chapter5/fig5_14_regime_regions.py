"""RQ2 — Basin-seen against basin-unseen test ground within Europe, the temperate and cold climates and the tropical climates
(rq2_regions.csv, ch5_regime_regions.py): IoU, POD and POFD of U-Net-T0 and ViT-T0 (seed means, pooled over
the scored cells of each group), and the flooded share of the scored cells. The partial spatial block is left out.
Ravi, 19 Oct 2026: the area-bias column is removed (IoU, POD and POFD only).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_14_regime_regions.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import regime_common as rg
import rq1_common as rc
from rq1_common import ps

GROUPS = (('Europe', 'Europe'), ('Temperate and cold', 'Temperate and cold'), ('Tropical', 'Tropical'))
METRICS = (('iou', 'IoU', None), ('pod', 'POD', None), ('pofd', 'POFD', None))


def main():
    rc.style()
    d = pd.read_csv(rg.D / 'rq2_regions.csv').set_index('group')
    fig, axes = plt.subplots(3, 4, figsize=(ps.TEXT_WIDTH_IN, 5.7), sharey='col',
                             gridspec_kw={'width_ratios': [2, 2, 2, 1.15]})
    tops = {col: 1.16 * max(d.loc[g, f'{f}_{col}_{r}'] for g, _ in GROUPS for f in rc.FAMILIES for r in rg.REG)
            for col, _, _ in METRICS}                                  # one value scale per column, from the data
    flood_top = 1.16 * 100 * max(d.loc[g, f'{r}_flood_fraction'] for g, _ in GROUPS for r in rg.REG)
    k = 0
    for i, (g, glab) in enumerate(GROUPS):
        row = d.loc[g]
        for j, (col, lab, ref) in enumerate(METRICS):
            ax = axes[i, j]
            vals = {(f, r): row[f'{f}_{col}_{r}'] for f in rc.FAMILIES for r in rg.REG}
            rg.family_bars(ax, vals, ref=ref, top=tops[col])
            rc.panel_title(ax, f'({"abcdefghijkl"[k]}) {lab}')
            k += 1
        ax = axes[i, 3]                                           # flooded share of the scored cells, per regime
        for x, r in enumerate(rg.REG):
            v = 100 * row[f'{r}_flood_fraction']
            ax.bar(x, v, 0.62, color=rg.face(r), edgecolor=rg.REG_COLOR[r], lw=0.6, zorder=2)
            ax.text(x, v, f'{v:.1f}', ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5)
        ax.set_xticks([0, 1], ['', ''])
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, flood_top)
        rc.ygrid(ax)
        rc.panel_title(ax, f'({"abcdefghijkl"[k]}) Flooded (%)')
        k += 1
        axes[i, 0].set_ylabel(f"{glab}\n({int(row['seen_blocks_flood'])} / {int(row['unseen_blocks_flood'])} blocks)",
                              fontsize=ps.FONT_SIZE)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.0, w_pad=0.6)
    rg.legend(fig, y=0.03)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
