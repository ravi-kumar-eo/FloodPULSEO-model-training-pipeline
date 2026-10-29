"""RQ2 — HydroBASINS Level-5 basins of the basin-seen and basin-unseen test spatial blocks (rq2_basins.gpkg,
ch5_regime_basins.py), and below the map the share of the test patches of each regime by continent
(rq2_composition.csv). The partial spatial block is left out.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_12_regime_continents.py
"""
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import regime_common as rg
import rq1_common as rc
from rq1_common import ps

CONTINENTS = ['Europe', 'Asia', 'Africa', 'South America', 'North America', 'Australia/Oceania']


def main():
    rc.style()
    comp = pd.read_csv(rg.D / 'rq2_composition.csv').set_index('regime')
    W = ps.TEXT_WIDTH_IN
    fig = plt.figure(figsize=(W, W * 0.74))
    ax = rg.world_axes(fig, [0, 0.355, 1, 0.61])
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='#F3F1EC', edgecolor='none'),
                   zorder=1)
    rg.draw_basins(ax)
    rc.panel_title(ax, '(a) Level-5 basins of the test spatial blocks')
    bx = fig.add_axes([0.075, 0.135, 0.915, 0.17])
    x = np.arange(len(CONTINENTS))
    for r, dx in (('seen', -0.19), ('unseen', 0.19)):
        v = 100 * comp.loc[r, [f'continent_{c}' for c in CONTINENTS]].to_numpy(float)
        bx.bar(x + dx, v, 0.36, color=rg.face(r), edgecolor=rg.REG_COLOR[r], lw=0.6, zorder=2)
        for xi, vi in zip(x + dx, v):
            bx.text(xi, vi, f'{vi:.1f}', ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5)
    bx.set_xticks(x, [c.replace('Australia/Oceania', 'Australia and\nOceania').replace('h A', 'h\nA') for c in CONTINENTS])
    bx.tick_params(axis='x', length=0)
    bx.set_ylim(0, 112)
    bx.set_ylabel('Test patches (%)')
    rc.ygrid(bx)
    rc.panel_title(bx, '(b) Share of the test patches by continent')
    rg.legend(fig, y=0.045)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
