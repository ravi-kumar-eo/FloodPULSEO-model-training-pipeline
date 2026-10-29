"""RQ2 — HydroBASINS Level-5 basins of the test spatial blocks of the two basin regimes (rq2_basins.gpkg, in the darker
regime colours) on the main Köppen–Geiger climate classes (Beck et al., 2018,
present-day map at 0.083°, data/koppen/Beck_KG_V1_present_0p083.tif; the dataset assigns each mapping product the most
frequent main class of the 1 km map within its AOI), with the share of the test patches of each regime by climate class
(rq2_composition.csv) drawn on the map. The partial spatial block is left out.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_13_regime_climates.py
"""
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import regime_common as rg
import rq1_common as rc
from rq1_common import ps

KG = Path(__file__).resolve().parents[1] / 'data' / 'koppen' / 'Beck_KG_V1_present_0p083.tif'
# no green or purple, which are the regime colours (Ravi, 13 Oct: cold purple hid the basin-unseen basins)
CLIMATES = [('A Tropical', 'Tropical', '#E8935A', (1, 3)), ('B Arid', 'Arid', '#F1D27A', (4, 7)),
            ('C Temperate', 'Temperate', '#8DC3E0', (8, 16)), ('D Cold', 'Cold', '#4F86B8', (17, 28)),
            ('E Polar', 'Polar', '#C8C8C8', (29, 30))]
MAP_FILL = 0.75                                                  # Ravi: 0.4 was washed out (13 Oct)


def main_classes():
    with rasterio.open(KG) as src:
        kg = src.read(1)
        b = src.bounds
    out = np.zeros(kg.shape, np.uint8)
    for i, (_, _, _, (lo, hi)) in enumerate(CLIMATES, start=1):
        out[(kg >= lo) & (kg <= hi)] = i
    return np.ma.masked_equal(out, 0), (b.left, b.right, b.bottom, b.top)


def main():
    rc.style()
    comp = pd.read_csv(rg.D / 'rq2_composition.csv').set_index('regime')
    W = ps.TEXT_WIDTH_IN
    fig = plt.figure(figsize=(W, W * 0.68))
    ax = rg.world_axes(fig, [0, 0.30, 1, 0.66])
    cls, extent = main_classes()
    cmap = ListedColormap([ps.alpha_on_white(c, MAP_FILL) for _, _, c, _ in CLIMATES])
    ax.imshow(cls, origin='upper', extent=extent, transform=ccrs.PlateCarree(), cmap=cmap, vmin=0.5,
              vmax=len(CLIMATES) + 0.5, interpolation='nearest', regrid_shape=3000, zorder=1)
    rg.draw_basins(ax, dark=True)
    cols = [c for _, _, c, _ in CLIMATES]
    rc.panel_title(ax, '(a) Level-5 basins of the test spatial blocks on the main climate classes')
    handles = [Patch(facecolor=rg.REG_DARK[r], edgecolor='black', lw=0.3, label=rg.REG_LABEL[r]) for r in rg.REG]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.245), ncol=2, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handletextpad=0.2, columnspacing=1.6)
    bx = fig.add_axes([0.2, 0.105, 0.72, 0.105])                    # below the map
    rg.share_bars(bx, comp, [f'climate_{k}' for k, _, _, _ in CLIMATES], cols, fill=MAP_FILL)
    bx.set_xticks([0, 20, 40, 60, 80, 100], ['0', '20', '40', '60', '80', '100'], fontsize=ps.FONT_SIZE - 2)
    bx.set_yticklabels([rg.REG_LABEL[r] for r in rg.REG], fontsize=ps.FONT_SIZE - 1)
    rc.panel_title(bx, '(b) Share of the test patches (%) by climate class')
    h2 = [Patch(color=ps.alpha_on_white(c, MAP_FILL), label=l) for _, l, c, _ in CLIMATES]
    fig.legend(handles=h2, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=0.9, handletextpad=0.4, columnspacing=1.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
