"""Appendix A.5 — The ten large test mapping products with the largest IoU advantage of ViT-T0 over U-Net-T0 and the ten
with the largest advantage of U-Net-T0 over ViT-T0 (Ravi, 19-20 Oct 2026: "top 10 events where U-Net beats ViT and vice
versa", as in the early presentation).
Large = at least 5,000 flood cells of the label (32 km2) and 40 patches (Section 4.7.1). Product IoU per seed from
rq3/products.csv (T0 rows); dots = mean of seeds 1-3, lines = range of the three seeds. Panel (a) ViT ahead, (b) U-Net
ahead, each sorted by the difference, one product per AOI (the one with the largest difference). Data written to plot/data/tables/top10_products.csv.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_13_top10_products.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import event_maps as em
import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data'


def table():
    p = pd.read_csv(D / 'rq3/products.csv')
    p = p[p.window == 't0']
    s = p.pivot_table(index='folder_name', columns=['family', 'seed'], values='iou')
    pt = pd.read_csv(em.C / 'patches_test.csv.gz', usecols=['folder_name', 'flood'])
    g = pt.groupby('folder_name').agg(patches=('flood', 'size'), flood=('flood', 'sum'))
    t = pd.DataFrame({f'{f}_s{k}': s[(f, k)] for f in rc.FAMILIES for k in rc.SEEDS}).join(g)
    t = t[(t.flood >= 5000) & (t.patches >= 40)]
    for f in rc.FAMILIES:
        t[f] = t[[f'{f}_s{k}' for k in rc.SEEDS]].mean(axis=1)
    t['diff'] = t.vit - t.unet
    t['aoi'] = t.index.str.split('_').str[:2].str.join('_')        # one product per AOI: the largest difference
    vit = t.sort_values('diff', ascending=False).groupby('aoi').head(1).head(10).assign(side='ViT-T0 ahead')
    unet = t.sort_values('diff').groupby('aoi').head(1).head(10).assign(side='U-Net-T0 ahead')
    out = pd.concat([vit, unet])
    out['flood_km2'] = out.flood * 0.0064
    out.to_csv(D / 'tables/top10_products.csv')
    return vit, unet


def label(product):
    code, d = em.product_title(product)
    code = code.split(',')[0]
    day, month, year = d.split()
    return f'{code}, {day} {month[:3]} {year}'


def panel(ax, t, title):
    y = np.arange(len(t))[::-1]
    for f, dy in (('unet', 0.12), ('vit', -0.12)):
        lo = t[[f'{f}_s{k}' for k in rc.SEEDS]].min(axis=1)
        hi = t[[f'{f}_s{k}' for k in rc.SEEDS]].max(axis=1)
        ax.hlines(y + dy, lo, hi, color=rc.COLOR[f], lw=1.2, alpha=0.6)
        ax.plot(t[f], y + dy, 'o', color=rc.COLOR[f], ms=4, label=f'{rc.LABEL[f]}-T0')
    ax.set_yticks(y)
    ax.set_yticklabels([label(i) for i in t.index], fontsize=ps.FONT_SIZE - 0.5)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel('IoU')
    ax.grid(axis='x', color=ps.GREY_GRID, lw=0.5)
    ax.set_axisbelow(True)
    rc.panel_title(ax, title)


if __name__ == '__main__':
    rc.style()
    vit, unet = table()
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 6.4))
    panel(axes[0], vit, '(a) Largest IoU advantage of ViT-T0')
    panel(axes[1], unet, '(b) Largest IoU advantage of U-Net-T0')
    axes[0].legend(loc='lower right', frameon=False, fontsize=ps.FONT_SIZE - 0.5)
    fig.tight_layout()
    ps.save(fig, __file__, subdir='appendix')
    print(pd.concat([vit, unet])[['unet', 'vit', 'diff', 'flood']].round(3).to_string())
