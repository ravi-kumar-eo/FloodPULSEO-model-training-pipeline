"""Appendix A.5 — Event maps of selected test mapping products (Ravi, 20 Oct 2026: show important, large floods that the
networks predict well, IoU at least 0.3, and trim the former atlas of twenty products).

Selection (rules fixed before drawing; Section 4.7.1):
  * large test products (at least 5,000 flood cells of the label and 40 patches) whose IoU, the mean of seeds 1-3, is at
    least 0.3 for all four of U-Net-T0, ViT-T0, U-Net-T30 and ViT-T30; one product per test activation, the one with the
    largest labelled flooded area; AOIs already mapped in Chapter 5 are left out;
  * for contrast, the large product with the lowest IoU (mean of the four networks) among the largest products of the
    twenty test activations with the largest labelled flooded area.
Each product one figure (Figures A.8 onwards, files figA_<n>_event.png), drawn with event_maps.draw_product as the chapter
maps: flood label, U-Net-T0, U-Net-T30, ViT-T0, ViT-T30, cells predicted by at least two of the three seeds, IoU, POD and
POFD under every map. Table of all ten configurations per product: event_maps.metric_table.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_8_event_atlas.py
"""
from pathlib import Path

import pandas as pd

import event_maps as em

D = Path(__file__).resolve().parents[1] / 'data'
COLUMNS = (None, ('unet', 't0'), ('unet', 't30'), ('vit', 't0'), ('vit', 't30'))
TITLES = ('Flood label', 'U-Net-T0', 'U-Net-T30', 'ViT-T0', 'ViT-T30')
CH5_AOIS = ('EMSR692_AOI03', 'EMSR629_AOI02', 'EMSR416_AOI02', 'EMSR788_AOI06', 'EMSR708_AOI01', 'EMSR417_AOI01',
            'EMSR848_AOI03', 'EMSR466_AOI01')


def products():
    p = pd.read_csv(D / 'rq3/products.csv')
    m = p.assign(cfg=p.family + '_' + p.window).groupby(['folder_name', 'cfg']).iou.mean().unstack()
    m = m[['unet_t0', 'vit_t0', 'unet_t30', 'vit_t30']]
    pt = pd.read_csv(em.C / 'patches_test.csv.gz', usecols=['folder_name', 'flood'])
    g = pt.groupby('folder_name').agg(patches=('flood', 'size'), flood=('flood', 'sum'))
    m = m.join(g)
    m['act'] = m.index.str.split('_').str[0]
    m['aoi'] = m.index.str.split('_').str[:2].str.join('_')
    large = m[(m.flood >= 5000) & (m.patches >= 40)]
    good = large[(large[['unet_t0', 'vit_t0', 'unet_t30', 'vit_t30']].min(axis=1) >= 0.3) & ~large.aoi.isin(CH5_AOIS)]
    good = good.sort_values('flood', ascending=False).groupby('act').head(1)
    top = m[m.flood > 0].sort_values('flood', ascending=False).groupby('act').head(1).head(20)
    top = top[(top.flood >= 5000) & (top.patches >= 40)]
    contrast = top[['unet_t0', 'vit_t0', 'unet_t30', 'vit_t30']].mean(axis=1).idxmin()
    return list(good.index) + [contrast], m


if __name__ == '__main__':
    ev, m = products()
    rows = []
    for k, e in enumerate(ev):
        em.draw_product(e, COLUMNS, ('Flood label',) + TITLES[1:], Path(__file__).with_name(f'figA_{8 + k}_event.py'),
                        subdir='appendix')
        w, h = em.size_km(e)
        rows.append({'figure': f'A.{8 + k}', 'product': e, 'width_km': round(w), 'height_km': round(h),
                     'flood_km2': round(m.loc[e, 'flood'] * 0.0064, 1),
                     **{f'iou_{t}': round(m.loc[e, c], 3) for t, c in zip(TITLES[1:], ('unet_t0', 'unet_t30', 'vit_t0', 'vit_t30'))}})
        print('\n'.join(em.metric_table(e, COLUMNS[1:])))
    pd.DataFrame(rows).to_csv(D / 'event_atlas.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
