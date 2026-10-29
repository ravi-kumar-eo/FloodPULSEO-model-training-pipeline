"""RQ1 — Paired IoU of the same mapping products: U-Net-T0 against ViT-T0.

(a) IoU of each flood-positive mapping product with U-Net-T0 (x) and ViT-T0 (y), each averaged over seeds 1-3,
    with the 1:1 line. (b) Histogram of the difference ViT - U-Net of the seed means, bins of 0.01 centred on whole hundredths.
Colour: the family with the higher seed-mean IoU of the product (grey: equal, all of them IoU 0 with both).
Test subset, full rows, flood-only cells, each fit at its validation-selected global threshold.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_5_paired_products.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

MIXED = ps.GREY_OTHER


def main():
    rc.style()
    c = rc.product_counts()
    c['iou'] = c.tp / (c.tp + c.fp + c.fn)
    W = c.pivot_table(index='folder_name', columns=['family', 'seed'], values='iou')
    u, v = W['unet'], W['vit']
    um, vm = u.mean(axis=1), v.mean(axis=1)
    d = vm - um
    cat = np.where(d > 0, 'vit', np.where(d < 0, 'unet', 'mixed'))
    col = {'vit': rc.COLOR['vit'], 'unet': rc.COLOR['unet'], 'mixed': MIXED}

    fig, (a, b) = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 2.75), gridspec_kw={'width_ratios': [1, 1]})
    for k in ('mixed', 'unet', 'vit'):
        m = cat == k
        a.scatter(um[m], vm[m], s=6, color=col[k], alpha=0.7, lw=0, zorder=3)
    a.plot([0, 1], [0, 1], color='black', lw=0.6, ls=(0, (3, 2)), zorder=2)
    a.set_xlim(0, 0.92); a.set_ylim(0, 0.92); a.set_box_aspect(1)
    a.set_xlabel('U-Net IoU'); a.set_ylabel('ViT IoU')
    rc.ygrid(a); a.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
    rc.panel_title(a, '(a) IoU per product')

    k = np.ceil(np.abs(d).max() * 100)               # bins of 0.01 centred on whole hundredths, 0 in the middle of one
    bins = (np.arange(-k, k + 2) - 0.5) / 100
    b.hist([d[cat == k] for k in ('mixed', 'unet', 'vit')], bins=bins, stacked=True,
           color=[col[k] for k in ('mixed', 'unet', 'vit')], alpha=0.8, lw=0, zorder=2)
    b.axvline(0, color='black', lw=0.6, zorder=3)
    b.set_box_aspect(1)
    b.set_xlabel('ViT − U-Net IoU'); b.set_ylabel('Products')
    rc.ygrid(b)
    rc.panel_title(b, '(b) Difference per product')

    h = [Patch(color=col['vit'], label='ViT higher'), Patch(color=col['unet'], label='U-Net higher'),
         Patch(color=MIXED, label='Equal')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.0, columnspacing=1.6)
    fig.tight_layout(w_pad=1.5)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    print(f'products {len(d)}; ViT higher {(cat == "vit").sum()}, U-Net higher {(cat == "unet").sum()}, '
          f'equal {(cat == "mixed").sum()}; median difference {d.median():+.4f}; |d|<=0.01 {(d.abs() <= 0.01).sum()}, '
          f'|d|<=0.05 {(d.abs() <= 0.05).sum()}; range {d.min():+.3f} to {d.max():+.3f}')


if __name__ == '__main__':
    main()
