"""Flood maps at three of the 21 expert-labelled sites of the comparison with the Global Flood Forecasting (GFF)
models: flood label, GFF U-TAE, GFF MaxViT U-Net, U-Net-T15 and ViT-T15 (Section 5.5.1).

Sites chosen by rule (plot/data/baseline/sites.csv): the site with the most labelled flood cells; the site where the
better of U-Net-T15 and ViT-T15 exceeds the better GFF model by the largest IoU margin; the site where the better GFF
model exceeds the better of U-Net-T15 and ViT-T15 by the largest margin. T15 is the input-history length of the models
of this thesis in Section 5.5 (Ravi, 27 Oct 2026). Each is drawn as the majority of seeds 1-3 at the
validation-selected thresholds, and the evaluation metrics under each map are those of the drawn map. Sites are named by
country, date and their number within the flood event (no dataset identifiers). Source (read only):
experiments/runs_baselines/gff/paired_maps_all/<site>.npz (80 m cells scored in Section 5.5: cls 1 flood, 0 dry,
2 permanent water or mixed, -1 not evaluated).
"""
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import baseline_common as bc
import event_maps as em
import rq1_common as rc
from rq1_common import ps

SRC = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_baselines/gff/paired_maps_all')
SITES = Path(__file__).resolve().parents[1] / 'data' / 'baseline' / 'sites.csv'
LABEL = {'unet': 'U-Net', 'vit': 'ViT'}


def choose():
    d = pd.read_csv(SITES)
    d = d[d.label_type == 'hand']
    piv = d.pivot_table(index='site', columns='group', values='IoU')
    fc = d.groupby('site').flood_cells.first()
    order = bc.site_order(d)
    names = dict(zip(order.site, bc.site_names(order, long=True)))
    gap = piv[['unet_t15', 'vit_t15']].max(axis=1) - piv[['gff_utae', 'gff_maxvit']].max(axis=1)
    picks = [fc.idxmax(), gap.idxmax(), gap.idxmin()]
    return [(s, 'unet_t15', 'vit_t15', names[s]) for s in picks]


def majority(z, cfg, thr):
    arch, w = cfg.split('_')
    v = [z[f'prob_final_{w}__s{s}__{arch}'].astype(np.float32) >= thr[f'final_{w}__s{s}__{arch}'] for s in (1, 2, 3)]
    return np.sum(v, axis=0) >= 2


def image(cls, yes=None):
    img = np.ones(cls.shape + (3,))
    img[cls == 2] = to_rgb(em.COL['pw'])
    if yes is None:
        img[cls == 1] = to_rgb(em.COL['tp'])
        img[cls == 0] = to_rgb(em.COL['dry'])
    else:
        img[(cls == 1) & yes] = to_rgb(em.COL['tp'])
        img[(cls == 1) & ~yes] = to_rgb(em.COL['fn'])
        img[(cls == 0) & yes] = to_rgb(em.COL['fp'])
        img[(cls == 0) & ~yes] = to_rgb(em.COL['dry'])
    return img


def metrics(cls, yes):
    y = cls[(cls == 0) | (cls == 1)]
    p = yes[(cls == 0) | (cls == 1)]
    tp, fp, fn, tn = np.sum(p & (y == 1)), np.sum(p & (y == 0)), np.sum(~p & (y == 1)), np.sum(~p & (y == 0))
    return tp / max(tp + fp + fn, 1), tp / max(tp + fn, 1), fp / max(fp + tn, 1)


def main():
    picks = choose()
    rows = []
    for site, cu, cv, name in picks:
        z = np.load(SRC / f'{site}.npz', allow_pickle=True)
        thr = json.loads(str(z['thr']))
        cls = z['cls']
        r, c = np.where(cls >= 0)
        sl = (slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1))
        cls = cls[sl]
        maps = {'gff_utae': z['gff_utae'][sl].astype(bool), 'gff_metnet': z['gff_metnet'][sl].astype(bool),
                'unet': majority(z, cu, thr)[sl], 'vit': majority(z, cv, thr)[sl]}
        rows.append(dict(site=site, name=name, cls=cls, maps=maps, cfg={'unet': bc.LABEL[cu], 'vit': bc.LABEL[cv]}))
        print(site, name, cu, cv, {k: tuple(round(float(x), 3) for x in metrics(cls, v)) for k, v in maps.items()})

    # One page per site (Ravi, 29 Oct 2026: the 15 maps on one page were too small), suffix _1 to _3; the LaTeX
    # figure number is marked A, B, C on the three pages, each with panels (a) to (e).
    ps.apply_clear()
    width, gap, side = ps.TEXT_WIDTH_IN, 0.25, 0.05
    pw = (width - 2 * side - gap) / 2
    head_h, tit_h, below_h = 0.30, 0.22, (0.40, 0.40, 0.40)
    for n, r in enumerate(rows, 1):
        letters = iter('abcde')
        H, W = r['cls'].shape
        h = min(pw * H / W, 1.9)
        w_ = h * W / H
        fig_h = head_h + 3 * (tit_h + h) + sum(below_h) + 0.05
        fig = plt.figure(figsize=(width, fig_h))
        fig.text(side / width, (fig_h - 0.04) / fig_h, r['name'], ha='left', va='top', fontsize=ps.FONT_SIZE,
                 weight='bold')
        grid = [[('label', 'Flood label'), None], [('gff_utae', 'GFF U-TAE'), ('gff_metnet', 'GFF MaxViT U-Net')],
                [('unet', r['cfg']['unet']), ('vit', r['cfg']['vit'])]]
        top = fig_h - head_h
        for row, cells in enumerate(grid):
            bottom = top - tit_h - h
            for col, cell in enumerate(cells):
                x0 = side + col * (pw + gap)
                if cell is None:
                    hd, heads = em.legend_handles()
                    leg = fig.legend(handles=hd, loc='upper left', bbox_to_anchor=(x0 / width, (top - 0.02) / fig_h),
                                     ncol=1, frameon=False, fontsize=ps.FONT_SIZE, handlelength=1.1,
                                     handleheight=0.8, labelspacing=0.35, borderaxespad=0)
                    em.bold_heads(leg, heads)
                    continue
                k, title = cell
                ax = fig.add_axes([(x0 + (pw - w_) / 2) / width, bottom / fig_h, w_ / width, h / fig_h])
                ax.imshow(image(r['cls'], None if k == 'label' else r['maps'][k]), interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor('black')
                    sp.set_linewidth(0.6)
                ax.set_title(f'({next(letters)}) {title}', fontsize=ps.FONT_SIZE, pad=3, loc='center')
                ev = (r['cls'] == 0) | (r['cls'] == 1)                 # evaluated cells: labelled flood or dry
                if k == 'label':
                    a, s = em.area_share(int((r['cls'] == 1).sum()), int(ev.sum()))
                    txt = f'Flooded area {a} km²\n{s}% of the mapped area'
                else:
                    i, p, q = metrics(r['cls'], r['maps'][k])
                    txt = em.predicted_text(int((r['maps'][k] & ev).sum()), int(ev.sum())) + \
                        f'\nIoU {i:.2f}   POD {p:.2f}   POFD {q:.3f}'
                ax.text(0.5, -0.03, txt, transform=ax.transAxes, ha='center', va='top', fontsize=ps.FONT_SIZE)
            top = bottom - below_h[row]
        with mpl.rc_context({'savefig.bbox': None}):
            ps.save(fig, __file__, suffix=f'_{n}', subdir='5_5')


if __name__ == '__main__':
    main()
