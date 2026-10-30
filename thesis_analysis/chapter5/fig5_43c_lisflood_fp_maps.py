"""Flood maps of three test mapping products: flood label, the LISFLOOD-FP event footprint of the EFAS chain,
U-Net-T15 and ViT-T15 (majority of seeds 1-3), on the evaluated 80 m cells (Section 5.5.2).

Mapping products by rule among the 21 of Table 5.7 (plot/data/lisflood_fp/products.csv): the one where the footprint
exceeds the better of U-Net-T15 and ViT-T15 most (Palamas); of those where a model of this thesis exceeds the footprint
and the footprint floods the AOI, the one with the most labelled flood (Westhoek); of those where the footprint has no
flood in the AOI, the one with the highest IoU of U-Net-T15 or ViT-T15 (Orentano). T15 is the input-history length of the
models of this thesis in Section 5.5 (Ravi, 27 Oct 2026). Footprint: no flood protections, wet where the 100 m depth is
above 0, flood on an 80 m cell where at least half of the cell is wet.
Sources (read only): experiments/runs_baselines/lisflood_fp/data/footprints (Scarpellini et al., 2026, Zenodo
10.5281/zenodo.19666073); plot/data/conditions/{cells_test.npz, preds_test_bits.npz, patches_test.csv.gz}.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import to_rgb
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

import event_maps as em
from rq1_common import ps

C = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
LF = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_baselines/lisflood_fp')
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
PROD = Path(__file__).resolve().parents[1] / 'data' / 'lisflood_fp' / 'products.csv'
NAME = {'unet': 'U-Net', 'vit': 'ViT'}


def footprint(event, g):
    with rasterio.open(LF / 'data' / 'footprints' / f'noprot__flooded_area_{event}.tif') as src:
        d = src.read(1)
        wet = np.where(np.isfinite(d) & (d > 0), 1.0, 0.0).astype('float32')
        out = np.zeros((len(g), 32, 32), bool)
        for k, r in enumerate(g.itertuples()):
            dst = np.zeros((32, 32), 'float32')
            reproject(wet, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=from_origin(r.bounds_minx, r.bounds_maxy, 80.0, 80.0), dst_crs=r.crs,
                      resampling=Resampling.average, dst_nodata=0)
            out[k] = dst >= 0.5
    return out


def mosaic(lab, pred, ri, ci, H, W):
    img = np.ones((H, W, 3))
    for k in range(len(lab)):
        l = lab[k]; t = np.ones((32, 32, 3))
        t[l == -1] = to_rgb(em.COL['pw'])
        if pred is None:
            t[l == 1] = to_rgb(em.COL['tp']); t[l == 0] = to_rgb(em.COL['dry'])
        else:
            q = pred[k]
            t[(l == 1) & q] = to_rgb(em.COL['tp']); t[(l == 1) & ~q] = to_rgb(em.COL['fn'])
            t[(l == 0) & q] = to_rgb(em.COL['fp']); t[(l == 0) & ~q] = to_rgb(em.COL['dry'])
        img[ri[k]:ri[k] + 32, ci[k]:ci[k] + 32] = t
    return img


def metrics(lab, pred):
    m = (lab == 0) | (lab == 1); f = lab == 1
    tp = (pred & f & m).sum(); fp = (pred & ~f & m).sum(); fn = (~pred & f & m).sum(); tn = m.sum() - tp - fp - fn
    return tp / (tp + fp + fn), tp / (tp + fn), fp / (fp + tn)


def choose():
    p = pd.read_csv(PROD)
    p = p[p.selected].set_index('product')
    best = p[['unet_t15_iou', 'vit_t15_iou']].max(axis=1)
    lead = p.lf_noprot_iou - best
    ours = p[(lead < 0) & (p.lf_noprot_iou > 0)]
    dry = best[p.lf_noprot_iou == 0]
    return [(k, p.loc[k]) for k in (lead.idxmax(), ours.km2.idxmax(), dry.idxmax())]


def label(cfg):
    a, w = cfg.split('_')
    return f'{NAME[a]}-{w.upper()}'


def main():
    picks = choose()
    p = pd.read_csv(C / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number', 'flood'])
    s = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_maxy'])
    lab_all = np.load(C / 'cells_test.npz')['label']
    bits = np.load(C / 'preds_test_bits.npz'); shape = tuple(bits['shape'])
    rows = []
    for prod, r in picks:
        g = p[p.folder_name == prod].merge(s, on=['folder_name', 'patch_number'])
        idx = g.row_index.to_numpy()
        lab = lab_all[idx]
        maps = {'lf': footprint(r.sim_event, g)}
        for arch in ('unet', 'vit'):
            cfg = f'{arch}_t15'
            v = [np.unpackbits(bits[f'{cfg}_s{sd}'])[:np.prod(shape)].reshape(shape).astype(bool)[idx] for sd in (1, 2, 3)]
            maps[arch] = np.sum(v, axis=0) >= 2
        x0, y1 = g.bounds_minx.min(), g.bounds_maxy.max()
        ci = ((g.bounds_minx - x0) / 80).round().astype(int).to_numpy()
        ri = ((y1 - g.bounds_maxy) / 80).round().astype(int).to_numpy()
        code, d = em.product_title(prod)
        code = code.replace(r.place.replace('-', ' '), r.place) if r.place else code   # Moorsele Wevelgem -> hyphen
        if r.place and r.place not in code:
            code = code.replace(r.aoi, f'{r.aoi} {r.place}')
        titles = {'lf': 'LISFLOOD-FP', 'unet': 'U-Net-T15', 'vit': 'ViT-T15'}
        rows.append(dict(prod=prod, head=f'{code}, {r.country}, {d}', lab=lab, maps=maps, titles=titles, ri=ri, ci=ci,
                         H=ri.max() + 32, W=ci.max() + 32))
        print(prod, {k: tuple(round(float(x), 3) for x in metrics(lab, v)) for k, v in maps.items()})

    ps.apply_clear()
    width, ncol, gap = ps.TEXT_WIDTH_IN, 4, 0.08
    pw = (width - gap * (ncol - 1) - 0.1) / ncol
    tit_h, lab_h, leg_h = 0.38, 0.42, 0.72
    heights = [min(pw * r['H'] / r['W'], 1.0 * pw) for r in rows]       # tall AOIs get narrower panels
    fig_h = sum(h + tit_h + lab_h for h in heights) + leg_h
    fig = plt.figure(figsize=(width, fig_h))
    top = fig_h
    letters = iter('abcdefghijkl')
    for r, h in zip(rows, heights):
        bottom = top - tit_h - h
        fig.text(0.05 / width, (top - 0.03) / fig_h, r['head'], ha='left', va='top', fontsize=ps.FONT_SIZE - 1.5)
        panels = [(None, 'Flood label')] + [(k, r['titles'][k]) for k in ('lf', 'unet', 'vit')]
        for col, (k, title) in enumerate(panels):
            w_ = h * r['W'] / r['H']                                    # panel width at the capped height
            ax = fig.add_axes([(0.05 + col * (pw + gap) + (pw - w_) / 2) / width, bottom / fig_h, w_ / width, h / fig_h])
            ax.imshow(mosaic(r['lab'], None if k is None else r['maps'][k], r['ri'], r['ci'], r['H'], r['W']),
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(0.6)
            ax.set_title(f'({next(letters)}) {title}', fontsize=ps.FONT_SIZE - 2, pad=3, loc='center')
            lab = r['lab']; ev = (lab == 0) | (lab == 1)                # evaluated cells: labelled flood or dry
            if k is None:
                a, s = em.area_share(int((lab == 1).sum()), int(ev.sum()))
                txt = f'Flooded area {a} km²\n{s}% of the mapped area'
            else:
                i, pod, pofd = metrics(r['lab'], r['maps'][k])
                txt = em.predicted_text(int((r['maps'][k] & ev).sum()), int(ev.sum())) + \
                    f'\nIoU {i:.2f}   POD {pod:.2f}\nPOFD {pofd:.3f}'
            ax.text(0.5, -0.04, txt, transform=ax.transAxes, ha='center', va='top', fontsize=ps.FONT_SIZE - 2.5)
        top = bottom - lab_h
    h, heads = em.legend_handles()
    leg = fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False,
                     fontsize=ps.FONT_SIZE - 2, handlelength=1.1, handleheight=0.8, columnspacing=1.2)
    em.bold_heads(leg, heads)
    with mpl.rc_context({'savefig.bbox': None}):
        ps.save(fig, __file__, subdir='5_5')


if __name__ == '__main__':
    main()
