"""Map of one CEMS test product: flood label, the LISFLOOD-FP event footprint and two thesis configurations
(majority of seeds 1-3), on the scored 80 m cells, colours of the Chapter 5 event maps.

    python map_lisflood_fp.py <product> <prefix> <tag> <out.png>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

sys.path.insert(0, '/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/scripts')
import event_maps as em
from rq1_common import ps

C = Path('/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data/conditions')
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
prod, prefix, tag, out = sys.argv[1:5]
H = Path(__file__).resolve().parents[1] / 'results'

p = pd.read_csv(C / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number'])
sel = p[p.folder_name.str.startswith(prefix)].reset_index(drop=True)
model_all = np.load(H / f'model_cells_{prefix}_{tag}.npy')
s = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'bounds_minx', 'bounds_maxy'])
g = sel[sel.folder_name == prod].merge(s, on=['folder_name', 'patch_number'])
idx = sel.index[sel.folder_name == prod].to_numpy()
rows = g.row_index.to_numpy()
lab = np.load(C / 'cells_test.npz')['label'][rows]
bits = np.load(C / 'preds_test_bits.npz'); shape = tuple(bits['shape'])


def maj(cfg):
    v = [np.unpackbits(bits[f'{cfg}_s{s}'])[:np.prod(shape)].reshape(shape).astype(bool)[rows] for s in (1, 2, 3)]
    return np.sum(v, axis=0) >= 2


x0, y1 = g.bounds_minx.min(), g.bounds_maxy.max()
ci = ((g.bounds_minx - x0) / 80).round().astype(int).to_numpy(); ri = ((y1 - g.bounds_maxy) / 80).round().astype(int).to_numpy()
Hh, Ww = ri.max() + 32, ci.max() + 32


def mosaic(pred):
    img = np.ones((Hh, Ww, 3))
    for k in range(len(g)):
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


def metr(pred):
    m = (lab == 0) | (lab == 1); f = lab == 1
    tp = (pred & f & m).sum(); fp = (pred & ~f & m).sum(); fn = (~pred & f & m).sum(); tn = m.sum() - tp - fp - fn
    return f'IoU {tp / (tp + fp + fn):.2f}   POD {tp / (tp + fn):.2f}\nPOFD {fp / (fp + tn):.3f}'


panels = [('Flood label', None), ('LISFLOOD-FP event footprint', model_all[idx]),
          ('ViT-T3, best here', maj('vit_t3')), ('ViT-T0', maj('vit_t0')), ('U-Net-T0', maj('unet_t0'))]
ps.apply_clear()
fig, axs = plt.subplots(1, 5, figsize=(ps.TEXT_WIDTH_IN * 1.6, 3.2))
for ax, (t, pr) in zip(axs, panels):
    ax.imshow(mosaic(pr), interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(t, fontsize=9, loc='left')
    if pr is not None:
        ax.text(0.5, -0.03, metr(pr), transform=ax.transAxes, ha='center', va='top', fontsize=8)
code, d = em.product_title(prod)
fig.suptitle(f'{code}, {d}', fontsize=10, x=0.01, ha='left')
h, heads = em.legend_handles()
leg = fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
em.bold_heads(leg, heads)
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(out, dpi=200, facecolor='white'); print('wrote', out)
