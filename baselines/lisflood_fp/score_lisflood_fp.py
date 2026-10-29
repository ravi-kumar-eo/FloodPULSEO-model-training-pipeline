"""Score an EFAS/LISFLOOD-FP event footprint (Scarpellini et al., Zenodo 19666073) and the thesis networks on the same
scored 80 m test cells of the matching CEMS products.

The footprint (100 m, depth in m) is turned into wet/dry, averaged onto the 80 m cell grid of each test patch and
counted as flood where at least half of the cell is wet (the majority rule of the flood label). Cells outside the
footprint raster count as dry. Our predictions are the stored test predictions of every fit at its validation-selected
threshold (plot/data/conditions/preds_test_bits.npz); the label is the stored 80 m CEMS label (cells_test.npz).

    python score_lisflood_fp.py <footprint.tif> <product prefix> <tag>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

C = Path('/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data/conditions')
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
tif, prefix, tag = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path(__file__).resolve().parents[1] / 'results'

p = pd.read_csv(C / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number'])
sel = p[p.folder_name.str.startswith(prefix)].copy()
s = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_maxy'])
sel = sel.merge(s, on=['folder_name', 'patch_number'], how='left', validate='one_to_one')
lab = np.load(C / 'cells_test.npz')['label'][sel.row_index.to_numpy()]
scored, flood = (lab == 0) | (lab == 1), lab == 1

with rasterio.open(tif) as src:
    depth = src.read(1)
    wet = np.where(np.isfinite(depth) & (depth > 0), 1.0, 0.0).astype('float32')
    model = np.zeros(lab.shape, bool)
    inside = np.zeros(lab.shape, bool)
    for k, r in enumerate(sel.itertuples()):
        dst = np.zeros((32, 32), 'float32')
        cov = np.zeros((32, 32), 'float32')
        tr = from_origin(r.bounds_minx, r.bounds_maxy, 80.0, 80.0)
        reproject(wet, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=r.crs,
                  resampling=Resampling.average, src_nodata=None, dst_nodata=0)
        reproject(np.ones_like(wet), cov, src_transform=src.transform, src_crs=src.crs, dst_transform=tr,
                  dst_crs=r.crs, resampling=Resampling.average, src_nodata=None, dst_nodata=0)
        model[k] = dst >= 0.5
        inside[k] = cov > 0.99

bits = np.load(C / 'preds_test_bits.npz')
shape = tuple(bits['shape'])
rows_idx = sel.row_index.to_numpy()


def score(pred, m):
    tp = int((pred & flood & m).sum()); fp = int((pred & ~flood & m).sum()); fn = int((~pred & flood & m).sum())
    tn = int(m.sum()) - tp - fp - fn
    return dict(iou=tp / max(tp + fp + fn, 1), pod=tp / max(tp + fn, 1), pofd=fp / max(fp + tn, 1),
                pred_share=(tp + fp) / max(m.sum(), 1))


res = []
m_all = scored
print(f'{prefix}: {sel.folder_name.nunique()} products, {len(sel):,} patches, scored cells {scored.sum():,}, '
      f'labelled flood {flood.sum():,} ({flood.sum() / scored.sum():.1%}); scored cells inside the footprint raster '
      f'{(scored & inside).sum() / scored.sum():.1%}')
res.append(dict(model=f'LISFLOOD-FP ({tag})', seed=0, **score(model, m_all)))
for fit in [k for k in bits.files if k != 'shape']:
    pred = np.unpackbits(bits[fit])[:np.prod(shape)].reshape(shape).astype(bool)[rows_idx]
    res.append(dict(model=fit[:-3], seed=int(fit[-1]), **score(pred, m_all)))
r = pd.DataFrame(res)
r.to_csv(OUT / f'scores_{prefix}_{tag}.csv', index=False)
main = ['unet_t0', 'vit_t0', 'unet_t3', 'vit_t3', 'unet_t7', 'vit_t7', 'unet_t15', 'vit_t15', 'unet_t30', 'vit_t30']
g = r[r.model.isin(main + [f'LISFLOOD-FP ({tag})'])].groupby('model')[['iou', 'pod', 'pofd', 'pred_share']].mean()
g = g.loc[[f'LISFLOOD-FP ({tag})'] + main]
pd.set_option('display.width', 200)
print(g.round(3).to_string())
# per product: LISFLOOD-FP vs the seed means of T0 and T30
rows = []
for f, gi in sel.reset_index(drop=True).groupby('folder_name'):
    ii = gi.index.to_numpy(); m = np.zeros(lab.shape, bool); m[ii] = scored[ii]
    d = dict(product=f, flood_share=flood[ii].sum() / max(scored[ii].sum(), 1), lisflood_fp=score(model, m)['iou'])
    for cfg in ('unet_t0', 'vit_t0', 'unet_t30', 'vit_t30'):
        d[cfg] = np.mean([score(np.unpackbits(bits[f'{cfg}_s{sd}'])[:np.prod(shape)].reshape(shape).astype(bool)[rows_idx], m)['iou'] for sd in (1, 2, 3)])
    rows.append(d)
pp = pd.DataFrame(rows).sort_values('product', key=lambda x: x.str[-8:])
pp.to_csv(OUT / f'per_product_{prefix}_{tag}.csv', index=False)
print(pp.round(3).to_string(index=False))
np.save(OUT / f'model_cells_{prefix}_{tag}.npy', model)
