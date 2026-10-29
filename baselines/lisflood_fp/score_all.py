"""Score the EFAS/LISFLOOD-FP event footprints (Scarpellini et al., Zenodo 19666073) and the thesis networks on every
CEMS test product matched to a simulated event (matches_test_products.csv: same date window, event centre within
60 km of the AOI centre).

Per product, on its scored 80 m cells: the footprint (no protections and FLOPROS), turned into wet where the 100 m
depth is > 0, averaged onto the 80 m cells of each patch and counted as flood where at least half of the cell is wet;
the ten thesis configurations as the mean over seeds 1-3 of the per-seed metrics, at the validation thresholds.
Also reported: the share of the product's scored cells that lie inside the footprint raster, and pooled metrics per
activation.

    python score_all.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

BASE = Path(__file__).resolve().parents[1]           # experiments/runs_baselines/lisflood_fp
C = Path('/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data/conditions')
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
ZIPS = {'noprot': None, 'flopros': None}       # footprints extracted from the two archives into tif/<tag>__<file>
CFG = ['unet_t0', 'vit_t0', 'unet_t3', 'vit_t3', 'unet_t7', 'vit_t7', 'unet_t15', 'vit_t15', 'unet_t30', 'vit_t30']

m = pd.read_csv(BASE / 'data' / 'matches_test_products.csv')
p = pd.read_csv(C / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number'])
s = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_maxy'])
p = p[p.folder_name.isin(m['product'])].merge(s, on=['folder_name', 'patch_number'], how='left', validate='one_to_one')
lab_all = np.load(C / 'cells_test.npz')['label']
bits = np.load(C / 'preds_test_bits.npz')
shape = tuple(bits['shape'])
PRED = {f: np.unpackbits(bits[f])[:np.prod(shape)].reshape(shape).astype(bool) for f in bits.files if f != 'shape'}


def counts(pred, lab):
    sc = (lab == 0) | (lab == 1); f = lab == 1
    tp = int((pred & f & sc).sum()); fp = int((pred & ~f & sc).sum()); fn = int((~pred & f & sc).sum())
    return tp, fp, fn, int(sc.sum()) - tp - fp - fn


def metrics(tp, fp, fn, tn):
    return dict(iou=tp / max(tp + fp + fn, 1), pod=tp / max(tp + fn, 1), pofd=fp / max(fp + tn, 1))


def footprint_cells(tag, event, g):
    """(n, 32, 32) flood cells and inside-raster mask of the footprint on the patches g, or None if absent."""
    f = BASE / 'data' / 'footprints' / f'{tag}__flooded_area_{event}.tif'
    if not f.exists():                       # FLOPROS archive holds no footprint: no flood under the protections
        return np.zeros((len(g), 32, 32), bool), np.ones((len(g), 32, 32), bool)
    with rasterio.open(f) as src:
        depth = src.read(1)
        wet = np.where(np.isfinite(depth) & (depth > 0), 1.0, 0.0).astype('float32')
        out = np.zeros((len(g), 32, 32), bool); ins = np.zeros((len(g), 32, 32), bool)
        for k, r in enumerate(g.itertuples()):
            dst = np.zeros((32, 32), 'float32'); cov = np.zeros((32, 32), 'float32')
            tr = from_origin(r.bounds_minx, r.bounds_maxy, 80.0, 80.0)
            reproject(wet, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs=r.crs,
                      resampling=Resampling.average, dst_nodata=0)
            reproject(np.ones_like(wet), cov, src_transform=src.transform, src_crs=src.crs, dst_transform=tr,
                      dst_crs=r.crs, resampling=Resampling.average, dst_nodata=0)
            out[k] = dst >= 0.5; ins[k] = cov > 0.99
    return out, ins


rows, pooled = [], {}
for (prod, event), mm in m.groupby(['product', 'sim_event']):
    g = p[p.folder_name == prod].reset_index(drop=True)
    lab = lab_all[g.row_index.to_numpy()]
    sc = (lab == 0) | (lab == 1)
    r = dict(product=prod, activation=prod.split('_')[0], date=prod[-8:], sim_event=event,
             dist_km=float(mm.dist_km.iloc[0]), patches=len(g), scored=int(sc.sum()), flood_cells=int((lab == 1).sum()))
    act = r['activation']
    for tag in ZIPS:
        fp_cells, ins = footprint_cells(tag, event, g)
        c = counts(fp_cells, lab)
        r.update({f'lf_{tag}_{k}': v for k, v in metrics(*c).items()})
        r[f'lf_{tag}_pred_share'] = (c[0] + c[1]) / max(sc.sum(), 1)
        if tag == 'noprot':
            r['inside_footprint_raster'] = float((sc & ins).sum() / max(sc.sum(), 1))
        pooled.setdefault((act, f'LISFLOOD-FP {tag}'), np.zeros(4, int))
        pooled[(act, f'LISFLOOD-FP {tag}')] += np.array(c)
    for cfg in CFG:
        per = []
        for sd in (1, 2, 3):
            c = counts(PRED[f'{cfg}_s{sd}'][g.row_index.to_numpy()], lab)
            per.append(metrics(*c))
            pooled.setdefault((act, f'{cfg}_s{sd}'), np.zeros(4, int))
            pooled[(act, f'{cfg}_s{sd}')] += np.array(c)
        for k in ('iou', 'pod', 'pofd'):
            r[f'{cfg}_{k}'] = float(np.mean([q[k] for q in per]))
    rows.append(r)
    print(prod, event, f"LF {r.get('lf_noprot_iou', np.nan):.3f}", f"ViT-T0 {r['vit_t0_iou']:.3f}", flush=True)

R = pd.DataFrame(rows).sort_values(['activation', 'date', 'product'])
R.to_csv(BASE / 'results' / 'lisflood_fp_all_products.csv', index=False)
# pooled per activation: LISFLOOD-FP and the seed means of the configurations
prow = []
for act in sorted({a for a, _ in pooled}):
    d = dict(activation=act, products=int((R.activation == act).sum()), scored=int(R.scored[R.activation == act].sum()),
             flood_cells=int(R.flood_cells[R.activation == act].sum()))
    for tag in ZIPS:
        if (act, f'LISFLOOD-FP {tag}') in pooled:
            d.update({f'lf_{tag}_{k}': v for k, v in metrics(*pooled[(act, f'LISFLOOD-FP {tag}')]).items()})
    for cfg in CFG:
        ms = [metrics(*pooled[(act, f'{cfg}_s{sd}')]) for sd in (1, 2, 3)]
        for k in ('iou', 'pod', 'pofd'):
            d[f'{cfg}_{k}'] = float(np.mean([q[k] for q in ms]))
    prow.append(d)
P = pd.DataFrame(prow)
P.to_csv(BASE / 'results' / 'lisflood_fp_by_activation.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
cols = ['activation', 'products', 'flood_cells', 'lf_noprot_iou', 'lf_flopros_iou'] + [f'{c}_iou' for c in CFG]
print(P[[c for c in cols if c in P]].round(3).to_string(index=False))

# pooled over all matched products, and over the activations whose AOIs lie inside their footprint raster
covered = set(R.groupby('activation').inside_footprint_raster.median().loc[lambda x: x >= 0.5].index)
orow = []
for name, acts in (('all matched products', set(R.activation)), ('activations inside the footprint raster', covered)):
    d = dict(subset=name, activations=len(acts), products=int(R.activation.isin(acts).sum()),
             flood_cells=int(R.flood_cells[R.activation.isin(acts)].sum()))
    for tag in ZIPS:
        c = sum(pooled[(a, f'LISFLOOD-FP {tag}')] for a in acts)
        d.update({f'lf_{tag}_{k}': v for k, v in metrics(*c).items()})
    for cfg in CFG:
        ms = [metrics(*sum(pooled[(a, f'{cfg}_s{sd}')] for a in acts)) for sd in (1, 2, 3)]
        for k in ('iou', 'pod', 'pofd'):
            d[f'{cfg}_{k}'] = float(np.mean([q[k] for q in ms]))
    orow.append(d)
O = pd.DataFrame(orow)
O.to_csv(BASE / 'results' / 'lisflood_fp_pooled.csv', index=False)
print(O.round(3).T.to_string())
