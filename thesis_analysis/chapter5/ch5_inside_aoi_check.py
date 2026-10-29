"""Sensitivity of the test evaluation to the cells outside the mapped AOI (26 Oct 2026, external review comment).

Patches that cross the AOI boundary keep the dry label 0 outside the AOI (Section 4.2.1), so these cells enter the
evaluation as dry without an observation. This script rescores the stored test predictions on the scored cells whose
centre lies inside the AOI polygon of their mapping product, and compares the micro IoU, POD and POFD with those on all
scored cells. No network is retrained.

Inputs (read only): plot/data/conditions/{cells_test.npz, preds_test_bits.npz, patches_test.csv.gz, fits.json};
experiments/metadata/splits/patches.csv (crs and bounds of each patch); the CEMS AOI polygon of each product,
pipeline/data/activations/cems_annotations/<EMSR>/<product>/aoi/aoi.shp.
Output: plot/data/conditions/inside_aoi_metrics.csv (one row per fit and cell set).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import from_origin

D = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
AOI = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/activations/cems_annotations')
CELL = 80.0


def inside_mask(p):
    """(N, 32, 32) bool: cell centre inside the AOI polygon of the product of the patch."""
    s = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_maxy'])
    p = p.merge(s, on=['folder_name', 'patch_number'], how='left', validate='one_to_one')
    assert p.crs.notna().all()
    mask = np.zeros((len(p), 32, 32), bool)
    for prod, g in p.groupby('folder_name'):
        aoi = gpd.read_file(AOI / prod.split('_')[0] / prod / 'aoi' / 'aoi.shp')
        for crs, gg in g.groupby('crs'):
            geom = aoi.to_crs(crs).geometry.union_all()
            for i, r in gg.iterrows():
                tr = from_origin(r.bounds_minx, r.bounds_maxy, CELL, CELL)
                mask[i] = features.rasterize([(geom, 1)], out_shape=(32, 32), transform=tr, fill=0,
                                             all_touched=False, dtype='uint8').astype(bool)
    return mask


def main():
    p = pd.read_csv(D / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number'])
    assert (p.row_index.to_numpy() == np.arange(len(p))).all()
    lab = np.load(D / 'cells_test.npz')['label']
    scored = (lab == 0) | (lab == 1)
    flood = lab == 1
    inside = inside_mask(p)
    print(f'scored cells {scored.sum():,}; inside the AOI {(scored & inside).sum():,} '
          f'({(scored & inside).sum() / scored.sum():.1%}); labelled flood outside the AOI {(flood & ~inside).sum():,}; '
          f'patches with scored cells outside the AOI {np.any(scored & ~inside, axis=(1, 2)).sum():,}')
    bits = np.load(D / 'preds_test_bits.npz')
    shape = tuple(bits['shape'])
    rows = []
    for fit in [k for k in bits.files if k != 'shape']:
        pred = np.unpackbits(bits[fit])[:np.prod(shape)].reshape(shape).astype(bool)
        for name, m in (('all scored', scored), ('inside AOI', scored & inside)):
            tp = int((pred & flood & m).sum()); fp = int((pred & ~flood & m).sum()); fn = int((~pred & flood & m).sum())
            tn = int(m.sum()) - tp - fp - fn
            rows.append(dict(fit=fit, cells=name, tp=tp, fp=fp, fn=fn, tn=tn, iou=tp / (tp + fp + fn),
                             pod=tp / (tp + fn), pofd=fp / (fp + tn)))
    r = pd.DataFrame(rows)
    r.to_csv(D / 'inside_aoi_metrics.csv', index=False)
    r['config'] = r.fit.str.replace(r'_s\d$', '', regex=True)
    s = r.groupby(['config', 'cells'])[['iou', 'pod', 'pofd']].mean().unstack('cells')
    pd.set_option('display.width', 200)
    print(s.round(4).to_string())
    # check: the stored fits reproduce the pooled counts of fits.json on all scored cells
    import json
    fj = json.loads((D / 'fits.json').read_text())
    a = r[r.cells == 'all scored'].set_index('fit')
    bad = [f for f in fj if f in a.index and (a.loc[f, 'tp'], a.loc[f, 'fp'], a.loc[f, 'fn']) != (fj[f]['tp'], fj[f]['fp'], fj[f]['fn'])]
    print('fits not matching fits.json:', bad)


if __name__ == '__main__':
    main()
