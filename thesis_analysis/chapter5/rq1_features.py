"""Characteristics of every test mapping product, for the RQ1 win-share figures (plot/data/rq1/product_features.csv).

From the test rows of the evaluator (51,606 patches, 404 products) and their 80 m labels (preds/test__labels: 1 flood,
0 dry, -1 permanent water or mixed):
  flood_cells, scored_cells, area_km2 (flood cells x 0.0064), flood_fraction (flood / scored cells),
  pw_share (permanent-water and mixed cells / all cells of the product's test patches),
  pw_distance_class (median distance of the product's flood cells to the nearest permanent-water or mixed cell of
  the same patch: '<= 80 m' (adjacent), '80-240 m', '240-800 m', '> 800 m or none'; Euclidean, in 80 m cells),
  elevation_m (median over the product's test patches of the patch median of input_80m band 0; the local elevation
  band equals the released one, chapters' figure-style memory, 1 Oct 2026),
  continent, climate (test index), resolution_class (released_events_metadata.csv), map_type ('first map' or
  'monitoring map': MONIT in the product name).
HAND is not included: the local input_80m HAND band is the pre-fix build.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/rq1_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

import ch5_data as cd

PT = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/patches/patches_tif')
CATALOGUE = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/metadata/released_events_metadata.csv')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq1' / 'product_features.csv'
DIST_EDGES = (1.0, 3.0, 10.0)                      # cells: <= 80 m, <= 240 m, <= 800 m
DIST_LABELS = ('<= 80 m', '80-240 m', '240-800 m', '> 800 m or none')


def main():
    ts = cd.test_set()
    ix = ts.index
    Y = np.load(cd.run_dir('unet', 't0', 1) / 'inference_final/preds/test__labels.i8.npy')
    keep, flood, pw = Y >= 0, Y == 1, Y < 0
    dist_counts = np.zeros((len(Y), 4), np.int64)
    elev = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        if flood[i].any():
            if pw[i].any():
                d = ndimage.distance_transform_edt(~pw[i])[flood[i]]
                k = np.digitize(d, DIST_EDGES, right=True)
            else:
                k = np.full(int(flood[i].sum()), 3)
            dist_counts[i] = np.bincount(k, minlength=4)
        r = ix.iloc[i]
        f = PT / r.folder_name.split('_')[0] / r.folder_name / f'patch_{int(r.patch_number):04d}_input_80m.tif'
        with rasterio.open(f) as src:
            e = src.read(1).astype(np.float64)
        e[(e >= 9999) | ~np.isfinite(e)] = np.nan
        elev[i] = np.nanmedian(e) if np.isfinite(e).any() else np.nan
        if i % 10000 == 0:
            print(i, flush=True)
    g = pd.DataFrame({'folder_name': ix.folder_name, 'scored': keep.sum((1, 2)), 'flood': flood.sum((1, 2)),
                      'pw': pw.sum((1, 2)), 'elev': elev})
    for k in range(4):
        g[f'd{k}'] = dist_counts[:, k]
    p = g.groupby('folder_name').agg(scored_cells=('scored', 'sum'), flood_cells=('flood', 'sum'), pw_cells=('pw', 'sum'),
                                     elevation_m=('elev', 'median'), d0=('d0', 'sum'), d1=('d1', 'sum'), d2=('d2', 'sum'),
                                     d3=('d3', 'sum'))
    p['area_km2'] = p.flood_cells * 0.0064
    p['flood_fraction'] = p.flood_cells / p.scored_cells
    p['pw_share'] = p.pw_cells / (p.scored_cells + p.pw_cells)
    cum = p[['d0', 'd1', 'd2', 'd3']].cumsum(axis=1).div(p.flood_cells.replace(0, np.nan), axis=0)
    p['pw_distance_class'] = [DIST_LABELS[int(np.argmax(row >= 0.5))] if np.isfinite(row).all() else None
                              for row in cum.to_numpy()]
    meta = ix.groupby('folder_name')[['continent', 'climate']].first()
    cat = pd.read_csv(CATALOGUE, usecols=['folder_name', 'resolution_class']).set_index('folder_name')
    p = p.join(meta).join(cat)
    p['map_type'] = np.where(p.index.str.contains('MONIT'), 'monitoring map', 'first map')
    p.drop(columns=['d0', 'd1', 'd2', 'd3']).to_csv(OUT)
    q = p[p.flood_cells > 0]
    print(f'{len(p)} products, {len(q)} flood-positive')
    for c in ('pw_distance_class', 'continent', 'climate', 'resolution_class', 'map_type'):
        print(q[c].value_counts().to_dict())
    print(q[['area_km2', 'flood_fraction', 'pw_share', 'elevation_m']].quantile([0, .1, .25, .5, .75, .9, 1]).round(4).to_string())


if __name__ == '__main__':
    main()
