"""Section 5.5.2 and Appendix A.13 — data of the comparison with the LISFLOOD-FP event footprints of the EFAS chain
(Scarpellini et al., 2026; Zenodo 10.5281/zenodo.19666073).

Reads the scores of the matched test mapping products (experiments/runs_baselines/lisflood_fp/results/
lisflood_fp_all_products.csv, written by scripts/score_all.py there) and keeps the mapping products whose mapped area
lies entirely within the footprint raster of the matched simulated flood (Ravi, 30 Oct 2026: the simulation must cover
the whole mapped area). Writes plot/data/lisflood_fp/:
  products.csv    one row per mapping product: place, country, labelled flooded area, the footprint without flood
                  protections and with the FLOPROS protections and every U-Net and ViT configuration (mean of seeds
                  1-3); `selected` marks the largest mapping product of each AOI (Figure 5.41).
  pooled.csv      cell counts pooled over all mapping products, for the footprint and every configuration and seed.
  agreement.csv   per selected mapping product, model (U-Net-T15, ViT-T15) and seed: labelled flood cells detected by
                  both the footprint and the model, by one of them or by neither, and the false detections of both or
                  of one.
U-Net-T15 and ViT-T15 are the models of Section 5.5 (Ravi, 27 Oct 2026: one fixed input-history length).
Places: AOI names of the CEMS products (activations_raw folders and AOI metadata).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_lisflood_fp_data.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

LF = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_baselines/lisflood_fp')
C = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'lisflood_fp'
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
MODELS = ('unet_t15', 'vit_t15')
CONFIGS = [f'{f}_{w}' for f in ('unet', 'vit') for w in ('t0', 't3', 't7', 't15', 't30')]
COUNTRY = {'EMSR251': 'Norway', 'EMSR287': 'Germany', 'EMSR327': 'Spain', 'EMSR397': 'Spain', 'EMSR411': 'France',
           'EMSR416': 'France', 'EMSR437': 'France', 'EMSR444': 'Ukraine', 'EMSR498': 'France',
           'EMSR640': 'United Kingdom', 'EMSR680': 'Slovenia', 'EMSR692': 'Greece', 'EMSR694': 'Spain',
           'EMSR698': 'United Kingdom', 'EMSR705': 'Italy', 'EMSR706': 'France', 'EMSR708': 'Belgium',
           'EMSR721': 'Italy'}
PLACE = {('EMSR397', 'AOI01'): 'Lleida', ('EMSR397', 'AOI07'): "L'Espluga de Francolí",
         ('EMSR411', 'AOI01'): 'Puget-sur-Argens', ('EMSR416', 'AOI01'): 'Peyrehorade', ('EMSR416', 'AOI02'): 'Dax',
         ('EMSR437', 'AOI06'): 'Roquefort', ('EMSR444', 'AOI03'): 'Khust', ('EMSR498', 'AOI01'): 'Argentat',
         ('EMSR640', 'AOI01'): 'Ballater', ('EMSR640', 'AOI02'): 'Aberdeen', ('EMSR640', 'AOI03'): 'Brechin'}


def aoi_place(product):
    t = product.split('_')
    if t[1].startswith('AOI'):
        aoi, words = t[1], []
        for x in t[2:]:
            if x in ('DEL', 'FEP', 'GRA'):
                break
            words.append(x)
        name = ' '.join(words).replace('Moorsele Wevelgem', 'Moorsele-Wevelgem')
    else:                                                        # old names such as 01ECIJA_01DELINEATION_MAP
        m = re.match(r'(\d+)([A-Z]+)', t[1])
        aoi, name = f'AOI{m.group(1)}', m.group(2).title()
    name = {'Ecija': 'Écija'}.get(name, name)
    return aoi, name or PLACE.get((t[0], aoi), '')


def footprint(event, g):
    """Flood cells of the footprint without protections on the 80 m cells of the patches g (wet where depth > 0,
    flood where at least half of the cell is wet), as score_all.py."""
    f = LF / 'data' / 'footprints' / f'noprot__flooded_area_{event}.tif'
    out = np.zeros((len(g), 32, 32), bool)
    with rasterio.open(f) as src:
        d = src.read(1)
        wet = np.where(np.isfinite(d) & (d > 0), 1.0, 0.0).astype('float32')
        for k, r in enumerate(g.itertuples()):
            dst = np.zeros((32, 32), 'float32')
            reproject(wet, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=from_origin(r.bounds_minx, r.bounds_maxy, 80.0, 80.0), dst_crs=r.crs,
                      resampling=Resampling.average, dst_nodata=0)
            out[k] = dst >= 0.5
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = pd.read_csv(LF / 'results' / 'lisflood_fp_all_products.csv')
    m = pd.read_csv(LF / 'data' / 'matches_test_products.csv')[['product', 'sim_event', 'start', 'end']]
    a = a.merge(m, on=['product', 'sim_event'], how='left', validate='one_to_one')
    pa = a['product'].map(aoi_place)
    a['aoi'], a['place'] = pa.str[0], pa.str[1]
    a['country'] = a.activation.map(COUNTRY)
    a['km2'] = a.flood_cells * 0.0064
    a = a[a.inside_footprint_raster == 1.0].copy()               # the simulation covers the whole mapped area
    big = a.sort_values('flood_cells', ascending=False).drop_duplicates(['activation', 'aoi'])
    a['selected'] = a['product'].isin(big['product'])
    a.sort_values(['selected', 'km2'], ascending=False).to_csv(OUT / 'products.csv', index=False)
    print(f"{len(a)} mapping products, {a.drop_duplicates(['activation', 'aoi']).shape[0]} AOIs, "
          f"{a.activation.nunique()} activations, {a.sim_event.nunique()} simulated floods")

    p = pd.read_csv(C / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number'])
    s = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_maxy'])
    lab_all = np.load(C / 'cells_test.npz')['label']
    bits = np.load(C / 'preds_test_bits.npz')
    shape = tuple(bits['shape'])
    rows, pool = [], []
    for r in a.itertuples():
        g = p[p.folder_name == r.product].merge(s, on=['folder_name', 'patch_number'], how='left',
                                                validate='one_to_one')
        idx = g.row_index.to_numpy()
        lab = lab_all[idx]
        F, D = lab == 1, lab == 0
        q = footprint(r.sim_event, g)
        pool.append(dict(product=r.product, config='footprint', seed=0, tp=int((q & F).sum()), fp=int((q & D).sum()),
                         fn=int((~q & F).sum()), dry=int(D.sum())))
        for cfg in CONFIGS:
            for sd in (1, 2, 3):
                o = np.unpackbits(bits[f'{cfg}_s{sd}'])[:np.prod(shape)].reshape(shape).astype(bool)[idx]
                pool.append(dict(product=r.product, config=cfg, seed=sd, tp=int((o & F).sum()), fp=int((o & D).sum()),
                                 fn=int((~o & F).sum()), dry=int(D.sum())))
                if r.selected and cfg in MODELS:
                    rows.append(dict(product=r.product, config=cfg, seed=sd, flood=int(F.sum()),
                                     both=int((o & q & F).sum()), lf_only=int((~o & q & F).sum()),
                                     ours_only=int((o & ~q & F).sum()), neither=int((~o & ~q & F).sum()),
                                     fa_both=int((o & q & D).sum()), fa_lf_only=int((~o & q & D).sum()),
                                     fa_ours_only=int((o & ~q & D).sum())))
    pd.DataFrame(rows).to_csv(OUT / 'agreement.csv', index=False)
    pool = pd.DataFrame(pool)
    pool.to_csv(OUT / 'pooled.csv', index=False)
    c = pool.groupby(['config', 'seed'])[['tp', 'fp', 'fn', 'dry']].sum()
    c['iou'], c['pod'], c['pofd'] = c.tp / (c.tp + c.fp + c.fn), c.tp / (c.tp + c.fn), c.fp / c.dry
    print(c.groupby('config')[['iou', 'pod', 'pofd']].mean().round(3).to_string())
    t = a[a.selected][['activation', 'aoi', 'place', 'country', 'date', 'km2', 'lf_noprot_iou', 'unet_t15_iou',
                       'vit_t15_iou']]
    pd.set_option('display.width', 250)
    print(t.sort_values('km2', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
