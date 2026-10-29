"""RQ2, inside the regimes: scores of each test spatial block, its distance to the training ground, the regime scores
without the dominant blocks, and a screen of the cell characteristics (soil, terrain, vegetation) by regime.
Runs on the laptop copy of the condition tables (ch5_cells_build.py):

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_regime_blocks.py

Inputs (plot/data/conditions/): cells_test.npz, preds_test_bits.npz, fits.json, patches_test.csv.gz, clusters_boot.npz;
experiments/metadata/splits/patches.csv for the patch footprints of all subsets. T0 fits only; the partial block is
left out.
Outputs (plot/data/conditions/):
  rq2_blocks.csv      per basin-seen and basin-unseen block: patches, scored and flood cells, flooded share, majority
                      continent and climate, distance from its patches to the nearest training patch (median and minimum
                      over its patches, km, patch centres), and per family the seed-mean IoU, POD, POFD and area bias
                      pooled over the block
  rq2_exclusion.csv   regime scores (seed means) and unseen - seen with 95% cluster bootstrap intervals for all blocks,
                      without the block holding the most basin-seen flood cells, and without the largest block of each
                      regime by flood cells
  rq2_features.csv    per characteristic (cell, patch and product level), class and regime: share of the regime's
                      scored cells, flooded share, seed-mean IoU, POD and POFD per family, and unseen - seen with its 95%
                      cluster bootstrap interval. Classes as in the RQ1 screen (ch5_cells_screen.py): physical bounds,
                      quintiles of the scored cells for clay and sand.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import BallTree

D = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
FAM = ('unet', 'vit')
T0 = {f: [f'{f}_t0_s{s}' for s in (1, 2, 3)] for f in FAM}
INF = np.inf
BINS = {'hand_m': [0, 1, 2, 5, 10, 20, 50, INF], 'elevation_m': [-INF, 10, 50, 200, 500, INF],
        'slope_deg': [0, 0.5, 1, 2, 5, 10, INF],
        'uda_log10': [0, np.log10(2), np.log10(11), np.log10(101), np.log10(1001), np.log10(10001), INF],
        'ndvi': [-1.0001, 0, 0.2, 0.4, 0.6, 1.0001], 'ndbi': [-1.0001, -0.2, 0, 0.2, 1.0001]}
QUINTILES = ('clay_pct', 'sand_pct')


def metrics(c):                                                      # c (..., 4): TP FP FN TN
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'iou': tp / (tp + fp + fn), 'pod': tp / (tp + fn), 'pofd': fp / (fp + tn),
                'area_bias': (tp + fp) / (tp + fn)}


def patch_centres(df):
    """Longitude and latitude of the centre of each patch footprint (bounds in the patch CRS)."""
    x = ((df.bounds_minx + df.bounds_maxx) / 2).to_numpy(); y = ((df.bounds_miny + df.bounds_maxy) / 2).to_numpy()
    lon, lat = np.empty(len(df)), np.empty(len(df))
    for crs, idx in df.groupby('crs').indices.items():
        t = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        lon[idx], lat[idx] = t.transform(x[idx], y[idx])
    return lon, lat


def main():
    cells = np.load(D / 'cells_test.npz')
    bits = np.load(D / 'preds_test_bits.npz')
    fits = [k for k in json.loads((D / 'fits.json').read_text()) if '_t0_' in k]
    cb = np.load(D / 'clusters_boot.npz')
    W, clus = cb['weights'].astype(float), cb['cluster'].astype(np.int64)
    K = W.shape[1]
    pt = pd.read_csv(D / 'patches_test.csv.gz', usecols=['row_index', 'folder_name', 'patch_number', 'ground_unit',
                                                          'basin_regime', 'continent', 'climate'])
    Y = cells['label']
    scored, flood, dry = Y >= 0, Y == 1, Y == 0
    reg = pt.basin_regime.to_numpy()
    keep = np.isin(reg, ['seen', 'unseen'])
    P = {k: np.unpackbits(bits[k])[:Y.size].reshape(Y.shape).astype(bool) for k in fits}
    rowc = {k: np.stack([(P[k] & flood).sum((1, 2)), (P[k] & dry).sum((1, 2)), (~P[k] & flood).sum((1, 2)),
                         (~P[k] & dry).sum((1, 2))], 1).astype(float) for k in fits}

    # ---- 1. per block scores and distance to the training ground
    sp = pd.read_csv(SPLIT, usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_miny', 'bounds_maxx',
                                     'bounds_maxy', 'cohort'])
    tr = sp[sp.cohort == 'train'].drop_duplicates(['crs', 'bounds_minx', 'bounds_miny'])
    tlon, tlat = patch_centres(tr)
    tree = BallTree(np.radians(np.c_[tlat, tlon]), metric='haversine')
    te = pt[['folder_name', 'patch_number']].merge(sp, on=['folder_name', 'patch_number'], how='left')
    assert te.crs.notna().all() and len(te) == len(pt)
    lon, lat = patch_centres(te)
    dist_km = tree.query(np.radians(np.c_[lat, lon]), k=1)[0][:, 0] * 6371.0
    blocks = []
    for b, g in pt[keep].groupby('ground_unit'):
        i = g.index.to_numpy()
        rec = {'block': b, 'regime': g.basin_regime.iloc[0], 'patches': len(i), 'products': g.folder_name.nunique(),
               'scored_cells': int(scored[i].sum()), 'flood_cells': int(flood[i].sum()),
               'continent': g.continent.mode().iloc[0], 'climate': g.climate.mode().iloc[0],
               'lon': float(np.mean(lon[i])), 'lat': float(np.mean(lat[i])),
               'dist_train_median_km': float(np.median(dist_km[i])), 'dist_train_min_km': float(dist_km[i].min()),
               'cluster': int(clus[i[0]])}
        rec['flooded_share'] = rec['flood_cells'] / rec['scored_cells']
        for f in FAM:
            m = [metrics(rowc[k][i].sum(0)) for k in T0[f]]
            for name in ('iou', 'pod', 'pofd', 'area_bias'):
                rec[f'{f}_{name}'] = float(np.mean([x[name] for x in m]))
        blocks.append(rec)
    bl = pd.DataFrame(blocks)
    bl.to_csv(D / 'rq2_blocks.csv', index=False)
    print(bl.groupby('regime')[['dist_train_median_km', 'dist_train_min_km']].describe().T.round(1))

    # ---- 2. regime scores without the dominant blocks
    seen_top = bl[bl.regime == 'seen'].sort_values('flood_cells').block.iloc[-1]
    unseen_top = bl[bl.regime == 'unseen'].sort_values('flood_cells').block.iloc[-1]
    gu = pt.ground_unit.to_numpy()
    out = []
    for label, drop in (('all blocks', []), (f'without {seen_top}', [seen_top]),
                        (f'without {seen_top} and {unseen_top}', [seen_top, unseen_top])):
        ok = keep & ~np.isin(gu, drop)
        for f in FAM:
            full, reps, vals = [], [], {}
            for k in T0[f]:
                c = rowc[k]
                ms = {r: metrics(c[ok & (reg == r)].sum(0)) for r in ('seen', 'unseen')}
                for r in ms:
                    for name, v in ms[r].items():
                        vals.setdefault((r, name), []).append(v)
                cl = {r: np.zeros((K, 4)) for r in ('seen', 'unseen')}
                for r in cl:
                    s = ok & (reg == r); np.add.at(cl[r], clus[s], c[s])
                full.append({n: ms['unseen'][n] - ms['seen'][n] for n in ms['seen']})
                mu, mse = metrics(W @ cl['unseen']), metrics(W @ cl['seen'])
                reps.append({n: mu[n] - mse[n] for n in mu})
            for name in ('iou', 'pod', 'pofd', 'area_bias'):
                rep = np.mean([r[name] for r in reps], 0)
                lo, hi = np.nanpercentile(rep, [2.5, 97.5])
                out.append({'set': label, 'family': f, 'metric': name, 'seen': np.mean(vals[('seen', name)]),
                            'unseen': np.mean(vals[('unseen', name)]),
                            'unseen_minus_seen': np.mean([x[name] for x in full]), 'lo': lo, 'hi': hi})
    ex = pd.DataFrame(out)
    ex.to_csv(D / 'rq2_exclusion.csv', index=False)
    print(ex[ex.metric.isin(['iou', 'pod', 'pofd'])].round(3).to_string(index=False))

    # ---- 3. screen of the characteristics by regime (cell values, patch values broadcast to their cells, product
    #         flooded area broadcast to the cells of the product)
    rows_idx = np.repeat(np.arange(len(pt)), Y.shape[1] * Y.shape[2])
    sc_flat, fl_flat = scored.ravel(), flood.ravel()
    regc = np.where(reg == 'seen', 0, np.where(reg == 'unseen', 1, -1))
    cell = lambda k: cells[k].astype(np.float32)
    patch = lambda v: np.repeat(np.asarray(v, np.float32), Y.shape[1] * Y.shape[2])
    pw = (Y == -1).mean((1, 2))
    area = pd.read_csv(D / 'product_features.csv').set_index('folder_name').area_km2.reindex(pt.folder_name).to_numpy()
    pmean = lambda k, how: getattr(np, how)(np.where(scored, cells[k].astype(np.float32), np.nan), axis=(1, 2))
    with np.errstate(all='ignore'):
        cand = {'hand_m': (cell('hand_m').ravel(), BINS['hand_m']),
                'elevation_m': (cell('elevation_m').ravel(), BINS['elevation_m']),
                'slope_deg': (cell('slope_deg').ravel(), BINS['slope_deg']),
                'uda_log10': (cell('uda_log10').ravel(), BINS['uda_log10']),
                'ndvi': (cell('ndvi').ravel(), BINS['ndvi']), 'ndbi': (cell('ndbi').ravel(), BINS['ndbi']),
                'clay_pct': (cell('clay_pct').ravel(), 'q'), 'sand_pct': (cell('sand_pct').ravel(), 'q'),
                'dist_pw_cells': (cell('dist_pw_cells').ravel(), [1, 2, 4, 11, 255, 256]),
                'patch_pw_share': (patch(pw), [0, 1e-9, 0.01, 0.05, 0.20, INF]),
                'patch_mean_slope': (patch(pmean('slope_deg', 'nanmean')), [0, 1, 2, 5, 10, INF]),
                'patch_max_uda': (patch(pmean('uda_log10', 'nanmax')), BINS['uda_log10']),
                'patch_median_ndvi': (patch(pmean('ndvi', 'nanmedian')), BINS['ndvi']),
                'patch_mean_clay': (patch(pmean('clay_pct', 'nanmean')), 'q'),
                'patch_mean_sand': (patch(pmean('sand_pct', 'nanmean')), 'q'),
                'product_flooded_area': (patch(area), [0, 1, 10, 100, INF])}
    feats = []
    for name, (x, edges) in cand.items():
        valid = sc_flat & np.isfinite(x) & (regc[rows_idx] >= 0)
        if isinstance(edges, str):
            edges = [-INF, *np.quantile(x[valid], [0.2, 0.4, 0.6, 0.8]), INF]
        cls = np.digitize(x, edges[1:-1])
        C = len(edges) - 1
        key = (regc[rows_idx] * K + clus[rows_idx]) * C + cls
        n_all = np.bincount(key[valid], minlength=2 * K * C).reshape(2, K, C)
        n_fl = np.bincount(key[valid & fl_flat], minlength=2 * K * C).reshape(2, K, C)
        per_fit = {}
        for k in fits:
            p = P[k].ravel()
            tp = np.bincount(key[valid & p & fl_flat], minlength=2 * K * C).reshape(2, K, C)
            fp = np.bincount(key[valid & p & ~fl_flat], minlength=2 * K * C).reshape(2, K, C)
            per_fit[k] = np.stack([tp, fp, n_fl - tp, n_all - n_fl - fp], -1).astype(float)   # (2, K, C, 4)
        for c in range(C):
            rec = {'feature': name, 'class': c, 'lower': edges[c], 'upper': edges[c + 1]}
            for ri, r in enumerate(('seen', 'unseen')):
                rec[f'{r}_share_cells'] = n_all[ri, :, c].sum() / n_all[ri].sum()
                rec[f'{r}_flooded_share'] = n_fl[ri, :, c].sum() / max(n_all[ri, :, c].sum(), 1)
            for f in FAM:
                full, reps, sv, uv = ({m: [] for m in ('iou', 'pod', 'pofd')} for _ in range(4))
                for k in T0[f]:
                    a_ = per_fit[k][:, :, c, :]                                  # (2, K, 4)
                    ms = {ri: metrics(a_[ri].sum(0)) for ri in (0, 1)}
                    rp = {ri: metrics(W @ a_[ri]) for ri in (0, 1)}
                    for m in full:
                        sv[m].append(ms[0][m]); uv[m].append(ms[1][m])
                        full[m].append(ms[1][m] - ms[0][m]); reps[m].append(rp[1][m] - rp[0][m])
                for m in full:
                    rec[f'{f}_{m}_seen'], rec[f'{f}_{m}_unseen'] = float(np.mean(sv[m])), float(np.mean(uv[m]))
                    rec[f'{f}_{m}_diff'] = float(np.mean(full[m]))
                    rec[f'{f}_{m}_lo'], rec[f'{f}_{m}_hi'] = np.nanpercentile(np.mean(reps[m], 0), [2.5, 97.5])
            feats.append(rec)
        print('screened', name)
    pd.DataFrame(feats).to_csv(D / 'rq2_features.csv', index=False)


if __name__ == '__main__':
    main()
