"""Change from T0 to T30 by the size of the mapping product relative to the grid of the dynamic hydroclimatic variables
(Ravi, 15 Oct 2026: small events fall within one IMERG or SMAP cell of about 10 km, large events cover many).

Per test mapping product: the number of distinct IMERG cells (0.1 degree) and SMAP cells (9 km) that its test patches
occupy, counted as the distinct 30-day series of local precipitation and of local soil moisture among its patches
(each patch takes the native cell containing its centre, Section 4.1.3; experiments/metadata/final/upstream/weather),
and its labelled flooded area. Groups: IMERG and SMAP cells 1, 2-3, 4-10, > 10; labelled flooded area < 1, 1-10,
10-100, >= 100 km2. Per group and family, pooled over the scored cells of the products of the group: POD, POFD, micro
IoU and predicted flood cells of T0 and T30 (seed means), and the T30 - T0 changes, paired by seed (predicted flood
cells in per cent), with 95% cluster bootstrap intervals (clusters_boot.npz: 61 clusters, 4,000 replicates).
Writes chapters/chapter5/plot/data/rq3/aoi_size.csv and aoi_size_products.csv.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/ch5_rq3_aoi_size_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

WX = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/final/upstream/weather')
D = Path(__file__).resolve().parents[1] / 'data'
GROUPS = {'imerg_cells': ([0, 1, 3, 10, np.inf], ('1', '2–3', '4–10', '>10')),
          'smap_cells': ([0, 1, 3, 10, np.inf], ('1', '2–3', '4–10', '>10')),
          'flood_km2': ([-1, 1, 10, 100, np.inf], ('<1', '1–10', '10–100', '≥100'))}


def distinct(a):
    """Number of distinct rows of a (patches x days), missing values included as a value of their own."""
    return len({np.nan_to_num(r, nan=-9999.0).tobytes() for r in a})


def products(pt):
    rows = []
    for f, g in pt.groupby('folder_name'):
        z = np.load(WX / f'{f}.npz')
        idx = {int(n): i for i, n in enumerate(z['patch_number'])}
        i = [idx[int(n)] for n in g.patch_number]
        rows.append({'folder_name': f, 'patches': len(g), 'imerg_cells': distinct(z['local_precip'][i]),
                     'smap_cells': distinct(z['local_sm'][i]), 'flood_km2': g.flood.sum() * 0.0064,
                     'scored_km2': g.scored.sum() * 0.0064})
    return pd.DataFrame(rows).set_index('folder_name')


def metrics(T, F, N, S):
    with np.errstate(all='ignore'):
        return {'iou': T / (T + F + N), 'pod': T / (T + N), 'pofd': F / (S - T - N), 'pred': T + F}


def main():
    pt = pd.read_csv(D / 'conditions/patches_test.csv.gz').sort_values('row_index').reset_index(drop=True)
    cb = np.load(D / 'conditions/clusters_boot.npz')
    W, clus = cb['weights'].astype(float), cb['cluster'].astype(int)
    K = W.shape[1]
    pr = products(pt)
    pr.to_csv(D / 'rq3/aoi_size_products.csv')
    print(pr.describe().round(1).to_string())
    print('Spearman IMERG cells vs scored area', round(pr[['imerg_cells', 'scored_km2']].corr('spearman').iloc[0, 1], 3))
    agg = lambda v: np.bincount(clus, v, minlength=K).astype(float)
    out = []
    for gname, (edges, labels) in GROUPS.items():
        cls = pd.cut(pr[gname], edges, labels=False)
        pc = pt.folder_name.map(cls).to_numpy()
        for ci, lab in enumerate(labels):
            m = (pc == ci).astype(float)
            sel = cls == ci
            rec = {'group': gname, 'class': lab, 'products': int(sel.sum()),
                   'flood_positive': int((sel & (pr.flood_km2 > 0)).sum()), 'flood_cells': int((pt.flood * m).sum()),
                   'clusters': len(set(clus[(m > 0) & (pt.flood.to_numpy() > 0)]))}
            for f in ('unet', 'vit'):
                full, reps = {}, {}
                for w in ('t0', 't30'):
                    full[w], reps[w] = [], []
                    for s in (1, 2, 3):
                        c = [agg(pt[f'{f}_{w}_s{s}_{x}'].to_numpy() * m) for x in ('tp', 'fp', 'fn')]
                        c.append(agg(pt.scored.to_numpy() * m))
                        full[w].append(metrics(*(x.sum() for x in c)))
                        reps[w].append(metrics(*(W @ x for x in c)))
                for w in ('t0', 't30'):
                    for k in ('iou', 'pod', 'pofd'):
                        rec[f'{f}_{w}_{k}'] = float(np.mean([d[k] for d in full[w]]))
                for k in ('iou', 'pod', 'pofd'):
                    rec[f'{f}_d{k}'] = float(np.mean([full['t30'][i][k] - full['t0'][i][k] for i in range(3)]))
                    r = np.mean([reps['t30'][i][k] - reps['t0'][i][k] for i in range(3)], 0)
                    rec[f'{f}_d{k}_lo'], rec[f'{f}_d{k}_hi'] = np.nanpercentile(r, [2.5, 97.5])
                rec[f'{f}_dpred'] = 100 * (np.mean([full['t30'][i]['pred'] / full['t0'][i]['pred'] for i in range(3)]) - 1)
                with np.errstate(all='ignore'):
                    r = 100 * (np.mean([reps['t30'][i]['pred'] / reps['t0'][i]['pred'] for i in range(3)], 0) - 1)
                rec[f'{f}_dpred_lo'], rec[f'{f}_dpred_hi'] = np.nanpercentile(r, [2.5, 97.5])
            out.append(rec)
    o = pd.DataFrame(out)
    o.to_csv(D / 'rq3/aoi_size.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['group', 'class', 'products', 'flood_cells', 'clusters'] + [f'{f}_{k}' for f in ('unet', 'vit') for k in
                                                                     ('dpred', 'dpod', 'dpofd', 'diou')]
    print(o[cols].round(3).to_string(index=False))
    print(o[[c for c in o.columns if c.endswith(('_lo', '_hi'))] + ['group', 'class']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
