"""RQ2: basin-seen against basin-unseen test ground within the same region or climate (T0, seed means per family).
Groups: Europe (continent of the product), temperate and cold (Koppen-Geiger main classes C and D) and tropical (A).
Per group and regime: spatial blocks (all and with flood), products, flood cells, flooded fraction; per family the seed
means of IoU, POD, POFD and area bias; unseen - seen of IoU, POD and POFD with a 95% cluster bootstrap interval (the
4,000 x 61 draws of ch5_data.boot_weights()). Reads plot/data/conditions/, writes rq2_regions.csv there.

    PYTHONDONTWRITEBYTECODE=1 nice -n 10 python chapters/chapter5/plot/scripts/ch5_regime_regions.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
FAM = {f: [f'{f}_t0_s{s}' for s in (1, 2, 3)] for f in ('unet', 'vit')}


def met(c):
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    with np.errstate(all='ignore'):
        return {'iou': tp / (tp + fp + fn), 'pod': tp / (tp + fn), 'pofd': fp / (fp + tn), 'bias': (tp + fp) / (tp + fn)}


def main():
    cells, bits, cb = np.load(D / 'cells_test.npz'), np.load(D / 'preds_test_bits.npz'), np.load(D / 'clusters_boot.npz')
    pt = pd.read_csv(D / 'patches_test.csv.gz', usecols=['ground_unit', 'basin_regime', 'continent', 'climate', 'folder_name'])
    Y = cells['label']
    scored, flood, dry = Y >= 0, Y == 1, Y == 0
    W, clus = cb['weights'].astype(float), cb['cluster'].astype(int)
    K = W.shape[1]
    C = {}
    for f in FAM:
        for k in FAM[f]:
            p = np.unpackbits(bits[k])[:Y.size].reshape(Y.shape).astype(bool)
            C[k] = np.stack([(p & flood).sum((1, 2)), (p & dry).sum((1, 2)), (~p & flood).sum((1, 2)),
                             (~p & dry).sum((1, 2))], 1).astype(float)
    groups = {'Europe': pt.continent.eq('Europe').values,
              'Temperate and cold': pt.climate.isin(['C Temperate', 'D Cold']).values,
              'Tropical': pt.climate.eq('A Tropical').values}
    rows = []
    for gname, g in groups.items():
        r = {'group': gname}
        sel = {reg: g & pt.basin_regime.eq(reg).values for reg in ('seen', 'unseen')}
        for reg, s in sel.items():
            fl_rows = s & (flood.sum((1, 2)) > 0)
            r.update({f'{reg}_blocks': pt.ground_unit[s].nunique(), f'{reg}_blocks_flood': pt.ground_unit[fl_rows].nunique(),
                      f'{reg}_products': pt.folder_name[s].nunique(), f'{reg}_flood_cells': int(flood[s].sum()),
                      f'{reg}_flood_fraction': flood[s].sum() / scored[s].sum()})
        for f in FAM:
            for m in ('iou', 'pod', 'pofd', 'bias'):
                for reg, s in sel.items():
                    r[f'{f}_{m}_{reg}'] = np.mean([met(C[k][s].sum(0))[m] for k in FAM[f]])
            for m in ('iou', 'pod', 'pofd'):
                reps = []
                for k in FAM[f]:
                    v = []
                    for reg in ('seen', 'unseen'):
                        pc = np.zeros((K, 4)); np.add.at(pc, clus[sel[reg]], C[k][sel[reg]])
                        v.append(met(W @ pc)[m])
                    reps.append(v[1] - v[0])
                r[f'{f}_{m}_diff'] = r[f'{f}_{m}_unseen'] - r[f'{f}_{m}_seen']
                r[f'{f}_{m}_lo'], r[f'{f}_{m}_hi'] = np.nanpercentile(np.mean(reps, 0), [2.5, 97.5])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(D / 'rq2_regions.csv', index=False)
    print(out.round(3).T.to_string())


if __name__ == '__main__':
    main()
