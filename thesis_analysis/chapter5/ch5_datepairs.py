"""Successive mapping dates of the same AOI, for Figure 5.5 (repeated observations, Chapter 4 Section 4.7.5).
Runs ON BOLZANO, where the split's patch footprints, the event metadata and every fit's predictions live:

    python3 ch5_datepairs.py > fig5_5_datepairs.csv           (then copy to chapters/chapter5/plot/data/)

For each test AOI mapped on two or more dates, the products are ordered by mapping date and every pair of successive
dates is compared on the cells scored on BOTH dates (patches matched by footprint: CRS and lower-left corner, as in
review_unet_s1/scripts/explore/a2_dates.py). Per pair: observed flooded cells on each date (O1, O2); per fit: cells
predicted flooded at the fit's validation-selected global threshold (A1, A2) and the IoU on each date, on the same
common cells. Area = cells x 0.0064 km2 (80 m cells). A fit's maps of both dates are compared with its own T0's
(same family and seed) through `t0_max_dp`: the largest probability change of that T0 between the two dates.
Output: one row per date pair and fit. Fits that are not finished are skipped (listed on stderr).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

E = Path('/home/ravi/shared_data/global_flood_dataset/ravi/experiments')
R = E / 'runs_final'
EVENTS = Path('/home/ravi/shared_data/global_flood_dataset/pipeline/data/metadata/released_events_metadata.csv')
FITS = {  # label -> run dir pattern (relative to runs_final), seeds
    'T0': ('final_t0/seed{s}/{f}', (1, 2, 3)), 'T30': ('final_t30/seed{s}/{f}', (1, 2, 3)),
    'T3': ('final_window/seed{s}/{f}_t3', (1, 2, 3)), 'T7': ('final_window/seed{s}/{f}_t7', (1, 2, 3)),
    'T15': ('final_window/seed{s}/{f}_t15', (1, 2, 3)),
    'T30 local': ('final_ablation/seed{s}/{f}_noup', (1,)), 'T30 daily': ('final_ablation/seed{s}/{f}_nosum', (1,)),
    'T30 statistics': ('final_ablation/seed{s}/{f}_sumonly', (1,)), 'T30 random init': ('final_scratch/seed{s}/{f}', (1,)),
}


def finished(d):
    return (d / 'COMPLETE').exists() and (d / 'inference_final/preds/test__prob.f16.npy').exists()


def main():
    ref = R / 'final_t0/seed1/unet/inference_final/preds'
    idx = pd.read_csv(ref / 'test__index.csv')
    Y = np.load(ref / 'test__labels.i8.npy')
    split = pd.read_csv(E / 'metadata/final/split/patches.csv', usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx', 'bounds_miny'])
    ev = pd.read_csv(EVENTS, usecols=['folder_name', 'mapping_date'])
    df = idx.merge(split, on=['folder_name', 'patch_number'], how='left', validate='1:1').merge(ev, on='folder_name', how='left', validate='m:1')
    assert df.crs.notna().all() and df.mapping_date.notna().all()
    df['key'] = df.crs + '|' + df.bounds_minx.round(1).astype(str) + '|' + df.bounds_miny.round(1).astype(str)
    fits, missing = [], []
    for f in ('unet', 'vit'):
        for lab, (pat, seeds) in FITS.items():
            for s in seeds:
                d = R / pat.format(s=s, f=f)
                (fits if finished(d) else missing).append((f, lab, s, d))
    fits = [x[:4] for x in fits]
    P, thr = {}, {}
    for f, lab, s, d in fits:
        if not np.array_equal(np.load(d / 'inference_final/preds/test__labels.i8.npy'), Y):
            raise SystemExit(f'{d}: labels differ from the reference')
        P[(f, lab, s)] = np.load(d / 'inference_final/preds/test__prob.f16.npy', mmap_mode='r')
        thr[(f, lab, s)] = np.float32(json.loads((d / 'inference_final/threshold.json').read_text())['threshold'])
    rows = []
    for aoi, g in df.groupby('aoi'):
        dates = g.groupby('folder_name').mapping_date.first().sort_values()
        if len(dates) < 2:
            continue
        names = list(dates.index)
        for a, b in zip(names[:-1], names[1:]):
            ia = pd.Series(g.index[g.folder_name == a], index=g.key[g.folder_name == a].values)
            ib = pd.Series(g.index[g.folder_name == b], index=g.key[g.folder_name == b].values)
            common = ia.index.intersection(ib.index)
            base = {'aoi': aoi, 'a': a, 'b': b, 'date_a': dates[a], 'date_b': dates[b],
                    'gap_days': (pd.Timestamp(dates[b]) - pd.Timestamp(dates[a])).days, 'n_common_patches': len(common)}
            if not len(common):
                rows.append({**base, 'fit': None})
                continue
            i1, i2 = ia[common].to_numpy(), ib[common].to_numpy()
            m = (Y[i1] >= 0) & (Y[i2] >= 0)
            t1, t2 = (Y[i1] == 1) & m, (Y[i2] == 1) & m
            base.update(cells=int(m.sum()), O1=int(t1.sum()), O2=int(t2.sum()))
            for (f, lab, s) in P:
                p1 = np.asarray(P[(f, lab, s)][i1], np.float32)
                p2 = np.asarray(P[(f, lab, s)][i2], np.float32)
                q1, q2 = (p1 >= thr[(f, lab, s)]) & m, (p2 >= thr[(f, lab, s)]) & m
                u1, u2 = (q1 | t1).sum(), (q2 | t2).sum()
                z0 = (f, 'T0', s)
                t0dp = float(np.abs(np.asarray(P[z0][i1], np.float32) - np.asarray(P[z0][i2], np.float32))[m].max()) \
                    if (z0 in P and m.any()) else np.nan
                rows.append({**base, 'family': f, 'fit': lab, 'seed': s, 'A1': int(q1.sum()), 'A2': int(q2.sum()),
                             'iou1': (q1 & t1).sum() / u1 if u1 else np.nan, 'iou2': (q2 & t2).sum() / u2 if u2 else np.nan,
                             't0_max_dp': t0dp})
    out = pd.DataFrame(rows)
    out.to_csv(sys.stdout, index=False)
    n = out.drop_duplicates(['a', 'b'])
    print(f'{out.aoi.nunique()} AOIs with >= 2 dates, {len(n)} successive pairs, {int((n.n_common_patches > 0).sum())} with '
          f'common patches; products {pd.unique(pd.concat([n.a, n.b])).size}; fits {len(P)}', file=sys.stderr)
    print('not finished (skipped): ' + (', '.join(f'{f} {lab} s{s}' for f, lab, s, _ in missing) or 'none'), file=sys.stderr)


if __name__ == '__main__':
    main()
