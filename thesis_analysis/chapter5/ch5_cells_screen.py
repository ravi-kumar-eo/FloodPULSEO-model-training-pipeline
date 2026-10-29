"""Screening of U-Net-T0 against ViT-T0 by input condition, per 80 m cell (RQ1). Runs where the condition tables are
(Bolzano: /home/ravi/shared_data/global_flood_dataset/ravi/ch5_conditions/, or plot/data/conditions/ locally):

    python3 ch5_cells_screen.py <conditions dir>

Inputs: cells_test.npz and preds_test_bits.npz (ch5_cells_build.py), clusters_boot.npz (the dependence cluster of
every test row and the 4,000 x 61 cluster draws of ch5_data.boot_weights(), the draws behind every other interval
of the chapter).
For every feature and class, on the scored cells (label >= 0) of the class:
  share of all scored cells and of all flood cells; clusters with flood cells in the class;
  per family, the mean over seeds 1-3 of: probability of detection POD = TP / (TP + FN) (= recall), probability of
  false detection POFD = FP / (FP + TN) (false alarms over all reference non-flood cells), IoU = TP / (TP + FP + FN),
  and the class's share of all FP and of all FN of the fit;
  ViT - U-Net of POD, POFD and IoU, with a 95% cluster bootstrap interval (whole clusters drawn
  with replacement, the same draws for both families, seed means recomputed in every replicate).
Classes: physical bounds (HAND, elevation, slope, UDA, NDVI, NDBI), quintiles of the scored cells (clay, sand),
distance in 80 m cells to the nearest permanent-water or mixed cell of the patch, and the patch class of the
largest UDA of the patch. Patch level (level 'patch'): every scored cell takes the class of its patch, from the
patch's median HAND, share of scored cells with HAND below 2 m (low ground), mean slope, largest UDA, share of
permanent-water and mixed cells, median NDVI, median elevation and mean clay and sand content. Cells without a value
form no class and are counted in 'no value'. Event level (level 'event'): every scored cell takes the class of its
mapping product: flooded area (km2 of 80 m flood cells; flood-free products have no class), Koppen-Geiger main class
and resolution class of the reference map (product_features.csv, built by rq1_features.py).
Output: screen_rq1.csv in the same directory.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(sys.argv[1])
FITS = {f: [f'{f}_t0_s{s}' for s in (1, 2, 3)] for f in ('unet', 'vit')}
INF = np.inf
BINS = {
    'hand_m': ([0, 1, 2, 5, 10, 20, 50, INF], '<1|1-2|2-5|5-10|10-20|20-50|>=50 m'),
    'elevation_m': ([-INF, 10, 50, 200, 500, INF], '<10|10-50|50-200|200-500|>=500 m'),
    'slope_deg': ([0, 0.5, 1, 2, 5, 10, INF], '<0.5|0.5-1|1-2|2-5|5-10|>=10 deg'),
    'uda_log10': ([0, np.log10(2), np.log10(11), np.log10(101), np.log10(1001), np.log10(10001), INF],
                  '<1|1-10|10-100|100-1000|1000-10000|>=10000 km2'),
    'ndvi': ([-1.0001, 0, 0.2, 0.4, 0.6, 1.0001], '<0|0-0.2|0.2-0.4|0.4-0.6|>=0.6'),
    'ndbi': ([-1.0001, -0.2, 0, 0.2, 1.0001], '<-0.2|-0.2-0|0-0.2|>=0.2'),
    'patch_median_hand': ([0, 2, 5, 10, 20, 50, INF], '<2|2-5|5-10|10-20|20-50|>=50 m'),
    'patch_lowland_share': ([0, 1e-9, 0.1, 0.25, 0.5, 1.0001], 'none|<10%|10-25%|25-50%|>=50%'),
    'patch_mean_slope': ([0, 1, 2, 5, 10, INF], '<1|1-2|2-5|5-10|>=10 deg'),
    'patch_pw_share': ([0, 1e-9, 0.01, 0.05, 0.2, 1.0001], 'none|<1%|1-5%|5-20%|>=20%'),
}
BINS['patch_max_uda'] = BINS['uda_log10']
BINS['patch_median_ndvi'] = BINS['ndvi']
BINS['patch_median_elevation'] = BINS['elevation_m']


def classes(name, x, scored):
    """Class index per cell (-1: no value) and the class labels."""
    x = x.astype(np.float32)
    if name in BINS:
        edges, labels = BINS[name]
        labels = labels.split('|')
    else:                                                       # quintiles of the scored cells
        q = np.nanquantile(x[scored], [0.2, 0.4, 0.6, 0.8])
        edges = [-INF, *q, INF]
        labels = [f'Q{i + 1} ({a:.0f}-{b:.0f})' if np.isfinite(a) and np.isfinite(b) else f'Q{i + 1}'
                  for i, (a, b) in enumerate(zip([np.nanmin(x[scored]), *q], [*q, np.nanmax(x[scored])]))]
    c = np.digitize(x, edges[1:-1]).astype(np.int8)
    c[~np.isfinite(x)] = -1
    return c, labels


def main():
    cells = np.load(D / 'cells_test.npz')
    bits = np.load(D / 'preds_test_bits.npz')
    cb = np.load(D / 'clusters_boot.npz')
    Y = cells['label']
    n = Y.shape[0]
    scored, flood, dry = Y >= 0, Y == 1, Y == 0
    clus = np.broadcast_to(cb['cluster'].astype(np.int64)[:, None, None], Y.shape)
    K = int(clus.max()) + 1
    W = cb['weights'].astype(np.float64)                        # (4000, 61)
    P = {k: np.unpackbits(bits[k])[:Y.size].reshape(Y.shape).astype(bool) for f in FITS for k in FITS[f]}

    feats = {k: cells[k] for k in ('hand_m', 'elevation_m', 'slope_deg', 'uda_log10', 'ndvi', 'ndbi',
                                    'clay_pct', 'sand_pct')}
    cls = {k: classes(k, v, scored) for k, v in feats.items()}
    d = cells['dist_pw_cells']
    dc = np.select([d <= 1, d <= 3, d <= 10, d < 255], [0, 1, 2, 3], 4).astype(np.int8)
    cls['dist_pw_cells'] = (dc, ['adjacent (1 cell)', '2-3 cells', '4-10 cells', '>10 cells', 'none in patch'])
    def per_patch(x, how):
        v = np.where(scored, x.astype(np.float32), np.nan).reshape(n, -1)
        with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore', RuntimeWarning)
            return {'median': np.nanmedian, 'mean': np.nanmean, 'max': np.nanmax}[how](v, axis=1)
    hand = cells['hand_m'].astype(np.float32)
    low = np.where(scored & np.isfinite(hand), hand < 2, np.nan).reshape(n, -1)
    with np.errstate(invalid='ignore'), __import__('warnings').catch_warnings():
        __import__('warnings').simplefilter('ignore', RuntimeWarning)
        lowshare = np.nanmean(low, axis=1)
    patch = {'patch_median_hand': per_patch(hand, 'median'), 'patch_lowland_share': lowshare,
             'patch_mean_slope': per_patch(cells['slope_deg'], 'mean'),
             'patch_max_uda': per_patch(cells['uda_log10'], 'max'),
             'patch_pw_share': (Y == -1).reshape(n, -1).mean(axis=1),
             'patch_median_ndvi': per_patch(cells['ndvi'], 'median'),
             'patch_median_elevation': per_patch(cells['elevation_m'], 'median'),
             'patch_mean_clay': per_patch(cells['clay_pct'], 'mean'),
             'patch_mean_sand': per_patch(cells['sand_pct'], 'mean')}
    for k, v in patch.items():
        pc, labs = classes(k, v, np.ones(n, bool))
        cls[k] = (np.broadcast_to(pc[:, None, None], Y.shape), labs)

    # event level: every cell takes the class of its mapping product (product_features.csv of rq1_features.py)
    pt = pd.read_csv(D / 'patches_test.csv.gz', usecols=['row_index', 'folder_name'])
    assert (pt.row_index.values == np.arange(n)).all()
    pf = pd.read_csv(D / 'product_features.csv').set_index('folder_name').reindex(pt.folder_name)
    area = pf.area_km2.to_numpy(float)
    ac = np.digitize(area, [1, 10, 100]).astype(np.int8)
    ac[~(area > 0)] = -1                                           # the flood-free products have no flooded-area class
    events = {'event_flooded_area': (ac, ['<1', '1-10', '10-100', '>=100 km2'])}
    for col, name, labs in (('climate', 'event_climate', ['A Tropical', 'B Arid', 'C Temperate', 'D Cold', 'E Polar']),
                            ('resolution_class', 'event_resolution', ['very-high', 'high', 'medium'])):
        code = {l: i for i, l in enumerate(labs)}
        events[name] = (np.array([code.get(v, -1) for v in pf[col]], np.int8), labs)
    for k, (pc, labs) in events.items():
        cls[k] = (np.broadcast_to(pc[:, None, None], Y.shape), labs)

    rows = []
    for name, (c, labels) in cls.items():
        C = len(labels)
        valid = scored & (c >= 0)
        key = np.where(valid, c.astype(np.int64) * K + clus, -1)
        def counts(mask):                                        # (C, K) counts of mask cells per class and cluster
            m = mask & valid
            return np.bincount(key[m], minlength=C * K).reshape(C, K).astype(np.float64)
        Nf, Nd = counts(flood), counts(dry)
        stats = {}
        for k in [k for f in FITS for k in FITS[f]]:
            tp, fp = counts(P[k] & flood), counts(P[k] & dry)
            stats[k] = (tp, fp, Nf - tp, Nd - fp)              # TP, FP, FN, TN per class and cluster
        def metric(tp, fp, fn, tn):
            with np.errstate(invalid='ignore', divide='ignore'):
                return {'pod': tp / (tp + fn), 'pofd': fp / (fp + tn), 'iou': tp / (tp + fp + fn)}
        full = {k: metric(*(a.sum(1) for a in v)) for k, v in stats.items()}
        boot = {k: metric(*(v_ @ W.T for v_ in v)) for k, v in stats.items()}   # (C, 4000)
        tot = {k: (stats[k][1].sum(), stats[k][2].sum()) for k in stats}       # all FP, all FN of the fit
        for i, lab in enumerate(labels):
            r = {'level': name.split('_')[0] if name.split('_')[0] in ('patch', 'event') else 'cell', 'feature': name, 'class': lab, 'cells': int((Nf + Nd)[i].sum()), 'flood_cells': int(Nf[i].sum()),
                 'share_cells': (Nf + Nd)[i].sum() / (Nf + Nd).sum(), 'share_flood': Nf[i].sum() / Nf.sum(),
                 'flood_fraction': Nf[i].sum() / (Nf + Nd)[i].sum(),
                 'clusters_with_flood': int((Nf[i] > 0).sum()), 'clusters_with_cells': int(((Nf + Nd)[i] > 0).sum())}
            for f in FITS:
                for m in ('pod', 'pofd', 'iou'):
                    r[f'{f}_{m}'] = np.mean([full[k][m][i] for k in FITS[f]])
                r[f'{f}_fp_share'] = np.mean([stats[k][1][i].sum() / tot[k][0] for k in FITS[f]])
                r[f'{f}_fn_share'] = np.mean([stats[k][2][i].sum() / tot[k][1] for k in FITS[f]])
            for m in ('pod', 'pofd', 'iou'):
                rep = np.mean([boot[k][m][i] for k in FITS['vit']], 0) - np.mean([boot[k][m][i] for k in FITS['unet']], 0)
                r[f'diff_{m}'] = r[f'vit_{m}'] - r[f'unet_{m}']
                r[f'diff_{m}_lo'], r[f'diff_{m}_hi'] = np.nanpercentile(rep, [2.5, 97.5])
            rows.append(r)
        nov = scored & (c < 0)
        rows.append({'level': name.split('_')[0] if name.split('_')[0] in ('patch', 'event') else 'cell', 'feature': name, 'class': 'no value', 'cells': int(nov.sum()), 'flood_cells': int((nov & flood).sum()),
                     'share_cells': nov.sum() / scored.sum(), 'share_flood': (nov & flood).sum() / flood.sum()})
        print(f'{name}: {C} classes, no value {nov.sum() / scored.sum():.4f} of scored cells', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(D / 'screen_rq1.csv', index=False)
    print(f'wrote {D / "screen_rq1.csv"}: {len(out)} rows')


if __name__ == '__main__':
    main()
