"""Check of flood-time imagery in the NDVI/NDBI inputs (Ravi, 21 Oct 2026, option b).

The Sentinel-2 median composite of a mapping product uses the images of the 30 days before its mapping date (window
extended to 60 or 90 days, or gaps filled from the same window of the previous year, when the coverage stays below 98%;
Github/scripts/2_submit_gee_tasks.py, build_s2_indices). The image dates of each composite were not stored for the full
dataset, so post-onset exposure is bounded from below by the days of the window after the first mapping date of the
activation, when the flood had certainly begun: exposure = min(30, mapping date - first mapping date of the activation).

Tests on the test subset (cells of plot/data/conditions/cells_test.npz, label -1 unscored, 0 dry, 1 flood):
  1. Per product: NDVI of the labelled flood cells minus NDVI of the dry scored cells with HAND below 2 m (median of
     each), and the share of flood cells with NDVI < 0, by exposure group.
  2. Consecutive products of the same AOI (bound-matched patches of the same-AOI analysis, plot/data/repeat/): change
     of NDVI from the earlier to the later composite on cells flooded in the earlier label, minus the change on cells dry
     in both labels (difference in differences). The later window covers the earlier flood date when the products lie
     less than 30 days apart, so flood-time images in the later composite would lower the NDVI of the earlier flood.
  3. T0 detection by exposure group (micro POD, POFD, IoU, seed means).
Cluster bootstrap over the dependence clusters of the thesis (clusters_boot.npz), 2,000 draws.

Output  plot/data/ndvi_exposure/{products,groups,pairs,pairs_summary,t0_by_group}.csv

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_ndvi_exposure_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data'
OUT = D / 'ndvi_exposure'
META = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/metadata/released_events_metadata.csv')
RNG = np.random.default_rng(20261021)
GROUPS = [(0, 0, 'first date'), (1, 10, '1-10 days'), (11, 29, '11-29 days'), (30, 30, '30 days')]


def boot(values, clusters, n=2000):
    """Mean over units with a cluster bootstrap; returns mean, lo, hi."""
    v = np.asarray(values, float); c = np.asarray(clusters)
    uc = np.unique(c); idx = {u: np.where(c == u)[0] for u in uc}
    est = [np.nanmean(np.concatenate([v[idx[u]] for u in RNG.choice(uc, len(uc))])) for _ in range(n)]
    return np.nanmean(v), np.nanpercentile(est, 2.5), np.nanpercentile(est, 97.5)


def main():
    OUT.mkdir(exist_ok=True)
    z = np.load(D / 'conditions/cells_test.npz')
    lab, ndvi, ndbi, hand = z['label'], z['ndvi'].astype(np.float32), z['ndbi'].astype(np.float32), z['hand_m'].astype(np.float32)
    p = pd.read_csv(D / 'conditions/patches_test.csv.gz')
    p['cluster'] = np.load(D / 'conditions/clusters_boot.npz')['cluster']
    meta = pd.read_csv(META, usecols=['folder_name', 'mapping_date'])
    meta['act'] = meta.folder_name.str.split('_').str[0]
    meta['date'] = pd.to_datetime(meta.mapping_date)
    first = meta.groupby('act').date.min()
    meta['exposure'] = (meta.date - meta.act.map(first)).dt.days.clip(upper=30)

    # 1. per-product contrasts
    rows = []
    for f, g in p.groupby('folder_name'):
        r = g.index.values
        L, N, B, H = lab[r], ndvi[r], ndbi[r], hand[r]
        fl, dl = (L == 1) & np.isfinite(N), (L == 0) & (H < 2) & np.isfinite(N)
        if fl.sum() < 50 or dl.sum() < 50:
            continue
        rows.append({'folder_name': f, 'cluster': g.cluster.iloc[0], 'flood_cells': int(fl.sum()),
                     'ndvi_flood': float(np.median(N[fl])), 'ndvi_dry_low': float(np.median(N[dl])),
                     'ndvi_contrast': float(np.median(N[fl]) - np.median(N[dl])),
                     'ndbi_contrast': float(np.median(B[fl]) - np.median(B[dl])),
                     'neg_ndvi_flood': float((N[fl] < 0).mean()), 'neg_ndvi_dry_low': float((N[dl] < 0).mean())})
    pr = pd.DataFrame(rows).merge(meta[['folder_name', 'exposure']], on='folder_name', how='left')
    pr['group'] = pd.cut(pr.exposure, [-1, 0, 10, 29, 30], labels=[g for *_, g in GROUPS])
    pr.to_csv(OUT / 'products.csv', index=False)
    grp = []
    for _, _, name in GROUPS:
        s = pr[pr.group == name]
        if s.empty:
            continue
        m, lo, hi = boot(s.ndvi_contrast, s.cluster)
        n0, nl, nh = boot(s.neg_ndvi_flood - s.neg_ndvi_dry_low, s.cluster)
        grp.append({'group': name, 'products': len(s), 'clusters': s.cluster.nunique(), 'ndvi_contrast': m,
                    'lo': lo, 'hi': hi, 'neg_ndvi_excess': n0, 'neg_lo': nl, 'neg_hi': nh,
                    'median_ndvi_flood': s.ndvi_flood.median(), 'median_ndvi_dry_low': s.ndvi_dry_low.median()})
    pd.DataFrame(grp).to_csv(OUT / 'groups.csv', index=False)

    # 2. consecutive products of the same AOI (difference in differences)
    pm = pd.read_csv(D / 'repeat/patch_matches.csv.gz')
    info = pd.read_csv(D / 'repeat/pairs.csv.gz', usecols=['pair_id', 'support_id', 'aoi', 'cluster', 'product_a',
                                                           'product_b', 'date_a', 'date_b', 'days']).drop_duplicates('pair_id')
    prs = []
    for pid, g in pm.groupby('pair_id'):
        a, b = g.row_a.values, g.row_b.values
        La, Lb, Na, Nb = lab[a], lab[b], ndvi[a], ndvi[b]
        ok = np.isfinite(Na) & np.isfinite(Nb)
        fl = (La == 1) & (Lb >= 0) & ok          # flooded at the earlier date, scored at the later one
        dr = (La == 0) & (Lb == 0) & ok          # dry at both dates
        if fl.sum() < 50 or dr.sum() < 50:
            continue
        prs.append({'pair_id': pid, 'flood_cells_a': int(fl.sum()), 'd_ndvi_flood': float((Nb[fl] - Na[fl]).mean()),
                    'd_ndvi_dry': float((Nb[dr] - Na[dr]).mean()),
                    'did': float((Nb[fl] - Na[fl]).mean() - (Nb[dr] - Na[dr]).mean()),
                    'neg_b_minus_a_flood': float((Nb[fl] < 0).mean() - (Na[fl] < 0).mean())})
    pp = pd.DataFrame(prs).merge(info, on='pair_id', how='left')
    pp['window_covers_a'] = pp.days < 30
    pp.to_csv(OUT / 'pairs.csv', index=False)
    summ = []
    for name, s in (('all pairs', pp), ('later window covers earlier date', pp[pp.window_covers_a]),
                    ('30 days or more apart', pp[~pp.window_covers_a])):
        if s.empty:
            continue
        m, lo, hi = boot(s.did, s.cluster)
        n0, nl, nh = boot(s.neg_b_minus_a_flood, s.cluster)
        summ.append({'set': name, 'pairs': len(s), 'aois': s.aoi.nunique(), 'clusters': s.cluster.nunique(),
                     'did_ndvi': m, 'lo': lo, 'hi': hi, 'neg_ndvi_change_flood': n0, 'neg_lo': nl, 'neg_hi': nh,
                     'd_ndvi_flood': s.d_ndvi_flood.mean(), 'd_ndvi_dry': s.d_ndvi_dry.mean()})
    pd.DataFrame(summ).to_csv(OUT / 'pairs_summary.csv', index=False)

    # 3. T0 detection by exposure group
    p = p.merge(meta[['folder_name', 'exposure']], on='folder_name', how='left')
    p['group'] = pd.cut(p.exposure, [-1, 0, 10, 29, 30], labels=[g for *_, g in GROUPS])
    t0 = []
    for name, s in p.groupby('group', observed=True):
        for fam in ('unet', 'vit'):
            iou, pod, pofd = [], [], []
            for sd in (1, 2, 3):
                tp, fp, fn = (s[f'{fam}_t0_s{sd}_{k}'].sum() for k in ('tp', 'fp', 'fn'))
                dry = s.scored.sum() - s.flood.sum()
                iou.append(tp / (tp + fp + fn)); pod.append(tp / (tp + fn)); pofd.append(fp / dry)
            t0.append({'group': name, 'family': fam, 'products': s.folder_name.nunique(), 'flood_share': s.flood.sum() / s.scored.sum(),
                       'iou': np.mean(iou), 'pod': np.mean(pod), 'pofd': np.mean(pofd)})
    pd.DataFrame(t0).to_csv(OUT / 't0_by_group.csv', index=False)
    for f in ('groups', 'pairs_summary', 't0_by_group'):
        print(f'--- {f}'); print(pd.read_csv(OUT / f'{f}.csv').round(4).to_string(index=False))


if __name__ == '__main__':
    main()


def water_signal():
    """Cells whose composite NDVI is negative (open water at the composite dates): their share of the labelled flood,
    the POD of the T0 and T30 networks on them, their share of the detected flood, and micro IoU without them.
    Output plot/data/ndvi_exposure/water_signal.csv"""
    z = np.load(D / 'conditions/cells_test.npz')
    lab, ndvi = z['label'], z['ndvi'].astype(np.float32)
    bits = np.load(D / 'conditions/preds_test_bits.npz')
    shape = tuple(bits['shape'])
    neg = ndvi < 0
    fl, dr = lab == 1, lab == 0
    rows = []
    for fam in ('unet', 'vit'):
        for w in ('t0', 't30'):
            vals = []
            for sd in (1, 2, 3):
                P = np.unpackbits(bits[f'{fam}_{w}_s{sd}'])[:int(np.prod(shape))].reshape(shape).astype(bool)
                tp, fp, fn = P & fl, P & dr, ~P & fl
                keep = ~neg
                vals.append({'pod_neg': tp[neg].sum() / fl[neg].sum(), 'pod_other': tp[keep].sum() / fl[keep].sum(),
                             'tp_share_neg': tp[neg].sum() / tp.sum(),
                             'iou_all': tp.sum() / (tp.sum() + fp.sum() + fn.sum()),
                             'iou_without_neg': tp[keep].sum() / (tp[keep].sum() + fp[keep].sum() + fn[keep].sum())})
            m = pd.DataFrame(vals).mean()
            rows.append({'family': fam, 'window': w, **m.to_dict()})
    out = pd.DataFrame(rows)
    out['flood_share_neg'] = fl[neg].sum() / fl.sum()
    out['dry_share_neg'] = dr[neg].sum() / dr.sum()
    out.to_csv(OUT / 'water_signal.csv', index=False)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__' and 'water' in __import__('sys').argv:
    water_signal()
