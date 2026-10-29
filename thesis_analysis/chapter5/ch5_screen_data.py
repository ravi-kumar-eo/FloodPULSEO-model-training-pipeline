"""Section 5.7 — exploratory screen (account 4, 19 Oct 2026, presentation day; moved into the thesis 20 Oct 2026 on Ravi's
request, Q8: add it with all its limitations). Where does ViT beat U-Net (T0), and where do daily inputs help (T30 or any length vs T0)? Patch-level classes,
micro IoU pooled over the patches of a class per fit, seed-paired differences, 95% cluster bootstrap over the 61
dependence clusters with the thesis draws (clusters_boot.npz). Read-only on thesis data."""
import numpy as np
import pandas as pd

D = '/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data'
p = pd.read_csv(f'{D}/conditions/patches_test.csv.gz')
cb = np.load(f'{D}/conditions/clusters_boot.npz')
clus, W = cb['cluster'], cb['weights']                       # (51606,), (4000, 61)
assert len(clus) == len(p)
ante = pd.read_csv(f'{D}/rq3/patch_antecedent.csv', usecols=['folder_name', 'patch_number', 'api_up_0.90',
                                                              'sm_before', 'area_km2'])
p = p.merge(ante, on=['folder_name', 'patch_number'], how='left', validate='1:1')
pf = pd.read_csv(f'{D}/rq1/product_features.csv')[['folder_name', 'flood_cells', 'flood_fraction']]
pf = pf.rename(columns={'flood_cells': 'prod_flood_cells', 'flood_fraction': 'prod_flood_fraction'})
p = p.merge(pf, on='folder_name', how='left', validate='m:1')
p['patch_flood_frac'] = p['flood'] / p['scored'].clip(lower=1)
p['pw_share'] = p['pw_mixed'] / (p['scored'] + p['pw_mixed']).clip(lower=1)
p['prod_flood_km2'] = p['prod_flood_cells'] * 0.0064

def tert(x):
    q = x.quantile([1 / 3, 2 / 3]).values
    return pd.cut(x, [-np.inf, q[0], q[1], np.inf], labels=[f'<{q[0]:.3g}', f'{q[0]:.3g}-{q[1]:.3g}', f'>={q[1]:.3g}'])

CLASSES = {
    'ndvi_median': pd.cut(p['ndvi_median'], [-1, 0.2, 0.4, 0.6, 1.01], labels=['<0.2', '0.2-0.4', '0.4-0.6', '>=0.6']),
    'hand_median': pd.cut(p['hand_m_median'], [-1, 2, 5, 10, 50, np.inf], labels=['<2', '2-5', '5-10', '10-50', '>=50']),
    'slope_mean': pd.cut(p['slope_deg_mean'], [-1, 1, 2, 5, np.inf], labels=['<1', '1-2', '2-5', '>=5']),
    'uda_max_km2': pd.cut(p['uda_log10_max'], [-1, 2, 3, 4, np.inf], labels=['<100', '100-1000', '1000-10000', '>=10000']),
    'pw_share': pd.cut(p['pw_share'], [-1, 0, 0.05, 0.2, 1.01], labels=['none', '<5%', '5-20%', '>=20%']),
    'ndbi_median': pd.cut(p['ndbi_median'], [-1.01, -0.2, 0, 1.01], labels=['<-0.2', '-0.2-0', '>=0']),
    'patch_flood_frac': pd.cut(p['patch_flood_frac'], [-1, 0, 0.1, 0.5, 1.01], labels=['dry', '<10%', '10-50%', '>=50%']),
    'prod_flood_km2': pd.cut(p['prod_flood_km2'], [-1, 10, 100, np.inf], labels=['<10', '10-100', '>=100']),
    'climate': p['climate'].astype('category'),
    'api_up': tert(p['api_up_0.90']),
    'sm_day-4': tert(p['sm_before']),
    'catchment_km2': tert(p['area_km2']),
    'sand_mean': tert(p['sand_pct_mean']),
}
nC = W.shape[1]

def counts(fit):
    return p[f'{fit}_tp'].values, p[f'{fit}_fp'].values, p[f'{fit}_fn'].values

def iou_by(mask, fit):
    tp, fp, fn = (c[mask] for c in counts(fit))
    cl = clus[mask]
    agg = np.zeros((3, nC))
    for k, c in enumerate((tp, fp, fn)):
        np.add.at(agg[k], cl, c)
    point = agg[0].sum() / max(agg.sum(), 1)
    rep = W @ agg.T                                          # (4000, 3)
    boot = rep[:, 0] / np.clip(rep.sum(1), 1, None)
    return point, boot

COMPARE = {
    'ViT-U-Net T0': [(f'vit_t0_s{s}', f'unet_t0_s{s}') for s in (1, 2, 3)],
    'U-Net T30-T0': [(f'unet_t30_s{s}', f'unet_t0_s{s}') for s in (1, 2, 3)],
    'ViT T30-T0': [(f'vit_t30_s{s}', f'vit_t0_s{s}') for s in (1, 2, 3)],
    'U-Net any-T0': [(f'unet_t{t}_s{s}', f'unet_t0_s{s}') for s in (1, 2, 3) for t in (3, 7, 15, 30)],
    'ViT any-T0': [(f'vit_t{t}_s{s}', f'vit_t0_s{s}') for s in (1, 2, 3) for t in (3, 7, 15, 30)],
}
rows = []
for feat, cls in CLASSES.items():
    for c in cls.cat.categories:
        m = (cls == c).values
        if m.sum() < 200:
            continue
        r = dict(feature=feat, cls=str(c), patches=int(m.sum()), clusters=int(len(np.unique(clus[m]))),
                 flood_share=p.loc[m, 'flood'].sum() / p['flood'].sum())
        for name, pairs in COMPARE.items():
            diffs, boots, base = [], [], []
            for a, b in pairs:
                pa, ba = iou_by(m, a)
                pb, bb = iou_by(m, b)
                diffs.append(pa - pb)
                boots.append(ba - bb)
                base.append(pb)
            d = np.mean(diffs)
            bt = np.mean(boots, axis=0)
            lo, hi = np.nanpercentile(bt, [2.5, 97.5])
            r[f'{name} base'] = np.mean(base)
            r[f'{name} d'] = d
            r[f'{name} lo'], r[f'{name} hi'] = lo, hi
            r[f'{name} pos'] = f'{sum(x > 0 for x in diffs)}/{len(diffs)}'
        rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(f'{D}/conditions/screen_patterns.csv', index=False)
pd.set_option('display.width', 260)
for name in COMPARE:
    t = out[['feature', 'cls', 'patches', 'clusters', 'flood_share', f'{name} base', f'{name} d', f'{name} lo',
             f'{name} hi', f'{name} pos']].copy()
    t['sig'] = np.where((t[f'{name} lo'] > 0) | (t[f'{name} hi'] < 0), '*', '')
    print(f'\n===== {name}'); print(t.round(3).to_string(index=False))
