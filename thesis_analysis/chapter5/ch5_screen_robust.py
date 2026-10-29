"""Section 5.7 — robustness of the patterns of the exploratory screen (ch5_screen_data.py): leave-one-dependence-cluster-out
range, flood-weighted share of clusters with a positive difference, and the patterns by climate and HAND.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_screen_robust.py
"""
import numpy as np, pandas as pd
D = '/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data'
p = pd.read_csv(f'{D}/conditions/patches_test.csv.gz')
cb = np.load(f'{D}/conditions/clusters_boot.npz'); clus = cb['cluster']
p['clus'] = clus

def micro(df, fit):
    tp, fp, fn = df[f'{fit}_tp'].sum(), df[f'{fit}_fp'].sum(), df[f'{fit}_fn'].sum()
    return tp / max(tp + fp + fn, 1)

def diff(df, pairs):
    return np.mean([micro(df, a) - micro(df, b) for a, b in pairs])

ARCH = [(f'vit_t0_s{s}', f'unet_t0_s{s}') for s in (1, 2, 3)]
TU = [(f'unet_t{t}_s{s}', f'unet_t0_s{s}') for s in (1, 2, 3) for t in (3, 7, 15, 30)]
TV = [(f'vit_t{t}_s{s}', f'vit_t0_s{s}') for s in (1, 2, 3) for t in (3, 7, 15, 30)]

def report(name, m, pairs):
    sub = p[m]
    d = diff(sub, pairs)
    loo = [diff(sub[sub.clus != c], pairs) for c in sub.clus.unique()]
    per = sub.groupby('clus')
    cl = [(c, g['flood'].sum(), diff(g, pairs)) for c, g in per if g['flood'].sum() > 0]
    w = np.array([f for _, f, _ in cl]); v = np.array([x for _, _, x in cl])
    top = sorted(cl, key=lambda x: -x[1])[:3]
    share = (w * (v > 0)).sum() / w.sum()
    print(f'{name:48s} d={d:+.3f}  leave-one-cluster-out {min(loo):+.3f}..{max(loo):+.3f}  '
          f'clusters with flood {len(cl)}, flood-weighted share positive {share:.2f}, '
          f'largest: ' + ', '.join(f'{c}:{f}:{x:+.2f}' for c, f, x in top))

nd = p['ndvi_median']
print('--- ViT - U-Net (T0)')
report('NDVI >= 0.6', nd >= 0.6, ARCH)
report('NDVI >= 0.6, climate A tropical', (nd >= 0.6) & (p.climate == 'A Tropical'), ARCH)
report('NDVI >= 0.6, climate C temperate', (nd >= 0.6) & (p.climate == 'C Temperate'), ARCH)
report('NDVI >= 0.6, climate D cold', (nd >= 0.6) & (p.climate == 'D Cold'), ARCH)
report('NDVI >= 0.6 & HAND < 2 m', (nd >= 0.6) & (p.hand_m_median < 2), ARCH)
report('NDVI >= 0.6 & HAND >= 2 m', (nd >= 0.6) & (p.hand_m_median >= 2), ARCH)
report('NDVI < 0.6 (all other)', nd < 0.6, ARCH)
for name, pairs in (('U-Net any - T0', TU), ('ViT any - T0', TV)):
    print('---', name)
    report('NDVI < 0', nd < 0, pairs)
    report('NDVI 0-0.2', (nd >= 0) & (nd < 0.2), pairs)
    report('NDVI < 0.2', nd < 0.2, pairs)
    report('NDVI < 0.2, not arid', (nd < 0.2) & (p.climate != 'B Arid'), pairs)
    report('NDVI < 0.2, arid', (nd < 0.2) & (p.climate == 'B Arid'), pairs)
    report('NDVI >= 0.2', nd >= 0.2, pairs)
print('NDVI<0.2 composition: climate flood share'); print((p[nd < 0.2].groupby('climate')['flood'].sum() / p[nd < 0.2]['flood'].sum()).round(2).to_dict())
print('NDVI<0.2 split: flood share NDVI<0', round(p[nd < 0]['flood'].sum() / p[nd < 0.2]['flood'].sum(), 2), ' pw share mean', round((p[nd<0.2].pw_mixed/(p[nd<0.2].scored+p[nd<0.2].pw_mixed)).mean(),3), 'vs all', round((p.pw_mixed/(p.scored+p.pw_mixed)).mean(),3))
print('NDVI<0.2 median NDBI', round(p[nd<0.2].ndbi_median.median(),3), ' all', round(p.ndbi_median.median(),3))
