"""Section 5.3.5 (b) — individual AOIs with consecutive mapping products of one activation (Ravi, 21 Oct 2026: "not only
all events combined … cases where the temporal inputs improved or worsened the prediction, and in which events, regions or
conditions they help").

Inputs (plot/data/repeat/, from ch5_repeat_data.py): products.csv.gz (IoU, POD, POFD and flooded shares of every product
and training run on the product's own scored cells), pairs.csv.gz (consecutive pairs with the labelled and predicted
changes and the hydroclimatic summaries of both dates), product_inputs.csv; pipeline metadata for continent and climate.

Rules, fixed before any map was drawn:
  gain of an AOI   mean over its flood-positive dates of IoU(input history) - IoU(T0), per architecture and seed
  helps            seed-mean gain >= +0.01 and the gain of every seed > 0
  harms            seed-mean gain <= -0.01 and the gain of every seed < 0
  otherwise        mixed
Cases (AOIs with at least three flood-positive dates): the AOI with the largest T30 gain averaged over both architectures
among AOIs that T30 helps in both, the AOI with the most negative T30 gain averaged over both architectures, and the AOI
with the largest advantage of T7 over T30 averaged over both architectures.

Output  plot/data/repeat/cases/{aoi_gains,aoi_table,conditions,cases}.csv

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_repeat_cases_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data'
R = D / 'repeat'
OUT = R / 'cases'
META = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/metadata/released_events_metadata.csv')
CONFIGS = ('t3', 't7', 't15', 't30')
RNG = np.random.default_rng(20261021)


def boot_mean(v, c, n=2000):
    v, c = np.asarray(v, float), np.asarray(c)
    uc = np.unique(c)
    if len(uc) < 2:
        return np.nanmean(v), np.nan, np.nan
    idx = {u: np.where(c == u)[0] for u in uc}
    est = [np.nanmean(np.concatenate([v[idx[u]] for u in RNG.choice(uc, len(uc))])) for _ in range(n)]
    return np.nanmean(v), np.nanpercentile(est, 2.5), np.nanpercentile(est, 97.5)


def main():
    OUT.mkdir(exist_ok=True)
    pr = pd.read_csv(R / 'products.csv.gz')
    pr = pr[(pr.kind == 'same_activation') & pr.config.isin(('t0',) + CONFIGS)]
    pr = pr[pr.label_cells > 0]
    # per AOI, family, config, seed: mean IoU gain over the dates
    base = pr[pr.config == 't0'][['aoi', 'family', 'seed', 'folder_name', 'iou']].rename(columns={'iou': 'iou_t0'})
    x = pr[pr.config != 't0'].merge(base, on=['aoi', 'family', 'seed', 'folder_name'])
    x['gain'] = x.iou - x.iou_t0
    g = x.groupby(['aoi', 'cluster', 'family', 'config', 'seed']).gain.mean().reset_index()
    s = g.groupby(['aoi', 'cluster', 'family', 'config']).gain.agg(['mean', 'min', 'max']).reset_index()
    s['dates'] = s.aoi.map(pr.groupby('aoi').folder_name.nunique())
    s['cls'] = np.where((s['mean'] >= 0.01) & (s['min'] > 0), 'helps',
                        np.where((s['mean'] <= -0.01) & (s['max'] < 0), 'harms', 'mixed'))
    s.to_csv(OUT / 'aoi_gains.csv', index=False)
    print('--- AOIs per class')
    print(s.groupby(['family', 'config']).cls.value_counts().unstack(fill_value=0))

    # AOI attributes
    meta = pd.read_csv(META, usecols=['folder_name', 'continent', 'climate'])
    first = pr.sort_values('date').groupby('aoi').folder_name.first()
    att = pd.DataFrame({'aoi': first.index, 'first_product': first.values}).merge(
        meta.rename(columns={'folder_name': 'first_product'}), on='first_product', how='left')
    t0 = pr[pr.config == 't0'].groupby(['aoi', 'folder_name', 'date']).agg(label_cells=('label_cells', 'first'),
                                                                         label_share=('label_share', 'first')).reset_index()
    t0 = t0.sort_values(['aoi', 'date'])
    att = att.merge(t0.groupby('aoi').agg(flood_km2=('label_cells', lambda v: v.mean() * 0.0064),
                                          first_share=('label_share', 'first'), last_share=('label_share', 'last')),
                    on='aoi')
    att['course'] = np.where(att.last_share > att.first_share * 1.1, 'expanding',
                             np.where(att.last_share < att.first_share / 1.1, 'receding', 'steady'))
    inp = pd.read_csv(R / 'product_inputs.csv')
    inp = inp[inp.folder_name.isin(pr.folder_name)]
    att = att.merge(inp.groupby('aoi').agg(precip_up_7=('precip_up_7', 'mean'), api_up_30=('api_up_30', 'mean'),
                                           sm_day4=('sm_day4', 'mean')), on='aoi', how='left')
    pa = pd.read_csv(R / 'pairs.csv.gz', usecols=['aoi', 'kind', 'fit', 'delta_precip_up_7', 'delta_sm_day4'])
    pa = pa[(pa.kind == 'same_activation') & (pa.fit == 'unet_t0_s1')]
    att = att.merge(pa.groupby('aoi').agg(abs_dprecip_up_7=('delta_precip_up_7', lambda v: np.abs(v).mean()),
                                          abs_dsm_day4=('delta_sm_day4', lambda v: np.abs(v).mean())),
                    on='aoi', how='left')
    wide = s.pivot_table(index=['aoi', 'cluster'], columns=['family', 'config'], values='mean')
    wide.columns = [f'{f}_{c}' for f, c in wide.columns]
    cls = s.pivot_table(index='aoi', columns=['family', 'config'], values='cls', aggfunc='first')
    cls.columns = [f'cls_{f}_{c}' for f, c in cls.columns]
    tab = att.merge(wide.reset_index(), on='aoi').merge(cls.reset_index(), on='aoi')
    tab['dates'] = tab.aoi.map(pr.groupby('aoi').folder_name.nunique())
    tab.to_csv(OUT / 'aoi_table.csv', index=False)

    # conditions: mean gain by groups of AOIs (cluster bootstrap)
    def terc(col):
        q = tab[col].quantile([1 / 3, 2 / 3]).values
        return pd.cut(tab[col], [-np.inf, q[0], q[1], np.inf], labels=['low', 'middle', 'high'])
    groupings = {'continent': tab.continent.where(tab.continent.isin(['Europe', 'Asia']), 'other'),
                 'climate': tab.climate.str[0].map({'A': 'tropical', 'B': 'arid', 'C': 'temperate', 'D': 'cold', 'E': 'polar'}),
                 'course': tab.course, 'flooded area': terc('flood_km2'), 'upstream precipitation, 7 days': terc('precip_up_7'),
                 'change of 7-day precipitation': terc('abs_dprecip_up_7'), 'soil moisture, day -4': terc('sm_day4')}
    rows = []
    for name, grp in groupings.items():
        for level in [l for l in pd.unique(grp.dropna())]:
            sub = tab[grp == level]
            for fam in ('unet', 'vit'):
                for cfg in ('t7', 't30'):
                    m, lo, hi = boot_mean(sub[f'{fam}_{cfg}'], sub.cluster)
                    rows.append({'condition': name, 'level': level, 'aois': len(sub), 'clusters': sub.cluster.nunique(),
                                 'family': fam, 'config': cfg, 'gain': m, 'lo': lo, 'hi': hi,
                                 'helps': int((sub[f'cls_{fam}_{cfg}'] == 'helps').sum()),
                                 'harms': int((sub[f'cls_{fam}_{cfg}'] == 'harms').sum())})
    cond = pd.DataFrame(rows)
    cond.to_csv(OUT / 'conditions.csv', index=False)

    # cases (AOIs with at least three flood-positive dates; both architectures averaged)
    e = tab[tab.dates >= 3].copy()
    e['t30_both'] = (e.unet_t30 + e.vit_t30) / 2
    e['t7_minus_t30'] = (e.unet_t7 + e.vit_t7) / 2 - e.t30_both
    both_help = e[(e.cls_unet_t30 == 'helps') & (e.cls_vit_t30 == 'helps')].sort_values('t30_both', ascending=False)
    sel = [('T30 helps both architectures', both_help.iloc[0], len(both_help)),
           ('most negative T30 gain', e.sort_values('t30_both').iloc[0], len(e)),
           ('largest T7 advantage over T30', e.sort_values('t7_minus_t30', ascending=False).iloc[0], len(e))]
    cases = [{'case': k, 'aoi': r.aoi, 'candidates': n, 'dates': r.dates, 'continent': r.continent, 'climate': r.climate,
              'course': r.course, 'flood_km2': r.flood_km2, 'unet_t7': r.unet_t7, 'vit_t7': r.vit_t7,
              'unet_t30': r.unet_t30, 'vit_t30': r.vit_t30} for k, r, n in sel]
    pd.DataFrame(cases).to_csv(OUT / 'cases.csv', index=False)
    print('--- cases'); print(pd.DataFrame(cases).round(3).to_string(index=False))
    print('--- conditions (T30)'); print(cond[cond.config == 't30'].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
