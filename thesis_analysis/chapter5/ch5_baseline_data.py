"""Section 5.6 and Appendix A.6 — data of the comparison with the GFF models (Victor et al., 2024).

Reads the scored sites of the GFF comparison (experiments/runs_baselines/gff/results, written by
experiments/src/baselines/gff/score_sites.py) and writes plot/data/baseline/:
  sites.csv   one row per scored site and model: site, event, label type, country, date, overlap with our data,
              scored and flood cells, IoU, POD, POFD, area bias. Ours = mean over seeds 1-3 of one architecture and
              input-history length (never across U-Net and ViT); GFF = the released model of the fold that held the
              site out.
Site rule of the thesis (Ravi, 16 Oct 2026): hand-labelled Kuro Siwo sites in the main text, generated sites in the
appendix; sites that touch our validation subset are left out (267-1, 4573, 4848).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_baseline_data.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path('/media/ravi/Jupyter/flood_vit')
RES = ROOT / 'experiments/runs_baselines/gff/results'
SITES = ROOT / 'experiments/src/baselines/gff/sites.csv'
COUNTRIES = ROOT / 'pipeline/data/continents/ne_110m_admin_0_countries.shp'
OUT = Path(__file__).resolve().parents[1] / 'data' / 'baseline'
VALIDATION_OVERLAP = ('267-1-', '4573-', '4848-')
GFF = {'gff:utae': 'gff_utae', 'gff:metnet': 'gff_maxvit'}       # the two GFF models of the thesis
METRICS = ['IoU', 'POD', 'POFD', 'area_bias']


def country(lon, lat, world):
    p = Point(lon, lat)
    hit = world[world.contains(p)]
    if len(hit):
        return hit.iloc[0].NAME
    return world.iloc[world.distance(p).argmin()].NAME          # coastal site outside the coarse coastline


def main():
    allsites = pd.read_csv(RES / 'all' / 'all_sites.csv')[['site', 'batch']]
    rows = []
    for b in ('ks', 'gen', 'more'):
        s = pd.read_csv(RES / b / 'scores.csv')
        rows.append(s[s.site != 'ALL'].assign(batch=b))
    s = pd.concat(rows).merge(allsites, on=['site', 'batch'])          # the batch in which each site was scored
    assert not s.duplicated(['site', 'model']).any()
    s = s[~s.site.str.startswith(VALIDATION_OVERLAP)]
    ours = s[s.model.str.startswith('ours:')].copy()
    p = ours.model.str.extract(r'ours:final_t(?P<window>\d+)__s(?P<seed>\d)__(?P<arch>unet|vit)')
    ours = pd.concat([ours, p], axis=1)
    ours['group'] = ours.arch + '_t' + ours.window
    ours = ours.groupby(['site', 'group'], as_index=False).agg(
        cells=('cells', 'first'), flood_cells=('flood_cells', 'first'), **{m: (m, 'mean') for m in METRICS})
    gff = s[s.model.isin(GFF)].assign(group=lambda d: d.model.map(GFF))[['site', 'group', 'cells', 'flood_cells']
                                                                        + METRICS]
    d = pd.concat([ours, gff], ignore_index=True)
    meta = pd.read_csv(SITES).rename(columns={'roi': 'site'})
    world = gpd.read_file(COUNTRIES)
    meta['country'] = [country(x, y, world) for x, y in zip(meta.lon, meta.lat)]
    meta.loc[meta.site.str.startswith('445-'), 'country'] = 'Ukraine/Romania'   # on the border; site list name
    meta['label_type'] = meta.label.map(lambda v: 'hand' if v.startswith('hand') else 'generated')
    meta['event'] = meta.site.str.split('-').str[0]
    meta['our_test_event'] = meta.our_split_overlap.eq('test')
    d = d.merge(meta[['site', 'event', 'label_type', 'country', 'flood_date', 'lon', 'lat', 'our_test_event',
                      'our_events']], on='site', how='left')
    assert d.label_type.notna().all()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / 'sites.csv', index=False)
    for lt, g in d.groupby('label_type'):
        print(lt, 'sites', g.site.nunique(), 'events', g.event.nunique(), 'test-event sites',
              g.drop_duplicates('site').our_test_event.sum())
        print(g.groupby('group')[METRICS].median().round(3).to_string())


if __name__ == '__main__':
    main()
