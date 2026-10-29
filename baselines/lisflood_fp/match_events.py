"""Match the CEMS test products of the thesis to the simulated floods of the EFAS/LISFLOOD-FP catalogue
(Scarpellini et al., 2026, Zenodo 10.5281/zenodo.19666073).

A European test product matches a simulated event when its mapping date lies between 3 days before the event start
and 10 days after the event end, and the centre of mass of the event lies within 60 km of the centroid of the
product's AOI; the nearest such event is taken. The footprint files of the matched events were then read from the two
archives (Simulation No Protections.zip, Simulation FLOPROS.zip) with rangezip.py, without downloading them whole.

    python match_events.py      ->  data/matches_test_products.csv
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
C = Path('/media/ravi/Jupyter/flood_vit/manuscript/PhD_Thesis/chapters/chapter5/plot/data/conditions')
ANN = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/activations/cems_annotations')


def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    s = pd.read_excel(BASE / 'data' / 'efas_event_tables' / 'Results Simulation No Protections.xlsx')
    s['start'] = pd.to_datetime(s['Start date']); s['end'] = pd.to_datetime(s['End date'])
    p = pd.read_csv(C / 'patches_test.csv.gz', usecols=['folder_name', 'flood', 'continent'])
    pr = p.groupby('folder_name').agg(flood=('flood', 'sum'), cont=('continent', 'first')).reset_index()
    pr = pr[pr.cont == 'Europe']
    loc = []
    for f in pr.folder_name:
        a = gpd.read_file(ANN / f.split('_')[0] / f / 'aoi' / 'aoi.shp').to_crs(4326)
        c = a.geometry.union_all().centroid
        loc.append((f, c.x, c.y))
    pr = pr.merge(pd.DataFrame(loc, columns=['folder_name', 'lon', 'lat']))
    pr['date'] = pd.to_datetime(pr.folder_name.str[-8:])
    out = []
    for _, r in pr.iterrows():
        m = s[(s.start - pd.Timedelta(days=3) <= r.date) & (r.date <= s.end + pd.Timedelta(days=10))]
        if m.empty:
            continue
        d = hav(r.lon, r.lat, m['Center of mass Lon'].values, m['Center of mass Lat'].values)
        k = int(np.argmin(d))
        if d[k] <= 60:
            e = m.iloc[k]
            out.append((r.folder_name, r.date.date(), int(r.flood), e['Event_id'], e['Start date'], e['End date'],
                        round(float(d[k]), 1), e['Flooded Area (km2)']))
    o = pd.DataFrame(out, columns=['product', 'date', 'our_flood_cells', 'sim_event', 'start', 'end', 'dist_km',
                                   'sim_area_km2'])
    o.to_csv(BASE / 'data' / 'matches_test_products.csv', index=False)
    print(f'{len(pr)} European test products; {len(o)} matched to a simulated event')


if __name__ == '__main__':
    main()
