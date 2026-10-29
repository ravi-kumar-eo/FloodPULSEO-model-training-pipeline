"""RQ2 — HydroBASINS Level-5 basins of the test spatial blocks, for the basin map (Figure 5.12).

The split records for each mapping product the Level-5 Pfafstetter codes of its basins (basin_tokens, the first five
digits of the Level-12 PFAF_ID of the Level-12 basins that intersect the product). Each Level-5 basin of a basin-seen
or basin-unseen test block is rebuilt by dissolving its Level-12 basins and carries the regime of its block. A basin
that belongs to blocks of both regimes is reported and kept as 'both'. The partial block is left out.

Output  plot/data/conditions/rq2_basins.gpkg (layer basins: pfaf5, regime, geometry in EPSG:4326, simplified 0.01°)

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_regime_basins.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

SPLIT = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/splits/patches.csv')
L12 = Path('/media/ravi/Jupyter/flood_vit/Github/data/hydrobasins/hybas_lev12_global.shp')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'conditions' / 'rq2_basins.gpkg'


def main():
    p = pd.read_csv(SPLIT, usecols=['folder_name', 'cohort', 'ground_unit', 'basin_regime', 'basin_tokens'], dtype=str)
    p['basin_regime'] = p.basin_regime.replace({'partial': 'seen'})  # Ravi, 20 Oct 2026: no partial regime
    p = p[(p.cohort == 'test') & p.basin_regime.isin(['seen', 'unseen'])].drop_duplicates('folder_name')
    rows = [(t, r) for toks, r in zip(p.basin_tokens, p.basin_regime) for t in str(toks).split('-')]
    b = pd.DataFrame(rows, columns=['pfaf5', 'regime']).drop_duplicates()
    reg = b.groupby('pfaf5').regime.agg(lambda s: s.iloc[0] if s.nunique() == 1 else 'both')
    print('Level-5 basins:', reg.value_counts().to_dict(), 'blocks:', p.groupby('basin_regime').ground_unit.nunique().to_dict())
    attr = pyogrio.read_dataframe(L12, columns=['PFAF_ID'], read_geometry=False)
    pre = attr.PFAF_ID.astype(np.int64).astype(str).str[:5]
    keep = np.where(pre.isin(reg.index))[0]
    print('Level-12 basins kept:', len(keep), 'of', len(attr))
    g = pyogrio.read_dataframe(L12, columns=['PFAF_ID'], fids=keep)
    g['pfaf5'] = g.PFAF_ID.astype(np.int64).astype(str).str[:5]
    g = g.dissolve('pfaf5').reset_index()[['pfaf5', 'geometry']]
    g['geometry'] = g.geometry.simplify(0.01)
    g['regime'] = g.pfaf5.map(reg)
    missing = set(reg.index) - set(g.pfaf5)
    print('codes without polygons:', sorted(missing))
    g.to_crs(4326).to_file(OUT, layer='basins', driver='GPKG')
    print('wrote', OUT, len(g))


if __name__ == '__main__':
    main()
