#!/usr/bin/env python3
"""Write metadata/splits_v2/ground_units_events.csv — the GROUND UNIT and AOI of every event.

WHY THIS EXISTS (Ravi, 2026-09-04)
----------------------------------
1,565 "events" do not describe 1,565 pieces of ground. Copernicus EMS re-maps the same area
on later dates (PRODUCT, then MONIT01, MONIT02, ...), and separate AOIs of one activation
often overlap at their edges. Two levels of identity matter and they are NOT the same:

  AOI          `activation_AOIid` — one mapped area. 845 of them. Its SIBLINGS are the same
               area re-mapped later. Measured: EMSR848_AOI04_Hue MONIT01/02/03 are
               BIT-IDENTICAL in every band the model sees (NDVI, NDBI, elevation, flow, UDA,
               HAND, clay, sand) and differ only in the mask, which recedes 0.085 -> 0.069
               -> 0.057 as the water falls. Across the dataset 749 events / 324,478 patches
               (57.3%) have such a twin. Training on those shows one input several
               contradictory targets, so the model can do no better than learn the mean
               flood extent for that terrain.
  GROUND UNIT  the transitive closure of footprint OVERLAP. 395 of them. Distinct AOIs that
               touch land in one unit (up to 39 AOIs in a single unit), so a unit is the
               coarsest thing that can be called independent ground.

Both are needed and for different jobs. The GROUND UNIT is what a split must keep atomic and
must balance in COUNT, because overlapping events are never independent samples — measured
on the first v1 fit, counting each unit once REVERSES the basin-familiarity result (pooled
2.16x for `seen`, per event 1.17x, per ground unit 0.67x, i.e. `unseen` ahead). The AOI is
what the EXP 1 row selection uses, because the sibling problem is an AOI-level problem.

WHY IT IS PRECOMPUTED
---------------------
The unit needs a geometric overlap test across 37 different CRS. Every patch box is
reprojected into ONE equal-area frame (EPSG:4087) — comparing coordinates within their own
CRS silently treats every cross-CRS pair as disjoint, the same bug that hid 33 real overlaps
in the v0 split. That costs geopandas + pyproj and tens of seconds, which has no business
running inside a 100-seed search, so it is written once and read by `build_split_v2.py`.

    python src/build_ground_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))   # src/ on the path
from flood.paths import METADATA as MD
SRC = MD / 'split_old' / 'splits_v0'   # any split's patches.csv — geometry is a property of
OUT = MD / 'splits'                    # the EVENT, not of the split. v0 is the raw table.
COMMON = 'EPSG:4087'            # World Equidistant Cylindrical: one metric, equal-area frame


def ground_units(patches: pd.DataFrame) -> pd.Series:
    """folder_name -> representative event of its overlap component."""
    geoms: dict[str, list] = {}
    for crs, g in patches.groupby('crs'):
        gd = gpd.GeoSeries([box(a, b, c, d) for a, b, c, d in
                            zip(g.bounds_minx, g.bounds_miny, g.bounds_maxx, g.bounds_maxy)],
                           crs=crs).to_crs(COMMON)
        for fn, geom in zip(g.folder_name.values, gd.values):
            geoms.setdefault(fn, []).append(geom)
    ev = sorted(geoms)
    # the event's true footprint is the UNION of its patch boxes, not its bounding box:
    # footprint_group in v0/v1 used bounding boxes and over-merges (one group swallowed five
    # separate Sri Lankan AOIs), so it is close to this but not the same thing.
    gdf = gpd.GeoDataFrame({'folder_name': ev},
                           geometry=[unary_union(geoms[e]) for e in ev], crs=COMMON)
    parent = {e: e for e in ev}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    hits = gpd.sjoin(gdf, gdf, how='inner', predicate='intersects')
    for a, b in zip(hits.folder_name_left, hits.folder_name_right):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return pd.Series({e: find(e) for e in ev}, name='raw_unit')


def main() -> int:
    p = pd.read_csv(SRC / 'patches.csv', low_memory=False,
                    usecols=['folder_name', 'patch_number', 'crs', 'bounds_minx',
                             'bounds_miny', 'bounds_maxx', 'bounds_maxy'])
    f = pd.read_csv(SRC / 'flood80_flags.csv')
    ev = p.groupby('folder_name').agg(n_patches=('patch_number', 'size'))
    ev = ev.join(f.groupby('folder_name').agg(flood80=('n_flood80', 'sum'),
                                              pw80=('n_pw80', 'sum')))
    ev['flood_fraction'] = ev.flood80 / (ev.n_patches * 1024)
    ev['activation'] = [i.split('_')[0] for i in ev.index]
    # AOI = first two name tokens. Verified: 845 keys, and NOT ONE of them spans a ground
    # unit, which is the check that the key really names a place rather than a product.
    ev['aoi'] = ['_'.join(i.split('_')[:2]) for i in ev.index]

    raw = ev.index.map(ground_units(p))
    order = {u: k for k, u in enumerate(sorted(set(raw)))}
    ev['ground_unit'] = [f'GU{order[u]:04d}' for u in raw]
    if int(ev.groupby('aoi').ground_unit.nunique().gt(1).sum()):
        sys.exit('an AOI spans more than one ground unit — the AOI key is wrong')

    ev['aoi_siblings'] = ev.groupby('aoi').n_patches.transform('size')
    ev['unit_events'] = ev.groupby('ground_unit').n_patches.transform('size')
    ev['unit_aois'] = ev.groupby('ground_unit').aoi.transform('nunique')
    # EXP 1 selection: one event per AOI, the sibling with the highest flood FRACTION
    # (Ravi, 2026-09-04). Recorded, not applied — build_split_v2.py reads the flag.
    ev['exp1_keep'] = False
    ev.loc[ev.groupby('aoi').flood_fraction.idxmax(), 'exp1_keep'] = True
    # kept only so the choice of rule stays visible: max flood CELLS instead of FRACTION
    # picks a different event for 15 of the 845 AOIs (215,300 vs 222,532 patches).
    ev['alt_keep_max_flood_cells'] = False
    ev.loc[ev.groupby('aoi').flood80.idxmax(), 'alt_keep_max_flood_cells'] = True

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ['activation', 'aoi', 'ground_unit', 'n_patches', 'flood80', 'pw80',
            'flood_fraction', 'aoi_siblings', 'unit_events', 'unit_aois', 'exp1_keep',
            'alt_keep_max_flood_cells']
    ev.reset_index()[['folder_name'] + cols].to_csv(OUT / 'ground_units_events.csv',
                                                    index=False)
    k = ev[ev.exp1_keep]
    print(f'{len(ev)} events -> {ev.aoi.nunique()} AOIs -> {ev.ground_unit.nunique()} '
          f'ground units')
    print(f'  {int((ev.aoi_siblings > 1).sum())} events have an AOI sibling; '
          f'{int((ev.unit_events > 1).sum())} share a ground unit with another event')
    print(f'  exp1 subset: {len(k)} events, {int(k.n_patches.sum()):,} of '
          f'{int(ev.n_patches.sum()):,} patches ({100 * k.n_patches.sum() / ev.n_patches.sum():.0f}%)')
    print(f'written to {OUT / "ground_units_events.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
