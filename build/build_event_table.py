#!/usr/bin/env python3
"""Write cohort_events.csv — one row per flood event, with cohort, geography and covariates.

patches.csv has 566,669 rows and is what the loader reads. Almost every question you
actually ask — which cohort, where on earth, how big, how wet, which basins — is an EVENT
question, and answering it from patches.csv means a groupby over half a million rows every
time. events.csv is 1,565 rows and answers those directly.

Includes the bbox in BOTH the event's native CRS and lat/lon, because events carry 37
different CRS and any map, distance or area calculation needs a common frame.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--patches-csv', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.patches_csv)
    ev = df.groupby('folder_name').agg(
        cohort=('cohort', 'first'),
        basin_tokens=('basin_tokens', 'first'),
        footprint_group=('footprint_group', 'first'),
        continent=('continent', 'first'),
        climate=('climate', 'first'),
        sensor_resolution_m=('sensor_resolution_m', 'first'),
        crs=('crs', 'first'),
        n_patches=('patch_number', 'size'),
        flood_pixels=('flood_pixels', 'sum'),
        x_min=('bounds_minx', 'min'), y_min=('bounds_miny', 'min'),
        x_max=('bounds_maxx', 'max'), y_max=('bounds_maxy', 'max'),
    ).reset_index()

    ev['n_basins'] = ev.basin_tokens.str.count('-') + 1
    ev['flood_frac'] = (ev.flood_pixels / ev.n_patches).round(2)
    ev['emsr'] = ev.folder_name.str.extract(r'^(EMSR\d+)', expand=False)

    # --- reproject each CRS group to EPSG:4326 so lat/lon are comparable across events
    parts = []
    for crs, g in ev.groupby('crs'):
        gdf = gpd.GeoDataFrame(
            g.copy(),
            geometry=[box(a_, b_, c_, d_) for a_, b_, c_, d_
                      in zip(g.x_min, g.y_min, g.x_max, g.y_max)],
            crs=crs).to_crs(4326)
        b = gdf.geometry.bounds
        gdf['lon_min'], gdf['lat_min'] = b.minx.values, b.miny.values
        gdf['lon_max'], gdf['lat_max'] = b.maxx.values, b.maxy.values
        c = gdf.geometry.centroid
        gdf['lon'], gdf['lat'] = c.x.values, c.y.values
        parts.append(pd.DataFrame(gdf.drop(columns='geometry')))
    ev = pd.concat(parts, ignore_index=True)

    for c in ('lon', 'lat', 'lon_min', 'lat_min', 'lon_max', 'lat_max'):
        ev[c] = ev[c].round(5)

    cols = ['folder_name', 'emsr', 'cohort', 'lat', 'lon',
            'lat_min', 'lat_max', 'lon_min', 'lon_max',
            'continent', 'climate', 'sensor_resolution_m',
            'n_patches', 'flood_pixels', 'flood_frac',
            'basin_tokens', 'n_basins', 'footprint_group',
            'crs', 'x_min', 'y_min', 'x_max', 'y_max']
    ev = ev[cols].sort_values(['cohort', 'folder_name'])
    ev.to_csv(a.out, index=False)

    print(f"wrote {a.out}  —  {len(ev):,} events × {len(cols)} columns")
    print(ev.groupby('cohort').agg(events=('folder_name', 'size'),
                                   patches=('n_patches', 'sum')).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
