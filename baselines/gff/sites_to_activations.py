#!/usr/bin/env python3
"""Turn GFF test sites into FloodPULSEO activation folders, so our own pipeline builds our inputs there.

For every site of `sites.csv` (phase given by --phase), from the GFF release (`rois/`):
  aoi/aoi.shp              union of the site's GFF visit tiles = the cells GFF scores (EPSG:4326)
  flood_extent/event.shp   the GFF label's flood class (2) inside those tiles, polygonised (EPSG:4326)
The folder name ends with the GFF post-event date (the acquisition date of the image the label was
drawn from), which our pipeline reads as the mapping date:
  activations/activations_reorganized/GFF<site>/GFF<site>_AOI01_DEL_PRODUCT_<YYYYMMDD>/
The flood polygons only let Step 3 run; scoring regrids the GFF label raster itself.
Writes <data>/gff_site_map.csv (site, folder_name, gff label file, visit tiles file, fold).

    python sites_to_activations.py --sites sites.csv --gff <gff/data> --data <batch data dir> --phase 1
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import shape


def site_folder(roi: str, post: str) -> tuple[str, str]:
    p = roi.split("-")
    code = f"GFFKS{p[0]}x{p[1]}" if len(p) == 5 else f"GFFGEN{p[0]}"
    return code, f"{code}_AOI01_DEL_PRODUCT_{post.replace('-', '')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True, type=Path)
    ap.add_argument("--gff", required=True, type=Path, help="GFF data folder (contains rois/)")
    ap.add_argument("--data", required=True, type=Path, help="pipeline batch DATA_DIR")
    ap.add_argument("--phase", type=int, default=1)
    a = ap.parse_args()

    sites = pd.read_csv(a.sites)
    sites = sites[sites.phase == a.phase]
    rows = []
    for s in sites.itertuples():
        mpath = a.gff / "rois" / f"{s.roi}-meta.json"
        meta = json.load(open(mpath))
        post = meta["post_date"][:10]
        code, folder = site_folder(s.roi, post)
        out = a.data / "activations" / "activations_reorganized" / code / folder
        tiles = gpd.read_file(a.gff / "rois" / meta["visit_tiles"])
        aoi = tiles.geometry.union_all()
        with rasterio.open(a.gff / "rois" / meta["floodmap"]) as r:
            if tiles.crs != r.crs:
                raise ValueError(f"{s.roi}: tiles {tiles.crs} vs label {r.crs}")
            win = from_bounds(*tiles.total_bounds, r.transform).round_offsets().round_lengths()
            lab = r.read(1, window=win)
            tr = r.window_transform(win)
            inside = features.geometry_mask([aoi], lab.shape, tr, invert=True)
            flood = (lab == 2) & inside
            polys = [shape(g) for g, v in features.shapes(flood.astype(np.uint8), mask=flood, transform=tr) if v == 1]
            crs = r.crs
        (out / "aoi").mkdir(parents=True, exist_ok=True)
        (out / "flood_extent").mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({"site": [s.roi]}, geometry=[aoi], crs=crs).to_crs(4326).to_file(out / "aoi" / "aoi.shp")
        gpd.GeoDataFrame({"site": [s.roi] * len(polys)}, geometry=polys, crs=crs).to_crs(4326) \
            .to_file(out / "flood_extent" / "event.shp")
        rows.append(dict(site=s.roi, folder_name=folder, label_file=meta["floodmap"],
                         visit_tiles=meta["visit_tiles"], gff_fold=s.gff_fold, post_date=post,
                         n_tiles=len(tiles), flood_polygons=len(polys)))
        print(f"{s.roi} -> {folder}: {len(tiles)} tiles, {len(polys)} flood polygons", flush=True)
    m = pd.DataFrame(rows)
    a.data.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.data / "gff_site_map.csv", index=False)


if __name__ == "__main__":
    main()
