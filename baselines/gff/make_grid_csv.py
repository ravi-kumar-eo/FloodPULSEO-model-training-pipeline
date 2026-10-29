#!/usr/bin/env python3
"""Release-style patch grid for new products, in the format of MERIT_data_prep/merit_event_grid.csv.

The released patches are anchored on the MERIT export, which the original pipeline pulled in the
product's UTM zone at 90 m over the AOI's lat/lon bounding box. Reproduced on 400 released products
(origin and grid rows/cols 400/400; right and bottom edge 396/400, the other 4 one 90 m cell off):
  crs          UTM zone of the AOI centroid (EPSG:4326)
  U            the AOI's EPSG:4326 bounding box projected to that zone (corners only, no densifying)
  origin_minx  floor(U.minx / 90) * 90        origin_maxy  ceil(U.maxy / 90) * 90
  aoi_maxx     ceil(U.maxx / 90) * 90         aoi_miny     floor(U.miny / 90) * 90
  grid_cols    (aoi_maxx - origin_minx) // 2560, grid_rows likewise
The current Github Step 2 exports MERIT in EPSG:4326, so its Step 4 grid is not the release grid;
1_download_merit.py with this CSV restores the UTM export and therefore the release grid.

Across all 1,535 released products with a local AOI the rule gives the recorded grid for 1,522; the 13
others (bottom edge one cell off, scattered positions, probably AOI files edited after the export) are
why --release takes the RECORDED row for any product that is in the release.

    python make_grid_csv.py --data <batch DATA_DIR> --out <batch>/merit_event_grid.csv [--release merit_event_grid.csv]
"""
import argparse
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from rasterio.warp import transform_bounds

COLS = ["folder_name", "crs", "origin_minx", "origin_maxy", "grid_rows", "grid_cols", "aoi_maxx", "aoi_miny",
        "merit_export_crs", "export_minx", "export_maxy", "export_cols", "export_rows"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--release", type=Path, default=None, help="shipped merit_event_grid.csv")
    a = ap.parse_args()
    rel = pd.read_csv(a.release, dtype=str).set_index("folder_name") if a.release else None
    rows = []
    for shp in sorted((a.data / "activations" / "activations_reorganized").glob("*/*/aoi/aoi.shp")):
        folder = shp.parent.parent.name
        if rel is not None and folder in rel.index:
            rows.append([folder] + rel.loc[folder].fillna("").tolist()); continue
        g = gpd.read_file(shp).to_crs(4326)
        c = g.union_all().centroid
        crs = f"EPSG:{(32600 if c.y >= 0 else 32700) + int((c.x + 180) // 6) + 1}"
        u = transform_bounds("EPSG:4326", crs, *g.total_bounds)
        x0, y1 = math.floor(u[0] / 90) * 90, math.ceil(u[3] / 90) * 90
        x1, y0 = math.ceil(u[2] / 90) * 90, math.floor(u[1] / 90) * 90
        rows.append([folder, crs, float(x0), float(y1), int((y1 - y0) // 2560), int((x1 - x0) // 2560),
                     float(x1), float(y0), "", "", "", "", ""])
    pd.DataFrame(rows, columns=COLS).to_csv(a.out, index=False)
    print(f"{len(rows)} products -> {a.out}")


if __name__ == "__main__":
    main()
