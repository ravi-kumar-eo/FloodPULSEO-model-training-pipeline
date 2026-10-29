#!/bin/bash
# T30 weather for a pipeline batch, with the thesis's own builders: weather_batch.sh <batch DATA_DIR>
# Step 2 wrote Precipitation_/SoilMoisture_<mapping-30>_<mapping>.tif (31 daily bands, AOI + 0.6 deg). The builders
# read them as GEE_exports_31day and want each band described by its ISO date; today's Step 2 describes them
# 'Precipitation_YYYYMMDD'. Copies with ISO descriptions go to GEE_exports_31day (values untouched).
set -e
B=$1; PY=/opt/dlami/nvme/pipeline/env/bin/python; S=/opt/dlami/nvme/flood-training/experiments/src
M=/opt/dlami/nvme/pipeline/merit_ihu_30s; F=$B/final
rm -rf $B/GEE_exports_31day
$PY - $B <<'P'
import re, shutil, sys
from pathlib import Path
import rasterio
B = Path(sys.argv[1]); n = 0
for f in sorted((B / "GEE_exports").glob("*/*/*.tif")):
    if not re.match(r"(Precipitation|SoilMoisture)_\d{8}_\d{8}\.tif$", f.name):
        continue
    dst = B / "GEE_exports_31day" / f.parent.parent.name / f.parent.name / f.name
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, dst)
    with rasterio.open(dst, "r+") as r:
        for i, d in enumerate(r.descriptions, 1):
            m = re.search(r"(\d{4})(\d{2})(\d{2})$", d or "")
            if not m:
                raise SystemExit(f"{f}: band {i} description {d!r} has no date")
            r.set_band_description(i, f"{m[1]}-{m[2]}-{m[3]}")
    n += 1
print(f"{n} weather files with ISO band dates -> {B / 'GEE_exports_31day'}")
P
echo "[$(date '+%F %T')] aoi_weather";  $PY $S/build/build_aoi_weather_final.py --data $B --out $F/aoi_weather --workers 8
echo "[$(date '+%F %T')] catchments";   $PY $S/build/build_upstream_catchments.py --split $F/split/patches.csv --merit $M --out $F/upstream/catchments --workers 8
echo "[$(date '+%F %T')] upstream weather"; $PY $S/build/build_upstream_weather.py --final $F --merit $M --out $F/upstream/weather --workers 8
echo "[$(date '+%F %T')] WEATHER DONE"
