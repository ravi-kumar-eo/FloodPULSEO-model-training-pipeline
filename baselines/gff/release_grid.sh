#!/bin/bash
# Re-cut a pipeline batch on the release grid: release_grid.sh <batch DATA_DIR>
#   1 grid CSV (make_grid_csv.py)  2 MERIT in UTM over the release footprint (MERIT_data_prep/1)
#   3 that MERIT.tif replaces Step 2's EPSG:4326 one  4 Step 4 again (old patches moved aside)
#   5 input_80m rebuilt exactly as the release's (MERIT_data_prep/2) + verify
set -e
B=$1; export FP_DATA_DIR=$B TMPDIR=/opt/dlami/nvme/tmp XDG_CACHE_HOME=/opt/dlami/nvme/cache
PY=/opt/dlami/nvme/pipeline/env/bin/python; FP=/opt/dlami/nvme/pipeline/FloodPULSEO
HERE=$(cd "$(dirname "$0")" && pwd)
echo "[$(date '+%F %T')] grid csv"; $PY $HERE/make_grid_csv.py --data $B --out $B/merit_event_grid.csv --release $FP/MERIT_data_prep/merit_event_grid.csv
echo "[$(date '+%F %T')] MERIT (UTM)"; $PY -u $FP/MERIT_data_prep/1_download_merit.py --events $B/merit_event_grid.csv --out $B/merit_raw --project flood-vit-project-8 \
  || echo "MERIT: some products failed (products without patches are harmless)"
for m in $B/merit_raw/*/MERIT.tif; do
  f=$(basename $(dirname $m)); g=$B/GEE_exports/${f%%_*}/$f
  [ -d $g ] || continue
  [ -f $g/MERIT_step2_4326.tif ] || mv $g/MERIT.tif $g/MERIT_step2_4326.tif
  cp $m $g/MERIT.tif
done
mkdir -p $B/_old_grid; [ -d $B/patches ] && mv $B/patches $B/_old_grid/patches_$(date +%H%M%S)
for c in released_patches_metadata.csv 4_patch_validation_issues.csv; do [ -f $B/metadata/$c ] && mv $B/metadata/$c $B/_old_grid/; done
echo "[$(date '+%F %T')] Step 4"; cd $FP; $PY -u scripts/4_make_patches.py < /dev/null; $PY -u scripts/4_make_patches.py --verify < /dev/null | tail -3
echo "[$(date '+%F %T')] input_80m (MERIT_data_prep/2)"
ONLY=$(ls -d $B/patches/*/* | xargs -n1 basename)   # products with patches (a product dropped in Step 3 has none)
$PY -u $FP/MERIT_data_prep/2_make_merit_patches.py --merit $B/merit_raw --patches $B/patches --only $ONLY
$PY -u $FP/MERIT_data_prep/2_make_merit_patches.py --merit $B/merit_raw --patches $B/patches --only $ONLY --verify | tail -3
echo "[$(date '+%F %T')] RELEASE GRID DONE"
