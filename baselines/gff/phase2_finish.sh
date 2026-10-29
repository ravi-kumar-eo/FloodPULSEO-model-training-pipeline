#!/bin/bash
# Finish phase 2 (12 generated sites) after release_grid stopped on the 3 dropped products (no patches):
# verify input_80m, loader split + labels, weather, our 12 fits, scoring. chain4 syncs everything to S3 at its end.
set -e
N=/opt/dlami/nvme; B=$N/baselines; G=$N/flood-training/experiments/src/baselines/gff; D=$N/pipeline/data_sites_gen
PY=$N/pipeline/env/bin/python; FP=$N/pipeline/FloodPULSEO; F=$N/flood-training/experiments/metadata/final
say() { echo "[$(date '+%F %T')] $*"; }
ONLY=$(ls -d $D/patches/*/* | xargs -n1 basename)
say "verify input_80m ($(echo $ONLY | wc -w) products)"
$PY -u $FP/MERIT_data_prep/2_make_merit_patches.py --merit $D/merit_raw --patches $D/patches --only $ONLY --verify | tail -2
say "loader split + labels"; $PY $G/make_batch_final.py --data $D --ref-final $F --out $D/final
$PY $N/flood-training/experiments/src/build/build_labels_80m.py --patches $D/patches --out $D/final/labels_80m_cls --workers 8 | tail -2
say "weather"; $G/weather_batch.sh $D
echo "[$(date '+%F %T')] PHASE 2 DATA DONE"
say "our 12 fits on generated sites"; $B/run_ours.sh $D $B/site_preds/ours_gen > $B/logs/ours_gen.log 2>&1
say "score generated sites"
$PY $G/score_sites.py --batch $D --ours $B/site_preds/ours_gen --gff-data $B/gff/data --gff-pred $B/gff_preds \
  --models utae metnet --out $B/results/gen > $B/logs/score_gen.log 2>&1
tail -16 $B/logs/score_gen.log
say "PHASE 2 FINISHED"
