#!/bin/bash
# Phase 3 (47 sites, 15 Oct 2026): build inputs with a pipeline copy whose Sentinel-2 step falls back from surface
# reflectance to L1C (COPERNICUS/S2_HARMONIZED) where SR gives no composite; then, after chain v2 has finished
# (one GPU job at a time), GFF U-TAE + MaxViT U-Net and our 12 fits on these sites, scoring, S3. No instance stop.
N=/opt/dlami/nvme; B=$N/baselines; G=$N/flood-training/experiments/src/baselines/gff; L=$B/logs
PYG=$B/envs/gff_gpu/bin/python; PYP=$N/pipeline/env/bin/python; export TMPDIR=$N/tmp
S3=s3://flood-patches-use1/experiments/runs_baselines/gff
say() { echo "[$(date '+%F %T')] $*"; }
say "phase 3 data (S2: SR, then L1C)"
FP_ROOT=$N/pipeline/FloodPULSEO_l1c FP_S2_COLLECTIONS="COPERNICUS/S2_SR_HARMONIZED,COPERNICUS/S2_HARMONIZED" \
  $G/build_phase.sh 3 $N/pipeline/data_sites_more > $N/pipeline/logs/phase3_data.log 2>&1 || say "phase 3 data FAILED (see pipeline/logs/phase3_data.log)"
say "waiting for chain v2 (GPU)"
until grep -q "MASTER CHAIN DONE" $L/master.log; do sleep 120; done
ROIS=$($PYP -c "import pandas as pd; s=pd.read_csv('$G/sites.csv'); print(' '.join(r + '-meta.json' for r in s[s.phase==3].roi))")
GA="--code $B/gff/code/GFF --data $B/gff/data --weights $B/gff/weights --out $B/gff_preds --device cuda:0 --workers 8 --rois $ROIS"
for m in utae metnet; do
  say "GFF $m on phase 3 sites"; $PYG $G/gff_infer.py $GA --model $m > $L/gff_infer_${m}_phase3.log 2>&1 || say "GFF $m phase 3 FAILED"
done
if [ -f $N/pipeline/data_sites_more/final/upstream/weather/COMPLETE ]; then
  say "our 12 fits on phase 3 sites"; $B/run_ours.sh $N/pipeline/data_sites_more $B/site_preds/ours_more > $L/ours_more.log 2>&1
  say "score phase 3 sites"
  $PYP $G/score_sites.py --batch $N/pipeline/data_sites_more --ours $B/site_preds/ours_more --gff-data $B/gff/data \
    --gff-pred $B/gff_preds --models utae metnet --out $B/results/more > $L/score_more.log 2>&1 || say "score more FAILED"
else
  say "phase 3 weather not COMPLETE; skipped predictions"
fi
say "sync results to S3"
aws s3 sync --only-show-errors $B/results $S3/results/
aws s3 sync --only-show-errors $B/site_preds $S3/site_preds/
aws s3 sync --only-show-errors $L $S3/logs/
aws s3 sync --only-show-errors $N/pipeline/logs $S3/logs/pipeline/
date -u > $B/FINALIZED && aws s3 cp --only-show-errors $B/FINALIZED $S3/FINALIZED
say "CHAIN 3 DONE"
