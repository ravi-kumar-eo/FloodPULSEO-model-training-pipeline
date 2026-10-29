#!/bin/bash
# Everything left (15 Oct 2026, ~19:50 UTC), one chain. CPU: finish phase 2 inputs (release grid onward) and build
# phase 3 inputs (L1C fallback, pre-2017 allowed), in parallel. GPU, one job at a time: GFF on the phase-3 sites,
# then our 12 fits + scoring for phase 2 and phase 3 as their inputs become ready. Results to S3. No instance stop.
N=/opt/dlami/nvme; B=$N/baselines; G=$N/flood-training/experiments/src/baselines/gff; L=$B/logs; PL=$N/pipeline/logs
PYG=$B/envs/gff_gpu/bin/python; PYP=$N/pipeline/env/bin/python; export TMPDIR=$N/tmp
S3=s3://flood-patches-use1/experiments/runs_baselines/gff; F=$N/flood-training/experiments/metadata/final
say() { echo "[$(date '+%F %T')] $*"; }
( set -e; D=$N/pipeline/data_sites_gen
  echo "[$(date '+%F %T')] phase 2 remainder: release grid"; $G/release_grid.sh $D
  $PYP $G/make_batch_final.py --data $D --ref-final $F --out $D/final
  $PYP $N/flood-training/experiments/src/build/build_labels_80m.py --patches $D/patches --out $D/final/labels_80m_cls --workers 8 | tail -2
  $G/weather_batch.sh $D
  echo "[$(date '+%F %T')] PHASE 2 DATA DONE" ) > $PL/phase2_rest.log 2>&1 &
( FP_ROOT=$N/pipeline/FloodPULSEO_l1c FP_S2_COLLECTIONS="COPERNICUS/S2_SR_HARMONIZED,COPERNICUS/S2_HARMONIZED" \
  $G/build_phase.sh 3 $N/pipeline/data_sites_more ) > $PL/phase3_data.log 2>&1 &
ROIS=$($PYP -c "import pandas as pd; s=pd.read_csv('$G/sites.csv'); print(' '.join(r + '-meta.json' for r in s[s.phase==3].roi))")
GA="--code $B/gff/code/GFF --data $B/gff/data --weights $B/gff/weights --out $B/gff_preds --device cuda:0 --workers 8 --rois $ROIS"
for m in utae metnet; do
  say "GFF $m on phase 3 sites"; $PYG $G/gff_infer.py $GA --model $m > $L/gff_infer_${m}_phase3.log 2>&1 || say "GFF $m phase 3 FAILED"
done
run_batch() {  # <tag> <batch dir> <data log>
  say "waiting for $1 inputs"
  until grep -q "DATA DONE" $3 || ! pgrep -f "$4" > /dev/null; do sleep 120; done
  if [ -f $2/final/upstream/weather/COMPLETE ]; then
    say "our 12 fits on $1"; $B/run_ours.sh $2 $B/site_preds/ours_$1 > $L/ours_$1.log 2>&1
    say "score $1"; $PYP $G/score_sites.py --batch $2 --ours $B/site_preds/ours_$1 --gff-data $B/gff/data \
      --gff-pred $B/gff_preds --models utae metnet --out $B/results/$1 > $L/score_$1.log 2>&1 || say "score $1 FAILED"
  else
    say "$1 inputs did not finish (see $3)"
  fi
}
run_batch gen  $N/pipeline/data_sites_gen  $PL/phase2_rest.log "phase 2 remainder|release_grid.sh $N/pipeline/data_sites_gen"
run_batch more $N/pipeline/data_sites_more $PL/phase3_data.log "build_phase.sh 3"
say "sync results to S3"
aws s3 sync --only-show-errors $B/results $S3/results/
aws s3 sync --only-show-errors $B/site_preds $S3/site_preds/
aws s3 sync --only-show-errors $L $S3/logs/
aws s3 sync --only-show-errors $PL $S3/logs/pipeline/
date -u > $B/FINALIZED && aws s3 cp --only-show-errors $B/FINALIZED $S3/FINALIZED
say "CHAIN 4 DONE"
