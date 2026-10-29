#!/bin/bash
# Unattended chain, v2 (15 Oct 2026, Ravi: GFF ONLY on our 23 test sites, not on GFF's whole test splits).
# GFF U-TAE + MaxViT U-Net on the 23 sites of sites.csv (each in the fold that held it out) -> score the 8
# hand-labelled sites -> wait for the generated sites' data -> our 12 fits on them -> score -> S3. Then finalize.sh.
N=/opt/dlami/nvme; B=$N/baselines; G=$N/flood-training/experiments/src/baselines/gff; L=$B/logs
PYG=$B/envs/gff_gpu/bin/python; PYP=$N/pipeline/env/bin/python; export TMPDIR=$N/tmp
ROIS=$($PYP -c "import pandas as pd; print(' '.join(r + '-meta.json' for r in pd.read_csv('$G/sites.csv').roi))")
GA="--code $B/gff/code/GFF --data $B/gff/data --weights $B/gff/weights --out $B/gff_preds --device cuda:0 --workers 8 --rois $ROIS"
say() { echo "[$(date '+%F %T')] $*"; }
say "chain v2: GFF on the $(echo $ROIS | wc -w) sites only"
for m in utae metnet; do
  say "GFF $m on the sites"; $PYG $G/gff_infer.py $GA --model $m > $L/gff_infer_${m}_sites.log 2>&1 || say "GFF $m FAILED"
done
say "score hand-labelled sites"
$PYP $G/score_sites.py --batch $N/pipeline/data_sites_ks --ours $B/site_preds/ours_ks --gff-data $B/gff/data \
  --gff-pred $B/gff_preds --models utae metnet --out $B/results/ks > $L/score_ks.log 2>&1 || say "score ks FAILED"
say "waiting for generated-site data"
until grep -qE "PHASE 2 DATA DONE" $N/pipeline/logs/phase2_data.log || ! pgrep -f "build_phase.sh 2" > /dev/null; do sleep 120; done
if grep -q "PHASE 2 DATA DONE" $N/pipeline/logs/phase2_data.log; then
  say "our 12 fits on generated sites"; $B/run_ours.sh $N/pipeline/data_sites_gen $B/site_preds/ours_gen > $L/ours_gen.log 2>&1
  say "score generated sites"
  $PYP $G/score_sites.py --batch $N/pipeline/data_sites_gen --ours $B/site_preds/ours_gen --gff-data $B/gff/data \
    --gff-pred $B/gff_preds --models utae metnet --out $B/results/gen > $L/score_gen.log 2>&1 || say "score gen FAILED"
else
  say "generated-site data did NOT finish (see pipeline/logs/phase2_data.log)"
fi
say "backup to S3"
aws s3 sync --only-show-errors $B/results s3://flood-patches-use1/experiments/runs_baselines/gff/results/
aws s3 sync --only-show-errors $L s3://flood-patches-use1/experiments/runs_baselines/gff/logs/
say "MASTER CHAIN DONE"
