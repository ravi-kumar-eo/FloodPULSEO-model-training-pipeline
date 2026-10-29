#!/bin/bash
# Unattended chain on the AWS box (15 Oct 2026). GPU jobs run one after another; data for the generated
# sites (build_phase.sh 2) is built in parallel by its own process. Progress: logs/master.log.
N=/opt/dlami/nvme; B=$N/baselines; G=$N/flood-training/experiments/src/baselines/gff; L=$B/logs
PYG=$B/envs/gff_gpu/bin/python; PYP=$N/pipeline/env/bin/python; export TMPDIR=$N/tmp
GA="--code $B/gff/code/GFF --data $B/gff/data --weights $B/gff/weights --out $B/gff_preds --device cuda:0 --workers 8"
say() { echo "[$(date '+%F %T')] $*"; }
for m in utae metnet; do
  say "GFF $m inference, folds 0-4";  $PYG $G/gff_infer.py $GA --model $m > $L/gff_infer_$m.log 2>&1 || say "GFF $m inference FAILED"
  say "GFF $m reproduction check";   $PYP $G/gff_eval.py --data $B/gff/data --pred $B/gff_preds/$m > $L/gff_eval_$m.log 2>&1 || say "GFF $m eval FAILED"
  tail -2 $L/gff_eval_$m.log
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
aws s3 sync --only-show-errors $B/logs s3://flood-patches-use1/experiments/runs_baselines/gff/logs/
aws s3 sync --only-show-errors $N/pipeline/logs s3://flood-patches-use1/experiments/runs_baselines/gff/logs/pipeline/
say "MASTER CHAIN DONE"
