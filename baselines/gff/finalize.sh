#!/bin/bash
# Runs on the AWS box after master_chain.sh: back everything up to S3, mark FINALIZED, and NOTHING else:
# the instance is never stopped by a script (Ravi, 2026-10-15). Gives up waiting at the DEADLINE and backs up what exists.
N=/opt/dlami/nvme; B=$N/baselines; L=$B/logs; S3=s3://flood-patches-use1/experiments/runs_baselines/gff
DEADLINE=$(date -d "2026-10-16 05:00 UTC" +%s)
say() { echo "[$(date '+%F %T')] $*"; }
until grep -q "MASTER CHAIN DONE" $L/master.log 2>/dev/null || [ $(date +%s) -ge $DEADLINE ]; do sleep 120; done
grep -q "MASTER CHAIN DONE" $L/master.log || say "DEADLINE reached before the chain finished; backing up what exists"
say "backup to $S3"
aws s3 sync --only-show-errors $B/results    $S3/results/
aws s3 sync --only-show-errors $B/site_preds $S3/site_preds/
aws s3 sync --only-show-errors $B/gff_preds  $S3/gff_preds/
aws s3 sync --only-show-errors $B/gate2_preds $S3/gate2_preds/
for b in data_gate2 data_sites_ks data_sites_gen; do
  [ -d $N/pipeline/$b ] || continue
  aws s3 sync --only-show-errors $N/pipeline/$b $S3/batches/$b/ --exclude "GEE_exports/*" --exclude "GEE_exports_31day/*" --exclude "_old_grid/*"
done
aws s3 sync --only-show-errors $N/flood-training/experiments/src/baselines $S3/code/baselines/ --exclude "__pycache__/*"
aws s3 sync --only-show-errors $N/pipeline/logs $S3/logs/pipeline/
aws s3 sync --only-show-errors $L $S3/logs/
date -u > $B/FINALIZED && aws s3 cp --only-show-errors $B/FINALIZED $S3/FINALIZED && say "FINALIZED marker written"
# (no instance stop and no credential removal here: Ravi stops instances himself, 2026-10-15)
