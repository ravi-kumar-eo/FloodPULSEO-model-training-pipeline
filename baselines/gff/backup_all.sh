#!/bin/bash
# Box -> S3: everything of the baseline comparison that lives only on the instance SSD, after gff_more_models.sh.
# Skipped: GFF's Zenodo data (public, re-downloadable) and the obsolete pre-release-grid patches (_old_grid).
N=/opt/dlami/nvme; B=$N/baselines; L=$B/logs; S3=s3://flood-patches-use1/experiments/runs_baselines/gff
say() { echo "[$(date '+%F %T')] $*"; }
until grep -q "GFF 5 MODELS DONE" $L/gff_more_models.log || ! pgrep -f "^/bin/bash $N/flood-training/experiments/src/baselines/gff/gff_more_models.sh" > /dev/null; do sleep 60; done
for d in results site_preds gff_preds gate2_preds; do say "sync $d"; aws s3 sync --only-show-errors $B/$d $S3/$d/; done
for b in data_gate2 data_sites_ks data_sites_gen data_sites_more; do
  say "sync batch $b"; aws s3 sync --only-show-errors $N/pipeline/$b $S3/batches/$b/ --exclude "_old_grid/*" --exclude "GEE_exports_31day/*"
done
aws s3 sync --only-show-errors $N/flood-training/experiments/src/baselines $S3/code/baselines/ --exclude "__pycache__/*"
aws s3 sync --only-show-errors $N/pipeline/FloodPULSEO_l1c $S3/code/FloodPULSEO_l1c/ --exclude "__pycache__/*"
aws s3 sync --only-show-errors $N/pipeline/logs $S3/logs/pipeline/
aws s3 sync --only-show-errors $L $S3/logs/
date -u > $B/BACKUP_DONE && aws s3 cp --only-show-errors $B/BACKUP_DONE $S3/BACKUP_DONE && say "BACKUP DONE"
