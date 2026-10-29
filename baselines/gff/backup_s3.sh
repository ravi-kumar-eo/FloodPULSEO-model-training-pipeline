#!/bin/bash
# Box -> S3 only (Ravi, 16 Oct): results (every number: scores, GFF measure, per-cell tables, combined table), our
# predictions and GFF's prediction rasters. Nothing else is copied; the instance stays up. Marker RESULTS_DONE at the end.
N=/opt/dlami/nvme; B=$N/baselines; L=$B/logs; S3=s3://flood-patches-use1/experiments/runs_baselines/gff
say() { echo "[$(date '+%F %T')] $*"; }
until grep -q "GFF 5 MODELS DONE" $L/gff_more_models.log || ! pgrep -f "^/bin/bash $N/flood-training/experiments/src/baselines/gff/gff_more_models.sh" > /dev/null; do sleep 60; done
for d in results site_preds gff_preds; do say "sync $d"; aws s3 sync --only-show-errors $B/$d $S3/$d/; done
date -u > $B/RESULTS_DONE && aws s3 cp --only-show-errors $B/RESULTS_DONE $S3/RESULTS_DONE && say "S3 DONE"
