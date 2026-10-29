#!/bin/bash
# Laptop: once the box has written BACKUP_DONE to S3, pull the whole baseline comparison into
# experiments/runs_baselines/gff/. Only on the home wifi (never on the phone hotspot); resumable, re-checks wifi per item.
S3=s3://flood-patches-use1/experiments/runs_baselines/gff
D=/media/ravi/Jupyter/flood_vit/experiments/runs_baselines/gff
export AWS_PAGER=""
home() { nmcli -t -f NAME connection show --active | grep -q "FRITZ!Box"; }
until aws s3 ls $S3/BACKUP_DONE > /dev/null 2>&1; do sleep 300; done
for p in results logs site_preds gate2_preds code gff_preds batches; do
  until home; do sleep 300; done
  echo "[$(date '+%F %T')] pulling $p"
  until aws s3 sync --only-show-errors $S3/$p $D/$p; do sleep 60; done
done
aws s3 cp --only-show-errors $S3/BACKUP_DONE $D/BACKUP_DONE
echo "[$(date '+%F %T')] ALL PULLED to $D ($(du -sh $D | cut -f1))"
