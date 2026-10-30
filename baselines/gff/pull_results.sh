#!/bin/bash
# Laptop side: wait for the AWS box's FINALIZED marker in S3, then pull the small outputs (scores, cell tables, our
# predictions, logs) into experiments/runs_baselines/gff/. Large items (patches, GFF rasters) stay in S3.
# Never downloads on the phone hotspot: only while connected to the home wifi.
S3=s3://flood-patches-use1/experiments/runs_baselines/gff
D=/media/ravi/Jupyter/flood_vit/experiments/runs_baselines/gff
export AWS_PAGER=""
HOME_WIFI=${HOME_WIFI:-}             # name of the network to download on; empty = any network
until aws s3 ls $S3/FINALIZED > /dev/null 2>&1; do sleep 600; done
until nmcli -t -f NAME connection show --active | grep -q "$HOME_WIFI"; do sleep 600; done
mkdir -p $D
for p in results logs site_preds; do aws s3 sync --only-show-errors $S3/$p $D/$p; done
aws s3 cp --only-show-errors $S3/FINALIZED $D/FINALIZED
echo "[$(date '+%F %T')] pulled to $D"
