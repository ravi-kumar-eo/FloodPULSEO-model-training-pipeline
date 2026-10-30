#!/bin/bash
# Laptop: only the results (the numbers), once the box has finished and written RESULTS_DONE to S3. Home wifi only.
S3=s3://flood-patches-use1/experiments/runs_baselines/gff
D=/media/ravi/Jupyter/flood_vit/experiments/runs_baselines/gff
export AWS_PAGER=""
HOME_WIFI=${HOME_WIFI:-}             # name of the network to download on; empty = any network
until aws s3 ls $S3/RESULTS_DONE > /dev/null 2>&1; do sleep 300; done
until nmcli -t -f NAME connection show --active | grep -q "$HOME_WIFI"; do sleep 300; done
until aws s3 sync --only-show-errors $S3/results $D/results; do sleep 60; done
echo "[$(date '+%F %T')] results pulled to $D/results ($(du -sh $D/results | cut -f1))"
