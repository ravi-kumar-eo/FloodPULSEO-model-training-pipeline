#!/bin/bash
# Build our inputs for one phase of sites.csv, end to end: build_phase.sh <phase> <batch DATA_DIR>
# activation folders -> Steps 2-4 -> release grid -> loader split + 80 m labels -> T30 weather
set -e
P=$1; B=$2; PY=/opt/dlami/nvme/pipeline/env/bin/python
S=/opt/dlami/nvme/flood-training/experiments/src; G=$S/baselines/gff; F=/opt/dlami/nvme/flood-training/experiments/metadata/final
echo "[$(date '+%F %T')] activation folders (phase $P)"
$PY $G/sites_to_activations.py --sites $G/sites.csv --gff /opt/dlami/nvme/baselines/gff/data --data $B --phase $P
echo "[$(date '+%F %T')] pipeline steps 2-4"; /opt/dlami/nvme/pipeline/run_steps.sh $B
echo "[$(date '+%F %T')] release grid";     $G/release_grid.sh $B
echo "[$(date '+%F %T')] loader split + labels"
$PY $G/make_batch_final.py --data $B --ref-final $F --out $B/final
$PY $S/build/build_labels_80m.py --patches $B/patches --out $B/final/labels_80m_cls --workers 8 | tail -2
echo "[$(date '+%F %T')] weather"; $G/weather_batch.sh $B
echo "[$(date '+%F %T')] PHASE $P DATA DONE"
