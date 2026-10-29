#!/bin/bash
# The other three released GFF models (LSTM U-Net, 3DConv U-Net, logistic regression) on the scored sites, after
# windows_chain.sh (one GPU job at a time); then rescore every batch with all 5 GFF models and all our fits. No stop.
N=/opt/dlami/nvme; B=$N/baselines; G=$N/flood-training/experiments/src/baselines/gff; L=$B/logs
PYG=$B/envs/gff_gpu/bin/python; PYP=$N/pipeline/env/bin/python; export TMPDIR=$N/tmp
S3=s3://flood-patches-use1/experiments/runs_baselines/gff
say() { echo "[$(date '+%F %T')] $*"; }
until grep -q "WINDOWS DONE" $L/windows_chain.log || ! pgrep -f "^/bin/bash $G/windows_chain.sh" > /dev/null; do sleep 60; done
cd $B/gff/weights && for k in 0 1 2 3 4; do for e in th yml; do [ -s lstmunet_$k.$e ] || curl -sS -L --retry 5 -o lstmunet_$k.$e https://github.com/Multihuntr/GFF/releases/download/v1/lstmunet_$k.$e; done; done
ROIS=$($PYP -c "import pandas as pd; print(' '.join(r + '-meta.json' for r in pd.read_csv('$G/sites.csv').roi))")
for m in lstmunet 3dunet logreg; do
  say "GFF $m on the sites"
  $PYG $G/gff_infer.py --code $B/gff/code/GFF --data $B/gff/data --weights $B/gff/weights --out $B/gff_preds --device cuda:0 \
    --workers 8 --rois $ROIS --model $m > $L/gff_infer_${m}_sites.log 2>&1 || say "GFF $m FAILED"
done
for b in ks:data_sites_ks gen:data_sites_gen more:data_sites_more; do
  tag=${b%%:*}; say "rescore $tag (5 GFF models, all our fits)"
  $PYP $G/score_sites.py --batch $N/pipeline/${b##*:} --ours $B/site_preds/ours_$tag --gff-data $B/gff/data --gff-pred $B/gff_preds \
    --models utae metnet lstmunet 3dunet logreg --out $B/results/$tag > $L/score_${tag}_5gff.log 2>&1 || say "score $tag FAILED"
done
say "combined table"; $PYP $G/combine_sites.py --results $B/results --sites $G/sites.csv --out $B/results/all > $L/combine.log 2>&1
aws s3 sync --only-show-errors $B/results $S3/results/
aws s3 sync --only-show-errors $L $S3/logs/
say "GFF 5 MODELS DONE"
