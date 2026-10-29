#!/bin/bash
# T3 / T7 / T15 fits (final_window, U-Net + ViT, seeds 1-3) on all scored sites, then rescore every batch with all
# fits and rebuild the combined table; results to S3. No instance stop (Ravi stops it himself).
N=/opt/dlami/nvme; B=$N/baselines; G=$N/flood-training/experiments/src/baselines/gff; L=$B/logs; E=$N/flood-training/experiments
PYP=$N/pipeline/env/bin/python; S3=s3://flood-patches-use1/experiments/runs_baselines/gff
say() { echo "[$(date '+%F %T')] $*"; }
FITS=$(for s in 1 2 3; do for f in unet vit; do for t in 3 7 15; do echo -n "final_window/seed$s/${f}_t$t "; done; done; done)
for b in ks:data_sites_ks gen:data_sites_gen more:data_sites_more; do
  tag=${b%%:*}; D=$N/pipeline/${b##*:}
  say "window fits on $tag"
  docker run --rm --gpus all --shm-size=16g -v $N:$N -w $E/src -e XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    -e TF_CPP_MIN_LOG_LEVEL=2 -e TF_FORCE_GPU_ALLOW_GROWTH=true flood-training:v5r \
    python baselines/gff/predict_ours.py --runs $E/runs_final --final $D/final --patches $D/patches \
      --out $B/site_preds/ours_$tag --fits $FITS > $L/ours_windows_$tag.log 2>&1 || say "window fits on $tag FAILED"
  say "rescore $tag (all fits)"
  $PYP $G/score_sites.py --batch $D --ours $B/site_preds/ours_$tag --gff-data $B/gff/data --gff-pred $B/gff_preds \
    --models utae metnet --out $B/results/$tag > $L/score_${tag}_all.log 2>&1 || say "score $tag FAILED"
done
say "combined table"; $PYP $G/combine_sites.py --results $B/results --sites $G/sites.csv --out $B/results/all > $L/combine.log 2>&1
aws s3 sync --only-show-errors $B/results $S3/results/
aws s3 sync --only-show-errors $B/site_preds $S3/site_preds/
aws s3 sync --only-show-errors $L $S3/logs/
say "WINDOWS DONE"
