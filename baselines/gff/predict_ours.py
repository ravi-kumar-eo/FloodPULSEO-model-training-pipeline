#!/usr/bin/env python3
"""Run our final fits, unchanged, on any set of patches and save their flood probabilities.

For each fit: its config.yaml, its selected checkpoint (run_manifest selection, = best_selected) and
its stored threshold (inference_final/threshold.json). The model is built and restored exactly as in
scripts/evaluate.py (restore is checked: a dummy forward pass must change), inputs come through
FinalFloodPatches with the fit's own settings (transforms, soil file, weather mode, window).

Writes <out>/<stage>__s<seed>__<family>/: prob.f16.npy (N,32,32), labels.i8.npy (N,32,32, the batch's
own 80 m class labels), index.csv (row_index, folder_name, patch_number), threshold.json (copied).

    python predict_ours.py --runs <runs_final> --final <batch final dir> --patches <patches dir> --out <dir>
                           [--fits final_t0/seed1/unet ...]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np                                                       # noqa: E402
import tensorflow as tf                                                  # noqa: E402
import yaml                                                              # noqa: E402

from flood.data.dataset_final import FinalFloodPatches, load_transforms  # noqa: E402
from flood.training.core_final import build_model, weather_shape         # noqa: E402
from scripts.train import FAMILIES                                 # noqa: E402
from scripts.evaluate import predict                               # noqa: E402

ALL = [f"final_{t}/seed{s}/{m}" for t in ("t0", "t30") for s in (1, 2, 3) for m in ("unet", "vit")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--final", required=True, type=Path, help="batch metadata/final-style folder")
    ap.add_argument("--patches", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fits", nargs="*", default=ALL)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tr = load_transforms(a.final / "input_transforms.json")
    for fit in a.fits:
        run = a.runs / fit
        man = json.loads((run / "run_manifest.json").read_text())
        cfg = yaml.safe_load(open(run / "config.yaml"))
        tf.keras.mixed_precision.set_global_policy(cfg.get("mixed_precision", "float32"))
        T = int(cfg.get("temporal_days", 0))
        mode = str(cfg.get("weather_mode", "aoi"))
        if T and mode != "upstream":
            raise SystemExit(f"{fit}: weather mode {mode} is not supported here")
        model = build_model(FAMILIES[man["family"]][0], cfg)
        w = run / "checkpoints" / man.get("selection", {}).get("checkpoint", "best_loss.weights.h5")
        dummy = [tf.zeros((1, 256, 256, 2)), tf.zeros((1, 32, 32, 5)), tf.zeros((1, 16, 16, 2))]
        if T:
            dummy.append(tf.zeros((1,) + weather_shape(cfg)))
        before = np.asarray(tf.cast(model(tuple(dummy), training=False)[0], tf.float32))
        model.load_weights(str(w))
        if np.array_equal(before, np.asarray(tf.cast(model(tuple(dummy), training=False)[0], tf.float32))):
            raise SystemExit(f"checkpoint restored nothing: {w}")
        fp = FinalFloodPatches(a.final / "split" / "patches.csv", a.patches, a.final / "labels_80m_cls", "test",
                               allow_test=True, temporal_days=T, transforms=tr,
                               mixed_as=str(cfg.get("mixed_as", "pw")),
                               soil_file=str(cfg.get("soil_file", "input_160m.tif")),
                               weather_mode=mode, upstream_dir=(a.final / "upstream" / "weather") if T else None)
        P, _, Y = predict(model, fp, a.batch, a.workers)
        name = f"final_t{T}__s{cfg['seed']}__{man['family']}"      # window in the name (final_window holds T3/T7/T15)
        o = a.out / name
        o.mkdir(parents=True, exist_ok=True)
        np.save(o / "prob.f16.npy", P)
        np.save(o / "labels.i8.npy", Y)
        idx = fp.df[["folder_name", "patch_number"]].reset_index(drop=True)
        idx.index.name = "row_index"
        idx.to_csv(o / "index.csv")
        shutil.copy2(run / "inference_final" / "threshold.json", o / "threshold.json")
        print(f"{fit} -> {o}: {len(idx)} rows, checkpoint {w.name}", flush=True)
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
