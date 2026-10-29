#!/usr/bin/env python3
"""Check 2 at the model level: do our fits give the same answer on rebuilt inputs as on released ones?

Per fit (folders written by predict_ours.py under --ref and --new):
  stored     max |p| difference between --ref and the fit's stored test predictions (same rows):
             0 means predict_ours.py reproduces scripts/evaluate.py exactly
  agreement  correlation of the probabilities and share of scored cells with the same yes/no at the
             fit's stored threshold, rebuilt vs released
  IoU        pooled flood-only IoU (label 1 vs 0; PW and mixed, label -1, excluded) on each set,
             both scored against the RELEASED labels so only the inputs differ

    python compare_preds.py --ref <dir> --new <dir> --runs <runs_final>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def iou(p, y, thr):
    m = y >= 0
    yes, t = p[m] >= thr, y[m] == 1
    inter = np.sum(yes & t); union = np.sum(yes | t)
    return inter / union if union else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for d in sorted(p for p in a.ref.iterdir() if p.is_dir()):
        n = a.new / d.name
        if not n.exists():
            continue
        stage, seed, fam = d.name.split("__")
        thr = json.loads((d / "threshold.json").read_text())["threshold"]
        ri = pd.read_csv(d / "index.csv"); ni = pd.read_csv(n / "index.csv")
        key = ["folder_name", "patch_number"]
        j = ri.merge(ni, on=key, suffixes=("_r", "_n"))
        Pr = np.load(d / "prob.f16.npy").astype(np.float32)[j.row_index_r]
        Pn = np.load(n / "prob.f16.npy").astype(np.float32)[j.row_index_n]
        Yr = np.load(d / "labels.i8.npy")[j.row_index_r]
        # stored predictions of the fit's own test evaluation
        st = a.runs / stage / seed.replace("s", "seed") / fam / "inference_final" / "preds"
        si = pd.read_csv(st / "test__index.csv")
        sj = j[key].merge(si[["row_index"] + key], on=key)
        Ps = np.load(st / "test__prob.f16.npy", mmap_mode="r")[sj.row_index.to_numpy()].astype(np.float32)
        m = Yr >= 0
        rows.append(dict(fit=d.name, rows=len(j), stored_max_abs_diff=float(np.abs(Ps - Pr[: len(Ps)]).max()) if len(Ps) == len(j) else np.nan,
                         r_prob=float(np.corrcoef(Pr[m], Pn[m])[0, 1]),
                         same_yes_no=float(np.mean((Pr[m] >= thr) == (Pn[m] >= thr))),
                         iou_released=iou(Pr, Yr, thr), iou_rebuilt=iou(Pn, Yr, thr), thr=thr))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
