#!/usr/bin/env python3
"""One table over all scored sites (results/{ks,gen,more}/scores.csv), ranked into good / medium / worst.

Per site: label type, scored and flood cells, and IoU / F1 / AUPRC of GFF U-TAE, GFF MaxViT U-Net and our ten
families (U-Net and ViT x T0, T3, T7, T15, T30; mean over seeds 1-3 within a family, never across families).
Sites are ranked by the mean IoU of the GFF models and our T0 and T30 families (the original six groups) (how hard the site is for every model) and cut into
thirds: good / medium / worst. Writes <out>/all_sites.csv and prints it.

    python combine_sites.py --results <baselines/results> --sites sites.csv --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = {"gff_utae": "gff:utae", "gff_maxvit": "gff:metnet", "gff_lstmunet": "gff:lstmunet", "gff_3dunet": "gff:3dunet",
          "gff_logreg": "gff:logreg"}
OURS = {f"{fam}_t{t}": (f"final_t{t}", fam) for t in (0, 3, 7, 15, 30) for fam in ("unet", "vit")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--sites", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    sc = pd.concat([pd.read_csv(p).assign(batch=p.parent.name) for p in sorted(a.results.glob("*/scores.csv"))],
                   ignore_index=True)
    sc = sc[(sc.site != "ALL") & (sc.cells > 0)]          # a site moved between batches leaves an empty row
    rows = []
    for site, g in sc.groupby("site"):
        if g.cells.max() == 0:
            continue
        r = dict(site=site, batch=g.batch.iloc[0], cells=int(g.cells.max()), flood_cells=int(g.flood_cells.max()))
        for k, m in GROUPS.items():
            x = g[g.model == m]
            if x.empty:
                continue
            x = x.iloc[0]
            r[f"{k}_IoU"], r[f"{k}_F1"], r[f"{k}_AUPRC"] = x.IoU, x.F1, x.get("AUPRC", np.nan)
        for k, (stage, fam) in OURS.items():
            x = g[g.model.str.match(rf"ours:{stage}__s\d__{fam}$")]
            if x.empty:
                continue
            r[f"{k}_IoU"], r[f"{k}_F1"], r[f"{k}_AUPRC"] = x.IoU.mean(), x.F1.mean(), x.AUPRC.mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    s = pd.read_csv(a.sites)[["roi", "label", "place_approx", "flood_date", "phase"]].rename(columns={"roi": "site"})
    t = s.merge(t, on="site", how="right")
    iou = ["gff_utae_IoU", "gff_maxvit_IoU", "unet_t0_IoU", "vit_t0_IoU", "unet_t30_IoU", "vit_t30_IoU"]   # tiers unchanged
    t["mean_IoU_all"] = t[iou].mean(axis=1)
    t["best_ours_IoU"] = t[[f"{k}_IoU" for k in OURS if f"{k}_IoU" in t.columns]].max(axis=1)
    t["best_gff_IoU"] = t[[f"{k}_IoU" for k in GROUPS if f"{k}_IoU" in t.columns]].max(axis=1)
    t = t.sort_values("mean_IoU_all", ascending=False).reset_index(drop=True)
    t["tier"] = pd.qcut(t.mean_IoU_all.rank(method="first", ascending=False), 3, labels=["good", "medium", "worst"])
    a.out.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out / "all_sites.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    cols = ["tier", "site", "label", "flood_cells"] + [f"{k}_IoU" for k in GROUPS if f"{k}_IoU" in t.columns] + \
           [f"{k}_IoU" for k in OURS if f"{k}_IoU" in t.columns] + ["mean_IoU_all"]
    v = t[cols].copy(); v["site"] = v.site.str[:22]; v["label"] = v.label.str.split().str[0]
    print(v.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
