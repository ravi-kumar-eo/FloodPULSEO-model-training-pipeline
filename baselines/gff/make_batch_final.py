#!/usr/bin/env python3
"""Give a pipeline batch the `metadata/final`-style folder our loader reads.

From <batch>/metadata/released_patches_metadata.csv (Step 4) write <out>/split/patches.csv with the
columns FinalFloodPatches and the evaluator expect. Rows follow the final split's rule
(aoi_fraction > 0); every row is cohort 'test' (inference only) and exp1_keep True. The transforms
and data version are copied unchanged from the reference metadata/final, so inputs are scaled exactly
as in training. Labels (labels_80m_cls) and weather (upstream/weather) are built afterwards with
build_labels_80m.py and the three weather builders.

    python make_batch_final.py --data <batch> --ref-final <metadata/final> --out <batch>/final
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--ref-final", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    p = pd.read_csv(a.data / "metadata" / "released_patches_metadata.csv")
    p = p[p.aoi_fraction > 0].copy()
    p["cohort"] = "test"
    p["exp1_keep"] = True
    p["aoi"] = p.folder_name.str.split("_").str[:2].str.join("_")
    p["ground_unit"] = p["aoi"]
    p["basin_regime"] = "external"
    cols = ["folder_name", "patch_number", "crs", "bounds_minx", "bounds_miny", "bounds_maxx", "bounds_maxy",
            "flood_pixels", "continent", "climate", "cohort", "aoi", "ground_unit", "basin_regime", "exp1_keep",
            "aoi_fraction"]
    (a.out / "split").mkdir(parents=True, exist_ok=True)
    p[cols].sort_values(["folder_name", "patch_number"]).to_csv(a.out / "split" / "patches.csv", index=False)
    for f in ("input_transforms.json", "data_version.json"):
        shutil.copy2(a.ref_final / f, a.out / f)
    print(f"{p.folder_name.nunique()} products, {len(p)} patches -> {a.out}/split/patches.csv")


if __name__ == "__main__":
    main()
