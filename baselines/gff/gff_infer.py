#!/usr/bin/env python3
"""Run a released GFF model, unchanged, over GFF test regions and keep its yes/no AND its water probability.

Same as GFF's infer.py (gff.evaluation.model_inference): the fold's config and checkpoint, the fold's
test partition, their FloodForecastDataset and model; per visit tile the argmax class is written into
a raster on the region's label grid. Additionally the softmax probability of class 1 (water) is kept.
Regions are processed one at a time, so only one region's rasters are in memory.

Output <out>/<model>/fold<k>/<floodmap stem>-pred.tif (uint8 class, 255 outside visit tiles) and
<floodmap stem>-prob.tif (uint8, water probability x 250, 255 outside visit tiles).

    python gff_infer.py --code <GFF repo> --data <gff data> --weights <dir> --model utae --folds 0 1 2 3 4
                        --out <dir> [--rois <meta json names>] [--device cuda:0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rois", nargs="*", default=None, help="only these meta json names")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, str(a.code))
    import gff.dataloaders
    import gff.models.creation
    import gff.util

    torch.set_grad_enabled(False)
    for k in a.folds:
        C = yaml.safe_load(open(a.weights / f"{a.model}_{k}.yml"))
        C.update(data_folder=str(a.data), device=a.device, num_workers=a.workers, batch_size=a.batch)
        model = gff.models.creation.create(C).to(a.device)
        ck = torch.load(a.weights / f"{a.model}_{k}.th", map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        _, _, test_fnames = gff.dataloaders.read_partitions(a.data, k)
        if a.rois is not None:
            test_fnames = [f for f in test_fnames if f in set(a.rois)]
        out = a.out / a.model / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"{a.model} fold {k}: {len(test_fnames)} regions", flush=True)
        for n, meta_fname in enumerate(sorted(test_fnames), 1):
            ds = gff.dataloaders.FloodForecastDataset(a.data, C, meta_fnames=[meta_fname])
            if len(ds) == 0:
                continue
            fmap = a.data / "rois" / ds.metas[0]["floodmap"]
            stem = fmap.stem
            if (out / f"{stem}-prob.tif").exists():
                continue
            with rasterio.open(fmap) as src:
                prof, tr, shape = src.profile.copy(), src.transform, src.shape
            cls = np.full(shape, 255, np.uint8)
            prob = np.full(shape, 255, np.uint8)
            dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                                             collate_fn=gff.dataloaders.sometimes_things_are_lists)
            t0 = time.time()
            for ex in dl:
                ex = gff.util.recursive_todevice(ex, a.device)
                logits = model(ex)
                p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                c = logits.argmax(dim=1).cpu().numpy()
                for i in range(len(c)):
                    (r0, _), (c0, _) = gff.util.shapely_bounds_to_rasterio_window(ex["geom"][i].bounds, tr)
                    r0, c0 = int(r0), int(c0)              # their writer's window ((ylo, yhi), (xlo, xhi))
                    h, wd = c[i].shape[-2:]
                    cls[r0:r0 + h, c0:c0 + wd] = c[i].reshape(h, wd)
                    prob[r0:r0 + h, c0:c0 + wd] = np.round(p[i].reshape(h, wd) * 250).astype(np.uint8)
            prof.update(dtype="uint8", count=1, nodata=255, compress="lzw", tiled=True, BIGTIFF="IF_SAFER")
            for arr, tag in ((cls, "pred"), (prob, "prob")):
                tmp = out / f"{stem}-{tag}.part.tif"
                with rasterio.open(tmp, "w", **prof) as dst:
                    dst.write(arr, 1)
                tmp.rename(out / f"{stem}-{tag}.tif")
            print(f"  [{n}/{len(test_fnames)}] {stem}: {len(ds)} tiles, {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache() if a.device.startswith("cuda") else None


if __name__ == "__main__":
    main()
