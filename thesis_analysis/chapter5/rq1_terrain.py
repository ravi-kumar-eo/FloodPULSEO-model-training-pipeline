"""Mean elevation and mean HAND of every test mapping product, from the released 80 m terrain recipe
(plot/data/rq1/product_terrain.csv).

The local input_80m.tif files are the pre-fix build (HAND resampled cubic, values below 0). The released terrain is
rebuilt here per product with the release code itself: Github/MERIT_data_prep/2_make_merit_patches.py,
event_grid_from_patches() on the product's input_10m patches and build_event_stack() on
pipeline/data/GEE_exports/<EMSR>/<product>/MERIT.tif (elevation cubic, HAND nearest), then sliced per test patch as
process_event() does. Cells counted: scored cells of the test patches (80 m label >= 0, i.e. not permanent water or
mixed) with a valid terrain value and a D8 flow direction (sea fill has none and is left out, as for Chapter 3).
Per product: mean elevation and mean HAND over these cells, and mean HAND over its flood cells.

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python chapters/chapter5/plot/scripts/rq1_terrain.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import ch5_data as cd

GH = Path('/media/ravi/Jupyter/flood_vit/Github/MERIT_data_prep/2_make_merit_patches.py')
PT = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/patches/patches_tif')
GEE = Path('/media/ravi/Jupyter/flood_vit/pipeline/data/GEE_exports')
OUT = Path(__file__).resolve().parents[1] / 'data' / 'rq1' / 'product_terrain.csv'


def load_release_code():
    spec = importlib.util.spec_from_file_location('merit_patches', GH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    mp = load_release_code()
    ts = cd.test_set()
    ix = ts.index
    Y = np.load(cd.run_dir('unet', 't0', 1) / 'inference_final/preds/test__labels.i8.npy')
    rows, failed = [], []
    for k, (prod, g) in enumerate(ix.groupby('folder_name')):
        ev = PT / prod.split('_')[0] / prod
        merit = GEE / prod.split('_')[0] / prod / 'MERIT.tif'
        try:
            files = sorted(p for p in ev.iterdir() if mp.PATCH_RE.search(p.name))
            crs, transform, width, height, (ox, oy) = mp.event_grid_from_patches(files)
            stack = mp.build_event_stack(merit, crs, transform, width, height)
        except Exception as e:                                 # report, never guess a value
            failed.append((prod, repr(e)))
            continue
        se = sh = n = shf = nf = 0.0
        for r, pn in zip(g.index, g.patch_number):
            with rasterio.open(ev / f'patch_{int(pn):04d}_input_10m.tif') as src:
                b = src.bounds
            col = int(round((b.left - ox) / mp.RES_80M))
            row = int(round((oy - b.top) / mp.RES_80M))
            t = stack[:, row:row + mp.PX_80M, col:col + mp.PX_80M]
            if t.shape[1:] != (mp.PX_80M, mp.PX_80M):
                failed.append((prod, f'patch {pn}: window {t.shape}'))
                continue
            valid = (t[0] != mp.NODATA) & (t[4] != mp.NODATA) & (t[1] != mp.NODATA) & (Y[r] >= 0)
            flood = valid & (Y[r] == 1)
            se += float(t[0][valid].sum()); sh += float(t[4][valid].sum()); n += int(valid.sum())
            shf += float(t[4][flood].sum()); nf += int(flood.sum())
        rows.append({'folder_name': prod, 'cells': int(n), 'elevation_mean_m': se / n if n else np.nan,
                     'hand_mean_m': sh / n if n else np.nan, 'hand_flood_mean_m': shf / nf if nf else np.nan})
        if k % 50 == 0:
            print(k, prod, flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'{len(rows)} products written, {len(failed)} problems')
    for p, e in failed[:20]:
        print('  ', p, e)


if __name__ == '__main__':
    main()
