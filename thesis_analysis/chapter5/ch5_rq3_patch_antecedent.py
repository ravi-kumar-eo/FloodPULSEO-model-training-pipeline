"""Antecedent conditions of every test patch, from the dynamic hydroclimatic variables that the T30 networks receive
(experiments/metadata/final/upstream/weather/<product>.npz, written by experiments/src/build/build_upstream_weather.py;
per patch and day from day -30 to day -1: local IMERG precipitation and SMAP soil moisture of the cell containing the
patch centre, and the precipitation over the upstream catchment of the patch).

Writes chapters/chapter5/plot/data/rq3/patch_antecedent.csv, one row per test patch (folder_name, patch_number):
  api_up_<k>, api_loc_<k>   antecedent precipitation index of the upstream and of the local precipitation (mm):
                            sum over the 30 days of P(day -t) * k**(t - 1), so day -1 has weight 1 and the weight
                            falls by the factor k per day back in time; k = 0.85, 0.90, 0.95
  p3_up, p3_loc             precipitation of the last 3 days, days -3 to -1 (mm)
  sm_before                 local soil moisture of day -4, before the last 3 days (m3 m-3)
  sm_last                   local soil moisture of day -1 (m3 m-3)
  area_km2                  upstream catchment area of the patch (km2)

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_rq3_patch_antecedent.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

W = Path('/media/ravi/Jupyter/flood_vit/experiments/metadata/final/upstream/weather')
HERE = Path(__file__).resolve().parents[1] / 'data'
OUT = HERE / 'rq3' / 'patch_antecedent.csv'
KS = (0.85, 0.90, 0.95)


def main():
    p = pd.read_csv(HERE / 'conditions' / 'patches_test.csv.gz', usecols=['folder_name', 'patch_number'])
    rows = []
    for f, g in p.groupby('folder_name'):
        z = np.load(W / f'{f}.npz')
        if z['local_precip'].shape[1] != 30:
            raise SystemExit(f'{f}: expected 30 days')
        idx = {int(n): i for i, n in enumerate(z['patch_number'])}
        lp, up, sm = z['local_precip'], z['upstream_precip'], z['local_sm']
        for n in g.patch_number:
            i = idx[int(n)]
            r = {'folder_name': f, 'patch_number': int(n)}
            for k in KS:
                w = k ** np.arange(30)[::-1]                  # the last column is day -1, weight 1
                r[f'api_up_{k:.2f}'] = float(np.nansum(up[i] * w))
                r[f'api_loc_{k:.2f}'] = float(np.nansum(lp[i] * w))
            r.update(p3_up=float(np.nansum(up[i, -3:])), p3_loc=float(np.nansum(lp[i, -3:])),
                     sm_before=float(sm[i, -4]), sm_last=float(sm[i, -1]), area_km2=float(z['upstream_area_km2'][i]))
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out) != len(p):
        raise SystemExit('patches without inputs')
    out.to_csv(OUT, index=False)
    print(f'{len(out)} patches written to {OUT}')


if __name__ == '__main__':
    main()
