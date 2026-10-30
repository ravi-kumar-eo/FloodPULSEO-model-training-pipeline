"""Section 5.5.2 — IoU, POD and POFD of the LISFLOOD-FP event footprints, U-Net-T15 and ViT-T15 on the largest
test mapping product of each AOI.

Horizontal bars, one group per AOI (its largest mapping product; the simulation covers the whole mapped area), ordered by
labelled flooded area, one bar per model:
the footprint without flood protections (Scarpellini et al., 2026) and U-Net-T15 and ViT-T15 (mean of seeds 1-3; T15 is
the input-history length used for the models of this thesis in Section 5.5, Ravi 27 Oct 2026).
Data: data/lisflood_fp/products.csv (ch5_lisflood_fp_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_43d_lisflood_fp_metrics.py
"""
import pandas as pd

import baseline_common as bc
import rq1_common as rc
from rq1_common import ps

D = bc.D.parent / 'lisflood_fp'
GROUPS = ['lf', 'unet_t15', 'vit_t15']


def main():
    rc.style()
    p = pd.read_csv(D / 'products.csv')
    p = p[p.selected].sort_values('km2', ascending=False).reset_index(drop=True)
    w = pd.DataFrame(index=p.index)
    for m in ('IoU', 'POD', 'POFD'):
        w[f'lf_{m}'] = p[f'lf_noprot_{m.lower()}']
        for g in GROUPS[1:]:
            w[f'{g}_{m}'] = p[f'{g}_{m.lower()}']
    names = [f'{r.place}, {r.country}\n{r.km2:,.1f} km²' if r.km2 >= 0.1 else f'{r.place}, {r.country}\n{r.km2:.2f} km²'
             for r in p.itertuples()]
    fig = bc.bar_rows(w, GROUPS, names, xmax=(0.7, 1.0, 1.0), bar=0.22, gap=0.24, height=8.0)
    print(w.round(3).to_string())
    ps.save(fig, __file__, subdir='5_5')


if __name__ == '__main__':
    main()
