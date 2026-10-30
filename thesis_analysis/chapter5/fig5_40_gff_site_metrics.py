"""Section 5.5.1 — IoU, POD and POFD of GFF U-TAE, GFF MaxViT U-Net, U-Net-T15 and ViT-T15 at each of the 21
expert-labelled sites of the comparison with the Global Flood Forecasting (GFF) models.

Horizontal bars, one group per site (Ravi, 29 Oct 2026: per site, so the figure replaces the former IoU table of the
section), sites ordered by flood date and, within a flood event, by labelled flooded area; one bar per model. Sites are
named as in Appendix Table A.9, with the labelled flooded area under the name. U-Net-T15 and ViT-T15 are the mean of
seeds 1-3 at a site; T15 is the input-history length of Section 5.5 (Ravi, 27 Oct 2026). A dagger marks the sites
whose flood is also a test flood of this thesis.
Data: data/baseline/sites.csv (ch5_baseline_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_40_gff_site_metrics.py
"""
import pandas as pd

import baseline_common as bc
import rq1_common as rc
from rq1_common import ps

GROUPS = ['gff_utae', 'gff_maxvit', 'unet_t15', 'vit_t15']


def main():
    rc.style()
    d = bc.read('hand')
    d = d[d.group.isin(GROUPS)]
    order = bc.site_order(d)
    w = pd.DataFrame(index=order.site)
    for m in ('IoU', 'POD', 'POFD'):
        piv = d.pivot_table(index='site', columns='group', values=m)
        for g in GROUPS:
            w[f'{g}_{m}'] = piv.loc[order.site, g].to_numpy()
    names = [f"{n}{'†' if r.our_test_event else ''}\n{r.flood_cells * 0.0064:,.1f} km²"
             for n, r in zip(bc.site_names(order), order.itertuples())]
    w = w.reset_index(drop=True)
    fig = bc.bar_rows(w, GROUPS, names, xmax=(0.6, 1.0, 0.5), bar=0.19, gap=0.26, height=8.0)
    print(w.round(3).to_string())
    ps.save(fig, __file__, subdir='5_5')


if __name__ == '__main__':
    main()
