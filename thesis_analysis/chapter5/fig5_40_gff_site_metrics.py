"""Section 5.5.1 — IoU, POD and POFD of GFF U-TAE, GFF MaxViT U-Net, U-Net-T15 and ViT-T15 for each of the 10 flood
events of the 21 expert-labelled sites of the comparison with the Global Flood Forecasting (GFF) models.

Horizontal bars, one group per flood event in date order (as Table 5.6), one bar per model; the value of an event is
the mean over its sites, and U-Net-T15 and ViT-T15 are the mean of seeds 1-3 at a site. T15 is the input-history length
used for the models of this thesis in Section 5.5 (Ravi, 27 Oct 2026: one fixed length, horizontal bars that can be
read; it replaces the per-site dot plot). A dagger marks the floods that are also test floods of this thesis.
Data: data/baseline/sites.csv (ch5_baseline_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_40_gff_site_metrics.py
"""
import pandas as pd

import baseline_common as bc
import rq1_common as rc
from rq1_common import ps

GROUPS = ['gff_utae', 'gff_maxvit', 'unet_t15', 'vit_t15']
MONTH = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


def main():
    rc.style()
    d = bc.read('hand')
    d = d[d.group.isin(GROUPS)]
    ev = d.groupby(['event', 'group'])[['IoU', 'POD', 'POFD']].mean().unstack('group')
    meta = d.drop_duplicates('site').groupby('event').agg(date=('flood_date', 'first'), country=('country', 'first'),
                                                         sites=('site', 'nunique'), test=('our_test_event', 'max'))
    meta = meta.sort_values('date')
    w = pd.DataFrame(index=meta.index)
    for g in GROUPS:
        for m in ('IoU', 'POD', 'POFD'):
            w[f'{g}_{m}'] = ev[(m, g)]
    names = []
    for r in meta.itertuples():
        y, mo, _ = r.date.split('-')
        names.append(f"{r.country}, {MONTH[int(mo) - 1]} {y}{'†' if r.test else ''}\n"
                     f"{r.sites} site{'s' if r.sites > 1 else ''}")
    fig = bc.bar_rows(w, GROUPS, names, xmax=(0.6, 1.0, 0.3))
    print(w.round(3).to_string())
    ps.save(fig, __file__, subdir='5_5')


if __name__ == '__main__':
    main()
