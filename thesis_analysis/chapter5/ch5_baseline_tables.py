"""Section 5.6 and Appendix A.6 — tables of the comparison with the GFF models.

Writes plot/data/baseline/:
  summary_hand.csv / summary_generated.csv  one row per model: median IoU, POD, POFD and area bias over the sites;
      sites (and events) on which the model has a higher IoU than GFF U-TAE and than GFF MaxViT U-Net. The IoU of an
      event is the mean over its sites.
  persite_hand.csv  IoU of every model at every hand-labelled site, with event, country and date.
Data: data/baseline/sites.csv (ch5_baseline_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/ch5_baseline_tables.py
"""
import pandas as pd

import baseline_common as bc

ORDER = ['gff_utae', 'gff_maxvit'] + [f'{a}_{w}' for w in ('t0', 't3', 't7', 't15', 't30') for a in ('unet', 'vit')]


def summary(d):
    piv = d.pivot_table(index='site', columns='group', values='IoU')
    ev = d.drop_duplicates('site').set_index('site').event
    epiv = piv.groupby(ev).mean()
    rows = []
    for g in ORDER:
        s = d[d.group == g]
        r = {'model': bc.LABEL[g], 'IoU': s.IoU.median(), 'POD': s.POD.median(), 'POFD': s.POFD.median(),
             'area_bias': s.area_bias.median()}
        for ref in ('gff_utae', 'gff_maxvit'):
            r[f'sites_above_{ref}'] = '' if g.startswith('gff') else int((piv[g] > piv[ref]).sum())
            r[f'events_above_{ref}'] = '' if g.startswith('gff') else int((epiv[g] > epiv[ref]).sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    out.attrs['n'] = (piv.shape[0], epiv.shape[0])
    return out


def main():
    for lt in ('hand', 'generated'):
        d = bc.read(lt)
        s = summary(d)
        s.to_csv(bc.D / f'summary_{lt}.csv', index=False)
        print(lt, 'sites, events =', s.attrs['n'])
        print(s.round(3).to_string(index=False))
    d = bc.read('hand')
    order = bc.site_order(d)
    piv = d.pivot_table(index='site', columns='group', values='IoU')[ORDER].loc[order.site]
    meta = order.set_index('site')[['event', 'country', 'flood_date', 'our_test_event', 'flood_cells']]
    pd.concat([meta, piv], axis=1).to_csv(bc.D / 'persite_hand.csv')
    print(pd.concat([meta, piv.round(3)], axis=1).to_string())


if __name__ == '__main__':
    main()
