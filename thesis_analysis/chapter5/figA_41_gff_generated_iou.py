"""Appendix A.7 — IoU of the GFF models and of the networks of this thesis at each site with a generated GFF label.

30 sites whose labels GFF generated with its automatic mapping models (the two sites that touch the validation subset
are left out), in date order, named by country, month of the flood and rank in labelled flooded area within a flood
event as in Table A.9 (names under panel b only); (a) T0, (b) T30, each with GFF U-TAE and MaxViT U-Net. Ours = mean of seeds 1-3. A dagger marks floods that are also test events of the thesis.
Data: data/baseline/sites.csv (ch5_baseline_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_41_gff_generated_iou.py
"""
import matplotlib.pyplot as plt

import baseline_common as bc
import rq1_common as rc
from rq1_common import ps


def main():
    rc.style()
    d = bc.read('generated')
    order = bc.site_order(d)
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 6.0), sharex=True, sharey=True)
    for ax, w, t in zip(axes, ('t0', 't30'), ('(a) Without the hydroclimatic input history (T0)', '(b) With the 30-day hydroclimatic input history (T30)')):
        groups = ['gff_utae', 'gff_maxvit', f'unet_{w}', f'vit_{w}']
        bc.site_panel(ax, d[d.group.isin(groups)], groups, order, t, event_labels=False)
        print(w, d[d.group.isin(groups)].groupby('group').IoU.median().round(3).to_dict())
    axes[0].set_ylim(0, 1)
    names = [n + ('†' if te else '') for n, te in zip(bc.site_names(order), order.our_test_event)]
    axes[1].set_xticks(range(len(order)), names, fontsize=ps.FONT_SIZE - 2, rotation=90)
    axes[0].tick_params(axis='x', labelbottom=False)
    fig.legend(handles=bc.handles(['gff_utae', 'gff_maxvit', 'unet_t30', 'vit_t30']),
               labels=['GFF U-TAE', 'GFF MaxViT U-Net', 'U-Net', 'ViT'],
               loc='upper center', bbox_to_anchor=(0.5, 0.025), ncol=4, frameon=False, fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.0)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
