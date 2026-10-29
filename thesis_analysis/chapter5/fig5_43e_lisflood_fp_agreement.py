"""Section 5.5.2 — Agreement of the predicted flood cells of the LISFLOOD-FP event footprints and U-Net-T15 and ViT-T15
on the 21 selected test mapping products, by outcome group.

Groups of mapping products by the IoU of the footprint against U-Net-T15 and ViT-T15: above both, between the two,
below both, and dry (the footprint has no flood in the AOI). One bar per group and model. (a) Labelled flood cells:
detected by both the footprint and the model, by the model only, by the footprint only, and by neither. (b) False
detections (labelled dry cells predicted as flood): raised by both, by the model only and by the footprint only, as
shares of the false detections of either. Shares are computed per mapping product and seed, averaged over seeds 1-3 and
then over the mapping products of a group, so no large flood dominates its group. T15 is the input-history length of the
models of this thesis in Section 5.5 (Ravi, 27 Oct 2026).
Data: data/lisflood_fp/agreement.csv (ch5_lisflood_fp_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_43e_lisflood_fp_agreement.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import baseline_common as bc
import rq1_common as rc
from rq1_common import ps

D = bc.D.parent / 'lisflood_fp'
GROUPS = [('above', 'Footprint above both'), ('between', 'Footprint between the two'), ('below', 'Footprint below both'),
          ('dry', 'Footprint dry')]
MODELS = ('unet_t15', 'vit_t15')
FLOOD = [('both', 'Detected by both', '#0257CE'), ('ours_only', 'Model of this thesis only', '#56B4E9'),
         ('lf_only', 'LISFLOOD-FP only', '#8C8C8C'), ('neither', 'Detected by neither', '#C1408F')]
FALSE = [('fa_both', 'Raised by both', '#EE8A00'), ('fa_ours_only', 'Model of this thesis only', '#F2B866'),
         ('fa_lf_only', 'LISFLOOD-FP only', '#8C8C8C')]


def main():
    rc.style()
    a = pd.read_csv(D / 'agreement.csv')
    for k, _, _ in FLOOD:
        a[k + '_s'] = a[k] / a.flood
    fa = a.fa_both + a.fa_ours_only + a.fa_lf_only
    for k, _, _ in FALSE:
        a[k + '_s'] = np.where(fa > 0, a[k] / fa.where(fa > 0, 1), np.nan)
    per = a.groupby(['group', 'config', 'product']).mean(numeric_only=True)       # seed mean per mapping product
    g = per.groupby(['group', 'config']).mean()                                    # mean over the mapping products
    n = a.groupby('group')['product'].nunique()
    fl, fd, names = [], [], []
    for grp, lab in GROUPS:
        for m in MODELS:
            r = g.loc[(grp, m)]
            fl.append([r[k + '_s'] for k, _, _ in FLOOD])
            fd.append([0 if np.isnan(r[k + '_s']) else r[k + '_s'] for k, _, _ in FALSE])
            names.append(f'{lab} ({n[grp]}), {bc.LABEL[m]}')
            print(f'{grp:8s} {m:9s} ' + ' '.join(f'{k} {v:.3f}' for (k, _, _), v in zip(FLOOD, fl[-1])) + ' | '
                  + ' '.join(f'{k} {v:.3f}' for (k, _, _), v in zip(FALSE, fd[-1])))
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 6.0))
    bc.stacked_shares(axes[0], fl, FLOOD, '(a) Labelled flood cells')
    bc.stacked_shares(axes[1], fd, FALSE, '(b) False detections')
    for ax, cats in zip(axes, (FLOOD, FALSE)):
        ax.set_yticks(range(len(names)), names, fontsize=ps.FONT_SIZE - 2)
        ax.tick_params(axis='y', length=0)
        for y in (1.5, 3.5, 5.5):
            ax.axhline(y, color='black', lw=0.5, zorder=4)
        ax.legend(handles=[Patch(facecolor=ps.alpha_on_white(c, 0.6), edgecolor=c, lw=0.6, label=lab)
                           for _, lab, c in cats],
                  loc='upper center', bbox_to_anchor=(0.38, -0.1), ncol=len(cats), frameon=False,
                  fontsize=ps.FONT_SIZE - 2, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(h_pad=2.2)
    ps.save(fig, __file__, subdir='5_5')


if __name__ == '__main__':
    main()
