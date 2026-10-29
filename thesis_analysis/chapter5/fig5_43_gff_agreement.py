"""Section 5.5.1 — Agreement of the predicted flood cells of the Global Flood Forecasting (GFF) models and the best
U-Net-T15 and ViT-T15 at the 21 expert-labelled sites.

One bar per pair of a model of this thesis and a GFF model, all scored cells of the 21 sites pooled. T15 is the
input-history length of the models of this thesis in Section 5.5 (Ravi, 27 Oct 2026).
(a) Labelled flood cells: detected by both, by the model of this thesis only, by the GFF model only, and by neither.
(b) False detections (labelled dry cells predicted as flood): raised by both, by the model of this thesis only and by
the GFF model only, as shares of the false detections of either. Ours: each seed at its validation-selected threshold,
shares averaged over seeds 1-3. Data: data/baseline/agreement.csv (ch5_baseline_cells_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_43_gff_agreement.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

import baseline_common as bc
import rq1_common as rc
from rq1_common import ps

PAIRS = [('unet_t15', 'gff_utae'), ('unet_t15', 'gff_maxvit'), ('vit_t15', 'gff_utae'), ('vit_t15', 'gff_maxvit')]
FLOOD = [('both', 'Detected by both', '#0257CE'), ('ours_only', 'Model of this thesis only', '#56B4E9'),
         ('gff_only', 'GFF model only', '#8C8C8C'), ('neither', 'Detected by neither', '#C1408F')]
FALSE = [('fa_both', 'Raised by both', '#EE8A00'), ('fa_ours_only', 'Model of this thesis only', '#F2B866'),
         ('fa_gff_only', 'GFF model only', '#8C8C8C')]


def main():
    rc.style()
    a = pd.read_csv(bc.D / 'agreement.csv').groupby(['group', 'gff']).mean(numeric_only=True)
    fl, fa = [], []
    for p in PAIRS:
        r = a.loc[p]
        fl.append([r[k] / r.flood for k, _, _ in FLOOD])
        tot = r.fa_both + r.fa_ours_only + r.fa_gff_only
        fa.append([r[k] / tot for k, _, _ in FALSE])
        print(p, ' '.join(f'{k} {v:.3f}' for (k, _, _), v in zip(FLOOD, fl[-1])), '|',
              ' '.join(f'{k} {v:.3f}' for (k, _, _), v in zip(FALSE, fa[-1])), f'| false detections {tot:,.0f}')
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 4.6))
    bc.stacked_shares(axes[0], fl, FLOOD, '(a) Labelled flood cells')
    bc.stacked_shares(axes[1], fa, FALSE, '(b) False detections')
    names = [f'{bc.LABEL[o]} and {bc.LABEL[g]}' for o, g in PAIRS]
    for ax, cats in zip(axes, (FLOOD, FALSE)):
        ax.set_yticks(range(len(PAIRS)), names, fontsize=ps.FONT_SIZE - 1.5)
        ax.tick_params(axis='y', length=0)
        ax.legend(handles=[Patch(facecolor=ps.alpha_on_white(c, 0.6), edgecolor=c, lw=0.6, label=lab)
                           for _, lab, c in cats],
                  loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=len(cats), frameon=False,
                  fontsize=ps.FONT_SIZE - 2, handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(h_pad=1.6)
    ps.save(fig, __file__, subdir='5_5')


if __name__ == '__main__':
    main()
