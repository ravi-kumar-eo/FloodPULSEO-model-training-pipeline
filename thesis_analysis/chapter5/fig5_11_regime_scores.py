"""RQ2 — Scores of U-Net-T0 and ViT-T0 on basin-seen and basin-unseen test ground (seed means, rq2_regime_fits.csv):
(a) IoU, (b) POD and (c) POFD, each pooled over the scored cells of the regime (Ravi, 19 Oct 2026: no area bias). The partial spatial block
is left out.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_11_regime_scores.py
"""
import matplotlib.pyplot as plt

import regime_common as rg
import rq1_common as rc
from rq1_common import ps

PANELS = (('iou', '(a) IoU', None), ('pod', '(b) POD', None), ('pofd', '(c) POFD', None))


def main():
    rc.style()
    m = rg.t0_means()
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.15))
    for ax, (col, title, ref) in zip(axes, PANELS):
        rg.family_bars(ax, {(f, r): m.loc[(f, r), col] for f in rc.FAMILIES for r in rg.REG}, ref=ref)
        rc.panel_title(ax, title)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
    rg.legend(fig, y=0.075)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
