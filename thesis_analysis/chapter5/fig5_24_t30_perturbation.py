"""RQ3 — Sensitivity to perturbed dynamic hydroclimatic variables: change of (a) the predicted flood cells, (b) the
probability of detection, (c) the probability of false detection and (d) the micro IoU of the T30 networks when their
inputs are perturbed on the test subset, relative to the unperturbed pass of the same run. Appendix Figure A.7: (a) share of the added or
removed predicted cells that are flood in the reference (seeds pooled; precipitation changes only, since the swap
barely changes the predicted area), with a black tick at the precision of all predicted flood cells of
the unperturbed pass and a dotted line at the flooded share of the scored cells; (b) mean absolute change of the
flood probability over the scored cells (perturb.csv, ch5_rq3_perturb_data.py).

Groups: no precipitation (local and upstream precipitation zero on all 30 days), half and double precipitation (x 0.5
and x 2), swapped inputs (precipitation, soil moisture and validity flag of a random test patch from another CEMS
activation; the catchment area stays the patch's own). Bars: U-Net and ViT, mean of seeds 1-3 (Ravi, 14 Oct 2026:
seed means in the figure, every seed in Appendix Table A.6), value at the bar end, whisker = 95% cluster bootstrap
interval. Each fit keeps its validation-selected threshold.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_24_t30_perturbation.py
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

D = Path(os.environ.get('PERTURB_CSV', Path(__file__).resolve().parents[1] / 'data' / 'rq3' / 'perturb.csv'))
GROUPS = (('p0', 'Zero'), ('p05', 'Half'), ('p2', 'Double'), ('swap', 'Swapped'))     # defined in the caption
PANELS = (('pred_change_pct', '(a) Predicted flood cells, change (%)', '{:+.1f}'),
          ('pod_change', '(b) Probability of detection, change', '{:+.3f}'),
          ('pofd_change', '(c) Probability of false detection, change', '{:+.4f}'),
          ('iou_change', '(d) Micro IoU, change', '{:+.3f}'),
          ('changed_flood_share', '(e) Flood share of the net change', '{:.2f}'),
          ('mean_abs_prob_change', '(f) Mean absolute probability change', '{:.4f}'))
FLOODED_SHARE = 2769452 / (2769452 + 46499674)        # scored test cells (cells_test.npz labels 1 and 0)


def draw(d, panels, appendix=False):
    ps.apply_clear()
    fig, axes = plt.subplots(1 if appendix else 2, 2,
                             figsize=(ps.TEXT_WIDTH_IN, 3.1 if appendix else 5.0))
    for ax, (col, title, fmt) in zip(axes.ravel(), panels):
        lo_all, hi_all = 0.0, 0.0
        for g, (key, _) in enumerate(GROUPS):
            for f, dx in (('unet', -0.19), ('vit', 0.19)):
                r = d.loc[(f, key)]
                v = r[col]
                if np.isnan(v) or (col == 'changed_flood_share' and key == 'swap'):   # swapped inputs barely change
                    continue                                                           # the predicted area (panel a)
                lo, hi = (r[f'{col}_lo'], r[f'{col}_hi']) if f'{col}_lo' in r.index else (v, v)
                if col == 'changed_flood_share':
                    ax.plot([g + dx - 0.17, g + dx + 0.17], [r.precision_none] * 2, color='black', lw=1.4, zorder=5)
                    hi_all = max(hi_all, r.precision_none)
                ax.bar(g + dx, v, 0.34, color=ps.alpha_on_white(rc.COLOR[f], 0.8), edgecolor=rc.COLOR[f], lw=0.6,
                       zorder=2)
                ax.plot([g + dx] * 2, [lo, hi], color='black', lw=0.7, zorder=3)
                end = hi if v >= 0 else lo
                txt = fmt.format(v)
                txt = (txt[1:] if float(txt) == 0 else txt).replace('-', '\u2212')   # '-0.000' -> '0.000'; true minus
                ax.annotate(txt, (g + dx, end), xytext=(0, 1.5 if v >= 0 else -1.5),
                            textcoords='offset points', ha='center', va='bottom' if v >= 0 else 'top',
                            fontsize=ps.FONT_SIZE - 3, zorder=4)
                lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
        pad = 0.22 * (hi_all - lo_all)
        ax.set_ylim(lo_all - pad if lo_all < 0 else 0, hi_all + pad)
        if col == 'changed_flood_share':
            ax.axhline(FLOODED_SHARE, color='black', lw=0.6, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(0, color='black', lw=0.6, zorder=1)
        ax.set_xticks(range(len(GROUPS)), [n for _, n in GROUPS])
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        rc.ygrid(ax)
        rc.panel_title(ax, title)
    h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], 0.8), edgecolor=rc.COLOR[f], lw=0.6, label=rc.LABEL[f])
         for f in rc.FAMILIES]
    if appendix:
        h += [Line2D([], [], color='black', lw=1.4, label='Unperturbed precision (a)'),
              Line2D([], [], color='black', lw=0.6, ls=(0, (1, 1.5)), label='Labelled flooded share (a)')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=2, frameon=False, fontsize=ps.FONT_SIZE - 1.5,
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0.17 if appendix else 0.055, 1, 1), h_pad=1.2, w_pad=1.2)
    output = Path(__file__).with_name('figA_7_t30_perturbation_details.py') if appendix else Path(__file__)
    ps.save(fig, output, subdir=rc.section_dir(output))


def main():
    d = pd.read_csv(D)
    d = d[d.seed.astype(str) == 'mean'].set_index(['family', 'perturbation'])
    draw(d, PANELS[:4])
    details = tuple((col, title.replace('(e)', '(a)').replace('(f)', '(b)'), fmt)
                    for col, title, fmt in PANELS[4:])
    draw(d, details, appendix=True)
    for f in rc.FAMILIES:
        print(rc.LABEL[f], ' | '.join(f'{key} ' + ' '.join(fmt.format(d.loc[(f, key), col]) for col, _, fmt in PANELS)
                                      for key, _ in GROUPS))


if __name__ == '__main__':
    main()
