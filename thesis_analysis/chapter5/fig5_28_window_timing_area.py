"""RQ4 — Change of the predicted flood cells from T0 for every input-history length within the classes of the timing
of the antecedent precipitation (timing.csv, ch5_rq4_timing_data.py; upstream precipitation; the classes of
Figure 5.35).

Bars: change of the predicted flood cells (TP + FP over the scored cells of the class) from T0 of the same family and
seed, in per cent, mean of seeds 1-3 (Ravi, 14-15 Oct 2026: seed means only; show where the networks predict more or
less), value at the bar end, whisker = 95% cluster bootstrap interval. T3 to T30 in the family colour, lighter for
shorter histories.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_28_window_timing_area.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

import rq1_common as rc
import rq4_common as r4
from rq1_common import ps

CLASSES = (('t3', 'Days −3 to −1'), ('t7', 'Days −7 to −4'), ('t15', 'Days −15 to −8'), ('t30', 'Days −30 to −16'))
LENGTHS = ('t3', 't7', 't15', 't30')
STRENGTH = {'t3': 0.35, 't7': 0.55, 't15': 0.78, 't30': 1.0}
WIDTH = 0.19


def main():
    rc.style()
    d = pd.read_csv(r4.D / 'timing.csv')
    d = d[d.source == 'upstream'].set_index(['class', 'family', 'window'])
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 5.0))
    for ax, (k, f) in zip(axes, enumerate(rc.FAMILIES)):
        lo_all, hi_all = 0.0, 0.0
        for g, (c, _) in enumerate(CLASSES):
            for j, w in enumerate(LENGTHS):
                r = d.loc[(c, f, w)]
                v, lo, hi = r.d_pred_t0_pct, r.d_pred_t0_pct_lo, r.d_pred_t0_pct_hi
                x = g + (j - 1.5) * (WIDTH + 0.01)
                ax.bar(x, v, WIDTH, color=ps.alpha_on_white(rc.COLOR[f], STRENGTH[w]), edgecolor=rc.COLOR[f],
                       lw=0.6, zorder=2)
                ax.plot([x, x], [lo, hi], color='black', lw=0.7, zorder=3)
                end = hi if v >= 0 else lo
                txt = '0' if round(v) == 0 else f'{v:+.0f}'.replace('-', '\u2212')
                ax.annotate(txt, (x, end), xytext=(0, 1.5 if v >= 0 else -1.5),
                            textcoords='offset points', ha='center', va='bottom' if v >= 0 else 'top',
                            fontsize=ps.FONT_SIZE - 3, zorder=4)
                lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
        pad = 0.15 * (hi_all - lo_all)
        ax.set_ylim(lo_all - pad, hi_all + pad)
        ax.axhline(0, color='black', lw=0.6, zorder=1)
        ax.set_xticks(range(len(CLASSES)), [n for _, n in CLASSES])
        ax.tick_params(axis='x', length=0, pad=3)
        ax.set_xlim(-0.55, len(CLASSES) - 0.45)
        ax.set_ylabel('Change from T0 (%)')
        rc.ygrid(ax)
        rc.panel_title(ax, f'({"ab"[k]}) {rc.LABEL[f]}, change in predicted flood cells from T0')
        h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], STRENGTH[w]), edgecolor=rc.COLOR[f], lw=0.6,
                   label=r4.WLABEL[w]) for w in LENGTHS]
        ax.legend(handles=h, loc='upper right', ncol=4, frameon=False, fontsize=ps.FONT_SIZE - 1.5,
                  handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(h_pad=1.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
