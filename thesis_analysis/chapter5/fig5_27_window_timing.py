"""RQ4 — Micro IoU of every input-history length within classes of the timing of the antecedent precipitation
(timing.csv, ch5_rq4_timing_data.py; upstream precipitation).

Timing class of a test patch: the shortest input history whose days hold at least half of the upstream precipitation
of the 30 days before the mapping date (days -3 to -1, -7 to -4, -15 to -8 or -30 to -16); the 10 patches with less
than 1 mm in 30 days are left out. Bars: micro IoU pooled over the scored cells of the class, mean of seeds 1-3
(Ravi, 14 Oct 2026: seed means only), value on top; T0 hatched, T3 to T30 in the family colour, lighter for shorter
histories. The number of patches of each class is written under its label.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_27_window_timing.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import rq1_common as rc
import rq4_common as r4
from rq1_common import ps

CLASSES = (('t3', 'Days −3 to −1'), ('t7', 'Days −7 to −4'), ('t15', 'Days −15 to −8'), ('t30', 'Days −30 to −16'))
STRENGTH = {'t3': 0.35, 't7': 0.55, 't15': 0.78, 't30': 1.0}
WIDTH = 0.16


def main():
    rc.style()
    d = pd.read_csv(r4.D / 'timing.csv')
    d = d[d.source == 'upstream'].set_index(['class', 'family', 'window'])
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 5.0))
    for ax, (k, f) in zip(axes, enumerate(rc.FAMILIES)):
        top = 0.0
        for g, (c, _) in enumerate(CLASSES):
            for j, w in enumerate(r4.WINDOWS):
                v = d.loc[(c, f, w), 'iou']
                x = g + (j - 2) * (WIDTH + 0.01)
                t0 = w == 't0'
                ax.bar(x, v, WIDTH, color='white' if t0 else ps.alpha_on_white(rc.COLOR[f], STRENGTH[w]),
                       edgecolor=rc.COLOR[f], hatch='////' if t0 else None, lw=0.6, zorder=2)
                txt = f'{v:.3f}'
                ax.annotate(txt[1:] if txt.startswith('0.') else txt, (x, v), xytext=(0, 1.5),
                            textcoords='offset points', rotation=90, ha='center', va='bottom',
                            fontsize=ps.FONT_SIZE - 2.5, zorder=4)
                top = max(top, v)
        ax.set_ylim(0, top * 1.3)
        ax.set_xticks(range(len(CLASSES)),
                      [f'{n}\n{d.loc[(c, f, "t0"), "patches"]:,} patches' for c, n in CLASSES])
        ax.tick_params(axis='x', length=0, pad=3)
        ax.set_xlim(-0.55, len(CLASSES) - 0.45)
        ax.set_ylabel('Micro IoU')
        rc.ygrid(ax)
        rc.panel_title(ax, f'({"ab"[k]}) {rc.LABEL[f]}')
        h = [Patch(facecolor='white' if w == 't0' else ps.alpha_on_white(rc.COLOR[f], STRENGTH[w]),
                   edgecolor=rc.COLOR[f], hatch='////' if w == 't0' else None, lw=0.6, label=r4.WLABEL[w])
             for w in r4.WINDOWS]
        ax.legend(handles=h, loc='upper left', ncol=5, frameon=False, fontsize=ps.FONT_SIZE - 1.5,
                  handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout(h_pad=1.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    for f in rc.FAMILIES:
        for c, n in CLASSES:
            print(rc.LABEL[f], n, ' '.join(f'{r4.WLABEL[w]} {d.loc[(c, f, w), "iou"]:.3f}' for w in r4.WINDOWS))


if __name__ == '__main__':
    main()
