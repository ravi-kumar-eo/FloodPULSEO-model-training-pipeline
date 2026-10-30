"""Share of flood patches whose most-attended day is a rain day (Section 5.4.3).

For every flood patch and network, the day with the highest attention is a rain day when its upstream precipitation
is at least 10 mm. Bars: share of the flood patches, mean of the three seeds (computed per seed, then averaged).
Black line: the same share with the upstream precipitation of a flood patch of another activation
(rq4_attention_common.control), the level that an unrelated precipitation series gives.
The same shares with the precipitation on the patch are printed for the text.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps
import rq4_attention_common as ac

LENGTHS = (30, 15, 7, 3)
OFF, BW = {'unet': -0.19, 'vit': 0.19}, 0.34


def shares(arch, T, fp, ctrl, P):
    own, oth = [], []
    for s in ac.SEEDS:
        top = ac.relative(np.load(ac.path(arch, T, s)), T)[fp].argmax(1)
        idx = np.arange(len(fp))
        own.append(np.mean(P[fp][idx, top] >= ac.RAIN_DAY))
        oth.append(np.mean(P[ctrl][idx, top] >= ac.RAIN_DAY))
    return 100 * np.mean(own), 100 * np.mean(oth)


def main():
    fn, pn, flood, act, area = ac.rows()
    fp = np.where(flood > 0)[0]
    ctrl = ac.control(fp, act)
    ps.apply_clear()
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, 2.7))
    for g, T in enumerate(LENGTHS):
        Pu, Pl = ac.upstream(T), ac.local(T)
        for f in rc.FAMILIES:
            v, cv = shares(f, T, fp, ctrl, Pu)
            vl, cl = shares(f, T, fp, ctrl, Pl)
            print(f'{rc.LABEL[f]}-T{T}: upstream {v:.1f}% (control {cv:.1f}%), on the patch {vl:.1f}% (control {cl:.1f}%)')
            x = g + OFF[f]
            ax.bar(x, v, BW, color=ps.alpha_on_white(rc.COLOR[f], ps.FILL_ALPHA), edgecolor=rc.COLOR[f], lw=0.6,
                   zorder=2)
            ax.hlines(cv, x - BW / 2, x + BW / 2, color='black', lw=1.4, zorder=4)
            ax.text(x, max(v, cv) + 1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2)
    ax.set_xticks(range(len(LENGTHS)), [f'T{T}' for T in LENGTHS])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(LENGTHS) - 0.4)
    ax.set_ylim(0, 55)
    rc.ygrid(ax)
    ax.set_ylabel('Flood patches (%)')
    ax.set_xlabel('Length of the hydroclimatic input history')
    h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], ps.FILL_ALPHA), edgecolor=rc.COLOR[f], lw=0.6,
               label=f'{rc.LABEL[f]}, mean of three seeds') for f in rc.FAMILIES]
    h.append(Line2D([], [], color='black', lw=1.4, label='Control (precipitation of another event)'))
    fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               fontsize=ps.FONT_SIZE - 1.5, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    with mpl.rc_context({'savefig.bbox': None}):
        ps.save(fig, __file__, subdir='5_4')


if __name__ == '__main__':
    main()
