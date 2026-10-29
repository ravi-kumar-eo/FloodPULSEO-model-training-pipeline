"""Attention of the flood patches around their wettest 3 days of upstream precipitation (Section 5.4.3).

Flood patches (>= 1 labelled flood cell) with at least one day of >= 10 mm upstream precipitation in the input
history. The wettest 3 days are the three consecutive days with the largest upstream total. Period 0 = those days,
-1 = the 3 days before, +1 = the 3 days after, and so on; the attention of a period is the mean over its 3 days,
relative to uniform, of the seed-mean attention. Only periods that lie fully inside the input history count, and a
period is drawn when at least 50 patches have it. Control (dotted): the same with the wettest 3 days of a flood patch
of another activation (rq4_attention_common.control). T3 has a single 3-day period and is left out.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import rq1_common as rc
from rq1_common import ps
import rq4_attention_common as ac

K = np.arange(-3, 4)


def periods(A, start, T):
    out = []
    for k in K:
        s0 = start + 3 * k
        ok = (s0 >= 0) & (s0 + 2 <= T - 1)
        r = np.where(ok)[0]
        out.append(np.mean([A[r, s0[ok] + j] for j in range(3)], axis=0).mean() if len(r) >= 50 else np.nan)
    return np.array(out)


def main():
    fn, pn, flood, act, area = ac.rows()
    fp = np.where(flood > 0)[0]
    ctrl = ac.control(fp, act)
    ps.apply_clear()
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.9))
    for ax, T, lab in zip(axes, (30, 15, 7), 'abc'):
        P = ac.upstream(T)
        m = P[fp].max(1) >= ac.RAIN_DAY
        S3 = P[:, :-2] + P[:, 1:-1] + P[:, 2:]
        own, oth = S3[fp][m].argmax(1), S3[ctrl][m].argmax(1)
        for f in rc.FAMILIES:
            A = ac.seed_mean(f, T)[fp][m]
            p_own, p_oth = periods(A, own, T), periods(A, oth, T)
            ax.plot(K, p_own, color=rc.COLOR[f], lw=1.6, marker='o', ms=3.2, zorder=3)
            ax.plot(K, p_oth, color=rc.COLOR[f], lw=1.1, ls=':', zorder=3)
            print(f'{rc.LABEL[f]}-T{T}: patches {m.sum():,}; own ' + ' '.join(f'{v:.3f}' for v in p_own)
                  + ' | control ' + ' '.join(f'{v:.3f}' for v in p_oth))
        ax.axvline(0, color='#999999', lw=0.5)
        ax.axhline(1, color='black', lw=0.4, ls=':')
        ax.set_xticks(K, ['−3', '−2', '−1', '0', '+1', '+2', '+3'])
        ax.set_xlim(-3.4, 3.4)
        rc.ygrid(ax)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.18 * (hi - lo))
        ax.text(0.03, 0.97, f'{int(m.sum()):,} flood patches', transform=ax.transAxes, ha='left', va='top',
                fontsize=ps.FONT_SIZE - 2)
        rc.panel_title(ax, f'({lab}) T{T}')
    axes[0].set_ylabel('Attention relative\nto uniform')
    fig.supxlabel('3-day period, counted from the wettest 3 days of upstream precipitation (0)', fontsize=ps.FONT_SIZE)
    h = [Line2D([], [], color=rc.COLOR[f], lw=1.6, label=f'{rc.LABEL[f]}, mean of three seeds') for f in rc.FAMILIES]
    h.append(Line2D([], [], color='#777777', lw=1.1, ls=':', label='Control (precipitation of another event)'))
    fig.legend(handles=h, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               fontsize=ps.FONT_SIZE - 1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.2)
    with mpl.rc_context({'savefig.bbox': None}):
        ps.save(fig, __file__, subdir='5_4')


if __name__ == '__main__':
    main()
