"""RQ4 — Precision-recall curves by input-history length per family on the test subset (micro, all scored cells
pooled), with the operating points at the validation-selected thresholds.

Curves from the probability histograms of ch5_rq4_hist.py (1,000 bins), as in Figure 5.2: precision averaged over
seeds 1-3 at each recall (the seed means only; Ravi, 14 Oct 2026). T0 dashed black, T3 to T30 solid in the family
colour, lighter for shorter histories. Markers: mean recall and precision of the three seeds at their selected
thresholds. Dotted line: flooded share of the scored cells.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_26_window_pr_curves.py
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import rq1_common as rc
import rq4_common as r4
from rq1_common import ps

GRID = np.linspace(0.02, 0.98, 97)
STRENGTH = {'t3': 0.35, 't7': 0.55, 't15': 0.78, 't30': 1.0}


def curve(h, k):
    fl, dr = h[f'{k}_flood'].astype(float), h[f'{k}_dry'].astype(float)
    tp, fp = np.cumsum(fl[::-1])[::-1], np.cumsum(dr[::-1])[::-1]
    rec, prec = tp / fl.sum(), tp / np.maximum(tp + fp, 1)
    o = np.argsort(rec)
    return np.interp(GRID, rec[o], prec[o]), fl.sum() / (fl.sum() + dr.sum())


def main():
    rc.style()
    h = np.load(r4.D / 'hist.npz')
    fi = r4.fits()
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH_IN, 2.9), sharey=True)
    for ax, (k, f) in zip(axes, enumerate(rc.FAMILIES)):
        handles = []
        for w in r4.WINDOWS:
            col = 'black' if w == 't0' else ps.alpha_on_white(rc.COLOR[f], STRENGTH[w])
            ls = (0, (4, 2)) if w == 't0' else '-'
            P = np.array([curve(h, f'{f}_{w}_s{s}')[0] for s in rc.SEEDS]).mean(0)
            ax.plot(GRID, P, color=col, lw=1.1, ls=ls, zorder=3)
            r = np.mean([fi.loc[(f, w, s), 'recall'] for s in rc.SEEDS])
            p = np.mean([fi.loc[(f, w, s), 'precision'] for s in rc.SEEDS])
            ax.scatter([r], [p], s=24, facecolor='white' if w == 't0' else col, edgecolor=col, lw=1.1, zorder=4)
            handles.append(Line2D([], [], color=col, lw=1.1, ls=ls, marker='o', ms=4,
                                  mfc='white' if w == 't0' else col, mec=col, label=r4.WLABEL[w]))
            print(f'{rc.LABEL[f]} {r4.WLABEL[w]}: precision at recall 0.3/0.5/0.7/0.9 '
                  + ' '.join(f'{P[np.argmin(np.abs(GRID - R))]:.3f}' for R in (0.3, 0.5, 0.7, 0.9))
                  + f' | operating point recall {r:.3f} precision {p:.3f}')
        base = curve(h, f'{f}_t0_s1')[1]
        ax.axhline(base, color='black', lw=0.6, ls=(0, (1, 1.5)), zorder=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.set_xlabel('Recall')
        if k == 0:
            ax.set_ylabel('Precision')
        rc.ygrid(ax)
        ax.xaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
        rc.panel_title(ax, f'({"ab"[k]}) {rc.LABEL[f]}')
        ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=ps.FONT_SIZE - 1.5, handlelength=1.8,
                  labelspacing=0.3)
    rc.yticks_everywhere(axes)
    fig.tight_layout(w_pad=1.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
