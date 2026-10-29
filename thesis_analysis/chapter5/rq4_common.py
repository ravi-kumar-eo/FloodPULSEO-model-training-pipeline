"""Shared data and drawing of the RQ4 figures (Section 5.4): networks without (T0) and with an input history of 3, 7,
15 and 30 days of the dynamic hydroclimatic variables (T3, T7, T15, T30), U-Net and ViT, seeds 1-3.
Data: plot/data/rq4/ (ch5_rq4_data.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data' / 'rq4'
WINDOWS = ('t0', 't3', 't7', 't15', 't30')
WLABEL = {'t0': 'T0', 't3': 'T3', 't7': 'T7', 't15': 'T15', 't30': 'T30'}


def fits():
    return pd.read_csv(D / 'fits.csv').set_index(['family', 'window', 'seed'])


def products():
    c = pd.read_csv(D / 'products.csv')
    c['f1'] = 2 * c.iou / (1 + c.iou)
    c['far'] = c.fp / (c.tp + c.fp)                   # NaN where the fit predicts no flood in the product
    c['miss'] = c.fn / (c.tp + c.fn)
    c['bias'] = (c.tp + c.fp) / (c.tp + c.fn)
    return c


def length_bars(panels, ref_line=None, height=5.6, windows=WINDOWS):
    """One row per measure, micro in the left and macro in the right column. In every panel each input-history length
    (T0, T3, T7, T15, T30) has a U-Net and a ViT bar, the mean of seeds 1-3, with its value on top; T0 hatched (Ravi,
    14 Oct 2026: bars as in the ablation figures, seed means only; the values of every seed are in Appendix Table A.2).
    panels: list of (title, format, value(column, family, window, seed)), column 0 micro, column 1 macro.
    ref_line: {row index: y} for a dashed reference line (area bias = 1)."""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(len(panels), 2, figsize=(ps.TEXT_WIDTH_IN, height))
    x = np.arange(len(windows))
    for r, (title, fmt, value) in enumerate(panels):
        for c in (0, 1):
            ax = axes[r, c]
            top = 0.0
            for f, dx in (('unet', -0.215), ('vit', 0.215)):
                for xi, w in zip(x, windows):
                    v = float(np.mean([value(c, f, w, s) for s in rc.SEEDS]))
                    t0 = w == 't0'
                    ax.bar(xi + dx, v, 0.36, color='white' if t0 else ps.alpha_on_white(rc.COLOR[f], 0.8),
                           edgecolor=rc.COLOR[f], hatch='////' if t0 else None, lw=0.6, zorder=2)
                    txt = fmt.format(v)
                    ax.annotate(txt[1:] if txt.startswith('0.') else txt, (xi + dx, v), xytext=(0, 1.5),
                                textcoords='offset points', ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5,
                                zorder=4)
                    top = max(top, v)
            if ref_line and r in ref_line:
                ax.axhline(ref_line[r], color='black', lw=0.6, ls=(0, (3, 2)), zorder=3)
            ax.set_ylim(0, top * 1.18)
            ax.set_xticks(x, [WLABEL[w] for w in windows])
            ax.tick_params(axis='x', length=0, pad=2)
            ax.set_xlim(-0.6, len(windows) - 0.4)
            rc.ygrid(ax)
            rc.panel_title(ax, f'({"abcdefgh"[2 * r + c]}) {title}, {("micro", "macro")[c]}')
    h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], 0.8), edgecolor=rc.COLOR[f], lw=0.6, label=rc.LABEL[f])
         for f in rc.FAMILIES]
    h += [Patch(facecolor='white', edgecolor='#555555', hatch='////', lw=0.6,
                label='T0, without dynamic hydroclimatic variables')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.0, w_pad=1.2)
    return fig, axes


def report(panels):
    """Seed means per family and length, micro and macro."""
    for title, _, val in panels:
        for c, grp in enumerate(('micro', 'macro')):
            print(f'{title:18s} {grp}: ' + ' | '.join(
                f'{rc.LABEL[f]} ' + ' '.join(f'{np.mean([val(c, f, w, s) for s in rc.SEEDS]):.4f}' for w in WINDOWS)
                for f in rc.FAMILIES))
