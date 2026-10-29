"""Shared data and drawing of the Section 5.5 figures (uncertainty and error analysis): U-Net and ViT without (T0)
and with (T30) the 30-day input history, seeds 1-3. Data: plot/data/uncertainty/ (ch5_uncertainty_data.py)."""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data' / 'uncertainty'
CONFIGS = (('unet', 't0'), ('unet', 't30'), ('vit', 't0'), ('vit', 't30'))
CLABEL = {('unet', 't0'): 'U-Net-T0', ('unet', 't30'): 'U-Net-T30', ('vit', 't0'): 'ViT-T0', ('vit', 't30'): 'ViT-T30'}
FILL = 0.8


def read(name):
    return pd.read_csv(D / f'{name}.csv')


def bar_style(f, w):
    """T0 hatched on white, T30 filled, in the family colour (as in Figure 5.16)."""
    t0 = w == 't0'
    return dict(color='white' if t0 else ps.alpha_on_white(rc.COLOR[f], FILL), edgecolor=rc.COLOR[f],
                hatch='////' if t0 else None, lw=0.6)


def grouped_bars(ax, labels, values, fmt='{:.0f}', width=0.19):
    """One group per label, four bars per group (U-Net-T0, U-Net-T30, ViT-T0, ViT-T30) with the value on top.
    values: {config: array over labels}."""
    x = np.arange(len(labels))
    offs = (-1.5 * width - 0.02, -0.5 * width - 0.02, 0.5 * width + 0.02, 1.5 * width + 0.02)
    top = 0.0
    for (cfg, dx) in zip(CONFIGS, offs):
        v = np.asarray(values[cfg], float)
        ax.bar(x + dx, v, width, zorder=2, **bar_style(*cfg))
        for xi, vi in zip(x, v):
            txt = fmt.format(vi)
            ax.annotate(txt[1:] if txt.startswith('0.') else txt, (xi + dx, vi), xytext=(0, 1.5),
                        textcoords='offset points', ha='center', va='bottom', fontsize=ps.FONT_SIZE - 3, zorder=4)
        top = max(top, np.nanmax(v))
    ax.set_xticks(x, labels)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, top * 1.18)
    rc.ygrid(ax)


def legend_handles():
    out = []
    for c in CONFIGS:
        st = bar_style(*c)
        st['facecolor'] = st.pop('color')
        out.append(Patch(label=CLABEL[c], **st))
    return out
