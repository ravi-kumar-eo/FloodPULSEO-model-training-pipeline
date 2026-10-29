"""Section 5.4.4 summary of consecutive-product change metrics."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from PIL import Image

from rq1_common import COLOR, ps

D = Path(__file__).resolve().parents[1] / 'data/repeat'
CONFIGS = ['t0', 't3', 't7', 't15', 't30']
LABELS = ['T0', 'T3', 'T7', 'T15', 'T30']
FAMILIES = ('unet', 'vit')
FAMILY_LABEL = {'unet': 'U-Net', 'vit': 'ViT'}
X = np.arange(len(CONFIGS))
BAR_WIDTH = 0.31
OFFSETS = {'unet': -0.18, 'vit': 0.18}


def metric_rows(data, metric):
    """Return the five seed-mean estimates and cluster intervals in plot order."""
    rows = {}
    for family in FAMILIES:
        q = data[(data.family == family) & (data.metric == metric)].set_index('config').loc[CONFIGS]
        assert (q.n_seeds == 3).all()
        rows[family] = q
    return rows


def panel(ax, data, metric, title, ylabel, scale=1, reference=None):
    rows = metric_rows(data, metric)
    lows, highs = [], []
    for family in FAMILIES:
        q = rows[family]
        x = X + OFFSETS[family]
        values = q.estimate.to_numpy() * scale
        lo = q.lo.to_numpy() * scale
        hi = q.hi.to_numpy() * scale
        bars = ax.bar(
            x, values, width=BAR_WIDTH, bottom=0,
            color=ps.alpha_on_white(COLOR[family], ps.CLEAR_FILL),
            edgecolor=COLOR[family], linewidth=0.8, zorder=3,
        )
        bars[0].set_hatch('////')
        ax.errorbar(
            x, values, yerr=np.vstack((values - lo, hi - values)), fmt='none',
            ecolor='black', elinewidth=0.85, capsize=2.2, capthick=0.85, zorder=5,
        )
        lows.extend(lo)
        highs.extend(hi)

    low = min(0.0, float(np.min(lows)))
    high = max(0.0, float(np.max(highs)))
    if reference is not None:
        high = max(high, reference[0])
    span = high - low
    ax.set_ylim(low - (0.08 * span if low < 0 else 0), high + 0.13 * span)
    ax.set_xlim(-0.62, X[-1] + 0.62)
    ax.set_xticks(X, LABELS)
    ax.tick_params(axis='x', length=0, pad=3)
    ax.axhline(0, color='black', lw=0.7, zorder=2)
    if reference is not None:
        value, label = reference
        ax.axhline(value, color='#666666', ls=(0, (4, 2.5)), lw=0.9, zorder=1)
        ax.annotate(
            label, xy=(0.015, value), xycoords=('axes fraction', 'data'),
            xytext=(0, -3 if '\n' in label else 2.5), textcoords='offset points', ha='left',
            va='top' if '\n' in label else 'bottom',
            color='#555555', fontsize=ps.FONT_SIZE - 1.5,
        )
    ax.set_title(title, loc='left', pad=6)
    ax.set_ylabel(ylabel)
    ps.ygrid(ax)


def output_report(path):
    with Image.open(path) as image:
        print(f'{path}: {image.width} x {image.height} px')


def main():
    ps.apply_clear()
    data = pd.read_csv(D / 'summary.csv')
    data = data[(data.scope == 'same_activation') & data.config.isin(CONFIGS)]
    label_change = float(data[data.metric == 'label_flip_share'].estimate.iloc[0]) * 100

    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 7.0))
    grid = fig.add_gridspec(3, 2, left=0.115, right=0.985, top=0.965, bottom=0.10,
                            hspace=0.54, wspace=0.34)
    axes = [
        fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, :]),
    ]
    specifications = [
        ('direction', '(a) Direction agreement', 'Agreement (%)', 100,
         (50.4, 'Always\nexpansion\n50.4%')),
        ('elasticity', '(b) Log-change slope', 'Slope', 1, None),
        ('change_pod', '(c) Change POD', 'Change POD (%)', 100, None),
        ('change_precision', '(d) Change precision', 'Change precision (%)', 100, None),
        ('flip_share', '(e) Predicted cells changing', 'Predicted cells changing (%)', 100,
         (label_change, f'Labelled cells changing: {label_change:.3f}%')),
    ]
    for ax, args in zip(axes, specifications):
        panel(ax, data, *args)

    handles = [
        Patch(facecolor=ps.alpha_on_white(COLOR[f], ps.CLEAR_FILL), edgecolor=COLOR[f], label=FAMILY_LABEL[f])
        for f in FAMILIES
    ]
    handles.append(Patch(facecolor='white', edgecolor='#555555', hatch='////', label='T0'))
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.012),
               columnspacing=1.5, handlelength=1.3)
    out_script = Path(__file__).with_name('fig5_33_repeat_summary.py')
    with mpl.rc_context({'savefig.bbox': None}):
        out = ps.save(fig, out_script, subdir='5_4')
    output_report(out)

    print('Figure 5.33 plotted seed means with 95% cluster intervals:')
    for metric, _, _, scale, _ in specifications:
        print(f'  {metric}:')
        rows = metric_rows(data, metric)
        for family in FAMILIES:
            q = rows[family]
            values = ', '.join(
                f'{label} {estimate * scale:.4f} [{lo * scale:.4f}, {hi * scale:.4f}]'
                for label, estimate, lo, hi in zip(LABELS, q.estimate, q.lo, q.hi)
            )
            print(f'    {FAMILY_LABEL[family]}: {values}')
    print(f'  labelled-cell changing share reference: {label_change:.4f}%')


if __name__ == '__main__':
    main()
