"""Shared drawing of the loss curves: Figure 5.3 (T0), Figure 5.17 (T30), Appendix Figures A.42 to A.44 (T3, T7, T15)
and A.45 (T30 ablations). Data: plot/data/training/ (built by ch5_loss_curves_data.py).

Layout (Ravi, 21 Oct 2026, second round): only the training and validation loss matter, so one wide panel per seed
(rows seed 1, 2, 3; for the seed-1 ablations one row per ablation), each with the training loss (solid) and the
validation loss (dashed) of U-Net (orange) and ViT (blue), thin lines so that the trend stays visible. Open circle:
lowest validation loss. Filled dot: selected checkpoint (highest validation IoU at the best of 49 thresholds), drawn on
the validation-loss curve. Grey band: the warm-up epochs of the ViT trained from random initialisation, not eligible for
the checkpoint (Section 4.5.3). The training loss comes from batches of about half flood-bearing patches and the
validation loss from the natural mix of the validation subset (Section 4.5.2), so the two losses compare in shape, not
in level.
"""
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import rq1_common as rc
from rq1_common import ps

DATA = Path(__file__).resolve().parents[1] / 'data' / 'training'
SEED_COLOR = {1: '#1B9E77', 2: '#7570B3', 3: '#E7298A'}          # Dark2 green, purple, magenta: colour-blind safe
ABLATION_COLOR = {'noup': '#1B9E77', 'nosum': '#7570B3', 'sumonly': '#E7298A', 'scratch': '#666666'}
WARMUP_BG = '#ECECEC'
DASH = (0, (3.0, 1.6))
LW, MS = 0.9, 14


def load():
    return pd.read_csv(DATA / 'curves.csv'), pd.read_csv(DATA / 'runs.csv')


def draw(ax, curves, runs, config, seed, title):
    """One panel: training (solid) and validation (dashed) loss of U-Net and ViT for one configuration and seed."""
    last = 0
    for f in rc.FAMILIES:
        c = curves[(curves.config == config) & (curves.family == f) & (curves.seed == seed)]
        if c.empty:
            continue
        r = runs[(runs.config == config) & (runs.family == f) & (runs.seed == seed)].iloc[0]
        col = rc.COLOR[f]
        ax.plot(c.epoch, c.train_loss, color=col, lw=LW, zorder=3)
        ax.plot(c.epoch, c.val_loss, color=col, lw=LW, ls=DASH, zorder=3)
        lo = c[c.epoch == r.lowest_val_loss_epoch].iloc[0]
        ax.scatter(lo.epoch, lo.val_loss, s=MS, facecolor='white', edgecolor=col, lw=0.9, zorder=5)
        sel = c[c.epoch == r.selected_epoch].iloc[0]
        ax.scatter(sel.epoch, sel.val_loss, s=MS + 4, facecolor=col, edgecolor=col, lw=0.6, zorder=6)
        if r.warmup_epochs:
            ax.axvspan(0.5, r.warmup_epochs + 0.5, color=WARMUP_BG, lw=0, zorder=0)
        last = max(last, int(c.epoch.max()))
    ax.set_xlim(0.5, last + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=14, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel('Loss')
    rc.ygrid(ax)
    rc.panel_title(ax, title)


def legend(fig, y, warmup=False):
    h = [Line2D([], [], color=rc.COLOR[f], lw=1.6, label=rc.LABEL[f]) for f in rc.FAMILIES]
    h += [Line2D([], [], color='#444444', lw=1.0, label='Training loss'),
          Line2D([], [], color='#444444', lw=1.0, ls=DASH, label='Validation loss'),
          Line2D([], [], color='#444444', lw=0, marker='o', markersize=4.2, markerfacecolor='white',
                 markeredgecolor='#444444', markeredgewidth=0.9, label='Lowest validation loss'),
          Line2D([], [], color='#444444', lw=0, marker='o', markersize=4.4, markerfacecolor='#444444',
                 markeredgecolor='#444444', label='Selected checkpoint')]
    if warmup:
        h.append(Patch(facecolor=WARMUP_BG, edgecolor='none', label='ViT warm-up'))
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, y), ncol=4, frameon=False,
               fontsize=ps.FONT_SIZE, handlelength=2.2, columnspacing=1.4)


def rows_figure(script_file, rows, warmup, height):
    """rows: (config, seed, panel title) per panel; one wide panel per row."""
    import matplotlib.pyplot as plt
    rc.style()
    curves, runs = load()
    fig, axes = plt.subplots(len(rows), 1, figsize=(ps.TEXT_WIDTH_IN, height))
    for k, (ax, (config, seed, title)) in enumerate(zip(axes, rows)):
        draw(ax, curves, runs, config, seed, f'({"abcdefgh"[k]}) {title}')
    axes[-1].set_xlabel('Epoch')
    fig.tight_layout(h_pad=1.0, rect=(0, 0.07, 1, 1))
    legend(fig, 0.06, warmup=warmup)
    ps.save(fig, script_file, subdir=rc.section_dir(script_file))


def network_figure(script_file, config, name, height=7.6):
    """Figures 5.3 (T0), 5.17 (T30) and A.42 to A.44 (T3, T7, T15): one panel per seed, U-Net and ViT together."""
    curves, runs = load()
    warm = bool(runs[runs.config == config].warmup_epochs.max() > 0)
    rows_figure(script_file, [(config, s, f'{name}, seed {s}') for s in rc.SEEDS], warm, height)
