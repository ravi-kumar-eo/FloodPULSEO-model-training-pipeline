"""Shared drawing helpers of the RQ1 figures (rq1_01 ... rq1_04). Data: plot/data/rq1/ (built by rq1_data.py).

Encoding used in every RQ1 figure: U-Net orange, ViT blue (data-paper blue). Where the seeds are shown, each seed is
its own bar in the family colour at three strengths (seed 1 strongest), with its value written above. "Micro" = all
test cells pooled; "macro" = mean over the flood-positive mapping products.
"""
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import plot_style as ps  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'data' / 'rq1'
FAMILIES = ('unet', 'vit')
LABEL = {'unet': 'U-Net', 'vit': 'ViT'}
COLOR = {'unet': '#E0942A', 'vit': '#0257CE'}
SEEDS = (1, 2, 3)
SEED_STRENGTH = {1: 0.85, 2: 0.55, 3: 0.3}          # fill strength on white; outline in the full family colour
OFFSETS = {('unet', 1): -0.385, ('unet', 2): -0.27, ('unet', 3): -0.155,
           ('vit', 1): 0.155, ('vit', 2): 0.27, ('vit', 3): 0.385}
BAR_W = 0.105
CH5_FONT_SIZE = 9                                  # shared style: 7


# Chapter 5 figure number -> folder of plot/png, one folder per section (Ravi, 14 Oct 2026). Appendix figures
# (figA_*) go to plot/png/appendix. Extend the last range when a section receives new figures.
SECTION_FIGURES = {'5_1': range(1, 11), '5_2': range(11, 15), '5_3': range(15, 25), '5_4': range(25, 40), '5_5': range(40, 44), '5_6': range(44, 49)}


def section_dir(script_file):
    """plot/png subfolder of a figure script, from its figure number (fig5_<n>_*.py) or 'appendix' (figA_*.py)."""
    stem = Path(script_file).stem
    if stem.startswith('figA_'):
        return 'appendix'
    n = int(stem.split('_')[1])
    for sec, nums in SECTION_FIGURES.items():
        if n in nums:
            return sec
    raise ValueError(f'{stem}: figure number {n} has no section in SECTION_FIGURES')


def style():
    """Thesis chart style (plot_style.apply_soft) with every text element in black: tick labels, axis labels, titles,
    legends and annotations (Ravi, 12 Oct 2026: grey text is hard to read), and black axis lines and ticks (13 Oct).
    Chapter 5 text is 2 pt larger than the shared style (Ravi, 13 Oct 2026): at print size the smallest text is 6.5 pt,
    ticks 7.5 pt, axis labels and legends 8 pt and panel titles 9 pt, within the 6-8 pt minimum of journal guidelines."""
    ps.FONT_SIZE = CH5_FONT_SIZE                       # read at draw time by apply_soft() and every fontsize below
    ps.TITLE_SIZE = CH5_FONT_SIZE + 1
    ps.apply_soft()
    mpl.rcParams.update({'text.color': 'black', 'axes.labelcolor': 'black', 'axes.titlecolor': 'black',
                         'xtick.labelcolor': 'black', 'ytick.labelcolor': 'black', 'legend.labelcolor': 'black',
                         # black axis lines and ticks, easier to read than the grey of the soft style (Ravi, 13 Oct 2026)
                         'axes.edgecolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})


def fits():
    return pd.read_csv(DATA / 'fits.csv')


def products():
    return pd.read_csv(DATA / 'products.csv')


def agreement():
    return pd.read_csv(DATA / 'agreement.csv')


def seed_face(f, s):
    return ps.alpha_on_white(COLOR[f], SEED_STRENGTH[s])


def panel_title(ax, text):
    ax.set_title(text, loc='left', fontsize=ps.FONT_SIZE, color='black', pad=4)


def yticks_everywhere(axes):
    """Value labels on the y-axis of every panel, also where panels share their y-axis (Ravi, 13 Oct 2026)."""
    for ax in np.ravel(axes):
        ax.tick_params(axis='y', labelleft=True)


def ygrid(ax):
    """Grid lines at the labelled values, and an unlabelled minor tick halfway between them on a linear axis."""
    from matplotlib.ticker import AutoMinorLocator
    ax.yaxis.grid(True, color=ps.GREY_GRID, lw=0.5)
    ax.set_axisbelow(True)
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def seed_bars(ax, groups, value, labels='above', fmt='{:.3f}'):
    """Six bars per group (U-Net seeds 1-3, ViT seeds 1-3).
    groups: list of x labels; value(group_index, family, seed) -> float.
    labels: 'above' writes each value above its bar, 'rotated' writes it upright along the bar top, None writes none."""
    top = 0.0
    for g in range(len(groups)):
        for f in FAMILIES:
            for s in SEEDS:
                v = value(g, f, s)
                x = g + OFFSETS[(f, s)]
                ax.bar(x, v, BAR_W, color=seed_face(f, s), edgecolor=COLOR[f], lw=0.6, zorder=2)
                if labels == 'above':
                    ax.text(x, v, fmt.format(v), ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5,
                            color='black')
                elif labels == 'rotated':
                    ax.annotate(fmt.format(v), (x, v), xytext=(0, 1.5), textcoords='offset points', rotation=90,
                                ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5, color='black')
                top = max(top, v)
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, top * (1.32 if labels == 'rotated' else 1.15))
    ygrid(ax)


def seed_legend(fig, y):
    h = []
    for s in SEEDS:                                   # column-major fill: rows become U-Net, ViT
        for f in FAMILIES:
            h.append(Patch(facecolor=seed_face(f, s), edgecolor=COLOR[f], lw=0.6, label=f'{LABEL[f]} seed {s}'))
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, y), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.0, columnspacing=1.6)


def macro(pr, column, f, s):
    """Mean of a per-product score over the flood-positive mapping products of one fit (where it is defined)."""
    return float(pr[(pr.family == f) & (pr.seed == s)][column].dropna().mean())


RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final/final_t0')


def product_counts():
    """TP, FP and FN of every flood-positive test product for the six T0 fits, from each run's
    plots/per_product_test.csv (the source of products.csv). Checked: the IoU from these counts equals products.csv."""
    out = []
    for f in FAMILIES:
        for s in SEEDS:
            d = pd.read_csv(RUNS / f'seed{s}' / f / 'plots' / 'per_product_test.csv')
            d = d[d.flood > 0].assign(family=f, seed=s)
            out.append(d[['folder_name', 'family', 'seed', 'flood', 'tp', 'fp', 'fn', 'auprc']])
    c = pd.concat(out, ignore_index=True)
    pr = products().rename(columns={'product': 'folder_name'})
    m = c.merge(pr[['folder_name', 'family', 'seed', 'iou']], on=['folder_name', 'family', 'seed'], validate='1:1')
    assert len(m) == len(pr) == len(c)
    assert np.allclose(m.tp / (m.tp + m.fp + m.fn), m.iou, atol=1e-6)
    return c


POS = {('unet', 1): 0.0, ('unet', 2): 1.0, ('unet', 3): 2.0, ('vit', 1): 3.7, ('vit', 2): 4.7, ('vit', 3): 5.7}


def metric_grid(panels, ref_line=None, yfmt=None):
    """Two rows (micro, macro) by one column per metric, each column on its own y-scale shared by its two rows.
    In every panel: U-Net seeds 1-3, then ViT seeds 1-3, the seed number under each bar, the family under its three
    bars and the value written horizontally above each bar. Seeds are told apart by position and number, so the
    figure needs no colour legend. panels: list of (title, fmt, value(row, family, seed)), row 0 micro, row 1 macro.
    ref_line: {column index: y} for a dashed reference line (e.g. area bias = 1).
    yfmt: tick format for every column (e.g. '%.2f'), so neighbouring panels use the same number of decimals."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.transforms import blended_transform_factory
    fig, axes = plt.subplots(2, len(panels), figsize=(ps.TEXT_WIDTH_IN, 3.55), sharey='col')
    for c, (title, fmt, value) in enumerate(panels):
        top = 0.0
        for r in (0, 1):
            ax = axes[r, c]
            for f in FAMILIES:
                for s in SEEDS:
                    v = value(r, f, s)
                    ax.bar(POS[(f, s)], v, 0.8, color=seed_face(f, s), edgecolor=COLOR[f], lw=0.6, zorder=2)
                    txt = fmt.format(v)
                    txt = txt[1:] if txt.startswith('0.') else txt   # '.194': fits the bar at 6.5 pt (exact values: Table 5.1)
                    ax.text(POS[(f, s)], v, txt, ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5,
                            color='black')
                    top = max(top, v)
            ax.set_xticks([POS[k] for k in POS], [str(k[1]) for k in POS])
            ax.tick_params(axis='x', length=0, pad=1.5, labelsize=ps.FONT_SIZE - 2.5)
            ax.set_xlim(-0.7, 6.4)
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            for f, x in (('unet', 1.0), ('vit', 4.7)):
                ax.text(x, -0.105, LABEL[f], transform=tr, ha='center', va='top', fontsize=ps.FONT_SIZE - 1.5,
                        color='black')
            if ref_line and c in ref_line:
                ax.axhline(ref_line[c], color='black', lw=0.6, ls=(0, (3, 2)), zorder=3)
            if yfmt:
                ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))
            ygrid(ax)
        axes[0, c].set_ylim(0, top * 1.14)
        panel_title(axes[0, c], title)
    for r, name in enumerate(('Micro', 'Macro')):
        axes[r, 0].set_ylabel(name)
    fig.tight_layout(h_pad=1.4, w_pad=1.0)
    return fig, axes
