"""Shared drawing of the condition figures (RQ1): U-Net-T0 and ViT-T0 by class of a characteristic of the flood event,
the patch or the 80 m cell. Data: plot/data/conditions/screen_rq1.csv (ch5_cells_screen.py, computed on Bolzano from
ch5_cells_build.py's tables). Every metric is pooled over the scored cells of the class and averaged over seeds 1-3.

Layout: one row per characteristic; columns: share of all reference flood cells in the class, POD = TP / (TP + FN),
POFD = FP / (FP + TN), IoU = TP / (TP + FP + FN). Each column shares its vertical scale across the rows. An asterisk
marks a class whose 95% cluster bootstrap interval of ViT - U-Net (61 clusters, 4,000 draws) excludes zero.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

BAR_FILL = 0.6             # bar fill strength on white (lighter, Ravi 13 Oct); the outline keeps the full colour
SCREEN = Path(__file__).resolve().parents[1] / 'data' / 'conditions' / 'screen_rq1.csv'
METRICS = (('pod', 'POD'), ('pofd', 'POFD'), ('iou', 'IoU'))
NICE = {'A Tropical': 'Tropical', 'B Arid': 'Arid', 'C Temperate': 'Temperate', 'D Cold': 'Cold', 'E Polar': 'Polar',
        'very-high': 'Very high', 'high': 'High', 'medium': 'Medium', 'adjacent (1 cell)': '1', '2-3 cells': '2–3',
        '4-10 cells': '4–10', '>10 cells': '>10', 'none in patch': 'None'}


def label(c):
    if c in NICE:
        return NICE[c]
    c = c.split(' (')[1].rstrip(')') if c.startswith('Q') and '(' in c else c
    for unit in (' km2', ' m', ' deg'):
        c = c.replace(unit, '')
    return c.replace('>=', '≥').replace('<-', '<−').replace('-', '–').replace('–0.', '−0.', 1) if c.startswith('-') \
        else c.replace('>=', '≥').replace('-', '–')


RELABEL = {'patch_max_uda': ['<1', '1–10', '10–10²', '10²–10³', '10³–10⁴', '≥10⁴'],
           'uda_log10': ['<1', '1–10', '10–10²', '10²–10³', '10³–10⁴', '≥10⁴'],
           'patch_pw_share': ['0', '<1', '1–5', '5–20', '≥20'],
           'patch_lowland_share': ['0', '<10', '10–25', '25–50', '≥50'],
           'ndbi': ['<−0.2', '−0.2–0', '0–0.2', '≥0.2']}


def draw(features, script_file, row_h=1.15):
    """features: list of (feature name in screen_rq1.csv, row title)."""
    rc.style()
    d = pd.read_csv(SCREEN)
    d = d[d['class'] != 'no value']
    n = len(features)
    fig, axes = plt.subplots(n, 4, figsize=(ps.TEXT_WIDTH_IN, row_h * n + 0.45), sharey='col', squeeze=False,
                             gridspec_kw={'width_ratios': [0.8, 1, 1, 1]})
    for i, (feat, title) in enumerate(features):
        g = d[d.feature == feat].reset_index(drop=True)
        x = np.arange(len(g))
        a = axes[i, 0]
        a.bar(x, 100 * g.share_flood, 0.6, color=ps.alpha_on_white('#4F5D6E', 0.35), lw=0, zorder=2)
        a.set_ylabel(title, fontsize=ps.FONT_SIZE - 1.5)
        for j, (m, lab) in enumerate(METRICS, start=1):
            ax = axes[i, j]
            for f, dx in (('unet', -0.13), ('vit', 0.13)):
                ax.scatter(x + dx, g[f'{f}_{m}'], s=13, color=rc.COLOR[f], lw=0, zorder=3)
            sig = (g[f'diff_{m}_lo'] > 0) | (g[f'diff_{m}_hi'] < 0)
            top = np.maximum(g[f'unet_{m}'], g[f'vit_{m}'])
            for k in np.where(sig)[0]:
                ax.annotate('*', (x[k], top[k]), xytext=(0, 1.5), textcoords='offset points', ha='center',
                            va='bottom', fontsize=ps.FONT_SIZE, color='black')
            if i == 0:
                rc.panel_title(ax, lab)
        if i == 0:
            rc.panel_title(a, 'Flood cells (%)')
        labs = RELABEL.get(feat, [label(c) for c in g['class']])
        turn = len(labs) >= 6 or max(len(t) for t in labs) > 4
        for ax in axes[i]:
            ax.set_xticks(x, labs, fontsize=ps.FONT_SIZE - 2.5, rotation=45 if turn else 0,
                          ha='right' if turn else 'center', rotation_mode='anchor')
            ax.tick_params(axis='x', length=0, pad=1.5)
            ax.set_xlim(-0.6, len(g) - 0.4)
            rc.ygrid(ax)
    for j in range(4):
        axes[0, j].set_ylim(0, None)
        top = max(ax.get_ylim()[1] for ax in axes[:, j])
        axes[0, j].set_ylim(0, top * 1.12)
    h = [Line2D([], [], ls='', marker='o', ms=4, color=rc.COLOR[f], label=rc.LABEL[f]) for f in rc.FAMILIES] + \
        [Line2D([], [], ls='', marker='$*$', ms=6, color='black', label='95% interval of ViT − U-Net excludes zero')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handletextpad=0.3, columnspacing=1.6)
    rc.yticks_everywhere(axes)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    ps.save(fig, script_file, subdir=rc.section_dir(script_file))


# ---------------------------------------------------------------- bar layout (Ravi, 13 Oct: POD and POFD only)
def counts(feat, g):
    """Share of all scored test cells that falls in each class, written under its label."""
    return [f'({100 * v:.0f}%)' if v >= 0.01 else f'({100 * v:.1f}%)' for v in g.share_cells]


def draw_bars(features, script_file, height=3.9):
    """One column per characteristic, rows POD and POFD; grouped bars of U-Net-T0 and ViT-T0 (seed means)."""
    rc.style()
    d = pd.read_csv(SCREEN)
    d = d[d['class'] != 'no value']
    n = len(features)
    widths = [len(d[d.feature == f]) + 1 for f, _ in features]
    fig, axes = plt.subplots(2, n, figsize=(ps.TEXT_WIDTH_IN, height), sharey='row', squeeze=False,
                             gridspec_kw={'width_ratios': widths})
    for j, (feat, title) in enumerate(features):
        g = d[d.feature == feat].reset_index(drop=True)
        x = np.arange(len(g))
        labs = RELABEL.get(feat, [label(c) for c in g['class']])
        ticks = [f'{a}\n{b}' for a, b in zip(labs, counts(feat, g))]
        # tilt the class labels when the widest one is wider than its bar group (6.5 pt Lato, about 0.55 em per character)
        slot_in = 0.85 * ps.TEXT_WIDTH_IN * widths[j] / sum(widths) / len(g)
        label_in = max(len(t) for tk in ticks for t in tk.split('\n')) * 0.55 * (ps.FONT_SIZE - 2.5) / 72
        tilt = label_in > 0.9 * slot_in
        if tilt:                                                  # one tilted line per class, e.g. '10²–10³ (18%)'
            ticks = [f'{a} {b}' for a, b in zip(labs, counts(feat, g))]
        for i, (m, lab) in enumerate((('pod', 'POD'), ('pofd', 'POFD'))):
            ax = axes[i, j]
            for f, dx in (('unet', -0.19), ('vit', 0.19)):
                ax.bar(x + dx, g[f'{f}_{m}'], 0.36, color=ps.alpha_on_white(rc.COLOR[f], BAR_FILL),
                       edgecolor=rc.COLOR[f], lw=0.5, zorder=2)
            ax.set_xticks(x, ticks if i == 1 else [''] * len(x), fontsize=ps.FONT_SIZE - 2.5,
                          rotation=45 if tilt else 0, ha='right' if tilt else 'center', rotation_mode='anchor')
            ax.tick_params(axis='x', length=0, pad=1.5)
            ax.set_xlim(-0.6, len(g) - 0.4)
            rc.ygrid(ax)
            if j == 0:
                ax.set_ylabel(lab)
        lines = max(t.count('\n') for _, t in features)          # equal line count keeps the titles aligned
        rc.panel_title(axes[0, j], f'({"abcdefgh"[j]}) {title}' + '\n' * (lines - title.count('\n')))
    for i in range(2):
        axes[i, 0].set_ylim(0, None)
    h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], BAR_FILL), edgecolor=rc.COLOR[f], lw=0.5, label=rc.LABEL[f])
         for f in rc.FAMILIES]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handlelength=1.0, columnspacing=1.6)
    rc.yticks_everywhere(axes)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    ps.save(fig, script_file, subdir=rc.section_dir(script_file))
