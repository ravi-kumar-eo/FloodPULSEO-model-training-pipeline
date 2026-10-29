"""Shared data and drawing of the RQ3 figures (Section 5.3): networks without (T0) and with a 30-day input history of
the dynamic hydroclimatic variables (T30), U-Net and ViT, seeds 1-3. Data: plot/data/rq3/ (ch5_rq3_data.py)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

import rq1_common as rc
from rq1_common import ps

D = Path(__file__).resolve().parents[1] / 'data' / 'rq3'
WINDOWS = ('t0', 't30')
WLABEL = {'t0': 'T0', 't30': 'T30'}


def fits():
    return pd.read_csv(D / 'fits.csv').set_index(['family', 'window', 'seed'])


def products():
    return pd.read_csv(D / 'products.csv')


def pair_grid(panels, ref_line=None, height=3.7):
    """Two rows (micro, macro) by one column per measure. In every panel each seed is one pair of markers, T0 open
    and T30 filled, joined by a line in the family colour; U-Net seeds 1-3, then ViT seeds 1-3, as in Figure 5.1.
    The y-axes do not start at zero and each panel has its own scale, so that the change from T0 to T30 is visible.
    panels: list of (title, value(row, family, window, seed)), row 0 micro, row 1 macro.
    ref_line: {column index: y} for a dashed reference line (area bias = 1)."""
    fig, axes = plt.subplots(2, len(panels), figsize=(ps.TEXT_WIDTH_IN, height))
    for c, (title, value) in enumerate(panels):
        for r in (0, 1):
            ax = axes[r, c]
            vals = []
            for f in rc.FAMILIES:
                for s in rc.SEEDS:
                    x = rc.POS[(f, s)]
                    v0, v1 = value(r, f, 't0', s), value(r, f, 't30', s)
                    vals += [v0, v1]
                    ax.plot([x, x], [v0, v1], color=rc.COLOR[f], lw=1.0, zorder=2)
                    ax.scatter(x, v0, s=22, facecolor='white', edgecolor=rc.COLOR[f], lw=1.0, zorder=3)
                    ax.scatter(x, v1, s=22, facecolor=rc.COLOR[f], edgecolor=rc.COLOR[f], lw=1.0, zorder=4)
            lo, hi = min(vals), max(vals)
            pad = 0.12 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            if ref_line and c in ref_line and lo - pad < ref_line[c] < hi + pad:
                ax.axhline(ref_line[c], color='black', lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.set_xticks([rc.POS[k] for k in rc.POS], [str(k[1]) for k in rc.POS])
            ax.tick_params(axis='x', length=0, pad=1.5, labelsize=ps.FONT_SIZE - 2.5)
            ax.set_xlim(-0.7, 6.4)
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            for f, x in (('unet', 1.0), ('vit', 4.7)):
                ax.text(x, -0.105, rc.LABEL[f], transform=tr, ha='center', va='top', fontsize=ps.FONT_SIZE - 1.5)
            rc.ygrid(ax)
        rc.panel_title(axes[0, c], title)
    for r, name in enumerate(('Micro', 'Macro')):
        axes[r, 0].set_ylabel(name)
    h = [Line2D([], [], ls='', marker='o', ms=4.5, mfc='white', mec='#555555', mew=1.0, label='T0'),
         Line2D([], [], ls='', marker='o', ms=4.5, mfc='#555555', mec='#555555', mew=1.0, label='T30')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False,
               fontsize=ps.FONT_SIZE - 1, handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.6, w_pad=1.0)
    return fig, axes


def report(panels):
    for title, val in panels:
        for g, grp in enumerate(('micro', 'macro')):
            print(f'{title:24s} {grp}: ' + ' | '.join(
                f'{rc.LABEL[f]}{s} {val(g, f, "t0", s):.4f}->{val(g, f, "t30", s):.4f}' for f in rc.FAMILIES
                for s in rc.SEEDS))


ABLATIONS = (('t0', 'T0'), ('t30', 'T30'), ('noup', 'T30 local'), ('nosum', 'T30 daily'),
             ('sumonly', 'T30 statistics'), ('scratch', 'T30 random init'))


def ablation_value(r, m):
    """A micro measure of one fit row of ablations.csv."""
    tp, fp, fn = r.tp, r.fp, r.fn
    if m in ('pod', 'pofd'):                         # Ravi, 19 Oct 2026: IoU, POD and POFD in the thesis
        import metrics_common as mc
        return tp / (tp + fn) if m == 'pod' else fp / mc.dry_total()
    return {'iou': tp / (tp + fp + fn), 'f1': 2 * tp / (2 * tp + fp + fn), 'auprc': r.auprc,
            'far': fp / (tp + fp), 'miss': fn / (tp + fn), 'bias': (tp + fp) / (tp + fn)}[m]


def ablation_bars(panels, script_file, ref_line=None, height=5.6):
    """One full-width panel per measure: for every configuration of seed 1, a U-Net and a ViT bar with its value on
    top. T0 hatched; T30 with a whisker over its three seeds. panels: (measure, title, format)."""
    from matplotlib.patches import Patch
    a = pd.read_csv(D / 'ablations.csv')
    fig, axes = plt.subplots(len(panels), 1, figsize=(ps.TEXT_WIDTH_IN, height))
    x = np.arange(len(ABLATIONS))
    for ax, (m, title, fmt) in zip(axes, panels):
        top = 0
        for f, dx in (('unet', -0.2), ('vit', 0.2)):
            q = a[a.family == f]
            seeds = [ablation_value(r, m) for _, r in q[q.config == 't30'].iterrows()]
            for xi, (k, _) in zip(x, ABLATIONS):
                v = ablation_value(q[(q.config == k) & (q.seed == 1)].iloc[0], m)
                t0 = k == 't0'
                ax.bar(xi + dx, v, 0.38, color='white' if t0 else ps.alpha_on_white(rc.COLOR[f], 0.8),
                       edgecolor=rc.COLOR[f], hatch='////' if t0 else None, lw=0.6, zorder=2)
                y = v
                if k == 't30':
                    ax.plot([xi + dx] * 2, [min(seeds), max(seeds)], color='black', lw=0.9, zorder=3)
                    for yy in (min(seeds), max(seeds)):
                        ax.plot([xi + dx - 0.06, xi + dx + 0.06], [yy, yy], color='black', lw=0.9, zorder=3)
                    y = max(v, max(seeds))
                s = fmt.format(v)
                ax.annotate(s[1:] if s.startswith('0.') else s, (xi + dx, y), xytext=(0, 2),
                            textcoords='offset points', ha='center', va='bottom', fontsize=ps.FONT_SIZE - 2.5,
                            zorder=4)
                top = max(top, y)
        if ref_line and m in ref_line:
            ax.axhline(ref_line[m], color='black', lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_ylim(0, top * 1.15)
        ax.set_xticks(x, [n for _, n in ABLATIONS], fontsize=ps.FONT_SIZE - 1)
        ax.tick_params(axis='x', length=0, pad=2)
        ax.set_xlim(-0.6, len(ABLATIONS) - 0.4)
        rc.ygrid(ax)
        rc.panel_title(ax, title)
    h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], 0.8), edgecolor=rc.COLOR[f], lw=0.6, label=rc.LABEL[f])
         for f in rc.FAMILIES]
    h += [Patch(facecolor='white', edgecolor='#555555', hatch='////', lw=0.6,
                label='T0, without dynamic hydroclimatic variables'),
          Line2D([], [], color='black', lw=0.9, label='Range of the three T30 seeds')]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False,
               fontsize=ps.FONT_SIZE - 1.5, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0)
    ps.save(fig, script_file, subdir=rc.section_dir(script_file))
    for f in rc.FAMILIES:
        q = a[(a.family == f) & (a.seed == 1)]
        print(rc.LABEL[f], {k: {m: round(ablation_value(q[q.config == k].iloc[0], m), 4) for m, _, _ in panels}
                            for k, _ in ABLATIONS})
