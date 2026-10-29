"""RQ3 — Cells changed by the dynamic hydroclimatic variables, as a waterfall per family: the count of T0, the cells
that T30 takes away, the cells that T30 adds, and the count of T30. Every count is the mean over seeds 1-3, each T30
network compared cell by cell with the T0 network of the same family and seed.

(a) False detections (reference non-flood cells predicted as flood): T0, removed by T30, added by T30, T30.
(b) Detected flood cells (reference flood cells predicted as flood): T0, lost by T30 (detected by T0, missed by T30),
    recovered by T30 (missed by T0, detected by T30), T30.
Totals are solid bars with their value above, changes are hatched bars that float between the running totals with
their value inside; values in millions of cells.
Test subset, full rows, scored cells (label >= 0), each fit at its validation-selected global threshold; predictions
from preds_test_bits.npz and labels from cells_test.npz (ch5_cells_build.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_17_t30_cell_changes.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

import rq1_common as rc
from rq1_common import ps

C = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
FILL = 0.8                                 # fill strength of the total bars on white


def changes():
    """Per family, the seed means of the cell counts (millions)."""
    L = np.load(C / 'cells_test.npz')['label'].ravel()
    B = np.load(C / 'preds_test_bits.npz')
    n = int(np.prod(B['shape']))
    keep = L >= 0
    y = (L == 1)[keep]
    pred = lambda k: np.unpackbits(B[k])[:n].astype(bool)[keep]
    out = {}
    for f in rc.FAMILIES:
        acc = []
        for s in rc.SEEDS:
            p0, p1 = pred(f'{f}_t0_s{s}'), pred(f'{f}_t30_s{s}')
            acc.append([(p0 & ~y).sum(), (p0 & ~p1 & ~y).sum(), (~p0 & p1 & ~y).sum(), (p1 & ~y).sum(),
                        (p0 & y).sum(), (p0 & ~p1 & y).sum(), (~p0 & p1 & y).sum(), (p1 & y).sum()])
        m = np.mean(acc, 0) / 1e6
        out[f] = dict(fp=m[:4], tp=m[4:])
    return out


def waterfall(ax, x0, vals, color):
    """Bars at x0..x0+3: total, decrease (floating, hatched), increase (floating, hatched), total."""
    t0, dec, inc, t1 = vals
    face = ps.alpha_on_white(color, FILL)
    ax.bar(x0, t0, 0.72, color=face, edgecolor=color, lw=0.6, zorder=2)
    ax.bar(x0 + 1, dec, 0.72, bottom=t0 - dec, color='white', edgecolor=color, hatch='////', lw=0.6, zorder=2)
    ax.bar(x0 + 2, inc, 0.72, bottom=t0 - dec, color='white', edgecolor=color, hatch='\\\\\\\\', lw=0.6, zorder=2)
    ax.bar(x0 + 3, t1, 0.72, color=face, edgecolor=color, lw=0.6, zorder=2)
    for xa, xb, yv in ((x0, x0 + 1, t0), (x0 + 1, x0 + 2, t0 - dec), (x0 + 2, x0 + 3, t1)):   # running totals
        ax.plot([xa + 0.36, xb - 0.36], [yv, yv], color='#777777', lw=0.6, ls=(0, (2, 1.5)), zorder=3)
    lab = dict(fontsize=ps.FONT_SIZE - 2, ha='center', zorder=4)
    box = dict(boxstyle='square,pad=0.15', facecolor='white', edgecolor='none')
    ax.text(x0, t0, f'{t0:.2f}', va='bottom', **lab)                          # totals above their bars
    ax.text(x0 + 3, t1, f'{t1:.2f}', va='bottom', **lab)
    ax.text(x0 + 1, t0 - dec / 2, f'−{dec:.2f}', va='center', bbox=box, **lab)  # changes inside their bars
    ax.text(x0 + 2, t0 - dec + inc / 2, f'+{inc:.2f}', va='center', bbox=box, **lab)


def main():
    rc.style()
    ch = changes()
    fig, axes = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_IN, 4.5))
    panels = (('(a) False detections', 'fp', ('T0', 'Removed', 'Added', 'T30')),
              ('(b) Detected flood cells', 'tp', ('T0', 'Lost', 'Recovered', 'T30')))
    starts = {'unet': 0, 'vit': 5}
    for ax, (title, key, names) in zip(axes, panels):
        top = 0
        for f in rc.FAMILIES:
            waterfall(ax, starts[f], ch[f][key], rc.COLOR[f])
            top = max(top, ch[f][key][0], ch[f][key][3])
        xs = [starts[f] + i for f in rc.FAMILIES for i in range(4)]
        ax.set_xticks(xs, list(names) * 2, fontsize=ps.FONT_SIZE - 1.5)
        ax.tick_params(axis='x', length=0, pad=1.5)
        ax.set_xlim(-0.7, 8.7)
        ax.set_ylim(0, top * 1.15)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        for f in rc.FAMILIES:
            ax.text(starts[f] + 1.5, -0.13, rc.LABEL[f], transform=tr, ha='center', va='top',
                    fontsize=ps.FONT_SIZE - 1)
        ax.set_ylabel('Cells (millions)')
        rc.ygrid(ax)
        rc.panel_title(ax, title)
    fig.tight_layout(h_pad=1.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    for f in rc.FAMILIES:
        a, b = ch[f]['fp'], ch[f]['tp']
        print(f'{rc.LABEL[f]}: FP {a[0]:.3f} -{a[1]:.3f} ({100 * a[1] / a[0]:.1f}%) +{a[2]:.3f} ({100 * a[2] / a[0]:.1f}%) '
              f'= {a[3]:.3f} | TP {b[0]:.3f} -{b[1]:.3f} ({100 * b[1] / b[0]:.1f}%) +{b[2]:.3f} = {b[3]:.3f}')


if __name__ == '__main__':
    main()
