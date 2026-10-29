"""RQ3 — Cell-level changes by height above nearest drainage (HAND): where the T30 networks remove and add false detections
and detected flood cells relative to the T0 network of the same family and seed.

Classes of the HAND of the 80 m cell (cells_test.npz, as in Figure A.1): <1, 1-2, 2-5, 5-10, 10-20, 20-50, >=50 m.
(a) Share of the false detections of T0 that lie in each class (%).
(b) Net change of the false detections from T0 to T30, (added - removed) / false detections of T0 in the class (%).
(c) Net change of the detected flood cells, (recovered - lost) / flood cells detected by T0 in the class (%).
Counts are means over seeds 1-3; scored test cells with a valid HAND. Above 20 m, fewer than 0.1% of the cells are
flood, so the changes in (c) there rest on fewer than 200 detected flood cells.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_18_t30_hand.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import rq1_common as rc
from rq1_common import ps

C = Path(__file__).resolve().parents[1] / 'data' / 'conditions'
EDGES = [0, 1, 2, 5, 10, 20, 50, np.inf]
LABELS = ['<1', '1–2', '2–5', '5–10', '10–20', '20–50', '≥50']
FILL = 0.8


def counts():
    L = np.load(C / 'cells_test.npz')
    lab, hand = L['label'].ravel(), L['hand_m'].ravel().astype(np.float32)
    B = np.load(C / 'preds_test_bits.npz')
    n = int(np.prod(B['shape']))
    keep = (lab >= 0) & np.isfinite(hand)
    y, cls = (lab == 1)[keep], np.digitize(hand[keep], EDGES[1:-1])
    pred = lambda k: np.unpackbits(B[k])[:n].astype(bool)[keep]
    out = {}
    for f in rc.FAMILIES:
        acc = []
        for s in rc.SEEDS:
            p0, p1 = pred(f'{f}_t0_s{s}'), pred(f'{f}_t30_s{s}')
            acc.append([[np.sum(m & p0 & ~y), np.sum(m & p0 & ~p1 & ~y), np.sum(m & ~p0 & p1 & ~y),
                         np.sum(m & p0 & y), np.sum(m & p0 & ~p1 & y), np.sum(m & ~p0 & p1 & y)]
                        for m in (cls == c for c in range(len(LABELS)))])
        a = np.mean(acc, 0)
        out[f] = dict(fp_share=100 * a[:, 0] / a[:, 0].sum(), fp_net=100 * (a[:, 2] - a[:, 1]) / a[:, 0],
                      tp_net=100 * (a[:, 5] - a[:, 4]) / a[:, 3])
    return out


def main():
    rc.style()
    ch = counts()
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.7))
    x = np.arange(len(LABELS))
    titles = (('fp_share', '(a) Share of the false\ndetections of T0 (%)'), ('fp_net', '(b) Net change in\nfalse detections (%)'),
              ('tp_net', '(c) Net change in detected\nflood cells (%)'))
    for ax, (k, title) in zip(axes, titles):
        for f, dx in (('unet', -0.19), ('vit', 0.19)):
            ax.bar(x + dx, ch[f][k], 0.36, color=ps.alpha_on_white(rc.COLOR[f], FILL), edgecolor=rc.COLOR[f],
                   lw=0.6, zorder=2)
        ax.axhline(0, color='black', lw=0.6, zorder=1)
        ax.set_xticks(x, LABELS, fontsize=ps.FONT_SIZE - 2.5, rotation=30, ha='right', rotation_mode='anchor')
        ax.tick_params(axis='x', length=0, pad=1.5)
        ax.set_xlim(-0.6, len(LABELS) - 0.4)
        ax.set_xlabel('HAND (m)', fontsize=ps.FONT_SIZE - 1.5)
        rc.ygrid(ax)
        rc.panel_title(ax, title)
    h = [Patch(facecolor=ps.alpha_on_white(rc.COLOR[f], FILL), edgecolor=rc.COLOR[f], lw=0.6, label=rc.LABEL[f])
         for f in rc.FAMILIES]
    fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=2, frameon=False,
               fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    for f in rc.FAMILIES:
        print(rc.LABEL[f], {k: np.round(v, 1).tolist() for k, v in ch[f].items()})


if __name__ == '__main__':
    main()
