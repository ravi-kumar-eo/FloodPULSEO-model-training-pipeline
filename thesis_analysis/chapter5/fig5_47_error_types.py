"""Section 5.5 — Types of error of U-Net and ViT without (T0) and with (T30) the 30-day input history.

Each error cell of a fit is assigned to the first matching type, on the mosaic of its mapping product (test patches at
their positions on the patch grid; chessboard distance, one cell = 80 m):
(a) false detections: within 80 m (1 cell) of a reference flood cell; within 160-240 m (2-3 cells); next to a permanent-
    water or mixed cell; on low ground with a HAND below 2 m; elsewhere.
(b) missed detections: within 80 m of a predicted flood cell of the same fit; within 160-240 m; next to a permanent-
    water or mixed cell; under dense vegetation (NDVI of at least 0.6); elsewhere.
(c) false-detection cells by the size of their connected false-detection patch (8-connected, cells of 80 m).
Shares of the false detections or missed cells of each fit, mean of seeds 1-3; T0 hatched.
Data: data/uncertainty/errors.csv and components.csv (ch5_uncertainty_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_47_error_types.py
"""
import matplotlib.pyplot as plt
import numpy as np

import rq1_common as rc
import unc_common as uc
from rq1_common import ps

FP = (('edge', '≤80 m\nfrom flood'), ('near', '160–240 m\nfrom flood'), ('pw', 'Next to\nperm. water'),
      ('low', 'HAND\n<2 m'), ('other', 'Else-\nwhere'))
FN = (('edge', '≤80 m from\nprediction'), ('near', '160–240 m\nfrom pred.'), ('pw', 'Next to\nperm. water'),
      ('veg', 'NDVI\n≥0.6'), ('other', 'Else-\nwhere'))
SIZE = ('1', '2–9', '10–99', '100–\n999', '≥1,000')


def shares(df, types):
    out = {}
    for cfg in uc.CONFIGS:
        g = df[(df.family == cfg[0]) & (df.window == cfg[1])]
        s = g.pivot_table(index='seed', columns='type', values='cells')
        s = 100 * s.div(s.sum(axis=1), axis=0)
        out[cfg] = s[[t for t, _ in types]].mean().to_numpy()
    return out


def main():
    rc.style()
    e = uc.read('errors')
    comp = uc.read('components')
    fp = shares(e[e.kind == 'fp'], FP)
    fn = shares(e[e.kind == 'fn'], FN)
    cs = {}
    for cfg in uc.CONFIGS:
        g = comp[(comp.family == cfg[0]) & (comp.window == cfg[1])].pivot_table(index='seed', columns='size_min',
                                                                                 values='cells')
        cs[cfg] = (100 * g.div(g.sum(axis=1), axis=0)).mean().to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(ps.TEXT_WIDTH_IN, 6.6))
    uc.grouped_bars(axes[0], [l for _, l in FP], fp)
    uc.grouped_bars(axes[1], [l for _, l in FN], fn)
    uc.grouped_bars(axes[2], SIZE, cs)
    for ax, t in zip(axes, ('(a) False detections by type (% of the false detections)',
                            '(b) Missed detections by type (% of the missed detections)',
                            '(c) False detections by size of their connected patch, in cells of 80 m (% of the false detections)')):
        rc.panel_title(ax, t)
        ax.tick_params(axis='x', labelsize=ps.FONT_SIZE - 1.5)
    fig.legend(handles=uc.legend_handles(), loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False,
               fontsize=ps.FONT_SIZE - 1)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.0)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    for name, d, types in (('FP', fp, [t for t, _ in FP]), ('FN', fn, [t for t, _ in FN]), ('size', cs, SIZE)):
        for cfg in uc.CONFIGS:
            print(name, uc.CLABEL[cfg], dict(zip(types, np.round(d[cfg], 1))))


if __name__ == '__main__':
    main()
