"""RQ1 — Distribution over the flood-positive mapping products: IoU, POD and POFD of U-Net-T0 and ViT-T0, one dot per
product, each product's score averaged over the three seeds of its family (Ravi, 19 Oct 2026: IoU, POD and POFD only).

Test subset, full rows, scored cells, each fit at its validation-selected global threshold. Per product and seed:
  IoU = TP / (TP + FP + FN), POD = TP / (TP + FN), POFD = FP / (dry scored cells of the product),
then the mean over seeds 1-3 (a seed-mean score, not an ensemble prediction), all 394 products.
Boxes: quartiles and median; whiskers to the furthest product within 1.5 IQR; every product is drawn as a dot.
Counts from runs_final/final_t0/seed*/<family>/plots/per_product_test.csv (rq1_common.product_counts).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_4_product_distributions.py
"""
import matplotlib.pyplot as plt
import numpy as np

import metrics_common as mc
import rq1_common as rc
from rq1_common import ps

PANELS = (('iou', '(a) IoU'), ('pod', '(b) POD'), ('pofd', '(c) POFD'))


def seed_means(c, f, col):
    w = c[c.family == f].pivot(index='folder_name', columns='seed', values=col)
    return w.dropna().mean(axis=1)


def main():
    rc.style()
    c = mc.add_pod_pofd(rc.product_counts())
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH_IN, 2.5))
    for ax, (col, title) in zip(axes, PANELS):
        for x, f in enumerate(rc.FAMILIES):
            v = seed_means(c, f, col).to_numpy()
            b = ax.boxplot(v, positions=[x], widths=0.5, whis=1.5, showfliers=False, patch_artist=True,
                           medianprops=dict(color='black', lw=1.0), zorder=2,
                           boxprops=dict(facecolor=ps.alpha_on_white(rc.COLOR[f], 0.18), edgecolor=rc.COLOR[f], lw=0.7),
                           whiskerprops=dict(color=rc.COLOR[f], lw=0.7), capprops=dict(color=rc.COLOR[f], lw=0.7))
            ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=3.0, color=rc.COLOR[f], alpha=0.7, lw=0, zorder=3)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            print(f'{title:22s} {rc.LABEL[f]:5s} n={len(v)}  q1 {q1:.3f}  median {med:.3f}  q3 {q3:.3f}  mean {v.mean():.3f}')
        ax.set_xticks([0, 1], [rc.LABEL[f] for f in rc.FAMILIES])
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.02, 1.02)
        rc.ygrid(ax)
        rc.panel_title(ax, title)
    fig.tight_layout(w_pad=1.2)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
