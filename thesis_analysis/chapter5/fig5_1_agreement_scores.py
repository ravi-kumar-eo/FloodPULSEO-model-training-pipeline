"""RQ1 — IoU, POD and POFD of U-Net-T0 and ViT-T0, micro and macro, every seed (Ravi, 17 and 19 Oct 2026: the thesis
reports IoU, POD and POFD; this figure replaces IoU, F1 and AUPRC and the former Figure 5.2 of false alarm ratio,
miss rate and area bias).
Test subset, full rows, scored cells, each fit at its validation-selected global threshold.
  IoU   TP / (TP + FP + FN)
  POD   TP / (TP + FN), the share of the flood cells of the label that are predicted
  POFD  FP / (FP + TN), the share of the dry cells of the label that are predicted as flood
Micro: counts summed over all scored test cells. Macro: the metric within each of the 394 flood-positive mapping
products, then the mean. Data: data/metrics/fits.csv (ch5_pod_pofd_data.py).
Top row micro, bottom row macro; seeds 1-3 under the bars of each family, values written above the bars.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_1_agreement_scores.py
"""
import metrics_common as mc
import rq1_common as rc
from rq1_common import ps


def main():
    rc.style()
    panels = tuple((f'({"abc"[i]}) {name}', '{:.3f}',
                    (lambda m: lambda g, f, s: mc.micro(f, 't0', s, m) if g == 0 else mc.macro(f, 't0', s, m))(m))
                   for i, (m, name) in enumerate((('iou', 'IoU'), ('pod', 'POD'), ('pofd', 'POFD'))))
    fig, _ = rc.metric_grid(panels, yfmt='%.2f')
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))
    for title, _, val in panels:
        print(f'{title:10s} ' + ' | '.join(
            f'{grp}: ' + ' '.join(f'{rc.LABEL[f]}{s} {val(g, f, s):.4f}' for f in rc.FAMILIES for s in rc.SEEDS)
            for g, grp in enumerate(('micro', 'macro'))))


if __name__ == '__main__':
    main()
