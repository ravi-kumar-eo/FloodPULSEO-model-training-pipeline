"""RQ3 — Hydroclimatic input ablations: micro IoU, POD and POFD of every configuration of seed 1, per family (the
metrics of Figure 5.1; Ravi, 19 Oct 2026: replaces IoU, F1 and AUPRC and the former figure of false alarm ratio, miss
rate and area bias). Data: ablations.csv (ch5_rq3_ablation_data.py); POFD over the dry scored test cells.
Configurations (Section 4.4.3), seed 1: T0 (without dynamic hydroclimatic variables, hatched); T30 (whisker: range
of the three T30 seeds); T30 local (T30 without upstream precipitation and catchment area); T30 daily (without
the summary statistics); T30 statistics (the summary statistics alone, as a single day); T30 random init (T30 trained
from random initialisation). Values on the bars; 95% cluster intervals of the differences from T30 in the text.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_48_t30_ablations.py
"""
import rq1_common as rc
import rq3_common as r3

if __name__ == '__main__':
    rc.style()
    r3.ablation_bars((('iou', '(a) IoU', '{:.3f}'), ('pod', '(b) POD', '{:.3f}'), ('pofd', '(c) POFD', '{:.3f}')),
                     __file__)
