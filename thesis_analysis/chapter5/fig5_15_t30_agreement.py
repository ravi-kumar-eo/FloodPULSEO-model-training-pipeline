"""RQ3 — IoU, POD and POFD without (T0) and with (T30) the dynamic hydroclimatic variables, U-Net and ViT, micro and
macro, mean of seeds 1-3 (the metrics of Figure 5.1; Ravi, 19 Oct 2026: this figure replaces IoU, F1 and AUPRC and the
former figure of false alarm ratio, miss rate and area bias).
Test subset, full rows, scored cells, each fit at its validation-selected global threshold. Data: data/metrics/fits.csv
(ch5_pod_pofd_data.py). Bars: the mean of seeds 1-3 per family and configuration, value on top, T0 hatched; the values
of every seed are in Appendix Table A.2.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_15_t30_agreement.py
"""
import metrics_common as mc
import rq1_common as rc
import rq4_common as r4
from rq1_common import ps


def main():
    rc.style()
    fig, _ = r4.length_bars(mc.three_panels(letters=False), windows=('t0', 't30'), height=5.0)
    ps.save(fig, __file__, subdir=rc.section_dir(__file__))


if __name__ == '__main__':
    main()
