"""Appendix — soil of the 80 m cell: U-Net-T0 and ViT-T0 by class: POD (top) and POFD (bottom), grouped bars, seed means; share of the
scored cells under the labels (cond_common.draw_bars).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_3_cell_soil.py
"""
import cond_common as cc

FEATURES = [('clay_pct', 'Clay (%)'),
    ('sand_pct', 'Sand (%)')]

if __name__ == '__main__':
    cc.draw_bars(FEATURES, __file__)
