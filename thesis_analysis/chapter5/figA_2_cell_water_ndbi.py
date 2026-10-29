"""Appendix — water proximity and built-up index of the 80 m cell: U-Net-T0 and ViT-T0 by class: POD (top) and POFD (bottom), grouped bars, seed means; share of the
scored cells under the labels (cond_common.draw_bars).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_2_cell_water_ndbi.py
"""
import cond_common as cc

FEATURES = [('dist_pw_cells', 'Distance to permanent\nwater (80 m cells)'),
    ('ndbi', 'NDBI')]

if __name__ == '__main__':
    cc.draw_bars(FEATURES, __file__)
