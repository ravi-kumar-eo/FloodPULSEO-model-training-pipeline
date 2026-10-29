"""RQ1 — Characteristics of the 80 m cell (cell level): U-Net-T0 and ViT-T0 by class: POD (top) and POFD (bottom), grouped
bars, seed means; class sizes under the labels (cond_common.draw_bars).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_10_cell_conditions.py
"""
import cond_common as cc

FEATURES = [('hand_m', 'HAND (m)'),
    ('ndvi', 'NDVI')]

if __name__ == '__main__':
    cc.draw_bars(FEATURES, __file__)
