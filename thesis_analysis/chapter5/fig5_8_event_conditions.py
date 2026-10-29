"""RQ1 — Characteristics of the mapping product (event level): U-Net-T0 and ViT-T0 by class: POD (top) and POFD (bottom), grouped
bars, seed means; class sizes under the labels (cond_common.draw_bars).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_8_event_conditions.py
"""
import cond_common as cc

FEATURES = [('event_flooded_area', 'Flooded area (km²)'),
    ('event_climate', 'Climate class'),
    ('event_resolution', 'Resolution of the\nmapping imagery')]

if __name__ == '__main__':
    cc.draw_bars(FEATURES, __file__)
