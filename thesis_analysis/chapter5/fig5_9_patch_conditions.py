"""RQ1 — Characteristics of the 2.56 km patch (patch level): U-Net-T0 and ViT-T0 by class: POD (top) and POFD (bottom), grouped
bars, seed means; class sizes under the labels (cond_common.draw_bars).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_9_patch_conditions.py
"""
import cond_common as cc

FEATURES = [('patch_mean_slope', 'Mean slope (°)'),
    ('patch_max_uda', 'Maximum upstream\ndrainage area (km²)'),
    ('patch_pw_share', 'Permanent-water\nfraction (%)')]

if __name__ == '__main__':
    cc.draw_bars(FEATURES, __file__)
