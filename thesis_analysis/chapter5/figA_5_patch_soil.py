"""Appendix — soil of the patch: U-Net-T0 and ViT-T0 by class: POD (top) and POFD (bottom), grouped bars, seed means; share of the
scored cells under the labels (cond_common.draw_bars).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_5_patch_soil.py
"""
import cond_common as cc

FEATURES = [('patch_mean_clay', 'Mean clay (%)'),
    ('patch_mean_sand', 'Mean sand (%)')]

if __name__ == '__main__':
    cc.draw_bars(FEATURES, __file__)
