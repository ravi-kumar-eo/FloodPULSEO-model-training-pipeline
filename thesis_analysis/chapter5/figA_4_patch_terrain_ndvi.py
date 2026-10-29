"""Appendix — terrain and vegetation of the patch: U-Net-T0 and ViT-T0 by class: POD (top) and POFD (bottom), grouped bars, seed means; share of the
scored cells under the labels (cond_common.draw_bars).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_4_patch_terrain_ndvi.py
"""
import cond_common as cc

FEATURES = [('patch_lowland_share', 'Cells with HAND\nbelow 2 m (%)'),
    ('patch_median_ndvi', 'Median NDVI'),
    ('patch_median_elevation', 'Median\nelevation (m)')]

if __name__ == '__main__':
    cc.draw_bars(FEATURES, __file__)
