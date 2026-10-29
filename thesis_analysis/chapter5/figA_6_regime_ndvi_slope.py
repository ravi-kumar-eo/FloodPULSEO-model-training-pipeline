"""Appendix — Basin-seen against basin-unseen test ground within the classes of NDVI of the 80 m cell and mean slope
of the patch (rq2_features.csv from ch5_regime_blocks.py); same layout as fig5_15_regime_soil.py.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_6_regime_ndvi_slope.py
"""
import regime_common as rg

if __name__ == '__main__':
    rg.feature_classes((('ndvi', 'NDVI', 'NDVI of the cell'),
                        ('patch_mean_slope', 'Mean slope of the patch', 'Mean slope of the patch (°)')),
                       __file__, height=5.4)
