"""Event maps, two mapping products in one figure (event_maps.draw_rows; Ravi, 20 Oct 2026, as on pre-defence slide 22):
(a) EMSR416_AOI02_DEL_PRODUCT_20191215, Adour floodplain at Dax, Landes, France: from the AOI of the large test product
    with the largest IoU of ViT-T0 above U-Net-T0.
(b) EMSR788_AOI06_Redon_DEL_MONIT03_20250204, Vilaine valley at Redon, Brittany, France: the large test product with the
    second-largest IoU of U-Net-T0 above ViT-T0 (the largest, EMSR629 AOI03, belongs to the activation of the
    highest-IoU map, Figure 5.9). Large = at least 5,000 flood cells (32 km2) and 40 patches.
Maps: cells predicted by at least two of the three seeds; evaluation metrics under every map (IoU, POD, POFD).
Replaces fig5_10_map_emsr416_aoi02.py.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_6_map_dax_redon.py
"""
import event_maps as em

if __name__ == '__main__':
    em.draw_rows(['EMSR416_AOI02_DEL_PRODUCT_20191215', 'EMSR788_AOI06_Redon_DEL_MONIT03_20250204'],
                 (None, ('unet', 't0'), ('vit', 't0')), ('Flood label', 'U-Net-T0', 'ViT-T0'), __file__, subdir='5_1',
                 row_labels=('(a)', '(b)'))
