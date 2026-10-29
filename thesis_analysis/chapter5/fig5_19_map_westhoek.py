"""Event map, one mapping product per figure (event_maps.draw_product; Ravi, 15 Oct 2026): EMSR708_AOI01_Westhoek_DEL_MONIT02_20231112.
RQ3: the large test product with the largest gain of IoU from T0 to T30, mean of the two families. Large = at least 5,000 flood cells (32 km2) and 40 patches; rules fixed before drawing (Section 5.1.2).
Maps: cells predicted by at least two of the three seeds; evaluation metrics under every map (IoU, POD, POFD).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_19_map_westhoek.py
"""
import event_maps as em

if __name__ == '__main__':
    em.draw_product('EMSR708_AOI01_Westhoek_DEL_MONIT02_20231112', (None, ('unet', 't0'), ('unet', 't30'), ('vit', 't0'), ('vit', 't30')), ('Flood label', 'U-Net-T0', 'U-Net-T30', 'ViT-T0', 'ViT-T30'), __file__, subdir='5_3')
