"""Event map, one mapping product per figure (event_maps.draw_product; Ravi, 15 Oct 2026): EMSR466_AOI01_DEL_PRODUCT_20200927.
RQ4: the large product of the timing class days -30 to -16 with the largest gain of IoU from T3 to T30. Large = at least 5,000 flood cells (32 km2) and 40 patches; rules fixed before drawing (Section 5.1.2).
Maps: cells predicted by at least two of the three seeds; evaluation metrics under every map (IoU, POD, POFD).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/fig5_29_map_emsr466_t3.py
"""
import event_maps as em

if __name__ == '__main__':
    em.draw_product('EMSR466_AOI01_DEL_PRODUCT_20200927', (None, ('unet', 't3'), ('unet', 't30'), ('vit', 't3'), ('vit', 't30')), ('Flood label', 'U-Net-T3', 'U-Net-T30', 'ViT-T3', 'ViT-T30'), __file__, subdir='5_4')
