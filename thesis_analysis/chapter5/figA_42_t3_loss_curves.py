"""Figure A.42 — loss curves of U-Net-T3 and ViT-T3, seeds 1-3 (redrawn 21 Oct 2026, Ravi: bigger, one colour per seed). Rows:
training loss, validation loss, validation IoU; columns U-Net and ViT. Encoding in loss_common.py; data
plot/data/training/ (ch5_loss_curves_data.py).

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_42_t3_loss_curves.py
"""
import loss_common as lc

if __name__ == '__main__':
    lc.network_figure(__file__, 't3', 'T3')
