"""Figure A.45 — loss curves of the four T30 ablations of Section 5.3.6, seed 1 (the ablations were trained with seed
1 only), redrawn 21 Oct 2026 (Ravi: bigger, curves easy to identify): one colour per ablation. Panels: U-Net above
ViT, training loss solid and validation loss dashed. Encoding in loss_common.py; data plot/data/training/.

    PYTHONDONTWRITEBYTECODE=1 python chapters/chapter5/plot/scripts/figA_45_ablation_loss_curves.py
"""
import loss_common as lc

ABLATIONS = (('noup', 'T30 local'), ('nosum', 'T30 daily'), ('sumonly', 'T30 statistics'),
             ('scratch', 'T30 random init'))


def main():
    lc.rows_figure(__file__, [(a, 1, f'{name}, seed 1') for a, name in ABLATIONS], True, 9.0)


if __name__ == '__main__':
    main()
