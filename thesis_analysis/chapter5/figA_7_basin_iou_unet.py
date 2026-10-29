"""Appendix Figure A.7: the basin map of Figure 5.14 for U-Net-T0 (Ravi, 25 Oct 2026: U-Net-T0 map in the appendix,
justified in Section 5.2.1). Runs fig_basin_iou.py with the U-Net-T0 basin IoU.
    python figA_7_basin_iou_unet.py
"""
import runpy, sys
from pathlib import Path

here = Path(__file__).resolve().parent
sys.argv = [str(here / 'fig_basin_iou.py'), str(here.parent / 'png' / 'appendix'), 'unet_t0', 'figA_7_basin_iou_unet.png']
runpy.run_path(str(here / 'fig_basin_iou.py'), run_name='__main__')
