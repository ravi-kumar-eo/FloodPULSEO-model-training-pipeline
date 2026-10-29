"""THE single source of truth for every project path.

WHY THIS EXISTS. Before the 2026-09-07 reorganisation, sixteen modules each computed the
project root themselves with `Path(__file__).resolve().parent` or `.parents[1]`. Every one of
those encodes how deep the file sits in the tree, so MOVING A FILE SILENTLY CHANGED WHERE IT
LOOKED FOR DATA — no import error, just a wrong directory. Import from here instead; then a
file can move anywhere and its paths still resolve.
"""
from pathlib import Path
import os

SRC = Path(__file__).resolve().parents[1]          # .../experiments/src
ROOT = SRC.parent                                  # .../experiments
METADATA = Path(os.environ.get('FLOOD_METADATA_DIR') or ROOT / 'metadata')
RUNS = ROOT / 'runs'
CONFIGS = SRC / 'configs'
