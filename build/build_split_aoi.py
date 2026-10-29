#!/usr/bin/env python3
"""Build metadata/splits_aoi/ — the thesis split restricted to patches that overlap a CEMS AOI.

WHY (Ravi, 2026-10-04). CEMS maps flood only inside its area of interest (AOI). Patches tile the
whole rectangle around it, so label 0 outside the AOI means "never mapped", not "dry":
52.3% of patch area lies outside the AOIs and 45.8% of patches lie entirely outside
(readme/01_data/DATASET.md, "Label 0 outside the AOI"). Training taught those cells as dry and
evaluation counted any prediction there as a false alarm.

WHAT. The assignment of every mapping product, AOI, ground unit and basin regime is unchanged:
this only DROPS the patches with aoi_fraction == 0. Patches that are partly inside keep their
outside cells, which need a cell mask in the 80 m labels (separate step).

    python build/build_split_aoi.py            # writes metadata/splits_aoi/, never touches splits/
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[2]                      # experiments/
SRC_SPLIT = EXP / 'metadata' / 'splits'
OUT = EXP / 'metadata' / 'splits_aoi'
CATALOGUE = EXP.parent / 'pipeline' / 'data' / 'metadata' / 'global' / 'released_patches_metadata.csv'
KEY = ['folder_name', 'patch_number']


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f'{OUT} exists and is not empty — refusing to overwrite')
    cat = pd.read_csv(CATALOGUE, low_memory=False, usecols=KEY + ['aoi_fraction'])
    sp = pd.read_csv(SRC_SPLIT / 'patches.csv', low_memory=False)
    ff = pd.read_csv(SRC_SPLIT / 'flood80_flags.csv')
    m = sp.merge(cat, on=KEY, how='left', validate='one_to_one')
    if m.aoi_fraction.isna().any():
        sys.exit(f'{int(m.aoi_fraction.isna().sum())} split rows have no aoi_fraction')
    keep = m.aoi_fraction > 0
    flood = sp[KEY].merge(ff, on=KEY, how='left').has_flood80.fillna(0).astype(bool)
    if (flood & ~keep).any():
        sys.exit('a flood-bearing patch lies fully outside its AOI — inspect before dropping')

    OUT.mkdir(parents=True, exist_ok=True)
    out = m[keep].copy()
    out.to_csv(OUT / 'patches.csv', index=False)
    ff.merge(out[KEY], on=KEY, how='inner').to_csv(OUT / 'flood80_flags.csv', index=False)

    summary = {'source_split': str(SRC_SPLIT), 'source_patches_md5': md5(SRC_SPLIT / 'patches.csv'),
               'catalogue': str(CATALOGUE), 'rule': 'aoi_fraction > 0', 'cohorts': {}}
    for co, g in m.groupby('cohort'):
        k = g[g.aoi_fraction > 0]
        summary['cohorts'][co] = {
            'patches_before': int(len(g)), 'patches_after': int(len(k)),
            'fully_inside': int((k.aoi_fraction >= 0.9999).sum()),
            'partly_inside': int((k.aoi_fraction < 0.9999).sum()),
            'products_before': int(g.folder_name.nunique()), 'products_after': int(k.folder_name.nunique()),
        }
    summary['patches_md5'] = md5(OUT / 'patches.csv')
    (OUT / 'manifest_aoi.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['cohorts'], indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
