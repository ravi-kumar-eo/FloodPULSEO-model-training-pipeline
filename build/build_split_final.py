#!/usr/bin/env python3
"""Build metadata/final/split/ — the split of the final programme (METHOD_PLAN_RERUN.md, D3–D5).

* The assignment of every product, AOI, ground unit and basin regime is the CURRENT split
  (`metadata/splits/patches.csv`), unchanged.
* Rows: patches that overlap their CEMS AOI (`aoi_fraction > 0`), used whole (D3).
* Flags: from the 32/64 class-map labels (`labels_80m_cls`, built by build_labels_80m.py):
  n_flood80 / n_pw80 / n_mixed80 cells per patch; has_flood80 = at least one flood cell (class 1).
* Single-observation selection (`exp1_keep`) recomputed on the retained rows and new labels: per
  AOI, the product with the highest share of flood cells (n_flood80 / (patches x 1024)); ties go to
  the first product in name order. The number of AOIs whose selected product changed is reported.

Categorical strings are preserved (`keep_default_na=False`: the training rows' basin_regime is the
literal 'n/a'). Fails (nonzero exit, no output) on duplicate keys, missing labels or lost units.

    python build_split_final.py --experiments <experiments dir> --catalogue <released_patches_metadata.csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['folder_name', 'patch_number']


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', required=True, type=Path)
    ap.add_argument('--catalogue', required=True, type=Path)
    a = ap.parse_args()
    md = a.experiments / 'metadata'
    src = md / 'splits' / 'patches.csv'
    lab = md / 'final' / 'labels_80m_cls'
    out = md / 'final' / 'split'
    if not (lab / 'COMPLETE').exists():
        sys.exit(f'{lab} is not COMPLETE — build the labels first')

    sp = pd.read_csv(src, keep_default_na=False, na_values=[''], low_memory=False)
    if sp.duplicated(KEY).any():
        sys.exit('duplicate (folder_name, patch_number) keys in the source split')
    cat = pd.read_csv(a.catalogue, usecols=KEY + ['aoi_fraction'], low_memory=False)
    m = sp.merge(cat, on=KEY, how='left', validate='one_to_one')
    if m.aoi_fraction.isna().any():
        sys.exit(f'{int(m.aoi_fraction.isna().sum())} rows without aoi_fraction')

    # per-patch class counts from the new labels
    counts = []
    for fn, g in m.groupby('folder_name'):
        p = lab / f'{fn}.npz'
        if not p.exists():
            sys.exit(f'missing label file {p}')
        z = np.load(p)
        for pn in g.patch_number:
            k = str(int(pn))
            if k not in z.files:
                sys.exit(f'missing label {fn} patch {k}')
            b = np.bincount(z[k].ravel(), minlength=4)
            counts.append((fn, int(pn), int(b[1]), int(b[2]), int(b[3])))
    c = pd.DataFrame(counts, columns=KEY + ['n_flood80', 'n_pw80', 'n_mixed80'])
    m = m.merge(c, on=KEY, how='left', validate='one_to_one')
    m['has_flood80'] = (m.n_flood80 > 0).astype(int)

    keep = m[m.aoi_fraction > 0].copy()
    if (m.has_flood80.astype(bool) & (m.aoi_fraction == 0)).any():
        n = int((m.has_flood80.astype(bool) & (m.aoi_fraction == 0)).sum())
        print(f'NOTE: {n} flood-bearing patches lie fully outside the AOI and are dropped')

    # single-observation selection on retained rows
    old_keep = keep.exp1_keep.astype(str).str.lower().isin(('true', '1'))
    ev = keep.groupby('folder_name').agg(n=('patch_number', 'size'), f=('n_flood80', 'sum'))
    ev['frac'] = ev.f / (ev.n * 1024)
    ev['aoi'] = ['_'.join(i.split('_')[:2]) for i in ev.index]
    ev = ev.sort_index()
    chosen = set(ev.groupby('aoi').frac.idxmax())
    keep['exp1_keep'] = keep.folder_name.isin(chosen)
    old_chosen = set(keep.loc[old_keep, 'folder_name'])
    changed_aois = len({'_'.join(f.split('_')[:2]) for f in chosen ^ old_chosen})

    # integrity: nothing lost at product / AOI / ground-unit level
    for col in ('folder_name', 'aoi', 'ground_unit'):
        if keep[col].nunique() != sp[col].nunique():
            sys.exit(f'{col}: {sp[col].nunique()} before, {keep[col].nunique()} after — units lost')

    out.mkdir(parents=True, exist_ok=True)
    keep.drop(columns=[]).to_csv(out / 'patches.csv', index=False)
    keep[KEY + ['n_flood80', 'n_pw80', 'n_mixed80', 'has_flood80']].to_csv(out / 'flood80_flags.csv', index=False)
    check = pd.read_csv(out / 'patches.csv', keep_default_na=False, na_values=[''], low_memory=False)
    assert (check.loc[check.cohort == 'train', 'basin_regime'] == 'n/a').all(), 'n/a regime lost'

    summary = {'rule_rows': 'aoi_fraction > 0', 'rule_labels': '32/64 class map (0 dry, 1 flood, 2 pw, 3 mixed)',
               'source_split_md5': md5(src), 'labels_manifest_md5': md5(lab / 'manifest.csv'),
               'patches_md5': md5(out / 'patches.csv'), 'flags_md5': md5(out / 'flood80_flags.csv'),
               'aois_with_changed_single_observation_product': changed_aois, 'cohorts': {}}
    for co, g in keep.groupby('cohort'):
        s1 = g[g.exp1_keep]
        summary['cohorts'][co] = {
            'patches_full': int(len(g)), 'patches_single_obs': int(len(s1)),
            'flood_bearing_full': int(g.has_flood80.sum()), 'flood_bearing_single_obs': int(s1.has_flood80.sum()),
            'steps_per_epoch_full': int(g.has_flood80.sum() // 16), 'steps_per_epoch_single_obs': int(s1.has_flood80.sum() // 16),
            'flood_cells_share_full': float(g.n_flood80.sum() / (len(g) * 1024)),
            'products': int(g.folder_name.nunique()), 'aois': int(g.aoi.nunique()), 'ground_units': int(g.ground_unit.nunique())}
    (out / 'manifest_final.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    (out / 'COMPLETE').write_text('ok\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
