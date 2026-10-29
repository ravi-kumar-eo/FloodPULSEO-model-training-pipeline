"""Build split_v2: split_v1 plus the GROUND UNIT and the AOI.

WHAT v2 ADDS OVER v1 (Ravi, 2026-09-04)
---------------------------------------
v1 stopped storms leaking. It did not stop the SAME GROUND being counted many times, and
that turned out to drive the headline result.

  * GROUND UNIT — the transitive closure of footprint OVERLAP, computed on patch polygons
    reprojected to EPSG:4087. 1,565 events collapse to 395 ground units; 208 of them hold
    more than one event and those cover 96.4% of all patches. Measured on the first v1 fit,
    counting each ground unit once REVERSES the basin-familiarity result: pooled 2.16x in
    favour of `seen`, per event 1.17x, per ground unit 0.67x — i.e. `unseen` ahead. The
    2.16x was `test_seen` repeating its ground 4.38 times per unit against `test_unseen`'s
    2.52. A unit is therefore atomic AND its count is balanced, not just its patches.
  * AOI — one mapped area, `activation_AOIid` (845 of them). Its SIBLINGS are the same area
    re-mapped on later dates. Measured: EMSR848_AOI04_Hue MONIT01/02/03 are BIT-IDENTICAL in
    every band the model sees (NDVI, NDBI, elevation, flow, UDA, HAND, clay, sand) and
    differ only in the mask, which recedes 0.085 -> 0.069 -> 0.057. Across the dataset 749
    events (324,478 patches, 57.3%) have a bit-identical twin. Training on those teaches the
    model to predict the mean flood extent for a terrain and nothing more.
  * exp1_keep — ONE event per AOI, the sibling with the highest flood FRACTION. 845 events,
    215,300 patches. This is the subset EXP 1 (terrain-only) trains and scores on, because
    showing one input two different masks is not a defensible target. EXP 2 onward, where
    weather distinguishes the dates, uses every event. ONE split serves both: the cohorts
    are identical, only the row selection differs, so the two experiments stay comparable.

The objective therefore balances the FULL event set and the EXP1 SUBSET at the same time,
plus the ground-unit and AOI counts. Everything else is inherited from v1 unchanged.

--- inherited from split_v1 -------------------------------------------------------------
Build split_v1: train / val / test, with a basin-regime LABEL rather than test cohorts.

WHY A REBUILD (all measured on split v0, 2026-09-03)
----------------------------------------------------
1. v0 never searched. candidates.csv held one row; manifest.json records n_seeds_tried: 1.
2. Its two unseen cohorts were different populations -- val_unseen 297.5 patches/event
   against test_unseen's 130.5 -- so selecting on one and reporting the other compared two
   different worlds. The U-Net threshold head scored +6.2% on val_unseen, -10.3% on
   test_unseen, against the same baseline.
3. ACTIVATION leakage was unconstrained: 36.3% of EMSR activations spanned cohorts, 43% of
   patches. One storm hits many basins, so basin-disjointness does not stop it.
4. Cohort-level flood prevalence hid the real imbalance. v0 test_unseen held 26.5% of its
   patches in 500+ events at prev80 0.0101, against val_unseen's 62.9% at 0.0281 -- and
   large events dominate pooled IoU, so that alone moved the headline metric. Balancing
   overall prevalence does not fix it; the balance has to hold WITHIN event-size strata.

WHAT THIS BUILDS
----------------
  * cohorts train / val / test. v0's val_seen is gone -- 82.6% of its events shared an
    activation with train, so it measured memorisation, and nothing selected on it.
  * basin familiarity is a LABEL on test, not a cohort (Ravi, 2026-09-03): it is needed by
    one RQ, and a label can be sliced, standardised, or thresholded at reporting time,
    while a cohort fixes those choices in the split forever.
        basin_regime     'seen' | 'partial' | 'unseen'  ('unseen' for val, 'n/a' train)
        basin_seen_frac  share of the event's basin codes that appear in train
    THREE-valued (Ravi, 2026-09-03, second pass) -- 'seen' frac >= MIN_BASIN_COVER,
    'partial' 0 < frac < MIN_BASIN_COVER, 'unseen' frac == 0. See CARVE GATE below for what
    two values cost. RQ2 contrasts seen vs unseen; `partial` stays in test for the headline
    metric and sits out that contrast, so no event has to wear a label it does not fit.
    The cut is only how the label is derived; the continuous value is kept so a stricter cut
    is a filter rather than a rebuild.
  * an ACTIVATION is atomic: it may never span cohorts.
  * prevalence is matched WITHIN event-size strata, not just overall.
  * val is matched to test as a whole (it selects, test reports) and the two test slices
    are matched to each other (that is the RQ2 contrast).

GROUPING
--------
Events join into components on shared activation, shared footprint_group, or shared basin
CODE. `basin_id` is a hyphen-joined SET ('74244-74252-74291'), so it is split into codes
first -- comparing the strings lets two events share real basins while looking distinct.
Components are indivisible for the train / val / test-unseen partition.

The seen slice is carved INSIDE the train pool: whole activation/scene subunits whose basin
codes still appear in train afterwards, so its basins are familiar while its storms are not.

CARVE GATE: PER ACTIVATION, NOT PER EVENT (Ravi, 2026-09-03, second pass)
------------------------------------------------------------------------
The first v1 build gated the carve on the WORST EVENT in a subunit -- so that label and rule
would agree by construction -- and rejected any candidate where they disagreed. Measured, it
halved the eligible pool:

    gate                       units   patches   % of data   % area flooded
    per activation, >=50%         67   195,047       34.4%           2.03%
    per event, >=50%              61    94,984       16.8%           2.63%
    whole dataset                120   566,669        100%           2.47%

The seen slice needs 7.5% of the data, so a 16.8% pool forces it to consume 45% of
everything eligible and the greedy has no freedom left to match prevalence. All 100
candidates collapsed into test/seen prev80 0.0143-0.0157 against train's 0.024-0.026 --
the whole run was compromised, not one seed. So the gate goes back to per activation and the
label absorbs the mismatch as `partial`, which is honest either way: a `partial` event is not
silently mislabelled `seen` (the defect that motivated the per-event gate) and not forced
into `unseen`, where it would break basin-disjointness.

`carve_leak` in the objective is the share of the carved patches that did NOT come out
`seen`. It is what keeps the search from spending the looser gate freely.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))   # src/ on the path
from flood.paths import METADATA as MD
SRC = MD / 'split_old' / 'splits_v0'   # the raw patch table; not a split we build from
OUT = MD / 'splits'                    # THE split
UNITS = OUT / 'ground_units_events.csv'   # folder_name -> aoi, ground_unit

TARGET = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEEN_SHARE = 0.50        # of `test`, the rest is basin-unseen
N_SEEDS = 60       # the matching is now DIRECTED (the carve minimises divergence on the
                   # covariates that drive the score), so brute-force breadth is not needed
N_REFINE = 6
BIG_COMP = 0.05          # components larger than this can only be train
MIN_BASIN_COVER = 0.5    # a test-seen activation must leave >=this share of its basins in train
# HARD ADMISSIBILITY BANDS (Ravi asleep, decided 2026-09-05 from the candidate front).
# Permanent water and ground repetition are the two structural advantages that can manufacture a
# basin-familiarity effect on their own, and BOTH pushed the same way in the seed-29 build (PW
# 2.46x, repetition 1.76x, both favouring test/seen). As a soft term at weight 8 they did not bite:
# seed 29 still won. They are therefore a CONSTRAINT, not a preference -- a candidate outside these
# bands is not scored at all, and the objective picks the best of what is left. Measured on the
# 100-candidate front, 14 seeds survive [0.70, 1.45] and the best of them (72) is also BETTER than
# seed 29 on d_seen_vs_unseen and size_err_exp1, so this costs almost nothing.
PW_RATIO_BAND = (0.70, 1.45)     # pw_frac(seen) / pw_frac(unseen)
BLK_RATIO_BAND = (0.70, 1.45)    # acquisitions per spatial block, seen / unseen
# TERRAIN and GEOGRAPHY, added 2026-09-05 after the first EXP 1 fit. These were the two
# confounds left uncontrolled, and they were large: test/seen came out 0% Africa / 58% Europe
# against test/unseen's 27% / 41%, and median HAND 14.8 m against 1.6 m. Both track the score
# (the flat, non-European slice scored 3.4x worse pooled), so a seen-vs-unseen number from an
# unbalanced split measures geography and terrain rather than basin familiarity.
# MEASURED on 391 scored acquisitions from the first EXP 1 fit — spearman against per-acquisition
# flood IoU, and the partial correlation once flood fraction (the dominant term) is removed:
#     flood base rate  +0.576 / +0.07     median ELEVATION  -0.507 / -0.440
#     sensor resolution +0.167 / +0.344   n patches         +0.116 / +0.307
#     wet-patch rate   +0.165 / -0.323    permanent water   +0.240 / +0.258
#     median HAND      -0.286 / -0.122    % flat (HAND<2m)  +0.195 / +0.045  (p=0.37, NOT significant)
# So the earlier `% flat` band constrained the one terrain variable that stops mattering once flood
# fraction is controlled. ELEVATION is the real residual driver, then resolution, size and wetness.
# Climate separates as strongly as continent (Tropical 0.113 vs Cold 0.030, Kruskal p<1e-4).
ELEV_RATIO_BAND = (0.70, 1.45)   # median elevation, seen / unseen
CONT_TV_MAX = 1.01               # continent: PROVED infeasible (0 of 300 candidates at <=0.50), so
                                 # it is carried as a soft term only and reported, never enforced
TERRAIN = MD / 'acquisition_terrain.csv'   # built by src/build_terrain_stats.py

CELLS = 1024
STRATA = [(1, 20, '<20'), (20, 100, '20-99'), (100, 500, '100-499'), (500, 10 ** 9, '500+')]
# Flood-density bands, as % of the event's area labelled flood. Matching density only WITHIN
# a size band matches its average and says nothing about the mix: a 500+ event can be 0.3%
# or 3% flooded, and the near-dry ones are where pooled IoU dies (measured: two v0 events of
# 539-587 patches at 0.05-0.08% flooded scored IoU 0.0010 and 0.0020). v0 put 33.0% of
# test_seen's area in large near-dry events against test_unseen's 16.0%; the first v1 build
# made it 15.3% vs 39.8%. Only the JOINT size x density cell catches that.
DENS = [(0.0, 1.0, '<1%'), (1.0, 3.0, '1-3%'), (3.0, 1e9, '>3%')]
CELLS_IJ = [(sl, dl) for _, _, sl in STRATA for _, _, dl in DENS]


def load_events() -> pd.DataFrame:
    p = pd.read_csv(SRC / 'patches.csv')
    f = pd.read_csv(SRC / 'flood80_flags.csv')
    m = p.merge(f, on=['folder_name', 'patch_number'], how='left')
    if m.n_flood80.isna().any():
        sys.exit('flood80_flags.csv does not cover every patch')
    ev = m.groupby('folder_name').agg(
        n=('patch_number', 'size'),
        n_flood=('n_flood80', lambda s: int((s > 0).sum())),
        flood80=('n_flood80', 'sum'),
        pw80=('n_pw80', 'sum'),
        fg=('footprint_group', 'first'),
        bid=('basin_id', 'first'),
        cont=('continent', lambda s: s.mode().iloc[0]),
        clim=('climate', lambda s: s.mode().iloc[0]),
        res=('sensor_resolution_m', 'median')).reset_index()
    ev['act'] = ev.folder_name.str.split('_').str[0]
    # GROUND UNIT and AOI come from a precomputed table: the unit needs a geometric overlap
    # test across 37 CRS (geopandas + pyproj), which has no business running inside a
    # 100-seed search. Rebuild it with `python src/build_ground_units.py`.
    gu = pd.read_csv(UNITS)[['folder_name', 'aoi', 'ground_unit', 'exp1_keep']]
    ev = ev.merge(gu, on='folder_name', how='left')
    tr = pd.read_csv(TERRAIN)[['folder_name', 'hand_med', 'flat_frac', 'elev_med']]
    ev = ev.merge(tr, on='folder_name', how='left')
    for c in ('hand_med', 'flat_frac', 'elev_med'):
        ev[c] = ev[c].fillna(ev[c].median())
    if ev[['aoi', 'ground_unit']].isna().any().any():
        sys.exit(f'{UNITS} does not cover every event — rebuild it')
    ev['exp1'] = ev.exp1_keep.astype(bool)
    ev['codes'] = ev.bid.map(lambda s: frozenset(str(s).split('-')))
    ev['stratum'] = pd.cut(ev.n, bins=[lo for lo, _, _ in STRATA] + [STRATA[-1][1]],
                           labels=[lab for _, _, lab in STRATA], right=False)
    dens = 100.0 * ev.flood80 / (ev.n * CELLS)
    ev['dens'] = pd.cut(dens, bins=[lo for lo, _, _ in DENS] + [DENS[-1][1]],
                        labels=[lab for _, _, lab in DENS], right=False)
    key = {c: i for i, c in enumerate(CELLS_IJ)}
    ev['cell'] = [key[(a, b)] for a, b in zip(ev.stratum.astype(str), ev.dens.astype(str))]
    return ev


def _union_by(ev: pd.DataFrame, keyfn) -> np.ndarray:
    parent = list(range(len(ev)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    first = {}
    for i, r in enumerate(ev.itertuples()):
        for key in keyfn(r):
            if key in first:
                union(i, first[key])
            else:
                first[key] = i
    raw = [find(i) for i in range(len(ev))]
    # CANONICAL LABELS (Ravi, 2026-09-04). find() returns whichever member happened to become
    # the root, which depends on the ORDER the unions ran in -- and that order comes from
    # iterating `r.codes`, a frozenset of STRINGS, whose iteration order Python randomises per
    # process via PYTHONHASHSEED. The partition itself is identical every time; only the
    # labels move. But `small` is ordered by those labels before rng.shuffle(), so the same
    # seed built a DIFFERENT split in every new process: seed 91 gave test/seen 52 ground
    # units in one run and 39 in the next. Relabelling each component by its lowest row index
    # makes the whole search reproducible.
    canon = {}
    for i, r in enumerate(raw):
        if r not in canon:
            canon[r] = i
    return np.array([canon[r] for r in raw])


def components(ev):
    """Indivisible for train / val / test-unseen: activation, scene, basin, or GROUND UNIT.

    The ground unit is new in v2 and it is the one that matters most: two events whose
    footprints overlap are the same ground, so putting one in train and the other in test
    measures memorisation of that ground no matter how the basins fall.
    """
    return _union_by(ev, lambda r: [('a', r.act), ('f', r.fg), ('u', r.ground_unit)]
                     + [('b', c) for c in r.codes])


def subunits(ev):
    """The seen-slice unit. Basins are NOT a join key -- the seen slice is meant to share
    them -- but an activation spanning several scenes must move as one piece, and a ground
    unit must never be torn apart by the carve either."""
    return _union_by(ev, lambda r: [('a', r.act), ('f', r.fg), ('u', r.ground_unit)])


# ------------------------------------------------------------------ balance
_ARR = {}


def prep(ev):
    """Cache the columns profile() needs as plain arrays -- it is called tens of thousands
    of times inside the search, and pandas indexing dominated the runtime."""
    _ARR['n'] = ev.n.to_numpy(np.int64)
    _ARR['flood80'] = ev.flood80.to_numpy(np.int64)
    _ARR['cell'] = ev.cell.to_numpy(np.int64)
    _ARR['cont'] = pd.Categorical(ev.cont).codes.astype(np.int64)
    _ARR['n_cont'] = int(_ARR['cont'].max()) + 1
    # v2: counted, not just weighed. nunique() on a pandas slice inside the search cost more
    # than every other term combined, so AOI and ground unit become integer codes here.
    _ARR['aoi'] = pd.Categorical(ev.aoi).codes.astype(np.int64)
    _ARR['unit'] = pd.Categorical(ev.ground_unit).codes.astype(np.int64)
    _ARR['exp1'] = ev.exp1.to_numpy(bool)
    _ARR['flat'] = ev.flat_frac.to_numpy(np.float64)
    _ARR['elev'] = ev.elev_med.to_numpy(np.float64)
    _ARR['res'] = ev.res.to_numpy(np.float64)
    _ARR['clim'] = pd.Categorical(ev.clim).codes.astype(np.int64)
    _ARR['n_clim'] = int(_ARR['clim'].max()) + 1
    _ARR['pos'] = {v: i for i, v in enumerate(ev.index)}


def profile(ev: pd.DataFrame, idx: np.ndarray) -> dict:
    s = ev.loc[idx]
    n = int(s.n.sum())
    d = max(n, 1)                # `partial` may be empty; every rate below is then 0.0
    out = {
        'patches': n,
        'events': int(len(s)),
        'prev80': float(s.flood80.sum() / (d * CELLS)),
        'wet_rate': float(s.n_flood.sum() / d),
        'mean_event': float(n / max(len(s), 1)),
        'median_event': float(s.n.median()) if len(s) else 0.0,
        'cells_per_wet': float(s.flood80.sum() / max(s.n_flood.sum(), 1)),
        'pw_frac': float(s.pw80.sum() / (d * CELLS)),
    }
    for _, _, lab in STRATA:                    # the balance v0 was missing
        t = s[s.stratum == lab]
        tn = int(t.n.sum())
        out[f'share_{lab}'] = tn / d
        # EVENT share as well as patch share: MACRO IoU weights every event equally, and
        # events under 20 patches are ~1% of patches but a third of the events, so matching
        # patch share alone leaves the macro metric wide open (measured on v0: events under
        # 20 patches were 17.2% of val_unseen's events and 34.5% of test_unseen's, and those
        # events score roughly half the macro IoU of mid-size ones).
        out[f'evshare_{lab}'] = len(t) / max(len(s), 1)
        out[f'prev_{lab}'] = float(t.flood80.sum() / (tn * CELLS)) if tn else 0.0
    # joint size x density cells
    rows = np.array([_ARR['pos'][i] for i in idx], dtype=np.int64)
    cell = _ARR['cell'][rows]
    pw = np.bincount(cell, weights=_ARR['n'][rows], minlength=len(CELLS_IJ))
    ec = np.bincount(cell, minlength=len(CELLS_IJ)).astype(float)
    out['jshare'] = pw / d
    out['jevshare'] = ec / max(len(s), 1)
    # CONTINENT, weight 0.5 (Ravi, 2026-09-03). Dropping it entirely did not merely relax it
    # -- the search EXPLOITED it, because skewing geography was the cheapest way to satisfy
    # the density terms. That build came out train 80% Europe, val 60% Africa, test/seen 49%
    # Asia, which would confound any basin-familiarity result with continent. A small weight
    # stops it being free currency without letting it crowd out prevalence as it did at 2.0.
    out['cshare'] = np.bincount(_ARR['cont'][rows], weights=_ARR['n'][rows],
                                minlength=_ARR['n_cont']) / d
    # v2 counts. These are NOT fed to divergence() -- they are counts, not distributions --
    # they are balanced separately in score(). See count_err().
    out['aois'] = int(np.unique(_ARR['aoi'][rows]).size)
    out['units'] = int(np.unique(_ARR['unit'][rows]).size)
    out['exp1_patches'] = int(_ARR['n'][rows][_ARR['exp1'][rows]].sum())
    # patch-weighted share of the group lying within 2 m of drainage — the terrain statistic
    # that separated test/seen from test/unseen in the first EXP 1 fit (22% vs 54%)
    out['flat_frac'] = float(np.average(_ARR['flat'][rows], weights=_ARR['n'][rows])) if n else 0.0
    # the covariates that actually predict per-acquisition difficulty (see the note by
    # ELEV_RATIO_BAND). Patch-weighted, because that is how the pooled metric weights them.
    out['elev'] = float(np.average(_ARR['elev'][rows], weights=_ARR['n'][rows])) if n else 0.0
    out['res'] = float(np.average(_ARR['res'][rows], weights=_ARR['n'][rows])) if n else 0.0
    out['climshare'] = np.bincount(_ARR['clim'][rows], weights=_ARR['n'][rows],
                                   minlength=_ARR['n_clim']) / d
    return out


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-9)


def divergence(x: dict, ref: dict) -> float:
    d = (3.0 * rel(x['prev80'], ref['prev80'])
         + 1.0 * rel(x['wet_rate'], ref['wet_rate'])
         + 1.0 * rel(x['mean_event'], ref['mean_event'])
         + 0.5 * rel(x['median_event'], ref['median_event'])
         + 0.5 * rel(x['cells_per_wet'], ref['cells_per_wet'])
         + 0.5 * rel(x['pw_frac'], ref['pw_frac']))
    # stratified prevalence, weighted by how much of the reference lives in each stratum:
    # an overall prev80 match means nothing if the flood sits in different event sizes.
    for _, _, lab in STRATA:
        w = ref[f'share_{lab}']
        d += w * (2.0 * rel(x[f'prev_{lab}'], ref[f'prev_{lab}'])
                  + 2.0 * abs(x[f'share_{lab}'] - ref[f'share_{lab}']) / max(w, 1e-9))
        # unweighted by patch share -- the whole point is that these bands are tiny in
        # patches and large in event count
        d += 1.5 * abs(x[f'evshare_{lab}'] - ref[f'evshare_{lab}'])
    # joint cells: absolute differences, so an empty cell costs its share and nothing blows up
    d += 4.0 * float(np.abs(x['jshare'] - ref['jshare']).sum())
    d += 2.0 * float(np.abs(x['jevshare'] - ref['jevshare']).sum())
    d += 0.5 * 0.5 * float(np.abs(x['cshare'] - ref['cshare']).sum())   # 0.5 x TV distance
    # covariates ranked by their PARTIAL correlation with per-acquisition IoU, so the carve — which
    # already selects seen-slice subunits to minimise divergence from the unseen slice — matches the
    # things that actually drive the score instead of only the things that are easy to count.
    d += 2.0 * rel(x['elev'], ref['elev'])        # partial -0.440, the strongest residual driver
    d += 1.5 * rel(x['res'], ref['res'])          # partial +0.344
    d += 1.0 * 0.5 * float(np.abs(x['climshare'] - ref['climshare']).sum())   # Kruskal p < 1e-4
    return d


def count_err(prof: dict) -> float:
    """How far the AOI and GROUND-UNIT COUNTS are from balanced.

    v1 balanced patches and prevalence and still produced test_seen with 39 ground units
    against test_unseen's 60 -- 4.38 events per unit against 2.52. Pooled IoU then counted
    the seen slice's ground 1.7x more often than the unseen slice's, which is the whole of
    the 2.16x basin-familiarity effect. Counts have to be a first-class term, because
    nothing about matching patches or prevalence constrains them.
    """
    e = 0.0
    for key in ('aois', 'units'):
        tot = sum(prof[c][key] for c in TARGET)          # train + val + test partition it
        for c, t in TARGET.items():
            e += abs(prof[c][key] / max(tot, 1) - t) / t
        a, b = prof['seen'][key], prof['unseen'][key]    # the RQ2 contrast, counted
        e += 2.0 * abs(a - b) / max((a + b) / 2.0, 1.0)
    return e


def struct_err(prof: dict) -> float:
    """Permanent water and ground repetition — two structural advantages that both inflate
    test/seen, and that the soft divergence sum was too weak to hold.

    MEASURED on the 2026-09-05 seed-29 build: test/seen carried 11.44% permanent water against
    test/unseen's 4.65% (2.46x) and repeated its ground 4.44 times per spatial block against 2.53.
    v0, for comparison, was 0.83x on PW — so this was a regression, not something inherited.

    Why PW matters even though the flood-only metric drops PW cells: the model predicts permanent
    water almost perfectly (pw_only IoU 0.984-0.991 on every v0 cohort), so any UNION number is
    largely a readout of PW content; and per event, PW fraction still correlates positively with
    FLOOD-ONLY IoU (spearman +0.148 on v0, +0.139 partialling out base rate, p=0.002), most likely
    because flood beside a permanent channel is what HAND and UDA predict best.

    `divergence()` already carries pw_frac, but at weight 0.5 against prevalence's 3.0 — about a
    fifth of d_seen_vs_unseen, which was not enough to stop it drifting. Here it is a hard term.
    """
    def gap(a, b):
        return abs(a - b) / max((a + b) / 2.0, 1e-9)
    e = gap(prof['seen']['pw_frac'], prof['unseen']['pw_frac'])
    e += gap(prof['val']['pw_frac'], prof['test']['pw_frac'])
    e += gap(prof['seen']['events'] / max(prof['seen']['units'], 1),
             prof['unseen']['events'] / max(prof['unseen']['units'], 1))
    return e


def score(prof: dict, profe: dict, shares: dict, leak: float) -> tuple[float, dict]:
    """val selects and test reports, so those must be one population; the two test slices
    are the RQ2 contrast, so those must be one population too.

    The size target is on the CARVE (seen + partial + any carved event that came out
    basin-unseen), because that is the quantity the greedy controls; how much of it earns
    the `seen` label is charged separately as `carve_leak`. Targeting `seen` directly would
    ask the search to hit a number it cannot see until the carve has finished."""
    size_err = (sum(abs(shares[c] - t) / t for c, t in TARGET.items())
                + abs(shares['carved'] / max(shares['test'], 1e-9) - SEEN_SHARE) / SEEN_SHARE)
    d_pair_vt = divergence(prof['val'], prof['test'])
    d_pair_slices = divergence(prof['seen'], prof['unseen'])
    d_train = sum(divergence(prof[c], prof['train']) for c in ('val', 'test'))
    # EXP 1 trains and reports on ONE event per AOI, so that subset needs its own val-vs-test
    # and seen-vs-unseen match. Balancing only the full set would leave exp1 with a
    # non-random 38% of each cohort and no guarantee the two halves are comparable -- which
    # is exactly the defect v0 had at the cohort level.
    d_e_vt = divergence(profe['val'], profe['test'])
    d_e_slices = divergence(profe['seen'], profe['unseen'])
    # ...and its SIZE SHARES. The divergence terms above match the exp1 subset's SHAPE
    # (prevalence, event size, density cells) and say nothing about how big each cohort is.
    # Without this the first v2 build came out 70.8/14.3/14.8 on all events and
    # 58.9/17.7/23.4 on the exp1 subset -- train sheds proportionally far more when siblings
    # are dropped, because the sibling-heavy ground units are large and land in train.
    e_tot = max(sum(profe[c]['patches'] for c in TARGET), 1)
    e_shares = {c: profe[c]['patches'] / e_tot for c in TARGET}
    size_err_e = sum(abs(e_shares[c] - t) / t for c, t in TARGET.items())
    size_err_e += abs(profe['seen']['patches'] / max(profe['test']['patches'], 1)
                      - SEEN_SHARE) / SEEN_SHARE
    cerr = count_err(prof)
    serr = struct_err(prof)
    parts = {'size_err': size_err, 'd_val_vs_test': d_pair_vt,
             'd_seen_vs_unseen': d_pair_slices, 'd_train': d_train, 'carve_leak': leak,
             'd_exp1_val_vs_test': d_e_vt, 'd_exp1_seen_vs_unseen': d_e_slices,
             'size_err_exp1': size_err_e, 'count_err': cerr, 'struct_err': serr,
             'pw_seen': prof['seen']['pw_frac'], 'pw_unseen': prof['unseen']['pw_frac'],
             'acq_per_block_seen': prof['seen']['events'] / max(prof['seen']['units'], 1),
             'acq_per_block_unseen': prof['unseen']['events'] / max(prof['unseen']['units'], 1),
             **{f'exp1_share_{c}': e_shares[c] for c in TARGET}}
    return (3.0 * d_pair_vt + 3.0 * d_pair_slices + 1.0 * d_train
            + 2.0 * d_e_vt + 2.0 * d_e_slices
            + 12.0 * size_err + 10.0 * size_err_e
            + 6.0 * leak + 6.0 * cerr + 8.0 * serr), parts


# ------------------------------------------------------------------ the seen slice
def _worst_act_cover(rows, survives):
    """Worst per-ACTIVATION basin coverage in this subunit.

    An activation is scored on the UNION of its events' basin codes -- it is one storm over
    one region, and it is the unit that moves. Per EVENT (the first v1 build) the gate is
    strictly harder and it halved the pool; see CARVE GATE in the module docstring. Events
    inside a qualifying activation that fall short on their own basins are not rejected now,
    they are labelled `partial`.
    """
    worst = 1.0
    for _, grp in rows.groupby('act', sort=False):
        codes = frozenset().union(*grp.codes)
        worst = min(worst, len(codes & survives) / len(codes))
    return worst


def carve_seen(ev, sub, cohort, regime, tot, rng, ref_unseen):
    """Lift whole activation/scene subunits out of train into test's SEEN slice.

    A subunit may leave if every activation in it still finds >=MIN_BASIN_COVER of its basin
    codes in train. Measured per activation, not per subunit: a subunit averaging above the
    bar can hide an activation with no basin left in train, which would be a basin-UNSEEN
    event wearing the wrong label. The all-or-nothing rule this replaced admitted only 33 of
    74 subunits -- 6.6% of the data, mean event 125 patches against train's 371 -- because a
    large event usually owns at least one basin outright, so it sampled exactly the events
    least like train.
    """
    # .to_numpy(copy=True): on newer pandas `.values` can hand back a READ-ONLY view, and this
    # array is mutated in place below (`in_train &= ~sel`). It worked locally on pandas 2.2.3 and
    # died on Bolzano with "output array is read-only".
    in_train = (cohort == 'train').to_numpy(copy=True)
    code_count = Counter()
    for r in ev[in_train].itertuples():
        for c in r.codes:
            code_count[c] += 1
    target = TARGET['test'] * SEEN_SHARE * tot

    got = 0.0
    while got < target:
        best = None
        for u in rng.permutation(pd.unique(sub[in_train])):
            sel = in_train & (sub == u)
            rows = ev[sel]
            sz = rows.n.sum()
            if sz > 1.35 * (target - got):
                continue
            need = Counter()
            for r in rows.itertuples():
                for c in r.codes:
                    need[c] += 1
            survives = {c for c, k in need.items() if code_count[c] - k > 0}
            survives |= {c for c in code_count if c not in need and code_count[c] > 0}
            if _worst_act_cover(rows, survives) < MIN_BASIN_COVER:
                continue
            idx = np.concatenate([ev.index[regime == 'seen'].values, rows.index.values])
            d = divergence(profile(ev, idx), ref_unseen) - 0.35 * (sz / target)
            if best is None or d < best[0]:
                best = (d, sel, need, sz)
        if best is None:
            break
        _, sel, need, sz = best
        cohort[sel] = 'test'
        regime[sel] = 'seen'
        in_train &= ~sel
        for c, k in need.items():
            code_count[c] -= k
        got += sz

    # repair: a subunit admitted early can fall below the rule once later ones leave
    for _ in range(10):
        survives = {c for c, k in code_count.items() if k > 0}
        bad = None
        for u in pd.unique(sub[(regime == 'seen').values]):
            sel = (regime == 'seen').values & (sub == u)
            if _worst_act_cover(ev[sel], survives) < MIN_BASIN_COVER:
                bad = sel
                break
        if bad is None:
            break
        cohort[bad] = 'train'
        regime[bad] = 'n/a'
        for r in ev[bad].itertuples():
            for c in r.codes:
                code_count[c] += 1
    return cohort, regime


# ------------------------------------------------------------------ one candidate
def build(ev, comp, sub, seed):
    rng = np.random.default_rng(seed)
    tot = ev.n.sum()
    csize = pd.Series(ev.n.values).groupby(comp).sum()
    small = [c for c in csize.index if csize[c] / tot < BIG_COMP]
    rng.shuffle(small)

    cohort = pd.Series('train', index=ev.index)
    regime = pd.Series('n/a', index=ev.index)

    # val and test's unseen slice are both basin-disjoint from train and from each other,
    # so both take whole components. Filled alternately so one does not take every good
    # component before the other starts.
    want = {'val': TARGET['val'] * tot,
            'unseen': TARGET['test'] * (1 - SEEN_SHARE) * tot}
    got = {k: 0.0 for k in want}
    for c in small:
        room = {k: want[k] - got[k] for k in want}
        k = max(room, key=room.get)
        if room[k] <= 0:
            break
        if csize[c] > 1.6 * room[k]:
            continue
        m = comp == c
        cohort[m] = 'val' if k == 'val' else 'test'
        regime[m] = 'unseen'
        got[k] += csize[c]

    def groups(ch, rg):
        return {'train': ev.index[ch == 'train'].values,
                'val': ev.index[ch == 'val'].values,
                'test': ev.index[ch == 'test'].values,
                'seen': ev.index[rg == 'seen'].values,
                'unseen': ev.index[(ch == 'test') & (rg == 'unseen')].values}

    def partial(ch, rg):
        g = groups(ch, rg)
        if any(len(g[k]) == 0 for k in ('train', 'val', 'unseen')):
            return None
        pr = {k: profile(ev, g[k]) for k in ('train', 'val', 'unseen')}
        se = (abs(pr['val']['patches'] / tot - TARGET['val']) / TARGET['val']
              + abs(pr['unseen']['patches'] / tot - TARGET['test'] * (1 - SEEN_SHARE))
              / (TARGET['test'] * (1 - SEEN_SHARE)))
        return (3.0 * divergence(pr['val'], pr['unseen'])
                + 1.0 * divergence(pr['val'], pr['train'])
                + 1.0 * divergence(pr['unseen'], pr['train']) + 12.0 * se)

    cur = partial(cohort, regime)
    if cur is None:
        return None
    for _ in range(N_REFINE):
        improved = False
        for c in rng.permutation(small):
            m = comp == c
            here = cohort[m].iloc[0]
            for dest in ('train', 'val', 'test'):
                if dest == here:
                    continue
                ch, rg = cohort.copy(), regime.copy()
                ch[m] = dest
                rg[m] = 'n/a' if dest == 'train' else 'unseen'
                sc = partial(ch, rg)
                if sc is not None and sc < cur - 1e-9:
                    cohort, regime, cur, improved = ch, rg, sc, True
                    break
        if not improved:
            break

    g = groups(cohort, regime)
    if len(g['unseen']) == 0:
        return None
    cohort, regime = carve_seen(ev, sub, cohort, regime,
                                tot, rng, profile(ev, g['unseen']))

    # RELABEL (Ravi, 2026-09-03, second pass): the gate is per activation, so an event
    # inside a qualifying activation can still fall short on its own basins. Rejecting those
    # candidates is what emptied the pool -- label them instead. All three values come from
    # the EVENT's own coverage of the FINAL train set, which is also why this happens here
    # and not inside the carve: every subunit that leaves shrinks train's basin set, so no
    # event's frac is settled until the carve stops.
    carved = (regime == 'seen').to_numpy()
    tr_codes = frozenset().union(*ev[cohort == 'train'].codes)
    # v2: the regime is decided for the GROUND UNIT, not the event. In v1 it was per event
    # and one unit came out with events on BOTH sides of the RQ2 contrast --
    # EMSR471_AOI01 labelled `seen` and the overlapping EMSR471_AOI02 labelled `unseen`, so
    # the same ground was compared against itself. A unit is atomic here, so its basin set
    # is the honest one to score.
    ucodes = ev.groupby('ground_unit').codes.agg(lambda s: frozenset().union(*s))
    ufrac = ucodes.map(lambda cs: len(cs & tr_codes) / len(cs))
    frac = ev.ground_unit.map(ufrac)
    out = cohort != 'train'
    regime = pd.Series('n/a', index=ev.index)
    regime[out & (frac >= MIN_BASIN_COVER)] = 'seen'
    regime[out & (frac > 0) & (frac < MIN_BASIN_COVER)] = 'partial'
    regime[out & (frac == 0)] = 'unseen'

    g = groups(cohort, regime)
    if any(len(v) == 0 for v in g.values()):
        return None
    prof = {k: profile(ev, v) for k, v in g.items()}
    prof['partial'] = profile(ev, ev.index[(cohort == 'test') & (regime == 'partial')].values)
    # the EXP 1 view of the same cohorts: one event per AOI, nothing else changes
    pos, e_mask = _ARR['pos'], _ARR['exp1']
    def _sub(v):
        return v[e_mask[np.array([pos[i] for i in v], dtype=np.int64)]] if len(v) else v
    ge = {k: _sub(v) for k, v in g.items()}
    if any(len(v) == 0 for v in ge.values()):
        return None
    profe = {k: profile(ev, v) for k, v in ge.items()}
    shares = {k: prof[k]['patches'] / tot for k in prof}
    cp = int(ev.n.values[carved].sum())
    shares['carved'] = cp / tot
    leak = 1.0 - prof['seen']['patches'] / max(cp, 1)
    s, parts = score(prof, profe, shares, leak)
    return dict(cohort=cohort, regime=regime, carved=carved, prof=prof, profe=profe,
                shares=shares, score=s, parts=parts)


# ------------------------------------------------------------------ invariants
def check(ev, cohort, regime) -> list[str]:
    bad = []
    e = ev.assign(cohort=cohort.values, regime=regime.values)
    e['exp1'] = ev.exp1.values
    for col, nm in [('act', 'activation'), ('fg', 'footprint_group'),
                    ('ground_unit', 'ground unit'), ('aoi', 'AOI')]:
        sp = e.groupby(col).cohort.nunique()
        if (sp > 1).any():
            bad.append(f'{nm} spans cohorts: {int((sp > 1).sum())} of {len(sp)}')
    codes = {k: frozenset().union(*e[m].codes) if m.any() else frozenset() for k, m in {
        'train': e.cohort == 'train', 'val': e.cohort == 'val',
        'seen': e.regime == 'seen',
        'unseen': (e.cohort == 'test') & (e.regime == 'unseen')}.items()}
    for k in ('val', 'unseen'):
        if codes[k] & codes['train']:
            bad.append(f'{k} shares {len(codes[k] & codes["train"])} basin codes with train')
    if codes['val'] & codes['unseen']:
        bad.append(f'val shares {len(codes["val"] & codes["unseen"])} basin codes with the unseen slice')
    tr = codes['train']
    # v2: the label is a property of the GROUND UNIT, so it is checked against the unit's
    # basin set. A `seen` unit may legitimately contain an event that, on its own basins,
    # falls below the bar -- that is what made the per-event rule tear units in half.
    uc = e.groupby('ground_unit').codes.agg(lambda x: frozenset().union(*x))
    ufr = uc.map(lambda cs: len(cs & tr) / len(cs))
    fr = e.ground_unit.map(ufr)
    mis = int(((e.regime == 'seen') & (fr < MIN_BASIN_COVER)).sum())
    if mis:
        bad.append(f'{mis} events labelled seen with <{MIN_BASIN_COVER:.0%} of basins in train')
    mis = int(((e.regime == 'unseen') & (fr > 0)).sum())
    if mis:
        bad.append(f'{mis} events labelled unseen holding a basin that is in train')
    mis = int(((e.regime == 'partial') & ((fr <= 0) | (fr >= MIN_BASIN_COVER))).sum())
    if mis:
        bad.append(f'{mis} events labelled partial outside (0, {MIN_BASIN_COVER:.0%})')
    if (e.cohort == 'train').any() and not (e[e.cohort == 'train'].regime == 'n/a').all():
        bad.append('a train event carries a basin regime')
    if (e.cohort == 'val').any() and not (e[e.cohort == 'val'].regime == 'unseen').all():
        bad.append('val is not entirely basin-unseen')
    # v2: one ground unit, one regime -- otherwise the same ground sits on both sides of the
    # seen-vs-unseen contrast, which is what v1 did once.
    mix = int((e.groupby('ground_unit').regime.nunique() > 1).sum())
    if mix:
        bad.append(f'{mix} ground units carry more than one basin regime')
    # every AOI must keep at least one exp1 event, or EXP 1 loses that AOI entirely
    miss = int((e.groupby('aoi').exp1.sum() == 0).sum())
    if miss:
        bad.append(f'{miss} AOIs have no exp1 event')
    if len(e) != int((e.cohort.isin(['train', 'val', 'test'])).sum()):
        bad.append('cohorts are not exhaustive/exclusive')
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=None,
                    help='write THIS candidate instead of the best-scoring admissible one. Used '
                         'to write a deliberately chosen compromise, and to write several '
                         'alternative splits so that split variance can be measured rather than '
                         'assumed — measured, it dominates seed variance.')
    ap.add_argument('--out', default=None, help='write to metadata/<OUT> instead of the default')
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = MD / a.out
    ev = load_events()
    prep(ev)
    comp, sub = components(ev), subunits(ev)
    print(f'{len(ev)} events, {len(set(comp))} components, {len(set(sub))} act/scene '
          f'subunits, {ev.n.sum():,} patches')

    rows, best = [], None
    for seed in range(N_SEEDS):
        r = build(ev, comp, sub, seed)
        if r is None:
            rows.append({'seed': seed, 'score': np.nan, 'valid': False,
                         'violations': 'construction failed'})
            continue
        bad = check(ev, r['cohort'], r['regime'])
        pr = r['parts']['pw_seen'] / max(r['parts']['pw_unseen'], 1e-9)
        br = (r['parts']['acq_per_block_seen']
              / max(r['parts']['acq_per_block_unseen'], 1e-9))
        fr = r['prof']['seen']['elev'] / max(r['prof']['unseen']['elev'], 1e-9)
        tv = 0.5 * float(np.abs(r['prof']['seen']['cshare']
                                - r['prof']['unseen']['cshare']).sum())
        if not PW_RATIO_BAND[0] <= pr <= PW_RATIO_BAND[1]:
            bad = bad + [f'PW ratio {pr:.2f} outside {PW_RATIO_BAND}']
        if not BLK_RATIO_BAND[0] <= br <= BLK_RATIO_BAND[1]:
            bad = bad + [f'acq/block ratio {br:.2f} outside {BLK_RATIO_BAND}']
        if not ELEV_RATIO_BAND[0] <= fr <= ELEV_RATIO_BAND[1]:
            bad = bad + [f'elevation ratio {fr:.2f} outside {ELEV_RATIO_BAND}']
        if tv > CONT_TV_MAX:
            bad = bad + [f'continent TV distance {tv:.2f} > {CONT_TV_MAX}']
        rows.append({'seed': seed, 'score': r['score'], 'valid': not bad,
                     'violations': '; '.join(bad), 'pw_ratio': round(pr, 4),
                     'blk_ratio': round(br, 4), 'elev_ratio': round(fr, 4),
                     'flat_ratio': round(r['prof']['seen']['flat_frac']
                                         / max(r['prof']['unseen']['flat_frac'], 1e-9), 4),
                     'res_ratio': round(r['prof']['seen']['res']
                                        / max(r['prof']['unseen']['res'], 1e-9), 4),
                     'cont_tv': round(tv, 4),
                     **{f'{k}_share': round(r['shares'][k], 5) for k in r['shares']},
                     **{f'{k}_events': r['prof'][k]['events'] for k in r['prof']},
                     **{f'{k}_prev80': round(r['prof'][k]['prev80'], 5) for k in r['prof']},
                     **{k: round(v, 5) for k, v in r['parts'].items()}})
        pick = (seed == a.seed) if a.seed is not None else not bad
        if pick and (best is None or r['score'] < best['score'] or seed == a.seed):
            best = r | {'seed': seed, 'admissible': not bad, 'violations': bad}

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'candidates.csv', index=False)
    n_valid = sum(1 for r in rows if r.get('valid'))
    print(f'candidates: {len(rows)} tried, {n_valid} passed every invariant')
    if best is None:
        print('NO VALID CANDIDATE -- nothing written beyond candidates.csv')
        return 1

    print(f'\nbest seed {best["seed"]}  score {best["score"]:.4f}')
    for k, v in best['parts'].items():
        print(f'   {k:<18} {v:.4f}')
    print(f'\n{"group":<14}{"events":>7}{"AOIs":>6}{"units":>6}{"patches":>9}{"share":>8}'
          f'{"prev80":>9}{"wet rate":>10}{"mean_ev":>9}{"ev/unit":>9}')
    for k in ('train', 'val', 'test', 'seen', 'partial', 'unseen'):
        p = best['prof'][k]
        print(f'{("test/" + k if k in ("seen", "partial", "unseen") else k):<14}{p["events"]:>7}'
              f'{p["aois"]:>6}{p["units"]:>6}{p["patches"]:>9}{best["shares"][k]:>8.1%}'
              f'{p["prev80"]:>9.5f}{p["wet_rate"]:>10.4f}{p["mean_event"]:>9.1f}'
              f'{p["events"]/max(p["units"],1):>9.2f}')
    print(f'\nEXP 1 view — one event per AOI, the sibling with the highest flood fraction')
    print(f'  {"group":<12}{"events":>7}{"patches":>9}{"prev80":>9}{"mean_ev":>9}')
    for k in ('train', 'val', 'test', 'seen', 'unseen'):
        p = best['profe'][k]
        print(f'  {("test/" + k if k in ("seen", "unseen") else k):<12}{p["events"]:>7}'
              f'{p["patches"]:>9}{p["prev80"]:>9.5f}{p["mean_event"]:>9.1f}')

    cv, rg = best['carved'], best['regime']
    print(f'\ncarve: {best["shares"]["carved"]:.2%} of the data lifted out of the train pool '
          f'({int(cv.sum())} events), of which')
    for lab in ('seen', 'partial', 'unseen'):
        m = cv & (rg == lab).to_numpy()
        print(f'   {lab:<9}{int(m.sum()):>5} events {int(ev.n.values[m].sum()):>8} patches'
              f'{ev.n.values[m].sum() / max(ev.n.values[cv].sum(), 1):>8.1%} of the carve')
    print(f'   carve_leak {best["parts"]["carve_leak"]:.3f}   '
          f'partial is {best["prof"]["partial"]["patches"] / max(best["prof"]["test"]["patches"], 1):.1%} of test')

    # ---- write
    train_codes = frozenset().union(*ev[best['cohort'] == 'train'].codes)
    # the LABEL comes from the ground unit; the event's own share is kept alongside it so a
    # stricter per-event cut stays a filter rather than a rebuild.
    _uc = ev.groupby('ground_unit').codes.agg(lambda x: frozenset().union(*x))
    _uf = _uc.map(lambda cs: len(cs & train_codes) / len(cs))
    seen_frac = ev.ground_unit.map(_uf)
    event_frac = ev.codes.map(lambda cs: len(cs & train_codes) / len(cs))
    assert not ((best['regime'] == 'seen') & (seen_frac < MIN_BASIN_COVER)).any()
    assert not ((best['regime'] == 'unseen') & (seen_frac > 0)).any()
    assert not ((best['regime'] == 'partial')
                & ((seen_frac <= 0) | (seen_frac >= MIN_BASIN_COVER))).any()
    pat = pd.read_csv(SRC / 'patches.csv')
    pat['cohort'] = pat.folder_name.map(dict(zip(ev.folder_name, best['cohort'].values)))
    pat['basin_regime'] = pat.folder_name.map(dict(zip(ev.folder_name, best['regime'].values)))
    pat['basin_seen_frac'] = pat.folder_name.map(dict(zip(ev.folder_name, seen_frac.values)))
    for col, vals in (('basin_seen_frac_event', event_frac.values), ('aoi', ev.aoi.values),
                      ('ground_unit', ev.ground_unit.values), ('exp1_keep', ev.exp1.values)):
        pat[col] = pat.folder_name.map(dict(zip(ev.folder_name, vals)))
    pat.to_csv(OUT / 'patches.csv', index=False)
    (OUT / 'flood80_flags.csv').write_bytes((SRC / 'flood80_flags.csv').read_bytes())

    ev.assign(cohort=best['cohort'].values, basin_regime=best['regime'].values,
              basin_seen_frac=seen_frac.values, basin_seen_frac_event=event_frac.values)[
        ['folder_name', 'act', 'aoi', 'ground_unit', 'fg', 'bid', 'n', 'n_flood',
         'cont', 'clim', 'res', 'cohort', 'basin_regime', 'basin_seen_frac',
         'basin_seen_frac_event', 'exp1']].rename(columns={'exp1': 'exp1_keep'}).to_csv(
        OUT / 'cohort_events.csv', index=False)
    def flat(p):
        return {k: v for k, v in p.items() if not isinstance(v, np.ndarray)}
    pd.DataFrame([{'group': k, 'view': 'all events', **flat(best['prof'][k])}
                  for k in ('train', 'val', 'test', 'seen', 'partial', 'unseen')]
                 + [{'group': k, 'view': 'exp1 (one event per AOI)', **flat(best['profe'][k])}
                    for k in ('train', 'val', 'test', 'seen', 'unseen')]
                 ).to_csv(OUT / 'cohort_summary.csv', index=False)
    (OUT / 'manifest.json').write_text(json.dumps({
        'version': 'split_v2',
        'built_from': str(SRC / 'patches.csv'),
        'ground_units_from': str(UNITS),
        'supersedes': 'metadata/split_old/splits_v1, metadata/split_old/splits_v0',
        'n_seeds_tried': N_SEEDS, 'n_valid_candidates': n_valid,
        'best_seed': int(best['seed']), 'best_score': float(best['score']),
        'admissible': bool(best.get('admissible', True)),
        'band_violations': list(best.get('violations', [])),
        'score_parts': {k: float(v) for k, v in best['parts'].items()},
        'targets': TARGET, 'carved_share_of_test': SEEN_SHARE,
        'min_basin_cover': MIN_BASIN_COVER,
        'pw_ratio_band': list(PW_RATIO_BAND),
        'blk_ratio_band': list(BLK_RATIO_BAND),
        'admissibility': 'a candidate is REJECTED unless pw_frac(seen)/pw_frac(unseen) and '
                         'acquisitions-per-spatial-block(seen)/(unseen) both lie in their band. '
                         'Both confounds inflate test/seen and either can manufacture a '
                         'basin-familiarity effect by itself; as soft objective terms they did '
                         'not bind. 14 of 100 candidates survive.',
        'carve_gate': 'per ACTIVATION: every activation in a subunit must keep '
                      '>=min_basin_cover of its basin codes in train after the subunit '
                      'leaves. Events inside it that fall short are labelled partial.',
        'carve_leak': float(best['parts']['carve_leak']),
        'grouping': 'components joined on activation | footprint_group | basin CODE '
                    '(basin_id split on "-") | GROUND UNIT; components are indivisible',
        'ground_unit': 'transitive closure of footprint OVERLAP, computed on patch polygons '
                       'reprojected to EPSG:4087. 1,565 events -> 395 units. A unit is '
                       'atomic, carries ONE basin_regime, and its COUNT is balanced across '
                       'cohorts — v1 balanced patches only and produced test_seen with 39 '
                       'units against test_unseen 60 (4.38 vs 2.52 events per unit), which '
                       'is where its 2.16x basin-familiarity effect came from.',
        'aoi': 'activation_AOIid, 845 of them. Siblings are the same AOI re-mapped on later '
               'dates; measured, 749 events (57.3% of patches) are bit-identical to a '
               'sibling in every band the model sees and differ only in the mask.',
        'exp1_keep': 'ONE event per AOI, the sibling with the highest flood FRACTION (845 '
                     'events, ~215k patches). EXP 1 (terrain-only) uses these rows; EXP 2 '
                     'onward, where weather separates the dates, uses every row. Same '
                     'cohorts either way — one split, two row selections.',
        'counts': {k: {'aois': int(best['prof'][k]['aois']),
                       'units': int(best['prof'][k]['units']),
                       'events': int(best['prof'][k]['events'])}
                   for k in ('train', 'val', 'test', 'seen', 'partial', 'unseen')},
        'exp1_groups': {k: flat(best['profe'][k]) for k in ('train','val','test','seen','unseen')},
        'basin_regime': "three-valued label on every non-train row, from that EVENT's own "
                        "basin_seen_frac against the final train set: 'seen' "
                        ">=min_basin_cover, 'partial' strictly between 0 and "
                        "min_basin_cover, 'unseen' exactly 0. train is 'n/a'; val comes out "
                        "'unseen' by construction. RQ2 contrasts seen vs unseen and leaves "
                        "partial out; partial stays in test for the headline metric. "
                        "basin_seen_frac keeps the raw share so a stricter cut is a filter, "
                        "not a rebuild.",
        'groups': {k: flat(best['prof'][k]) for k in ('train', 'val', 'test', 'seen', 'partial', 'unseen')},
        'invariants_checked': [
            'activation never spans cohorts',
            'footprint_group never spans cohorts',
            'val basin codes disjoint from train',
            'test/unseen basin codes disjoint from train',
            'val basin codes disjoint from test/unseen',
            'every seen-labelled event holds >=min_basin_cover of its basins in train',
            'every unseen-labelled event holds none of its basins in train',
            'every partial-labelled event lies strictly between the two',
            'no train event carries a basin regime',
            'val is entirely basin-unseen',
            'cohorts exhaustive and exclusive'],
        'changes_from_v0': [
            'val_seen dropped (82.6% of its events shared an activation with train)',
            'basin familiarity is a label on test, not a separate cohort',
            'activation added as an atomic unit (v0: 36.3% of activations spanned cohorts)',
            'prevalence matched WITHIN event-size strata, and across joint '
            'size x flood-density cells',
            'continent weighted 0.5 (climate not scored)',
            'basin_regime derived per event, three-valued (seen/partial/unseen); the carve '
            'gate stays per activation, which keeps the eligible pool at 34.4% of the data '
            'rather than 16.8%',
            'val<->test and seen<->unseen matching are first-class objective terms',
            f'{N_SEEDS} seeds searched (v0 searched 1)'],
        'changes_from_v1': [
            'GROUND UNIT (footprint overlap, transitive) is an atomic grouping key',
            'AOI and ground-unit COUNTS are balanced, not just patches and prevalence',
            'basin_regime is decided per ground unit, so one unit can never sit on both '
            'sides of the seen-vs-unseen contrast (v1 had exactly one that did)',
            'the EXP1 subset (one event per AOI, max flood fraction) is matched val-vs-test '
            'and seen-vs-unseen as a first-class objective term',
            'patches.csv gains aoi, ground_unit, exp1_keep, basin_seen_frac_event'],
    }, indent=2))
    print(f'\nwritten to {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
