"""Experiment invariants that a config CANNOT override.

WHY. Exp 2, 3 and 4 run the SAME weather model and differ only in `mono_temporal` and
`temporal_days`. That means one edited YAML line silently turns the Exp 3 control into another
Exp 2 arm, producing a plausible-looking result that answers nothing. The defining conditions
of an experiment therefore live in its entry-point script, are asserted here before a single
batch is read, and the run directory is derived from them rather than from a free-text name.
"""
from __future__ import annotations

VALID = {
    # experiment: (temporal_days, mono_temporal, purpose)
    'exp1': ((0,),            True,  'TERRAIN only, D_mono — the terrain baseline'),
    'exp2': ((30,),           False, 'terrain + daily weather, D_multi — the weather model'),
    'exp3': ((30,),           True,  'terrain + daily weather, D_mono — the RQ3b control'),
    'exp4': ((3, 7, 15, 30),  False, 'window study, D_multi — RQ4'),
    # Antara's critique 1 (2026-09-22): the 10-epoch freeze was never tested against NOT
    # freezing. Identical to exp2 in every defining condition -- same rows, same window, same
    # warm start -- and differs ONLY in freeze_epochs, so it is the ablation and not a new arm.
    'exp2_non_freeze': ((30,),  False, 'exp2 with freeze_epochs=0 — the no-freeze ablation'),
    # U-TAE weather path for the U-Net (temporal_mode: utae), added 2026-10-04
    'exp2_utae': ((30,),          False, 'U-Net U-TAE weather model, D_multi, T30'),
    'exp4_utae': ((3, 7, 15),     False, 'U-Net U-TAE window study, D_multi'),
}


def check(experiment: str, backbone: str, cfg: dict) -> None:
    """Raise unless `cfg` is genuinely the experiment the entry point claims to be."""
    if experiment not in VALID:
        raise ValueError(f'unknown experiment {experiment!r}; expected one of {sorted(VALID)}')
    days_ok, mono_ok, purpose = VALID[experiment]

    got_bb = str(cfg.get('backbone', ''))
    if got_bb != backbone:
        raise ValueError(f'{experiment}/{backbone}: config declares backbone {got_bb!r}. '
                         f'Each backbone has its own entry point; use that one.')

    got_days = int(cfg.get('temporal_days', 0))
    if got_days not in days_ok:
        raise ValueError(f'{experiment}: temporal_days={got_days} but {experiment} is defined '
                         f'at {days_ok}. {purpose}.')

    got_mono = bool(cfg.get('mono_temporal', False))
    if got_mono is not mono_ok:
        rows = 'D_mono' if got_mono else 'D_multi'
        want = 'D_mono' if mono_ok else 'D_multi'
        raise ValueError(f'{experiment}: config trains on {rows}, but {experiment} is defined on '
                         f'{want}. {purpose}. Running it on {rows} makes it a different '
                         f'experiment that answers a different question.')
