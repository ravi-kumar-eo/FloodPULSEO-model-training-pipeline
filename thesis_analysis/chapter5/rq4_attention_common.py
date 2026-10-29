"""Day attention of the networks with an input history on all 51,606 test patches (Section 5.4.3, 26 Oct 2026).

Source (read only): experiments/runs_final/{final_t30,final_window}/seed*/<fit>/inference_attention_all/attention.npz,
one forward pass of each fit over every test row (scripts/attention_all.py): att (N, T), the
weather as the network receives it (N, T, 5: local precipitation / 400, soil moisture, valid, upstream precipitation
/ 400, log10(1 + catchment area km2) / 7), folder_name, patch_number and the labelled flood cells of each patch.
Weights are expressed relative to a uniform weighting of the days (1 = 1/T per day) and averaged over seeds 1-3.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path('/media/ravi/Jupyter/flood_vit/experiments/runs_final')
SEEDS = (1, 2, 3)
PDIV, UDIV = 400.0, 7.0
RAIN_DAY = 10.0                                    # mm of upstream precipitation that make a rain day


def path(arch, T, seed):
    fit = f'final_t30/seed{seed}/{arch}' if T == 30 else f'final_window/seed{seed}/{arch}_t{T}'
    return RUNS / fit / 'inference_attention_all' / 'attention.npz'


def relative(z, T):
    a = z['att'].astype(float)
    return a / a.sum(1, keepdims=True) * T


def seed_mean(arch, T):
    """(N, T) attention relative to uniform, mean of the three seeds."""
    return np.mean([relative(np.load(path(arch, T, s)), T) for s in SEEDS], axis=0)


def upstream(T):
    """(N, T) upstream precipitation in mm/day of the input history of length T."""
    return np.load(path('vit', T, 1))['weather'][:, :, 3].astype(float) * PDIV


def local(T):
    return np.load(path('vit', T, 1))['weather'][:, :, 0].astype(float) * PDIV


def rows():
    """folder names, patch numbers, labelled flood cells, activation code and catchment area (km2) of every test row."""
    z = np.load(path('vit', 30, 1), allow_pickle=True)
    fn = z['folder_name'].astype(str)
    act = pd.Series(fn).str.extract(r'^(EMSR\d+)')[0].to_numpy()
    area = 10 ** (z['weather'][:, -1, 4].astype(float) * UDIV) - 1
    return fn, z['patch_number'].astype(int), z['flood'], act, area


def control(fp, act, seed=1):
    """For every flood patch, a random flood patch of another activation: its precipitation is paired with the
    attention of the first patch, which gives the result of an unrelated precipitation series."""
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(fp[act[fp] != act[i]]) for i in fp])
