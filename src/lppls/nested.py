"""Orchestration for the numba LPPLS kernel: job building, the chunked
parallel driver, vectorized confidence indicators, and the adapter back to
the legacy list-of-dicts result shape.

This module must not import lppls.lppls (the LPPLS class imports us).
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import _kernel as K

DEFAULT_FILTER_CONDITIONS: dict[str, float] = {
    "m_min": 0.0,
    "m_max": 1.0,
    "w_min": 2.0,
    "w_max": 15.0,
    "O_min": 2.5,
    "D_min": 0.5,
    "tc_min_days": 60.0,
    "tc_max_days": 252.0,
    "tc_min_frac": 0.5,
    "tc_max_frac": 0.5,
}


def resolve_filter_conditions_config(
    filter_conditions_config: dict[str, Any] | None,
) -> dict[str, float]:
    """Validate and merge filter condition thresholds for indicators."""
    defaults = DEFAULT_FILTER_CONDITIONS.copy()
    if filter_conditions_config is None:
        return defaults

    if not isinstance(filter_conditions_config, dict):
        raise TypeError("filter_conditions_config must be a dict[str, float] or None.")

    unknown_keys = set(filter_conditions_config.keys()) - set(defaults.keys())
    if unknown_keys:
        raise ValueError(
            "Unknown filter condition keys: "
            f"{sorted(unknown_keys)}. Supported keys: {sorted(defaults.keys())}."
        )

    resolved = defaults.copy()
    for key, value in filter_conditions_config.items():
        resolved[key] = float(value)

    if resolved["m_min"] >= resolved["m_max"]:
        raise ValueError("m_min must be < m_max.")
    if resolved["w_min"] >= resolved["w_max"]:
        raise ValueError("w_min must be < w_max.")
    if resolved["tc_min_days"] < 0 or resolved["tc_max_days"] < 0:
        raise ValueError("tc_min_days and tc_max_days must be >= 0.")
    if resolved["tc_min_frac"] < 0 or resolved["tc_max_frac"] < 0:
        raise ValueError("tc_min_frac and tc_max_frac must be >= 0.")

    return resolved


_FIT_COLUMNS = [
    "tc",
    "m",
    "w",
    "a",
    "b",
    "c",
    "c1",
    "c2",
    "O",
    "D",
    "t1",
    "t2",
    "sse",
    "success",
]


@dataclass
class NestedFitResult:
    """Raw output of run_nested_fits.

    out has one row per fitted (outer window, shrinking sub-window) pair, in
    outer-major order; job_outer maps each row to its outer window index.
    """

    out: np.ndarray  # float64[J, N_COLS]
    jobs: np.ndarray  # int64[J, 2] (start, end_exclusive)
    job_outer: np.ndarray  # int64[J] outer-window index per job
    outer_t1: np.ndarray  # float64[W] outer window start times
    outer_t2: np.ndarray  # float64[W] outer window end times
    outer_p2: np.ndarray  # float64[W] observed value at outer window end
    window_size: int
    smallest_window_size: int
    outer_increment: int
    inner_increment: int
    max_searches: int
    seed: int

    @property
    def n_outer(self) -> int:
        return len(self.outer_t2)

    def fits_dataframe(self) -> pd.DataFrame:
        """All fits as a flat DataFrame (one row per sub-window fit)."""
        df = pd.DataFrame(self.out, columns=_FIT_COLUMNS)
        df.insert(0, "outer", self.job_outer)
        return df

    def tc_as_timestamps(self, index: pd.DatetimeIndex) -> pd.Series:
        """Map fitted tc values (in position units) to Timestamps.

        Positions inside the observed range interpolate the given index;
        positions beyond the last observation extrapolate with business-day
        steps. Only meaningful when the fits were run on positional time.
        """
        tc = self.out[:, K.COL_TC]
        success = self.out[:, K.COL_SUCCESS] == 1.0
        n = len(index)
        result = pd.Series(pd.NaT, index=range(len(tc)), dtype="datetime64[ns]")
        for i, (tci, ok) in enumerate(zip(tc, success)):
            if not ok:
                continue
            pos = int(round(tci))
            if 0 <= pos < n:
                result.iloc[i] = index[pos]
            elif pos >= n:
                result.iloc[i] = index[-1] + pd.offsets.BDay(pos - (n - 1))
        return result


def build_jobs(
    n_obs: int,
    window_size: int,
    smallest_window_size: int,
    outer_increment: int,
    inner_increment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten the nested-fit loops into a job list.

    Replicates the legacy loop semantics exactly, including the quirk that
    the inner range stops before window_delta, so with the defaults the
    smallest fitted window is smallest_window_size + inner_increment points.

    Returns (jobs int64[J,2], job_outer int64[J], outer_starts int64[W]).
    """
    if window_size > n_obs:
        raise ValueError(
            f"window_size ({window_size}) exceeds series length ({n_obs})."
        )
    if smallest_window_size >= window_size:
        raise ValueError("smallest_window_size must be < window_size.")

    outer_starts = np.arange(
        0, n_obs - window_size + 1, outer_increment, dtype=np.int64
    )
    window_delta = window_size - smallest_window_size
    inner_offsets = np.arange(0, window_delta, inner_increment, dtype=np.int64)

    n_outer = len(outer_starts)
    n_inner = len(inner_offsets)
    jobs = np.empty((n_outer * n_inner, 2), dtype=np.int64)
    job_outer = np.empty(n_outer * n_inner, dtype=np.int64)
    k = 0
    for oi, i in enumerate(outer_starts):
        for j in inner_offsets:
            jobs[k, 0] = i + j
            jobs[k, 1] = i + window_size
            job_outer[k] = oi
            k += 1
    return jobs, job_outer, outer_starts


def run_nested_fits(
    t: np.ndarray,
    p: np.ndarray,
    *,
    window_size: int = 80,
    smallest_window_size: int = 20,
    outer_increment: int = 5,
    inner_increment: int = 2,
    max_searches: int = 25,
    rescale: bool = True,
    workers: int | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> NestedFitResult:
    """Run all nested LPPLS fits through the numba kernel.

    t, p: 1-D float64 arrays (time axis and log-price). The same base seed
    gives bit-identical results regardless of workers or chunking.
    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    p = np.ascontiguousarray(p, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 1 or len(t) != len(p):
        raise ValueError("t and p must be 1-D arrays of equal length.")

    jobs, job_outer, outer_starts = build_jobs(
        len(t),
        window_size,
        smallest_window_size,
        outer_increment,
        inner_increment,
    )
    n_jobs = len(jobs)
    n_inner = n_jobs // len(outer_starts)

    if seed is None:
        seed = _random.randrange(2**64)
    seeds = K.derive_seeds(seed, n_jobs)
    out = np.zeros((n_jobs, K.N_COLS), dtype=np.float64)

    if workers is not None:
        numba.set_num_threads(max(1, min(workers, numba.config.NUMBA_NUM_THREADS)))

    # prange cannot tick a progress bar, so chunk the job list at
    # outer-window boundaries and loop chunks in Python. Seeds are derived
    # from global job indices, so chunking never changes results.
    n_threads = numba.get_num_threads()
    n_outer = len(outer_starts)
    # Both are lower bounds on the chunk size: at most ~40 progress ticks,
    # and enough jobs per chunk (~8 per thread) to keep prange saturated.
    outers_per_chunk = (
        max(
            -(-n_outer // 40),
            -(-8 * n_threads // n_inner),
            1,
        )
        if progress
        else n_outer
    )
    chunk_bounds = [
        (oi * n_inner, min(oi + outers_per_chunk, n_outer) * n_inner)
        for oi in range(0, n_outer, outers_per_chunk)
    ]

    iterator = tqdm(chunk_bounds, disable=not progress)
    for a, b in iterator:
        K.batch_fit(t, p, jobs[a:b], max_searches, rescale, seeds[a:b], out[a:b])

    outer_ends = outer_starts + window_size - 1
    return NestedFitResult(
        out=out,
        jobs=jobs,
        job_outer=job_outer,
        outer_t1=t[outer_starts],
        outer_t2=t[outer_ends],
        outer_p2=p[outer_ends],
        window_size=window_size,
        smallest_window_size=smallest_window_size,
        outer_increment=outer_increment,
        inner_increment=inner_increment,
        max_searches=max_searches,
        seed=seed,
    )


def indicators_from_matrix(
    result: NestedFitResult,
    filter_conditions_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Vectorized equivalent of LPPLS.compute_indicators.

    Same qualification rules (strict inequalities, O treated as +inf when
    b == 0 or c == 0, NaN comparisons disqualify) applied over the result
    matrix with no Python loop.
    """
    cfg = resolve_filter_conditions_config(filter_conditions_config)
    out = result.out

    tc = out[:, K.COL_TC]
    m = out[:, K.COL_M]
    w = out[:, K.COL_W]
    b = out[:, K.COL_B]
    c = out[:, K.COL_C]
    O = out[:, K.COL_O]
    D = out[:, K.COL_D]
    t1 = out[:, K.COL_T1]
    t2 = out[:, K.COL_T2]

    lo = np.maximum(t2 - cfg["tc_min_days"], t2 - cfg["tc_min_frac"] * (t2 - t1))
    hi = np.minimum(t2 + cfg["tc_max_days"], t2 + cfg["tc_max_frac"] * (t2 - t1))
    Oeff = np.where((b != 0) & (c != 0), O, np.inf)

    qual = (
        (lo < tc)
        & (tc < hi)
        & (cfg["m_min"] < m)
        & (m < cfg["m_max"])
        & (cfg["w_min"] < w)
        & (w < cfg["w_max"])
        & (Oeff > cfg["O_min"])
        & (D > cfg["D_min"])
    )
    pos = b < 0
    neg = b > 0

    n_outer = result.n_outer
    pos_count = np.bincount(result.job_outer[pos], minlength=n_outer)
    neg_count = np.bincount(result.job_outer[neg], minlength=n_outer)
    pos_qual = np.bincount(result.job_outer[pos & qual], minlength=n_outer)
    neg_qual = np.bincount(result.job_outer[neg & qual], minlength=n_outer)

    pos_conf = np.divide(
        pos_qual, pos_count, out=np.zeros(n_outer), where=pos_count > 0
    )
    neg_conf = np.divide(
        neg_qual, neg_count, out=np.zeros(n_outer), where=neg_count > 0
    )

    return pd.DataFrame(
        {
            "time": result.outer_t2,
            "price": result.outer_p2,
            "pos_conf": pos_conf,
            "neg_conf": neg_conf,
        }
    )


def to_legacy_dicts(result: NestedFitResult) -> list[dict[str, Any]]:
    """Rebuild the legacy mp_compute_nested_fits return structure.

    [{t1, t2, p2, res: [{tc, m, w, a, b, c, c1, c2, t1, t2, O, D}, ...]}, ...]
    so compute_indicators / plot_confidence_indicators work unchanged.
    """
    out = result.out
    legacy: list[dict[str, Any]] = []
    j = 0
    n_inner = len(out) // result.n_outer
    for oi in range(result.n_outer):
        res = []
        for _ in range(n_inner):
            row = out[j]
            res.append(
                {
                    "tc": row[K.COL_TC],
                    "m": row[K.COL_M],
                    "w": row[K.COL_W],
                    "a": row[K.COL_A],
                    "b": row[K.COL_B],
                    "c": row[K.COL_C],
                    "c1": row[K.COL_C1],
                    "c2": row[K.COL_C2],
                    "t1": row[K.COL_T1],
                    "t2": row[K.COL_T2],
                    "O": row[K.COL_O],
                    "D": row[K.COL_D],
                }
            )
            j += 1
        legacy.append(
            {
                "t1": result.outer_t1[oi],
                "t2": result.outer_t2[oi],
                "p2": result.outer_p2[oi],
                "res": res,
            }
        )
    return legacy
