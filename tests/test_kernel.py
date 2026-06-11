#!/usr/bin/env python3
"""Tests for the numba fitting kernel, its parity with the legacy scipy
implementation, determinism, and the pandas-first API."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from lppls import data_loader, lppls
from lppls import _kernel as K
from lppls import nested
from lppls.api import bubble_confidence


@pytest.fixture(scope="module")
def series():
    data = data_loader.nasdaq_dotcom()
    t = np.arange(len(data), dtype=np.float64)
    p = np.log(data["Adj Close"].to_numpy(dtype=np.float64))
    return t, p


@pytest.fixture(scope="module")
def model(series):
    t, p = series
    return lppls.LPPLS(observations=np.array([t, p]))


def _scratch(n):
    return (
        np.empty(n),
        np.empty(n),
        np.empty(n),
        np.empty(4),
        np.empty((4, 3)),
        np.empty(4),
        np.empty(3),
        np.empty(3),
        np.empty(3),
    )


# ---------------------------------------------------------------------------
# Objective parity: fused kernel vs func_restricted + matrix_equation
# ---------------------------------------------------------------------------


def test_objective_parity(series, model):
    t, p = series
    rng = np.random.default_rng(7)
    fi, gi, hi, beta = (np.empty(120), np.empty(120), np.empty(120), np.empty(4))
    for _ in range(50):
        n = int(rng.integers(20, 121))
        i0 = int(rng.integers(0, len(t) - n))
        tw, pw = t[i0 : i0 + n], p[i0 : i0 + n]
        t1, t2 = tw[0], tw[-1]
        tc = float(rng.uniform(t2 - 0.2 * (t2 - t1), t2 + 0.2 * (t2 - t1)))
        m = float(rng.uniform(0.1, 1.0))
        w = float(rng.uniform(6.0, 13.0))

        obs = np.array([tw, pw])
        sse_legacy = model.func_restricted(np.array([tc, m, w]), obs)
        beta_legacy = model.matrix_equation(obs, tc, m, w)[:, 0]

        sse_kernel = K._lppls_sse(tw, pw, tc, m, w, fi[:n], gi[:n], hi[:n], beta)

        assert sse_kernel == pytest.approx(sse_legacy, rel=1e-9)
        np.testing.assert_allclose(beta, beta_legacy, rtol=1e-6, atol=1e-10)


# ---------------------------------------------------------------------------
# Nelder-Mead parity vs scipy (same x0, rescale off)
# ---------------------------------------------------------------------------


def test_nelder_mead_parity(series, model):
    t, p = series
    rng = np.random.default_rng(11)
    flag_agree = 0
    sse_close = 0
    n_cases = 30
    both_converged = 0
    scipy_aborted = 0
    for _ in range(n_cases):
        n = int(rng.integers(40, 121))
        i0 = int(rng.integers(0, len(t) - n))
        tw = np.ascontiguousarray(t[i0 : i0 + n])
        pw = np.ascontiguousarray(p[i0 : i0 + n])
        t1, t2 = tw[0], tw[-1]
        x0 = np.array(
            [
                rng.uniform(t2 - 0.2 * (t2 - t1), t2 + 0.2 * (t2 - t1)),
                rng.uniform(0.1, 1.0),
                rng.uniform(6.0, 13.0),
            ]
        )

        obs = np.array([tw, pw])
        # Legacy aborts the whole search when the objective overflows
        # mid-trajectory (LinAlgError); the kernel maps that point to +inf
        # and keeps optimizing. That's an intended robustness improvement,
        # so abort cases don't count against flag agreement.
        try:
            ref = minimize(
                args=obs, fun=model.func_restricted, x0=x0.copy(), method="Nelder-Mead"
            )
            ref_success = bool(ref.success)
            ref_fun = ref.fun
        except np.linalg.LinAlgError:
            scipy_aborted += 1
            continue

        fi, gi, hi, beta, sim, fsim, xbar, xr, xtr = _scratch(n)
        converged, _ = K._nm_minimize(
            tw,
            pw,
            fi,
            gi,
            hi,
            beta,
            sim,
            fsim,
            xbar,
            xr,
            xtr,
            x0.copy(),
            1e-4,
            1e-4,
            600,
            600,
        )

        if converged == ref_success:
            flag_agree += 1
        if converged and ref_success:
            both_converged += 1
            if fsim[0] == pytest.approx(ref_fun, rel=1e-4):
                sse_close += 1

    n_compared = n_cases - scipy_aborted
    # NM trajectories on float64 can diverge chaotically after enough
    # iterations, so require strong but not perfect agreement.
    assert flag_agree >= int(0.9 * n_compared)
    assert both_converged > 0
    assert sse_close >= int(0.9 * both_converged)


# ---------------------------------------------------------------------------
# Determinism of the batch driver
# ---------------------------------------------------------------------------


def test_batch_determinism(series):
    t, p = series
    t, p = t[:200], p[:200]

    runs = []
    for workers in (1, 4):
        for _ in range(2):
            res = nested.run_nested_fits(
                t,
                p,
                max_searches=5,
                workers=workers,
                seed=123,
                progress=False,
            )
            runs.append(res.out.copy())

    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)


def test_chunking_neutral(series):
    """Chunked (progress=True) and unchunked runs give identical results."""
    t, p = series
    t, p = t[:200], p[:200]
    a = nested.run_nested_fits(t, p, max_searches=5, seed=9, progress=False)
    b = nested.run_nested_fits(t, p, max_searches=5, seed=9, progress=True)
    np.testing.assert_array_equal(a.out, b.out)


# ---------------------------------------------------------------------------
# Job construction mirrors the legacy loops
# ---------------------------------------------------------------------------


def test_build_jobs_defaults():
    jobs, job_outer, outer_starts = nested.build_jobs(100, 80, 20, 5, 2)
    assert len(outer_starts) == 5  # range(0, 21, 5)
    assert len(jobs) == 5 * 30  # range(0, 60, 2) -> 30 inner windows
    # first outer window: full window then shrinking from the left
    assert jobs[0].tolist() == [0, 80]
    assert jobs[1].tolist() == [2, 80]
    assert jobs[29].tolist() == [58, 80]  # smallest fitted window is 22 pts
    assert jobs[30].tolist() == [5, 85]
    assert job_outer[29] == 0 and job_outer[30] == 1


def test_build_jobs_validation():
    with pytest.raises(ValueError):
        nested.build_jobs(50, 80, 20, 5, 2)
    with pytest.raises(ValueError):
        nested.build_jobs(100, 80, 80, 5, 2)


# ---------------------------------------------------------------------------
# Vectorized indicators match the legacy dict-based compute_indicators
# ---------------------------------------------------------------------------


def test_indicators_match_legacy(series, model):
    t, p = series
    t, p = t[:300], p[:300]
    res = nested.run_nested_fits(t, p, max_searches=10, seed=42, progress=False)

    fast = nested.indicators_from_matrix(res)
    legacy = model.compute_indicators(nested.to_legacy_dicts(res))

    np.testing.assert_allclose(fast["pos_conf"], legacy["pos_conf"])
    np.testing.assert_allclose(fast["neg_conf"], legacy["neg_conf"])
    np.testing.assert_allclose(fast["time"], legacy["time"])
    np.testing.assert_allclose(fast["price"], legacy["price"])


def test_indicators_custom_config(series):
    t, p = series
    t, p = t[:200], p[:200]
    res = nested.run_nested_fits(t, p, max_searches=10, seed=42, progress=False)
    default = nested.indicators_from_matrix(res)
    loose = nested.indicators_from_matrix(res, {"O_min": 0.0, "D_min": 0.0})
    assert (loose["pos_conf"] >= default["pos_conf"]).all()
    with pytest.raises(ValueError):
        nested.indicators_from_matrix(res, {"bogus": 1.0})


# ---------------------------------------------------------------------------
# bubble_confidence pandas API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def price_series():
    data = data_loader.nasdaq_dotcom().head(300)
    idx = pd.DatetimeIndex(pd.to_datetime(data["Date"]))
    return pd.Series(data["Adj Close"].to_numpy(), index=idx, name="nasdaq")


def test_bubble_confidence_index_alignment(price_series):
    df = bubble_confidence(price_series, max_searches=5, seed=1, progress=False)
    assert df.index.equals(price_series.index)
    assert list(df.columns) == ["price", "pos_conf", "neg_conf"]

    # confidence exists exactly at outer-window ends
    expected_positions = set(range(79, 300, 5))
    have = set(np.flatnonzero(df["pos_conf"].notna().to_numpy()))
    assert have == expected_positions

    conf = df["pos_conf"].dropna()
    assert ((conf >= 0) & (conf <= 1)).all()
    # price column is the log price
    np.testing.assert_allclose(df["price"].to_numpy(), np.log(price_series.to_numpy()))


def test_bubble_confidence_nan_handling(price_series):
    withnan = price_series.copy()
    withnan.iloc[10] = np.nan
    with pytest.warns(UserWarning, match="Dropped 1 NaN"):
        df = bubble_confidence(withnan, max_searches=2, seed=1, progress=False)
    assert df.index.equals(withnan.index)
    assert np.isnan(df["price"].iloc[10])


def test_bubble_confidence_ordinal_scale(price_series):
    df = bubble_confidence(
        price_series, time_scale="ordinal", max_searches=2, seed=1, progress=False
    )
    assert df.index.equals(price_series.index)
    conf = df["pos_conf"].dropna()
    assert ((conf >= 0) & (conf <= 1)).all()


def test_bubble_confidence_validation(price_series):
    with pytest.raises(TypeError):
        bubble_confidence(price_series.to_frame())
    with pytest.raises(ValueError, match="monotonically"):
        bubble_confidence(price_series.iloc[::-1])
    with pytest.raises(ValueError, match="duplicate"):
        dup = pd.concat([price_series, price_series.iloc[[0]]]).sort_index()
        bubble_confidence(dup)
    with pytest.raises(ValueError, match="window_size"):
        bubble_confidence(price_series.head(50))
    with pytest.raises(ValueError, match="positive"):
        neg = price_series.copy()
        neg.iloc[0] = -1.0
        bubble_confidence(neg)
    with pytest.raises(ValueError, match="time_scale"):
        bubble_confidence(price_series, time_scale="bogus")
    with pytest.raises(ValueError, match="DatetimeIndex"):
        bubble_confidence(pd.Series(price_series.to_numpy()), time_scale="ordinal")


def test_bubble_confidence_reproducible(price_series):
    df1 = bubble_confidence(price_series, max_searches=3, seed=5, progress=False)
    df2 = bubble_confidence(price_series, max_searches=3, seed=5, progress=False)
    pd.testing.assert_frame_equal(df1, df2)


def test_bubble_confidence_return_fits(price_series):
    df, res = bubble_confidence(
        price_series, max_searches=3, seed=5, progress=False, return_fits=True
    )
    assert isinstance(res, nested.NestedFitResult)
    fits = res.fits_dataframe()
    assert {"tc", "m", "w", "sse", "success", "outer"} <= set(fits.columns)
    assert len(fits) == len(res.jobs)
    ts = res.tc_as_timestamps(price_series.index)
    ok = res.out[:, K.COL_SUCCESS] == 1.0
    assert ts[ok].notna().all()
