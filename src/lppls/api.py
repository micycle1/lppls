"""Pandas-first convenience API for LPPLS bubble detection.

The single entry point, bubble_confidence, takes a price series with a
(typically datetime) index and returns the LPPLS confidence indicators on
that same index.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from lppls import nested as _nested
from lppls.nested import NestedFitResult


def bubble_confidence(
    prices: pd.Series,
    *,
    log_prices: bool = True,
    time_scale: str = "positions",
    window_size: int = 80,
    smallest_window_size: int = 20,
    outer_increment: int = 5,
    inner_increment: int = 2,
    max_searches: int = 25,
    filter_conditions: dict[str, Any] | None = None,
    workers: int | None = None,
    seed: int | None = None,
    progress: bool = True,
    return_fits: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, NestedFitResult]:
    """Compute LPPLS bubble confidence indicators for a price series.

    Args:
        prices: Price series with a monotonically increasing index
            (typically a DatetimeIndex of daily bars). NaNs are dropped with
            a warning.
        log_prices: If True (default), apply np.log to the input — pass raw
            prices. Set False if the series is already in log space.
        time_scale: Internal time axis for the fits.
            - "positions" (default): t = 0, 1, 2, ... per observation
              ("trading-day time"). No weekend/holiday gaps distorting the
              log-periodic oscillations; the tc filter bounds
              (tc_min_days=60, tc_max_days=252) read as trading days.
            - "ordinal": calendar-day ordinals (legacy library behavior;
              requires a DatetimeIndex). Filter bounds read as calendar days.
        window_size, smallest_window_size, outer_increment, inner_increment,
        max_searches: Nested-fit parameters, as in mp_compute_nested_fits.
        filter_conditions: Optional indicator thresholds (see
            LPPLS.compute_indicators for supported keys).
        workers: numba thread count (default: all cores).
        seed: Base RNG seed for fully reproducible results.
        progress: Show a tqdm progress bar.
        return_fits: If True, also return the NestedFitResult for drill-down
            into individual fits (tc, m, w, ... per sub-window).

    Returns:
        DataFrame on the SAME index as ``prices`` with columns:
          - 'price': log price (or the input values if log_prices=False)
          - 'pos_conf': positive bubble confidence in [0, 1]
          - 'neg_conf': negative bubble confidence in [0, 1]
        Confidence values exist at outer-window end dates (every
        ``outer_increment``-th bar from bar ``window_size - 1``); other rows
        are NaN. Use ``outer_increment=1`` for a dense signal.
        With return_fits=True, returns (DataFrame, NestedFitResult).
    """
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pd.Series, got {type(prices)}")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be monotonically increasing.")
    if prices.index.has_duplicates:
        raise ValueError("prices index contains duplicate entries.")

    clean = prices.dropna()
    if len(clean) < len(prices):
        warnings.warn(
            f"Dropped {len(prices) - len(clean)} NaN observation(s) from the "
            "input series."
        )
    if len(clean) < window_size:
        raise ValueError(
            f"Need at least window_size={window_size} non-NaN observations, "
            f"got {len(clean)}."
        )

    values = clean.to_numpy(dtype=np.float64)
    if log_prices:
        if np.any(values <= 0):
            raise ValueError(
                "Prices must be strictly positive to take logs "
                "(or pass log_prices=False)."
            )
        values = np.log(values)

    if time_scale == "positions":
        t = np.arange(len(clean), dtype=np.float64)
    elif time_scale == "ordinal":
        if not isinstance(clean.index, pd.DatetimeIndex):
            raise ValueError('time_scale="ordinal" requires a DatetimeIndex.')
        t = np.array([ts.toordinal() for ts in clean.index], dtype=np.float64)
    else:
        raise ValueError(
            f'time_scale must be "positions" or "ordinal", got {time_scale!r}'
        )

    result = _nested.run_nested_fits(
        t,
        values,
        window_size=window_size,
        smallest_window_size=smallest_window_size,
        outer_increment=outer_increment,
        inner_increment=inner_increment,
        max_searches=max_searches,
        workers=workers,
        seed=seed,
        progress=progress,
    )
    indicators = _nested.indicators_from_matrix(result, filter_conditions)

    # outer window ends as positions into the cleaned series
    end_positions = np.arange(result.n_outer) * outer_increment + window_size - 1
    df = pd.DataFrame(
        {
            "price": values,
            "pos_conf": np.nan,
            "neg_conf": np.nan,
        },
        index=clean.index,
    )
    df.iloc[end_positions, df.columns.get_loc("pos_conf")] = indicators[
        "pos_conf"
    ].to_numpy()
    df.iloc[end_positions, df.columns.get_loc("neg_conf")] = indicators[
        "neg_conf"
    ].to_numpy()
    # back to the caller's exact index (NaN at rows dropped as NaN input)
    df = df.reindex(prices.index)

    if return_fits:
        return df, result
    return df
