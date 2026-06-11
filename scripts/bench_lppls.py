#!/usr/bin/env python3
"""Benchmark: legacy multiprocessing nested fits vs the numba kernel.

Usage: python scripts/bench_lppls.py [n_obs ...]
Defaults to 200, 500, and the full Nasdaq dot-com series.
"""

import multiprocessing
import sys
import time

import numpy as np

from lppls import data_loader, lppls, nested


class LegacyLPPLS(lppls.LPPLS):
    """Subclass so type(self) is not LPPLS -> forces the legacy Pool path."""


def load(n_obs=None):
    data = data_loader.nasdaq_dotcom()
    if n_obs is not None:
        data = data.head(n_obs)
    t = np.arange(len(data), dtype=np.float64)
    p = np.log(data["Adj Close"].to_numpy(dtype=np.float64))
    return t, p


def bench_legacy(t, p, workers):
    model = LegacyLPPLS(observations=np.array([t, p]))
    start = time.perf_counter()
    res = model.mp_compute_nested_fits(workers=workers)
    elapsed = time.perf_counter() - start
    n_fits = sum(len(r["res"]) for r in res)
    return elapsed, n_fits


def bench_kernel(t, p, workers):
    start = time.perf_counter()
    res = nested.run_nested_fits(t, p, workers=workers, seed=42, progress=False)
    elapsed = time.perf_counter() - start
    return elapsed, len(res.out)


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [200, 500, None]
    workers = multiprocessing.cpu_count()
    print(f"workers/threads: {workers}")

    # JIT warm-up (cached after first ever run, but exclude from timings)
    t, p = load(120)
    warm_start = time.perf_counter()
    nested.run_nested_fits(t, p, max_searches=2, progress=False, seed=0)
    print(f"kernel warm-up (JIT/cache load): {time.perf_counter() - warm_start:.1f}s\n")

    header = (
        f"{'N':>6} {'fits':>7} {'legacy (s)':>11} {'kernel (s)':>11} "
        f"{'speedup':>8} {'fits/s':>9}"
    )
    print(header)
    print("-" * len(header))
    for size in sizes:
        t, p = load(size)
        legacy_s, n_fits = bench_legacy(t, p, workers)
        kernel_s, n_fits_k = bench_kernel(t, p, workers)
        assert n_fits == n_fits_k, (n_fits, n_fits_k)
        print(
            f"{len(t):>6} {n_fits:>7} {legacy_s:>11.1f} {kernel_s:>11.2f} "
            f"{legacy_s / kernel_s:>7.0f}x {n_fits / kernel_s:>9.0f}"
        )


if __name__ == "__main__":
    main()
