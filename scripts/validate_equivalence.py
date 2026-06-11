#!/usr/bin/env python3
"""Statistical equivalence of the numba kernel vs the legacy scipy path.

Runs nested fits on the Nasdaq dot-com series through:
  (a) the legacy multiprocessing/scipy path (multiple runs -> its own
      seed-to-seed spread, since it cannot be seeded),
  (b) the kernel with rescale=False (isolates the NM reimplementation),
  (c) the kernel with rescale=True (adds the conditioning change; default),
then compares the pos/neg confidence series and the pooled distributions of
m, w, and (tc - t2)/(t2 - t1) over successful fits.

Usage: python scripts/validate_equivalence.py [n_obs] (default 600)
"""

import sys
import time

import numpy as np
from scipy.stats import ks_2samp, pearsonr

from lppls import data_loader, lppls, nested


class LegacyLPPLS(lppls.LPPLS):
    """Forces the legacy Pool/scipy path (type(self) is not LPPLS)."""


def load(n_obs):
    data = data_loader.nasdaq_dotcom()
    if n_obs is not None:
        data = data.head(n_obs)
    t = np.arange(len(data), dtype=np.float64)
    p = np.log(data["Adj Close"].to_numpy(dtype=np.float64))
    return t, p


def legacy_run(t, p, workers=None):
    import multiprocessing

    model = LegacyLPPLS(observations=np.array([t, p]))
    res = model.mp_compute_nested_fits(workers=workers or multiprocessing.cpu_count())
    ind = model.compute_indicators(res)
    fits = np.array(
        [
            [f["tc"], f["m"], f["w"], f["b"], f["t1"], f["t2"]]
            for r in res
            for f in r["res"]
        ]
    )
    return ind, fits


def kernel_run(t, p, seed, rescale):
    res = nested.run_nested_fits(t, p, seed=seed, rescale=rescale, progress=False)
    ind = nested.indicators_from_matrix(res)
    out = res.out
    ok = out[:, 13] == 1.0  # COL_SUCCESS
    fits = out[np.ix_(np.arange(len(out)), [0, 1, 2, 4, 10, 11])]
    fits = np.where(ok[:, None], fits, 0.0)
    return ind, fits


def summarize_fits(fits):
    """Pooled m, w, scaled tc over successful fits (b != 0)."""
    ok = fits[:, 3] != 0
    f = fits[ok]
    tc, m, w, _b, t1, t2 = f.T
    tc_scaled = (tc - t2) / (t2 - t1)
    return m, w, tc_scaled


def compare(name, base_ind, base_fits, test_ind, test_fits):
    r_pos = pearsonr(base_ind["pos_conf"], test_ind["pos_conf"])[0]
    r_neg = pearsonr(base_ind["neg_conf"], test_ind["neg_conf"])[0]
    d_pos = np.abs(base_ind["pos_conf"] - test_ind["pos_conf"])
    d_neg = np.abs(base_ind["neg_conf"] - test_ind["neg_conf"])
    m0, w0, tc0 = summarize_fits(base_fits)
    m1, w1, tc1 = summarize_fits(test_fits)
    ks_m = ks_2samp(m0, m1).statistic
    ks_w = ks_2samp(w0, w1).statistic
    ks_tc = ks_2samp(tc0, tc1).statistic
    print(f"\n{name}")
    print(
        f"  pos_conf: r={r_pos:.4f}  mean|d|={d_pos.mean():.4f}  "
        f"max|d|={d_pos.max():.4f}"
    )
    print(
        f"  neg_conf: r={r_neg:.4f}  mean|d|={d_neg.mean():.4f}  "
        f"max|d|={d_neg.max():.4f}"
    )
    print(f"  KS stats: m={ks_m:.4f}  w={ks_w:.4f}  tc_scaled={ks_tc:.4f}")
    print(
        f"  success rate: base={np.mean(base_fits[:, 3] != 0):.3f}  "
        f"test={np.mean(test_fits[:, 3] != 0):.3f}"
    )


def main():
    n_obs = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    t, p = load(n_obs)
    print(f"series length: {len(t)}")

    print("running legacy x2 (measures its own run-to-run spread)...")
    start = time.perf_counter()
    leg1_ind, leg1_fits = legacy_run(t, p)
    leg2_ind, leg2_fits = legacy_run(t, p)
    print(f"legacy 2 runs: {time.perf_counter() - start:.0f}s")

    print("running kernel x3 seeds x2 rescale modes...")
    start = time.perf_counter()
    kernel = {
        (rescale, seed): kernel_run(t, p, seed, rescale)
        for rescale in (False, True)
        for seed in (1, 2, 3)
    }
    print(f"kernel 6 runs: {time.perf_counter() - start:.0f}s")

    compare(
        "legacy run2 vs legacy run1 (baseline spread)",
        leg1_ind,
        leg1_fits,
        leg2_ind,
        leg2_fits,
    )
    for rescale in (False, True):
        for seed in (1, 2, 3):
            ind, fits = kernel[(rescale, seed)]
            compare(
                f"kernel rescale={rescale} seed={seed} vs legacy run1",
                leg1_ind,
                leg1_fits,
                ind,
                fits,
            )


if __name__ == "__main__":
    main()
