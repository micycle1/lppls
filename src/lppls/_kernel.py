"""Numba-native LPPLS fitting kernel.

Everything in this module is @njit(cache=True) and depends only on
numpy/numba. It replaces the scipy.optimize.minimize(Nelder-Mead) hot path
with a fused objective (basis sums + 4x4 Cholesky solve + SSE in one pass)
and a hand-rolled Nelder-Mead, so a full fit never leaves compiled code.

The semantics deliberately mirror LPPLS.fit / estimate_params /
func_restricted / matrix_equation in lppls.py: random restarts with the same
init bounds, retry on non-convergence or non-finite O/D, first successful
search wins, all-zeros on exhaustion.

fastmath is intentionally OFF: inf/NaN are load-bearing for the retry logic.
"""

import math

import numpy as np
from numba import njit, prange

# Column layout of a result row (one row per fitted window).
COL_TC = 0
COL_M = 1
COL_W = 2
COL_A = 3
COL_B = 4
COL_C = 5
COL_C1 = 6
COL_C2 = 7
COL_O = 8
COL_D = 9
COL_T1 = 10
COL_T2 = 11
COL_SSE = 12
COL_SUCCESS = 13
N_COLS = 14

# splitmix64 constants (kept as uint64 so numba never promotes to float64)
_U_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_U_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_U_MIX2 = np.uint64(0x94D049BB133111EB)
_U30 = np.uint64(30)
_U27 = np.uint64(27)
_U31 = np.uint64(31)
_U11 = np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0  # 2**-53


def derive_seeds(base_seed, n_jobs):
    """Per-job uint64 seeds from a base seed (pure Python, exact uint64 math).

    Seeds depend only on the global job index, so results are identical
    regardless of thread count or how the job list is chunked.
    """
    mask = (1 << 64) - 1
    golden = 0x9E3779B97F4A7C15
    seeds = np.empty(n_jobs, dtype=np.uint64)
    for j in range(n_jobs):
        z = (base_seed ^ ((j + 1) * golden)) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        seeds[j] = (z ^ (z >> 31)) & mask
    return seeds


@njit(cache=True)
def _sm64_next(state):
    """splitmix64 step: returns (new_state, uniform double in [0, 1))."""
    state = state + _U_GOLDEN
    z = state
    z = (z ^ (z >> _U30)) * _U_MIX1
    z = (z ^ (z >> _U27)) * _U_MIX2
    z = z ^ (z >> _U31)
    return state, np.float64(z >> _U11) * _INV53


@njit(cache=True)
def _lppls_sse(t, p, tc, m, w, fi, gi, hi, beta):
    """Fused LPPLS objective: solve the linear params and return the SSE.

    Equivalent to matrix_equation + lppls + residual sum in lppls.py, in a
    single pass. On success, beta holds (a, b, c1, c2). Any numerical failure
    (overflow, non-SPD normal equations) returns +inf, which the optimizer
    treats as a worst-case point — the compiled equivalent of the
    LinAlgError/overflow exception path.

    fi/gi/hi are caller-provided scratch (len >= n); they cache the basis so
    the residual pass needs no second transcendental sweep.
    """
    n = t.shape[0]

    s_f = 0.0
    s_g = 0.0
    s_h = 0.0
    s_ff = 0.0
    s_gg = 0.0
    s_hh = 0.0
    s_fg = 0.0
    s_fh = 0.0
    s_gh = 0.0
    s_y = 0.0
    s_yf = 0.0
    s_yg = 0.0
    s_yh = 0.0

    for i in range(n):
        dt = abs(tc - t[i]) + 1e-8
        lg = math.log(dt)
        f = math.exp(m * lg)  # dt**m, reusing lg
        ph = w * lg
        g = f * math.cos(ph)
        h = f * math.sin(ph)
        fi[i] = f
        gi[i] = g
        hi[i] = h
        y = p[i]
        s_f += f
        s_g += g
        s_h += h
        s_ff += f * f
        s_gg += g * g
        s_hh += h * h
        s_fg += f * g
        s_fh += f * h
        s_gh += g * h
        s_y += y
        s_yf += y * f
        s_yg += y * g
        s_yh += y * h

    # inf + (-inf) -> nan, inf + inf -> inf: a single combined check catches
    # any non-finite accumulator.
    total = s_f + s_g + s_h + s_ff + s_gg + s_hh + s_fg + s_fh + s_gh
    if not math.isfinite(total + s_y + s_yf + s_yg + s_yh):
        return np.inf

    # Normal equations A @ beta = rhs with the same 1e-8 ridge as
    # matrix_equation. A = X^T X + 1e-8 I is SPD, so use an unrolled 4x4
    # Cholesky (no LAPACK call, no allocation). A non-positive pivot is the
    # compiled equivalent of LinAlgError.
    a00 = n + 1e-8
    a10 = s_f
    a20 = s_g
    a30 = s_h
    a11 = s_ff + 1e-8
    a21 = s_fg
    a31 = s_fh
    a22 = s_gg + 1e-8
    a32 = s_gh
    a33 = s_hh + 1e-8

    if a00 <= 0.0:
        return np.inf
    l00 = math.sqrt(a00)
    l10 = a10 / l00
    l20 = a20 / l00
    l30 = a30 / l00

    d1 = a11 - l10 * l10
    if d1 <= 0.0 or not math.isfinite(d1):
        return np.inf
    l11 = math.sqrt(d1)
    l21 = (a21 - l20 * l10) / l11
    l31 = (a31 - l30 * l10) / l11

    d2 = a22 - l20 * l20 - l21 * l21
    if d2 <= 0.0 or not math.isfinite(d2):
        return np.inf
    l22 = math.sqrt(d2)
    l32 = (a32 - l30 * l20 - l31 * l21) / l22

    d3 = a33 - l30 * l30 - l31 * l31 - l32 * l32
    if d3 <= 0.0 or not math.isfinite(d3):
        return np.inf
    l33 = math.sqrt(d3)

    # forward solve L y = rhs
    y0 = s_y / l00
    y1 = (s_yf - l10 * y0) / l11
    y2 = (s_yg - l20 * y0 - l21 * y1) / l22
    y3 = (s_yh - l30 * y0 - l31 * y1 - l32 * y2) / l33

    # back solve L^T beta = y
    b3 = y3 / l33
    b2 = (y2 - l32 * b3) / l22
    b1 = (y1 - l21 * b2 - l31 * b3) / l11
    b0 = (y0 - l10 * b1 - l20 * b2 - l30 * b3) / l00

    beta[0] = b0
    beta[1] = b1
    beta[2] = b2
    beta[3] = b3

    sse = 0.0
    for i in range(n):
        r = b0 + b1 * fi[i] + b2 * gi[i] + b3 * hi[i] - p[i]
        sse += r * r

    if not math.isfinite(sse):
        return np.inf
    return sse


@njit(cache=True)
def _obj(t, p, x, fi, gi, hi, beta):
    """Objective wrapper: NaN maps to +inf so NM ordering stays sane."""
    v = _lppls_sse(t, p, x[0], x[1], x[2], fi, gi, hi, beta)
    if v != v:  # NaN
        return np.inf
    return v


@njit(cache=True)
def _nm_sort(sim, fsim):
    """Stable insertion sort of the 4 simplex vertices by objective value."""
    for i in range(1, 4):
        fv = fsim[i]
        v0 = sim[i, 0]
        v1 = sim[i, 1]
        v2 = sim[i, 2]
        j = i - 1
        while j >= 0 and fsim[j] > fv:
            fsim[j + 1] = fsim[j]
            sim[j + 1, 0] = sim[j, 0]
            sim[j + 1, 1] = sim[j, 1]
            sim[j + 1, 2] = sim[j, 2]
            j -= 1
        fsim[j + 1] = fv
        sim[j + 1, 0] = v0
        sim[j + 1, 1] = v1
        sim[j + 1, 2] = v2


@njit(cache=True)
def _nm_minimize(
    t, p, fi, gi, hi, beta, sim, fsim, xbar, xr, xtr, x0, xatol, fatol, maxiter, maxfev
):
    """Nelder-Mead on (tc, m, w), faithful to scipy's _minimize_neldermead.

    Standard (non-adaptive) coefficients, scipy's default initial simplex
    (5% multiplicative / 0.00025 absolute steps) and termination tests.
    Returns (converged, nfev); the best vertex ends in sim[0]/fsim[0].
    converged=False corresponds to scipy success=False -> caller retries.
    """
    rho = 1.0
    chi = 2.0
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025

    for k in range(3):
        sim[0, k] = x0[k]
    for k in range(3):
        for j in range(3):
            sim[k + 1, j] = x0[j]
        if x0[k] != 0.0:
            sim[k + 1, k] = x0[k] * (1.0 + nonzdelt)
        else:
            sim[k + 1, k] = zdelt

    for i in range(4):
        fsim[i] = _obj(t, p, sim[i], fi, gi, hi, beta)
    nfev = 4
    _nm_sort(sim, fsim)

    iterations = 1
    while nfev < maxfev and iterations < maxiter:
        # convergence test (scipy: max |sim[1:]-sim[0]| and max |fsim[0]-fsim[1:]|)
        xmax = 0.0
        fmax = 0.0
        for i in range(1, 4):
            for j in range(3):
                d = abs(sim[i, j] - sim[0, j])
                if d > xmax:
                    xmax = d
            df = abs(fsim[0] - fsim[i])
            if df > fmax:
                fmax = df
        if xmax <= xatol and fmax <= fatol:
            return True, nfev

        for j in range(3):
            xbar[j] = (sim[0, j] + sim[1, j] + sim[2, j]) / 3.0
        for j in range(3):
            xr[j] = (1.0 + rho) * xbar[j] - rho * sim[3, j]
        fxr = _obj(t, p, xr, fi, gi, hi, beta)
        nfev += 1
        doshrink = False

        if fxr < fsim[0]:
            for j in range(3):
                xtr[j] = (1.0 + rho * chi) * xbar[j] - rho * chi * sim[3, j]
            fxe = _obj(t, p, xtr, fi, gi, hi, beta)
            nfev += 1
            if fxe < fxr:
                for j in range(3):
                    sim[3, j] = xtr[j]
                fsim[3] = fxe
            else:
                for j in range(3):
                    sim[3, j] = xr[j]
                fsim[3] = fxr
        else:
            if fxr < fsim[2]:
                for j in range(3):
                    sim[3, j] = xr[j]
                fsim[3] = fxr
            else:
                if fxr < fsim[3]:
                    # contraction
                    for j in range(3):
                        xtr[j] = (1.0 + psi * rho) * xbar[j] - psi * rho * sim[3, j]
                    fxc = _obj(t, p, xtr, fi, gi, hi, beta)
                    nfev += 1
                    if fxc <= fxr:
                        for j in range(3):
                            sim[3, j] = xtr[j]
                        fsim[3] = fxc
                    else:
                        doshrink = True
                else:
                    # inside contraction
                    for j in range(3):
                        xtr[j] = (1.0 - psi) * xbar[j] + psi * sim[3, j]
                    fxcc = _obj(t, p, xtr, fi, gi, hi, beta)
                    nfev += 1
                    if fxcc < fsim[3]:
                        for j in range(3):
                            sim[3, j] = xtr[j]
                        fsim[3] = fxcc
                    else:
                        doshrink = True
                if doshrink:
                    for i in range(1, 4):
                        for j in range(3):
                            sim[i, j] = sim[0, j] + sigma * (sim[i, j] - sim[0, j])
                        fsim[i] = _obj(t, p, sim[i], fi, gi, hi, beta)
                        nfev += 1

        _nm_sort(sim, fsim)
        iterations += 1

    # ran out of iterations/evaluations: scipy reports success=False
    return False, nfev


@njit(cache=True)
def fit_window(t, p, i0, i1, max_searches, rescale, state, out_row):
    """Fit one window [i0, i1) with random restarts; write a result row.

    Mirrors LPPLS.fit: random (tc, m, w) inits within the same bounds, retry
    on NM non-convergence / numerical failure / non-finite O or D, first
    success wins, zeros (success=0) on exhaustion.

    With rescale=True the optimizer works on tau = (t - t1)/(t2 - t1) (m and
    w are invariant; tc init band becomes [0.8, 1.2]); the linear params are
    then re-solved once on the raw t axis so all outputs stay in original
    units, exactly like estimate_params re-calling matrix_equation at the
    optimum.
    """
    n = i1 - i0
    tw_raw = t[i0:i1]
    pw = p[i0:i1]
    t1 = tw_raw[0]
    t2 = tw_raw[n - 1]
    s = t2 - t1

    out_row[COL_T1] = t1
    out_row[COL_T2] = t2

    # per-job scratch (never allocated inside the NM loop)
    fi = np.empty(n, dtype=np.float64)
    gi = np.empty(n, dtype=np.float64)
    hi = np.empty(n, dtype=np.float64)
    beta = np.empty(4, dtype=np.float64)
    sim = np.empty((4, 3), dtype=np.float64)
    fsim = np.empty(4, dtype=np.float64)
    xbar = np.empty(3, dtype=np.float64)
    xr = np.empty(3, dtype=np.float64)
    xtr = np.empty(3, dtype=np.float64)
    x0 = np.empty(3, dtype=np.float64)

    if rescale and s > 0.0:
        tw = (tw_raw - t1) / s
        lo_tc = 0.8
        hi_tc = 1.2
    else:
        tw = tw_raw.copy()
        lo_tc = t2 - 0.2 * s
        hi_tc = t2 + 0.2 * s

    for _search in range(max_searches):
        state, u1 = _sm64_next(state)
        state, u2 = _sm64_next(state)
        state, u3 = _sm64_next(state)
        x0[0] = lo_tc + u1 * (hi_tc - lo_tc)
        x0[1] = 0.1 + u2 * 0.9
        x0[2] = 6.0 + u3 * 7.0

        converged, _nfev = _nm_minimize(
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
            x0,
            1e-4,
            1e-4,
            600,
            600,
        )
        if not converged:
            continue

        tc = sim[0, 0]
        m = sim[0, 1]
        w = sim[0, 2]
        if rescale and s > 0.0:
            tc = t1 + s * tc

        # re-solve the linear params in original units at the optimum
        sse = _lppls_sse(tw_raw, pw, tc, m, w, fi, gi, hi, beta)
        if not math.isfinite(sse):
            continue
        a = beta[0]
        b = beta[1]
        c1 = beta[2]
        c2 = beta[3]

        # get_c verbatim (truthiness branch included)
        if c1 != 0.0 and c2 != 0.0:
            c = c1 / math.cos(math.atan(c2 / c1))
        else:
            c = 0.0

        # get_oscillations verbatim
        denom = tc - t2
        if denom != 0.0:
            ratio = (tc - t1) / denom
        else:
            ratio = 0.0
        if ratio <= 0.0:
            O = np.nan
        else:
            O = (w / (2.0 * np.pi)) * math.log(ratio)

        # get_damping verbatim (division by zero -> inf, like numpy floats)
        D = (m * abs(b)) / (w * abs(c)) if w * abs(c) != 0.0 else np.inf

        if not (math.isfinite(O) and math.isfinite(D)):
            continue

        out_row[COL_TC] = tc
        out_row[COL_M] = m
        out_row[COL_W] = w
        out_row[COL_A] = a
        out_row[COL_B] = b
        out_row[COL_C] = c
        out_row[COL_C1] = c1
        out_row[COL_C2] = c2
        out_row[COL_O] = O
        out_row[COL_D] = D
        out_row[COL_SSE] = sse
        out_row[COL_SUCCESS] = 1.0
        return

    # exhausted: params stay zero, success stays 0 (== legacy all-zeros tuple)


@njit(cache=True, parallel=True)
def batch_fit(t, p, jobs, max_searches, rescale, seeds, out):
    """Fit every job (window) in parallel with numba threads.

    jobs: int64[J, 2] of (start, end_exclusive) into t/p.
    seeds: uint64[J] per-job RNG states (global-index derived).
    out:   float64[J, N_COLS], zero-filled; each job writes only its row.
    """
    for j in prange(jobs.shape[0]):
        fit_window(
            t, p, jobs[j, 0], jobs[j, 1], max_searches, rescale, seeds[j], out[j]
        )
