"""Spot-free option pricing utilities used by the panel builder (paper Section 2).

Vendor greeks in the free EOD source fail internal-consistency checks, so implied
volatility and delta are recomputed from quoted prices alone:

  - implied_carry:   (r - q) per date from the ratio of put-call-parity forwards across
                     two expiries. The ratio cancels the spot price, so nothing depends
                     on the end-of-day timestamp mismatch between option and stock closes.
  - implied_vol_b76: Black-76 implied volatility per strike by vectorised bisection,
                     priced off the parity forward, taken from the OTM leg.
  - greeks_b76:      spot delta without a spot price: dC/dS = D*N(d1)*(F/S), with
                     F/S = exp(carry*T); delta_C - delta_P = D*exp(carry*T) holds by
                     construction.

These functions are extracted verbatim from the research codebase's shared pricing
kernel; the discount-factor/forward fit itself (`parity_surface_study`) lives in
tfm3_surface_panel.py because this study widens its sanity bound (paper Appendix A,
amendment A1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

__all__ = ["implied_carry", "implied_vol_b76", "greeks_b76"]


def implied_carry(d: pd.DataFrame, lo: int = 15, hi: int = 120) -> pd.Series:
    """(r - q) per date, from the ratio of parity forwards across two expiries."""
    x = d[d.dte.between(lo, hi) & (d.bid > 0.01)].copy()
    piv = x.pivot_table(index=["date", "expiration", "strike"],
                        columns="is_put", values="mid").dropna()
    if piv.empty:
        return pd.Series(dtype=float)
    k = piv.index.get_level_values("strike")
    fwd = (piv[False] - piv[True] + k).rename("f")          # C - P + K
    rel = ((piv[False] - piv[True]).abs() / k)
    f = fwd[rel < 0.10].reset_index()
    f = (f.groupby(["date", "expiration"])
         .agg(f=("f", "median"), n=("f", "size")).reset_index())
    f = f[f.n >= 3]
    f["dte"] = (f.expiration - f.date).dt.days
    near = f[f.dte.between(lo, 45)].sort_values("dte").groupby("date").first()
    far = f[f.dte.between(60, hi)].sort_values("dte").groupby("date").first()
    j = near.join(far, lsuffix="_n", rsuffix="_f", how="inner")
    dt = (j.dte_f - j.dte_n) / 365.0
    c = np.log(j.f_f / j.f_n) / dt
    return c[(dt > 0.08) & c.abs().lt(0.25)].rename("carry")


def _b76(F, K, T, s, is_put, D):
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    d2 = d1 - v
    c = D * (F * _norm.cdf(d1) - K * _norm.cdf(d2))
    return np.where(is_put, c - D * (F - K), c)


def implied_vol_b76(px, F, K, T, D, is_put, lo=0.01, hi=4.0, iters=60):
    """Vectorised bisection. Robust where Newton is not, and fast enough at this scale."""
    a = np.full_like(px, lo, dtype=float)
    b = np.full_like(px, hi, dtype=float)
    for _ in range(iters):
        m = 0.5 * (a + b)
        up = _b76(F, K, T, m, is_put, D) < px
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    s = 0.5 * (a + b)
    return np.where((s > lo * 1.01) & (s < hi * 0.99), s, np.nan)


def greeks_b76(F, K, T, D, s, is_put, carry):
    """Spot delta WITHOUT a spot price: dC/dS = D*N(d1)*(F/S), and F/S = exp(carry*T)."""
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    fs = np.exp(carry * T)
    dfwd = np.where(is_put, _norm.cdf(d1) - 1.0, _norm.cdf(d1))
    return D * dfwd * fs, D * F * _norm.pdf(d1) * np.sqrt(T) / 100.0   # delta, vega/1volpt
