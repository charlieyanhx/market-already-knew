"""TFM3 study — Track B: daily SSVI fit on the kernel-recomputed strike slices.

Prereg: PREREG.md (repository root). Power-law phi SSVI, global (rho, eta,
gamma) + theta per expiry, dte in [5,120]; butterfly bound eta*(1+|rho|)<=2 via
reparameterization; calendar no-arb by isotonic projection of theta(T); warm start from
t-1 (past info only). Gate B0: reconstruct the 12 Track-A nodes to median |err| <= 1 vp.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

DIR = Path(__file__).resolve().parents[1] / "data"
TENORS = [7, 30, 60, 90]
GAMMA_MAX = 0.995


def ssvi_w(k: np.ndarray, th: float, rho: float, eta: float, gam: float) -> np.ndarray:
    phi = eta * th ** (-gam)
    return 0.5 * th * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))


def unpack(p: np.ndarray, n_exp: int):
    th = np.exp(p[:n_exp])
    rho = np.tanh(p[n_exp])
    eta = 2.0 / (1 + abs(rho)) / (1 + np.exp(-p[n_exp + 1]))     # eta*(1+|rho|) <= 2
    gam = GAMMA_MAX / (1 + np.exp(-p[n_exp + 2]))
    return th, rho, eta, gam


def fit_day(slices: list[tuple[float, np.ndarray, np.ndarray]], p0: np.ndarray | None):
    """slices: [(T_years, k, w_mkt)]; returns (params_vec, theta_arr, rho, eta, gam, rmse)."""
    n = len(slices)
    if p0 is None or len(p0) != n + 3:
        th0 = [max(np.interp(0.0, np.sort(k), w[np.argsort(k)]), 1e-6) for _, k, w in slices]
        p0 = np.r_[np.log(th0), -0.9, 0.0, 0.0]  # rho starts ~ tanh(-0.9) = -0.72 (equity skew)

    def resid(p):
        th, rho, eta, gam = unpack(p, n)
        out = []
        for i, (_, k, w) in enumerate(slices):
            r = ssvi_w(k, th[i], rho, eta, gam) - w
            out.append(r / np.sqrt(len(k)))       # each expiry counts equally
        return np.concatenate(out)

    sol = least_squares(resid, p0, method="lm", max_nfev=4000)
    th, rho, eta, gam = unpack(sol.x, n)
    rmse = float(np.sqrt(np.mean(np.concatenate(
        [(ssvi_w(k, th[i], rho, eta, gam) - w) ** 2 for i, (_, k, w) in enumerate(slices)]))))
    return sol.x, th, rho, eta, gam, rmse


def node_iv_from_ssvi(th: float, tau_y: float, rho: float, eta: float, gam: float,
                      tgt_delta: float) -> float:
    """Invert Black-76 forward delta N(d1) (D~1, carry~0 approx) for the node k."""
    def dgap(k):
        w = ssvi_w(np.array([k]), th, rho, eta, gam)[0]
        s = np.sqrt(max(w, 1e-10) / tau_y)
        d1 = (-k + 0.5 * w) / np.sqrt(max(w, 1e-10))
        d = norm.cdf(d1) if tgt_delta > 0 else norm.cdf(d1) - 1.0
        return d - tgt_delta
    try:
        k = brentq(dgap, -2.0, 2.0, xtol=1e-6)
    except ValueError:
        return np.nan
    w = ssvi_w(np.array([k]), th, rho, eta, gam)[0]
    return float(np.sqrt(w / tau_y))


def main() -> None:
    xs = pd.read_parquet(DIR / "slices_strikes.parquet")
    panel = pd.read_parquet(DIR / "panel_trackA.parquet")
    print(f"strike rows {len(xs):,}, {xs.date.nunique():,} days", flush=True)

    rows, recon_errs = [], []
    p_last: np.ndarray | None = None
    for dt, g in xs.groupby("date", sort=True):
        slices = []
        for ex, s in g.groupby("expiration"):
            if len(s) < 6 or s.k.abs().min() > 0.05:
                continue
            T = s.dte.iloc[0] / 365.0
            k, iv = s.k.to_numpy(), s.iv.to_numpy()
            m = np.abs(k) <= 1.0
            if m.sum() < 6:
                continue
            slices.append((T, k[m], (iv[m] ** 2) * T))
        if len(slices) < 3:
            continue
        slices.sort(key=lambda t: t[0])
        p0 = p_last if p_last is not None and len(p_last) == len(slices) + 3 else None
        try:
            pvec, th, rho, eta, gam, rmse = fit_day(slices, p0)
        except Exception:
            continue
        p_last = pvec
        Ts = np.array([t for t, _, _ in slices])
        th = np.maximum.accumulate(th)            # isotonic calendar projection
        dtes = Ts * 365.0
        rec = {"date": dt, "rho": rho, "eta": eta, "gamma": gam,
               "rmse_w": rmse, "n_slices": len(slices)}
        for tau in TENORS:
            if dtes.min() <= tau <= dtes.max():
                rec[f"th{tau}"] = float(np.interp(tau, dtes, th))
            else:
                rec[f"th{tau}"] = np.nan
        rows.append(rec)
        # B0 reconstruction check on a 10% date subsample
        if hash(dt) % 10 == 0 and dt in panel.index:
            for tau in TENORS:
                if np.isnan(rec[f"th{tau}"]):
                    continue
                ty = tau / 365.0
                for nm, tgt in (("p25", -0.25), ("atm", 0.50), ("c25", 0.25)):
                    ref = panel.loc[dt, f"{nm}_{tau}"]
                    if pd.isna(ref):
                        continue
                    iv = node_iv_from_ssvi(rec[f"th{tau}"], ty, rho, eta, gam, tgt)
                    if np.isfinite(iv):
                        recon_errs.append(abs(iv - ref))
        if len(rows) % 500 == 0:
            print(f"  {len(rows):,} days fit; last {dt:%Y-%m-%d} rmse_w {rmse:.2e}", flush=True)

    F = pd.DataFrame(rows).set_index("date").sort_index()
    F.to_parquet(DIR / "params_trackB.parquet")
    re_ = np.array(recon_errs)
    print(f"\nSSVI fits: {len(F):,} days {F.index.min():%Y-%m-%d} -> {F.index.max():%Y-%m-%d}")
    print(f"  rmse_w median {F.rmse_w.median():.2e}; rho median {F.rho.median():+.3f}; "
          f"eta median {F.eta.median():.3f}; gamma median {F.gamma.median():.3f}")
    for tau in TENORS:
        print(f"  th{tau}: avail {F[f'th{tau}'].notna().mean():.1%}")
    print(f"GATE B0 — node reconstruction |err| (n={len(re_):,}): "
          f"median {np.median(re_)*100:.2f} vp, p90 {np.percentile(re_, 90)*100:.2f} vp "
          f"-> {'PASS' if np.median(re_) <= 0.01 else 'FAIL'} @ 1.0 vp")


if __name__ == "__main__":
    main()
