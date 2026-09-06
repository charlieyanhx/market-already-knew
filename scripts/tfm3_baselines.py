"""TFM3 study — walk-forward baselines (models 5-9): RW, AR1, HAR, PCA-VAR, FWD.

Prereg: PREREG.md (repository root). Rolling 756d refit every step;
point forecasts wrapped into 9 quantiles with each model's own trailing 252 resolved
h-step errors (past-only; err at origin s resolves at s+h, so origin t may use errors
from origins <= t-h, implemented as shift(h) before the rolling window). All series are
log-IV (Track A) or transformed SSVI params (Track B). No fit ever sees post-origin data.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tfm3_common import DIR, FIT_WIN, HS, QS  # noqa: E402

ERR_WIN, ERR_MIN = 252, 200


def rw_points(y: pd.DataFrame, h: int) -> pd.DataFrame:
    return y.copy()


def ar1_points(y: pd.DataFrame, h: int) -> pd.DataFrame:
    dy = y.diff()
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for j, col in enumerate(y.columns):
        d = dy[col].to_numpy()
        lvl = y[col].to_numpy()
        for i in range(FIT_WIN, len(y)):
            w = d[i - FIT_WIN + 1:i + 1]
            m = np.isfinite(w[1:]) & np.isfinite(w[:-1])
            if m.sum() < 300 or not np.isfinite(lvl[i]) or not np.isfinite(d[i]):
                continue
            a = np.c_[np.ones(m.sum()), w[:-1][m]]
            (c, phi), *_ = np.linalg.lstsq(a, w[1:][m], rcond=None)
            step, acc = d[i], 0.0
            for _ in range(h):
                step = c + phi * step
                acc += step
            out.iloc[i, j] = lvl[i] + acc
    return out


def har_points(y: pd.DataFrame, h: int) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    yw = y.rolling(5, min_periods=4).mean()
    ym = y.rolling(22, min_periods=18).mean()
    for j, col in enumerate(y.columns):
        Y, W, M = y[col].to_numpy(), yw[col].to_numpy(), ym[col].to_numpy()
        for i in range(FIT_WIN, len(y)):
            s0 = i - FIT_WIN + 1
            yy = Y[s0 + h:i + 1]                      # targets y_{s+h}, s in [s0, i-h]
            x1, x2, x3 = Y[s0:i + 1 - h], W[s0:i + 1 - h], M[s0:i + 1 - h]
            m = np.isfinite(yy) & np.isfinite(x1) & np.isfinite(x2) & np.isfinite(x3)
            if m.sum() < 300 or not (np.isfinite(Y[i]) and np.isfinite(W[i]) and np.isfinite(M[i])):
                continue
            a = np.c_[np.ones(m.sum()), x1[m], x2[m], x3[m]]
            b, *_ = np.linalg.lstsq(a, yy[m], rcond=None)
            out.iloc[i, j] = b[0] + b[1] * Y[i] + b[2] * W[i] + b[3] * M[i]
    return out


def pcavar_points(y: pd.DataFrame, h: int, n_fac: int = 3) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    Yv = y.to_numpy()
    for i in range(FIT_WIN, len(y)):
        W = Yv[i - FIT_WIN + 1:i + 1]
        rows = np.isfinite(W).all(axis=1)
        if rows.sum() < 500 or not np.isfinite(Yv[i]).all():
            continue
        Wc = W[rows]
        mu = Wc.mean(axis=0)
        _, _, vt = np.linalg.svd(Wc - mu, full_matrices=False)
        L = vt[:n_fac].T                              # (n_series, n_fac)
        f = (Wc - mu) @ L
        A = np.c_[np.ones(len(f) - 1), f[:-1]]
        B, *_ = np.linalg.lstsq(A, f[1:], rcond=None)  # (1+n_fac, n_fac)
        fc = (Yv[i] - mu) @ L
        for _ in range(h):
            fc = B[0] + fc @ B[1:]
        out.iloc[i] = mu + fc @ L.T
    return out


def fwd_points(y: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward-curve-implied expected future ATM IV (martingale forward variance).

    ATM nodes only; wings have no market-implied forecast. Curve from the per-expiry
    ATM slices (no extrapolation: NaN unless [h_c, h_c+tau] is inside the quoted range).
    """
    S = pd.read_parquet(DIR / "slices_atm.parquet")
    S = S[S.atm.notna() & (S.dte >= 1)]
    curves = {dt: (g.dte.to_numpy(float), (g.atm.to_numpy() ** 2) * g.dte.to_numpy() / 365.0)
              for dt, g in S.sort_values("dte").groupby("date")}
    h_c = int(round(h * 7 / 5))
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    atm_cols = [c for c in y.columns if c.startswith("atm_")]
    for dt in y.index:
        cv = curves.get(dt)
        if cv is None or len(cv[0]) < 3:
            continue
        dte, w = cv
        order = np.argsort(dte)
        dte, w = dte[order], np.maximum.accumulate(w[order])   # enforce monotone total var
        dte, w = np.r_[0.0, dte], np.r_[0.0, w]   # V(0)=0 boundary condition, not extrapolation
        for c in atm_cols:
            tau = int(c.split("_")[1])
            if dte[0] <= h_c and dte[-1] >= h_c + tau:
                v0 = np.interp(h_c, dte, w)
                v1 = np.interp(h_c + tau, dte, w)
                fv = (v1 - v0) / (tau / 365.0)
                if fv > 1e-8:
                    out.loc[dt, c] = np.log(np.sqrt(fv))
    return out


def wrap_quantiles(point: pd.DataFrame, y: pd.DataFrame, h: int) -> np.ndarray:
    """(n_dates, n_series, 9) quantile forecasts = point + trailing empirical error qs."""
    realized = y.shift(-h)
    err = realized - point                             # indexed by origin, resolves at +h
    q = np.full((len(y), y.shape[1], len(QS)), np.nan)
    for j, col in enumerate(y.columns):
        e = err[col].shift(h)                          # usable at origin t: origins <= t-h
        roll = e.rolling(ERR_WIN, min_periods=ERR_MIN)
        for qi, qq in enumerate(QS):
            q[:, j, qi] = point[col].to_numpy() + roll.quantile(qq).to_numpy()
    return q


def run_track(name: str, y: pd.DataFrame) -> None:
    print(f"\n=== track {name}: {y.shape[1]} series, {len(y):,} days "
          f"{y.index.min():%Y-%m-%d} -> {y.index.max():%Y-%m-%d}", flush=True)
    models = {"rw": rw_points, "ar1": ar1_points, "har": har_points, "pcavar": pcavar_points}
    if name == "trackA":
        models["fwd"] = fwd_points
    for h in HS:
        for mname, fn in models.items():
            t0 = time.time()
            pf = fn(y, h)
            q = wrap_quantiles(pf, y, h)
            np.savez_compressed(
                DIR / f"fc_{name}_{mname}_h{h}.npz",
                dates=np.asarray(y.index, dtype="datetime64[ns]").astype("int64"),
                cols=np.array(y.columns, dtype=object),
                point=pf.to_numpy(), quant=q, qs=QS)
            n_ok = int(np.isfinite(q).all(axis=2).sum())
            print(f"  {mname:7s} h={h}: {time.time()-t0:6.1f}s, "
                  f"{n_ok:,} (origin,series) forecasts with full quantiles", flush=True)


if __name__ == "__main__":
    from tfm3_common import load_trackA, load_trackB
    run_track("trackA", load_trackA())
    if (DIR / "params_trackB.parquet").exists():
        run_track("trackB", load_trackB())
    else:
        print("params_trackB.parquet not yet built — Track B skipped this run")
