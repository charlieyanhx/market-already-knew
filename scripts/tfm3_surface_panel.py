"""TFM3 study — Track A panel: 12-node constant-maturity SPY IV surface, 2008-2025.

Prereg: PREREG.md (repository root). Nodes {p25, atm, c25} x {7,30,60,90}d
from the free EOD chains with kernel-recomputed IV/delta (parity forward + Black-76 OTM
leg). Validation-only extras p10_30/c10_30 for the RR10 gate. No extrapolation anywhere.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from parity_b76 import greeks_b76, implied_carry, implied_vol_b76  # noqa: E402


def parity_surface_study(d: pd.DataFrame, min_pairs: int = 5) -> pd.DataFrame:
    """Study-local fork of kernel.parity_surface with a wider D bound (0.85, 1.06].

    The free EOD source carries a systematic put/call quote shift that biases the fitted
    discount factor ~+2-4% high at long DTE; kernel's (0.90, 1.02] cap silently drops
    100-140 DTE slices in the high-rate era (2023+: fitted D ~1.023), starving the 90d
    node of its upper bracket. Near-dated slices carry the same artifact level and still
    calibrated at 0.99+ corr vs the owned grid, so admitting long slices with it is
    consistent with what the gate validated. kernel.py itself is untouched.
    """
    import numpy as _np
    x = d[d.bid > 0.02]
    piv = x.pivot_table(index=["date", "expiration", "strike"],
                        columns="is_put", values="mid").dropna()
    if piv.empty:
        return pd.DataFrame()
    cp = (piv[False] - piv[True]).rename("cp").reset_index()
    out = []
    for (dt, ex), g in cp.groupby(["date", "expiration"]):
        if len(g) < min_pairs:
            continue
        k = g.strike.to_numpy()
        y = g.cp.to_numpy()
        m = _np.abs(y) < 0.35 * _np.median(k)
        if m.sum() < min_pairs:
            continue
        A = _np.c_[_np.ones(m.sum()), k[m]]
        b, *_ = _np.linalg.lstsq(A, y[m], rcond=None)
        D = -b[1]
        if not (0.85 < D <= 1.06):
            continue
        out.append((dt, ex, D, b[0] / D))
    return pd.DataFrame(out, columns=["date", "expiration", "D", "F"])

# raw free EOD chains — see data/README.md for the download; not redistributed here
SRC = str(Path(__file__).resolve().parents[1] / "data/raw/SPY_options.parquet")
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "panel_trackA.parquet"

TENORS = [7, 30, 60, 90]
DTE_MAX = 140          # need expiries beyond 90 to bracket the 90d node
BRACKET_MAX = 100      # widest allowed dte gap between bracketing expiries
NODES = ["p25", "atm", "c25"]

print("loading free chains ...", flush=True)
o = pd.read_parquet(SRC, columns=["date", "expiration", "strike", "type", "bid", "ask"])
o["date"] = pd.to_datetime(o.date)
o["expiration"] = pd.to_datetime(o.expiration)
o["dte"] = (o.expiration - o.date).dt.days
n0 = len(o)
o = o[(o.bid > 0) & (o.ask > o.bid) & o.dte.between(1, DTE_MAX)]
o["mid"] = (o.bid + o.ask) / 2
o["is_put"] = o.type.eq("put")
print(f"  rows {n0:,} -> {len(o):,} after bid/ask/dte filter; "
      f"{o.date.nunique():,} days {o.date.min():%Y-%m-%d} -> {o.date.max():%Y-%m-%d}", flush=True)

print("parity surface (D, F per date x expiry) ...", flush=True)
pf = parity_surface_study(o)
print(f"  {len(pf):,} (date,expiry) slices with valid parity fit", flush=True)

carry = implied_carry(o)
print(f"  carry: {len(carry):,} days, median {carry.median():+.2%}", flush=True)

# one row per (date, expiry, strike): the OTM leg, priced off the parity forward
x = o.merge(pf, on=["date", "expiration"])
x = x[(~x.is_put & (x.strike >= x.F)) | (x.is_put & (x.strike < x.F))]
x["T"] = x.dte / 365.0
x["iv"] = implied_vol_b76(x.mid.to_numpy(), x.F.to_numpy(), x.strike.to_numpy(),
                          x["T"].to_numpy(), x.D.to_numpy(), x.is_put.to_numpy())
x = x[x.iv.between(0.01, 3.0)]
x["carry"] = x.date.map(carry).fillna(0.0)
d_otm, _ = greeks_b76(x.F.to_numpy(), x.strike.to_numpy(), x["T"].to_numpy(),
                      x.D.to_numpy(), x.iv.to_numpy(), x.is_put.to_numpy(),
                      x.carry.to_numpy())
# both-side deltas at every strike: delta_C - delta_P = D*exp(carry*T) by construction
span = x.D.to_numpy() * np.exp(x.carry.to_numpy() * x["T"].to_numpy())
x["d_call"] = np.where(x.is_put, d_otm + span, d_otm)
x["d_put"] = x.d_call - span
print(f"  {len(x):,} OTM-leg strikes with valid IV", flush=True)

# strike-level slices for the Track-B SSVI fit (dte 5-120, k = ln(K/F))
xs = x[x.dte.between(5, 120)][["date", "expiration", "dte", "strike", "F", "D", "iv"]].copy()
xs["k"] = np.log(xs.strike / xs.F)
xs.to_parquet(OUT_DIR / "slices_strikes.parquet")
print(f"  saved {len(xs):,} strike rows -> slices_strikes.parquet", flush=True)


def node_interp(deltas: np.ndarray, ivs: np.ndarray, tgt: float) -> float:
    m = (np.abs(deltas) > 0.02) & (np.abs(deltas) < 0.98)
    if m.sum() < 4:
        return np.nan
    d, v = deltas[m], ivs[m]
    if not (d.min() < tgt < d.max()):
        return np.nan
    idx = np.argsort(d)
    return float(np.interp(tgt, d[idx], v[idx]))


print("per-expiry delta nodes ...", flush=True)
slices = []
for (dt, ex), g in x.groupby(["date", "expiration"], sort=True):
    dp, dc, iv = g.d_put.to_numpy(), g.d_call.to_numpy(), g.iv.to_numpy()
    a_p, a_c = node_interp(dp, iv, -0.50), node_interp(dc, iv, 0.50)
    rec = {"date": dt, "dte": int(g.dte.iloc[0]),
           "p25": node_interp(dp, iv, -0.25), "c25": node_interp(dc, iv, 0.25),
           "atm": np.nanmean([a_p, a_c]) if not (np.isnan(a_p) and np.isnan(a_c)) else np.nan,
           "p10": node_interp(dp, iv, -0.10), "c10": node_interp(dc, iv, 0.10)}
    slices.append(rec)
S = pd.DataFrame(slices)
S.to_parquet(OUT_DIR / "slices_atm.parquet")   # per-expiry ATM curve, for the FWD baseline
print(f"  {len(S):,} expiry slices; node availability "
      + " ".join(f"{c}:{S[c].notna().mean():.0%}" for c in ["p25", "atm", "c25", "p10", "c10"]),
      flush=True)


def cm_interp(sub: pd.DataFrame, col: str, tau: int) -> float:
    s = sub[sub[col].notna()]
    if s.empty:
        return np.nan
    exact = s[s.dte == tau]
    if len(exact):
        return float(exact[col].iloc[0])
    lo, hi = s[s.dte < tau], s[s.dte > tau]
    if lo.empty or hi.empty:
        return np.nan
    a, b = lo.iloc[-1], hi.iloc[0]
    if b.dte - a.dte > BRACKET_MAX:
        return np.nan
    wa, wb = a[col] ** 2 * a.dte / 365.0, b[col] ** 2 * b.dte / 365.0
    w = wa + (wb - wa) * (tau - a.dte) / (b.dte - a.dte)
    return float(np.sqrt(max(w, 1e-8) / (tau / 365.0)))


print("constant-maturity tenor interpolation ...", flush=True)
rows = []
for dt, sub in S.sort_values("dte").groupby("date", sort=True):
    rec = {"date": dt}
    for tau in TENORS:
        for c in NODES:
            rec[f"{c}_{tau}"] = cm_interp(sub, c, tau)
    for c in ("p10", "c10"):                       # validation-only, 30d
        rec[f"{c}_30"] = cm_interp(sub, c, 30)
    rows.append(rec)
P = pd.DataFrame(rows).set_index("date").sort_index()
P.to_parquet(OUT)

cols12 = [f"{c}_{t}" for t in TENORS for c in NODES]
print(f"\nPANEL -> {OUT}: {len(P):,} days {P.index.min():%Y-%m-%d} -> {P.index.max():%Y-%m-%d}")
print("coverage (non-NaN share) per node:")
for t in TENORS:
    print("  " + "  ".join(f"{c}_{t}: {P[f'{c}_{t}'].notna().mean():6.1%}" for c in NODES))
comp = P[cols12].notna().all(axis=1)
print(f"all-12-node-complete days: {comp.sum():,} ({comp.mean():.1%}); "
      f"first complete: {P.index[comp.argmax()]:%Y-%m-%d}" if comp.any() else "NONE")
yr = P.assign(ok=comp).groupby(P.index.year).ok.sum()
print("complete days by year: " + " ".join(f"{y}:{int(v)}" for y, v in yr.items()))

# ------------------------------------------------------------- CALIBRATION GATE
print("\n" + "=" * 88)
print("CALIBRATION GATE vs proprietary CBOE 16:15 reference grid (not redistributed)")
print("=" * 88)
REF = Path(__file__).resolve().parents[1] / "data/raw/surface_grid_eod.parquet"
if not REF.exists():
    print("reference 16:15 CBOE grid is proprietary and not redistributed; the gate\n"
          "numbers it produced are reported in the paper (Section 2) and PREREG.md.")
    raise SystemExit(0)
g = pd.read_parquet(REF)
ref = pd.DataFrame({"atm_30_ref": (g.P50_30 + g.C50_30) / 2,
                    "rr25_30_ref": g.P25_30 - g.C25_30,
                    "rr10_30_ref": g.P10_30 - g.C10_30,
                    "atm_60_ref": (g.P50_60 + g.C50_60) / 2,
                    "rr25_60_ref": g.P25_60 - g.C25_60})
mine = pd.DataFrame({"atm_30": P.atm_30, "rr25_30": P.p25_30 - P.c25_30,
                     "rr10_30": P.p10_30 - P.c10_30,
                     "atm_60": P.atm_60, "rr25_60": P.p25_60 - P.c25_60})
j = mine.join(ref, how="inner").dropna()
print(f"overlap {len(j):,} days {j.index.min():%Y-%m-%d} -> {j.index.max():%Y-%m-%d}")
for a in ["atm_30", "rr25_30", "rr10_30", "atm_60", "rr25_60"]:
    r = a + "_ref"
    lc = j[a].corr(j[r])
    dj = np.log(j[[a, r]].clip(lower=1e-4)).diff().dropna() if a.startswith("atm") \
        else j[[a, r]].diff().dropna()
    cc = dj[a].corr(dj[r])
    bias = (j[a] - j[r]).mean() * 100
    print(f"  {a:8s} level corr {lc:+.4f} ({'PASS' if lc > 0.95 else 'FAIL'} @0.95) | "
          f"1d-change corr {cc:+.4f} | mean diff {bias:+5.2f} vp | sd {(j[a]-j[r]).std()*100:4.2f} vp")
