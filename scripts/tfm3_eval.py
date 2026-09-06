"""TFM3 study — evaluation: pinball/RMSE/MAE/hit/coverage, DM (NW-HAC), Hansen MCS,
rolling past-only Mincer-Zarnowitz recalibration, loss-series archive.

Prereg: PREREG.md (repository root). Evaluation origin set per track x h =
origins where EVERY model has a full 9-quantile forecast, the current level is finite,
and the realized target is finite. Aggregation: per-node loss ratios to PCA-VAR averaged
across nodes (never pooled raw losses); DM/MCS on per-origin surface-mean loss series.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tfm3_common import DIR, HS, QS, load_track  # noqa: E402

MZ_WIN, MZ_MIN = 504, 200
REF = "pcavar"


def nw_se(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    n = len(d)
    s = d @ d / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (d[:-k] @ d[k:]) / n
    return float(np.sqrt(max(s, 1e-18) / n))


def dm_test(la: np.ndarray, lb: np.ndarray, h: int) -> tuple[float, float]:
    """DM stat and 2-sided p for H0: E[la - lb] = 0. Negative stat = A better."""
    from scipy.stats import norm
    d = la - lb
    se = nw_se(d, 2 * (h - 1) + 5)
    t = d.mean() / se
    return float(t), float(2 * norm.sf(abs(t)))


def pinball(realized: np.ndarray, quant: np.ndarray) -> np.ndarray:
    """(T, J) mean pinball over the 9 levels. realized (T,J), quant (T,J,9)."""
    r = realized[:, :, None]
    diff = r - quant
    loss = np.where(diff >= 0, QS[None, None, :] * diff, (QS[None, None, :] - 1) * diff)
    return loss.mean(axis=2)


def mz_recalibrate(point: np.ndarray, realized: np.ndarray, quant: np.ndarray,
                   h: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling past-only MZ: realized_s = a + b*point_s on resolved origins <= t-h.
    Affine map applied to point and all quantiles (re-sorted if b < 0)."""
    T, J = point.shape
    p2, q2 = np.full_like(point, np.nan), np.full_like(quant, np.nan)
    for j in range(J):
        f, r = point[:, j], realized[:, j]
        ok = np.isfinite(f) & np.isfinite(r)
        for t in range(T):
            if not np.isfinite(f[t]):
                continue
            lo = max(0, t - h - MZ_WIN)
            sl = slice(lo, max(lo, t - h + 1))
            m = ok[sl]
            if m.sum() < MZ_MIN:
                continue
            x, y = f[sl][m], r[sl][m]
            vx = x.var()
            if vx < 1e-12:
                a, b = y.mean(), 0.0
            else:
                b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
                a = y.mean() - b * x.mean()
            p2[t, j] = a + b * f[t]
            q2[t, j] = np.sort(a + b * quant[t, j]) if b < 0 else a + b * quant[t, j]
    return p2, q2


def evaluate(track: str, h: int) -> pd.DataFrame | None:
    files = sorted(DIR.glob(f"fc_{track}_*_h{h}.npz"))
    if not files:
        return None
    Y = load_track(track)
    yv = Y.to_numpy()
    realized = np.full_like(yv, np.nan)
    realized[:-h] = yv[h:]
    models = {}
    for f in files:
        z = np.load(f, allow_pickle=True)
        name = f.stem.replace(f"fc_{track}_", "").replace(f"_h{h}", "")
        assert list(z["cols"]) == list(Y.columns), f"column mismatch in {f.name}"
        models[name] = {"point": z["point"], "quant": z["quant"],
                        "wall": float(z["wall_mean"]) if "wall_mean" in z else np.nan}
    J = yv.shape[1]
    atm_js = [j for j, c in enumerate(Y.columns) if c.startswith("atm_")]

    # ---- evaluation mask: FWD is ATM-only by design -> two masks (surface + ATM subset)
    full_models = [m for m in models if m != "fwd"]
    ok = np.isfinite(yv) & np.isfinite(realized)
    for m in full_models:
        ok &= np.isfinite(models[m]["quant"]).all(axis=2)
    ok_atm = ok.copy()
    if "fwd" in models:
        ok_atm &= np.isfinite(models["fwd"]["quant"]).all(axis=2)
        ok_atm[:, [j for j in range(J) if j not in atm_js]] = False
    rows_full = ok.all(axis=1)
    rows_atm = ok_atm[:, atm_js].all(axis=1) if atm_js else np.zeros(len(yv), bool)
    print(f"\n### {track} h={h}: {rows_full.sum():,} surface origins "
          f"({Y.index[rows_full].min():%Y-%m-%d} -> {Y.index[rows_full].max():%Y-%m-%d}), "
          f"{rows_atm.sum():,} ATM-subset origins", flush=True)

    out, arch_losses = [], {}
    for stage in ("raw", "mz"):
        surf, atm_ser, node_means, stage_rows = {}, {}, {}, []
        for m, d in models.items():
            pt, q = d["point"], d["quant"]
            if stage == "mz":
                pt, q = mz_recalibrate(pt, realized, q, h)
            pl = pinball(realized, q)
            dchg = pt - yv
            rchg = realized - yv
            if m != "fwd":
                pl_f = np.where(ok, pl, np.nan)[rows_full]
                surf[m] = np.nanmean(pl_f, axis=1)
                node_means[m] = np.nanmean(pl_f, axis=0)
                rmse = np.sqrt(np.nanmean(np.where(ok, (dchg - rchg) ** 2, np.nan)[rows_full]))
                mae = np.nanmean(np.where(ok, np.abs(dchg - rchg), np.nan)[rows_full])
                nz = ok & (np.abs(dchg) > 1e-10)
                hit = np.nanmean(np.where(nz, np.sign(dchg) == np.sign(rchg), np.nan)[rows_full])
                cov = [float(np.nanmean(np.where(ok[:, :, None], realized[:, :, None] <= q,
                       np.nan)[rows_full][:, :, qi])) for qi in (0, 4, 8)]
                row = {"track": track, "h": h, "stage": stage, "model": m,
                       "pinball": float(np.nanmean(node_means[m])),
                       "rmse_chg": float(rmse), "mae_chg": float(mae),
                       "hit": float(hit) if np.isfinite(hit) else np.nan,
                       "cov10": cov[0], "cov50": cov[1], "cov90": cov[2],
                       "wall_s": d["wall"]}
                out.append(row)
                stage_rows.append(row)
            pl_a = np.where(ok_atm, pl, np.nan)[rows_atm][:, atm_js]
            atm_ser[m] = np.nanmean(pl_a, axis=1)
        arch_losses[stage] = {"surface": surf, "atm": atm_ser}

        # loss ratios vs REF (per-node ratio averaged, never pooled) + DM p-values
        for r in stage_rows:
            m = r["model"]
            r["ratio_vs_ref"] = float(np.nanmean(node_means[m] / node_means[REF]))
            if m != REF:
                t, p = dm_test(surf[m], surf[REF], h)
                r["dm_t_vs_ref"], r["dm_p_vs_ref"] = t, p

        # MCS at 90% on surface losses
        try:
            from arch.bootstrap import MCS
            ldf = pd.DataFrame(surf)
            # a duplicated loss column makes the MCS elimination degenerate (it returns
            # empty). Drop exact duplicates and say so rather than reporting a silent NaN.
            dup = ldf.T.duplicated()
            if dup.any():
                print(f"  ⚠ identical loss columns dropped from MCS: "
                      f"{list(ldf.columns[dup])} — check the run that produced them")
                ldf = ldf.loc[:, ~dup.to_numpy()]
            mcs = MCS(ldf, size=0.10, block_size=max(5, 2 * h), method="R")
            mcs.compute()
            inc = set(mcs.included)
            for r in out:
                if r["stage"] == stage and r["track"] == track and r["h"] == h:
                    r["in_mcs90"] = r["model"] in inc
        except Exception as e:  # noqa: BLE001
            print(f"  MCS failed ({stage}): {e}", flush=True)

        # FWD comparison on the ATM subset (kill criterion 3)
        if "fwd" in atm_ser:
            for m in sorted(models):
                if m == "fwd":
                    continue
                t, p = dm_test(atm_ser[m], atm_ser["fwd"], h)
                print(f"  [{stage}] ATM subset: {m:12s} vs fwd  DM t {t:+6.2f} p {p:.4f} "
                      f"(neg = beats fwd)  loss {np.nanmean(atm_ser[m]):.5f} "
                      f"vs {np.nanmean(atm_ser['fwd']):.5f}", flush=True)

    # archive loss series (deliverable 3)
    np.savez_compressed(
        DIR / f"losses_{track}_h{h}.npz",
        dates_full=np.asarray(Y.index[rows_full], dtype="datetime64[ns]").astype("int64"),
        dates_atm=np.asarray(Y.index[rows_atm], dtype="datetime64[ns]").astype("int64"),
        **{f"{st}_{sc}_{m}": s for st, d0 in arch_losses.items()
           for sc, d1 in d0.items() for m, s in d1.items()})
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in out])
    return df


if __name__ == "__main__":
    tables = []
    for track in ("trackA", "trackB"):
        for h in HS:
            t = evaluate(track, h)
            if t is not None:
                tables.append(t)
    if tables:
        allt = pd.concat(tables, ignore_index=True)
        allt.to_csv(DIR / "results_table.csv", index=False)
        cols = ["track", "h", "stage", "model", "pinball", "ratio_vs_ref", "dm_t_vs_ref",
                "dm_p_vs_ref", "in_mcs90", "rmse_chg", "hit", "cov10", "cov50", "cov90",
                "wall_s"]
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print("\n" + allt[[c for c in cols if c in allt.columns]]
                  .sort_values(["track", "h", "stage", "pinball"]).to_string(index=False))
        print(f"\nsaved -> {DIR/'results_table.csv'}")
