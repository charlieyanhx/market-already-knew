"""TFM3 study — Follow-up W (POST-HOC, see prereg amendment): the 25Δ wing edge.

W1 per-node pinball + DM on wing nodes; W2 point-forecast IC/hit on wing changes;
W3 Gate-0-style toll math on the 30d wings (owned-grid 16:15 toll, conservative).
Bar (pre-stated): amp/toll >= 2 at h=5 to propose a lane; else the scrap closes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tfm3_common import DIR, HS, load_trackA  # noqa: E402
from tfm3_eval import dm_test, mz_recalibrate, pinball  # noqa: E402

WING = [f"{c}_{t}" for t in (7, 30, 60, 90) for c in ("p25", "c25")]


def main() -> None:
    Y = load_trackA()
    yv = Y.to_numpy()
    cols = list(Y.columns)
    wing_j = [cols.index(c) for c in WING]

    # 25-delta/30d round-trip spread in vol points: median over 1,430 days of a
    # proprietary 16:15 CBOE snapshot grid (a basis measured ~+83% rich vs intraday,
    # i.e. conservative against the candidate). The grid itself is not redistributed.
    toll_vp = 0.173

    for h in HS:
        files = sorted(DIR.glob(f"fc_trackA_*_h{h}.npz"))
        models = {}
        for f in files:
            z = np.load(f, allow_pickle=True)
            name = f.stem.replace("fc_trackA_", "").replace(f"_h{h}", "")
            if name == "fwd":
                continue                       # no wing forecast by construction
            models[name] = {"point": z["point"], "quant": z["quant"]}
        realized = np.full_like(yv, np.nan)
        realized[:-h] = yv[h:]
        ok = np.isfinite(yv) & np.isfinite(realized)
        for d in models.values():
            ok &= np.isfinite(d["quant"]).all(axis=2)
        rows = ok.all(axis=1)
        print(f"\n================ h={h}: {rows.sum():,} origins, models: {sorted(models)}")

        # ---- W1: per-node pinball (raw & MZ) + DM vs baselines, wing nodes only
        for stage in ("raw", "mz"):
            pls = {}
            for m, d in models.items():
                q = d["quant"] if stage == "raw" else \
                    mz_recalibrate(d["point"], realized, d["quant"], h)[1]
                pls[m] = np.where(ok, pinball(realized, q), np.nan)[rows]
            print(f"--- W1 [{stage}] per-wing-node pinball (x1000) & DM t of tfm3-mv vs:")
            hdr = f"{'node':8s}" + "".join(f"{m:>10s}" for m in sorted(pls)) + \
                "   | dm(har) dm(rw) dm(pcavar)"
            print(hdr)
            for c in WING:
                j = cols.index(c)
                vals = "".join(f"{np.nanmean(pls[m][:, j])*1000:10.3f}" for m in sorted(pls))
                dms = ""
                if "tfm3-mv" in pls:
                    for ref in ("har", "rw", "pcavar"):
                        t, _ = dm_test(pls["tfm3-mv"][:, j], pls[ref][:, j], h)
                        dms += f" {t:+6.1f}"
                print(f"{c:8s}{vals}   |{dms}")

        # ---- W2: IC and hit on wing changes (point forecasts, raw)
        print("--- W2 IC (pred chg vs real chg) / hit%, wing nodes:")
        for c in WING:
            j = cols.index(c)
            rc = (realized - yv)[rows][:, j]
            line = f"{c:8s}"
            for m in sorted(models):
                pc = (models[m]["point"] - yv)[rows][:, j]
                mfin = np.isfinite(pc) & np.isfinite(rc) & (np.abs(pc) > 1e-12)
                ic = np.corrcoef(pc[mfin], rc[mfin])[0, 1] if mfin.sum() > 50 else np.nan
                hit = np.mean(np.sign(pc[mfin]) == np.sign(rc[mfin])) if mfin.sum() > 50 else np.nan
                line += f"  {m}:{ic:+.3f}/{hit:.0%}" if np.isfinite(ic) else f"  {m}: --"
            print(line)

        # ---- W3: toll math, 30d wings + RR25 (volpts), Gate-0 style
        jP, jC = cols.index("p25_30"), cols.index("c25_30")
        ivP, ivC = np.exp(yv[rows][:, jP]), np.exp(yv[rows][:, jC])
        realP = ivP * (realized - yv)[rows][:, jP] * 100    # volpts
        realC = ivC * (realized - yv)[rows][:, jC] * 100
        real_rr = realP - realC
        # toll_opt is $/structure round-trip, vega is $/volpt -> toll_vp is ALREADY volpts
        print(f"--- W3 toll math (h={h}): 25Δ/30d toll {toll_vp:.3f} volpts round-trip "
              f"(owned grid, 16:15 basis)")
        preds = {}
        for m in sorted(models):
            predP = ivP * (models[m]["point"] - yv)[rows][:, jP] * 100
            predC = ivC * (models[m]["point"] - yv)[rows][:, jC] * 100
            pred_rr = predP - predC
            fin = np.isfinite(pred_rr) & np.isfinite(real_rr) & (np.abs(pred_rr) > 1e-12)
            if fin.sum() < 100:
                print(f"  {m:10s} rr: no active forecast (RW-style)")
                continue
            preds[m] = (pred_rr, fin)
            ic = np.corrcoef(pred_rr[fin], real_rr[fin])[0, 1]
            amp = ic * 2 * np.std(real_rr[fin])
            print(f"  {m:10s} rr25_30 GROSS: IC {ic:+.3f}  2σ(realΔ) "
                  f"{2*np.std(real_rr[fin]):.2f}vp  amp {amp:+.3f}vp  "
                  f"amp/toll {amp/toll_vp:+.2f}")
        # Champion comparators (W3b): trailing-z fade of rr25 itself — the classic
        # skew mean-reversion signal. Signal = -z (fade), past-only trailing windows.
        rr_full = (np.exp(yv[:, jP]) - np.exp(yv[:, jC])) * 100      # volpts, all rows
        rr_s = pd.Series(rr_full, index=Y.index)
        for win in (63, 252):
            mu = rr_s.rolling(win, min_periods=win // 2).mean()
            sd = rr_s.rolling(win, min_periods=win // 2).std()
            z = ((rr_s - mu) / sd).to_numpy()[rows]
            fin = np.isfinite(z) & np.isfinite(real_rr)
            ic = np.corrcoef(-z[fin], real_rr[fin])[0, 1]
            amp = ic * 2 * np.std(real_rr[fin])
            preds[f"z{win}"] = (-z, np.isfinite(z))
            print(f"  {'z'+str(win):10s} rr25_30 GROSS: IC {ic:+.3f}  amp {amp:+.3f}vp  "
                  f"amp/toll {amp/toll_vp:+.2f}   (trailing-z fade, champion class)")
        # Orthogonal increment of tfm3-mv over each comparator (A5/W3b: bar applies to
        # the increment over the BEST of these; gross rr fade is a well-known trade).
        for cand in ("tfm3-mv", "tfm3-mv-cov", "tfm3-uni", "tfm25"):
            if cand not in preds:
                continue
            print(f"  ---- candidate: {cand}")
            pt, ft = preds[cand]
            for ref in ("har", "pcavar", "z63", "z252"):
                if ref not in preds:
                    continue
                pr, fr = preds[ref]
                fin = ft & fr & np.isfinite(real_rr)
                b = np.polyfit(pr[fin], pt[fin], 1)
                resid = pt[fin] - np.polyval(b, pr[fin])
                pic = np.corrcoef(resid, real_rr[fin])[0, 1]
                pamp = pic * 2 * np.std(real_rr[fin])
                print(f"  {cand} | {ref:7s} PARTIAL: IC {pic:+.3f}  amp {pamp:+.3f}vp  "
                      f"amp/toll {pamp/toll_vp:+.2f}  "
                      f"{'clears bar' if pamp/toll_vp >= 2 else 'below bar'}")
            # Adversarial audits: joint orthogonalization + skip-1-day (mark-bounce guard)
            refs = [r for r in ("har", "pcavar", "z63", "z252") if r in preds]
            fin = ft & np.isfinite(real_rr)
            for r in refs:
                fin &= preds[r][1]
            X = np.c_[np.ones(fin.sum())] if fin.sum() else None
            if X is not None and fin.sum() > 300:
                X = np.c_[np.ones(fin.sum()), *[preds[r][0][fin] for r in refs]]
                beta, *_ = np.linalg.lstsq(X, pt[fin], rcond=None)
                resid = pt[fin] - X @ beta
                pic = np.corrcoef(resid, real_rr[fin])[0, 1]
                pamp = pic * 2 * np.std(real_rr[fin])
                print(f"  {cand} | ALL-4  JOINT:   IC {pic:+.3f}  amp {pamp:+.3f}vp  "
                      f"amp/toll {pamp/toll_vp:+.2f}  "
                      f"{'clears bar' if pamp/toll_vp >= 2 else 'below bar'}")
                # skip-1: realized change measured t+1 -> t+h+1 (in full-row index space)
                rr_vp_full = rr_full
                idx_rows = np.flatnonzero(rows)
                sel = idx_rows[fin]
                okskip = sel + h + 1 < len(rr_vp_full)
                sel = sel[okskip]
                real_skip = rr_vp_full[sel + h + 1] - rr_vp_full[sel + 1]
                fs = np.isfinite(real_skip)
                pic_s = np.corrcoef(resid[okskip][fs], real_skip[fs])[0, 1]
                pamp_s = pic_s * 2 * np.nanstd(real_skip[fs])
                print(f"  {cand} | ALL-4  SKIP-1:  IC {pic_s:+.3f}  amp {pamp_s:+.3f}vp  "
                      f"amp/toll {pamp_s/toll_vp:+.2f}  "
                      f"{'SURVIVES mark-bounce guard' if pamp_s/toll_vp >= 2 else 'DIES on skip-1 -> bounce/denoising artifact'}")


if __name__ == "__main__":
    main()
