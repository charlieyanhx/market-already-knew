"""TFM3 study — TimesFM forecast runner (runs in .venv-timesfm, NOT the repo venv).

Prereg: PREREG.md (repository root). Zero-shot; context = full available
history at each origin (ffill of interior gaps <= 5 days; origin skipped if any NaN
remains). One horizon-5 call per origin supplies both h=1 and h=5. Outputs the same npz
schema as tfm3_baselines.py. Wall-clock per forecast recorded.

Usage: .venv-timesfm/bin/python scripts/research/tfm3_timesfm_run.py \
    --model tfm3-mv --track trackA [--device cpu] [--limit 40] [--start-row 756]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tfm3_common import DIR, FIT_WIN, HS, QS  # noqa: E402

MAX_FFILL = 5
MIN_CTX = 256
HMAX = max(HS)


_CAL: pd.DataFrame | None = None


def build_covariates(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic + macro-release covariates, known past AND future (prereg model 3)."""
    global _CAL
    if _CAL is None:
        _CAL = pd.read_csv(DIR / "macro_event_calendar.csv", parse_dates=["date"])
    cal = _CAL
    cov = pd.DataFrame(index=index)
    for ev in ("FOMC", "CPI", "NFP"):
        dts = set(cal[cal.event == ev].date)
        cov[f"is_{ev.lower()}"] = [1.0 if d in dts else 0.0 for d in index]
    fri3 = pd.DatetimeIndex([d for d in index if d.weekday() == 4 and 15 <= d.day <= 21])
    cov["is_opex"] = index.isin(fri3).astype(float)
    cov["is_qtr_opex"] = (index.isin(fri3) & index.month.isin([3, 6, 9, 12])).astype(float)
    for wd in range(1, 5):
        cov[f"dow_{wd}"] = (index.weekday == wd).astype(float)
    gap_next = np.r_[np.diff(index.values).astype("timedelta64[D]").astype(int), 1]
    gap_prev = np.r_[1, np.diff(index.values).astype("timedelta64[D]").astype(int)]
    cov["holiday_adj"] = ((gap_next > 3) | (gap_prev > 3)).astype(float)
    return cov


def extend_covariates(last: pd.Timestamp, h: int) -> np.ndarray:
    """Future h business days after `last`, same construction."""
    fut = pd.bdate_range(last, periods=h + 1)[1:]
    return build_covariates(fut).to_numpy().T


def index_i8(idx: pd.Index) -> np.ndarray:
    return np.asarray(idx, dtype="datetime64[ns]").astype("int64")


def make_context_fn(yv: np.ndarray):
    """Context = longest gap-free suffix ending at the origin, after bounded ffill.

    ffill(limit=MAX_FFILL) is past-only, so precomputing once over the full panel is
    point-in-time safe. Rows still containing NaN (gaps > MAX_FFILL, e.g. the sparse
    2008-2010 7-DTE era) break the context; the context starts after the last break.
    """
    yf = pd.DataFrame(yv).ffill(limit=MAX_FFILL).to_numpy().astype(np.float32)
    bad = ~np.isfinite(yf).all(axis=1)
    last_bad = np.full(len(yf), -1)
    cur = -1
    for t in range(len(yf)):
        if bad[t]:
            cur = t
        last_bad[t] = cur
    n_filled = int((~np.isfinite(yv) & np.isfinite(yf)).sum())
    print(f"context prep: {n_filled} node-days ffilled (limit {MAX_FFILL}), "
          f"{bad.sum()} break rows; last break at row {int(bad.nonzero()[0][-1]) if bad.any() else -1}",
          flush=True)

    def prepare(i: int) -> np.ndarray | None:
        if bad[i]:
            return None
        start = last_bad[i] + 1
        L = i - start + 1
        if L < MIN_CTX:
            return None
        # bucket context length to a multiple of 64 (drop <=63 OLDEST days) so tensor
        # shapes repeat across origins — otherwise MPS recompiles kernels every call
        L = (L // 64) * 64
        return yf[i + 1 - L: i + 1].T

    return prepare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["tfm3-uni", "tfm3-mv", "tfm3-mv-cov", "tfm25"])
    ap.add_argument("--track", required=True, choices=["trackA", "trackB"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--start-row", type=int, default=FIT_WIN)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from tfm3_common import load_track
    Y = load_track(args.track)
    yv = Y.to_numpy()
    n, V = yv.shape
    origins = list(range(args.start_row, n - 0))     # forecast every row; eval intersects
    if args.limit:
        origins = origins[:: max(1, len(origins) // args.limit)][: args.limit]

    import timesfm
    if args.model.startswith("tfm3"):
        # tfm3-uni is TRUE per-series (prereg model 1). A4's flag-based ablation was
        # withdrawn (A6): use_variate_attention is consumed at construction, so passing it
        # to from_pretrained is a silent no-op that returned forecasts bit-identical to MV.
        fc = timesfm.TimesFM3Forecaster.from_pretrained(
            "google/timesfm-3.0-pytorch", device=args.device)
    else:
        import torch
        fc = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        if args.device != "cpu":
            fc.model.device = torch.device(args.device)   # class only knows cuda/cpu at init
            fc.model.to(fc.model.device)
        fc.compile(timesfm.ForecastConfig(
            max_context=4608, max_horizon=16, normalize_inputs=True,
            per_core_batch_size=V,                        # no dummy-series padding
            use_continuous_quantile_head=True, fix_quantile_crossing=True))

    cov_full = build_covariates(Y.index).to_numpy().T if args.model == "tfm3-mv-cov" else None

    point = {h: np.full((n, V), np.nan) for h in HS}
    quant = {h: np.full((n, V, len(QS)), np.nan) for h in HS}
    prepare_context = make_context_fn(yv)
    times, t_all = [], time.time()
    done = 0
    for i in origins:
        ctx = prepare_context(i)
        if ctx is None:
            continue
        L = ctx.shape[1]
        t0 = time.time()
        if args.model == "tfm25":
            pt, q = fc.forecast(horizon=HMAX, inputs=[ctx[j] for j in range(V)])
            q, pt = np.asarray(q), np.asarray(pt)
            if done == 0:                            # one-time layout check: cols 1..9 monotone
                assert (np.diff(q[0, 0, 1:]) >= -1e-9).all(), "tfm2.5 quantile layout changed"
            q = q[:, :, 1:]                          # col 0 = mean, cols 1..9 = q10..q90
            for h in HS:
                point[h][i] = pt[:, h - 1]
                quant[h][i] = q[:, h - 1, :]
        else:
            if args.model == "tfm3-uni":
                outs = list(fc.predict_batch([ctx[j] for j in range(V)], horizon=HMAX,
                                             return_quantiles=True))
                fcast = np.stack([o.forecast for o in outs])          # (V, H)
                qs = np.stack([o.quantiles for o in outs])            # (V, H, 9)
            else:
                kw = {}
                if cov_full is not None:
                    past = cov_full[:, i - L + 1: i + 1]
                    fut = extend_covariates(Y.index[i], HMAX)
                    kw["past_future_covariates"] = np.concatenate([past, fut], axis=1)
                out = fc.predict(ctx, horizon=HMAX, return_quantiles=True, **kw)
                fcast, qs = out.forecast, out.quantiles               # (V,H), (V,H,9)
            for h in HS:
                point[h][i] = fcast[:, h - 1]
                quant[h][i] = qs[:, h - 1, :]
        times.append(time.time() - t0)
        done += 1
        if args.device == "mps" and done % 250 == 0:
            import torch
            torch.mps.empty_cache()
        if done % 200 == 0:
            print(f"  {done}/{len(origins)} origins, mean {np.mean(times):.2f}s/forecast, "
                  f"last200 {np.mean(times[-200:]):.2f}s, "
                  f"elapsed {(time.time()-t_all)/60:.1f}m", flush=True)

    for h in HS:
        np.savez_compressed(
            DIR / f"fc_{args.track}_{args.model}_h{h}.npz",
            dates=index_i8(Y.index), cols=np.array(Y.columns, dtype=object),
            point=point[h], quant=quant[h], qs=QS,
            wall_mean=np.mean(times) if times else np.nan,
            wall_p90=np.percentile(times, 90) if times else np.nan)
    if times:
        print(f"DONE {args.model} {args.track}: {done} origins, "
              f"mean {np.mean(times):.2f}s p90 {np.percentile(times, 90):.2f}s per forecast, "
              f"total {(time.time()-t_all)/60:.1f}m", flush=True)
    else:
        print(f"DONE {args.model} {args.track}: ZERO origins produced forecasts — "
              "check context availability", flush=True)


if __name__ == "__main__":
    main()
