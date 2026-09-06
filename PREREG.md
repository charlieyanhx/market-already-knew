# PREREG — Does TimesFM-3 forecast the SPY IV surface better than a 3-factor VAR?

Date registered: 2026-08-31 (TimesFM-3 release date). Registered BEFORE any model was run
on the panel. This is a falsification test of a forecasting claim, not a tape study — no
Sharpe is produced or quoted at any stage before the (conditional) economic test, which has
its own bar and correction.

License constraint: TimesFM 3.0 weights ship under TimesFM Non-Commercial License v1.0 —
research-only, no production/live use. TimesFM 2.5 (Apache-2.0) is the only deployable
version and is included as a control. Nothing from this study touches live infra.

## Objective

Decide whether TimesFM-3 (multivariate mode) forecasts the SPY implied-vol surface better
than cheap econometric baselines. "Better" = lower pinball loss with statistical
significance, after removing calibration effects (Mincer-Zarnowitz), per the kill ladder.

## Data (placeholders resolved)

- Source: `data/external/philippdubach/SPY_options.parquet` — free EOD chains,
  2008-01-02 → 2025-12-12, 24.68M rows. Program status: **screen-grade** (validated for
  IV-node / IC measurement; failed calibration twice for tape building — acceptable here
  because the deliverable is forecast loss, not P&L).
- Vendor greeks in this source are known-unreliable → **IV and delta are recomputed** with
  the program's kernel (`scripts/research/kernel.py`): per-(date,expiry) discount factor D
  and forward F from same-strike parity regression (near-money band), Black-76 bisection IV
  per strike **from the OTM leg**, spot delta `D·N(d1)·exp(carry·T)` with carry from
  `implied_carry` (two-expiry, spot-free). T = dte/365 (ACT/365). This sidesteps both the
  16:15/16:00 non-synchronicity and the American-exercise parity break (near-ATM band only
  for D,F estimation, per the standing rule).
- Cross-check store: `data/cache/surface_grid_eod.parquet` (owned CBOE 16:15 grid,
  2020-09 → 2026-05). Used only for the calibration gate on the overlap, not as panel.
- Period: 2008-01-02 → 2025-12-12, daily EOD. The owned store cannot host the study: its
  extractor clips DTE at 90 (90-DTE node unbracketed on ~60% of days) and it is 5.5× shorter.

### Panel construction (Track A targets)

12 nodes: {25Δ put, 50Δ (ATM), 25Δ call} × {7, 30, 60, 90} calendar-day constant maturity.

- Quality filter: bid>0, ask>bid, mid=(bid+ask)/2; per-expiry slice needs a valid parity
  fit (≥5 same-strike pairs, 0.90<D≤1.02) and ≥4 usable strikes per wing.
- Delta axis: within each expiry, IV interpolated linearly vs recomputed delta to the
  target (put wing to −0.25; call wing to +0.25; ATM = mean of interp at call Δ=+0.50 and
  put Δ=−0.50). **No extrapolation** — node NaN unless the target delta is spanned.
- Tenor axis: for each target tenor τ, bracketing expiries (1 ≤ dte, both sides of τ),
  linear interpolation in **total variance** at fixed delta; NaN if unbracketed.
- Point-in-time by construction: each row uses only that day's chain.
- Target variable: log(IV) per node. Forecast in log space.

### Calibration gate (must pass before any forecasting)

Overlap join vs `surface_grid_eod.parquet` (2020-09 → 2025-12): ATM30 and RR10-equivalents
corr ≥ 0.95 (program's standing bar). RR25 corr reported against the same bar with the
recorded caveat that the vendor-based free build failed it at 0.906; the kernel recompute
is expected to close that gap — if RR25 still fails, the 25Δ nodes carry a data-quality
flag in every downstream table (both candidate and baselines are hit by the same noise, so
comparisons remain internally valid; absolute losses on 25Δ nodes are not comparable
across studies). Also reported: correlation of 1-day log-IV **changes** per node (the
actual target), and per-node availability by year (coverage line).

### Known-leak guard

`free_spy_iv30_panel.parquet::vrp_proxy` is a recorded forward-looking column — not used.
No feature other than the panel's own history and the deterministic event calendar enters
any model.

## Track B — SSVI

Daily SSVI fit (power-law phi): w(k) = θ/2 · [1 + ρφ(θ)k + √((φ(θ)k+ρ)² + 1−ρ²)],
φ(θ) = η·θ^(−γ). Fit per day on all expiries with dte ∈ [5, 120] that pass the slice
filter, k = ln(K/F), on kernel-recomputed total variances; θ_exp free per expiry, global
(ρ, η, γ); constraints ρ∈(−1,1), γ∈[0,0.995], η(1+|ρ|)≤2 (butterfly), θ monotone in T
enforced by isotonic projection (calendar). Warm-start from t−1 (past info only).
θ at the 4 target tenors by monotone interpolation of θ(T).

Forecast series (7): log θ_7, log θ_30, log θ_60, log θ_90, atanh(ρ), log η, logit(γ/0.995).
Track B pinball is computed on these transformed parameters (all models forecast the same
series). Cross-track comparison uses point RMSE on reconstructed node log-IVs (median
path only — quantiles are not mapped through the nonlinear reconstruction).
Track B exists to price the arbitrage-free constraint; Track A is where the kill criteria
are evaluated (12 raw nodes, pinball).
Gate B0: the daily SSVI fit must reconstruct the 12 Track-A nodes with median |error|
≤ 1 vol pt; if it cannot, Track B is reported as "SSVI fit inadequate" and the study
proceeds on Track A alone (this is a fit-quality statement, not a forecasting result).

## Horizons

h = 1 and h = 5 trading days, same origin. One TimesFM call at horizon 5 supplies both
(steps 1 and 5). Baselines: iterated for AR/VAR, direct per-h for HAR, trivial for RW.

## Models (9)

1. TFM3-UNI — TimesFM-3 zero-shot, each series independently.
2. TFM3-MV — TimesFM-3 multivariate (variate attention across 12 nodes / 7 SSVI params).
3. TFM3-MV-COV — TFM3-MV + past-future covariates: FOMC, CPI, NFP dates (actual release
   calendars, compiled from primary sources), monthly (3rd-Friday) and quarterly expiry,
   day-of-week, market-holiday-adjacent flag. Isolates the covariate slot (the only genuinely
   new capability vs 2.5).
4. TFM25-UNI — TimesFM-2.5 (Apache-2.0), univariate. The deployable control.
5. RW — zero-change forecast.
6. AR1 — AR(1) on log-IV changes per node, rolling refit.
7. HAR — log-HAR (D/W/M components) per node, direct h-step, rolling refit.
8. PCA-VAR — 3-factor PCA (level/slope/curvature) on the 12 log-IV nodes, VAR(1) on
   factors, reconstruct. **Primary benchmark.**
9. FWD — forward-curve-implied: expected future ATM total variance under martingale
   forward variance from today's term structure (ATM nodes only; wings have no
   market-implied forecast — FWD is scored on the 4 ATM nodes and enters kill
   criterion 3 on that subset).

Zero-shot for TimesFM (no fine-tuning). Context: full available history at each origin
(≤ 16k supported; actual ≤ ~4.5k). NaN gaps in context ffilled ≤ 5 days (count reported);
origins whose target is NaN are dropped for that node, for all models alike.

## Quantile recipe (pre-registered, uniform)

TimesFM emits 9 quantiles (10%…90%) — used directly (sort_quantiles=True; TFM2.5's
10-column output has its mean column dropped after verifying column order on synthetic
data). Point-forecast baselines (5–9) are wrapped identically: forecast quantiles =
point forecast + trailing empirical quantiles of that model's own resolved h-step
forecast errors (window 252, past-only; origins enter evaluation only once the wrapper
window is full). No Gaussian assumption anywhere.

## Walk-forward

- Rolling refit window for fitted baselines (6,7,8): 756 trading days, refit every step.
- Error-wrapper burn-in: 252 resolved forecasts.
- Evaluation origins: every trading day from (first all-12-node-complete date + 756 +
  252 + h) through 2025-12-12 − h. Same origin set for every model within a track/horizon.
- Embargo: forecast at origin t uses data ≤ t only; wrapper at t uses errors of origins
  ≤ t − h. No parameter is ever fit on post-origin data. Test set touched once: the loss
  series is computed once per model and archived; no model is re-tuned after seeing it.

## Metrics

- Primary: pinball loss averaged over the 9 quantile levels, on log-IV (level at t+h;
  identical to change-loss given the known level at t).
- Point: RMSE, MAE on h-step log-IV changes. Directional: sign hit rate on changes.
- Calibration: PIT histogram, empirical coverage per quantile level.
- Cost: wall-clock per forecast per model (recorded inline).
- Aggregation: per-node loss ratio to PCA-VAR, then averaged across nodes (never pooled
  raw losses). Surface-level series for DM/MCS = per-origin mean across nodes of
  per-node losses (nodes share the log-IV scale).

## Statistical tests

1. Pairwise Diebold-Mariano, Newey-West HAC (lag = 2(h−1)+5), per node and surface-level.
2. Hansen MCS at 90% (arch.bootstrap.MCS, block bootstrap, per-origin surface losses),
   over the full model family, per track × horizon.
3. Mincer-Zarnowitz recalibration: rolling past-only (504d) regression of realized on
   point forecast per (model, node, h); affine map applied to all quantiles; steps 1–2
   re-run on recalibrated losses. Expectation from Brini (2026): short-horizon
   foundation-model gains vanish after MZ. This step is not optional.
4. Specification count: every configuration run (including failures) appended to the
   study spec ledger and to docs/GLOBAL_TEST_LEDGER_2026_08.md; feeds any later
   multiple-testing correction. Bars are fixed by this prereg BEFORE results exist.

## Kill criteria — checked in order, stop at first failure

1. TFM3-MV (or MV-COV) fails to enter the 90% MCS against PCA-VAR on pinball (Track A,
   either horizon) → STOP: "variate attention finds nothing beyond linear factor dynamics."
2. Surviving edge disappears after MZ recalibration (DM no longer rejects at 5%, or drops
   from the MCS) → STOP: "it was calibration, not information."
3. Surviving edge does not also beat FWD on the ATM subset (DM 5%) → STOP: "the market
   already knew."
4. Forward holdout after 2026-08-31 (release date = today): **zero days exist at
   registration.** Criterion 4 CANNOT be evaluated in this session; any survival through
   1–3 is provisional ("historical-only") until re-scored on ≥ 6 months of post-release
   data. The walk-forward loss series is archived to make that re-score mechanical.

Leakage context for criterion 4: TimesFM-3 pretrain (GiftEvalPretrain, Wikipedia
pageviews to 2023-11, Google Trends to 2022, synthetic) does not obviously contain SPY
IV, but correlated financial series may be present; "zero-shot on history" ≠ OOS.

## Economic test — only if 1–3 pass (and provisionally, given 4)

Map forecast quantiles to bull-put-spread entry/sizing on the owned tape (Line-3
accounting, MTM-daily, full-calendar), vs the unconditioned book; deflated Sharpe / SPA
correction using the spec count. Reported separately from forecast-loss significance.
Not run if any kill criterion fires.

## Deliverables

1. Results table: model × track × horizon — pinball, MCS membership, DM p vs PCA-VAR,
   before/after MZ. 2. One-paragraph verdict naming the kill criterion hit (or none).
3. Full walk-forward loss series archived (data/cache/tfm3_study/). 4. Inference-cost
note. 5. Spec ledger with counts.

## Amendments (all recorded before any candidate-vs-baseline result existed)

- A1 (2026-08-31, data stage): the free source carries a systematic put/call quote shift
  that biases parity-fitted discount factors +2-4% high at long DTE; kernel's D bound
  (0.90, 1.02] silently dropped 100-140d slices in the high-rate era and starved the 90d
  node's upper bracket. The panel uses a study-local parity fork with D ∈ (0.85, 1.06]
  (`parity_surface_study`); kernel.py untouched. Calibration gate re-passed unchanged.
- A2 (2026-08-31, fit stage): Gate B0 FAILED — SSVI reconstruction median |err| 2.03 vp
  (bar 1.0), p90 8.25 vp. Strict single-(ρ,η,γ) power-law SSVI cannot match SPY wings
  across 5-120 DTE. Per the pre-registered contingency, kill criteria are evaluated on
  Track A only; Track B is reported as supplementary (within-track comparisons remain
  internally valid — every model forecasts the same parameter series).
- A3 (2026-08-31, fit stage): θ_7 is available only 20-30% of days pre-2017 (shortest
  SSVI-fit slice is usually > 7 DTE before the weeklies/dailies era). θ_7 is dropped from
  the Track B forecast set → 6 series (log θ_30/60/90, atanh ρ, log η, logit γ).

- A4 (2026-09-01, inference stage): TFM3-UNI implemented as an ABLATION — the same model
  and pipeline with `use_variate_attention=False` on the stacked 12-series input — rather
  than strict per-series calls. Strict per-series runs sequentially inside the library
  (~21s/origin, infeasible); the ablation is ~10x faster and differs from per-series
  output by ≤1.4% of forecast scale (residual coupling via joint normalization). This
  makes UNI-vs-MV a controlled isolation of variate attention itself. Decided on wall-
  clock grounds before any UNI result existed.
- A5 (2026-09-01, follow-up W3 clarification, recorded before the partial-IC number was
  computed): the GROSS rr25 mean-reversion amp/toll is a known, already-deployed lane
  family (skew fade) and proposes nothing; the W3 bar (amp/toll ≥ 2) applies to the
  ORTHOGONAL increment of TFM3-MV over the best cheap model (partial IC after
  regressing TFM3's forecast on the cheap model's forecast).
- **A6 (2026-09-01) — A4 WITHDRAWN, original spec restored.** A4 replaced the registered
  per-series univariate with a `use_variate_attention=False` flag ablation on wall-clock
  grounds. Two defects: (i) the flag is consumed at model construction, so passing it to
  `from_pretrained` is a silent no-op — the run returned forecasts **bit-identical** to
  TFM3-MV (max |diff| 0.0 over 389,772 quantile values), and the duplicate loss column
  additionally broke the MCS (returned empty); (ii) A4's infeasibility measurement
  (~21s/forecast) was taken on CPU, where per-series is 3.5× slower than stacked; on the
  MPS device actually used it is **1.0-1.2×** (0.85s vs 0.71s at ctx 1024). The registered
  per-series construction is therefore affordable and is restored. The invalid run is
  quarantined at data/cache/tfm3_study/quarantine/, not deleted. Verified before re-running:
  per-series and stacked forecasts differ by ~1-2% of scale, so the comparison is live.

- **A7 (2026-09-02) — TimesFM-2.5 on Track B omitted, not run.** Track B is already
  supplementary (Gate B0 failed, A2), and the 2.5 control there bears on no kill criterion.
  The preceding Track B stage took **19.5 hours** against 2.2 hours for the identical
  configuration on Track A, under system-wide memory pressure (481M swapouts, 8GB machine);
  the p90 stayed at 2.84s while the mean rose to 21.16s, i.e. stall-driven, not compute-
  driven. Continuing was judged disproportionate. Track B is reported with the three
  TimesFM-3 configurations and all baselines; the 2.5 row there is stated as omitted rather
  than left blank. No Track A row is affected.

## Follow-up W (2026-09-01, POST-HOC — recorded after headline results)

Observation that triggered it: TFM3-MV beat all econometric baselines on the 25Δ wing
nodes, where no market-implied (FWD) benchmark exists. This is a post-selection
observation and inherits the study's selection bias; it is labeled exploratory and can
at most PROPOSE a new lane, never open one.

Pre-stated analyses and bar (stated BEFORE any wing dollar number was computed):
- W1: per-node pinball (raw + MZ), TFM3-MV vs each baseline, DM per wing node.
- W2: point-forecast quality on wing log-IV changes: IC (corr of predicted vs realized
  change), hit rate, and the post-MZ margin vs RW.
- W3: economic pre-screen, Gate-0 style, NO tape: predictable move per 2σ signal
  (IC × 2σ of realized h-step change, in vol pts) vs the round-trip 25Δ toll in vol pts
  taken from the owned-data skew grid (`skew_delta_grid_panel.parquet`, 16:15 basis —
  a toll known to run ~+83% rich vs intraday, i.e. conservative against the edge).
  BAR: amp/toll ≥ 2 at h=5 for "proposes a lane" (per the two-gate pre-screen law);
  anything less closes the scrap with a number. h=1 reported for completeness.
- W3b (recorded 2026-09-01 before computing): the partial-IC comparator set must include
  the program's champion class — the trailing z-score fade of rr25 itself (63d and 252d
  trailing windows, past-only), the deployed S-signal construction. The bar applies to
  TFM3-MV's increment orthogonal to the BEST of {HAR, PCA-VAR, trailing-z}. If the
  orthogonal increment vs trailing-z fails the bar, the scrap closes as "rediscovered
  the deployed skew fade."

## Pre-committed expectations (priors, falsifiable)

Program priors say: persistence dominates (RW/AR hard to beat at h=1 on IV), simple beats
complex on this data (GRU < trailing z; ML framework closure), and foundation-model RV
gains were calibration (Brini). The most likely outcome is a kill at criterion 1 or 2.
If TFM3-MV survives 1–3, that is a genuinely new fact about cross-node structure beyond
3 linear factors — worth the compute either way. Null result gets reported as prominently
as a positive.
