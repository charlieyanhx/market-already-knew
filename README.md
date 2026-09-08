# The Market Already Knew — reproducibility package

Code, pre-registration, and archived walk-forward results for:

> **The Market Already Knew: A Pre-Registered Falsification of TimesFM-3 on the SPY
> Implied-Volatility Surface.** Charlie Yan, September 2026. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7406218

The study was registered on the model's release day (2026-08-31), before any arm ran;
[PREREG.md](PREREG.md) is the registration verbatim, including all seven dated
amendments. The paper is in [paper/](paper/).

## What re-runs without anything but this repository

Every statistical result in the paper — the loss tables, Diebold–Mariano tests, Model
Confidence Sets, Mincer–Zarnowitz recalibration, per-year robustness, and the wing
follow-up — recomputes from the **archived forecasts** shipped in `data/` (`fc_*.npz`,
one file per model × track × horizon; ~56 MB total). No GPU, no model download:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cd scripts
../.venv/bin/python tfm3_eval.py            # main tables (results_table.csv, losses_*.npz)
../.venv/bin/python tfm3_robustness.py      # per-year loss-ratio concentration check
../.venv/bin/python tfm3_wing_followup.py   # wing anomaly: orthogonalization + audits
```

`data/results_table.csv` and `data/losses_*.npz` as shipped are the paper's numbers;
re-running overwrites them in place with identical values (evaluation is deterministic).

## Regenerating the forecasts themselves

The TimesFM runs need the second environment (the models pin different library
versions than the evaluation stack):

```bash
python -m venv .venv-timesfm && .venv-timesfm/bin/pip install -r requirements-timesfm.txt
cd scripts
../.venv-timesfm/bin/python tfm3_timesfm_run.py --model tfm3-mv --track trackA --device mps
```

Models: `tfm3-mv`, `tfm3-mv-cov`, `tfm3-uni`, `tfm25`; tracks `trackA`/`trackB`;
`--device cpu|mps|cuda`. TimesFM-3.0 weights download from
`google/timesfm-3.0-pytorch` and are licensed for **non-commercial use only**; 2.5
(`google/timesfm-2.5-200m-pytorch`) is Apache-2.0. On an M1 GPU a full Track A
walk-forward is ~1–2 h per configuration (see the paper's cost section for the two
throughput pitfalls: context-length bucketing, and sustained-run memory pressure).
Baselines regenerate with `tfm3_baselines.py` (minutes, CPU).

## Rebuilding the panel from raw chains

`data/panel_trackA.parquet` (the 12-node surface), `data/slices_atm.parquet`, and the
Track-B SSVI inputs rebuild from the free raw chains with `tfm3_surface_panel.py` and
`tfm3_ssvi_fit.py` — see [data/README.md](data/README.md) for the raw-data download
(~630 MB, not redistributed here). The panel's calibration gate against a proprietary
CBOE 16:15 snapshot grid cannot be re-run from this package; the gate's numbers are in
the paper (Section 2) and the registration.

## Layout

```
PREREG.md                  registration + amendments A1–A7 (verbatim)
paper/                     the paper PDF
scripts/
  parity_b76.py            spot-free pricing (parity carry, Black-76 IV, deltas)
  tfm3_surface_panel.py    raw chains → 12-node constant-maturity panel (+ gate)
  tfm3_ssvi_fit.py         Track B: daily SSVI fits (failed its fit gate; see paper)
  tfm3_common.py           shared series definitions and paths
  tfm3_baselines.py        RW / AR(1) / log-HAR / PCA-VAR / forward-curve + wrapper
  tfm3_timesfm_run.py      TimesFM 2.5 / 3.0 walk-forward runner
  tfm3_eval.py             pinball, DM/HAC, MCS90, rolling past-only MZ
  tfm3_robustness.py       per-year loss-ratio concentration
  tfm3_wing_followup.py    post-hoc wing study (registered bars; skip-1 audit)
data/
  panel_trackA.parquet     12-node log-IV surface, 2008–2025 (built, gated)
  params_trackB.parquet    daily SSVI parameters
  slices_atm.parquet       per-expiry ATM curve (forward-curve baseline input)
  macro_event_calendar.csv FOMC/CPI/NFP release dates 2008–2026, primary sources
  fc_*.npz                 archived forecasts: point + 9 quantiles per model/track/h
  losses_*.npz             archived per-origin loss series (the criterion-4 re-score input)
  results_table.csv        the paper's main table, machine-readable
```

## Basis and scope

All results are screen-basis forecast-loss comparisons on end-of-day quoted surfaces —
no fills, no P&L, no performance claims. The one number imported from proprietary data
is a single constant: the 0.173 vol-point 25Δ/30d round-trip spread used to scale the
wing follow-up (provenance in `tfm3_wing_followup.py`; the basis it comes from is
conservative against the candidate). This research is unaffiliated with Google;
TimesFM weights were used under their respective licenses for research only.

## License

Code and derived data files: MIT (see LICENSE). The raw free chains and the TimesFM
weights carry their own upstream terms.
