# Data notes

## Shipped (derived, MIT)
Everything listed in the top-level README's layout section. All derived files were
built by the scripts in this repository from the free raw chains below.

## Not shipped

**Raw free EOD chains (~630 MB).** `tfm3_surface_panel.py` expects
`data/raw/SPY_options.parquet` with columns
`contract_id, symbol, expiration, strike, type, bid, ask, date, ...`
(one row per contract per day, 2008-01-02 → 2025-12-12). The file used in the paper is
the freely distributed SPY end-of-day chain parquet ("philippdubach" free options
dataset). Any source with the same schema and coverage reproduces the panel; the
paper's Section 2 gate statistics tell you whether your copy matches.

**Proprietary references.** The CBOE 16:15 snapshot grid used for the panel's
calibration gate, and the quote-grid used to measure the 0.173 vol-point 25Δ toll, are
purchased data and are not redistributed. Neither is needed for any computation here:
the gate is a validation (its numbers are in the paper), and the toll enters only as
the disclosed constant.

## Event calendar
`macro_event_calendar.csv` — FOMC statement dates (scheduled flag included), CPI and
NFP release dates, 2008-01 → 2026-09, compiled from Federal Reserve and BLS primary
sources; per-year provenance in `macro_event_calendar_SOURCES.md`. Cross-validated
against an independently scraped calendar: CPI 215/215 and NFP 215/215 exact
agreement.
