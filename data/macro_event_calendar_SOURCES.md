# macro_event_calendar.csv — provenance

Compiled 2026-08-31. Coverage window: 2008-01-01 .. 2026-09-30.
Columns: `date` (YYYY-MM-DD), `event` (FOMC / CPI / NFP), `scheduled` (FOMC: 1 =
regularly scheduled meeting, 0 = unscheduled/emergency action with a statement;
CPI/NFP: always 1). 604 rows, no duplicate (date, event) pairs.

## FOMC (statement-release dates; second day of two-day meetings)

- 2008–2020: Federal Reserve historical-materials pages, one per year:
  `https://www.federalreserve.gov/monetarypolicy/fomchistorical{YEAR}.htm`
  (fetched 2026-08-31 for YEAR = 2008..2020).
- 2021–2026: `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm`
  (fetched 2026-08-31).
- Scheduled meetings: the date recorded is the final (statement) day of the
  meeting as listed by the Fed.
- Unscheduled entries (`scheduled = 0`), dated by statement release:
  - 2008-01-22 (Jan 21 videoconference; statement URL `.../20080122b.htm`)
  - 2008-03-11 (Mar 10 call; statement URL `.../20080311a.htm`)
  - 2008-10-08 (Oct 7 call; statement URL `.../20081008a.htm`)
  - 2010-05-09 (swap-lines statement; a Sunday)
  - 2019-10-11 (Statement Regarding Monetary Policy Implementation following the
    Oct 4 videoconference — a balance-sheet/technical statement, not a rate
    decision; drop this row if only rate decisions are wanted)
  - 2020-03-03 and 2020-03-15 (emergency cuts; Mar 15 is a Sunday)
- Deliberately excluded: conference calls with no statement (2008 Jan 9/Jul 24/
  Sep 29; 2009 Jan 16/Feb 7/Jun 3; 2010 Oct 15; 2011 Aug 1/Nov 28; 2013 Oct 16;
  2014 Mar 4) and notation votes (2020 Mar 19/Mar 23/Mar 31/Aug 27; 2025 Aug 22),
  since they are not meetings with policy statements. The 2020 Mar 17-18
  scheduled meeting was cancelled (hence 7 scheduled meetings in 2020).

## CPI and NFP (BLS news-release dates)

- 2008–2025: BLS archived yearly release schedules
  `https://www.bls.gov/schedule/{YEAR}/home.htm` (linked from
  `https://www.bls.gov/bls/archived_sched.htm`), rows "Consumer Price Index for
  <month>" and "Employment Situation for <month>". Accessed 2026-08-31 via a
  real browser session (bls.gov returns HTTP 403 to non-browser clients; the
  older `/schedule/archives/cpi_nr.htm`-style URLs no longer exist).
  These pages reflect the schedule as last modified during/after each year and
  therefore carry ACTUAL post-shutdown dates — verified on 2013 (Sep-2013
  Employment Situation shown 2013-10-22 and Sep-2013 CPI shown 2013-10-30, the
  known post-shutdown release dates, not the original Oct 4 / Oct 16 slots) and
  on 2025 (see gaps below).
- 2026: current BLS schedule pages, accessed 2026-08-31:
  - CPI: `https://www.bls.gov/schedule/news_release/cpi.htm`
  - Employment Situation: `https://www.bls.gov/schedule/news_release/empsit.htm`

## Known gaps / caveats

- 2025 government shutdown (Oct–Nov 2025): the September-2025 CPI was released
  2025-10-24 and the September-2025 Employment Situation 2025-11-20. There was
  NO standalone release for the October-2025 reference month of either series
  (October CPI never published; October payrolls folded into the November
  report released 2025-12-16). Hence 11 CPI and 11 NFP rows in 2025.
- Rows after 2026-08-31 (NFP 2026-09-04, CPI 2026-09-11, FOMC 2026-09-16) are
  scheduled dates from the primary sources, not yet realized at compile time.
- 2026 counts are 9 CPI / 9 NFP / 6 FOMC because the window ends 2026-09-30.
- NFP 2026-02-11 falls on a Wednesday — that is the date BLS lists (post-
  shutdown schedule), not a transcription error.
- Cross-check: the independently maintained repo files
  `research/close_fade_program/fomc_dates.txt` (46 dates) and
  `research/close_fade_program/cpi_dates.txt` (68 dates), covering 2020-09 →
  2026-05, are both exact subsets of this calendar (0 mismatches).

Build script (data tables + sanity checks): session scratchpad
`build_calendar.py`; checks enforced at build time: dates parse, all within
[2008-01-01, 2026-09-30], no duplicate (date, event) pairs.
