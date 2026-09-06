"""TFM3 study — robustness: per-year surface pinball loss ratios (concentration check).

Brini (2026) found foundation-model gains concentrated in a few outlier assets; the
analogue here is concentration in a few calendar episodes. Reads the archived loss
series (deliverable 3) and prints per-year loss ratios vs pcavar / har / rw, plus
full-sample and ex-2020 (COVID) aggregates, raw and MZ stages.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tfm3_common import DIR, HS  # noqa: E402


def main() -> None:
    for h in HS:
        f = DIR / f"losses_trackA_h{h}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        dates = pd.DatetimeIndex(z["dates_full"].astype("datetime64[ns]"))
        for stage in ("raw", "mz"):
            models = sorted({k.split("_", 2)[2] for k in z.files
                             if k.startswith(f"{stage}_surface_")})
            cands = [m for m in models if m.startswith("tfm")]
            if not cands:
                continue
            print(f"\n=== trackA h={h} [{stage}] per-year loss ratio (candidate / ref):")
            for cand in cands:
                lc = z[f"{stage}_surface_{cand}"]
                for ref in ("pcavar", "har", "rw"):
                    lr = z[f"{stage}_surface_{ref}"]
                    s = pd.DataFrame({"c": lc, "r": lr}, index=dates).dropna()
                    yr = s.groupby(s.index.year).mean()
                    ratios = (yr.c / yr.r)
                    line = " ".join(f"{y}:{v:.3f}" for y, v in ratios.items())
                    full = s.c.mean() / s.r.mean()
                    ex20 = s[s.index.year != 2020]
                    ex = ex20.c.mean() / ex20.r.mean()
                    worst = ratios.max()
                    print(f"  {cand:12s}/{ref:7s} full {full:.3f}  ex-2020 {ex:.3f}  "
                          f"worst-yr {worst:.3f}\n    {line}")


if __name__ == "__main__":
    main()
