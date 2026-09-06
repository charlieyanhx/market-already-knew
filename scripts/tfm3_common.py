"""TFM3 study — shared series definitions. Both venvs import this; keep it dependency-light."""
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1] / "data"
TENORS = [7, 30, 60, 90]
NODES = [f"{c}_{t}" for t in TENORS for c in ("p25", "atm", "c25")]
QS = np.arange(0.1, 0.91, 0.1)
HS = [1, 5]
FIT_WIN = 756


def load_trackA() -> pd.DataFrame:
    panel = pd.read_parquet(DIR / "panel_trackA.parquet")
    return np.log(panel[NODES].clip(lower=1e-4))


def load_trackB() -> pd.DataFrame:
    # th7 excluded per prereg amendment A3 (sparse pre-2017); Track B is supplementary (A2)
    F = pd.read_parquet(DIR / "params_trackB.parquet")
    g = F.gamma.clip(1e-4, 0.994)
    return pd.DataFrame({
        **{f"lth{t}": np.log(F[f"th{t}"].clip(lower=1e-8)) for t in (30, 60, 90)},
        "arho": np.arctanh(F.rho.clip(-0.999, 0.999)),
        "leta": np.log(F.eta.clip(lower=1e-6)),
        "lgam": np.log(g / (0.995 - g)),
    }, index=F.index)


def load_track(name: str) -> pd.DataFrame:
    return load_trackA() if name == "trackA" else load_trackB()
