"""Loader for the UCI Heart Disease processed files (4 sites)."""

from __future__ import annotations

import numpy as np
import pandas as pd

COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
SITES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}


def load_site(root: str, site: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(f"{root}/{SITES[site]}", header=None, names=COLS, na_values="?")
    df.loc[df["chol"] == 0, "chol"] = np.nan  # 0 mg/dl is a missing-value code
    y = (df["num"] > 0).astype(int).to_numpy()
    return df[FEATURES].astype(float), y
