"""Scoring step and index construction shared by the experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data
from .config import INTERIM, load_config
from .corpus import load_corpus
from .index import composite, monthly_index
from .scorers import SCORERS

SCORES_PATH = INTERIM / "scores.parquet"
TEXT_SCORERS = ["lm", "vader", "finbert", "clf"]
INDEX_COLS = TEXT_SCORERS + ["gdelt_tone"]


def _score_unique(texts: pd.Series, name: str) -> np.ndarray:
    """Score each distinct text once and map the result back to every row."""
    uniq = pd.unique(texts)
    scores = SCORERS[name](list(uniq))
    return pd.Series(scores, index=uniq).reindex(texts.values).values


def score_corpus() -> pd.DataFrame:
    """Score every kept headline, and every page title where one exists. Writes scores.parquet."""
    df = load_corpus()
    out = df[["gkg_id"]].copy()
    has_title = df["title"].notna() & (df["title"].str.len() > 0)
    for name in TEXT_SCORERS:
        print(f"scoring {len(df)} headlines with {name}", flush=True)
        out[name] = _score_unique(df["headline"], name)
        t = np.full(len(df), np.nan)
        if has_title.any():
            print(f"scoring {int(has_title.sum())} titles with {name}", flush=True)
            t[has_title.values] = _score_unique(df.loc[has_title, "title"], name)
        out[f"title_{name}"] = t
    out.to_parquet(SCORES_PATH, index=False)
    return out


def scored_corpus() -> pd.DataFrame:
    """Deduplicated corpus joined with its scores."""
    df = load_corpus()
    dedup = df.attrs.get("dedup", {})
    scores = pd.read_parquet(SCORES_PATH)
    if len(scores) != len(df) or not (scores["gkg_id"].values == df["gkg_id"].values).all():
        raise RuntimeError("scores.parquet does not match the corpus; rerun the score step")
    out = pd.concat([df.reset_index(drop=True), scores.drop(columns="gkg_id")], axis=1)
    out["gdelt_tone"] = out["tone"]
    out.attrs["dedup"] = dedup
    return out


def build_monthly(df: pd.DataFrame, tau: pd.Series, *, weighting: str = "day", cols: list[str] | None = None,
                  prefix: str = "") -> pd.DataFrame:
    """Monthly indices for the given score columns; prefix selects title-based scores."""
    cfg = load_config()["index"]
    cols = INDEX_COLS if cols is None else cols
    src = [f"{prefix}{c}" if c in TEXT_SCORERS else c for c in cols]
    work = df.dropna(subset=src) if prefix else df
    m = monthly_index(work, src, tau, min_headlines=cfg["min_headlines_per_day"],
                      min_days=cfg["min_days_per_month"], weighting=weighting)
    return m.rename(columns=dict(zip(src, cols)))


def add_composite(monthly: pd.DataFrame, origins: pd.PeriodIndex, n0: int) -> pd.DataFrame:
    """Append the composite, standardised on the first n0 origins of the sample."""
    members = load_config()["index"]["composite_members"]
    out = monthly.copy()
    out["composite"] = composite(monthly, members, origins[:n0])
    return out


def forecast_dates() -> pd.Series:
    return data.forecast_dates()
