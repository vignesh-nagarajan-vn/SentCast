"""Market, macro and published-index data: download and load (docs/data.md)."""
from __future__ import annotations

import warnings

import pandas as pd

from .config import RAW, load_config
from .net import fetch

FILES = {
    "goyal": "goyal_predictor_data.xlsx",
    "sp500": "fred_SP500.csv",
    "tbill": "fred_TB3MS.csv",
    "calendar": "fred_NASDAQCOM.csv",
    "ics": "umich_tbmics.csv",
    "sffed": "sffed_news_sentiment.xlsx",
    "epu": "epu_us_monthly.xlsx",
    "lm": "lm_master_dictionary.csv",
    "phrasebank": "FinancialPhraseBank-v1.0.zip",
}


def download_all(refresh: bool = False) -> None:
    src = load_config()["sources"]
    sc = load_config()["scorers"]
    urls = {
        "goyal": src["goyal_xlsx"],
        "sp500": src["fred_csv"].format(series=src["fred_sp500"]),
        "tbill": src["fred_csv"].format(series=src["fred_tbill"]),
        "calendar": src["fred_csv"].format(series=src["fred_calendar"]),
        "ics": src["umich_ics_csv"],
        "sffed": src["sffed_xlsx"],
        "epu": src["epu_xlsx"],
        "lm": sc["lm"]["dictionary_url"],
        "phrasebank": sc["clf"]["train_url"],
    }
    for key, url in urls.items():
        path = fetch(url, RAW / FILES[key], refresh=refresh)
        print(f"{key}: {path.name} ({path.stat().st_size} bytes)")


def _fred(name: str) -> pd.Series:
    df = pd.read_csv(RAW / FILES[name])
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def trading_days() -> pd.DatetimeIndex:
    """Dates with a Nasdaq Composite close; used only as a US trading calendar."""
    return _fred("calendar").index


def forecast_dates(days: pd.DatetimeIndex | None = None) -> pd.Series:
    """tau(m): the last trading day of each calendar month, indexed by month."""
    days = trading_days() if days is None else days
    s = pd.Series(days, index=days.to_period("M"))
    tau = s.groupby(level=0).max()
    last_full = days.max().to_period("M")
    if days.max() < (last_full.to_timestamp("M").normalize() - pd.Timedelta(days=3)):
        tau = tau[tau.index < last_full]   # the current month is not over
    return tau


def sp500_month_end(tau: pd.Series | None = None) -> pd.DataFrame:
    """S&P 500 level on tau(m). FRED where it covers the whole month, Welch-Goyal before."""
    tau = forecast_dates() if tau is None else tau
    fred = _fred("sp500")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = pd.read_excel(RAW / FILES["goyal"], sheet_name="Monthly")
    g = g[pd.to_numeric(g["Index"], errors="coerce").notna()]
    goyal = pd.Series(g["Index"].astype(float).values,
                      index=pd.PeriodIndex(g["yyyymm"].astype(int).astype(str), freq="M"))
    first_full = (fred.index.min() + pd.offsets.MonthBegin(1)).to_period("M") \
        if fred.index.min().day > 3 else fred.index.min().to_period("M")
    fred_month = fred.index.to_period("M")
    rows = []
    for m, day in tau.items():
        if m >= first_full:
            seg = fred[(fred_month == m) & (fred.index <= day)]   # close on tau(m), or the last one before it
            if len(seg):
                rows.append((m, float(seg.iloc[-1]), "fred"))
        elif m in goyal.index:
            rows.append((m, float(goyal[m]), "goyal"))
    out = pd.DataFrame(rows, columns=["month", "price", "source"]).set_index("month")
    return out


def tbill_monthly() -> pd.Series:
    s = _fred("tbill")
    s.index = s.index.to_period("M")
    return s


def ics_monthly() -> pd.Series:
    df = pd.read_csv(RAW / FILES["ics"])
    df.columns = [c.strip() for c in df.columns]
    idx = pd.PeriodIndex(pd.to_datetime(df["Month"].str.strip() + " " + df["YYYY"].astype(int).astype(str),
                                        format="%B %Y"), freq="M")
    s = pd.Series(pd.to_numeric(df["ICS_ALL"], errors="coerce").values, index=idx).dropna()
    return s[s.index >= "1978-01"]


def sffed_daily() -> pd.Series:
    df = pd.read_excel(RAW / FILES["sffed"], sheet_name="Data")
    return pd.Series(df["News Sentiment"].astype(float).values, index=pd.to_datetime(df["date"])).dropna()


def epu_monthly() -> pd.Series:
    df = pd.read_excel(RAW / FILES["epu"])
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    idx = pd.PeriodIndex([f"{int(y)}-{int(m):02d}" for y, m in zip(df["Year"], df["Month"])], freq="M")
    col = [c for c in df.columns if c not in ("Year", "Month")][0]
    return pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=idx).dropna().sort_index()
