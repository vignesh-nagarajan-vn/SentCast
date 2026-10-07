"""Phase 1 data access check.

Downloads a small sample from every candidate source into data/samples/ and writes
results/data_check.json. Run: python -m sentcast.datacheck [--include-yahoo]
"""
from __future__ import annotations

import datetime as dt
import io
import time
import zipfile
from typing import Any, Callable

import pandas as pd
import requests

from .config import SAMPLES, load_config, write_result
from .net import USER_AGENT

def _get(url: str, *, timeout: int = 90, **kw: Any) -> requests.Response:
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, **kw)


def _save(name: str, content: bytes) -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    (SAMPLES / name).write_bytes(content)


def check_yahoo(refresh: bool = False) -> list[dict]:
    """Yahoo forbids automated collection without permission, so the samples are only
    downloaded when asked for (--include-yahoo). Otherwise the saved samples are described."""
    out = []
    for sym, label in [("^GSPC", "S&P 500 price index"), ("^SP500TR", "S&P 500 total return index")]:
        path = SAMPLES / f"yahoo_{sym.strip('^')}.csv"
        if refresh:
            import yfinance as yf

            df = yf.download(sym, period="max", interval="1d", progress=False, auto_adjust=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.to_csv(path)
        if not path.exists():
            out.append({"source": f"Yahoo Finance {sym}", "what": label, "ok": False,
                        "note": "no saved sample; run with --include-yahoo to download"})
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        out.append({"source": f"Yahoo Finance {sym}", "what": label, "access": "yfinance (unofficial API), no key",
                    "ok": len(df) > 0, "rows": len(df), "frequency": "daily",
                    "coverage_start": str(df.index.min().date()), "coverage_end": str(df.index.max().date()),
                    "note": "downloaded in this run" if refresh else "saved sample, not refreshed"})
    return out


def check_fred() -> list[dict]:
    cfg = load_config()
    out = []
    for series, label in [("SP500", "S&P 500 daily close"), ("UMCSENT", "UMich consumer sentiment"),
                          ("TB3MS", "3-month Treasury bill rate"),
                          ("NASDAQCOM", "Nasdaq Composite daily close (used for trading dates only)")]:
        url = cfg["sources"]["fred_csv"].format(series=series)
        r = _get(url)
        _save(f"fred_{series}.csv", r.content)
        df = pd.read_csv(io.BytesIO(r.content))
        df = df[pd.to_numeric(df.iloc[:, 1], errors="coerce").notna()]
        out.append({"source": f"FRED {series}", "what": label, "url": url,
                    "access": "CSV endpoint, no key", "status": r.status_code,
                    "ok": r.status_code == 200 and len(df) > 0, "bytes": len(r.content), "rows": len(df),
                    "coverage_start": str(df.iloc[0, 0]), "coverage_end": str(df.iloc[-1, 0])})
    return out


def check_goyal() -> list[dict]:
    cfg = load_config()["sources"]
    r = _get(cfg["goyal_xlsx"], timeout=180)
    _save("goyal_predictor_data.xlsx", r.content)
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Monthly")
    df = df[pd.to_numeric(df["Index"], errors="coerce").notna()]
    return [{"source": "Welch-Goyal predictor data (A. Goyal's website)", "what": "S&P 500 month-end index level, monthly",
             "url": cfg["goyal_page"], "access": "Google Sheets XLSX export, no key", "status": r.status_code,
             "ok": len(df) > 0, "bytes": len(r.content), "rows": len(df), "frequency": "monthly",
             "coverage_start": str(int(df["yyyymm"].iloc[0])), "coverage_end": str(int(df["yyyymm"].iloc[-1])),
             "columns": [str(c) for c in df.columns]}]


def check_umich() -> list[dict]:
    url = load_config()["sources"]["umich_ics_csv"]
    r = _get(url)
    _save("umich_tbmics.csv", r.content)
    df = pd.read_csv(io.BytesIO(r.content))
    df.columns = [c.strip() for c in df.columns]
    first = f"{df.iloc[0, 0].strip()} {int(df.iloc[0, 1])}"
    last = f"{df.iloc[-1, 0].strip()} {int(df.iloc[-1, 1])}"
    monthly = df[df.iloc[:, 1] >= 1978]
    return [{"source": "University of Michigan Surveys of Consumers", "what": "Index of Consumer Sentiment (tbmics.csv)",
             "url": url, "access": "direct CSV, no key", "status": r.status_code, "ok": r.status_code == 200,
             "bytes": len(r.content), "rows": len(df), "rows_since_1978": len(monthly),
             "coverage_start": first, "coverage_end": last, "last_value": float(df.iloc[-1, 2])}]


def check_stooq() -> list[dict]:
    url = "https://stooq.com/q/d/l/?s=^spx&i=d"
    r = _get(url)
    is_csv = r.text[:4].lower() == "date"
    return [{"source": "Stooq ^SPX", "what": "S&P 500 daily history", "url": url, "access": "CSV endpoint",
             "status": r.status_code, "ok": is_csv, "bytes": len(r.content),
             "note": "CSV returned" if is_csv else f"no CSV returned (HTTP {r.status_code})"}]


def check_sffed() -> list[dict]:
    url = load_config()["sources"]["sffed_xlsx"]
    r = _get(url)
    _save("sffed_news_sentiment.xlsx", r.content)
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data")
    return [{"source": "San Francisco Fed Daily News Sentiment Index", "what": "published benchmark index",
             "url": url, "access": "direct XLSX, no key", "status": r.status_code, "ok": len(df) > 0,
             "bytes": len(r.content), "rows": len(df), "frequency": "daily",
             "coverage_start": str(df["date"].min().date()), "coverage_end": str(df["date"].max().date())}]


def check_aaii() -> list[dict]:
    """AAII serves its spreadsheet to browsers only; the honest user agent is refused."""
    url = "https://www.aaii.com/files/surveys/sentiment.xls"
    r = _get(url)
    is_xls = r.content[:8] == bytes.fromhex("d0cf11e0a1b11ae1")  # OLE2 header of a real .xls
    return [{"source": "AAII Investor Sentiment Survey", "what": "weekly bullish/neutral/bearish shares", "url": url,
             "access": "direct XLS", "status": r.status_code, "ok": is_xls, "bytes": len(r.content),
             "note": None if is_xls else "refused for scripted access (bot protection)"}]


def check_epu() -> list[dict]:
    url = load_config()["sources"]["epu_xlsx"]
    r = _get(url)
    _save("epu_us_monthly.xlsx", r.content)
    df = pd.read_excel(io.BytesIO(r.content))
    df = df[pd.to_numeric(df["Year"], errors="coerce").notna()]
    months = [f"{int(y)}-{int(m):02d}" for y, m in zip(df["Year"], df["Month"])]
    return [{"source": "Economic Policy Uncertainty index (US, monthly)", "what": "published news-based index",
             "url": url, "access": "direct XLSX, no key", "status": r.status_code, "ok": len(df) > 0,
             "bytes": len(r.content), "rows": len(df), "frequency": "monthly", "columns": [str(c) for c in df.columns],
             "coverage_start": min(months), "coverage_end": max(months)}]


def check_gdelt_gkg() -> list[dict]:
    cfg = load_config()["gkg"]
    out = []
    r = _get(cfg["base_url"] + "lastupdate.txt")
    latest = [ln.split()[-1] for ln in r.text.strip().splitlines() if ".gkg.csv.zip" in ln]
    for ts in ["20150219120000", "20190701120000", "20191001120000", "20260930120000"]:
        url = f"{cfg['base_url']}{ts}.gkg.csv.zip"
        t0 = time.time()
        r = _get(url, timeout=180)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        raw = z.read(z.namelist()[0])
        rows = [x for x in raw.split(b"\n") if x]
        _save(f"gkg_{ts}.csv.zip", r.content)
        out.append({"source": "GDELT GKG 2.0 file", "what": f"15-minute file {ts}", "url": url,
                    "access": "direct ZIP, no key", "status": r.status_code, "ok": len(rows) > 0,
                    "bytes": len(r.content), "bytes_unzipped": len(raw), "rows": len(rows),
                    "rows_with_page_title": sum(1 for x in rows if b"<PAGE_TITLE>" in x),
                    "seconds": round(time.time() - t0, 2)})
    out.append({"source": "GDELT GKG 2.0 lastupdate.txt", "what": "latest file", "ok": bool(latest),
                "note": latest[0] if latest else "none"})
    return out


def check_gdelt_doc() -> list[dict]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": '(economy OR "stock market") sourcecountry:US sourcelang:english', "mode": "artlist",
              "maxrecords": 250, "format": "json", "startdatetime": "20170301000000",
              "enddatetime": "20170301235959", "sort": "dateasc"}
    t0 = time.time()
    n, attempts, statuses = None, 0, []
    for attempts in range(1, 4):
        r = _get(url, params=params)
        statuses.append(r.status_code)
        try:
            n = len(r.json().get("articles", []))
            _save("gdelt_doc_20170301.json", r.content)
            break
        except ValueError:
            time.sleep(20)
    return [{"source": "GDELT DOC 2.0 API", "what": "article list for one day (max 250 per call)", "url": url,
             "access": "JSON API, no key; one request per 5 seconds", "status": r.status_code, "ok": bool(n),
             "rows": n, "bytes": len(r.content), "seconds": round(time.time() - t0, 2),
             "attempts": attempts, "statuses": statuses, "note": None if n else r.text[:160]}]


def check_reddit() -> list[dict]:
    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    out = []
    for after, before in [("2008-09-15", "2008-09-16"), ("2019-03-04", "2019-03-05"), ("2026-09-01", "2026-09-02")]:
        try:
            r = _get(url, params={"subreddit": "economics", "after": after, "before": before, "limit": 100, "sort": "asc"})
            data = r.json().get("data") or []
            note = None if data else r.text[:120]
            status = r.status_code
        except (requests.RequestException, ValueError) as exc:
            data, note, status = [], f"{type(exc).__name__}", None
        out.append({"source": "Reddit via Arctic Shift archive API", "what": f"r/economics posts {after}", "url": url,
                    "access": "unofficial archive API, no key", "status": status, "ok": bool(data), "rows": len(data),
                    "note": note})
        time.sleep(2)
    return out


def check_wikipedia() -> list[dict]:
    url = "https://en.wikipedia.org/w/api.php"
    out = []
    for title in ["Portal:Current events/2006 May 10", "Portal:Current events/2019 August 14"]:
        r = _get(url, params={"action": "query", "prop": "revisions", "titles": title, "rvprop": "timestamp|content",
                              "rvslots": "main", "rvlimit": 1, "format": "json", "formatversion": 2})
        page = r.json()["query"]["pages"][0]
        text = page["revisions"][0]["slots"]["main"]["content"] if "revisions" in page else ""
        out.append({"source": "Wikipedia Current Events portal", "what": title, "url": url,
                    "access": "MediaWiki API, no key", "status": r.status_code, "ok": bool(text),
                    "rows": sum(1 for ln in text.splitlines() if ln.startswith("*"))})
    return out


def check_models() -> list[dict]:
    cfg = load_config()["scorers"]
    out = []
    r = _get(cfg["lm"]["dictionary_url"], timeout=180)
    head = r.content[:200].decode("utf-8", errors="replace")
    ok = head.startswith("Word,")
    if ok:
        _save("lm_master_dictionary.csv", r.content)
        lm = pd.read_csv(io.BytesIO(r.content))
        note = f"{len(lm)} words; {int((lm['Negative'] > 0).sum())} negative; {int((lm['Positive'] > 0).sum())} positive"
    else:
        note = "not a CSV"
    out.append({"source": "Loughran-McDonald master dictionary", "what": "finance sentiment word lists",
                "url": cfg["lm"]["dictionary_page"], "access": "Google Drive direct link, no key",
                "status": r.status_code, "ok": ok, "bytes": len(r.content), "note": note})
    r = _get(cfg["clf"]["train_url"])
    z = zipfile.ZipFile(io.BytesIO(r.content))
    _save("FinancialPhraseBank-v1.0.zip", r.content)
    lines = z.read(cfg["clf"]["train_file"]).decode("latin-1").strip().splitlines()
    out.append({"source": "Financial PhraseBank", "what": "labelled sentences for the trained classifier",
                "url": cfg["clf"]["train_url"], "access": "Hugging Face direct file, no key",
                "status": r.status_code, "ok": len(lines) > 0, "bytes": len(r.content), "rows": len(lines)})
    r = _get(f"https://huggingface.co/api/models/{cfg['finbert']['model']}")
    j = r.json()
    out.append({"source": f"Hugging Face model {cfg['finbert']['model']}", "what": "FinBERT weights",
                "access": "Hugging Face Hub, no key", "status": r.status_code,
                "ok": r.status_code == 200 and not j.get("gated"), "note": f"sha {j.get('sha')}"})
    return out


def cross_checks() -> dict:
    """Consistency checks between sources, computed from the samples just saved."""
    out: dict[str, Any] = {}
    y = pd.read_csv(SAMPLES / "yahoo_GSPC.csv", index_col=0, parse_dates=True)["Close"].dropna()
    f = pd.read_csv(SAMPLES / "fred_SP500.csv", index_col=0, parse_dates=True).iloc[:, 0]
    f = pd.to_numeric(f, errors="coerce").dropna()
    both = pd.concat([y.rename("yahoo"), f.rename("fred")], axis=1, sort=True).dropna()
    rel = (both["yahoo"] / both["fred"] - 1).abs()
    out["sp500_yahoo_vs_fred"] = {"overlap_days": len(both), "start": str(both.index.min().date()),
                                  "end": str(both.index.max().date()), "max_abs_rel_diff": float(rel.max()),
                                  "days_rel_diff_above_1e-4": int((rel > 1e-4).sum())}
    g = pd.read_excel(SAMPLES / "goyal_predictor_data.xlsx", sheet_name="Monthly")
    g.index = pd.PeriodIndex(g["yyyymm"].astype(int).astype(str), freq="M")
    for name, daily in [("fred", f), ("yahoo", y)]:
        me = daily.groupby(daily.index.to_period("M")).last()
        me = me[me.index <= g.index.max()].iloc[1:]  # first month of a daily series may be partial
        both = pd.concat([g["Index"].rename("goyal"), me.rename(name)], axis=1, sort=True).dropna()
        if name == "yahoo":
            both = both[both.index >= "1950-01"]
        rel = (both["goyal"] / both[name] - 1).abs()
        big = both[rel > 1e-3]
        out[f"sp500_month_end_goyal_vs_{name}"] = {
            "overlap_months": len(both), "start": str(both.index.min()), "end": str(both.index.max()),
            "max_abs_rel_diff": float(rel.max()), "months_rel_diff_above_1e-3": int((rel > 1e-3).sum()),
            "months_rel_diff_above_1e-4": int((rel > 1e-4).sum()),
            "months_above_1e-3": [{"month": str(m), "goyal": float(r["goyal"]), name: float(r[name])}
                                  for m, r in big.iterrows()]}
    u = pd.read_csv(SAMPLES / "umich_tbmics.csv")
    u.columns = [c.strip() for c in u.columns]
    u["date"] = pd.to_datetime(u["Month"].str.strip() + " " + u["YYYY"].astype(int).astype(str), format="%B %Y")
    u = u.set_index("date")["ICS_ALL"]
    fr = pd.read_csv(SAMPLES / "fred_UMCSENT.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fr = pd.to_numeric(fr, errors="coerce").dropna()
    both = pd.concat([u.rename("umich"), fr.rename("fred")], axis=1, sort=True).dropna()
    both = both[both.index >= "1978-01-01"]
    out["ics_umich_vs_fred"] = {"overlap_months": len(both), "max_abs_diff": float((both["umich"] - both["fred"]).abs().max()),
                                "umich_last": str(u.index.max().date()), "fred_last": str(fr.index.max().date())}
    # revision check: ALFRED vintage of 2015-06-30 against current values
    try:
        r = _get("https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=UMCSENT&vintage_date=2015-06-30")
        v = pd.read_csv(io.BytesIO(r.content), index_col=0, parse_dates=True).iloc[:, 0]
        v = pd.to_numeric(v, errors="coerce").dropna()
        both = pd.concat([v.rename("vintage"), u.rename("current")], axis=1, sort=True).dropna()
        diff = (both["vintage"] - both["current"]).abs()
        out["ics_vintage_2015_06_30_vs_current"] = {"months": len(both), "vintage_last": str(v.index.max().date()),
                                                    "months_differing": int((diff > 0.05).sum()),
                                                    "max_abs_diff": float(diff.max())}
    except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
        out["ics_vintage_2015_06_30_vs_current"] = {"error": type(exc).__name__}
    return out


CHECKS: list[tuple[str, Callable[[], list[dict]]]] = [
    ("yahoo", check_yahoo), ("fred", check_fred), ("goyal", check_goyal), ("umich", check_umich),
    ("stooq", check_stooq),
    ("sffed", check_sffed), ("epu", check_epu), ("aaii", check_aaii), ("gdelt_gkg", check_gdelt_gkg), ("gdelt_doc", check_gdelt_doc),
    ("reddit", check_reddit), ("wikipedia", check_wikipedia), ("models", check_models),
]


def main() -> None:
    import sys

    include_yahoo = "--include-yahoo" in sys.argv
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, fn in CHECKS:
        try:
            got = check_yahoo(refresh=include_yahoo) if name == "yahoo" else fn()
        except Exception as exc:  # noqa: BLE001 - a failed source is a finding, not a crash
            got = [{"source": name, "ok": False, "note": f"{type(exc).__name__}: {str(exc)[:200]}"}]
        for g in got:
            g["check"] = name
            print(f"{'OK ' if g.get('ok') else 'FAIL'} {g.get('source')}: "
                  f"{g.get('coverage_start', '')}..{g.get('coverage_end', '')} rows={g.get('rows')} {g.get('note') or ''}")
        rows.extend(got)
    try:
        cross = cross_checks()
    except Exception as exc:  # noqa: BLE001
        cross = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    path = write_result("data_check", {"checks": rows, "cross_checks": cross})
    print("wrote", path)


if __name__ == "__main__":
    main()
