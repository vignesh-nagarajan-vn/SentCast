"""Numbered experiments E01 .. E12 (PLAN.md section 9).

Each experiment writes results/E##_<name>.json through config.write_result, which adds
the git commit and config hash. Nothing here chooses a setting from the evaluation data.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import data, features, pipeline, scorers
from . import evaluate as ev
from .config import INTERIM, RESULTS, load_config, write_result
from .corpus import FUNNEL, load_manifest
from .forecast import logit_forecasts, ols_forecasts

PREDICTORS = pipeline.INDEX_COLS + ["composite"]
PRIMARY = "composite"


# --- shared inputs ----------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _tau() -> pd.Series:
    return data.forecast_dates()


@functools.lru_cache(maxsize=1)
def _base() -> pd.DataFrame:
    return features.base_table(_tau())


@functools.lru_cache(maxsize=1)
def _text() -> pd.DataFrame:
    return features.text_sample(_base())


@functools.lru_cache(maxsize=1)
def _scored() -> pd.DataFrame:
    return pipeline.scored_corpus()


@functools.lru_cache(maxsize=4)
def _monthly(weighting: str = "day", prefix: str = "") -> pd.DataFrame:
    return pipeline.build_monthly(_scored(), _tau(), weighting=weighting, prefix=prefix)


def _indices(n0: int, origins: pd.PeriodIndex | None = None, weighting: str = "day", prefix: str = "") -> pd.DataFrame:
    """Monthly indices on the given origins, with the composite standardised on the first n0 of them."""
    origins = _text().index if origins is None else origins
    m = _monthly(weighting, prefix).reindex(origins)
    return pipeline.add_composite(m, origins, n0)


def _common_config(**extra) -> dict:
    cfg = load_config()
    return {"seed": cfg["seed"], "sample": cfg["sample"], "index": cfg["index"], "evaluation": cfg["evaluation"],
            "release_lag_months": cfg["release_lag_months"], "publication_lag_days": cfg["publication_lag_days"],
            **extra}


# --- evaluation helpers -----------------------------------------------------------------

def _block(h: int) -> int:
    e = load_config()["evaluation"]
    return int(e["bootstrap_mean_block"]) * (2 if h > 1 else 1)


def compare(y, f_bench, f_model, *, h: int, key: str, nested: bool = True, subset=None) -> dict:
    """Out-of-sample comparison of a model with a benchmark on the origins where both forecast."""
    cfg = load_config()
    e = cfg["evaluation"]
    y, fb, fm = (np.asarray(a, dtype=float) for a in (y, f_bench, f_model))
    mask = np.isfinite(y) & np.isfinite(fb) & np.isfinite(fm)
    if subset is not None:
        mask &= np.asarray(subset, dtype=bool)
    n = int(mask.sum())
    if n < 12:
        return {"n_oos": n, "note": "too few out-of-sample forecasts"}
    y, fb, fm = y[mask], fb[mask], fm[mask]
    eb, em = y - fb, y - fm
    lo, hi = ev.bootstrap_ci_r2(em, eb, mean_block=_block(h), reps=e["bootstrap_reps"], level=e["ci_level"],
                                seed=cfg["seed"], key=key)
    if nested:
        stat, p = ev.clark_west(y, fb, fm, h)
        test = "clark_west_one_sided"
    else:
        stat, p = ev.diebold_mariano(np.square(em), np.square(eb), h)
        test = "diebold_mariano_two_sided"
    return {"n_oos": n, "rmse_model": ev.rmse(em), "rmse_bench": ev.rmse(eb), "mae_model": ev.mae(em),
            "mae_bench": ev.mae(eb), "r2_os": ev.r2_os(em, eb), "r2_ci_lo": lo, "r2_ci_hi": hi,
            "test": test, "stat": stat, "p": p}


def _bonferroni(res: dict, comparisons: list[str]) -> None:
    """Add Bonferroni-adjusted p-values across the individual text scorers, in place."""
    k = len(pipeline.TEXT_SCORERS)
    for name in pipeline.TEXT_SCORERS:
        for c in comparisons:
            block = res.get(name, {}).get(c)
            if block and block.get("p") is not None:
                block["p_bonferroni"] = min(1.0, block["p"] * k)


def eval_continuous(tab: pd.DataFrame, y_col: str, lag_col: str, predictors: pd.DataFrame, *, n0: int, h: int = 1,
                    scheme: str = "expanding", window: int | None = None, tag: str, subset=None,
                    names: list[str] | None = None) -> tuple[dict, pd.DataFrame]:
    """M0..M3 for each predictor column. Returns (results, forecasts by origin)."""
    tab = tab[tab[y_col].notna()]
    y = tab[y_col].values
    ylag = tab[lag_col].values
    kw = dict(n0=n0, h=h, scheme=scheme, window=window)
    f0 = ols_forecasts(y, None, **kw)
    f1 = ols_forecasts(y, ylag, **kw)
    sub = None if subset is None else np.asarray(subset.reindex(tab.index).values, dtype=bool)
    out = {"baselines": {"M1_vs_M0": compare(y, f0, f1, h=h, key=f"{tag}|M1", subset=sub)}}
    fc = pd.DataFrame({"y": y, "M0": f0, "M1": f1}, index=tab.index)
    for name in (names or list(predictors.columns)):
        x = predictors[name].reindex(tab.index).values.astype(float)
        f2 = ols_forecasts(y, x, **kw)
        f3 = ols_forecasts(y, np.column_stack([ylag, x]), **kw)
        out[name] = {"M2_vs_M0": compare(y, f0, f2, h=h, key=f"{tag}|{name}|M2", subset=sub),
                     "M3_vs_M1": compare(y, f1, f3, h=h, key=f"{tag}|{name}|M3", subset=sub),
                     "x_missing_origins": int(np.isnan(x).sum())}
        fc[f"M2_{name}"], fc[f"M3_{name}"] = f2, f3
    _bonferroni(out, ["M2_vs_M0", "M3_vs_M1"])
    return out, fc


def _direction_stats(u, p, p0, *, h: int, key: str, subset=None) -> dict:
    cfg = load_config()
    e = cfg["evaluation"]
    u, p, p0 = (np.asarray(a, dtype=float) for a in (u, p, p0))
    mask = np.isfinite(u) & np.isfinite(p) & np.isfinite(p0)
    if subset is not None:
        mask &= np.asarray(subset, dtype=bool)
    n = int(mask.sum())
    if n < 12:
        return {"n_oos": n, "note": "too few out-of-sample forecasts"}
    u, p, p0 = u[mask], p[mask], p0[mask]
    pred, pred0 = (p >= 0.5).astype(float), (p0 >= 0.5).astype(float)
    hits = (pred == u).astype(float)
    lo, hi = ev.bootstrap_ci_mean(hits, mean_block=_block(h), reps=e["bootstrap_reps"], level=e["ci_level"],
                                  seed=cfg["seed"], key=key)
    pt_stat, pt_p = ev.pesaran_timmermann(u, pred)
    dm_stat, dm_p = ev.diebold_mariano(np.square(p - u), np.square(p0 - u), h)
    return {"n_oos": n, "hit_rate": float(hits.mean()), "hit_ci_lo": lo, "hit_ci_hi": hi,
            "hit_rate_base": float((pred0 == u).mean()), "share_up_actual": float(u.mean()),
            "share_up_predicted": float(pred.mean()), "pt_stat": pt_stat, "pt_p": pt_p,
            "brier": float(np.mean(np.square(p - u))), "brier_base": float(np.mean(np.square(p0 - u))),
            "dm_brier_stat": dm_stat, "dm_brier_p": dm_p}


def eval_direction(tab: pd.DataFrame, predictors: pd.DataFrame, *, n0: int, h: int = 1, scheme: str = "expanding",
                   window: int | None = None, tag: str, subset=None, names: list[str] | None = None,
                   u_col: str = "y_up", lag_col: str = "lag_ret_excess") -> tuple[dict, pd.DataFrame]:
    """L0..L3 for each predictor column."""
    tab = tab[tab[u_col].notna()]
    u = tab[u_col].values
    lag = tab[lag_col].values
    kw = dict(n0=n0, h=h, scheme=scheme, window=window)
    p0 = logit_forecasts(u, None, **kw)
    p1 = logit_forecasts(u, lag, **kw)
    sub = None if subset is None else np.asarray(subset.reindex(tab.index).values, dtype=bool)
    out = {"baselines": {"L0": _direction_stats(u, p0, p0, h=h, key=f"{tag}|L0", subset=sub),
                         "L1": _direction_stats(u, p1, p0, h=h, key=f"{tag}|L1", subset=sub)}}
    fc = pd.DataFrame({"u": u, "L0": p0, "L1": p1}, index=tab.index)
    for name in (names or list(predictors.columns)):
        x = predictors[name].reindex(tab.index).values.astype(float)
        p2 = logit_forecasts(u, x, **kw)
        p3 = logit_forecasts(u, np.column_stack([lag, x]), **kw)
        out[name] = {"L2": _direction_stats(u, p2, p0, h=h, key=f"{tag}|{name}|L2", subset=sub),
                     "L3": _direction_stats(u, p3, p0, h=h, key=f"{tag}|{name}|L3", subset=sub)}
        fc[f"L2_{name}"], fc[f"L3_{name}"] = p2, p3
    k = len(pipeline.TEXT_SCORERS)
    for name in pipeline.TEXT_SCORERS:
        for m in ("L2", "L3"):
            b = out.get(name, {}).get(m)
            if b and b.get("pt_p") is not None and np.isfinite(b["pt_p"]):
                b["pt_p_bonferroni"] = min(1.0, b["pt_p"] * k)
    return out, fc


def decide_continuous(res: dict) -> dict:
    alpha = load_config()["evaluation"]["alpha"]
    a, b = res[PRIMARY]["M2_vs_M0"], res[PRIMARY]["M3_vs_M1"]
    ok_a = bool(a.get("r2_os", -1) > 0 and (a.get("p") is not None) and a["p"] < alpha)
    ok_b = bool(b.get("r2_os", -1) > 0 and (b.get("p") is not None) and b["p"] < alpha)
    outcome = "positive" if (ok_a and ok_b) else "mixed" if (ok_a or ok_b) else "null"
    return {"primary_predictor": PRIMARY, "alpha": alpha, "M2_beats_M0": ok_a, "M3_beats_M1": ok_b, "outcome": outcome}


def decide_direction(res: dict) -> dict:
    alpha = load_config()["evaluation"]["alpha"]
    b = res[PRIMARY]["L2"]
    p = b.get("pt_p")
    ok = bool(b.get("hit_rate", 0) > b.get("hit_rate_base", 1) and p is not None and np.isfinite(p) and p < alpha)
    return {"primary_predictor": PRIMARY, "alpha": alpha, "L2_hit_above_base": bool(b.get("hit_rate", 0) > b.get("hit_rate_base", 1)),
            "pt_significant": bool(p is not None and np.isfinite(p) and p < alpha), "outcome": "positive" if ok else "null"}


def _sample_info(tab: pd.DataFrame, n0: int, h: int = 1) -> dict:
    first_oos = n0 + h - 1
    return {"first_origin": str(tab.index[0]), "last_origin": str(tab.index[-1]), "n_origins": len(tab),
            "initial_window": n0, "horizon": h, "n_oos": max(0, len(tab) - first_oos),
            "first_oos_target": str(tab.index[first_oos] + 1) if len(tab) > first_oos else None,
            "last_oos_target": str(tab.index[-1] + h)}


def _save_forecasts(name: str, fc: pd.DataFrame, tab: pd.DataFrame) -> str:
    out = fc.copy()
    out.insert(0, "forecast_date", pd.to_datetime(tab["forecast_date"].reindex(out.index)).dt.strftime("%Y-%m-%d").values)
    out.insert(0, "target_month", [str(m + 1) for m in out.index])
    out.index = out.index.astype(str)
    out.index.name = "origin_month"
    path = RESULTS / f"{name}.csv"
    out.round(6).to_csv(path, lineterminator="\n")
    return path.name


# --- E01 corpus -------------------------------------------------------------------------

def e01_corpus() -> None:
    cfg = load_config()
    man = load_manifest()
    df = pipeline.load_corpus() if not (INTERIM / "scores.parquet").exists() else _scored()
    dedup = df.attrs.get("dedup", {})
    tau = _tau()
    from .index import assign_windows

    df = df.assign(window=assign_windows(df["date"], tau))
    origins = _text().index
    icfg = cfg["index"]
    per_day = df.groupby(["window", "date"]).size()
    valid = per_day[per_day >= icfg["min_headlines_per_day"]]
    n_days = valid.groupby(level="window").size().reindex(origins).fillna(0).astype(int)
    n_head = valid.groupby(level="window").sum().reindex(origins).fillna(0).astype(int)
    has_title = df["title"].notna() & (df["title"].str.len() > 0)
    share_title = has_title.groupby(df["window"]).mean().reindex(origins)
    counts = pd.DataFrame({"n_headlines": n_head, "n_valid_days": n_days, "share_with_title": share_title.round(4)})
    counts.index = counts.index.astype(str)
    counts.index.name = "index_month"
    counts.to_csv(RESULTS / "E01_monthly_counts.csv", lineterminator="\n")

    ok = man[man["ok"].astype(bool)]
    year = df["date"].dt.year
    dom = df["domain"].value_counts()
    title_months = share_title[share_title >= icfg["title_month_min_share"]]
    by_year = {}
    for y, g in df.groupby(year):
        m = ok[pd.to_datetime(ok["date"]).dt.year == y]
        by_year[int(y)] = {"files": int(len(m)), "kept_per_file": float(m["kept"].mean()) if len(m) else None,
                           "headlines_after_dedup": int(len(g)), "distinct_domains": int(g["domain"].nunique()),
                           "share_with_title": float((g["title"].notna() & (g["title"].str.len() > 0)).mean())}
    # how much the set of sources changes: first and last calendar year with at least ten months
    months_in_year = df.groupby(year)["date"].apply(lambda d: d.dt.month.nunique())
    full = [int(y) for y in months_in_year.index[months_in_year >= 10]]
    ya, yb = full[0], full[-1]
    da, db = df.loc[year == ya, "domain"], df.loc[year == yb, "domain"]
    source_mix = {
        "year_a": ya, "year_b": yb,
        "share_of_year_b_headlines_from_domains_seen_in_year_a": float(db.isin(set(da)).mean()),
        "share_of_year_a_headlines_from_domains_seen_in_year_b": float(da.isin(set(db)).mean()),
        "top20_domains_in_common": int(len(set(da.value_counts().head(20).index)
                                           & set(db.value_counts().head(20).index))),
    }
    failed_draw = ~man["ok"].astype(bool)
    failed_per_day = failed_draw.groupby(man["date"]).sum()
    payload = {
        "experiment": "E01", "name": "corpus",
        "files": {"requested": int(len(man)), "downloaded": int(len(ok)), "failed": int(len(man) - len(ok)),
                  "unavailable_by_month": {str(k): int(v) for k, v in
                                           man[failed_draw].groupby(
                                               man["date"].astype(str).str[:7]).size().items()},
                  "days_with_no_file": int((failed_per_day == cfg["gkg"]["files_per_day"]).sum()),
                  "days_with_one_file": int(((failed_per_day > 0)
                                             & (failed_per_day < cfg["gkg"]["files_per_day"])).sum()),
                  "max_slots_tried": int(man["slots_tried"].max()) if "slots_tried" in man else None,
                  "replaced_by_next_slot": int(ok["replaced"].astype(bool).sum()),
                  "bytes_downloaded": int(ok["bytes"].sum()), "first_day": str(man["date"].min()),
                  "last_day": str(man["date"].max()), "days": int(man["date"].nunique())},
        "funnel": {k: int(ok[k].sum()) for k in FUNNEL},
        "dedup": dedup,
        "headlines_per_index_month": {"months": int(len(origins)), "min": int(n_head.min()),
                                      "median": float(n_head.median()), "mean": float(n_head.mean()),
                                      "max": int(n_head.max()), "total": int(n_head.sum())},
        "valid_days_per_index_month": {"min": int(n_days.min()), "month_of_min": str(n_days.idxmin()),
                                       "median": float(n_days.median()), "max": int(n_days.max())},
        "index_months_failing_coverage": [str(m) for m in origins[n_days.values < icfg["min_days_per_month"]]],
        "n_index_months_failing_coverage": int((n_days.values < icfg["min_days_per_month"]).sum()),
        "titles": {"share_with_title_overall": float(has_title.mean()),
                   "first_month_at_threshold": str(title_months.index.min()) if len(title_months) else None,
                   "months_at_threshold": int(len(title_months)), "threshold": icfg["title_month_min_share"]},
        "domains": {"distinct": int(dom.size), "top10_share": float(dom.head(10).sum() / dom.sum()),
                    "top20": [{"domain": d, "share": float(n / dom.sum())} for d, n in dom.head(20).items()]},
        "source_mix": source_mix,
        "by_year": by_year,
        "files_out": ["E01_monthly_counts.csv"],
    }
    write_result("E01_corpus", payload, _common_config(gkg={k: v for k, v in cfg["gkg"].items() if k != "keywords"},
                                                       n_keywords=len(cfg["gkg"]["keywords"])))


# --- E02 measurement --------------------------------------------------------------------

def _corr(a: pd.Series, b: pd.Series) -> dict:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 8:
        return {"n": int(len(d))}
    return {"n": int(len(d)), "pearson": float(d.iloc[:, 0].corr(d.iloc[:, 1])),
            "spearman": float(d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman"))}


def e02_measurement() -> None:
    cfg = load_config()
    df = _scored()
    text = _text()
    origins = text.index
    n0 = cfg["sample"]["initial_window"]
    idx = _indices(n0)
    ts_cols = pipeline.TEXT_SCORERS
    all_cols = pipeline.INDEX_COLS

    # (a) agreement between scorers
    head = df[all_cols].dropna()
    agree_headline = {f"{a}~{b}": {"n": int(len(head)), "pearson": float(head[a].corr(head[b])),
                                   "spearman": float(head[a].corr(head[b], method="spearman"))}
                      for i, a in enumerate(all_cols) for b in all_cols[i + 1:]}
    agree_month = {f"{a}~{b}": _corr(idx[a], idx[b]) for i, a in enumerate(all_cols) for b in all_cols[i + 1:]}
    score_stats = {c: {"mean": float(df[c].mean()), "sd": float(df[c].std()), "share_zero": float((df[c] == 0).mean())}
                   for c in all_cols}

    # (b) URL headline against page title
    tcols = [f"title_{c}" for c in ts_cols]
    both = df.dropna(subset=tcols)
    url_vs_title_headline = {c: {"n": int(len(both)), "pearson": float(both[c].corr(both[f"title_{c}"])),
                                 "spearman": float(both[c].corr(both[f"title_{c}"], method="spearman"))}
                             for c in ts_cols}
    title_monthly = _monthly("day", "title_").reindex(origins)
    counts = pd.read_csv(RESULTS / "E01_monthly_counts.csv", index_col=0)
    tmonths = pd.PeriodIndex(counts.index[counts["share_with_title"] >= cfg["index"]["title_month_min_share"]], freq="M")
    url_vs_title_month = {c: _corr(idx.loc[idx.index.isin(tmonths), c], title_monthly.loc[title_monthly.index.isin(tmonths), c])
                          for c in ts_cols}
    same_text = float((both["headline"] == both["title"].map(scorers.normalise)).mean()) if len(both) else None

    # (c) split-half reliability from the two daily files
    half_min = int(np.ceil(cfg["index"]["min_headlines_per_day"] / 2))
    from .index import monthly_index

    halves = [monthly_index(df[df["rank"] == r], all_cols, _tau(), min_headlines=half_min,
                            min_days=cfg["index"]["min_days_per_month"]).reindex(origins) for r in (0, 1)]
    split_half = {}
    for c in all_cols:
        r = _corr(halves[0][c], halves[1][c])
        if "pearson" in r:
            r["spearman_brown"] = float(2 * r["pearson"] / (1 + r["pearson"]))
        split_half[c] = r

    # (d) relation to published measures and to the market, by origin month
    t = features.targets(_tau())
    ext = pd.DataFrame(index=origins)
    ext["sffed_nsi"] = text["sffed_nsi"]
    ext["ics"] = t["ics"].reindex(origins)
    ext["epu"] = data.epu_monthly().reindex(origins)
    ext["ret_same_month"] = t["ret_excess"].reindex(origins)
    ext["ret_next_month"] = text["y_ret_excess"]
    ext["ics_change_same_month"] = t["ics_change"].reindex(origins)
    ext["ics_change_next_month"] = text["y_ics_change"]
    external = {}
    for c in all_cols + ["composite"]:
        external[c] = {f"level~{e}": _corr(idx[c], ext[e]) for e in ext.columns}
        external[c].update({f"diff~diff_{e}": _corr(idx[c].diff(), ext[e].diff()) for e in ("sffed_nsi", "ics", "epu")})

    # (e) accuracy on labelled data
    pb = scorers.load_phrasebank()
    texts, gold = pb["text"].tolist(), pb["label"].tolist()
    from sklearn.metrics import accuracy_score, f1_score

    labelled = {"clf_5fold": scorers.clf_cross_validation()}
    preds = {"lm": scorers.labels_from_scores(scorers.score_lm(texts)),
             "vader": scorers.labels_from_scores(scorers.score_vader(texts)),
             "finbert": scorers.finbert_labels(texts)}
    for k, p in preds.items():
        labelled[k] = {"n": len(gold), "accuracy": float(accuracy_score(gold, p)),
                       "macro_f1": float(f1_score(gold, p, average="macro"))}
    labelled["finbert"]["note"] = "FinBERT was fine-tuned on this dataset; this is not a held-out score"
    labelled["majority_class_share"] = float(pb["label"].value_counts(normalize=True).max())
    labelled["lexicon_class_threshold"] = 0.05   # score above +0.05 positive, below -0.05 negative

    # monthly indices for the figures; published series only as z-scores
    out = idx.copy()
    for e in ("sffed_nsi", "ics", "epu"):
        out[f"{e}_z"] = (ext[e] - ext[e].mean()) / ext[e].std(ddof=0)
    out.insert(0, "forecast_date", pd.to_datetime(text["forecast_date"]).dt.strftime("%Y-%m-%d").values)
    out.index = out.index.astype(str)
    out.index.name = "index_month"
    out.round(6).to_csv(RESULTS / "E02_indices_monthly.csv", lineterminator="\n")

    write_result("E02_measurement", {
        "experiment": "E02", "name": "measurement", "n_headlines": int(len(df)), "n_index_months": int(len(origins)),
        "score_stats": score_stats,
        "a_scorer_agreement": {"headline_level": agree_headline, "month_level": agree_month},
        "b_url_headline_vs_title": {"headline_level": url_vs_title_headline, "month_level": url_vs_title_month,
                                    "title_months": int(len(tmonths)),
                                    "share_url_headline_equal_to_normalised_title": same_text},
        "c_split_half": {"min_headlines_per_day_in_each_half": half_min, "by_index": split_half},
        "d_external": external,
        "e_labelled_data": labelled,
        "files_out": ["E02_indices_monthly.csv"],
    }, _common_config())


# --- E03, E04, E05 main forecasts -------------------------------------------------------

def e03_sp500_return() -> None:
    n0 = load_config()["sample"]["initial_window"]
    tab, idx = _text(), _indices(n0)
    res, fc = eval_continuous(tab, "y_ret_excess", "lag_ret_excess", idx, n0=n0, tag="E03", names=PREDICTORS)
    write_result("E03_sp500_return", {
        "experiment": "E03", "name": "sp500_return", "target": "S&P 500 log excess return, percent", "sample": _sample_info(tab, n0),
        "results": res, "decision": decide_continuous(res), "files_out": [_save_forecasts("E03_forecasts", fc, tab)],
    }, _common_config(models="M0 mean; M1 own lag; M2 index; M3 own lag + index"))


def e04_sp500_direction() -> None:
    n0 = load_config()["sample"]["initial_window"]
    tab, idx = _text(), _indices(n0)
    res, fc = eval_direction(tab, idx, n0=n0, tag="E04", names=PREDICTORS)
    write_result("E04_sp500_direction", {
        "experiment": "E04", "name": "sp500_direction", "target": "1 if the S&P 500 month-end close is above the previous one",
        "sample": _sample_info(tab, n0), "results": res, "decision": decide_direction(res),
        "files_out": [_save_forecasts("E04_forecasts", fc, tab)],
    }, _common_config(models="L0 base rate; L1 lagged excess return; L2 index; L3 both"))


def e05_ics_change() -> None:
    n0 = load_config()["sample"]["initial_window"]
    tab, idx = _text(), _indices(n0)
    res, fc = eval_continuous(tab, "y_ics_change", "lag_ics_change", idx, n0=n0, tag="E05", names=PREDICTORS)
    write_result("E05_ics_change", {
        "experiment": "E05", "name": "ics_change", "target": "monthly change in the Index of Consumer Sentiment, points",
        "sample": _sample_info(tab, n0), "results": res, "decision": decide_continuous(res),
        "files_out": [_save_forecasts("E05_forecasts", fc, tab)],
    }, _common_config(models="M0 mean; M1 own lag; M2 index; M3 own lag + index"))


# --- E06 published indices --------------------------------------------------------------

def e06_published() -> None:
    cfg = load_config()
    n0 = cfg["sample"]["long_initial_window"]
    tab = features.long_sample(_base())
    short = pd.Series(tab.index.isin(_text().index[cfg["sample"]["initial_window"]:]), index=tab.index)
    px = pd.DataFrame({"sffed_nsi": tab["sffed_nsi"], "epu": tab["epu_for_sp500"], "ics_level": tab["ics_level_for_sp500"]})
    pi = pd.DataFrame({"sffed_nsi": tab["sffed_nsi"], "epu": tab["epu_for_ics"]})
    out = {}
    for label, sub in (("long", None), ("text_window", short)):
        r1, _ = eval_continuous(tab, "y_ret_excess", "lag_ret_excess", px, n0=n0, tag=f"E06|{label}|ret", subset=sub)
        r2, _ = eval_direction(tab, px, n0=n0, tag=f"E06|{label}|dir", subset=sub)
        r3, _ = eval_continuous(tab, "y_ics_change", "lag_ics_change", pi, n0=n0, tag=f"E06|{label}|ics", subset=sub)
        out[label] = {"sp500_return": r1, "sp500_direction": r2, "ics_change": r3}
    write_result("E06_published", {
        "experiment": "E06", "name": "published", "sample": _sample_info(tab, n0),
        "text_window": {"first_target": str(_text().index[cfg["sample"]["initial_window"]] + 1),
                        "last_target": str(_text().index[-1] + 1), "n_oos": int(short.sum())},
        "predictors": {"sffed_nsi": "San Francisco Fed Daily News Sentiment Index, window mean shifted back 7 days",
                       "epu": "news-based US Economic Policy Uncertainty index, lagged one month",
                       "ics_level": "Index of Consumer Sentiment level, lagged one month (S&P 500 targets only)"},
        "results": out,
    }, _common_config(note="models estimated on the long sample; text_window evaluates the same forecasts on the text-sample months"))


# --- E07 incremental value --------------------------------------------------------------

def _incremental(tab: pd.DataFrame, y_col: str, controls: list[str], x: pd.Series, *, n0: int, key: str) -> dict:
    y = tab[y_col].values
    C = tab[controls].values.astype(float)
    xv = x.reindex(tab.index).values.astype(float)
    fb = ols_forecasts(y, C, n0=n0)
    fm = ols_forecasts(y, np.column_stack([C, xv]), n0=n0)
    r = compare(y, fb, fm, h=1, key=key)
    r["controls"] = controls
    return r


def e07_incremental() -> None:
    n0 = load_config()["sample"]["initial_window"]
    tab, idx = _text(), _indices(n0)
    x = idx[PRIMARY]
    res = {
        "sp500_return": {
            "a_sffed": _incremental(tab, "y_ret_excess", ["sffed_nsi"], x, n0=n0, key="E07|ret|a"),
            "b_all_published": _incremental(tab, "y_ret_excess", ["sffed_nsi", "epu_for_sp500", "ics_level_for_sp500"], x,
                                            n0=n0, key="E07|ret|b"),
        },
        "ics_change": {
            "a_sffed": _incremental(tab, "y_ics_change", ["lag_ics_change", "sffed_nsi"], x, n0=n0, key="E07|ics|a"),
            "b_all_published": _incremental(tab, "y_ics_change", ["lag_ics_change", "sffed_nsi", "epu_for_ics"], x,
                                            n0=n0, key="E07|ics|b"),
            "c_lagged_sp500_return": _incremental(tab, "y_ics_change", ["lag_ics_change", "lag_ret_excess_for_ics"], x,
                                                  n0=n0, key="E07|ics|c"),
        },
    }
    write_result("E07_incremental", {
        "experiment": "E07", "name": "incremental", "sample": _sample_info(tab, n0), "added_predictor": PRIMARY,
        "results": res,
    }, _common_config(note="benchmark = constant + controls; model = benchmark + composite; ICS models keep the own lag"))


# --- E08 .. E11 robustness --------------------------------------------------------------

def _trio(tab: pd.DataFrame, idx: pd.DataFrame, *, n0: int, tag: str, scheme: str = "expanding",
          window: int | None = None) -> dict:
    """E03, E04 and E05 on one variant; compact output."""
    kw = dict(n0=n0, scheme=scheme, window=window, names=PREDICTORS)
    r1, _ = eval_continuous(tab, "y_ret_excess", "lag_ret_excess", idx, tag=f"{tag}|ret", **kw)
    r2, _ = eval_direction(tab, idx, tag=f"{tag}|dir", **kw)
    r3, _ = eval_continuous(tab, "y_ics_change", "lag_ics_change", idx, tag=f"{tag}|ics", **kw)
    return {"sample": _sample_info(tab, n0), "sp500_return": r1, "sp500_direction": r2, "ics_change": r3,
            "decisions": {"sp500_return": decide_continuous(r1), "sp500_direction": decide_direction(r2),
                          "ics_change": decide_continuous(r3)}}


def e08_estimation() -> None:
    tab = _text()
    res = {
        "initial_36": _trio(tab, _indices(36), n0=36, tag="E08|n36"),
        "initial_60": _trio(tab, _indices(60), n0=60, tag="E08|n60"),
        "rolling_48": _trio(tab, _indices(48), n0=48, tag="E08|roll48", scheme="rolling", window=48),
    }
    write_result("E08_estimation", {"experiment": "E08", "name": "estimation", "results": res},
                 _common_config(note="composite standardised on the initial window of each variant"))


def e09_index_definition() -> None:
    cfg = load_config()
    n0 = cfg["sample"]["initial_window"]
    tab = _text()
    level = _indices(n0)
    res = {"a_first_difference": _trio(tab.iloc[1:], level.diff().iloc[1:], n0=n0, tag="E09|diff"),
           "b_headline_weighted": _trio(tab, _indices(n0, weighting="headline"), n0=n0, tag="E09|hw")}
    counts = pd.read_csv(RESULTS / "E01_monthly_counts.csv", index_col=0)
    tmonths = pd.PeriodIndex(counts.index[counts["share_with_title"] >= cfg["index"]["title_month_min_share"]], freq="M")
    tmonths = tab.index[tab.index.isin(tmonths)]
    if len(tmonths) >= 36 + 12:
        sub = tab.loc[tmonths]
        res["c_title_index"] = _trio(sub, _indices(36, origins=tmonths, prefix="title_"), n0=36, tag="E09|title")
        res["d_url_index_same_months"] = _trio(sub, _indices(36, origins=tmonths), n0=36, tag="E09|url")
    else:
        res["c_title_index"] = {"status": "not run", "reason": f"only {len(tmonths)} months have page titles"}
        res["d_url_index_same_months"] = res["c_title_index"]
    write_result("E09_index_definition", {"experiment": "E09", "name": "index_definition",
                                          "title_months": [str(m) for m in tmonths[[0, -1]]] if len(tmonths) else [],
                                          "results": res}, _common_config())


def e10_target_definition() -> None:
    n0 = load_config()["sample"]["initial_window"]
    tab, idx = _text(), _indices(n0)
    kw = dict(n0=n0, names=PREDICTORS)
    a, _ = eval_continuous(tab, "y_ret_raw", "lag_ret_raw", idx, tag="E10|raw", **kw)
    b, _ = eval_continuous(tab, "y_ret_simple_excess", "lag_ret_simple_excess", idx, tag="E10|simple", **kw)
    # ICS at release lag 1: at the forecast date of origin m the latest known value is ICS(m-1).
    # So the own lag is one month older, and estimation stops one origin earlier (h=2 in the
    # training slice), because the target of origin m-1 needs ICS(m).
    strict = tab.copy()
    strict["lag_ics_change"] = tab["lag_ics_change"].shift(1)
    c, _ = eval_continuous(strict.iloc[1:], "y_ics_change", "lag_ics_change", idx, tag="E10|icslag1", h=2, **kw)
    res = {"a_raw_log_return": {"results": a, "decision": decide_continuous(a)},
           "b_simple_excess_return": {"results": b, "decision": decide_continuous(b)},
           "c_ics_own_lag_release_lag_1": {
               "results": c, "decision": decide_continuous(c),
               "note": "own lag and estimation targets both respect a one-month release lag; "
                       "Newey-West with 1 lag; bootstrap mean block 12"}}
    write_result("E10_target_definition", {"experiment": "E10", "name": "target_definition",
                                           "sample": _sample_info(tab, n0), "results": res}, _common_config())


def e11_horizon3() -> None:
    n0 = load_config()["sample"]["initial_window"]
    tab, idx = _text(), _indices(n0)
    h = 3
    r1, f1 = eval_continuous(tab, "y3_ret_excess", "lag_ret_excess", idx, n0=n0, h=h, tag="E11|ret", names=PREDICTORS)
    r2, f2 = eval_continuous(tab, "y3_ics_change", "lag_ics_change", idx, n0=n0, h=h, tag="E11|ics", names=PREDICTORS)
    sub = tab[tab["y3_ret_excess"].notna()]
    write_result("E11_horizon3", {
        "experiment": "E11", "name": "horizon3", "sample": _sample_info(sub, n0, h),
        "results": {"sp500_return_3m": {"results": r1, "decision": decide_continuous(r1)},
                    "ics_change_3m": {"results": r2, "decision": decide_continuous(r2)}},
    }, _common_config(note="targets overlap; estimation uses only targets realised by the forecast date; "
                           "Newey-West with 2 lags; bootstrap mean block 12"))


# --- E12 LLM scorer ---------------------------------------------------------------------

def llm_post_cutoff(origins: pd.PeriodIndex, cutoff: str) -> pd.Series:
    """True for origins whose text window, tau(m-1) to tau(m), lies wholly after the cutoff month."""
    return pd.Series(origins > pd.Period(cutoff, freq="M") + 1, index=origins)


def e12_llm() -> None:
    cfg = load_config()
    lcfg = cfg["scorers"]["llm"]
    if not scorers.llm_available():
        write_result("E12_llm", {"experiment": "E12", "name": "llm", "status": "skipped",
                                 "reason": f"environment variable {lcfg['env_key']} is not set"},
                     _common_config(llm=lcfg))
        return
    first, *rest = scorers.llm_configs()
    out = {"experiment": "E12", "name": "llm", "status": "ran", **_llm_block(first, "E12")}
    if rest:
        out["second_model"] = _llm_block(rest[0], "E12b")
    write_result("E12_llm", out, _common_config(llm=lcfg, note="llm index built like the other scorer indices "
                                                              "(PLAN.md section 5) from every kept headline"))


def _llm_block(mcfg: dict, tag: str) -> dict:
    """E03 to E05 with the index of one LLM, on all test origins and split at its training cutoff."""
    df = _scored()
    tab = _text()
    n0 = load_config()["sample"]["initial_window"]
    scores, usage = scorers.score_llm(df["headline"].tolist(), mcfg)
    labelled = df.assign(llm=scores).dropna(subset=["llm"])
    idx = pipeline.build_monthly(labelled, _tau(), cols=["llm"])[["llm"]].reindex(tab.index)

    def trio(t: str, subset=None) -> dict:
        r1, _ = eval_continuous(tab, "y_ret_excess", "lag_ret_excess", idx, n0=n0, tag=f"{t}|ret", subset=subset)
        r2, _ = eval_direction(tab, idx, n0=n0, tag=f"{t}|dir", subset=subset)
        r3, _ = eval_continuous(tab, "y_ics_change", "lag_ics_change", idx, n0=n0, tag=f"{t}|ics", subset=subset)
        return {"sp500_return": r1, "sp500_direction": r2, "ics_change": r3}

    post = llm_post_cutoff(tab.index, mcfg["training_cutoff"])
    oos = pd.Series(np.arange(len(tab)) >= n0, index=tab.index)
    return {
        "usage": usage, "index_months_with_llm_index": int(idx["llm"].notna().sum()),
        "sample": _sample_info(tab, n0),
        "caveat": "the model's training data may postdate the headlines it scores",
        "results": trio(tag),
        "by_training_cutoff": {
            "training_cutoff": mcfg["training_cutoff"],
            "rule": "post-cutoff origins are those whose whole text window lies after the cutoff month",
            "post_cutoff": {"n_test_origins": int((post & oos).sum()),
                            "first_origin": str(tab.index[post.values][0]) if post.any() else None,
                            "results": trio(f"{tag}|post", post)},
            "pre_cutoff": {"n_test_origins": int((~post & oos).sum()), "results": trio(f"{tag}|pre", ~post)},
            "note": "the split restricts evaluation only; models are still estimated on all earlier origins, "
                    "which include pre-cutoff index values",
        },
    }


EXPERIMENTS = {
    "E01": e01_corpus, "E02": e02_measurement, "E03": e03_sp500_return, "E04": e04_sp500_direction,
    "E05": e05_ics_change, "E06": e06_published, "E07": e07_incremental, "E08": e08_estimation,
    "E09": e09_index_definition, "E10": e10_target_definition, "E11": e11_horizon3, "E12": e12_llm,
}


def run(only: list[str] | None = None) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for key, fn in EXPERIMENTS.items():
        if only and key not in only:
            continue
        print(f"== {key} {fn.__name__}", flush=True)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a failed experiment is reported, not hidden
            write_result(f"{key}_FAILED", {"experiment": key, "status": "failed",
                                           "error": f"{type(exc).__name__}: {exc}"}, _common_config())
            print(f"   FAILED: {type(exc).__name__}: {exc}", flush=True)
            raise
        (RESULTS / f"{key}_FAILED.json").unlink(missing_ok=True)
