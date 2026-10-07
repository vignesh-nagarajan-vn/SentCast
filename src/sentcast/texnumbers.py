"""Numbers for the paper text, generated from files in results/ only.

Writes results/tables/numbers.tex, which defines \\res{key} for every value the running
text of the paper may cite, and results/paper_numbers.csv with the source of each key.
The report step copies the .tex file to paper/tables/. No result is typed in paper/main.tex:
the text calls \\res{key}, and an unknown key stops the LaTeX run.

Keys, by example:
    corpus.dedup                 E01, headlines after de-duplication
    meas.split.finbert           E02, split-half reliability of one index
    E03.composite.m0.r2          E03, composite against M0: r2, lo, hi (interval), p, n
    E03.finbert.m1.padj          Bonferroni-adjusted p of an individual scorer against M1
    E04.composite.hit            E04, L2: hit, lo, hi, base, up, ptp, brier, brierbase, dmp, n
    E06.long.sp.sffed.m0.r2      E06, long sample or 'text' window; sp, dir or ics
    E08.initial36.sp.m0.r2       E08 and E09 variants; outcome.sp, outcome.dir, outcome.ics
    E12.a.post.ics.m1.r2         E12, first (a) or second (b) model; all, post or pre cutoff
    power.sp.r80                 power simulation: zero, at1, at2, at5, at10 (power), r50, r80 (true R2)

Run: python -m sentcast.report (or python -m sentcast.texnumbers)
"""
from __future__ import annotations

import calendar
import csv
import json

from .config import RESULTS, TABLES

NAME = "numbers.tex"
SCORERS = {"composite": "composite", "finbert": "finbert", "clf": "clf", "lm": "lm", "vader": "vader",
           "gdelt": "gdelt_tone"}
PUBLISHED = {"sffed": "sffed_nsi", "epu": "epu", "icslevel": "ics_level"}
R2_LEAVES = (("r2", "r2_os", "pct"), ("lo", "r2_ci_lo", "pct"), ("hi", "r2_ci_hi", "pct"), ("p", "p", "p"),
             ("padj", "p_bonferroni", "p"), ("n", "n_oos", "int"))
DIR_LEAVES = (("hit", "hit_rate", "pct1"), ("lo", "hit_ci_lo", "pct1"), ("hi", "hit_ci_hi", "pct1"),
              ("base", "hit_rate_base", "pct1"), ("up", "share_up_predicted", "pct1"), ("ptp", "pt_p", "p"),
              ("brier", "brier", "num4"), ("brierbase", "brier_base", "num4"), ("dmp", "dm_brier_p", "p"),
              ("n", "n_oos", "int"))
MISSING = object()


def _load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _get(node, path: str):
    for key in path.split("."):
        if isinstance(node, list):
            node = node[int(key)] if key.isdigit() and int(key) < len(node) else MISSING
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return MISSING
        if node is MISSING:
            return MISSING
    return node


def _minus(text: str) -> str:
    """A leading hyphen is a minus sign; \\ensuremath keeps it right in text and in maths."""
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def fmt(x, kind: str) -> str:
    if kind == "str":
        return str(x)
    if kind == "month":                       # 2019-04 -> April 2019
        year, month = str(x).split("-")[:2]
        return f"{calendar.month_name[int(month)]} {year}"
    if kind == "int":
        return f"{int(round(x)):,}"
    if kind == "count":                       # a median of counts can end in .5
        return f"{int(x):,}" if float(x) == int(x) else f"{x:,.1f}"
    if kind == "gb":
        return f"{x / 1e9:.1f}"
    if kind == "p":
        return "\\ensuremath{<}0.001" if x < 0.001 else f"{x:.3f}"
    if kind == "pct":
        return _minus(f"{100 * x:.2f}")
    if kind == "pct1":
        return _minus(f"{100 * x:.1f}")
    if kind in ("num", "num1", "num4"):
        return _minus(f"{x:.{ {'num': 2, 'num1': 1, 'num4': 4}[kind] }f}")
    if kind == "len":
        return f"{len(x):,}"
    raise ValueError(f"unknown format {kind!r}")


def collect() -> list[dict]:
    """Every key with its source and rendered value. Keys whose source is absent are left out."""
    rows: list[dict] = []
    files: dict[str, dict | None] = {}

    def add(key: str, name: str, path: str, kind: str) -> None:
        if name not in files:
            files[name] = _load(name)
        value = _get(files[name], path) if files[name] is not None else MISSING
        if value is MISSING or value is None:
            return
        rows.append({"key": key, "file": f"results/{name}.json", "path": path, "format": kind,
                     "value": fmt(value, kind)})

    def r2(key: str, name: str, path: str) -> None:
        for suffix, leaf, kind in R2_LEAVES:
            add(f"{key}.{suffix}", name, f"{path}.{leaf}", kind)

    def direction(key: str, name: str, path: str) -> None:
        for suffix, leaf, kind in DIR_LEAVES:
            add(f"{key}.{suffix}", name, f"{path}.{leaf}", kind)

    def sample(key: str, name: str, path: str) -> None:
        for suffix, leaf, kind in (("n", "n_oos", "int"), ("origins", "n_origins", "int"),
                                   ("window", "initial_window", "int"), ("first", "first_oos_target", "month"),
                                   ("last", "last_oos_target", "month"), ("firstorigin", "first_origin", "month"),
                                   ("lastorigin", "last_origin", "month")):
            add(f"{key}.{suffix}", name, f"{path}.{leaf}", kind)

    def continuous(key: str, name: str, path: str, predictors: dict[str, str]) -> None:
        r2(f"{key}.ar", name, f"{path}.baselines.M1_vs_M0")
        for short, long in predictors.items():
            r2(f"{key}.{short}.m0", name, f"{path}.{long}.M2_vs_M0")
            r2(f"{key}.{short}.m1", name, f"{path}.{long}.M3_vs_M1")

    def directional(key: str, name: str, path: str, predictors: dict[str, str]) -> None:
        direction(f"{key}.L0", name, f"{path}.baselines.L0")
        direction(f"{key}.L1", name, f"{path}.baselines.L1")
        for short, long in predictors.items():
            direction(f"{key}.{short}", name, f"{path}.{long}.L2")

    # E01 corpus
    c = "E01_corpus"
    for key, path, kind in (
            ("requested", "files.requested", "int"), ("downloaded", "files.downloaded", "int"),
            ("replaced", "files.replaced_by_next_slot", "int"), ("nofile", "files.failed", "int"),
            ("daysnofile", "files.days_with_no_file", "int"), ("days", "files.days", "int"),
            ("gb", "files.bytes_downloaded", "gb"), ("rows", "funnel.rows", "int"), ("us", "funnel.us", "int"),
            ("headline", "funnel.headline", "int"), ("keyword", "funnel.keyword", "int"),
            ("kept", "funnel.kept", "int"), ("dedup", "dedup.after_headline_dedup", "int"),
            ("inindex", "headlines_per_index_month.total", "int"),
            ("months", "headlines_per_index_month.months", "int"),
            ("permonth.min", "headlines_per_index_month.min", "int"),
            ("permonth.median", "headlines_per_index_month.median", "count"),
            ("permonth.max", "headlines_per_index_month.max", "int"),
            ("validdays.min", "valid_days_per_index_month.min", "int"),
            ("validdays.minmonth", "valid_days_per_index_month.month_of_min", "month"),
            ("validdays.median", "valid_days_per_index_month.median", "count"),
            ("failcoverage", "n_index_months_failing_coverage", "int"),
            ("title.share", "titles.share_with_title_overall", "pct1"),
            ("title.first", "titles.first_month_at_threshold", "month"),
            ("title.months", "titles.months_at_threshold", "int"), ("domains", "domains.distinct", "int"),
            ("top10", "domains.top10_share", "pct1"), ("mix.yeara", "source_mix.year_a", "str"),
            ("mix.yearb", "source_mix.year_b", "str"),
            ("mix.share", "source_mix.share_of_year_b_headlines_from_domains_seen_in_year_a", "pct1"),
            ("mix.top20", "source_mix.top20_domains_in_common", "int"),
            ("keywords", "config.n_keywords", "int"), ("pressdomains", "config.gkg.excluded_domains", "len")):
        add(f"corpus.{key}", c, path, kind)

    # E02 measurement
    m = "E02_measurement"
    add("meas.n", m, "n_headlines", "int")
    for short, long in SCORERS.items():
        add(f"meas.zero.{short}", m, f"score_stats.{long}.share_zero", "pct1")
        add(f"meas.split.{short}", m, f"c_split_half.by_index.{long}.spearman_brown", "num")
        add(f"meas.title.headline.{short}", m, f"b_url_headline_vs_title.headline_level.{long}.pearson", "num")
        add(f"meas.title.month.{short}", m, f"b_url_headline_vs_title.month_level.{long}.pearson", "num")
        for other, col in (("sffed", "level~sffed_nsi"), ("ics", "level~ics"), ("epu", "level~epu"),
                           ("retsame", "level~ret_same_month"), ("retnext", "level~ret_next_month"),
                           ("icssame", "level~ics_change_same_month"), ("icsnext", "level~ics_change_next_month"),
                           ("dsffed", "diff~diff_sffed_nsi"), ("dics", "diff~diff_ics")):
            add(f"meas.corr.{short}.{other}", m, f"d_external.{long}.{col}.pearson", "num")
    add("meas.split.n", m, "c_split_half.by_index.finbert.n", "int")
    add("meas.title.headline.n", m, "b_url_headline_vs_title.headline_level.finbert.n", "int")
    add("meas.title.month.n", m, "b_url_headline_vs_title.month_level.finbert.n", "int")
    add("meas.title.same", m, "b_url_headline_vs_title.share_url_headline_equal_to_normalised_title", "pct1")
    pairs = (("lm", "vader"), ("lm", "finbert"), ("lm", "clf"), ("vader", "finbert"), ("vader", "clf"), ("finbert", "clf"))
    for a, b in pairs:
        add(f"meas.agree.headline.{a}.{b}", m, f"a_scorer_agreement.headline_level.{a}~{b}.pearson", "num")
        add(f"meas.agree.month.{a}.{b}", m, f"a_scorer_agreement.month_level.{a}~{b}.pearson", "num")
    add("meas.agree.month.n", m, "a_scorer_agreement.month_level.finbert~clf.n", "int")
    for key, path, kind in (("clf.n", "clf_5fold.n", "int"), ("clf.acc", "clf_5fold.accuracy_mean", "pct1"),
                            ("clf.f1", "clf_5fold.macro_f1_mean", "pct1"), ("lm.acc", "lm.accuracy", "pct1"),
                            ("vader.acc", "vader.accuracy", "pct1"), ("finbert.acc", "finbert.accuracy", "pct1"),
                            ("majority", "majority_class_share", "pct1")):
        add(f"meas.labelled.{key}", m, f"e_labelled_data.{path}", kind)

    # E03 to E05, the primary experiments
    for exp, name in (("E03", "E03_sp500_return"), ("E05", "E05_ics_change")):
        continuous(exp, name, "results", SCORERS)
        sample(exp, name, "sample")
        add(f"{exp}.outcome", name, "decision.outcome", "str")
    directional("E04", "E04_sp500_direction", "results", SCORERS)
    sample("E04", "E04_sp500_direction", "sample")
    add("E04.outcome", "E04_sp500_direction", "decision.outcome", "str")
    add("E04.upshare", "E04_sp500_direction", "results.baselines.L0.share_up_actual", "pct1")

    # months with the largest gain and loss of the composite against M0 (what Figure 2 cumulates)
    for exp in ("E03", "E05"):
        path = RESULTS / f"{exp}_forecasts.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            months = [r for r in csv.DictReader(fh) if r["M0"] and r["M2_composite"]]
        gain = {r["target_month"]: (float(r["y"]) - float(r["M0"])) ** 2 - (float(r["y"]) - float(r["M2_composite"])) ** 2
                for r in months}
        what = "(y - M0)^2 - (y - M2_composite)^2"
        source = f"results/{exp}_forecasts.csv"
        for key, month in (("best", max(gain, key=gain.get)), ("worst", min(gain, key=gain.get))):
            side = "largest" if key == "best" else "smallest"
            rows.append({"key": f"{exp}.cssed.{key}", "file": source, "path": f"target month with the {side} {what}",
                         "format": "month", "value": fmt(month, "month")})
            rows.append({"key": f"{exp}.cssed.{key}value", "file": source, "path": f"{side} monthly {what}",
                         "format": "num1", "value": fmt(gain[month], "num1")})
        rows.append({"key": f"{exp}.cssed.total", "file": source, "path": f"sum over the test months of {what}",
                     "format": "num1", "value": fmt(sum(gain.values()), "num1")})

    # E06 published indices, E07 composite added to them
    p = "E06_published"
    sample("E06", p, "sample")
    add("E06.text.n", p, "text_window.n_oos", "int")
    for block, source in (("long", "long"), ("text", "text_window")):
        continuous(f"E06.{block}.sp", p, f"results.{source}.sp500_return", PUBLISHED)
        continuous(f"E06.{block}.ics", p, f"results.{source}.ics_change", PUBLISHED)
        directional(f"E06.{block}.dir", p, f"results.{source}.sp500_direction", PUBLISHED)
    i = "E07_incremental"
    for target, long in (("sp", "sp500_return"), ("ics", "ics_change")):
        for short, block in (("sffed", "a_sffed"), ("all", "b_all_published"), ("lagret", "c_lagged_sp500_return")):
            r2(f"E07.{target}.{short}", i, f"results.{long}.{block}")

    # E08 and E09 variants of the three primary experiments
    for exp, name, variants in (
            ("E08", "E08_estimation", {"initial36": "initial_36", "initial60": "initial_60", "rolling48": "rolling_48"}),
            ("E09", "E09_index_definition", {"diff": "a_first_difference", "weighted": "b_headline_weighted",
                                             "title": "c_title_index", "urlsame": "d_url_index_same_months"})):
        for short, long in variants.items():
            base = f"results.{long}"
            r2(f"{exp}.{short}.sp.m0", name, f"{base}.sp500_return.composite.M2_vs_M0")
            r2(f"{exp}.{short}.sp.m1", name, f"{base}.sp500_return.composite.M3_vs_M1")
            r2(f"{exp}.{short}.ics.m0", name, f"{base}.ics_change.composite.M2_vs_M0")
            r2(f"{exp}.{short}.ics.m1", name, f"{base}.ics_change.composite.M3_vs_M1")
            direction(f"{exp}.{short}.dir", name, f"{base}.sp500_direction.composite.L2")
            sample(f"{exp}.{short}", name, f"{base}.sample")
            for target, key in (("sp", "sp500_return"), ("dir", "sp500_direction"), ("ics", "ics_change")):
                add(f"{exp}.{short}.outcome.{target}", name, f"{base}.decisions.{key}.outcome", "str")
    add("E09.titlemonths.first", "E09_index_definition", "title_months.0", "month")
    add("E09.titlemonths.last", "E09_index_definition", "title_months.1", "month")

    # E10 target definitions, E11 three-month horizon
    t = "E10_target_definition"
    for short, long in (("raw", "a_raw_log_return"), ("simple", "b_simple_excess_return"),
                        ("strict", "c_ics_own_lag_release_lag_1")):
        r2(f"E10.{short}.m0", t, f"results.{long}.results.composite.M2_vs_M0")
        r2(f"E10.{short}.m1", t, f"results.{long}.results.composite.M3_vs_M1")
        add(f"E10.{short}.outcome", t, f"results.{long}.decision.outcome", "str")
    h = "E11_horizon3"
    sample("E11", h, "sample")
    for short, long in (("sp", "sp500_return_3m"), ("ics", "ics_change_3m")):
        r2(f"E11.{short}.m0", h, f"results.{long}.results.composite.M2_vs_M0")
        r2(f"E11.{short}.m1", h, f"results.{long}.results.composite.M3_vs_M1")
        add(f"E11.{short}.outcome", h, f"results.{long}.decision.outcome", "str")

    # E12 LLM scorers: first model (a), second model (b); all test months, post and pre cutoff
    e = "E12_llm"
    add("E12.status", e, "status", "str")
    for model, root in (("a", ""), ("b", "second_model.")):
        for key, path, kind in (("model", "usage.model", "str"), ("labelled", "usage.labelled", "int"),
                                ("texts", "usage.texts", "int"), ("requests", "usage.requests", "int"),
                                ("unusable", "usage.discarded_batches", "int"), ("cost", "usage.cost_usd", "num"),
                                ("lastrun", "usage.requested_this_run", "int"),
                                ("months", "index_months_with_llm_index", "int"),
                                ("cutoff", "by_training_cutoff.training_cutoff", "month"),
                                ("post.months", "by_training_cutoff.post_cutoff.n_test_origins", "int"),
                                ("post.firstorigin", "by_training_cutoff.post_cutoff.first_origin", "month"),
                                ("pre.months", "by_training_cutoff.pre_cutoff.n_test_origins", "int")):
            add(f"E12.{model}.{key}", e, f"{root}{path}", kind)
        for subset, path in (("all", "results"), ("post", "by_training_cutoff.post_cutoff.results"),
                             ("pre", "by_training_cutoff.pre_cutoff.results")):
            base = f"{root}{path}"
            r2(f"E12.{model}.{subset}.sp.m0", e, f"{base}.sp500_return.llm.M2_vs_M0")
            r2(f"E12.{model}.{subset}.sp.m1", e, f"{base}.sp500_return.llm.M3_vs_M1")
            r2(f"E12.{model}.{subset}.ics.m0", e, f"{base}.ics_change.llm.M2_vs_M0")
            r2(f"E12.{model}.{subset}.ics.m1", e, f"{base}.ics_change.llm.M3_vs_M1")
            direction(f"E12.{model}.{subset}.dir", e, f"{base}.sp500_direction.llm.L2")
    # power simulation (not pre-specified)
    add("power.reps", "power", "config.reps", "int")
    for short, target in (("sp", "sp500_return"), ("ics", "ics_change")):
        base = f"results.{target}"
        add(f"power.{short}.zero", "power", f"{base}.at_zero.power", "pct1")
        for key, leaf in (("at1", "r2_1pct"), ("at2", "r2_2pct"), ("at5", "r2_5pct"), ("at10", "r2_10pct")):
            add(f"power.{short}.{key}", "power", f"{base}.selected.{leaf}.power", "pct1")
        add(f"power.{short}.r50", "power", f"{base}.smallest_r2.power_50", "pct1")
        add(f"power.{short}.r80", "power", f"{base}.smallest_r2.power_80", "pct1")
    return rows


def render(rows: list[dict]) -> str:
    # nothing under paper/ may name the project (double-blind), so the header stays generic
    lines = ["% Generated by the report step of the pipeline from the files in results/. Do not edit.",
             "% \\res{key} prints one value; an unknown key stops the LaTeX run.",
             "% Source of every key: results/paper_numbers.csv.",
             "\\makeatletter",
             "\\DeclareRobustCommand{\\res}[1]{\\@ifundefined{res@#1}{\\errmessage{Unknown result key: #1}}{\\@nameuse{res@#1}}}"]
    lines += [f"\\@namedef{{res@{r['key']}}}{{{r['value']}}}" for r in rows]
    return "\n".join(lines + ["\\makeatother", ""])


def write() -> list[dict]:
    rows = collect()
    keys = [r["key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate result keys: " + ", ".join(sorted({k for k in keys if keys.count(k) > 1})))
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / NAME).write_text(render(rows), encoding="utf-8", newline="\n")
    with open(RESULTS / "paper_numbers.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "file", "path", "format", "value"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    print(f"wrote {len(write())} keys to {TABLES / NAME}")
