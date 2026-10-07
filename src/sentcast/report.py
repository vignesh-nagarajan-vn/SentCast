"""Tables, figures and results/SUMMARY.md, generated from files in results/ only.

No number is typed here. Tables are LaTeX tabular bodies without captions; figures carry
axis and legend labels only. Run: python -m sentcast.report
"""
from __future__ import annotations

import json
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from . import texnumbers  # noqa: E402
from .config import DRY_RUN, FIGURES, RESULTS, ROOT, TABLES, load_config  # noqa: E402
from .features import release_lag  # noqa: E402

PAPER = ROOT / "paper"
LABEL = {"composite": "Composite", "finbert": "FinBERT", "clf": "Classifier", "lm": "LM lexicon",
         "vader": "VADER", "gdelt_tone": "GDELT tone", "sffed_nsi": "SF Fed index", "epu": "EPU index",
         "ics_level": "ICS level", "llm": "LLM"}
TEXT_ORDER = ["composite", "finbert", "clf", "lm", "vader", "gdelt_tone"]
PUB_ORDER = ["sffed_nsi", "epu", "ics_level"]
# validated with the dataviz palette checker on a white surface (light mode): all checks pass
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#e3e2de"
PALE_BLUE = "#d5e5f8"
INDEX_ORDER = ["lm", "vader", "finbert", "clf", "gdelt_tone"]
PDF_META = {"Creator": None, "Producer": None, "CreationDate": None, "Title": None, "Author": None}


def _load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --- number formatting ------------------------------------------------------------------

def pct(x, d: int = 2) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}"


def num(x, d: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def pval(x) -> str:
    if x is None:
        return "n/a"
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def r2_cell(b: dict | None) -> tuple[str, str, str]:
    """(R2 in percent, interval, p) for one comparison block."""
    if not b or "r2_os" not in b:
        return "n/a", "n/a", "n/a"
    return pct(b["r2_os"]), f"[{pct(b['r2_ci_lo'])}, {pct(b['r2_ci_hi'])}]", pval(b.get("p"))


# a hyphen that starts a number is a minus sign; hyphens inside dates and names stay as they are
_MINUS = re.compile(r"(?<![\w$\\}])-(?=\d)")


def _groups(*spans: tuple[int, int, str]) -> list[str]:
    """Header lines that name groups of columns: (first column, last column, title), columns counted from 1."""
    cells, col = [], 1
    for first, last, title in spans:
        cells += [""] * (first - col) + [f"\\multicolumn{{{last - first + 1}}}{{c}}{{{title}}}"]
        col = last + 1
    return [" & ".join(cells) + " \\\\", "".join(f"\\cmidrule(lr){{{a}-{b}}}" for a, b, _ in spans)]


def _tabular(spec: str, header: list[str], rows: list[list[str]], groups: list[str] | None = None) -> str:
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", *(groups or []), " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == ["---"]:
            if lines[-1] != "\\midrule":      # never two rules in a row
                lines.append("\\midrule")
        else:
            lines.append(" & ".join(_MINUS.sub("$-$", c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _write(name: str, text: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(text, encoding="utf-8", newline="\n")


def _int(x) -> str:
    return f"{int(x):,}".replace(",", "{,}")


def _count(x) -> str:
    """A count, with one decimal when it is not a whole number (a median can end in .5)."""
    return _int(x) if float(x) == int(x) else f"{x:,.1f}".replace(",", "{,}")


def _count_text(x) -> str:
    return f"{int(x):,}" if float(x) == int(x) else f"{x:,.1f}"


# --- tables -----------------------------------------------------------------------------

def table_corpus() -> None:
    e = _load("E01_corpus")
    if not e:
        return
    f, fu, d, h, t = e["files"], e["funnel"], e["dedup"], e["headlines_per_index_month"], e["titles"]
    rows = [
        ["Calendar days in the draw", _int(f["days"])],
        ["Files requested", _int(f["requested"])],
        ["Files downloaded", _int(f["downloaded"])],
        ["Files replaced by another slot of the same day", _int(f["replaced_by_next_slot"])],
        ["Draws left without a file", _int(f["failed"])],
        ["Days with one file / with no file", f"{_int(f['days_with_one_file'])} / {_int(f['days_with_no_file'])}"],
        ["Download volume (GB)", num(f["bytes_downloaded"] / 1e9, 1)],
        ["---"],
        ["Article rows read", _int(fu["rows"])],
        ["Web articles", _int(fu["web"])],
        ["With a US location", _int(fu["us"])],
        ["With a URL headline of 4+ words", _int(fu["headline"])],
        ["With an economy or market term", _int(fu["keyword"])],
        ["Not from a press-release domain", _int(fu["kept"])],
        ["After URL de-duplication", _int(d["after_url_dedup"])],
        ["After same-day headline de-duplication", _int(d["after_headline_dedup"])],
        ["---"],
        ["Index months", _int(h["months"])],
        ["Headlines per index month: min / median / max", f"{_int(h['min'])} / {_count(h['median'])} / {_int(h['max'])}"],
        ["Valid days per index month: min / median / max",
         " / ".join(_count(e["valid_days_per_index_month"][k]) for k in ("min", "median", "max"))],
        ["Index months failing the coverage rule", _int(len(e["index_months_failing_coverage"]))],
        ["Share of headlines with a page title (\\%)", pct(t["share_with_title_overall"], 1)],
        ["First index month with titles on $\\geq$95\\% of headlines", str(t["first_month_at_threshold"])],
        ["Distinct source domains", _int(e["domains"]["distinct"])],
        ["Share of the ten largest domains (\\%)", pct(e["domains"]["top10_share"], 1)],
    ]
    _write("tab1_corpus.tex", _tabular("lr", ["Item", "Value"], rows))


def table_measurement() -> None:
    e = _load("E02_measurement")
    if not e:
        return
    rows = []
    for k in ["finbert", "clf", "lm", "vader", "gdelt_tone", "composite"]:
        sh = e["c_split_half"]["by_index"].get(k, {})
        ut_h = e["b_url_headline_vs_title"]["headline_level"].get(k, {})
        ut_m = e["b_url_headline_vs_title"]["month_level"].get(k, {})
        ex = e["d_external"][k]
        st = e["score_stats"].get(k, {})
        rows.append([LABEL[k],
                     pct(st.get("share_zero"), 1) if st else "n/a",
                     num(sh.get("spearman_brown")),
                     num(ut_h.get("pearson")), num(ut_m.get("pearson")),
                     num(ex["level~sffed_nsi"].get("pearson")), num(ex["level~ics"].get("pearson")),
                     num(ex["level~ret_same_month"].get("pearson")), num(ex["level~ret_next_month"].get("pearson"))])
    header = ["Index", "Zero (\\%)", "Split-half", "Headline", "Month", "SF Fed", "ICS", "Same month", "Next month"]
    groups = _groups((4, 5, "URL vs title"), (6, 7, "Published series"), (8, 9, "S\\&P 500 return"))
    _write("tab2_measurement.tex", _tabular("lrrrrrrrr", header, rows, groups))
    lab = e["e_labelled_data"]
    rows = [["Classifier (5-fold)", pct(lab["clf_5fold"]["accuracy_mean"], 1), pct(lab["clf_5fold"]["macro_f1_mean"], 1)],
            ["LM lexicon", pct(lab["lm"]["accuracy"], 1), pct(lab["lm"]["macro_f1"], 1)],
            ["VADER", pct(lab["vader"]["accuracy"], 1), pct(lab["vader"]["macro_f1"], 1)],
            ["FinBERT (in sample)", pct(lab["finbert"]["accuracy"], 1), pct(lab["finbert"]["macro_f1"], 1)],
            ["Majority class", pct(lab["majority_class_share"], 1), "n/a"]]
    _write("tab2b_labelled.tex", _tabular("lrr", ["Scorer", "Accuracy (\\%)", "Macro-F1 (\\%)"], rows))


def _adj(block: dict | None, key: str = "p_bonferroni") -> str:
    """Bonferroni-adjusted p-value where the plan asks for one (the four individual scorers), else blank."""
    return pval(block[key]) if block and block.get(key) is not None else ""


def _continuous_rows(res: dict, order: list[str]) -> list[list[str]]:
    rows = []
    for k in order:
        if k not in res:
            continue
        m2, m3 = res[k].get("M2_vs_M0"), res[k].get("M3_vs_M1")
        rows.append([LABEL[k], *r2_cell(m2), _adj(m2), *r2_cell(m3), _adj(m3)])
    return rows


def table_continuous(exp: str, section: str, out: str) -> None:
    e, pub = _load(exp), _load("E06_published")
    if not e:
        return
    header = ["Predictor", "$R^2_{OS}$ vs M0 (\\%)", "90\\% interval", "$p$", "$p$ adj.",
              "$R^2_{OS}$ vs M1 (\\%)", "90\\% interval", "$p$", "$p$ adj."]
    rows = _continuous_rows(e["results"], TEXT_ORDER)
    if pub:
        for title, block in _published_blocks(pub):
            rows += [["---"], [f"\\multicolumn{{9}}{{l}}{{{title}}}"]]
            rows += _continuous_rows(pub["results"][block][section], PUB_ORDER)
    _write(out, _tabular("lrcrrrcrr", header, rows))


def _published_blocks(pub: dict) -> list[tuple[str, str]]:
    """Titles of the two E06 blocks. Their models and benchmarks are estimated on the long sample."""
    since = pub["sample"]["first_origin"][:4]
    return [(f"Published indices and their benchmarks, estimated from {since}: same "
             f"{pub['text_window']['n_oos']} test months", "text_window"),
            (f"Published indices and their benchmarks, estimated from {since}: long sample, "
             f"{pub['sample']['n_oos']} test months", "long")]


def _direction_baselines(base: dict) -> list[list[str]]:
    """L0 and L1 of one block. The Brier and DM cells of the rows below refer to this L0."""
    rows = []
    for k, label in (("L0", "Base rate (L0)"), ("L1", "Lagged return (L1)")):
        b = base[k]
        rows.append([label, pct(b["hit_rate"], 1), f"[{pct(b['hit_ci_lo'], 1)}, {pct(b['hit_ci_hi'], 1)}]",
                     pct(b["share_up_predicted"], 1), pval(b["pt_p"]), "", num(b["brier"], 4),
                     "n/a" if k == "L0" else pval(b["dm_brier_p"])])
    return rows


def _direction_rows(res: dict, order: list[str], adjusted: tuple[str, ...] = ()) -> list[list[str]]:
    """L2 of each predictor. `adjusted` names the scorers that carry a Bonferroni-adjusted p-value."""
    rows = []
    for k in order:
        if k not in res:
            continue
        b = res[k]["L2"]
        if "hit_rate" not in b:
            continue
        adj = _adj(b, "pt_p_bonferroni") or ("n/a" if k in adjusted and b.get("pt_p") is None else "")
        rows.append([LABEL[k], pct(b["hit_rate"], 1), f"[{pct(b['hit_ci_lo'], 1)}, {pct(b['hit_ci_hi'], 1)}]",
                     pct(b["share_up_predicted"], 1), pval(b["pt_p"]), adj,
                     num(b["brier"], 4), pval(b["dm_brier_p"])])
    return rows


def table_direction() -> None:
    e, pub = _load("E04_sp500_direction"), _load("E06_published")
    if not e:
        return
    header = ["Predictor", "Hit rate (\\%)", "90\\% interval", "Predicted up (\\%)", "PT $p$", "PT $p$ adj.",
              "Brier", "DM $p$ vs L0"]
    rows = _direction_baselines(e["results"]["baselines"]) + [["---"]]
    rows += _direction_rows(e["results"], TEXT_ORDER, tuple(e["config"]["index"]["composite_members"]))
    if pub:
        for title, block in _published_blocks(pub):
            res = pub["results"][block]["sp500_direction"]
            rows += [["---"], [f"\\multicolumn{{8}}{{l}}{{{title}}}"]]
            rows += _direction_baselines(res["baselines"]) + _direction_rows(res, PUB_ORDER)
    _write("tab4_sp500_direction.tex", _tabular("lrcrrrrr", header, rows))


def _n(*blocks: dict) -> str:
    """Number of test months behind the cells of one row; more than one value if they differ."""
    return " / ".join(str(k) for k in sorted({b["n_oos"] for b in blocks}))


_WIDE_GROUPS = _groups((3, 5, "S\\&P 500"), (6, 9, "ICS change"))     # the robustness and LLM tables share columns
_WIDE_HEADER = ["$n$", "$R^2_{OS}$ vs M0", "$p$", "Hit rate / base", "$R^2_{OS}$ vs M0", "$p$", "$R^2_{OS}$ vs M1", "$p$"]


def table_robustness() -> None:
    header = ["Variant", *_WIDE_HEADER]
    rows: list[list[str]] = []

    def trio(label: str, block: dict) -> None:
        if "sp500_return" not in block:
            rows.append([label] + ["not run"] * 8)
            return
        sp, d = block["sp500_return"]["composite"]["M2_vs_M0"], block["sp500_direction"]["composite"]["L2"]
        i0, i1 = block["ics_change"]["composite"]["M2_vs_M0"], block["ics_change"]["composite"]["M3_vs_M1"]
        a, c0, c1 = r2_cell(sp), r2_cell(i0), r2_cell(i1)
        hit = f"{pct(d['hit_rate'], 1)} / {pct(d['hit_rate_base'], 1)}" if "hit_rate" in d else "n/a"
        rows.append([label, _n(sp, d, i0, i1), a[0], a[2], hit, c0[0], c0[2], c1[0], c1[2]])

    for exp, labels in (("E08_estimation", {"initial_36": "Initial window 36", "initial_60": "Initial window 60",
                                             "rolling_48": "Rolling window 48"}),
                        ("E09_index_definition", {"a_first_difference": "Index in first differences",
                                                  "b_headline_weighted": "Headline-weighted index",
                                                  "c_title_index": "Page-title index (title months)",
                                                  "d_url_index_same_months": "URL-headline index (title months)"})):
        e = _load(exp)
        if e:
            for k, lab in labels.items():
                trio(lab, e["results"][k])
            rows.append(["---"])
    e = _load("E10_target_definition")
    if e:
        r = e["results"]
        ra = r["a_raw_log_return"]["results"]["composite"]["M2_vs_M0"]
        rb = r["b_simple_excess_return"]["results"]["composite"]["M2_vs_M0"]
        i0 = r["c_ics_own_lag_release_lag_1"]["results"]["composite"]["M2_vs_M0"]
        i1 = r["c_ics_own_lag_release_lag_1"]["results"]["composite"]["M3_vs_M1"]
        a, b, c0, c1 = r2_cell(ra), r2_cell(rb), r2_cell(i0), r2_cell(i1)
        rows += [["Raw log return", _n(ra), a[0], a[2], "", "", "", "", ""],
                 ["Simple excess return", _n(rb), b[0], b[2], "", "", "", "", ""],
                 ["ICS own lag, release lag 1", _n(i0, i1), "", "", "", c0[0], c0[2], c1[0], c1[2]], ["---"]]
    e = _load("E11_horizon3")
    if e:
        r = e["results"]
        sp = r["sp500_return_3m"]["results"]["composite"]["M2_vs_M0"]
        i0 = r["ics_change_3m"]["results"]["composite"]["M2_vs_M0"]
        i1 = r["ics_change_3m"]["results"]["composite"]["M3_vs_M1"]
        a, c0, c1 = r2_cell(sp), r2_cell(i0), r2_cell(i1)
        rows += [["Three-month horizon", _n(sp, i0, i1), a[0], a[2], "", c0[0], c0[2], c1[0], c1[2]], ["---"]]
    while rows and rows[-1] == ["---"]:
        rows.pop()
    if rows:
        _write("tab6_robustness.tex", _tabular("lrrrrrrrr", header, rows, _WIDE_GROUPS))


def table_incremental() -> None:
    """E07: the composite added to a benchmark that already holds other predictors."""
    e = _load("E07_incremental")
    if not e:
        return
    name = {"sffed_nsi": "SF Fed index", "epu_for_sp500": "EPU index", "epu_for_ics": "EPU index",
            "ics_level_for_sp500": "ICS level", "lag_ics_change": "own lag",
            "lag_ret_excess_for_ics": "lagged S\\&P 500 return"}
    rows = []
    for target, label in (("sp500_return", "S\\&P 500 excess return"), ("ics_change", "ICS change")):
        for block in e["results"][target].values():
            controls = ", ".join(name[c] for c in block["controls"])
            rows.append([label, controls, *r2_cell(block)])
        rows.append(["---"])
    while rows and rows[-1] == ["---"]:
        rows.pop()
    header = ["Target", "Benchmark holds", "$R^2_{OS}$ vs benchmark (\\%)", "90\\% interval", "$p$"]
    _write("tab6b_incremental.tex", _tabular("llrcr", header, rows))


def table_llm() -> None:
    """E12: the index of each LLM in the models of E03 to E05, on all test months and split at its training cutoff."""
    name = "tab7_llm.tex"
    e = _load("E12_llm")
    if not e or e.get("status") != "ran":          # a skipped E12 leaves no table behind
        (TABLES / name).unlink(missing_ok=True)
        if not DRY_RUN:
            (PAPER / "tables" / name).unlink(missing_ok=True)
        return
    header = ["Model and test months", *_WIDE_HEADER]
    rows: list[list[str]] = []

    def row(label: str, res: dict) -> None:
        a, d = res["sp500_return"]["llm"]["M2_vs_M0"], res["sp500_direction"]["llm"]["L2"]
        c0, c1 = res["ics_change"]["llm"]["M2_vs_M0"], res["ics_change"]["llm"]["M3_vs_M1"]
        hit = f"{pct(d['hit_rate'], 1)} / {pct(d['hit_rate_base'], 1)}" if "hit_rate" in d else "n/a"
        ra, r0, r1 = r2_cell(a), r2_cell(c0), r2_cell(c1)
        rows.append([label, _n(a, d, c0, c1), ra[0], ra[2], hit, r0[0], r0[2], r1[0], r1[2]])

    for blk in (e, e.get("second_model")):
        if not blk:
            continue
        cut = blk["by_training_cutoff"]
        row(f"{blk['usage']['model']}, all test months", blk["results"])
        row(f"\\quad after the cutoff month ({cut['training_cutoff']})", cut["post_cutoff"]["results"])
        row("\\quad other test months", cut["pre_cutoff"]["results"])
        rows.append(["---"])
    rows.pop()
    _write(name, _tabular("lrrrrrrrr", header, rows, _WIDE_GROUPS))


# --- figures ----------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "grid.linestyle": "-", "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
    })


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", metadata=PDF_META)
    fig.savefig(FIGURES / f"{name}.png", dpi=200, metadata={"Software": None})
    plt.close(fig)


def figure_indices() -> None:
    path = RESULTS / "E02_indices_monthly.csv"
    if not path.exists():
        return
    d = pd.read_csv(path, index_col=0)
    x = pd.PeriodIndex(d.index, freq="M").to_timestamp()
    keys = ["finbert", "clf", "lm", "vader", "gdelt_tone", "composite"]
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 5.0), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), keys):
        z = (d[k] - d[k].mean()) / d[k].std(ddof=0)
        ax.plot(x, d["sffed_nsi_z"], color=MUTED, lw=1.0, ls=(0, (4, 2)), label="SF Fed index")
        ax.plot(x, z, color=BLUE, lw=1.4, label="Headline index")
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.set_title(LABEL[k], loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("z-score")
    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, handlelength=2.4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    _save(fig, "fig1_indices")


def figure_cssed() -> None:
    specs = [("E03_forecasts", "S&P 500 excess return"), ("E05_forecasts", "ICS change")]
    if not all((RESULTS / f"{n}.csv").exists() for n, _ in specs):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, (name, title) in zip(axes, specs):
        d = pd.read_csv(RESULTS / f"{name}.csv", index_col=0).dropna(subset=["M0", "M1", "M2_composite", "M3_composite"])
        x = pd.PeriodIndex(d["target_month"], freq="M").to_timestamp()
        d2 = np.cumsum(np.square(d["y"] - d["M0"]) - np.square(d["y"] - d["M2_composite"]))
        d3 = np.cumsum(np.square(d["y"] - d["M1"]) - np.square(d["y"] - d["M3_composite"]))
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(x, d2, color=BLUE, lw=1.5, label="Index vs historical mean (M2 - M0)")
        ax.plot(x, d3, color=ORANGE, lw=1.5, ls=(0, (5, 2)), label="Own lag + index vs own lag (M3 - M1)")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Cumulative squared-error difference")
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, handlelength=2.6, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig2_cssed")


def figure_r2() -> None:
    e3, e5, pub = _load("E03_sp500_return"), _load("E05_ics_change"), _load("E06_published")
    if not (e3 and e5):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    panels = [(e3["results"], "sp500_return", "S&P 500 excess return"), (e5["results"], "ics_change", "ICS change")]
    keys = TEXT_ORDER + (PUB_ORDER if pub else [])     # one row order for both panels
    ypos = {k: len(keys) - 1 - i for i, k in enumerate(keys)}
    for ax, (res, section, title) in zip(axes, panels):
        entries = [(k, res[k]) for k in TEXT_ORDER if k in res]
        if pub:
            entries += [(k, pub["results"]["text_window"][section][k]) for k in PUB_ORDER
                        if k in pub["results"]["text_window"][section]]
        for k, block in entries:
            y = ypos[k]
            for off, comp, color, marker, fill in ((0.16, "M2_vs_M0", BLUE, "o", BLUE), (-0.16, "M3_vs_M1", ORANGE, "s", "white")):
                b = block.get(comp, {})
                if "r2_os" not in b:
                    continue
                ax.plot([100 * b["r2_ci_lo"], 100 * b["r2_ci_hi"]], [y + off, y + off], color=color, lw=1.2)
                ax.plot(100 * b["r2_os"], y + off, marker=marker, ms=4.5, mfc=fill, mec=color, mew=1.2, color=color, ls="none")
        ax.axvline(0, color=INK, lw=0.7)
        ax.set_yticks([ypos[k] for k in keys])
        ax.set_yticklabels([LABEL[k] for k in keys])
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.set_xlabel("Out-of-sample R-squared (%), 90% interval")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.06)
    handles = [Line2D([0], [0], color=BLUE, marker="o", ms=4.5, mfc=BLUE, lw=1.2, label="Index vs historical mean (M2 vs M0)"),
               Line2D([0], [0], color=ORANGE, marker="s", ms=4.5, mfc="white", mew=1.2, lw=1.2,
                      label="Own lag + index vs own lag (M3 vs M1)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig3_r2")


def _months(col: pd.Series) -> pd.DatetimeIndex:
    return pd.PeriodIndex(col, freq="M").to_timestamp()


def _years(ax, step: int = 1) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def figure_coverage() -> None:
    path, e = RESULTS / "E01_monthly_counts.csv", _load("E01_corpus")
    if not (path.exists() and e):
        return
    ix = load_config()["index"]
    d = pd.read_csv(path)
    x = _months(d["index_month"])
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 4.2), sharex=True, height_ratios=[1.4, 1, 1])
    axes[0].bar(x, d["n_headlines"], width=24, color=BLUE, lw=0)
    axes[0].set_ylabel("Headlines")
    axes[0].set_title("Headlines per index month", loc="left")
    axes[1].plot(x, d["n_valid_days"], color=BLUE, lw=1.3)
    axes[1].axhline(ix["min_days_per_month"], color=ORANGE, lw=1.0, ls=(0, (4, 2)))
    axes[1].text(x[0], ix["min_days_per_month"] + 0.8, f"required: {ix['min_days_per_month']} days", fontsize=6.5,
                 color=INK, va="bottom")
    axes[1].set_ylim(bottom=min(10, ix["min_days_per_month"] - 5))
    axes[1].set_ylabel("Days")
    axes[1].set_title(f"Valid days per index month (at least {ix['min_headlines_per_day']} headlines)", loc="left")
    axes[2].plot(x, 100 * d["share_with_title"], color=BLUE, lw=1.3)
    axes[2].axhline(100 * ix["title_month_min_share"], color=ORANGE, lw=1.0, ls=(0, (4, 2)))
    axes[2].text(x[0], 100 * ix["title_month_min_share"] - 4, f"title months: {round(100 * ix['title_month_min_share'])}% or more",
                 fontsize=6.5, color=INK, va="top")
    axes[2].set_ylabel("Percent")
    axes[2].set_title("Share of headlines with a page title", loc="left")
    _years(axes[2])
    fig.tight_layout(h_pad=0.6)
    _save(fig, "fig4_coverage")


def figure_timing() -> None:
    e3, e6 = _load("E03_sp500_return"), _load("E06_published")
    if not (e3 and e6):
        return
    cfg = load_config()
    days = int(cfg["publication_lag_days"]["sffed_nsi"])
    shift = days / 30.0                                     # drawn to scale for a month of 30 days
    tau = {-2: 0.0, -1: 1.0, 0: 2.0, 1: 3.0}                # x position of tau(m + k)

    def month(lag: int) -> tuple[float, float]:             # reference month m - lag
        return tau[-lag - 1], tau[-lag]

    def name(lag: int) -> str:
        return "month $m$" if lag == 0 else f"month $m-{lag}$"

    epu, ics = release_lag("epu", "sp500", cfg), release_lag("ics", "sp500", cfg)
    rows = [("Headlines (text window)", (tau[-1], tau[0]), BLUE, "dated before $\\tau(m)$"),
            ("SF Fed index", (tau[-1] - shift, tau[0] - shift), PALE_BLUE, f"ends {days} days before $\\tau(m)$"),
            ("EPU index", month(epu), PALE_BLUE, name(epu)),
            ("ICS level, for the S&P 500", month(ics), PALE_BLUE, name(ics)),
            ("Own lag of the target", month(0), PALE_BLUE, name(0)),
            ("Target", (tau[0], tau[1]), ORANGE, "month $m+1$")]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.3, 3.9), height_ratios=[2.5, 1])
    for i, (label, (lo, hi), color, note) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, hi - lo, left=lo, height=0.56, color=color, edgecolor=MUTED if color == PALE_BLUE else color, lw=0.5)
        ax.text((lo + hi) / 2, y, note, ha="center", va="center", fontsize=6.5,
                color="white" if color in (BLUE, ORANGE) else INK)
    ax.axvline(tau[0], color=INK, lw=1.0)
    ax.text(tau[0] + 0.03, len(rows) - 0.45, "forecast date", ha="left", va="bottom", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xticks(list(tau.values()))
    ax.set_xticklabels(["$\\tau(m-2)$", "$\\tau(m-1)$", "$\\tau(m)$", "$\\tau(m+1)$"])
    ax.set_xlim(tau[-2] - 0.05, tau[1] + 0.05)
    ax.set_ylim(-0.55, len(rows) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("(a) Information used by the forecast made at $\\tau(m)$", loc="left")

    def spans(s: dict) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        first = (pd.Period(s["first_origin"], "M") + 1).to_timestamp()
        split = pd.Period(s["first_oos_target"], "M").to_timestamp()
        end = (pd.Period(s["last_oos_target"], "M") + 1).to_timestamp()
        return first, split, end

    for y, (label, s) in enumerate((("Text sample", e3["sample"]), ("Long sample", e6["sample"]))):
        first, split, end = spans(s)
        bx.barh(y, (split - first).days, left=mdates.date2num(first), height=0.5, color=MUTED, lw=0)
        bx.barh(y, (end - split).days, left=mdates.date2num(split), height=0.5, color=BLUE, lw=0)
        text = f"{s['initial_window']} months for the first estimate, {s['n_oos']} test months"
        if y == 0:
            bx.text(mdates.date2num(first) - 250, y, text, ha="right", va="center", fontsize=6.5)
        else:
            bx.text(mdates.date2num(first) + 250, y + 0.48, text, ha="left", va="center", fontsize=6.5)
    bx.set_yticks([0, 1])
    bx.set_yticklabels(["Text sample", "Long sample"])
    bx.set_ylim(-0.45, 1.75)
    bx.xaxis_date()
    bx.xaxis.set_major_locator(mdates.YearLocator(5))
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    bx.grid(axis="y", visible=False)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Target month")
    bx.set_title("(b) Estimation and test months", loc="left")
    bx.legend(handles=[Patch(color=MUTED, label="First estimation window"), Patch(color=BLUE, label="Test months")],
              loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.32), handlelength=1.2)
    fig.tight_layout(h_pad=1.0)
    _save(fig, "fig5_timing")


def figure_agreement() -> None:
    e = _load("E02_measurement")
    if not e:
        return
    agree = e["a_scorer_agreement"]
    cmap = LinearSegmentedColormap.from_list("agreement", ["#ffffff", BLUE])
    rows, cols = INDEX_ORDER[1:], INDEX_ORDER[:-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75), sharey=True)
    for ax, (level, title) in zip(axes, (("headline_level", "Per headline"), ("month_level", "Per index month"))):
        m = np.full((len(rows), len(cols)), np.nan)
        for i, a in enumerate(rows):
            for j, b in enumerate(cols):
                if INDEX_ORDER.index(b) < INDEX_ORDER.index(a):
                    pair = agree[level].get(f"{b}~{a}") or agree[level][f"{a}~{b}"]
                    m[i, j] = pair["pearson"]
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if m[i, j] > 0.55 else INK)
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        n = next(iter(agree[level].values()))["n"]
        ax.set_title(f"{title} (n = {n:,})", loc="left")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([LABEL[k] for k in cols])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([LABEL[k] for k in rows])
        ax.grid(visible=False)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(w_pad=1.0)
    bar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("Pearson correlation")
    bar.outline.set_visible(False)
    _save(fig, "fig6_agreement")


def figure_probability() -> None:
    path = RESULTS / "E04_forecasts.csv"
    if not path.exists():
        return
    d = pd.read_csv(path).dropna(subset=["L0", "L2_composite"])
    x = _months(d["target_month"])
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    for k in INDEX_ORDER:
        ax.plot(x, d[f"L2_{k}"], color=MUTED, lw=0.7, alpha=0.7)
    ax.plot(x, d["L0"], color=INK, lw=1.1, ls=(0, (4, 2)))
    ax.plot(x, d["L2_composite"], color=BLUE, lw=1.6)
    ax.axhline(0.5, color=ORANGE, lw=1.1)
    floor = min(0.5, float(d[[f"L2_{k}" for k in INDEX_ORDER]].min().min())) - 0.08
    fell = d["u"] == 0
    ax.plot(x[fell.values], np.full(int(fell.sum()), floor + 0.025), ls="none", marker="|", ms=6, mew=1.2, color=INK)
    ax.set_ylim(floor, None)
    ax.set_ylabel("Predicted probability of a rise")
    _years(ax)
    handles = [Line2D([0], [0], color=BLUE, lw=1.6, label="Composite index (L2)"),
               Line2D([0], [0], color=MUTED, lw=0.7, label="Individual indices (L2)"),
               Line2D([0], [0], color=INK, lw=1.1, ls=(0, (4, 2)), label="Base rate (L0)"),
               Line2D([0], [0], color=ORANGE, lw=1.1, label="Threshold for predicting a rise"),
               Line2D([0], [0], color=INK, ls="none", marker="|", ms=6, mew=1.2, label="Month in which the index fell")]
    fig.legend(handles=handles, loc="lower center", ncol=3, handlelength=2.2, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    _save(fig, "fig7_probability")


# --- summary ----------------------------------------------------------------------------

def _r2_text(b: dict | None) -> str:
    if not b or "r2_os" not in b:
        return "n/a"
    return (f"{pct(b['r2_os'])}% [90% interval {pct(b['r2_ci_lo'])}, {pct(b['r2_ci_hi'])}], "
            f"p = {pval(b.get('p'))}, n = {b['n_oos']}")


def _hit_text(b: dict) -> str:
    if "hit_rate" not in b:
        return "n/a"
    return (f"hit rate {pct(b['hit_rate'], 1)}% [90% interval {pct(b['hit_ci_lo'], 1)}, {pct(b['hit_ci_hi'], 1)}] "
            f"against base rate {pct(b['hit_rate_base'], 1)}%, PT p = {pval(b['pt_p'])}, n = {b['n_oos']}")


def summary() -> None:
    rows = []

    def add(exp: str, tested: str, headline: str, source: str, commit: str) -> None:
        rows.append(f"| {exp} | {tested} | {headline} | `{source}` | `{commit[:7]}{'-dirty' if commit.endswith('-dirty') else ''}` |")

    e = _load("E01_corpus")
    if e:
        h = e["headlines_per_index_month"]
        add("E01", "Corpus: seeded sample of GDELT GKG files, filters, coverage",
            f"{e['files']['downloaded']:,} of {e['files']['requested']:,} files; {e['dedup']['after_headline_dedup']:,} "
            f"headlines after filters and de-duplication; {h['min']:,} / {_count_text(h['median'])} / {h['max']:,} per index month "
            f"(min / median / max); {len(e['index_months_failing_coverage'])} index months fail the coverage rule",
            "results/E01_corpus.json", e["git_commit"])
    e = _load("E02_measurement")
    if e:
        sh = e["c_split_half"]["by_index"]
        ut = e["b_url_headline_vs_title"]["month_level"]
        ex = e["d_external"]["composite"]["level~sffed_nsi"]
        add("E02", "Measurement: scorer agreement, URL headline vs page title, split-half reliability, relation to published series",
            f"split-half reliability (Spearman-Brown) FinBERT {num(sh['finbert'].get('spearman_brown'))}, classifier "
            f"{num(sh['clf'].get('spearman_brown'))}, LM {num(sh['lm'].get('spearman_brown'))}, VADER "
            f"{num(sh['vader'].get('spearman_brown'))} (n = {sh['finbert'].get('n')} months); monthly URL-vs-title "
            f"correlation FinBERT {num(ut['finbert'].get('pearson'))} (n = {ut['finbert'].get('n')}); composite vs SF Fed "
            f"index r = {num(ex.get('pearson'))} (n = {ex.get('n')})",
            "results/E02_measurement.json", e["git_commit"])
    for exp, name, label in (("E03", "E03_sp500_return", "S&P 500 excess return, h = 1"),
                             ("E05", "E05_ics_change", "ICS change, h = 1")):
        e = _load(name)
        if e:
            c = e["results"]["composite"]
            add(exp, f"{label}: composite index against historical mean (M2 vs M0) and own-lag model (M3 vs M1)",
                f"R2_OS vs M0 {_r2_text(c['M2_vs_M0'])}; R2_OS vs M1 {_r2_text(c['M3_vs_M1'])}; outcome: "
                f"**{e['decision']['outcome']}**", f"results/{name}.json", e["git_commit"])
        if exp == "E03":
            e4 = _load("E04_sp500_direction")
            if e4:
                add("E04", "S&P 500 direction, h = 1: composite index in a logistic model (L2) against the base rate (L0)",
                    f"{_hit_text(e4['results']['composite']['L2'])}; outcome: **{e4['decision']['outcome']}**",
                    "results/E04_sp500_direction.json", e4["git_commit"])
    e = _load("E06_published")
    if e:
        lg, tw = e["results"]["long"], e["results"]["text_window"]
        add("E06", "Published indices as predictors, long sample and the text-sample window",
            f"SF Fed index, long sample: S&P R2_OS vs M0 {_r2_text(lg['sp500_return']['sffed_nsi']['M2_vs_M0'])}; ICS R2_OS vs M1 "
            f"{_r2_text(lg['ics_change']['sffed_nsi']['M3_vs_M1'])}. Text window: S&P {_r2_text(tw['sp500_return']['sffed_nsi']['M2_vs_M0'])}; "
            f"ICS vs M1 {_r2_text(tw['ics_change']['sffed_nsi']['M3_vs_M1'])}",
            "results/E06_published.json", e["git_commit"])
    e = _load("E07_incremental")
    if e:
        r = e["results"]
        add("E07", "Composite added to models that already hold published indices or the lagged S&P 500 return",
            f"S&P, added to SF Fed index: {_r2_text(r['sp500_return']['a_sffed'])}; ICS, added to own lag + SF Fed index: "
            f"{_r2_text(r['ics_change']['a_sffed'])}; ICS, added to own lag + lagged S&P return: "
            f"{_r2_text(r['ics_change']['c_lagged_sp500_return'])}", "results/E07_incremental.json", e["git_commit"])

    def trio_text(block: dict) -> str:
        if "decisions" not in block:
            return "not run"
        d = block["decisions"]
        return (f"S&P R2_OS vs M0 {pct(block['sp500_return']['composite']['M2_vs_M0'].get('r2_os'))}% ({d['sp500_return']['outcome']}), "
                f"direction {d['sp500_direction']['outcome']}, ICS R2_OS vs M1 "
                f"{pct(block['ics_change']['composite']['M3_vs_M1'].get('r2_os'))}% ({d['ics_change']['outcome']})")

    e = _load("E08_estimation")
    if e:
        add("E08", "Robustness to the estimation scheme (composite)",
            "; ".join(f"{k}: {trio_text(v)}" for k, v in e["results"].items()), "results/E08_estimation.json", e["git_commit"])
    e = _load("E09_index_definition")
    if e:
        add("E09", "Robustness to the index definition (composite)",
            "; ".join(f"{k}: {trio_text(v)}" for k, v in e["results"].items()), "results/E09_index_definition.json", e["git_commit"])
    e = _load("E10_target_definition")
    if e:
        r = e["results"]
        add("E10", "Robustness to the target definition (composite)",
            f"raw log return: R2_OS vs M0 {_r2_text(r['a_raw_log_return']['results']['composite']['M2_vs_M0'])}; simple excess "
            f"return: {_r2_text(r['b_simple_excess_return']['results']['composite']['M2_vs_M0'])}; ICS with own lag at release "
            f"lag 1: R2_OS vs M1 {_r2_text(r['c_ics_own_lag_release_lag_1']['results']['composite']['M3_vs_M1'])}",
            "results/E10_target_definition.json", e["git_commit"])
    e = _load("E11_horizon3")
    if e:
        r = e["results"]
        add("E11", "Three-month horizon (composite)",
            f"S&P R2_OS vs M0 {_r2_text(r['sp500_return_3m']['results']['composite']['M2_vs_M0'])} "
            f"({r['sp500_return_3m']['decision']['outcome']}); ICS R2_OS vs M1 "
            f"{_r2_text(r['ics_change_3m']['results']['composite']['M3_vs_M1'])} ({r['ics_change_3m']['decision']['outcome']})",
            "results/E11_horizon3.json", e["git_commit"])
    e = _load("E12_llm")
    if e:
        if e["status"] == "skipped":
            add("E12", "LLM scorer", f"skipped: {e['reason']}", "results/E12_llm.json", e["git_commit"])
        else:
            for label, blk in (("LLM scorer", e), ("LLM scorer, second model", e.get("second_model"))):
                if not blk:
                    continue
                r, c = blk["results"], blk["by_training_cutoff"]
                p = c["post_cutoff"]["results"]
                add("E12", f"{label} ({blk['usage']['model']}, every kept headline)",
                    f"S&P R2_OS vs M0 {_r2_text(r['sp500_return']['llm']['M2_vs_M0'])}; ICS R2_OS vs M1 "
                    f"{_r2_text(r['ics_change']['llm']['M3_vs_M1'])}; test months whose whole text window lies "
                    f"after the training cutoff month ({c['training_cutoff']}; {c['post_cutoff']['n_test_origins']}): "
                    f"S&P R2_OS vs M0 {_r2_text(p['sp500_return']['llm']['M2_vs_M0'])}, ICS R2_OS vs M1 "
                    f"{_r2_text(p['ics_change']['llm']['M3_vs_M1'])}; labelled {blk['usage']['labelled']:,} of "
                    f"{blk['usage']['texts']:,} headlines; cost computed from token counts USD "
                    f"{blk['usage']['cost_usd']:.2f}",
                    "results/E12_llm.json", e["git_commit"])
    e = _load("power")
    if e:
        def power_text(r: dict) -> str:
            s, sm = r["selected"], r["smallest_r2"]
            return (f"power {pct(r['at_zero']['power'], 1)}% at a true R2 of 0, {pct(s['r2_1pct']['power'], 1)}% at 1%, "
                    f"{pct(s['r2_5pct']['power'], 1)}% at 5%; 80% power from a true R2 of {pct(sm['power_80'], 1)}%")

        r = e["results"]
        add("Power", f"Not pre-specified: simulated power of the M2 vs M0 comparison for the composite, "
            f"{e['config']['reps']:,} samples per value of the true R2",
            f"S&P 500 return: {power_text(r['sp500_return'])}; ICS change: {power_text(r['ics_change'])}",
            "results/power.json", e["git_commit"])
    failed = sorted(p.name for p in RESULTS.glob("E*_FAILED.json"))
    text = ["# Results summary", "",
            "Generated by `python -m sentcast.report` from the files in `results/`. Do not edit by hand.", "",
            "R2_OS is the out-of-sample R-squared in percent. M0 is the historical mean, M1 the own-lag model. "
            "p-values for R2_OS are one-sided Clark-West. Intervals are 90% stationary-bootstrap intervals. "
            "Outcomes follow the rule fixed in PLAN.md section 1.", "",
            "| ID | What was tested | Headline number with uncertainty | Source | Commit |", "|---|---|---|---|---|", *rows, ""]
    if failed:
        text += ["Failed runs on record: " + ", ".join(f"`results/{f}`" for f in failed), ""]
    (RESULTS / "SUMMARY.md").write_text("\n".join(text), encoding="utf-8", newline="\n")


def mirror_to_paper() -> None:
    if DRY_RUN:
        return
    for src, dst in ((TABLES, PAPER / "tables"), (FIGURES, PAPER / "figures")):
        if not src.exists() or not PAPER.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix in (".tex", ".pdf"):
                shutil.copyfile(f, dst / f.name)


def main() -> None:
    _style()
    table_corpus()
    table_measurement()
    table_continuous("E03_sp500_return", "sp500_return", "tab3_sp500_return.tex")
    table_direction()
    table_continuous("E05_ics_change", "ics_change", "tab5_ics_change.tex")
    table_robustness()
    table_incremental()
    table_llm()
    figure_indices()
    figure_cssed()
    figure_r2()
    figure_coverage()
    figure_timing()
    figure_agreement()
    figure_probability()
    summary()
    texnumbers.write()
    mirror_to_paper()
    print("report written to", RESULTS)


if __name__ == "__main__":
    main()
