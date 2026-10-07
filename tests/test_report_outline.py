"""The parts of the brief and the tables that depend on whether the LLM scorer ran (E12)."""
import json

import numpy as np
import pandas as pd
import pytest

from sentcast import experiments as ex
from sentcast import outline, report

TEMPLATE = ("before\n"
            "<!-- if E12_llm:status == ran -->\n"
            "ran {{E12_llm:usage.labelled:int}}\n"
            "<!-- else -->\n"
            "status {{E12_llm:status:str}}\n"
            "<!-- end -->\n"
            "after\n")


@pytest.fixture()
def results_dir(monkeypatch, tmp_path):
    for mod in (outline, report):
        monkeypatch.setattr(mod, "RESULTS", tmp_path)
    monkeypatch.setattr(report, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(report, "DRY_RUN", True)             # never touch paper/
    outline._file.cache_clear()
    yield tmp_path
    outline._file.cache_clear()


def _write(path, payload: dict) -> None:
    (path / "E12_llm.json").write_text(json.dumps(payload), encoding="utf-8")
    outline._file.cache_clear()


def _llm_payload() -> dict:
    """An E12 result with the structure experiments._llm_block writes, from synthetic data."""
    rng = np.random.default_rng(5)
    n, n0 = 138, 48
    origins = pd.period_range("2015-03", periods=n, freq="M")
    y = rng.normal(size=n)
    tab = pd.DataFrame({"y_ret_excess": y, "lag_ret_excess": np.r_[0.0, y[:-1]], "y_up": (y > 0).astype(float),
                        "y_ics_change": rng.normal(size=n), "lag_ics_change": rng.normal(size=n)}, index=origins)
    idx = pd.DataFrame({"llm": rng.normal(size=n)}, index=origins)

    def trio(subset=None) -> dict:
        r1, _ = ex.eval_continuous(tab, "y_ret_excess", "lag_ret_excess", idx, n0=n0, tag="t|ret", subset=subset)
        r2, _ = ex.eval_direction(tab, idx, n0=n0, tag="t|dir", subset=subset)
        r3, _ = ex.eval_continuous(tab, "y_ics_change", "lag_ics_change", idx, n0=n0, tag="t|ics", subset=subset)
        return {"sp500_return": r1, "sp500_direction": r2, "ics_change": r3}

    def block(model: str, cutoff: str) -> dict:
        post = ex.llm_post_cutoff(origins, cutoff)
        return {"usage": {"model": model, "labelled": 1234, "texts": 1250},
                "results": trio(),
                "by_training_cutoff": {"training_cutoff": cutoff,
                                       "post_cutoff": {"n_test_origins": int(post.sum()), "results": trio(post)},
                                       "pre_cutoff": {"n_test_origins": int((~post).sum()) - n0, "results": trio(~post)}}}

    first = block("model-a", "2025-07")
    return {"status": "ran", **first, "second_model": block("model-b", "2026-06")}


def test_a_conditional_block_follows_the_result_file(results_dir):
    _write(results_dir, {"status": "skipped"})
    text, used = outline.render(TEMPLATE)
    assert text == "before\nstatus skipped\nafter\n" and [u["rendered"] for u in used] == ["skipped"]
    _write(results_dir, {"status": "ran", "usage": {"labelled": 1234}})
    text, used = outline.render(TEMPLATE)
    assert text == "before\nran 1,234\nafter\n" and [u["rendered"] for u in used] == ["1,234"]


def test_a_placeholder_without_a_source_fails_outside_a_dry_run(results_dir, monkeypatch):
    monkeypatch.setattr(outline, "DRY_RUN", False)
    _write(results_dir, {"status": "ran"})                   # the number the kept part needs is absent
    with pytest.raises(KeyError):
        outline.render(TEMPLATE)


def test_negative_numbers_get_a_minus_sign_and_other_hyphens_stay():
    row = ["claude-haiku-4-5 (2025-07)", "-1.49", "[-2.44, 4.80]", "61.1 / 63.3", "2019-10", "n/a"]
    line = report._tabular("lrrrrr", ["a"] * 6, [row]).splitlines()[4]
    assert line == "claude-haiku-4-5 (2025-07) & $-$1.49 & [$-$2.44, 4.80] & 61.1 / 63.3 & 2019-10 & n/a \\\\"


def test_published_blocks_say_where_their_benchmarks_come_from():
    pub = {"sample": {"first_origin": "1980-02", "n_oos": 439}, "text_window": {"n_oos": 90}}
    (short, a), (long, b) = report._published_blocks(pub)
    assert (a, b) == ("text_window", "long")
    assert "estimated from 1980" in short and "same 90 test months" in short
    assert "estimated from 1980" in long and "439 test months" in long


def test_an_undefined_direction_test_reads_na_only_where_an_adjustment_applies():
    block = {"hit_rate": 0.6, "hit_ci_lo": 0.5, "hit_ci_hi": 0.7, "share_up_predicted": 1.0, "pt_p": None,
             "brier": 0.24, "dm_brier_p": 0.5}
    res = {"finbert": {"L2": block}, "gdelt_tone": {"L2": block}}
    rows = report._direction_rows(res, ["finbert", "gdelt_tone"], adjusted=("finbert",))
    assert [r[5] for r in rows] == ["n/a", ""]               # the column of adjusted p-values
    base = {"L0": {**block, "dm_brier_p": None}, "L1": block}
    assert [r[0] for r in report._direction_baselines(base)] == ["Base rate (L0)", "Lagged return (L1)"]


def test_the_llm_table_exists_only_when_e12_ran(results_dir):
    table = results_dir / "tables" / "tab7_llm.tex"
    _write(results_dir, _llm_payload())
    report.table_llm()
    lines = [ln.removesuffix(" \\\\") for ln in table.read_text(encoding="utf-8").splitlines()
             if " & " in ln and "multicolumn" not in ln]
    assert len(lines) == 1 + 6                              # header, then three rows for each model
    assert lines[1].startswith("model-a, all test months & 90 & ")
    assert "(2025-07)" in lines[2] and lines[2].split(" & ")[1] == "12"
    assert lines[3].split(" & ")[1] == "78"
    # one test month lies after the second model's cutoff: too few for a comparison
    assert lines[5].split(" & ")[1:] == ["1"] + ["n/a"] * 7
    _write(results_dir, {"status": "skipped"})
    report.table_llm()
    assert not table.exists()
