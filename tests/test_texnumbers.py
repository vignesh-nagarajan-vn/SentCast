"""Numbers for the paper text: formats, the generated macro file and the check on typed numbers."""
import json
import re

import pytest

from sentcast import texnumbers, verify


def test_formats_print_what_the_text_needs():
    assert texnumbers.fmt(-0.0181, "pct") == "\\ensuremath{-}1.81"
    assert texnumbers.fmt(0.0103, "pct") == "1.03"
    assert texnumbers.fmt(0.6333, "pct1") == "63.3"
    assert texnumbers.fmt(0.0004, "p") == "\\ensuremath{<}0.001"
    assert texnumbers.fmt(0.149, "p") == "0.149"
    assert texnumbers.fmt(125105, "int") == "125,105"
    assert texnumbers.fmt(870.5, "count") == "870.5" and texnumbers.fmt(30.0, "count") == "30"
    assert texnumbers.fmt("2019-04", "month") == "April 2019"


@pytest.fixture()
def results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(texnumbers, "RESULTS", tmp_path)
    monkeypatch.setattr(texnumbers, "TABLES", tmp_path / "tables")
    block = {"r2_os": -0.0181, "r2_ci_lo": -0.0785, "r2_ci_hi": 0.031, "p": 0.45, "n_oos": 90}
    payload = {"results": {"baselines": {"M1_vs_M0": block},
                           "composite": {"M2_vs_M0": block, "M3_vs_M1": {"n_oos": 3, "note": "too few"}}},
               "sample": {"n_oos": 90, "first_oos_target": "2019-04"}, "decision": {"outcome": "null"}}
    (tmp_path / "E05_ics_change.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


def test_keys_exist_only_where_the_result_file_has_a_value(results_dir):
    rows = {r["key"]: r for r in texnumbers.write()}
    assert rows["E05.composite.m0.r2"]["value"] == "\\ensuremath{-}1.81"
    assert rows["E05.composite.m0.r2"]["path"] == "results.composite.M2_vs_M0.r2_os"
    assert rows["E05.first"]["value"] == "April 2019" and rows["E05.outcome"]["value"] == "null"
    assert "E05.composite.m1.n" in rows and "E05.composite.m1.r2" not in rows      # too few forecasts: no number
    assert not any(k.startswith(("E03.", "E12.")) for k in rows)                   # files that are absent
    text = (results_dir / "tables" / texnumbers.NAME).read_text(encoding="utf-8")
    assert "\\@namedef{res@E05.composite.m0.p}{0.450}" in text
    assert "Unknown result key" in text                                            # a wrong key stops LaTeX
    assert not re.search(verify.FORBIDDEN[7], text, flags=re.I)                    # the project is not named


def test_the_check_rejects_typed_decimals_and_unknown_keys(results_dir, monkeypatch):
    texnumbers.write()
    paper = results_dir / "paper"
    paper.mkdir()
    monkeypatch.setattr(verify, "PAPER", paper)
    monkeypatch.setattr(verify, "TABLES", results_dir / "tables")
    good = ("The level is 0.05, Attribution 4.0 (CC BY 4.0) applies and the tool is Opus\n5.5. % a comment with 1.23\n"
            "The index gives \\res{E05.composite.m0.r2}\\% over \\res{E05.n} months, see Table~\\ref{tab:x}.\n")
    (paper / "main.tex").write_text(good, encoding="utf-8")
    report = verify.check_paper_numbers()
    assert report["ok"] and report["keys_used"] == 2 and report["typed_decimals"] == []
    (paper / "main.tex").write_text(good + "It gives 1.03\\% and \\res{E05.nothing}.\n", encoding="utf-8")
    report = verify.check_paper_numbers()
    assert not report["ok"] and report["typed_decimals"] == ["1.03"] and report["unknown_keys"] == ["E05.nothing"]
