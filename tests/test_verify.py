"""The anonymity check of the verification pass, in its named and double-blind modes."""
import pytest

from sentcast import verify

NAME = "Vignesh Nagarajan"


def _paper(tmp_path, tex: str, pdf_text: str = "", meta: dict | None = None, raw: bytes = b"%PDF-1.5",
           monkeypatch=None):
    """A paper/ folder with main.tex and a stand-in main.pdf whose extraction is faked."""
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
    (paper / "main.pdf").write_bytes(raw)
    monkeypatch.setattr(verify, "_pdf_extract", lambda pdf: (meta or {}, pdf_text, raw))
    return paper


NAMED_TEX = (f"\\author{{{NAME}}}\n% Texas A&M University, vigneshn26@tamu.edu\n"
             "\\footnote{https://github.com/vignesh-nagarajan-vn/SentCast}\n\\section*{Acknowledgements}\n")
NAMED_PDF = f"Out-of-Sample Evaluation\n{NAME}\nTexas A&M University\nSentCast github.com\n"


def test_named_allows_name_affiliation_repo(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX, NAMED_PDF, {"Author": NAME, "Title": "SentCast"}, monkeypatch=monkeypatch)
    res = verify.check_anonymity("named", paper)
    assert res["ok"], res["violations"]
    assert res["review"] == "named"
    assert res["author_name_in_pdf_text"]


def test_named_name_may_wrap_across_lines(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX, "by Vignesh\n   Nagarajan\n", monkeypatch=monkeypatch)
    assert verify.check_anonymity("named", paper)["author_name_in_pdf_text"]


def test_named_fails_without_name_in_pdf(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX, "Anonymous\n", monkeypatch=monkeypatch)
    res = verify.check_anonymity("named", paper)
    assert not res["ok"]
    assert not res["author_name_in_pdf_text"]
    assert res["violations"] == []


@pytest.mark.parametrize("leak", ["C:\\Users\\someone\\paper", "/c/Users/someone/paper", "c:/users/x",
                                  "home of nrvig", "Users\\someone"])
def test_named_fails_on_local_paths_in_source(tmp_path, monkeypatch, leak):
    paper = _paper(tmp_path, NAMED_TEX + f"% {leak}\n", NAMED_PDF, monkeypatch=monkeypatch)
    res = verify.check_anonymity("named", paper)
    assert not res["ok"]
    assert any(v["where"] == "paper/main.tex" for v in res["violations"])


def test_named_fails_on_local_path_in_metadata(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX, NAMED_PDF, {"Producer": "pdfTeX C:\\Users\\x\\miktex"},
                   monkeypatch=monkeypatch)
    res = verify.check_anonymity("named", paper)
    assert not res["ok"]
    assert res["violations"][0]["where"] == "paper/main.pdf metadata Producer"


def test_named_fails_on_local_path_in_pdf_text(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX, NAMED_PDF + "see /c/Users/x/figures\n", monkeypatch=monkeypatch)
    assert not verify.check_anonymity("named", paper)["ok"]


@pytest.mark.parametrize("raw", [b"/PTEX.FileName (C:/Users/x/fig.pdf)", b"(C:\\\\Users\\\\x)", b"nrvig"])
def test_named_fails_on_local_path_in_raw_bytes(tmp_path, monkeypatch, raw):
    paper = _paper(tmp_path, NAMED_TEX, NAMED_PDF, raw=b"%PDF-1.5\n" + raw, monkeypatch=monkeypatch)
    res = verify.check_anonymity("named", paper)
    assert not res["ok"]
    assert res["violations"][0]["where"] == "paper/main.pdf raw bytes"


def test_named_fails_on_private_email(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX + "% someone@gmail.com\n", NAMED_PDF, monkeypatch=monkeypatch)
    assert not verify.check_anonymity("named", paper)["ok"]


def test_double_blind_fails_on_name_and_repo(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX, NAMED_PDF, monkeypatch=monkeypatch)
    res = verify.check_anonymity("double-blind", paper)
    assert not res["ok"]
    assert res["review"] == "double-blind"
    patterns = {v["pattern"] for v in res["violations"]}
    assert {r"vignesh", r"nagarajan", r"github\.com", r"sentcast", r"acknowledg", r"tamu"} <= patterns


def test_double_blind_passes_anonymous_paper_with_pebsa_citation(tmp_path, monkeypatch):
    tex = "\\author{Anonymous}\nas proposed in~\\cite{nagarajan2024pebsa}.\n"
    pdf = "Anonymous\n[3] V. Nagarajan, PEBSA: forecasting with posts, 2024.\n"
    paper = _paper(tmp_path, tex, pdf, {"Author": "Anonymous"}, monkeypatch=monkeypatch)
    res = verify.check_anonymity("double-blind", paper)
    assert res["ok"], res["violations"]
    assert len(res["allowed_occurrences"]) == 1


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        verify.check_anonymity("open")


def test_default_mode_is_named(tmp_path, monkeypatch):
    paper = _paper(tmp_path, NAMED_TEX, NAMED_PDF, monkeypatch=monkeypatch)
    assert verify.check_anonymity(paper=paper)["review"] == "named"
