"""Mechanical checks for the verification pass. Writes results/verification.json.

    python -m sentcast.verify            # checks on files
    python -m sentcast.verify --compile  # also compile paper/main.tex first

Checks:
  1. every experiment E01..E12 has a result file, from one clean commit, and a SUMMARY row
  2. every number in docs/outline.md still equals its source in results/
  2b. the paper text takes every result from the generated \\res{key} macros: the macro file
      is current, every key used exists, and no decimal number is typed in paper/main.tex
  3. tables and figures under paper/ are identical to those under results/
  4. nothing under paper/ (sources, PDF text, PDF metadata) names the author, an
     affiliation, the repository, or acknowledgements; the cited PEBSA reference is the
     one allowed place for the author's surname
  5. with --compile: paper/main.tex compiles without errors and embeds Libertinus fonts
"""
from __future__ import annotations

import csv
import filecmp
import json
import re
import subprocess
import sys

from . import outline, texnumbers
from .config import FIGURES, RESULTS, ROOT, TABLES, write_result

PAPER = ROOT / "paper"
EXPERIMENTS = [f"E{i:02d}" for i in range(1, 13)]
# identity strings that must not appear under paper/ (case-insensitive)
FORBIDDEN = [r"vignesh", r"nagarajan", r"nrvig", r"tamu", r"texas\s*a\s*&?\s*m", r"basis\s+phoenix",
             r"github\.com", r"sentcast", r"acknowledg", r"@gmail"]
TEXT_SUFFIXES = {".tex", ".bib", ".bbl", ".txt", ".md"}
PEBSA_KEY = "nagarajan2024pebsa"
# decimal numbers the paper text may contain as typed: the significance level and the
# probability at which a rise is predicted. Every other decimal is
# a result and has to come from \res{key}.
TYPED_DECIMALS = {"0.05", "0.5"}


def check_experiments() -> dict:
    files = {}
    commits = set()
    for e in EXPERIMENTS:
        hits = sorted(p.name for p in RESULTS.glob(f"{e}_*.json"))
        files[e] = hits
        for h in hits:
            commits.add(json.load(open(RESULTS / h, encoding="utf-8"))["git_commit"])
    summary = (RESULTS / "SUMMARY.md").read_text(encoding="utf-8") if (RESULTS / "SUMMARY.md").exists() else ""
    missing_rows = [e for e in EXPERIMENTS if f"| {e} |" not in summary]
    failed = [f for fs in files.values() for f in fs if "FAILED" in f]
    return {"ok": all(len(v) == 1 for v in files.values()) and not failed and not missing_rows
            and len(commits) == 1 and not any(c.endswith("-dirty") for c in commits),
            "result_files": files, "commits": sorted(commits), "failed_files": failed,
            "experiments_missing_from_summary": missing_rows}


def check_outline_numbers() -> dict:
    path = RESULTS / "outline_numbers.csv"
    if not path.exists():
        return {"ok": False, "note": "results/outline_numbers.csv is missing; run python -m sentcast.outline"}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    text = outline.OUTPUT.read_text(encoding="utf-8")
    fresh, _ = outline.render(outline.TEMPLATE.read_text(encoding="utf-8"))
    body = text.split("-->\n\n", 1)[-1]
    mismatched = []
    for r in rows:
        parts = r["placeholder"].split(":")
        if parts[0] in ("r2", "hit"):
            now = (outline._r2 if parts[0] == "r2" else outline._hit)(outline._get(parts[1], parts[2]))
        else:
            now = outline._fmt(outline._get(parts[0], parts[1]), parts[2])
        if now != r["rendered"] or now not in text:
            mismatched.append(r["placeholder"])
    return {"ok": not mismatched and body == fresh and "{{" not in text, "values_checked": len(rows),
            "mismatched": mismatched, "outline_equals_fresh_render": body == fresh,
            "unfilled_placeholders": text.count("{{")}


def check_mirror() -> dict:
    out = {"ok": True, "differing": [], "missing": []}
    for src, dst in ((TABLES, PAPER / "tables"), (FIGURES, PAPER / "figures")):
        if not src.exists():
            out["ok"] = False
            out["missing"].append(str(src.relative_to(ROOT)))
            continue
        for f in sorted(src.iterdir()):
            if f.suffix not in (".tex", ".pdf"):
                continue
            g = dst / f.name
            if not g.exists():
                out["missing"].append(f"paper/{dst.name}/{f.name}")
            elif not filecmp.cmp(f, g, shallow=False):
                out["differing"].append(f.name)
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    # a float wrapped in \IfFileExists is optional: its table exists only if that experiment ran
    used -= {re.sub(r"\.tex$", "", o) for o in re.findall(r"\\IfFileExists\{([^}]+)\}", tex)}
    for u in sorted(used):
        if not any((PAPER / (u + ext)).exists() for ext in ("", ".tex", ".pdf")):
            out["missing"].append(f"paper/{u} (referenced in main.tex)")
    out["ok"] = not out["differing"] and not out["missing"]
    return out


def check_paper_numbers() -> dict:
    """The paper text takes its results from generated macros and types none of them."""
    path = TABLES / texnumbers.NAME
    if not path.exists():
        return {"ok": False, "note": f"{path.name} is missing; run python -m sentcast.report"}
    rows = texnumbers.collect()
    fresh = path.read_text(encoding="utf-8") == texnumbers.render(rows)
    defined = {r["key"] for r in rows}
    body = re.sub(r"(?<!\\)%.*", "", (PAPER / "main.tex").read_text(encoding="utf-8"))      # drop comments
    used = re.findall(r"\\res\{([^}]*)\}", body)
    unknown = sorted(set(used) - defined)
    prose = re.sub(r"\\(?:res|ref|label|cite[tp]?|input|includegraphics(?:\[[^\]]*\])?)\{[^}]*\}", "", body)
    prose = re.sub(r"(?:CC BY|Attribution|Opus)\s+\d+\.\d+", "", prose)   # version numbers inside names
    typed = sorted({m for m in re.findall(r"(?<![\w.])\d+\.\d+", prose) if m not in TYPED_DECIMALS})
    return {"ok": fresh and not unknown and not typed, "keys_defined": len(defined), "keys_used": len(set(used)),
            "uses": len(used), "numbers_file_equals_fresh_render": fresh, "unknown_keys": unknown,
            "typed_decimals": typed}




def _scan(text: str, where: str, ignore: str = "") -> list[dict]:
    """Every line that matches a forbidden pattern. Callers decide which hits are allowed.

    `ignore` is removed from each line first: the citation key of the PEBSA reference holds the
    author's surname, and citing that reference by its key in the LaTeX source names nobody.
    """
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        probe = line.replace(ignore, "") if ignore else line
        for pat in FORBIDDEN:
            if re.search(pat, probe, flags=re.I):
                hits.append({"where": where, "line": i, "pattern": pat, "text": line.strip()[:120], "allowed": False})
    return hits


def check_anonymity() -> dict:
    hits: list[dict] = []
    for p in sorted(PAPER.rglob("*")):
        if p.is_file() and p.suffix in TEXT_SUFFIXES:
            text = p.read_text(encoding="utf-8", errors="replace")
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            for h in _scan(text, rel, ignore=PEBSA_KEY if p.suffix == ".tex" else ""):
                # the PEBSA reference is cited in the third person: its author field and key are expected
                in_pebsa = False
                if p.name in ("references.bib", "main.bbl"):
                    lines = text.splitlines()
                    lo = max(0, h["line"] - 4)
                    in_pebsa = "pebsa" in " ".join(lines[lo:h["line"] + 6]).lower()
                h["allowed"] = bool(in_pebsa and h["pattern"] in (r"nagarajan", r"vignesh"))
                hits.append(h)
    pdf = PAPER / "main.pdf"
    meta = {}
    if pdf.exists():
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, errors="replace").stdout
        for line in info.splitlines():
            k, _, v = line.partition(":")
            if k.strip() in ("Title", "Author", "Subject", "Keywords", "Creator", "Producer"):
                meta[k.strip()] = v.strip()
        for k, v in meta.items():
            for pat in FORBIDDEN:
                if re.search(pat, v, flags=re.I):
                    hits.append({"where": f"paper/main.pdf metadata {k}", "line": 0, "pattern": pat, "text": v, "allowed": False})
        txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, errors="replace").stdout
        lines = txt.splitlines()
        for h in _scan(txt, "paper/main.pdf text"):
            ctx = " ".join(lines[max(0, h["line"] - 2):h["line"] + 2]).lower()
            h["allowed"] = bool("pebsa" in ctx and h["pattern"] in (r"nagarajan",))
            hits.append(h)
        raw = pdf.read_bytes()
        for pat in (rb"Vignesh", rb"nrvig", rb"SentCast", rb"github\.com"):
            if re.search(pat, raw):
                hits.append({"where": "paper/main.pdf raw bytes", "line": 0, "pattern": pat.decode(), "text": "", "allowed": False})
    bad = [h for h in hits if not h["allowed"]]
    return {"ok": not bad, "pdf_checked": pdf.exists(), "pdf_metadata": meta, "violations": bad,
            "allowed_occurrences": [h for h in hits if h["allowed"]]}


def compile_paper() -> dict:
    def run(cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=PAPER, capture_output=True, text=True, errors="replace")

    steps = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], ["bibtex", "main"],
             ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
             ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]]
    codes = []
    last = None
    for s in steps:
        last = run(s)
        codes.append(last.returncode)
    log = (PAPER / "main.log").read_text(encoding="utf-8", errors="replace") if (PAPER / "main.log").exists() else ""
    errors = [ln for ln in log.splitlines() if ln.startswith("!")]
    warnings = [ln.strip() for ln in log.splitlines() if "Overfull" in ln or "undefined" in ln.lower()]
    fonts = run(["pdffonts", "main.pdf"]).stdout if (PAPER / "main.pdf").exists() else ""
    font_names = sorted({ln.split()[0].split("+")[-1] for ln in fonts.splitlines()[2:] if ln.strip()})
    info = run(["pdfinfo", "main.pdf"]).stdout
    pages = next((int(ln.split()[-1]) for ln in info.splitlines() if ln.startswith("Pages:")), None)
    not_embedded = [ln.split()[0] for ln in fonts.splitlines()[2:] if ln.strip() and " yes " not in ln[36:]]
    return {"ok": all(c == 0 for c in codes) and not errors and any("Libertinus" in f for f in font_names),
            "return_codes": codes, "latex_errors": errors, "warnings": warnings[:20], "pages": pages,
            "fonts": font_names, "fonts_not_embedded": not_embedded}


def main() -> None:
    report = {"experiments": check_experiments(), "outline_numbers": check_outline_numbers(),
              "paper_numbers": check_paper_numbers(), "paper_mirror": check_mirror()}
    if "--compile" in sys.argv:
        report["compile"] = compile_paper()
    report["anonymity"] = check_anonymity()
    report["all_ok"] = all(v["ok"] for v in report.values() if isinstance(v, dict))
    write_result("verification", report)
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{'OK  ' if v['ok'] else 'FAIL'} {k}")
            if not v["ok"]:
                print("     ", json.dumps({a: b for a, b in v.items() if a != "ok"}, default=str)[:1500])
    print("all ok:", report["all_ok"])


if __name__ == "__main__":
    main()
