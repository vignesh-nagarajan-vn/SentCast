# SentCast

Tests whether sentiment in public text improves forecasts of the S&P 500 monthly return
and of US consumer sentiment.

## Prior work

PEBSA: Nagarajan and Mittal (2024), IJECS 13(12),
https://doi.org/10.18535/ijecs/v13i12.4950. The paper is in `legacy/`. SentCast cites it
as motivating prior work and reuses none of its text, numbers, figures, code or data.
Audit: `docs/audit.md`.

## Status

As of 2026-10-06: Phase 1 (audit, data check, plan), Phase 2
(pipeline, corpus, experiments E01 to E12) and Phase 3 (paper skeleton, writing brief,
verification) are complete. Results: `results/SUMMARY.md`. All twelve experiments come
from one commit. They were run again after an independent check of a first run found
errors; the corrections are listed in `PLAN.md` under "Deviations", and the outcome
labels did not change. The paper is in `paper/main.tex`.

## Run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m sentcast.run
```

`sentcast.run` goes from raw downloads to every table and figure, then renders the writing
brief. Steps can be run one at a time: `python -m sentcast.run --help`. Tests:
`python -m pytest`. Verification pass: `python -m sentcast.verify --compile`.

Everything except E12 uses free data. E12 scores the headlines with two hosted language
models and runs only if `ANTHROPIC_API_KEY` is set (`pip install -e ".[llm]"`); without
the key it is recorded as skipped and its table and brief entries are left out.

## Layout

| Path | Content |
|---|---|
| `PLAN.md` | Research plan, frozen before the first experiment |
| `config/sentcast.yaml` | All parameters |
| `src/sentcast/` | Pipeline: download, score, index, forecast, evaluate, report |
| `tests/` | Unit tests, including the leakage tests |
| `results/` | One file per experiment, tables, figures, `SUMMARY.md` |
| `data/` | Downloads and intermediate files; gitignored |
| `paper/` | Paper source, generated tables and figures |
| `docs/` | Audit, data check, writing brief |
| `legacy/` | The PEBSA paper |

`results/` holds values derived from third-party series (monthly returns, index changes).
Check the source terms in `docs/data.md` before making the repository public.
