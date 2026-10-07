# SentCast

- **Paper title:** Out-of-Sample Evaluation of Textual News Sentiment for Predicting the
  Equity Premium and Consumer Sentiment
  - Referred to in this repository as SentCast, short for Sentiment Forecast
- **Paper info:** 17 pages, 9 tables, 7 figures, 25 references (`paper/main.pdf`)
- **Author:** Vignesh Nagarajan
- **Target venue:** [BDAI 2026 Workshop](https://www.univda.it/bdai2026/)

## Methodology

- **Question:** does a sentiment index built from free news headlines improve
  one-month-ahead forecasts of the S&P 500 excess return, its direction and the change in
  US consumer sentiment?
- **Data:** GDELT headlines for 2015 to 2026, FRED, the Welch-Goyal file and three
  published indices. All free.
- **Design:** plan, models, tests and decision rule frozen before any experiment ran
  (`PLAN.md`); twelve experiments, all reported; leakage tests in `tests/`.
- **Result:** null for all three targets at one month. Details: `results/SUMMARY.md`.

## Prior work

PEBSA: Nagarajan and Mittal (2024), IJECS 13(12),
https://doi.org/10.18535/ijecs/v13i12.4950. The paper is in `legacy/`. SentCast cites it
as motivating prior work and reuses none of its text, numbers, figures, code or data.
Audit: `docs/audit.md`.

## Layout

```
PLAN.md        research plan, frozen before the first experiment
config/        all parameters
src/sentcast/  pipeline, from downloads to tables, figures and checks
tests/         unit tests, including the leakage tests
results/       one file per experiment, tables, figures, SUMMARY.md
paper/         paper source and PDF, generated tables and figures
docs/          PEBSA audit, data check, paper outline
legacy/        the PEBSA paper
data/          downloads and intermediate files; gitignored
```

`results/` holds values derived from third-party series. Check the source terms in
`docs/data.md` before making the repository public.

## Run

```bash
pip install -e ".[dev]"
python -m sentcast.run
```

Runs every step from raw downloads to tables and figures; `--help` lists them. E12 needs
`ANTHROPIC_API_KEY` and is skipped without it.
