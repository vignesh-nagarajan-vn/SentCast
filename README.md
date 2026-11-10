# SentCast

- **Paper title:** Out-of-Sample Evaluation of Textual News Sentiment for Predicting the
  Equity Premium and Consumer Sentiment
  - Referred to in this repository as SentCast, short for Sentiment Forecast
- **Paper info:** 21 pages, 9 tables, 7 figures, 25 references (`paper/main.pdf`)
- **Author:** Vignesh Nagarajan
- **Submitted to:** [BDAI 2026 Workshop](https://www.univda.it/bdai2026/) on 2026-10-07; the workshop was cancelled
- **Target:** Journal of Forecasting (Wiley), not yet submitted

## Methodology

- **Question:** does a sentiment index built from free news headlines improve
  one-month-ahead forecasts of the S&P 500 excess return, its direction and the change in
  US consumer sentiment?
- **Data:** GDELT headlines for 2015 to 2026, FRED, the Welch-Goyal file and three
  published indices. All free.
- **Design:** models, tests and decision rule fixed before any experiment ran; twelve
  experiments, all reported; leakage tests in `tests/`.
- **Result:** null for all three targets at one month. Details: `results/SUMMARY.md`.

## Layout

```
config/        all parameters
src/sentcast/  pipeline, from downloads to tables, figures and checks
tests/         unit tests, including the leakage tests
results/       one file per experiment, tables, figures, SUMMARY.md
paper/         paper source and PDF, generated tables and figures
docs/          data check, paper outline
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

E12 reproduces from the cached labels in `data/interim/`, since a scorer model may be retired.
