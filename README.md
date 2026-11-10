# SentCast

**Sentiment Forecast: an out-of-sample test of news headline sentiment as a predictor of the
S&P 500 and US consumer sentiment.**

Does a monthly sentiment index built from free news headlines improve one-month-ahead
forecasts of the S&P 500 excess return, its direction and the change in the University of
Michigan Index of Consumer Sentiment? SentCast answers with a test fixed in advance: the
sample, models, statistical tests and decision rule were written down before any experiment
ran, and all twelve experiments are reported. The result is null for all three targets at
one month.

*Built by Vignesh Nagarajan.*

* **Paper:** *Out-of-Sample Evaluation of Textual News Sentiment for Predicting the Equity
  Premium and Consumer Sentiment* ([`paper/main.pdf`](paper/main.pdf)), 21 pages, 9 tables,
  7 figures, 25 references.
* **Workshop:** submitted to BDAI 2026 on 7 Oct 2026; the workshop was cancelled.
* **Journal:** Journal of Forecasting (Wiley), not yet submitted.

## Tech Stack

<!--- ML / NLP & scientific computing --->
<p align="center"><sub><b>ML / NLP & Scientific Computing:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white">
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
</p>

<!--- Language, config & testing --->
<p align="center"><sub><b>Language, Config & Testing:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/python%203.11+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
  <img src="https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515">
  <img src="https://img.shields.io/badge/pytest-%230A9EDC.svg?style=for-the-badge&logo=pytest&logoColor=white">
</p>

<!--- Tooling & docs --->
<p align="center"><sub><b>Tooling & Docs:</b></sub></p>
<p align="center">
  <img src="https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white">
  <img src="https://img.shields.io/badge/LaTeX-008080?style=for-the-badge&logo=latex&logoColor=white">
  <img src="https://img.shields.io/badge/BibTeX-008080?style=for-the-badge&logo=latex&logoColor=white">
</p>

## Functionality

- **Headline corpus:** a seeded draw of two GDELT files a day, 2015 to 2026, with headlines
  rebuilt from article URLs and filtered to US economy and market news.
- **Four scorers and a composite:** the Loughran-McDonald word lists, VADER, FinBERT and a
  classifier trained on the Financial PhraseBank, standardised and averaged; two large
  language models as a secondary comparison.
- **Leakage-checked forecasts:** expanding-window OLS and logit models; tests fail if a
  headline dated on or after a forecast date, or a series without a release lag, enters a
  forecast.
- **Pre-specified evaluation:** out-of-sample R-squared, Clark-West, Pesaran-Timmermann and
  Diebold-Mariano tests, stationary bootstrap intervals, and a decision rule that maps the
  results to positive, mixed or null.
- **Generated paper numbers:** every result in the paper is printed from `results/` through
  `\res{key}` macros, and `sentcast.verify` checks them.

## Repository layout

    SentCast/
    |- config/        <- every parameter, frozen
    |- src/sentcast/  <- pipeline, from downloads to tables, figures and checks
    |- tests/         <- unit tests, including the leakage tests
    |- results/       <- one file per experiment, tables, figures, SUMMARY.md
    |- paper/         <- paper source and PDF; archive/ holds the workshop versions
    |- docs/          <- data check, paper outline
    \- data/          <- downloads and intermediate files; gitignored

`results/` holds values derived from third-party series; the source terms are in
[`docs/data.md`](docs/data.md).

## Run

```bash
pip install -e ".[dev]"
python -m sentcast.run
python -m sentcast.verify --compile
```

`sentcast.run` runs every step from raw downloads to tables and figures; `--help` lists
them. E12 needs `ANTHROPIC_API_KEY` and is skipped without it. E12 reproduces from the
cached labels in `data/interim/`, since a scorer model may be retired.
