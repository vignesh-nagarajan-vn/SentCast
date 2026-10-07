# SentCast research plan

Frozen at the commit that adds this file. From that commit on, the questions, targets,
sample period, metrics and tests below do not change. Any later change is recorded in
"Deviations" at the end, dated, with the reason. Parameter values live in
`config/sentcast.yaml`, committed with this file and frozen with it.

Inputs to this plan: `docs/audit.md` (what PEBSA leaves open) and `docs/data.md` (what
data can be had for free).

## 1. Research questions

- RQ1. Does a monthly sentiment index built from free news headlines improve
  one-month-ahead out-of-sample forecasts of the S&P 500, for (a) the excess return and
  (b) its direction?
- RQ2. Does the same index improve one-month-ahead out-of-sample forecasts of the change
  in the University of Michigan Index of Consumer Sentiment (ICS)?
- RQ3. How do the constructed indices compare with published indices as predictors, and
  do they add anything beyond them?
- RQ4. Is the index a sound measurement: do scorers agree, is it reliable given the
  sampling, and does it track published measures?

### What counts as a positive or a null result

The primary predictor is the composite index (section 5). Models M0 to M3 and L0 to L3
are defined in section 6. Tests are in section 7. Significance level 0.05.

| Target | Positive | Mixed | Null |
|---|---|---|---|
| S&P 500 excess return (RQ1a) | M2 beats M0 and M3 beats M1: out-of-sample R-squared above zero and Clark-West p below 0.05 in both | exactly one of the two | neither |
| S&P 500 direction (RQ1b) | L2 hit rate above L0 hit rate and Pesaran-Timmermann p below 0.05 | n/a | otherwise |
| ICS change (RQ2) | as RQ1a | as RQ1a | as RQ1a |

Results for the four individual scorers are secondary. Their p-values are multiplied by
four (Bonferroni) within each target and benchmark. A null on the primary test is
reported as a null even if an individual scorer is significant.

## 2. Targets, horizons, sample

Forecast date. tau(m) is the last trading day of calendar month m, taken from the dates
of FRED `NASDAQCOM`. A forecast of target month m+1 is dated tau(m). Section 8 gives the
one qualification, for T2.

Targets:
- T1a, S&P 500 excess return, percent:
  R(m) = 100 * [ ln(P(m) / P(m-1)) - ln(1 + TB3MS(m-1) / 1200) ],
  with P(m) the close on tau(m). Price index, no dividends.
- T1b, S&P 500 direction: U(m) = 1 if P(m) > P(m-1), else 0.
- T2, ICS change, index points: C(m) = ICS(m) - ICS(m-1), final values.

Horizons: one month (primary). Three months (secondary, E11): the sum of R over m+1 to
m+3, and ICS(m+3) - ICS(m).

Samples:
- Text sample. Index months 2015-03 to 2026-08, target months 2015-04 to 2026-09: 138
  forecasts. The first 48 are the initial estimation window. Out-of-sample evaluation
  covers 2019-04 to 2026-09: 90 months.
- Long sample, published indices only. Target months 1980-03 to 2026-09: 559 forecasts.
  The first 120 are the initial estimation window. Out-of-sample evaluation covers
  1990-03 to 2026-09: 439 months. Results are also reported on the 90-month window of the
  text sample.

Data sources: `docs/data.md`.

## 3. Text corpus

Source: GDELT Global Knowledge Graph 2.0, 15-minute files, 2015-02-19 to 2026-09-30.

Sampling: two files per UTC day, drawn without replacement from the 96 slots by a random
generator seeded with the global seed and the date. A missing or unreadable file is
replaced by the next slot of the same day and recorded.

Headline text: page titles exist only from the third quarter of 2019, so the headline is
rebuilt from the article URL for the whole sample. The path segment with the most
hyphens or underscores is split into words; tokens that mix letters and digits, or are
numbers longer than six digits, are dropped; the result is lower-cased. Titles are stored
where present and used only in E02 and E09. A title is scored as written, after
removing HTML entities and a trailing site name.

Filters, in order:
1. the row is a web article with a URL;
2. GDELT tags at least one US location;
3. the rebuilt headline has at least 4 alphabetic words;
4. the headline contains at least one economy or market term from the fixed list in
   `config/sentcast.yaml`;
5. the source domain is not on the fixed press-release list in the same file;
6. exact duplicates are removed by URL and by (UTC date, headline).

Timestamp: the UTC time of the 15-minute file in which GDELT recorded the article. This
is at or after publication.

## 4. Sentiment scorers

Each scorer maps one headline to a number. Higher is more positive.

| Key | Scorer | Definition |
|---|---|---|
| `lm` | Lexicon, finance | Loughran-McDonald positive and negative word lists. Score = (pos - neg) / (pos + neg); 0 if no listed word. No negation handling. |
| `vader` | Lexicon, general | VADER compound score. |
| `finbert` | FinBERT | `ProsusAI/finbert`, P(positive) - P(negative), 64 tokens maximum. |
| `clf` | Simple trained classifier | TF-IDF of word 1- and 2-grams with multinomial logistic regression (C = 1), trained once on the 4,846 sentences of the Financial PhraseBank (50% agreement file), normalised like the headlines. Score = P(positive) - P(negative). |
| `llm` | LLM | Runs only if `ANTHROPIC_API_KEY` is set. Model `claude-haiku-4-5`, every kept headline (see Deviations; originally 200 per index month), 40 per request, integer label from -2 to 2 rescaled to [-1, 1]. Hard cost cap USD 30.00 (originally 5.00), computed from the token counts the API returns; the scorer stops at the cap and reports coverage. |
| `gdelt_tone` | External benchmark | GDELT's own tone for the full article. Not a headline scorer. |

No scorer is tuned on the forecast targets. FinBERT and `clf` are trained on labelled
sentiment data that predate the evaluation window. An LLM can know what happened after a
headline was written; any `llm` result carries that caveat.

## 5. From scores to an index

- Daily index: the mean score of the day's headlines (UTC date). Days with fewer than 5
  headlines are missing.
- Monthly index I(m): the mean of the daily index over the window from tau(m-1)
  inclusive to tau(m) exclusive. At least 15 valid days are required.
- Composite: `lm`, `vader`, `finbert` and `clf` monthly indices, each standardised with
  the mean and standard deviation of the initial estimation window only, then averaged
  with equal weights.

The window ends the day before the forecast date, so every headline used for a forecast
is dated before it.

## 6. Forecast models and baselines

All models are re-estimated every month on an expanding window (all data up to the
forecast date). y is the target, x the sentiment predictor available at the forecast date.

Continuous targets (T1a, T2), ordinary least squares:
- M0, historical mean: the mean of y so far.
- M1, own past: y(t+1) on a constant and y(t).
- M2, sentiment: y(t+1) on a constant and x(t).
- M3, own past plus sentiment: y(t+1) on a constant, y(t) and x(t).

Direction (T1b), logistic regression with predictors standardised inside each estimation
window, class predicted at probability 0.5:
- L0, base rate: the share of up months so far.
- L1: lagged excess return. L2: x. L3: both.

Published indices, used as x in the same models (E06) and as controls (E07):
- San Francisco Fed Daily News Sentiment Index: mean of daily values over the window
  shifted back 7 days.
- Economic Policy Uncertainty, news-based US index: lagged one month.
- ICS level: predictor for T1 only, lagged one month.

## 7. Evaluation

Scheme: expanding window, one forecast per month, no re-use of evaluation data for any
choice. All settings in this plan and the config file are fixed in advance.

Metrics:
- root mean squared error and mean absolute error;
- out-of-sample R-squared against a benchmark: 1 - SSE(model) / SSE(benchmark);
- directional accuracy (hit rate) and Brier score for T1b.

Tests:
- Nested models (M2 against M0, M3 against M1, augmented against control in E07):
  Clark-West MSPE-adjusted statistic, one-sided, Newey-West variance with h - 1 lags.
- Non-nested comparisons and Brier score differences: Diebold-Mariano, two-sided, same
  variance.
- Direction: Pesaran-Timmermann test, one-sided.

Uncertainty: 90% intervals for the out-of-sample R-squared and the hit rate from a
stationary bootstrap of the out-of-sample months (mean block 6 months at h = 1, 12 at
h = 3; 10,000 resamples; fixed seed).

Multiplicity: Bonferroni across the four individual scorers, as in section 1.

## 8. Leakage controls

1. Text. A headline enters the forecast dated tau(m) only if its UTC timestamp is before
   tau(m). The feature builder checks this and stops otherwise.
2. Macro release lags. Every monthly series used as a predictor or own lag has an entry
   in `release_lag_months` for the target it serves. A missing entry is an error.
   - ICS as its own lag: lag 0. ICS forecasts are dated at the later of tau(m) and the
     final release of the month-m survey. Text is still cut off at tau(m), so the text
     index never holds information that the own-past benchmark lacks.
   - ICS as a predictor of the S&P 500: lag 1, because the final value can be released
     after tau(m).
   - EPU: lag 1.
3. Daily published series are cut off 7 days before the forecast date.
4. Standardisation for the composite uses the initial estimation window only.
5. Scorer training data are unrelated to the targets.
6. At h = 3 only targets fully realised by the forecast date enter estimation.
7. Tests in `tests/` fail if any text timestamp is on or after its forecast date, or if
   a release lag is missing.

## 9. Experiments

Each experiment writes `results/E##_<name>.json` with its config and the git commit.
All are reported, including nulls and failures.

| ID | Name | What it does | Answers |
|---|---|---|---|
| E01 | corpus | Files drawn, missing files, filter funnel, headlines per month, share with titles, top domains | RQ4 |
| E02 | measurement | (a) scorer agreement at headline and month level; (b) URL headline against page title, per headline and per month; (c) split-half reliability of the monthly index from the two daily files; (d) correlation with the San Francisco Fed index, ICS, EPU and the same-month S&P 500 return; (e) accuracy on the Financial PhraseBank: 5-fold for `clf`, full set for the others | RQ4 |
| E03 | sp500_return | T1a, h = 1, text sample: M0 to M3 for each index and the composite | RQ1a |
| E04 | sp500_direction | T1b, h = 1, text sample: L0 to L3 for each index and the composite | RQ1b |
| E05 | ics_change | T2, h = 1, text sample: M0 to M3 for each index and the composite | RQ2 |
| E06 | published | T1a, T1b, T2 with each published index as x, long sample; also evaluated on the 90-month window | RQ3 |
| E07 | incremental | Composite added to models that already hold (a) the San Francisco Fed index, (b) all published indices, and for T2 (c) the lagged S&P 500 return | RQ3 |
| E08 | estimation | E03 to E05 with initial windows of 36 and 60 months and with a 48-month rolling window. The composite is standardised on the initial window of each variant. | robustness |
| E09 | index_definition | E03 to E05 with (a) the first difference of the index, (b) a headline-weighted monthly mean, (c) an index from page titles on the months where at least 95% of kept headlines have one and (d) the URL-headline index on those same months, both with a 36-month initial window | robustness |
| E10 | target_definition | (a) T1a without subtracting the bill rate, (b) T1a as a simple excess return, (c) T2 with the ICS own lag at release lag 1 | robustness |
| E11 | horizon3 | T1a and T2 at h = 3 | secondary horizon |
| E12 | llm | E03 to E05 with the `llm` index, also evaluated separately on origins before and after the model's training cutoff (see Deviations); recorded as skipped if no key is set | RQ1, RQ2 |

Planned outputs, all generated from results files:
- Table 1 data and corpus (E01). Table 2 measurement (E02). Table 3 S&P 500 return (E03,
  E06). Table 4 S&P 500 direction (E04, E06). Table 5 ICS change (E05, E06). Table 6
  incremental value and robustness (E07 to E11).
- Figure 1 monthly indices with the San Francisco Fed index (E02). Figure 2 cumulative
  difference in squared forecast errors against the benchmark, T1a and T2 (E03, E05).
  Figure 3 out-of-sample R-squared with intervals by scorer and target (E03, E05, E06).

## 10. Related work

Each source was opened on 2026-10-06 (publisher or Crossref record with abstract, arXiv,
RePEc, or the paper itself) and says what the line states.

| Key | Source | What it shows |
|---|---|---|
| nagarajan2024pebsa | Nagarajan, Mittal (2024). PEBSA: Predicting Economic Behavior via Sentiment Analysis. IJECS 13(12). https://doi.org/10.18535/ijecs/v13i12.4950 | Proposes forecasting US economic behaviour from social media sentiment; reports four months of 2024. Motivating prior work. |
| tetlock2007giving | Tetlock (2007). Giving Content to Investor Sentiment. J. Finance 62(3), 1139-1168. https://doi.org/10.1111/j.1540-6261.2007.01232.x | Pessimism in a daily newspaper column predicts lower market prices followed by reversal. |
| garcia2013sentiment | Garcia (2013). Sentiment during Recessions. J. Finance 68(3), 1267-1300. https://doi.org/10.1111/jofi.12027 | Word-count sentiment from New York Times columns predicts daily stock returns, mostly in recessions (1905 to 2005). |
| bollen2011twitter | Bollen, Mao, Zeng (2011). Twitter mood predicts the stock market. J. Comput. Sci. 2(1), 1-8. https://doi.org/10.1016/j.jocs.2010.12.007 | Some Twitter mood dimensions improve daily Dow Jones direction forecasts; 87.6% accuracy on a 2008 sample. |
| huang2015investor | Huang, Jiang, Tu, Zhou (2015). Investor Sentiment Aligned. Rev. Financ. Stud. 28(3), 791-837. https://doi.org/10.1093/rfs/hhu080 | An investor sentiment index built for prediction forecasts aggregate stock returns in and out of sample. |
| loughran2011liability | Loughran, McDonald (2011). When Is a Liability Not a Liability? J. Finance 66(1), 35-65. https://doi.org/10.1111/j.1540-6261.2010.01625.x | General word lists misclassify financial text; finance-specific lists are provided. |
| hutto2014vader | Hutto, Gilbert (2014). VADER. Proc. ICWSM 8(1), 216-225. https://doi.org/10.1609/icwsm.v8i1.14550 | A rule-based general sentiment lexicon validated on social media text. |
| malo2014good | Malo, Sinha, Korhonen, Wallenius, Takala (2014). Good debt or bad debt. JASIST 65(4), 782-796. https://doi.org/10.1002/asi.23062 | A human-annotated financial phrase bank for training and evaluating sentiment models. |
| araci2019finbert | Araci (2019). FinBERT. arXiv:1908.10063. https://arxiv.org/abs/1908.10063 | BERT further trained on financial text improves financial sentiment classification. |
| shapiro2022measuring | Shapiro, Sudhof, Wilson (2022). Measuring news sentiment. J. Econometrics 228(2), 221-243. https://doi.org/10.1016/j.jeconom.2020.07.053 | A newspaper sentiment index for 1980 to 2015; news sentiment predicts consumer sentiment movements. |
| baker2016measuring | Baker, Bloom, Davis (2016). Measuring Economic Policy Uncertainty. Q. J. Econ. 131(4), 1593-1636. https://doi.org/10.1093/qje/qjw024 | A newspaper-based uncertainty index; its innovations foreshadow declines in investment, output and employment. |
| gentzkow2019text | Gentzkow, Kelly, Taddy (2019). Text as Data. J. Econ. Lit. 57(3), 535-574. https://doi.org/10.1257/jel.20181020 | Survey of statistical methods and applications for text in economics. |
| ke2019predicting | Ke, Kelly, Xiu (2019). Predicting Returns with Text Data. NBER WP 26186. https://doi.org/10.3386/w26186 | A supervised method that extracts return-relevant sentiment from news articles. |
| kalamara2022making | Kalamara, Turrell, Redl, Kapetanios, Kapadia (2022). Making text count. J. Appl. Econometrics 37(5), 896-919. https://doi.org/10.1002/jae.2907 | Newspaper text improves UK forecasts of GDP, inflation and unemployment, most in periods of stress. |
| welch2008comprehensive | Welch, Goyal (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction. Rev. Financ. Stud. 21(4), 1455-1508. https://doi.org/10.1093/rfs/hhm014 | Most proposed predictors of the equity premium fail out of sample against the historical mean. |
| campbell2008predicting | Campbell, Thompson (2008). Predicting Excess Stock Returns Out of Sample. Rev. Financ. Stud. 21(4), 1509-1531. https://doi.org/10.1093/rfs/hhm055 | With sign restrictions many predictors beat the historical mean; small out-of-sample R-squared can matter to investors. |
| goyal2024comprehensive | Goyal, Welch, Zafirov (2024). A Comprehensive 2022 Look at the Empirical Performance of Equity Premium Prediction. Rev. Financ. Stud. 37(11), 3490-3557. https://doi.org/10.1093/rfs/hhae044 | Of 29 newer predictors, most lose significance or perform poorly out of sample through 2021. |
| clark2007approximately | Clark, West (2007). Approximately normal tests for equal predictive accuracy in nested models. J. Econometrics 138(1), 291-311. https://doi.org/10.1016/j.jeconom.2006.05.023 | An adjustment to the squared-error comparison of nested models that allows normal critical values. |
| diebold1995comparing | Diebold, Mariano (1995). Comparing Predictive Accuracy. J. Bus. Econ. Stat. 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599 | A test of equal forecast accuracy under general loss functions and serially correlated errors. |
| pesaran1992simple | Pesaran, Timmermann (1992). A Simple Nonparametric Test of Predictive Performance. J. Bus. Econ. Stat. 10(4), 461-465. https://doi.org/10.1080/07350015.1992.10509922 | A distribution-free test of whether direction forecasts are better than chance. |
| newey1987simple | Newey, West (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. Econometrica 55(3), 703-708. https://doi.org/10.2307/1913610 | A covariance estimator that stays valid under serial correlation. |
| politis1994stationary | Politis, Romano (1994). The Stationary Bootstrap. J. Am. Stat. Assoc. 89(428), 1303-1313. https://doi.org/10.1080/01621459.1994.10476870 | A block resampling scheme for weakly dependent stationary data. |
| lopezlira2023chatgpt | Lopez-Lira, Tang (2023). Can ChatGPT Forecast Stock Price Movements? arXiv:2304.07619. https://arxiv.org/abs/2304.07619 | LLM scores of news headlines predict stock returns after the model's knowledge cutoff. |
| glasserman2023assessing | Glasserman, Lin (2023). Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT Sentiment Analysis. arXiv:2309.17322. https://arxiv.org/abs/2309.17322 | LLM sentiment backtests are biased when the training window overlaps the test period; proposes anonymising headlines. |
| leetaru2013gdelt | Leetaru, Schrodt (2013). GDELT: Global Data on Events, Location and Tone, 1979-2012. ISA Annual Convention. http://data.gdeltproject.org/documentation/ISA.2013.GDELT.pdf | Describes the GDELT event and tone dataset built from world news. |

## 11. Page budget

Target 13 pages in the single-column style, inside the author's limit of 12 to 14.

| Section | Pages |
|---|---|
| Title, abstract, keywords | 0.5 |
| 1 Introduction | 1.25 |
| 2 Related work | 1.5 |
| 3 Data | 1.75 |
| 4 Method | 2.5 |
| 5 Results | 3.0 |
| 6 Robustness and limitations | 1.0 |
| 7 Conclusion | 0.4 |
| Declaration on Generative AI | 0.1 |
| References | 1.0 |
| Total | 13.0 |

## 12. Main risks

| Risk | Handling |
|---|---|
| 90 out-of-sample months give low power; a null is the likely outcome for returns | Intervals on every headline number; long-sample benchmark with a published index (E06) |
| March and April 2020 dominate squared-error comparisons | Figure 2 shows where gains and losses accrue |
| The mix of GDELT sources drifts over time | Reported in E01; first-difference index in E09 |
| URL-derived headlines differ from real titles | Measured in E02; title-based index in E09 |
| Two files per day is a small sample of each day | Split-half reliability in E02 |
| Several scorers and targets invite selective reporting | One pre-specified primary predictor; Bonferroni for the rest; every experiment reported |
| ICS release dates are not machine-verified | Lag rules in section 8; stricter variant in E10 |
| Compute or bandwidth runs short | Fall back to one file per day (the first drawn), recorded as a deviation |
| FRED keeps only ten years of S&P 500 data; earlier months come from the Welch-Goyal file | Cross-checked in `docs/data.md` |
## Deviations

- 2026-10-06, before any experiment ran. Clarification of section 4: the `llm` monthly
  index is the mean label over the month's 200 sampled headlines, because 200 headlines
  a month cannot meet the 5-per-day rule of section 5.
- 2026-10-06, during the corpus download, before any forecast experiment ran. In
  `config/sentcast.yaml`, `gkg.base_url` changed from http to https and `gkg.workers`
  from 8 to 16. Reason: the GDELT server began redirecting http to https, and a network
  outage left 19 calendar months with missing files. The downloader now retries with
  increasing waits and stops on a network failure instead of recording the file as
  missing. The affected months were deleted and downloaded again. The seeded draw, the
  filters and every other setting are unchanged.
- 2026-10-06, before any experiment ran and before any headline was sent to the LLM.
  Section 4 and E12, two changes to the `llm` scorer. Neither touches the primary
  predictor: `llm` stays outside the composite and E12 stays a secondary comparison.
  (1) The scorer labels every kept headline instead of 200 per index month, and the cost
  cap rises from USD 5.00 to USD 30.00. The `llm` index is then built by the rule of
  section 5 like the other scorer indices, which replaces the clarification in the first
  entry above. Reason: the corpus holds about 1,200 headlines per index month, so full
  coverage is affordable and puts the `llm` index on the same footing as the others.
  (2) E12 is also evaluated on two subsets of the test origins: those whose whole text
  window lies after the model's training data cutoff (July 2025 for `claude-haiku-4-5`,
  from Anthropic's model table read on 2026-10-06; origins 2025-09 to 2026-08, 12
  months) and those before it. Reason: section 4 notes that an LLM can know what followed
  a headline; the split shows the result on months where it cannot. The split restricts
  evaluation only. Models are still estimated on all earlier origins.
- 2026-10-06, before any experiment ran and before any headline was sent to an LLM. E12
  gains a second model, `claude-sonnet-5-5` (effort low, same prompt, labels and index
  rule, own cost cap of USD 50.00), reported next to `claude-haiku-4-5` in the same
  result file. Reason: a check that the E12 result is not specific to one model. Its
  training data cutoff is June 2026, so one test origin lies after it; the cutoff split
  is reported but cannot be evaluated for this model, and its whole result carries the
  look-ahead caveat of section 4. It stays outside the composite and the decision rule.
- 2026-10-06, note on the entries above. A dry run at about 14:41, on the 2015 to 2017
  part of the corpus and with outputs outside the repository, exercised all experiment
  code. It produced the E06 numbers and the baseline forecasts on real data; no forecast
  from a text index existed, because 33 index months are fewer than the initial window.
  Entries dated after that time that say "before any experiment ran" mean before any
  experiment of record ran. No design choice was changed after the dry run.
- 2026-10-06, after the first full run (results at commit `1d3e830`), following an
  independent check. Three corrections, then all experiments were run again:
  (1) Corpus. The downloader stopped looking for a replacement after three later slots;
  section 3 sets no limit. It now tries every later slot of the day. The months with a
  failed draw were downloaded again.
  (2) E10c. Only the own lag had been moved back one month; estimation still used the
  target that needs the month-m survey value. Estimation now stops one origin earlier.
  (3) Tables. Bonferroni-adjusted p-values for the four scorers are printed next to the
  raw ones, and E07 moved from Table 6 to its own Table 6b, because its benchmarks are
  not M0 or M1.
  The outcome labels of the first run and of the rerun are both in the git history.
- 2026-10-06, after E01 to E11 were run again on the corrected corpus. Section 9, planned
  outputs: a Table 7 is added for E12, which the list did not cover because E12 runs
  only with an API key. The table and the E12 entries of the writing brief are written
  only when E12 has run. E01 also records the index month with the fewest valid days.
  No experiment, model, test or rule changed. Because the reporting code changed, E01
  to E12 were run once more so that every result file carries one commit.
- 2026-10-06, after the run of record, following a second independent check of the final
  files. A record of how E12 was run, and clarifications. No experiment, model, test or
  rule changed, and no experiment was run again.
  (1) E12, two runs. The labels come from two runs of the same labelling code
  (`src/sentcast/scorers.py`, unchanged since commit `ea4d331`). The first run started
  before the corpus correction was finished and labelled the corpus as it stood before
  the nine months were downloaded again. The second run, at the commit of record, asked
  only for headlines without a label: those new in the corrected corpus and those of
  requests whose reply was unusable in the first run. The E12 result of record was
  computed in the second run, from all labels, on the corrected corpus. The result file
  of the first run is kept as `results/superseded/E12_llm_run1.json`; it was written
  from commit `9174353` with uncommitted reporting changes.
  (2) Wording of entries above. "Origins after the cutoff" are the origins whose whole
  text window lies after the cutoff month; an origin whose window starts inside the
  cutoff month belongs to the other group. The entry on the `llm` scorer gives about
  1,200 headlines per index month; E01 reports the measured count after de-duplication,
  which is lower. The dry run of 14:41 held index values for 34 months, 2015-03 to
  2017-12, not 33.
  (3) Section 3, replacement. The search for a replacement file continues at 00:00 after
  the last slot of the day, so a replacement can lie earlier in the same UTC day than
  the drawn slot. A file that cannot be read counts as missing.
  (4) Choices made in the code that sections 1 to 9 do not state. A comparison is
  computed only with 12 or more forecasts. E10c is evaluated with the settings of a
  two-month information lag (mean block 12, one Newey-West lag). Split-half reliability
  requires 3 headlines per day in each half and is stepped up with the Spearman-Brown
  formula. Lexicon and VADER scores are turned into classes on the labelled sentences at
  0.05 and -0.05. A monthly San Francisco Fed value needs 15 daily values. The downloader
  makes 8 attempts per file; `gkg.max_retries` in the config has no effect. The second
  LLM runs with `max_tokens` 4096. RMSE and MAE are in the result files and in no table.
  Table 2b (accuracy on the labelled sentences, E02) is not in the list of section 9.
- 2026-10-07. Sections 11 and 12 and this list: notes on the submission venue and on
  how the paper text is written are removed. The research design, the experiments and
  the decision rule are unchanged.
- 2026-10-07, after the run of record and after the first paper draft. Two additions to
  the reporting. No experiment, model, test or rule changed, and no experiment was run
  again.
  (1) Power. A simulation of the power of the main test is added (`src/sentcast/power.py`,
  `results/power.json`). It is not pre-specified and was designed after the outcomes
  were known. The composite is held at its observed values, the target is simulated for
  a grid of true R-squared values with noise drawn from the standardised observed
  target, and each sample goes through the M0 and M2 forecasts and the Clark-West test
  of section 7. Power is the share of samples that meet both conditions of the rule of
  section 1 for M2 against M0. It covers the composite at the one-month horizon for the
  S&P 500 return and the consumer sentiment change. Reason: the outcome is null on 90
  test months, and the power states which effect sizes the test could have found. It
  has no outcome label and does not enter the decision rule.
  (2) Section 9, planned outputs. Four figures are added, drawn from existing result
  files: corpus coverage by index month (E01), the timing of the inputs of one forecast
  and the two samples (config, E03, E06), pairwise scorer agreement (E02) and the
  predicted probabilities of the direction models (E04).
