# Writing brief

One entry per section, in page-budget order (13 pages in total).
Entries are notes to draft from, not paper text. Numbers are filled from `results/`
by `python -m sentcast.outline`; every number filled from `results/` is listed with its
source in `results/outline_numbers.csv`. Design constants (window lengths, thresholds,
dates) come from `config/sentcast.yaml`. Two typed numbers come from
elsewhere: the minimum of 12 forecasts for a comparison (`src/sentcast/experiments.py`)
and the first FRED month of the S&P 500, 2016-11 (`docs/data.md`,
`results/data_check.json`; it moves with the download date).

Conventions:
- M0 historical mean; M1 own lag; M2 index; M3 own lag + index. L0 to L3: logistic
  models of direction with the same predictors (L0 base rate; L1 uses the lagged excess
  return). Composite: the pre-specified primary predictor.
- R2_OS: out-of-sample R-squared, percent, written as value (90% interval, p, n) =
  stationary-bootstrap interval, one-sided Clark-West p, number of test months.
- Outcome: label from the rule fixed in advance (positive, mixed, null). A
  rule output; its meaning is yours to state.
- Sources: BibTeX keys in `paper/references.bib`.
- Tables and figures are named here by their files: Table 2b is
  `paper/tables/tab2b_labelled.tex`, Table 6b `tab6b_incremental.tex`, Table 7
  `tab7_llm.tex`. LaTeX numbers floats in order of appearance, so the compiled PDF shows
  other numbers; the labels are in `paper/main.tex`.

---

## 0. Title, abstract, keywords (0.5 page)

Purpose: question, method, result, for a reader with no background in finance,
statistics or machine learning.

Title: set in `paper/main.tex`; chosen by the author on 2026-10-07 from candidates the AI
tool proposed.

Facts the abstract needs:
- Question: does sentiment measured from free news headlines improve forecasts of
  (a) next month's S&P 500 return and direction, (b) next month's change in US consumer
  sentiment.
- What was done:
  - {{E01_corpus:dedup.after_headline_dedup:int}} headlines from
    {{E01_corpus:files.downloaded:int}} GDELT files, {{E01_corpus:files.first_day:str}}
    to {{E01_corpus:files.last_day:str}} (E01, `results/E01_corpus.json`)
  - four scorers (lexicon, VADER and FinBERT off the shelf; one classifier trained here
    on public labelled sentences), one monthly index
  - forecasts built from text dated before the forecast date and from series under the
    release rules
  - {{E03_sp500_return:sample.n_oos:int}} test months,
    {{E03_sp500_return:sample.first_oos_target:str}} to
    {{E03_sp500_return:sample.last_oos_target:str}} (E03)
  - plan and decision rule fixed before any experiment; later changes are dated
    (the E12 design, before any LLM request; corrections to the
    downloader and to E10c after a first full run)
<!-- if E12_llm:status == ran -->
  - secondary comparison: the same forecasts with an index from each of two LLM scorers
    (E12, section 6)
<!-- end -->
- Main numbers, pick one or two:
  - S&P 500 return, composite vs historical mean: R2_OS
    {{r2:E03_sp500_return:results.composite.M2_vs_M0}}; outcome
    {{E03_sp500_return:decision.outcome:str}} (E03, `results/E03_sp500_return.json`)
  - S&P 500 direction, composite: hit rate {{hit:E04_sp500_direction:results.composite.L2}};
    outcome {{E04_sp500_direction:decision.outcome:str}} (E04,
    `results/E04_sp500_direction.json`)
  - consumer sentiment change, composite vs historical mean: R2_OS
    {{r2:E05_ics_change:results.composite.M2_vs_M0}}; composite added to own-lag model:
    {{r2:E05_ics_change:results.composite.M3_vs_M1}}; outcome
    {{E05_ics_change:decision.outcome:str}} (E05, `results/E05_ics_change.json`)
- Plain reading of R2_OS for this audience: percent by which summed squared forecast
  errors are smaller (positive) or larger (negative) than the baseline's.

Candidate keywords: sentiment analysis; news text; stock return forecasting; consumer
sentiment; out-of-sample evaluation; GDELT.

Questions only you can answer:
- Which target leads the abstract?
- How to word an outcome of "null" or "mixed" for a general reader?
- Does the abstract mention the prior paper?

---

## 1. Introduction (1.25 pages)

Purpose: motivate the question, state it, say what is different about this test,
preview the numbers.

Points, in order:
1. Text as a timely signal about the economy and markets.
   Sources: gentzkow2019text, tetlock2007giving, shapiro2022measuring.
2. Motivating prior work, third person: nagarajan2024pebsa. Facts available:
   - proposes forecasting US economic behaviour from social media sentiment
   - reports forecasts for four months, January to April 2024
   - own stated limitations: overlap of training and input data; social media users not
     representative of the public; US only; hardware
   - own stated future work: longer sample, model comparison
3. Difficulty of the task in the literature:
   - most return predictors fail out of sample: welch2008comprehensive,
     goyal2024comprehensive
   - with sign restrictions many beat the historical mean, and a small out-of-sample
     R-squared can matter to investors: campbell2008predicting
   - sentiment and aggregate returns: huang2015investor
   - news sentiment and consumer sentiment: shapiro2022measuring
4. The four questions RQ1 to RQ4.
5. Features of the test: plan and decision rule fixed in advance, later changes dated;
   free, re-downloadable data only, except the LLM scorers of
   E12; month-by-month forecasts from past data; automated look-ahead checks; all
   experiments reported.
6. Preview, numbers only:
   - S&P 500 return: {{r2:E03_sp500_return:results.composite.M2_vs_M0}}; outcome
     {{E03_sp500_return:decision.outcome:str}} (E03)
   - S&P 500 direction: {{hit:E04_sp500_direction:results.composite.L2}}; outcome
     {{E04_sp500_direction:decision.outcome:str}} (E04)
   - consumer sentiment: vs M0 {{r2:E05_ics_change:results.composite.M2_vs_M0}}; vs M1
     {{r2:E05_ics_change:results.composite.M3_vs_M1}}; outcome
     {{E05_ics_change:decision.outcome:str}} (E05)
   - published index, long sample, San Francisco Fed index: S&P 500 vs M0
     {{r2:E06_published:results.long.sp500_return.sffed_nsi.M2_vs_M0}}; consumer
     sentiment vs M1 {{r2:E06_published:results.long.ics_change.sffed_nsi.M3_vs_M1}}
     (E06, `results/E06_published.json`)
7. Contributions, as facts: one-command pipeline on free data (E12 also needs a paid
   API); pre-specified test with
   all outcomes reported; measurement checks of a URL-derived headline corpus (E02);
   side-by-side with published indices on {{E06_published:sample.n_oos:int}} test months
   (E06).
8. Roadmap.

Terms needing a plain lead-in:
- Out-of-sample: fitted only on months before the forecast month; judged on unseen
  months; in-sample fit does not show this.
- Baseline: the two simplest forecasts, the average so far (M0) and a regression on the
  target's own previous value (M1).

Questions only you can answer:
- How strongly to state the answer to each question?
- Framing: hypothesis test, check of earlier claims, or system description?

---

## 2. Related work (1.5 pages)

Purpose: place the study among work on text and markets, text and the macroeconomy,
out-of-sample return prediction, and sentiment scoring.

Groups:
1. News and social media sentiment and stock returns: tetlock2007giving (daily column
   pessimism, reversal); garcia2013sentiment (daily, concentrated in recessions, 1905 to
   2005); bollen2011twitter (daily Dow Jones direction, 2008 sample); ke2019predicting
   (supervised sentiment from news articles).
2. Sentiment indices and aggregate stock returns: huang2015investor.
3. Out-of-sample evidence on return prediction: welch2008comprehensive,
   campbell2008predicting, goyal2024comprehensive.
4. Text for macroeconomic measurement and forecasting: shapiro2022measuring,
   baker2016measuring, kalamara2022making, gentzkow2019text.
5. Scorers used here: loughran2011liability, hutto2014vader, malo2014good,
   araci2019finbert.
6. Large language models as scorers; look-ahead: lopezlira2023chatgpt,
   glasserman2023assessing.
7. Data: leetaru2013gdelt.
8. Prior work, third person: nagarajan2024pebsa.

Facts for positioning:
- frequency: tetlock2007giving, garcia2013sentiment and bollen2011twitter use daily data;
  this study monthly
- sample: text {{E03_sp500_return:sample.first_origin:str}} to
  {{E03_sp500_return:sample.last_origin:str}}; published indices from 1980
- inputs: free, re-downloadable sources only, apart from the LLM scorers of E12; no
  vendor news feed
- evaluation: out-of-sample R-squared (campbell2008predicting) and the nested-model test
  (clark2007approximately)

Terms needing a plain lead-in:
- Lexicon: fixed list of words tagged positive or negative; scoring by counting.
- Excess return (equity premium): stock return minus the return on a safe short-term
  government bill over the same month.
- Sentiment index: one number per period summarising many texts or survey answers.

Questions only you can answer:
- Which gap in this literature do you claim to address?
- How to refer to the prior paper without self-identification?

---

## 3. Data (1.75 pages)

Purpose: every series and the corpus, in enough detail to rebuild them. Table 1 here
(`paper/tables/tab1_corpus.tex`, E01).

Points, in order:
1. Targets.
   - S&P 500 close on the last trading day of the month; FRED from 2016-11, Welch-Goyal
     file before (`docs/data.md`); price index, no dividends
   - excess return = log price change minus monthly 3-month Treasury bill rate
   - direction = 1 if the close is above the previous month-end close; share of up
     months in the test window
     {{E04_sp500_direction:results.baselines.L0.share_up_actual:pct1}}% (E04)
   - consumer sentiment = monthly change in the University of Michigan index, final values
   - {{E03_sp500_return:sample.n_origins:int}} forecasts; first
     {{E03_sp500_return:sample.initial_window:int}} for the initial estimate;
     {{E03_sp500_return:sample.n_oos:int}} test months (E03)
2. Text corpus (E01, `results/E01_corpus.json`). Source: leetaru2013gdelt.
   - two 15-minute files per UTC day, seeded draw: {{E01_corpus:files.requested:int}}
     requested; {{E01_corpus:files.downloaded:int}} downloaded;
     {{E01_corpus:files.replaced_by_next_slot:int}} replaced by another slot of the same
     UTC day; {{E01_corpus:files.failed:int}} draws left without a file;
     {{E01_corpus:files.bytes_downloaded:gb}} GB
   - replacement: the next slot that has a file, continuing at 00:00 after 23:45, so a
     replacement can be earlier in the day than the drawn slot
   - a draw is left without a file only when no slot of that day gives a readable file
     (all 96 slots tried; the manifest records how many, not why; `docs/data.md`
     section 4): days with no file {{E01_corpus:files.days_with_no_file:int}}, days
     with one file {{E01_corpus:files.days_with_one_file:int}}; by month in
     `results/E01_corpus.json`, `files.unavailable_by_month`
   - filter funnel, article rows: {{E01_corpus:funnel.rows:int}} read;
     {{E01_corpus:funnel.us:int}} with a US location;
     {{E01_corpus:funnel.headline:int}} with a URL headline of four or more words;
     {{E01_corpus:funnel.keyword:int}} with an economy or market term;
     {{E01_corpus:funnel.kept:int}} after dropping press-release domains;
     {{E01_corpus:dedup.after_headline_dedup:int}} after de-duplication
   - headlines per index month: min {{E01_corpus:headlines_per_index_month.min:int}},
     median {{E01_corpus:headlines_per_index_month.median:count}}, max
     {{E01_corpus:headlines_per_index_month.max:int}}; index months failing the coverage
     rule: {{E01_corpus:n_index_months_failing_coverage:int}}
   - headlines that enter the {{E01_corpus:headlines_per_index_month.months:int}} index
     months: {{E01_corpus:headlines_per_index_month.total:int}} of the
     {{E01_corpus:dedup.after_headline_dedup:int}}; the rest are dated outside those
     windows or on days with fewer than 5 headlines
   - sources: {{E01_corpus:domains.distinct:int}} domains; ten largest hold
     {{E01_corpus:domains.top10_share:pct1}}% of headlines; distinct domains 2015:
     {{E01_corpus:by_year.2015.distinct_domains:int}}; 2025:
     {{E01_corpus:by_year.2025.distinct_domains:int}}
   - page titles: on {{E01_corpus:titles.share_with_title_overall:pct1}}% of headlines;
     first index month with titles on at least 95%:
     {{E01_corpus:titles.first_month_at_threshold:str}}; reason for the URL-derived headline
3. Published indices (`docs/data.md`): San Francisco Fed Daily News Sentiment Index
   (shapiro2022measuring); news-based Economic Policy Uncertainty index
   (baker2016measuring); consumer sentiment level as predictor of the S&P 500.
4. Scorer inputs: Loughran-McDonald word lists (loughran2011liability); Financial
   PhraseBank (malo2014good); FinBERT weights (araci2019finbert).
5. Access and terms, briefly; details in `docs/data.md`.

Terms needing a plain lead-in:
- Log return, excess return: as in section 2; bill rate known at the start of the month.
- Index of Consumer Sentiment: monthly household survey; final value released at the
  end of the month or in the first days of the next.
- GDELT: open database monitoring world news; one file every 15 minutes; one row per
  article with URL, detected locations, tone score.
- URL-derived headline: words in the article's web address; used because titles exist
  for part of the sample only; compared with real titles in section 5.1.

Questions only you can answer:
- How much of the keyword and press-release lists goes in the paper, and where?
- Do you report why Yahoo Finance and Stooq were not used?

---

## 4. Method (2.5 pages)

Purpose: scorers, index, models, evaluation, look-ahead controls, each readable by both
audiences.

Points, in order:
1. Scorers: input and output of each. Sources:
   loughran2011liability, hutto2014vader, araci2019finbert, malo2014good. Share of
   headlines scored exactly zero: lexicon
   {{E02_measurement:score_stats.lm.share_zero:pct1}}%, VADER
   {{E02_measurement:score_stats.vader.share_zero:pct1}}% (E02,
   `results/E02_measurement.json`).
2. Index: daily mean; monthly mean of daily means over the window
   ending the day before the forecast date; composite of four standardised indices.
3. Models: M0 to M3; L0 to L3; published indices in the same models.
4. Evaluation: expanding window; R2_OS; Clark-West; Diebold-Mariano;
   Pesaran-Timmermann; bootstrap intervals; Bonferroni over the four scorers. Sources:
   campbell2008predicting, clark2007approximately, diebold1995comparing,
   pesaran1992simple, newey1987simple, politis1994stationary.
5. Decision rule fixed in advance.
6. Leakage controls; tests that enforce them (`tests/`).

Terms needing a plain lead-in before the formal definition, with the facts the lead-in
must get right:

- Out-of-sample testing, expanding window
  - each month-end: fit on all months up to that date, forecast the next month, add the
    month, repeat
  - forecast month never in the fitting data
  - first forecast after {{E03_sp500_return:sample.initial_window:int}} months;
    {{E03_sp500_return:sample.n_oos:int}} forecasts
  - purpose: accuracy is measured only on months that were not used for fitting
  - rolling = fixed-length window, the variant in E08

- Out-of-sample R-squared
  - one minus (model's summed squared forecast errors / benchmark's), over test months
  - above zero: model errors smaller; zero: equal; below zero: larger; no lower bound
  - unlike in-sample R-squared, can be negative
  - source: campbell2008predicting

- Significance test for nested models (Clark-West)
  - nested: benchmark = larger model with the extra coefficient fixed at zero
  - larger model estimates one more coefficient; if the true value is zero the estimate
    only adds noise, so its squared errors are expected to be slightly larger in test
    months
  - the test removes that expected penalty, then asks whether the adjusted difference
    is above zero; one-sided; level 0.05
  - R2_OS can sit at or just below zero while the test rejects; the decision rule needs
    both R2_OS above zero and p below 0.05
  - source: clark2007approximately; overlap at the three-month horizon handled with
    newey1987simple

- Leakage, release lags
  - leakage: information dated after a forecast was made entering that forecast
  - text rule: headline used only if GDELT recorded it before the forecast date (last
    trading day of the month); GDELT's timestamp is at or after publication
  - release lag: a statistic about month m is published after month m
  - rules: consumer sentiment treated as known at its release, and lagged one month
    when it predicts the S&P 500; policy uncertainty lagged one month; San Francisco
    Fed index cut off seven days before the forecast date
  - composite standardised on the first
    {{E03_sp500_return:sample.initial_window:int}} months only
  - repository tests fail on a headline dated on or after its forecast date, or on a
    missing release lag

- Each sentiment scorer
  - lexicon (Loughran-McDonald): two fixed word lists built for financial text; score =
    (positive count - negative count) / (positive + negative); zero if no listed word;
    negation not handled
  - VADER: general-purpose, rule-based; word list plus rules for emphasis and negation;
    one number in [-1, 1]
  - FinBERT: neural language model further trained on financial text and labelled
    financial sentences; outputs probabilities of positive, negative, neutral; score =
    P(positive) - P(negative); not trained by this project
  - classifier: logistic regression on weighted counts of words and word pairs; trained
    by this project on {{E02_measurement:e_labelled_data.clf_5fold.n:int}} labelled
    sentences; five-fold accuracy
    {{E02_measurement:e_labelled_data.clf_5fold.accuracy_mean:pct1}}%; majority-class
    share {{E02_measurement:e_labelled_data.majority_class_share:pct1}}% (E02)
  - GDELT tone: GDELT's own score of the whole article; a benchmark, not a headline
    scorer
<!-- if E12_llm:status == ran -->
  - LLM scorers (E12, `results/E12_llm.json`): two models of one provider,
    {{E12_llm:usage.model:str}} and {{E12_llm:second_model.usage.model:str}}, the second
    at its low effort setting
    - request: up to 40 headlines; reply: one integer per headline from -2 (very negative) to 2
      (very positive), rescaled to [-1, 1]; the prompt asks for a judgement of the
      wording only, as news for the economic or market outlook
    - every kept headline goes to each model; a distinct text is sent until it has a
      label, and the stored label is then reused; a request with an unusable reply gives
      no labels, its headlines are sent again in the next run, and no other model is
      asked
    - the labels of record come from two runs of the same labelling code: the first on
      the corpus before nine months were downloaded again, the second for the headlines
      still without a label
    - requests over both runs: {{E12_llm:usage.requests:int}} (first model) and
      {{E12_llm:second_model.usage.requests:int}} (second), of which unusable
      {{E12_llm:usage.discarded_batches:int}} and
      {{E12_llm:second_model.usage.discarded_batches:int}}; distinct texts requested in
      the run that wrote the result file: {{E12_llm:usage.requested_this_run:int}} and
      {{E12_llm:second_model.usage.requested_this_run:int}}; labelled:
      {{E12_llm:usage.labelled:int}} of {{E12_llm:usage.texts:int}} headlines (first
      model) and {{E12_llm:second_model.usage.labelled:int}} (second)
    - index built by the same rule as for the other scorers; not part of the composite
      or the decision rule
    - caveat: the model may know what followed a headline (glasserman2023assessing);
      training data cutoffs from the provider's model table:
      {{E12_llm:by_training_cutoff.training_cutoff:str}} and
      {{E12_llm:second_model.by_training_cutoff.training_cutoff:str}}; results are also
      split at the cutoff (section 6)
    - the one step that uses a paid service; cost over both runs, computed from the
      token counts the API returned and the prices typed in `src/sentcast/scorers.py`,
      not a billed amount: USD {{E12_llm:usage.cost_usd:num}} and USD
      {{E12_llm:second_model.usage.cost_usd:num}}
<!-- else -->
  - LLM scorer: specified in the plan; status {{E12_llm:status:str}} (E12,
    `results/E12_llm.json`); caveat: the model may know what followed a headline
    (glasserman2023assessing)
<!-- end -->

- Index construction
  - score per headline; day's index = mean over the day's headlines; days under 5
    headlines dropped
  - month's index = mean of daily values from the previous forecast date to the day
    before this one; at least 15 valid days
  - composite: each of four indices rescaled with its mean and standard deviation from
    the first 48 months, then averaged

- Also first used here
  - Bonferroni: p multiplied by four, the number of individual scorers
  - bootstrap interval: test months resampled in blocks; 90%
  - direction test (Pesaran-Timmermann): do up and down calls match outcomes more often
    than chance
  - Brier score: mean squared error of a probability forecast

Schematic to draw yourself, if wanted. Content: one time line for a single forecast,
with (1) the text window from the previous forecast date to the day before this one,
(2) the forecast date on the last trading day of month m, (3) the target month m+1 from
that close to the next month-end close, (4) underneath, the growing estimation sample,
(5) a release-lagged series entering one month earlier. Optional second panel: boxes in
order for GDELT files, filters, four scorers, daily index, monthly index, forecast
models, evaluation.

Questions only you can answer:
- How much formal notation for a mixed audience?
- Is the decision rule presented as pre-registration, and in what words?

---

## 5. Results (3.0 pages)

Purpose: report the experiments, measurement first, then forecasts. Tables 2 to 5 and
Figures 1 to 3 here.

### 5.1 Measurement (E02, `results/E02_measurement.json`; Table 2, Figure 1)

- Scorer agreement, headline level, Pearson, n = {{E02_measurement:n_headlines:int}}:
  FinBERT-classifier
  {{E02_measurement:a_scorer_agreement.headline_level.finbert~clf.pearson:num}};
  FinBERT-lexicon {{E02_measurement:a_scorer_agreement.headline_level.lm~finbert.pearson:num}};
  FinBERT-VADER {{E02_measurement:a_scorer_agreement.headline_level.vader~finbert.pearson:num}};
  FinBERT-GDELT tone
  {{E02_measurement:a_scorer_agreement.headline_level.finbert~gdelt_tone.pearson:num}}
- Scorer agreement, month level, Pearson, n =
  {{E02_measurement:a_scorer_agreement.month_level.finbert~clf.n:int}}: FinBERT-classifier
  {{E02_measurement:a_scorer_agreement.month_level.finbert~clf.pearson:num}};
  FinBERT-lexicon {{E02_measurement:a_scorer_agreement.month_level.lm~finbert.pearson:num}};
  FinBERT-VADER {{E02_measurement:a_scorer_agreement.month_level.vader~finbert.pearson:num}}
- URL headline vs page title, same scorer, headline level, n =
  {{E02_measurement:b_url_headline_vs_title.headline_level.finbert.n:int}}: FinBERT
  {{E02_measurement:b_url_headline_vs_title.headline_level.finbert.pearson:num}}; lexicon
  {{E02_measurement:b_url_headline_vs_title.headline_level.lm.pearson:num}}; VADER
  {{E02_measurement:b_url_headline_vs_title.headline_level.vader.pearson:num}}; classifier
  {{E02_measurement:b_url_headline_vs_title.headline_level.clf.pearson:num}}
- URL headline vs page title, month level, n =
  {{E02_measurement:b_url_headline_vs_title.month_level.finbert.n:int}}: FinBERT
  {{E02_measurement:b_url_headline_vs_title.month_level.finbert.pearson:num}}; lexicon
  {{E02_measurement:b_url_headline_vs_title.month_level.lm.pearson:num}}; VADER
  {{E02_measurement:b_url_headline_vs_title.month_level.vader.pearson:num}}; classifier
  {{E02_measurement:b_url_headline_vs_title.month_level.clf.pearson:num}}
- Share of URL headlines identical to the normalised title:
  {{E02_measurement:b_url_headline_vs_title.share_url_headline_equal_to_normalised_title:pct1}}%
- Split-half reliability of the monthly index (Spearman-Brown, two daily files), n =
  {{E02_measurement:c_split_half.by_index.finbert.n:int}} months: FinBERT
  {{E02_measurement:c_split_half.by_index.finbert.spearman_brown:num}}; classifier
  {{E02_measurement:c_split_half.by_index.clf.spearman_brown:num}}; lexicon
  {{E02_measurement:c_split_half.by_index.lm.spearman_brown:num}}; VADER
  {{E02_measurement:c_split_half.by_index.vader.spearman_brown:num}}; GDELT tone
  {{E02_measurement:c_split_half.by_index.gdelt_tone.spearman_brown:num}}
- Composite vs published series, levels, Pearson, n =
  {{E02_measurement:d_external.composite.level~sffed_nsi.n:int}} months: San Francisco
  Fed index {{E02_measurement:d_external.composite.level~sffed_nsi.pearson:num}}; consumer
  sentiment {{E02_measurement:d_external.composite.level~ics.pearson:num}}; policy
  uncertainty {{E02_measurement:d_external.composite.level~epu.pearson:num}}
- Composite vs published series, first differences: San Francisco Fed index
  {{E02_measurement:d_external.composite.diff~diff_sffed_nsi.pearson:num}}; consumer
  sentiment {{E02_measurement:d_external.composite.diff~diff_ics.pearson:num}}
- Composite vs S&P 500 excess return: same month
  {{E02_measurement:d_external.composite.level~ret_same_month.pearson:num}}; next month
  {{E02_measurement:d_external.composite.level~ret_next_month.pearson:num}}
- Composite vs consumer sentiment change: same month
  {{E02_measurement:d_external.composite.level~ics_change_same_month.pearson:num}}; next
  month {{E02_measurement:d_external.composite.level~ics_change_next_month.pearson:num}}
- Accuracy on the Financial PhraseBank, three classes: classifier five-fold
  {{E02_measurement:e_labelled_data.clf_5fold.accuracy_mean:pct1}}% (macro-F1
  {{E02_measurement:e_labelled_data.clf_5fold.macro_f1_mean:pct1}}%); lexicon
  {{E02_measurement:e_labelled_data.lm.accuracy:pct1}}%; VADER
  {{E02_measurement:e_labelled_data.vader.accuracy:pct1}}%; FinBERT
  {{E02_measurement:e_labelled_data.finbert.accuracy:pct1}}%, in-sample (trained on this
  set); majority class {{E02_measurement:e_labelled_data.majority_class_share:pct1}}%

### 5.2 S&P 500 return (E03, `results/E03_sp500_return.json`; Table 3, Figures 2 and 3)

- Own-lag model vs historical mean: {{r2:E03_sp500_return:results.baselines.M1_vs_M0}}
- Composite vs M0: {{r2:E03_sp500_return:results.composite.M2_vs_M0}}
- Composite added to M1: {{r2:E03_sp500_return:results.composite.M3_vs_M1}}
- Individual scorers vs M0 (p before Bonferroni; times four after):
  - FinBERT {{r2:E03_sp500_return:results.finbert.M2_vs_M0}}
  - classifier {{r2:E03_sp500_return:results.clf.M2_vs_M0}}
  - lexicon {{r2:E03_sp500_return:results.lm.M2_vs_M0}}
  - VADER {{r2:E03_sp500_return:results.vader.M2_vs_M0}}
- GDELT tone vs M0: {{r2:E03_sp500_return:results.gdelt_tone.M2_vs_M0}}
- Outcome under the rule: {{E03_sp500_return:decision.outcome:str}}
- Figure 2, left panel: when the composite gains or loses against each benchmark; data
  in `results/E03_forecasts.csv`

### 5.3 S&P 500 direction (E04, `results/E04_sp500_direction.json`; Table 4)

- Base rate L0: hit rate {{E04_sp500_direction:results.baselines.L0.hit_rate:pct1}}%;
  predicts up in {{E04_sp500_direction:results.baselines.L0.share_up_predicted:pct1}}%
  of test months
- Lagged-return model L1: {{hit:E04_sp500_direction:results.baselines.L1}}
- Composite L2: {{hit:E04_sp500_direction:results.composite.L2}}; Brier
  {{E04_sp500_direction:results.composite.L2.brier:num4}} vs
  {{E04_sp500_direction:results.composite.L2.brier_base:num4}} for L0; Diebold-Mariano
  p = {{E04_sp500_direction:results.composite.L2.dm_brier_p:p}}
- Individual scorers, L2:
  - FinBERT {{hit:E04_sp500_direction:results.finbert.L2}}
  - classifier {{hit:E04_sp500_direction:results.clf.L2}}
  - lexicon {{hit:E04_sp500_direction:results.lm.L2}}
  - VADER {{hit:E04_sp500_direction:results.vader.L2}}
- Fact for the table note: a model that calls the same direction every month has an
  undefined direction test, shown as n/a
- Outcome under the rule: {{E04_sp500_direction:decision.outcome:str}}

### 5.4 Consumer sentiment (E05, `results/E05_ics_change.json`; Table 5, Figures 2 and 3)

- Own-lag model vs historical mean: {{r2:E05_ics_change:results.baselines.M1_vs_M0}}
- Composite vs M0: {{r2:E05_ics_change:results.composite.M2_vs_M0}}
- Composite added to M1: {{r2:E05_ics_change:results.composite.M3_vs_M1}}
- Individual scorers vs M0 (p before Bonferroni):
  - FinBERT {{r2:E05_ics_change:results.finbert.M2_vs_M0}}
  - classifier {{r2:E05_ics_change:results.clf.M2_vs_M0}}
  - lexicon {{r2:E05_ics_change:results.lm.M2_vs_M0}}
  - VADER {{r2:E05_ics_change:results.vader.M2_vs_M0}}
- GDELT tone vs M0: {{r2:E05_ics_change:results.gdelt_tone.M2_vs_M0}}
- Outcome under the rule: {{E05_ics_change:decision.outcome:str}}
- Figure 2, right panel: data in `results/E05_forecasts.csv`

### 5.5 Published indices (E06, `results/E06_published.json`; lower blocks of Tables 3 to 5)

- Long sample: {{E06_published:sample.n_oos:int}} test months,
  {{E06_published:sample.first_oos_target:str}} to
  {{E06_published:sample.last_oos_target:str}}
  - S&P 500 return vs M0: San Francisco Fed index
    {{r2:E06_published:results.long.sp500_return.sffed_nsi.M2_vs_M0}}; policy
    uncertainty {{r2:E06_published:results.long.sp500_return.epu.M2_vs_M0}}; consumer
    sentiment level {{r2:E06_published:results.long.sp500_return.ics_level.M2_vs_M0}}
  - S&P 500 direction, L2, San Francisco Fed index:
    {{hit:E06_published:results.long.sp500_direction.sffed_nsi.L2}}
  - consumer sentiment change, San Francisco Fed index: vs M0
    {{r2:E06_published:results.long.ics_change.sffed_nsi.M2_vs_M0}}; added to M1
    {{r2:E06_published:results.long.ics_change.sffed_nsi.M3_vs_M1}}
  - consumer sentiment change, policy uncertainty added to M1:
    {{r2:E06_published:results.long.ics_change.epu.M3_vs_M1}}
- Same forecasts on the {{E06_published:text_window.n_oos:int}} months of the text sample
  - S&P 500 return vs M0, San Francisco Fed index:
    {{r2:E06_published:results.text_window.sp500_return.sffed_nsi.M2_vs_M0}}
  - consumer sentiment change, San Francisco Fed index: vs M0
    {{r2:E06_published:results.text_window.ics_change.sffed_nsi.M2_vs_M0}}; added to M1
    {{r2:E06_published:results.text_window.ics_change.sffed_nsi.M3_vs_M1}}
- Fact for any side-by-side: these models are estimated on data from 1980; the
  text-index models on data from 2015

### 5.6 Composite added to published measures (E07, `results/E07_incremental.json`; Table 6b)

- S&P 500 return: added to San Francisco Fed index
  {{r2:E07_incremental:results.sp500_return.a_sffed}}; added to all published indices
  {{r2:E07_incremental:results.sp500_return.b_all_published}}
- Consumer sentiment change: added to own lag + San Francisco Fed index
  {{r2:E07_incremental:results.ics_change.a_sffed}}; added to own lag + all published
  indices {{r2:E07_incremental:results.ics_change.b_all_published}}; added to own lag +
  lagged S&P 500 return {{r2:E07_incremental:results.ics_change.c_lagged_sp500_return}}

Terms needing a plain lead-in:
- Hit rate, base rate: share of months with the direction called correctly; base rate =
  what "always the more common direction" achieves.
- Split-half reliability: index built twice, once from each of the day's two files
  (after de-duplication across both), then correlated; Spearman-Brown estimates the
  reliability of the full index.

Questions only you can answer:
- What do you conclude for RQ1, RQ2 and RQ3, and how strongly?
- If outcomes differ between targets, which goes first and why?
- How to treat a scorer significant before Bonferroni and not after?
- How to read a result driven by the 2020 months, if Figure 2 shows that?
- Is the measurement evidence enough to call the index valid, and by what criterion?

---

## 6. Robustness and limitations (1.0 page)

Purpose: how the main numbers move under other reasonable choices; what the study
cannot show. Table 6 here, and Table 7 when E12 has run.

Robustness of the composite (Table 6), then the LLM index (Table 7):
- Estimation scheme (E08, `results/E08_estimation.json`); outcomes in the order return,
  direction, consumer sentiment
  - initial window 36: S&P 500 vs M0
    {{r2:E08_estimation:results.initial_36.sp500_return.composite.M2_vs_M0}}; consumer
    sentiment added to M1
    {{r2:E08_estimation:results.initial_36.ics_change.composite.M3_vs_M1}}; outcomes
    {{E08_estimation:results.initial_36.decisions.sp500_return.outcome:str}},
    {{E08_estimation:results.initial_36.decisions.sp500_direction.outcome:str}},
    {{E08_estimation:results.initial_36.decisions.ics_change.outcome:str}}
  - initial window 60: S&P 500 vs M0
    {{r2:E08_estimation:results.initial_60.sp500_return.composite.M2_vs_M0}}; consumer
    sentiment added to M1
    {{r2:E08_estimation:results.initial_60.ics_change.composite.M3_vs_M1}}; outcomes
    {{E08_estimation:results.initial_60.decisions.sp500_return.outcome:str}},
    {{E08_estimation:results.initial_60.decisions.sp500_direction.outcome:str}},
    {{E08_estimation:results.initial_60.decisions.ics_change.outcome:str}}
  - rolling 48: S&P 500 vs M0
    {{r2:E08_estimation:results.rolling_48.sp500_return.composite.M2_vs_M0}}; consumer
    sentiment added to M1
    {{r2:E08_estimation:results.rolling_48.ics_change.composite.M3_vs_M1}}; outcomes
    {{E08_estimation:results.rolling_48.decisions.sp500_return.outcome:str}},
    {{E08_estimation:results.rolling_48.decisions.sp500_direction.outcome:str}},
    {{E08_estimation:results.rolling_48.decisions.ics_change.outcome:str}}
- Index definition (E09, `results/E09_index_definition.json`)
  - first difference: S&P 500 vs M0
    {{r2:E09_index_definition:results.a_first_difference.sp500_return.composite.M2_vs_M0}};
    consumer sentiment added to M1
    {{r2:E09_index_definition:results.a_first_difference.ics_change.composite.M3_vs_M1}}
  - headline-weighted: S&P 500 vs M0
    {{r2:E09_index_definition:results.b_headline_weighted.sp500_return.composite.M2_vs_M0}};
    consumer sentiment added to M1
    {{r2:E09_index_definition:results.b_headline_weighted.ics_change.composite.M3_vs_M1}}
  - page-title index, months {{E09_index_definition:title_months.0:str}} to
    {{E09_index_definition:title_months.1:str}}, initial window 36: S&P 500 vs M0
    {{r2:E09_index_definition:results.c_title_index.sp500_return.composite.M2_vs_M0}};
    consumer sentiment added to M1
    {{r2:E09_index_definition:results.c_title_index.ics_change.composite.M3_vs_M1}}
  - URL-headline index, same months: S&P 500 vs M0
    {{r2:E09_index_definition:results.d_url_index_same_months.sp500_return.composite.M2_vs_M0}};
    consumer sentiment added to M1
    {{r2:E09_index_definition:results.d_url_index_same_months.ics_change.composite.M3_vs_M1}}
- Target definition (E10, `results/E10_target_definition.json`)
  - raw log return vs M0:
    {{r2:E10_target_definition:results.a_raw_log_return.results.composite.M2_vs_M0}}
  - simple excess return vs M0:
    {{r2:E10_target_definition:results.b_simple_excess_return.results.composite.M2_vs_M0}}
  - consumer sentiment, own lag one month older, composite added to M1:
    {{r2:E10_target_definition:results.c_ics_own_lag_release_lag_1.results.composite.M3_vs_M1}};
    design of this variant: the own lag and the estimation targets are one month
    older (survey value of month m unknown at the forecast date); the text window
    still ends the day before the forecast date
- Three-month horizon (E11, `results/E11_horizon3.json`),
  {{E11_horizon3:sample.n_oos:int}} overlapping test months: S&P 500 vs M0
  {{r2:E11_horizon3:results.sp500_return_3m.results.composite.M2_vs_M0}}, outcome
  {{E11_horizon3:results.sp500_return_3m.decision.outcome:str}}; consumer sentiment
  added to M1 {{r2:E11_horizon3:results.ics_change_3m.results.composite.M3_vs_M1}},
  outcome {{E11_horizon3:results.ics_change_3m.decision.outcome:str}}
- Power of the main test (`results/power.json`; not pre-specified,
  2026-10-07): simulation with the composite at its observed values,
  {{power:config.reps:int}} samples per value of the true R-squared; power is the share
  of samples in which M2 against M0 has R2_OS above zero and a Clark-West p-value below
  0.05
  - S&P 500 return: {{power:results.sp500_return.at_zero.power:pct1}}% at a true
    R-squared of zero, {{power:results.sp500_return.selected.r2_1pct.power:pct1}}% at 1%,
    {{power:results.sp500_return.selected.r2_2pct.power:pct1}}% at 2%,
    {{power:results.sp500_return.selected.r2_5pct.power:pct1}}% at 5%; 50% power from
    {{power:results.sp500_return.smallest_r2.power_50:pct1}}%, 80% power from
    {{power:results.sp500_return.smallest_r2.power_80:pct1}}%
  - consumer sentiment: {{power:results.ics_change.at_zero.power:pct1}}% at zero,
    {{power:results.ics_change.selected.r2_1pct.power:pct1}}% at 1%,
    {{power:results.ics_change.selected.r2_5pct.power:pct1}}% at 5%; 80% power from
    {{power:results.ics_change.smallest_r2.power_80:pct1}}%
<!-- if E12_llm:status == ran -->
- LLM scorers (E12, `results/E12_llm.json`; Table 7): the `llm` index in the models of
  E03 to E05; outside the composite and the decision rule, so no outcome label;
  p-values not adjusted for multiple comparisons
  - {{E12_llm:usage.model:str}}; index in {{E12_llm:index_months_with_llm_index:int}} of
    {{E12_llm:sample.n_origins:int}} index months
    - all test months: S&P 500 vs M0 {{r2:E12_llm:results.sp500_return.llm.M2_vs_M0}};
      direction {{hit:E12_llm:results.sp500_direction.llm.L2}}; consumer sentiment vs M0
      {{r2:E12_llm:results.ics_change.llm.M2_vs_M0}}, added to M1
      {{r2:E12_llm:results.ics_change.llm.M3_vs_M1}}
    - test months whose whole text window lies after the training cutoff
      ({{E12_llm:by_training_cutoff.training_cutoff:str}}; origins from
      {{E12_llm:by_training_cutoff.post_cutoff.first_origin:str}};
      {{E12_llm:by_training_cutoff.post_cutoff.n_test_origins:int}} months): S&P 500 vs M0
      {{r2:E12_llm:by_training_cutoff.post_cutoff.results.sp500_return.llm.M2_vs_M0}};
      direction
      {{hit:E12_llm:by_training_cutoff.post_cutoff.results.sp500_direction.llm.L2}};
      consumer sentiment added to M1
      {{r2:E12_llm:by_training_cutoff.post_cutoff.results.ics_change.llm.M3_vs_M1}}
    - the other {{E12_llm:by_training_cutoff.pre_cutoff.n_test_origins:int}} test months:
      S&P 500 vs M0
      {{r2:E12_llm:by_training_cutoff.pre_cutoff.results.sp500_return.llm.M2_vs_M0}};
      consumer sentiment added to M1
      {{r2:E12_llm:by_training_cutoff.pre_cutoff.results.ics_change.llm.M3_vs_M1}}
  - {{E12_llm:second_model.usage.model:str}}; index in
    {{E12_llm:second_model.index_months_with_llm_index:int}} of
    {{E12_llm:second_model.sample.n_origins:int}} index months
    - all test months: S&P 500 vs M0
      {{r2:E12_llm:second_model.results.sp500_return.llm.M2_vs_M0}}; direction
      {{hit:E12_llm:second_model.results.sp500_direction.llm.L2}}; consumer sentiment vs
      M0 {{r2:E12_llm:second_model.results.ics_change.llm.M2_vs_M0}}, added to M1
      {{r2:E12_llm:second_model.results.ics_change.llm.M3_vs_M1}}
    - test months whose whole text window lies after its cutoff month
      ({{E12_llm:second_model.by_training_cutoff.training_cutoff:str}}):
      {{E12_llm:second_model.by_training_cutoff.post_cutoff.n_test_origins:int}}; the
      code compares only 12 or more forecasts, so S&P 500 vs M0 reads
      {{r2:E12_llm:second_model.by_training_cutoff.post_cutoff.results.sp500_return.llm.M2_vs_M0}}
  - design facts: the split restricts evaluation only, and the models are still
    estimated on all earlier months, whose index values include months before the
    cutoff; a test month whose text window starts inside the cutoff month is counted
    with the other months; 12 forecasts is the smallest number the code compares
<!-- else -->
- LLM scorer (E12, `results/E12_llm.json`): {{E12_llm:status:str}}; specified and not run
<!-- end -->

Limitations, as facts:
- {{E03_sp500_return:sample.n_oos:int}} test months, including March and April 2020
- one country, one stock index, one survey series
- headlines from URLs, not titles; agreement with titles in section 5.1
- two files a day cover a small part of each day's news; reliability in section 5.1
- source mix: {{E01_corpus:source_mix.share_of_year_b_headlines_from_domains_seen_in_year_a:pct1}}%
  of {{E01_corpus:source_mix.year_b:str}} headlines come from domains that also appear in
  {{E01_corpus:source_mix.year_a:str}};
  {{E01_corpus:source_mix.top20_domains_in_common:int}} of the 20 largest domains are the
  same in both years (E01)
- gaps in the GDELT archive: {{E01_corpus:files.days_with_no_file:int}} days with no
  file and {{E01_corpus:files.days_with_one_file:int}} with one (section 3); the index
  month with the fewest valid days,
  {{E01_corpus:valid_days_per_index_month.month_of_min:str}}, has
  {{E01_corpus:valid_days_per_index_month.min:int}} (the rule asks for 15)
- keyword filter fixed by hand, not validated against human labels; the list was written
  in 2026, after the sample period, and holds tariff terms, for example
- S&P 500 series is a price index without dividends
- consumer sentiment release dates not machine-verified (`docs/data.md` section 3)
- San Francisco Fed index, policy uncertainty index and Loughran-McDonald word lists
  used as downloaded on 2026-10-06; revisions of their history unverified
- several scorers and targets tested; one primary predictor fixed in advance; Bonferroni
  adjustment for the four individual scorers only; p-values of GDELT tone, the published
  indices and E07 to E12 are unadjusted
<!-- if E12_llm:status == ran -->
- E12: a request holds up to 40 headlines in corpus order and can mix headlines dated
  on both sides of a forecast date; in the second run the requests mixed new headlines
  from nine separate months with re-sent ones, so some span months or years; how many
  labels were produced next to later-dated headlines is not recorded in `results/`;
  every other scorer scores each headline on its own
- E12: the training cutoffs are the provider's statements and are not checked here; the
  test months whose whole text window lies after the cutoff month number
  {{E12_llm:by_training_cutoff.post_cutoff.n_test_origins:int}} for the first model and
  {{E12_llm:second_model.by_training_cutoff.post_cutoff.n_test_origins:int}} for the
  second
- E12 needs a paid API and hosted models; E01 to E11 use free data only
<!-- end -->

Questions only you can answer:
- Which robustness results change your reading of section 5, if any?
- Which limitations matter most for the claim you make?
- What future work do you propose?

---

## 7. Conclusion (0.4 page)

Purpose: answer the introduction's questions in a few sentences.

Facts available to restate (no new numbers):
- outcome label per target: S&P 500 return {{E03_sp500_return:decision.outcome:str}};
  S&P 500 direction {{E04_sp500_direction:decision.outcome:str}}; consumer sentiment
  {{E05_ics_change:decision.outcome:str}}
- measurement findings, section 5.1
- published-index comparison, section 5.5
- E01 to E11 rerun from free data with one command; E12 also needs an API key

Questions only you can answer:
- What is the one-sentence answer to the paper's question?
- What should a practitioner or a researcher take from it?
- What comes next?

---

## Declaration on Generative AI (0.1 page)

Above the references. Wording is yours.

---

## References (1.0 page)

`paper/references.bib`: 25 entries, each checked at the original.
The draft in `paper/main.tex` cites all 25. Keys by group: section 2 above.
