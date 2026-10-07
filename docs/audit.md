# PEBSA audit

Audited file: `legacy/pebsa_ijecs_2024.pdf` (22 pages, numbered 26656-26677).
Nagarajan and Mittal, "PEBSA: Predicting Economic Behavior via Sentiment Analysis",
IJECS 13(12), December 2024, https://doi.org/10.18535/ijecs/v13i12.4950.
Crossref lists the pages as 26656-26676; the PDF runs to 26677.

Method: all 22 pages read, including the five code screenshots, three charts and three
figures. Arithmetic checked by `src/sentcast/audit.py`, output in
`results/audit_checks.json`. Cited sources checked at the original on 2026-10-06
(Crossref and OpenAlex records, arXiv, PubMed Central full text, publisher and agency
pages). "Not located" means a search did not find the item. "Not opened" means the item
exists but its content was not checked.

## 1. Claims, methods, datasets, results

| # | Claim in the paper | Where | Supported by the paper? |
|---|---|---|---|
| 1 | Posts and comments from X and Facebook were collected with Hootsuite and exported to Excel | 1, 3.2, 3.4 | Stated only. No query, schema, per-platform counts, date range, or sample. No data survives. |
| 2 | "Scientific sampling techniques" ensured unbiased data | Abstract, 1 | No. No sampling design is described anywhere. |
| 3 | Dataset size | 3.1, 3.2 vs 3.4, 4.2 | Inconsistent. "over 147,000 unique data points" twice, "over 88,000 data points" twice. Not reconciled. |
| 4 | The data represent "a significant majority of the American public" | 3.4 | No. 88,000 posts are not a majority of the public. 6.2 says social media users are about 68% of the population. |
| 5 | Collection period | 3.4, Chart 1, 6.4 | Inconsistent. "since January 2024"; Chart 1 says January 1 to April 30, 2024; 6.4 says training and input data overlapped from November 2023 to January 2024. |
| 6 | The model combines sentiment analysis, neural networks and random forests | 3.1, 4.2 | No. The code shown contains neither a neural network nor a random forest (section 2.4 below). |
| 7 | The model was trained on Google Cloud Vertex AI | 3.1, 3.2, 4.2 | Contradicted. 6.1 says training ran on an i5-13400F, 16 GB RAM, RTX 4060. The code is ML.NET, a local library. |
| 8 | Comments were classified as positive or negative | 3.1, 3.2 | Partly. Code 1 trains a binary classifier on a labelled column whose origin is not described. No accuracy, AUC or F1 value is reported, although Code 4 prints them. |
| 9 | "Whiteboard macroeconomic reasoning" turns sentiment into a forecast | 1 | No. No formula, model or procedure is given (section 2.2 below). |
| 10 | Predicted S&P 500 monthly changes for January to April 2024: +1.76, +5.11, +3.19, -4.20 percent | Chart 1 | Stated only. Four numbers, not reproducible, no statement of when each prediction was made. |
| 11 | Accuracy of 97.53% | 3.2, 4.3 | The arithmetic reproduces. It is 100 minus a mean relative error over four months, not a classification accuracy (section 2.1). |
| 12 | The model beats professional forecasters, whose accuracy is "around 23%" | 4.3 | No. The 23% is a calibration statistic from a different setting (section 2.6, Campbell and Moore). |
| 13 | A "meta-analysis" found 1.3% error in normal markets and 8.6% in recessions | 3.2, 4.3 | No. Both numbers come from another paper's model (section 2.3). |
| 14 | Chart 2, Chart 3, Figure 2, Figure 3 | 4.3, 6 | Third-party images. Charts 2-3 are from Malladi (2022); Figures 2-3 are from Smalter Hall (2018), which is not in the reference list. |
| 15 | Machine learning outperforms traditional econometric models and professional forecasters | 2.2, 2.5, 2.6 | Rests on citations that mostly do not say this (section 2.6). |

The abstract is written in the future tense ("will be applied") and describes a plan, not
results.

## 2. Specific checks

### 2.1 The 97.53% figure

Chart 1 lists four months. The "% Error" column is the absolute gap between the predicted
and actual percent change, divided by the absolute actual percent change:

| Month | S&P 500 (printed) | PEBSA (printed) | Error (printed) | Error (recomputed) |
|---|---|---|---|---|
| Jan 2024 | +1.7 | +1.76 | 3.53 | 3.5294 |
| Feb 2024 | +5.3 | +5.11 | 3.59 | 3.5849 |
| Mar 2024 | +3.2 | +3.19 | 0.31 | 0.3125 |
| Apr 2024 | -4.1 | -4.20 | 2.44 | 2.4390 |

Mean of the printed errors: 2.4675. The paper's 97.5325% is 100 - 2.4675. Number of
observations: 4.

Consequences:
- The statistic is a mean absolute percentage error on returns. It is undefined when the
  actual return is zero and grows without bound as the return approaches zero.
- The absolute gaps are 0.06, 0.19, 0.01 and 0.10 percentage points. For scale, the mean
  absolute monthly price return of the S&P 500 over 1950 to 2023 is 3.30 percent (888
  months, `results/audit_checks.json`). The paper gives no procedure that could produce
  forecasts this close.
- No baseline is reported (historical mean, previous month, zero).
- The printed S&P 500 figures match the total return index rounded to one decimal
  (1.6804, 5.3396, 3.2175, -4.0845), not the price index (1.5896, 5.1721, 3.1019,
  -4.1615). Sources: `^SP500TR` sample for total return, Welch-Goyal monthly file for
  price, both in `results/audit_checks.json`. The paper does not say which it used.

### 2.2 How sentiment becomes a predicted percent change

Not described. The only output of the code shown is a count of comments labelled
"Optimistic" and "Pessimistic", drawn as a bar chart (Code 5, where `data1` and `data2`
are undefined). The text refers to "whiteboard macroeconomic reasoning" and to
"condensing the output data ... into a format suitable for economic reasoning". No
mapping from two counts to a percent change with two decimals appears anywhere.

### 2.3 The 1.3% and 8.6% error rates

Both are false discovery rates from Table 2 of Malladi (2022), reproduced as an image in
Chart 2 ("Created by California State University"). In that paper a linear support vector
machine classifies each month as recession or normal three months ahead, using 134
FRED-MD macroeconomic variables for 1959 to 2020. The full text states a false discovery
rate of 8.6% for recession and 2.9% for crash, a positive predictive value of 91.4% for
recession months (85 of 93) and 98.7% for normal months (626 of 634). 100 - 98.7 = 1.3.

The numbers therefore describe another author's model, on macroeconomic data, in its
training and validation sample. They are not PEBSA results, and no meta-analysis was
carried out. Malladi (2022) uses no social media or sentiment data.

### 2.4 Which model the code implements

| Screenshot | What it does |
|---|---|
| Code 1 | ML.NET pipeline: `FeaturizeText` on a text column, then `BinaryClassification.Trainers.SdcaLogisticRegression`. This is a logistic regression on bag-of-words style features. Reads `path_to_data.csv`, a placeholder. |
| Code 2 | Loads the first sheet of an Excel file with ExcelDataReader. |
| Code 3 | Sets `Sentiment = Opinion.Contains("positive") ? "Positive" : "Negative"`. A substring rule, not a trained model. It also calls `LoadFromTextFile` with a comma separator on an `.xlsx` file. |
| Code 4 | Loads `model.zip` and prints accuracy, AUC and F1. No values are reported in the paper. |
| Code 5 | Windows Forms bar chart of two counts. |

There is no neural network, no random forest, no Vertex AI call, no S&P 500 data, and no
forecasting step in the code shown.

### 2.5 Dataset size

"Over 147,000 unique data points" (3.1, 3.2) and "over 88,000 data points" (3.4, 4.2).
The paper does not say whether these are different sets. No per-platform, per-month or
per-label counts are given. The unit ("data point") is not defined.

### 2.6 Whether each cited source says what it is cited for

Verdicts: Supports; Partly (real source, claim goes beyond it); No (does not say it);
Not located; Not opened.

| # | Reference as listed | Cited for | What the original says | Verdict |
|---|---|---|---|---|
| 1 | Abbasi (2021), Medium | PEBSA's own algorithm integration and choice of C# (4.2) | Not found by search. A tutorial cannot support claims about PEBSA's own model. | Not located |
| 2 | Agrawal, Gans, Goldfarb (2018), Prediction Machines | Policymakers using ML; scalability (2.5, 2.6) | Book exists. | Not opened |
| 3 | Athey (2018), NBER | ML improves economic forecasts (2.2); public opinion shapes economic outcomes (4.4) | Single-author chapter on ML and economics: prediction policy problems, causal inference, future impact. Nothing on public opinion or sentiment. Cited in text as "Athley et al." | Partly |
| 4 | Bishop (2006) | Data augmentation, stratified sampling, cleaning (2.6) | Textbook. | Not opened |
| 5 | Bollen, Mao, Zeng (2011) | Sentiment influences markets; validation against the S&P 500 (2.3-2.5) | Twitter mood and the Dow Jones Industrial Average, not the S&P 500. Reports 87.6% daily direction accuracy. | Supports (general claim) |
| 6 | Breiman (2001) | Random forests suit economic forecasting (2.2) | Introduces random forests. No economics. | Partly |
| 7 | Bureau of Economic Analysis (2024) | PEBSA is "a highly accurate predictive model" (4.2) | A statistics page cannot support a claim about PEBSA. No URL given. | No |
| 8 | Campbell and Moore (2024) | ML models cut forecast errors "by up to 30%" versus econometric models (2.5) | Sandy Campbell and Don A. Moore, "Overprecision in the Survey of Professional Forecasters", Collabra: Psychology 10(1). Forecasters report 53% confidence and are correct 23% of the time (n = 16,559). No ML comparison. Initials in the reference are wrong. Source of the "23%" in 4.3. | No |
| 9 | Choi and Varian (2012) | Google Trends improves predictions "when incorporated into ML models" (2.2) | Search data help forecast near-term indicators, using simple autoregressive models. | Partly |
| 10 | Costola et al. (2023) | Policymakers can anticipate market shifts (4.4) | BERT-based sentiment of 203,886 COVID-19 news articles, January to June 2020; positive, significant relation with S&P 500 returns. | Partly |
| 11 | Coulombe et al. (2019), Review of Economic Dynamics 31, 156-168 | ML beats time-series models and professional forecasters; neural networks and random forests predict downturns (2.2); PEBSA's own accuracy (4.3) | Goulet Coulombe, Leroux, Stevanovic, Surprenant, "How is machine learning useful for macroeconomic forecasting?", Journal of Applied Econometrics 37(5), 920-964 (2022). Finds nonlinearity is the main source of ML gains. No comparison with professional forecasters. The DOI given (10.1016/j.red.2018.09.009) does not resolve; journal, volume and pages are wrong. | Partly |
| 12 | Desai (2023), Bank of Canada | Automation improves economic management (4.4) | Staff Analytical Note 2023-16: a review of ML tools for economics research. | Partly |
| 13 | Foy et al. (2023), MIT News | Better hardware raises "training efficiency and model performance by over 80%" (6.1); PEBSA's accuracy (7) | Kylie Foy, "New tools are available to help reduce the energy that AI models devour", October 5, 2023. Early stopping cut training energy by 80%; power capping cut energy by 12-15%. Nothing on hardware upgrades raising performance. | No |
| 14 | Goodfellow, Bengio, Courville (2016) | Pruning, quantization, distributed training (2.6) | Textbook. | Not opened |
| 15 | Gottfried, Barthel, Mitchell (2024), Pew | A study that applied NLP, CNNs and RNNs to Twitter and Facebook posts to forecast economic trends (2.3) | Jeffrey Gottfried (sole author), "Americans' Social Media Use", January 31, 2024. A survey of 5,733 adults on platform use. No sentiment analysis, no forecasting. Title and coauthors as listed do not match. It reports that 68% of U.S. adults use Facebook, which matches the 68% in 6.2. | No |
| 16 | Han et al. (2015) | Pruning cuts computation without losing accuracy (2.5) | Parameters reduced 9x (AlexNet) and 13x (VGG-16) with no accuracy loss. | Supports |
| 17 | Hastie, Tibshirani, Friedman (2009) | Overfitting and regularization (2.6) | Textbook. | Not opened |
| 18 | Hochreiter and Schmidhuber (1997) | LSTMs for economic time series (2.5) | Introduces LSTM. No economics. | Partly |
| 19 | Kapoor (2019), IT Exchange | Overfitting needs new datasets (6.4) | Not found by search. | Not located |
| 20 | Kumar (2024), SciSpace | Geographic limits reduce applicability (6.3); ML sets a precedent (7) | Not found by search. | Not located |
| 21 | Liu (2012) | Sentiment analysis methods (2.3) | Book. | Not opened |
| 22 | Malladi, Nanduri, Agrawal (2022) | Models trained on "economic indicators and sentiment data from social media", accuracy over 90% (2.3); PEBSA's "meta-analysis" (4.3) | Rama K. Malladi (sole author), Computational Economics 63(3), 1021-1045. Uses 134 FRED-MD variables. No social media or sentiment data. Best model validation accuracy 97.8%. Coauthors as listed are not on the paper. | No |
| 23 | Microsoft (2020), ML.NET documentation | ML.NET capabilities (2.3) | Product documentation. | Not opened |
| 24 | Mullainathan and Spiess (2017) | "Stanford" researchers showed ML improves forecasts of economic indicators (2.2); ML beats econometric models in the short term (2.5) | A methodological overview of ML as a prediction tool in econometrics. Reports no forecasting comparison of that kind. | Partly |
| 25 | Northumbria University (2024, February 2) | Social media users are about 68% of Americans, skewed young (6.2); insight into "complex economic systems" (7) | Press release dated 27 November 2023 on platform API restrictions threatening research. Contains neither statement. | No |
| 26 | Olteanu et al. (2019) | Bias in social data (2.6) | Survey of biases and pitfalls in social data. | Supports |
| 27 | Pang and Lee (2008) | Opinion mining background (2.3) | Survey of opinion mining and sentiment analysis. | Supports |
| 28 | Sasaki (2007) | F-measure definition (2.4) | Search did not return the report. | Not opened |
| 29 | Scikit-learn (2019) | Dimensionality reduction; "more actionable economic predictions" (5) | Software site. | Not opened |
| 30 | Silver et al. (2016), ICML 32 | Reinforcement learning improves adaptive economic forecasting (2.5) | "Deterministic Policy Gradient Algorithms", 31st ICML, 2014, PMLR 32:387-395. No economics. Year is wrong. | No |
| 31 | Smolic (2022), Graphite Note | Larger datasets improve accuracy (4.2) | Blog post on how much data ML needs; rule of thumb of ten samples per feature. | Supports (blog) |
| 32 | Shtar and Margel (2017), Imperva | Clustering and dimensionality reduction (5) | Blog post applying both to file-access security. | Supports (blog) |
| 33 | Stock and Watson (2001) | Traditional models are linear and weak in volatile periods (2.5) | Review of vector autoregressions: reliable for description and forecasting, less so for structural inference. | Partly |
| 34 | Strubell, Ganesh, McCallum (2019) | Never cited in the text | Energy cost of training NLP models. | Uncited |
| 35 | Varian (2014) | Hybrid models give more robust forecasts (2.2, 2.5, 2.6) | Overview of big-data tools for econometrics. The DOI as listed is truncated (10.1257/jep.28.2.3). | Partly |

Totals: Supports 6, Partly 10, No 7, Not located 3, Not opened 8, Uncited 1.

In-text citations with no reference-list entry: Brownlee (4.2), Stanford (4.4), EMB (4.4),
National Bureau of Economic Research (5), Google Cloud 2021 (2.3), Stock and Watson 2018
(2.2), and "Hall et al. 2018" in the captions of Figures 2 and 3. The last is Aaron
Smalter Hall (sole author), "Machine Learning Approaches to Macroeconomic Forecasting",
Federal Reserve Bank of Kansas City Economic Review, Q4 2018, 63-81.

## 3. What carries forward

Hypothesis: sentiment expressed in public text carries information about future market
and economic outcomes.

Targets: the S&P 500 monthly change, which PEBSA used for validation, and one
real-economy series standing in for "the American economy's behavior".

Limitations PEBSA states about itself, and the SentCast response:

| PEBSA limitation (section) | SentCast response |
|---|---|
| Training and input data overlapped (6.4) | Expanding-window out-of-sample forecasts; tests that fail on any text dated on or after its forecast date |
| Social media users skew young and are not the whole public (6.2) | News text with documented sources; the corpus and its filters are described and reproducible |
| US only (6.3) | Kept, and stated as scope |
| Hardware limits (6.1) | Small models that run on one laptop GPU |

Future work PEBSA proposes (section 5), and what SentCast does with it:

| PEBSA future work | SentCast |
|---|---|
| Test on a 17-year dataset, 2006 to 2023 | As much history as free sources give: 2015 to 2026 for text, 1980 to 2026 for a published index |
| Compare models, including regression | Regression forecasts against two baselines, with four scorers |
| Better data collection | Seeded, documented sampling of a public corpus |
| Python tooling | Python pipeline, one command |

Nothing else carries forward. SentCast reuses no PEBSA text, number, figure, code or data,
and cites PEBSA in the third person as motivating prior work.
