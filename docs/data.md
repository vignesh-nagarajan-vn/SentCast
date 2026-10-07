# Data access check

Run on 2026-10-06 with `python -m sentcast.datacheck`. Samples are in `data/samples/`
(gitignored). The machine-readable record is `results/data_check.json`. Every number
below comes from that file unless marked otherwise. Terms of use were read on the
source's own pages on the same day.

Requirements: free, re-downloadable by script, no paid key, long history.

## 1. Candidates

### S&P 500

| Source | Result | Coverage | Access | Terms |
|---|---|---|---|---|
| Welch-Goyal predictor file (A. Goyal's website), column `Index` | OK, 1,860 monthly rows | 1871-01 to 2025-12, month-end level | Google Sheets XLSX export, no key | No terms stated on the page; standard academic dataset, cite Welch and Goyal (2008) |
| FRED `SP500` | OK, 2,512 daily rows | 2016-10-06 to 2026-10-05 | CSV endpoint, no key | Copyright S&P Dow Jones Indices; FRED carries a rolling 10 years; FRED terms allow personal, non-commercial, educational use and forbid disruptive scraping |
| Yahoo Finance `^GSPC` | OK, 24,808 daily rows | 1927-12-30 to 2026-10-06 | `yfinance`, unofficial API, no key | Yahoo terms (section 2.4(i)) forbid collecting data by automated means without permission |
| Yahoo Finance `^SP500TR` | OK, 9,763 daily rows | 1988-01-04 to 2026-10-06 | same | same |
| Stooq `^SPX` | Fails, HTTP 404 | n/a | The CSV endpoint serves no data to a script (a browser gets a JavaScript verification page; exploratory observation) | n/a |

Cross-checks:
- Yahoo and FRED daily closes: 2,512 overlapping days; 2 days differ by more than 0.01%,
  largest gap 0.12%.
- Welch-Goyal and FRED month-end levels: 110 overlapping months (2016-11 to 2025-12);
  one month differs, by 2.37%. The month is 2017-11, where the Welch-Goyal value is
  2584.84 and FRED and Yahoo both give 2647.58 (identified in an exploratory comparison,
  not stored in `data_check.json`).
- Welch-Goyal and Yahoo month-end levels since 1950: 912 months; 11 differ by more than
  0.01%, 9 by more than 0.1%.

Chosen: FRED `SP500` month-end closes for every month FRED covers in full, and the
Welch-Goyal `Index` column before that. Yahoo is not used by the pipeline because of its
terms; its samples were downloaded once for this check.

Trading dates come from FRED `NASDAQCOM` (daily, 1971-02-05 to 2026-10-05, 14,034 rows),
used only as a calendar. The risk-free rate is FRED `TB3MS` (1934-01 to 2026-09, 1,113
monthly rows).

Caveat: FRED's S&P 500 window rolls forward. A download made after late 2027 will no
longer cover 2017-11, and the Welch-Goyal value for that month would then be used unless
its file has been corrected.

### Consumer sentiment

| Source | Result | Coverage | Access | Terms |
|---|---|---|---|---|
| University of Michigan `tbmics.csv` | OK, 677 rows, 585 since 1978 | November 1952 to September 2026; monthly from 1978 | direct CSV, no key | Copyright the Regents of the University of Michigan; cite the Surveys of Consumers |
| FRED `UMCSENT` | OK, 676 rows | 1952-11 to 2026-08 | CSV endpoint, no key | "Copyrighted: citation required". FRED delays the series by one month at the source's request. |

Cross-check: the two agree exactly on all 584 overlapping months since 1978. The
University of Michigan file has one more month (September 2026, 48.1). An ALFRED vintage
dated 2015-06-30 matches today's values in all 450 months it covers, so published final
values were not revised over that span.

Chosen: the University of Michigan file, because it is not delayed. FRED is the fallback.

FRED quirk (exploratory observation, not in `data_check.json`): the CSV endpoint reset
connections that sent a browser user agent and answered a plain one in about 0.1 s. The
pipeline sends `SentCast/0.1` to every source.

### News headlines: GDELT

| Route | Result | Coverage | Access | Notes |
|---|---|---|---|---|
| GKG 2.0 15-minute files | OK. Four sample files, 5.9 to 8.3 MB zipped, 1,341 to 2,151 articles each, under 1 s each | 2015-02-19 to present, 96 files per day | direct ZIP, no key, no rate limit | Page titles are absent in the 2015-02-19 and 2019-07-01 samples and present in every row of the 2019-10-01 and 2026-09-30 samples. Every row has the article URL, themes, locations and GDELT's tone score. |
| DOC 2.0 API | Intermittent. The recorded run got HTTP 429 on all three attempts, 20 s apart. Earlier runs the same day returned 250 articles for one day, once at the first attempt and once at the second. | 2017-01-01 to present | JSON API, no key | At most 250 articles per call and one call per 5 seconds. A daily sample of the full period needs about 3,560 calls. Results can change when GDELT reindexes. |

Terms: GDELT data are free for any academic, commercial or governmental use and may be
redistributed, with a citation to the GDELT Project and a link to
https://www.gdeltproject.org/.

Throughput (exploratory benchmark, not in `data_check.json`): 32 randomly chosen GKG
files downloaded and unzipped in 6.1 s on 8 threads, 251 MB in total.

### Reddit (optional)

Arctic Shift archive API, `r/economics`, three one-day queries: all three succeeded in
the recorded run (100, 55 and 40 posts for 2008-09-15, 2019-03-04 and 2026-09-01). In
earlier runs the same day one or two of the three returned "Timeout. Maybe slow down a
bit". The archive is unofficial and Reddit's data terms were not reviewed. Not used.

### Published benchmarks

| Source | Result | Coverage | Access | Terms |
|---|---|---|---|---|
| San Francisco Fed Daily News Sentiment Index | OK, 17,060 daily rows | 1980-01-01 to 2026-10-04, updated weekly | direct XLSX, no key | No terms stated on the page; cite Shapiro, Sudhof and Wilson, and Buckman et al. |
| Economic Policy Uncertainty, news-based US index | OK, 1,521 monthly rows | 1900-01 to 2026-09 | direct XLSX, no key | CC BY 4.0; the file asks for attribution to Baker, Bloom and Davis, the paper and the website |
| AAII Investor Sentiment Survey | Refused, HTTP 403 | n/a | The spreadsheet is served to browsers only | Not used. The site says the full history is for members. |

### Scorer inputs

| Source | Result | Access | Terms |
|---|---|---|---|
| Loughran-McDonald master dictionary 1993-2025 | OK, 86,553 words, 2,345 negative, 347 positive | Google Drive direct link from the authors' page | Free for academic research; commercial use needs a licence |
| Financial PhraseBank v1.0 | OK, 4,846 labelled sentences (50% agreement file) | Hugging Face direct file | CC BY-NC-SA 3.0 |
| FinBERT `ProsusAI/finbert` | OK, commit 4556d13 | Hugging Face Hub | No licence field on the model card; weights are PyTorch pickle files, loaded with `weights_only` |

### Considered and not used

| Source | Why not |
|---|---|
| Wikipedia Current Events portal | Reachable through the MediaWiki API (2 items on 2006-05-10, 15 on 2019-08-14). Too few items per day, and most are not about the US economy. |
| New York Times Archive API, Guardian Open Platform | Need a registered key, which needs an account. Not tested. |
| FNSPID (Hugging Face) | 5.7 GB of firm-level headlines ending in 2023. Not tested beyond the file listing. |
| Shiller monthly data | Monthly averages of daily prices, not month-end closes. |

## 2. Choice of text source

Primary: GDELT GKG 2.0 files.
- Bulk files are immutable and addressed by timestamp, so a seeded sample can be drawn
  again exactly.
- No rate limit and no key.
- Coverage starts in February 2015, two years before the DOC API.
- Terms allow research use and redistribution with citation.
- Each row carries GDELT's own tone score, which gives one more benchmark at no cost.

Fallback: GDELT DOC 2.0 API article lists. It returns real headlines and a source-country
filter from 2017, but it is rate limited, capped at 250 articles per call, and not exactly
repeatable. It is used only if the bulk files become unreachable.

Consequences of the primary choice:
- Headline text. Page titles exist only from the third quarter of 2019. To keep one
  definition for the whole sample, the headline is rebuilt from the words in the article
  URL. Titles are used to validate this where both exist.
- Volume. The full archive is about 96 files of 5 to 12 MB per day. The plan draws two
  files per day by a seeded rule, about 8,500 files and roughly 67 GB of transfer.
- History. 2015-03 to 2026-08 gives 138 monthly forecasts. The San Francisco Fed index
  and the Economic Policy Uncertainty index supply a 1980 to 2026 benchmark for the same
  targets.

## 3. Release timing

| Series | Timing | Handling |
|---|---|---|
| Economic Policy Uncertainty | Monthly, published early in the following month (not machine-verified) | Lagged one month |
| S&P 500 month-end close | Known at the close | The last trading day of month m is the forecast date for month m+1 |
| Trading calendar | FRED `NASDAQCOM` observation dates | Defines the last trading day of each month |
| University of Michigan index, final | Released in the last week of the month or the first days of the next (not machine-verified; no free archive of release dates was found) | As the target's own lag: forecasts are dated at the final release. As a predictor of the S&P 500: lagged one month. |
| TB3MS | Monthly average of daily rates | Defines the excess return only; not a predictor |
| San Francisco Fed index | Weekly updates | Daily values from the 7 days before the forecast date are excluded |
| GKG article timestamp | Time GDELT processed the article, at or after publication | Text is used only if its UTC timestamp is before the forecast date |

## 4. Notes from the full download

The corpus step ran on 2026-10-06. Counts are in `results/E01_corpus.json` under `files`
and in Table 1. The points below are observations from that run. Unless a results file
is named, they are not stored anywhere else.

- Missing files are gaps in the archive. Probes of drawn timestamps in 2015-10, 2020-10,
  2020-11, 2021-08 and 2025-06 returned HTTP 404 on both `data.gdeltproject.org` and
  `storage.googleapis.com/data.gdeltproject.org`.
- A draw whose file is missing takes the next slot of the same day that has a file. The
  search continues at 00:00 after the last slot, so a replacement can lie earlier in the
  same UTC day than the drawn slot, and a file that cannot be read counts as missing
  The first version of the downloader stopped
  after three later slots and left 90 draws without a file (`results/E01_corpus.json` at
  commit `1d3e830`). After the correction the nine calendar months with such draws were
  downloaded again: 2015-10, 2016-06, 2020-10, 2020-11, 2021-08, 2022-11, 2023-03,
  2025-06 and 2025-07.
- The draws still without a file all fall on the days from 2025-06-15 to 2025-07-01. On
  each of those days all 96 slots were tried and none gave a readable file; the
  manifests under `data/interim/corpus/` record the number of slots tried, not the cause
  (`files.unavailable_by_month` and `files.days_with_no_file` in
  `results/E01_corpus.json`). The index month of June 2025 therefore rests on the days
  from 2025-05-30 to 2025-06-14, one more valid day than the coverage rule needs
  (`results/E01_monthly_counts.csv`).
- The Google Storage address serves the same files and can stand in for the GDELT host.
- During the run the GDELT host began answering http with a redirect to https. The base
  URL was changed to https.
- A network outage of about 19 minutes left 19 calendar months with missing files under
  the first version of the downloader, which treated any failure as a missing file. Those
  months were deleted and downloaded again.
- Later, two connections slowed to about 30 KB/s and held one month open for 40 minutes
  while new connections ran at about 16 MB/s. The downloader now drops a file that takes
  over 120 seconds and retries on a new connection.
- Throughput on the same line ranged from about 4 to 40 MB/s over the afternoon.
