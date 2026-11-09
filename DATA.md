# Data card

What the source files actually contain, found by inspecting them. Counts that come out of the
pipeline live in `reports/data_profile.json`; this file describes the schema and caveats.

## Sources

| Source | Files | Zipped size | Pinned by |
|---|---|---|---|
| CFPB Narratives Archive (FOIA Reading Room) | 21 ZIPs, `CCDB_Export_{N}_{period}.zip`, N = 1 (Dec 2011 to Apr 2018) to 21 (Aug 2026) | 1,430 MB | sha256 in `reports/data_manifest.json` |
| Live complaints file | `complaints.csv.zip` | 364 MB | sha256 and download time in `reports/data_manifest.json` |

URLs are in `config/sources.yaml`, copied from the archive page on 2026-11-08. Each ZIP holds
exactly one CSV with the same base name.

## Archive files: full records, not narratives only

Every archive CSV has the same 16-column header (checked on all 21 files), and each file holds
**every complaint received in its period, with or without a narrative**. Only a minority of rows
carry narrative text (file 17, April 2026: 21,052 of 618,293 rows; file 2, May 2018 to
April 2021: 439,860 of 1,041,632).

| CSV header | Column | Type | Notes |
|---|---|---|---|
| Date received | `date_received` | date | ISO `YYYY-MM-DD`; matches the file's period |
| Product | `product` | text | Product names changed over time (file 2 uses "Credit reporting, credit repair services, or other personal consumer reports"; file 17 uses "Credit reporting or other personal consumer reports") |
| Sub-product | `sub_product` | text | A few nulls; "I do not know" is a real value |
| Issue | `issue` | text | |
| Sub-issue | `sub_issue` | text | Null in about 1% (file 17) to 11% (file 2) of rows |
| Consumer complaint narrative | `narrative` | text | Archive only. Redactions are runs of capital X (`XXXX`, `XX/XX/XXXX`), often glued to digits or words (`XXXX1234`, `XXXXaccount`); in a 30,000-narrative sample there were 440,593 capital-X runs and 6 lowercase ones. `text.py` removes the capital runs |
| Company public response | `company_public_response` | text | Outcome; banned as a feature |
| Company | `company` | text | Some names carry a leading non-breaking space |
| State | `state` | text | Includes military and territory codes (AA, AE, ...) |
| ZIP code | `zip_code` | text | Can be `XXXXX` or partly redacted; keep as text |
| Tags | `tags` | text | `Servicemember`, `Older American`, both, or null |
| Submitted via | `submitted_via` | text | Web, Referral, Phone, Postal mail and two rarer channels (6 values) |
| Date sent to company | `date_sent_to_company` | date | Can be months after `date_received` |
| Company response to consumer | `company_response_to_consumer` | text | Outcome; banned as a feature |
| Timely response? | `timely_response` | text | `Yes` / `No`; outcome, used only to build the label |
| Complaint ID | `complaint_id` | bigint | Unique within a file |

Encoding: read as UTF-8 by DuckDB without errors on the files inspected. Empty fields are read
as null; `ingest.py` also turns blank strings into null.

## Live file

`complaints.csv` (5.77 GB unzipped) has the archive's columns **minus the narrative**: 15
columns, same header names, same order otherwise. The copy downloaded 2026-11-08 has
19,020,152 rows, one per complaint ID, received 2011-12-01 through 2026-11-08 (it is updated
daily, so rows keep arriving).

Values found in the live copy:
- `timely_response`: `Yes` or `No`, never missing (113,542 `No`).
- `company_response_to_consumer`: 8 values, including `In progress` (641,049 rows, mostly
  recent complaints whose outcome isn't settled) and a bare `Closed`; 21 nulls.
- `product`: 21 names, because the CFPB renamed and merged products over the years (for example
  "Credit reporting, credit repair services, or other personal consumer reports" became
  "Credit reporting or other personal consumer reports").
- `issue`: 179 values; `sub_issue`: 272, null in about 5% of rows.
- `company`: 8,211 names; `submitted_via`: 6 channels (Web is 97%).
- `zip_code`: messy free text (values like `(1352` exist); not used as given.

**Neither source has `Consumer disputed?`.** The live file no longer publishes it, and the archive
files never did. The disputed-or-late label that v1 used up to April 2017 can't be rebuilt from
these files, so the profile has no disputed series. The late-only label from 2017-05-01 doesn't
need it.

## Join

Archive narrative rows are deduplicated by `complaint_id` (the latest export wins), then
inner-joined to the live file. The narrative comes from the archive; every other field, including
outcomes, comes from the live file. `reports/data_profile.json` (`ingest`) records archive rows,
live rows, matched rows, archive-only rows, duplicates, and how often the archive's outcome fields
disagree with the live file's.

Counts from the 2026-11-08 build (`reports/data_profile.json`, `ingest`):
- The 21 archive files hold 17,533,918 complaints, of which 3,851,415 have a narrative. No
  complaint ID appears in more than one file, so deduplication drops nothing.
- 3,851,366 narratives match a live row; 49 are archive-only and are dropped.
- The total matches the 3,851,415 narratives reported by the `cfpb-complaints-analysis` project
  exactly.
- On matched rows the archive's outcomes almost always agree with the live file:
  `timely_response` differs on 3 rows and `company_response_to_consumer` on 298.

## Label

`late = timely_response == 'No'` from the live file, for complaints with a narrative received
from 2017-05-01 through `data_end`. Rows dropped by each rule, the chosen `data_end` and the
monthly series are in `reports/data_profile.json` (`label`). Rows still "In progress" are
dropped; among narrative rows there are only 107, because the archive was published after
those complaints were settled. In the live file as a whole, almost all "In progress" rows
(639,943 of 641,049) were received from September 2026 on.

## Caveat: the narrative share collapses from late 2025

Total complaint volume in the live file is steady at 500,000 to 620,000 a month through 2026,
but the share of complaints with a published narrative falls from about 26% (mid-2025) to
about 15% (autumn 2025), 11% (December 2025) and about 3% from January 2026. Within that thin
2026 subset the late rate is about 3.5 to 4.5%, against about 0.8% for narrative complaints in
mid-2025, while the late rate across all complaints stays near 0.5%. So the narrative
complaints of 2026 are a different, much smaller selection, not just fewer of the same.

The `data_end` rule compares each month with the median of the six before it, so it adapts to
the new level and does not flag this drop: it chooses 2026-06-30 (July 2026, about 10,000
narratives, is the first month to fail). Any test window from 2026 is drawn from the thin
subset; results should be read per window with that in mind.

Decision: keep the rule and its `data_end` of 2026-06-30, and flag the shift wherever results
are shown rather than cutting the data early. The monthly series in `reports/data_profile.json`
(`label.timeline`) carries total complaints, narrative share and both late rates, so the flag is
drawn from pipeline output.

## Windows

`split.py` generates five rolling windows from `data_end` and `config/default.yaml`. The test
periods are the last five whole quarters; validation is the three months ending 31 days before
test starts, and training ends 31 days before validation starts, so 30 clear days separate each
split (an outcome is known about 15 days after a complaint arrives). Training is capped at the
most recent year before that cutoff (`windows.train_years`), so each window's TF-IDF matrix fits
in memory on an 8 GB laptop; earlier complaints are not used for training. Row counts,
prevalence and narrative share per split are in `reports/windows.json`; every row's window tags
are in `data/features/splits.parquet`.

## Features

Only fields known when a complaint arrives. Outcome columns are dropped by `label.py`, and
`features.py` refuses a source that still has them.

| Block | Columns | Fitted on |
|---|---|---|
| TF-IDF | word 1-2 grams from `text.py`; vocabulary of up to 100,000 terms chosen from a seeded 150,000-document train sample (min df 5); idf from all train rows; sublinear tf, L2 rows | train rows of the window |
| Categorical | product, sub-product, issue, sub-issue, state, submitted via, tags; one-hot, categories seen fewer than 20 times share an "infrequent" column, missing is its own category | train rows of the window |
| Structured numeric | days from receipt to sending to the company; company late rate; company log volume | nothing (scaled inside the model Pipeline) |
| Text numeric | urgency and harm lexicon counts (`text.LEXICON`, whole-token matches, negations kept); VADER compound score on the raw narrative | nothing |

Company history uses the live file (complaints with and without narratives). For a complaint
received on day d, it counts the same company's settled complaints received in the 365 days
before d - 30, and smooths the late rate toward the late rate of all companies over the same
days: (late + 50 x overall) / (n + 50). The volume feature is ln(1 + n).

Feature sets: `text` = TF-IDF + text numerics; `struct` = categorical + structured numerics;
`both` = all of them.

The days-to-company feature is known only once the CFPB forwards the complaint. For complaints
received from November 2023, 95.2% were sent the same day and 96.5% within a day; the 99th
percentile is 16 days and the maximum 446. It is kept as a field of intake, as planned, with
that caveat.
