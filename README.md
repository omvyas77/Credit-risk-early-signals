# Does a complaint's text predict a late company response?

When a consumer files a complaint with the Consumer Financial Protection Bureau (CFPB), the
company has 15 days to respond. This project asks whether what the consumer writes helps
predict which complaints will get a late response, beyond what the complaint form already
says (company, product, issue, state, channel) and the company's own track record.

**[Read the results and try the scorer in your browser](https://omvyas77.github.io/Credit-risk-early-signals/)**

This is about **complaint-handling risk**: how companies respond to complaints. It is not a
credit-risk model and must not be used for credit decisions or to judge individual consumers.
See [MODEL_CARD.md](MODEL_CARD.md).

## The answer

<!-- generated:answer -->
Across 5 rolling quarterly test windows, a logistic regression on the intake fields and the company's history reaches a mean test PR-AUC of 0.614. Adding the narrative gives 0.605 (its paired 95% CI is below zero in 4 of 5 windows), and the narrative alone gives 0.180, against 0.020 by chance. Most of the signal is the company: its own trailing late rate, with no model, reaches 0.564.
<!-- /generated:answer -->

## Results

<!-- generated:results -->
| Model | Features | Mean test PR-AUC, 5 windows | W5 test PR-AUC (95% CI) |
|---|---|---|---|
| Chance (share of late responses) | none | 0.020 | 0.040 |
| Company late rate alone | company history | 0.564 | 0.660 (0.641 to 0.679) |
| LR | narrative only | 0.180 | 0.328 (0.308 to 0.348) |
| LR | intake fields + company history | 0.614 | 0.713 (0.695 to 0.732) |
| LR | narrative + intake fields + company history | 0.605 | 0.705 (0.687 to 0.724) |
| LightGBM | narrative only | 0.172 | 0.318 (0.299 to 0.338) |
| LightGBM | intake fields + company history | 0.509 | 0.607 (0.586 to 0.628) |
| LightGBM | narrative + intake fields + company history | 0.460 | 0.626 (0.607 to 0.647) |
| In-browser scorer (LR, no VADER, pruned vocabulary) | narrative + intake fields + company history | W5 only | 0.704 (0.686 to 0.723) |
<!-- /generated:results -->

PR-AUC is the area under the precision-recall curve; a model that guesses scores the share of
late responses. Every test window has its own per-window table with bootstrap CIs in
[reports/summary.md](reports/summary.md), and the model card has calibration and thresholds.
The numbers in this file are filled in from `reports/` by `make writeup`, and a test fails if
they drift.

## How it works

- **Data.** Complaint narratives from the CFPB's archive, joined on complaint ID to the live
  complaints file, which supplies every other field and the outcome.
- **Label.** A complaint is late if the live file marks its response untimely. Outcome fields
  are never features; only what is known when a complaint arrives is.
- **Windows.** Five rolling quarterly test windows generated from the data's end date. Each has
  its own training year and validation quarter, with a 30-day gap between them.
- **Features.** TF-IDF word 1-2 grams, urgency and harm word counts and VADER sentiment from
  the narrative; product, issue, state, channel and tags from the form; and the company's late
  rate over the year ending 30 days before the complaint.
- **Models.** Logistic regression and LightGBM on narrative only, intake fields only, and both;
  the company's late rate alone as a baseline; class-weighted LR with Platt and isotonic
  recalibration. MiniLM embeddings and a fine-tuned DistilBERT run on a GPU in Colab.
- **Evaluation.** Thresholds, recalibration, early stopping and regularisation are chosen on
  validation; test is scored once. 95% CIs come from a stratified bootstrap of each test
  quarter, paired across models.
- **Explanations.** Coefficient contributions by feature group, LightGBM SHAP, a cross-product
  transfer matrix and month-by-month drift.
- **Site.** A static page in `docs/` reads every number from `docs/data/*.json`. Its scorer is
  a JavaScript port of the LR whose tokenizer is tested against the Python one in CI.

## Run it

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.12), `make`, and Node.js for
the scorer parity test.

```bash
git clone https://github.com/omvyas77/Credit-risk-early-signals.git
```

```bash
cd Credit-risk-early-signals && uv sync
```

```bash
make all
```

`make all` downloads the data and runs every stage in order. Each stage skips work whose
outputs are newer than its inputs and were built with the same settings, so after an
interruption run it again and it picks up where it stopped; a rerun with nothing new takes
seconds. After changing pipeline code, rerun the affected stage with
`uv run python -m cres <stage> --force`.

| Target | What it does |
|---|---|
| `make download` | Raw ZIPs into `$CRES_DATA_DIR/raw/`, checksums into `reports/data_manifest.json` |
| `make data` | Unzip, join the archive to the live file, build the label |
| `make features` | Windows and feature matrices into `data/features/` |
| `make train` | Classical models and the in-browser scorer's model into `data/preds/` and `models/` |
| `make evaluate` | Every prediction file into `reports/metrics.json` and `reports/summary.md` |
| `make explain` | Coefficients, SHAP, cross-product transfer and drift into `reports/` |
| `make export` | Site data into `docs/data/` and the scorer into `docs/scorer/` |
| `make writeup` | The generated numbers in this README and the model card |
| `make deep-input` | The slim input for the GPU models on Colab |
| `make deep-baseline` | TF-IDF LR on DistilBERT's training subsample, for a same-data comparison |
| `make test`, `make lint` | pytest (including the scorer parity check under Node) and ruff |
| `make help` | This list |

**Where the data goes.** Raw downloads go to `$CRES_DATA_DIR/raw/`; set `CRES_DATA_DIR` to put
them on another disk (it defaults to `data/`). Everything derived goes under `data/` and
`models/`, which are not tracked. The unzipped CSVs and feature matrices take tens of gigabytes,
so check free disk space first.

**On an 8 GB laptop.** The pipeline was built on one. DuckDB spills to disk past
`duckdb_memory_limit`, each window trains on one year (`windows.train_years`), the TF-IDF
vocabulary is capped, matrices are float32 sparse, and each model is fitted in a fresh process
so memory is returned between fits. Lower `features.workers` and `models.lgbm.num_threads` in
`config/default.yaml` on a machine with fewer cores.
<!-- generated:compute -->
The largest single fit peaked at 4.9 GB of memory, and the recorded model fits took 9.2 hours in all (from `reports/metrics.json`, `training`).
<!-- /generated:compute -->

**Deep models.** MiniLM and DistilBERT need a GPU. `notebooks/colab_deep.ipynb` runs them on
Colab; [notebooks/README.md](notebooks/README.md) has the steps. Copy their prediction files into
`data/preds/` and run `make evaluate` again.

## Data

Two public CFPB sources, listed in [config/sources.yaml](config/sources.yaml):

1. The [Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)
   in the FOIA Reading Room: one ZIP per period with the narrative text, which the public
   complaints export no longer includes.
2. The [live complaints file](https://files.consumerfinance.gov/ccdb/complaints.csv.zip), which
   supplies every other field, including the outcome.

<!-- generated:data -->
- Downloaded 2026-11-08: 21 archive ZIPs and the live file (19,020,152 complaints). Their sha256 checksums and download times are in `reports/data_manifest.json`.
- 3,851,366 narratives join to the live file; 3,674,640 of them were received 2017-05-01 to 2026-06-30 with a settled outcome and are labelled, and 36,901 (1.00%) of those were answered late.
<!-- /generated:data -->

The live file is updated daily, so a fresh clone downloads a newer copy and can shift the
results slightly; the manifest records which download the published numbers come from.
[DATA.md](DATA.md) describes the schema as found and the caveats. Narrative text never leaves
`data/`: it is not in git, `reports/` or the site.

## What changed from v1

<!-- generated:v1 -->
v1 reported a test PR-AUC of 0.314 (LightGBM, TF-IDF and structured features) on a "disputed or late" label, with 287 test positives and its threshold chosen on test. Its data export is gone, so those numbers can't be reproduced or compared with this version's.

| v1 | This version |
|---|---|
| Disputes and late replies mixed in one label | Late replies only |
| Settings picked on the test data | Settings picked on an earlier quarter |
| One test period, about 287 late cases, no error ranges | Rolling test quarters, each with a 95% range |
| 4,336 columns, one per company | The company's own past late rate |
<!-- /generated:v1 -->

v1's notebook, report and figures are in [archive/](archive/).

## Limits

**Narratives thin out in 2026.**
<!-- generated:shift -->
W4 and W5 test on 2026-01-01 to 2026-06-30, when 3.4% and 3.0% of complaints carry a published narrative and 3.4% and 4.0% of those are late; W1's test quarter had 26.3% and 1.0%. Those windows are drawn from a different, much smaller selection of complaints, so compare each window's PR-AUC with its own chance level.
<!-- /generated:shift -->

**One year of training per window.** Each window trains on the most recent year before its
training cutoff, so the matrices fit in memory on an 8 GB laptop. Earlier complaints are not
used.

**Deep models are pending.**
<!-- generated:deep -->
Not run yet: `minilm__text__W5`, `minilm__both__W5`, `distilbert__text__W5`, `distilbert__both__W5`. They need a GPU; `notebooks/README.md` has the Colab steps, and `make evaluate` adds them to every table once their prediction files are in `data/preds/`.
<!-- /generated:deep -->

**The in-browser scorer has no VADER.**
<!-- generated:scorer_gap -->
The in-browser scorer leaves out VADER sentiment (it has no JavaScript port) and keeps 50,000 terms. On W5 test its PR-AUC is 0.704 against 0.705 for the full LR; the paired difference is -0.001 (-0.001 to -0.000).
<!-- /generated:scorer_gap -->

**The label is about companies.** Whether a response is late depends on the company, so the
strongest predictor is which company the complaint is about, and the narrative's most useful
words are mostly company names. The model card has the fairness and proxy caveats.

## Repository

| Path | Contents |
|---|---|
| `src/cres/` | The pipeline, one module per stage (`python -m cres --help`) |
| `config/` | Every setting and the one seed; download URLs |
| `tests/` | pytest on small hand-built fixtures, plus the scorer parity check |
| `reports/` | Pipeline outputs: metrics, curves, explanations, data profile and manifest |
| `docs/` | The site, served by GitHub Pages |
| `notebooks/` | The Colab notebook for the GPU models |
| `archive/` | v1 |

MIT licensed. The CFPB complaint data is public; complaints are not verified by the CFPB.
