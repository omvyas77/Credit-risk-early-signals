# Model card: late-response risk for CFPB complaints

Every number between the generated markers is filled in from `reports/` by `make writeup`; a
test fails if they drift. The full tables are in [reports/summary.md](reports/summary.md).

## Model

- **Task.** Given a consumer complaint as it arrives at the CFPB, score the probability that
  the company will miss the 15-day response window.
- **Main model.** Logistic regression (scikit-learn `Pipeline`) on TF-IDF word 1-2 grams,
  urgency and harm word counts and VADER sentiment from the narrative; one-hot product,
  sub-product, issue, sub-issue, state, channel and tags; days from receipt to sending; and the
  company's smoothed late rate and complaint volume over the year ending 30 days before the
  complaint. Regularisation is chosen on validation average precision.
- **Also evaluated.** The same LR on narrative only and on intake fields only; LightGBM on
  each feature set; the company's late rate alone, with no model; class-weighted LR with and
  without Platt or isotonic recalibration. MiniLM embeddings with LR on the narrative
  (latest window); MiniLM with the intake fields and a fine-tuned DistilBERT are built but
  were not run.
- **In-browser scorer.** The LR refitted on the latest window without VADER and with a pruned
  vocabulary, shipped as `docs/scorer/model.json` and run by `docs/js/scorer.js`.
- **Version.** v2.0, by Om Vyas. MIT licensed.

## Intended use

- Triage of complaint handling: ranking incoming complaints so that the ones most likely to
  get a late response are looked at first, by a complaint team or by someone studying how
  companies handle complaints.
- Research on whether complaint text carries signal beyond the structured fields.

## Out of scope

- **Credit decisions** of any kind: lending, pricing, limits or collections. The label is about
  how a company responds to a complaint, not about anyone's creditworthiness.
- **Judging individual consumers.** A high score says the company is likely to be late, not
  that the consumer or the complaint is in any way suspect.
- Conclusions about a company's legal compliance. The CFPB does not verify the facts in
  complaints, and
  "late" is the company's own reported timing.
- Complaints from other sources, or from periods after the training data without retraining
  and checking calibration.

## Training data

Complaints with a published narrative from the CFPB Narratives Archive, joined on complaint ID
to the live complaints file, which supplies every other field and the outcome. The label is
`late = timely_response == "No"` from the live file, for complaints received from 2017-05-01
through a data end date chosen from the narrative counts. Rows still "In progress" are dropped.
Sources, schema and caveats: [DATA.md](DATA.md).

<!-- generated:data -->
- Downloaded 2026-11-08: 21 archive ZIPs and the live file (19,020,152 complaints). Their sha256 checksums and download times are in `reports/data_manifest.json`.
- 3,851,366 narratives join to the live file; 3,674,640 of them were received 2017-05-01 to 2026-06-30 with a settled outcome and are labelled, and 36,901 (1.00%) of those were answered late.
<!-- /generated:data -->

Five rolling windows are generated from the data end date. Validation ends 31 days before test
starts and training ends 31 days before validation starts, so each pair of splits has 30 clear
days between them; training is capped at the most recent year.

<!-- generated:windows -->
| Window | Train | Validation | Test | Train rows | Test rows | Test late share | Test narrative share |
|---|---|---|---|---|---|---|---|
| W1 | 2023-11-02 to 2024-11-01 | 2024-12-02 to 2025-03-01 | 2025-04-01 to 2025-06-30 | 740,384 | 324,614 | 1.0% | 26.3% |
| W2 | 2024-01-30 to 2025-01-29 | 2025-03-01 to 2025-05-31 | 2025-07-01 to 2025-09-30 | 936,773 | 316,020 | 0.8% | 21.5% |
| W3 | 2024-05-02 to 2025-05-01 | 2025-06-01 to 2025-08-31 | 2025-10-01 to 2025-12-31 | 1,074,903 | 210,719 | 0.9% | 13.8% |
| W4 | 2024-08-03 to 2025-08-02 | 2025-09-02 to 2025-12-01 | 2026-01-01 to 2026-03-31 | 1,206,806 | 55,988 | 3.4% | 3.4% (low) |
| W5 | 2024-11-02 to 2025-11-01 | 2025-12-02 to 2026-03-01 | 2026-04-01 to 2026-06-30 | 1,240,869 | 55,380 | 4.0% | 3.0% (low) |
<!-- /generated:windows -->

## Metrics

Test PR-AUC with 95% CIs from a stratified bootstrap of each test quarter
(`evaluate.bootstrap_resamples` in the config); differences are paired on the same resamples. "Chance" is the test share of late responses.

<!-- generated:metrics -->
| Window | Chance | Company late rate alone | LR, struct | LR, both | both - struct | LR, both: AUROC |
|---|---|---|---|---|---|---|
| W1 | 0.010 | 0.564 (0.550 to 0.579) | 0.546 (0.531 to 0.561) | 0.541 (0.527 to 0.556) | -0.005 (-0.006 to -0.004) | 0.986 (0.985 to 0.987) |
| W2 | 0.008 | 0.482 (0.464 to 0.499) | 0.547 (0.528 to 0.565) | 0.530 (0.512 to 0.548) | -0.016 (-0.022 to -0.011) | 0.987 (0.985 to 0.988) |
| W3 | 0.009 | 0.500 (0.480 to 0.521) | 0.587 (0.568 to 0.607) | 0.570 (0.550 to 0.590) | -0.017 (-0.025 to -0.008) | 0.986 (0.985 to 0.987) |
| W4 | 0.034 | 0.613 (0.593 to 0.634) | 0.679 (0.659 to 0.699) | 0.677 (0.659 to 0.698) | -0.002 (-0.008 to +0.004) | 0.958 (0.955 to 0.962) |
| W5 | 0.040 | 0.660 (0.641 to 0.679) | 0.713 (0.695 to 0.732) | 0.705 (0.687 to 0.724) | -0.008 (-0.012 to -0.004) | 0.956 (0.952 to 0.959) |
<!-- /generated:metrics -->

<!-- generated:answer -->
Across 5 rolling quarterly test windows, a logistic regression on the intake fields and the company's history reaches a mean test PR-AUC of 0.614. Adding the narrative gives 0.605 (its paired 95% CI is below zero in 4 of 5 windows), and the narrative alone gives 0.180, against 0.020 by chance. Most of the signal is the company: its own trailing late rate, with no model, reaches 0.564.
<!-- /generated:answer -->

**Thresholds on the latest window** (LR, `both`). Each threshold is chosen on validation and
applied unchanged to test, so the share flagged on test can differ from the target.

<!-- generated:thresholds -->
Costs for the cost-optimal rule: a missed late response counts 10 times a needless review.

| Rule (chosen on validation) | Share of test flagged | Precision (95% CI) | Recall (95% CI) |
|---|---|---|---|
| F2-optimal | 5.9% | 0.507 (0.494 to 0.520) | 0.742 (0.724 to 0.758) |
| cost-optimal | 9.4% | 0.351 (0.343 to 0.360) | 0.813 (0.798 to 0.830) |
| top 1% of validation scores | 2.0% | 0.886 (0.867 to 0.905) | 0.431 (0.409 to 0.452) |
| top 5% of validation scores | 10.0% | 0.332 (0.324 to 0.340) | 0.822 (0.806 to 0.838) |
<!-- /generated:thresholds -->

## Calibration

Expected calibration error (ECE) uses ten equal-count bins of test scores. LR, `both`:

<!-- generated:calibration -->
| Window | ECE (95% CI) | Brier (95% CI) | Log loss (95% CI) |
|---|---|---|---|
| W1 | 0.0048 (0.0046 to 0.0049) | 0.0071 (0.0070 to 0.0072) | 0.0270 (0.0263 to 0.0276) |
| W2 | 0.0024 (0.0023 to 0.0025) | 0.0054 (0.0053 to 0.0055) | 0.0204 (0.0199 to 0.0210) |
| W3 | 0.0016 (0.0014 to 0.0019) | 0.0056 (0.0055 to 0.0058) | 0.0211 (0.0204 to 0.0218) |
| W4 | 0.0067 (0.0059 to 0.0076) | 0.0178 (0.0172 to 0.0184) | 0.0696 (0.0672 to 0.0719) |
| W5 | 0.0048 (0.0038 to 0.0058) | 0.0196 (0.0189 to 0.0203) | 0.0770 (0.0741 to 0.0796) |

Mean ± sd across windows, `both`:

| Model | PR-AUC | ECE | Brier |
|---|---|---|---|
| LR | 0.6048 ± 0.0808 | 0.0040 ± 0.0021 | 0.0111 ± 0.0070 |
| LR, class-weighted | 0.5721 ± 0.0810 | 0.1336 ± 0.1201 | 0.0924 ± 0.0838 |
| LR, class-weighted + Platt | 0.5721 ± 0.0810 | 0.0030 ± 0.0013 | 0.0118 ± 0.0077 |
| LR, class-weighted + isotonic | 0.5455 ± 0.0852 | 0.0033 ± 0.0013 | 0.0119 ± 0.0078 |
<!-- /generated:calibration -->

Class weighting inflates the scores and lowers PR-AUC. Platt recalibration on validation
repairs the scores but, being monotone, cannot repair the ranking; isotonic loses a little more.
The unweighted LR is already well calibrated.

## In-browser scorer

<!-- generated:scorer_gap -->
The in-browser scorer leaves out VADER sentiment (it has no JavaScript port) and keeps 50,000 terms. On W5 test its PR-AUC is 0.704 against 0.705 for the full LR; the paired difference is -0.001 (-0.001 to -0.000).
<!-- /generated:scorer_gap -->

<!-- generated:scorer -->
| Model | PR-AUC | AUROC | Log loss | ECE |
|---|---|---|---|---|
| LR | 0.705 (0.687 to 0.724) | 0.956 (0.952 to 0.959) | 0.0770 (0.0741 to 0.0796) | 0.0048 (0.0038 to 0.0058) |
| In-browser scorer (LR, no VADER, pruned vocabulary) | 0.704 (0.686 to 0.723) | 0.956 (0.952 to 0.959) | 0.0771 (0.0743 to 0.0797) | 0.0048 (0.0039 to 0.0059) |
<!-- /generated:scorer -->

The scorer fills dropdowns the visitor leaves empty, and fields the page doesn't ask for, with
training means, and gives unlisted companies a default history; `docs/data/scorer.json` records
the rules.

## Known shift

<!-- generated:shift -->
W4 and W5 test on 2026-01-01 to 2026-06-30, when 3.4% and 3.0% of complaints carry a published narrative and 3.4% and 4.0% of those are late; W1's test quarter had 26.3% and 1.0%. Those windows are drawn from a different, much smaller selection of complaints, so compare each window's PR-AUC with its own chance level.
<!-- /generated:shift -->

The data end date is chosen by comparing each month's narrative count with the median of the
six before it, which adapts to the new, lower level instead of flagging it. The decision was to
keep that date and flag the shift wherever results are shown rather than cut the data early.

## Fairness and proxy caveats

**The label reflects company behaviour.** Lateness is decided by the company, so the model
mostly learns which companies are late.

<!-- generated:proxies -->
On W5 test, company-history features carry 76% of the LR's contribution (product 10%, issue 5%, narrative terms 5%, state 2%, tags 1%). The narrative terms that raise the score most are "mohela", "cash app", "cash", "cashapp", "app" and "credit collection": mostly the names of companies and their apps.
<!-- /generated:proxies -->

- **Proxies.** State and the Servicemember and Older American tags are features. They carry
  little weight here, but they can stand in for protected characteristics, and the way people
  write can correlate with language, education or disability. Performance has not been audited
  by group.
- **Selection.** Complaints are self-selected, and only consumers who consent have their
  narrative published, so the data does not represent all customers or all problems.
- **Company history** is computed from complaints with and without narratives, counting only
  those received at least 30 days before the one being scored, so it uses no future outcomes.

## Other limits

- Each window trains on one year, so the matrices fit on an 8 GB laptop.
- Days from receipt to sending the complaint is known only once the CFPB forwards it; most are
  sent the same day, but not all.
- LightGBM's parameters were not tuned, and it often stops early; LR's chosen regularisation
  sometimes sits at the edge of the searched grid. Both are listed in `reports/summary.md`.

<!-- generated:deep -->
`minilm__text__W5` reached a test PR-AUC of 0.296 (0.276 to 0.315), -0.032 (-0.043 to -0.020) against LR on the same features. Not run: `minilm__both__W5`, `distilbert__text__W5`, `distilbert__both__W5`. No GPU was available to finish them; `notebooks/README.md` has the steps.
<!-- /generated:deep -->

## Maintenance

The live complaints file changes daily, and the CFPB adds files to the narrative archive,
which have to be added to `config/sources.yaml` by hand. Results are pinned to the downloads in `reports/data_manifest.json`, which
records each file's checksum and download time. `make download` keeps files whose checksums
match, so a rebuild on the same machine reuses them; a fresh clone downloads the current live
file and can shift the results slightly. The model is not retrained on a schedule.
