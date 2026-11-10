import numpy as np
import pytest

from cres import robustness


def test_largest_breaks_ties_by_name():
    counts = {"Mortgage": 5, "Credit card": 9, "Debt collection": 9, "Student loan": 1}
    assert robustness.largest(counts, 3) == ["Credit card", "Debt collection", "Mortgage"]


def test_score_cell_reports_metrics_and_handles_targets_without_both_classes():
    y = np.array([0, 0, 0, 1, 0, 1, 0, 0])
    perfect = robustness.score_cell(y * 0.9 + 0.05, y, 50, [42, 5, 0])
    assert perfect["pr_auc"] == pytest.approx(1)
    assert perfect["prevalence"] == pytest.approx(0.25)
    assert perfect["ci95"]["pr_auc"] == pytest.approx([1, 1])
    empty = robustness.score_cell(np.full(4, 0.3), np.zeros(4, int), 50, [42, 5, 0])
    assert empty["pr_auc"] is None and empty["late"] == 0


def test_score_cell_resamples_depend_only_on_seed_and_labels():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    s1, s2 = rng.random(300), y * 0.3 + rng.random(300)
    a = robustness.score_cell(s1, y, 100, [42, 5, 1])
    b = robustness.score_cell(s1, y, 100, [42, 5, 1])
    assert a == b
    assert robustness.score_cell(s2, y, 100, [42, 5, 1])["pr_auc"] > a["pr_auc"]


def _cell(pr_auc: float, rows: int, late: int) -> dict:
    return {
        "rows": rows,
        "late": late,
        "prevalence": late / rows,
        "pr_auc": pr_auc,
        "auroc": 0.5 + pr_auc / 2,
        "ci95": {"pr_auc": [pr_auc - 0.1, pr_auc + 0.1], "auroc": [0.5, 1.0]},
    }


def test_matrix_puts_sources_on_rows_targets_on_columns_with_references():
    products = ["A", "B", "C"]
    size = {"A": (100, 10), "B": (200, 4), "C": (50, 5)}
    runs = {
        a: {
            "train_rows": 1000,
            "train_late": 20,
            "C": 0.1,
            "val_average_precision": 0.5,
            # PR-AUC encodes (source, target) so the layout can be checked.
            "targets": {
                b: _cell(0.1 * (i + 1) + 0.01 * (j + 1), *size[b]) for j, b in enumerate(products)
            },
        }
        for i, a in enumerate(products)
    }
    pooled = {b: _cell(0.9, *size[b]) for b in products}
    m = robustness.matrix(products, runs, pooled)
    assert np.array(m["pr_auc"]).shape == (3, 3)
    assert m["pr_auc"][0][2] == pytest.approx(0.13)  # trained on A, tested on C
    assert m["pr_auc"][2][0] == pytest.approx(0.31)
    assert m["in_domain"] == [m["pr_auc"][i][i] for i in range(3)]
    assert m["chance"] == [0.1, 0.02, 0.1]  # each target's test prevalence
    assert m["pooled"] == [0.9, 0.9, 0.9]
    assert [t["rows"] for t in m["test"]] == [100, 200, 50]
    assert np.array(m["pr_auc_ci95"]).shape == (3, 3, 2)


def test_error_cells_counts_mistakes_by_category():
    y = np.array([1, 1, 0, 0, 0, 1])
    score = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.05])
    product = np.array(
        ["Mortgage", "Mortgage", "Credit card", "Credit card", "Mortgage", "Credit card"]
    )
    issue = np.array(["Fees", "Fees", "Billing", "Billing", "Fees", "Fraud"])
    out = robustness.error_cells(y, score, 0.5, product, issue)
    assert out["counts"] == {"tp": 1, "fp": 2, "fn": 2, "tn": 1}
    assert out["false_positives"] == [
        {
            "product": "Credit card",
            "issue": "Billing",
            "count": 2,
            "share": 1.0,
            "test_rows": 2,
            "test_late": 0,
        }
    ]
    assert [(c["product"], c["issue"], c["count"]) for c in out["false_negatives"]] == [
        ("Credit card", "Fraud", 1),
        ("Mortgage", "Fees", 1),
    ]


def test_monthly_drift_by_month():
    months = np.array(["2026-04", "2026-04", "2026-05", "2026-05", "2026-05"])
    y = np.array([0, 1, 0, 0, 0])
    out = robustness.monthly(months, y, {"lr__both": np.array([0.1, 0.9, 0.2, 0.2, 0.5])})
    assert [r["month"] for r in out] == ["2026-04", "2026-05"]
    assert out[0]["prevalence"] == 0.5 and out[0]["lr__both:pr_auc"] == 1.0
    assert out[1]["lr__both:mean_score"] == pytest.approx(0.3)
    assert out[1]["lr__both:pr_auc"] is None  # no late complaints that month
