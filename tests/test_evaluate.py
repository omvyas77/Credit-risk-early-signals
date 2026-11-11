import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from cres.evaluate import (
    bootstrap_weights,
    choose_thresholds,
    ci,
    flagged_counts,
    metrics,
    prepare,
)

# Hand-built scores with ties and both classes; no real data.
SCORE = np.array([0.9, 0.8, 0.8, 0.7, 0.6, 0.6, 0.4, 0.3, 0.2, 0.2, 0.1, 0.05])
Y = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])


def test_unweighted_metrics_match_sklearn():
    m = metrics(prepare(SCORE, Y, 3))
    assert m["pr_auc"] == pytest.approx(average_precision_score(Y, SCORE))
    assert m["auroc"] == pytest.approx(roc_auc_score(Y, SCORE))
    assert m["log_loss"] == pytest.approx(log_loss(Y, SCORE))
    assert m["brier"] == pytest.approx(brier_score_loss(Y, SCORE))
    assert m["prevalence"] == pytest.approx(4 / 12)


def test_integer_weights_equal_repeating_rows():
    rng = np.random.default_rng(0)
    w = rng.integers(0, 4, len(SCORE)).astype(float)
    w[[0, 2]] = [1, 2]  # keep both classes present
    p = prepare(SCORE, Y, 3)
    weighted = metrics(p, w[p["order"]])
    rep = np.repeat(np.arange(len(SCORE)), w.astype(int))
    assert weighted["pr_auc"] == pytest.approx(average_precision_score(Y[rep], SCORE[rep]))
    assert weighted["auroc"] == pytest.approx(roc_auc_score(Y[rep], SCORE[rep]))
    assert weighted["log_loss"] == pytest.approx(log_loss(Y[rep], SCORE[rep]))


def test_ece_uses_equal_count_bins():
    # Two bins of two rows: |mean score - observed| = |0.85 - 1| and |0.15 - 0|.
    m = metrics(prepare(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]), 2))
    assert m["ece"] == pytest.approx(0.5 * 0.15 + 0.5 * 0.15)
    perfect = metrics(prepare(np.array([1.0, 1.0, 0.0, 0.0]), np.array([1, 1, 0, 0]), 2))
    assert perfect["ece"] == pytest.approx(0)


def test_capacity_metrics_flag_the_top_fraction():
    m = metrics(prepare(SCORE, Y, 3), capacities=(0.25,))
    # The top 3 of 12 rows are 0.9, 0.8, 0.8: two positives out of four.
    assert m["precision_top0.25"] == pytest.approx(2 / 3)
    assert m["recall_top0.25"] == pytest.approx(2 / 4)


def test_flagged_counts_at_a_threshold():
    p = prepare(SCORE, Y, 1)
    tp, fp, fn, tn = flagged_counts(p, np.ones(len(SCORE)), 0.6)
    assert (tp, fp, fn, tn) == (3, 3, 1, 5)


def test_f2_threshold_is_the_best_on_validation():
    t = choose_thresholds(SCORE, Y, (), {"fp": 1, "fn": 1})["f2"]

    def f2(th):
        flag = SCORE >= th
        tp = (flag & (Y == 1)).sum()
        prec, rec = tp / flag.sum(), tp / Y.sum()
        return 5 * prec * rec / (4 * prec + rec) if tp else 0

    assert f2(t) == pytest.approx(max(f2(th) for th in np.unique(SCORE)))


def test_cost_threshold_minimises_validation_cost():
    def cost(th, c):
        flag = SCORE >= th
        return c["fp"] * (flag & (Y == 0)).sum() + c["fn"] * (~flag & (Y == 1)).sum()

    for c in ({"fp": 1, "fn": 1}, {"fp": 1, "fn": 10}, {"fp": 5, "fn": 1}):
        t = choose_thresholds(SCORE, Y, (), c)["cost"]
        best = min([cost(th, c) for th in np.unique(SCORE)] + [c["fn"] * Y.sum()])
        assert cost(t, c) == best


def test_cost_threshold_flags_nothing_when_false_positives_are_too_dear():
    score, y = np.array([0.9, 0.8, 0.1]), np.array([0, 1, 0])
    t = choose_thresholds(score, y, (), {"fp": 1000, "fn": 1})["cost"]
    assert t == float("inf")
    tp, fp, _, _ = flagged_counts(prepare(score, y, 1), np.ones(3), t)
    assert tp + fp == 0


def test_capacity_threshold_flags_that_share_of_validation():
    t = choose_thresholds(SCORE, Y, (0.25, 0.5), {"fp": 1, "fn": 1})
    assert (SCORE >= t["capacity_0.25"]).sum() == 3
    assert (SCORE >= t["capacity_0.5"]).sum() == 6


def test_thresholds_from_validation_are_applied_unchanged_to_test():
    t = choose_thresholds(SCORE, Y, (0.25,), {"fp": 1, "fn": 1})
    test_score = np.array([0.95, 0.5, 0.85, 0.1])
    test_y = np.array([1, 0, 0, 1])
    m = metrics(prepare(test_score, test_y, 1), thresholds={"capacity_0.25": t["capacity_0.25"]})
    flag = test_score >= t["capacity_0.25"]  # 0.8: flags 0.95 and 0.85
    assert m["capacity_0.25:flagged"] == pytest.approx(flag.mean())
    assert m["capacity_0.25:precision"] == pytest.approx(0.5)
    assert m["capacity_0.25:recall"] == pytest.approx(0.5)


def test_bootstrap_keeps_class_counts_and_is_seeded():
    y = np.array([1] * 5 + [0] * 95)
    a = bootstrap_weights(y, np.random.default_rng([42, 1]))
    b = bootstrap_weights(y, np.random.default_rng([42, 1]))
    np.testing.assert_array_equal(a, b)
    assert a[y == 1].sum() == 5
    assert a[y == 0].sum() == 95
    assert np.all(a == np.round(a)) and a.min() >= 0


def test_bootstrap_ci_brackets_the_estimate_and_collapses_for_a_perfect_model():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    noisy = np.clip(0.3 * y + rng.random(2000) * 0.7, 0, 1)
    for score, exact in ((noisy, False), (y * 0.9 + 0.05, True)):
        p = prepare(score, y, 10)
        point = metrics(p)["pr_auc"]
        draws = np.array(
            [metrics(p, bootstrap_weights(y, rng)[p["order"]])["pr_auc"] for _ in range(200)]
        )
        assert not np.isnan(draws).any()
        lo, hi = ci(draws)
        assert lo <= point <= hi
        if exact:
            assert lo == hi == pytest.approx(1.0)
        else:
            assert hi - lo > 0.01


def test_evaluate_stage_on_a_hand_built_window(tmp_path, monkeypatch):
    import json

    from cres import config, evaluate, features, models

    monkeypatch.setattr(features, "FEATURES", tmp_path / "features")
    monkeypatch.setattr(features, "REPORT", tmp_path / "windows.json")
    monkeypatch.setattr(models, "PREDS", tmp_path / "preds")
    monkeypatch.setattr(models, "MODELS", tmp_path / "models")
    monkeypatch.setattr(evaluate, "METRICS", tmp_path / "metrics.json")
    monkeypatch.setattr(evaluate, "CURVES", tmp_path / "curves")
    monkeypatch.setattr(evaluate, "SUMMARY", tmp_path / "summary.md")
    monkeypatch.setattr(config, "STAMPS", tmp_path / "stamps")
    rng = np.random.default_rng(0)
    d = features.FEATURES / "W1"
    d.mkdir(parents=True)
    ys = {}
    for s, n in (("val", 400), ("test", 300)):
        ids = rng.permutation(np.arange(n) + (0 if s == "val" else 1000))  # unsorted on disk
        ys[s] = (rng.random(n) < 0.1).astype(np.int8)
        np.save(d / f"ids_{s}.npy", ids)
        np.save(d / f"y_{s}.npy", ys[s])
        cells = (
            ("lr", "struct", 1.0),
            ("lr", "both", 1.5),
            ("lr", "text", 0.3),
            ("lr_sub", "text", 0.4),  # a deep-table cell; the GPU ones are missing
        )
        for m, fs, strength in cells:
            score = 1 / (1 + np.exp(-(strength * ys[s] + rng.normal(0, 1, n) - 2)))
            path = models.pred_path(m, fs, "W1")
            if s == "test":
                old = models.read_preds(path)["val"]
                models.write_preds(
                    path, {"val": old[0], "test": ids}, {"val": old[1], "test": score}
                )
            else:
                models.write_preds(
                    path, {"val": ids, "test": ids[:0]}, {"val": score, "test": score[:0]}
                )
    features.REPORT.write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window": "W1",
                        "dates": {"test": ["2025-04-01", "2025-06-30"]},
                        "test": {"prevalence": float(ys["test"].mean()), "rows": 300},
                        "val": {"prevalence": float(ys["val"].mean())},
                        "narrative_share": {"test": 0.25},
                    }
                ]
            }
        )
    )
    cfg = {
        "seed": 42,
        "evaluate": {
            "capacity": [0.05],
            "bootstrap_resamples": 50,
            "ece_bins": 5,
            "costs": {"fp": 1, "fn": 10},
        },
        "deep": {"windows": ["W1"]},
    }
    assert evaluate.main(cfg) == 0
    report = json.loads(evaluate.METRICS.read_text())
    cell = report["cells"]["lr__both__W1"]
    lo, hi = cell["ci95"]["pr_auc"]
    assert lo <= cell["point"]["pr_auc"] <= hi
    assert set(cell["thresholds"]) == {"f2", "cost", "capacity_0.05"}
    pair = next(
        c for c in report["comparisons"] if c["a"] == ["lr", "both"] and c["b"] == ["lr", "struct"]
    )
    assert pair["metrics"]["pr_auc"]["diff"] == pytest.approx(
        cell["point"]["pr_auc"] - report["cells"]["lr__struct__W1"]["point"]["pr_auc"]
    )
    curve = json.loads((evaluate.CURVES / "lr__both__W1.json").read_text())
    for row in curve["thresholds"]:
        assert sum(row["test"].values()) == 300 and sum(row["val"].values()) == 400
    text = evaluate.SUMMARY.read_text()
    assert "both - struct" in text and "W1" in text
    # Deep cells without prediction files are listed and shown as "not run".
    assert report["not_run"] == [
        "lr_sub__both__W1",
        "minilm__text__W1",
        "minilm__both__W1",
        "distilbert__text__W1",
        "distilbert__both__W1",
    ]
    assert "| W1 | DistilBERT, fine-tuned | train subsample | not run | not run |  |  |" in text
    sub_row = next(r for r in text.splitlines() if "DistilBERT's training subsample |" in r)
    assert sub_row.count("not run") == 1  # text was run, both was not
    assert "Not run yet: lr_sub__both__W1, minilm__text__W1" in text
    # A rerun with nothing new is skipped; a removed prediction file makes it run again.
    paths = sorted(models.PREDS.glob("*.parquet"))
    assert evaluate.current(paths, cfg)
    assert not evaluate.current(paths[1:], cfg)
    assert evaluate.current(paths, {**cfg, "scorer": {"digits": 3}})  # a section it doesn't read
    assert not evaluate.current(paths, {**cfg, "seed": 7})


def test_training_caveats_report_stopping_rounds_and_c_at_the_grid_edge():
    from cres.evaluate import training_caveats

    report = {
        "models": {"lr": {"C": [0.01, 0.1, 1.0]}, "lgbm": {"num_boost_round": 500}},
        "training": {
            "lgbm__both__W1": {"best_iteration": 40},
            "lgbm__text__W1": {"best_iteration": 7},
            "lgbm__struct__W1": {"best_iteration": 300},
            "lr__struct__W1": {"model": "lr", "featureset": "struct", "window": "W1", "C": 0.01},
            "lr__both__W1": {"model": "lr", "featureset": "both", "window": "W1", "C": 0.1},
        },
    }
    lgbm, lr = training_caveats(report)
    assert "from 7 to 300 (median 40) out of up to 500" in lgbm
    assert "lr struct W1 (C=0.01)" in lr and "both" not in lr
    assert training_caveats({"models": {}, "training": {}}) == []
