"""Site JSON is shaped from pipeline reports by plain functions; hand-built reports here."""

import json
import math

import pytest

from cres import evaluate, export


def test_tidy_rounds_and_drops_non_finite():
    assert export.tidy({"a": 0.123456789, "b": [math.inf, 3], "c": "x"}) == {
        "a": 0.123457,
        "b": [None, 3],
        "c": "x",
    }


def test_write_refuses_nan_free_json_only(tmp_path):
    export.write(tmp_path / "a.json", {"t": float("inf")})
    assert json.loads((tmp_path / "a.json").read_text()) == {"t": None}


def cell(pr):
    return {"point": {"pr_auc": pr}, "ci95": {"pr_auc": [pr - 0.1, pr + 0.1]}, "thresholds": {}}


def test_metrics_grid_marks_missing_deep_models_not_run():
    report = {
        "generated": "2026-01-01",
        "bootstrap_resamples": 10,
        "costs": {"fp": 1, "fn": 10},
        "capacity": [0.01],
        "ece_bins": 10,
        "windows": {"W1": {}},
        "cells": {"lr__both__W1": cell(0.6), "company_rate__struct__W1": cell(0.5)},
        "comparisons": [],
        "across_windows": {},
        "not_run": ["distilbert__text__W1"],
    }
    grid = export.metrics_grid(report)
    status = {(c["model"], c["featureset"], c["window"]): c["status"] for c in grid["cells"]}
    assert status == {
        ("company_rate", "struct", "W1"): "ok",
        ("lr", "both", "W1"): "ok",
        ("distilbert", "text", "W1"): "not run",
    }
    assert "scorer" in grid["model_labels"]


def window(name, share):
    split = {"rows": 10, "late": 1, "prevalence": 0.1}
    return {
        "window": name,
        "dates": {"train": ["a", "b"], "val": ["c", "d"], "test": ["e", "f"]},
        "train": split,
        "val": split,
        "test": split,
        "narrative_share": {"train": 0.3, "val": 0.3, "test": share},
    }


def test_windows_flag_the_low_narrative_share_period():
    report = {
        "data_end": "2026-06-30",
        "train_years": 1,
        "windows": [window("W1", 0.26), window("W5", 0.03)],
    }
    info = export.windows_info(report)
    assert [w["low_narrative_share"] for w in info["windows"]] == [False, True]
    assert f"{evaluate.LOW_NARRATIVE_SHARE:.0%}" in info["low_narrative_share_rule"]


def test_compact_curve_keeps_threshold_counts():
    curve = {
        "pr": [{"recall": 0.5, "precision": 0.4, "threshold": 0.3}],
        "reliability": [{"mean_score": 0.1, "observed": 0.2, "n": 5}],
        "thresholds": [
            {
                "threshold": 0.3,
                "val": {"tp": 1, "fp": 2, "fn": 3, "tn": 4},
                "test": {"tp": 5, "fp": 6, "fn": 7, "tn": 8},
            }
        ],
    }
    out = export.compact_curve(curve)
    assert out["thresholds"] == [[0.3, 1, 2, 3, 4, 5, 6, 7, 8]]
    assert len(export.THRESHOLD_COLUMNS) == 9


@pytest.mark.parametrize("name", ["lr__both__W5", "scorer__both__W5"])
def test_split_cell(name):
    assert export.split_cell(name)["window"] == "W5"
