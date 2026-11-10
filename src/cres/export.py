"""Site data for docs/, every number read from pipeline outputs.

Reads reports/*.json, reports/curves/, data/preds/, the feature cache (with the company column
of data/interim/labelled.parquet, never its narratives) and the scorer's files in models/.
Writes
  docs/data/metrics.json      every model x feature set x window: test metrics with 95% CIs,
                              paired comparisons, and the expected deep-model cells that have
                              no prediction file marked "not run"
  docs/data/curves.json       per cell: downsampled PR curve, reliability bins, and val/test
                              TP, FP, FN, TN at a grid of thresholds (for the slider)
  docs/data/calibration.json  reliability before (LR, class-weighted LR) and after
                              recalibration (Platt, isotonic), with ECE, Brier and log loss
  docs/data/timeline.json     monthly complaints, narrative share and late rates; data_end
  docs/data/windows.json      split dates, rows, prevalence and narrative share per window,
                              with the low-narrative-share windows flagged
  docs/data/companies.json    company explorer for the latest window
  docs/data/robustness.json   cross-product matrix, drift by test month, error counts
  docs/data/explain.json      LR group shares, LightGBM SHAP shares and top terms
  docs/data/scorer.json       how the in-browser scorer was fitted, and its metrics next to
                              the full LR
  docs/data/manifest.json     the raw files: name, size, sha256 and download time
  docs/data/v1.json           v1's published numbers and the fixes, from archive/v1.json
  docs/scorer/model.json      the scorer bundle, copied from models/scorer/model.json
"""

import json
import math
import shutil
from pathlib import Path

import duckdb

from cres import evaluate, features, models, scorer
from cres.ingest import INTERIM

DOCS = Path("docs")
DATA = DOCS / "data"
SCORER_DIR = DOCS / "scorer"
REPORTS = Path("reports")
ARCHIVE = Path("archive")
DIGITS = 6
FEATURESET_LABELS = {
    "text": "Narrative only",
    "struct": "Structured fields only",
    "both": "Narrative and structured fields",
}
CALIBRATION_MODELS = ("lr", "lr_balanced", "lr_balanced_platt", "lr_balanced_isotonic")


def tidy(obj, digits: int = DIGITS):
    """Round floats to `digits` significant digits; non-finite floats become null."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return float(f"{obj:.{digits}g}")
    if isinstance(obj, dict):
        return {k: tidy(v, digits) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [tidy(v, digits) for v in obj]
    return obj


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(tidy(obj), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return len(payload)


def split_cell(name: str) -> dict:
    model, featureset, window = name.split("__")
    return {"model": model, "featureset": featureset, "window": window}


# --- metrics -----------------------------------------------------------------------------


def metrics_grid(report: dict) -> dict:
    """Every evaluated cell with point metrics and CIs, plus "not run" cells."""
    cells = [
        {**split_cell(name), "status": "ok", **{k: c[k] for k in ("point", "ci95", "thresholds")}}
        for name, c in sorted(report["cells"].items())
    ]
    cells += [{**split_cell(name), "status": "not run"} for name in report.get("not_run", [])]
    return {
        "generated": report["generated"],
        "bootstrap_resamples": report["bootstrap_resamples"],
        "costs": report["costs"],
        "capacity": report["capacity"],
        "ece_bins": report["ece_bins"],
        "model_labels": evaluate.MODEL_LABELS,
        "featureset_labels": FEATURESET_LABELS,
        "windows": sorted(report["windows"]),
        "cells": cells,
        "comparisons": report["comparisons"],
        "across_windows": report["across_windows"],
        "not_run": report.get("not_run", []),
    }


# --- curves ------------------------------------------------------------------------------

THRESHOLD_COLUMNS = [
    "threshold",
    *(f"{s}_{k}" for s in ("val", "test") for k in ("tp", "fp", "fn", "tn")),
]


def compact_curve(curve: dict) -> dict:
    return {
        "pr": [[p["recall"], p["precision"], p["threshold"]] for p in curve["pr"]],
        "reliability": [[b["mean_score"], b["observed"], b["n"]] for b in curve["reliability"]],
        "thresholds": [
            [t["threshold"], *(t[s][k] for s in ("val", "test") for k in ("tp", "fp", "fn", "tn"))]
            for t in curve["thresholds"]
        ],
    }


def all_curves(curves_dir: Path, cells: dict) -> dict:
    return {
        "columns": {
            "pr": ["recall", "precision", "threshold"],
            "reliability": ["mean_score", "observed", "n"],
            "thresholds": THRESHOLD_COLUMNS,
        },
        "note": "Thresholds flag scores >= threshold; the grid comes from validation scores "
        "(flag rates from 0.1% to 50%) plus the thresholds each rule chose on validation.",
        "cells": {
            name: compact_curve(read(curves_dir / f"{name}.json"))
            for name in sorted(cells)
            if (curves_dir / f"{name}.json").exists()
        },
    }


def calibration(curves_dir: Path, report: dict) -> dict:
    out = {"before": ["lr", "lr_balanced"], "after": ["lr_balanced_platt", "lr_balanced_isotonic"]}
    rows = []
    for name, c in sorted(report["cells"].items()):
        cell = split_cell(name)
        if cell["model"] not in (*CALIBRATION_MODELS, scorer.MODEL):
            continue
        curve = read(curves_dir / f"{name}.json")
        rows.append(
            {
                **cell,
                "reliability": [
                    [b["mean_score"], b["observed"], b["n"]] for b in curve["reliability"]
                ],
                **{
                    k: {"point": c["point"][k], "ci95": c["ci95"][k]} for k in evaluate.DIFF_METRICS
                },
            }
        )
    out["columns"] = ["mean_score", "observed", "n"]
    out["bins"] = f"{report['ece_bins']} equal-count bins of test scores"
    out["cells"] = rows
    return out


# --- timeline and windows ----------------------------------------------------------------


def timeline(profile: dict, windows_report: dict) -> dict:
    label = profile["label"]
    return {
        "data_end": label["data_end"],
        "data_end_rule": label["data_end_rule"],
        "received_from": label["received_from"],
        "months": label["timeline"],
        "windows": [{"window": w["window"], **w["dates"]} for w in windows_report["windows"]],
    }


def windows_info(windows_report: dict) -> dict:
    out = []
    for w in windows_report["windows"]:
        share = w["narrative_share"]
        out.append(
            {
                "window": w["window"],
                "dates": w["dates"],
                **{
                    s: {k: w[s][k] for k in ("rows", "late", "prevalence")}
                    for s in ("train", "val", "test")
                },
                "narrative_share": share,
                "low_narrative_share": share["test"] < evaluate.LOW_NARRATIVE_SHARE,
            }
        )
    return {
        "data_end": windows_report["data_end"],
        "train_years": windows_report["train_years"],
        "low_narrative_share_rule": "share of the test period's complaints that carry a "
        f"published narrative below {evaluate.LOW_NARRATIVE_SHARE:.0%}",
        "windows": out,
    }


# --- companies ---------------------------------------------------------------------------


def companies(cfg: dict, window: str) -> dict:
    """Company explorer: volume and late rate over the window's labelled complaints, and the
    mean test score of the full LR and of the scorer."""
    stats = scorer.company_stats(cfg, window)
    con = duckdb.connect()
    con.execute(f"set memory_limit = '{cfg['duckdb_memory_limit']}'")
    scores = {}
    for model in ("lr", scorer.MODEL):
        path = models.pred_path(model, "both", window)
        if not path.exists():
            continue
        rows = con.execute(
            f"""
            select l.company, avg(p.score), count(*), sum(l.late::int)
            from read_parquet('{path}') p
            join read_parquet('{INTERIM / "labelled.parquet"}') l using (complaint_id)
            where p.split = 'test'
            group by l.company""",
        ).fetchall()
        scores[model] = {r[0]: r[1:] for r in rows}
    con.close()
    out = []
    for c in stats:
        test = scores.get("lr", {}).get(c["company"])
        out.append(
            {
                "company": c["name"],
                "complaints": c["complaints"],
                "late": c["late"],
                "late_rate": c["late_rate_observed"],
                "test_complaints": c["test_complaints"],
                "test_late": test[2] if test else 0,
                "mean_score": test[0] if test else None,
                "mean_score_scorer": scores.get(scorer.MODEL, {}).get(c["company"], [None])[0],
                "history_late_rate": c["late_rate"],
                "history_log_volume": c["log_volume"],
            }
        )
    return {
        "window": window,
        "min_complaints": cfg["scorer"]["company_min_complaints"],
        "about": [
            "Labelled complaints with a narrative across the window's train, validation and "
            "test splits.",
            "Mean scores are over the company's test complaints: LR on narrative and structured "
            "fields, and the in-browser scorer.",
        ],
        "history_about": "Company late rate (smoothed) and log volume over the 365 days ending "
        "30 days before the company's latest complaint, as the models see them.",
        "companies": out,
    }


# --- robustness, explanations, scorer ----------------------------------------------------


def robustness(report: dict) -> dict:
    return {k: report[k] for k in ("cross_product", "drift", "errors")}


def explain(report: dict) -> dict:
    keep = ("window", "featureset", "C", "test_rows", "sample_rows", "sample_late", "groups")
    return {
        "main_window": report["main_window"],
        "groups": report["groups"],
        "method": report["method"],
        "lr": {n: {k: v for k, v in c.items() if k in keep} for n, c in report["lr"].items()},
        "lgbm_shap": {
            n: {k: v for k, v in c.items() if k in keep} for n, c in report["lgbm_shap"].items()
        },
        "top_terms": report["top_terms"],
    }


def scorer_metrics(report: dict, window: str) -> dict:
    """The scorer's test metrics next to the full LR's, and their paired difference."""
    mine = models.cell_name(scorer.MODEL, scorer.FEATURESET, window)
    full = models.cell_name("lr", "both", window)
    cells = report["cells"]
    compare = next(
        (
            c["metrics"]
            for c in report["comparisons"]
            if c["window"] == window
            and c["a"] == [scorer.MODEL, "both"]
            and c["b"] == ["lr", "both"]
        ),
        None,
    )
    return {
        "scorer": {k: cells[mine][k] for k in ("point", "ci95")} if mine in cells else None,
        "full": {k: cells[full][k] for k in ("point", "ci95")} if full in cells else None,
        "scorer_minus_full": compare,
    }


def scorer_info(report: dict, fit: dict, bundle: dict, size: int) -> dict:
    return {
        "window": fit["window"],
        "train_dates": bundle["train_dates"],
        "C": fit["C"],
        "C_whole_vocabulary": fit["full"]["C"],
        "terms_whole_vocabulary": fit["full"]["terms"],
        "terms": len(bundle["vocabulary"]),
        "bundle_bytes": size,
        "numeric": fit["numeric"],
        "left_out": ["vader"],
        "baseline_rule": bundle["categorical"]["rule"],
        "company_default_rule": bundle["company_default_rule"],
        "companies": len(bundle["companies"]),
        "days_to_company_default": bundle["defaults"]["days_to_company"],
        "text_check_max_gap": fit["text_check_max_gap"],
        "rounding_max_gap": fit["rounding_max_gap"],
        "metrics": scorer_metrics(report, fit["window"]),
    }


# --- manifest and v1 history --------------------------------------------------------------


def manifest(report: dict) -> dict:
    """The raw files in download order: archive ZIPs by number, then the live file."""

    def order(name: str) -> tuple:
        parts = name.split("_")
        return (0, int(parts[2])) if name.startswith("CCDB_Export_") else (1, 0)

    files = [
        {"name": name, **{k: report[name][k] for k in ("url", "bytes", "sha256", "downloaded")}}
        for name in sorted(report, key=order)
    ]
    return {"files": files}


def from_tracked(reports: Path = REPORTS, archive: Path = ARCHIVE) -> dict:
    """Site files built only from tracked inputs (reports/ and archive/), so a test can rebuild
    them and compare with docs/data."""
    report = read(reports / "metrics.json")
    windows_report = read(reports / "windows.json")
    curves_dir = reports / "curves"
    return {
        "metrics.json": metrics_grid(report),
        "curves.json": all_curves(curves_dir, report["cells"]),
        "calibration.json": calibration(curves_dir, report),
        "timeline.json": timeline(read(reports / "data_profile.json"), windows_report),
        "windows.json": windows_info(windows_report),
        "robustness.json": robustness(read(reports / "robustness.json")),
        "explain.json": explain(read(reports / "explain.json")),
        "manifest.json": manifest(read(reports / "data_manifest.json")),
        "v1.json": read(archive / "v1.json"),
    }


def main(cfg: dict) -> int:
    report = read(REPORTS / "metrics.json")
    windows_report = read(features.REPORT)
    window = windows_report["windows"][-1]["window"]
    sizes = {name: write(DATA / name, obj) for name, obj in from_tracked().items()}
    sizes["companies.json"] = write(DATA / "companies.json", companies(cfg, window))

    bundle = read(scorer.BUNDLE)
    if bundle["window"] != window:
        raise SystemExit(f"export: the scorer bundle is for {bundle['window']}, not {window}")
    fit = read(models.MODELS / f"{models.cell_name(scorer.MODEL, scorer.FEATURESET, window)}.json")
    SCORER_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scorer.BUNDLE, SCORER_DIR / "model.json")
    size = (SCORER_DIR / "model.json").stat().st_size
    sizes["scorer.json"] = write(DATA / "scorer.json", scorer_info(report, fit, bundle, size))
    for name, n in sizes.items():
        print(f"export: docs/data/{name} {n / 1e3:,.0f} kB")
    print(f"export: docs/scorer/model.json {size / 1e6:.2f} MB")
    return 0
