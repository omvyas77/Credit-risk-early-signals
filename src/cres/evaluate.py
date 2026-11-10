"""Score every prediction file in data/preds/.

For each file (model x feature set x window):
  - test metrics: PR-AUC, AUROC, log loss, Brier, ECE, precision and recall in the top 1% and
    5% of test scores, and the chance PR-AUC (test prevalence);
  - thresholds chosen on validation by three rules (F2-optimal, cost-optimal, capacity), then
    applied to test;
  - 95% CIs from a stratified bootstrap of the test split (resampling positives and negatives
    separately). Every file in a window uses the same resamples, so differences between
    files (for example `both` minus `struct`) get paired CIs.
Writes reports/metrics.json, reports/curves/*.json and reports/summary.md.
"""

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from cres import features, models

REPORTS = Path("reports")
METRICS = REPORTS / "metrics.json"
CURVES = REPORTS / "curves"
SUMMARY = REPORTS / "summary.md"
EPS = 1e-15
DIFF_METRICS = ("pr_auc", "auroc", "log_loss", "brier", "ece")


# --- metrics on score-sorted arrays ------------------------------------------------------


def prepare(score: np.ndarray, y: np.ndarray, ece_bins: int) -> dict:
    """Sort by descending score once; every metric below reads this, with optional weights."""
    order = np.argsort(-score, kind="mergesort")
    s = score[order].astype(np.float64)
    yy = y[order].astype(np.float64)
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last row of each distinct score
    bins = np.arange(len(s)) * ece_bins // len(s)  # equal-count bins over sorted scores
    return {"s": s, "y": yy, "ends": ends, "order": order, "bins": bins, "ece_bins": ece_bins}


def flagged_counts(p: dict, w: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    """Weighted TP, FP, FN, TN when rows with score >= threshold are flagged."""
    k = int(np.searchsorted(-p["s"], -threshold, side="right"))  # rows flagged
    wp = w * p["y"]
    tp, fl = wp[:k].sum(), w[:k].sum()
    pos, total = wp.sum(), w.sum()
    return tp, fl - tp, pos - tp, total - pos - (fl - tp)


def metrics(
    p: dict,
    w: np.ndarray | None = None,
    capacities: tuple = (),
    thresholds: dict | None = None,
    costs: dict | None = None,
) -> dict:
    """Metrics for one prepared score vector; w are per-row weights in sorted order."""
    s, y = p["s"], p["y"]
    w = np.ones_like(s) if w is None else w.astype(np.float64)
    wp = w * y
    cum_p, cum_w = np.cumsum(wp), np.cumsum(w)
    tp, fl = cum_p[p["ends"]], cum_w[p["ends"]]
    fp = fl - tp
    pos, total = tp[-1], fl[-1]
    neg = total - pos
    # Leading rows can carry zero bootstrap weight; nothing is flagged there, so precision 0
    # (its recall step is 0 too).
    precision = np.divide(tp, fl, out=np.zeros_like(tp), where=fl > 0)
    recall = tp / pos
    out = {
        "pr_auc": float(np.sum(np.diff(np.r_[0.0, recall]) * precision)),
        "auroc": float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fp / neg])),
        "prevalence": float(pos / total),
    }
    q = np.clip(s, EPS, 1 - EPS)
    out["log_loss"] = float(-np.sum(w * (y * np.log(q) + (1 - y) * np.log(1 - q))) / total)
    out["brier"] = float(np.sum(w * (s - y) ** 2) / total)
    nb = p["ece_bins"]
    bw = np.bincount(p["bins"], weights=w, minlength=nb)
    bs = np.bincount(p["bins"], weights=w * s, minlength=nb)
    by = np.bincount(p["bins"], weights=wp, minlength=nb)
    ok = bw > 0
    out["ece"] = float(np.sum(np.abs(bs[ok] - by[ok])) / total)
    for c in capacities:
        k = min(int(np.searchsorted(cum_w, c * total)), len(s) - 1)  # top c of the weight
        out[f"precision_top{c:g}"] = float(cum_p[k] / cum_w[k]) if cum_w[k] else 0.0
        out[f"recall_top{c:g}"] = float(cum_p[k] / pos)
    for rule, t in (thresholds or {}).items():
        tp_t, fp_t, fn_t, _ = flagged_counts(p, w, t)
        out[f"{rule}:precision"] = float(tp_t / (tp_t + fp_t)) if tp_t + fp_t else 0.0
        out[f"{rule}:recall"] = float(tp_t / pos)
        out[f"{rule}:flagged"] = float((tp_t + fp_t) / total)
        if costs:
            out[f"{rule}:cost_per_1000"] = float(
                1000 * (costs["fp"] * fp_t + costs["fn"] * fn_t) / total
            )
    return out


# --- thresholds chosen on validation -----------------------------------------------------


def choose_thresholds(score: np.ndarray, y: np.ndarray, capacities, costs: dict) -> dict:
    """Thresholds (flag score >= t) chosen on validation by each rule.

    f2:          maximises F2 (recall weighted four times precision)
    cost:        minimises fp_cost * FP + fn_cost * FN; may flag nothing (t = inf)
    capacity_c:  flags the top fraction c of validation scores
    """
    p = prepare(score, y, 1)
    s, ends = p["s"], p["ends"]
    tp = np.cumsum(p["y"])[ends]
    fl = ends + 1.0
    fp = fl - tp
    pos = tp[-1]
    precision, recall = tp / fl, tp / pos
    f2 = np.where(tp > 0, 5 * precision * recall / (4 * precision + recall + EPS), 0.0)
    out = {"f2": float(s[ends[np.argmax(f2)]])}
    cost = costs["fp"] * fp + costs["fn"] * (pos - tp)
    i = int(np.argmin(cost))
    out["cost"] = float(s[ends[i]]) if cost[i] < costs["fn"] * pos else float("inf")
    for c in capacities:
        k = max(int(np.ceil(c * len(s))), 1)
        out[f"capacity_{c:g}"] = float(s[k - 1])
    return out


# --- bootstrap ---------------------------------------------------------------------------


def bootstrap_weights(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stratified bootstrap as row weights: positives and negatives resampled separately."""
    w = np.zeros(len(y), dtype=np.float64)
    for idx in (np.flatnonzero(y == 1), np.flatnonzero(y == 0)):
        w[idx] = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx))
    return w


def ci(values: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def window_seed(seed: int, window: str) -> list[int]:
    return [seed, int(window.lstrip("W"))]


# --- curves ------------------------------------------------------------------------------


def curves(p: dict, val: dict, thresholds: dict, n_points: int = 200) -> dict:
    """Downsampled PR points and reliability bins on test, and confusion counts on validation
    and test over a grid of thresholds taken from validation scores."""
    s, y, ends = p["s"], p["y"], p["ends"]
    tp = np.cumsum(y)[ends]
    fl = ends + 1.0
    recall, precision = tp / tp[-1], tp / fl
    keep = np.unique(np.searchsorted(recall, np.linspace(0, 1, n_points), side="left"))
    keep = keep[keep < len(ends)]
    pr = [
        {
            "recall": float(recall[i]),
            "precision": float(precision[i]),
            "threshold": float(s[ends[i]]),
        }
        for i in keep
    ]
    nb = p["ece_bins"]
    cnt = np.bincount(p["bins"], minlength=nb)
    rel = [
        {
            "mean_score": float(np.bincount(p["bins"], weights=s, minlength=nb)[b] / cnt[b]),
            "observed": float(np.bincount(p["bins"], weights=y, minlength=nb)[b] / cnt[b]),
            "n": int(cnt[b]),
        }
        for b in range(nb)
        if cnt[b]
    ]
    # Thresholds at validation flag rates from 0.1% to 50%, plus the rule thresholds.
    vs = val["s"]
    rates = np.geomspace(0.001, 0.5, 60)
    grid = {float(vs[max(int(np.ceil(r * len(vs))), 1) - 1]) for r in rates}
    grid |= {t for t in thresholds.values() if np.isfinite(t)}
    ones_t, ones_v = np.ones_like(s), np.ones_like(vs)
    table = []
    for t in sorted(grid, reverse=True):
        row = {"threshold": t}
        for name, prep, w in (("val", val, ones_v), ("test", p, ones_t)):
            row[name] = dict(
                zip(("tp", "fp", "fn", "tn"), map(int, flagged_counts(prep, w, t)), strict=True)
            )
        table.append(row)
    return {"pr": pr, "reliability": rel, "thresholds": table}


# --- stage -------------------------------------------------------------------------------


def labels(window: str, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    d = features.FEATURES / window
    ids = np.load(d / f"ids_{split_name}.npy")
    order = np.argsort(ids)
    return ids[order], np.load(d / f"y_{split_name}.npy")[order]


def parse_name(path: Path) -> tuple[str, str, str]:
    model, featureset, window = path.stem.split("__")
    return model, featureset, window


def evaluate_window(window: str, paths: list[Path], cfg: dict) -> tuple[dict, dict]:
    """Point metrics, CIs and curves for every file in one window, plus bootstrap draws."""
    ecfg = cfg["evaluate"]
    caps = tuple(ecfg["capacity"])
    costs = ecfg["costs"]
    ids = {s: labels(window, s) for s in ("val", "test")}
    results, draws, preps = {}, {}, {}
    for path in paths:
        preds = models.read_preds(path)
        for s in ("val", "test"):
            if not np.array_equal(preds[s][0], ids[s][0]):
                raise RuntimeError(f"{path.name}: {s} ids don't match the {window} feature cache")
        y_val, y_test = ids["val"][1], ids["test"][1]
        thresholds = choose_thresholds(preds["val"][1], y_val, caps, costs)
        p = prepare(preds["test"][1], y_test, ecfg["ece_bins"])
        pv = prepare(preds["val"][1], y_val, 1)
        point = metrics(p, None, caps, thresholds, costs)
        results[path.stem] = {"point": point, "thresholds": thresholds}
        preps[path.stem] = (p, thresholds)
        CURVES.mkdir(parents=True, exist_ok=True)
        (CURVES / f"{path.stem}.json").write_text(
            json.dumps(curves(p, pv, thresholds)) + "\n", encoding="utf-8"
        )
    y_test = ids["test"][1]
    rng = np.random.default_rng(window_seed(cfg["seed"], window))
    for _ in range(ecfg["bootstrap_resamples"]):
        w = bootstrap_weights(y_test, rng)  # in complaint_id order, shared by every file
        for name, (p, thresholds) in preps.items():
            m = metrics(p, w[p["order"]], caps, thresholds, costs)
            for k, v in m.items():
                draws.setdefault(name, {}).setdefault(k, []).append(v)
    for name, d in draws.items():
        results[name]["ci95"] = {k: ci(np.array(v)) for k, v in d.items() if k != "prevalence"}
    return results, {n: {k: np.array(v) for k, v in d.items()} for n, d in draws.items()}


def compare(draws: dict, point: dict, a: str, b: str) -> dict | None:
    """Paired difference a - b for each metric, with a bootstrap CI."""
    if a not in draws or b not in draws:
        return None
    out = {}
    for k in DIFF_METRICS:
        out[k] = {
            "diff": point[a]["point"][k] - point[b]["point"][k],
            "ci95": ci(draws[a][k] - draws[b][k]),
        }
    return out


def comparisons(window: str, draws: dict, results: dict) -> list[dict]:
    out = []
    model_names = sorted({n.split("__")[0] for n in results})
    pairs = []
    for m in model_names:  # what text adds, and what the structured fields add
        pairs += [(m, "both", m, "struct"), (m, "both", m, "text"), (m, "struct", m, "text")]
    for m in ("lr", "lgbm"):  # what everything else adds to the company's own record
        pairs += [(m, fs, "company_rate", "struct") for fs in features.FEATURE_SETS]
    for fs in features.FEATURE_SETS:
        pairs += [("lgbm", fs, "lr", fs), ("lr_balanced", fs, "lr", fs)]
        pairs += [(c, fs, "lr", fs) for c in models.CALIBRATED]
    for ma, fa, mb, fb in pairs:
        a, b = models.cell_name(ma, fa, window), models.cell_name(mb, fb, window)
        d = compare(draws, results, a, b)
        if d:
            out.append({"window": window, "a": [ma, fa], "b": [mb, fb], "metrics": d})
    return out


def across_windows(cells: dict) -> dict:
    """Mean and sd across windows of each point metric, per model and feature set."""
    groups: dict = {}
    for name, r in cells.items():
        model, fs, _ = name.split("__")
        groups.setdefault(f"{model}__{fs}", []).append(r["point"])
    out = {}
    for key, points in sorted(groups.items()):
        out[key] = {"n_windows": len(points)}
        for k in points[0]:
            v = np.array([p[k] for p in points])
            sd = float(v.std(ddof=1)) if len(v) > 1 else None
            out[key][k] = {"mean": float(v.mean()), "sd": sd}
    return out


def training_log() -> dict:
    out = {}
    for p in sorted(models.MODELS.glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        m.pop("val_average_precision_by_round", None)
        out[p.stem] = m
    return out


def main(cfg: dict) -> int:
    ecfg = cfg["evaluate"]
    if not ecfg.get("costs"):
        raise SystemExit("evaluate: set evaluate.costs.fp and .fn in config")
    paths = sorted(models.PREDS.glob("*.parquet"))
    if not paths:
        raise SystemExit("evaluate: no prediction files in data/preds/")
    by_window: dict = {}
    for path in paths:
        by_window.setdefault(parse_name(path)[2], []).append(path)
    windows_report = json.loads(features.REPORT.read_text(encoding="utf-8"))
    cells, comps = {}, []
    for window, wpaths in sorted(by_window.items()):
        t0 = time.time()
        results, draws = evaluate_window(window, wpaths, cfg)
        cells.update(results)
        comps += comparisons(window, draws, results)
        print(f"evaluate: {window}, {len(wpaths)} files ({time.time() - t0:,.0f}s)", flush=True)
    report = {
        "generated": dt.date.today().isoformat(),
        "bootstrap_resamples": ecfg["bootstrap_resamples"],
        "ece_bins": ecfg["ece_bins"],
        "capacity": ecfg["capacity"],
        "costs": ecfg["costs"],
        "models": cfg.get("models", {}),
        "windows": {
            w["window"]: {
                "dates": w["dates"],
                "test_prevalence": w["test"]["prevalence"],
                "val_prevalence": w["val"]["prevalence"],
                "test_rows": w["test"]["rows"],
                "narrative_share_test": w["narrative_share"]["test"],
            }
            for w in windows_report["windows"]
        },
        "cells": cells,
        "across_windows": across_windows(cells),
        "comparisons": comps,
        "training": training_log(),
    }
    METRICS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    SUMMARY.write_text(summary(report), encoding="utf-8")
    return 0


# --- summary ------------------------------------------------------------------------------

MODEL_LABELS = {
    "lr": "LR",
    "lgbm": "LightGBM",
    "lr_balanced": "LR, class-weighted",
    "lr_balanced_platt": "LR, class-weighted + Platt",
    "lr_balanced_isotonic": "LR, class-weighted + isotonic",
    "company_rate": "Company late rate alone",
}


def _ci(cell: dict, k: str, digits: int = 3) -> str:
    lo, hi = cell["ci95"][k]
    return f"{cell['point'][k]:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def _diff(d: dict, digits: int = 3) -> str:
    lo, hi = d["ci95"]
    return f"{d['diff']:+.{digits}f} ({lo:+.{digits}f} to {hi:+.{digits}f})"


def _find(report: dict, window: str, a: list, b: list) -> dict | None:
    for c in report["comparisons"]:
        if c["window"] == window and c["a"] == a and c["b"] == b:
            return c["metrics"]
    return None


def training_caveats(report: dict) -> list[str]:
    """Bullets on how the fits ended: LightGBM stopping rounds and LR C at the grid's edge."""
    runs = report.get("training", {}).values()
    out = []
    rounds = sorted(m["best_iteration"] for m in runs if m.get("best_iteration") is not None)
    if rounds:
        out.append(
            f"- LightGBM's best validation round ranges from {rounds[0]} to {rounds[-1]} "
            f"(median {rounds[len(rounds) // 2]}) out of up to "
            f"{report['models']['lgbm']['num_boost_round']:,}: validation average precision "
            "peaks early and then falls, so these trees fit the training year more than the "
            "later quarter. Its parameters were not tuned."
        )
    grid = report.get("models", {}).get("lr", {}).get("C")
    if grid:
        edge = sorted(
            f"{m['model']} {m['featureset']} {m['window']} (C={m['C']:g})"
            for m in runs
            if m.get("C") in (min(grid), max(grid))
        )
        if edge:
            out.append(
                "- LR's chosen C sits at the edge of the searched grid "
                f"({min(grid):g} to {max(grid):g}) for: {', '.join(edge)}."
            )
    return out


def summary(report: dict) -> str:
    """reports/summary.md, every number read from the metrics report."""
    cells, wins = report["cells"], report["windows"]
    windows = sorted({n.split("__")[2] for n in cells})
    fitted = [m for m in ("lr", "lgbm") if any(n.startswith(m + "__") for n in cells)]
    out = [
        "# Results summary",
        "",
        "Generated by `python -m cres evaluate` from `reports/metrics.json`; do not edit by hand.",
        "",
        "**Question.** Does what a consumer writes help predict a late company response, beyond "
        "knowing the company, product and issue?",
        "",
    ]
    # The answer: both - struct, paired, per model.
    lines = []
    for m in fitted:
        diffs = [_find(report, w, [m, "both"], [m, "struct"]) for w in windows]
        diffs = [d["pr_auc"] for d in diffs if d]
        if not diffs:
            continue
        mean = np.mean([d["diff"] for d in diffs])
        above = sum(d["ci95"][0] > 0 for d in diffs)
        below = sum(d["ci95"][1] < 0 for d in diffs)
        lines.append(
            f"- {MODEL_LABELS[m]}: adding the narrative to the structured fields changes test "
            f"PR-AUC by {mean:+.3f} on average across {len(diffs)} windows; the paired 95% CI is "
            f"above zero in {above}, below zero in {below} and spans zero in "
            f"{len(diffs) - above - below}."
        )
        means = {
            fs: report["across_windows"].get(f"{m}__{fs}", {}).get("pr_auc", {}).get("mean")
            for fs in features.FEATURE_SETS
        }
        if all(v is not None for v in means.values()):
            lines.append(
                f"  Mean test PR-AUC: text {means['text']:.3f}, struct {means['struct']:.3f}, "
                f"both {means['both']:.3f}."
            )
    chance = np.mean([wins[w]["test_prevalence"] for w in windows])
    out += [
        "**Answer, in numbers.**",
        "",
        *lines,
        "",
        f"Chance PR-AUC (test prevalence) averages {chance:.3f}.",
        "",
    ]
    base = report["across_windows"].get("company_rate__struct", {}).get("pr_auc")
    if base:
        out += [
            "**Where the signal comes from.** Scoring complaints by the company's own trailing "
            f"late rate alone, with no model, gives a mean test PR-AUC of {base['mean']:.3f}. "
            "That column counts only the company's complaints received at least 30 days before "
            "the one being scored (checked by a unit test and by recomputing sampled rows from "
            "the live file). A few companies miss the deadline on most of their complaints, so "
            "high PR-AUCs here mostly reflect the company, not leakage.",
            "",
        ]

    for m in fitted:
        out += [
            f"## {MODEL_LABELS[m]}: test PR-AUC by window (95% bootstrap CI)",
            "",
            "| Window | Test period | Test rows | Prevalence (chance) | Narrative share "
            "| Company late rate alone | text | struct | both | both - struct |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for w in windows:
            row = [
                w,
                " to ".join(wins[w]["dates"]["test"]),
                f"{wins[w]['test_rows']:,}",
                f"{wins[w]['test_prevalence']:.3f}",
                f"{wins[w]['narrative_share_test']:.1%}",
            ]
            base = cells.get(models.cell_name("company_rate", "struct", w))
            row.append(_ci(base, "pr_auc") if base else "not run")
            for fs in features.FEATURE_SETS:
                c = cells.get(models.cell_name(m, fs, w))
                row.append(_ci(c, "pr_auc") if c else "not run")
            d = _find(report, w, [m, "both"], [m, "struct"])
            row.append(_diff(d["pr_auc"]) if d else "")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    # AUROC for the same cells, compactly.
    out += [
        "## Test AUROC by window",
        "",
        "| Window | "
        + " | ".join(f"{MODEL_LABELS[m]} {fs}" for m in fitted for fs in features.FEATURE_SETS)
        + " |",
        "|---" * (1 + len(fitted) * 3) + "|",
    ]
    for w in windows:
        row = [w]
        for m in fitted:
            for fs in features.FEATURE_SETS:
                c = cells.get(models.cell_name(m, fs, w))
                row.append(f"{c['point']['auroc']:.3f}" if c else "")
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # Class weighting and recalibration.
    variants = [
        m
        for m in ("lr", "lr_balanced", "lr_balanced_platt", "lr_balanced_isotonic")
        if f"{m}__both" in report["across_windows"]
    ]
    if len(variants) > 1:
        out += [
            "## Class weighting and recalibration (`both`, mean ± sd across windows)",
            "",
            'Weighted LR is fitted with `class_weight="balanced"`; Platt and isotonic '
            "recalibrate it on validation.",
            "",
            "| Model | PR-AUC | Log loss | Brier | ECE |",
            "|---|---|---|---|---|",
        ]
        for m in variants:
            a = report["across_windows"][f"{m}__both"]
            out.append(
                f"| {MODEL_LABELS[m]} | "
                + " | ".join(
                    f"{a[k]['mean']:.4f} ± {a[k]['sd']:.4f}"
                    if a[k]["sd"] is not None
                    else f"{a[k]['mean']:.4f}"
                    for k in ("pr_auc", "log_loss", "brier", "ece")
                )
                + " |"
            )
        out.append("")

    # Thresholds chosen on validation, applied to test.
    # The model with the higher mean PR-AUC on `both`.
    both = {
        m: report["across_windows"].get(f"{m}__both", {}).get("pr_auc", {}).get("mean")
        for m in fitted
    }
    both = {m: v for m, v in both.items() if v is not None}
    if both:
        best = max(both, key=both.get)
        present = [w for w in windows if models.cell_name(best, "both", w) in cells]
        rules = list(cells[models.cell_name(best, "both", present[0])]["thresholds"])
        out += [
            f"## Thresholds chosen on validation, applied to test ({MODEL_LABELS[best]}, `both`)",
            "",
            f"Costs: false positive {report['costs']['fp']}, false negative "
            f"{report['costs']['fn']}.",
            "",
            "| Window | Rule | Flagged share | Precision | Recall |",
            "|---|---|---|---|---|",
        ]
        for w in present:
            c = cells[models.cell_name(best, "both", w)]
            for r in rules:
                out.append(
                    f"| {w} | {r} | {c['point'][f'{r}:flagged']:.3%} | "
                    f"{_ci(c, f'{r}:precision')} | {_ci(c, f'{r}:recall')} |"
                )
        out.append("")

    shifted = [w for w in windows if wins[w]["narrative_share_test"] < 0.1]
    out += [
        "## Read with care",
        "",
        f"- Windows {', '.join(shifted) or 'none'} test on complaints from the period when only "
        "a small share of complaints carried a published narrative, and that subset is late "
        "far more often; compare each window's PR-AUC with its own prevalence.",
        "- Each window trains on the most recent year before its training cutoff.",
        "- LR's C is chosen on validation average precision; LightGBM early-stops on validation "
        "average precision. Test is scored once.",
        *training_caveats(report),
        *(
            [
                f"- LightGBM sees the {terms:,} most frequent TF-IDF terms; LR sees the whole "
                "vocabulary."
            ]
            if (terms := report.get("models", {}).get("lgbm", {}).get("max_text_terms"))
            else []
        ),
        f"- CIs: stratified bootstrap of the test split, {report['bootstrap_resamples']:,} "
        "resamples; differences are paired on the same resamples.",
        "",
    ]
    return "\n".join(out)
