"""Robustness checks on the latest window, plus drift across every window's test period.

cross_product   LR trained on one product's training rows and scored on each product's test
                rows, for the products with the most test rows in the latest window. Every run
                loads its own feature matrices and fits its own Pipeline, in a fresh process.
                C is chosen on the source product's validation rows. References: in-domain
                (the diagonal), chance (each target's test prevalence) and the window's pooled
                LR (trained on every product) scored on the same target rows. The TF-IDF
                vocabulary and idf come from the window's whole training year (no labels).
drift           per test month of every window: rows, prevalence, and each model's mean score
                and PR-AUC.
errors          the latest window's LR (`both`) at its validation-chosen F2 threshold: counts
                of false positives and false negatives by product x issue, at the threshold
                and among the most confident mistakes. Categories only; no narrative text.
Runs are cached in data/robustness/ so an interrupted stage resumes. Writes
reports/robustness.json.
"""

import datetime as dt
import json
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import duckdb
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

from cres import evaluate, features, models

OUT = Path("reports/robustness.json")
CACHE = Path("data/robustness")
N_PRODUCTS = 5
CROSS_MODEL = "lr"
ERROR_MODEL = ("lr", "both")
ERROR_RULE = "f2"
MOST_CONFIDENT = 500
TOP_CELLS = 15
DRIFT_MODELS = [
    ("lr", "both"),
    ("lr", "struct"),
    ("lr", "text"),
    ("lgbm", "both"),
    ("lgbm", "struct"),
    ("company_rate", "struct"),
]


# --- products ----------------------------------------------------------------------------


def product_rows(window: str, split_name: str) -> dict[str, np.ndarray]:
    """Product -> row positions in the split, read from the product one-hot columns."""
    d = features.FEATURES / window
    names = json.loads((d / "meta.json").read_text("utf-8"))["categorical_names"]
    cat = sp.load_npz(d / f"cat_{split_name}.npz").tocsc()
    out = {}
    for j, n in enumerate(names):
        if n.startswith("product_") and n != "product_infrequent_sklearn":
            col = cat[:, j]
            out[n.removeprefix("product_")] = np.sort(col.indices[col.data == 1])
    return out


def largest(counts: dict[str, int], n: int) -> list[str]:
    """The n keys with the largest counts (ties by name)."""
    return sorted(counts, key=lambda k: (-counts[k], k))[:n]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --- scoring ------------------------------------------------------------------------------


def score_cell(scores: np.ndarray, y: np.ndarray, resamples: int, seed: list[int]) -> dict:
    """PR-AUC and AUROC with stratified-bootstrap 95% CIs. The bootstrap depends only on
    (seed, y), so every source scored on the same target shares its resamples."""
    out = {"rows": int(len(y)), "late": int(y.sum()), "prevalence": float(y.mean())}
    if out["late"] == 0 or out["late"] == len(y):
        return out | {"pr_auc": None, "auroc": None, "ci95": None}
    p = evaluate.prepare(np.asarray(scores, np.float64), y, 1)
    point = evaluate.metrics(p)
    rng = np.random.default_rng(seed)
    draws = {"pr_auc": [], "auroc": []}
    for _ in range(resamples):
        w = evaluate.bootstrap_weights(y, rng)
        m = evaluate.metrics(p, w[p["order"]])
        for k in draws:
            draws[k].append(m[k])
    return out | {
        "pr_auc": point["pr_auc"],
        "auroc": point["auroc"],
        "ci95": {k: evaluate.ci(np.array(v)) for k, v in draws.items()},
    }


def target_seed(seed: int, window: str, target: int) -> list[int]:
    return [*evaluate.window_seed(seed, window), target]


def fit_source(window: str, featureset: str, source: str, products: list[str], cfg: dict) -> dict:
    """Fit LR on the source product's training rows (C chosen on its validation rows) and
    score every product's test rows. Builds all feature matrices fresh for this run."""
    t0 = time.time()
    seed = cfg["seed"]
    params = cfg["models"]["lr"]
    train = features.load(window, featureset, "train", rows=product_rows(window, "train")[source])
    val = features.load(window, featureset, "val", rows=product_rows(window, "val")[source])
    numeric_names = [train["names"][i].removeprefix("num:") for i in train["numeric_cols"]]
    by_c = {}

    def score(C):
        pipe = models.fit_lr(
            train["X"], train["y"], train["numeric_cols"], numeric_names, C, params, False, seed
        )
        ap = float(average_precision_score(val["y"], pipe.predict_proba(val["X"])[:, 1]))
        by_c[str(C)] = ap
        print(f"    {source[:30]} C={C}: val AP {ap:.4f}", flush=True)
        return ap, pipe

    best_c, pipe = models.search_c(params["C"], score)
    out = {
        "source": source,
        "C": best_c,
        "val_average_precision": by_c[str(best_c)],
        "train_rows": int(len(train["y"])),
        "train_late": int(train["y"].sum()),
    }
    train = val = None  # free the source rows before loading test
    test = features.load(window, featureset, "test")
    scores = pipe.predict_proba(test["X"])[:, 1]
    rows = product_rows(window, "test")
    resamples = cfg["evaluate"]["bootstrap_resamples"]
    out["targets"] = {
        t: score_cell(scores[rows[t]], test["y"][rows[t]], resamples, target_seed(seed, window, i))
        for i, t in enumerate(products)
    }
    out["seconds"] = time.time() - t0
    return out


def matrix(products: list[str], runs: dict[str, dict], pooled: dict[str, dict]) -> dict:
    """Assemble the source x target matrix with its references.

    runs[source]["targets"][target] and pooled[target] are score_cell results."""
    cells = [[runs[a]["targets"][b] for b in products] for a in products]
    return {
        "rows_are": "training product",
        "columns_are": "test product",
        "pr_auc": [[c["pr_auc"] for c in row] for row in cells],
        "pr_auc_ci95": [[c["ci95"] and c["ci95"]["pr_auc"] for c in row] for row in cells],
        "auroc": [[c["auroc"] for c in row] for row in cells],
        "in_domain": [cells[i][i]["pr_auc"] for i in range(len(products))],
        "chance": [runs[products[0]]["targets"][b]["prevalence"] for b in products],
        "pooled": [pooled[b]["pr_auc"] for b in products],
        "pooled_ci95": [pooled[b]["ci95"] and pooled[b]["ci95"]["pr_auc"] for b in products],
        "test": [
            {k: runs[products[0]]["targets"][b][k] for k in ("rows", "late", "prevalence")}
            for b in products
        ],
        "train": [
            {k: runs[a][k] for k in ("train_rows", "train_late", "C", "val_average_precision")}
            for a in products
        ],
    }


def test_scores(model: str, featureset: str, window: str, ids: np.ndarray) -> np.ndarray | None:
    """Test scores from a prediction file, in the order of `ids`; None if there is no file."""
    path = models.pred_path(model, featureset, window)
    if not path.exists():
        return None
    pid, s = models.read_preds(path)["test"]
    pos = np.searchsorted(pid, ids)
    if not np.array_equal(pid[np.minimum(pos, len(pid) - 1)], ids):
        raise RuntimeError(f"{path.name}: test ids don't match the {window} feature cache")
    return s[pos]


def cross_product(window: str, cfg: dict) -> dict:
    counts = {p: len(r) for p, r in product_rows(window, "test").items()}
    products = largest(counts, N_PRODUCTS)
    d = features.FEATURES / window
    ids, y = np.load(d / "ids_test.npy"), np.load(d / "y_test.npy")
    rows = product_rows(window, "test")
    resamples = cfg["evaluate"]["bootstrap_resamples"]
    out = {
        "window": window,
        "model": CROSS_MODEL,
        "products": products,
        "selection": f"the {N_PRODUCTS} products with the most test rows in {window}",
        "featuresets": {},
    }
    ctx = get_context("spawn")
    meta = d / "meta.json"
    for fs in features.FEATURE_SETS:
        runs = {}
        for i, source in enumerate(products):
            path = CACHE / window / f"{fs}__{slug(source)}.json"
            if not features.stale(path, [meta, Path(models.__file__), Path(__file__)]):
                print(f"robustness: {fs} {source} is current", flush=True)
                runs[source] = json.loads(path.read_text("utf-8"))
                continue
            print(f"robustness: {fs}, train on {source} ({i + 1}/{len(products)})", flush=True)
            with ProcessPoolExecutor(1, mp_context=ctx) as pool:
                run = pool.submit(fit_source, window, fs, source, products, cfg).result()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
            print(f"robustness: {fs} {source} done ({run['seconds']:.0f}s)", flush=True)
            runs[source] = run
        scores = test_scores(CROSS_MODEL, fs, window, ids)
        pooled = {
            t: score_cell(
                scores[rows[t]], y[rows[t]], resamples, target_seed(cfg["seed"], window, i)
            )
            for i, t in enumerate(products)
        }
        out["featuresets"][fs] = matrix(products, runs, pooled)
    return out


# --- drift and errors ----------------------------------------------------------------------


def categories(window: str, cols: list[str], cfg: dict) -> dict[str, np.ndarray]:
    """Columns of data/features/rows.parquet for the window's test rows, in feature order."""
    ids = np.load(features.FEATURES / window / "ids_test.npy")
    test = {"complaint_id": ids, "pos": np.arange(len(ids))}  # noqa: F841 (scanned by DuckDB)
    con = duckdb.connect()
    con.execute(f"set memory_limit = '{cfg['duckdb_memory_limit']}'")
    rows = con.execute(f"""
        select t.pos, {", ".join("r." + c for c in cols)}
        from test t join read_parquet('{features.FEATURES / "rows.parquet"}') r using (complaint_id)
        order by t.pos""").fetchnumpy()
    con.close()
    if len(rows["pos"]) != len(ids):
        raise RuntimeError(f"{window}: {len(ids) - len(rows['pos'])} test rows not in rows.parquet")
    return {c: np.asarray(rows[c]) for c in cols}


def monthly(months: np.ndarray, y: np.ndarray, scores: dict[str, np.ndarray]) -> list[dict]:
    """Per month: rows, late, prevalence, and each model's mean score and PR-AUC."""
    out = []
    for m in sorted(set(months)):
        idx = months == m
        row = {"month": m, "rows": int(idx.sum()), "late": int(y[idx].sum())}
        row["prevalence"] = row["late"] / row["rows"]
        for name, s in scores.items():
            row[f"{name}:mean_score"] = float(s[idx].mean())
            row[f"{name}:pr_auc"] = (
                float(average_precision_score(y[idx], s[idx]))
                if 0 < row["late"] < row["rows"]
                else None
            )
        out.append(row)
    return out


def drift(windows: list[str], cfg: dict) -> dict:
    out = {"models": [f"{m}__{fs}" for m, fs in DRIFT_MODELS], "windows": {}}
    for w in windows:
        d = features.FEATURES / w
        ids, y = np.load(d / "ids_test.npy"), np.load(d / "y_test.npy")
        dates = categories(w, ["date_received"], cfg)["date_received"]
        months = np.array([str(x)[:7] for x in dates])
        scores = {}
        for m, fs in DRIFT_MODELS:
            s = test_scores(m, fs, w, ids)
            if s is not None:
                scores[f"{m}__{fs}"] = s
        out["windows"][w] = monthly(months, y, scores)
    return out


def crosstab(
    product: np.ndarray, issue: np.ndarray, mask: np.ndarray, y: np.ndarray, top: int
) -> list:
    """The `top` product x issue cells among rows in `mask`, by count, with each cell's size
    and late count over the whole test split for context."""
    picked = Counter(zip(product[mask], issue[mask], strict=True))
    everyone = Counter(zip(product, issue, strict=True))
    late = Counter(zip(product[y == 1], issue[y == 1], strict=True))
    n = int(mask.sum())
    return [
        {
            "product": p,
            "issue": i,
            "count": c,
            "share": c / n,
            "test_rows": everyone[(p, i)],
            "test_late": late[(p, i)],
        }
        for (p, i), c in sorted(picked.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    ]


def error_cells(
    y: np.ndarray, score: np.ndarray, threshold: float, product: np.ndarray, issue: np.ndarray
) -> dict:
    """False positives and negatives at `threshold` (flag score >= threshold), by category,
    plus the MOST_CONFIDENT highest-scored on-time and lowest-scored late complaints."""
    flagged = score >= threshold
    fp, fn = flagged & (y == 0), ~flagged & (y == 1)
    neg, pos = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    top_fp = np.zeros(len(y), bool)
    top_fp[neg[np.argsort(-score[neg], kind="mergesort")[:MOST_CONFIDENT]]] = True
    top_fn = np.zeros(len(y), bool)
    top_fn[pos[np.argsort(score[pos], kind="mergesort")[:MOST_CONFIDENT]]] = True
    return {
        "threshold": float(threshold),
        "counts": {
            "tp": int((flagged & (y == 1)).sum()),
            "fp": int(fp.sum()),
            "fn": int(fn.sum()),
            "tn": int((~flagged & (y == 0)).sum()),
        },
        "false_positives": crosstab(product, issue, fp, y, TOP_CELLS),
        "false_negatives": crosstab(product, issue, fn, y, TOP_CELLS),
        "most_confident": MOST_CONFIDENT,
        "most_confident_false_positives": crosstab(product, issue, top_fp, y, TOP_CELLS),
        "most_confident_false_negatives": crosstab(product, issue, top_fn, y, TOP_CELLS),
    }


def errors(window: str, cfg: dict) -> dict:
    model, fs = ERROR_MODEL
    name = models.cell_name(model, fs, window)
    cell = json.loads(evaluate.METRICS.read_text("utf-8"))["cells"][name]
    d = features.FEATURES / window
    ids, y = np.load(d / "ids_test.npy"), np.load(d / "y_test.npy")
    cats = categories(window, ["product", "issue"], cfg)
    return {
        "model": name,
        "rule": ERROR_RULE,
        "rule_chosen_on": "validation",
        **error_cells(
            y,
            test_scores(model, fs, window, ids),
            cell["thresholds"][ERROR_RULE],
            cats["product"],
            cats["issue"],
        ),
    }


def main(cfg: dict, force: bool = False) -> int:
    # The prediction files it reads (the error threshold in metrics.json comes from them too).
    preds = {models.cell_name(m, fs, "*") for m, fs in DRIFT_MODELS}
    inputs = [
        *(p for pattern in sorted(preds) for p in models.PREDS.glob(f"{pattern}.parquet")),
        *features.FEATURES.glob("W*/meta.json"),
        features.REPORT,
    ]
    if not force and not features.stale(OUT, inputs):
        print(f"robustness: {OUT} is newer than its inputs, nothing to do")
        return 0
    windows = [w["window"] for w in json.loads(features.REPORT.read_text("utf-8"))["windows"]]
    latest = windows[-1]
    t0 = time.time()
    report = {
        "generated": dt.date.today().isoformat(),
        "cross_product": cross_product(latest, cfg),
        "drift": drift(windows, cfg),
        "errors": errors(latest, cfg),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"robustness: wrote {OUT} ({time.time() - t0:,.0f}s)", flush=True)
    return 0
