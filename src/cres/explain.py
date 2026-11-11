"""Explanations of the fitted models, by feature group. Refits nothing.

For every saved LR (models/lr__{featureset}__{window}.joblib):
  each group's contribution on the test split. A row's contribution from group g is
  w_g . (x_g - mean of x_g over test), using the Pipeline's scaled inputs, which is the exact
  SHAP value of a linear model with the test rows as background. The share of a group is its
  mean absolute contribution over the sum of all groups'.
For the latest window: the TF-IDF terms with the largest and smallest coefficients.
For every saved LightGBM booster: TreeSHAP (LightGBM's own pred_contrib) on a seeded sample of
test rows, summed within each group, then averaged in absolute value.
Writes reports/explain.json. Feature names and category values only; no narrative text.
"""

import datetime as dt
import json
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp

from cres import evaluate, features, models

OUT = Path("reports/explain.json")
SHAP_ROWS = 5000
TOP_TERMS = 25
TOP_FEATURES = 20
CHUNK = 1000

# Group -> what it holds. Order is the order reported.
GROUPS = {
    "text": "TF-IDF of narrative 1-2 grams",
    "lexicon": "urgency and harm phrase counts",
    "sentiment": "VADER compound score",
    "product": "product and sub-product",
    "issue": "issue and sub-issue",
    "company history": "company trailing late rate and log volume",
    "days to company": "days between receipt and sending to the company",
    "state": "consumer state",
    "channel": "how the complaint was submitted",
    "tags": "older American and servicemember tags",
}
NUMERIC_GROUPS = {
    "urgency": "lexicon",
    "harm": "lexicon",
    "vader": "sentiment",
    "company_late_rate": "company history",
    "company_log_volume": "company history",
    "days_to_company": "days to company",
}
CATEGORICAL_GROUPS = {
    "product": "product",
    "sub_product": "product",
    "issue": "issue",
    "sub_issue": "issue",
    "state": "state",
    "submitted_via": "channel",
    "tags": "tags",
}


def group_of(name: str) -> str:
    """The group of a feature name from features.load (tfidf:, cat:, num:)."""
    kind, _, rest = name.partition(":")
    if kind == "tfidf":
        return "text"
    if kind == "num" and rest in NUMERIC_GROUPS:
        return NUMERIC_GROUPS[rest]
    if kind == "cat":
        for col, group in CATEGORICAL_GROUPS.items():
            if rest.startswith(col + "_"):
                return group
    raise ValueError(f"feature {name!r} has no group")


def group_matrix(names: list[str], weights: np.ndarray | None = None) -> tuple[sp.csr_matrix, list]:
    """(n_features x n_groups) matrix with weights[j] (default 1) at feature j's group, and the
    groups present, in GROUPS order. X @ it sums each row's weighted features per group."""
    of = [group_of(n) for n in names]
    present = [g for g in GROUPS if g in set(of)]
    col = np.array([present.index(g) for g in of])
    w = np.ones(len(names)) if weights is None else np.asarray(weights, dtype=np.float64)
    return sp.csr_matrix(
        (w, (np.arange(len(names)), col)), shape=(len(names), len(present))
    ), present


def group_table(contrib: np.ndarray, groups: list[str], names: list[str], y: np.ndarray) -> list:
    """Per group: features, mean |contribution|, share of the total, and mean contribution
    among late and on-time rows. `contrib` is (rows x groups), already centered."""
    of = [group_of(n) for n in names]
    mean_abs = np.abs(contrib).mean(axis=0)
    total = mean_abs.sum()
    late = y.astype(bool)
    return [
        {
            "group": g,
            "features": of.count(g),
            "mean_abs": float(mean_abs[i]),
            "share": float(mean_abs[i] / total) if total else 0.0,
            "mean_late": float(contrib[late, i].mean()) if late.any() else None,
            "mean_on_time": float(contrib[~late, i].mean()) if (~late).any() else None,
        }
        for i, g in enumerate(groups)
    ]


def lr_groups(pipe, X, y: np.ndarray, names: list[str]) -> dict:
    """Group contributions of a fitted LR Pipeline on X (unscaled, as features.load returns)."""
    coef = pipe[-1].coef_[0]
    if len(coef) != len(names):
        raise ValueError(f"model has {len(coef)} coefficients for {len(names)} features")
    G, groups = group_matrix(names, coef)
    contrib = np.asarray((pipe[:-1].transform(X) @ G).todense())
    contrib -= contrib.mean(axis=0)
    return {
        "intercept": float(pipe[-1].intercept_[0]),
        "groups": group_table(contrib, groups, names, y),
    }


def top_terms(coef: np.ndarray, names: list[str], X, k: int = TOP_TERMS) -> dict:
    """The k TF-IDF terms with the largest and the smallest coefficients, with the number of
    test documents containing each."""
    cols = np.array([i for i, n in enumerate(names) if n.startswith("tfidf:")], dtype=np.int64)
    docs = np.diff(X.tocsc()[:, cols].indptr)
    order = np.argsort(coef[cols], kind="mergesort")

    def rows(idx):
        return [
            {
                "term": names[cols[i]].removeprefix("tfidf:"),
                "coef": float(coef[cols[i]]),
                "test_docs": int(docs[i]),
            }
            for i in idx
        ]

    return {"up": rows(order[::-1][:k]), "down": rows(order[:k])}


def shap_groups(booster, X, y: np.ndarray, names: list[str], n: int, seed: list[int]) -> dict:
    """TreeSHAP on a seeded sample of n rows of X, summed per group (log-odds units)."""
    if booster.num_feature() != len(names):
        raise ValueError(f"booster has {booster.num_feature()} features for {len(names)} names")
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False))
    G, groups = group_matrix(names)
    grouped, abs_sum, bias = [], np.zeros(len(names)), []
    for i in range(0, len(sample), CHUNK):
        Xs = X[sample[i : i + CHUNK]]
        c = booster.predict(Xs, pred_contrib=True)
        c = c.tocsr() if sp.issparse(c) else sp.csr_matrix(c)
        raw = booster.predict(Xs, raw_score=True)
        if not np.allclose(np.asarray(c.sum(axis=1)).ravel(), raw, atol=1e-3):
            raise RuntimeError("SHAP values don't add up to the raw score")
        feats, last = c[:, :-1], c[:, -1].toarray().ravel()
        grouped.append(np.asarray((feats @ G).todense()))
        abs_sum += np.asarray(abs(feats).sum(axis=0)).ravel()
        bias.append(last)
    grouped = np.vstack(grouped)
    mean_abs = abs_sum / len(sample)
    top = np.argsort(-mean_abs, kind="mergesort")[:TOP_FEATURES]
    return {
        "sample_rows": int(len(sample)),
        "sample_late": int(y[sample].sum()),
        "seed": seed,
        "expected_value": float(np.concatenate(bias).mean()),
        "groups": group_table(grouped, groups, names, y[sample]),
        "top_features": [
            {"feature": names[j], "group": group_of(names[j]), "mean_abs": float(mean_abs[j])}
            for j in top
        ],
    }


def main(cfg: dict, force: bool = False) -> int:
    import lightgbm as lgb

    # The LR and LightGBM fits it reads; a change to their features or config refits them.
    inputs = [*models.MODELS.glob("lr__*"), *models.MODELS.glob("lgbm__*"), features.REPORT]
    if not force and not features.stale(OUT, inputs):
        print(f"explain: {OUT} is newer than the fitted models, nothing to do")
        return 0
    windows = [w["window"] for w in json.loads(features.REPORT.read_text("utf-8"))["windows"]]
    latest = windows[-1]
    max_terms = cfg["models"]["lgbm"]["max_text_terms"]
    report = {
        "generated": dt.date.today().isoformat(),
        "main_window": latest,
        "groups": GROUPS,
        "method": {
            "lr": "mean |w_g . (x_g - test mean of x_g)| on the Pipeline's scaled inputs; "
            "share = group / sum of groups",
            "lgbm": f"LightGBM TreeSHAP (pred_contrib) on {SHAP_ROWS:,} seeded test rows, "
            "summed within each group; log-odds units",
        },
        "lr": {},
        "top_terms": {},
        "lgbm_shap": {},
    }
    for window in windows:
        for fs in features.FEATURE_SETS:
            name = models.cell_name("lr", fs, window)
            path = models.MODELS / f"{name}.joblib"
            if path.exists():
                t0 = time.time()
                pipe = joblib.load(path)
                d = features.load(window, fs, "test")
                meta = json.loads((models.MODELS / f"{name}.json").read_text("utf-8"))
                report["lr"][name] = {
                    "window": window,
                    "featureset": fs,
                    "C": meta["C"],
                    "test_rows": int(len(d["y"])),
                    **lr_groups(pipe, d["X"], d["y"], d["names"]),
                }
                if window == latest and fs != "struct":
                    report["top_terms"][name] = top_terms(pipe[-1].coef_[0], d["names"], d["X"])
                del pipe, d
                print(f"explain: {name} ({time.time() - t0:.0f}s)", flush=True)
            name = models.cell_name("lgbm", fs, window)
            path = models.MODELS / f"{name}.txt"
            if path.exists():
                t0 = time.time()
                booster = lgb.Booster(model_file=str(path))
                d = features.load(window, fs, "test", max_terms)
                report["lgbm_shap"][name] = {
                    "window": window,
                    "featureset": fs,
                    **shap_groups(
                        booster,
                        d["X"],
                        d["y"],
                        d["names"],
                        SHAP_ROWS,
                        evaluate.window_seed(cfg["seed"], window),
                    ),
                }
                del booster, d
                print(f"explain: {name} SHAP ({time.time() - t0:.0f}s)", flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0
