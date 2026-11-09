"""Classical models: logistic regression and LightGBM, one prediction file per cell.

A cell is (model, feature set, window). Each cell writes
  data/preds/{model}__{featureset}__{window}.parquet   complaint_id, split (val|test), score
  models/{model}__{featureset}__{window}.joblib|.txt   the fitted model
  models/{model}__{featureset}__{window}.json          fit time, iterations, peak memory
Models:
  lr                    logistic regression, unweighted
  lr_balanced           logistic regression with class_weight="balanced"
  lgbm                  LightGBM, early-stopped on the validation split
  lr_balanced_platt     lr_balanced recalibrated on validation (Platt scaling)
  lr_balanced_isotonic  lr_balanced recalibrated on validation (isotonic regression)
  company_rate          baseline, no fitting: the company's trailing late rate as the score
                        (feature set `struct`, where that column lives)
The recalibrated models are fitted on validation scores only. Their validation scores are
cross-fitted (each fold calibrated by the others) so thresholds chosen on validation don't
see their own labels; their test scores use a calibrator fitted on all of validation.
"""

import json
import os
import resource
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import duckdb
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from cres import features

PREDS = Path("data/preds")
MODELS = Path("models")
FITTED = ("lr", "lr_balanced", "lgbm")
CALIBRATED = {"lr_balanced_platt": "platt", "lr_balanced_isotonic": "isotonic"}
BASELINES = ("company_rate",)
ALL_MODELS = BASELINES + FITTED + tuple(CALIBRATED)


def cell_name(model: str, featureset: str, window: str) -> str:
    return f"{model}__{featureset}__{window}"


def pred_path(model: str, featureset: str, window: str) -> Path:
    return PREDS / f"{cell_name(model, featureset, window)}.parquet"


# --- logistic regression -----------------------------------------------------------------


# Heavy-tailed numeric columns are reshaped before standardising, so a few extreme rows
# don't dominate the fit or the solver's conditioning.
NUMERIC_TRANSFORMS = {
    "days_to_company": "log1p",
    "urgency": "log1p",
    "harm": "log1p",
    "company_late_rate": "logit",
}


def _reshape(values: np.ndarray, how: str) -> np.ndarray:
    if how == "log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    if how == "logit":
        v = np.clip(values.astype(np.float64), 1e-9, 1 - 1e-9)
        return np.log(v / (1 - v))
    if how == "none":
        return values
    raise ValueError(f"unknown transform {how!r}")


class NumericBlock(TransformerMixin, BaseEstimator):
    """Reshape, center and scale the numeric columns of a CSR matrix; leave the rest alone.

    TF-IDF rows are already L2-normalised and one-hot columns are 0/1, so only the numeric
    block needs this. features.load stores every numeric entry (zeros included), so centering
    happens in place without changing the sparsity pattern.
    """

    def __init__(self, cols: list[int], transforms: list[str]):
        self.cols = cols
        self.transforms = transforms

    def _entries(self, X) -> tuple[np.ndarray, np.ndarray]:
        lo, k = self.cols[0], len(self.cols)
        pos = np.flatnonzero(X.indices >= lo)
        col = X.indices[pos] - lo
        if np.any(np.bincount(col, minlength=k) != X.shape[0]):
            raise ValueError("every numeric entry must be stored, zeros included")
        return pos, col

    def _reshaped(self, X, pos, col) -> np.ndarray:
        values = X.data[pos].astype(np.float64)
        for j, how in enumerate(self.transforms):
            values[col == j] = _reshape(values[col == j], how)
        return values

    def fit(self, X, y=None):
        cols = list(self.cols)
        if cols != list(range(X.shape[1] - len(cols), X.shape[1])):
            raise ValueError("the numeric columns must be the last columns of X")
        if len(self.transforms) != len(cols):
            raise ValueError("one transform per numeric column")
        pos, col = self._entries(X)
        values = self._reshaped(X, pos, col)
        n = X.shape[0]
        self.mean_ = np.bincount(col, weights=values, minlength=len(cols)) / n
        var = np.bincount(col, weights=values**2, minlength=len(cols)) / n - self.mean_**2
        self.scale_ = np.sqrt(np.maximum(var, 0))
        self.scale_[self.scale_ == 0] = 1
        return self

    def transform(self, X):
        X = X.tocsr()
        pos, col = self._entries(X)
        data = X.data.copy()
        data[pos] = (self._reshaped(X, pos, col) - self.mean_[col]) / self.scale_[col]
        return type(X)((data, X.indices, X.indptr), shape=X.shape)


def fit_lr(
    X,
    y,
    numeric_cols: list[int],
    numeric_names: list[str],
    C: float,
    params: dict,
    balanced: bool,
    seed: int,
) -> Pipeline:
    """Fit an LR Pipeline (numeric block standardised, lbfgs); raise if it doesn't converge.

    lbfgs works on the float32 CSR matrix without copying it. saga, tried first, hit its
    iteration cap on the structured block (rare one-hot categories converge slowly).
    """
    transforms = [NUMERIC_TRANSFORMS.get(n, "none") for n in numeric_names]
    pipe = Pipeline(
        [
            ("numeric", NumericBlock(numeric_cols, transforms)),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=params["max_iter"],
                    tol=params["tol"],
                    class_weight="balanced" if balanced else None,
                    random_state=seed,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipe.fit(X, y)
    n_iter = int(np.max(pipe[-1].n_iter_))
    if n_iter >= params["max_iter"]:
        raise RuntimeError(f"logistic regression did not converge in {n_iter} iterations")
    return pipe


def search_c(grid: list[float], score) -> tuple[float, object]:
    """Choose C on validation by climbing the sorted grid from its middle pair.

    `score(C)` returns (validation score, fitted model). The climb moves toward the better of
    the middle pair and stops at the first value that doesn't improve, so the slow, weakly
    regularised end of the grid is fitted only when the scores point there. Assumes the
    validation score is unimodal in C. Keeps only the best model in memory.
    """
    grid = sorted(grid)
    if len(grid) == 1:
        return grid[0], score(grid[0])[1]
    i = (len(grid) - 1) // 2
    lo_score, lo_model = score(grid[i])
    hi_score, hi_model = score(grid[i + 1])
    if hi_score > lo_score:
        best, best_score, best_model, step = i + 1, hi_score, hi_model, 1
    else:
        best, best_score, best_model, step = i, lo_score, lo_model, -1
    del lo_model, hi_model
    j = best + step
    while 0 <= j < len(grid):
        sc, m = score(grid[j])
        if sc <= best_score:
            break
        best, best_score, best_model = j, sc, m
        j += step
    return grid[best], best_model


# --- LightGBM ----------------------------------------------------------------------------


def fit_lgbm(train: dict, val: dict, params: dict, seed: int):
    """Train LightGBM with early stopping on the validation split's average precision.

    Takes the feature dicts from features.load and drops their X once LightGBM has binned
    it, so the raw training matrix doesn't stay in memory while trees are grown.
    """
    import lightgbm as lgb

    p = dict(params)
    rounds = p.pop("num_boost_round")
    stop = p.pop("early_stopping_rounds")
    p.update(objective="binary", metric="average_precision", seed=seed, verbosity=-1)
    p.setdefault("deterministic", True)
    dtrain = lgb.Dataset(train.pop("X"), train["y"], params=p, free_raw_data=True)
    dval = lgb.Dataset(val.pop("X"), val["y"], reference=dtrain, free_raw_data=True)
    dtrain.construct()
    dval.construct()
    evals: dict = {}
    booster = lgb.train(
        p,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stop, first_metric_only=True, verbose=False),
            lgb.record_evaluation(evals),
            _progress,
        ],
    )
    log = evals["val"]["average_precision"]
    return booster, log


def _progress(env) -> None:
    if env.iteration % 50 == 0:
        (_, metric, value, _), *_ = env.evaluation_result_list
        print(f"    round {env.iteration}: val {metric} {value:.4f}", flush=True)


# --- calibration -------------------------------------------------------------------------


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(s.astype(np.float64), 1e-12, 1 - 1e-12)
    return np.log(s / (1 - s))


def _calibrator(method: str, scores: np.ndarray, y: np.ndarray):
    if method == "platt":
        m = LogisticRegression(C=np.inf).fit(_logit(scores)[:, None], y)
        return lambda s: m.predict_proba(_logit(s)[:, None])[:, 1]
    if method == "isotonic":
        m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(scores, y)
        return m.predict
    raise ValueError(f"unknown calibration method {method!r}")


def calibrate(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    test_scores: np.ndarray,
    method: str,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Recalibrate scores on validation. Returns (cross-fitted val scores, test scores)."""
    val_out = np.empty(len(val_scores), dtype=np.float64)
    for fit_idx, out_idx in StratifiedKFold(folds, shuffle=True, random_state=seed).split(
        val_scores, y_val
    ):
        val_out[out_idx] = _calibrator(method, val_scores[fit_idx], y_val[fit_idx])(
            val_scores[out_idx]
        )
    test_out = _calibrator(method, val_scores, y_val)(test_scores)
    return val_out, test_out


# --- prediction files --------------------------------------------------------------------


def write_preds(path: Path, ids: dict, scores: dict) -> None:
    """Write complaint_id, split, score for the val and test splits, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    val = {"complaint_id": ids["val"], "score": np.asarray(scores["val"], np.float64)}  # noqa: F841
    test = {"complaint_id": ids["test"], "score": np.asarray(scores["test"], np.float64)}  # noqa: F841
    con = duckdb.connect()
    con.execute(f"""
        copy (select complaint_id, 'val' as split, score from val
              union all
              select complaint_id, 'test' as split, score from test)
        to '{tmp}' (format parquet)""")
    con.close()
    os.replace(tmp, path)


def read_preds(path: Path) -> dict:
    """Return {split: (ids, scores)} sorted by complaint_id."""
    con = duckdb.connect()
    out = {}
    for s in ("val", "test"):
        rows = con.execute(
            f"select complaint_id, score from read_parquet('{path}') "
            "where split = ? order by complaint_id",
            [s],
        ).fetchnumpy()
        out[s] = (rows["complaint_id"], rows["score"])
    con.close()
    return out


# --- cells -------------------------------------------------------------------------------


def _peak_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**30 if sys.platform == "darwin" else rss / 2**20  # bytes on macOS, KB on Linux


def run_fitted(model: str, featureset: str, window: str, cfg: dict) -> dict:
    """Fit one model on one window's train split and score its val and test splits."""
    seed = cfg["seed"]
    name = cell_name(model, featureset, window)
    t0 = time.time()
    # LightGBM gets only the most frequent TF-IDF terms: grouping 100k sparse columns takes it
    # longer than training does (over 35 CPU-minutes before the first tree on W1).
    max_terms = cfg["models"]["lgbm"]["max_text_terms"] if model == "lgbm" else None
    train = features.load(window, featureset, "train", max_terms)
    print(
        f"{name}: train {train['X'].shape[0]:,} x {train['X'].shape[1]:,}, "
        f"nnz {train['X'].nnz:,} ({time.time() - t0:.0f}s to load)",
        flush=True,
    )
    meta = {"model": model, "featureset": featureset, "window": window}
    MODELS.mkdir(exist_ok=True)
    t1 = time.time()
    if model in ("lr", "lr_balanced"):
        params = cfg["models"]["lr"]
        numeric_names = [train["names"][i].removeprefix("num:") for i in train["numeric_cols"]]
        val = features.load(window, featureset, "val")
        by_c = {}

        def score(C):
            t2 = time.time()
            pipe = fit_lr(
                train["X"],
                train["y"],
                train["numeric_cols"],
                numeric_names,
                C,
                params,
                model == "lr_balanced",
                seed,
            )
            ap = float(average_precision_score(val["y"], pipe.predict_proba(val["X"])[:, 1]))
            n_iter = int(np.max(pipe[-1].n_iter_))
            by_c[str(C)] = {
                "val_average_precision": ap,
                "n_iter": n_iter,
                "seconds": time.time() - t2,
            }
            print(f"    C={C}: val AP {ap:.4f}, {n_iter} iterations", flush=True)
            return ap, pipe

        best_c, fitted = search_c(params["C"], score)
        del train, val
        meta["fit_seconds"] = time.time() - t1
        meta["C"] = best_c
        meta["by_C"] = by_c
        meta["n_iter"] = by_c[str(best_c)]["n_iter"]
        meta["params"] = params
        joblib.dump(fitted, MODELS / f"{name}.joblib")

        def predict(X):
            return fitted.predict_proba(X)[:, 1]

    elif model == "lgbm":
        params = cfg["models"]["lgbm"]
        val = features.load(window, featureset, "val", max_terms)
        lgb_params = {k: v for k, v in params.items() if k != "max_text_terms"}
        fitted, log = fit_lgbm(train, val, lgb_params, seed)
        del train, val
        meta["fit_seconds"] = time.time() - t1
        meta["best_iteration"] = fitted.best_iteration
        meta["val_average_precision_by_round"] = log
        meta["params"] = params
        fitted.save_model(str(MODELS / f"{name}.txt"), num_iteration=fitted.best_iteration)

        def predict(X):
            return fitted.predict(X, num_iteration=fitted.best_iteration)

    else:
        raise ValueError(f"unknown fitted model {model!r}")

    ids, scores = {}, {}
    for s in ("val", "test"):
        d = features.load(window, featureset, s, max_terms)
        ids[s] = d["ids"]
        scores[s] = predict(d["X"])
        del d
    write_preds(pred_path(model, featureset, window), ids, scores)
    meta["total_seconds"] = time.time() - t0
    meta["peak_rss_gb"] = _peak_gb()
    (MODELS / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"{name}: fit {meta['fit_seconds']:,.0f}s, total {meta['total_seconds']:,.0f}s, "
        f"peak {meta['peak_rss_gb']:.2f} GB",
        flush=True,
    )
    return meta


def run_calibrated(model: str, featureset: str, window: str, cfg: dict) -> None:
    """Recalibrate lr_balanced's scores for one cell on its validation split."""
    base = read_preds(pred_path("lr_balanced", featureset, window))
    d = features.FEATURES / window
    y_val = np.load(d / "y_val.npy")
    ids_val = np.load(d / "ids_val.npy")
    order = np.argsort(ids_val)
    if not np.array_equal(ids_val[order], base["val"][0]):
        raise RuntimeError(f"lr_balanced val ids don't match the {window} feature cache")
    val_scores, test_scores = calibrate(
        base["val"][1],
        y_val[order],
        base["test"][1],
        CALIBRATED[model],
        cfg["models"]["calibration_folds"],
        cfg["seed"],
    )
    write_preds(
        pred_path(model, featureset, window),
        {"val": base["val"][0], "test": base["test"][0]},
        {"val": val_scores, "test": test_scores},
    )


def run_baseline(model: str, featureset: str, window: str, cfg: dict) -> None:
    """Score val and test with the company's trailing late rate, unchanged."""
    col = features.NUMERIC.index("company_late_rate")
    d = features.FEATURES / window
    ids = {s: np.load(d / f"ids_{s}.npy") for s in ("val", "test")}
    scores = {s: np.load(d / f"num_{s}.npy")[:, col] for s in ("val", "test")}
    write_preds(pred_path(model, featureset, window), ids, scores)


def cells(models: list[str], featuresets: list[str], windows: list[str]) -> list[tuple]:
    """Cells in run order: baselines, then cheap feature sets first, calibrated models after
    their base."""
    out = [(m, "struct", w) for m in BASELINES if m in models for w in windows]
    for fs in sorted(featuresets, key=("struct", "text", "both").index):
        for w in windows:
            out += [(m, fs, w) for m in FITTED if m in models]
            out += [(m, fs, w) for m in CALIBRATED if m in models]
    return out


def main(cfg: dict, models=None, featuresets=None, windows=None, force: bool = False) -> int:
    report = json.loads(features.REPORT.read_text(encoding="utf-8"))
    windows = windows or [w["window"] for w in report["windows"]]
    todo = cells(models or list(ALL_MODELS), featuresets or list(features.FEATURE_SETS), windows)
    # Each fitted cell runs in a fresh process, so memory is returned between cells and the
    # recorded peak belongs to that cell alone.
    ctx = get_context("spawn")
    for i, (model, fs, w) in enumerate(todo, 1):
        path = pred_path(model, fs, w)
        inputs = [features.FEATURES / w / "meta.json"]
        if model in CALIBRATED:
            inputs.append(pred_path("lr_balanced", fs, w))
        if not force and not features.stale(path, inputs):
            print(f"[{i}/{len(todo)}] {path.name} is current", flush=True)
            continue
        print(f"[{i}/{len(todo)}] {cell_name(model, fs, w)}", flush=True)
        if model in BASELINES:
            run_baseline(model, fs, w, cfg)
        elif model in CALIBRATED:
            run_calibrated(model, fs, w, cfg)
        else:
            with ProcessPoolExecutor(1, mp_context=ctx) as pool:
                pool.submit(run_fitted, model, fs, w, cfg).result()
    return 0
