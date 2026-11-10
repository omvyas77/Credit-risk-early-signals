"""Deep text models and their same-data baseline, one prediction file per cell.

Models (prediction files and run notes as in models.py):
  minilm      embeddings (all-MiniLM-L6-v2) of every train, val and test narrative, then LR:
              `text` on the embeddings, `both` on the embeddings plus the structured block.
              C is chosen on validation.
  distilbert  DistilBERT fine-tuned on the train subsample: every positive plus
              `deep.negatives` negatives drawn with the seed. The checkpoint is the epoch with
              the best validation average precision. `text` is its score; `both` stacks that
              score with the structured block (see below).
  lr_sub      the same-data baseline: TF-IDF LR (feature set `text`) trained on exactly
              DistilBERT's subsample, C chosen on validation, `both` stacked the same way.
              Needs no GPU and reads the local feature cache.
Scores of models trained on the subsample are shifted back to the full training prior.
Stacked `both`: an LR on the structured block plus the logit of the text score, fitted on
validation. Its validation scores are cross-fitted (each fold scored by an LR fitted on the
other folds), C is chosen by their average precision, and test is scored by an LR fitted on
all of validation.

Narratives go through text.clean: URLs, emails and redactions removed; case, punctuation and
stopwords kept.

Inputs: data/deep_input/ when it holds the window (written by `python -m cres deep-input`
and uploaded for Colab), else data/interim/labelled.parquet and data/features/.
Embedding chunks and DistilBERT checkpoints go to data/deep/, so an interrupted run resumes.
torch, transformers and sentence-transformers come from the `deep` extra and are imported
only by the functions that use them.
"""

import contextlib
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np
import scipy.sparse as sp
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from cres import features, models, text
from cres.evaluate import window_seed
from cres.ingest import INTERIM

INPUT = Path("data/deep_input")
WORK = Path("data/deep")
DEEP_MODELS = ("minilm", "distilbert")
ALL_MODELS = (*DEEP_MODELS, "lr_sub")
FEATURE_SETS = ("text", "both")
SPLITS = ("train", "val", "test")
# Per-window files the deep runs read from the feature cache: the structured block, ids,
# labels and column names. The TF-IDF matrices stay behind.
STRUCT_FILES = ["meta.json"] + [
    f"{block}_{s}.{ext}"
    for s in SPLITS
    for block, ext in (("cat", "npz"), ("num", "npy"), ("ids", "npy"), ("y", "npy"))
]


# --- Colab input ---------------------------------------------------------------------------


def export_input(
    windows: list[str], labelled: Path, features_root: Path, out: Path, memory_limit: str
) -> dict[str, int]:
    """Write the slim deep-model input to `out` and return {relative path: bytes}.

    narratives.parquet  complaint_id, narrative for every row of the windows' splits
    splits.parquet      complaint_id, window, split for those windows
    {window}/           the structured block, ids and labels from the feature cache
    """
    for w in windows:
        if not (w.startswith("W") and w[1:].isdigit()):
            raise ValueError(f"not a window name: {w!r}")
    out.mkdir(parents=True, exist_ok=True)
    tmp_dir = out / "tmp"
    con = duckdb.connect()
    con.execute(f"set memory_limit = '{memory_limit}'")
    con.execute(f"set temp_directory = '{tmp_dir}'")
    wanted = ", ".join(f"'{w}'" for w in windows)
    splits_out = out / "splits.parquet"
    con.execute(f"""
        copy (select complaint_id, "window", split
              from read_parquet('{features_root / "splits.parquet"}')
              where "window" in ({wanted})
              order by complaint_id, "window")
        to '{splits_out}.tmp' (format parquet)""")
    os.replace(f"{splits_out}.tmp", splits_out)
    # No ORDER BY: sorting with the narrative text attached runs out of memory.
    narr_out = out / "narratives.parquet"
    con.execute(f"""
        copy (select complaint_id, narrative from read_parquet('{labelled}')
              where complaint_id in (select complaint_id from read_parquet('{splits_out}')))
        to '{narr_out}.tmp' (format parquet, compression zstd)""")
    os.replace(f"{narr_out}.tmp", narr_out)
    n_ids = con.execute(
        f"select count(distinct complaint_id) from read_parquet('{splits_out}')"
    ).fetchone()[0]
    n_narr = con.execute(f"select count(*) from read_parquet('{narr_out}')").fetchone()[0]
    con.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if n_ids != n_narr:
        raise RuntimeError(f"{n_ids:,} tagged rows but {n_narr:,} narratives")
    for w in windows:
        (out / w).mkdir(exist_ok=True)
        for name in STRUCT_FILES:
            shutil.copy2(features_root / w / name, out / w / name)
    sizes = {
        str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob("*")) if p.is_file()
    }
    print(f"deep-input: {n_narr:,} narratives for {', '.join(windows)}", flush=True)
    return sizes


def main_export(cfg: dict, windows: list[str] | None) -> int:
    windows = windows or cfg["deep"]["windows"]
    sizes = export_input(
        windows,
        INTERIM / "labelled.parquet",
        features.FEATURES,
        INPUT,
        cfg["duckdb_memory_limit"],
    )
    for name, size in sizes.items():
        print(f"  {name:<28} {size / 2**20:9.1f} MB")
    print(f"deep-input: {sum(sizes.values()) / 2**20:,.0f} MB in {INPUT}/", flush=True)
    return 0


def sources(window: str) -> tuple[Path, Path]:
    """(feature root, narratives parquet): the Colab input if it holds the window, else the
    local pipeline outputs."""
    if (INPUT / window / "meta.json").exists():
        return INPUT, INPUT / "narratives.parquet"
    return features.FEATURES, INTERIM / "labelled.parquet"


def narratives(path: Path, ids: np.ndarray, memory_limit: str) -> list[str]:
    """Narratives for `ids`, in the order of `ids`."""
    con = duckdb.connect()
    con.execute(f"set memory_limit = '{memory_limit}'")
    want = {"complaint_id": np.asarray(ids, dtype=np.int64)}  # noqa: F841 (scanned by DuckDB)
    got = con.execute(f"""
        select complaint_id, narrative from read_parquet('{path}')
        where complaint_id in (select complaint_id from want)""").fetchnumpy()
    con.close()
    got_ids, texts = np.asarray(got["complaint_id"]), got["narrative"]
    order = np.argsort(got_ids)
    pos = np.searchsorted(got_ids[order], ids)
    if len(got_ids) != len(ids) or not np.array_equal(got_ids[order][pos], ids):
        raise RuntimeError(f"{path.name}: found {len(got_ids):,} of {len(ids):,} narratives")
    return [texts[i] for i in order[pos]]


def prepare(narrative: str) -> str:
    """text.clean with runs of whitespace collapsed: what the deep models read."""
    return " ".join(text.clean(narrative).split())


# --- subsample and prior -----------------------------------------------------------------


def subsample(y: np.ndarray, negatives: int, seed: int, window: str) -> np.ndarray:
    """Sorted positions of every positive plus `negatives` negatives drawn without
    replacement (all negatives if there are fewer)."""
    rng = np.random.default_rng(window_seed(seed, window))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if negatives < len(neg):
        neg = rng.choice(neg, negatives, replace=False)
    return np.sort(np.concatenate([pos, neg]))


def prior_shift(scores: np.ndarray, kept_negatives: int, all_negatives: int) -> np.ndarray:
    """Undo negative subsampling: logit + ln(kept / all), so scores match the training
    prior. A monotone map, so rankings (and PR-AUC) are unchanged."""
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-12, 1 - 1e-12)
    return expit(logit(s) + np.log(kept_negatives / all_negatives))


def subsample_note(ids: np.ndarray, y: np.ndarray, keep: np.ndarray) -> dict:
    """What the run trained on, with a fingerprint so runs on different machines can be
    checked to share the subsample."""
    chosen = np.sort(ids[keep]).astype("<i8")
    return {
        "rows": int(len(keep)),
        "positives": int(y[keep].sum()),
        "negatives": int(len(keep) - y[keep].sum()),
        "negatives_available": int((y == 0).sum()),
        "ids_sha256": hashlib.sha256(chosen.tobytes()).hexdigest()[:16],
    }


# --- stacking on validation --------------------------------------------------------------


def crossfit(fit: Callable, X, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold scores: each fold is scored by `fit(X, y)` trained on the other folds.

    `fit` returns a function that scores rows."""
    out = np.empty(len(y), dtype=np.float64)
    split = StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    for fit_idx, out_idx in split:
        out[out_idx] = fit(X[fit_idx], y[fit_idx])(X[out_idx])
    return out


def with_score(struct: dict, score: np.ndarray) -> tuple[sp.csr_matrix, list[int], list[str]]:
    """The structured block with logit(score) appended as one more numeric column (stored
    densely and last, as NumericBlock expects). Returns X, numeric columns, their names."""
    n = len(score)
    s = np.clip(np.asarray(score, dtype=np.float64), 1e-12, 1 - 1e-12)
    col = sp.csr_matrix(
        (logit(s).astype(np.float32), np.zeros(n, dtype=np.int32), np.arange(n + 1)),
        shape=(n, 1),
    )
    X = sp.hstack([struct["X"], col], format="csr", dtype=np.float32)
    cols = [*struct["numeric_cols"], X.shape[1] - 1]
    names = [struct["names"][i].removeprefix("num:") for i in struct["numeric_cols"]]
    return X, cols, [*names, "text_score"]


def stack(
    score_val: np.ndarray,
    struct_val: dict,
    y_val: np.ndarray,
    score_test: np.ndarray,
    struct_test: dict,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stack a text score with the structured block on validation.

    Returns (cross-fitted validation scores, test scores, notes)."""
    params, seed = cfg["models"]["lr"], cfg["seed"]
    folds = cfg["models"]["calibration_folds"]
    X_val, cols, names = with_score(struct_val, score_val)
    X_test, _, _ = with_score(struct_test, score_test)

    def fitter(C):
        def fit(X, y):
            pipe = models.fit_lr(X, y, cols, names, C, params, False, seed)
            return lambda Z: pipe.predict_proba(Z)[:, 1]

        return fit

    by_c = {}

    def score(C):
        out = crossfit(fitter(C), X_val, y_val, folds, seed)
        ap = float(average_precision_score(y_val, out))
        by_c[str(C)] = {"val_average_precision": ap}
        print(f"    stack C={C}: cross-fitted val AP {ap:.4f}", flush=True)
        return ap, out

    best_c, val_out = models.search_c(params["C"], score)
    test_out = fitter(best_c)(X_val, y_val)(X_test)
    return val_out, test_out, {"C": best_c, "by_C": by_c, "stacked_on": "val", "folds": folds}


# --- LR fits -----------------------------------------------------------------------------


def fit_lr_on_val(train: dict, val: dict, cfg: dict, label: str) -> tuple[float, object, dict]:
    """models.fit_lr on train with C chosen on validation AP; returns (C, Pipeline, by C)."""
    params, seed = cfg["models"]["lr"], cfg["seed"]
    names = [train["names"][i].removeprefix("num:") for i in train["numeric_cols"]]
    by_c = {}

    def score(C):
        t0 = time.time()
        pipe = models.fit_lr(
            train["X"], train["y"], train["numeric_cols"], names, C, params, False, seed
        )
        ap = float(average_precision_score(val["y"], pipe.predict_proba(val["X"])[:, 1]))
        n_iter = int(np.max(pipe[-1].n_iter_))
        by_c[str(C)] = {"val_average_precision": ap, "n_iter": n_iter, "seconds": time.time() - t0}
        print(f"    {label} C={C}: val AP {ap:.4f}, {n_iter} iterations", flush=True)
        return ap, pipe

    best_c, pipe = models.search_c(params["C"], score)
    return best_c, pipe, by_c


def fit_dense_lr(X: np.ndarray, y: np.ndarray, C: float, params: dict, seed: int):
    """Unweighted lbfgs LR on a dense, already standardised matrix; raises if it doesn't
    converge (as models.fit_lr does)."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    lr = LogisticRegression(
        C=C, solver="lbfgs", max_iter=params["max_iter"], tol=params["tol"], random_state=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        lr.fit(X, y)
    if int(np.max(lr.n_iter_)) >= params["max_iter"]:
        raise RuntimeError("logistic regression did not converge")
    return lr


def standardise(emb: np.ndarray, mean: np.ndarray, scale: np.ndarray, chunk: int = 100_000):
    """(emb - mean) / scale as float32, a chunk at a time (emb may be memory-mapped)."""
    out = np.empty(emb.shape, dtype=np.float32)
    for a in range(0, len(emb), chunk):
        out[a : a + chunk] = (emb[a : a + chunk] - mean) / scale
    return out


def moments(emb: np.ndarray, chunk: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and standard deviation (zero sd replaced by 1), a chunk at a time."""
    total = np.zeros(emb.shape[1])
    sq = np.zeros(emb.shape[1])
    for a in range(0, len(emb), chunk):
        block = np.asarray(emb[a : a + chunk], dtype=np.float64)
        total += block.sum(0)
        sq += (block**2).sum(0)
    mean = total / len(emb)
    scale = np.sqrt(np.maximum(sq / len(emb) - mean**2, 0))
    scale[scale == 0] = 1
    return mean, scale


def hstack_dense(
    emb: np.ndarray, mean: np.ndarray, scale: np.ndarray, X: sp.csr_matrix, chunk: int = 100_000
) -> sp.csr_matrix:
    """CSR of [standardised emb | X], filled a chunk of rows at a time.

    Equivalent to sp.hstack([dense block, X]) with every embedding entry stored, but without
    holding a second full copy: on a year of training rows the result alone is about 4 GB.
    """
    n, d = emb.shape
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(d + np.diff(X.indptr), out=indptr[1:])
    idx = np.int32 if indptr[-1] < 2**31 - 1 else np.int64
    indptr = indptr.astype(idx)
    data = np.empty(indptr[-1], dtype=np.float32)
    indices = np.empty(indptr[-1], dtype=idx)
    cols = np.arange(d, dtype=idx)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        block = ((emb[a:b] - mean) / scale).astype(np.float32)
        x = X[a:b]
        lengths = np.diff(x.indptr)
        # Rows of the chunk: d embedding entries, then that row's entries of X.
        starts = indptr[a:b] - indptr[a]
        lo, hi = indptr[a], indptr[b]
        seg_data = np.empty(hi - lo, dtype=np.float32)
        seg_idx = np.empty(hi - lo, dtype=idx)
        emb_pos = (starts[:, None] + cols).ravel()
        seg_data[emb_pos] = block.ravel()
        seg_idx[emb_pos] = np.tile(cols, b - a)
        x_pos = np.repeat(starts + d, lengths) + (
            np.arange(x.nnz) - np.repeat(x.indptr[:-1], lengths)
        )
        seg_data[x_pos] = x.data
        seg_idx[x_pos] = x.indices + d
        data[lo:hi] = seg_data
        indices[lo:hi] = seg_idx
    return sp.csr_matrix((data, indices, indptr), shape=(n, d + X.shape[1]))


# --- run notes and outputs ---------------------------------------------------------------


def write_note(name: str, note: dict) -> None:
    models.MODELS.mkdir(exist_ok=True)
    (models.MODELS / f"{name}.json").write_text(json.dumps(note, indent=2) + "\n", "utf-8")


def done(model: str, window: str) -> bool:
    return all(models.pred_path(model, fs, window).exists() for fs in FEATURE_SETS)


def split_ids(root: Path, window: str) -> dict:
    return {s: np.load(root / window / f"ids_{s}.npy") for s in SPLITS}


def split_y(root: Path, window: str) -> dict:
    return {s: np.load(root / window / f"y_{s}.npy") for s in SPLITS}


def check_tags(root: Path, window: str, ids: dict) -> None:
    """When the Colab input carries splits.parquet, its tags must match the ids files."""
    path = root / "splits.parquet"
    if not path.exists():
        return
    con = duckdb.connect()
    for s in SPLITS:
        tagged = con.execute(
            f"select complaint_id from read_parquet('{path}') "
            'where "window" = ? and split = ? order by complaint_id',
            [window, s],
        ).fetchnumpy()["complaint_id"]
        if not np.array_equal(np.asarray(tagged), np.sort(ids[s])):
            raise RuntimeError(f"{path}: {window} {s} tags don't match ids_{s}.npy")
    con.close()


def write_both_stacked(
    model: str, window: str, root: Path, ids: dict, y: dict, scores: dict, cfg: dict, note: dict
) -> None:
    """Stack `scores` (val, test) with the structured block and write the `both` cell."""
    print(f"{model}__both__{window}: stacking on validation", flush=True)
    t0 = time.time()
    struct = {s: features.load(window, "struct", s, root=root) for s in ("val", "test")}
    for s in ("val", "test"):
        if not np.array_equal(struct[s]["ids"], ids[s]):
            raise RuntimeError(f"structured {s} rows are out of order")
    val_out, test_out, stack_note = stack(
        scores["val"], struct["val"], y["val"], scores["test"], struct["test"], cfg
    )
    name = models.cell_name(model, "both", window)
    models.write_preds(
        models.pred_path(model, "both", window),
        {"val": ids["val"], "test": ids["test"]},
        {"val": val_out, "test": test_out},
    )
    write_note(
        name,
        {
            **note,
            "model": model,
            "featureset": "both",
            "window": window,
            **stack_note,
            "fit_seconds": time.time() - t0,
        },
    )


# --- lr_sub: the same-data baseline ------------------------------------------------------


def run_lr_sub(window: str, cfg: dict) -> None:
    """TF-IDF LR on DistilBERT's training subsample, from the local feature cache."""
    root = features.FEATURES
    ids, y = split_ids(root, window), split_y(root, window)
    keep = subsample(y["train"], cfg["deep"]["negatives"], cfg["seed"], window)
    sub = subsample_note(ids["train"], y["train"], keep)
    print(
        f"lr_sub__text__{window}: {sub['rows']:,} train rows "
        f"({sub['positives']:,} positives, {sub['negatives']:,} negatives)",
        flush=True,
    )
    t0 = time.time()
    train = features.load(window, "text", "train", rows=keep, root=root)
    val = features.load(window, "text", "val", root=root)
    best_c, pipe, by_c = fit_lr_on_val(train, val, cfg, "lr_sub text")
    del train
    test = features.load(window, "text", "test", root=root)
    scores = {
        s: prior_shift(
            pipe.predict_proba(d["X"])[:, 1], sub["negatives"], sub["negatives_available"]
        )
        for s, d in (("val", val), ("test", test))
    }
    del val, test
    name = models.cell_name("lr_sub", "text", window)
    models.write_preds(models.pred_path("lr_sub", "text", window), ids, scores)
    note = {"subsample": sub, "trained_on": "train subsample"}
    write_note(
        name,
        {
            **note,
            "model": "lr_sub",
            "featureset": "text",
            "window": window,
            "C": best_c,
            "by_C": by_c,
            "params": cfg["models"]["lr"],
            "fit_seconds": time.time() - t0,
        },
    )
    write_both_stacked("lr_sub", window, root, ids, y, scores, cfg, note)


# --- MiniLM ------------------------------------------------------------------------------


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_minilm(cfg: dict) -> Callable[[list[str]], np.ndarray]:
    """A function from texts to L2-normalised float32 embeddings."""
    from sentence_transformers import SentenceTransformer

    mcfg = cfg["deep"]["minilm"]
    device = device_name()
    model = SentenceTransformer(mcfg["model"], device=device)
    model.max_seq_length = mcfg["max_length"]
    if device == "cuda":
        model.half()
    print(f"minilm: {mcfg['model']} on {device}", flush=True)

    def encode(texts: list[str]) -> np.ndarray:
        out = model.encode(
            texts,
            batch_size=mcfg["batch_size"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)

    return encode


def embed(
    encode: Callable, source: Path, ids: np.ndarray, path: Path, chunk: int, memory_limit: str
) -> np.ndarray:
    """Embeddings of the narratives of `ids`, cached at `path` (memory-mapped on return).

    Chunks are saved as they finish, so an interrupted run picks up where it stopped. The
    ids are saved next to the cache and checked on reuse."""
    ids_path = path.with_name(path.stem + "_ids.npy")
    if path.exists():
        if not np.array_equal(np.load(ids_path), ids):
            raise RuntimeError(f"{path} was built for other rows; delete it and rerun")
        return np.load(path, mmap_mode="r")
    path.parent.mkdir(parents=True, exist_ok=True)
    texts = narratives(source, ids, memory_limit)
    parts, t0, n = [], time.time(), len(ids)
    for k, a in enumerate(range(0, n, chunk)):
        part = path.with_name(f"{path.stem}.part{k:04d}.npy")
        if not part.exists():
            vec = encode([prepare(t) for t in texts[a : a + chunk]])
            np.save(part.with_name(part.name + ".tmp.npy"), vec)
            os.replace(part.with_name(part.name + ".tmp.npy"), part)
        parts.append(part)
        b = min(a + chunk, n)
        print(f"  {path.stem}: {b:,} / {n:,} ({time.time() - t0:,.0f}s)", flush=True)
    del texts
    first = np.load(parts[0], mmap_mode="r")
    out = np.lib.format.open_memmap(
        path.with_name(path.name + ".tmp.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(n, first.shape[1]),
    )
    a = 0
    for part in parts:
        block = np.load(part)
        out[a : a + len(block)] = block
        a += len(block)
    out.flush()
    del out
    os.replace(path.with_name(path.name + ".tmp.npy"), path)
    np.save(ids_path, ids)
    for part in parts:
        part.unlink()
    return np.load(path, mmap_mode="r")


def run_minilm(window: str, cfg: dict) -> None:
    root, source = sources(window)
    ids, y = split_ids(root, window), split_y(root, window)
    check_tags(root, window, ids)
    mcfg = cfg["deep"]["minilm"]
    t0 = time.time()
    encode = load_minilm(cfg)
    emb = {
        s: embed(
            encode,
            source,
            ids[s],
            WORK / window / f"minilm_{s}.npy",
            mcfg["chunk"],
            cfg["duckdb_memory_limit"],
        )
        for s in SPLITS
    }
    del encode
    embed_seconds = time.time() - t0
    mean, scale = moments(emb["train"])
    params, seed = cfg["models"]["lr"], cfg["seed"]
    base = {
        "window": window,
        "model": "minilm",
        "embedding_model": mcfg["model"],
        "max_length": mcfg["max_length"],
        "embed_seconds": embed_seconds,
        "trained_on": "all train rows",
        "params": params,
    }

    # text: LR on the standardised embeddings.
    t1 = time.time()
    X = {s: standardise(emb[s], mean, scale) for s in SPLITS}
    by_c = {}

    def score(C):
        lr = fit_dense_lr(X["train"], y["train"], C, params, seed)
        ap = float(average_precision_score(y["val"], lr.predict_proba(X["val"])[:, 1]))
        by_c[str(C)] = {"val_average_precision": ap, "n_iter": int(np.max(lr.n_iter_))}
        print(f"    minilm text C={C}: val AP {ap:.4f}", flush=True)
        return ap, lr

    best_c, lr = models.search_c(params["C"], score)
    scores = {s: lr.predict_proba(X[s])[:, 1] for s in ("val", "test")}
    X.clear()  # free the dense copies before building the `both` matrices
    models.write_preds(models.pred_path("minilm", "text", window), ids, scores)
    write_note(
        models.cell_name("minilm", "text", window),
        {**base, "featureset": "text", "C": best_c, "by_C": by_c, "fit_seconds": time.time() - t1},
    )

    # both: one LR on the embeddings plus the structured block, fitted on train.
    t1 = time.time()
    data = {}
    for s in SPLITS:
        struct = features.load(window, "struct", s, root=root)
        if not np.array_equal(struct["ids"], ids[s]):
            raise RuntimeError(f"structured {s} rows are out of order")
        d = emb[s].shape[1]
        data[s] = {
            "X": hstack_dense(emb[s], mean, scale, struct["X"]),
            "y": y[s],
            "names": [f"emb:{i}" for i in range(d)] + struct["names"],
            "numeric_cols": [c + d for c in struct["numeric_cols"]],
        }
        del struct
    best_c, pipe, by_c = fit_lr_on_val(data["train"], data["val"], cfg, "minilm both")
    del data["train"]
    scores = {s: pipe.predict_proba(data[s]["X"])[:, 1] for s in ("val", "test")}
    models.write_preds(models.pred_path("minilm", "both", window), ids, scores)
    write_note(
        models.cell_name("minilm", "both", window),
        {**base, "featureset": "both", "C": best_c, "by_C": by_c, "fit_seconds": time.time() - t1},
    )


# --- DistilBERT --------------------------------------------------------------------------


def tokenize(tokenizer, texts: list[str], max_length: int, chunk: int = 20_000):
    """Token ids of each text as one flat int32 array plus offsets (no padding)."""
    flat, lengths = [], []
    for a in range(0, len(texts), chunk):
        enc = tokenizer(
            [prepare(t) for t in texts[a : a + chunk]],
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        lengths += [len(e) for e in enc]
        flat += [np.asarray(e, dtype=np.int32) for e in enc]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return np.concatenate(flat), offsets


def fit_distilbert(
    texts: dict, y_train: np.ndarray, y_val: np.ndarray, cfg: dict, work: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fine-tune on texts["train"], choose the epoch on validation AP, score val and test.

    Each finished epoch is checkpointed in `work`, so a rerun resumes after the last one.
    Returns raw (val, test) probabilities of the chosen checkpoint and run notes."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dcfg, seed = cfg["deep"]["distilbert"], cfg["seed"]
    work.mkdir(parents=True, exist_ok=True)
    final = work / "scores.npz"
    if final.exists():
        z = np.load(final)
        return z["val"], z["test"], json.loads((work / "notes.json").read_text("utf-8"))
    torch.manual_seed(seed)
    device = device_name()
    tok = AutoTokenizer.from_pretrained(dcfg["model"])
    t0 = time.time()
    enc = {s: tokenize(tok, texts[s], dcfg["max_length"]) for s in texts}
    print(f"distilbert: tokenized in {time.time() - t0:,.0f}s on {device}", flush=True)
    pad = tok.pad_token_id

    def batch(split_name: str, rows: np.ndarray):
        flat, off = enc[split_name]
        width = int((off[rows + 1] - off[rows]).max())
        ids = np.full((len(rows), width), pad, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            seq = flat[off[r] : off[r + 1]]
            ids[i, : len(seq)] = seq
            mask[i, : len(seq)] = 1
        return torch.from_numpy(ids).to(device), torch.from_numpy(mask).to(device)

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()

    model = AutoModelForSequenceClassification.from_pretrained(dcfg["model"], num_labels=1)
    model.to(device)

    @torch.no_grad()
    def predict(split_name: str) -> np.ndarray:
        model.eval()
        off = enc[split_name][1]
        order = np.argsort(-(np.diff(off)), kind="stable")  # longest first: less padding
        out = np.empty(len(order), dtype=np.float64)
        bs = dcfg["eval_batch_size"]
        for a in range(0, len(order), bs):
            rows = order[a : a + bs]
            ids, mask = batch(split_name, rows)
            with autocast():
                logits = model(input_ids=ids, attention_mask=mask).logits[:, 0]
            out[rows] = torch.sigmoid(logits.float()).cpu().numpy()
        model.train()
        return out

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": dcfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(groups, lr=dcfg["learning_rate"])
    n_train, bs = len(y_train), dcfg["batch_size"]
    steps_per_epoch = -(-n_train // bs)
    total = steps_per_epoch * dcfg["epochs"]
    warmup = max(int(dcfg["warmup_ratio"] * total), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min((step + 1) / warmup, max(total - step, 0) / max(total - warmup, 1))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y_t = torch.tensor(y_train, dtype=torch.float32)

    history, start_epoch = [], 0
    last = work / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, start_epoch = state["history"], state["epoch"] + 1
        print(f"distilbert: resuming after epoch {state['epoch'] + 1}", flush=True)

    model.train()
    for epoch in range(start_epoch, dcfg["epochs"]):
        order = np.random.default_rng([seed, epoch]).permutation(n_train)
        t1, running = time.time(), 0.0
        for step, a in enumerate(range(0, n_train, bs), 1):
            rows = order[a : a + bs]
            ids, mask = batch("train", rows)
            with autocast():
                logits = model(input_ids=ids, attention_mask=mask).logits[:, 0]
            loss = loss_fn(logits.float(), y_t[rows].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 200 == 0 or step == steps_per_epoch:
                rate = step * bs / (time.time() - t1)
                eta = (steps_per_epoch - step) * bs / rate
                print(
                    f"  epoch {epoch + 1}/{dcfg['epochs']} step {step:,}/{steps_per_epoch:,}: "
                    f"loss {running / step:.4f}, {rate:,.0f} rows/s, {eta / 60:.0f} min left",
                    flush=True,
                )
        val_scores = predict("val")
        ap = float(average_precision_score(y_val, val_scores))
        history.append(
            {
                "epoch": epoch + 1,
                "val_average_precision": ap,
                "train_loss": running / steps_per_epoch,
                "seconds": time.time() - t1,
            }
        )
        print(f"distilbert: epoch {epoch + 1} val AP {ap:.4f}", flush=True)
        if ap >= max(h["val_average_precision"] for h in history):
            torch.save(model.state_dict(), work / "best.pt.tmp")
            os.replace(work / "best.pt.tmp", work / "best.pt")
            np.save(work / "best_val.npy", val_scores)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "epoch": epoch,
            },
            work / "last.pt.tmp",
        )
        os.replace(work / "last.pt.tmp", last)

    best = max(history, key=lambda h: h["val_average_precision"])
    model.load_state_dict(torch.load(work / "best.pt", map_location=device))
    val_scores = np.load(work / "best_val.npy")
    test_scores = predict("test")
    notes = {
        "base_model": dcfg["model"],
        "params": dcfg,
        "history": history,
        "best_epoch": best["epoch"],
        "device": torch.cuda.get_device_name() if device == "cuda" else device,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    np.savez(final, val=val_scores, test=test_scores)
    (work / "notes.json").write_text(json.dumps(notes, indent=2), "utf-8")
    return val_scores, test_scores, notes


def run_distilbert(window: str, cfg: dict) -> None:
    root, source = sources(window)
    ids, y = split_ids(root, window), split_y(root, window)
    check_tags(root, window, ids)
    keep = subsample(y["train"], cfg["deep"]["negatives"], cfg["seed"], window)
    sub = subsample_note(ids["train"], y["train"], keep)
    print(
        f"distilbert__text__{window}: {sub['rows']:,} train rows "
        f"({sub['positives']:,} positives, {sub['negatives']:,} negatives)",
        flush=True,
    )
    work = WORK / window / "distilbert"
    run_key = {"subsample": sub, "params": cfg["deep"]["distilbert"], "seed": cfg["seed"]}
    key_path = work / "run.json"
    if key_path.exists() and json.loads(key_path.read_text("utf-8")) != run_key:
        raise RuntimeError(f"{work} holds a run with other settings; delete it and rerun")
    work.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(run_key, indent=2), "utf-8")
    t0 = time.time()
    limit = cfg["duckdb_memory_limit"]
    texts = {
        "train": narratives(source, ids["train"][keep], limit),
        "val": narratives(source, ids["val"], limit),
        "test": narratives(source, ids["test"], limit),
    }
    val_raw, test_raw, notes = fit_distilbert(texts, y["train"][keep], y["val"], cfg, work)
    del texts
    scores = {
        "val": prior_shift(val_raw, sub["negatives"], sub["negatives_available"]),
        "test": prior_shift(test_raw, sub["negatives"], sub["negatives_available"]),
    }
    models.write_preds(models.pred_path("distilbert", "text", window), ids, scores)
    note = {"subsample": sub, "trained_on": "train subsample"}
    write_note(
        models.cell_name("distilbert", "text", window),
        {
            **note,
            **notes,
            "model": "distilbert",
            "featureset": "text",
            "window": window,
            "total_seconds": time.time() - t0,
        },
    )
    write_both_stacked("distilbert", window, root, ids, y, scores, cfg, note)


# --- stage -------------------------------------------------------------------------------

RUNNERS = {"minilm": run_minilm, "distilbert": run_distilbert, "lr_sub": run_lr_sub}


def main(cfg: dict, model: str, window: str, force: bool = False) -> int:
    if model not in RUNNERS:
        raise SystemExit(f"deep: unknown model {model!r}; choose from {', '.join(RUNNERS)}")
    if done(model, window) and not force:
        print(f"deep: {model} {window} prediction files exist (use --force to redo)")
        return 0
    t0 = time.time()
    RUNNERS[model](window, cfg)
    print(f"deep: {model} {window} done in {(time.time() - t0) / 60:,.1f} min", flush=True)
    for fs in FEATURE_SETS:
        print(f"  wrote {models.pred_path(model, fs, window)}")
    return 0
