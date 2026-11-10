"""The in-browser scorer: a logistic regression small enough to ship as JSON.

Fitted on the latest window's train split with TF-IDF, the lexicon counts and the structured
fields, but not VADER (it has no JavaScript port):
  1. fit on the whole vocabulary, C chosen on validation average precision;
  2. keep the `scorer.max_terms` terms with the largest mean |coefficient x TF-IDF| on train;
  3. refit on just those terms, rows re-normalised over them (the TF-IDF that vocabulary gives
     by itself), C chosen on validation again.
The bundle's numbers are rounded, and the prediction file is scored with the rounded numbers,
so the reported metrics are the shipped model's.

Writes
  data/preds/scorer__both__{window}.parquet  val and test scores, evaluated like any model
  models/scorer__both__{window}.json         fit log
  models/scorer/model.json                   the bundle docs/js/scorer.js reads; export copies it

`score()` is the float64 reference that the parity test compares with scorer.js.
"""

import json
import math
import time
from collections import Counter

import duckdb
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

from cres import features, models, text
from cres.ingest import INTERIM

MODEL = "scorer"
FEATURESET = "both"
BUNDLE = models.MODELS / "scorer" / "model.json"
FORMAT = 1
# Categorical variables the scorer's dropdowns set; the rest are held at their train baseline.
DROPDOWNS = ("product", "issue")
# Within each dropdown, the finer variable is averaged over train rows with the chosen value.
NESTED = {"product": "sub_product", "issue": "sub_issue"}
INFREQUENT = "infrequent_sklearn"
MISSING = "(missing)"
BASELINE_RULE = [
    "Sub-product and sub-issue are set to their train-split average among complaints with "
    "the chosen product and issue.",
    "State, submission channel and tags are set to their train-split average.",
    "With no product or issue chosen, each is set to its train-split average.",
]


def latest_window() -> str:
    report = json.loads(features.REPORT.read_text(encoding="utf-8"))
    return report["windows"][-1]["window"]


def rounder(digits: int):
    def r(v: float) -> float:
        return float(f"{float(v):.{digits}g}")

    return r


# --- companies ---------------------------------------------------------------------------


def company_stats(cfg: dict, window: str) -> list[dict]:
    """Companies with at least `scorer.company_min_complaints` labelled complaints across the
    window's splits: counts, late rate, and the company-history features of their latest
    complaint (the history a new complaint would get, give or take the last few weeks).

    Reads only the id, company and date columns of labelled.parquet (no narratives).
    """
    con = duckdb.connect()
    con.execute(f"set memory_limit = '{cfg['duckdb_memory_limit']}'")
    rows = con.execute(
        f"""
        with s as (
          select complaint_id, split from read_parquet('{features.FEATURES / "splits.parquet"}')
          where "window" = ?),
        l as (
          select complaint_id, company
          from read_parquet('{INTERIM / "labelled.parquet"}')),
        r as (
          select complaint_id, date_received, late, company_late_rate, company_log_volume
          from read_parquet('{features.FEATURES / "rows.parquet"}'))
        select l.company, count(*) as complaints, sum(r.late::int) as late,
               sum((s.split = 'test')::int) as test_complaints,
               arg_max(r.company_late_rate, (r.date_received, r.complaint_id)) as late_rate,
               arg_max(r.company_log_volume, (r.date_received, r.complaint_id)) as log_volume,
               max(r.date_received) as latest
        from s join l using (complaint_id) join r using (complaint_id)
        group by l.company
        having count(*) >= ?
        order by complaints desc, l.company""",
        [window, cfg["scorer"]["company_min_complaints"]],
    ).fetchall()
    con.close()
    keys = ("company", "complaints", "late", "test_complaints", "late_rate", "log_volume")
    out = []
    for row in rows:
        d = dict(zip(keys, row[:6], strict=True))
        d["name"] = d["company"].strip()  # some names carry a leading non-breaking space
        d["late_rate_observed"] = d["late"] / d["complaints"]
        d["latest"] = str(row[6])
        out.append(d)
    names = [d["name"] for d in out]
    if len(set(names)) != len(names):
        raise RuntimeError("two company names differ only by surrounding whitespace")
    return out


def default_company(cfg: dict, window: str, listed: list[str]) -> tuple[float, float]:
    """Median company-history features over train rows of companies outside the list."""
    con = duckdb.connect()
    con.execute(f"set memory_limit = '{cfg['duckdb_memory_limit']}'")
    row = con.execute(
        f"""
        select median(r.company_late_rate), median(r.company_log_volume)
        from read_parquet('{features.FEATURES / "splits.parquet"}') s
        join read_parquet('{INTERIM / "labelled.parquet"}') l using (complaint_id)
        join read_parquet('{features.FEATURES / "rows.parquet"}') r using (complaint_id)
        where s."window" = ? and s.split = 'train'
          and not list_contains(?, l.company)""",
        [window, listed],
    ).fetchone()
    con.close()
    return float(row[0]), float(row[1])


# --- categorical baselines ---------------------------------------------------------------


def split_categorical(names: list[str]) -> dict[str, list[tuple[int, str]]]:
    """{variable: [(column, category)]} from one-hot names like "sub_product_Store credit"."""
    out: dict[str, list[tuple[int, str]]] = {v: [] for v in features.CATEGORICAL}
    by_length = sorted(features.CATEGORICAL, key=len, reverse=True)
    for j, n in enumerate(names):
        var = next(v for v in by_length if n.startswith(v + "_"))
        out[var].append((j, n[len(var) + 1 :]))
    return out


def one_hot_index(block: sp.csr_matrix) -> np.ndarray:
    """Column of each row's single 1 in a one-hot block."""
    block = block.tocsr()
    if not np.all(np.diff(block.indptr) == 1):
        raise ValueError("each row must have exactly one category")
    return block.indices


def categorical_part(
    x_cat: sp.csr_matrix, coef: np.ndarray, names: list[str], min_count: int, r
) -> dict:
    """Dropdown coefficients, train baselines for everything else, and issues by product.

    `coef` are the LR coefficients of the one-hot columns `names`; `x_cat` is the train block.
    """
    x_cat = x_cat.tocsr()
    cols = split_categorical(names)
    contrib = {}  # per train row, each variable's contribution to the log-odds
    index = {}
    for var, pairs in cols.items():
        idx = [j for j, _ in pairs]
        block = x_cat[:, idx]
        contrib[var] = block @ coef[idx]
        index[var] = one_hot_index(block)
    out = {"rule": BASELINE_RULE}
    for var in DROPDOWNS:
        nested = NESTED[var]
        cats = [c for _, c in cols[var]]
        sums = np.bincount(index[var], weights=contrib[nested], minlength=len(cats))
        counts = np.bincount(index[var], minlength=len(cats))
        out[var] = {
            c: [r(coef[j]), r(sums[k] / counts[k])]
            for k, (j, c) in enumerate(cols[var])
            if c not in (INFREQUENT, MISSING) and counts[k] > 0
        }
        out[f"{var}_default"] = r(np.mean(contrib[var] + contrib[nested]))
    held = [v for v in features.CATEGORICAL if v not in DROPDOWNS and v not in NESTED.values()]
    out["held"] = held
    out["held_baseline"] = r(sum(np.mean(contrib[v]) for v in held))
    # Issues seen with each product at least min_count times, for the issue dropdown.
    products = [c for _, c in cols["product"]]
    issues = [c for _, c in cols["issue"]]
    pair = np.bincount(
        index["product"] * len(issues) + index["issue"], minlength=len(products) * len(issues)
    ).reshape(len(products), len(issues))
    out["issues_by_product"] = {
        p: [issues[k] for k in np.flatnonzero(pair[i] >= min_count) if issues[k] in out["issue"]]
        for i, p in enumerate(products)
        if p in out["product"]
    }
    return out


# --- bundle ------------------------------------------------------------------------------


def build_bundle(
    pipe,
    terms: list[str],
    idf: np.ndarray,
    cat_names: list[str],
    numeric_names: list[str],
    x_cat_train: sp.csr_matrix,
    companies: dict[str, tuple[float, float]],
    company_default: tuple[float, float],
    cfg: dict,
    info: dict,
) -> dict:
    """The JSON bundle for scorer.js, from an LR Pipeline over [TF-IDF | one-hot | numeric]."""
    r = rounder(cfg["scorer"]["digits"])
    numeric, lr = pipe[0], pipe[-1]
    coef = lr.coef_.ravel().astype(np.float64)
    v, c = len(terms), len(cat_names)
    if len(coef) != v + c + len(numeric_names):
        raise ValueError("coefficients don't match the blocks")
    return {
        "format": FORMAT,
        **info,
        "tokenizer": {
            "version": text.VERSION,
            "hash": text.spec_hash(),
            "lexicon": text.LEXICON,
        },
        "intercept": r(lr.intercept_[0]),
        "tfidf": {"sublinear_tf": True, "norm": "l2"},
        # Every kept term stays, whatever its coefficient: each one counts in the L2 norm.
        "vocabulary": {t: [r(idf[i]), r(coef[i])] for i, t in enumerate(terms)},
        "numeric": [
            {
                "name": n,
                "transform": numeric.transforms[k],
                "mean": r(numeric.mean_[k]),
                "scale": r(numeric.scale_[k]),
                "coef": r(coef[v + c + k]),
            }
            for k, n in enumerate(numeric_names)
        ],
        "categorical": categorical_part(
            x_cat_train,
            coef[v : v + c],
            cat_names,
            cfg["features"]["categorical_min_frequency"],
            r,
        ),
        "companies": {name: [r(a), r(b)] for name, (a, b) in companies.items()},
        "company_default": [r(company_default[0]), r(company_default[1])],
        "defaults": {"days_to_company": 0},
    }


def apply_bundle(pipe, bundle: dict, terms: list[str], cat_names: list[str]) -> None:
    """Set the Pipeline's coefficients, intercept and scaler to the bundle's rounded values,
    so it scores exactly as the bundle does (with each row's own categories)."""
    numeric, lr = pipe[0], pipe[-1]
    coef = lr.coef_.ravel()
    vocab = bundle["vocabulary"]
    rounded = np.array([vocab[t][1] for t in terms])
    r = rounder(bundle["digits"])
    rounded_cat = np.array([r(x) for x in coef[len(terms) : len(terms) + len(cat_names)]])
    rounded_num = np.array([n["coef"] for n in bundle["numeric"]])
    lr.coef_[0] = np.concatenate([rounded, rounded_cat, rounded_num]).astype(lr.coef_.dtype)
    lr.intercept_[0] = bundle["intercept"]
    numeric.mean_ = np.array([n["mean"] for n in bundle["numeric"]])
    numeric.scale_ = np.array([n["scale"] for n in bundle["numeric"]])


# --- reference scorer (float64; scorer.js does the same) ----------------------------------


def transform(value: float, how: str) -> float:
    if how == "log1p":
        return math.copysign(math.log1p(abs(value)), value)
    if how == "logit":
        v = min(max(value, 1e-9), 1 - 1e-9)
        return math.log(v / (1 - v))
    if how == "none":
        return value
    raise ValueError(f"unknown transform {how!r}")


def score(
    bundle: dict,
    narrative: str,
    product: str | None = None,
    issue: str | None = None,
    company: str | None = None,
    days_to_company: float | None = None,
) -> dict:
    """Probability of a late response, its log-odds parts and each term's contribution.

    A product, issue or company the bundle doesn't list falls back to the bundle's default.
    """
    if bundle["tokenizer"]["lexicon"] != text.LEXICON:
        raise ValueError("the bundle's lexicon differs from text.LEXICON")
    vocab = bundle["vocabulary"]
    toks = text.tokens(narrative)
    counts = Counter(g for g in text.ngrams(toks) if g in vocab)
    weights = {g: (1 + math.log(n)) * vocab[g][0] for g, n in counts.items()}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    terms = []
    for g, w in weights.items():
        x = w / norm
        terms.append({"term": g, "count": counts[g], "tfidf": x, "contribution": x * vocab[g][1]})
    lex = text.lexicon_counts(toks)
    late_rate, log_volume = bundle["companies"].get(company, bundle["company_default"])
    values = {
        "days_to_company": bundle["defaults"]["days_to_company"]
        if days_to_company is None
        else days_to_company,
        "company_late_rate": late_rate,
        "company_log_volume": log_volume,
        **lex,
    }
    numeric = {
        n["name"]: n["coef"]
        * (transform(values[n["name"]], n["transform"]) - n["mean"])
        / n["scale"]
        for n in bundle["numeric"]
    }
    cat = bundle["categorical"]

    def dropdown(var: str, choice: str | None) -> float:
        if choice not in cat[var]:  # nothing chosen, or a value the model doesn't list
            return cat[f"{var}_default"]
        coef, nested = cat[var][choice]
        return coef + nested

    parts = {
        "intercept": bundle["intercept"],
        "text": sum(t["contribution"] for t in terms),
        "lexicon": sum(numeric[g] for g in text.LEXICON),
        "company": numeric["company_late_rate"] + numeric["company_log_volume"],
        "days_to_company": numeric["days_to_company"],
        "product": dropdown("product", product),
        "issue": dropdown("issue", issue),
        "held": cat["held_baseline"],
    }
    logit = sum(parts.values())
    return {
        "probability": 1 / (1 + math.exp(-logit)),
        "logit": logit,
        "parts": parts,
        "tokens": toks,
        "lexicon": lex,
        "terms": terms,
    }


# --- fitting -----------------------------------------------------------------------------


def fit_on_val(train: dict, val: dict, numeric_names: list[str], cfg: dict, log: dict):
    """LR with C chosen on validation average precision (models.search_c)."""
    params = cfg["models"]["lr"]

    def fit(C):
        t0 = time.time()
        pipe = models.fit_lr(
            train["X"],
            train["y"],
            train["numeric_cols"],
            numeric_names,
            C,
            params,
            False,
            cfg["seed"],
        )
        ap = float(average_precision_score(val["y"], pipe.predict_proba(val["X"])[:, 1]))
        n_iter = int(np.max(pipe[-1].n_iter_))
        log[str(C)] = {"val_average_precision": ap, "n_iter": n_iter, "seconds": time.time() - t0}
        print(f"    C={C}: val AP {ap:.4f}, {n_iter} iterations", flush=True)
        return ap, pipe

    return models.search_c(params["C"], fit)


def rank_terms(pipe, x_train: sp.csr_matrix, n_terms: int) -> np.ndarray:
    """Term columns ranked by mean |coefficient x TF-IDF| over train rows (TF-IDF >= 0)."""
    coef = np.abs(pipe[-1].coef_.ravel()[:n_terms]).astype(np.float64)
    sums = np.asarray(x_train.sum(axis=0), dtype=np.float64).ravel()[:n_terms]
    mean_x = sums / x_train.shape[0]
    return np.argsort(-(coef * mean_x), kind="stable")


def check_text_part(
    cfg: dict, window: str, bundle: dict, val: dict, pipe, n_terms: int, numeric_names: list
) -> float:
    """Re-score seeded validation narratives through `score()` and compare its text and
    lexicon log-odds with the fitted Pipeline's on the cached matrix. Returns the largest gap.

    Narratives are read into memory only; nothing derived from them is written.
    """
    n = min(cfg["scorer"]["check_rows"], len(val["ids"]))
    pick = np.sort(np.random.default_rng(cfg["seed"]).choice(len(val["ids"]), n, replace=False))
    ids = [int(i) for i in val["ids"][pick]]
    con = duckdb.connect()
    con.execute(f"set memory_limit = '{cfg['duckdb_memory_limit']}'")
    wanted = {"complaint_id": np.array(ids, dtype=np.int64)}  # noqa: F841 (scanned below)
    narratives = dict(
        con.execute(
            f"select l.complaint_id, l.narrative "
            f"from read_parquet('{INTERIM / 'labelled.parquet'}') l "
            f"semi join wanted w on l.complaint_id = w.complaint_id"
        ).fetchall()
    )
    con.close()
    coef = pipe[-1].coef_.ravel().astype(np.float64)
    x = pipe[0].transform(val["X"][pick]).astype(np.float64)
    lex_cols = [x.shape[1] - len(numeric_names) + numeric_names.index(g) for g in text.LEXICON]
    expected = x[:, :n_terms] @ coef[:n_terms] + x[:, lex_cols] @ coef[lex_cols]
    gaps = []
    for k, cid in enumerate(ids):
        s = score(bundle, narratives[cid])
        gaps.append(abs(s["parts"]["text"] + s["parts"]["lexicon"] - expected[k]))
    return float(max(gaps))


def main(cfg: dict, force: bool = False) -> int:
    window = latest_window()
    name = models.cell_name(MODEL, FEATURESET, window)
    path = models.pred_path(MODEL, FEATURESET, window)
    d = features.FEATURES / window
    inputs = [d / "meta.json", features.FEATURES / "rows.parquet"]
    if not force and not features.stale(path, inputs) and not features.stale(BUNDLE, inputs):
        print(f"scorer: {path.name} and the bundle are current")
        return 0
    scfg = cfg["scorer"]
    numeric_names = scfg["numeric"]
    seed = cfg["seed"]
    t0 = time.time()
    meta = {"model": MODEL, "featureset": FEATURESET, "window": window, "numeric": numeric_names}

    print(f"scorer {window}: fit 1 of 2, whole vocabulary, no VADER", flush=True)
    train = features.load(window, FEATURESET, "train", numeric=numeric_names)
    val = features.load(window, FEATURESET, "val", numeric=numeric_names)
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    n_vocab = len(vocab["terms"])
    meta["full"] = {"by_C": {}, "terms": n_vocab}
    c_full, pipe = fit_on_val(train, val, numeric_names, cfg, meta["full"]["by_C"])
    meta["full"]["C"] = c_full
    ranked = rank_terms(pipe, train["X"], n_vocab)
    keep = np.sort(ranked[: scfg["max_terms"]])
    del train, val, pipe

    print(f"scorer {window}: fit 2 of 2, {len(keep):,} terms", flush=True)
    train = features.load(window, FEATURESET, "train", terms=keep, numeric=numeric_names)
    val = features.load(window, FEATURESET, "val", terms=keep, numeric=numeric_names)
    meta["by_C"] = {}
    c_best, pipe = fit_on_val(train, val, numeric_names, cfg, meta["by_C"])
    meta["C"] = c_best
    meta["n_iter"] = meta["by_C"][str(c_best)]["n_iter"]
    del train

    terms = [vocab["terms"][i] for i in keep]
    idf = np.asarray(vocab["idf"], dtype=np.float64)[keep]
    cat_names = json.loads((d / "meta.json").read_text(encoding="utf-8"))["categorical_names"]
    stats = company_stats(cfg, window)
    companies = {c["name"]: (c["late_rate"], c["log_volume"]) for c in stats}
    default = default_company(cfg, window, [c["company"] for c in stats])
    report = json.loads(features.REPORT.read_text(encoding="utf-8"))
    dates = next(w["dates"] for w in report["windows"] if w["window"] == window)
    info = {
        "digits": scfg["digits"],
        "window": window,
        "train_dates": dates["train"],
        "C": c_best,
        "terms_full": n_vocab,
        "company_min_complaints": scfg["company_min_complaints"],
        "company_default_rule": "median over train rows of companies not in the list",
    }
    bundle = build_bundle(
        pipe,
        terms,
        idf,
        cat_names,
        numeric_names,
        sp.load_npz(d / "cat_train.npz"),
        companies,
        default,
        cfg,
        info,
    )
    meta["text_check_max_gap"] = check_text_part(
        cfg, window, bundle, val, pipe, len(terms), numeric_names
    )
    print(f"scorer: text part re-scored from narratives, max gap {meta['text_check_max_gap']:.2e}")
    if meta["text_check_max_gap"] > 1e-3:
        raise RuntimeError("the bundle doesn't reproduce the fitted model's text part")
    del val

    before = {}
    ids, scores = {}, {}
    for s in ("val", "test"):
        m = features.load(window, FEATURESET, s, terms=keep, numeric=numeric_names)
        before[s] = pipe.predict_proba(m["X"])[:, 1]
        ids[s] = m["ids"]
        del m
    apply_bundle(pipe, bundle, terms, cat_names)
    for s in ("val", "test"):
        m = features.load(window, FEATURESET, s, terms=keep, numeric=numeric_names)
        scores[s] = pipe.predict_proba(m["X"])[:, 1]
        del m
    meta["rounding_max_gap"] = float(
        max(np.max(np.abs(scores[s] - before[s])) for s in ("val", "test"))
    )
    print(f"scorer: rounding moves scores by at most {meta['rounding_max_gap']:.2e}")

    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    size = len(payload.encode("utf-8"))
    if size > scfg["max_bundle_mb"] * 1e6:
        raise RuntimeError(f"bundle is {size / 1e6:.2f} MB, over {scfg['max_bundle_mb']} MB")
    BUNDLE.write_text(payload + "\n", encoding="utf-8")
    models.write_preds(path, ids, scores)
    meta.update(
        terms=len(bundle["vocabulary"]),
        bundle_bytes=size,
        companies=len(companies),
        total_seconds=time.time() - t0,
        peak_rss_gb=models._peak_gb(),
        seed=seed,
    )
    (models.MODELS / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    print(
        f"scorer: {meta['terms']:,} terms, C={c_best}, bundle {size / 1e6:.2f} MB, "
        f"{meta['total_seconds']:,.0f}s, peak {meta['peak_rss_gb']:.2f} GB",
        flush=True,
    )
    return 0
