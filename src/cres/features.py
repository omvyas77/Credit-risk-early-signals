"""Windows and feature blocks, cached under data/features/.

Built once for every labelled row in any window (data/features/rows.parquet):
  structured fields known at intake, days until sent to the company, company history,
  lexicon counts and the VADER compound score.
Built per window (data/features/W*/), fitted on that window's train rows only:
  tfidf_{split}.npz   TF-IDF of 1-2 grams (text.py), vocabulary and idf from train
  cat_{split}.npz     one-hot structured categories
  num_{split}.npy     numeric columns (unscaled; models scale them inside a Pipeline)
  y_{split}.npy, ids_{split}.npy, meta.json
`load(window, featureset, split)` assembles the `text`, `struct` or `both` matrix.
"""

import datetime as dt
import json
import time
from collections import Counter
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from cres import split, text
from cres.ingest import INTERIM

FEATURES = Path("data/features")
REPORT = Path("reports/windows.json")

# Banned as features (CLAUDE.md rule 2). The only outcome-derived feature is company history,
# which uses other complaints received before date - gap_days.
BANNED = {
    "timely_response",
    "company_response_to_consumer",
    "company_public_response",
    "consumer_disputed",
    "late",
}
CATEGORICAL = ["product", "sub_product", "issue", "sub_issue", "state", "submitted_via", "tags"]
STRUCT_NUMERIC = ["days_to_company", "company_late_rate", "company_log_volume"]
TEXT_NUMERIC = ["urgency", "harm", "vader"]
NUMERIC = STRUCT_NUMERIC + TEXT_NUMERIC
FEATURE_SETS = ("text", "struct", "both")
CHUNK = 20_000


# --- company history -------------------------------------------------------------------


def company_history(
    con: duckdb.DuckDBPyConnection,
    live: str,
    rows: str,
    gap_days: int,
    days: int,
    prior_strength: float,
    unsettled: list[str],
) -> None:
    """Create table `company_history` (complaint_id, company_late_rate, company_log_volume).

    For a row received on date d, with cutoff c = d - gap_days, counts settled complaints in
    `live` against the same company received in [c - days, c). The late rate is smoothed
    toward the overall late rate over the same days: (late + a * overall) / (n + a).
    """
    drop = ", ".join("'" + u.replace("'", "''") + "'" for u in unsettled)
    con.execute(f"""
        create or replace table company_history as
        with daily as (
          select company, date_received as d, count(*) as n,
                 sum((timely_response = 'No')::int) as k
          from {live}
          where timely_response is not null
            and (company_response_to_consumer is null
                 or company_response_to_consumer not in ({drop}))
          group by all),
        cum as (
          select company, d, sum(n) over w as cn, sum(k) over w as ck
          from daily window w as (partition by company order by d
                                  rows between unbounded preceding and current row)),
        ocum as (
          select d, sum(n) over (order by d) as cn, sum(k) over (order by d) as ck
          from (select d, sum(n) as n, sum(k) as k from daily group by d)),
        r as (
          select complaint_id, company,
                 date_received - interval {int(gap_days)} day as hi,
                 date_received - interval {int(gap_days) + int(days)} day as lo
          from {rows}),
        a as (
          select r.*, coalesce(c.cn, 0) as cn_hi, coalesce(c.ck, 0) as ck_hi
          from r asof left join cum c on r.company = c.company and r.hi > c.d),
        b as (
          select a.*, coalesce(c.cn, 0) as cn_lo, coalesce(c.ck, 0) as ck_lo
          from a asof left join cum c on a.company = c.company and a.lo > c.d),
        o as (
          select b.*, coalesce(oh.cn, 0) - coalesce(ol.cn, 0) as on_,
                 coalesce(oh.ck, 0) - coalesce(ol.ck, 0) as ok_
          from b
          asof left join ocum oh on b.hi > oh.d
          asof left join ocum ol on b.lo > ol.d)
        select complaint_id,
          ((ck_hi - ck_lo) + {float(prior_strength)} * (ok_ / nullif(on_, 0)))
            / ((cn_hi - cn_lo) + {float(prior_strength)}) as company_late_rate,
          ln(1 + (cn_hi - cn_lo)) as company_log_volume
        from o""")


# --- per-row text features ---------------------------------------------------------------

_vader = None


def _vader_scores(batch: list[tuple[int, str]]) -> list[tuple[int, float]]:
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader = SentimentIntensityAnalyzer()
    return [(cid, _vader.polarity_scores(narrative)["compound"]) for cid, narrative in batch]


def _lexicon(batch: list[tuple[int, str]]) -> list[tuple[int, int, int]]:
    out = []
    for cid, narrative in batch:
        counts = text.lexicon_counts(text.tokens(narrative))
        out.append((cid, counts["urgency"], counts["harm"]))
    return out


def batches(cursor: duckdb.DuckDBPyConnection, size: int = CHUNK):
    while rows := cursor.fetchmany(size):
        yield rows


def pooled(pool: Pool, fn, items, group: int):
    """pool.map over `items` a few at a time, so only `group` batches are in memory at once
    (Pool.imap would read the whole iterable eagerly)."""
    items = iter(items)
    while chunk := list(islice(items, group)):
        yield from pool.map(fn, chunk)


def per_narrative(con: duckdb.DuckDBPyConnection, source: str, fn, workers: int, what: str):
    """Run `fn` over (complaint_id, narrative) batches of `source` on `workers` processes."""
    n = con.execute(f"select count(*) from {source}").fetchone()[0]
    print(f"features: {what} for {n:,} narratives on {workers} workers", flush=True)
    cursor = con.cursor()
    cursor.execute(f"select complaint_id, narrative from {source}")  # sorted source
    out, start = [], time.time()
    with Pool(workers) as pool:
        for result in pooled(pool, fn, batches(cursor), 2 * workers):
            out.extend(result)
            if len(out) % (10 * CHUNK) < CHUNK:
                rate = len(out) / (time.time() - start)
                print(f"  {len(out):,} / {n:,} ({rate:,.0f}/s)", flush=True)
    return out


def build_rows(
    con: duckdb.DuckDBPyConnection, labelled: str, workers: int, vader_path: Path
) -> None:
    """Create table `rows`: structured fields, company history and text-derived numerics.

    VADER reads the raw narrative and doesn't depend on text.py, so its scores are cached in
    `vader_path` and recomputed only when that file is missing.
    """
    cols = {r[0] for r in con.execute(f"describe select * from {labelled}").fetchall()}
    leaked = cols & (BANNED - {"late"})
    if leaked:
        raise ValueError(f"outcome columns reached the feature source: {sorted(leaked)}")
    if not vader_path.exists():
        scores = per_narrative(con, labelled, _vader_scores, workers, "VADER")
        vader = {  # noqa: F841 (scanned by DuckDB below)
            "complaint_id": np.array([s[0] for s in scores], dtype=np.int64),
            "vader": np.array([s[1] for s in scores], dtype=np.float32),
        }
        con.execute(
            f"copy (select * from vader order by complaint_id) to '{vader_path}' (format parquet)"
        )
    lex = per_narrative(con, labelled, _lexicon, workers, "lexicon")
    lexicon = {  # noqa: F841 (scanned by DuckDB below)
        "complaint_id": np.array([r[0] for r in lex], dtype=np.int64),
        "urgency": np.array([r[1] for r in lex], dtype=np.float32),
        "harm": np.array([r[2] for r in lex], dtype=np.float32),
    }
    cat = ", ".join(f"coalesce(l.{c}, '(missing)') as {c}" for c in CATEGORICAL)
    con.execute(f"""
        create or replace table rows as
        select l.complaint_id, l.date_received, l.late, {cat},
          date_diff('day', l.date_received, l.date_sent_to_company)::float as days_to_company,
          h.company_late_rate::float as company_late_rate,
          h.company_log_volume::float as company_log_volume,
          t.urgency, t.harm, v.vader
        from {labelled} l
        join company_history h using (complaint_id)
        join lexicon t using (complaint_id)
        join read_parquet('{vader_path}') v using (complaint_id)""")
    missing = (
        con.execute(f"select count(*) from {labelled}").fetchone()[0]
        - con.execute("select count(*) from rows").fetchone()[0]
    )
    if missing:
        raise RuntimeError(f"{missing} rows lost in the join (stale {vader_path.name}?)")


# --- TF-IDF ------------------------------------------------------------------------------

_vocab: dict[str, int] = {}


def _doc_freq(texts: list[str]) -> Counter:
    c = Counter()
    for t in texts:
        c.update(set(text.analyze(t)))
    return c


def _init_vocab(vocab: dict[str, int]) -> None:
    global _vocab
    _vocab = vocab


def _counts(texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths, indices, counts = [], [], []
    for t in texts:
        c = Counter(i for g in text.analyze(t) if (i := _vocab.get(g)) is not None)
        lengths.append(len(c))
        indices.extend(c.keys())
        counts.extend(c.values())
    return (
        np.array(lengths, dtype=np.int64),
        np.array(indices, dtype=np.int32),
        np.array(counts, dtype=np.float32),
    )


def choose_vocab(doc_freq: Counter, min_df: int, max_features: int) -> list[str]:
    """Terms with df >= min_df, most frequent first (ties alphabetical), capped."""
    kept = [(t, df) for t, df in doc_freq.items() if df >= min_df]
    kept.sort(key=lambda x: (-x[1], x[0]))
    return [t for t, _ in kept[:max_features]]


def count_matrix(pool: Pool, texts, group: int) -> sp.csr_matrix:
    parts = list(pooled(pool, _counts, texts, group))
    lengths = np.concatenate([p[0] for p in parts])
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    indices = np.concatenate([p[1] for p in parts])
    data = np.concatenate([p[2] for p in parts])
    del parts
    return sp.csr_matrix((data, indices, indptr), shape=(len(lengths), len(_vocab)))


def idf_from(train_counts: sp.csr_matrix) -> np.ndarray:
    """Smooth idf: ln((1 + n) / (1 + df)) + 1, df counted on train rows."""
    n = train_counts.shape[0]
    df = np.bincount(train_counts.indices, minlength=train_counts.shape[1])
    return (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)


def tfidf(counts: sp.csr_matrix, idf: np.ndarray, sublinear: bool) -> sp.csr_matrix:
    """Apply sublinear tf, idf weights and row L2 normalisation, in place."""
    if sublinear:
        np.log(counts.data, out=counts.data)
        counts.data += 1
    counts.data *= idf[counts.indices]
    norms = np.sqrt(np.asarray(counts.multiply(counts).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    counts.data /= np.repeat(norms, np.diff(counts.indptr)).astype(np.float32)
    return counts


# --- per window --------------------------------------------------------------------------


def build_window(con: duckdb.DuckDBPyConnection, win: dict, cfg: dict, labelled: str) -> dict:
    name = win["window"]
    out = FEATURES / name
    out.mkdir(parents=True, exist_ok=True)
    fcfg = cfg["features"]
    tcfg = fcfg["tfidf"]
    seed = cfg["seed"]
    meta = {"window": name, "dates": {s: [str(d) for d in win[s]] for s in split.SPLITS}}

    def in_split(s: str) -> str:
        lo, hi = win[s]
        return f"date_received between date '{lo}' and date '{hi}'"

    # Vocabulary from a seeded sample of train documents.
    sample = con.execute(f"""
        select narrative from {labelled}
        where {in_split("train")}
        order by hash(complaint_id + {seed}) limit {int(tcfg["vocab_sample"])}""").fetchall()
    texts = [r[0] for r in sample]
    del sample
    print(f"{name}: vocabulary from {len(texts):,} sampled train documents", flush=True)
    with Pool(fcfg["workers"]) as pool:
        doc_freq = Counter()
        for c in pool.imap(_doc_freq, [texts[i : i + CHUNK] for i in range(0, len(texts), CHUNK)]):
            doc_freq.update(c)
    del texts
    vocab_terms = choose_vocab(doc_freq, tcfg["min_df"], tcfg["max_features"])
    del doc_freq
    vocab = {t: i for i, t in enumerate(vocab_terms)}
    _init_vocab(vocab)

    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=fcfg["categorical_min_frequency"],
        sparse_output=True,
        dtype=np.float32,
    )
    idf = None
    with Pool(fcfg["workers"], initializer=_init_vocab, initargs=(vocab,)) as pool:
        for s in split.SPLITS:  # train first: idf and the encoder are fitted on it
            # rows.parquet and labelled.parquet are both sorted by complaint_id, so two
            # filtered scans return the same rows in the same order without a join or sort.
            row_cur, text_cur = con.cursor(), con.cursor()
            row_cur.execute(f"""
                select complaint_id, late, {", ".join(CATEGORICAL)}, {", ".join(NUMERIC)}
                from rows where {in_split(s)}""")
            text_cur.execute(f"select complaint_id, narrative from {labelled} where {in_split(s)}")
            ids, y, cats, nums = [], [], [], []

            def texts(row_cur=row_cur, text_cur=text_cur, ids=ids, y=y, cats=cats, nums=nums):
                for rows in batches(row_cur):
                    narratives = text_cur.fetchmany(len(rows))
                    if [r[0] for r in rows] != [n[0] for n in narratives]:
                        raise RuntimeError("rows and narratives are out of step")
                    ids.extend(r[0] for r in rows)
                    y.extend(r[1] for r in rows)
                    cats.extend(r[2 : 2 + len(CATEGORICAL)] for r in rows)
                    nums.extend(r[2 + len(CATEGORICAL) :] for r in rows)
                    yield [n[1] for n in narratives]
                if text_cur.fetchmany(1):
                    raise RuntimeError("more narratives than rows")

            t0 = time.time()
            counts = count_matrix(pool, texts(), 2 * fcfg["workers"])
            if s == "train":
                idf = idf_from(counts)
                encoder.fit(np.array(cats, dtype=object))
            x_text = tfidf(counts, idf, tcfg["sublinear_tf"])
            sp.save_npz(out / f"tfidf_{s}.npz", x_text, compressed=False)
            x_cat = encoder.transform(np.array(cats, dtype=object)).tocsr()
            sp.save_npz(out / f"cat_{s}.npz", x_cat, compressed=False)
            np.save(out / f"num_{s}.npy", np.array(nums, dtype=np.float32))
            np.save(out / f"y_{s}.npy", np.array(y, dtype=np.int8))
            np.save(out / f"ids_{s}.npy", np.array(ids, dtype=np.int64))
            meta[s] = {
                "rows": len(ids),
                "late": int(sum(y)),
                "prevalence": sum(y) / len(ids) if ids else None,
                "tfidf_nnz": int(x_text.nnz),
            }
            print(
                f"{name} {s}: {len(ids):,} rows, prevalence {meta[s]['prevalence']:.4f}, "
                f"tfidf nnz {x_text.nnz:,} ({time.time() - t0:,.0f}s)",
                flush=True,
            )
            del counts, x_text, x_cat, cats, nums

    (out / "vocab.json").write_text(
        json.dumps({"terms": vocab_terms, "idf": idf.tolist(), "text_version": text.VERSION}),
        encoding="utf-8",
    )
    meta["vocab_size"] = len(vocab_terms)
    meta["categorical_names"] = encoder.get_feature_names_out(CATEGORICAL).tolist()
    meta["numeric_names"] = NUMERIC
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load(window: str, featureset: str, split_name: str) -> dict:
    """Assemble one feature set for one window and split.

    Returns X (CSR, float32), y, ids, feature names, and the column indices of the numeric
    block, which models should scale inside their Pipeline.
    """
    if featureset not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {featureset!r}")
    d = FEATURES / window
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))["terms"]
    num = np.load(d / f"num_{split_name}.npy")
    blocks, names = [], []
    if featureset in ("text", "both"):
        blocks.append(sp.load_npz(d / f"tfidf_{split_name}.npz"))
        names += [f"tfidf:{t}" for t in vocab]
    if featureset in ("struct", "both"):
        blocks.append(sp.load_npz(d / f"cat_{split_name}.npz"))
        names += [f"cat:{c}" for c in meta["categorical_names"]]
    numeric = {"text": TEXT_NUMERIC, "struct": STRUCT_NUMERIC, "both": NUMERIC}[featureset]
    cols = [NUMERIC.index(c) for c in numeric]
    first_numeric = len(names)
    blocks.append(sp.csr_matrix(num[:, cols]))
    names += [f"num:{c}" for c in numeric]
    return {
        "X": sp.hstack(blocks, format="csr", dtype=np.float32),
        "y": np.load(d / f"y_{split_name}.npy"),
        "ids": np.load(d / f"ids_{split_name}.npy"),
        "names": names,
        "numeric_cols": list(range(first_numeric, len(names))),
    }


# --- stage -------------------------------------------------------------------------------


def stale(output: Path, inputs: list[Path]) -> bool:
    return not output.exists() or any(output.stat().st_mtime < p.stat().st_mtime for p in inputs)


def main(cfg: dict) -> int:
    from cres.config import DEFAULT

    profile = json.loads(Path("reports/data_profile.json").read_text(encoding="utf-8"))
    data_end = dt.date.fromisoformat(profile["label"]["data_end"])
    wins = split.from_config(cfg, data_end)
    labelled_path = INTERIM / "labelled.parquet"
    labelled = f"read_parquet('{labelled_path}')"
    live = f"read_parquet('{INTERIM / 'live.parquet'}')"
    code = [Path(__file__), Path(text.__file__), Path(split.__file__)]
    inputs = [labelled_path, DEFAULT, *code]

    FEATURES.mkdir(parents=True, exist_ok=True)
    scratch = FEATURES / "features.duckdb"
    for leftover in (scratch, scratch.with_name(scratch.name + ".wal")):
        leftover.unlink(missing_ok=True)  # scratch database from an interrupted run
    con = duckdb.connect(str(scratch))
    con.execute(f"set temp_directory = '{FEATURES / 'tmp'}'")
    con.execute(f"set memory_limit = '{cfg['duckdb_memory_limit']}'")
    split.tag_rows(con, labelled, wins)
    con.execute(
        f'copy (select * from splits order by complaint_id, "window") '
        f"to '{FEATURES / 'splits.parquet'}' (format parquet)"
    )
    first = min(w["train"][0] for w in wins)

    rows_path = FEATURES / "rows.parquet"
    if stale(rows_path, inputs):
        tagged = (
            f"(select * from {labelled} "
            f"where date_received between date '{first}' and date '{data_end}')"
        )
        ccfg = cfg["features"]["company_history"]
        print("features: company history", flush=True)
        company_history(
            con,
            live,
            tagged,
            cfg["windows"]["gap_days"],
            ccfg["days"],
            ccfg["prior_strength"],
            cfg["label"]["drop_company_response"],
        )
        vader_path = FEATURES / "vader.parquet"
        if vader_path.exists() and stale(vader_path, [labelled_path]):
            vader_path.unlink()  # derived cache; the narratives changed
        build_rows(con, tagged, cfg["features"]["workers"], vader_path)
        con.execute(
            f"copy (select * from rows order by complaint_id) to '{rows_path}' (format parquet)"
        )
        con.execute("drop table rows")
    else:
        print("features: rows.parquet is current")
    con.execute(f"create or replace view rows as select * from read_parquet('{rows_path}')")

    report = {
        "data_end": str(data_end),
        "train_years": cfg["windows"]["train_years"],
        "windows": [],
    }
    for win in wins:
        meta_path = FEATURES / win["window"] / "meta.json"
        if stale(meta_path, [*inputs, rows_path]):
            meta = build_window(con, win, cfg, labelled)
        else:
            print(f"{win['window']}: features are current")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = {k: v for k, v in meta.items() if k not in ("categorical_names", "numeric_names")}
        meta["narrative_share"] = narrative_share(con, live, win, labelled)
        report["windows"].append(meta)
    con.close()
    scratch.unlink()
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def narrative_share(con: duckdb.DuckDBPyConnection, live: str, win: dict, labelled: str) -> dict:
    """Labelled narrative rows / all live complaints, per split date range."""
    out = {}
    for s in split.SPLITS:
        lo, hi = win[s]
        n_live = con.execute(
            f"select count(*) from {live} where date_received between ? and ?", [lo, hi]
        ).fetchone()[0]
        n_lab = con.execute(
            f"select count(*) from {labelled} where date_received between ? and ?", [lo, hi]
        ).fetchone()[0]
        out[s] = n_lab / n_live if n_live else None
    return out
