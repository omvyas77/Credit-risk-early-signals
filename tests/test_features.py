import datetime as dt
import json
import math

import duckdb
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from cres import features, text
from cres.features import (
    BANNED,
    CATEGORICAL,
    NUMERIC,
    STRUCT_NUMERIC,
    TEXT_NUMERIC,
    build_rows,
    choose_vocab,
    company_history,
    idf_from,
    tfidf,
)

D = dt.date


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("""create table live (company varchar, date_received date,
        timely_response varchar, company_response_to_consumer varchar)""")
    con.executemany(
        "insert into live values (?, ?, ?, ?)",
        [
            ["A", D(2025, 1, 30), "No", "Closed"],  # on the cutoff: excluded
            ["A", D(2025, 1, 29), "No", "Closed"],  # day before the cutoff: counted
            ["A", D(2024, 1, 31), "Yes", "Closed"],  # first day of the 365-day window: counted
            ["A", D(2024, 1, 30), "No", "Closed"],  # before the window: excluded
            ["A", D(2025, 1, 29), "No", "In progress"],  # unsettled: excluded
            ["A", D(2025, 3, 1), "No", "Closed"],  # the row itself, after the cutoff
            ["B", D(2025, 1, 1), "No", "Closed"],  # other company: overall rate only
        ],
    )
    con.execute("create table r (complaint_id bigint, company varchar, date_received date)")
    con.executemany(
        "insert into r values (?, ?, ?)", [[1, "A", D(2025, 3, 1)], [2, "C", D(2025, 3, 1)]]
    )
    return con


def history(con):
    company_history(
        con, "live", "r", gap_days=30, days=365, prior_strength=50, unsettled=["In progress"]
    )
    return {r[0]: r[1:] for r in con.execute("select * from company_history").fetchall()}


def test_company_history_only_looks_back_past_the_gap(con):
    h = history(con)
    prior = 2 / 3  # overall in [2024-01-31, 2025-01-30): A's 2 settled rows and B's 1
    assert h[1][0] == pytest.approx((1 + 50 * prior) / (2 + 50))
    assert h[1][1] == pytest.approx(math.log(1 + 2))


def test_company_without_history_gets_the_overall_rate(con):
    h = history(con)
    assert h[2][0] == pytest.approx(2 / 3)
    assert h[2][1] == 0


def test_later_complaints_never_change_history(con):
    before = history(con)
    con.execute("insert into live values ('A', '2025-02-15', 'No', 'Closed')")
    con.execute("insert into live values ('A', '2026-01-01', 'No', 'Closed')")
    assert history(con) == before


def test_no_banned_column_is_a_feature():
    assert not (set(CATEGORICAL) | set(NUMERIC)) & BANNED
    assert set(NUMERIC) == set(STRUCT_NUMERIC) | set(TEXT_NUMERIC)


def test_build_rows_refuses_outcome_columns():
    con = duckdb.connect()
    con.execute(
        "create table bad (complaint_id bigint, narrative varchar, timely_response varchar)"
    )
    with pytest.raises(ValueError, match="timely_response"):
        build_rows(con, "bad", workers=1, vader_path=None)


def test_choose_vocab_orders_by_df_then_term():
    df = {"b": 5, "a": 5, "c": 9, "rare": 1}
    assert choose_vocab(df, min_df=2, max_features=2) == ["c", "a"]


def test_tfidf_matches_sklearn():
    train = ["I did not authorize this charge", "the charge is not mine not mine", "thanks"]
    vec = TfidfVectorizer(analyzer=text.analyze, sublinear_tf=True, norm="l2", smooth_idf=True)
    expected = vec.fit_transform(train).toarray()
    vocab = {t: i for i, t in enumerate(vec.get_feature_names_out())}
    features._init_vocab(vocab)
    lengths, indices, counts = features._counts(train)
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    m = sp.csr_matrix((counts, indices, indptr), shape=(len(train), len(vocab)))
    idf = idf_from(m)
    np.testing.assert_allclose(idf, vec.idf_, rtol=1e-6)
    np.testing.assert_allclose(tfidf(m, idf, sublinear=True).toarray(), expected, atol=1e-6)


def test_load_assembles_each_feature_set(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "FEATURES", tmp_path)
    d = tmp_path / "W1"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"categorical_names": ["product_x", "product_y"]}))
    (d / "vocab.json").write_text(json.dumps({"terms": ["fraud", "not mine", "late"]}))
    sp.save_npz(d / "tfidf_val.npz", sp.csr_matrix(np.ones((2, 3), dtype=np.float32)))
    sp.save_npz(d / "cat_val.npz", sp.csr_matrix(np.eye(2, dtype=np.float32)))
    np.save(d / "num_val.npy", np.arange(12, dtype=np.float32).reshape(2, 6))
    np.save(d / "y_val.npy", np.array([0, 1], dtype=np.int8))
    np.save(d / "ids_val.npy", np.array([10, 11]))
    shapes = {}
    for fs in features.FEATURE_SETS:
        m = features.load("W1", fs, "val")
        shapes[fs] = m["X"].shape
        assert len(m["names"]) == m["X"].shape[1]
        assert [m["names"][i] for i in m["numeric_cols"]] == [
            n for n in m["names"] if n.startswith("num:")
        ]
        # "late" as a word in the vocabulary is fine; no column may be an outcome.
        cols = [n.split(":", 1)[1] for n in m["names"] if not n.startswith("tfidf:")]
        assert not set(cols) & BANNED
        # The numeric block is stored densely (the 0 in row 0 is an explicit entry).
        numeric = m["X"][:, m["numeric_cols"]]
        assert numeric.nnz == 2 * len(m["numeric_cols"])
    assert shapes == {"text": (2, 6), "struct": (2, 5), "both": (2, 11)}
    capped = features.load("W1", "both", "val", max_terms=2)
    assert capped["X"].shape == (2, 10)
    assert [n for n in capped["names"] if n.startswith("tfidf:")] == [
        "tfidf:fraud",
        "tfidf:not mine",
    ]
    text_names = features.load("W1", "text", "val")["names"]
    assert not any(n.startswith("cat:") for n in text_names)
    assert "num:company_late_rate" not in text_names
    subset = features.load("W1", "both", "val", rows=np.array([1]))
    np.testing.assert_array_equal(
        subset["X"].toarray(), features.load("W1", "both", "val")["X"][[1]].toarray()
    )
    assert subset["ids"].tolist() == [11] and subset["y"].tolist() == [1]
