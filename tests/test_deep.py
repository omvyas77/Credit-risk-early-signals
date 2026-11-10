"""Deep-model plumbing, with the models mocked: no torch, no real narratives."""

import json

import duckdb
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

from cres import config, deep, features, models

SIZES = {"train": 240, "val": 160, "test": 120}


def make_cache(root, window: str, first_id: int, seed: int = 0) -> dict:
    """A feature cache for one window: two one-hot columns, the six numeric columns, ids, y.

    Every split has the same prevalence; the label leans on the second numeric column."""
    rng = np.random.default_rng(seed)
    d = root / window
    d.mkdir(parents=True)
    out, next_id = {}, first_id
    for s, n in SIZES.items():
        ids = np.arange(next_id, next_id + n, dtype=np.int64)
        next_id += n
        y = (rng.random(n) < 0.2).astype(np.int8)
        num = rng.normal(size=(n, len(features.NUMERIC))).astype(np.float32)
        num[:, 1] = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.1, n), 0.01, 0.99)
        cat = sp.csr_matrix(np.eye(2, dtype=np.float32)[rng.integers(0, 2, n)])
        sp.save_npz(d / f"cat_{s}.npz", cat, compressed=False)
        np.save(d / f"num_{s}.npy", num)
        np.save(d / f"ids_{s}.npy", ids)
        np.save(d / f"y_{s}.npy", y)
        out[s] = (ids, y)
    meta = {"window": window, "categorical_names": ["product_a", "product_b"]}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    sp.save_npz(d / "tfidf_train.npz", sp.csr_matrix((SIZES["train"], 3), dtype=np.float32))
    return out


def story(cid: int, late: int) -> str:
    """A made-up narrative; late ones say so, with a redaction to clean."""
    tail = "They never replied to me." if late else "They answered within a week."
    return f"Complaint {cid} about my account on XX/XX/XXXX.  {tail} See https://example.org"


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Feature cache with windows W1 and W2, labelled.parquet and splits.parquet in tmp_path,
    with the module paths pointed there."""
    root = tmp_path / "features"
    w1 = make_cache(root, "W1", 1_000, seed=1)
    w2 = make_cache(root, "W2", 5_000, seed=2)
    tags = [(cid, w, s) for w, c in (("W1", w1), ("W2", w2)) for s in SIZES for cid in c[s][0]]
    labels = {
        int(c): int(v)
        for w in (w1, w2)
        for ids, y in w.values()
        for c, v in zip(ids, y, strict=True)
    }
    con = duckdb.connect()
    tag_table = {  # noqa: F841 (scanned by DuckDB)
        "complaint_id": np.array([t[0] for t in tags], dtype=np.int64),
        "window": np.array([t[1] for t in tags]),
        "split": np.array([t[2] for t in tags]),
    }
    con.execute(f"copy (select * from tag_table) to '{root / 'splits.parquet'}' (format parquet)")
    extra = 99_999  # a labelled row in no window
    ids = sorted(labels) + [extra]
    labelled = {  # noqa: F841 (scanned by DuckDB)
        "complaint_id": np.array(ids, dtype=np.int64),
        "narrative": np.array([story(c, labels.get(c, 0)) for c in ids]),
        "product": np.array(["Mortgage"] * len(ids)),
    }
    path = tmp_path / "labelled.parquet"
    con.execute(f"copy (select * from labelled) to '{path}' (format parquet)")
    con.close()
    monkeypatch.setattr(deep, "INPUT", tmp_path / "deep_input")
    monkeypatch.setattr(deep, "WORK", tmp_path / "deep")
    monkeypatch.setattr(models, "PREDS", tmp_path / "preds")
    monkeypatch.setattr(models, "MODELS", tmp_path / "models")
    return {"root": root, "labelled": path, "W2": w2, "tmp": tmp_path}


def small_cfg() -> dict:
    cfg = config.load()
    cfg["models"]["lr"]["C"] = [0.1, 1.0]
    cfg["deep"]["negatives"] = 60
    cfg["deep"]["minilm"]["chunk"] = 70  # several chunks per split
    return cfg


def export(p) -> dict:
    return deep.export_input(["W2"], p["labelled"], p["root"], deep.INPUT, "1GB")


# --- export --------------------------------------------------------------------------------


def test_export_writes_only_the_windows_rows_and_columns(pipeline):
    sizes = export(pipeline)
    con = duckdb.connect()
    narr = con.execute(f"select * from read_parquet('{deep.INPUT / 'narratives.parquet'}')")
    assert [c[0] for c in narr.description] == ["complaint_id", "narrative"]
    got = sorted(r[0] for r in narr.fetchall())
    want = sorted(int(c) for ids, _ in pipeline["W2"].values() for c in ids)
    assert got == want  # no W1 rows, no untagged row
    tags = con.execute(
        f"select distinct \"window\" from read_parquet('{deep.INPUT / 'splits.parquet'}')"
    ).fetchall()
    assert tags == [("W2",)]
    assert sorted(p.name for p in (deep.INPUT / "W2").iterdir()) == sorted(deep.STRUCT_FILES)
    assert not (deep.INPUT / "W1").exists()
    assert "W2/tfidf_train.npz" not in sizes and sizes["narratives.parquet"] > 0
    assert not (deep.INPUT / "tmp").exists()


def test_export_rejects_a_bad_window_name(pipeline):
    with pytest.raises(ValueError, match="window"):
        deep.export_input(["W2'; drop"], pipeline["labelled"], pipeline["root"], deep.INPUT, "1GB")


def test_narratives_come_back_in_the_order_asked(pipeline):
    ids = np.array([5_003, 5_000, 5_241])
    texts = deep.narratives(pipeline["labelled"], ids, "1GB")
    assert [t.split()[1] for t in texts] == ["5003", "5000", "5241"]
    with pytest.raises(RuntimeError, match="found 1 of 2"):
        deep.narratives(pipeline["labelled"], np.array([5_000, 123]), "1GB")


def test_prepare_cleans_like_the_tokenizer_and_keeps_case():
    out = deep.prepare("Paid  XXXX1234 to me@bank.com on XX/XX/2025,\nsee www.x.org now")
    assert out == "Paid 1234 to on / /2025, see now"


# --- subsample and prior -------------------------------------------------------------------


def test_subsample_keeps_every_positive_and_a_seeded_set_of_negatives():
    y = np.array([0] * 90 + [1] * 10)
    keep = deep.subsample(y, 20, seed=42, window="W5")
    assert np.all(np.diff(keep) > 0)
    assert set(np.flatnonzero(y == 1)) <= set(keep)
    assert (y[keep] == 0).sum() == 20
    np.testing.assert_array_equal(keep, deep.subsample(y, 20, seed=42, window="W5"))
    assert not np.array_equal(keep, deep.subsample(y, 20, seed=7, window="W5"))
    assert not np.array_equal(keep, deep.subsample(y, 20, seed=42, window="W4"))
    np.testing.assert_array_equal(deep.subsample(y, 500, 42, "W5"), np.arange(100))


def test_subsample_reads_the_config_seed_and_size(pipeline):
    cfg = small_cfg()
    y = np.load(pipeline["root"] / "W2" / "y_train.npy")
    keep = deep.subsample(y, cfg["deep"]["negatives"], cfg["seed"], "W2")
    assert (y[keep] == 0).sum() == cfg["deep"]["negatives"]


def test_prior_shift_undoes_negative_sampling_and_keeps_the_order():
    s = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(deep.prior_shift(s, 10, 10), s)
    out = deep.prior_shift(s, 10, 100)
    assert np.all(np.diff(out) > 0) and np.all(out < s)
    # Odds fall by the sampling rate.
    np.testing.assert_allclose(out / (1 - out), s / (1 - s) * 0.1)


# --- stacking ------------------------------------------------------------------------------


def test_crossfit_scores_each_row_with_a_model_that_never_saw_it():
    X = np.arange(50, dtype=float)[:, None]
    y = np.array([0, 1] * 25)
    seen = []

    def fit(X_fit, y_fit):
        fitted_on = set(X_fit[:, 0].astype(int))
        seen.append(fitted_on)

        def score(X_out):
            rows = set(X_out[:, 0].astype(int))
            assert not rows & fitted_on
            assert rows | fitted_on == set(range(50))
            return X_out[:, 0] + 1000 * len(seen)

        return score

    out = deep.crossfit(fit, X, y, folds=5, seed=0)
    assert len(seen) == 5
    folds = (out // 1000).astype(int)
    assert sorted(np.bincount(folds)[1:]) == [10] * 5  # every row scored once, by one fold
    np.testing.assert_array_equal(out % 1000, np.arange(50))


def test_stack_on_validation_cross_fits_val_and_fits_test_on_all_of_it(pipeline):
    cfg = small_cfg()
    rng = np.random.default_rng(3)
    root = pipeline["root"]
    struct = {s: features.load("W2", "struct", s, root=root) for s in ("val", "test")}
    score = {
        s: np.clip(0.2 + 0.4 * struct[s]["y"] + rng.normal(0, 0.2, len(struct[s]["y"])), 0.01, 0.99)
        for s in ("val", "test")
    }
    y_val = struct["val"]["y"]
    val_out, test_out, note = deep.stack(
        score["val"], struct["val"], y_val, score["test"], struct["test"], cfg
    )
    assert val_out.shape == y_val.shape and test_out.shape == struct["test"]["y"].shape
    assert average_precision_score(struct["test"]["y"], test_out) > 0.5
    # Test scores come from one LR on all of validation at the chosen C.
    X_val, cols, names = deep.with_score(struct["val"], score["val"])
    X_test, _, _ = deep.with_score(struct["test"], score["test"])
    pipe = models.fit_lr(
        X_val, y_val, cols, names, note["C"], cfg["models"]["lr"], False, cfg["seed"]
    )
    np.testing.assert_allclose(test_out, pipe.predict_proba(X_test)[:, 1])
    # Validation scores are out of fold, so they differ from that in-sample fit.
    assert not np.allclose(val_out, pipe.predict_proba(X_val)[:, 1])
    assert note["stacked_on"] == "val" and set(note["by_C"]) == {"0.1", "1.0"}


def test_with_score_appends_a_stored_logit_column_last(pipeline):
    struct = features.load("W2", "struct", "val", root=pipeline["root"])
    score = np.full(len(struct["y"]), 0.5)  # logit 0: must still be stored
    X, cols, names = deep.with_score(struct, score)
    assert X.shape[1] == struct["X"].shape[1] + 1
    assert cols[-1] == X.shape[1] - 1 and names[-1] == "text_score"
    assert X[:, -1].nnz == X.shape[0]


def test_hstack_dense_matches_scipy():
    rng = np.random.default_rng(4)
    emb = rng.normal(size=(23, 4)).astype(np.float32)
    dense = rng.random((23, 5)) * (rng.random((23, 5)) < 0.3)
    dense[[3, 10]] = 0  # rows with no sparse entries
    X = sp.csr_matrix(dense.astype(np.float32))
    mean, scale = deep.moments(emb, chunk=5)
    np.testing.assert_allclose(mean, emb.mean(0), atol=1e-6)
    np.testing.assert_allclose(scale, emb.std(0), atol=1e-5)
    got = deep.hstack_dense(emb, mean, scale, X, chunk=6)
    want = np.hstack([(emb - mean) / scale, dense])
    np.testing.assert_allclose(got.toarray(), want, atol=1e-5)
    assert got.nnz == emb.size + X.nnz
    np.testing.assert_allclose(deep.standardise(emb, mean, scale, chunk=5), want[:, :4], atol=1e-5)


# --- runs with mocked models ---------------------------------------------------------------


def fake_encoder(calls: list):
    """Stands in for MiniLM: a few hashed word counts, plus a 'never replied' flag."""

    def encode(texts):
        calls.append(len(texts))
        assert all("XX" not in t and "https" not in t for t in texts)
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = "never replied" in t
            out[i, 1:] = [len(t) % 7, t.count("a") % 5, hash(t.split()[1]) % 3]
        return out

    return encode


def test_minilm_run_writes_both_cells_and_resumes_from_its_cache(pipeline, monkeypatch, capsys):
    cfg = small_cfg()
    export(pipeline)
    calls = []
    monkeypatch.setattr(deep, "load_minilm", lambda cfg: fake_encoder(calls))
    assert deep.main(cfg, "minilm", "W2") == 0
    assert calls == [70, 70, 70, 30, 70, 70, 20, 70, 50]  # chunked per split
    for fs in ("text", "both"):
        preds = models.read_preds(models.pred_path("minilm", fs, "W2"))
        for s in ("val", "test"):
            np.testing.assert_array_equal(preds[s][0], pipeline["W2"][s][0])
            y = pipeline["W2"][s][1]
            assert average_precision_score(y, preds[s][1]) > 0.9
        note = json.loads((models.MODELS / f"minilm__{fs}__W2.json").read_text())
        assert note["C"] in cfg["models"]["lr"]["C"] and note["trained_on"] == "all train rows"
    assert not list(deep.WORK.rglob("*.part*"))
    # A rerun with --force reads the cached embeddings instead of encoding again.
    calls.clear()
    assert deep.main(cfg, "minilm", "W2", force=True) == 0
    assert calls == []
    assert deep.main(cfg, "minilm", "W2") == 0
    assert "exist" in capsys.readouterr().out


def test_distilbert_run_trains_on_the_subsample_and_stacks(pipeline, monkeypatch):
    cfg = small_cfg()
    export(pipeline)
    seen = {}

    def fake_fit(texts, y_train, y_val, cfg_, work):
        seen.update(texts=texts, y_train=y_train)
        raw = {
            s: np.array([0.8 if "never replied" in t else 0.3 for t in texts[s]])
            for s in ("val", "test")
        }
        return raw["val"], raw["test"], {"best_epoch": 1, "history": []}

    monkeypatch.setattr(deep, "fit_distilbert", fake_fit)
    assert deep.main(cfg, "distilbert", "W2") == 0
    ids, y = pipeline["W2"]["train"]
    keep = deep.subsample(y, cfg["deep"]["negatives"], cfg["seed"], "W2")
    assert len(seen["texts"]["train"]) == len(keep)
    np.testing.assert_array_equal(seen["y_train"], y[keep])
    assert [int(t.split()[1]) for t in seen["texts"]["train"]] == list(ids[keep])

    text = models.read_preds(models.pred_path("distilbert", "text", "W2"))
    val_ids, y_val = pipeline["W2"]["val"]
    order = np.argsort(val_ids)
    raw = np.where(y_val[order] == 1, 0.8, 0.3)
    n_neg = int((y == 0).sum())
    np.testing.assert_allclose(text["val"][1], deep.prior_shift(raw, 60, n_neg))
    both = json.loads((models.MODELS / "distilbert__both__W2.json").read_text())
    assert both["stacked_on"] == "val" and both["subsample"]["negatives"] == 60
    assert models.pred_path("distilbert", "both", "W2").exists()
    run = json.loads((deep.WORK / "W2" / "distilbert" / "run.json").read_text())
    assert run["subsample"]["ids_sha256"] == both["subsample"]["ids_sha256"]


def test_lr_sub_trains_on_the_same_subsample_as_distilbert(pipeline, monkeypatch):
    """lr_sub reads the local feature cache; its subsample fingerprint must equal the one a
    DistilBERT run on the exported input records."""
    cfg = small_cfg()
    root = pipeline["root"]
    # A minimal TF-IDF block, one column carrying the label, for each split.
    for s, (_, y) in pipeline["W2"].items():
        x = sp.csr_matrix(np.column_stack([y, 1 - y, np.ones(len(y))]).astype(np.float32))
        sp.save_npz(root / "W2" / f"tfidf_{s}.npz", x, compressed=False)
    (root / "W2" / "vocab.json").write_text(json.dumps({"terms": ["late", "on", "time"]}))
    monkeypatch.setattr(features, "FEATURES", root)
    assert deep.main(cfg, "lr_sub", "W2") == 0
    note = json.loads((models.MODELS / "lr_sub__text__W2.json").read_text())
    ids, y = pipeline["W2"]["train"]
    keep = deep.subsample(y, cfg["deep"]["negatives"], cfg["seed"], "W2")
    assert note["subsample"] == deep.subsample_note(ids, y, keep)
    for fs in ("text", "both"):
        preds = models.read_preds(models.pred_path("lr_sub", fs, "W2"))
        np.testing.assert_array_equal(preds["test"][0], pipeline["W2"]["test"][0])


def test_unknown_model_is_refused(pipeline):
    with pytest.raises(SystemExit, match="unknown model"):
        deep.main(small_cfg(), "gpt", "W2")
