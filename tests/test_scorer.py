"""The scorer bundle reproduces the fitted Pipeline, using small hand-built data."""

import json
import random
from collections import Counter

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder

from cres import features, models, scorer, text

WORDS = "payment late fee account card loan never received refund called again".split()
COMPANIES = {"Acme": (0.2, 6.0), "Beta": (0.01, 9.5)}
NUMERIC = ["days_to_company", "company_late_rate", "company_log_volume", "urgency", "harm"]
CFG = {
    "features": {"categorical_min_frequency": 3},
    "models": {"lr": {"C": [1.0], "max_iter": 2000, "tol": 1e-8}},
    "scorer": {"digits": 7},
    "seed": 0,
}


def synthetic_rows(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        words = [rng.choice(WORDS) for _ in range(rng.randint(0, 12))]
        if rng.random() < 0.2:
            words += ["urgent", "fraud"]
        product = rng.choice(["Card", "Loan"])
        issue = rng.choice(["Billing", "Fees", "Other"])
        company = rng.choice(list(COMPANIES))
        rows.append(
            {
                "text": " ".join(words),
                # Held variables are constant and each sub-variable follows its parent, so the
                # bundle's baselines equal every row's own contribution.
                "cats": [product, product + " sub", issue, issue + " sub", "IL", "Web", "none"],
                "product": product,
                "issue": issue,
                "company": company,
                "days": rng.choice([0, 0, 1, 4]),
                "y": int(rng.random() < 0.3 + 0.3 * ("late" in words) + 0.2 * (company == "Acme")),
                "id": i,
            }
        )
    return rows


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "FEATURES", tmp_path)
    rows = synthetic_rows(300, 1)
    terms = sorted({g for r in rows for g in text.analyze(r["text"])})
    vocab = {t: i for i, t in enumerate(terms)}
    counts = sp.lil_matrix((len(rows), len(terms)), dtype=np.float32)
    for i, r in enumerate(rows):
        for g, c in Counter(text.analyze(r["text"])).items():
            counts[i, vocab[g]] = c
    counts = counts.tocsr()
    idf = features.idf_from(counts)
    x_text = features.tfidf(counts, idf, sublinear=True)
    enc = OneHotEncoder(sparse_output=True, dtype=np.float32).fit([r["cats"] for r in rows])
    x_cat = enc.transform([r["cats"] for r in rows]).tocsr()
    num = []
    for r in rows:
        lex = text.lexicon_counts(text.tokens(r["text"]))
        num.append([r["days"], *COMPANIES[r["company"]], lex["urgency"], lex["harm"], 0.5])
    d = tmp_path / "W1"
    d.mkdir()
    cat_names = enc.get_feature_names_out(features.CATEGORICAL).tolist()
    (d / "meta.json").write_text(json.dumps({"categorical_names": cat_names}))
    (d / "vocab.json").write_text(json.dumps({"terms": terms, "idf": idf.tolist()}))
    sp.save_npz(d / "tfidf_train.npz", x_text)
    sp.save_npz(d / "cat_train.npz", x_cat)
    np.save(d / "num_train.npy", np.array(num, dtype=np.float32))
    np.save(d / "y_train.npy", np.array([r["y"] for r in rows], dtype=np.int8))
    np.save(d / "ids_train.npy", np.array([r["id"] for r in rows]))
    data = features.load("W1", "both", "train", numeric=NUMERIC)
    pipe = models.fit_lr(
        data["X"], data["y"], data["numeric_cols"], NUMERIC, 1.0, CFG["models"]["lr"], False, 0
    )
    bundle = scorer.build_bundle(
        pipe,
        terms,
        idf.astype(np.float64),
        cat_names,
        NUMERIC,
        x_cat,
        COMPANIES,
        (0.05, 3.0),
        CFG,
        {"digits": 7, "window": "W1"},
    )
    return rows, data, pipe, bundle, terms, cat_names


def test_reference_score_matches_the_pipeline(fitted):
    rows, data, pipe, bundle, _, _ = fitted
    expected = pipe.decision_function(data["X"])
    for i, r in enumerate(rows):
        s = scorer.score(bundle, r["text"], r["product"], r["issue"], r["company"], r["days"])
        assert s["logit"] == pytest.approx(expected[i], abs=1e-4)
        assert s["logit"] == pytest.approx(sum(s["parts"].values()), abs=1e-12)
        assert s["probability"] == pytest.approx(1 / (1 + np.exp(-s["logit"])), abs=1e-12)


def test_apply_bundle_scores_exactly_as_the_bundle(fitted):
    rows, data, pipe, bundle, terms, cat_names = fitted
    scorer.apply_bundle(pipe, bundle, terms, cat_names)
    expected = pipe.decision_function(data["X"])
    for i, r in enumerate(rows[:50]):
        s = scorer.score(bundle, r["text"], r["product"], r["issue"], r["company"], r["days"])
        assert s["logit"] == pytest.approx(expected[i], abs=2e-6)


def test_term_contributions_sum_to_the_text_part(fitted):
    rows, _, _, bundle, _, _ = fitted
    s = scorer.score(bundle, "late fee late fee urgent")
    terms = ["late", "fee", "urgent", "late fee", "fee late", "fee urgent"]
    assert [t["term"] for t in s["terms"]] == terms
    assert s["terms"][0]["count"] == 2
    assert sum(t["tfidf"] ** 2 for t in s["terms"]) == pytest.approx(1)
    assert s["parts"]["text"] == pytest.approx(sum(t["contribution"] for t in s["terms"]))
    assert s["lexicon"] == {"urgency": 1, "harm": 0}


def test_unknown_choices_fall_back_to_defaults(fitted):
    _, _, _, bundle, _, _ = fitted
    s = scorer.score(bundle, "", product="Nope", issue=None, company="Nobody")
    cat = bundle["categorical"]
    assert s["parts"]["product"] == cat["product_default"]
    assert s["parts"]["issue"] == cat["issue_default"]
    assert s["parts"]["text"] == 0
    assert s["parts"]["company"] == scorer.score(bundle, "", company=None)["parts"]["company"]


def test_bundle_lists_dropdowns_and_states_its_rule(fitted):
    _, _, _, bundle, _, _ = fitted
    cat = bundle["categorical"]
    assert set(cat["product"]) == {"Card", "Loan"}
    assert set(cat["issue"]) == {"Billing", "Fees", "Other"}
    assert cat["held"] == ["state", "submitted_via", "tags"]
    assert set(cat["issues_by_product"]["Card"]) <= set(cat["issue"])
    assert all(len(line) < 200 for line in cat["rule"])
    assert bundle["tokenizer"]["hash"] == text.spec_hash()


def test_split_categorical_handles_prefixes():
    names = ["product_Card", "sub_product_Card sub", "issue_Fees", "sub_issue_Fees sub"]
    out = scorer.split_categorical(names)
    assert out["product"] == [(0, "Card")]
    assert out["sub_product"] == [(1, "Card sub")]
    assert out["sub_issue"] == [(3, "Fees sub")]
