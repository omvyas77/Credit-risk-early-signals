import numpy as np
import pytest
import scipy.sparse as sp

from cres import explain
from cres.models import fit_lr

NAMES = [
    "tfidf:refund",
    "tfidf:not mine",
    "cat:product_Mortgage",
    "cat:sub_product_Conventional home mortgage",
    "cat:issue_Trouble during payment process",
    "cat:sub_issue_Problem with fees",
    "cat:state_IL",
    "cat:submitted_via_Web",
    "cat:tags_Servicemember",
    "num:days_to_company",
    "num:company_late_rate",
    "num:company_log_volume",
    "num:urgency",
    "num:harm",
    "num:vader",
]


def test_every_feature_kind_gets_its_group():
    assert [explain.group_of(n) for n in NAMES] == [
        "text",
        "text",
        "product",
        "product",  # sub-product is not read as product "sub_product..."
        "issue",
        "issue",
        "state",
        "channel",
        "tags",
        "days to company",
        "company history",
        "company history",
        "lexicon",
        "lexicon",
        "sentiment",
    ]
    with pytest.raises(ValueError, match="no group"):
        explain.group_of("num:timely_response")


def test_group_matrix_sums_weighted_features_per_group():
    names = ["tfidf:a", "num:urgency", "tfidf:b", "num:harm"]
    G, groups = explain.group_matrix(names, np.array([1.0, 2.0, 3.0, 4.0]))
    assert groups == ["text", "lexicon"]  # GROUPS order, only groups present
    X = sp.csr_matrix(np.array([[1, 1, 1, 1], [0, 2, 1, 0]], dtype=float))
    np.testing.assert_array_equal((X @ G).toarray(), [[4, 6], [3, 4]])


def _toy(n: int = 600, seed: int = 0):
    """Two TF-IDF columns, two product one-hots and the company late rate; only the rate
    carries signal."""
    rng = np.random.default_rng(seed)
    rate = rng.uniform(0.001, 0.5, n)
    y = (rng.random(n) < rate).astype(int)
    words = rng.random((n, 2)) * (rng.random((n, 2)) < 0.3)
    product = np.eye(2)[rng.integers(0, 2, n)]
    X = sp.csr_matrix(np.column_stack([words, product, rate]).astype(np.float32))
    names = ["tfidf:a", "tfidf:b", "cat:product_A", "cat:product_B", "num:company_late_rate"]
    return X, y, names


def test_lr_groups_shares_add_up_and_follow_the_signal():
    X, y, names = _toy()
    pipe = fit_lr(X, y, [4], ["company_late_rate"], 1.0, {"max_iter": 500, "tol": 1e-6}, False, 0)
    out = explain.lr_groups(pipe, X, y, names)
    groups = {g["group"]: g for g in out["groups"]}
    assert list(groups) == ["text", "product", "company history"]
    assert sum(g["share"] for g in groups.values()) == pytest.approx(1)
    assert [groups[g]["features"] for g in groups] == [2, 2, 1]
    assert max(groups, key=lambda g: groups[g]["share"]) == "company history"
    # Late complaints get a higher company-history contribution than on-time ones.
    assert groups["company history"]["mean_late"] > groups["company history"]["mean_on_time"]
    # Centered group contributions plus the mean decision value rebuild the decision value.
    G, _ = explain.group_matrix(names, pipe[-1].coef_[0])
    contrib = (pipe[:-1].transform(X) @ G).toarray()
    decision = pipe.decision_function(X)
    rebuilt = (contrib - contrib.mean(0)).sum(1) + decision.mean()
    np.testing.assert_allclose(rebuilt, decision, atol=1e-6)


def test_lr_groups_refuses_mismatched_names():
    X, y, names = _toy()
    pipe = fit_lr(X, y, [4], ["company_late_rate"], 1.0, {"max_iter": 500, "tol": 1e-6}, False, 0)
    with pytest.raises(ValueError, match="coefficients"):
        explain.lr_groups(pipe, X, y, names[1:])


def test_top_terms_ranks_text_coefficients_and_counts_documents():
    names = ["tfidf:refund", "tfidf:thank", "tfidf:fraud", "cat:product_A"]
    coef = np.array([0.5, -2.0, 3.0, 9.0])  # the category's coefficient is ignored
    X = sp.csr_matrix(np.array([[1, 0, 1, 1], [1, 1, 0, 1], [0, 0, 0, 1]], dtype=float))
    out = explain.top_terms(coef, names, X, k=2)
    assert [t["term"] for t in out["up"]] == ["fraud", "refund"]
    assert [t["term"] for t in out["down"]] == ["thank", "refund"]
    assert {t["term"]: t["test_docs"] for t in out["up"]} == {"fraud": 1, "refund": 2}


def test_shap_groups_is_seeded_and_shares_add_up():
    lgb = pytest.importorskip("lightgbm")
    X, y, names = _toy(n=2000)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 0, "min_child_samples": 20},
        lgb.Dataset(X, y),
        num_boost_round=20,
    )
    a = explain.shap_groups(booster, X, y, names, 300, [42, 5])
    b = explain.shap_groups(booster, X, y, names, 300, [42, 5])
    assert a == b
    assert a["sample_rows"] == 300
    assert sum(g["share"] for g in a["groups"]) == pytest.approx(1)
    assert a["top_features"][0]["feature"] == "num:company_late_rate"
    assert explain.shap_groups(booster, X, y, names, 10_000, [42, 5])["sample_rows"] == 2000
