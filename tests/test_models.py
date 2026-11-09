import numpy as np
import pytest
import scipy.sparse as sp

from cres import models
from cres.models import NumericBlock, calibrate, cells, fit_lr, read_preds, write_preds


def matrix(numeric: np.ndarray) -> sp.csr_matrix:
    """Two one-hot columns followed by a densely stored numeric block, as features.load does."""
    n, k = numeric.shape
    onehot = sp.csr_matrix(np.eye(2, dtype=np.float32)[np.arange(n) % 2])
    block = sp.csr_matrix(
        (numeric.ravel(), np.tile(np.arange(k, dtype=np.int32), n), np.arange(0, n * k + 1, k)),
        shape=(n, k),
    )
    return sp.hstack([onehot, block], format="csr", dtype=np.float32)


def test_numeric_block_standardises_only_the_numeric_columns():
    num = np.array([[0, 0.01], [3, 0.02], [0, 0.5], [10, 0.001]], dtype=np.float32)
    X = matrix(num)
    out = NumericBlock([2, 3], ["log1p", "logit"]).fit_transform(X).toarray()
    np.testing.assert_array_equal(out[:, :2], X.toarray()[:, :2])
    expected = np.column_stack([np.log1p(num[:, 0]), np.log(num[:, 1] / (1 - num[:, 1]))])
    expected = (expected - expected.mean(0)) / expected.std(0)
    np.testing.assert_allclose(out[:, 2:], expected, atol=1e-5)
    assert X.toarray()[0, 2] == 0  # the input is not modified


def test_numeric_block_uses_training_statistics_on_new_rows():
    block = NumericBlock([2], ["none"]).fit(matrix(np.array([[1.0], [3.0]], dtype=np.float32)))
    out = block.transform(matrix(np.array([[5.0]], dtype=np.float32))).toarray()
    assert out[0, 2] == pytest.approx((5 - 2) / 1)


def test_numeric_block_refuses_implicit_zeros():
    X = sp.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.float32))  # zeros not stored
    with pytest.raises(ValueError, match="stored"):
        NumericBlock([1], ["none"]).fit(X)


def test_lr_converges_and_raises_when_it_cannot():
    rng = np.random.default_rng(0)
    num = rng.normal(size=(400, 2)).astype(np.float32)
    y = (num[:, 0] + rng.normal(size=400) > 0).astype(int)
    X = matrix(num)
    params = {"max_iter": 200, "tol": 1e-6}
    pipe = fit_lr(X, y, [2, 3], ["a", "b"], 1.0, params, False, 0)
    assert pipe[-1].n_iter_.max() < params["max_iter"]
    with pytest.raises(Exception, match="converge"):
        fit_lr(X, y, [2, 3], ["a", "b"], 1e6, {"max_iter": 1, "tol": 1e-12}, False, 0)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_cross_fits_validation_and_fits_test_on_all_of_it(method):
    rng = np.random.default_rng(1)
    y_val = (rng.random(2000) < 0.1).astype(int)
    # Class-weighted scores run high: a positive-heavy shift that calibration should undo.
    val = np.clip(0.5 + 0.3 * y_val + rng.normal(0, 0.15, 2000), 0.01, 0.99)
    test = np.clip(0.5 + rng.normal(0, 0.15, 500), 0.01, 0.99)
    v, t = calibrate(val, y_val, test, method, folds=5, seed=42)
    assert v.shape == val.shape and t.shape == test.shape
    assert abs(v.mean() - y_val.mean()) < 0.02  # calibrated toward the base rate
    assert val.mean() > 0.5
    v2, _ = calibrate(val, y_val, test, method, folds=5, seed=42)
    np.testing.assert_array_equal(v, v2)


def test_prediction_files_round_trip(tmp_path):
    path = tmp_path / "lr__both__W1.parquet"
    write_preds(
        path,
        {"val": np.array([3, 1]), "test": np.array([2])},
        {"val": np.array([0.3, 0.1]), "test": np.array([0.2])},
    )
    back = read_preds(path)
    np.testing.assert_array_equal(back["val"][0], [1, 3])
    np.testing.assert_allclose(back["val"][1], [0.1, 0.3])
    np.testing.assert_allclose(back["test"][1], [0.2])
    assert not list(tmp_path.glob("*.tmp"))


def test_calibrated_cells_follow_their_base_model():
    order = cells(list(models.ALL_MODELS), ["both", "text", "struct"], ["W1"])
    assert order[0] == ("company_rate", "struct", "W1")
    assert [c[1] for c in order if c[0] == "lr"] == ["struct", "text", "both"]
    names = [c[0] for c in order if c[1] == "both"]
    assert names.index("lr_balanced") < names.index("lr_balanced_platt")


@pytest.mark.parametrize(
    ("peak", "expected_fits"),
    [
        (0.01, [0.1, 1.0, 0.01]),
        (0.1, [0.1, 1.0, 0.01]),
        (1.0, [0.1, 1.0, 10.0]),
        (10.0, [0.1, 1.0, 10.0]),
    ],
)
def test_c_search_climbs_toward_the_validation_peak(peak, expected_fits):
    fitted = []

    def score(C):
        fitted.append(C)
        return -abs(np.log10(C) - np.log10(peak)), f"model at {C}"

    best, model = models.search_c([10.0, 0.01, 1.0, 0.1], score)
    assert best == peak and model == f"model at {peak}"
    assert fitted == expected_fits


def test_c_search_walks_past_one_step_when_scores_keep_improving():
    fitted = []

    def score(C):
        fitted.append(C)
        return C, None

    assert models.search_c([0.01, 0.1, 1, 10, 100], score)[0] == 100
    assert fitted == [1, 10, 100]
