"""docs/js/scorer.js and the Python reference (cres.scorer.score, float64) agree on 1,000
synthetic texts: identical tokens and lexicon counts, probabilities within 1e-6."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from parity_fixtures import OUT, generate

from cres import config, scorer, text

BUNDLE = Path("docs/scorer/model.json")
RUNNER = Path(__file__).parent / "scorer_parity.mjs"
TOL = 1e-6


def node() -> str:
    exe = shutil.which("node")
    if exe is None:
        if os.environ.get("CI"):
            pytest.fail("node is required for the scorer parity test in CI")
        pytest.skip("node is not installed")
    return exe


def test_fixtures_match_their_generator():
    texts = json.loads(OUT.read_text(encoding="utf-8"))
    assert len(texts) == 1000
    assert texts == generate(config.load()["seed"])


def inputs(bundle: dict, texts: list[str]) -> list[dict]:
    """Each text with a product, issue, company and days to company, cycling through the
    bundle's choices and including unlisted and missing values."""
    cat = bundle["categorical"]
    products = [*sorted(cat["product"]), None, "Not a listed product"]
    companies = [*sorted(bundle["companies"]), None, "Not a listed company"]
    days = [None, 0, 1, 3, 16, 446]
    out = []
    for i, t in enumerate(texts):
        product = products[i % len(products)]
        issues = [*cat["issues_by_product"].get(product, []), None, "Not a listed issue"]
        out.append(
            {
                "text": t,
                "product": product,
                "issue": issues[(i // len(products)) % len(issues)],
                "company": companies[(i * 7) % len(companies)],
                "daysToCompany": days[i % len(days)],
            }
        )
    return out


def test_scorer_js_matches_python(tmp_path):
    exe = node()
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    cases = inputs(bundle, json.loads(OUT.read_text(encoding="utf-8")))
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    run = subprocess.run(
        [exe, str(RUNNER), str(BUNDLE), str(path)], capture_output=True, text=True, check=True
    )
    js = json.loads(run.stdout)
    assert len(js) == len(cases)
    worst = 0.0
    for case, got in zip(cases, js, strict=True):
        want = scorer.score(
            bundle,
            case["text"],
            case["product"],
            case["issue"],
            case["company"],
            case["daysToCompany"],
        )
        where = repr(case["text"][:60])
        assert got["tokens"] == want["tokens"], where
        assert got["lexicon"] == want["lexicon"], where
        assert [(t["term"], t["count"]) for t in got["terms"]] == [
            (t["term"], t["count"]) for t in want["terms"]
        ], where
        for g, w in zip(got["terms"], want["terms"], strict=True):
            assert abs(g["contribution"] - w["contribution"]) <= TOL, where
        for k, v in want["parts"].items():
            assert abs(got["parts"][k] - v) <= TOL, (where, k)
        assert abs(got["logit"] - want["logit"]) <= TOL, where
        assert abs(got["probability"] - want["probability"]) <= TOL, where
        worst = max(worst, abs(got["probability"] - want["probability"]))
    print(f"largest probability gap: {worst:.2e}")


def test_parity_cases_exercise_the_model():
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    texts = json.loads(OUT.read_text(encoding="utf-8"))
    scored = [scorer.score(bundle, t) for t in texts]
    assert sum(bool(s["terms"]) for s in scored) > 900  # most texts hit the vocabulary
    assert sum(s["lexicon"]["harm"] > 0 for s in scored) > 50
    assert sum(not s["tokens"] for s in scored) >= 3  # empty, blank and redaction-only texts


def test_scorer_js_implements_this_tokenizer():
    js = Path("docs/js/scorer.js").read_text(encoding="utf-8")
    assert re.search(r'TOKENIZER_HASH = "([0-9a-f]+)"', js).group(1) == text.spec_hash()
    assert re.search(r"TOKENIZER_VERSION = (\d+)", js).group(1) == str(text.VERSION)


def test_shipped_bundle_matches_this_tokenizer():
    bundle = json.loads(Path("docs/scorer/model.json").read_text(encoding="utf-8"))
    assert bundle["tokenizer"]["hash"] == text.spec_hash()
    assert bundle["tokenizer"]["lexicon"] == text.LEXICON
    assert "vader" not in [n["name"] for n in bundle["numeric"]]
