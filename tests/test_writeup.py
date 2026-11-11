"""The numbers in README.md and MODEL_CARD.md come from reports/, never from a keyboard."""

import re

import pytest

from cres import writeup

# A metric-like number: a decimal in [0, 1], a percentage or a thousands-separated count.
NUMBER = re.compile(r"(?<![\w.])(?:0\.\d+|\d+(?:\.\d+)?%|\d{1,3}(?:,\d{3})+)(?![\w.])")


@pytest.fixture(scope="module")
def inputs():
    return writeup.load()


@pytest.mark.parametrize("path", writeup.FILES, ids=str)
def test_generated_blocks_match_the_reports(path, inputs):
    text = path.read_text(encoding="utf-8")
    assert writeup.BLOCK.search(text), f"{path} has no generated blocks"
    assert writeup.fill(text, inputs) == text, f"{path} is stale: run `make writeup`"


@pytest.mark.parametrize("path", writeup.FILES, ids=str)
def test_no_metric_is_typed_outside_a_generated_block(path):
    outside = writeup.BLOCK.sub("", path.read_text(encoding="utf-8"))
    outside = outside.replace("95% CI", "CI")  # the confidence level, not a result
    assert NUMBER.findall(outside) == []


def test_fill_replaces_only_between_markers(inputs):
    text = "Before 1.\n<!-- generated:deep -->\nold\n<!-- /generated:deep -->\nAfter.\n"
    out = writeup.fill(text, inputs)
    assert out.startswith("Before 1.\n<!-- generated:deep -->\n")
    assert out.endswith("\n<!-- /generated:deep -->\nAfter.\n")
    assert "old" not in out
    assert writeup.fill(out, inputs) == out


def test_unknown_or_unclosed_block_is_an_error(inputs):
    with pytest.raises(KeyError):
        writeup.fill("<!-- generated:nope -->\n<!-- /generated:nope -->", inputs)
    with pytest.raises(ValueError):
        writeup.fill("<!-- generated:v1 -->\n<!-- /generated:v2 -->", inputs)


def test_number_pattern():
    assert NUMBER.findall("PR-AUC 0.614, 3.4% late, 19,020,152 rows") == [
        "0.614",
        "3.4%",
        "19,020,152",
    ]
    assert NUMBER.findall("Python 3.12, 8 GB, 15 days, 2017-05-01, W5") == []
