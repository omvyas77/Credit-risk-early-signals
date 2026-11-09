import pytest

from cres.text import LEXICON, analyze, lexicon_counts, tokens


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I can't PAY", ["i", "can't", "pay"]),
        ("On XX/XX/2024 XXXX XXXX bank", ["on", "bank"]),
        ("Exxon xx-ray", ["exxon", "xx", "ray"]),  # lowercase x's are text
        ("EXXON", ["e", "on"]),  # documented cost of the capital-X rule
        ("acct XXXX1234 and XXXX_acct, 12XXXX", ["acct", "and", "acct"]),  # glued to digits
        ("XXXXaccount XXXXThe NUMBERXXXX", ["account", "the", "number"]),  # glued to words
        ("See HTTPS://Example.com/X now", ["see", "now"]),  # URLs in any case
        ("see https://example.com/a?b=1 or www.example.org now", ["see", "or", "now"]),
        ("mail me@example.com today", ["mail", "today"]),
        ("ｆｒａｕｄ ﬁle", ["fraud", "file"]),  # NFKC folds full-width letters and ligatures
        ("'quoted' consumer's 3rd", ["quoted", "consumer's", "rd"]),
        ("café", ["caf"]),  # letters outside a-z separate tokens
        ("not\tmine\n", ["not", "mine"]),
        ("", []),
    ],
)
def test_tokens(text, expected):
    assert tokens(text) == expected


def test_analyze_adds_bigrams_after_unigrams():
    assert analyze("did not pay") == ["did", "not", "pay", "did not", "not pay"]


@pytest.mark.parametrize(("group", "phrase"), [(g, p) for g, ps in LEXICON.items() for p in ps])
def test_each_phrase_matches_on_whole_tokens(group, phrase):
    assert lexicon_counts(tokens(f"Start. {phrase.upper()}, end"))[group] == 1
    # Glued to other letters it must not match.
    assert lexicon_counts(tokens(f"pre{phrase}post"))[group] == 0


def test_counts_repeats_and_keeps_negation():
    counts = lexicon_counts(tokens("This is not mine. NOT MINE! I did not authorize it"))
    assert counts == {"urgency": 0, "harm": 3}


def test_no_substring_matches():
    assert lexicon_counts(tokens("helpful fraudulently unauthorizedly")) == {
        "urgency": 0,
        "harm": 0,
    }


def test_lexicon_has_no_product_names():
    products = {"credit card", "mortgage", "credit report", "debt collection", "student loan"}
    phrases = {p for ps in LEXICON.values() for p in ps}
    assert not phrases & products
