"""Tokenizer shared with the in-browser scorer (docs/js/scorer.js must match it exactly).

Spec, version 1:
  1. Unicode NFKC normalisation.
  2. Replace URLs (`http://`, `https://` or `www.`, any case, up to the next whitespace) and
     emails (non-space runs containing `@`) with a space.
  3. Replace redactions, every run of two or more capital X's, with a space. The CFPB redacts
     with capital X's (`XXXX`, `XX/XX/XXXX`), often glued to digits or words (`XXXX1234`,
     `XXXXaccount`, `NUMBERXXXX`). Lowercase x's are left alone, so `Exxon` survives, but an
     all-caps `EXXON` becomes `E ON`.
  4. Lowercase.
  5. Tokens are runs of a-z letters, optionally joined by single apostrophes
     (`don't`, `company's`); everything else separates tokens. Stopwords and negations are
     kept; nothing is stemmed or lemmatised.
  6. Features are the unigrams plus adjacent-pair bigrams joined by one space.
"""

import re
import unicodedata
from itertools import pairwise

VERSION = 1

URL = re.compile(r"(?:https?://|www\.)\S*", re.IGNORECASE)
EMAIL = re.compile(r"\S*@\S*")
REDACTION = re.compile(r"X{2,}")
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")


def tokens(text: str) -> list[str]:
    """Tokens of `text` per the spec above."""
    text = unicodedata.normalize("NFKC", text)
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = REDACTION.sub(" ", text)
    return TOKEN.findall(text.lower())


def ngrams(toks: list[str]) -> list[str]:
    """Unigrams followed by bigrams."""
    return toks + [f"{a} {b}" for a, b in pairwise(toks)]


def analyze(text: str) -> list[str]:
    """The TF-IDF analyzer: unigrams and bigrams of `tokens(text)`."""
    return ngrams(tokens(text))


# Phrases matched as whole token sequences. No product or company names.
LEXICON: dict[str, list[str]] = {
    "urgency": [
        "urgent",
        "urgently",
        "immediately",
        "asap",
        "emergency",
        "right away",
        "as soon as possible",
        "cannot wait",
        "can't wait",
        "desperate",
        "deadline",
    ],
    "harm": [
        "fraud",
        "fraudulent",
        "identity theft",
        "stolen",
        "scam",
        "unauthorized",
        "not mine",
        "did not authorize",
        "never authorized",
        "harassment",
        "harassing",
        "threatened",
        "eviction",
        "foreclosure",
        "garnishment",
        "charged twice",
    ],
}


def _group_pattern(phrases: list[str]) -> re.Pattern:
    # Longest phrases first, so "did not authorize" wins over shorter overlaps.
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return re.compile(rf"(?<= )(?:{alts})(?= )")


_PATTERNS = {group: _group_pattern(ps) for group, ps in LEXICON.items()}


def lexicon_counts(toks: list[str]) -> dict[str, int]:
    """Non-overlapping occurrences of each group's phrases, matched on whole tokens."""
    joined = " " + " ".join(toks) + " "
    return {group: len(p.findall(joined)) for group, p in _PATTERNS.items()}
