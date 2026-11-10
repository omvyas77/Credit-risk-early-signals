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

import hashlib
import json
import re
import unicodedata
from itertools import pairwise

VERSION = 1

URL = re.compile(r"(?:https?://|www\.)\S*", re.IGNORECASE)
EMAIL = re.compile(r"\S*@\S*")
REDACTION = re.compile(r"X{2,}")
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")


def clean(text: str) -> str:
    """Steps 1 to 3 of the spec: the text before lowercasing and tokenising.

    The deep models read this, so they see the same narrative as TF-IDF minus URLs, emails
    and redactions, with case, punctuation and stopwords intact.
    """
    text = unicodedata.normalize("NFKC", text)
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    return REDACTION.sub(" ", text)


def tokens(text: str) -> list[str]:
    """Tokens of `text` per the spec above."""
    return TOKEN.findall(clean(text).lower())


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


def spec_hash() -> str:
    """Fingerprint of this tokenizer: the spec above, the patterns and the lexicon.

    The scorer bundle carries it and docs/js/scorer.js refuses a bundle whose hash differs
    from its own, so a change here can't ship without a matching change there.
    """
    spec = {
        "version": VERSION,
        "spec": __doc__,
        "patterns": [URL.pattern, EMAIL.pattern, REDACTION.pattern, TOKEN.pattern],
        "lexicon": LEXICON,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()[:16]
