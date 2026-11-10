"""Synthetic texts for the scorer parity test, written by hand: no real narratives.

Run `uv run python tests/parity_fixtures.py` to regenerate tests/fixtures/scorer_texts.json
after changing the tokenizer; test_scorer_parity.py checks the file matches this generator.
"""

import json
import random
from pathlib import Path

from cres import config

OUT = Path(__file__).parent / "fixtures" / "scorer_texts.json"
N = 1000

# Everyday words of a complaint about a financial product, written for this file.
WORDS = """
i my the a an and or but not no never to of in on for with from by at about after before
account accounts bank card credit debit loan loans mortgage payment payments paid pay late
fee fees interest charge charged charges balance statement statements report reported
reporting bureau dispute disputed disputes collection collections collector debt owed owe
company companies servicer lender customer service representative rep called call calls
phone email letter letters mail sent send received receive told said asked request
requested refund refunded month months year years day days week weeks time again still
closed close open opened transfer transferred deposit check checking savings fraud
fraudulent unauthorized stolen identity theft scam harassment harassing threatened urgent
immediately emergency deadline desperate foreclosure eviction garnishment wage application
denied approved approval score inquiry inquiries information incorrect error errors wrong
inaccurate correct fix fixed removed remove delete deleted student vehicle car auto house
home insurance escrow modification forbearance hardship help please complaint resolve
resolved response respond responded nothing anything everything they them their we our
you your he she it is was were be been being have has had do does did can could would should
""".split()

# Shapes the tokenizer has to get right, one per entry.
EDGES = [
    # apostrophes
    "don't",
    "company's",
    "'quoted'",
    "rock 'n' roll",
    "o''brien",
    "it's'",
    "students'",
    "''",
    "y'all'd've",
    "can't wait",
    "' leading",
    # redactions and capital X's
    "XXXX",
    "XX/XX/XXXX",
    "XXXX1234",
    "XXXXaccount",
    "NUMBERXXXX",
    "EXXON",
    "Exxon",
    "xx",
    "X",
    "XxX",
    "XXX-XX-XXXX",
    "{$XXXX.00}",
    "XXXXXXXX",
    "XXXX XXXX XXXX",
    "aXXb",
    "TAXXES",
    # URLs
    "https://example.com/a?b=c",
    "HTTP://EXAMPLE.ORG/PATH",
    "www.example.net/x",
    "Www.Example.com",
    "see:http://x.y/z",
    "(https://a.b)",
    "hTtPs://mixed.case/page",
    "ftp://not.matched/file",
    "wwwexample",
    "http:/broken",
    "http://",
    "https://example.com/path\twith tab",
    # emails
    "name@example.com",
    "a@b",
    "@handle",
    "user@@x",
    "first.last+tag@sub.example.co.uk",
    "mail me at foo@bar.",
    "@",
    "late@",
    # unicode that NFKC changes (or leaves alone)
    "ﬁnancial",  # fi ligature
    "ﬀ",  # ff ligature
    "ｆｕｌｌｗｉｄｔｈ",  # fullwidth "fullwidth"
    "①",  # circled one
    "x²",  # superscript two
    "Ⅻ",  # roman numeral twelve
    "café",
    "naïve",
    "é",  # e + combining acute
    "İstanbul",  # dotted capital I
    "Straße",
    "™",  # trade mark
    "a b",  # no-break space
    "a b",  # em space
    "a　b",  # ideographic space
    "don’t",  # right single quote: not an apostrophe
    "can＇t",  # fullwidth apostrophe: becomes one
    "a﻿b",  # zero-width no-break space: not whitespace in Python
    "line break",
    "next\x85line",
    "sep\x1cgroup",
    "ＸＸＸＸ",  # fullwidth XXXX: a redaction after NFKC
    "ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ",
    "Ｘ",
    "ΣΑΣ",  # Greek capitals
    "ǅ",  # Dz digraph
    "ℌello",  # script H
    "㎏",  # square kg
    "½",
    "a b",  # ogham space mark
    "\U0001f600",  # emoji
    "\U0001d41b\U0001d428\U0001d425\U0001d41d",  # mathematical bold "bold"
    "Ⓐ",  # circled A
    "ﬅ",  # long s t ligature
    "ſecure",  # long s
    "K",  # Kelvin sign
    # names on Object.prototype
    "constructor",
    "toString",
    "__proto__",
    "hasOwnProperty",
    "valueOf",
    "constructor prototype",
    # lexicon phrases
    "urgent",
    "as soon as possible",
    "did not authorize",
    "not mine",
    "identity theft",
    "charged twice",
    "right away",
    "never authorized",
    "fraud fraudulent",
    "did not authorize did not authorize",
    "as soon as",
    # punctuation and digits
    "!!! ??? ...",
    "12345 67.89",
    "$1,000.00",
    "-- // ** ##",
    "a-b_c.d",
]

SPECIAL = [
    "",
    " ",
    "\n\n\t",
    "XXXX",
    "late " * 40,
    "fraud " * 25,
    "i i i i i i i i",
    "the payment the payment the payment",
    "".join(EDGES),
    " ".join(EDGES),
]


def generate(seed: int) -> list[str]:
    rng = random.Random(seed)
    texts = SPECIAL + EDGES
    while len(texts) < N:
        words = []
        for _ in range(rng.randint(0, 120)):
            r = rng.random()
            if r < 0.12:
                words.append(rng.choice(EDGES))
            elif r < 0.17 and words:
                words.append(words[-1])  # repeats
            else:
                w = rng.choice(WORDS)
                style = rng.random()
                if style < 0.08:
                    w = w.upper()
                elif style < 0.18:
                    w = w.capitalize()
                words.append(w)
            if rng.random() < 0.05:
                words[-1] = rng.choice(["XXXX", "XX", "XXXXXX"]) + words[-1]  # glued redaction
        text = ""
        for w in words:
            text += w + rng.choice([" ", " ", " ", ", ", ". ", "\n", "  ", "/", "-", "("])
        texts.append(text)
    return texts[:N]


def main() -> None:
    texts = generate(config.load()["seed"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(texts, ensure_ascii=True, indent=0) + "\n", encoding="utf-8")
    print(f"wrote {len(texts)} texts to {OUT}")


if __name__ == "__main__":
    main()
