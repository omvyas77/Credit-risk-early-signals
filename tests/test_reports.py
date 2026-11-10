"""Nothing in reports/ may carry narrative text. Narratives are long free text, usually with
XXXX redactions; report strings are names, numbers and short generated sentences."""

import json
from pathlib import Path

import pytest

REPORTS = Path("reports")
MAX_JSON_STRING = 200  # longest legitimate value today: a 100-character download URL
MAX_TEXT_LINE = 1000  # generated markdown, e.g. a list of caveats
ALLOWED = {".json", ".md"}


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def problems(root: Path) -> list[str]:
    """Every file under root that could hold narrative text, with the reason."""
    out = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == ".DS_Store":
            continue
        if path.suffix not in ALLOWED:
            out.append(f"{path}: only {sorted(ALLOWED)} files belong in reports/")
            continue
        content = path.read_text(encoding="utf-8")
        if "XXXX" in content:
            out.append(f"{path}: contains a redaction marker")
        if path.suffix == ".json":
            long = [s for s in _strings(json.loads(content)) if len(s) > MAX_JSON_STRING]
            out += [f"{path}: {len(s)}-character string {s[:40]!r}..." for s in long]
        else:
            long = [line for line in content.splitlines() if len(line) > MAX_TEXT_LINE]
            out += [f"{path}: {len(line)}-character line" for line in long]
    return out


def test_reports_hold_no_long_strings():
    assert REPORTS.is_dir()
    assert problems(REPORTS) == []


@pytest.mark.parametrize(
    "name, content",
    [
        ("a.json", json.dumps({"top": [{"term": "I called them " * 30}]})),
        ("b.json", json.dumps({"I called them " * 30: 1})),
        ("c.json", json.dumps({"term": "on XX/XX/XXXX"})),
        ("d.md", "| cell |\n" + "word " * 300 + "\n"),
        ("e.csv", "complaint_id,narrative\n"),
    ],
)
def test_scan_catches_text_that_looks_like_a_narrative(tmp_path, name, content):
    (tmp_path / name).write_text(content, encoding="utf-8")
    assert len(problems(tmp_path)) == 1


def test_scan_accepts_names_and_numbers(tmp_path):
    report = {"groups": [{"group": "company history", "share": 0.76}], "term": "cash app"}
    (tmp_path / "ok.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "ok.md").write_text("| W5 | 0.713 |\n", encoding="utf-8")
    assert problems(tmp_path) == []
