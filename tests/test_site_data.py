"""docs/data must match the reports it is built from. The files that come only from tracked
inputs are rebuilt here and compared byte for byte; the two that also need data/ and models/
(companies.json, scorer.json) are checked against the parts that are tracked."""

import json
from pathlib import Path

import pytest

from cres import export

DATA = Path("docs/data")


def payload(obj) -> str:
    return json.dumps(export.tidy(obj), separators=(",", ":"), ensure_ascii=False) + "\n"


@pytest.mark.parametrize("name", sorted(export.from_tracked()))
def test_site_file_matches_its_reports(name):
    built = export.from_tracked()[name]
    assert (DATA / name).read_text(encoding="utf-8") == payload(built), (
        f"docs/data/{name} is out of date with reports/ or archive/: run `make export`"
    )


def test_scorer_file_matches_metrics_and_bundle():
    site = json.loads((DATA / "scorer.json").read_text(encoding="utf-8"))
    report = export.read(export.REPORTS / "metrics.json")
    assert site["metrics"] == export.tidy(export.scorer_metrics(report, site["window"]))
    assert site["bundle_bytes"] == (export.SCORER_DIR / "model.json").stat().st_size


def test_companies_file_is_for_the_latest_window():
    site = json.loads((DATA / "companies.json").read_text(encoding="utf-8"))
    windows = export.read(export.REPORTS / "windows.json")["windows"]
    assert site["window"] == windows[-1]["window"]


def test_every_site_file_is_written_by_export():
    known = set(export.from_tracked()) | {"companies.json", "scorer.json"}
    assert {p.name for p in DATA.glob("*.json")} == known


def test_manifest_lists_archive_files_by_number_then_the_live_file():
    report = {
        "complaints.csv.zip": {"url": "u", "bytes": 3, "sha256": "c", "downloaded": "t"},
        "CCDB_Export_10_May.zip": {"url": "u", "bytes": 2, "sha256": "b", "downloaded": "t"},
        "CCDB_Export_2_May.zip": {"url": "u", "bytes": 1, "sha256": "a", "downloaded": "t"},
    }
    names = [f["name"] for f in export.manifest(report)["files"]]
    assert names == ["CCDB_Export_2_May.zip", "CCDB_Export_10_May.zip", "complaints.csv.zip"]
