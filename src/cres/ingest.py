"""Unzip the raw files and load them with DuckDB; join archive narratives to live records.

Outputs (under data/interim/):
  live.parquet        every live-file row, snake_case columns, typed dates
  narratives.parquet  archive rows with a narrative, deduplicated by complaint_id
  complaints.parquet  narrative from the archive, every other field from live
"""

import json
import re
import zipfile
from pathlib import Path

import duckdb

from cres.config import data_dir, raw_dir

INTERIM = Path("data/interim")
PROFILE = Path("reports/data_profile.json")
MANIFEST = Path("reports/data_manifest.json")

# CSV header -> column name. Same header in all 21 archive files; live lacks the narrative.
COLUMNS = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response_to_consumer",
    "Timely response?": "timely_response",
    "Complaint ID": "complaint_id",
}
DATES = {"date_received", "date_sent_to_company"}
LIVE_ZIP = "complaints.csv.zip"
EXPORT_N = re.compile(r"CCDB_Export_(\d+)_")


def select_list(headers: list[str]) -> str:
    """SQL that renames CSV headers to snake_case and casts ids and dates."""
    out = []
    for header in headers:
        name = COLUMNS[header]
        src = '"' + header + '"'
        if name == "complaint_id":
            out.append(f"cast({src} as bigint) as complaint_id")
        elif name in DATES:
            out.append(f"cast({src} as date) as {name}")
        else:
            out.append(f"nullif(trim({src}), '') as {name}")
    return ",\n  ".join(out)


def extract(zip_path: Path, out_dir: Path) -> Path:
    """Unzip the single CSV in `zip_path`, skipping it if already extracted at full size."""
    with zipfile.ZipFile(zip_path) as z:
        (info,) = z.infolist()
        dest = out_dir / info.filename
        if dest.exists() and dest.stat().st_size == info.file_size:
            return dest
        print(f"  unzipping {zip_path.name} ({info.file_size / 1e9:.2f} GB)", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        z.extract(info, out_dir)
    return dest


def csv_headers(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(
        "describe select * from read_csv(?, all_varchar = true, header = true)", [str(path)]
    ).fetchall()
    return [r[0] for r in rows]


def read_csv_sql(paths: list[Path], headers: list[str]) -> str:
    files = ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths)
    return (
        f"select {select_list(headers)}, regexp_extract(filename, '{EXPORT_N.pattern}', 1) "
        f"as export_file from read_csv([{files}], all_varchar = true, header = true, "
        f"filename = true)"
    )


def dedupe_narratives(con: duckdb.DuckDBPyConnection, archive: str) -> dict:
    """Create table `narratives` from `archive`: rows with a narrative, one per complaint_id,
    keeping the latest export. Returns counts."""
    con.execute(f"""
        create or replace table narratives as
        select * exclude (rn) from (
          select *, row_number() over (
            partition by complaint_id order by export_n desc) as rn
          from {archive} where narrative is not null)
        where rn = 1""")
    archive_rows, with_narrative = con.execute(
        f"select count(*), count(narrative) from {archive}"
    ).fetchone()
    unique = con.execute("select count(*) from narratives").fetchone()[0]
    return {
        "archive_rows": archive_rows,
        "archive_rows_with_narrative": with_narrative,
        "duplicate_narrative_rows_dropped": with_narrative - unique,
        "narratives_unique": unique,
    }


def join_live(con: duckdb.DuckDBPyConnection, live: str) -> dict:
    """Create table `complaints`: narrative from `narratives`, every other field from `live`."""
    con.execute(f"""
        create or replace table complaints as
        select l.*, n.narrative
        from narratives n join {live} l using (complaint_id)""")
    live_rows = con.execute(f"select count(*) from {live}").fetchone()[0]
    matched = con.execute("select count(*) from complaints").fetchone()[0]
    unique = con.execute("select count(*) from narratives").fetchone()[0]
    return {"live_rows": live_rows, "matched": matched, "archive_only": unique - matched}


def compare_outcomes(con: duckdb.DuckDBPyConnection, live: str) -> dict:
    """How often the archive's outcome fields differ from the live file's, on matched rows."""
    row = con.execute(f"""
        select count(*),
          count(*) filter (where n.timely_response is distinct from l.timely_response),
          count(*) filter (where n.company_response_to_consumer
                           is distinct from l.company_response_to_consumer)
        from narratives n join {live} l using (complaint_id)""").fetchone()
    return {
        "matched": row[0],
        "timely_response_differs": row[1],
        "company_response_differs": row[2],
    }


def stale(output: Path) -> bool:
    """True if `output` is missing or older than the download manifest."""
    return not output.exists() or output.stat().st_mtime < MANIFEST.stat().st_mtime


def write_profile(section: str, data: dict) -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8")) if PROFILE.exists() else {}
    profile[section] = data
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILE.write_text(json.dumps(profile, indent=2, default=str) + "\n", encoding="utf-8")


def main(cfg: dict) -> int:
    raw, extracted = raw_dir(cfg), data_dir(cfg) / "extracted"
    INTERIM.mkdir(parents=True, exist_ok=True)
    outputs = [INTERIM / f for f in ("live.parquet", "narratives.parquet", "complaints.parquet")]
    if not any(stale(p) for p in outputs):
        print("ingest: outputs are newer than the manifest, nothing to do")
        return 0

    archive_zips = sorted(
        raw.glob("CCDB_Export_*.zip"), key=lambda p: int(EXPORT_N.search(p.name)[1])
    )
    print(f"ingest: extracting {len(archive_zips)} archive files and the live file", flush=True)
    archive_csvs = [extract(z, extracted) for z in archive_zips]
    live_csv = extract(raw / LIVE_ZIP, extracted)

    con = duckdb.connect(str(INTERIM / "ingest.duckdb"))
    con.execute(f"set temp_directory = '{INTERIM / 'tmp'}'")
    con.execute("set preserve_insertion_order = false")

    print("ingest: live file -> live.parquet", flush=True)
    live_sql = read_csv_sql([live_csv], csv_headers(con, live_csv))
    con.execute(
        f"copy (select * exclude (export_file) from ({live_sql})) to '{outputs[0]}' "
        "(format parquet)"
    )
    con.execute(f"create or replace view live as select * from read_parquet('{outputs[0]}')")

    print("ingest: loading archive files", flush=True)
    headers = csv_headers(con, archive_csvs[0])
    for path in archive_csvs[1:]:
        if csv_headers(con, path) != headers:
            raise ValueError(f"{path.name}: header differs from {archive_csvs[0].name}")
    con.execute(f"""
        create or replace table archive as
        select complaint_id, date_received, narrative, timely_response,
          company_response_to_consumer, cast(export_file as integer) as export_n
        from ({read_csv_sql(archive_csvs, headers)})""")

    print("ingest: deduplicating and joining", flush=True)
    counts = dedupe_narratives(con, "archive")
    con.execute(
        "copy (select complaint_id, date_received, narrative, export_n from narratives) "
        f"to '{outputs[1]}' (format parquet)"
    )
    counts |= join_live(con, "live")
    counts["outcomes_archive_vs_live"] = compare_outcomes(con, "live")
    con.execute(f"copy complaints to '{outputs[2]}' (format parquet)")
    con.close()
    (INTERIM / "ingest.duckdb").unlink()

    write_profile("ingest", counts)
    for key, value in counts.items():
        print(f"  {key}: {value}")
    return 0
