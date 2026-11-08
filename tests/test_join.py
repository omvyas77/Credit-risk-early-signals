import duckdb
import pytest

from cres.ingest import compare_outcomes, dedupe_narratives, join_live


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("""create table archive (complaint_id bigint, date_received date,
        narrative varchar, timely_response varchar, company_response_to_consumer varchar,
        export_n integer)""")
    con.executemany(
        "insert into archive values (?, '2024-01-05', ?, ?, 'Closed with explanation', ?)",
        [
            [1, "old text", "Yes", 3],
            [1, "new text", "Yes", 4],  # duplicate: the later export wins
            [2, "only in archive", "Yes", 4],  # no live row
            [3, None, "Yes", 4],  # no narrative
            [4, "outcome changed", "Yes", 4],
        ],
    )
    con.execute("""create table live (complaint_id bigint, product varchar,
        timely_response varchar, company_response_to_consumer varchar)""")
    con.executemany(
        "insert into live values (?, 'Mortgage', ?, 'Closed with explanation')",
        [[1, "Yes"], [3, "Yes"], [4, "No"], [5, "Yes"]],
    )
    return con


def test_dedupe_keeps_latest_export(con):
    counts = dedupe_narratives(con, "archive")
    assert counts == {
        "archive_rows": 5,
        "archive_rows_with_narrative": 4,
        "duplicate_narrative_rows_dropped": 1,
        "narratives_unique": 3,
    }
    text = con.execute("select narrative from narratives where complaint_id = 1").fetchone()[0]
    assert text == "new text"


def test_join_counts_archive_only_and_takes_outcomes_from_live(con):
    dedupe_narratives(con, "archive")
    counts = join_live(con, "live")
    assert counts == {"live_rows": 4, "matched": 2, "archive_only": 1}
    rows = dict(con.execute("select complaint_id, timely_response from complaints").fetchall())
    assert rows == {1: "Yes", 4: "No"}  # 4 is 'Yes' in the archive, 'No' in live


def test_compare_outcomes(con):
    dedupe_narratives(con, "archive")
    assert compare_outcomes(con, "live") == {
        "matched": 2,
        "timely_response_differs": 1,
        "company_response_differs": 0,
    }
