import datetime as dt

import duckdb
import pytest

from cres.label import OUTCOME_COLUMNS, apply_label

START, END = dt.date(2017, 5, 1), dt.date(2026, 3, 31)

# complaint_id, date_received, narrative, timely_response, company_response_to_consumer
ROWS = [
    (1, "2020-01-10", "late one", "No", "Closed with explanation"),
    (2, "2020-01-10", "on time", "Yes", "Closed with explanation"),
    (3, "2017-04-30", "too early", "No", "Closed with explanation"),
    (4, "2017-05-01", "first day", "Yes", "Closed with explanation"),
    (5, "2026-04-01", "after end", "No", "Closed with explanation"),
    (6, "2026-03-31", "last day", "No", "Untimely response"),
    (7, "2020-01-10", None, "No", "Closed with explanation"),
    (8, "2020-01-10", "   ", "No", "Closed with explanation"),
    (9, "2020-01-10", "missing outcome", None, "Closed with explanation"),
    (10, "2020-01-10", "pending", "Yes", "In progress"),
    (11, "2020-01-10", "no company response", "Yes", None),
]


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("""create table src (complaint_id bigint, date_received date, narrative varchar,
        timely_response varchar, company_response_to_consumer varchar,
        company_public_response varchar, product varchar)""")
    con.executemany(
        "insert into src values (?, ?, ?, ?, ?, null, 'Mortgage')", [list(r) for r in ROWS]
    )
    return con


def labels(con):
    return dict(con.execute("select complaint_id, late from labelled").fetchall())


def test_every_branch(con):
    drops = apply_label(con, "src", START, END, ["In progress"])
    assert labels(con) == {1: True, 2: False, 4: False, 6: True, 11: False}
    assert drops == {
        "rows_in": 11,
        "no_narrative": 2,
        "received_before_start": 1,
        "received_after_data_end": 1,
        "timely_response_missing": 1,
        "company_response_dropped": 1,
        "rows_out": 5,
    }


def test_outcome_columns_are_dropped(con):
    apply_label(con, "src", START, END, ["In progress"])
    cols = {r[0] for r in con.execute("describe labelled").fetchall()}
    assert not cols & set(OUTCOME_COLUMNS)
    assert {"complaint_id", "narrative", "product", "late"} <= cols


def test_unexpected_timely_value_fails(con):
    con.execute("insert into src values (12, '2020-01-10', 'odd', 'Maybe', 'x', null, 'x')")
    with pytest.raises(ValueError):
        apply_label(con, "src", START, END, ["In progress"])
