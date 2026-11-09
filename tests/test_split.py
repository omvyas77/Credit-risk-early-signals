import datetime as dt
from itertools import pairwise

import duckdb
import pytest

from cres.split import add_months, tag_rows, windows

D = dt.date


def test_matches_the_worked_example_for_march_2026():
    w = windows(D(2026, 3, 31), 5, 3, 3, 30, 1)
    assert w[0]["test"] == (D(2025, 1, 1), D(2025, 3, 31))
    assert w[0]["val"] == (D(2024, 9, 2), D(2024, 12, 1))
    assert w[0]["train"][1] == D(2024, 8, 2)
    assert w[2]["val"] == (D(2025, 3, 1), D(2025, 5, 31))
    assert w[2]["train"][1] == D(2025, 1, 29)
    assert w[4]["test"] == (D(2026, 1, 1), D(2026, 3, 31))


@pytest.mark.parametrize("data_end", [D(2026, 6, 30), D(2026, 3, 31), D(2025, 12, 31)])
@pytest.mark.parametrize("gap", [0, 30, 45])
def test_gaps_hold_between_splits(data_end, gap):
    for w in windows(data_end, 5, 3, 3, gap, 1):
        assert w["train"][1] + dt.timedelta(days=gap) < w["val"][0]
        assert w["val"][1] + dt.timedelta(days=gap) < w["test"][0]
        for s in ("train", "val", "test"):
            assert w[s][0] <= w[s][1]
    assert windows(data_end, 5, 3, 3, gap, 1)[-1]["test"][1] == data_end


def test_test_windows_are_consecutive_quarters():
    w = windows(D(2026, 6, 30), 5, 3, 3, 30, 1)
    for a, b in pairwise(w):
        assert a["test"][1] + dt.timedelta(days=1) == b["test"][0]


def test_train_years_sets_the_start():
    (w,) = windows(D(2026, 6, 30), 1, 3, 3, 30, 2)
    assert w["train"][0] == add_months(w["train"][1] + dt.timedelta(days=1), -24)


def test_add_months_clamps_day():
    assert add_months(D(2025, 5, 31), -3) == D(2025, 2, 28)
    assert add_months(D(2024, 1, 31), 1) == D(2024, 2, 29)


def test_tag_rows_puts_each_date_in_the_right_split():
    (w,) = windows(D(2026, 6, 30), 1, 3, 3, 30, 1)
    con = duckdb.connect()
    con.execute("create table r (complaint_id bigint, date_received date)")
    dates = {
        1: w["train"][0],
        2: w["train"][1],
        3: w["train"][1] + dt.timedelta(days=1),  # in the gap: untagged
        4: w["val"][0],
        5: w["val"][1] + dt.timedelta(days=5),  # gap
        6: w["test"][1],
        7: w["test"][1] + dt.timedelta(days=1),  # after data_end
        8: w["train"][0] - dt.timedelta(days=1),  # before the train cap
    }
    con.executemany("insert into r values (?, ?)", list(dates.items()))
    tag_rows(con, "r", [w])
    got = dict(con.execute("select complaint_id, split from splits").fetchall())
    assert got == {1: "train", 2: "train", 4: "val", 6: "test"}
