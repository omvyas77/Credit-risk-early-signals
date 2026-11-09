"""Rolling-origin windows, generated from data_end; never written out by hand.

For each window (oldest first), with all ranges inclusive:
  test  = `test_months` whole months; the last window ends on data_end
  val   = `val_months` months ending gap_days + 1 days before test starts
  train = `train_years` years ending gap_days + 1 days before val starts
So the last train day + gap_days < the first val day, and likewise val -> test.
"""

import calendar
import datetime as dt

import duckdb

SPLITS = ("train", "val", "test")


def add_months(day: dt.date, months: int) -> dt.date:
    """Shift by whole months, clamping the day to the target month's length."""
    total = day.year * 12 + day.month - 1 + months
    year, month = divmod(total, 12)
    month += 1
    return dt.date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def windows(
    data_end: dt.date,
    n_windows: int,
    test_months: int,
    val_months: int,
    gap_days: int,
    train_years: int,
) -> list[dict]:
    """Window boundaries, oldest first, as dicts of inclusive (start, end) dates per split."""
    gap = dt.timedelta(days=gap_days)
    one = dt.timedelta(days=1)
    out = []
    last_month = data_end.replace(day=1)
    for k in range(n_windows):
        offset = (n_windows - 1 - k) * test_months
        test_end_month = add_months(last_month, -offset)
        test_end = test_end_month.replace(
            day=calendar.monthrange(test_end_month.year, test_end_month.month)[1]
        )
        test_start = add_months(test_end_month, -(test_months - 1))
        val_end = test_start - gap - one
        val_start = add_months(val_end, -val_months) + one
        train_end = val_start - gap - one
        train_start = add_months(train_end + one, -12 * train_years)
        out.append(
            {
                "window": f"W{k + 1}",
                "train": (train_start, train_end),
                "val": (val_start, val_end),
                "test": (test_start, test_end),
            }
        )
    return out


def from_config(cfg: dict, data_end: dt.date) -> list[dict]:
    w = cfg["windows"]
    return windows(
        data_end,
        w["n_windows"],
        w["test_months"],
        w["val_months"],
        w["gap_days"],
        w["train_years"],
    )


def tag_rows(con: duckdb.DuckDBPyConnection, source: str, wins: list[dict]) -> None:
    """Create table `splits` (complaint_id, window, split) for rows of `source` in any range."""
    values = ", ".join(
        f"('{w['window']}', '{s}', date '{w[s][0]}', date '{w[s][1]}')"
        for w in wins
        for s in SPLITS
    )
    con.execute(f"""
        create or replace table splits as
        select r.complaint_id, b."window", b.split
        from {source} r
        join (values {values}) b("window", split, lo, hi)
          on r.date_received between b.lo and b.hi""")
