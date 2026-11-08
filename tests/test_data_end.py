import datetime as dt

import pytest

from cres.label import choose_data_end


def months(counts, start=dt.date(2025, 1, 1)):
    out, m = {}, start
    for c in counts:
        out[m] = c
        m = (m + dt.timedelta(days=32)).replace(day=1)
    return out


def test_picks_last_month_before_the_tail_thins():
    # Jan..Jun steady at 100, Jul 90 passes, Aug 50 and Sep 1 fail.
    counts = months([100, 100, 100, 100, 100, 100, 90, 50, 1])
    assert choose_data_end(counts, 0.7, 6) == dt.date(2025, 7, 31)


def test_exactly_at_ratio_passes():
    counts = months([100] * 6 + [70])
    assert choose_data_end(counts, 0.7, 6) == dt.date(2025, 7, 31)


def test_missing_month_counts_as_zero():
    counts = months([100] * 7)
    counts[dt.date(2025, 9, 1)] = 100  # August missing -> 0, September still passes
    assert choose_data_end(counts, 0.7, 6) == dt.date(2025, 9, 30)


def test_no_month_passes():
    with pytest.raises(ValueError):
        choose_data_end(months([100] * 6 + [10]), 0.7, 6)
