"""The late-response label and the data_end rule. The label is defined here and nowhere else."""

import calendar
import datetime as dt
import statistics

import duckdb


def month_end(month: dt.date) -> dt.date:
    return month.replace(day=calendar.monthrange(month.year, month.month)[1])


def choose_data_end(
    monthly_counts: dict[dt.date, int], min_ratio: float, trailing_months: int
) -> dt.date:
    """Last day of the latest month whose count is >= min_ratio x the median of the
    `trailing_months` months before it. Keys are first-of-month dates; missing months count 0.
    """
    months = sorted(monthly_counts)
    first, last = months[0], months[-1]
    series, m = [], first
    while m <= last:
        series.append((m, monthly_counts.get(m, 0)))
        m = (m + dt.timedelta(days=32)).replace(day=1)
    for i in range(len(series) - 1, trailing_months - 1, -1):
        month, count = series[i]
        median = statistics.median(c for _, c in series[i - trailing_months : i])
        if median > 0 and count >= min_ratio * median:
            return month_end(month)
    raise ValueError("no month passes the data_end rule")


# Outcome columns that must never reach a feature matrix. timely_response builds the label
# below and is then dropped with the rest.
OUTCOME_COLUMNS = ["timely_response", "company_response_to_consumer", "company_public_response"]


def apply_label(
    con: duckdb.DuckDBPyConnection,
    source: str,
    received_from: dt.date,
    data_end: dt.date,
    drop_company_response: list[str],
) -> dict[str, int]:
    """Create table `labelled` from `source` and return how many rows each rule dropped.

    Rules, applied in this order: no narrative, received before `received_from`, received
    after `data_end`, missing timely_response, company response in `drop_company_response`.
    late = timely_response == 'No'. Outcome columns are dropped from the result.
    """
    rules = [
        ("no_narrative", "narrative is not null and trim(narrative) <> ''"),
        ("received_before_start", f"date_received >= date '{received_from}'"),
        ("received_after_data_end", f"date_received <= date '{data_end}'"),
        ("timely_response_missing", "timely_response is not null"),
        (
            "company_response_dropped",
            "company_response_to_consumer is null or company_response_to_consumer not in ("
            + ", ".join("'" + r.replace("'", "''") + "'" for r in drop_company_response)
            + ")",
        ),
    ]
    con.execute(f"create or replace temp view label_step_0 as select * from {source}")
    drops = {"rows_in": con.execute("select count(*) from label_step_0").fetchone()[0]}
    for i, (name, keep) in enumerate(rules, 1):
        con.execute(
            f"create or replace temp view label_step_{i} as "
            f"select * from label_step_{i - 1} where {keep}"
        )
        before = con.execute(f"select count(*) from label_step_{i - 1}").fetchone()[0]
        after = con.execute(f"select count(*) from label_step_{i}").fetchone()[0]
        drops[name] = before - after
    bad = con.execute(
        f"select distinct timely_response from label_step_{len(rules)} "
        "where timely_response not in ('Yes', 'No')"
    ).fetchall()
    if bad:
        raise ValueError(f"unexpected timely_response values: {bad}")
    con.execute(f"""
        create or replace table labelled as
        select * exclude ({", ".join(OUTCOME_COLUMNS)}), timely_response = 'No' as late
        from label_step_{len(rules)}""")
    drops["rows_out"] = con.execute("select count(*) from labelled").fetchone()[0]
    return drops


NARRATIVES_REF = 3_851_415  # total narratives reported by the cfpb-complaints-analysis project


def main(cfg: dict) -> int:
    from cres.config import DEFAULT
    from cres.ingest import INTERIM, write_profile

    out = INTERIM / "labelled.parquet"
    inputs = [INTERIM / "complaints.parquet", INTERIM / "narratives.parquet", DEFAULT]
    if out.exists() and all(out.stat().st_mtime > p.stat().st_mtime for p in inputs):
        print("label: labelled.parquet is newer than its inputs, nothing to do")
        return 0

    con = duckdb.connect()
    narratives = f"read_parquet('{INTERIM / 'narratives.parquet'}')"
    complaints = f"read_parquet('{INTERIM / 'complaints.parquet'}')"

    monthly = dict(
        con.execute(
            f"select date_trunc('month', date_received)::date, count(*) from {narratives} "
            "group by 1 order by 1"
        ).fetchall()
    )
    rule = cfg["data_end"]
    data_end = choose_data_end(monthly, rule["min_ratio"], rule["trailing_months"])
    print(f"label: data_end = {data_end}", flush=True)

    lab = cfg["label"]
    received_from = dt.date.fromisoformat(lab["received_from"])
    drops = apply_label(con, complaints, received_from, data_end, lab["drop_company_response"])
    con.execute(f"copy labelled to '{out}' (format parquet)")

    settled = ", ".join("'" + r + "'" for r in lab["drop_company_response"])
    timeline = con.execute(f"""
        select strftime(date_trunc('month', date_received), '%Y-%m') as month,
          count(*) as narratives,
          count(*) filter (where timely_response is not null
                           and company_response_to_consumer not in ({settled})) as settled,
          avg(case when timely_response is null
                     or company_response_to_consumer in ({settled}) then null
                   else (timely_response = 'No')::int end) as late_rate
        from {complaints}
        where date_received >= date '2015-01-01'
        group by 1 order by 1""").fetchall()
    late_rate = con.execute("select avg(late::int), sum(late::int) from labelled").fetchone()
    total = sum(monthly.values())
    write_profile(
        "label",
        {
            "data_end": data_end.isoformat(),
            "data_end_rule": rule,
            "received_from": lab["received_from"],
            "rows_dropped": drops,
            "late_positives": late_rate[1],
            "late_rate": late_rate[0],
            "narratives_total": total,
            "narratives_reference": NARRATIVES_REF,
            "narratives_vs_reference": total - NARRATIVES_REF,
            "narratives_per_month": {m.strftime("%Y-%m"): n for m, n in monthly.items()},
            "timeline": [
                {"month": m, "narratives": n, "settled": s, "late_rate": r}
                for m, n, s, r in timeline
            ],
        },
    )
    for key, value in drops.items():
        print(f"  {key}: {value:,}")
    print(f"  late: {late_rate[1]:,} ({late_rate[0]:.4f})")
    return 0
