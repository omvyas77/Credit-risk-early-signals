"""Fill the generated blocks of README.md and MODEL_CARD.md from reports/ and archive/v1.json.

Each block sits between `<!-- generated:NAME -->` and `<!-- /generated:NAME -->`. Only the
lines between the markers are replaced, so the prose around them is written by hand and every
number inside them comes from a pipeline output. A test renders the blocks from the tracked
reports and fails if either file differs.
"""

import re
from pathlib import Path

import numpy as np

from cres import evaluate, export
from cres.evaluate import MODEL_LABELS, _ci, _diff, _find

FILES = (Path("README.md"), Path("MODEL_CARD.md"))
BLOCK = re.compile(r"(<!-- generated:([a-z0-9_]+) -->\n)(.*?)(<!-- /generated:\2 -->)", re.DOTALL)
OPENING = re.compile(r"<!-- generated:(\S+) -->")
FEATURESET_LABELS = {
    "text": "narrative only",
    "struct": "intake fields + company history",
    "both": "narrative + intake fields + company history",
}
RULE_LABELS = {
    "f2": "F2-optimal",
    "cost": "cost-optimal",
    "capacity_0.01": "top 1% of validation scores",
    "capacity_0.05": "top 5% of validation scores",
}


def load(reports: Path = export.REPORTS, archive: Path = export.ARCHIVE) -> dict:
    """Every input the blocks read."""
    return {
        "metrics": export.read(reports / "metrics.json"),
        "windows": export.read(reports / "windows.json"),
        "profile": export.read(reports / "data_profile.json"),
        "manifest": export.read(reports / "data_manifest.json"),
        "explain": export.read(reports / "explain.json"),
        "v1": export.read(archive / "v1.json"),
    }


def _mean(r: dict, cell: str, k: str = "pr_auc") -> float:
    return r["metrics"]["across_windows"][cell][k]["mean"]


def _cell(r: dict, model: str, fs: str, window: str) -> dict | None:
    return r["metrics"]["cells"].get(f"{model}__{fs}__{window}")


def _windows(r: dict) -> list[str]:
    return [w["window"] for w in r["windows"]["windows"]]


def _low_share(r: dict) -> list[str]:
    wins = r["metrics"]["windows"]
    return [
        w for w in _windows(r) if wins[w]["narrative_share_test"] < evaluate.LOW_NARRATIVE_SHARE
    ]


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def answer(r: dict) -> str:
    windows = _windows(r)
    diffs = [_find(r["metrics"], w, ["lr", "both"], ["lr", "struct"])["pr_auc"] for w in windows]
    below = sum(d["ci95"][1] < 0 for d in diffs)
    chance = np.mean([r["metrics"]["windows"][w]["test_prevalence"] for w in windows])
    return (
        f"Across {len(windows)} rolling quarterly test windows, a logistic regression on the "
        f"intake fields and the company's history reaches a mean test PR-AUC of "
        f"{_mean(r, 'lr__struct'):.3f}. Adding the narrative gives {_mean(r, 'lr__both'):.3f} "
        f"(its paired 95% CI is below zero in {below} of {len(windows)} windows), and the "
        f"narrative alone gives {_mean(r, 'lr__text'):.3f}, against {chance:.3f} by chance. "
        f"Most of the signal is the company: its own trailing late rate, with no model, reaches "
        f"{_mean(r, 'company_rate__struct'):.3f}."
    )


def results(r: dict) -> str:
    latest = _windows(r)[-1]
    wins = r["metrics"]["windows"]
    chance = np.mean([wins[w]["test_prevalence"] for w in _windows(r)])
    rows = [
        f"| Model | Features | Mean test PR-AUC, {len(_windows(r))} windows "
        f"| {latest} test PR-AUC (95% CI) |",
        "|---|---|---|---|",
        f"| Chance (share of late responses) | none | {chance:.3f} "
        f"| {wins[latest]['test_prevalence']:.3f} |",
    ]
    cells = [("company_rate", "struct")] + [
        (m, fs) for m in ("lr", "lgbm") for fs in ("text", "struct", "both")
    ]
    for m, fs in cells:
        feats = "company history" if m == "company_rate" else FEATURESET_LABELS[fs]
        rows.append(
            f"| {MODEL_LABELS[m]} | {feats} | {_mean(r, f'{m}__{fs}'):.3f} "
            f"| {_ci(_cell(r, m, fs, latest), 'pr_auc')} |"
        )
    scorer = _cell(r, "scorer", "both", latest)
    if scorer:
        rows.append(
            f"| {MODEL_LABELS['scorer']} | {FEATURESET_LABELS['both']} | {latest} only "
            f"| {_ci(scorer, 'pr_auc')} |"
        )
    return "\n".join(rows)


def data(r: dict) -> str:
    man, ing, lab = r["manifest"], r["profile"]["ingest"], r["profile"]["label"]
    live = man["complaints.csv.zip"]
    archive = [n for n in man if n.startswith("CCDB_Export_")]
    return "\n".join(
        [
            f"- Downloaded {live['downloaded'][:10]}: {len(archive)} archive ZIPs and the live "
            f"file ({ing['live_rows']:,} complaints). Their sha256 checksums and download times "
            "are in `reports/data_manifest.json`.",
            f"- {ing['matched']:,} narratives join to the live file; "
            f"{lab['rows_dropped']['rows_out']:,} of them were received {lab['received_from']} "
            f"to {lab['data_end']} with a settled outcome and are labelled, and "
            f"{lab['late_positives']:,} ({lab['late_rate']:.2%}) of those were answered late.",
        ]
    )


def shift(r: dict) -> str:
    wins, low = r["metrics"]["windows"], _low_share(r)
    if not low:
        return "No test window falls in a period with a low narrative share."
    first = _windows(r)[0]
    start, end = wins[low[0]]["dates"]["test"][0], wins[low[-1]]["dates"]["test"][1]
    shares = _and([f"{wins[w]['narrative_share_test']:.1%}" for w in low])
    late = _and([f"{wins[w]['test_prevalence']:.1%}" for w in low])
    return (
        f"{_and(low)} test on {start} to {end}, when {shares} of complaints carry a published "
        f"narrative and {late} of those are late; {first}'s test quarter had "
        f"{wins[first]['narrative_share_test']:.1%} and {wins[first]['test_prevalence']:.1%}. "
        "Those windows are drawn from a different, much smaller selection of complaints, so "
        "compare each window's PR-AUC with its own chance level."
    )


def scorer_gap(r: dict) -> str:
    latest = _windows(r)[-1]
    s, lr = _cell(r, "scorer", "both", latest), _cell(r, "lr", "both", latest)
    d = _find(r["metrics"], latest, ["scorer", "both"], ["lr", "both"])["pr_auc"]
    terms = r["metrics"]["training"][f"scorer__both__{latest}"]["terms"]
    return (
        f"The in-browser scorer leaves out VADER sentiment (it has no JavaScript port) and keeps "
        f"{terms:,} terms. On {latest} test its PR-AUC is {s['point']['pr_auc']:.3f} against "
        f"{lr['point']['pr_auc']:.3f} for the full LR; the paired difference is {_diff(d)}."
    )


def deep(r: dict) -> str:
    missing = r["metrics"]["not_run"]
    if not missing:
        return "Every deep-model cell the config expects has a prediction file."
    return (
        f"Not run yet: {', '.join(f'`{c}`' for c in missing)}. They need a GPU; "
        "`notebooks/README.md` has the Colab steps, and `make evaluate` adds them to every "
        "table once their prediction files are in `data/preds/`."
    )


def compute(r: dict) -> str:
    runs = r["metrics"]["training"].values()
    peak = max(m.get("peak_rss_gb") or 0 for m in runs)
    hours = sum(m.get("total_seconds") or m.get("fit_seconds") or 0 for m in runs) / 3600
    return (
        f"The largest single fit peaked at {peak:.1f} GB of memory, and the recorded model "
        f"fits took {hours:.1f} hours in all (from `reports/metrics.json`, `training`)."
    )


def v1(r: dict) -> str:
    v = r["v1"]
    best = max(v["models"], key=lambda m: m["pr_auc"])
    rows = [
        f"v1 reported a test PR-AUC of {best['pr_auc']:.3f} ({best['model']}) on a "
        f'"{v["label"]}" label, with {v["test"]["positives"]:,} test positives and its '
        f"threshold chosen on {v['threshold']['chosen_on']}. Its data export is gone, so those "
        "numbers can't be reproduced or compared with this version's.",
        "",
        "| v1 | This version |",
        "|---|---|",
    ]
    rows += [f"| {f['v1']} | {f['v2']} |" for f in v["fixes"]]
    return "\n".join(rows)


def windows(r: dict) -> str:
    wins, low = r["metrics"]["windows"], _low_share(r)
    rows = [
        "| Window | Train | Validation | Test | Train rows | Test rows | Test late share "
        "| Test narrative share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for w in r["windows"]["windows"]:
        name, d = w["window"], w["dates"]
        flag = " (low)" if name in low else ""
        rows.append(
            f"| {name} | {' to '.join(d['train'])} | {' to '.join(d['val'])} "
            f"| {' to '.join(d['test'])} | {w['train']['rows']:,} | {w['test']['rows']:,} "
            f"| {wins[name]['test_prevalence']:.1%} "
            f"| {wins[name]['narrative_share_test']:.1%}{flag} |"
        )
    return "\n".join(rows)


def metrics(r: dict) -> str:
    rows = [
        "| Window | Chance | Company late rate alone | LR, struct | LR, both | both - struct "
        "| LR, both: AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in _windows(r):
        lr = _cell(r, "lr", "both", w)
        d = _find(r["metrics"], w, ["lr", "both"], ["lr", "struct"])["pr_auc"]
        rows.append(
            f"| {w} | {r['metrics']['windows'][w]['test_prevalence']:.3f} "
            f"| {_ci(_cell(r, 'company_rate', 'struct', w), 'pr_auc')} "
            f"| {_ci(_cell(r, 'lr', 'struct', w), 'pr_auc')} | {_ci(lr, 'pr_auc')} "
            f"| {_diff(d)} | {_ci(lr, 'auroc')} |"
        )
    return "\n".join(rows)


def thresholds(r: dict) -> str:
    latest = _windows(r)[-1]
    c = _cell(r, "lr", "both", latest)
    costs = r["metrics"]["costs"]
    rows = [
        "Costs for the cost-optimal rule: a missed late response counts "
        f"{costs['fn'] / costs['fp']:g} times a needless review.",
        "",
        "| Rule (chosen on validation) | Share of test flagged | Precision (95% CI) "
        "| Recall (95% CI) |",
        "|---|---|---|---|",
    ]
    for rule in c["thresholds"]:
        rows.append(
            f"| {RULE_LABELS.get(rule, rule)} | {c['point'][f'{rule}:flagged']:.1%} "
            f"| {_ci(c, f'{rule}:precision')} | {_ci(c, f'{rule}:recall')} |"
        )
    return "\n".join(rows)


def calibration(r: dict) -> str:
    rows = [
        "| Window | ECE (95% CI) | Brier (95% CI) | Log loss (95% CI) |",
        "|---|---|---|---|",
    ]
    for w in _windows(r):
        c = _cell(r, "lr", "both", w)
        rows.append(
            f"| {w} | {_ci(c, 'ece', 4)} | {_ci(c, 'brier', 4)} | {_ci(c, 'log_loss', 4)} |"
        )
    rows += [
        "",
        "Mean ± sd across windows, `both`:",
        "",
        "| Model | PR-AUC | ECE | Brier |",
        "|---|---|---|---|",
    ]
    for m in export.CALIBRATION_MODELS:
        a = r["metrics"]["across_windows"][f"{m}__both"]
        rows.append(
            f"| {MODEL_LABELS[m]} | "
            + " | ".join(
                f"{a[k]['mean']:.4f} ± {a[k]['sd']:.4f}" for k in ("pr_auc", "ece", "brier")
            )
            + " |"
        )
    return "\n".join(rows)


def scorer(r: dict) -> str:
    latest = _windows(r)[-1]
    rows = [
        "| Model | PR-AUC | AUROC | Log loss | ECE |",
        "|---|---|---|---|---|",
    ]
    for m in ("lr", "scorer"):
        c = _cell(r, m, "both", latest)
        rows.append(
            f"| {MODEL_LABELS[m]} | {_ci(c, 'pr_auc')} | {_ci(c, 'auroc')} "
            f"| {_ci(c, 'log_loss', 4)} | {_ci(c, 'ece', 4)} |"
        )
    return "\n".join(rows)


def proxies(r: dict) -> str:
    e = r["explain"]
    latest = e["main_window"]
    groups = {g["group"]: g["share"] for g in e["lr"][f"lr__both__{latest}"]["groups"]}
    terms = [t["term"] for t in e["top_terms"][f"lr__both__{latest}"]["up"][:6]]
    return (
        f"On {latest} test, company-history features carry {groups['company history']:.0%} of "
        f"the LR's contribution (product {groups['product']:.0%}, issue {groups['issue']:.0%}, "
        f"narrative terms {groups['text']:.0%}, state {groups['state']:.0%}, tags "
        f"{groups['tags']:.0%}). The narrative terms that raise the score most are "
        f"{_and([f'"{t}"' for t in terms])}: mostly the names of companies and their apps."
    )


BLOCKS = {
    "answer": answer,
    "results": results,
    "data": data,
    "shift": shift,
    "scorer_gap": scorer_gap,
    "deep": deep,
    "compute": compute,
    "v1": v1,
    "windows": windows,
    "metrics": metrics,
    "thresholds": thresholds,
    "calibration": calibration,
    "scorer": scorer,
    "proxies": proxies,
}


def fill(text: str, r: dict) -> str:
    """`text` with every generated block re-rendered; unknown or unclosed blocks are errors."""
    opened = OPENING.findall(text)
    if opened != [m.group(2) for m in BLOCK.finditer(text)]:
        raise ValueError(f"unclosed or malformed generated block among {opened}")

    def sub(m: re.Match) -> str:
        if m.group(2) not in BLOCKS:
            raise KeyError(f"unknown generated block {m.group(2)!r}")
        return f"{m.group(1)}{BLOCKS[m.group(2)](r)}\n{m.group(4)}"

    return BLOCK.sub(sub, text)


def main(cfg: dict) -> int:
    r = load()
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        new = fill(text, r)
        if new != text:
            path.write_text(new, encoding="utf-8")
        print(f"writeup: {path} {'updated' if new != text else 'is current'}")
    return 0
