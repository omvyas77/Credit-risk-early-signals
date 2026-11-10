"""Command line: python -m cres <stage>."""

import argparse
import sys

from cres import config

# Stage name -> one-line help. Order is pipeline order.
STAGES = {
    "download": "fetch raw ZIPs into $CRES_DATA_DIR/raw/ and record checksums",
    "ingest": "unzip, load with DuckDB and join the archive to the live file",
    "label": "build the late-response label and choose data_end",
    "features": "build windows and feature sets",
    "train": "train classical models and write prediction files",
    "deep-input": "write the slim deep-model input (narratives, window tags) for Colab",
    "deep": "train MiniLM, DistilBERT (on a GPU) or the same-data LR; write prediction files",
    "scorer": "fit the in-browser scorer's LR (no VADER, pruned) and write its bundle",
    "evaluate": "score every prediction file into reports/metrics.json",
    "explain": "coefficients by group and SHAP",
    "robustness": "cross-product matrix and test-period drift",
    "export": "write site JSON and the scorer bundle into docs/",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cres", description=__doc__)
    parser.add_argument("--config", default=str(config.DEFAULT), help="path to config YAML")
    sub = parser.add_subparsers(dest="stage", required=True, metavar="stage")
    stages = {
        name: sub.add_parser(name, help=text, description=text) for name, text in STAGES.items()
    }
    stages["download"].add_argument(
        "--only", nargs="+", metavar="KEY", help="download only files whose name contains KEY"
    )
    train = stages["train"]
    train.add_argument("--models", nargs="+", metavar="MODEL", help="models to train")
    train.add_argument("--featuresets", nargs="+", metavar="SET", help="text, struct, both")
    train.add_argument("--windows", nargs="+", metavar="W", help="windows, e.g. W1 W5")
    train.add_argument("--force", action="store_true", help="retrain cells that are current")
    stages["deep-input"].add_argument(
        "--windows", nargs="+", metavar="W", help="windows to export (default: deep.windows)"
    )
    deep = stages["deep"]
    deep.add_argument("--model", required=True, choices=["minilm", "distilbert", "lr_sub"])
    deep.add_argument("--window", help="window, e.g. W5 (default: the first of deep.windows)")
    deep.add_argument("--force", action="store_true", help="redo cells whose files exist")
    stages["scorer"].add_argument("--force", action="store_true", help="refit when current")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config.load(args.config)
    if args.stage == "download":
        from cres import download

        return download.main(cfg, args.only)
    if args.stage == "ingest":
        from cres import ingest

        return ingest.main(cfg)
    if args.stage == "label":
        from cres import label

        return label.main(cfg)
    if args.stage == "features":
        from cres import features

        return features.main(cfg)
    if args.stage == "train":
        from cres import models

        return models.main(cfg, args.models, args.featuresets, args.windows, args.force)
    if args.stage == "deep-input":
        from cres import deep

        return deep.main_export(cfg, args.windows)
    if args.stage == "deep":
        from cres import deep

        return deep.main(cfg, args.model, args.window or cfg["deep"]["windows"][0], args.force)
    if args.stage == "scorer":
        from cres import scorer

        return scorer.main(cfg, args.force)
    if args.stage == "evaluate":
        from cres import evaluate

        return evaluate.main(cfg)
    if args.stage == "explain":
        from cres import explain

        return explain.main(cfg)
    if args.stage == "robustness":
        from cres import robustness

        return robustness.main(cfg)
    if args.stage == "export":
        from cres import export

        return export.main(cfg)
    print(f"cres {args.stage}: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
