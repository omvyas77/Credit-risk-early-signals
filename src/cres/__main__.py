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
    "deep": "train MiniLM or DistilBERT (run on a GPU) and write prediction files",
    "evaluate": "score every prediction file into reports/metrics.json",
    "explain": "coefficients by group and SHAP",
    "robustness": "cross-product matrix and test-period drift",
    "export": "write site JSON and the scorer bundle into docs/",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cres", description=__doc__)
    parser.add_argument("--config", default=str(config.DEFAULT), help="path to config YAML")
    sub = parser.add_subparsers(dest="stage", required=True, metavar="stage")
    for name, text in STAGES.items():
        sub.add_parser(name, help=text, description=text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.load(args.config)
    print(f"cres {args.stage}: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
