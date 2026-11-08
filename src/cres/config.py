"""Load the YAML config that every stage reads, and resolve data paths."""

import os
from pathlib import Path

import yaml

DEFAULT = Path("config/default.yaml")


def load(path: Path | str = DEFAULT) -> dict:
    """Return the config at `path` as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir(cfg: dict) -> Path:
    """Root for raw downloads: $CRES_DATA_DIR if set, else data/ in the repo."""
    return Path(os.environ.get(cfg["data_dir_env"]) or "data")


def raw_dir(cfg: dict) -> Path:
    return data_dir(cfg) / "raw"
