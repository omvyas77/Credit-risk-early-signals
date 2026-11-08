"""Load the YAML config that every stage reads."""

from pathlib import Path

import yaml

DEFAULT = Path("config/default.yaml")


def load(path: Path | str = DEFAULT) -> dict:
    """Return the config at `path` as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
