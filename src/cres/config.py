"""Load the YAML config that every stage reads, and resolve data paths."""

import json
import os
from pathlib import Path

import yaml

DEFAULT = Path("config/default.yaml")
# The settings each stage last finished with, so editing an unrelated config section doesn't
# make its outputs look stale.
STAMPS = Path("data/stamps")


def load(path: Path | str = DEFAULT) -> dict:
    """Return the config at `path` as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir(cfg: dict) -> Path:
    """Root for raw downloads: $CRES_DATA_DIR if set, else data/ in the repo."""
    return Path(os.environ.get(cfg["data_dir_env"]) or "data")


def raw_dir(cfg: dict) -> Path:
    return data_dir(cfg) / "raw"


def settings(cfg: dict, keys: tuple[str, ...], **extra) -> dict:
    """The config sections in `keys` (plus any `extra` values) in a comparable form."""
    return json.loads(json.dumps({**{k: cfg.get(k) for k in keys}, **extra}, default=str))


def settings_changed(stage: str, current: dict) -> bool:
    """True if `current` differs from the settings `stage` last finished with. Outputs built
    before stages recorded their settings have no record, and count as unchanged."""
    path = STAMPS / f"{stage}.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")) != current


def record_settings(stage: str, current: dict) -> None:
    STAMPS.mkdir(parents=True, exist_ok=True)
    (STAMPS / f"{stage}.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
