"""Fetch the raw ZIPs into $CRES_DATA_DIR/raw/ and record their checksums.

A file is downloaded to `<name>.part`, resumed with an HTTP Range request if the
`.part` file exists, and renamed only once complete. A finished file whose
checksum matches the manifest is skipped. A file whose checksum doesn't match is
renamed to `<name>.bad-<date>` (never deleted) and downloaded again.
"""

import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import yaml

from cres.config import raw_dir

MANIFEST = Path("reports/data_manifest.json")
USER_AGENT = "cres-download (github.com/omvyas77/Credit-risk-early-signals)"
CHUNK = 1 << 20


def source_urls(path: Path | str = "config/sources.yaml") -> list[str]:
    """Every URL in sources.yaml: the live file first, then the archive in order."""
    with open(path, encoding="utf-8") as f:
        sources = yaml.safe_load(f)
    return [sources["live"], *sources["archive"]]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict, path: Path = MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = dict(sorted(manifest.items()))
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def fetch(url: str, dest: Path) -> None:
    """Download `url` to `dest`, resuming from `dest.part` if it exists."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if have and response.status != 206:
            have = 0  # server ignored the range; start over
        total = have + int(response.headers.get("Content-Length", 0))
        mode = "ab" if have else "wb"
        done, last_report = have, have
        with open(part, mode) as f:
            while block := response.read(CHUNK):
                f.write(block)
                done += len(block)
                if done - last_report >= 25 * CHUNK:
                    print(f"  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)
                    last_report = done
    if total and part.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {total}")
    part.rename(dest)


def download(urls: list[str], raw: Path, manifest_path: Path = MANIFEST) -> dict:
    """Make sure every URL is in `raw` with a checksum that matches the manifest."""
    raw.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    for i, url in enumerate(urls, 1):
        name = url.rsplit("/", 1)[1]
        dest = raw / name
        entry = manifest.get(name)
        print(f"[{i}/{len(urls)}] {name}", flush=True)
        if dest.exists() and entry:
            if sha256(dest) == entry["sha256"]:
                print("  checksum matches, skipped", flush=True)
                continue
            bad = dest.with_name(f"{name}.bad-{dt.date.today().isoformat()}")
            print(f"  checksum mismatch, kept as {bad.name} and downloading again", flush=True)
            dest.rename(bad)
        if not dest.exists():
            fetch(url, dest)
        manifest[name] = {
            "url": url,
            "bytes": dest.stat().st_size,
            "sha256": sha256(dest),
            "downloaded": dt.datetime.fromtimestamp(dest.stat().st_mtime, dt.UTC).isoformat(
                timespec="seconds"
            ),
        }
        save_manifest(manifest, manifest_path)
        print(f"  {manifest[name]['bytes'] / 1e6:,.1f} MB, sha256 recorded", flush=True)
    return manifest


def main(cfg: dict, only: list[str] | None = None) -> int:
    urls = source_urls()
    if only:
        urls = [u for u in urls if any(key in u.rsplit("/", 1)[1] for key in only)]
        if not urls:
            print("no source matches --only", file=sys.stderr)
            return 1
    download(urls, raw_dir(cfg))
    return 0
