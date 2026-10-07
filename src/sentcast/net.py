"""HTTP helpers with on-disk caching and a download manifest."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from .config import RAW

# FRED resets connections that present a browser user agent; an honest one works.
USER_AGENT = "SentCast/0.1 (academic research; python-requests)"
MANIFEST = RAW / "manifest.jsonl"


def fetch(url: str, dest: Path, *, refresh: bool = False, timeout: int = 120, retries: int = 3,
          params: dict | None = None) -> Path:
    """Download url to dest unless it is already cached. Appends to the manifest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            dest.write_bytes(r.content)
            _log(url, dest, r.content)
            return dest
        except requests.RequestException as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} tries: {url}: {last}")


def _log(url: str, dest: Path, content: bytes) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rec = {"url": url, "file": dest.name, "bytes": len(content),
           "sha256": hashlib.sha256(content).hexdigest(),
           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(MANIFEST, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
