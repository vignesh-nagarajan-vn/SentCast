"""Paths, configuration loading and run provenance."""
from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "sentcast.yaml"
# SENTCAST_DATA_DIR points the pipeline at another data folder (dry runs on a copy).
DATA = Path(os.environ["SENTCAST_DATA_DIR"]) if os.environ.get("SENTCAST_DATA_DIR") else ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
SAMPLES = DATA / "samples"
# SENTCAST_RESULTS_DIR redirects all outputs, for dry runs that must not touch results/ or paper/
DRY_RUN = bool(os.environ.get("SENTCAST_RESULTS_DIR"))
RESULTS = Path(os.environ["SENTCAST_RESULTS_DIR"]) if DRY_RUN else ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"


@functools.lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash() -> str:
    """Hash of the config file with line endings normalised, so it is the same on every platform."""
    return hashlib.sha256(CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]


def git_commit() -> str:
    """HEAD commit, with a '-dirty' suffix if anything under src/ or config/ is uncommitted."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "config"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance() -> dict[str, Any]:
    return {
        "git_commit": git_commit(),
        "config_sha256_12": config_hash(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _clean(obj: Any) -> Any:
    """Make numpy and pandas scalars JSON serialisable; NaN becomes null."""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def write_result(name: str, payload: dict[str, Any], config: dict[str, Any] | None = None) -> Path:
    """Write results/<name>.json with provenance and the config that produced it."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"result": name, **provenance(), "config": config if config is not None else {}, **payload}
    path = RESULTS / f"{name}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_clean(out), fh, indent=1, allow_nan=False)
        fh.write("\n")
    return path


def read_result(name: str) -> dict[str, Any]:
    with open(RESULTS / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)
