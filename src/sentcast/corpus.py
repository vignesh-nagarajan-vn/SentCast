"""Text corpus from GDELT GKG 2.0 files (PLAN.md section 3).

Two 15-minute files per UTC day are drawn by a seeded rule, downloaded, filtered and
reduced to one row per kept article. Output: data/interim/corpus/<YYYY-MM>.parquet and
data/interim/corpus_manifest.csv.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import io
import random
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from urllib.parse import urlparse

import pandas as pd
import requests

from .config import INTERIM, load_config
from .net import USER_AGENT

CORPUS_DIR = INTERIM / "corpus"
MANIFEST = INTERIM / "corpus_manifest.csv"
FUNNEL = ["rows", "web", "us", "headline", "keyword", "kept"]

_EXT = re.compile(r"\.(html?|php|aspx?|cms|shtml|story|amp|jsp|ece|stm)$", re.I)
_SPLIT = re.compile(r"[-_+.~,;:!()\s]+")
_ALPHA = re.compile(r"[a-z]+\Z")
_NUM = re.compile(r"\d{1,6}\Z")
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


def slots_for_day(day: dt.date, seed: int, slots_per_day: int = 96) -> list[int]:
    """Seeded permutation of the day's 15-minute slots. The first k entries are the draw."""
    rng = random.Random(f"{seed}-{day:%Y%m%d}")
    return rng.sample(range(slots_per_day), slots_per_day)


def slot_timestamp(day: dt.date, slot: int) -> str:
    return f"{day:%Y%m%d}{slot // 4:02d}{(slot % 4) * 15:02d}00"


def headline_from_url(url: str, min_words: int = 4) -> str:
    """Rebuild a headline from the words in a URL path. Returns '' if too little is left."""
    try:
        path = urlparse(url).path
    except ValueError:
        return ""
    segments = [s for s in path.split("/") if s]
    if not segments:
        return ""
    best = max(segments, key=lambda s: s.count("-") + s.count("_"))
    best = _EXT.sub("", best).lower()
    words = [w for w in _SPLIT.split(best) if w and (_ALPHA.match(w) or _NUM.match(w))]
    if sum(1 for w in words if _ALPHA.match(w)) < min_words:
        return ""
    return " ".join(words)


def clean_title(raw: str) -> str:
    """Page title as written, without HTML entities or a trailing site name after ' | '."""
    title = html.unescape(html.unescape(raw)).strip()
    if " | " in title:
        head = title.rsplit(" | ", 1)[0].strip()
        if len(head.split()) >= 3:
            title = head
    return re.sub(r"\s+", " ", title)


@lru_cache(maxsize=1)
def keyword_pattern() -> re.Pattern:
    terms = load_config()["gkg"]["keywords"]
    return re.compile(r"\b(?:" + "|".join(terms) + r")\b")


def parse_gkg(raw: bytes, cfg: dict) -> tuple[list[dict], dict]:
    """Filter one unzipped GKG file. Returns (kept rows, funnel counts)."""
    us = cfg["us_location_pattern"]
    excluded = set(cfg["excluded_domains"])
    kw = keyword_pattern()
    funnel = dict.fromkeys(FUNNEL, 0)
    out = []
    for line in raw.decode("utf-8", errors="replace").split("\n"):
        if not line:
            continue
        funnel["rows"] += 1
        f = line.split("\t")
        if len(f) < 27 or f[2] != "1" or not f[4].startswith("http"):
            continue
        funnel["web"] += 1
        if us not in f[9]:
            continue
        funnel["us"] += 1
        headline = headline_from_url(f[4], cfg["min_slug_words"])
        if not headline:
            continue
        funnel["headline"] += 1
        if not kw.search(headline):
            continue
        funnel["keyword"] += 1
        domain = f[3].lower()
        if domain in excluded:
            continue
        funnel["kept"] += 1
        tone = f[15].split(",")
        m = _TITLE.search(f[26])
        out.append({
            "gkg_id": f[0], "ts": f[1], "domain": domain, "url": f[4], "headline": headline,
            "title": clean_title(m.group(1)) if m else None,
            "tone": float(tone[0]) if tone and tone[0] else None,
            "tone_pos": float(tone[1]) if len(tone) > 1 and tone[1] else None,
            "tone_neg": float(tone[2]) if len(tone) > 2 and tone[2] else None,
            "wordcount": int(tone[6]) if len(tone) > 6 and tone[6] else None,
        })
    return out, funnel


class NetworkError(RuntimeError):
    """The server could not be reached. Never recorded as a missing file; the run stops instead."""


_BACKOFF = [2, 5, 15, 30, 60, 120, 120, 120]
_FILE_DEADLINE_S = 120      # a file is 5 to 15 MB; a connection slower than this is dropped and retried


def _fetch(url: str, retries: int) -> bytes | None:
    """File content, or None if the server says the file does not exist (HTTP 404).

    Any other failure is retried with increasing waits and then raised, so that a network
    outage cannot be mistaken for a gap in the archive. A download that runs past the
    per-file deadline counts as a failure: on 2026-10-06 single connections slowed to
    about 30 KB/s and held a month open for 40 minutes while new connections were fast.
    """
    last: Exception | None = None
    for attempt in range(max(retries, len(_BACKOFF))):
        try:
            start = time.monotonic()
            with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(20, 30), stream=True) as r:
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                chunks = []
                for chunk in r.iter_content(chunk_size=1 << 18):
                    chunks.append(chunk)
                    if time.monotonic() - start > _FILE_DEADLINE_S:
                        raise requests.Timeout(f"download exceeded {_FILE_DEADLINE_S} s")
                return b"".join(chunks)
        except requests.RequestException as exc:
            last = exc
            time.sleep(_BACKOFF[min(attempt, len(_BACKOFF) - 1)])
    raise NetworkError(f"{url}: {type(last).__name__}: {last}")


def process_day(args: tuple[str, int, dict]) -> tuple[list[dict], list[dict]]:
    """Download and filter the day's draw. Runs in a worker process."""
    day_s, seed, cfg = args
    day = dt.date.fromisoformat(day_s)
    n = cfg["slots_per_day"]
    drawn = slots_for_day(day, seed, n)[:cfg["files_per_day"]]
    used: set[int] = set()
    rows: list[dict] = []
    manifest: list[dict] = []
    for rank, slot in enumerate(drawn):
        content, raw, used_slot, tried = None, None, None, 0
        # the drawn slot first, then each following slot of the same day, wrapping at midnight,
        # until a file exists. A draw fails only if the day has no other file left.
        for s in [(slot + j) % n for j in range(n)]:
            if s in used or (s != slot and s in drawn):
                continue
            used.add(s)
            tried += 1
            content = _fetch(f"{cfg['base_url']}{slot_timestamp(day, s)}.gkg.csv.zip", cfg["max_retries"])
            if content is None:
                continue
            try:
                z = zipfile.ZipFile(io.BytesIO(content))
                raw = z.read(z.namelist()[0])
            except (zipfile.BadZipFile, IndexError, KeyError):
                content, raw = None, None
                continue
            used_slot = s
            break
        ok = raw is not None
        file_ts = slot_timestamp(day, used_slot) if ok else None
        rec = {"date": day_s, "rank": rank, "slot_drawn": slot, "slot_used": used_slot, "ts": file_ts,
               "slots_tried": tried,
               "replaced": bool(ok and used_slot != slot), "ok": ok, "bytes": len(content) if ok else 0,
               "sha256": hashlib.sha256(content).hexdigest() if ok else None}
        if ok:
            kept, funnel = parse_gkg(raw, cfg)
            for r in kept:
                r["rank"] = rank
                r["ts"] = file_ts   # the time GDELT recorded the article
            rows.extend(kept)
            rec.update(funnel)
        manifest.append(rec)
    return rows, manifest


def months_between(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def build_corpus(files_per_day: int | None = None, limit_months: int | None = None) -> None:
    """Download and filter the whole sample, one parquet file per calendar month. Resumable.

    A month is written only when every day in it was processed. A network failure stops
    the run; running the step again continues from the first unfinished month.
    """
    cfg_all = load_config()
    cfg = dict(cfg_all["gkg"])
    if files_per_day is not None:
        cfg["files_per_day"] = files_per_day
    seed = cfg_all["seed"]
    start = dt.date.fromisoformat(cfg_all["sample"]["corpus_start"])
    end = dt.date.fromisoformat(cfg_all["sample"]["corpus_end"])
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        for y, m in months_between(start, end):
            out_path = CORPUS_DIR / f"{y}-{m:02d}.parquet"
            man_path = CORPUS_DIR / f"{y}-{m:02d}.manifest.csv"
            if out_path.exists() and man_path.exists():
                continue
            first = max(start, dt.date(y, m, 1))
            last = min(end, (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)))
            days = [(first + dt.timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]
            rows, manifest = [], []
            for r, man in pool.map(process_day, [(d, seed, cfg) for d in days]):
                rows.extend(r)
                manifest.extend(man)
            df = pd.DataFrame(rows, columns=["gkg_id", "ts", "domain", "url", "headline", "title", "tone",
                                             "tone_pos", "tone_neg", "wordcount", "rank"])
            df.to_parquet(out_path, index=False)
            pd.DataFrame(manifest).to_csv(man_path, index=False)
            ok = sum(1 for x in manifest if x["ok"])
            print(f"{y}-{m:02d}: files {ok}/{len(manifest)}, kept rows {len(df)}", flush=True)
            done += 1
            if limit_months and done >= limit_months:
                break
    load_manifest().to_csv(MANIFEST, index=False)


def load_manifest() -> pd.DataFrame:
    """One row per drawn file, from the per-month manifests."""
    parts = sorted(CORPUS_DIR.glob("*.manifest.csv"))
    if not parts:
        raise FileNotFoundError("no corpus manifests; run the corpus step first")
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def load_corpus() -> pd.DataFrame:
    """All kept articles, deduplicated by URL and by (UTC date, headline), in time order."""
    parts = sorted(CORPUS_DIR.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError("no corpus files; run the corpus step first")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], format="%Y%m%d%H%M%S", utc=True)
    df["date"] = df["ts"].dt.tz_localize(None).dt.normalize()
    df = df.sort_values(["ts", "gkg_id"], kind="stable")
    n_raw = len(df)
    df = df.drop_duplicates("url", keep="first")
    n_url = len(df)
    df = df.drop_duplicates(["date", "headline"], keep="first").reset_index(drop=True)
    df.attrs["dedup"] = {"kept_before_dedup": n_raw, "after_url_dedup": n_url, "after_headline_dedup": len(df)}
    return df
