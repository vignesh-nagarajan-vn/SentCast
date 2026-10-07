"""One command from raw downloads to every table and figure.

    python -m sentcast.run                  # all steps
    python -m sentcast.run --steps score,experiments,report
    python -m sentcast.run --only E03,E05   # selected experiments, then nothing else

Steps, in order:
    download     market, macro, published-index and scorer input files -> data/raw
    corpus       seeded sample of GDELT GKG files -> data/interim/corpus (resumable)
    score        headline and title scores -> data/interim/scores.parquet
    experiments  E01 .. E12 -> results/E##_*.json
    power        simulated power of the main test -> results/power.json (not pre-specified)
    report       tables, figures, results/SUMMARY.md, copies under paper/
    outline      docs/outline.md from docs/outline.template.md, numbers from results/
"""
from __future__ import annotations

import argparse
import random

import numpy as np

from .config import load_config

STEPS = ["download", "corpus", "score", "experiments", "power", "report", "outline"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of: " + ", ".join(STEPS))
    ap.add_argument("--only", default="", help="comma-separated experiment ids, e.g. E03,E05 (implies --steps experiments)")
    ap.add_argument("--refresh", action="store_true", help="download market and macro files again")
    args = ap.parse_args(argv)
    steps = ["experiments"] if args.only else [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps: {sorted(unknown)}")

    seed = load_config()["seed"]
    random.seed(seed)
    np.random.seed(seed)

    if "download" in steps:
        from . import data
        data.download_all(refresh=args.refresh)
    if "corpus" in steps:
        from . import corpus
        corpus.build_corpus()
    if "score" in steps:
        from . import pipeline
        pipeline.score_corpus()
    if "experiments" in steps:
        from . import experiments
        experiments.run([e.strip() for e in args.only.split(",") if e.strip()] or None)
    if "power" in steps:
        from . import power
        power.main()
    if "report" in steps:
        from . import report
        report.main()
    if "outline" in steps:
        from . import outline
        outline.main()


if __name__ == "__main__":
    main()
