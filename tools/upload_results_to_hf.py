#!/usr/bin/env python
"""Upload the AM-benchmark evaluation result DBs to the Hugging Face Hub.

Courtesy release for master's-thesis transparency. Creates the dataset repo
PRIVATE by default — review it on the Hub, then flip to public from the dataset
settings when you're ready (that public/indexed step is deliberately left to you).

    python tools/upload_results_to_hf.py            # private (default)
    python tools/upload_results_to_hf.py --public   # create/keep public

Requires `huggingface_hub` and a logged-in token with write access
(`hf auth login`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "Wilsonuep/claims_processing_results"

# local path  ->  filename on the Hub
# Pipeline to (re)generate the staged Parquet files:
#   1. tools/_build_hf_dbs.py    -> cleaned .db copies (drop web_tool_arch + question text)
#   2. tools/_db_to_parquet.py   -> the .parquet files uploaded below
FILES = {
    "README_DATASET.md": "README.md",  # rendered as the dataset card
    "results/hf_staging/subsample_analyzed.parquet": "subsample_analyzed.parquet",
    "results/hf_staging/full_benchmark.parquet": "full_benchmark.parquet",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true",
                    help="create the dataset as public (default: private)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    api = HfApi()
    who = api.whoami()
    print(f"Authenticated as: {who.get('name')}")

    api.create_repo(REPO_ID, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    print(f"Repo ready: {REPO_ID}  (private={not args.public})")

    for local, remote in FILES.items():
        path = repo_root / local
        if not path.exists():
            print(f"  SKIP (missing): {local}")
            continue
        size_mb = path.stat().st_size / 1e6
        print(f"  uploading {local}  ->  {remote}  ({size_mb:,.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {remote}",
        )
        print(f"  done: {remote}")

    print(f"\nFinished -> https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
