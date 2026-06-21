# Data pipeline

Flow: **scrape → prepare → SQLite DBs → evaluation**. All data lives under the
`data/` hierarchy (gitignored); paths resolve through
`claims_processing.paths`.

```
data/
├── raw/          # scraped/downloaded dumps (jsonl, json)
├── benchmarks/   # am_benchmark.csv/.db/_4k.db, demagog.db
└── wiki/         # wiki.db (Wikipedia vector + BM25 corpus, ~85 GB)
```

## AM benchmark (active)

```bash
# Download from HuggingFace Hub (amu-cai/llmzszl-dataset) into data/benchmarks/
python -m claims_processing.pipeline.prepare.am_benchmark_loader

# Ingest CSV -> SQLite (data/benchmarks/am_benchmark.db, table `claims`)
python -m claims_processing.pipeline.prepare.am_benchmark_db --input data/benchmarks/am_benchmark.csv
```

A 4,000-claim debugging subset can be built with:

```bash
python tools/build_am_benchmark_subset.py        # -> data/benchmarks/am_benchmark_4k.db
```

## Wikipedia (for BM25 / RAG agents)

```bash
# 1. Scrape Polish Wikipedia -> data/raw/polish_wikipedia_articles.jsonl
python -m claims_processing.pipeline.scrape.polish_wikipedia_webscrapper

# 2. Stream JSONL -> chunk -> embed -> sqlite-vec (data/wiki/wiki.db). Resume-safe.
python -m claims_processing.pipeline.prepare.build_wikipedia_db \
    --input data/raw/polish_wikipedia_articles.jsonl --db data/wiki/wiki.db
```

`build_wikipedia_db` orchestrates `wikipeda_chunking` (sentence-level chunks),
`wikipedia_embedding` (`sdadas/mmlw-retrieval-roberta-large-v2`, cached model),
and `wikipedia_db` (SQLite + sqlite-vec, WAL mode, batch inserts, `knn_search`).

Agents locate `wiki.db` via `paths.bm25_wiki_db()` / `paths.rag_wiki_db()`,
which honor the `BM25_WIKI_DB` / `RAG_WIKI_DB` env overrides (default
`data/wiki/wiki.db`).

## Demagog (archived)

The Demagog scrapers and DB builder live in `extras/demagog/`. See
[`extras/README.md`](../extras/README.md).
