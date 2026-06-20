# claims_processing

A Polish-language **claim-verification / fact-checking benchmark framework**.
LLM agents — from zero-shot to multi-step ReAct + RAG pipelines — are evaluated
against the **AM/UAM** benchmark (AMU-CAI multiple-choice, labels 0–3) using a
Wikipedia retrieval knowledge base.

> Master's-thesis research codebase. The active workflow is the AM/UAM
> benchmark; the parallel **Demagog** fact-checking benchmark is archived under
> [`extras/`](extras/README.md).

## Repository layout

```
claims_processing/
├── src/claims_processing/      # the installable package (see docs/architecture.md)
│   ├── paths.py                # single source of truth for all data/results paths
│   ├── core/                   # BaseAgent, LLM client, ReAct, BM25/RAG retrieval
│   ├── agents/uam/             # the 7 active agents (uam_ga1–uam_ga7)
│   ├── pipeline/{scrape,prepare}/   # data scraping + DB building
│   ├── evaluation/             # generic eval loop
│   ├── monitoring/             # push-notification progress/crash alerts
│   └── cli/                    # run_eval_am_benchmark + analysis CLIs
├── tools/                      # operational scripts (backup, subset builders)
├── tests/                      # integration test suite
├── examples/                   # .env.example
├── docs/                       # architecture, data pipeline, running, results DB
├── notebooks/                  # exploratory + evaluation-analysis notebooks
├── extras/                     # archived: Demagog benchmark, local_builder, one-offs
├── data/{raw,benchmarks,wiki}/ # datasets + Wikipedia DB (gitignored)
└── results/                    # evaluation result DBs (gitignored)
```

## Quick start

```bash
# 1. Create .venv and install everything (Windows / macOS / Linux)
python loader.py
#    (loader.py runs `pip install -e .`, so the claims_processing package and
#     the `run-am-eval` command become available)

# 2. Configure credentials / paths
cp examples/.env.example .env        # then edit .env (Together.ai key, wiki DB paths)

# 3. Run a small evaluation
run-am-eval --limit 10 --agents uam_ga1,uam_ga3
```

To build the data first, see [docs/data_pipeline.md](docs/data_pipeline.md)
(AM benchmark ingest and the Wikipedia retrieval DB).

## Documentation

| Topic | File |
|-------|------|
| Package layout, agents, eval loop, RAG/BM25 caches | [docs/architecture.md](docs/architecture.md) |
| Scrape → prepare → SQLite data flow | [docs/data_pipeline.md](docs/data_pipeline.md) |
| Running evaluations: modes, tiers, resume, distributed | [docs/running_evaluations.md](docs/running_evaluations.md) |
| Results DB schema, invariants, corruption/repair | [docs/results_db.md](docs/results_db.md) |
| Archived Demagog / local_builder / one-offs | [extras/README.md](extras/README.md) |

## Tests

```bash
python tests/tester.py               # full integration suite
python tests/test_11_am_agent_config.py   # AM agent config checks (no network)
```

## Configuration

Copy `examples/.env.example` to `.env`. Key variables:

- `LLM_BACKEND` — `together` (default) | `ollama` | `vllm` | `llamacpp`
- `LLM_MODEL` — model name string
- `together_api_key` — required for cloud mode
- `BM25_WIKI_DB`, `RAG_WIKI_DB` — wiki.db paths (default `data/wiki/wiki.db`)
- `MONITORING_ACTIVE`, `BRRR_WEBHOOK_URL`, `MACHINE_NAME` — push notifications
