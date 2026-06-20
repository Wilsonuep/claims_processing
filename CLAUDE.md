# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polish-language claim verification / fact-checking benchmark framework. The
**active** benchmark is **AM/UAM** (AMU-CAI multiple-choice, labels 0–3). A
second family, **Demagog** (Polish fact-checks, text labels), is **archived**
under `extras/demagog/` and is not part of the active workflow. Agents range
from zero-shot to multi-step ReAct + RAG pipelines.

## Repository layout (src/ package)

All active code is one installable package, `claims_processing`, under `src/`.
Imports are `claims_processing.<layer>.<module>`.

```
src/claims_processing/
├── paths.py            # SINGLE SOURCE OF TRUTH for data/results/wiki paths
├── core/               # base_agent, llm_client, react  (+ retrieval/{bm25,rag})
├── agents/uam/         # uam_ga1–uam_ga7
├── pipeline/{scrape,prepare}/   # scrapers + DB builders/chunking/embedding
├── evaluation/         # eval_loop.py
├── monitoring/         # monitor.py
└── cli/                # run_eval_am_benchmark (entry point) + analyze/fix/merge
tools/        # operational scripts (backup, subset/subsample builders)
tests/        # integration suite
examples/     # .env.example
docs/         # architecture / data_pipeline / running_evaluations / results_db
notebooks/    # exploratory + evaluation analysis
extras/       # archived: demagog/, local_builder/, oneoff/
data/{raw,benchmarks,wiki}/   # datasets + wiki.db (gitignored)
results/      # result DBs (gitignored)
```

**Path rule:** never hardcode `data/...` or `results/...` paths or recompute
`PROJECT_ROOT` via `Path(__file__).parent.parent`. Import from
`claims_processing.paths` (`AM_BENCHMARK_DB`, `AM_BENCHMARK_4K_DB`, `WIKI_DB`,
`RESULTS_AM_DB`, `bm25_wiki_db()`, `rag_wiki_db()`, …). The data hierarchy and
deep package nesting are both safe to change because every path resolves there.

## Setup

```bash
python loader.py          # creates .venv, installs deps, AND `pip install -e .`
python loader.py --force  # rebuild from scratch
```

`pip install -e .` (run by loader.py) is **required** — without the editable
install `import claims_processing` fails. Copy `examples/.env.example` to `.env`
at the repo root. Key env vars:
- `LLM_BACKEND` — `together` (default) | `ollama` | `vllm` | `llamacpp`
- `LLM_MODEL` — model name string
- `together_api_key` — required for cloud mode
- `BM25_WIKI_DB`, `RAG_WIKI_DB` — wiki.db paths (default `data/wiki/wiki.db`)
- `STRIP_THINKING_TAGS` — tags to strip (default `think,reasoning,scratchpad`)
- `MONITORING_ACTIVE`, `BRRR_WEBHOOK_URL`, `MACHINE_NAME` — push notifications

## Running evaluations

```bash
# AM benchmark (primary). Console script `run-am-eval` == the module form.
python -m claims_processing.cli.run_eval_am_benchmark
run-am-eval --limit 10 --agents uam_ga1,uam_ga2 --models llama3.1:8b
run-am-eval --subset            # use data/benchmarks/am_benchmark_4k.db

# Merge results from multiple machines
python -m claims_processing.cli.merge_results --target results/merged.db \
    --sources results/results_am_benchmark.db other.db

# Generic eval-loop CLI (advanced — wrappers above call into this)
python -m claims_processing.evaluation.eval_loop --benchmarks am_benchmark --agents uam_ga1,uam_ga2
python -m claims_processing.evaluation.eval_loop --mode cloud --workers 10
python -m claims_processing.evaluation.eval_loop --clear --export-csv   # destructive; scope with --agents

# Analyze / repair corrupted results
python -m claims_processing.cli.analyze_results
python -m claims_processing.cli.fix_corrupted_results --dry-run
python -m claims_processing.cli.fix_corrupted_results

# Backup result/benchmark DBs (non-blocking, safe during eval runs)
python tools/backup_dbs.py                  # -> D:\claims_backup\ by default
python tools/schedule_backup.py --interval 3
```

`run_eval_am_benchmark.py` uses `data/benchmarks/am_benchmark.db` as input and
defaults to `--mode local`.

## Tests

```bash
python tests/tester.py                       # full integration suite
python tests/test_04_eval_local.py           # LLM ping + local eval
python tests/test_09_bm25_polish.py          # Polish BM25 (no network)
python tests/test_11_am_agent_config.py      # AM agent config checks (no network)
python tests/eval_completeness_test.py --results-db results/merged.db
```

Tests import siblings directly, so run them from the `tests/` directory (or via
`tester.py`). The Demagog DB test moved to `extras/demagog/test_demagog_db.py`.

## Data pipeline

See [docs/data_pipeline.md](docs/data_pipeline.md). Summary:

```bash
# Wikipedia: scrape -> chunk+embed -> sqlite-vec (data/wiki/wiki.db)
python -m claims_processing.pipeline.scrape.polish_wikipedia_webscrapper
python -m claims_processing.pipeline.prepare.build_wikipedia_db \
    --input data/raw/polish_wikipedia_articles.jsonl --db data/wiki/wiki.db

# AM Benchmark
python -m claims_processing.pipeline.prepare.am_benchmark_loader
python -m claims_processing.pipeline.prepare.am_benchmark_db --input data/benchmarks/am_benchmark.csv
```

## Architecture

Full detail in [docs/architecture.md](docs/architecture.md). Key points:

- **Agent contract** (`core/base_agent.py`): `eval(claim)` returns `model_label`,
  `original_label`, `is_correct`, `total_tokens`, `prompt_tokens`,
  `completion_tokens`, `time_thought`, `raw_output`, `model_name`.
- `core/llm_client.py` is a universal factory with a module-level `client`
  singleton; agents pass `model_override` to `make_client()` for multi-model runs.
- `core/react.py` is the universal ReAct loop; `*_web.py` agents pass
  `max_steps=8` (low step counts cause most `ERROR_MAX_STEPS`).
- `core/retrieval/{bm25,rag}.py` use **process-level caches** (`_INDEX_CACHE`,
  `_MODEL_CACHE`, `_embed_cache`) to share the 4–6 GB BM25 index and embedding
  model across agents. Never load these twice in one process.
- **Agent registration**: edit `_register_default_agents()` in
  `src/claims_processing/cli/run_eval_am_benchmark.py`.
- **AM benchmark quirk**: agents use
  `claim.get("label_original", "") or claim.get("label", "")` for the
  ground-truth label and call `_build_question_with_answers()` to inject answer
  choices from `claim["metadata"]["answers"]`.
- **Multi-model runs**: agent name becomes `uam_ga2__model-name`; globals are
  patched during `eval()` then restored. Safe sequential; not thread-safe in
  cloud-parallel mode.

### Evaluation loop (`evaluation/eval_loop.py`)

Modes: **sequential** (tests), **cloud** (`ThreadPoolExecutor(workers)`),
**local** (tiered: tier-1 full dataset, tier-2 `--tier2-limit`, tier-3
`--tier3-limit`). Wrapper defaults are **20000 / 20000** (effectively no cap vs
the 18,820-row benchmark); the direct eval-loop CLI defaults to **2000 / 500**.

**Crash recovery**: on startup `get_evaluated_claim_ids(model_name=…)` deletes
`model_label='ERROR'` rows and returns processed IDs so they are skipped.
`ERROR_MAX_STEPS` rows are **not** auto-deleted — use
`claims_processing.cli.fix_corrupted_results`.

## Results database — invariants (do not break)

See [docs/results_db.md](docs/results_db.md). Location:
`results/results_am_benchmark.db`, table `agent_results`. A UNIQUE INDEX on
`(agent_name, claim_id, benchmark_name, model_name)` prevents duplicates.

- **Always `INSERT OR IGNORE`** in any new write path — never plain `INSERT`.
- **Never `DELETE FROM agent_results` unscoped** — always scope by
  `(agent_name, benchmark_name, model_name)`.
- **`agent_name` always carries `__<model-suffix>`** (e.g. `uam_ga6__llama3.1-8b`);
  `register_agent()` applies it.
- **`get_evaluated_claim_ids` takes a `model_name` arg**.

These exist because a bare `--clear` once permanently wiped Bielik results for
`uam_ga6`/`uam_ga7` before a llama re-run. Bielik suffix:
`__hf.co-speakleash-Bielik-11B-v2.3-Instruct-GGUF-Q4_K_M`; llama: `__llama3.1-8b`.

### Current data state (2026-04-28, post-naming-sweep)

All rows in `agent_results` carry a `__<model-suffix>` (no un-suffixed legacy
rows).

| Agent | Bielik (Q4_K_M) | llama3.1:8b |
|-------|-----------------|-------------|
| uam_ga1 | complete (18,820) | complete (18,820) |
| uam_ga2 | complete (18,820) | registered for run |
| uam_ga3 | complete (18,820) | registered for run |
| uam_ga4 | complete (18,820) | registered for run |
| uam_ga5 | complete (18,820) | registered for run |
| uam_ga6 | needs re-run (Bielik lost) | complete (18,820) |
| uam_ga7 | needs re-run (Bielik lost) | resume from 3,397/18,820 |

Bielik re-run for ga6/ga7 (after the llama batch, without `--clear`) — set
`LLM_MODEL=hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M` in `.env` (or
change `model_override` in `_register_default_agents()`), then:
```bash
run-am-eval --agents uam_ga6,uam_ga7
```

## Archived assets (`extras/`)

The Demagog benchmark (`agents_dem`, `run_eval_demagog`, `demagog_db`,
scrapers), `local_builder/` (Ollama setup), and one-off scripts live under
`extras/`. They depend on the installed `claims_processing` package; the demagog
code runs with `PYTHONPATH=extras/demagog`. See [extras/README.md](extras/README.md).
