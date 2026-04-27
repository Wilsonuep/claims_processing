# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polish-language claim verification / fact-checking benchmark framework. Two benchmark families: **UAM** (AM-CAI multiple-choice, labels 0–3) and **Demagog** (Polish fact-checks, labels PRAWDA/CZĘŚCIOWA_PRAWDA/FAŁSZ/MANIPULACJA/NIEWERYFIKOWALNE). Agents range from zero-shot to multi-step ReAct + RAG pipelines.

## Setup

```bash
python loader.py          # creates .venv, installs all deps
python loader.py --force  # rebuild from scratch
```

Copy `.env.example` to `.env`. Key env vars:
- `LLM_BACKEND` — `together` (default) | `ollama` | `vllm` | `llamacpp`
- `LLM_MODEL` — model name string
- `together_api_key` — required for cloud mode
- `BM25_WIKI_DB`, `RAG_WIKI_DB` — paths to wiki.db for retrieval agents
- `STRIP_THINKING_TAGS` — comma-separated tags to strip (default: `think,reasoning,scratchpad`)
- `MONITORING_ACTIVE`, `BRRR_WEBHOOK_URL`, `MACHINE_NAME` — push notifications

## Running Evaluations

```bash
# Run AM benchmark (primary)
python scripts/run_eval_am_benchmark.py
python scripts/run_eval_am_benchmark.py --limit 10 --agents uam_ga1,uam_ga2 --models llama3.1:8b

# Run Demagog benchmark
python scripts/run_eval_demagog.py
python scripts/run_eval_demagog.py --limit 10

# Parallel cloud mode / local tiered mode are set inside the scripts
# Merge results from multiple machines
python -m scripts.merge_results --target results/merged.db --sources results/results_am_benchmark.db other.db

# Analyze corrupted results
python scripts/analyze_results.py
python scripts/fix_corrupted_results.py --dry-run
python scripts/fix_corrupted_results.py
```

## Tests

```bash
python tests/tester.py          # full integration suite (12 tests)
python tests/test_04_eval_local.py   # LLM ping + local eval
python tests/test_05_eval_cloud.py   # parallel eval
python tests/test_08_crash_recovery.py
python tests/eval_completeness_test.py --results-db results/merged_eval.db
```

## Data Pipeline

```bash
# Wikipedia: scrape → chunk+embed → SQLite
python datascrap/polish_wikipedia_webscrapper.py
python dataprep/build_wikipedia_db.py --input polish_wikipedia_articles.jsonl

# Demagog
python datascrap/demagog_webscrapper.py && python datascrap/demagog_det_webscrapper.py
python dataprep/demagog_db.py --input data/demagog_wypowiedzi_detailed.json

# AM Benchmark
python data/am_benchmark_loader.py
python dataprep/am_benchmark_db.py --input data/am_benchmark.csv
```

## Architecture

### Agent Infrastructure (`gen_agent/`)

All agents inherit from `BaseAgent` (`gen_agent/base_agent.py`) and must return these keys from `eval(claim)`: `model_label`, `original_label`, `is_correct`, `total_tokens`, `prompt_tokens`, `completion_tokens`, `time_thought`, `raw_output`, `model_name`.

`llm_client.py` is a universal LLM factory — a module-level `client` singleton is created on import from env vars. Agents pass `model_override` to `make_client()` to create per-agent clients for multi-model evaluation runs.

`react.py` implements a universal ReAct loop used by all `*_web.py` agents. `parse_react_output()` strips configurable thinking tags then extracts JSON (prefers ` ```json ``` ` blocks, falls back to first `{` / last `}`). Max steps default is 8 (was 5 — low step counts cause most `ERROR_MAX_STEPS` failures).

`bm25.py` and `rag.py` use **process-level caches** (`_INDEX_CACHE`, `_MODEL_CACHE`, `_embed_cache`) to share the 4–6 GB BM25 index and embedding model across all agents. Never load these objects twice in the same process.

### Agent Families

**UAM agents** (`agents_uam/`, names `uam_ga1`–`uam_ga7`):
| Agent | File | Strategy | cost_tier |
|-------|------|----------|-----------|
| uam_ga1 | single.py | Zero-shot JSON | 1 |
| uam_ga2 | single_web.py | ReAct + DuckDuckGo | 1 |
| uam_ga3 | single_bm25.py | BM25 Wikipedia retrieval | 1 |
| uam_ga4 | rag_claim_decomp.py | Claim decomp + vector RAG | 2 |
| uam_ga5 | bm25_claim_decomp.py | Claim decomp + BM25 | 2 |
| uam_ga6 | fewshot_cot_rag.py | Few-shot CoT + 3 reasoners + RAG | 2 |
| uam_ga7 | fewshot_cot_debate_rag.py | Adversarial debate + judge + RAG | 3 |

**Demagog agents** (`agents_dem/`, names `dem_ga1`–`dem_ga7`) mirror the UAM structure with Polish fact-checking prompts and text labels instead of 0–3.

**AM benchmark quirk**: Agents must use `claim.get("label_original", "") or claim.get("label", "")` for the ground-truth label, and call `_build_question_with_answers()` to inject answer choices from `claim["metadata"]["answers"]` into the question string.

**Multi-model runs**: When `model_override` is passed, `__init__` creates a private `(client, model)` pair, patches module-level globals during `eval()`, then restores them. Agent name becomes `uam_ga2__model-name`. Safe in sequential mode; thread-safety is not guaranteed in cloud-parallel mode.

### Evaluation Loop (`eval/eval_loop.py`)

Three modes wired in `run_eval_*.py` scripts:
- **sequential** — single-threaded, used by tests
- **cloud** — `ThreadPoolExecutor(workers)`, for Together.ai / cloud APIs
- **local** — tiered: tier-1 agents on full dataset, tier-2 on `--tier2-limit` (default 2000), tier-3 on `--tier3-limit` (default 500)

**Crash recovery**: On startup, `get_evaluated_claim_ids()` queries existing results, deletes `model_label='ERROR'` rows, and returns already-processed claim IDs so they are skipped. `ERROR_MAX_STEPS` rows are **not** auto-deleted — use `scripts/fix_corrupted_results.py` to handle them.

Results DB schema is in `eval/eval_loop.py`; the `model_name` column is auto-migrated if absent.

### RAG Pipeline (`gen_agent/rag.py`)

Three retrieval modes: `bm25`, `vector`, `hybrid`. Hybrid uses Reciprocal Rank Fusion (RRF, k=60) to merge BM25 and vector rankings. Two-stage: fetch `k_initial=20` candidates, re-rank/filter by `score_threshold`, return `k_final=5`. Embedding model is `sdadas/mmlw-retrieval-roberta-large-v2` (768-dim), loaded once and cached in `dataprep/wikipedia_embedding.py`.

### Monitoring (`monitoring/monitor.py`)

`MonitoringAgent` fires brrr push notifications at scheduled times (08:00, 14:00, 19:00) and on crash/done events via background daemon threads. The eval loop calls `monitoring.update()` after each claim — never blocks. Disabled when `MONITORING_ACTIVE != "true"`.

## Results Database

Location: `results/results_am_benchmark.db`. Table: `agent_results`.

Valid labels: `0`–`3` (UAM), `PRAWDA/CZĘŚCIOWA_PRAWDA/FAŁSZ/MANIPULACJA/NIEWERYFIKOWALNE` (Demagog). Common corruptions:
- `ERROR_MAX_STEPS` — agent hit step limit; trajectory in `raw_output` often has a recoverable label
- `ERROR` — exception; auto-deleted on next eval run
- Float labels (`"1.0"`), prefix labels (`"Output: 2"`) — `_normalize_uam_label()` in `agents_uam/single_web.py` handles these going forward

Use `scripts/analyze_results.py` for a full breakdown and `scripts/fix_corrupted_results.py --dry-run` before applying repairs.

### Results DB invariants — do not break these

A UNIQUE INDEX on `(agent_name, claim_id, benchmark_name, model_name)` prevents duplicate rows at the DB level.

- **Always use `INSERT OR IGNORE`** in any new write path — never plain `INSERT`.
- **Never use bare `DELETE FROM agent_results`** — always scope by `(agent_name, benchmark_name, model_name)`.
- **`agent_name` always carries a `__<model-suffix>`** (e.g. `uam_ga6__llama3.1-8b`). `register_agent()` applies this automatically so Bielik and llama rows for the same agent are distinguishable by name alone, not just by `model_name` column.
- **`get_evaluated_claim_ids` takes a `model_name` arg** — different models for the same agent accumulate as independent row sets.

These rules exist because a bare `--clear` once wiped Bielik results for `uam_ga6` and `uam_ga_7` before a llama re-run, losing data permanently.

### Current data state (2026-04-28)

| Agent | Model | Status |
|-------|-------|--------|
| uam_ga1–ga5 | Bielik | complete |
| uam_ga6 | llama3.1:8b | complete — **Bielik run lost, needs re-run** |
| uam_ga_7 | llama3.1:8b | in progress (~3k/18820) — **Bielik run lost, needs re-run** |

Re-run command (after ga_7 llama finishes, without `--clear`):
```bash
python -m scripts.run_eval_am_benchmark --agents uam_ga6,uam_ga_7 --models "hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M"
```
