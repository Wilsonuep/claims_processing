# Architecture

The active framework is a single installable package, `claims_processing`, laid
out under `src/`. Everything imports as `claims_processing.<layer>.<module>`.

```
src/claims_processing/
├── paths.py            # single source of truth for data/results/wiki locations
├── core/               # shared agent infrastructure
│   ├── base_agent.py   # BaseAgent contract
│   ├── llm_client.py   # universal LLM factory (+ module-level `client` singleton)
│   ├── react.py        # universal ReAct loop + output parsing
│   └── retrieval/
│       ├── bm25.py     # BM25 index (process-level _INDEX_CACHE)
│       └── rag.py      # RAG retriever: bm25 | vector | hybrid (RRF)
├── agents/uam/         # the 6 active AM/UAM agents (ga1–ga6)
├── pipeline/
│   ├── scrape/         # polish_wikipedia_webscrapper.py
│   └── prepare/        # *_db builders, chunking, embedding, am_benchmark_loader
├── evaluation/         # eval_loop.py (generic eval engine)
├── monitoring/         # monitor.py (brrr push notifications)
└── cli/                # run_eval_am_benchmark (entry point) + analysis CLIs
```

## Agent infrastructure (`core/`)

All agents inherit from `BaseAgent` (`core/base_agent.py`) and return these keys
from `eval(claim)`: `model_label`, `original_label`, `is_correct`,
`total_tokens`, `prompt_tokens`, `completion_tokens`, `time_thought`,
`raw_output`, `model_name`.

`core/llm_client.py` is a universal LLM factory — a module-level `client`
singleton is created on import from env vars. Agents pass `model_override` to
`make_client()` to create per-agent clients for multi-model runs.

`core/react.py` implements a universal ReAct loop used by all `*_web.py` agents.
`parse_react_output()` strips configurable thinking tags then extracts JSON
(prefers ```` ```json ```` blocks, falls back to first `{` / last `}`).
`run_react_agent()` defaults to `max_steps=5`; `*_web.py` agents pass
`max_steps=8` explicitly — low step counts cause most `ERROR_MAX_STEPS` failures.

`core/retrieval/bm25.py` and `rag.py` use **process-level caches**
(`_INDEX_CACHE`, `_MODEL_CACHE`, `_embed_cache`) to share the 4–6 GB BM25 index
and embedding model across all agents. Never load these objects twice in the
same process.

## Agent families (`agents/uam/`)

| Agent   | File                       | Strategy                              | cost_tier |
|---------|----------------------------|---------------------------------------|-----------|
| uam_ga1 | single.py                  | Zero-shot JSON                        | 1 |
| uam_ga2 | single_bm25.py             | BM25 Wikipedia retrieval              | 1 |
| uam_ga3 | rag_claim_decomp.py        | Claim decomp + vector RAG             | 2 |
| uam_ga4 | bm25_claim_decomp.py       | Claim decomp + BM25                   | 2 |
| uam_ga5 | fewshot_cot_rag.py         | Few-shot CoT + 3 reasoners + RAG      | 3 |
| uam_ga6 | fewshot_cot_debate_rag.py  | Adversarial debate + judge + RAG      | 3 |

> **Discontinued:** the former **uam_ga2** (`single_web.py`, ReAct + DuckDuckGo
> web tool) was misconfigured and removed from the benchmark. It is archived as
> **`uam_ga_web_tool_arch`** in [`extras/discontinued/single_web.py`](../extras/discontinued/single_web.py);
> its existing result rows were renamed to `uam_ga_web_tool_arch__*` rather than
> deleted. The remaining agents were renumbered down to ga1–ga6 (the table above
> reflects the current numbering).

**AM benchmark quirk**: agents use `claim.get("label_original", "") or
claim.get("label", "")` for the ground-truth label, and call
`_build_question_with_answers()` to inject answer choices from
`claim["metadata"]["answers"]` into the question string.

**Multi-model runs**: when `model_override` is passed, `__init__` creates a
private `(client, model)` pair, patches module-level globals during `eval()`,
then restores them. Agent name becomes `uam_ga2__model-name`. Safe in
sequential mode; thread-safety is not guaranteed in cloud-parallel mode.

The archived **Demagog** family (`dem_ga1`–`dem_ga7`) mirrors this structure
with Polish fact-checking prompts; it now lives in `extras/demagog/` (see
[`extras/README.md`](../extras/README.md)).

## Agent registration

Agents for a run are wired in `_register_default_agents()` inside
`src/claims_processing/cli/run_eval_am_benchmark.py`. To add or remove agents
from a run, edit that function.

## Evaluation loop (`evaluation/eval_loop.py`)

The generic engine, invokable directly:

```bash
python -m claims_processing.evaluation.eval_loop --benchmarks am_benchmark --agents uam_ga1,uam_ga2
```

The wrapper `run_eval_am_benchmark.py` calls the same `eval_benchmark*`
functions and registers its own agents — for normal use prefer the wrapper,
since direct invocation skips `_register_default_agents()`.

Three modes:
- **sequential** — single-threaded, used by tests
- **cloud** — `ThreadPoolExecutor(workers)`, for Together.ai / cloud APIs
- **local** — tiered: tier-1 on full dataset, tier-2 on `--tier2-limit`,
  tier-3 on `--tier3-limit`

See [running_evaluations.md](running_evaluations.md) for modes/tiers and
[results_db.md](results_db.md) for the results schema and invariants.

## RAG pipeline (`core/retrieval/rag.py`)

Three retrieval modes: `bm25`, `vector`, `hybrid`. Hybrid uses Reciprocal Rank
Fusion (RRF, k=60) to merge BM25 and vector rankings. Two-stage: fetch
`k_initial=20` candidates, re-rank/filter by `score_threshold`, return
`k_final=5`. Embedding model is `sdadas/mmlw-retrieval-roberta-large-v2`
(768-dim), loaded once and cached in `pipeline/prepare/wikipedia_embedding.py`.

## Monitoring (`monitoring/monitor.py`)

`MonitoringAgent` fires brrr push notifications at scheduled times (08:00,
14:00, 19:00) and on crash/done events via background daemon threads. The eval
loop calls `monitoring.update()` after each claim — never blocks. Disabled when
`MONITORING_ACTIVE != "true"`.
