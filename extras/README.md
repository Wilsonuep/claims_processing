# extras/ — archived, not-actively-used assets

These components are kept for reference and remain runnable, but they are **not
part of the active AM/UAM benchmark workflow**. They depend on the installed
`claims_processing` package (run `pip install -e .` at the repo root first).

## demagog/ — Polish fact-checking benchmark (Demagog)

A full parallel benchmark family (`dem_ga1`–`dem_ga7`) mirroring the UAM agents
with Polish fact-checking prompts and text labels
(`PRAWDA / CZĘŚCIOWA_PRAWDA / FAŁSZ / MANIPULACJA / NIEWERYFIKOWALNE`).

```
demagog/
├── agents_dem/        # 7 agents + shared prompts.py
├── demagog_db.py      # build demagog.db from scraped JSON
├── run_eval_demagog.py
└── scrape/            # demagog.pl web scrapers
```

The agents import shared infrastructure from `claims_processing.*` (installed
package) and reference each other via the flat `agents_dem.*` namespace, so run
them with `extras/demagog` on `PYTHONPATH`:

```bash
# from the repo root
PYTHONPATH=extras/demagog python extras/demagog/run_eval_demagog.py --limit 10
```

Input/output DB paths resolve via `claims_processing.paths`
(`DEMAGOG_DB`, `RESULTS_DEMAGOG_DB`).

## local_builder/ — local Ollama model management

Helpers to pull/verify local models and benchmark them
(`model_registry.py`, `setup_ollama.py`, `benchmark.py`). Optional; only needed
for local (Ollama) inference setup.

## discontinued/ — retired agents

Both retired agents were removed from the active AM benchmark and the remaining
agents renumbered down after each archival; their result rows were **renamed,
not deleted**, in the results DBs. Kept for reference only — they still import
`claims_processing.core.*` but are no longer registered for runs.

`single_web.py` — the former **uam_ga2** agent (zero-shot + ReAct + DuckDuckGo
web tool), renamed **`uam_ga_web_tool_arch`**. It was misconfigured; after its
removal old ga3–ga7 were renumbered to ga2–ga6 and its result rows renamed to
`uam_ga_web_tool_arch__*`.

`rag_claim_decomp.py` — the former **uam_ga3** agent (claim decomposition
through `RAGRetriever`), renamed **`uam_ga_rag_decomp_arch`**. `RAG_MODE` was
never set to `vector`/`hybrid`, so every stored result used lexical BM25 —
making it functionally redundant with `bm25_claim_decomp.py` (identical
prompts; only the retrieval code path differed). After its removal old ga4–ga6
were renumbered to ga3–ga5 and its result rows renamed to
`uam_ga_rag_decomp_arch__*` (migration: `oneoff/archive_ga3_renumber.py`).

## oneoff/ — one-off utilities

`test_api.py` (quick LLM API ping), `json_file_check.py` (JSON validation) and
`archive_ga3_renumber.py` (the 2026-07 results-DB migration that archived
uam_ga3 and renumbered ga4–ga6 → ga3–ga5).
