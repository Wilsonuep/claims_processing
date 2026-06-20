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

## oneoff/ — one-off utilities

`test_api.py` (quick LLM API ping) and `json_file_check.py` (JSON validation).
