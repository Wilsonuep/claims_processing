# Results database

Location: `results/results_am_benchmark.db`. Table: `agent_results`. The schema
lives in `src/claims_processing/evaluation/eval_loop.py`; the `model_name`
column is auto-migrated if absent.

Valid labels:
- `0`–`3` (UAM / AM benchmark)
- `PRAWDA / CZĘŚCIOWA_PRAWDA / FAŁSZ / MANIPULACJA / NIEWERYFIKOWALNE` (Demagog)

## Common corruptions

- `ERROR_MAX_STEPS` — agent hit the ReAct step limit; the trajectory in
  `raw_output` often has a recoverable label.
- `ERROR` — exception; auto-deleted on the next eval run.
- Float labels (`"1.0"`), prefix labels (`"Output: 2"`) —
  `_normalize_uam_label()` in `agents/uam/single_web.py` handles these going
  forward.

Inspect and repair:

```bash
python -m claims_processing.cli.analyze_results
python -m claims_processing.cli.fix_corrupted_results --dry-run
python -m claims_processing.cli.fix_corrupted_results
```

## Invariants — do not break these

A UNIQUE INDEX on `(agent_name, claim_id, benchmark_name, model_name)` prevents
duplicate rows at the DB level.

- **Always use `INSERT OR IGNORE`** in any new write path — never plain `INSERT`.
- **Never use bare `DELETE FROM agent_results`** — always scope by
  `(agent_name, benchmark_name, model_name)`.
- **`agent_name` always carries a `__<model-suffix>`** (e.g.
  `uam_ga6__llama3.1-8b`). `register_agent()` applies this automatically so
  Bielik and llama rows for the same agent are distinguishable by name alone,
  not just by the `model_name` column.
- **`get_evaluated_claim_ids` takes a `model_name` arg** — different models for
  the same agent accumulate as independent row sets.

These rules exist because a bare `--clear` once wiped Bielik results for
`uam_ga6`/`uam_ga7` before a llama re-run, losing data permanently.

Model suffixes in use: Bielik =
`__hf.co-speakleash-Bielik-11B-v2.3-Instruct-GGUF-Q4_K_M`, llama =
`__llama3.1-8b`.

## Backups

DBs are backed up non-blocking (safe during eval runs):

```bash
python tools/backup_dbs.py                 # backs up results/ + data/benchmarks/ to D:\claims_backup\
python tools/backup_dbs.py --dest E:\backup
python tools/schedule_backup.py --interval 3
```
