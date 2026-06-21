# Running evaluations

The primary entry point is the AM benchmark wrapper. After `pip install -e .`
it is available both as a module and as the `run-am-eval` console script.

```bash
# Full run (all registered agents, local tiered mode by default)
python -m claims_processing.cli.run_eval_am_benchmark
run-am-eval                                   # equivalent console script

# Common options
run-am-eval --limit 10                        # cap claims (debugging)
run-am-eval --agents uam_ga1,uam_ga2          # subset of agents
run-am-eval --mode cloud --workers 10         # parallel cloud mode
run-am-eval --subset                          # use data/benchmarks/am_benchmark_4k.db
run-am-eval --clear                           # clear previous results first (scoped)
```

`run_eval_am_benchmark.py` uses `data/benchmarks/am_benchmark.db` as input and
writes to `results/results_am_benchmark.db`. Paths come from
`claims_processing.paths` (`AM_BENCHMARK_DB`, `AM_BENCHMARK_4K_DB`,
`RESULTS_AM_DB`).

## Direct eval-loop CLI (advanced)

The wrapper calls into the generic engine; you can invoke it directly, but you
must register agents from the calling code (the wrapper's
`_register_default_agents()` is skipped):

```bash
python -m claims_processing.evaluation.eval_loop --benchmarks am_benchmark --agents uam_ga1,uam_ga2
python -m claims_processing.evaluation.eval_loop --mode cloud --workers 10
python -m claims_processing.evaluation.eval_loop --mode local --tier2-limit 500 --tier3-limit 100
python -m claims_processing.evaluation.eval_loop --clear --export-csv   # destructive; scope with --agents
```

## Execution modes

- **sequential** — single-threaded; used by tests.
- **cloud** — `ThreadPoolExecutor(workers)`; for Together.ai / cloud APIs.
- **local** — tiered scheduling: tier-1 agents run the full dataset, tier-2 are
  capped at `--tier2-limit`, tier-3 at `--tier3-limit`. Defaults in the wrapper
  are **20000 / 20000** (effectively "no cap" against the 18,820-row benchmark);
  the direct eval-loop CLI defaults to **2000 / 500**. Pass smaller values to
  throttle tiers 2/3.

## Resume & crash recovery

On startup, `get_evaluated_claim_ids(model_name=…)` queries existing results,
deletes `model_label='ERROR'` rows, and returns already-processed claim IDs so
they are skipped. `ERROR_MAX_STEPS` rows are **not** auto-deleted — use
`python -m claims_processing.cli.fix_corrupted_results` for those.

Long AM benchmark runs are I/O- and memory-heavy and have hard-locked the
machine in the past — keep periodic backups running during a long run:

```bash
python tools/schedule_backup.py --interval 3      # backs up DBs every 3 h
```

## Distributed evaluation

Run on multiple machines, then merge result DBs:

```bash
python -m claims_processing.cli.merge_results \
    --target results/merged.db \
    --sources results/results_am_benchmark.db other_machine.db
```

## Checking completeness

```bash
python tests/eval_completeness_test.py --results-db results/merged.db
```
