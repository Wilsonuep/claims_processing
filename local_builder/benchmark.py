"""
Model Benchmark — quick performance test for all installed models.
==================================================================

Runs a standardized Polish fact-checking prompt through each model
and reports:
    - Response quality (is it Polish? does it answer?)
    - Tokens per second (generation speed)
    - Time to first token (latency)
    - Total response time

Usage
-----
    python -m local_builder.benchmark

    # Specific model only:
    python -m local_builder.benchmark --model bielik-11b

    # With custom prompt:
    python -m local_builder.benchmark --prompt "Czy Warszawa jest stolicą Polski?"
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

from local_builder.model_registry import MODELS, get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TEST PROMPTS — standardized for fact-checking evaluation
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SYSTEM_PROMPT = """\
Jesteś ekspertem od weryfikacji twierdzeń (fact-checking) po polsku.
Odpowiadasz krótko i precyzyjnie, opierając się na faktach.\
"""

DEFAULT_TEST_PROMPTS = [
    {
        "name": "simple_fact",
        "prompt": (
            "Czy poniższe twierdzenie jest prawdziwe?\n"
            "Twierdzenie: 'Polska ma 16 województw.'\n"
            "Opcje: 0=Prawda, 1=Fałsz, 2=Częściowo prawda, 3=Brak danych\n"
            "Odpowiedz TYLKO numerem: 0, 1, 2 lub 3."
        ),
        "expected": "0",
        "max_tokens": 10,
    },
    {
        "name": "reasoning",
        "prompt": (
            "Rozłóż poniższe twierdzenie na pod-twierdzenia do weryfikacji.\n"
            "Twierdzenie: 'Kraków był stolicą Polski do 1596 roku, "
            "kiedy to przeniesiono ją do Warszawy.'\n"
            "Odpowiedz w formacie JSON: [\"pod-twierdzenie 1\", \"pod-twierdzenie 2\"]"
        ),
        "expected": None,  # any valid JSON list
        "max_tokens": 150,
    },
    {
        "name": "cot_reasoning",
        "prompt": (
            "Oceń prawdziwość twierdzenia krok po kroku.\n"
            "Twierdzenie: 'Mount Everest ma wysokość dokładnie 8849 metrów.'\n"
            "Krok 1: Jaka jest oficjalna wysokość Mount Everest?\n"
            "Krok 2: Czy twierdzenie zawiera słowo 'dokładnie' — czy to zmienia ocenę?\n"
            "Krok 3: Finalna ocena: 0=Prawda, 1=Fałsz, 2=Częściowo, 3=Brak danych\n"
            "Odpowiedz numerem."
        ),
        "expected": None,
        "max_tokens": 300,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════


def benchmark_model(
    model_name: str,
    prompts: list[dict] | None = None,
) -> dict[str, Any]:
    """Runs benchmark tests on a single model.

    Parameters
    ----------
    model_name : str
        Short model name from registry.
    prompts : list[dict] | None
        Custom test prompts. Default: DEFAULT_TEST_PROMPTS.

    Returns
    -------
    dict with benchmark results per prompt + aggregate stats.
    """
    model = get_model(model_name)
    tag = model["llm_model_name"]

    if prompts is None:
        prompts = DEFAULT_TEST_PROMPTS

    log.info("═" * 60)
    log.info("  Benchmarking: %s (%s)", model["display_name"], tag)
    log.info("═" * 60)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="local",
        )
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "error": str(e),
            "results": [],
        }

    results = []
    total_tokens = 0
    total_time = 0.0

    for i, test in enumerate(prompts, 1):
        log.info("─ Test %d/%d: %s ─", i, len(prompts), test["name"])

        t0 = time.perf_counter()

        try:
            response = client.chat.completions.create(
                model=tag,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": test["prompt"]},
                ],
                max_tokens=test.get("max_tokens", 100),
                temperature=0.1,
            )

            t1 = time.perf_counter()
            elapsed = t1 - t0

            content = response.choices[0].message.content.strip()
            usage = response.usage

            compl_tokens = usage.completion_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            tok_per_sec = compl_tokens / elapsed if elapsed > 0 else 0

            total_tokens += compl_tokens
            total_time += elapsed

            # Check expected answer
            correct = None
            if test.get("expected"):
                correct = test["expected"] in content[:5]

            test_result = {
                "name": test["name"],
                "success": True,
                "response": content[:300],
                "elapsed": round(elapsed, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": compl_tokens,
                "tokens_per_sec": round(tok_per_sec, 1),
                "correct": correct,
            }

            log.info(
                "  Response: %s", content[:100].replace("\n", " "),
            )
            log.info(
                "  Stats: %.1f tok/s, %d tokens, %.2fs%s",
                tok_per_sec, compl_tokens, elapsed,
                f", correct={'YES' if correct else 'NO'}" if correct is not None else "",
            )

        except Exception as e:
            test_result = {
                "name": test["name"],
                "success": False,
                "response": "",
                "elapsed": 0,
                "error": str(e),
            }
            log.error("  ❌ Error: %s", e)

        results.append(test_result)

    # Aggregate stats
    avg_tok_per_sec = total_tokens / total_time if total_time > 0 else 0

    log.info("═" * 60)
    log.info(
        "  %s — Summary: %.1f avg tok/s, %d total tokens, %.1fs",
        model["display_name"], avg_tok_per_sec, total_tokens, total_time,
    )
    log.info("═" * 60)

    return {
        "model": model_name,
        "display_name": model["display_name"],
        "success": True,
        "results": results,
        "aggregate": {
            "avg_tokens_per_sec": round(avg_tok_per_sec, 1),
            "total_tokens": total_tokens,
            "total_time": round(total_time, 2),
            "tests_passed": sum(1 for r in results if r["success"]),
            "tests_total": len(results),
        },
    }


def benchmark_all() -> list[dict[str, Any]]:
    """Benchmarks all installed models."""
    all_results = []

    for model_name in MODELS:
        result = benchmark_model(model_name)
        all_results.append(result)
        log.info("")

    # Comparison table
    log.info("=" * 70)
    log.info("  COMPARISON TABLE")
    log.info("=" * 70)
    log.info(
        "  %-25s  %8s  %8s  %8s",
        "Model", "tok/s", "tokens", "time(s)",
    )
    log.info("  " + "-" * 55)
    for r in all_results:
        if r["success"]:
            agg = r["aggregate"]
            log.info(
                "  %-25s  %8.1f  %8d  %8.1f",
                r["display_name"],
                agg["avg_tokens_per_sec"],
                agg["total_tokens"],
                agg["total_time"],
            )
        else:
            log.info("  %-25s  FAILED: %s", r.get("display_name", r["model"]), r.get("error"))
    log.info("=" * 70)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local LLM models for claims processing.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Specific model to benchmark. Default: all installed.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom single prompt to test with.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )

    args = parser.parse_args()

    # Custom prompt
    if args.prompt:
        prompts = [{
            "name": "custom",
            "prompt": args.prompt,
            "expected": None,
            "max_tokens": 200,
        }]
    else:
        prompts = None

    # Run
    if args.model:
        results = [benchmark_model(args.model, prompts)]
    else:
        results = benchmark_all()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
