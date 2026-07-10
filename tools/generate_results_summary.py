#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate docs/results_summary.md — agent × model summary tables.

Reads both result DBs (full benchmark + fair 4k subsample), excludes the
archived agents (uam_ga_web_tool_arch, uam_ga_rag_decomp_arch), and writes
markdown pivot tables: accuracy, mean total tokens, mean inference time and
row coverage. Model labels follow the shortened-Ollama-tag convention used
in the notebooks.

    python tools/generate_results_summary.py
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from claims_processing import paths

OUTPUT = paths.PROJECT_ROOT / "docs" / "results_summary.md"

# Shortened Ollama-tag labels (mirrors _short_model() in the notebooks).
MODEL_LABELS = {
    "hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M": "Bielik-11B-v2.3",
    "hf.co/mradermacher/Llama-PLLuM-8B-instruct-GGUF:Q4_K_M": "Llama-PLLuM-8B",
    "llama3.1:8b": "llama3.1:8b",
    "qwen2.5:7b": "qwen2.5:7b",
}
MODEL_ORDER = list(MODEL_LABELS.values())

AGENT_LABELS = {
    "uam_ga1": "ga1 Single (zero-shot)",
    "uam_ga2": "ga2 Single + BM25",
    "uam_ga3": "ga3 Decomp + BM25",
    "uam_ga4": "ga4 FewShot CoT",
    "uam_ga5": "ga5 Debate",
}
AGENT_ORDER = list(AGENT_LABELS.values())

QUERY = """
    SELECT agent_name, model_name, is_correct, total_tokens, time_thought
    FROM agent_results
    WHERE agent_name NOT LIKE 'uam_ga_web_tool_arch%'
      AND agent_name NOT LIKE 'uam_ga_rag_decomp_arch%'
"""


def load(db: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(QUERY, conn)
    finally:
        conn.close()
    df["agent"] = (
        df["agent_name"].str.extract(r"^(uam_ga\d+)")[0].map(AGENT_LABELS)
    )
    df["model"] = df["model_name"].map(MODEL_LABELS)
    return df


def pivot(df: pd.DataFrame, value: str, agg: str, fmt: str) -> str:
    p = df.pivot_table(value, "agent", "model", aggfunc=agg)
    p = p.reindex(index=AGENT_ORDER, columns=MODEL_ORDER)
    p.index.name = "Agent"
    return p.map(lambda v: "—" if pd.isna(v) else format(v, fmt)).to_markdown()


def section(title: str, df: pd.DataFrame, note: str) -> str:
    acc = df.assign(acc=df["is_correct"] * 100)
    return "\n".join([
        f"## {title}",
        "",
        note,
        "",
        "### Dokładność (%)",
        "",
        pivot(acc, "acc", "mean", ".1f"),
        "",
        "### Średnia liczba tokenów (total) na pytanie",
        "",
        pivot(df, "total_tokens", "mean", ",.0f"),
        "",
        "### Średni czas inferencji na pytanie (s)",
        "",
        pivot(df, "time_thought", "mean", ".1f"),
        "",
        "### Pokrycie (liczba wierszy wyników)",
        "",
        pivot(df, "is_correct", "count", ",.0f"),
        "",
    ])


def main() -> None:
    full = load(paths.RESULTS_AM_DB)
    sub = load(paths.RESULTS_AM_SUBSAMPLE_DB)

    doc = "\n".join([
        "# Podsumowanie wyników — AM benchmark (agenci × modele)",
        "",
        f"Wygenerowano {date.today().isoformat()} przez "
        "`python tools/generate_results_summary.py`. Nie edytować ręcznie.",
        "",
        "Aktywni agenci **uam_ga1–uam_ga5**; zarchiwizowani "
        "(`uam_ga_web_tool_arch`, `uam_ga_rag_decomp_arch`) są wykluczeni. "
        "Wszyscy agenci z retrievalem (ga2–ga5) używają BM25 — `RAG_MODE` "
        "nigdy nie był ustawiony na vector/hybrid. Etykiety modeli w stylu "
        "skróconych tagów Ollama.",
        "",
        section(
            "Pełny benchmark (`results/results_am_benchmark.db`)",
            full,
            "Bielik i llama3.1 pokrywają pełne 18 820 pytań; qwen2.5 i PLLuM "
            "tylko podzbiór ~4 000 pytań (seed 42) — kolumn nie należy "
            "porównywać 1:1 w tej tabeli.",
        ),
        section(
            "Wspólny podzbiór 4k (`results/results_am_subsample.db`)",
            sub,
            "Sprawiedliwe porównanie 1:1 — te same 4 000 pytań dla każdej "
            "pary agent × model.",
        ),
    ])

    OUTPUT.write_text(doc, encoding="utf-8")
    print(f"written {OUTPUT}")


if __name__ == "__main__":
    main()
