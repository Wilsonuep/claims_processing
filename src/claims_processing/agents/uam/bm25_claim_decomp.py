"""
System agentowy: BM25 + dekompozycja twierdzeń (Claim Decomposition)
=====================================================================

Uproszczona wersja ``rag_claim_decomp.py`` wykorzystująca bezpośrednio
BM25 zamiast pełnego RAG (bez embeddingów, bez sqlite-vec).

Pipeline:
    Agent 1 — DECOMPOSER  → rozbija twierdzenie na pod-twierdzenia
    Agent 2 — RETRIEVER   → BM25 search per pod-twierdzenie
    Agent 3 — VERIFIER    → finalna odpowiedź na podstawie dowodów

Kompatybilność z eval_loop.py
-------------------------------
    Klasa ``ClaimDecompBM25Agent`` dziedziczy po ``BaseAgent``.

Użycie
------
    from claims_processing.agents.uam.bm25_claim_decomp import ClaimDecompBM25Agent
    from claims_processing.evaluation.eval_loop import register_agent

    register_agent(ClaimDecompBM25Agent())

Wymaga
------
    - Together AI API key (together_api_key w .env)
    - Baza SQLite z wiki_chunks (tabela wiki_chunks z tekstem)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

from claims_processing.core.llm_client import client, MODEL

from claims_processing.core.base_agent import BaseAgent

load_dotenv()

# ---------------------------------------------------------------------------
# Konfiguracja logowania
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Together AI client
# ---------------------------------------------------------------------------

client = client  # re-export from llm_client
model = MODEL

# ---------------------------------------------------------------------------
# Ścieżki do baz danych
# ---------------------------------------------------------------------------

from claims_processing.paths import bm25_wiki_db

_WIKI_DB_PATH = bm25_wiki_db()

# ---------------------------------------------------------------------------
# System prompty
# ---------------------------------------------------------------------------

DECOMPOSER_PROMPT = """\
Jesteś ekspertem od analizy twierdzeń (fact-checking).

Zadanie: Rozłóż poniższe twierdzenie na mniejsze, weryfikowalne \
pod-twierdzenia (sub-claims). Każde pod-twierdzenie powinno być \
prostym stwierdzeniem faktycznym, które można niezależnie zweryfikować.

Zasady:
- Wyodrębnij 1-5 pod-twierdzeń.
- Każde pod-twierdzenie musi być samodzielne (zrozumiałe bez kontekstu).
- Zachowaj język oryginału.
- Odpowiedz TYLKO w formacie JSON: listą stringów.
- Nie dodawaj żadnego tekstu poza JSON.

Przykład odpowiedzi:
["Kraków był stolicą Polski", "Stolica została przeniesiona w 1596 roku"]
"""

VERIFIER_PROMPT = """\
Jesteś ekspertem od weryfikacji twierdzeń (fact-checking).

Zadanie: Na podstawie oryginalnego pytania/twierdzenia, jego dekompozycji \
na pod-twierdzenia oraz dostarczonego kontekstu z Wikipedii, wybierz \
najbardziej poprawną odpowiedź.

Zasady:
- Przeanalizuj każde pod-twierdzenie w kontekście dostarczonej wiedzy.
- Jeśli kontekst potwierdza pod-twierdzenie, traktuj je jako prawdziwe.
- Jeśli kontekst zaprzecza — jako fałszywe.
- Jeśli brak kontekstu — polegaj na wiedzy ogólnej.
- Na końcu wybierz odpowiedź, która najlepiej pasuje do oryginalnego pytania.

Output: Podaj TYLKO numer odpowiedzi: 0, 1, 2 lub 3.
"""

# ---------------------------------------------------------------------------
# Pomocnicze — wywołanie LLM
# ---------------------------------------------------------------------------


def _call_llm(
    system_prompt: str,
    user_content: str,
) -> tuple[str, int, int, int]:
    """Wywołuje LLM i zwraca (odpowiedź, total, prompt, completion)."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    choice = response.choices[0]
    usage = response.usage
    return (
        choice.message.content.strip(),
        usage.total_tokens if usage else 0,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def _parse_json_list(text: str) -> list[str]:
    """Parsuje odpowiedź LLM jako listę stringów JSON."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip("`").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass

    return []


def _extract_label(answer: str) -> str:
    """Wyciąga numer odpowiedzi (0–3) z odpowiedzi LLM."""
    stripped = answer.strip()
    if stripped in ("0", "1", "2", "3"):
        return stripped

    match = re.search(r"[0-3]", stripped)
    if match:
        return match.group()

    return stripped


def _build_question_with_answers(claim_text: str, claim: dict) -> str:
    """Buduje tekst pytania z dołączonymi odpowiedziami z metadanych (AM benchmark).

    Jeśli metadata nie zawiera listy odpowiedzi, zwraca sam claim_text.
    """
    import ast
    raw_meta = claim.get("metadata") or ""
    if not raw_meta:
        return claim_text
    try:
        meta = json.loads(raw_meta)
    except Exception:
        return claim_text

    raw_answers = meta.get("answers", "")
    try:
        answers_list = ast.literal_eval(raw_answers) if isinstance(raw_answers, str) else raw_answers
    except Exception:
        answers_list = []

    if not answers_list or len(answers_list) != 4:
        return claim_text

    answers_block = "\n".join(f"{i}: {a}" for i, a in enumerate(answers_list))
    return f"{claim_text}\n\nOdpowiedzi:\n{answers_block}"


# ---------------------------------------------------------------------------
# Agent 1: DECOMPOSER
# ---------------------------------------------------------------------------


def decompose_claim(claim_text: str) -> tuple[list[str], int, int, int]:
    """Rozbija twierdzenie na pod-twierdzenia."""
    raw_answer, total, prompt, completion = _call_llm(
        DECOMPOSER_PROMPT, claim_text,
    )

    sub_claims = _parse_json_list(raw_answer)
    if not sub_claims:
        log.warning("Decomposer fallback → oryginał. Raw: %s", raw_answer[:200])
        sub_claims = [claim_text]

    log.info("Decomposer: %d pod-twierdzeń.", len(sub_claims))
    return sub_claims, total, prompt, completion


# ---------------------------------------------------------------------------
# Agent 2: RETRIEVER (BM25)
# ---------------------------------------------------------------------------

_bm25_index = None


def _get_bm25():
    """Lazy-loading indeksu BM25."""
    global _bm25_index
    if _bm25_index is None:
        from claims_processing.core.retrieval.bm25 import BM25Index
        _bm25_index = BM25Index.from_sqlite(_WIKI_DB_PATH)
    return _bm25_index


def retrieve_contexts(
    sub_claims: list[str],
    k_per_claim: int = 3,
    max_context_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Wyszukuje kontekst BM25 per pod-twierdzenie."""
    bm25 = _get_bm25()
    results: list[dict[str, Any]] = []

    for sc in sub_claims:
        context = bm25.search_and_format(
            sc, k=k_per_claim, max_context_chars=max_context_chars,
        )
        results.append({
            "sub_claim": sc,
            "context": context,
            "num_results": len(bm25.search(sc, k=k_per_claim)),
        })

    log.info("BM25 Retriever: kontekst dla %d pod-twierdzeń.", len(results))
    return results


# ---------------------------------------------------------------------------
# Agent 3: VERIFIER
# ---------------------------------------------------------------------------


def verify_claim(
    original_question: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, int, int, int]:
    """Weryfikuje twierdzenie na podstawie pod-twierdzeń z kontekstem."""
    evidence_sections: list[str] = []
    for i, e in enumerate(evidence, start=1):
        section = f"--- Pod-twierdzenie {i}: {e['sub_claim']} ---\n"
        if e["context"]:
            section += f"Kontekst z Wikipedii:\n{e['context']}\n"
        else:
            section += "Brak kontekstu w bazie wiedzy.\n"
        evidence_sections.append(section)

    user_content = (
        f"Oryginalne pytanie/twierdzenie:\n{original_question}\n\n"
        f"Dekompozycja i zebrane dowody:\n{''.join(evidence_sections)}\n\n"
        f"Na podstawie powyższych informacji wybierz odpowiedź."
    )

    answer, total, prompt, completion = _call_llm(VERIFIER_PROMPT, user_content)
    log.info("Verifier: '%s' (tokens=%d)", answer[:20], total)
    return answer, total, prompt, completion


# ---------------------------------------------------------------------------
# AGENT_CONFIG (kompatybilność z zero_shot1/2/3)
# ---------------------------------------------------------------------------

AGENT_CONFIG = {
    "name": "uam_ga4",
    "model": model,
    "system_prompt": "Multi-agent: Decomposer → BM25 Retriever → Verifier",
    "tools": ["bm25_wikipedia", "claim_decomposition"],
}

# ---------------------------------------------------------------------------
# Parametry retrieval
# ---------------------------------------------------------------------------

BM25_K_PER_CLAIM: int = 3
BM25_MAX_CONTEXT_CHARS: int = 2000


# ---------------------------------------------------------------------------
# ask() — pełny pipeline
# ---------------------------------------------------------------------------


def ask(question: str) -> dict:
    """Pełny pipeline: decompose → BM25 retrieve → verify."""
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    # === Agent 1: Dekompozycja ===
    sub_claims, t1_total, t1_prompt, t1_comp = decompose_claim(question)
    total_tokens += t1_total
    prompt_tokens += t1_prompt
    completion_tokens += t1_comp

    # === Agent 2: BM25 Retrieval ===
    evidence = retrieve_contexts(
        sub_claims,
        k_per_claim=BM25_K_PER_CLAIM,
        max_context_chars=BM25_MAX_CONTEXT_CHARS,
    )

    # === Agent 3: Weryfikacja ===
    answer, t3_total, t3_prompt, t3_comp = verify_claim(question, evidence)
    total_tokens += t3_total
    prompt_tokens += t3_prompt
    completion_tokens += t3_comp

    return {
        "answer": answer,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "sub_claims": sub_claims,
        "evidence_summary": [
            {"sub_claim": e["sub_claim"], "num_results": e["num_results"]}
            for e in evidence
        ],
    }


# ---------------------------------------------------------------------------
# BaseAgent — integracja z eval_loop.py
# ---------------------------------------------------------------------------


class ClaimDecompBM25Agent(BaseAgent):
    """Agent wieloetapowy: dekompozycja + BM25 retrieval.

    Uproszczona wersja ``ClaimDecompRAGAgent`` — bez embeddingów,
    bez sqlite-vec. Wymaga jedynie bazy SQLite z tabelą ``wiki_chunks``.
    """

    name = AGENT_CONFIG["name"]
    cost_tier = 2

    def __init__(self, model_override: str | None = None) -> None:
        from claims_processing.core.llm_client import make_client, MODEL as _DEFAULT_MODEL
        if model_override is not None:
            self._override_client, self._override_model = make_client(model_override)
            suffix = model_override.replace("/", "-").replace(":", "-")
            self.name = f"{AGENT_CONFIG['name']}__{suffix}"
            self.model_name = model_override
        else:
            self._override_client = None
            self._override_model = None
            self.model_name = _DEFAULT_MODEL

    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        if self._override_client is not None:
            import claims_processing.agents.uam.bm25_claim_decomp as _m
            _orig_client, _orig_model = _m.client, _m.model
            _m.client = self._override_client
            _m.model = self._override_model
            try:
                return self._eval_inner(claim)
            finally:
                _m.client = _orig_client
                _m.model = _orig_model
        return self._eval_inner(claim)

    def _eval_inner(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label_original", "") or claim.get("label", "")

        t0 = time.perf_counter()

        # Build question with answer choices from metadata (AM benchmark)
        question_with_answers = _build_question_with_answers(claim_text, claim)

        try:
            result = ask(question_with_answers)
        except Exception as exc:
            log.error("Błąd w pipeline: %s", exc)
            elapsed = time.perf_counter() - t0
            return {
                "model_label": "ERROR",
                "original_label": original_label,
                "is_correct": False,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "time_thought": elapsed,
                "raw_output": f"ERROR: {exc}",
                "model_name": self.model_name or "",
            }

        elapsed = time.perf_counter() - t0
        model_label = _extract_label(result["answer"])

        raw_output = json.dumps(
            {
                "answer": result["answer"],
                "sub_claims": result.get("sub_claims", []),
                "evidence_summary": result.get("evidence_summary", []),
            },
            ensure_ascii=False,
            indent=2,
        )

        return {
            "model_label": model_label,
            "original_label": original_label,
            "is_correct": str(model_label) == str(original_label),
            "total_tokens": result["total_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "time_thought": elapsed,
            "raw_output": raw_output,
            "model_name": self.model_name or "",
        }
