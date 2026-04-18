"""
System agentowy: RAG + dekompozycja twierdzeń (Claim Decomposition)
====================================================================

Wieloetapowy pipeline agentowy składający się z 3 agentów:

    Agent 1 — DECOMPOSER
        Rozbija twierdzenie na mniejsze, weryfikowalne pod-twierdzenia.

    Agent 2 — RETRIEVER
        Dla każdego pod-twierdzenia wyszukuje kontekst z bazy wiedzy
        (Wikipedia) za pomocą RAGRetriever (BM25 / vector / hybrid).

    Agent 3 — VERIFIER
        Na podstawie oryginalnego twierdzenia, pod-twierdzeń oraz
        znalezionego kontekstu podejmuje finalną decyzję.

Flow
----
    claim_text + options
        → Agent 1: dekompozycja → lista pod-twierdzeń [JSON]
        → Agent 2: RAG retrieval → kontekst per pod-twierdzenie
        → Agent 3: weryfikacja → odpowiedź (0, 1, 2, 3)

Kompatybilność z eval_loop.py
-------------------------------
    Klasa ``ClaimDecompRAGAgent`` dziedziczy po ``BaseAgent``
    i implementuje ``eval(claim)`` — gotowa do rejestracji
    w pętli ewaluacyjnej.

Użycie
------
    from agents_uam.rag_claim_decomp import ClaimDecompRAGAgent

    agent = ClaimDecompRAGAgent()
    result = agent.eval(claim_dict)

    # Lub z eval_loop:
    from eval.eval_loop import register_agent
    register_agent(ClaimDecompRAGAgent())

Wymaga
------
    - Together AI API key (together_api_key w .env)
    - Baza SQLite z wiki_chunks (BM25)
    - Opcjonalnie: sqlite-vec + model embeddingowy (dla trybu hybrid)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

from gen_agent.llm_client import client, MODEL

from gen_agent.base_agent import BaseAgent

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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_WIKI_DB_PATH = os.getenv(
    "RAG_WIKI_DB",
    os.path.join(_PROJECT_ROOT, "dataprep", "wiki.db"),
)

# Tryb RAG: "bm25", "vector", "hybrid"
_RAG_MODE = os.getenv("RAG_MODE", "bm25")


# ---------------------------------------------------------------------------
# System prompty dla poszczególnych agentów
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
# Agent 1: DECOMPOSER — rozbija twierdzenie na pod-twierdzenia
# ---------------------------------------------------------------------------

_CALL_LLM_RETRIES = 5
_CALL_LLM_BACKOFF = [5, 10, 20, 40, 60]  # seconds between retries


def _call_llm(
    system_prompt: str,
    user_content: str,
) -> tuple[str, int, int, int]:
    """Wywołuje LLM i zwraca (odpowiedź, total_tokens, prompt_tokens, completion_tokens)."""
    import time as _time

    last_exc: Exception | None = None
    for attempt in range(_CALL_LLM_RETRIES):
        try:
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
        except (ConnectionError, OSError) as exc:
            last_exc = exc
            wait = _CALL_LLM_BACKOFF[min(attempt, len(_CALL_LLM_BACKOFF) - 1)]
            log.warning(
                "LLM connection error (attempt %d/%d): %s — retry in %ds",
                attempt + 1, _CALL_LLM_RETRIES, exc, wait,
            )
            _time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def decompose_claim(claim_text: str) -> tuple[list[str], int, int, int]:
    """Agent 1: Rozbija twierdzenie na pod-twierdzenia.

    Parametry
    ---------
    claim_text : str
        Oryginalne twierdzenie / pytanie.

    Zwraca
    ------
    (sub_claims, total_tokens, prompt_tokens, completion_tokens)
    """
    raw_answer, total, prompt, completion = _call_llm(
        DECOMPOSER_PROMPT,
        claim_text,
    )

    # Parsowanie JSON z odpowiedzi
    sub_claims = _parse_json_list(raw_answer)

    if not sub_claims:
        # Fallback: użyj oryginalnego twierdzenia jako jedynego pod-twierdzenia
        log.warning(
            "Decomposer nie zwrócił poprawnego JSON. Fallback → oryginał. "
            "Raw: %s",
            raw_answer[:200],
        )
        sub_claims = [claim_text]

    log.info(
        "Decomposer: %d pod-twierdzeń z '%s…'",
        len(sub_claims),
        claim_text[:50],
    )

    return sub_claims, total, prompt, completion


def _parse_json_list(text: str) -> list[str]:
    """Parsuje odpowiedź LLM jako listę stringów JSON.

    Obsługuje przypadki, gdy LLM otacza JSON markdownowym blokiem kodu.
    """
    # Usuń markdown code block jeśli istnieje
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip("`").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    except json.JSONDecodeError:
        pass

    # Fallback: szukaj czegokolwiek w nawiasach kwadratowych
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# Agent 2: RETRIEVER — wyszukuje kontekst dla pod-twierdzeń
# ---------------------------------------------------------------------------

_rag_retriever = None


def _get_rag():
    """Lazy-loading RAGRetriever."""
    global _rag_retriever
    if _rag_retriever is None:
        from gen_agent.rag import RAGRetriever

        rag_kwargs: dict[str, Any] = {
            "mode": _RAG_MODE,
            "bm25_db_path": _WIKI_DB_PATH,
            "text_field": "text",
            "title_field": "title",
            "section_field": "section_title",
            "id_field": "chunk_id",
        }

        # W trybie vector/hybrid potrzebujemy też bazy wektorowej
        if _RAG_MODE in ("vector", "hybrid"):
            rag_kwargs["vector_db_path"] = _WIKI_DB_PATH

        _rag_retriever = RAGRetriever(**rag_kwargs)

    return _rag_retriever


def retrieve_contexts(
    sub_claims: list[str],
    k_per_claim: int = 3,
    max_context_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Agent 2: Wyszukuje kontekst RAG dla każdego pod-twierdzenia.

    Parametry
    ---------
    sub_claims : list[str]
        Lista pod-twierdzeń.
    k_per_claim : int
        Liczba dokumentów do pobrania per pod-twierdzenie.
    max_context_chars : int
        Maksymalna długość kontekstu per pod-twierdzenie.

    Zwraca
    ------
    list[dict]
        Lista z wynikami per pod-twierdzenie::

            [
                {
                    "sub_claim": "Kraków był stolicą",
                    "context": "[1] Kraków — Historia (score: 0.45)\\n...",
                    "num_results": 3,
                },
                ...
            ]
    """
    rag = _get_rag()
    results: list[dict[str, Any]] = []

    for sc in sub_claims:
        context = rag.retrieve_and_format(
            sc, k=k_per_claim, max_context_chars=max_context_chars,
        )
        results.append({
            "sub_claim": sc,
            "context": context,
            "num_results": len(rag.retrieve(sc, k=k_per_claim)),
        })

    log.info(
        "Retriever: pobrano kontekst dla %d pod-twierdzeń.",
        len(results),
    )

    return results


# ---------------------------------------------------------------------------
# Agent 3: VERIFIER — finalna weryfikacja
# ---------------------------------------------------------------------------

def verify_claim(
    original_question: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, int, int, int]:
    """Agent 3: Weryfikuje twierdzenie na podstawie pod-twierdzeń z kontekstem.

    Parametry
    ---------
    original_question : str
        Oryginalne pytanie / twierdzenie z opcjami odpowiedzi.
    evidence : list[dict]
        Lista z pod-twierdzeniami i kontekstem (wynik retrieve_contexts).

    Zwraca
    ------
    (answer, total_tokens, prompt_tokens, completion_tokens)
    """
    # Budowanie sekcji dowodów
    evidence_sections: list[str] = []
    for i, e in enumerate(evidence, start=1):
        section = f"--- Pod-twierdzenie {i}: {e['sub_claim']} ---\n"
        if e["context"]:
            section += f"Kontekst z Wikipedii:\n{e['context']}\n"
        else:
            section += "Brak kontekstu w bazie wiedzy.\n"
        evidence_sections.append(section)

    evidence_text = "\n".join(evidence_sections)

    user_content = (
        f"Oryginalne pytanie/twierdzenie:\n{original_question}\n\n"
        f"Dekompozycja i zebrane dowody:\n{evidence_text}\n\n"
        f"Na podstawie powyższych informacji wybierz odpowiedź."
    )

    answer, total, prompt, completion = _call_llm(
        VERIFIER_PROMPT,
        user_content,
    )

    log.info(
        "Verifier: odpowiedź='%s' (tokens: total=%d, prompt=%d, completion=%d)",
        answer[:20],
        total,
        prompt,
        completion,
    )

    return answer, total, prompt, completion


# ---------------------------------------------------------------------------
# AGENT_CONFIG (kompatybilność z zero_shot1/2/3)
# ---------------------------------------------------------------------------

AGENT_CONFIG = {
    "name": "uam_ga4",
    "model": model,
    "system_prompt": "Multi-agent: Decomposer → Retriever (RAG) → Verifier",
    "tools": ["rag_hybrid", "claim_decomposition"],
}


# ---------------------------------------------------------------------------
# Funkcja ask() — pełny pipeline (kompatybilność z zero_shot1/2/3)
# ---------------------------------------------------------------------------

# Parametry retrieval
RAG_K_PER_CLAIM: int = 3
RAG_MAX_CONTEXT_CHARS: int = 2000


def ask(question: str) -> dict:
    """Pełny pipeline: decompose → retrieve → verify.

    Parametry
    ---------
    question : str
        Pytanie / twierdzenie z opcjami odpowiedzi.

    Zwraca
    ------
    dict
        Odpowiedź z metadanymi (answer, total_tokens, prompt_tokens,
        completion_tokens, sub_claims, evidence_summary).
    """
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    # === Agent 1: Dekompozycja ===
    sub_claims, t1_total, t1_prompt, t1_completion = decompose_claim(question)
    total_tokens += t1_total
    prompt_tokens += t1_prompt
    completion_tokens += t1_completion

    # === Agent 2: Retrieval ===
    evidence = retrieve_contexts(
        sub_claims,
        k_per_claim=RAG_K_PER_CLAIM,
        max_context_chars=RAG_MAX_CONTEXT_CHARS,
    )

    # === Agent 3: Weryfikacja ===
    answer, t3_total, t3_prompt, t3_completion = verify_claim(question, evidence)
    total_tokens += t3_total
    prompt_tokens += t3_prompt
    completion_tokens += t3_completion

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
# Klasa BaseAgent — integracja z eval_loop.py
# ---------------------------------------------------------------------------

class ClaimDecompRAGAgent(BaseAgent):
    """Agent wieloetapowy z dekompozycją twierdzeń i RAG retrieval.

    Kompatybilny z ``eval_loop.py`` — implementuje ``BaseAgent.eval()``.

    Pipeline:
        1. Decomposer (LLM) → pod-twierdzenia
        2. Retriever (RAG)  → kontekst per pod-twierdzenie
        3. Verifier  (LLM)  → finalna odpowiedź

    Użycie
    ------
        from eval.eval_loop import register_agent
        from agents_uam.rag_claim_decomp import ClaimDecompRAGAgent

        register_agent(ClaimDecompRAGAgent())
    """

    name = AGENT_CONFIG["name"]
    cost_tier = 2

    def __init__(self, model_override: str | None = None) -> None:
        from gen_agent.llm_client import make_client, MODEL as _DEFAULT_MODEL
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
            import agents_uam.rag_claim_decomp as _m
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
        """Ewaluacja pojedynczego twierdzenia.

        Parametry
        ---------
        claim : dict
            Słownik z co najmniej: ``claim_text``, ``label``.

        Zwraca
        ------
        dict
            Wynik z kluczami wymaganymi przez ``BaseAgent``.
        """
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

        # Ekstrakcja numerycznej odpowiedzi (0, 1, 2 lub 3)
        model_label = _extract_label(result["answer"])

        # Budowanie raw_output z pełnym trace'em pipeline'u
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


def _extract_label(answer: str) -> str:
    """Wyciąga numer odpowiedzi (0–3) z odpowiedzi LLM.

    Szuka pierwszego wystąpienia cyfry 0–3 w tekście.
    Jeśli nie znajdzie, zwraca pełną odpowiedź.
    """
    # Najprostszy przypadek: odpowiedź jest samą cyfrą
    stripped = answer.strip()
    if stripped in ("0", "1", "2", "3"):
        return stripped

    # Szukaj pierwszej cyfry 0–3
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
