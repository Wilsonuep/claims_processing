"""
Few-Shot Chain-of-Thought z Multi-Voter, RAG i Evidence Provenance
====================================================================

Zaawansowany system 6-agentowy z provenance tracking, grounded
reasoning i opcjonalnym NLI verifier:

    Agent 1 — DECOMPOSER
        Rozbija twierdzenie na pod-twierdzenia.

    Agent 2 — RETRIEVER (two-stage)
        Wyszukuje K_initial kandydatów (BM25/vector/hybrid),
        filtruje po score_threshold, ucina do K_final.
        Wyniki w ustrukturyzowanej formie z chunk_id / title / section.

    Agent 3a — REASONER (temp=0.2, konserwatywny)
    Agent 3b — REASONER (temp=0.7, zbalansowany)
    Agent 3c — REASONER (temp=1.0, kreatywny)
        Trzech niezależnych reasonerów z few-shot CoT.
        Prompt wymaga odwoływania się do kontekstu (grounded reasoning).

    Agent 4 — CONSOLIDATOR
        Ocenia argumenty trzech reasonerów z naciskiem na provenance:
        preferuje odpowiedzi ugruntowane w kontekście.

    (Opcjonalnie) VERIFIER (NLI placeholder)
        Interfejs do przyszłego NLI-based verification.

Evidence Provenance
--------------------
    Pipeline śledzi pełne metadane dokumentów (chunk_id, title,
    section, score) od retrievera przez reasonerów do finału.
    Wszystko zapisywane w raw_output do analizy.

Kompatybilność z eval_loop.py
-------------------------------
    Klasa ``FewShotCoTAgent`` dziedziczy po ``BaseAgent``.

Aligned with
-------------
    - OpenFactCheck (Wang et al., 2024)
    - RAG-augmented political fact-checking (Russo et al., 2024)
    - Provenance lightweight fact-checker for RAG
    - RAG robustness literature (best practices)

Wymaga
------
    - Together AI API key (together_api_key w .env)
    - Baza SQLite z wiki_chunks
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Callable

from dotenv import load_dotenv

from claims_processing.core.llm_client import client as _llm_client, MODEL as _LLM_MODEL

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

# ═══════════════════════════════════════════════════════════════════════════
# KONFIGURACJA EKSPERYMENTALNA (łatwa do tuningu z jednego miejsca)
# ═══════════════════════════════════════════════════════════════════════════

# --- LLM client (z llm_client.py) ---
client = _llm_client
MODEL = _LLM_MODEL

# --- Ścieżki ---
from claims_processing.paths import rag_wiki_db

_WIKI_DB_PATH = rag_wiki_db()
_RAG_MODE = os.getenv("RAG_MODE", "bm25")

# --- Retrieval (two-stage) ---
RAG_K_INITIAL: int = 20         # Stage 1: fetch this many candidates
RAG_K_PER_CLAIM: int = 5        # Stage 2: keep top K per sub-claim
RAG_SCORE_THRESHOLD: float | None = None  # None = no threshold
RAG_MAX_CONTEXT_CHARS: int = 2000

# --- Reasoner config ---
REASONER_CONFIGS: list[dict[str, Any]] = [
    {"temperature": 0.2, "label": "konserwatywny", "top_p": 0.9, "max_tokens": 1024},
    {"temperature": 0.7, "label": "zbalansowany",  "top_p": 0.95, "max_tokens": 1024},
    {"temperature": 1.0, "label": "kreatywny",     "top_p": 1.0, "max_tokens": 1024},
]

# --- Consolidator config ---
CONSOLIDATOR_TEMPERATURE: float = 0.2
CONSOLIDATOR_MAX_TOKENS: int = 1024

# --- NLI Verifier ---
NLI_ENABLED: bool = False  # set True when NLI model is available


# ═══════════════════════════════════════════════════════════════════════════
# FEW-SHOT CHAIN-OF-THOUGHT EXAMPLES (Polish, 4 labels)
# ═══════════════════════════════════════════════════════════════════════════

COT_EXAMPLES = """\
--- Przykład 1 (Prawda → 0) ---
Twierdzenie: "Wisła jest najdłuższą rzeką w Polsce."
Pod-twierdzenia: ["Wisła jest rzeką w Polsce", "Wisła jest najdłuższą rzeką w Polsce"]
Rozumowanie:
- Pod-twierdzenie 1: "Wisła jest rzeką w Polsce" → Wg kontekstu z artykułu \
"Wisła — Geografia": Wisła to rzeka w Polsce o długości 1047 km. POTWIERDZONE.
- Pod-twierdzenie 2: "Wisła jest najdłuższą rzeką w Polsce" → Wg kontekstu: \
Wisła jest najdłuższą rzeką Polski (1047 km). POTWIERDZONE.
Wniosek: Oba pod-twierdzenia potwierdzone przez kontekst.
Odpowiedź: 0

--- Przykład 2 (Fałsz → 1) ---
Twierdzenie: "Kraków jest obecnie stolicą Polski."
Pod-twierdzenia: ["Kraków jest stolicą Polski", "Jest nią obecnie"]
Rozumowanie:
- Pod-twierdzenie 1: "Kraków jest stolicą Polski" → Wg kontekstu z artykułu \
"Kraków — Historia": Kraków był stolicą do 1596 r. Obecnie stolicą jest Warszawa. ZAPRZECZONE.
- Pod-twierdzenie 2: "Jest nią obecnie" → Kontekst jasno wskazuje, że stolicą \
jest Warszawa od 1596 r. ZAPRZECZONE.
Wniosek: Twierdzenie jest fałszywe — Kraków nie jest obecną stolicą.
Odpowiedź: 1

--- Przykład 3 (Częściowo prawda → 2) ---
Twierdzenie: "Polska ma 40 milionów mieszkańców i jest największym krajem UE."
Pod-twierdzenia: ["Polska ma 40 mln mieszkańców", "Polska jest największym krajem UE"]
Rozumowanie:
- Pod-twierdzenie 1: "Polska ma 40 mln mieszkańców" → Wg kontekstu z artykułu \
"Polska — Demografia": Polska ma ok. 38 mln mieszkańców. Blisko, ale nie 40 mln. \
CZĘŚCIOWO PRAWDA.
- Pod-twierdzenie 2: "Polska jest największym krajem UE" → Wg kontekstu: \
największym krajem UE pod względem powierzchni jest Francja. ZAPRZECZONE.
Wniosek: Jedno pod-twierdzenie częściowo prawdziwe, drugie fałszywe.
Odpowiedź: 2

--- Przykład 4 (Brak danych → 3) ---
Twierdzenie: "W 2025 roku Polska będzie miała najwyższe PKB w regionie."
Pod-twierdzenia: ["Polska będzie miała najwyższe PKB w regionie", "Stanie się to w 2025"]
Rozumowanie:
- Pod-twierdzenie 1: "Polska będzie miała najwyższe PKB w regionie" → Brak \
kontekstu dotyczącego przyszłych prognoz PKB. Kontekst nie zawiera danych \
prospektywnych.
- Pod-twierdzenie 2: "Stanie się to w 2025" → Brak kontekstu do weryfikacji.
Wniosek: Kontekst nie zawiera informacji do weryfikacji tego twierdzenia.
Odpowiedź: 3
"""

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTY
# ═══════════════════════════════════════════════════════════════════════════

DECOMPOSER_PROMPT = """\
Jesteś ekspertem od analizy twierdzeń (fact-checking).

Zadanie: Rozłóż poniższe twierdzenie na mniejsze, weryfikowalne \
pod-twierdzenia (sub-claims). Każde pod-twierdzenie musi być prostym \
stwierdzeniem faktycznym, weryfikowalnym niezależnie.

Zasady:
- Wyodrębnij 1-5 pod-twierdzeń.
- Każde pod-twierdzenie musi być samodzielne (zrozumiałe bez kontekstu).
- Zachowaj język oryginału.
- Odpowiedz TYLKO w formacie JSON: listą stringów.

Przykład: ["Kraków był stolicą Polski", "Stolica przeniesiona w 1596 roku"]\
"""

REASONER_PROMPT = """\
Jesteś ekspertem od weryfikacji twierdzeń (fact-checking). Stosuj \
rozumowanie krok po kroku (chain-of-thought).

Otrzymujesz:
1. Oryginalne pytanie/twierdzenie z opcjami odpowiedzi.
2. Dekompozycję na pod-twierdzenia.
3. Kontekst z Wikipedii dla każdego pod-twierdzenia (z tytułem i sekcją źródła).

ZASADY GROUNDED REASONING:
- PREFERUJ informacje z dostarczonego kontekstu nad własną wiedzę ogólną.
- Odwołuj się do konkretnych źródeł: "Wg kontekstu z artykułu «Tytuł — Sekcja»: ..."
- Jeśli kontekst jest sprzeczny lub niewystarczający, wybierz "Częściowo prawda" \
  lub "Brak danych" zamiast zgadywać.
- Nie wymyślaj faktów, których nie ma w kontekście.
- Jeśli kontekst potwierdza — powiedz POTWIERDZONE.
- Jeśli kontekst zaprzecza — powiedz ZAPRZECZONE.
- Jeśli brak danych — powiedz BRAK KONTEKSTU.

{examples}

Format odpowiedzi:
Rozumowanie:
- [krok po kroku analiza każdego pod-twierdzenia z odwołaniem do źródeł]
Wniosek: [podsumowanie]
Odpowiedź: [0, 1, 2 lub 3]\
"""

CONSOLIDATOR_PROMPT = """\
Jesteś ekspertem od konsolidacji opinii wielu analityków fact-checkingowych.

Otrzymujesz trzy niezależne analizy tego samego twierdzenia. Każda zawiera \
rozumowanie krok po kroku i finalną odpowiedź.

KRYTERIA OCENY (Provenance-based verification):
1. PREFERUJ odpowiedzi, których rozumowanie JAWNIE odwołuje się do \
   dostarczonego kontekstu (cytuje artykuły, sekcje, lub parafrazuje kontekst).
2. ODRZUCAJ odpowiedzi, które wyglądają na halucynacje — podają fakty \
   nieobecne w kontekście.
3. Jeśli mniejszość ma lepsze ugruntowanie w kontekście niż większość — \
   PREFERUJ mniejszość.
4. Jeśli wszyscy eksperci opierają się na tych samych dowodach, ale dochodzą \
   do różnych wniosków — wybierz ten z najbardziej logicznym rozumowaniem.
5. W razie wątpliwości preferuj ostrożniejszą ocenę (Częściowo prawda / \
   Brak danych) nad pewną (Prawda / Fałsz).

{pre_analysis}

Format odpowiedzi:
Analiza:
- [ocena argumentów i grounding'u każdego eksperta]
Decyzja: [dlaczego ta odpowiedź jest najlepsza]
Odpowiedź: [0, 1, 2 lub 3]\
"""

# ═══════════════════════════════════════════════════════════════════════════
# POMOCNICZE — LLM, parsowanie
# ═══════════════════════════════════════════════════════════════════════════


def _call_llm(
    system_prompt: str,
    user_content: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 1024,
) -> tuple[str, int, int, int]:
    """Wywołuje LLM z konfigurowalnymi parametrami generacji.

    Zwraca
    ------
    (answer, total_tokens, prompt_tokens, completion_tokens)
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
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
    """Parsuje odpowiedź LLM jako listę stringów JSON.

    Obsługuje markdown code blocks i inne artefakty LLM.
    """
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
    """Wyciąga numer odpowiedzi (0–3) z odpowiedzi LLM.

    Kolejność heurystyk:
      1. Cały tekst to jedna cyfra.
      2. Pattern "Odpowiedź: X".
      3. Ostatnia cyfra 0–3 w tekście.
    """
    stripped = answer.strip()
    if stripped in ("0", "1", "2", "3"):
        return stripped

    match = re.search(r"[Oo]dpowied[źz]:\s*([0-3])", stripped)
    if match:
        return match.group(1)

    matches = re.findall(r"[0-3]", stripped)
    if matches:
        return matches[-1]

    return stripped


def _build_question_with_answers(claim_text: str, claim: dict) -> str:
    """Buduje tekst pytania z dołączonymi odpowiedziami z metadanych (AM benchmark).

    Jeśli metadata nie zawiera listy odpowiedzi, zwraca sam claim_text.
    """
    import ast
    import json as _json
    raw_meta = claim.get("metadata") or ""
    if not raw_meta:
        return claim_text
    try:
        meta = _json.loads(raw_meta)
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


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1: DECOMPOSER
# ═══════════════════════════════════════════════════════════════════════════


def decompose_claim(claim_text: str) -> tuple[list[str], int, int, int]:
    """Agent 1: Rozbija twierdzenie na pod-twierdzenia.

    Zwraca
    ------
    (sub_claims, total_tokens, prompt_tokens, completion_tokens)
    """
    raw, total, prompt, compl = _call_llm(
        DECOMPOSER_PROMPT, claim_text,
        temperature=0.3, max_tokens=512,
    )

    sub_claims = _parse_json_list(raw)
    if not sub_claims:
        log.warning("Decomposer fallback → oryginał. Raw: %s", raw[:200])
        sub_claims = [claim_text]

    log.info(
        "Agent 1 (Decomposer): %d pod-twierdzeń z '%s…'",
        len(sub_claims), claim_text[:50],
    )
    return sub_claims, total, prompt, compl


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2: RETRIEVER (two-stage, structured evidence)
# ═══════════════════════════════════════════════════════════════════════════

_rag_retriever = None


def _get_rag():
    """Lazy-loading RAGRetriever."""
    global _rag_retriever
    if _rag_retriever is None:
        from claims_processing.core.retrieval.rag import RAGRetriever

        rag_kwargs: dict[str, Any] = {
            "mode": _RAG_MODE,
            "bm25_db_path": _WIKI_DB_PATH,
            "text_field": "text",
            "title_field": "title",
            "section_field": "section_title",
            "id_field": "chunk_id",
        }
        if _RAG_MODE in ("vector", "hybrid"):
            rag_kwargs["vector_db_path"] = _WIKI_DB_PATH

        _rag_retriever = RAGRetriever(**rag_kwargs)
    return _rag_retriever


def retrieve_evidence(
    sub_claims: list[str],
    *,
    k_initial: int = RAG_K_INITIAL,
    k_final: int = RAG_K_PER_CLAIM,
    score_threshold: float | None = RAG_SCORE_THRESHOLD,
    max_context_chars: int = RAG_MAX_CONTEXT_CHARS,
) -> list[dict[str, Any]]:
    """Agent 2: Two-stage retrieval z ustrukturyzowanymi wynikami.

    Dla każdego pod-twierdzenia:
      1. Pobiera k_initial kandydatów via RAG.
      2. Filtruje po threshold, ucina do k_final.
      3. Formatuje kontekst do tekstu (zachowując strukturę).

    Parametry
    ---------
    sub_claims : list[str]
        Lista pod-twierdzeń do wyszukania.
    k_initial, k_final : int
        Parametry two-stage retrieval.
    score_threshold : float | None
        Minimalny score (None = brak filtracji).
    max_context_chars : int
        Max długość kontekstu tekstowego na pod-twierdzenie.

    Zwraca
    ------
    list[dict]
        Dla każdego pod-twierdzenia::

            {
                "sub_claim": str,
                "chunks": [                         # structured evidence
                    {"chunk_id": str, "title": str, "section": str,
                     "score": float, "text": str},
                    ...
                ],
                "context_formatted": str,            # text for LLM prompt
                "num_results": int,
            }
    """
    from claims_processing.core.retrieval.rag import RAGRetriever

    rag = _get_rag()
    results: list[dict[str, Any]] = []

    for sc in sub_claims:
        # Two-stage retrieval
        chunks = rag.retrieve_structured(
            sc,
            k_initial=k_initial,
            k_final=k_final,
            score_threshold=score_threshold,
        )

        # Format to text (from structured evidence)
        context_text = RAGRetriever.format_evidence(
            chunks, max_context_chars=max_context_chars,
        )

        results.append({
            "sub_claim": sc,
            "chunks": chunks,
            "context_formatted": context_text,
            "num_results": len(chunks),
        })

    total_chunks = sum(e["num_results"] for e in results)
    log.info(
        "Agent 2 (Retriever): %d sub-claims → %d chunks total "
        "(k_initial=%d, k_final=%d, threshold=%s).",
        len(sub_claims), total_chunks,
        k_initial, k_final, score_threshold,
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 3 (×3): REASONERS — few-shot CoT, grounded reasoning
# ═══════════════════════════════════════════════════════════════════════════


def _build_reasoner_user_prompt(
    original_question: str,
    evidence: list[dict[str, Any]],
) -> str:
    """Buduje prompt dla reasonera z kontekstem i provenance."""
    parts: list[str] = []
    for i, e in enumerate(evidence, start=1):
        part = f"=== Pod-twierdzenie {i}: {e['sub_claim']} ===\n"
        if e["context_formatted"]:
            part += f"Kontekst z Wikipedii:\n{e['context_formatted']}\n"
        else:
            part += "⚠ Brak kontekstu w bazie wiedzy dla tego pod-twierdzenia.\n"
        parts.append(part)

    return (
        f"Pytanie/twierdzenie do weryfikacji:\n{original_question}\n\n"
        f"Dekompozycja i zebrane dowody:\n\n{''.join(parts)}\n"
        f"Przeanalizuj krok po kroku i podaj odpowiedź. "
        f"Odwołuj się do kontekstu z konkretnych artykułów."
    )


def run_reasoners(
    original_question: str,
    evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Uruchamia 3 reasonerów z różnymi temperaturami i few-shot CoT.

    Każdy reasoner dostaje ten sam prompt, ale generuje z inną
    temperaturą, co daje różnorodność odpowiedzi do konsolidacji.

    Zwraca
    ------
    (outputs, total_tokens, prompt_tokens, completion_tokens)
        outputs - lista per reasoner::

            {
                "label": str,              # extracted label (0-3)
                "temperature": float,
                "top_p": float,
                "style": str,              # "konserwatywny" / ...
                "reasoning": str,          # full CoT reasoning
            }
    """
    user_prompt = _build_reasoner_user_prompt(original_question, evidence)
    system_prompt = REASONER_PROMPT.format(examples=COT_EXAMPLES)

    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    outputs: list[dict[str, Any]] = []

    for cfg in REASONER_CONFIGS:
        raw, t_total, t_prompt, t_compl = _call_llm(
            system_prompt,
            user_prompt,
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_tokens"],
        )

        extracted = _extract_label(raw)

        outputs.append({
            "label": extracted,
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "style": cfg["label"],
            "reasoning": raw,
        })

        total_tokens += t_total
        prompt_tokens += t_prompt
        completion_tokens += t_compl

        log.info(
            "Agent 3 [%s, temp=%.1f, top_p=%.2f]: label='%s' (tokens=%d)",
            cfg["label"], cfg["temperature"], cfg["top_p"], extracted, t_total,
        )

    return outputs, total_tokens, prompt_tokens, completion_tokens


# ═══════════════════════════════════════════════════════════════════════════
# PRE-CONSOLIDATION ANALYSIS (evidence grounding heuristic)
# ═══════════════════════════════════════════════════════════════════════════


def _compute_evidence_grounding(
    reasoner_outputs: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> str:
    """Heurystyczna analiza ugruntowania odpowiedzi w kontekście.

    Liczy ile termów z tytułów i sekcji artykułów pojawia się
    w rozumowaniu każdego reasonera. Wynik dodawany do promptu
    consolidatora jako podpowiedź.

    Zwraca
    ------
    str
        Tekst pre-analizy do wstrzyknięcia w prompt consolidatora.
        Pusty string jeśli wszyscy zgodni.
    """
    labels = [r["label"] for r in reasoner_outputs]
    if len(set(labels)) == 1:
        return ""

    # Zbierz termy z evidence titles/sections
    evidence_terms: set[str] = set()
    for e in evidence:
        for chunk in e.get("chunks", []):
            title = chunk.get("title", "").lower()
            section = chunk.get("section", "").lower()
            for word in re.split(r"\W+", f"{title} {section}"):
                if len(word) > 3:
                    evidence_terms.add(word)

    if not evidence_terms:
        return ""

    # Policz grounding score per reasoner
    grounding_scores: list[tuple[str, int, str]] = []
    for r in reasoner_outputs:
        reasoning_lower = r["reasoning"].lower()
        count = sum(1 for t in evidence_terms if t in reasoning_lower)
        grounding_scores.append((r["style"], count, r["label"]))

    # Build pre-analysis text
    lines = ["Wstępna analiza grounding'u (automatyczna):"]
    for style, score, label in grounding_scores:
        lines.append(
            f"  - Ekspert {style}: {score} odwołań do artykułów "
            f"z kontekstu, odpowiedź: {label}"
        )

    # Identify majority vs minority
    label_counts = Counter(labels)
    majority_label, majority_count = label_counts.most_common(1)[0]
    if majority_count >= 2:
        lines.append(
            f"  Większość ({majority_count}/3) wybrała: {majority_label}."
        )

        # Check if minority is better grounded
        for style, score, label in grounding_scores:
            if label != majority_label:
                majority_scores = [
                    s for st, s, l in grounding_scores if l == majority_label
                ]
                avg_majority = sum(majority_scores) / len(majority_scores)
                if score > avg_majority * 1.3:  # minority 30%+ better grounded
                    lines.append(
                        f"  ⚠ Ekspert {style} (mniejszość) ma wyższy "
                        f"grounding ({score}) niż średnia większości "
                        f"({avg_majority:.0f}). Rozważ jakość argumentów."
                    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 4: CONSOLIDATOR
# ═══════════════════════════════════════════════════════════════════════════


def consolidate(
    original_question: str,
    reasoner_outputs: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> tuple[str, str, int, int, int]:
    """Konsoliduje odpowiedzi trzech reasonerów.

    Fast path: jeśli wszyscy zgodni → pomija LLM.
    Otherwise: provenance-aware consolidation z pre-analizą grounding'u.

    Zwraca
    ------
    (final_label, full_reasoning, total_tokens, prompt_tokens, completion_tokens)
    """
    labels = [r["label"] for r in reasoner_outputs]
    label_counts = Counter(labels)

    # --- Fast path: unanimity ---
    if len(set(labels)) == 1:
        log.info(
            "Agent 4 (Consolidator): unanimity → '%s' (skip LLM).", labels[0],
        )
        return (
            labels[0],
            f"Unanimity: wszyscy 3 eksperci odpowiedzieli {labels[0]}.",
            0, 0, 0,
        )

    log.info(
        "Agent 4 (Consolidator): disagreement %s → wywołuję LLM.",
        dict(label_counts),
    )

    # --- Pre-analysis ---
    pre_analysis = _compute_evidence_grounding(reasoner_outputs, evidence)

    # --- Build prompt ---
    expert_sections: list[str] = []
    for i, r in enumerate(reasoner_outputs, start=1):
        expert_sections.append(
            f"=== Ekspert {i} ({r['style']}, temp={r['temperature']}, "
            f"top_p={r['top_p']}) ===\n"
            f"Odpowiedź: {r['label']}\n"
            f"Rozumowanie:\n{r['reasoning']}\n"
        )

    user_content = (
        f"Oryginalne pytanie:\n{original_question}\n\n"
        f"Analizy ekspertów:\n\n{''.join(expert_sections)}\n"
        f"Oceń argumenty i jakość grounding'u, podaj finalną odpowiedź."
    )

    system_prompt = CONSOLIDATOR_PROMPT.format(
        pre_analysis=pre_analysis if pre_analysis else "",
    )

    raw, total, prompt, compl = _call_llm(
        system_prompt,
        user_content,
        temperature=CONSOLIDATOR_TEMPERATURE,
        max_tokens=CONSOLIDATOR_MAX_TOKENS,
    )

    final = _extract_label(raw)
    log.info(
        "Agent 4 (Consolidator): final='%s' (votes=%s, tokens=%d)",
        final, dict(label_counts), total,
    )

    return final, raw, total, prompt, compl


# ═══════════════════════════════════════════════════════════════════════════
# NLI VERIFIER PLACEHOLDER (opcjonalny)
# ═══════════════════════════════════════════════════════════════════════════


class NLIVerifier:
    """Placeholder interface for NLI-based verification.

    Sprawdza czy finalna odpowiedź i wyjaśnienie są wspierane (entailed)
    przez dostarczony kontekst. Do podłączenia w przyszłości konkretnego
    modelu NLI (np. cross-encoder/nli-distilroberta-base lub
    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

    Użycie
    ------
        verifier = NLIVerifier()
        score = verifier.verify(
            context="Kraków był stolicą do 1596 roku...",
            explanation="Wg kontekstu Kraków nie jest stolicą.",
            label="1",
        )
        # score ∈ [0.0, 1.0] — stopień entailment

    Integration point
    -----------------
        W pipeline wywoływany po consolidator, przed zwróceniem wyniku.
        Jeśli score < threshold → można flagować wynik jako "low confidence".
    """

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Inicjalizacja.

        Parametry
        ---------
        model_name : str | None
            Nazwa modelu NLI z HuggingFace Hub. None = dummy (zawsze 1.0).
        threshold : float
            Minimalny score poniżej którego wynik jest "low confidence".
        """
        self.model_name = model_name
        self.threshold = threshold
        self._model = None  # lazy-loading

    def verify(
        self,
        context: str,
        explanation: str,
        label: str,
    ) -> float:
        """Sprawdza entailment między kontekstem a wyjaśnieniem.

        Parametry
        ---------
        context : str
            Kontekst z bazy wiedzy (concatenated evidence).
        explanation : str
            Wyjaśnienie/rozumowanie modelu (reasoning z consolidatora).
        label : str
            Label predykcji (0-3).

        Zwraca
        ------
        float
            Entailment score ∈ [0.0, 1.0].
            1.0 = fully entailed, 0.0 = contradiction.
        """
        if self._model is None and self.model_name is not None:
            self._load_model()

        if self._model is None:
            # Dummy: brak modelu → zawsze 1.0
            return 1.0

        # --- Tutaj podłącz konkretny model NLI ---
        # premise = context
        # hypothesis = f"Odpowiedź to {label}. {explanation}"
        # score = self._model.predict([(premise, hypothesis)])[0]
        # return float(score)

        return 1.0  # placeholder

    def _load_model(self) -> None:
        """Lazy-loading modelu NLI."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            log.info("NLI model loaded: %s", self.model_name)
        except ImportError:
            log.warning(
                "sentence-transformers not available. NLI verifier disabled."
            )
            self._model = None
        except Exception as exc:
            log.warning("NLI model load failed: %s", exc)
            self._model = None

    def is_confident(self, score: float) -> bool:
        """Czy score jest powyżej threshold (= sufficient confidence)."""
        return score >= self.threshold


# Singleton NLI verifier
_nli_verifier: NLIVerifier | None = None


def _get_nli_verifier() -> NLIVerifier:
    """Lazy-loading NLI verifier."""
    global _nli_verifier
    if _nli_verifier is None:
        _nli_verifier = NLIVerifier()  # dummy by default
    return _nli_verifier


# ═══════════════════════════════════════════════════════════════════════════
# AGENT_CONFIG
# ═══════════════════════════════════════════════════════════════════════════

AGENT_CONFIG = {
    "name": "uam_ga5",
    "model": MODEL,
    "system_prompt": "Few-Shot CoT: Decomposer → RAG (two-stage) → 3×Reasoner → Consolidator",
    "tools": ["rag_two_stage", "claim_decomposition", "cot_multi_voter"],
}


# ═══════════════════════════════════════════════════════════════════════════
# ask() — PEŁNY PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def ask(question: str) -> dict[str, Any]:
    """Pełny pipeline: decompose → retrieve → 3× reason → consolidate.

    Zwraca
    ------
    dict
        Klucze:
        - answer: str — final label
        - total_tokens, prompt_tokens, completion_tokens: int
        - sub_claims: list[str]
        - evidence: list[dict] — structured evidence per sub-claim
        - reasoner_votes: list[dict] — per reasoner
        - consolidation: str — consolidator reasoning
        - nli_score: float — NLI confidence (placeholder)
    """
    total_tok = 0
    prompt_tok = 0
    compl_tok = 0

    # ═══ Agent 1: Dekompozycja ═══
    sub_claims, t1, p1, c1 = decompose_claim(question)
    total_tok += t1
    prompt_tok += p1
    compl_tok += c1

    # ═══ Agent 2: Two-stage RAG Retrieval ═══
    evidence = retrieve_evidence(sub_claims)

    # ═══ Agent 3 (×3): Chain-of-Thought Reasoning ═══
    reasoner_outputs, t3, p3, c3 = run_reasoners(question, evidence)
    total_tok += t3
    prompt_tok += p3
    compl_tok += c3

    # ═══ Agent 4: Konsolidacja ═══
    final_answer, consolidation_reasoning, t4, p4, c4 = consolidate(
        question, reasoner_outputs, evidence,
    )
    total_tok += t4
    prompt_tok += p4
    compl_tok += c4

    # ═══ (Opcjonalnie) NLI Verification ═══
    nli_score = 1.0
    if NLI_ENABLED:
        all_context = "\n".join(
            e["context_formatted"] for e in evidence if e["context_formatted"]
        )
        nli_score = _get_nli_verifier().verify(
            context=all_context,
            explanation=consolidation_reasoning,
            label=final_answer,
        )
        log.info("NLI Verifier: score=%.3f", nli_score)

    # ═══ Structured evidence for output ═══
    evidence_output: list[dict[str, Any]] = []
    for e in evidence:
        evidence_output.append({
            "sub_claim": e["sub_claim"],
            "chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "title": c["title"],
                    "section": c["section"],
                    "score": c["score"],
                }
                for c in e["chunks"]
            ],
        })

    return {
        "answer": final_answer,
        "total_tokens": total_tok,
        "prompt_tokens": prompt_tok,
        "completion_tokens": compl_tok,
        "sub_claims": sub_claims,
        "evidence": evidence_output,
        "reasoner_votes": [
            {
                "style": r["style"],
                "temp": r["temperature"],
                "top_p": r["top_p"],
                "label": r["label"],
            }
            for r in reasoner_outputs
        ],
        "consolidation": consolidation_reasoning,
        "nli_score": nli_score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BaseAgent — INTEGRACJA Z eval_loop.py
# ═══════════════════════════════════════════════════════════════════════════


class FewShotCoTAgent(BaseAgent):
    """Few-Shot Chain-of-Thought z provenance tracking.

    Pipeline 6-agentowy:
        Decomposer → Retriever (two-stage RAG) → 3× Reasoner (CoT)
        → Consolidator (provenance-aware)

    Evidence provenance propagowane od retrievera do finału.
    raw_output zawiera pełny trace z chunk_ids, votes, i NLI score.
    """

    name = AGENT_CONFIG["name"]
    cost_tier = 3

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
            import claims_processing.agents.uam.fewshot_cot_rag as _m
            _orig_client, _orig_model = _m.client, _m.MODEL
            _m.client = self._override_client
            _m.MODEL = self._override_model
            try:
                return self._eval_inner(claim)
            finally:
                _m.client = _orig_client
                _m.MODEL = _orig_model
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
            Wynik z kluczami wymaganymi przez ``BaseAgent``::

                model_label, original_label, is_correct,
                total_tokens, prompt_tokens, completion_tokens,
                time_thought, raw_output
        """
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label_original", "") or claim.get("label", "")

        t0 = time.perf_counter()

        # Build question with answer choices from metadata (AM benchmark)
        question_with_answers = _build_question_with_answers(claim_text, claim)

        try:
            result = ask(question_with_answers)
        except Exception as exc:
            log.error("Pipeline error: %s", exc, exc_info=True)
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
                "evidence": result.get("evidence", []),
                "reasoner_votes": result.get("reasoner_votes", []),
                "consolidation": result.get("consolidation", ""),
                "nli_score": result.get("nli_score", 1.0),
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
