"""
Few-Shot Chain-of-Thought z Debate i RAG
==========================================

System wieloagentowy z debatą adversarialną:

    Agent 1 — DECOMPOSER (reuse)
        Rozbija twierdzenie na pod-twierdzenia.

    Agent 2 — RETRIEVER (reuse)
        Two-stage RAG retrieval z provenance tracking.

    Agent 3a — DEBATER-PROPONENT
        Argumentuje ZA prawdziwością twierdzenia (grounded w evidence).

    Agent 3b — DEBATER-OPPONENT
        Argumentuje PRZECIW (grounded w evidence).

    (Hook) Agent 3c — DEBATER-CRITIC (neutral, przyszła rozbudowa)

    Agent 4 — JUDGE
        Ocenia transcript debaty, jakość argumentów, grounding
        w evidence. Podejmuje finalną decyzję.

Debate flow (3 rundy)
----------------------
    Opening   → każdy debater prezentuje argument + provisional label
    Rebuttal  → każdy debater odpowiada na argument drugiego
    Closing   → każdy debater podsumowuje + final label

Kompatybilność z eval_loop.py
-------------------------------
    Klasa ``DebateCoTAgent`` dziedziczy po ``BaseAgent``.

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
from typing import Any

from dotenv import load_dotenv

from claims_processing.core.base_agent import BaseAgent

# Reuse shared primitives from fewshot_cot_rag
from claims_processing.agents.uam.fewshot_cot_rag import (
    _call_llm,
    _extract_label,
    _build_question_with_answers,
    _parse_json_list,
    decompose_claim,
    retrieve_evidence,
    # Config & constants reused for retrieval:
    RAG_K_INITIAL,
    RAG_K_PER_CLAIM,
    RAG_SCORE_THRESHOLD,
    RAG_MAX_CONTEXT_CHARS,
)

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
# KONFIGURACJA DEBATY
# ═══════════════════════════════════════════════════════════════════════════

# --- Debate rounds ---
DEBATE_NUM_ROUNDS: int = 3      # Opening, Rebuttal, Closing
DEBATE_ROUND_NAMES: list[str] = ["Opening", "Rebuttal", "Closing"]

# --- Max tokens per round per debater ---
DEBATE_OPENING_MAX_TOKENS: int = 768
DEBATE_REBUTTAL_MAX_TOKENS: int = 768
DEBATE_CLOSING_MAX_TOKENS: int = 512

DEBATE_ROUND_MAX_TOKENS: dict[str, int] = {
    "Opening": DEBATE_OPENING_MAX_TOKENS,
    "Rebuttal": DEBATE_REBUTTAL_MAX_TOKENS,
    "Closing": DEBATE_CLOSING_MAX_TOKENS,
}

# --- Temperatures per role ---
DEBATE_TEMPERATURES: dict[str, float] = {
    "proponent": 0.5,
    "opponent": 0.5,
}

DEBATE_TOP_P: float = 0.95

# --- Judge ---
JUDGE_TEMPERATURE: float = 0.2
JUDGE_MAX_TOKENS: int = 1024


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTY — DEBATERS
# ═══════════════════════════════════════════════════════════════════════════

DEBATER_PROPONENT_PROMPT = """\
Jesteś ekspertem fact-checkingowym w roli PROPONENTA (obrońcy twierdzenia).

Twoje zadanie: argumentować, że dane twierdzenie jest PRAWDZIWE lub \
CZĘŚCIOWO PRAWDZIWE, opierając się na dostarczonym kontekście z Wikipedii.

ZASADY GROUNDED REASONING:
- OPIERAJ się wyłącznie na dostarczonym kontekście z Wikipedii.
- Odwołuj się do konkretnych artykułów: "Wg artykułu «Tytuł — Sekcja»: ..."
- NIE wymyślaj faktów nieobecnych w kontekście — to dyskwalifikuje argument.
- Jeśli kontekst nie wspiera twierdzenia, przyznaj to uczciwie, ale szukaj \
  aspektów częściowo prawdziwych.
- Jeśli kontekst jest niewystarczający, argumentuj za "Brak danych" \
  zamiast wymyślać poparcie.

KONTEKST DEBATY:
{debate_context}

Format odpowiedzi:
Argument:
- [punkt po punkcie argumentacja z odwołaniami do źródeł]
Konkluzja: [podsumowanie stanowiska]
Proponowana odpowiedź: [0, 1, 2 lub 3]\
"""

DEBATER_OPPONENT_PROMPT = """\
Jesteś ekspertem fact-checkingowym w roli OPONENTA (krytyka twierdzenia).

Twoje zadanie: argumentować, że dane twierdzenie jest FAŁSZYWE, \
CZĘŚCIOWO PRAWDZIWE lub NIEWERYFIKOWALNE, opierając się na dostarczonym \
kontekście z Wikipedii.

ZASADY GROUNDED REASONING:
- OPIERAJ się wyłącznie na dostarczonym kontekście z Wikipedii.
- Odwołuj się do konkretnych artykułów: "Wg artykułu «Tytuł — Sekcja»: ..."
- NIE wymyślaj faktów nieobecnych w kontekście — to dyskwalifikuje argument.
- Szukaj rozbieżności, nieścisłości, brakujących danych.
- Jeśli kontekst potwierdza twierdzenie, przyznaj to uczciwie, ale zwróć \
  uwagę na niuanse lub niedokładności.
- Jeśli kontekst jest niewystarczający, argumentuj za "Brak danych".

KONTEKST DEBATY:
{debate_context}

Format odpowiedzi:
Argument:
- [punkt po punkcie argumentacja z odwołaniami do źródeł]
Konkluzja: [podsumowanie stanowiska]
Proponowana odpowiedź: [0, 1, 2 lub 3]\
"""

DEBATER_PROMPTS: dict[str, str] = {
    "proponent": DEBATER_PROPONENT_PROMPT,
    "opponent": DEBATER_OPPONENT_PROMPT,
}


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — JUDGE
# ═══════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """\
Jesteś sędzią debaty fact-checkingowej. Otrzymujesz pełny transcript \
debaty między proponentem (obrońcą) a oponentem (krytykiem) twierdzenia.

Twoje zadanie: na podstawie argumentów obu stron oraz dostarczonego \
kontekstu z Wikipedii, podejmij finalną decyzję o prawdziwości twierdzenia.

KRYTERIA OCENY:
1. PREFERUJ argumenty ugruntowane w dostarczonym kontekście (cytujące \
   artykuły, sekcje, lub parafrazujące konkretne fragmenty).
2. ODRZUCAJ argumenty oparte na halucynacjach — faktach nieobecnych \
   w kontekście.
3. OCEŃ logiczność rozumowania — nawet dobrze ugruntowany argument \
   może zawierać błędy wnioskowania.
4. Jeśli obie strony mają równie silne argumenty, preferuj ostrożniejszą \
   ocenę (Częściowo prawda / Brak danych).
5. Jeśli debater mniejszościowy ma lepsze ugruntowanie w kontekście, \
   PREFERUJ jego stanowisko.
6. Weź pod uwagę, jak debaterzy reagowali na kontrargumenty w rundzie \
   Rebuttal — skuteczne odpieranie zarzutów wzmacnia wiarygodność.

Format odpowiedzi:
Ocena proponenta:
- [mocne i słabe strony argumentacji]
Ocena oponenta:
- [mocne i słabe strony argumentacji]
Werdykt: [uzasadnienie decyzji]
Odpowiedź: [0, 1, 2 lub 3]\
"""


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: BUILD EVIDENCE CONTEXT FOR DEBATERS
# ═══════════════════════════════════════════════════════════════════════════


def _build_evidence_summary(
    original_question: str,
    sub_claims: list[str],
    evidence: list[dict[str, Any]],
) -> str:
    """Builds the evidence context block shared by all debaters.

    Includes the original question, sub-claims, and formatted
    Wikipedia context for each sub-claim.
    """
    parts: list[str] = []
    parts.append(f"Twierdzenie do weryfikacji:\n{original_question}\n")
    parts.append(f"Pod-twierdzenia: {json.dumps(sub_claims, ensure_ascii=False)}\n")

    for i, e in enumerate(evidence, start=1):
        section = f"=== Pod-twierdzenie {i}: {e['sub_claim']} ===\n"
        if e.get("context_formatted"):
            section += f"Kontekst z Wikipedii:\n{e['context_formatted']}\n"
        else:
            section += "⚠ Brak kontekstu w bazie wiedzy.\n"
        parts.append(section)

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# DEBATER EXECUTION
# ═══════════════════════════════════════════════════════════════════════════


def _build_debater_prompt(
    role: str,
    evidence_summary: str,
    debate_history: str,
    round_name: str,
) -> tuple[str, str]:
    """Builds system and user prompts for a debater in a given round.

    Returns
    -------
    (system_prompt, user_prompt)
    """
    base_prompt = DEBATER_PROMPTS[role]

    # Context varies by round
    if round_name == "Opening":
        debate_context = "To jest runda otwarcia. Zaprezentuj swój argument."
    elif round_name == "Rebuttal":
        debate_context = (
            "To jest runda odpowiedzi. Przeanalizuj argument przeciwnika "
            "i odpowiedz na jego punkty.\n\n"
            f"Dotychczasowa debata:\n{debate_history}"
        )
    elif round_name == "Closing":
        debate_context = (
            "To jest runda zamknięcia. Podsumuj debatę i podaj finalną "
            "odpowiedź, uwzględniając argumenty obu stron.\n\n"
            f"Pełna debata:\n{debate_history}"
        )
    else:
        debate_context = debate_history

    system_prompt = base_prompt.format(debate_context=debate_context)
    user_prompt = evidence_summary

    return system_prompt, user_prompt


def run_debater(
    role: str,
    evidence_summary: str,
    debate_history: str,
    round_name: str,
) -> tuple[dict[str, Any], int, int, int]:
    """Runs a single debater for a given round.

    Parameters
    ----------
    role : str
        "proponent", "opponent", or "critic".
    evidence_summary : str
        Pre-built evidence context.
    debate_history : str
        Accumulated debate transcript so far.
    round_name : str
        "Opening", "Rebuttal", or "Closing".

    Returns
    -------
    (output_dict, total_tokens, prompt_tokens, completion_tokens)
        output_dict contains: role, round, label, argument.
    """
    system_prompt, user_prompt = _build_debater_prompt(
        role, evidence_summary, debate_history, round_name,
    )

    temperature = DEBATE_TEMPERATURES.get(role, 0.5)
    max_tokens = DEBATE_ROUND_MAX_TOKENS.get(round_name, 768)

    raw, total, prompt, compl = _call_llm(
        system_prompt,
        user_prompt,
        temperature=temperature,
        top_p=DEBATE_TOP_P,
        max_tokens=max_tokens,
    )

    label = _extract_label(raw)

    log.info(
        "Debater [%s, %s]: label='%s' (tokens=%d)",
        role, round_name, label, total,
    )

    output = {
        "role": role,
        "round": round_name,
        "label": label,
        "argument": raw,
    }

    return output, total, prompt, compl


# ═══════════════════════════════════════════════════════════════════════════
# DEBATE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════


def run_debate_rounds(
    original_question: str,
    sub_claims: list[str],
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, int, int]:
    """Orchestrates the full debate: Opening → Rebuttal → Closing.

    Parameters
    ----------
    original_question : str
        Original claim/question.
    sub_claims : list[str]
        Decomposed sub-claims.
    evidence : list[dict]
        Structured evidence from retriever.

    Returns
    -------
    (debate_result, total_tokens, prompt_tokens, completion_tokens)
        debate_result contains:
        - transcript: str (full human-readable debate transcript)
        - rounds: list[dict] (structured per-round data)
        - final_labels: {"proponent": str, "opponent": str}
    """
    evidence_summary = _build_evidence_summary(
        original_question, sub_claims, evidence,
    )

    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    rounds_data: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    debate_history = ""

    # Active debater roles (hook: add "critic" here to enable)
    active_roles = ["proponent", "opponent"]

    for round_name in DEBATE_ROUND_NAMES[:DEBATE_NUM_ROUNDS]:
        log.info("─── Debate Round: %s ───", round_name)

        round_outputs: list[dict[str, Any]] = []

        for role in active_roles:
            try:
                output, t_total, t_prompt, t_compl = run_debater(
                    role=role,
                    evidence_summary=evidence_summary,
                    debate_history=debate_history,
                    round_name=round_name,
                )
            except Exception as debater_exc:
                log.warning(
                    "Debater [%s, %s] failed (%s) — using NIEWERYFIKOWALNE fallback.",
                    role, round_name, debater_exc,
                )
                output = {
                    "role": role,
                    "round": round_name,
                    "label": "NIEWERYFIKOWALNE",
                    "argument": f"[BŁĄD DEBATORA: {debater_exc}]",
                }
                t_total = t_prompt = t_compl = 0

            round_outputs.append(output)
            total_tokens += t_total
            prompt_tokens += t_prompt
            completion_tokens += t_compl

            # Append to transcript
            role_label = "PROPONENT" if role == "proponent" else "OPONENT"
            entry = (
                f"── {role_label} ({round_name}) ──\n"
                f"{output['argument']}\n"
            )
            transcript_parts.append(entry)

        # Update debate history for next round
        debate_history = "\n".join(transcript_parts)

        rounds_data.append({
            "round": round_name,
            "outputs": round_outputs,
        })

    # Extract final labels (from Closing round, or last available)
    final_labels: dict[str, str] = {}
    if rounds_data:
        last_round = rounds_data[-1]
        for output in last_round["outputs"]:
            final_labels[output["role"]] = output["label"]

    transcript = "\n".join(transcript_parts)

    log.info(
        "Debate complete: %d rounds, final_labels=%s, total_tokens=%d.",
        len(rounds_data), final_labels, total_tokens,
    )

    debate_result = {
        "transcript": transcript,
        "rounds": rounds_data,
        "final_labels": final_labels,
    }

    return debate_result, total_tokens, prompt_tokens, completion_tokens


# ═══════════════════════════════════════════════════════════════════════════
# JUDGE
# ═══════════════════════════════════════════════════════════════════════════


def judge_debate(
    original_question: str,
    evidence: list[dict[str, Any]],
    debate_transcript: str,
    final_labels: dict[str, str],
) -> tuple[str, str, int, int, int]:
    """Judge evaluates the debate and produces a final verdict.

    Parameters
    ----------
    original_question : str
        Original claim.
    evidence : list[dict]
        Structured evidence from retriever.
    debate_transcript : str
        Full debate transcript (all rounds).
    final_labels : dict
        Final labels from each debater, e.g. {"proponent": "0", "opponent": "1"}.

    Returns
    -------
    (final_label, judge_reasoning, total_tokens, prompt_tokens, completion_tokens)
    """
    # Fast path: if both debaters agree
    labels = list(final_labels.values())
    if len(set(labels)) == 1:
        agreed = labels[0]
        log.info("Judge: debaters agree → '%s' (skip LLM).", agreed)
        return (
            agreed,
            f"Consensus: obaj debaterzy odpowiedzieli {agreed}.",
            0, 0, 0,
        )

    log.info("Judge: disagreement %s → wywołuję LLM.", final_labels)

    # Build evidence summary for judge
    evidence_context_parts: list[str] = []
    for e in evidence:
        if e.get("context_formatted"):
            evidence_context_parts.append(
                f"Pod-twierdzenie: {e['sub_claim']}\n"
                f"{e['context_formatted']}\n"
            )

    evidence_text = "\n".join(evidence_context_parts) if evidence_context_parts else "(brak)"

    # Build user prompt
    labels_text = "\n".join(
        f"  - {role.upper()}: {label}" for role, label in final_labels.items()
    )

    user_content = (
        f"Oryginalne twierdzenie:\n{original_question}\n\n"
        f"Kontekst z Wikipedii:\n{evidence_text}\n\n"
        f"Transcript debaty:\n{debate_transcript}\n\n"
        f"Finalne odpowiedzi debaterów:\n{labels_text}\n\n"
        f"Oceń argumenty obu stron i podaj finalny werdykt."
    )

    raw, total, prompt, compl = _call_llm(
        JUDGE_PROMPT,
        user_content,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    final = _extract_label(raw)
    log.info(
        "Judge: final='%s' (debater_labels=%s, tokens=%d)",
        final, final_labels, total,
    )

    return final, raw, total, prompt, compl


# ═══════════════════════════════════════════════════════════════════════════
# AGENT_CONFIG
# ═══════════════════════════════════════════════════════════════════════════

AGENT_CONFIG = {
    "name": "uam_ga7",
    "model": "openai/gpt-oss-20b",
    "system_prompt": "Debate CoT: Decomposer → RAG → Proponent vs Opponent → Judge",
    "tools": ["rag_two_stage", "claim_decomposition", "adversarial_debate"],
}


# ═══════════════════════════════════════════════════════════════════════════
# debate_ask() — PEŁNY PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def debate_ask(question: str) -> dict[str, Any]:
    """Full debate pipeline: decompose → retrieve → debate → judge.

    Returns
    -------
    dict
        Keys:
        - answer: str — final label from judge
        - total_tokens, prompt_tokens, completion_tokens: int
        - sub_claims: list[str]
        - evidence: list[dict] — structured evidence per sub-claim
        - debate_transcript: str — full debate text
        - debater_labels: dict — final labels per debater
        - judge_reasoning: str — judge's full reasoning
    """
    total_tok = 0
    prompt_tok = 0
    compl_tok = 0

    # ═══ Agent 1: Decomposition (reuse) ═══
    sub_claims, t1, p1, c1 = decompose_claim(question)
    total_tok += t1
    prompt_tok += p1
    compl_tok += c1

    # ═══ Agent 2: Two-stage RAG Retrieval (reuse) ═══
    evidence = retrieve_evidence(sub_claims)

    # ═══ Agent 3: Debate (proponent vs opponent) ═══
    debate_result, t3, p3, c3 = run_debate_rounds(
        question, sub_claims, evidence,
    )
    total_tok += t3
    prompt_tok += p3
    compl_tok += c3

    # ═══ Agent 4: Judge ═══
    final_answer, judge_reasoning, t4, p4, c4 = judge_debate(
        question,
        evidence,
        debate_result["transcript"],
        debate_result["final_labels"],
    )
    total_tok += t4
    prompt_tok += p4
    compl_tok += c4

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
                for c in e.get("chunks", [])
            ],
        })

    return {
        "answer": final_answer,
        "total_tokens": total_tok,
        "prompt_tokens": prompt_tok,
        "completion_tokens": compl_tok,
        "sub_claims": sub_claims,
        "evidence": evidence_output,
        "debate_transcript": debate_result["transcript"],
        "debate_rounds": [
            {
                "round": rd["round"],
                "debaters": [
                    {"role": o["role"], "label": o["label"]}
                    for o in rd["outputs"]
                ],
            }
            for rd in debate_result["rounds"]
        ],
        "debater_labels": debate_result["final_labels"],
        "judge_reasoning": judge_reasoning,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BaseAgent — INTEGRACJA Z eval_loop.py
# ═══════════════════════════════════════════════════════════════════════════


class DebateCoTAgent(BaseAgent):
    """Adversarial Debate agent z Proponent/Opponent i Judge.

    Pipeline:
        Decomposer → Retriever (two-stage RAG)
        → 3 debate rounds (Proponent vs Opponent)
        → Judge (provenance-aware verdict)

    LLM calls per claim: 1 (decompose) + 6 (2 debaters × 3 rounds)
    + 0-1 (judge, 0 if consensus) = 7-8.
    """

    name = AGENT_CONFIG["name"]
    cost_tier = 3  # 7-8 LLM calls per claim (decompose + 6 debate + 0-1 judge)

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
            # Debate agent uses _call_llm from fewshot_cot_rag → patch that module
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
        """Evaluate a single claim via the debate pipeline.

        Parameters
        ----------
        claim : dict
            Must contain at least: ``claim_text``, ``label``.

        Returns
        -------
        dict
            Standard eval_loop result with model_label, original_label,
            is_correct, token stats, time_thought, raw_output.
        """
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label_original", "") or claim.get("label", "")

        t0 = time.perf_counter()

        # Build question with answer choices from metadata (AM benchmark)
        question_with_answers = _build_question_with_answers(claim_text, claim)

        try:
            result = debate_ask(question_with_answers)
        except Exception as exc:
            log.error("Debate pipeline error: %s", exc, exc_info=True)
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
                "debater_labels": result.get("debater_labels", {}),
                "debate_rounds": result.get("debate_rounds", []),
                "debate_transcript": result.get("debate_transcript", ""),
                "judge_reasoning": result.get("judge_reasoning", ""),
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
