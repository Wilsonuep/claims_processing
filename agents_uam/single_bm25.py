"""
Agent open-book z wyszukiwaniem BM25 w lokalnej bazie Wikipedii
Brak narzędzi online — kontekst pochodzi z offline'owego indeksu BM25
Do benchmarku z UAMu
"""

from __future__ import annotations

import ast
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

log = logging.getLogger(__name__)

model = MODEL

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WIKI_DB_PATH = os.getenv(
    "BM25_WIKI_DB",
    os.path.join(_PROJECT_ROOT, "dataprep", "wiki.db"),
)

_bm25_index = None


def _get_bm25():
    global _bm25_index
    if _bm25_index is None:
        from gen_agent.bm25 import BM25Index
        _bm25_index = BM25Index.from_sqlite(_WIKI_DB_PATH)
    return _bm25_index


AGENT_CONFIG = {
    "name": "uam_ga3",
    "model": model,
    "system_prompt": (
        "Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi z wykorzystaniem "
        "dostarczonego kontekstu z Wikipedii.\n"
        "Input: Wypowiedź/pytanie którego prawdziwość masz ocenić wraz z 4 opcjami do wyboru "
        "oraz kontekst z Wikipedii.\n"
        "Instructions: Dokonaj oceny prawdziwości wypowiedzi/pytania i wybierz najbardziej "
        "odpowiednią opcję. Wykorzystujesz dostarczony kontekst z Wikipedii do weryfikacji "
        "informacji. Jeśli kontekst nie zawiera wystarczających informacji, polegaj na swojej "
        "wiedzy ogólnej.\n"
        "Output: 0, 1, 2 or 3"
    ),
    "tools": ["bm25_wikipedia"],
}

BM25_TOP_K: int = 5
BM25_MAX_CONTEXT_CHARS: int = 3000


def ask(question: str) -> dict:
    """Wysyła pytanie do agenta z kontekstem BM25 i zwraca odpowiedź wraz z metadanymi."""
    bm25 = _get_bm25()
    context = bm25.search_and_format(
        question, k=BM25_TOP_K, max_context_chars=BM25_MAX_CONTEXT_CHARS
    )

    if context:
        user_content = (
            f"Kontekst z Wikipedii:\n{context}\n\n"
            f"Pytanie:\n{question}"
        )
    else:
        user_content = question

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AGENT_CONFIG["system_prompt"]},
            {"role": "user", "content": user_content},
        ],
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "answer": choice.message.content.strip(),
        "total_tokens": usage.total_tokens if usage else 0,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }


def _extract_label(answer: str) -> str:
    stripped = answer.strip()
    if stripped in ("0", "1", "2", "3"):
        return stripped
    match = re.search(r"[0-3]", stripped)
    if match:
        return match.group()
    return stripped


def _build_question_with_answers(claim_text: str, claim: dict) -> str:
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


class SingleBM25Agent(BaseAgent):
    """Open-book BM25 agent (uam_ga3)."""

    name = AGENT_CONFIG["name"]
    cost_tier = 1

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
            import agents_uam.single_bm25 as _m
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
        question_with_answers = _build_question_with_answers(claim_text, claim)

        t0 = time.perf_counter()
        try:
            result = ask(question_with_answers)
        except Exception as exc:
            log.error("Błąd agenta: %s", exc)
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
        return {
            "model_label": model_label,
            "original_label": original_label,
            "is_correct": str(model_label) == str(original_label),
            "total_tokens": result["total_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "time_thought": elapsed,
            "raw_output": result["answer"],
            "model_name": self.model_name or "",
        }
