"""
Najbardziej podstawowa wersja agenta
Brak narzędzi
Do benchmarku z UAMu
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from typing import Any

from dotenv import load_dotenv

from claims_processing.core.llm_client import client, MODEL
from claims_processing.core.base_agent import BaseAgent

load_dotenv()

log = logging.getLogger(__name__)

model = MODEL

AGENT_CONFIG = {
    "name": "uam_ga1",
    "model": model,
    "system_prompt": (
        "Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi bez wykorzystania "
        "jakichkolwiek narzędzi.\n"
        "Input: Wypowiedź/pytanie którego prawdziwość masz ocenić wraz z 4 opcjami do wyboru\n"
        "Instructions: Dokonaj oceny prawdziwości wypowiedzi/pytania i wybierz najbardziej "
        "odpowiednią opcję. Nie wykorzystujesz żadnych narzędzi i masz polegać tylko na swojej "
        "wiedzy ogólnej.\n"
        "Output: 0, 1, 2 or 3"
    ),
    "tools": [],
}


def ask(question: str, model_name: str | None = None) -> dict:
    """Wysyła pytanie do agenta i zwraca odpowiedź wraz z metadanymi."""
    response = client.chat.completions.create(
        model=model_name or model,
        messages=[
            {"role": "system", "content": AGENT_CONFIG["system_prompt"]},
            {"role": "user", "content": question},
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


class SingleAgent(BaseAgent):
    """Zero-shot agent bez narzędzi (uam_ga1)."""

    name = AGENT_CONFIG["name"]
    cost_tier = 1

    def __init__(self, model_override: str | None = None) -> None:
        from claims_processing.core.llm_client import make_client, MODEL as _DEFAULT_MODEL
        if model_override is not None:
            _, self._model = make_client(model_override)
            suffix = model_override.replace("/", "-").replace(":", "-")
            self.name = f"uam_ga1__{suffix}"
            self.model_name = model_override
        else:
            self._model = MODEL
            self.model_name = _DEFAULT_MODEL

    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label_original", "") or claim.get("label", "")
        question_with_answers = _build_question_with_answers(claim_text, claim)

        t0 = time.perf_counter()
        try:
            result = ask(question_with_answers, self._model)
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