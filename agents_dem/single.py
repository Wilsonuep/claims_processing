import os
import json
import logging
import time
from typing import Any
from dotenv import load_dotenv

from gen_agent.llm_client import client, MODEL
from gen_agent.base_agent import BaseAgent
from agents_dem.prompts import FACTCHECK_PROMPT

load_dotenv()

log = logging.getLogger(__name__)

"""
Najbardziej podstawowa wersja agenta.
Brak narzędzi.
Dostosowany do formatu Demagog (wynik to JSON z etykietą PRAWDA/FAŁSZ/itd.).
"""

model = MODEL

AGENT_CONFIG = {
    "name": "dem_ga1",
    "model": model,
    "system_prompt": FACTCHECK_PROMPT,
    "tools": [],
}

def ask(question: str) -> dict:
    # Wymuszamy format JSON za pomocą mechanizmu Together AI / OpenAI
    response = client.chat.completions.create(
        model=AGENT_CONFIG["model"],
        messages=[
            {"role": "system", "content": AGENT_CONFIG["system_prompt"]},
            {"role": "user", "content": json.dumps({"statement": question}, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
    )
    choice = response.choices[0]
    usage = response.usage
    return {
        "answer": choice.message.content.strip(),
        "total_tokens": usage.total_tokens if usage else 0,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }

class SingleAgent(BaseAgent):
    name = AGENT_CONFIG["name"]

    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label", "")
        
        t0 = time.perf_counter()
        
        try:
            result = ask(claim_text)
            parsed_answer = json.loads(result["answer"])
            model_label = parsed_answer.get("label", "ERROR")
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
            }
            
        elapsed = time.perf_counter() - t0
        
        return {
            "model_label": model_label,
            "original_label": original_label,
            "is_correct": str(model_label).upper() == str(original_label).upper(),
            "total_tokens": result["total_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "time_thought": elapsed,
            "raw_output": result["answer"],
        }
