import os
import json
import logging
import re
import time
from typing import Any
from dotenv import load_dotenv

from gen_agent.llm_client import MODEL
from gen_agent.base_agent import BaseAgent
from gen_agent.react import run_react_agent

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

load_dotenv()

log = logging.getLogger(__name__)

"""
Wersja agenta UAM z dostępem do internetu (ReAct).
Zastępuje przestarzałe, jednostopniowe wywołanie `ask()` pełnoprawną pętlą ReAct.
"""

def perform_web_search(query: str) -> str:
    if not DDGS:
        return "Błąd: pakiet duckduckgo-search nie jest zainstalowany."
    try:
        log.info("Wyszukiwanie w duckduckgo: %s", query)
        results = DDGS().text(query, max_results=3)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Błąd wyszukiwania: {e}"

REACT_TOOLS = {
    "web_search": perform_web_search
}

SYSTEM_PROMPT = """Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi w benchmarku UAM.
Dokonaj oceny prawdziwości wypowiedzi/pytania korzystając z dowodów uzyskanych w sieci. Oczekiwany jest pojedynczy, ostateczny wyrok: {0, 1, 2, 3}.

Aby wyszukać informacje, wykonaj użycie narzędzia:
```json
{
  "action": "web_search",
  "action_input": {"query": "Zapytanie"}
}
```

Kiedy system zwróci wyniki (`Observation:`), przeanalizuj je.
Gdy będziesz gotowy wydać ostateczny werdykt (0, 1, 2 lub 3), użyj akcji `final_answer`. Twoje `action_input` musi zawierać końcowe rozwiązanie w "label".

```json
{
  "action": "final_answer",
  "action_input": {
    "label": "2",
    "reasoning": "Uzasadnienie decyzji dla wybranej opcji"
  }
}
```

Jeśli twój model używa tagów <think> (np DeepSeek), wstaw blok json na zewnątrz / po tagach think.
"""


def _normalize_uam_label(raw: str) -> str:
    s = str(raw).strip()
    # "1.0" → "1" (only for exact integer floats)
    try:
        f = float(s)
        rounded = str(round(f))
        if rounded in {"0", "1", "2", "3"} and abs(f - round(f)) < 0.01:
            return rounded
    except ValueError:
        pass
    # "Output: 2" or similar prefix → extract digit
    m = re.search(r"\b([0-3])\b", s)
    if m:
        return m.group(1)
    return s  # leave ERROR / ERROR_MAX_STEPS unchanged


class SingleWebAgent(BaseAgent):
    name = "uam_ga2"
    cost_tier = 1

    def __init__(self, model_override: str | None = None) -> None:
        from gen_agent.llm_client import make_client, MODEL as _DEFAULT_MODEL
        if model_override is not None:
            _, self._model = make_client(model_override)
            suffix = model_override.replace("/", "-").replace(":", "-")
            self.name = f"uam_ga2__{suffix}"
            self.model_name = model_override
        else:
            self._model = MODEL
            self.model_name = _DEFAULT_MODEL

    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label_original", "") or claim.get("label", "")

        t0 = time.perf_counter()

        try:
            result = run_react_agent(
                model=self._model,
                system_prompt=SYSTEM_PROMPT,
                user_query=claim_text,
                available_tools=REACT_TOOLS,
                max_steps=8
            )
            model_label = _normalize_uam_label(result.get("label") or "ERROR")
            raw_trajectory = json.dumps(result.get("trajectory", []), ensure_ascii=False)

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
        return {
            "model_label": str(model_label),
            "original_label": str(original_label),
            "is_correct": str(model_label).strip() == str(original_label).strip(),
            "total_tokens": result.get("total_tokens", 0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "time_thought": elapsed,
            "raw_output": raw_trajectory,
            "model_name": self.model_name or "",
        }