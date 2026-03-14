import os
import json
import logging
import time
from typing import Any
from dotenv import load_dotenv

from gen_agent.llm_client import MODEL
from gen_agent.base_agent import BaseAgent
from gen_agent.react import run_react_agent
from agents_dem.prompts import FACTCHECK_PROMPT

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

load_dotenv()

log = logging.getLogger(__name__)

"""
Wersja agenta Demagog z dostępem do internetu (ReAct).
Wykorzystuje DuckDuckGo do wyszukiwania w sieci i uniwersalną pętlę ReAct.
"""

def perform_web_search(query: str) -> str:
    if not DDGS:
        return "Błąd: pakiet duckduckgo-search nie jest zainstalowany. Zainstaluj go aby używać web_search."
    try:
        log.info("Wyszukiwanie w duckduckgo: %s", query)
        results = DDGS().text(query, max_results=3)
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return f"Błąd wyszukiwania: {e}"

REACT_TOOLS = {
    "web_search": perform_web_search
}

REACT_INSTRUCTION = """
Masz dostęp do następujących narzędzi:
- `web_search(query: str)`: Wyszukuje w internecie informacje na podany temat, zwracając listę wyników.

Aby użyć narzędzia, **MUSISZ** zwrócić blok JSON w poniższym formacie i NIC WIĘCEJ:
```json
{
  "action": "web_search",
  "action_input": {"query": "Twoje zapytanie do wyszukiwarki"}
}
```

Kiedy system zwróci Ci wyniki narzędzia (jako `Observation:`), przeanalizuj je.
Gdy będziesz gotowy wydać ostateczny werdykt (zgodny z wytycznymi), **MUSISZ** zwrócić odpowiedź końcową korzystając z `final_answer`. Sekcja `action_input` musi zawierać wymagane pola.

```json
{
  "action": "final_answer",
  "action_input": {
    "label": "PRAWDA | CZĘŚCIOWA_PRAWDA | FAŁSZ | MANIPULACJA | NIEWERYFIKOWALNE",
    "justification": "Krótkie, rzeczowe uzasadnienie, maks 5 zdań.",
    "evidence": []
  }
}
```

ZASADY:
1. Wykonuj tylko JEDNO użycie narzędzia naraz.
2. Zawsze umieszczaj wywołanie akcji w formacie ```json ... ```.
3. Jeśli korzystasz z modelu DeepSeek (R1), swoje przemyślenia umieszczaj w tagach <think>...</think>, a zaraz pod nimi wstaw blok JSON z akcją.
"""

REACT_SYSTEM_PROMPT = FACTCHECK_PROMPT + "\n\n" + REACT_INSTRUCTION

class SingleWebAgent(BaseAgent):
    name = "dem_ga2"

    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label", "")
        
        t0 = time.perf_counter()
        
        try:
            result = run_react_agent(
                model=MODEL,
                system_prompt=REACT_SYSTEM_PROMPT,
                user_query=claim_text,
                available_tools=REACT_TOOLS,
                max_steps=5
            )
            model_label = result.get("label", "ERROR")
            # In result, reasoning is returned, but we mapped justification to reasoning optionally
            # Let's stringify the trajectory to be saved in raw_output
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
            }
            
        elapsed = time.perf_counter() - t0
        
        return {
            "model_label": model_label,
            "original_label": original_label,
            "is_correct": str(model_label).upper() == str(original_label).upper(),
            "total_tokens": result.get("total_tokens", 0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "time_thought": elapsed,
            "raw_output": raw_trajectory,
        }
