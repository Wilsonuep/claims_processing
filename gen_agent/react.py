import json
import logging
import re
from typing import Any, Callable

from gen_agent.llm_client import client

log = logging.getLogger(__name__)

# Tags to strip before JSON parsing. Can be extended via STRIP_THINKING_TAGS env var
# (comma-separated tag names, e.g. "think,reasoning,scratchpad").
import os as _os
_THINKING_TAGS: tuple[str, ...] = tuple(
    t.strip() for t in _os.getenv("STRIP_THINKING_TAGS", "think,reasoning,scratchpad").split(",") if t.strip()
)
_THINKING_TAG_RE = re.compile(
    "|".join(rf"<{t}>.*?</{t}>" for t in _THINKING_TAGS),
    flags=re.DOTALL | re.IGNORECASE,
)


def parse_react_output(text: str) -> dict[str, Any]:
    """
    Parses a ReAct text output looking for a JSON block.
    Strips model-internal thinking tags (<think>, <reasoning>, <scratchpad>, …)
    before parsing. Configurable via STRIP_THINKING_TAGS env var.
    """
    # 1. Strip thinking tags (DeepSeek, QwQ, and similar reasoning models)
    text_no_think = _THINKING_TAG_RE.sub("", text).strip()
    
    # 2. Extract JSON block. 
    # Match ```json \n { ... } \n ``` or fallback to finding raw {} brackets.
    json_match = re.search(r'```(?:json)?(.*?)```', text_no_think, flags=re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Fallback: find the first { and last }
        start = text_no_think.find('{')
        end = text_no_think.rfind('}')
        if start != -1 and end != -1:
            json_str = text_no_think[start:end+1]
        else:
            raise ValueError("Nie znaleziono bloku JSON w odpowiedzi modelu.")
            
    return json.loads(json_str)

def run_react_agent(
    model: str,
    system_prompt: str,
    user_query: str,
    available_tools: dict[str, Callable],
    max_steps: int = 5
) -> dict[str, Any]:
    """
    A universal ReAct loop that executes tools and collects tokens.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Twierdzenie do sprawdzenia:\n{user_query}"}
    ]
    
    total_tokens, prompt_tokens, comp_tokens = 0, 0, 0
    full_trajectory = []
    
    for step in range(max_steps):
        # 1. Call LLM
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        msg = response.choices[0].message
        content = msg.content or ""
        
        # Track Tokens
        if response.usage:
            total_tokens += response.usage.total_tokens
            prompt_tokens += response.usage.prompt_tokens
            comp_tokens += response.usage.completion_tokens
        
        messages.append({"role": "assistant", "content": content})
        full_trajectory.append({"role": "assistant", "content": content})
        
        # 2. Parse Action
        try:
            parsed = parse_react_output(content)
        except Exception as e:
            # If the model fails formatting, give it an error observation to correct itself
            err_msg = f"Observation: Błąd parsowania odpowiedzi. Odpowiedz w poprawnym formacie JSON. Szczegóły: {e}"
            messages.append({"role": "user", "content": err_msg})
            full_trajectory.append({"role": "user", "content": err_msg})
            continue
            
        action = parsed.get("action")
        action_input = parsed.get("action_input", {})
        
        # 3. Check for Final Answer
        if action == "final_answer":
            return {
                "label": action_input.get("label", "ERROR"),
                "reasoning": action_input.get("reasoning", ""),
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tokens,
                "trajectory": full_trajectory
            }
            
        # 4. Execute Tool
        if action in available_tools:
            try:
                # web_search, etc
                observation = available_tools[action](**action_input)
            except Exception as e:
                observation = f"Błąd wykonania narzędzia '{action}': {e}"
        else:
            observation = f"Nieznane narzędzie '{action}'."
            
        # 5. Append Observation
        obs_msg = f"Observation: {observation}"
        messages.append({"role": "user", "content": obs_msg})
        full_trajectory.append({"role": "tool", "content": obs_msg})
        
    # If Max Steps reached
    return {
        "label": "ERROR_MAX_STEPS",
        "reasoning": "Osiągnięto limit kroków ReAct.",
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": comp_tokens,
        "trajectory": full_trajectory
    }
