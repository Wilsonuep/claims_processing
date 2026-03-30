import json
import logging
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

from gen_agent.base_agent import BaseAgent
from agents_dem.prompts import FACTCHECK_PROMPT

from agents_dem.fewshot_cot_rag import (
    _call_llm,
    _extract_label,
    _parse_json_list,
    decompose_claim,
    retrieve_evidence,
    RAG_K_INITIAL,
    RAG_K_PER_CLAIM,
    RAG_SCORE_THRESHOLD,
    RAG_MAX_CONTEXT_CHARS,
)

load_dotenv()
log = logging.getLogger(__name__)

DEBATE_NUM_ROUNDS: int = 3      
DEBATE_ROUND_NAMES: list[str] = ["Opening", "Rebuttal", "Closing"]
DEBATE_ROUND_MAX_TOKENS: dict[str, int] = {"Opening": 768, "Rebuttal": 768, "Closing": 512}
DEBATE_TEMPERATURES: dict[str, float] = {"proponent": 0.5, "opponent": 0.5}
DEBATE_TOP_P: float = 0.95
JUDGE_TEMPERATURE: float = 0.2
JUDGE_MAX_TOKENS: int = 1500

DEBATER_PROPONENT_PROMPT = """\
Jesteś ekspertem fact-checkingowym w roli PROPONENTA (obrońcy twierdzenia).
Twoje zadanie: argumentować, że dane twierdzenie to PRAWDA lub CZĘŚCIOWA_PRAWDA, opierając się na kontekście.
NIE wymyślaj faktów nieobecnych w kontekście — to dyskwalifikuje argument.
Jeśli kontekst jest niewystarczający, argumentuj za "NIEWERYFIKOWALNE".

KONTEKST DEBATY:
{debate_context}

Format odpowiedzi (koniecznie musi się kończyć linijką Odpowiedź: ETYKIETA):
Argument:
- [punkt po punkcie]
Odpowiedź: [ETYKIETA]\
"""

DEBATER_OPPONENT_PROMPT = """\
Jesteś ekspertem fact-checkingowym w roli OPONENTA (krytyka twierdzenia).
Twoje zadanie: argumentować, że dane twierdzenie to FAŁSZ, MANIPULACJA lub NIEWERYFIKOWALNE, opierając się na kontekście.
NIE wymyślaj faktów nieobecnych w kontekście.
Szukaj rozbieżności, nieścisłości, brakujących danych.

KONTEKST DEBATY:
{debate_context}

Format odpowiedzi (koniecznie musi się kończyć linijką Odpowiedź: ETYKIETA):
Argument:
- [punkt po punkcie]
Odpowiedź: [ETYKIETA]\
"""

JUDGE_PROMPT = FACTCHECK_PROMPT + """

DODATKOWE INSTRUKCJE DO DEBATY:
Otrzymujesz pełny transcript debaty między proponentem (obrońcą) a oponentem (krytykiem) twierdzenia.
Twoje zadanie: na podstawie argumentów obu stron oraz dostarczonego kontekstu, podejmij finalną decyzję o prawdziwości twierdzenia.
PREFERUJ argumenty ugruntowane w dostarczonym kontekście. ODRZUCAJ argumenty oparte na halucynacjach.
Zwróć wynik jako obiekt JSON zgodnie ze specyfikacją.
"""

DEBATER_PROMPTS: dict[str, str] = {
    "proponent": DEBATER_PROPONENT_PROMPT,
    "opponent": DEBATER_OPPONENT_PROMPT,
}

def _build_evidence_summary(original_question: str, sub_claims: list[str], evidence: list[dict[str, Any]]) -> str:
    parts = []
    parts.append(f"Twierdzenie do weryfikacji:\n{original_question}\n")
    parts.append(f"Pod-twierdzenia: {json.dumps(sub_claims, ensure_ascii=False)}\n")
    for i, e in enumerate(evidence, start=1):
        section = f"=== Pod-twierdzenie {i}: {e['sub_claim']} ===\n"
        if e.get("context_formatted"): section += f"Kontekst:\n{e['context_formatted']}\n"
        else: section += "⚠ Brak kontekstu.\n"
        parts.append(section)
    return "\n".join(parts)

def _build_debater_prompt(role: str, evidence_summary: str, debate_history: str, round_name: str) -> tuple[str, str]:
    base_prompt = DEBATER_PROMPTS[role]
    if round_name == "Opening": debate_context = "Runda otwarcia. Zaprezentuj argument."
    elif round_name == "Rebuttal": debate_context = f"Runda odpowiedzi. Przeanalizuj argument.\nDebata:\n{debate_history}"
    else: debate_context = f"Runda zamknięcia. Podsumuj i podaj finalną odpowiedź.\nDebata:\n{debate_history}"
    return base_prompt.format(debate_context=debate_context), evidence_summary

def run_debater(role: str, evidence_summary: str, debate_history: str, round_name: str) -> tuple[dict[str, Any], int, int, int]:
    sys_prompt, user_prompt = _build_debater_prompt(role, evidence_summary, debate_history, round_name)
    raw, t_tot, t_prm, t_cmp = _call_llm(sys_prompt, user_prompt, temperature=DEBATE_TEMPERATURES.get(role, 0.5), top_p=DEBATE_TOP_P, max_tokens=DEBATE_ROUND_MAX_TOKENS.get(round_name, 768))
    label = _extract_label(raw)
    return {"role": role, "round": round_name, "label": label, "argument": raw}, t_tot, t_prm, t_cmp

def run_debate_rounds(original_question: str, sub_claims: list[str], evidence: list[dict]) -> tuple[dict, int, int, int]:
    ev_summary = _build_evidence_summary(original_question, sub_claims, evidence)
    tot, prm, cmp = 0, 0, 0
    rounds_data, transcript_parts = [], []
    history = ""
    for rd in DEBATE_ROUND_NAMES[:DEBATE_NUM_ROUNDS]:
        r_outs = []
        for role in ["proponent", "opponent"]:
            try:
                out, t_tot, t_prm, t_cmp = run_debater(role, ev_summary, history, rd)
            except Exception as debater_exc:
                log.warning("Debater [%s, %s] failed (%s) — fallback NIEWERYFIKOWALNE.", role, rd, debater_exc)
                out = {"role": role, "round": rd, "label": "NIEWERYFIKOWALNE", "argument": f"[BŁĄD: {debater_exc}]"}
                t_tot = t_prm = t_cmp = 0
            r_outs.append(out)
            tot += t_tot; prm += t_prm; cmp += t_cmp
            transcript_parts.append(f"── {role.upper()} ({rd}) ──\n{out['argument']}\n")
        history = "\n".join(transcript_parts)
        rounds_data.append({"round": rd, "outputs": r_outs})
    
    fin_labels = {o["role"]: o["label"] for o in rounds_data[-1]["outputs"]} if rounds_data else {}
    return {"transcript": history, "rounds": rounds_data, "final_labels": fin_labels}, tot, prm, cmp

def judge_debate(original_question: str, evidence: list[dict], trans: str, fin_labels: dict) -> tuple[dict, str, int, int, int]:
    evidence_text = "\n".join([f"Pod-twierdzenie: {e['sub_claim']}\n{e['context_formatted']}\n" for e in evidence if e.get("context_formatted")])
    labels_text = "\n".join(f"  - {r.upper()}: {l}" for r, l in fin_labels.items())
    
    user_content = json.dumps({
        "statement": original_question,
        "evidence": evidence_text,
        "debate_transcript": trans,
        "debater_labels": labels_text
    }, ensure_ascii=False)

    raw, total, prompt, compl = _call_llm(JUDGE_PROMPT, user_content, temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS, json_format=True)
    try: final_json = json.loads(raw)
    except: final_json = {"label": "ERROR"}

    return final_json, raw, total, prompt, compl

AGENT_CONFIG = {
    "name": "dem_ga7",
    "model": "openai/gpt-oss-20b",
    "system_prompt": "Debate CoT RAG for Demagog",
    "tools": ["rag_two_stage", "claim_decomposition", "adversarial_debate"],
}

class DebateCoTAgent(BaseAgent):
    name = AGENT_CONFIG["name"]
    cost_tier = 3  # 7-8 LLM calls per claim

    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label", "")
        t0 = time.perf_counter()

        try:
            sub_claims, t1, p1, c1 = decompose_claim(claim_text)
            evidence = retrieve_evidence(sub_claims)
            deb_res, t3, p3, c3 = run_debate_rounds(claim_text, sub_claims, evidence)
            final_json, raw_judge, t4, p4, c4 = judge_debate(claim_text, evidence, deb_res["transcript"], deb_res["final_labels"])
            
            tot = t1 + t3 + t4
            prm = p1 + p3 + p4
            cmp = c1 + c3 + c4
            model_label = final_json.get("label", "ERROR")
        except Exception as exc:
            log.error("Debate error: %s", exc)
            return {"model_label": "ERROR", "original_label": original_label, "is_correct": False, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "time_thought": time.perf_counter()-t0, "raw_output": f"ERROR: {exc}"}

        elapsed = time.perf_counter() - t0
        raw_output_dump = json.dumps({
            "final_json": final_json,
            "debater_labels": deb_res["final_labels"],
        }, ensure_ascii=False)

        return {
            "model_label": model_label,
            "original_label": original_label,
            "is_correct": str(model_label).upper() == str(original_label).upper(),
            "total_tokens": tot,
            "prompt_tokens": prm,
            "completion_tokens": cmp,
            "time_thought": elapsed,
            "raw_output": raw_output_dump,
        }
