import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any

from dotenv import load_dotenv

from gen_agent.llm_client import client, MODEL
from gen_agent.base_agent import BaseAgent
from agents_dem.prompts import FACTCHECK_PROMPT

load_dotenv()
log = logging.getLogger(__name__)

model = MODEL

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WIKI_DB_PATH = os.getenv(
    "RAG_WIKI_DB",
    os.path.join(_PROJECT_ROOT, "dataprep", "wiki.db"),
)
_RAG_MODE = os.getenv("RAG_MODE", "bm25")

RAG_K_INITIAL: int = 20
RAG_K_PER_CLAIM: int = 5
RAG_SCORE_THRESHOLD: float | None = None
RAG_MAX_CONTEXT_CHARS: int = 2000

REASONER_CONFIGS: list[dict[str, Any]] = [
    {"temperature": 0.2, "label": "konserwatywny", "top_p": 0.9, "max_tokens": 1024},
    {"temperature": 0.7, "label": "zbalansowany",  "top_p": 0.95, "max_tokens": 1024},
    {"temperature": 1.0, "label": "kreatywny",     "top_p": 1.0, "max_tokens": 1024},
]
CONSOLIDATOR_TEMPERATURE: float = 0.2
CONSOLIDATOR_MAX_TOKENS: int = 1500

COT_EXAMPLES = """\
--- Przykład 1 ---
Twierdzenie: "Wisła jest najdłuższą rzeką w Polsce."
Pod-twierdzenia: ["Wisła jest rzeką w Polsce", "Wisła jest najdłuższą rzeką w Polsce"]
Rozumowanie:
- Pod-twierdzenie 1: "Wisła jest rzeką w Polsce" → Wg kontekstu z artykułu "Wisła — Geografia": Wisła to rzeka w Polsce o długości 1047 km. POTWIERDZONE.
- Pod-twierdzenie 2: "Wisła jest najdłuższą rzeką w Polsce" → Wg kontekstu: Wisła jest najdłuższą rzeką Polski (1047 km). POTWIERDZONE.
Wniosek: Oba pod-twierdzenia potwierdzone przez kontekst.
Odpowiedź: PRAWDA

--- Przykład 2 ---
Twierdzenie: "Kraków jest obecnie stolicą Polski."
Pod-twierdzenia: ["Kraków jest stolicą Polski", "Jest nią obecnie"]
Rozumowanie:
- Pod-twierdzenie 1: "Kraków jest stolicą Polski" → Wg kontekstu z artykułu "Kraków — Historia": Kraków był stolicą do 1596 r. Obecnie stolicą jest Warszawa. ZAPRZECZONE.
- Pod-twierdzenie 2: "Jest nią obecnie" → Kontekst jasno wskazuje, że stolicą jest Warszawa od 1596 r. ZAPRZECZONE.
Wniosek: Twierdzenie jest fałszywe — Kraków nie jest obecną stolicą.
Odpowiedź: FAŁSZ

--- Przykład 3 ---
Twierdzenie: "Polska ma 40 milionów mieszkańców i jest największym krajem UE."
Pod-twierdzenia: ["Polska ma 40 mln mieszkańców", "Polska jest największym krajem UE"]
Rozumowanie:
- Pod-twierdzenie 1: "Polska ma 40 mln mieszkańców" → Wg kontekstu z artykułu "Polska — Demografia": Polska ma ok. 38 mln mieszkańców. Blisko, ale nie 40 mln. CZĘŚCIOWO PRAWDA.
- Pod-twierdzenie 2: "Polska jest największym krajem UE" → Wg kontekstu: największym krajem UE pod względem powierzchni jest Francja. ZAPRZECZONE.
Wniosek: Jedno pod-twierdzenie częściowo prawdziwe, drugie fałszywe. Główny wydźwięk nie zgadza się z prawdą. 
Odpowiedź: CZĘŚCIOWA_PRAWDA

--- Przykład 4 ---
Twierdzenie: "W 2025 roku Polska będzie miała najwyższe PKB w regionie."
Pod-twierdzenia: ["Polska będzie miała najwyższe PKB w regionie", "Stanie się to w 2025"]
Rozumowanie:
- Pod-twierdzenie 1: "Polska będzie miała najwyższe PKB w regionie" → Brak kontekstu dotyczącego przyszłych prognoz PKB.
- Pod-twierdzenie 2: "Stanie się to w 2025" → Brak kontekstu do weryfikacji.
Wniosek: Kontekst nie zawiera informacji do weryfikacji tego twierdzenia.
Odpowiedź: NIEWERYFIKOWALNE
"""

DECOMPOSER_PROMPT = """\
Jesteś ekspertem od analizy twierdzeń (fact-checking).

Zadanie: Rozłóż poniższe twierdzenie na mniejsze, weryfikowalne pod-twierdzenia (sub-claims).
Zasady:
- Wyodrębnij 1-5 pod-twierdzeń.
- Odpowiedz TYLKO w formacie JSON: listą stringów.
"""

REASONER_PROMPT = """\
Jesteś ekspertem od weryfikacji twierdzeń (fact-checking). Stosuj rozumowanie krok po kroku (chain-of-thought).

Otrzymujesz:
1. Oryginalne pytanie/twierdzenie z opcjami odpowiedzi.
2. Dekompozycję na pod-twierdzenia.
3. Kontekst z bazy wiedzy.

ZASADY GROUNDED REASONING:
- PREFERUJ informacje z dostarczonego kontekstu nad własną wiedzę ogólną.
- Jeśli kontekst potwierdza — powiedz POTWIERDZONE.
- Jeśli kontekst zaprzecza — powiedz ZAPRZECZONE.
- Jeśli brak danych — powiedz BRAK KONTEKSTU.

Dozwolone etykiety: PRAWDA, CZĘŚCIOWA_PRAWDA, FAŁSZ, MANIPULACJA, NIEWERYFIKOWALNE.

{examples}

Format odpowiedzi (musi kończyć się dokładnie taką linijką, ze słowem Odpowiedź: ETYKIETA):
Rozumowanie:
- [krok po kroku analiza każdego pod-twierdzenia]
Wniosek: [podsumowanie]
Odpowiedź: [ETYKIETA]\
"""

CONSOLIDATOR_PROMPT = FACTCHECK_PROMPT + """

DODATKOWE INSTRUKCJE DO KONSOLIDACJI:
Otrzymujesz trzy niezależne analizy tego samego twierdzenia od innych ekspertów.
Oceniaj je na podstawie "ugruntowania" w dostarczonym kontekście. 
Pamiętaj, by ODRZUCAĆ halucynacje i preferować ostrożniejsze opcje (CZĘŚCIOWA_PRAWDA / NIEWERYFIKOWALNE) przy konfliktach, jeśli to uzasadnione.
Skomponuj ostateczną odpowiedź w formacie JSON wymaganym powyżej.

{pre_analysis}\
"""

def _call_llm(system_prompt: str, user_content: str, temperature=0.7, top_p=0.95, max_tokens=1024, json_format=False) -> tuple[str, int, int, int]:
    kwargs = {}
    if json_format:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **kwargs
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
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list): return [str(item) for item in parsed if item]
    except: pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list): return [str(item) for item in parsed if item]
        except: pass
    return []

def _extract_label(answer: str) -> str:
    match = re.search(r"[Oo]dpowied[źz]:\s*(PRAWDA|CZĘŚCIOWA_PRAWDA|FAŁSZ|MANIPULACJA|NIEWERYFIKOWALNE)", answer)
    if match: return match.group(1).upper()
    for lbl in ["PRAWDA", "CZĘŚCIOWA_PRAWDA", "FAŁSZ", "MANIPULACJA", "NIEWERYFIKOWALNE"]:
        if lbl in answer.upper(): return lbl
    return "ERROR"

def decompose_claim(claim_text: str) -> tuple[list[str], int, int, int]:
    raw, total, prompt, compl = _call_llm(DECOMPOSER_PROMPT, claim_text, temperature=0.3, max_tokens=512)
    sub_claims = _parse_json_list(raw)
    if not sub_claims: sub_claims = [claim_text]
    return sub_claims, total, prompt, compl

_rag_retriever = None

def _get_rag():
    global _rag_retriever
    if _rag_retriever is None:
        from gen_agent.rag import RAGRetriever
        rag_kwargs = {
            "mode": _RAG_MODE,
            "bm25_db_path": _WIKI_DB_PATH,
            "text_field": "text", "title_field": "title",
            "section_field": "section_title", "id_field": "chunk_id",
        }
        if _RAG_MODE in ("vector", "hybrid"):
            rag_kwargs["vector_db_path"] = _WIKI_DB_PATH
        _rag_retriever = RAGRetriever(**rag_kwargs)
    return _rag_retriever

def retrieve_evidence(sub_claims: list[str]) -> list[dict[str, Any]]:
    from gen_agent.rag import RAGRetriever
    rag = _get_rag()
    results = []
    for sc in sub_claims:
        chunks = rag.retrieve_structured(sc, k_initial=RAG_K_INITIAL, k_final=RAG_K_PER_CLAIM, score_threshold=RAG_SCORE_THRESHOLD)
        context_text = RAGRetriever.format_evidence(chunks, max_context_chars=RAG_MAX_CONTEXT_CHARS)
        results.append({"sub_claim": sc, "chunks": chunks, "context_formatted": context_text, "num_results": len(chunks)})
    return results

def run_reasoners(original_question: str, evidence: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
    parts = []
    for i, e in enumerate(evidence, start=1):
        part = f"=== Pod-twierdzenie {i}: {e['sub_claim']} ===\n"
        if e["context_formatted"]: part += f"Kontekst:\n{e['context_formatted']}\n"
        else: part += "⚠ Brak kontekstu.\n"
        parts.append(part)
    
    user_prompt = (f"Pytanie/twierdzenie: {original_question}\n\nDowody:\n{''.join(parts)}\nPrzeanalizuj to krok po kroku.")
    system_prompt = REASONER_PROMPT.format(examples=COT_EXAMPLES)

    tot, prm, cmp = 0, 0, 0
    outputs = []
    for cfg in REASONER_CONFIGS:
        raw, t_t, t_p, t_c = _call_llm(system_prompt, user_prompt, temperature=cfg["temperature"], top_p=cfg["top_p"], max_tokens=cfg["max_tokens"])
        extracted = _extract_label(raw)
        outputs.append({"label": extracted, "temperature": cfg["temperature"], "top_p": cfg["top_p"], "style": cfg["label"], "reasoning": raw})
        tot += t_t; prm += t_p; cmp += t_c

    return outputs, tot, prm, cmp

def _compute_evidence_grounding(reasoner_outputs: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> str:
    labels = [r["label"] for r in reasoner_outputs]
    if len(set(labels)) == 1: return ""
    
    evidence_terms = set()
    for e in evidence:
        for chunk in e.get("chunks", []):
            for word in re.split(r"\W+", f"{chunk.get('title','')} {chunk.get('section','')} ".lower()):
                if len(word) > 3: evidence_terms.add(word)
    if not evidence_terms: return ""

    gs = [(r["style"], sum(1 for t in evidence_terms if t in r["reasoning"].lower()), r["label"]) for r in reasoner_outputs]
    lines = ["Wstępna analiza grounding'u:"]
    for style, score, label in gs:
        lines.append(f"  - Ekspert {style}: {score} odwołań, odp: {label}")
    return "\n".join(lines)

def consolidate(original_question: str, reasoner_outputs: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> tuple[dict, str, int, int, int]:
    labels = [r["label"] for r in reasoner_outputs]
    
    pre_analysis = _compute_evidence_grounding(reasoner_outputs, evidence)
    expert_sections = []
    for i, r in enumerate(reasoner_outputs, start=1):
        expert_sections.append(f"=== Ekspert {i} ({r['style']}) ===\nOdpowiedź: {r['label']}\nRozumowanie:\n{r['reasoning']}\n")

    user_content = json.dumps({
        "statement": original_question,
        "experts": expert_sections
    }, ensure_ascii=False)

    system_prompt = CONSOLIDATOR_PROMPT.format(pre_analysis=pre_analysis if pre_analysis else "")

    raw, total, prompt, compl = _call_llm(system_prompt, user_content, temperature=CONSOLIDATOR_TEMPERATURE, max_tokens=CONSOLIDATOR_MAX_TOKENS, json_format=True)
    
    try:
        final_json = json.loads(raw)
    except:
        final_json = {"label": "ERROR", "justification": "", "evidence": []}

    return final_json, raw, total, prompt, compl

AGENT_CONFIG = {
    "name": "dem_ga6",
    "model": model,
    "system_prompt": "Few-Shot CoT RAG (Demagog JSON)",
    "tools": ["rag_two_stage", "claim_decomposition", "multiple_reasoners"],
}

class FewShotCoTAgent(BaseAgent):
    name = AGENT_CONFIG["name"]

    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_text = claim.get("claim_text", "")
        original_label = claim.get("label", "")
        t0 = time.perf_counter()

        try:
            sub_claims, t1, p1, c1 = decompose_claim(claim_text)
            evidence = retrieve_evidence(sub_claims)
            reasoner_outputs, t2, p2, c2 = run_reasoners(claim_text, evidence)
            final_json, raw_judge, t3, p3, c3 = consolidate(claim_text, reasoner_outputs, evidence)
            model_label = final_json.get("label", "ERROR")
            
            total_tokens = t1 + t2 + t3
            prompt_tokens = p1 + p2 + p3
            completion_tokens = c1 + c2 + c3
        except Exception as exc:
            log.error("Pipeline error: %s", exc)
            return {
                "model_label": "ERROR", "original_label": original_label, "is_correct": False,
                "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "time_thought": time.perf_counter()-t0, "raw_output": f"ERROR: {exc}"
            }

        elapsed = time.perf_counter() - t0
        raw_output_dump = json.dumps({
            "final_json": final_json,
            "reasoners": [{"style": r["style"], "label": r["label"]} for r in reasoner_outputs],
        }, ensure_ascii=False)

        return {
            "model_label": model_label,
            "original_label": original_label,
            "is_correct": str(model_label).upper() == str(original_label).upper(),
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "time_thought": elapsed,
            "raw_output": raw_output_dump,
        }
