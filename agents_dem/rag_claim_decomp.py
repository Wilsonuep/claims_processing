import json
import logging
import os
import re
import time
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

DECOMPOSER_PROMPT = """\
Jesteś ekspertem od analizy twierdzeń (fact-checking).

Zadanie: Rozłóż poniższe twierdzenie na mniejsze, weryfikowalne pod-twierdzenia (sub-claims). Każde pod-twierdzenie powinno być prostym stwierdzeniem faktycznym, które można niezależnie zweryfikować.

Zasady:
- Wyodrębnij 1-5 pod-twierdzeń.
- Każde pod-twierdzenie musi być samodzielne (zrozumiałe bez kontekstu).
- Zachowaj język oryginału.
- Odpowiedz TYLKO w formacie JSON: listą stringów.
- Nie dodawaj żadnego tekstu poza JSON.

Przykład odpowiedzi:
["Kraków był stolicą Polski", "Stolica została przeniesiona w 1596 roku"]
"""

VERIFIER_PROMPT = FACTCHECK_PROMPT + """

DODATKOWE INSTRUKCJE:
Otrzymujesz oryginalne twierdzenie (statement) do weryfikacji. Oprócz tego otrzymujesz jego dekompozycję oraz dowody (kontekst z lokalnej bazy wiedzy RAG).
Na podstawie powiązanych dowodów i swojej wiedzy zastosuj się do standardowych procedur. Zwróć finalny JSON.
"""

def _call_llm_json(system_prompt: str, user_content: str, max_tokens=1024) -> tuple[str, int, int, int]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
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

def _call_llm(system_prompt: str, user_content: str, max_tokens=1024) -> tuple[str, int, int, int]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
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

def decompose_claim(claim_text: str) -> tuple[list[str], int, int, int]:
    raw_answer, total, prompt, completion = _call_llm(DECOMPOSER_PROMPT, claim_text, 512)
    sub_claims = _parse_json_list(raw_answer)
    if not sub_claims:
        log.warning("Decomposer fallback → oryginał. Raw: %s", raw_answer[:200])
        sub_claims = [claim_text]
    log.info("Decomposer: %d pod-twierdzeń.", len(sub_claims))
    return sub_claims, total, prompt, completion

_rag_retriever = None

def _get_rag():
    global _rag_retriever
    if _rag_retriever is None:
        from gen_agent.rag import RAGRetriever
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

def retrieve_contexts(
    sub_claims: list[str],
    k_per_claim: int = 3,
    max_context_chars: int = 2000,
) -> list[dict[str, Any]]:
    rag = _get_rag()
    results: list[dict[str, Any]] = []
    for sc in sub_claims:
        context = rag.retrieve_and_format(sc, k=k_per_claim, max_context_chars=max_context_chars)
        results.append({
            "sub_claim": sc,
            "context": context,
            "num_results": len(rag.retrieve(sc, k=k_per_claim)),
        })
    log.info("Retriever: pobrano kontekst dla %d pod-twierdzeń.", len(results))
    return results

def verify_claim(
    original_question: str,
    evidence: list[dict[str, Any]],
) -> tuple[str, int, int, int]:
    evidence_sections: list[str] = []
    for i, e in enumerate(evidence, start=1):
        section = f"--- Pod-twierdzenie {i}: {e['sub_claim']} ---\n"
        if e["context"]:
            section += f"Kontekst referencyjny:\n{e['context']}\n"
        else:
            section += "Brak kontekstu w bazie wiedzy.\n"
        evidence_sections.append(section)

    user_content = json.dumps({
        "statement": original_question,
        "decomposition_and_evidence": "".join(evidence_sections),
    }, ensure_ascii=False)

    answer, total, prompt, completion = _call_llm_json(VERIFIER_PROMPT, user_content)
    log.info("Verifier: '%s' (tokens=%d)", answer[:20], total)
    return answer, total, prompt, completion

AGENT_CONFIG = {
    "name": "dem_ga4",
    "model": model,
    "system_prompt": "Multi-agent: Decomposer → RAG Retriever → Verifier",
    "tools": ["rag_hybrid", "claim_decomposition"],
}

RAG_K_PER_CLAIM: int = 3
RAG_MAX_CONTEXT_CHARS: int = 2000

def ask(question: str) -> dict:
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    sub_claims, t1_total, t1_prompt, t1_comp = decompose_claim(question)
    total_tokens += t1_total
    prompt_tokens += t1_prompt
    completion_tokens += t1_comp

    evidence = retrieve_contexts(
        sub_claims,
        k_per_claim=RAG_K_PER_CLAIM,
        max_context_chars=RAG_MAX_CONTEXT_CHARS,
    )

    answer, t3_total, t3_prompt, t3_comp = verify_claim(question, evidence)
    total_tokens += t3_total
    prompt_tokens += t3_prompt
    completion_tokens += t3_comp

    return {
        "answer": answer,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "sub_claims": sub_claims,
        "evidence_summary": [
            {"sub_claim": e["sub_claim"], "num_results": e["num_results"]}
            for e in evidence
        ],
    }

class ClaimDecompRAGAgent(BaseAgent):
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
            log.error("Błąd w pipeline: %s", exc)
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
        raw_output = json.dumps(
            {
                "answer": result["answer"],
                "sub_claims": result.get("sub_claims", []),
                "evidence_summary": result.get("evidence_summary", []),
            },
            ensure_ascii=False,
            indent=2,
        )

        return {
            "model_label": model_label,
            "original_label": original_label,
            "is_correct": str(model_label).upper() == str(original_label).upper(),
            "total_tokens": result["total_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "time_thought": elapsed,
            "raw_output": raw_output,
        }
