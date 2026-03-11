import os
from dotenv import load_dotenv
from gen_agent.bm25 import BM25Index

from gen_agent.llm_client import client, MODEL

load_dotenv()

"""
Agent open-book z wyszukiwaniem BM25 w lokalnej bazie Wikipedii
Brak narzędzi online — kontekst pochodzi z offline'owego indeksu BM25
Do benchmarku z UAMu
Wykorzystuje Together AI API z modelem openai/gpt-oss-20b
"""

model = MODEL

# ---------------------------------------------------------------------------
# Ładowanie indeksu BM25 (lazy — przy pierwszym zapytaniu)
# ---------------------------------------------------------------------------

_WIKI_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataprep",
    "wiki.db",
)

_bm25_index: BM25Index | None = None


def _get_bm25() -> BM25Index:
    """Lazy-loading indeksu BM25 — buduje się raz przy pierwszym wywołaniu."""
    global _bm25_index
    if _bm25_index is None:
        db_path = os.getenv("WIKI_BM25_DB", _WIKI_DB_PATH)
        _bm25_index = BM25Index.from_sqlite(db_path)
    return _bm25_index


# ---------------------------------------------------------------------------
# Konfiguracja agenta
# ---------------------------------------------------------------------------

AGENT_CONFIG = {
    "name": "uam_ga3",
    "model": model,
    "system_prompt": """Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi z wykorzystaniem dostarczonego kontekstu z Wikipedii.
    Input: Wypowiedź/pytanie którego prawdziwość masz ocenić wraz z 4 opcjami do wyboru oraz kontekst z Wikipedii.
    Instructions: Dokonaj oceny prawdziwości wypowiedzi/pytania i wybierz najbardziej odpowiednią opcję. 
    Wykorzystujesz dostarczony kontekst z Wikipedii do weryfikacji informacji.
    Jeśli kontekst nie zawiera wystarczających informacji, polegaj na swojej wiedzy ogólnej.
    Output: 0, 1, 2 or 3""",
    "tools": ["bm25_wikipedia"],  # BM25 Wikipedia retrieval
}

# ---------------------------------------------------------------------------
# Parametry retrieval
# ---------------------------------------------------------------------------

BM25_TOP_K: int = 5
BM25_MAX_CONTEXT_CHARS: int = 3000


def ask(question: str) -> dict:
    """Wysyła pytanie do agenta z kontekstem BM25 i zwraca odpowiedź wraz z metadanymi."""

    # 1. Wyszukiwanie kontekstu BM25
    bm25 = _get_bm25()
    context = bm25.search_and_format(
        question, k=BM25_TOP_K, max_context_chars=BM25_MAX_CONTEXT_CHARS
    )

    # 2. Budowanie promptu z kontekstem
    if context:
        user_content = (
            f"Kontekst z Wikipedii:\n{context}\n\n"
            f"Pytanie:\n{question}"
        )
    else:
        user_content = question

    # 3. Wywołanie LLM
    response = client.chat.completions.create(
        model=AGENT_CONFIG["model"],
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
