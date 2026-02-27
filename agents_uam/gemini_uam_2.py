import os
from dotenv import load_dotenv
from together import Together

load_dotenv()

"""
Najbardziej podstawowa wersja agenta
Z dostępem do web_search (Together AI)
Wykorzystywany do benchmarku z UAMu
Wykorzystuje Together AI API z modelem openai/gpt-oss-20b
"""

client = Together(api_key=os.getenv("together_api_key"))

AGENT_CONFIG = {
    "name": "uam_ga2",
    "model": "openai/gpt-oss-20b",
    "system_prompt": """Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi z wykorzystaniem narzędzi.
    Input: Wypowiedź/pytanie którego prawdziwość masz ocenić wraz z 4 opcjami do wyboru
    Instructions: Dokonaj oceny prawdziwości wypowiedzi/pytania i wybierz najbardziej odpowiednią opcję. 
    Wykorzystujesz dostępne narzędzia do wyszukiwania informacji.
    Output: 0, 1, 2 or 3""",
    "tools": ["web_search"],  # Together AI web search
}


def ask(question: str) -> dict:
    """Wysyła pytanie do agenta i zwraca odpowiedź wraz z metadanymi."""
    response = client.chat.completions.create(
        model=AGENT_CONFIG["model"],
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