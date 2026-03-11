import os
from dotenv import load_dotenv

from gen_agent.llm_client import client, MODEL

load_dotenv()

"""
Najbardziej podstawowa wersja agenta
Brak narzędzi
Do benchmarku z UAMu
Wykorzystuje Together AI API z modelem openai/gpt-oss-20b
"""

model = MODEL

AGENT_CONFIG = {
    "name": "uam_ga1",
    "model": model,
    "system_prompt": """Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi bez wykorzystania jakichkolwiek narzędzi.
    Input: Wypowiedź/pytanie którego prawdziwość masz ocenić wraz z 4 opcjami do wyboru
    Instructions: Dokonaj oceny prawdziwości wypowiedzi/pytania i wybierz najbardziej odpowiednią opcję. Nie wykorzystujesz żadnych narzędzi i masz polegać tylko na swojej wiedzy ogólnej.
    Output: 0, 1, 2 or 3""",
    "tools": [],  # brak narzędzi
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