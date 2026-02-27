import os
from dotenv import load_dotenv
from together import Together

load_dotenv()

"""
Najbardziej podstawowa wersja agenta
Brak narzędzi
Do benchmarku z Demagoga
Wykorzystuje Together AI API z modelem openai/gpt-oss-20b
"""

client = Together(api_key=os.getenv("together_api_key"))

AGENT_CONFIG = {
    "name": "dem_ga1",
    "model": "openai/gpt-oss-20b",
    "system_prompt": """Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi polityka bez wykorzystania jakichkolwiek narzędzi.
    Input: Wypowiedź polityka której prawdziwość masz ocenić.
    Instructions: Dokonaj oceny prawdziwości wypowiedzi polityka. 
    Nie wykorzystujesz żadnych narzędzi i masz polegać tylko na swojej wiedzy ogólnej.
    Output: Prawda, Fałsz, Manipulacja, Brak danych""",
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