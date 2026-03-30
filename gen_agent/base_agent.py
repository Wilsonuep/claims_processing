"""
Abstrakcyjna klasa bazowa agenta (BaseAgent)
=============================================

Każdy agent ewaluowany przez pętlę ewaluacyjną musi dziedziczyć
po ``BaseAgent`` i zaimplementować metodę ``eval()``.

Użycie
------
    from gen_agent.base_agent import BaseAgent

    class MojAgent(BaseAgent):
        name = "moj_agent"

        def eval(self, claim: dict) -> dict:
            ...

Wymaga
------
    Python 3.10+
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Wymagane klucze w słowniku zwracanym przez eval()
# ---------------------------------------------------------------------------

REQUIRED_RESULT_KEYS: frozenset[str] = frozenset(
    [
        "model_label",
        "original_label",
        "is_correct",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "time_thought",
        "raw_output",
    ]
)


# ---------------------------------------------------------------------------
# Klasa bazowa agenta
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Interfejs bazowy dla agentów poddawanych ewaluacji.

    Atrybuty
    --------
    name : str
        Unikalna nazwa agenta — używana w logach i tabeli wyników.
    """

    name: str
    cost_tier: int = 2  # 1=fast (≤1 LLM call), 2=moderate (2 calls), 3=expensive (4+ calls)

    @abstractmethod
    def eval(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Ewaluacja pojedynczego twierdzenia.

        Parametry
        ---------
        claim : dict
            Słownik reprezentujący wiersz z tabeli claims benchmarku.
            Zawiera co najmniej: ``id``, ``claim_text``, ``label``
            (= ground truth).

        Zwraca
        ------
        dict
            Słownik z co najmniej kluczami:
            - ``model_label``       (str)  — etykieta przewidziana przez agenta
            - ``original_label``    (str)  — etykieta ground-truth (pass-through)
            - ``is_correct``        (bool) — czy predykcja jest poprawna
            - ``total_tokens``      (int)  — łączna liczba tokenów
            - ``prompt_tokens``     (int)  — tokeny promptu
            - ``completion_tokens`` (int)  — tokeny odpowiedzi
            - ``time_thought``      (float)— czas ewaluacji w sekundach
            - ``raw_output``        (str)  — pełna odpowiedź modelu / uzasadnienie
        """


# ---------------------------------------------------------------------------
# Walidacja wyniku
# ---------------------------------------------------------------------------


def validate_result(result: dict[str, Any], agent_name: str) -> None:
    """Sprawdza, czy wynik agenta zawiera wszystkie wymagane klucze.

    Parametry
    ---------
    result : dict
        Słownik zwrócony przez ``agent.eval()``.
    agent_name : str
        Nazwa agenta — do komunikatów o błędach.

    Rzuca
    -----
    ValueError
        Gdy brakuje wymaganego klucza.
    """
    missing = REQUIRED_RESULT_KEYS - result.keys()
    if missing:
        raise ValueError(
            f"Agent '{agent_name}' zwrócił wynik bez kluczy: {sorted(missing)}"
        )
