"""
Skrypt do ewaluacji benchmarku UAM.
Iteruje po pytaniach z am_benchmark.csv, wysyła je do wybranego agenta
(Together AI / openai/gpt-oss-20b) i zapisuje wyniki do CSV.

Struktura wynikowego CSV:
    pytanie, odpowiedz_oryginalna, odpowiedz_agenta, tokeny_uzyte, czas_odpowiedzi_s

Użycie:
    python eval/uam_benchmark_loop.py                       # domyślnie agent ga1, wszystkie pytania
    python eval/uam_benchmark_loop.py --agent ga2           # agent z web_search
    python eval/uam_benchmark_loop.py --limit 50            # pierwsze 50 pytań
    python eval/uam_benchmark_loop.py --agent ga2 --limit 10
"""

import ast
import os
import sys
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from together import Together

load_dotenv()

# Reconfigure stdout for UTF-8 on Windows (prevents UnicodeEncodeError
# when printing emoji like '⚠' in cmd.exe / PowerShell)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Konfiguracja API
# ---------------------------------------------------------------------------
api_key = os.getenv("together_api_key")
if not api_key:
    print("⚠ Brak klucza API! Ustaw zmienną together_api_key w pliku .env")
    sys.exit(1)

client = Together(api_key=api_key)
MODEL = "openai/gpt-oss-20b"

# ---------------------------------------------------------------------------
# Definicje agentów (system prompts)
# ---------------------------------------------------------------------------
AGENTS = {
    "ga1": {
        "name": "uam_ga1",
        "system_prompt": (
            "Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi "
            "bez wykorzystania jakichkolwiek narzędzi.\n"
            "Input: Wypowiedź/pytanie którego prawdziwość masz ocenić wraz z 4 opcjami do wyboru\n"
            "Instructions: Dokonaj oceny prawdziwości wypowiedzi/pytania i wybierz "
            "najbardziej odpowiednią opcję. Nie wykorzystujesz żadnych narzędzi "
            "i masz polegać tylko na swojej wiedzy ogólnej.\n"
            "Output: 0, 1, 2 or 3"
        ),
    },
    "ga2": {
        "name": "uam_ga2",
        "system_prompt": (
            "Jesteś agentem który ma za zadanie ocenić prawdziwość wypowiedzi "
            "z wykorzystaniem narzędzi.\n"
            "Input: Wypowiedź/pytanie którego prawdziwość masz ocenić wraz z 4 opcjami do wyboru\n"
            "Instructions: Dokonaj oceny prawdziwości wypowiedzi/pytania i wybierz "
            "najbardziej odpowiednią opcję. Wykorzystujesz dostępne narzędzia "
            "do wyszukiwania informacji.\n"
            "Output: 0, 1, 2 or 3"
        ),
    },
}


def format_question(row: pd.Series) -> str:
    """Formatuje pytanie wraz z opcjami odpowiedzi do postaci tekstowej."""
    question = row["question"]
    answers_raw = row["answers"]

    # Odpowiedzi są zapisane jako string w formacie listy Pythona
    try:
        answers = ast.literal_eval(answers_raw)
    except (ValueError, SyntaxError):
        answers = [answers_raw]

    # Budowanie promptu z pytaniem i opcjami
    prompt = f"Pytanie: {question}\n\nOpcje:\n"
    for idx, ans in enumerate(answers):
        prompt += f"  {idx}: {ans}\n"
    prompt += "\nOdpowiedz TYLKO numerem opcji (0, 1, 2 lub 3)."

    return prompt


def get_correct_answer(row: pd.Series) -> str:
    """Zwraca indeks poprawnej odpowiedzi jako string."""
    return str(int(row["correct_answer_index"]))


def run_agent_on_question(system_prompt: str, prompt: str) -> tuple[str, int]:
    """
    Wysyła pytanie do Together AI i zwraca (odpowiedź, łączna liczba tokenów).
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    # Odpowiedź
    answer = response.choices[0].message.content.strip()

    # Zużycie tokenów
    total_tokens = 0
    if response.usage:
        total_tokens = response.usage.total_tokens or 0

    return answer, total_tokens


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja benchmarku UAM (Together AI)")
    parser.add_argument(
        "--agent",
        type=str,
        default="ga1",
        choices=["ga1", "ga2"],
        help="Wybór agenta: ga1 (bez narzędzi) lub ga2 (z web_search). Domyślnie: ga1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maksymalna liczba pytań do ewaluacji. Domyślnie: wszystkie",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ścieżka do pliku wynikowego CSV. Domyślnie: eval/wyniki_{agent}_{timestamp}.csv",
    )
    args = parser.parse_args()

    # Ładowanie benchmarku
    benchmark_path = str(Path(__file__).resolve().parent.parent / "data" / "am_benchmark.csv")
    df = pd.read_csv(benchmark_path)
    print(f"Załadowano benchmark: {len(df)} pytań")
    print(f"Model: {MODEL}")

    if args.limit:
        df = df.head(args.limit)
        print(f"Ograniczono do {args.limit} pytań")

    # Wybór agenta
    agent_cfg = AGENTS[args.agent]
    agent_name = args.agent
    print(f"Wybrany agent: {agent_name} ({agent_cfg['name']})")

    # Ścieżka do pliku wynikowego
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(
            Path(__file__).resolve().parent / f"wyniki_{agent_name}_{timestamp}.csv"
        )

    # Otwarcie pliku wynikowego i start ewaluacji
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pytanie", "odpowiedz_oryginalna", "odpowiedz_agenta", "tokeny_uzyte", "czas_odpowiedzi_s"])

        total = len(df)
        correct_count = 0
        total_tokens_all = 0

        for idx, row in df.iterrows():
            prompt = format_question(row)
            correct_answer = get_correct_answer(row)

            print(f"\n[{idx + 1}/{total}] Przetwarzanie pytania...")
            print(f"  Pytanie: {row['question'][:80]}...")

            tokens_used = 0
            t_start = time.perf_counter()

            try:
                agent_answer, tokens_used = run_agent_on_question(
                    system_prompt=agent_cfg["system_prompt"],
                    prompt=prompt,
                )
            except Exception as e:
                agent_answer = f"BŁĄD: {str(e)}"
                print(f"  ⚠ Błąd: {e}")

            elapsed = round(time.perf_counter() - t_start, 2)
            total_tokens_all += tokens_used

            # Zapis wiersza do CSV
            writer.writerow([row["question"], correct_answer, agent_answer, tokens_used, elapsed])
            csvfile.flush()  # Zapisuj na bieżąco

            # Prosta weryfikacja czy odpowiedź zawiera poprawny numer
            if correct_answer in agent_answer:
                correct_count += 1

            print(f"  Poprawna: {correct_answer} | Agent: {agent_answer[:50]}")
            print(f"  Tokeny: {tokens_used} | Czas: {elapsed}s")
            print(
                f"  Trafność dotychczasowa: {correct_count}/{idx + 1}"
                f" ({correct_count / (idx + 1) * 100:.1f}%)"
            )

    print(f"\n{'=' * 60}")
    print(f"Ewaluacja zakończona!")
    print(f"Wyniki zapisano do: {output_path}")
    print(f"Łączna trafność: {correct_count}/{total} ({correct_count / total * 100:.1f}%)")
    print(f"Łączne zużycie tokenów: {total_tokens_all}")


if __name__ == "__main__":
    main()
