"""
Automatyczny instalator środowiska wirtualnego
================================================

Tworzy .venv w katalogu projektu i instaluje wszystkie zależności
wymagane przez moduły w repozytorium claims_processing.

Kompatybilny z Windows, macOS i Linux.

Użycie
------
    python loader.py           # tworzy .venv i instaluje pakiety
    python loader.py --force   # usuwa istniejący .venv i tworzy od nowa
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"

# Pakiety wymagane przez poszczególne moduły repozytorium:
#
#   agents_uam/*            → together, python-dotenv
#   data/am_benchmark_loader.py → pandas
#   datascrap/*             → cloudscraper, beautifulsoup4
#   dataprep/wikipedia_embedding.py → sentence-transformers, tqdm, numpy
#   dataprep/wikipedia_db.py → sqlite-vec, pysqlite3-binary
#   eval/uam_benchmark_loop.py → pandas, together, python-dotenv
#
REQUIREMENTS: list[str] = [
    # --- API & CLI ---
    "together",
    "python-dotenv",
    # --- Data ---
    "pandas",
    # --- Web scraping ---
    "cloudscraper",
    "beautifulsoup4",
    # --- ML / Embeddings ---
    "sentence-transformers",
    "tqdm",
    "numpy",
    # --- SQLite extensions ---
    "sqlite-vec",
]

if platform.system() != "Windows":
    REQUIREMENTS.append("pysqlite3-binary")


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def _print_header(msg: str) -> None:
    """Wyswietla naglowek sekcji."""
    width = 60
    print()
    print("=" * width)
    print(f"  {msg}")
    print("=" * width)


def _get_venv_python() -> Path:
    """Zwraca sciezke do interpretera Python w .venv (cross-platform)."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _get_venv_pip() -> Path:
    """Zwraca sciezke do pip w .venv (cross-platform)."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def _run(cmd: list[str], description: str) -> None:
    """Uruchamia komende subprocess i wyswietla status."""
    print(f"\n  >> {description}")
    print(f"     {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"\n  [!] Komenda zakonczona bledem (kod: {result.returncode})")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Glowna logika
# ---------------------------------------------------------------------------

def create_venv(force: bool = False) -> None:
    """Tworzy srodowisko wirtualne .venv."""
    if VENV_DIR.exists():
        if force:
            print(f"  Usuwam istniejacy .venv: {VENV_DIR}")
            shutil.rmtree(VENV_DIR)
        else:
            print(f"  .venv juz istnieje: {VENV_DIR}")
            print("  Uzyj --force aby utworzyc od nowa.")
            return

    _run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        "Tworzenie srodowiska wirtualnego .venv",
    )
    print(f"  [OK] .venv utworzony: {VENV_DIR}")


def install_packages() -> None:
    """Instaluje wszystkie zaleznosci w .venv."""
    pip_path = _get_venv_pip()
    python_path = _get_venv_python()

    if not python_path.exists():
        print(f"  [!] Nie znaleziono interpretera: {python_path}")
        print("      Uruchom ponownie z --force aby odtworzyc .venv")
        sys.exit(1)

    # Upgrade pip
    _run(
        [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
        "Aktualizacja pip",
    )

    # Install all requirements
    _run(
        [str(python_path), "-m", "pip", "install"] + REQUIREMENTS,
        f"Instalacja {len(REQUIREMENTS)} pakietow",
    )

    print(f"\n  [OK] Zainstalowano {len(REQUIREMENTS)} pakietow")


def generate_requirements_txt() -> None:
    """Generuje plik requirements.txt w katalogu projektu."""
    req_path = PROJECT_ROOT / "requirements.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("# Zależności projektu claims_processing\n")
        f.write("# Wygenerowano automatycznie przez loader.py\n\n")
        for pkg in REQUIREMENTS:
            f.write(f"{pkg}\n")
    print(f"  [OK] Zapisano requirements.txt: {req_path}")


def verify_installation() -> None:
    """Weryfikuje poprawnosc instalacji — importuje kluczowe pakiety."""
    python_path = _get_venv_python()

    test_imports = [
        "together",
        "dotenv",
        "pandas",
        "cloudscraper",
        "bs4",
        "tqdm",
        "numpy",
    ]

    print("\n  Weryfikacja importow:")
    all_ok = True
    for mod in test_imports:
        result = subprocess.run(
            [str(python_path), "-c", f"import {mod}; print('  [OK] {mod}')"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"  [!] {mod} — blad importu")
            all_ok = False

    # pysqlite3 — opcjonalny, wymagany na macOS/Windows
    result = subprocess.run(
        [str(python_path), "-c", "import pysqlite3; print('  [OK] pysqlite3')"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("  [~] pysqlite3 — niedostepny (uzyje stdlib sqlite3)")

    # sqlite_vec
    result = subprocess.run(
        [str(python_path), "-c", "import sqlite_vec; print('  [OK] sqlite_vec')"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("  [!] sqlite_vec — blad importu (wymagany dla RAG)")
        all_ok = False

    if all_ok:
        print("\n  Wszystkie kluczowe pakiety zainstalowane poprawnie!")
    else:
        print("\n  [!] Niektore pakiety nie zostaly zainstalowane poprawnie.")
        print("      Sprawdz komunikaty bledow powyzej.")


def print_activation_instructions() -> None:
    """Wyswietla instrukcje aktywacji .venv."""
    _print_header("Aktywacja srodowiska")
    system = platform.system()

    if system == "Windows":
        print("  CMD:        .venv\\Scripts\\activate.bat")
        print("  PowerShell: .venv\\Scripts\\Activate.ps1")
    else:
        print("  bash/zsh:   source .venv/bin/activate")
        print("  fish:       source .venv/bin/activate.fish")

    print(f"\n  Python:     {_get_venv_python()}")
    print(f"  Pip:        {_get_venv_pip()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tworzy .venv i instaluje zaleznosci projektu claims_processing."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Usun istniejacy .venv i utworz od nowa.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Pomin weryfikacje importow po instalacji.",
    )
    args = parser.parse_args()

    _print_header("claims_processing — loader.py")
    print(f"  System:  {platform.system()} {platform.release()}")
    print(f"  Python:  {sys.version}")
    print(f"  Projekt: {PROJECT_ROOT}")

    # 1. Srodowisko wirtualne
    _print_header("Tworzenie .venv")
    create_venv(force=args.force)

    # 2. Instalacja pakietow
    _print_header("Instalacja pakietow")
    install_packages()

    # 3. requirements.txt
    _print_header("Generowanie requirements.txt")
    generate_requirements_txt()

    # 4. Weryfikacja
    if not args.skip_verify:
        _print_header("Weryfikacja instalacji")
        verify_installation()

    # 5. Instrukcje
    print_activation_instructions()

    _print_header("Gotowe!")
    print("  Srodowisko jest gotowe do pracy.\n")


if __name__ == "__main__":
    main()
