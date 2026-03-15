import os
import ast
import sqlite3
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def find_agent_names(directory: Path) -> list[str]:
    """Skanuje pliki .py w poszukiwaniu klas dziedziczących z BaseAgent i atrybutu name."""
    names = set()
    if not directory.exists():
        return list(names)
        
    for p in directory.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                    # Szukamy stringów uam_... i dem_... we wszystkich węzłach
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        val = node.value
                        prefix = "uam_" if "uam" in directory.name else "dem_"
                        if val.startswith(prefix):
                            names.add(val)
        except Exception as e:
            print(f"Błąd parsowania {p.name}: {e}")
            
    return sorted(list(names))

def get_total_claims(db_path: Path) -> int:
    """Zwraca liczbę wszystkich rekordów w tabeli claims."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM claims")
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        if 'conn' in locals():
            conn.close()

def get_agent_completed_claims(results_db_path: Path, agent_name: str, benchmark_name: str) -> int:
    """Zwraca liczbę unikalnych rozwiązanych claim_id (bez błędów)."""
    if not results_db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(results_db_path)
        cur = conn.cursor()
        
        # Sprawdzamy czy w ogóle jest taka tabela
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_results'")
        if not cur.fetchone():
            return 0
            
        cur.execute('''
            SELECT COUNT(DISTINCT claim_id) 
            FROM agent_results 
            WHERE agent_name = ? AND benchmark_name = ? AND model_label != 'ERROR'
        ''', (agent_name, benchmark_name))
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        if 'conn' in locals():
            conn.close()

def print_report(title: str, total_expected: int, agents: list[str], results_db_path: Path, benchmark_name: str):
    print(f"\n{'='*70}")
    print(f"[{title}]")
    print(f" Oczekiwana liczba wierszy (twierdzeń): {total_expected}")
    print(f" Czerpie z bazy wynikowej: {results_db_path.name}")
    print(f"{'='*70}")
    
    if total_expected == 0:
        print(" OSTRZEŻENIE: Baza testowa ma 0 rekordów lub nie istnieje!")
        
    print(f"{'AGENT NAME':<25} | {'GOTOWE':<10} | {'STATUS'}")
    print("-" * 70)
    
    all_completed = True
    
    for agent in agents:
        completed = get_agent_completed_claims(results_db_path, agent, benchmark_name)
        if total_expected == 0:
            status = "❔ BRAK BAZY WEJŚCIOWEJ"
            all_completed = False
        elif completed == total_expected:
            status = "✅ ZAKOŃCZONO"
        else:
            status = f"❌ BRAKUJE {total_expected - completed}"
            all_completed = False
            
        print(f"{agent:<25} | {completed:<10} | {status}")
        
    print("-" * 70)
    if all_completed and agents:
        print("🌟 Wszystkie agenty w tej grupie zakończyły ewaluację!")
    elif not agents:
        print("⚠️ Brak zdefiniowanych agentów w tym folderze.")
    else:
        print("⚠️ Niektórzy agenci nie ukończyli jeszcze wszystkich zadań.")

def main():
    parser = argparse.ArgumentParser(description="Test kompletności ewaluacji agentów.")
    parser.add_argument("--results-db", type=str, default=None, 
                        help="Ścieżka do niestandardowej bazy wynikowej (np. merged_eval.db). "
                             "Domyślnie używa standardowych baz dla Demagog i AM Benchmark.")
    args = parser.parse_args()

    # Lokacje folderów
    uam_dir = PROJECT_ROOT / "agents_uam"
    dem_dir = PROJECT_ROOT / "agents_dem"
    
    # Lokacje baz
    am_bench_input = PROJECT_ROOT / "dataprep" / "am_benchmark.db"
    demagog_input = PROJECT_ROOT / "dataprep" / "demagog.db"
    
    if args.results_db:
        # User explicitly passed a merged database
        results_db_path_am = Path(args.results_db)
        results_db_path_dem = Path(args.results_db)
    else:
        # Standard workflow databases
        results_db_path_am = PROJECT_ROOT / "results" / "results_am_benchmark.db"
        results_db_path_dem = PROJECT_ROOT / "results" / "results_demagog.db"

    # Wykrywamy agentów
    uam_agents = find_agent_names(uam_dir)
    dem_agents = find_agent_names(dem_dir)
    
    # Odpytujemy oryginalne bazy o totale
    am_total = get_total_claims(am_bench_input)
    demagog_total = get_total_claims(demagog_input)
    
    # Raport UAM
    print_report("AGENCI UAM (am_benchmark)", am_total, uam_agents, results_db_path_am, "am_benchmark")
    
    # Raport DEMAGOG
    print_report("AGENCI DEMAGOG (demagog)", demagog_total, dem_agents, results_db_path_dem, "demagog")
    print("\n")

if __name__ == "__main__":
    main()
