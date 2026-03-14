import sys
import time

from test_01_wikipedia_db import test_wikipedia_db
from test_02_demagog_db import test_demagog_db
from test_03_am_benchmark_db import test_am_benchmark_db
from test_04_eval_local import test_eval_local
from test_05_eval_cloud import test_eval_cloud

def run_tests():
    print("=" * 60)
    print("Rozpoczynanie testów integracyjnych (Integrations & Eval Pipeline)")
    print("=" * 60)
    
    tests = [
        ("Rozbicie i wektoryzacja Wikipedii do bazy", test_wikipedia_db),
        ("Załadunek wyników Demagog JSON do bazy DB", test_demagog_db),
        ("Załadunek AM Benchmark CSV do bazy DB", test_am_benchmark_db),
        ("Evaluacja agentowa (tryb lokalny - tiered)", test_eval_local),
        ("Evaluacja agentowa (tryb w chmurze - parallel)", test_eval_cloud)
    ]
    
    all_passed = True
    for name, test_func in tests:
        # print function name right aligned for readibility
        print(f"[{name}]".ljust(55), end="", flush=True)
        try:
            success, elapsed, err = test_func()
            if success:
                print(f"PASSED ({elapsed:.3f}s)")
            else:
                print(f"FAILED ({elapsed:.3f}s)")
                print(f"    - Błąd: {err}")
                all_passed = False
        except Exception as e:
            print(f"FAILED (Exception: {e})")
            all_passed = False
            
    print("=" * 60)
    if all_passed:
        print("WSZYSTKIE TESTY ZAKOŃCZONE POMYŚLNIE! (\u2713)")
        sys.exit(0)
    else:
        print("NIEKTÓRE TESTY ZAKOŃCZYŁY SIĘ NIEPOWODZENIEM! (\u2717)")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
