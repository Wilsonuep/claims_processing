import sys
import time

from test_01_wikipedia_db import test_wikipedia_db
from test_03_am_benchmark_db import test_am_benchmark_db
from test_04_eval_local import test_eval_local
from test_05_eval_cloud import test_eval_cloud
from test_06_cuda_gpu import test_cuda_gpu
from test_07_monitoring import test_monitoring
from test_08_crash_recovery import test_crash_recovery
from test_09_bm25_polish import test_bm25_polish
from test_10_monitoring_progress import test_monitoring_progress
from test_10b_bm25_cache import test_bm25_cache
from test_11_am_agent_config import test_am_agent_config

def run_tests():
    print("=" * 60)
    print("Rozpoczynanie testów integracyjnych (Integrations & Eval Pipeline)")
    print("=" * 60)

    tests = [
        ("Rozbicie i wektoryzacja Wikipedii do bazy", test_wikipedia_db),
        ("Załadunek AM Benchmark CSV do bazy DB", test_am_benchmark_db),
        ("Evaluacja agentowa (tryb lokalny - tiered)", test_eval_local),
        ("Evaluacja agentowa (tryb w chmurze - parallel)", test_eval_cloud),
        ("CUDA GPU — dostępność i wydajność NVIDIA", test_cuda_gpu),
        ("Monitoring - powiadomienia i uaktualnienia (brrr)", test_monitoring),
        ("Crash recovery i resume pętli ewaluacyjnej", test_crash_recovery),
        ("BM25 tokenizer — polska morfologia i stopwords", test_bm25_polish),
        ("Monitoring — dokładność progressu i ETA (live payload check)", test_monitoring_progress),
        ("BM25 index cache — OOM regression (jeden obiekt na proces)", test_bm25_cache),
        ("AM Benchmark agenci — konfiguracja label i odpowiedzi", test_am_agent_config),
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
