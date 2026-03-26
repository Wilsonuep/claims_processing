"""
test_06_cuda_gpu.py
====================

Weryfikuje dostępność i wydajność karty graficznej NVIDIA (CUDA) na
maszynie ewaluacyjnej.

Sprawdzenia:
    1. Dostępność biblioteki torch z obsługą CUDA.
    2. Wykrycie co najmniej jednej karty CUDA (nvidia-smi / torch).
    3. Poprawność podstawowych operacji tensorowych na GPU.
    4. Przepustowość mnożenia macierzy (GEMM) – referencyjna miara
       zdolności GPU do zadań embeddingowych / LLM.
    5. Zużycie pamięci VRAM przez duże tensory.
    6. Poprawność wyników GPU vs CPU (numeryczna weryfikacja).

Konwencja powrotu (zgodna z resztą plików w /tests):
    success : bool   – True = test zdany
    elapsed : float  – czas wykonania w sekundach
    err     : str|None – opis błędu lub None gdy sukces

Uruchomienie:
    python tests/test_06_cuda_gpu.py          # standalone
    python tests/tester.py                    # przez runner (po dodaniu do listy)
"""

from __future__ import annotations

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Stałe konfiguracyjne
# ---------------------------------------------------------------------------

# Minimalna wymagana wolna pamięć VRAM w MB
MIN_FREE_VRAM_MB: int = 500

# Rozmiar macierzy do testu GEMM (N×N, float32)
# 4096 × 4096 × 4B ≈ 64 MB → bezpieczne nawet dla kart 2 GB VRAM
GEMM_SIZE: int = 4096

# Minimalna oczekiwana przepustowość GEMM w GFLOPS
# ~500 GFLOPS to dolna granica dla starej karty klasy GTX 1060;
# RTX 3090 osiąga ~35 000 GFLOPS w fp32.
MIN_GFLOPS: float = 200.0

# Tolerancja numeryczna (GPU fp32 vs CPU fp64)
NUMERIC_TOLERANCE: float = 1e-2


# ---------------------------------------------------------------------------
# Pomocnicze funkcje
# ---------------------------------------------------------------------------

def _get_device_info(device_idx: int = 0) -> dict:
    """Zbiera informacje o karcie CUDA o podanym indeksie."""
    import torch

    props = torch.cuda.get_device_properties(device_idx)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    return {
        "name": props.name,
        "total_vram_mb": total_bytes // (1024 ** 2),
        "free_vram_mb": free_bytes // (1024 ** 2),
        "cuda_capability": f"{props.major}.{props.minor}",
        "multiprocessors": props.multi_processor_count,
    }


def _run_gemm_benchmark(size: int, device: str) -> float:
    """
    Zwraca szacowaną przepustowość GEMM w GFLOPS.

    Operacja: C = A @ B, gdzie A, B są macierzami float32 rozmiaru size×size.
    Liczba operacji zmiennoprzecinkowych (FLOPs) dla mnożenia macierzy N×N:
        2 * N^3
    """
    import torch

    a = torch.randn(size, size, dtype=torch.float32, device=device)
    b = torch.randn(size, size, dtype=torch.float32, device=device)

    # Rozgrzewka – synchronizacja przed pomiarem
    _ = torch.mm(a, b)
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    c = torch.mm(a, b)
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    flops = 2.0 * (size ** 3)
    gflops = flops / elapsed / 1e9

    # Zwolnienie pamięci GPU
    del a, b, c
    if device != "cpu":
        torch.cuda.empty_cache()

    return gflops


def _verify_numeric_correctness(size: int = 512) -> tuple[bool, float]:
    """
    Sprawdza zgodność wyników GPU i CPU dla małej macierzy.

    Zwraca (ok: bool, max_diff: float).
    """
    import torch

    a = torch.randn(size, size, dtype=torch.float32)
    b = torch.randn(size, size, dtype=torch.float32)

    cpu_result = torch.mm(a, b)
    gpu_result = torch.mm(a.cuda(), b.cuda()).cpu()

    max_diff = float((cpu_result - gpu_result).abs().max().item())
    ok = max_diff < NUMERIC_TOLERANCE
    return ok, max_diff


# ---------------------------------------------------------------------------
# Główna funkcja testowa
# ---------------------------------------------------------------------------

def test_cuda_gpu() -> tuple[bool, float, str | None]:
    """
    Kompleksowy test CUDA GPU.

    Zwraca
    -------
    (success, elapsed, error_message)
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Import torch
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        return False, time.time() - start_time, (
            "PyTorch nie jest zainstalowany. "
            "Uruchom: pip install torch"
        )

    # ------------------------------------------------------------------
    # 2. Dostępność CUDA
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        return False, time.time() - start_time, (
            "CUDA niedostępne. Sprawdź sterownik NVIDIA i instalację "
            "PyTorch z obsługą CUDA (np. pip install torch --index-url "
            "https://download.pytorch.org/whl/cu121)."
        )

    # ------------------------------------------------------------------
    # 3. Informacje o karcie
    # ------------------------------------------------------------------
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False, time.time() - start_time, "Brak kart CUDA (device_count=0)."

        info = _get_device_info(0)
    except Exception as e:
        return False, time.time() - start_time, f"Błąd przy odczycie informacji o GPU: {e}"

    # ------------------------------------------------------------------
    # 4. Minimalna wolna pamięć VRAM
    # ------------------------------------------------------------------
    if info["free_vram_mb"] < MIN_FREE_VRAM_MB:
        return False, time.time() - start_time, (
            f"Za mało wolnej pamięci VRAM: {info['free_vram_mb']} MB "
            f"(wymagane ≥ {MIN_FREE_VRAM_MB} MB). "
            "Zamknij inne procesy korzystające z GPU."
        )

    # ------------------------------------------------------------------
    # 5. Podstawowe operacje tensorowe na GPU
    # ------------------------------------------------------------------
    try:
        device = torch.device("cuda:0")

        # Tworzenie tensorów
        t = torch.ones(1000, 1000, dtype=torch.float32, device=device)
        if not torch.all(t == 1.0).item():
            return False, time.time() - start_time, "Błędne wartości tensora ones() na GPU."

        # Operacja elementarna
        t2 = t * 2.0 + 1.0
        expected_val = 3.0
        if abs(float(t2[0, 0].item()) - expected_val) > 1e-5:
            return False, time.time() - start_time, (
                f"Błędny wynik operacji arytmetycznej na GPU: "
                f"oczekiwano {expected_val}, otrzymano {t2[0,0].item()}"
            )

        del t, t2
        torch.cuda.empty_cache()
    except Exception as e:
        return False, time.time() - start_time, f"Błąd operacji tensorowej na GPU: {e}"

    # ------------------------------------------------------------------
    # 6. Test GEMM (przepustowość)
    # ------------------------------------------------------------------
    try:
        gflops = _run_gemm_benchmark(GEMM_SIZE, device="cuda:0")
    except Exception as e:
        return False, time.time() - start_time, f"Błąd benchmarku GEMM: {e}"

    if gflops < MIN_GFLOPS:
        return False, time.time() - start_time, (
            f"Zbyt niska przepustowość GEMM: {gflops:.1f} GFLOPS "
            f"(wymagane ≥ {MIN_GFLOPS} GFLOPS). GPU może być przeciążone."
        )

    # ------------------------------------------------------------------
    # 7. Weryfikacja numeryczna GPU vs CPU
    # ------------------------------------------------------------------
    try:
        numerically_ok, max_diff = _verify_numeric_correctness(size=512)
    except Exception as e:
        return False, time.time() - start_time, f"Błąd weryfikacji numerycznej: {e}"

    if not numerically_ok:
        return False, time.time() - start_time, (
            f"Wyniki GPU i CPU różnią się zbyt mocno: max_diff={max_diff:.6f} "
            f"(tolerancja={NUMERIC_TOLERANCE}). Możliwy błąd sterownika."
        )

    # ------------------------------------------------------------------
    # Wszystko OK — raport
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    return True, elapsed, None


# ---------------------------------------------------------------------------
# Raport szczegółowy (wywoływany w trybie standalone)
# ---------------------------------------------------------------------------

def _print_full_report() -> None:
    """Wyświetla pełny raport CUDA dla uruchomienia standalone."""
    try:
        import torch

        print()
        print("=" * 65)
        print("  RAPORT DIAGNOSTYCZNY CUDA GPU")
        print("=" * 65)

        print(f"  PyTorch:          {torch.__version__}")
        print(f"  CUDA dostępne:    {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("  Brak kart CUDA — przerwano raport.")
            print("=" * 65)
            return

        print(f"  Liczba kart GPU:  {torch.cuda.device_count()}")
        print()

        for idx in range(torch.cuda.device_count()):
            info = _get_device_info(idx)
            print(f"  ─── GPU #{idx}: {info['name']} ───")
            print(f"    CUDA Capability: {info['cuda_capability']}")
            print(f"    Multiprocessors: {info['multiprocessors']}")
            print(f"    VRAM całkowita:  {info['total_vram_mb']:,} MB")
            print(f"    VRAM wolna:      {info['free_vram_mb']:,} MB")

        print()
        print(f"  ─── Benchmark GEMM ({GEMM_SIZE}×{GEMM_SIZE}, float32) ───")
        gflops = _run_gemm_benchmark(GEMM_SIZE, device="cuda:0")
        print(f"    Przepustowość:   {gflops:,.1f} GFLOPS")
        print(f"    Min. wymagane:   {MIN_GFLOPS:,.1f} GFLOPS")
        print(f"    Status:          {'✅ OK' if gflops >= MIN_GFLOPS else '❌ ZBYT WOLNE'}")

        print()
        print("  ─── Weryfikacja numeryczna GPU vs CPU (512×512) ───")
        ok, diff = _verify_numeric_correctness(512)
        print(f"    Max. różnica:    {diff:.2e}")
        print(f"    Tolerancja:      {NUMERIC_TOLERANCE:.2e}")
        print(f"    Status:          {'✅ OK' if ok else '❌ BŁĄD'}")

        print()
        print("=" * 65)

    except ImportError:
        print("  [BŁĄD] PyTorch nie jest zainstalowany.")
        print("=" * 65)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _print_full_report()

    success, elapsed, err = test_cuda_gpu()
    if success:
        print(f"PASSED (Time: {elapsed:.2f}s)")
    else:
        print(f"FAILED (Time: {elapsed:.2f}s, Error: {err})")
