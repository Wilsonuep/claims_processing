# Podsumowanie wyników — AM benchmark (agenci × modele)

Wygenerowano 2026-07-10 przez `python tools/generate_results_summary.py`. Nie edytować ręcznie.

Aktywni agenci **uam_ga1–uam_ga5**; zarchiwizowani (`uam_ga_web_tool_arch`, `uam_ga_rag_decomp_arch`) są wykluczeni. Wszyscy agenci z retrievalem (ga2–ga5) używają BM25 — `RAG_MODE` nigdy nie był ustawiony na vector/hybrid. Etykiety modeli w stylu skróconych tagów Ollama.

## Pełny benchmark (`results/results_am_benchmark.db`)

Bielik i llama3.1 pokrywają pełne 18 820 pytań; qwen2.5 i PLLuM tylko podzbiór ~4 000 pytań (seed 42) — kolumn nie należy porównywać 1:1 w tej tabeli.

### Dokładność (%)

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                     45   |                    26.7 |          35.3 |         43   |
| ga2 Single + BM25      |                     43.6 |                    30.9 |          35.4 |         40.7 |
| ga3 Decomp + BM25      |                     45.9 |                    32.9 |          28.6 |         43.5 |
| ga4 FewShot CoT        |                     29.2 |                    29.9 |          29   |         28.4 |
| ga5 Debate             |                     24   |                    19.6 |          26.2 |         28.9 |

### Średnia liczba tokenów (total) na pytanie

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                      645 |                     329 |           431 |          282 |
| ga2 Single + BM25      |                    1,154 |                     893 |           976 |          900 |
| ga3 Decomp + BM25      |                    2,414 |                   1,388 |         1,705 |        1,282 |
| ga4 FewShot CoT        |                   17,168 |                  10,884 |        14,094 |       10,615 |
| ga5 Debate             |                   27,947 |                  15,043 |        23,942 |       18,798 |

### Średni czas inferencji na pytanie (s)

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                      4.5 |                     0.6 |           1.7 |          0.6 |
| ga2 Single + BM25      |                      3.7 |                     2.9 |           2.5 |          2.4 |
| ga3 Decomp + BM25      |                      7.6 |                     2.7 |           2.4 |          1.6 |
| ga4 FewShot CoT        |                     46.5 |                    16.9 |          22.3 |         17.5 |
| ga5 Debate             |                     61   |                    17.7 |          32.8 |         31.9 |

### Pokrycie (liczba wierszy wyników)

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                   18,820 |                   4,005 |        18,820 |        4,006 |
| ga2 Single + BM25      |                   18,820 |                   4,000 |        18,820 |        4,000 |
| ga3 Decomp + BM25      |                   18,820 |                   4,000 |        18,820 |        4,000 |
| ga4 FewShot CoT        |                   18,820 |                   4,000 |        18,820 |        4,000 |
| ga5 Debate             |                   18,820 |                   4,000 |        18,820 |        4,000 |

## Wspólny podzbiór 4k (`results/results_am_subsample.db`)

Sprawiedliwe porównanie 1:1 — te same 4 000 pytań dla każdej pary agent × model.

### Dokładność (%)

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                     45.9 |                    26.7 |          35   |         42.9 |
| ga2 Single + BM25      |                     44   |                    30.9 |          35.2 |         40.7 |
| ga3 Decomp + BM25      |                     44.9 |                    32.9 |          28.8 |         43.5 |
| ga4 FewShot CoT        |                     29.4 |                    29.9 |          28.6 |         28.4 |
| ga5 Debate             |                     23.4 |                    19.6 |          26.4 |         28.9 |

### Średnia liczba tokenów (total) na pytanie

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                      654 |                     329 |           433 |          282 |
| ga2 Single + BM25      |                    1,150 |                     893 |           981 |          900 |
| ga3 Decomp + BM25      |                    2,423 |                   1,388 |         1,715 |        1,282 |
| ga4 FewShot CoT        |                   17,193 |                  10,884 |        14,084 |       10,615 |
| ga5 Debate             |                   28,000 |                  15,043 |        23,906 |       18,798 |

### Średni czas inferencji na pytanie (s)

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                      4.6 |                     0.6 |           1.8 |          0.6 |
| ga2 Single + BM25      |                      3.6 |                     2.9 |           2.5 |          2.4 |
| ga3 Decomp + BM25      |                      7.7 |                     2.7 |           2.4 |          1.6 |
| ga4 FewShot CoT        |                     46.5 |                    16.9 |          22.2 |         17.5 |
| ga5 Debate             |                     61.4 |                    17.7 |          32.7 |         31.9 |

### Pokrycie (liczba wierszy wyników)

| Agent                  |   Bielik-11B-v2.3:Q4_K_M |   Llama-PLLuM-8B:Q4_K_M |   llama3.1:8b |   qwen2.5:7b |
|:-----------------------|-------------------------:|------------------------:|--------------:|-------------:|
| ga1 Single (zero-shot) |                    4,000 |                   4,000 |         4,000 |        4,000 |
| ga2 Single + BM25      |                    4,000 |                   4,000 |         4,000 |        4,000 |
| ga3 Decomp + BM25      |                    4,000 |                   4,000 |         4,000 |        4,000 |
| ga4 FewShot CoT        |                    4,000 |                   4,000 |         4,000 |        4,000 |
| ga5 Debate             |                    4,000 |                   4,000 |         4,000 |        4,000 |
