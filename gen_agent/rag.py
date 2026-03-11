"""
Ogólna klasa RAG (Retrieval-Augmented Generation)
==================================================

Łączy wyszukiwanie BM25 i/lub wektorowe (embedding k-NN) w jeden
reużywalny komponent. Zaprojektowany do użycia przez dowolne systemy
agentowe potrzebujące kontekstu z baz danych przed generowaniem
odpowiedzi.

Tryby wyszukiwania
-------------------
    - ``bm25``    — tylko BM25 (szybkie, leksykalne, bez embeddingów)
    - ``vector``  — tylko k-NN na embeddingach (sqlite-vec)
    - ``hybrid``  — BM25 + vector, fuzja wyników przez Reciprocal Rank
                    Fusion (RRF)

Two-stage retrieval (retrieve_structured)
-----------------------------------------
    1. Pobierz K_initial kandydatów (np. 20) via BM25/vector/hybrid.
    2. Opcjonalnie odfiltruj wyniki z score < threshold.
    3. Utnij do K_final (np. 3–5) — te trafiają do LLM.
    Wyniki zwracane w ustrukturyzowanej formie z pełnymi metadanymi
    (chunk_id, title, section, score, text) — provenance tracking.

Użycie
------
    from gen_agent.rag import RAGRetriever

    rag = RAGRetriever(mode="bm25", bm25_db_path="wiki.db")

    # Stare API (bez zmian):
    context_str = rag.retrieve_and_format("Kraków stolica", k=5)

    # Nowe API — structured evidence:
    evidence = rag.retrieve_structured(
        "Kraków stolica",
        k_initial=20,
        k_final=5,
        score_threshold=0.1,
    )
    # evidence = [{"chunk_id": ..., "title": ..., "section": ...,
    #              "score": ..., "text": ...}, ...]

    # Sformatuj structured evidence do tekstu:
    context_str = RAGRetriever.format_evidence(evidence)

Zależności
----------
    - gen_agent.bm25               (BM25Index — zawsze dostępne)
    - dataprep.wikipedia_db       (knn_search — tylko w trybie vector/hybrid)
    - sentence-transformers       (tylko gdy embed_fn nie jest podane)
    - sqlite-vec + pysqlite3      (tylko w trybie vector/hybrid)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Konfiguracja logowania
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    id_field: str = "chunk_id",
    k_rrf: int = 60,
) -> list[dict[str, Any]]:
    """Łączy wiele posortowanych list wyników przez Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) po wszystkich listach, gdzie rank
    jest 1-bazowy.

    Parametry
    ---------
    ranked_lists : list[list[dict]]
        Lista list wyników (każda posortowana od najlepszego).
    id_field : str
        Pole identyfikujące dokument do deduplikacji.
    k_rrf : int
        Stała RRF (domyślnie 60 — standardowa wartość z literatury).

    Zwraca
    ------
    list[dict]
        Połączona lista wyników posortowana malejąco po ``rrf_score``.
        Każdy wynik zawiera dodatkowe pole ``rrf_score``.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = str(doc.get(id_field, id(doc)))
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
            # Zachowaj pełny dokument (pierwszy napotkany)
            if doc_id not in doc_map:
                doc_map[doc_id] = dict(doc)

    # Sortuj malejąco po RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results: list[dict[str, Any]] = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc["rrf_score"] = scores[doc_id]
        results.append(doc)

    return results


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """Uniwersalny retriever RAG — BM25, vector, lub hybrid.

    Atrybuty
    --------
    mode : str
        Tryb wyszukiwania: ``"bm25"``, ``"vector"``, ``"hybrid"``.

    Parametry
    ---------
    mode : str
        Tryb wyszukiwania.
    bm25_db_path : str | None
        Ścieżka do bazy SQLite z danymi dla BM25.
        Wymagane w trybach ``"bm25"`` i ``"hybrid"``.
    bm25_preset : str | None
        Preset BM25 (``"wiki_chunks"``, ``"demagog"``, ``"am_benchmark"``).
        Jeśli None, używa domyślnych (wiki_chunks).
    bm25_kwargs : dict | None
        Dodatkowe parametry dla ``BM25Index.from_sqlite()``, np.
        ``{"table": "claims", "text_column": "claim_text"}``.
    vector_db_path : str | None
        Ścieżka do bazy SQLite z embeddingami (sqlite-vec).
        Wymagane w trybach ``"vector"`` i ``"hybrid"``.
    embed_fn : Callable[[str], list[float]] | None
        Funkcja embeddingu zapytania. Wymagana w trybach
        ``"vector"`` i ``"hybrid"``.
    embed_model_name : str | None
        Nazwa modelu sentence-transformers (używana jako fallback
        gdy ``embed_fn`` nie jest podane). Domyślnie:
        ``sdadas/mmlw-retrieval-roberta-large-v2``.
    embed_device : str | None
        Urządzenie dla modelu embeddingowego (cuda/mps/cpu/None=auto).
    text_field : str
        Nazwa pola z tekstem do wyświetlania (domyślnie ``"text"``).
    title_field : str
        Nazwa pola z tytułem (domyślnie ``"title"``).
    section_field : str | None
        Nazwa pola z sekcją (domyślnie ``"section_title"``).
    id_field : str
        Pole identyfikujące dokument (do RRF fusion, domyślnie ``"chunk_id"``).
    """

    VALID_MODES = ("bm25", "vector", "hybrid")

    def __init__(
        self,
        mode: str = "bm25",
        *,
        # BM25
        bm25_db_path: str | None = None,
        bm25_preset: str | None = None,
        bm25_kwargs: dict[str, Any] | None = None,
        # Vector
        vector_db_path: str | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        embed_model_name: str | None = None,
        embed_device: str | None = None,
        embedding_dim: int = 1024,
        # Shared
        text_field: str = "text",
        title_field: str = "title",
        section_field: str | None = "section_title",
        id_field: str = "chunk_id",
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Nieznany tryb '{mode}'. Dostępne: {self.VALID_MODES}"
            )

        self.mode = mode
        self.text_field = text_field
        self.title_field = title_field
        self.section_field = section_field
        self.id_field = id_field

        # ----- BM25 -----
        self._bm25 = None
        if mode in ("bm25", "hybrid"):
            if bm25_db_path is None:
                raise ValueError(
                    f"Tryb '{mode}' wymaga parametru bm25_db_path."
                )
            self._init_bm25(bm25_db_path, bm25_preset, bm25_kwargs or {})

        # ----- Vector -----
        self._vector_conn = None
        self._embed_fn = embed_fn
        self._embed_model = None
        self._embed_model_name = (
            embed_model_name or "sdadas/mmlw-retrieval-roberta-large-v2"
        )
        self._embed_device = embed_device
        self._embedding_dim = embedding_dim

        if mode in ("vector", "hybrid"):
            if vector_db_path is None:
                raise ValueError(
                    f"Tryb '{mode}' wymaga parametru vector_db_path."
                )
            self._init_vector(vector_db_path)

        log.info(
            "RAGRetriever zainicjalizowany: mode=%s, text_field='%s'.",
            self.mode, self.text_field,
        )

    # ------------------------------------------------------------------
    # Inicjalizacja komponentów
    # ------------------------------------------------------------------

    def _init_bm25(
        self,
        db_path: str,
        preset: str | None,
        kwargs: dict[str, Any],
    ) -> None:
        """Inicjalizuje indeks BM25."""
        from gen_agent.bm25 import BM25Index

        from_sqlite_kwargs: dict[str, Any] = dict(kwargs)
        if preset is not None:
            from_sqlite_kwargs["preset"] = preset

        self._bm25 = BM25Index.from_sqlite(db_path, **from_sqlite_kwargs)
        log.info("BM25 gotowy: %d dokumentów.", self._bm25.corpus_size)

    def _init_vector(self, db_path: str) -> None:
        """Inicjalizuje połączenie z bazą wektorową (sqlite-vec)."""
        from dataprep.wikipedia_db import init_db

        self._vector_conn = init_db(db_path, embedding_dim=self._embedding_dim)
        log.info("Vector DB gotowa: %s", db_path)

    def _get_embed_fn(self) -> Callable[[str], list[float]]:
        """Zwraca funkcję embeddingu (lazy-loading modelu)."""
        if self._embed_fn is not None:
            return self._embed_fn

        if self._embed_model is None:
            from dataprep.wikipedia_embedding import load_model
            self._embed_model = load_model(
                self._embed_model_name, device=self._embed_device
            )
            log.info("Model embeddingowy załadowany: %s", self._embed_model_name)

        model = self._embed_model

        def _embed(text: str) -> list[float]:
            return model.encode(
                text, normalize_embeddings=True
            ).tolist()

        return _embed

    # ------------------------------------------------------------------
    # Retrieval: BM25
    # ------------------------------------------------------------------

    def _retrieve_bm25(self, query: str, k: int) -> list[dict[str, Any]]:
        """Wyszukiwanie BM25."""
        if self._bm25 is None:
            raise RuntimeError("BM25 nie jest zainicjalizowany.")
        return self._bm25.search(query, k=k)

    # ------------------------------------------------------------------
    # Retrieval: Vector (k-NN)
    # ------------------------------------------------------------------

    def _retrieve_vector(self, query: str, k: int) -> list[dict[str, Any]]:
        """Wyszukiwanie wektorowe (k-NN) przez sqlite-vec."""
        if self._vector_conn is None:
            raise RuntimeError("Baza wektorowa nie jest zainicjalizowana.")

        from dataprep.wikipedia_db import knn_search

        embed_fn = self._get_embed_fn()
        query_embedding = embed_fn(query)

        results = knn_search(self._vector_conn, query_embedding, k=k)

        # Ujednolicenie: dodaj score (odwrotność distance)
        for r in results:
            r["vector_score"] = 1.0 / (1.0 + r.get("distance", 0.0))

        return results

    # ------------------------------------------------------------------
    # Retrieval: Hybrid (BM25 + Vector → RRF)
    # ------------------------------------------------------------------

    def _retrieve_hybrid(
        self, query: str, k: int, k_rrf: int = 60
    ) -> list[dict[str, Any]]:
        """Wyszukiwanie hybrydowe: BM25 + vector, fuzja przez RRF."""
        # Pobieramy 2x k z każdego źródła, żeby RRF miał więcej do fuzji
        fetch_k = min(k * 2, 50)

        bm25_results = self._retrieve_bm25(query, k=fetch_k)
        vector_results = self._retrieve_vector(query, k=fetch_k)

        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            id_field=self.id_field,
            k_rrf=k_rrf,
        )

        log.info(
            "Hybrid retrieval: BM25=%d + Vector=%d → RRF=%d (zwracam top-%d).",
            len(bm25_results), len(vector_results), len(fused), k,
        )

        return fused[:k]

    # ------------------------------------------------------------------
    # Publiczne API: retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Wyszukuje top-k dokumentów w wybranym trybie.

        Parametry
        ---------
        query : str
            Zapytanie w języku naturalnym.
        k : int
            Liczba wyników do zwrócenia.

        Zwraca
        ------
        list[dict]
            Lista dokumentów z metadanymi i score'ami.
        """
        if self.mode == "bm25":
            return self._retrieve_bm25(query, k)
        elif self.mode == "vector":
            return self._retrieve_vector(query, k)
        elif self.mode == "hybrid":
            return self._retrieve_hybrid(query, k)
        else:
            raise ValueError(f"Nieznany tryb: {self.mode}")

    # ------------------------------------------------------------------
    # Publiczne API: retrieve_and_format
    # ------------------------------------------------------------------

    def retrieve_and_format(
        self,
        query: str,
        k: int = 5,
        max_context_chars: int = 3000,
    ) -> str:
        """Wyszukuje top-k i zwraca sformatowany tekst do wstrzyknięcia w prompt LLM.

        Parametry
        ---------
        query : str
            Zapytanie w języku naturalnym.
        k : int
            Liczba wyników.
        max_context_chars : int
            Maksymalna łączna długość kontekstu (znaki).

        Zwraca
        ------
        str
            Sformatowany kontekst, np.::

                [1] Kraków — Historia (score: 0.034)
                Kraków jest jednym z najstarszych miast w Polsce...
        """
        results = self.retrieve(query, k=k)
        if not results:
            return ""

        parts: list[str] = []
        total_chars = 0

        for i, r in enumerate(results, start=1):
            title = r.get(self.title_field, "")
            section = (
                r.get(self.section_field, "") if self.section_field else ""
            )
            text = r.get(self.text_field, "")

            # Wybierz najlepszy dostępny score
            score = (
                r.get("rrf_score")
                or r.get("bm25_score")
                or r.get("vector_score")
                or 0.0
            )

            header = f"[{i}] {title}" if title else f"[{i}]"
            if section:
                header += f" — {section}"
            header += f" (score: {score:.4f})"

            entry = f"{header}\n{text}\n"

            if total_chars + len(entry) > max_context_chars:
                remaining = max_context_chars - total_chars
                if remaining > 50:
                    parts.append(entry[:remaining] + "…")
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Publiczne API: retrieve_structured (two-stage)
    # ------------------------------------------------------------------

    def retrieve_structured(
        self,
        query: str,
        *,
        k_initial: int = 20,
        k_final: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Two-stage retrieval: fetch → filter/rerank → truncate.

        Stage 1: Pobierz ``k_initial`` kandydatów (BM25/vector/hybrid).
        Stage 2: Odfiltruj wyniki poniżej ``score_threshold`` (opcjonalne),
                 następnie utnij do ``k_final``.

        Wyniki mają ustandaryzowaną strukturę z pełnym provenance::

            [
                {
                    "chunk_id":  str,
                    "title":     str,
                    "section":   str,
                    "score":     float,
                    "text":      str,
                    "source":    str,      # "bm25" | "vector" | "hybrid"
                },
                ...
            ]

        Parametry
        ---------
        query : str
            Zapytanie w języku naturalnym.
        k_initial : int
            Liczba kandydatów do pobrania w stage 1.
        k_final : int
            Liczba wyników do zwrócenia (po filtracji/rerankingu).
        score_threshold : float | None
            Minimalny score, aby zachować chunk. ``None`` = bez filtracji.

        Zwraca
        ------
        list[dict]
            Ustrukturyzowane wyniki z provenance metadata.
        """
        # Stage 1: fetch k_initial candidates
        raw_results = self.retrieve(query, k=k_initial)

        # Normalize to standard structure
        structured: list[dict[str, Any]] = []
        for r in raw_results:
            score = (
                r.get("rrf_score")
                or r.get("bm25_score")
                or r.get("vector_score")
                or 0.0
            )
            structured.append({
                "chunk_id": r.get(self.id_field, ""),
                "title":    r.get(self.title_field, ""),
                "section":  r.get(self.section_field, "") if self.section_field else "",
                "score":    score,
                "text":     r.get(self.text_field, ""),
                "source":   self.mode,
            })

        # Stage 2a: filter by score threshold
        if score_threshold is not None:
            before_count = len(structured)
            structured = [
                s for s in structured if s["score"] >= score_threshold
            ]
            if len(structured) < before_count:
                log.info(
                    "Score filter: %d → %d (threshold=%.4f)",
                    before_count, len(structured), score_threshold,
                )

        # Stage 2b: truncate to k_final (already sorted by retrieve())
        final = structured[:k_final]

        log.info(
            "retrieve_structured: query='%s' | k_initial=%d → fetched=%d → final=%d",
            query[:50], k_initial, len(raw_results), len(final),
        )

        return final

    # ------------------------------------------------------------------
    # Static helper: format structured evidence for LLM prompt
    # ------------------------------------------------------------------

    @staticmethod
    def format_evidence(
        evidence: list[dict[str, Any]],
        *,
        max_context_chars: int = 3000,
    ) -> str:
        """Formatuje ustrukturyzowane wyniki do tekstu dla promptu LLM.

        Generuje kontekst z pełnym provenance (tytuł, sekcja, score)
        z listy zwróconej przez ``retrieve_structured()``.

        Parametry
        ---------
        evidence : list[dict]
            Lista z polami: chunk_id, title, section, score, text.
        max_context_chars : int
            Maksymalna łączna długość (znaki).

        Zwraca
        ------
        str
            Sformatowany kontekst gotowy do wstrzyknięcia w prompt.
        """
        if not evidence:
            return ""

        parts: list[str] = []
        total_chars = 0

        for i, e in enumerate(evidence, start=1):
            title = e.get("title", "")
            section = e.get("section", "")
            score = e.get("score", 0.0)
            chunk_id = e.get("chunk_id", "")
            text = e.get("text", "")

            header = f"[{i}] {title}" if title else f"[{i}]"
            if section:
                header += f" — {section}"
            header += f" (score: {score:.4f}, id: {chunk_id})"

            entry = f"{header}\n{text}\n"

            if total_chars + len(entry) > max_context_chars:
                remaining = max_context_chars - total_chars
                if remaining > 50:
                    parts.append(entry[:remaining] + "…")
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Zamykanie zasobów
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Zamyka połączenia z bazami danych."""
        if self._vector_conn is not None:
            self._vector_conn.close()
            self._vector_conn = None
            log.info("Vector DB zamknięta.")

    def __enter__(self) -> "RAGRetriever":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Szybki autotest / demo (tylko BM25 — nie wymaga embeddingów)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from gen_agent.bm25 import BM25Index

    print("=" * 60)
    print("DEMO: RAGRetriever (tryb BM25)")
    print("=" * 60)

    # --- Przygotowanie demo danych ---
    demo_docs = [
        {
            "chunk_id": "42_historia_0_0",
            "title": "Kraków",
            "section_title": "Historia",
            "text": "Kraków jest jednym z najstarszych miast w Polsce. "
                    "Pierwsza wzmianka pochodzi z X wieku.",
        },
        {
            "chunk_id": "42_historia_0_1",
            "title": "Kraków",
            "section_title": "Historia",
            "text": "Miasto było stolicą Polski do 1596 roku. "
                    "Wawel stanowił siedzibę królów polskich.",
        },
        {
            "chunk_id": "99_warszawa_0_0",
            "title": "Warszawa",
            "section_title": "Historia",
            "text": "Warszawa jest stolicą Polski od 1596 roku. "
                    "Leży nad Wisłą w centralnej części kraju.",
        },
    ]

    # Tworzenie RAGRetriever z ręcznie zbudowanym BM25
    rag = RAGRetriever.__new__(RAGRetriever)
    rag.mode = "bm25"
    rag.text_field = "text"
    rag.title_field = "title"
    rag.section_field = "section_title"
    rag.id_field = "chunk_id"
    rag._bm25 = BM25Index(demo_docs, text_field="text")
    rag._vector_conn = None
    rag._embed_fn = None
    rag._embed_model = None

    # --- retrieve ---
    query = "stolica Polski"
    print(f"\nZapytanie: '{query}'")
    results = rag.retrieve(query, k=3)

    print(f"\nZnaleziono {len(results)} wyników:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['chunk_id']}] score={r.get('bm25_score', 0):.4f}")
        print(f"     {r['text'][:80]}")

    # --- retrieve_and_format ---
    print("\n--- Formatowany kontekst (gotowy do wstrzyknięcia w prompt) ---")
    ctx = rag.retrieve_and_format(query, k=3)
    print(ctx)

    print("\n[OK] Demo zakonczone pomyslnie.")
