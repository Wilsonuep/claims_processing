"""
Generyczny BM25 retriever (zoptymalizowany — inverted index)
============================================================

Uniwersalna klasa ``BM25Index`` do wyszukiwania BM25 (Okapi BM25)
na dowolnym korpusie dokumentów — Wikipedia, Demagog, AM benchmark
lub dowolne inne źródło tekstowe.

Optymalizacje pamięciowe
------------------------
    - **Inverted index** zamiast per-document TF.
    - **array.array** dla długości dokumentów.
    - **Brak przechowywania tokenizowanego korpusu**.
    - **heapq.nlargest** do top-k.

Szacunki RAM dla 1.5M chunków (~100 tokenów/chunk)
---------------------------------------------------
    ~4–6 GB  (w tym ~1–2 GB na sam tekst dokumentów)

Użycie
------
    from dataprep.wikipedia_bm25 import BM25Index

    # --- z listy słowników ---
    docs = [{"text": "foo bar"}, {"text": "baz qux"}, ...]
    bm25 = BM25Index(docs)
    results = bm25.search("foo", k=3)

    # --- z bazy SQLite (Wikipedia wiki_chunks) ---
    bm25 = BM25Index.from_sqlite("dataprep/wiki.db")

    # --- z bazy SQLite (Demagog claims) ---
    bm25 = BM25Index.from_sqlite(
        "dataprep/demagog.db",
        table="claims",
        text_column="claim_text",
        columns=["id", "claim_text", "speaker", "label", "topic"],
    )

    # --- z bazy SQLite (AM benchmark claims) ---
    bm25 = BM25Index.from_sqlite(
        "dataprep/am_benchmark.db",
        table="claims",
        text_column="claim_text",
        columns=["id", "claim_text", "label", "topic", "metadata"],
    )

Alias
-----
    WikipediaBM25 = BM25Index   # wsteczna kompatybilność

Zależności
----------
    Brak zewnętrznych — tylko stdlib (sqlite3, array, math, re, heapq).
"""

from __future__ import annotations

import array
import heapq
import logging
import math
import re
import sqlite3
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
# Domyślny tokenizer
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"\W+", re.UNICODE)


def default_tokenize(text: str) -> list[str]:
    """Prosty tokenizer: lowercase + podział na znakach nie-alfanumerycznych.

    Wystarczający do BM25 na polskim tekście. W razie potrzeby podmień
    na tokenizer morfologiczny (np. Morfeusz, spaCy).
    """
    return [tok for tok in _SPLIT_RE.split(text.lower()) if tok]


# ---------------------------------------------------------------------------
# Domyślne konfiguracje from_sqlite dla znanych źródeł
# ---------------------------------------------------------------------------

SQLITE_PRESETS: dict[str, dict[str, Any]] = {
    "wiki_chunks": {
        "table": "wiki_chunks",
        "text_column": "text",
        "columns": [
            "chunk_id", "page_id", "title", "section_title",
            "paragraph_index", "text", "num_tokens",
        ],
    },
    "demagog": {
        "table": "claims",
        "text_column": "claim_text",
        "columns": [
            "id", "claim_text", "speaker", "speaker_role",
            "claim_date", "label", "label_original", "topic", "url",
        ],
    },
    "am_benchmark": {
        "table": "claims",
        "text_column": "claim_text",
        "columns": [
            "id", "claim_text", "label", "label_original",
            "topic", "claim_date", "metadata",
        ],
    },
}


# ---------------------------------------------------------------------------
# Klasa BM25Index (generyczna, zoptymalizowana — inverted index)
# ---------------------------------------------------------------------------


class BM25Index:
    """In-memory BM25 retriever z inverted index na dowolnym korpusie tekstowym.

    Optymalizacje vs. naiwna implementacja
    ----------------------------------------
    1. **Inverted index** (``term → [(doc_idx, tf), ...]``) zamiast
       przechowywania pełnych per-document TF słowników.
    2. **Brak _tokenized_corpus** — tokeny przetwarzane strumieniowo
       i odrzucane po obliczeniu TF.
    3. **array.array('I')** dla ``_doc_len``.
    4. **heapq.nlargest** do wyciągania top-k.

    Atrybuty
    --------
    documents : list[dict]
        Lista dokumentów. Każdy słownik zawiera co najmniej klucz
        z tekstem (domyślnie ``text``).
    corpus_size : int
        Liczba dokumentów w korpusie.
    text_field : str
        Nazwa klucza w słowniku dokumentu, z którego pobierany jest
        tekst do tokenizacji (domyślnie ``"text"``).

    Parametry BM25
    ---------------
    k1 : float   (domyślnie 1.5)
    b  : float   (domyślnie 0.75)
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        *,
        text_field: str = "text",
        tokenize_fn: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Inicjalizuje indeks BM25 na podanym zbiorze dokumentów.

        Parametry
        ---------
        documents : list[dict]
            Lista słowników z co najmniej kluczem ``text_field``.
        text_field : str
            Nazwa klucza, z którego pobierany jest tekst do indeksowania.
            Domyślnie ``"text"`` (dla wiki_chunks). Dla claims użyj
            ``"claim_text"``.
        tokenize_fn : Callable | None
            Funkcja tokenizacji (tekst → lista tokenów).
            Domyślnie: ``default_tokenize``.
        k1, b : float
            Parametry Okapi BM25.
        """
        if not documents:
            raise ValueError("Lista dokumentów nie może być pusta.")

        self.documents = documents
        self.corpus_size = len(documents)
        self.text_field = text_field
        self._tokenize = tokenize_fn or default_tokenize
        self._k1 = k1
        self._b = b

        # --- Budowa indeksu ---
        self._build_index()
        log.info(
            "BM25Index: zbudowano indeks — %d dokumentów, "
            "śr. długość = %.1f tokenów, unikalne termy = %d, "
            "text_field = '%s'.",
            self.corpus_size,
            self._avgdl,
            len(self._inverted_index),
            self.text_field,
        )

    # ------------------------------------------------------------------
    # Budowa inverted index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Buduje inverted index i struktury powiązane.

        Struktury wynikowe
        -------------------
        _doc_len : array.array('I')
            Długości dokumentów (w tokenach). Unsigned int, 4 bajty
            per dokument.
        _inverted_index : dict[str, list[tuple[int, int]]]
            Mapowanie: term → lista (doc_idx, tf).
        _idf : dict[str, float]
            IDF per term (precomputed).
        _avgdl : float
            Średnia długość dokumentu w tokenach.
        """
        self._doc_len = array.array("I")

        inv_build: dict[str, list[tuple[int, int]]] = {}
        total_tokens = 0

        for doc_idx, doc in enumerate(self.documents):
            tokens = self._tokenize(doc.get(self.text_field, ""))
            doc_len = len(tokens)
            self._doc_len.append(doc_len)
            total_tokens += doc_len

            # Zlicz TF w bieżącym dokumencie
            local_tf: dict[str, int] = {}
            for tok in tokens:
                local_tf[tok] = local_tf.get(tok, 0) + 1

            # Dodaj do inverted index
            for term, tf_val in local_tf.items():
                if term not in inv_build:
                    inv_build[term] = []
                inv_build[term].append((doc_idx, tf_val))

        self._inverted_index = inv_build

        self._avgdl = total_tokens / self.corpus_size if self.corpus_size else 1.0

        # Precompute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        N = self.corpus_size
        self._idf: dict[str, float] = {}
        for term, postings in self._inverted_index.items():
            df = len(postings)
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    # ------------------------------------------------------------------
    # Wyszukiwanie z inverted index
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Wyszukuje top-k dokumentów najbardziej pasujących do zapytania.

        Wykorzystuje inverted index — iteruje tylko po posting listach
        termów z zapytania.

        Parametry
        ---------
        query : str
            Zapytanie w języku naturalnym.
        k : int
            Liczba wyników do zwrócenia.

        Zwraca
        ------
        list[dict]
            Lista słowników (kopia oryginalnego dokumentu + klucz
            ``bm25_score``), posortowana malejąco po score.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            log.warning("Puste zapytanie po tokenizacji.")
            return []

        k1 = self._k1
        b = self._b
        avgdl = self._avgdl

        scores: dict[int, float] = {}

        for qt in query_tokens:
            if qt not in self._inverted_index:
                continue

            idf = self._idf[qt]
            postings = self._inverted_index[qt]

            for doc_idx, tf_val in postings:
                dl = self._doc_len[doc_idx]
                numerator = tf_val * (k1 + 1.0)
                denominator = tf_val + k1 * (1.0 - b + b * dl / avgdl)
                term_score = idf * (numerator / denominator)

                if doc_idx in scores:
                    scores[doc_idx] += term_score
                else:
                    scores[doc_idx] = term_score

        if not scores:
            return []

        top_k = heapq.nlargest(k, scores.items(), key=lambda x: x[1])

        results: list[dict[str, Any]] = []
        for doc_idx, score in top_k:
            result = dict(self.documents[doc_idx])
            result["bm25_score"] = score
            results.append(result)

        log.info(
            "BM25 search: query='%s' → %d kandydatów, zwracam top-%d.",
            query[:60],
            len(scores),
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Statystyki pamięci
    # ------------------------------------------------------------------

    def memory_stats(self) -> dict[str, Any]:
        """Zwraca przybliżone statystyki zużycia pamięci przez indeks."""
        total_postings = sum(
            len(pl) for pl in self._inverted_index.values()
        )
        posting_bytes = total_postings * 72
        doc_len_bytes = self._doc_len.itemsize * len(self._doc_len)
        idf_bytes = len(self._idf) * 100
        inv_overhead = len(self._inverted_index) * 100

        total_index_bytes = posting_bytes + doc_len_bytes + idf_bytes + inv_overhead

        return {
            "corpus_size": self.corpus_size,
            "unique_terms": len(self._inverted_index),
            "total_postings": total_postings,
            "avg_postings_per_term": total_postings / max(len(self._inverted_index), 1),
            "doc_len_bytes": doc_len_bytes,
            "estimated_index_mb": total_index_bytes / (1024 * 1024),
        }

    # ------------------------------------------------------------------
    # Factory: ładowanie z bazy SQLite (generyczne)
    # ------------------------------------------------------------------

    @classmethod
    def from_sqlite(
        cls,
        db_path: str,
        *,
        table: str = "wiki_chunks",
        text_column: str = "text",
        columns: list[str] | None = None,
        preset: str | None = None,
        tokenize_fn: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        limit: int | None = None,
    ) -> "BM25Index":
        """Tworzy indeks BM25 z dowolnej tabeli SQLite.

        Parametry
        ---------
        db_path : str
            Ścieżka do pliku ``.db``.
        table : str
            Nazwa tabeli SQLite (domyślnie ``"wiki_chunks"``).
        text_column : str
            Kolumna z tekstem do indeksowania (domyślnie ``"text"``).
        columns : list[str] | None
            Lista kolumn do załadowania. Jeśli ``None``, ładuje ``*``.
            Kolumna ``text_column`` powinna być w tej liście.
        preset : str | None
            Nazwa presetu z ``SQLITE_PRESETS`` (np. ``"wiki_chunks"``,
            ``"demagog"``, ``"am_benchmark"``). Jeśli podany, nadpisuje
            ``table``, ``text_column`` i ``columns``.
        tokenize_fn : Callable | None
            Opcjonalny niestandardowy tokenizer.
        k1, b : float
            Parametry BM25.
        limit : int | None
            Opcjonalny limit wierszy (debugging).

        Zwraca
        ------
        BM25Index
            Gotowy indeks BM25.

        Przykłady
        ---------
        >>> # Wikipedia (domyślne)
        >>> bm25 = BM25Index.from_sqlite("wiki.db")

        >>> # Wikipedia (preset)
        >>> bm25 = BM25Index.from_sqlite("wiki.db", preset="wiki_chunks")

        >>> # Demagog (preset)
        >>> bm25 = BM25Index.from_sqlite("demagog.db", preset="demagog")

        >>> # AM benchmark (preset)
        >>> bm25 = BM25Index.from_sqlite("am_benchmark.db", preset="am_benchmark")

        >>> # Dowolna tabela (ręcznie)
        >>> bm25 = BM25Index.from_sqlite(
        ...     "my.db",
        ...     table="articles",
        ...     text_column="body",
        ...     columns=["id", "body", "author"],
        ... )
        """
        # Zastosuj preset jeśli podany
        if preset is not None:
            if preset not in SQLITE_PRESETS:
                raise ValueError(
                    f"Nieznany preset '{preset}'. "
                    f"Dostępne: {list(SQLITE_PRESETS.keys())}"
                )
            cfg = SQLITE_PRESETS[preset]
            table = cfg["table"]
            text_column = cfg["text_column"]
            columns = cfg["columns"]

        log.info(
            "Ładowanie z bazy: %s (tabela=%s, text_column=%s)",
            db_path, table, text_column,
        )
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        col_list = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_list} FROM {table}"
        if limit:
            query += f" LIMIT {limit}"

        rows = conn.execute(query).fetchall()
        conn.close()

        documents: list[dict[str, Any]] = [dict(row) for row in rows]

        log.info("Załadowano %d dokumentów z tabeli '%s'.", len(documents), table)

        return cls(
            documents,
            text_field=text_column,
            tokenize_fn=tokenize_fn,
            k1=k1,
            b=b,
        )

    # ------------------------------------------------------------------
    # Formatowanie kontekstu dla LLM
    # ------------------------------------------------------------------

    def search_and_format(
        self,
        query: str,
        k: int = 5,
        max_context_chars: int = 3000,
        title_field: str = "title",
        section_field: str | None = "section_title",
    ) -> str:
        """Wyszukuje top-k i zwraca sformatowany kontekst tekstowy.

        Przydatne do bezpośredniego wstrzyknięcia w prompt LLM.

        Parametry
        ---------
        query : str
            Zapytanie.
        k : int
            Liczba wyników.
        max_context_chars : int
            Maksymalna łączna długość kontekstu (znaki).
        title_field : str
            Nazwa pola w dokumencie służącego jako tytuł w nagłówku.
            Domyślnie ``"title"`` (Wikipedia). Dla Demagog ustaw
            np. ``"speaker"``.
        section_field : str | None
            Opcjonalne pole sekcji/tematu. ``None`` = brak.

        Zwraca
        ------
        str
            Sformatowany kontekst, np.::

                [1] Kraków — Historia (score: 3.42)
                Kraków jest jednym z najstarszych miast w Polsce...
        """
        results = self.search(query, k=k)
        if not results:
            return ""

        parts: list[str] = []
        total_chars = 0

        for i, r in enumerate(results, start=1):
            title = r.get(title_field, "")
            section = r.get(section_field, "") if section_field else ""
            score = r.get("bm25_score", 0.0)
            text = r.get(self.text_field, "")

            header = f"[{i}] {title}" if title else f"[{i}]"
            if section:
                header += f" — {section}"
            header += f" (score: {score:.2f})"

            entry = f"{header}\n{text}\n"

            if total_chars + len(entry) > max_context_chars:
                remaining = max_context_chars - total_chars
                if remaining > 50:
                    parts.append(entry[:remaining] + "…")
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Alias wstecznej kompatybilności
# ---------------------------------------------------------------------------

WikipediaBM25 = BM25Index


# ---------------------------------------------------------------------------
# Szybki autotest / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DEMO: BM25Index (generyczny retriever)")
    print("=" * 60)

    # ---- Demo 1: Wikipedia chunks (text_field="text") ----
    wiki_docs = [
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

    print("\n--- Wikipedia (text_field='text') ---")
    bm25_wiki = BM25Index(wiki_docs, text_field="text")
    for r in bm25_wiki.search("stolica Polski", k=2):
        print(f"  [{r['chunk_id']}] score={r['bm25_score']:.4f}  {r['text'][:70]}")

    # ---- Demo 2: Demagog claims (text_field="claim_text") ----
    demagog_docs = [
        {
            "id": 1,
            "claim_text": "Polska jest największym krajem w Europie.",
            "speaker": "Jan Kowalski",
            "label": "REFUTES",
            "topic": "geografia",
        },
        {
            "id": 2,
            "claim_text": "Warszawa jest stolicą Polski od XVI wieku.",
            "speaker": "Anna Nowak",
            "label": "SUPPORTS",
            "topic": "historia",
        },
        {
            "id": 3,
            "claim_text": "Kraków leży nad Wisłą w południowej Polsce.",
            "speaker": "Piotr Zieliński",
            "label": "SUPPORTS",
            "topic": "geografia",
        },
    ]

    print("\n--- Demagog (text_field='claim_text') ---")
    bm25_dem = BM25Index(demagog_docs, text_field="claim_text")
    results = bm25_dem.search("stolica Polski", k=2)
    for r in results:
        print(f"  [id={r['id']}] score={r['bm25_score']:.4f}  "
              f"{r['speaker']}: {r['claim_text'][:60]}")

    # ---- Demo 3: search_and_format z różnymi polami ----
    print("\n--- Formatowany kontekst (Demagog, title=speaker) ---")
    ctx = bm25_dem.search_and_format(
        "stolica Polski", k=2,
        title_field="speaker", section_field="topic",
    )
    print(ctx)

    # ---- Statystyki ----
    stats = bm25_wiki.memory_stats()
    print(f"\n--- Statystyki (Wikipedia) ---")
    print(f"  Dokumenty:      {stats['corpus_size']}")
    print(f"  Unikalne termy: {stats['unique_terms']}")
    print(f"  RAM indeksu:    {stats['estimated_index_mb']:.2f} MB")

    print("\n[OK] Demo zakonczone pomyslnie.")
