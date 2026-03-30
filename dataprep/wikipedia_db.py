"""
Moduł lokalnej bazy danych dla systemu RAG opartego na Wikipedii
=================================================================

Konfiguruje bazę SQLite z rozszerzeniem sqlite-vec do wyszukiwania
wektorowego. Jeden plik .db przechowuje zarówno metadane / tekst
chunków, jak i embeddingi w osobnej wirtualnej tabeli vec0.

Schemat
-------
    wiki_chunks          — metadane + tekst (tabela zwykła)
    wiki_chunk_vectors   — embeddingi (tabela wirtualna vec0, rowid ↔ wiki_chunks.id)

Użycie
------
    from wikipedia_db import init_db, insert_chunks_with_embeddings, knn_search

    conn = init_db("wiki.db")
    insert_chunks_with_embeddings(conn, chunks, embed_fn=my_embed)
    results = knn_search(conn, query_embedding, k=10)
    conn.close()

Wymagane pakiety
-----------------
    pip install sqlite-vec pysqlite3

Uwaga (macOS / Windows)
-----------------------
Domyślny Python na macOS i wielu instalacjach Windows nie wspiera
rozszerzeń SQLite (``enable_load_extension`` jest zablokowane).
Moduł automatycznie używa ``pysqlite3`` jako zamiennika — zawiera
własną kompilację SQLite z pełnym wsparciem rozszerzeń.
Jeśli ``pysqlite3`` nie jest zainstalowane, moduł spróbuje użyć
standardowego ``sqlite3``, ale może to się nie powieść.

Zależności
----------
    - sqlite-vec      (rozszerzenie SQLite do wektorów)
    - pysqlite3       (zamiennik sqlite3 z obsługą rozszerzeń, macOS/Windows)
    - wikipeda_chunking.Chunk   (dataclass chunka)
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Callable

# pysqlite3 zawiera kompilację SQLite z obsługą rozszerzeń — wymagane
# na macOS i Windows, gdzie systemowy sqlite3 często blokuje
# enable_load_extension().
try:
    import pysqlite3 as sqlite3  # type: ignore[no-redef]
except ImportError:
    import sqlite3

from dataprep.wikipeda_chunking import Chunk

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
# Monitoring integration (lazy — no side-effects on import)
# ---------------------------------------------------------------------------

_monitoring = None  # type: ignore[assignment]


def _get_monitoring():
    """Return module-level MonitoringAgent singleton (created on first call)."""
    global _monitoring
    if _monitoring is None:
        try:
            from monitoring.monitor import MonitoringAgent
            _monitoring = MonitoringAgent()
        except Exception:
            class _NoOp:
                def update(self, **_): pass
                def report_crash(self, *_, **__): pass
                def report_done(self, *_, **__): pass
            _monitoring = _NoOp()
    return _monitoring


def _fmt_dur(seconds: float) -> str:
    """Format seconds → '1h 23m', '45m 10s', or '8s'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h {m:02d}m"


# ---------------------------------------------------------------------------
# Stałe konfiguracyjne
# ---------------------------------------------------------------------------

# TODO: Ustaw wymiar embeddingu zgodny z używanym modelem.
#       - sdadas/mmlw-retrieval-roberta-large-v2  →  1024
#       - intfloat/multilingual-e5-large           →  1024
#       - BAAI/bge-m3                              →  1024
#       - sentence-transformers/all-MiniLM-L6-v2   →  384
EMBEDDING_DIM: int = 1024


# ---------------------------------------------------------------------------
# Schemat bazy danych
# ---------------------------------------------------------------------------

_CREATE_WIKI_CHUNKS_SQL = """\
CREATE TABLE IF NOT EXISTS wiki_chunks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id          TEXT    NOT NULL UNIQUE,
    page_id           INTEGER NOT NULL,
    title             TEXT    NOT NULL,
    section_title     TEXT,
    paragraph_index   INTEGER NOT NULL,
    sentence_indices  TEXT    NOT NULL,   -- JSON array np. [0, 1, 2]
    text              TEXT    NOT NULL,
    num_tokens        INTEGER NOT NULL
);
"""

_CREATE_IDX_PAGE_CHUNK_SQL = """\
CREATE INDEX IF NOT EXISTS idx_wiki_chunks_page_chunk
    ON wiki_chunks (page_id, chunk_id);
"""

# Tabela wirtualna vec0 — wymiar jest wstawiany dynamicznie.
_CREATE_VECTORS_SQL_TEMPLATE = """\
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_chunk_vectors
USING vec0(
    embedding float[{dim}]
);
"""


# ---------------------------------------------------------------------------
# Serializacja embeddingów
# ---------------------------------------------------------------------------

def serialize_embedding(embedding: list[float]) -> bytes:
    """Serializuje listę floatów do formatu binarnego (raw float32 / little-endian).

    Jest to format wymagany przez sqlite-vec do operacji MATCH na wektorach.
    Odpowiednik ``serialize_f32`` z oficjalnych przykładów sqlite-vec.
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


# ---------------------------------------------------------------------------
# Inicjalizacja bazy danych
# ---------------------------------------------------------------------------

def init_db(db_path: str, embedding_dim: int = EMBEDDING_DIM) -> sqlite3.Connection:
    """Tworzy (lub otwiera) bazę SQLite, ładuje sqlite-vec i zakłada tabele.

    Parametry
    ---------
    db_path : str
        Ścieżka do pliku .db (zostanie utworzony, jeśli nie istnieje).
    embedding_dim : int
        Wymiar wektora embeddingu (musi odpowiadać modelowi).

    Zwraca
    ------
    sqlite3.Connection
        Otwarte połączenie z załadowanym rozszerzeniem sqlite-vec.

    Uwagi
    -----
    Na macOS domyślny Python nie obsługuje rozszerzeń SQLite.
    Użyj Pythona z Homebrew lub pysqlite3.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # wygodny dostęp po nazwie kolumny
    conn.isolation_level = None      # autocommit — transakcje zarządzamy ręcznie (BEGIN/COMMIT)

    # --- Ładowanie rozszerzenia sqlite-vec ---
    # TODO: Jeśli sqlite_vec.load() nie działa w Twoim środowisku,
    #       możesz załadować rozszerzenie ręcznie:
    #       conn.load_extension("/ścieżka/do/vec0")
    try:
        import sqlite_vec
    except ImportError:
        raise ImportError(
            "Pakiet 'sqlite-vec' nie jest zainstalowany. "
            "Zainstaluj go: pip install sqlite-vec"
        )

    try:
        conn.enable_load_extension(True)
    except AttributeError:
        raise RuntimeError(
            "Twoja instalacja Pythona nie obsługuje rozszerzeń SQLite "
            "(enable_load_extension niedostępne).\n"
            "Zainstaluj pysqlite3: pip install pysqlite3-binary"
        )

    try:
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as exc:
        raise RuntimeError(
            f"Nie udało się załadować rozszerzenia sqlite-vec: {exc}\n"
            "Upewnij się, że pysqlite3-binary jest zainstalowane: "
            "pip install pysqlite3-binary"
        ) from exc

    vec_version = conn.execute("SELECT vec_version()").fetchone()[0]
    log.info("sqlite-vec załadowany (wersja: %s), baza: %s", vec_version, db_path)

    # --- Tworzenie tabel ---
    cur = conn.cursor()
    cur.execute(_CREATE_WIKI_CHUNKS_SQL)
    cur.execute(_CREATE_IDX_PAGE_CHUNK_SQL)

    create_vectors_sql = _CREATE_VECTORS_SQL_TEMPLATE.format(dim=embedding_dim)
    cur.execute(create_vectors_sql)

    conn.commit()
    log.info(
        "Tabele gotowe: wiki_chunks + wiki_chunk_vectors (dim=%d)",
        embedding_dim,
    )

    return conn


# ---------------------------------------------------------------------------
# Wstawianie chunków z embeddingami
# ---------------------------------------------------------------------------

def insert_chunks_with_embeddings(
    conn: sqlite3.Connection,
    chunks: list[Chunk],
    embed_fn: Callable[[str], list[float]],
    *,
    batch_size: int = 500,
) -> None:
    """Wstawia paczkę chunków do wiki_chunks i ich embeddingów do wiki_chunk_vectors.

    Dla każdego chunka:
    1. Wstawia wiersz do ``wiki_chunks`` i pobiera ``id`` (lastrowid).
    2. Oblicza embedding za pomocą ``embed_fn(chunk.text)``.
    3. Wstawia embedding do ``wiki_chunk_vectors`` z tym samym rowid.

    Parametry
    ---------
    conn : sqlite3.Connection
        Połączenie z bazą (z załadowanym sqlite-vec).
    chunks : list[Chunk]
        Lista chunków do wstawienia (patrz ``wikipeda_chunking.Chunk``).
    embed_fn : Callable[[str], list[float]]
        Funkcja embeddingu: tekst → lista floatów.
    batch_size : int
        Rozmiar paczki do przetwarzania w jednej transakcji.

    Uwagi
    -----
    - ``sentence_indices`` jest serializowane jako JSON string.
    - Embedding jest serializowany do formatu binarnego float32
      wymaganego przez sqlite-vec.
    - Wstawianie odbywa się w transakcjach po ``batch_size`` chunków.
    """
    if not chunks:
        log.warning("Brak chunków do wstawienia.")
        return

    total = len(chunks)
    inserted = 0
    errors = 0
    mon = _get_monitoring()
    t_start = __import__('time').perf_counter()

    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        cur = conn.cursor()
        try:
            cur.execute("BEGIN")

            for chunk in batch:
                # 1. Wstaw metadane + tekst do wiki_chunks (INSERT OR IGNORE — pomija duplikaty chunk_id)
                cur.execute(
                    """
                    INSERT OR IGNORE INTO wiki_chunks
                        (chunk_id, page_id, title, section_title,
                         paragraph_index, sentence_indices, text, num_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.page_id,
                        chunk.title,
                        chunk.section_title,
                        chunk.paragraph_index,
                        json.dumps(chunk.sentence_indices),
                        chunk.text,
                        chunk.num_tokens,
                    ),
                )

                # Jeśli INSERT OR IGNORE pominął duplikat — rowcount == 0 → skip embedding
                if cur.rowcount == 0:
                    log.debug("Pominięto duplikat chunk_id: %s", chunk.chunk_id)
                    continue

                row_id = cur.lastrowid

                # 2. Oblicz embedding
                try:
                    embedding = embed_fn(chunk.text)
                except Exception as exc:
                    errors += 1
                    mon.report_crash(
                        exc,
                        context=f"wikipedia_db/insert_chunks/embed_fn/chunk={chunk.chunk_id}",
                    )
                    raise

                # 3. Wstaw embedding do wiki_chunk_vectors (rowid = wiki_chunks.id)
                cur.execute(
                    """
                    INSERT INTO wiki_chunk_vectors (rowid, embedding)
                    VALUES (?, ?)
                    """,
                    (row_id, serialize_embedding(embedding)),
                )

            conn.commit()
            inserted += len(batch)
            elapsed = __import__('time').perf_counter() - t_start
            rate = inserted / max(elapsed, 0.1)
            eta_sec = (total - inserted) / max(rate, 0.001)
            mon.update(
                mode="build_db",
                phase="inserting chunks",
                agent_name="wikipedia_db/insert",
                benchmark="embedding",
                done=inserted,
                total=total,
                errors=errors,
                elapsed_sec=elapsed,
            )
            log.info(
                "[%*d/%d]  %5.1f%%  |  %5.0f ch/s  |  elapsed %-10s |  ETA %s",
                len(str(total)), inserted, total,
                inserted / total * 100,
                rate,
                _fmt_dur(elapsed),
                _fmt_dur(eta_sec),
            )

        except Exception:
            conn.rollback()
            log.exception(
                "Błąd przy wstawianiu paczki %d–%d. Transakcja wycofana.",
                batch_start, batch_start + len(batch),
            )
            raise

    elapsed_total = __import__('time').perf_counter() - t_start
    avg_rate = inserted / max(elapsed_total, 0.1)
    log.info(
        "Zakończono wstawianie: %d chunków w %s (śr. %.0f ch/s).",
        inserted, _fmt_dur(elapsed_total), avg_rate,
    )
    mon.report_done(
        context="wikipedia_db / insert_chunks",
        lines=[
            f"Inserted {inserted:,} chunks in {_fmt_dur(elapsed_total)}",
            f"Avg speed: {avg_rate:.0f} ch/s",
        ],
    )


# ---------------------------------------------------------------------------
# Wyszukiwanie k-NN
# ---------------------------------------------------------------------------

def knn_search(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    k: int = 5,
) -> list[dict]:
    """Wyszukuje k najbliższych chunków do podanego wektora zapytania.

    Parametry
    ---------
    conn : sqlite3.Connection
        Połączenie z bazą (z załadowanym sqlite-vec).
    query_embedding : list[float]
        Wektor zapytania (musi mieć taki sam wymiar jak embeddingi w bazie).
    k : int
        Liczba wyników do zwrócenia.

    Zwraca
    ------
    list[dict]
        Lista słowników z polami:
        - chunk_id, page_id, title, section_title, paragraph_index,
          sentence_indices (list[int]), text, num_tokens, distance (float).
        Posortowane od najbliższego (najmniejszy distance).
    """
    query_blob = serialize_embedding(query_embedding)

    rows = conn.execute(
        """
        SELECT
            wc.chunk_id,
            wc.page_id,
            wc.title,
            wc.section_title,
            wc.paragraph_index,
            wc.sentence_indices,
            wc.text,
            wc.num_tokens,
            v.distance
        FROM wiki_chunk_vectors AS v
        JOIN wiki_chunks AS wc ON wc.id = v.rowid
        WHERE v.embedding MATCH ?
            AND k = ?
        ORDER BY v.distance
        """,
        (query_blob, k),
    ).fetchall()

    results: list[dict] = []
    for row in rows:
        results.append({
            "chunk_id":         row["chunk_id"],
            "page_id":          row["page_id"],
            "title":            row["title"],
            "section_title":    row["section_title"],
            "paragraph_index":  row["paragraph_index"],
            "sentence_indices": json.loads(row["sentence_indices"]),
            "text":             row["text"],
            "num_tokens":       row["num_tokens"],
            "distance":         row["distance"],
        })

    log.info("k-NN: znaleziono %d wyników (k=%d).", len(results), k)
    return results


# ---------------------------------------------------------------------------
# Szybki autotest / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import tempfile
    import os

    print("=" * 60)
    print("DEMO: wikipedia_db.py")
    print("=" * 60)

    # --- Przygotowanie demo chunków ---
    demo_chunks = [
        Chunk(
            chunk_id="42_historia_0_0",
            page_id=42,
            title="Kraków",
            section_title="Historia",
            paragraph_index=0,
            sentence_indices=[0, 1],
            text="Kraków jest jednym z najstarszych miast w Polsce. "
                 "Pierwsza wzmianka pochodzi z X wieku.",
            num_tokens=18,
        ),
        Chunk(
            chunk_id="42_historia_0_1",
            page_id=42,
            title="Kraków",
            section_title="Historia",
            paragraph_index=0,
            sentence_indices=[2, 3],
            text="Miasto było stolicą Polski do 1596 roku. "
                 "Wawel stanowił siedzibę królów polskich.",
            num_tokens=15,
        ),
        Chunk(
            chunk_id="42_geografia_0_0",
            page_id=42,
            title="Kraków",
            section_title="Geografia",
            paragraph_index=0,
            sentence_indices=[0],
            text="Kraków leży nad Wisłą, w południowej Polsce.",
            num_tokens=8,
        ),
    ]

    # TODO: Zamień na prawdziwą funkcję embeddingu, np.:
    #       from sentence_transformers import SentenceTransformer
    #       model = SentenceTransformer("sdadas/mmlw-retrieval-roberta-large-v2")
    #       embed_fn = lambda text: model.encode(text).tolist()
    DIM = 128  # mały wymiar do demo

    def dummy_embed(text: str) -> list[float]:
        """Pseudo-embedding do testów (losowy wektor)."""
        random.seed(hash(text) % 2**32)
        return [random.gauss(0, 1) for _ in range(DIM)]

    # --- Uruchomienie ---
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "wiki_demo.db")
        print(f"\nBaza tymczasowa: {db_path}")

        conn = init_db(db_path, embedding_dim=DIM)

        print("\n--- Wstawianie chunków ---")
        insert_chunks_with_embeddings(conn, demo_chunks, embed_fn=dummy_embed)

        print("\n--- Wyszukiwanie k-NN ---")
        query_vec = dummy_embed("stolica Polski")
        results = knn_search(conn, query_vec, k=2)

        print(f"\nZnaleziono {len(results)} wyników:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['chunk_id']}] dist={r['distance']:.4f}")
            print(f"     {r['text'][:100]}{'…' if len(r['text']) > 100 else ''}")

        # Sprawdzenie liczby rekordów
        count = conn.execute("SELECT COUNT(*) FROM wiki_chunks").fetchone()[0]
        print(f"\nŁącznie w bazie: {count} chunków")

        conn.close()

    print("\n[OK] Demo zakonczone pomyslnie.")
