"""
Readiness check: Wikipedia RAG pipeline (real embedding model).

Runs the full streaming pipeline on 3 synthetic Polish articles:
  Phase 1 — Loads the configured sentence-transformers model
  Phase 2 — Init SQLite+sqlite-vec DB
  Phase 3 — build_article_chunks → insert_chunk_batch (one article at a time)
  Phase 4 — knn_search: 'stolica Polski Warszawa' must hit page_id=1
  Phase 5 — MonitoringAgent state is correct after all insert phases

Passes ↔ embedding model loads, sqlite-vec pipeline builds, knn_search works.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataprep.wikipeda_chunking import build_article_chunks
from dataprep.wikipedia_db import init_db, insert_chunk_batch, knn_search
from dataprep.wikipedia_embedding import load_model
from monitoring.monitor import MonitoringAgent


# ---------------------------------------------------------------------------
# Synthetic corpus — 3 distinct Polish topics so knn_search results are clear
# ---------------------------------------------------------------------------

_ARTICLES = [
    {
        "page_id": 1,
        "title": "Warszawa",
        "sections": [
            {
                "section_title": "Wprowadzenie",
                "paragraphs": [
                    "Warszawa jest stolicą i największym miastem Polski. "
                    "Leży nad Wisłą, w centrum Mazowsza.",
                ],
            },
            {
                "section_title": "Historia",
                "paragraphs": [
                    "Prawa miejskie Warszawa uzyskała w XIV wieku. "
                    "Od 1596 roku jest stolicą Rzeczypospolitej.",
                ],
            },
        ],
    },
    {
        "page_id": 2,
        "title": "Astronomia",
        "sections": [
            {
                "section_title": "Wprowadzenie",
                "paragraphs": [
                    "Astronomia to nauka badająca ciała niebieskie. "
                    "Obejmuje gwiazdy, planety, galaktyki i inne obiekty kosmiczne.",
                ],
            },
            {
                "section_title": "Układ Słoneczny",
                "paragraphs": [
                    "Układ Słoneczny składa się ze Słońca i ośmiu planet. "
                    "Ziemia jest trzecią planetą od Słońca.",
                ],
            },
        ],
    },
    {
        "page_id": 3,
        "title": "Kuchnia polska",
        "sections": [
            {
                "section_title": "Wprowadzenie",
                "paragraphs": [
                    "Kuchnia polska jest różnorodna i bogata w tradycje. "
                    "Charakteryzuje się potrawami z mięsa, ziemniaków i kapusty.",
                ],
            },
            {
                "section_title": "Potrawy",
                "paragraphs": [
                    "Do najpopularniejszych potraw należą bigos, pierogi i żurek. "
                    "Polska kuchnia słynie z bigosu i pierogów.",
                ],
            },
        ],
    },
]


def test_wikipedia_db() -> tuple[bool, float, str | None]:
    start = time.time()
    db_path = "test_wiki_real.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        # ── Phase 1: Load embedding model ─────────────────────────────────────
        embed_model_name = os.getenv("EMBED_MODEL", "sdadas/mmlw-retrieval-roberta-large-v2")
        try:
            embed_model = load_model(embed_model_name)
        except Exception as e:
            return False, time.time() - start, (
                f"Failed to load embedding model '{embed_model_name}': {e}\n"
                "Check: pip install sentence-transformers  |  HuggingFace connectivity"
            )

        embed_dim = len(embed_model.encode("test").tolist())

        # ── Phase 2: Init DB + monitoring agent ───────────────────────────────
        mon = MonitoringAgent(active=False, webhook_url="", machine_name="test-wiki")
        mon.update(mode="build_db", phase="init DB", done=0, total=len(_ARTICLES))

        try:
            conn = init_db(db_path, embedding_dim=embed_dim)
        except Exception as e:
            return False, time.time() - start, f"init_db failed: {e}"

        # ── Phase 3: Stream articles → chunk → embed → insert ────────────────
        total_inserted = 0
        t0 = time.perf_counter()
        mon.update(mode="build_db", phase="inserting chunks", done=0, total=len(_ARTICLES))

        for art_idx, article in enumerate(_ARTICLES, 1):
            chunks = build_article_chunks(article)
            if not chunks:
                conn.close()
                return False, time.time() - start, (
                    f"build_article_chunks returned no chunks for '{article['title']}'"
                )

            texts = [c.text for c in chunks]
            embeddings = embed_model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            pairs = [(c, emb.tolist()) for c, emb in zip(chunks, embeddings)]
            insert_chunk_batch(conn, pairs)
            total_inserted += len(chunks)

            elapsed = time.perf_counter() - t0
            mon.update(
                mode="build_db",
                phase="inserting chunks",
                done=art_idx,
                total=len(_ARTICLES),
                elapsed_sec=elapsed,
            )

        # ── Phase 4: Verify DB chunk count ────────────────────────────────────
        db_count = conn.execute("SELECT COUNT(*) FROM wiki_chunks").fetchone()[0]
        if db_count != total_inserted:
            conn.close()
            return False, time.time() - start, (
                f"DB has {db_count} chunks, expected {total_inserted}"
            )

        # ── Phase 5: knn_search — Warszawa query must rank page_id=1 first ───
        query_vec = embed_model.encode(
            "stolica Polski Warszawa Wisła",
            normalize_embeddings=True,
        ).tolist()
        results = knn_search(conn, query_vec, k=3)

        if not results:
            conn.close()
            return False, time.time() - start, "knn_search returned no results"

        top_page_id = results[0]["page_id"]
        if top_page_id != 1:
            conn.close()
            return False, time.time() - start, (
                f"knn_search top result is page_id={top_page_id} ('{results[0]['title']}'), "
                f"expected page_id=1 (Warszawa). All hits: "
                + ", ".join(f"{r['title']} d={r['distance']:.4f}" for r in results)
            )

        # ── Phase 6: MonitoringAgent state check ──────────────────────────────
        state = mon._snapshot()
        if state["done"] != len(_ARTICLES):
            conn.close()
            return False, time.time() - start, (
                f"MonitoringAgent done={state['done']}, expected {len(_ARTICLES)}"
            )
        if state["phase"] != "inserting chunks":
            conn.close()
            return False, time.time() - start, (
                f"MonitoringAgent phase='{state['phase']}', expected 'inserting chunks'"
            )
        if state["elapsed_sec"] <= 0:
            conn.close()
            return False, time.time() - start, "MonitoringAgent elapsed_sec not updated"

        conn.close()
        return True, time.time() - start, None

    except Exception as e:
        import traceback
        return False, time.time() - start, f"{e}\n{traceback.format_exc()}"
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError:
                pass


if __name__ == "__main__":
    success, elapsed, err = test_wikipedia_db()
    if success:
        print(f"PASSED ({elapsed:.2f}s)")
    else:
        print(f"FAILED ({elapsed:.2f}s): {err}")
