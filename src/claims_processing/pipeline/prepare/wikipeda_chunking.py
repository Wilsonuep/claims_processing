"""
Moduł chunkowania Wikipedii
===========================

Konwertuje sparsowane artykuły z polskiej Wikipedii na jednostki
na poziomie zdań i chunków, gotowe do embeddingu i wyszukiwania
wektorowego w offline'owym pipeline'ie RAG.

Użycie
------
    from wikipeda_chunking import build_wiki_chunks, build_article_chunks

    chunks = build_wiki_chunks(articles)   # list[Chunk]
    rows   = [c.to_dict() for c in chunks] # gotowe do DB / JSONL

Zależności
----------
Wywołujący musi dostarczyć dwie funkcje (wstrzykiwane przez zmienne
modułowe lub monkey-patching):

    split_into_sentences(text: str) -> list[str]
    tokenize(text: str) -> list[str]

Jeśli nie zostaną podane, używane są lekkie domyślne implementacje
(regex do zdań i tokenizer białoznakowy). Zamień je przed użyciem
produkcyjnym.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import TypedDict, Callable

# ---------------------------------------------------------------------------
# Definicje typów dla struktury artykułu wejściowego
# ---------------------------------------------------------------------------


class Section(TypedDict):
    section_title: str          # np. "Biografia"
    paragraphs: list[str]       # akapity jako tekst (już oczyszczone z markupu)


class Article(TypedDict):
    page_id: int
    title: str
    sections: list[Section]


# ---------------------------------------------------------------------------
# Wymienne funkcje NLP (zamień przed użyciem produkcyjnym)
# ---------------------------------------------------------------------------

def _default_split_into_sentences(text: str) -> list[str]:
    """Naiwny splitter zdań oparty na regexie – wystarczający do testów.
    Zamień na właściwy tokenizer zdaniowy dla polskiego (np. spaCy,
    Stanza lub pySBD skonfigurowany dla języka polskiego).
    """
    # Dziel na znakach końca zdania, po których następuje biały znak lub koniec tekstu.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _default_tokenize(text: str) -> list[str]:
    """Placeholder tokenizera białoznakowego.
    Zamień na *ten sam* tokenizer, którego używa model embeddingowy
    (np. tiktoken, SentencePiece, tokenizer HuggingFace).
    """
    return text.split()


# Referencje na poziomie modułu – nadpisz je przed wywołaniem publicznego API.
split_into_sentences: Callable[[str], list[str]] = _default_split_into_sentences
tokenize: Callable[[str], list[str]] = _default_tokenize


# ---------------------------------------------------------------------------
# Parametry chunkowania (wartości domyślne)
# ---------------------------------------------------------------------------

MAX_SENT_PER_CHUNK: int = 3       # maks. liczba zdań na chunk
MAX_TOKENS_PER_CHUNK: int = 256   # maks. liczba tokenów na chunk
OVERLAP_SENT: int = 1             # zakładka zdaniowa między kolejnymi chunkami


# ---------------------------------------------------------------------------
# Dataclass: Sentence (zdanie)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Sentence:
    """Pojedyncze zdanie z artykułu Wikipedii wraz z pełnym pochodzeniem."""
    page_id: int
    title: str
    section_title: str
    paragraph_index: int     # indeks 0-bazowy w obrębie sekcji
    sentence_index: int      # indeks 0-bazowy w obrębie akapitu
    text: str


# ---------------------------------------------------------------------------
# Dataclass: Chunk (fragment)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Chunk:
    """Grupa kolejnych zdań gotowa do embeddingu."""
    chunk_id: str
    page_id: int
    title: str
    section_title: str
    paragraph_index: int
    sentence_indices: list[int]
    text: str
    num_tokens: int

    # ------------------------------------------------------------------
    # Metody serializacji
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Zwraca zwykły słownik gotowy do zapisu JSONL / wstawienia do bazy."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Funkcje pomocnicze
# ---------------------------------------------------------------------------

def normalize_for_id(text: str) -> str:
    """Przekształca tytuł sekcji na bezpieczny fragment chunk_id.

    - normalizacja NFKD, usunięcie akcentów
    - zamiana na małe litery
    - zamiana białych znaków / ciągów nie-alfanumerycznych na '_'
    - usunięcie wiodących / końcowych podkreśleń
    """
    # Normalizacja Unicode i usunięcie znaków łączących (akcentów)
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    lowered = ascii_text.lower()
    cleaned = re.sub(r'[^a-z0-9]+', '_', lowered)
    return cleaned.strip('_') or "untitled"


# ---------------------------------------------------------------------------
# article_to_sentences — rozbijanie artykułu na zdania
# ---------------------------------------------------------------------------

def article_to_sentences(article: Article) -> list[Sentence]:
    """Spłaszcza artykuł do listy obiektów :class:`Sentence` z metadanymi.

    Kolejność przetwarzania:
    1. Iteruj po sekcjach w kolejności.
    2. W każdej sekcji iteruj po akapitach w kolejności.
    3. Dla każdego akapitu podziel tekst na zdania za pomocą
       :func:`split_into_sentences`.
    4. Pomiń puste / białoznakowe zdania.
    5. ``sentence_index`` jest lokalny w obrębie akapitu (0, 1, 2, …).
    """
    sentences: list[Sentence] = []
    page_id = article["page_id"]
    title = article["title"]

    for section in article["sections"]:
        section_title = section["section_title"]
        for para_idx, paragraph in enumerate(section["paragraphs"]):
            if not paragraph or not paragraph.strip():
                continue

            raw_sents = split_into_sentences(paragraph)
            sent_idx = 0
            for raw in raw_sents:
                text = raw.strip()
                if not text:
                    continue
                sentences.append(
                    Sentence(
                        page_id=page_id,
                        title=title,
                        section_title=section_title,
                        paragraph_index=para_idx,
                        sentence_index=sent_idx,
                        text=text,
                    )
                )
                sent_idx += 1

    return sentences


# ---------------------------------------------------------------------------
# chunk_paragraph_sentences — chunkowanie zdań jednego akapitu
# ---------------------------------------------------------------------------

def chunk_paragraph_sentences(
    sentences: list[Sentence],
    max_sent_per_chunk: int = MAX_SENT_PER_CHUNK,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    overlap_sent: int = OVERLAP_SENT,
) -> list[Chunk]:
    """Dzieli listę :class:`Sentence` z *jednego* akapitu na chunki.

    Algorytm (okno przesuwne z zakładką):
    1. Wszystkie zdania mają ten sam ``page_id``, ``title``,
       ``section_title`` i ``paragraph_index``.
    2. Utrzymuj indeks ``i`` — początek bieżącego okna.
    3. Zachłannie dodawaj zdania, aż osiągniesz *max_sent_per_chunk*
       lub *max_tokens_per_chunk*.
    4. Jeśli pojedyncze zdanie przekracza limit tokenów, umieść je
       samodzielnie w chunku (przypadek brzegowy).
    5. Po wyemitowaniu chunka przesuń ``i`` o
       ``max(num_sent_in_chunk - overlap_sent, 1)``, aby zawsze
       posuwać się do przodu, umożliwiając lekką zakładkę.

    Zwraca listę obiektów :class:`Chunk`.
    """
    if not sentences:
        return []

    # Wstępne obliczenie liczby tokenów na zdanie (unikamy powtórnej tokenizacji).
    token_counts: list[int] = [len(tokenize(s.text)) for s in sentences]

    # Metadane (stałe w obrębie akapitu).
    ref = sentences[0]
    page_id = ref.page_id
    title = ref.title
    section_title = ref.section_title
    paragraph_index = ref.paragraph_index
    section_id_part = normalize_for_id(section_title)

    chunks: list[Chunk] = []
    local_chunk_idx = 0
    i = 0

    while i < len(sentences):
        # ------ budowa bieżącego okna ------
        window_sents: list[Sentence] = []
        window_tokens = 0
        j = i

        while j < len(sentences):
            # Sprawdź limity *przed* dodaniem kolejnego zdania.
            if len(window_sents) >= max_sent_per_chunk:
                break
            next_tokens = token_counts[j]
            if window_sents and (window_tokens + next_tokens) > max_tokens_per_chunk:
                break
            window_sents.append(sentences[j])
            window_tokens += next_tokens
            j += 1

        # Przypadek brzegowy: nawet pierwsze zdanie się nie zmieściło?
        # Nie powinno się zdarzyć, bo zawsze dodajemy przynajmniej jedno, ale zabezpieczamy.
        if not window_sents:
            window_sents = [sentences[i]]
            window_tokens = token_counts[i]

        # ------ emisja chunka ------
        chunk_text = " ".join(s.text for s in window_sents)
        sent_indices = [s.sentence_index for s in window_sents]

        chunk = Chunk(
            chunk_id=f"{page_id}_{section_id_part}_{paragraph_index}_{local_chunk_idx}",
            page_id=page_id,
            title=title,
            section_title=section_title,
            paragraph_index=paragraph_index,
            sentence_indices=sent_indices,
            text=chunk_text,
            num_tokens=window_tokens,
        )
        chunks.append(chunk)
        local_chunk_idx += 1

        # ------ przesunięcie okna z zakładką ------
        num_sent_in_chunk = len(window_sents)
        if overlap_sent > 0:
            i = max(i + num_sent_in_chunk - overlap_sent, i + 1)
        else:
            i += num_sent_in_chunk

    return chunks


# ---------------------------------------------------------------------------
# build_article_chunks — chunkowanie całego artykułu
# ---------------------------------------------------------------------------

def build_article_chunks(
    article: Article,
    max_sent_per_chunk: int = MAX_SENT_PER_CHUNK,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    overlap_sent: int = OVERLAP_SENT,
) -> list[Chunk]:
    """Konwertuje pojedynczy :class:`Article` na płaską listę :class:`Chunk`.

    Kroki:
    1. Iteruj po sekcjach i akapitach.
    2. Zamień każdy akapit na obiekty :class:`Sentence` (z metadanymi).
    3. Wywołaj :func:`chunk_paragraph_sentences` dla listy zdań
       każdego akapitu.
    4. Połącz wszystkie wynikowe chunki w jedną listę.

    Granice akapitów i sekcji **nigdy** nie są przekraczane.
    """
    all_chunks: list[Chunk] = []
    page_id = article["page_id"]
    title = article["title"]

    for section in article["sections"]:
        section_title = section["section_title"]
        for para_idx, paragraph in enumerate(section["paragraphs"]):
            if not paragraph or not paragraph.strip():
                continue

            # --- budowa obiektów Sentence dla tego akapitu ---
            raw_sents = split_into_sentences(paragraph)
            para_sentences: list[Sentence] = []
            sent_idx = 0
            for raw in raw_sents:
                text = raw.strip()
                if not text:
                    continue
                para_sentences.append(
                    Sentence(
                        page_id=page_id,
                        title=title,
                        section_title=section_title,
                        paragraph_index=para_idx,
                        sentence_index=sent_idx,
                        text=text,
                    )
                )
                sent_idx += 1

            if not para_sentences:
                continue

            # --- chunkowanie akapitu ---
            para_chunks = chunk_paragraph_sentences(
                para_sentences,
                max_sent_per_chunk=max_sent_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
                overlap_sent=overlap_sent,
            )
            all_chunks.extend(para_chunks)

    return all_chunks


# ---------------------------------------------------------------------------
# build_wiki_chunks  (batch / wysoki poziom)
# ---------------------------------------------------------------------------

def build_wiki_chunks(
    articles: list[Article],
    max_sent_per_chunk: int = MAX_SENT_PER_CHUNK,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    overlap_sent: int = OVERLAP_SENT,
) -> list[Chunk]:
    """Funkcja wysokopoziomowa: chunkowanie wielu artykułów naraz.

    Dla każdego artykułu wywołuje :func:`build_article_chunks` i zwraca
    jedną płaską listę wszystkich obiektów :class:`Chunk`.
    """
    all_chunks: list[Chunk] = []
    for article in articles:
        all_chunks.extend(
            build_article_chunks(
                article,
                max_sent_per_chunk=max_sent_per_chunk,
                max_tokens_per_chunk=max_tokens_per_chunk,
                overlap_sent=overlap_sent,
            )
        )
    return all_chunks


# ---------------------------------------------------------------------------
# Szybki autotest / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Mały przykładowy artykuł do demonstracji.
    demo_article: Article = {
        "page_id": 42,
        "title": "Kraków",
        "sections": [
            {
                "section_title": "Historia",
                "paragraphs": [
                    (
                        "Kraków jest jednym z najstarszych miast w Polsce. "
                        "Pierwsza wzmianka pochodzi z X wieku. "
                        "Miasto było stolicą Polski do 1596 roku. "
                        "Wawel stanowił siedzibę królów polskich."
                    ),
                    (
                        "W XIX wieku Kraków znalazł się pod zaborem austriackim. "
                        "Mimo to miasto zachowało swój polski charakter."
                    ),
                ],
            },
            {
                "section_title": "Geografia",
                "paragraphs": [
                    "Kraków leży nad Wisłą, w południowej Polsce.",
                ],
            },
        ],
    }

    print("=" * 60)
    print("SENTENCES")
    print("=" * 60)
    sents = article_to_sentences(demo_article)
    for s in sents:
        print(f"  [{s.section_title} | p{s.paragraph_index} s{s.sentence_index}] {s.text}")

    print()
    print("=" * 60)
    print(f"CHUNKS  (max_sent={MAX_SENT_PER_CHUNK}, "
          f"max_tok={MAX_TOKENS_PER_CHUNK}, overlap={OVERLAP_SENT})")
    print("=" * 60)
    chunks = build_article_chunks(demo_article)
    for c in chunks:
        print(f"  {c.chunk_id}  ({c.num_tokens} tok, sents={c.sentence_indices})")
        print(f"    → {c.text[:120]}{'…' if len(c.text) > 120 else ''}")

    print()
    print(f"Total: {len(sents)} sentences → {len(chunks)} chunks")
