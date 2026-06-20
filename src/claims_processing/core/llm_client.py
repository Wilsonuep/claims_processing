"""
Universal LLM Client Factory
==============================

Abstrakcja nad backendem LLM — przełącza się między Together.ai,
Ollama, vLLM i llama.cpp za pomocą zmiennych środowiskowych.

Zmienne środowiskowe
---------------------
    LLM_BACKEND   — "together" | "ollama" | "vllm" | "llamacpp"
                     (domyślnie: "together")
    LLM_MODEL     — nazwa modelu (domyślnie: "openai/gpt-oss-20b")
    LLM_BASE_URL  — nadpisuje domyślny URL backendu (opcjonalne)

Przykłady .env
--------------
    # Together.ai (cloud):
    LLM_BACKEND=together
    LLM_MODEL=openai/gpt-oss-20b

    # Ollama (local, Bielik):
    LLM_BACKEND=ollama
    LLM_MODEL=bielik-11b

    # Ollama (local, Llama):
    LLM_BACKEND=ollama
    LLM_MODEL=llama3.1:8b

    # Ollama (local, DeepSeek):
    LLM_BACKEND=ollama
    LLM_MODEL=deepseek-r1:7b

    # vLLM server:
    LLM_BACKEND=vllm
    LLM_MODEL=speakleash/Bielik-11B-v2.3-Instruct
    LLM_BASE_URL=http://localhost:8000/v1

    # llama.cpp server:
    LLM_BACKEND=llamacpp
    LLM_MODEL=bielik-11b-q4
    LLM_BASE_URL=http://localhost:8080/v1

Użycie w agentach
-----------------
    from claims_processing.core.llm_client import client, MODEL

    response = client.chat.completions.create(
        model=MODEL,
        messages=[...],
    )
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

LLM_BACKEND: str = os.getenv("LLM_BACKEND", "together")
MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")

# Domyślne URL-e per backend
_DEFAULT_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
    "llamacpp": "http://localhost:8080/v1",
}

# ---------------------------------------------------------------------------
# Informacja o backendzie — przydatne w logach
# ---------------------------------------------------------------------------

IS_LOCAL: bool = LLM_BACKEND in ("ollama", "vllm", "llamacpp")
"""True jeśli backend to self-hosted model (nie cloud API)."""

BACKEND_INFO: dict[str, str] = {
    "backend": LLM_BACKEND,
    "model": MODEL,
    "base_url": LLM_BASE_URL or _DEFAULT_URLS.get(LLM_BACKEND, ""),
    "is_local": str(IS_LOCAL),
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _create_client():
    """Tworzy klienta LLM na podstawie konfiguracji."""
    if LLM_BACKEND == "together":
        from together import Together

        api_key = os.getenv("together_api_key")
        if not api_key:
            log.warning(
                "together_api_key not set — Together client may fail."
            )
        log.info(
            "LLM backend: Together.ai (model=%s)", MODEL,
        )
        return Together(api_key=api_key)

    if LLM_BACKEND in ("ollama", "vllm", "llamacpp"):
        from openai import OpenAI

        base_url = LLM_BASE_URL or _DEFAULT_URLS[LLM_BACKEND]
        log.info(
            "LLM backend: %s (model=%s, url=%s)",
            LLM_BACKEND, MODEL, base_url,
        )
        return OpenAI(base_url=base_url, api_key="local")

    raise ValueError(
        f"Unknown LLM_BACKEND: '{LLM_BACKEND}'. "
        f"Expected: together, ollama, vllm, llamacpp."
    )


client = _create_client()
"""Globalny klient LLM — importowany przez agentów."""


def make_client(
    model: str | None = None,
    backend: str | None = None,
    base_url: str | None = None,
):
    """Tworzy nowego klienta LLM dla podanego modelu.

    Parametry
    ---------
    model : str | None
        Nazwa modelu. None = użyj globalnego MODEL.
    backend : str | None
        Backend LLM. None = użyj globalnego LLM_BACKEND.
    base_url : str | None
        Nadpisuje URL backendu. None = domyślny URL dla backendu.

    Zwraca
    ------
    (client, model_name) : tuple
        Nowy klient i nazwa modelu.
    """
    _backend = backend or LLM_BACKEND
    _model = model or MODEL
    _url = base_url or LLM_BASE_URL

    if _backend == "together":
        from together import Together
        api_key = os.getenv("together_api_key")
        log.info("make_client: Together.ai (model=%s)", _model)
        return Together(api_key=api_key), _model

    if _backend in ("ollama", "vllm", "llamacpp"):
        from openai import OpenAI
        resolved_url = _url or _DEFAULT_URLS[_backend]
        log.info("make_client: %s (model=%s, url=%s)", _backend, _model, resolved_url)
        return OpenAI(base_url=resolved_url, api_key="local"), _model

    raise ValueError(
        f"Unknown LLM_BACKEND: '{_backend}'. "
        f"Expected: together, ollama, vllm, llamacpp."
    )
