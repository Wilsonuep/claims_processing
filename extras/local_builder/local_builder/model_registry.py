"""
Model Registry — metadata for all supported self-hosted models.
================================================================

Central source of truth for model names, Ollama tags, HuggingFace repos,
VRAM requirements, quantization variants, and LLM_MODEL values used
by gen_agent/llm_client.py.

Usage
-----
    from local_builder.model_registry import MODELS, get_model

    bielik = get_model("bielik-11b")
    print(bielik["ollama_tag"])       # "hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M"
    print(bielik["llm_model_name"])   # model name for LLM_MODEL env var
"""

from __future__ import annotations

from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

MODELS: dict[str, dict[str, Any]] = {
    # ─────────────────────────────────────────────────────────────────────
    # Bielik 11B — Polish specialist (best for Polish fact-checking)
    # ─────────────────────────────────────────────────────────────────────
    "bielik-11b": {
        "display_name": "Bielik 11B v2.3 Instruct",
        "family": "bielik",
        "params": "11B",
        "language": "Polish (native)",
        "description": (
            "Polish-language LLM trained by SpeakLeash/AGH Cyfronet. "
            "Best-in-class for Polish NLP tasks."
        ),

        # --- Ollama ---
        "ollama_tag": "hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M",
        "ollama_pull_cmd": "ollama pull hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M",

        # --- HuggingFace ---
        "hf_repo": "speakleash/Bielik-11B-v2.3-Instruct",
        "hf_gguf_repo": "speakleash/Bielik-11B-v2.3-Instruct-GGUF",

        # --- llm_client.py compatibility ---
        "llm_model_name": "hf.co/speakleash/Bielik-11B-v2.3-Instruct-GGUF:Q4_K_M",

        # --- Hardware ---
        "quantization": "Q4_K_M",
        "size_gb": 7.0,
        "min_vram_gb": 8,
        "recommended_vram_gb": 12,

        # --- Performance (Q4_K_M, consumer GPU) ---
        "est_tokens_per_sec": {
            "RTX 3060 12GB": 30,
            "RTX 3070 8GB": 28,
            "RTX 4060 8GB": 32,
            "RTX 4070 12GB": 38,
        },

        # --- Quantization variants ---
        "variants": {
            "Q4_K_M": {"size_gb": 7.0, "min_vram_gb": 8},
            "Q5_K_M": {"size_gb": 8.2, "min_vram_gb": 10},
            "Q8_0":   {"size_gb": 12.0, "min_vram_gb": 14},
            "F16":    {"size_gb": 22.0, "min_vram_gb": 24},
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # Llama 3.1 8B — Strong multilingual baseline
    # ─────────────────────────────────────────────────────────────────────
    "llama-3.1-8b": {
        "display_name": "Llama 3.1 8B Instruct",
        "family": "llama",
        "params": "8B",
        "language": "Multilingual (strong Polish)",
        "description": (
            "Meta's Llama 3.1 8B — strong multilingual model with "
            "good Polish capability. Solid baseline for comparison."
        ),

        # --- Ollama ---
        "ollama_tag": "llama3.1:8b",
        "ollama_pull_cmd": "ollama pull llama3.1:8b",

        # --- HuggingFace ---
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "hf_gguf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",

        # --- llm_client.py compatibility ---
        "llm_model_name": "llama3.1:8b",

        # --- Hardware ---
        "quantization": "Q4_K_M",
        "size_gb": 4.9,
        "min_vram_gb": 6,
        "recommended_vram_gb": 8,

        # --- Performance (Q4_K_M, consumer GPU) ---
        "est_tokens_per_sec": {
            "RTX 3060 12GB": 42,
            "RTX 3070 8GB": 45,
            "RTX 4060 8GB": 42,
            "RTX 4070 12GB": 55,
        },

        # --- Quantization variants ---
        "variants": {
            "Q4_K_M": {"size_gb": 4.9, "min_vram_gb": 6},
            "Q5_K_M": {"size_gb": 5.7, "min_vram_gb": 7},
            "Q8_0":   {"size_gb": 8.5, "min_vram_gb": 10},
            "F16":    {"size_gb": 16.1, "min_vram_gb": 18},
        },
    },

    # ─────────────────────────────────────────────────────────────────────
    # DeepSeek R1 Distill 7B — Reasoning specialist
    # ─────────────────────────────────────────────────────────────────────
    "deepseek-r1-7b": {
        "display_name": "DeepSeek R1 Distill Qwen 7B",
        "family": "deepseek",
        "params": "7B",
        "language": "Multilingual (reasoning-focused)",
        "description": (
            "DeepSeek R1 distilled into Qwen 7B — built-in chain-of-thought "
            "reasoning capability. Interesting for CoT/debate pipelines."
        ),

        # --- Ollama ---
        "ollama_tag": "deepseek-r1:7b",
        "ollama_pull_cmd": "ollama pull deepseek-r1:7b",

        # --- HuggingFace ---
        "hf_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "hf_gguf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",

        # --- llm_client.py compatibility ---
        "llm_model_name": "deepseek-r1:7b",

        # --- Hardware ---
        "quantization": "Q4_K_M",
        "size_gb": 4.7,
        "min_vram_gb": 6,
        "recommended_vram_gb": 8,

        # --- Performance (Q4_K_M, consumer GPU) ---
        "est_tokens_per_sec": {
            "RTX 3060 12GB": 40,
            "RTX 3070 8GB": 42,
            "RTX 4060 8GB": 43,
            "RTX 4070 12GB": 52,
        },

        # --- Quantization variants ---
        "variants": {
            "Q4_K_M": {"size_gb": 4.7, "min_vram_gb": 6},
            "Q5_K_M": {"size_gb": 5.5, "min_vram_gb": 7},
            "Q8_0":   {"size_gb": 8.1, "min_vram_gb": 10},
            "F16":    {"size_gb": 15.0, "min_vram_gb": 17},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def get_model(name: str) -> dict[str, Any]:
    """Returns model metadata by short name.

    Parameters
    ----------
    name : str
        Short model name: "bielik-11b", "llama-3.1-8b", "deepseek-r1-7b".

    Raises
    ------
    KeyError
        If model is not found.
    """
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise KeyError(
            f"Model '{name}' not found. Available: {available}"
        )
    return MODELS[name]


def list_models() -> list[dict[str, Any]]:
    """Returns a list of all models with their key metadata."""
    result = []
    for key, m in MODELS.items():
        result.append({
            "key": key,
            "display_name": m["display_name"],
            "params": m["params"],
            "language": m["language"],
            "size_gb": m["size_gb"],
            "min_vram_gb": m["min_vram_gb"],
            "ollama_tag": m["ollama_tag"],
            "llm_model_name": m["llm_model_name"],
        })
    return result


def get_env_config(model_name: str, backend: str = "ollama") -> dict[str, str]:
    """Returns .env variable values for a given model + backend.

    Parameters
    ----------
    model_name : str
        Short model name.
    backend : str
        "ollama", "vllm", or "llamacpp".

    Returns
    -------
    dict with keys: LLM_BACKEND, LLM_MODEL, LLM_BASE_URL
    """
    m = get_model(model_name)

    base_urls = {
        "ollama": "http://localhost:11434/v1",
        "vllm": "http://localhost:8000/v1",
        "llamacpp": "http://localhost:8080/v1",
    }

    if backend == "ollama":
        model_val = m["llm_model_name"]
    elif backend == "vllm":
        model_val = m["hf_repo"]
    elif backend == "llamacpp":
        model_val = m["llm_model_name"]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return {
        "LLM_BACKEND": backend,
        "LLM_MODEL": model_val,
        "LLM_BASE_URL": base_urls[backend],
    }
