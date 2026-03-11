"""
Ollama Model Setup — downloads and prepares models for local inference.
========================================================================

Automates the full setup flow:
    1. Check if Ollama is installed and running
    2. Pull specified models (or all 3 by default)
    3. Verify each model responds correctly
    4. Generate .env configuration for llm_client.py

Usage (CLI)
-----------
    # Install all 3 models:
    python -m local_builder.setup_ollama

    # Install specific model:
    python -m local_builder.setup_ollama --model bielik-11b

    # Just verify existing models:
    python -m local_builder.setup_ollama --verify-only

    # Generate .env for a model:
    python -m local_builder.setup_ollama --model llama-3.1-8b --gen-env

Usage (Python)
--------------
    from local_builder.setup_ollama import setup_model, setup_all

    setup_model("bielik-11b")
    # or
    setup_all()

Requirements
------------
    - Ollama installed (https://ollama.com/download)
    - Sufficient VRAM / disk space for chosen models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from local_builder.model_registry import (
    MODELS,
    get_env_config,
    get_model,
    list_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# OLLAMA DETECTION
# ═══════════════════════════════════════════════════════════════════════════


def check_ollama_installed() -> bool:
    """Checks if the Ollama CLI is available in PATH."""
    return shutil.which("ollama") is not None


def check_ollama_running() -> bool:
    """Checks if the Ollama server is responding."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_installed_models() -> list[str]:
    """Returns list of model tags currently installed in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        # Parse output: NAME    ID    SIZE    MODIFIED
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


def is_model_installed(ollama_tag: str) -> bool:
    """Checks if a specific model is already pulled in Ollama."""
    installed = get_installed_models()
    # Ollama tags can match partially (e.g., "llama3.1:8b" matches "llama3.1:8b")
    for m in installed:
        if ollama_tag in m or m in ollama_tag:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# MODEL INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════


def pull_model(model_name: str) -> bool:
    """Pulls a model via Ollama CLI.

    Parameters
    ----------
    model_name : str
        Short name from registry: "bielik-11b", "llama-3.1-8b", "deepseek-r1-7b"

    Returns
    -------
    bool
        True if pull succeeded.
    """
    model = get_model(model_name)
    tag = model["ollama_tag"]

    log.info(
        "═" * 50
        + "\n  Pulling: %s"
        + "\n  Tag:     %s"
        + "\n  Size:    ~%.1f GB"
        + "\n" + "═" * 50,
        model["display_name"], tag, model["size_gb"],
    )

    if is_model_installed(tag):
        log.info("  ✅ Already installed — skipping pull.")
        return True

    log.info("  ⏳ Downloading... (this may take several minutes)")

    try:
        process = subprocess.Popen(
            ["ollama", "pull", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                log.info("  [ollama] %s", line)

        process.wait()

        if process.returncode == 0:
            log.info("  ✅ Pull complete: %s", model["display_name"])
            return True
        else:
            log.error("  ❌ Pull failed with code %d", process.returncode)
            return False

    except FileNotFoundError:
        log.error(
            "  ❌ 'ollama' not found. Install from: https://ollama.com/download"
        )
        return False
    except Exception as e:
        log.error("  ❌ Pull error: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# MODEL VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════


def verify_model(model_name: str) -> dict[str, Any]:
    """Sends a test prompt to a model and verifies it responds correctly.

    Parameters
    ----------
    model_name : str
        Short name from registry.

    Returns
    -------
    dict with keys:
        - model: str
        - success: bool
        - response: str (first 200 chars)
        - tokens_per_sec: float (estimated)
        - error: str | None
    """
    model = get_model(model_name)
    tag = model["llm_model_name"]

    log.info("Verifying: %s (%s)...", model["display_name"], tag)

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="local",
        )

        t0 = time.perf_counter()

        response = client.chat.completions.create(
            model=tag,
            messages=[
                {"role": "system", "content": "Odpowiadaj krótko po polsku."},
                {"role": "user", "content": "Ile województw ma Polska? Odpowiedz jednym zdaniem."},
            ],
            max_tokens=50,
            temperature=0.1,
        )

        t1 = time.perf_counter()

        content = response.choices[0].message.content.strip()
        usage = response.usage
        elapsed = t1 - t0

        compl_tokens = usage.completion_tokens if usage else len(content.split())
        tokens_per_sec = compl_tokens / elapsed if elapsed > 0 else 0

        log.info(
            "  ✅ Response: '%s' (%.1f tok/s, %.2fs)",
            content[:100], tokens_per_sec, elapsed,
        )

        return {
            "model": model_name,
            "success": True,
            "response": content[:200],
            "tokens_per_sec": round(tokens_per_sec, 1),
            "elapsed": round(elapsed, 2),
            "error": None,
        }

    except Exception as e:
        log.error("  ❌ Verification failed: %s", e)
        return {
            "model": model_name,
            "success": False,
            "response": "",
            "tokens_per_sec": 0,
            "elapsed": 0,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════
# .ENV GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def generate_env_lines(model_name: str, backend: str = "ollama") -> str:
    """Generates .env lines for a given model.

    Returns
    -------
    str
        Ready-to-paste .env content.
    """
    config = get_env_config(model_name, backend)
    model = get_model(model_name)

    lines = [
        f"# {model['display_name']} ({model['params']}, {model['quantization']})",
        f"LLM_BACKEND={config['LLM_BACKEND']}",
        f"LLM_MODEL={config['LLM_MODEL']}",
        f"LLM_BASE_URL={config['LLM_BASE_URL']}",
    ]
    return "\n".join(lines)


def write_env_file(
    model_name: str,
    env_path: str | None = None,
    backend: str = "ollama",
    preserve_other_vars: bool = True,
) -> str:
    """Writes/updates .env file with model configuration.

    If preserve_other_vars is True, only LLM_* vars are replaced,
    other variables (like together_api_key) are preserved.

    Returns
    -------
    str
        Path to the .env file.
    """
    if env_path is None:
        project_root = Path(__file__).resolve().parent.parent
        env_path = str(project_root / ".env")

    config = get_env_config(model_name, backend)

    # Read existing .env
    existing_lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            existing_lines = f.readlines()

    # Filter out old LLM_* lines
    llm_keys = {"LLM_BACKEND", "LLM_MODEL", "LLM_BASE_URL"}
    if preserve_other_vars:
        filtered = []
        for line in existing_lines:
            stripped = line.strip()
            # Keep non-LLM lines and comments/blanks
            if not any(stripped.startswith(k + "=") for k in llm_keys):
                filtered.append(line)
        existing_lines = filtered

    # Append new LLM config
    model = get_model(model_name)
    new_lines = existing_lines.copy()
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines.append("\n")

    new_lines.append(f"\n# --- Local LLM: {model['display_name']} ---\n")
    new_lines.append(f"LLM_BACKEND={config['LLM_BACKEND']}\n")
    new_lines.append(f"LLM_MODEL={config['LLM_MODEL']}\n")
    new_lines.append(f"LLM_BASE_URL={config['LLM_BASE_URL']}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    log.info("Updated %s → %s (%s)", env_path, model_name, backend)
    return env_path


# ═══════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL ORCHESTRATORS
# ═══════════════════════════════════════════════════════════════════════════


def setup_model(model_name: str) -> dict[str, Any]:
    """Full setup for a single model: pull + verify.

    Returns
    -------
    dict with setup results.
    """
    result = {"model": model_name, "pulled": False, "verified": False, "error": None}

    if not check_ollama_installed():
        result["error"] = (
            "Ollama not found. Install from https://ollama.com/download"
        )
        log.error(result["error"])
        return result

    if not check_ollama_running():
        log.warning(
            "Ollama server not running. Attempting to start..."
        )
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(3)
            if not check_ollama_running():
                result["error"] = "Could not start Ollama server."
                log.error(result["error"])
                return result
            log.info("Ollama server started.")
        except Exception as e:
            result["error"] = f"Failed to start Ollama: {e}"
            log.error(result["error"])
            return result

    # Pull
    result["pulled"] = pull_model(model_name)
    if not result["pulled"]:
        result["error"] = "Model pull failed."
        return result

    # Verify
    verify_result = verify_model(model_name)
    result["verified"] = verify_result["success"]
    result["tokens_per_sec"] = verify_result.get("tokens_per_sec", 0)
    if not result["verified"]:
        result["error"] = verify_result.get("error", "Verification failed")

    return result


def setup_all() -> list[dict[str, Any]]:
    """Sets up all 3 models: pull + verify each.

    Returns
    -------
    list[dict] with per-model results.
    """
    log.info("=" * 60)
    log.info("  LOCAL MODEL SETUP — installing 3 models via Ollama")
    log.info("=" * 60)

    all_results = []
    for model_name in MODELS:
        result = setup_model(model_name)
        all_results.append(result)
        log.info("")

    # Summary
    log.info("=" * 60)
    log.info("  SETUP SUMMARY")
    log.info("=" * 60)
    for r in all_results:
        model = get_model(r["model"])
        status = "✅" if r["verified"] else "❌"
        speed = f"{r.get('tokens_per_sec', 0):.0f} tok/s" if r["verified"] else r.get("error", "failed")
        log.info(
            "  %s  %-30s  %s",
            status, model["display_name"], speed,
        )
    log.info("=" * 60)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point for model setup."""
    parser = argparse.ArgumentParser(
        description="Set up local LLM models via Ollama for claims processing.",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Specific model to install. Default: all 3.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify installed models, don't pull.",
    )
    parser.add_argument(
        "--gen-env",
        action="store_true",
        help="Generate .env configuration for the model.",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write model config to project .env file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="List all available models.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show Ollama status and installed models.",
    )

    args = parser.parse_args()

    # --- List models ---
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 70)
        for m in list_models():
            print(
                f"  {m['key']:<18s}  {m['display_name']:<35s}  "
                f"{m['size_gb']:.1f} GB  VRAM ≥{m['min_vram_gb']} GB"
            )
        print()
        return

    # --- Status ---
    if args.status:
        print(f"\nOllama installed: {check_ollama_installed()}")
        print(f"Ollama running:   {check_ollama_running()}")
        installed = get_installed_models()
        print(f"Installed models: {len(installed)}")
        for m in installed:
            print(f"  - {m}")
        print()
        return

    # --- Gen env ---
    if args.gen_env:
        model_name = args.model
        if not model_name:
            print("ERROR: --gen-env requires --model")
            sys.exit(1)
        print(generate_env_lines(model_name))
        return

    # --- Write env ---
    if args.write_env:
        model_name = args.model
        if not model_name:
            print("ERROR: --write-env requires --model")
            sys.exit(1)
        path = write_env_file(model_name)
        print(f"Written to: {path}")
        return

    # --- Verify only ---
    if args.verify_only:
        targets = [args.model] if args.model else list(MODELS.keys())
        for name in targets:
            verify_model(name)
        return

    # --- Full setup ---
    if args.model:
        setup_model(args.model)
    else:
        setup_all()


if __name__ == "__main__":
    main()
