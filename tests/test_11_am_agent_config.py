"""
Test AM benchmark agent configuration.

Verifies for each registered agent:
1. eval() uses label_original (not label) for the is_correct comparison.
2. A correct prediction (model_label == label_original) → is_correct=True.
3. A wrong prediction → is_correct=False.
4. The question passed to the pipeline includes answer choices from metadata.
5. original_label in the result is the answer index, NOT "SUPPORTS".

All LLM/BM25/RAG calls are mocked — no external services required.
"""
from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_claim(correct_index: str = "2") -> dict:
    """Returns a fake AM benchmark claim dict as returned by eval_loop."""
    answers = ["Odpowiedź A", "Odpowiedź B", "Odpowiedź C", "Odpowiedź D"]
    return {
        "id": 1,
        "claim_text": "Jakie jest pytanie testowe?",
        "label": "SUPPORTS",          # always SUPPORTS in AM benchmark DB
        "label_original": correct_index,
        "metadata": json.dumps({
            "answers": str(answers),
            "correct_answer_index": correct_index,
        }),
        "topic": "Test",
        "source": "am_benchmark",
    }


def _make_ask_mock(answer: str) -> MagicMock:
    """Returns a mock for the ask() pipeline function."""
    mock = MagicMock()
    mock.return_value = {
        "answer": answer,
        "total_tokens": 100,
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "sub_claims": ["sub-twierdzenie testowe"],
        "evidence_summary": [],
        "evidence": [],
        "reasoner_votes": [],
        "consolidation": "",
        "nli_score": 1.0,
    }
    return mock


# ---------------------------------------------------------------------------
# Individual agent tests
# ---------------------------------------------------------------------------

def _test_agent(agent_module: str, ask_fn: str, agent_class_name: str) -> tuple[bool, str | None]:
    """Generic test for a single agent."""
    import importlib
    mod = importlib.import_module(agent_module)
    AgentClass = getattr(mod, agent_class_name)
    agent = AgentClass()

    correct_index = "2"
    wrong_index = "0"
    claim = _make_fake_claim(correct_index)

    patch_target = f"{agent_module}.{ask_fn}"

    # ── Test 1 & 2: correct prediction → is_correct=True ─────────────────────
    mock_ask = _make_ask_mock(correct_index)
    with patch(patch_target, mock_ask):
        result = agent.eval(claim)

    if result.get("is_correct") is not True and result.get("is_correct") != 1:
        return False, (
            f"{agent_class_name}: correct prediction '{correct_index}' → "
            f"is_correct={result.get('is_correct')!r} (expected True). "
            f"original_label={result.get('original_label')!r}"
        )

    if result.get("original_label") != correct_index:
        return False, (
            f"{agent_class_name}: original_label={result.get('original_label')!r}, "
            f"expected '{correct_index}'. Bug: using 'label' instead of 'label_original'."
        )

    # ── Test 3: wrong prediction → is_correct=False ───────────────────────────
    mock_ask_wrong = _make_ask_mock(wrong_index)
    with patch(patch_target, mock_ask_wrong):
        result_wrong = agent.eval(claim)

    if result_wrong.get("is_correct") is not False and result_wrong.get("is_correct") != 0:
        return False, (
            f"{agent_class_name}: wrong prediction '{wrong_index}' → "
            f"is_correct={result_wrong.get('is_correct')!r} (expected False)"
        )

    # ── Test 4: answer choices included in the call to ask() ──────────────────
    mock_ask_capture = _make_ask_mock(correct_index)
    with patch(patch_target, mock_ask_capture):
        agent.eval(claim)

    call_args = mock_ask_capture.call_args
    if call_args is None:
        return False, f"{agent_class_name}: ask() was never called"

    passed_question = call_args[0][0] if call_args[0] else ""
    if "Odpowiedź A" not in passed_question or "0:" not in passed_question:
        return False, (
            f"{agent_class_name}: answer choices not injected into ask() call. "
            f"Got: {passed_question[:200]!r}"
        )

    return True, None


def test_am_agent_config() -> tuple[bool, float, str | None]:
    start = time.time()

    agents_to_test = [
        ("claims_processing.agents.uam.bm25_claim_decomp", "ask", "ClaimDecompBM25Agent"),
        ("claims_processing.agents.uam.rag_claim_decomp",  "ask", "ClaimDecompRAGAgent"),
        ("claims_processing.agents.uam.fewshot_cot_rag",   "ask", "FewShotCoTAgent"),
        ("claims_processing.agents.uam.fewshot_cot_debate_rag", "debate_ask", "DebateCoTAgent"),
    ]

    for module, ask_fn, class_name in agents_to_test:
        ok, err = _test_agent(module, ask_fn, class_name)
        if not ok:
            return False, time.time() - start, err

    return True, time.time() - start, None


if __name__ == "__main__":
    ok, elapsed, err = test_am_agent_config()
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] test_am_agent_config ({elapsed:.2f}s)")
    if err:
        print(f"  Error: {err}")
    sys.exit(0 if ok else 1)
