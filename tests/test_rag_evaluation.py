"""검색 평가가 누락·오류·빈 결과를 성공으로 세지 않는지 검증한다."""
import importlib
from collections import Counter
from types import SimpleNamespace

import pytest


def score(case, result):
    return importlib.import_module("eval.run_rag_evaluation").score(case, result)


def test_partial_retrieval_and_context_are_scored_separately():
    case = {"expected_action": "retrieve", "required_document_ids": ["a", "b"],
            "allowed_extra_document_ids": [], "evidence": [
                {"document_id": "a", "required_evidence": [{"text": "1. 첫 절차"}]},
                {"document_id": "b", "required_evidence": [{"text": "2. 조건부 조치"}]}]}
    result = {"status": "ok", "documents": [{"document_id": "a", "text": "1. 첫 절차"}]}
    assert score(case, result) == {
        "recall": 0.5, "precision": 1.0, "complete": False,
        "evidence_exact_recall": 0.5, "correct_abstention": None}


def test_error_is_not_correct_abstention():
    case = {"expected_action": "abstain", "required_document_ids": [], "evidence": []}
    assert score(case, {"status": "error", "documents": []}) is None
    assert score(case, {"status": "ok", "documents": []})["correct_abstention"] is True


def test_empty_success_is_zero_recall_for_positive_case():
    case = {"expected_action": "retrieve", "required_document_ids": ["a"],
            "allowed_extra_document_ids": [], "evidence": []}
    result = score(case, {"status": "ok", "documents": []})
    assert result["recall"] == result["precision"] == 0
    assert result["complete"] is False


def test_comparison_pairs_same_input_and_alternates_order():
    module = importlib.import_module("eval.run_rag_evaluation")
    cases = [{"id": "A"}, {"id": "B"}]
    runs = list(module.evaluation_inputs(cases, "compare", 2))
    assert [(r, c["id"], policy) for r, c, policy in runs] == [
        (1, "A", "model"), (1, "A", "explicit_title"),
        (1, "B", "explicit_title"), (1, "B", "model"),
        (2, "A", "explicit_title"), (2, "A", "model"),
        (2, "B", "model"), (2, "B", "explicit_title"),
    ]
    assert runs[0][1] is runs[1][1] is cases[0]


def test_summary_counts_skipped_calls_without_inventing_missing_measurements():
    module = importlib.import_module("eval.run_rag_evaluation")
    rows = [
        {"status": "ok", "scores": None, "elapsed_ms": 100,
         "usage": {"selector_called": False, "timings_ms": {"embedding": 90, "selection": 1}}},
        {"status": "ok", "scores": None, "elapsed_ms": 300,
         "usage": {"selector_called": True, "timings_ms": {"embedding": 95, "selection": 200}}},
        {"status": "error", "scores": None, "elapsed_ms": 400},
    ]
    result = module.summarize(rows)
    assert result["errors"] == 1
    assert result["selector_calls"] == {"called": 1, "skipped": 1, "unobserved": 1}
    assert result["latency_ms"]["mean"] == 200
    assert result["stages_ms"]["selection"] == {"mean": 100.5, "n": 2}


def test_four_repetitions_balance_each_case_in_both_positions():
    module = importlib.import_module("eval.run_rag_evaluation")
    runs = list(module.evaluation_inputs([{"id": "A"}, {"id": "B"}], "compare", 4))
    counts = Counter((case["id"], policy, index % 2)
                     for index, (_, case, policy) in enumerate(runs))
    assert len(counts) == 8
    assert set(counts.values()) == {2}


@pytest.mark.parametrize("repetitions", [0, 1, 3, -2])
def test_unbalanced_comparison_is_rejected_before_creating_output(tmp_path, monkeypatch, repetitions):
    import asyncio
    module = importlib.import_module("eval.run_rag_evaluation")
    def forbidden_client():
        pytest.fail("잘못된 반복 수로 MCP 초기화를 시도했습니다")
    monkeypatch.setattr(module, "MCPClient", forbidden_client)
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="짝수"):
        asyncio.run(module.run(SimpleNamespace(
            phase="compare", repetitions=repetitions, output=str(output))))
    assert not output.exists()


def test_warmup_uses_a_separate_input_and_always_loads_the_selector():
    import asyncio
    import json
    module = importlib.import_module("eval.run_rag_evaluation")
    dataset = json.loads((module.ROOT / "eval/rag-dataset.json").read_text())
    calls = []

    class Retriever:
        explicit_title_selection = True

        async def search(self, **payload):
            calls.append((self.explicit_title_selection, payload))

    retriever = Retriever()
    result = asyncio.run(module.warmup_comparison(retriever))
    assert len(calls) == 1
    assert calls[0][0] is False
    assert calls[0][1] not in [case["input"] for case in dataset["cases"]]
    assert set(calls[0][1]) == {"query", "observations"}
    assert result["input"] == calls[0][1]
    assert result["elapsed_ms"] >= 0
    assert retriever.explicit_title_selection is True


@pytest.mark.parametrize("original_policy", [True, False])
def test_warmup_failure_restores_policy_and_is_not_a_scored_result(original_policy):
    import asyncio
    module = importlib.import_module("eval.run_rag_evaluation")

    class Retriever:
        explicit_title_selection = original_policy

        async def search(self, **payload):
            raise TimeoutError("warmup failed")

    retriever = Retriever()
    with pytest.raises(TimeoutError, match="warmup failed"):
        asyncio.run(module.warmup_comparison(retriever))
    assert retriever.explicit_title_selection is original_policy
