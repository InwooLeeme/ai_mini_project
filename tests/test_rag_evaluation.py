"""검색 평가가 누락·오류·빈 결과를 성공으로 세지 않는지 검증한다."""
import importlib


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
