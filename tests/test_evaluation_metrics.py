"""누락·실패를 분모에서 숨기거나 검색 점수를 답변 점수로 바꾸는 회귀를 막는다."""
import copy

import pytest

from evaluation.metrics import score_record, summarize


def turn(**changes):
    value = {"turn": 1, "message": "payments-service 장애", "expected_route": "incident",
             "expected_behavior": "diagnose", "required_monitoring_targets": ["payments-service"],
             "allowed_monitoring_targets": ["payments-service"],
             "expected_retrieval_document_union": ["cpu-spike", "memory-leak"],
             "required_runbooks_by_system": {"payments-service": ["cpu-spike", "memory-leak"]},
             "allowed_extra_runbooks_by_system": {}, "criteria": ["추가 확인 안내"], "forbidden": []}
    value.update(changes)
    return value


def record(**changes):
    value = {"case_id": "C01", "turn": 1, "repetition": 1, "status": "ok",
             "response": {"reply": "1) 확인", "metadata": {"route": "incident"}, "events": [
                 {"type": "TaskCompleted", "source": "monitoring", "target": "payments-service"},
                 {"type": "RunbookRetrieved", "document_ids": ["cpu-spike", "cpu-spike"]}]},
             "http_elapsed_ms": 100, "trace_elapsed_ms": 70, "model_usage": None,
             "state": None, "review": None}
    value.update(changes)
    return value


def test_missing_documents_and_duplicate_chunks_have_correct_denominator():
    result = score_record(turn(), record())
    assert result["retrieval_recall"] == 0.5
    assert result["retrieval_found"] == 1
    assert result["retrieval_required"] == 2
    assert result["retrieval_precision"] == 1
    assert result["answer_pass"] is None


def test_failed_request_stays_in_recall_and_target_denominators():
    result = score_record(turn(), record(status="error", response=None))
    assert result["request_success"] == 0
    assert result["retrieval_recall"] == 0
    assert result["target_exact"] == 0


def test_non_retrieval_turn_is_not_zero_recall():
    expected = turn(expected_retrieval_document_union=[], required_runbooks_by_system={})
    result = score_record(expected, record())
    assert result["retrieval_recall"] is None


def test_clarification_requires_event_and_no_monitoring_or_runbook():
    expected = turn(expected_behavior="clarify_target", required_monitoring_targets=[],
                    allowed_monitoring_targets=[], expected_retrieval_document_union=[])
    r = record()
    r["response"]["events"].append({"type": "TargetConfirmationRequired"})
    assert score_record(expected, r)["clarification_safe"] == 0
    r["response"]["events"] = [{"type": "TargetConfirmationRequired"}]
    assert score_record(expected, r)["clarification_safe"] == 1


def test_unmeasured_usage_is_not_zero_and_latency_boundaries_are_separate():
    r = record()
    scored = [{**r, "scores": score_record(turn(), r)}]
    result = summarize(scored)
    assert result["openai_total_tokens"] is None
    assert result["http_latency_ms"]["mean"] == 100
    assert result["trace_latency_ms"]["mean"] == 70
    assert result["answer_pass"]["eligible"] == 0


def test_incomplete_run_is_visible_in_summary():
    rows = [record(), record(case_id="C02", status="not_run", response=None,
                            http_elapsed_ms=None, trace_elapsed_ms=None)]
    for r in rows:
        r["scores"] = score_record(turn(), r)
    result = summarize(rows)
    assert result["request_success"] == {"passed": 1, "eligible": 2, "rate": 0.5}
    assert result["http_latency_ms"]["count"] == 1
    assert result["retrieval_micro_recall"] == 0.25
    assert result["not_run"] == 1


def test_extra_monitoring_target_fails_even_when_required_target_present():
    r = record()
    r["response"]["events"].append({"type": "TaskCompleted", "source": "monitoring", "target": "auth-service"})
    assert score_record(turn(), r)["target_exact"] == 0


def test_manual_pass_requires_complete_criteria_and_grounding_not_just_format():
    r = record(review={"reviewer": "manual", "notes": "근거 대조", "answer_behavior_grade": "pass",
                       "claim_grounding_grade": "fail", "target_grade": "pass",
                       "criteria": [{"criterion": "추가 확인 안내", "grade": "pass"}]})
    assert score_record(turn(), r)["answer_pass"] == 0
    r["review"]["claim_grounding_grade"] = "pass"
    assert score_record(turn(), r)["answer_pass"] == 1
    r["review"]["criteria"] = []
    assert score_record(turn(), r)["answer_pass"] is None


def test_model_usage_does_not_double_count_cached_or_local_tokens():
    r = record(model_usage=[
        {"provider": "openai", "requested_model": "gpt-4o", "resolved_model": "gpt-4o-version",
         "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                   "input_token_details": {"cache_read": 50}}},
        {"provider": "ollama", "usage": {"input_tokens": 20, "output_tokens": 3, "total_tokens": 23}}])
    r["scores"] = score_record(turn(), r)
    result = summarize([r])
    assert result["openai_total_tokens"] == 110
    assert result["ollama_total_tokens"] == 23
    assert result["model_calls"] == 2


def test_scoring_does_not_modify_evidence():
    expected, r = turn(), record()
    snapshot = copy.deepcopy((expected, r))
    score_record(expected, r)
    assert (expected, r) == snapshot


def test_not_applicable_manual_criterion_does_not_hide_a_real_failure():
    r = record(review={"reviewer": "manual", "notes": "첫 기준 실패, 둘째는 해당 없음",
                       "answer_behavior_grade": "fail", "claim_grounding_grade": "pass", "target_grade": "pass",
                       "criteria": [{"criterion": "추가 확인 안내", "grade": "fail"},
                                    {"criterion": "선택한 경우의 기준", "grade": "not_applicable"}]})
    assert score_record(turn(criteria=["추가 확인 안내", "선택한 경우의 기준"]), r)["answer_pass"] == 0


def test_forbidden_claim_failure_overrides_overall_manual_pass():
    r = record(review={"reviewer": "manual", "notes": "금지 주장 발견", "answer_behavior_grade": "pass",
                       "claim_grounding_grade": "pass", "target_grade": "pass",
                       "criteria": [{"criterion": "추가 확인 안내", "grade": "pass"}],
                       "forbidden_checks": [{"criterion": "재시작 완료 주장", "grade": "fail"}]})
    assert score_record(turn(forbidden=["재시작 완료 주장"]), r)["answer_pass"] == 0
