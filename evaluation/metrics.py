"""규칙 점수·수동 의미 판정·측정값을 분리한다. None은 미측정/해당 없음이다."""
import math
from statistics import mean, median


def _grade(value):
    return {"pass": 1, "fail": 0}.get(value)


def _manual_score(expected, review):
    if not review or not review.get("reviewer") or not review.get("notes"):
        return None
    criteria = review.get("criteria", [])
    if [c.get("criterion") for c in criteria] != expected.get("criteria", []):
        return None
    grades = [_grade(c.get("grade")) for c in criteria if c.get("grade") != "not_applicable"]
    if "forbidden_checks" in review:
        forbidden = review["forbidden_checks"]
        if [c.get("criterion") for c in forbidden] != expected.get("forbidden", []):
            return None
        grades += [_grade(c.get("grade")) for c in forbidden if c.get("grade") != "not_applicable"]
    grades += [_grade(review.get(k)) for k in (
        "answer_behavior_grade", "claim_grounding_grade", "target_grade")]
    if not grades or any(g is None for g in grades):
        return None
    # answer_behavior_grade는 금지 사항까지 포함한 수동 전체 판정이다.
    return int(all(grades))


def score_record(expected, record):
    response = record.get("response") or {}
    events = response.get("events", [])
    ok = record["status"] == "ok"
    targets = {e["target"] for e in events if e.get("type") == "TaskCompleted"
               and e.get("source") == "monitoring" and isinstance(e.get("target"), str)}
    documents = {e["target"] for e in events if e.get("type") == "TaskCompleted"
                 and e.get("source") == "runbook" and isinstance(e.get("target"), str)}
    for event in events:
        if event.get("type") == "RunbookRetrieved":
            documents.update(event.get("document_ids", []))
    required = set(expected["expected_retrieval_document_union"])
    allowed = required | {doc for docs in expected.get("allowed_extra_runbooks_by_system", {}).values()
                          for doc in docs}
    found = len(required & documents) if ok else 0
    required_targets = set(expected["required_monitoring_targets"])
    allowed_targets = set(expected["allowed_monitoring_targets"])
    clarify = expected["expected_behavior"] == "clarify_target"
    clarification = any(e.get("type") == "TargetConfirmationRequired" for e in events)
    no_research = not targets and not documents
    # 대상 선택 이벤트만으로 자연어 설명의 정확성까지 판정하지 않는다.
    return {
        "request_success": int(ok),
        "route_correct": int(ok and response.get("metadata", {}).get("route") == expected["expected_route"]),
        "target_exact": int(ok and required_targets <= targets <= allowed_targets),
        "clarification_safe": int(ok and clarification and no_research) if clarify else None,
        "retrieval_required": len(required), "retrieval_found": found,
        "retrieval_recall": found / len(required) if required else None,
        "retrieval_precision": len(documents & allowed) / len(documents) if documents and required and ok else None,
        "unnecessary_retrieval": int(bool(documents)) if not required and ok else None,
        "answer_pass": _manual_score(expected, record.get("review")) if ok else 0,
        "numeric_grounding_pass": _grade((record.get("review") or {}).get("numeric_grounding_grade")) if ok else None,
        "system_document_link_accuracy": None,
    }


def distribution(values):
    values = sorted(v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0)
    return {"count": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "p95": values[math.ceil(len(values) * .95) - 1] if values else None}


def summarize(records):
    result = {"requests": len(records), "not_run": sum(r["status"] == "not_run" for r in records),
              "errors": sum(r["status"] == "error" for r in records)}
    for key in ("request_success", "route_correct", "target_exact", "clarification_safe",
                "answer_pass", "numeric_grounding_pass"):
        values = [r["scores"][key] for r in records if r["scores"][key] is not None]
        result[key] = {"passed": sum(values), "eligible": len(values),
                       "rate": mean(values) if values else None}
    required = sum(r["scores"]["retrieval_required"] for r in records)
    found = sum(r["scores"]["retrieval_found"] for r in records)
    recalls = [r["scores"]["retrieval_recall"] for r in records if r["scores"]["retrieval_recall"] is not None]
    result.update(retrieval_micro_recall=found / required if required else None,
                  retrieval_macro_recall=mean(recalls) if recalls else None,
                  retrieval_required_documents=required, retrieval_found_documents=found)
    for kind in ("http", "trace"):
        result[f"{kind}_latency_ms"] = distribution(
            [r.get(f"{kind}_elapsed_ms") for r in records if r["status"] == "ok"])
    known_usage = all(r.get("model_usage") is not None for r in records)
    calls = [call for r in records for call in (r.get("model_usage") or [])]
    result["model_calls"] = len(calls) if known_usage else None
    for provider in ("openai", "ollama"):
        usages = [c.get("usage") for c in calls if c["provider"] == provider]
        complete = known_usage and all(u is not None for u in usages)
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            result[f"{provider}_{field}"] = sum(u[field] for u in usages) if complete else None
    for key in ("retry_count", "validation_attempts", "replan_rounds", "cost_usd"):
        values = [(r.get("response") or {}).get("metadata", {}).get(key) for r in records]
        result["simulated_cost_usd" if key == "cost_usd" else key] = (
            sum(values) if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values) else None)
    result["actual_billed_usd"] = None
    estimates = [r.get("langsmith_estimated_cost_usd") for r in records]
    result["langsmith_estimated_cost_usd"] = sum(float(v) for v in estimates) if all(v is not None for v in estimates) else None
    scenarios = {}
    for r in records:
        scenarios.setdefault((r["case_id"], r["repetition"]), []).append(r["scores"]["answer_pass"])
    judged = [int(all(v)) for v in scenarios.values() if all(x is not None for x in v)]
    result["scenario_answer_pass"] = {"passed": sum(judged), "eligible": len(judged),
                                      "total": len(scenarios), "rate": mean(judged) if judged else None}
    return result
