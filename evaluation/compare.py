"""python -m evaluation.compare --before DIR --after DIR --output NEW_DIR"""
import argparse
from pathlib import Path

from .contracts import data_hashes, load_bundle, row_key, write_json
from .metrics import summarize


def compare_bundles(before, after):
    a, b = before["manifest"], after["manifest"]
    for key in ("dataset_sha256", "backend", "requested_fast_model", "requested_quality_model",
                "temperature", "repetitions", "concurrency", "client_retry", "tracing_enabled", "warmup"):
        if a.get(key) != b.get(key):
            raise ValueError(f"comparison condition mismatch: {key}")
    if data_hashes(a) != data_hashes(b):
        raise ValueError("comparison condition mismatch: data hashes")
    for key in ("monitoring_metrics", "last_checked", "research_artificial_delay_seconds", "forced_first_failure_default"):
        if a.get("conditions", {}).get(key) != b.get("conditions", {}).get(key):
            raise ValueError(f"comparison condition mismatch: {key}")
    def models(bundle):
        found = {}
        for r in bundle["records"]:
            for u in r.get("model_usage") or []:
                if u["provider"] == "openai" and u.get("resolved_model"):
                    found.setdefault(u.get("requested_model"), set()).add(u["resolved_model"])
        return found
    first, second = models(before), models(after)
    if any(first[m] != second[m] for m in first.keys() & second.keys()):
        raise ValueError("comparison condition mismatch: resolved models")
    left = {row_key(r): r for r in before["records"]}
    right = {row_key(r): r for r in after["records"]}
    if left.keys() != right.keys():
        raise ValueError("comparison request keys mismatch")
    return {"before_run_id": a["run_id"], "after_run_id": b["run_id"],
            "before": summarize(before["records"]), "after": summarize(after["records"]),
            "pairs": [{"case_id": k[0], "turn": k[1], "repetition": k[2],
                       "before_status": left[k]["status"], "after_status": right[k]["status"],
                       "before_scores": left[k]["scores"], "after_scores": right[k]["scores"]} for k in left],
            "limitations": ["작은 공개 회귀셋의 관측값이며 통계적 우위·운영 정확도를 뜻하지 않음",
                            "HTTP와 trace 지연은 별도 측정 경계. 성공 응답 지연에는 실패·미실행을 포함하지 않음",
                            "검색 점수는 최종 답변의 근거성을 보장하지 않음. 의미 정확도는 별도 수동 판정",
                            "After에 추가된 로컬 RAG 연산·토큰은 OpenAI 사용량과 별도 표시",
                            "실제 청구액은 미확인. 가상 단가 비용과 LangSmith 가격표 추정 비용을 구분",
                            "공급자 캐시·장비 부하·모델 적재 상태는 완전히 통제되지 않음"]}


def markdown_report(result):
    rows = ["# 자동화 평가 비교", "", f"Before: `{result['before_run_id']}` · After: `{result['after_run_id']}`", "",
            "| 지표 | Before | After |", "|---|---:|---:|"]
    def display(value):
        if value is None:
            return "미측정/해당 없음"
        if isinstance(value, dict) and "rate" in value:
            return f"{value['passed']}/{value['eligible']}" + (f" ({value['rate']:.1%})" if value['rate'] is not None else " 미판정")
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    for key in ("request_success", "target_exact", "clarification_safe", "retrieval_micro_recall", "retrieval_macro_recall",
                "answer_pass", "scenario_answer_pass", "numeric_grounding_pass", "model_calls", "openai_total_tokens",
                "ollama_total_tokens", "retry_count", "validation_attempts", "replan_rounds", "simulated_cost_usd",
                "langsmith_estimated_cost_usd", "actual_billed_usd"):
        rows.append(f"| {key} | {display(result['before'][key])} | {display(result['after'][key])} |")
    for kind in ("http", "trace"):
        for metric in ("count", "mean", "median", "p95"):
            rows.append(f"| {kind} latency {metric} (ms) | {display(result['before'][kind+'_latency_ms'][metric])} | {display(result['after'][kind+'_latency_ms'][metric])} |")
    rows += ["", "## 요청별 점수", "", "| 사례/턴 | Before 상태 | After 상태 | 대상 전→후 | 검색 전→후 | 답변 전→후 |", "|---|---|---|---|---|---|"]
    for pair in result["pairs"]:
        a, b = pair["before_scores"], pair["after_scores"]
        cells = [f"{display(a[k])} → {display(b[k])}" for k in ("target_exact", "retrieval_recall", "answer_pass")]
        rows.append(f"| {pair['case_id']}/{pair['turn']} | {pair['before_status']} | {pair['after_status']} | " + " | ".join(cells) + " |")
    rows += ["", "## 해석 범위", ""] + [f"- {v}" for v in result["limitations"]]
    return "\n".join(rows) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_bundles(load_bundle(args.before), load_bundle(args.after))
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.before, args.after)):
        raise ValueError("comparison output must be outside source runs")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "comparison.json", result)
    (args.output / "REPORT.md").write_text(markdown_report(result), encoding="utf-8")
    print(args.output / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
