"""고정 정상·오류 초안의 실제 검증 정확도·시간·토큰을 비교한다 (--run 필요)."""

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]


def comparison_cases(include_context_edges=False):
    from evaluation.alert_regression import load_fixture, regression_cases

    result = regression_cases(load_fixture())
    old = load_fixture()
    for case in result:
        case.update(query=old["query"], targets=old["targets"])
    recorded = json.loads(
        (ROOT / "evaluation/fixtures/validator-context.json").read_text()
    )
    result.append(
        {
            "id": "recorded_context",
            "expected": "passed",
            **{k: recorded[k] for k in ("query", "draft", "targets", "collected")},
        }
    )
    for name, draft, expected in [
        (
            "explicit_context",
            recorded["draft"].replace(
                "모니터링 알림에 따르면 메모리 사용량이",
                "cache-service의 모니터링 알림에 따르면 메모리 사용량이",
            ),
            "passed",
        ),
        (
            "wrong_system_attribution",
            "1) payments-service: 모니터링 알림에 따르면 메모리가 최근 6시간 동안 증가했습니다.\n2) cache-service 메모리 89%입니다. 공통 원인은 확인되지 않아 추가 로그가 필요합니다.",
            "temporal",
        ),
        (
            "wrong_period",
            "1) payments-service CPU 91%입니다.\n2) cache-service의 모니터링 알림에 따르면 메모리가 최근 12시간 동안 증가했습니다. 공통 원인은 미확인입니다.",
            "temporal",
        ),
        (
            "fabricated_logs",
            "1) payments-service CPU 91%이며 cache-service 메모리 89%입니다.\n2) 두 시스템의 로그를 직접 조회해 공통 원인이 최근 배포임을 확인했습니다.",
            "grounding",
        ),
        (
            "wrong_numeric",
            "1) payments-service CPU 50%이며 cache-service 메모리 89%입니다.\n2) 공통 원인은 미확인이므로 로그 확인이 필요합니다.",
            "numeric",
        ),
    ]:
        result.append(
            {
                "id": name,
                "expected": expected,
                "query": recorded["query"],
                "targets": recorded["targets"],
                "collected": recorded["collected"],
                "draft": draft,
            }
        )
    if include_context_edges:
        result.extend(json.loads((ROOT / "evaluation/fixtures/validator-context-edges.json").read_text()))
    return result


async def evaluate(validator, model, repeats, emit, *, include_context_edges=False):
    """동일 순서로 반복하며 규칙 검사와 실제 모델 호출을 구분해 기록한다."""
    for repeat in range(repeats):
        order = comparison_cases(True) if include_context_edges else comparison_cases()
        random.Random(20261021 + repeat).shuffle(order)
        for case in order:
            model.last_usage = None
            model.last_model = None
            started = time.perf_counter()
            result = await validator.validate(
                case["query"], case["draft"], case["collected"], case["targets"]
            )
            elapsed = (time.perf_counter() - started) * 1000
            expected = case["expected"]
            correct = (
                result.status == "passed"
                if expected == "passed"
                else result.status == "rejected"
                and any(
                    i.code == ("temporal" if expected == "attribution" else expected)
                    for i in result.issues
                )
            )
            if expected == "attribution":
                correct = correct and any(
                    c["criterion"] == "temporal"
                    and c["reason"].startswith("ALERT_ATTRIBUTION:")
                    for c in result.checks
                )
            emit(
                {
                    "id": case["id"],
                    "repeat": repeat + 1,
                    "expected": expected,
                    "correct": correct,
                    "elapsed_ms": elapsed,
                    "result": asdict(result),
                    "provider_usage": model.last_usage,
                    "model": model.last_model,
                }
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-context-edges", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.run:
        print(
            f"사전 검사: {len(comparison_cases(args.include_context_edges))}개 사례 × {args.repeats}회. 모델 호출 없음."
        )
        return 0
    if args.output is None or args.output.exists():
        parser.error("--output must name a new result file")

    from dotenv import dotenv_values
    from agent.openai_llm import OpenAILLM
    from agent.validation import AnswerValidator

    config = {**dotenv_values(ROOT / ".env"), **os.environ}
    if not config.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY missing")
    os.environ.update(
        OPENAI_API_KEY=config["OPENAI_API_KEY"],
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
    )

    class MeasuredReviewer(OpenAILLM):
        def _response(self, response):
            body = response.json()
            self.last_usage, self.last_model = body.get("usage"), body.get("model")
            return super()._response(response)

    model = MeasuredReviewer(config.get("OPENAI_VALIDATOR_MODEL") or "gpt-4o", 0.005)
    validator = AnswerValidator(model)
    names = [
        "agent/validation.py",
        "agent/openai_llm.py",
        "evaluation/validator_latency.py",
        "evaluation/alert_regression.py",
        "evaluation/fixtures/validator-context.json",
        "evaluation/fixtures/alert-feedback.json",
        "evaluation/fixtures/validator-context-edges.json",
    ]
    manifest = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names
    }
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        records.append(row)
        args.output.write_text(
            json.dumps(
                {
                    "source_sha256": manifest,
                    "requested_model": model.model,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(
            f"{row['id']} #{row['repeat']}: {'PASS' if row['correct'] else 'FAIL'} "
            f"({row['elapsed_ms']:.1f}ms)",
            flush=True,
        )

    asyncio.run(evaluate(validator, model, args.repeats, emit,
                         include_context_edges=args.include_context_edges))
    return 0 if all(r["correct"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
