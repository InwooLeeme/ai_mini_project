"""Prepared Before 측정기. 기본은 오프라인 검사, --run만 Ollama/v2를 호출한다."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from urllib import error, request
import uuid


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8100"
COLUMNS = [
    "sequence", "repetition", "case_id", "turn", "thread_id", "status",
    "http_status", "elapsed_ms", "route", "simulated_cost_usd", "retry_count",
    "validation_attempts", "rewrite_count", "replan_rounds",
    "last_research_elapsed_seconds", "monitoring_targets", "runbook_types",
    "langsmith_run_id", "actual_input_tokens", "actual_output_tokens", "actual_billed_usd",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight(root):
    """HTTP·모델·애플리케이션 import 없이 파일/보존본과 데이터셋을 검사한다."""
    baseline = json.loads((root / "baseline/prepared-manifest.json").read_text(encoding="utf-8"))
    hashes = baseline["files_sha256"]
    if not hashes or "eval/dataset.json" not in hashes:
        raise ValueError("baseline에 평가 데이터셋 해시가 필요합니다")
    expected = dict(hashes)
    expected["baseline/" + baseline["archive"]] = baseline["archive_sha256"]
    expected["baseline/" + baseline["tracked_patch"]] = baseline["tracked_patch_sha256"]
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"프로젝트 밖의 기준 경로: {name}")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Before 기준 파일이 달라졌습니다: {name}")
    conditions = baseline["conditions"]
    if (conditions["research_artificial_delay_seconds"] != 0
            or conditions["forced_first_failure_default"] is not False):
        raise ValueError("정리된 baseline의 지연/강제 실패 조건이 아닙니다")
    dataset = json.loads((root / "eval/dataset.json").read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if len(cases) != dataset["scenario_count"] or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("시나리오 수 또는 ID 불일치")
    if sum(len(c["turns"]) for c in cases) != dataset["request_count_per_repetition"]:
        raise ValueError("요청 수 불일치")
    for case in cases:
        if [t["turn"] for t in case["turns"]] != list(range(1, len(case["turns"]) + 1)):
            raise ValueError(f"턴 순서 불일치: {case['id']}")
        if not all(isinstance(t["message"], str) and t["message"].strip() for t in case["turns"]):
            raise ValueError(f"빈 질문: {case['id']}")
    return baseline, dataset


def measure(item, timeout):
    payload = {"message": item["message"], "thread_id": item["thread_id"]}
    req = request.Request(
        BASE_URL + "/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    body, http_status, failure, result = None, None, None, {}
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout) as response:
            http_status = response.status
            body = response.read().decode("utf-8")
        elapsed = (time.perf_counter() - started) * 1000
        result = json.loads(body)
        if (not isinstance(result, dict) or not isinstance(result.get("reply"), str)
                or not isinstance(result.get("metadata"), dict)
                or not isinstance(result.get("events"), list)
                or not all(isinstance(e, dict) for e in result["events"])):
            raise ValueError("응답 계약 불일치")
        for key in ("retry_count", "validation_attempts", "replan_rounds"):
            value = result["metadata"].get(key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"잘못된 횟수: {key}")
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        failure = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, error.HTTPError):
            http_status = exc.code
            body = exc.read().decode("utf-8", errors="replace")
        result = {}

    meta = result.get("metadata", {})
    events = result.get("events", [])
    attempts = meta.get("validation_attempts")

    def targets(source):
        return sorted({e["target"] for e in events
                       if e.get("type") == "TaskCompleted" and e.get("source") == source
                       and isinstance(e.get("target"), str)})

    row = {key: item[key] for key in ("sequence", "repetition", "case_id", "turn", "thread_id")}
    row.update(
        status="error" if failure else "response_received", http_status=http_status,
        elapsed_ms=round(elapsed, 3), route=meta.get("route"),
        simulated_cost_usd=meta.get("cost_usd"), retry_count=meta.get("retry_count"),
        validation_attempts=attempts,
        rewrite_count=max(0, attempts - 1) if attempts is not None else None,
        replan_rounds=meta.get("replan_rounds"),
        last_research_elapsed_seconds=meta.get("parallel_elapsed"),
        monitoring_targets=json.dumps(targets("monitoring")),
        runbook_types=json.dumps(targets("runbook")),
        langsmith_run_id=meta.get("langsmith_run_id"),
        actual_input_tokens=None, actual_output_tokens=None, actual_billed_usd=None,
    )
    return {**item, "measurement": row, "response": result, "raw_body": body, "error": failure}


def collect(root, baseline, dataset, repetitions, timeout):
    with request.urlopen(BASE_URL + "/health", timeout=10) as response:
        health = json.load(response)
    if health.get("llm_backend") != "ollama" or health.get("supervisor_version") != "v2":
        raise ValueError("Ollama/v2 서버만 허용합니다. 유료·fake 호출을 시작하지 않습니다.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    out = root / "eval/results" / ("before-prepared-" + run_id)
    out.mkdir(parents=True, exist_ok=False)
    plan = []
    for repetition in range(1, repetitions + 1):
        for case in dataset["cases"]:
            thread_id = f"eval-{run_id}-{repetition}-{case['id']}"
            for turn in case["turns"]:
                plan.append({
                    "sequence": len(plan) + 1, "repetition": repetition,
                    "case_id": case["id"], "turn": turn["turn"],
                    "thread_id": thread_id, "message": turn["message"],
                })
    manifest = {
        "phase": "before", "profile": baseline["profile"], "run_id": run_id,
        "dataset_id": dataset["dataset_id"], "dataset_version": dataset["version"],
        "dataset_sha256": sha256(root / "eval/dataset.json"),
        "source_manifest_sha256": sha256(root / "baseline/prepared-manifest.json"),
        "source_manifest": baseline, "conditions": baseline["conditions"], "health": health,
        "client_python": platform.python_version(), "client_platform": platform.platform(),
        "repetitions": repetitions, "planned_requests": len(plan),
        "concurrency": 1, "client_retry": 0, "request_timeout_seconds": timeout,
        "model_load_state": "not_recorded", "hardware": "not_recorded",
        "server_package_versions": "not_verified", "ollama_version_and_model_ids": "not_recorded",
        "tracing_enabled": "not_verified", "completed_plan": False,
        "recorded_requests": 0, "successful_responses": 0, "unexecuted_requests": len(plan),
        "measurement_status": "running", "stop_reason": None,
    }
    write_json(out / "request-plan.json", plan)
    write_json(out / "manifest.json", manifest)
    print("저장 위치:", out)
    try:
        with (out / "raw.jsonl").open("w", encoding="utf-8") as raw, (
            out / "measurements.csv"
        ).open("w", newline="", encoding="utf-8") as table:
            writer = csv.DictWriter(table, fieldnames=COLUMNS)
            writer.writeheader()
            for item in plan:
                result = measure(item, timeout)
                raw.write(json.dumps(result, ensure_ascii=False) + "\n")
                raw.flush()
                writer.writerow(result["measurement"])
                table.flush()
                manifest["recorded_requests"] += 1
                print(item["case_id"], item["turn"], result["measurement"]["status"],
                      result["measurement"]["elapsed_ms"], "ms")
                if result["error"]:
                    manifest["stop_reason"] = result["error"]
                    break
                manifest["successful_responses"] += 1
    except BaseException as exc:
        manifest["stop_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["unexecuted_requests"] = len(plan) - manifest["recorded_requests"]
        manifest["completed_plan"] = (
            manifest["successful_responses"] == len(plan) and manifest["stop_reason"] is None
        )
        manifest["measurement_status"] = "completed" if manifest["completed_plan"] else "stopped"
        write_json(out / "manifest.json", manifest)
    if not manifest["completed_plan"]:
        print("측정 중단:", manifest["stop_reason"], file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="오프라인 사전 검사만 수행 (기본)")
    mode.add_argument("--run", action="store_true", help="실제 로컬 Ollama/v2 측정 실행")
    parser.add_argument("--repetitions", type=int, default=os.environ.get("EVAL_REPETITIONS", "3"))
    parser.add_argument("--timeout", type=int, default=900, help="요청 제한 시간(초)")
    args = parser.parse_args(argv)
    if not 1 <= args.repetitions <= 10 or args.timeout < 1:
        parser.error("반복은 1~10, timeout은 양수여야 합니다")
    try:
        baseline, dataset = preflight(ROOT)
        print(f"사전 검사 통과: {len(baseline['files_sha256'])}개 파일, "
              f"{dataset['scenario_count']}개 시나리오 / "
              f"{dataset['request_count_per_repetition']}개 요청")
        if not args.run:
            print("HTTP·모델 호출 없음. 측정은 --run으로 명시적으로 실행하세요.")
            return 0
        return collect(ROOT, baseline, dataset, args.repetitions, args.timeout)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"중단: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("사용자가 중단했습니다. 서버의 진행 중 요청 상태를 확인하세요.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
