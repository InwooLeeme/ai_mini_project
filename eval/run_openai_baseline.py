"""OpenAI prepared baseline 1회 측정. --run 없이는 오프라인 검사만 수행."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib import request
import uuid

import run_prepared_baseline as prepared
from langsmith_usage import COST_POLICY


ROOT = Path(__file__).resolve().parents[1]


def tracing_settings(config, enabled, project_override=None):
    value = "true" if enabled else "false"
    env = {"LANGCHAIN_TRACING_V2": value, "LANGSMITH_TRACING": value}
    if not enabled:
        return env, None
    key = config.get("LANGSMITH_API_KEY") or config.get("LANGCHAIN_API_KEY")
    if not key:
        raise ValueError("LangSmith API key 미설정")
    project = project_override or config.get("LANGSMITH_PROJECT") or config.get("LANGCHAIN_PROJECT") or "prepared-baseline-openai"
    endpoint = config.get("LANGSMITH_ENDPOINT") or config.get("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
    env.update(LANGSMITH_API_KEY=key, LANGCHAIN_API_KEY=key,
               LANGSMITH_PROJECT=project, LANGCHAIN_PROJECT=project,
               LANGSMITH_ENDPOINT=endpoint, LANGCHAIN_ENDPOINT=endpoint,
               LANGSMITH_TRACING_SAMPLING_RATE="1")
    if config.get("LANGSMITH_WORKSPACE_ID"):
        env["LANGSMITH_WORKSPACE_ID"] = config["LANGSMITH_WORKSPACE_ID"]
    return env, {"project": project, "endpoint": endpoint, "sampling_rate": 1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="명시적으로 유료 OpenAI 평가 9요청 실행")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--langsmith", action="store_true", help="질문·응답·도구 추적을 LangSmith로 전송")
    parser.add_argument("--langsmith-project", help="이번 실행에 사용할 LangSmith 프로젝트")
    args = parser.parse_args(argv)
    baseline, dataset = prepared.preflight(ROOT)
    if not args.run:
        print("사전 검사 통과. 모델 호출 없음. --run만 OpenAI 유료 호출을 수행합니다.")
        return 0
    if not 1024 <= args.port <= 65535:
        raise ValueError("잘못된 포트")
    from dotenv import dotenv_values
    config = {**dotenv_values(ROOT / ".env"), **os.environ}
    if not config.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 미설정")
    trace_env, trace_public = tracing_settings(config, args.langsmith, args.langsmith_project)
    if args.langsmith:
        from langsmith import Client
        client = Client(api_key=trace_env["LANGSMITH_API_KEY"],
                        api_url=trace_env["LANGSMITH_ENDPOINT"],
                        workspace_id=trace_env.get("LANGSMITH_WORKSPACE_ID"),
                        timeout_ms=10000)
        project = client.read_project(project_name=trace_public["project"])
        trace_public["project_id"] = str(project.id)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "eval/results" / ("before-prepared-openai-" + run_id)
    output.mkdir(parents=True, exist_ok=False)
    plan = []
    for case in dataset["cases"]:
        for turn in case["turns"]:
            plan.append({
                "sequence": len(plan) + 1, "repetition": 1, "case_id": case["id"],
                "turn": turn["turn"], "thread_id": f"eval-{run_id}-{case['id']}",
                "message": turn["message"],
            })
    manifest = {
        "phase": "before", "profile": baseline["profile"], "run_id": run_id,
        "source_manifest": baseline,
        "source_manifest_sha256": prepared.sha256(ROOT / "baseline/prepared-manifest.json"),
        "dataset_sha256": prepared.sha256(ROOT / "eval/dataset.json"),
        "dataset_id": dataset["dataset_id"], "dataset_version": dataset["version"],
        "conditions": baseline["conditions"], "backend": "openai",
        "requested_fast_model": config.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
        "requested_quality_model": config.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
        "temperature": 0, "repetitions": 1, "concurrency": 1, "client_retry": 0,
        "planned_requests": len(plan), "recorded_requests": 0, "successful_responses": 0,
        "unexecuted_requests": len(plan), "completed_plan": False,
        "measurement_status": "starting", "tracing_enabled": args.langsmith,
        "langsmith": trace_public,
        "warmup": "none; fresh application and MCP processes",
        "cache_policy": "no application cache; provider prompt caching not controlled",
        "python": platform.python_version(), "platform": platform.platform(),
        "hardware": {"architecture": platform.machine(), "cpu_model": "not_recorded"},
        "instrumentation": "local Tool/graph evidence and OpenAI usage recording; overhead included in HTTP latency",
        "actual_billed_usd": None,
        "cost_policy": COST_POLICY,
        "langsmith_usage_adapter_sha256": prepared.sha256(ROOT / "eval/langsmith_usage.py"),
        "runner_sha256": prepared.sha256(Path(__file__)),
        "recorder_sha256": prepared.sha256(ROOT / "eval/openai_baseline_server.py"),
        "stop_reason": None,
    }
    prepared.write_json(output / "manifest.json", manifest)
    prepared.write_json(output / "request-plan.json", plan)
    shutil.copyfile(ROOT / "eval/dataset.json", output / "dataset.json")
    shutil.copyfile(Path(__file__), output / "run_openai_baseline.py")
    shutil.copyfile(ROOT / "eval/openai_baseline_server.py", output / "openai_baseline_server.py")
    shutil.copyfile(ROOT / "eval/langsmith_usage.py", output / "langsmith_usage.py")
    print("결과 폴더:", output, flush=True)
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="openai-baseline-") as temp:
            snapshot = Path(temp)
            with tarfile.open(ROOT / "baseline" / baseline["archive"]) as archive:
                archive.extractall(snapshot, filter="data")
            for name in ("prepared-manifest.json", baseline["archive"], baseline["tracked_patch"]):
                shutil.copyfile(ROOT / "baseline" / name, snapshot / "baseline" / name)
            prepared.preflight(snapshot)
            env = dict(os.environ)
            env.update(
                OPENAI_API_KEY=config["OPENAI_API_KEY"], LLM_BACKEND="openai",
                SUPERVISOR_VERSION="v2", PYTHONDONTWRITEBYTECODE="1",
                OPENAI_FAST_MODEL=manifest["requested_fast_model"],
                OPENAI_QUALITY_MODEL=manifest["requested_quality_model"],
            )
            env.update(trace_env)
            with (output / "server.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, str(ROOT / "eval/openai_baseline_server.py"),
                     str(snapshot), str(output), str(args.port)],
                    cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                prepared.BASE_URL = f"http://127.0.0.1:{args.port}"
                for _ in range(60):
                    if process.poll() is not None:
                        raise RuntimeError("평가 서버 기동 실패: server.log 확인")
                    try:
                        with request.urlopen(prepared.BASE_URL + "/evaluation-identity", timeout=1) as response:
                            identity = json.load(response)
                        if identity != {"pid": process.pid, "run_directory": output.name}:
                            raise ValueError("다른 프로세스의 평가 서버입니다")
                        with request.urlopen(prepared.BASE_URL + "/health", timeout=1) as response:
                            health = json.load(response)
                        break
                    except OSError:
                        time.sleep(0.5)
                else:
                    raise TimeoutError("평가 서버 기동 시간 초과")
                if health["llm_backend"] != "openai" or health["supervisor_version"] != "v2":
                    raise ValueError("OpenAI/v2가 아닌 서버")
                manifest["health"] = health
                manifest["measurement_status"] = "running"
                prepared.write_json(output / "manifest.json", manifest)
                with (output / "raw.jsonl").open("w", encoding="utf-8") as raw, (
                    output / "measurements.csv"
                ).open("w", encoding="utf-8", newline="") as table:
                    writer = csv.DictWriter(table, fieldnames=prepared.COLUMNS)
                    writer.writeheader()
                    for item in plan:
                        result = prepared.measure(item, timeout=900)
                        raw.write(json.dumps(result, ensure_ascii=False) + "\n")
                        raw.flush()
                        writer.writerow(result["measurement"])
                        table.flush()
                        manifest["recorded_requests"] += 1
                        print(item["case_id"], item["turn"], result["measurement"]["status"],
                              result["measurement"]["elapsed_ms"], "ms", flush=True)
                        if result["error"]:
                            manifest["stop_reason"] = result["error"]
                            break
                        manifest["successful_responses"] += 1
                        prepared.write_json(output / "manifest.json", manifest)
                process.terminate()
                process.wait(timeout=15)
                process = None
    except BaseException as exc:
        manifest["stop_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        manifest["unexecuted_requests"] = len(plan) - manifest["recorded_requests"]
        manifest["completed_plan"] = manifest["successful_responses"] == len(plan)
        manifest["measurement_status"] = "completed" if manifest["completed_plan"] else "stopped"
        prepared.write_json(output / "manifest.json", manifest)
    return 0 if manifest["completed_plan"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
