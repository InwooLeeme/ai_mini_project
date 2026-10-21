"""현재 코드를 독립 HTTP 서버에서 평가한다. --run 없이는 파일 사전 검사만 한다."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib import request
from uuid import uuid4

from .contracts import DEFAULT_BEFORE, ROOT, load_bundle, preflight, read_jsonl, request_plan, row_key, sha256, write_json
from .metrics import summarize


def prepare_output(output, before):
    output, before = Path(output).resolve(), Path(before).resolve()
    if output != before and output.is_relative_to(before):
        raise ValueError("output must not be inside Before")
    output.mkdir(parents=True, exist_ok=False)
    return output


def snapshot_sources(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    files = [root / name for name in ("server.py", "mcp_server.py", "mcp_server_docs.py")]
    files += sorted((root / "agent").glob("*.py"))
    files += sorted(p for p in (root / "data").rglob("*") if p.suffix in (".json", ".md"))
    hashes = {}
    for source in files:
        name = source.relative_to(root)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(name)] = sha256(target)
    return hashes


def validate_health(health, manifest):
    expected = {"llm_backend": "openai", "supervisor_version": "v2",
                "fast_llm": "openai:" + manifest["requested_fast_model"],
                "quality_llm": "openai:" + manifest["requested_quality_model"]}
    if "requested_validator_model" in manifest:
        expected["validator_llm"] = "openai:" + manifest["requested_validator_model"]
    if any(health.get(k) != v for k, v in expected.items()):
        raise ValueError("evaluation server backend/model/version mismatch")


def collect(plan, output, measure):
    rows, stopped = [], False
    with (Path(output) / "records.jsonl").open("x", encoding="utf-8") as f:
        for item in plan:
            row = ({**item, "status": "not_run", "response": None, "state": None, "model_usage": None,
                    "http_elapsed_ms": None, "trace_elapsed_ms": None, "error": "previous_request_failed"}
                   if stopped else measure(item))
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            stopped |= row["status"] != "ok"
            print(f"{item['case_id']}/{item['turn']}: {row['status']}", flush=True)
    return rows


def measure_http(item, url, timeout):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    row = {**item, "status": "error", "response": None, "state": None, "model_usage": None,
           "trace_elapsed_ms": None, "started_at": started_at}
    req = request.Request(url + "/chat", headers={"Content-Type": "application/json"},
                          data=json.dumps({"message": item["message"], "thread_id": item["thread_id"]}).encode())
    try:
        with request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read())
        if (not isinstance(result, dict) or not isinstance(result.get("reply"), str)
                or not isinstance(result.get("events"), list) or not isinstance(result.get("metadata"), dict)
                or not all(isinstance(e, dict) for e in result["events"])):
            raise ValueError("response contract mismatch")
        row.update(status="ok", response=result, source_run_id=result["metadata"].get("langsmith_run_id"))
    except Exception as exc:
        # 서버 오류 본문이나 인증 헤더가 결과·터미널에 노출되지 않는다.
        row["error"] = type(exc).__name__
    row["http_elapsed_ms"] = (time.perf_counter() - started) * 1000
    return row


def enrich(output, rows):
    observations = {row_key(r): r for r in read_jsonl(output / "evidence.jsonl")}
    calls = read_jsonl(output / "model-usage.jsonl")
    for row in rows:
        observed = observations.get(row_key(row))
        if observed:
            row.update(state=observed.get("state"), trace_elapsed_ms=observed.get("trace_elapsed_ms"),
                       source_run_id=observed.get("source_run_id"), trace_started_at=observed.get("trace_started_at"))
        if observed or any(row_key(c) == row_key(row) for c in calls):
            row["model_usage"] = [c for c in calls if row_key(c) == row_key(row)]
    temp = output / "records.tmp"
    with temp.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(output / "records.jsonl")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-dir", type=Path, default=DEFAULT_BEFORE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true", help="고정 9요청의 실제 OpenAI/Ollama 평가 및 기존 조건의 LangSmith 추적")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=0, help="0: 사용 가능한 별도 로컬 포트")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    baseline = preflight(ROOT, args.before_dir)
    before = baseline["manifest"]
    if before["backend"] != "openai" or before["repetitions"] != 1 or before["concurrency"] != 1 or before["client_retry"] != 0:
        raise ValueError("supported baseline: openai, repetition=1, concurrency=1, client_retry=0")
    if not args.run:
        print("사전 검사 통과: 기존 데이터·질문 해시 일치. 모델 호출·LangSmith 전송 없음.")
        return 0
    if args.timeout <= 0 or args.timeout > 600 or not (args.port == 0 or 1024 <= args.port <= 65535):
        raise ValueError("invalid timeout/port")
    from dotenv import dotenv_values
    config = {**dotenv_values(ROOT / ".env"), **os.environ}
    if not config.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY missing")
    # 명시 모델 설정이 기준과 다르면 조용히 덮어쓰지 않는다.
    for env_key, key in (("OPENAI_FAST_MODEL", "requested_fast_model"), ("OPENAI_QUALITY_MODEL", "requested_quality_model")):
        if config.get(env_key) and config[env_key] != before[key]:
            raise ValueError(f"model mismatch: {env_key}")
    sys.path.insert(0, str(ROOT / "eval"))
    from run_openai_baseline import tracing_settings
    trace_env, trace_public = tracing_settings(config, before["tracing_enabled"], (before.get("langsmith") or {}).get("project"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = prepare_output(args.output or ROOT / "evaluation/results" / ("after-current-" + run_id), args.before_dir)
    plan = request_plan(baseline["dataset"], run_id)
    manifest = {k: before[k] for k in ("backend", "requested_fast_model", "requested_quality_model", "temperature",
                "repetitions", "concurrency", "client_retry", "tracing_enabled", "warmup", "conditions", "dataset_sha256")}
    manifest["conditions"] = {k: v for k, v in before["conditions"].items() if k != "seven_feature_improvements"}
    manifest.update(run_id=run_id, phase="after", profile="current", before_run_id=before["run_id"],
                    requested_validator_model=config.get("OPENAI_VALIDATOR_MODEL") or "gpt-4o",
                    data_sha256=baseline["data_sha256"], measurement_status="starting", completed_plan=False,
                    langsmith=trace_public, planned_requests=len(plan), actual_billed_usd=None,
                    recorded_requests=0, source_sha256=snapshot_sources(ROOT, output / "source"),
                    runner_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "evaluation").glob("*.py")},
                    rag={"embedding_model": config.get("RAG_EMBEDDING_MODEL", "embeddinggemma:300m"),
                         "selection_model": config.get("RAG_SELECTION_MODEL", "qwen3:8b")},
                    feature_scope="current checkout; content validation enabled; specialist/log tools not implemented")
    write_json(output / "manifest.json", manifest)
    shutil.copyfile(args.before_dir / "dataset.json", output / "dataset.json")
    write_json(output / "request-plan.json", plan)
    write_json(output / "accuracy-review.json", {"policy": baseline["dataset"].get("evaluation_policy", {}), "reviews": [
        {"case_id": p["case_id"], "turn": p["turn"], "repetition": p["repetition"], "reviewer": "", "notes": "",
         "answer_behavior_grade": "not_run", "claim_grounding_grade": "not_run", "target_grade": "not_run",
         "numeric_grounding_grade": "not_run", "criteria": [{"criterion": c, "grade": "not_run"} for c in t["criteria"]],
         "forbidden_checks": [{"criterion": c, "grade": "not_run"} for c in t.get("forbidden", [])]}
        for p, t in zip(plan, [t for c in baseline["dataset"]["cases"] for t in c["turns"]])]})
    process, rows = None, []
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", args.port))
            port = listener.getsockname()[1]
        env = dict(os.environ)
        env.update(OPENAI_API_KEY=config["OPENAI_API_KEY"], LLM_BACKEND="openai", SUPERVISOR_VERSION="v2",
                   OPENAI_FAST_MODEL=before["requested_fast_model"], OPENAI_QUALITY_MODEL=before["requested_quality_model"],
                   OPENAI_VALIDATOR_MODEL=manifest["requested_validator_model"],
                   PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1")
        env.update(trace_env)
        for name in ("RAG_OLLAMA_URL", "RAG_EMBEDDING_MODEL", "RAG_SELECTION_MODEL"):
            if config.get(name):
                env[name] = config[name]
        with (output / "server.log").open("x") as log:
            started = time.perf_counter()
            process = subprocess.Popen([sys.executable, str(ROOT / "evaluation/recording_server.py"),
                                        str(output / "source"), str(output), str(port)],
                                       cwd=output / "source", env=env, stdout=log, stderr=subprocess.STDOUT)
            url = f"http://127.0.0.1:{port}"
            while time.perf_counter() - started < args.timeout:
                if process.poll() is not None:
                    raise RuntimeError("evaluation server startup failed; see server.log")
                try:
                    with request.urlopen(url + "/evaluation-identity", timeout=1) as r:
                        identity = json.load(r)
                    if identity != {"pid": process.pid, "run_directory": output.name}:
                        raise ValueError("unrelated server on evaluation port")
                    with request.urlopen(url + "/health", timeout=1) as r:
                        health = json.load(r)
                    validate_health(health, manifest)
                    manifest.update(health=health, startup_ms=(time.perf_counter() - started) * 1000)
                    break
                except OSError:
                    time.sleep(.25)
            else:
                raise TimeoutError("evaluation server startup timeout")
            print("결과 폴더:", output, flush=True)
            rows = collect(plan, output, lambda item: measure_http(item, url, args.timeout))
            manifest["completed_plan"] = all(r["status"] == "ok" for r in rows)
            manifest["measurement_status"] = "completed" if manifest["completed_plan"] else "failed"
    except Exception as exc:
        manifest.update(measurement_status="failed", error=type(exc).__name__)
        print(f"평가 실패: {type(exc).__name__}; {output}", file=sys.stderr)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                manifest["shutdown_forced"] = True
        manifest["recorded_requests"] = sum(r["status"] != "not_run" for r in rows)
        write_json(output / "manifest.json", manifest)
        if rows:
            enrich(output, rows)
        bundle = load_bundle(output)
        write_json(output / "summary.json", summarize(bundle["records"]))
    return 0 if manifest["completed_plan"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
