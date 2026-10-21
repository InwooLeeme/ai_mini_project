"""평가 입력·기록 계약. 사전 검사는 파일만 읽고 서버·모델을 실행하지 않는다."""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from .metrics import score_record


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE = ROOT / "eval/results/before-prepared-openai-20261020T053746Z-7939ce28"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()] if Path(path).exists() else []


def row_key(row):
    return row["case_id"], row["turn"], row.get("repetition", 1)


def dataset_turns(dataset):
    cases = dataset["cases"]
    if len(cases) != dataset["scenario_count"] or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("dataset scenario count/IDs mismatch")
    turns = {}
    for case in cases:
        if [t["turn"] for t in case["turns"]] != list(range(1, len(case["turns"]) + 1)):
            raise ValueError("dataset turn order mismatch")
        for turn in case["turns"]:
            if not isinstance(turn["message"], str) or not turn["message"].strip():
                raise ValueError("dataset message missing")
            turns[case["id"], turn["turn"]] = turn
    if len(turns) != dataset["request_count_per_repetition"]:
        raise ValueError("dataset request count mismatch")
    return turns


def request_plan(dataset, run_id, repetitions=1):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be 1..10")
    turns = dataset_turns(dataset)
    return [{"sequence": (repetition - 1) * len(turns) + i + 1, "case_id": case_id,
             "turn": number, "repetition": repetition, "message": turn["message"],
             "thread_id": f"eval-{run_id}-{repetition}-{case_id}"}
            for repetition in range(1, repetitions + 1)
            for i, ((case_id, number), turn) in enumerate(turns.items())]


def data_hashes(manifest):
    hashes = manifest.get("data_sha256") or manifest.get("source_manifest", {}).get("files_sha256", {})
    return {k: v for k, v in hashes.items() if k.startswith("data/")}


def preflight(root, before_dir):
    root, before_dir = Path(root).resolve(), Path(before_dir).resolve()
    manifest = read_json(before_dir / "manifest.json")
    expected = manifest["dataset_sha256"]
    if sha256(before_dir / "dataset.json") != expected or sha256(root / "eval/dataset.json") != expected:
        raise ValueError("dataset hash mismatch")
    dataset = read_json(before_dir / "dataset.json")
    dataset_turns(dataset)
    hashes = data_hashes(manifest)
    if "data/monitoring.json" not in hashes:
        raise ValueError("missing data/monitoring.json hash")
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"data hash mismatch: {name}")
    # 기존 해시 목록이 있으면 평가에 사용하는 원자료를 대조한다.
    checksum_path = before_dir / "artifact-sha256.json"
    if checksum_path.exists():
        checksums = read_json(checksum_path)
        for name in ("manifest.json", "dataset.json", "raw.jsonl", "evidence.jsonl", "model-usage.jsonl", "accuracy-review.json"):
            if name in checksums and sha256(before_dir / name) != checksums[name]:
                raise ValueError(f"Before artifact hash mismatch: {name}")
    return {"manifest": manifest, "dataset": dataset, "data_sha256": hashes}


def _usage(call):
    from eval.langsmith_usage import normalize_usage
    return {"provider": "openai", "requested_model": call.get("requested_model"),
            "resolved_model": call.get("resolved_model"), "usage": normalize_usage(call.get("usage")),
            "http_status": call.get("http_status")}


def load_bundle(directory):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    dataset = read_json(directory / "dataset.json")
    if sha256(directory / "dataset.json") != manifest["dataset_sha256"]:
        raise ValueError("dataset hash mismatch")
    expected = dataset_turns(dataset)
    plan = request_plan(dataset, manifest["run_id"], manifest.get("repetitions", 1))
    normalized = directory / "records.jsonl"
    raw = read_jsonl(normalized if normalized.exists() else directory / "raw.jsonl")
    by_key = {}
    allowed_keys = {row_key(p) for p in plan}
    for r in raw:
        key = row_key(r)
        if key in by_key:
            raise ValueError(f"duplicate evaluation row: {key}")
        if key not in allowed_keys or r.get("message") != expected[key[:2]]["message"]:
            raise ValueError(f"foreign or changed evaluation row: {key}")
        by_key[key] = r
    evidence = read_jsonl(directory / "evidence.jsonl")
    calls = read_jsonl(directory / "model-usage.jsonl")
    spans = read_json(directory / "langsmith-spans.json") if (directory / "langsmith-spans.json").exists() else []
    span_ids = {(s["case_id"], s["turn"]): {n["id"] for n in s["nodes"]} for s in spans}
    roots = {(s["case_id"], s["turn"]): next((n for n in s["nodes"] if n.get("parent_run_id") is None), {}) for s in spans}
    remote = read_json(directory / "langsmith-verification.json") if (directory / "langsmith-verification.json").exists() else {}
    traces = {(r["case_id"], r["turn"]): r for r in remote.get("traces", [])}
    reviews_file = directory / "accuracy-review.json"
    reviews = read_json(reviews_file).get("reviews", []) if reviews_file.exists() else []
    by_review = {}
    for review in reviews:
        key = row_key(review)
        if key in by_review or key not in allowed_keys:
            raise ValueError(f"duplicate or foreign review: {key}")
        by_review[key] = review
    records = []
    for item in plan:
        key = row_key(item)
        old = by_key.get(key)
        if old is None:
            row = {**item, "status": "not_run", "response": None, "state": None, "model_usage": None,
                   "http_elapsed_ms": None, "trace_elapsed_ms": None}
        elif normalized.exists():
            row = dict(old)
        else:
            measurement = old["measurement"]
            matching = [e for e in evidence if e["thread_id"] == old["thread_id"] and e["query"] == old["message"]]
            ids = span_ids.get(key[:2])
            thread_turns = sum(r["thread_id"] == old["thread_id"] for r in raw)
            usage = [_usage(c) for c in calls if c["thread_id"] == old["thread_id"]
                     and (ids is None or c.get("langsmith_llm_run_id") in ids)]
            if not (directory / "model-usage.jsonl").exists() or (thread_turns > 1 and ids is None):
                usage = None
            trace = traces.get(key[:2], {})
            end = roots.get(key[:2], {}).get("end_time")
            trace_start = ((datetime.fromisoformat(end) - timedelta(milliseconds=trace["remote_latency_ms"])).isoformat()
                           if end and trace.get("remote_latency_ms") is not None else None)
            row = {**{k: old.get(k, v) for k, v in item.items()}, "status": "ok" if measurement["status"] == "response_received" else "error",
                   "response": old.get("response"), "state": matching[0]["state"] if len(matching) == 1 else None,
                   "http_elapsed_ms": measurement.get("elapsed_ms"), "trace_elapsed_ms": trace.get("remote_latency_ms"),
                   "model_usage": usage, "source_run_id": measurement.get("langsmith_run_id"), "trace_started_at": trace_start,
                   "source_run_url": trace.get("url"), "langsmith_estimated_cost_usd": trace.get("langsmith_total_cost")}
        row["review"] = by_review.get(key)
        row["scores"] = score_record(expected[key[:2]], row)
        records.append(row)
    return {"manifest": manifest, "dataset": dataset, "records": records}
