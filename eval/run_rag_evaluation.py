"""고정 검색 입력으로 Before/After를 측정한다. 정답은 검색기에 전달하지 않는다."""
import argparse
import asyncio
import hashlib
import inspect
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

# 평가 시 외부 추적 전송을 하지 않는다. 애플리케이션 설정 파일을 읽지 않는다.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from agent.graph import _classify_incident_type
from agent.mcp_client import MCPClient

ROOT = Path(__file__).resolve().parents[1]


def score(case, result):
    """실행 실패는 미관측으로 두며, 정답 없는 사례를 recall에서 제외한다."""
    if result["status"] != "ok":
        return None
    required = set(case["required_document_ids"])
    actual = {doc["document_id"] for doc in result["documents"]}
    if not required:
        return {"recall": None, "precision": None, "complete": None,
                "evidence_exact_recall": None, "correct_abstention": not actual}
    allowed = required | set(case.get("allowed_extra_document_ids", []))
    contexts = {doc["document_id"]: doc["text"] for doc in result["documents"]}
    anchors = [(e["document_id"], a["text"]) for e in case["evidence"]
               for a in e["required_evidence"]]
    return {
        "recall": len(actual & required) / len(required),
        "precision": len(actual & allowed) / len(actual) if actual else 0,
        "complete": required <= actual,
        # 원문 그대로 공급하는 현재 방식용 지표. 의미 보존에 대한 모델 판정이 아니다.
        "evidence_exact_recall": sum(text in contexts.get(doc, "") for doc, text in anchors)
        / len(anchors) if anchors else None,
        "correct_abstention": None,
    }


def summarize(rows):
    """평균과 분모, 오류를 함께 기록한다."""
    summary = {"requested": len(rows), "errors": sum(r["status"] != "ok" for r in rows)}
    for key in ("recall", "precision", "complete", "evidence_exact_recall", "correct_abstention"):
        values = [r["scores"][key] for r in rows if r["scores"] and r["scores"][key] is not None]
        summary[key] = {"mean": statistics.mean(values) if values else None, "n": len(values)}
    times = [r["elapsed_ms"] for r in rows if r["status"] == "ok"]
    summary["latency_ms"] = {"median": statistics.median(times) if times else None,
                            "min": min(times) if times else None,
                            "max": max(times) if times else None}
    return summary


async def run(args):
    """연결·색인 비용을 분리하고 각 요청의 실제 본문·오류·경과 시간을 보존한다."""
    dataset_path = ROOT / "eval/rag-dataset.json"
    dataset = json.loads(dataset_path.read_text())
    for doc in dataset["corpus"]:
        if hashlib.sha256((ROOT / doc["path"]).read_bytes()).hexdigest() != doc["sha256"]:
            raise ValueError(f"평가 코퍼스 변경: {doc['id']}")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "phase": args.phase, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "corpus": dataset["corpus"], "python": platform.python_version(),
        "platform": platform.platform(), "repetitions": args.repetitions,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_diff": subprocess.check_output(["git", "diff", "--stat"], text=True),
        "classifier_sha256": hashlib.sha256(inspect.getsource(_classify_incident_type).encode()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "agent").glob("*.py"))},
        "tracing": False, "external_api_calls": 0, "completed": False,
        "scope": "retrieval_component_synthetic_fixtures",
    }
    rows = []
    client = MCPClient()
    started = time.perf_counter()
    try:
        await client.start()
        manifest["mcp_start_ms"] = (time.perf_counter() - started) * 1000
        retriever = None
        if args.phase == "after":
            from agent.rag import build_retriever
            started = time.perf_counter()
            retriever = await build_retriever(client)
            manifest["index_build_ms"] = (time.perf_counter() - started) * 1000
            manifest["retriever"] = retriever.metadata
        with (out / "raw.jsonl").open("w") as stream:
            for repetition in range(1, args.repetitions + 1):
                for case in dataset["cases"]:
                    row = {"case_id": case["id"], "repetition": repetition,
                           "category": case["category"], "input": case["input"]}
                    started = time.perf_counter()
                    try:
                        if retriever:
                            result = await retriever.search(**case["input"])
                            row.update(asdict(result))
                        else:
                            collected = [{"source": "monitoring", "data": o}
                                         for o in case["input"]["observations"]]
                            doc_id = _classify_incident_type(collected)
                            server, template = await client.find_resource_template("get_incident_runbook")
                            uri = template.format(incident_type=doc_id)
                            text = await client.read_resource(uri, server=server)
                            row.update(documents=[{"document_id": doc_id, "text": text, "uri": uri}],
                                       candidates=[], reason="legacy_keyword_rule")
                        row["status"] = "ok"
                    except Exception as exc:
                        row.update(status="error", error_type=type(exc).__name__, documents=[])
                    row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    row["scores"] = score(case, row)
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(args.phase, repetition, case["id"], row["status"],
                          [d["document_id"] for d in row["documents"]], row["elapsed_ms"], flush=True)
        manifest["completed"] = True
    finally:
        await client.stop()
        (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        summary = summarize(rows)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    if any(row["status"] != "ok" for row in rows):
        raise SystemExit("오류가 포함되어 있습니다. raw.jsonl 확인 필요")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    options = parser.parse_args()
    if options.repetitions < 1:
        parser.error("repetitions는 1 이상이어야 합니다")
    asyncio.run(run(options))
