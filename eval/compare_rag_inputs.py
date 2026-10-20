"""동일 색인에서 Git 기준 선택기와 현재 선택기의 입력 최적화 효과를 비교한다."""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

from eval.run_rag_evaluation import ROOT, score, summarize
from agent.mcp_client import MCPClient
from agent.rag import build_retriever


def paired_inputs(cases, repetitions):
    """각 사례에서 두 방식의 선행·후행 횟수를 같게 한다."""
    if repetitions < 2 or repetitions % 2:
        raise ValueError("repetitions는 2 이상의 짝수여야 합니다")
    for repeat in range(1, repetitions + 1):
        for index, case in enumerate(cases):
            policies = ["baseline", "optimized"]
            if (repeat + index) % 2 == 0:
                policies.reverse()
            for policy in policies:
                yield repeat, case, policy


def token_summary(rows):
    """미측정은 0으로 추정하지 않고 합계와 측정 개수를 함께 기록한다."""
    result = {}
    for key in ("selector_input_tokens", "selector_output_tokens"):
        values = [r.get("usage", {}).get(key) for r in rows]
        values = [v for v in values if type(v) is int and v >= 0]
        result[key] = {"sum": sum(values) if values else None,
                       "mean": statistics.mean(values) if values else None, "n": len(values)}
    return result


def load_baseline(ref):
    """사용자가 지정한 로컬 Git 커밋의 선택기를 읽는다. 작업 트리는 변경하지 않는다."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "show", f"{commit}:agent/rag.py"], cwd=ROOT)
    return load_selector(source, "_rag_input_baseline"), commit, source


def load_selector(source, name):
    """평가자가 지정한 로컬 Python 소스에서 선택기를 구성한다."""
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / "agent/rag.py")
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.OllamaModels()


async def run(args):
    """모델·문서·색인은 공유하고 선택기만 교체한다. 정답은 채점에만 사용한다."""
    baseline, commit, source = load_baseline(args.baseline_ref)
    candidate_path = Path(args.candidate_file) if args.candidate_file else ROOT / "agent/rag.py"
    candidate_source = candidate_path.read_bytes()
    candidate = (load_selector(candidate_source, "_rag_input_candidate")
                 if args.candidate_file else None)
    cases, datasets = [], {}
    for name in ("rag-dataset.json", "rag-diagnostic-dataset.json"):
        path = ROOT / "eval" / name
        dataset = json.loads(path.read_text())
        datasets[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for doc in dataset["corpus"]:
            if hashlib.sha256((ROOT / doc["path"]).read_bytes()).hexdigest() != doc["sha256"]:
                raise ValueError(f"코퍼스 변경: {doc['id']}")
        cases.extend({**case, "dataset": name} for case in dataset["cases"])
    schedule = list(paired_inputs(cases, args.repetitions))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "baseline-rag.py").write_bytes(source)
    (out / "candidate-rag.py").write_bytes(candidate_source)
    (out / "evaluator.py").write_bytes(Path(__file__).read_bytes())
    manifest = {
        "baseline_commit": commit, "baseline_sha256": hashlib.sha256(source).hexdigest(),
        "optimized_sha256": hashlib.sha256(candidate_source).hexdigest(),
        "candidate_path": str(candidate_path),
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "datasets": datasets, "repetitions": args.repetitions, "completed": False,
        "scope": "retrieval_component_fixed_fixtures", "external_paid_calls": 0,
        "order": "balanced_per_case", "warmups": [],
        "limitations": "공개 회귀셋. 내부 캐시·장비 부하를 통제하지 않음. 첫 측정도 cold start는 아님.",
    }
    rows = []
    client = MCPClient()
    try:
        await client.start()
        retriever = await build_retriever(client)
        optimized = candidate if candidate is not None else retriever.selector
        assert baseline.selection_model == optimized.selection_model
        assert baseline.embedding_model == optimized.embedding_model
        assert baseline.url == optimized.url
        selectors = {"baseline": baseline, "optimized": optimized}
        manifest["retriever"] = retriever.metadata
        warmup = {"query": "검색 준비 확인. 알려진 장애와 요청한 절차는 없습니다.",
                  "observations": []}
        for policy, selector in selectors.items():
            retriever.selector = selector
            started = time.perf_counter()
            result = await retriever.search(**warmup)
            manifest["warmups"].append({
                "policy": policy, "input": warmup,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "usage": result.usage})
        with (out / "raw.jsonl").open("w") as stream:
            for repeat, case, policy in schedule:
                retriever.selector = selectors[policy]
                row = {"case_id": case["id"], "dataset": case["dataset"],
                       "repetition": repeat, "policy": policy, "input": case["input"]}
                started = time.perf_counter()
                try:
                    result = await retriever.search(**case["input"])
                    row.update(asdict(result), status="ok")
                except Exception as exc:
                    row.update(status="error", error_type=type(exc).__name__, documents=[])
                row["elapsed_ms"] = (time.perf_counter() - started) * 1000
                row["scores"] = score(case, row)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(repeat, case["id"], policy, row["status"],
                      round(row["elapsed_ms"]), flush=True)
        manifest["completed"] = True
    finally:
        try:
            await client.stop()
        finally:
            (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            summary = {}
            for group in ("all", *datasets):
                summary[group] = {}
                for policy in ("baseline", "optimized"):
                    selected = [r for r in rows if r["policy"] == policy
                                and (group == "all" or r["dataset"] == group)]
                    summary[group][policy] = {**summarize(selected),
                                               "tokens": token_summary(selected)}
            (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(r["status"] != "ok" for r in rows):
        raise SystemExit("실행 오류가 있습니다. raw.jsonl 확인 필요")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="develop")
    parser.add_argument("--candidate-file", help="보존한 로컬 agent/rag.py 후보 소스. 생략하면 현재 코드")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))
