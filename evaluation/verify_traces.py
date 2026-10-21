"""After 원본 trace의 원격 입력·출력·usage를 조회해 로컬 기록과 대조한다."""
import argparse
import copy
import json
import os
from pathlib import Path

from .contracts import ROOT, load_bundle, write_json


def verify_records(records, client):
    verified, checks = [], []
    for row in records:
        if not row.get("source_run_id"):
            raise ValueError("missing source trace; request is not remotely verified")
        run = client.read_run(row["source_run_id"])
        if run.inputs != {"message": row["message"], "thread_id": row["thread_id"]}:
            raise ValueError("remote input mismatch")
        if row["status"] == "ok":
            output = (run.outputs or {}).get("output")
            if (run.error or not isinstance(output, (list, tuple)) or not output
                    or not isinstance(output[0], dict) or output[0].get("final") != row["response"]["reply"]):
                raise ValueError("remote output mismatch")
        if run.end_time is None:
            raise ValueError("source trace has not finished")
        usages = row.get("model_usage")
        tokens = (sum(c["usage"]["total_tokens"] for c in usages)
                  if usages is not None and all(c.get("usage") is not None for c in usages) else None)
        if tokens is not None and run.total_tokens != tokens:
            raise ValueError("remote tokens differ from provider records (or aggregation not ready)")
        enriched = copy.deepcopy(row)
        enriched.update(source_run_url=client.get_run_url(run=run), trace_started_at=run.start_time.isoformat(),
                        trace_elapsed_ms=(run.end_time - run.start_time).total_seconds() * 1000,
                        langsmith_estimated_cost_usd=str(run.total_cost) if run.total_cost is not None else None)
        verified.append(enriched)
        checks.append({"case_id": row["case_id"], "turn": row["turn"], "source_run_id": row["source_run_id"],
                       "inputs_match": True, "outputs_match": True if row["status"] == "ok" else None,
                       "provider_tokens": tokens, "remote_tokens": run.total_tokens,
                       "token_match": True if tokens is not None else None})
    return verified, {"verified_requests": len(verified), "checks": checks, "actual_billed_usd": None,
                      "cost_basis": "provider_usage_x_LangSmith_pricing_estimate"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", required=True, type=Path)
    args = parser.parse_args(argv)
    bundle = load_bundle(args.after)
    if bundle["manifest"].get("phase") != "after":
        raise ValueError("only new After artifacts may be enriched; Before is immutable")
    from dotenv import dotenv_values
    from langsmith import Client
    config = {**dotenv_values(ROOT / ".env"), **os.environ}
    key = config.get("LANGSMITH_API_KEY") or config.get("LANGCHAIN_API_KEY")
    if not key:
        raise ValueError("LangSmith API key missing")
    client = Client(api_key=key, api_url=config.get("LANGSMITH_ENDPOINT") or config.get("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com",
                    workspace_id=config.get("LANGSMITH_WORKSPACE_ID"), auto_batch_tracing=False, timeout_ms=15000)
    try:
        records, report = verify_records(bundle["records"], client)
    finally:
        client.close()
    target = args.after / "records.verified.tmp"
    with target.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    target.replace(args.after / "records.jsonl")
    write_json(args.after / "remote-verification.json", report)
    print(f"원격 조회 검증: {len(records)}개 요청. 원본 LangSmith trace 수정 없음.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
