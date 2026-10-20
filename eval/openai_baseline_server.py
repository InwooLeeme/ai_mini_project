"""보존본 서버에 로컬 근거/usage 기록만 추가하는 평가 계측. 진단 코드는 수정하지 않는다."""
import contextvars
import json
import os
from pathlib import Path
import sys
import time

from langsmith_usage import COST_POLICY, attach_openai_usage, with_virtual_cost_metadata


def make_recorded_post(original_post, append, get_thread):
    """HTTP 부수 효과를 한 곳에서 계측하며 원래 응답 객체를 그대로 돌려준다."""
    def recorded_post(url, *args, **kwargs):
        if url != "https://api.openai.com/v1/chat/completions":
            return original_post(url, *args, **kwargs)
        started = time.perf_counter()
        response = original_post(url, *args, **kwargs)
        entry = {
            "thread_id": get_thread(), "requested_model": kwargs.get("json", {}).get("model"),
            "http_status": response.status_code,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if response.is_success:
            data = response.json()
            entry.update(response_id=data.get("id"), resolved_model=data.get("model"),
                         usage=data.get("usage"))
            attach_openai_usage(data, entry["requested_model"])
            from langsmith import get_current_run_tree
            run = get_current_run_tree()
            entry["langsmith_llm_run_id"] = str(run.id) if run and run.run_type == "llm" else None
        append("model-usage.jsonl", entry)
        return response
    return recorded_post


def main():
    snapshot, output, port = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
    sys.path.insert(0, str(snapshot))
    os.chdir(snapshot)
    import httpx
    import server
    import uvicorn
    from agent.openai_llm import OpenAILLM
    from langsmith import tracing_context

    if Path(server.__file__).resolve() != snapshot / "server.py":
        raise RuntimeError("보존본 이외의 서버 import 거부")
    thread = contextvars.ContextVar("evaluation_thread", default=None)

    @server.app.get("/evaluation-identity")
    async def evaluation_identity():
        return {"pid": os.getpid(), "run_directory": output.name}

    def append(name, value):
        with (output / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")

    httpx.post = make_recorded_post(httpx.post, append, thread.get)
    OpenAILLM._chat = with_virtual_cost_metadata(OpenAILLM._chat)
    original_handle = server._handle_chat

    async def recorded_handle(message, thread_id):
        token = thread.set(thread_id)
        try:
            with tracing_context(metadata={**COST_POLICY, "thread_id": thread_id}):
                result, run_id = await original_handle(message, thread_id)
            append("evidence.jsonl", {"thread_id": thread_id, "query": message, "state": result})
            return result, run_id
        finally:
            thread.reset(token)

    server._handle_chat = recorded_handle
    try:
        uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        if os.environ.get("LANGSMITH_TRACING") == "true":
            from langsmith.run_trees import get_cached_client
            get_cached_client().flush(timeout=10)


if __name__ == "__main__":
    main()
