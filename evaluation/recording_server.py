"""독립 자식 프로세스의 계측. 원본 앱·모델 동작과 호출 결과를 그대로 유지한다."""
import contextvars
from functools import wraps
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse


class UsageRecorder:
    def __init__(self, output):
        self.output = Path(output)
        self.current = contextvars.ContextVar("evaluation_request", default=None)

    def append(self, name, value):
        with (self.output / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False) + "\n")

    def capture(self, url, payload, response, elapsed, error=None):
        context = self.current.get()
        parsed = urlparse(str(url))
        provider = ("openai" if str(url) == "https://api.openai.com/v1/chat/completions"
                    else "ollama" if parsed.path == "/api/chat" else None)
        if context is None or provider is None:
            return
        from eval.langsmith_usage import normalize_usage
        from langsmith import get_current_run_tree
        body = {}
        if response is not None and response.is_success:
            try:
                body = response.json()
            except (ValueError, TypeError):
                pass
        if not isinstance(body, dict):
            body = {}
        raw = (body.get("usage") if provider == "openai" else
               {"prompt_tokens": body.get("prompt_eval_count"), "completion_tokens": body.get("eval_count")})
        run = get_current_run_tree()
        row = {**context, "provider": provider, "requested_model": payload.get("model"),
               "resolved_model": body.get("model"), "usage": normalize_usage(raw),
               "http_status": response.status_code if response is not None else None,
               "elapsed_ms": elapsed, "llm_run_id": str(run.id) if run is not None else None}
        if error:
            row["error"] = type(error).__name__
        self.append("model-usage.jsonl", row)

    def wrap_sync(self, original):
        @wraps(original)
        def wrapped(url, *args, **kwargs):
            started, response, error = time.perf_counter(), None, None
            try:
                response = original(url, *args, **kwargs)
                return response
            except BaseException as exc:
                error = exc
                raise
            finally:
                self.capture(url, kwargs.get("json") or {}, response, (time.perf_counter() - started) * 1000, error)
        return wrapped

    def wrap_async(self, original):
        @wraps(original)
        async def wrapped(url, *args, **kwargs):
            started, response, error = time.perf_counter(), None, None
            try:
                response = await original(url, *args, **kwargs)
                return response
            except BaseException as exc:
                error = exc
                raise
            finally:
                self.capture(url, kwargs.get("json") or {}, response, (time.perf_counter() - started) * 1000, error)
        return wrapped


def instrument_handler(original, recorder, plan, manifest):
    lookup = {(p["thread_id"], p["message"]): p for p in plan}

    async def recorded(app, message, thread_id):
        item = lookup[thread_id, message]
        identity = {k: item[k] for k in ("case_id", "turn", "repetition")}
        token = recorder.current.set(identity)
        observation = {**identity, "thread_id": thread_id, "query": message, "state": None}

        def finished(run):
            observation.update(source_run_id=str(run.id), trace_started_at=run.start_time.isoformat(),
                               trace_elapsed_ms=(run.end_time - run.start_time).total_seconds() * 1000 if run.end_time else None)

        try:
            state, run_id = await original(app, message, thread_id, langsmith_extra={
                "metadata": {**identity, "evaluation_run_id": manifest["run_id"], "evaluation_profile": "current",
                             "dataset_sha256": manifest["dataset_sha256"]}, "on_end": finished})
            observation.update(state=state, source_run_id=run_id)
            return state, run_id
        except BaseException as exc:
            observation["error"] = type(exc).__name__
            raise
        finally:
            recorder.append("evidence.jsonl", observation)
            recorder.current.reset(token)
    return recorded


def main():
    snapshot, output, port = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
    # 계측 패키지는 이 실행기의 위치, 제품 모듈은 보존된 현재 소스에서 가져온다.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(snapshot))
    os.chdir(snapshot)
    import httpx
    import server
    import uvicorn
    from langsmith.run_trees import get_cached_client

    if Path(server.__file__).resolve() != snapshot / "server.py":
        raise RuntimeError("refusing non-snapshot server")
    recorder = UsageRecorder(output)
    httpx.post = recorder.wrap_sync(httpx.post)
    original_async_post = httpx.AsyncClient.post

    async def async_post(client, url, *args, **kwargs):
        return await recorder.wrap_async(original_async_post.__get__(client))(url, *args, **kwargs)

    httpx.AsyncClient.post = async_post
    manifest = json.loads((output / "manifest.json").read_text())
    plan = json.loads((output / "request-plan.json").read_text())
    server._handle_chat = instrument_handler(server._handle_chat, recorder, plan, manifest)

    @server.app.get("/evaluation-identity")
    async def identity():
        return {"pid": os.getpid(), "run_directory": output.name}

    try:
        uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        if os.environ.get("LANGSMITH_TRACING") == "true":
            get_cached_client().flush(timeout=10)


if __name__ == "__main__":
    main()
