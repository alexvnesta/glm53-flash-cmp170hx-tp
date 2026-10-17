"""Local OpenAI API entry point. External bearer authentication stays in glm-auth-proxy."""
import asyncio
import contextlib
import json
import os
from pathlib import Path
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from controller import Controller

GLM = os.environ.get("SWITCHER_GLM_URL", "http://127.0.0.1:8014")
QWEN = os.environ.get("SWITCHER_QWEN_URL", "http://127.0.0.1:18420")
MAX_BODY = 32 * 1024 * 1024
QWEN_IDS = {"qwen", "qwen3.8-flash-next", "qwen3.8-flashnext", "qwen3.8-flash-next-abliterated"}
HOP = {"host", "authorization", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "content-length"}


class Services:
    def __init__(self, client):
        self.client = client
        self.qwen_key = Path(os.environ.get("SWITCHER_QWEN_KEY_FILE", "/home/alex/.qwen-api-key")).read_text().strip()
        if not self.qwen_key:
            raise RuntimeError("Qwen key file is empty")
        self.qwen_model = "qwen3.8-flash-next"
        self.records = {}

    def headers(self, target):
        return {"Authorization": "Bearer " + self.qwen_key} if target == "qwen" else {}

    def url(self, target):
        return QWEN if target == "qwen" else GLM

    async def command(self, argv, timeout=1800, allow_failure=False):
        p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            code = await asyncio.wait_for(p.wait(), timeout)
        except TimeoutError:
            p.kill()
            await p.wait()
            raise RuntimeError("service command timeout") from None
        if code and not allow_failure:
            raise RuntimeError("service command failed")

    async def free_gpus(self):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            p = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            try:
                out, _ = await asyncio.wait_for(p.communicate(), timeout=10)
            except TimeoutError:
                p.kill()
                await p.wait()
                raise RuntimeError("GPU ownership query timed out") from None
            try:
                used = [int(x.strip()) for x in out.decode().splitlines()]
            except ValueError:
                used = []
            if p.returncode == 0 and len(used) == 2 and max(used) < 2000:
                return
            await asyncio.sleep(3)
        raise RuntimeError("GPUs did not become free")

    async def active(self, target):
        cmd = ["systemctl", "--user", "is-active", "--quiet", "glm53-b2.service"] if target == "glm" else ["systemctl", "is-active", "--quiet", "qwen-flash-vllm.service"]
        p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        return await p.wait() == 0

    async def ready(self, target):
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            try:
                r = await self.client.get(self.url(target) + "/v1/models", headers=self.headers(target), timeout=5)
                if r.status_code == 200 and r.json().get("data"):
                    self.records[target] = r.json()["data"]
                    if target == "qwen":
                        self.qwen_model = r.json()["data"][0]["id"]
                    return
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            await asyncio.sleep(3)
        raise RuntimeError("backend readiness timeout")

    async def switch(self, target):
        if target == "qwen":
            # GLM's qualified unit is SIGKILLed on its 15-second stop timeout.
            # The GPU-free check, not systemctl's timeout exit, decides safety.
            await self.command(["systemctl", "--user", "stop", "glm53-b2.service"], allow_failure=True)
            await self.free_gpus()
            await self.command(["sudo", "-n", "systemctl", "start", "qwen-flash-vllm.service"])
        else:
            await self.command(["sudo", "-n", "systemctl", "stop", "qwen-flash-vllm.service"], allow_failure=True)
            await self.free_gpus()
            await self.command(["systemctl", "--user", "start", "glm53-b2.service"])
        await self.ready(target)


def make_app(services=None, controller=None, http_client=None):
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=10), trust_env=False)
        backend = services or Services(client)
        ctrl = controller
        if ctrl is None:
            glm, qwen = await backend.active("glm"), await backend.active("qwen")
            if glm and qwen:
                await client.aclose()
                raise RuntimeError("both GPU owners active")
            ctrl = Controller(backend, "qwen" if qwen else "glm" if glm else None,
                              ttl=float(os.environ.get("SWITCHER_QWEN_IDLE_SECONDS", "900")))
            if ctrl.active is None:
                await ctrl._change("glm")
            else:
                await backend.ready(ctrl.active)
        state.update(client=client, backend=backend, controller=ctrl)

        async def idle_return():
            while True:
                await asyncio.sleep(10)
                try:
                    await ctrl.expire()
                except Exception:
                    # The controller attempts GLM recovery; /health exposes the failure.
                    pass
        timer = asyncio.create_task(idle_return())
        try:
            yield
        finally:
            timer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await timer
            await client.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health():
        c, b, client = state["controller"], state["backend"], state["client"]
        def unavailable(status="unavailable"):
            # Preserve admission visibility even while the backend is loading.
            return JSONResponse({"status": status, "busy": True,
                "switcher": {"active_model": c.active, "inflight": c.inflight,
                             "waiting": c.waiters, "qwen_idle_seconds": c.ttl}}, status_code=503)
        if c.switching or c.active is None:
            return unavailable("switching" if c.switching else "unavailable")
        try:
            r = await client.get(b.url(c.active) + "/health", headers=b.headers(c.active), timeout=5)
            if r.status_code != 200:
                return unavailable()
            try:
                h = r.json()
            except ValueError:
                h = {}
            if not isinstance(h, dict):
                h = {}
            if h.get("status", "ok") != "ok" or (h.get("prompt_cache") or {}).get("terminal"):
                return unavailable()
            h.update(status=h.get("status", "ok"), busy=bool(h.get("busy") or c.inflight or c.waiters),
                     switcher={"active_model": c.active, "inflight": c.inflight, "waiting": c.waiters, "qwen_idle_seconds": c.ttl})
            return JSONResponse(h)
        except httpx.HTTPError:
            return unavailable()

    @app.get("/v1/models")
    async def models():
        ids = ["GLM-5.3-Flash", "GLM-5.3-Flash-low", "GLM-5.3-Flash-high", "GLM-5.3-Flash-max", "qwen3.8-flash-next"]
        records = getattr(state["backend"], "records", {})
        glm = {r["id"]: r for r in records.get("glm", [])}
        qwen = next(iter(records.get("qwen", [])), {})
        data = []
        for name in ids:
            original = qwen if name.startswith("qwen") else glm.get(name, {})
            record = {k: v for k, v in original.items() if k not in ("root", "parent")}
            record.update(id=name, object="model", owned_by="local")
            record.setdefault("created", 0)
            data.append(record)
        return {"object": "list", "data": data}

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def forward(path, request: Request):
        allowed = {"v1/chat/completions", "v1/completions", "tokenize"} if request.method == "POST" else {"metrics"}
        if path not in allowed:
            return JSONResponse({"error": "not found"}, status_code=404)
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > MAX_BODY:
                return JSONResponse({"error": "request body too large"}, status_code=413)
        c, b, client = state["controller"], state["backend"], state["client"]
        target = None  # GET /metrics observes the current owner; it never selects a model.
        payload = None
        if request.method == "POST":
            try:
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError()
                model = payload.get("model", "GLM-5.3-Flash")
                if not isinstance(model, str):
                    raise ValueError()
                if model.lower().startswith("qwen") and model.lower() not in QWEN_IDS:
                    return JSONResponse({"error": "Qwen model is not available"}, status_code=404)
                target = "qwen" if model.lower() in QWEN_IDS else "glm"
                payload.setdefault("model", "GLM-5.3-Flash")
            except ValueError:
                return JSONResponse({"error": "invalid JSON request"}, status_code=400)
        async def prepare():
            lease = c.lease(target, activity=request.method == "POST")
            entered = False
            try:
                await lease.__aenter__()
                entered = True
                owner = target or c.active
                if request.method == "POST":
                    # An operator or an automatic unit restart may be loading the
                    # current owner without changing the selected model. Hold its
                    # lease and keep SSE comments flowing until it is ready.
                    # Never retry a forwarded generation: that could duplicate it.
                    await b.ready(owner)
                outgoing = bytes(body) or None
                if payload is not None:
                    obj = dict(payload)
                    if owner == "qwen":
                        obj["model"] = b.qwen_model
                    outgoing = json.dumps(obj).encode()
                headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
                headers.pop("content-encoding", None)  # outgoing JSON was re-encoded
                headers["Accept-Encoding"] = "identity"  # SSE comments must remain plain bytes
                headers.update(b.headers(owner))
                upstream = await client.send(client.build_request(request.method, b.url(owner) + "/" + path,
                                                                   content=outgoing, headers=headers), stream=True)
                return lease, upstream
            except BaseException:
                if entered:
                    await lease.__aexit__(None, None, None)
                raise

        async def cleanup(lease, upstream):
            try:
                await upstream.aclose()
            finally:
                await lease.__aexit__(None, None, None)

        if payload is not None and payload.get("stream") is True:
            async def event_stream():
                task = asyncio.create_task(prepare())
                lease = upstream = next_chunk = None
                try:
                    # Headers and comments arrive while systemd loads the model.
                    # The public auth proxy can therefore keep Cloudflare alive.
                    while not task.done():
                        done, _ = await asyncio.wait([task], timeout=15)
                        if not done:
                            yield b": waiting for model\n\n"
                    lease, upstream = task.result()
                    if upstream.headers.get("content-encoding", "identity") != "identity":
                        raise RuntimeError("compressed event stream refused")
                    if upstream.status_code >= 400:
                        raw = await upstream.aread()
                        try:
                            error = json.loads(raw)
                        except ValueError:
                            error = {"error": {"message": "backend request failed"}}
                        yield b"data: " + json.dumps(error).encode() + b"\n\n"
                        yield b"data: [DONE]\n\n"
                        return
                    iterator = upstream.aiter_raw().__aiter__()
                    tail = b"\n\n"
                    while True:
                        next_chunk = asyncio.create_task(iterator.__anext__())
                        while not next_chunk.done():
                            done, _ = await asyncio.wait([next_chunk], timeout=15)
                            if not done and (tail.endswith(b"\n\n") or tail.endswith(b"\r\n\r\n")):
                                yield b": keepalive\n\n"
                        try:
                            chunk = next_chunk.result()
                        except StopAsyncIteration:
                            break
                        tail = (tail + chunk)[-4:]
                        yield chunk
                except Exception:
                    yield b'data: {"error":{"message":"backend unavailable or switch failed"}}\n\ndata: [DONE]\n\n'
                finally:
                    async def release():
                        if next_chunk and not next_chunk.done():
                            next_chunk.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await next_chunk
                        if lease is not None:
                            await cleanup(lease, upstream)
                        else:
                            if not task.done():
                                task.cancel()
                            try:
                                pair = await task
                            except BaseException:
                                return
                            await cleanup(*pair)
                    await asyncio.shield(asyncio.create_task(release()))
            return StreamingResponse(event_stream(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        try:
            lease, upstream = await prepare()
        except asyncio.CancelledError:
            raise
        except Exception:
            return JSONResponse({"error": "backend unavailable or switch failed"}, status_code=503)

        async def stream():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await asyncio.shield(asyncio.create_task(cleanup(lease, upstream)))
        headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP}
        return StreamingResponse(stream(), status_code=upstream.status_code, headers=headers)

    return app


app = make_app()
