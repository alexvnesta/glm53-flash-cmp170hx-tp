import asyncio
import json
import unittest
import httpx
from controller import Controller
from switcher import make_app


class SlowStream(httpx.AsyncByteStream):
    def __init__(self, start, finish, closed):
        self.start, self.finish, self.closed = start, finish, closed
    async def __aiter__(self):
        self.start.set()
        yield b'data: {"token": "first"}\n\n'
        await self.finish.wait()
        yield b'data: [DONE]\n\n'
    async def aclose(self): self.closed.set()


class Backend:
    qwen_model = "actual-qwen-id"
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.finish.set()
        self.ready_started = asyncio.Event()
        self.ready_finish = asyncio.Event()
        self.ready_finish.set()
    async def ready(self, target):
        self.ready_started.set()
        await self.ready_finish.wait()
    def url(self, target): return "http://mock/" + target
    def headers(self, target): return {"Authorization": "Bearer private-test-key"} if target == "qwen" else {}
    async def switch(self, target):
        self.started.set()
        await self.finish.wait()
        self.calls.append(target)


class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = Backend(); self.ctrl = Controller(self.backend)
        self.start, self.finish, self.closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.requests = []; self.slow = False; self.fail = False; self.terminal = False; self.unavailable = False
        async def upstream(req):
            self.requests.append(req)
            if req.url.path.endswith('/health'):
                return httpx.Response(503 if self.unavailable else 200, json={"status": "ok", "busy": False, "prompt_cache": {"terminal": self.terminal}})
            if self.fail: raise httpx.ConnectError('unavailable')
            if self.slow and req.url.path.startswith('/glm'):
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowStream(self.start, self.finish, self.closed))
            return httpx.Response(200, stream=httpx.ByteStream(b'{"ok":true}'))
        self.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.app = make_app(self.backend, self.ctrl, self.upstream)
        self.life = self.app.router.lifespan_context(self.app); await self.life.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://entry')
    async def asyncTearDown(self):
        self.finish.set()
        await self.client.aclose(); await self.life.__aexit__(None, None, None)

    async def test_default_and_auth_header_boundary(self):
        r = await self.client.post('/v1/chat/completions', json={"messages": []}, headers={"Authorization": "Bearer caller-key"})
        self.assertEqual(r.status_code, 200)
        req = self.requests[-1]
        self.assertEqual(req.url.path, '/glm/v1/chat/completions')
        self.assertNotIn('Authorization', req.headers)
        self.assertEqual(json.loads(req.content)['model'], 'GLM-5.3-Flash')
        self.assertEqual(self.ctrl.inflight, 0)

    async def test_qwen_is_explicit_and_uses_backend_key_and_id(self):
        r = await self.client.post('/v1/completions', json={"model": "qwen3.8-flash-next", "prompt": "x"})
        self.assertEqual(r.status_code, 200); self.assertEqual(self.backend.calls, ['qwen'])
        req = self.requests[-1]
        self.assertEqual(req.headers['Authorization'], 'Bearer private-test-key')
        self.assertEqual(json.loads(req.content)['model'], 'actual-qwen-id')

    async def test_open_stream_blocks_switch_until_last_byte(self):
        self.slow = True
        glm = asyncio.create_task(self.client.post('/v1/completions', json={"prompt": "x"}))
        await self.start.wait()
        qwen = asyncio.create_task(self.client.post('/v1/completions', json={"model": "qwen", "prompt": "x"}))
        await asyncio.sleep(.03)
        self.assertEqual(self.backend.calls, []); self.assertEqual(self.ctrl.inflight, 1)
        h = await self.client.get('/health'); self.assertEqual(h.status_code, 503)
        self.finish.set(); a, b = await asyncio.gather(glm, qwen)
        self.assertIn(b'[DONE]', a.content); self.assertEqual(b.status_code, 200)
        self.assertEqual(self.backend.calls, ['qwen']); self.assertEqual(self.ctrl.inflight, 0)

    async def test_disconnect_closes_upstream_and_releases_lease(self):
        self.slow = True
        task = asyncio.create_task(self.client.post('/v1/completions', json={"prompt": "x"}))
        await self.start.wait(); task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        await asyncio.wait_for(self.closed.wait(), 1)
        await asyncio.sleep(.01)
        self.assertEqual(self.ctrl.inflight, 0)

    async def test_connect_error_releases_lease(self):
        self.fail = True
        r = await self.client.post('/v1/completions', json={"prompt": "x"})
        self.assertEqual(r.status_code, 503); self.assertEqual(self.ctrl.inflight, 0)

    async def test_existing_owner_reload_holds_request_and_exposes_lease(self):
        self.backend.ready_finish.clear(); self.unavailable = True
        task = asyncio.create_task(self.client.post('/v1/completions', json={"prompt": "x"}))
        await asyncio.wait_for(self.backend.ready_started.wait(), 1)
        h = await self.client.get('/health')
        self.assertEqual(h.status_code, 503)
        self.assertEqual(h.json()['switcher']['inflight'], 1)
        self.assertEqual(h.json()['switcher']['active_model'], 'glm')
        self.assertFalse(any(r.method == 'POST' for r in self.requests))
        self.assertFalse(task.done()); self.assertEqual(self.backend.calls, [])
        self.unavailable = False; self.backend.ready_finish.set()
        self.assertEqual((await task).status_code, 200)
        self.assertEqual(sum(r.method == 'POST' for r in self.requests), 1)
        self.assertEqual(self.ctrl.inflight, 0)

    async def test_disconnect_while_existing_owner_reloads_releases_lease(self):
        self.backend.ready_finish.clear()
        task = asyncio.create_task(self.client.post('/v1/completions', json={"prompt": "x"}))
        await asyncio.wait_for(self.backend.ready_started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(self.ctrl.inflight, 0)
        self.assertFalse(any(r.method == 'POST' for r in self.requests))

    async def test_stream_headers_arrive_while_existing_owner_reloads(self):
        self.backend.ready_finish.clear()
        headers = asyncio.Event(); sent = []; first = True
        body = json.dumps({"prompt": "x", "stream": True}).encode()
        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()
        async def send(message):
            sent.append(message)
            if message['type'] == 'http.response.start': headers.set()
        scope = {"type": "http", "method": "POST", "path": "/v1/completions", "raw_path": b"/v1/completions",
                 "query_string": b"", "headers": [(b"content-type", b"application/json")], "http_version": "1.1",
                 "scheme": "http", "server": ("test", 80), "client": ("test", 1)}
        task = asyncio.create_task(self.app(scope, receive, send))
        await asyncio.wait_for(headers.wait(), 1)
        await asyncio.wait_for(self.backend.ready_started.wait(), 1)
        self.assertEqual(sent[0]['status'], 200)
        self.assertEqual(self.ctrl.inflight, 1)
        self.assertFalse(any(r.method == 'POST' for r in self.requests))
        self.backend.ready_finish.set(); await asyncio.wait_for(task, 1)
        self.assertEqual(self.ctrl.inflight, 0)
        self.assertFalse(any(b'backend unavailable' in m.get('body', b'') for m in sent))

    async def test_unavailable_health_preserves_waiter_counts(self):
        self.unavailable = True; self.ctrl.waiters = 2
        r = await self.client.get('/health')
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()['switcher']['waiting'], 2)
        self.assertTrue(r.json()['busy'])
        self.ctrl.waiters = 0

    async def test_stream_headers_arrive_before_model_load_finishes(self):
        self.backend.finish.clear()
        headers = asyncio.Event(); sent = []
        body = json.dumps({"model": "qwen", "prompt": "x", "stream": True}).encode()
        first = True
        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()
        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.start": headers.set()
        scope = {"type": "http", "method": "POST", "path": "/v1/completions", "raw_path": b"/v1/completions",
                 "query_string": b"", "headers": [(b"content-type", b"application/json")], "http_version": "1.1",
                 "scheme": "http", "server": ("test", 80), "client": ("test", 1)}
        task = asyncio.create_task(self.app(scope, receive, send))
        await asyncio.wait_for(headers.wait(), 1)
        await asyncio.wait_for(self.backend.started.wait(), 1)
        self.assertEqual(sent[0]["status"], 200)
        self.assertFalse(task.done()); self.assertEqual(self.backend.calls, [])
        self.backend.finish.set(); await asyncio.wait_for(task, 1)
        self.assertEqual(self.ctrl.inflight, 0)

    async def test_sse_disconnect_releases_lease(self):
        self.slow = True
        task = asyncio.create_task(self.client.post('/v1/completions', json={"prompt": "x", "stream": True}))
        await self.start.wait(); task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        await asyncio.wait_for(self.closed.wait(), 1)
        await asyncio.sleep(.01)
        self.assertEqual(self.ctrl.inflight, 0)

    async def test_terminal_health_is_unavailable_and_models_do_not_switch(self):
        self.terminal = True
        self.assertEqual((await self.client.get('/health')).status_code, 503)
        r = await self.client.get('/v1/models')
        self.assertEqual(len(r.json()['data']), 5); self.assertEqual(self.backend.calls, [])

    async def test_rejects_invalid_and_unknown_routes(self):
        for data in ('[]', '{"model":1}', '{'):
            self.assertEqual((await self.client.post('/v1/completions', content=data)).status_code, 400)
        self.assertEqual((await self.client.post('/admin/drain', json={})).status_code, 404)
        self.assertEqual((await self.client.post('/v1/completions', json={"model": "qwen2.5-7b", "prompt": "x"})).status_code, 404)
        self.assertEqual(self.backend.calls, [])


if __name__ == '__main__': unittest.main()
