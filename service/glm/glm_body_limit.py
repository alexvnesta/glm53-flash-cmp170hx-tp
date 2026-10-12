"""Bound the HTTP request body before any handler parses it (stdlib only, pure ASGI).

FastAPI reads and JSON-parses a whole request body, and the API had no limit, so one request could make the single
serving process buffer any amount of memory. This middleware reads at most the current limit, answers 413 in the API's
error shape for a larger body, and otherwise hands the application the same body as one message (then the server's
later messages, such as a disconnect, unchanged). Its state
(the limit and the count of refused bodies) is one dict that the API keeps in runtime["request_body"] for /health.
"""
import json
from collections import deque

DEFAULT_LIMIT = 32 * 1024 * 1024
MIN_LIMIT, MAX_LIMIT = 1024 * 1024, 1024 * 1024 * 1024


def body_limit_from_environment(environ):
    """GLM_MAX_REQUEST_BYTES (default 32 MiB). Any invalid value stops the service at start."""
    raw = environ.get("GLM_MAX_REQUEST_BYTES", "").strip()
    if not raw:
        return DEFAULT_LIMIT
    if not (raw.isascii() and raw.isdigit()) or not MIN_LIMIT <= int(raw) <= MAX_LIMIT:
        raise ValueError(f"GLM_MAX_REQUEST_BYTES must be an integer from {MIN_LIMIT} through {MAX_LIMIT}")
    return int(raw)


def body_limit_state(environ):
    return {"max_request_bytes": body_limit_from_environment(environ), "request_body_rejects": 0}


class RequestBodyLimit:
    def __init__(self, app, state):
        self.app = app
        self.state = state    # read on every request: image input raises the limit once, at start

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = self.state["max_request_bytes"]
        declared = [value for key, value in scope.get("headers", ()) if key.lower() == b"content-length"]
        if declared:
            if not declared[0].isdigit():
                return await self.reject(send, 400, "Invalid Content-Length header")
            if int(declared[0]) > limit:
                return await self.too_large(send, limit)
        # One bytearray, not one message per chunk: a body sent as a million tiny chunks costs no more than one chunk.
        body, ended, pending = bytearray(), False, deque()
        while not ended:
            message = await receive()
            if message["type"] != "http.request":
                pending.append(message)   # the client left before the body ended; the application sees that too
                break
            body += message.get("body", b"")
            if len(body) > limit:         # chunked or understated body
                return await self.too_large(send, limit)
            ended = not message.get("more_body", False)
        pending.appendleft({"type": "http.request", "body": bytes(body), "more_body": not ended})
        del body

        async def replay():
            return pending.popleft() if pending else await receive()
        await self.app(scope, replay, send)

    async def too_large(self, send, limit):
        self.state["request_body_rejects"] += 1
        await self.reject(send, 413, f"Request body exceeds {limit} bytes")

    @staticmethod
    async def reject(send, status, message):
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": status}}).encode()
        # The unread rest of the body makes the connection unusable for another request.
        await send({"type": "http.response.start", "status": status, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})
