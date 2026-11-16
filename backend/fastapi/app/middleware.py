from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import error_response


BODY_LIMIT = 16 * 1024


class RequestBoundaryMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        started = False

        async def send_with_headers(message: Message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["Cache-Control"] = "no-store"
                headers["X-Content-Type-Options"] = "nosniff"
            await send(message)

        async def reject(status_code: int, code: str, message: str):
            response = error_response(status_code, code, message, request_id)
            await response(scope, receive, send_with_headers)

        async def invoke(inner_receive):
            try:
                await self.app(scope, inner_receive, send_with_headers)
            except Exception:
                if not started:
                    await reject(500, "INFERENCE_FAILED", "The request could not be processed.")

        if scope["path"].startswith("/api/") and scope["method"] in {"POST", "PUT", "PATCH"}:
            headers = Headers(scope=scope)
            lengths = headers.getlist("content-length")
            if lengths:
                value = lengths[0]
                if len(lengths) != 1 or not value or not value.isascii() or not value.isdecimal():
                    await reject(422, "INVALID_REQUEST", "Invalid request framing.")
                    return
                normalized = value.lstrip("0") or "0"
                if len(normalized) > 5 or int(normalized) > BODY_LIMIT:
                    await reject(413, "BODY_TOO_LARGE", "The request body exceeds 16 KiB.")
                    return

            media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                await reject(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json.")
                return

            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > BODY_LIMIT:
                    await reject(413, "BODY_TOO_LARGE", "The request body exceeds 16 KiB.")
                    return
                body.extend(chunk)
                if not message.get("more_body", False):
                    break

            if lengths and int(normalized) != len(body):
                await reject(422, "INVALID_REQUEST", "Invalid request framing.")
                return

            delivered = False

            async def replay_body():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await invoke(replay_body)
            return

        await invoke(receive)
