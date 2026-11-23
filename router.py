#!/usr/bin/env python3
"""Loopback HTTP adapter for the local agent fairness experiment."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import re
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from aiohttp import ClientError, ClientSession, ClientTimeout, DummyCookieJar, TCPConnector, web
from yarl import URL

from scheduler import Event, Rejected, Request, Scheduler

MODEL = "openrelay/deepseek-v4.1-flash-abliterated"
SERVED_MODEL = "local-model"
CHAT = "/v1/chat/completions"
MESSAGES = "/v1/messages"
MODES = ("bypass", "fifo", "fair")
QUEUE_WAIT_HEADER = "X-Queue-Wait-Seconds"
MAX_SSE_EVENT_BYTES = 1 << 20
ID = re.compile(r"[a-zA-Z0-9_.:-]{1,128}\Z")


def local_endpoint(value: str) -> URL:
    """Accept only a numeric loopback HTTP origin on an explicit port."""
    parts = urlsplit(value)
    if (
        parts.scheme != "http"
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or not parts.hostname
        or "%" in parts.hostname
        or parts.port is None
        or parts.port == 0
    ):
        raise ValueError("use a numeric loopback HTTP origin with an explicit port")
    host = ipaddress.ip_address(parts.hostname)
    if not host.is_loopback:
        raise ValueError("upstream must be a numeric loopback address")
    return URL.build(scheme="http", host=str(host), port=parts.port)


@dataclass(frozen=True)
class RouterConfig:
    upstream: str = "http://127.0.0.1:8001"
    mode: str = "bypass"
    capacity: int = 128
    model: str = MODEL
    served_model: str = SERVED_MODEL
    max_body_bytes: int = 10 << 20
    max_pending: int = 256
    max_pending_bytes: int = 16 << 20
    max_pending_per_root: int = 32
    max_roots: int = 4096
    queue_timeout: float = 30
    root_ttl: float = 600
    request_timeout: float = 180


class Gate:
    """One event loop owns admission. All scheduler transitions are synchronous."""

    def __init__(self, config: RouterConfig, clock: Callable[[], float]):
        self.clock = clock
        self.queue_timeout = config.queue_timeout
        self.scheduler = Scheduler(
            config.mode,
            config.capacity,
            max_pending=config.max_pending,
            max_pending_bytes=config.max_pending_bytes,
            max_pending_per_root=config.max_pending_per_root,
            max_roots=config.max_roots,
            queue_timeout=config.queue_timeout,
            root_ttl=config.root_ttl,
        )
        self.waiters: dict[str, asyncio.Future[str]] = {}

    def _dispatch(self, events: Iterable[Event]) -> None:
        for event in events:
            future = self.waiters.get(event.id)
            if future is not None and not future.done():
                future.set_result(event.status)

    def advance(self) -> None:
        self._dispatch(self.scheduler.advance(self.clock()))

    async def admit(self, spec: Request) -> float:
        now = self.clock()
        self._dispatch(self.scheduler.advance(now))
        if self.scheduler.submit(spec, now) == "running":
            return 0.0

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.waiters[spec.id] = future
        deadline = now + self.queue_timeout
        timer = None

        def expire() -> None:
            nonlocal timer
            remaining = deadline - self.clock()
            if remaining > 0:
                # Event-loop timers may fire slightly ahead of their deadline.
                timer = loop.call_later(remaining, expire)
            else:
                self.advance()

        timer = loop.call_later(max(0, deadline - self.clock()), expire)
        try:
            if await future == "expired":
                raise Rejected("queue_timeout")
            return self.clock() - now
        finally:
            timer.cancel()
            del self.waiters[spec.id]

    def finish(self, id: str, completed_work: float | None) -> None:
        now = self.clock()
        if completed_work is None:
            self._dispatch(self.scheduler.cancel(id, now))
        else:
            self._dispatch(self.scheduler.finish(id, now, completed_work))


def decode_sse_payload(payload: bytes) -> dict | str:
    """The proxy and replay agree on malformed data, whitespace and DONE."""
    payload = payload.strip()
    if payload == b"[DONE]":
        return "[DONE]"
    event = json.loads(payload)
    if not isinstance(event, dict):
        raise ValueError("SSE data must be a JSON object or DONE")
    return event


def chat_choices(event: dict) -> list[dict]:
    choices = event.get("choices", [])
    if not isinstance(choices, list):
        raise ValueError("choices must be an array")
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("delta", {}), dict):
            raise ValueError("each choice and delta must be an object")
        reason = choice.get("finish_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("finish_reason must be a string or null")
    return choices


def unary_success(payload: bytes, path: str) -> bool:
    try:
        value = json.loads(payload)
    except (ValueError, RecursionError):
        return False
    if not isinstance(value, dict) or "error" in value:
        return False
    if path == CHAT:
        try:
            choices = chat_choices(value)
        except ValueError:
            return False
        return bool(choices) and all(isinstance(choice.get("message"), dict)
                                     and bool(choice.get("finish_reason")) for choice in choices)
    return value.get("type") == "message" and isinstance(value.get("content"), list)


class StreamCompletion:
    """Observe SSE terminal and error events without modifying the relayed bytes."""

    def __init__(self, path: str):
        self.chat = path == CHAT
        self.terminal = False
        self.failed = False
        self.finished = False
        self.partial = b""
        self.event = b""
        self.data = bytearray()
        self.size = 0

    def feed(self, chunk: bytes) -> None:
        *lines, self.partial = (self.partial + chunk).split(b"\n")
        for line in lines:
            line = line.removesuffix(b"\r")
            if not line:
                self._finish_event()
                continue
            self.size += len(line) + 1
            if self.size > MAX_SSE_EVENT_BYTES:
                raise web.HTTPBadGateway()
            field, _, value = line.partition(b":")
            value = value.removeprefix(b" ")
            if field == b"event":
                self.event = value
            elif field == b"data":
                self.data.extend(value + b"\n")
        if self.size + len(self.partial) > MAX_SSE_EVENT_BYTES:
            raise web.HTTPBadGateway()

    def _finish_event(self) -> None:
        payload = self.data.rstrip(b"\n")
        self.failed |= self.terminal and bool(payload or self.event)
        self.failed |= self.event == b"error"
        if payload:
            try:
                event = decode_sse_payload(payload)
            except (ValueError, RecursionError):
                self.failed = True
            else:
                if event == "[DONE]":
                    self.terminal |= self.chat
                    self.failed |= not self.chat or not self.finished
                else:
                    self.failed |= "error" in event or event.get("type") == "error"
                    self.terminal |= not self.chat and event.get("type") == "message_stop"
                    if self.chat:
                        try:
                            self.finished |= any(choice.get("finish_reason") for choice in chat_choices(event))
                        except ValueError:
                            self.failed = True
        self.event = b""
        self.data.clear()
        self.size = 0

    def close(self) -> None:
        if self.partial or self.data or self.event:
            self.failed = True


def error(status: int, code: str, queued: float = 0) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": code.replace("_", " ")}},
        status=status,
        headers={QUEUE_WAIT_HEADER: f"{queued:.6f}"},
    )


def admission_fields(request: web.Request) -> tuple[str, str, float]:
    root = request.headers.get("X-Workflow-Id", "")
    agent = request.headers.get("X-Agent-Id", root)
    work = float(request.headers.get("X-Solo-Work-Seconds", "nan"))
    if not ID.fullmatch(root) or not ID.fullmatch(agent) or not math.isfinite(work) or work <= 0:
        raise ValueError("expected local workflow, agent and solo-work fields")
    return root, agent, work


class Router:
    def __init__(self, config: RouterConfig, clock: Callable[[], float]):
        self.upstream = local_endpoint(config.upstream)
        self.config = config
        self.gate = Gate(config, clock)
        self.handlers: set[asyncio.Task] = set()

    async def lifecycle(self, _app: web.Application):
        async with ClientSession(
            connector=TCPConnector(limit=self.config.capacity),
            timeout=ClientTimeout(total=self.config.request_timeout, connect=5),
            cookie_jar=DummyCookieJar(),
            auto_decompress=False,
            trust_env=False,
        ) as self.client:
            yield

    async def shutdown(self, _app: web.Application) -> None:
        handlers = tuple(self.handlers)
        for handler in handlers:
            handler.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)

    async def state(self, _request: web.Request) -> web.Response:
        self.gate.advance()
        snapshot = self.gate.scheduler.snapshot(self.gate.clock())
        snapshot.update(request_timeout=self.config.request_timeout, max_body_bytes=self.config.max_body_bytes)
        return web.json_response(snapshot)

    async def handle(self, request: web.Request) -> web.StreamResponse:
        handler = asyncio.current_task()
        self.handlers.add(handler)
        spec = downstream = work_done = queue_start = None
        queued = 0.0
        try:
            async with asyncio.timeout(self.config.request_timeout):
                try:
                    root, agent, work = admission_fields(request)
                    raw = await request.read()
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("expected a JSON object")
                    if payload.get("model") != self.config.model:
                        return error(404, "model_not_supported")
                    payload["model"] = self.config.served_model
                    stream = payload.get("stream") is True
                    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
                    if len(body) > self.config.max_body_bytes:
                        return error(413, "request_too_large")
                except (ValueError, RecursionError):
                    return error(400, "invalid_request_or_workflow_metadata")

                # aiohttp retains the original body for the handler lifetime.
                spec = Request(uuid.uuid4().hex, root, agent, work, len(raw) + len(body))
                del payload, raw
                queue_start = self.gate.clock()
                queued = await self.gate.admit(spec)
                work_done = 0.0
                headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
                for name in ("Anthropic-Version", "Anthropic-Beta"):
                    if name in request.headers:
                        headers[name] = request.headers[name]

                async with self.client.post(
                    self.upstream.with_path(request.path),
                    data=body,
                    headers=headers,
                    allow_redirects=False,
                ) as upstream:
                    if 300 <= upstream.status < 400:
                        return error(502, "upstream_redirect", queued)
                    ok = 200 <= upstream.status < 300
                    is_sse = upstream.content_type == "text/event-stream"
                    encoding = upstream.headers.get("Content-Encoding", "identity").lower()
                    if ok and (
                        (stream and not is_sse)
                        or (is_sse and encoding != "identity")
                    ):
                        return error(502, "upstream_stream_format", queued)
                    completion = StreamCompletion(request.path) if ok and is_sse else None
                    unary = bytearray() if ok and not is_sse else None
                    forwarded = {
                        name: upstream.headers[name]
                        for name in ("Content-Type", "Content-Encoding", "Retry-After")
                        if name in upstream.headers
                    }
                    forwarded[QUEUE_WAIT_HEADER] = f"{queued:.6f}"
                    downstream = web.StreamResponse(status=upstream.status, headers=forwarded)
                    await downstream.prepare(request)
                    async for chunk in upstream.content.iter_chunked(64 << 10):
                        if completion:
                            completion.feed(chunk)
                        if unary is not None:
                            if len(unary) + len(chunk) <= self.config.max_body_bytes:
                                unary.extend(chunk)
                            else:
                                unary = None
                        await downstream.write(chunk)
                    if completion:
                        completion.close()
                    if completion and not (completion.terminal or completion.failed):
                        raise web.HTTPBadGateway()
                    await downstream.write_eof()
                    if ok and ((completion is not None and completion.terminal and not completion.failed)
                               or (unary is not None and unary_success(unary, request.path))):
                        work_done = work
                    return downstream
        except Rejected as exc:
            queued = self.gate.clock() - queue_start if exc.code == "queue_timeout" else 0
            return error(429, exc.code, queued)
        except web.HTTPRequestEntityTooLarge:
            return error(413, "request_too_large")
        except asyncio.CancelledError:
            work_done = None
            raise
        except (TimeoutError, ClientError, ConnectionError, web.HTTPBadGateway) as exc:
            if downstream is not None and downstream.prepared:
                # Headers are committed. Break the transfer so EOF cannot look like success.
                if request.transport is not None:
                    request.transport.abort()
                return downstream
            if isinstance(exc, TimeoutError):
                if queue_start is not None and work_done is None:
                    queued = self.gate.clock() - queue_start
                return error(504, "request_timeout", queued)
            return error(502, "upstream_error", queued)
        finally:
            if spec is not None:
                self.gate.finish(spec.id, work_done)
            self.handlers.remove(handler)


ROUTER = web.AppKey("router", Router)


async def health(_request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def create_app(config: RouterConfig, *, clock: Callable[[], float] = time.monotonic) -> web.Application:
    if config.max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be positive")
    if not math.isfinite(config.request_timeout) or config.request_timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    router = Router(config, clock)
    app = web.Application(
        client_max_size=config.max_body_bytes + 1,
        handler_args={"handler_cancellation": True, "access_log": None},
    )
    app[ROUTER] = router
    app.cleanup_ctx.append(router.lifecycle)
    app.on_shutdown.append(router.shutdown)
    app.add_routes([
        web.get("/healthz", health),
        web.get("/state", router.state),
        web.post(CHAT, router.handle),
        web.post(MESSAGES, router.handle),
    ])
    return app


def main() -> None:
    defaults = RouterConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default=defaults.upstream)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--mode", choices=MODES, default=defaults.mode)
    parser.add_argument("--capacity", type=int, default=defaults.capacity)
    parser.add_argument("--served-model", default=defaults.served_model)
    parser.add_argument("--max-pending", type=int, default=defaults.max_pending)
    parser.add_argument("--max-pending-bytes", type=int, default=defaults.max_pending_bytes)
    parser.add_argument("--max-pending-per-root", type=int, default=defaults.max_pending_per_root)
    parser.add_argument("--queue-timeout", type=float, default=defaults.queue_timeout)
    parser.add_argument("--request-timeout", type=float, default=defaults.request_timeout)
    args = vars(parser.parse_args())
    port = args.pop("port")
    try:
        app = create_app(RouterConfig(**args))
    except ValueError as exc:
        parser.error(str(exc))
    web.run_app(app, host="127.0.0.1", port=port, shutdown_timeout=5)


if __name__ == "__main__":
    main()
