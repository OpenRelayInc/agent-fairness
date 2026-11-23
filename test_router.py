import asyncio
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace

from aiohttp import ClientConnectionError, ClientPayloadError, ClientSession, ClientTimeout, web

from router import (CHAT, MESSAGES, MODEL, QUEUE_WAIT_HEADER, ROUTER, SERVED_MODEL,
                    RouterConfig, StreamCompletion, create_app, local_endpoint, unary_success)

HANG_GUARD = 10


@asynccontextmanager
async def serve(app):
    runner = web.AppRunner(app, access_log=None, handler_cancellation=True, shutdown_timeout=HANG_GUARD)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


@asynccontextmanager
async def rig(handler, *, clock=None, **overrides):
    engine = web.Application()
    engine.router.add_post("/{path:.*}", handler)
    async with serve(engine) as upstream:
        config = RouterConfig(upstream=upstream, **overrides)
        app = create_app(config) if clock is None else create_app(config, clock=clock)
        async with serve(app) as router, ClientSession(timeout=ClientTimeout(total=HANG_GUARD)) as client:
            yield app[ROUTER], client, router


async def post(client, router, root, *, path=CHAT, stream=False, **kwargs):
    headers = {
        "X-Workflow-Id": root,
        "X-Solo-Work-Seconds": "0.5",
        "Authorization": "Bearer local-client-sentinel",
        "X-Api-Key": "local-client-sentinel",
        "Cookie": "local-client-sentinel",
        "Anthropic-Version": "2023-06-01",
    }
    headers.update(kwargs.pop("headers", {}))
    return await client.post(router + path, headers=headers, json={"model": MODEL, "stream": stream, **kwargs})


async def state_until(client, router, predicate):
    async with asyncio.timeout(HANG_GUARD):
        while True:
            async with client.get(router + "/state") as response:
                state = await response.json()
            if predicate(state):
                return state


class RouterTest(unittest.IsolatedAsyncioTestCase):
    def test_malformed_or_post_terminal_sse_never_earns_work_credit(self):
        for payload in (
            b'data: [DONE]\n\n',
            b'data: {"choices":true}\n\ndata: [DONE]\n\n',
            b'data: bad-json\n\ndata: [DONE]\n\n',
            b'data:  {"error":"failed"}\n\ndata: [DONE]\n\n',
            b'data: []\n\ndata: [DONE]\n\n',
            b'data: [DONE]\n\ndata: {"choices":[]}\n\n',
            b'data: [DONE]\n\ntrailing-partial-event',
        ):
            with self.subTest(payload=payload):
                completion = StreamCompletion(CHAT)
                completion.feed(payload)
                completion.close()
                self.assertTrue(completion.failed)

    async def test_unary_error_in_success_status_does_not_earn_work_credit(self):
        async def reply(_request):
            return web.json_response({"error": {"message": "failed"}})

        async with rig(reply) as (engine, client, router):
            async with await post(client, router, "root") as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(await response.json(), {"error": {"message": "failed"}})
            await state_until(client, router, lambda state: state["running"] == 0)
            self.assertEqual(engine.gate.scheduler.roots["root"].work, 0)
        self.assertFalse(unary_success(b'{"choices":[],"usage":{}}', CHAT))
        self.assertTrue(unary_success(b'{"choices":[{"message":{},"finish_reason":"stop"}]}', CHAT))
        self.assertTrue(unary_success(b'{"type":"message","content":[]}', MESSAGES))

    async def test_relays_chat_and_messages_in_chunks_without_leaking_headers(self):
        for path, terminal in (
            (CHAT, b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'),
            (MESSAGES, b"event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n"),
        ):
            with self.subTest(path=path):
                release = asyncio.Event()
                first = b'data: {"delta":"first"}\r\n\r\n'
                seen = asyncio.Future()

                async def reply(request):
                    seen.set_result((dict(request.headers), await request.json(), request.path))
                    stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                    await stream.prepare(request)
                    await stream.write(first)
                    await release.wait()
                    await stream.write(terminal[:4])
                    await stream.write(terminal[4:])
                    return stream

                async with rig(reply) as (engine, client, router):
                    try:
                        async with await post(client, router, "root", path=path, stream=True, tools=[{"name": "check"}]) as response:
                            self.assertEqual(response.status, 200)
                            self.assertEqual(float(response.headers[QUEUE_WAIT_HEADER]), 0)
                            self.assertEqual(await response.content.readexactly(len(first)), first)
                            release.set()
                            self.assertEqual(await response.read(), terminal)
                    finally:
                        release.set()
                    headers, body, sent_path = await seen
                    headers = {name.lower(): value for name, value in headers.items()}
                    for name in ("authorization", "x-api-key", "cookie", "x-workflow-id", "x-solo-work-seconds"):
                        self.assertNotIn(name, headers)
                    self.assertEqual(headers["anthropic-version"], "2023-06-01")
                    self.assertEqual((body, sent_path), ({"model": SERVED_MODEL, "stream": True, "tools": [{"name": "check"}]}, path))
                    state = await state_until(client, router, lambda value: value["running"] == 0)
                    self.assertEqual(state["stats"]["completed"], 1)
                    self.assertEqual(engine.gate.scheduler.roots["root"].work, 0.5)

    async def test_truncation_breaks_transfer_and_errors_do_not_earn_progress(self):
        for name, payload, incomplete in (
            ("missing terminal", b'data: {"choices":[{"text":"partial"}]}\n\n', True),
            ("explicit error", b'event: error\ndata: {"error":{"message":"failed"}}\n\n', False),
        ):
            with self.subTest(name=name):
                release = asyncio.Event()

                async def reply(request):
                    await request.read()
                    stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                    await stream.prepare(request)
                    await stream.write(payload)
                    await release.wait()
                    return stream

                async with rig(reply, capacity=1) as (engine, client, router):
                    try:
                        async with await post(client, router, "root", stream=True) as response:
                            self.assertEqual(await response.content.readexactly(len(payload)), payload)
                            release.set()
                            if incomplete:
                                with self.assertRaises(ClientPayloadError):
                                    await response.read()
                            else:
                                self.assertEqual(await response.read(), b"")
                    finally:
                        release.set()
                    state = await state_until(client, router, lambda value: value["running"] == 0)
                    self.assertEqual(state["stats"]["failed"], 1)
                    self.assertEqual(engine.gate.scheduler.roots["root"].work, 0)

    async def test_refusal_and_redirect_responses_release_capacity_and_do_not_retry(self):
        seen = []

        async def reply(request):
            seen.append(request.path)
            body = await request.json()
            if body.get("redirect"):
                raise web.HTTPTemporaryRedirect(location="/unexpected")
            return web.json_response({"error": "overloaded"}, status=503, headers={"Retry-After": "1"})

        async with rig(reply, capacity=1) as (_, client, router):
            async with await post(client, router, "root") as response:
                self.assertEqual(response.status, 503)
                self.assertEqual(response.headers["Retry-After"], "1")
                self.assertEqual(await response.json(), {"error": "overloaded"})
            async with await post(client, router, "root", redirect=True) as response:
                self.assertEqual(response.status, 502)
                self.assertEqual((await response.json())["error"]["code"], "upstream_redirect")
            self.assertEqual(seen, [CHAT, CHAT])

    async def test_bypass_refuses_at_capacity_without_queueing(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def reply(request):
            await request.read()
            entered.set()
            await release.wait()
            return web.json_response({"ok": True})

        async with rig(reply, capacity=1) as (_, client, router):
            busy = asyncio.create_task(post(client, router, "busy"))
            try:
                await asyncio.wait_for(entered.wait(), HANG_GUARD)
                async with await post(client, router, "overflow") as response:
                    self.assertEqual(response.status, 429)
                    self.assertEqual((await response.json())["error"]["code"], "at_capacity")
                release.set()
                async with await busy as response:
                    self.assertEqual(await response.json(), {"ok": True})
            finally:
                release.set()
                if not busy.done():
                    busy.cancel()
                await asyncio.gather(busy, return_exceptions=True)

    async def test_queue_bounds_expiry_and_client_disconnect(self):
        entered = asyncio.Event()
        upstream_canceled = asyncio.Event()
        now = 100.0

        async def reply(request):
            await request.read()
            entered.set()
            try:
                await asyncio.Future()
            finally:
                upstream_canceled.set()

        async with rig(reply, mode="fifo", capacity=1, max_pending=1, clock=lambda: now) as (engine, client, router):
            active = asyncio.create_task(post(client, router, "running"))
            queued = None
            try:
                await asyncio.wait_for(entered.wait(), HANG_GUARD)
                queued = asyncio.create_task(post(client, router, "queued"))
                await state_until(client, router, lambda value: value["pending"] == 1)
                async with await post(client, router, "full") as response:
                    self.assertEqual(response.status, 429)
                    self.assertEqual((await response.json())["error"]["code"], "queue_full")
                queued.cancel()
                await asyncio.gather(queued, return_exceptions=True)
                state = await state_until(client, router, lambda value: value["pending"] == 0)
                self.assertEqual(state["pending_bytes"], 0)

                queued = asyncio.create_task(post(client, router, "expires"))
                await state_until(client, router, lambda value: value["pending"] == 1)
                now += 30
                engine.gate.advance()
                async with await queued as response:
                    self.assertEqual(response.status, 429)
                    self.assertEqual((await response.json())["error"]["code"], "queue_timeout")
                    self.assertEqual(float(response.headers[QUEUE_WAIT_HEADER]), 30)

                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
                await asyncio.wait_for(upstream_canceled.wait(), HANG_GUARD)
                state = await state_until(client, router, lambda value: value["running"] == 0)
                self.assertEqual(state["stats"]["canceled"], 2)
                self.assertEqual(state["stats"]["expired"], 1)
            finally:
                tasks = [task for task in (active, queued) if task is not None]
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_rejects_invalid_workflow_headers_and_oversized_bodies(self):
        async def reply(_request):
            self.fail("rejected requests must not reach the upstream")

        async with rig(reply, max_body_bytes=256) as (_, client, router):
            async with await post(client, router, "root", headers={"X-Solo-Work-Seconds": "inf"}) as response:
                self.assertEqual(response.status, 400)
            async with await post(client, router, "root", messages=[{"content": "x" * 256}]) as response:
                self.assertEqual(response.status, 413)

    async def test_shutdown_releases_running_and_pending_slots(self):
        entered = asyncio.Event()
        disconnected = asyncio.Event()

        async def reply(request):
            await request.read()
            entered.set()
            try:
                await asyncio.Future()
            finally:
                disconnected.set()

        engine = web.Application()
        engine.router.add_post(CHAT, reply)
        async with serve(engine) as upstream, ClientSession(timeout=ClientTimeout(total=HANG_GUARD)) as client:
            app = create_app(RouterConfig(upstream=upstream, mode="fair", capacity=1))
            tasks = []
            try:
                async with serve(app) as router:
                    tasks.append(asyncio.create_task(post(client, router, "running")))
                    await asyncio.wait_for(entered.wait(), HANG_GUARD)
                    tasks.append(asyncio.create_task(post(client, router, "queued")))
                    await state_until(client, router, lambda value: value["pending"] == 1)
                await asyncio.wait_for(disconnected.wait(), HANG_GUARD)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                self.assertTrue(all(isinstance(result, ClientConnectionError) for result in results), results)
                gate = app[ROUTER].gate
                state = gate.scheduler.snapshot(gate.clock())
                self.assertEqual((state["running"], state["pending"], state["pending_bytes"]), (0, 0, 0))
                self.assertEqual(gate.waiters, {})
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    def test_accepts_only_numeric_loopback_http_origins(self):
        self.assertEqual(str(local_endpoint("http://127.0.0.1:8080/")), "http://127.0.0.1:8080")
        self.assertEqual(str(local_endpoint("http://[::1]:8080")), "http://[::1]:8080")
        for value in (
            "https://127.0.0.1:8080", "http://127.0.0.1", "http://localhost:8080",
            "http://inference.example.com:80", "http://192.168.0.1:8080",
            "http://user@127.0.0.1:8080", "http://127.0.0.1:8080/v1",
            "http://127.0.0.1:8080?arg=1",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                create_app(replace(RouterConfig(), upstream=value))


if __name__ == "__main__":
    unittest.main()
