import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web

from replay import run_replay, solo_costs, summarize
from router import CHAT, QUEUE_WAIT_HEADER, SERVED_MODEL, RouterConfig, create_app
from scheduler import Request
from simulate import Turn, Workflow
from test_router import serve


COMPLETE = (
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n'
    b'data: {"choices":[\r\n'
    b'data: {"delta":{"reasoning_content":"first"}}]}\r\n\r\n'
    b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":5,'
    b'"prompt_tokens_details":{"cached_tokens":20}}}\n\n'
    b'data: [DONE]\n\n'
)


def turn(id_, after=(), agent="leader", cancel_after=None):
    return Turn(Request(id_, "r", f"r/{agent}", 1, 100), after, 0, 10, 5, cancel_after)


async def replay(url, turns, **overrides):
    trace = (Workflow("r", 0, tuple(turns)),)
    options = dict(time_scale=1, max_inflight=4, request_timeout=5, root_timeout=10)
    options.update(overrides)
    return await run_replay(trace, url, solo_costs(trace, None), **options)


@asynccontextmanager
async def fake_router(handler, *, capacity=4, mode="fifo"):
    async def state(_request):
        return web.json_response({"mode": mode, "capacity": capacity, "workers": 1})

    app = web.Application()
    app.router.add_post(CHAT, handler)
    app.router.add_get("/state", state)
    async with serve(app) as url:
        yield url


class ReplayTest(unittest.IsolatedAsyncioTestCase):
    async def test_tracks_causal_children_and_usage_through_router(self):
        seen = {}
        completed = set()

        async def engine(request):
            body = await request.json()
            id_ = body["messages"][-1]["content"].split(":", 1)[0].split()[-1]
            self.assertEqual(body["model"], SERVED_MODEL)
            self.assertTrue(body["stream_options"]["include_usage"])
            self.assertNotIn("X-Agent-Id", request.headers)
            seen[id_] = (body["messages"], completed.copy())
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(COMPLETE[:95])
            completed.add(id_)
            await response.write(COMPLETE[95:])
            return response

        app = web.Application()
        app.router.add_post(CHAT, engine)
        async with serve(app) as upstream:
            router = create_app(RouterConfig(upstream=upstream, capacity=2, mode="fair"))
            async with serve(router) as origin:
                settings, roots, rows = await replay(origin, [
                    turn("start"), turn("a", ("start",), "a"), turn("b", ("start",), "b"),
                    turn("join", ("a", "b")),
                ])

        self.assertEqual(set(seen), {"start", "a", "b", "join"})
        self.assertEqual(settings, {"mode": "fair", "capacity": 2, "workers": 1})
        self.assertEqual(seen["a"][0][:1], seen["start"][0])
        self.assertEqual(seen["join"][0][:1], seen["start"][0])
        self.assertTrue({"a", "b"} <= seen["join"][1])
        self.assertEqual(roots[0]["outcome"], "success")
        self.assertEqual(roots[0]["solo_work_s"], 4)
        self.assertEqual(roots[0]["completed_turns"], 4)
        for row in rows:
            self.assertEqual(row.outcome, "success")
            self.assertEqual((row.prompt_tokens, row.cached_tokens, row.output_tokens), (40, 20, 5))
            self.assertIsNotNone(row.ttft_s)
            self.assertIsNotNone(row.router_wait_s)
        report = summarize(roots, rows)
        self.assertEqual(report["cache_hit_share_where_reported"], 0.5)
        self.assertEqual(report["turn_outcomes"], {"success": 4})

    async def test_http_reject_and_committed_stream_failures_count_every_root(self):
        responses = {
            "rejected": web.Response(status=429, headers={QUEUE_WAIT_HEADER: "0.125"}),
            "expired": web.json_response({"error": {"code": "queue_timeout"}}, status=429),
            "stream_error": web.Response(body=b'event: error\ndata: {}\n\n' + COMPLETE,
                                         content_type="text/event-stream"),
            "truncated_stream": web.Response(body=COMPLETE.split(b"data: [DONE]")[0],
                                              content_type="text/event-stream"),
            "malformed_stream": web.Response(body=COMPLETE + b'data: {}\n', content_type="text/event-stream"),
            "missing_usage": web.Response(body=b'data: {"choices":[{"finish_reason":"length"}]}\n\n'
                                               b'data: [DONE]\n\n', content_type="text/event-stream"),
        }
        for outcome, response in responses.items():
            with self.subTest(outcome=outcome):
                issued = []

                async def reject(request):
                    issued.append(await request.json())
                    return response

                async with fake_router(reject) as url:
                    _, roots, rows = await replay(url, [turn("start"), turn("next", ("start",))])
                self.assertEqual(len(issued), 1)
                self.assertEqual([r.outcome for r in rows], [outcome, "dependency_failed"])
                self.assertEqual(roots[0]["outcome"], "failed")
                self.assertIsNone(roots[0]["normalized_slowdown"])
                self.assertEqual(summarize(roots, rows)["completion_fraction"], 0)
                if outcome == "expired":
                    self.assertEqual(rows[0].reason, "queue_timeout")

    async def test_replay_does_not_follow_http_redirects(self):
        destination_calls = []

        async def destination(request):
            destination_calls.append(request.method)
            return web.Response()

        async def redirect(_request):
            raise web.HTTPTemporaryRedirect("/destination")

        async def state(_request):
            return web.json_response({"mode": "bypass", "capacity": 1, "workers": 1})

        app = web.Application()
        app.router.add_post(CHAT, redirect)
        app.router.add_post("/destination", destination)
        app.router.add_get("/state", state)
        async with serve(app) as url:
            _, roots, rows = await replay(url, [turn("start")])
        self.assertEqual(destination_calls, [])
        self.assertEqual((rows[0].status, rows[0].outcome), (307, "http_error"))
        self.assertEqual(roots[0]["outcome"], "failed")

    async def test_client_deadline_keeps_the_unfinished_workflow_in_results(self):
        async def hang(_request):
            await asyncio.Future()

        async with fake_router(hang) as url:
            _, roots, rows = await replay(url, [turn("start", cancel_after=0), turn("next", ("start",))])
        self.assertEqual([r.outcome for r in rows], ["client_canceled", "dependency_failed"])
        self.assertEqual((roots[0]["outcome"], roots[0]["completed_turns"]), ("failed", 0))

    async def test_rejects_nonloopback_before_connecting(self):
        with self.assertRaises(ValueError):
            await replay("http://192.0.2.1:8000", [turn("start")])

    async def test_connector_allows_configured_concurrency_above_aiohttp_default(self):
        concurrency = 101
        seen = 0
        full, release = asyncio.Event(), asyncio.Event()

        async def engine(request):
            nonlocal seen
            await request.read()
            seen += 1
            if seen == concurrency:
                full.set()
            await release.wait()
            return web.Response(body=COMPLETE, content_type="text/event-stream")

        trace = tuple(Workflow(key, 0, (Turn(Request(key, key, "leader", 1, 100), (), 0, 1, 5),))
                      for key in map(str, range(concurrency)))
        async with fake_router(engine, capacity=concurrency) as url:
            async with asyncio.TaskGroup() as group:
                run = group.create_task(run_replay(
                    trace, url, solo_costs(trace, None), time_scale=1, max_inflight=concurrency,
                    request_timeout=30, root_timeout=30,
                ))
                try:
                    await asyncio.wait_for(full.wait(), 10)
                finally:
                    release.set()
        self.assertEqual(len(run.result()[1]), concurrency)
        self.assertTrue(all(row.outcome == "success" for row in run.result()[2]))


class CalibrationTest(unittest.TestCase):
    def test_fixed_costs_must_cover_exactly_the_same_trace(self):
        trace = (Workflow("r", 0, (turn("start"), turn("next", ("start",)))),)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solo.json"
            values = {"start": 1.2, "next": 2.3}
            path.write_text(json.dumps(values))
            self.assertEqual(solo_costs(trace, path), values)
            path.write_text(json.dumps({"start": 1.2}))
            with self.assertRaisesRegex(ValueError, "turn IDs"):
                solo_costs(trace, path)
            path.write_text(json.dumps({"start": 1.2, "next": 0}))
            with self.assertRaisesRegex(ValueError, "positive seconds"):
                solo_costs(trace, path)
