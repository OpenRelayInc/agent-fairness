#!/usr/bin/env python3
"""Replay seeded synthetic agent workflows through a loopback router."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import uuid

import aiohttp

from router import CHAT, MODEL, MODES, QUEUE_WAIT_HEADER, chat_choices, decode_sse_payload, local_endpoint
from simulate import DEFAULT_WORKLOADS, fair_entitlement, load_scenarios, make_trace, trace_fingerprint, write_rows


WORDS = "the task requires checking files tests state logs and report results before changing any code".split()
MAX_TURNS = 4096
MAX_EVENT_BYTES = 1024 * 1024
MAX_REPLAY_BODY_BYTES = 256 * 1024 * 1024
MAX_ERROR_BYTES = 4096


@dataclass
class TurnResult:
    id: str
    root: str
    agent: str
    solo_work_s: float
    input_target: int
    output_target: int
    after: str = ""
    outcome: str = "not_issued"
    reason: str | None = None
    ready_s: float | None = None
    sent_s: float | None = None
    done_s: float | None = None
    status: int | None = None
    router_wait_s: float | None = None
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None


def chat_history(turn, previous):
    """Use a causal parent's entire history as the prefix for the next turn."""
    req = turn.request
    parents = [previous[id_] for id_ in turn.after]
    same_agent = [history for agent, history in parents if agent == req.agent]
    history = max(same_agent or [history for _, history in parents], key=len, default=())
    if history:
        history += ({"role": "assistant", "content": "I checked the previous fictional step."},)
    rng = random.Random(req.id)
    words = " ".join(rng.choices(WORDS, k=turn.input_tokens))
    kind = "Fictional tool results" if parents else "Synthetic task"
    message = f"{kind} for {req.id}: {words}"
    return history + ({"role": "user", "content": message},)


def prepare_bodies(trace):
    """Render fictional input before starting the client clock."""
    bodies = {}
    size = 0
    for workflow in trace:
        previous = {}
        for turn in workflow.turns:
            req = turn.request
            history = chat_history(turn, previous)
            previous[req.id] = (req.agent, history)
            body = json.dumps({
                "model": MODEL,
                "messages": history,
                "max_tokens": turn.output_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }, separators=(",", ":")).encode()
            size += len(body)
            if size > MAX_REPLAY_BODY_BYTES:
                raise ValueError("replay bodies exceed 256 MiB; reduce the workflow count")
            bodies[req.id] = body
    return bodies


def optional_float(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def token_count(value):
    return value if type(value) is int and value >= 0 else None


async def sse_events(content):
    parts = []
    event = b""
    size = 0
    async for raw in content:
        if not raw.endswith(b"\n"):
            raise ValueError("partial SSE line at EOF")
        line = raw.rstrip(b"\r\n")
        if not line:
            if parts or event == b"error":
                yield event, b"\n".join(parts)
            parts.clear()
            event = b""
            size = 0
            continue
        size += len(raw)
        if size > MAX_EVENT_BYTES:
            raise ValueError("SSE event too large")
        field, _, value = line.partition(b":")
        value = value.removeprefix(b" ")
        if field == b"data":
            parts.append(value)
        elif field == b"event":
            event = value
    if parts or event:
        raise ValueError("partial SSE event at EOF")


async def read_stream(response, result, now):
    if response.content_type != "text/event-stream":
        return "unexpected_content_type"
    finished = False
    done = False
    try:
        async for name, data in sse_events(response.content):
            if done:
                return "malformed_stream"
            if name == b"error":
                return "stream_error"
            event = decode_sse_payload(data)
            if event == "[DONE]":
                done = True
                continue
            if "error" in event or event.get("type") == "error":
                return "stream_error"
            for choice in chat_choices(event):
                delta = choice.get("delta", {})
                if result.ttft_s is None and any(delta.get(key) for key in (
                    "content", "reasoning", "reasoning_content", "tool_calls", "function_call"
                )):
                    result.ttft_s = now() - result.sent_s
                finished |= bool(choice.get("finish_reason"))
            if usage := event.get("usage"):
                result.prompt_tokens = token_count(usage.get("prompt_tokens"))
                result.output_tokens = token_count(usage.get("completion_tokens"))
                details = usage.get("prompt_tokens_details") or {}
                result.cached_tokens = token_count(details.get("cached_tokens"))
    except (ValueError, AttributeError, TypeError, RecursionError):
        return "malformed_stream"
    except aiohttp.ClientPayloadError:
        return "truncated_stream"
    if not (done and finished):
        return "truncated_stream"
    if result.prompt_tokens is None or result.output_tokens is None:
        return "missing_usage"
    return "success"


async def read_error_code(response):
    if response.content_type != "application/json":
        return None
    body = bytearray()
    while len(body) <= MAX_ERROR_BYTES:
        chunk = await response.content.read(MAX_ERROR_BYTES + 1 - len(body))
        if not chunk:
            break
        body.extend(chunk)
    if len(body) > MAX_ERROR_BYTES:
        return None
    try:
        code = json.loads(body).get("error", {}).get("code")
    except (ValueError, AttributeError, TypeError, RecursionError):
        return None
    if isinstance(code, str) and 0 < len(code) <= 64 and code.isascii() and code.replace("_", "").isalnum():
        return code
    return None


async def call_router(session, url, body, result, now, run_id):
    result.sent_s = now()
    headers = {
        "Content-Type": "application/json",
        "X-Workflow-Id": f"{run_id}:{result.root}",
        "X-Agent-Id": hashlib.sha256(result.agent.encode()).hexdigest()[:32],
        "X-Solo-Work-Seconds": str(result.solo_work_s),
    }
    try:
        async with session.post(url, data=body, headers=headers, allow_redirects=False) as response:
            result.status = response.status
            result.router_wait_s = optional_float(response.headers.get(QUEUE_WAIT_HEADER))
            if response.status == 200:
                return await read_stream(response, result, now)
            result.reason = await read_error_code(response)
            if response.status == 429:
                return "expired" if result.reason == "queue_timeout" else "rejected"
            if response.status == 504 and result.reason == "request_timeout":
                return "request_timeout"
            return "http_error"
    except aiohttp.ClientError:
        return "transport_error"


async def router_settings(session, origin):
    try:
        async with asyncio.timeout(5):
            async with session.get(origin.with_path("/state"), allow_redirects=False) as response:
                if response.status != 200:
                    raise ValueError(f"router /state returned HTTP {response.status}")
                state = await response.json()
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise ValueError("could not read local router /state") from exc
    if not isinstance(state, dict) or state.get("mode") not in MODES:
        raise ValueError("router /state did not report a recognized mode")
    if any(type(state.get(key)) is not int or state[key] <= 0 for key in ("capacity", "workers")):
        raise ValueError("router /state did not report positive capacity and worker counts")
    return {key: state[key] for key in ("mode", "capacity", "workers")}


async def replay_workflow(workflow, session, url, gate, clock, time_scale, request_timeout, root_timeout,
                          costs, bodies, run_id):
    arrival = workflow.arrival * time_scale
    await asyncio.sleep(max(0, arrival - clock()))
    started = clock()
    rows = {
        turn.request.id: TurnResult(
            turn.request.id, workflow.id, turn.request.agent, costs[turn.request.id],
            turn.input_tokens, turn.output_tokens, " ".join(turn.after),
        ) for turn in workflow.turns
    }
    finished = {id_: asyncio.Event() for id_ in rows}

    async def one_turn(turn):
        req = turn.request
        result = rows[req.id]
        try:
            for predecessor in turn.after:
                await finished[predecessor].wait()
            if any(rows[id_].outcome != "success" for id_ in turn.after):
                result.outcome = "dependency_failed"
                return
            await asyncio.sleep(turn.tool_gap * time_scale)
            result.ready_s = clock()
            deadline = request_timeout
            cancel = turn.cancel_after is not None and turn.cancel_after <= request_timeout
            if cancel:
                deadline = turn.cancel_after
            try:
                async with asyncio.timeout(deadline):
                    async with gate:
                        result.outcome = await call_router(session, url, bodies[req.id], result, clock, run_id)
            except TimeoutError:
                result.outcome = "client_canceled" if cancel else "request_timeout"
        except asyncio.CancelledError:
            result.outcome = "root_timeout"
            raise
        finally:
            if result.ready_s is not None:
                result.done_s = clock()
            finished[req.id].set()

    try:
        async with asyncio.timeout(root_timeout):
            async with asyncio.TaskGroup() as group:
                for turn in workflow.turns:
                    group.create_task(one_turn(turn))
    except TimeoutError:
        outcome = "root_timeout"
    else:
        outcomes = {r.outcome for r in rows.values()}
        outcome = "success" if outcomes == {"success"} else "failed"
    return {
        "id": workflow.id,
        "outcome": outcome,
        "scheduled_s": arrival,
        "started_s": started,
        "done_s": clock(),
        "planned_turns": len(rows),
        "offered_turns": sum(r.ready_s is not None for r in rows.values()),
        "completed_turns": sum(r.outcome == "success" for r in rows.values()),
        "solo_work_s": sum(r.solo_work_s for r in rows.values()),
        "inference_s": sum(r.done_s - r.ready_s for r in rows.values() if r.ready_s is not None),
    }, list(rows.values())


def apply_entitlement(roots, turns, workers):
    entitlement = fair_entitlement((
        (row.root, row.agent, row.ready_s, row.done_s)
        for row in turns if row.ready_s is not None
    ), workers)
    for root in roots:
        root["entitlement_s"] = entitlement[root["id"]]
        success = root["outcome"] == "success"
        root["raw_slowdown"] = root["inference_s"] / root["solo_work_s"] if success else None
        root["normalized_slowdown"] = root["entitlement_s"] / root["solo_work_s"] if success else None


async def run_replay(trace, url, costs, *, time_scale, max_inflight, request_timeout, root_timeout):
    origin = local_endpoint(url)
    endpoint = origin.with_path(CHAT)
    bodies = prepare_bodies(trace)
    gate = asyncio.Semaphore(max_inflight)
    loop = asyncio.get_running_loop()
    run_id = uuid.uuid4().hex

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max_inflight), timeout=aiohttp.ClientTimeout(total=None),
        cookie_jar=aiohttp.DummyCookieJar(), trust_env=False,
    ) as client:
        settings = await router_settings(client, origin)
        t0 = loop.time()

        def clock():
            return loop.time() - t0

        results = await asyncio.gather(*(
            replay_workflow(workflow, client, endpoint, gate, clock, time_scale, request_timeout, root_timeout,
                            costs, bodies, run_id)
            for workflow in trace
        ))
    roots = [root for root, _ in results]
    turns = [turn for _, rows in results for turn in rows]
    apply_entitlement(roots, turns, settings["workers"])
    return settings, roots, turns


def percentiles(values):
    ordered = sorted(value for value in values if value is not None)
    return {
        f"p{p}": ordered[math.ceil(p / 100 * len(ordered)) - 1] if ordered else None
        for p in (50, 95, 99)
    }


def summarize(roots, turns):
    completed = [row for row in turns if row.outcome == "success"]
    elapsed = max(root["done_s"] for root in roots)
    output_tokens = sum(row.output_tokens for row in completed)
    cached_tokens = sum(row.cached_tokens for row in completed if row.cached_tokens is not None)
    cache_reported_tokens = sum(row.prompt_tokens for row in completed if row.cached_tokens is not None)
    return {
        "elapsed_s": elapsed,
        "assigned_roots": len(roots),
        "root_outcomes": dict(Counter(root["outcome"] for root in roots)),
        "turn_outcomes": dict(Counter(row.outcome for row in turns)),
        "completion_fraction": sum(row["outcome"] == "success" for row in roots) / len(roots),
        "successful_output_tokens_per_s": output_tokens / elapsed,
        "completed_roots_per_s": sum(row["outcome"] == "success" for row in roots) / elapsed,
        "cache_reported_successful_turns": sum(row.cached_tokens is not None for row in completed),
        "cache_hit_share_where_reported": cached_tokens / cache_reported_tokens if cache_reported_tokens else None,
        "successful_turn_ttft_s": percentiles(row.ttft_s for row in completed),
        "successful_turn_e2e_s": percentiles(row.done_s - row.ready_s for row in completed),
        "client_wait_s": percentiles(row.sent_s - row.ready_s for row in turns if row.sent_s is not None),
        "router_wait_s": percentiles(row.router_wait_s for row in turns),
        "completed_root_raw_slowdown": percentiles(root["raw_slowdown"] for root in roots),
        "completed_root_normalized_slowdown": percentiles(root["normalized_slowdown"] for root in roots),
    }


def positive(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return n


def solo_costs(trace, path):
    expected = {turn.request.id: turn.request.solo_work for workflow in trace for turn in workflow.turns}
    if path is None:
        return expected
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict) or supplied.keys() != expected.keys():
        raise ValueError("standalone calibration must list exactly the generated turn IDs")
    if any(type(n) not in (int, float) or not 0 < n < math.inf for n in supplied.values()):
        raise ValueError("standalone calibration values must be finite positive seconds")
    return supplied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="router origin on numeric loopback")
    parser.add_argument("--out", type=Path, required=True, help="new directory for JSON and CSV outputs")
    parser.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS)
    parser.add_argument("--scenario", default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-scale", type=positive, default=1.0, help="scale root arrivals and tool gaps only")
    parser.add_argument("--max-inflight", type=int, default=256, help="client connection limit")
    parser.add_argument("--request-timeout", type=positive, default=120.0, help="seconds including client/router wait")
    parser.add_argument("--root-timeout", type=positive, help="wall seconds per workflow (default: scenario deadline)")
    parser.add_argument("--solo-costs", type=Path, help="JSON object of measured standalone seconds by turn ID")
    args = parser.parse_args()
    try:
        local_endpoint(args.url)
        scenario = load_scenarios(args.workloads)[args.scenario]
        trace = make_trace(scenario, args.seed)
        if not math.isfinite(scenario.horizon * args.time_scale):
            raise ValueError("scaled workload horizon must be finite")
        turns = [turn for workflow in trace for turn in workflow.turns]
        if not 1 <= len(turns) <= MAX_TURNS:
            raise ValueError(f"replay needs between 1 and {MAX_TURNS} turns")
        if not 1 <= args.max_inflight <= MAX_TURNS:
            raise ValueError(f"max-inflight must be between 1 and {MAX_TURNS}")
        costs = solo_costs(trace, args.solo_costs)
        args.out.mkdir(parents=True, exist_ok=False)
    except (ValueError, KeyError, OSError) as err:
        parser.error(str(err))
    try:
        settings, roots, records = asyncio.run(run_replay(
            trace, args.url, costs, time_scale=args.time_scale,
            max_inflight=args.max_inflight, request_timeout=args.request_timeout,
            root_timeout=args.root_timeout or scenario.root_deadline,
        ))
    except ValueError as err:
        parser.error(str(err))
    report = {
        "model": MODEL,
        "scenario": scenario.name,
        "seed": args.seed,
        "trace_sha256": trace_fingerprint(trace),
        "solo_cost_source": "supplied calibration" if args.solo_costs else "synthetic trace estimate",
        "solo_cost_sha256": hashlib.sha256(json.dumps(costs, sort_keys=True).encode()).hexdigest(),
        "time_scale": args.time_scale,
        "router": settings,
        "max_inflight": args.max_inflight,
        "request_timeout_s": args.request_timeout,
        "root_timeout_s": args.root_timeout or scenario.root_deadline,
        **summarize(roots, records),
    }
    write_rows(args.out / "requests.csv", [asdict(record) for record in records])
    write_rows(args.out / "roots.csv", roots)
    rendered = json.dumps(report, indent=2)
    (args.out / "summary.json").write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
