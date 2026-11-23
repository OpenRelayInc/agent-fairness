#!/usr/bin/env python3
"""Run when comparing admission policies on a private vLLM replica through loopback.

Measure a frozen solo reference before scoring an engine or policy change. This
tool does not provision capacity, change an engine configuration, or dial public
origins.
"""

import argparse
import asyncio
import csv
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import uuid

import aiohttp

from replay import (MAX_TURNS, TurnResult, apply_entitlement, call_router, percentiles,
                    positive, prepare_bodies, replay_workflow, solo_costs, summarize)
from router import CHAT, MODES, local_endpoint
from simulate import load_scenarios, make_trace, trace_fingerprint, write_rows


ROUTER_COUNT_SETTINGS = ("capacity", "workers", "max_pending", "max_pending_bytes",
                         "max_pending_per_root", "max_roots", "max_body_bytes")
ROUTER_TIME_SETTINGS = ("queue_timeout", "root_ttl", "request_timeout")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_bytes(encoded(value))
    temp.replace(path)


def digest(value):
    return sha256(encoded(value)).hexdigest()


def cache_namespace(trace, bodies, run_id):
    """Separate roots and experimental passes while preserving tokens and causal prefixes."""
    for root in trace:
        salt = sha256(f"{run_id}/{root.id}".encode()).hexdigest()
        for turn in root.turns:
            id_ = turn.request.id
            if b'"cache_salt"' in bodies[id_]:
                raise ValueError("prepared request unexpectedly contains a cache salt")
            bodies[id_] = bodies[id_][:-1] + f',"cache_salt":"{salt}"}}'.encode()
    return bodies


def client(limit=1):
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=limit),
                                 timeout=aiohttp.ClientTimeout(total=None),
                                 cookie_jar=aiohttp.DummyCookieJar(), trust_env=False)


async def read_state(session, origin, mode, capacity):
    async with asyncio.timeout(5), session.get(origin.with_path("/state"), allow_redirects=False) as response:
        if response.status != 200:
            raise ValueError(f"router /state returned HTTP {response.status}")
        state = await response.json()
    if (not isinstance(state, dict) or state.get("mode") != mode or state.get("capacity") != capacity
            or not isinstance(state.get("stats"), dict)):
        raise ValueError(f"expected mode={mode} and capacity={capacity} in router /state")
    return state


def router_settings(state):
    settings = {key: state.get(key) for key in (*ROUTER_COUNT_SETTINGS, *ROUTER_TIME_SETTINGS)}
    for key in ROUTER_COUNT_SETTINGS:
        value = settings[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"router /state did not report a positive integer {key}")
    for key in ROUTER_TIME_SETTINGS:
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"router /state did not report finite positive {key}")
    return settings


async def drain_router(session, origin, mode, capacity):
    async with asyncio.timeout(30):
        while True:
            state = await read_state(session, origin, mode, capacity)
            if state.get("running") == 0 and state.get("pending") == 0:
                return state
            await asyncio.sleep(0.1)


async def engine_metrics(session, origin):
    async with asyncio.timeout(10), session.get(origin.with_path("/metrics"), allow_redirects=False) as response:
        if response.status != 200:
            raise ValueError(f"engine /metrics returned HTTP {response.status}")
        body = bytearray()
        async for chunk in response.content.iter_chunked(64 << 10):
            body.extend(chunk)
            if len(body) > 4 << 20:
                raise ValueError("engine metrics response exceeds 4 MiB")
    return bytes(body)


async def prepare_engine(session, origin, reset):
    if origin is None:
        return
    if reset:
        async with asyncio.timeout(30), session.post(origin.with_path("/reset_prefix_cache"),
                                                     allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError(f"engine /reset_prefix_cache returned HTTP {response.status}")


async def measure_root(session, endpoint, root, bodies, run_id, request_timeout):
    """Measure each generated turn once, replaying preceding turns on the isolated engine."""
    now = asyncio.get_running_loop().time
    rows = []
    for turn in root.turns:
        request = turn.request
        row = TurnResult(request.id, root.id, request.agent, request.solo_work,
                         turn.input_tokens, turn.output_tokens, " ".join(turn.after))
        async with asyncio.timeout(request_timeout):
            row.outcome = await call_router(session, endpoint, bodies[request.id], row, now, run_id)
        row.done_s = now()
        if row.outcome != "success" or row.router_wait_s != 0 or row.output_tokens != turn.output_tokens:
            raise ValueError(f"solo turn {request.id} failed calibration: {row.outcome}, HTTP {row.status}, "
                             f"router wait {row.router_wait_s}, {row.output_tokens}/{turn.output_tokens} output tokens")
        rows.append({"id": request.id, "root": root.id, "seconds": row.done_s - row.sent_s,
                     "ttft_s": row.ttft_s, "prompt_tokens": row.prompt_tokens,
                     "cached_tokens": row.cached_tokens, "output_tokens": row.output_tokens})
    return rows


async def calibrate(args, trace, metadata):
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "checkpoint.json"
    checkpoint = json.loads(path.read_text()) if path.exists() else {"metadata": metadata, "roots": {}}
    if checkpoint.get("metadata") != metadata or not isinstance(checkpoint.get("roots"), dict):
        raise ValueError("saved calibration does not match this trace, engine manifest and cache policy")
    roots = checkpoint["roots"]
    if roots.keys() - {root.id for root in trace}:
        raise ValueError("checkpoint contains roots not in the trace")
    origin = local_endpoint(args.url)
    engine = local_endpoint(args.engine_url) if args.engine_url else None
    run_id = uuid.uuid4().hex
    bodies = cache_namespace(trace, prepare_bodies(trace), run_id)
    async with client() as session:
        await drain_router(session, origin, "bypass", args.capacity)
        await prepare_engine(session, engine, args.reset_prefix_cache)
        warmup = trace[:args.warmup_roots]
        warm_id = uuid.uuid4().hex
        warmup_bodies = cache_namespace(warmup, prepare_bodies(warmup), warm_id)
        for root in warmup:
            await measure_root(session, origin.with_path(CHAT), root, warmup_bodies, warm_id, args.request_timeout)
        for root in trace:
            if root.id in roots:
                rows = roots[root.id]
                if [row.get("id") for row in rows] != [turn.request.id for turn in root.turns]:
                    raise ValueError(f"incomplete checkpoint for {root.id}")
                continue
            roots[root.id] = await measure_root(session, origin.with_path(CHAT), root, bodies, run_id,
                                                args.request_timeout)
            write_json(path, checkpoint)
            if len(roots) % 10 == 0 or len(roots) == len(trace):
                print(f"Calibrated {len(roots)}/{len(trace)} workflows", flush=True)
        await drain_router(session, origin, "bypass", args.capacity)
    rows = [row for root in trace for row in roots[root.id]]
    costs = args.out / "solo-costs.json"
    write_json(costs, {row["id"]: row["seconds"] for row in rows})
    solo_costs(trace, costs)
    write_rows(args.out / "solo-requests.csv", rows)
    write_json(args.out / "calibration.json", {**metadata, "solo_cost_sha256": digest(solo_costs(trace, costs)),
                                              "calibrated_turns": len(rows)})


async def run_arm(args, trace, metadata):
    if args.max_inflight <= args.capacity:
        raise ValueError("max-inflight must exceed router capacity so client limits cannot hide overload")
    calibration = json.loads((args.calibration / "calibration.json").read_text())
    if any(calibration.get(key) != value for key, value in metadata.items()):
        raise ValueError("calibration differs from the requested trace, engine manifest or cache policy")
    costs = solo_costs(trace, args.calibration / "solo-costs.json")
    if digest(costs) != calibration.get("solo_cost_sha256"):
        raise ValueError("calibration cost checksum does not match")
    args.out.mkdir(parents=True, exist_ok=False)
    origin = local_endpoint(args.url)
    engine = local_endpoint(args.engine_url) if args.engine_url else None
    run_id = uuid.uuid4().hex
    bodies = cache_namespace(trace, prepare_bodies(trace), run_id)
    async with client(args.max_inflight) as session:
        settings = router_settings(await drain_router(session, origin, args.mode, args.capacity))
        await prepare_engine(session, engine, args.reset_prefix_cache)
        warmup = trace[:args.warmup_roots]
        warm_id = uuid.uuid4().hex
        warmup_bodies = cache_namespace(warmup, prepare_bodies(warmup), warm_id)
        for root in warmup:
            await measure_root(session, origin.with_path(CHAT), root, warmup_bodies, warm_id, args.request_timeout)
        before = await drain_router(session, origin, args.mode, args.capacity)
        if engine:
            (args.out / "engine-before.prom").write_bytes(await engine_metrics(session, engine))
        loop = asyncio.get_running_loop()
        t0 = loop.time()

        def clock():
            return loop.time() - t0

        gate = asyncio.Semaphore(args.max_inflight)
        results = await asyncio.gather(*(replay_workflow(
            workflow, session, origin.with_path(CHAT), gate, clock, args.time_scale,
            args.request_timeout, args.root_timeout, costs, bodies, run_id) for workflow in trace))
        after = await drain_router(session, origin, args.mode, args.capacity)
        if router_settings(before) != settings or router_settings(after) != settings:
            raise ValueError("router settings changed during the experiment arm")
        if engine:
            (args.out / "engine-after.prom").write_bytes(await engine_metrics(session, engine))
    roots = [root for root, _ in results]
    turns = [turn for _, rows in results for turn in rows]
    # A different output length would change effective work in the paired trace.
    targets = {turn.request.id: turn.output_tokens for root in trace for turn in root.turns}
    mismatches = [turn.id for turn in turns if turn.outcome == "success"
                  and turn.output_tokens != targets[turn.id]]
    apply_entitlement(roots, turns, settings["workers"])
    report = {
        **metadata, "mode": args.mode, "router_capacity": args.capacity, "workers": before["workers"],
        "time_scale": args.time_scale, "root_timeout_s": args.root_timeout,
        "request_timeout_s": args.request_timeout, "max_inflight": args.max_inflight,
        "solo_cost_sha256": digest(costs), "output_length_mismatches": mismatches,
        "router_settings": settings, "router_before": before, "router_after": after, **summarize(roots, turns),
    }
    report["successful_turn_router_wait_s"] = percentiles(
        turn.router_wait_s for turn in turns if turn.outcome == "success")
    write_rows(args.out / "requests.csv", [asdict(turn) for turn in turns])
    write_rows(args.out / "roots.csv", roots)
    write_json(args.out / "summary.json", report)
    print(encoded(report).decode(), end="")


def compare(paths, out):
    require_equal = ("trace_sha256", "engine_manifest_sha256", "scenario", "seed", "time_scale",
                     "solo_cost_sha256", "router_capacity", "workers", "root_timeout_s", "request_timeout_s",
                     "max_inflight", "cache_policy", "warmup_roots", "router_settings")
    reports = {path: json.loads((path / "summary.json").read_text()) for path in paths}
    summaries = {report["mode"]: report for report in reports.values()}
    if set(summaries) != set(MODES) or len(reports) != len(MODES):
        raise ValueError("compare requires one bypass, one FIFO and one fair output directory")
    reference = summaries["fifo"]
    for key in require_equal:
        if any(report[key] != reference[key] for report in summaries.values()):
            raise ValueError(f"experiment arms differ at {key}")
    for mode, report in summaries.items():
        router_settings(report["router_settings"])
        if report["output_length_mismatches"]:
            raise ValueError(f"{mode} arm has output length mismatches, which change the work being measured")
    roots = {}
    for path, report in reports.items():
        with (path / "roots.csv").open(newline="") as source:
            roots[report["mode"]] = {row["id"]: row for row in csv.DictReader(source)}
        if len(roots[report["mode"]]) != report["assigned_roots"]:
            raise ValueError(f"root count differs from summary for {path}")
    if any(rows.keys() != roots["fifo"].keys() for rows in roots.values()):
        raise ValueError("arms do not have identical assigned root IDs")

    def paired(modes):
        shared = {id_ for id_ in roots["fifo"] if all(roots[mode][id_]["outcome"] == "success" for mode in modes)}
        metrics = {}
        for mode in modes:
            metrics[mode] = {key: percentiles(float(roots[mode][id_][key]) for id_ in shared)
                             for key in ("raw_slowdown", "normalized_slowdown")}
        return {"completed_roots": len(shared), "arms": metrics}

    result = {
        "scope": "Private GPU replay; completed-root tails require checking failures and queue exposure",
        "matched_inputs": {key: reference[key] for key in require_equal},
        "arms": summaries,
        "fifo_vs_fair_paired_completions": paired(("fifo", "fair")),
        "all_arms_paired_completions": paired(MODES),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "run-arm"):
        command = commands.add_parser(name)
        command.add_argument("--url", required=True, help="numeric loopback origin of private router")
        command.add_argument("--engine-url", help="numeric loopback origin of private engine for metrics/reset")
        command.add_argument("--reset-prefix-cache", action="store_true")
        command.add_argument("--warmup-roots", type=int, default=2)
        command.add_argument("--workloads", type=Path, required=True)
        command.add_argument("--scenario", required=True)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--engine-manifest", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--capacity", type=int, default=128)
        command.add_argument("--request-timeout", type=positive, default=180)
        if name == "run-arm":
            command.add_argument("--calibration", type=Path, required=True)
            command.add_argument("--mode", choices=MODES, required=True)
            command.add_argument("--time-scale", type=positive, default=1.0)
            command.add_argument("--root-timeout", type=positive, default=300)
            command.add_argument("--max-inflight", type=int, default=256)
    compare_command = commands.add_parser("compare")
    compare_command.add_argument("--out", type=Path, required=True)
    compare_command.add_argument("arms", type=Path, nargs=3)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            compare(args.arms, args.out)
            return
        local_endpoint(args.url)
        if args.engine_url:
            local_endpoint(args.engine_url)
        if args.reset_prefix_cache and not args.engine_url:
            raise ValueError("cache reset requires --engine-url pointing to the private replica")
        scenario = load_scenarios(args.workloads)[args.scenario]
        trace = make_trace(scenario, args.seed)
        if not 1 <= sum(len(root.turns) for root in trace) <= MAX_TURNS:
            raise ValueError(f"replay needs between 1 and {MAX_TURNS} turns")
        if not 0 <= args.warmup_roots <= len(trace) or not 1 <= args.capacity <= MAX_TURNS:
            raise ValueError("warmup roots or router capacity is outside the supported range")
        if args.command == "run-arm" and (not 1 <= args.max_inflight <= MAX_TURNS
                                           or not math.isfinite(scenario.horizon * args.time_scale)):
            raise ValueError("invalid scaled workload horizon or max-inflight")
        manifest = json.loads(args.engine_manifest.read_text())
        metadata = {
            "trace_sha256": trace_fingerprint(trace), "scenario": args.scenario, "seed": args.seed,
            "engine_manifest_sha256": digest(manifest), "warmup_roots": args.warmup_roots,
            "cache_policy": "reset_then_salted_roots" if args.reset_prefix_cache else "salted_roots",
        }
        asyncio.run((calibrate if args.command == "calibrate" else run_arm)(args, trace, metadata))
    except (ValueError, KeyError, OSError, TimeoutError, aiohttp.ClientError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
