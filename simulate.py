#!/usr/bin/env python3
"""Compare three admission policies on seeded causal workflows and a synthetic engine."""

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from scheduler import Rejected, Request, Scheduler


MODEL = "openrelay/deepseek-v4.1-flash-abliterated"
MODES = ("bypass", "fifo", "fair")
DEFAULT_WORKLOADS = Path(__file__).with_name("workloads.json")
ROOT_OUTCOMES = ("completed", "rejected", "canceled", "expired", "incomplete")


@dataclass(frozen=True)
class Scenario:
    name: str
    roots: int
    capacity: int
    workers: int
    roots_per_burst: int
    burst_period: float
    spacing: float
    queue_timeout: float
    root_deadline: float
    horizon: float
    cancel_fraction: float = 0
    cancel_after: float = 5

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.name):
            raise ValueError("scenario names must be lowercase alphanumeric slugs")
        for field in ("roots", "capacity", "workers", "roots_per_burst"):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= 4096:
                raise ValueError(f"{field} must be an integer between 1 and 4096")
        for field in ("burst_period", "queue_timeout", "root_deadline", "horizon", "cancel_after"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if not math.isfinite(self.spacing) or not 0 <= self.spacing * self.roots_per_burst <= self.burst_period:
            raise ValueError("spacing must be nonnegative and fit within the burst period")
        if self.workers > self.capacity or not 0 <= self.cancel_fraction <= 1:
            raise ValueError("workers must fit within capacity and cancel_fraction must be between 0 and 1")
        if not self.arrival(self.roots - 1) + self.root_deadline <= self.horizon <= 86_400:
            raise ValueError("horizon must cover the deadline of every offered root and be at most one day")

    def arrival(self, index):
        burst, offset = divmod(index, self.roots_per_burst)
        return burst * self.burst_period + offset * self.spacing


@dataclass(frozen=True)
class Turn:
    request: Request
    after: tuple[str, ...]
    tool_gap: float
    input_tokens: int
    output_tokens: int
    cancel_after: float | None = None


@dataclass(frozen=True)
class Workflow:
    id: str
    arrival: float
    turns: tuple[Turn, ...]


def load_scenarios(path=DEFAULT_WORKLOADS):
    with open(path) as src:
        return {name: Scenario(name=name, **values) for name, values in json.load(src).items()}


def make_trace(scenario, seed):
    """Build a paired trace with fixed work and open-loop root starts.

    Each fork has two child agents followed by a join. New input includes
    synthetic tool output. Costs are illustrative, with no GPU calibration.
    """
    rng = random.Random(seed)
    workflows = []
    for index in range(scenario.roots):
        root = f"root-{index:04d}"
        long_context = rng.random() < 0.30
        fanout = rng.random() < 0.45
        canceled = rng.random() < scenario.cancel_fraction

        def turn(label, after=(), *, agent="leader", first=False, cancel_after=None):
            input_tokens = rng.randint(40_000, 120_000) if first and long_context else rng.randint(128, 8_192)
            output_tokens = rng.randint(45, 540) if long_context else rng.randint(15, 190)
            solo_work = 0.08 + input_tokens / 80_000 + output_tokens / 180
            request_id = f"{root}/{label}"
            return Turn(
                Request(request_id, root, f"{root}/{agent}", solo_work, input_tokens * 4 + 256),
                tuple(f"{root}/{parent}" for parent in after),
                0 if first else rng.uniform(0.025, 0.35),
                input_tokens, output_tokens, cancel_after,
            )

        turns = [turn("start", first=True, cancel_after=scenario.cancel_after if canceled else None)]
        if fanout:
            turns.extend((
                turn("child-a", ("start",), agent="a"),
                turn("child-b", ("start",), agent="b"),
                turn("join", ("child-a", "child-b")),
            ))
        else:
            turns.append(turn("continue", ("start",)))
        turns.append(turn("end", ("join" if fanout else "continue",)))
        workflows.append(Workflow(root, scenario.arrival(index), tuple(turns)))
    return tuple(workflows)


def trace_fingerprint(trace):
    return hashlib.sha256(json.dumps([asdict(root) for root in trace], sort_keys=True).encode()).hexdigest()


@dataclass
class Attempt:
    turn: Turn
    status: str = "unreleased"
    submitted: float | None = None
    admitted: float | None = None
    ended: float | None = None
    reason: str = ""

    def row(self):
        request = self.turn.request
        return {
            "id": request.id, "root": request.root, "agent": request.agent,
            "after": " ".join(self.turn.after), "tool_gap_s": self.turn.tool_gap,
            "solo_work_s": request.solo_work, "body_bytes": request.body_bytes,
            "target_input_tokens": self.turn.input_tokens, "target_output_tokens": self.turn.output_tokens,
            "status": self.status, "reason": self.reason, "submitted_s": self.submitted,
            "admitted_s": self.admitted, "ended_s": self.ended,
            "wait_s": None if self.admitted is None else self.admitted - self.submitted,
            "outstanding_s": None if self.submitted is None else self.ended - self.submitted,
        }


class Simulation:
    """Use the router's scheduler with an ideal processor-sharing engine.

    The event order at a time is expiry, cancellation, completion, then
    causal releases. IDs break ties. Unfinished requests end at the horizon
    for accounting, so they remain visible in the assigned root population.
    """

    def __init__(self, scenario, trace, mode):
        self.scenario = scenario
        self.scheduler = Scheduler(mode, scenario.capacity, workers=scenario.workers,
                                   queue_timeout=scenario.queue_timeout,
                                   root_ttl=max(600, scenario.horizon))
        self.now = 0.0
        self.attempts = {}
        self.dependents = defaultdict(list)
        self.releases = []
        self.cancels = []
        self.remaining = {}
        for root in trace:
            for turn in root.turns:
                request_id = turn.request.id
                self.attempts[request_id] = Attempt(turn)
                if not turn.after:
                    heapq.heappush(self.releases, (root.arrival + turn.tool_gap, request_id))
                for dependency in turn.after:
                    self.dependents[dependency].append(request_id)

    def start(self, request_id):
        attempt = self.attempts[request_id]
        attempt.status, attempt.admitted = "running", self.now
        self.remaining[request_id] = attempt.turn.request.solo_work

    def terminal(self, request_id, status, reason=""):
        attempt = self.attempts[request_id]
        attempt.status, attempt.ended, attempt.reason = status, self.now, reason
        for child in self.dependents[request_id]:
            dependent = self.attempts[child]
            if dependent.status != "unreleased":
                continue
            if status != "completed":
                self.terminal(child, "blocked", "dependency_failed")
            elif all(self.attempts[parent].status == "completed" for parent in dependent.turn.after):
                heapq.heappush(self.releases, (self.now + dependent.turn.tool_gap, child))

    def apply(self, events):
        for event in events:
            if event.status == "admitted":
                self.start(event.id)
            elif event.status == "expired":
                self.terminal(event.id, "expired", "queue_timeout")
            else:
                raise ValueError(f"unexpected scheduler event {event.status}")

    def offer(self, request_id):
        attempt = self.attempts[request_id]
        attempt.submitted = self.now
        try:
            status = self.scheduler.submit(attempt.turn.request, self.now)
        except Rejected as refused:
            self.terminal(request_id, "rejected", refused.code)
            return
        if status == "running":
            self.start(request_id)
        else:
            attempt.status = "queued"
        if attempt.turn.cancel_after is not None:
            heapq.heappush(self.cancels, (self.now + attempt.turn.cancel_after, request_id))

    def next_event(self):
        while self.cancels and self.attempts[self.cancels[0][1]].status not in ("queued", "running"):
            heapq.heappop(self.cancels)
        times = []
        if self.releases:
            times.append(self.releases[0][0])
        if self.cancels:
            times.append(self.cancels[0][0])
        if self.remaining:
            times.append(self.now + min(self.remaining.values()) / self.service_rate())
        times.extend(self.attempts[request_id].submitted + self.scenario.queue_timeout
                     for request_id in self.scheduler.pending)
        return min(times, default=math.inf)

    def service_rate(self):
        return min(1.0, self.scenario.workers / len(self.remaining)) if self.remaining else 0.0

    def run(self):
        horizon = self.scenario.horizon
        while (at := self.next_event()) <= horizon:
            work = (at - self.now) * self.service_rate()
            for request_id in self.remaining:
                self.remaining[request_id] -= work
            self.now = at
            completed = sorted(key for key, left in self.remaining.items() if left <= 1e-9)
            self.apply(self.scheduler.advance(at))
            while self.cancels and self.cancels[0][0] <= at:
                _, request_id = heapq.heappop(self.cancels)
                if self.attempts[request_id].status in ("queued", "running"):
                    self.remaining.pop(request_id, None)
                    self.terminal(request_id, "canceled", "client_deadline")
                    self.apply(self.scheduler.cancel(request_id, at))
            for request_id in completed:
                if request_id in self.remaining:
                    del self.remaining[request_id]
                    self.terminal(request_id, "completed")
                    self.apply(self.scheduler.finish(request_id, at, self.attempts[request_id].turn.request.solo_work))
            while self.releases and self.releases[0][0] <= at:
                _, request_id = heapq.heappop(self.releases)
                self.offer(request_id)
        for attempt in self.attempts.values():
            if attempt.status in ("queued", "running", "unreleased"):
                attempt.status, attempt.ended, attempt.reason = "incomplete", horizon, "horizon"
        return list(self.attempts.values())


def fair_entitlement(intervals, workers):
    """Accumulate ideal share from (root, agent, submitted, ended) intervals."""
    events = []
    for root, agent, submitted, ended in intervals:
        if ended > submitted:
            events.extend(((submitted, 1, root, agent), (ended, -1, root, agent)))
    entitlement = defaultdict(float)
    active = defaultdict(Counter)
    last = 0.0
    for at, change, root, agent in sorted(events):
        elapsed = at - last
        if elapsed and active:
            share = workers / len(active)
            for key, agents in active.items():
                entitlement[key] += elapsed * min(share, len(agents))
        active[root][agent] += change
        if active[root][agent] == 0:
            del active[root][agent]
        if not active[root]:
            del active[root]
        last = at
    return entitlement


def root_rows(trace, attempts, scenario):
    by_id = {attempt.turn.request.id: attempt for attempt in attempts}
    intervals = ((attempt.turn.request.root, attempt.turn.request.agent, attempt.submitted, attempt.ended)
                 for attempt in attempts if attempt.submitted is not None)
    entitlement = fair_entitlement(intervals, scenario.workers)
    rows = []
    for workflow in trace:
        turns = [by_id[turn.request.id] for turn in workflow.turns]
        solo_work = sum(turn.request.solo_work for turn in workflow.turns)
        outstanding = sum(turn.ended - turn.submitted for turn in turns if turn.submitted is not None)
        failures = [turn for turn in turns if turn.status in ("rejected", "canceled", "expired")]
        completed = all(turn.status == "completed" for turn in turns)
        status = "completed" if completed else "incomplete"
        if failures:
            status = min(failures, key=lambda turn: (turn.ended, turn.turn.request.id)).status
        elapsed = max(turn.ended for turn in turns) - workflow.arrival
        rows.append({
            "root": workflow.id, "status": status, "turns": len(turns), "arrival_s": workflow.arrival,
            "elapsed_s": elapsed, "by_deadline": completed and elapsed <= scenario.root_deadline,
            "completed_turns": sum(turn.status == "completed" for turn in turns),
            "solo_work_s": solo_work, "outstanding_s": outstanding,
            "entitled_work_s": entitlement[workflow.id],
            "raw_slowdown": outstanding / solo_work if completed else None,
            "normalized_slowdown": entitlement[workflow.id] / solo_work if completed else None,
        })
    return rows


def p99(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.99 * len(ordered)) - 1] if ordered else None


def record_arm(scenario, trace, mode):
    attempts = Simulation(scenario, trace, mode).run()
    roots = root_rows(trace, attempts, scenario)
    requests = [attempt.row() for attempt in attempts]
    counts = Counter(row["status"] for row in roots)
    elapsed = max(row["arrival_s"] + row["elapsed_s"] for row in roots) - min(row["arrival_s"] for row in roots)
    successes = [row for row in requests if row["status"] == "completed"]
    completed_roots = [row for row in roots if row["status"] == "completed"]
    stats = {
        "assigned_roots": len(roots),
        "root_outcomes": {outcome: counts[outcome] for outcome in ROOT_OUTCOMES},
        "completed_by_deadline": sum(row["by_deadline"] for row in roots),
        "root_deadline_s": scenario.root_deadline,
        "simulated_elapsed_s": elapsed,
        "p99_raw_slowdown_completed_roots": p99(row["raw_slowdown"] for row in completed_roots),
        "p99_normalized_slowdown_completed_roots": p99(row["normalized_slowdown"] for row in completed_roots),
        "p99_elapsed_s_completed_roots": p99(row["elapsed_s"] for row in completed_roots),
        "p99_wait_s_completed_requests": p99(row["wait_s"] for row in successes),
        "p99_outstanding_s_completed_requests": p99(row["outstanding_s"] for row in successes),
        "completed_roots_per_s": len(completed_roots) / elapsed if elapsed else 0,
        "successful_output_tokens_per_s": sum(row["target_output_tokens"] for row in successes) / elapsed if elapsed else 0,
    }
    return stats, requests, roots


def write_rows(path, rows):
    with path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(scenarios, seed, out):
    out.mkdir(parents=True, exist_ok=True)
    summary = {"model": MODEL, "engine": "synthetic_processor_sharing", "seed": seed,
               "verdict": "synthetic_only_no_deployment_decision", "scenarios": {}}
    for scenario in scenarios:
        trace = make_trace(scenario, seed)
        result = {"config": asdict(scenario), "trace_sha256": trace_fingerprint(trace), "arms": {}}
        for mode in MODES:
            stats, requests, roots = record_arm(scenario, trace, mode)
            arm_dir = out / scenario.name / mode
            arm_dir.mkdir(parents=True, exist_ok=True)
            write_rows(arm_dir / "requests.csv", requests)
            write_rows(arm_dir / "roots.csv", roots)
            result["arms"][mode] = stats
        fifo, fair = result["arms"]["fifo"], result["arms"]["fair"]
        norm = "p99_normalized_slowdown_completed_roots"
        wait = "p99_wait_s_completed_requests"
        result["fair_vs_fifo"] = {
            "completed_roots_delta": fair["root_outcomes"]["completed"] - fifo["root_outcomes"]["completed"],
            "completed_by_deadline_delta": fair["completed_by_deadline"] - fifo["completed_by_deadline"],
            "p99_normalized_slowdown_delta": fair[norm] - fifo[norm] if None not in (fair[norm], fifo[norm]) else None,
            "p99_wait_s_delta": fair[wait] - fifo[wait] if None not in (fair[wait], fifo[wait]) else None,
        }
        summary["scenarios"][scenario.name] = result
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def print_summary(summary):
    print("Synthetic engine. Slowdown and wait percentiles cover successful completions only.")
    print(f"{'scenario':<11} {'mode':<7} {'done/all':>9} {'SLO':>5} {'reject':>6} {'cancel':>6} {'expire':>6} {'open':>5} "
          f"{'p99 S':>8} {'p99 U':>8} {'p99 wait':>9} {'out tok/s':>9}")
    for name, data in summary["scenarios"].items():
        for mode, arm in data["arms"].items():
            counts = arm["root_outcomes"]
            def fmt(key):
                value = arm[key]
                return f"{value:.2f}" if value is not None else "n/a"
            print(f"{name:<11} {mode:<7} {counts['completed']:>4}/{arm['assigned_roots']:<4} "
                  f"{arm['completed_by_deadline']:>5} {counts['rejected']:>6} {counts['canceled']:>6} "
                  f"{counts['expired']:>6} {counts['incomplete']:>5} "
                  f"{fmt('p99_raw_slowdown_completed_roots'):>8} {fmt('p99_normalized_slowdown_completed_roots'):>8} "
                  f"{fmt('p99_wait_s_completed_requests'):>9} {fmt('successful_output_tokens_per_s'):>9}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS)
    parser.add_argument("--scenario", action="append", help="choose a scenario, or repeat to select several")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        scenarios = load_scenarios(args.workloads)
        chosen = [scenarios[name] for name in args.scenario] if args.scenario else scenarios.values()
        print_summary(compare(chosen, args.seed, args.out))
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
