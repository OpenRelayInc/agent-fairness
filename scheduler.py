"""Single-owner admission policies, shared by the HTTP prototype and virtual-clock replay.

All mutations take an explicit monotonic timestamp. Calls do no I/O and never run
background tasks. The owner delivers returned events to queued request waiters.
"""

from collections import Counter
from dataclasses import dataclass
import math


class Rejected(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Request:
    id: str
    root: str
    agent: str
    solo_work: float
    body_bytes: int

    def __post_init__(self):
        for value in (self.id, self.root, self.agent):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("request, root and agent IDs must contain 1 to 256 characters")
        if not math.isfinite(self.solo_work) or self.solo_work <= 0:
            raise ValueError("solo_work must be finite and positive")
        if type(self.body_bytes) is not int or self.body_bytes < 0:
            raise ValueError("body_bytes must be a nonnegative integer")


@dataclass(frozen=True)
class Event:
    id: str
    status: str


@dataclass
class RootState:
    entitlement: float = 0.0
    work: float = 0.0
    last_seen: float = 0.0


class Scheduler:
    """A bounded, non-preemptive router, not a runtime/GPU scheduler.

    ``bypass`` approximates today's sole-pool admission: refuse at the hard cap.
    ``fifo`` and ``fair`` add the same bounded queue. Fair orders by accumulated
    ideal processor-sharing entitlement divided by completed estimated solo work.
    Before a root completes any work, its cheapest outstanding turn bootstraps the
    denominator. This turn-boundary estimate is not Duru's per-iteration signal.

    An agent may have one outstanding request; parallel children need distinct
    agent IDs under the same root. Tool gaps have no outstanding request and
    accrue no entitlement. Root history expires only when idle.
    """

    def __init__(self, mode, capacity, workers=1, max_pending=256,
                 max_pending_bytes=16 * 1024 * 1024, max_pending_per_root=32,
                 max_roots=4096, queue_timeout=30.0, root_ttl=600.0):
        if mode not in ("bypass", "fifo", "fair"):
            raise ValueError("mode must be bypass, fifo or fair")
        for name, value in (("capacity", capacity), ("workers", workers),
                            ("max_pending", max_pending), ("max_pending_bytes", max_pending_bytes),
                            ("max_pending_per_root", max_pending_per_root), ("max_roots", max_roots)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("queue_timeout", queue_timeout), ("root_ttl", root_ttl)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.mode, self.capacity, self.workers = mode, capacity, workers
        self.max_pending, self.max_pending_bytes = max_pending, max_pending_bytes
        self.max_pending_per_root, self.max_roots = max_pending_per_root, max_roots
        self.queue_timeout, self.root_ttl = queue_timeout, root_ttl
        self.running = {}
        self.pending = {}
        self.roots = {}
        self.stats = Counter()
        self.pending_bytes = 0
        self._enqueued = {}
        self._now = None

    def _tick(self, now):
        if not math.isfinite(now) or (self._now is not None and now < self._now):
            raise ValueError("time must be finite and monotonic")
        elapsed = 0 if self._now is None else now - self._now
        counts = Counter(r.root for r in (*self.running.values(), *self.pending.values()))
        for root, agents in counts.items():
            # Equal share between root workflows, then their outstanding agents;
            # each agent is capped at one worker's standalone processing rate.
            share = min(1.0, self.workers / len(counts) / agents)
            self.roots[root].entitlement += elapsed * share * agents
        self._now = now

    def _reject(self, code):
        self.stats["rejected"] += 1
        self.stats["rejected_" + code] += 1
        raise Rejected(code)

    def submit(self, request, now):
        """Return running/queued. Call advance after submitting to deliver any admissions."""
        self._tick(now)
        self.stats["submitted"] += 1
        outstanding = (*self.running.values(), *self.pending.values())
        if any(r.id == request.id for r in outstanding):
            self._reject("duplicate_request")
        if any((r.root, r.agent) == (request.root, request.agent) for r in outstanding):
            self._reject("agent_busy")
        active_roots = {r.root for r in outstanding}
        self.roots = {key: state for key, state in self.roots.items()
                      if key in active_roots or now < state.last_seen + self.root_ttl}
        if request.root not in self.roots and len(self.roots) >= self.max_roots:
            self._reject("root_limit")
        can_run = len(self.running) < self.capacity and not self.pending
        if not can_run:
            if self.mode == "bypass":
                self._reject("at_capacity")
            if len(self.pending) >= self.max_pending:
                self._reject("queue_full")
            if self.pending_bytes + request.body_bytes > self.max_pending_bytes:
                self._reject("queue_bytes_full")
            if sum(r.root == request.root for r in self.pending.values()) >= self.max_pending_per_root:
                self._reject("root_queue_full")
        self.roots.setdefault(request.root, RootState()).last_seen = now
        if can_run:
            self.running[request.id] = request
            self.stats["admitted"] += 1
            self.stats["peak_running"] = max(self.stats["peak_running"], len(self.running))
            return "running"
        self.pending[request.id] = request
        self._enqueued[request.id] = now
        self.pending_bytes += request.body_bytes
        self.stats["queued"] += 1
        self.stats["peak_pending"] = max(self.stats["peak_pending"], len(self.pending))
        return "queued"

    def _score(self, root):
        state = self.roots[root]
        denominator = state.work
        if denominator == 0:
            denominator = min((r.solo_work for r in (*self.running.values(), *self.pending.values())
                               if r.root == root), default=1.0)
        return state.entitlement / denominator

    def score(self, root, now):
        self._tick(now)
        return self._score(root)

    def _pop_pending(self, request_id):
        request = self.pending.pop(request_id)
        self.pending_bytes -= request.body_bytes
        del self._enqueued[request_id]
        return request

    def _drain(self, now):
        events = []
        for request_id in list(self.pending):
            if now >= self._enqueued[request_id] + self.queue_timeout:
                request = self._pop_pending(request_id)
                self.roots[request.root].last_seen = now
                self.stats["expired"] += 1
                events.append(Event(request_id, "expired"))
        while self.pending and len(self.running) < self.capacity:
            # dict insertion order breaks ties by arrival, identically in both queues.
            request_id = (max(self.pending, key=lambda key: self._score(self.pending[key].root))
                          if self.mode == "fair" else next(iter(self.pending)))
            request = self._pop_pending(request_id)
            self.running[request_id] = request
            self.stats["admitted"] += 1
            self.stats["peak_running"] = max(self.stats["peak_running"], len(self.running))
            events.append(Event(request_id, "admitted"))
        return events

    def advance(self, now):
        self._tick(now)
        return self._drain(now)

    def finish(self, request_id, now, completed_work=0.0):
        if not math.isfinite(completed_work) or completed_work < 0:
            raise ValueError("completed_work must be finite and nonnegative")
        self._tick(now)
        request = self.running.pop(request_id, None)
        if request is not None:
            state = self.roots[request.root]
            state.work += completed_work
            state.last_seen = now
            self.stats["completed" if completed_work > 0 else "failed"] += 1
        return self._drain(now)

    def cancel(self, request_id, now):
        self._tick(now)
        request = self.running.pop(request_id, None)
        if request_id in self.pending:
            request = self._pop_pending(request_id)
        if request is not None:
            self.roots[request.root].last_seen = now
            self.stats["canceled"] += 1
        return self._drain(now)

    def snapshot(self, now):
        self._tick(now)
        return {"mode": self.mode, "capacity": self.capacity, "workers": self.workers,
                "max_pending": self.max_pending, "max_pending_bytes": self.max_pending_bytes,
                "max_pending_per_root": self.max_pending_per_root, "max_roots": self.max_roots,
                "queue_timeout": self.queue_timeout, "root_ttl": self.root_ttl,
                "running": len(self.running), "pending": len(self.pending),
                "pending_bytes": self.pending_bytes, "roots": len(self.roots),
                "stats": dict(self.stats)}
