"""Policy and accounting checks driven by explicit event times, not elapsed time."""

import math
import unittest

from scheduler import Request, Scheduler, Rejected


def req(key, work=1.0, root=None, agent=None, size=10):
    return Request(key, root or key, agent or key, work, size)


class SchedulerTest(unittest.TestCase):
    def test_bypass_refuses_while_fifo_queues_at_the_identical_hard_cap(self):
        for mode in ("bypass", "fifo", "fair"):
            with self.subTest(mode=mode):
                s = Scheduler(mode, capacity=1)
                self.assertEqual(s.submit(req("running"), 0), "running")
                if mode == "bypass":
                    with self.assertRaises(Rejected) as failure:
                        s.submit(req("next"), 0)
                    self.assertEqual(failure.exception.code, "at_capacity")
                    self.assertEqual(s.stats["rejected"], 1)
                else:
                    self.assertEqual(s.submit(req("next"), 0), "queued")
                    events = s.finish("running", 1, 1)
                    self.assertEqual([(e.id, e.status) for e in events], [("next", "admitted")])
                self.assertEqual(len(s.running), 1)

    def test_fair_orders_by_workflow_deficit_instead_of_arrival(self):
        winners = {}
        for mode in ("fifo", "fair"):
            s = Scheduler(mode, capacity=1)
            s.submit(req("first", 10, root="returning"), 0)
            s.finish("first", 10, 10)
            s.submit(req("second", 10, root="returning"), 10)
            s.submit(req("large", 100), 10)
            s.submit(req("small", 1), 10)
            self.assertAlmostEqual(s.score("returning", 20), 4 / 3)
            self.assertAlmostEqual(s.score("small", 20), 10 / 3)
            winners[mode] = s.finish("second", 20, 10)[0].id
        self.assertEqual(winners, {"fifo": "large", "fair": "small"})

    def test_root_share_is_split_between_parallel_children(self):
        s = Scheduler("fair", capacity=3)
        s.submit(req("a", root="family", agent="child-a"), 0)
        s.submit(req("b", root="family", agent="child-b"), 0)
        s.submit(req("c", root="other"), 0)
        s.advance(4)
        self.assertEqual(s.roots["family"].entitlement, 2)
        self.assertEqual(s.roots["other"].entitlement, 2)
        s.finish("a", 4, 1)
        s.finish("b", 4, 1)
        self.assertEqual(s.score("family", 4), 1)

    def test_tool_gaps_do_not_accumulate_service_entitlement(self):
        s = Scheduler("fair", capacity=1)
        s.submit(req("turn-1", root="root"), 0)
        s.finish("turn-1", 1, 1)
        self.assertEqual(s.score("root", 100), 1)
        s.submit(req("turn-2", root="root"), 100)
        s.finish("turn-2", 101, 1)
        self.assertEqual(s.roots["root"].work, 2)
        self.assertEqual(s.roots["root"].entitlement, 2)

    def test_expiration_at_exact_deadline_precedes_admission(self):
        s = Scheduler("fifo", capacity=1, queue_timeout=3)
        s.submit(req("a"), 0)
        s.submit(req("b"), 0)
        events = s.finish("a", 3, 1)
        self.assertEqual([(e.id, e.status) for e in events], [("b", "expired")])
        self.assertEqual((s.pending_bytes, len(s.running), len(s.pending)), (0, 0, 0))
        self.assertEqual(s.stats["expired"], 1)

    def test_cancellation_releases_exactly_one_slot_and_is_idempotent(self):
        s = Scheduler("fair", capacity=1)
        s.submit(req("a"), 0)
        s.submit(req("b"), 0)
        s.submit(req("c"), 0)
        self.assertEqual([e.id for e in s.cancel("a", 1)], ["b"])
        self.assertEqual(s.cancel("a", 1), [])
        self.assertEqual(s.finish("a", 1, 1), [])
        self.assertEqual(s.stats["canceled"], 1)
        self.assertEqual(s.roots["a"].work, 0)
        s.cancel("c", 1)
        self.assertEqual(s.pending_bytes, 0)
        self.assertEqual(set(s.running), {"b"})

    def test_each_queue_bound_rejects_without_leaking_bytes_or_roots(self):
        cases = [
            ({"max_pending": 1}, req("c"), "queue_full"),
            ({"max_pending_bytes": 15}, req("c"), "queue_bytes_full"),
            ({"max_pending_per_root": 1}, req("c", root="b"), "root_queue_full"),
        ]
        for options, candidate, reason in cases:
            with self.subTest(reason=reason):
                s = Scheduler("fifo", 1, **options)
                s.submit(req("a"), 0)
                s.submit(req("b"), 0)
                with self.assertRaises(Rejected) as failure:
                    s.submit(candidate, 0)
                self.assertEqual(failure.exception.code, reason)
                self.assertEqual(s.pending_bytes, 10)
                self.assertEqual(len(s.roots), 2)

    def test_idle_history_is_bounded_and_active_roots_never_evicted(self):
        s = Scheduler("fair", 1, max_roots=1, root_ttl=5)
        s.submit(req("a"), 0)
        with self.assertRaises(Rejected) as failure:
            s.submit(req("b"), 10)
        self.assertEqual(failure.exception.code, "root_limit")
        s.finish("a", 10, 1)
        with self.assertRaises(Rejected):
            s.submit(req("b"), 14)
        s.submit(req("b"), 15)
        self.assertEqual(set(s.roots), {"b"})

    def test_duplicate_agent_cannot_create_extra_parallel_entitlement(self):
        s = Scheduler("fair", 2)
        s.submit(req("a", root="r", agent="child"), 0)
        with self.assertRaises(Rejected) as failure:
            s.submit(req("b", root="r", agent="child"), 0)
        self.assertEqual(failure.exception.code, "agent_busy")
        self.assertEqual(s.stats["submitted"], 2)
        self.assertEqual(len(s.running), 1)

    def test_equal_scores_keep_fifo_order(self):
        s = Scheduler("fair", 1)
        for key in ("a", "b", "c"):
            s.submit(req(key), 0)
        self.assertEqual(s.finish("a", 1, 1)[0].id, "b")

    def test_oversized_root_or_nonfinite_work_and_time_fail_loud(self):
        for work in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                req("a", work)
        with self.assertRaises(ValueError):
            req("a", root="r" * 257)
        s = Scheduler("fifo", 1)
        s.submit(req("a"), 1)
        for now in (0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                s.advance(now)
        self.assertEqual(s.snapshot(1)["running"], 1)


if __name__ == "__main__":
    unittest.main()
