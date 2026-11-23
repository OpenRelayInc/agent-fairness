import csv
import json
import tempfile
import unittest
from pathlib import Path

from scheduler import Request
from simulate import (
    MODES, Scenario, Turn, Workflow, compare, fair_entitlement,
    load_scenarios, make_trace, record_arm, trace_fingerprint,
)


def scenario(**changes):
    return Scenario(**({"name": "fixture", "roots": 2, "capacity": 1, "workers": 1,
                        "roots_per_burst": 2, "burst_period": 10, "spacing": 0,
                        "queue_timeout": 10, "root_deadline": 10, "horizon": 20} | changes))


def turn(root, label, *, agent="leader", after=(), gap=0, cost=1, cancel_after=None):
    request = Request(f"{root}/{label}", root, f"{root}/{agent}", cost, 100)
    return Turn(request, tuple(f"{root}/{parent}" for parent in after), gap, 5, 10, cancel_after)


class CompareTest(unittest.TestCase):
    def test_two_roots_match_hand_calculated_processor_sharing_reference(self):
        trace = tuple(Workflow(root, 0, (turn(root, "turn"),)) for root in ("first", "second"))
        stats, _, rows = record_arm(scenario(), trace, "fifo")
        self.assertEqual([row["normalized_slowdown"] for row in rows], [0.5, 1.5])
        self.assertEqual([row["raw_slowdown"] for row in rows], [1, 2])
        self.assertEqual(stats["p99_normalized_slowdown_completed_roots"], 1.5)
        self.assertEqual(stats["p99_wait_s_completed_requests"], 1)

    def test_agent_fanout_does_not_increase_root_share(self):
        intervals = [("root-a", "leader", 0, 4), ("root-a", "child", 1, 3), ("root-b", "leader", 1, 4)]
        # The first second has one active agent, capped at one worker. Once
        # root-b arrives, root-a receives one worker whether it has 1 or 2 agents.
        self.assertEqual(fair_entitlement(intervals, 2), {"root-a": 4, "root-b": 3})

    def test_fork_join_delays_dependents_and_excludes_tool_gaps(self):
        trace = (Workflow("root", 0, (
            turn("root", "initial"),
            turn("root", "one", agent="child-a", after=("initial",), gap=3),
            turn("root", "two", agent="child-b", after=("initial",), gap=3),
            turn("root", "join", after=("one", "two"), gap=5),
        )),)
        stats, requests, roots = record_arm(scenario(capacity=2), trace, "fair")
        self.assertEqual([row["submitted_s"] for row in requests], [0, 4, 4, 11])
        self.assertEqual([row["ended_s"] for row in requests], [1, 6, 6, 12])
        self.assertEqual(roots[0]["elapsed_s"], 12)
        self.assertEqual(roots[0]["outstanding_s"], 6)
        self.assertEqual(roots[0]["raw_slowdown"], 1.5)
        self.assertEqual(roots[0]["normalized_slowdown"], 1)
        self.assertEqual(stats["completed_by_deadline"], 0)

    def test_baseline_rejects_overflow_and_denominator_keeps_rejected_root(self):
        trace = (Workflow("a", 0, (turn("a", "only"),)),
                 Workflow("b", 0, (turn("b", "initial"), turn("b", "next", after=("initial",)))))
        stats, requests, roots = record_arm(scenario(), trace, "bypass")
        self.assertEqual(stats["assigned_roots"], 2)
        self.assertEqual(stats["root_outcomes"]["completed"], 1)
        self.assertEqual(stats["root_outcomes"]["rejected"], 1)
        self.assertEqual(stats["completed_by_deadline"], 1)
        self.assertEqual([row["status"] for row in requests], ["completed", "rejected", "blocked"])
        self.assertEqual(roots[1]["solo_work_s"], 2)
        self.assertIsNone(roots[1]["normalized_slowdown"])

    def test_expiration_and_cancellation_do_not_release_dependent_turns(self):
        trace = (Workflow("a", 0, (turn("a", "only", cost=5, cancel_after=2),)),
                 Workflow("b", 0, (turn("b", "initial"), turn("b", "next", after=("initial",)))),
                 Workflow("c", 2, (turn("c", "only"),)))
        stats, requests, _ = record_arm(scenario(queue_timeout=1), trace, "fair")
        self.assertEqual([row["status"] for row in requests], ["canceled", "expired", "blocked", "completed"])
        self.assertEqual([row["ended_s"] for row in requests], [2, 1, 1, 3])
        self.assertEqual(stats["root_outcomes"], {"completed": 1, "rejected": 0, "canceled": 1,
                                                  "expired": 1, "incomplete": 0})

    def test_horizon_keeps_unfinished_workflow_and_its_full_solo_cost(self):
        trace = (Workflow("a", 0, (turn("a", "initial", cost=21), turn("a", "next", after=("initial",)))),)
        stats, requests, roots = record_arm(scenario(), trace, "fifo")
        self.assertEqual(stats["root_outcomes"]["incomplete"], 1)
        self.assertEqual(stats["simulated_elapsed_s"], 20)
        self.assertIsNone(stats["p99_raw_slowdown_completed_roots"])
        self.assertEqual(roots[0]["solo_work_s"], 22)
        self.assertEqual(roots[0]["outstanding_s"], 20)
        self.assertEqual([row["status"] for row in requests], ["incomplete", "incomplete"])

    def test_light_workload_has_identical_arms_and_reproducible_artifacts(self):
        light = load_scenarios()["light"]
        trace = make_trace(light, 42)
        self.assertEqual(trace_fingerprint(trace), trace_fingerprint(make_trace(light, 42)))
        self.assertNotEqual(trace_fingerprint(trace), trace_fingerprint(make_trace(light, 43)))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = compare([light], 42, out)
            self.assertEqual(result, json.loads((out / "summary.json").read_text()))
            arms = result["scenarios"][light.name]["arms"]
            self.assertEqual(arms["bypass"]["root_outcomes"]["completed"], light.roots)
            self.assertEqual(arms["fifo"], arms["bypass"])
            self.assertEqual(arms["fair"], arms["bypass"])
            for mode in MODES:
                with (out / light.name / mode / "roots.csv").open() as src:
                    self.assertEqual(len(list(csv.DictReader(src))), light.roots)

    def test_workload_name_cannot_escape_artifact_directory(self):
        with self.assertRaisesRegex(ValueError, "scenario names"):
            scenario(name="../other")


if __name__ == "__main__":
    unittest.main()
