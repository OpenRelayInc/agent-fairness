import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from aiohttp import web

from gpu_replay import cache_namespace, calibrate, compare, run_arm
from replay import prepare_bodies
from router import CHAT, MODES, RouterConfig, create_app
from simulate import Workflow
from test_replay import COMPLETE, turn
from test_router import serve


class GPUReplayTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.trace = (Workflow("r", 0, (turn("start"), turn("next", ("start",)))),)

    def test_namespaces_do_not_change_payload_tokens_or_same_root_cache_reuse(self):
        original = prepare_bodies(self.trace)
        first = cache_namespace(self.trace, original.copy(), "loop-1")
        second = cache_namespace(self.trace, original.copy(), "loop-2")
        decoded_first = {id_: json.loads(body) for id_, body in first.items()}
        decoded_second = {id_: json.loads(body) for id_, body in second.items()}
        self.assertEqual(decoded_first["start"]["cache_salt"], decoded_first["next"]["cache_salt"])
        self.assertNotEqual(decoded_first["start"]["cache_salt"], decoded_second["start"]["cache_salt"])
        for id_, payload in decoded_first.items():
            payload.pop("cache_salt")
            self.assertEqual(payload, json.loads(original[id_]))

    async def test_three_modes_reuse_frozen_calibration_and_keep_all_root_metrics(self):
        seen = []

        async def engine(request):
            seen.append(await request.json())
            return web.Response(body=COMPLETE, content_type="text/event-stream")

        app = web.Application()
        app.router.add_post(CHAT, engine)
        metadata = dict(trace_sha256="trace", scenario="tiny", seed=42, engine_manifest_sha256="engine",
                        warmup_roots=0, cache_policy="salted_roots")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(tmp)
            calibration = directory / "calibration"
            common = dict(engine_url=None, reset_prefix_cache=False, capacity=2, warmup_roots=0, request_timeout=5)
            arm_dirs = []
            async with serve(app) as upstream:
                for mode in MODES:
                    async with serve(create_app(RouterConfig(upstream=upstream, mode=mode, capacity=2))) as url:
                        if mode == "bypass":
                            cal_args = SimpleNamespace(url=url, out=calibration, **common)
                            await calibrate(cal_args, self.trace, metadata)
                            self.assertEqual(len(seen), 2)
                            await calibrate(cal_args, self.trace, metadata)
                            self.assertEqual(len(seen), 2)
                            with self.assertRaisesRegex(ValueError, "saved calibration"):
                                await calibrate(cal_args, self.trace, dict(metadata, scenario="other"))
                        arm_dir = directory / mode
                        arm_dirs.append(arm_dir)
                        await run_arm(SimpleNamespace(url=url, out=arm_dir, calibration=calibration, mode=mode,
                                                      time_scale=1, root_timeout=10, max_inflight=4, **common),
                                      self.trace, metadata)
            self.assertEqual(len(seen), 8)
            salts = [payload["cache_salt"] for payload in seen]
            self.assertEqual(salts[::2], salts[1::2])
            self.assertEqual(len(set(salts)), 4)
            report = directory / "result.json"
            compare(arm_dirs, report)
            summary = json.loads(report.read_text())
            self.assertEqual(summary["fifo_vs_fair_paired_completions"]["completed_roots"], 1)
            self.assertEqual(summary["all_arms_paired_completions"]["completed_roots"], 1)
            self.assertTrue(all(arm["root_outcomes"] == {"success": 1} for arm in summary["arms"].values()))
            self.assertTrue(all(arm["router_after"]["stats"]["completed"]
                                - arm["router_before"]["stats"].get("completed", 0) == 2
                                for arm in summary["arms"].values()))
            path = arm_dirs[-1] / "summary.json"
            original = json.loads(path.read_text())
            self.assertEqual(original["router_settings"]["max_pending_bytes"], 16 << 20)
            self.assertEqual(original["router_settings"]["queue_timeout"], 30)
            for mutation, message in (
                ({"time_scale": 2}, "time_scale"),
                ({"router_settings": dict(original["router_settings"], queue_timeout=31)}, "router_settings"),
                ({"output_length_mismatches": ["start"]}, "output length mismatches"),
            ):
                path.write_text(json.dumps({**original, **mutation}))
                with self.assertRaisesRegex(ValueError, message):
                    compare(arm_dirs, report)


if __name__ == "__main__":
    unittest.main()
