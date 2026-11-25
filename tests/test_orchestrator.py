"""Integration tests of controller decisions with mocked external boundaries.

Every image, manifest, and model response here is an explicit unit fixture.
No real model, browser, external process, or devset run is performed. All run
artifacts live in TemporaryDirectory and are deleted after each test.
"""

from copy import deepcopy
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxy_harness.engine import run_agent
from proxy_harness.orchestrator import load_config, run_task
from proxy_harness.util import read_json, write_json


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j"
    "A5sAAAAASUVORK5CYII="
)
_SESSION_ID = "unit-fixture-exact-session-7d812"


def _spec():
    return {
        "schema_version": "1.0",
        "task_id": "orchestrator_unit_fixture",
        "prompt": "A required sphere identified as a.",
        "duration_s": 2,
        "fps": 12,
        "width": 640,
        "height": 360,
        "sample_times": [0, 1],
        "subjects": [{"id": "a"}],
        "evaluation": {"pass_score": 75, "hard_fail_score_cap": 40},
        "split": "demo",
    }


def _config(iterations=2, evaluator=True):
    config = load_config()
    config["max_iterations"] = iterations
    config["coder"] = {"backend": "codex"}
    config["evaluator"] = {"backend": "command", "argv": ["mock-unit-fixture-no-process"]} if evaluator else {"backend": "none"}
    config["export_video"] = False
    return config


def _assets(root):
    assets = root / "fixture_assets"
    assets.mkdir()
    (assets / "runtime.js").write_text("// immutable unit-test runtime fixture\n", encoding="utf-8")
    (assets / "primitives.js").write_text("// immutable primitive fixture\n", encoding="utf-8")
    (assets / "AGENTS.md").write_text("Only edit scene.html and appearance_prompt.txt.\n", encoding="utf-8")
    (assets / "scene.html").write_text("<html>initial unit fixture</html>\n", encoding="utf-8")
    return assets


def _coder(*, exit_codes=(0, 0), mutate_protected=False):
    calls = []

    def run(config, prompt, workspace, trace_dir, *, session_id=None, images=()):
        index = len(calls)
        calls.append({"session_id": session_id, "workspace": Path(workspace), "images": list(images), "prompt": prompt})
        (workspace / "scene.html").write_text(
            f"<!doctype html><canvas id='scene'></canvas><!-- mocked candidate {index} -->\n",
            encoding="utf-8",
        )
        (workspace / "appearance_prompt.txt").write_text("a: the requested sphere; unit fixture only.\n", encoding="utf-8")
        if mutate_protected:
            (workspace / "runtime.js").write_text("// unauthorized unit-fixture mutation\n", encoding="utf-8")
        result = {
            "backend": "codex", "exit_code": exit_codes[min(index, len(exit_codes) - 1)],
            "session_id": _SESSION_ID, "evidence_mode": "mock_unit_fixture",
            "output": "Mocked response; no model was invoked.",
        }
        write_json(trace_dir / "coder_result.json", result)
        (trace_dir / "coder.stdout.jsonl").write_text("", encoding="utf-8")
        (trace_dir / "coder.stderr.log").write_text("UNIT FIXTURE: no model invoked.\n", encoding="utf-8")
        return result

    run.calls = calls
    return run


def _renderer(hard_passes=(True, True)):
    calls = []

    def render(scene_path, spec, output_dir, **kwargs):
        index = len(calls)
        calls.append({"source": Path(scene_path).read_text(encoding="utf-8"), "output_dir": Path(output_dir), "kwargs": kwargs})
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        hard_pass = hard_passes[min(index, len(hard_passes) - 1)]
        l1 = {
            "schema_version": "1.0", "hard_pass": hard_pass,
            "failures": [] if hard_pass else [{
                "code": "mock_fixture_geometry_failure",
                "message": "Synthetic failure used only to test controller decisions.",
                "repair": "Not a real scene diagnosis.",
            }],
            "warnings": [],
            "metrics": {"evidence_mode": "mock_unit_fixture", "primitive_count": 1},
        }
        write_json(output_dir / "l1_report.json", l1)
        write_json(output_dir / "manifests.json", [{"audit": {"primitive_count": 1}, "evidence_mode": "mock_unit_fixture"}])
        (output_dir / "contact_sheet.png").write_bytes(_ONE_PIXEL_PNG)
        (output_dir / "top_view.png").write_bytes(_ONE_PIXEL_PNG)
        return deepcopy(l1)

    render.calls = calls
    return render


def _judge(scores=(60, 99)):
    calls = []

    def run(config, prompt, workspace, trace_dir, *, role=None, images=(), output_schema=None, **kwargs):
        index = len(calls)
        calls.append({"role": role, "workspace": Path(workspace), "kwargs": kwargs})
        response = {
            "score": scores[min(index, len(scores) - 1)],
            "subject_complete": True,
            "issues": [],
            "summary": "MOCK UNIT FIXTURE: no VLM inspected these images.",
        }
        return {"exit_code": 0, "output": json.dumps(response)}

    run.calls = calls
    return run


class OrchestratorIntegrationTests(unittest.TestCase):
    def test_keeps_first_valid_candidate_when_high_scoring_repair_breaks_l1(self):
        coder = _coder()
        renderer = _renderer((True, False))
        judge = _judge((60, 99))
        with tempfile.TemporaryDirectory(prefix="proxy-orchestrator-unit-fixture-") as tmp:
            root = Path(tmp)
            assets = _assets(root)
            run_dir = root / "run"
            with (
                patch("proxy_harness.orchestrator.run_agent", side_effect=coder),
                patch("proxy_harness.orchestrator.render_scene", side_effect=renderer),
                patch("proxy_harness.evaluate.run_agent", side_effect=judge),
            ):
                summary = run_task(_spec(), run_dir, _config(), assets_dir=assets)
            history = read_json(run_dir / "history.json")
            self.assertEqual(summary["selected_iteration"], 0)
            self.assertEqual(summary["best_valid_iteration"], 0)
            self.assertEqual(summary["score"], 60)
            self.assertEqual(summary["status"], "rejected")
            self.assertTrue(summary["hard_pass"])
            self.assertEqual(summary["iterations"], 2)
            self.assertFalse(any(item["accepted"] for item in history))
            self.assertEqual(history[1]["score"], 99)
            self.assertEqual(history[1]["effective_score"], 40)
            self.assertFalse(history[1]["hard_pass"])
            self.assertIn("mocked candidate 0", (run_dir / "best_valid" / "scene.html").read_text())
            self.assertIn("mocked candidate 1", (run_dir / "workspace" / "scene.html").read_text())
            self.assertEqual(read_json(run_dir / "summary.json")["status"], "rejected")
            # Controller uses the exact first thread id for the next writer.
            self.assertEqual([item["session_id"] for item in coder.calls], [None, _SESSION_ID])
            self.assertTrue(all(item["role"] == "evaluator" for item in judge.calls))
            self.assertTrue(all("session_id" not in item["kwargs"] for item in judge.calls))

    def test_missing_evaluator_cannot_accept_even_renderable_scene(self):
        coder = _coder()
        renderer = _renderer()
        with tempfile.TemporaryDirectory(prefix="proxy-orchestrator-unit-fixture-") as tmp:
            root = Path(tmp)
            assets = _assets(root)
            run_dir = root / "run"
            with (
                patch("proxy_harness.orchestrator.run_agent", side_effect=coder),
                patch("proxy_harness.orchestrator.render_scene", side_effect=renderer),
                patch("proxy_harness.evaluate.run_agent") as external_judge,
            ):
                summary = run_task(_spec(), run_dir, _config(evaluator=False), assets_dir=assets)
            self.assertEqual(summary["status"], "unverified")
            self.assertIsNone(summary["best_valid_iteration"])
            self.assertIsNone(summary["score"])
            self.assertEqual(summary["iterations"], 1)
            self.assertFalse((run_dir / "best_valid").exists())
            self.assertFalse(any(row["accepted"] for row in read_json(run_dir / "history.json")))
            self.assertEqual(read_json(run_dir / "iter_00" / "eval_request.json")["history"], [])
            external_judge.assert_not_called()

    def test_protected_asset_mutation_is_rejected_and_aborts_repairs(self):
        coder = _coder(mutate_protected=True)
        renderer = _renderer()
        judge = _judge((99,))
        with tempfile.TemporaryDirectory(prefix="proxy-orchestrator-unit-fixture-") as tmp:
            root = Path(tmp)
            assets = _assets(root)
            original_runtime = (assets / "runtime.js").read_bytes()
            run_dir = root / "run"
            with (
                patch("proxy_harness.orchestrator.run_agent", side_effect=coder),
                patch("proxy_harness.orchestrator.render_scene", side_effect=renderer),
                patch("proxy_harness.evaluate.run_agent", side_effect=judge),
            ):
                summary = run_task(_spec(), run_dir, _config(iterations=3), assets_dir=assets)
            self.assertEqual(summary["status"], "rejected")
            self.assertFalse(summary["hard_pass"])
            self.assertEqual(summary["iterations"], 1)
            self.assertEqual(len(coder.calls), 1)
            self.assertFalse((run_dir / "iter_01").exists())
            self.assertIsNone(summary["best_valid_iteration"])
            l1 = read_json(run_dir / "iter_00" / "l1_report.json")
            self.assertTrue(any(item["code"] == "output_integrity" and "runtime.js" in item["message"] for item in l1["failures"]))
            self.assertEqual((assets / "runtime.js").read_bytes(), original_runtime)
            self.assertEqual((run_dir / "assets_snapshot" / "runtime.js").read_bytes(), original_runtime)

    def test_nonzero_coder_exit_rejects_an_otherwise_renderable_candidate(self):
        coder = _coder(exit_codes=(7,))
        renderer = _renderer((True,))
        judge = _judge((99,))
        with tempfile.TemporaryDirectory(prefix="proxy-orchestrator-unit-fixture-") as tmp:
            root = Path(tmp)
            assets = _assets(root)
            run_dir = root / "run"
            with (
                patch("proxy_harness.orchestrator.run_agent", side_effect=coder),
                patch("proxy_harness.orchestrator.render_scene", side_effect=renderer),
                patch("proxy_harness.evaluate.run_agent", side_effect=judge),
            ):
                summary = run_task(_spec(), run_dir, _config(iterations=1), assets_dir=assets)
            self.assertEqual(len(renderer.calls), 1)
            self.assertEqual(summary["status"], "rejected")
            self.assertFalse(summary["hard_pass"])
            self.assertIsNone(summary["best_valid_iteration"])
            self.assertEqual(summary["score"], 40)
            l1 = read_json(run_dir / "iter_00" / "l1_report.json")
            self.assertTrue(any("Coder exited unsuccessfully: 7" in item["message"] for item in l1["failures"]))
            self.assertFalse(read_json(run_dir / "history.json")[0]["accepted"])

    def test_codex_resume_argv_uses_exact_session_id_and_never_last(self):
        process_calls = []

        def process(argv, *, cwd, stdin, timeout):
            process_calls.append({"argv": list(argv), "cwd": Path(cwd), "stdin": stdin})
            return {
                "exit_code": 0, "timed_out": False, "elapsed_s": 0,
                "stdout": json.dumps({"type": "thread.started", "thread_id": _SESSION_ID}) + "\n",
                "stderr": "",
            }

        with tempfile.TemporaryDirectory(prefix="proxy-engine-unit-fixture-") as tmp:
            root = Path(tmp)
            with (
                patch("proxy_harness.engine.shutil.which", return_value="/mock-unit-fixture/codex"),
                patch("proxy_harness.engine.run_process", side_effect=process),
            ):
                first = run_agent({"backend": "codex"}, "first mocked turn", root / "work", root / "first")
                second = run_agent({"backend": "codex"}, "repair mocked turn", root / "work", root / "second", session_id=first["session_id"])
            first_argv, second_argv = [item["argv"] for item in process_calls]
            self.assertNotIn("resume", first_argv)
            resume_index = second_argv.index("resume")
            self.assertEqual(second_argv[resume_index + 1], _SESSION_ID)
            self.assertNotIn("--last", second_argv)
            self.assertEqual(second["session_id"], _SESSION_ID)
            self.assertTrue(second["session_continued"])
            self.assertEqual(process_calls[0]["cwd"], process_calls[1]["cwd"])
            self.assertEqual(process_calls[1]["stdin"], "repair mocked turn")
            stored = read_json(root / "second" / "coder_result.json")
            self.assertEqual(stored["argv"], second_argv)
            self.assertNotIn("--last", stored["argv"])


if __name__ == "__main__":
    unittest.main()
