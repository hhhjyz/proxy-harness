"""Admission integrity tests using temporary synthetic JSON fixtures.

No live model is invoked here.  Fixture flags exercise the provenance schema;
these tests and their scores are not scene-generation benchmark results.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from proxy_harness.outerloop import (
    EVIDENCE_FILES, OuterLoopError, admit_candidate, bind_candidate, canonical_sha256,
    condition_id, hash_tree, load_manifest, prepare_distillation,
)
from proxy_harness.spec import validate_spec


ROOT = Path(__file__).resolve().parent.parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class OuterLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="outerloop-unit-fixture-")
        self.root = Path(self.temp.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        for name in ("runtime.js", "primitives.js", "scene.html", "AGENTS.md"):
            (self.assets / name).write_text("unit fixture only: " + name + "\n", encoding="utf-8")
        self.config = {"coder": {"backend": "unit-test-coder"}, "evaluator": {"backend": "unit-test-judge"},
                       "max_iterations": 2, "render": {"backend": "unit-test-fixture"}}
        self.manifest_path = self.root / "suite" / "manifest.json"
        tasks = []
        for split, count in (("train", 2), ("dev", 3), ("holdout", 1)):
            for index in range(count):
                task_id = f"{split}_unit_{index}"
                spec = validate_spec({"schema_version": "1.0", "task_id": task_id, "split": split,
                                      "seed": 42 + index, "prompt": f"Unique {split} unit fixture prompt {index}.",
                                      "duration_s": 3, "fps": 12, "width": 640, "height": 360,
                                      "sample_times": [0, 1, 2], "subjects": [{"id": "box"}]})
                relative = f"{split}/{task_id}.spec.json"
                write(self.manifest_path.parent / relative, spec)
                tasks.append({"task_id": task_id, "split": split, "seed": spec["seed"],
                              "spec_path": relative, "spec_sha256": canonical_sha256(spec)})
        manifest = {"schema_version": "1.0", "dataset_id": "temporary-unit-fixture", "tasks": tasks}
        write(self.manifest_path, manifest)
        self.manifest_path.with_suffix(".sha256").write_text(canonical_sha256(manifest) + "  manifest.json\n")
        self.manifest = load_manifest(self.manifest_path)
        self.tasks = {split: [task for task in self.manifest["tasks"] if task["split"] == split]
                      for split in ("train", "dev", "holdout")}
        self.run_count = 0

    def tearDown(self):
        self.temp.cleanup()

    def run_record(self, task, *, assets=None, status="accepted", hard=True, complete=True,
                   score=90, lines=10, primitives=4, mode="live", eval_status="evaluated"):
        self.run_count += 1
        assets = assets or self.assets
        run = self.root / "runs" / f"run_{self.run_count}"
        iteration = run / "iter_00"
        iteration.mkdir(parents=True)
        write(run / "spec.json", task["spec"])
        write(run / "config.json", self.config)
        shutil.copytree(assets, run / "assets_snapshot")
        (iteration / "scene.html").write_text("".join(f"// unit fixture line {i}\n" for i in range(lines)))
        (iteration / "appearance_prompt.txt").write_text("box: unit fixture appearance\n")
        (iteration / "coder_prompt.txt").write_text(task["spec"]["prompt"])
        (iteration / "coder.stdout.jsonl").write_text('{"fixture_only":true,"not_a_live_result":true}\n')
        (iteration / "coder.stderr.log").write_text("")
        (iteration / "console.log").write_text("")
        (iteration / "diff.patch").write_text("fixture diff\n")
        (iteration / "contact_sheet.png").write_bytes(b"unit image presence fixture; not a rendered scene")
        (iteration / "top_view.png").write_bytes(b"unit image presence fixture; not a rendered scene")
        write(iteration / "coder_result.json", {"backend": "fixture" if mode == "demo" else "unit-test-coder",
                                                "exit_code": 0, "session_id": "temporary-unit-fixture"})
        write(iteration / "l1_report.json", {"schema_version": "1.0", "hard_pass": hard,
                                            "failures": [] if hard else [{"code": "fixture_failure"}]})
        write(iteration / "eval.json", {"status": eval_status, "score": score if eval_status == "evaluated" else None,
                                       "subject_complete": complete, "summary": "Synthetic unit fixture, not measured quality.", "issues": []})
        write(iteration / "manifests.json", [{"t": 0, "audit": {"primitive_count": primitives}}])
        write(iteration / "eval_request.json", {"fixture_only": True, "history": []})
        write(iteration / "eval_response.json", {"fixture_only": True})
        effective = (score if hard else min(score, task["spec"]["evaluation"]["hard_fail_score_cap"])) if eval_status == "evaluated" else None
        summary = {"schema_version": "1.0", "task_id": task["task_id"], "split": task["split"],
                   "seed": task["seed"], "spec_sha256": task["spec_sha256"], "status": status,
                   "hard_pass": hard, "subject_complete": complete, "score": effective,
                   "code_lines": lines, "primitive_count": primitives, "selected_iteration": 0,
                   "evidence_dir": "iter_00", "model_mode": mode, "asset_sha256": hash_tree(assets),
                   "iterations": 1, "fixture_only": True}
        write(run / "summary.json", summary)
        write(run / "history.json", [{"iteration": 0, "hard_pass": hard,
                                      "score": score, "eval_status": eval_status, "fixture_only": True}])
        return {**summary, "run_dir": str(run)}

    def repaired_train_record(self):
        """Two synthetic rounds: a failing draft followed by a selected repair."""
        record = self.run_record(self.tasks["train"][0])
        run = Path(record["run_dir"])
        shutil.copytree(run / "iter_00", run / "iter_01")
        write(run / "iter_00" / "l1_report.json", {"schema_version": "1.0", "hard_pass": False,
                                                   "failures": [{"code": "unit_fixture_missing_subject"}]})
        write(run / "iter_00" / "eval.json", {"status": "evaluated", "score": 30,
                                              "subject_complete": False, "summary": "Synthetic failed draft", "issues": []})
        (run / "iter_00" / "scene.html").write_text("// original failed unit draft\n")
        (run / "iter_00" / "diff.patch").write_text("+ original failed unit draft\n")
        (run / "iter_01" / "diff.patch").write_text("- original failed unit draft\n+ repaired unit draft\n")
        for index in (0, 1):
            iteration = run / f"iter_{index:02d}"
            (iteration / "frames").mkdir()
            (iteration / "frames" / "sample_00.png").write_bytes(f"unit sampled frame from round {index}".encode())
            (iteration / "video_frames").mkdir()
            (iteration / "video_frames" / "000000.png").write_bytes(f"unit full video frame from round {index}".encode())
            (iteration / "sample_detail.png").write_bytes(f"unit detail frame from round {index}".encode())
            write(iteration / "eval_wire_request.json", {"round": index, "fixture_only": True})
            write(iteration / "evaluator_result.json", {"round": index, "fixture_only": True})
            (iteration / "evaluator_prompt.txt").write_text(f"Unit evaluator request for train round {index}\n")
            (iteration / "evaluator.stdout.jsonl").write_text(f'{{"round":{index},"fixture_only":true}}\n')
            write(iteration / "render_meta.json", {"round": index, "fixture_only": True})
        summary = json.loads((run / "summary.json").read_text())
        summary.update(selected_iteration=1, evidence_dir="iter_01", iterations=2)
        write(run / "summary.json", summary)
        write(run / "history.json", [{"iteration": 0, "hard_pass": False, "accepted": False, "fixture_only": True},
                                      {"iteration": 1, "hard_pass": True, "accepted": True, "fixture_only": True}])
        record.update(selected_iteration=1, evidence_dir="iter_01", iterations=2)
        return record

    def batch(self, records, *, assets=None):
        result = {"schema_version": "1.0", "manifest_sha256": self.manifest["manifest_sha256"],
                  "config_sha256": canonical_sha256(self.config), "harness_sha256": "b" * 64,
                  "asset_sha256": hash_tree(assets or self.assets), "records": records}
        result["condition_id"] = condition_id(result)
        return result

    def stage(self):
        train = self.batch([self.run_record(task) for task in self.tasks["train"]])
        stage = self.root / "staging"
        prepared = prepare_distillation(train, self.manifest_path, stage, candidate_name="unit-candidate", baseline_assets=self.assets)
        candidate_assets = Path(prepared["candidate_assets"])
        (candidate_assets / "primitives.js").write_text("unit fixture candidate primitive\n")
        bind_candidate(stage, candidate_assets)
        return stage, candidate_assets

    def pair(self, baseline_options=None, candidate_options=None):
        stage, assets = self.stage()
        baseline_options = baseline_options or [{} for _ in self.tasks["dev"]]
        candidate_options = candidate_options or [{} for _ in self.tasks["dev"]]
        baseline = self.batch([self.run_record(task, **options) for task, options in zip(self.tasks["dev"], baseline_options)])
        candidate = self.batch([self.run_record(task, assets=assets, **options) for task, options in zip(self.tasks["dev"], candidate_options)], assets=assets)
        return stage, assets, baseline, candidate

    def admit(self, pair):
        stage, assets, baseline, candidate = pair
        return admit_candidate(baseline, candidate, self.manifest_path,
                               proposal_path=stage / "proposal.json", candidate_assets=assets)

    def assert_rejected(self, report, code):
        self.assertFalse(report["eligible"], report)
        self.assertEqual(report["promotion"], "not_performed")
        self.assertIn(code, [reason["code"] for reason in report["reasons"]], report)

    def test_shipped_suite_is_original_and_pinned_6_24_6(self):
        manifest = load_manifest(ROOT / "devset" / "manifest.json")
        self.assertEqual({split: sum(task["split"] == split for task in manifest["tasks"])
                          for split in ("train", "dev", "holdout")}, {"train": 6, "dev": 24, "holdout": 6})
        self.assertIn("not CodeVideoBench", manifest["provenance"])
        self.assertEqual(len({task["spec"]["prompt"] for task in manifest["tasks"]}), 36)

    def test_changed_spec_is_rejected_without_repinning(self):
        task = self.tasks["train"][0]
        spec = deepcopy(task["spec"])
        spec["prompt"] += " tampered"
        write(self.manifest_path.parent / task["spec_path"], spec)
        with self.assertRaisesRegex(OuterLoopError, "SPEC_HASH_MISMATCH"):
            load_manifest(self.manifest_path)

    def test_changed_manifest_is_rejected(self):
        manifest = json.loads(self.manifest_path.read_text())
        manifest["tasks"] = manifest["tasks"][:-1]
        write(self.manifest_path, manifest)
        with self.assertRaisesRegex(OuterLoopError, "MANIFEST_HASH_MISMATCH"):
            load_manifest(self.manifest_path)

    def test_mixed_distillation_never_opens_or_copies_dev_holdout(self):
        train = self.run_record(self.tasks["train"][0], status="rejected", hard=False, score=95)
        other = [{"task_id": task["task_id"], "split": task["split"], "run_dir": "/not/a/real/run"}
                 for split in ("dev", "holdout") for task in self.tasks[split]]
        prepared = prepare_distillation(self.batch([train, *other]), self.manifest_path, self.root / "stage",
                                        candidate_name="train-only", baseline_assets=self.assets)
        self.assertEqual(prepared["training_case_count"], 1)
        self.assertTrue(Path(prepared["prompt_path"]).is_file())
        content = "\n".join(path.read_text(errors="replace") for path in (self.root / "stage").rglob("*")
                            if path.is_file() and path.suffix != ".png")
        for split in ("dev", "holdout"):
            for task in self.tasks[split]:
                self.assertNotIn(task["task_id"], content)
                self.assertNotIn(task["spec"]["prompt"], content)
        source = Path(train["run_dir"]) / "iter_00" / "scene.html"
        copied = self.root / "stage" / "train_evidence" / train["task_id"] / "iter_00" / "scene.html"
        self.assertEqual(source.read_bytes(), copied.read_bytes())

    def test_split_spoofing_cannot_feed_holdout_to_distiller(self):
        record = self.run_record(self.tasks["holdout"][0])
        record["split"] = "train"
        with self.assertRaisesRegex(OuterLoopError, "SPLIT_SPOOFING"):
            prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage", candidate_name="bad")

    def test_demo_does_not_become_train_evidence(self):
        record = self.run_record(self.tasks["train"][0], mode="demo", status="unverified", eval_status="skipped", score=None, complete=None)
        with self.assertRaisesRegex(OuterLoopError, "NO_VERIFIED_TRAIN_EVIDENCE"):
            prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage", candidate_name="bad")

    def test_leaked_holdout_in_raw_train_trace_is_excluded(self):
        record = self.run_record(self.tasks["train"][0])
        (Path(record["run_dir"]) / "iter_00" / "coder.stdout.jsonl").write_text(self.tasks["holdout"][0]["spec"]["prompt"])
        with self.assertRaisesRegex(OuterLoopError, "NO_VERIFIED_TRAIN_EVIDENCE"):
            prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage", candidate_name="bad")

    def test_complete_failed_to_repaired_raw_trajectory_is_preserved(self):
        record = self.repaired_train_record()
        prepared = prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage",
                                        candidate_name="all-raw-rounds", baseline_assets=self.assets)
        stage = Path(prepared["request_path"]).parent
        request = json.loads(Path(prepared["request_path"]).read_text())
        case = request["training_cases"][0]
        self.assertEqual(case["selected_iteration"], 1)
        self.assertEqual(case["trace_iterations"], [0, 1])
        self.assertEqual(case["trace_policy"], "all_recorded_iterations")
        prefix = f"train_evidence/{record['task_id']}/"
        run = Path(record["run_dir"])
        wanted = ["history.json", "config.json", "summary.json", "spec.json"]
        for index in (0, 1):
            iteration = f"iter_{index:02d}"
            wanted += [f"{iteration}/{name}" for name in EVIDENCE_FILES if (run / iteration / name).is_file()]
            wanted += [f"{iteration}/frames/sample_00.png", f"{iteration}/video_frames/000000.png", f"{iteration}/sample_detail.png"]
        proposal = json.loads((stage / "proposal.json").read_text())
        import hashlib
        for relative in wanted:
            listed = prefix + relative
            self.assertIn(listed, case["evidence_paths"])
            source = (run / relative).read_bytes()
            self.assertEqual((stage / listed).read_bytes(), source)
            self.assertEqual(proposal["evidence_sha256"][listed], hashlib.sha256(source).hexdigest())
        self.assertNotEqual((stage / (prefix + "iter_00/diff.patch")).read_bytes(),
                            (stage / (prefix + "iter_01/diff.patch")).read_bytes())

    def test_dev_prompt_in_earlier_failed_iteration_rejects_whole_trace(self):
        record = self.repaired_train_record()
        (Path(record["run_dir"]) / "iter_00" / "evaluator_prompt.txt").write_text(self.tasks["dev"][0]["spec"]["prompt"])
        with self.assertRaisesRegex(OuterLoopError, "NO_VERIFIED_TRAIN_EVIDENCE"):
            prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage", candidate_name="bad")
        self.assertFalse((self.root / "stage").exists())

    def test_earlier_frame_symlink_is_not_copied(self):
        record = self.repaired_train_record()
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside the recorded trajectory")
        frame = Path(record["run_dir"]) / "iter_00" / "frames" / "sample_00.png"
        frame.unlink()
        frame.symlink_to(outside)
        with self.assertRaisesRegex(OuterLoopError, "NO_VERIFIED_TRAIN_EVIDENCE"):
            prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage", candidate_name="bad")

    def test_missing_failed_iteration_is_not_a_complete_raw_trace(self):
        record = self.repaired_train_record()
        shutil.rmtree(Path(record["run_dir"]) / "iter_00")
        with self.assertRaisesRegex(OuterLoopError, "NO_VERIFIED_TRAIN_EVIDENCE"):
            prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage", candidate_name="bad")

    def test_no_model_proof_or_missing_raw_file_cannot_be_distilled(self):
        record = self.run_record(self.tasks["train"][0])
        (Path(record["run_dir"]) / "iter_00" / "coder.stdout.jsonl").unlink()
        with self.assertRaisesRegex(OuterLoopError, "NO_VERIFIED_TRAIN_EVIDENCE"):
            prepare_distillation(self.batch([record]), self.manifest_path, self.root / "stage", candidate_name="bad")

    def test_valid_pass_rate_improvement_is_eligible_but_not_promoted(self):
        pair = self.pair([{}, {"status": "rejected", "score": 70}, {"status": "rejected", "score": 50}],
                         [{}, {}, {"status": "rejected", "score": 50}])
        original = (self.assets / "primitives.js").read_bytes()
        report = self.admit(pair)
        self.assertTrue(report["eligible"], report)
        self.assertEqual(report["admission_basis"], "pass_rate")
        self.assertEqual(report["metrics"]["baseline_passed"], 1)
        self.assertEqual(report["metrics"]["candidate_passed"], 2)
        self.assertEqual(report["promotion"], "not_performed")
        self.assertEqual((self.assets / "primitives.js").read_bytes(), original)

    def test_complexity_reduction_on_same_passed_tasks_is_eligible(self):
        pair = self.pair(candidate_options=[{"lines": 8} for _ in self.tasks["dev"]])
        report = self.admit(pair)
        self.assertTrue(report["eligible"], report)
        self.assertEqual(report["admission_basis"], "passed_intersection_complexity")
        self.assertEqual(report["metrics"]["intersection_complexity_totals"]["candidate"]["code_lines"], 24)

    def test_incomplete_dev_set_cannot_cherry_pick_successes(self):
        pair = self.pair()
        pair[3]["records"].pop()
        self.assert_rejected(self.admit(pair), "INCOMPLETE_DEV_SET")

    def test_duplicate_task_is_not_a_complete_dev_set(self):
        pair = self.pair()
        pair[3]["records"][-1] = deepcopy(pair[3]["records"][0])
        self.assert_rejected(self.admit(pair), "DUPLICATE_TASK")

    def test_holdout_is_never_an_admission_split(self):
        pair = self.pair()
        pair[3]["records"].append(self.run_record(self.tasks["holdout"][0], assets=pair[1]))
        self.assert_rejected(self.admit(pair), "NON_DEV_ADMISSION")

    def test_rehashed_different_config_still_fails_pairing(self):
        pair = self.pair()
        pair[3]["config_sha256"] = "c" * 64
        pair[3]["condition_id"] = condition_id(pair[3])
        self.assert_rejected(self.admit(pair), "PAIRED_CONDITION_MISMATCH")

    def test_wrong_seed_is_rejected(self):
        pair = self.pair()
        pair[3]["records"][0]["seed"] += 1
        self.assert_rejected(self.admit(pair), "RECORD_IDENTITY_MISMATCH")

    def test_same_metadata_cannot_hide_changed_run_config(self):
        pair = self.pair()
        write(Path(pair[3]["records"][0]["run_dir"]) / "config.json", {"different_budget": True})
        self.assert_rejected(self.admit(pair), "RUN_CONFIG_MISMATCH")

    def test_condition_id_cannot_be_reused_for_other_assets(self):
        pair = self.pair()
        pair[3]["condition_id"] = pair[2]["condition_id"]
        self.assert_rejected(self.admit(pair), "CONDITION_MISMATCH")

    def test_post_bind_asset_edit_invalidates_trial(self):
        pair = self.pair()
        (pair[1] / "primitives.js").write_text("changed after candidate batch\n")
        self.assert_rejected(self.admit(pair), "CANDIDATE_ASSET_MISMATCH")

    def test_declared_complexity_cannot_override_raw_source(self):
        pair = self.pair()
        record = pair[3]["records"][0]
        record["code_lines"] = 1
        summary_path = Path(record["run_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["code_lines"] = 1
        write(summary_path, summary)
        self.assert_rejected(self.admit(pair), "COMPLEXITY_MISMATCH")

    def test_deleting_subject_for_fewer_primitives_is_rejected(self):
        pair = self.pair(candidate_options=[{"status": "rejected", "complete": False, "lines": 2, "primitives": 1}, {}, {}])
        report = self.admit(pair)
        self.assert_rejected(report, "SUBJECT_COMPLETENESS_REGRESSION")
        self.assertIn("ACCEPTANCE_REGRESSION", [item["code"] for item in report["reasons"]])

    def test_failing_complex_case_does_not_improve_complexity_denominator(self):
        pair = self.pair(candidate_options=[{"status": "rejected", "score": 60, "lines": 1}, {"lines": 8}, {"lines": 8}])
        self.assert_rejected(self.admit(pair), "ACCEPTANCE_REGRESSION")

    def test_hard_gate_failure_cannot_be_rescued_by_high_score(self):
        pair = self.pair(candidate_options=[{"status": "rejected", "hard": False, "score": 99}, {}, {}])
        self.assert_rejected(self.admit(pair), "HARD_GATE_REGRESSION")

    def test_skipped_evaluation_is_never_a_success(self):
        pair = self.pair(candidate_options=[{"status": "unverified", "eval_status": "skipped", "score": None, "complete": None, "mode": "unverified"}, {}, {}])
        self.assert_rejected(self.admit(pair), "NOT_LIVE")

    def test_false_accepted_label_with_skipped_evaluator_is_rejected(self):
        pair = self.pair(candidate_options=[{"status": "accepted", "eval_status": "skipped", "score": None, "complete": None}, {}, {}])
        self.assert_rejected(self.admit(pair), "FALSE_ACCEPTANCE")

    def test_no_change_in_measured_outcome_is_not_an_improvement(self):
        self.assert_rejected(self.admit(self.pair()), "NO_MEASURED_IMPROVEMENT")

    def test_complexity_tradeoff_is_not_automatic_admission(self):
        pair = self.pair(candidate_options=[{"lines": 8, "primitives": 5} for _ in self.tasks["dev"]])
        self.assert_rejected(self.admit(pair), "NO_MEASURED_IMPROVEMENT")

    def test_protected_runtime_cannot_be_changed(self):
        stage, assets = self.stage()
        (assets / "runtime.js").write_text("do not disable measurement\n")
        with self.assertRaisesRegex(OuterLoopError, "FORBIDDEN_ASSET_CHANGE"):
            bind_candidate(stage, assets)

    def test_staged_evidence_cannot_be_edited_by_distiller(self):
        stage, assets = self.stage()
        next((stage / "train_evidence").rglob("scene.html")).write_text("rewritten evidence\n")
        with self.assertRaisesRegex(OuterLoopError, "STAGED_EVIDENCE_CHANGED"):
            bind_candidate(stage, assets)

    def test_resealed_proposal_cannot_forge_baseline_asset_inventory(self):
        stage, assets = self.stage()
        proposal_path = stage / "proposal.json"
        proposal = json.loads(proposal_path.read_text())
        proposal["baseline_asset_inventory"]["runtime.js"]["sha256"] = "f" * 64
        proposal.pop("proposal_sha256")
        proposal["proposal_sha256"] = canonical_sha256(proposal)
        write(proposal_path, proposal)
        with self.assertRaisesRegex(OuterLoopError, "BASELINE_INVENTORY_MISMATCH"):
            bind_candidate(stage, assets)

    def test_selected_iteration_cannot_point_to_different_evidence(self):
        pair = self.pair()
        record = pair[3]["records"][0]
        summary_path = Path(record["run_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["selected_iteration"] = 1
        write(summary_path, summary)
        self.assert_rejected(self.admit(pair), "ITERATION_MISMATCH")

    def test_project_skill_is_staged_and_checks_local_references(self):
        stage, assets = self.stage()
        skill = assets / "skills" / "trajectory-guide" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("---\nname: trajectory-guide\ndescription: Project guide for deterministic motion.\n---\nRead [case](references/case.json).\n")
        write(skill.parent / "references" / "case.json", {"unit_fixture": True})
        result = bind_candidate(stage, assets)
        self.assertIn("skills/trajectory-guide/SKILL.md", result["changed_assets"])
        skill.write_text("---\nname: trajectory-guide\ndescription: Project guide.\n---\nRead [secret](../../../../outside.json).\n")
        with self.assertRaisesRegex(OuterLoopError, "UNSAFE_SKILL_REFERENCE"):
            bind_candidate(stage, assets)

    def test_project_skill_metadata_must_match_index_contract(self):
        stage, assets = self.stage()
        skill = assets / "skills" / "trajectory-guide" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("---\nname: different-name\ndescription: Project guide.\n---\n")
        with self.assertRaisesRegex(OuterLoopError, "INVALID_SKILL_NAME"):
            bind_candidate(stage, assets)

    def test_python_import_caches_do_not_change_harness_hash(self):
        before = hash_tree(self.assets)
        cache = self.assets / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-311.pyc").write_bytes(b"temporary unit fixture cache")
        self.assertEqual(hash_tree(self.assets), before)


if __name__ == "__main__":
    unittest.main()
