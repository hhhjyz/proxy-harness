"""Contract tests for input validation; no model or browser is invoked."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from proxy_harness.spec import SpecValidationError, is_active, load_spec, validate_spec


def basic_spec():
    return {
        "schema_version": "1.0",
        "task_id": "unit_scene",
        "prompt": "A ball moves in front of a stationary camera.",
        "duration_s": 2,
        "fps": 12,
        "width": 640,
        "height": 360,
        "sample_times": [0, 0.5, 1, 1.5],
        "subjects": [{"id": "ball"}],
    }


class SpecTests(unittest.TestCase):
    def test_defaults_do_not_mutate_input_or_invent_screen_thresholds(self):
        original = basic_spec()
        before = deepcopy(original)
        normalized = validate_spec(original)
        self.assertEqual(original, before)
        self.assertEqual(normalized["subjects"][0]["role"], "primary")
        self.assertTrue(normalized["subjects"][0]["required"])
        self.assertFalse(normalized["subjects"][0]["fully_in_frame"])
        self.assertNotIn("min_area_ratio", normalized["subjects"][0])
        self.assertNotIn("max_area_ratio", normalized["subjects"][0])
        self.assertEqual(normalized["collision"]["forbidden_pairs"], [])
        self.assertEqual(normalized["split"], "demo")

    def test_explicit_camera_and_area_policy_is_preserved(self):
        spec = basic_spec()
        spec["camera"] = {"pitch_deg_range": [-50, -10], "height_range": [2, 5]}
        spec["subjects"][0].update(fully_in_frame=True, min_area_ratio=0.01, max_area_ratio=0.8)
        self.assertEqual(validate_spec(spec)["camera"], spec["camera"])
        self.assertEqual(validate_spec(spec)["subjects"][0]["max_area_ratio"], 0.8)

    def test_rejects_nonfinite_numbers_anywhere_in_spec(self):
        for value in (float("nan"), float("inf"), float("-inf"), 10 ** 1000):
            with self.subTest(value_type=type(value).__name__):
                spec = basic_spec()
                spec["duration_s"] = value
                with self.assertRaises(SpecValidationError):
                    validate_spec(spec)
        spec = basic_spec()
        spec["camera"] = {"height_range": [0, float("nan")]}
        with self.assertRaises(SpecValidationError):
            validate_spec(spec)

    def test_boolean_is_not_a_number_or_integer(self):
        for field in ("fps", "duration_s", "width", "seed"):
            spec = basic_spec()
            spec[field] = True
            with self.subTest(field=field), self.assertRaises(SpecValidationError):
                validate_spec(spec)

    def test_dimensions_are_positive_even_integers(self):
        for value in (0, -2, 641, 640.0):
            spec = basic_spec()
            spec["width"] = value
            with self.subTest(value=value), self.assertRaises(SpecValidationError):
                validate_spec(spec)

    def test_unknown_fields_and_missing_required_fields_fail(self):
        spec = basic_spec()
        spec["sample_time"] = [0]
        with self.assertRaisesRegex(SpecValidationError, "unknown field"):
            validate_spec(spec)
        del spec["sample_time"]
        del spec["sample_times"]
        with self.assertRaisesRegex(SpecValidationError, "missing required"):
            validate_spec(spec)

    def test_sample_times_must_be_ordered_unique_and_below_duration(self):
        for times in ([0, 2], [0.5, 0], [0, 0], [-0.1, 0], []):
            spec = basic_spec()
            spec["sample_times"] = times
            with self.subTest(times=times), self.assertRaises(SpecValidationError):
                validate_spec(spec)

    def test_duplicate_and_blank_subject_ids_fail(self):
        spec = basic_spec()
        spec["subjects"].append({"id": "ball"})
        with self.assertRaisesRegex(SpecValidationError, "duplicate"):
            validate_spec(spec)
        spec["subjects"] = [{"id": "   "}]
        with self.assertRaisesRegex(SpecValidationError, "blank"):
            validate_spec(spec)

    def test_active_intervals_are_half_open(self):
        spec = basic_spec()
        spec["subjects"][0]["active_intervals"] = [[0.5, 1.0], [1.5, 2]]
        subject = validate_spec(spec)["subjects"][0]
        self.assertFalse(is_active(subject, 0))
        self.assertTrue(is_active(subject, 0.5))
        self.assertFalse(is_active(subject, 1))
        self.assertTrue(is_active(subject, 1.999999))
        self.assertFalse(is_active(subject, 2))

    def test_active_intervals_cannot_be_invalid_or_vacuously_evaluated(self):
        for intervals in ([[1, 1]], [[-1, 1]], [[1, 3]], [[0, 1], [0.5, 1.5]], [[0.1, 0.2]]):
            spec = basic_spec()
            spec["subjects"][0]["active_intervals"] = intervals
            with self.subTest(intervals=intervals), self.assertRaises(SpecValidationError):
                validate_spec(spec)

    def test_optional_subject_can_have_no_active_sample(self):
        spec = basic_spec()
        spec["subjects"][0].update(required=False, active_intervals=[[0.1, 0.2]])
        validate_spec(spec)

    def test_bad_camera_ranges_fail(self):
        for camera in ({"pitch_deg_range": [-100, 0]}, {"height_range": [3, 2]}, {"pitch_deg_range": [0, float("inf")]}):
            spec = basic_spec()
            spec["camera"] = camera
            with self.subTest(camera=camera), self.assertRaises(SpecValidationError):
                validate_spec(spec)

    def test_bad_area_range_fails(self):
        spec = basic_spec()
        spec["subjects"][0].update(min_area_ratio=0.8, max_area_ratio=0.2)
        with self.assertRaises(SpecValidationError):
            validate_spec(spec)

    def test_pairs_are_known_nonself_unique_and_noncontradictory(self):
        cases = [
            {"forbidden_pairs": [["ball", "missing"]]},
            {"forbidden_pairs": [["ball", "ball"]]},
            {"forbidden_pairs": [["ball", "box"], ["box", "ball"]]},
            {"forbidden_pairs": [["ball", "box"]], "allowed_pairs": [["box", "ball"]]},
        ]
        for collision in cases:
            spec = basic_spec()
            spec["subjects"].append({"id": "box"})
            spec["collision"] = collision
            with self.subTest(collision=collision), self.assertRaises(SpecValidationError):
                validate_spec(spec)

    def test_contact_tolerance_cannot_be_negative(self):
        spec = basic_spec()
        spec["collision"] = {"tolerance": -0.001}
        with self.assertRaises(SpecValidationError):
            validate_spec(spec)

    def test_hard_fail_cap_stays_below_acceptance(self):
        spec = basic_spec()
        spec["evaluation"] = {"pass_score": 40, "hard_fail_score_cap": 40}
        with self.assertRaises(SpecValidationError):
            validate_spec(spec)

    def test_split_can_be_holdout_without_rewriting_it(self):
        spec = basic_spec()
        spec["split"] = "holdout"
        self.assertEqual(validate_spec(spec)["split"], "holdout")

    def test_load_spec_rejects_nonstandard_json_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.json"
            path.write_text(json.dumps({**basic_spec(), "duration_s": float("nan")}), encoding="utf-8")
            with self.assertRaisesRegex(SpecValidationError, "non-finite JSON"):
                load_spec(path)
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(SpecValidationError, "invalid JSON"):
                load_spec(path)
            path.write_text(json.dumps(basic_spec()), encoding="utf-8")
            self.assertEqual(load_spec(path)["task_id"], "unit_scene")

    def test_every_validation_error_contains_a_fix(self):
        spec = basic_spec()
        spec["fps"] = 0
        with self.assertRaises(SpecValidationError) as caught:
            validate_spec(spec)
        self.assertIn("Fix:", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
