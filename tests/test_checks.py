"""Geometry and evidence tests using explicit synthetic manifest fixtures.

These unit fixtures are not presented as browser measurements or live-model
results. Integration tests must obtain their manifests from the real renderer.
"""

from copy import deepcopy
import math
import unittest

from proxy_harness.checks import check_scene


SOURCE = """<!doctype html><canvas id="scene"></canvas>
<script type="module">
import { createHarness } from './runtime.js';
createHarness({scene, camera, renderer, update, subjects, seed: 42});
</script>"""


def spec_for(*ids):
    return {
        "schema_version": "1.0",
        "task_id": "unit_fixture",
        "prompt": "A deliberately specified unit-test scene.",
        "duration_s": 2,
        "fps": 12,
        "width": 640,
        "height": 360,
        "sample_times": [0, 1],
        "subjects": [{"id": subject_id} for subject_id in (ids or ("a",))],
    }


def determinism_ok():
    return {
        "passed": True,
        "tested_times": [0, 1],
        "comparisons": [
            {"t": 0, "passed": True, "pixels_equal": True, "manifest_equal": True},
            {"t": 1, "passed": True, "pixels_equal": True, "manifest_equal": True},
        ],
    }


def sphere(center=(0, 1, 0), radius=0.5):
    return {"type": "sphere", "center": list(center), "radius": radius, "exact": True}


def obb(center=(0, 1, 0), half=(1, 1, 1), angle=0):
    c, s = math.cos(angle), math.sin(angle)
    return {
        "type": "obb", "center": list(center), "half_extents": list(half),
        "axes": [[c, s, 0], [-s, c, 0], [0, 0, 1]], "exact": True,
    }


def bounds(collider):
    if collider["type"] == "sphere":
        extents = [collider["radius"]] * 3
    else:
        extents = [
            sum(abs(collider["axes"][axis][coordinate]) * collider["half_extents"][axis] for axis in range(3))
            for coordinate in range(3)
        ]
    return {
        "min": [value - extent for value, extent in zip(collider["center"], extents)],
        "max": [value + extent for value, extent in zip(collider["center"], extents)],
    }


def obj(subject_id="a", collider=None, role="primary"):
    shape = sphere() if collider is None else collider
    return {
        "id": subject_id, "role": role, "world_aabb": bounds(shape),
        "frustum_intersects": True, "in_front": True,
        "fully_in_frame": True, "projection_valid": True,
        "projected_area_ratio": 0.04,
        "ndc_box": {"min": [-0.2, -0.2], "max": [0.2, 0.2]},
        "visible": True, "collider": shape, "primitive_count": 1,
    }


def frame(t=0, objects=None):
    objects = [obj()] if objects is None else objects
    return {
        "schema_version": "1.0", "t": t,
        "camera": {
            "position": [0, 4, 8],
            "quaternion": [math.sin(-math.pi / 12), 0, 0, math.cos(-math.pi / 12)],
            "forward": [0, -0.5, -math.sqrt(3) / 2], "pitch_deg": -30,
            "fov_deg": 45, "near": 0.1, "far": 100,
        },
        "objects": deepcopy(objects),
        "audit": {
            "mesh_count": len(objects), "registered_mesh_count": len(objects),
            "unregistered_mesh_count": 0, "primitive_count": len(objects),
        },
    }


def frames(objects=None):
    return [frame(0, objects), frame(1, objects)]


def evaluate(spec=None, manifests=None, **kwargs):
    return check_scene(
        spec_for() if spec is None else spec,
        frames() if manifests is None else manifests,
        determinism=kwargs.pop("determinism", determinism_ok()),
        source=kwargs.pop("source", SOURCE),
        **kwargs,
    )


def codes(report, category="failures"):
    return {issue["code"] for issue in report[category]}


def hide_projection(item):
    item.update(
        frustum_intersects=False, in_front=False, fully_in_frame=False,
        projection_valid=False, projected_area_ratio=None, ndc_box=None,
        visible=False, projection_reason="behind_camera",
    )


class CheckTests(unittest.TestCase):
    def test_complete_valid_evidence_passes(self):
        report = evaluate()
        self.assertTrue(report["hard_pass"], report)
        self.assertEqual(report["metrics"]["valid_manifest_count"], 2)

    def test_actual_runtime_error_is_a_hard_failure(self):
        report = evaluate(runtime_errors=["ReferenceError: sceneObject is not defined"])
        self.assertFalse(report["hard_pass"])
        self.assertIn("runtime_error", codes(report))

    def test_missing_determinism_never_passes(self):
        self.assertIn("determinism_missing", codes(evaluate(determinism=None)))

    def test_failed_comparison_cannot_be_overridden_by_top_level_pass(self):
        evidence = determinism_ok()
        evidence["comparisons"][0]["pixels_equal"] = False
        report = evaluate(determinism=evidence)
        self.assertIn("determinism_inconsistent", codes(report))
        evidence["comparisons"][0]["passed"] = False
        evidence["passed"] = False
        self.assertIn("determinism_failed", codes(evaluate(determinism=evidence)))

    def test_determinism_needs_actual_pixel_and_manifest_evidence(self):
        evidence = {"passed": True, "tested_times": [0], "comparisons": [{"passed": True}]}
        self.assertIn("determinism_malformed", codes(evaluate(determinism=evidence)))
        evidence = determinism_ok()
        evidence["tested_times"] = [float("nan")]
        self.assertIn("determinism_malformed", codes(evaluate(determinism=evidence)))

    def test_missing_frame_and_duplicate_time_are_detected(self):
        report = evaluate(manifests=[frame(0)])
        self.assertIn("missing_sample", codes(report))
        report = evaluate(manifests=[frame(0), frame(0)])
        self.assertIn("duplicate_manifest_time", codes(report))
        self.assertIn("missing_sample", codes(report))

    def test_required_id_completeness_is_checked_at_each_time(self):
        scene = frames()
        scene[1]["objects"] = []
        scene[1]["audit"] = dict(mesh_count=0, registered_mesh_count=0, unregistered_mesh_count=0, primitive_count=0)
        report = evaluate(manifests=scene)
        self.assertIn("required_subject_missing", codes(report))
        self.assertEqual(next(x for x in report["failures"] if x["code"] == "required_subject_missing")["t"], 1)

    def test_an_empty_group_does_not_satisfy_required_subject_presence(self):
        scene = frames()
        scene[0]["objects"][0]["primitive_count"] = 0
        self.assertIn("required_subject_empty", codes(evaluate(manifests=scene)))

    def test_invalid_projection_does_not_need_fabricated_numbers(self):
        scene = frames()
        hide_projection(scene[0]["objects"][0])
        report = evaluate(manifests=scene)
        self.assertIn("subject_projection_invalid", codes(report))
        self.assertNotIn("manifest_malformed", codes(report))

    def test_passive_role_invalid_projection_is_advisory(self):
        spec = spec_for("ground")
        spec["subjects"][0]["role"] = "ground"
        scene = frames([obj("ground", role="ground")])
        for sample in scene:
            hide_projection(sample["objects"][0])
        report = evaluate(spec, scene)
        self.assertTrue(report["hard_pass"], report)
        self.assertIn("passive_projection_invalid", codes(report, "warnings"))

    def test_optional_subject_invalid_projection_is_advisory(self):
        spec = spec_for()
        spec["subjects"][0]["required"] = False
        scene = frames()
        hide_projection(scene[0]["objects"][0])
        report = evaluate(spec, scene)
        self.assertTrue(report["hard_pass"], report)
        self.assertIn("subject_projection_invalid", codes(report, "warnings"))

    def test_explicit_passive_screen_constraint_is_enforced(self):
        spec = spec_for("ground")
        spec["subjects"][0].update(role="ground", fully_in_frame=True)
        scene = frames([obj("ground", role="ground")])
        scene[0]["objects"][0]["fully_in_frame"] = False
        self.assertIn("subject_not_fully_in_frame", codes(evaluate(spec, scene)))

    def test_inactive_interval_does_not_require_visibility_or_presence(self):
        spec = spec_for()
        spec["subjects"][0]["active_intervals"] = [[0.5, 1.5]]
        scene = frames()
        hide_projection(scene[0]["objects"][0])
        self.assertTrue(evaluate(spec, scene)["hard_pass"])
        scene[0]["objects"] = []
        scene[0]["audit"] = dict(mesh_count=0, registered_mesh_count=0, unregistered_mesh_count=0, primitive_count=0)
        self.assertTrue(evaluate(spec, scene)["hard_pass"])

    def test_full_frame_is_opt_in_and_independent_from_area(self):
        scene = frames()
        for sample in scene:
            sample["objects"][0].update(fully_in_frame=False, projected_area_ratio=0.9)
            sample["objects"][0]["ndc_box"]["max"] = [1.2, 0.5]
        self.assertTrue(evaluate(manifests=scene)["hard_pass"])
        spec = spec_for()
        spec["subjects"][0]["fully_in_frame"] = True
        self.assertIn("subject_not_fully_in_frame", codes(evaluate(spec, scene)))

    def test_explicit_area_thresholds_are_enforced(self):
        spec = spec_for()
        spec["subjects"][0]["min_area_ratio"] = 0.1
        self.assertIn("subject_area_too_small", codes(evaluate(spec)))
        spec["subjects"][0] = {"id": "a", "max_area_ratio": 0.01}
        self.assertIn("subject_area_too_large", codes(evaluate(spec)))

    def test_zero_area_does_not_satisfy_visible_primary_subject(self):
        scene = frames()
        scene[0]["objects"][0]["projected_area_ratio"] = 0
        self.assertIn("subject_zero_projected_area", codes(evaluate(manifests=scene)))

    def test_camera_height_and_signed_pitch_use_independent_ranges(self):
        spec = spec_for()
        spec["camera"] = {"pitch_deg_range": [-20, -10], "height_range": [1, 3]}
        report = evaluate(spec)
        self.assertIn("camera_pitch_out_of_range", codes(report))
        self.assertIn("camera_height_out_of_range", codes(report))
        spec["camera"] = {"pitch_deg_range": [-40, -20], "height_range": [3, 5]}
        self.assertTrue(evaluate(spec)["hard_pass"])

    def test_unregistered_mesh_is_not_hidden_by_declared_subjects(self):
        scene = frames()
        scene[0]["audit"].update(mesh_count=2, unregistered_mesh_count=1, primitive_count=2)
        report = evaluate(manifests=scene)
        self.assertIn("unregistered_meshes", codes(report))

    def test_no_forbidden_pairs_means_no_unsolicited_collision_hard_gate(self):
        spec = spec_for("a", "b")
        report = evaluate(spec, frames([obj("a", sphere(radius=1)), obj("b", sphere(radius=1))]))
        self.assertTrue(report["hard_pass"], report)
        self.assertEqual(report["metrics"]["exact_collision_checks"], 0)

    def test_allowed_pair_does_not_create_a_collision_failure(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"allowed_pairs": [["a", "b"]]}
        report = evaluate(spec, frames([obj("a"), obj("b")]))
        self.assertTrue(report["hard_pass"], report)

    def test_sphere_sphere_contact_tolerance_and_penetration(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]], "tolerance": 0.001}
        for distance, should_pass in ((2, True), (1.9995, True), (1.998, False)):
            scene = frames([obj("a", sphere((0, 1, 0), 1)), obj("b", sphere((distance, 1, 0), 1))])
            report = evaluate(spec, scene)
            with self.subTest(distance=distance):
                self.assertEqual(report["hard_pass"], should_pass, report)
                if not should_pass:
                    self.assertIn("forbidden_penetration", codes(report))

    def test_rotated_obb_aabb_false_positive_is_not_penetration(self):
        # Parallel thin boxes rotated 45 degrees. Their world AABBs overlap,
        # while the boxes are separated along their short local Y axis.
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]]}
        a = obj("a", obb((0, 1, 0), (2, 0.1, 0.1), math.pi / 4))
        b = obj("b", obb((0, 1.5, 0), (2, 0.1, 0.1), math.pi / 4))
        self.assertTrue(all(min(a["world_aabb"]["max"][i], b["world_aabb"]["max"][i]) > max(a["world_aabb"]["min"][i], b["world_aabb"]["min"][i]) for i in range(3)))
        report = evaluate(spec, frames([a, b]))
        self.assertTrue(report["hard_pass"], report)
        self.assertEqual(report["metrics"]["exact_collision_checks"], 2)
        self.assertNotIn("collision_unverified", codes(report, "warnings"))

    def test_rotated_obbs_that_really_intersect_fail(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]]}
        scene = frames([
            obj("a", obb((0, 1, 0), (2, 0.1, 0.1), math.pi / 4)),
            obj("b", obb((0, 1.1, 0), (2, 0.1, 0.1), math.pi / 4)),
        ])
        self.assertIn("forbidden_penetration", codes(evaluate(spec, scene)))

    def test_obb_face_contact_is_allowed(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]], "tolerance": 0.001}
        for distance in (2, 1.9995):
            scene = frames([obj("a", obb((0, 1, 0))), obj("b", obb((distance, 1, 0)))])
            self.assertTrue(evaluate(spec, scene)["hard_pass"])

    def test_sphere_obb_corner_distance_avoids_aabb_false_positive(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]]}
        box = obj("a", obb((0, 0, 0)))
        for coordinate, should_pass in ((1.6, True), (1.4, False)):
            ball = obj("b", sphere((coordinate, coordinate, 0), 0.75))
            report = evaluate(spec, frames([box, ball]))
            self.assertEqual(report["hard_pass"], should_pass, report)

    def test_sphere_inside_obb_is_penetration(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]]}
        report = evaluate(spec, frames([obj("a", obb((0, 1, 0))), obj("b", sphere((0, 1, 0), 0.2))]))
        self.assertIn("forbidden_penetration", codes(report))

    def test_unknown_shape_aabb_overlap_is_warning_only(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]]}
        objects = [obj("a"), obj("b")]
        objects[0]["collider"] = None
        report = evaluate(spec, frames(objects))
        self.assertTrue(report["hard_pass"], report)
        self.assertIn("collision_unverified", codes(report, "warnings"))
        self.assertEqual(report["metrics"]["exact_collision_checks"], 0)

    def test_nonexact_collider_cannot_trigger_precise_collision_failure(self):
        spec = spec_for("a", "b")
        spec["collision"] = {"forbidden_pairs": [["a", "b"]]}
        objects = [obj("a"), obj("b")]
        objects[0]["collider"]["exact"] = False
        report = evaluate(spec, frames(objects))
        self.assertTrue(report["hard_pass"], report)
        self.assertIn("collision_unverified", codes(report, "warnings"))

    def test_collision_is_not_enforced_outside_actor_active_interval(self):
        spec = spec_for("a", "b")
        spec["subjects"][1]["active_intervals"] = [[0.5, 1.5]]
        spec["collision"] = {"forbidden_pairs": [["a", "b"]]}
        scene = [
            frame(0, [obj("a"), obj("b")]),
            frame(1, [obj("a"), obj("b", sphere((4, 1, 0)))]),
        ]
        self.assertTrue(evaluate(spec, scene)["hard_pass"])

    def test_nonfinite_or_malformed_manifests_fail_closed(self):
        edits = [
            lambda m: m["camera"].update(position=[0, float("nan"), 8]),
            lambda m: m["objects"][0]["world_aabb"].update(min=[float("-inf"), 0, 0]),
            lambda m: m["objects"][0].update(projected_area_ratio="0.1"),
            lambda m: m["objects"][0].update(visible="true"),
            lambda m: m.pop("audit"),
            lambda m: m["audit"].update(mesh_count=2),
            lambda m: m["objects"][0].update(collider={"type": "sphere", "exact": True}),
            lambda m: m["objects"][0].update(collider=obb(half=(1, -1, 1))),
            lambda m: m["camera"].update(quaternion=[0, 0, 0, 0]),
            lambda m: m["objects"][0].pop("collider"),
            lambda m: m.update(t=float("inf")),
            lambda m: m.update(extra={"unexpected_nan": float("nan")}),
        ]
        for index, edit in enumerate(edits):
            scene = frames()
            edit(scene[0])
            report = evaluate(manifests=scene)
            with self.subTest(case=index):
                self.assertFalse(report["hard_pass"])
                self.assertIn("manifest_malformed", codes(report))
                self.assertIn("missing_sample", codes(report))

    def test_invalid_exact_obb_axes_do_not_fall_back_to_safe_looking_warning(self):
        scene = frames()
        shape = obb()
        shape["axes"] = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
        scene[0]["objects"][0]["collider"] = shape
        self.assertIn("manifest_malformed", codes(evaluate(manifests=scene)))

    def test_invalid_spec_returns_report_instead_of_crashing(self):
        spec = spec_for()
        spec["duration_s"] = float("nan")
        self.assertIn("spec_invalid", codes(evaluate(spec)))

    def test_source_contract_rejects_wall_clock_random_and_owned_loops(self):
        cases = [
            ("requestAnimationFrame(update);", "source_animation_loop"),
            ("renderer.setAnimationLoop(update);", "source_animation_loop"),
            ("setInterval(update, 10);", "source_timer"),
            ("const x=performance.now();", "source_wall_clock"),
            ("const x=Math.random();", "source_unseeded_random"),
            ("window.renderAt = myCapture;", "source_hook_override"),
            ("window['exportManifest'] = fake;", "source_hook_override"),
            ("import x from 'https://cdn.invalid/three.js';", "source_remote_dependency"),
        ]
        for extra, expected in cases:
            source = SOURCE.replace("</script>", extra + "</script>")
            with self.subTest(extra=extra):
                self.assertIn(expected, codes(evaluate(source=source)))

    def test_remote_import_map_is_rejected(self):
        source = '<script type="importmap">{"imports":{"three":"https://cdn.invalid/three.js"}}</script>' + SOURCE
        self.assertIn("source_remote_dependency", codes(evaluate(source=source)))

    def test_comments_and_plain_strings_do_not_trigger_source_lint(self):
        source = SOURCE.replace("</script>", """
// Do not call requestAnimationFrame(update) or Math.random().
const label = "Date.now() performance.now() setTimeout(foo, 1)";
/* window.renderAt = fake; */
</script>""")
        report = evaluate(source=source)
        self.assertTrue(report["hard_pass"], report)

    def test_missing_source_is_reported_as_unchecked_not_fabricated_success(self):
        report = evaluate(source="")
        self.assertTrue(report["hard_pass"], report)
        self.assertIn("source_not_checked", codes(report, "warnings"))

    def test_every_failure_and_warning_has_actionable_repair(self):
        spec = spec_for()
        spec["subjects"][0]["fully_in_frame"] = True
        scene = frames()
        hide_projection(scene[0]["objects"][0])
        report = evaluate(spec, scene, runtime_errors=["TypeError: broken"], source="")
        for issue in report["failures"] + report["warnings"]:
            self.assertTrue(issue["repair"])
            self.assertTrue(issue["code"])
            self.assertTrue(issue["message"])


if __name__ == "__main__":
    unittest.main()
