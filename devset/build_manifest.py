"""Rebuild the original, pinned 6/24/6 task suite after an intentional revision.

This is a dataset authoring utility, not part of a model or admission run.
Changing any prompt/spec creates a different manifest hash and invalidates old
paired comparisons.  None of these prompts were copied from CodeVideoBench.
"""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from proxy_harness.outerloop import canonical_sha256  # noqa: E402
from proxy_harness.spec import validate_spec  # noqa: E402


def case(name, prompt, ids, coverage, *, allowed=(), camera=None, active=None):
    return dict(name=name, prompt=prompt, ids=ids, coverage=coverage,
                allowed=list(allowed), camera=camera, active=active or {})


TRAIN = [
    case("static_pair", "Show a red sphere on the left and a blue cube on the right, separated on a plain floor. Both remain stationary. Use a fixed three-quarter camera; show both complete silhouettes throughout.", ["sphere", "cube"], ["registration", "framing", "basic_geometry"]),
    case("straight_cart", "A simple yellow cart travels smoothly from left to right beside a stationary cone. It slows to a stop before reaching the right edge. Keep the cart and cone fully visible, with no contact and a fixed camera.", ["cart", "cone"], ["linear_motion", "boundary_framing", "stopping"]),
    case("orbit_marker", "A small sphere completes one horizontal orbit around a stationary cylinder. Leave a clear gap between the objects. Use an elevated fixed camera so the orbit and both complete objects remain easy to see.", ["orbiter", "cylinder"], ["periodic_motion", "separation", "absolute_time"]),
    case("gentle_dolly", "A stationary orange cube sits on a floor beside a small blue sphere. The camera moves gently forward for three seconds without changing its aim abruptly. Both objects remain completely in frame and neither becomes an extreme close-up.", ["cube", "sphere"], ["camera_dolly", "screen_coverage", "target_tracking"]),
    case("staggered_lift", "Two separated blocks start on the floor. The left block rises smoothly during the first half; the right block rises during the second half. Both finish suspended at the same height. Use a fixed camera and keep both blocks fully visible.", ["left_block", "right_block"], ["timing", "multiple_subjects", "independent_state"]),
    case("roll_and_stop", "A ball rolls across a flat floor toward a stationary cube, then stops with a clearly visible gap. The ball rotates consistently with its forward motion. Keep the whole ball and cube visible in a stable side three-quarter view.", ["ball", "cube"], ["rolling", "collision_clearance", "stop_timing"]),
]


DEV = [
    case("three_silhouettes", "Place a sphere, a cone and a tall box side by side on a floor, with clear gaps and distinct silhouettes. Let each rotate slowly in place. The fixed camera must retain all three complete objects for the whole clip.", ["sphere", "cone", "box"], ["layout", "subject_completeness", "silhouette"]),
    case("depth_stagger", "Arrange three boxes at near, middle and far depths, offset horizontally so no box is hidden by another. They remain still while a fixed oblique camera makes the depth arrangement clear. Keep every box wholly visible.", ["near_box", "middle_box", "far_box"], ["depth_order", "framing", "multiple_subjects"]),
    case("wide_composition", "A large cube stands on the left and a small sphere on the right. They slowly move slightly farther apart and then return. Preserve a visible margin around both objects and keep the small sphere legible throughout.", ["large_cube", "small_sphere"], ["screen_coverage", "boundary_framing", "scale_contrast"]),
    case("foreground_landmark", "A short cylinder remains near the front of the scene while a cube moves in a shallow arc behind it. Offset the path so the cylinder never hides the cube. Keep both complete subjects visible from a fixed camera.", ["cylinder", "cube"], ["occlusion_avoidance", "depth", "curved_motion"]),
    case("curved_car", "A simple car follows a smooth quarter-circle road around a stationary pillar. Its heading follows the road tangent. Neither car nor pillar should collide, leave the frame or disappear. Use a fixed elevated camera.", ["car", "pillar"], ["trajectory", "orientation", "clearance"]),
    case("figure_eight", "A sphere follows a smooth figure-eight path on a horizontal plane. A stationary cube sits well outside the path as a scale reference. Keep both fully visible; the sphere should move continuously through the crossing.", ["sphere", "reference_cube"], ["trajectory", "continuity", "absolute_time"]),
    case("pause_resume", "A box moves from left to right, pauses visibly around the middle second, then resumes. A stationary cone remains behind and to one side, separated from the path. Keep both complete objects visible with a fixed camera.", ["moving_box", "cone"], ["temporal_control", "pause", "continuity"]),
    case("bounce_near_marker", "A ball makes two smooth diminishing bounces on a flat floor beside a stationary cube. Contact with the floor should be visually clear; the ball must never enter the cube. Keep the entire bounce and both subjects in a fixed frame.", ["ball", "cube"], ["vertical_motion", "event_timing", "floor_contact"]),
    case("camera_orbit", "A sphere and a tall box stay motionless with a clear gap between them. The camera makes a slow quarter orbit around the pair while always showing both complete objects. The objects themselves must not drift.", ["sphere", "box"], ["camera_orbit", "world_state", "framing"]),
    case("low_tracking", "A small box-shaped vehicle travels forward along a straight floor path beside a stationary cone. A low camera tracks the vehicle gently, keeping both vehicle and cone completely visible. Avoid a top-down angle and any collision.", ["vehicle", "cone"], ["low_camera", "tracking", "camera_height"], camera={"pitch_deg_range": [-20, 15], "height_range": [0.5, 3]}),
    case("high_follow", "A sphere moves along a shallow S-shaped path beside a stationary box. An elevated camera follows gently from above while keeping both complete objects in view. The scene should read as a clear overhead oblique view, not a ground-level shot.", ["sphere", "box"], ["high_camera", "tracking", "camera_pitch"], camera={"pitch_deg_range": [-70, -25], "height_range": [4, 12]}),
    case("two_camera_shots", "Two stationary objects, a sphere and a cone, are shown first from the front-left and then from the front-right, with one clean camera cut at 1.5 seconds. Keep both objects fully visible in each shot; their world positions and sizes remain unchanged.", ["sphere", "cone"], ["camera_cut", "state_preservation", "timing"]),
    case("approach_contact", "A sphere rolls toward a stationary box and comes to rest at its side without passing through it. Show the approach, the contact and the settled state. Use a fixed camera that keeps both complete objects visible.", ["sphere", "box"], ["contact", "causality", "nonpenetration"], allowed=[("sphere", "box")]),
    case("passing_lanes", "Two box-shaped vehicles travel in opposite directions in two clearly separated parallel lanes. They pass one another without touching. Keep both complete vehicles visible throughout with a fixed elevated camera.", ["vehicle_a", "vehicle_b"], ["multiple_motion", "collision_clearance", "direction"]),
    case("ramp_roll", "A ball rolls down a broad simple ramp from its upper part to its lower part. Keep the ball visually on the ramp and show its rolling rotation. A fixed oblique camera must show the entire ramp and ball for the whole clip.", ["ball", "ramp"], ["support_contact", "inclined_motion", "geometry"], allowed=[("ball", "ramp")]),
    case("pendulum", "A pendulum bob swings twice beneath a small fixed support. Make the supporting connection readable and keep the bob attached at a constant distance from its pivot. Keep both bob and support fully in frame with a stable side-oblique camera.", ["bob", "support"], ["articulation", "periodic_motion", "connection"]),
    case("door_opening", "A rectangular door panel opens smoothly from closed to a right angle inside a simple freestanding frame. The panel rotates about its side hinge and never detaches. Show the complete panel and frame throughout from a fixed oblique view.", ["door_panel", "door_frame"], ["hinge", "articulation", "structural_completeness"], allowed=[("door_panel", "door_frame")]),
    case("drawer_slide", "A drawer slides outward from a simple open-front cabinet and then stops. Keep the cabinet fixed and make the drawer's opening direction clear. Show the complete cabinet and drawer from a fixed three-quarter camera.", ["drawer", "cabinet"], ["relative_motion", "containment", "stopping"], allowed=[("drawer", "cabinet")]),
    case("cargo_lift", "A low platform raises a small cube smoothly upward. The cube stays resting on the platform with no sliding or levitation relative to it. Keep both complete subjects visible throughout the lift from a fixed camera.", ["platform", "cargo"], ["support_contact", "coupled_motion", "relative_state"], allowed=[("platform", "cargo")]),
    case("turntable", "A rectangular box sits off-center on a round turntable. The turntable rotates slowly through half a turn and carries the box with it. Preserve the contact and relative placement. Keep the full box and turntable visible from a fixed elevated view.", ["turntable", "box"], ["parent_transform", "rotation", "support_contact"], allowed=[("turntable", "box")]),
    case("counter_rotation", "An elongated box rotates slowly clockwise in place while the camera moves a short distance counterclockwise around it. A stationary sphere to one side provides a world reference. Keep both complete objects visible and separated.", ["box", "reference_sphere"], ["camera_object_decoupling", "world_reference", "rotation"]),
    case("crossing_heights", "Two spheres follow crossing horizontal paths at clearly different heights, so they pass without touching. Use an oblique camera that makes the height separation obvious. Keep both complete spheres visible throughout.", ["upper_sphere", "lower_sphere"], ["3d_clearance", "multiple_motion", "depth_cues"]),
    case("scheduled_entry", "A stationary cube is visible for the entire clip. A sphere enters smoothly from above around 1 second and then moves beside it without contact. Before 1 second the sphere may be outside the frame; from 1 second onward both complete objects must be visible.", ["cube", "sphere"], ["active_intervals", "entry_timing", "framing"], active={"sphere": [[1, 3]]}),
    case("reversible_scale", "A cube grows smoothly to about one and a half times its starting size during the first half, then shrinks back during the second half while staying planted on the floor. A stationary sphere remains beside it. Keep both complete objects visible and separated.", ["cube", "sphere"], ["absolute_state", "scale_change", "grounding"]),
]


HOLDOUT = [
    case("carousel_pair", "Two different small boxes are carried on opposite sides of a rotating circular platform. The platform makes a quarter turn while a fixed elevated camera keeps the complete assembly visible. The boxes retain their spacing and stay supported.", ["platform", "box_a", "box_b"], ["coupled_motion", "multiple_subjects", "support"], allowed=[("platform", "box_a"), ("platform", "box_b")]),
    case("offset_crossing", "A sphere moves left-to-right in front while a cube moves right-to-left farther back. Time the motions so they pass at the middle of the clip without touching or hiding one another. Keep both complete subjects visible from a slightly elevated fixed camera.", ["sphere", "cube"], ["trajectory", "depth_separation", "timing"]),
    case("lift_and_orbit", "A small cube rises vertically beside a stationary cylinder while the camera makes a gentle quarter orbit. Show both complete subjects throughout. The cube's path stays vertical in world space and remains separated from the cylinder.", ["cube", "cylinder"], ["camera_object_decoupling", "vertical_motion", "framing"]),
    case("hinged_barrier", "A simple barrier arm rotates upward around one end on a short post. A stationary sphere beside the post provides scale. Keep the arm attached and show the complete arm, post and sphere from a fixed oblique view.", ["arm", "post", "sphere"], ["articulation", "structural_completeness", "scale"], allowed=[("arm", "post")]),
    case("follow_stop_return", "A box vehicle moves forward, stops briefly, and then reverses to near its initial position beside a stationary cone. The camera follows gently without losing either complete subject. Preserve the vehicle's heading while reversing and avoid contact.", ["vehicle", "cone"], ["temporal_control", "tracking", "reverse_motion"]),
    case("expanding_triangle", "Three spheres start at the corners of a small triangle on a horizontal plane. They move smoothly outward to enlarge the triangle and then settle. Keep their relative order, avoid contact and show all three complete spheres from a fixed elevated camera.", ["sphere_a", "sphere_b", "sphere_c"], ["multiple_subjects", "layout_change", "boundary_framing"]),
]


def build(destination=ROOT / "devset"):
    destination = Path(destination)
    tasks = []
    for split, entries, first_seed in (("train", TRAIN, 1100), ("dev", DEV, 2100), ("holdout", HOLDOUT, 3100)):
        for index, entry in enumerate(entries, 1):
            task_id = f"{split}_{index:02d}_{entry['name']}"
            allowed = [list(pair) for pair in entry["allowed"]]
            allowed_set = {tuple(sorted(pair)) for pair in allowed}
            subjects = []
            for subject_id in entry["ids"]:
                subject = {"id": subject_id, "role": "primary", "required": True,
                           "fully_in_frame": True, "min_area_ratio": 0.005, "max_area_ratio": 0.7}
                if subject_id in entry["active"]:
                    subject["active_intervals"] = entry["active"][subject_id]
                subjects.append(subject)
            spec = validate_spec({
                "schema_version": "1.0", "task_id": task_id, "prompt": entry["prompt"],
                "split": split, "seed": first_seed + index, "duration_s": 3, "fps": 12,
                "width": 640, "height": 360, "sample_times": [0, 0.5, 1, 1.5, 2, 2.5, 2.916667],
                "subjects": subjects,
                "camera": entry["camera"] or {"pitch_deg_range": [-70, 15], "height_range": [0.5, 12]},
                "collision": {"forbidden_pairs": [list(pair) for pair in combinations(entry["ids"], 2)
                               if tuple(sorted(pair)) not in allowed_set], "allowed_pairs": allowed, "tolerance": 0.001},
                "evaluation": {"pass_score": 75, "hard_fail_score_cap": 40},
            })
            relative = f"{split}/{task_id}.spec.json"
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tasks.append({"task_id": task_id, "split": split, "seed": spec["seed"],
                          "spec_path": relative, "spec_sha256": canonical_sha256(spec), "coverage": entry["coverage"]})
    manifest = {
        "schema_version": "1.0", "dataset_id": "proxy-harness-original-36-v1",
        "provenance": "Original synthetic scene specifications authored for this project; not CodeVideoBench tasks or an official benchmark reproduction.",
        "hash_policy": "SHA-256 over canonical UTF-8 JSON: sorted keys, comma/colon separators, ensure_ascii=False, no NaN. Spec hashes use validate_spec normalized objects.",
        "split_policy": {"train": "6 cases; the only source of distillation evidence", "dev": "24 cases; complete paired admission only", "holdout": "6 cases; reserved final evaluation, never distillation or admission"},
        "tasks": tasks,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = canonical_sha256(manifest)
    (destination / "manifest.sha256").write_text(digest + "  manifest.json\n", encoding="ascii")
    return digest


if __name__ == "__main__":
    print(build())
