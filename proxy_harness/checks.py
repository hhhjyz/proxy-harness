"""Evidence-based hard gates for proxy renders.

Only explicitly forbidden pairs receive collision checks. Broad-phase AABB
overlap is never classified as penetration. Hard collision failures require
validated exact sphere or OBB colliders derived by the trusted runtime.
"""

from __future__ import annotations

from html.parser import HTMLParser
import json
import math
import re
from typing import Any

from .spec import is_active, validate_spec


_EPS = 1e-8
_GEOMETRY_EPS = 1e-7
_SCREEN_ROLES = {"primary", "secondary"}
_PASSIVE_ROLES = {"ground", "decoration", "background"}


def _issue(code: str, message: str, repair: str, **context: Any) -> dict:
    return {"code": code, "message": message, "repair": repair, **context}


def _finite(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _vec(value: Any, size: int) -> bool:
    return isinstance(value, list) and len(value) == size and all(_finite(x) for x in value)


def _dot(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: list, b: list) -> list:
    return [x - y for x, y in zip(a, b)]


def _norm(a: list) -> float:
    return math.hypot(*a)


def _cross(a: list, b: list) -> list:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _finite_json(value: Any, path: str = "manifest", seen: set | None = None) -> str | None:
    """Locate non-JSON, cyclic, or non-finite evidence before doing arithmetic."""
    if value is None or isinstance(value, (str, bool)):
        return None
    if isinstance(value, (int, float)):
        return None if _finite(value) else f"{path} contains a non-finite or unrepresentable number"
    if not isinstance(value, (list, dict)):
        return f"{path} is not a JSON value"
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return f"{path} is cyclic"
    seen.add(identity)
    iterator = value.items() if isinstance(value, dict) else enumerate(value)
    for key, item in iterator:
        if isinstance(value, dict) and not isinstance(key, str):
            return f"{path} has a non-string key"
        result = _finite_json(item, f"{path}.{key}", seen)
        if result:
            return result
    seen.remove(identity)
    return None


def _valid_box(box: Any, dimensions: int) -> bool:
    return (
        isinstance(box, dict)
        and _vec(box.get("min"), dimensions)
        and _vec(box.get("max"), dimensions)
        and all(low <= high for low, high in zip(box["min"], box["max"]))
    )


def _collider_problem(collider: Any) -> str | None:
    if collider is None:
        return None
    if not isinstance(collider, dict):
        return "collider must be an object or null"
    if not isinstance(collider.get("type"), str):
        return "collider.type must identify its geometry"
    if "exact" in collider and not isinstance(collider["exact"], bool):
        return "collider.exact must be a boolean"
    if collider.get("exact") is not True:
        return None
    kind = collider["type"]
    if kind not in {"sphere", "obb"}:
        return None  # Unknown exact shape still cannot be certified by this checker.
    if not _vec(collider.get("center"), 3):
        return "exact collider.center must have three finite coordinates"
    if kind == "sphere":
        if not _finite(collider.get("radius")) or collider["radius"] < 0:
            return "exact sphere.radius must be a finite nonnegative number"
    else:
        if not _vec(collider.get("half_extents"), 3) or any(x < 0 for x in collider["half_extents"]):
            return "exact obb.half_extents must be finite and nonnegative"
        axes = collider.get("axes")
        if not isinstance(axes, list) or len(axes) != 3 or any(not _vec(axis, 3) for axis in axes):
            return "exact obb.axes must be three finite 3-vectors"
        if any(abs(_norm(axis) - 1) > 1e-4 for axis in axes):
            return "exact obb axes must have unit length"
        if any(abs(_dot(axes[i], axes[j])) > 1e-4 for i in range(3) for j in range(i)):
            return "exact obb axes must be mutually orthogonal"
    return None


def _manifest_problems(manifest: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    try:
        problem = _finite_json(manifest)
    except RecursionError:
        problem = "manifest nesting exceeds the supported depth"
    if problem:
        return [problem]
    if manifest.get("schema_version") != "1.0":
        problems.append("manifest.schema_version must be '1.0'")
    if not _finite(manifest.get("t")):
        problems.append("manifest.t must be finite")

    camera = manifest.get("camera")
    if not isinstance(camera, dict):
        problems.append("camera must be an object")
    else:
        for field, count in (("position", 3), ("quaternion", 4), ("forward", 3)):
            if not _vec(camera.get(field), count):
                problems.append(f"camera.{field} must be a finite {count}-vector")
        for field in ("pitch_deg", "fov_deg", "near", "far"):
            if not _finite(camera.get(field)):
                problems.append(f"camera.{field} must be finite")
        if _vec(camera.get("quaternion"), 4) and abs(_norm(camera["quaternion"]) - 1) > 1e-4:
            problems.append("camera.quaternion must be normalized")
        if _vec(camera.get("forward"), 3):
            if abs(_norm(camera["forward"]) - 1) > 1e-4:
                problems.append("camera.forward must be normalized")
            elif _finite(camera.get("pitch_deg")):
                actual_pitch = math.degrees(math.asin(max(-1, min(1, camera["forward"][1]))))
                if abs(actual_pitch - camera["pitch_deg"]) > 1e-3:
                    problems.append("camera.pitch_deg disagrees with the signed elevation of camera.forward")
        if _finite(camera.get("fov_deg")) and not 0 < camera["fov_deg"] < 180:
            problems.append("camera.fov_deg must be inside (0, 180)")
        if _finite(camera.get("near")) and camera["near"] <= 0:
            problems.append("camera.near must be positive")
        if _finite(camera.get("near")) and _finite(camera.get("far")) and camera["far"] <= camera["near"]:
            problems.append("camera.far must exceed camera.near")

    objects = manifest.get("objects")
    if not isinstance(objects, list):
        problems.append("objects must be an array")
    else:
        seen: set[str] = set()
        for index, obj in enumerate(objects):
            prefix = f"objects[{index}]"
            if not isinstance(obj, dict):
                problems.append(f"{prefix} must be an object")
                continue
            subject_id = obj.get("id")
            if not isinstance(subject_id, str) or not subject_id.strip():
                problems.append(f"{prefix}.id must be nonempty")
            elif subject_id in seen:
                problems.append(f"{prefix}.id duplicates {subject_id!r}")
            else:
                seen.add(subject_id)
            if not isinstance(obj.get("role"), str) or not obj["role"]:
                problems.append(f"{prefix}.role must be nonempty")
            if not _valid_box(obj.get("world_aabb"), 3):
                problems.append(f"{prefix}.world_aabb must have finite ordered min/max 3-vectors")
            for flag in ("frustum_intersects", "in_front", "fully_in_frame", "projection_valid", "visible"):
                if not isinstance(obj.get(flag), bool):
                    problems.append(f"{prefix}.{flag} must be a boolean")
            if not _integer(obj.get("primitive_count")):
                problems.append(f"{prefix}.primitive_count must be a nonnegative integer")
            if "collider" not in obj:
                problems.append(f"{prefix}.collider must be present (null for unsupported geometry)")
            else:
                collider_error = _collider_problem(obj["collider"])
                if collider_error:
                    problems.append(f"{prefix}: {collider_error}")
            if obj.get("projection_valid") is True:
                if not _valid_box(obj.get("ndc_box"), 2):
                    problems.append(f"{prefix}.ndc_box must have finite ordered min/max 2-vectors for a valid projection")
                if not _finite(obj.get("projected_area_ratio")) or obj["projected_area_ratio"] < 0:
                    problems.append(f"{prefix}.projected_area_ratio must be finite and nonnegative")
                if obj.get("in_front") is not True:
                    problems.append(f"{prefix} claims a valid projection while in_front is false")
            elif obj.get("projection_valid") is False:
                if obj.get("ndc_box") is not None and not _valid_box(obj["ndc_box"], 2):
                    problems.append(f"{prefix}.ndc_box must be null or a finite box")
                if obj.get("projected_area_ratio") is not None and (
                    not _finite(obj["projected_area_ratio"]) or obj["projected_area_ratio"] < 0
                ):
                    problems.append(f"{prefix}.projected_area_ratio must be null or finite and nonnegative")
                if obj.get("fully_in_frame") is True:
                    problems.append(f"{prefix} cannot be fully_in_frame with an invalid projection")
            if obj.get("fully_in_frame") is True:
                if obj.get("frustum_intersects") is not True:
                    problems.append(f"{prefix} cannot be fully_in_frame outside the frustum")
                if _valid_box(obj.get("ndc_box"), 2):
                    if any(x < -1 - 1e-5 for x in obj["ndc_box"]["min"]) or any(x > 1 + 1e-5 for x in obj["ndc_box"]["max"]):
                        problems.append(f"{prefix}.fully_in_frame disagrees with ndc_box")

    audit = manifest.get("audit")
    if not isinstance(audit, dict):
        problems.append("audit must be an object")
    else:
        fields = ("mesh_count", "registered_mesh_count", "unregistered_mesh_count", "primitive_count")
        for field in fields:
            if not _integer(audit.get(field)):
                problems.append(f"audit.{field} must be a nonnegative integer")
        if all(_integer(audit.get(field)) for field in fields[:3]):
            if audit["registered_mesh_count"] + audit["unregistered_mesh_count"] != audit["mesh_count"]:
                problems.append("audit registered + unregistered mesh counts must equal mesh_count")
        if _integer(audit.get("primitive_count")) and isinstance(objects, list):
            for obj in objects:
                if isinstance(obj, dict) and _integer(obj.get("primitive_count")) and obj["primitive_count"] > audit["primitive_count"]:
                    problems.append("a subject primitive_count exceeds the actual scene primitive_count")
    return problems


def _aabb_penetrates(a: dict, b: dict, tolerance: float) -> bool:
    return all(min(a["max"][i], b["max"][i]) - max(a["min"][i], b["min"][i]) > tolerance + _GEOMETRY_EPS for i in range(3))


def _obb_penetration(a: dict, b: dict) -> float:
    """Minimum overlap on all SAT axes; nonpositive means disjoint/contact."""
    delta = _sub(b["center"], a["center"])
    axes = list(a["axes"]) + list(b["axes"])
    axes.extend(_cross(x, y) for x in a["axes"] for y in b["axes"])
    minimum = math.inf
    for candidate in axes:
        length = _norm(candidate)
        if length < 1e-10:  # Parallel edges give a zero cross-product axis.
            continue
        axis = [x / length for x in candidate]
        radius_a = sum(extent * abs(_dot(base, axis)) for base, extent in zip(a["axes"], a["half_extents"]))
        radius_b = sum(extent * abs(_dot(base, axis)) for base, extent in zip(b["axes"], b["half_extents"]))
        overlap = radius_a + radius_b - abs(_dot(delta, axis))
        if overlap <= 0:
            return overlap
        minimum = min(minimum, overlap)
    return minimum


def _sphere_obb_penetration(sphere: dict, box: dict) -> float:
    delta = _sub(sphere["center"], box["center"])
    local = [_dot(delta, axis) for axis in box["axes"]]
    outside = [max(abs(value) - extent, 0) for value, extent in zip(local, box["half_extents"])]
    distance = _norm(outside)
    if distance > 0:
        return sphere["radius"] - distance
    # An interior center needs to cross its nearest box face plus its radius.
    return sphere["radius"] + min(extent - abs(value) for value, extent in zip(local, box["half_extents"]))


def _exact_penetration(a: Any, b: Any) -> tuple[float, str] | None:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    if a.get("exact") is not True or b.get("exact") is not True:
        return None
    kinds = (a.get("type"), b.get("type"))
    if kinds == ("sphere", "sphere"):
        return a["radius"] + b["radius"] - _norm(_sub(a["center"], b["center"])), "sphere_sphere"
    if kinds == ("obb", "obb"):
        return _obb_penetration(a, b), "obb_sat"
    if kinds == ("sphere", "obb"):
        return _sphere_obb_penetration(a, b), "sphere_obb"
    if kinds == ("obb", "sphere"):
        return _sphere_obb_penetration(b, a), "sphere_obb"
    return None


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.scripts: list[tuple[dict, str]] = []
        self._attrs: dict | None = None
        self._body: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() == "script":
            self._attrs = dict(attrs)
            self._body = []

    def handle_data(self, data: str) -> None:
        if self._attrs is not None:
            self._body.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._attrs is not None:
            self.scripts.append((self._attrs, "".join(self._body)))
            self._attrs = None
            self._body = []


def _lexical_js(source: str, *, mask_strings: bool) -> str:
    """Remove comments and optionally string bodies for narrow contract lint."""
    output: list[str] = []
    index = 0
    length = len(source)
    quote: str | None = None
    while index < length:
        char = source[index]
        if quote is not None:
            if char == "\\" and index + 1 < length:
                output.append("  " if mask_strings else source[index:index + 2])
                index += 2
                continue
            if char == quote:
                output.append(" " if mask_strings else char)
                quote = None
            else:
                output.append("\n" if char == "\n" else (" " if mask_strings else char))
            index += 1
            continue
        if char in ("'", '"', chr(96)):
            quote = char
            output.append(" " if mask_strings else char)
            index += 1
        elif source.startswith("//", index):
            end = source.find("\n", index)
            end = length if end == -1 else end
            output.append(" " * (end - index))
            index = end
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            end = length if end == -1 else end + 2
            output.append("".join("\n" if c == "\n" else " " for c in source[index:end]))
            index = end
        else:
            output.append(char)
            index += 1
    return "".join(output)


def _lint_source(source: str) -> list[dict]:
    if not source:
        return []
    parser = _Scripts()
    parser.feed(source)
    scripts = [body for attrs, body in parser.scripts if attrs.get("type", "").lower() != "importmap"]
    javascript = "\n".join(scripts) if parser.scripts else source
    code = _lexical_js(javascript, mask_strings=True)
    literal_code = _lexical_js(javascript, mask_strings=False)
    problems: list[dict] = []
    if not re.search(r"\bcreateHarness\s*\(", code):
        problems.append(_issue(
            "source_harness_missing", "Scene source does not initialize createHarness(...).",
            "Import createHarness from ./runtime.js and pass scene, camera, renderer, update, subjects, and seed.",
        ))
    checks = [
        (r"\brequestAnimationFrame\s*\(|\.\s*setAnimationLoop\s*\(", "source_animation_loop", "Scene owns a continuous animation loop.", "Remove the loop; let the capture controller call renderAt(t), and derive state in update(t)."),
        (r"\b(?:setInterval|setTimeout)\s*\(", "source_timer", "Scene uses a wall-clock timer.", "Remove scene timers; express event timing as explicit conditions on update(t)."),
        (r"\b(?:Date|performance)\s*\.\s*now\s*\(|\bnew\s+(?:THREE\s*\.\s*)?Clock\s*\(|\bnew\s+Date\s*\(", "source_wall_clock", "Scene reads wall-clock time.", "Use only the supplied absolute t for animation; keep wall-clock work outside scene code."),
        (r"\bMath\s*\.\s*random\s*\(", "source_unseeded_random", "Scene calls unseeded Math.random().", "Use the provided seeded random helper and initialize random scene parameters once."),
        (r"\b(?:window|globalThis)\s*\.\s*(?:renderAt|exportManifest|renderTopAt|__harnessReady)\s*=(?!=)", "source_hook_override", "Scene attempts to replace a trusted capture hook.", "Keep capture hooks owned by runtime.js; edit only update(t) and registered scene objects."),
        (r"\b(?:window|globalThis)\s*\[\s*['\"](?:renderAt|exportManifest|renderTopAt|__harnessReady)['\"]\s*\]\s*=(?!=)", "source_hook_override", "Scene attempts to replace a trusted capture hook.", "Keep capture hooks owned by runtime.js; edit only update(t) and registered scene objects."),
    ]
    for pattern, code_name, message, repair in checks:
        text = literal_code if "\\[" in pattern else code
        if re.search(pattern, text):
            problems.append(_issue(code_name, message, repair))
    remote_scripts = [
        attrs.get("src", "") for attrs, _ in parser.scripts
        if re.match(r"^(?:https?:)?//", attrs.get("src", ""), re.IGNORECASE)
    ]
    remote_imports = re.findall(r"\b(?:from\s*|import\s*(?:\(\s*)?)[\"']((?:https?:)?//[^\"']+)[\"']", literal_code)
    for attrs, body in parser.scripts:
        if attrs.get("type", "").lower() != "importmap":
            continue
        try:
            import_map = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            problems.append(_issue("source_importmap_invalid", "Scene import map is not valid JSON.", "Restore the pinned local import map."))
            continue
        if isinstance(import_map, dict):
            mappings = [import_map.get("imports", {})]
            if isinstance(import_map.get("scopes"), dict):
                mappings.extend(import_map["scopes"].values())
            for mapping in mappings:
                if isinstance(mapping, dict):
                    remote_imports.extend(value for value in mapping.values() if isinstance(value, str) and re.match(r"^(?:https?:)?//", value, re.IGNORECASE))
    if remote_scripts or remote_imports:
        problems.append(_issue(
            "source_remote_dependency", "Scene imports an unpinned remote script or module.",
            "Use the local pinned import map and ./runtime.js, ./primitives.js, and ./vendor/ assets.",
            dependencies=sorted(set(remote_scripts + remote_imports)),
        ))
    return problems


def _check_determinism(determinism: Any, duration: float) -> list[dict]:
    repair = "Repeat frame and manifest capture at the same times, including after out-of-order seeks, and return the actual comparisons."
    if determinism is None:
        return [_issue("determinism_missing", "No determinism evidence was supplied.", repair)]
    if not isinstance(determinism, dict) or not isinstance(determinism.get("passed"), bool):
        return [_issue("determinism_malformed", "Determinism report must contain a boolean passed field.", repair)]
    try:
        malformed = _finite_json(determinism, "determinism")
    except RecursionError:
        malformed = "determinism nesting exceeds the supported depth"
    if malformed:
        return [_issue("determinism_malformed", malformed, repair)]
    comparisons = determinism.get("comparisons")
    times = determinism.get("tested_times")
    if (
        not isinstance(comparisons, list) or not comparisons
        or not isinstance(times, list) or not times
        or any(not _finite(t) or not 0 <= t < duration for t in times)
        or any(not isinstance(item, dict) or any(not isinstance(item.get(flag), bool) for flag in ("passed", "pixels_equal", "manifest_equal")) for item in comparisons)
    ):
        return [_issue("determinism_malformed", "Determinism needs finite in-range tested_times and comparisons containing boolean passed, pixels_equal, and manifest_equal fields.", repair)]
    actual_passes = [item["pixels_equal"] and item["manifest_equal"] for item in comparisons]
    if any(item["passed"] != actual for item, actual in zip(comparisons, actual_passes)) or determinism["passed"] != all(actual_passes):
        return [_issue("determinism_inconsistent", "Determinism pass flags disagree with pixel/manifest comparison evidence.", repair)]
    if determinism["passed"] is not True or any(item["passed"] is not True for item in comparisons):
        return [_issue("determinism_failed", "Repeated capture does not reproduce the same frame and manifest.", "Replace accumulated position/rotation/time updates with absolute functions of t, reset simulation state before seeking, and seed randomness.")]
    return []


def check_scene(
    spec: dict,
    manifests: list[dict],
    *,
    runtime_errors: list[str] = (),
    determinism: dict | None = None,
    source: str = "",
) -> dict:
    """Return hard gates, warnings, measured metrics, and actionable repairs.

    A passing result concerns the supplied sample times and verified geometry.
    It does not certify unsampled events, occlusion-free visibility, arbitrary
    mesh collision, physical correctness, or downstream generative-video quality.
    """
    failures: list[dict] = []
    warnings: list[dict] = []
    metrics: dict = {"manifest_count": 0, "valid_manifest_count": 0, "exact_collision_checks": 0, "unverified_collision_checks": 0}
    report = {"schema_version": "1.0", "hard_pass": False, "failures": failures, "warnings": warnings, "metrics": metrics}
    try:
        spec = validate_spec(spec)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        failures.append(_issue("spec_invalid", str(exc), "Correct the scene specification before generating or evaluating a scene."))
        return report
    metrics["requested_sample_count"] = len(spec["sample_times"])
    metrics["required_subject_count"] = sum(subject["required"] for subject in spec["subjects"])
    metrics["determinism_passed"] = False
    if not isinstance(runtime_errors, (list, tuple)) or any(not isinstance(item, str) for item in runtime_errors):
        failures.append(_issue("runtime_errors_malformed", "Runtime error evidence must be an array of strings.", "Return actual browser, asset-load, and render exceptions as strings."))
    else:
        for error in runtime_errors:
            failures.append(_issue("runtime_error", error or "The runtime reported an unspecified error.", "Fix the reported JavaScript, asset-load, browser, or rendering error and render again."))
    deterministic_errors = _check_determinism(determinism, spec["duration_s"])
    failures.extend(deterministic_errors)
    metrics["determinism_passed"] = not deterministic_errors
    if not isinstance(source, str):
        failures.append(_issue("source_malformed", "Source must be a string.", "Pass the actual scene.html text to the checker."))
    elif source:
        failures.extend(_lint_source(source))
    else:
        warnings.append(_issue("source_not_checked", "No source was supplied for capture-contract lint.", "Pass scene.html to check_scene; runtime evidence is still checked independently."))

    if not isinstance(manifests, list):
        failures.append(_issue("manifests_malformed", "Manifests must be an array.", "Export real manifests at every requested sample time."))
        return report
    metrics["manifest_count"] = len(manifests)
    valid: list[dict] = []
    for index, manifest in enumerate(manifests):
        problems = _manifest_problems(manifest)
        if problems:
            failures.append(_issue("manifest_malformed", "; ".join(problems), "Repair the runtime or scene so it exports complete finite geometry, projection, camera, and audit data; never replace missing evidence with guessed values.", manifest_index=index))
            continue
        t = manifest["t"]
        if not 0 <= t < spec["duration_s"]:
            failures.append(_issue("manifest_time_out_of_range", "Manifest time is outside [0, duration_s).", "Capture only times inside the specified duration.", t=t))
            continue
        valid.append(manifest)
    metrics["valid_manifest_count"] = len(valid)
    time_tolerance = max(1e-7, min(1e-5, spec["duration_s"] * 1e-7))
    seen_times: list[float] = []
    for manifest in valid:
        t = manifest["t"]
        if any(abs(t - previous) <= time_tolerance for previous in seen_times):
            failures.append(_issue("duplicate_manifest_time", "A sample time appears more than once.", "Keep one primary manifest per requested time; store repeat captures inside determinism evidence.", t=t))
        seen_times.append(t)
    for t in spec["sample_times"]:
        if not any(abs(t - actual) <= time_tolerance for actual in seen_times):
            failures.append(_issue("missing_sample", "No valid manifest exists for a requested sample.", "Render and export the missing sample; do not omit failed frames.", t=t))
    if len(spec["sample_times"]) == 1:
        warnings.append(_issue("single_temporal_sample", "Only one time is sampled, so temporal behavior is weakly observed.", "Add samples around the start, motion changes, interactions, and end."))

    subjects = {item["id"]: item for item in spec["subjects"]}
    areas: dict[str, list[float]] = {subject_id: [] for subject_id in subjects}
    primitives: list[int] = []
    camera_heights: list[float] = []
    camera_pitches: list[float] = []
    declared_ids = set(subjects)
    undeclared_reported: set[str] = set()
    tolerance = spec["collision"]["tolerance"]
    forbidden_pairs = spec["collision"]["forbidden_pairs"]
    metrics["forbidden_pair_count"] = len(forbidden_pairs)
    for manifest in valid:
        t = manifest["t"]
        camera = manifest["camera"]
        camera_heights.append(camera["position"][1])
        camera_pitches.append(camera["pitch_deg"])
        primitives.append(manifest["audit"]["primitive_count"])
        for config_name, value, name in (
            ("pitch_deg_range", camera["pitch_deg"], "pitch"),
            ("height_range", camera["position"][1], "height"),
        ):
            if config_name in spec["camera"]:
                low, high = spec["camera"][config_name]
                if value < low - _EPS or value > high + _EPS:
                    failures.append(_issue(f"camera_{name}_out_of_range", f"Camera {name} {value:.6g} is outside [{low}, {high}].", f"Adjust the camera {name} at this time to the configured range; preserve the specified camera convention.", t=t, measured=value, expected=[low, high]))
        audit = manifest["audit"]
        if audit["unregistered_mesh_count"] > 0:
            failures.append(_issue("unregistered_meshes", f"{audit['unregistered_mesh_count']} actual mesh(es) are absent from the subject registry.", "Register every rendered mesh under a stable subject id; register ground and decoration with their explicit roles.", t=t, count=audit["unregistered_mesh_count"]))
        by_id = {obj["id"]: obj for obj in manifest["objects"]}
        extras = set(by_id) - declared_ids
        for subject_id in sorted(extras - undeclared_reported):
            warnings.append(_issue("undeclared_subject", f"Registered subject {subject_id!r} is not described by the spec.", "Declare it in subjects if its behavior should be evaluated; background/ground/decoration can use passive roles.", subject=subject_id))
        undeclared_reported.update(extras)
        for subject_id, subject in subjects.items():
            active = is_active(subject, t)
            obj = by_id.get(subject_id)
            if obj is None:
                if subject["required"] and active:
                    failures.append(_issue("required_subject_missing", f"Required subject {subject_id!r} is missing.", "Create the subject and register its actual object with the exact required id before this active interval.", t=t, subject=subject_id))
                continue
            if obj["role"] != subject["role"]:
                failures.append(_issue("subject_role_mismatch", f"Subject role {obj['role']!r} disagrees with spec role {subject['role']!r}.", "Use the role declared in the immutable specification; do not change roles to bypass visibility checks.", t=t, subject=subject_id))
            if subject["required"] and active and obj["primitive_count"] == 0:
                failures.append(_issue("required_subject_empty", "A required subject has no actual renderable primitives.", "Attach actual mesh geometry to the registered subject; an empty Group does not satisfy presence.", t=t, subject=subject_id))
            if not active:
                continue
            if obj["projection_valid"] and obj["projected_area_ratio"] is not None:
                areas[subject_id].append(obj["projected_area_ratio"])
            explicit_screen = subject["fully_in_frame"] or "min_area_ratio" in subject or "max_area_ratio" in subject
            visual_obligation = subject["role"] in _SCREEN_ROLES or explicit_screen
            # Optional subjects may be absent; their screen issues are advisory.
            visual_issues = failures if subject["required"] else warnings
            if not visual_obligation:
                if not obj["projection_valid"]:
                    warnings.append(_issue("passive_projection_invalid", f"Passive subject projection is unavailable: {obj.get('projection_reason', 'not specified')}.", "Inspect the ground/background if relevant; add explicit screen constraints only when the task requires them.", t=t, subject=subject_id))
                continue
            checks = (
                ("frustum_intersects", "subject_outside_frustum", "Subject does not intersect the primary camera frustum.", "Move the subject or camera so the subject intersects the intended view during its active interval."),
                ("in_front", "subject_behind_camera", "Subject is not in front of the primary camera.", "Place the subject in front of the camera and outside the near clipping plane."),
                ("visible", "subject_not_visible", "Subject is not eligible for rendering in the primary camera view.", "Check object and parent visibility, camera layers, and the camera frustum."),
                ("projection_valid", "subject_projection_invalid", "Subject has no valid projected bounds.", "Move geometry clear of the near plane and camera, and eliminate non-finite or degenerate transforms."),
            )
            for flag, code_name, message, repair in checks:
                if obj[flag] is not True:
                    visual_issues.append(_issue(code_name, message, repair, t=t, subject=subject_id))
            if subject["fully_in_frame"] and not obj["fully_in_frame"]:
                visual_issues.append(_issue("subject_not_fully_in_frame", "Subject is clipped despite a fully_in_frame requirement.", "Widen the camera view or reposition the subject so its projected bounds stay inside the frame.", t=t, subject=subject_id))
            if obj["projection_valid"]:
                area = obj["projected_area_ratio"]
                if area <= 0:
                    visual_issues.append(_issue("subject_zero_projected_area", "Subject has zero projected screen area.", "Give the subject nondegenerate visible geometry and a camera view that covers it.", t=t, subject=subject_id))
                if "min_area_ratio" in subject and area < subject["min_area_ratio"] - _EPS:
                    visual_issues.append(_issue("subject_area_too_small", f"Projected area ratio {area:.6g} is below {subject['min_area_ratio']}.", "Increase projected subject size through camera distance/FOV or geometry scale while preserving other constraints.", t=t, subject=subject_id, measured=area))
                if "max_area_ratio" in subject and area > subject["max_area_ratio"] + _EPS:
                    visual_issues.append(_issue("subject_area_too_large", f"Projected area ratio {area:.6g} exceeds {subject['max_area_ratio']}.", "Reduce projected subject size or adjust the camera while preserving required framing.", t=t, subject=subject_id, measured=area))
        for first, second in forbidden_pairs:
            if not is_active(subjects[first], t) or not is_active(subjects[second], t):
                continue
            a, b = by_id.get(first), by_id.get(second)
            if a is None or b is None:
                # Missing required actors are already rejected. Optional absent
                # actors do not create a collision.
                continue
            exact = _exact_penetration(a["collider"], b["collider"])
            if exact is not None:
                penetration, algorithm = exact
                metrics["exact_collision_checks"] += 1
                if not _finite(penetration):
                    failures.append(_issue("collision_evidence_invalid", "Exact collision arithmetic produced a non-finite result.", "Use finite scene scales and valid exact colliders.", t=t, pair=[first, second]))
                elif penetration > tolerance + _GEOMETRY_EPS:
                    failures.append(_issue("forbidden_penetration", f"Exact {algorithm} test found penetration {penetration:.6g} above tolerance {tolerance}.", "Separate the two forbidden colliders or correct their trajectories; touching within tolerance is allowed.", t=t, pair=[first, second], penetration=penetration, tolerance=tolerance, algorithm=algorithm))
            elif _aabb_penetrates(a["world_aabb"], b["world_aabb"], tolerance):
                metrics["unverified_collision_checks"] += 1
                warnings.append(_issue("collision_unverified", "AABBs overlap, but these shapes have no supported exact collider. AABB overlap is not proof of penetration.", "Inspect the rendered views or add a geometry-aware narrow-phase checker for this shape; do not label broad-phase overlap as a collision.", t=t, pair=[first, second]))
    metrics["primitive_count"] = max(primitives, default=0)
    metrics["sampled_primitive_count_range"] = [min(primitives), max(primitives)] if primitives else None
    metrics["camera_height_range"] = [min(camera_heights), max(camera_heights)] if camera_heights else None
    metrics["camera_pitch_deg_range"] = [min(camera_pitches), max(camera_pitches)] if camera_pitches else None
    metrics["subject_area_ranges"] = {key: [min(values), max(values)] for key, values in areas.items() if values}
    metrics["sample_times_checked"] = sorted(manifest["t"] for manifest in valid)
    metrics["scope"] = "sampled_times; projection is not an occlusion test; exact collision only for supported colliders"
    report["hard_pass"] = not failures
    return report
