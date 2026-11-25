"""Scene specification validation without third-party dependencies.

The shipped JSON Schema is the structural source of truth. The deliberately
small validator implements every validation keyword used in that schema; it
does not claim to be a general-purpose implementation of JSON Schema.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "scene-spec.schema.json"


class SpecValidationError(ValueError):
    """An invalid or internally inconsistent scene specification."""


def _fail(path: str, message: str, repair: str) -> None:
    raise SpecValidationError(f"{path}: {message}. Fix: {repair}")


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _resolve(schema: dict, root: dict) -> dict:
    if "$ref" not in schema:
        return schema
    ref = schema["$ref"]
    if not ref.startswith("#/"):
        raise RuntimeError("Only local schema references are supported")
    node = root
    for key in ref[2:].split("/"):
        node = node[key.replace("~1", "/").replace("~0", "~")]
    return {**node, **{key: value for key, value in schema.items() if key != "$ref"}}


def _validate(value: Any, schema: dict, root: dict, path: str) -> None:
    schema = _resolve(schema, root)
    kind = schema.get("type")
    predicates = {
        "object": lambda x: isinstance(x, dict) and all(isinstance(k, str) for k in x),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "boolean": lambda x: isinstance(x, bool),
        "number": _number,
        "integer": lambda x: isinstance(x, int) and _number(x),
    }
    if kind is not None and not predicates[kind](value):
        _fail(path, f"expected {kind}", "use the documented JSON type and finite numbers")
    if "const" in schema and value != schema["const"]:
        _fail(path, f"must equal {schema['const']!r}", "use the supported schema version")
    if "enum" in schema and value not in schema["enum"]:
        _fail(path, f"must be one of {schema['enum']!r}", "choose a documented value")
    if kind == "object":
        for key in schema.get("required", []):
            if key not in value:
                _fail(path, f"missing required field {key!r}", f"provide {key}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key not in properties:
                if schema.get("additionalProperties") is False:
                    _fail(f"{path}.{key}", "unknown field", "remove the field or correct its spelling")
            else:
                _validate(item, properties[key], root, f"{path}.{key}")
    elif kind == "array":
        if len(value) < schema.get("minItems", 0):
            _fail(path, "too few items", f"provide at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            _fail(path, "too many items", f"provide at most {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if item in value[:index]:
                    _fail(path, "duplicate item", "remove duplicate items")
        for index, item in enumerate(value):
            if "items" in schema:
                _validate(item, schema["items"], root, f"{path}[{index}]")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0):
            _fail(path, "string is too short", "supply a nonempty string")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            _fail(path, "string is too long", f"use at most {schema['maxLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            _fail(path, "invalid identifier", "use letters, digits, underscores, periods, or hyphens; start with a letter or digit")
    elif kind in {"number", "integer"}:
        if "minimum" in schema and value < schema["minimum"]:
            _fail(path, "below minimum", f"use a value >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            _fail(path, "above maximum", f"use a value <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            _fail(path, "must be strictly positive", f"use a value > {schema['exclusiveMinimum']}")
        if "multipleOf" in schema and value % schema["multipleOf"] != 0:
            _fail(path, "invalid dimension", f"use a multiple of {schema['multipleOf']}")


def _defaults(value: Any, schema: dict, root: dict) -> None:
    schema = _resolve(schema, root)
    if isinstance(value, dict):
        for key, child_schema in schema.get("properties", {}).items():
            effective = _resolve(child_schema, root)
            if key not in value and "default" in effective:
                value[key] = deepcopy(effective["default"])
            if key in value:
                _defaults(value[key], effective, root)
    elif isinstance(value, list) and "items" in schema:
        for item in value:
            _defaults(item, schema["items"], root)


def is_active(subject: dict, t: float) -> bool:
    """Whether screen/collision obligations apply at t; intervals are [a, b)."""
    intervals = subject.get("active_intervals")
    return intervals is None or any(start <= t < end for start, end in intervals)


def validate_spec(spec: dict) -> dict:
    """Validate JSON structure and semantics, returning an independent copy.

    Defaults enable subject registration and primary-subject visibility, but
    never invent screen-area thresholds or camera limits. Explicit sample
    times must be increasing, distinct, and inside [0, duration_s).
    """
    with SCHEMA_PATH.open(encoding="utf-8") as stream:
        schema = json.load(stream)
    _validate(spec, schema, schema, "spec")
    result = deepcopy(spec)
    _defaults(result, schema, schema)
    duration = result["duration_s"]
    if not result["prompt"].strip():
        _fail("spec.prompt", "blank prompt", "describe the intended scene")
    if result["task_id"] in {".", ".."}:
        _fail("spec.task_id", "invalid path-like identifier", "choose a task name")
    times = result["sample_times"]
    for index, t in enumerate(times):
        if t >= duration:
            _fail(f"spec.sample_times[{index}]", "sample is outside [0, duration_s)", "choose a time strictly below duration_s")
        if index and t <= times[index - 1]:
            _fail("spec.sample_times", "times are not strictly increasing", "sort the sample times and remove duplicates")

    seen: set[str] = set()
    for index, subject in enumerate(result["subjects"]):
        path = f"spec.subjects[{index}]"
        subject_id = subject["id"]
        if not subject_id.strip():
            _fail(f"{path}.id", "blank subject id", "use a stable nonempty id")
        if subject_id in seen:
            _fail(f"{path}.id", f"duplicate subject {subject_id!r}", "give each subject a unique id")
        seen.add(subject_id)
        if subject.get("min_area_ratio", 0) > subject.get("max_area_ratio", math.inf):
            _fail(path, "min_area_ratio exceeds max_area_ratio", "put the lower area bound first")
        intervals = subject.get("active_intervals", [])
        for interval_index, (start, end) in enumerate(intervals):
            if not 0 <= start < end <= duration:
                _fail(f"{path}.active_intervals[{interval_index}]", "invalid interval", "use 0 <= start < end <= duration_s")
            if interval_index and start < intervals[interval_index - 1][1]:
                _fail(f"{path}.active_intervals", "intervals overlap or are unsorted", "sort intervals and merge overlaps")
        if subject["required"] and intervals and not any(is_active(subject, t) for t in times):
            _fail(f"{path}.active_intervals", "required subject has no active sample", "add a sample time inside an active interval")

    camera = result["camera"]
    for name, bounds in camera.items():
        low, high = bounds
        if low > high:
            _fail(f"spec.camera.{name}", "range is reversed", "put the lower bound first")
        if name == "pitch_deg_range" and not (-90 <= low <= high <= 90):
            _fail("spec.camera.pitch_deg_range", "pitch is outside [-90, 90]", "use signed elevation degrees; downward pitch is negative")

    collision = result["collision"]
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    for key in ("forbidden_pairs", "allowed_pairs"):
        canonical: set[tuple[str, str]] = set()
        for index, pair in enumerate(collision[key]):
            path = f"spec.collision.{key}[{index}]"
            if pair[0] == pair[1]:
                _fail(path, "self-collision pair", "name two different subjects")
            if any(subject_id not in seen for subject_id in pair):
                _fail(path, "unknown subject id", "register both ids in subjects")
            normalized = tuple(sorted(pair))
            if normalized in canonical:
                _fail(path, "duplicate unordered pair", "list each pair only once")
            canonical.add(normalized)
        pair_sets[key] = canonical
    if pair_sets["forbidden_pairs"] & pair_sets["allowed_pairs"]:
        _fail("spec.collision", "a pair is both allowed and forbidden", "give each pair exactly one policy")
    evaluation = result["evaluation"]
    if evaluation["hard_fail_score_cap"] >= evaluation["pass_score"]:
        _fail("spec.evaluation", "hard_fail_score_cap is not below pass_score", "set the cap strictly below the passing score")
    return result


def load_spec(path: str | Path) -> dict:
    """Read a specification, rejecting nonstandard JSON NaN/Infinity values."""
    def reject_constant(value: str) -> None:
        raise SpecValidationError(f"spec: non-finite JSON constant {value}. Fix: use finite numbers")

    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise SpecValidationError(f"spec: invalid JSON at line {exc.lineno}, column {exc.colno}. Fix: correct the JSON syntax") from exc
    return validate_spec(data)
