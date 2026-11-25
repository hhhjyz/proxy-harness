"""Evidence-preserving train distillation and paired dev admission.

This module never calls a model, installs a personal skill, or replaces the
working asset tree.  It prepares a concrete proposal for an external coding
agent, binds its assets to an experimental condition, and checks paired runs.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

from .spec import validate_spec


SCHEMA_VERSION = "1.0"
HASH_FIELDS = ("manifest_sha256", "config_sha256", "harness_sha256", "asset_sha256")
RECORD_FIELDS = ("task_id", "split", "seed", "spec_sha256", "status", "hard_pass",
                 "subject_complete", "score", "code_lines", "primitive_count")
STATUSES = {"accepted", "rejected", "error", "unverified", "skipped"}
EVIDENCE_FILES = (
    "scene.html", "appearance_prompt.txt", "diff.patch", "l1_report.json",
    "console.log", "contact_sheet.png", "top_view.png", "eval.json",
    "manifests.json", "coder_prompt.txt", "coder.stdout.jsonl", "coder.stderr.log",
    "coder_result.json", "eval_request.json", "eval_response.json",
    "eval_wire_request.json", "eval.stdout.jsonl", "eval.stderr.log",
    "evaluator_prompt.txt", "evaluator.stdout.jsonl", "evaluator.stderr.log",
    "evaluator_result.json", "eval.schema.json", "determinism.json",
    "render_meta.json", "render_job.json", "render_worker.log", "ffmpeg.log",
    "proxy.mp4",
)
ROOT_TRACE_FILES = ("spec.json", "summary.json", "config.json", "history.json", "protected_hashes.json")
FRAME_TRACE_DIRS = ("frames", "video_frames")
BINARY_TRACE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4"}
REQUIRED_TRAIN_FILES = (
    "scene.html", "appearance_prompt.txt", "l1_report.json", "eval.json",
    "coder_prompt.txt", "coder.stdout.jsonl", "coder_result.json",
)
REQUIRED_SUCCESS_FILES = (
    "scene.html", "appearance_prompt.txt", "l1_report.json", "eval.json",
    "manifests.json", "contact_sheet.png", "top_view.png",
    "coder_prompt.txt", "coder.stdout.jsonl", "coder_result.json",
    "eval_request.json", "eval_response.json",
)


class OuterLoopError(ValueError):
    """A fail-closed provenance, pairing, or admission error."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON, including normalized specs, without NaN."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise OuterLoopError("NONFINITE_JSON", f"{path.name} contains {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, json.JSONDecodeError) as exc:
        raise OuterLoopError("MISSING_OR_INVALID_JSON", f"Cannot read {path}: {exc}") from exc


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OuterLoopError("INVALID_HASH", f"{label} must be a lowercase SHA-256 digest")
    return value


def asset_inventory(root: str | Path) -> dict[str, dict]:
    """Return a deterministic inventory; symlinks are deliberately unsupported."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise OuterLoopError("INVALID_ASSET_TREE", f"Expected a real directory: {root}")
    inventory = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise OuterLoopError("ASSET_SYMLINK", f"Asset symlinks are not allowed: {relative}")
        if path.is_file():
            inventory[relative] = {"sha256": _file_sha(path), "size": path.stat().st_size}
    if not inventory:
        raise OuterLoopError("EMPTY_ASSET_TREE", "The asset tree has no files")
    return inventory


def hash_tree(root: str | Path) -> str:
    """Hash {POSIX-relative path: raw-file SHA-256}, excluding Python caches.

    Asset inventories still include every file when checking the permitted
    edit scope.  The exclusion keeps imports from changing a runner hash.
    """
    return canonical_sha256({name: info["sha256"] for name, info in asset_inventory(root).items()
                             if "__pycache__" not in Path(name).parts and Path(name).suffix not in {".pyc", ".pyo"}})


def condition_id(metadata: dict) -> str:
    """Bind the dataset, comparable configuration, runner, and full asset tree."""
    return canonical_sha256({key: _sha(metadata.get(key), key) for key in HASH_FIELDS})


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise OuterLoopError("UNSAFE_PATH", "Expected a nonempty relative path")
    if ".." in Path(relative).parts:
        raise OuterLoopError("UNSAFE_PATH", f"Parent traversal is not allowed: {relative}")
    path = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise OuterLoopError("EVIDENCE_SYMLINK", f"Symlink not allowed: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise OuterLoopError("UNSAFE_PATH", f"Path escapes its root: {relative}")
    return path


def load_manifest(path: str | Path) -> dict:
    """Validate the pinned manifest and every normalized spec, returning copies.

    The disk manifest contains no self-hash.  Its canonical hash is pinned in
    manifest.sha256; the return value additionally has manifest_sha256 and
    manifest_path.  Each returned task additionally contains its parsed spec.
    """
    path = Path(path).resolve()
    data = _read_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise OuterLoopError("INVALID_MANIFEST", "Unsupported manifest schema")
    if "manifest_sha256" in data or not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise OuterLoopError("INVALID_MANIFEST", "Expected nonempty tasks and an external hash pin")
    digest = canonical_sha256(data)
    pin_path = path.with_suffix(".sha256")
    try:
        pinned = pin_path.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as exc:
        raise OuterLoopError("MISSING_MANIFEST_PIN", f"Missing {pin_path.name}") from exc
    if _sha(pinned, "manifest pin") != digest:
        raise OuterLoopError("MANIFEST_HASH_MISMATCH", "The manifest differs from its checked-in pin")
    seen = set()
    seen_paths = set()
    for task in data["tasks"]:
        if not isinstance(task, dict) or task.get("split") not in {"train", "dev", "holdout"}:
            raise OuterLoopError("INVALID_MANIFEST", "Every task needs an explicit dataset split")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id) is None:
            raise OuterLoopError("INVALID_MANIFEST", "Unsafe task identifier")
        if task_id in seen or task.get("spec_path") in seen_paths:
            raise OuterLoopError("DUPLICATE_TASK", "A task or specification path occurs twice")
        seen.add(task_id)
        seen_paths.add(task.get("spec_path"))
        spec = validate_spec(_read_json(_inside(path.parent, task.get("spec_path"))))
        for key in ("task_id", "split", "seed"):
            if task.get(key) != spec.get(key):
                raise OuterLoopError("SPEC_IDENTITY_MISMATCH", f"{task_id}: manifest/spec {key} differ")
        if canonical_sha256(spec) != _sha(task.get("spec_sha256"), "spec_sha256"):
            raise OuterLoopError("SPEC_HASH_MISMATCH", f"{task_id}: normalized spec changed")
        task["spec"] = spec
    data["manifest_sha256"] = digest
    data["manifest_path"] = str(path)
    return data


def _load_batch(value: dict | str | Path) -> dict:
    data = deepcopy(value) if isinstance(value, dict) else _read_json(Path(value))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise OuterLoopError("INVALID_BATCH", "Unsupported batch schema")
    if not isinstance(data.get("records"), list):
        raise OuterLoopError("INVALID_BATCH", "Batch records must be a list")
    if data.get("condition_id") != condition_id(data):
        raise OuterLoopError("CONDITION_MISMATCH", "Condition ID does not match its four immutable hashes")
    return data


def _number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _record_identity(record: dict, task: dict) -> None:
    for key in ("task_id", "split", "seed", "spec_sha256"):
        if record.get(key) != task.get(key):
            raise OuterLoopError("RECORD_IDENTITY_MISMATCH", f"{task['task_id']}: incorrect {key}")
    if record.get("status") not in STATUSES:
        raise OuterLoopError("INVALID_STATUS", f"{task['task_id']}: unsupported status")
    for key in ("code_lines", "primitive_count"):
        value = record.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise OuterLoopError("INVALID_COMPLEXITY", f"{task['task_id']}: invalid {key}")
    for key in ("hard_pass", "subject_complete"):
        if record.get(key) is not None and type(record[key]) is not bool:
            raise OuterLoopError("INVALID_QUALITY", f"{task['task_id']}: {key} must be boolean or null")
    if record.get("score") is not None and (not _number(record["score"]) or not 0 <= record["score"] <= 100):
        raise OuterLoopError("INVALID_SCORE", f"{task['task_id']}: invalid score")


def _inspect_run(record: dict, task: dict, *, require_live: bool, condition: dict | None = None) -> dict:
    """Cross-check batch claims against immutable run artifacts, not model prose."""
    _record_identity(record, task)
    raw_dir = record.get("run_dir")
    if not isinstance(raw_dir, str) or not raw_dir:
        raise OuterLoopError("MISSING_RUN", f"{task['task_id']}: no run directory")
    run_dir = Path(raw_dir)
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise OuterLoopError("MISSING_RUN", f"{task['task_id']}: missing real run directory")
    summary = _read_json(_inside(run_dir, "summary.json"))
    for key in RECORD_FIELDS:
        if summary.get(key) != record.get(key):
            raise OuterLoopError("SUMMARY_MISMATCH", f"{task['task_id']}: batch/summary {key} differ")
    actual_spec = validate_spec(_read_json(_inside(run_dir, "spec.json")))
    if canonical_sha256(actual_spec) != task["spec_sha256"]:
        raise OuterLoopError("RUN_SPEC_MISMATCH", f"{task['task_id']}: run used another specification")
    if condition is not None:
        if canonical_sha256(_read_json(_inside(run_dir, "config.json"))) != condition["config_sha256"]:
            raise OuterLoopError("RUN_CONFIG_MISMATCH", f"{task['task_id']}: recorded configuration differs from the batch")
        if hash_tree(_inside(run_dir, "assets_snapshot")) != condition["asset_sha256"]:
            raise OuterLoopError("RUN_ASSET_MISMATCH", f"{task['task_id']}: asset snapshot differs from the batch")
        if summary.get("asset_sha256", condition["asset_sha256"]) != condition["asset_sha256"]:
            raise OuterLoopError("RUN_ASSET_MISMATCH", f"{task['task_id']}: summary reports other assets")
    if require_live and summary.get("model_mode") != "live":
        raise OuterLoopError("NOT_LIVE", f"{task['task_id']}: demo/unverified runs cannot establish an improvement")
    if summary.get("model_mode") not in {"live", "demo", "unverified"}:
        raise OuterLoopError("MISSING_MODEL_PROVENANCE", f"{task['task_id']}: unknown model mode")
    evidence_name = summary.get("evidence_dir")
    if not evidence_name:
        if record["status"] not in {"error", "unverified", "skipped"} or any(
            record.get(k) not in (None, False, 0) for k in ("hard_pass", "subject_complete", "score", "code_lines", "primitive_count")
        ):
            raise OuterLoopError("MISSING_EVIDENCE", f"{task['task_id']}: positive claims without evidence")
        return {"summary": summary, "run_dir": run_dir, "evidence_dir": None,
                "success": False, "hard_pass": False, "subject_complete": False,
                "code_lines": 0, "primitive_count": 0, "eval_status": "unverified"}
    if not isinstance(evidence_name, str) or re.fullmatch(r"iter_\d+", evidence_name) is None:
        raise OuterLoopError("UNSAFE_EVIDENCE_DIR", "Evidence must name one iter_NN directory")
    selected = summary.get("selected_iteration")
    if type(selected) is not int or selected < 0 or int(evidence_name[5:]) != selected:
        raise OuterLoopError("ITERATION_MISMATCH", "Selected iteration and evidence directory disagree")
    evidence_dir = _inside(run_dir, evidence_name)
    if not evidence_dir.is_dir():
        raise OuterLoopError("MISSING_EVIDENCE", f"{task['task_id']}: missing iteration directory")
    l1 = _read_json(_inside(evidence_dir, "l1_report.json"))
    evaluation = _read_json(_inside(evidence_dir, "eval.json"))
    if type(l1.get("hard_pass")) is not bool or l1["hard_pass"] != record.get("hard_pass"):
        raise OuterLoopError("HARD_GATE_MISMATCH", f"{task['task_id']}: L1 result disagrees with the record")
    subject_complete = evaluation.get("subject_complete")
    if subject_complete is not None and type(subject_complete) is not bool:
        raise OuterLoopError("INVALID_EVALUATION", "subject_complete must be boolean or null")
    if subject_complete != record.get("subject_complete"):
        raise OuterLoopError("SUBJECT_MISMATCH", f"{task['task_id']}: completeness claims disagree")
    eval_status = evaluation.get("status")
    if eval_status not in {"evaluated", "skipped", "error"}:
        raise OuterLoopError("INVALID_EVALUATION", f"{task['task_id']}: invalid evaluation status")
    raw_score = evaluation.get("score")
    if eval_status == "evaluated":
        if not _number(raw_score) or not 0 <= raw_score <= 100 or type(subject_complete) is not bool:
            raise OuterLoopError("INVALID_EVALUATION", f"{task['task_id']}: evaluated result lacks valid fields")
        effective = raw_score if l1["hard_pass"] else min(raw_score, task["spec"]["evaluation"]["hard_fail_score_cap"])
    else:
        effective = None
    if record.get("score") != effective:
        raise OuterLoopError("SCORE_MISMATCH", f"{task['task_id']}: score does not follow the hard-gate cap")
    scene = _inside(evidence_dir, "scene.html")
    code_lines = sum(bool(line.strip()) for line in scene.read_text(encoding="utf-8").splitlines()) if scene.is_file() else 0
    manifests_path = _inside(evidence_dir, "manifests.json")
    manifests = _read_json(manifests_path) if manifests_path.is_file() else []
    if not isinstance(manifests, list):
        raise OuterLoopError("INVALID_MANIFEST_EVIDENCE", "manifests.json must be the actual sampled list")
    counts = []
    for frame in manifests:
        count = frame.get("audit", {}).get("primitive_count") if isinstance(frame, dict) else None
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise OuterLoopError("INVALID_PRIMITIVE_COUNT", "Every sampled manifest needs a measured primitive count")
        counts.append(count)
    primitive_count = max(counts, default=0)
    if (code_lines, primitive_count) != (record["code_lines"], record["primitive_count"]):
        raise OuterLoopError("COMPLEXITY_MISMATCH", f"{task['task_id']}: complexity differs from source/manifest evidence")
    success = (record["status"] == "accepted" and l1["hard_pass"] and
               eval_status == "evaluated" and subject_complete is True and
               effective >= task["spec"]["evaluation"]["pass_score"] and summary.get("model_mode") == "live")
    if record["status"] == "accepted" and not success:
        raise OuterLoopError("FALSE_ACCEPTANCE", f"{task['task_id']}: accepted label does not satisfy all gates")
    if success:
        for filename in REQUIRED_SUCCESS_FILES:
            file = _inside(evidence_dir, filename)
            if not file.is_file() or file.stat().st_size == 0:
                raise OuterLoopError("MISSING_SUCCESS_EVIDENCE", f"{task['task_id']}: missing {filename}")
        if not manifests:
            raise OuterLoopError("MISSING_SUCCESS_EVIDENCE", "Accepted scene has no sampled manifests")
    return {"summary": summary, "run_dir": run_dir, "evidence_dir": evidence_dir,
            "success": success, "hard_pass": l1["hard_pass"],
            "subject_complete": subject_complete is True, "code_lines": code_lines,
            "primitive_count": primitive_count, "eval_status": eval_status}


def _seal_proposal(proposal: dict) -> dict:
    result = deepcopy(proposal)
    result.pop("proposal_sha256", None)
    result["proposal_sha256"] = canonical_sha256(result)
    return result


def _read_proposal(path: Path) -> dict:
    value = _read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise OuterLoopError("INVALID_PROPOSAL", "Unsupported proposal")
    if _seal_proposal(value)["proposal_sha256"] != value.get("proposal_sha256"):
        raise OuterLoopError("PROPOSAL_HASH_MISMATCH", "Proposal metadata changed after being sealed")
    return value


def _validate_evidence_pin(stage: Path, proposal: dict) -> None:
    for relative, digest in proposal.get("evidence_sha256", {}).items():
        path = _inside(stage, relative)
        if not path.is_file() or _file_sha(path) != digest:
            raise OuterLoopError("STAGED_EVIDENCE_CHANGED", f"Pinned evidence changed: {relative}")


def _collect_train_trace(checked: dict, forbidden: list[tuple[str, str]]) -> list[dict]:
    """Inventory every recorded iteration, including failed/pre-selected ones.

    Only named raw artifacts and image-frame directories enter the inventory;
    workspaces, other runs, and arbitrary sibling files are never traversed.
    Hashes are made before staging so concurrent source mutation is detectable.
    """
    run_dir = checked["run_dir"]
    iteration_dirs = []
    for child in sorted(run_dir.iterdir()):
        match = re.fullmatch(r"iter_(\d+)", child.name)
        if match is None:
            continue
        directory = _inside(run_dir, child.name)
        if not directory.is_dir():
            raise OuterLoopError("INVALID_TRACE_ITERATION", f"Iteration path is not a directory: {child.name}")
        iteration_dirs.append((int(match.group(1)), directory))
    iteration_dirs.sort(key=lambda item: item[0])
    numbers = [number for number, _ in iteration_dirs]
    if not numbers or numbers != list(range(len(numbers))):
        raise OuterLoopError("INCOMPLETE_TRAIN_TRACE", "Raw iteration directories must be distinct and contiguous from iter_00")
    recorded_count = checked["summary"].get("iterations")
    if recorded_count is not None and (type(recorded_count) is not int or recorded_count != len(numbers)):
        raise OuterLoopError("INCOMPLETE_TRAIN_TRACE", "Summary iteration count does not match the raw trajectory")
    if checked["summary"]["selected_iteration"] not in numbers:
        raise OuterLoopError("INCOMPLETE_TRAIN_TRACE", "Selected iteration is absent from the raw trajectory")
    relatives = set()
    for name in ROOT_TRACE_FILES:
        path = _inside(run_dir, name)
        if path.exists():
            if not path.is_file():
                raise OuterLoopError("INVALID_TRACE_FILE", f"Expected a regular trace file: {name}")
            relatives.add(name)
    for _, directory in iteration_dirs:
        for name in EVIDENCE_FILES:
            relative = f"{directory.name}/{name}"
            path = _inside(run_dir, relative)
            if path.exists():
                if not path.is_file():
                    raise OuterLoopError("INVALID_TRACE_FILE", f"Expected a regular trace file: {relative}")
                relatives.add(relative)
        # Some renderers put diagnostic sample PNGs directly in the iteration.
        for child in sorted(directory.iterdir()):
            if child.name.startswith("sample_") and child.suffix.lower() == ".png":
                relative = child.relative_to(run_dir).as_posix()
                if not _inside(run_dir, relative).is_file():
                    raise OuterLoopError("INVALID_TRACE_FILE", f"Expected a sampled image: {relative}")
                relatives.add(relative)
        for name in FRAME_TRACE_DIRS:
            frame_dir = _inside(run_dir, f"{directory.name}/{name}")
            if not frame_dir.exists():
                continue
            if not frame_dir.is_dir():
                raise OuterLoopError("INVALID_TRACE_FILE", f"Expected a frame directory: {directory.name}/{name}")
            for child in sorted(frame_dir.rglob("*")):
                relative = child.relative_to(run_dir).as_posix()
                path = _inside(run_dir, relative)
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    relatives.add(relative)
    inventory = []
    for relative in sorted(relatives):
        path = _inside(run_dir, relative)
        content = path.read_bytes()
        if path.suffix.lower() not in BINARY_TRACE_SUFFIXES:
            text = content.decode("utf-8", errors="replace")
            if any(task_id in text or prompt in text for task_id, prompt in forbidden):
                raise OuterLoopError("HELDOUT_CONTENT_IN_TRAIN", "A train trajectory artifact contains a held-out task identifier or prompt")
        inventory.append({"relative": relative, "sha256": hashlib.sha256(content).hexdigest()})
    checked["trace_iterations"] = numbers
    return inventory


def prepare_distillation(batch_results: dict | str | Path, manifest_path: str | Path,
                         staging_dir: str | Path, *, candidate_name: str,
                         max_examples: int = 6, baseline_assets: str | Path | None = None) -> dict:
    """Stage raw live train evidence and a proposal request; do not run an agent.

    Mixed batches may be supplied, but only manifest-confirmed train records
    are ever opened.  Dev/holdout source paths and contents never enter the
    staged request. All recorded iterations of each selected run are retained,
    including failures before/after its selected result. A failed live train
    run is useful evidence, not a success.
    """
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples < 1:
        raise OuterLoopError("INVALID_BUDGET", "max_examples must be a positive integer")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", candidate_name):
        raise OuterLoopError("INVALID_CANDIDATE_NAME", "Use a simple candidate identifier")
    manifest = load_manifest(manifest_path)
    batch = _load_batch(batch_results)
    if batch["manifest_sha256"] != manifest["manifest_sha256"]:
        raise OuterLoopError("MANIFEST_MISMATCH", "Batch belongs to a different pinned task set")
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    # Deny exact held-out task identifiers or prompt copies in raw teacher inputs.
    forbidden = [(task["task_id"], task["spec"]["prompt"]) for task in manifest["tasks"] if task["split"] != "train"]
    selected = []
    excluded = []
    seen = set()
    for record in batch["records"]:
        task = tasks.get(record.get("task_id")) if isinstance(record, dict) else None
        if task is None:
            raise OuterLoopError("UNKNOWN_TASK", "Batch contains a task outside the pinned manifest")
        if task["task_id"] in seen:
            raise OuterLoopError("DUPLICATE_TASK", "Duplicate source task in distillation batch")
        seen.add(task["task_id"])
        if record.get("split") != task["split"]:
            raise OuterLoopError("SPLIT_SPOOFING", "Record split disagrees with the pinned manifest")
        if task["split"] != "train":
            # No run directory is touched and no held-out identifier is exported.
            excluded.append({"reason": "NON_TRAIN_SPLIT"})
            continue
        try:
            checked = _inspect_run(record, task, require_live=True, condition=batch)
            evidence = checked["evidence_dir"]
            if evidence is None:
                raise OuterLoopError("MISSING_TRAIN_EVIDENCE", "No raw coding trajectory")
            for filename in REQUIRED_TRAIN_FILES:
                path = _inside(evidence, filename)
                if not path.is_file() or path.stat().st_size == 0:
                    raise OuterLoopError("MISSING_TRAIN_EVIDENCE", f"Missing {filename}")
            coder = _read_json(_inside(evidence, "coder_result.json"))
            backend = coder.get("backend")
            if not isinstance(backend, str) or not backend or backend.lower() in {"demo", "mock", "fixture", "unverified"}:
                raise OuterLoopError("NOT_LIVE", "No real coding backend is recorded")
            if not isinstance(coder.get("exit_code"), int) or isinstance(coder.get("exit_code"), bool):
                raise OuterLoopError("MISSING_TRAIN_EVIDENCE", "No process result is recorded")
            checked["trace_inventory"] = _collect_train_trace(checked, forbidden)
            selected.append((record, task, checked))
        except (OuterLoopError, OSError) as exc:
            excluded.append({"task_id": task["task_id"], "reason": getattr(exc, "code", "UNREADABLE_TRAIN_EVIDENCE")})
    selected.sort(key=lambda item: (item[2]["success"], item[1]["task_id"]))
    selected = selected[:max_examples]
    if not selected:
        raise OuterLoopError("NO_VERIFIED_TRAIN_EVIDENCE", "No live train case with sufficient original evidence is available")
    inventory = None
    if baseline_assets is not None:
        baseline_assets = Path(baseline_assets).resolve()
        inventory = asset_inventory(baseline_assets)
        if hash_tree(baseline_assets) != batch["asset_sha256"]:
            raise OuterLoopError("BASELINE_ASSET_MISMATCH", "The supplied baseline assets were not used by this batch")
    stage = Path(staging_dir).resolve()
    if stage.exists() and any(stage.iterdir()):
        raise OuterLoopError("STAGING_NOT_EMPTY", "Use a fresh staging directory; existing proposals are never overwritten")
    if baseline_assets is not None and stage.is_relative_to(baseline_assets):
        raise OuterLoopError("UNSAFE_STAGING", "Staging must be outside the baseline asset tree")
    stage.mkdir(parents=True, exist_ok=True)
    cases = []
    pins = {}
    for record, task, checked in selected:
        destination = stage / "train_evidence" / task["task_id"]
        destination.mkdir(parents=True)
        evidence_paths = []
        for item in checked["trace_inventory"]:
            source = _inside(checked["run_dir"], item["relative"])
            target = _inside(destination, item["relative"])
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise OuterLoopError("SOURCE_TRACE_CHANGED", f"Source evidence changed during staging: {item['relative']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            relative = target.relative_to(stage).as_posix()
            pins[relative] = item["sha256"]
            evidence_paths.append(relative)
        cases.append({"task_id": task["task_id"], "split": "train", "seed": task["seed"],
                      "spec_sha256": task["spec_sha256"], "status": record["status"],
                      "verified_success": checked["success"],
                      "selected_iteration": checked["summary"]["selected_iteration"],
                      "trace_iterations": checked["trace_iterations"],
                      "trace_policy": "all_recorded_iterations", "evidence_paths": evidence_paths})
    if baseline_assets is not None:
        shutil.copytree(baseline_assets, stage / "candidate_assets")
    request = {
        "schema_version": SCHEMA_VERSION, "purpose": "staged_train_only_proposal",
        "candidate_name": candidate_name, "training_cases": cases,
        "candidate_assets": "candidate_assets", "allowed_changes": ["primitives.js", "skills/**/*.md", "skills/**/*.txt", "skills/**/*.json"],
        "constraints": [
            "Read only the listed train evidence and the staged candidate asset tree.",
            "Read the complete per-case iteration sequence and history, including failed attempts and repair diffs; selected_iteration identifies the verified result, not the whole training trajectory.",
            "Do not inspect dev or holdout tasks, batches, results, or directories.",
            "Do not change runtime.js, scene.html, AGENTS.md, vendor files, validators, or evaluation thresholds.",
            "Preserve required subjects, timing, camera and collision constraints; do not simplify by deleting required content.",
            "A failed, skipped or unverified source is not a positive example.",
            "Propose reusable project guidance or primitive improvements; do not hard-code task ids or answers.",
            "Write candidate_notes.json with rationale, exact source evidence, expected benefits and risks.",
            "Project skills may be written only under candidate_assets/skills/<name>/SKILL.md with name/description frontmatter and safe local references.",
            "Do not install or modify a personal assistant skill; do not overwrite the main assets.",
        ],
        "output_status": "proposal_requires_binding_and_paired_dev_runs",
    }
    _write_json(stage / "proposal_request.json", request)
    pins["proposal_request.json"] = _file_sha(stage / "proposal_request.json")
    prompt = (
        "Prepare a small reusable project asset improvement from the live TRAIN evidence in this isolated staging directory.\n"
        "Read proposal_request.json and follow its exact source-file list and change restrictions.\n"
        "Read only those train artifacts plus candidate_assets. Do not inspect dev/holdout tasks, run results, or other project paths.\n"
        "Learn from the full ordered raw trajectory, including earlier failures, repair diffs, sampled frames and evaluator traces; do not reduce it to the selected final iteration.\n"
        "Edit only candidate_assets/primitives.js and project guidance under candidate_assets/skills/.\n"
        "For a project SKILL.md use valid name/description frontmatter and references that stay within candidate_assets.\n"
        "Write candidate_notes.json with evidence-based rationale, changes, expected benefit and risks.\n"
        "Do not edit proposal.json, proposal_request.json, this prompt, or train_evidence.\n"
        "Do not run admission or claim measured improvement; a separate controller will bind assets and run the complete paired dev split.\n"
        "Do not install personal skills or overwrite main assets. Leave the proposal staged for review.\n"
    )
    (stage / "distill_prompt.txt").write_text(prompt, encoding="utf-8")
    pins["distill_prompt.txt"] = _file_sha(stage / "distill_prompt.txt")
    proposal = _seal_proposal({
        "schema_version": SCHEMA_VERSION, "candidate_name": candidate_name, "status": "staged",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": {**{key: batch[key] for key in HASH_FIELDS}, "condition_id": batch["condition_id"]},
        "training_cases": [{key: case[key] for key in ("task_id", "split", "seed", "spec_sha256", "status", "verified_success", "selected_iteration", "trace_iterations", "trace_policy")} for case in cases],
        "baseline_asset_inventory": inventory, "evidence_sha256": pins,
        "excluded_counts": {reason: sum(item["reason"] == reason for item in excluded) for reason in sorted({item["reason"] for item in excluded})},
        "candidate_asset_sha256": None, "candidate_condition_id": None,
        "promotion": "never_automatic",
    })
    _write_json(stage / "proposal.json", proposal)
    return {"status": "staged", "proposal_path": str(stage / "proposal.json"),
            "request_path": str(stage / "proposal_request.json"),
            "prompt_path": str(stage / "distill_prompt.txt"),
            "candidate_assets": str(stage / "candidate_assets"), "training_case_count": len(cases),
            "excluded_counts": proposal["excluded_counts"], "proposal_sha256": proposal["proposal_sha256"]}


def _mutable_asset(relative: str) -> bool:
    path = Path(relative)
    if relative == "primitives.js":
        return True
    return (len(path.parts) >= 2 and path.parts[0] == "skills" and
            path.suffix in {".md", ".txt", ".json"} and
            all(not part.startswith(".") for part in path.parts))


def _validate_project_skills(candidate: Path) -> None:
    """Validate project skill metadata and explicit local Markdown references.

    These are staged runtime assets, not personal installed skills.  This
    conservative check validates Markdown links and backticked references/
    scripts paths; it does not claim to interpret arbitrary natural language.
    """
    skill_root = candidate / "skills"
    if not skill_root.is_dir():
        return
    for path in sorted(skill_root.rglob("SKILL.md")):
        if path.parent.parent != skill_root:
            raise OuterLoopError("INVALID_PROJECT_SKILL", "Use skills/<name>/SKILL.md")
        content = path.read_text(encoding="utf-8")
        if not content.startswith("---\n") or "\n---" not in content[4:]:
            raise OuterLoopError("INVALID_SKILL_FRONTMATTER", "Project skills need YAML name/description frontmatter")
        front = content.split("---", 2)[1]
        fields = {}
        for line in front.splitlines():
            if ":" in line and not line.startswith((" ", "\t")):
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip().strip("\"'")
        name, description = fields.get("name", ""), fields.get("description", "")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64 or name != path.parent.name:
            raise OuterLoopError("INVALID_SKILL_NAME", "Project skill name must match its lowercase-hyphen directory")
        if not description or description in {"|", ">", "|-", ">-"}:
            raise OuterLoopError("INVALID_SKILL_DESCRIPTION", "Use a nonempty single-line description for on-demand indexing")
    for path in sorted(skill_root.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        links += re.findall(r"`((?:references|scripts)/[^`\n]+)`", content)
        for link in links:
            target = link.strip().split(' "', 1)[0].strip("<>")
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target:
                continue
            if ":" in target or "\\" in target or Path(target).is_absolute() or target.startswith("~"):
                raise OuterLoopError("UNSAFE_SKILL_REFERENCE", f"Unsafe local reference in {path.name}")
            resolved = (path.parent / target).resolve()
            if not resolved.is_relative_to(candidate.resolve()) or not resolved.exists():
                raise OuterLoopError("UNSAFE_SKILL_REFERENCE", f"Reference escapes assets or is missing: {target}")
            _inside(candidate, resolved.relative_to(candidate.resolve()).as_posix())


def _candidate_scope(proposal: dict, candidate: Path) -> tuple[dict, list[str]]:
    baseline = proposal.get("baseline_asset_inventory")
    if not isinstance(baseline, dict) or not baseline:
        raise OuterLoopError("MISSING_BASELINE_INVENTORY", "Prepare with baseline_assets before binding a candidate")
    mapping = {}
    for name, info in baseline.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            raise OuterLoopError("INVALID_BASELINE_INVENTORY", "Invalid asset inventory")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise OuterLoopError("INVALID_BASELINE_INVENTORY", "Unsafe inventory path")
        digest = _sha(info.get("sha256"), "baseline asset file")
        if "__pycache__" not in Path(name).parts and Path(name).suffix not in {".pyc", ".pyo"}:
            mapping[name] = digest
    if canonical_sha256(mapping) != proposal["baseline"]["asset_sha256"]:
        raise OuterLoopError("BASELINE_INVENTORY_MISMATCH", "Baseline inventory does not match the bound baseline asset hash")
    current = asset_inventory(candidate)
    _validate_project_skills(candidate)
    changed = [name for name in sorted(set(baseline) | set(current)) if baseline.get(name) != current.get(name)]
    if not changed:
        raise OuterLoopError("NO_CANDIDATE_CHANGES", "The candidate is identical to the baseline")
    forbidden = [name for name in changed if not _mutable_asset(name)]
    if forbidden:
        raise OuterLoopError("FORBIDDEN_ASSET_CHANGE", "Changes outside project guidance/primitives: " + ", ".join(forbidden))
    if "primitives.js" in baseline and "primitives.js" not in current:
        raise OuterLoopError("DELETED_PRIMITIVES", "The primitive module cannot be deleted")
    return current, changed


def bind_candidate(staging_dir: str | Path, candidate_assets: str | Path) -> dict:
    """Seal the full candidate tree and condition after reviewing its change scope."""
    stage = Path(staging_dir).resolve()
    candidate = Path(candidate_assets).resolve()
    if not candidate.is_relative_to(stage):
        raise OuterLoopError("CANDIDATE_NOT_STAGED", "Candidate assets must be in the proposal's staging directory")
    proposal = _read_proposal(stage / "proposal.json")
    _validate_evidence_pin(stage, proposal)
    current, changed = _candidate_scope(proposal, candidate)
    digest = hash_tree(candidate)
    metadata = {**proposal["baseline"], "asset_sha256": digest}
    proposal.update({"status": "bound", "candidate_asset_sha256": digest,
                     "candidate_condition_id": condition_id(metadata), "changed_assets": changed,
                     "candidate_assets_relative": candidate.relative_to(stage).as_posix(),
                     "candidate_asset_inventory": current})
    notes = stage / "candidate_notes.json"
    if notes.is_file():
        _read_json(notes)
        proposal["candidate_notes_sha256"] = _file_sha(notes)
    proposal = _seal_proposal(proposal)
    _write_json(stage / "proposal.json", proposal)
    return {"status": "bound", "proposal_path": str(stage / "proposal.json"),
            "candidate_asset_sha256": digest, "candidate_condition_id": proposal["candidate_condition_id"],
            "changed_assets": changed, "proposal_sha256": proposal["proposal_sha256"]}


def _dev_records(batch: dict, manifest: dict) -> dict[str, tuple[dict, dict]]:
    expected = {task["task_id"]: task for task in manifest["tasks"] if task["split"] == "dev"}
    if not expected:
        raise OuterLoopError("EMPTY_DEV_SET", "The pinned manifest has no development tasks")
    actual = {}
    for record in batch["records"]:
        if not isinstance(record, dict) or record.get("split") != "dev":
            raise OuterLoopError("NON_DEV_ADMISSION", "Admission accepts only the complete dev split; holdout is separate")
        task_id = record.get("task_id")
        if task_id in actual:
            raise OuterLoopError("DUPLICATE_TASK", "Duplicate task in paired batch")
        if task_id not in expected:
            raise OuterLoopError("UNEXPECTED_DEV_TASK", "Batch contains a task outside the pinned dev split")
        _record_identity(record, expected[task_id])
        actual[task_id] = (record, expected[task_id])
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        raise OuterLoopError("INCOMPLETE_DEV_SET", "Missing development tasks: " + ", ".join(missing))
    return actual


def admit_candidate(baseline: dict | str | Path, candidate: dict | str | Path,
                    manifest_path: str | Path, *, proposal_path: str | Path,
                    candidate_assets: str | Path) -> dict:
    """Return an admission report; never publish, install, or copy to main.

    Eligibility requires complete paired dev evidence, no task-level hard-gate,
    subject-completeness or acceptance regressions, and either more accepted
    tasks or a Pareto reduction of measured complexity on their passed
    intersection.  Skipped/unverified outcomes always count as failures.
    """
    report = {"schema_version": SCHEMA_VERSION, "status": "rejected", "eligible": False,
              "reasons": [], "promotion": "not_performed", "metrics": {}}
    try:
        manifest = load_manifest(manifest_path)
        base = _load_batch(baseline)
        cand = _load_batch(candidate)
        for key in ("manifest_sha256", "config_sha256", "harness_sha256"):
            if base[key] != cand[key]:
                raise OuterLoopError("PAIRED_CONDITION_MISMATCH", f"Paired runs differ in {key}")
        if base["manifest_sha256"] != manifest["manifest_sha256"]:
            raise OuterLoopError("MANIFEST_MISMATCH", "Paired batches use another dataset version")
        proposal_path = Path(proposal_path).resolve()
        proposal = _read_proposal(proposal_path)
        _validate_evidence_pin(proposal_path.parent, proposal)
        if proposal.get("status") != "bound":
            raise OuterLoopError("UNBOUND_CANDIDATE", "Bind candidate assets before running the paired experiment")
        for key in (*HASH_FIELDS, "condition_id"):
            if proposal["baseline"].get(key) != base[key]:
                raise OuterLoopError("BASELINE_BINDING_MISMATCH", f"Proposal baseline differs in {key}")
        actual_assets = hash_tree(candidate_assets)
        if actual_assets != cand["asset_sha256"] or actual_assets != proposal.get("candidate_asset_sha256"):
            raise OuterLoopError("CANDIDATE_ASSET_MISMATCH", "The candidate assets changed or were not used by the candidate batch")
        if cand["condition_id"] != proposal.get("candidate_condition_id"):
            raise OuterLoopError("CANDIDATE_CONDITION_MISMATCH", "Candidate batch is not bound to this proposal")
        current_inventory, changed_assets = _candidate_scope(proposal, Path(candidate_assets).resolve())
        if current_inventory != proposal.get("candidate_asset_inventory") or changed_assets != proposal.get("changed_assets"):
            raise OuterLoopError("CANDIDATE_INVENTORY_MISMATCH", "Bound candidate inventory or allowed edit list changed")
        if base["asset_sha256"] == cand["asset_sha256"]:
            raise OuterLoopError("IDENTICAL_ASSETS", "There is no distinct candidate to admit")
        base_records = _dev_records(base, manifest)
        cand_records = _dev_records(cand, manifest)
        measured_base = {key: _inspect_run(record, task, require_live=True, condition=base) for key, (record, task) in base_records.items()}
        measured_cand = {key: _inspect_run(record, task, require_live=True, condition=cand) for key, (record, task) in cand_records.items()}
        passed_base = {key for key, value in measured_base.items() if value["success"]}
        passed_cand = {key for key, value in measured_cand.items() if value["success"]}
        intersection = sorted(passed_base & passed_cand)
        hard_regressions = sorted(key for key in measured_base if measured_base[key]["hard_pass"] and not measured_cand[key]["hard_pass"])
        subject_regressions = sorted(key for key in measured_base if measured_base[key]["subject_complete"] and not measured_cand[key]["subject_complete"])
        acceptance_regressions = sorted(passed_base - passed_cand)
        totals = {name: {metric: sum(values[key][metric] for key in intersection)
                         for metric in ("code_lines", "primitive_count")}
                  for name, values in (("baseline", measured_base), ("candidate", measured_cand))}
        complexity_improved = bool(intersection) and all(totals["candidate"][metric] <= totals["baseline"][metric]
            for metric in ("code_lines", "primitive_count")) and any(totals["candidate"][metric] < totals["baseline"][metric]
            for metric in ("code_lines", "primitive_count"))
        pass_improved = len(passed_cand) > len(passed_base)
        report["metrics"] = {
            "task_count": len(base_records), "baseline_passed": len(passed_base), "candidate_passed": len(passed_cand),
            "baseline_pass_rate": len(passed_base) / len(base_records), "candidate_pass_rate": len(passed_cand) / len(base_records),
            "passed_intersection": intersection, "intersection_complexity_totals": totals,
            "complexity_pareto_improved": complexity_improved, "pass_rate_improved": pass_improved,
            "hard_gate_regressions": hard_regressions, "subject_completeness_regressions": subject_regressions,
            "acceptance_regressions": acceptance_regressions,
            "baseline_condition_id": base["condition_id"], "candidate_condition_id": cand["condition_id"],
        }
        for code, items in (("HARD_GATE_REGRESSION", hard_regressions), ("SUBJECT_COMPLETENESS_REGRESSION", subject_regressions),
                            ("ACCEPTANCE_REGRESSION", acceptance_regressions)):
            if items:
                report["reasons"].append({"code": code, "tasks": items})
        if not pass_improved and not complexity_improved:
            report["reasons"].append({"code": "NO_MEASURED_IMPROVEMENT", "message": "Neither pass rate nor complexity on the passed intersection improved"})
        if not report["reasons"]:
            report.update({"status": "eligible", "eligible": True,
                           "admission_basis": "pass_rate" if pass_improved else "passed_intersection_complexity",
                           "candidate_asset_sha256": actual_assets, "proposal_sha256": proposal["proposal_sha256"]})
    except (OuterLoopError, OSError, KeyError, TypeError, ValueError) as exc:
        report["reasons"].append({"code": getattr(exc, "code", "INVALID_EXPERIMENT_EVIDENCE"), "message": str(exc)})
    return report
