"""A bounded workflow around a mature CLI, not another coding-agent tool loop."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import re
import shutil
import sys
import time

from .engine import run_agent
from .evaluate import evaluate
from .render import render_scene
from .spec import validate_spec
from .util import ASSETS, ROOT, canonical_hash, file_hash, read_json, write_json


DEFAULT_CONFIG = {
    "coder":{"backend":"codex","timeout_s":240},
    "evaluator":{"backend":"none"},
    "max_iterations":3,
    "render":{"backend":"swiftshader","timeout_s":120},
    "export_video":False,
    "ablations":{"feedback":True,"l1_gate":True,"independent_eval":True},
}


def load_config(path=None):
    value = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        incoming = read_json(path)
        for key, item in incoming.items():
            if key not in value:
                raise ValueError(f"Unknown configuration key: {key}")
            if isinstance(value[key], dict) and isinstance(item, dict):
                value[key].update(item)
            else:
                value[key] = item
    if type(value["max_iterations"]) is not int or not 1 <= value["max_iterations"] <= 4:
        raise ValueError("max_iterations must be 1..4")
    if set(value["ablations"]) != {"feedback","l1_gate","independent_eval"} or not all(type(v) is bool for v in value["ablations"].values()):
        raise ValueError("Ablation flags must be booleans: feedback, l1_gate, independent_eval")
    return value


def candidate_rank(l1, evaluation):
    """Only L1 + complete-subject + valid evaluator candidates enter the best pool."""
    eligible = bool(l1.get("hard_pass") and evaluation.get("status") == "evaluated" and evaluation.get("subject_complete") is True)
    score = evaluation.get("effective_score")
    return (eligible, float(score) if eligible and isinstance(score, (int,float)) else -1.0)


def _make_workspace(spec, work, assets):
    shutil.copytree(assets, work)
    write_json(work / "spec.json", spec)
    (work / "appearance_prompt.txt").write_text(spec["prompt"] + "\nBind every subject by its Scene Spec id. Structural proxy colors are not final appearance.\n", encoding="utf-8")
    shim = ("from pathlib import Path\nimport sys\n"
            + f"sys.path.insert(0, {str(ROOT)!r})\n"
            + "from proxy_harness.__main__ import main\n"
            + "main(['render-check','--scene',str(Path(__file__).with_name('scene.html')),'--spec',str(Path(__file__).with_name('spec.json')),'--out',str(Path(__file__).with_name('local_check'))] + sys.argv[1:])\n")
    (work / "render_and_check.py").write_text(shim, encoding="utf-8")
    skills = []
    for path in sorted((work / "skills").rglob("SKILL.md")) if (work / "skills").exists() else []:
        text = path.read_text(encoding="utf-8")
        front = text.split("---", 2)[1] if text.startswith("---") and text.count("---") >= 2 else ""
        fields = {}
        for line in front.splitlines():
            if ":" in line:
                key, value = line.split(":",1)
                if key.strip() in ("name","description"):
                    fields[key.strip()] = value.strip().strip("\"'")
        skills.append({"path":path.relative_to(work).as_posix(),"name":fields.get("name",path.parent.name),
                       "description":fields.get("description","Read this project skill only if relevant to the scene.")})
    write_json(work / "skills-index.json",skills)
    return {p.relative_to(work).as_posix():file_hash(p) for p in work.rglob("*")
            if p.is_file() and p.name not in ("scene.html","appearance_prompt.txt")}


def _protected_changes(work, expected):
    return [name for name, digest in expected.items()
            if not (work / name).is_file() or (work / name).is_symlink() or file_hash(work / name) != digest]


def coder_prompt(spec, iteration, feedback, *, use_feedback):
    intro = """Implement a deterministic structural proxy video in this workspace.
Read AGENTS.md, spec.json, runtime.js and primitives.js. Only edit scene.html and appearance_prompt.txt.
Read skills-index.json for available project skill metadata; load only the relevant SKILL.md files on demand.
scene.html is the single generated scene. Preserve the local import map and trusted runtime contract.
Use primitive neutral materials to encode geometry, occlusion, causal order and camera motion.
Do not optimize for prettiness. Set all animated state from absolute t, including camera targets.
The appearance_prompt.txt must bind every subject id to the requested final appearance without leaking proxy gray colors.
You may run python render_and_check.py for diagnostics; the controller separately re-renders your snapshot.
Never edit spec.json, AGENTS.md, the checker, runtime, vendor files or measurement/evaluation results.
Do not delegate coding or launch parallel writers. Finish both output files, then briefly report the changes.
"""
    text = intro + "\nScene Spec:\n" + json.dumps(spec, ensure_ascii=False, indent=2)
    if iteration:
        text += "\nThis is a sequential repair in the same coder session. Preserve working behavior.\n"
        if use_feedback:
            text += "Independent measured feedback (use timestamps/subject ids for local fixes):\n" + json.dumps(feedback, ensure_ascii=False, indent=2)
        else:
            text += "Feedback ablation: inspect the spec and revise without the previous L1 or VLM feedback.\n"
    return text


def _record_complexity(iter_dir):
    lines = sum(bool(x.strip()) for x in (iter_dir / "scene.html").read_text(encoding="utf-8").splitlines())
    manifests = read_json(iter_dir / "manifests.json") if (iter_dir / "manifests.json").exists() else []
    counts = [m.get("audit", {}).get("primitive_count",0) for m in manifests if isinstance(m, dict)]
    counts = [x for x in counts if type(x) is int and x >= 0]
    return lines, max(counts, default=0)


def run_task(spec, run_dir, config=None, *, assets_dir=None, demo=False):
    spec = validate_spec(spec)
    config = config or load_config()
    assets = Path(assets_dir or ASSETS).resolve()
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Run directory must be empty; immutable prior evidence will not be overwritten: " + str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "config.json", config)
    # Snapshot the exact trusted assets for reproducibility; candidate writes use another copy.
    shutil.copytree(assets, run_dir / "assets_snapshot")
    work = run_dir / "workspace"
    protected = _make_workspace(spec, work, assets)
    write_json(run_dir / "protected_hashes.json", protected)
    session_id, feedback, previous_source = None, None, ""
    best_valid, best_rank, diagnostic, history = None, (False,-1.0), None, []
    rounds = 1 if demo else config["max_iterations"]
    ablation = config["ablations"]
    is_ablation = not all(ablation.values())
    started = time.monotonic()
    for k in range(rounds):
        iteration = run_dir / f"iter_{k:02d}"
        iteration.mkdir()
        print(f"[{spec['task_id']}] iteration {k+1}/{rounds}: {'fixture' if demo else 'coder'}", flush=True)
        prompt = coder_prompt(spec,k,feedback,use_feedback=ablation["feedback"])
        try:
            if demo:
                (iteration / "coder_prompt.txt").write_text(prompt, encoding="utf-8")
                (iteration / "coder.stdout.jsonl").write_text("", encoding="utf-8")
                (iteration / "coder.stderr.log").write_text("No coding model invoked: deterministic template demonstration.\n", encoding="utf-8")
                result = {"backend":"fixture","exit_code":0,"session_id":None,"evidence_mode":"demo","output":"Pre-authored template; not generated by a model."}
                write_json(iteration / "coder_result.json", result)
            else:
                repair_images = []
                if k and ablation["feedback"]:
                    repair_images = [run_dir / f"iter_{k-1:02d}" / name for name in ("contact_sheet.png","top_view.png")]
                    repair_images = [path for path in repair_images if path.is_file()]
                result = run_agent(config["coder"],prompt,work,iteration,session_id=session_id,images=repair_images)
                session_id = result.get("session_id")
        except Exception as exc:
            result = {"exit_code":127,"backend":config["coder"].get("backend"),"error":str(exc),"evidence_mode":"live"}
            write_json(iteration / "coder_result.json", result)
        changed = _protected_changes(work, protected)
        output_errors = []
        for name in ("scene.html","appearance_prompt.txt"):
            path = work / name
            if not path.is_file() or path.is_symlink():
                output_errors.append("Missing or symlinked generated output: " + name)
                (iteration / name).write_text("", encoding="utf-8")
            else:
                shutil.copyfile(path,iteration / name)
        source = (iteration / "scene.html").read_text(encoding="utf-8")
        (iteration / "diff.patch").write_text("".join(difflib.unified_diff(previous_source.splitlines(True),source.splitlines(True),fromfile=f"iter_{k-1:02d}/scene.html",tofile=f"iter_{k:02d}/scene.html")),encoding="utf-8")
        previous_source = source
        if not (iteration / "appearance_prompt.txt").read_text(encoding="utf-8").strip():
            output_errors.append("appearance_prompt.txt is empty")
        appearance = (iteration / "appearance_prompt.txt").read_text(encoding="utf-8")
        for subject in spec["subjects"]:
            if subject.get("required",True) and subject["id"] not in appearance:
                output_errors.append("Appearance prompt lacks required subject id: " + subject["id"])
        print(f"[{spec['task_id']}] rendering and L1 measurement",flush=True)
        l1 = render_scene(iteration / "scene.html",spec,iteration,assets_dir=run_dir / "assets_snapshot",**config["render"])
        if changed:
            output_errors.append("Protected files changed: " + ", ".join(changed))
        if result.get("exit_code") != 0:
            output_errors.append(f"Coder exited unsuccessfully: {result.get('exit_code')}")
        for error in output_errors:
            l1["failures"].append({"code":"output_integrity","message":error,"repair":"Restore protected files and produce both requested scene outputs."})
        if output_errors:
            l1["hard_pass"] = False
            write_json(iteration / "l1_report.json",l1)
        print(f"[{spec['task_id']}] independent evaluation (L1={l1['hard_pass']})",flush=True)
        evaluator_config = config["evaluator"] if not demo and ablation["independent_eval"] else {"backend":"none"}
        evaluation = evaluate(spec,l1,iteration / "contact_sheet.png",iteration / "top_view.png",iteration,evaluator_config)
        rank = candidate_rank(l1,evaluation)
        if rank[0] and rank > best_rank:
            best_rank, best_valid = rank, k
        diagnostic = k
        accepted = rank[0] and rank[1] >= spec["evaluation"]["pass_score"]
        # Ablations are measurements only and can never be admitted as production results.
        experiment_accept = accepted
        if not ablation["l1_gate"] and evaluation.get("status") == "evaluated":
            experiment_accept = evaluation.get("subject_complete") is True and evaluation["score"] >= spec["evaluation"]["pass_score"]
        history.append({"iteration":k,"hard_pass":l1["hard_pass"],"score":evaluation.get("score"),
                        "effective_score":evaluation.get("effective_score"),"subject_complete":evaluation.get("subject_complete"),
                        "accepted":accepted,"experiment_accept":experiment_accept,"eval_status":evaluation["status"]})
        write_json(run_dir / "history.json",history)
        feedback = {"l1":l1,"evaluation":evaluation}
        if experiment_accept or demo or changed or (evaluation["status"] == "skipped" and l1["hard_pass"]):
            break
        # Restore guard files before any subsequent sequential repair.
        for name in changed:
            original = run_dir / "assets_snapshot" / name
            if original.exists():
                (work / name).unlink(missing_ok=True)
                shutil.copyfile(original,work / name)
            elif name == "spec.json":
                write_json(work / name,spec)
        if not session_id and config["coder"].get("backend","codex") == "codex":
            # Do not quietly turn a required continuation into independent writers.
            break
    selected = best_valid if best_valid is not None else diagnostic
    selected_dir = run_dir / f"iter_{selected:02d}"
    l1, ev = read_json(selected_dir / "l1_report.json"), read_json(selected_dir / "eval.json")
    code_lines, primitive_count = _record_complexity(selected_dir)
    accepted = candidate_rank(l1,ev)[0] and ev["effective_score"] >= spec["evaluation"]["pass_score"]
    status = "accepted" if accepted and not is_ablation and not demo else "unverified" if demo or ev["status"] == "skipped" else "rejected"
    if ev["status"] == "error":
        status = "error"
    from .outerloop import hash_tree
    summary = {"schema_version":"1.0","task_id":spec["task_id"],"split":spec["split"],"seed":spec["seed"],
               "spec_sha256":canonical_hash(spec),"status":status,"hard_pass":l1["hard_pass"],
               "subject_complete":ev["subject_complete"],"score":ev.get("effective_score"),
               "code_lines":code_lines,"primitive_count":primitive_count,"selected_iteration":selected,
               "best_valid_iteration":best_valid,"evidence_dir":selected_dir.name,
               "model_mode":"demo" if demo else "live" if ev["status"] == "evaluated" else "unverified",
               "coder_backend":"fixture" if demo else config["coder"]["backend"],"evaluator_backend":evaluator_config["backend"],
               "is_ablation":is_ablation,"asset_sha256":hash_tree(run_dir / "assets_snapshot"),
               "iterations":len(history),"elapsed_s":round(time.monotonic()-started,3)}
    if best_valid is not None:
        target = run_dir / "best_valid"
        target.mkdir()
        for name in ("scene.html","appearance_prompt.txt","contact_sheet.png","top_view.png","l1_report.json","eval.json"):
            shutil.copyfile(selected_dir / name,target / name)
    if config.get("export_video") and (accepted or demo):
        final_dir = run_dir / "export"
        report = render_scene(selected_dir / "scene.html",spec,final_dir,video=True,assets_dir=run_dir / "assets_snapshot",**config["render"])
        summary["video_exported"] = report["hard_pass"] and (final_dir / "proxy.mp4").is_file()
        if not report["hard_pass"]:
            summary["export_failure"] = report["failures"]
    summary["elapsed_s"] = round(time.monotonic()-started,3)
    write_json(run_dir / "summary.json",summary)
    return summary
