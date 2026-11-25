from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

from .util import ASSETS, ROOT, canonical_hash, read_json, run_process, write_json


def _config(args):
    from .orchestrator import load_config
    config = load_config(getattr(args,"config",None))
    if getattr(args,"browser",None):
        config["render"]["browser"] = args.browser
    if getattr(args,"video",False):
        config["export_video"] = True
    return config


def _batch(args):
    from .outerloop import load_manifest, hash_tree, condition_id
    from .orchestrator import run_task
    from .spec import load_spec
    config = _config(args)
    manifest = load_manifest(Path(args.manifest))
    root = Path(args.manifest).resolve().parent
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Batch output must be a new empty directory")
    output.mkdir(parents=True,exist_ok=True)
    assets = Path(args.assets or ASSETS)
    metadata = {"schema_version":"1.0","manifest_sha256":manifest["manifest_sha256"],
                "config_sha256":canonical_hash(config),
                "harness_sha256":canonical_hash({"code":hash_tree(ROOT / "proxy_harness"),"schemas":hash_tree(ROOT / "schemas")}),
                "asset_sha256":hash_tree(assets),"records":[]}
    metadata["condition_id"] = condition_id(metadata)
    tasks = [x for x in manifest["tasks"] if x["split"] == args.split]
    for entry in tasks:
        spec = load_spec(root / entry["spec_path"])
        run_dir = output / spec["task_id"]
        summary = run_task(spec,run_dir,config,assets_dir=assets)
        metadata["records"].append({**summary,"run_dir":str(run_dir)})
        # Checkpoint every task; a partial batch cannot pass admission.
        write_json(output / "batch_results.json",metadata)
    return metadata


def _plan(args):
    from .engine import run_agent
    from .spec import validate_spec
    from .util import parse_json_response
    config = _config(args)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    target = Path(args.out).resolve()
    if target.exists():
        raise FileExistsError(str(target))
    trace = target.parent / (target.stem + ".planner_trace")
    schema = read_json(ROOT / "schemas" / "scene-spec.schema.json")
    task = ("Convert the following video brief into a compact Scene Spec JSON. Return JSON only. "
            "Do not write code or call a video model. Use schema_version 1.0, split train, seed 42, 3 seconds, "
            "12 fps, 640x360 unless requested otherwise, and 7 timestamp samples in [0,duration). "
            "Include subject IDs, required intervals, camera bounds only if justified, and explicit non-contact pairs. "
            "Do not impose a universal projected-size threshold. Use pass_score=75 and hard_fail_score_cap=40.\n"
            + "Schema:\n" + json.dumps(schema) + "\nBrief:\n" + prompt)
    with tempfile.TemporaryDirectory(prefix="proxy-plan-") as tmp:
        result = run_agent(config["coder"],task,Path(tmp),trace,role="planner")
    if result["exit_code"] != 0:
        raise RuntimeError("Planner CLI failed; inspect its raw trace")
    spec = validate_spec(parse_json_response(result["output"]))
    write_json(target,spec)
    return {"spec":str(target),"planner_trace":str(trace)}


def _handoff(args):
    from .render import render_scene
    from .spec import load_spec
    run = Path(args.run).resolve()
    summary = read_json(run / "summary.json")
    if summary["status"] != "accepted":
        raise ValueError("Only an accepted live-evaluated run may produce a downstream handoff")
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Handoff output directory must be empty")
    out.mkdir(parents=True,exist_ok=True)
    spec = load_spec(run / "spec.json")
    selected = run / summary["evidence_dir"]
    config = read_json(run / "config.json")
    if args.browser:
        config["render"]["browser"] = args.browser
    report = render_scene(selected / "scene.html",spec,out,video=True,assets_dir=run / "assets_snapshot",**config["render"])
    if not report["hard_pass"]:
        raise RuntimeError("Final proxy re-render failed L1; no downstream handoff created")
    shutil.copyfile(selected / "appearance_prompt.txt",out / "appearance_prompt.txt")
    write_json(out / "scene_spec.json",spec)
    info = {"schema_version":"1.0","proxy_video":"proxy.mp4","appearance_prompt":"appearance_prompt.txt",
            "scene_spec":"scene_spec.json","source_run":str(run),"status":"ready_for_backend_configuration",
            "width":spec["width"],"height":spec["height"],"fps":spec["fps"],"duration_s":spec["duration_s"],
            "downstream_model_invoked":False,"note":"Configure the actual video-to-video or structural-conditioning backend. No MiniMax/H3 API endpoint or input capability is assumed."}
    write_json(out / "handoff.json",info)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic Three.js proxy-video harness")
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor",help="Inspect local dependencies without model calls")
    for name in ("demo","run"):
        p = sub.add_parser(name)
        p.add_argument("--out",required=True)
        p.add_argument("--spec",default=str(ROOT / "examples" / "demo.spec.json"),required=name=="run")
        p.add_argument("--config")
        p.add_argument("--assets")
        p.add_argument("--browser")
        p.add_argument("--video",action="store_true")
    p = sub.add_parser("render-check")
    p.add_argument("--scene",required=True)
    p.add_argument("--spec",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--assets")
    p.add_argument("--browser")
    p.add_argument("--backend",choices=["swiftshader","native"],default="swiftshader")
    p.add_argument("--video",action="store_true")
    p = sub.add_parser("plan")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt")
    group.add_argument("--prompt-file")
    p.add_argument("--out",required=True)
    p.add_argument("--config")
    p = sub.add_parser("batch")
    p.add_argument("--manifest",default=str(ROOT / "devset" / "manifest.json"))
    p.add_argument("--split",choices=["train","dev","holdout"],required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--assets")
    p.add_argument("--browser")
    p = sub.add_parser("distill")
    p.add_argument("--batch",required=True)
    p.add_argument("--manifest",default=str(ROOT / "devset" / "manifest.json"))
    p.add_argument("--staging",required=True)
    p.add_argument("--name",required=True)
    p.add_argument("--assets",default=str(ASSETS))
    p.add_argument("--execute",action="store_true",help="Invoke the configured CLI to write staged candidates")
    p.add_argument("--config")
    p = sub.add_parser("bind")
    p.add_argument("--staging",required=True)
    p.add_argument("--assets",required=True)
    p = sub.add_parser("admit")
    p.add_argument("--baseline",required=True)
    p.add_argument("--candidate",required=True)
    p.add_argument("--proposal",required=True)
    p.add_argument("--assets",required=True)
    p.add_argument("--manifest",default=str(ROOT / "devset" / "manifest.json"))
    p.add_argument("--out",required=True)
    p = sub.add_parser("handoff")
    p.add_argument("--run",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--browser")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            deps = {}
            for name in ("playwright","Pillow","httpx"):
                try:
                    deps[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    deps[name] = None
            result = {"python":sys.version.split()[0],"dependencies":deps,
                      "executables":{x:shutil.which(x) for x in ("codex","claude","opencode","ffmpeg")},
                      "assets_present":all((ASSETS / x).is_file() for x in ("runtime.js","primitives.js","vendor/three.module.js")),
                      "next":"Run demo to verify WebGL. An executable on PATH does not prove browser launch succeeds."}
        elif args.command in ("demo","run"):
            from .orchestrator import run_task
            from .spec import load_spec
            result = run_task(load_spec(args.spec),Path(args.out),_config(args),assets_dir=args.assets,demo=args.command=="demo")
        elif args.command == "render-check":
            from .render import render_scene
            from .spec import load_spec
            result = render_scene(Path(args.scene),load_spec(args.spec),Path(args.out),assets_dir=args.assets,browser=args.browser,backend=args.backend,video=args.video)
        elif args.command == "plan":
            result = _plan(args)
        elif args.command == "batch":
            result = _batch(args)
        elif args.command == "distill":
            from .outerloop import prepare_distillation
            result = prepare_distillation(Path(args.batch),Path(args.manifest),Path(args.staging),candidate_name=args.name,baseline_assets=Path(args.assets))
            if args.execute:
                from .engine import run_agent
                staging = Path(args.staging).resolve()
                prompt_path = staging / "distill_prompt.txt"
                if not prompt_path.exists():
                    raise RuntimeError("Staging did not contain distill_prompt.txt; inspect preparation output")
                result["execution"] = run_agent(_config(args)["coder"],prompt_path.read_text(encoding="utf-8"),staging,staging / "trace",role="distiller")
        elif args.command == "bind":
            from .outerloop import bind_candidate
            result = bind_candidate(Path(args.staging),Path(args.assets))
        elif args.command == "admit":
            from .outerloop import admit_candidate
            result = admit_candidate(Path(args.baseline),Path(args.candidate),Path(args.manifest),proposal_path=Path(args.proposal),candidate_assets=Path(args.assets))
            write_json(args.out,result)
        elif args.command == "handoff":
            result = _handoff(args)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if args.command == "render-check" and not result["hard_pass"]:
            return 2
        if args.command == "run" and result["status"] != "accepted":
            return 2
        return 0
    except Exception as exc:
        print(json.dumps({"error":type(exc).__name__,"message":str(exc)},ensure_ascii=False),file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
