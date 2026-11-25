"""Thin subprocess adapters; the coding CLI owns its own tool-calling loop."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .util import run_process, write_json


def run_agent(config: dict, prompt: str, workspace: Path, trace_dir: Path, *,
              role="coder", session_id=None, images=(), output_schema: Path | None = None) -> dict:
    workspace, trace_dir = Path(workspace).resolve(), Path(trace_dir).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = trace_dir / f"{role}_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    final_path = workspace / f".{role}-last-message.txt"
    final_path.unlink(missing_ok=True)
    backend = config.get("backend", "codex")
    argv = []
    if backend == "codex":
        executable = config.get("executable", "codex")
        if not shutil.which(executable):
            raise RuntimeError(f"Coding CLI not found: {executable}")
        argv = [executable, "-a", "never", "-s", "workspace-write" if role in ("coder", "distiller") else "read-only", "exec"]
        if session_id:
            if role != "coder":
                raise ValueError("Evaluator and planner must never resume coder sessions")
            argv += ["resume", str(session_id)]
        else:
            argv += ["-C", str(workspace)]
        argv += ["--json", "--skip-git-repo-check", "-o", str(final_path)]
        if role not in ("coder", "distiller"):
            argv += ["--ephemeral"]
        if config.get("model"):
            argv += ["-m", config["model"]]
        if output_schema:
            argv += ["--output-schema", str(Path(output_schema).resolve())]
        # Only non-interactive CLI-supported images are passed; no hidden history.
        for image in images:
            argv += ["-i", str(Path(image).resolve())]
        argv += list(config.get("extra_args", []))
        argv += ["-"]
    elif backend == "command":
        template = config.get("resume_argv") if session_id else config.get("argv")
        if not isinstance(template, list) or not template or not all(isinstance(x, str) for x in template):
            raise ValueError("command backend requires a JSON argv array (and resume_argv for continuation)")
        substitutions = {"workspace":str(workspace),"prompt_file":str(prompt_path),"output_file":str(final_path),
                         "session_id":session_id or "", "images_json":json.dumps([str(Path(p).resolve()) for p in images])}
        # Only named placeholders are replaced: braces in user arguments remain literal.
        argv = list(template)
        for k, value in substitutions.items():
            argv = [x.replace("{" + k + "}", value) for x in argv]
    else:
        raise ValueError("Agent backend must be codex or command")
    result = run_process(argv, cwd=workspace, stdin=prompt, timeout=float(config.get("timeout_s", 240)))
    (trace_dir / f"{role}.stdout.jsonl").write_text(result.pop("stdout"), encoding="utf-8")
    (trace_dir / f"{role}.stderr.log").write_text(result.pop("stderr"), encoding="utf-8")
    output = final_path.read_text(encoding="utf-8") if final_path.exists() else ""
    returned_session = session_id
    for line in (trace_dir / f"{role}.stdout.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if backend == "codex" and event.get("type") == "thread.started":
            returned_session = event.get("thread_id") or returned_session
        elif backend == "command" and isinstance(event.get("session_id"), str):
            returned_session = event["session_id"]
        # Command adapters may emit a final event instead of writing output_file.
        if backend == "command" and event.get("type") == "result" and isinstance(event.get("text"), str):
            output = event["text"]
    result.update(backend=backend, role=role, session_id=returned_session,
                  session_continued=bool(session_id), argv=argv, output=output,
                  evidence_mode="live", output_file=str(final_path))
    write_json(trace_dir / f"{role}_result.json", result)
    return result
