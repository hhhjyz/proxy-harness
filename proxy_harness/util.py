from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reject NaN/Infinity rather than silently writing non-standard JSON.
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path, *, exclude=()):
    path = Path(path)
    files = {p.relative_to(path).as_posix(): file_hash(p) for p in sorted(path.rglob("*"))
             if p.is_file() and not p.is_symlink() and not any(x in p.relative_to(path).parts for x in exclude)}
    return canonical_hash(files)


def run_process(argv, *, cwd, stdin="", timeout=180, env=None):
    """No shell; terminate the process group on timeout, including browser children."""
    started = time.monotonic()
    try:
        proc = subprocess.Popen([str(x) for x in argv], cwd=str(cwd), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", env=env, start_new_session=True)
    except OSError as exc:
        return {"exit_code": 127, "timed_out": False, "stdout": "", "stderr": str(exc), "elapsed_s": 0}
    timed_out = False
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The process may have exited between timeout and termination.
        out, err = proc.communicate()
    return {"exit_code": proc.returncode, "timed_out": timed_out, "stdout": out,
            "stderr": err, "elapsed_s": round(time.monotonic() - started, 3)}


def parse_json_response(text):
    """Allow a single Markdown JSON fence, never guess at arbitrary prose braces."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines[0].strip().lower() not in ("```", "```json"):
            raise ValueError("Unexpected response fence")
        text = "\n".join(lines[1:-1])
    value = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
