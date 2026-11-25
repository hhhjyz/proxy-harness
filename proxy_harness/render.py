"""Trusted capture process. Generated JS executes in a separately timed-out worker."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import random
import shutil
import sys
import tempfile
import threading
from urllib.parse import unquote, urlsplit

from .util import ASSETS, ROOT, canonical_hash, file_hash, read_json, run_process, write_json


class LocalHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        root = Path(self.directory).resolve()
        requested = (root / unquote(urlsplit(self.path).path).lstrip("/")).resolve()
        if not requested.is_relative_to(root) or not requested.is_file():
            self.send_error(404)
            return
        return super().do_GET()


def _pixel_hash(png):
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    return hashlib.sha256(im.tobytes()).hexdigest()


def _contact_sheet(paths, times, target):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=17)
    tiles = []
    for path, t in zip(paths, times):
        im = Image.open(path).convert("RGB")
        im.thumbnail((384, 216))
        tile = Image.new("RGB", (384, 248), "#e9edf2")
        tile.paste(im, ((384-im.width)//2, 30+(216-im.height)//2))
        ImageDraw.Draw(tile).text((12, 5), f"t = {t:.3f} s", font=font, fill="#17253b")
        tiles.append(tile)
    cols = min(4, max(1, len(tiles)))
    sheet = Image.new("RGB", (cols*384, math.ceil(len(tiles)/cols)*248), "#d7dde5")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols)*384, (i//cols)*248))
    sheet.save(target)


def _worker(job):
    from playwright.sync_api import sync_playwright
    from .checks import check_scene
    spec, out = job["spec"], Path(job["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    errors, logs, manifests, frame_paths = [], [], [], []
    determinism = {"passed": False, "comparisons": [], "tested_times": []}
    meta = {"schema_version": "1.0", "backend_requested": job["backend"],
            "seed": spec["seed"], "viewport": [spec["width"], spec["height"]],
            "scope": "Same browser process and backend; sampled frames only, not a continuous-time proof."}
    scene_path = Path(job["scene_path"])
    source = scene_path.read_text(encoding="utf-8")
    try:
        with tempfile.TemporaryDirectory(prefix="proxy-render-") as temp:
            root = Path(temp)
            shutil.copyfile(scene_path, root / "scene.html")
            assets = Path(job.get("assets_dir") or ASSETS)
            # No candidate-writable runtime is ever served directly.
            for name in ("runtime.js", "primitives.js"):
                shutil.copyfile(assets / name, root / name)
            shutil.copytree(assets / "vendor", root / "vendor")
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(LocalHandler, directory=str(root)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            origin = f"http://127.0.0.1:{server.server_port}"
            try:
                with sync_playwright() as p:
                    args = ["--force-device-scale-factor=1", "--hide-scrollbars"]
                    if job["backend"] == "swiftshader":
                        args += ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
                    elif job["backend"] != "native":
                        raise ValueError("backend must be swiftshader or native")
                    # An executable named chromium can be a broken distro wrapper.
                    # Prefer Playwright's tested browser unless explicitly configured.
                    executable = job.get("browser") or None
                    browser = p.chromium.launch(headless=True, executable_path=executable, args=args)
                    meta.update(browser_version=browser.version, browser_executable=executable or "playwright-bundled", launch_args=args)
                    context = browser.new_context(viewport={"width":spec["width"], "height":spec["height"]},
                                                  device_scale_factor=1, locale="en-US", timezone_id="UTC",
                                                  reduced_motion="reduce", service_workers="block")
                    def route_request(route):
                        if route.request.url.startswith(origin + "/"):
                            route.continue_()
                        else:
                            errors.append("Blocked external network: " + route.request.url[:500])
                            route.abort()
                    context.route("**/*", route_request)
                    if hasattr(context, "route_web_socket"):
                        context.route_web_socket("**/*", lambda ws: ws.close())
                    page = context.new_page()
                    page.set_default_timeout(12000)
                    def on_console(msg):
                        logs.append(f"[{msg.type}] {msg.text}")
                        if msg.type == "error":
                            errors.append(msg.text)
                    page.on("console", on_console)
                    page.on("pageerror", lambda e: errors.append("JS: " + str(e)))
                    page.on("requestfailed", lambda r: errors.append(f"Request failed: {r.url} {r.failure}"))
                    page.on("response", lambda r: errors.append(f"HTTP {r.status}: {r.url}") if r.status >= 400 else None)
                    page.goto(origin + "/scene.html", wait_until="load", timeout=15000)
                    page.wait_for_function("window.__harnessReady === true && typeof window.renderAt === 'function' && typeof window.exportManifest === 'function'", timeout=12000)
                    meta["runtime"] = page.evaluate("window.__harnessMeta || null")
                    canvas = page.locator("canvas#scene")
                    if canvas.count() != 1:
                        raise ValueError("Exactly one canvas#scene is required")
                    (out / "frames").mkdir(exist_ok=True)
                    baseline = {}
                    def capture(t):
                        manifest = page.evaluate("t => {window.renderAt(t); return window.exportManifest(t);}", t)
                        png = canvas.screenshot(type="png", animations="disabled", timeout=12000)
                        return manifest, png
                    for index, t in enumerate(spec["sample_times"]):
                        manifest, png = capture(t)
                        path = out / "frames" / f"sample_{index:02d}.png"
                        path.write_bytes(png)
                        manifests.append(manifest)
                        frame_paths.append(path)
                        baseline[t] = (_pixel_hash(png), canonical_hash(manifest))
                    # Repeat every sampled time after a seeded shuffle, including first/middle/last.
                    order = list(spec["sample_times"])
                    random.Random(spec["seed"]).shuffle(order)
                    if order == spec["sample_times"] and len(order) > 1:
                        order.reverse()
                    for t in order:
                        manifest, png = capture(t)
                        actual = (_pixel_hash(png), canonical_hash(manifest))
                        determinism["comparisons"].append({"t":t, "pixels_equal":actual[0] == baseline[t][0],
                                                          "manifest_equal":actual[1] == baseline[t][1],
                                                          "passed":actual == baseline[t],
                                                          "pixel_sha256":actual[0], "manifest_sha256":actual[1],
                                                          "baseline_pixel_sha256":baseline[t][0],
                                                          "baseline_manifest_sha256":baseline[t][1]})
                    determinism["tested_times"] = order
                    determinism["passed"] = all(c["passed"] for c in determinism["comparisons"])
                    midpoint = spec["sample_times"][len(spec["sample_times"])//2]
                    page.evaluate("t => window.renderTopAt(t)", midpoint)
                    canvas.screenshot(path=str(out / "top_view.png"), animations="disabled")
                    _contact_sheet(frame_paths, spec["sample_times"], out / "contact_sheet.png")
                    from PIL import Image
                    if Image.open(frame_paths[0]).size != (spec["width"], spec["height"]):
                        errors.append("Canvas dimensions differ from the fixed Scene Spec viewport")
                    if job.get("video"):
                        if not shutil.which("ffmpeg"):
                            raise RuntimeError("ffmpeg is required for proxy.mp4 export")
                        video_frames = out / "video_frames"
                        video_frames.mkdir(exist_ok=True)
                        count = math.ceil(spec["duration_s"] * spec["fps"])
                        for i in range(count):
                            _, png = capture(i / spec["fps"])
                            (video_frames / f"{i:06d}.png").write_bytes(png)
                        cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(spec["fps"]),
                               "-i", str(video_frames / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                               "-movflags", "+faststart", str(out / "proxy.mp4")]
                        ff = run_process(cmd, cwd=out, timeout=90)
                        (out / "ffmpeg.log").write_text(ff["stderr"], encoding="utf-8")
                        if ff["exit_code"] != 0:
                            raise RuntimeError("ffmpeg encode failed; see ffmpeg.log")
                        meta["video"] = {"fps":spec["fps"],"frames":count,"duration_s":count/spec["fps"],
                                         "sha256":file_hash(out / "proxy.mp4")}
                        # All video frames retained: raw evidence can be selectively archived by the user later.
                    context.close()
                    browser.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
    except Exception as exc:
        errors.append(f"Renderer {type(exc).__name__}: {exc}")
    write_json(out / "manifests.json", manifests)
    write_json(out / "determinism.json", determinism)
    write_json(out / "render_meta.json", meta)
    (out / "console.log").write_text("\n".join(logs + ["[ERROR] " + x for x in errors]) + "\n", encoding="utf-8")
    report = check_scene(spec, manifests, runtime_errors=errors, determinism=determinism, source=source)
    write_json(out / "l1_report.json", report)
    return report


def render_scene(scene_path: Path, spec: dict, output_dir: Path, *, browser: str | None = None,
                 backend: str = "swiftshader", video: bool = False, assets_dir: Path | None = None,
                 timeout_s: float = 120) -> dict:
    from .checks import check_scene
    from .spec import validate_spec
    spec = validate_spec(spec)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Never consume stale successful reports after a failed worker.
    for name in ("l1_report.json", "proxy.mp4", "contact_sheet.png", "top_view.png"):
        (output_dir / name).unlink(missing_ok=True)
    job = {"scene_path":str(Path(scene_path).resolve()), "spec":spec, "output_dir":str(output_dir),
           "browser":browser, "backend":backend, "video":video,
           "assets_dir":str(Path(assets_dir).resolve()) if assets_dir else str(ASSETS)}
    write_json(output_dir / "render_job.json", job)
    result = run_process([sys.executable, "-m", "proxy_harness.render", "--worker", str(output_dir / "render_job.json")],
                         cwd=ROOT, timeout=timeout_s)
    (output_dir / "render_worker.log").write_text(result["stdout"] + result["stderr"], encoding="utf-8")
    if result["exit_code"] != 0 or not (output_dir / "l1_report.json").exists():
        error = "Render worker timed out" if result["timed_out"] else f"Render worker exited {result['exit_code']}; see render_worker.log"
        report = check_scene(spec, [], runtime_errors=[error], determinism={"passed":False,"comparisons":[],"tested_times":[]})
        write_json(output_dir / "l1_report.json", report)
        (output_dir / "console.log").write_text(error + "\n", encoding="utf-8")
        return report
    return read_json(output_dir / "l1_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    _worker(read_json(parser.parse_args().worker))
