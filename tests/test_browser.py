"""Real Chromium checks for the trusted Three.js runtime.

Run with PROXY_BROWSER=/path/to/chromium python -m unittest discover -s tests
     -p test_browser.py -v
Without PROXY_BROWSER these browser-dependent tests are explicitly skipped.
"""

from __future__ import annotations

import functools
import http.server
import os
from pathlib import Path
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
BROWSER = os.environ.get("PROXY_BROWSER")
FIXTURE_HTML = """<!doctype html><html><head>
<meta charset="utf-8"><style>body{margin:0}canvas{display:block}</style>
<script type="importmap">{"imports":{"three":"./vendor/three.module.js"}}</script>
</head><body><canvas id="scene"></canvas><script type="module">
import * as THREE from 'three';
import { createHarness } from './runtime.js';
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 640 / 360, .05, 100);
const renderer = new THREE.WebGLRenderer({
  canvas: document.getElementById('scene'), preserveDrawingBuffer: true
});
renderer.setSize(640, 360);
const object = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial());
scene.add(object);
object.position.z = -5;
window.fixture = { THREE, scene, camera, object, renderer };
createHarness({ scene, camera, renderer, update(t) { object.position.x = t; },
  subjects: [{ id: 'test', object }], seed: 42 });
window.renderAt(0);
</script></body></html>"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@unittest.skipUnless(BROWSER, "Set PROXY_BROWSER to a working Chromium executable to run real browser tests")
class BrowserRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright

        if not Path(BROWSER).is_file():
            raise RuntimeError("PROXY_BROWSER was provided but is not an existing browser executable")
        handler = functools.partial(_QuietHandler, directory=str(ROOT / "assets"))
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch(
                executable_path=BROWSER,
                headless=True,
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
        except BaseException:
            cls.playwright.stop()
            cls.server.shutdown()
            cls.server.server_close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 640, "height": 360}, device_scale_factor=1)
        self.page.set_default_timeout(15_000)
        self.page.route("**/case.html", lambda route: route.fulfill(body=FIXTURE_HTML, content_type="text/html"))
        self.page.goto(self.base_url + "/case.html")
        self.page.wait_for_function("window.__harnessReady === true")

    def tearDown(self):
        self.page.close()

    def measure(self, mutation: str):
        self.page.evaluate(mutation)
        return self.page.evaluate("window.exportManifest(0)")

    def test_01_box_nonuniform_rotated_exact(self):
        m = self.measure("fixture.object.scale.set(2,1,3); fixture.object.rotation.y=.4;")
        collider = m["objects"][0]["collider"]
        self.assertEqual(collider["type"], "obb")
        self.assertTrue(collider["exact"])
        for actual, expected in zip(collider["half_extents"], [1, .5, 1.5]):
            self.assertAlmostEqual(actual, expected)

    def test_02_sphere_uniform_exact(self):
        m = self.measure("fixture.object.geometry=new fixture.THREE.SphereGeometry(.5); fixture.object.scale.setScalar(2);")
        collider = m["objects"][0]["collider"]
        self.assertEqual(collider["type"], "sphere")
        self.assertTrue(collider["exact"])
        self.assertAlmostEqual(collider["radius"], 1)

    def test_03_sphere_nonuniform_rejected(self):
        m = self.measure("fixture.object.geometry=new fixture.THREE.SphereGeometry(.5); fixture.object.scale.set(2,1,1);")
        self.assertIsNone(m["objects"][0]["collider"])
        self.assertEqual(m["objects"][0]["collider_reason"], "nonuniform_sphere_scale")

    def test_04_sheared_box_rejected(self):
        m = self.measure("""(() => {
          const {THREE,scene,object}=fixture;
          const parent=new THREE.Group(); scene.add(parent); parent.add(object);
          parent.scale.set(2,1,1); object.rotation.y=.4;
        })()""")
        self.assertIsNone(m["objects"][0]["collider"])
        self.assertEqual(m["objects"][0]["collider_reason"], "sheared_world_transform")

    def test_05_near_clip_unknown_projection(self):
        m = self.measure("fixture.object.geometry=new fixture.THREE.BoxGeometry(.04,.04,.04); fixture.object.position.z=-.06;")
        obj = m["objects"][0]
        self.assertFalse(obj["projection_valid"])
        self.assertFalse(obj["fully_in_frame"])
        self.assertIsNone(obj["ndc_box"])
        self.assertIsNone(obj["projected_area_ratio"])
        self.assertEqual(obj["projection_reason"], "near_plane_crossing")

    def test_06_behind_unknown_projection(self):
        m = self.measure("fixture.object.position.z=1;")
        obj = m["objects"][0]
        self.assertFalse(obj["in_front"])
        self.assertFalse(obj["projection_valid"])
        self.assertIsNone(obj["ndc_box"])
        self.assertIsNone(obj["projected_area_ratio"])

    def test_07_modified_sphere_no_exact_collider(self):
        m = self.measure("""(() => {
          const {THREE,object}=fixture; object.geometry=new THREE.SphereGeometry(.5);
          const a=object.geometry.attributes.position;
          a.setX(3,a.getX(3)+.3); a.needsUpdate=true;
        })()""")
        self.assertIsNone(m["objects"][0]["collider"])

    def test_08_audit_unregistered_mesh(self):
        m = self.measure("""(() => {
          const {THREE,scene}=fixture;
          scene.add(new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshBasicMaterial()));
        })()""")
        self.assertEqual(m["audit"]["mesh_count"], 2)
        self.assertEqual(m["audit"]["registered_mesh_count"], 1)
        self.assertEqual(m["audit"]["unregistered_mesh_count"], 1)

    def test_09_downward_pitch_is_negative(self):
        m = self.measure("fixture.camera.lookAt(0,-1,-1);")
        self.assertAlmostEqual(m["camera"]["pitch_deg"], -45, places=6)

    def test_10_top_render_restores_scene_and_primary_camera(self):
        self.page.evaluate("window.renderAt(1.5)")
        snapshot = "({x:fixture.object.position.x,camera:fixture.camera.matrixWorld.elements.slice()})"
        before = self.page.evaluate(snapshot)
        self.page.evaluate("window.renderTopAt(.5)")
        self.assertEqual(self.page.evaluate(snapshot), before)


if __name__ == "__main__":
    unittest.main()
