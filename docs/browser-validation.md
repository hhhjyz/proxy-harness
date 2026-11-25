# Real browser validation

The trusted runtime was executed in a real headless Chromium process using the
SwiftShader software WebGL backend. The machine-readable record is
`browser-validation.json`. This is infrastructure validation of hand-authored
fixtures. It does not claim a live coding-model, visual-evaluator or video-model
run.

## Reproduce the same ten checks

Install the Python requirements, supply a working Chromium executable, then run:

```bash
PROXY_BROWSER=/path/to/chromium python -m unittest discover -s tests -p test_browser.py -v
```

Without `PROXY_BROWSER`, these ten tests are explicitly skipped; a skipped test
is not recorded as a successful browser validation. An explicitly configured
browser that fails to launch causes a test error.

The tests cover rotated/nonuniform boxes, uniform and nonuniform spheres, shear,
near-plane clipping, behind-camera geometry, mutated sphere buffers, unregistered
meshes, signed camera pitch, and restoring the primary state after a top view.
They serve the actual checked-in runtime and pinned Three.js files from a local
HTTP server. They do not replace measurements with expected data.

## Browser used for the recorded run

The standard Playwright browser download endpoint was unavailable in this
environment. The recorded run used the publicly distributed Linux browser from
the npm package `@sparticuz/chromium@158.0.0`, with its SHA-512 package integrity
verified before extraction. The executable reported `Chromium 158.0.8127.0`.
The package source and integrity are recorded in the JSON file. This browser
binary is not redistributed with the project and is not a project dependency.
Users may instead use a browser installed by Playwright or their own Chromium.

The demo was also checked by the project's `render-check` CLI: seven sampled
times and their shuffled repeats produced identical decoded pixels and identical
manifests, with no L1 failures or warnings. This establishes reproducibility for
the tested scenes, browser and backend; it is not a cross-GPU or continuous-time
guarantee. The contact sheet and top view were visually inspected.
