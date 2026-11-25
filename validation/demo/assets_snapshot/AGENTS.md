# Proxy scene authoring contract

Edit only `scene.html` and `appearance_prompt.txt` in the task directory.
The controller owns `runtime.js`, `primitives.js`, `vendor/` and this contract.
Do not replace, shadow, monkeypatch or modify those trusted assets or capture hooks.

## Required scene interface

- One editable HTML page, one `<canvas id="scene">`.
- Import Three.js from the local import map: `./vendor/three.module.js`.
- Import `createHarness` from `./runtime.js`; call it once with scene, camera,
  renderer, update, subjects and the task seed.
- Use `THREE.PerspectiveCamera`, `THREE.WebGLRenderer`, built-in materials and
  rigid mesh geometry. Keep output at the supplied viewport dimensions.
- Register every mesh exactly once, including ground or decoration under its
  respective role. Subject IDs must match the task specification.
- Each `update(t)` must assign all animated state from absolute seconds `t`.
  Calling it for 2, then 0, then 2 seconds must recover the same state and image.
- `update(t)` is synchronous. Finish any asset loading before creating the harness.
- No `requestAnimationFrame`, timers, real-time clock, `Date.now`,
  `performance.now`, unseeded randomness or incremental animation state.
- Initialize seeded random values once; do not consume random draws in update.
- Do not replace `renderAt`, `exportManifest`, `renderTopAt`, `__harnessReady`
  or `__harnessMeta`. The harness installs them as non-writable properties.
- Keep camera and scene transforms finite. Camera pitch is forward elevation
  above the horizontal XZ plane: looking down means a negative pitch in degrees.

## Measured evidence

Manifest bounds are derived from real position buffers and world transforms.
Screen area is the viewport-clipped projected vertex bounding-box area, not a
segmentation-mask area. Visibility flags do not prove an object is unoccluded.
Near/far clipping or geometry crossing the camera plane invalidates projection;
the manifest then reports null NDC/area and a reason. Keep required subjects well
inside the camera frustum instead of relying on clipping or oversized bounds.

Exact collision proxies are emitted only for unmodified full BoxGeometry or
SphereGeometry on a single mesh. Shear, nonuniform sphere scaling, composites,
other geometry and instancing do not get an exact collider. AABB overlap alone
does not establish a collision. Use separated simple geometry for hard gates.
Skinned/batched meshes, morph targets, custom shader materials, displacement maps
and custom onBeforeCompile callbacks are unsupported in this v1 runtime.
Object/material render callbacks are also unsupported; mutate state in update(t).
Use full draw ranges and one built-in material per mesh; split material groups
into separate registered meshes if necessary.

## Deliverables

`scene.html`: neutral, readable geometry and required motion, with local imports.
`appearance_prompt.txt`: target finished-video appearance in plain text.
Do not claim visual quality, task success or model execution without the emitted
runtime and evaluation evidence. Do not alter validators to make a task pass.
