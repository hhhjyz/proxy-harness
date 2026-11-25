# Original scene suite

The suite contains **6 train, 24 dev and 6 holdout scene specifications** written
for this project. They are **not copied CodeVideoBench questions**, are not an
official benchmark reproduction, and do not establish performance until a live
model actually runs them.

| Split | Purpose | Coverage |
| --- | --- | --- |
| Train, 6 | Evidence for staged guidance/primitive proposals | Basic registration, fixed framing, straight/periodic motion, a camera dolly, staggered timing and collision clearance |
| Dev, 24 | Complete, paired admission | Four layout/framing cases; four trajectory/timing cases; four camera cases; four interaction/constraint cases; four articulation/support cases; four combined state/camera/visibility cases |
| Holdout, 6 | Separate final check after a candidate is chosen | New combinations of multiple subjects, support, articulation, camera motion, reversal and layout change |

All scenes are short, low-resolution structural proxies with explicit subject
IDs and fixed seeds. The suite intentionally exercises the capture contract
and visible structure before photorealism. Train/dev/holdout use disjoint task
IDs and prompts. They share task families: this is a small engineering devset,
not evidence of broad distributional generalization.

The manifest pins each **normalized specification** and separately pins the
manifest itself in `manifest.sha256`. `load_manifest()` checks both. Re-running
`build_manifest.py` is an intentional dataset-authoring operation; changing a
specification or manifest invalidates comparisons against older hashes. Do not
regenerate the manifest during an experiment to make edited tasks appear valid.

## What the checks do and do not establish

- Geometry and frame checks enforce registered subjects, specified camera
  ranges, declared screen-area limits and eligible exact collision checks.
- Passing an AABB/frustum check does not establish pixel-level visibility or
  correct physical causality. Visual evaluation still has to inspect actual
  frames and the temporal prompt.
- Allowed contact pairs allow necessary support or articulation. They do not
  certify nonpenetration, contact dynamics or conservation laws.
- Sparse sample times can miss short violations. The determinism and render
  checks do not turn the three-second scene into a validated simulator.
- Holdout specs ship for reproducibility. Access separation in the outer loop
  prevents this workflow from feeding them into distillation; it is not a
  secrecy guarantee against an operator who opens the files manually.

Coverage labels in `manifest.json` are descriptive metadata. They are never
substituted for measured scores or interpreted as successful outcomes.
