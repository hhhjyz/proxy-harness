# Staged outer loop: train evidence, asset proposal, paired admission

The outer loop does not silently improve or replace the current harness. It
prepares evidence for a coding agent, stages a proposed asset change, and
decides whether a **complete paired dev experiment** makes that candidate
eligible for review. Eligibility never copies files into the main asset tree.

## Dataset boundary

`devset/manifest.json` contains 6 original train, 24 original dev and 6 original
holdout scene specifications. `devset/COVERAGE.md` explains their scope. These
are project-authored tasks, not CodeVideoBench questions or a claimed benchmark
reproduction. No generated score is shipped as a measured model result.

- **Train:** the only source of model-facing distillation evidence. A live
  failed attempt is useful evidence and is not relabeled as a success.
- **Dev:** all 24 tasks, at their pinned seeds/specifications, are required on
  both sides of admission. Partial batches and duplicate tasks are rejected.
- **Holdout:** separate final evaluation after choosing a candidate. It cannot
  enter either the distillation request or the admission comparison.

The files are public within this project for reproducibility; the separation
is enforced by the outer-loop code and model request, not by concealing the
holdout files from an operator with filesystem access.

## Hash and condition contract

`canonical_sha256(value)` hashes UTF-8 JSON with sorted keys,
`separators=(",", ":")`, `ensure_ascii=False` and non-finite values rejected.
Specification hashes apply to `validate_spec(spec)`, including its defaults.
The manifest hash is pinned separately in `manifest.sha256` to avoid a
self-referential JSON object. `load_manifest()` verifies both layers.

`hash_tree(directory)` hashes a mapping from POSIX-relative file paths to raw
file SHA-256 digests. Python import caches are ignored so importing a module
does not change a source hash. Symlinks are rejected. Asset-change inventories
still inspect all files, including names that must not be added to a candidate.

```python
condition_id = canonical_sha256({
    "manifest_sha256": manifest_sha256,
    "config_sha256": config_sha256,
    "harness_sha256": harness_sha256,
    "asset_sha256": asset_sha256,
})
```

The baseline and candidate have different condition IDs when their assets
differ. Pairing requires identical manifest, configuration and harness hashes;
the proposal separately binds each asset hash and condition ID. The comparable
configuration includes the actual coder/evaluator settings, iteration budget,
render settings and experimental flags. Specifications bind each task's seed,
time samples and quality thresholds. A dataset or harness change calls for a
new baseline, not a reinterpretation of old results.

## 1. Stage verified raw train evidence

```python
from pathlib import Path
from proxy_harness.outerloop import prepare_distillation

prepared = prepare_distillation(
    Path("runs/train/batch_results.json"),
    Path("devset/manifest.json"),
    Path("staging/motion-v1"),
    candidate_name="motion-v1",
    baseline_assets=Path("assets"),
)
```

The function verifies each selected run against its root `summary.json`,
normalized `spec.json`, `config.json`, asset snapshot and selected iteration.
It requires `model_mode="live"` and original coding evidence. A copied demo,
an absent raw coding response, a spoofed split, or a mismatched score/geometry
claim cannot provide a training example.

Mixed batches can be supplied. Non-train records are excluded **without
opening their run directories**. For an eligible train run, the stage retains
**every recorded `iter_XX`**, including failed drafts, repair attempts and any
later attempts after the selected result. `selected_iteration` verifies the
summary and never cuts the training trajectory down to its final good scene.
Missing iteration gaps or a contradictory recorded iteration count are
rejected.

Only the raw-artifact allowlist is copied from each iteration: source and
appearance text, diffs, coding/evaluator requests and responses (including
`eval_wire_request.json` when present), raw CLI traces, console and render
records, L1 reports, manifests, image sheets, sample images, retained frame
sequences and any proxy MP4. Root `history.json` and `config.json` are retained
with the specification and summary. Each source path is checked, including
early-round frame paths; symlinks cannot introduce external files. Every file
is hashed before staging, rechecked during copying and preserved byte for
byte in the staged evidence pins.

Exact dev/holdout task IDs and complete prompt copies are checked in **all
copied text across the trajectory**, including earlier failed iterations and
root history/config. This catches direct contamination; it is not a claim
that arbitrary semantic paraphrases can be detected.

The stage contains:

- `distill_prompt.txt`: the concrete task that the CLI may give to an agent.
- `proposal_request.json`: selected train artifacts and permissible changes.
- `train_evidence/`: byte-preserved complete per-case iteration histories,
  sources, diffs, raw coding/evaluation traces, measurements and all available
  sampled/rendered observations from the permitted artifact paths.
- `candidate_assets/`: a complete copy of the baseline asset tree.
- `proposal.json`: sealed source metadata, file pins and candidate state.

Preparation does not invoke a model. The CLI's explicit `distill --execute`
option can invoke the configured coding agent using `distill_prompt.txt`.
There is one proposal writer; evaluating or promoting the proposal is outside
that writer's task.

The proposal may modify `primitives.js` or project guidance under `skills/`.
It may not modify runtime code, the renderer/checker, base scene template,
vendor files, task specs, thresholds or copied source evidence. A future
distiller can create `skills/<name>/SKILL.md` as a **project asset**; nothing is
installed into the assistant's personal skill collection. No untrained skill
is shipped as a proven improvement.

## 2. Bind the candidate before evaluating it

```python
from proxy_harness.outerloop import bind_candidate

bound = bind_candidate(
    Path("staging/motion-v1"),
    Path("staging/motion-v1/candidate_assets"),
)
```

Binding verifies unchanged source evidence and the allowed asset-edit scope.
Project skills need a matching lowercase-hyphen name and nonempty single-line
description in frontmatter. Explicit Markdown and `references/` or `scripts/`
references must resolve safely inside the candidate asset tree and exist.
The validation does not interpret arbitrary natural language references or
execute any skill.

Binding hashes the **whole** candidate asset tree and derives its condition
ID. Changing an asset after the candidate experiment invalidates admission.
Rebinding is possible only as a new experimental condition: a batch generated
with the previous hash no longer matches.

## 3. Compare complete paired dev runs

```python
from proxy_harness.outerloop import admit_candidate

report = admit_candidate(
    Path("runs/dev-baseline/batch_results.json"),
    Path("runs/dev-candidate/batch_results.json"),
    Path("devset/manifest.json"),
    proposal_path=Path("staging/motion-v1/proposal.json"),
    candidate_assets=Path("staging/motion-v1/candidate_assets"),
)
```

Both batches must contain exactly the pinned dev task set. Each record must
match its manifest identity, seed, normalized spec and run summary. The code
also checks the original run configuration and asset snapshot. Identical
models or equal average scores alone are insufficient pairing evidence.

Acceptance requires an actual evaluated result, a passing L1 report, complete
required subjects, a valid score above that spec's threshold, and live-run
provenance. A high VLM score cannot override a hard failure. Skipped or
unverified runs never become successes; this first version requires live
verified runs throughout an admission batch and rejects incomplete evaluation
provenance instead of treating it as proof of an improvement.

Complexity is remeasured from stored artifacts:

- Code lines: the count of nonempty lines in the selected `scene.html`.
- Primitive count: the maximum `audit.primitive_count` across its sampled
  `manifests.json` entries.

No task may regress on a previously passing hard gate, required-subject
completeness or acceptance. Subject deletion cannot be sold as simplification.
After these requirements, either condition can establish eligibility:

1. **More accepted dev tasks.** The pass rate rises on the same full task set.
2. **Less complexity on the shared passed tasks.** Sum code lines and sum
   primitive count over the baseline/candidate passed intersection. Neither
   sum may increase and at least one must strictly decrease. An empty
   intersection, a code/primitive tradeoff, or dropping a complex failed case
   from the denominator does not satisfy the rule.

The result includes task counts, pass rates, the exact passed intersection,
complexity totals, any regressions and the two bound condition IDs. It returns
`eligible=true` only for a candidate satisfying all gates. The output always
says `promotion="not_performed"`; a reviewer still decides whether to adopt
the proposed assets. This small deterministic policy is an engineering gate,
not a statistical significance claim or a guarantee of generalization.

## Verification

`tests/test_outerloop.py` uses clearly labeled temporary fixtures to exercise
contamination prevention, split spoofing, immutable evidence, incomplete or
duplicated task sets, mismatched conditions, stale candidate hashes, invented
complexity reductions, hard/subject regressions and skipped evaluators. The
fixtures exercise code paths and do not claim that any live model generated
their scenes or scores.
