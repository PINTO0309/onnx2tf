# `flatbuffer_direct` refactor continuation checkpoint — 2026-07-14

## Status

The active branch is `fb-refactor5`, created from `main` after pull request
`#949` merged the complete `fb-refactor4` checkpoint. The Goal is active again;
subsequent work uses coherent commits and pushes without opening another pull
request.

The latest implementation unit centralizes the repeated post-rewrite literal
static-shape cache update for aligned binary, Resize, and Pool statements in
the Torch-free `pytorch_fast_precanonicalize_policy.py` owner. The orchestrator
remains 308 lines at this checkpoint, down from 482 lines at Goal
resumption, 1,025 lines at the beginning of the previous continuation, and
1,608 lines before the broader fast-precanonicalize extraction.

## Completed work

The merged `fb-refactor4` checkpoints included:

- `062ddc4` — centralized DepthToSpace/Gather repair and the permuted-Conv
  statement decoder;
- `a522f1b` — centralized static NHWC Pool layout selection;
- `5eb86e1` — centralized CF Pool-neighbor repair with an explicit
  short-circuit contract;
- `d16265b` — centralized dynamic Pool layout repair and added the exact
  aligned-rank4 decoder;
- `e0bc280` — centralized simple-alias layout repair and moved all
  permuted-Conv decoder consumers into the policy owner;
- `afb5bb5` — centralizes aligned scalar-binary
  shape reconciliation and removes the now-unused aligned-rank4 and Softmax
  parser imports from the exporter.

The current `fb-refactor5` work contains five coherent continuations:

- `3ac19b40` centralizes the ordered fallback that repairs aligned binary
  shapes only when general binary repair made no change and the immediate next
  statement supplies matching BN, direct-return, or channel-first Resize
  evidence;
- `008e4ad0` centralizes the following Resize fallback, including
  exact direct and reshaped BN-constant parsing, preferred-channel selection,
  input-layout guards, and CF evidence propagation;
- `80d1d6a5` centralizes the aligned BatchNorm-constant rewrite itself while
  preserving the different direct and already-reshaped guards;
- `91a0a52d` centralizes LRN output evidence propagation without changing or
  broadening the generated source grammar;
- the current checkpoint centralizes literal static-shape recording while
  retaining each update at its original point in the ordered scan.

The extraction preserves the ordered source-rewrite behavior. Layout evidence
continues to mutate only the per-run CF/NHWC sets; repair context maps remain
shared. Rules that formerly used `continue` return an explicit short-circuit
result to the exporter. Exact generated-statement grammars remain rule-local or
use the shared Torch-free parser owner.

No dependency was added, no TensorFlow path was introduced, and no model
conversion or inference was run during these checkpoints.

## Current branch and changed files

Branch: `fb-refactor5`, tracking `origin/fb-refactor5`.

The final checkpoint changes:

- `onnx2tf/tflite_builder/pytorch_exporter.py`;
- `onnx2tf/tflite_builder/pytorch_fast_precanonicalize_policy.py`;
- `tests/test_flatbuffer_direct_pytorch_fast_precanonicalize_policy.py`;
- `tests/test_flatbuffer_direct_architecture.py`;
- `docs/flatbuffer_direct_architecture.md`;
- this handoff document.

The expected handoff state after committing and pushing is an empty `git
status --short` with local `fb-refactor5` equal to `origin/fb-refactor5`.

## Important design decisions

- The exporter remains the ordered orchestration owner; match/guard/rewrite
  decisions move to `pytorch_fast_precanonicalize_policy.py` one coherent
  family at a time.
- Indexed helpers receive the current line index, shared source lines, mutable
  layout evidence, and the shared repair context. They do not rescan the full
  generated source unless the preserved rule already required a bounded scan.
- Former loop `continue` behavior is represented explicitly in helper results;
  extraction must not silently allow later rules to run.
- General binary repair remains first. The downstream-evidence fallback is
  called only from its unchanged no-rewrite branch, and its returned CF
  evidence is visible to the following Resize repair in the same scan.
- The fallback deliberately retains its narrower positional grammar and legacy
  `_in` naming evidence. It additionally requires an immediate matching BN,
  direct return, or channel-first Resize; mismatched channels and mixed-layout
  names remain no-ops.
- General Resize repair also remains first. The input/BN-evidence fallback runs
  only afterward, uses an immediate matching direct or reshaped BN constant as
  the preferred channel hint when available, and otherwise retains the legacy
  input/source channel fallback. Its returned CF evidence remains visible to
  Pool and later aligned-constant decisions in the same ordered scan.
- Explicit NHWC Resize inputs and already-channel-first target shapes remain
  no-ops. BN evidence refines the preferred channel count but is not a
  prerequisite for the legacy CF-input repair.
- Direct aligned BatchNorm constants require a registered channel count that
  matches the generated target channel before a reshape is introduced.
  Already-reshaped constants intentionally retain the older, narrower rule:
  their explicit reshape channel drives normalization without requiring a
  registered-buffer channel lookup. Both forms still require CF input and a
  BatchNorm-derived attribute name.
- LRN output propagation is state-only: exact CF input evidence adds the output
  to the CF set, removes stale NHWC evidence, and copies only a known rank-four
  static input shape. It does not mark the source file changed or rewrite the
  LRN statement.
- Rewritten-shape caching accepts only a literal `target_shape=[...]` or the
  exact trailing aligned shape. Dynamic and unparseable expressions do not
  replace existing cache entries, and binary/Resize/Pool callers still update
  the shared context immediately after their successful rewrite.
- Shared parsers preserve the exact old generated syntax when broadening would
  change rule eligibility. Parser ownership tests prevent duplicate exporter
  implementations and unused compatibility imports.
- No real-model conversion gate is required for these mechanical checkpoints
  under the current instruction to prioritize implementation and minimize
  conversion tests. This does not prove broad corpus regression safety.

## Tests executed

The resumed downstream-binary, Resize-evidence, aligned-BatchNorm, LRN, and
static-shape-cache checkpoints passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pytorch_fast_precanonicalize_policy.py \
  tests/test_flatbuffer_direct_architecture.py

124 passed
```

Seven pre/post-extraction characterization cases also preserve the exact
orchestrator output for matching BN, direct return, channel-first Resize,
channel mismatch, channel-last Resize, mixed operands, and an already-CF shape.
Five additional pre/post-extraction cases preserve direct BN, reshaped BN,
no-BN fallback, NHWC-input no-op, and already-CF Resize behavior.
Seven aligned-BatchNorm cases preserve direct rewrite, non-BN no-op, channel
mismatch no-op, NHWC-input no-op, reshaped rewrite, already-CF behavior, and
reshaped non-BN no-op.
The LRN checkpoint additionally passed a four-test selection covering CF/NHWC
and static-shape state, Pool/LRN interaction, architecture ownership, and the
existing generated-source integration case.
The cache checkpoint passed four focused cases covering aligned binary,
Resize/Pool, literal recording, parse-failure no-op, and architecture ownership.
The exporter and policy pass `python -m py_compile`, and `git diff --check`
passes. The immediately preceding DepthToSpace, Pool, dynamic-Pool,
simple-alias, and aligned-scalar checkpoints passed their focused synthetic and
ownership selections.

## Failing tests and known issues

- No newly failing focused test is known at this checkpoint.
- A whole-file Ruff run on `pytorch_exporter.py` reports 282 pre-existing
  compatibility re-export, unused scaffold, and undefined-name findings. It is
  not used as the scoped checkpoint gate; changed owners/tests pass Ruff and
  the exporter passes syntax compilation.
- The optional PyTorch exporter suite runs when the host's Python 3.10
  `LD_LIBRARY_PATH` and `PYTHONPATH` are removed from the command environment.
  The focused results, restored native-codegen bindings, real-model artifact
  gate, and remaining inherited failures are recorded in
  `docs/flatbuffer_direct_pytorch_regression_2026-07-14.md`.
- The optional TensorFlow suite was not synchronized or run.
- Recent PyTorch source-policy checkpoints have not been followed by a Tier
  corpus conversion run. This is intentional under the current minimal-
  conversion instruction, but broad model-level regression remains unproven.

## Unfinished work

The full Goal is not complete. The fast-precanonicalize orchestrator still has
308 lines. Its remaining body is primarily the intended ordered helper
orchestration. Remaining state-update glue includes:

- applying NHWC AveragePool bridge layout/shape results before subsequent Pool
  decisions.

The broader fixed-pipeline, exporter, artifact-matrix, optional TensorFlow,
PyTorch/TorchScript/Dynamo/ExportedProgram, and full Tier regression work also
remains subject to the original refactor plan and its verification gates.

## Next work

1. Confirm `git status --short --branch` is clean and local `fb-refactor5`
   matches `origin/fb-refactor5`.
2. Inspect NHWC AveragePool-to-binary bridge state application in
   `_apply_fast_precanonicalize_repairs`.
3. Characterize the returned-name, CF/NHWC-set, normalized-shape, and no-op
   behavior before co-locating the state update with its policy rewrite.
4. Preserve the immediate reparse of the rewritten Pool statement before the
   following Pool layout decisions.
5. Run only the focused synthetic/ownership/static checks unless the user asks
   for broader conversion validation. Use `uv`, run inference sequentially if
   any is explicitly requested, commit and push coherent units, and do not
   create a pull request.
