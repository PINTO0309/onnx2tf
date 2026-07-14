# `flatbuffer_direct` refactor continuation checkpoint — 2026-07-14

## Status

The active branch is `fb-refactor5`, created from `main` after pull request
`#949` merged the complete `fb-refactor4` checkpoint. The Goal is active again;
subsequent work uses coherent commits and pushes without opening another pull
request.

The latest implementation unit moves the downstream-evidence aligned-binary
fallback out of `_apply_fast_precanonicalize_repairs` and into the Torch-free
`pytorch_fast_precanonicalize_policy.py` owner. The orchestrator is 421 lines at
this checkpoint, down from 482 lines at Goal resumption, 1,025 lines at the
beginning of the previous continuation, and 1,608 lines before the broader
fast-precanonicalize extraction.

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

The current `fb-refactor5` checkpoint centralizes the ordered fallback that
repairs aligned binary shapes only when general binary repair made no change
and the immediate next statement supplies matching BN, direct-return, or
channel-first Resize evidence.

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
- Shared parsers preserve the exact old generated syntax when broadening would
  change rule eligibility. Parser ownership tests prevent duplicate exporter
  implementations and unused compatibility imports.
- No real-model conversion gate is required for these mechanical checkpoints
  under the current instruction to prioritize implementation and minimize
  conversion tests. This does not prove broad corpus regression safety.

## Tests executed

The resumed downstream-evidence checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pytorch_fast_precanonicalize_policy.py \
  tests/test_flatbuffer_direct_architecture.py

120 passed
```

Seven pre/post-extraction characterization cases also preserve the exact
orchestrator output for matching BN, direct return, channel-first Resize,
channel mismatch, channel-last Resize, mixed operands, and an already-CF shape.
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
421 lines. Its largest remaining in-loop blocks are:

- 65 lines: Resize shape repair selected by following BN constant evidence;
- 32 and 26 lines: reshaped and direct aligned BN-constant repair;
- smaller Pool state-update glue and local-response-normalization shape repair.

The broader fixed-pipeline, exporter, artifact-matrix, optional TensorFlow,
PyTorch/TorchScript/Dynamo/ExportedProgram, and full Tier regression work also
remains subject to the original refactor plan and its verification gates.

## Next work

1. Confirm `git status --short --branch` is clean and local `fb-refactor5`
   matches `origin/fb-refactor5`.
2. Inspect the 65-line Resize repair selected by following BN constant
   evidence in `_apply_fast_precanonicalize_repairs`.
3. Characterize direct and reshaped BN constant evidence, channel-count
   agreement, and no-op cases before moving the exact decision to the policy
   owner.
4. Preserve the current ordering after downstream binary evidence and before
   the later aligned BN-constant repairs.
5. Run only the focused synthetic/ownership/static checks unless the user asks
   for broader conversion validation. Use `uv`, run inference sequentially if
   any is explicitly requested, commit and push coherent units, and do not
   create a pull request.
