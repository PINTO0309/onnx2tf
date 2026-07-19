# `fb-refactor8` handoff

## Current state

- Branch: `fb-refactor8`.
- Starting checkpoint: `6ca2de11`, the merge of `fb-refactor7` into `main` and
  the `2.6.6` tag.
- Continue with coherent commits and pushes to `origin/fb-refactor8` only.
- Do not create, reopen, or update a pull request.
- Inference remains strictly sequential; no model-inference ProcessPool or
  parallel worker is permitted.

## Terminal Expand/Squeeze reconciliation characterization

The raw `_reconcile_static_tensor_shapes(model_ir)` immediately after
`_terminal_expand_squeeze_stats` has been inventoried. It currently discards
the legacy one-key result before `_advance_post_progress()`.

The characterization freezes the current boundary and proves why the call
cannot be guarded only by `_terminal_expand_squeeze_stats`: a graph with no
Expand/Squeeze operators can still contain stale Reshape output metadata from
an earlier late-layout rewrite, and the reconciliation repairs it. The test
fixture records zero Expand/Squeeze rewrites followed by one tensor-shape and
one complete static-shape mutation. Its existing explicit shape signature is
preserved, matching the current reconciler contract.

The owner has no caller-visible live graph index. Its only index is created
locally when dynamic Squeeze pre-operators must be inserted, so reusing an
index would require separately characterizing and changing the owner contract.
This unit deliberately selects only complete result retention.

The strict expected-failure contract requires the raw expression to become an
unconsumed assignment named
`_terminal_expand_squeeze_static_shape_stats`, requesting
`include_mutation_count=True`. Reconciliation remains unconditional and in the
same position; no consumer, guard, pass, scan, index, dependency, public API,
or TensorFlow behavior may change.

Characterization validation completed sequentially under `uv`:

- dedicated contract: `2 passed, 1 xfailed in 0.54s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result
assignment. No production source changed in this checkpoint.

The first all-direct-test collection subsequently stopped on an inherited
test-only import error. `test_flatbuffer_direct_shape_resolution.py` still
looked for `_set_operator_outputs` in the central lowerer, while the canonical
owner is `core.model_ir_utils`. `main` has the same stale import. The test
reference is repaired without restoring a production compatibility alias.

After the characterization tests pass with exactly one strict xfail, commit
and push the characterization checkpoint. Then implement only the selected
assignment, remove the xfail marker, repeat the focused and branch-wide gates,
record the results here and in `docs/fb_refactor8_improvements.md`, commit, and
push. Do not create or update a pull request.

## Terminal Expand/Squeeze reconciliation implementation checkpoint

The unconditional reconciliation now retains its opt-in complete dictionary
as `_terminal_expand_squeeze_static_shape_stats`. The target is unconsumed and
observation-only. No guard, index, scan, mutation, pass, order, progress,
dependency, public API, or TensorFlow behavior changed.

Implementation validation completed sequentially under `uv`:

- dedicated contract: `3 passed in 0.55s`;
- focused affected-owner, core, architecture, pass-efficiency, and
  TensorFlow-import-blocked gate: `381 passed in 27.03s`;
- implementation/test-repair changed files: `17 passed in 0.68s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The first focused run found one stale AST expectation and was `380 passed, 1
failed`; after updating that structural contract, the identical gate passed.
An attempted all-direct glob was not used as evidence because it included the
entire PyTorch exporter suite and its detached execution output could not be
recovered reliably. No model conversion was necessary for this
observation-only assignment.

At resume, inventory the next raw result-returning post-lowering boundary
without changing production behavior. Prefer a local assignment-only or
shared-index opportunity whose current mutation and cleanup evidence can be
fully characterized. Continue with commits and pushes only; do not create,
reopen, or update a pull request.
