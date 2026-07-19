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

## Core dynamic-Reshape result characterization

The next selected raw boundary is `_resolve_dynamic_reshape_shapes(model_ir)`
inside core cleanup. It has a fixed one-key result, no cleanup or layout-state
side effect, and is positioned between `_core_cleanup_conv_activation_stats`
and `_core_cleanup_squeeze_reshape_identity_stats`.

The characterization fixture freezes a positive resolution of `[-1, 2]` to
`[2, 2]`, including the operator option, shape constant, output shape, and
output signature. A strict expected failure requires observation-only
assignment to `_core_cleanup_dynamic_reshape_stats` with the same sole
`model_ir` argument. Do not add a consumer, guard, graph index, cleanup, or
pass, and do not change the surrounding core-cleanup order.

Characterization validation completed sequentially under `uv`: the dedicated
contract is `2 passed, 1 xfailed in 0.54s`, and targeted Ruff, bytecode
compilation, and whitespace checks pass. The sole expected failure is the
unimplemented assignment. Commit and push this characterization before
changing production code; do not create or update a pull request.

## Core dynamic-Reshape result implementation checkpoint

The core-cleanup call now retains its unchanged dictionary as
`_core_cleanup_dynamic_reshape_stats`. It remains unconsumed. Arguments,
execution count, graph mutation, order, following Squeeze/Reshape cleanup,
dependency set, public behavior, and TensorFlow isolation are unchanged.

Implementation validation completed sequentially under `uv`:

- dedicated contract: `3 passed in 0.54s`;
- focused affected-owner, convergence, core, architecture, pass-efficiency,
  terminal orchestration, and TensorFlow-import-blocked gate:
  `452 passed in 28.87s`;
- all `fb-refactor8` changed test files: `20 passed in 0.76s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No model conversion was required for this observation-only assignment. After
this checkpoint is committed and pushed, inventory the raw
`_apply_safe_transpose_reduction_lite(model_ir)` result before changing it.
Freeze its schema, mutation/cleanup semantics, surrounding conditional
boundary, and whether the existing pass already owns all required state.
Continue with commits and pushes only; do not create or update a pull request.

## Safe-transpose reduction result characterization

The raw no-layout fallback call to
`_apply_safe_transpose_reduction_lite(model_ir)` has been selected. Its fixed
three-key schema represents the applied pass count, net Transpose reduction,
and unbound-input rollback count. The owner itself owns the complete snapshot,
curated pass sequence, prune/reconcile work, validation, and rollback, so this
boundary has no live state to share safely.

A strict expected failure requires assignment to
`_no_layout_safe_transpose_reduction_stats` immediately before
`_no_layout_fallback_affine_prepost_stats`. The result remains unconsumed; do
not change the `elif` condition, arguments, transaction, pass sequence,
following affine cleanup, dependency set, public behavior, or TensorFlow
boundary.

Characterization validation completed sequentially under `uv`: the dedicated
contract is `2 passed, 1 xfailed in 0.55s`, and targeted Ruff, bytecode
compilation, and whitespace checks pass. The sole expected failure is the
unimplemented assignment. Commit and push this contract before changing
production code; do not create or update a pull request.
