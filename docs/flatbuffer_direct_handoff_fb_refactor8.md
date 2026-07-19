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

## Fallback norm reconciliation characterization

The remaining raw `_reconcile_static_tensor_shapes(fallback_ir)` inside the
positive `fallback_norm_stats` guard is selected. Its exact predecessors are
the retained indexed binary-adapter pair and the retained
singleton/consecutive-Reshape tuple; its immediate successor is the
unconditional fallback topological sort.

The characterization freezes a complete two-key schema and one stale-Reshape
shape repair while preserving the current explicit shape signature. A strict
expected failure requires the unconsumed target
`_fallback_norm_static_shape_stats` with
`include_mutation_count=True`. Do not narrow the existing norm guard or add a
new guard, index, pass, cleanup, consumer, or sort. The preceding owners do not
expose a live index to reuse at this boundary.

Characterization validation completed sequentially under `uv`: the dedicated
contract is `2 passed, 1 xfailed in 0.56s`, and targeted Ruff, bytecode
compilation, and whitespace checks pass. The initial test-only AST lookup was
corrected to search the nested fallback block. The sole expected failure is
the unimplemented result assignment. Commit and push before changing
production code; do not create or update a pull request.

## Fallback norm reconciliation implementation checkpoint

The guarded reconciliation now retains its complete dictionary as
`_fallback_norm_static_shape_stats`. The result remains unconsumed. The
positive norm guard, predecessors, unconditional reconciliation, topological
sort, arguments, graph behavior, dependency set, public API, and TensorFlow
boundary are unchanged.

Implementation validation completed sequentially under `uv`:

- dedicated contract: `3 passed in 0.58s`;
- focused fallback/reconciliation, terminal/core/architecture,
  pass-efficiency, and TensorFlow-import-blocked gate:
  `461 passed in 28.73s`;
- all `fb-refactor8` changed test files: `38 passed in 1.13s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

The first focused run found one stale singleton-boundary AST expectation and
was `460 passed, 1 failed`; after requiring the assigned complete result, the
same gate passed. No model conversion was required.

At resume, audit the remaining raw result-returning topology and layout refresh
boundaries as a family before selecting another assignment. In particular,
`_topologically_sort_operators()` returns a dictionary at several primary and
fallback sites, while `infer_model_ir_logical_layouts()` has a different
contract. Do not mechanically retain or combine them until schemas, mutation
semantics, caller conditions, and current consumers are inventoried. Continue
with commits and pushes only; do not create or update a pull request.

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

## Safe-transpose reduction result implementation checkpoint

The no-layout fallback now retains the existing owner result as
`_no_layout_safe_transpose_reduction_stats`. It remains unconsumed. No branch
condition, pass, snapshot, rollback, prune/reconciliation, argument, ordering,
dependency, public behavior, or TensorFlow boundary changed.

Implementation validation completed sequentially under `uv`:

- dedicated contract: `3 passed in 0.56s`;
- focused safe-reduction, related layout owners, terminal orchestration, core,
  architecture, pass-efficiency, and TensorFlow-import-blocked gate:
  `461 passed in 29.10s`;
- all `fb-refactor8` changed test files: `23 passed in 0.87s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No model conversion was required for this observation-only assignment. After
this checkpoint is committed and pushed, inventory the remaining raw
fallback `_reconcile_static_tensor_shapes(fallback_ir)` inside the positive
`fallback_norm_stats` branch. Its relationship to the preceding binary
adapter and singleton/consecutive-Reshape results, complete mutation schema,
and unconditional topological sort must be fixed before implementation.
Continue with commits and pushes only; do not create or update a pull request.

## Topology/layout refresh characterization

The family audit found twenty-one raw topological-sort calls and six adjacent
`_topologically_sort_operators(...)` / `infer_model_ir_logical_layouts(...)`
pairs. The selected six are the fallback post-dynamic-rank-one boundary,
fallback broadcast-positive boundary, absolute-final primary boundary, and the
positive ConvInteger, InstanceNorm, and broadcast repair boundaries.

Topological sort returns two integer counters. Layout inference returns a full
layout map in addition to its required tensor/metadata mutations, so retaining
that map would unnecessarily increase live memory. The strict expected-failure
contract instead requires `run_topology_layout_refresh()` to preserve both
operations and return only the original sort dictionary. The six lowerer calls
retain small, unconsumed phase-specific results. Fifteen sort-only sites remain
unchanged.

Before implementation, preserve the exact six model arguments, predecessor
targets, conditions, mutation order, and layout effects. Do not return or
retain the full layout map, add a graph scan, alter sort cycle handling, or
touch sort-only boundaries.

Characterization validation completed sequentially under `uv`: the dedicated
family contract is `2 passed, 1 xfailed in 0.56s`, and targeted Ruff, bytecode
compilation, and whitespace checks pass. The sole expected failure is the
absent owner and assignments. Commit and push this characterization before
changing production code; do not create or update a pull request.

## Topology/layout refresh implementation checkpoint

`run_topology_layout_refresh()` now owns the six selected adjacent pairs. It
returns only `reordered_operators` and `cycle_detected`; the full layout map is
released after its side effects. The layout refresh remains unconditional even
when sort reports a cycle. Six phase-specific results are retained without
consumers, while all fifteen sort-only sites remain unchanged.

Implementation validation completed sequentially under `uv`:

- dedicated owner/boundary contract: `4 passed in 0.54s`;
- focused fallback/terminal, affected op families, core, architecture,
  pass-efficiency, and TensorFlow-import-blocked gate:
  `493 passed in 28.99s`;
- all `fb-refactor8` changed test files: `123 passed in 2.91s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

The first focused run found six stale raw-pair AST expectations and was
`487 passed, 6 failed`; after updating those exact structural contracts, the
same gate passed. No model conversion was required for the differentially
equivalent owner.

After this checkpoint is committed and pushed, audit the fifteen sort-only
boundaries by semantic phase. Do not retain all results mechanically: select
only repeated sequences or mutation evidence that can drive a proven safety or
efficiency improvement without extending large result lifetimes. Continue
with commits and pushes only; do not create or update a pull request.
