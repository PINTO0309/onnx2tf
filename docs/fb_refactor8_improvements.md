# `fb-refactor8` improvement summary

## Purpose and compatibility boundary

`fb-refactor8` continues the characterize-first refactoring of the
TensorFlow-free `flatbuffer_direct` pipeline from the merged `2.6.6`
checkpoint. Public CLI/Python behavior, artifact names, pass order, dependency
set, optional TensorFlow boundary, and strictly sequential inference policy
remain fixed.

Development on this branch uses coherent commits and pushes only. No pull
request is to be created, reopened, or updated.

## Terminal Expand/Squeeze reconciliation characterization

The first unit inventories the raw static-shape reconciliation immediately
after `_terminal_expand_squeeze_stats` and before `_advance_post_progress()`.
The current boundary discards the default one-key result.

The new contracts establish that:

- the default result remains
  `{"reconciled_static_tensor_shapes": 0}` on a stable graph;
- the opt-in complete schema additionally reports
  `reconciled_static_shape_mutations`;
- reconciliation must remain unconditional because a stale shape produced by
  an earlier late-layout owner can require repair even when the terminal
  Expand/Squeeze owner reports zero rewrites;
- the fixture's static tensor shape is repaired while its existing explicit
  shape signature remains unchanged, matching the current reconciler
  contract;
- the call remains directly adjacent to the retained Expand/Squeeze result and
  `_advance_post_progress()`;
- there is no reusable live `ModelIRGraphIndex` at this boundary. The
  Expand/Squeeze owner creates a local index only for dynamic pre-operator
  insertion and does not expose it. Index sharing therefore requires a
  separate owner-level characterization and is not mixed into result
  retention.

The selected implementation is observation-only assignment to
`_terminal_expand_squeeze_static_shape_stats` with
`include_mutation_count=True`. It must not add a guard, consumer, graph scan,
pass, or index construction.

Characterization validation completed sequentially under `uv`:

- dedicated contract: `2 passed, 1 xfailed in 0.54s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result
assignment. No production source, public API, dependency, pass order,
TensorFlow boundary, or model artifact changed in this checkpoint.

## Inherited shape-resolution test import repair

The first all-direct-test collection exposed an inherited test-only stale
import: `test_flatbuffer_direct_shape_resolution.py` still imported
`_set_operator_outputs` from the central lowerer after that helper had moved to
`core.model_ir_utils`. `main` contains the same stale import.

The test now imports the canonical owner used throughout production pass
modules. No lowerer compatibility alias is restored, and no production code or
runtime behavior changes in this repair.

## Terminal Expand/Squeeze reconciliation result retention

The unconditional terminal reconciliation now retains its complete result as
`_terminal_expand_squeeze_static_shape_stats`. The call requests
`include_mutation_count=True`, so parameter-only and metadata mutations remain
observable in addition to legacy tensor-shape updates.

The result is intentionally unconsumed. No guard was added because preceding
late-layout owners can require reconciliation even when the immediately
adjacent Expand/Squeeze owner reports zero. Call order, graph mutation,
reconciliation work, progress reporting, layout state, and artifact behavior
are unchanged. No graph index or additional scan is introduced.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.55s`;
- focused Expand/Squeeze, shape reconciliation, surrounding orchestration,
  core, pass-efficiency, architecture, and TensorFlow-import-blocked gate:
  `381 passed in 27.03s`;
- all files changed by this branch's implementation and inherited test repair:
  `17 passed in 0.68s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The first focused run produced `380 passed, 1 failed` because one existing AST
test still required the raw reconciliation expression. Its structural
expectation was updated to require the new observation target and complete
schema; the unchanged gate then passed. No model-corpus conversion was run
because assigning an already-computed dictionary cannot affect ModelIR or an
artifact.

## Core dynamic-Reshape result characterization

The core-cleanup call to `_resolve_dynamic_reshape_shapes(model_ir)` returns a
fixed `resolved_dynamic_reshape_shapes` counter but currently discards it. The
call sits between the retained Conv-activation result and the retained
Squeeze/Reshape identity cleanup result.

The new contract freezes the zero schema and a positive mutation that resolves
`[-1, 2]` to `[2, 2]` across the operator option, constant shape tensor, and
output tensor metadata. The selected implementation is an unconsumed
assignment named `_core_cleanup_dynamic_reshape_stats`. It must not add a
guard, graph index, cleanup, pass, scan, or consumer, and it must preserve the
exact call arguments and phase position.

Characterization validation completed sequentially under `uv`:

- dedicated contract: `2 passed, 1 xfailed in 0.54s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented assignment. No
production source changed in this checkpoint.

## Core dynamic-Reshape result retention

The core-cleanup invocation now retains its unchanged one-key dictionary as
`_core_cleanup_dynamic_reshape_stats`. The target is unconsumed and
observation-only. No call argument, graph index, guard, cleanup, pass order,
mutation, layout state, or artifact behavior changed.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.54s`;
- focused dynamic-Reshape, surrounding cleanup, indexed convergence, core,
  architecture, pass-efficiency, terminal orchestration, and
  TensorFlow-import-blocked gate: `452 passed in 28.87s`;
- all test files changed on `fb-refactor8`: `20 passed in 0.76s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No model conversion was run because retaining an already-computed dictionary
cannot change ModelIR or serialization.

## Safe-transpose reduction result characterization

The no-layout fallback calls `_apply_safe_transpose_reduction_lite(model_ir)`
as a raw expression before the retained affine pre/post cleanup. The owner
returns three counters covering executed passes, removed Transposes, and
rollback-triggering unbound inputs.

The owner already contains its complete safety transaction: it snapshots the
ModelIR, runs the curated pass sequence, prunes and reconciles, rejects unbound
inputs, and restores the snapshot when no safe reduction is achieved. No live
caller index or external cleanup is available to combine with this boundary.

The selected implementation is observation-only assignment to
`_no_layout_safe_transpose_reduction_stats`. The conditional branch, sole
`model_ir` argument, owner transaction, following affine cleanup, and result
schema must remain unchanged.

Characterization validation completed sequentially under `uv`:

- dedicated contract: `2 passed, 1 xfailed in 0.55s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the unimplemented assignment. No production
source changed in this checkpoint.

## Safe-transpose reduction result retention

The no-layout fallback now retains the unchanged three-key dictionary as
`_no_layout_safe_transpose_reduction_stats`. It remains unconsumed and
observation-only. The `elif` condition, owner transaction, pass sequence,
snapshot rollback, prune/reconciliation work, arguments, and following affine
cleanup are unchanged.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.56s`;
- focused safe-reduction, indexed Concat adapter, Conv/Pool result, affine,
  terminal orchestration, core, architecture, pass-efficiency, and
  TensorFlow-import-blocked gate: `461 passed in 29.10s`;
- all test files changed on `fb-refactor8`: `23 passed in 0.87s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No model conversion was run because assigning the already-returned transaction
summary cannot affect ModelIR or artifacts.
