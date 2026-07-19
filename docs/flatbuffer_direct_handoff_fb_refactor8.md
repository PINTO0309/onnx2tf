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

## Static-shape/topology reconciliation characterization

The fifteen remaining sort-only boundaries were classified before any
production edit. Eight are exact adjacent
`_reconcile_static_tensor_shapes(..., include_mutation_count=True)` then
`_topologically_sort_operators(...)` pairs over the same `ModelIR`. The eight
sites are fallback norm, fallback high-rank BatchMatMul, and the final
high-rank BatchMatMul, Pad, Conv-input, mixed-Concat, Concat-axis, and
binary-layout repair guards. The other seven sites have materially different
predecessors or scope and were excluded from this unit.

`tests/test_flatbuffer_direct_static_shape_topology_reconciliation.py` fixed
the exact eight targets and model arguments and exercised both observable
shape convergence and operator reordering. The existing iterative reconciler
reports three updates for that fixture; do not simplify that result to a count
of distinct output tensors. Characterization passed as `2 passed in 0.51s`
and was committed and pushed as `f8431200` before production changes.

## Static-shape/topology reconciliation implementation checkpoint

`run_static_shape_topology_reconciliation()` now lives with the reconciler in
`passes/static_shape_reconciliation.py`. It runs the unchanged reconciler and
sorter in their original order and returns one normalized four-counter result:

- `reconciled_static_tensor_shapes`;
- `reconciled_static_shape_mutations`;
- `reordered_operators`;
- `cycle_detected`.

The same eight lowerer targets retain the combined result. Guard predicates,
repair calls, fallback recursion, following owners, graph mutations, public
behavior, dependencies, and TensorFlow isolation are unchanged. A dedicated
cycle fixture proves that a detected cycle is reported and the original
operator order remains intact.

Validation completed sequentially under core-only `uv`:

- dedicated owner/boundary/cycle contract: `3 passed`;
- affected fallback, terminal, singleton/Reshape, topology/layout, and shape
  resolution contracts: `114 passed in 3.17s`;
- lowerer architecture contracts: `258 passed in 18.50s`;
- direct-builder topology/reconciliation selection:
  `17 passed, 724 deselected in 0.68s`.

The unfiltered direct-builder attempt also selected two `tf_converter` matrix
tests. Their failures were the expected missing optional TensorFlow dependency
in the core-only environment, not a `flatbuffer_direct` regression. No
real-model conversion was run for this differentially equivalent ownership
change.

Seven raw sort-only boundaries remain:

1. fallback after placeholder-MatMul restoration;
2. fallback after the late Conv/Concat/binary repair family;
3. fallback after indexed binary-layout convergence;
4. the primary post-lowering baseline sort;
5. the guarded no-layout safe-reduction sequence;
6. final placeholder-MatMul restoration;
7. the terminal sort before layout validation.

On resume, characterize the two unconditional validation-boundary sorts and
the three fallback-wide sorts separately. Do not fold them into the new owner
merely because they call the same sorter: their predecessors, guards, and
validation successors differ. Continue with coherent commits and pushes only;
never create, update, or reopen a pull request.

## Terminal topology/layout validation characterization

The fallback and primary terminal paths both contained the exact sequence
`topological sort → layout annotation validation → validation metadata set/pop
→ ModelIR finalization`. The validator is pure; the surrounding lowerer block
owns the metadata update. The characterization contract fixed both complete
boundaries and a graph that exercises operator reordering and one rank/layout
diagnostic. It passed as `2 passed in 0.51s` and was committed and pushed as
`400071a7` before implementation.

## Terminal topology/layout validation implementation checkpoint

`passes/topology_layout_validation.py` now owns the duplicated terminal
invariant boundary. `run_topology_layout_validation()` performs the unchanged
sort and validation, writes or clears `logical_layout_validation_errors`, and
returns only:

- `reordered_operators`;
- `cycle_detected`;
- `layout_validation_errors`.

The full error list remains solely in ModelIR metadata. The fallback and
primary call sites retain compact phase-local results immediately before their
unchanged `_finalize_model_ir(...)` returns. The lowerer no longer imports the
layout validator directly. Cycle handling and stale-error clearing are covered
explicitly.

Validation completed sequentially under core-only `uv`:

- dedicated boundary/effect/cycle contract: `3 passed`;
- affected fallback, terminal, shape, and topology contracts:
  `117 passed in 2.68s`;
- lowerer architecture contracts: `258 passed in 16.94s`;
- direct-builder topology/reconciliation selection:
  `17 passed, 724 deselected in 0.60s`.

No real-model conversion was needed for this differentially equivalent
ownership extraction. Five raw sort-only boundaries remain:

1. fallback after placeholder-MatMul restoration;
2. fallback after the late Conv/Concat/binary repair family;
3. the primary post-lowering baseline sort;
4. the guarded no-layout safe-reduction sequence;
5. final placeholder-MatMul restoration.

On resume, audit the two unconditional fallback-wide sorts first. Determine
which preceding repairs can actually mutate operator order and whether one of
the sorts is redundant; do not remove or guard a traversal without a
differential test proving identical operator order and downstream behavior.
Continue with coherent commits and pushes only; never create, update, or reopen
a pull request.

## Fallback topology checkpoint characterization

The two unconditional fallback-wide sorts were audited separately from the
other remaining sort calls. The first follows the guarded
placeholder-MatMul reconciliation and precedes precision cleanup. The second
follows the guarded late binary-layout reconciliation and precedes fallback
metadata/high-rank BatchMatMul handling. Because repair owners between these
checkpoints can mutate topology, this unit does not remove, merge, or guard
either traversal.

The characterization contract fixes both predecessor guards, successors,
arguments, and the stable `reordered_operators` / `cycle_detected` schema. It
passed as `2 passed in 0.50s` and was committed and pushed as `fe529c63` before
the lowerer changed.

## Fallback topology checkpoint implementation

The existing sort results are now retained as:

- `_fallback_post_placeholder_topology_stats`;
- `_fallback_post_layout_repair_topology_stats`.

Both assignments call the same sorter at the same unconditional locations.
There is no additional graph scan and no result consumer. This phase evidence
can support a future measured redundancy decision without changing current
behavior.

Validation completed sequentially under core-only `uv`:

- dedicated checkpoint and fallback orchestration contracts:
  `20 passed in 0.94s`;
- affected fallback, terminal, shape, and topology contracts:
  `119 passed in 2.77s`;
- lowerer architecture contracts: `258 passed in 16.55s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required for these observation-only assignments.
Three topological-sort results remain discarded in the lowerer:

1. the primary post-lowering baseline sort;
2. the guarded no-layout safe-reduction sort;
3. final placeholder-MatMul restoration.

On resume, characterize these three primary-path checkpoints together only at
the result-contract level, then decide separately whether each result should be
retained. Do not change their guards or remove a sort. Continue with coherent
commits and pushes only; never create, update, or reopen a pull request.

## Primary topology checkpoint characterization

The last three discarded sort results were classified as the primary
post-lowering baseline, guarded no-layout cleanup, and guarded final
placeholder-MatMul checkpoints. They share the sort result schema but not their
execution conditions, so the characterization requires three separate
contexts and does not propose a shared wrapper. It passed as
`2 passed in 0.55s` and was committed and pushed as `806ccc50` before
production changes.

## Primary topology checkpoint implementation

The existing results are now retained under three phase-specific names:

- `_primary_post_lowering_topology_stats`;
- `_no_layout_post_reduction_topology_stats`;
- `_final_placeholder_topology_stats`.

Their sort calls, guards, arguments, predecessors, and successors are
unchanged. Together with the two fallback checkpoint assignments, no direct
lowerer topological-sort result is now discarded. The results are small,
unconsumed observation dictionaries; no scan was added or removed.

Validation completed sequentially under core-only `uv`:

- primary checkpoint and terminal orchestration contracts:
  `65 passed in 1.89s`;
- affected fallback, terminal, safe-reduction, shape, and topology contracts:
  `124 passed in 2.92s`;
- lowerer architecture contracts: `258 passed in 18.34s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

The first architecture run found one stale expression-only AST expectation and
was `257 passed, 1 failed`; after requiring
`_final_placeholder_topology_stats`, it passed completely. No real-model
conversion was needed for observation-only assignments.

On resume, do not consume these counters to skip sorts yet. First inventory all
phase-local observation results introduced on `fb-refactor8` and determine a
single bounded diagnostics sink that does not expose internal types or retain
large maps. Any decision to skip a graph scan must be characterized separately
and must preserve cycle behavior. Continue with coherent commits and pushes
only; never create, update, or reopen a pull request.

## Bounded phase-result store characterization

The result inventory confirmed that the five direct topology checkpoint
results and two topology/layout validation results were all unconsumed locals.
The existing `ConversionSession.diagnostics` list is not a compatible sink:
the private metrics path exports it and requires every event to have
`stage=model_ir_pass`. Therefore the selected contract uses a separate session
store and leaves diagnostics numbering and summaries untouched.

The strict contract requires no more than 128 phases, no more than 32 counters
per phase, integer-only values, copied inputs, and isolated snapshots. It also
fixes the seven old local targets and owners before migration. Characterization
passed as `1 passed, 1 xfailed in 0.56s` and was committed and pushed as
`8a3245e3` before implementation.

## Bounded phase-result store implementation

`ConversionSession.record_phase_result()` now validates and stores compact
integer counter mappings, while `phase_results_snapshot()` returns an isolated
copy. The store is conversion-local and is not copied into ModelIR metadata,
the existing pass-diagnostics sink, public conversion results, or generated
reports.

Seven observations migrated from unconsumed locals to stable phase IDs:

- `topology.fallback.post_placeholder`;
- `topology.fallback.post_layout_repair`;
- `layout_validation.fallback.terminal`;
- `topology.primary.post_lowering`;
- `topology.primary.no_layout_post_reduction`;
- `topology.primary.final_placeholder`;
- `layout_validation.primary.terminal`.

The owner expression remains the second argument of each record call, so it is
evaluated once at the original position. No graph scan, sort, validation,
metadata mutation, guard, or successor changed. Only small integer dictionaries
are retained.

Validation completed sequentially under core-only `uv`:

- phase store, seven migrations, fallback/terminal orchestration, and core
  diagnostics compatibility: `145 passed in 2.97s`;
- lowerer architecture contracts: `258 passed in 18.98s`;
- additional affected shape/topology contracts: `36 passed in 1.04s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was needed for this observation-storage change. On
resume, inventory the remaining unconsumed `fb-refactor8` observation locals
and migrate only one homogeneous result family at a time. The six
topology/layout refresh results are the next suitable family because they all
share the same two-counter schema; do not migrate shape-reconciliation results
in the same checkpoint. Continue with coherent commits and pushes only; never
create, update, or reopen a pull request.
