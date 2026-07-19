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

## Topology/layout refresh result migration

The six already-characterized `run_topology_layout_refresh()` results now use
the bounded session store rather than unconsumed lowerer locals. Their stable
phase IDs are:

- `topology_layout.fallback.post_dynamic_rank1`;
- `topology_layout.fallback.broadcast`;
- `topology_layout.primary.absolute_final`;
- `topology_layout.primary.final_convinteger`;
- `topology_layout.primary.final_instancenorm`;
- `topology_layout.primary.final_broadcast`.

The migration reuses the owner and differential contracts from `5def1684` and
`b7fe39e0`, plus the bounded-store contract from `8a3245e3` and `0772d42c`.
No new characterization commit was needed because both sides of the migration
were already fixed. Owner calls, guard conditions, predecessor results,
logical-layout side effects, cycle handling, and full-layout-map release are
unchanged.

Validation completed sequentially under core-only `uv`:

- phase store, topology/layout owner, fallback/terminal orchestration, and core
  diagnostics contracts: `149 passed in 2.97s`;
- lowerer architecture contracts: `258 passed in 16.74s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required for this result-destination-only change.
On resume, migrate the three remaining single-call observation locals
(`_core_cleanup_dynamic_reshape_stats`,
`_no_layout_safe_transpose_reduction_stats`, and
`_terminal_expand_squeeze_static_shape_stats`) only after confirming that each
owner returns integer-only bounded counters. Keep fallback/final combined
shape-topology results as a separate later family. Continue with coherent
commits and pushes only; never create, update, or reopen a pull request.

## Single-call phase-result migration

The three proposed owners were confirmed to return bounded integer-only
schemas: Dynamic Reshape returns one counter, safe Transpose reduction returns
three, and terminal Expand/Squeeze reconciliation returns two. They now record:

- `shape_resolution.core.dynamic_reshape`;
- `layout.no_layout.safe_transpose_reduction`;
- `shape_reconciliation.terminal.expand_squeeze`.

All three calls remain in place. Their predecessors, successors, arguments,
guards, graph mutations, and TensorFlow-free boundary are unchanged. The
bounded-store contract now covers sixteen phase IDs in the lowerer.

Validation completed sequentially under core-only `uv`:

- dedicated result/schema/boundary contracts: `14 passed in 0.88s`;
- phase store, topology/layout, fallback/terminal orchestration, and core
  contracts: `149 passed in 3.49s`;
- lowerer architecture contracts: `258 passed in 16.73s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required for result-destination-only changes. On
resume, audit the eight combined static-shape/topology reconciliation result
targets as one family. Before removing their zero defaults, explicitly decide
whether the bounded store represents only invoked phases or also records
guard-skipped zero results. Do not silently conflate "not invoked" with "ran
and changed nothing". Continue with coherent commits and pushes only; never
create, update, or reopen a pull request.

## Static-shape/topology phase-result migration

The eight combined reconciliation results now use the bounded session store.
The selected contract is invoked-phase-only: a skipped guard creates no entry,
whereas an invoked owner records its complete four-counter result even when
all counters are zero. This avoids conflating "not invoked" with "invoked and
stable". The eight old unconsumed zero-default locals were removed.

Stable phase IDs cover fallback norm, fallback high-rank BatchMatMul, and the
six primary final repair boundaries for high-rank BatchMatMul, Pad, Conv input,
mixed Concat, Concat axis, and binary layout. All guards, owner calls,
arguments, predecessors, successors, and ModelIR effects remain unchanged.
The bounded store now covers 24 phase IDs and remains isolated from public
results, reports, diagnostics, metadata, and artifacts.

Validation completed sequentially under core-only `uv`:

- dedicated and directly affected contracts: `101 passed in 2.66s`;
- broader phase-result and related owner/orchestration contracts:
  `124 passed in 3.21s`;
- lowerer architecture contracts: `258 passed in 16.64s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was needed for this result-destination-only change.
After committing and pushing this checkpoint, inventory remaining unconsumed
lowerer observations before selecting another homogeneous family. Do not use
stored counters to skip graph scans without a separate differential
characterization. Continue with coherent commits and pushes only; never
create, update, or reopen a pull request.

## Fallback static-shape phase-result characterization

The next homogeneous family has been selected: seven fallback-only guarded
static-shape reconciliation observations for broadcast, SE/FC/Gather,
placeholder-MatMul, Conv input, mixed Concat, Concat axis, and binary layout.
Each uses the same complete two-counter owner call and stores an unconsumed
all-zero default when its guard is skipped.

The new contract fixes all seven target names, their source order, zero schema,
owner arguments, `include_mutation_count=True`, and absence of consumers. Its
strict expected failure requires seven stable
`shape_reconciliation.fallback.*` records and removal of the old targets.
Invoked-phase-only semantics remain selected: skipped guards create no phase
entry.

Validation completed sequentially under core-only `uv`: the dedicated test is
`1 passed, 1 xfailed in 0.15s`, and targeted Ruff, bytecode compilation, and
whitespace checks pass. No production code changed.

Commit and push this characterization before implementation. Then replace only
the seven guarded assignments with `session.record_phase_result(...)`, remove
only their unconsumed zero defaults, update the already-existing fallback
orchestration contracts, and repeat the focused and architecture gates. Do not
change any guard, owner call, repair order, graph scan, or successor. Never
create, update, or reopen a pull request.

## Fallback static-shape phase-result implementation

All seven characterized results now use stable
`shape_reconciliation.fallback.*` phase IDs. The old zero-default dictionaries
and lowerer-local targets are gone. Guard-skipped phases remain absent, while
invoked owners record the complete two-counter result. Arguments, guards,
owner evaluation count, repair order, broadcast topology/layout refresh, and
all successors remain unchanged.

Validation completed sequentially under core-only `uv`:

- family and safety-fallback contracts: `20 passed in 0.93s`;
- SE/FC/Gather, topology/layout, and phase-store contracts:
  `18 passed in 0.82s`;
- broader affected contracts: `138 passed in 3.45s`;
- lowerer architecture contracts: `258 passed in 17.69s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

The bounded store now covers 31 phase IDs. No real-model conversion was needed
for this result-destination-only change. After committing and pushing, audit
the remaining primary-path static-shape-only result locals. Split them into
bounded semantic families rather than migrating all model-specific final
repairs at once, and preserve invoked-only semantics. Never create, update, or
reopen a pull request.

## Primary final layout-refresh reconciliation characterization

The primary inventory found 20 remaining unconsumed static-shape-only results.
The first selected family is deliberately limited to the three final
ConvInteger, InstanceNorm, and broadcast reconciliations because each is
immediately followed by its already-characterized topology/layout refresh.

The new contract fixes the three zero defaults, target order, mutation guards,
`model_ir` argument, `include_mutation_count=True`, successor refresh order,
and absence of consumers. Its strict expected failure requires stable
`shape_reconciliation.primary.final_*` phase records without changing the
existing refresh records.

Validation completed sequentially under core-only `uv`: the dedicated test is
`1 passed, 1 xfailed in 0.16s`, and targeted Ruff, bytecode compilation, and
whitespace checks pass. No production source changed.

Commit and push this characterization, then migrate only these three guarded
assignments and remove only their zero defaults. Update the topology/layout
predecessor contract and bounded-store inventory, run focused and architecture
gates, document, commit, and push. Never create, update, or reopen a pull
request.

## Primary final layout-refresh reconciliation implementation

The final ConvInteger, InstanceNorm, and broadcast reconciliations now use
stable `shape_reconciliation.primary.final_*` records immediately before their
unchanged topology/layout refresh records. Their zero defaults and local
targets were removed; invoked-only semantics and all mutation guards are
preserved. The bounded store now covers 34 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct family, terminal, refresh, and store contracts:
  `71 passed in 2.50s`;
- broader affected contracts: `140 passed in 3.57s`;
- lowerer architecture contracts: `258 passed in 17.03s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required. After committing and pushing, continue
the primary static-shape inventory with another small semantic family. The
remaining SiNet-specific final repair chain should not be migrated together
with generic placeholder, PReLU, or consecutive-Reshape boundaries. Preserve
all guards and do not use observations to skip work without separate
differential characterization. Never create, update, or reopen a pull request.

## Primary final cleanup reconciliation characterization

The next selected primary family contains only final PReLU and
consecutive-Reshape reconciliation. Both have the same complete two-counter
owner and unconsumed zero defaults. Their guards differ intentionally: PReLU
also observes cleanup-only tensor pruning, while consecutive Reshape sums its
three declared mutation counters.

The new contract fixes both targets, source order, zero schema, owner call,
`include_mutation_count=True`, and absence of consumers. Its strict expected
failure requires stable `shape_reconciliation.primary.final_*` records without
changing either guard.

Validation completed sequentially under core-only `uv`: the dedicated test is
`1 passed, 1 xfailed in 0.15s`, and targeted Ruff, bytecode compilation, and
whitespace checks pass. No production source changed.

Commit and push this characterization before replacing only these two guarded
assignments and removing their defaults. Update the terminal and architecture
contracts that currently require the old locals, then repeat the bounded-store,
focused, and architecture gates. Never create, update, or reopen a pull
request.

## Primary final cleanup reconciliation implementation

The final PReLU and consecutive-Reshape results now use invoked-only
`shape_reconciliation.primary.final_*` records. Their original guards remain
complete, including PReLU cleanup-only tensor pruning and all three
consecutive-Reshape mutation counters. The zero defaults and local targets are
removed, and the bounded store now covers 36 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct pair, terminal, and store contracts: `67 passed in 1.96s`;
- broader affected contracts: `142 passed in 3.62s`;
- lowerer architecture contracts: `258 passed in 19.09s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required. After committing and pushing, select a
separate generic final family for mixed-singleton Concat,
placeholder/binary repair, and SE/FC/Gather before considering the six
SiNet-specific final repairs. Preserve every mutation and pruning guard. Never
create, update, or reopen a pull request.

## Primary generic final reconciliation characterization

The next selected family is mixed-singleton Concat,
placeholder/binary-adapter, and SE/FC/Gather reconciliation. All three selected
results are unconsumed complete two-counter dictionaries behind existing
mutation/pruning guards.

The placeholder path also retains
`_final_placeholder_matmul_static_shape_stats`, which is consumed to construct
`final_placeholder_reconcile_stats` and decide whether the nested binary
reconciliation runs. The new contract explicitly proves that this consumed
result remains loaded and must not be migrated with the selected three.

Characterization fixes target order, zero schema, owner arguments,
`include_mutation_count=True`, absence of selected-result consumers, and the
consumed placeholder result. Its strict expected failure requires three stable
`shape_reconciliation.primary.final_*` records. The dedicated test is
`1 passed, 1 xfailed in 0.17s`; Ruff, bytecode compilation, and whitespace
checks pass. No production source changed.

Commit and push before implementation. Replace only the three selected guarded
assignments and remove only their defaults. Preserve the consumed
placeholder-MatMul assignment and all nested guard logic. Then update terminal,
SE/FC/Gather, architecture, and bounded-store contracts, validate, document,
commit, and push. Never create, update, or reopen a pull request.

## Primary generic final reconciliation implementation

The mixed-singleton Concat, nested placeholder/binary, and SE/FC/Gather results
now use stable invoked-only `shape_reconciliation.primary.final_*` records.
Their zero defaults and local targets were removed. The separately consumed
`_final_placeholder_matmul_static_shape_stats` assignment remains unchanged
and still controls the nested binary-reconciliation guard.

Validation completed sequentially under core-only `uv`:

- direct family, terminal, SE/FC/Gather, and store contracts:
  `79 passed in 2.57s`;
- broader affected contracts: `144 passed in 3.83s`;
- lowerer architecture contracts: `258 passed in 19.17s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

The bounded store now covers 39 phase IDs. No real-model conversion was
required. After committing and pushing, treat the six final SiNet-specific
reconciliations as their own semantic family. Characterize the complete chain
and successor order before migration; do not mix it with earlier late/static
shape observations. Never create, update, or reopen a pull request.

## Primary final SiNet reconciliation characterization

The six final SiNet repair/reconciliation pairs have been characterized as one
ordered family: late residual, pre-add fanout, dual resize, shared post,
deep-skip tail, and concat-resize. Each selected result has the same complete
two-counter schema, a dedicated positive mutation guard, a `model_ir` owner
argument, and no consumer.

The new contract fixes repair order, result order, zero defaults,
`include_mutation_count=True`, and absence of loads. Its strict expected
failure requires six `shape_reconciliation.primary.final_sinet_*` records and
does not include earlier late/static shape observations.

Validation completed sequentially under core-only `uv`: the dedicated test is
`1 passed, 1 xfailed in 0.16s`, and targeted Ruff, bytecode compilation, and
whitespace checks pass. No production source changed.

Commit and push before replacing only these six guarded assignments and
removing their defaults. Update the parameterized SiNet terminal and
architecture contracts plus the bounded-store inventory, then validate,
document, commit, and push. Never create, update, or reopen a pull request.

## Primary final SiNet reconciliation implementation

All six final SiNet reconciliations now use stable invoked-only
`shape_reconciliation.primary.final_sinet_*` records. Their dedicated guards,
repair order, owner arguments, and successors are unchanged. The old zero
defaults and local targets are removed, and the bounded store now covers 45
phase IDs.

Validation completed sequentially under core-only `uv`:

- direct ordered-chain, terminal, and store contracts:
  `67 passed in 2.32s`;
- broader affected contracts: `146 passed in 3.84s`;
- lowerer architecture contracts: `258 passed in 18.28s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required. After committing and pushing, re-run
the unconsumed static-shape inventory. The remaining earlier late boundaries
have different unconditional, split-fallback, shared-mutation, and binary
recovery semantics; classify them before selecting another family. Never
create, update, or reopen a pull request.

## Unconditional very-late reconciliation characterization

The remaining six unconsumed static-shape observations were classified by
execution semantics. The selected family contains only the two unconditional
calls after very-late broadcast repair and after the final very-late
dynamic-rank-one rewrite.

The new contract fixes their exact predecessors, successors, order,
`model_ir` argument, `include_mutation_count=True`, absence of consumers, and
unconditional execution. Its strict expected failure requires stable
`shape_reconciliation.primary.very_late_*` records.

Validation completed sequentially under core-only `uv`: the dedicated test is
`1 passed, 1 xfailed in 0.14s`, and targeted Ruff, bytecode compilation, and
whitespace checks pass. No production source changed.

Commit and push before replacing only the two assignment targets with phase
records. Do not add guards or combine them with shared, binary-recovery, or
split-fallback results. Update their existing surrounding contracts and the
bounded-store inventory, validate, document, commit, and push. Never create,
update, or reopen a pull request.

## Unconditional very-late reconciliation implementation

The two unconditional very-late results now use stable
`shape_reconciliation.primary.very_late_*` records. Their calls remain
unconditional and between the same predecessor/successor assignments. Only the
unconsumed local destinations changed; no defaults existed at these boundaries.
The bounded store now covers 47 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct boundary, terminal, very-late, and store contracts:
  `90 passed in 2.77s`;
- broader affected contracts: `171 passed in 4.32s`;
- lowerer architecture contracts: `258 passed in 18.11s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required. After committing and pushing, the four
remaining unconsumed static-shape observations are guarded shared-late,
late-binary repair, late-binary recovery, and post-split fallback boundaries.
Classify the two binary-related boundaries together first; keep shared-late and
post-split separate. Never create, update, or reopen a pull request.

## Late binary reconciliation characterization

The two binary-related remaining results have been characterized together.
The repair result is guarded by signature/adapter mutations or cleanup-only
tensor pruning. The layout-recovery result is nested under the existing late
layout enablement condition and a positive recovery-summary guard. Neither has
a default or consumer.

The contract fixes both guard layers, order, `model_ir` argument,
`include_mutation_count=True`, pruning evidence, and absence of loads. Its
strict expected failure requires
`shape_reconciliation.primary.late_binary_repair` and
`shape_reconciliation.primary.late_binary_layout_recovery` records.

Validation completed sequentially under core-only `uv`: the dedicated test is
`1 passed, 1 xfailed in 0.18s`, and targeted Ruff, bytecode compilation, and
whitespace checks pass. No production source changed.

Commit and push before replacing only these two guarded assignments with phase
records. Preserve both outer and inner guards, then update binary-recovery,
terminal, architecture, and bounded-store contracts, validate, document,
commit, and push. Never create, update, or reopen a pull request.

## Late binary reconciliation implementation

The two guarded reconciliation results now use stable
`shape_reconciliation.primary.late_binary_repair` and
`shape_reconciliation.primary.late_binary_layout_recovery` records. The repair
record retains its mutation-or-pruning guard. The recovery record retains both
the outer late-layout enablement guard and the inner positive-summary guard.
Only the unconsumed result destinations changed; neither boundary had a zero
default. The bounded store now covers 49 phase IDs.

Validation completed sequentially under core-only `uv`:

- focused late-binary, terminal, and store contracts:
  `72 passed in 2.40s`;
- broader affected contracts: `187 passed in 4.82s`;
- lowerer architecture contracts: `258 passed in 18.40s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required. After committing and pushing, only the
shared-late and post-split fallback static-shape observations remain
unconsumed. Characterize these separately before changing either destination;
their guards and execution semantics differ. Never create, update, or reopen a
pull request.

## Shared-late reconciliation characterization

The guarded shared-late result has been characterized independently from the
post-split fallback boundary. Its predicate retains all nine ordered mutation
results plus the tensor-count decrease for prune-only cleanup. The result has
no zero default and no consumer.

The contract fixes the evidence owners and order, guard, `model_ir` argument,
`include_mutation_count=True`, preceding tensor-count snapshot, following
late-binary tensor-count snapshot, and absence of loads. Its strict expected
failure requires `shape_reconciliation.primary.shared_late`.

Validation completed sequentially under core-only `uv`: the dedicated result
is `1 passed, 1 xfailed in 0.15s`; the characterization plus runtime, terminal,
and architecture boundary contracts are `4 passed, 1 xfailed in 0.72s`;
targeted Ruff, bytecode compilation, and whitespace checks pass. No production
source changed.

Commit and push this characterization before replacing only the guarded result
destination. Preserve the exact evidence predicate and both tensor-count
boundaries. Then update terminal, architecture, and bounded-store contracts,
validate, document, commit, and push. Do not alter the post-split fallback
boundary in the same implementation checkpoint, and never create, update, or
reopen a pull request.

## Shared-late reconciliation implementation

The shared-late result now uses the stable
`shape_reconciliation.primary.shared_late` record. Its exact nine-result plus
prune-delta predicate and both surrounding tensor-count snapshots remain
unchanged. Only the unconsumed result destination changed; no default existed.
The bounded store now covers 50 phase IDs.

Validation completed sequentially under core-only `uv`:

- focused shared-late, late-binary, terminal, runtime, and store contracts:
  `75 passed in 2.31s`;
- broader affected contracts: `190 passed in 5.03s`;
- lowerer architecture contracts: `258 passed in 16.57s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required. The post-split fallback result is the
only remaining unconsumed static-shape observation. Characterize its
unconditional zero default, conditional invocation, fallback owner argument,
and successor before changing its destination. Never create, update, or reopen
a pull request.

## Post-split fallback reconciliation characterization

The last unconsumed static-shape result has been characterized. It follows the
very-late unsupported Split-to-Slice fallback, starts with an unconsumed
all-zero two-counter default, and invokes the complete static-shape reconciler
only when at least one Split was replaced. The result has no consumer.

The contract fixes the fallback owner and layout-state argument, zero schema,
positive guard, `model_ir` and `include_mutation_count=True`, absence of loads,
and following unbound-input safety check. Its strict expected failure requires
`shape_reconciliation.primary.post_split_fallback` with invoked-only semantics.

Validation completed sequentially under core-only `uv`: characterization,
existing orchestration, and Split fallback unit contracts are
`6 passed, 1 xfailed in 0.64s`; targeted Ruff, bytecode compilation, and
whitespace checks pass. No production source changed.

Commit and push this characterization before removing only the unconsumed zero
default and replacing only the guarded result destination. Preserve the Split
owner, positive guard, and safety-fallback successor. Then update the existing
orchestration and bounded-store inventories, validate, document, commit, and
push. Never create, update, or reopen a pull request.

## Post-split fallback reconciliation implementation

The final unconsumed static-shape result now uses the stable
`shape_reconciliation.primary.post_split_fallback` record. Its unconsumed zero
default was removed, so a skipped guard leaves the phase absent while an
invoked stable reconciliation is still recorded. The bounded store now covers
51 phase IDs.

Validation completed sequentially under core-only `uv`:

- focused post-split, very-late, Split fallback, terminal, and store contracts:
  `96 passed in 2.39s`;
- broader affected contracts: `196 passed in 5.25s`;
- lowerer architecture contracts: `258 passed in 16.47s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No real-model conversion was required. An AST audit reports no unconsumed
`*static_shape_stats` assignments in `lower_onnx_to_ir`. The sole remaining
matching local, `_final_placeholder_matmul_static_shape_stats`, is consumed by
its established downstream guard and must remain local unless that control-
flow contract is redesigned separately. After committing and pushing this
checkpoint, inventory the next unconsumed result family rather than altering
that consumed value. Never create, update, or reopen a pull request.

## Stale Squeeze/Reshape boundary assertion correction

The broader core-cleanup characterization gate found one stale test assertion:
the core Squeeze/Reshape result contract still expected a raw dynamic-Reshape
call as its predecessor. Since `dd58fd84`, that predecessor has been the exact
`shape_resolution.core.dynamic_reshape` session record. The assertion was
updated to the current stable boundary; production source is unchanged.

The same related gate now reports `72 passed, 1 xfailed in 2.18s`. The only
xfail is the deliberate characterization gate for the not-yet-implemented
core-cleanup result migration. Ruff, bytecode compilation, and whitespace
checks pass.

Commit and push this stale-contract correction independently. Then commit the
core-cleanup characterization separately before changing production source.
Never create, update, or reopen a pull request.

## Core cleanup phase-result characterization

The next family contains only the nine unconditional mapping results in the
`core cleanup passes` progress stage. It excludes composite cluster results,
guarded layout-pass results, and later cleanup stages. The owners cover pseudo-
LeakyReLU, YOLO decode, consecutive Mul, terminal Dequantize/QDQ, Conv affine,
Conv activation, Squeeze/Reshape identity, and indexed prune/reconcile cleanup.

The contract fixes exact targets, owner arguments and keywords, source order,
progress boundaries, absence of loads, and the already-recorded dynamic-
Reshape phase between owner seven and owner eight. Its strict expected failure
requires nine `cleanup.core.*` phase records.

Validation completed sequentially under core-only `uv`: the dedicated test is
`1 passed, 1 xfailed in 0.15s`; the broader relevant structural gate is
`72 passed, 1 xfailed in 2.18s`; Ruff, bytecode compilation, and whitespace
checks pass. No production source changed.

Commit and push this characterization before replacing only the nine result
destinations. Preserve owner calls, arguments, source order, unconditional
execution, progress boundaries, and the dynamic-Reshape record position. Then
update all existing result-retention contracts and the bounded-store inventory,
validate, document, commit, and push. Never create, update, or reopen a pull
request.

## Core cleanup phase-result implementation

The nine unconditional core-cleanup mapping results now use stable
`cleanup.core.*` records. Their owner expressions, order, progress stage, and
the intervening dynamic-Reshape record are unchanged. No defaults existed in
this family. The bounded store now covers 60 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct affected structural contracts: `76 passed in 2.65s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader affected contracts: `257 passed in 6.27s`;
- lowerer architecture contracts: `258 passed in 16.75s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite exercised the stored schemas. After committing
and pushing, inventory the next mapping-only family by execution guard and
return schema. Keep composite cluster results separate, and account for the
128-phase session limit before selecting further migrations. Never create,
update, or reopen a pull request.

## Terminal cleanup phase-result characterization

The next mapping-only family is restricted to the four unconditional results
at the beginning of `terminal cleanup passes`: Dequantize sanitization, exact-
grid Q/DQ cleanup, Conv affine folding, and Conv activation folding. These
reuse the owner schemas validated by the preceding core-cleanup migration.

The contract fixes exact targets, owner arguments and keywords, source order,
progress predecessor, pre-ArgMax successor, and absence of loads. Its strict
expected failure requires four `cleanup.terminal.*` records; later terminal
layout results are explicitly excluded.

Validation completed sequentially under core-only `uv`: the dedicated and
related boundary gate is `5 passed, 1 xfailed in 0.68s`; Ruff, bytecode
compilation, and whitespace checks pass. No production source changed.

Commit and push this characterization before replacing only the four result
destinations. Preserve the owner expressions, unconditional execution,
progress boundary, and pre-ArgMax successor. Then update existing occurrence
contracts and the bounded-store inventory, validate, document, commit, and
push. Never create, update, or reopen a pull request.

## Terminal cleanup phase-result implementation

The four unconditional terminal-cleanup mapping results now use stable
`cleanup.terminal.*` records. Their owner expressions, source order, progress
predecessor, and pre-ArgMax successor remain unchanged. No defaults existed in
this family. The bounded store now covers 64 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct affected structural contracts: `68 passed in 4.13s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader affected contracts: `259 passed in 6.15s`;
- lowerer architecture contracts: `258 passed in 16.54s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite executes the terminal-cleanup stage. After
committing and pushing, select the next family by common execution guard and
mapping schema. Keep the 64 current records and 128-phase hard limit visible
when sizing further migrations. Never create, update, or reopen a pull request.

## Layout pass-set 2 cleanup characterization

The selected pair contains only Squeeze/Reshape identity cleanup and indexed
prune/reconcile cleanup at the end of the second layout recovery pass-set. Both
are bounded integer mappings behind the same
`optimize_layout_transpose_chains` guard, with no defaults or consumers.

The contract fixes the guard, owner arguments and keywords, source order,
preceding two-iteration normalization convergence loop, progress successor,
and absence of loads. Its strict expected failure requires two
`cleanup.layout_pass_set_2.*` records with invoked-only semantics.

Validation completed sequentially under core-only `uv`: the characterization
and existing owner contracts are `6 passed, 1 xfailed in 0.77s`; Ruff,
bytecode compilation, and whitespace checks pass. No production source changed.

Commit and push this characterization before replacing only the two result
destinations. Preserve the outer guard, convergence predecessor, owner calls,
and progress successor. Then update the two existing multi-occurrence owner
contracts and bounded-store inventory, validate, document, commit, and push.
Never create, update, or reopen a pull request.

## Layout pass-set 2 cleanup implementation

The guarded pair now uses stable
`cleanup.layout_pass_set_2.squeeze_reshape_identity` and
`cleanup.layout_pass_set_2.prune_reconcile` records. The outer layout guard,
convergence predecessor, owner expressions, and progress successor remain
unchanged. No defaults existed. The bounded store now covers 66 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct affected contracts: `9 passed in 0.85s`;
- synthetic core runtime contracts: `55 passed in 0.99s`;
- broader affected contracts: `261 passed in 6.27s`;
- lowerer architecture contracts: `258 passed in 16.57s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite exercises the guarded path. After committing and
pushing, select the next family without mixing compound cluster return values
with mapping-only results, and retain the 128-phase limit in the sizing audit.
Never create, update, or reopen a pull request.

## Layout pass-set 1 cleanup characterization

The selected pair contains only the adjacent InstanceNorm pre/post and
Squeeze/Reshape identity mapping results near the end of the first layout
pass-set. Both are behind `optimize_layout_transpose_chains`, have no defaults
or consumers, and are bounded by composite attention-cluster results that are
not part of this family.

The contract fixes the guard, owner arguments and keywords, adjacency,
composite predecessor and successor targets, and absence of loads. Its strict
expected failure requires two `cleanup.layout_pass_set_1.*` records.

Validation completed sequentially under core-only `uv`: the characterization
and existing owner/architecture boundaries are
`6 passed, 1 xfailed in 0.80s`; Ruff, bytecode compilation, and whitespace
checks pass. No production source changed.

Commit and push this characterization before replacing only the two mapping
destinations. Preserve the outer guard and both composite boundaries; do not
attempt to store the composite return values in the bounded integer mapping
store. Then update existing contracts and the phase inventory, validate,
document, commit, and push. Never create, update, or reopen a pull request.

## Layout pass-set 1 cleanup implementation

The adjacent mapping pair now uses stable
`cleanup.layout_pass_set_1.instancenorm_prepost` and
`cleanup.layout_pass_set_1.squeeze_reshape_identity` records. The common guard,
owner expressions, adjacency, and composite attention boundaries remain
unchanged. No defaults existed. The bounded store now covers 68 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct affected contracts: `9 passed in 2.63s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- broader affected contracts: `265 passed in 6.34s`;
- lowerer architecture contracts: `258 passed in 16.53s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite executes the guarded path. After committing and
pushing, keep composite return values outside the integer mapping store and
select the next mapping family with the 128-phase cap in view. Never create,
update, or reopen a pull request.

## Layout pass-set 1 quantized cleanup characterization

The next family is restricted to the three consecutive quantized mapping
results in the first layout pass-set: quantized PReLU, Dequantize→TransposeConv
→Quantize, and quantized Reshape cleanup. They share the existing layout guard,
have explicit bounded integer schemas, and have no defaults or consumers.
Composite attention/QDQ results on both sides are excluded.

The contract fixes owner arguments and keywords, adjacency, composite
boundaries, guard, and absence of loads. Its strict expected failure requires
three `cleanup.layout_pass_set_1.*` quantized records.

Validation completed sequentially under core-only `uv`: the characterization
and existing owner/schema contracts are `7 passed, 1 xfailed in 0.83s`; Ruff,
bytecode compilation, and whitespace checks pass. No production source changed.

Commit and push this characterization before replacing only the three mapping
destinations. Preserve the outer guard and composite boundaries. Update the
three multi-occurrence owner contracts and bounded-store inventory, then
validate, document, commit, and push. Never create, update, or reopen a pull
request.

## Layout pass-set 1 quantized cleanup implementation

The three consecutive mapping results now use stable
`cleanup.layout_pass_set_1.*` quantized records. Their outer guard, owner
expressions, adjacency, and composite boundaries remain unchanged. No defaults
existed. The bounded store now covers 71 phase IDs.

The first architecture run reported two failures in stale adjacent-boundary
assertions that assumed every direct owner had an `ast.Name` outer call. Both
now unwrap and verify the exact owner nested in the phase record. The targeted
correction is `2 passed in 2.20s`; production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- direct affected contracts: `10 passed in 0.95s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader affected contracts: `273 passed in 6.62s`;
- lowerer architecture contracts: `258 passed in 16.61s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite exercises the guarded path. After committing and
pushing, keep composite results outside the mapping store and retain the
71/128 capacity audit when choosing the next family. Never create, update, or
reopen a pull request.

## Layout pass-set 2 quantized cleanup characterization

The sole remaining direct Dequantize→TransposeConv→Quantize mapping result is
inside layout pass-set 2. It shares the existing layout guard, has no default or
consumer, and remains between composite attention-gate/QDQ and quantized-
activation recovery results.

The contract fixes the owner expression and layout-state keyword, guard,
composite boundaries, sole Store occurrence, and absence of loads. Its strict
expected failure requires
`cleanup.layout_pass_set_2.dequant_transposeconv_quantize`.

Validation completed sequentially under core-only `uv`: the characterization,
multi-occurrence owner, and adjacent architecture boundaries are
`5 passed, 1 xfailed in 0.76s`; Ruff, bytecode compilation, and whitespace
checks pass. No production source changed.

Commit and push this characterization before replacing only the single mapping
destination. Preserve the guard and both composite boundaries, then update the
multi-occurrence owner and bounded-store contracts, validate, document, commit,
and push. Never create, update, or reopen a pull request.

## Layout pass-set 2 quantized cleanup implementation

The sole mapping result now uses the stable
`cleanup.layout_pass_set_2.dequant_transposeconv_quantize` record. Its outer
guard, owner expression, and composite recovery boundaries remain unchanged.
No default existed. The bounded store now covers 72 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct affected contracts: `8 passed in 0.82s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader affected contracts: `275 passed in 6.94s`;
- lowerer architecture contracts: `258 passed in 17.39s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only the observation destination
changed and the runtime suite exercises the guarded path. After committing and
pushing, retain the 72/128 phase-cap audit and keep composite results outside
the bounded integer mapping store. Never create, update, or reopen a pull
request.

## Layout pass-set 1 affine cleanup characterization

The next family is restricted to five mapping-only affine results under the
first layout guard: two affine-chain folds plus affine pre/post, pre-unary
affine fan-out, and mean-affine pre/post cleanup. Existing tests fix their
bounded integer schemas. They have no defaults or consumers.

The contract fixes exact owner arguments and keywords, the four-result prefix,
the separate post-binary fold, both composite recovery boundaries, the common
guard, and absence of loads. Composite return values remain excluded. Its
strict expected failure requires five `cleanup.layout_pass_set_1.*` affine
records.

The first focused run exposed one stale adjacency assertion for the previously
migrated safe-transpose phase record. Its helper now unwraps and verifies the
exact nested owner; production behavior was not implicated. The dedicated
characterization and existing owner/schema contracts are
`9 passed, 1 xfailed in 0.97s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only the five mapping
destinations. Preserve the guard and every composite boundary, then update
multi-occurrence owner and bounded-store contracts, validate, document,
commit, and push. Never create, update, or reopen a pull request.

## Layout pass-set 1 affine cleanup implementation

The five mapping results now use stable `cleanup.layout_pass_set_1.*` affine
records. The shared layout guard, four-result prefix, isolated post-binary
fold, owner expressions, and every composite recovery boundary remain
unchanged. No default existed. The bounded store now covers 77 phase IDs.

The broader result-contract run exposed a second stale AST helper for the
already-migrated no-layout safe-transpose record. The ConvPool boundary helper
now unwraps and verifies its exact nested owner; the targeted correction is
`2 passed in 0.14s` and production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- direct affected contracts: `13 passed in 2.90s`;
- synthetic core runtime contracts: `55 passed in 1.00s`;
- broader result contracts: `180 passed in 8.24s`;
- lowerer architecture contracts: `258 passed in 16.87s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite exercises the guarded path. After committing and
pushing, retain the 77/128 phase-cap audit and keep composite results outside
the bounded integer mapping store. Never create, update, or reopen a pull
request.

## Layout pass-set 1 residual cleanup characterization

The four remaining direct mapping observations in the first layout pass-set
are primary layout-Transpose cleanup, guarded Transpose/binary bridge cleanup,
duplicate fan-out cleanup, and Dequantize→Mean→Quantize bridge cleanup. Their
bounded integer schemas are already explicit, and none has a default or
consumer.

The contract fixes the outer and nested guards, exact owner expressions,
policy and composite boundaries, and absence of loads. Composite results stay
outside the migration. Its strict expected failure requires four stable
`cleanup.layout_pass_set_1.*` records.

Validation completed sequentially under core-only `uv`: the dedicated
characterization and existing owner/schema contracts are
`9 passed, 1 xfailed in 0.99s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only the four mapping
destinations. Preserve both guards and every boundary, then update all
multi-occurrence owner and bounded-store contracts, validate, document,
commit, and push. Never create, update, or reopen a pull request.

## Layout pass-set 1 residual cleanup implementation

The four remaining direct mapping results now use stable
`cleanup.layout_pass_set_1.*` records. The outer layout guard, nested binary
feature guard, owner expressions, policy boundary, and all composite recovery
boundaries remain unchanged. No default existed. The bounded store now covers
81 phase IDs.

Expanded architecture and orchestration validation exposed three stale AST
assertions that assumed outer `ast.Name` calls or three assignment statements.
They now verify the exact nested owner and the primary-record/two-late-
assignment split. Targeted corrections are `2 passed in 2.28s` and
`1 passed in 0.56s`; production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- direct affected contracts: `14 passed in 1.61s`;
- synthetic core runtime contracts: `55 passed in 1.06s`;
- broader result contracts: `182 passed in 8.70s`;
- QLinear and terminal-layout orchestration contracts:
  `71 passed in 1.96s`;
- lowerer architecture contracts: `258 passed in 17.42s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite exercises both guarded paths. After committing
and pushing, retain the 81/128 phase-cap audit and keep composite results
outside the bounded integer mapping store. Never create, update, or reopen a
pull request.

## Layout pass-set 2 residual cleanup characterization

The remaining direct mapping family in the second layout pass-set contains
eight consecutive elementwise/Concat/SPP/layout cleanup observations and one
later SA/PA MirrorPad observation. All nine share the outer layout guard, have
explicit bounded integer schemas, and have no defaults or consumers.

The contract fixes exact owner expressions, the eight-result adjacency,
composite boundaries on both clusters, the isolated SA/PA position, and
absence of loads. Composite recovery results remain excluded. Its strict
expected failure requires nine stable `cleanup.layout_pass_set_2.*` records.

Validation completed sequentially under core-only `uv`: the owner/schema
baseline is `22 passed in 1.42s`; the characterization plus those contracts is
`23 passed, 1 xfailed in 1.59s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only the nine mapping
destinations. Preserve the outer guard, adjacency, and every composite
boundary, then update multi-occurrence owner and bounded-store contracts,
validate, document, commit, and push. Never create, update, or reopen a pull
request.

## Layout pass-set 2 residual cleanup implementation

The nine mapping results now use stable `cleanup.layout_pass_set_2.*` records.
The eight-result adjacency, isolated SA/PA position, outer layout guard, owner
expressions, and every composite recovery boundary remain unchanged. No
default existed. The bounded store now covers 90 phase IDs.

Expanded orchestration validation exposed three stale test helpers for
previously migrated quantized and Dequantize→Mean→Quantize phase owners. They
now unwrap and verify the exact nested call; the targeted correction is
`3 passed in 0.62s` and production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- direct nine-result contracts: `24 passed in 1.60s`;
- expanded affected contracts: `64 passed in 2.38s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result contracts: `184 passed in 11.76s`;
- lowerer architecture contracts: `258 passed in 16.96s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite exercises the guarded path. After committing and
pushing, retain the 90/128 phase-cap audit and keep composite results outside
the bounded integer mapping store. Never create, update, or reopen a pull
request.

## Terminal boundary cleanup characterization

The next family is restricted to seven consecutive unconditional terminal
mapping results: pre-ArgMax, Transpose/Gather fan-out, Softmax/Transpose,
boundary normalization, two channel-slice cleanups, and channel-slice Mul/Add
bridge cleanup. Existing tests fix their bounded integer schemas; none has a
default or consumer.

The contract fixes exact owner expressions, adjacency, the preceding terminal
Conv-activation phase record, the following Slice/Concat composite, and absence
of loads. Its strict expected failure requires seven stable
`cleanup.terminal.*` records.

Validation completed sequentially under core-only `uv`: the related baseline
is `152 passed in 2.31s`; characterization plus related contracts is
`153 passed, 1 xfailed in 2.51s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only the seven mapping
destinations. Preserve adjacency and both boundaries, then update terminal
orchestration and bounded-store contracts, validate, document, commit, and
push. Never create, update, or reopen a pull request.

## Terminal boundary cleanup implementation

The seven consecutive mapping results now use stable `cleanup.terminal.*`
records. Their unconditional execution, owner expressions, adjacency,
preceding Conv-activation phase, and following Slice/Concat composite remain
unchanged. No default existed. The bounded store now covers 97 phase IDs.

Expanded validation exposed 13 stale AST assertions that assumed assignment
targets or outer owner calls. They now verify exact phase IDs/nested owners and
preserve expectations for later non-migrated assignments. Targeted corrections
are `11 passed in 1.13s`, `2 passed in 0.35s`, and `1 passed in 2.67s`;
production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- terminal boundary and related contracts: `156 passed in 2.72s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result contracts: `186 passed in 9.83s`;
- lowerer architecture contracts: `258 passed in 18.19s`;
- targeted Ruff, bytecode compilation, and whitespace checks: passed.

No root-model conversion was required because only observation destinations
changed and the runtime suite exercises the terminal path. After committing
and pushing, retain the 97/128 phase-cap audit and keep composite results
outside the bounded integer mapping store. Never create, update, or reopen a
pull request.

## Terminal activation cleanup characterization

The next family is restricted to four consecutive unconditional results after
the terminal Slice/Concat composite: boundary StridedSlice/QDQ/Concat, Swish
residual/Concat, Dequantize/Logistic/Mul/Quantize, and Swish QDQ-island
cleanup. Existing contracts fix their bounded integer schemas; none has a
default or consumer.

The contract fixes exact owner expressions, adjacency, both composite/cleanup
boundaries, and absence of loads. Its strict expected failure requires four
stable `cleanup.terminal.*` records.

Validation completed sequentially under core-only `uv`: the related baseline
is `72 passed in 1.99s`; characterization plus related contracts is
`73 passed, 1 xfailed in 2.08s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only the four mapping
destinations. Preserve adjacency and both boundaries, then update terminal
orchestration and bounded-store contracts, validate, document, commit, and
push. Never create, update, or reopen a pull request.

## Terminal activation cleanup implementation

The four characterized results now use the following stable records in their
original source order:

- `cleanup.terminal.boundary_stridedslice_qdq_concat`;
- `cleanup.terminal.swish_residual_concat_closure`;
- `cleanup.terminal.dequant_logistic_mul_quantize_bridge`;
- `cleanup.terminal.swish_qdq_island`.

The change replaces only unconsumed local mapping destinations. It preserves
all owner calls, arguments, keywords, unconditional execution, adjacency, the
Slice/Concat and InstanceNorm boundaries, graph behavior, public contracts,
dependencies, and TensorFlow isolation. The phase store now contains 101/128
records, leaving 27 slots. Composite `*_results` remain outside the bounded
integer mapping store.

Six representation-dependent assertions were updated to unwrap the record
while retaining exact owner, phase, argument, keyword, and boundary checks.
Production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- focused activation/store/terminal/Slice-Concat contracts:
  `75 passed in 2.42s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader result contracts: `187 passed in 8.61s`;
- lowerer architecture contracts: `258 passed in 18.51s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required for this observation-only destination
migration. Commit and push this implementation as a self-contained unit. The
next refactor unit must begin with a fresh characterize-first audit and must
not create, update, or reopen a pull request.

## Terminal normalization cleanup characterization

The next unit is restricted to five consecutive unconditional results between
terminal Swish QDQ-island cleanup and the terminal boundary-layout composite:
InstanceNorm post-bias, normalization Pad, InstanceNorm residual Add,
InstanceNorm residual Mul/Concat, and InstanceNorm dual-stat cleanup. All five
owners return bounded integer mappings; the locals have no defaults or loads.

The strict contract fixes exact owner expressions and keywords, adjacency,
both outer boundaries, and the five proposed `cleanup.terminal.*` phase IDs.
No production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `97 passed in 2.59s`; characterization plus related contracts is
`98 passed, 1 xfailed in 2.59s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push the characterization before replacing only these five mapping
destinations. Preserve both boundaries and source order, update all
representation-dependent terminal contracts, run the sequential gates,
document, commit, and push. Never create, update, or reopen a pull request.

## Terminal normalization cleanup implementation

The five characterized results now use stable `cleanup.terminal.*` records in
their original source order. Only their unconsumed local destinations changed.
Owner calls, arguments, keywords, unconditional execution, evaluation count,
the preceding Swish QDQ-island record, the following terminal boundary-layout
composite, graph behavior, public contracts, dependencies, and TensorFlow
isolation remain unchanged.

The bounded store now contains 106/128 records, leaving 22 slots. Composite
`*_results` stay outside the integer mapping store. Seventeen stale structural
assertions were updated to unwrap the record while preserving exact phase,
owner, call-count, later-assignment, and boundary checks.

Validation completed sequentially under core-only `uv`:

- focused normalization/store/terminal/boundary contracts:
  `100 passed in 2.81s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader result contracts: `188 passed in 8.64s`;
- lowerer architecture contracts: `258 passed in 18.46s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required for this observation-only destination
migration. Commit and push this implementation as a self-contained unit. The
next unit must begin with a fresh characterize-first audit and must never
create, update, or reopen a pull request.

## Guarded terminal BatchMatMul characterization

The next unit is restricted to three consecutive mapping results inside the
existing `optimize_layout_transpose_chains` guard: BatchMatMul affine-input,
Reshape/SE, and adjoint-flag cleanup. Their bounded single-integer schemas and
no-op behavior are already covered; the local results have no defaults or
loads.

The strict contract fixes the shared guard, exact owner expressions,
adjacency, the Mean-attention and QKV-attention composite boundaries, and the
three proposed `cleanup.terminal.*` phase IDs. Both composites remain outside
the bounded mapping store. No production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `23 passed in 0.94s`; characterization plus related contracts is
`24 passed, 1 xfailed in 1.14s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only these three
guarded mapping destinations. Preserve the guard, both composites, and source
order; update representation-dependent contracts; run the sequential gates;
document, commit, and push. Never create, update, or reopen a pull request.

## Guarded terminal BatchMatMul implementation

The three characterized BatchMatMul observations now use stable
`cleanup.terminal.*` records inside the unchanged
`optimize_layout_transpose_chains` guard. Their owner calls, arguments, order,
Mean-attention predecessor, QKV-attention successor, post-SiNet counterparts,
graph behavior, public contracts, dependencies, and TensorFlow isolation are
unchanged. The composite results remain outside the bounded mapping store.

The phase store now contains 109/128 records, leaving 19 slots. Five stale
structural assertions were changed to unwrap nested owners and locate the
guard through retained composites while preserving both-call-site checks.

Validation completed sequentially under core-only `uv`:

- focused BatchMatMul/QKV/store contracts: `26 passed in 1.22s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- broader result contracts: `189 passed in 9.37s`;
- lowerer architecture contracts: `258 passed in 17.61s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required for this observation-only destination
migration. Commit and push this unit. Start the next unit with a fresh
characterize-first audit and never create, update, or reopen a pull request.

## Guarded terminal QKV bridge characterization

The next unit contains only the guarded QKV Split/Conv/Concat bridge mapping
result between the existing QKV-attention and singleton-reshape composites.
Its indexed owner contract already covers the bounded counter schema, no-op,
dynamic, fan-out, quantized, invariant, and rollback paths. The result has no
default or load.

The strict contract fixes the guard, exact owner expression, both composite
boundaries, and proposed
`cleanup.terminal.qkv_split_conv_concat_bridge` record. No production source
changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `103 passed in 1.05s`; characterization plus related contracts is
`104 passed, 1 xfailed in 1.13s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only the one mapping
destination. Preserve the guard and both composites, update structural
contracts, run sequential gates, document, commit, and push. Never create,
update, or reopen a pull request.

## Guarded terminal QKV bridge implementation

The one characterized result now records under
`cleanup.terminal.qkv_split_conv_concat_bridge` inside the unchanged layout
optimization guard. Its owner, argument, layout-state keyword, evaluation
count, QKV and singleton composite boundaries, graph behavior, later owner
calls, public contracts, dependencies, and TensorFlow isolation are unchanged.

The bounded store now contains 110/128 records, leaving 18 slots. Two stale
structural assertions now unwrap the nested owner while retaining three-call-
site and boundary checks.

Validation completed sequentially under core-only `uv`:

- focused bridge/QKV/singleton/store contracts: `106 passed in 1.37s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader result contracts: `190 passed in 9.42s`;
- lowerer architecture contracts: `258 passed in 17.17s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because the indexed synthetic suite
executes this bridge. Commit and push this unit. Begin the next unit with a
fresh characterize-first audit and never create, update, or reopen a pull
request.

## Terminal HardSwish/HardSigmoid characterization

The next unit contains the consecutive unconditional SiNet HardSwish-SE and
Dequantize/HardSigmoid/Quantize bridge mappings between the retained SiNet
terminal and pre-Add/Resize composites. Existing result and owner contracts
already cover their schemas, cleanup, production forms, arguments, boundaries,
and lack of consumers.

The existing HardSwish-SE result module now includes a strict expected failure
for two exact `cleanup.terminal.*` records. This avoids introducing a redundant
large structural module. No production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `21 passed in 1.00s`; characterization plus related contracts is
`21 passed, 1 xfailed in 1.10s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push the characterization before replacing only the two mapping
destinations. Preserve both SiNet composites and source order, update stale
structural contracts, run sequential gates, document, commit, and push. Never
create, update, or reopen a pull request.

## Terminal HardSwish/HardSigmoid implementation

The pair now records as `cleanup.terminal.sinet_hardswish_se` and
`cleanup.terminal.dequant_hardsigmoid_bridge` in the original unconditional
positions. Owner calls, arguments, order, both SiNet composite boundaries,
later production forms, graph behavior, public contracts, dependencies, and
TensorFlow isolation remain unchanged.

The bounded store now contains 112/128 records, leaving 16 slots. Six stale
structural assertions now unwrap phase records while preserving exact phase,
owner, later-form, and composite-name checks. The temporary characterization
expectation was converted in the existing focused module rather than leaving
a redundant new test file.

Validation completed sequentially under core-only `uv`:

- focused HardSwish/HardSigmoid/SiNet/store contracts:
  `23 passed in 1.21s`;
- synthetic core runtime contracts: `55 passed in 1.07s`;
- broader result contracts: `190 passed in 9.59s`;
- lowerer architecture contracts: `258 passed in 20.53s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because focused runtime tests cover both
owners. Commit and push this unit. Start the next unit with a fresh
characterize-first audit and never create, update, or reopen a pull request.

## Post-terminal indexed shape convergence characterization

The next unit contains the single top-level indexed shape/topology convergence
mapping between the post-terminal singleton and very-late SiNet composites.
Its owner reuses one graph index for pruning, dynamic Reshape resolution, and
static-shape reconciliation, returning three bounded integer counters. The
local has no default or load.

The existing focused result module now strictly expects
`shape_topology.terminal.indexed_convergence` with exact owner arguments and
both composite boundaries while preserving the separate nested convergence
result. No production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `10 passed in 0.71s`; characterization plus related contracts is
`10 passed, 1 xfailed in 0.78s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push the characterization before replacing only the top-level
mapping destination. Preserve both composites and the nested result, update
structural contracts, run sequential gates, document, commit, and push. Never
create, update, or reopen a pull request.

## Post-terminal indexed shape convergence implementation

The top-level result now records under
`shape_topology.terminal.indexed_convergence` in its original position. The
owner, arguments, one-index convergence behavior, three-counter schema,
nested result, singleton/SiNet composite boundaries, graph behavior, public
contracts, dependencies, and TensorFlow isolation are unchanged.

The bounded store now contains 113/128 records, leaving 15 slots. Four stale
structural assertions now unwrap the record and preserve exact phase, owner,
composite, and nested-result checks.

Validation completed sequentially under core-only `uv`:

- focused indexed-convergence/SiNet/store contracts: `12 passed in 0.81s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader result contracts: `190 passed in 8.96s`;
- lowerer architecture contracts: `258 passed in 19.09s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because synthetic contracts exercise
the owner. Commit and push this unit. Begin the next unit with a fresh
characterize-first audit and never create, update, or reopen a pull request.

## Very-late residual cleanup characterization

The next unit contains three consecutive unconditional results between the
very-late and post-cleanup SiNet pre-Add/Resize composites: residual affine
PReLU, residual affine Transpose fan-out, and indexed prune/reconcile cleanup.
Existing tests cover their integer schemas, cleanup, routing, arguments,
one-index behavior, and absence of loads.

The prune/reconcile result module now strictly expects three exact
`cleanup.very_late.*` records with adjacency and both composite boundaries.
No production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `17 passed in 0.98s`; characterization plus related contracts is
`17 passed, 1 xfailed in 1.01s`; targeted Ruff, bytecode compilation, and
whitespace checks pass.

Commit and push this characterization before replacing only the three mapping
destinations. Preserve both composites and source order, update structural
contracts, run sequential gates, document, commit, and push. Never create,
update, or reopen a pull request.

## Very-late residual cleanup implementation

The three results now record under the characterized `cleanup.very_late.*`
phase IDs in their original unconditional order. Owner calls, arguments,
indexed prune/reconcile behavior, both SiNet composite boundaries, graph
behavior, public contracts, dependencies, and TensorFlow isolation remain
unchanged.

The bounded store now contains 116/128 records, leaving 12 slots. Affected
residual, prune/reconcile, SiNet, and architecture contracts now unwrap the
records while preserving exact phase, owner, and composite checks.

Validation completed sequentially under core-only `uv`:

- focused residual/prune/SiNet/store contracts: `20 passed in 1.10s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result contracts: `191 passed in 9.62s`;
- lowerer architecture contracts: `258 passed in 18.50s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because focused runtime contracts cover
all three owners. Commit and push this unit. Begin the next unit with a fresh
characterize-first audit and never create, update, or reopen a pull request.

## Post-cleanup attention result characterization

The next bounded-store unit contains the consecutive top-level CSP attention
and SA/PA MirrorPad cleanup observations immediately after
`_post_cleanup_sinet_preadd_resize_results`. Both owners already have focused
schema and cleanup coverage, return a single integer counter, receive
`session.layout_state`, and have no result consumer.

The existing CSP attention result module now has a strict expected-failure
contract for:

- `cleanup.post_cleanup.csp_attention`;
- `cleanup.post_cleanup.sa_pa_mirrorpad`.

It fixes exact nested owner expressions, source adjacency, the preceding SiNet
composite, the following `_post_sinet_batchmatmul_affine_input_stats` result,
and removal of both old local targets. Production source is unchanged.

The related pre-characterization baseline is `14 passed in 0.88s`. Run the
same four focused modules and expect one additional strict xfail, then run
targeted Ruff, bytecode compilation, and whitespace validation. Commit and
push that characterization before changing the lowerer. On implementation,
change only the two result destinations, grow the store contract from 116 to
118 records, keep the existing owner calls and phase boundaries, run all
sequential gates, document, commit, and push. Never create, update, or reopen
a pull request.

## Post-cleanup attention result implementation

The two results now record under the characterized
`cleanup.post_cleanup.csp_attention` and
`cleanup.post_cleanup.sa_pa_mirrorpad` phase IDs. The change removes only the
two unconsumed local destinations; calls, arguments, execution count, order,
layout-state use, surrounding composite and BatchMatMul boundaries, graph
behavior, public contracts, dependencies, and TensorFlow isolation are
unchanged.

The bounded store now contains 118/128 records, leaving 10 slots. Affected
structural assertions unwrap phase records and retain exact phase, nested
owner, argument, and boundary checks. The strict characterization expectation
now passes.

Validation completed sequentially under core-only `uv`:

- focused boundary/store contracts: `17 passed in 1.12s`;
- focused CSP/MirrorPad runtime and orchestration contracts:
  `68 passed in 1.20s`;
- synthetic core runtime contracts: `55 passed in 1.12s`;
- broader result contracts: `192 passed in 9.46s`;
- lowerer architecture contracts: `258 passed in 21.25s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because this unit changes only the
destination of already-computed one-counter mappings and focused runtime
tests cover both owners. Commit and push this unit. Begin the next unit with a
fresh characterize-first audit and never create, update, or reopen a pull
request.

## Post-SiNet BatchMatMul result characterization

The next bounded-store unit is limited to the three consecutive post-SiNet
BatchMatMul locals after `cleanup.post_cleanup.sa_pa_mirrorpad` and before the
retained `_post_sinet_qkv_attention_results` composite. The affine-input,
Reshape/SE, and adjoint-flag owners each return one bounded integer counter;
their runtime semantics are already covered, and none of the three locals is
loaded.

The affine-input test module now has a strict expected-failure contract for
the three exact `cleanup.post_sinet.batchmatmul_*` records. It fixes owner
expressions, adjacency, both outer boundaries, and absence of consumers. No
production source changed.

The related baseline is `28 passed in 1.20s`. Run the same six focused modules
and expect one additional strict xfail, then run targeted Ruff, bytecode
compilation, and whitespace validation. Commit and push characterization
before implementation. Then change only the three destinations, expand the
store contract from 118 to 121, preserve the QKV composite, run sequential
gates, document, commit, and push. Never create, update, or reopen a pull
request.
