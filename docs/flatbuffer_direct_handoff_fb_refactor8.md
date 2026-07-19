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

## Terminal clamp/SiNet layout implementation checkpoint

`passes/terminal_clamp_sinet_layout_orchestration.py` now implements the fixed
two-child boundary. The embedded shared pass context is passed to terminal
Clamp/unary/ReLU; the exact original SiNet context and pre-add/resize callback
are passed to SiNet terminal-layout recovery. Both complete results are
returned unchanged and in their original order.

The lowerer retains one observation-only `_terminal_clamp_sinet_layout_results`
value in place of the two child locals. The terminal layout conditional and
following SiNet hard-swish/SE phase record remain immediate outer boundaries.
Both zero-argument wrappers remain defined, and the independent very-late SiNet
route remains unchanged. The characterized unconsumed-result inventory is now
54; the phase store remains exactly 128 IDs and 128 owners.

Sequential validation passed: focused 3, affected 464,
terminal-layout/efficiency 92, core 55, result contracts 196, phase store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff, bytecode
compilation, and whitespace checks pass. Runtime injection proves order,
embedded pass-context identity, original context/callback identity, and both
raw-result identities. No production test or known issue is failing, and no
real-model conversion was repeated for this ownership-only extraction.

At resume, rerun the read-only inventory of the 54 remaining characterized
unconsumed lowerer results and select the next smallest source-adjacent,
semantically closed cluster whose children already have pass-module owners.
Characterize it before production changes, keep all tests sequential under
`uv`, and commit/push only at complete checkpoints. Do not create, update, or
reopen a pull request.

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

## Terminal clamp/SiNet layout characterization checkpoint

The current characterized inventory contains 55 unconsumed lowerer results.
The next selected adjacent pair is terminal Clamp/unary/ReLU followed by SiNet
terminal-layout recovery. The SiNet context embeds the same shared pass context
used by the first child and preserves the existing pre-add/resize callback.
There is no intervening branch, phase record, progress update, result consumer,
or other mutation.

The contract requires one two-child owner with fixed order, exact embedded
pass-context and callback preservation, and identity preservation for both
complete results. The terminal layout conditional remains the predecessor and
the phase-recorded SiNet hard-swish/SE cleanup remains the successor. Both
lowerer wrappers remain compatibility routes.

Focused characterization reports `1 passed, 1 xfailed`; the affected suite
reports `462 passed, 1 xfailed`. The sole xfail is the intentionally absent
`passes/terminal_clamp_sinet_layout_orchestration.py`. Production and the
exactly 128-ID/128-owner phase store remain unchanged.

At resume, implement `run_terminal_clamp_sinet_layout_cleanup(context)` as a
straight-line owner. Pass `context.pass_context` to terminal
Clamp/unary/ReLU, pass the exact original context to SiNet terminal-layout
recovery, return both raw results unchanged, and replace only the two
characterized lowerer locals. Preserve the preceding conditional, following
phase record, callback, and both wrappers. Add runtime
order/context/callback/result-identity injection, run affected and standard
gates sequentially, then commit and push only. Do not create, update, or reopen
a pull request.

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

## Terminal affine/Slice-SPP composite implementation checkpoint

The characterized three-stage owner is implemented in
`passes/terminal_affine_slice_spp_orchestration.py`. It passes the exact shared
context to terminal affine and late SPP summaries, passes the shared model to
the strict Slice bridge, and returns the three original mappings without
copying. The lowerer now stores one ordered composite tuple in place of the
three unconsumed child locals while retaining all wrappers and context aliases
needed by earlier independent routes.

The pre-terminal composite and QKV shape-extract result remain the exact outer
boundaries. Owner-aware tests preserve specialized child behavior and mutation
counts, and runtime injection verifies call order, context/model identity, and
raw-result identity. Sequential validation passed: focused 4, affected 520,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow isolation/default-direct/`-cotof` 11. No test is failing, no
phase entry changed, and the store remains exactly 128 IDs and 128 owners.

At resume, rerun the read-only unconsumed-result inventory after removal of
these three locals. Select the next smallest adjacent semantically closed
cluster whose execution guard and context policy are uniform, characterize it
before production changes, and keep all `uv` validation sequential and
single-process. Commit and push each checkpoint only. Never create, update,
reopen, or otherwise modify a pull request.

## Terminal QKV shape/attention composite characterization checkpoint

The post-removal inventory selected the adjacent QKV shape-extract result and
terminal QKV summary immediately after
`_terminal_affine_slice_spp_results`. Both are unconditional and unconsumed.
The QKV summary uses the exact shared context with
`include_layout_transpose=optimize_layout_transpose_chains` and
`include_prefix=False`; shape cleanup receives the shared model. The indexed
Split/Conv/Concat bridge is the fixed successor.

The new strict contract also preserves the later independent shape-extract
call and existing raw QKV wrapper routes. It fixes exact arguments, outer
boundaries, result schemas, both layout-option variants, and absence of
consumers. Focused validation reports `3 passed, 1 xfailed`; affected
sequential validation reports `443 passed, 1 xfailed`. The sole expected
failure requires `passes/terminal_qkv_shape_attention_orchestration.py`.

At resume, implement that module as a two-stage owner accepting the shared
`ModelIRPassContext` and keyword-only `include_layout_transpose`. Pass
`context.model_ir` to shape cleanup and `context` to the QKV summary, preserve
`include_prefix=False`, return both raw mapping objects unchanged, and replace
only the two unconsumed locals. Keep the later shape call and raw QKV wrappers
independent. Run affected and standard gates sequentially, then commit and
push only. Never create or modify a pull request.

## Terminal QKV shape/attention composite implementation checkpoint

The characterized two-stage owner is implemented in
`passes/terminal_qkv_shape_attention_orchestration.py`. It forwards the shared
model to terminal shape-extract cleanup and the exact shared context to the
QKV summary, retaining the layout option and `include_prefix=False`. The
lowerer stores one ordered tuple instead of two unconsumed locals.

The later independent shape-extract invocation and all raw QKV wrapper routes
remain intact. The terminal affine/Slice-SPP predecessor and indexed
Split/Conv/Concat successor are unchanged. Runtime identity tests and
owner-aware structural coverage verify all state, policy, result, route, and
boundary contracts.

Sequential validation passed: focused 5, affected 445,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow isolation/default-direct/`-cotof` 11. No test is failing, no
phase result changed, and the store remains exactly 128 IDs and 128 owners.

At resume, rerun the unconsumed-result inventory after removal of this pair.
Select the next smallest adjacent terminal cluster with uniform guard and
context policy, characterize it before implementation, and keep every `uv`
test sequential and single-process. Commit and push only; never create,
update, reopen, or otherwise modify a pull request.

## Terminal activation-bridge composite characterization checkpoint

The refreshed inventory selected indexed Split/Conv/Concat bridge,
HardSwish-SE summary, and late hard-activation summary immediately after the
terminal QKV composite. All three are unconditional and unconsumed. Their
argument policy is shared model/LayoutState, shared model, and shared context
plus the layout option, respectively. Absolute-final pre-ConCat cleanup is the
fixed successor.

The strict contract preserves exact call order, argument and context identity,
both layout-option schemas, outer boundaries, and raw mapping schemas of
lengths `(1, 2, 8)`. Focused validation reports `3 passed, 1 xfailed` and
affected validation reports `443 passed, 1 xfailed`. The sole expected failure
requires `passes/terminal_activation_bridge_orchestration.py`.

At resume, implement that module as a three-stage owner accepting the shared
`ModelIRPassContext` and keyword-only `include_layout_transpose`. Forward the
shared model/LayoutState to the Split bridge, the shared model to HardSwish-SE,
and the context/option to hard activation. Return all raw mappings unchanged,
replace only the three unconsumed locals, and retain both outer boundaries and
all compatibility routes. Run affected and standard gates sequentially,
commit, and push only. Never create or modify a pull request.

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

## Post-SiNet BatchMatMul result implementation

The three results now record under their characterized
`cleanup.post_sinet.batchmatmul_*` phase IDs. Only their unused destinations
changed. Owner calls and arguments, unconditional order, outer MirrorPad and
QKV boundaries, graph behavior, public contracts, artifacts, dependencies,
and TensorFlow isolation are unchanged. The QKV composite remains a retained
local and is not copied into the bounded store.

The store now contains 121/128 records, leaving 7 slots. Affected structural
tests unwrap the records and retain exact phase, owner, argument, adjacency,
and composite assertions. The strict characterization expectation now passes.

Validation completed sequentially under core-only `uv`:

- focused BatchMatMul/QKV/attention/store contracts: `31 passed in 1.46s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- broader result contracts: `192 passed in 9.08s`;
- lowerer architecture contracts: `258 passed in 17.00s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because only the destination of three
already-computed one-counter mappings changed and focused runtime tests cover
all owners. Commit and push this unit. Start the next unit with a fresh
characterize-first audit and never create, update, or reopen a pull request.

## Post-SiNet ReLU/Split result characterization

The next bounded-store unit contains the three top-level ReLU/Split results
between `_post_sinet_qkv_attention_results` and
`_post_sinet_mix_attention_stats`. All-Split-output propagation,
Split/Conv/ReLU/Concat propagation, and Split/Conv/Concat bridge cleanup each
return one integer counter, receive `session.layout_state`, and have no result
consumer.

The existing ReLU/Split/Conv/Concat result module now has a strict
expected-failure contract for their exact `cleanup.post_sinet.*` records,
owner expressions, adjacency, outer boundaries, and absence of loads. No
production source changed.

The related baseline is `72 passed in 1.20s`. Re-run the five focused modules
and expect one additional strict xfail, then run targeted Ruff, bytecode
compilation, and whitespace validation. Commit and push characterization
first. Implementation must change only the three destinations, move the store
from 121 to 124 records, preserve the QKV composite and mix-attention result,
run sequential gates, document, commit, and push. Never create, update, or
reopen a pull request.

## Post-SiNet ReLU/Split result implementation

The three results now record under their characterized
`cleanup.post_sinet.*` phase IDs. Only their unused local destinations
changed. All owner calls, layout-state arguments, execution order, outer QKV
and mix-attention boundaries, graph behavior, public contracts, artifacts,
dependencies, and TensorFlow isolation remain unchanged.

The bounded store now contains 124/128 records, leaving 4 slots. Affected
structural tests unwrap records and retain exact phase, owner, argument,
adjacency, and composite assertions. The strict characterization expectation
now passes.

Validation completed sequentially under core-only `uv`:

- focused ReLU/Split/QKV/mix-attention/store contracts:
  `75 passed in 1.51s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader result contracts: `193 passed in 9.15s`;
- lowerer architecture contracts: `258 passed in 17.38s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because only three already-computed
one-counter result destinations changed and focused runtime tests cover their
owners. Commit and push this unit. Begin the next unit with a fresh
characterize-first audit and never create, update, or reopen a pull request.

## Post-SiNet attention and activation result characterization

The next bounded-store unit contains the consecutive SiNet mix-attention,
mixed-attention layout, and Dequantize/HardSigmoid bridge results between
`cleanup.post_sinet.split_conv_concat_bridge` and creation of
`late_ndhwc_cost_volume_state_scope`. All three mappings contain only one
integer counter. The mixed-attention layout owner still writes normal pass
diagnostics separately, and none of these result locals has a consumer.

The existing mixed-attention result module now has a strict expected-failure
contract for exact `cleanup.post_sinet.*` records, owner expressions and
arguments, adjacency, outer boundaries, and absence of loads. No production
source changed.

The related baseline is `14 passed in 1.04s`. Re-run the same five focused
modules and expect one strict xfail, then run targeted Ruff, bytecode
compilation, and whitespace validation. Commit and push characterization
first. Implementation must change only the three destinations, preserve the
diagnostics stream and state-scope boundary, move the store from 124 to 127
records, run sequential gates, document, commit, and push. Never create,
update, or reopen a pull request.

## Post-SiNet attention and activation result implementation

The three results now record under their characterized
`cleanup.post_sinet.*` phase IDs. Only unused destinations changed. Owner
calls, layout-state and diagnostics arguments, execution order, outer
Split/Conv/Concat and NDHWC/cost-volume scope boundaries, diagnostic events,
graph behavior, public contracts, dependencies, and TensorFlow isolation are
unchanged.

The store now contains 127/128 records, leaving one slot. During the broad
gate, one architecture-only assertion failed because it assumed the two
pre-scope calls were assignments. Focused, core, and result suites were green.
The assertion was made phase-aware and now verifies exact phase IDs plus the
nested owners; the full architecture suite then passed.

Validation completed sequentially under core-only `uv`:

- focused attention/activation/state-scope/store contracts:
  `17 passed in 1.23s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader result contracts: `194 passed in 9.03s`;
- repaired architecture contract: `1 passed in 2.17s`;
- full architecture contracts: `258 passed in 16.99s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model conversion was required because only three already-computed
one-counter result destinations changed and focused runtime tests cover all
owners. Commit and push this unit. With only one store slot remaining, audit
the next candidate independently before using it; never create, update, or
reopen a pull request.

## Late NDHWC/cost-volume pair characterization

The final available store slot must not be used for only half of the adjacent
NDHWC gate and cost-volume ScatterND pair. Both currently share one
`ModelIRPassStateScope`; their mappings contain two and one distinct integer
counters respectively.

The chosen contract is one orchestration owner,
`run_late_ndhwc_cost_volume_layout_cleanup`, which creates the shared scope,
calls the same two owners in order with unchanged model/layout/diagnostics
inputs, and returns the merged three-counter mapping. The lowerer will retain
that result as `cleanup.late.ndhwc_cost_volume` between the post-SiNet
HardSigmoid phase and late Conv-affine result. The old scope and two result
locals must disappear.

The existing result module now contains a strict expected-failure contract
for this owner, phase, nested expression, boundaries, and old-local removal.
Production source is unchanged. The related baseline is
`16 passed in 0.86s`. Re-run it with one expected xfail and run targeted Ruff,
bytecode compilation, and whitespace validation before committing and
pushing characterization.

Implementation must add a focused runtime test proving callback order, shared
scope identity, arguments, and merged schema; then move the store from 127 to
128 records, run all sequential gates, document, commit, and push. Never
create, update, or reopen a pull request.

## Late NDHWC/cost-volume pair implementation

The new `run_late_ndhwc_cost_volume_layout_cleanup` owner creates one internal
state scope, calls the NDHWC and cost-volume owners in the original order, and
returns their merged three-counter mapping. The lowerer records it as
`cleanup.late.ndhwc_cost_volume` through
`shared_model_ir_pass_context`. The old scope and two result locals are
removed; direct low-level imports remain compatibility re-exports.

A runtime contract uses isolated callbacks to prove exact order, identical
model/layout/diagnostics objects, shared scope identity, and result schema.
The bounded store is now full at 128/128. Do not raise the limit or add another
record as a mechanical follow-up; future observation work needs an explicit
retention or aggregation decision.

Broad validation found only stale structural contracts: one former
three-statement boundary assertion, two former scope-successor assertions,
and two owner-count assertions. They now verify the combined phase, nested
owner, internal shared scope, and distinct late orchestration pass-ID sequence.
No runtime or numerical failure occurred.

Final sequential validation under core-only `uv`:

- focused pair/gate/store/architecture contracts: `20 passed in 2.76s`;
- pass-efficiency and terminal-layout contracts: `94 passed in 2.12s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- focused repaired boundaries: `11 passed in 0.87s`;
- broader result contracts: `196 passed in 9.12s`;
- focused architecture ownership contracts: `2 passed in 2.29s`;
- full architecture contracts: `258 passed in 16.88s`;
- targeted Ruff, bytecode compilation, full-capacity AST audit, and
  whitespace checks: passed.

No root-model conversion was required because this is a characterized
two-call owner extraction with focused runtime equivalence. Commit and push
this unit. Resume with a non-store refactoring audit, keeping the 128-phase
bound fixed, and never create, update, or reopen a pull request.

## Late Concat shared-scope characterization

The next unit is intentionally outside the full phase store. Four adjacent
late Concat/layout owners share `late_concat_layout_state_scope`: axis-3
constant Concat, Dequantize/Concat/Quantize, LayerNorm statistics, and layout
Transpose cleanup. Their four locals have no consumers.

The new focused module fixes current order, arguments, shared-scope loads,
outer boundaries, and unconsumed results. Its strict expected-failure contract
requires a new `run_late_concat_layout_cleanup` orchestration owner and one
`_late_concat_layout_results` tuple assignment in the lowerer. The owner must
create the scope internally and return all four mappings in order; this
composite remains outside `ConversionSession.phase_results`.

The related baseline is `72 passed in 0.86s`. Run the focused characterization
with one expected xfail, plus targeted Ruff, bytecode compilation, and
whitespace validation. Commit and push characterization first. Implementation
must leave the store at 128/128, add a runtime shared-scope/order/tuple test,
update structural ownership counts, run sequential gates, document, commit,
and push. Never create, update, or reopen a pull request.

## Late Concat shared-scope implementation

`run_late_concat_layout_cleanup` now creates the scope internally, calls the
four owners in the fixed order, and returns their four mappings as an ordered
tuple. The lowerer keeps one `_late_concat_layout_results` composite via the
shared pass context. It remains outside the phase store, which stays fixed at
128/128.

The former scope and four unconsumed locals are gone, shortening lowerer state
and the scope lifetime. Low-level imports remain compatibility re-exports;
the unused lowerer core scope import was removed. A runtime test proves order,
context identity, shared scope identity, and tuple order.

Four focused and two architecture assertions initially reflected the old
source representation. They now require the composite, internal scope,
nested owner arguments, and the new four-ID orchestration sequence. No runtime
or numerical failure occurred.

Final sequential validation under core-only `uv`:

- focused late-Concat/owner contracts: `76 passed in 3.10s`;
- terminal-layout/pass-efficiency contracts: `94 passed in 1.94s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- broader result contracts: `196 passed in 9.20s`;
- focused architecture ownership repairs: `2 passed in 2.29s`;
- full architecture contracts: `258 passed in 16.86s`;
- targeted Ruff, bytecode compilation, fixed-capacity AST audit, and
  whitespace checks: passed.

No root-model conversion was required because focused runtime equivalence
covers the mechanical four-call extraction. Commit and push this unit. Resume
with another non-store orchestration audit, keep the 128-phase bound fixed,
and never create, update, or reopen a pull request.

## Late reshape-layout composite characterization

The next selected unit is another non-store extraction. It owns the adjacent
late ExpandDims-compatible, Flatten-HW-compatible, and NHWC-collapse reshape
passes while preserving their current order and arguments. Their three result
locals have no consumers.

`tests/test_flatbuffer_direct_late_reshape_layout_orchestration.py` fixes the
current calls and outer boundaries. Its strict expected failure requires a
new `run_late_reshape_layout_cleanup(shared_model_ir_pass_context)` composite
and removal of the three old locals. The ordered tuple must remain outside
`ConversionSession.phase_results`, which stays fixed at 128/128.

The audit baseline also found a stale channel-shuffle structure assertion,
not a production regression. Commit `2107f972` updated it to inspect the
existing phase ID and nested pass owner, with `24 passed` plus Ruff, compile,
and whitespace checks.

The characterization gate completed with `93 passed, 1 xfailed in 1.09s`;
the sole xfail is the intentionally absent composite. Targeted Ruff, bytecode
compilation, and whitespace validation passed. Commit and push this checkpoint
first. Implementation must add a focused runtime test for call order, context
identity, layout arguments, and tuple order; then update only structural
boundary contracts, run sequential focused/core/result/architecture gates,
document, commit, and push. Never create, update, or reopen a pull request.

## Late reshape-layout composite implementation

`run_late_reshape_layout_cleanup` now executes the three characterized owners
in their original order and returns their mappings as an ordered tuple. The
first two receive `context.layout_state`; the private collapse owner remains
model-only. The lowerer keeps one `_late_reshape_layout_results` composite via
`shared_model_ir_pass_context`, outside the full phase store.

The three old result locals are gone. Compatibility wrappers remain intact.
Focused runtime coverage proves callback order, ModelIR/layout identity, exact
argument policy, and tuple ordering. Broader gates exposed only stale source-
representation assertions; no graph or numerical behavior failed.

Final sequential validation under core-only `uv`:

- focused reshape owners and affected boundaries: `95 passed in 1.11s`;
- terminal-layout and pass-efficiency contracts: `94 passed in 2.11s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- result and phase-result contracts: `196 passed in 9.07s`;
- full architecture contracts: `258 passed in 17.48s`;
- phase-store capacity contracts: `2 passed in 0.51s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

The store remains fixed at 128/128. Commit and push this implementation unit.
Resume with the next non-store late orchestration audit; do not create, update,
or reopen a pull request.

## Terminal affine/QKV/layout-shape implementation checkpoint

`passes/terminal_affine_qkv_layout_shape_orchestration.py` now implements the
characterized two-child boundary. The exact shared context is passed to both
children, the unchanged layout-Transpose option is passed only to terminal
QKV/activation/layout/shape, and both complete nested results are returned in
fixed order with their raw identities preserved.

The lowerer retains one observation-only
`_terminal_affine_qkv_layout_shape_results` value in place of the two child
locals. The optional late-binary-layout reconciliation branch remains the
predecessor; terminal Expand/Squeeze reconciliation and the progress callback
remain successors outside the owner. Child owners, wrappers, callbacks,
guards, graph behavior, public behavior, and TensorFlow isolation remain
unchanged. The characterized unconsumed-result inventory is now 55, and the
phase store remains exactly 128 IDs and 128 owners.

Sequential validation passed: focused 5, expanded affected 1143,
terminal-layout/efficiency 92, core 55, result contracts 196, phase store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff, bytecode
compilation, and whitespace checks pass. Runtime injection covers both layout
policies and proves order, context identity, option forwarding, complete nested
schemas, and raw-result identity. No production test or known issue is failing,
and no real-model conversion was repeated for this ownership-only extraction.

At resume, rerun the read-only inventory of the 55 remaining characterized
unconsumed lowerer results and select the next smallest source-adjacent,
semantically closed cluster whose children already have pass-module owners.
Characterize it before production changes, keep all tests sequential under
`uv`, and commit/push only at complete checkpoints. Do not create, update, or
reopen a pull request.

## Late attention-layout composite characterization

The next selected unit owns the adjacent late QKV reshape, attention-Gather
cleanup, axis-0 Gather reshape, and attention pre-projection rank-lift passes.
Their mappings are unconsumed, and their argument policy is
layout/model/layout/model.

`tests/test_flatbuffer_direct_late_attention_layout_orchestration.py` fixes
the current order, exact arguments, late channel-shuffle predecessor,
window-partition successor, and absence of consumers. Its strict expected
failure requires one
`run_late_attention_layout_cleanup(shared_model_ir_pass_context)` tuple outside
the full phase store and removal of the four old locals.

The characterization gate completed with
`214 passed, 1 xfailed in 1.18s`; Ruff, bytecode compilation, and whitespace
checks passed. Commit and push this checkpoint first. Implementation must add
runtime order/context/argument/tuple coverage, update only source-
representation contracts, run the sequential focused/core/result/architecture
gates, document, commit, and push. Keep the store at 128/128 and never create,
update, or reopen a pull request.

## Late attention-layout composite implementation

`run_late_attention_layout_cleanup` now invokes the four characterized owners
in their original order with the exact layout/model/layout/model argument
policy. It returns their mappings as an ordered tuple, retained by the lowerer
as `_late_attention_layout_results` through the shared pass context and outside
the full phase store.

The four old result locals are gone, while compatibility wrappers remain.
Focused runtime coverage proves call order, ModelIR/layout identity, model-only
callbacks, and tuple order. The only broad-gate failures were stale source-
representation boundaries; no graph or numerical behavior failed.

Final sequential validation under core-only `uv`:

- focused attention owners and affected boundaries: `238 passed in 1.35s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 2.15s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- result and phase-result contracts: `196 passed in 9.42s`;
- full architecture contracts: `258 passed in 17.67s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

The store remains fixed at 128/128. Commit and push this implementation unit.
Resume with the next non-store late orchestration audit; do not create, update,
or reopen a pull request.

## Late window-layout composite characterization

The next selected unit owns the adjacent late window-partition and
window-reverse repairs. Their mappings are unconsumed, and both receive the
same ModelIR and conversion-local layout state.

`tests/test_flatbuffer_direct_late_window_layout_orchestration.py` fixes their
order and arguments, the late attention-composite predecessor, final
shape/activation convergence successor, and absence of consumers. Its strict
expected failure requires
`run_late_window_layout_cleanup(shared_model_ir_pass_context)` and removal of
the two old locals, while keeping the ordered tuple outside the full store.

The gate completed with `103 passed, 1 xfailed in 0.84s`; Ruff, bytecode
compilation, and whitespace checks passed. Commit and push this checkpoint
first. Implementation must add runtime order/context/layout/tuple coverage,
update structural boundaries and ownership, run sequential gates, document,
commit, and push. Keep the store at 128/128 and never create, update, or reopen
a pull request.

## Late window-layout composite implementation

`run_late_window_layout_cleanup` now invokes the window-partition and
window-reverse owners in their original order with the same ModelIR and layout
state. It returns both mappings as an ordered tuple retained by the lowerer as
`_late_window_layout_results`, outside the full phase store.

The two old result locals are gone, while lowerer wrappers and optional
`graph_index` behavior remain intact. Focused runtime coverage proves call
order, ModelIR/layout identity, and tuple order. Broad failures were limited to
stale source-boundary assertions; no graph or numerical behavior failed.

Final sequential validation under core-only `uv`:

- focused window owners and affected boundaries: `110 passed in 3.11s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 2.06s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- result and phase-result contracts: `196 passed in 9.14s`;
- full architecture contracts: `258 passed in 19.39s`;
- phase-store capacity contracts: `2 passed in 0.51s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

The store remains fixed at 128/128. Commit and push this implementation unit.
Resume by auditing the next non-store late boundary; do not create, update, or
reopen a pull request.

## Final boundary-channel composite characterization

The next selected unit owns final boundary-input normalization, internal
channel-slice propagation, and the channel-slice Mul/Add bridge. The first
call receives layout and diagnostics; the latter two final calls intentionally
remain model-only, unlike the earlier terminal instances.

`tests/test_flatbuffer_direct_final_boundary_channel_layout_orchestration.py`
fixes the exact order and argument policy, final shape/activation predecessor,
slice/Concat recovery successor, and absence of consumers. Its strict expected
failure requires one
`run_final_boundary_channel_layout_cleanup(shared_model_ir_pass_context)`
tuple outside the full store and removal of the three old locals.

The gate completed with `77 passed, 1 xfailed in 2.20s`; Ruff, bytecode
compilation, and whitespace checks passed. Commit and push this checkpoint
first. Implementation must add runtime order/context/argument/tuple coverage,
update structural boundaries and ownership, run sequential gates, document,
commit, and push. Keep the store at 128/128 and never create, update, or reopen
a pull request.

## Final boundary-channel composite implementation

`run_final_boundary_channel_layout_cleanup` now invokes final boundary-input
normalization, internal channel-slice propagation, and the channel-slice
Mul/Add bridge in the original order. Only normalization receives layout and
diagnostics; the latter two remain model-only. The ordered tuple is retained
as `_final_boundary_channel_layout_results` outside the full store.

The three old final locals are gone, while the earlier terminal phase records
and layout-aware calls remain unchanged. Focused runtime coverage proves order,
context identity, exact argument policy, and tuple order. Broad failures were
only stale source-representation assertions; no graph or numerical behavior
failed.

Final sequential validation under core-only `uv`:

- focused boundary/channel owners and affected boundaries:
  `79 passed in 4.15s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.97s`;
- synthetic core runtime contracts: `55 passed in 1.11s`;
- result and phase-result contracts: `196 passed in 9.49s`;
- full architecture contracts: `258 passed in 19.29s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

The store remains fixed at 128/128. Commit and push this implementation unit.
Resume by auditing the next final non-store boundary; do not create, update, or
reopen a pull request.

## Terminal Concat-bridge composite characterization

The final Slice/pre-Concat pair was audited but left unchanged. The pre-Concat
composite implementation still resides in the lowerer, so a pass-module owner
and compatibility-wrapper checkpoint must precede any pair extraction; avoid
circular imports or callback injection.

The next selected unit instead owns six adjacent pass-module callbacks:
all-output ReLU/Split, ReLU/Split/Conv/Concat, mixed Split/Concat, Concat input
adaptation, Concat-unary-Conv, and Shape extract. Their argument policy is
layout/layout/layout/layout/(layout+diagnostics)/model-only.

`tests/test_flatbuffer_direct_terminal_concat_bridge_layout_orchestration.py`
fixes exact order and arguments, final pre-Concat predecessor, guarded
elementwise-fanout successor, and absence of consumers. Its strict expected
failure requires one
`run_terminal_concat_bridge_layout_cleanup(shared_model_ir_pass_context)`
tuple outside the full store and removal of the six old locals.

The gate completed with `15 passed, 1 xfailed in 0.86s`; Ruff, bytecode
compilation, and whitespace checks passed. Commit and push this checkpoint
first. Implementation must add runtime order/context/argument/tuple coverage,
update source-boundary and ownership contracts, run sequential gates,
document, commit, and push. Keep the store at 128/128 and never create, update,
or reopen a pull request.

## Terminal Concat-bridge composite implementation

`run_terminal_concat_bridge_layout_cleanup` now invokes the six characterized
pass-module owners in their original order. The first four receive layout
state, Concat-unary-Conv receives layout state and diagnostics, and Shape
extract remains model-only. The ordered tuple is retained as
`_terminal_concat_bridge_layout_results` outside the full store.

The six old result locals are gone. Existing compatibility wrappers, guards,
owner implementations, graph-index behavior, the retained pre-Concat result,
and the guarded elementwise-fanout successor are unchanged. Focused runtime
coverage proves order, shared context identity, exact argument policy, and
tuple order. Broad failures were limited to stale source-representation and
direct-call-count assertions; no graph or numerical behavior failed.

Final sequential validation under core-only `uv`:

- focused composite and affected result contracts: `17 passed in 1.48s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.99s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- result contracts: `196 passed in 9.40s`;
- full architecture contracts: `258 passed in 19.32s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

The store remains fixed at 128/128. Commit and push this implementation unit.
On resume, first characterize moving the lowerer-resident pre-Concat
implementation behind a pass-module compatibility wrapper; only then reassess
the deferred final Slice/pre-Concat pair. Continue with coherent commits and
pushes only; do not create, update, or reopen a pull request.

## Pre-Concat NHWC pass-module owner characterization

The prerequisite audit found that
`_optimize_transpose_pre_concat_nhwc_chains` still owns a three-stage composite
inside the lowerer: indexed NHWC Concat cleanup, quantized indexed cleanup, and
legacy fallback. The first two receive layout state and diagnostics; the last
is model-only. Only the named indexed and quantized detail keys plus the legacy
aggregate contribute to the returned one-counter mapping.

`tests/test_flatbuffer_direct_pre_concat_nhwc_owner.py` fixes stage order,
argument identity, aggregation and ignored-detail behavior, return schema, and
the public lowerer compatibility name. Its strict expected failure requires a
new `passes/pre_concat_nhwc_layout.py` owner and a one-return lowerer wrapper.
No production source changed.

Run the dedicated characterization, targeted Ruff, bytecode compilation, and
whitespace checks, then commit and push this test-first checkpoint. During
implementation, import the three existing owners directly into the new module,
preserve all four production uses and the legacy lowerer wrapper, and avoid
callback injection or lowerer imports. Keep the store fixed at 128/128 and
never create, update, or reopen a pull request.

The characterization gate completed with `2 passed, 1 xfailed in 0.57s`;
the sole xfail is the intentionally absent pass-module owner. Targeted Ruff,
bytecode compilation, and whitespace checks passed. Commit and push this
checkpoint before production changes.

## Pre-Concat NHWC pass-module owner implementation

`passes/pre_concat_nhwc_layout.py` now owns indexed NHWC Concat cleanup,
quantized indexed cleanup, legacy fallback, and their aggregate counter. It
preserves the characterized order and exact layout/diagnostics/model-only
argument policy. The recognized counter-key lists moved with the owner, so
unrelated details are still ignored.

The lowerer function `_optimize_transpose_pre_concat_nhwc_chains` is now a
one-return compatibility wrapper with its original signature. Its three direct
production uses and recovery callback are unchanged, and the legacy lowerer
wrapper remains available. The new module imports existing owners directly;
it does not import the lowerer or inject callbacks.

Final sequential validation under core-only `uv`:

- focused owner and compatibility contracts: `3 passed in 0.54s`;
- all NHWC Concat family runtime contracts: `285 passed in 1.33s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- result contracts: `196 passed in 8.99s`;
- full architecture contracts: `258 passed in 19.02s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No graph, numerical, diagnostics, public API, or artifact behavior changed.
Commit and push this implementation checkpoint. On resume, characterize the
now-safe final Slice/pre-Concat pair between final slice/Concat recovery and
the terminal Concat-bridge composite. Continue with coherent commits and
pushes only; never create, update, or reopen a pull request.

## Final Slice/pre-ConCat composite characterization

The pre-ConCat owner extraction removed the prior circular-dependency risk.
The selected pair is now two adjacent pass-module calls: model-only final
Slice/pre-post passthrough cleanup, then layout-and-diagnostics-aware final
pre-ConCat NHWC cleanup. Both mappings are unconsumed.

`tests/test_flatbuffer_direct_final_slice_pre_concat_layout_orchestration.py`
fixes adjacency, arguments, the final slice/Concat recovery predecessor, the
terminal Concat-bridge successor, and absence of consumers. Its strict
expected failure requires one
`run_final_slice_pre_concat_layout_cleanup(shared_model_ir_pass_context)`
call and one ordered `_final_slice_pre_concat_layout_results` tuple outside
the full store. No production source changed.

Run the dedicated characterization, targeted Ruff, bytecode compilation, and
whitespace checks, then commit and push before implementation. The production
owner must import both existing pass owners directly, preserve compatibility
wrappers and result order, keep the store at 128/128, and never create, update,
or reopen a pull request.

The characterization gate completed with `1 passed, 1 xfailed in 0.14s`;
the sole xfail is the intentionally absent composite owner. Targeted Ruff,
bytecode compilation, and whitespace checks passed. Commit and push this
checkpoint before production changes.

## Final Slice/pre-ConCat composite implementation

`run_final_slice_pre_concat_layout_cleanup` now owns the characterized pair.
It calls Slice/pre-post passthrough model-only, then pre-ConCat NHWC cleanup
with the shared layout state and diagnostics, returning both mappings in order.

The lowerer retains `_final_slice_pre_concat_layout_results` outside the full
store. The two old unconsumed locals are gone; both compatibility wrappers and
the surrounding final slice/Concat recovery and terminal Concat-bridge
composites are unchanged. Focused runtime coverage proves order, context
identity, exact argument policy, and tuple order.

Final sequential validation under core-only `uv`:

- focused composite and affected boundaries: `20 passed in 1.13s`;
- Slice/pre-post mutation contracts: `9 passed in 0.52s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.81s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- result contracts: `196 passed in 9.28s`;
- full architecture contracts: `258 passed in 18.28s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No graph, numerical, diagnostics, public API, or artifact behavior changed.
Commit and push this implementation checkpoint. Resume by auditing the next
non-store late/terminal source-order unit. Continue with coherent commits and
pushes only; never create, update, or reopen a pull request.

## Late Conv1D/decoder composite characterization

The next audit selected eight adjacent unconditional layout repairs between
late Swish passthrough and very-late Pad cleanup: three Conv1D unary variants,
Conv1D InstanceNorm/unary, tencoder merge, Conv1D BatchMatMul, decoder
deconvolution input, and terminal Squeeze/Mean. Each receives the same ModelIR
and layout state, each already has a pass-module owner, and all result mappings
are unconsumed.

`tests/test_flatbuffer_direct_late_conv1d_decoder_layout_orchestration.py`
fixes the exact order, argument policy, boundaries, and absence of consumers.
Its strict expected failure requires one
`run_late_conv1d_decoder_layout_cleanup(shared_model_ir_pass_context)` call
and one ordered `_late_conv1d_decoder_layout_results` tuple outside the full
store. No production source changed.

Run the dedicated characterization, targeted Ruff, bytecode compilation, and
whitespace checks, then commit and push before implementation. The owner must
import all eight existing callbacks directly, preserve compatibility wrappers
and tuple order, keep the store at 128/128, and never create, update, or reopen
a pull request.

The characterization gate completed with `1 passed, 1 xfailed in 0.17s`;
the sole xfail is the intentionally absent composite owner. Targeted Ruff,
bytecode compilation, and whitespace checks passed. Commit and push this
checkpoint before production changes.

## Late Conv1D/decoder composite implementation

`run_late_conv1d_decoder_layout_cleanup` now owns all eight characterized
callbacks. Each receives the same ModelIR and layout state, and the independent
counter mappings are returned in their original order.

The lowerer retains `_late_conv1d_decoder_layout_results` outside the full
store. The eight old unconsumed locals and long inline call block are gone;
compatibility wrappers, indexed owners, late Swish, and very-late Pad remain
unchanged. Focused runtime coverage proves callback order, context identity,
and tuple ordering.

Final sequential validation under core-only `uv`:

- focused composite and boundaries: `3 passed in 0.65s`;
- indexed Conv1D/decoder and affected result contracts:
  `431 passed in 2.01s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.93s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- result contracts: `196 passed in 9.27s`;
- full architecture contracts: `258 passed in 19.39s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No graph, numerical, diagnostics, public API, or artifact behavior changed.
Commit and push this implementation checkpoint. Resume by auditing the
adjacent very-late Pad/InstanceNorm unit before the singleton-reshape cluster.
Continue with coherent commits and pushes only; never create, update, or reopen
a pull request.

## Very-late Pad/InstanceNorm composite characterization

The selected unit is four adjacent unconditional calls after the late
Conv1D/decoder composite: Pad cleanup, InstanceNorm post-bias,
InstanceNorm residual-Mul/Concat, and InstanceNorm dual-stat residual. Pad
receives layout and diagnostics; the other three receive layout only. Every
mapping is unconsumed, and the singleton/consecutive-Reshape cluster is the
fixed successor.

`tests/test_flatbuffer_direct_very_late_pad_instancenorm_layout_orchestration.py`
fixes exact order, argument policy, boundaries, and absence of consumers. Its
strict expected failure requires one
`run_very_late_pad_instancenorm_layout_cleanup(shared_model_ir_pass_context)`
call and one ordered `_very_late_pad_instancenorm_layout_results` tuple outside
the full store. No production source changed.

Run the dedicated characterization, targeted Ruff, bytecode compilation, and
whitespace checks, then commit and push before implementation. Import all four
existing pass owners directly, preserve wrappers and tuple order, keep the
store at 128/128, and never create, update, or reopen a pull request.

The characterization gate completed with `1 passed, 1 xfailed in 0.15s`;
the sole xfail is the intentionally absent composite owner. Targeted Ruff,
bytecode compilation, and whitespace checks passed. Commit and push this
checkpoint before production changes.

## Very-late Pad/InstanceNorm composite implementation

The new pass-module owner calls Pad cleanup followed by the three
InstanceNorm layout repairs in the characterized order. Pad keeps its layout
and diagnostics arguments; every InstanceNorm owner keeps its layout-only
argument policy. All calls receive the same conversion-local ModelIR and
LayoutState, and their four mappings are returned as an ordered tuple.

The lowerer now has one
`_very_late_pad_instancenorm_layout_results` assignment outside the bounded
store instead of four unconsumed locals. The store remains exactly 128/128.
The late Conv1D/decoder and singleton/consecutive-Reshape boundaries,
compatibility wrappers, existing pass owners, and all graph behavior remain
unchanged.

Final sequential validation under core-only `uv`:

- focused composite and affected Pad/InstanceNorm boundaries:
  `424 passed in 2.53s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.83s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- result contracts: `196 passed in 9.25s`;
- full architecture contracts: `258 passed in 17.05s`;
- phase-store capacity contracts: `2 passed in 0.56s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No real-model conversion was required for this owner-only extraction. Commit
and push this checkpoint. At resume, audit the next coherent non-store unit
after the singleton/consecutive-Reshape cluster before changing production
behavior. Continue with coherent commits and pushes only; never create,
update, or reopen a pull request.

## Very-late layout/broadcast composite characterization

The next selected source-order unit is the optional final layout-Transpose
cleanup and unconditional rank-four channelwise broadcast-constant repair
between the singleton/consecutive-Reshape composite and the already-recorded
very-late broadcast reconciliation. Both result mappings are unconsumed.

`tests/test_flatbuffer_direct_very_late_layout_broadcast_orchestration.py`
fixes the original guard, call order, exact arguments, outer boundaries, and
absence of consumers. Its strict expected failure requires one
`run_very_late_layout_broadcast_cleanup(shared_model_ir_pass_context,
include_layout_transpose=optimize_layout_transpose_chains)` call and an ordered
`_very_late_layout_broadcast_results` tuple outside the full store. The
existing reconciliation record must stay unconditional and immediately
follow the composite.

The implementation owner must import both pass owners directly, preserve the
layout cleanup skip exactly, return `None` for the skipped optional result,
preserve the unconditional broadcast result, retain compatibility wrappers,
and keep the store exactly 128/128. The characterization gate completed with
`1 passed, 1 xfailed in 0.15s`; the sole xfail is the intentionally absent
owner. Targeted Ruff, bytecode compilation, and whitespace checks passed.

Commit and push this characterization before production changes. Never
create, update, or reopen a pull request.

## Very-late layout/broadcast composite implementation

The new pass-module owner preserves the normalized layout-option guard around
layout-Transpose cleanup, including its layout and diagnostics arguments. It
always follows with rank-four channelwise broadcast-constant repair. Disabled
layout cleanup returns `None`; enabled cleanup and broadcast repair retain
their original mappings in a fixed two-item tuple.

The lowerer replaces the two old unconsumed locals and inline guard with one
`_very_late_layout_broadcast_results` assignment outside the full store. The
singleton/consecutive-Reshape predecessor and unconditional very-late
broadcast shape-reconciliation record remain unchanged and adjacent. The
phase-result store remains exactly 128/128, and all compatibility wrappers
remain intact.

Final sequential validation under core-only `uv`:

- focused enabled/disabled owner contracts: `4 passed in 0.57s`;
- affected layout, broadcast, reconciliation, singleton, and store contracts:
  `97 passed in 2.82s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- result contracts: `196 passed in 9.20s`;
- full architecture contracts: `258 passed in 19.24s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No real-model conversion was required for this owner-only extraction. Commit
and push this checkpoint. At resume, audit the next coherent non-store unit
after the very-late broadcast reconciliation without changing production
behavior first. Continue with coherent commits and pushes only; never create,
update, or reopen a pull request.

## Shared-late reconciliation decision characterization

The next selected unit owns the cleanup decision immediately after the
very-late broadcast reconciliation. Four direct sanitizers, the two-result
indexed binary adapter owner, and the three-result singleton/consecutive-
Reshape owner produce nine ordered dictionaries. Their positive counters or a
decrease from the initial tensor count trigger the existing shared-late
reconciliation record.

`tests/test_flatbuffer_direct_shared_late_reconciliation_orchestration.py`
fixes every evidence position, the prune-delta check, owner order, phase ID,
direct reconciliation call, and late-binary successor. Its strict expected
failure requires one
`run_shared_late_reconciliation_cleanup(shared_model_ir_pass_context)` boolean
assignment followed by the same conditional record.

The owner must not record phase evidence or perform reconciliation itself.
That keeps `session.record_phase_result` in the lowerer with the same nested
`_reconcile_static_tensor_shapes` call and preserves invoked-phase-only store
semantics. It must import the six pass-module owners directly and retain their
exact argument policy. The characterization gate completed with
`1 passed, 1 xfailed in 0.13s`; the sole xfail is the intentionally absent
owner. Targeted Ruff, bytecode compilation, and whitespace checks passed.

Commit and push this characterization before production changes. Keep the
store at 128/128 and never create, update, or reopen a pull request.

## Shared-late reconciliation decision implementation

The new pass-module owner runs the four direct sanitizers, indexed binary
adapter pair, and singleton/consecutive-Reshape triple in their original
order. It preserves every argument contract and returns one boolean derived
from all nine positive-counter mappings plus the original prune-only tensor
count decrease.

The lowerer replaces nine evidence locals and one tensor-count snapshot with
`_shared_late_requires_reconciliation`. The following guard and nested
`_reconcile_static_tensor_shapes` phase record remain in the lowerer exactly
to preserve invoked-phase-only store semantics. The store remains 128/128.

Final sequential validation under core-only `uv`:

- focused stable, nine-evidence, prune-only, order, and boundary contracts:
  `13 passed in 0.55s`;
- affected shared-late, adapter, singleton, terminal, core, and store
  contracts: `150 passed in 3.13s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.89s`;
- synthetic core runtime contracts: `55 passed in 0.96s`;
- result contracts: `196 passed in 9.19s`;
- full architecture contracts: `258 passed in 18.45s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No real-model conversion was required because all ten decision triggers and
the lowerer integration branch are covered directly. Commit and push this
checkpoint. At resume, characterize the adjacent late-binary repair decision
before changing production code. Continue with coherent commits and pushes
only; never create, update, or reopen a pull request.

## Late-binary repair decision characterization

The next unit is the three-evidence plus prune-delta decision immediately
after shared-late reconciliation. Static-signature sanitization contributes
one named counter, indexed binary adapter cleanup contributes two, and a
tensor-count decrease captures cleanup-only pruning. Any of these conditions
invokes the existing `shape_reconciliation.primary.late_binary_repair`
record.

`tests/test_flatbuffer_direct_late_binary_repair_orchestration.py` fixes the
counter keys, call order, tensor snapshot, direct reconciliation owner,
shared-late predecessor, and optional late-binary recovery successor. Its
strict expected failure requires one
`run_late_binary_repair_cleanup(shared_model_ir_pass_context)` boolean
assignment followed by the unchanged conditional record.

The new owner must not absorb reconciliation or either guard of the following
late-binary layout recovery. It must import the two existing pass-module
owners directly, keep both model-only argument contracts, and preserve the
three specific counter-key checks plus prune delta. The characterization gate
completed with `1 passed, 1 xfailed in 0.14s`; the sole xfail is the absent
owner. Targeted Ruff, bytecode compilation, and whitespace checks passed.

Commit and push this characterization before production changes. Keep the
store at 128/128 and never create, update, or reopen a pull request.

## Late-binary repair decision implementation

The new `run_late_binary_repair_cleanup` pass owner preserves the original
static-signature sanitizer and indexed binary adapter order, model identity,
three exact counter keys, and tensor-pruning fallback. It returns one boolean
and does not perform reconciliation or record phase evidence.

The lowerer now consumes that boolean and retains the existing conditional
`shape_reconciliation.primary.late_binary_repair` record with its direct
static-shape reconciliation call. Both guards around the following optional
late-binary layout recovery are unchanged. Three evidence locals and one
tensor-count snapshot left the lowerer, all compatibility wrappers remain,
and the bounded store remains exactly 128/128.

Final sequential validation under core-only `uv`:

- focused owner contracts: `7 passed in 0.56s`;
- affected boundary contracts: `132 passed in 2.78s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.88s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.09s`;
- architecture contracts: `258 passed in 19.02s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No real-model conversion was required because the runtime matrix covers the
stable path, all three evidence triggers, prune-only cleanup, and the lowerer
integration branch. Commit and push this checkpoint. At resume, characterize
the adjacent optional late-binary layout-recovery decision before production
changes. Continue with coherent commits and pushes only; never create,
update, or reopen a pull request.

## Optional late-binary layout-recovery decision characterization

The next boundary has two control-flow layers: the normalized layout-option
predicate controls whether aggregate late-binary recovery runs, and a
positive-summary predicate controls the already-recorded static-shape
reconciliation. The direct reconciliation call, phase ID, preceding
late-binary repair decision, and following pre-terminal InstanceNorm cleanup
must remain fixed.

`tests/test_flatbuffer_direct_optional_late_binary_layout_recovery_orchestration.py`
captures the current boundary and strictly xfails until one focused boolean
owner replaces only the optional recovery call and its decision. The owner
must preserve disabled-path skipping, ModelIR/LayoutState/diagnostics identity,
the independent layout-Transpose flag, and all positive mutation values. It
must not reconcile shapes or write to the bounded store.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.16s`; targeted Ruff, bytecode compilation, and
whitespace checks passed. No production source changed.

Commit and push this characterization before implementation. Keep the store
at 128/128. On resume, add the owner, retain the direct conditional
reconciliation in the lowerer, update owner-aware contracts, and run the
affected gates sequentially. Continue with commits and pushes only; never
create, update, or reopen a pull request.

## Optional late-binary layout-recovery decision implementation

The new boolean owner preserves disabled-path skipping and forwards the exact
ModelIR/LayoutState/diagnostics objects plus the independent layout-Transpose
flag to the existing aggregate recovery owner. For enabled recovery it returns
whether any aggregate mutation count is positive; it neither reconciles shapes
nor records phase evidence.

The lowerer retains the direct conditional
`shape_reconciliation.primary.late_binary_layout_recovery` record. One consumed
aggregate-result local and the two nested decision branches are replaced by
`_late_binary_layout_recovery_requires_reconciliation`; the predecessor and
successor remain adjacent and the store remains 128/128.

Final sequential validation under core-only `uv`:

- focused owner contracts: `6 passed in 0.57s`;
- affected boundary contracts: `138 passed in 2.82s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.99s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.15s`;
- architecture contracts: `258 passed in 18.96s`;
- phase-store capacity contracts: `2 passed in 0.55s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No real-model conversion was required because the runtime matrix covers both
enablement states, stable and positive summaries, both layout-flag values,
context identity, and lowerer integration. Commit and push this checkpoint. At
resume, characterize the adjacent pre-terminal affine/InstanceNorm decision
boundary before production changes. Continue with commits and pushes only;
never create, update, or reopen a pull request.

## Pre-terminal InstanceNorm layout composite characterization

Three adjacent InstanceNorm layout owners run after the optional late-binary
recovery decision and before the first terminal-affine tensor-count snapshot.
They share one ModelIR/LayoutState argument policy and return three independent,
unconsumed mappings.

`tests/test_flatbuffer_direct_pre_terminal_instancenorm_layout_orchestration.py`
fixes their targets, order, argument identity, predecessor, successor, and
absence of result consumers. Its strict xfail requires one ordered composite
owner outside the full store. The owner may absorb only these three calls and
must preserve each mapping without aggregation. No production source changed.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.14s`; targeted Ruff, bytecode compilation, and
whitespace checks passed.

Commit and push this characterization before implementation. Keep the store
at 128/128. On resume, add the direct pass-module owner, update owner-aware
InstanceNorm/terminal-affine contracts, validate sequentially, then document,
commit, and push. Never create, update, or reopen a pull request.

## Pre-terminal InstanceNorm layout composite implementation

The new composite owner calls the existing post-bias,
residual-Mul/Concat, and dual-stat InstanceNorm pass owners in source order. It
forwards one shared ModelIR/LayoutState context and returns the three original
mappings without aggregation.

The lowerer now retains one ordered composite result instead of three
unconsumed locals. The preceding optional recovery guard and following
terminal-affine tensor snapshot remain adjacent; compatibility wrappers and
the 128/128 phase store remain unchanged.

Final sequential validation under core-only `uv`:

- focused owner contracts: `3 passed in 0.59s`;
- affected boundary contracts: `151 passed in 3.50s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.88s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.13s`;
- architecture contracts: `258 passed in 19.45s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No real-model conversion was required because focused runtime and owner-aware
architecture tests prove exact result order, object identity, and unchanged
total production call counts. Commit and push this checkpoint. At resume,
characterize the first terminal-affine recovery evidence boundary before
production changes. Continue with commits and pushes only; never create,
update, or reopen a pull request.

## Terminal-affine prune-aware summary characterization

The late pipeline repeats two identical terminal-affine evidence triples:
tensor-count snapshot, raw eleven-result recovery, and prune-aware normalized
summary. The six intermediate locals are confined to those triples; the two
summary mappings are unconsumed. The existing nested raw wrapper is retained
as a compatibility boundary.

`tests/test_flatbuffer_direct_terminal_affine_recovery_summary_orchestration.py`
fixes both triples, raw wrapper dispatch, exact prune expression, source order,
and all four neighboring boundaries. Its strict xfail requires one pass-module
summary owner used at both sites while keeping the raw wrapper defined. No
production source changed.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.13s`; targeted Ruff, bytecode compilation, and
whitespace checks passed.

Commit and push before implementation. Keep the store at 128/128. On resume,
add the prune-aware owner, replace only the two evidence triples, update
owner-aware terminal-affine contracts, validate sequentially, then document,
commit, and push. Never create, update, or reopen a pull request.

## Terminal-affine prune-aware summary implementation

The pass-module summary owner now owns tensor-count capture, raw eleven-pass
recovery, and the existing strict prune-aware normalization. The two lowerer
sites retain their original summary targets but no longer retain four consumed
intermediate locals or duplicate the prune expression.

The raw lowerer wrapper remains defined and dispatches exactly as before. Both
predecessors, both successors, all raw result schemas, and the 128/128 store
remain unchanged.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `4 passed in 0.59s`;
- affected boundary contracts: `182 passed in 2.14s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 2.02s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.17s`;
- architecture contracts: `258 passed in 18.81s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No real-model conversion was required because focused tests cover stable and
prune-only summary paths while the existing raw-owner matrix covers every pass
result. Commit and push this checkpoint. At resume, characterize the adjacent
pre-terminal pre-add prune-aware evidence boundary before production changes.
Continue with commits and pushes only; never create, update, or reopen a pull
request.

## Pre-terminal pre-add prune-evidence characterization

The adjacent boundary is now characterized without a production change. The
lowerer still snapshots `len(model_ir.tensors)`, calls the existing pre-add
NHWC-chain cleanup with `model_ir` and `session.layout_state`, and adds the
exact non-negative tensor-count delta to its otherwise unchanged result
mapping. The first terminal-affine summary remains the predecessor and the
channel Slice/Pad/Mul cluster remains the successor.

`tests/test_flatbuffer_direct_pre_terminal_pre_add_orchestration.py` contains
one passing current-boundary contract and one strict expected failure for the
future pass-module owner. Sequential core-only validation completed with
`1 passed, 1 xfailed in 0.13s`; targeted Ruff, bytecode compilation, and
whitespace checks also passed. The xfail is intentional and represents the
next production unit rather than a regression.

Commit and push this test-and-documentation checkpoint. At resume, implement
only `run_pre_terminal_pre_add_cleanup(context)`, preserve the lowerer
compatibility wrapper, update owner-aware neighboring contracts, and run the
affected gates sequentially. Keep the phase-result store at exactly 128/128.
Continue with commits and pushes only; never create, update, or reopen a pull
request.

## Pre-terminal pre-add prune-evidence implementation

The characterized owner is implemented in
`passes/pre_terminal_pre_add_orchestration.py`. It captures the initial tensor
count, calls the existing pre-add NHWC cleanup once with the shared
ModelIR/LayoutState context, and returns the original mapping with the same
non-negative `pruned_unused_tensors` delta. The lowerer now uses one owner call
at the same result target and source location; its compatibility wrapper and
all other call sites remain intact.

The old lowerer-local tensor-count snapshot is removed. The first
terminal-affine summary remains the predecessor, the channel Slice/Pad/Mul
cluster remains the successor, and `_pre_terminal_pre_add_stats` remains
outside the full 128/128 phase-result store. No graph, pass, layout,
diagnostics, API, artifact, dependency, or TensorFlow behavior changed.

Final sequential validation under core-only `uv`:

- focused owner contracts: `4 passed in 0.58s`;
- affected boundary contracts: `186 passed in 2.26s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.98s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.37s`;
- architecture contracts: `258 passed in 19.63s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 10.20s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. On resume, inspect the next
adjacent non-store evidence boundary and characterize it before production
changes. Continue with coherent commits and pushes only; never create, update,
or reopen a pull request.

## Channel Slice/Pad/Mul direct-summary characterization

The direct late site currently assigns the raw two-result cluster tuple and
immediately feeds it to `summarize_channel_slice_pad_mul_mutations`. The tuple
has no other consumer, and the normalized summary is unconsumed. The nested
raw wrapper remains a required callback for terminal Slice/Concat recovery and
must not be removed.

The new characterization fixes that exact boundary and adds a strict xfail
requiring a pass-module `run_channel_slice_pad_mul_summary(context)` owner at
the direct site while preserving the raw wrapper. Sequential core-only
validation completed with `1 passed, 1 xfailed in 0.14s`; targeted Ruff,
bytecode compilation, and whitespace checks passed.

Commit and push this characterization separately. At resume, add the direct
summary owner, replace only the two direct evidence statements, update
owner-aware neighboring contracts, and retain the raw callback wrapper.
Continue with commits and pushes only; never create, update, or reopen a pull
request.

## Channel Slice/Pad/Mul direct-summary implementation

The pass module now exposes `run_channel_slice_pad_mul_summary(context)`, which
runs the existing raw pair and feeds its tuple to the existing strict
normalizer. The direct lowerer site retains its summary target but no longer
keeps the consumed raw-result local. The nested raw wrapper remains intact for
terminal recovery callback composition.

The direct summary receives the same `channel_slice_pad_mul_context`; raw pass
order, shared state scope, diagnostics/layout identity, normalized keys,
predecessor, successor, and total invocation count are unchanged. The summary
stays outside the 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `3 passed in 0.57s`;
- affected boundary and callback contracts: `195 passed in 2.42s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.86s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.18s`;
- architecture contracts: `258 passed in 18.46s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 10.31s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Pre-terminal affine-tail composite characterization

The two calls immediately after the channel Slice/Pad/Mul summary are now
characterized as one possible ordered boundary: affine post-Add cleanup with
ModelIR/LayoutState, then strict StridedSlice/Pad/Concat cleanup with ModelIR
only. Their mappings are unconsumed, and the following terminal-affine summary
is unchanged.

The focused contract passes for the current two-call representation and has
one strict xfail for a future
`run_pre_terminal_affine_tail_cleanup(shared_model_ir_pass_context)` owner.
Sequential validation completed with `1 passed, 1 xfailed in 0.14s`; targeted
Ruff, bytecode compilation, and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the ordered pair, retain both lowerer compatibility wrappers and all other
call sites, update owner-aware neighboring contracts, and validate
sequentially. Continue with commits and pushes only; never create, update, or
reopen a pull request.

## Pre-terminal affine-tail composite implementation

The new pass-module owner runs affine post-Add cleanup with
ModelIR/LayoutState and then strict StridedSlice/Pad/Concat cleanup with
ModelIR only. It returns both mappings in source order. The lowerer now keeps
one ordered composite target instead of the two old unconsumed targets.

The channel Slice/Pad/Mul summary predecessor, terminal-affine summary
successor, lowerer wrappers, other direct call sites, and declared total call
counts remain unchanged. The composite remains outside the 128/128 phase
store.

Final sequential validation under core-only `uv`:

- focused owner contracts: `3 passed in 0.56s`;
- affected boundary and call-count contracts: `237 passed in 3.14s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.88s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.23s`;
- architecture contracts: `258 passed in 18.98s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.81s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Late SPP/Concat/Unary direct-summary characterization

The direct late SPP pair currently assigns its raw two-result tuple and
immediately normalizes it into `_late_spp_stats`. The tuple has no other
consumer and the summary is unconsumed. The existing raw wrapper remains a
required compatibility boundary.

The focused contract fixes this representation and adds one strict xfail for
`run_late_spp_concat_unary_conv_summary(context)`. Sequential validation
completed with `1 passed, 1 xfailed in 0.14s`; targeted Ruff, bytecode
compilation, and whitespace checks passed.

Commit and push this characterization separately. At resume, add only the
direct summary owner, preserve the raw wrapper, update owner-aware neighboring
contracts, and validate sequentially. Continue with commits and pushes only;
never create, update, or reopen a pull request.

## Late SPP/Concat/Unary direct-summary implementation

The pass module now exposes
`run_late_spp_concat_unary_conv_summary(context)`. It runs the existing raw
pair and normalizes its tuple through the existing strict two-key summary. The
lowerer retains `_late_spp_stats` but removes the consumed raw-result local.

The nested raw wrapper, shared pass scope, terminal Slice/Pad/Concat
predecessor, pre-QKV shape-extract successor, raw result schema, and total
owner count remain unchanged. The summary stays outside the 128/128 store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `3 passed in 0.60s`;
- affected boundary and owner contracts: `187 passed in 1.97s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.91s`;
- synthetic core runtime contracts: `55 passed in 0.98s`;
- result contracts: `196 passed in 9.42s`;
- architecture contracts: `258 passed in 18.89s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.86s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Late QKV prune-aware summary characterization

The late QKV site is now fixed as one evidence triple: tensor-count snapshot,
flagged raw QKV owner with prefix disabled, and strict prune-aware summary. The
two consumed intermediate values are local to this triple, while the summary
is unconsumed. Other default-policy raw-wrapper uses remain outside the scope.

The focused contract passes for the current representation and has one strict
xfail for `run_qkv_attention_summary(context, flags...)`. Sequential
validation completed with `1 passed, 1 xfailed in 0.17s`; targeted Ruff,
bytecode compilation, and whitespace checks passed.

Commit and push this characterization separately. At resume, add the generic
prune-aware owner, replace only this late triple, retain the raw lowerer
wrapper and both default uses, and validate sequentially. Continue with commits
and pushes only; never create, update, or reopen a pull request.

## Late QKV prune-aware summary implementation

The pass module now exposes `run_qkv_attention_summary(...)`. It snapshots
tensor count, invokes the existing raw QKV owner with the supplied policy
flags, and applies the existing strict prune-aware summary. The late lowerer
site keeps `_late_qkv_stats` but no longer exposes the consumed tensor-count
and raw-result locals.

The raw lowerer wrapper and both default-policy uses remain intact. The late
site still receives the runtime layout-Transpose flag and disables prefix
cleanup. Its shape-extract predecessor, terminal Split/Conv/Concat successor,
pass order, shared context, graph behavior, result schema, and 128/128 store
are unchanged.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `5 passed in 0.59s`;
- affected owner, boundary, core, store, and architecture contracts:
  `405 passed in 20.77s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.90s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.37s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.69s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Terminal HardSwish/SE prune-aware summary characterization

The selected next unit is the two-statement terminal HardSwish/SE evidence
boundary immediately after the terminal QKV Split/Conv/Concat bridge. It
snapshots tensor count, calls the existing raw owner, and adds the non-negative
prune delta to the unchanged raw mapping. The summary remains unconsumed and
the following late hard-activation triple is outside this unit.

The focused characterization fixes the current representation, exact prune
expression, predecessor and successor, raw wrapper retention, and absence of a
summary consumer. One strict xfail requires
`run_hardswish_se_layout_summary(model_ir)` in the pass module.

Sequential validation under core-only `uv` completed with
`76 passed, 1 xfailed in 1.22s` across the new characterization and related
HardSwish/SE, late hard-activation, indexed bridge, and store contracts. The
sole expected failure is the unimplemented summary owner.

Commit and push this characterization separately. At resume, implement only
the prune-aware owner and late direct site, retain the raw lowerer wrapper and
the earlier phase-store raw call, update owner-aware structural contracts, and
validate sequentially. Continue with commits and pushes only; never create,
update, or reopen a pull request.

## Terminal HardSwish/SE prune-aware summary implementation

The pass module now exposes `run_hardswish_se_layout_summary(model_ir)`. It
captures tensor count, invokes the existing raw HardSwish/SE owner, and returns
the raw one-key mapping plus `pruned_unused_tensors`. The late lowerer site
keeps `_terminal_hardswish_se_stats` but removes its local count and inline
mapping composition.

The raw lowerer wrapper and earlier phase-store call remain intact. The
terminal QKV bridge predecessor, late hard-activation successor, raw pruning,
graph behavior, mapping schema, and 128/128 store are unchanged.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `4 passed in 0.57s`;
- affected owner, boundary, store, and architecture contracts:
  `337 passed in 19.54s`;
- related HardSwish/SE and late recovery contracts: `87 passed in 1.42s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.93s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.13s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.73s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Late hard-activation prune-aware summary characterization

The next selected unit is the three-statement late hard-activation evidence
boundary. It snapshots tensor count, runs the existing raw ordered owner with
the runtime layout-Transpose flag, and applies the existing strict prune-aware
normalizer. The summary is unconsumed.

The focused contract fixes the current representation, exact flag and prune
expression, raw wrapper retention, terminal HardSwish/SE predecessor, and
absolute-final pre-ConCat successor. One strict xfail requires
`run_late_hard_activation_layout_summary(context, flags...)`.

Sequential validation under core-only `uv` completed with
`22 passed, 1 xfailed in 1.15s` across the new characterization and related
late hard-activation, HardSwish/SE, pre-ConCat result, and store contracts. The
sole expected failure is the unimplemented summary owner.

Commit and push this characterization separately. At resume, implement only
the prune-aware owner and direct late site, retain the raw lowerer wrapper,
update owner-aware structural contracts, and validate sequentially. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Late hard-activation prune-aware summary implementation

The pass module now exposes
`run_late_hard_activation_layout_summary(context, flags...)`. It captures
tensor count, invokes the existing raw ordered owner, and applies the strict
normalizer with the same runtime layout-Transpose policy. The lowerer keeps
`_late_hard_activation_stats` but removes its consumed count and raw-result
locals.

The raw lowerer wrapper remains intact. Shared context and pass-state identity,
hard-activation option policy, pruning, summary schema, terminal HardSwish/SE
predecessor, absolute-final pre-ConCat successor, and 128/128 store are
unchanged.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `5 passed in 0.56s`;
- affected owner, boundary, store, and architecture contracts:
  `294 passed in 19.48s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.92s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.40s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.80s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Late layout-cluster prune-aware summary characterization

The next selected unit is the three-statement late
layout/Mean/SPP/Gather/constant-fold/Cast evidence boundary. It snapshots tensor
count, runs the existing ordered owner with the runtime layout-Transpose flag,
and applies the existing strict prune-aware normalizer. The summary is
unconsumed.

The focused contract fixes the current representation, exact flag and prune
expression, raw wrapper retention, shape-extract predecessor, and terminal
Expand/Squeeze successor. One strict xfail requires
`run_late_layout_mean_spp_gather_constant_cast_summary(context, flags...)`.

Sequential validation under core-only `uv` completed with
`21 passed, 1 xfailed in 0.98s` across the new characterization and related
late-layout, shape-extract result, and store contracts. The sole expected
failure is the unimplemented summary owner.

Commit and push this characterization separately. At resume, implement only
the prune-aware owner and direct late site, retain the raw lowerer wrapper,
update owner-aware structural contracts, and validate sequentially. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Late layout-cluster prune-aware summary implementation

The pass module now exposes
`run_late_layout_mean_spp_gather_constant_cast_summary(context, flags...)`. It
captures tensor count, invokes the existing ordered owner, and applies the
strict normalizer with the same runtime layout-Transpose policy. The lowerer
keeps `_late_layout_cluster_stats` but removes its consumed count and raw-result
locals.

The raw lowerer wrapper remains intact. Shared context and pass-state identity,
child constant-fold/Cast orchestration, pruning, summary schema, shape-extract
predecessor, terminal Expand/Squeeze successor, and 128/128 store are
unchanged.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `5 passed in 0.55s`;
- affected owner, boundary, store, and architecture contracts:
  `283 passed in 20.14s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.33s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.57s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Very-late normalization prune-aware summary characterization

The next selected unit is the three-statement very-late
Gather/constant-fold/Cast/normalization evidence boundary. It snapshots tensor
count, runs the existing four-result ordered owner, and applies the existing
strict prune-aware normalizer. The summary is unconsumed.

The focused contract fixes the current representation, exact prune expression,
raw wrapper retention, affine post-Add predecessor, and dynamic-Reshape
successor. One strict xfail requires
`run_very_late_gather_constant_normalization_summary(context)`.

Sequential validation under core-only `uv` completed with
`40 passed, 1 xfailed in 1.34s` across the new characterization and related
very-late normalization, absolute-final normalization/attention, late input
repair, and store contracts. The sole expected failure is the unimplemented
summary owner.

Commit and push this characterization separately. At resume, implement only
the prune-aware owner and direct late site, retain the raw lowerer wrapper,
update owner-aware structural contracts, and validate sequentially. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Very-late normalization prune-aware summary implementation

The pass module now exposes
`run_very_late_gather_constant_normalization_summary(context)`. It captures
tensor count, invokes the existing four-result ordered owner, and applies the
strict normalizer. The lowerer keeps `_very_late_normalization_stats` but
removes its consumed count and raw-result locals.

The raw lowerer wrapper remains intact. Shared context and pass-state identity,
child constant-fold/Cast orchestration, normalization policy, pruning, summary
schema, affine predecessor, dynamic-Reshape successor, later repair order, and
128/128 store are unchanged.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `4 passed in 0.60s`;
- affected owner, boundary, store, and architecture contracts:
  `301 passed in 19.52s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.82s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.03s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.68s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next adjacent non-store evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Indexed Conv-input prune-aware summary family characterization

The next selected family is the repeated indexed Conv-input count-plus-mapping
boundary at the very-late primary and fallback sites. Both use the same
two-repair indexed owner and two-key schema. The final primary site uses only
the stale-Transpose repair and remains outside this family.

The focused contract fixes both current representations, model arguments,
prune expressions, predecessors, very-late successor, fallback conditional
reconciliation, and raw wrapper retention. One strict xfail requires
`run_indexed_conv_input_adapter_repairs_summary(model_ir)`.

Sequential validation under core-only `uv` completed with
`115 passed, 1 xfailed in 2.65s` across the new characterization and related
indexed Conv-input, very-late normalization, fallback, terminal-layout, and
store contracts. The sole expected failure is the unimplemented shared summary
owner.

Commit and push this characterization separately. At resume, implement only
the shared owner and the two compatible sites, retain the raw lowerer wrapper,
leave the final one-key site unchanged, update owner-aware contracts, and
validate sequentially. Continue with commits and pushes only; never create,
update, or reopen a pull request.

## Indexed Conv-input prune-aware summary family implementation

The pass module now exposes
`run_indexed_conv_input_adapter_repairs_summary(model_ir)`. It captures tensor
count, invokes the existing one-index two-repair owner, and returns the exact
raw mapping plus prune evidence. The very-late and fallback stats targets
remain, while their count locals and inline mapping extensions are removed.

The raw lowerer wrapper remains intact. The fallback reconciliation guard,
very-late neighboring repairs, indexed graph behavior, two-key schema, final
one-key site, and 128/128 store are unchanged.

Final sequential validation under core-only `uv`:

- focused shared-summary contracts: `4 passed in 0.57s`;
- affected family, boundary, store, and architecture contracts:
  `376 passed in 21.20s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.81s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 8.93s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.58s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next compatible repeated evidence family before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Stale channelwise-binary adapter summary family characterization

The next selected family is the repeated raw-repair-plus-prune mapping at the
fallback and final-primary stale channelwise-binary adapter sites. Both use the
same raw repair and exact one-key schema. The indexed binary convergence loop
is explicitly excluded because it owns a shared graph index and iterative
broadcast, adapter, and shape reconciliation semantics.

The focused contract fixes both current representations, model arguments,
prune expressions, preceding concat-axis guards, following conditional
reconciliation, fallback topology successor, final progress successor, and raw
wrapper retention. One strict xfail requires
`run_stale_binary_adapter_repair_summary(model_ir)`.

Sequential validation under core-only `uv` completed with
`93 passed, 1 xfailed in 2.30s` across the new characterization and related
fallback, terminal-layout, indexed binary-convergence, and phase-store
contracts. The sole expected failure is the unimplemented shared summary
owner. Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the shared owner and the two compatible sites, retain the raw lowerer wrapper,
leave indexed convergence unchanged, update owner-aware structural contracts,
and validate sequentially. Continue with commits and pushes only; never create,
update, or reopen a pull request.

## Stale channelwise-binary adapter summary family implementation

The pass module now exposes
`run_stale_binary_adapter_repair_summary(model_ir)`. It captures tensor count,
invokes the existing raw stale channelwise-binary adapter repair once, and
returns the exact raw mapping plus prune evidence. The fallback and
final-primary stats targets remain, while both count locals and inline mapping
extensions are removed.

The raw lowerer wrapper and its optional graph-index forwarding remain intact.
The indexed convergence loop, mutation-positive reconciliation guards,
phase-result IDs, fallback topology successor, final progress successor, and
128/128 store are unchanged.

Final sequential validation under core-only `uv`:

- focused shared-summary contracts: `4 passed in 0.57s`;
- affected family, boundary, store, and architecture contracts:
  `354 passed in 20.61s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.80s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.11s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.60s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next compatible repeated evidence family before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Final stale Conv-input summary characterization

The next selected boundary is the final-primary one-repair stale Conv-input
count-plus-mapping site that was deliberately excluded from the indexed
two-repair family. Its raw one-key schema and optional graph-index wrapper are
distinct and remain independently owned.

The focused contract fixes the current representation, model argument, prune
expression, preceding final-Pad guard, following conditional reconciliation,
mixed-Concat successor, and raw wrapper retention. One strict xfail requires
`run_stale_conv_input_adapter_repair_summary(model_ir)`.

Sequential validation under core-only `uv` completed with
`336 passed, 1 xfailed in 18.50s` across the new characterization and related
indexed Conv-input, terminal-layout, phase-store, and architecture contracts.
The sole expected failure is the unimplemented dedicated summary owner.
Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the dedicated one-repair summary owner and final-primary site, retain the raw
lowerer wrapper, leave both indexed sites unchanged, update owner-aware
structural contracts, and validate sequentially. Continue with commits and
pushes only; never create, update, or reopen a pull request.

## Final stale Conv-input summary implementation

The pass module now exposes
`run_stale_conv_input_adapter_repair_summary(model_ir)`. It captures tensor
count, invokes the existing raw stale Conv-input Transpose repair once, and
returns the exact raw mapping plus prune evidence. The final-primary stats
target remains, while its count local and inline mapping extension are removed.

The raw lowerer wrapper and optional graph-index forwarding remain intact. The
two indexed sites, indexed two-repair owner, final-Pad predecessor,
mutation-positive reconciliation guard, mixed-Concat successor, and 128/128
store are unchanged.

Final sequential validation under core-only `uv`:

- focused dedicated-summary contracts: `4 passed in 0.56s`;
- affected family, boundary, store, and architecture contracts:
  `339 passed in 20.54s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.17s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.65s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next compatible repeated evidence family before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Final PRELU prune-aware summary characterization

The next selected boundary is the absolute-final PRELU passthrough repair. Its
raw owner prunes on every invocation, while the lowerer currently owns the
tensor snapshot and rewrite-or-prune reconciliation condition.

The focused contract fixes the current representation, ModelIR/LayoutState
arguments, preceding SE-FC/Gather guard, following consecutive-Reshape cleanup,
raw wrapper retention, and exact guard semantics. One strict xfail requires
`run_prelu_transpose_passthrough_summary(model_ir, layout_state=...)` and the
existing generic positive-count predicate.

Sequential validation under core-only `uv` completed with
`389 passed, 1 xfailed in 19.45s` across the new characterization and related
terminal-layout, SE-FC/Gather, core runtime, phase-store, and architecture
contracts. The sole expected failure is the unimplemented dedicated summary
owner. Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the dedicated PRELU summary owner and final-primary site, retain the raw
lowerer wrapper and all other PRELU callers, update owner-aware structural
contracts, and validate sequentially. Continue with commits and pushes only;
never create, update, or reopen a pull request.

## Final PRELU prune-aware summary implementation

The pass module now exposes
`run_prelu_transpose_passthrough_summary(model_ir, layout_state=...)`. It
captures tensor count, invokes the existing raw PRELU owner once with the exact
layout state, and returns the raw mapping plus prune evidence. The final stats
target remains, while its count local is removed and the existing generic
positive-count predicate now owns the rewrite-or-prune decision.

The raw lowerer wrapper and both other PRELU production paths remain intact.
The SE-FC/Gather predecessor, consecutive-Reshape successor, reconciliation
phase ID, and 128/128 store are unchanged.

Final sequential validation under core-only `uv`:

- focused dedicated-summary contracts: `4 passed in 0.56s`;
- affected boundary, runtime, store, and architecture contracts:
  `392 passed in 21.86s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.37s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.68s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next compatible repeated evidence family before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Fallback norm-subgraph Pad summary characterization

The next selected boundary is the safety-fallback norm-only Pad layout cleanup.
It uses the shared Pad owner with `include_pad=False`, `include_unary=False`,
`include_norm=True`, no layout state, and conversion diagnostics, then records
prune-only evidence beside the raw schema.

The focused contract fixes the current representation, fixed flags and
arguments, recursive fallback predecessor, conditional norm reconciliation,
dynamic rank-one successor, and distinct ownership from all other Pad callers.
One strict xfail requires
`run_norm_subgraph_pad_layout_summary(model_ir, diagnostics=...)`.

Sequential validation under core-only `uv` completed with
`300 passed, 1 xfailed in 18.26s` across the new characterization and related
fallback, Pad result/orchestration, norm reconciliation, singleton-Reshape,
phase-store, and architecture contracts. The sole expected failure is the
unimplemented dedicated summary owner. Targeted Ruff and whitespace checks
passed.

Commit and push this characterization separately. At resume, implement only
the dedicated norm-subgraph Pad summary owner and fallback site, leave all
other Pad callers unchanged, update owner-aware structural contracts, and
validate sequentially. Continue with commits and pushes only; never create,
update, or reopen a pull request.

## Fallback norm-subgraph Pad summary implementation

The Pad pass module now exposes
`run_norm_subgraph_pad_layout_summary(model_ir, diagnostics=...)`. It captures
tensor count, invokes the existing Pad owner once with Pad/unary disabled and
norm enabled, forwards diagnostics, and returns the raw mapping plus prune
evidence. The fallback stats target remains, while its count local and inline
mapping extension are removed.

The raw Pad runner remains a lowerer compatibility re-export, all other Pad
routes remain unchanged, and owner-aware architecture coverage preserves the
total ordered-runner count. The fallback guard, reconciliation phase ID,
dynamic rank-one successor, and 128/128 store are unchanged.

Final sequential validation under core-only `uv`:

- focused dedicated-summary contracts: `4 passed in 0.57s`;
- affected boundary, Pad, norm, store, and architecture contracts:
  `303 passed in 19.72s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.82s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.17s`;
- phase-store capacity contracts: `2 passed in 0.55s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.98s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next compatible repeated evidence family before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Final placeholder binary-adapter summary characterization

The next selected boundary is the indexed exact/singleton binary-adapter pair
inside final placeholder-MatMul recovery. The pair's counter keys are disjoint;
the current lowerer separately owns its tensor snapshot and merges pair evidence
with the preceding placeholder reconciliation only in the following guard.

The focused contract fixes count and pair representation, pair order, model
argument, preceding mapping, rewrite-or-prune condition, topology successor,
and continued raw pair ownership for all other callers. One strict xfail
requires `run_indexed_binary_layout_adapter_summary(model_ir)`.

Sequential validation under core-only `uv` completed with
`380 passed, 1 xfailed in 21.00s` across the new characterization and related
indexed binary-adapter, terminal-layout, core runtime, phase-store, and
architecture contracts. The sole expected failure is the unimplemented merged
summary owner. Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the merged indexed binary-adapter summary and final-placeholder site, retain
the raw pair owner and all other pair callers, update owner-aware structural
contracts, and validate sequentially. Continue with commits and pushes only;
never create, update, or reopen a pull request.

## Final placeholder binary-adapter summary implementation

The binary-adapter pass module now exposes
`run_indexed_binary_layout_adapter_summary(model_ir, graph_index=...,
layout_state=...)`. It captures tensor count, invokes the existing indexed pair
once, merges both disjoint counter mappings, and adds prune evidence. The final
placeholder site replaces its count and two raw-result locals with one summary
mapping while retaining the preceding placeholder reconciliation mapping.

The raw pair owner and its shared-late, late-binary, and fallback callers remain
unchanged. The rewrite-or-prune guard, topology successor, and 128/128 store are
unchanged.

Final sequential validation under core-only `uv`:

- focused merged-summary contracts: `4 passed in 0.55s`;
- affected boundary, indexed adapter, runtime, store, and architecture
  contracts: `383 passed in 19.66s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.83s`;
- synthetic core runtime contracts: `55 passed in 0.95s`;
- result contracts: `196 passed in 9.25s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.92s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, characterize the
next compatible repeated evidence family before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## SiNet SE-FC/Gather shared-summary characterization

The next selected family is the duplicated fallback and absolute-final
SiNet/SE-FC/Gather cleanup. Each site currently owns a tensor snapshot, one
SiNet shuffle-tail result, the ordered SE-FC/Gather pair results, and a
rewrite-or-prune reconciliation condition. The ModelIR and layout-state
arguments remain path-specific.

The focused contract fixes the two current sequences, exact arguments,
predecessor/successor boundaries, reconciliation semantics, and retained raw
pair helper. One strict xfail requires a pass-module
`run_sinet_se_fc_gather_summary(context)` owner and one merged stats mapping at
each lowerer site.

Sequential validation under core-only `uv` completed with
`407 passed, 1 xfailed in 19.96s` across the new characterization and related
SE-FC/Gather, fallback, terminal-layout, core runtime, phase-store, and
architecture contracts. The sole expected failure is the unimplemented shared
summary owner. Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the shared owner and these two compatible sites, preserve the raw pair helper
and every other caller, update owner-aware structural/runtime contracts, and
validate sequentially. Continue with commits and pushes only; never create,
update, or reopen a pull request.

## SiNet SE-FC/Gather shared-summary implementation

The SE-FC/Gather pass module now exposes
`run_sinet_se_fc_gather_summary(context)`. It captures tensor count, invokes
the SiNet shuffle-tail owner once, invokes the existing ordered SE-FC/Gather
pair once with the same context, and returns the three normalized rewrite
counters plus prune evidence.

The fallback and absolute-final sites now consume one summary mapping each and
use the existing generic positive-count predicate. Their path-specific
ModelIR/LayoutState arguments, rewrite-or-prune reconciliation semantics,
phase IDs, and neighboring boundaries are unchanged. The raw lowerer SiNet
wrapper and raw pair helper remain defined, and the phase-result store remains
exactly 128/128.

Final sequential validation under core-only `uv`:

- focused shared-summary contracts: `4 passed in 0.54s`;
- affected boundary, owner, runtime, store, and architecture contracts:
  `410 passed in 21.87s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.83s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.32s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.66s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, audit the next
compatible repeated lowerer evidence family before production changes. Keep
all validation sequential and continue with commits and pushes only; never
create, update, or reopen a pull request.

## Shared precision-cleanup sequence characterization

The next selected family is the duplicated fallback and primary-final
DIV-to-reciprocal → consecutive-MUL → precision-sensitive-DIV-restore
sequence. The fallback path uses `fallback_ir`, no layout state, and diagnostics
only for consecutive-MUL. The primary path uses `model_ir`, the conversion
layout state for all three stages, and diagnostics only for consecutive-MUL.

The focused contract fixes all six current result locals, raw call order and
occurrence counts, exact argument policy, fallback topology/unbound-repair
boundaries, and the primary topological-progress successor. One strict xfail
requires a pass-module `run_precision_cleanup_sequence(context)` owner that
returns the three independent raw mappings in order. GraphIndex sharing is
explicitly excluded because the middle stage owns a transactional pass state.

Sequential validation under core-only `uv` completed with
`365 passed, 1 xfailed in 18.94s` across the new characterization and related
precision, graph-cleanup, fallback, topology, terminal-layout, phase-store,
and architecture contracts. The sole expected failure is the unimplemented
shared sequence owner. Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the shared owner and these two compatible three-stage sites, retain the raw
lowerer compatibility re-exports and the earlier core consecutive-MUL caller,
update owner-aware contracts, and validate sequentially. Continue with commits
and pushes only; never create, update, or reopen a pull request.

## Shared precision-cleanup sequence implementation

The new precision-cleanup pass module exposes
`run_precision_cleanup_sequence(context)`. It invokes DIV-to-reciprocal,
transactional consecutive-MUL, and sensitive-DIV restore once each in source
order and returns their three raw mappings unchanged. Fallback omits the layout
keyword exactly as before; primary-final forwards the conversion layout state;
only consecutive-MUL receives diagnostics.

The two production sites now retain one ordered-result tuple each instead of
six individual mapping locals. The fallback topology/unbound-repair boundaries,
primary progress/sort boundary, independent core consecutive-MUL call, private
precision compatibility re-exports, and 128/128 phase-result store are
unchanged. GraphIndex sharing remains intentionally excluded across the
transactional middle stage.

Final sequential validation under core-only `uv`:

- focused shared-sequence contracts: `4 passed in 0.55s`;
- affected precision, cleanup, fallback, topology, terminal, store, and
  architecture contracts: `368 passed in 20.89s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.89s`;
- synthetic core runtime contracts: `55 passed in 0.96s`;
- result contracts: `196 passed in 9.39s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.88s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, audit the next
compatible adjacent unconsumed-result cluster before production changes. Keep
all validation sequential and continue with commits and pushes only; never
create, update, or reopen a pull request.

## Inherited shared-late successor contract repair

The broader signature-characterization gate exposed a stale test-only
successor constant. It still named the removed
`late_binary_repair_tensor_count` snapshot, while the earlier late-binary
decision extraction replaced that boundary with
`_late_binary_repair_requires_reconciliation`.

The constant now follows the existing production boundary. The focused file is
`13 passed in 0.56s`; no production behavior or 128/128 store entry changed.
Commit and push this repair separately, then resume the boundary-signature
characterization. Continue with commits and pushes only; never create, update,
or reopen a pull request.

## Absolute-final boundary-signature pair characterization

The next selected cluster is the adjacent absolute-final dynamic-boundary
realignment and static-signature consistency sanitizer. Both existing owners
take only `model_ir`, return independent bounded mappings, and precede the
absolute-final affine post-Add cleanup. Later terminal realignment and
late-binary sanitizer callers remain separate.

The focused contract fixes both current targets, raw call order and counts,
the following affine boundary, and both raw lowerer wrappers. One strict xfail
requires `run_boundary_shape_signature_cleanup(model_ir)` in the existing
signature module and one ordered-result target in the lowerer.

Sequential validation under core-only `uv` completed with
`361 passed, 1 xfailed in 18.70s` across the new characterization and related
signature, terminal-layout, late-binary, shared-late, phase-store, and
architecture contracts. The sole expected failure is the unimplemented
ordered pair owner. Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the pair owner and absolute-final site, retain both raw wrappers and every
other caller, update owner-aware contracts, and validate sequentially.
Continue with commits and pushes only; never create, update, or reopen a pull
request.

## Absolute-final boundary-signature pair implementation

The existing signature-sanitization module now exposes
`run_boundary_shape_signature_cleanup(model_ir)`. It returns dynamic-boundary
realignment and static-signature sanitization mappings unchanged in their
existing order. The absolute-final site now retains one ordered tuple instead
of two individual locals.

The following affine cleanup, later terminal realignment, shared-late
realignment, late-binary sanitizer, and both lowerer wrappers remain unchanged.
Owner-aware architecture coverage preserves three realign routes and two
sanitize routes. No context/index/layout/diagnostics handoff or 128/128 store
entry is added.

Final sequential validation under core-only `uv`:

- focused ordered-pair contracts: `3 passed in 0.57s`;
- affected signature, terminal, late recovery, store, and architecture
  contracts: `374 passed in 20.32s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.80s`;
- synthetic core runtime contracts: `55 passed in 0.91s`;
- result contracts: `196 passed in 9.05s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.90s`;
- Ruff, bytecode compilation, 128/128 audit, and whitespace checks: passed.

Commit and push this implementation checkpoint. At resume, audit the next
small semantically closed unconsumed-result cluster before production changes.
Keep validation sequential and continue with commits and pushes only; never
create, update, or reopen a pull request.

## No-layout final SE-FC/affine pair characterization

The next selected cluster is the guarded no-layout final SE-FC cleanup and
affine pre/post cleanup pair. Both receive the primary ModelIR and layout state;
only SE-FC receives diagnostics. The pair precedes the guarded topology
checkpoint and following boundary-signature cleanup.

The focused contract fixes the guard, two current targets, exact arguments,
raw order, both topology boundaries, signature successor, affine wrapper, and
SE-FC compatibility import. One strict xfail requires
`run_no_layout_final_cleanup(shared_model_ir_pass_context)` and one ordered
result target.

Sequential validation under core-only `uv` completed with
`367 passed, 1 xfailed in 18.74s` across the new characterization and related
terminal, topology, affine, SE-layout, efficiency, store, architecture, and
signature contracts. The sole expected failure is the unimplemented shared
context owner. Targeted Ruff and whitespace checks passed.

Commit and push this characterization separately. At resume, implement only
the guarded pair owner and site, retain the affine wrapper, SE-FC compatibility
import, all other callers, and both topology boundaries, then validate
sequentially. Continue with commits and pushes only; never create, update, or
reopen a pull request.

## No-layout final SE-FC/affine pair implementation

`passes/no_layout_final_cleanup_orchestration.py` now provides
`run_no_layout_final_cleanup(shared_model_ir_pass_context)`. It invokes SE-FC
layout cleanup before affine pre/post cleanup, forwards the shared ModelIR and
layout state to both, forwards diagnostics only to SE-FC, and returns both raw
mappings unchanged as an ordered tuple.

The guarded lowerer site retains that tuple instead of two individual
unconsumed locals. The option guard, preceding and following topology
checkpoints, boundary-signature successor, affine lowerer wrapper, SE-FC
compatibility re-export, every other raw caller, and exact graph mutation order
remain intact. The context owner stays outside the already-full 128/128
phase-result store.

Final sequential validation under core-only `uv`:

- focused owner contracts: `3 passed in 0.57s`;
- affected owner-aware contracts: `369 passed in 18.98s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.78s`;
- synthetic core runtime contracts: `55 passed in 0.91s`;
- result contracts: `196 passed in 9.15s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.62s`;
- targeted Ruff passed.

No real-model conversion was repeated; runtime callback injection and the
owner-aware structural suite cover exact context forwarding, raw schemas,
order, guards, neighboring boundaries, and independent routes. Commit and push
this checkpoint. On resume, audit the next small semantically closed
unconsumed-result cluster before production changes. Keep validation
sequential and continue with commits and pushes only; never create, update, or
reopen a pull request.

## Inherited late-binary successor contract repair

The next-cluster affected gate found one stale test-only boundary after the
late-binary reconciliation guard. Production already uses
`_pre_terminal_instancenorm_layout_results` with
`run_pre_terminal_instancenorm_layout_cleanup(shared_model_ir_pass_context)`,
while the old assertion still named the removed direct post-bias result and
wrapper.

The test now follows the established owner and exact argument policy. Focused
validation is `5 passed in 0.55s`; no production behavior or 128/128 store
entry changed. Commit and push this repair separately before committing the
absolute-final affine/InstanceNorm characterization. Continue with commits
and pushes only; never create, update, or reopen a pull request.

## Absolute-final affine/InstanceNorm pair characterization

The next selected cluster is the adjacent absolute-final affine post-ADD and
decomposed-InstanceNorm post-bias cleanup pair. Both raw owners receive the
primary ModelIR and layout state, return independent one-counter mappings, and
sit between boundary-signature cleanup and the normalization/attention owner.

The focused contract fixes both current targets, exact arguments and order,
the signature predecessor, normalization/attention successor, and both lowerer
wrappers. One strict xfail requires
`run_absolute_final_affine_instancenorm_cleanup(shared_model_ir_pass_context)`
and one ordered result target.

Sequential validation under core-only `uv` completed with
`676 passed, 1 xfailed in 21.46s` across all affected owner, boundary,
terminal, late-binary, store, and architecture contracts. The sole expected
failure is the unimplemented context owner. Focused Ruff and whitespace checks
passed.

Commit and push this characterization separately. At resume, implement only
the two-pass context owner and absolute-final site, retain both raw lowerer
wrappers and every other caller, update owner-aware occurrence contracts, and
validate sequentially. Keep the phase-result store at 128/128 and never
create, update, or reopen a pull request.

## Absolute-final affine/InstanceNorm pair implementation

`passes/absolute_final_affine_instancenorm_orchestration.py` now owns the
adjacent affine post-ADD and decomposed-InstanceNorm post-bias cleanups through
`run_absolute_final_affine_instancenorm_cleanup(shared_model_ir_pass_context)`.
It forwards the same ModelIR/LayoutState to both callbacks and returns their
unchanged one-counter mappings in the original order.

The lowerer uses one ordered tuple in place of two unconsumed locals. The
boundary-signature predecessor, normalization/attention successor, both raw
lowerer wrappers, every other caller, and exact mutation order remain intact.
The new owner stays outside the already-full 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused context-owner contracts: `3 passed in 0.58s`;
- affected owner-aware contracts: `678 passed in 23.97s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.81s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.18s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.61s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model conversion was repeated; runtime callback injection and the
owner-aware structural suite cover the complete unchanged contract. Commit
and push this checkpoint. On resume, inventory the next small semantically
closed unconsumed-result cluster before changing production. Keep all
validation sequential and continue with commits and pushes only; never create,
update, or reopen a pull request.

## Inherited target-context construction contract repair

The following characterization gate found one stale test-only count for
target-specific `ModelIRPassContext` construction. Production already has four
such helpers: SE-FC/Gather, SiNet summary, precision cleanup, and
singleton/consecutive-Reshape. Each constructs the same explicit target
ModelIR/LayoutState plus session diagnostics contract.

The test now names all four helpers and requires one construction per helper.
Focused validation is `29 passed in 0.55s`; no production behavior or 128/128
store entry changed. Commit and push this repair separately before the next
characterization checkpoint. Continue with commits and pushes only; never
create, update, or reopen a pull request.

## Absolute-final normalization/attention rank-one characterization

The next selected cluster is the existing absolute-final normalization/pad and
mixed-attention owner followed by dynamic rank-one Unsqueeze/Reshape repair.
Both share the primary ModelIR/LayoutState context and run immediately before
the absolute-final topology/layout refresh. The intended result is a nested
ordered pair that preserves the existing two-mapping tuple and one-counter
mapping without flattening either schema.

The focused contract fixes both current targets, callback order and arguments,
the affine/InstanceNorm predecessor, topology/layout successor, lowerer closure
and context alias, and raw dynamic-rank-one wrapper. One strict xfail requires
`run_absolute_final_normalization_attention_rank1_cleanup(shared_model_ir_pass_context)`.

Sequential validation under core-only `uv` completed with
`423 passed, 1 xfailed in 19.29s` across all affected context, boundary,
dynamic-Reshape, terminal, store, and architecture contracts. The sole
expected failure is the missing composite owner. Focused Ruff and whitespace
checks passed.

Commit and push this characterization separately. At resume, add the composite
function to the existing normalization/attention module, replace the two
lowerer results with one nested pair, remove only the now-redundant lowerer
closure and context alias, retain the raw dynamic-rank-one wrapper and all
other callers, then validate sequentially. Keep the store at 128/128 and never
create, update, or reopen a pull request.

## Absolute-final normalization/attention rank-one implementation

`run_absolute_final_normalization_attention_rank1_cleanup(context)` now lives
in the existing normalization/attention orchestration module. It invokes the
unchanged two-pass owner first and dynamic rank-one repair second, forwarding
the same ModelIR/LayoutState and returning `(normalization_attention_results,
dynamic_rank1_result)` without flattening either schema.

The lowerer replaces two unconsumed locals with one composite target and
removes the now-redundant zero-argument closure and context alias. The
affine/InstanceNorm predecessor, topology/layout successor, raw lowerer
wrapper, other rank-one callers, shared normalization/attention state scope,
and exact mutation order remain intact. The store stays 128/128.

Final sequential validation under core-only `uv`:

- focused nested-schema contracts: `3 passed in 0.69s`;
- affected owner-aware contracts: `425 passed in 20.73s`;
- terminal-layout/pass-efficiency contracts: `92 passed in 1.84s`;
- synthetic core runtime contracts: `55 passed in 0.95s`;
- result contracts: `196 passed in 9.30s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras blocker, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.98s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model conversion was repeated; runtime and owner-aware structural
coverage prove the unchanged composite contract. Commit and push this
checkpoint. On resume, inventory the next small semantically closed
unconsumed-result cluster before production changes. Keep all validation
sequential and continue with commits and pushes only; never create, update, or
reopen a pull request.

## Indexed binary-layout convergence owner characterization

The next selected boundary is `_run_indexed_binary_layout_convergence`, whose
full three-stage convergence loop still lives in the lowerer. It shares one
graph index across broadcast-constant repair, stale binary-adapter repair, and
static-shape reconciliation; repeats for at most three rounds; stops on the
first stable round; and accumulates three counters. Fallback and primary
terminal paths are its two callers.

The focused contract fixes index lifetime, callback order and arguments, round
cap, result-key order, and caller inputs. One strict xfail requires a new
`passes/binary_layout_convergence.py` owner with the lowerer function retained
as a compatibility wrapper.

Sequential validation under core-only `uv` completed with
`399 passed, 1 xfailed in 19.19s` across all affected convergence, fallback,
terminal, adapter, store, and architecture contracts. The sole expected
failure is the unimplemented pass-module owner. Focused Ruff and whitespace
checks passed.

Commit and push this characterization separately. At resume, move the loop
mechanically into the pass module, import the three raw pass owners directly,
retain the lowerer wrapper and both call sites, redirect runtime monkeypatch
tests to the new owner module, then validate sequentially. Keep the store at
128/128 and never create, update, or reopen a pull request.

## Indexed binary-layout convergence owner implementation

`passes/binary_layout_convergence.py` now owns the full convergence loop. It
constructs one `ModelIRGraphIndex`, forwards that same object to broadcast-
constant repair, stale binary-adapter repair, and static-shape reconciliation,
preserves that exact order, caps execution at three rounds, stops on the first
stable round, and returns the unchanged ordered three-counter mapping.

The lowerer retains `_run_indexed_binary_layout_convergence` as a one-return
compatibility wrapper. Both existing callers remain in place and still pass
`fallback_ir` followed later by primary `model_ir`. Runtime monkeypatch tests
were redirected to the pass-module owner so callback order, index identity,
stable stopping, and the hard round cap are exercised at the new ownership
boundary. No phase result was added; the store remains exactly 128/128.

Sequential validation under core-only `uv` completed with:

- focused owner contracts: `2 passed in 0.15s`;
- affected contracts: `400 passed in 18.70s`;
- terminal-layout/pass-efficiency: `92 passed in 1.85s`;
- core runtime: `55 passed in 0.95s`;
- result contracts: `196 passed in 8.93s`;
- phase-store capacity: `2 passed in 0.53s`;
- TensorFlow isolation/default direct/`-cotof`: `11 passed in 9.53s`.

On resume, characterize the adjacent terminal stabilization triple before any
production edit: indexed binary-layout convergence, static high-rank binary
coalescing, and dynamic boundary-signature realignment immediately before
primary terminal topology/layout validation. Preserve the three independent
raw mappings and both compatibility wrappers, keep the store at 128/128, and
never create, update, or reopen a pull request.

## Terminal stabilization composite characterization

The next contract is now fixed in
`tests/test_flatbuffer_direct_terminal_stabilization_orchestration.py`. It
preserves the exact final-primary sequence:

1. indexed binary-layout convergence with `model_ir`;
2. static high-rank binary coalescing with `model_ir` and
   `session.layout_state`;
3. dynamic boundary-signature realignment with `model_ir`;
4. terminal topology/layout validation;
5. ModelIR finalization.

The contract also fixes all three raw result names and mapping identities. One
strict xfail requires a new
`passes/terminal_stabilization_orchestration.py` context owner that returns the
three mappings as an ordered tuple and one lowerer composite result using
`shared_model_ir_pass_context`.

Sequential affected validation completed with
`388 passed, 1 xfailed in 18.72s`; the sole expected failure is the absent
owner. Ruff and whitespace checks passed. Production and the full 128/128
phase-result store are unchanged.

Commit and push this characterization separately. At resume, implement the
straight-line context owner using the three existing pass-module owners,
replace the three terminal locals with one ordered composite result, update
owner-aware structural/runtime coverage, and run all affected and standard
gates sequentially. Never create, update, or reopen a pull request.

## Terminal stabilization composite implementation

`passes/terminal_stabilization_orchestration.py` now runs the characterized
triple through `shared_model_ir_pass_context`. It preserves convergence →
high-rank binary coalescing → boundary-signature realignment order, forwards
LayoutState only to coalescing, and returns all three unchanged mappings as an
ordered tuple.

The lowerer replaces the three old result locals with
`_final_terminal_stabilization_results`. Both raw lowerer compatibility
wrappers remain defined, the fallback convergence path is unchanged, and
primary terminal topology/layout validation plus finalization remain directly
after the new composite. No phase result was added; the store remains exactly
128/128.

Sequential validation under core-only `uv` completed with:

- focused context-owner contracts: `3 passed in 0.54s`;
- affected contracts: `390 passed in 18.66s`;
- terminal-layout/pass-efficiency: `92 passed in 1.74s`;
- core runtime: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.03s`;
- phase-store capacity: `2 passed in 0.52s`;
- TensorFlow isolation/default direct/`-cotof`: `11 passed in 9.64s`.

Runtime injection proves model/layout identity, callback order, tuple shape,
and raw mapping identity. On resume, perform a fresh AST inventory of remaining
unconsumed lowerer results and select the next small semantically closed non-
store boundary before changing production. Keep validation sequential, the
store at 128/128, and never create, update, or reopen a pull request.

## Absolute-final cleanup composite characterization

The fresh inventory selected the three adjacent results directly before
`topology_layout.primary.absolute_final`:

1. `run_boundary_shape_signature_cleanup(model_ir)`;
2. `run_absolute_final_affine_instancenorm_cleanup(`
   `shared_model_ir_pass_context)`;
3. `run_absolute_final_normalization_attention_rank1_cleanup(`
   `shared_model_ir_pass_context)`.

The new focused contract preserves their result names and nested schemas,
exact argument policy and order, shared-context identity, and immediate
topology/layout refresh successor. One strict xfail requires
`passes/absolute_final_cleanup_orchestration.py` to own an ordered triple and
the lowerer to retain one `_absolute_final_cleanup_results` target.

Sequential affected validation completed with
`387 passed, 1 xfailed in 19.31s`; the only expected failure is the missing
owner. Ruff and whitespace checks passed. Production and the full 128/128
store are unchanged.

Commit and push this characterization separately. At resume, implement the
straight-line context owner by composing the three existing pass-module
owners, replace only the three lowerer targets, add runtime identity/order/
nested-tuple coverage, and run affected plus standard gates sequentially.
Never create, update, or reopen a pull request.

## Absolute-final cleanup composite implementation

`passes/absolute_final_cleanup_orchestration.py` now owns the ordered outer
triple. Boundary-signature cleanup receives `context.model_ir`; the existing
affine/InstanceNorm and normalization/attention/rank-one composites receive the
same context object. All three nested result objects are returned unchanged.

The lowerer retains one `_absolute_final_cleanup_results` target and removes
its three direct sub-owner imports. The no-layout predecessor and absolute-
final topology/layout refresh successor remain adjacent, all sub-owner and raw
wrapper contracts remain available, and the phase-result store stays 128/128.

Sequential validation under core-only `uv` completed with:

- focused top-level owner contracts: `3 passed in 0.55s`;
- affected contracts: `389 passed in 18.97s`;
- terminal-layout/pass-efficiency: `92 passed in 1.75s`;
- core runtime: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.21s`;
- phase-store capacity: `2 passed in 0.53s`;
- TensorFlow isolation/default direct/`-cotof`: `11 passed in 9.57s`.

On resume, rerun the read-only unconsumed-result inventory after this removal
and choose the next small source-adjacent non-store cluster. Do not combine a
guarded decision with an unconditional cleanup merely to reduce locals. Keep
validation sequential, the store at 128/128, and never create, update, or
reopen a pull request.

## Very-late dynamic/adapter composite characterization

The inventory skipped the preceding orphan/unbound/affine group because the
unbound repair still depends on a lowerer-only mapping wrapper. The selected
six-result boundary is fully pass-module-owned and runs:

1. dynamic Reshape resolution with runtime-inferable ONNX raw shapes enabled;
2. indexed Conv-input adapter summary;
3. stale NCHW channel-shuffle repair with LayoutState and diagnostics;
4. Concat/Transpose/Conv axis repair with LayoutState;
5. Concat/global-pool/Conv axis repair with LayoutState;
6. dynamic rank-one Unsqueeze/Reshape repair with LayoutState.

The new contract also fixes the immediately following
`shape_reconciliation.primary.very_late_final` record and split-fallback
assignment. One strict xfail requires
`passes/very_late_dynamic_adapter_orchestration.py` to return the six raw
mappings as an ordered context-owned tuple.

Sequential affected validation completed with
`497 passed, 1 xfailed in 18.94s`; the only expected failure is the missing
owner. Ruff and whitespace checks passed. Production and the full 128/128
store are unchanged.

Commit and push this characterization separately. At resume, implement the
straight-line context owner using existing pass-module callbacks, alias the two
private Concat-axis callbacks locally, retain all lowerer compatibility
wrappers and independent callers, add runtime identity/order/tuple coverage,
and run affected plus standard gates sequentially. Never create, update, or
reopen a pull request.

## Very-late dynamic/adapter composite implementation

`passes/very_late_dynamic_adapter_orchestration.py` now runs the characterized
six-stage sequence through `shared_model_ir_pass_context`. It preserves the
dynamic-Reshape flag, ModelIR/LayoutState/diagnostics argument policy, exact
order, and all six mapping objects. Private Concat-axis callbacks are imported
under module-local public aliases only.

The lowerer replaces six result targets with
`_very_late_dynamic_adapter_results`. All compatibility wrappers and fallback/
independent callers remain, while the now-unneeded direct stale channel-shuffle
import is removed. Mandatory very-late static-shape reconciliation and split
fallback remain immediate successors. The phase-result store stays 128/128.

Sequential validation under core-only `uv` completed with:

- focused context-owner contracts: `3 passed in 0.54s`;
- affected contracts: `499 passed in 18.86s`;
- terminal-layout/pass-efficiency: `92 passed in 1.76s`;
- core runtime: `55 passed in 0.93s`;
- result contracts: `196 passed in 8.97s`;
- phase-store capacity: `2 passed in 0.53s`;
- TensorFlow isolation/default direct/`-cotof`: `11 passed in 9.52s`.

On resume, rerun the unconsumed-result inventory. The preceding orphan/
unbound-input/affine/normalization group remains deliberately deferred until
the lowerer-only unbound-input mapping wrapper has a pass-module owner. Select
another fully owned source-adjacent cluster if available; otherwise
characterize that wrapper extraction separately before composing the group.
Keep validation sequential, the store at 128/128, and never create, update, or
reopen a pull request.

## Unbound-input repair owner characterization

The deferred prerequisite is now characterized without production changes.
The lowerer-local wrapper performs one indexed unbound-input layout repair,
reconciles static shapes only when its repaired count is positive using the
returned graph index, and normalizes the result to the existing one-key mapping.
Both callers remain fixed: primary `model_ir` and safety-path `fallback_ir`.

The focused contract completed with `1 passed, 1 xfailed`; the affected suite
completed with `392 passed, 1 xfailed`. The strict expected failure requires
`passes/unbound_input_repair_orchestration.py` and a one-return compatibility
wrapper. Production, public behavior, TensorFlow isolation, and the full
128-ID/128-owner phase-result store are unchanged; no model conversion was
run.

At resume, implement only this mechanical owner extraction. Preserve the
lowerer wrapper and both callers, keep the repair → conditional reconciliation
order and graph-index identity exact, then run focused, affected, and standard
gates sequentially before considering the larger adjacent result group. Do
not create, update, or reopen a pull request.

## Unbound-input repair owner implementation

The mechanical prerequisite extraction is complete.
`passes/unbound_input_repair_orchestration.py` owns raw indexed repair followed
by mutation-positive static-shape reconciliation. It preserves the returned
GraphIndex identity and exact one-key result mapping. The lowerer compatibility
wrapper and both primary/fallback callers remain in place.

Sequential gates passed: focused `3`, affected `394`, terminal/efficiency
`92`, core runtime `55`, result contracts `196`, phase-store `2`, and
TensorFlow-isolation/default-direct/`-cotof` `11`. The store remains exactly
128 IDs and 128 owners. No real-model conversion was run.

At resume, rerun the read-only unconsumed-result inventory. Re-evaluate the
adjacent orphan/unbound/affine/normalization region now that every constituent
has a pass-module owner, but characterize only a small semantically closed
straight-line unit before production changes. Keep every test sequential and
do not create, update, or reopen a pull request.

## Recurrent-alias repair mapping-owner characterization

The refreshed inventory found that raw recurrent-alias mutation is already
pass-module-owned, but its direct-TFLite integer-to-mapping normalization still
lives in the lowerer. A focused strict contract now fixes raw call arguments,
optional GraphIndex forwarding, exact one-key schema, the sole primary caller,
and the required one-return compatibility adapter.

Sequential focused validation is `1 passed, 1 xfailed`; affected validation is
`307 passed, 1 xfailed`. Production, the independent PyTorch raw-owner path,
public behavior, TensorFlow isolation, and the 128-ID/128-owner store remain
unchanged. No model conversion was run.

At resume, implement only
`passes/recurrent_alias_repair_orchestration.py`, retain the lowerer wrapper
and primary caller, and leave PyTorch's raw owner path unchanged. Run focused,
affected, and standard gates sequentially, then reassess the now-fully-owned
adjacent late-result cluster. Never create, update, or reopen a pull request.

## Recurrent-alias repair mapping-owner implementation

`passes/recurrent_alias_repair_orchestration.py` now owns the direct-TFLite
integer-to-mapping conversion. The lowerer retains a one-return compatibility
adapter and its sole primary caller. The raw indexed mutation owner and the
independent PyTorch wrapper remain unchanged, including GraphIndex identity and
PyTorch's void return convention.

Sequential gates passed: focused `3`, affected `309`, terminal/efficiency
`92`, core runtime `55`, results `196`, phase-store `2`, and TensorFlow
isolation/default-direct/`-cotof` `11`. The store remains 128 IDs and 128
owners; no model conversion was run.

At resume, rerun the unconsumed-result inventory and characterize the smallest
semantically closed prefix of the adjacent late orphan/unbound/affine/
normalization sequence, whose individual mapping boundaries are now all
pass-module-owned. Preserve its progress predecessor and dynamic-adapter
successor, validate sequentially, and never create, update, or reopen a pull
request.

## Late input/affine/normalization composite characterization

The now-fully-owned late prefix contains four adjacent unconditional mappings:
recurrent alias, unbound-input repair, affine post-Add, and prune-aware
normalization. They share the existing `ModelIRPassContext`, have no consumers,
follow `_advance_post_progress()`, and directly precede the very-late dynamic-
adapter composite.

The focused contract reports `1 passed, 1 xfailed`; the affected sequential
suite reports `394 passed, 1 xfailed`. The strict expected failure requires a
four-stage context owner and one replacement lowerer result. Production,
TensorFlow isolation, and the 128-ID/128-owner store remain unchanged; no
model conversion was run.

At resume, implement the straight-line owner without flattening or copying any
mapping. Pass ModelIR to the two repair summaries, ModelIR plus LayoutState to
affine cleanup, and the exact shared context to normalization. Preserve the
progress predecessor and dynamic-adapter successor, then run affected and
standard gates sequentially. Never create, update, or reopen a pull request.

## Late input/affine/normalization composite implementation

`passes/late_input_affine_normalization_orchestration.py` now runs recurrent
repair summary → unbound-input repair summary → affine post-Add cleanup →
prune-aware normalization through one shared context and returns their four raw
mappings unchanged. The lowerer retains one composite result; all wrappers,
fallback and independent routes, progress predecessor, and dynamic-adapter
successor remain intact.

Sequential gates passed: focused `3`, affected `396`, terminal/efficiency
`92`, core runtime `55`, results `196`, phase-store `2`, and TensorFlow
isolation/default-direct/`-cotof` `11`. The store remains exactly 128 IDs and
128 owners; no model conversion was run.

At resume, rerun the read-only unconsumed-result inventory after removal of
these four locals. Select the next smallest source-adjacent, semantically
closed, fully pass-module-owned non-store cluster and characterize it before
production changes. Keep all validation sequential and never create, update,
or reopen a pull request.

## Pre-terminal cleanup composite characterization

The fresh unconsumed-result inventory selected five adjacent unconditional
pre-terminal owners: InstanceNorm layout, affine/Concat/Split summary, pre-Add,
channel Slice/Pad/Mul summary, and affine tail. All receive aliases of the same
shared context and return unconsumed mappings or nested tuples.

The strict contract preserves every child schema and argument, the preceding
optional late-binary reconciliation guard, and the following separate terminal
affine rerun. Focused validation reports `1 passed, 1 xfailed`; affected
sequential validation reports `338 passed, 1 xfailed`. Production, TensorFlow
isolation, and the 128-ID/128-owner store are unchanged; no model conversion
was run.

At resume, implement `passes/pre_terminal_cleanup_orchestration.py` as a
straight-line five-stage owner. Pass the exact shared context to every child,
return all five raw objects unchanged, retain child owners and independent
callers, and preserve both outer boundaries. Run affected and standard gates
sequentially. Never create, update, or reopen a pull request.

## Pre-terminal cleanup composite implementation

The five-stage owner is now implemented in
`passes/pre_terminal_cleanup_orchestration.py`. It runs InstanceNorm layout,
affine/Concat/Split recovery, pre-Add, channel Slice/Pad/Mul, and affine-tail
cleanup with the exact shared context and returns each raw nested result in
source order. The lowerer replaces the five unconsumed locals with
`_pre_terminal_cleanup_results`; the preceding optional late-binary
reconciliation guard and following independent terminal-affine rerun remain
adjacent and unchanged.

Owner-aware AST coverage and runtime callback injection prove child order,
context identity, result identity, nested schemas, and both outer boundaries.
Sequential validation passed: focused `3`, affected `340`, focused stale-
boundary coverage `293`, terminal/efficiency `92`, core runtime `55`, result
contracts `196`, phase-store `2`, and TensorFlow isolation/default-direct/
`-cotof` `11`. Ruff and whitespace checks passed. The store remains exactly
128 IDs and 128 owners; no model conversion was run.

At resume, start with a read-only inventory of the remaining unconsumed
lowerer results after this removal. Select the smallest source-adjacent,
semantically closed cluster whose children are already pass-module-owned,
characterize it before production changes, and keep all verification under
`uv` sequential and single-process. Do not create, update, or reopen a pull
request; use appropriately scoped commits and pushes only.

## Late reshape/shuffle/attention/window composite characterization

The fresh inventory selected four adjacent unconsumed late-layout results:
reshape, base-only channel shuffle/Gather, attention, and window cleanup. They
all use the existing shared pass context; the base channel policy explicitly
disables two-way and NHWC shuffle. The preceding optional Concat elementwise-
fanout guard and following indexed final shape/activation convergence define
the fixed outer boundary. The separate guarded full channel-shuffle policy is
unchanged.

The strict contract preserves exact child owners, arguments, flags, nested raw
tuple schema `(3, 2, 4, 2)`, source order, and absence of consumers. Focused
validation reports `2 passed, 1 xfailed`; affected sequential validation
reports `403 passed, 1 xfailed`. The sole expected failure requires the new
context owner. A pre-existing layout-recovery AST test was independently
updated to unwrap already-recorded phase owners; it failed on the unchanged
production baseline. Ruff and whitespace checks passed. Production,
TensorFlow isolation, and the 128-ID/128-owner store remain unchanged; no model
conversion was run.

At resume, implement
`passes/late_reshape_shuffle_attention_window_orchestration.py` as a
straight-line four-stage owner. Pass the exact shared context to every child,
preserve the base-only channel flags and all raw tuple identities, retain the
lowerer channel-shuffle wrapper for its guarded full-policy and callback users,
and preserve both outer boundaries. Run affected and standard gates
sequentially. Never create, update, or reopen a pull request.

## Late reshape/shuffle/attention/window composite implementation

The four-stage owner is now implemented in
`passes/late_reshape_shuffle_attention_window_orchestration.py`. It invokes
reshape, base-only channel shuffle/Gather, attention, and window cleanup with
the exact shared context, retains both disabled base-channel flags, and returns
all four raw tuples unchanged in source order. The lowerer replaces four
unconsumed locals with one composite result and removes only imports made
redundant by that move.

The lowerer channel-shuffle wrapper remains for the guarded full-policy route
and argument-free layout-recovery callback. The optional elementwise-fanout
guard and indexed final shape/activation convergence remain the exact outer
boundaries. Owner-aware AST coverage and runtime injection prove order,
context identity, raw-result identity, nested schema, keyword policy, and
independent routes.

Sequential validation passed: focused `4`, owner-aware focused `356`, affected
`405`, terminal/efficiency `92`, core runtime `55`, result contracts `196`,
phase-store `2`, and TensorFlow isolation/default-direct/`-cotof` `11`. Ruff,
bytecode compilation, and whitespace checks passed. The store remains exactly
128 IDs and 128 owners; no model conversion was run.

At resume, rerun the read-only unconsumed-result inventory after removal of
these four locals. Select the next smallest source-adjacent and semantically
closed cluster whose children are already pass-module-owned, characterize it
before production changes, and keep every verification under `uv` sequential
and single-process. Never create, update, or reopen a pull request; use
appropriately scoped commits and pushes only.

## Final boundary/Slice/Concat composite characterization

The fresh inventory selected four adjacent unconsumed final-layout results:
boundary channel, terminal Slice/Concat recovery, final Slice/pre-Concat, and
terminal Concat bridge cleanup. The three layout owners use the exact shared
pass context. Slice/Concat recovery retains its custom context containing that
same pass context and the channel Slice/Pad/Mul callback. The indexed final-
shape result and optional terminal elementwise-fanout guard define the outer
boundaries; the earlier Slice/Concat wrapper call remains independent.

The strict contract preserves exact owner order, shared/custom context
identity, callback route, raw tuple schema `(3, 14, 2, 6)`, source adjacency,
and absence of consumers. Focused validation reports `2 passed, 1 xfailed`;
affected sequential validation reports `398 passed, 1 xfailed`. The sole
expected failure requires the new owner. Ruff and whitespace checks passed.
Production, TensorFlow isolation, and the 128-ID/128-owner store remain
unchanged; no model conversion was run.

At resume, implement
`passes/final_boundary_slice_concat_orchestration.py` as a straight-line
four-stage owner accepting `TerminalSliceConcatRecoveryContext`. Pass
`context.pass_context` to the three layout owners, pass the full context to
Slice/Concat recovery, return all four raw tuples unchanged, retain the lowerer
wrapper and earlier independent caller, and preserve both outer boundaries.
Run affected and standard gates sequentially. Never create, update, or reopen
a pull request.

## Final boundary/Slice/Concat composite implementation

The four-stage owner is now implemented in
`passes/final_boundary_slice_concat_orchestration.py`. It accepts the existing
callback-bearing terminal Slice/Concat context, forwards that complete context
to recovery, forwards the exact nested shared pass context to the three layout
children, and returns all four raw tuples unchanged. The lowerer replaces four
unconsumed locals with one result and removes only redundant direct imports.

The compatibility wrapper and earlier independent Slice/Concat invocation
remain. Indexed final shape/activation convergence and the optional terminal
elementwise-fanout guard remain the exact outer boundaries. Owner-aware AST
coverage and runtime injection prove order, custom/shared context identity,
raw-result identity, nested schema, and both recovery routes.

Sequential validation passed: focused `4`, owner-aware focused `349`, affected
`400`, terminal/efficiency `92`, core runtime `55`, result contracts `196`,
phase-store `2`, and TensorFlow isolation/default-direct/`-cotof` `11`. Ruff,
bytecode compilation, and whitespace checks passed. The store remains exactly
128 IDs and 128 owners; no model conversion was run.

At resume, rerun the read-only unconsumed-result inventory after removal of
these four locals. Select the next smallest source-adjacent and semantically
closed pass-module-owned cluster, characterize it before production changes,
and keep all verification under `uv` sequential and single-process. Never
create, update, or reopen a pull request; use appropriately scoped commits and
pushes only.

## Very-late layout-tail composite characterization

The fresh inventory selected four adjacent unconsumed results after late Swish
cleanup: Conv1D/decoder, Pad/InstanceNorm, singleton/consecutive Reshape, and
layout/broadcast cleanup. The primary calls share ModelIR, LayoutState, and
diagnostics; broadcast retains its option-dependent layout-Transpose flag. The
late-Swish result and phase-recorded very-late broadcast reconciliation define
the outer boundaries. The singleton wrapper's fallback call remains
independent with `fallback_ir` and no LayoutState.

The strict contract preserves exact child order, context and flag policy, both
singleton routes, raw tuple schema `(8, 4, 3, 2)`, both broadcast schema
variants, and absence of consumers. Focused validation reports
`3 passed, 1 xfailed`; affected sequential validation reports
`435 passed, 1 xfailed`. Two pre-existing stale expectations were corrected to
their already-extracted owners; production remained unchanged. Ruff and
whitespace checks passed. TensorFlow isolation and the 128-ID/128-owner store
remain unchanged; no model conversion was run.

At resume, implement `passes/very_late_layout_tail_orchestration.py` as a
straight-line four-stage owner accepting the shared `ModelIRPassContext` and
keyword-only `include_layout_transpose`. Call the singleton pass-module owner
directly with that exact context, retain the lowerer wrapper and fallback
caller, preserve the broadcast flag and all raw tuples, and retain both outer
boundaries. Run affected and standard gates sequentially. Never create,
update, or reopen a pull request.

## Very-late layout-tail composite implementation checkpoint

The characterized owner is now implemented in
`passes/very_late_layout_tail_orchestration.py`. It runs Conv1D/decoder,
Pad/InstanceNorm, singleton/consecutive Reshape, and optional
layout-Transpose/broadcast cleanup in fixed order with the exact shared
`ModelIRPassContext`. The lowerer retains the late-Swish predecessor, the
phase-recorded broadcast-reconciliation successor, and the independent
fallback singleton wrapper, but replaces the four unconsumed child locals
with `_very_late_layout_tail_results`.

Important design decisions:

- the new owner is a straight-line orchestration boundary and does not change
  matching, rewriting, graph mutation, or result schemas;
- child raw tuples are returned without copying or normalization;
- `include_layout_transpose` is forwarded exactly to the broadcast child;
- the primary singleton route uses the shared context directly, while the
  fallback wrapper keeps `(fallback_ir, None)`;
- specialized child owners remain independently tested and all ownership
  assertions are now aware of the outer composite;
- no phase result was added or removed, so the bounded store remains exactly
  128 IDs and 128 owners.

Changed production files are
`onnx2tf/tflite_builder/lower_from_onnx2tf.py` and
`onnx2tf/tflite_builder/passes/very_late_layout_tail_orchestration.py`.
Characterization and owner-aware contract tests were updated across the four
child orchestration families, singleton fallback coverage, terminal layout
validation, architecture coverage, mutation-evidence accounting, and affected
result-boundary tests. This handoff and the branch improvement/description
documents record the checkpoint.

Sequential `uv` validation passed: focused previously failing contracts 380,
complete affected contracts 437, terminal-layout/efficiency 92, core 55,
result contracts 196, phase-store 2, and TensorFlow import-blocking/default
direct/`-cotof` 11. No test is failing and no new known production issue was
introduced. No real-model conversion was run for this ownership-only move.

At resume, first rerun the read-only unconsumed-result inventory against the
committed lowerer. Select the next smallest source-adjacent, semantically
closed cluster whose children are already pass-module-owned, add a strict
characterization checkpoint before production changes, and keep all tests
sequential and single-process under `uv`. Continue with scoped commits and
pushes only. Never create, update, reopen, or otherwise modify a pull request.

## Terminal affine/Slice-SPP composite characterization checkpoint

The refreshed AST inventory selected three adjacent unconditional and
unconsumed results immediately after `_pre_terminal_cleanup_results`:
terminal affine/Concat/Split summary, strict StridedSlice/Pad/Concat cleanup,
and late SPP/Concat/Unary summary. The next statement is the QKV shape-extract
cleanup. Both existing summary context aliases are the exact shared pass
context, and the middle bridge operates on its `model_ir`.

The new strict characterization records child order, exact current arguments,
context-alias identity, both outer boundaries, absence of result consumers,
and complete empty-model mapping schemas of lengths `(13, 1, 2)`. Focused
validation reports `2 passed, 1 xfailed`; affected sequential validation
reports `518 passed, 1 xfailed`. The sole expected failure requires
`passes/terminal_affine_slice_spp_orchestration.py` and one replacement
lowerer result.

Two inherited StridedSlice AST tests were corrected during inventory. They
referenced pre-terminal child locals removed by an earlier committed composite
and a direct post-Add invocation now owned by existing orchestration modules.
The owner-aware replacements preserve the original mutation-evidence totals
and pass on the unchanged production baseline.

At resume, implement
`passes/terminal_affine_slice_spp_orchestration.py` as a straight-line
three-stage owner accepting `ModelIRPassContext`. Pass `context` unchanged to
the two summary owners, pass `context.model_ir` to the bridge owner, return the
three raw mapping objects unchanged, replace only the three unconsumed lowerer
locals, and preserve both outer boundaries. Run all affected and standard
gates sequentially. Commit and push only; never create or modify a pull
request.

## Terminal activation-bridge composite implementation checkpoint

The terminal affine/Slice-SPP and terminal QKV characterization checkpoints
have now both been implemented and committed. The newest implementation adds
`passes/terminal_activation_bridge_orchestration.py`, a straight-line owner
for the indexed Split/Conv/Concat bridge, prune-aware HardSwish-SE cleanup, and
late hard-activation layout summary.

Important design decisions:

- the exact shared `ModelIRPassContext`, its `model_ir`, and its `layout_state`
  are forwarded; no replacement context or graph copy is created;
- `include_layout_transpose` is forwarded unchanged to the hard-activation
  child;
- all three child mapping objects are returned unchanged and in source order;
- the lowerer replaces only three unconsumed observation locals;
- raw compatibility wrappers and earlier independent routes remain available;
- terminal QKV and absolute-final pre-ConCat cleanup remain the exact outer
  boundaries;
- no phase result was added or removed, so the bounded store remains exactly
  128 IDs and 128 owners.

Changed production files in this checkpoint are
`onnx2tf/tflite_builder/lower_from_onnx2tf.py` and
`onnx2tf/tflite_builder/passes/terminal_activation_bridge_orchestration.py`.
Owner-aware tests were updated for the indexed bridge, HardSwish-SE, hard
activation, terminal QKV successor, pre-ConCat predecessor, shared context,
architecture, terminal validation, and result retention.

Sequential `uv` validation passed: focused 5, complete affected 445,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. No test is failing,
no new production issue is known, and no real-model conversion was repeated
for this ownership-only move.

At resume, first rerun the read-only inventory of remaining unconsumed lowerer
results. Characterize the next smallest source-adjacent, semantically closed
cluster before changing production, then implement it as a separate
checkpoint with all validation sequential and single-process under `uv`.

## Terminal layout/shape composite characterization checkpoint

The next candidate is now characterized but not implemented. It consists of
four consecutive unconsumed results after terminal activation:

1. absolute-final pre-ConCat cleanup;
2. late NHWC-to-NCHW Shape-extract cleanup;
3. prune-aware late layout/Mean/SPP/Gather/constant-fold/Cast summary;
4. terminal Expand/Squeeze-to-Reshape cleanup.

The contract requires one `ModelIRPassContext` owner, exact child order, the
same ModelIR/LayoutState/diagnostics objects, unchanged
`include_layout_transpose` forwarding, and raw mapping identity. The owner
must not absorb the following phase-recorded complete static-shape
reconciliation or `_advance_post_progress`; those are the fixed successor
boundary. The terminal activation composite is the fixed predecessor.

Focused characterization reports `3 passed, 1 xfailed`; complete affected
characterization reports `404 passed, 1 xfailed`. The only xfail is the
intentionally absent owner. A stale terminal-QKV successor expectation in the
Shape-extract suite was corrected to the already-committed terminal activation
owner; production was unchanged. The phase-result store remains exactly 128
IDs and 128 owners.

At resume, implement
`passes/terminal_layout_shape_orchestration.py` as a straight-line four-stage
owner. Accept the shared context plus keyword-only
`include_layout_transpose`, forward exact state to every child, return all raw
mappings unchanged, replace only the four characterized lowerer locals, and
preserve both outer boundaries. Run affected and standard gates sequentially,
then commit and push only. Do not create or modify a pull request.

## Terminal layout/shape composite implementation checkpoint

The characterized four-stage owner is now implemented in
`passes/terminal_layout_shape_orchestration.py`. It runs absolute pre-ConCat,
Shape-extract, prune-aware late layout/Mean/SPP/Gather/constant-fold/Cast, and
Expand/Squeeze-to-Reshape cleanup with the exact shared
`ModelIRPassContext`. It forwards the same ModelIR, LayoutState, diagnostics,
and layout-Transpose option required by the original calls and returns all
four raw mapping objects unchanged in source order.

The lowerer replaces only the four unconsumed result locals with
`_terminal_layout_shape_results`. Compatibility wrappers and independent
routes remain. Terminal activation remains the predecessor; the complete
static-shape reconciliation phase record and progress update remain separate
successors. No phase-store entry was added or removed, so capacity remains
exactly 128 IDs and 128 owners.

Sequential `uv` validation passed: focused 5, complete affected 406,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff, bytecode, and
whitespace checks passed. No test is failing and no new production issue is
known. No real-model conversion was repeated for this ownership-only move.

At resume, rerun the read-only inventory of remaining unconsumed lowerer
results. Select the next smallest source-adjacent, semantically closed cluster
whose children already have pass-module owners, characterize it before
production changes, and keep every test sequential and single-process under
`uv`. Commit and push only; never create, update, or reopen a pull request.

## Final input/dynamic composite characterization checkpoint

The next two-stage candidate is characterized but not implemented. It joins
the existing late input/affine/normalization composite with the existing
very-late dynamic adapter composite. Both receive the exact same shared
`ModelIRPassContext`; their raw nested tuple lengths are four and six.

The fixed predecessor is `_advance_post_progress`. The fixed successor is the
phase-recorded complete static-shape reconciliation
`shape_reconciliation.primary.very_late_final`, followed immediately by the
Split fallback assignment. The new owner must not absorb either successor or
change any child schema, state identity, order, or result object.

Focused characterization reports `1 passed, 1 xfailed`; complete affected
characterization reports `399 passed, 1 xfailed`. The only xfail is the
intentionally absent `passes/final_input_dynamic_orchestration.py`. Production
and the 128-ID/128-owner phase store are unchanged.

At resume, implement `run_final_input_dynamic_cleanup(context)` as a
straight-line two-child owner, return both raw nested tuples unchanged, and
replace only the two unconsumed lowerer locals. Preserve progress,
reconciliation, Split fallback, all child owners, and all independent routes.
Run affected and standard gates sequentially, then commit and push only. Never
create, update, or reopen a pull request.

## Final input/dynamic composite implementation checkpoint

`passes/final_input_dynamic_orchestration.py` now owns the fixed two-child
tail. It passes the exact shared `ModelIRPassContext` to the existing late
input/affine/normalization and very-late dynamic-adapter owners in source order
and returns both raw nested tuples without copying or flattening them.

The lowerer replaces only the two unconsumed locals with
`_final_input_dynamic_results`. `_advance_post_progress`, the
`shape_reconciliation.primary.very_late_final` phase record, and Split
fallback remain outside the owner. Child owners, specialized pass contracts,
public behavior, and the 128-ID/128-owner phase store are unchanged.

Sequential `uv` validation passed: focused 3, complete affected 401,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff, bytecode, and
whitespace checks passed. No test is failing and no new production issue is
known. No real-model conversion was repeated for this ownership-only move.

At resume, rerun the read-only unconsumed-result inventory. Select the next
smallest source-adjacent, semantically closed cluster whose children are
already pass-module-owned, characterize it before changing production, and
keep all testing sequential and single-process under `uv`. Commit and push
only; never create, update, or reopen a pull request.

## Late affine/Concat composite characterization checkpoint

The next two-stage candidate is characterized but not implemented. It joins
indexed Conv Mul/Add-affine folding with the existing late Concat/layout
composite. The first child receives the shared ModelIR and LayoutState with
`enable_conv_add_only_fold=True`; the second receives the exact shared
`ModelIRPassContext`.

The fixed predecessor is the `cleanup.late.ndhwc_cost_volume` phase record.
The fixed successor is the optional `optimize_layout_transpose_chains` fanout
guard. The new owner must not absorb the guard, change its policy, flatten the
Concat tuple, or copy either raw result.

Focused characterization reports `1 passed, 1 xfailed`; complete affected
characterization reports `376 passed, 1 xfailed`. The sole xfail is the
intentionally absent `passes/late_affine_concat_orchestration.py`. Production
and the 128-ID/128-owner store are unchanged.

At resume, implement `run_late_affine_concat_cleanup(context)` as a
straight-line two-stage owner with the fixed affine flag. Return the affine
mapping and raw four-result Concat tuple unchanged, replace only the two
unconsumed lowerer locals, and preserve both outer boundaries. Run affected
and standard gates sequentially, then commit and push only. Never create,
update, or reopen a pull request.

## Late affine/Concat composite implementation checkpoint

The characterized two-stage owner is now implemented in
`passes/late_affine_concat_orchestration.py`. It forwards the shared model and
LayoutState to indexed Conv Mul/Add-affine folding with the original fixed
`enable_conv_add_only_fold=True` policy, then forwards the exact shared
`ModelIRPassContext` to the existing late Concat/layout composite. It returns
the affine mapping and the complete nested Concat tuple unchanged and in the
original order.

The lowerer replaces only the two unconsumed observation locals with
`_late_affine_concat_results`. The `cleanup.late.ndhwc_cost_volume` phase
record remains the immediate predecessor; the optional elementwise-fanout
guard remains the immediate successor. Independent affine routes,
compatibility wrappers, child owners, graph mutation behavior, and the
128-ID/128-owner phase store remain unchanged.

Sequential core-only `uv` validation passed: focused 3, complete affected
378, terminal-layout/efficiency 92, core 55, result contracts 196,
phase-store 2, and TensorFlow import-blocking/default-direct/`-cotof` 11.
Owner-aware structural coverage and runtime callback injection prove exact
order, state identity, fixed policy, nested schema, outer boundaries, and raw
result identity. No test is failing and no new production issue is known. No
real-model conversion was repeated for this ownership-only move.

At resume, rerun the read-only inventory of remaining unconsumed lowerer
results. Characterize the next smallest source-adjacent, semantically closed
cluster before changing production, and keep all validation sequential and
single-process under `uv`. Commit and push each completed checkpoint. Never
create, update, or reopen a pull request.

## Pre-terminal affine/Slice/SPP composite characterization checkpoint

The read-only lowerer inventory found 61 remaining unconsumed underscore
assignment targets. Earlier candidates in the late-layout tail were skipped
because they cross optional guards or still depend on a lowerer-local indexed
convergence helper. The first eligible adjacent pair whose children already
have pass-module owners is `run_pre_terminal_cleanup(context)` followed by
`run_terminal_affine_slice_spp_cleanup(context)`.

The contract requires one shared-context owner, exact two-child order, the
same `ModelIRPassContext` object, and preservation of the complete nested
five-result and three-result schemas without copying or flattening. The
preceding `_late_binary_layout_recovery_requires_reconciliation` guard and the
following terminal QKV shape/attention composite are fixed outer boundaries
and must remain outside the new owner.

Focused characterization reports `1 passed, 1 xfailed`; complete affected
characterization reports `512 passed, 1 xfailed`. The sole xfail is the
intentionally absent
`passes/pre_terminal_affine_slice_spp_orchestration.py`. Eight pre-existing
stale AST expectations were corrected to resolve terminal QKV and later
Shape-extract operations through their already-implemented owners. No
production source, mutation route, or phase-store entry changed; the store
remains exactly 128 IDs and 128 owners.

At resume, implement
`run_pre_terminal_affine_slice_spp_cleanup(context)` as a straight-line
two-child owner. Return both raw nested tuples unchanged, replace only
`_pre_terminal_cleanup_results` and `_terminal_affine_slice_spp_results`, and
preserve both outer boundaries. Update child-family ownership tests, add
runtime order/context/result-identity injection, run affected and standard
gates sequentially, then commit and push only. Never create, update, or
reopen a pull request.

## Pre-terminal affine/Slice/SPP composite implementation checkpoint

`passes/pre_terminal_affine_slice_spp_orchestration.py` now owns the fixed
two-child sequence. It passes the exact shared `ModelIRPassContext` to the
existing pre-terminal cleanup and terminal affine/Slice/SPP owners in source
order and returns both nested raw tuples without copying or flattening them.

The lowerer replaces only `_pre_terminal_cleanup_results` and
`_terminal_affine_slice_spp_results` with one
`_pre_terminal_affine_slice_spp_results` tuple. The preceding optional
late-binary layout-recovery reconciliation guard and following terminal QKV
shape/attention composite remain separate. Child owners, raw wrappers,
callbacks, independent routes, graph behavior, and the 128-ID/128-owner phase
store are unchanged.

Sequential core-only `uv` validation passed: focused 3, complete affected
514, terminal-layout/efficiency 92, core 55, result contracts 196,
phase-store 2, and TensorFlow import-blocking/default-direct/`-cotof` 11.
Owner-aware structural coverage and runtime injection prove exact order,
shared-context identity, nested schemas, outer boundaries, independent route
counts, and both raw-result identities. No test is failing and no new
production issue is known. No real-model conversion was repeated for this
ownership-only move.

At resume, rerun the read-only inventory of remaining unconsumed lowerer
results. Select the next smallest source-adjacent, semantically closed cluster
whose children already have pass-module owners. Characterize it before any
production change, keep all tests sequential and single-process under `uv`,
and commit/push completed checkpoints only. Never create, update, or reopen a
pull request.

## Terminal QKV/activation-bridge characterization checkpoint

The current inventory contains 60 unconsumed underscore assignment targets.
The next characterized pair is
`run_terminal_qkv_shape_attention_cleanup(context, ...)` followed by
`run_terminal_activation_bridge_cleanup(context, ...)`. Both receive the
exact shared `ModelIRPassContext` and the same layout-Transpose option.

The contract requires one two-child owner, fixed order, unchanged option
forwarding, and identity preservation for the nested two-result QKV tuple and
three-result activation tuple. The pre-terminal affine/Slice/SPP composite is
the fixed predecessor; terminal layout/shape is the fixed successor. Neither
outer boundary may be absorbed.

Focused characterization reports `2 passed, 1 xfailed`; complete affected
characterization reports `532 passed, 1 xfailed`. The sole xfail is the
intentionally absent
`passes/terminal_qkv_activation_bridge_orchestration.py`. Eleven stale AST
expectations were corrected to resolve already-moved predecessor, activation,
and terminal layout/shape operations through their existing owners.
Production, graph mutations, independent routes, and the 128-ID/128-owner
phase store are unchanged.

At resume, implement
`run_terminal_qkv_activation_bridge_cleanup(context, *,
include_layout_transpose)` as a straight-line two-child owner. Return both raw
nested tuples unchanged, replace only `_terminal_qkv_shape_attention_results`
and `_terminal_activation_bridge_results`, preserve both outer boundaries,
add runtime order/context/option/result-identity injection, and update
child-family ownership tests. Run all affected and standard gates
sequentially, then commit and push only. Never create, update, or reopen a
pull request.

## Terminal QKV/activation-bridge implementation checkpoint

`passes/terminal_qkv_activation_bridge_orchestration.py` now owns the fixed
QKV-then-activation sequence. Both existing children receive the exact shared
`ModelIRPassContext` and the unchanged layout-Transpose option. Their complete
two-result and three-result tuples are returned unchanged in one ordered outer
tuple, preserving both child identities and all nested schemas.

The lowerer replaces only the two unconsumed child locals with
`_terminal_qkv_activation_bridge_results`. Pre-terminal affine/Slice/SPP and
terminal layout/shape remain separate immediate boundaries. Specialized child
owners, independent and compatibility routes, callbacks, guards, graph
behavior, public behavior, and TensorFlow isolation remain unchanged. The
unconsumed assignment inventory is now 59; the phase-result store remains
exactly 128 IDs and 128 owners.

Sequential core-only `uv` validation passed: focused 5, complete affected 535,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff, bytecode
compilation, and whitespace checks pass. Runtime injection proves exact
order, shared-context identity, option forwarding, nested schemas, outer
boundaries, and both raw-result identities. No test is failing and no new
production issue is known. No real-model conversion was repeated for this
ownership-only move.

At resume, rerun the read-only inventory of the 59 remaining unconsumed
lowerer results. Characterize the next smallest source-adjacent, semantically
closed cluster whose children already have pass-module owners before changing
production. Keep all tests sequential and single-process under `uv`, and
commit/push completed checkpoints only. Never create, update, or reopen a pull
request.

## Terminal QKV/activation/layout/shape characterization checkpoint

The current inventory contains 59 unconsumed underscore assignment targets.
The next characterized pair is the terminal QKV/activation composite followed
by terminal layout/shape. Both receive the exact shared `ModelIRPassContext`
and the same layout-Transpose option, with no intervening guard, phase record,
progress update, callback, or result consumer.

The contract requires one two-child owner, fixed order, unchanged option
forwarding, and identity preservation for both complete nested child tuples.
Pre-terminal affine/Slice/SPP is the fixed predecessor. The recorded terminal
Expand/Squeeze static-shape reconciliation and its following progress update
are fixed successors and must remain outside the owner.

Focused characterization reports `2 passed, 1 xfailed`; complete affected
characterization reports `507 passed, 1 xfailed`. The sole xfail is the
intentionally absent
`passes/terminal_qkv_activation_layout_shape_orchestration.py`. Three stale
AST expectations in two late-layout test modules were corrected to resolve
the already-moved activation route through its current QKV/activation owner.
Production, graph mutations, independent routes, and the 128-ID/128-owner
phase store are unchanged.

At resume, implement
`run_terminal_qkv_activation_layout_shape_cleanup(context, *,
include_layout_transpose)` as a straight-line two-child owner. Return both raw
nested tuples unchanged, replace only
`_terminal_qkv_activation_bridge_results` and
`_terminal_layout_shape_results`, preserve both outer boundaries, add runtime
order/context/option/result-identity injection, and update child-family
ownership tests. Run all affected and standard gates sequentially, then
commit and push only. Never create, update, or reopen a pull request.

## Terminal QKV/activation/layout/shape implementation checkpoint

`passes/terminal_qkv_activation_layout_shape_orchestration.py` now owns the
fixed QKV/activation-then-layout/shape sequence. Both existing composites
receive the exact shared `ModelIRPassContext` and unchanged layout-Transpose
option. Their complete nested tuples are returned unchanged in one ordered
outer tuple, preserving both child identities and all result schemas.

The lowerer replaces only the two unconsumed child locals with
`_terminal_qkv_activation_layout_shape_results`. Pre-terminal
affine/Slice/SPP remains the immediate predecessor. The phase-recorded terminal
Expand/Squeeze reconciliation and following progress update remain separate
immediate successors. Specialized child owners, independent and compatibility
routes, callbacks, guards, graph behavior, progress behavior, public behavior,
and TensorFlow isolation remain unchanged. The unconsumed assignment inventory
is now 58; the phase-result store remains exactly 128 IDs and 128 owners.

Sequential core-only `uv` validation passed: focused 5, complete affected 510,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff, bytecode
compilation, and whitespace checks pass. Runtime injection proves exact order,
shared-context identity, option forwarding, nested schemas, phase/progress
boundaries, and both raw-result identities. No test is failing and no new
production issue is known. No real-model conversion was repeated for this
ownership-only move.

At resume, rerun the read-only inventory of the 58 remaining unconsumed
lowerer results. Characterize the next smallest source-adjacent, semantically
closed cluster whose children already have pass-module owners before changing
production. Keep all tests sequential and single-process under `uv`, and
commit/push completed checkpoints only. Never create, update, or reopen a pull
request.

## Late dequant hard-sigmoid/unary characterization checkpoint

The current inventory contains 58 unconsumed underscore assignment targets.
The next characterized pair is late transpose/dequant/hard-sigmoid/quantize
bridge cleanup followed by the existing three-stage late
dequant/unary/fan-out composite. The current calls use a model-only
compatibility wrapper and a zero-argument helper backed by the exact shared
`ModelIRPassContext`.

The contract requires one two-child owner, fixed order, exact shared context,
and identity preservation for the bridge mapping and nested three-result
tuple. The preceding layout/no-layout conditional and following swish
transpose-passthrough call remain fixed outer boundaries. Both existing
lowerer wrappers remain compatibility routes outside the new owner.

Focused characterization reports `1 passed, 1 xfailed`; complete affected
characterization reports `374 passed, 1 xfailed`. The sole xfail is the
intentionally absent
`passes/late_dequant_hardsigmoid_unary_orchestration.py`. Production, graph
mutations, wrappers, independent routes, and the 128-ID/128-owner phase store
are unchanged.

At resume, implement
`run_late_dequant_hardsigmoid_unary_cleanup(context)` as a straight-line
two-child owner. Call the public hard-sigmoid bridge pass with
`context.model_ir`, pass the same context to late dequant/unary/fan-out, return
both raw results unchanged, replace only the two characterized locals, and
preserve both outer boundaries and wrappers. Add runtime
order/context/result-identity injection, run affected and standard gates
sequentially, then commit and push only. Never create, update, or reopen a
pull request.

## Late dequant hard-sigmoid/unary implementation checkpoint

`passes/late_dequant_hardsigmoid_unary_orchestration.py` now owns the fixed
hard-sigmoid-bridge-then-unary/fan-out sequence. The public bridge owner
receives `context.model_ir`; the existing three-stage composite receives the
exact same shared `ModelIRPassContext`. Their mapping and nested tuple are
returned unchanged in one ordered outer tuple, preserving both raw identities
and every result schema.

The lowerer replaces only the two unconsumed child locals with
`_late_dequant_hardsigmoid_unary_results`. The layout/no-layout conditional and
following swish passthrough remain separate immediate boundaries. The
model-only hard-sigmoid wrapper and zero-argument fan-out helper remain intact
as compatibility routes. Independent callers, callbacks, graph behavior,
public behavior, and TensorFlow isolation remain unchanged. The unconsumed
assignment inventory is now 57; the phase-result store remains exactly 128 IDs
and 128 owners.

Sequential core-only `uv` validation passed: focused 3, complete affected 376,
terminal-layout/efficiency 92, core 55, result contracts 196, phase-store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff, bytecode
compilation, and whitespace checks pass. Runtime injection proves exact order,
model/context identity, nested schemas, wrapper preservation, outer boundaries,
and both raw-result identities. No test is failing and no new production issue
is known. No real-model conversion was repeated for this ownership-only move.

At resume, rerun the read-only inventory of the 57 remaining unconsumed
lowerer results. Characterize the next smallest source-adjacent, semantically
closed cluster whose children already have pass-module owners before changing
production. Keep all tests sequential and single-process under `uv`, and
commit/push completed checkpoints only. Never create, update, or reopen a pull
request.

## Late swish/very-late layout-tail characterization checkpoint

The current inventory contains 57 unconsumed underscore assignment targets.
The next characterized pair is late swish transpose-passthrough cleanup
followed by the existing four-stage very-late layout-tail composite. The
current calls share the same ModelIR/LayoutState; the tail additionally
receives the exact shared `ModelIRPassContext` and unchanged layout-Transpose
option.

The contract requires one two-child owner, fixed order, exact shared state and
context, unchanged option forwarding, and identity preservation for the swish
mapping and complete nested tail tuple. The preceding late dequant
hard-sigmoid/unary composite and following phase-recorded very-late broadcast
reconciliation remain fixed outer boundaries. The existing swish lowerer
wrapper remains a compatibility route.

Focused characterization reports `2 passed, 1 xfailed`; complete affected
characterization reports `476 passed, 1 xfailed`. The sole xfail is the
intentionally absent `passes/late_swish_layout_tail_orchestration.py`. Two
stale terminal-affine AST expectations were corrected to resolve the already
implemented terminal QKV/activation/layout/shape boundary. Production, graph
mutations, wrappers, independent routes, and the 128-ID/128-owner phase store
are unchanged.

At resume, implement `run_late_swish_layout_tail_cleanup(context, *,
include_layout_transpose)` as a straight-line two-child owner. Call the public
swish pass with the context model and LayoutState, pass the same context and
option to the tail, return both raw results unchanged, replace only the two
characterized locals, and preserve both outer boundaries and the swish
wrapper. Add runtime order/state/context/option/result-identity injection, run
affected and standard gates sequentially, then commit and push only. Never
create, update, or reopen a pull request.

## Late swish/very-late layout-tail implementation checkpoint

`passes/late_swish_layout_tail_orchestration.py` now implements the fixed
swish-then-layout-tail boundary. It passes `context.model_ir` and the shared
LayoutState to the public swish owner, then passes the exact same shared
context and unchanged layout-Transpose option to the existing very-late tail.
Both complete raw results are returned unchanged and in their original order.

The lowerer now retains one observation-only
`_late_swish_layout_tail_results` value in place of the two characterized child
locals. The preceding late dequant hard-sigmoid/unary composite and following
phase-recorded very-late broadcast reconciliation remain immediate outer
boundaries. The lowerer swish wrapper, nested child owners, independent routes,
guards, graph behavior, public behavior, and TensorFlow isolation remain
unchanged. The characterized unconsumed-result inventory is now 56, and the
phase store remains exactly 128 IDs and 128 owners.

Sequential validation passed: focused 5, complete affected 479,
terminal-layout/efficiency 92, core 55, result contracts 196, phase store 2,
and TensorFlow import-blocking/default-direct/`-cotof` 11. Ruff passed for all
affected Python files. Runtime injection covers both layout-policy values and
proves exact execution order, model/LayoutState/context identity, option
forwarding, full nested schemas, and raw-result identity. No production test
or known issue is failing, and no real-model conversion was repeated for this
ownership-only extraction.

At resume, rerun the read-only inventory of the 56 remaining characterized
unconsumed lowerer results and select the next smallest source-adjacent,
semantically closed cluster whose children already have pass-module owners.
Characterize it before changing production, keep every test sequential under
`uv`, and commit/push only at complete checkpoints. Do not create or modify a
pull request unless the user explicitly requests that operation in the active
turn.

## Terminal affine/QKV/layout-shape characterization checkpoint

The current characterized inventory contains 56 unconsumed lowerer results.
The next selected adjacent pair is pre-terminal affine/Slice/SPP followed by
terminal QKV/activation/layout/shape. Both use the exact shared context, and
the second child receives the unchanged layout-Transpose option. There is no
intervening phase record, branch, progress update, result consumer, or other
mutation.

The contract requires one two-child owner with fixed order, exact context and
option forwarding, and identity preservation for both complete nested result
tuples. The optional late-binary-layout reconciliation branch remains the
predecessor. The terminal Expand/Squeeze reconciliation phase record and
post-progress callback remain successors outside the owner.

Focused characterization reports `2 passed, 1 xfailed`; the expanded affected
suite reports `1140 passed, 1 xfailed`. The sole xfail is the intentionally
absent `passes/terminal_affine_qkv_layout_shape_orchestration.py`. The expanded
suite also identified 24 stale tests still naming the already-replaced
terminal QKV/shape/attention boundary. Their expectations now resolve the
current terminal QKV/activation/layout/shape owner without changing production.
The phase store remains exactly 128 IDs and 128 owners.

At resume, implement
`run_terminal_affine_qkv_layout_shape_cleanup(context, *,
include_layout_transpose)` as a straight-line owner. Pass the exact context to
both children, forward the option only to the terminal child, return both raw
results unchanged, and replace only the two characterized lowerer locals.
Preserve the guard, phase record, progress callback, and all child owners. Add
runtime order/context/option/result-identity injection, run affected and
standard gates sequentially, then commit and push only. Do not create, update,
or reopen a pull request.

## Very-late SiNet recovery-tail characterization checkpoint

After the terminal Clamp/SiNet extraction, the characterized inventory is 54
unconsumed lowerer-result assignments. The next safe boundary is the adjacent
absolute-end SiNet terminal-layout recovery and pre-add/resize callback. The
recorded `shape_topology.terminal.indexed_convergence` call is the predecessor;
the recorded `cleanup.very_late.residual_affine_prelu` call is the successor.
There is no intervening decision, phase-store write, progress callback, or
consumer.

Focused characterization reports `1 passed, 1 xfailed`; the affected suite
reports `456 passed, 1 xfailed`. The only xfail requires the intentionally
absent `passes/very_late_sinet_recovery_tail_orchestration.py`. The tests fix
the exact shared `SINetTerminalLayoutRecoveryContext`, its original
pre-add/resize callback, source order, both complete nested schemas, and the
unchanged outer boundaries. Production and the 128-ID/128-owner phase store
are unchanged.

At resume, implement
`run_very_late_sinet_recovery_tail_cleanup(context)` as a straight-line owner:
call `run_sinet_terminal_layout_recovery(context)`, then
`context.preadd_resize_recovery()`, and return both raw results unchanged.
Replace only the two characterized lowerer locals, retain both zero-argument
wrappers and all independent routes, update only stale direct-boundary tests,
and run all affected and standard gates sequentially. Commit and push the
completed unit only. Do not create, update, reopen, or otherwise modify a pull
request.

## Very-late SiNet recovery-tail implementation checkpoint

`run_very_late_sinet_recovery_tail_cleanup(context)` is implemented in
`passes/very_late_sinet_recovery_tail_orchestration.py`. It runs the existing
terminal-layout owner with the exact context and then invokes the exact
context-owned pre-add/resize callback. Both raw results, their order, and their
identities are preserved. The lowerer now has one
`_very_late_sinet_recovery_tail_results` observation in place of the two
characterized locals; both lowerer wrappers and all independent recovery
routes remain available.

Focused, owner-aware, affected, and standard sequential gates pass with
`3`, `277`, `458`, and `92 / 55 / 196 / 2 / 11` tests respectively. The
result gate initially exposed three additional stale neighbor-target
assertions; they now identify the composite without changing production. The
phase store remains exactly 128 IDs and 128 owners, and the characterized
unconsumed lowerer-result inventory is now 53. No real-model conversion was
repeated for this exact orchestration-only extraction.

At resume, refresh the 53-result inventory and select the next smallest
straight-line observation-only boundary. Characterize its state, guards,
schemas, independent routes, and outer phase/progress boundaries before any
production change. Continue to use core-only `uv`, run tests sequentially,
commit and push only at complete checkpoints, and do not create, update,
reopen, or otherwise modify a pull request.

## Terminal SiNet/singleton-Reshape characterization checkpoint

The current inventory is 53 unconsumed lowerer-result assignments. The next
safe pair is terminal SiNet pre-add/resize recovery followed immediately by
post-terminal singleton-Reshape recovery. Both context aliases are the same
`shared_model_ir_pass_context`. The second child alone has the fixed duplicate
fan-out/spatial-Concat policy. Recorded terminal dequant/hard-sigmoid cleanup
and terminal indexed-shape convergence remain the outer phase boundaries.

Focused characterization reports `1 passed, 1 xfailed`; the affected suite
reports `455 passed, 1 xfailed`. The sole xfail requires the intentionally
absent `passes/terminal_sinet_singleton_reshape_orchestration.py`. The complete
six-result and eight-result schemas, exact shared identity, options, wrapper
routes, and phase adjacency are fixed. Production and the 128-ID/128-owner
phase store are unchanged.

At resume, implement
`run_terminal_sinet_singleton_reshape_cleanup(context)` as a straight-line
owner. Call `run_sinet_preadd_resize_recovery(context)`, then
`run_singleton_reshape(context, include_duplicate_fanout=True,
include_spatial_concat_post_transpose=False)`, returning both raw tuples
unchanged. Replace only the two characterized lowerer locals, retain the
wrappers and independent routes, update only stale boundary contracts, and
run affected and standard gates sequentially. Commit and push only; do not
create, update, reopen, or otherwise modify a pull request.

## Terminal SiNet/singleton-Reshape implementation checkpoint

`run_terminal_sinet_singleton_reshape_cleanup(context)` is implemented in
`passes/terminal_sinet_singleton_reshape_orchestration.py`. It passes the exact
shared context through SiNet recovery and singleton-Reshape recovery in source
order and fixes the second child's duplicate-fanout/spatial-Concat options.
Both raw tuples and identities are preserved. The lowerer now has one
`_terminal_sinet_singleton_reshape_results` observation in place of the two
characterized locals; wrappers and independent routes remain available.

Focused, owner-aware, affected, and standard sequential gates pass with
`3`, `314`, `457`, and `92 / 55 / 196 / 2 / 11` tests respectively. Eleven
stale direct-call or neighbor assertions now resolve the composite and still
verify every independent route and fixed policy. The phase store remains
exactly 128 IDs and 128 owners, and the characterized unconsumed lowerer-result
inventory is now 52. No real-model conversion was repeated for this exact
orchestration-only extraction.

At resume, refresh the 52-result inventory and select the next smallest
straight-line observation-only boundary. Characterize exact state, options,
guards, schemas, independent routes, and outer phase/progress boundaries before
production changes. Continue with core-only `uv`, sequential tests, and
complete checkpoint commits/pushes only. Do not create, update, reopen, or
otherwise modify a pull request.

## Late dequant/swish-layout-tail characterization checkpoint

The current inventory is 52 unconsumed lowerer-result assignments. The next
safe pair is late dequant/hard-sigmoid/unary cleanup followed immediately by
late swish/very-late-layout-tail cleanup. Both use the exact
`shared_model_ir_pass_context`, and the second alone receives the normalized
layout-Transpose option. The preceding layout/no-layout branch and following
recorded very-late broadcast reconciliation remain outside the owner.

Focused characterization reports `2 passed, 1 xfailed`; the affected suite
reports `432 passed, 1 xfailed`. The sole xfail requires the intentionally
absent `passes/late_dequant_swish_layout_tail_orchestration.py`. Both policy
paths, shared identity, complete nested outer schemas, options, independent
routes, and phase adjacency are fixed. Production and the 128-ID/128-owner
phase store are unchanged.

At resume, implement
`run_late_dequant_swish_layout_tail_cleanup(context, *,
include_layout_transpose)` as a straight-line owner. Call the existing late
dequant child first and late swish/layout-tail child second, forward the option
only to the second, and return both raw results unchanged. Replace only the two
characterized lowerer locals, retain nested owners and wrappers, update only
stale boundary contracts, and run affected and standard gates sequentially.
Commit and push only; do not create, update, reopen, or otherwise modify a pull
request.

## Late dequant/swish-layout-tail implementation checkpoint

`run_late_dequant_swish_layout_tail_cleanup(context, *,
include_layout_transpose)` is implemented in
`passes/late_dequant_swish_layout_tail_orchestration.py`. It preserves the
exact shared context, dequant-before-swish order, second-child-only option, and
both complete nested result identities. The lowerer has one
`_late_dequant_swish_layout_tail_results` observation in place of the two
characterized locals; child owners, wrappers, and independent routes remain
available.

Focused, owner-aware, affected, and standard sequential gates pass with
`5`, `373`, `435`, and `92 / 55 / 196 / 2 / 11` tests respectively. Thirty-six
stale entry-point assertions now resolve the outer composite while retaining
direct validation of every nested pass family. The phase store remains exactly
128 IDs and 128 owners, and the characterized unconsumed lowerer-result
inventory is now 51. No real-model conversion was repeated for this exact
orchestration-only extraction.

At resume, refresh the 51-result inventory and select the next smallest
straight-line observation-only boundary. Characterize state, options, guards,
schemas, nested and independent routes, and outer phase/progress boundaries
before production changes. Continue with core-only `uv`, sequential tests,
and complete checkpoint commits/pushes only. Do not create, update, reopen, or
otherwise modify a pull request.

## Late final shape/boundary characterization checkpoint

The refreshed AST inventory confirms 51 unconsumed lowerer-result assignments.
The next selected boundary is the adjacent late reshape/shuffle/attention/window
composite, indexed final shape/activation convergence, and final boundary
Slice/Concat composite. The first stage receives the shared `ModelIRPassContext`,
the middle stage receives its exact ModelIR and LayoutState while reusing one
`ModelIRGraphIndex`, and the final stage receives the existing terminal
Slice/Concat recovery context whose pass context is the same session context.

The optional late Concat elementwise-fanout branch remains the immediate
predecessor, and the optional terminal elementwise-fanout branch remains the
immediate successor. The characterization fixes the three-stage order, both
context identities, all raw nested result schemas, the eleven-key convergence
mapping, the callback-bearing recovery context, and both outer guards.
Production is unchanged pending extraction of the convergence logic to a
dedicated pass owner and one straight-line three-stage composite owner.

Focused characterization reports `1 passed, 1 xfailed`; the complete
reference-based affected suite reports `402 passed, 1 xfailed`. The sole
expected failure requires the intentionally absent convergence/composite
owner modules.

At resume, implement the dedicated convergence owner with lowerer compatibility
wrappers, implement the composite context owner, replace only the three
observation-only lowerer locals, and update stale structural entry assertions.
Preserve exact GraphIndex sharing, result identities, callbacks, guards, and
the 128-ID/128-owner phase-result store. Commit and push only; do not create,
update, reopen, or otherwise modify a pull request.

## Late final shape/boundary implementation checkpoint

`passes/indexed_final_shape_activation_convergence.py` now owns both indexed
shape convergence and final shape/activation convergence. The implementation
still builds exactly one `ModelIRGraphIndex` for final convergence and forwards
that same object through pruning, static-shape reconciliation, dynamic Reshape
resolution, HardSwish metadata repair, and activation fusion. Mutation guards,
the fusion prune fallback, and the exact three-key and eleven-key result schemas
are unchanged. Both former lowerer functions remain as one-return compatibility
wrappers.

`passes/late_final_shape_boundary_orchestration.py` adds the frozen
`LateFinalShapeBoundaryContext` and the straight-line
`run_late_final_shape_boundary_cleanup()` owner. It invokes late
reshape/shuffle/attention/window cleanup with the shared pass context, final
shape/activation convergence with that context's exact ModelIR and LayoutState,
and final boundary Slice/Concat cleanup with the original callback-bearing
terminal context. All three raw results are returned unchanged and in source
order.

The lowerer replaces the three observation-only locals with
`_late_final_shape_boundary_results`. The optional late and terminal
elementwise-fanout guards remain the immediate outer boundaries. Thirty-one
stale structural assertions now traverse the new outer owner while continuing
to validate the two nested composites, the dedicated convergence owner, all
compatibility wrappers, and independent routes.

Sequential core-only validation passes: focused convergence/composite `18`,
complete affected `404`, terminal-layout/efficiency `92`, core `55`, result
contracts `196`, phase-store `2`, and TensorFlow import-blocking/default-direct/
`-cotof` `11`. Ruff, bytecode compilation, whitespace checks, exact result
identity injection, legacy-sequence equivalence, and one-index reuse all pass.
The phase store remains exactly 128 IDs and 128 owners. The characterized
unconsumed lowerer-result inventory decreases from 51 to 49. No real-model
conversion was repeated for this ownership-only extraction.

At resume, refresh the 49-result inventory and select the next smallest
source-adjacent, semantically closed observation-only boundary. Characterize
its context, option, guard, callback, result-schema, independent-route, and
outer-boundary contracts before changing production. Continue with sequential
core-only `uv` validation and complete checkpoint commits/pushes only. Do not
create, update, reopen, or otherwise modify a pull request.

## Fallback precision/unbound characterization checkpoint

The refreshed 49-result inventory selects the two adjacent unconditional
observations immediately after the fallback post-placeholder topology
checkpoint: precision cleanup followed by unbound-input repair. Both operate
on the exact `fallback_ir`; precision cleanup additionally receives no
LayoutState and the current session diagnostics, while unbound repair retains
its model-only contract. The following indexed Conv-input summary does not
consume either result.

The characterization fixes the topology phase predecessor, exact child order,
ModelIR identity, no-layout precision policy, diagnostics source, complete
three-mapping precision tuple, one-key unbound mapping, observation-only
status, and Conv-input successor. Production is unchanged pending one frozen
`ModelIRPassContext` and straight-line two-child owner.

Focused characterization reports `1 passed, 1 xfailed`; the complete
reference-based affected suite reports `29 passed, 1 xfailed`. The sole
expected failure requires the intentionally absent owner module.

At resume, implement the owner with direct public child imports, retain both
lowerer compatibility wrappers and the final-primary precision route, update
stale entry assertions, and run all gates sequentially. Commit and push only;
do not create, update, reopen, or otherwise modify a pull request.

## Fallback precision/unbound implementation checkpoint

`passes/fallback_precision_unbound_orchestration.py` now owns the characterized
pair. It receives one `ModelIRPassContext` whose model is the recursively
rebuilt `fallback_ir`, whose LayoutState is explicitly `None`, and whose
diagnostics object is the active session list. It invokes the existing public
precision owner first and model-only unbound-input repair second, returning the
complete three-mapping tuple and one-key mapping unchanged and by identity.

The fallback context is created immediately after recursive relowering, so the
post-placeholder topology checkpoint remains the direct execution predecessor.
The lowerer replaces only the two observation-only child locals with
`_fallback_precision_unbound_results`; indexed Conv-input repair remains the
direct successor. Both lowerer compatibility wrappers, the final-primary
precision route, and the unbound repair module remain available.

Sequential validation passes: focused `3`, affected `31`, and standard
`92 / 55 / 196 / 2 / 11`. Ruff, bytecode compilation, whitespace checks,
runtime context/order/result-identity injection, TensorFlow import blocking,
default direct conversion, and `-cotof` all pass. The phase store remains
exactly 128 IDs and 128 owners, while the characterized unconsumed
lowerer-result inventory decreases from 49 to 48. No real-model conversion was
repeated for this straight-line ownership extraction.

At resume, refresh the 48-result inventory and select the next smallest
source-adjacent semantically closed boundary. Preserve fallback recursion,
guards, result-driven reconciliation, and all independent wrappers. Continue
with sequential core-only `uv` tests and complete commits/pushes only. Do not
create, update, reopen, or otherwise modify a pull request.

## Next candidate audit (not yet characterized)

The read-only 48-result audit selects the adjacent layout-pass-set-1 mean/
attention and attention/gate/QDQ observations. They are unconditional within
the existing `optimize_layout_transpose_chains` guard and have no intervening
phase, progress, or result consumer. The first currently calls the mean/
attention owner with `include_layernorm=True` and the default enabled Conv
attention policy. The second receives the existing `AttentionRecoveryContext`,
which embeds the exact same `ModelIRPassContext` plus the original gate and
transpose-unary callback identities.

The recorded mean-affine-prepost phase is the immediate predecessor, and the
recorded quantized-PRELU phase is the immediate successor. A safe owner can
therefore accept the existing `AttentionRecoveryContext`, call
`run_mean_attention(context.pass_context, include_layernorm=True)` first, call
`run_attention_gate_qdq_recovery(context)` second, and return both complete raw
tuples unchanged. Do not implement this directly on resume: first add a strict
characterization covering the two policy defaults, embedded pass-context and
callback identity, child schemas, outer phase IDs, observation-only status,
and independent wrapper routes. No production or test change was made during
this audit.

## Layout-pass-set-1 mean/attention gate characterization checkpoint

The strict characterization now fixes the current two-assignment boundary.
It proves that both calls remain consecutive inside the sole
`optimize_layout_transpose_chains` guard, are observation-only, and are
surrounded immediately by the recorded
`cleanup.layout_pass_set_1.mean_affine_prepost` and
`cleanup.layout_pass_set_1.quantized_prelu` phases. Mean/attention receives
`include_layernorm=True` while omitting `include_conv_attention`, thereby
retaining its enabled default. Attention/gate/QDQ receives the existing
`AttentionRecoveryContext` with the exact session pass context and original
mean, gate, and transpose-unary callback identities.

The empty-ModelIR schema probe freezes all seven mean/attention mappings and
all ten attention/gate/QDQ slots, including the eight-result gate-layout and
three-result transpose-unary nested tuples. The proposed owner must call
`run_mean_attention(context.pass_context, include_layernorm=True)` and then
`run_attention_gate_qdq_recovery(context)`, return both raw tuples unchanged,
and replace only the two characterized lowerer locals. Both lowerer wrappers
and every independent route must remain available.

The expanded reference-based suite initially exposed six stale tests from
completed earlier work: two still counted direct fallback precision/unbound
wrappers after their composite extraction, while four assumed phase-recorded
owners were raw lowerer calls. Their expectations now follow the existing
fallback composite owner and unwrap `session.record_phase_result` only for
structural boundary inspection. No production source changed.

Sequential validation passes: focused `1 passed, 1 xfailed` and complete
affected `373 passed, 1 xfailed`. The sole expected failure requires the
intentionally absent
`passes/layout_pass_set_1_mean_attention_gate_orchestration.py`. Ruff,
bytecode compilation, and whitespace validation pass. The inventory remains
48, and the phase-result store remains unchanged.

At resume, implement the owner with direct public child imports, replace only
the two layout-pass-set-1 locals with
`_layout_pass_set_1_mean_attention_gate_results`, and convert the strict xfail
to runtime order/context/result-identity coverage. Update only stale structural
entry assertions, then run affected and standard gates sequentially under
`uv`. Commit and push only; do not create, update, reopen, or otherwise modify
a pull request.

## Layout-pass-set-1 mean/attention gate implementation checkpoint

`passes/layout_pass_set_1_mean_attention_gate_orchestration.py` now provides
`run_layout_pass_set_1_mean_attention_gate_cleanup()`. It accepts the existing
`AttentionRecoveryContext`, passes `context.pass_context` directly to
`run_mean_attention(..., include_layernorm=True)`, and deliberately omits the
`include_conv_attention` keyword so the characterized enabled default remains
authoritative. It then passes the exact original context to
`run_attention_gate_qdq_recovery()`. Both complete raw tuples are returned in
source order without copying, flattening, or normalization.

The lowerer replaces only
`_layout_pass_set_1_mean_attention_results` and
`_layout_pass_set_1_attention_gate_qdq_results` with
`_layout_pass_set_1_mean_attention_gate_results`. The recorded
`cleanup.layout_pass_set_1.mean_affine_prepost` and
`cleanup.layout_pass_set_1.quantized_prelu` calls remain the immediate
neighbors. Both lowerer wrappers and their independent pre-add, suffix,
layout-pass-set-2, and terminal routes remain available. No phase-result entry
or control-flow decision was added.

Runtime injection proves exact child order, pass-context and recovery-context
identity, LayerNorm policy forwarding, omission of the Conv policy override,
and both raw-result identities. Twelve stale structural assertions now count
the public child calls through the new owner and continue to cover all direct
and nested routes.

Sequential validation passes: focused `3`, complete affected `375`,
terminal-layout/efficiency `92`, core `55`, result contracts `196`, phase store
`2`, and TensorFlow import-blocking/default-direct/`-cotof` `11`. Ruff,
bytecode compilation, and whitespace validation pass. The phase-result store
remains exactly 128 IDs and 128 owners. The read-only unconsumed lowerer-result
inventory decreases from 48 to 47. No real-model conversion was repeated for
this straight-line ownership-only extraction.

At resume, refresh the 47-result inventory and select the next smallest
source-adjacent, semantically closed observation-only boundary. Characterize
all guards, options, context/callback identities, raw schemas, independent
routes, and outer phase/progress boundaries before changing production.
Continue with sequential `uv` validation and complete checkpoint commits and
pushes only. Do not create, update, reopen, or otherwise modify a pull request.

## Next 47-result candidate audit (not yet characterized)

The refreshed read-only inventory selects the first layout-pass-set-1
post-binary pair: the 13-stage layout/attention/quantized suffix followed
immediately by safe-binary recovery. Both observations are unconditional
inside the existing `optimize_layout_transpose_chains` guard, are not consumed,
and share the exact session `ModelIRPassContext`. There is no intervening
phase, progress update, branch, or other mutation.

The suffix receives the existing callback-bearing
`LayoutAttentionQuantizedSuffixContext` and the normalized
`enable_duplicate_transpose_fanout_optimizations` policy. Its embedded pass
context is identical to `quantized_recovery_context`, which the current
safe-binary wrapper uses. The recorded
`cleanup.layout_pass_set_1.post_binary_affine_chain_fold` phase is the direct
predecessor, and recorded
`cleanup.layout_pass_set_1.dequant_mean_quantize` is the direct successor.

A safe dedicated owner can therefore accept the existing suffix context and
`include_duplicate_transpose` option, call
`run_layout_attention_quantized_suffix(context,
include_duplicate_transpose=include_duplicate_transpose)` first, call
`run_safe_binary_recovery(context.pass_context)` second, and return both raw
tuples unchanged. Retain both lowerer wrappers and all independent suffix and
safe-binary routes. In particular, do not include the later final suffix and
safe-binary calls: a transpose-unary/fan-out stage lies between them, so they
are a different semantic boundary.

Before production changes, add strict characterization for the guard, option,
embedded pass-context and three callback identities, complete nested schemas,
raw-result identity requirement, phase neighbors, observation-only status,
and independent routes. The expected inventory reduction after a correct
implementation is 47 to 46. No production or test change was made during this
audit.

## Post-binary quantized-suffix/safe-binary characterization checkpoint

The strict characterization now covers both values of
`include_duplicate_transpose`. It fixes the current two wrapper calls and their
source adjacency, the exact normalized option expression, the shared session
pass context, `quantized_recovery_context` alias identity, and the suffix
context's mean, attention/gate/QDQ, and duplicate-PRELU callback identities.

The schema contract includes all 13 suffix slots and recursively freezes the
six-result mean tuple, ten-result attention/gate/QDQ tuple, option-dependent
two-result duplicate/PRELU tuple, and all remaining quantized mappings. It also
freezes the one-result safe-binary tuple. The recorded post-binary affine phase
and dequant-Mean phase remain the immediate outer boundaries, and the later
final suffix/transpose-unary/final-safe sequence remains a distinct contiguous
three-stage route.

Production is unchanged pending
`passes/layout_pass_set_1_attention_quantized_safe_binary_orchestration.py`.
The planned owner must accept the existing suffix context plus the normalized
duplicate-Transpose option, call the public suffix owner first, call
`run_safe_binary_recovery(context.pass_context)` second, and return both raw
results unchanged. Both lowerer wrappers and every independent route must
remain available.

Sequential validation passes: focused `2 passed, 1 xfailed` and complete
reference-based affected `426 passed, 1 xfailed`. The sole expected failure
requires the intentionally absent owner. Ruff, bytecode compilation, and
whitespace validation pass. The inventory remains 47 and the phase store is
unchanged.

At resume, implement the straight-line owner, replace only the first suffix
and safe-binary locals with one composite result, convert the xfail to runtime
order/context/option/result-identity injection, and update only stale entry
assertions. Run affected and standard gates sequentially under `uv`, then
confirm the 46-result inventory. Commit and push only; do not create, update,
reopen, or otherwise modify a pull request.

## Post-binary quantized-suffix/safe-binary implementation checkpoint

`passes/layout_pass_set_1_attention_quantized_safe_binary_orchestration.py`
now provides
`run_layout_pass_set_1_attention_quantized_safe_binary_cleanup()`. It receives
the original `LayoutAttentionQuantizedSuffixContext`, forwards it unchanged to
`run_layout_attention_quantized_suffix()` with the exact normalized
duplicate-Transpose policy, then passes `context.pass_context` directly to
`run_safe_binary_recovery()`. Both complete results are returned in source
order without copying, flattening, or normalization.

The lowerer replaces only
`_layout_pass_set_1_attention_quantized_suffix_results` and
`_layout_pass_set_1_safe_binary_results` with
`_layout_pass_set_1_attention_quantized_safe_binary_results`. The recorded
post-binary affine phase remains the direct predecessor and recorded
dequant-Mean phase remains the direct successor. Both lowerer wrappers and the
quantized-activation nested route remain available. The later final suffix,
transpose-unary/fan-out, and final safe-binary calls remain three separate
contiguous observations with their original policy and order.

Runtime injection covers both duplicate-Transpose policy values and proves
exact child order, suffix-context identity, embedded pass-context identity,
option forwarding, and both raw-result identities. Eleven stale structural
assertions now count the new public-child route and retain all independent
wrapper coverage.

Sequential validation passes: focused `5`, complete affected `429`,
terminal-layout/efficiency `92`, core `55`, result contracts `196`, phase store
`2`, and TensorFlow import-blocking/default-direct/`-cotof` `11`. Ruff,
bytecode compilation, and whitespace validation pass. The phase-result store
remains exactly 128 IDs and 128 owners. The read-only unconsumed lowerer-result
inventory decreases from 47 to 46. No real-model conversion was repeated for
this straight-line ownership-only extraction.

At resume, refresh the 46-result inventory and select the next smallest
source-adjacent, semantically closed observation-only boundary. Preserve all
result-driven branches, progress updates, phase records, wrappers, and
independent routes. Characterize before production changes, run every test
sequentially under `uv`, and commit/push only. Do not create, update, reopen,
or otherwise modify a pull request.
