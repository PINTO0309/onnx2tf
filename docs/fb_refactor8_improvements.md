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

## Fallback norm reconciliation characterization

After a positive fallback norm rewrite, the lowerer retains the exact and
singleton binary-adapter results plus the singleton/consecutive-Reshape tuple,
then discards one static-shape reconciliation dictionary before topological
sorting.

The new contract freezes the guarded four-statement order, the complete
two-key reconciliation schema, and a positive stale-Reshape metadata repair.
The existing explicit shape signature remains unchanged, matching the current
reconciler contract.

The selected implementation is observation-only assignment to
`_fallback_norm_static_shape_stats` with `include_mutation_count=True`.
Reconciliation remains unconditional inside the existing positive norm guard.
The preceding owners construct internal state that is not exposed to this
boundary, so graph-index sharing is not mixed into this result-retention unit.

Characterization validation completed sequentially under `uv`:

- dedicated contract: `2 passed, 1 xfailed in 0.56s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The first test run exposed only an incorrect top-level-only AST search for the
nested fallback guard. After selecting the unique guard recursively, the sole
expected failure is the intentionally unimplemented assignment. No production
source changed in this checkpoint.

## Fallback norm reconciliation result retention

The positive fallback norm branch now retains the opt-in complete result as
`_fallback_norm_static_shape_stats`. It remains unconsumed. The norm guard,
binary-adapter and singleton/Reshape predecessors, unconditional
reconciliation, topological-sort successor, arguments, and graph mutations are
unchanged. No graph index or additional pass is introduced.

Implementation validation completed sequentially under `uv`:

- dedicated result contract: `3 passed in 0.58s`;
- focused fallback owners, binary adapters, singleton/Reshape orchestration,
  shape reconciliation, terminal/core/architecture, pass-efficiency, and
  TensorFlow-import-blocked gate: `461 passed in 28.73s`;
- all test files changed on `fb-refactor8`: `38 passed in 1.13s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The first focused run was `460 passed, 1 failed` because an existing singleton
boundary test still required a raw reconciliation expression. Its structural
expectation now requires the new observation target and complete schema. No
model conversion was run because the result assignment does not affect ModelIR
or artifacts.

## Topology/layout refresh characterization

The post-lowering pipeline contains twenty-one raw topological-sort calls.
Six of them are immediately followed by logical-layout inference: two in the
recursive fallback and four in the primary absolute-final repair sequence.
The other fifteen sort-only boundaries have different successors and are not
included in this unit.

Topological sort returns the small fixed schema `reordered_operators` and
`cycle_detected`. Logical-layout inference returns a full tensor-name-to-layout
dictionary while also mutating tensor layout annotations and public-layout
metadata. Retaining that full dictionary in the lowerer would unnecessarily
extend its memory lifetime.

The selected design is a focused `run_topology_layout_refresh(model_ir)` owner
that executes the existing sort and layout inference in order, discards the
large layout-map return after its side effects, and returns only the unchanged
two-key sort result. The six call sites retain small observation-only results.
No sort-only site, layout algorithm, graph mutation, condition, or following
owner changes.

Characterization validation completed sequentially under `uv`:

- dedicated family contract: `2 passed, 1 xfailed in 0.56s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the absent shared owner and six assignments. No
production source changed in this checkpoint.

## Topology/layout refresh implementation

`passes/topology_layout_refresh.py` now owns the six selected adjacent
boundaries. It calls topological sort and logical-layout inference in the
unchanged order, does not retain the full layout map, and returns only the
normalized two-key sort result. Layout refresh still runs when sort detects a
cycle, matching the former unconditional pair.

The six lowerer positions retain phase-specific, unconsumed result dictionaries.
All fifteen sort-only boundaries remain raw and unchanged. No layout algorithm,
condition, graph mutation, successor, dependency, public API, or TensorFlow
boundary changed.

Implementation validation completed sequentially under `uv`:

- dedicated owner and six-boundary contract: `4 passed in 0.54s`;
- focused fallback/terminal, Dynamic Reshape, broadcast, ConvInteger,
  InstanceNorm, core, architecture, pass-efficiency, and
  TensorFlow-import-blocked gate: `493 passed in 28.99s`;
- all test files changed on `fb-refactor8`: `123 passed in 2.91s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The first focused run was `487 passed, 6 failed`; every failure was a stale AST
expectation for the intentionally replaced two-expression boundary. After the
six contracts required the new phase-specific result targets, the same gate
passed. No model conversion was run because the owner is differentially
equivalent to the existing adjacent calls.

## Static-shape/topology reconciliation owner

The follow-up audit classified the fifteen sort-only boundaries left after the
topology/layout refresh extraction. Eight of those boundaries had one exact
contract: a guarded static-shape reconciliation with
`include_mutation_count=True`, immediately followed by an unconditional
topological sort of the same `ModelIR`. They cover fallback norm and high-rank
BatchMatMul recovery plus the final high-rank BatchMatMul, Pad, Conv-input,
mixed-Concat, Concat-axis, and binary-layout repairs.

The characterization checkpoint fixed the eight locations, model arguments,
result targets, reconciliation result schema, sort result schema, mutation
order, and a graph where reconciliation and reordering are both observable.
It also exposed that the existing iterative reconciler reports three updates
for the deliberately stale Reshape fixture rather than one; that existing
convergence result is preserved exactly.

`passes/static_shape_reconciliation.py` now owns
`run_static_shape_topology_reconciliation(model_ir)`. The owner:

- runs the existing shape reconciler with mutation counting enabled;
- runs the existing topological sorter immediately afterward;
- returns the normalized four-counter result
  `reconciled_static_tensor_shapes`,
  `reconciled_static_shape_mutations`, `reordered_operators`, and
  `cycle_detected`;
- preserves the old cycle behavior, including leaving operator order unchanged
  when a cycle is detected.

The eight lowerer branches now retain that compact result under their existing
phase-local targets. Their zero-result defaults contain the same four keys.
No guard, repair owner, graph mutation, following pass, public API, artifact,
dependency, or TensorFlow boundary changed. This reduces the remaining raw
topological-sort sites from fifteen to seven without adding a graph traversal;
the runtime still performs the same reconciliation and sort at the same eight
boundaries.

Implementation validation completed sequentially under `uv`:

- dedicated owner and eight-boundary contract: `3 passed`;
- affected fallback, terminal, singleton/Reshape, topology/layout, and shape
  resolution contracts: `114 passed in 3.17s`;
- lowerer architecture contracts: `258 passed in 18.50s`;
- selected direct-builder topology and reconciliation tests:
  `17 passed, 724 deselected in 0.68s`.

An attempted unfiltered run of `test_tflite_builder_direct.py` reached its two
`tf_converter` matrix tests, which correctly require the optional TensorFlow
extra and fail in the core-only environment. Those tests are outside this
TensorFlow-free unit; the directly affected selection above passes. No
real-model conversion was run because the new owner is differentially
equivalent to the former adjacent calls.

## Terminal topology/layout validation owner

The fallback and primary terminal paths each ended with the same invariant
boundary: topologically sort operators, validate logical-layout annotations,
store the full problem list in `logical_layout_validation_errors` or remove a
stale value, and then finalize the ModelIR. The duplicated lowerer blocks also
discarded the topological-sort result.

The characterization checkpoint fixed both complete four-statement sequences,
their model arguments, metadata behavior, finalize boundary, and a fixture that
requires both operator reordering and an invalid rank/layout diagnostic. It
passed as `2 passed in 0.51s` before production code changed.

`passes/topology_layout_validation.py` now owns this invariant boundary through
`run_topology_layout_validation(model_ir)`. The owner preserves the exact
execution order and metadata payload while returning only three small integer
counters: `reordered_operators`, `cycle_detected`, and
`layout_validation_errors`. The complete error strings remain in ModelIR
metadata and are not duplicated in the returned result. A cycle fixture also
proves that operator order remains unchanged and a stale validation-error entry
is removed when the current validation is clean.

The fallback and primary lowerer paths retain the result as
`_fallback_topology_layout_validation_stats` and
`_terminal_topology_layout_validation_stats`. Direct layout-validator imports
were removed from the lowerer. No validator rule, finalize behavior, graph
mutation, artifact, dependency, public API, or TensorFlow boundary changed.
Five raw topological-sort sites remain after this extraction.

Implementation validation completed sequentially under `uv`:

- dedicated owner, boundary, metadata, and cycle contracts: `3 passed`;
- affected fallback, terminal, shape, and topology contracts:
  `117 passed in 2.68s`;
- lowerer architecture contracts: `258 passed in 16.94s`;
- selected direct-builder topology and reconciliation tests:
  `17 passed, 724 deselected in 0.60s`.

No real-model conversion was run because this owner is differentially
equivalent to the former terminal sequences.

## Fallback topology checkpoint evidence

After terminal validation ownership was extracted, the fallback path still had
two unconditional topological sorts spanning different repair families. The
first normalizes topology after the SE/FC/Gather and placeholder-MatMul repair
sequence before precision cleanup. The second normalizes topology after the
late Conv-input, Concat, Concat-axis, and binary-layout repair sequence before
fallback metadata and high-rank BatchMatMul handling. They are intentionally
kept as separate checkpoints because operators between them can mutate graph
topology and may rely on producer-before-consumer order.

The characterization checkpoint fixed both predecessor guards, successors,
model arguments, and the two-key sort result schema. It passed as
`2 passed in 0.50s` before production code changed.

The lowerer now retains the existing results as
`_fallback_post_placeholder_topology_stats` and
`_fallback_post_layout_repair_topology_stats`. No helper, extra traversal,
condition, graph mutation, repair order, fallback metadata, validation,
artifact, dependency, public API, or TensorFlow boundary was added or changed.
This is observation-only evidence for deciding whether future topology scans
are redundant; neither checkpoint was removed, merged, or made conditional.

Implementation validation completed sequentially under `uv`:

- dedicated checkpoint and fallback orchestration contracts:
  `20 passed in 0.94s`;
- affected fallback, terminal, shape, and topology contracts:
  `119 passed in 2.77s`;
- lowerer architecture contracts: `258 passed in 16.55s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run for these observation-only assignments.

## Primary topology checkpoint evidence

The final three discarded topological-sort results belonged to distinct
primary-path checkpoints: the unconditional post-lowering baseline sort, the
sort inside the no-layout final cleanup guard, and the sort inside final
placeholder-MatMul restoration. Their guards and locations remain separate;
no wrapper or common condition was introduced.

The characterization checkpoint fixed all three enclosing contexts,
predecessors, `model_ir` arguments, and the stable two-key sort result schema.
It passed as `2 passed in 0.55s` before production code changed.

The lowerer now retains the existing results as:

- `_primary_post_lowering_topology_stats`;
- `_no_layout_post_reduction_topology_stats`;
- `_final_placeholder_topology_stats`.

Every direct lowerer call to `_topologically_sort_operators()` now has an
explicit result owner. The calls still execute at the same locations and under
the same guards. No graph traversal, mutation, repair, successor, public API,
artifact, dependency, or TensorFlow boundary changed. These small results are
observation-only and remain unconsumed.

Implementation validation completed sequentially under `uv`:

- primary checkpoint plus terminal orchestration contracts:
  `65 passed in 1.89s`;
- affected fallback, terminal, safe-reduction, shape, and topology contracts:
  `124 passed in 2.92s`;
- lowerer architecture contracts: `258 passed in 18.34s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The first architecture run was `257 passed, 1 failed`; the sole failure was an
AST contract that still required the final placeholder sort to be a discarded
expression. It now requires the phase-specific result target, and the same gate
passes. No real-model conversion was run for these observation-only
assignments.

## Bounded ConversionSession phase-result storage

The observation audit found that all seven topology/validation result locals
introduced by the preceding checkpoints were intentionally unconsumed. The
existing `session.diagnostics` list was not reused: its private metrics export
has an established contract that every exported entry is a `model_ir_pass`
event. Mixing phase counters into that list would change internal metrics and
existing tests.

`ConversionSession` now provides a separate bounded store through
`record_phase_result()` and `phase_results_snapshot()`. The store:

- accepts at most 128 phase IDs;
- accepts at most 32 counters per phase;
- accepts integer counters only and normalizes them to built-in `int`;
- copies caller mappings on record and returns isolated snapshots;
- remains internal to one conversion session and is not placed in ModelIR
  metadata, public results, reports, or artifacts.

The five direct topology checkpoints and two terminal topology/layout
validation results now record seven stable phase IDs instead of creating seven
unconsumed lowerer locals. Their owner calls remain nested directly in the
record operation, so evaluation and graph mutation order are unchanged. The
store contains only the already-small counter dictionaries; no tensor layout
map, graph object, tensor data, or diagnostic string list is retained.

Characterization passed as `1 passed, 1 xfailed in 0.56s`. The strict expected
failure covered the absent bounded store while the existing seven locals and
their schemas were still fixed. Implementation validation completed
sequentially under `uv`:

- bounded store, migrated topology/validation contracts, fallback/terminal
  orchestration, and core diagnostics compatibility:
  `145 passed in 2.97s`;
- lowerer architecture contracts: `258 passed in 18.98s`;
- additional static-shape, safe-reduction, topology/layout refresh, and shape
  resolution contracts: `36 passed in 1.04s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The existing private pass-diagnostics sink remains unchanged and its
`model_ir_pass`-only contract still passes. No real-model conversion was run
because pass owners and ModelIR mutations are unchanged.

## Topology/layout refresh phase-result migration

The six topology/layout refresh results already shared one characterized owner
and the same two-counter schema. They were therefore migrated as the next
homogeneous bounded-store family without reopening owner behavior or mixing in
shape-reconciliation results.

The former unconsumed locals now use these phase IDs:

- `topology_layout.fallback.post_dynamic_rank1`;
- `topology_layout.fallback.broadcast`;
- `topology_layout.primary.absolute_final`;
- `topology_layout.primary.final_convinteger`;
- `topology_layout.primary.final_instancenorm`;
- `topology_layout.primary.final_broadcast`.

Each `run_topology_layout_refresh()` call remains at the same predecessor and
under the same guard. The owner still sorts operators, refreshes logical layout
annotations, releases the full layout map, and returns only
`reordered_operators` and `cycle_detected`. The bounded store copies those two
integers. No extra layout map, tensor reference, graph object, scan, public
result, report field, artifact, dependency, or TensorFlow import was added.

Validation completed sequentially under `uv`:

- bounded-store, topology/layout owner, fallback/terminal orchestration, and
  core diagnostics contracts: `149 passed in 2.97s`;
- lowerer architecture contracts: `258 passed in 16.74s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because only the destination of an already
small, unconsumed result changed.

## Single-call phase-result migration

Three remaining observation-only locals each wrapped one already-characterized
owner call. Their dedicated contracts prove bounded integer-only schemas:

- core Dynamic Reshape resolution: one counter;
- no-layout safe Transpose reduction: three counters;
- terminal Expand/Squeeze static-shape reconciliation: two counters.

They now record these stable phase IDs:

- `shape_resolution.core.dynamic_reshape`;
- `layout.no_layout.safe_transpose_reduction`;
- `shape_reconciliation.terminal.expand_squeeze`.

The calls remain at their original positions between the same predecessors and
successors. Dynamic Reshape resolution and terminal reconciliation remain
unconditional; safe Transpose reduction remains under its existing no-layout
guard. No owner argument, graph mutation, pass ordering, condition, public
result, report, artifact, dependency, or TensorFlow boundary changed.

Validation completed sequentially under `uv`:

- dedicated owner/schema/boundary contracts: `14 passed in 0.88s`;
- bounded-store, topology/layout, fallback/terminal orchestration, and core
  contracts: `149 passed in 3.49s`;
- lowerer architecture contracts: `258 passed in 16.73s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because the owner effects are unchanged and
only their small result destination moved.

## Static-shape/topology phase-result migration

The remaining eight observation-only results all use the previously
characterized `run_static_shape_topology_reconciliation()` owner. They have
now moved from lowerer-local variables to the bounded `ConversionSession`
phase-result store under these stable phase IDs:

- `shape_topology.fallback.norm`;
- `shape_topology.fallback.high_rank_batch_matmul`;
- `shape_topology.primary.final_high_rank_batch_matmul`;
- `shape_topology.primary.final_pad_layout`;
- `shape_topology.primary.final_conv_input`;
- `shape_topology.primary.final_mixed_concat`;
- `shape_topology.primary.final_concat_axis`;
- `shape_topology.primary.final_binary_layout`.

The store follows invoked-phase-only semantics. A guarded phase that does not
run is absent from the snapshot; it is not represented as an artificial
all-zero result. The former all-zero default dictionaries were unconsumed and
have therefore been removed. This preserves the important distinction between
"not invoked" and "invoked but made no change" while reducing lowerer-local
state.

Every existing guard, owner argument, owner evaluation count, graph mutation,
topological sort, predecessor, successor, and execution order remains
unchanged. The reconciliation owner still returns the same four bounded
integer counters for static tensor reconciliation, mutation count, operator
reordering, and cycle detection. No scan was added or removed, and the stored
results are not exposed through ModelIR metadata, public APIs, reports, or
artifacts.

Validation completed sequentially under core-only `uv`:

- dedicated and directly affected orchestration contracts:
  `101 passed in 2.66s`;
- broader phase-result, owner, fallback, terminal, and topology contracts:
  `124 passed in 3.21s`;
- lowerer architecture contracts: `258 passed in 16.64s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because this checkpoint changes only the
destination of already-computed bounded dictionaries. The bounded session
contract now covers 24 phase IDs.

## Fallback static-shape phase-result characterization

The remaining unconsumed lowerer observations were inventoried after the
24-phase checkpoint. The next selected family contains seven fallback-only
static-shape reconciliation results for broadcast, SE/FC/Gather,
placeholder-MatMul, Conv input, mixed Concat, Concat axis, and binary layout.

All seven boundaries currently share the same contract:

- an unconsumed all-zero default dictionary;
- an existing mutation-positive guard;
- one `_reconcile_static_tensor_shapes(fallback_ir,
  include_mutation_count=True)` call;
- the same bounded two-counter integer schema;
- no load of the result elsewhere in the lowerer.

The characterization fixes the exact source order, targets, default schema,
owner arguments, keyword arguments, and absence of consumers. A strict
expected failure specifies migration to seven stable
`shape_reconciliation.fallback.*` phase IDs with invoked-phase-only semantics.
No production source changed in this checkpoint.

Validation completed sequentially under core-only `uv`:

- dedicated family contract: `1 passed, 1 xfailed in 0.15s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented bounded-store
migration.

## Layout pass-set 1 affine cleanup characterization

The next migration is restricted to five mapping-only affine observations in
layout pass-set 1: the initial and post-binary affine-chain folds, affine
pre/post propagation, pre-unary affine fan-out, and mean-affine pre/post
propagation. All five share the existing `optimize_layout_transpose_chains`
guard, return bounded integer mappings with explicit schema tests, and have no
defaults or consumers.

The characterization fixes the exact owner calls and keywords, the contiguous
four-result prefix, both composite recovery boundaries, the isolated
post-binary fold boundary, and absence of loads. Composite attention,
quantized, and safe-binary recovery results remain outside this migration. A
strict expected failure requires five stable
`cleanup.layout_pass_set_1.*` affine records. No production source changed.

The first focused run exposed one stale adjacent-owner assertion that still
expected the previously migrated safe-transpose result to be an outer
`ast.Name` call. The test helper now unwraps the exact nested phase-record owner;
production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- dedicated characterization and existing owner/schema contracts:
  `9 passed, 1 xfailed in 0.97s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented five-result
destination migration.

## Terminal normalization cleanup implementation

The five characterized observations now record under:

- `cleanup.terminal.instancenorm_post_bias`;
- `cleanup.terminal.normalization_pad`;
- `cleanup.terminal.instancenorm_residual_add`;
- `cleanup.terminal.instancenorm_residual_mul_concat`;
- `cleanup.terminal.instancenorm_dualstats`.

Only the unused local destinations changed. Owner calls, arguments, keywords,
unconditional execution, evaluation count, five-statement order, the
preceding Swish QDQ-island record, the following terminal boundary-layout
composite, ModelIR mutations, public outputs, reports, artifacts,
dependencies, and TensorFlow isolation are unchanged. The bounded store now
covers 106 of 128 phase slots, leaving 22.

Seventeen representation-dependent assertions expected the migrated terminal
call to be an outer assignment. They now unwrap the record and continue to
verify exact phase IDs, owners, arguments, keywords, call counts, later
non-migrated assignments, and both outer boundaries. Production behavior was
not implicated.

Validation completed sequentially under core-only `uv`:

- focused normalization/store/terminal/boundary contracts:
  `100 passed in 2.81s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader result and phase-result contracts: `188 passed in 8.64s`;
- lowerer architecture contracts: `258 passed in 18.46s`;
- targeted Ruff, Python bytecode compilation, AST capacity audit, and
  whitespace validation: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and the synthetic runtime suite exercises
the terminal path.

## Guarded terminal BatchMatMul characterization

The next bounded family contains the three consecutive BatchMatMul mapping
observations inside the existing `optimize_layout_transpose_chains` guard:
affine Transpose input cleanup, Reshape/SE NHWC cleanup, and Transpose-input
adjoint-flag folding.

Existing owner tests fix each single-counter integer schema and no-op graph
behavior. None of the three locals has a default or consumer. The
characterization fixes the shared guard, owner expressions, three-statement
adjacency, the preceding Mean-attention composite, the following QKV-attention
composite, and absence of loads. A strict expected failure requires three
stable `cleanup.terminal.*` records inside the same guard. Composite results
are intentionally excluded and no production source changed.

Validation completed sequentially under core-only `uv`:

- related BatchMatMul/QKV baseline: `23 passed in 0.94s`;
- characterization plus related contracts: `24 passed, 1 xfailed in 1.14s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented three-result
destination migration.

## Very-late residual cleanup implementation

The three characterized observations now record consecutively under:

- `cleanup.very_late.residual_affine_prelu`;
- `cleanup.very_late.residual_affine_fanout`;
- `cleanup.very_late.prune_reconcile`.

Only the unused local destinations changed. Owner calls, arguments,
unconditional execution, source order, indexed prune/reconcile behavior, both
SiNet composite boundaries, ModelIR mutations, public outputs, reports,
artifacts, dependencies, and TensorFlow isolation remain unchanged. The
bounded store now covers 116/128 phase IDs, leaving 12 slots.

Affected residual, prune/reconcile, SiNet boundary, and architecture contracts
now unwrap phase records and verify exact phase IDs and nested owners while
retaining composite target checks. The existing characterization expectation
was converted to a passing contract.

Validation completed sequentially under core-only `uv`:

- focused residual/prune/SiNet/store contracts: `20 passed in 1.10s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result and phase-result contracts: `191 passed in 9.62s`;
- lowerer architecture contracts: `258 passed in 18.50s`;
- targeted Ruff, Python bytecode compilation, AST capacity audit, and
  whitespace validation: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and existing synthetic runtime contracts
exercise all three owners.

## Post-cleanup attention result characterization

The next unit is limited to the two consecutive top-level observations after
the post-cleanup SiNet pre-Add/Resize recovery composite: CSP attention NHWC
cleanup and SA/PA MirrorPad NHWC propagation. Each owner returns exactly one
bounded integer counter, receives the existing conversion-session layout
state, and performs its existing positive-rewrite cleanup internally. Neither
lowerer-local result is loaded.

The CSP attention result contract now strictly expects the two exact
`cleanup.post_cleanup.*` records, their owner expressions and adjacency, the
preceding SiNet composite, the following post-SiNet BatchMatMul affine-input
result, and removal of both unconsumed local targets. No production source
changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `14 passed in 0.88s`. The characterization must retain those passes plus
one intentional strict expected failure until implementation. Targeted Ruff,
bytecode compilation, and whitespace validation are required before the
characterization checkpoint is committed and pushed.

After that checkpoint, replace only the two mapping destinations, preserve
all owner calls and surrounding composite boundaries, update structural
contracts, run the sequential gates, document, commit, and push. Never create,
update, or reopen a pull request.

## Post-cleanup attention result implementation

The two characterized observations now record consecutively under:

- `cleanup.post_cleanup.csp_attention`;
- `cleanup.post_cleanup.sa_pa_mirrorpad`.

Only the unused local destinations changed. Both owner calls, layout-state
arguments, unconditional execution, source adjacency, preceding SiNet
pre-Add/Resize composite, following post-SiNet BatchMatMul result, ModelIR
mutations, public outputs, reports, artifacts, dependencies, and TensorFlow
isolation remain unchanged. The bounded store now covers 118/128 phase IDs,
leaving 10 slots.

Affected CSP, MirrorPad, SiNet-boundary, BatchMatMul, store, and architecture
contracts now unwrap the phase records and preserve exact phase, owner,
argument, and boundary checks. The strict characterization expectation is now
a passing contract.

Validation completed sequentially under core-only `uv`:

- focused boundary and phase-store contracts: `17 passed in 1.12s`;
- focused CSP and indexed MirrorPad runtime/orchestration contracts:
  `68 passed in 1.20s`;
- synthetic core runtime contracts: `55 passed in 1.12s`;
- broader result and phase-result contracts: `192 passed in 9.46s`;
- lowerer architecture contracts: `258 passed in 21.25s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and focused runtime contracts exercise
both owners.

## Post-SiNet BatchMatMul result characterization

The next unit contains the three consecutive unconditional BatchMatMul
observations immediately after post-cleanup SA/PA MirrorPad propagation:
affine Transpose input cleanup, Reshape/SE NHWC cleanup, and Transpose-input
adjoint-flag folding. The same owner schemas and graph effects are already
covered at the guarded terminal boundary and by focused runtime tests. The
three post-SiNet locals have no consumers.

The affine-input result module now strictly expects the exact
`cleanup.post_sinet.batchmatmul_*` records, nested owner expressions,
three-statement adjacency, preceding post-cleanup MirrorPad phase record,
following QKV-attention composite, and absence of result loads. The composite
itself remains outside the bounded store. No production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `28 passed in 1.20s`. Characterization must preserve those passes and add
one intentional strict expected failure. Targeted Ruff, bytecode compilation,
and whitespace validation are required before committing and pushing this
checkpoint.

Implement only the three result destinations after that checkpoint. Preserve
the owner calls, unconditional order, adjacent phase/composite boundaries,
and all graph behavior; update the store from 118 to 121 records, run the
sequential gates, document, commit, and push. Never create, update, or reopen
a pull request.

## Post-SiNet BatchMatMul result implementation

The three characterized observations now record consecutively under:

- `cleanup.post_sinet.batchmatmul_affine_input`;
- `cleanup.post_sinet.batchmatmul_reshape_se`;
- `cleanup.post_sinet.batchmatmul_adj_flags`.

Only the unused local destinations changed. All three owners, arguments,
unconditional execution, source order, preceding MirrorPad phase, following
QKV-attention composite, ModelIR mutations, public behavior, reports,
artifacts, dependencies, and TensorFlow isolation remain unchanged. The QKV
composite remains deliberately outside the store. The bounded store now
covers 121/128 phase IDs, leaving 7 slots.

Affected affine-input, Reshape/SE, adjoint-flag, QKV, attention-boundary, and
store contracts now unwrap phase records while preserving exact phase, owner,
argument, and composite checks. The strict characterization expectation is a
passing contract.

Validation completed sequentially under core-only `uv`:

- focused BatchMatMul/QKV/attention/store contracts: `31 passed in 1.46s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- broader result and phase-result contracts: `192 passed in 9.08s`;
- lowerer architecture contracts: `258 passed in 17.00s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and the focused suite exercises all three
owners.

## Post-SiNet ReLU/Split result characterization

The next unit is limited to the three consecutive ReLU/Split observations
after the retained post-SiNet QKV-attention composite: all-Split-output NHWC
propagation, Split/Conv/ReLU/Concat NHWC propagation, and the
Split/Conv/Concat bridge collapse. Each owner returns one bounded integer
counter and receives the existing layout state. Their local results have no
consumers.

The ReLU/Split/Conv/Concat result module now strictly expects three exact
`cleanup.post_sinet.*` records with nested owner expressions, adjacency, the
preceding QKV composite, the following mix-attention local, and absence of
loads. The composite and following result remain outside this unit. No
production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `72 passed in 1.20s`. Characterization must retain those passes plus one
intentional strict expected failure. Targeted Ruff, bytecode compilation, and
whitespace validation are required before the checkpoint is committed and
pushed.

Implement only these three result destinations, preserve both outer
boundaries and every owner call, grow the store from 121 to 124 records, run
the sequential gates, document, commit, and push. Never create, update, or
reopen a pull request.

## Post-SiNet ReLU/Split result implementation

The three characterized observations now record consecutively under:

- `cleanup.post_sinet.relu_split_all_outputs`;
- `cleanup.post_sinet.relu_split_conv_concat`;
- `cleanup.post_sinet.split_conv_concat_bridge`.

Only the unused local destinations changed. Owner calls, layout-state
arguments, unconditional order, the preceding QKV composite, following
mix-attention result, ModelIR behavior, public outputs, reports, artifacts,
dependencies, and TensorFlow isolation are unchanged. The bounded store now
covers 124/128 phase IDs, leaving 4 slots.

Affected ReLU/Split, indexed bridge, QKV, mix-attention, store, and
architecture contracts now unwrap the records and preserve exact phase,
owner, argument, adjacency, and composite checks. The strict
characterization expectation now passes.

Validation completed sequentially under core-only `uv`:

- focused ReLU/Split/QKV/mix-attention/store contracts:
  `75 passed in 1.51s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader result and phase-result contracts: `193 passed in 9.15s`;
- lowerer architecture contracts: `258 passed in 17.38s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and the focused suite exercises all three
owners.

## Post-SiNet attention and activation result characterization

The next unit is limited to the three consecutive observations after the
post-SiNet Split/Conv/Concat bridge: SiNet double-Logistic mix-attention,
mixed Mean/ReduceMax/Concat/MirrorPad layout cleanup, and the
Dequantize/HardSigmoid/Quantize bridge. Their fixed schemas contain one
bounded integer each; the mixed-attention owner continues to receive the
conversion diagnostics stream, but diagnostic events are not copied into its
returned mapping. None of the three locals is loaded.

The mixed-attention result module now strictly expects the exact
`cleanup.post_sinet.*` records, nested owner expressions including diagnostics
arguments, adjacency, preceding Split/Conv/Concat phase, following shared
NDHWC/cost-volume state-scope creation, and absence of result loads. No
production source changed.

Validation completed sequentially under core-only `uv`: the related baseline
is `14 passed in 1.04s`. Characterization must preserve those passes plus one
intentional strict expected failure. Targeted Ruff, bytecode compilation, and
whitespace validation are required before committing and pushing this
checkpoint.

Implement only the three destinations after the checkpoint. Preserve
diagnostic behavior, outer boundaries, owner arguments, and execution order;
grow the store from 124 to 127 records, run sequential gates, document,
commit, and push. Never create, update, or reopen a pull request.

## Post-SiNet attention and activation result implementation

The three characterized observations now record consecutively under:

- `cleanup.post_sinet.mix_attention`;
- `cleanup.post_sinet.mixed_attention_layout`;
- `cleanup.post_sinet.dequant_hardsigmoid_bridge`.

Only the unused local destinations changed. Owner calls, layout-state and
diagnostics arguments, unconditional order, preceding Split/Conv/Concat
phase, following shared state-scope creation, diagnostics events, ModelIR
behavior, public outputs, artifacts, dependencies, and TensorFlow isolation
remain unchanged. The bounded store now covers 127/128 phase IDs, leaving one
slot.

One architecture assertion initially failed because it treated the two calls
immediately before the NDHWC/cost-volume scope as raw assignments. The
failure was representation-only: all focused, core, and result suites passed.
The assertion now unwraps phase records and additionally verifies both exact
phase IDs before checking nested owners and shared-scope order. The full
architecture suite then passed.

Validation completed sequentially under core-only `uv`:

- focused attention/activation/state-scope/store contracts:
  `17 passed in 1.23s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader result and phase-result contracts: `194 passed in 9.03s`;
- focused repaired architecture contract: `1 passed in 2.17s`;
- full lowerer architecture contracts after repair: `258 passed in 16.99s`;
- targeted Ruff, bytecode compilation, AST capacity audit, and whitespace
  checks: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and focused runtime contracts exercise
all three owners.

## Late NDHWC/cost-volume pair characterization

The phase-result store has one remaining slot. The next two lowerer results
are a semantic pair: NDHWC gate cleanup and cost-volume ScatterND cleanup run
consecutively with the same `ModelIRPassStateScope`. Recording only one would
make the observation contract asymmetric, while recording both separately
would exceed the 128-phase bound.

The selected contract therefore requires one small orchestration owner that
creates the shared state scope internally, invokes the existing two owners in
the same order with the same model/layout/diagnostics inputs, combines their
three distinct integer counters, and returns one bounded mapping. The lowerer
must record that mapping as `cleanup.late.ndhwc_cost_volume` between the
post-SiNet HardSigmoid phase and late Conv-affine result, with all three old
locals absent.

The existing late NDHWC/cost-volume result module now carries this as a strict
expected failure. No production source changed. Validation completed
sequentially under core-only `uv`: the related baseline is
`16 passed in 0.86s`. Characterization must preserve those passes plus one
intentional strict xfail, and targeted Ruff, bytecode compilation, and
whitespace validation must pass before commit and push.

Implementation must preserve owner order and diagnostics, prove that both
callbacks receive the same internally scoped state object, update the store
from 127 to its fixed 128-record capacity, run sequential gates, document,
commit, and push. Never create, update, or reopen a pull request.

## Late NDHWC/cost-volume pair implementation

`run_late_ndhwc_cost_volume_layout_cleanup(context)` now owns the adjacent
pair. It creates one short-lived `ModelIRPassStateScope`, invokes NDHWC gate
cleanup and cost-volume ScatterND cleanup in the original order with identical
model/layout/diagnostics values, and merges their two-plus-one distinct
integer counters. The lowerer records this mapping under
`cleanup.late.ndhwc_cost_volume` using the existing shared pass context.

The former scope local and two unconsumed result locals are gone. Low-level
owner imports remain compatibility re-exports. A dedicated runtime test
proves callback order, shared scope identity, exact context members, and the
merged schema. The phase-result store now contains exactly 128/128 phase IDs;
its bound was not increased and no further phase may be added without an
explicit retention-policy decision.

Three groups of stale structural assumptions surfaced during broad gates:

- one terminal-layout assertion expected the former three-statement
  scope/result sequence;
- two result assertions expected the former scope target immediately after
  post-SiNet HardSigmoid cleanup;
- two architecture assertions counted only direct lowerer calls plus the
  general gate cluster.

They were updated to require the exact combined phase, nested owner, shared
scope ownership, and the separate two-pass late orchestration ID sequence.
No graph, numerical, diagnostics, or artifact failure occurred.

Validation completed sequentially under core-only `uv`:

- focused pair/gate/store/architecture contracts: `20 passed in 2.76s`;
- pass-efficiency and terminal-layout contracts after repair:
  `94 passed in 2.12s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- focused repaired boundary contracts: `11 passed in 0.87s`;
- broader result and phase-result contracts after repair:
  `196 passed in 9.12s`;
- focused architecture ownership repairs: `2 passed in 2.29s`;
- full lowerer architecture contracts after repair: `258 passed in 16.88s`;
- targeted Ruff, bytecode compilation, full-capacity AST audit, and
  whitespace checks: passed.

No root-model corpus conversion was run because this mechanical owner
extraction preserves both already-tested callbacks and now has a focused
runtime equivalence contract.

## Late Concat shared-scope characterization

With the phase store fixed at 128/128, the next refactoring does not add a
phase record. The selected lowerer cluster creates one
`ModelIRPassStateScope` and runs four consecutive owners: axis-3 constant
Concat layout, Dequantize/Concat/Quantize layout, LayerNorm statistics layout,
and layout-Transpose cleanup. All four result locals are unconsumed, while the
scope is loaded exactly once by each owner.

A new focused contract fixes the current four-owner order, targets, arguments,
shared-scope identity, preceding late cost-volume Conv-affine result, following
layout-optimization guard, and absence of result consumers. A strict expected
failure requires one `run_late_concat_layout_cleanup` owner that creates the
scope internally and a single retained `_late_concat_layout_results`
composite. The composite deliberately remains outside the full phase store.

No production source changed. Validation completed sequentially under
core-only `uv`: the related baseline is `72 passed in 0.86s`. The focused
characterization must retain those passes plus one intentional strict xfail;
targeted Ruff, bytecode compilation, and whitespace validation must pass
before commit and push.

Implementation must return the four mappings as an ordered tuple without
merging schemas, prove callback order and shared scope identity, keep the
128-phase bound unchanged, run sequential gates, document, commit, and push.
Never create, update, or reopen a pull request.

## Late Concat shared-scope implementation

The new `run_late_concat_layout_cleanup(context)` owner creates one internal
`ModelIRPassStateScope` and invokes the four characterized owners in order.
It returns their mappings as an ordered tuple, preserving each independent
schema. The lowerer retains that tuple as `_late_concat_layout_results`
through `shared_model_ir_pass_context`; it is deliberately not written to the
full phase-result store.

The former scope and four unconsumed locals are absent, and the scope now
falls out of lifetime immediately after the composite owner returns. Direct
low-level imports remain compatibility re-exports where necessary. The now
unused lowerer-level `ModelIRPassStateScope` import was removed. A focused
runtime contract proves callback order, identical model/layout/diagnostics
objects, shared scope identity, and tuple ordering.

Broad gates exposed only stale representation contracts: four focused
assertions expected the old scope/four-assignment sequence or direct
layout-Transpose call, and two architecture assertions expected a direct
axis-3 call or omitted it from orchestrated ownership. They now verify the
composite, internal scope, nested owner arguments, and
`LATE_CONCAT_LAYOUT_PASS_IDS`. No graph or numerical failure occurred.

Validation completed sequentially under core-only `uv`:

- focused late-Concat owner and affected owner contracts:
  `76 passed in 3.10s`;
- terminal-layout and pass-efficiency contracts: `94 passed in 1.94s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- broader result and phase-result contracts: `196 passed in 9.20s`;
- focused architecture ownership repairs: `2 passed in 2.29s`;
- full lowerer architecture contracts after repair: `258 passed in 16.86s`;
- targeted Ruff, bytecode compilation, fixed-capacity AST audit, and
  whitespace checks: passed.

The phase store remains exactly 128/128. No root-model conversion was run
because this is a characterized four-call orchestration extraction with a
focused runtime equivalence test.

## Late reshape-layout composite characterization

The next non-store unit covers three adjacent late layout repairs: the
ExpandDims-compatible Transpose/Reshape collapse, the Flatten-HW-compatible
collapse, and the private NHWC Reshape collapse. Each returns one bounded
integer mapping, and all three lowerer locals are unconsumed.

The focused characterization fixes their order, exact model/layout arguments,
guarded elementwise-fanout predecessor, channel-shuffle successor, and absence
of result consumers. A strict expected failure requires one
`run_late_reshape_layout_cleanup` owner and one ordered
`_late_reshape_layout_results` tuple outside the already-full session store.
No production source changed.

The initial baseline exposed one independent stale channel-shuffle assertion:
its predecessor had already moved inside `session.record_phase_result`, while
the test still required a direct call. Commit `2107f972` now verifies both the
stable phase ID and its nested owner; that module passes all 24 contracts.

Sequential validation under core-only `uv` completed with
`93 passed, 1 xfailed in 1.09s`; the sole xfail is the intentionally absent
composite. Targeted Ruff, bytecode compilation, and whitespace validation
also passed. Commit and push the characterization before implementing the
owner. Keep the phase store at 128/128 and do not create, update, or reopen a
pull request.

## Late reshape-layout composite implementation

`run_late_reshape_layout_cleanup(context)` now owns the characterized
ExpandDims-compatible, Flatten-HW-compatible, and NHWC-collapse repairs. It
passes the shared conversion-local `LayoutState` to the first two owners,
invokes the model-only collapse owner third, and returns the three independent
counter mappings as an ordered tuple.

The lowerer retains one `_late_reshape_layout_results` composite through
`shared_model_ir_pass_context`. The three old unconsumed locals are absent,
and the composite remains outside `ConversionSession.phase_results`; the
bounded store stays exactly 128/128. Compatibility wrappers remain available
for existing direct callers and tests.

Focused runtime coverage proves callback order, identical ModelIR identity,
layout argument identity for the first two callbacks, the model-only third
callback, and tuple ordering. Existing structural contracts now require the
composite at the elementwise-fanout/channel-shuffle boundaries, account for
its three nested owners, and reject the old locals. No graph, numerical,
diagnostics, or artifact failure occurred.

Validation completed sequentially under core-only `uv`:

- focused reshape owners and affected boundaries: `95 passed in 1.11s`;
- terminal-layout and pass-efficiency contracts: `94 passed in 2.11s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result and phase-result contracts: `196 passed in 9.07s`;
- full lowerer architecture contracts: `258 passed in 17.48s`;
- phase-result capacity contracts: `2 passed in 0.51s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No root-model conversion was run because this is a characterized three-call
owner extraction with focused runtime equivalence and unchanged serialization
inputs.

## Late attention-layout composite characterization

The next non-store unit covers four adjacent attention repairs: QKV
Reshape/Transpose simplification, attention-Gather cleanup, axis-0 Gather to
Reshape input normalization, and pre-projection BatchMatMul rank lifting. The
four unconsumed mappings use the fixed layout/model/layout/model argument
policy.

The focused characterization fixes their exact order and arguments, the late
channel-shuffle predecessor, the window-partition successor, and the absence
of result consumers. A strict expected failure requires one
`run_late_attention_layout_cleanup` owner and one ordered
`_late_attention_layout_results` tuple outside the full phase store. No
production source changed.

Sequential validation under core-only `uv` completed with
`214 passed, 1 xfailed in 1.18s`; the sole xfail is the intentionally absent
composite owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed. Commit and push this characterization before implementation. Keep the
phase store fixed at 128/128 and never create, update, or reopen a pull request.

## Late attention-layout composite implementation

`run_late_attention_layout_cleanup(context)` now owns the characterized QKV
reshape, attention-Gather cleanup, axis-0 Gather reshape, and pre-projection
rank-lift passes. It preserves the layout/model/layout/model argument policy
and returns all four independent counter mappings as an ordered tuple.

The lowerer retains one `_late_attention_layout_results` value through
`shared_model_ir_pass_context`. The four old unconsumed locals are absent, and
the composite remains outside `ConversionSession.phase_results`; the store is
still exactly 128/128. Existing lowerer wrappers remain compatibility exports.

Focused runtime coverage proves exact callback order, ModelIR identity, layout
identity at the first and third callbacks, model-only invocation at the second
and fourth callbacks, and tuple ordering. Structural contracts now require the
composite between late channel-shuffle and window partition and account for
its four nested owners. No graph, numerical, diagnostics, or artifact failure
occurred.

Validation completed sequentially under core-only `uv`:

- focused attention owners and affected boundaries: `238 passed in 1.35s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 2.15s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- broader result and phase-result contracts: `196 passed in 9.42s`;
- full lowerer architecture contracts: `258 passed in 17.67s`;
- phase-result capacity contracts: `2 passed in 0.52s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No root-model conversion was run because this is a characterized four-call
owner extraction with focused runtime equivalence and unchanged serialization
inputs.

## Late window-layout composite characterization

The next non-store unit covers the adjacent window-partition and window-reverse
layout repairs. Both return one unconsumed integer mapping and receive the same
ModelIR and conversion-local `LayoutState`.

The focused characterization fixes their order, exact arguments, late
attention-composite predecessor, final shape/activation convergence successor,
and absence of consumers. A strict expected failure requires one
`run_late_window_layout_cleanup` owner and one ordered
`_late_window_layout_results` tuple outside the full phase store. No production
source changed.

Sequential validation under core-only `uv` completed with
`103 passed, 1 xfailed in 0.84s`; the sole xfail is the intentionally absent
composite. Targeted Ruff, bytecode compilation, and whitespace checks passed.
Commit and push characterization before implementation, keep the store fixed
at 128/128, and never create, update, or reopen a pull request.

## Late window-layout composite implementation

`run_late_window_layout_cleanup(context)` now owns the characterized
window-partition and window-reverse repairs. Both receive the shared ModelIR
and conversion-local `LayoutState`; their independent counter mappings are
returned as an ordered tuple.

The lowerer retains one `_late_window_layout_results` composite through
`shared_model_ir_pass_context`. The two old unconsumed locals are absent, and
the composite remains outside `ConversionSession.phase_results`; the store is
still exactly 128/128. Compatibility wrappers and optional `graph_index`
behavior remain unchanged.

Focused runtime coverage proves call order, ModelIR/layout identity, and tuple
ordering. Structural contracts now require the composite between late
attention and final shape/activation convergence and account for its two nested
owners. No graph, numerical, diagnostics, or artifact failure occurred.

Validation completed sequentially under core-only `uv`:

- focused window owners and affected boundaries: `110 passed in 3.11s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 2.06s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- broader result and phase-result contracts: `196 passed in 9.14s`;
- full lowerer architecture contracts: `258 passed in 19.39s`;
- phase-result capacity contracts: `2 passed in 0.51s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No root-model conversion was run because this is a characterized two-call
owner extraction with focused runtime equivalence and unchanged serialization
inputs.

## Final boundary-channel composite characterization

The next non-store unit covers final boundary-input normalization, internal
channel-slice propagation, and the channel-slice Mul/Add bridge. The first
owner receives layout state and diagnostics; unlike their earlier terminal
invocations, the final two owners intentionally remain model-only. All three
result mappings are unconsumed.

The focused characterization fixes this exact argument policy and order, the
final shape/activation convergence predecessor, the slice/Concat recovery
successor, and absence of consumers. A strict expected failure requires one
`run_final_boundary_channel_layout_cleanup` owner and one ordered
`_final_boundary_channel_layout_results` tuple outside the full store. No
production source changed.

Sequential validation under core-only `uv` completed with
`77 passed, 1 xfailed in 2.20s`; the sole xfail is the intentionally absent
composite. Targeted Ruff, bytecode compilation, and whitespace checks passed.
Commit and push characterization first, keep the store fixed at 128/128, and
never create, update, or reopen a pull request.

## Final boundary-channel composite implementation

`run_final_boundary_channel_layout_cleanup(context)` now owns the
characterized boundary-input normalization, internal channel-slice, and
channel-slice Mul/Add bridge calls. It passes layout state and diagnostics only
to normalization and preserves the model-only final invocations of the latter
two owners. Their independent mappings are returned as an ordered tuple.

The lowerer retains one `_final_boundary_channel_layout_results` composite via
`shared_model_ir_pass_context`. The three old unconsumed locals are absent, and
the composite remains outside `ConversionSession.phase_results`; the store is
still exactly 128/128. Earlier terminal phase records and their layout-aware
arguments are unchanged.

Focused runtime coverage proves call order, ModelIR/layout/diagnostics identity,
the model-only second and third callbacks, and tuple ordering. Structural
contracts now distinguish the retained terminal calls from the new final
composite and require it between final convergence and slice/Concat recovery.
No graph, numerical, diagnostics, or artifact failure occurred.

Validation completed sequentially under core-only `uv`:

- focused boundary/channel owners and affected boundaries:
  `79 passed in 4.15s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.97s`;
- synthetic core runtime contracts: `55 passed in 1.11s`;
- broader result and phase-result contracts: `196 passed in 9.49s`;
- full lowerer architecture contracts: `258 passed in 19.29s`;
- phase-result capacity contracts: `2 passed in 0.52s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No root-model conversion was run because this is a characterized three-call
owner extraction with focused runtime equivalence and unchanged serialization
inputs.

## Terminal Concat-bridge composite characterization

The initially considered final Slice/pre-Concat pair was deferred without
source changes. Its pre-Concat implementation still lives in the lowerer and
combines indexed, quantized, and legacy owners; moving the pair now would
require either a circular import or callback injection. The pre-ConCat owner
must first move behind a pass-module compatibility wrapper in a separate
characterized checkpoint.

The selected non-store unit instead covers six adjacent owners that already
have pass-module boundaries: all-output ReLU/Split, ReLU/Split/Conv/Concat,
mixed Split/Concat, Concat input adaptation, Concat-unary-Conv, and Shape
extract. Their fixed argument policy is four layout-aware calls, one
layout-and-diagnostics call, and one model-only call. All six mappings are
unconsumed.

The focused characterization fixes exact order and arguments, final
pre-Concat predecessor, guarded elementwise-fanout successor, and absence of
consumers. A strict expected failure requires one
`run_terminal_concat_bridge_layout_cleanup` owner and one ordered
`_terminal_concat_bridge_layout_results` tuple outside the full store. No
production source changed.

Sequential validation under core-only `uv` completed with
`15 passed, 1 xfailed in 0.86s`; the sole xfail is the intentionally absent
composite. Targeted Ruff, bytecode compilation, and whitespace checks passed.
Commit and push characterization first, keep the store at 128/128, and never
create, update, or reopen a pull request.

## Terminal Concat-bridge composite implementation

`run_terminal_concat_bridge_layout_cleanup(context)` now owns the six
characterized callbacks in their original order. The first four receive the
shared ModelIR and conversion-local layout state, the Concat-unary-Conv owner
also receives the shared diagnostics list, and the final Shape-extract owner
remains model-only. Their independent counter mappings are returned as an
ordered tuple.

The lowerer retains one `_terminal_concat_bridge_layout_results` composite via
`shared_model_ir_pass_context`. The six old unconsumed locals are absent, and
the composite remains outside `ConversionSession.phase_results`; the store is
still exactly 128/128. Existing lowerer compatibility wrappers, guards, graph
index behavior, owner implementations, and the following guarded
elementwise-fanout phase remain unchanged.

Focused runtime coverage proves callback order, ModelIR/layout/diagnostics
identity, the final model-only argument policy, and tuple ordering. Structural
contracts now account for nested composite ownership and require the composite
between the retained final pre-Concat result and the guarded fanout successor.
No graph, numerical, diagnostics, or artifact failure occurred.

Validation completed sequentially under core-only `uv`:

- focused composite and affected result contracts: `17 passed in 1.48s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.99s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader result contracts: `196 passed in 9.40s`;
- full lowerer architecture contracts: `258 passed in 19.32s`;
- phase-result capacity contracts: `2 passed in 0.54s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No root-model conversion was run because this is a characterized six-call
owner extraction with focused runtime equivalence and unchanged serialization
inputs. The deferred final Slice/pre-Concat pair still requires a separate
pass-module compatibility-wrapper checkpoint before safe extraction.

## Pre-Concat NHWC pass-module owner characterization

The lowerer-resident `_optimize_transpose_pre_concat_nhwc_chains` composite was
selected as the prerequisite for the deferred final Slice/pre-Concat pair. It
currently invokes indexed NHWC Concat cleanup, quantized indexed cleanup, and
the legacy fallback in that order, then returns one aggregate integer counter.
The first two stages share the caller's layout state and diagnostics; the
legacy stage remains model-only.

The focused characterization fixes this order, argument identity, recognized
counter aggregation, ignored-detail behavior, public lowerer compatibility
name, and final one-counter schema. A strict expected failure requires the
implementation to move to `passes/pre_concat_nhwc_layout.py` while the lowerer
function becomes a one-return compatibility wrapper. No production source
changed.

Implementation must import the three existing pass-module owners directly,
without callback injection or a lowerer import, preserve all four production
uses, and leave the existing legacy lowerer wrapper available to callers.
Keep the phase-result store at 128/128 and do not create, update, or reopen a
pull request.

Sequential characterization under core-only `uv` completed with
`2 passed, 1 xfailed in 0.57s`; the sole xfail is the intentionally absent
pass-module owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

## Pre-Concat NHWC pass-module owner implementation

`passes/pre_concat_nhwc_layout.py` now owns the complete three-stage composite.
It invokes indexed NHWC Concat cleanup, quantized indexed cleanup, and the
legacy fallback in the characterized order, preserves layout/diagnostics only
for the first two stages, and returns the same aggregate integer schema. The
recognized detail-key lists moved unchanged with the owner, so unrelated
detail counters remain excluded from the aggregate.

The lowerer keeps `_optimize_transpose_pre_concat_nhwc_chains` as a one-return
compatibility wrapper with the same signature. All three direct production
calls and the recovery-orchestration callback still resolve through that name.
The separate legacy lowerer wrapper also remains available. The new module
imports the three existing pass owners directly and has no lowerer import or
callback injection.

Sequential validation under core-only `uv` completed with:

- focused owner order, aggregation, runtime identity, and wrapper contracts:
  `3 passed in 0.54s`;
- indexed, quantized, and legacy NHWC Concat family runtime contracts:
  `285 passed in 1.33s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result contracts: `196 passed in 8.99s`;
- full lowerer architecture contracts: `258 passed in 19.02s`;
- phase-result capacity contracts: `2 passed in 0.52s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No graph rewrite, pass ordering, result value, diagnostics destination,
public compatibility name, phase identity, artifact, dependency, or
TensorFlow boundary changed. No root-model conversion was repeated because
the family-level runtime suite exercises all moved dispatch paths. The store
remains exactly 128/128.

## Final Slice/pre-ConCat composite characterization

With the pre-ConCat implementation now behind a pass-module owner, the
previously deferred final pair can be extracted without a lowerer import,
callback injection, or circular dependency. The pair contains final
Slice/pre-post passthrough cleanup followed by final pre-ConCat NHWC cleanup.
The first call is model-only; the second receives the same conversion-local
layout state and diagnostics as before. Both result mappings are unconsumed.

The focused characterization fixes adjacency, exact argument policy, final
slice/Concat recovery predecessor, terminal Concat-bridge successor, and
absence of consumers. A strict expected failure requires one
`run_final_slice_pre_concat_layout_cleanup(shared_model_ir_pass_context)`
owner and an ordered `_final_slice_pre_concat_layout_results` tuple outside
the full phase-result store. No production source changed.

Implementation must import both existing pass owners directly, preserve their
order and independent result mappings, retain both lowerer compatibility
wrappers, and keep the store fixed at 128/128. Do not create, update, or reopen
a pull request.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.14s`; the sole xfail is the intentionally absent
composite owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

## Final Slice/pre-ConCat composite implementation

`run_final_slice_pre_concat_layout_cleanup(context)` now invokes the existing
Slice/pre-post passthrough and pre-ConCat NHWC owners in their original order.
The first remains model-only; the second receives the shared ModelIR, layout
state, and diagnostics. Their independent mappings are returned as an ordered
tuple.

The lowerer retains one `_final_slice_pre_concat_layout_results` composite via
`shared_model_ir_pass_context`, outside the full phase-result store. The two
old unconsumed locals are absent. Both compatibility wrappers, the preceding
final slice/Concat recovery composite, and the following terminal Concat-bridge
composite remain intact.

Sequential validation under core-only `uv` completed with:

- focused composite and affected boundary/result contracts:
  `20 passed in 1.13s`;
- Slice/pre-post owner mutation contracts: `9 passed in 0.52s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.81s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader result contracts: `196 passed in 9.28s`;
- full lowerer architecture contracts: `258 passed in 18.28s`;
- phase-result capacity contracts: `2 passed in 0.52s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No pass implementation, graph rewrite, call count, result value, layout or
diagnostics identity, public compatibility name, artifact, dependency, or
TensorFlow boundary changed. No root-model conversion was repeated because
the focused runtime owner test and existing Slice/pre-post mutation suite cover
the extracted dispatch. The store remains exactly 128/128.

## Late Conv1D/decoder composite characterization

The next non-store audit selected eight adjacent late layout repairs: three
Conv1D unary variants, the Conv1D InstanceNorm/unary bridge, tencoder residual
merge, Conv1D BatchMatMul tail, decoder deconvolution input, and terminal
Squeeze/Mean tail. Every owner already lives in a pass module, every call is
unconditional, every call receives the same ModelIR and conversion-local
layout state, and all eight result mappings are unconsumed.

The focused characterization fixes exact source order and arguments, the late
Swish predecessor, very-late Pad successor, and absence of consumers. A strict
expected failure requires one
`run_late_conv1d_decoder_layout_cleanup(shared_model_ir_pass_context)` owner
and an ordered `_late_conv1d_decoder_layout_results` tuple outside the full
store. No production source changed.

Implementation must import all eight existing owners directly, preserve their
independent mappings and compatibility wrappers, and keep the phase-result
store fixed at 128/128. Do not create, update, or reopen a pull request.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.17s`; the sole xfail is the intentionally absent
composite owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

## Late Conv1D/decoder composite implementation

`run_late_conv1d_decoder_layout_cleanup(context)` now invokes the eight
characterized pass owners in their original order. Every callback receives the
shared ModelIR and conversion-local layout state, and their independent
mapping results are returned as an ordered tuple.

The lowerer retains one `_late_conv1d_decoder_layout_results` composite via
`shared_model_ir_pass_context`, outside the full phase-result store. The eight
old unconsumed locals and their long inline call block are absent. All eight
compatibility wrappers, owner implementations, indexed graph behavior, late
Swish predecessor, and very-late Pad successor remain intact.

Sequential validation under core-only `uv` completed with:

- focused composite order, context identity, tuple, and boundary contracts:
  `3 passed in 0.65s`;
- indexed Conv1D/decoder mutations plus affected result contracts:
  `431 passed in 2.01s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.93s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- broader result contracts: `196 passed in 9.27s`;
- full lowerer architecture contracts: `258 passed in 19.39s`;
- phase-result capacity contracts: `2 passed in 0.53s`;
- targeted Ruff, bytecode compilation, fixed-capacity audit, and whitespace
  checks: passed.

No callback, graph rewrite, execution count, ordering, result mapping,
LayoutState identity, public compatibility name, artifact, dependency, or
TensorFlow boundary changed. No root-model conversion was repeated because the
431-test indexed family gate exercises the extracted owner paths. The store
remains exactly 128/128.

## Very-late Pad/InstanceNorm composite characterization

The next selected unit contains very-late Pad cleanup followed by three
InstanceNorm repairs: post-bias, residual-Mul/Concat, and dual-stat residual.
All four calls are unconditional and adjacent between the late Conv1D/decoder
composite and singleton/consecutive-Reshape cluster. Pad receives layout state
and diagnostics; the three InstanceNorm owners receive layout state only. All
four result mappings are unconsumed.

The focused characterization fixes source order, exact argument policy, outer
boundaries, and absence of consumers. A strict expected failure requires one
`run_very_late_pad_instancenorm_layout_cleanup(shared_model_ir_pass_context)`
owner and an ordered `_very_late_pad_instancenorm_layout_results` tuple outside
the full store. No production source changed.

Implementation must import all four existing pass owners directly, preserve
their compatibility wrappers and independent mappings, keep the store fixed at
128/128, and never create, update, or reopen a pull request.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.15s`; the sole xfail is the intentionally absent
composite owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

## Very-late Pad/InstanceNorm composite implementation checkpoint

`run_very_late_pad_instancenorm_layout_cleanup` now owns the four
characterized callbacks. It runs Pad cleanup first with the conversion-local
layout state and diagnostics, then runs the post-bias, residual-Mul/Concat,
and dual-stat InstanceNorm repairs with the same ModelIR and layout state.
The four independent counter mappings are returned in their original order.

The lowerer retains `_very_late_pad_instancenorm_layout_results` as an ordered
tuple outside the already-full phase-result store. The four old unconsumed
locals are gone. Existing pass owners, lowerer compatibility wrappers, the
late Conv1D/decoder predecessor, and the singleton/consecutive-Reshape
successor remain unchanged. Focused runtime tests prove exact call order,
context identity, argument policy, and tuple order.

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

No graph traversal, mutation, numerical behavior, diagnostics, public API,
artifact, dependency, or TensorFlow boundary changed. No real-model
conversion was run because this checkpoint only extracts four already
adjacent calls behind a focused owner. Commit and push this implementation
checkpoint. On resume, audit the next coherent non-store source-order unit
after the singleton/consecutive-Reshape cluster. Continue with commits and
pushes only; never create, update, or reopen a pull request.

## Very-late layout/broadcast composite characterization

The next non-store boundary consists of the guarded final layout-Transpose
cleanup followed by unconditional rank-four channelwise broadcast-constant
repair. Both results are unconsumed. The existing unconditional
`shape_reconciliation.primary.very_late_broadcast` record remains immediately
after them and is explicitly outside this extraction.

The focused characterization fixes the normalized
`optimize_layout_transpose_chains` guard, exact ModelIR/layout/diagnostics
arguments, unconditional broadcast call, singleton/consecutive-Reshape
predecessor, reconciliation successor, and absence of result consumers. A
strict expected failure requires one
`run_very_late_layout_broadcast_cleanup(shared_model_ir_pass_context,
include_layout_transpose=optimize_layout_transpose_chains)` assignment and an
ordered `_very_late_layout_broadcast_results` tuple outside the full store.

Implementation must preserve the conditional execution count, import the two
existing pass owners directly, return `None` for the skipped optional result,
keep the broadcast result mapping intact, retain all compatibility wrappers,
and leave the reconciliation record and 128/128 store unchanged. No
production source changed in this checkpoint.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.15s`; the sole xfail is the intentionally absent
composite owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

## Very-late layout/broadcast composite implementation

`run_very_late_layout_broadcast_cleanup` now owns the characterized boundary.
It invokes layout-Transpose cleanup only when the normalized layout option is
enabled, with the same ModelIR, LayoutState, and diagnostics. It then invokes
rank-four channelwise broadcast-constant repair unconditionally with the same
ModelIR. The ordered result contains `None` when the optional first pass is
skipped and always retains the broadcast mapping.

The lowerer now retains one `_very_late_layout_broadcast_results` tuple
outside the full phase store. The old conditional local, unconditional local,
and inline guard are gone. The singleton/consecutive-Reshape predecessor and
unconditional `shape_reconciliation.primary.very_late_broadcast` successor
remain adjacent to the new owner. Both existing pass owners and all lowerer
compatibility wrappers remain available.

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

No pass execution count, graph traversal, mutation, numerical behavior,
diagnostics, public API, artifact, dependency, or TensorFlow boundary changed.
No real-model conversion was run because the focused owner tests exercise both
flag paths and the affected suites cover both underlying repair families.
Commit and push this implementation checkpoint. On resume, audit the next
coherent non-store boundary after the very-late broadcast reconciliation.
Continue with commits and pushes only; never create, update, or reopen a pull
request.

## Shared-late reconciliation decision characterization

The next boundary is not an observation-only tuple. It runs six owners that
produce nine mutation dictionaries, compares the tensor count before and
after cleanup to catch prune-only changes, and conditionally invokes the
already-recorded shared-late static-shape reconciliation. The predicate is an
actual execution guard and must remain invoked-phase-only.

The focused characterization fixes all nine evidence positions, the initial
tensor-count snapshot, exact positive-counter predicate, prune-delta fallback,
direct reconciliation call, stable phase ID, and late-binary successor. A
strict expected failure requires one boolean
`run_shared_late_reconciliation_cleanup(shared_model_ir_pass_context)` result
between the very-late broadcast record and the unchanged reconciliation
guard.

The proposed owner may absorb only the six cleanup calls and decision. It
must call boundary-signature, HardSwish, Squeeze, Conv-input, indexed binary
adapter, and singleton/consecutive-Reshape owners in the same order and with
the same arguments. The lowerer must continue to own the conditional
`session.record_phase_result` call so the nested reconciliation owner and
invoked-phase-only store semantics remain unchanged. No production source
changed in this checkpoint.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.13s`; the sole xfail is the intentionally absent
boolean owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

## Shared-late reconciliation decision implementation

`run_shared_late_reconciliation_cleanup` now owns the six characterized
cleanup calls and their decision. It preserves the four model-only sanitizer
calls, model-only indexed binary adapter call, and shared-context
singleton/consecutive-Reshape call in source order. The owner evaluates all
nine integer mappings and the original tensor-count decrease, returning only
a boolean reconciliation requirement.

The lowerer retains `_shared_late_requires_reconciliation` and continues to
own the conditional `shape_reconciliation.primary.shared_late` record. Its
record still directly invokes `_reconcile_static_tensor_shapes(model_ir,
include_mutation_count=True)`, so invoked-phase-only semantics, owner identity,
and store representation remain unchanged. Nine evidence locals and the
tensor-count local were removed from the lowerer without hiding the actual
reconciliation behind the new owner.

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

No cleanup or reconciliation execution count, graph traversal, guard result,
ModelIR mutation, diagnostics, public API, artifact, dependency, or TensorFlow
boundary changed. No real-model conversion was run because focused unit tests
force every evidence path and an integration test proves the boolean still
adds exactly one conditional reconciliation. Commit and push this checkpoint.
On resume, audit the late-binary repair decision immediately following this
boundary. Continue with commits and pushes only; never create, update, or
reopen a pull request.

## Late-binary repair decision characterization

The adjacent late-binary repair boundary snapshots tensor count, sanitizes
static shape signatures, runs the indexed binary adapter pair, and triggers
its already-recorded reconciliation when one of three named counters is
positive or cleanup pruned a tensor. This predicate is control flow and
remains separate from the following optional late-binary layout recovery.

The focused characterization fixes the three counter keys, tensor-count
fallback, evidence order, direct reconciliation record, shared-late guard
predecessor, and optional recovery successor. A strict expected failure
requires one
`run_late_binary_repair_cleanup(shared_model_ir_pass_context)` boolean result
followed by the same lowerer-owned reconciliation guard.

Implementation may absorb only signature sanitization, the indexed adapter
pair, and their decision. The lowerer must continue to own the direct
`_reconcile_static_tensor_shapes` record, and the optional layout-recovery
outer and inner guards must remain untouched. No production source changed in
this checkpoint.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.14s`; the sole xfail is the intentionally absent
boolean owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

## Late-binary repair decision implementation

`run_late_binary_repair_cleanup` now owns the characterized signature
sanitizer, indexed binary adapter pair, and their reconciliation decision. It
captures the tensor count before either repair, calls both existing pass
owners in the original order with the same ModelIR instance, and reduces the
three exact mutation counters plus cleanup-only tensor pruning to one boolean.

The lowerer replaces three consumed evidence mappings and one tensor-count
snapshot with `_late_binary_repair_requires_reconciliation`. It continues to
own the conditional `shape_reconciliation.primary.late_binary_repair` record
and its direct `_reconcile_static_tensor_shapes(model_ir,
include_mutation_count=True)` call. The following normalized option guard and
the mutation-positive inner guard for late-binary layout recovery are
unchanged. The phase-result store remains exactly 128/128.

Final sequential validation under core-only `uv`:

- focused stable, three-evidence, prune-only, order, and boundary contracts:
  `7 passed in 0.56s`;
- affected late-binary, shared-late, adapter, terminal, core, and store
  contracts: `132 passed in 2.78s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.88s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.09s`;
- full architecture contracts: `258 passed in 19.02s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No pass call, graph traversal, reconciliation trigger, ModelIR mutation,
diagnostic, public API, artifact, dependency, or TensorFlow boundary changed.
No real-model conversion was run because focused runtime tests exercise the
stable path, each of the three evidence paths, and prune-only cleanup, while
the lowerer integration test proves that the returned boolean adds exactly
one reconciliation record. Commit and push this checkpoint. On resume,
characterize the following optional late-binary layout-recovery decision
before changing production code. Continue with commits and pushes only; never
create, update, or reopen a pull request.

## Optional late-binary layout-recovery decision characterization

The next adjacent boundary conditionally runs the existing aggregate
late-binary layout-recovery owner when either full layout optimization or the
safe no-layout reduction mode is enabled. A second guard reconciles static
shapes only when the returned recovery summary contains a positive mutation
count. The reconciliation record remains the direct lowerer-owned
`shape_reconciliation.primary.late_binary_layout_recovery` boundary.

The focused characterization fixes the normalized outer enablement predicate,
aggregate-owner arguments, positive-summary predicate, direct reconciliation
call, late-binary repair predecessor, and pre-terminal InstanceNorm successor.
A strict expected failure requires one boolean
`run_optional_late_binary_layout_recovery_cleanup` result that may absorb only
the optional aggregate call and its mutation decision.

The future owner must skip the aggregate call completely when disabled, pass
the same ModelIR/LayoutState/diagnostics objects when enabled, preserve the
independent `include_layout_transpose` flag, and return `False` for a stable
summary. It must not perform reconciliation or record phase evidence. No
production source changed in this checkpoint.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.16s`; the sole xfail is the intentionally absent
boolean owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

Commit and push this characterization before production changes. Keep the
store exactly 128/128 and continue with commits and pushes only; never create,
update, or reopen a pull request.

## Optional late-binary layout-recovery decision implementation

`run_optional_late_binary_layout_recovery_cleanup` now owns the normalized
enablement check, the existing aggregate recovery call, and its positive-count
decision. When disabled it returns `False` without invoking recovery. When
enabled it forwards the same ModelIR, LayoutState, diagnostics list, and
independent layout-Transpose flag, then returns whether any aggregate counter
is positive.

The lowerer replaces the nested option branch, consumed aggregate result, and
positive-summary branch with
`_late_binary_layout_recovery_requires_reconciliation`. It continues to own
the conditional `shape_reconciliation.primary.late_binary_layout_recovery`
record and its direct `_reconcile_static_tensor_shapes(model_ir,
include_mutation_count=True)` call. The preceding late-binary repair decision
and following pre-terminal InstanceNorm cleanup remain adjacent. The bounded
store remains exactly 128/128.

Final sequential validation under core-only `uv`:

- focused disabled, stable, positive, flag, identity, and boundary contracts:
  `6 passed in 0.57s`;
- affected recovery, repair, terminal, core, and store contracts:
  `138 passed in 2.82s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.99s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.15s`;
- full architecture contracts: `258 passed in 18.96s`;
- phase-store capacity contracts: `2 passed in 0.55s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No recovery call, repair order, mutation counter, reconciliation trigger,
ModelIR mutation, diagnostic, public API, artifact, dependency, or TensorFlow
boundary changed. No real-model conversion was run because runtime tests cover
the disabled path, stable enabled path, positive enabled paths with both layout
flag values, exact context identity, and lowerer reconciliation integration.
Commit and push this checkpoint. On resume, characterize the next coherent
pre-terminal affine/InstanceNorm decision boundary before production changes.
Continue with commits and pushes only; never create, update, or reopen a pull
request.

## Pre-terminal InstanceNorm layout composite characterization

The next source-order cluster runs three InstanceNorm layout repairs directly
after optional late-binary layout recovery: post-bias, residual-Mul/Concat,
and dual-stat residual-Add/Resize. All three receive the same ModelIR and
conversion-local LayoutState. Their result mappings are retained only in
lowerer locals and have no consumers.

The focused characterization fixes the three targets, exact pass order,
model/layout argument policy, late-binary recovery predecessor, first
terminal-affine tensor-count successor, and absence of result loads. A strict
expected failure requires one ordered
`run_pre_terminal_instancenorm_layout_cleanup(shared_model_ir_pass_context)`
composite result outside the full phase store.

The future owner must import the three pass-module owners directly, preserve
their independent mappings in a fixed tuple, and forward the identical
ModelIR/LayoutState objects to every call. It must not summarize counters,
record phase evidence, or absorb either neighboring decision. No production
source changed in this checkpoint.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.14s`; the sole xfail is the intentionally absent
composite owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

Commit and push this characterization before production changes. Keep the
store at 128/128 and continue with commits and pushes only; never create,
update, or reopen a pull request.

## Pre-terminal InstanceNorm layout composite implementation

`run_pre_terminal_instancenorm_layout_cleanup` now owns the three
characterized InstanceNorm repairs. It imports the pass-module owners directly,
calls them in post-bias, residual-Mul/Concat, and dual-stat order, forwards the
same ModelIR and LayoutState to every call, and returns all three independent
mappings in a fixed tuple.

The lowerer replaces three unconsumed locals with one
`_pre_terminal_instancenorm_layout_results` assignment outside the full phase
store. The optional late-binary recovery reconciliation remains immediately
before the composite, and the first terminal-affine tensor-count snapshot
remains immediately after it. Existing lowerer compatibility wrappers and all
other production call sites are unchanged. The store remains exactly 128/128.

Final sequential validation under core-only `uv`:

- focused order, mapping, identity, and boundary contracts:
  `3 passed in 0.59s`;
- affected InstanceNorm, terminal-affine, absolute-final, core, and store
  contracts: `151 passed in 3.50s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.88s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.13s`;
- full architecture contracts: `258 passed in 19.45s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No graph traversal, repair call, mutation mapping, callback order, layout
state, diagnostics, public API, artifact, dependency, or TensorFlow boundary
changed. No real-model conversion was run because the owner runtime contract
proves exact call order, mapping identity, and shared context identity, while
owner-aware architecture contracts prove the unchanged total call counts.
Commit and push this checkpoint. On resume, characterize the adjacent first
terminal-affine recovery evidence boundary before production changes. Continue
with commits and pushes only; never create, update, or reopen a pull request.

## Terminal-affine prune-aware summary characterization

The terminal-affine/Concat/Split recovery sequence runs twice in the late
pipeline. Each boundary currently repeats the same three-step evidence logic:
snapshot tensor count, retain the raw eleven-pass result tuple, and normalize
its declared counters plus cleanup-only tensor pruning. Both normalized
mappings are unconsumed, while the raw lowerer wrapper remains a compatibility
boundary.

The focused characterization fixes both triples, their exact summary
expressions, raw wrapper dispatch, source order, pre-terminal InstanceNorm and
slice/pad predecessors, and the pre-add and terminal slice/pad successors. A
strict expected failure requires both summary targets to call one reusable
`run_terminal_affine_concat_split_recovery_summary` owner.

The future owner must snapshot the tensor count before invoking the existing
raw pass-module owner, reuse the existing strict summary function, preserve
all eleven result schemas and prune accounting, and return only the normalized
mapping. The raw lowerer wrapper must remain defined with its existing
dispatch for compatibility. No production source changed in this checkpoint.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.13s`; the sole xfail is the intentionally absent
summary owner. Targeted Ruff, bytecode compilation, and whitespace checks
passed.

Commit and push this characterization before production changes. Keep the
store at 128/128 and continue with commits and pushes only; never create,
update, or reopen a pull request.

## Terminal-affine prune-aware summary implementation

`run_terminal_affine_concat_split_recovery_summary` now snapshots tensor
count, invokes the existing raw eleven-pass recovery owner, and delegates to
the existing strict summary function with the original cleanup-only prune
delta. It preserves the raw tuple schema, declared mutation-key validation,
integer normalization, and non-negative prune count.

Both lowerer boundaries now assign their existing `_pre_terminal_affine_stats`
and `_terminal_affine_stats` targets directly from this owner. Four consumed
intermediate locals and two duplicated summary expressions were removed. The
nested `_run_terminal_affine_concat_split_recovery_sequence` compatibility
wrapper remains defined and continues to dispatch to the raw owner with the
same context. All four neighboring boundaries and the 128/128 store remain
unchanged.

Final sequential validation under core-only `uv`:

- focused stable/pruned owner, wrapper, and dual-boundary contracts:
  `4 passed in 0.59s`;
- affected terminal-affine, pre-add, slice/pad, InstanceNorm, core, and store
  contracts: `182 passed in 2.14s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 2.02s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.17s`;
- full architecture contracts: `258 passed in 18.81s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- Ruff, bytecode compilation, 128/128 capacity audit, and whitespace checks:
  passed.

No recovery invocation, pass order, mutation schema, prune semantics, ModelIR
mutation, layout state, diagnostics, public API, artifact, dependency, or
TensorFlow boundary changed. No real-model conversion was run because focused
runtime tests cover stable and prune-only paths, and the existing raw recovery
tests cover all eleven pass results. Commit and push this checkpoint. On
resume, characterize the adjacent pre-terminal pre-add prune-aware evidence
boundary before production changes. Continue with commits and pushes only;
never create, update, or reopen a pull request.

## Guarded terminal BatchMatMul implementation

The three characterized results now record inside their original guard under:

- `cleanup.terminal.batchmatmul_affine_input`;
- `cleanup.terminal.batchmatmul_reshape_se`;
- `cleanup.terminal.batchmatmul_adj_flags`.

Only the unused local destinations changed. The
`optimize_layout_transpose_chains` guard, owner calls, arguments,
three-statement order, preceding Mean-attention composite, following
QKV-attention composite, ModelIR mutations, post-SiNet observations, public
outputs, reports, artifacts, dependencies, and TensorFlow isolation remain
unchanged. Because records remain inside the guard, invoked-phase-only
semantics are preserved. The bounded store now covers 109/128 phase IDs,
leaving 19 slots.

Five representation-dependent owner and QKV boundary assertions now unwrap
the phase record and identify the guard through the retained composites. They
continue to verify both production call sites and all non-migrated post-SiNet
assignments.

Validation completed sequentially under core-only `uv`:

- focused BatchMatMul/QKV/store contracts: `26 passed in 1.22s`;
- synthetic core runtime contracts: `55 passed in 1.05s`;
- broader result and phase-result contracts: `189 passed in 9.37s`;
- lowerer architecture contracts: `258 passed in 17.61s`;
- targeted Ruff, Python bytecode compilation, AST capacity audit, and
  whitespace validation: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and the synthetic runtime suite exercises
the guarded terminal path.

## Guarded terminal QKV bridge characterization

The next source-order semantic unit is the single Split/Conv/Concat bridge
mapping observation between the retained QKV-attention and singleton-reshape
composites inside `optimize_layout_transpose_chains`.

Existing indexed owner tests fix its single integer counter, no-op behavior,
fan-out handling, dynamic shapes, quantization preservation, invariant
validation, and transactional rollback. The local has no default or consumer.
The characterization fixes the guard, exact owner expression and keyword,
both composite boundaries, and absence of loads. A strict expected failure
requires `cleanup.terminal.qkv_split_conv_concat_bridge` in the same position.
No production source changed.

Validation completed sequentially under core-only `uv`:

- related bridge/QKV/singleton baseline: `103 passed in 1.05s`;
- characterization plus related contracts: `104 passed, 1 xfailed in 1.13s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented destination
migration.

## Guarded terminal QKV bridge implementation

The characterized bridge observation now records as
`cleanup.terminal.qkv_split_conv_concat_bridge` inside its original
`optimize_layout_transpose_chains` guard.

Only the unused local destination changed. The indexed owner call, argument,
layout-state keyword, evaluation count, QKV-attention predecessor,
singleton-reshape successor, post-SiNet and later bridge observations, ModelIR
mutation, public outputs, reports, artifacts, dependencies, and TensorFlow
isolation remain unchanged. The retained guard preserves invoked-phase-only
semantics. The bounded store now covers 110/128 phase IDs, leaving 18 slots.

Two representation-dependent bridge and singleton boundary tests now unwrap
the record while continuing to validate all three owner call sites and both
composite boundaries.

Validation completed sequentially under core-only `uv`:

- focused bridge/QKV/singleton/store contracts: `106 passed in 1.37s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader result and phase-result contracts: `190 passed in 9.42s`;
- lowerer architecture contracts: `258 passed in 17.17s`;
- targeted Ruff, Python bytecode compilation, AST capacity audit, and
  whitespace validation: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and the indexed synthetic runtime suite
exercises the bridge implementation.

## Terminal HardSwish/HardSigmoid characterization

The next source-order pair contains the unconditional SiNet HardSwish-SE and
Dequantize/HardSigmoid/Quantize bridge observations between retained SiNet
terminal-layout and pre-Add/Resize recovery composites.

Existing focused contracts already fix both integer schemas, owner cleanup,
all production forms, call arguments, adjacency, and absence of consumers. To
avoid adding another large structural file, the strict phase-record contract
extends the existing HardSwish-SE result module. It requires two consecutive
stable `cleanup.terminal.*` records with exact owners and both composite
boundaries. No production source changed.

Validation completed sequentially under core-only `uv`:

- related HardSwish/HardSigmoid/SiNet baseline: `21 passed in 1.00s`;
- characterization plus related contracts: `21 passed, 1 xfailed in 1.10s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented two-result
destination migration.

## Terminal HardSwish/HardSigmoid implementation

The two characterized observations now record consecutively under:

- `cleanup.terminal.sinet_hardswish_se`;
- `cleanup.terminal.dequant_hardsigmoid_bridge`.

Only the unused local destinations changed. Both owner calls, arguments,
unconditional execution, evaluation count, SiNet terminal-layout predecessor,
pre-Add/Resize successor, later owner forms, ModelIR mutations, public outputs,
reports, artifacts, dependencies, and TensorFlow isolation are unchanged. The
bounded store now covers 112/128 phase IDs, leaving 16 slots.

Six representation-dependent result, SiNet orchestration, and architecture
assertions now unwrap phase records. They continue to verify the exact two
phase IDs, owner names, all later production forms, and retained composite
result names. The characterization was folded into the existing focused test,
so no additional structural test file remains.

Validation completed sequentially under core-only `uv`:

- focused HardSwish/HardSigmoid/SiNet/store contracts:
  `23 passed in 1.21s`;
- synthetic core runtime contracts: `55 passed in 1.07s`;
- broader result and phase-result contracts: `190 passed in 9.59s`;
- lowerer architecture contracts: `258 passed in 20.53s`;
- targeted Ruff, Python bytecode compilation, AST capacity audit, and
  whitespace validation: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and the existing focused runtime tests
exercise both owners.

## Post-terminal indexed shape convergence characterization

The next source-order non-composite boundary is the top-level indexed
shape/topology convergence observation between the post-terminal singleton-
reshape composite and very-late SiNet recovery composite.

The owner shares one `ModelIRGraphIndex` across dead-operator pruning, dynamic
Reshape resolution, and static-shape reconciliation, and returns three bounded
integer counters. Its local result has no default or consumer. The existing
result module now adds a strict expected failure for
`shape_topology.terminal.indexed_convergence`, fixing the exact owner
arguments, both composite boundaries, and preservation of the separate nested
convergence result. No production source changed.

Validation completed sequentially under core-only `uv`:

- related indexed-convergence/SiNet baseline: `10 passed in 0.71s`;
- characterization plus related contracts: `10 passed, 1 xfailed in 0.78s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented top-level
destination migration.

## Post-terminal indexed shape convergence implementation

The characterized top-level observation now records as
`shape_topology.terminal.indexed_convergence` between its original singleton
and very-late SiNet composites.

Only the unused top-level local destination changed. The owner call, model and
layout-state arguments, one-index convergence logic, three-counter schema,
nested `convergence_stats` result, evaluation count, ModelIR mutations, public
outputs, reports, artifacts, dependencies, and TensorFlow isolation remain
unchanged. The bounded store now covers 113/128 phase IDs, leaving 15 slots.

Four representation-dependent focused and architecture assertions now unwrap
the record and verify the exact phase and owner while retaining both composite
and nested-result checks. The temporary expected failure was converted in the
existing focused module.

Validation completed sequentially under core-only `uv`:

- focused indexed-convergence/SiNet/store contracts: `12 passed in 0.81s`;
- synthetic core runtime contracts: `55 passed in 1.02s`;
- broader result and phase-result contracts: `190 passed in 8.96s`;
- lowerer architecture contracts: `258 passed in 19.09s`;
- targeted Ruff, Python bytecode compilation, AST capacity audit, and
  whitespace validation: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and existing synthetic contracts exercise
the indexed convergence owner.

## Very-late residual cleanup characterization

The next source-order family contains three consecutive unconditional mapping
observations after very-late SiNet recovery: residual affine PReLU cleanup,
residual affine Transpose fan-out cleanup, and indexed prune/reconcile cleanup.

Existing focused contracts fix their integer schemas, unconditional tensor
cleanup, orchestration routes, exact arguments, one-index prune/reconcile
behavior, and absence of consumers. The existing prune/reconcile result module
now strictly requires three `cleanup.very_late.*` records with exact owners,
adjacency, the preceding very-late SiNet pre-Add/Resize composite, and the
following post-cleanup SiNet composite. No production source changed.

Validation completed sequentially under core-only `uv`:

- related residual/prune/SiNet baseline: `17 passed in 0.98s`;
- characterization plus related contracts: `17 passed, 1 xfailed in 1.01s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented three-result
destination migration.

## Layout pass-set 1 affine cleanup implementation

The five characterized observations now record under stable
`cleanup.layout_pass_set_1.*` affine phase IDs. The first four records remain a
contiguous prefix between initial attention recovery and mean-attention
recovery. The post-binary affine-chain fold remains between post-binary
attention recovery and the attention/quantized suffix.

Only the five unconsumed mapping destinations changed. Owner calls, arguments,
keywords, evaluation counts, the outer layout guard, graph traversal, ModelIR
mutation, composite recovery boundaries, public results, reports, artifacts,
dependencies, and TensorFlow import boundaries are unchanged. No defaults
existed. The bounded store now covers 77 phase IDs.

The broader result-contract run exposed one additional stale test helper that
read the previously migrated no-layout safe-transpose phase record as an outer
call. The ConvPool boundary helper now unwraps and verifies the exact nested
owner. Its targeted correction is `2 passed in 0.14s`; production behavior was
not implicated.

Validation completed sequentially under core-only `uv`:

- direct affine, multi-occurrence owner, phase-store, and adjacent architecture
  contracts: `13 passed in 2.90s`;
- synthetic core runtime contracts: `55 passed in 1.00s`;
- broader result and phase-result contracts: `180 passed in 8.24s`;
- lowerer architecture contracts: `258 passed in 16.87s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the synthetic runtime suite executes the guarded
layout path.

## Layout pass-set 1 residual cleanup characterization

The remaining direct mapping observations in layout pass-set 1 are limited to
four results: primary layout-Transpose cleanup, the conditionally enabled
Transpose/binary bridge, duplicate fan-out cleanup, and
Dequantize→Mean→Quantize bridge cleanup. Each has an existing bounded integer
schema contract and no default or consumer.

The characterization fixes the outer layout guard, nested binary-feature
guard, exact owner arguments and keywords, policy predecessor, composite
attention/quantized and safe-binary boundaries, and absence of loads.
Composite recovery results remain excluded. A strict expected failure requires
four stable `cleanup.layout_pass_set_1.*` records. No production source
changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization and existing owner/schema contracts:
  `9 passed, 1 xfailed in 0.99s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented four-result
destination migration.

## Layout pass-set 1 residual cleanup implementation

The four remaining direct observations now record under stable
`cleanup.layout_pass_set_1.*` phase IDs. Primary layout-Transpose cleanup,
duplicate fan-out cleanup, and Dequantize→Mean→Quantize cleanup remain in the
outer layout guard. The Transpose/binary bridge record remains inside its
original feature guard and is absent when that guard is false.

Only the four unconsumed mapping destinations changed. Owner calls, arguments,
keywords, evaluation counts, both guards, policy selection, graph traversal,
ModelIR mutation, composite recovery boundaries, public results, reports,
artifacts, dependencies, and TensorFlow import boundaries are unchanged. No
defaults existed. The bounded store now covers 81 phase IDs.

Architecture and orchestration expansion exposed three stale structural
assertions that accessed an adjacent owner as an outer `ast.Name` or required
all three layout-cleanup owners to be assignments. They now verify the exact
nested phase owner and distinguish the primary record from the two retained
late assignments. Targeted corrections are `2 passed in 2.28s` and
`1 passed in 0.56s`; production behavior was not implicated.

Validation completed sequentially under core-only `uv`:

- direct residual, owner, phase-store, QLinear-boundary, and terminal-boundary
  contracts: `14 passed in 1.61s`;
- synthetic core runtime contracts: `55 passed in 1.06s`;
- broader result and phase-result contracts: `182 passed in 8.70s`;
- QLinear and terminal-layout orchestration contracts:
  `71 passed in 1.96s`;
- lowerer architecture contracts: `258 passed in 17.42s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the synthetic runtime suite executes the guarded
layout paths.

## Layout pass-set 2 residual cleanup characterization

The remaining direct mapping observations in layout pass-set 2 are limited to
nine results. Eight consecutive results cover elementwise/Concat/Conv, SPP,
pre-Concat, NDHWC Concat, StridedSlice pre-Concat, mixed-Split pre-Concat,
Concat input-adapter, and Slice/Logistic/Concat-tail cleanup. The ninth covers
SA/PA MirrorPad propagation after the channel-shuffle and attention composite
clusters.

All nine share the existing `optimize_layout_transpose_chains` guard, have
bounded integer schemas fixed by existing owner tests, and have no defaults or
consumers. The characterization fixes exact owner calls and keywords, the
eight-result adjacency, composite predecessor/successor boundaries, the
isolated SA/PA boundary, and absence of loads. Composite recovery results stay
outside the migration. A strict expected failure requires nine stable
`cleanup.layout_pass_set_2.*` records. No production source changed.

Validation completed sequentially under core-only `uv`:

- existing owner/schema baseline: `22 passed in 1.42s`;
- dedicated characterization plus existing owner/schema contracts:
  `23 passed, 1 xfailed in 1.59s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented nine-result
destination migration.

## Layout pass-set 2 residual cleanup implementation

The nine characterized observations now record under stable
`cleanup.layout_pass_set_2.*` phase IDs. The first eight records remain
consecutive between the quantized-activation/binary composite and the
channel-shuffle/Gather composite. The SA/PA MirrorPad record remains between
the pre-add/mean-attention and gate-layout composites.

Only the nine unconsumed mapping destinations changed. Owner calls, arguments,
keywords, evaluation counts, the outer layout guard, graph traversal, ModelIR
mutation, composite recovery boundaries, public results, reports, artifacts,
dependencies, and TensorFlow import boundaries are unchanged. No defaults
existed. The bounded store now covers 90 phase IDs.

Expanded orchestration validation exposed three stale helper expectations that
could not unwrap previously migrated quantized and Dequantize→Mean→Quantize
phase owners. The shared test helpers now resolve the exact nested owner. The
targeted correction is `3 passed in 0.62s`; production behavior was not
implicated.

Validation completed sequentially under core-only `uv`:

- direct nine-result owner/schema contracts: `24 passed in 1.60s`;
- expanded owner, phase-store, attention, gate, quantized-recovery, and
  architecture-boundary contracts: `64 passed in 2.38s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result and phase-result contracts: `184 passed in 11.76s`;
- lowerer architecture contracts: `258 passed in 16.96s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the synthetic runtime suite executes the guarded
layout path.

## Terminal boundary cleanup characterization

The next selected family contains seven consecutive unconditional mapping
observations immediately after the existing terminal Conv-activation record:
pre-ArgMax, Transpose/Gather channel fan-out, terminal Softmax/Transpose,
boundary-input normalization, boundary-input channel slicing, internal channel
slicing, and channel-slice Mul/Add bridge cleanup.

All seven have explicit bounded integer schemas, no defaults, and no
consumers. The characterization fixes exact owner calls and keywords,
seven-statement adjacency, the preceding `cleanup.terminal.conv_activation`
record, the following terminal Slice/Concat composite, and absence of loads. A
strict expected failure requires seven stable `cleanup.terminal.*` records. No
production source changed.

Validation completed sequentially under core-only `uv`:

- related owner/runtime baseline: `152 passed in 2.31s`;
- characterization plus related owner/runtime contracts:
  `153 passed, 1 xfailed in 2.51s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented seven-result
destination migration.

## Terminal boundary cleanup implementation

The seven characterized observations now record under stable
`cleanup.terminal.*` phase IDs. They remain consecutive and unconditional,
directly after terminal Conv activation and directly before the terminal
Slice/Concat recovery composite.

Only the seven unconsumed mapping destinations changed. Owner calls,
arguments, keywords, evaluation counts, graph traversal, ModelIR mutation,
source order, composite boundary, public results, reports, artifacts,
dependencies, and TensorFlow import boundaries are unchanged. No defaults
existed. The bounded store now covers 97 phase IDs.

Expanded validation exposed 13 stale structural assertions across terminal
owner, Slice/Concat boundary, prior terminal-cleanup boundary, and architecture
contracts. They expected outer assignments or outer `ast.Name` calls. The
contracts now verify exact phase IDs and nested owners while retaining the
later non-migrated assignments. Targeted corrections are `11 passed in 1.13s`,
`2 passed in 0.35s`, and `1 passed in 2.67s`; production behavior was not
implicated.

Validation completed sequentially under core-only `uv`:

- terminal boundary, phase-store, owner, and runtime contracts:
  `156 passed in 2.72s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader result and phase-result contracts: `186 passed in 9.83s`;
- lowerer architecture contracts: `258 passed in 18.19s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the synthetic runtime suite executes the terminal
path.

## Terminal activation cleanup characterization

The next family contains four consecutive unconditional mapping observations
after terminal Slice/Concat recovery: boundary-input
StridedSlice/QDQ/Concat cleanup, Swish residual/Concat closure,
Dequantize/Logistic/Mul/Quantize bridging, and Swish QDQ-island cleanup.

All four have bounded integer schemas fixed by existing terminal contracts,
have no defaults or consumers, and remain between the Slice/Concat composite
and terminal InstanceNorm cleanup. The characterization fixes owner calls and
keywords, four-statement adjacency, both boundaries, and absence of loads. A
strict expected failure requires four stable `cleanup.terminal.*` records. No
production source changed.

Validation completed sequentially under core-only `uv`:

- related terminal baseline: `72 passed in 1.99s`;
- characterization plus related terminal contracts:
  `73 passed, 1 xfailed in 2.08s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented four-result
destination migration.

## Terminal activation cleanup implementation

The four characterized terminal activation results now record under:

- `cleanup.terminal.boundary_stridedslice_qdq_concat`;
- `cleanup.terminal.swish_residual_concat_closure`;
- `cleanup.terminal.dequant_logistic_mul_quantize_bridge`;
- `cleanup.terminal.swish_qdq_island`.

Only the unused local destinations changed. The owner calls, arguments,
keywords, unconditional execution, evaluation count, four-statement order,
preceding Slice/Concat composite, following InstanceNorm cleanup, graph
mutation, public outputs, reports, artifacts, dependencies, and TensorFlow
boundary remain unchanged. The bounded store now covers 101 of its 128 phase
slots.

Six stale structural assertions expected an outer assignment or direct owner
call. They now unwrap the bounded record and continue to verify the exact
phase ID, owner, arguments, keywords, and neighboring pass boundaries.

Validation completed sequentially under core-only `uv`:

- direct activation, phase-store, terminal-owner, and Slice/Concat contracts:
  `75 passed in 2.42s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader result and phase-result contracts: `187 passed in 8.61s`;
- lowerer architecture contracts: `258 passed in 18.51s`;
- targeted Ruff, Python bytecode compilation, AST capacity audit, and
  whitespace validation: passed.

No root-model corpus conversion was run because this is an
observation-destination-only change and the synthetic runtime suite exercises
the terminal path.

## Terminal normalization cleanup characterization

The next bounded family contains five consecutive unconditional mapping
observations after terminal activation cleanup: InstanceNorm post-bias,
normalization Pad, InstanceNorm residual Add, InstanceNorm residual
Mul/Concat, and InstanceNorm dual-stat cleanup.

Existing owner tests fix each bounded integer schema. None of the five locals
has a default or consumer. The characterization fixes all owner expressions,
arguments and keywords, five-statement adjacency, the preceding Swish
QDQ-island phase, the following terminal boundary-layout composite, and
absence of loads. A strict expected failure requires five stable
`cleanup.terminal.*` records. No production source changed.

Validation completed sequentially under core-only `uv`:

- related terminal baseline: `97 passed in 2.59s`;
- characterization plus the related terminal contracts:
  `98 passed, 1 xfailed in 2.59s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented five-result
destination migration.

## Primary final SiNet reconciliation implementation

The six ordered SiNet results now record under:

- `shape_reconciliation.primary.final_sinet_late_residual`;
- `shape_reconciliation.primary.final_sinet_preadd_fanout`;
- `shape_reconciliation.primary.final_sinet_dual_resize`;
- `shape_reconciliation.primary.final_sinet_shared_post`;
- `shape_reconciliation.primary.final_sinet_deep_skip`;
- `shape_reconciliation.primary.final_sinet_concat_resize`.

Each record remains directly behind its original dedicated mutation guard and
between the same neighboring SiNet repairs. The six zero defaults and local
targets were removed. No earlier late/static-shape observation was included.

No owner argument, guard, reconciliation, repair order, graph scan, mutation,
successor, public result, report, artifact, dependency, or TensorFlow boundary
changed. The bounded store now covers 45 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct ordered-chain, terminal, and store contracts:
  `67 passed in 2.32s`;
- broader affected contracts: `146 passed in 3.84s`;
- lowerer architecture contracts: `258 passed in 18.28s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because only six already-computed
two-counter result destinations changed.

## Unconditional very-late reconciliation characterization

The remaining unconsumed static-shape inventory contains six results with
different execution semantics. The next family is limited to the two
unconditional reconciliations: one immediately after very-late broadcast
repair and one after the final very-late dynamic-rank-one rewrite.

The characterization fixes each predecessor, successor, source order,
`model_ir` argument, `include_mutation_count=True`, and absence of consumers.
It explicitly requires both owner calls to remain unconditional. A strict
expected failure requires stable
`shape_reconciliation.primary.very_late_broadcast` and
`shape_reconciliation.primary.very_late_final` records. No production source
changed.

Validation completed sequentially under core-only `uv`:

- dedicated boundary contract: `1 passed, 1 xfailed in 0.14s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result-destination
migration.

## Unconditional very-late reconciliation implementation

The two unconditional results now record as:

- `shape_reconciliation.primary.very_late_broadcast`;
- `shape_reconciliation.primary.very_late_final`.

Both owner calls remain top-level, unconditional, and directly between the
same predecessors and successors. Only their unconsumed assignment targets
were replaced; unlike guarded families, there were no zero defaults to remove.

No guard, owner argument, reconciliation, graph scan, mutation, pass order,
public result, report, artifact, dependency, or TensorFlow boundary changed.
The bounded store now covers 47 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct boundary, terminal, very-late, and store contracts:
  `90 passed in 2.77s`;
- broader affected contracts: `171 passed in 4.32s`;
- lowerer architecture contracts: `258 passed in 18.11s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because the two reconciliation calls and
their ModelIR effects are unchanged.

## Late binary reconciliation characterization

The next family contains the two guarded late binary reconciliation results.
The first follows static-signature and indexed binary-adapter repair and also
guards on cleanup-only tensor pruning. The second is nested under the existing
late-layout enablement condition and runs only when the late binary recovery
summary reports a positive mutation.

Neither result has a default assignment or a consumer. The characterization
fixes both nested guard structures, source order, owner arguments,
`include_mutation_count=True`, pruning evidence, and absence of loads. A strict
expected failure requires stable
`shape_reconciliation.primary.late_binary_repair` and
`shape_reconciliation.primary.late_binary_layout_recovery` records. No
production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated nested-guard contract: `1 passed, 1 xfailed in 0.18s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result-destination
migration.

## Primary generic final reconciliation implementation

The three selected results now record as:

- `shape_reconciliation.primary.final_mixed_singleton_concat`;
- `shape_reconciliation.primary.final_placeholder_binary`;
- `shape_reconciliation.primary.final_se_fc_gather`.

All records remain inside their original mutation/pruning guards. The consumed
`_final_placeholder_matmul_static_shape_stats` assignment remains unchanged
and still feeds `final_placeholder_reconcile_stats`; only the nested,
unconsumed binary reconciliation result moved. The three selected zero defaults
and local targets were removed.

No guard, owner argument, reconciliation, adapter cleanup, topology checkpoint,
graph scan, mutation, successor, public result, report, artifact, dependency,
or TensorFlow boundary changed. The bounded store now covers 39 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct family, terminal, SE/FC/Gather, and store contracts:
  `79 passed in 2.57s`;
- broader affected contracts: `144 passed in 3.83s`;
- lowerer architecture contracts: `258 passed in 19.17s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because this is a result-destination-only
change with the consumed placeholder control-flow input explicitly preserved.

## Primary final SiNet reconciliation characterization

The six final SiNet-specific reconciliation boundaries form one contiguous
semantic chain: late residual, pre-add fanout, dual resize, shared post,
deep-skip tail, and concat-resize. Each repair has an unconsumed two-counter
zero default and invokes
`_reconcile_static_tensor_shapes(model_ir, include_mutation_count=True)` only
when its dedicated mutation counter is positive.

The characterization fixes all six repair targets, result targets, source
order, zero schemas, owner arguments, keyword arguments, and absence of result
consumers. A strict expected failure requires six stable
`shape_reconciliation.primary.final_sinet_*` records. Earlier late/static
shape boundaries are deliberately outside this family. No production source
changed.

Validation completed sequentially under core-only `uv`:

- dedicated ordered-chain contract: `1 passed, 1 xfailed in 0.16s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented bounded-store
migration.

## Late binary reconciliation implementation

The two characterized late binary reconciliation results now record under:

- `shape_reconciliation.primary.late_binary_repair`;
- `shape_reconciliation.primary.late_binary_layout_recovery`.

The first record remains guarded by static-signature repair, indexed binary
adapter repair, or cleanup-only tensor pruning. The second remains nested
inside both the existing late-layout enablement condition and the positive
late binary recovery-summary guard. Neither boundary had an all-zero default
to remove.

Only the destinations of the already-computed reconciliation counters changed.
Both `_reconcile_static_tensor_shapes` calls retain `model_ir` and
`include_mutation_count=True`; their guards, evaluation count, source order,
graph traversal, ModelIR mutations, successors, public results, reports,
artifacts, dependencies, and TensorFlow import boundaries are unchanged. The
bounded store now covers 49 phase IDs.

Validation completed sequentially under core-only `uv`:

- focused late-binary, terminal, and bounded-store contracts:
  `72 passed in 2.40s`;
- broader phase-result, owner, fallback, terminal, shape, and topology
  contracts: `187 passed in 4.82s`;
- lowerer architecture contracts: `258 passed in 18.40s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because this implementation changes only two
unconsumed result destinations. The remaining unconsumed static-shape
observations are the guarded shared-late boundary and the post-split fallback
boundary; they have different execution semantics and must be characterized
separately.

## Shared-late reconciliation characterization

The next result is the guarded shared-late static-shape reconciliation. Its
existing predicate combines nine mutation-result dictionaries with a tensor-
count decrease that covers cleanup-only pruning. The result has no default and
no consumer.

The dedicated contract fixes the exact evidence order, tensor-count boundary,
guard, `model_ir` argument, `include_mutation_count=True`, immediate late-
binary successor, and absence of loads. A strict expected failure requires a
stable `shape_reconciliation.primary.shared_late` record. Existing runtime
coverage independently forces each positive evidence dictionary and the
prune-only path. No production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization contract: `1 passed, 1 xfailed in 0.15s`;
- characterization plus runtime, terminal, and architecture boundary
  contracts: `4 passed, 1 xfailed in 0.72s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result-destination
migration.

## Shared-late reconciliation implementation

The guarded result now records under
`shape_reconciliation.primary.shared_late`. Its predicate still contains the
same nine mutation-result dictionaries in the same order plus the tensor-count
decrease for cleanup-only pruning. The record remains between the same
`shared_late_tensor_count` and `late_binary_repair_tensor_count` snapshots.

Only the destination of the already-computed reconciliation counters changed.
The `_reconcile_static_tensor_shapes(model_ir,
include_mutation_count=True)` owner, evaluation count, guard, graph traversal,
ModelIR mutations, successors, public results, reports, artifacts,
dependencies, and TensorFlow import boundaries are unchanged. No default
existed at this boundary. The bounded store now covers 50 phase IDs.

Validation completed sequentially under core-only `uv`:

- focused shared-late, late-binary, terminal, runtime, and bounded-store
  contracts: `75 passed in 2.31s`;
- broader phase-result, owner, fallback, terminal, shape, and topology
  contracts: `190 passed in 5.03s`;
- lowerer architecture contracts: `258 passed in 16.57s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because this is a result-destination-only
change at a runtime-characterized boundary. The post-split fallback result is
now the sole remaining unconsumed static-shape observation and must be
characterized separately before migration.

## Post-split fallback reconciliation characterization

The final unconsumed static-shape result follows the very-late unsupported
Split-to-Slice fallback. It has an unconditional all-zero two-counter default,
then invokes `_reconcile_static_tensor_shapes(model_ir,
include_mutation_count=True)` only when the fallback reports at least one
replacement. Neither the default nor the invoked result is consumed.

The dedicated contract fixes the Split owner and layout-state argument, exact
zero schema, positive replacement guard, reconciliation owner arguments,
absence of loads, and the immediate unbound-input safety-fallback successor. A
strict expected failure requires a stable
`shape_reconciliation.primary.post_split_fallback` record with invoked-phase-
only semantics. No production source changed.

Validation completed sequentially under core-only `uv`:

- characterization, existing orchestration boundary, and Split fallback unit
  contracts: `6 passed, 1 xfailed in 0.64s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result-destination
migration.

## Post-split fallback reconciliation implementation

The guarded result now records under
`shape_reconciliation.primary.post_split_fallback`. The unconsumed all-zero
default was removed, and the phase remains absent when the existing positive
replacement guard is false. This matches the invoked-phase-only semantics of
the other guarded reconciliation records.

Only the default and result destination changed. The Split-to-Slice owner,
`layout_state=session.layout_state`, replacement-count predicate,
`_reconcile_static_tensor_shapes(model_ir,
include_mutation_count=True)` call, evaluation count, ModelIR mutations,
unbound-input safety check, public results, reports, artifacts, dependencies,
and TensorFlow import boundaries are unchanged. The bounded store now covers
51 phase IDs.

Validation completed sequentially under core-only `uv`:

- focused post-split, very-late, Split fallback, terminal, and bounded-store
  contracts: `96 passed in 2.39s`;
- broader phase-result, owner, fallback, terminal, shape, and topology
  contracts: `196 passed in 5.25s`;
- lowerer architecture contracts: `258 passed in 16.47s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because this is a result-destination-only
change at an already-characterized boundary. An AST audit confirms that
`lower_onnx_to_ir` now has no unconsumed `*static_shape_stats` assignment. The
remaining `_final_placeholder_matmul_static_shape_stats` local is intentionally
retained because its counters are loaded by an existing downstream guard.

## Stale Squeeze/Reshape boundary assertion correction

The expanded core-cleanup characterization gate exposed one stale structural
assertion in `test_flatbuffer_direct_squeeze_reshape_identity_results.py`.
That test still expected a raw `_resolve_dynamic_reshape_shapes(model_ir)` call
immediately before the core Squeeze/Reshape cleanup, although the call had
already moved to the stable `shape_resolution.core.dynamic_reshape` phase
record in commit `dd58fd84`.

The assertion now fixes the exact existing session record. No production code,
owner call, graph mutation, phase order, public result, artifact, dependency,
or TensorFlow boundary changed. The expanded related suite moved from one
failure to `72 passed, 1 xfailed in 2.18s`; the sole xfail is the separately
characterized, intentionally unimplemented core-cleanup result migration.
Targeted Ruff, bytecode compilation, and whitespace validation pass.

## Core cleanup phase-result characterization

The next selected family is limited to the nine direct `Dict[str, int]`
results in the unconditional `core cleanup passes` stage:

- pseudo-LeakyReLU and YOLO decode cleanup;
- consecutive-Mul folding;
- terminal Dequantize and exact-grid Q/DQ cleanup;
- Conv affine and Conv activation folding;
- Squeeze/Reshape identity cleanup;
- indexed prune/reconcile cleanup.

Composite cluster `*_results`, guarded layout-pass results, and later terminal
families are intentionally excluded. The dedicated contract fixes all nine
targets, owner expressions and keyword arguments, source order, progress
boundaries, absence of consumers, and the existing
`shape_resolution.core.dynamic_reshape` record between the seventh and eighth
owners. A strict expected failure requires nine stable `cleanup.core.*` phase
records. No production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization: `1 passed, 1 xfailed in 0.15s`;
- characterization plus dynamic-Reshape, Squeeze/Reshape, indexed
  prune/reconcile, and terminal orchestration contracts:
  `72 passed, 1 xfailed in 2.18s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented nine-result
destination migration.

## Core cleanup phase-result implementation

The nine characterized results now record under stable `cleanup.core.*` phase
IDs. All records remain top-level and unconditional inside the existing
`core cleanup passes` progress stage. The established
`shape_resolution.core.dynamic_reshape` record remains between Conv activation
and Squeeze/Reshape identity cleanup.

Only the nine unconsumed assignment destinations changed. Every owner call,
argument, keyword, evaluation count, graph traversal, ModelIR mutation, source
order, progress boundary, public result, report, artifact, dependency, and
TensorFlow import boundary is unchanged. No all-zero default existed in this
family. The bounded store now covers 60 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct core-cleanup, phase-store, dynamic-Reshape, Squeeze/Reshape, indexed
  prune/reconcile, terminal, and architecture-boundary contracts:
  `76 passed in 2.65s`;
- synthetic core runtime contracts, including all nine stored mapping schemas:
  `55 passed in 1.02s`;
- broader phase-result, owner, cleanup, fallback, terminal, shape, and topology
  contracts: `257 passed in 6.27s`;
- lowerer architecture contracts: `258 passed in 16.75s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because owner execution and ModelIR
effects are unchanged and the existing synthetic runtime suite exercises the
new storage boundary.

## Terminal cleanup phase-result characterization

The next family contains only the four unconditional mapping results at the
start of the `terminal cleanup passes` stage:

- terminal Transpose-before-Dequantize sanitization;
- exact-grid terminal Q/DQ cleanup;
- Conv affine folding;
- Conv activation folding.

These are the same owner schemas already exercised by the core-cleanup runtime
gate. The characterization fixes their targets, owner expressions and keyword
arguments, source order, progress predecessor, pre-ArgMax successor, and
absence of consumers. A strict expected failure requires four stable
`cleanup.terminal.*` records. Later terminal layout results remain outside this
family. No production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization plus terminal orchestration and indexed-owner
  architecture boundaries: `5 passed, 1 xfailed in 0.68s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented four-result
destination migration.

## Terminal cleanup phase-result implementation

The four characterized results now record under stable `cleanup.terminal.*`
phase IDs. They remain top-level and unconditional immediately after the
`terminal cleanup passes` progress marker and immediately before pre-ArgMax
cleanup.

Only the four unconsumed assignment destinations changed. All owner calls,
arguments, keywords, evaluation counts, graph traversals, ModelIR mutations,
source order, progress boundary, successor, public results, reports, artifacts,
dependencies, and TensorFlow import boundaries are unchanged. No default
existed in this family. The bounded store now covers 64 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct terminal-cleanup, phase-store, terminal orchestration, and indexed-
  owner architecture-boundary contracts: `68 passed in 4.13s`;
- synthetic core runtime contracts, which execute the terminal cleanup stage:
  `55 passed in 1.01s`;
- broader phase-result, owner, cleanup, fallback, terminal, shape, and topology
  contracts: `259 passed in 6.15s`;
- lowerer architecture contracts: `258 passed in 16.54s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the synthetic runtime suite executes all four
stored owners.

## Layout pass-set 2 cleanup characterization

The next family is the adjacent Squeeze/Reshape identity and indexed
prune/reconcile cleanup pair at the end of layout pass-set 2. Both return the
already-validated bounded integer mapping schemas, run only inside the existing
`optimize_layout_transpose_chains` guard, have no defaults, and have no
consumers.

The characterization fixes the common guard, owner expressions and keywords,
source order, preceding two-iteration normalization convergence loop,
following progress advance, and absence of loads. A strict expected failure
requires stable `cleanup.layout_pass_set_2.*` records with invoked-phase-only
semantics. No production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization plus Squeeze/Reshape and indexed prune/reconcile
  contracts: `6 passed, 1 xfailed in 0.77s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented two-result
destination migration.

## Layout pass-set 2 cleanup implementation

The pair now records under
`cleanup.layout_pass_set_2.squeeze_reshape_identity` and
`cleanup.layout_pass_set_2.prune_reconcile`. Both records remain inside the
existing `optimize_layout_transpose_chains` guard, immediately after the same
normalization convergence loop and before the same progress advance.

Only the two unconsumed assignment destinations changed. Owner calls,
arguments, keywords, evaluation counts, graph traversals, ModelIR mutations,
guard, source order, public results, reports, artifacts, dependencies, and
TensorFlow import boundaries are unchanged. No defaults existed. The bounded
store now covers 66 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct pair, Squeeze/Reshape, indexed prune/reconcile, and phase-store
  contracts: `9 passed in 0.85s`;
- synthetic core runtime contracts: `55 passed in 0.99s`;
- broader phase-result, owner, cleanup, fallback, terminal, shape, and topology
  contracts: `261 passed in 6.27s`;
- lowerer architecture contracts: `258 passed in 16.57s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because only observation destinations
changed and the synthetic runtime suite exercises the guarded layout path.

## Layout pass-set 1 cleanup characterization

The next family is limited to the adjacent InstanceNorm pre/post and
Squeeze/Reshape identity mapping results near the end of layout pass-set 1.
Both run under the existing `optimize_layout_transpose_chains` guard, have no
defaults or consumers, and sit between two composite attention-cluster results
that remain explicitly outside this migration.

The characterization fixes the common guard, exact owner expressions and
keywords, adjacency, composite predecessor and successor targets, and absence
of loads. A strict expected failure requires stable
`cleanup.layout_pass_set_1.*` records with invoked-phase-only semantics. No
production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization plus InstanceNorm, Squeeze/Reshape, and attention-
  prefix architecture boundaries: `6 passed, 1 xfailed in 0.80s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented two-result
destination migration.

## Layout pass-set 1 cleanup implementation

The adjacent pair now records under
`cleanup.layout_pass_set_1.instancenorm_prepost` and
`cleanup.layout_pass_set_1.squeeze_reshape_identity`. Both remain inside the
existing layout-optimization guard and between the same composite attention
prefix/suffix result assignments.

Only the two unconsumed mapping destinations changed. Owner calls, arguments,
keywords, evaluation counts, graph traversals, ModelIR mutations, guard,
adjacency, composite boundaries, public results, reports, artifacts,
dependencies, and TensorFlow import boundaries are unchanged. No defaults
existed. The bounded store now covers 68 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct pair, InstanceNorm, Squeeze/Reshape, phase-store, and attention-prefix
  architecture-boundary contracts: `9 passed in 2.63s`;
- synthetic core runtime contracts: `55 passed in 1.04s`;
- broader phase-result, owner, cleanup, fallback, terminal, shape, and topology
  contracts: `265 passed in 6.34s`;
- lowerer architecture contracts: `258 passed in 16.53s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the runtime suite executes the guarded layout path.

## Layout pass-set 1 quantized cleanup characterization

The next family contains the three consecutive mapping-only quantized cleanup
results in layout pass-set 1:

- quantized PReLU cleanup;
- Dequantize→TransposeConv→Quantize folding;
- quantized Reshape cleanup.

All three run under `optimize_layout_transpose_chains`, return bounded integer
mappings with existing explicit schema tests, have no defaults or consumers,
and are bounded by composite attention/QDQ results that remain outside this
migration.

The characterization fixes the common guard, owner expressions and keywords,
three-statement adjacency, composite predecessor/successor targets, and absence
of loads. A strict expected failure requires stable
`cleanup.layout_pass_set_1.*` quantized phase records. No production source
changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization plus quantized PReLU, Reshape, and TransposeConv
  owner/schema contracts: `7 passed, 1 xfailed in 0.83s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented three-result
destination migration.

## Layout pass-set 1 quantized cleanup implementation

The trio now records under stable
`cleanup.layout_pass_set_1.quantized_prelu`,
`cleanup.layout_pass_set_1.dequant_transposeconv_quantize`, and
`cleanup.layout_pass_set_1.quantized_reshape` phase IDs. All remain inside the
same layout guard, consecutive, and between the same composite attention/QDQ
result assignments.

Only the three unconsumed mapping destinations changed. Owner calls,
arguments, keywords, evaluation counts, graph traversals, ModelIR mutations,
guard, source order, composite boundaries, public results, reports, artifacts,
dependencies, and TensorFlow import boundaries are unchanged. No defaults
existed. The bounded store now covers 71 phase IDs.

The first full architecture run exposed two stale adjacent-boundary assertions.
Both treated the new outer `session.record_phase_result` call as a direct
`ast.Name` owner. They were updated to unwrap and verify the exact nested owner
while retaining the composite boundary assertions. The focused correction
passes `2 passed in 2.20s`; no production fix was required.

Validation completed sequentially under core-only `uv`:

- direct trio, quantized owner/schema, and phase-store contracts:
  `10 passed in 0.95s`;
- synthetic core runtime contracts: `55 passed in 1.01s`;
- broader phase-result, owner, cleanup, fallback, terminal, shape, and topology
  contracts: `273 passed in 6.62s`;
- lowerer architecture contracts after the stale-assertion correction:
  `258 passed in 16.61s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the runtime suite exercises the guarded layout
path.

## Layout pass-set 2 quantized cleanup characterization

The remaining direct Dequantize→TransposeConv→Quantize mapping result belongs
to layout pass-set 2. It runs under `optimize_layout_transpose_chains`, has no
default or consumer, and is bounded by composite attention-gate/QDQ and
quantized-activation recovery results.

The characterization fixes the guard, exact owner expression and layout-state
argument, composite predecessor/successor targets, sole Store occurrence, and
absence of loads. A strict expected failure requires
`cleanup.layout_pass_set_2.dequant_transposeconv_quantize`. No production source
changed.

Validation completed sequentially under core-only `uv`:

- dedicated characterization, multi-occurrence owner, and adjacent architecture
  boundaries: `5 passed, 1 xfailed in 0.76s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result-destination
migration.

## Layout pass-set 2 quantized cleanup implementation

The remaining direct result now records under
`cleanup.layout_pass_set_2.dequant_transposeconv_quantize`. It remains inside
the existing layout guard and between the same composite attention-gate/QDQ and
quantized-activation recovery results.

Only the unconsumed mapping destination changed. The owner call, argument,
layout-state keyword, evaluation count, graph traversal, ModelIR mutation,
guard, composite boundaries, public results, reports, artifacts, dependencies,
and TensorFlow import boundaries are unchanged. No default existed. The bounded
store now covers 72 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct result, multi-occurrence owner, phase-store, and adjacent architecture
  contracts: `8 passed in 0.82s`;
- synthetic core runtime contracts: `55 passed in 1.03s`;
- broader phase-result, owner, cleanup, fallback, terminal, shape, and topology
  contracts: `275 passed in 6.94s`;
- lowerer architecture contracts: `258 passed in 17.39s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No root-model corpus conversion was run because this is an observation-
destination-only change and the runtime suite executes the guarded layout path.

## Primary final cleanup reconciliation implementation

The final PReLU and consecutive-Reshape reconciliation results now record as:

- `shape_reconciliation.primary.final_prelu`;
- `shape_reconciliation.primary.final_consecutive_reshape`.

Both records remain inside their original guards. The PReLU condition still
accounts for either a rewrite or cleanup-only tensor pruning, and the
consecutive-Reshape condition still sums all three declared mutation counters.
The unconsumed zero defaults and local targets were removed.

No owner argument, reconciliation, cleanup, graph scan, mutation, guard,
successor, public result, report, artifact, dependency, or TensorFlow boundary
changed. The bounded store now covers 36 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct pair, terminal, and phase-store contracts: `67 passed in 1.96s`;
- broader affected contracts: `142 passed in 3.62s`;
- lowerer architecture contracts: `258 passed in 19.09s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because only the destination of existing
two-counter results changed.

## Primary generic final reconciliation characterization

The next primary family contains mixed-singleton Concat,
placeholder/binary-adapter, and SE/FC/Gather reconciliation. Each selected
result is an unconsumed two-counter zero default followed by a guarded
`_reconcile_static_tensor_shapes(model_ir, include_mutation_count=True)` call.

The placeholder boundary also has a separate
`_final_placeholder_matmul_static_shape_stats` result that is consumed to
decide whether binary adapters require a second reconciliation. That consumed
result is explicitly excluded from migration and protected by the new
contract.

The characterization fixes the three selected targets, source order, zero
schema, owner arguments, keyword arguments, absence of consumers, and continued
load of the placeholder-MatMul result. A strict expected failure requires
three stable `shape_reconciliation.primary.final_*` records without changing
the consumed value or any guard. No production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated family contract: `1 passed, 1 xfailed in 0.17s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented bounded-store
migration.

## Fallback static-shape phase-result implementation

The seven characterized fallback reconciliation results now record directly
to the bounded session store under:

- `shape_reconciliation.fallback.broadcast`;
- `shape_reconciliation.fallback.se_fc_gather`;
- `shape_reconciliation.fallback.placeholder_matmul`;
- `shape_reconciliation.fallback.conv_input`;
- `shape_reconciliation.fallback.mixed_concat`;
- `shape_reconciliation.fallback.concat_axis`;
- `shape_reconciliation.fallback.binary_layout`.

Their unconsumed zero-default dictionaries and lowerer-local targets were
removed. Existing mutation-positive guards still decide whether the owner is
invoked, so skipped phases are absent from the snapshot. Every owner call
still receives `fallback_ir` and `include_mutation_count=True` at the same
location. The broadcast boundary still performs topology/layout refresh
immediately after reconciliation, and the remaining successors are unchanged.

No graph scan, mutation, repair, guard, layout refresh, topology sort,
dependency, public result, report, metadata field, or artifact changed. The
bounded store now covers 31 stable phase IDs.

Validation completed sequentially under core-only `uv`:

- new family plus safety-fallback contracts: `20 passed in 0.93s`;
- SE/FC/Gather, topology/layout predecessor, and phase-store contracts:
  `18 passed in 0.82s`;
- broader affected owner and orchestration contracts:
  `138 passed in 3.45s`;
- lowerer architecture contracts: `258 passed in 17.69s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because this implementation only changes the
destination of already-computed two-counter dictionaries.

## Primary final layout-refresh reconciliation characterization

The primary-path inventory found 20 remaining unconsumed static-shape-only
results. They were split by semantic boundary rather than treated as one large
mechanical migration. The next selected family contains only the three
reconciliations immediately preceding the final ConvInteger, InstanceNorm, and
broadcast topology/layout refreshes.

All three currently share an all-zero default, a single mutation-positive
guard, the complete two-counter
`_reconcile_static_tensor_shapes(model_ir, include_mutation_count=True)` owner,
and a same-guard `run_topology_layout_refresh(model_ir)` successor. Their
results are never loaded.

The characterization fixes the target and source order, zero schema, owner
arguments, keyword arguments, refresh phase order, and absence of consumers.
A strict expected failure requires three stable
`shape_reconciliation.primary.final_*` records before their existing refresh
records. No production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated family contract: `1 passed, 1 xfailed in 0.16s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented session-store
migration.

## Primary final layout-refresh reconciliation implementation

The three characterized reconciliations now record under:

- `shape_reconciliation.primary.final_convinteger`;
- `shape_reconciliation.primary.final_instancenorm`;
- `shape_reconciliation.primary.final_broadcast`.

Each record remains inside its original mutation-positive guard and directly
precedes the matching `topology_layout.primary.final_*` refresh record. The
three unconsumed zero-default dictionaries and local targets were removed.
Owner arguments, complete mutation accounting, refresh calls, repair order,
and all subsequent phases are unchanged.

No graph scan, reconciliation, layout inference, topology sort, mutation,
public result, report, artifact, dependency, or TensorFlow boundary changed.
The bounded store now covers 34 phase IDs.

Validation completed sequentially under core-only `uv`:

- direct family, terminal, refresh, and phase-store contracts:
  `71 passed in 2.50s`;
- broader affected contracts: `140 passed in 3.57s`;
- lowerer architecture contracts: `258 passed in 17.03s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

No real-model conversion was run because this is a result-destination-only
change at already-characterized owner boundaries.

## Primary final cleanup reconciliation characterization

The next generic primary family is limited to the final PReLU and
consecutive-Reshape cleanup boundaries. Both use an unconsumed two-counter
zero default followed by a guarded
`_reconcile_static_tensor_shapes(model_ir, include_mutation_count=True)` call.
The PReLU guard includes both the rewrite counter and cleanup-only tensor
pruning; the consecutive-Reshape guard includes all three mutation counters.

The characterization fixes both targets, their source order, zero schema,
owner arguments, keyword arguments, and absence of consumers. A strict
expected failure requires stable `shape_reconciliation.primary.final_prelu`
and `shape_reconciliation.primary.final_consecutive_reshape` records while
preserving the existing guards. No production source changed.

Validation completed sequentially under core-only `uv`:

- dedicated pair contract: `1 passed, 1 xfailed in 0.15s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented bounded-store
migration.

## Pre-terminal pre-add prune-evidence characterization

The next late-pipeline extraction candidate is the single pre-add NHWC-chain
cleanup immediately after the first terminal-affine recovery summary and
before the channel Slice/Pad/Mul cluster. Its current lowerer boundary takes a
tensor-count snapshot, runs
`_optimize_transpose_pre_add_nhwc_chains(...)`, and extends the returned
mapping with an exact `pruned_unused_tensors` delta. The mapping is retained
only as `_pre_terminal_pre_add_stats` and is not used for control flow.

`tests/test_flatbuffer_direct_pre_terminal_pre_add_orchestration.py` fixes the
current source order, exact tensor-count expression, owner arguments,
prune-delta expression, neighboring boundaries, and absence of any additional
count use. A strict expected failure describes the smallest safe follow-up: a
pass-module owner must capture the tensor count, call the same existing pass
once with the same ModelIR/LayoutState objects, return the same mapping, and
replace only the local evidence construction. No production source, graph
mutation, pass order, guard, store entry, public result, artifact, dependency,
or TensorFlow boundary changed in this characterization checkpoint.

Validation completed sequentially under core-only `uv`:

- dedicated boundary contract: `1 passed, 1 xfailed in 0.13s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally absent prune-aware owner.

## Pre-terminal pre-add prune-evidence implementation

`run_pre_terminal_pre_add_cleanup(context)` now owns the characterized
tensor-count snapshot, the single pre-add NHWC-chain cleanup call, and the
non-negative `pruned_unused_tensors` delta. It imports the existing pass owner
directly, forwards the exact conversion-local ModelIR and LayoutState objects,
and returns the original pass mapping extended with the same prune evidence.

The lowerer retains `_pre_terminal_pre_add_stats` at its original location but
now assigns it from the focused owner. The lowerer-local tensor-count variable
and duplicate inline mapping construction are gone. The preceding first
terminal-affine summary and following channel Slice/Pad/Mul cluster remain
adjacent. The existing lowerer compatibility wrapper remains defined for its
other callers, and the direct pass's total production call count is unchanged
through the declared orchestration pass-ID sequence.

No pass execution, scan, graph mutation, prune behavior, result key, source
order, layout identity, diagnostics, public API, artifact, dependency, or
TensorFlow boundary changed. The phase-result store remains exactly 128/128;
this result stays outside it as before.

Final sequential validation under core-only `uv`:

- focused stable/pruned owner and boundary contracts: `4 passed in 0.58s`;
- affected pre-add, channel Slice/Pad/Mul, StridedSlice/Pad/Concat,
  terminal-affine, InstanceNorm, core, and store contracts:
  `186 passed in 2.26s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.98s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.37s`;
- full lowerer architecture contracts: `258 passed in 19.63s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 10.20s`;
- targeted Ruff, bytecode compilation, 128/128 capacity audit, and whitespace
  checks: passed.

No real-model conversion was repeated because this is a one-call orchestration
extraction whose focused runtime matrix proves both stable and prune-only
paths, while the broader structural gates preserve the exact call count and
neighboring boundaries.

## Channel Slice/Pad/Mul direct-summary characterization

The next adjacent evidence pair stores the raw ordered results from the
channel Slice/Pad/Mul cluster and immediately normalizes them into
`_pre_terminal_channel_slice_pad_mul_stats`. The raw tuple is consumed only by
that summary call, while the normalized mapping is not read by later control
flow. The lowerer-local raw wrapper must remain because terminal Slice/Concat
recovery still receives it as a callback.

`tests/test_flatbuffer_direct_channel_slice_pad_mul_summary_orchestration.py`
fixes the two-statement boundary, exact wrapper and summarizer calls, result
use counts, predecessor, successor, and the retained raw-wrapper dispatch. Its
strict expected failure describes one pass-module summary owner used only at
the direct site. No production source, pass, graph, result, callback, store,
public API, artifact, dependency, or TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.14s`; targeted Ruff, bytecode compilation, and
whitespace checks passed. The sole expected failure is the intentionally
absent direct-summary owner.

## Channel Slice/Pad/Mul direct-summary implementation

`run_channel_slice_pad_mul_summary(context)` now composes the existing raw
ordered cluster owner with the existing strict four-counter normalizer. The
lowerer direct site calls this owner with the same conversion-local context and
retains `_pre_terminal_channel_slice_pad_mul_stats` at the same source
boundary. The consumed `channel_slice_pad_mul_results` local and duplicate
two-statement composition are removed.

The nested `_run_channel_slice_pad_mul_layout_pass_cluster` wrapper remains
defined and still dispatches to the raw owner. Terminal Slice/Concat recovery
continues to receive that wrapper as its callback, so the raw cluster executes
the same total number of times and still shares one pass-state scope per
invocation. The pre-terminal pre-add predecessor and affine post-Add successor
remain adjacent.

No pass call, pass order, state-scope lifetime, result schema, graph scan,
mutation, layout or diagnostics identity, callback, public API, artifact,
dependency, or TensorFlow boundary changed. The normalized mapping remains
outside the already-full 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `3 passed in 0.57s`;
- affected channel Slice/Pad/Mul, pre-add, StridedSlice/Pad/Concat, terminal
  recovery, core, and store contracts: `195 passed in 2.42s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.86s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.18s`;
- full lowerer architecture contracts: `258 passed in 18.46s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 10.31s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because the focused runtime test
proves the raw context and normalized schema, while the affected recovery and
architecture gates prove the unchanged callback and production invocation
counts.

## Pre-terminal affine-tail composite characterization

The next two adjacent mappings cover affine post-Add cleanup followed by the
strict StridedSlice/Pad/Concat affine bridge. Both results are unconsumed, but
their argument policies differ: the first receives ModelIR plus LayoutState,
while the second receives ModelIR only. They run after the normalized channel
Slice/Pad/Mul summary and immediately before the second terminal-affine
recovery summary.

`tests/test_flatbuffer_direct_pre_terminal_affine_tail_orchestration.py` fixes
the two targets, exact call expressions, argument policies, order, neighboring
boundaries, and absence of result consumers. Its strict expected failure
requires one direct pass-module owner returning both mappings in order. No
production source, pass, graph, result, store, public API, artifact,
dependency, or TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.14s`; targeted Ruff, bytecode compilation, and
whitespace checks passed. The sole expected failure is the intentionally
absent ordered owner.

## Pre-terminal affine-tail composite implementation

`run_pre_terminal_affine_tail_cleanup(context)` now owns the two characterized
repairs. It calls affine post-Add cleanup with the shared ModelIR/LayoutState,
then calls strict StridedSlice/Pad/Concat cleanup with ModelIR only, returning
both original mappings as an ordered tuple.

The lowerer replaces the two unconsumed direct result targets with
`_pre_terminal_affine_tail_results` at the same boundary. The normalized
channel Slice/Pad/Mul summary remains the predecessor and the second
terminal-affine recovery summary remains the successor. Both lowerer
compatibility wrappers, all other direct sites, and total production call
counts remain unchanged through the declared orchestration pass IDs.

No pass execution, order, graph scan, mutation, result mapping, layout or
diagnostics identity, public API, artifact, dependency, or TensorFlow boundary
changed. The ordered tuple remains outside the already-full 128/128
phase-result store.

Final sequential validation under core-only `uv`:

- focused owner, order, argument, and boundary contracts: `3 passed in 0.56s`;
- affected affine-tail, channel Slice/Pad/Mul, StridedSlice/Pad/Concat,
  terminal recovery, very-late, absolute-final, core, and store contracts:
  `237 passed in 3.14s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.88s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.23s`;
- full lowerer architecture contracts: `258 passed in 18.98s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.81s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because the focused runtime
contract proves exact order, argument identity, and tuple preservation, while
the affected and architecture gates cover the other production call sites and
total invocation counts.

## Late SPP/Concat/Unary direct-summary characterization

The next late evidence pair stores the raw ordered SPP and Concat/Unary/Conv
results and immediately normalizes them into `_late_spp_stats`. The raw tuple
is consumed only by that summary call, and the normalized mapping is not used
for control flow. The existing raw wrapper remains a separate owner boundary.

`tests/test_flatbuffer_direct_late_spp_summary_orchestration.py` fixes the
two-statement representation, wrapper and summarizer calls, result use counts,
predecessor, successor, and retained wrapper dispatch. Its strict expected
failure requires one pass-module direct-summary owner. No production source,
pass, graph, result, store, public API, artifact, dependency, or TensorFlow
boundary changed.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.14s`; targeted Ruff, bytecode compilation, and
whitespace checks passed. The sole expected failure is the intentionally
absent direct-summary owner.

## Late SPP/Concat/Unary direct-summary implementation

`run_late_spp_concat_unary_conv_summary(context)` now composes the existing
raw ordered owner with its existing strict two-counter normalizer. The lowerer
direct site retains `_late_spp_stats` at the same boundary and passes the same
conversion-local context, while the consumed `late_spp_results` local and
duplicate two-statement composition are removed.

The nested `_run_late_spp_concat_unary_conv_pass_pair` wrapper remains defined
and still dispatches to the raw owner. The terminal Slice/Pad/Concat
predecessor, pre-QKV shape-extract successor, shared pass-state scope, and raw
result schema remain unchanged.

No pass execution, order, state-scope lifetime, graph scan, mutation, layout
or diagnostics identity, public API, artifact, dependency, or TensorFlow
boundary changed. The normalized summary remains outside the already-full
128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `3 passed in 0.60s`;
- affected late SPP, shape-extract, StridedSlice/Pad/Concat, terminal recovery,
  core, and store contracts: `187 passed in 1.97s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.91s`;
- synthetic core runtime contracts: `55 passed in 0.98s`;
- result contracts: `196 passed in 9.42s`;
- full lowerer architecture contracts: `258 passed in 18.89s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.86s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact context and normalization, while affected and architecture gates
preserve the wrapper, adjacent boundaries, shared scope, and total pass
ownership.

## Late QKV prune-aware summary characterization

The next evidence triple captures tensor count, invokes the QKV owner with the
runtime layout-Transpose flag and `include_prefix=False`, then builds a stable
prune-aware summary. The count and raw tuple are consumed only by that summary;
the normalized mapping is not used for control flow. Two default-policy raw
wrapper uses elsewhere must remain unchanged.

`tests/test_flatbuffer_direct_late_qkv_summary_orchestration.py` fixes the
three statements, exact flags, prune expression, result use counts,
predecessor, successor, and retained raw wrapper. Its strict expected failure
requires a generic pass-module prune-aware summary owner. No production source,
pass, graph, result, store, public API, artifact, dependency, or TensorFlow
boundary changed.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.17s`; targeted Ruff, bytecode compilation, and
whitespace checks passed. The sole expected failure is the intentionally
absent summary owner.

## Late QKV prune-aware summary implementation

`run_qkv_attention_summary(context, *, include_layout_transpose,
include_prefix)` now owns the characterized tensor snapshot, raw QKV
invocation, and strict prune-aware normalization. The lowerer retains
`_late_qkv_stats` at the same boundary and forwards the same runtime
layout-Transpose flag with prefix cleanup disabled, while the consumed
`late_qkv_tensor_count` and `late_qkv_results` locals are removed.

The nested `_run_qkv_attention_layout_pass_cluster` compatibility wrapper and
its two default-policy production uses remain unchanged. Pass selection,
execution order, shared context identity, result schema, graph pruning,
neighboring shape-extract and terminal bridge boundaries, public behavior,
artifacts, dependencies, and TensorFlow isolation are unchanged. The summary
remains outside the already-full 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `5 passed in 0.59s`;
- affected QKV, neighboring owner, core, store, and architecture contracts:
  `405 passed in 20.77s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.90s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.37s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.69s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact flag/context forwarding, stable and prune-only behavior, and
schema preservation, while the affected and architecture gates preserve all
production boundaries and total pass ownership.

## Terminal HardSwish/SE prune-aware summary characterization

The next late evidence pair snapshots tensor count and then extends the raw
HardSwish/SE layout result with `pruned_unused_tensors`. The raw owner already
performs its required unused-tensor pruning; the lowerer snapshot observes that
mutation without controlling later execution. The normalized result is not
loaded after assignment.

`tests/test_flatbuffer_direct_terminal_hardswish_se_summary_orchestration.py`
fixes the exact two-statement representation, one-key raw mapping plus prune
delta, terminal QKV-bridge predecessor, late hard-activation successor, and
retained raw lowerer wrapper. Its strict expected failure requires one generic
pass-module prune-aware summary owner. No production source, graph mutation,
pass order, store entry, public API, artifact, dependency, or TensorFlow
boundary changed.

Sequential characterization under core-only `uv` completed with
`76 passed, 1 xfailed in 1.22s` across the dedicated contract and related
HardSwish/SE, late hard-activation, indexed bridge, and phase-store contracts.
The sole expected failure is the intentionally absent summary owner.

## Terminal HardSwish/SE prune-aware summary implementation

`run_hardswish_se_layout_summary(model_ir)` now owns the characterized tensor
snapshot, raw HardSwish/SE layout invocation, and non-negative prune delta. The
late lowerer site retains `_terminal_hardswish_se_stats` at the same boundary,
while the lowerer-local `terminal_hardswish_se_tensor_count` and inline mapping
extension are removed.

The existing lowerer compatibility wrapper remains the dispatch boundary for
the earlier phase-store call. The raw owner still executes once at each of the
two production policies: indirectly through the wrapper at the earlier site
and through the new summary owner at the late site. Its internal pruning,
one-key raw schema, graph mutations, terminal QKV-bridge predecessor, late
hard-activation successor, public behavior, artifacts, dependencies, and
TensorFlow isolation are unchanged. The summary remains outside the already-
full 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `4 passed in 0.57s`;
- affected owner, boundary, store, and architecture contracts:
  `337 passed in 19.54s`;
- related HardSwish/SE, late hard-activation, indexed bridge, and recovery
  contracts: `87 passed in 1.42s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.93s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.13s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.73s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because the focused runtime tests
prove stable and prune-only summary behavior, while affected owner and
architecture gates preserve both production policies and their boundaries.

## Late hard-activation prune-aware summary characterization

The immediately following evidence triple snapshots tensor count, invokes the
late hard-activation/layout raw owner with the runtime layout-Transpose flag,
and normalizes the ordered results through the existing strict summarizer. The
count and raw tuple are consumed only by that summary; the normalized mapping
does not control later work.

`tests/test_flatbuffer_direct_late_hard_activation_summary_orchestration.py`
fixes the exact three statements, flag forwarding, prune expression, retained
raw wrapper, terminal HardSwish/SE predecessor, absolute-final pre-ConCat
successor, and absence of a summary consumer. Its strict expected failure
requires one pass-module prune-aware summary owner. No production source, pass
selection, graph mutation, store entry, public API, artifact, dependency, or
TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`22 passed, 1 xfailed in 1.15s` across the dedicated contract and related late
hard-activation, HardSwish/SE, pre-ConCat result, and phase-store contracts.
The sole expected failure is the intentionally absent summary owner.

## Late hard-activation prune-aware summary implementation

`run_late_hard_activation_layout_summary(context, *,
include_layout_transpose)` now owns the characterized tensor snapshot, raw
ordered invocation, and strict prune-aware normalization. The lowerer retains
`_late_hard_activation_stats` at the same boundary and passes the same shared
context and runtime layout-Transpose flag, while the consumed
`late_hard_activation_tensor_count` and `late_hard_activation_results` locals
are removed.

The nested `_run_late_hard_activation_layout_pass_pair` wrapper remains
defined and still dispatches to the raw owner. Active pass selection, shared
pass-state scope, hard-activation options, result schema, graph pruning,
terminal HardSwish/SE predecessor, absolute-final pre-ConCat successor, public
behavior, artifacts, dependencies, and TensorFlow isolation are unchanged.
The normalized summary remains outside the full 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `5 passed in 0.56s`;
- affected owner, boundary, store, and architecture contracts:
  `294 passed in 19.48s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.92s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.40s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.80s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact context/flag forwarding plus stable and prune-only behavior, while
the affected and architecture gates preserve raw ownership and both neighboring
boundaries.

## Late layout-cluster prune-aware summary characterization

The next evidence triple snapshots tensor count, invokes the late
layout/Mean/SPP/Gather/constant-fold/Cast ordered owner with the runtime
layout-Transpose flag, and applies its existing strict prune-aware normalizer.
The count and raw tuple are consumed only by the normalized mapping, which does
not control subsequent work.

`tests/test_flatbuffer_direct_late_layout_cluster_summary_orchestration.py`
fixes the exact three statements, flag and prune expression, retained raw
wrapper, pre-cluster shape-extract predecessor, terminal Expand/Squeeze
successor, and absence of a summary consumer. Its strict expected failure
requires one pass-module prune-aware summary owner. No production source, pass
selection, graph mutation, store entry, public API, artifact, dependency, or
TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`21 passed, 1 xfailed in 0.98s` across the dedicated contract and related late
layout-cluster, shape-extract result, and phase-store contracts. The sole
expected failure is the intentionally absent summary owner.

## Late layout-cluster prune-aware summary implementation

`run_late_layout_mean_spp_gather_constant_cast_summary(context, *,
include_layout_transpose)` now owns the characterized tensor snapshot, raw
ordered invocation, and strict prune-aware normalization. The lowerer retains
`_late_layout_cluster_stats` at the same boundary and passes the same shared
context and runtime layout-Transpose flag, while the consumed
`late_layout_cluster_tensor_count` and `late_layout_cluster_results` locals are
removed.

The nested raw wrapper remains defined and still dispatches to the ordered
owner. Active pass selection, shared pass-state scope, constant-fold/Cast child
builder, result schema, pruning, pre-cluster shape-extract predecessor,
terminal Expand/Squeeze successor, public behavior, artifacts, dependencies,
and TensorFlow isolation are unchanged. The normalized summary remains outside
the full 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `5 passed in 0.55s`;
- affected owner, boundary, store, and architecture contracts:
  `283 passed in 20.14s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.33s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.57s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact context/flag forwarding plus stable and prune-only behavior, while
the affected and architecture gates preserve child orchestration, raw
ownership, and both neighboring boundaries.

## Very-late normalization prune-aware summary characterization

The next evidence triple snapshots tensor count, invokes the very-late
Gather/constant-fold/Cast/normalization ordered owner, and applies its existing
strict four-result prune-aware normalizer. The count and raw tuple are consumed
only by the normalized mapping, which does not control subsequent work.

`tests/test_flatbuffer_direct_very_late_normalization_summary_orchestration.py`
fixes the exact three statements, prune expression, retained raw wrapper,
very-late affine post-Add predecessor, dynamic-Reshape successor, and absence
of a summary consumer. Its strict expected failure requires one pass-module
prune-aware summary owner. No production source, pass selection, graph
mutation, store entry, public API, artifact, dependency, or TensorFlow boundary
changed.

Sequential characterization under core-only `uv` completed with
`40 passed, 1 xfailed in 1.34s` across the dedicated contract and related
very-late normalization, absolute-final normalization/attention, late input
repair result, and phase-store contracts. The sole expected failure is the
intentionally absent summary owner.

## Very-late normalization prune-aware summary implementation

`run_very_late_gather_constant_normalization_summary(context)` now owns the
characterized tensor snapshot, four-result ordered invocation, and strict
prune-aware normalization. The lowerer retains `_very_late_normalization_stats`
at the same boundary and passes the same shared context, while the consumed
`very_late_normalization_tensor_count` and `very_late_normalization_results`
locals are removed.

The nested raw wrapper remains defined and still dispatches to the ordered
owner. Shared pass-state scope, constant-fold/Cast child builder, normalization
Pad policy, four-result schema, pruning, affine post-Add predecessor,
dynamic-Reshape successor, subsequent very-late repairs, public behavior,
artifacts, dependencies, and TensorFlow isolation are unchanged. The summary
remains outside the full 128/128 phase-result store.

Final sequential validation under core-only `uv`:

- focused summary-owner contracts: `4 passed in 0.60s`;
- affected owner, boundary, store, and architecture contracts:
  `301 passed in 19.52s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.82s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.03s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.68s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact context forwarding plus stable and prune-only behavior, while the
affected and architecture gates preserve child orchestration, raw ownership,
and all neighboring repair boundaries.

## Indexed Conv-input prune-aware summary family characterization

The next repeated family contains two compatible sites: very-late primary and
fallback. Both snapshot tensor count, invoke the shared indexed
singleton-Reshape plus stale-Transpose Conv-input owner, and extend its exact
two-key mapping with `pruned_unused_tensors`. The final primary site is
explicitly excluded because it invokes only the stale-Transpose repair and has
a different one-key schema.

`tests/test_flatbuffer_direct_indexed_conv_input_summary_orchestration.py`
fixes both current two-statement representations, model arguments, prune
expressions, predecessors, very-late successor, fallback reconciliation guard,
and retained raw lowerer wrapper. Its strict expected failure requires one
shared pass-module prune-aware summary owner. No production source, indexed
repair, graph mutation, guard, store entry, public API, artifact, dependency,
or TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`115 passed, 1 xfailed in 2.65s` across the dedicated family contract and
related indexed Conv-input, very-late normalization, fallback, terminal-layout,
and phase-store contracts. The sole expected failure is the intentionally
absent shared summary owner.

## Indexed Conv-input prune-aware summary family implementation

`run_indexed_conv_input_adapter_repairs_summary(model_ir)` now owns the tensor
snapshot, shared indexed two-repair invocation, and non-negative prune delta.
The very-late primary and fallback sites retain their existing stats targets
and model arguments while removing both lowerer-local count variables and
inline mapping extensions.

The raw lowerer compatibility wrapper remains defined. The indexed owner still
constructs one `ModelIRGraphIndex` and returns the same two repair counters.
The fallback mutation-positive reconciliation guard, very-late dynamic-Reshape
predecessor, stale-channel successor, public behavior, artifacts, dependencies,
and TensorFlow isolation are unchanged. The final primary site continues to
invoke only the stale-Transpose repair and retains its distinct one-key-plus-
prune schema. Both shared summaries remain outside the full 128/128 store.

Final sequential validation under core-only `uv`:

- focused shared-summary contracts: `4 passed in 0.57s`;
- affected family, boundary, store, and architecture contracts:
  `376 passed in 21.20s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.81s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 8.93s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.58s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves stable and prune-only schema preservation, while family and architecture
gates preserve the one-index owner, both compatible sites, fallback guard, and
excluded final site.

## Stale channelwise-binary adapter summary family characterization

The next repeated evidence family contains the fallback and final-primary
stale channelwise-binary adapter sites. Both snapshot tensor count, invoke the
same raw stale NCHW-to-NHWC Transpose repair, and extend its exact one-key
mapping with `pruned_unused_tensors`. The indexed convergence loop is excluded:
it shares a `ModelIRGraphIndex` across multiple repairs and iterations and
therefore has materially different ownership and ordering semantics.

`tests/test_flatbuffer_direct_stale_binary_adapter_summary_orchestration.py`
fixes both current two-statement representations, exact model arguments and
prune expressions, preceding concat-axis reconciliation guards, following
mutation-positive reconciliation guards, fallback topology successor, final
progress successor, and the retained raw lowerer wrapper. Its strict expected
failure requires one shared pass-module prune-aware summary owner. No
production source, graph mutation, guard, store entry, public API, artifact,
dependency, or TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`93 passed, 1 xfailed in 2.30s` across the dedicated family contract and the
related fallback, terminal-layout, indexed binary-convergence, and phase-store
contracts. The sole expected failure is the intentionally absent shared
summary owner. Targeted Ruff and whitespace checks passed.

## Stale channelwise-binary adapter summary family implementation

`run_stale_binary_adapter_repair_summary(model_ir)` now owns the tensor-count
snapshot, one raw stale-adapter repair invocation, and the non-negative prune
delta. The fallback and final-primary stats targets remain, while their two
lowerer-local count variables and inline mapping extensions are removed.

The raw lowerer wrapper remains defined and continues to forward an optional
`ModelIRGraphIndex`. The indexed convergence owner remains unchanged and still
shares one graph index across its broadcast, adapter, and shape-reconciliation
steps. Both mutation-positive reconciliation guards, their phase IDs, the
fallback topology successor, final progress successor, public behavior,
artifacts, dependencies, TensorFlow isolation, and the full 128/128
phase-result store are unchanged. The shared summaries remain outside that
store.

Final sequential validation under core-only `uv`:

- focused shared-summary contracts: `4 passed in 0.57s`;
- affected fallback, terminal-layout, indexed convergence, store, and
  architecture contracts: `354 passed in 20.61s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.80s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.11s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.60s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves stable and prune-only schema preservation, while the affected and
architecture gates preserve raw-wrapper forwarding, iterative indexed
convergence, both conditional reconciliations, and neighboring boundaries.

## Final stale Conv-input summary characterization

The final-primary stale Conv-input site is now characterized as a dedicated
one-repair family. It snapshots tensor count, invokes only the stale
NCHW-to-NHWC Conv-input Transpose repair, and extends that exact one-key mapping
with `pruned_unused_tensors`. This keeps it separate from the already-extracted
indexed owner, which also runs singleton-Reshape repair and returns a two-key
raw schema.

`tests/test_flatbuffer_direct_final_conv_input_summary_orchestration.py` fixes
the current two-statement representation, exact model argument and prune
expression, preceding final-Pad reconciliation guard, following
mutation-positive reconciliation guard, subsequent mixed-Concat repair, and
retained raw lowerer wrapper with optional graph-index forwarding. Its strict
expected failure requires a dedicated pass-module summary owner. No production
source, graph mutation, guard, store entry, public API, artifact, dependency,
or TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`336 passed, 1 xfailed in 18.50s` across the dedicated contract and related
indexed Conv-input, terminal-layout, phase-store, and architecture contracts.
The sole expected failure is the intentionally absent dedicated summary owner.
Targeted Ruff and whitespace checks passed.

## Final stale Conv-input summary implementation

`run_stale_conv_input_adapter_repair_summary(model_ir)` now owns the
tensor-count snapshot, one raw stale Conv-input Transpose repair invocation,
and the non-negative prune delta. The final-primary stats target remains while
its lowerer-local count variable and inline mapping extension are removed.

The raw lowerer wrapper remains defined and continues to forward an optional
`ModelIRGraphIndex`. The two indexed summary sites retain their shared
singleton-Reshape plus stale-Transpose owner and two-key schema. The final-Pad
predecessor, mutation-positive reconciliation guard and phase ID, mixed-Concat
successor, public behavior, artifacts, dependencies, TensorFlow isolation, and
the full 128/128 phase-result store are unchanged. The dedicated summary
remains outside that store.

Final sequential validation under core-only `uv`:

- focused dedicated-summary contracts: `4 passed in 0.56s`;
- affected indexed Conv-input, terminal-layout, store, and architecture
  contracts: `339 passed in 20.54s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.17s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.65s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves stable and prune-only schema preservation, while the affected and
architecture gates preserve the raw and indexed owners, all three production
summary sites, optional graph-index forwarding, and neighboring boundaries.

## Final PRELU prune-aware summary characterization

The next isolated boundary is the absolute-final PRELU passthrough repair. The
raw owner always performs unused-tensor pruning, including on a zero-match
call. The lowerer currently snapshots tensor count around that one-key result
and reconciles shapes after either a PRELU rewrite or prune-only cleanup.

`tests/test_flatbuffer_direct_final_prelu_summary_orchestration.py` fixes the
current count-plus-call representation, exact ModelIR and `LayoutState`
arguments, preceding SE-FC/Gather reconciliation guard, rewrite-or-prune
condition, following consecutive-Reshape cleanup, and retained raw lowerer
wrapper. Its strict expected failure requires one dedicated pass-module
prune-aware summary and the existing generic positive-count predicate. No
production source, graph mutation, pass order, store entry, public API,
artifact, dependency, or TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`389 passed, 1 xfailed in 19.45s` across the dedicated contract and related
terminal-layout, SE-FC/Gather, core runtime, phase-store, and architecture
contracts. The sole expected failure is the intentionally absent dedicated
summary owner. Targeted Ruff and whitespace checks passed.

## Final PRELU prune-aware summary implementation

`run_prelu_transpose_passthrough_summary(model_ir, layout_state=...)` now owns
the tensor-count snapshot, one raw PRELU passthrough invocation, exact
`LayoutState` forwarding, and the non-negative prune delta. The final stats
target remains while its lowerer-local count variable is removed. The lowerer
uses the existing integer-mapping `_stats_have_positive_count` predicate, so a
rewrite or prune-only cleanup still triggers the same reconciliation phase.

The raw lowerer wrapper and its graph-index/layout/max-rewrite/candidate
forwarding remain defined. The layout-recovery and late-binary-recovery PRELU
paths remain raw and unchanged; architecture coverage fixes the total raw
owner use at those two paths plus the new summary owner. The preceding
SE-FC/Gather guard, following consecutive-Reshape cleanup, public behavior,
artifacts, dependencies, TensorFlow isolation, and the full 128/128
phase-result store are unchanged. The summary remains outside that store.

Final sequential validation under core-only `uv`:

- focused dedicated-summary contracts: `4 passed in 0.56s`;
- affected terminal-layout, SE-FC/Gather, core runtime, store, and architecture
  contracts: `392 passed in 21.86s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.37s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.68s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact layout-state forwarding plus stable and prune-only schema
preservation, while affected and architecture gates preserve the raw owner use
count, both other production paths, reconciliation semantics, and neighboring
boundaries.

## Fallback norm-subgraph Pad summary characterization

The next isolated boundary is the safety-fallback norm-only Pad layout cleanup.
It snapshots tensor count, invokes `run_pad_layout_cleanup` with Pad and unary
stages disabled, norm enabled, no layout state, and the conversion diagnostics,
then extends the exact three-key raw mapping with prune evidence. Its existing
guard consumes only the norm rewrite counter; prune-only evidence remains
observational.

`tests/test_flatbuffer_direct_fallback_norm_summary_orchestration.py` fixes the
current count-plus-mapping representation, all fixed flags and arguments,
recursive fallback predecessor, conditional reconciliation guard, dynamic
rank-one successor, and future dedicated summary contract. Its strict expected
failure requires `run_norm_subgraph_pad_layout_summary(model_ir,
diagnostics=...)`. Other Pad-family callers and schemas remain independent. No
production source, graph mutation, pass order, store entry, public API,
artifact, dependency, or TensorFlow boundary changed.

Sequential characterization under core-only `uv` completed with
`300 passed, 1 xfailed in 18.26s` across the dedicated contract and related
fallback, Pad result/orchestration, norm reconciliation, singleton-Reshape,
phase-store, and architecture contracts. The sole expected failure is the
intentionally absent summary owner. Targeted Ruff and whitespace checks passed.

## Fallback norm-subgraph Pad summary implementation

`run_norm_subgraph_pad_layout_summary(model_ir, diagnostics=...)` now owns the
tensor-count snapshot, norm-only fixed flag policy, diagnostics forwarding, one
raw Pad-family invocation, and the non-negative prune delta. The fallback stats
target remains while its lowerer-local count and inline mapping extension are
removed. The existing guard still consumes only the norm rewrite counter, so
prune-only evidence remains observational exactly as before.

The raw `run_pad_layout_cleanup` lowerer import remains an explicit
compatibility re-export. Every other Pad-family orchestration route and result
schema is unchanged. Architecture coverage counts the raw runner inside the
summary, preserving the total 120 ordered runner invocations. The recursive
fallback predecessor, conditional reconciliation phase, dynamic rank-one
successor, public behavior, artifacts, dependencies, TensorFlow isolation, and
the full 128/128 phase-result store are unchanged. The summary remains outside
that store.

Final sequential validation under core-only `uv`:

- focused dedicated-summary contracts: `4 passed in 0.57s`;
- affected fallback, Pad, norm, singleton-Reshape, store, and architecture
  contracts: `303 passed in 19.72s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.82s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.17s`;
- phase-store capacity contracts: `2 passed in 0.55s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.98s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact flag and diagnostics forwarding plus stable and prune-only schema
preservation, while affected and architecture gates preserve all other Pad
routes, runner count, reconciliation semantics, and neighboring boundaries.

## Final placeholder binary-adapter summary characterization

The next isolated boundary is the indexed exact/singleton binary-adapter pair
inside the final placeholder-MatMul recovery guard. Its two raw mappings have
disjoint counter keys. The lowerer currently snapshots tensor count, retains
both mappings separately, and combines them with the preceding placeholder
shape-reconciliation mapping plus prune-only change.

`tests/test_flatbuffer_direct_final_placeholder_binary_summary_orchestration.py`
fixes the current count-plus-pair representation, pair order and model argument,
preceding reconciliation mapping, exact rewrite-or-prune guard, following
topology checkpoint, and future merged summary contract. Its strict expected
failure requires `run_indexed_binary_layout_adapter_summary(model_ir)` to
return both raw counters plus prune evidence while leaving the existing pair
owner available to every other caller. No production source, graph mutation,
pass order, store entry, public API, artifact, dependency, or TensorFlow
boundary changed.

Sequential characterization under core-only `uv` completed with
`380 passed, 1 xfailed in 21.00s` across the dedicated contract and related
indexed binary-adapter, terminal-layout, core runtime, phase-store, and
architecture contracts. The sole expected failure is the intentionally absent
merged summary owner. Targeted Ruff and whitespace checks passed.

## Final placeholder binary-adapter summary implementation

`run_indexed_binary_layout_adapter_summary(model_ir, graph_index=...,
layout_state=...)` now owns the tensor-count snapshot, one indexed pair
invocation, optional graph-index and layout-state forwarding, ordered merge of
the two disjoint raw counter mappings, and the non-negative prune delta. The
final placeholder site replaces its count and two raw-result locals with one
merged mapping. Its guard still combines the preceding placeholder
reconciliation mapping with all binary rewrite and prune evidence.

The raw pair owner remains available and unchanged. Shared-late, late-binary,
and fallback callers continue to consume independent ordered results;
owner-aware tests preserve four total raw pair uses including the new summary.
The following topology checkpoint, public behavior, artifacts, dependencies,
TensorFlow isolation, and full 128/128 phase-result store are unchanged. The
summary remains outside that store.

Final sequential validation under core-only `uv`:

- focused merged-summary contracts: `4 passed in 0.55s`;
- affected indexed adapter, terminal-layout, core runtime, store, and
  architecture contracts: `383 passed in 19.66s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.83s`;
- synthetic core runtime contracts: `55 passed in 0.95s`;
- result contracts: `196 passed in 9.25s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.92s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves exact schema merge, context forwarding, and stable/prune-only behavior,
while affected and architecture gates preserve pair order, raw use count,
other callers, reconciliation semantics, and topology order.

## SiNet SE-FC/Gather shared-summary characterization

The next isolated family is the duplicated fallback and absolute-final
SiNet/SE-FC/Gather cleanup sequence. Both sites currently snapshot tensor
count, run the SiNet shuffle-tail optimizer, run the ordered SE-FC and Gather
fanout pair, and reconcile static shapes after either a reported rewrite or
prune-only cleanup. The two sites differ only in their ModelIR and layout-state
arguments.

`tests/test_flatbuffer_direct_sinet_se_fc_gather_summary_orchestration.py`
fixes both current call sequences, exact argument forwarding, rewrite-or-prune
guards, neighboring fallback/final boundaries, and continued availability of
the raw pair helper. Its one strict expected failure describes the future
`run_sinet_se_fc_gather_summary(context)` owner: one tensor snapshot, one SiNet
owner call, one ordered pair-owner call, a merged bounded mapping, and one
generic positive-count guard at each production site.

Sequential characterization under core-only `uv` completed with
`407 passed, 1 xfailed in 19.96s` across the dedicated contract and related
SE-FC/Gather, safety-fallback, terminal-layout, synthetic core, phase-store,
and architecture contracts. The sole expected failure is the intentionally
absent shared summary owner. Targeted Ruff and whitespace checks passed.

This checkpoint changes no production source, graph mutation, pass order,
phase-result entry, public API, artifact, dependency, or TensorFlow boundary.
No real-model conversion was repeated because it is a characterization-only
change.

## SiNet SE-FC/Gather shared-summary implementation

`run_sinet_se_fc_gather_summary(context)` now owns the tensor-count snapshot,
one SiNet shuffle-tail optimizer call, and the existing ordered SE-FC/Gather
fanout pair. It forwards the exact ModelIR and `LayoutState` to the SiNet
owner, forwards the unchanged `ModelIRPassContext` (including diagnostics) to
the pair owner, and returns the three normalized rewrite counters plus a
non-negative `pruned_unused_tensors` counter.

The fallback and absolute-final sites each replace one tensor snapshot, one
SiNet result, and two pair-result locals with one bounded summary mapping. Both
sites use the existing `_stats_have_positive_count` predicate, preserving
reconciliation after any of the three rewrites or prune-only cleanup. Their
path-specific ModelIR/LayoutState arguments, SiNet-before-pair order,
reconciliation phase IDs, and neighboring pass boundaries are unchanged.

The raw lowerer SiNet wrapper and raw SE-FC/Gather context helper remain
defined for compatibility. The shared summary is deliberately outside the
already-full phase-result store, which remains exactly 128 phase IDs and 128
owners. Public behavior, artifacts, dependencies, TensorFlow isolation, and
single-process validation policy are unchanged.

Final sequential validation under core-only `uv`:

- focused shared-summary contracts: `4 passed in 0.54s`;
- affected SE-FC/Gather, fallback, terminal-layout, core runtime, store, and
  architecture contracts: `410 passed in 21.87s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.83s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.32s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.66s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because focused runtime coverage
proves ordered owner invocation, exact context forwarding, stable schema, all
three rewrite-evidence paths, and prune-only reconciliation, while affected
structural gates preserve both production boundaries and compatibility
wrappers.

## Shared precision-cleanup sequence characterization

The next compatible repeated family is the fallback and primary-final
precision cleanup sequence. Both paths rewrite eligible constant DIV operators
to reciprocal MUL, transactionally fold consecutive constant MUL chains, and
restore precision-sensitive reciprocal divisions in that exact order. The
fallback path intentionally omits layout state; the primary-final path forwards
the conversion layout state. Only the middle transactional stage receives
conversion diagnostics.

`tests/test_flatbuffer_direct_precision_cleanup_orchestration.py` fixes both
three-call sequences, all six current result targets, exact ModelIR/layout/
diagnostics arguments, the fallback topology predecessor and unbound-repair
successor, the primary topological-progress successor, and the total raw owner
occurrence counts. Its one strict expected failure describes a future
`run_precision_cleanup_sequence(context)` owner that returns all three raw
mappings in order rather than merging their independent schemas.

No GraphIndex sharing is proposed: the middle cleanup is transactional and
owns its own pass state, while mutations on either side require each indexed
owner to construct state from the current graph. This checkpoint changes no
production source, graph mutation, pass order, phase-result entry, public API,
artifact, dependency, or TensorFlow boundary.

Sequential characterization under core-only `uv` completed with
`365 passed, 1 xfailed in 18.94s` across the dedicated contract and related
precision, graph-cleanup, fallback, topology, terminal-layout, phase-store,
and architecture contracts. The sole expected failure is the intentionally
absent shared sequence owner. Targeted Ruff and whitespace checks passed. No
real-model conversion was repeated for this characterization-only change.

## Shared precision-cleanup sequence implementation

`run_precision_cleanup_sequence(context)` now owns the ordered
DIV-to-reciprocal, transactional consecutive-MUL, and precision-sensitive DIV
restore calls. It returns the three raw mappings unchanged as a tuple, so their
independent schemas and source order remain explicit. A conditional layout
keyword mapping preserves the prior callback contract exactly: no layout
keyword is sent for fallback, while the conversion `LayoutState` is sent to all
three primary-final stages. Diagnostics continue to reach only the middle
transactional stage.

The fallback and primary-final sites each replace three individual unconsumed
mapping locals with one ordered-result local. The preceding fallback topology
checkpoint, following fallback unbound-input repair, following primary progress
description and sort, graph mutations, pruning, layout synchronization, and
diagnostics are unchanged. The earlier core consecutive-MUL phase-result call
remains independent and direct.

The lowerer retains both private precision imports as explicit compatibility
re-exports. No GraphIndex is shared across the transaction boundary, no result
is added to the already-full 128/128 phase-result store, and no new guard,
scan, dependency, TensorFlow import, public API, or artifact behavior is
introduced.

Final sequential validation under core-only `uv`:

- focused shared-sequence contracts: `4 passed in 0.55s`;
- affected precision, graph-cleanup, fallback, topology, terminal-layout,
  store, and architecture contracts: `368 passed in 20.89s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.89s`;
- synthetic core runtime contracts: `55 passed in 0.96s`;
- result contracts: `196 passed in 9.39s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.88s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because runtime coverage proves
exact raw schema/order, both layout policies, and diagnostics forwarding,
while affected structural gates preserve all neighboring boundaries and the
independent core caller.

## Inherited shared-late successor contract repair

An expanded characterization gate exposed one stale test-only boundary from
before the late-binary boolean-owner extraction. The shared-late contract still
expected the removed `late_binary_repair_tensor_count` snapshot even though
production, the late-binary owner tests, and all neighboring orchestration now
use `_late_binary_repair_requires_reconciliation`.

The single test constant now names the current boolean successor. Both affected
assertions pass (`13 passed in 0.56s`). No production source, graph behavior,
public API, artifact, dependency, TensorFlow boundary, or phase-result store
changed.

## Absolute-final boundary-signature pair characterization

The next selected cluster is the adjacent absolute-final dynamic-boundary
signature realignment and static-signature consistency sanitization. Both raw
owners are already independently characterized, return bounded mappings, and
require only the current ModelIR. The pair sits immediately before the
absolute-final affine post-Add cleanup.

`tests/test_flatbuffer_direct_boundary_signature_cleanup_orchestration.py`
fixes the two current result targets, realign-before-sanitize order, exact
ModelIR argument, other lowerer occurrence counts, following affine boundary,
and continued availability of both lowerer wrappers. Its one strict expected
failure requires `run_boundary_shape_signature_cleanup(model_ir)` in the
existing signature-sanitization module, returning both raw mappings in order.
The later terminal realignment and the late-binary sanitizer caller remain
independent.

Sequential characterization under core-only `uv` completed with
`361 passed, 1 xfailed in 18.70s` across the dedicated contract and related
signature, terminal-layout, late-binary, shared-late, phase-store, and
architecture contracts. The sole expected failure is the intentionally absent
ordered pair owner. Targeted Ruff and whitespace checks passed.

This checkpoint changes no production source, metadata/tensor mutation, pass
order, store entry, public API, artifact, dependency, or TensorFlow boundary.
No real-model conversion was repeated for this characterization-only change.

## Absolute-final boundary-signature pair implementation

`run_boundary_shape_signature_cleanup(model_ir)` now lives beside the two raw
signature owners and returns their mappings unchanged as an ordered pair. It
always realigns the dynamic-boundary signature map before sanitizing static
shape-signature consistency, preserving the exact metadata/tensor mutation
order.

The absolute-final lowerer site replaces its two individual result locals with
one ordered-result tuple. The following affine post-Add cleanup, later terminal
dynamic-boundary realignment, shared-late realignment, and late-binary
sanitization remain unchanged. Both lowerer wrappers remain available for
compatibility, and owner-aware architecture tests preserve three total realign
routes and two total sanitize routes.

The new owner requires no context object, graph index, layout state,
diagnostics, reconciliation, or guard. It remains outside the full 128/128
phase-result store and adds no dependency, TensorFlow import, public API, or
artifact behavior.

Final sequential validation under core-only `uv`:

- focused ordered-pair contracts: `3 passed in 0.57s`;
- affected signature, terminal, normalization/attention, late-binary,
  shared-late, store, and architecture contracts: `374 passed in 20.32s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.80s`;
- synthetic core runtime contracts: `55 passed in 0.91s`;
- result contracts: `196 passed in 9.05s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.90s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because runtime coverage proves
exact raw order, ModelIR identity, and schema preservation, while affected
structural gates preserve all independent callers and neighboring boundaries.

## No-layout final SE-FC/affine pair characterization

The next selected cluster is the guarded no-layout final SE-FC cleanup followed
by affine pre/post cleanup. Both calls use the same primary ModelIR and
`LayoutState`; only SE-FC receives conversion diagnostics. Their raw mappings
are unconsumed, and the pair is immediately followed by the guarded topology
checkpoint.

`tests/test_flatbuffer_direct_no_layout_final_cleanup_orchestration.py` fixes
the guard, both current result targets, exact callback arguments, SE-FC-before-
affine order, preceding primary topology checkpoint, guarded topology
successor, following boundary-signature cleanup, affine wrapper retention, and
SE-FC compatibility import. Its one strict expected failure requires
`run_no_layout_final_cleanup(shared_model_ir_pass_context)` to return both raw
mappings in order.

Sequential characterization under core-only `uv` completed with
`367 passed, 1 xfailed in 18.74s` across the dedicated contract and related
terminal-layout, topology, affine, SE-layout, pass-efficiency, phase-store,
architecture, and boundary-signature contracts. The sole expected failure is
the intentionally absent ordered context owner. Targeted Ruff and whitespace
checks passed.

This checkpoint changes no production source, graph mutation, pass order,
guard, topology phase, store entry, public API, artifact, dependency, or
TensorFlow boundary. No real-model conversion was repeated.

## No-layout final SE-FC/affine pair implementation

`passes/no_layout_final_cleanup_orchestration.py` now owns the guarded final
no-layout cleanup pair through
`run_no_layout_final_cleanup(shared_model_ir_pass_context)`. The owner forwards
the same primary ModelIR and `LayoutState` to both raw passes, forwards
conversion diagnostics only to SE-FC cleanup, and returns the two unchanged raw
mappings in their original SE-FC-before-affine order.

The lowerer now retains one ordered result tuple instead of the two former
unconsumed locals. The existing option guard, preceding primary topology
checkpoint, following guarded topology checkpoint, boundary-signature
successor, affine lowerer wrapper, SE-FC compatibility re-export, and every
other raw caller remain unchanged. The owner adds no graph scan, result
normalization, control-flow decision, phase-store entry, dependency,
TensorFlow import, public API, or artifact behavior; the phase-result store
remains exactly 128/128.

Final sequential validation under core-only `uv`:

- focused owner contracts: `3 passed in 0.57s`;
- affected terminal-layout, topology, affine, SE-layout, efficiency, store,
  architecture, and boundary-signature contracts: `369 passed in 18.98s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.78s`;
- synthetic core runtime contracts: `55 passed in 0.91s`;
- result contracts: `196 passed in 9.15s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.62s`;
- targeted Ruff passed.

No real-model corpus conversion was repeated because the runtime contract
proves exact context identity, argument policy, callback order, and raw result
schema, while the affected structural gates preserve both neighboring
topology boundaries and all independent callers.

## Inherited late-binary successor contract repair

The broader next-cluster characterization gate exposed one stale test-only
successor assertion. Production already routes the first post-late-binary
boundary through `_pre_terminal_instancenorm_layout_results` and
`run_pre_terminal_instancenorm_layout_cleanup(shared_model_ir_pass_context)`,
but the older test still expected the removed direct InstanceNorm post-bias
target and raw wrapper call.

The assertion now follows the existing production owner, including its exact
shared-context argument and empty keyword policy. The focused file passes
(`5 passed in 0.55s`). No production source, graph behavior, pass order,
public API, artifact, dependency, TensorFlow boundary, or 128/128 phase-result
store changed.

## Absolute-final affine/InstanceNorm pair characterization

The next selected cluster is the adjacent absolute-final affine post-ADD
cleanup and decomposed-InstanceNorm post-bias cleanup. Both calls receive the
same primary ModelIR and `LayoutState`, return independent one-counter raw
mappings, and run immediately after boundary-signature cleanup and before the
existing normalization/attention owner.

`tests/test_flatbuffer_direct_absolute_final_affine_instancenorm_orchestration.py`
fixes both current result targets, affine-before-InstanceNorm order, exact
ModelIR/layout arguments, signature predecessor, normalization/attention
successor, and continued availability of both lowerer wrappers. Its one strict
expected failure requires
`run_absolute_final_affine_instancenorm_cleanup(shared_model_ir_pass_context)`
to return both raw mappings unchanged in order.

Sequential characterization under core-only `uv` completed with
`676 passed, 1 xfailed in 21.46s` across the dedicated contract and affected
boundary-signature, normalization/attention, affine, InstanceNorm, terminal,
late-binary, phase-store, and architecture contracts. The sole expected
failure is the intentionally absent context owner. Focused Ruff and whitespace
checks passed.

This checkpoint changes no production source, graph mutation, pass order,
argument policy, store entry, public API, artifact, dependency, or TensorFlow
boundary. No real-model conversion was repeated.

## Absolute-final affine/InstanceNorm pair implementation

`passes/absolute_final_affine_instancenorm_orchestration.py` now provides
`run_absolute_final_affine_instancenorm_cleanup(shared_model_ir_pass_context)`.
It runs indexed affine post-ADD cleanup before decomposed-InstanceNorm
post-bias cleanup, forwards the same primary ModelIR and `LayoutState` to both,
and returns both one-counter raw mappings unchanged as an ordered tuple.

The absolute-final lowerer site now retains one tuple instead of the two former
unconsumed locals. Boundary-signature cleanup remains the immediate
predecessor and the existing normalization/attention owner remains the
immediate successor. Both private lowerer wrappers and every other direct or
orchestrated raw caller remain available. The extraction adds no graph scan,
normalization, diagnostics forwarding, control-flow decision, phase-store
entry, dependency, TensorFlow import, public API, or artifact behavior; the
store remains exactly 128/128.

Final sequential validation under core-only `uv`:

- focused context-owner contracts: `3 passed in 0.58s`;
- affected signature, normalization/attention, affine, InstanceNorm,
  terminal, late-binary, store, and architecture contracts:
  `678 passed in 23.97s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.81s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 9.18s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.61s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model corpus conversion was repeated because injected runtime tests
prove exact context identity, callback order, arguments, and raw schemas,
while the owner-aware affected suite preserves all independent occurrences and
neighboring boundaries.

## Inherited target-context construction contract repair

The next characterization gate exposed one stale test-only construction
count. The lowerer already contains four target-specific
`ModelIRPassContext(...)` constructions for SE-FC/Gather, SiNet summary,
precision cleanup, and singleton/consecutive-Reshape helpers; the old test
still expected only two.

The contract now identifies all four owning helpers explicitly, requires
exactly one construction in each, and preserves the identical target ModelIR,
target LayoutState, and session diagnostics arguments. The focused file passes
(`29 passed in 0.55s`). No production source, context lifetime, graph behavior,
pass order, API, artifact, dependency, TensorFlow boundary, or 128/128 store
entry changed.

## Absolute-final normalization/attention rank-one characterization

The next selected cluster is the existing absolute-final normalization/pad and
mixed-attention owner followed immediately by dynamic rank-one
Unsqueeze/Reshape shape-input repair. The first result is already an ordered
two-mapping tuple; the second is an independent one-counter mapping. Both use
the same primary ModelIR/LayoutState context and precede the absolute-final
topology/layout refresh.

`tests/test_flatbuffer_direct_absolute_final_normalization_attention_rank1_orchestration.py`
fixes the two current result targets, exact owner-before-rank-one order,
ModelIR/layout arguments, the affine/InstanceNorm predecessor, topology/layout
successor, current lowerer closure and context alias, and retention of the raw
dynamic-rank-one wrapper. Its one strict expected failure requires
`run_absolute_final_normalization_attention_rank1_cleanup(shared_model_ir_pass_context)`
to return the existing normalization/attention tuple and rank-one mapping as a
nested ordered pair.

Sequential characterization under core-only `uv` completed with
`423 passed, 1 xfailed in 19.29s` across the dedicated contract and affected
normalization/attention, affine/InstanceNorm, shared-context, topology/layout,
safety-fallback, dynamic-Reshape, terminal-layout, store, and architecture
contracts. The sole expected failure is the intentionally absent composite
context owner. Focused Ruff and whitespace checks passed.

This checkpoint changes no production source, callback, context lifetime,
graph mutation, pass order, result schema, store entry, API, artifact,
dependency, or TensorFlow boundary. No real-model conversion was repeated.

## Absolute-final normalization/attention rank-one implementation

The existing
`passes/absolute_final_normalization_attention_orchestration.py` now exposes
`run_absolute_final_normalization_attention_rank1_cleanup(context)`. It first
runs the existing normalization/pad plus mixed-attention owner, then runs
dynamic rank-one Unsqueeze/Reshape repair with the same ModelIR/LayoutState.
It returns a nested ordered pair: the original two-mapping normalization/
attention tuple followed by the unchanged rank-one mapping. No result is
flattened or normalized.

The lowerer now retains one composite result instead of two unconsumed locals.
Its redundant zero-argument closure and dedicated shared-context alias are
removed. The affine/InstanceNorm predecessor, absolute-final topology/layout
successor, raw dynamic-rank-one lowerer wrapper, safety-fallback and very-late
rank-one callers, internal shared state scope, and all result schemas remain
unchanged. The composite remains outside the full 128/128 phase-result store
and adds no scan, control-flow decision, dependency, TensorFlow import, public
API, or artifact behavior.

Final sequential validation under core-only `uv`:

- focused nested-schema contracts: `3 passed in 0.69s`;
- affected normalization/attention, affine/InstanceNorm, shared-context,
  topology/layout, safety-fallback, dynamic-Reshape, terminal, store, and
  architecture contracts: `425 passed in 20.73s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.84s`;
- synthetic core runtime contracts: `55 passed in 0.95s`;
- result contracts: `196 passed in 9.30s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.98s`;
- targeted Ruff, bytecode compilation, 128/128 audit, and whitespace checks:
  passed.

No real-model conversion was repeated because runtime callback injection
proves the exact nested schema, context identity, callback order, and argument
policy, while owner-aware structural coverage accounts for every independent
dynamic-rank-one caller and both neighboring boundaries.

## Indexed binary-layout convergence owner characterization

The next selected boundary is the lowerer-local indexed binary-layout
convergence implementation. It creates one `ModelIRGraphIndex`, runs broadcast-
constant repair, stale binary-adapter repair, and static-shape reconciliation
in that order for at most three rounds, stops after a stable round, and returns
three accumulated counters. Both the safety fallback and primary terminal path
call the same implementation.

`tests/test_flatbuffer_direct_binary_layout_convergence_owner.py` fixes the
single-index construction, three-round cap, exact callback order and
graph-index forwarding, result-key order, and the two existing caller
arguments. Its one strict expected failure requires
`passes/binary_layout_convergence.py` to own
`run_indexed_binary_layout_convergence(model_ir)` while the private lowerer
function becomes a one-return compatibility wrapper.

Sequential characterization under core-only `uv` completed with
`399 passed, 1 xfailed in 19.19s` across the dedicated contract and affected
convergence runtime, terminal-layout, safety-fallback, binary-adapter,
stale-repair, phase-store, and architecture contracts. The sole expected
failure is the intentionally absent pass-module owner. Focused Ruff and
whitespace checks passed.

This checkpoint changes no production source, graph-index lifetime, callback,
round count, stable-stop rule, result schema, caller, store entry, API,
artifact, dependency, or TensorFlow boundary. No real-model conversion was
repeated.

## Indexed binary-layout convergence owner implementation

The complete indexed binary-layout convergence loop now lives in
`passes/binary_layout_convergence.py`. The owner constructs exactly one
`ModelIRGraphIndex`, shares it across broadcast-constant repair, stale
binary-adapter repair, and static-shape reconciliation, and preserves the
original stage order. It still executes no more than three rounds, stops after
the first round with no positive mutation evidence, and returns the same three
accumulated counters in the same order.

The private lowerer function remains as a one-return compatibility adapter,
and its safety-fallback and primary-terminal callers are unchanged. Runtime
monkeypatch coverage now targets the pass-module owner, while structural tests
verify the lowerer wrapper, both caller arguments, single-index lifetime,
callback forwarding, stable-stop behavior, round cap, and result schema. The
change adds no phase result: the bounded store remains exactly 128 phase IDs
and 128 owners.

Final sequential validation under core-only `uv`:

- focused owner contracts: `2 passed in 0.15s`;
- affected convergence, fallback, terminal-layout, binary-adapter,
  stale-repair, phase-store, and architecture contracts:
  `400 passed in 18.70s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.85s`;
- synthetic core runtime contracts: `55 passed in 0.95s`;
- result contracts: `196 passed in 8.93s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.53s`.

No real-model conversion was repeated for this mechanical ownership move.
Dedicated runtime tests exercise the exact convergence behavior and shared
graph-index identity, and the affected structural/runtime suite covers both
production call paths.

## Terminal stabilization composite characterization

The next selected boundary is the final primary stabilization triple
immediately before terminal topology/layout validation. The lowerer currently
retains three independent results while it runs indexed binary-layout
convergence, static high-rank binary coalescing, and dynamic boundary-signature
realignment in that exact order.

`tests/test_flatbuffer_direct_terminal_stabilization_orchestration.py` fixes
the three raw result names and schemas, model/layout argument policy, adjacency,
terminal validation successor, and finalizer successor. Its one strict
expected failure requires
`passes/terminal_stabilization_orchestration.py` to expose a single context
owner returning the three unchanged mappings as an ordered tuple, with the
lowerer retaining one composite result through the existing shared
`ModelIRPassContext`.

Sequential characterization under core-only `uv` completed with
`388 passed, 1 xfailed in 18.72s` across the dedicated contract and affected
binary-convergence, high-rank-binary, boundary-signature, terminal-validation,
shared-context, phase-store, and architecture contracts. The sole expected
failure is the intentionally absent composite owner. Focused Ruff and
whitespace checks passed.

This checkpoint changes no production source, callback, context lifetime,
graph mutation, pass order, argument policy, result schema, validation
boundary, store entry, API, artifact, dependency, or TensorFlow boundary. No
real-model conversion was repeated.

## Terminal stabilization composite implementation

`passes/terminal_stabilization_orchestration.py` now owns the final primary
stabilization triple. Using the existing shared `ModelIRPassContext`, it runs
indexed binary-layout convergence, static high-rank binary coalescing, and
dynamic boundary-signature realignment in the original order. It forwards
ModelIR to all three owners, forwards LayoutState only to high-rank binary
coalescing, and returns the three original mappings unchanged as an ordered
tuple.

The lowerer now retains one `_final_terminal_stabilization_results` composite
instead of three individual unconsumed result locals. The two raw lowerer
compatibility wrappers remain available for independent callers, the safety-
fallback convergence call remains unchanged, and terminal topology/layout
validation and ModelIR finalization remain the immediate successors. The
lowerer no longer imports the high-rank binary owner solely for this terminal
site.

Runtime coverage injects all three callbacks and proves ModelIR/LayoutState
identity, exact call order, raw mapping object identity, and ordered tuple
shape. Owner-aware structural coverage accounts for the remaining wrapper
callers and each pass-module invocation. The composite remains outside the
full phase-result store, which stays exactly 128 IDs and 128 owners.

Final sequential validation under core-only `uv`:

- focused context-owner runtime and structure contracts:
  `3 passed in 0.54s`;
- affected binary-convergence, high-rank-binary, boundary-signature,
  terminal-validation, shared-context, phase-store, and architecture
  contracts: `390 passed in 18.66s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.74s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.03s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.64s`.

No real-model conversion was repeated because this is a straight-line
ownership move and dedicated runtime injection proves the complete terminal
sequence, state identity, and raw result preservation.

## Absolute-final cleanup composite characterization

The remaining-result inventory selected the three adjacent absolute-final
cleanups immediately before `topology_layout.primary.absolute_final`. They run
boundary-signature cleanup, the existing affine/InstanceNorm composite, and the
existing normalization/attention/rank-one composite in that order. All three
results are independent and unconsumed; the latter two already receive the
same shared `ModelIRPassContext`.

`tests/test_flatbuffer_direct_absolute_final_cleanup_orchestration.py` fixes
the three raw result names and nested schemas, exact model/context argument
policy, adjacency, and topology/layout refresh successor. Its one strict
expected failure requires
`passes/absolute_final_cleanup_orchestration.py` to expose one context owner
that returns the three unchanged composite results as an ordered tuple, with
the lowerer retaining one replacement result.

Sequential characterization under core-only `uv` completed with
`387 passed, 1 xfailed in 19.31s` across the dedicated contract and affected
boundary-signature, affine/InstanceNorm, normalization/attention/rank-one,
terminal-layout, shared-context, phase-store, and architecture contracts. The
sole expected failure is the intentionally absent top-level context owner.
Focused Ruff and whitespace checks passed.

This checkpoint changes no production source, callback, context identity,
graph mutation, pass order, nested result schema, topology/layout refresh,
store entry, API, artifact, dependency, or TensorFlow boundary. No real-model
conversion was repeated.

## Absolute-final cleanup composite implementation

`passes/absolute_final_cleanup_orchestration.py` now composes the three
characterized absolute-final owners through one shared `ModelIRPassContext`.
It runs boundary-signature cleanup with `context.model_ir`, then passes the
same context object to affine/InstanceNorm cleanup and normalization/attention/
rank-one cleanup. Their nested return values are neither flattened nor copied;
the owner returns the three original objects as an ordered outer tuple.

The lowerer now retains one `_absolute_final_cleanup_results` target instead
of the three former results. It no longer imports the three sub-owners solely
for this site. The guarded no-layout predecessor, every sub-owner and raw
compatibility wrapper, the absolute-final topology/layout refresh, and all
later repair boundaries remain unchanged. The composite stays outside the full
phase-result store, which remains exactly 128 IDs and 128 owners.

Runtime callback injection proves shared context identity, exact three-stage
order, nested tuple shape, and object identity for each result. Owner-aware
structural tests continue to validate every sub-owner's internal order and
schema while accounting for the new top-level caller.

Final sequential validation under core-only `uv`:

- focused top-level owner runtime and structure contracts:
  `3 passed in 0.55s`;
- affected boundary-signature, affine/InstanceNorm,
  normalization/attention/rank-one, terminal-layout, shared-context,
  phase-store, and architecture contracts: `389 passed in 18.97s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.75s`;
- synthetic core runtime contracts: `55 passed in 0.92s`;
- result contracts: `196 passed in 9.21s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.57s`.

No real-model conversion was repeated because the production change only
moves a straight-line sequence behind a context owner, and runtime injection
proves the complete order, argument identity, and nested results.

## Very-late dynamic/adapter composite characterization

The fresh inventory intentionally skipped the preceding orphan/unbound-input
group because one repair still exposes only a lowerer-local mapping wrapper.
The selected boundary instead contains six adjacent pass-module owners:
dynamic Reshape resolution, indexed Conv-input adapter summary, stale channel-
shuffle repair, Concat/Transpose/Conv axis repair, Concat/global-pool/Conv axis
repair, and dynamic rank-one Unsqueeze/Reshape repair. Mandatory static-shape
reconciliation follows immediately, then split fallback.

`tests/test_flatbuffer_direct_very_late_dynamic_adapter_orchestration.py`
fixes all six raw result names, exact callback order, ModelIR/LayoutState/
diagnostics argument policy, runtime-inferable dynamic-Reshape flag, mandatory
reconciliation successor, and split-fallback successor. Its one strict
expected failure requires
`passes/very_late_dynamic_adapter_orchestration.py` to expose one context owner
returning the six unchanged mappings as an ordered tuple, with one replacement
lowerer result.

Sequential characterization under core-only `uv` completed with
`497 passed, 1 xfailed in 18.94s` across the dedicated contract and affected
dynamic-Reshape, Conv-input, channel-shuffle, indexed Concat-axis, Conv-layout,
reconciliation, safety-fallback, shared-context, phase-store, efficiency, and
architecture contracts. The sole expected failure is the intentionally absent
context owner. Focused Ruff and whitespace checks passed.

This checkpoint changes no production source, callback, context identity,
graph mutation, pass order, flag, result schema, reconciliation, split
fallback, store entry, API, artifact, dependency, or TensorFlow boundary. No
real-model conversion was repeated.

## Very-late dynamic/adapter composite implementation

`passes/very_late_dynamic_adapter_orchestration.py` now owns the six-stage
sequence. It uses the existing shared `ModelIRPassContext`, preserves the
runtime-inferable dynamic-Reshape flag, forwards diagnostics only to stale
channel-shuffle repair, forwards LayoutState to the four layout-sensitive
repairs, and returns all six raw mappings unchanged as an ordered tuple. The
two private Concat-axis callbacks have clear module-local aliases; their
underlying owners are unchanged.

The lowerer now retains one `_very_late_dynamic_adapter_results` target instead
of six individual unconsumed results. Its dynamic-Reshape, Concat-axis, and
dynamic-rank-one compatibility wrappers remain available, the fallback Conv-
input summary call remains direct, and other independent callers are
unchanged. The very-late normalization predecessor, mandatory static-shape
reconciliation, and split fallback remain adjacent. The lowerer no longer
imports stale channel-shuffle repair solely for this site.

Runtime injection proves all six callback arguments, exact order, shared
ModelIR/LayoutState/diagnostics identity, the dynamic-Reshape flag, mapping
object identity, and outer tuple shape. Owner-aware structural tests account
for direct, orchestrated, fallback, and compatibility-wrapper callers. No
phase result was added; the store remains exactly 128 IDs and 128 owners.

Final sequential validation under core-only `uv`:

- focused context-owner runtime and structure contracts:
  `3 passed in 0.54s`;
- affected dynamic-Reshape, Conv-input, channel-shuffle, indexed Concat-axis,
  Conv-layout, reconciliation, safety-fallback, shared-context, phase-store,
  efficiency, and architecture contracts: `499 passed in 18.86s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.76s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 8.97s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.52s`.

No real-model conversion was repeated because this is a straight-line owner
move and dedicated runtime injection covers every callback, flag, shared-state
identity, raw result, and both neighboring boundaries.

## Unbound-input repair owner characterization

The next prerequisite checkpoint fixes the remaining lowerer-local mapping
wrapper before any larger orphan/unbound/affine group is composed. The wrapper
calls `repair_unbound_nonconstant_inputs_with_layout_transpose(model_ir,
graph_index=graph_index)`, conditionally reconciles static shapes with the
returned graph index when `result.repaired > 0`, and returns exactly
`repaired_unbound_nonconstant_inputs_with_layout_transpose: int(result.repaired)`.
Its two production callers remain the primary `model_ir` path and the
`fallback_ir` safety path.

`tests/test_flatbuffer_direct_unbound_input_repair_owner.py` first locks the
current lowerer behavior, then uses one strict expected failure to require a
pass-module owner plus a one-return lowerer compatibility adapter. Sequential
focused validation completed with `1 passed, 1 xfailed in 0.15s`. After an
independent owner-aware repair to the inherited very-late dynamic-adapter AST
contracts, the full affected set completed with
`392 passed, 1 xfailed in 19.37s`; the sole expected failure is the deliberately
absent pass-module owner.

This characterization changes no production callback, graph mutation,
reconciliation guard, result schema, caller, pass order, API, artifact,
dependency, or TensorFlow boundary. No real-model conversion was run. The
phase-result store remains exactly 128 IDs and 128 owners.

## Unbound-input repair owner implementation

`passes/unbound_input_repair_orchestration.py` now owns the characterized
repair and conditional reconciliation sequence. It forwards the optional
GraphIndex to the raw repair, reconciles only after a positive repaired count,
and forwards the exact GraphIndex returned by that repair. The existing
one-key integer mapping is returned unchanged.

The lowerer retains
`_repair_unbound_nonconstant_operator_inputs_with_layout_transpose` as a
one-return compatibility adapter, and its primary plus safety-fallback callers
remain unchanged. The lowerer no longer imports the raw layout repair solely
to implement this mapping. Runtime injection proves repair-before-reconcile
order and returned-index identity; inherited indexed-layout and architecture
contracts now inspect the pass-module owner at its actual boundary.

Final sequential validation under core-only `uv`:

- focused owner structure and runtime contracts: `3 passed in 0.58s`;
- affected unbound-input, QLinear, safety-fallback, very-late, terminal,
  architecture, and store contracts: `394 passed in 19.20s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.71s`;
- synthetic core runtime contracts: `55 passed in 0.91s`;
- result contracts: `196 passed in 8.86s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.46s`.

No phase result was added: the store remains exactly 128 IDs and 128 owners.
No public API, artifact, dependency, TensorFlow boundary, pass order, guard,
or result schema changed. No real-model conversion was repeated because this
mechanical owner move is covered by runtime identity/order checks and the full
affected synthetic suite.

## Recurrent-alias repair mapping-owner characterization

The refreshed adjacent-result inventory found one remaining boundary mismatch
before composing the late orphan/unbound/affine/normalization region. The raw
indexed recurrent-alias repair already lives in `passes/recurrent_alias.py`,
but the direct-TFLite lowerer still owns conversion of its integer return value
to the public `repaired_orphan_recurrent_step_tensors` mapping. The lowerer has
one primary production caller; PyTorch normalization continues to use the raw
owner through its independent compatibility wrapper.

`tests/test_flatbuffer_direct_recurrent_alias_repair_owner.py` locks the exact
raw call, optional GraphIndex forwarding, one-key integer schema, sole lowerer
caller, and compatibility-wrapper shape. Its strict expected failure requires
`passes/recurrent_alias_repair_orchestration.py` to own the mapping and the
lowerer wrapper to become one return dispatch.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.17s` focused and
`307 passed, 1 xfailed in 17.91s` across recurrent alias, late input repair,
unbound input, PyTorch recurrent normalization, direct lowering, very-late,
architecture, and phase-store contracts. The sole expected failure is the
intentionally absent mapping owner. Ruff and whitespace checks passed.

This checkpoint changes no production source, mutation, result schema,
GraphIndex policy, caller, PyTorch behavior, API, artifact, dependency,
TensorFlow boundary, or pass order. No real-model conversion was run, and the
phase-result store remains exactly 128 IDs and 128 owners.

## Recurrent-alias repair mapping-owner implementation

`passes/recurrent_alias_repair_orchestration.py` now owns direct-TFLite result
normalization for the shared indexed recurrent-alias repair. It forwards the
optional GraphIndex unchanged, invokes the existing raw mutation owner once,
and returns the same one-key integer mapping.

The lowerer's `_repair_orphan_recurrent_step_tensors` remains available as a
one-return compatibility adapter, and its sole primary production caller is
unchanged. `passes/recurrent_alias.py` remains the only graph-mutation owner.
The PyTorch recurrent wrapper continues to call that raw owner directly, so
its return convention and normalization pipeline are unaffected. Owner-aware
architecture coverage distinguishes these two intentional routes.

Final sequential validation under core-only `uv`:

- focused mapping-owner structure and runtime identity: `3 passed in 0.56s`;
- affected recurrent alias, late input repair, unbound input, PyTorch
  recurrent normalization, direct lowering, very-late, architecture, and
  phase-store contracts: `309 passed in 19.23s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.73s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 8.89s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.55s`.

No phase result, public API, artifact, dependency, TensorFlow boundary, graph
mutation, pass order, GraphIndex policy, or result schema changed. The store
remains exactly 128 IDs and 128 owners. No real-model conversion was repeated
because runtime injection and affected direct/PyTorch synthetic coverage prove
the complete mechanical owner move.

## Late input/affine/normalization composite characterization

With both result-normalization prerequisites extracted, the refreshed
inventory selected four adjacent unconditional mappings immediately after the
post-progress boundary and before the existing very-late dynamic-adapter
composite:

1. recurrent-alias repair summary;
2. unbound-input layout-repair summary;
3. affine post-Add cleanup with LayoutState;
4. prune-aware gather/constant/normalization summary.

All four use the same `shared_model_ir_pass_context`; the former dedicated
very-late normalization context is already an identity alias of that object.
Every result is unconsumed, and each child now has a pass-module owner.

`tests/test_flatbuffer_direct_late_input_affine_normalization_orchestration.py`
fixes exact source adjacency, progress predecessor, dynamic-adapter successor,
raw mapping schemas, callback order, ModelIR/LayoutState argument policy,
shared-context identity, and the absence of result consumers. Its strict
expected failure requires one context owner returning the four raw mapping
objects as an ordered tuple and one replacement lowerer target.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.57s` focused and
`394 passed, 1 xfailed in 17.86s` across late input repair, recurrent/unbound
indexed repair, affine post-Add, very-late normalization/dynamic adapter,
shared-context, architecture, and phase-store contracts. The sole expected
failure is the intentionally absent composite. An inherited normalization
test's stale pre-composite successor was corrected independently before this
checkpoint. Ruff and whitespace checks passed.

No production callback, graph mutation, result schema, context identity,
pass order, API, artifact, dependency, TensorFlow boundary, or store entry
changed. No real-model conversion was run; the phase-result store remains
exactly 128 IDs and 128 owners.

## Final boundary/Slice/Concat composite implementation

`passes/final_boundary_slice_concat_orchestration.py` now owns the
characterized four-stage final-layout sequence. It accepts the existing
`TerminalSliceConcatRecoveryContext`, passes that complete callback-bearing
context to terminal Slice/Concat recovery, and passes its exact
`context.pass_context` object to boundary channel, final Slice/pre-Concat, and
terminal Concat bridge cleanup. The four raw result tuples are returned
unchanged inside one ordered outer tuple.

The lowerer replaces four unconsumed locals with
`_final_boundary_slice_concat_results` and removes only the three direct child
imports made redundant by the move. Its terminal Slice/Concat compatibility
wrapper remains available and retains the earlier independent invocation. The
indexed final shape/activation result remains the predecessor, and the
optional terminal elementwise-fanout guard remains the successor.

Child-family and affected result tests now follow both ownership levels: the
new final composite owns the specialized layout composites, while each child
continues to own its existing pass family. Runtime callback injection proves
exact order, callback-context identity, shared pass-context identity, and raw
result identity. The lowerer wrapper and direct composite recovery call are
counted independently.

Final sequential validation under core-only `uv`:

- focused composite boundary, nested schema, route, and runtime identity:
  `4 passed in 0.55s`;
- focused owner-aware child, result, terminal-boundary, and architecture
  coverage: `349 passed in 20.39s`;
- complete affected suite: `400 passed in 19.35s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.75s`;
- synthetic core runtime contracts: `55 passed in 0.91s`;
- result contracts: `196 passed in 8.91s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.48s`.

Ruff, Python bytecode compilation, and whitespace checks passed. No phase
result, graph mutation, callback behavior, guard, public API, artifact,
dependency, TensorFlow boundary, or nested result schema changed. The store
remains exactly 128 IDs and 128 owners. No real-model conversion was repeated
because runtime injection directly proves the custom/shared context, results,
order, independent route, and boundary contracts.

## Very-late layout-tail composite characterization

The refreshed inventory selected four consecutive unconditional results after
late Swish passthrough cleanup: Conv1D/decoder layout cleanup, Pad/InstanceNorm
layout cleanup, singleton/consecutive Reshape cleanup, and optional layout-
Transpose plus broadcast repair. All four results are observation-only and
unconsumed.

The Conv1D, Pad/InstanceNorm, and broadcast owners already receive the shared
`ModelIRPassContext`. The primary singleton wrapper constructs an equivalent
context from the same ModelIR, LayoutState, and diagnostics; its fallback
caller remains independent and uses `fallback_ir` with no LayoutState. The
broadcast policy must preserve
`include_layout_transpose=optimize_layout_transpose_chains`. The following
very-late broadcast reconciliation remains phase-recorded and outside the
cluster.

`tests/test_flatbuffer_direct_very_late_layout_tail_orchestration.py` fixes
the four child owners, exact primary arguments and option policy, late-Swish
predecessor, recorded reconciliation successor, both singleton wrapper routes,
absence of consumers, and both empty-graph nested schemas. The result lengths
are `(8, 4, 3, 2)`; the final first element is `None` when layout-Transpose is
disabled and a mapping when enabled. Its strict expected failure requires one
shared-context owner returning all four raw tuples in order.

Two inherited tests were corrected independently before this checkpoint. The
Conv1D composite successor had already become the Pad/InstanceNorm composite,
and an affine post-Add occurrence had already moved from a direct lowerer call
to terminal Slice/Concat recovery. Both failed against unchanged production.

Sequential characterization under core-only `uv` completed with
`3 passed, 1 xfailed in 0.61s` focused and
`435 passed, 1 xfailed in 19.98s` across all four child families, option and
fallback routes, related result contracts, shared-late and absolute-final
owners, terminal layout, shared-context, architecture, and phase-store
contracts. The sole expected failure is the intentionally absent composite.
Ruff and whitespace checks passed.

No production callback, graph mutation, nested schema, context identity,
option policy, pass order, public API, artifact, dependency, TensorFlow
boundary, or store entry changed. No real-model conversion was run; the
phase-result store remains exactly 128 IDs and 128 owners.

## Late input/affine/normalization composite implementation

`passes/late_input_affine_normalization_orchestration.py` now owns the
characterized four-stage sequence through one `ModelIRPassContext`. It passes
`context.model_ir` to both repair summaries, forwards `context.layout_state`
only to affine post-Add cleanup, passes the exact context object to prune-aware
normalization, and returns all four original mapping objects in one ordered
tuple without copying or flattening.

The lowerer replaces four unconsumed locals with
`_late_input_affine_normalization_results`. Its recurrent, unbound-input, and
affine compatibility wrappers remain available; the fallback unbound-input
caller and every independent affine caller are unchanged. The raw very-late
normalization compatibility cluster and its shared-context alias also remain.
Only the now-redundant direct summary import was removed. The progress
predecessor and very-late dynamic-adapter successor remain adjacent.

Runtime callback injection proves exact four-stage order, ModelIR/LayoutState/
context identity, raw mapping identity, and outer tuple shape. Owner-aware
result, architecture, and normalization contracts now count the composite,
fallback, compatibility, and independent routes at their actual boundaries.

Final sequential validation under core-only `uv`:

- focused context-owner structure, empty schema, and runtime identity:
  `3 passed in 0.56s`;
- affected late input, recurrent/unbound indexed repair, affine post-Add,
  very-late normalization/dynamic adapter, shared-context, architecture, and
  phase-store contracts: `396 passed in 17.86s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.74s`;
- synthetic core runtime contracts: `55 passed in 0.91s`;
- result contracts: `196 passed in 9.10s`;
- phase-store capacity contracts: `2 passed in 0.54s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.47s`.

No phase result, graph mutation, callback, guard, public API, artifact,
dependency, TensorFlow boundary, or result schema changed. The store remains
exactly 128 IDs and 128 owners. No real-model conversion was repeated because
the move is straight-line and the dedicated runtime test proves all arguments,
state identities, raw results, and boundaries.

## Pre-terminal cleanup composite characterization

The post-removal AST inventory selected the five consecutive unconditional
results that form the existing pre-terminal cleanup stage. They run
InstanceNorm layout cleanup, affine/Concat/Split recovery summary, pre-Add
cleanup, channel Slice/Pad/Mul summary, and affine-tail cleanup. Each current
context variable is an identity alias of `shared_model_ir_pass_context`; all
five results are unconsumed and every child is pass-module-owned.

The cluster begins immediately after the optional late-binary reconciliation
guard and ends immediately before the intentionally repeated terminal affine
summary. This excludes that terminal rerun because its comment and position
define a separate post-pre-terminal boundary.

`tests/test_flatbuffer_direct_pre_terminal_cleanup_orchestration.py` fixes the
five child owners and current context arguments, exact adjacency, both outer
boundaries, absence of consumers, and empty-graph nested result schemas. Its
strict expected failure requires one shared-context owner returning all five
raw result objects as an ordered tuple and one replacement lowerer result.

Sequential characterization under core-only `uv` completed with
`1 passed, 1 xfailed in 0.55s` focused and
`338 passed, 1 xfailed in 17.67s` across the five child families, optional
late-binary recovery, shared-context, architecture, and phase-store contracts.
The sole expected failure is the intentionally absent composite. Ruff and
whitespace checks passed.

No production callback, nested schema, context identity, guard, graph
mutation, pass order, API, artifact, dependency, TensorFlow boundary, or store
entry changed. No real-model conversion was run; the phase-result store
remains exactly 128 IDs and 128 owners.

## Pre-terminal cleanup composite implementation

`passes/pre_terminal_cleanup_orchestration.py` now owns the characterized
five-stage sequence through one `ModelIRPassContext`. It invokes InstanceNorm
layout cleanup, affine/Concat/Split recovery, pre-Add cleanup, channel
Slice/Pad/Mul recovery, and affine-tail cleanup in their original order. The
exact same context object is passed to every child, and the five original raw
objects are returned as one ordered tuple without flattening, normalization,
or copying. This preserves the nested tuple and mapping schemas already
produced by the specialized child owners.

The lowerer now keeps a single `_pre_terminal_cleanup_results` local instead
of five unconsumed locals. The optional late-binary reconciliation guard still
immediately precedes the stage. The deliberately separate terminal affine
recovery rerun still immediately follows it and continues to feed terminal
Slice/Pad/Concat recovery. Existing lowerer compatibility helpers, child
owners, callbacks, independent invocation paths, pass IDs, and phase-result
recording remain unchanged.

The characterization tests were made owner-aware instead of weakening their
contracts: they inspect the new composite for the four moved child summaries
and cleanup calls, while continuing to inspect the lowerer for the separate
terminal rerun. Runtime callback injection proves exact five-stage order,
context identity, raw-result identity, and outer tuple order. Empty-graph
coverage fixes all nested result schemas.

Final sequential validation under core-only `uv`:

- focused composite boundary, schema, and runtime identity: `3 passed`;
- affected pre-terminal, optional late-binary, shared-context, architecture,
  and phase-store contracts: `340 passed in 18.10s`;
- focused owner-aware stale-boundary set: `293 passed in 19.13s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.71s`;
- synthetic core runtime contracts: `55 passed in 0.93s`;
- result contracts: `196 passed in 9.07s`;
- phase-store capacity contracts: `2 passed in 0.52s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.54s`.

Ruff and whitespace checks passed. No phase result, graph mutation, callback,
guard, public API, artifact, dependency, TensorFlow boundary, or result schema
changed. The store remains exactly 128 IDs and 128 owners. No real-model
conversion was repeated because this checkpoint is a straight-line ownership
move and the dedicated runtime test proves all arguments, state identities,
raw results, and boundaries.

## Late reshape/shuffle/attention/window composite characterization

The post-removal inventory selected four consecutive unconditional late-layout
results: reshape cleanup, the base NCHW channel-shuffle/Gather policy,
attention cleanup, and window partition/reverse cleanup. The first, third, and
fourth calls already receive `shared_model_ir_pass_context`; the channel
shuffle wrapper delegates through an identity alias of that same context with
`include_two_way_shuffle=False` and `include_nhwc_shuffle=False`. All four
results are observation-only and unconsumed.

The cluster starts immediately after the optional late Concat elementwise-
fanout guard and ends immediately before indexed final shape/activation
convergence. The guarded full channel-shuffle/Gather route remains independent
and retains `include_post_gather_cleanup=True`.

`tests/test_flatbuffer_direct_late_reshape_shuffle_attention_window_orchestration.py`
fixes the four child owners, exact base-only policy, source adjacency, both
outer boundaries, absence of consumers, and the empty-graph nested result
schema `(3, 2, 4, 2)`. Its strict expected failure requires one shared-context
owner returning the four raw tuples in order and one replacement lowerer
result.

An inherited layout-recovery test was also corrected to resolve the effective
owner inside `session.record_phase_result(...)`. Its previous direct-call-only
AST expectation was stale and failed before any production change; the new
helper preserves the same expected owner names for both direct and recorded
calls.

Sequential characterization under core-only `uv` completed with
`2 passed, 1 xfailed in 0.60s` focused and
`403 passed, 1 xfailed in 19.20s` across the four child families, channel
shuffle policies, terminal boundaries, callback composition, layout recovery,
shared-context, architecture, result, and phase-store contracts. The sole
expected failure is the intentionally absent composite. Ruff and whitespace
checks passed.

No production callback, graph mutation, result schema, context identity, pass
order, public API, artifact, dependency, TensorFlow boundary, or store entry
changed. No real-model conversion was run; the phase-result store remains
exactly 128 IDs and 128 owners.

## Late reshape/shuffle/attention/window composite implementation

`passes/late_reshape_shuffle_attention_window_orchestration.py` now owns the
characterized four-stage late-layout sequence. It passes the exact same
`ModelIRPassContext` object to reshape cleanup, base-only channel
shuffle/Gather, attention cleanup, and window cleanup in their original order.
The base channel policy still sets `include_two_way_shuffle=False` and
`include_nhwc_shuffle=False`. All four raw result tuples are returned unchanged
inside one ordered outer tuple, preserving child identities and nested schemas.

The lowerer replaces four unconsumed locals with
`_late_reshape_shuffle_attention_window_results` and removes the three
now-unused direct child imports. The generic channel-shuffle wrapper remains:
its guarded full-policy caller still requests post-Gather cleanup, and its
argument-free callback remains wired into layout recovery. The optional late
Concat elementwise-fanout guard remains the predecessor, and indexed final
shape/activation convergence remains the successor.

Existing child-family contracts now inspect their specialized owner for pass
internals and the new composite for top-level ownership. Channel-policy tests
count the guarded wrapper and base composite independently. Runtime callback
injection proves exact four-stage order, shared-context identity, raw-result
identity, and exact keyword policy.

Final sequential validation under core-only `uv`:

- focused composite boundary, schema, policy, and runtime identity:
  `4 passed in 0.58s`;
- focused owner-aware child, terminal-boundary, and architecture coverage:
  `356 passed in 20.24s`;
- complete affected suite: `405 passed in 18.74s`;
- terminal-layout and pass-efficiency contracts: `92 passed in 1.73s`;
- synthetic core runtime contracts: `55 passed in 0.94s`;
- result contracts: `196 passed in 8.91s`;
- phase-store capacity contracts: `2 passed in 0.53s`;
- TensorFlow/tf-keras import blocking, default/direct conversion, and `-cotof`
  contracts: `11 passed in 9.57s`.

Ruff, Python bytecode compilation, and whitespace checks passed. No phase
result, graph mutation, callback, guard, public API, artifact, dependency,
TensorFlow boundary, or result schema changed. The store remains exactly 128
IDs and 128 owners. No real-model conversion was repeated because this is a
straight-line ownership move and runtime injection proves all state, result,
policy, order, and boundary contracts.

## Final boundary/Slice/Concat composite characterization

The refreshed inventory selected four consecutive unconditional final-layout
results immediately after indexed final shape/activation convergence: boundary
channel cleanup, terminal Slice/Concat recovery, final Slice/pre-Concat
cleanup, and terminal Concat bridge cleanup. All results are observation-only
and unconsumed.

The three layout children use the existing shared `ModelIRPassContext`. The
Slice/Concat child uses the existing `TerminalSliceConcatRecoveryContext`,
whose `pass_context` is that exact shared object and whose callback preserves
the channel Slice/Pad/Mul compatibility route. The earlier terminal
Slice/Concat invocation remains independent. The cluster ends before the
optional terminal elementwise-fanout guard.

`tests/test_flatbuffer_direct_final_boundary_slice_concat_orchestration.py`
fixes the four child owners, exact shared/custom context arguments, source
adjacency, both outer boundaries, the independent wrapper route, absence of
consumers, and the empty-graph nested result schema `(3, 14, 2, 6)`. Its strict
expected failure requires one custom-context owner returning all four raw
tuples in order and one replacement lowerer result.

Sequential characterization under core-only `uv` completed with
`2 passed, 1 xfailed in 0.60s` focused and
`398 passed, 1 xfailed in 19.40s` across the four child families, callback
composition, terminal layout, affected result contracts, shared-context,
architecture, and phase-store contracts. The sole expected failure is the
intentionally absent composite. Ruff and whitespace checks passed.

No production callback, graph mutation, nested schema, context identity, pass
order, public API, artifact, dependency, TensorFlow boundary, or store entry
changed. No real-model conversion was run; the phase-result store remains
exactly 128 IDs and 128 owners.

## Very-late layout-tail composite implementation

`passes/very_late_layout_tail_orchestration.py` now owns the characterized
four-stage tail after late Swish cleanup. It forwards the exact shared
`ModelIRPassContext` object to Conv1D/decoder cleanup, Pad/InstanceNorm
cleanup, singleton/consecutive Reshape cleanup, and layout/broadcast cleanup
in the original order. The option-dependent layout-Transpose policy is
forwarded unchanged through the keyword-only `include_layout_transpose`
argument.

The lowerer replaces four observation-only result locals with the single
`_very_late_layout_tail_results` tuple and removes only the three imports made
redundant by that ownership move. Every child returns its original raw tuple;
the new owner preserves those tuple objects and their nested empty-model
schema lengths `(8, 4, 3, 2)`. The late-Swish result remains the immediate
predecessor and phase-recorded static-shape reconciliation remains the
immediate successor.

The lowerer-local singleton compatibility wrapper remains available for the
independent fallback path. Its `fallback_ir` invocation still supplies no
`LayoutState`, while the primary route now calls the pass-module owner with
the shared context directly. Existing specialized owners, callback policies,
pass IDs, graph mutations, public APIs, artifacts, dependency boundaries, and
TensorFlow-free default direct/`-cotof` behavior are unchanged.

Owner-aware structural tests were updated to distinguish the new top-level
owner from its specialized child owners. Runtime callback injection proves
the four-stage order, shared-context identity, result identity, and exact
broadcast flag. Sequential validation under `uv` completed with 437 affected
tests, 92 terminal-layout/efficiency tests, 55 core tests, 196 result-contract
tests, 2 phase-store tests, and 11 TensorFlow-isolation/default-direct/`-cotof`
tests all passing. The phase-result store remains exactly 128 IDs and 128
owners. No real-model conversion was repeated because the change is a
straight-line ownership extraction with explicit state, boundary, schema,
fallback, and result-identity coverage.

## Terminal affine/Slice-SPP composite characterization

The post-removal inventory selected the next three adjacent unconditional
observation results after pre-terminal cleanup: the prune-aware terminal
affine/Concat/Split summary, strict StridedSlice/Pad/Concat bridge cleanup, and
late SPP/Concat/Unary summary. Both summary contexts are identity aliases of
`shared_model_ir_pass_context`; the bridge receives that context's `model_ir`.
The existing pre-terminal composite and the following QKV shape-extract
cleanup define the fixed outer boundaries.

`tests/test_flatbuffer_direct_terminal_affine_slice_spp_orchestration.py`
fixes exact source adjacency, child order, current raw arguments, shared
context aliases, absence of consumers, and the complete empty-model mapping
schemas with lengths `(13, 1, 2)`. Its strict expected failure requires one
shared-context owner that invokes the two summary owners with `context`, the
bridge owner with `context.model_ir`, returns all three raw mappings in order,
and replaces the three lowerer locals with one composite result.

The affected inventory also exposed two stale structural expectations in the
StridedSlice mutation-evidence suite. They still searched for child locals
removed by the already-committed pre-terminal composite and for a direct
post-Add call already owned by three orchestration modules. The corrected
tests resolve the existing pre-terminal and terminal Slice/Concat owners while
preserving the same total mutation-route counts; no production behavior was
changed.

Sequential characterization under core-only `uv` completed with
`2 passed, 1 xfailed in 0.57s` focused and
`518 passed, 1 xfailed in 19.98s` across terminal affine, StridedSlice bridge,
late SPP, pre-terminal ownership, QKV boundary, shared-context, architecture,
terminal validation, and phase-store contracts. The sole expected failure is
the intentionally absent composite. Ruff and whitespace checks passed.

No production callback, graph mutation, result schema, context identity, pass
order, public API, artifact, dependency, TensorFlow boundary, or phase-store
entry changed. No real-model conversion was run; the store remains exactly
128 IDs and 128 owners.
