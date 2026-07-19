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
