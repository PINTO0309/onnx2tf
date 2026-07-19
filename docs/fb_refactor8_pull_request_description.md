# Make `flatbuffer_direct` phase evidence bounded, explicit, and reusable

## Summary

This branch continues the staged, characterize-first refactoring of the
TensorFlow-free `flatbuffer_direct` backend. It does not change conversion
policy, public interfaces, artifacts, pass order, or numerical behavior.
Instead, it makes several important post-lowering boundaries explicit and
consolidates their small observation results in a bounded conversion-session
store.

The branch improves the pipeline in four related ways:

1. previously discarded pass results are retained under stable phase IDs;
2. repeated shape/topology and topology/layout sequences have focused owners;
3. terminal topology/layout validation has one shared invariant boundary;
4. unconsumed lowerer-local result dictionaries are replaced by bounded,
   conversion-local phase evidence.

These changes reduce implicit state in the central lowerer and provide a safe
foundation for future scan-elimination work. This branch deliberately does
not use the new counters to skip any graph traversal; that requires separate
differential characterization.

## Motivation

The direct FlatBuffer lowerer contains many interacting shape, layout,
topology, fallback, and compatibility passes. Several calls returned useful
mutation or validation evidence that was either discarded or stored in local
variables with no consumer. Other boundaries repeated the same adjacent
operations directly in the lowerer.

That structure made later optimization risky:

- a discarded result could not show whether a phase actually changed the
  graph;
- an unconsumed local increased lowerer state without establishing a durable
  phase identity;
- guard-skipped work could be confused with work that ran and returned zero;
- duplicated adjacent operations could drift apart during maintenance;
- terminal invariants were harder to review because ownership was spread
  across the lowerer.

`fb-refactor8` addresses these issues without adding scans, changing guards,
or altering ModelIR mutations.

## Detailed changes

### Complete observation at existing boundaries

The branch first characterized and retained results from three independent
single-call boundaries:

- core Dynamic Reshape resolution;
- safe Transpose reduction in the no-layout path;
- terminal Expand/Squeeze static-shape reconciliation.

It also characterized the fallback norm reconciliation boundary. The complete
static-shape mutation schema is used where needed, so option, constant, and
tensor-metadata updates remain observable rather than relying only on a legacy
shape counter.

These calls remain at the same locations with the same arguments and guards.
No result is used as a control-flow input in this branch.

### Shared topology/layout refresh owner

`run_topology_layout_refresh(model_ir)` owns the repeated sequence that:

1. topologically sorts ModelIR operators;
2. refreshes logical layout annotations;
3. releases the full temporary layout map;
4. returns only bounded integer counters.

Six fallback and primary boundaries use this owner. The extracted function
preserves cycle behavior, operator order, layout updates, and temporary-map
lifetime while removing duplicated orchestration from the lowerer.

### Shared static-shape/topology reconciliation owner

`run_static_shape_topology_reconciliation(model_ir)` owns the repeated
sequence that performs complete static-shape reconciliation and then restores
producer-before-consumer order. It returns four integer counters:

- `reconciled_static_tensor_shapes`;
- `reconciled_static_shape_mutations`;
- `reordered_operators`;
- `cycle_detected`.

Eight fallback and primary repair boundaries use this owner. Their existing
guards and phase ordering are unchanged.

### Shared terminal topology/layout validation owner

`run_topology_layout_validation(model_ir)` now owns the fallback and primary
terminal invariant boundary:

1. topologically sort operators;
2. validate logical-layout annotations;
3. set or clear `logical_layout_validation_errors` in ModelIR metadata;
4. return compact validation counters.

Full validation strings remain only in ModelIR metadata. The returned mapping
contains bounded integer evidence, avoiding duplicate retention of diagnostic
text. Cycle behavior and stale-error removal are covered explicitly.

### Explicit topology checkpoints

Five existing direct topological-sort calls now have stable phase identities:

- fallback after placeholder restoration;
- fallback after late layout repair;
- primary post-lowering baseline;
- primary no-layout post-reduction;
- primary final placeholder restoration.

The calls remain distinct because intervening repair families can mutate
topology. This branch does not merge, guard, or remove any of them.

### Bounded `ConversionSession` phase-result store

`ConversionSession` now provides two internal methods:

- `record_phase_result(phase_id, counters)`;
- `phase_results_snapshot()`.

The store is intentionally small and defensive:

- at most 128 phase IDs;
- at most 32 counters per phase;
- integer values only, normalized to built-in `int`;
- copied input mappings;
- isolated snapshots;
- conversion-session lifetime only.

This store is separate from `session.diagnostics`. The diagnostics stream has
an existing private metrics contract in which events represent ModelIR pass
execution. Mixing observation counters into that stream would change event
numbering and report semantics.

The phase store is not written to ModelIR metadata and is not exposed through
the public API, conversion result, reports, or generated artifacts.

### Thirty-six stable phase IDs

The lowerer now records 36 bounded observations covering:

- core shape resolution;
- safe no-layout Transpose reduction;
- terminal static-shape reconciliation;
- fallback and primary topology checkpoints;
- fallback and primary topology/layout refresh;
- primary final ConvInteger, InstanceNorm, and broadcast reconciliation before
  their matching topology/layout refresh;
- primary final PReLU and consecutive-Reshape reconciliation;
- fallback and primary terminal layout validation;
- fallback broadcast, SE/FC/Gather, placeholder-MatMul, Conv-input,
  mixed-Concat, Concat-axis, and binary-layout static-shape reconciliation;
- fallback norm and high-rank BatchMatMul shape/topology reconciliation;
- primary final high-rank BatchMatMul, Pad, Conv-input, mixed-Concat,
  Concat-axis, and binary-layout shape/topology reconciliation.

The guarded shape-reconciliation and shape/topology phases use
invoked-phase-only semantics. A phase omitted by its guard is absent from the
snapshot. An invoked phase is recorded even when all counters are zero. This
preserves the distinction between "not invoked" and "invoked but stable" and
allowed 20 unconsumed all-zero default dictionaries to be removed.

## Safety and compatibility

- Public CLI and Python APIs are unchanged.
- `flatbuffer_direct` remains the default backend.
- Artifact names, report formats, return behavior, and exceptions are
  unchanged.
- All affected owner calls keep their original arguments, guards, evaluation
  count, predecessors, and successors.
- No graph traversal, reconciliation, sort, validation, or layout inference
  was added or removed.
- No stored counter is used to change control flow.
- Normal direct TFLite conversion and `-cotof` remain independent of
  TensorFlow.
- Optional TensorFlow exporters remain behind the existing optional boundary.
- No dependency was added.
- No multiprocessing or parallel inference behavior was introduced.

## Characterize-first implementation strategy

Each owner extraction and observation migration was preceded by a focused
contract that fixed the relevant schema, graph effects, cycle behavior,
metadata behavior, phase position, arguments, and no-op behavior. Production
changes were then limited to the characterized boundary.

Structural tests also ensure that:

- raw duplicated operation pairs no longer remain at migrated sites;
- all 36 phase IDs and owners appear in deterministic source order;
- old unconsumed result targets are absent from the lowerer;
- the bounded store does not alias caller mappings or snapshots;
- diagnostics and public output contracts remain independent of the store.

## Validation

All validation was executed sequentially under `uv`.

Final checkpoint results:

- fallback static-shape family and safety-fallback contracts:
  **20 passed**;
- direct primary final-layout family, terminal, refresh, and store contracts:
  **71 passed**;
- direct PReLU/consecutive-Reshape, terminal, and store contracts:
  **67 passed**;
- broader phase-store, owner, fallback, terminal, shape, and topology suite:
  **142 passed**;
- lowerer architecture suite: **258 passed**;
- targeted Ruff checks: **passed**;
- Python bytecode compilation: **passed**;
- whitespace validation: **passed**.

Earlier checkpoints also ran larger focused gates that covered core contracts,
pass efficiency, architecture constraints, and TensorFlow-import blocking.
Their exact commands and results are recorded in
`docs/fb_refactor8_improvements.md` and
`docs/flatbuffer_direct_handoff_fb_refactor8.md`.

No real-model corpus conversion was repeated for these checkpoints because
the implementation only extracts previously adjacent operations or changes
the destination of already-computed bounded dictionaries. The owner-effect
tests and structural gates verify that ModelIR mutations and serialization
inputs remain unchanged.

## Scope intentionally deferred

This branch does not attempt to remove redundant graph scans based on the new
evidence. A future change may consume phase counters only after a separate
differential test proves identical operator order, layout state, cycle
handling, ModelIR digest, and downstream artifacts for both mutation-positive
and stable paths.

The broader multi-phase `flatbuffer_direct` refactor remains ongoing. This
checkpoint supplies more explicit ownership and bounded evidence for that
work without changing current converter behavior.
