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

### Late composite orchestration owners

Nineteen late lowerer clusters now have focused orchestration owners. The first
combines adjacent NDHWC gate and cost-volume ScatterND cleanup into the final
bounded phase result while sharing one short-lived pass state. The second runs
four late Concat/layout owners with one internal state scope and returns their
independent mappings as an ordered composite tuple outside the full phase
store.

The third runs the adjacent ExpandDims-compatible, Flatten-HW-compatible, and
NHWC-collapse reshape repairs and returns their independent mappings as an
ordered tuple outside the store. It preserves the shared layout argument for
the first two repairs and the model-only contract of the third.

The fourth runs the adjacent QKV reshape, attention-Gather cleanup, axis-0
Gather reshape, and pre-projection rank-lift repairs. It preserves their
layout/model/layout/model argument policy and returns the four mappings as an
ordered tuple outside the store.

The fifth runs the adjacent window-partition and window-reverse repairs with
the same conversion-local layout state and returns their two mappings as an
ordered tuple outside the store.

The sixth runs final boundary-input normalization, internal channel-slice
propagation, and the channel-slice Mul/Add bridge. It preserves the final-only
model-only policy for the latter two calls and returns all three mappings as an
ordered tuple outside the store.

The seventh runs the adjacent all-output ReLU/Split,
ReLU/Split/Conv/Concat, mixed Split/Concat, Concat-input adapter,
Concat-unary-Conv, and Shape-extract repairs. It preserves the exact
layout/layout/layout/layout/(layout+diagnostics)/model-only argument policy and
returns all six mappings as an ordered tuple outside the store.

The eighth runs final Slice/pre-post passthrough followed by final pre-ConCat
NHWC cleanup. It preserves the model-only/(layout+diagnostics) argument policy
and returns both independent mappings as an ordered tuple outside the store.

The ninth runs eight adjacent late Conv1D and decoder-tail repairs with one
shared ModelIR/LayoutState context. It returns every independent counter
mapping in source order while preserving all indexed pass owners and public
compatibility wrappers.

The tenth runs very-late Pad cleanup followed by the post-bias,
residual-Mul/Concat, and dual-stat InstanceNorm layout repairs. It preserves
Pad's layout-and-diagnostics contract, the three InstanceNorm owners'
layout-only contracts, and the singleton/consecutive-Reshape successor. Its
four independent mappings remain an ordered composite outside the full phase
store.

The eleventh preserves the normalized option guard around very-late
layout-Transpose cleanup and then runs rank-four channelwise
broadcast-constant repair unconditionally. It returns `None` for the skipped
optional result, retains the broadcast mapping on every path, and leaves the
unconditional broadcast shape reconciliation immediately afterward.

The twelfth runs four shared-late sanitizers, the indexed binary adapter pair,
and singleton/consecutive-Reshape triple, then reduces all nine mutation
mappings plus prune-only tensor-count change to one boolean. The lowerer keeps
the conditional reconciliation record itself, preserving its direct owner
call and invoked-phase-only store semantics.

The thirteenth runs static-signature sanitization followed by the indexed
binary adapter pair, then reduces their three exact mutation counters plus
prune-only tensor-count change to one boolean. The lowerer again retains the
conditional reconciliation record, and both guards around the following
optional late-binary layout recovery remain unchanged.

The fourteenth preserves the normalized enablement predicate around aggregate
late-binary layout recovery and reduces the aggregate mapping to one boolean.
Disabled recovery is still skipped completely; enabled recovery receives the
same ModelIR, LayoutState, diagnostics, and independent layout-Transpose flag.
The lowerer retains the direct conditional reconciliation record.

The fifteenth runs the pre-terminal post-bias, residual-Mul/Concat, and
dual-stat InstanceNorm layout repairs with one shared ModelIR/LayoutState
context. It returns all three independent mappings in source order outside the
full store and leaves both neighboring decision boundaries untouched.

The sixteenth owns the prune-aware summary around terminal-affine recovery at
both late call sites. It snapshots tensor count, invokes the existing raw
eleven-pass owner, and reuses the strict summary schema. The raw lowerer wrapper
remains available as a compatibility boundary.

The seventeenth owns the immediately following pre-terminal pre-add cleanup.
It snapshots tensor count, invokes the existing pre-add NHWC-chain pass once,
and returns the original mapping extended with the same non-negative
`pruned_unused_tensors` delta. The lowerer result target and both neighboring
boundaries remain unchanged, while the lowerer-local snapshot and inline
mapping construction are removed.

The eighteenth composes the adjacent channel Slice/Pad/Mul raw cluster with
its strict normalized summary at the direct late site. The raw lowerer wrapper
remains available to terminal recovery callback composition, while one
consumed raw-result local is removed.

The nineteenth runs affine post-Add cleanup followed by strict
StridedSlice/Pad/Concat cleanup as one ordered pre-terminal tail. It preserves
the layout-aware/model-only argument policy and returns both mappings outside
the full store.

These extractions preserve callback order, model/layout/diagnostics identity,
and result schemas while removing forty-five former unconsumed locals and two
lowerer scope locals. They also replace sixteen consumed mutation-evidence or
aggregate-result locals and five tensor-count snapshots with three explicit
boolean decisions, three reusable summary calls, and one prune-aware cleanup
call.
Focused runtime tests verify shared scope identity, exact argument policy,
ordered results, every positive-evidence path, and prune-only cleanup.

### Shared pre-Concat NHWC composite owner

The three-stage pre-ConCat NHWC composite now lives in
`passes/pre_concat_nhwc_layout.py`. It runs indexed cleanup, quantized indexed
cleanup, and the legacy fallback in the original order, forwards layout state
and diagnostics only to the first two stages, and produces the same bounded
aggregate counter from the same recognized detail keys.

The existing lowerer function remains as a one-return compatibility wrapper,
so its three direct production uses, recovery-orchestration callback, public
test imports, arguments, and result schema are unchanged. The legacy wrapper
also remains available. The pass module imports existing owners directly and
does not depend on the lowerer or callback injection.

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

### 128 stable phase IDs

The lowerer now records 128 bounded observations covering:

- nine unconditional core cleanup results covering pseudo-LeakyReLU, YOLO
  decode, consecutive Mul, terminal Dequantize/QDQ, Conv affine/activation,
  Squeeze/Reshape, and indexed prune/reconcile cleanup;
- four unconditional terminal cleanup results covering terminal Dequantize/QDQ
  and Conv affine/activation cleanup;
- guarded layout pass-set 2 Squeeze/Reshape and indexed prune/reconcile cleanup;
- guarded layout pass-set 1 InstanceNorm pre/post and Squeeze/Reshape cleanup;
- guarded layout pass-set 1 quantized PReLU, TransposeConv, and Reshape cleanup;
- guarded layout pass-set 1 affine-chain fold, affine pre/post, pre-unary
  affine fan-out, and mean-affine pre/post cleanup;
- guarded layout pass-set 1 layout-Transpose, Transpose/binary bridge,
  duplicate fan-out, and Dequantize→Mean→Quantize cleanup;
- guarded layout pass-set 2 quantized TransposeConv cleanup;
- guarded layout pass-set 2 elementwise/Concat/SPP, input-adapter,
  Slice/Logistic-tail, and SA/PA MirrorPad cleanup;
- unconditional terminal ArgMax, Gather fan-out, Softmax, boundary-input,
  channel-slice, and channel-slice Mul/Add bridge cleanup;
- unconditional terminal boundary StridedSlice/QDQ/Concat, Swish
  residual/Concat, Dequantize/Logistic/Mul/Quantize, and Swish QDQ-island
  cleanup;
- unconditional terminal InstanceNorm post-bias, normalization Pad,
  InstanceNorm residual Add, InstanceNorm residual Mul/Concat, and
  InstanceNorm dual-stat cleanup;
- guarded terminal BatchMatMul affine-input, Reshape/SE, and adjoint-flag
  cleanup between the retained Mean- and QKV-attention composites;
- guarded terminal QKV Split/Conv/Concat bridge cleanup between the retained
  QKV-attention and singleton-reshape composites;
- unconditional terminal SiNet HardSwish-SE and
  Dequantize/HardSigmoid/Quantize bridge cleanup between retained SiNet
  recovery composites;
- post-terminal indexed shape/topology convergence between the singleton and
  very-late SiNet composites;
- very-late residual affine PReLU, residual Transpose fan-out, and indexed
  prune/reconcile cleanup between retained SiNet composites;
- post-cleanup CSP attention and SA/PA MirrorPad cleanup between the retained
  SiNet pre-Add/Resize composite and post-SiNet BatchMatMul observations;
- post-SiNet BatchMatMul affine-input, Reshape/SE, and adjoint-flag cleanup
  before the retained QKV-attention composite;
- post-SiNet all-output ReLU/Split, Split/Conv/ReLU/Concat, and
  Split/Conv/Concat bridge cleanup after the retained QKV composite;
- post-SiNet mix-attention, mixed-attention layout, and
  Dequantize/HardSigmoid bridge cleanup before shared late pass-state scopes;
- one aggregated late NDHWC gate/cost-volume ScatterND phase whose internal
  owner shares a short-lived pass state and returns three integer counters;
- core shape resolution;
- safe no-layout Transpose reduction;
- terminal static-shape reconciliation;
- fallback and primary topology checkpoints;
- fallback and primary topology/layout refresh;
- primary final ConvInteger, InstanceNorm, and broadcast reconciliation before
  their matching topology/layout refresh;
- primary final PReLU and consecutive-Reshape reconciliation;
- primary final mixed-singleton Concat, nested placeholder/binary, and
  SE/FC/Gather reconciliation;
- the ordered six-boundary primary final SiNet reconciliation chain;
- two unconditional very-late reconciliation boundaries;
- guarded shared-late reconciliation over nine mutation-evidence sources and
  cleanup-only pruning;
- guarded late binary repair and nested layout-recovery reconciliation;
- guarded post-split fallback reconciliation before the unbound-input safety
  check;
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
allowed 30 unconsumed all-zero default dictionaries to be removed.

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

The latest thirty-one records cover terminal boundary StridedSlice/QDQ/Concat,
activation bridge, InstanceNorm, normalization, and guarded BatchMatMul
and QKV bridge plus SiNet HardSwish/HardSigmoid and indexed convergence
and very-late residual, post-cleanup attention, plus post-SiNet BatchMatMul
and ReLU/Split plus attention/activation observations. They retain
deterministic order and original boundaries.

Structural tests also ensure that:

- raw duplicated operation pairs no longer remain at migrated sites;
- all 128 phase IDs and owners appear in deterministic source order;
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
- direct generic-final, terminal, SE/FC/Gather, and store contracts:
  **79 passed**;
- direct final-SiNet, terminal, and store contracts: **67 passed**;
- direct unconditional, terminal, very-late, and store contracts:
  **90 passed**;
- focused late-binary, terminal, and bounded-store contracts:
  **72 passed**;
- focused shared-late, late-binary, terminal, runtime, and bounded-store
  contracts: **75 passed**;
- focused post-split, very-late, Split fallback, terminal, and bounded-store
  contracts: **96 passed**;
- direct core-cleanup, phase-store, dynamic-Reshape, Squeeze/Reshape, indexed
  prune/reconcile, terminal, and architecture-boundary contracts:
  **76 passed**;
- synthetic core runtime contracts: **55 passed**;
- direct terminal-cleanup, phase-store, terminal orchestration, and indexed-
  owner architecture-boundary contracts: **68 passed**;
- direct layout pass-set 2 cleanup and owner contracts: **9 passed**;
- direct layout pass-set 1 cleanup and owner contracts: **9 passed**;
- direct layout pass-set 1 quantized cleanup and owner contracts:
  **10 passed**;
- direct layout pass-set 2 quantized cleanup and owner contracts:
  **8 passed**;
- direct layout pass-set 1 affine cleanup and owner contracts:
  **13 passed**;
- direct residual layout pass-set 1 cleanup and owner contracts:
  **14 passed**;
- expanded residual layout pass-set 2 and orchestration contracts:
  **64 passed**;
- terminal boundary, owner, phase-store, and runtime contracts:
  **156 passed**;
- QLinear and terminal-layout orchestration contracts: **71 passed**;
- terminal activation, phase-store, owner, and Slice/Concat contracts:
  **75 passed**;
- terminal normalization, phase-store, owner, and boundary contracts:
  **100 passed**;
- terminal BatchMatMul, QKV, and phase-store contracts: **26 passed**;
- indexed QKV bridge, QKV, singleton, and phase-store contracts:
  **106 passed**;
- terminal HardSwish, HardSigmoid, SiNet, and phase-store contracts:
  **23 passed**;
- indexed convergence, SiNet, and phase-store contracts: **12 passed**;
- very-late residual, prune/reconcile, SiNet, and phase-store contracts:
  **20 passed**;
- post-cleanup CSP/MirrorPad boundary and phase-store contracts:
  **17 passed**;
- focused CSP/MirrorPad runtime and orchestration contracts: **68 passed**;
- post-SiNet BatchMatMul/QKV/attention/store contracts: **31 passed**;
- post-SiNet ReLU/Split/QKV/mix-attention/store contracts: **75 passed**;
- post-SiNet attention/activation/state-scope/store contracts: **17 passed**;
- late NDHWC/cost-volume pair, gate, store, and architecture contracts:
  **20 passed**;
- late Concat composite, owner, terminal-layout, and architecture contracts:
  **76 passed**;
- late reshape-layout composite and affected owner contracts:
  **95 passed**;
- late attention-layout composite and affected owner contracts:
  **238 passed**;
- late window-layout composite and affected owner contracts:
  **110 passed**;
- final boundary-channel composite and affected owner contracts:
  **79 passed**;
- terminal Concat-bridge composite and affected result contracts:
  **17 passed**;
- final Slice/pre-ConCat composite and affected boundary contracts:
  **20 passed**;
- Slice/pre-post mutation contracts: **9 passed**;
- late Conv1D/decoder composite contracts: **3 passed**;
- indexed Conv1D/decoder and affected result contracts: **431 passed**;
- very-late Pad/InstanceNorm composite and affected boundary contracts:
  **424 passed**;
- very-late layout/broadcast composite and affected boundary contracts:
  **97 passed**;
- shared-late reconciliation decision and affected boundary contracts:
  **150 passed**;
- late-binary repair decision and affected boundary contracts:
  **132 passed**;
- optional late-binary layout-recovery decision and affected contracts:
  **138 passed**;
- pre-terminal InstanceNorm composite and affected contracts:
  **151 passed**;
- terminal-affine prune-aware summary and affected contracts:
  **182 passed**;
- pre-terminal pre-add prune-aware owner and boundary contracts:
  **4 passed**;
- affected pre-add, channel Slice/Pad/Mul, terminal-affine, and related
  contracts: **186 passed**;
- channel Slice/Pad/Mul direct-summary owner contracts: **3 passed**;
- affected channel Slice/Pad/Mul, pre-add, terminal recovery, and related
  contracts: **195 passed**;
- pre-terminal affine-tail owner contracts: **3 passed**;
- affected affine-tail, terminal recovery, very-late, and related contracts:
  **237 passed**;
- TensorFlow/tf-keras import blocker, default/direct conversion, and `-cotof`
  contracts: **11 passed**;
- pre-Concat NHWC pass-owner and compatibility contracts: **3 passed**;
- indexed, quantized, and legacy NHWC Concat family contracts:
  **285 passed**;
- broader result and phase-result contracts: **196 passed**;
- broader phase-store, owner, fallback, terminal, shape, and topology suite:
  **275 passed**;
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

The latest checkpoint implements the characterized pre-terminal pre-add owner.
The existing pass call, tensor-count delta, source order, ModelIR/LayoutState
identity, and unconsumed result remain intact; only their orchestration moves
behind a focused pass-module boundary. Runtime tests cover both stable and
prune-only paths, and the already-full 128/128 phase-result store is unchanged.

The following checkpoint implements the adjacent direct channel Slice/Pad/Mul
raw-to-summary owner. It removes only the consumed direct raw-result local and
explicitly preserves the existing raw wrapper for terminal recovery callback
composition.

The latest checkpoint implements the adjacent affine post-Add and strict
StridedSlice/Pad/Concat pair as one ordered owner. It preserves the
layout-aware/model-only argument policy, all other call sites, and the full
128/128 store.
