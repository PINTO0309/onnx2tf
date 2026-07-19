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

Forty-one late lowerer clusters now have focused orchestration owners. The
first combines adjacent NDHWC gate and cost-volume ScatterND cleanup into the final
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

The twentieth composes the direct late SPP/Concat/Unary raw pair with its
strict two-counter summary. It preserves the raw wrapper and shared pass-state
scope while removing one consumed raw-result local.

The twenty-first owns the late QKV prune-aware summary. It snapshots tensor
count inside the pass module, invokes the existing QKV owner with the runtime
layout-Transpose flag and prefix cleanup disabled, and returns the unchanged
strict summary schema. The two default-policy raw-wrapper uses and the raw
lowerer compatibility wrapper remain unchanged.

The twenty-second owns terminal HardSwish/SE prune-aware evidence. It snapshots
tensor count inside the pass module, invokes the existing raw layout owner, and
extends the unchanged one-key mapping with a non-negative prune delta. The raw
lowerer wrapper and its earlier phase-store call remain unchanged.

The twenty-third composes late hard-activation/layout cleanup with its strict
prune-aware summary. It preserves the runtime layout-Transpose policy, shared
ModelIR/LayoutState/diagnostics context, raw wrapper, and normalized schema
while removing the consumed raw tuple from lowerer scope.

The twenty-fourth composes the late
layout/Mean/SPP/Gather/constant-fold/Cast cluster with its strict prune-aware
summary. It preserves the runtime layout-Transpose policy, shared pass-state
scope, child constant-fold/Cast builder, raw wrapper, and normalized schema.

The twenty-fifth composes very-late
Gather/constant-fold/Cast/normalization cleanup with its strict four-result
prune-aware summary. It preserves the shared pass-state scope, child
constant-fold/Cast builder, normalization policy, raw wrapper, and normalized
schema.

The twenty-sixth provides one reusable prune-aware summary for the compatible
very-late and fallback indexed Conv-input sites. It preserves the shared
single-index pair, exact two-key raw schema, fallback reconciliation guard, and
raw wrapper while explicitly leaving the different final one-key site intact.

The twenty-seventh provides one reusable prune-aware summary for the fallback
and final-primary stale channelwise-binary adapter sites. It preserves their
exact raw schema and mutation-positive reconciliation guards, keeps the raw
wrapper and optional graph-index forwarding available, and explicitly leaves
the iterative indexed convergence owner unchanged.

The twenty-eighth provides a dedicated prune-aware summary for the
final-primary one-repair stale Conv-input site. It preserves the distinct
one-key raw schema, raw wrapper and optional graph-index forwarding, final-Pad
predecessor, reconciliation guard, and mixed-Concat successor while leaving
the indexed two-repair summary sites unchanged.

The twenty-ninth provides a dedicated prune-aware summary for absolute-final
PRELU passthrough cleanup. It preserves exact layout-state forwarding, the raw
wrapper and both other raw PRELU paths, rewrite-or-prune reconciliation
semantics, and both neighboring cleanup boundaries.

The thirtieth provides a dedicated prune-aware summary for the safety-fallback
norm-subgraph Pad cleanup. It preserves the norm-only fixed flags, diagnostics
forwarding, raw Pad compatibility re-export, rewrite-only reconciliation guard,
all other Pad-family routes, and neighboring fallback boundaries.

The thirty-first provides a merged prune-aware summary for the final
placeholder-MatMul indexed binary-adapter pair. It preserves pair order,
optional graph-index/layout-state forwarding, all other raw pair callers, the
preceding placeholder reconciliation mapping, and following topology checkpoint.

The thirty-second provides one shared prune-aware summary for the fallback and
absolute-final SiNet/SE-FC/Gather sequences. It preserves path-specific
ModelIR/LayoutState forwarding, SiNet-before-pair ordering, diagnostics and
shared pair state, all three rewrite counters, prune-only reconciliation, raw
compatibility wrappers, and neighboring boundaries.

The thirty-third owns both duplicated three-stage precision-cleanup sequences.
It preserves DIV-to-reciprocal → consecutive-MUL → sensitive-DIV restore
ordering, keeps all three raw mappings independent, omits layout state for
fallback, forwards it for primary-final, and limits diagnostics to the
transactional middle stage.

The thirty-fourth owns the absolute-final dynamic-boundary realignment and
static-signature sanitizer pair. It preserves both raw schemas, exact mutation
order, every other signature caller, wrapper compatibility, and the following
affine cleanup.

The thirty-fifth owns the guarded no-layout final SE-FC and affine pre/post
cleanup pair. It preserves shared ModelIR/LayoutState identity, SE-FC-only
diagnostics, raw mapping schemas and order, the option guard, both topology
boundaries, every other raw caller, compatibility symbols, and the following
boundary-signature cleanup.

The thirty-sixth owns the adjacent absolute-final affine post-ADD and
decomposed-InstanceNorm post-bias cleanups. It preserves shared
ModelIR/LayoutState identity, affine-before-InstanceNorm order, both raw
schemas, all independent callers, compatibility wrappers, and both neighboring
orchestration boundaries.

The thirty-seventh composes the existing absolute-final normalization/pad and
mixed-attention owner with dynamic rank-one Unsqueeze/Reshape repair. It
preserves the existing inner tuple plus raw rank-one mapping as a nested pair,
shared context identity, callback order, all independent rank-one callers, and
both neighboring boundaries while removing a lowerer-only closure and alias.

The thirty-eighth owns indexed binary-layout convergence. It constructs one
graph index, shares it across broadcast-constant repair, stale binary-adapter
repair, and static-shape reconciliation, preserves that exact order, retains
the three-round cap and stable-stop rule, and returns the unchanged ordered
three-counter mapping. The lowerer compatibility wrapper and both fallback and
primary behaviors remain intact; the fallback still uses the wrapper and the
primary use is composed by the following terminal owner.

The thirty-ninth composes final-primary indexed binary-layout convergence,
static high-rank binary coalescing, and dynamic boundary-signature realignment.
It preserves model/layout argument policy, exact callback order, every raw
mapping object, both raw lowerer compatibility wrappers, the fallback
convergence path, and terminal validation/finalization successors while
replacing three lowerer results with one ordered context-owned tuple.

The fortieth composes the absolute-final boundary-signature,
affine/InstanceNorm, and normalization/attention/rank-one owners. It preserves
the shared context object, exact three-stage order, every nested result object,
all sub-owner and raw-wrapper contracts, the guarded no-layout predecessor,
and topology/layout refresh successor while replacing three lowerer targets
with one ordered outer tuple.

The forty-first composes very-late dynamic Reshape, indexed Conv-input, stale
channel-shuffle, two Concat-axis, and dynamic rank-one repairs. It preserves
the runtime-inferable flag, exact ModelIR/LayoutState/diagnostics policy,
callback order, every raw mapping object, compatibility wrappers, fallback and
independent callers, mandatory reconciliation, and split fallback while
replacing six lowerer results with one context-owned tuple.

These extractions preserve callback order, model/layout/diagnostics identity,
and result schemas while removing seventy-one former unconsumed locals and three
lowerer scope locals. They also replace twenty-nine consumed mutation-evidence
or aggregate-result locals and twenty tensor-count snapshots with three
explicit boolean decisions, nineteen reusable summary calls, and one prune-aware
cleanup call.
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
- late SPP/Concat/Unary direct-summary owner contracts: **3 passed**;
- affected late SPP, shape-extract, terminal recovery, and related contracts:
  **187 passed**;
- late QKV prune-aware summary owner and boundary contracts: **5 passed**;
- affected late QKV, neighboring owners, core, phase-store, and architecture
  contracts: **405 passed**;
- terminal HardSwish/SE prune-aware summary characterization and related
  contracts: **76 passed, 1 intentional strict xfail**;
- terminal HardSwish/SE prune-aware summary owner contracts: **4 passed**;
- affected terminal HardSwish/SE, late hard-activation, indexed bridge, store,
  and architecture contracts: **337 passed**;
- late hard-activation prune-aware summary characterization and related
  contracts: **22 passed, 1 intentional strict xfail**;
- late hard-activation prune-aware summary owner contracts: **5 passed**;
- affected late hard-activation, HardSwish/SE, pre-ConCat, store, and
  architecture contracts: **294 passed**;
- late layout-cluster prune-aware summary characterization and related
  contracts: **21 passed, 1 intentional strict xfail**;
- late layout-cluster prune-aware summary owner contracts: **5 passed**;
- affected late layout-cluster, shape-extract, store, and architecture
  contracts: **283 passed**;
- very-late normalization prune-aware summary characterization and related
  contracts: **40 passed, 1 intentional strict xfail**;
- very-late normalization prune-aware summary owner contracts: **4 passed**;
- affected very-late normalization, adjacent repair, store, and architecture
  contracts: **301 passed**;
- indexed Conv-input prune-aware summary family characterization and related
  contracts: **115 passed, 1 intentional strict xfail**;
- indexed Conv-input shared prune-aware summary owner contracts: **4 passed**;
- affected indexed Conv-input, very-late, fallback, terminal-layout, store, and
  architecture contracts: **376 passed**;
- stale channelwise-binary adapter summary family characterization and related
  contracts: **93 passed, 1 intentional strict xfail**;
- stale channelwise-binary adapter shared-summary contracts: **4 passed**;
- affected stale binary-adapter, fallback, terminal-layout, indexed
  convergence, store, and architecture contracts: **354 passed**;
- final stale Conv-input dedicated-summary characterization and related
  contracts: **336 passed, 1 intentional strict xfail**;
- final stale Conv-input dedicated-summary contracts: **4 passed**;
- affected indexed Conv-input, terminal-layout, store, and architecture
  contracts: **339 passed**;
- final PRELU dedicated-summary characterization and related contracts:
  **389 passed, 1 intentional strict xfail**;
- final PRELU dedicated-summary contracts: **4 passed**;
- affected terminal-layout, SE-FC/Gather, core runtime, store, and architecture
  contracts: **392 passed**;
- fallback norm-subgraph Pad summary characterization and related contracts:
  **300 passed, 1 intentional strict xfail**;
- fallback norm-subgraph Pad dedicated-summary contracts: **4 passed**;
- affected fallback, Pad, norm, singleton-Reshape, store, and architecture
  contracts: **303 passed**;
- final placeholder binary-adapter summary characterization and related
  contracts: **380 passed, 1 intentional strict xfail**;
- final placeholder binary-adapter merged-summary contracts: **4 passed**;
- affected indexed adapter, terminal-layout, core runtime, store, and
  architecture contracts: **383 passed**;
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

The latest checkpoint implements the adjacent late SPP/Concat/Unary raw tuple
and normalized summary owner while explicitly retaining the existing raw
wrapper and shared pass-state scope.

The latest checkpoint implements the late QKV prune-aware summary owner. It
moves the tensor snapshot and raw-to-summary composition behind one pass-module
boundary while retaining both default-policy raw-wrapper uses, the raw lowerer
wrapper, runtime flag forwarding, exact summary schema, and the full 128/128
phase-result store.

The next characterization fixes the terminal HardSwish/SE tensor snapshot,
raw mapping, prune delta, and neighboring pass boundaries. It intentionally
leaves production unchanged until the generic prune-aware summary owner is
implemented in a separate checkpoint.

The latest checkpoint implements that terminal HardSwish/SE summary owner. It
moves only the tensor snapshot and mapping extension into the pass module,
retaining the raw wrapper, earlier phase-store raw call, pass order, exact
mapping schema, neighboring boundaries, and the full 128/128 phase-result
store.

The next characterization fixes the late hard-activation tensor snapshot,
flagged raw ordered result, strict prune-aware summary, and neighboring pass
boundaries. Production remains unchanged until the direct summary owner is
implemented separately.

The latest checkpoint implements the late hard-activation prune-aware summary
owner. It removes the lowerer-local tensor snapshot and consumed raw tuple
while retaining runtime option forwarding, raw wrapper dispatch, strict schema,
neighboring boundaries, and the full 128/128 phase-result store.

The next characterization fixes the late layout-cluster tensor snapshot,
flagged raw ordered result, strict prune-aware summary, and neighboring pass
boundaries. Production remains unchanged until its direct summary owner is
implemented separately.

The latest checkpoint implements that late layout-cluster summary owner. It
removes the lowerer-local tensor snapshot and consumed raw tuple while
retaining runtime option forwarding, raw wrapper dispatch, child-builder and
shared-scope behavior, strict schema, neighboring boundaries, and the full
128/128 phase-result store.

The next characterization fixes the very-late normalization tensor snapshot,
four-result raw tuple, strict prune-aware summary, and neighboring pass
boundaries. Production remains unchanged until its direct summary owner is
implemented separately.

The latest checkpoint implements that very-late normalization summary owner.
It removes the lowerer-local tensor snapshot and consumed raw tuple while
retaining raw wrapper dispatch, child-builder and shared-scope behavior, strict
schema, neighboring repair boundaries, and the full 128/128 phase-result store.

The next characterization fixes the two compatible indexed Conv-input
count-plus-mapping sites and explicitly excludes the final one-key repair site.
Production remains unchanged until the shared prune-aware summary owner is
implemented separately.

The latest checkpoint implements that shared indexed Conv-input summary owner.
It removes both compatible tensor snapshots and inline mapping extensions while
retaining the indexed repair pair, fallback guard, raw wrapper, final one-key
site, and the full 128/128 phase-result store.

The next characterization fixes the two compatible stale channelwise-binary
adapter count-plus-mapping sites. It preserves their concat-axis predecessors,
mutation-positive reconciliation guards, fallback topology successor, final
progress successor, and raw wrapper, while explicitly excluding the iterative
indexed convergence owner. Production remains unchanged until the shared
prune-aware summary owner is implemented separately.

The latest checkpoint implements that shared stale binary-adapter summary
owner. It removes both compatible tensor snapshots and inline mapping
extensions while retaining the raw wrapper, optional graph-index forwarding,
indexed convergence loop, both reconciliation guards, neighboring boundaries,
and the full 128/128 phase-result store.

The next characterization fixes the dedicated final-primary stale Conv-input
count-plus-mapping boundary. It preserves its one-repair schema, final-Pad
predecessor, mutation-positive reconciliation guard, mixed-Concat successor,
and raw wrapper with optional graph-index forwarding. It remains separate from
the indexed two-repair family, and production is unchanged until its dedicated
summary owner is implemented separately.

The latest checkpoint implements that dedicated final stale Conv-input summary
owner. It removes the final tensor snapshot and inline mapping extension while
retaining the raw wrapper, optional graph-index forwarding, both indexed
summary sites, neighboring boundaries, and the full 128/128 phase-result store.

The next characterization fixes the absolute-final PRELU count-plus-result
boundary. It preserves ModelIR/LayoutState forwarding, the preceding
SE-FC/Gather guard, rewrite-or-prune reconciliation semantics, the following
consecutive-Reshape cleanup, and the raw wrapper. Production remains unchanged
until its dedicated prune-aware summary owner is implemented separately.

The latest checkpoint implements that dedicated final PRELU summary owner. It
removes the tensor snapshot, adds bounded prune evidence to the raw mapping,
and uses the existing positive-count predicate while retaining the raw wrapper,
both other PRELU paths, neighboring boundaries, and the full 128/128 store.

The next characterization fixes the safety-fallback norm-only Pad cleanup
count-plus-mapping boundary. It preserves the fixed stage flags, diagnostics
forwarding, conditional norm reconciliation, recursive fallback predecessor,
dynamic rank-one successor, and every other Pad-family caller. Production
remains unchanged until its dedicated prune-aware summary owner is implemented
separately.

The latest checkpoint implements that dedicated fallback norm-subgraph Pad
summary owner. It removes the tensor snapshot and inline mapping extension
while retaining the fixed flags, diagnostics, raw compatibility re-export,
rewrite-only guard, every other Pad route, neighboring boundaries, and the full
128/128 store.

The next characterization fixes the final placeholder-MatMul indexed
binary-adapter count-plus-pair boundary. It preserves pair order, both disjoint
counter schemas, the preceding placeholder reconciliation mapping,
rewrite-or-prune guard, following topology checkpoint, and every other raw pair
caller. Production remains unchanged until its merged prune-aware summary owner
is implemented separately.

The latest checkpoint implements that merged final placeholder binary-adapter
summary owner. It removes the tensor snapshot and two raw-result locals while
retaining optional context forwarding, all other raw pair callers, the
preceding reconciliation mapping, rewrite-or-prune guard, topology successor,
and the full 128/128 store.

The next characterization fixes both compatible SiNet/SE-FC/Gather
count-plus-result sequences. It preserves path-specific ModelIR/LayoutState
forwarding, SiNet-before-pair ordering, rewrite-or-prune reconciliation,
fallback and final neighboring boundaries, the raw pair helper, and the full
128/128 store. Production remains unchanged until a shared pass-module summary
owner is implemented and validated in a separate checkpoint.

The latest checkpoint implements the shared SiNet/SE-FC/Gather summary owner.
It removes two tensor snapshots and six consumed result locals from the two
compatible production sites while preserving path-specific context, ordered
mutations, all rewrite and prune-only reconciliation paths, compatibility
wrappers, neighboring boundaries, and the full 128/128 store.

The next characterization fixes both compatible three-stage precision-cleanup
sequences. It preserves DIV-to-reciprocal → consecutive-MUL → sensitive-DIV
restore ordering, path-specific layout policy, diagnostics forwarding only to
the transactional middle stage, independent result schemas, neighboring
fallback/final boundaries, and the full 128/128 store. Production remains
unchanged until a shared pass-module sequence owner is implemented separately.

The latest checkpoint implements that shared precision-cleanup sequence owner.
It replaces six individual unconsumed result locals with two ordered tuples
while preserving exact raw schemas, callback order, path-specific layout and
diagnostics policy, the independent core consecutive-MUL caller, compatibility
re-exports, neighboring boundaries, and the full 128/128 store.

An inherited shared-late structural test now follows the already-established
late-binary boolean successor instead of the removed tensor-count snapshot.
This is a test-only contract repair with no production change.

The next characterization fixes the adjacent absolute-final dynamic-boundary
realignment and static-signature sanitizer pair. It preserves independent raw
schemas, realign-before-sanitize order, the following affine boundary, all
other signature-owner callers, compatibility wrappers, and the full 128/128
store. Production remains unchanged until an ordered pair owner is implemented
separately.

The latest checkpoint implements that ordered boundary-signature pair owner.
It replaces two individual unconsumed result locals with one ordered tuple
while preserving metadata/tensor mutation order, all other realign/sanitize
routes, lowerer wrappers, the following affine boundary, and the full 128/128
store.

The next characterization fixes the guarded no-layout final SE-FC and affine
pre/post pair. It preserves shared ModelIR/LayoutState identity, SE-FC-only
diagnostics, raw result schemas and order, both topology boundaries, the
following signature owner, compatibility symbols, and the full 128/128 store.
Production remains unchanged until a shared context owner is implemented
separately.

The latest checkpoint implements that shared no-layout final cleanup owner. It
replaces the two individual unconsumed result locals with one ordered tuple
while preserving the option guard, shared ModelIR/LayoutState identity,
SE-FC-only diagnostics, raw order and schemas, both topology boundaries, every
other caller, compatibility symbols, the signature successor, and the full
128/128 store.

An inherited late-binary structural test now follows the already-established
pre-terminal InstanceNorm orchestration owner instead of the removed direct
post-bias result target. This is a test-only contract repair with no production
change.

The next characterization fixes the adjacent absolute-final affine post-ADD
and decomposed-InstanceNorm post-bias cleanup pair. It preserves shared
ModelIR/LayoutState identity, affine-before-InstanceNorm order, both raw result
schemas, the boundary-signature predecessor, normalization/attention
successor, compatibility wrappers, and the full 128/128 store. Production
remains unchanged until a shared context owner is implemented separately.

The latest checkpoint implements that shared absolute-final
affine/InstanceNorm owner. It replaces two unconsumed result locals with one
ordered tuple while preserving shared context identity, raw schemas and order,
both neighboring owners, every independent raw caller, compatibility wrappers,
and the full 128/128 store.

An inherited shared-context test now identifies all four existing
target-specific context-building helpers instead of relying on the stale count
of two. This is a test-only contract repair with no production change.

The next characterization fixes the existing absolute-final
normalization/attention owner plus the following dynamic rank-one
Unsqueeze/Reshape repair. It preserves the nested raw result schemas, shared
ModelIR/LayoutState identity, exact callback order, affine/InstanceNorm
predecessor, topology/layout successor, dynamic-rank-one compatibility wrapper,
and the full 128/128 store. Production remains unchanged until the composite
context owner is implemented separately.

The latest checkpoint implements that composite normalization/attention plus
rank-one owner. It replaces two unconsumed locals with one nested result,
removes a lowerer-only closure and context alias, and preserves the existing
inner tuple, raw rank-one mapping, callback order, every independent caller,
both neighboring boundaries, compatibility wrapper, and full 128/128 store.

The next characterization fixes the lowerer-local indexed binary-layout
convergence loop. It preserves one shared graph index, broadcast → stale-
adapter → shape-reconciliation order, the three-round cap and stable-stop rule,
three-counter schema, fallback and primary callers, and the full 128/128 store.
Production remains unchanged until the pass-module owner is implemented
separately.

The latest checkpoint implements the indexed binary-layout convergence owner.
The full loop moves mechanically to `passes/binary_layout_convergence.py`,
while the private lowerer function becomes a one-return compatibility adapter
and both production call sites remain unchanged. Runtime and structural tests
fix single-index identity, callback order and forwarding, the stable-stop rule,
three-round cap, ordered result schema, and fallback/primary arguments. The
already-full phase-result store remains exactly 128 IDs and 128 owners.

The next characterization fixes the final-primary stabilization triple before
terminal topology/layout validation: indexed binary-layout convergence,
static high-rank binary coalescing, and dynamic boundary-signature realignment.
It preserves raw mapping identities and order, model/layout argument policy,
the validation and finalizer successors, shared context identity, and the full
128/128 store. Production remains unchanged until the composite context owner
is implemented separately.

The latest checkpoint implements that terminal stabilization context owner.
It replaces the three final-primary locals with one ordered composite, removes
the direct high-rank-binary import from the lowerer, and preserves exact pass
order, ModelIR/LayoutState identity, all raw mappings, both compatibility
wrappers, fallback behavior, terminal validation, finalization, and the full
128/128 store. Runtime injection and owner-aware structural tests cover the
new boundary directly.

The next characterization fixes the adjacent absolute-final boundary-signature,
affine/InstanceNorm, and normalization/attention/rank-one composite sequence
immediately before topology/layout refresh. It preserves exact model/context
arguments, shared identity, nested raw result schemas and order, the refresh
successor, and the full 128/128 store. Production remains unchanged until the
top-level context owner is implemented separately.

The latest checkpoint implements that absolute-final context owner. It keeps
all nested results unchanged inside one outer tuple, removes the lowerer's
three direct sub-owner calls and imports, and preserves context identity, pass
order, guarded predecessor, topology/layout refresh successor, compatibility
wrappers, and the full 128/128 store. Runtime injection proves the complete
identity and nesting contract directly.

The next characterization fixes the six-stage very-late dynamic/adapter
sequence immediately before mandatory static-shape reconciliation: dynamic
Reshape, indexed Conv-input, stale channel shuffle, two Concat-axis repairs,
and dynamic rank-one repair. It preserves exact model/layout/diagnostics
arguments, runtime-inferable flag, raw result schemas and order,
reconciliation and split-fallback successors, and the full 128/128 store.
Production remains unchanged until the context owner is implemented
separately.

The latest checkpoint implements that very-late dynamic/adapter context owner.
It moves the six callbacks without changing flags or argument policy, retains
all wrappers and independent callers, removes one now-unused lowerer import,
and preserves mandatory reconciliation, split fallback, raw result identity,
and the full 128/128 store. Runtime injection proves all six stages directly.

The next characterization fixes the remaining lowerer-local unbound-input
repair mapping contract. It preserves indexed repair, mutation-positive static
shape reconciliation with the returned graph index, the exact one-key result
schema, and both primary/fallback callers. One strict expected failure requires
a pass-module owner and one-return compatibility wrapper; production and the
full 128-ID/128-owner store remain unchanged. Sequential affected validation
completed with `392 passed, 1 xfailed`, and no model conversion was run.

The latest checkpoint implements that unbound-input repair owner. The lowerer
keeps a one-return compatibility adapter and both production callers, while
the pass module now owns raw indexed repair, mutation-positive reconciliation,
returned-GraphIndex forwarding, and the unchanged one-key result mapping.
Runtime injection and owner-aware structural coverage prove exact order and
identity. Sequential focused/affected/standard gates passed, no phase entry
was added, and the bounded store remains exactly 128 IDs and 128 owners.

The next characterization identifies the final owner-boundary prerequisite in
the late orphan/unbound/affine/normalization region: recurrent-alias mutation
is already indexed and pass-module-owned, while direct-TFLite result mapping
still resides in the lowerer. The new strict contract preserves raw arguments,
GraphIndex forwarding, exact mapping schema, the sole primary caller, the
independent PyTorch path, and the full 128-ID/128-owner store. Production is
unchanged pending the separate mapping-owner extraction.

The latest checkpoint implements that recurrent-alias mapping owner. The raw
indexed graph mutation remains uniquely shared by direct TFLite and PyTorch,
while a small orchestration module now owns only direct-TFLite's existing
one-key integer result. The lowerer keeps its compatibility wrapper and sole
caller; PyTorch behavior, GraphIndex identity, result schemas, pass order, and
the 128-ID/128-owner store are unchanged. Focused, affected, and standard
sequential gates all pass.

The next characterization fixes the fully pass-module-owned late prefix:
recurrent-alias summary, unbound-input repair summary, affine post-Add cleanup,
and prune-aware gather/constant/normalization cleanup. It preserves exact raw
schemas and order, shared ModelIR/LayoutState/context identity, the progress
predecessor, the dynamic-adapter successor, and the full 128-ID/128-owner
store. Production remains unchanged until the separate context-owner
implementation.

The latest checkpoint implements that late input/affine/normalization context
owner. Four unconsumed lowerer locals become one ordered tuple while exact
repair, affine, and normalization callbacks remain unchanged. The owner
preserves shared ModelIR/LayoutState/context identity and raw mapping identity;
all compatibility, fallback, and independent routes remain available. The
progress predecessor, dynamic-adapter successor, TensorFlow isolation, public
behavior, and full 128-ID/128-owner store are unchanged. Focused, affected,
and standard sequential gates all pass.

The next characterization fixes the existing five-stage pre-terminal cleanup:
InstanceNorm layout, affine/Concat/Split recovery, pre-Add, channel
Slice/Pad/Mul, and affine-tail cleanup. It preserves nested raw schemas, exact
shared-context identity, source order, the optional late-binary guard
predecessor, the separate terminal-affine successor, and the full
128-ID/128-owner store. Production remains unchanged pending a separate
context-owner implementation.

The latest checkpoint implements that pre-terminal cleanup context owner. One
small orchestration module now owns the exact five-child order and forwards the
same `ModelIRPassContext` object to every child. It returns the original nested
tuples and mappings unchanged inside one ordered outer tuple, allowing the
lowerer to replace five unconsumed locals with one composite result. The
optional late-binary reconciliation guard remains the predecessor, while the
separate terminal affine recovery rerun remains the successor and is not
absorbed into the composite. Existing wrappers, specialized owners, callbacks,
pass IDs, phase results, public APIs, artifacts, dependency boundaries, and
TensorFlow-free direct/`-cotof` behavior are preserved. Runtime identity tests,
340 affected contracts, and the complete sequential standard gate set all
pass; the phase-result store remains exactly 128 IDs and 128 owners.

The next characterization fixes four adjacent late-layout composites:
reshape, base-only channel shuffle/Gather, attention, and window cleanup. It
preserves exact shared-context identity, the channel policy flags, all nested
raw tuple schemas and order, the optional elementwise-fanout predecessor, the
indexed final-shape successor, independent full-policy and callback routes,
and the full 128-ID/128-owner store. Production remains unchanged pending a
separate context-owner implementation.

The latest checkpoint implements that late reshape/shuffle/attention/window
context owner. One orchestration module now forwards the same
`ModelIRPassContext` to all four children, preserves the base-only channel
flags, and returns every raw nested tuple unchanged in source order. The
lowerer replaces four observation-only locals with one composite result while
retaining the generic channel wrapper for its guarded full-policy and callback
routes. The optional elementwise-fanout predecessor, indexed final-shape
successor, child owners, pass IDs, phase results, public behavior, artifacts,
dependency boundaries, and TensorFlow-free direct/`-cotof` behavior remain
unchanged. Runtime identity tests, 405 affected contracts, and the full
sequential standard gate set pass; the phase-result store remains exactly 128
IDs and 128 owners.

The next characterization fixes four adjacent final-layout composites:
boundary channel cleanup, terminal Slice/Concat recovery, final
Slice/pre-Concat cleanup, and terminal Concat bridge cleanup. It preserves the
shared pass context nested in the existing callback-bearing recovery context,
the independent earlier wrapper route, all raw tuple schemas and order, the
indexed final-shape predecessor, the optional elementwise-fanout successor,
and the full 128-ID/128-owner store. Production remains unchanged pending a
separate context-owner implementation.
