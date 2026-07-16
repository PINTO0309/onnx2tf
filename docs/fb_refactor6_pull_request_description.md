# Refactor flatbuffer-direct terminal shape and binary layout passes

## Summary

This branch continues the staged `flatbuffer_direct` refactor by moving thirty-eight
fully characterized compatibility rules out of the central ONNX-to-ModelIR
lowerer and into focused pass modules:

- static Squeeze shape sanitization;
- LiteRT.js ExpandDims/Squeeze-to-Reshape conversion;
- exact rank-four binary NHWC/NCHW adaptation;
- singleton-broadcast rank-four binary NHWC/NCHW adaptation;
- rank-three/four channelwise broadcast-constant runtime-layout repair;
- Conv/Pool output elementwise-passthrough compatibility recovery;
- indexed-first Conv/Mul/Add affine-fold compatibility orchestration;
- final static runtime-shape/signature consistency;
- dynamic boundary-signature map realignment;
- Transpose/QDQ bridge and residual-closure optimization;
- quantized Swish NHWC-island and residual-Concat orchestration;
- HardSwish/SE/HardSigmoid gating-block layout recovery;
- the remaining generic NHWC pre-Concat compatibility matcher;
- strict rank-four Transpose/Slice/inverse-Transpose passthrough;
- Shape-extraction layout recovery for Gather and Slice consumers;
- the complete indexed-first pre-Add compatibility composite;
- late dual-pre-Add to single-post adapter recovery;
- terminal affine/Reshape/FullyConnected layout recovery;
- terminal PReLU/Reshape/BatchMatMul layout recovery;
- terminal Transpose/Mul/Add/PReLU compatibility recovery;
- Transpose/Mean/Mul/Add compatibility recovery;
- dual affine-input BatchMatMul compatibility recovery;
- BatchMatMul-to-SE layout compatibility recovery;
- rank-three BatchMatMul input-adapter to adjoint-flag recovery;
- probable-NHWC axis-sensitive metadata and output sanitization;
- NHWC/NCHW elementwise fan-out compatibility recovery;
- residual Add/Mul/Add/PReLU compatibility recovery;
- residual Add/Mul/Add/post-Transpose fan-out compatibility recovery;
- pre-unary Mul/Add/post-Transpose fan-out compatibility recovery;
- indexed-first pre-Add/Mul/Reshape-suffix compatibility recovery;
- indexed-first Swish/plain-unary Reshape-suffix compatibility recovery;
- indexed-first Swish/plain-unary Squeeze-suffix compatibility recovery;
- indexed-first factorized/singleton ExpandDims compatibility recovery;
- indexed-first static/dynamic flatten-HW compatibility recovery;
- indexed-first static/relaxed attention-QKV compatibility recovery;
- stale NCHW-to-NHWC channelwise-binary Transpose repair;
- QLinear SiLU prefix NHWC propagation and legacy-adapter recovery;
- Mean/MaxPool/Concat/Conv NHWC recovery.

The change reduces the amount of mutable implementation embedded in
`lower_from_onnx2tf.py` while preserving its private compatibility entry
points and every existing production call position. This is an ownership and
testability checkpoint, not a public API or conversion-policy redesign.

## Motivation

The direct FlatBuffer backend historically accumulated shape repair, graph
rewriting, layout compatibility, and lowering logic in one large module. Even
when an individual rule was local, its implementation was difficult to test
without importing the entire lowerer, and changing it risked interaction with
unrelated terminal cleanup stages.

This branch establishes smaller op-family owners with explicit synthetic
contracts. The central lowerer now dispatches through thin wrappers, so call
order and private API compatibility remain stable while each rule can be
validated independently. Architecture tests make that ownership boundary
enforceable and prevent the implementations from drifting back into the
central module.

## Detailed changes

### Static Squeeze shape sanitization

`passes/squeeze_shape_sanitization.py` owns the final static Squeeze metadata
repair. It preserves negative-axis normalization, duplicate and out-of-range
axis handling, constant-payload authority, shape/signature reconciliation,
counter semantics, and idempotence. The pass intentionally performs one
operator traversal: this terminal rule does not require producer or consumer
queries, so building another graph index would add work without improving its
contract.

### ExpandDims/Squeeze Reshape compatibility

`passes/expand_squeeze_reshape.py` owns the LiteRT.js compatibility rewrite for
ExpandDims and Squeeze. Static target shapes, speculative inactive-If Squeeze
handling, dynamic SHAPE/GATHER target construction, semantic-axis metadata,
unused-tensor pruning, and LayoutState synchronization remain together in one
owner.

Dynamic pre-operators are inserted through `ModelIRGraphIndex` in deterministic
original-index order. This preserves the required runtime sequence
`SHAPE -> GATHER -> RESHAPE` without replacing the complete operator list.
The graph index is created only when dynamic pre-operators are actually
needed.

### Exact rank-four binary layout adaptation

`passes/binary_layout_adapter.py` owns the existing compatibility rule for two
full static rank-four shapes that are exact NHWC/NCHW permutations. For
ADD/MUL/SUB/DIV/MAXIMUM/MINIMUM it inserts the historical input-1 Transpose,
clones quantization independently, restarts fixed-point matching, and prunes
unused tensors. Both permutation directions and all no-op guards remain
unchanged.

### Singleton-broadcast binary layout adaptation

The same op-family module now owns the related but semantically distinct
singleton-channel policy. It supports either operand position and selects one
of four branches from the declared output shape:

- NCHW output: transpose the NHWC operand to NCHW;
- NHWC output: reshape the singleton NCHW operand to `[N,H,W,1]`.

All six supported binary operators retain the existing branch precedence,
unique-name behavior, tensor metadata, quantization cloning, insertion and
rewiring order, fixed-point restart, stats key, and conditional pruning. A
NumPy broadcast guard continues to leave an operator unchanged when its inputs
already broadcast to the declared output. The exact and singleton policies
remain separate public pass functions and separate statistics because merging
them would obscure their different operand and output-layout semantics.

### Channelwise broadcast-constant runtime-layout repair

`passes/binary_layout_adapter.py` also owns the complete former central repair
for rank-three and rank-four channelwise constants consumed by binary
operators. The move is mechanical: after normalizing only the function name,
the module owner AST is identical to the previous lowerer implementation. The
lowerer retains its private wrapper, all three direct production calls, and the
call inside the existing three-round indexed binary-layout convergence helper.

The owner preserves reuse of a caller-provided `ModelIRGraphIndex`, indexed
binary-op traversal, logical-layout/name/producer hints for ambiguous
broadcasts, standard NCHW-to-NHWC rotation, exact inverse recovery for a stale
NHWC constant, and the historical statistic. Exclusive constants are still
updated in place. Shared constants retain snapshot-based copy-on-write,
quantization cloning, deterministic names, and differential consumer updates
through `_set_operator_inputs(..., graph_index=graph_index)`.

### Conv/Pool output elementwise passthrough

`passes/convpool_output_passthrough_compat.py` owns the complete corrected
compatibility helper for a Conv/Pool-family NHWC output followed by a leading
NHWC-to-NCHW Transpose and an elementwise region. After normalizing only the
function name, its 556-line AST is identical to the corrected lowerer owner.
The lowerer retains one private wrapper and its single ordered production call.

Forward elementwise discovery, external-runtime input adapters, channel-last
metadata hints, legacy boundary adapters, keepdims Mean-axis absorption,
Reshape/Transpose/Squeeze follow-up handling, mutation order, pruning, and the
historical statistic remain unchanged. The pre-extraction atomicity correction
is retained: all external runtime shapes are planned before the first metadata
or graph mutation. The focused success and rejection corpus compares module
owner and compatibility wrapper over complete ModelIR fingerprints.

### Conv/Mul/Add affine-fold orchestration

`passes/conv_mul_affine_fold_compat.py` now owns the complete indexed-first
orchestration formerly embedded in the lowerer. It invokes the existing bounded
`conv_mul_affine_fold.py` owner with one `ModelIRGraphIndex` and caller
`LayoutState`, then runs the unchanged raw compatibility fallback for relaxed
scalar/dynamic coefficients, missing bias, fused ReLU, Add-only, Mul/Add, and
historically shared constants. The single prune and removal of pruned names
from LayoutState remain at the orchestration exit.

The moved 381-line function is AST-identical after normalizing only its name;
two comments document legacy unused locals without changing that AST. The
lowerer retains a signature-compatible wrapper at all three ordered production
positions. Focused tests compare compatibility owner and wrapper across the
complete ModelIR and LayoutState while retaining the indexed atomicity,
determinism, signed-zero, and raw fallback contracts.

### Static shape-signature consistency

`passes/static_shape_signature_sanitization.py` owns the final metadata-only
consistency pass. It builds one producer map and preserves dynamic signatures
only when they belong to an ONNX boundary contract, a runtime-dependent
WHERE/RANGE/RESHAPE/TOPK_V2 root, a graph-output leading axis, or a recursively
reachable dynamic lineage. The ancestry walk is memoized and cycle-safe, and
constant payloads stop propagation. Missing, rank-mismatched, stale, or
unjustified dynamic signatures on fully static internal tensors are repaired.
Both historical production positions and all four statistics remain unchanged.

The companion boundary-map realigner lives in the same module while reusing
the core ONNX-analysis alignment primitive. It preserves malformed-entry and
rank guards, deterministic repeated-extent placement, exact in-place map
mutation, and all three historical production positions. It does not inspect
or mutate graph topology.

### Transpose/QDQ bridge optimization

`passes/transpose_qdq_bridge_layout.py` owns the former central QDQ bridge
optimizer. It is deliberately separate from terminal exact-grid, Concat-input,
Mean, activation, PReLU, Reshape, and TransposeConv quantization cleanup. The
complete owner moves as one unit because behavior depends on a shared
fixed-point loop with A→B→C→D priority and restart after every rewrite:

- complete Transpose/Quantize/Dequantize/Transpose round trips;
- single Quantize or Dequantize bridges with guarded post fan-out;
- two-QDQ-branch Add residual closure;
- mixed float/QDQ Add residual closure, including legacy fan-out.

The existing private wrapper, four runtime sweep positions, inverse-permutation
and per-tensor-grid guards, public-boundary protection, metadata permutation,
quantization cloning, mutation order, pruning, and statistics are unchanged.
The owner remains a mechanical compatibility implementation; differential
GraphIndex mutation is a separate follow-up requiring family-level evidence.

### Quantized Swish orchestration

`passes/quantized_swish_layout.py` now owns the complete orchestration around
its previously extracted indexed phase owners. The public module owner retains
the exact ordered sequence of primary branch/metadata rewriting, first inverse
post-Transpose cleanup, late mixed-input Concat normalization and post cleanup,
the independent wrong-way Conv-input safety valve, and final tensor pruning.
Its statistics still aggregate rewritten branches, removed pre-Transposes,
the union of propagated tensors, primary plus late Concat-axis changes, and
all post/safety-valve removals under the historical keys.

The spatial-agnostic residual-Concat closure remains a separate entry point.
It invokes the same owner with `min_spatial_stage=0` and
`require_concat_closure=True`, then remaps the established result to the four
closure-specific keys. Both private lowerer symbols are retained as thin
compatibility wrappers, and their two production call positions are
unchanged. After normalizing only function and safety-owner names, both moved
function ASTs are identical to their pre-extraction implementations.

### HardSwish SE gating layout

`passes/hardswish_se_layout.py` owns the former 463-line central compatibility
implementation for HardSwish plus squeeze/excitation Conv and HardSigmoid
gating blocks. Its direct and decomposed activation roots, expanded and fused
gate forms, keepdims Mean axis rotation, Conv bridge removal, residual output
renaming, tensor metadata and quantization propagation, graph-order restart,
pruning, and statistic remain one cohesive contract. The lowerer retains one
thin private wrapper and both production calls remain unchanged. The moved
owner AST is identical to its pre-extraction implementation after normalizing
only the function name.

### Generic NHWC pre-Concat legacy compatibility owner

`passes/nhwc_concat_legacy_layout.py` now owns the complete former 2,452-line
central compatibility matcher. The move deliberately keeps this fallback as a
single semantic unit because its nested input planners, indexed-family
exclusion contracts, action ordering, fixed-point restart, constant
materialization, metadata and quantization propagation, operator removal, and
pruning are interdependent. This is not a source-line limit and does not alter
the already indexed float or quantized families. The lowerer retains the
historical private symbol as a one-call wrapper; the composite wrapper still
runs float indexed, quantized indexed, and legacy owners in that order and
still occupies four production positions. The old and new legacy-owner ASTs
are identical after function-name normalization.

### Slice pre/post NHWC passthrough

`passes/slice_prepost_layout.py` owns the former 148-line central matcher for a
strict rank-four NHWC→NCHW Transpose, constant Slice, and inverse Transpose.
The exact permutation, exclusive consumer, public-boundary, constant arity,
shape-reproduction, as-is versus remapped-parameter selection, fixed-point,
operator-removal, conditional-pruning, and statistic behavior remain intact.
It reuses the existing static-shape Slice inference owner directly. The lowerer
retains a one-call private wrapper at the unchanged single production position,
and the moved AST is identical after function-name normalization.

### Shape-extraction layout recovery

`passes/shape_extract_layout.py` owns the former 285-line central matcher for
Shape consumers behind an NHWC-to-NCHW Transpose. It preserves all three
historical families: Gather-index remapping, contiguous Slice remapping, and
non-contiguous Slice-to-Gather conversion. Shared constants remain
clone-on-write with dtype and quantization metadata preserved. The complete
all-users-supported rule, public-boundary and fan-out guards, fixed-point
restart, conditional pruning, and historical statistic are unchanged.

The lowerer retains a one-call private wrapper at all three unchanged
production positions. After normalizing only the function name, the old and
new complete owner ASTs are identical.

### Indexed-first pre-Add compatibility composite

`passes/pre_add_layout.py` owns the complete former 1,593-line lowerer
implementation. It remains one semantic compatibility unit: the bounded
direct/unary indexed owner runs first, followed by the historical Swish,
unary, Mul-constant, Mul/Sub-constant, Gather, constant-Add, nested-Add, PReLU,
direct-NCHW bridge, post-alias, and legacy-consumer fallback families. Their
fixed-point order, producer/consumer rebuilds, shared-constant copy-on-write,
metadata, quantization, mutation order, marker behavior, pruning, and statistic
are unchanged.

The lowerer retains a one-call private wrapper at all four production
positions and in the safe-transpose bundle. After normalizing only the function
name, the old and new complete owner ASTs are identical. This move is not a
source-line limit and does not broaden the indexed owner.

### Dual-pre-Add to single-post adapter recovery

`passes/dual_pre_add_layout.py` owns the former 166-line late helper. It
preserves the strict two-exclusive-input-adapter contract, public Add/output
guards, existing inverse-post rejection, NHWC Add metadata, quantization
cloning, one inserted NCHW compatibility adapter, fixed-point order,
unconditional prune boundary, and statistic. The lowerer retains a one-call
private wrapper at the unchanged single late production position. The complete
old and new owner ASTs are identical after function-name normalization.

The historical fixed permutation name is deliberately unchanged. An existing
`__nhwc_to_nchw_perm_rank4__` tensor is not yet validated for dtype, payload,
producer, or graph visibility; that latent collision risk requires a separate
semantic hardening checkpoint rather than being folded into this ownership
move.

### Terminal affine/Reshape/FullyConnected recovery

`passes/terminal_affine_fc_layout.py` owns the former 293-line late helper for
NHWC-to-NCHW Transpose, Mul, Add, Reshape, and FullyConnected. It preserves
chain exclusivity and public-boundary checks, channel-constant rotation,
shared-constant copy-on-write, both FullyConnected weight orientations,
flatten-order permutation, shape/dtype/quantization metadata, fixed-point
restart, pruning, statistic, and the single production position. The lowerer
retains a one-call private wrapper, and the complete owner AST is identical
after function-name normalization.

One historical transactional risk remains explicit: an exclusive Mul constant
can be rotated in place before the Add constant is found invalid, leaving a
partial mutation with a zero statistic. Correcting this requires a separate
plan/revalidation checkpoint and is not mixed into the mechanical move.

### Terminal PReLU/Reshape/BatchMatMul recovery

`passes/terminal_prelu_bmm_layout.py` owns the former 263-line late helper. It
preserves scalar, rank-three CHW, rank-four NCHW, and already-NHWC alpha
handling; shared alpha/RHS copy-on-write; NHWC flatten-order RHS permutation;
adjX/adjY rejection; metadata and quantization cloning; fixed-point restart;
pruning; statistic; and the conditional single production position. The
lowerer retains a one-call private wrapper, and the old/new complete owner ASTs
are identical after function-name normalization.

Existing alpha and RHS tensors are still not fully validated for producers,
variable state, or graph visibility. Those ownership guards require a separate
semantic hardening checkpoint and are not silently added by this mechanical
move.

### Terminal Transpose/Mul/Add/PReLU compatibility recovery

`passes/terminal_affine_prelu_layout.py` owns the former 295-line terminal
helper. It preserves commutative affine inputs, NCHW-to-NHWC channel-constant
rotation, shared-constant copy-on-write, multiple post-Transpose aliases,
retained legacy NCHW consumers through one reverse adapter, metadata and
quantization propagation, fixed-point restart, pruning, statistics, and its
single ordered production statement reached through four runtime recovery
invocations. The lowerer retains a one-call private wrapper, and the old/new
complete owner ASTs are identical after function-name normalization.

The raw owner still rebuilds complete maps in an unbounded loop and can rotate
an earlier constant before a later constant rejects. Transactional hardening is
kept separate from this exact ownership move.

### Transpose/Mean/Mul/Add compatibility recovery

`passes/mean_affine_prepost_layout.py` owns the former 359-line helper. It
preserves NCHW-to-NHWC reduction-axis remapping, commutative affine inputs,
static broadcast validation, channel-constant rotation and copy-on-write,
post-Transpose alias collapse, tensor metadata and quantization propagation,
fixed-point restart, pruning, statistics, and all three ordered source call
positions reached through five runtime invocations. The lowerer retains a one-
call private wrapper, and the old/new complete owner ASTs are identical after
function-name normalization.

The raw owner still rebuilds the complete consumer map in an unbounded loop and
performs in-place axes/constant updates without an immutable all-or-nothing
plan. Transactional hardening is kept separate from this exact ownership move.

### Dual affine-input BatchMatMul compatibility recovery

`passes/batchmatmul_affine_input_layout.py` owns the former 317-line helper. It
preserves commutative Mul/Add inputs, exact exclusive branch matching, NCHW-to-
NHWC channel-constant rotation, rank-three Reshape shape reversal, left post-
Transpose removal, `adjY=True` conversion, metadata propagation, fixed-point
restart, pruning, statistics, and both ordered production positions. The
lowerer retains a one-call private wrapper, and the old/new complete owner ASTs
are identical after function-name normalization.

The raw owner still mutates both branches sequentially before every shape
constant is known valid. Transactional hardening is kept separate from this
exact ownership move.

### BatchMatMul-to-SE layout compatibility recovery

`passes/batchmatmul_se_layout.py` owns the former 363-line helper. It preserves
the BatchMatMul/Reshape source, NCHW Mean and axis remap, NHWC Conv gate branch,
reverse gate adapter, Logistic and residual Mul merge, constant updates, alias
rewiring, metadata and quantization propagation, fixed-point restart, pruning,
statistics, and both ordered production positions. The lowerer retains a one-
call private wrapper, and the old/new complete owner ASTs are identical after
function-name normalization.

The raw owner still performs its long mutation sequence without an immutable
all-or-nothing plan. Transactional hardening is kept separate from this exact
ownership move.

### Rank-three BatchMatMul input-adapter recovery

`passes/batchmatmul_adjoint_layout.py` owns the former 145-line helper. It
preserves exclusive Transpose-output ownership, graph-output and fully-known-
shape guards, exact rank-three permutation validation, direct `[0,2,1]`
Transpose removal, singleton-preserving Transpose-to-Reshape conversion, new
INT32 shape-tensor creation, the corresponding `adjX` or `adjY` toggle, fixed-
point restart, conditional pruning, statistics, and both ordered production
positions. The lowerer retains a one-call private wrapper, and the complete
old/new owner ASTs are identical after function-name normalization.

The focused fixture covers both input positions and both rewrite forms, runs
the module owner and compatibility wrapper on deep copies, compares the full
ModelIR, and fixes idempotence. The mechanical owner still rebuilds complete
producer/consumer maps after every accepted adapter and directly mutates or
deletes operators without an invariant transaction. Indexed transactional
migration is kept separate from this exact ownership move.

### Probable-NHWC axis-sensitive sanitization

`passes/probable_nhwc_axis_sanitizer.py` owns the former 245-line helper. It
preserves the probable-NHWC shape heuristic, SPLIT axis repair and shared-axis
copy-on-write, CONCATENATION axis repair, SLICE begin/size rotation, unary and
binary output-metadata propagation, explicit/public NCHW guards, conditional
terminal NHWC-to-NCHW graph-output adapters, fixed-point restart, both
statistics, and both ordered production positions. The lowerer retains a one-
call private wrapper, and the complete old/new owner ASTs are identical after
function-name normalization.

The four-case focused fixture runs the module owner and compatibility wrapper
on deep copies and compares the complete ModelIR for every positive and no-op
contract. The mechanical owner still rebuilds full graph maps, mutates SLICE
constants without copy-on-write, and inserts terminal operators directly
without an invariant transaction. Semantic hardening and indexed migration are
kept separate from this exact ownership move.

### NCHW/NHWC elementwise roundtrip root-metadata correction

The still-central
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains` helper now skips
its private root output during the intermediate-tensor metadata permutation.
Previously it permuted that tensor once with the elementwise subgraph, copied
the already-permuted metadata to the canonical post-Transpose output, and
permuted it a second time. A `[1,8,8,3]` NHWC root therefore became the invalid
`[1,8,3,8]` shape instead of `[1,3,8,8]`.

The focused contract fixes a positive multi-input elementwise closure,
embedded constant retention, canonical alias rewiring, pruning and idempotence,
plus fan-out and public-output rejection. The implementation remains in the
lowerer because positive production ownership has not been observed; this safe
semantic correction is not mixed with an ownership extraction.

### NHWC/NCHW elementwise fan-out compatibility recovery

`passes/elementwise_fanout_layout.py` owns the former 555-line helper. It
preserves forward elementwise-DAG discovery, the conservative external-runtime-
input rejection, local and shared per-channel constant rotation, inverse
boundary-Transpose collapse, legacy NCHW adapter insertion, canonical aliases,
metadata and quantization propagation, candidate snapshots, unbound-input
rollback, fixed-point restart, pruning, statistics, and all three ordered
production positions. The lowerer retains a one-call private wrapper, and the
complete old/new owner ASTs are identical after function-name normalization.

The independent unbound-input validator is imported through a same-name
compatibility alias, so the pass does not depend on the central lowerer. The
focused fan-out fixture runs the module owner and compatibility wrapper on deep
copies and compares the complete ModelIR. The mechanical owner still rebuilds
full graph maps and deep-copies the complete ModelIR per candidate; indexed
transactional redesign is kept separate from this exact ownership move.

### Residual Add/Mul/Add/PReLU compatibility recovery

`passes/residual_affine_prelu_layout.py` owns the former 415-line compatibility
helper. It preserves dual pre-Add input planning, broadcast-aware Mul/Add/PReLU
constant prevalidation and copy-on-write, post aliases, legacy NCHW consumer
adapter retention, tensor metadata and quantization, mutation/removal order,
fixed-point restart, pruning, statistic, and all three source call positions.
The lowerer retains a one-call private wrapper. The complete old/new owner ASTs
are identical after function-name normalization, and the separate indexed SiNet
late-residual owner remains in its existing order.

The raw compatibility owner still has looser producer, variable-state, and
graph-visibility contracts for constants than newer indexed passes. Hardening
those boundaries remains a separate transactional checkpoint.

### Residual Add/Mul/Add/post-Transpose fan-out recovery

`passes/residual_affine_fanout_layout.py` owns the former 477-line compatibility
helper. It preserves the shared dual pre-Add prefix, multiple
Mul-constant→Add-constant→post-Transpose branches, optional legacy NCHW
consumers, the exact profitability decision, broadcast-aware constant
prevalidation and rotation, shared-constant copy-on-write, retained legacy
adapter, tensor metadata and quantization, mutation/removal order, fixed-point
restart, pruning, statistic, and all three source call positions. The lowerer
retains a one-call private wrapper. The complete old/new owner ASTs are
identical after function-name normalization.

The compatibility owner still does not fully validate constant producers,
variable state, or graph visibility. Those ownership guards remain a separate
semantic-hardening checkpoint so this mechanical move cannot change artifact
behavior.

### Pre-unary Mul/Add/post-Transpose fan-out recovery

`passes/pre_unary_affine_fanout_layout.py` owns the former 401-line
compatibility helper. It preserves the strict private NHWC-to-NCHW Transpose,
the complete seven-operation unary allowlist, all
Mul-constant→Add-constant→post-Transpose branches, broadcast-aware constant
prevalidation and rotation, shared-constant copy-on-write, tensor metadata and
quantization, mutation/removal order, fixed-point restart, pruning, statistic,
and all three source call positions. The lowerer retains a one-call private
wrapper. The complete old/new owner ASTs are identical after function-name
normalization.

The raw compatibility contract still does not share the Session GraphIndex or
LayoutState and does not fully validate constant producers, variable state, or
graph visibility. Those improvements require an indexed immutable plan and an
independent semantic checkpoint.

### Indexed-first pre-Add/Mul/Reshape-suffix compatibility recovery

`passes/pre_add_mulconst_reshape_suffix_compat_layout.py` owns the former
509-line composite. It preserves the existing indexed owner as the first
dispatch, one per-call `ModelIRGraphIndex`, caller `LayoutState` forwarding,
the unchanged direct/direct and direct/Mul-constant raw fallback, Mul and
Reshape constant copy-on-write, legacy NCHW adapter handling, combined
statistic, fixed-point restart, and the single prune/report boundary. The
lowerer retains one private wrapper at its unchanged production position. The
complete old/new composite ASTs are identical after function-name
normalization; the indexed semantic owner is unchanged.

The raw fallback still rebuilds whole-graph maps and has looser constant
producer, variable-state, and graph-visibility contracts than the indexed
plan. Converting it to an immutable transaction is a separate semantic task.

### Indexed-first Swish/plain-unary Reshape-suffix compatibility recovery

`passes/pre_unary_reshape_suffix_compat_layout.py` owns the former 302-line
composite. It preserves the existing indexed Swish owner as the first dispatch,
one per-call `ModelIRGraphIndex`, caller `LayoutState`, the thirteen-operation
plain-unary and relaxed Swish fallback, reshape constant/option mutation,
combined statistic, fixed-point restart, one prune/report boundary, and removal
of pruned names from LayoutState. The lowerer retains one private wrapper at its
unchanged production position. The complete old/new composite ASTs are
identical after function-name normalization; the indexed owner is unchanged.

The raw fallback still performs whole-graph scans and in-place relaxed constant
updates without an immutable differential-index transaction. That semantic
hardening remains separate from this ownership checkpoint.

### Indexed-first Swish/plain-unary Squeeze-suffix compatibility recovery

`passes/pre_unary_squeeze_suffix_compat_layout.py` owns the former 297-line
composite. It preserves the indexed static Swish owner as the first dispatch,
one per-call `ModelIRGraphIndex`, caller `LayoutState`, plain-unary, axis-3,
dynamic-signature, and relaxed Swish fallback behavior, Squeeze axis option
remapping, combined statistic, fixed-point restart, one prune/report boundary,
and LayoutState cleanup. The lowerer retains one private wrapper at the
unchanged production position. The complete old/new composite ASTs are
identical after function-name normalization; the indexed owner is unchanged.

Whole-graph fallback scans and relaxed in-place Squeeze axis updates remain a
separate immutable-indexed migration task.

### Indexed-first factorized/singleton ExpandDims compatibility recovery

`passes/expanddims_reshape_compat_layout.py` owns the former 271-line
composite. It preserves the strict indexed factorized Case B owner as the first
dispatch, one per-call `ModelIRGraphIndex`, caller `LayoutState`, singleton Case
A and relaxed raw fallback behavior, Reshape/permutation constant and option
updates, combined statistic, fixed-point restart, one prune/report boundary,
and LayoutState cleanup. The lowerer retains one private wrapper at both
unchanged production call positions. The complete old/new composite ASTs are
identical after function-name normalization; the indexed owner is unchanged.

Whole-graph fallback scans and relaxed in-place constant updates remain a
separate immutable-indexed migration task.

### Indexed-first static/relaxed attention-QKV compatibility recovery

`passes/attention_qkv_reshape_compat_layout.py` owns the former 245-line
composite. It preserves the strict indexed static HAD owner as the first
dispatch, one per-call `ModelIRGraphIndex`, caller `LayoutState`, HDA,
shared-constant copy-on-write, dynamic-signature, and relaxed raw fallback
behavior, shape/permutation constant cloning and updates, combined statistic,
fixed-point restart, one prune/report boundary, and LayoutState cleanup. The
lowerer retains one private wrapper at both unchanged production call
positions. The complete old/new composite ASTs are identical after function-
name normalization; the indexed owner is unchanged.

Whole-graph fallback scans and relaxed clone-on-write mutation remain a
separate immutable-indexed migration task.

### Indexed-first static/dynamic flatten-HW compatibility recovery

`passes/flatten_hw_reshape_compat_layout.py` owns the former 175-line
composite. It preserves the strict indexed static flatten-HW owner as the first
dispatch, one per-call `ModelIRGraphIndex`, caller `LayoutState`, dynamic-
signature and relaxed raw fallback behavior, Reshape constant and option
updates, combined statistic, fixed-point restart, one prune/report boundary,
and LayoutState cleanup. The lowerer retains one private wrapper at both
unchanged production call positions. The complete old/new composite ASTs are
identical after function-name normalization; the indexed owner is unchanged.

Whole-graph fallback scans and relaxed in-place constant updates remain a
separate immutable-indexed migration task.

### Stale channelwise-binary Transpose repair ownership

`passes/stale_binary_adapter_repair.py` now owns the corrected 132-line stale
NCHW-to-NHWC channelwise-binary Transpose repair. Its owner body is
AST-identical to the corrected predecessor at checkpoint `c869c410`. The
lowerer retains its historical private symbol as a compatibility wrapper and
forwards an optional caller-owned `ModelIRGraphIndex` without rebuilding it.
Both standalone fallback/final calls remain in their original positions.

The fixed three-round `_run_indexed_binary_layout_convergence` coordinator
intentionally remains in the lowerer: it owns the ordering boundary between
the separate broadcast-constant repair, stale-Transpose repair, and static
shape reconciliation. Direct owner and wrapper execution on deep-copied
multi-match ModelIR graphs produces identical statistics and complete ModelIR
fingerprints. Architecture tests prevent a lowerer import cycle, require the
rank and signature guards to precede mutation in the module owner, preserve
the wrapper's index forwarding, and freeze the runner plus both standalone
production call boundaries.

### QLinear SiLU prefix characterization

The remaining raw 419-line
`_optimize_nhwc_prefix_qlinear_silu_chains` helper now has an explicit
pre-change synthetic contract. LOGISTIC and decomposed HardSigmoid branches
both prove NHWC propagation, pre/post Transpose removal, metadata permutation,
and exact statistics. A two-candidate fixture freezes fixed-point restart and
candidate order. A separate legacy-consumer case freezes insertion of the
NHWC-to-NCHW adapter. Eight rejection cases cover permutation, public/fan-out,
quantization, shared sigmoid output, layout-sensitive consumer, and
non-singleton-constant boundaries. Architecture coverage records the raw
owner's current 419-line location, full-scan consumer-map ownership, mutation
APIs, fixed-point loop, and ordered QLinear recovery call boundary.

Four strict xfails record pre-existing problems before implementation changes:
rejected and second no-rewrite calls create then prune an otherwise unused
internal permutation tensor and append lineage metadata; an unrelated public
tensor colliding with the reserved internal name is reused without payload
validation; and a rank-two Mul output signature is accepted before graph
rewiring and a rank-four legacy adapter insertion. Production source remains
unchanged at this characterization checkpoint. The existing sequential real-
model audit already established zero rewrites and zero SWAP for every measured
candidate, so no redundant conversion was run.

### QLinear SiLU prefix transactional correction

The four characterized defects are corrected without changing valid topology
matching or statistics. All rank-four metadata targets and effective
signatures are now resolved before the first ModelIR mutation. Legacy adapters,
their collision-free tensor names, cumulative consumer input updates, and the
permutation dependency are fully planned before commit. The reserved
permutation is reused only when dtype, shape, signature, constant payload,
variable state, and quantization state are exact; otherwise a new validated
constant receives a collision-safe name. No permutation is created for a
candidate without legacy consumers, and unused tensors are pruned only after a
successful rewrite. Rejected and repeated zero-rewrite calls are therefore
complete metadata-preserving no-ops.

All four former strict xfails now pass, including four separate malformed
target-signature cases and exact valid-permutation reuse. A new strict xfail
records a separate pre-existing duplicate-consumer issue: when one downstream
operator consumes the Mul output in two slots, the legacy consumer map returns
that operator index twice and the helper plans both slots twice, leaving four
adapters instead of two. That behavior is documented before a separate fix;
it is not silently combined with this transactional correction.

### QLinear SiLU legacy-consumer deduplication

The remaining strict xfail is fixed by deduplicating final-Mul consumer
operator indices in first-observed order before classifying their input slots.
The consumer map may still represent every matching edge, but each downstream
operator is planned exactly once and every matching slot within that operator
receives exactly one adapter. A same-consumer two-slot ADD now creates two
adapters rather than four. A separate two-consumer fixture proves distinct
consumer order, deterministic adapter naming, and one adapter per edge. Single-
slot behavior, fixed-point counts, permutation reuse/collision handling, all
ordinary rejections, and production call boundaries remain unchanged. No
strict xfail remains for this owner.

### QLinear SiLU prefix ownership extraction

The corrected 513-line owner now resides in
`passes/qlinear_silu_prefix_layout.py`. Its function AST is identical to the
corrected lowerer predecessor at checkpoint `0cf699fd`. The lowerer retains
the historical private name as a one-return compatibility wrapper, so the
ordered QLinear recovery helper and both of its production boundaries remain
unchanged. The owner module imports only ModelIR utilities and cannot import
the central lowerer.

Direct owner and wrapper executions are compared on deep-copied LOGISTIC,
decomposed HardSigmoid, legacy-adapter, and reserved-name collision graphs.
They produce identical statistics, complete graph/tensor fingerprints, layout
state, and diagnostic metadata. Architecture checks retain pre-mutation
planning, conditional pruning, ordered consumer deduplication, exact 513-line
ownership, the thin wrapper, and the existing recovery sequence. This is a
mechanical ownership move after the separately committed correctness fixes.

### Mean/MaxPool/Concat/Conv characterization

The next raw 310-line
`_optimize_transpose_mean_maxpool_concat_conv_chains` owner now has an explicit
pre-change contract. Synthetic positive cases freeze NHWC rewiring of both
DEQUANTIZE branches, Mean axes `[2,3]` to `[1,2]`, pool-adapter removal,
Concat axis `1` to `3`, dynamic batch signatures, per-axis quantized dimension
`1` to `3`, multiple post-Quantize adapters, fixed-point multiple matches,
pruning, statistics, and idempotence. Ten ordinary rejection cases preserve
wrong permutation, boundary/fan-out, keepDims/axes, Concat-axis, quantization,
and post-consumer guards. Architecture coverage records the raw 310-line
owner, both whole-graph maps, mutation utilities, fixed-point loop, and its
existing ordered QLinear recovery boundary.

Nine strict xfails record pre-existing correctness problems before source
changes. Short source or pool signatures are read after mutations; missing,
rank-three, and short-signature additional Concat inputs are validated only
after both branches and Concat have been rewritten; and the Mean axes constant
is modified without excluding shared, graph-input, graph-output, or variable
ownership. These cases require a zero statistic and complete unchanged ModelIR
but currently raise or leave partial/nonlocal mutations. The earlier
sequential QLinear group audit already established zero production rewrites
and zero SWAP, so no duplicate model conversion was run.

### Mean/MaxPool/Concat/Conv transactional correction

All nine characterized defects are corrected. The Mean axes tensor must now be
an unquantized, non-variable INT32 tensor backed by an INT32 buffer, absent from
graph inputs/outputs, and consumed only by the matched Mean. Every source,
intermediate, removed-adapter output, post-adapter output, and planned Concat
input is resolved with rank-four shape and effective signature before mutation.
The pool substitution, Mean axes/shape/signature, Concat shape/signature,
per-axis QDIM update, aliases, and removal indices are calculated as one local
plan. Only a complete plan commits setters, constant data, options, tensor
metadata, aliases, and operator removals. Pruning runs only after a non-zero
rewrite, making rejection a complete metadata-preserving no-op.

The former nine strict xfails and three additional axes dtype/buffer/
quantization guards now pass. The owner also stops building its previously
unused whole-graph producer map, reducing one full scan per fixed-point round
and lowering the central lowerer's pre-existing Ruff findings from eight to
seven. Valid candidate order, statistics, multiple-post and multiple-chain
behavior, ordered production boundaries, and TensorFlow isolation are
unchanged.

### Mean/MaxPool/Concat/Conv ownership extraction

The corrected 382-line owner now resides in
`passes/mean_maxpool_concat_layout.py`. Its function AST is identical to the
corrected lowerer predecessor at checkpoint `7b0f08a9`. The lowerer retains a
one-return private compatibility wrapper, and the ordered QLinear recovery
sequence plus both production boundaries continue to call the historical name
in the same position. The focused module cannot import the lowerer.

Direct owner and wrapper executions are compared on deep-copied static,
dynamic-batch, multiple-post, multiple-chain, and rejection graphs. They
produce identical statistics and complete ModelIR including constants,
options, quantization, tensors, topology, and metadata. Architecture checks
retain axes ownership, transactional planning, conditional prune, exact
382-line module ownership, the thin wrapper, and the existing ordered
sequence. Extraction removes the now-unused `_quant_scale_count` lowerer import
and preserves its seven remaining Ruff findings.

### Dependency metadata

`uv.lock` now reports the repository version as 2.6.4, matching the current
project metadata. No package was added or removed.

## Compatibility and safety

- Public CLI and Python APIs are unchanged.
- The default backend and artifact names/formats are unchanged.
- Existing private lowerer symbols remain as compatibility wrappers.
- All historical production call positions and ordering are unchanged.
- No new dependency is introduced.
- Direct TFLite conversion and `-cotof` retain the TensorFlow-free boundary.
- ModelIR tensor shape, signature, dtype, quantization, and layout metadata
  behavior is preserved.
- Validation was performed under `uv` and all model inference was strictly
  sequential; no inference worker pool or concurrent model process was used.

## Test coverage

The new focused tests cover:

- static and dynamic Squeeze/ExpandDims cases, including speculative branches;
- exact dynamic pre-operator order and LayoutState validity;
- constant-payload authority, metadata counters, idempotence, and no-op guards;
- all six binary types in both exact permutation directions;
- all six binary types across all four singleton-broadcast branches;
- selected-operand rewiring, exact adapter constants, output metadata, and
  independent quantization cloning;
- existing NumPy-broadcast no-op behavior;
- direct pass owner versus compatibility-wrapper equivalence;
- boundary-map normalization, all dynamic-lineage root families, recursive and
  cyclic lineage, constant termination, and static signature completion;
- boundary-map realignment for same-axis, layout-moved, repeated, insufficient,
  malformed, missing, and rank-mismatched cases;
- direct QDQ Pattern A/B rewrites and guards plus end-to-end A/B/C/D and legacy-
  fan-out closure fixtures;
- quantized Swish primary/post/late/safety/prune phase order, option forwarding,
  statistics aggregation, closure remapping, and wrapper equality;
- all four HardSwish root/gate combinations, public/axis/fan-out guards,
  idempotence, and owner/wrapper equivalence;
- the positive pseudo-LeakyRelu plus Pad legacy boundary, idempotence, and
  direct-owner/private-wrapper equality;
- remap-required and already-NHWC Slice constants, idempotence, public/fan-out/
  shape/permutation guards, and direct-owner/private-wrapper equality;
- exclusive and shared Gather indices, contiguous Slice remapping,
  non-contiguous Slice-to-Gather conversion, idempotence, public/fan-out/axis/
  constant/index guards, and Shape-owner/private-wrapper equality;
- indexed-direct and forced-fallback pre-Add owner/wrapper equality, plus
  Gather/shared-constant, shared-Concat/unary, LeakyReLU, nested affine,
  QLinear, stale-plan, bounded-dispatch, and cleanup-boundary behavior;
- dual-pre-Add positive rewrite, quantization clone, idempotence, public,
  permutation, rank, fan-out, existing-post guards, and owner/wrapper equality;
- terminal affine/FC weight-axis variants, shared affine/weight copy-on-write,
  quantization cloning, idempotence, boundary/permutation/shape/weight/fan-out
  guards, and owner/wrapper equality;
- terminal PReLU/BMM supported alpha ranks, shared alpha/RHS copy-on-write,
  quantization cloning, idempotence, boundary/permutation/shape/RHS/adjoint/
  fan-out guards, one-dimensional alpha rejection, and owner/wrapper equality;
- residual affine/PReLU direct module-owner/private-wrapper full ModelIR
  equality plus the complete indexed SiNet residual contract suite;
- residual affine fan-out multi-branch, legacy-adapter, shared-constant
  copy-on-write, idempotence, public-output no-op, and owner/wrapper equality;
- pre-unary affine fan-out coverage for all seven unary types, two branches,
  shared-constant copy-on-write, idempotence, unsupported/public-boundary
  rejection, and owner/wrapper equality;
- indexed-first pre-Add/Mul/Reshape-suffix dispatch, direct and Mul-constant
  indexed families, forced raw fallback, single-prune behavior, LayoutState,
  GraphIndex, and complete compatibility-owner/wrapper equality;
- indexed-first Swish/plain-unary Reshape-suffix dispatch, indexed immutable
  guards, plain LEAKY_RELU fallback, single-prune/LayoutState cleanup, direct
  legacy fixture, and compatibility-owner/wrapper equality;
- indexed-first static Swish/plain-unary Squeeze-suffix dispatch, atomic and
  bounded indexed behavior, plain/axis-3/dynamic fallbacks, single-prune/
  LayoutState cleanup, and compatibility-owner/wrapper equality;
- one-owner/no-import-cycle architecture boundaries and unchanged production
  call counts.

Latest checkpoint results:

- focused binary adapter and legacy singleton tests: `45 passed`;
- focused static shape-signature owner, legacy fixtures, and ownership selector:
  `21 passed` after adding boundary realignment;
- focused Transpose-QDQ owner, A/B/C/D integrations, and ownership selector:
  `12 passed`;
- focused indexed Swish owners, legacy variants, safety delegation, and
  ownership selectors: `26 passed`;
- focused HardSwish SE owner and architecture selector: `9 passed`;
- complete NHWC Concat float/quantized family corpus: `285 passed`;
- complete flatbuffer-direct architecture suite: `226 passed`;
- NHWC Concat family plus architecture gate: `511 passed`;
- final combined branch gate across all extracted owners, active legacy
  selectors, shape reconciliation, NHWC Concat families, and architecture:
  `616 passed`;
- focused Slice pre/post owner plus architecture selector: `10 passed`;
- complete flatbuffer-direct architecture suite after Slice extraction:
  `227 passed`;
- final branch gate after Slice extraction: `626 passed`;
- focused Shape-extraction owner plus architecture selector: `14 passed`;
- complete flatbuffer-direct architecture suite after Shape extraction:
  `228 passed`;
- final branch gate after Shape extraction: `640 passed`;
- focused pre-Add owners, compatibility fixtures, and architecture suite:
  `256 passed`;
- final branch gate after pre-Add composite extraction: `668 passed`;
- focused dual-pre-Add owner plus architecture suite: `238 passed`;
- final branch gate after dual-pre-Add extraction: `678 passed`;
- focused terminal affine/FC owner plus architecture suite: `243 passed`;
- final branch gate after terminal affine/FC extraction: `692 passed`;
- focused terminal PReLU/BMM owner plus architecture suite: `249 passed`;
- final branch gate after terminal PReLU/BMM extraction: `711 passed`;
- focused terminal affine/PReLU owner, wrapper, and production-order gate:
  `3 passed`;
- changed-file branch regression gate after terminal affine/PReLU extraction:
  `510 passed`;
- focused Mean/Mul/Add owner and production-boundary gate: `2 passed`;
- changed-file branch regression gate after Mean/Mul/Add extraction:
  `512 passed`;
- focused BatchMatMul affine-input owner and production-boundary gate:
  `2 passed`;
- changed-file branch regression gate after BatchMatMul affine-input
  extraction: `514 passed`;
- focused BatchMatMul SE owner and production-boundary gate: `2 passed`;
- changed-file branch regression gate after BatchMatMul SE extraction:
  `516 passed`;
- focused BatchMatMul adjoint owner and production-boundary gate: `2 passed`;
- changed-file branch regression gate after BatchMatMul adjoint extraction:
  `518 passed`;
- focused probable-NHWC axis sanitizer characterization: `4 passed`;
- changed-file branch regression gate after sanitizer characterization:
  `522 passed`;
- focused probable-NHWC owner/wrapper and production-boundary gate: `5 passed`;
- changed-file branch regression gate after sanitizer extraction: `523 passed`;
- focused NCHW/NHWC elementwise roundtrip correction and guards: `3 passed`;
- changed-file branch regression gate after root-metadata correction:
  `526 passed`;
- focused opposite-direction elementwise fan-out owner and architecture gate:
  `2 passed`;
- changed-file branch regression gate after fan-out extraction: `528 passed`;
- focused channelwise broadcast-constant owner/wrapper, GraphIndex,
  convergence, and architecture gates: `9 passed`;
- historical direct-builder rank-three/rank-four broadcast cases: `4 passed`;
- changed-file branch regression gate after broadcast-constant extraction:
  `532 passed`;
- focused Conv/Pool output passthrough characterization and raw-owner gate:
  `11 passed, 1 xfailed`;
- changed-file branch regression gate after Conv/Pool characterization:
  `543 passed, 1 xfailed`;
- focused Conv/Pool atomicity correction and raw-owner gate: `12 passed`;
- changed-file branch regression gate after the atomicity correction:
  `544 passed`;
- focused Conv/Pool owner/wrapper and production-boundary gate: `12 passed`;
- changed-file branch regression gate after Conv/Pool extraction: `544 passed`;
- focused indexed/compatibility Conv/Mul affine owner and architecture gate:
  `12 passed`;
- changed-file branch regression gate after Conv/Mul affine extraction:
  `544 passed`;
- focused Mean/HardSigmoid/MulAdd characterization and raw-owner gate:
  `10 passed, 2 xfailed`;
- changed-file branch regression gate after Mean/HardSigmoid characterization:
  `565 passed, 2 xfailed`;
- focused Mean-axis atomicity correction and raw-owner gate:
  `12 passed, 1 xfailed`;
- changed-file branch regression gate after the Mean-axis correction:
  `567 passed, 1 xfailed`;
- focused public residual-output guard and raw-owner gate: `13 passed`;
- changed-file branch regression gate after the public-output correction:
  `568 passed`;
- focused Mean/HardSigmoid/MulAdd owner/wrapper and architecture gate:
  `14 passed`;
- changed-file branch regression gate after the ownership extraction:
  `569 passed`;
- focused QLinear Concat/Conv characterization and raw-owner gate:
  `16 passed, 3 xfailed`;
- changed-file branch regression gate including the QLinear fixture:
  `587 passed, 3 xfailed`;
- focused QLinear required-output atomicity gate:
  `18 passed, 1 xfailed`;
- changed-file branch regression after required-output prevalidation:
  `589 passed, 1 xfailed`;
- focused QLinear public Dequantize-output guard: `19 passed`;
- changed-file branch regression after the public-output correction:
  `590 passed`;
- focused QLinear Concat/Conv owner/wrapper and architecture gate:
  `20 passed`;
- changed-file branch regression after the ownership extraction:
  `591 passed`;
- focused indexed Conv-input adapter characterization:
  `3 passed, 2 xfailed`;
- changed-file focused branch regression including that characterization:
  `594 passed, 2 xfailed`;
- focused Conv-input source-signature atomicity correction: `5 passed`;
- changed-file focused branch regression after the correction: `596 passed`;
- focused Conv-input adapter owner/wrapper and architecture gate: `8 passed`;
- changed-file focused branch regression after the ownership extraction:
  `599 passed`;
- focused mixed NHWC-input/NCHW-Concat characterization and raw-owner gate:
  `10 passed, 3 xfailed`;
- changed-file focused branch regression including that characterization:
  `606 passed, 3 xfailed`;
- focused mixed-Concat transactional-plan correction: `13 passed`;
- changed-file focused branch regression after the correction: `609 passed`;
- focused mixed-Concat owner/wrapper and architecture gate: `14 passed`;
- changed-file focused branch regression after the ownership extraction:
  `610 passed`;
- focused stale binary-Transpose characterization:
  `5 passed, 1 xfailed`;
- changed-file focused branch regression including that characterization:
  `614 passed, 1 xfailed`;
- focused stale binary source-signature atomicity correction: `6 passed`;
- changed-file focused branch regression after the correction: `615 passed`;
- focused channelwise-constant binary rank characterization:
  `6 passed, 2 xfailed`;
- changed-file focused branch regression including that characterization:
  `615 passed, 2 xfailed`;
- focused channelwise-constant binary rank correction: `8 passed`;
- changed-file focused branch regression after the correction: `617 passed`;
- focused stale binary owner/wrapper and architecture extraction gate:
  `9 passed`;
- changed-file focused branch regression after the ownership extraction:
  `618 passed`;
- focused QLinear SiLU prefix characterization and architecture gate:
  `14 passed, 4 xfailed`;
- changed-file focused branch regression including the characterization:
  `631 passed, 4 xfailed`;
- focused QLinear SiLU transactional correction and architecture gate:
  `22 passed, 1 xfailed`;
- changed-file focused branch regression after the correction:
  `639 passed, 1 xfailed`;
- focused QLinear SiLU consumer deduplication and architecture gate:
  `24 passed`;
- changed-file focused branch regression after consumer deduplication:
  `641 passed`;
- focused QLinear SiLU owner/wrapper and architecture extraction gate:
  `28 passed`;
- changed-file focused branch regression after ownership extraction:
  `645 passed`;
- focused Mean/MaxPool/Concat characterization and architecture gate:
  `17 passed, 9 xfailed`;
- changed-file focused branch regression including the characterization:
  `661 passed, 9 xfailed`;
- focused Mean/MaxPool/Concat transactional correction and architecture gate:
  `29 passed`;
- changed-file focused branch regression after the correction: `673 passed`;
- focused Mean/MaxPool/Concat owner/wrapper and architecture extraction gate:
  `34 passed`;
- changed-file focused branch regression after ownership extraction:
  `678 passed`;
- residual affine/PReLU direct owner plus architecture suite: `233 passed`;
- complete indexed SiNet residual suite: `207 passed`;
- final branch gate after residual affine/PReLU extraction: `713 passed`;
- residual affine fan-out direct owner plus architecture suite: `235 passed`;
- complete indexed SiNet residual suite after fan-out extraction: `207 passed`;
- final branch gate after residual affine fan-out extraction: `716 passed`;
- focused pre-unary affine fan-out and adjacent ownership selector: `12 passed`;
- complete indexed SiNet residual suite after pre-unary extraction:
  `207 passed`;
- final branch gate after pre-unary affine fan-out extraction: `727 passed`;
- complete indexed/compatibility pre-Add/Mul/Reshape-suffix suite:
  `13 passed`;
- final branch gate after pre-Add/Mul/Reshape-suffix extraction: `740 passed`;
- focused indexed/compatibility pre-unary Reshape-suffix gate: `9 passed`;
- final branch gate after pre-unary Reshape-suffix extraction: `748 passed`;
- focused indexed/compatibility pre-unary Squeeze-suffix gate: `9 passed`;
- final branch gate after pre-unary Squeeze-suffix extraction: `756 passed`;
- focused indexed/compatibility ExpandDims gate: `10 passed`;
- changed-file branch regression gate including both historical Case A direct
  fixtures after ExpandDims compatibility extraction: `494 passed`;
- focused indexed/compatibility flatten-HW gate: `9 passed`;
- changed-file branch regression gate after flatten-HW compatibility
  extraction: `500 passed`;
- focused indexed/compatibility attention-QKV gate: `9 passed`;
- changed-file branch regression gate after attention-QKV compatibility
  extraction: `508 passed`;
- old helper versus new owner differential comparison: 250 generated ModelIR
  cases matched in both statistics and every tensor shape signature;
- boundary realigner differential comparison: 250 generated maps matched in
  both statistics and final metadata;
- full old/new Transpose-QDQ owner AST comparison: identical after function-
  name normalization;
- both old/new Swish-QDQ orchestration ASTs: identical after normalizing the
  public function and safety-owner names;
- old/new HardSwish SE owner ASTs: identical after function-name normalization;
- TensorFlow-import-blocked optional-boundary suite, including direct
  conversion and direct `-cotof`: `11 passed`;
- Ruff checks, Python compilation, and whitespace checks: passed.

Earlier checkpoints on this branch also passed their complete focused plus
architecture gates (`226`, `238`, and `237` tests respectively) before each
commit.

## Sequential model validation

The Squeeze and ExpandDims/Squeeze extractions were validated with `GRU.onnx`.
The generated artifacts were byte-identical to their pre-change controls, and
the measured maximum absolute errors remained below `9e-08` with zero
process-tree SWAP.

Both binary adapter extractions used six short real-model controls during
characterization. Their production counters were zero, making them explicit
zero-owner controls; positive behavior is therefore fixed by exhaustive
synthetic fixtures rather than inferred from the corpus.

After each binary checkpoint, `FastestDet.onnx` was converted with
`-tb flatbuffer_direct -cotof` in a single isolated subprocess. The latest run
passed in 3.739 seconds with:

- `evaluation_pass = true`;
- `max_abs = 1.3113021850585938e-05`;
- process-tree peak SWAP = 0 KiB;
- byte-identical float32, float16, and tensor-correspondence artifacts.

The shape-signature extraction additionally used `osnet025_Nx3x256x128.onnx`
because it is a positive real owner of dynamic leading-axis preservation. It
passed in 4.053 seconds with `max_abs = 2.193450927734375e-05`, zero SWAP, and
byte-identical float32, float16, and tensor-correspondence artifacts.

The QDQ extraction used `face_detection_yunet_2023mar_int8.onnx`, whose first
sweep removes nine bridge pairs. It passed after extraction in 5.204 seconds
with `max_abs = 0`, zero SWAP, and byte-identical float32, float16, and tensor-
correspondence artifacts.

No active passing Tier 0-4 model up to 100 MiB contains the complete ONNX
Transpose/Q/DQ/Sigmoid/Mul source family. `dequantize_linear.onnx` was
therefore used only as a bounded zero-owner artifact and known-failure control,
not as evidence of numeric success. Its post-extraction conversion still
exited zero in 12.789 seconds,
recorded zero process-tree SWAP, retained its established
`max_abs=58.7506103515625` and normalized failure signature, and reproduced
byte-identical float32, float16, tensor-correspondence, schema, and generated-
schema artifacts.

The HardSwish SE extraction used
`ssdlite320_mobilenet_v3_large.onnx` as its closest production control. Both
unchanged production invocations were zero-owner calls. Before and after the
move it passed with `max_abs=3.0517578125e-05` and zero process-tree SWAP. Its
float32, float16, tensor-correspondence, schema, and generated-schema files are
byte-identical. `inference_ops15.onnx` and
`mobilenetv3_large_pytorch.onnx` supplied four additional sequential zero-owner
invocations without widening the conversion set further.

The generic pre-Concat legacy extraction used `FastestDet.onnx` as its fixed
artifact control. Its seven measured runtime invocations were all zero for the
float indexed, quantized indexed, and legacy owners. Before and after the move
it passed with `max_abs=1.3113021850585938e-05`, zero process-tree SWAP, and
byte-identical float32, float16, tensor-correspondence, schema, and generated-
schema files. `osnet025_Nx3x256x128.onnx` supplied seven additional sequential
zero-owner invocations and passed with `max_abs=2.193450927734375e-05` and zero
SWAP. The checkpoint does not claim a non-zero production legacy owner; the
positive synthetic compatibility fixture fixes that behavior.

The Slice pre/post extraction used Tier 0 `UM_best_model.onnx` as its fixed
artifact control. Its single measured production call was zero before the
move. Before and after extraction it passed with
`max_abs=2.384185791015625e-07`, zero process-tree SWAP, and byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.
Tier 2 `alike_t_opset11_192x320.onnx` supplied one additional sequential zero-
owner call and passed with `max_abs=2.345442771911621e-05` and zero SWAP. No
non-zero production owner is claimed; the focused synthetic corpus fixes the
positive behavior.

The Shape-extraction move used Tier 2 `retinaface_onnx_dynamic.onnx` as its
positive artifact control. Its three production calls reported `1,0,0` before
and after extraction. Both sequential runs passed with identical
`max_abs=4.4405460357666016e-06`, zero process-tree SWAP, and byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.
Tier 2 `alike_t_opset11_192x320.onnx` supplied three zero-owner calls and
passed with `max_abs=2.345442771911621e-05` and zero SWAP.

The pre-Add composite extraction used `FastestDet.onnx` as its positive
compatibility-fallback artifact control. Its eight composite calls remain
`1,0,0,0,0,0,0,0`, while all eight bounded indexed-owner calls remain zero.
Before and after extraction, sequential `-cotof` passed with identical
`max_abs=1.3113021850585938e-05`, process-tree SWAP zero, and byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.

The dual-pre-Add extraction characterized FastestDet, OSNet, and HumanSeg
strictly sequentially. Each reached the helper once, reported zero rewrites,
and recorded process-tree SWAP zero. FastestDet is the artifact control: its
pre/post conversion-only runs completed in 1.614 and 1.593 seconds and produced
byte-identical float32, float16, tensor-correspondence, schema, and generated-
schema outputs. Its immediately preceding sequential `-cotof` checkpoint had
already passed at `max_abs=1.3113021850585938e-05`; no redundant inference run
was added because the executed TFLite artifact is byte-identical.

The terminal affine/FC extraction used OSNet as its single short artifact
control. Its late helper call remained zero before and after extraction, both
conversion-only runs recorded process-tree SWAP zero, and durations were 2.171
and 2.195 seconds. Float32, float16, tensor-correspondence, schema, and
generated-schema outputs are byte-identical. The immediately preceding OSNet
accuracy baseline remains `max_abs=2.193450927734375e-05`; no redundant
inference run was added because the executed TFLite artifact is unchanged. A
read-only topology scan of root ONNX files up to 50 MiB found no complete raw
source chain, so positive behavior is fixed synthetically.

The terminal PReLU/BMM extraction used `inference_ops15.onnx` as its short
artifact control. Its conditional helper call remained zero before and after
extraction, both conversion-only runs recorded process-tree SWAP zero, and
durations were 1.823 and 1.860 seconds. Float32, float16, tensor-
correspondence, schema, and generated-schema outputs are byte-identical. Its
immediately preceding accuracy baseline remains
`max_abs=1.9073486328125e-06`; no duplicate inference was run because the
executed TFLite is unchanged. A read-only scan of root ONNX files up to 50 MiB
found no complete raw source chain, so positive behavior remains synthetic.

The terminal affine/PReLU extraction used `sinet_320_op.onnx` as its fixed
zero-owner artifact control. Its four runtime results remain `0,0,0,0` before
and after extraction. The pre/post conversion-only runs both completed in
2.413 seconds, recorded process-tree SWAP zero, and produced byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.
The immediately preceding SiNet accuracy baseline remains
`max_abs=2.572051016613841e-09`; no duplicate inference was run because the
executed TFLite is unchanged. The positive owner contract is fixed by the
relocated direct fixture.

The Mean/Mul/Add extraction used `LINEA.onnx` as its fixed zero-owner artifact
control. Its five runtime results remain `0,0,0,0,0` before and after
extraction. The pre/post conversion-only runs completed in 7.869 and 7.868
seconds, recorded process-tree SWAP zero, and produced byte-identical float32,
float16, tensor-correspondence, schema, and generated-schema outputs. The
immediately preceding LINEA accuracy baseline remains
`max_abs=0.002297189086675644`; no duplicate inference was run because the
executed TFLite is unchanged. The positive axis-remap contract is fixed by the
relocated direct fixture.

The BatchMatMul affine-input extraction also used `LINEA.onnx` as its fixed
zero-owner artifact control. Its two runtime results remain `0,0` before and
after extraction. The pre/post conversion-only runs completed in 7.827 and
7.917 seconds, recorded process-tree SWAP zero, and produced byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.
The immediately preceding LINEA accuracy baseline remains
`max_abs=0.002297189086675644`; no duplicate inference was run because the
executed TFLite is unchanged. The positive dual-branch contract is fixed by the
relocated direct fixture.

The BatchMatMul SE extraction also used `LINEA.onnx` as its fixed zero-owner
artifact control. Its two runtime results remain `0,0` before and after
extraction. The pre/post conversion-only runs completed in 7.903 and 7.939
seconds, recorded process-tree SWAP zero, and produced byte-identical float32,
float16, tensor-correspondence, schema, and generated-schema outputs. The
immediately preceding LINEA accuracy baseline remains
`max_abs=0.002297189086675644`; no duplicate inference was run because the
executed TFLite is unchanged. The positive SE contract is fixed by the
relocated direct fixture.

The BatchMatMul adjoint-input extraction used Tier 0
`speech_command_classifier_trained.onnx` as a positive artifact control. Its
two runtime results remain `1,0` before and after extraction. The pre/post
conversion-only runs completed in 0.240 and 0.239 seconds, recorded process-
tree SWAP zero, and produced byte-identical float32, float16, tensor-
correspondence, schema, and generated-schema outputs. A single sequential
pre-move accuracy evaluation passed with
`max_abs=2.86102294921875e-06`, `evaluation_pass=true`, and no skipped output;
no duplicate post-move inference was run because the executed TFLite artifact
is byte-identical.

The probable-NHWC sanitizer extraction used FastestDet as its fixed zero-owner
artifact control. Its four runtime results remain zero before and after
extraction. The pre/post conversion-only runs completed in 0.802 and 0.783
seconds, recorded process-tree SWAP zero, and produced byte-identical float32,
float16, tensor-correspondence, schema, and generated-schema outputs. The
preceding FastestDet accuracy baseline remains
`max_abs=1.3113021850585938e-05`; no duplicate inference was run because the
executed TFLite artifact is unchanged. Positive production ownership is not
claimed; all positive branches are fixed by the four-case focused contract.

The elementwise roundtrip root-metadata correction used Tier 1
`gaze_estimation_adas_0002.onnx` as its fixed zero-owner artifact control. Its
four runtime results remain zero before and after the correction. The pre/post
conversion-only runs completed in 0.398 and 0.395 seconds, recorded process-
tree SWAP zero, and produced byte-identical float32, float16, tensor-
correspondence, schema, and generated-schema outputs. The active Tier 0-4
baseline remains `max_abs=1.2665987014770508e-07`; no duplicate inference was
run because the executed TFLite artifact is unchanged. Positive semantics are
fixed by the focused synthetic closure and guard corpus.

The opposite-direction elementwise fan-out extraction used Tier 0
`shadowformer_istd_160x240_split.onnx` as its fixed zero-owner artifact
control. Its six runtime results remain zero before and after extraction. The
pre/post conversion-only runs completed in 0.259 and 0.261 seconds, recorded
process-tree SWAP zero, and produced byte-identical float32, float16, tensor-
correspondence, schema, and generated-schema outputs. The active Tier 0-4
baseline remains `max_abs=4.0531158447265625e-06`; no duplicate inference was
run because the executed TFLite artifact is unchanged. Positive production
ownership is not claimed; the full fan-out rewrite is fixed by the relocated
focused contract.

The residual affine/PReLU extraction used `sinet_320_op.onnx` as its positive
artifact control. Its fourteen runtime results remain
`0,0,0,1,1,0,0,0,0,0,0,0,0,0` across extraction. The pre/post conversion-only
runs completed in 2.048 and 1.959 seconds, recorded process-tree SWAP zero, and
produced byte-identical float32, float16, tensor-correspondence, schema, and
generated-schema outputs. The immediately preceding SiNet accuracy baseline
remains `max_abs=2.572051016613841e-09`; no duplicate inference was run because
the executed TFLite is unchanged.

The residual affine fan-out extraction also used `sinet_320_op.onnx` as its
fixed artifact control. Its fourteen runtime results are all zero before and
after extraction. The pre/post conversion-only runs completed in 2.438 and
2.587 seconds, recorded process-tree SWAP zero, and produced byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.
The immediately preceding SiNet accuracy baseline remains
`max_abs=2.572051016613841e-09`; no duplicate inference was run because the
executed TFLite is unchanged. The positive owner contract is synthetic.

The pre-unary affine fan-out extraction again used `sinet_320_op.onnx` as the
minimal fixed artifact control. Its five runtime results are all zero before
and after extraction. The pre/post conversion-only runs completed in 2.430 and
2.503 seconds, recorded process-tree SWAP zero, and produced byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.
The immediately preceding SiNet accuracy baseline remains
`max_abs=2.572051016613841e-09`; no duplicate inference was run because the
executed TFLite is unchanged. The positive owner contract is synthetic and the
earlier broader zero-owner characterization remains authoritative.

The pre-Add/Mul/Reshape-suffix composite extraction used
`iat_llie_180x320.onnx` as its positive fixed artifact control. Its three
combined and indexed counts remain `5,4,4`, while raw fallback counts remain
`0,0,0`, before and after extraction. The pre/post conversion-only runs
completed in 2.162 and 2.074 seconds, recorded process-tree SWAP zero, and
produced byte-identical float32, float16, tensor-correspondence, schema, and
generated-schema outputs. The preceding sequential accuracy checkpoint remains
`max_abs=4.470348358154297e-07`; no duplicate inference was run because the
executed TFLite is unchanged.

The pre-unary Reshape-suffix composite extraction used `LINEA.onnx` as its
positive fixed artifact control. Its combined and indexed counts remain
`1,0,0`, while raw fallback counts remain `0,0,0`, before and after extraction.
The pre/post conversion-only runs completed in 7.958 and 7.751 seconds,
recorded process-tree SWAP zero, and produced byte-identical float32, float16,
tensor-correspondence, schema, and generated-schema outputs. The preceding
sequential accuracy checkpoint remains `max_abs=0.002297189086675644`; no
duplicate inference was run because the executed TFLite is unchanged.

The pre-unary Squeeze-suffix composite extraction used
`inference_ops15.onnx` as its positive fixed artifact control. Its combined and
indexed counts remain `1,0,0`, while raw fallback counts remain `0,0,0`, before
and after extraction. The pre/post conversion-only runs completed in 2.280 and
2.266 seconds, recorded process-tree SWAP zero, and produced byte-identical
float32, float16, tensor-correspondence, schema, and generated-schema outputs.
The preceding sequential accuracy checkpoint remains
`max_abs=1.9073486328125e-06`; no duplicate inference was run because the
executed TFLite is unchanged.

The factorized/singleton ExpandDims composite extraction used `yolo_test.onnx`
as its positive fixed artifact control. Its established indexed counts remain
`3,0,0,0`, with no residual raw-fallback rewrites. The pre/post conversion-only
runs completed in 3.347 and 3.378 seconds, recorded process-tree SWAP zero, and
produced byte-identical float32, float16, tensor-correspondence, schema, and
generated-schema outputs. The preceding sequential accuracy checkpoint remains
`max_abs=2.4437904357910156e-06`; no duplicate inference was run because the
executed TFLite is unchanged.

The static/dynamic flatten-HW composite extraction used `LINEA.onnx` as its
positive fixed artifact control. Its established indexed invocation counts
remain `2,0,0,0`, with no residual raw-fallback rewrites for those accepted
candidates. The pre/post conversion-only runs completed in 7.758 and 7.975
seconds, recorded process-tree SWAP zero, and produced byte-identical float32,
float16, tensor-correspondence, schema, and generated-schema outputs. The
preceding sequential accuracy checkpoint remains
`max_abs=0.002297189086675644`; no duplicate inference was run because the
executed TFLite is unchanged.

The static/relaxed attention-QKV composite extraction used
`rf-detr-nano.onnx` as its positive fixed artifact control. Its established
indexed invocation counts remain `5,0,0,0`, with no residual raw-fallback
rewrites for those accepted candidates. The pre/post conversion-only runs
completed in 10.928 and 11.288 seconds, recorded process-tree SWAP zero, and
produced byte-identical float32, float16, tensor-correspondence, schema, and
generated-schema outputs. The preceding sequential accuracy checkpoint remains
`max_abs=0.000102996826171875`; no duplicate inference was run because the
executed TFLite is unchanged.

The channelwise broadcast-constant extraction used `FastestDet.onnx` as its
strictly sequential zero-owner artifact control. Its five runtime results are
all zero before and after extraction. The post conversion completed in 0.815
seconds and recorded process-tree SWAP zero. The pre/post float32, float16,
tensor-correspondence, schema, and generated-schema artifacts are byte-
identical. Their SHA-256 values remain respectively
`3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b`,
`a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617`,
`2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074`,
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
The preceding FastestDet accuracy baseline remains
`max_abs=1.3113021850585938e-05`; no duplicate inference was run because the
executed TFLite artifact is unchanged. Positive production ownership is not
claimed; the focused synthetic cases are the semantic authority.

The Conv/Pool output passthrough helper is now mechanically extracted.
FastestDet, HumanSeg, OSNet, and inference_ops15 each reached its single
production position with a zero rewrite result. Their characterization runs
completed in 0.789, 0.513, 1.239, and 0.764 seconds respectively, and every
process-tree SWAP monitor reported zero. The extraction-specific FastestDet
pre/post runs completed in 0.788 and 0.807 seconds with zero SWAP and byte-
identical float32, float16, tensor-correspondence, schema, and generated-schema
artifacts. Positive production ownership is not claimed; the focused synthetic
corpus is the semantic authority.

The characterization exposed one pre-existing atomicity defect: the helper
rewired its elementwise root before rejecting a non-rank-four external runtime
input. The following narrow correction now validates every external runtime
tensor and computes its NHWC shape before the first graph or metadata mutation.
The former strict xfail is a passing multi-input atomicity contract; successful
rewrite order, statistics, and serialized artifacts are unchanged.

The Conv/Mul affine orchestration extraction used Tier 2
`iat_llie_180x320.onnx` as its positive artifact control. Its three ordered
results remain `12,0,0`: all twelve first-pass folds are indexed Mul-only folds
and the raw fallback has no residual rewrite. Pre/post conversion-only runs
completed in 0.808 and 0.791 seconds, recorded process-tree SWAP zero, and
produced byte-identical float32, float16, tensor-correspondence, schema, and
generated-schema artifacts. The preceding accuracy checkpoint remains
`max_abs=4.470348358154297e-07`; duplicate inference was not run because the
executed TFLite artifact is identical.

The quantized Mean/HardSigmoid/MulAdd helper is now owned by the focused
`mean_hardsigmoid_muladd_layout.py` module. The full synthetic graph fixes both
quantized branches, Mean-axis remap, decomposed HardSigmoid clamp, residual
Mul/Add rewiring, bridge removal, legacy adapter insertion, idempotence, and
eight rejection boundaries. Both pre-existing defects are fixed. Mean axes,
including a rejected no-change constant update, are validated before any graph
rewiring or dependent metadata mutation. A declared public residual output
rejects the candidate before the axes write, preserving its NCHW boundary and
the complete ModelIR fingerprint.

YuNet INT8, PPHumanSeg INT8, and SSD MobileNet INT8 each reached both ordered
runtime boundaries with zero rewrites. Their strictly sequential conversions
completed in 1.007, 1.715, and 2.107 seconds, every process-tree monitor
reported SWAP zero, and all conversions succeeded. This matches the earlier
broader zero-owner survey; positive production ownership is not claimed.

The Mean-axis correction used YuNet INT8 as a strictly sequential before/after
artifact control. Both runs passed with `max_abs=0` and process-tree SWAP zero
in 5.280 and 5.320 seconds. Pass metrics are identical; float32/float16 TFLite,
tensor-correspondence, op-error CSV, schema, and generated-schema artifacts are
byte-identical. The remaining JSON differences contain output or temporary
paths only. The TensorFlow-import-blocked optional-boundary suite remains
`11 passed`.

The public-output correction repeated the same strictly sequential YuNet INT8
control. Before and after runs passed with `max_abs=0`, zero process-tree SWAP,
and durations of 6.323 and 5.148 seconds. Internal pass metrics and all
non-path artifact content remain identical, including byte-identical float32
and float16 TFLite files.

The final ownership move is mechanical. The old 496-line lowerer function and
new owner have identical ASTs after normalizing only the function name. The
lowerer retains a two-line private wrapper, its single syntactic call, and both
ordered runtime boundaries. A third strictly sequential YuNet INT8 before/
after control passed with `max_abs=0`, zero SWAP, and durations of 6.290 and
5.215 seconds. Pass metrics and all non-path artifact content remain identical.

The corrected 612-line QLinear Concat/Conv propagation helper is now owned by
`qlinear_concat_conv_compat.py`. Its function body is AST-identical to the
corrected lowerer predecessor after normalizing only the function name. The
lowerer retains a two-line private wrapper, its single syntactic call, and both
ordered runtime boundaries. Dedicated tests cover all four input adapter forms,
multiple post-Quantize adapters, an additional direct Concat adapter, dynamic
batch signatures, per-axis quantization-dimension remap, idempotence, and nine
complete no-op guards. The former 119-line giant ModelIR fixture moved to the
focused qlinear module with an identical AST. Required Concat and Quantize
output tensors are validated before the first candidate mutation, making both
missing-tensor cases complete no-ops. A public tensor that would receive a
pending layout metadata update rejects before mutation, while an already-NHWC
public Dequantize output remains eligible. Direct owner and compatibility-
wrapper calls produce identical complete ModelIR fingerprints and statistics.
No strict xfail remains.

The required-output correction used YuNet INT8 as a strictly sequential before/
after artifact control. Both runs passed with `max_abs=0`, zero process-tree
SWAP, and durations of 6.329 and 5.291 seconds. Internal pass metrics and all
non-path artifact content remain identical, including byte-identical float32
and float16 TFLite files.

The public-output correction repeated the YuNet INT8 control. Before and after
runs passed with `max_abs=0`, zero process-tree SWAP, and durations of 6.426 and
5.259 seconds. Internal pass metrics and all non-path artifact content remain
identical.

The ownership extraction repeated the strictly sequential YuNet INT8 control
at the corrected `e2ccb4ac` checkpoint and after the move. Both runs passed with
`max_abs=0`, zero process-tree SWAP, and durations of 6.389 and 5.279 seconds.
Internal pass metrics are exact. Float32/float16 TFLite, tensor-correspondence,
op-error CSV, schema, and generated-schema artifacts are byte-identical; the
three JSON differences contain output or temporary paths only.

The corrected Conv-input adapter repair pair and shared-index runner are now
owned by `conv_input_adapter_repair.py`. Their 104-, 122-, and 23-line function
bodies are individually AST-identical to the corrected lowerer predecessors.
The lowerer retains private compatibility wrappers for all three names, the
primary and fallback runner calls, and the later standalone stale-Transpose
call. Direct owners and wrappers produce identical complete ModelIR
fingerprints and statistics for singleton-Reshape, stale-Transpose, and grouped
execution.

Tier 1 `face_blendshapes.onnx`, the historical positive singleton-repair model,
was run strictly sequentially at corrected checkpoint `a76ad6ff` and after the
move. Both runs passed with `max_abs=1.3709068298339844e-06`, zero process-tree
SWAP, and durations of 4.933 and 3.867 seconds. Pass metrics are exact. Float32/
float16 TFLite, tensor-correspondence, op-error CSV, schema, and generated-
schema artifacts are byte-identical; the three JSON differences contain output
or temporary paths only.

The following mixed NHWC-input/NCHW-Concat repair now builds a complete adapter
plan before mutation. It resolves the required output tensor, materializes all
source signatures, reserves collision-free names, computes adapter/output
metadata, clones quantization, and remaps a per-axis NHWC dimension to NCHW.
Only a fully valid plan commits tensors, Transpose operators, Concat inputs, and
output metadata in the historical order. The three former strict xfails are
green.

Tier 3 `sgscsh.onnx`, the historical positive mixed-Concat repair model, was
run strictly sequentially at characterization checkpoint `ec9f6bf0` and after
the correction. Both runs passed with
`max_abs=2.5331974029541016e-07`, zero process-tree SWAP, and durations of
15.216 and 14.375 seconds. Pass metrics are exact. Float32/float16 TFLite,
tensor-correspondence, op-error CSV, schema, and generated-schema artifacts are
byte-identical; the three JSON differences contain output or temporary paths
only.

The corrected 223-line mixed-Concat owner is now in
`mixed_concat_input_repair.py`. Its body is AST-identical to the corrected
lowerer predecessor, while the lowerer retains a two-line private wrapper and
both fallback/final production calls. Direct owner and wrapper execution on a
multi-adapter ModelIR produces identical complete fingerprints and statistics.
A second strictly sequential `sgscsh.onnx` control at corrected checkpoint
`55f1a541` and after extraction passed with the same
`max_abs=2.5331974029541016e-07`, zero SWAP, and durations of 15.582 and 14.434
seconds. Pass metrics and all non-path artifact content are exact.

## Scope and follow-up

This branch deliberately avoids semantic generalization and does not claim a
performance improvement for the two binary compatibility rules. Their
mechanical ownership is established first. A future differential-index rewrite
must independently prove candidate order, restart behavior, pruning behavior,
and non-zero ownership before replacing the current insertion logic.

The adjacent 218-line rank-3-to-NHWC reshape helper, 293-line attention
Gather/Transpose/Reshape cleanup, and 190-line attention pre-projection rank-
lift remain intentionally unchanged under their recorded no-owner decisions.
The corrected 207-line
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains` helper now has a
dedicated positive and rejection contract, but the structurally matching gaze
model still records zero production rewrites. Do not mechanically extract it
until positive production ownership is observed or a later checkpoint
explicitly accepts zero-owner evidence. The 496-line
`_optimize_transpose_mean_hardsigmoid_muladd_chains` helper and the following
612-line `_optimize_nhwc_propagation_qlinear_concat_conv` helper have now
completed separately approved zero-owner mechanical extractions. Their exact
bodies, wrapper signatures, production call order, statistics, and artifact
controls are preserved. Resume by inventorying and characterizing the next raw
source-order owner before changing it. That boundary is the existing indexed
Conv-input adapter repair pair and its shared-index runner. Characterization
found that both repair functions read a malformed source `shape_signature`
after rewriting the Conv input; the resulting `IndexError` leaves a partial
mutation. Both repairs now materialize and rank-validate that signature before
the first indexed edge mutation, so both former strict xfails are green with a
zero statistic and complete ModelIR no-op. Their exact mechanical ownership
move is complete with bodies, wrappers, sequence, shared-index behavior,
statistics, and production boundaries preserved. Resume by characterizing the
next raw source-order owner,
`_repair_mixed_nhwc_inputs_for_nchw_concat`, before changing it. That
characterization now freezes its three-input and two-input output-contract
success paths, idempotence, four no-op guards, two production calls, and three
pre-existing defects as strict xfails: partial insertion before a later invalid
source signature, mutation without a required Concat output tensor, and stale
per-axis quantized dimension after NHWC-to-NCHW adaptation. A complete
prevalidation plan now corrects all three paths with no remaining strict xfail.
The mechanical ownership extraction is complete with the corrected body, two
production calls, statistics, and artifact control preserved. Resume by
inventorying the next raw source-order owner before changing it. That 132-line
stale channelwise-binary Transpose repair already has indexed positive,
multiple-match, fan-out, and convergence-runner coverage. A new malformed
source-signature characterization is a strict xfail because the repair rewires
the binary input before assigning a rank-two signature to a rank-four output.
The source signature is now materialized and rank-validated before the indexed
setter, so the former strict xfail is green with a zero statistic and complete
ModelIR no-op. A final branch-specific audit found that the channelwise-
constant matcher still indexes short source/adapter shapes before its rank
guard; both rank-four checks now precede channel evidence, so the two former
strict xfails are green complete no-ops. The corrected 132-line owner is now
mechanically extracted to `passes/stale_binary_adapter_repair.py` with an
AST-identical body, a lowerer compatibility wrapper, both standalone calls,
and the central three-round convergence runner preserved. The next raw
source-order owner, `_optimize_nhwc_prefix_qlinear_silu_chains` (419 lines),
now has positive, multiple-match, legacy-adapter, rejection, idempotence,
collision, malformed-signature, and architecture characterization. Resume by
correcting its four strict xfails without changing valid candidate order,
statistics, production call boundaries, or artifacts. That correction is now
complete: metadata and adapter planning precede mutation, permutation creation
is lazy and collision-safe, and all four xfails are green. One newly isolated
strict xfail remains for duplicate planning when the same legacy consumer uses
the Mul output in two input slots. Consumer indices are now deduplicated in
first-observed order; same-consumer multi-slot and distinct-consumer order
fixtures are green, and no strict xfail remains. Resume by mechanically
extracting the corrected 513-line owner into a focused pass module. That move
is now complete with exact AST identity, a one-return private wrapper, direct
owner/wrapper equality, and unchanged ordered production boundaries. Resume by
inventorying and characterizing the next raw source-order owner,
`_optimize_transpose_mean_maxpool_concat_conv_chains` (310 lines), before any
semantic or ownership change. That characterization now freezes positive,
fixed-point, rejection, metadata, quantization, pruning, and ordered-boundary
behavior and isolates nine strict xfails. Resume by enforcing local immutable
axes ownership and building a complete rank-four branch/Concat metadata plan
before the first edge, constant, option, or tensor mutation. That correction is
now complete, all xfails are green, and the unused producer-map scan is gone.
Resume by mechanically extracting the corrected 382-line owner into a focused
pass module while preserving its exact AST, private lowerer name, ordered
recovery sequence, statistics, and both production boundaries. That extraction
is now complete with direct owner/wrapper equality and seven lowerer Ruff
findings preserved. Resume by inventorying and characterizing the next raw
source-order owner, `_canonicalize_softmax_transpose_chains` (190 lines),
before any semantic or ownership change. No broad conversion sweep is implied
by this checkpoint. Its characterization is now complete without production
source changes: 16 positive, rejection, alias, pruning, fixed-point, and
ordered-boundary cases pass, while 24 concrete safety cases are recorded as
strict xfails. They isolate incomplete Softmax-output metadata propagation,
unsafe non-last or malformed axis acceptance, missing/rank-invalid metadata,
mutable or public permutation ownership, public constant-output mutation,
duplicate producers, reverse topology, and public internal aliases. Resume by
building the full topology/metadata/permutation plan before the first mutation
and turning every strict xfail green while preserving valid statistics and
both production boundaries. That transactional correction is now complete.
One shared `ModelIRGraphIndex` replaces repeated full producer/consumer scans;
last-axis semantics, unique topological ownership, complete rank-four
metadata, and immutable INT32 permutation ownership are prevalidated. Both
permutation actions, clone names, Softmax input/output metadata, post-
Transpose metadata, and marker options are planned before commit. All 24
former strict xfails are green, including public-output cloning and post-plan
atomicity, while normalized axis `-1`, valid statistics, pruning, fixed-point
behavior, and both ordered boundaries remain intact. Resume by mechanically
extracting the corrected 343-line owner into a focused pass module. That move
is now complete with an AST-identical module owner, a one-return private
lowerer wrapper, direct owner/wrapper equality across ten positive and
rejection families, the shared marker import, and both ordered production
positions preserved. Seven pre-existing lowerer Ruff findings remain.
The next raw source-order owner is the 394-line
`_optimize_concat_mul_add_transpose_nhwc_bridge_chains`, which already has
ordinary and legacy-consumer public fixtures and must be characterized before
editing. That characterization is now complete without production changes.
The two public fixtures moved to a focused module, reducing the giant direct
test by 211 lines; 18 positive/rejection/boundary cases pass and 16 concrete
safety cases are strict xfails. The latter isolate a non-topological legacy
adapter, incomplete metadata, mutable/public constant ownership, missing
per-axis QDIM remaps, reserved adapter-name overwrite, late partial mutation,
duplicate producers, reverse order, and public internal aliases. Resume by
building the complete topology/metadata/constant/adapter plan before mutation
and turning all strict xfails green. That correction is now complete in the
652-line raw owner. A single differential `ModelIRGraphIndex`, complete
rank-four and topological preflight, ownership-aware constant rotation/cloning,
NCHW-to-NHWC QDIM remapping, private collision-safe adapter constants, and
producer-before-consumer adapter insertion turn all 16 xfails green. Valid
statistics, fixed point, public Concat outputs, pruning behavior, and both
ordered boundaries remain intact. Resume by mechanically extracting the
corrected owner into a focused pass module; six pre-existing lowerer Ruff
findings remain. That extraction is now complete in
`passes/concat_mul_add_bridge_layout.py`. Its 652-line function AST is exact to
checkpoint `5193fc11`, the lowerer retains a one-return compatibility wrapper,
and both ordered recovery-sequence positions are unchanged. Fifteen direct
owner/wrapper comparison families cover valid static/dynamic and multi-match
rewrites, constant and adapter ownership, per-axis quantization, pruning, and
transactional rejection; complete normalized ModelIR state and statistics are
identical. Resume by inventorying and characterizing the next raw source-order
owner, `_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains` (452 lines),
before any semantic or ownership change. That characterization is now
complete without production changes. Its legacy fixture moved out of the
giant direct test, 18 positive/rejection/boundary cases pass, and 27 strict
xfails isolate adapter order, incomplete metadata, affine-constant ownership,
per-axis QDIM, unsafe reserved adapter constants, late partial mutation,
malformed axis handling, duplicate producers, reverse topology, and a public
internal alias. Resume by building a complete indexed topology, metadata,
constant, quantization, name, adapter, removal, and insertion plan before the
first mutation, then turn all 27 xfails green while preserving valid behavior
and both ordered recovery boundaries.
