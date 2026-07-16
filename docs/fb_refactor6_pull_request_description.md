# Refactor flatbuffer-direct terminal shape and binary layout passes

## Summary

This branch continues the staged `flatbuffer_direct` refactor by moving eleven
fully characterized compatibility rules out of the central ONNX-to-ModelIR
lowerer and into focused pass modules:

- static Squeeze shape sanitization;
- LiteRT.js ExpandDims/Squeeze-to-Reshape conversion;
- exact rank-four binary NHWC/NCHW adaptation;
- singleton-broadcast rank-four binary NHWC/NCHW adaptation;
- final static runtime-shape/signature consistency;
- dynamic boundary-signature map realignment;
- Transpose/QDQ bridge and residual-closure optimization;
- quantized Swish NHWC-island and residual-Concat orchestration;
- HardSwish/SE/HardSigmoid gating-block layout recovery;
- the remaining generic NHWC pre-Concat compatibility matcher;
- strict rank-four Transpose/Slice/inverse-Transpose passthrough.

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

## Scope and follow-up

This branch deliberately avoids semantic generalization and does not claim a
performance improvement for the two binary compatibility rules. Their
mechanical ownership is established first. A future differential-index rewrite
must independently prove candidate order, restart behavior, pruning behavior,
and non-zero ownership before replacing the current insertion logic.

The next raw source-order boundary is the 285-line Shape-extraction NHWC→NCHW
helper. Its Gather/Slice remapping families, constant materialization,
production positions, and real-model ownership must be characterized before
extraction; no broad conversion sweep is implied by this mechanical checkpoint.
