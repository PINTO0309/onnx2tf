# Refactor flatbuffer-direct terminal shape and binary layout passes

## Summary

This branch continues the staged `flatbuffer_direct` refactor by moving four
fully characterized compatibility rules out of the central ONNX-to-ModelIR
lowerer and into focused pass modules:

- static Squeeze shape sanitization;
- LiteRT.js ExpandDims/Squeeze-to-Reshape conversion;
- exact rank-four binary NHWC/NCHW adaptation;
- singleton-broadcast rank-four binary NHWC/NCHW adaptation.

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
- one-owner/no-import-cycle architecture boundaries and unchanged production
  call counts.

Latest checkpoint results:

- focused binary adapter and legacy singleton tests: `45 passed`;
- complete flatbuffer-direct architecture suite: `221 passed`;
- final combined branch gate across all extracted owners, active legacy
  selectors, shape reconciliation, and architecture: `280 passed`;
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

## Scope and follow-up

This branch deliberately avoids semantic generalization and does not claim a
performance improvement for the two binary compatibility rules. Their
mechanical ownership is established first. A future differential-index rewrite
must independently prove candidate order, restart behavior, pruning behavior,
and non-zero ownership before replacing the current insertion logic.

The next candidate for characterization is the adjacent
`_sanitize_static_shape_signature_consistency` helper. It is outside this
branch's completed checkpoint.
