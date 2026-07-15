# flatbuffer_direct architecture

`flatbuffer_direct` keeps its public CLI and Python API in `onnx2tf.onnx2tf`.
New implementation work belongs behind the internal contracts in
`onnx2tf.tflite_builder.core`; those contracts must not be exported as public
API.

## Pipeline

The required stage order is:

1. normalize the public arguments into `ConversionRequest` and `ArtifactPlan`;
2. preprocess ONNX;
3. create one `ConversionSession`, `GraphIndex`, and `LayoutState`;
4. execute registered passes in `PassPhase` order;
5. lower operators into ModelIR;
6. validate ModelIR invariants;
7. invoke only exporters selected by `ArtifactPlan`;
8. adapt `ConversionResult` back to the legacy return dictionary.

Optional artifact work is guarded at the call site as well as inside each
exporter. In particular, op-coverage report writing is invoked only when
`ArtifactPlan.op_coverage_report` is true on success and both failure paths;
an unrequested report does not enter even the no-op wrapper. Export progress
advances for SavedModel only inside the SavedModel request guard, so the runtime
step count remains aligned with the request-derived label count.
Split thresholds and quantization calibration controls are resolved by the
TensorFlow- and Torch-free artifact policy only when the corresponding
`ArtifactPlan` flag is set. An unrequested artifact does not read or parse its
CLI values or environment variables, while requested values retain the legacy
conversion and default semantics. The resolved quantization mapping is
immutable and raw option dictionaries do not propagate into builders.
The same guarded quantization mapping owns `quant_type`, input quant dtype, and
output quant dtype. Their legacy `per-channel`/`int8` defaults are materialized
without reading the request when no quantized output is selected; requested
dynamic-range or integer artifacts receive the original explicit values.
The direct export boundary constructs `ConversionRequest` once. Every option
read after that boundary, including unsupported-quantization validation, uses
the request's immutable mapping or typed `ArtifactPlan`; the original `kwargs`
object is referenced only as the input to `ConversionRequest.from_kwargs`.
This keeps legacy keys and default values intact while preventing later
enhancements from bypassing normalization and reintroducing raw option
propagation.
SavedModel- and PyTorch-specific preparation options use a second requested-
exporter control resolver in `artifact_preparation.py`. It does not read output
paths, persistence, native PyTorch timeout, shape hints, or test data unless
the corresponding artifact is requested. The custom input data option is read
only for integer calibration or PyTorch-derived artifacts. Consequently, an
invalid unused PyTorch timeout cannot fail a TFLite-only conversion, while a
requested artifact retains the existing default paths, coercions, and error
behavior.

Both artifact-control resolvers and export-progress planning receive the
normalized `ArtifactPlan` itself rather than parallel request booleans. Split,
quantization, SavedModel, PyTorch, and integer-calibration dependencies are
therefore derived from one immutable object. TorchScript, Dynamo ONNX, and
ExportedProgram continue to imply the base PyTorch artifact only in
`ArtifactPlan.from_options`; downstream preparation cannot accidentally omit
that dependency or read options for an unrequested artifact. The default
correspondence report and float32/float16 outputs retain their compatibility
order, and all optional progress labels come directly from the same plan used
to guard execution.

The compatibility layer treats the direct builder's returned artifact mapping
as the only source of TFLite paths for accuracy evaluation. The seven legacy
variant keys and their stable evaluation order live in
`artifact_metadata.select_tflite_evaluation_artifact_paths`; the terminal
direct conversion path calls that owner once. Missing optional variants remain
absent instead of being reconstructed from output-directory conventions, so
report execution follows the actual `ConversionResult` artifacts.

The same compatibility boundary has one validation and completion-log owner
for OP coverage, tensor correspondence, dynamic-range quantization, integer
quantization, and both int16-activation variants. The sole direct-result
finalizer calls that owner, preserving required artifact keys, int16 skip
semantics, and the six established failure messages. Tensor correspondence
remains a compatibility-default artifact and is logged whenever the builder
returns it.

The direct fast path is terminal: successful finalization returns from
`convert`, while every failure propagates through its cleanup `finally` block.
The backend is normalized once, and an explicit post-boundary assertion limits
the remaining legacy graph, SavedModel, and TFLite-converter pipeline to
`tf_converter`. Historical direct retries after TensorFlow node-conversion or
SavedModel failure and the duplicate direct TFLite serialization block were
therefore unreachable and have been removed. Direct builder invocation,
artifact evaluation selection, and direct-result finalization now each have
one live compatibility-layer call site.

A pass has a stable ID, phase, priority, maximum iteration count, and explicit
`changed` result. Repeating passes must use a graph fingerprint so a cycle
terminates deterministically. Risky rewrites use the transactional mode and
must leave the graph unchanged when invariant validation fails.

`ModelIRPassState` also provides one-shot prepared pass data for expensive
preconditions that produce the exact object consumed by their callback. The
data belongs to one session, is removed on first consumption, and is cleared
on rollback, so it cannot leak across conversions or survive restored graph
objects.

`ModelIRPassStateScope` is the explicit reuse boundary for adjacent registered
pass groups that have no raw ModelIR mutation between them. It is lazy, so a
sequence whose model-only preflights all fail still constructs no graph index.
The first matching group constructs one `ModelIRPassState`; later groups reuse
its differentially maintained `ModelIRGraphIndex` and `LayoutState`. A scope
rejects a different ModelIR or LayoutState identity and must end before any
legacy/raw mutator. The repeated Mean/LayerNorm/terminal-Mean/SE/Conv-attention
cluster is the first production consumer, preserving its original runner order
while replacing up to seven identical index constructions with one.

The repeated mixed-attention/elementwise-gate/Pad/dual-postconv-gate/NDHWC-
gate/cost-volume-scatter/Add-Concat-suffix/dual-Mul-Concat sequence is the
second production consumer. Four occurrences retain all eight runners and one
occurrence retains the original seven-runner suffix without mixed attention.
Each occurrence now constructs at most one index instead of as many as eight.
Later isolated calls remain standalone because the scope must not cross the
legacy ModelIR mutators surrounding them.

The late NDHWC-gate/cost-volume-scatter pair uses a separate two-runner scope.
The preceding mixed-attention runner is intentionally excluded because a
legacy dequantize/HardSigmoid/quantize mutator separates it from the pair. The
scope ends before the following raw convolution-affine fold.

Immediately after that fold, the axis-3 constant-Concat, Dequantize/Concat/
Quantize, LayerNorm-statistics, and generic transpose-cleanup runners share a
third late scope. All four implementations mutate through the differential
index. The scope ends before the conditional legacy elementwise-roundtrip
optimizer.

Five repeated two-way/NHWC/NCHW channel-shuffle plus Gather-axis sequences
share one scope per occurrence. The final occurrence preserves its contiguous
generic transpose and unary fan-out suffix inside the same scope, while the
other four scopes end before their next legacy mutator. Four separate repeated
unary-passthrough/unary-fan-out/binary-fan-out sequences likewise share one
scope per occurrence.

Four repeated boundary-input BatchMatMul followed by leading-input unary
passthrough pairs share one scope per occurrence. Each scope is limited to the
two adjacent registered runners; the legacy layout transforms before and after
each occurrence remain hard boundaries.

Three repeated strict channel-slice-merge followed by Pad/Mul cleanup pairs
share one scope per occurrence. The three-spec channel-slice group and the
single-spec Pad/Mul group keep their original order and diagnostic grouping.

Two long terminal singleton/Reshape cleanup sequences share one scope per
occurrence through a flag-controlled helper. The helper retains the original
relative order of generic transpose cleanup, singleton-channel Transpose
canonicalization, optional reshape-only duplicate fan-out cleanup, singleton
Reshape cleanup, singleton MaxPool cleanup, flatten/Concat/Reshape cleanup,
general consecutive Reshape cleanup, Squeeze/Reshape identity cleanup,
singleton-spatial Reshape cleanup, and optional multi-branch gate cleanup.
One occurrence keeps the leading generic transpose and terminal gate; the
other keeps the duplicate-fan-out pass and disables the spatial post-Concat
variant exactly as before. Three later singleton-channel/duplicate-fan-out/
consecutive-Reshape triplets use a smaller target-parameterized helper, so the
fallback ModelIR receives its own identity-bound scope without inheriting the
primary Session layout state. No scope crosses the legacy rewrites around
these sequences. The separate terminal singleton-MaxPool/consecutive-Reshape
pair uses its own scope between the conditional elementwise-roundtrip and
Conv/Pool-output legacy rewrites. Its two runners retain their order and
three-spec diagnostic grouping while constructing one graph index.

The terminal scalar-clamp, unary-passthrough, and maximum-zero-to-ReLU
sequence shares one scope between the conditional terminal layout-recovery
block and the raw SiNet rewrites. Clamp and maximum-zero canonicalization use
`ModelIRGraphIndex.replace_operator_type()` so the shared operator-type index
remains current after changing `MAXIMUM`/`MINIMUM` operators to their RELU
forms. Both runners retain standalone behavior through optional scope
arguments.

The conditional generic-Transpose, late Mean/Mul/Add/Conv, generic SPP,
Gather-axis, constant-fold, and redundant-Cast sequence shares one scope
between the raw shape-extract and ExpandDims/Squeeze replacement rewrites.
When layout optimization is disabled, the same helper skips only generic
Transpose and lets the Mean runner lazily construct the state. Generic SPP
cleanup and the constant-fold/Cast helper accept the shared scope; their
existing differential mutations keep the index current across all nine pass
events without a blanket refresh.

The late generic SPP and Concat/unary/Conv pair shares one scope between the
raw StridedSlice/Pad/Concat and shape-extract rewrites. Concat/unary/Conv
cleanup now accepts an optional scope, and its existing differential input and
operator-removal updates keep the shared index current.

The absolute-final flattened-normalization Pad and mixed-attention pair shares
one scope between the raw InstanceNorm bias-add rewrite and dynamic-rank
Unsqueeze/Reshape shape rewrite. Normalization-Pad cleanup now accepts an
optional scope. The helper preserves `include_instance=False` and
`include_flatten=True`, so the final invocation still runs only the intended
flattened normalization spec before mixed attention.

The post-QDQ layout-transpose, unary-fan-out, and unary/binary-fan-out
sequence reuses the existing unary-fan-out helper in a second mode. The helper
defaults still run unary passthrough for its four prior call sites; the new
call enables generic transpose cleanup and disables unary passthrough. Its
scope remains between raw Softmax canonicalization and transpose-binary
rewrites. Consolidation reduces the lowerer's registered-runner call
characterization from 139 to 137.

The late NCHW channel-shuffle and Gather-axis pair reuses the existing
channel-shuffle helper with its two-way and NHWC modes disabled. Five prior
helper invocations retain both modes by default. NHWC and NCHW shuffle
rewrites now use `replace_operator_type()` when converting the surviving
operator to `GATHER`, keeping the shared type index current. This consolidation
reduces the registered-runner call characterization from 137 to 135.

The conditional late generic-transpose and QKV-bridge pair reuses the QKV
helper in a bridge-only mode. Two prior invocations retain the default
prefix-plus-bridge path. The new invocation forwards the runtime layout flag,
disables the prefix, and always runs the bridge. Its scope stays between the
raw shape-extract and split/Conv/Concat rewrites, reducing the registered-
runner call characterization from 135 to 134.

The very-late Gather-axis, constant-fold/Cast, and flattened-normalization Pad
sequence shares one scope between the raw unbound-input transpose repair and
dynamic-Reshape resolution. The constant-fold/Cast helper accepts an optional
external scope. Both production callers now provide their wrapper-owned scope:
the earlier combined layout/Mean/SPP/Gather sequence and this later
Gather/normalization sequence. Seven diagnostic events in the latter reuse one
index without changing the registered-runner call characterization.

Four AST-identical layout-recovery prefixes now use one ordered orchestration
helper. The helper fixes the existing 19-call sequence from Q/DQ transpose
bridges through boundary BatchMatMul/unary cleanup, hard-activation and generic
SPP/NDHWC Concat recovery, and channel-shuffle/Gather cleanup. All four call
sites retain their original surrounding rewrites and execute the sequence once.
The hard-activation, SPP, and NDHWC runners intentionally remain unscoped
inside this helper because raw ModelIR mutators separate them. Consolidation
reduces the lowerer's registered-runner AST characterization from 134 to 125
without crossing an index-validity boundary or changing runtime pass count.

Two repeated QKV attention prefix/bridge pairs share one scope per occurrence.
The four prefix specs (Gather-layout hoist, Gather-to-Slice, Slice-to-Split,
and Split/Reshape collapse) retain their order before the two bridge specs
(shared pre-Transpose and weighted-sum bridge). The separate later bridge-only
invocation remains standalone. Both runners keep optional scope arguments, and
each production scope ends before the following legacy attention/layout
rewriter.

Two repeated duplicate-fan-out/quantized-PReLU pairs share one scope per
occurrence. The helper receives the existing QDQ-derived
`include_transpose` decision and forwards it unchanged to duplicate cleanup
before the four quantized-PReLU specs. The earlier duplicate-fan-out invocation
that is separated from PReLU by legacy QDQ rewrites remains standalone. Each
shared scope ends before dequantize/TransposeConv/quantize cleanup.

Two repeated constant-input-fold/redundant-Cast pairs share the enclosing
wrapper scope at each occurrence. The three constant Pad/Pool/Cast specs retain
their order before the widening-alias and narrowing-chain Cast specs. The first
scope also covers its preceding conditional layout/Mean/SPP/Gather sequence
and ends before the legacy ExpandDims/Squeeze replacement; the second starts at
Gather-axis and continues through normalization-Pad. Both runners remain
standalone-compatible through optional scope arguments.

The fallback and primary absolute-final SE-FC/Gather-channel-fan-out pairs use
a target-parameterized helper. The fallback invocation supplies
`fallback_ir` with no Session LayoutState; the primary invocation supplies the
main ModelIR and Session layout state. Both runners share one identity-bound
scope for their respective target, and each scope ends before static-shape
reconciliation.

The terminal dual-Mul/Concat, boundary-input adapter, Pad-layout, generic
transpose, and Gather-channel-fan-out sequence shares one scope. Its five
runners retain their exact order and seven-spec diagnostic grouping. The
scope begins after the final raw InstanceNorm residual/resize rewrite and ends
before the conditional terminal Mean/attention stage. Boundary-input cleanup
now accepts an optional scope while remaining standalone-compatible.

The late Dequantize/Concat/Quantize, unary-passthrough, and unary-fan-out
sequence also shares one scope. The scope begins after the raw Dequantize/
HardSigmoid/Quantize bridge rewrite and ends before the raw swish rewrite, so
no legacy ModelIR mutation can invalidate its index. The three runners retain
their exact order and diagnostics while constructing one graph index instead
of up to three.

`GraphIndex` and `ModelIRGraphIndex` provide differential mutation contracts.
ONNX rewriters notify node input/output updates and node registration/removal;
ModelIR rewriters can replace inputs/outputs or insert/remove operators while
producer, consumer, duplicate-producer, operator-position, and operator-type
indices remain consistent. The operator-type index returns graph-order
positions and is shifted together with all edge indices on insertion/removal,
so bounded passes can enumerate only relevant operator families instead of
rescanning the full ModelIR operator list. A full `refresh()` is retained only
for compatibility with external mutations that bypass these APIs.
`ConversionSession.tensor_consumer_count` is also the sole consumer-count
source passed into `LoweringContext`. This preserves the pre-session safety
contract used by inverse-transpose elision, including repeated uses of one
input, without rebuilding a second ONNX edge-count map.
Lowering-time logical and physical layout changes use
`LoweringContext.set_tensor_layout()`. The method updates the tensor metadata
and the Session-owned `LayoutState` together, so the first post-lowering pass
receives current layout evidence without relying on that pass to hide stale
state through a full resynchronization. Op-family builders must not assign
`TensorIR.logical_layout` or `TensorIR.physical_layout` directly.
`LoweringContext.add_operator()` and `remove_operator()` likewise own the
lowering-time operator list and maintain differential IR producer and consumer
maps. Inverse-Transpose generation uses the ONNX `GraphIndex` count for an ONNX
edge and the current IR count plus the pending use for a synthetic edge. It may
therefore remove an exclusive producer without rescanning the partial graph,
while retaining that producer when a previously emitted side branch still
consumes its synthetic output. Op-family builders must not mutate
`model_ir.operators` directly.
Lineage-aware graph mutation helpers accept an optional ModelIR index and
update it atomically.
`operator_indices_for_types()` returns a sorted, deduplicated union for
multi-type roots without scanning the operator list. Cast cleanup and
constant-input Cast/Pool/Pad/ScatterND/binary folding use these single- or
multi-type indices; each successful mutation restarts against the updated
index.

The final ModelIR validation pipeline accepts an optional current caller-owned
index. Float32 and float16 serialization preparation passes the same index used
and differentially maintained by terminal ScatterND/binary constant folding,
so invariant validation does not reconstruct identical producer/consumer and
duplicate-producer state immediately before writing. Callers without a current
index retain the original self-contained validation behavior.

Indexed root enumeration is used by the rank-four float and quantized Concat
families and by the bounded axis-3 constant-Concat, Add/Concat suffix, rank-five
NDHWC Concat, Concat/unary/Conv, SPP, and Dequantize/Concat/Quantize passes.
Each visits only graph-order `CONCATENATION` positions; its semantic and
boundary guards remain unchanged.
Quantized Reshape and the four quantized PReLU bridge/fusion matchers likewise
enumerate only indexed `DEQUANTIZE` or `TRANSPOSE` roots. Each successful
rewrite restarts its outer loop against the differentially updated index.

Operator additions keep capability validation and lowering in the same
op-family module. Do not add model-name checks. A model-specific failure should
be reduced to a semantic graph pattern with explicit guards and a focused ONNX
fixture generated with `onnx.helper`.

Coverage and tensor-correspondence report construction lives in
`tflite_builder/reporting.py`. `lower_from_onnx2tf.py` retains thin wrappers
with the legacy signatures so existing Python imports and callers remain
compatible. Schema policy classification, dispatch diagnostics, lineage
tracing, downstream correspondence inference, and JSON writers have one owner
in the reporting module. The module is part of the TensorFlow-free dependency
boundary.

Static rank-greater-than-five BatchMatMul compression lives in
`passes/high_rank_matmul.py`. `lower_from_onnx2tf.py` keeps the legacy helper
name as a delegating wrapper for compatibility. The pass uses the canonical
shape predicate and tensor-pruning utility from `core/model_ir_utils.py`.
Precision, constant-fold, layout, and high-rank passes share that pruning
implementation so lineage events and unused-tensor removal have one owner.
Dynamic broadcast shape-signature reconciliation is likewise canonical in
`core/model_ir_utils.py`; the lowerer imports and compatibility re-exports the
helper instead of owning a private implementation, allowing op-family passes
to use it without a circular dependency.

NCHW channel-shuffle canonicalization begins its own op-family ownership in
`passes/channel_shuffle.py`. The strict static
Reshape→Transpose(group/channel swap)→Reshape block becomes one
`GATHER(axis=1)` with deterministic shuffle indices, while exclusive
intermediate edges and rank/shape/group invariants are preserved. The full
146-line implementation moved with an identical AST; the lowerer exposes only
a thin compatibility wrapper. This module is the intended home for later NHWC
shuffle and stale Concat/Gather repairs, avoiding further growth in the central
lowerer.

All six production positions now call `run_nchw_channel_shuffle_cleanup`,
registered as `canonicalize.nchw_channel_shuffle_gather` in `CANONICALIZE`.
Model-only Reshape+Transpose preflight and an exact indexed guard enforce both
exclusive intermediate edges, swap permutation, static ranks/shapes, valid
group/channel factorization, and non-identity shuffle indices before
snapshotting. Gather mutation, deterministic index tensor creation, structural
removal, and pruning share one differential index and `LayoutState` under a
transactional invariant check. The module performs no whole-graph consumer-map
rebuild and no direct operator-list deletion.

The adjacent stale Concat/Gather repair is mechanically owned by the same
module. It restores NCHW Concat axis 1 only when the downstream
`GATHER(axis=1)` index count exactly equals the summed input channels and all
non-channel dimensions agree, then reconciles Concat/Gather metadata. Its
62-line implementation moved with an identical AST; the lowerer keeps a thin
compatibility wrapper.

The production position now calls `run_stale_nchw_channel_shuffle_repair`,
registered as `layout.repair_nchw_channel_shuffle_concat` in `LAYOUT_PLAN`.
Model-only Concat+Gather preflight and an exact indexed guard check producer
identity, axis/batch dimensions, four-dimensional compatible input shapes, and
exact index/channel cardinality before snapshotting. Axis and tensor-metadata
repair reuse one differential index and synchronize `LayoutState`
transactionally. These indexed channel-shuffle implementations perform no
whole-graph index rebuilds or direct operator-list mutation.

The strict ShuffleNet NHWC adapter form is mechanically owned by the same
family. It recognizes NHWC→NCHW Transpose, optional unary passthrough,
five-dimensional group/channel Reshape+swap, NCHW collapse, and final NHWC
Transpose, replacing the branch with `GATHER(axis=3)`. Shared leading
Transpose users are preserved, while intermediate fan-out and public
intermediates remain guarded. Its full 268-line implementation moved with an
identical AST before indexed migration; the lowerer keeps a compatibility
wrapper.

All five production positions now call `run_nhwc_channel_shuffle_cleanup`,
registered as `canonicalize.nhwc_channel_shuffle_gather` in `CANONICALIZE`.
Model-only Reshape+Transpose preflight and an exact indexed guard cover direct
and optional-unary prefixes, shared leading Transpose users, exclusive
intermediate edges, public intermediates, permutations, static ranks/shapes,
group/channel factorization, and non-identity indices before snapshotting.
Gather mutation, optional unary rewiring/metadata permutation, conditional
leading-Transpose retention, structural removal, and pruning share one
differential index and `LayoutState` transactionally. The entire extracted
indexed channel-shuffle subset has no whole-graph map builders or direct
operator-list deletion.

Generic multi-branch gate/Add-tree propagation, historically named for OSNet,
is owned by `passes/multi_branch_gate_layout.py`. The rule is
topology-driven: each rank-four branch has a layout adapter, Relu, keep-dims
Mean, Logistic gate adapter, and Mul leaf; two or more leaves feed an Add tree
and inverse output bridge. `run_multi_branch_gate_layout_cleanup` registers
stable `LAYOUT_PLAN` ID `layout.multi_branch_gate_add_tree_nhwc`. A model-only
required-op scan and indexed Add-tree/leaf guard reject incomplete and fan-out
graphs before snapshot creation; the complete matcher retains the deeper
shape, axes, exclusivity, and adapter checks. Input/output rewrites, independent
constant cloning, structural removals, pruning, and layout reconciliation use
one shared differential graph index and `LayoutState`. The lowerer keeps a
compatibility wrapper, and its single production position calls the runner
with session layout state and diagnostics.

The generic complementary-gate/two-output propagation rule is mechanically
owned by `passes/dual_postconv_gate_layout.py`. It recognizes three independent
NHWC-to-NCHW adapters, a Logistic/Sub complementary gate, two Mul/Add branches,
and inverse adapters feeding both downstream convolution branches. Compact
characterization fixes successful propagation plus gate fan-out, data-adapter
fan-out, and public-intermediate rejection.
`run_dual_postconv_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.dual_postconv_complementary_gate_nhwc`. Model-only required-op
preflight and an indexed complementary-gate/two-branch guard reject incomplete
or unsafe graphs before snapshotting. All input/output and alias rewrites,
structural removals, pruning, metadata, and layout reconciliation use one
shared differential graph index and `LayoutState`. The lowerer retains its
compatibility wrapper, and all five production positions supply session layout
state and diagnostics to the runner.

The rank-five LeakyRelu/Logistic/two-Add propagation rule is mechanically
owned by `passes/ndhwc_gate_layout.py`. It maps a four-dimensional base through
Reshape into NDHWC, removes independent skip/gate NCDHW adapters, and
canonicalizes both inverse output bridges. Its dedicated compact corpus
replaces a 177-line central inline fixture and fixes base fan-out, gate fan-out,
public-intermediate, permutation, and reshape-rank rejection.
`run_ndhwc_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.ndhwc_leaky_logistic_gate`. A model-only required-op scan and indexed
two-Add/base/skip/gate guard reject incomplete graphs before snapshotting.
Reshape, LeakyRelu, and Logistic input rewrites, both Add output rewrites,
constant-shape remapping, structural removals, pruning, metadata, and layout
reconciliation share one differential index and `LayoutState`. The lowerer
retains a compatibility wrapper, and all six production calls supply session
layout state and diagnostics to the runner.

The same family module owns the adjacent Conv3D/LeakyRelu/gate variant. It
removes a rank-five Conv adapter, accepts either a rank-four NHWC or rank-five
NDHWC semantic adapter before the gate Reshape, remaps the five-dimensional
shape, and canonicalizes the gated Mul output. Its dedicated corpus replaces a
176-line central inline fixture and fixes both accepted input ranks plus
Conv-adapter, LeakyRelu, and Reshape fan-out, public-intermediate, permutation,
and reshape-rank rejection. The family runner registers it as the second
ordered `LAYOUT_PLAN` spec with stable ID
`layout.ndhwc_conv3d_leaky_unsqueeze_gate`, after the rank-five gate spec.
Indexed producer/consumer traversal, input/output rewrites, structural
removals, pruning, metadata, and layout reconciliation reuse the same
differential graph index and `LayoutState`. Each of the six production groups
invokes the runner once, preserving the original rule order; the lowerer
compatibility wrapper remains available.

The adjacent cost-volume/ScatterND rule is mechanically owned by
`passes/cost_volume_scatter_layout.py`. Its dedicated compact corpus fixes the
complete two-input NHWC-to-NCHW descriptor path, Slice/Mean/Reshape and
ScatterND constant remapping, NDHWC output canonicalization, and downstream
Conv3D contract. Whole-ModelIR no-op checks cover leading-adapter fan-out, both
sides of the trailing adapter, a public intermediate, an invalid leading
permutation, a non-Conv3D downstream consumer, invalid ScatterND shape or
coordinate rank, and out-of-bounds coordinates. Pure indexed candidate
planning validates every constant before mutation, including conditions that
the legacy matcher checked only after earlier constants had changed.
`run_cost_volume_scatter_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.cost_volume_scatter_ndhwc`. All traversal, rewrites, structural
removals, pruning, metadata, and layout reconciliation use one differential
graph index and `LayoutState`. The six production positions now call the
transactional runner; the lowerer compatibility wrapper remains available.

The adjacent Add/Concat/constant-suffix rule is mechanically owned by
`passes/add_concat_suffix_layout.py`. Its dedicated compact corpus fixes two
branch adapters, a shared base adapter, two Add fan-ins, channel Concat,
rank-four MUL/ADD suffix constants, inverse output adaptation, and downstream
consumer aliasing. Whole-ModelIR no-op cases cover branch and intermediate
fan-out, public pre/post outputs, invalid leading permutation, invalid Concat
axis, and a missing suffix constant. Shared suffix constants use copy-on-write
so unrelated NCHW consumers remain unchanged. Pure indexed candidate planning,
all input/output rewrites, structural removal, pruning, metadata correction,
and layout synchronization share one graph index and `LayoutState`.
`run_add_concat_suffix_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.add_concat_const_suffix_nhwc`; all five production positions call the
transactional runner and the lowerer compatibility wrapper remains available.

The adjacent dual-Mul/Concat round-trip rule is mechanically owned by
`passes/dual_mul_concat_layout.py`. Its dedicated compact corpus fixes a
shared NHWC-to-NCHW data adapter, two Mul branches, channel Concat, inverse
output adapter, downstream consumer, and copy-on-write for an NCHW constant
shared outside the island. Whole-ModelIR no-op cases cover adapter/Mul/Concat
fan-out, public intermediate/post outputs, both invalid permutations, invalid
Concat axis, missing constant data, and branches that do not share the adapted
input. Pure indexed candidate planning validates topology and constant
broadcast feasibility before mutation. Copy-on-write, input/output rewrites,
structural removal, pruning, corrected output metadata, and layout
reconciliation share one graph index and `LayoutState`.
`run_dual_mul_concat_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.dual_mul_concat_nhwc`; all six production positions call the
transactional runner and the compatibility wrapper remains available.

The following axis-3 constant-Concat bridge rule is mechanically owned by
`passes/axis3_const_concat_layout.py` and has a dedicated compact corpus in
`tests/test_flatbuffer_direct_axis3_const_concat_layout.py`. It fixes the
exclusive input-adapter rewrite, conversion of NCHW constant inputs, axis-2
NHWC Concat, bypass of every inverse post adapter, retention of a shared input
adapter, and insertion of one NHWC-to-NCHW bridge for legacy consumers.
Whole-ModelIR no-op cases cover public Concat and post-adapter tensors, invalid
pre/post permutations, invalid Concat axis, invalid constant rank or
incompatible shape, missing constant data, and a constant shared outside the
Concat. Indexed candidate planning validates every conversion and optional
legacy bridge before mutation, including protection for a public adapter or
constant. Constant rewrites, Concat/post/legacy input rewrites, differential
operator removal/insertion, pruning, and layout reconciliation share one
`ModelIRGraphIndex` and `LayoutState`. The stable transactional runner ID is
`layout.axis3_const_concat_bridge_nhwc`; the single production position calls
the runner and the lowerer compatibility wrapper remains available.

The adjacent Dequantize/Concat/Quantize round-trip rule is mechanically owned
by `passes/dequant_concat_quantize_layout.py` and has a dedicated compact
corpus in
`tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py`. It fixes two
quantized NHWC input adapters, exclusive Dequantize branches, channel Concat,
Quantize, inverse post fan-out, canonical quantized-output aliasing, and
preservation of `QuantParamIR`. A shared leading adapter remains for unrelated
NCHW consumers. Whole-ModelIR no-op cases cover Dequantize, Concat, and
quantized-output fan-out; public pre/Dequantize/Concat/Quantize/post tensors;
invalid pre/post permutations; invalid Concat axis; and a non-Dequantize
branch. Pure indexed planning now validates every adapter, Dequantize branch,
rank-four Concat→Quantize edge, quantization record, post branch, and canonical
alias before mutation. Dequantize inputs, Concat axis, Quantize output, post
aliases, adapter/post removals, pruning, and metadata/layout reconciliation use
one `ModelIRGraphIndex` and `LayoutState`. The stable transactional runner ID is
`layout.dequant_concat_quantize_nhwc`; both production positions call it and
the lowerer compatibility wrapper remains available.

The adjacent Concat/optional-unary/post-adapter/Conv propagation rule is
mechanically owned by `passes/concat_unary_conv_layout.py` and has a dedicated
compact corpus in
`tests/test_flatbuffer_direct_concat_unary_conv_layout.py`. It fixes both the
unary-free path and a two-unary, two-post path ending in Conv2D and
DepthwiseConv2D consumers. Whole-ModelIR no-op cases cover leading-adapter,
Concat, and unary fan-out; public adapter/Concat/unary/post tensors; invalid
pre/post permutations and Concat axis; a non-Transpose input; an unsupported
unary; and a non-Conv post consumer. Pure indexed planning validates the entire
adapter/Concat/unary/post/Conv island and rank-four metadata before mutation.
Concat input/axis changes, metadata permutation, post aliasing, adapter
removal, pruning, and layout reconciliation use one `ModelIRGraphIndex` and
`LayoutState`. The stable transactional runner ID is
`layout.concat_unary_conv_nhwc`; both production positions call it and the
lowerer compatibility wrapper remains available.

The larger SPP-style Resize/Add/two-Concat/two-affine/two-Conv rule now has a
generic compact corpus in `tests/test_flatbuffer_direct_spp_layout.py`. The
success graph fixes four Resize branches sharing one base adapter, the first
channel Concat/Mul and NHWC affine/Conv island, the base/Conv second Concat/Mul,
and the final affine/Conv island. Whole-ModelIR no-op cases cover fan-out at
both branch and island boundaries, public base/first-Concat tensors, invalid
leading permutation or either Concat axis, a non-Resize branch producer, and
missing Mul constants. The complete matcher is mechanically owned by
`passes/spp_layout.py`. Pure indexed planning validates both islands and every
constant before mutation; shared constants use copy-on-write and per-axis
quantization dimensions move with their NCHW→NHWC transpose. Differential
mutation uses one `ModelIRGraphIndex` and `LayoutState`, and stable runner
`layout.generic_spp_nhwc` occupies all seven production positions while the
lowerer keeps its compatibility wrapper.

The adjacent rank-five NDHWC pre-Concat rule has a compact corpus in
`tests/test_flatbuffer_direct_ndhwc_concat_layout.py`. It fixes mixed direct
and unary inputs, two inverse post branches, axis-4 canonicalization, and
complete no-op behavior for fan-out, public, permutation, axis, unary, rank,
and spatial-shape boundaries. The implementation is owned by
`passes/ndhwc_concat_layout.py`; indexed candidate planning, differential
mutation, `LayoutState` reconciliation, and transactional runner
`layout.ndhwc_pre_concat` replace all five raw production calls. Per-axis
quantization dimensions move from NCDHW dimension 1 to NDHWC dimension 4.

The much larger rank-four generic NHWC pre-Concat matcher is being migrated by
semantic family rather than as one monolithic rule. Its strict float-path
direct-adapter, unary, Pad-plus-direct, Dequantize, PReLU, Softmax, and
expanded-Swish families, plus the bounded direct-source Slice and Split
families and the bounded recursive direct/supported-unary/expanded-Swish/
bounded-Split Add family, are
owned by `passes/nhwc_concat_layout.py`. The exact pseudo-LeakyRelu
decomposition is owned there as a bounded family as well.
Every Concat consumer must be an inverse
NCHW→NHWC Transpose. Direct inputs come from NHWC→NCHW Transpose; the unary
family permits one or more RELU, RELU6, LOGISTIC, TANH, or GELU operations
between exclusive leading adapters and Concat. The Pad family accepts one or
more strict Pad inputs together with at least one direct input. Candidate
planning uses one shared `ModelIRGraphIndex`, retains shared or public direct
adapters, rejects unsafe unary/Pad fan-out and public boundaries, validates
compatible spatial shapes, and applies the rewrites transactionally under
stable IDs `layout.nhwc_pre_concat_direct`,
`layout.nhwc_pre_concat_unary`, `layout.nhwc_pre_concat_pad`, and
`layout.nhwc_pre_concat_dequantize`, plus
`layout.nhwc_pre_concat_prelu`, `layout.nhwc_pre_concat_softmax`, and
`layout.nhwc_pre_concat_swish`, `layout.nhwc_pre_concat_slice`, and
`layout.nhwc_pre_concat_split`, plus `layout.nhwc_pre_concat_add`.
One ordered table owns all eleven family names, statistics keys, and priorities;
frozen callbacks, preconditions, `PassSpec` objects, defaults, and preflight are
derived once at module import. Candidate enumeration uses the shared
operator-type index and visits only `CONCATENATION` positions. A successful
precondition passes its validated candidate through session-local prepared
data, so the callback does not immediately repeat the same search. The legacy
family-specific optimizer wrappers remain signature-compatible while the
runner itself uses the common family optimizer. This declarative conversion
keeps the module below its original 3,120-line state without changing pass
order or fallback ownership.
Input-plan dispatch is also declarative for non-Add families: one map owns the
common resolver signature, one owns resolvers requiring public tensor names,
and one owns Swish/Leaky resolvers that permit unary companions. An import-time
invariant requires these maps to cover every declared family except direct and
recursive Add. Add remains explicit because it carries the candidate-wide
consumer set through recursive planning.
Within the Add root-companion path, one ordered resolver tuple preserves the
existing Dequantize→PReLU→Softmax→Leaky→Pad→Slice precedence. Split remains the
final explicit resolver because it alone receives the candidate-wide allowed
consumer set.
Candidate combination acceptance is a separate immutable contract table. Each
family declares allowed and required input kinds, exact-count constraints, and
whether spatial shapes must be reconciled. One validator replaces the repeated
per-family count branches; an import-time invariant requires the contract and
pass-family sets to match. Slice retains only its plan-aware unique-operator
guard, while recursive Add retains its consumer-ownership planning.
Unary, Swish, Softmax, and Dequantize application share one common-signature
applier map. Its keys plus the contextual direct/Pad/PReLU/Slice/Split/Add/
Leaky paths must equal the union of contract kinds at import time. Split, Add,
and Leaky retain their applied-operator sets so shared plans are mutated once.
Adapter liveness and recursive plan walking are top-level helpers rather than
closures rebuilt for every candidate.
Exclusive pads constants are remapped in place; shared or public pads
constants use copy-on-write so unrelated Pad consumers preserve NCHW
semantics. Dequantize inputs retain source scale and zero-point provenance
while their output metadata moves to NHWC. PReLU selects the first
broadcastable rank-4, unchanged, or rank-3 alpha representation in legacy
order; shared/public transformed alpha uses copy-on-write and its per-axis
dimension follows the applied permutation. Exactly one Softmax plus at least
one direct input is lifted through local NHWC→NHCW→Softmax→NHWC adapters,
preserving the original NCHW last-axis semantics while reducing total
Transpose count. Canonical Concat and branch/intermediate per-axis
quantization metadata follow their actual permutations. Expanded Swish accepts
one or more exact `Logistic(x) * x` diamonds in either Mul input order, with
only direct or supported unary companion inputs. It rewires both diamond edges
to the NHWC source and remaps Logistic and Mul output metadata. Adapter,
Logistic, or Mul fan-out and public boundaries are rejected before mutation;
public Logistic output rejection deliberately closes an unsafe legacy case.
The bounded Slice family accepts one or more rank-four Slice inputs whose
output consumers are limited to the selected Concat and exact inverse
adapters, optionally together with direct inputs. Begin and size vectors are
remapped from NCHW to NHWC;
shared/public parameter tensors use copy-on-write with tensor provenance
retained, shared/public source adapters remain for their external consumers,
exact inverse output adapters are bypassed, and Slice-output per-axis
quantization follows the layout change.
The bounded Split family accepts one or more outputs from a rank-four Split
whose outputs are unused or consumed only by the selected Concat. It remaps
the Split axis to 3 once per operator, uses copy-on-write for a shared/public
axis tensor, retains shared/public source adapters for external consumers, and
moves every Split output's shape and per-axis quantization into NHWC while
bypassing exact inverse output adapters. Swish-source Slice and
broader Split/Add interactions remain in the legacy matcher. The bounded Add
family accepts a bounded acyclic two-input Add graph whose leaves come from
rank-four
NHWC→NCHW adapters, optionally through a supported unary operation, exact
expanded-Swish diamond, Dequantize, PReLU, semantics-preserving Softmax,
exact pseudo-LeakyRelu diamond, exact Pad, bounded direct-source Slice, or
bounded Split. Dequantize, PReLU, Softmax, pseudo-LeakyRelu, Pad, and Slice
plans may also be companion inputs of the same root Concat. Add inputs and
bounded operand
branches are rewired together, exclusive adapters are removed, shared/public
adapters remain for external consumers, exact inverse output adapters are
bypassed, and every Add output shape and per-axis quantization moves into
NHWC. Candidate-wide planning collects every selected Add plus the root
Concat, so an Add output may feed multiple selected Add branches or both a
parent Add and the root Concat. Consumers outside that set must be exact
inverse adapters or the candidate rejects. Recursive planning tracks visited
Add outputs, rejects cycles, and
stops at a maximum depth of 64. Shared application state ensures that nested
Add and Split operators and cloned integer parameters are materialized only
once; the same candidate-wide Pad materialization map is reused by nested and
root-companion Pad plans. Slice begin/size tensors use the same shared integer
materialization map as Split axes, retaining copy-on-write behavior. One Split
may feed different Add nodes and a
separate input of the same root Concat. Candidate planning first collects the
bounded Add operator set,
adds the root Concat, and permits Split-output consumers only inside that set
or through exact inverse adapters. This candidate-wide set makes matching
independent of Concat input order; any external consumer rejects the whole
candidate. Each nested output post-adapter remains associated with its own
Add; cleanup walks the complete input-plan tree only after the rewrites
succeed.
Source-adapter removal is decided from the post-rewrite GraphIndex, allowing an
adapter shared with the root Concat to be removed only after every selected
consumer is rewired.
When selected PReLU plans share one transformed alpha, a candidate-wide key of
source, permutation, and shape reuses one provenance-preserving clone.
Softmax operands reuse the existing local NHWC↔NHCW adapters, retaining the
original NCHW last-axis meaning. The exact pseudo-LeakyRelu diamond reuses its
existing internal-edge and singleton-alpha guards. Other uncharacterized Add
operands and broader mixed-input quantized-post families remain in legacy
until independently characterized.
The indexed
pseudo-LeakyRelu family recognizes the complete
`ReLU(x) - alpha * ReLU(-x)` diamond with either Mul operand order and direct
or supported unary Concat companions. It preserves the non-commutative Sub
order, rewires Neg and positive Relu to the NHWC source, and remaps all five
internal/output tensor shapes and per-axis quantization metadata. Public or
fan-out internal edges, invalid ranks, non-singleton alpha, and incomplete
diamonds reject before mutation under stable ID
`layout.nhwc_pre_concat_leaky`. Pad/Add/Split mixed companions and broader
mixed-input quantized-post families remain in legacy.

The strict direct, unary, Pad-plus-direct, mixed unary-plus-Pad, all-Pad,
expanded-Swish, Dequantize, PReLU, Softmax, and exact pseudo-LeakyRelu
plus bounded Slice/Split/Add quantized-post paths are independently owned by
`passes/nhwc_concat_quantized_layout.py`. They recognize rank-four direct
NHWC→NCHW inputs, optionally followed by RELU, RELU6, LOGISTIC, TANH, GELU, or
an exact constant PAD, the exact `Logistic(x) * x` expanded-Swish diamond,
Dequantize, PReLU, Softmax, or the exact pseudo-LeakyRelu diamond, then channel
Concat, one Quantize, and one or more inverse
post Transposes. The
mixed pass requires at least one unary and one Pad branch; additional direct
branches are allowed. The all-Pad pass requires at least two Pad branches. The
Swish pass accepts either Mul operand order and reuses the float path's strict
resolver and apply operation, including its public/fan-out boundary checks.
The Dequantize pass likewise shares the float resolver/apply pair and retains a
leading adapter only when its quantized NCHW output has another consumer.
PReLU also shares the float resolver/apply pair, including broadcast-safe alpha
selection, per-axis metadata remapping, and provenance-preserving copy-on-write.
Softmax reuses the axis-preserving float plan: local NHWC→NHCW→Softmax→NHWC
adapters retain the original NCHW last-axis meaning while the outer Concat and
Quantize path remains NHWC.
The pseudo-LeakyRelu pass shares the exact float matcher/apply pair, preserving
the non-commutative Sub order, scalar alpha guard, and all internal fan-out and
public-boundary invariants.
Expanded-Swish, pseudo-LeakyRelu, and bounded Add may also combine with
supported unary root-Concat companions through the same unary resolver/apply.
The bounded Slice pass accepts a rank-four channel Slice with constant
begin/size and no secondary output adapter, reorders the parameters to NHWC,
and uses the float path's provenance-preserving copy-on-write materializer.
The bounded Split pass accepts a constant channel axis and no secondary output
adapter. Multiple outputs from one Split may feed the root Concat; shared
application state rewrites the axis and all output metadata exactly once.
The bounded Add pass accepts a depth-guarded Add tree whose leaves are direct,
unary, expanded-Swish, Dequantize, PReLU, Softmax, exact pseudo-LeakyRelu,
Pad, Slice, or Split plans without secondary output adapters, plus direct
Concat companions. Shared application state
rewrites each Add once. Shared-plan cleanup walks nested operand plans and
removes every now-dead adapter while retaining public or still-used boundaries.
The transactional passes
rewire Concat and bounded branches to NHWC, retain shared/public input
adapters, redirect Quantize to one canonical post output, and coalesce
additional post aliases. Pad constants are reordered from NCHW to NHWC and use
provenance-preserving copy-on-write when shared or public. Match, constant
materialization, and Pad metadata updates are shared with the float path
through `passes/nhwc_concat_pad.py`, so the two pipelines cannot drift
independently. The float Concat, unary, and Pad output metadata, along with
quantized output per-axis dimensions, follow NCHW→NHWC.
Public Concat/Quantize/post or branch boundaries, non-Transpose fan-out,
invalid ranks/spatial metadata, and partial chains reject before mutation
under stable IDs
`layout.nhwc_pre_concat_quantized_direct` and
`layout.nhwc_pre_concat_quantized_unary`, and
`layout.nhwc_pre_concat_quantized_pad`, followed by
`layout.nhwc_pre_concat_quantized_unary_pad`, and
`layout.nhwc_pre_concat_quantized_all_pad`, followed by
`layout.nhwc_pre_concat_quantized_swish` and
`layout.nhwc_pre_concat_quantized_dequantize`, followed by
`layout.nhwc_pre_concat_quantized_prelu` and
`layout.nhwc_pre_concat_quantized_softmax`, followed by
`layout.nhwc_pre_concat_quantized_leaky` and
`layout.nhwc_pre_concat_quantized_slice`, followed by
`layout.nhwc_pre_concat_quantized_split` and
`layout.nhwc_pre_concat_quantized_add`. Shared pads constants are
materialized once and reused by every selected Pad. Other broader mixed
quantized inputs remain in legacy.

The quantized-post runner is declarative: one ordered table owns each family
name, stable statistics key, and priority. Pass IDs, callbacks, preconditions,
and default diagnostics are generated from that table. Shared float-plan
resolution is likewise declarative: one resolver map covers the common
signature, a second covers resolvers that require public tensor names, and one
adapter maps every selected plan into `_QuantizedInputPlan`. The additional
rank-four source guard for Dequantize and the no-secondary-post-adapter guard
for Slice/Split are explicit family sets. Add remains separate because it
recursively validates bounded leaf plans. These staged declarative changes
keep `nhwc_concat_quantized_layout.py` materially below its original
1,549 lines without changing the ordered pass contract or the legacy fallback
boundary.

Candidate acceptance is also declarative. Each family now records allowed and
required input kinds, minimum arity, exact-count constraints, and whether
spatial shapes must be reconciled. One validator replaces thirteen count
branches, and an import-time invariant requires the input-contract family set
to match the pass table. The resolver maps and input contracts remain separate:
the former select safe graph plans, while the latter validate the complete
Concat input combination. Unary and Pad companion resolution is enabled
directly from each contract's allowed kinds instead of maintaining duplicate
family lists. The four shared apply operations with a common signature use a
matching applier map; an import-time invariant prevents resolver/apply coverage
from drifting while contextual PReLU/Slice/Split application remains explicit.
The thirteen frozen `PassSpec` objects, callbacks, preconditions, default
statistics, and model-only preflight are constructed once at module import
rather than on every runner call. Adapter liveness and recursive shared-plan
walking are top-level functions rather than closures recreated for every
candidate. Each invocation therefore allocates only its pass manager/session
state while retaining the same transactional snapshots and diagnostics.
Quantized-post candidate planning enumerates only indexed `CONCATENATION`
positions; the thirteen family preconditions no longer scan unrelated
operators in large graphs. A successful precondition stores its fully validated
candidate in the session-local prepared data and the callback consumes it,
avoiding immediate duplicate resolution. Only the post-rewrite search for an
additional candidate runs again.

The legacy ownership boundary uses the same compact style. Simple quantized
families are described by allowed and required action-kind sets after one
action multiset is built. Only Slice, Split, and Add retain plan-aware
predicates. This replaces the repeated per-family count/subset blocks while
preserving the exact fallback boundary for broader legacy combinations.
The seven analogous float families now use a second allowed/required-kind
contract table over the same action multiset. Softmax's single-branch benefit
gate and Pad's required direct companion remain explicit invariants;
Slice/Split/Add remain plan-aware. This removes another net 74 lines from the
central lowerer.
The public compatibility wrapper also aggregates its eleven float and thirteen
quantized family counters through explicit key tuples and two sums, then adds
the legacy counter. This preserves the single historical return key while
removing another net 110 repeated lowerer lines.

The same family module mechanically owns the adjacent post-Add variant, where
the two Mul outputs cross inverse adapters before their downstream NHWC Add and
Conv. Compact characterization fixes successful two-output canonicalization
and gate fan-out, data-adapter fan-out, and public-intermediate rejection. The
family runner now owns a second ordered `LAYOUT_PLAN` spec with stable ID
`layout.postadd_complementary_gate_nhwc`, after the dual-postconv spec. Both
guards share one indexed resolver for the three-adapter Logistic/Sub/two-Mul
prefix, then validate only their distinct output topology. Postadd input/output
and alias rewrites, structural removals, pruning, metadata, and layout
reconciliation use the same differential state. Each of the five production
groups invokes the runner once, preserving dual-postconv-before-postadd order;
the lowerer compatibility wrapper remains available.

The larger generic two-way shuffle/branch/Concat propagation rule is now
mechanically owned by the same family. It accepts rank-five Gather selectors or
rank-four channel Slice selectors, traces one split through an NHWC
Conv/elementwise branch and the other as a skip, converts both selectors to
axis-3 Gathers, moves the final Concat to NHWC, and restores the downstream
NCHW boundary. The complete 609-line implementation moved with an identical
AST; its two existing positive tests cover both selector representations. The
lowerer retains a compatibility wrapper.

All five production positions now call `run_two_way_channel_shuffle_cleanup`,
registered as `canonicalize.two_way_channel_shuffle_branch` in
`CANONICALIZE`. Model-only Concat/Reshape/Transpose/selector preflight avoids
state construction for irrelevant graphs. An indexed prefix guard proves the
exclusive Concat→layout-Transpose→Reshape→shuffle-Transpose→Reshape→two-selector
shape before snapshotting; the existing deep matcher then validates selector
constants, branch/skip topology, op allowlists, layouts, shapes, and public
boundaries. Every consumer/producer lookup, input/output rewrite, Gather and
boundary insertion, id-based structural removal replacement, and pruning now
uses one differential index and `LayoutState` transactionally. The family has
no whole-graph map builders, full operator searches for removal, or direct
operator-list insert/delete operations.

The adjacent Mean layout family has begun its staged migration in
`passes/mean_layout.py`. The generic Transpose→Mean→inverse-Transpose
passthrough rewrite and the Transpose→Mean→Mul(const)→Reshape→Add(const)→Conv
NHWC propagation rewrite are mechanically owned there. Their implementations
are AST-equivalent to checkpoint `c99418a` after excluding docstrings; rank,
constant-axis, fan-out, public-boundary, metadata, and conditional leading
Transpose behavior are unchanged. The lowerer retains compatibility wrappers.

Both rewrites now reuse a differential `ModelIRGraphIndex` and active
`LayoutState`; consumer reads, input/output rewrites, alias replacement,
operator removal, tensor pruning, and layout synchronization update shared
state. `run_transpose_mean_passthrough_cleanup` registers
`layout.transpose_mean_prepost`, while
`run_mean_mul_add_conv_layout_cleanup` registers
`layout.mean_mul_add_conv_nhwc`; both execute in `LAYOUT_PLAN`. Cheap model-only
capability scans avoid state construction on irrelevant graphs, and indexed
guards prove the full relevant fan-out, axes, constant, shape, and Conv prefix
before transaction creation. All six and seven respective production
positions use the ordered runners and session diagnostics. The module performs
no whole-graph map construction or direct operator-list insertion/deletion.

The adjacent LayerNorm-statistics layout pair is mechanically owned by
`passes/layernorm_layout.py`. One rewrite propagates NHWC through two keep-dims
Mean statistics branches behind an NHWC→NCHW adapter, conditionally removing
that adapter; the other reuses an existing NCHW→NHWC projection. Both preserve
the strict centered-tensor fan-out guard, constant-axis remapping, rank-four
contract, public boundaries, and tensor metadata. Their complete ASTs match
checkpoint `d7866d2`; the lowerer keeps compatibility wrappers.

`run_layernorm_statistics_layout_cleanup` registers
`layout.layernorm_statistics_from_pre_transpose` and
`layout.layernorm_statistics_from_existing_post` in `LAYOUT_PLAN`. Because the
legacy rules were adjacent at both production positions, the two specs share
one pass state, one differential graph index, one layout state, and one
model-only capability scan per invocation. Indexed guards prove each complete
statistics topology before snapshotting. All input rewrites, conditional
Transpose removal, pruning, and metadata/layout reconciliation update shared
state; duplicate consumer entries from a self-Mul are normalized by operator
identity in the guard. The module has no whole-graph map construction or
direct operator-list mutation.

Terminal unary/Mean layout propagation is mechanically owned by
`passes/terminal_mean_layout.py`. It recognizes a canonical NHWC→NCHW adapter,
a linear layout-agnostic unary chain, a rank-four constant-axis Mean, and a
constant-shape terminal Reshape. It preserves shared leading adapters, rejects
unary fan-out, accepts shape-proven rank preservation when keepDims metadata is
missing, and defers inverse-Transpose tails to the dedicated paired-adapter
rewrite. The complete 259-line matcher is AST-identical to checkpoint
`8bce913`; the lowerer retains a compatibility wrapper.

All six production positions now call `run_terminal_mean_layout_cleanup`,
registered as `layout.terminal_unary_mean_reshape` in `LAYOUT_PLAN`. A
model-only Transpose/Mean/Reshape capability scan avoids pass-state creation on
irrelevant graphs. Its indexed guard proves the complete permutation, rank,
linear-unary, axes, keep-dims/shape fallback, exclusive Mean consumer, and
constant Reshape suffix before snapshotting. Consumer reads, unary/Mean input
rewrites, conditional adapter removal, pruning, metadata permutation, and
layout synchronization reuse one differential index and active layout state.
The module has no whole-graph map construction or direct operator-list
mutation.

EfficientNet-style squeeze/excitation propagation is mechanically owned by
`passes/se_layout.py`. The SE-Conv matcher covers Logistic and affine gates,
explicit Transpose and Squeeze/Reshape adapters, Mean axis remapping, shared
leading adapters, and terminal fan-out bridges. The broader SE-FC matcher
covers pooled or Mean statistics, stacked FC/1x1-Conv heads, optional unary
gates/output bridges, and its existing nested specialized variants. Their
complete 482-line and 977-line implementations are AST-identical to checkpoint
`5f5de07`; the lowerer retains compatibility wrappers.

All six SE-Conv positions now call `run_se_conv_layout_cleanup`, registered as
`layout.se_conv_gate_nhwc` in `LAYOUT_PLAN`. Model-only capability preflight
avoids state construction on irrelevant graphs. Its indexed common-region
guard proves the leading Swish-style gate, Mean branch/adapters, second gate
exclusivity, and complete inverse-Transpose output fan-out before snapshotting;
the existing deep matcher retains validation of Logistic, affine, and
Squeeze/Reshape gate variants. All graph mutations and removals use one
differential index and active layout state. The SE-Conv implementation has no
whole-graph map construction or direct operator-list deletion.

SE-FC is now also indexed through `run_se_fc_layout_cleanup`, registered as
`layout.se_fc_gate_nhwc` in `LAYOUT_PLAN`. Eight main-model positions share the
active session layout state; the fallback-IR position intentionally builds a
separate state because it owns a different ModelIR. Model-only capability
preflight and an indexed guard cover the normal gate Reshape/output bridge and
the common alternate affine/Conv prefix before the existing deep matcher.
Constants cloned for Mean-axis isolation, every input/output rewrite, alias
replacement, structural removal, pruning, and layout synchronization update
the differential index/state. The complete SE module now has no whole-graph
map builders or direct operator-list deletion.

The adjacent rank-four elementwise gate family is mechanically owned by
`passes/elementwise_gate_layout.py`. It contains SUM/Logistic/Sub/Mul/Add,
weighted-Add Swish, nested weighted-Add Swish, and Logistic/Mul/Add propagation
rules. Their complete 342-, 321-, 298-, and 346-line implementations are
AST-identical to checkpoint `1623762`; all five production positions per rule
are now replaced by five calls to `run_elementwise_gate_layout_cleanup`.
Scalar-like tensor recognition is a shared `core/model_ir_utils.py` utility,
with the lowerer preserving its compatibility import.

The group registers four stable `LAYOUT_PLAN` IDs for SUM/Logistic/MulAdd,
weighted Swish, nested weighted Swish, and Logistic/MulAdd, preserving their
legacy priority while sharing one graph index and layout state per production
position. Model-only capability preflight and indexed topology guards avoid
snapshots for irrelevant or protected graphs. All input/output rewrites, alias
replacement, structural removals, pruning, and layout synchronization use
differential state. The module contains no whole-graph map builder or direct
operator-list deletion.

Synthetic input-boundary transpose elision lives in
`passes/boundary_input_layout.py`. It only removes the adapter when public and
internal tensor metadata agree and no axis-sensitive gather/slice consumer
requires the boundary. Consumer indexing, transpose-permutation reads, and
lineage-aware input replacement are canonical utilities in
`core/model_ir_utils.py`; reporting, precision, and layout passes do not keep
private copies.

The lowerer invokes this rewrite through `run_boundary_input_layout_cleanup`,
registered as `layout.boundary_input_adapter` in `LAYOUT_PLAN`. It shares one
ModelIRPassState, uses indexed consumers and structural removal, synchronizes
pruned adapter tensors with the session LayoutState, and validates the public
input contract transactionally. A topology precondition avoids snapshots on
graphs without synthetic boundary adapters.

Channel-slice layout propagation and boundary StridedSlice/QDQ/Concat cleanup
live together in `passes/channel_slice_layout.py`. The family owns the guarded
boundary channel-slice rewrite, internal NHWC propagation, Mul/Add bridge
rewrites, strict dual-Add bridges, and the StridedSlice/QDQ/Concat round-trip
rewrite. The strict slice-MulAdd/Conv/merge-Add and
slice-MulAdd/merge-Add/post-Transpose implementations are also owned by this
module; their legacy lowerer symbols are thin compatibility wrappers. This
places the three adjacent late channel-slice rewrites under one ownership
boundary. They execute through `run_channel_slice_merge_layout_cleanup` at
their three unchanged recovery positions. The runner registers stable
`LAYOUT_PLAN` IDs `layout.channel_slice_dual_add_strict`,
`layout.slice_muladd_conv_mergeadd_strict`, and
`layout.slice_muladd_mergeadd_posttranspose_strict` with priorities 10/20/30.
All producer/consumer reads, input/output rewrites, structural removal and
insertion, constant cloning, and pruning share one differential
`ModelIRGraphIndex` and `LayoutState`. Model-only capability scanning avoids
state construction on irrelevant graphs; indexed prefix and downstream-chain
guards reject incomplete two-Slice regions before snapshotting. Include flags
allow isolated compatibility tests without changing the production order.
Constant-vector reads/writes, operator input/output mutation,
broadcast checks, metadata permutation, and lineage recording remain shared
core utilities. Legacy function names delegate to the family module.

The earlier synthetic-boundary channel-Slice rewrite also accepts one shared
`ModelIRGraphIndex` and the session `LayoutState`. It discovers only indexed
Transpose roots, obtains consumers from the index, and restricts iterative
NHWC propagation to the supported operator-type union. Slice rewiring, local
bridge rewiring, bridge insertion, and removal of the shared boundary adapter
all update the index differentially. The original graph-order insertion rules,
channel-axis constant conversion, and bridge-localization semantics are
unchanged. Pruning and final layout synchronization cover every newly created
adapter tensor without rebuilding producer/consumer maps.

The corresponding internal Transpose/channel-Slice propagation rewrite uses
the same optional state contract. Indexed Transpose roots and consumers replace
its repeated map construction, while the propagation loop visits only its
declared unary, binary, Concat, Slice, Conv, and Pool operator families. A lazy
per-constant consumer snapshot preserves the original copy-on-write decision
when multiple converted binary operators share one NCHW constant. Input
rewrites, cloned-constant rewires, NCHW bridge insertion, and removal of the
internal stem are differential index operations; final pruning synchronizes
the active `LayoutState`.

The adjacent channel-Slice/Mul/post-Transpose bridge family now shares that
indexed contract too. Root discovery is restricted to Transpose indices, all
Slice/Mul/post aliases are rewired through the index-aware canonical helpers,
and obsolete pre/post adapters are removed differentially. A lightweight
consumer-index snapshot is taken from the existing index at each retry because
the legacy matcher intentionally evaluates every branch against the same
pre-rewrite consumer view. This preserves branch selection and shared-constant
copy-on-write behavior without reconstructing producer/consumer maps or
scanning operators for roots. Successful cleanup prunes and synchronizes the
session `LayoutState`.

The boundary StridedSlice/QDQ/Concat round-trip rewrite completes the indexed
migration of this module. It validates the complete multi-branch topology from
one live index before mutation, then applies StridedSlice inputs, quantized
Concat output aliasing, and all post-alias rewires through index-aware helpers.
The shared boundary and trailing Transposes are removed differentially, with
layout-aware pruning after the retry loop. Consequently
`passes/channel_slice_layout.py` no longer builds whole-graph consumer maps or
inserts/deletes operators directly; its legacy function entry points remain
signature compatible through optional keyword state.

Boundary input normalization chains live in
`passes/boundary_input_chains.py`. The module owns the guarded
Transpose/Mul/Sum/Reshape NHWC rewrite and the exclusive
Transpose/BatchMatMul boundary rewrite. Both passes retain their fan-out,
model-output, permutation, constant-shape, axis, and metadata guards. Their
legacy lowerer names are delegating wrappers, while graph mutation and
constant-vector access use the canonical `core/model_ir_utils.py` helpers.

The boundary Mul/Sum/Reshape normalization rewrite executes through
`run_boundary_input_normalization_cleanup` at its two unchanged terminal
positions, registered as `layout.boundary_input_mul_sum_reshape` in
`LAYOUT_PLAN`. Its model-only preflight requires the common boundary
Transpose/Mul/Sum/Reshape capability. The indexed guard then proves the
synthetic input adapter, scalar or channelwise constant, single-consumer
normalization chain, kept reduction dimensions, channel axis, and terminal
Reshape before snapshotting. Rewiring and structural removal update one
`ModelIRGraphIndex`, and pruning synchronizes the session `LayoutState`.

The BatchMatMul rewrite executes through
`run_boundary_input_batchmatmul_cleanup`, registered as
`layout.boundary_input_batchmatmul` in `LAYOUT_PLAN`. Its shared candidate
matcher proves the boundary permutation, exclusive model-input adapter, and
BatchMatMul-only fan-out before a transaction is opened. Input rewiring and
Transpose removal update one `ModelIRGraphIndex` differentially, while unused
adapter tensors are removed from the shared `LayoutState`. The four production
retry positions remain separate and in their original order so intervening
layout rewrites can expose a later candidate.

Generic input-boundary passthrough folding lives in
`passes/input_passthrough_layout.py`. It moves strictly linear chains of
layout-agnostic unary and constant-side binary operations across a synthetic
input transpose, then removes the matching output transpose. Fan-out,
nonconstant side input, per-axis quantization, noninverse permutation, and
model-output guards remain part of the pass contract. Permutation inversion is
a canonical graph utility in `core/model_ir_utils.py`; the legacy lowerer pass
name remains a delegating wrapper.

The same module owns the guarded ASIN/ACOS decomposition passthrough rewrite:
`Mul(x,x) → Sub → Sqrt → Atan2`. It requires the original two-branch topology,
a singleton subtraction constant, strict inner consumers, and either a model
output or an inverse terminal transpose. Singleton constant detection is a
canonical `core/model_ir_utils.py` predicate shared by the remaining legacy
rewrite families.

Standalone HardSigmoid decomposition passthrough also lives in
`passes/input_passthrough_layout.py`. It accepts only the strict scalar
`Mul → Add → Relu0To1` form or scalar `Mul → Add → Maximum → Minimum` form and
requires an inverse terminal transpose. Intermediate metadata and output
lineage are updated through the shared graph mutation utilities.

The strict ERF polynomial decomposition passthrough is colocated in
`passes/input_passthrough_layout.py`. Its two ABS/SIGN branches, reciprocal
prelude, exponential branch, four-stage Horner chain, final sign merge, scalar
constants, exact consumer counts, and inverse terminal permutation are all
validated before any mutation. This remains a semantic graph pattern and does
not depend on a model name.

At the four production locations where leading-input, Asin/Acos, and Erf
passthrough were consecutive, `run_input_unary_passthrough_cleanup` registers
them as one `LAYOUT_PLAN` group with priorities 10/20/30 and stable IDs
`layout.leading_input_passthrough`, `layout.asin_passthrough`, and
`layout.erf_passthrough`. All three rewrites use one ModelIRGraphIndex and
LayoutState for input/output mutation, boundary-transpose removal, and pruning.
A model-only relevant-op scan avoids state construction on unrelated graphs;
indexed topology guards distinguish the single-consumer leading chain from the
two-branch Asin and Erf decompositions before taking snapshots. HardSwish and
HardSigmoid remain outside this group because intervening production rewrites
separate the two sequences.

Pseudo-expanded HardSwish passthrough is another member of
`passes/input_passthrough_layout.py`. It accepts only the residual
`Add → optional Relu6 → Div-or-Mul → Mul(original, branch)` topology with
singleton side constants, linear branch consumers, and an inverse terminal
transpose. Both residual inputs are rewired through lineage-aware helpers.

The HardSigmoid-plus-residual-Mul passthrough family is also owned by
`passes/input_passthrough_layout.py`. Its stricter multi-user analysis handles
the expanded HardSigmoid clamp, residual multiplication, optional Mean branch,
axis remapping, and cases where a legacy NCHW adapter must be retained for
other consumers. Constant-vector and graph mutations use only shared core
helpers.

Hard activation passthrough executes through
`run_hard_activation_passthrough_cleanup`. Four unchanged production sites
register HardSwish, standalone HardSigmoid, then HardSigmoid-residual-Mul as
`layout.hardswish_passthrough`, `layout.hardsigmoid_passthrough`, and
`layout.hardsigmoid_mul_passthrough`. The late recovery site enables only the
last two specs and reverses their priorities to preserve its original
HardSigmoid-Mul→HardSigmoid order. All variants share one ModelIRGraphIndex and
LayoutState. Indexed guards trace singleton constants, Add/clamp topology,
residual multiplication, optional Mean fan-out, and inverse terminal
transposes before snapshotting; operator rewires and boundary-transpose removal
update the differential index directly.

The terminal hard-activation recovery and its immediately following optional
generic Transpose cleanup share one lazy `ModelIRPassStateScope`. The hard
runner keeps its exact late configuration: HardSwish is disabled, both
HardSigmoid variants are enabled, and their order is reversed. When generic
layout optimization is disabled, the hard-activation runner still executes by
itself as before; when enabled, the Transpose runner reuses the already-built
graph index and layout state. The scope begins after the raw
HardSwish/SE/HardSigmoid rewrite and ends before the raw pre-Concat rewrite, so
no unindexed legacy mutation can invalidate the shared state.

Pad layout ownership is centralized in `passes/pad_layout.py`. In addition to
repairing a proven channel-last input/channel-first Pad mismatch, the module
owns direct inverse-transpose Pad folding and the guarded unary-to-Pad tail
rewrite that can retain one local NCHW adapter for legacy consumers. Padding
axis rotation, dynamic metadata, quantization, fan-out slots, and output names
are preserved; lowerer symbols remain compatibility wrappers.

The final channel-last-input/channel-first-Pad repair also uses this indexed
contract. It enumerates only `PAD`, `PADV2`, and `MIRROR_PAD` roots, retains the
existing full static input/output/padding shape proof, rewires the selected Pad
input through `ModelIRGraphIndex`, and inserts the required NHWC-to-NCHW
Transpose differentially. Each successful insertion restarts from the updated
index, preserving graph order without rebuilding producer/consumer maps or
mutating the operator list directly. The late lowerer call supplies the active
`LayoutState`, which is synchronized after newly materialized adapter tensors
are added.

The same Pad module owns the guarded
`Transpose → Pad → Mul → Transpose → Add` NHWC propagation rewrite. It proves
static broadcast compatibility, rotates Pad and rank-four Mul constants only
when safe, preserves shared constants by cloning, updates metadata, and keeps
the legacy lowerer entry point as a wrapper.

Normalization-subgraph Pad propagation is also contained in
`passes/pad_layout.py`. Its matcher validates the complete reduction and
channelwise normalization region, clones constants that are shared outside the
transactional region, remaps axes and Pad specifications, and preserves legacy
fan-out adapters. The module has no source-line limit; the 2,000 threshold is
reserved exclusively for ONNX corpus node-tier classification.

InstanceNorm decomposition followed by Pad is owned by the same Pad family.
The matcher retains its strict reduction, epsilon, coefficient, optional
legacy-consumer adapter, Pad-axis, and terminal-transpose guards. Shared tensor
mutation utilities maintain quantization, metadata, and lineage while the
lowerer exposes only a compatibility wrapper.

Flatten/global-normalization followed by Pad is likewise owned by
`passes/pad_layout.py`. Its complete reshape, reduction, reciprocal, affine,
Pad, and inverse-transpose topology is validated before rewriting; scalar and
layout-sensitive constants are handled by its nested guarded helpers.

The three most frequently repeated Pad layout passes now execute through
`run_pad_layout_cleanup`. At seven unchanged production positions it registers
`layout.pad_prepost_nhwc`, `layout.unary_pad_prepost_nhwc`, and
`layout.norm_subgraph_pad_prepost_nhwc` with priorities 10/20/30 in
`LAYOUT_PLAN`; the fallback recovery invokes the same runner with only the norm
spec enabled. Each rewrite accepts the state-owned ModelIRGraphIndex and
LayoutState. Input/output mutation, transpose removal, and required adapter
insertion update the differential index directly, while tensor pruning removes
layout entries. A shared model-only Pad/Transpose scan avoids state allocation
on irrelevant graphs, and indexed topology guards prevent transactional
snapshots unless the corresponding direct, unary, or normalization region can
exist. Existing compatibility wrappers remain available.

The strict `Transpose → Pad|MirrorPad → Mul(const) → Transpose → Add(const)`
family executes through `run_pad_mul_layout_cleanup` at its three unchanged
recovery positions. Its stable `LAYOUT_PLAN` ID is
`layout.pad_mul_posttranspose_add_nhwc`. The precondition proves the complete
producer/consumer topology, both inverse permutations, constant availability,
Pad specification, and static broadcast compatibility before opening a
transaction. Constant cloning and input/output rewiring update the shared
`ModelIRGraphIndex`; both structural Transpose removals use that same index,
and tensor pruning synchronizes the session `LayoutState`. This is a complete
indexed migration of the helper rather than a partial deletion-only change.

Decomposed InstanceNorm-Pad and flattened global-normalization-Pad propagation
use `run_normalization_pad_layout_cleanup`. The two unchanged pair positions
register `layout.instancenorm_pad_prepost_nhwc` then
`layout.flatten_globalnorm_pad_prepost_nhwc`; two later recovery positions
enable only the flatten spec. The first pair remains inside its convergence
loop and consumes the runner's per-pass statistics exactly as before. Both
rewrites now use indexed input/output mutation, structural transpose and
adapter changes, and LayoutState-aware pruning. Their model-only preflight
requires the common Transpose/Pad/Mean capability, while state guards trace the
actual upstream graph and distinguish rank-4 InstanceNorm decomposition from
the two-Reshape flattened normalization form before snapshotting.

Attention-specific layout propagation lives in
`passes/attention_layout.py`. Its first family member reconciles parallel
channel Mean and ReduceMax branches before Concat/MirrorPad/Conv. The pass is
defined by producer/consumer topology, reduction semantics, permutations, and
padding pairs—not by model names—and rewrites axes only after the entire
region is proven.

Singleton-channel MaxPool layout cleanup lives in
`passes/singleton_maxpool_layout.py`. The module owns the repeated
Reshape→MaxPool→Reshape/Binary/Cast bridge cleanup and the complete strict
SuperPoint-style singleton NMS MaxPool ladder. Both implementations preserve
their rank-4 singleton layout, fan-out, public-output, binary/logical topology,
and terminal NCHW adapter guards. Shared shape, constant, quantization, graph
mutation, and pruning helpers remain canonical; both legacy lowerer symbols
are thin compatibility wrappers. The two rewrites remain adjacent in their
three production positions.

Those adjacent rewrites now run through one ordered pass group with stable IDs
`layout.singleton_maxpool_binary_cast` and
`layout.singleton_nms_maxpool_nhwc`. The group builds one differential
`ModelIRGraphIndex` and one `LayoutState` only after its model-only preflight
finds both Reshape and MaxPool operators. Each spec then applies a local
producer/consumer topology guard before taking a transactional snapshot. All
input/output rewiring, operator insertion/removal, and tensor pruning update
the shared index and layout state; the lowerer no longer calls either raw
implementation directly. This preserves the legacy order while exposing
per-spec diagnostics and avoiding snapshots on irrelevant or guard-rejected
graphs.

The two immediately preceding singleton-Reshape cleanups live in
`passes/singleton_reshape_layout.py`. That family owns the strict
Reshape→unary→inverse-Reshape passthrough and consecutive inverse singleton
layout Reshape pair implementations. Their match conditions, static rank-4
shape requirements, singleton memory-order guards, fan-out/public-output
handling, rewiring, and pruning remain unchanged. The legacy lowerer symbols
are thin compatibility wrappers, so the family module is the single
implementation owner while public test imports continue to work.

The same family also owns the fully static 4D→2D Reshape/Concat/2D→4D Reshape
rewrite that keeps the concatenation directly in NHWC. Its rank, singleton
spatial, batch/channel, axis, single-consumer, and public-output guards remain
unchanged, as do its inserted side-input Reshape and metadata propagation. The
193-line implementation moved mechanically with an identical AST; both legacy
production positions retain the same relative order.

Those positions now call `run_flatten_concat_reshape_cleanup`, registered as
`layout.flatten_concat_expanddims_nhwc` in `LAYOUT_PLAN`. Model-only preflight
requires Reshape and Concat; an indexed strict-linear
Reshape→Concat→Reshape guard prevents snapshots on fan-out or unrelated graphs.
The callback shares one differential index for producer/consumer lookup,
inserted side-Reshape placement, Concat input/output mutation, downstream edge
replacement, structural removal, and layout-aware pruning. The legacy helper
remains a compatibility wrapper only.

Two related singleton-spatial rewrites are mechanically owned by the same
family. The first removes an NHWC→NCHW Transpose (optionally through Identity)
before a 2D flattening Reshape when both spatial dimensions are one. The second
keeps repeated singleton 2D→4D Reshape inputs, Concat, and a terminal
NHWC→NCHW Transpose directly in NHWC while reusing side adapters. Their 392
implementation lines moved with identical ASTs; all three production calls and
their original adjacency remain unchanged.

Production uses `run_singleton_spatial_reshape_cleanup` at two positions. The
first registers `layout.singleton_spatial_flatten` then
`layout.singleton_reshape_concat_nhwc`; the second disables the Concat spec and
runs only the spatial-flatten spec. Model-only Transpose preflight, exact
permutation/topology guards, transactional snapshots, and a shared differential
index preserve that legacy ordering. Edge rewrites, adapter insertion, output
retargeting, structural removal, and pruning all update `LayoutState`. Any
single-consumer removal decision needed after rewiring is captured before the
index mutation changes its live consumer set.

The family also mechanically owns singleton-safe layout Transpose→Reshape
canonicalization. It recognizes canonical and generalized singleton
memory-order equivalence, preserves dynamic batch signatures, remaps shape
metadata and constant side inputs, and protects the quantized
Logistic→Concat(axis=1) case. The 291-line implementation moved with an
identical AST; its five production positions still call a thin lowerer wrapper
in their original order for compatibility.

Production now uses `run_singleton_channel_transpose_cleanup`, registered as
`layout.singleton_channel_transpose_as_reshape` in `LAYOUT_PLAN`, at all five
positions including fallback IR. Model-only Transpose preflight and a rank-4
singleton memory-order guard avoid index/snapshot construction on unrelated or
non-singleton graphs. The callback shares a differential index for producer/
consumer protection checks and input-edge replacement, synchronizes the added
shape tensor through `LayoutState`, and leaves the op output identity stable.

Production uses one ordered pass group for that family, with stable IDs
`layout.singleton_reshape_unary_passthrough` and
`layout.consecutive_inverse_singleton_reshapes`. A model-only preflight scans
until it finds two Reshape operators; otherwise neither graph index nor
transaction state is built. Local single-user topology guards precede each
transactional callback. Both callbacks share one differential
`ModelIRGraphIndex` and `LayoutState`, including indexed edge rewrites, tensor
renames, operator removals, and layout-aware pruning. The lowerer has two
runner calls and no raw production calls for either implementation.

The mixed attention MirrorPad pass is the first post-lowering rewrite migrated
to the differential ModelIR index. It builds producer/consumer state once,
updates input edges through indexed lineage helpers, and removes the redundant
transpose through `ModelIRGraphIndex.remove_operator`, avoiding a full edge
rescan after each successful match.

All lowerer invocations of that pass now use
`run_mixed_attention_layout_cleanup`, registered as
`layout.mixed_attention_mirrorpad` in `LAYOUT_PLAN`. A cheap topology
precondition guards snapshots; successful rewrites update the session-owned
LayoutState as logical annotations change, and the shared pass state validates
graph and layout invariants transactionally.

Generic Conv-attention NHWC propagation is independently registered as
`layout.conv_attention_nhwc`. Its five production positions remain unchanged;
it is not grouped with the nearby SiNet-named tail rewrite because the sixth
mixed-attention recovery call has a different neighborhood and such grouping
would change the recovery order. The pass performs one model-only scan for the
common Transpose/Mean/Conv capability before constructing ModelIRPassState. A
candidate transaction reuses the state-owned `ModelIRGraphIndex` and refreshes
it once after each complete structural rewrite iteration, replacing the former
independent producer and consumer map builds. The legacy lowerer function
remains a compatibility wrapper around the same implementation.

The contiguous generic QKV bridge tail is registered as one ordered group at
all three production positions. `layout.qkv_shared_pretranspose` (priority 10)
hoists three sibling Slice branches behind one NHWC-to-NCHW transpose;
`layout.qkv_weighted_sum_bridge` (priority 20) then moves the weighted Sum tail
back to NHWC. Both passes use the same ModelIRGraphIndex and LayoutState,
perform indexed input mutation and structural remove/insert operations, and
prune layouts with tensors. Their state-level guards inspect the exact local
producer/consumer topology before taking a transactional snapshot. A broader
single model-only scan still avoids constructing state when neither Slice nor
weighted-Sum capability is present. The four earlier QKV canonicalization
steps remain outside this group until their own indexed migration is complete.

The QKV `SPLIT → RESHAPE × N` collapse implementation is also owned by
`passes/attention_layout.py`; the lowerer exposes only its compatibility
wrapper. It recognizes static, rank-preserving reshapes whose apparent
permutation moves singleton axes without changing memory order, inserts one
pre-Split Reshape, retargets the Split outputs, and removes every post-Split
Reshape. Input/output mutation and structural insertion/removal update one
ModelIRGraphIndex throughout the rewrite, while tensor pruning updates the
optional LayoutState. This pass remains at its two original raw call positions
until the other three QKV prefix rewrites are indexed and can join one ordered
prefix group without changing recovery order.

The preceding QKV/KV sibling-Slice canonicalization is differential-index
aware as well. It enumerates consumers from the shared ModelIRGraphIndex,
inserts the replacement Split through the structural index API, resolves each
matched Slice by object identity through that index, and removes the branches
without an edge-map refresh. Its strict two-or-three equal contiguous chunk,
static-shape, rank, and output-shape guards are unchanged. LayoutState entries
for pruned begin/size tensors are removed when a session state is supplied.

The earlier QKV/KV gather-to-slice rewrite follows the same contract. It reads
source and branch consumers from one ModelIRGraphIndex, converts each
shape-restoring Reshape operator in place to Slice using indexed input
replacement, resolves matched Gather operators by indexed identity, and
removes them structurally. Output names, shapes, the axis-zero scalar Gather
guard, exclusive Gather-output consumer guard, and two/three-branch handling
remain unchanged; pruning also removes stale LayoutState entries.

The initial shared QKV gather/reshape/transpose hoist is differential-index
aware too. It inserts the shared Reshape and Transpose through the structural
index API, retargets each Gather through indexed input mutation, rewires the
old branch-transpose output consumers through the index, and removes the
per-branch Reshape/Transpose pairs by indexed identity. Its public outputs and
two/three-branch order remain stable and LayoutState follows tensor pruning.

At the two production locations where these four rewrites were contiguous,
`run_qkv_attention_prefix_cleanup` now registers them as one `LAYOUT_PLAN`
group with priorities 10–40 and stable IDs
`layout.qkv_gather_layout_hoist`, `layout.qkv_gather_to_slice`,
`layout.qkv_slice_to_split`, and `layout.qkv_split_reshape_collapse`. The group
shares one ModelIRGraphIndex and LayoutState across the dependency chain. A
model-only op summary avoids state construction on irrelevant graphs; precise
indexed topology guards inspect Gather index sets, Slice chunks, and
Split/Reshape consumers before taking each transactional snapshot. The three
later QKV bridge-tail calls remain a separate two-spec runner because one is a
standalone terminal recovery position.

Generic structural deduplication lives in `passes/graph_cleanup.py`. Duplicate
Transpose fan-out cleanup uses one `ModelIRGraphIndex`, rewires only indexed
consumers through the lineage-aware bulk input replacement helper, and removes
duplicates through the structural index API. A focused test instruments
`refresh()` and requires exactly one initial full index build regardless of the
successful rewrite.

Duplicate Reshape fan-out cleanup shares the same indexed contract. Target
shapes are compared semantically from options or constant inputs, public output
names are protected, and only indexed consumers of a duplicate output are
rewired before structural removal. Its instrumentation likewise permits only
the initial index refresh.

Core layout-Transpose chain cleanup is mechanically isolated in
`passes/layout_transpose.py`. The family owns identity-permutation and inverse-
permutation helpers plus elimination of identity Transposes, strict inverse
pairs, inverse fan-out branches, and composition of consecutive pairs. It
preserves public outputs, explicit layout-boundary markers, and the specialized
Softmax chain. All three function ASTs are identical to their former lowerer
definitions; the lowerer re-exports both shared helpers and retains a thin
optimizer wrapper for compatibility.

All thirteen production positions now call `run_layout_transpose_cleanup`,
registered as `layout.transpose_chain_cleanup` in `LAYOUT_PLAN`. Model-only
Transpose preflight and an indexed identity-or-consecutive candidate guard
avoid state/snapshot construction for unrelated or isolated non-identity
Transposes. Identity removal, strict inverse-pair removal, fan-out branch
bypass, and pair composition share one differential index per invocation;
permutation tensor updates, edge rewrites, structural removals, and pruning are
validated transactionally against `LayoutState`. The runner derives `changed`
only from rewrite counters, never from the diagnostic iteration count.

The same family mechanically owns strict
NHWC→NCHW-Transpose→Gather→NCHW→NHWC-Transpose axis remapping. It remaps the
Gather axis into NHWC, preserves `batchDims=0`, retargets the final output, and
removes the pre-Transpose only when it has no remaining users. Its 134-line
implementation moved with an identical AST; all eight production positions
retain their original order.

Production now calls `run_transpose_gather_axis_cleanup`, registered as
`layout.transpose_gather_axis_nhwc` in `LAYOUT_PLAN`, at all eight positions.
Model-only Transpose+Gather preflight and an exact indexed pre/Gather/post guard
avoid snapshots for unrelated, fan-out, or nonzero-`batchDims` graphs. Gather
input/output retargeting, axis mutation, conditional pre-Transpose retention,
post-Transpose removal, and pruning share one differential index and
`LayoutState`; the legacy helper remains a compatibility wrapper.

The stricter channel-axis/multi-post variant is mechanically adjacent in the
same module. It requires an exclusive NHWC→NCHW pre-Transpose, rank-four
Gather with `axis=1`/`batchDims=0`, and one or more non-public inverse-post
Transposes; it rewrites the Gather to NHWC axis 3 and coalesces post aliases.
The full 134-line implementation moved with an identical AST, while the
lowerer keeps a compatibility wrapper and its four production positions stay
unchanged. It remains separate from the general single-post axis-remap runner
so pass ordering and public-output behavior do not change implicitly.

All four positions now call `run_transpose_gather_channel_fanout_cleanup`,
registered as `layout.transpose_gather_channel_fanout` in `LAYOUT_PLAN`.
Model-only Transpose+Gather preflight and an exact indexed guard enforce the
exclusive pre edge, rank, axis, batch dimensions, complete inverse-post fan-out,
and public-output protection before snapshotting. Gather retargeting, alias
rewiring, structural removal, and pruning share one differential index and
`LayoutState` transactionally. Three normal positions use the conversion
session state; the separately constructed no-layout fallback IR lets the
runner construct its own LayoutState while still recording session diagnostics.

Strict canonical NHWC→NCHW Transpose→unary-chain→inverse Transpose passthrough
is mechanically owned by the same family. Only linear layout-agnostic unary
chains are eligible; public intermediate outputs and pre-Transpose fan-out are
rejected, while final dtype/quantization/shape metadata is preserved. The
157-line implementation moved with an identical AST before its indexed
migration; the lowerer keeps the compatibility wrapper for external callers.

All six production positions now call
`run_transpose_unary_passthrough_cleanup`, registered as
`layout.transpose_unary_passthrough` in `LAYOUT_PLAN`, in their original order.
A model-only Transpose preflight avoids state construction when no Transpose is
present. The indexed guard exactly verifies the canonical pre-permutation, a
strictly linear supported unary chain, protected graph outputs, and the inverse
post-permutation before a transactional snapshot is taken. Rewiring and both
structural removals use the shared differential `ModelIRGraphIndex`; tensor
pruning and layout reconciliation use the invocation's `LayoutState`. Thus the
rewrite no longer rebuilds a consumer map for every iteration or mutates the
operator list outside the structural index API.

The adjacent unary fan-out bridge rewrite is also mechanically owned by
`passes/layout_transpose.py`. It folds one pre-Transpose and one unary op into
a representative inverse-post branch, coalesces additional inverse-post
branches, and preserves non-Transpose legacy consumers through one adapter.
The 149-line implementation moved with an identical AST, while the lowerer
retains only a thin compatibility wrapper.

All seven production positions now call
`run_transpose_unary_fanout_bridge_cleanup`, registered as
`layout.transpose_unary_fanout_bridge` in `LAYOUT_PLAN`. Model-only preflight
requires both a Transpose and a supported unary family before building pass
state. The indexed guard preserves support for any valid inverse permutation,
verifies the single pre→unary edge, classifies every unary-output consumer as a
matching inverse post or a legacy consumer, and protects public tensors before
snapshotting. Edge rewrites, branch coalescing, optional adapter retargeting,
and structural removal share one differential index and `LayoutState` under a
transactional invariant check. The implementation no longer rebuilds a global
consumer map inside its rewrite loop.

The related unary→binary full-post fan-out implementation is mechanically
owned by the same family. It preserves non-commutative operand order, matching
pre-permutations, static/dynamic broadcast metadata, quantization-grid guards,
multiple inverse-post branches, and the optional legacy adapter. Its complete
305-line implementation moved with an identical AST after broadcast signature
reconciliation became a shared core helper. The lowerer retains a thin
compatibility wrapper.

All six production positions now invoke
`run_transpose_unary_binary_fanout_bridge_cleanup`, registered as
`layout.transpose_unary_binary_fanout_bridge` in `LAYOUT_PLAN`. Model-only
preflight requires Transpose, the supported unary family, and the supported
binary family. Before snapshotting, the indexed guard checks both candidate
operand orientations, exact producer/consumer exclusivity, matching and inverse
permutations, public outputs, per-tensor quantization compatibility, static
broadcasts, and intermediate/post shapes. The rewrite shares one differential
index and `LayoutState` for all edge retargeting, fan-out coalescing, optional
adapter retention, structural removals, and pruning. These indexed
implementations have no whole-graph producer/consumer-map rebuilds or direct
operator-list deletes.

Trailing output-side layout Transpose passthrough is mechanically owned by the
same family. It handles direct terminal output Transposes and strictly linear
unary/singleton-binary tails while preserving protected layout boundaries,
Softmax output contracts, symmetric binary bridge candidates, public names,
and inverse-permuted metadata. The full 271-line implementation moved with an
identical AST before indexed migration; the lowerer keeps a compatibility
wrapper.

All four production positions now invoke
`run_trailing_output_transpose_cleanup`, registered as
`layout.trailing_output_transpose_passthrough` in `LAYOUT_PLAN`. Model-only
Transpose preflight avoids state construction on irrelevant graphs. The
indexed guard follows the exact direct-terminal or unary/singleton-binary chain
topology, shares the symmetric binary bridge predicate with the rewrite, and
checks layout permutations, protected boundaries, public outputs, internal
fan-out, and Softmax before snapshotting. Producer retargeting, chain-head
rewiring, structural removal, metadata reconciliation, and pruning use one
differential index and `LayoutState` transactionally. With this migration the
family module again has no full producer/consumer map builders or direct
operator-list deletion.

General consecutive Reshape passthrough cleanup is also owned by
`passes/graph_cleanup.py`. It covers metadata-identical no-op Reshapes,
fan-out-safe bypass of a second Reshape, and strict single-user chain collapse,
while preserving public names, mutable/dynamic boundaries, semantic-rank
markers, and ONNX `0`/`-1` target semantics. The seven legacy lowerer call
sites remain available through a thin compatibility wrapper; the implementation
moved mechanically with an identical AST.

Production now invokes `run_consecutive_reshape_cleanup` at all seven positions
with stable ID `cleanup.consecutive_reshape_passthrough` in
`POST_LOWERING_CLEANUP`. Model-only preflight rejects graphs without Reshape;
the indexed guard detects metadata no-ops or Reshape consumers before taking a
transactional snapshot. Producer/consumer lookup, output retargeting, fan-out
input bypass, structural removal, and layout-aware pruning all use one
differential `ModelIRGraphIndex` and `LayoutState` per invocation. The fallback
IR position builds local state but reports into the same session diagnostics.

The four later reshape-only recovery positions also execute through
`run_duplicate_fanout_cleanup(include_transpose=False)` instead of calling the
compatibility helper directly. This preserves their original placement and
does not introduce Transpose deduplication at those phase boundaries. The
three primary-IR calls reuse the session `LayoutState` and diagnostics; the
fallback-IR call builds local pass state while reporting through the same
session diagnostics. An AST contract fixes the four reshape-only invocations.

Scalar clamp canonicalization is the first linear fusion in
`passes/graph_cleanup.py`. It replaces the strictly guarded
`Maximum(x, 0) → Minimum(..., 1)` chain with `Relu0To1`, updates the surviving
operator through the index-aware input helper, and removes the obsolete
producer without rebuilding edge maps. Singleton finite-float reading has one
canonical owner in `core/model_ir_utils.py`. Its production call runs through
`run_clamp_cleanup` with stable ID
`canonicalize.scalar_clamp_relu0to1`. The runner reuses the session LayoutState,
skips transactional snapshots when no Maximum-to-Minimum edge exists, and
removes pruned intermediate layouts together with their tensors. The legacy
raw rewrite remains available as a compatibility wrapper target.

The adjacent float-only `Maximum(data, scalar-zero) → Relu` canonicalization is
owned by the same graph cleanup family. Its input2-only constant guard and
FLOAT16/FLOAT32 restriction remain unchanged. Production invokes
`run_maximum_zero_relu_cleanup` at the original terminal position with stable
ID `canonicalize.maximum_zero_relu`; indexed input mutation, LayoutState-aware
pruning, transaction rollback, precondition skip, and internal diagnostics use
the same contracts as scalar zero-to-one clamp cleanup. The legacy lowerer
symbol delegates to the family implementation.

Consecutive floating Mul constant folding is also owned by
`passes/graph_cleanup.py`. It requires two strict binary Mul operators, an
exclusive non-output intermediate, floating path tensors, broadcast-compatible
floating constants, and a finite fused value. The implementation shares one
differential index, updates the surviving Mul input slots through indexed
mutation, removes the first Mul structurally, preserves constant quantization,
and registers the new fused tensor in LayoutState before pruning. All three
production and fallback positions call `run_consecutive_mul_constants_cleanup`
with stable ID `canonicalize.fold_consecutive_mul_constants`; their relative
order is unchanged and the legacy lowerer helper remains a wrapper.

Redundant integer Cast cleanup is isolated in `passes/cast_cleanup.py`. The
first ordered spec removes exclusive INT32/UINT32-to-64-bit alias casts while
retargeting the producer output and all downstream Cast input metadata. The
second collapses immediate INT64/UINT64-to-32-bit narrowing chains by retargeting
the first Cast. `run_redundant_cast_cleanup` fixes this original order with
priorities 10 and 20 and stable IDs `cleanup.cast_widening_alias` and
`cleanup.cast_narrowing_chain`. Both specs share one differential index,
LayoutState-aware pruning, transaction validation, and diagnostics. Public
aliases/outputs, fan-out, non-Cast consumers, signed/unsigned dtype pairing,
shape signatures, and quantization remain guarded. The two legacy lowerer
symbols are wrappers, and both terminal sweep pairs are one-to-one group calls.

Accuracy-sensitive terminal Quantize/Dequantize cleanup lives in
`passes/quantization_cleanup.py`. It removes a terminal Q→DQ pair only when the
incoming float tensor is itself produced by Dequantize, both quantized tensors
have identical dtype, quantized dimension, scale array, and zero-point array,
and every boundary/fan-out condition is exclusive. The output-preserving tensor
rename uses indexed producer/consumer mutation and LayoutState rename; both
terminal operators are then removed structurally. The two production positions
call `run_terminal_quantize_dequantize_cleanup` with stable ID
`cleanup.terminal_quantize_dequantize`. The exact-grid helper and raw rewrite
have one owner while lowerer symbols remain compatibility wrappers. Runtime
rounding equivalence is a mandatory integration gate.

Quantized PReLU bridge and fusion ownership lives in
`passes/quantized_prelu.py`. It contains the four repeatedly adjacent
Transpose→Dequantize→PReLU→Quantize/Transpose and
Dequantize→PReLU→Quantize/DepthwiseConv→Quantize rewrites. The implementations
retain their linearity, inverse-permutation, public-output, per-tensor
quantization, dtype, alpha remapping/quantization, and depthwise weight/bias
guards. Canonical quantization and graph mutation helpers come from
`core/model_ir_utils.py`; inverse-permutation validation comes from `ir.py`.
The legacy lowerer symbols are thin compatibility wrappers. The later
Dequantize→Reshape→Quantize rewrite remains separate because a TransposeConv
fusion occurs between it and this four-pass block at every production site.
The three production quadruples execute through `run_quantized_prelu_cleanup`
with stable `LAYOUT_PLAN` IDs `layout.transpose_dequant_prelu_quantize`,
`layout.transpose_dequant_prelu_transpose`,
`layout.dequant_prelu_quantize`, and
`layout.dequant_prelu_depthwise_quantize` at priorities 10–40. All
producer/consumer lookup, input/output mutation, structural removal, constant
cloning, and pruning share one differential `ModelIRGraphIndex` and
`LayoutState`. Model-only DQ/PReLU preflight avoids state creation on unrelated
graphs; indexed topology, inverse-permutation, public-output, dtype,
per-tensor-quantization, and constant guards run before snapshots. Include
flags retain isolated compatibility tests without changing production order.

Quantized Reshape fusion ownership lives in `passes/quantized_reshape.py`.
It folds only the linear Dequantize→Reshape→Quantize chain whose quantized
input/output dtypes and per-tensor quantization parameters are identical, while
protecting observable float intermediates and preserving output shape metadata.
The three production calls remain after their corresponding quantized
TransposeConv fusion, so this family does not reorder that dependency. The
legacy lowerer symbol is a thin compatibility wrapper.
All three production positions call `run_quantized_reshape_cleanup`, registered
as `layout.dequant_reshape_quantize` in `LAYOUT_PLAN`. The exact indexed guard
proves the exclusive DQ→Reshape→Q topology, protected intermediates, matching
non-float dtype, and identical per-tensor quantization before snapshotting.
Consumer lookup, Reshape input/output mutation, structural DQ/Q removal, and
pruning share one differential `ModelIRGraphIndex` and `LayoutState`; a
model-only three-op capability scan avoids state construction on unrelated
graphs.

Squeeze/Reshape identity cleanup also uses the differential graph index. It
normalizes explicit or inferred squeeze axes, proves the squeezed and restored
shapes, rewires only consumers of the round-trip output, then removes both
operators without rebuilding maps. Squeeze-axis normalization has one shared
implementation in `core/model_ir_utils.py`. All recovery-sweep invocations use
`run_squeeze_reshape_identity_cleanup`, registered as
`cleanup.squeeze_reshape_identity` in `POST_LOWERING_CLEANUP`. The existing
eight call positions remain separate because intervening rewrites can expose
new round trips; each invocation shares the session LayoutState and avoids its
transaction snapshot unless a Squeeze has exactly one Reshape consumer.

The adjacent `Squeeze(axis=0) → unary → shape-restoring Reshape` passthrough is
now owned by the same graph-cleanup family. Single-path chains collapse to the
unary operator; fan-out chains reorder to `unary(4D) → Squeeze(3D)` so existing
rank-3 consumers remain valid. Indexed input/output mutation and structural
remove/reinsert operations preserve topology without map rebuilds. At the six
production sites where this pass immediately preceded identity cleanup, the
runner registers `cleanup.squeeze_unary_reshape_passthrough` at priority 10 and
identity cleanup at priority 20 in one state/index group. The two standalone
identity sites remain identity-only, and the legacy unary helper is a wrapper.

Consecutive duplicate Transpose/Reshape cleanup calls are executed by
`run_duplicate_fanout_cleanup`. The runner registers stable
`cleanup.duplicate_*` pass IDs in `POST_LOWERING_CLEANUP` order, shares one
ModelIR index, validates invariants against that index, and deep-snapshots each
transaction. Invalid state restores the original ModelIR and refreshes the
index only on rollback. Standalone legacy pass functions remain compatible.

`LayoutState` has explicit synchronization, rename, removal, and ModelIR
validation operations. Session index refresh also refreshes layouts. The
ordered duplicate-cleanup group receives the session-owned state, synchronizes
it at the phase boundary, removes pruned tensors from it, validates it after
each pass, and resynchronizes it after transactional rollback. Canonical global
tensor rename and pruning helpers accept the same optional state.

`PassSpec.precondition` provides an explicit, deterministic prerequisite gate.
When false, the manager records `skipped_by_precondition` with zero iterations
and performs no fingerprint, snapshot, callback, or validation work. The
duplicate cleanup group uses cheap candidate scans so transactional deep copies
are reserved for graphs that may actually change.

Each current ModelIR runner also supplies a broader model-only `preflight` to
`run_model_ir_pass_group`. A false preflight emits the same ordered zero-iteration
skip results and diagnostics for every registered spec but does not construct
ModelIRPassState, GraphIndex, or LayoutState. The state-level precondition remains
the second, more precise semantic gate after a broad candidate is found. This
keeps irrelevant recovery sweeps to one cheap operator-type/topology scan and
eliminates repeated index builds and tensor-layout synchronization on large
no-candidate graphs.

Preflight and manager work is observable through internal diagnostic metrics,
without timers or public schema changes. Each pass event records operators
visited by model-only preflight, whether ModelIRPassState was constructed, and
the exact snapshot and fingerprint counts reported by OrderedPassManager.
`ModelIRPreflightResult` and shared any-operator/required-op-type scanners make
visited counts deterministic and allow early-match scans. These counters feed
Tier profiling while remaining absent from ModelIR metadata and public reports.

Managed corpus runs can collect an aggregate through a private sink without
changing CLI/API artifacts. `lower_onnx_to_ir` accepts an optional private
diagnostic list; the flatbuffer builder summarizes only model-ir-pass events
when `ONNX2TF_INTERNAL_PASS_METRICS_PATH` is set. The sequential bulk runner sets
that environment variable only around one subprocess, restores the prior value
in `finally`, reads the per-run JSON, and stores it under the managed entry's
optional `pass_metrics`. Bulk summary totals combine visited operators,
actual state builds, snapshots, and fingerprints. Metrics schema version 2
assigns a `group_sequence` to every runner call so shared preflight/state work
is counted once for multi-spec groups rather than once per event. Normal conversions do not
allocate the sink or write a metrics file.

Production constant-input materialization uses
`run_constant_input_fold_cleanup` from `passes/constant_fold.py`. Its three
specs preserve the original dependency order with priorities 10/20/30:
`canonicalize.constant_input_pad`, `canonicalize.constant_input_pool`, then
`canonicalize.constant_input_cast`. A materialized Pad output can therefore feed
Pool folding, whose output can feed Cast folding, all through one differential
index and LayoutState. Each operator is removed structurally and unused source
constants are pruned transactionally. The two former lowerer triplets are
one-to-one group calls; ScatterND and binary constant folding remain independent
compatibility helpers outside that pass group. They now accept an optional
shared GraphIndex, use differential operator removal and LayoutState-aware
pruning, and contain no direct operator-list deletion. Float32 and float16
artifact preparation builds one index per serialization artifact and shares it
across the consecutive ScatterND and binary folds. The terminal precision IR
is also the serialization IR when no later unsplit exporter needs the
pre-serialization form. Float16 is always terminal at this point. Float32 is
isolated by the TensorFlow- and Torch-free policy in `artifact_preparation.py`
only for an unsplit SavedModel or PyTorch request, because those exporters still
consume the pre-fold float32 graph; split exporters consume the original
partition source instead. This removes one complete tensor, operator, subgraph,
metadata, and constant-buffer clone from the normal direct path while retaining
copy isolation at the only later mutation boundary.

Serialization artifact lifetime is sequential as well as construction order.
Each write-only GraphIndex and precision/quantized ModelIR reference is released
immediately after its final writer or exporter consumer, before the next
precision or quantization variant is built. Strict-integer calibration sample
arrays are released as soon as their range report has captured the sample
count; calibration ranges and the in-memory report remain only through the four
variant builders and JSON write. Paths, timing records, and copied report
dictionaries are the only state retained for later weight export and result
assembly. This prevents otherwise sequential float32, float16, dynamic-range,
integer, full-integer, and INT16-activation graphs from accumulating in one
function frame.

`ir.py` is the single owner of `OperatorIR` and `TensorIR` element cloning.
The shared helpers copy every structural, axis-semantic, variable,
quantization, layout, and ONNX-provenance field plus NumPy constant buffers.
Precision wrappers retain their established normalized-layout and recursive
subgraph/metadata behavior; the quantization root clone deliberately retains
its legacy raw-layout, root-only, empty-metadata behavior. Quantization chooses
that policy but no longer duplicates the complete field list, so adding ModelIR
provenance cannot silently diverge across precision and quantized artifacts.
Split partition construction uses the same element helpers while making its
ownership policy explicit: final artifacts copy NumPy buffers, binary-search
size probes may borrow them, quantization metadata remains shared and immutable,
and raw logical/physical layout strings are preserved. The partition builder
therefore cannot omit a newly added common tensor or operator provenance field,
without changing the established buffer and quantization alias behavior.

Strict-integer report construction is request-scoped. A small internal
reporter owns the existing `tensor_ranges`, `quantized_tensors`, and
`quantized_ops` schema and serializes qparams only when the public builder was
called with `return_report=True`. The two orchestrator variants that feed the
calibration JSON keep full reports; default public model-only calls and both
INT16-activation variants perform no report-range or qparam serialization.
Quantization mutation never depends on report state, and the private builder's
default still collects a report for compatibility with direct internal callers.
Full-integer activation dtype resolution also materializes graph-input and
graph-output name sets once per variant. Every operator/tensor decision reuses
those immutable boundary sets instead of rebuilding them for each lookup.
Final strict-model validation collects referenced input/output tensor names in
the same pass that validates operator support. It preserves the legacy raw-name
and empty-name boundary behavior while removing the preceding duplicate
operator/edge traversal.

The strict builder's mutable `ModelIRGraphIndex` is lazy. Float-IO or mixed
external/internal dtype boundaries request it when consumer/producer rewiring
or pre/post operator insertion first occurs and then share that one instance.
Full-integer variants whose boundary dtype already matches the internal dtype
perform no boundary mutation and therefore build no graph index.

Late precision conversion in `passes/precision.py` is differential as well.
Constant floating DIV roots are captured from one operator-type index and each
eligible DIV is replaced in graph order by an optional Cast, reciprocal Mul,
and optional output Cast using indexed remove/insert operations. Precision-
sensitive reciprocal Mul restoration traverses indexed consumers, updates its
inputs, and changes MUL to DIV through `ModelIRGraphIndex.replace_operator_type`.
That primitive updates only the operator-type dispatch index while leaving
producer/consumer edges intact. Neither precision family rebuilds consumer
maps or replaces the complete operator list; the main lowerer path also keeps
the session `LayoutState` synchronized with materialized and pruned constants.

Fully static high-rank binary coalescing in `passes/high_rank_binary.py` no
longer rebuilds the complete operator list. It captures the original supported
binary objects from the type index, validates static broadcast coalescing, and
replaces each selected operator at its current graph position with two input
Reshapes, the rank-at-most-four binary, and one restoring Reshape. All
remove/insert operations update one `ModelIRGraphIndex` differentially; the
main lowerer path synchronizes the session `LayoutState` for the generated
shape and intermediate tensors.

Static high-rank BatchMatMul compression in `passes/high_rank_matmul.py` uses
the same contract. Initial `BATCH_MATMUL` objects are indexed once; each proven
rank-above-five operator has its inputs and output changed through differential
edge updates, then receives two input Reshapes and one output-restoring Reshape
at the same graph position. The original BatchMatMul object and options remain
intact. Complete operator-list replacement and direct input/output assignment
are forbidden by the ownership gate, and the production lowerer call supplies
the session `LayoutState`.

PyTorch native-runtime SAME AveragePool compatibility in
`passes/pytorch_compat.py` also follows the differential ModelIR contract. It
captures the initial indexed `AVERAGE_POOL_2D` objects, changes each proven
pool output through `ModelIRGraphIndex.replace_operator_outputs()`, and inserts
the correction `MUL` immediately after the live pool position. It no longer
rebuilds the complete operator list or assigns operator outputs directly. The
entry point accepts an optional shared graph index and layout state while
retaining its existing single-argument exporter call. This keeps the optional
PyTorch artifact boundary independent of TensorFlow and permits focused pass
validation without importing Torch.

Dynamic rank-one Unsqueeze shape materialization is isolated in
`passes/dynamic_reshape.py`. The lowerer retains its private compatibility
wrapper, while the pass indexes only initial `RESHAPE` candidates. A proven
rank-one case rewires the existing Reshape through the differential graph
index and inserts its runtime `SHAPE` and shape `CONCATENATION` immediately
before the live operator position. Folded higher-rank inputs retain the
existing direct `-1` shape-constant repair. Complete operator-list rebuilding
and direct input-slot assignment are forbidden by the ownership gate, and the
two main production calls synchronize the session `LayoutState`.

The same module owns fallback MatMul-flatten restoration. Placeholder
`RESHAPE` candidates and their sole consumers come from one graph index; a
proven rank-recovered `BATCH_MATMUL` is rewired differentially to the original
source and the obsolete Reshape is removed by object identity. Tensor pruning
is layout-aware. This removes another lowerer-wide consumer-map construction,
complete operator-list filter, and direct MatMul input mutation while retaining
the existing compatibility wrapper and call order.

`ModelIRGraphIndex.remove_operators()` provides batch removal for graph-wide
cleanup. It detaches all selected edges against the original index, filters the
operator list once, and remaps producer, consumer, object, and type indices
through one old-to-new table without calling `refresh()`. Dead-code pruning is
owned by `passes/graph_cleanup.py`: one reverse liveness walk retains graph
outputs and live variable-state mutations, then sends all dead positions to
that batch primitive and prunes tensors with the active `LayoutState`. Main
lowering calls pass session layout state; the compatibility wrapper remains in
the lowerer.

Unsupported-dtype Split fallback is isolated in `passes/split_fallback.py`.
It enumerates only initial `SPLIT` objects and retains the established LiteRT
dtype, constant-axis, rank, output-count, and split-size guards. Each proven
operator is replaced at its live position with an optional output-dtype `CAST`
and ordered `SLICE` operators through differential remove/insert operations.
Generated begin/size constants and layout state are reconciled once after the
rewrite. The lowerer retains a private compatibility wrapper and passes the
session `LayoutState`; complete operator-list construction is forbidden by the
ownership gate.

Dynamic Squeeze runtime-shape plumbing no longer rebuilds the lowerer's
operator list. The established matcher still converts each eligible Squeeze to
the same Reshape and records its `SHAPE`/`GATHER` prefix. After all direct
operator metadata changes are complete, one `ModelIRGraphIndex` is constructed
and the prefixes are inserted in reverse candidate order at their original
positions. Tensor pruning and newly generated shape tensors are reconciled with
the session `LayoutState`. Consequently, the only complete `model_ir.operators`
assignments left in the central lowerer are explicit snapshot rollback paths.

Dynamic-range quantization creates one lazy `ModelIRGraphIndex` over the already
deep-cloned graph only when an elementwise quantized constant requires a
`DEQUANTIZE` insertion or input rewire. All later uses share that tensor and
the same differential index. Conv/Depthwise/FC kernel-only weight quantization
does not mutate graph topology and builds no index. The former second
operator-cloning pass and complete list assignment remain removed, reducing
copy volume while preserving operator axis semantics and ONNX provenance
already carried by the initial ModelIR clone.

Full-integer Identity elision also uses one differential index. Its ordered
replacement table preserves Identity-chain resolution and the special graph-
output producer promotion rule. Retained operator inputs/outputs are updated
through index primitives, then all elided Identity objects are removed in one
batch compaction. A no-Identity preflight avoids index construction entirely.
The quantization module no longer assigns a rebuilt operator list for this
cleanup.

Strict integer boundary construction shares one graph index after Identity
elision. Graph-input replacement visits only indexed consumers, and full-
integer graph-output conversion updates the indexed producer rather than
rescanning every operator. Boundary `QUANTIZE` operators are inserted in input
order before the unchanged core graph and output `QUANTIZE`/`DEQUANTIZE`
operators are appended in output order after quantization analysis, preserving
report indices. The former `pre_ops + clone.operators + post_ops` assignment is
removed.

Serialization sanitization reuses the same `prune_dead_operators()` liveness
implementation. Its shallow container clone requests operator-only pruning,
then preserves the established constant-input stripping and tensor-pruning
order. Thus reusable source ModelIR objects and weight buffers remain untouched
while the serializer no longer owns a duplicate liveness walk or complete
operator-list filter. The shared pass keeps tensor pruning enabled by default
for lowerer cleanup and exposes the opt-out only for this serialization order.

Split/export structural rewrites use an append-only `_ModelIRRewriteBuilder`.
It deep-clones tensors, subgraphs, and metadata but starts with an empty
operator stream, avoiding an immediately discarded deep copy of every source
operator. Group-convolution expansion, BatchMatMul unfolding, and recurrent
unrolling emit directly into that stream and no longer retain a second
`rewritten_ops` list or assign it at completion. Unchanged operators are copied
with options, version, axis semantics, and ONNX node/op provenance intact.
Boundary-based partition cropping follows the same operator contract: its
newly constructed cropped ModelIR copies axis semantics and ONNX node/op
provenance in addition to inputs, outputs, options, and version. The crop
preflight materializes the original runtime-input set once for all requested
boundaries, and does not collect an unused second set of kept operator outputs.
Top-level boundary names no longer trigger recursive nested-subgraph name
collection; that diagnostic-only traversal is deferred until a requested name
is actually missing from the top level. Producer discovery and forward
reachability share one operator-stream scan, while missing-input validation,
required-tensor collection, and kept-operator materialization share a second
scan over only the retained indices. Cropped operators use the common element
clone contract with explicit deep-copied options and axis semantics, preserving
the crop API's independent-mutation behavior.

Dependency-safe split-point discovery uses one producer scan and one consumer
edge scan. Backward dependencies are represented as invalid-boundary range
deltas, while forward tensors enter and leave an active crossing set through
boundary events. This replaces the former full edge rescan for every possible
boundary and preserves the exact ordered report, including first-producer
selection, non-topological invalid boundaries, fan-out, external inputs, and
duplicate tensor names. Runtime is proportional to operators, edges, and the
crossing tensors that must be written to the report rather than
`boundary_count * edge_count`. The complete split plan constructs this
`ModelIRGraphIndex` once and reuses it for candidate discovery, partition-range
validation, and manifest edge construction. Its first-producer map is likewise
materialized once and passed through all three phases instead of being rebuilt
from the shared index. Standalone helpers accept optional caller-owned index
and producer-map state while retaining their existing call behavior.
When split output is requested by the artifact plan, this caller-owned source
index is created before final ModelIR validation and then remains current and
shared across validation, planning, and partition-file writing. The normal
non-split path creates no source split index. Standalone planning and writing
calls still construct their own index when none is supplied.
Partition input/output tensor collection likewise uses insertion-ordered
`seen` sets. It preserves first-seen ordering and duplicate suppression while
removing the quadratic list-membership cost from every binary-search candidate
and final partition build. Candidate boundary-output discovery queries the
shared consumer index instead of rescanning the complete graph suffix for every
candidate. The final split artifact writer also constructs one index and reuses
it across all emitted partitions. Binary-search size-estimation candidates
borrow immutable NumPy constant buffers from the source ModelIR instead of
copying every weight for every probe. The public partition builder and final
artifact writer retain the established independent-buffer default. Partition
dead-branch liveness still runs for every candidate, but input/output and
boundary collection is repeated only when that liveness result removes at
least one operator. Required indices are a sorted unique subset of the local
range, so equal cardinality proves the full range without another elementwise
comparison. Fully required ranges reuse their initial collections. Boundary
output discovery uses `ModelIRGraphIndex.has_consumer_at_or_after()`: consumer
indices are maintained in sorted graph order, so a suffix query reads only the
last index instead of allocating an `any()` generator over the consumer list.

Custom-op result metadata has a single TensorFlow- and Torch-free owner in
`artifact_metadata.py`. One operator-stream pass produces both the legacy raw
custom-code list and the normalized, deduplicated node-detail list, preserving
their intentionally different whitespace/empty-code behavior and deterministic
sort order. Export orchestration no longer performs two consecutive ModelIR
scans for these result fields.

PyTorch layout-only Transpose cleanup is owned by the Torch-free
`passes/pytorch_compat.py` boundary. It indexes initial `TRANSPOSE` objects and
their live consumers once. Internal adapters rewire consumers and are removed
differentially; public-output adapters are replaced in place by an Identity
through indexed remove/insert while retaining the established output-tensor
clone semantics. The PyTorch exporter no longer owns a consumer-map builder or
complete operator-list deletion, and the pass can be tested without importing
Torch.

Native-PyTorch WHILE compatibility expansion is isolated in the Torch-free
`passes/pytorch_control_flow.py` boundary. It owns subgraph lookup,
constant/alias guards, static and counter-bounded matching, shape-literal
creation, root-graph cloning, and both rewrite entry points.
`_clone_model_ir_without_root_operators()` deep-clones
tensors, metadata, and complete subgraphs but deliberately leaves the root
operator stream empty. Static trip-count and counter-bounded expanders consume
the original root stream once, deep-copy retained operators once, and append
retained or generated operators directly in established order. This keeps the
source and result independent and leaves WHILE body/condition subgraphs intact,
without cloning a root operator stream that would immediately be discarded.
Static-WHILE, counter-bounded-WHILE, and recurrent-sequence entry points use
copy-on-write preflight: a no-op returns the borrowed input, which is safe
because the following channel-first normalizer creates its own deep copy before
mutation. A proven expansion still clones before modifying any graph state.
The exporter imports only the two ordered rewrite entry points, so control-flow
canonicalization can be validated with synthetic ModelIR without importing
Torch.
Both matchers build one `ModelIRGraphIndex` per candidate body and reuse it for
all producer and Reshape-alias queries. Required edges with duplicate producers
are rejected before cloning, making the rewrite deterministic while replacing
the former repeated linear scans of the body operator list.
Each root operator is matched exactly once into an immutable-by-convention
rewrite-plan map before copy-on-write construction begins. The static and
counter-bounded emit loops consume those plans instead of invoking the matcher
again against the cloned graph, so candidate body indexes and structural guards
are not rebuilt during emission.
The exporter architecture gate detects attribute assignments through the AST,
so alternate local names cannot silently reintroduce complete operator-list
replacement.

Recurrent-sequence canonicalization and capability selection are separately
owned by Torch-free `passes/pytorch_recurrent.py`. The module defines the
supported 15/24-input unidirectional and 29/48-input bidirectional LSTM index
contracts, constant and optional-input validation, direct native RNN/LSTM
selection, and copy-on-write delegation to
`rewrite_model_ir_unroll_recurrent_ops()`. The exporter imports these functions
for both preparation and its generated-code execution environment instead of
implementing a parallel capability policy. Unsupported recurrent forms still
use the shared append-only split-planner rewrite, while directly supported or
absent recurrent ops return the borrowed graph for the subsequent owning
normalizer copy.

Native recurrent constant preparation has a separate Torch-free owner in
`pytorch_recurrent_codegen_policy.py`. It enforces required constant arrays and
builds LSTM gate-bias buffers in index order. Fully omitted biases produce one
zero synthetic tensor, while partially omitted bias groups remain an explicit
export error.

The same recurrent module owns legacy orphan step-tensor repair. It preflights
`_h_step_`/`_c_step_` names before allocating graph state, then uses one
`ModelIRGraphIndex` to find the corresponding shape-driven Reshape and update
all orphan consumers differentially. It no longer constructs ad hoc producer
and consumer maps or scans every operator once per candidate; graphs without a
candidate return without index construction.

Softmax-specific channel-first validation is isolated in Torch-free
`passes/pytorch_layout_validation.py`. Attention-like Softmax consumers and
Transpose-sandwich producers/consumers are resolved from one shared
`ModelIRGraphIndex`. The top-level exportability validator creates that index
only when it encounters an unknown-layout Softmax and reuses it thereafter,
removing the former per-tensor full graph scans. Duplicate sandwich producers
reject the exception path rather than selecting one by graph order.

The same module owns feature-last region closure. Seed discovery constructs one
`ModelIRGraphIndex`; `_propagate_feature_last_tensor_names()` then runs a
deterministic worklist over only adjacent producer/consumer edges. Its
bidirectional passthrough semantics and standard channel-layout Transpose and
factorized rank-three Reshape barriers match the former monotonic fixed-point
loop, while removing repeated complete operator scans as the region grows.

Forward channel-last annotation also uses a consumer worklist. Existing
channel-last tensors seed the queue; only indexed consumers in the unchanged
safe-op allowlist are inspected, and newly annotated rank-three through
rank-five outputs extend the queue. This preserves unsupported-op boundaries
and works for non-topological operator order without the previous repeated
complete scans.

The layout-application entry point is also owned by
`passes/pytorch_layout_validation.py`. It consumes a caller-provided graph
index when available and otherwise builds one `ModelIRGraphIndex`; an empty
preserve set returns before index construction. Preserved output producers,
including duplicate producers, are enumerated in graph order from that index
and reused for both initial Transpose/Reshape decisions and raw ONNX Reshape
shape restoration. Unrelated operators are not scanned. Constant-shape updates
remain in this single Torch-free canonicalization owner, while the exporter
contains only the ordered invocations.

Feature-last seed discovery and its rank-four island policy are co-located in
the same module. The module also owns preserved-region shrinking and restoration
of non-preserved tensors to channel-first layouts. These functions share the
collector's graph index or accept the normalizer's existing producer/consumer
maps, so the exporter no longer implements a second layout-region policy or ad
hoc fallback map construction.

Public PyTorch input/output layout bridges are now owned by the same Torch-free
layout module. Bridge eligibility is resolved before graph indexing, so a graph
whose public shape and layout already match its contract pays no indexing cost.
When a bridge is required, one lazily created `ModelIRGraphIndex` is shared
across every public input and output: indexed consumers and producers are
rewired differentially, input Transposes are inserted in the established
front-of-graph order, and output Transposes are appended in public-output order.
Tensor names, bridge metadata, layout permutations, and operator provenance are
unchanged while the former complete operator scans are removed.

General PyTorch-friendly unary, binary, Concat, Pack/Unpack, Split, resize, and
pool layout propagation is co-located in this module as well. One graph index
seeds a deterministic operator worklist; a changed tensor schedules only its
indexed producer and consumers. This retains Concat peer-layout back-propagation
and works for non-topological graph order without the former repeated complete
operator sweep to a fixed point.

The channel-first normalizer now creates one `ModelIRGraphIndex` after its
owning deep copy and shares it across feature-last collection, both friendly-
layout propagations, redundant-Transpose removal, ATAN2 canonicalization,
recurrent orphan repair, and final validation. Structural and edge changes use
the index mutation API, so the final residual-Transpose check consumes the
still-current consumer table instead of rebuilding an ad hoc map. The ATAN2
ones-like rewrite is owned by `passes/pytorch_compat.py`; irrelevant graphs
return before index construction when it is called independently.
An internal normalization result carries this current index into native
preparation's public-boundary bridge and shape-alignment stage, removing the
former second index over the same prepared graph. The public
`normalize_model_ir_for_pytorch_channel_first()` compatibility function still
returns only ModelIR, and the layout-agnostic error fallback creates its own
index after constructing a different fallback graph.
Native model-file generation likewise creates one index and supplies it to
feature-last collection before placing the index itself in the writer context.
Read-only producer and consumer compatibility properties expose that same
object's maps to the generated pipeline. Code generation therefore does not
build a second index or detach raw tables from their owner.

PyTorch convolution-filter physicalization is also owned by the Torch-free
layout module. It queries only Conv2D, depthwise Conv2D, transpose-Conv2D,
Conv3D, and transpose-Conv3D indices from the shared graph index, preserves
graph order, and tracks shared weight names so each buffer is permuted exactly
once. The normalizer supplies its existing index; no extra operator scan or
index build is required. The pre-permutation exclusion set uses the same
op-family declaration and shared index, so unrelated operators are not scanned
a second time merely to identify kernel buffers.

Residual channel-layout Transpose validation and its Reshape-only exception are
owned by `passes/pytorch_compat.py`. The normalizer queries only indexed
Transpose operators and reuses the current consumer table. Public bridge,
preserved feature-last, recurrent rank-three, unknown-layout, shape-only, and
Reshape-only exceptions retain their established guards and error signature.

Layout-sensitive axis, vector, pad-matrix, Transpose-permutation,
transpose-convolution output-shape, and Reshape-target canonicalization is owned
by `passes/pytorch_layout_validation.py`. It queries only the affected op-family
indices from the normalizer's shared graph index and retains the one-rewrite
rule for shared constant tensors. Reshape target selection and channel-last
name recognition live in `pytorch_layout_utils.py`, so canonicalization and
native code generation share one policy without an exporter callback.
The subsequent Reshape target synchronization step is co-located with that
rewrite and enumerates only indexed Reshape operators, while preserving its
feature-last exclusion and constant dtype behavior.

Channel-first exportability validation is now part of the same Torch-free
layout owner. It enumerates only layout-sensitive op-family indices, reuses the
normalizer's graph index for attention and Transpose-sandwich Softmax
exceptions, and retains the existing recurrent, ScatterND, degenerate sequence,
public-bridge, preserved-region, and safe rank-four island guards. The exporter
contains only the ordered validation call and shared error type.

Public boundary shape/layout reconciliation and recurrent-sequence context
detection are co-located in the layout owner. The channel-first normalizer uses
its existing graph index to identify recurrent op families, while the
layout-agnostic fallback path retains direct detection. Explicit boundary
signatures, public layout metadata, dynamic-dimension concretization, and the
rank-three recurrent NWC default remain unchanged.

The complete channel-first normalization orchestration is owned by the new
Torch-free `passes/pytorch_normalization.py` module. It creates the owning deep
copy, constructs exactly one graph index, invokes the ordered layout,
constant/filter, compatibility, recurrent-repair, boundary, and validation
steps, and returns the validated copy. The remaining public-output Transpose
inspection uses the same index rather than a raw operator scan. The exporter
imports this single orchestration entry point, making the full normalizer
directly testable without importing Torch.

Native PyTorch preparation is owned by the same module. Static-WHILE,
counter-bounded-WHILE, and recurrent compatibility rewrites execute in their
established order before the channel-first normalizer. One root op-family scan
dispatches only the rewrite entry points that can apply; after WHILE expansion,
the root types are recomputed so recurrent operators introduced from a body are
still handled. Direct recurrent capability selection itself is a single scan
that stops at the first operator requiring unroll. The normalizer's current
index then continues through public bridge insertion and final public
shape/layout alignment; only the distinct layout-agnostic fallback constructs
its own index. Recursive root/subgraph op-type collection and the fallback
capability policy are also centralized here; the exporter retains compatibility
imports for the public internal entry points and artifact generation only.

Native PyTorch operation emission is being separated behind a Torch-free,
callback-driven boundary in `pytorch_emitters.py`. The unary operator family is
the first owner: its complete expression table and channel-first/channel-last
bridge emitter moved together, while tensor naming, alignment, omission, and
shape policy remain explicit callbacks supplied by the exporter. This keeps the
generated Python source and runtime-helper selection unchanged, permits direct
ModelIR code-generation tests without importing Torch, and prevents the central
exporter from regaining a duplicate unary implementation through an architecture
ownership gate. Subsequent binary and shape-transform families can move through
the same boundary independently.

The shape-transform family now follows that boundary as well. ReverseV2,
ExpandDims, Squeeze, Pack, Unpack, and Split source emission is owned by
`pytorch_emitters.py`, including runtime axis normalization and multi-output
statement construction. Constant integer-vector decoding moved to the shared
Torch-free `pytorch_codegen_utils.py` owner so the emitter and the generated
pipeline's remaining families use one policy. The generated pipeline keeps the
same imported global function names, avoiding a compatibility wrapper or source
template change.

Binary expression selection and source emission are also owned by
`pytorch_emitters.py`. Integer division, scalar comparison coercion,
channel-first aliases, fused activations, channel-last materialization, and
runtime broadcast alignment now live together with the complete binary dispatch
table. The stored generated pipeline still calls its original exporter global;
that name is a thin compatibility adapter which injects the exporter's existing
broadcast target-shape policy into the Torch-free emitter. The adapter contains
no statement-emission logic, and its shape policy can be separated later without
changing the pipeline contract.

Transpose source emission is directly owned by `pytorch_emitters.py` with no
exporter adapter. It reuses the Torch-free layout and compatibility policies for
permutation decoding, inconsistent-layout elision, residual Reshape bridges, and
channel-first aliases. Fold-only Slice/binary consumers, omitted materialized
channel-last aliases, constant/runtime permutations, and runtime-helper imports
retain their established behavior. The exporter and stored generated pipeline
continue to resolve the same imported function name.

Concat source emission and its channel-last axis-sensitive consumer guard are
co-located in `pytorch_emitters.py`. GatherElements coordinate construction,
channel-first concatenation and fused activation, NHWC/NDHWC materialization,
omitted aliases, and exact-shape runtime fallback therefore share one focused
owner. Gather, Split, and Unpack channel-axis consumers continue to block unsafe
channel-first emission. The generated pipeline again resolves the unchanged
imported emitter name directly.

Generic output expression finalization is also owned by
`pytorch_emitters.py`. Shape-preserving expressions bypass alignment, while
shape mismatches import the runtime alignment helper; direct-module outputs
first apply the declared rank-3/4/5 logical-layout permutation and then align
to the shared target-shape literal. Both helpers retain callback-based shape
access and have no exporter-local duplicate.
Fused activation statement generation is co-located with these emitters.
Direct ReLU, clamp, SiLU, and Tanh forms retain their compact source, while
unknown activation names use the shared generated-runtime fallback.

The native codegen entrypoint retains only writer-reachable orchestration
wrappers. The former `_assemble_native_model_source` stub had no caller and
self-recursed under its own name, so it is removed and guarded from
reintroduction by the architecture test.

Fast generated-source precanonicalization begins with a small Torch-free value
policy in `pytorch_fast_precanonicalize_policy.py`. It owns reversible NHWC/NCHW
rank-four pad-axis conversion, conservative channel-count inference, and source
identifier extraction. The larger repair context consumes these shared values
without maintaining local duplicates. Its dataclass, one-pass source-line
collector, alias resolver, and CF/NHWC classification helpers are co-located in
the same owner. The collector records static shapes, consumers, buffer channel
counts, Conv block arity, module edges, and propagated alias layout evidence.
It also records complete rank-four registered-buffer shapes during that same
source scan. The orchestrator reuses this map for constant-alignment repair and
does not compile another buffer regex or rescan every generated source line.
Buffer channel/shape maps, Conv output channels, and module producers remain
read-only after context construction, so the orchestrator queries those four
maps directly instead of allocating full dictionary copies. Dynamic CF/NHWC
sets remain per-pass mutable copies.
Preferred channel selection, scored consumer-layout inference, and recursive
channel-last spatial-consumer detection query the same context in this owner.
They remain conservative when source evidence ties or cycles.
Split-axis repair and channel-first Resize/Pool target-shape repair are also
co-located with these queries. Split voting inspects only later consumers;
Resize and Pool preserve immediate layout bridges and rewrite only when CF
input or consumer evidence is stronger than the declared channel-last form.
Static Pool NHWC selection is co-located here as one ordered decision. A MaxPool
with a channel-last spatial consumer takes priority; otherwise an immediate
NHWC bridge or unambiguous NHWC input evidence enables channel-last execution
and shape normalization. An immediate CF bridge continues to block the second
case. The exporter applies the returned statement and layout evidence before
its existing CF repair.
Two CF Pool-neighbor corrections are also one indexed policy decision. An
exact CF constant-Pad→channel-last MaxPool→permuted-Conv chain restores the
Pool to CF and returns an explicit short-circuit so later precanonicalization
rules retain the old `continue` boundary. A CF Pool followed by local response
normalization repairs its static target without stopping the scan. Both use the
shared statement decoders and repair context.
Dynamic-shape Pool layout repair is another indexed decision with the same
explicit short-circuit contract. Immediate NHWC/CF bridge evidence controls
channel-last execution; CF average pooling derives a concrete target from a
bounded direct aligned consumer or one binary hop followed by an aligned
consumer. The binary hop and aligned statements use shared parsers instead of
orchestrator-local regexes.
Simple generated aliases are handled by one indexed layout decision. It owns
the guarded CF→NHWC boundary needed before a rank-three reshape and before the
channel-last PReLU/permuted-Conv consumer forms, including exact-shape runtime
alignment when known. Aliases not rewritten propagate existing CF/NHWC
evidence. All permuted-Conv statement consumers now reside in this policy
owner, so the exporter no longer imports that decoder directly.
Aligned scalar-binary shape reconciliation is also policy-owned. It rewrites
only when the previous aligned assignment and the next aligned or Softmax
consumer agree on one rank-four shape and the current shape is the exact H/W
swap. The rule keeps its positional scalar grammar and uses the shared aligned
and Softmax statement decoders; their remaining consumers are policy-local, so
the exporter does not import them.
The subsequent aligned-binary fallback is policy-owned as a separate decision.
It preserves the narrower generated-statement grammar and runs only when the
general binary alignment repair made no change. Two channel-first operand names
are not sufficient by themselves: an immediate matching BN constant operation,
direct return, or channel-first Resize with matching channel evidence is also
required. Channel mismatches, channel-last Resize, mixed-layout operand names,
and already-channel-first shapes remain no-ops. The exporter applies returned
layout evidence before its existing Resize rule, preserving the established
general-repair → downstream-fallback → Resize ordering.
Channel-first Resize repair now has two explicit policy stages. The general
`_repair_cf_resize_target_shape` decision remains first; when it does not
rewrite the statement, the exact input/BN-evidence fallback may normalize the
target and publish CF evidence for the following Pool and aligned-constant
rules. Immediate direct and reshaped BN statements share policy-owned parsers,
and a registered BN constant channel count is preferred when present but is
not required for the legacy input-evidence fallback. Explicit NHWC input and
already-channel-first target shapes remain no-ops, preserving the established
Resize-to-Pool scan behavior.
The two aligned BatchNorm-constant forms are handled by one policy decision
after Resize, Pool, and Concat evidence has been applied. A direct constant is
reshaped only when its registered channel count matches the generated target;
an already-reshaped constant retains the legacy normalization based on its
explicit reshape channel. Both forms require a CF-like input and a generated
BatchNorm attribute name. Their exact direct/reshaped statement grammars are
shared with the preceding Resize evidence rule, and the repaired output is
published as CF evidence for later normalization decisions.
Local-response-normalization output propagation is policy-owned as a state-only
decision. It preserves the generated parser grammar, marks an output CF only
when the input has exact dynamic or suffix evidence, copies only rank-four
static shapes, and removes stale NHWC evidence for that output. The source line
is not rewritten, so the exporter does not mark the file changed solely for
this propagation; the updated evidence is consumed by the subsequent
Softmax/ReduceMax and Pool decisions.
Successful aligned-binary, Resize, and Pool rewrites record literal output
shapes through one policy-owned cache helper. It accepts only a literal
`target_shape=[...]` or the exact trailing aligned-shape form and leaves the
existing cache untouched for dynamic or unparseable expressions. The exporter
still invokes the update immediately after each rewrite, so every later rule in
the ordered scan observes the same shape evidence as before extraction.
The NHWC AveragePool-to-binary bridge repair and its channel-last spatial-pool
restoration wrapper use the same owner. AveragePool, binary-anchor, and multiply
target shapes are normalized as one chain only when NHWC producer and consumer
evidence agree; otherwise the repair remains a no-op. Constant-pad axis repair
and reshape-to-permute replacement reuse the shared parser and layout context.
On a successful bridge rewrite, the same policy decision now publishes all
four affected names as NHWC, removes their stale CF evidence, and records their
legacy normalized state shape. The state shape is deliberately recomputed from
the pre-rewrite Pool shape after the layout-set update, matching the old
exporter-side sequence rather than silently switching the cache to the rendered
rewrite target. The exporter retains only the changed flag and immediate Pool
reparse required by the following ordered decisions.
Channel-first binary alignment repair is co-located with that context as well.
It normalizes only rank-four targets backed by CF operands or CF consumer
evidence, preserves already normalized targets, and leaves explicit binary
anchor chains and any NHWC operand untouched.
Concat-axis and terminal-classifier tail repair complete the standalone layout
helpers in this owner. Concat rewrites only when every input is CF-like; the
tail repair adds an explicit singleton channel or removes a redundant trailing-
singleton reshape only when its source is CF-like. The exporter retains the
ordered precanonicalization orchestrator and imports these focused decisions.
The older post-helper `torch.cat(dim=3)` regex block was a strict subset of the
shared concat helper's alias-aware CF guard and is removed; concat axis repair
therefore has one decision point.
The singleton-reshape-to-CF-binary repair is another focused rule in this
owner. Its local patterns, feature-axis guard, following binary relationship,
rewrite, and CF evidence update are applied atomically through one indexed
helper; the orchestrator retains only its ordered call.
PReLU output evidence propagation and channel-last Gather-slice repair are also
focused rules in this owner. PReLU retains one rule-local module-call pattern;
Gather reuses the shared assignment decoder and rewrites only CF-like inputs
from axis 3 to axis 1 while recording the output as CF-like.
Rank-four NHWC registered-buffer binary alignment is likewise indexed here. It
consults the context's shared buffer-shape map and inserts the constant
NHWC-to-NCHW Permute only when `[1,H,W,2]` exactly matches the requested
`[1,2,H,W]` alignment.
DepthToSpace-adjacent Gather repair is indexed here as well. One policy helper
owns structural channel-first inference through a bounded preceding Concat,
Gather output shape/layout evidence, guarded removal of a following Conv input
Permute, and channel-last DepthToSpace index correction. It reuses the shared
permuted-Conv assignment decoder and parses each preceding assignment once.
Channel-first Softmax and ReduceMax axis repair also use focused parser-backed
helpers in this owner. Softmax preserves beta and moves the rank-four target
channel; ReduceMax preserves keepdims and changes only axis 3 to axis 1. Pool
lookbehind and scalar-binary lookahead consume shared Pad/Softmax parser tuples,
so the exporter keeps no duplicate regex for these statements.

The direct-module dispatcher is being decomposed by operator family rather than
moved as another monolith. Its unidirectional RNN, unidirectional LSTM, and
bidirectional LSTM source emission now delegates to the Torch-free recurrent
emitter. State-slot selection continues to use the shared recurrent arity/index
contract, and the generated alignment call and runtime-helper import are
unchanged. Conv, transpose-Conv, fully connected, and PReLU branches remain in
the dispatcher for subsequent independent extraction.

FullyConnected and PReLU are the next separated direct-module families.
FullyConnected owns its direct module call and fused-activation ordering in a
small emitter. PReLU owns parameter-count detection, NHWC/NWC/NDHWC parameter-
axis bridges, shape-preserving fast paths, and alignment fallback in a separate
emitter. The dispatcher retains only ordered delegation for both families;
convolution families and fused-module emission remain its final substantial
responsibilities.

TransposeConv2D and TransposeConv3D now have separate focused emitters. They own
constant or tensor-metadata fallback output shapes, module weight/bias/stride/
padding/dilation/output-padding arguments, target logical layout, and fused-
activation ordering. The legacy named-NHWC override for a channel-first
TransposeConv2D tensor remains explicit in the 2D emitter. Only fused modules
and regular Conv2D/depthwise/Conv3D remain substantial direct-dispatch logic.

Regular Conv3D direct and runtime-helper emission is now a focused emitter as
well. It owns NCDHW raw-output annotation, NDHWC materialization aliases,
direct-module capability fallback, module-output bridging, and activation order.
The only remaining `CONV_3D` check in the dispatcher belongs to the still-
central fused-module raw-layout classification; the ordinary Conv3D branch is
gone.

Regular Conv2D and DepthwiseConv2D emission is now focused in the same emitter
module. Input layout/shape shortcuts, existing channel-first aliases, planned or
folded pre-permutations, explicit module padding, direct/runtime capability,
NCHW aliases, public-output correction, and fused activation remain ordered as
before. The direct-module dispatcher has consequently fallen from 374 lines at
the preceding checkpoint to roughly 255 lines, with the fused-module branch as
its only remaining large implementation block.

Fused-module emission is now separated too. Planned input permutation folding,
legacy NHWC Conv fallback, raw-output layout classification, channel-first alias
retention, public-output correction, channel-last materialization, and generic
module-output fallback are owned by one focused emitter. The former direct-
module implementation is now an approximately 171-line ordered dispatcher: it
contains no generated statement append and no permutation expression, only the
capability guard, attribute/spec lookup, and family emitter delegation.

That final direct-module dispatcher and its supported-op table now live beside
the family emitters. The exporter no longer defines any direct-module emission
function or module-family table; it imports the unchanged names required by the
stored generated pipeline. The dispatcher itself remains statement-free and
preserves recurrent, fused, Conv2D, TransposeConv2D, Conv3D,
TransposeConv3D, FullyConnected, and PReLU priority exactly.

The final binary compatibility adapter has also been removed. The stored native
codegen source injects the exporter's existing broadcast target-shape resolver
as one explicit callback argument at the binary call site, then invokes the
Torch-free binary emitter directly. A structural gate verifies exactly one
injected keyword and parses the transformed function source as Python. Thus the
exporter no longer defines any of the extracted native unary, binary, transpose,
shape-transform, Concat, or direct-module emitters.

Generated encoder-stage composition now has a separate Torch-free owner in
`pytorch_codegen_stages.py`. It groups staged BERT encoder statements, derives
their live inputs and outputs, and emits the optional attention/FFN submodule
source, initialization lines, and forward calls without importing Torch at
module import time. The stored native-codegen pipeline still resolves the same
imported function name, so its contract is unchanged. The former non-composite
encoder builder had no production or test callers and was removed rather than
preserved as a second implementation. Deterministic differential validation
over 300 generated stage specifications fixes exact return-value equivalence
with the previous composite builder.

Forward-source partitioning and its single-use static Reshape-chain folding now
share that stage-codegen owner. Large forward bodies retain the same liveness-
based 18/28/36-line partition policy, deterministic boundary score, generated
method signatures, and inline fallback. Adjacent foldable Reshape pairs are
kept within one stage and reduced using the shared reshape-target policy. The
exporter imports only the partitioning entry point used by the stored pipeline;
the nested folder is no longer exporter-owned. Differential checks cover 250
deterministic forward bodies and representative literal, inferred, fan-out, and
multi-step Reshape chains with exact old/new return equality.

TorchScript package export now has a focused artifact owner in
`pytorch_artifact_exporters.py`. Public arguments, artifact naming, trace-then-
script fallback, timeout handling, metadata schema, skip behavior, error
contract, and return path are unchanged; `pytorch_exporter.py` re-exports the
imported function name for compatibility. Shared example-input metadata,
dynamic-boundary detection, native-package eligibility, torch.export-specific
skip policy, and sequential child-process execution live in
`pytorch_export_support.py` for reuse by later Dynamo ONNX and ExportedProgram
separation. That support module no longer imports Torch at module load time;
only image resize and requested artifact execution import it locally. The five
moved helpers are AST-identical to their previous exporter implementations, and
the TorchScript body is identical after its lazy Torch availability guard.
The same owner now builds the generated package metadata schema, including
public ONNX boundary shape/layout restoration and NumPy-backed operator option
serialization. Tensor metadata, recursive value serialization, and the payload
builder are AST-identical to their former exporter definitions.

Dynamo ONNX package export now shares the same artifact owner. The exporter
keeps a signature-compatible wrapper whose only responsibility is to supply the
existing generated-model source-rewrite context and final-repair callback;
input construction, skip/error metadata, the sequential child invocation,
timeout handling, artifact naming, sanitization, and return behavior live in
the artifact module. ONNX external-data preservation, missing public-output
shape restoration, graph cleanup, layout-bridge folding, and final metadata
sanitization are isolated in the Torch-free
`pytorch_onnx_artifact_support.py`. After normalizing the two explicit callback
names, the artifact function is AST-identical to the previous implementation,
and all four moved ONNX support functions are AST-identical without
normalization.

The ExportedProgram child process source has its own inert payload module,
`pytorch_exported_program_child.py`. The 1,813-line embedded literal previously
obscured the approximately 140-line host orchestration; the exporter now assigns
the imported `_EXPORTED_PROGRAM_CHILD_SCRIPT` constant before invoking the same
single child runner. The 71,054-byte string is byte-for-byte identical to the
previous literal, has fixed SHA-256
`548c123d658c61780a134e34dbc02939f07d1db7e6bccc81db08fddf6cf77d5e`, and
parses as Python. Importing the payload module does not import or execute Torch.

That exposed ExportedProgram host orchestration now lives beside TorchScript and
Dynamo ONNX in `pytorch_artifact_exporters.py`. The public exporter wrapper
supplies two established hooks: temporary generated-source rewriting and final
model repair. All
metadata, native/torch.export skip policy, example-input construction,
sequential child execution, timeout behavior, cleanup ordering, and return/error
contracts remain in the focused artifact implementation. After normalizing the
two callback names, the moved host function is AST-identical to its former
exporter implementation.

ExportedProgram stack-trace removal is directly owned by the import-safe
`pytorch_exported_program_archive.py`. It rewrites only `models/model.json`
entries in a temporary archive, removes every nested `stack_trace`, preserves
other JSON fields and archive entries, leaves the original untouched when no
field is found, and atomically replaces it after a change. The moved 46-line
implementation is AST-identical; the artifact host calls it directly instead
of receiving it through the exporter wrapper.

The same archive owner now contains inverse-permute and related FX cleanup. The
2,015-line optimizer is mechanically identical to its exporter implementation;
the only added statement is a local `import torch` after archive-existence
validation, so importing artifact policy does not load Torch. The artifact host
calls it directly and preserves its best-effort exception handling. The
exporter retains an imported private alias for compatibility but owns no archive
algorithm. Full execution of this optimizer remains covered by the optional
Torch suite; the current Python 3.12 environment cannot load its Python 3.10
libtorch, so this checkpoint uses exact AST equivalence and an import-free
missing-archive contract rather than new Torch execution.

Reusable generated-Python expression parsing now has a Torch-free owner in
`pytorch_source_parser.py`. Top-level CSV splitting, balanced outer-parenthesis
removal, binary/alignment arguments, cached simple assignments, rank-four shape
literals, runtime Concat/`torch.cat` arguments, integer lists, and permutation
dimension normalization form one 12-function boundary. The exporter imports the
same private names used throughout canonicalization and source rewrites. Every
moved function, including the 131,072-entry assignment-parser cache decorator,
is AST-identical to the previous top-level implementation; nested canonicalizer
helpers remain local where their semantics are intentionally narrower.

The parser owner also handles eight complete generated-call forms: channel-last
Gather slicing, static/dynamic-batch rank-four shapes, resize, pool arguments and
assignments, tensor split assignments, softmax, and NHWC-to-NCHW bridge source
resolution. The shared ShadowFormer batch-expression pattern moved with this
group and remains imported by the exporter's narrower structural rewrites. All
eight function ASTs and the pattern value are identical to their previous
definitions, bringing the common top-level parser boundary to 20 functions.

The remaining 12 pure top-level decoders now use that boundary too: buffer
`copy_`, aligned assignment, cached functional/method permute assignment,
local-response-normalization input, compact pool/resize/softmax inputs,
constant Pad, dynamic/static binary alignment, and anchor alignment. All ASTs,
including the permute parser's 131,072-entry LRU decorator, are unchanged. The
shared owner therefore contains 32 reusable parsers while graph-aware signature
and rewrite decisions remain in the exporter.

Source line splitting, regex presence/count scanning, and balanced extraction of
prefixed nested calls are co-located with those parsers. The four utility ASTs
are unchanged, bringing the directly testable generated-source boundary to 36
functions without moving graph-aware canonicalization policy.

Seven capture-free fast-precanonicalize statement decoders extend this boundary
to 43 functions: aligned binary assignment, simple return, dynamic Pool,
local-response normalization, Softmax, Resize, and ReduceMax. Their moved ASTs
are unchanged. The orchestrator's local binary-anchor decoder was identical to
the existing shared decoder apart from its parameter name, so it now uses the
shared implementation. The fast-precanonicalize orchestrator contains no nested
function definitions; graph-aware ordering and mutation remain in the exporter.
The repair-context collector and channel-first Resize repair also call the
shared complete Resize decoder; neither keeps a nested statement parser. The
shared decoder's explicit outer-call guard subsumes the lower-level Resize
decoder guard previously relied on by the repair-local copy.
Generic dynamic binary alignment is the 44th shared decoder and covers Add,
Mul, Sub, Div, Minimum, and Maximum while preserving the existing Add-specific
adapter contract. Fast precanonicalization uses one policy repair for a
dynamic-target binary anchor both during the main scan and during the required
post-scan revisit; the revisit is a named phase rather than duplicated rewrite
logic in a second exporter loop.
Permuted Conv input assignment is the 45th shared decoder. It returns the
indentation, destination, Conv block, and unpermuted input as one tuple, so
lookahead rules do not compile or interpret separate copies of the same
generated-statement regex.
Exact positional rank-four `_align_tensor_to_target_shape` assignment is the
46th shared decoder. It preserves the generated format's four-integer grammar
and supplies destination, expression, and shape tuples to both scalar-binary
lookaround and dynamic-Pool repair.
The context collector likewise uses the shared complete Softmax and constant
Pad assignment decoders for shape and CF/NHWC evidence, instead of compiling
narrower duplicate statement regexes.
Each remaining generic source assignment is decoded once through the cached
shared parser. Consumer identifiers, simple aliases, layout-name hints, and
aligned rank-four shapes derive from that tuple, replacing three overlapping
line regexes and accepting the same positional or keyword alignment syntax as
the shared parser.
Parser migration also removes its dead matching scaffold: unused module/return/
Resize/Pool/LRN/reshape regexes and discarded terminal/Conv match results are
not retained beside the shared decoders. An AST load check guards this boundary
by requiring every remaining simple local assignment in the orchestrator to be
referenced.

Pure generated-source rewrites now live in the Torch-free
`pytorch_source_rewrites.py`. Channel-first GAP-to-Conv bridge folding, explicit
channel-last GAP output materialization, SE scale/binary bridge rewriting,
channel-last affine-to-Conv bridge folding, and rank-3/rank-4 channel-last GAP
mean rewriting share the common parser boundary and retain their established
ordering. Boundary transpose/Conv folding, redundant permute-chain collapse,
public layout-bridge alias inlining, channel-last PReLU bridge folding, and
rank-4 reshape/permute/Conv folding use the same owner. The channel-first
hard-sigmoid gate/Conv rewrite is co-located with these pure transforms as
well. Channel-last binary bridge-chain folding uses the same callback-based
boundary for local-name and constant-layout decisions. All twelve ASTs are
identical to their former exporter definitions. Channel-last GAP/Conv input
repair now uses the shared rank-4 shape policy and is the thirteenth pure
rewrite owned here; its AST is likewise unchanged. Backward liveness pruning of
generated forward lines is also owned here and accepts explicit public input and
output variable names instead of consulting graph state. ExportedProgram's
direct-Conv channel-first Add-target repair is also owned here; it recognizes
only a declared Conv block, its input-channel count, the exact Add/ReLU chain,
and a nearby direct consumer before changing the static target. Its AST is
unchanged from the exporter implementation. Direct fixtures fix each
rewrite's representative success form and unmatched-source no-op behavior.
Graph-aware GatherND boundary repair is deliberately not mixed into this pure
source-rewrite boundary. It lives in the Torch-free
`pytorch_source_graph_rewrites.py` beside its ModelIR-backed GatherND shape
query. This keeps graph/layout inspection explicit while removing both
implementations from the exporter.

Read-only ModelIR codegen queries have a separate Torch-free owner in
`pytorch_graph_policy.py`. Gather boundary pre-permutation compares static
output signatures, while effective rank-4 runtime layout follows indexed
producer/consumer edges through a fixed passthrough family to a Conv boundary.
Neither query mutates ModelIR or emits source, keeping graph inspection out of
both the monolithic exporter and the source-rewrite module.
The same owner stores the writer's shared `ModelIRGraphIndex` in the native
codegen cache. Producer queries use `ModelIRGraphIndex.producer()` directly,
and expected Conv2D channel discovery enumerates only the indexed Conv2D
family. No detached producer-to-operator table or repeated all-operator scan is
created during writer preparation.
Public target-shape, base-signature, channel-first shape, and named-tensor
resolution form one mutually recursive family in this owner. It uses the same
graph/cache queries for binary broadcast and Conv channel evidence, while
returning public tensors in their declared logical layout. Compact target and
Resize shape literal rendering plus alias-aware rank-3/4/5 channel-first shape
queries are co-located with that contract; statement emission remains outside
this read-only graph-policy boundary.
The same owner decides whether a package graph is a single public-input/public-
output chain whose non-data inputs are all constants. Transpose-convolution data
input positions retain their explicit exception; branching, multiple outputs,
or dynamic side inputs reject the sequential fast path.

Generated CONCAT and adjacent Slice layout policy has a focused Torch-free
owner in `pytorch_concat_policy.py`. It owns channel-first input expressions,
static Slice/StridedSlice alias guards, channel-axis validation and recovery,
and the decision to keep a Slice result channel-first across all indexed
Concat consumers. ModelIR shape and expression access remain explicit
callbacks, so the module neither mutates the graph nor owns statement emission.

Generated RESHAPE policy has a focused Torch-free owner in
`pytorch_reshape_policy.py`. It distinguishes plain data reshapes from layout-
sensitive plans, resolves exact static signatures and sequence lengths, and
follows indexed unary chains to an adjX BatchMatMul that requires feature-last
storage. Runtime-layout and shape-plan collaborators are explicit callbacks;
the policy owns no graph mutation or source statement emission.

Native NMS postprocess recognition has a focused Torch-free owner in
`pytorch_nms_policy.py`. It validates the unit-step RANGE that supplies Gather
indices and requires every indexed consumer to be the matching selected-index
Gather. Alias, producer/consumer, and scalar-literal evidence are explicit
inputs; unrelated RANGE or Gather chains remain ordinary generated ops.

Native Affine LayerNorm and expanded Swish recognition have a focused Torch-
free owner in `pytorch_fusion_policy.py`. The LayerNorm matcher requires the
named constant gamma/beta Mul→Add form and returns a module specification without
mutating the graph. The Swish matcher requires an exclusive
`Logistic(x) * x` diamond and rejects additional Logistic consumers. Producer,
consumer, constant, canonical-name, and attribute-allocation collaborators stay
explicit at this boundary.
The axis-0 tensor-mux Slice matcher is co-located here. It recognizes the exact
Cast/Sub condition arithmetic, then/else first-dimension products, merged size,
axis-0 value Concat, and terminal Slice before returning the three conditional
input names; it does not rewrite the graph.

Generated constant and shape-tensor evaluation has a focused Torch-free owner
in `pytorch_constant_policy.py`. It resolves direct integer constants and the
bounded SHAPE, CAST/identity, GATHER/GATHER_ND, SLICE/STRIDED_SLICE,
CONCATENATION/PACK, and MINIMUM/MAXIMUM forms used by native code generation.
The same owner tracks whether a Reshape shape expression contains runtime
dimensions, including SPLIT and UNPACK propagation, while rejecting cycles and
unsupported expressions conservatively. It reads the shared producer index but
does not mutate ModelIR or emit generated statements.
Constant Pad and scalar literal rendering, static/runtime axis expressions, and
static mirror-Pad expression planning use the same owner. Runtime helper imports
are explicit mutable inputs, and tensor-expression lookup remains a callback;
the policy therefore contains no Torch import and does not own generated-model
state.

Constant-buffer alias shape and broadcast-permutation decisions have a separate
Torch-free owner in `pytorch_constant_alias_policy.py`. It validates trailing-
axis vectors, rank-four singleton channel constants, public/produced/inlined
exclusions, and declared NCW/NCHW/NCDHW permutation preference. Constants that
already express an intentional singleton broadcast remain unpermuted.

Generated tensor-expression selection has a separate Torch-free owner in
`pytorch_expression_policy.py`. Explicit aliases, channel-first aliases,
registered buffers, inline constants, and tensor variables retain one ordered
resolution contract. The same owner resolves channel-first/permuted/transposed
constant alias expressions and collision-free cached local names; it emits only
Python source strings and runtime-helper requirements.

Generated reduction axis and direct-expression planning has a separate Torch-
free owner in `pytorch_reduction_policy.py`. Static channel-last spatial axes
map to their channel-first equivalents only with unambiguous layout/shape
evidence. Negative and duplicate axes normalize before compact direct Mean source
selection; unsupported axes remain on the runtime path.

Static channel-first expression tracing has a separate Torch-free owner in
`pytorch_channel_first_policy.py`. It follows exact CF→CL Transpose bridges,
direct convolution-family producers, and shape-preserving unary chains while
honoring precomputed aliases and cycle bounds. Unary direct-emission eligibility
uses the same trace and relaxed shape equality.

Binary runtime-shape alignment policy has a separate Torch-free owner in
`pytorch_binary_policy.py`. It recognizes all-ones passthrough operands, compares
materialized shapes with dynamic signatures and channel-first broadcast shapes,
and selects the operand whose signature best anchors the declared output. The
same owner decides whether every consumer can remain a channel-first binary op
and whether a materialized channel-last alias can be omitted. Recursive alias
elision is conservative at graph outputs and unsupported fan-out, and preserves
the existing Transpose, Conv, spatial-reduction, unary, binary-broadcast, rank,
layout, and cycle boundaries through injected capability callbacks. Constant,
scalar, passthrough, and explicit NHWC→NCHW binary input-expression selection
uses that owner as well. Direct channel-first binary capability is granted only
when every dynamic input resolves through this policy and their relaxed
broadcast exactly matches the declared output shape. Binary output target
literals are derived by broadcasting the same channel-first input shapes and
then mapping them to the declared output layout; unknown layouts retain the
established expected-channel and tensor-name guards. Operand expression policy
also owns explicit pair aliases, channel-first constant aliases, public-input
Transpose reuse, vector-to-channel reshape, and deterministic constant-axis
permutation fallback while receiving all generated names through callbacks.

Generated-code layout bridge recognition has a separate Torch-free owner in
`pytorch_layout_bridge_policy.py`. A public-input Transpose is folded only when
its bridge marker, public source, single consumer, and valid permutation all
agree. Downstream bridge matching likewise requires one consumer, distinct
known logical layouts, and an exact permutation derived from those layouts.
The same owner follows conservative consumer hints for ambiguous same-shape
channel Transposes and bounded producer traces for batchless rank-3 public
outputs; explicit metadata boundaries take precedence over structural hints.

Generated shape-expression reconstruction has a separate Torch-free owner in
`pytorch_shape_expression_policy.py`. It rebuilds constant and runtime shape
lists/scalars through Shape, Slice, StridedSlice, Gather, Concat, reshape-like,
comparison, arithmetic, and ReduceProd producers with explicit cycle guards.
Runtime shape-list helper imports are recorded only when static reconstruction
is unavailable.

Rank-4 generated-source shape policy has a Torch-free shared owner in
`pytorch_shape_policy.py`. Layout hinting and CF/NHWC shape normalization are
used by both exporter policy and source rewrites without importing the exporter
or duplicating heuristics. Special Reshape layout planning is co-located with
this policy so pre/post permutations and reshape targets are not inferred in a
separate exporter-only branch. The four function ASTs are identical to their
former exporter definitions, and direct tests cover non-rank-4 passthrough,
preferred channel selection, singleton-channel hints, ambiguous layouts,
spatial dimension preservation, and each supported special Reshape form.
Shape-preserving unary alignment elision also lives here: it requires identical
logical layouts and either equal shapes or equal element counts before allowing
the emitter to bypass runtime target-shape alignment.
TopK rank-3/4/5 layout-bridge planning also uses this owner. It searches
deterministic input permutations from static value-output shape evidence and
returns an optional inverse permutation when the index output retains the
original layout.
Pure Conv2D and Conv3D output-spatial calculations share one rank-parameterized
forward-convolution helper in the same module; transpose-Conv3D retains its
distinct formula beside it. The exporter imports the three compatibility names
and no longer owns parallel 2-D and 3-D implementations.
Asymmetric Conv2D SAME-padding planning is co-located here because it consumes
only shapes, options, and logical layouts. The exporter retains the compatibility
name through an import. The older symmetric-only SAME-padding helper had no
callers and is removed instead of establishing a second padding policy.
Feature-last sequence Reshape permutations, MatMul batch broadcasting,
BatchMatMul result shapes, and reduction result shapes are pure policies in
this module as well. Their compatibility names are imported by the generated
pipeline; ModelIR-backed adjX consumer discovery remains in the exporter.
Conv3D and transpose-Conv3D constructor inference is co-located with the
spatial formulas it validates. The policy owns channel/layout interpretation,
weight-axis search, group selection, and the established conservative fallback;
the exporter only consumes the selected constructor parameters.
The corresponding Conv2D family is also centralized here: one layout-candidate
search feeds both input pre-permutation and regular/depthwise constructor
selection. Shape, option, and logical-layout inputs remain explicit, and no
ModelIR or emitter state enters this policy boundary.

`ModelIRPassState.fingerprint()` provides deterministic cycle state for
repeating passes. It covers graph/subgraph topology, public boundaries, tensor
shape/dtype/layout/quantization/provenance, operator options/axis semantics,
and constant content while deliberately excluding non-semantic lineage
metadata. Constant ndarray buffers become read-only when fingerprinting first
occurs, and their content SHA-256 is cached by object identity; replacement
buffers receive a new digest. The manager computes fingerprints only when
`max_iterations > 1`, so current one-shot production passes perform no
fingerprint serialization or constant freezing.

`ModelIRPassState` is the shared state object for ordered post-lowering groups.
It owns one ModelIR graph index and one LayoutState, provides combined invariant
validation, and centralizes deep snapshot/restore with index and layout
resynchronization. `run_model_ir_pass_group` is the common execution boundary:
it creates that state, registers a supplied ordered spec set, executes the
manager, preserves caller-provided zero-valued statistics, and strips manager
control fields from returned semantic diagnostics. Duplicate fan-out, mixed
attention layout, and boundary-input layout runners therefore own only their
match guards, rewrite callbacks, stable pass specifications, and typed legacy
result adapters rather than duplicating manager and result-aggregation
plumbing.

The common runner can append normalized internal events to
`ConversionSession.diagnostics`. Every event carries the stable pass ID,
phase, status, iteration count, changed flag, cycle flag, and precondition-skip
flag. A conversion-wide `sequence` and stable-ID-specific `invocation` number
distinguish repeated recovery-sweep calls without changing their execution
order. Transactional invariant failures use `PassInvariantError`, a
`RuntimeError`-compatible type that retains the pass ID, phase, iteration, and
complete validator problem list after rollback. Failure events record the same
problem list before re-raising. These diagnostics are conversion-session state
only: they are not inserted into ModelIR metadata, legacy return dictionaries,
coverage/correspondence reports, or public JSON schemas.

The attention module also owns the QKV Slice canonicalization pair. One pass
replaces compatible Slice branches with Gather/Reshape views; the next replaces
three compatible branches with a single Split. Both require fully known,
compatible dimensions and exact consumers before changing ModelIR, and both
preserve the legacy lowerer entry points.

Shared pre-Transpose QKV slicing is normalized in the same attention module.
The pass proves a single shared permutation and compatible Slice begin/size
vectors, rewrites those vectors into NCHW order, preserves shared constants by
cloning, and removes only the now-redundant adapter.

The attention weighted-sum bridge is also owned there. It validates the QKV
branch producers, coefficient constants, reduction and merge topology, then
rotates only proven layout-sensitive metadata and constants into NHWC while
preserving shared values through cloned tensors.

QKV Gather/Reshape/Transpose hoisting is centralized in the attention module.
It accepts only compatible two- or three-branch projections, hoists their
shared transpose after proving axis and shape equivalence, and preserves every
branch output contract. The moved implementation removes one pre-existing
unused local without changing graph semantics.

Conv-based attention NHWC propagation is also owned by
`passes/attention_layout.py`. Its guarded variants cover basic reduction
attention, expanded HardSigmoid gates, HardSwish activation, and self-HardSwish
Mean chains. Each nested matcher proves the full region and legacy consumers
before any transpose removal or coefficient/axis rewrite.
All variants now mutate through their runner-owned `ModelIRGraphIndex` without
an end-of-iteration refresh. Index-aware input/output rewrites and differential
removals retain live topology throughout the transaction. Optional legacy NCHW
consumers are tracked by operator object across removals, rewired to one local
adapter, and preceded by an indexed Transpose insertion. Both candidate scans
enumerate only indexed Transpose roots. The attention module consequently has
no whole-graph map builder, direct operator-list insertion/deletion, or routine
index refresh; an architecture gate preserves those constraints.

CSP attention propagation is the final currently characterized large member
of `passes/attention_layout.py`. It validates both residual forms, expanded
HardSigmoid or sigmoid-self-Mul gates, singleton-spatial reshape adapters,
branch fan-out, and terminal layout before rewiring the region to NHWC.
Its root discovery and complete producer/consumer proof now reuse one
`ModelIRGraphIndex`. Every branch, gate, Conv, terminal-output, and alias edge
rewrite uses index-aware helpers; all proven bridge Transposes are removed
differentially in reverse graph order. The lowerer supplies the active
`LayoutState`, which is synchronized after pruning. The optional state keywords
preserve the legacy single-argument compatibility entry point.

## Dependency boundaries

Default TFLite conversion and ONNX/TFLite accuracy checking must import neither
TensorFlow nor tf-keras. TensorFlow is allowed only after the user explicitly
requests a TensorFlow-family artifact. PyTorch remains an optional dependency
and is loaded only for a requested PyTorch-family artifact.

No new third-party dependency may be added for this refactor. Use `uv sync` for
the core profile, `uv sync --extra torch` for PyTorch exports, and
`uv sync --extra tensorflow` for optional TensorFlow exports.

PyTorch-family example-input construction and export metadata persistence live
in `pytorch_export_support.py`; shared export exceptions live in
`pytorch_export_errors.py`. These modules are bounded internal contracts and
must not import TensorFlow. Image trace data is resized with the existing
PyTorch dependency so TorchScript, Dynamo ONNX, and ExportedProgram generation
does not require the TensorFlow optional extra.

PyTorch op capability selection lives in the Torch-free
`pytorch_capabilities.py`. It composes direct emitter module/unary/binary
declarations with the existing runtime-kernel set, owns explicit CUSTOM
rejection and unsupported-op diagnostics, and returns a defensive copy from the
public supported-kernel query. Runtime-wrapper selection and the explicit
`ONNX_SLICE` custom-code allowance use this same capability owner. The registry
expressions, native direct-codegen validation, and the typed fallback-error
classifier use the same owner. This keeps runtime-kernel support distinct from
the smaller native source-emitter set without duplicating the registry. All six
functions are AST-identical to their former exporter definitions.
Model-specific direct-module eligibility is co-located here as well. Only
channel-first Conv2D, depthwise Conv2D, and Conv3D tensors with the expected rank
and positive static channel dimensions select direct module calls.
Native package preparation combines root-CUSTOM rejection and recursive
supported-op validation in one traversal. The compatibility validators remain
available independently, while production and debug export paths use only the
combined validator and preserve the established error precedence.

Generated PyTorch identifiers have a Torch-free single owner in
`pytorch_naming.py`. Tensor variables, constant-buffer attributes, serialized
tensor storage names, keyword/digit sanitization, long-name hashing, semantic
suffix preservation, and deterministic collision resolution share the same
policy. Direct module-attribute base names are co-located with this generated
identifier policy. Affine-module attribute canonicalization and collision
resolution also use this owner, including conflicts with direct-module and
already planned LayerNorm attributes. Twelve function ASTs and four policy-
constant ASTs are identical to their former exporter definitions. The obsolete
exporter-local full module-attribute-name shim had no callers and is removed
rather than retained as a second naming path.

Native codegen value policy has a Torch-free single owner in
`pytorch_codegen_values.py`. Small constant eligibility, Python and scalar
literal rendering, TFLite-to-Torch pad ordering, Torch dtype spelling, and
Conv-block fused-activation selection no longer live in the monolithic
exporter. All seven function ASTs are identical to their former exporter
definitions, including non-finite float handling and unsupported-dtype errors.

Native Slice, StridedSlice, Gather, and Gather-plus-Reshape code generation has
a Torch-free single owner in `pytorch_indexing_codegen.py`. It owns the direct
index expression builders, suffix-flatten recognition, singleton-axis-drop
recognition, and the guarded CRD-to-DCR Gather elision before DepthToSpace. All
nine function ASTs are identical to their former exporter definitions; emitted
source strings and ModelIR layout/consumer guards therefore remain unchanged.

Native model-file generation builds its producer/consumer view through the
shared `ModelIRGraphIndex`. The exporter no longer owns a second complete graph
scan or a separate indexing policy. Codegen context owns the shared index and
exposes its maps through read-only compatibility properties, preserving
operator-index semantics while aligning package generation with the fixed
conversion-session graph contract.

Generated-package import and native state-dict reconciliation live in
`pytorch_state_dict_support.py`. Import-name sanitization, stale generated
module cleanup, dtype/shape alignment, state-key validation, and ModelIR buffer
mapping have one owner. The module has no eager Torch import: Torch is loaded
only inside tensor materialization after a PyTorch artifact has been requested.
All three function ASTs are identical to their former exporter definitions.

Generated package scaffolding lives in the Torch-free
`pytorch_package_sources.py`. The common package initializer/runtime bridge,
wrapper model source, native runtime source assembly, and idempotent Pool2D
channel-last recovery patch are shared by native, TFLite-backed, and
SavedModel-backed packages. All four function ASTs are identical to their
former exporter definitions; Torch references exist only in generated source
strings and do not import Torch while the conversion modules are loaded.

Runtime-wrapper artifact generation lives in
`pytorch_runtime_wrapper_exporter.py`. It consumes the shared capability,
metadata, naming, and package-source contracts, writes only the requested
wrapper package, and imports Torch locally only when serializing the state dict.
Its function AST is identical to the former exporter definition, while an
unsupported ModelIR is rejected before any artifact directory is created.

TFLite-backed and SavedModel-backed PyTorch package exports live with the other
artifact writers in `pytorch_artifact_exporters.py`. Their public-boundary-only
metadata builders live in `pytorch_export_support.py`. Both paths reuse the
shared package scaffolding, copy only the explicitly requested backing artifact,
and reject missing sources before creating output. All four function ASTs are
identical to their former exporter definitions; neither path imports TensorFlow.

Fallback package preference lives in the Torch-free
`pytorch_package_selection.py`. Recurrent/control and length-input exclusions,
transpose-convolution and channel-first Softmax signals, and the established
large-graph structural thresholds are evaluated independently from artifact
writing. TFLite-backed and SavedModel-backed selection share this single policy;
the saved-model entry remains a direct alias of that policy. Root operator
types, counts, and Softmax candidates are collected in one traversal before the
established guard order is evaluated. Once the final fallback has selected the
TFLite artifact unconditionally, it writes that package directly without a
redundant preference evaluation whose branches produced the same artifact.
The early native TFLite-import preference for control-flow and recurrent ops is
owned here as well. Export orchestration invokes its short-circuit scan only
after a non-empty, custom-op-free fallback TFLite path is established, so the
ordinary native path does not collect root op types for an unavailable
fallback.

Reference ONNX public-boundary inference lives in
`pytorch_onnx_artifact_support.py`. Transpose permutation decoding, short
layout-preserving boundary walks, input/output layout inference, and batchless
rank-three Squeeze/Unsqueeze detection share the ONNX artifact boundary rather
than fallback orchestration. All four function ASTs are identical to their
former exporter definitions and build their producer/consumer views once per
reference graph. Reference ModelIR/ONNX shape and layout reconciliation is
co-located here as well: it restores public names/signatures, materializes any
required public layout bridges through the shared validation pass, preserves
batchless metadata, and forces recurrent rank-three boundaries feature-last.
Its function AST is also identical to the former exporter definition.

Single-op ONNX StringNormalizer package fallback lives in the Torch- and
TensorFlow-free `pytorch_string_normalizer_exporter.py`. ONNX attribute
decoding, validation, shared package scaffolding, and public tensor metadata
serialization are isolated from the native code generator and reuse the
bounded export-support/package-source contracts. Both functions are
AST-identical to their former exporter definitions. Invalid graphs are rejected
before creating an artifact directory, and importing the module cannot load
Torch or TensorFlow.

## Runtime-check memory boundary

The direct backend releases the legacy GraphSurgeon graph before ModelIR
lowering because no direct stage consumes it. After artifact export,
unreachable ModelIR/serialization clones are collected and allocator caches
are returned to the operating system when the platform exposes a safe
heap-trim operation. Heap trimming is optional and must never affect conversion
correctness.

Isolated ONNX/TFLite checking remains strictly sequential: the ONNX worker
finishes and is reaped before the TFLite worker starts. Large prepared ONNX
graphs are materialized once in a managed temporary file and opened directly
by ONNX Runtime; do not send serialized model bytes through a multiprocessing
pipe. Evaluation permits the standard LiteRT delegate only when every runtime
input dimension is statically positive. Dynamic-input models retain the
delegate-free interpreter path to preserve its crash-isolation behavior.

## Regression workflow

Create the local corpus inventory without copying model files into Git:

```bash
uv run python -m onnx2tf.utils.flatbuffer_direct_corpus_manifest \
  --root_dir . -o flatbuffer_direct_corpus.json
```

Run inference sequentially by node tier. For example, Tier 1 is:

```bash
uv run python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir . -o flatbuffer_direct_bulk_report_tier1 \
  --min_nodes 50 --max_nodes 199 --tflite_only --root_only
```

The number `2,000` in the validation policy refers to the number of ONNX graph
nodes (operations) in a model. It is not a source-file line limit. Source and
test modules may be split when that improves ownership or reviewability, but
there is no 2,000-line structural gate.

The active improvement and regression scope is every root model in Tier 0
through Tier 4, including models whose baseline classification is a conversion
error, accuracy failure, or missing report. Recorded timeout models are kept
for provenance but excluded from subsequent runs. Tier 5 is a historical
baseline only and must not be run as part of ongoing refactoring.
Do not introduce a process pool or parallel model runner. Every successful
baseline model must remain successful; known Tier 0-4 failures are improvement
targets and must retain a normalized signature until they improve. Accuracy
reports retain their existing stricter judgement and every comparable float
output must remain below a maximum absolute error of `1e-1`.

Use the managed Tier 0-4 profile for a full active regression:

```bash
uv run python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir . -o flatbuffer_direct_active_regression \
  --regression_profile docs/baselines/flatbuffer_direct_active_tier0_4.json \
  --tflite_only
```

The profile fixes root-only discovery, the 1–1,999 node range, all 420 managed
historical model records, and inference concurrency of one. Models classified
as `timeout` in the current managed baseline remain recorded for provenance but
are automatically excluded from subsequent runs. The active run therefore
contains 394 models: 368 expected passes and 26 expected non-passes, excluding
26 recorded timeouts. The active non-passes are 20 accuracy failures and 6
missing reports. Tier 5 models cannot be added because the profile loader
rejects tiers above 4 and node ranges above 1,999.

`silero_vad.onnx` is validated with `-kat input state sr` and has a recorded
float32 maximum absolute error of `1.3750978e-6`. The similarly named dynamic
`silero_vad (1).onnx` remains a non-pass input-model defect: it references 14
lexically captured tensors that do not exist in the serialized ONNX, including
all four LSTM weight/bias captures. It cannot provide an ONNX reference result
for an accuracy-preserving promotion.

`inverse_11.onnx` also remains an active non-pass, with the normalized reason
`unsupported_exact_inverse_matrix_size_224`. Stock TFLite has matrix-diagonal
and batch-matmul operators but no matrix solve/inverse operator. The model asks
for three 224x224 inverses after bilinear upsampling. With the fixed evaluation
seed, the matrices have condition numbers from approximately `8.0e9` to
`9.2e10`, and ONNX Runtime produces magnitudes up to approximately `8.3e7`.
Even a float64 NumPy inverse rounded to float32 differs from the ONNX Runtime
reference by `1.7e7` to `1.2e8`, so substituting a different approximate
algorithm cannot satisfy the required `1e-1` absolute-error limit. The existing
custom-op fallback is retained; no inaccurate builtin approximation is emitted.

`string_normalizer_11.onnx` remains an active non-pass with the normalized
reason `unsupported_stock_tflite_string_normalizer`. It requires runtime string
case conversion, locale-aware comparison, and stopword filtering. Stock TFLite
supports string tensors and hash tables but has no string normalization,
case-folding, tokenization, or filtering builtin, so the existing
`ONNX_STRINGNORMALIZER` custom fallback is the only semantics-preserving
artifact. The reference model additionally requests the `en_US` locale, which
is not installed in the core validation environment. No locale package or
string-processing dependency is added by this project.

### Recorded Tier 0 baseline

The recursive Tier 0 corpus at commit `c4f3b7a` contains 190 models. Its
managed result is `docs/baselines/flatbuffer_direct_tier0_c4f3b7a.json`:
137 passed, 10 conversion errors, 1 timeout, 6 accuracy failures, and 36
missing accuracy reports. The median per-model duration was 2.165 seconds.
Failures are fixed by normalized signature; local run artifacts remain outside Git.

The formal root-only Tier 0 gate at commit `32c3277` contains 120 readable
models; the planned 121st entry has no readable node count. The managed result
is `docs/baselines/flatbuffer_direct_tier0_root_32c3277.json`: 96 passed, 9
conversion errors, 4 accuracy failures, 11 missing reports, and no timeouts.
Its median duration was 2.281 seconds. All 120 classifications and normalized
signatures match the same models in the recursive baseline.

The root-only Tier 1 gate at commit `a1fd301` contains 86 models. The managed
result is `docs/baselines/flatbuffer_direct_tier1_root_a1fd301.json`: 57
passed, 5 conversion errors, 13 accuracy failures, 11 missing reports, and
no timeouts. Median and maximum durations were 4.046 and 79.296 seconds.

`ssd_mobilenet_v1_12-int8.onnx` remains an active Tier 1 non-pass with the
normalized reason `invalid_onnx_missing_loop_captures_186`. Conversion succeeds
when its dynamic NHWC input is fixed to `inputs:1,300,300,3` and preserved with
`-kat inputs`, but the source model itself fails both the ONNX checker and ONNX
Runtime validation. Its two `Loop` bodies reference 186 unique tensors that
are neither defined locally nor available from the parent graph. These include
the preprocessor loop increment and numerous postprocessor slice bounds,
thresholds, and other constants. Supplying only the evident `int32(1)` loop
increment exposes the next missing capture immediately. Reconstructing all
missing values heuristically would not provide a trustworthy ONNX reference,
so the generated TFLite artifact is not promoted without the required
accuracy comparison.

`efficientnet-lite4-11-int8.onnx` remains an active Tier 1 accuracy non-pass
with the normalized reason
`onnxruntime_u8s8_saturating_pair_accumulation`. Its input Transpose and
QuantizeLinear outputs match exactly, while the first QLinearConv is the first
divergent tensor. The model combines UINT8 activations with INT8 weights. On
the validation host, ONNX Runtime's CPU kernel performs saturating signed
16-bit pair accumulation for this U8S8 path, whereas the direct TFLite
compatibility lowering computes the mathematical integer convolution before
requantization. A minimal 520-channel reproduction using activation `255` and
weight `-127` yields mathematical accumulator output `129`, but ONNX Runtime
returns `212`, consistent with clipping each two-product sum to `-32768` before
the final accumulation. Emulating that host-specific SIMD behavior with a
large expanded graph would degrade portability and pipeline efficiency, so no
such approximation is emitted. The fixed-seed final output remains below the
required `1e-1` maximum absolute-error ceiling, but it stays a non-pass because
the stricter cosine-similarity gate is not satisfied.

`version-RFB-320-int8.onnx` exhibits the same host-runtime U8S8 limitation and
uses the same normalized reason. The input QuantizeLinear result matches
exactly. For the immediately following QLinearConv, the direct TFLite result
and ONNX ReferenceEvaluator differ at only one of 307,200 elements by one
quantum, while ONNX Runtime CPU and the reference differ at 97,431 elements by
as much as 32 quanta. The current fixed-seed final maximum absolute error is
`0.14972958900034428`, improved from `0.15208351612091064` but still above the
required `1e-1` ceiling. A full ReferenceEvaluator run is not a practical
fallback: ONNX lacks built-in implementations for this model's QLinearConcat
and opset-12 DequantizeLinear, and even with local diagnostic implementations
the sequential run remained inside the pure-NumPy QLinearConv loop after more
than 90 seconds. No host-SIMD emulation or slow evaluator substitution is added
to the production pipeline.

`text_recognition_CRNN_CN_2021nov_int8.onnx` now executes through both
bidirectional LSTMs. Its second LSTM is preceded by a runtime Reshape with the
ONNX target `[0, 0, -1]`. Stale metadata previously described the result as
`[1,1,1]`, although the upstream tensor is `[25,1,2,256]` and the LSTM weights
require an input width of 512. This produced a 6,400-to-256 element Reshape
failure after the LSTM. The centralized shape reconciliation pass now resolves
that contract to `[25,1,512]` from the source tensor and the LSTM input-weight
shape, then propagates the sequence length through the bidirectional output.
The normal sequential `-cotof` path consequently completes and emits its
reports. The model remains an active accuracy non-pass with normalized reason
`lstm_float_drift_crosses_quantization_boundary_before_qlinear_matmul`. The
second fused LSTM differs by at most `4.181463737040758e-06`; one value at
`[23,266]` then crosses the following QuantizeLinear boundary by one quantum.
That changes six QLinearMatMul outputs by one quantum. ONNX Runtime and direct
TFLite QLinearMatMul each match an explicit INT32 NumPy calculation exactly
when given their respective quantized input, proving that the matmul lowering
itself is not the source. The difference propagates to final maximum absolute
error `0.14842605590820312`; mean absolute error is
`1.3509051097190739e-5`, RMSE is `0.0011565761101850747`, and cosine similarity
is `0.9999999968466379`. A rounding bias would change valid quantization
semantics, so none is added.

`fcn-resnet50-12-int8.onnx` is the third Tier 1 model with the normalized
`onnxruntime_u8s8_saturating_pair_accumulation` reason. Its input quantization
and shape operations match, and the first difference appears at the first
UINT8-activation/INT8-weight QLinearConv. With a fixed 224x224 diagnostic
shape, ONNX Runtime CPU and ONNX ReferenceEvaluator differ at 46,743 of
802,816 elements by up to 36 quanta, while the direct TFLite tensor matches
the reference at all 802,816 elements. Under the unchanged managed dynamic
input conditions, the fixed-seed final maximum absolute error has improved
from `1.5115007311105728` to `0.5471203327178955`, but it remains above the
required ceiling and therefore stays an active non-pass.

The root-only Tier 2 gate at commit `ad1d508` contains 113 models. The managed
result is `docs/baselines/flatbuffer_direct_tier2_root_ad1d508.json`: 80
passed, 4 conversion errors, 3 timeouts, 6 accuracy failures, and 20 missing
reports. Median and maximum durations were 7.124 and 120.360 seconds.

`afhq_generator.v11.quant.onnx` now executes and produces a comparison report.
The layout planner records channel-last provenance through elementwise and
decomposed DynamicQuantizeLinear regions, and a bounded quantized-layout pass
removes stale ConvInteger NCHW-to-NHWC input bridges only when that provenance
proves the input is already NHWC. The repair now builds one
`ModelIRGraphIndex`, enumerates only indexed Transpose roots, updates the Conv
input and removes the stale adapter differentially, and synchronizes the active
LayoutState after pruning. It no longer rebuilds producer/consumer maps on each
iteration or mutates the operator list directly. This eliminates the double
transpose and
improves the fixed-seed maximum error from `2.7819780111312866` to
`0.22717905044555664`, while cosine similarity improves from
`0.7283386855698387` to `0.999042264918951`. DynamicQuantizeLinear now rounds
`x / scale` to nearest-even before adding the integer zero point, matching ONNX
Runtime even when adding a large zero point would erase a just-below-half
fraction in FLOAT32. The model's input quantization consequently matches
exactly, and the first remaining mismatch moves to a later quantizer after an
InstanceNormalization difference of at most `4.76837158203125e-07`. Sparse
one-quantum decisions are still amplified through the decoder; the final
maximum error is `0.21375656127929688` with cosine similarity
`0.999052579433886`. No model-specific normalization or GAN rewrite is applied.
The normalized reason is now
`instance_normalization_drift_amplified_by_dynamic_quantization_decoder`.

The DynamicQuantizeLinear builder now lives in the dedicated 391-line
`op_builders/dynamic_quantize.py` family module instead of the legacy combined
quantized builder. The move reduces `op_builders/quantized.py` from 3,235 to
2,850 lines without changing dispatch or its ModelIR contract. A normalized
fingerprint over every operator, tensor, constant buffer, option, shape,
signature, dtype, and quantization field is identical before and after the
move: `a83d642e4aa7903f9b34495fec2c1edb5ff8779ba6735bedde382578152657f5`
for 22 operators and 27 tensors. An architecture test preserves the op-family
ownership and prevents the builder from returning to the legacy file; source
line count is not an acceptance criterion.

QLinearMatMul and QGemm now share the dedicated 238-line
`op_builders/qlinear_fc.py` family module. This second mechanical extraction
reduces `op_builders/quantized.py` from 2,850 to 2,634 lines while preserving
the public builder imports and registry dispatch. Pre-extraction normalized
ModelIR fingerprints are fixed as executable tests:
`633d083445fcf765023a948c038c0956c7a0b7646b73bdac0bb65cf4c14173c8`
for QLinearMatMul and
`bf71085f2cc3a5981b209b6d5b02cc65ea55a41251465229a5ef1636a319f70f`
for QGemm; each contains 9 operators and 16 tensors. The CRNN corpus model
also retains its exact fixed-seed metrics through the new dispatch path.

QLinearAveragePool and QLinearGlobalAveragePool are isolated in
`op_builders/qlinear_pool.py` without changing their registry names or public
builder imports. Their normalized ModelIR fingerprints are locked by focused
tests at `0bb8b9064ae208810addbcebb27846b05873d817e947a5af212f3fd8ee4a6b7c`
and `1b066e8245cb45f79df76dbc052ecf7485f07d7910fb789cff38b47c298b7f19`,
respectively. The module is part of the TensorFlow-import boundary. This split
is an ownership improvement, not a source-line requirement.

QLinearAdd and QLinearMul are isolated in
`op_builders/qlinear_binary.py`. Their shared validation, quantization
metadata, builtin/float-path selection, and ONNX-compatible requantization now
have one op-family owner while continuing to use the common quantization
primitives. Pre-extraction ModelIR fingerprints are fixed at
`d2f0714a44b2dc376827b845269a217c1df894986f3957128994a2913d611c24`
for QLinearAdd and
`b4d9d1a39202474faf52ab43fbde4938fe892a0a38c5739a87b6da2d9b882b34`
for QLinearMul. Fingerprint normalization and serialization are shared by the
op-family tests instead of being copied into each test module.

QLinearSigmoid, QLinearLeakyRelu, and QLinearSoftmax are isolated in
`op_builders/qlinear_activation.py`. The family owns activation validation,
quantization metadata, float-domain activation lowering, and the existing
ONNX-compatible Softmax requantization path. Their pre-extraction ModelIR
fingerprints are
`67e5b3d23cf2cfe03ae8ef1a006ac5fecf221f328553d3c1904ceebad9a7d902`,
`f1d0b1b74e6f0f056ca595912efcceb2827da416b059dc12992fd06ed137ab09`,
and `56aef3cabbed33cabcaba95d36058a37b6a12428102f7e83b0aef334eadbb4ec`,
respectively. Public builder imports and registry dispatch remain unchanged.

QLinearConcat is isolated in `op_builders/qlinear_concat.py`, including its
input-group validation, per-input quantization metadata, output shape/signature
reconciliation, and float-domain concat bridge. Its pre-extraction ModelIR
fingerprint is
`924e1470c62f93ba44dde277144d84bf796f40c5123839b59b44e4cd89c5b927`.
Layout propagation remains a later shared pass and is not duplicated in the
family builder.

QuantizeLinear and DequantizeLinear are isolated in
`op_builders/quantize_linear.py`. This family owns Q/DQ axis handling, tensor
dtype and quantization metadata, UINT8-to-INT8 boundary promotion, and the
existing ONNX-compatible scalar requantization selection. A two-node Q/DQ
fixture fixes the pre-extraction ModelIR fingerprint at
`333343018c7bb32db3138cefdf4007353140b044472017ae6c3b4cce762e8f91`.

QLinearConv is isolated in `op_builders/qlinear_conv.py`. Its family owns
quantized convolution validation, weight/bias preparation, padding resolution,
layout-boundary construction, and output requantization. The mixed UINT8
activation / INT8 filter fixture fixes its pre-extraction ModelIR fingerprint
at `c752a5b1e31744e65d483733f55a688f2189d6bf11436cabd498cfc6a2ef5019`.

ConvInteger is isolated in `op_builders/conv_integer.py`, with its
zero-point subtraction, filter preparation, padding, layout, and INT32 output
contract fixed at the pre-extraction fingerprint
`587f53091ce42815e43946d7b73324fe31ec7d5aeb1c3d2d749097351106dfb5`.

After the last builder extraction, the old combined
`op_builders/quantized.py` module was replaced by
`op_builders/quantized_common.py`. It contains only shared quantization,
shape/signature, padding, and requantization primitives; no `build_*` function
remains there. Every quantized op-family imports those primitives explicitly,
and the common module is included in the TensorFlow-free import boundary.

`dynamics_rife_sim.onnx` remains an active non-pass with the normalized reason
`invalid_onnx_concat_spatial_mismatch_64_128`. The source passes the structural
ONNX checker but ONNX Runtime rejects it during shape inference at
`Concat_378`: six inputs declare 128x128 spatial dimensions while
`onnx::Concat_193` declares 64x64, and the node concatenates only along the
channel axis. The direct TFLite artifact preserves the same mismatch and
LiteRT rejects allocation at the corresponding CONCATENATION. Removing
intermediate value-info and clearing graph-output shapes does not make ONNX
Runtime accept the structurally inconsistent node. The converter does not
guess which branch should be resized because no valid ONNX reference exists
to prove the required `1e-1` accuracy ceiling.

`yolov3-12-int8.onnx` now produces and executes a direct TFLite artifact under
the fixed `input_1:1,3,416,416` and `image_shape:1,2` contract. Six tf2onnx
arange Loop bodies referenced one optimizer-pruned default `delta=1` capture;
the ONNX Runtime compatibility graph restores only that canonical capture.
The same canonical Loop pattern is lowered to a dynamic TFLite RANGE using
`limit = start + trip_count * delta`, preserving its scan output without
unrolling or a custom `ONNX_LOOP`. Dynamic rank-5 elementwise inputs also no
longer use static high-rank coalescing based on all-one placeholder shapes.
The previous missing report is therefore upgraded to an active accuracy
non-pass with normalized reason `u8s8_detector_strict_metric_mismatch`. Its
fixed-seed overall maximum absolute error is `0.09563881158828735`, below the
independent `1e-1` ceiling, with mean error `7.735504565813028e-5`, RMSE
`0.0009708107564596241`, and cosine similarity `0.9999678197697215`. It remains
a non-pass because the existing stricter per-output thresholds reject a
`0.0956388` box-output outlier and the low-energy score output has cosine
similarity `0.8350063294036085`.

`arcfaceresnet100-11-int8.onnx` also uses the normalized reason
`onnxruntime_u8s8_saturating_pair_accumulation`. Its preprocessing and the
first two QLinearConv boundaries are exact. The first divergence occurs at
`stage1_unit3_conv1_quant`, whose UINT8 activation input still matches before
the INT8-weight convolution. When this node is isolated with that exact input,
the direct TFLite output matches ONNX ReferenceEvaluator at all 200,704
elements, while ONNX Runtime CPU differs at 29 elements by up to two quanta.
Those small host-kernel differences accumulate through 103 QLinearConv nodes;
the current fixed-seed final maximum absolute error is
`0.3681950643658638`, improved from `0.8229759186506271` but still above the
required ceiling. As with the other U8S8 cases, the converter does not emulate
host-specific saturating SIMD pair accumulation.

`object_detection_nanodet_2022nov_int8.onnx` is promoted from an active
accuracy failure to a pass. Its first QLinearConv output shape was absent from
ONNX value-info and initially materialized as `[1,1,1,1]`. The following
quantized LeakyRelu and DequantizeLinear retained that placeholder while the
MaxPool lowering selected its padding strategy. For the actual `[1,24,208,208]`
input, ONNX `kernel=3`, `stride=2`, `pads=[1,1,1,1]` starts its windows one
element before the input, whereas TFLite SAME needs only one total padding
element and places it at the end. This shifted every pooling window. For a
quantized pass-through chain feeding a stride-greater-than-one MaxPool,
QLinearConv now derives static output geometry eagerly and pass-through ops
replace same-rank all-one placeholders from their source. The scope is kept at
that semantic planning boundary so unrelated quantized layout chains are not
prematurely materialized. MaxPool therefore emits explicit negative-infinity
padding followed by VALID pooling. All six
outputs pass sequential comparison; maximum absolute error improved from
`2.570713758468628` to `4.76837158203125e-7`.

`object_detection_yolox_2022nov_int8.onnx` is also promoted to a pass. A late
quantized layout rewrite left one two-input QLinearConcat branch in NCHW
`[1,32,160,160]` and the other in NHWC `[1,160,160,32]`, while the Concat still
used NCHW axis 1. The existing mixed-layout repair required three inputs so it
could elect a spatial majority and therefore skipped this unambiguous
two-input case. For a channel-axis Concat with a fully known NCHW output
contract, the repair now uses that output's spatial dimensions as the
canonical contract and inserts a local NHWC-to-NCHW adapter only for the
permuted branch. LiteRT allocation and inference then complete; maximum
absolute error improved from `3.234160177409649` to
`2.384185791015625e-7`, with all metric gates passing.

`text_detection_en_ppocrv3_2023may_int8.onnx` remains active with the
normalized reason
`int8_requantization_outliers_amplified_by_transpose_conv`. Under its recorded
`x:1,3,480,640` input contract, the first INT8-activation/INT8-weight
QLinearConv differs from ONNX Runtime at only two of 614,400 elements by one
quantum. The final output differs at 16 of 307,200 pixels, concentrated in one
small spatial cluster, but the model's two final stride-2 ConvTranspose stages
amplify six of those pixels above `1e-1` and produce maximum error
`0.7411765307188034`. Mean absolute error remains
`9.280535896323273e-6` and RMSE `0.0022202128069130776`. An isolated first-Conv
comparison shows both LiteRT and ONNX Runtime one-quantum tie differences from
ONNX ReferenceEvaluator. Using LiteRT's native fixed-point quantized Conv for
all QLinearConv nodes reduced the maximum only to `0.7137255659326911` while
worsening mean error and cosine similarity, so the explicit ONNX-style
requantization remains the less degrading general implementation.

`yolov5s.onnx` remains active with the normalized reason
`float16_decode_rounding_boundary`. Its public input, intermediate graph, and
decoded `[1,25200,85]` output use FLOAT16 in ONNX, while the compatibility
float32 TFLite artifact intentionally exposes FLOAT32 inputs and outputs. The
confidence and class fields remain close, but the decoded coordinate fields
combine small convolution differences with large coordinate magnitudes. The
fixed-seed comparison has maximum absolute error `0.32965087890625`, mean
absolute error `0.0015324083874539186`, RMSE `0.012606690502504028`, and cosine
similarity `0.9999999759961475`. Adding only a FLOAT16 round-trip at the public
output reduced mean error to `6.284908615017916e-5` but quantized the remaining
outliers to an adjacent half-precision value, increasing maximum error to
`0.5`. Reproducing FLOAT16 storage after every node increased maximum error to
`1.5` because LiteRT and ONNX Runtime convolution accumulation differences
then crossed additional half-precision boundaries. Both experiments were
rejected: they cannot satisfy the independent `1e-1` ceiling and would impose
extra CAST operators on every FLOAT16 model.

`yolox_nano.onnx` remains an active Tier 2 non-pass with the normalized reason
`float_conv_accumulation_amplified_by_exp_stride`. The normal sequential
comparison satisfies the aggregate metric gate with mean absolute error
`1.3679266313265127e-5`, RMSE `0.0003010161768765817`, and cosine similarity
`0.9999999999699016`. However, ordinary FLOAT32 convolution accumulation-order
differences reach approximately `0.0021` in the regression head, then the
decoded size path amplifies them through Exp (approximately `0.0085`) and its
stride multiplication to a final maximum absolute error of
`0.1362457275390625`. This exceeds the project's independent `1e-1` ceiling,
so the model is not promoted despite the aggregate evaluator result.

`alike_l_opset11_192x320_post.onnx` remains active with the normalized reason
`topk_index_instability_from_float_ties`. Both previous-frame matching outputs
are exact. In the new-keypoint path, the TopK values differ by at most
`1.8298625946044922e-5`, but near-tied scores select or order different indices.
That turns into an index difference up to 59,908 and a decoded coordinate
difference up to `290.0000057220459`. Of the 5,000 returned keypoints, 4,955
coordinates are common as sets and many raw-order differences are adjacent
pair swaps; the associated descriptors move with those indices. Rounding the
scores before TopK would alter the source model's selection semantics, so the
converter does not add such a model-specific stabilization rule.

The related `tmp_alike_debug3.onnx` and `tmp_alike_debug4.onnx` probes now run
to completion after preserving the dynamic `[-1, 1]` output contract when a
late Squeeze/Unsqueeze fold leaves stale higher-rank input metadata. Before the
repair, LiteRT attempted to reshape 1,868 runtime elements to `[1, 1]`. The
serialized shape tensor is now `[-1, 1]`, independent of the stale input rank.
Their ordinary FLOAT32 outputs remain close: `scores_map` differs by at most
`2.4020671844482422e-5`, `descriptors` by `3.732740879058838e-6`, and `scores`
by `7.748603820800781e-7`; `keypoints` is exact. The debug-only exact-equality
and derived boolean masks can nevertheless flip from those accumulation-order
differences, producing boolean maximum error `1.0`. Both probes therefore
remain active non-passes with the normalized reason
`exact_equality_mask_instability_from_float_accumulation`. Introducing a
tolerance into ONNX `Equal` would change model semantics, so no such rewrite is
applied.

The root-only Tier 3 gate at commit `c838b42` contains 71 models. The managed
result is `docs/baselines/flatbuffer_direct_tier3_root_c838b42.json`: 22
passed, 15 conversion errors, 17 timeouts, 1 accuracy failure, and 16 missing
reports. Median and maximum durations were 17.248 and 120.662 seconds.

`new_encoder.onnx` is promoted from a missing report to a pass. Its static
rank-6 attention DIV was unresolved when op lowering ran, then became rank-6
only after final shape reconciliation; LiteRT aborted in `BroadcastDivSlow<5>`.
A bounded final ModelIR pass now coalesces fully-static broadcast-equivalent
axes to rank 4 while leaving negative/dynamic signatures unchanged. Evaluation
also derives deterministic multi-scale controls from the graph: Split sizes
`[8352,2088,522,135]` imply spatial shapes
`[[72,116],[36,58],[18,29],[9,15]]`, valid ratios are one, and float inputs
cast directly to boolean masks are zero. The fixed-seed one-sample result has
maximum error `0.0008774623274803162` and cosine similarity
`0.9999999999724198`. Its managed profile uses `eval_num_samples: 1` because
the ten-sample sequential run exceeds 180 seconds, and `accuracy_only: true`
selects the existing `--eval_with_onnx` path so the much heavier intermediate
operator report is not generated for this model. Inference remains strictly
single-process.

`encoder.onnx` is also promoted from a missing report to a pass. Unlike
`new_encoder.onnx`, its 24 GridSample image tensors retain runtime spatial
dimensions derived from `spatial_shapes`. The TensorFlow-free dynamic rank-4
GridSample lowering reads N/C/H/W with `SHAPE`, flattens the NHWC image once,
and builds runtime batch/spatial gather indices for bilinear or nearest
interpolation. Zeros and border padding and both align-corners modes are
supported without fabricating static dimensions or retaining an
`ONNX_GRIDSAMPLE` custom op. Sequential verification compares its only output
with no skip at `max_abs=1.9293278455734253e-05`, RMSE
`2.1648828175950605e-07`, and cosine similarity `0.9999999999964916`.

`fasterrcnn_resnet50_fpn.onnx` is promoted from a missing report to a pass.
`LoweringContext.ensure_tensor()` previously expanded a producer-confirmed
rank-2 tensor `[1,12543]` back to a stale shape hint `[-1,1,12543]`. Slice was
then lowered with rank-3 vectors, and a downstream TopK inserted a rank-3
transpose whose runtime input was rank 2. The context now indexes ModelIR
producers as operators are added and retains an all-static producer rank when
the higher-rank hint can only add singleton or dynamic axes. Known
non-singleton mismatches and tensors without a producer retain the previous
behavior. The resulting Slice is rank 2 (`[1,12543]` to `[1,9408]`) and no
TopK axis transpose is needed. A fixed-seed one-sample sequential run compares
all three outputs without skips, with maximum absolute error `0` and cosine
similarity `1`. Its managed profile uses `eval_num_samples: 1` and
`accuracy_only: true`; ten sequential samples exceed 180 seconds.

`fasterrcnn_test4_new.onnx` is also promoted from a missing report to a pass.
For its fixed-seed input the two ROI-level ScatterElements nodes operate on
empty `(0,256,7,7)` data, indices, and updates. Empty ScatterElements is valid
and must return an empty tensor, but LiteRT's reference SCATTER_ND kernel raises
`SIGFPE` when its shape contains zero. Dynamic ScatterElements lowering now
clamps only the internal SCATTER_ND shape vector to a minimum of one. The final
combination with the original data restores every zero-sized dimension; for
non-empty inputs the clamp is an identity. A local runtime test covers both a
non-empty value case and a zero-sized input, and the real one-sample managed
run compares all three outputs without skips, with maximum absolute error `0`
and cosine similarity `1`. Its profile uses `eval_num_samples: 1` and
`accuracy_only: true`.

`conv_tasnet.onnx` remains an active missing-report model with the normalized
reason `invalid_onnx_scatterelements_rank_mismatch_4_6`. Its terminal
`/decoder/ScatterElements` receives FLOAT data and updates shaped
`[256,2,2,10]`, but the constant indices are shaped
`[2,1,256,2,2,10]`. ONNX requires data, indices, and updates to have the same
rank, so ONNX Runtime rejects the original model at execution even though the
structural checker accepts it. The direct artifact retains the incompatible
scatter and LiteRT aborts during invocation. Squeezing or selecting either
leading indices dimension would choose unrecorded source semantics, and no
valid ONNX result exists to prove the `1e-1` accuracy ceiling, so the converter
does not fabricate a rank repair.

`dequantize_linear.onnx` remains an active non-pass with the normalized reason
`onnxruntime_qdq_fusion_and_float_conv_decode_amplification`. Its input
QuantizeLinear uses an equivalent signed internal representation and the
following DequantizeLinear is exact. The first and second float Conv boundaries
differ by only `1.1444091796875e-5` and `5.340576171875e-5`, respectively, so
the failure is not an input layout or per-channel scale-axis mismatch. Small
float Conv and requantization differences accumulate through 63 Conv stages and
are amplified by the terminal detector decode; the current managed-seed final
maximum absolute error is `81.25048828125`. ONNX Runtime's Basic graph retains
550 nodes including 63 Conv, 264 DequantizeLinear, and 110 QuantizeLinear,
whereas Extended optimization reduces it to 146 nodes by forming 63
QLinearConv, 16 QLinearConcat, and 6 QLinearAdd nodes. For the same manual input,
the direct artifact differs from the Extended graph by maximum `58.7506103515625`
and from the optimization-disabled graph by `22.74102783203125`; the two ONNX
Runtime paths differ by `59.61541748046875`. Emulating one host optimizer's
quantized fusion would therefore neither reproduce the unfused ONNX path nor
meet the `1e-1` ceiling, so the portable explicit-Q/DQ lowering is retained.

`rtdetrv4_s.onnx` now executes after wide-integer Sign lowering routes
INT64/UINT64 values through FLOAT32 SIGN and casts the `{-1,0,1}` result back to
the ONNX integer dtype. Casting through FLOAT32 is exact for sign semantics even
at integer extrema, unlike narrowing to INT32. The model moves from a missing
report to an active non-pass with normalized reason
`builtin_conv_accumulation_amplified_by_topk`. Under the evaluator's required
builtin-only LiteRT path, 278 of 300 labels differ after TopK and the score
maximum error is `0.37020038068294525`; the aggregate maximum is `79.0` from
integer labels. With the same artifact and inputs under LiteRT's default
XNNPACK delegate, all labels and zero-scaled boxes match exactly and score
maximum error is only `4.954636096954346e-6`. The evaluator remains
builtin-only for portability and dynamic-shape crash isolation; the converter
does not make XNNPACK a hidden accuracy dependency or alter TopK tie semantics.

`yolov9_n_wholebody15_Nx3HxW.onnx` and
`yolov9_t_wholebody28_Nx3HxW.onnx` are promoted from missing reports to passes
with the explicit input shape `images:1,3,640,640`. Their symbolic height and
width previously remained unresolved placeholders; no converter rewrite was
required. Sequential fixed-seed evaluation reports maximum absolute errors of
`0.0008544921875` and `0.0081634521484375`, respectively, with cosine
similarity above `0.9999999999995` for both models.

`yolo11x-obb.onnx` is likewise promoted from a missing report to a pass by
fixing its symbolic batch dimension with `input_image:1,3,1024,1024`. The
sequential fixed-seed comparison reports maximum absolute error
`6.103515625e-05`, RMSE `2.986316236369173e-06`, and cosine similarity
`0.9999999999999999`. No converter rewrite was required.

The root-only Tier 4 gate at commit `0a8ee88` contains 30 models. The managed
result is `docs/baselines/flatbuffer_direct_tier4_root_0a8ee88.json`: 12
passed, 7 conversion errors, 4 timeouts, 2 accuracy failures, and 5 missing
reports. Median and maximum durations were 28.100 and 121.826 seconds. All 12
passing models remained below the required `1e-1` maximum absolute error.

`campp_vin.onnx` is promoted to a pass after isolated accuracy evaluation adds
a sequential builtin-interpreter retry when a default delegate invocation
fails. XNNPACK cannot reshape this model's concretized dynamic-time graph, but
the same artifact runs with the builtin interpreter and matches its single
output at maximum absolute error `3.3020973205566406e-05`. A successful default
delegate remains the first choice, and an already-builtin failure is never
retried.

`bertsquad-12-int8.onnx` remains a managed accuracy failure because ONNX
Runtime's CPU U8×S8 MatMulInteger kernel does not match the exact ONNX integer
product. At the first encoder MatMulInteger, direct TFLite, an explicit INT32
NumPy product, and ONNX `ReferenceEvaluator` agree exactly, while ONNX Runtime
differs by as much as 11,772 regardless of graph optimization level. The
converter retains portable exact semantics rather than emulating this
host-specific saturation behavior. After correcting DynamicQuantizeLinear's
round-before-zero-point order, its fixed-seed final maximum error is
`2.001576066017151`; the failure classification and signature remain unchanged.

The root-only Tier 5 gate at commit `95aa61b` contains 34 models. The managed
result is `docs/baselines/flatbuffer_direct_tier5_root_95aa61b.json`: 6
passed, 4 conversion errors, 10 timeouts, 2 accuracy failures, and 12 missing
reports. Median and maximum durations were 51.621 and 123.486 seconds. All 6
passing models remained below the required `1e-1` maximum absolute error; the
largest passing error was `2.3543834686279297e-4`.

The Tier 4 result is part of the active improvement gate. The Tier 5 result is
retained only as historical evidence; do not rerun Tier 5 or use its failed
models as current improvement gates.

The rank-five NDHWC pre-Concat layout matcher is mechanically isolated in
`passes/ndhwc_concat_layout.py`. Its extracted function AST matches the
characterized central implementation at SHA-256
`0b0c625290f2ed31351ca204b0bbc5f2a463fa09ffe1bf1eccb8ff15de6aee17`.
The indexed checkpoint retains the lowerer compatibility wrapper but replaces
all five raw production calls with transactional runner
`layout.ndhwc_pre_concat`. Candidate planning now uses one shared
`ModelIRGraphIndex`, validates every adapter, fan-out, public boundary,
permutation, rank, and spatial-shape condition before mutation, removes
operators through differential index updates, and reconciles `LayoutState`
after pruning. Rank-five unary and canonical Concat tensors also clone and
remap per-axis quantization metadata from NCDHW dimension 1 to NDHWC dimension
4. The original implementation and the indexed implementation produce
identical ModelIR for all sixteen non-quantized characterization cases; the
quantized cases intentionally differ only by the corrected quantized axis.
Focused characterization, runner instrumentation, ownership, and architecture
validation pass 60 tests. A single sequential `superpoint.onnx` smoke retains
`evaluation_pass=true`, maximum absolute error
`1.6666017472743988e-06`, RMSE `1.6207873294228388e-07`, and cosine similarity
`1.0`; the NDHWC pre-Concat precondition skips all five production positions
without snapshots or fingerprints for that unrelated rank-four graph.

Two AST-identical 22-call post-lowering recovery suffixes are owned by
`_run_layout_attention_quantized_recovery_suffix`. The helper preserves the
original ordering from NHWC Mul/Add and Mean/attention recovery through gate,
TransposeConv, Q/DQ bridge, quantized PReLU/Reshape, and final Softmax
canonicalization. Its duplicate-Transpose feature flag is forwarded explicitly
from both original call sites. Registered runners inside the suffix remain
unscoped because raw ModelIR mutators separate them, so runtime pass-state and
mutation boundaries are unchanged. A third similar sequence retains its
separate `include_layernorm=True` behavior and is deliberately not merged.
The extraction removes 21 net lowerer lines and reduces direct registered-
runner call sites from 125 to 123 without changing the runtime pass sequence.

Three AST-identical 16-call layout/reshape/attention recovery prefixes are
owned by `_run_layout_reshape_attention_recovery_prefix`. The helper begins
with the established 19-call layout-recovery sequence and preserves the
following pre-Add, reshape/attention, window partition/reverse, unary-Squeeze,
and Squeeze/Reshape cleanup order. The registered Squeeze/Reshape runner is not
given a shared state scope because raw ModelIR mutators precede it. Two call
sites retain their immediately following affine fold, while the third retains
its distinct InstanceNorm recovery; those variant successors remain outside
the helper and are asserted structurally. The extraction removes 37 net
lowerer lines and reduces direct registered-runner call sites from 123 to 121
without changing runtime invocation count or order.

Two AST-identical 14-call terminal slice/Concat layout-recovery sequences are
owned by `_run_terminal_slice_concat_layout_recovery_sequence`. The helper
preserves channel-slice/Pad/Mul cleanup, post-Transpose Add, Concat affine,
split-tail, NHWC-axis sanitation, StridedSlice, pre-Add, and final layout-
Transpose cleanup order. The final registered runner remains unscoped because
raw ModelIR mutators separate it from the first registered cluster. The
immediately preceding channel-slice bridge remains outside because only the
first site passes `layout_state`; the following boundary-QDQ and slice-
passthrough calls also remain outside as distinct successors. The extraction
removes 14 net lowerer lines and reduces direct registered-runner call sites
from 121 to 120 without changing runtime invocation count or order.

Two AST-identical 11-call absolute-terminal affine/Concat/split recovery
sequences are owned by
`_run_terminal_affine_concat_split_recovery_sequence`. The helper preserves
affine folding, constant Mul/Add Transpose recovery, four Concat-affine
variants, singleton-gate Concat, three split-tail variants, and NHWC-axis
sanitation. It contains only existing raw ModelIR mutators, so it creates no
new pass-state scope. The first site remains after InstanceNorm recovery and
before pre-Add; the second remains between two StridedSlice/Pad recovery calls.
Those differing boundaries are asserted outside the helper. The extraction
removes 7 net lowerer lines without changing runtime invocation count, order,
or the registered-runner call-site count of 120.

Three AST-identical 10-call attention/gate/QDQ recovery sequences are owned by
`_run_attention_gate_qdq_recovery_sequence`, including the copy previously
embedded in the broader layout-attention quantized suffix. The helper preserves
SA/PA MirrorPad propagation, SiNet attention, gate, TransposeConv, unary-
fanout, Q/DQ activation, trailing-output Transpose, and quantized-PReLU bridge
order. Its registered trailing-output runner remains unscoped because raw
ModelIR mutators separate it from the surrounding registered clusters. The
three callers retain their distinct LayerNorm-plus-quantized-PReLU, duplicate-
PReLU, and pass-set-2 TransposeConv successors. The extraction removes 23 net
lowerer lines and reduces direct registered-runner call sites from 120 to 118
without changing runtime invocation count or order.

Two AST-identical 10-call quantized-activation/binary-bridge recovery sequences
are owned by
`_run_quantized_activation_binary_bridge_recovery_sequence`. The helper keeps
Dequantize/HardSigmoid, MaxPool, Softmax, and Logistic recovery ahead of
Softmax-Transpose canonicalization and the five safe binary-bridge variants.
It contains only raw ModelIR mutators and therefore creates no pass-state
scope. The first caller remains after quantized Reshape and before the
conditional full binary-bridge optimization; the second remains after
Dequantize/TransposeConv and before Concat recovery. The extraction removes 6
net lowerer lines without changing runtime invocation count, order, conditions,
or the registered-runner call-site count of 118.

Two AST-identical 8-call SiNet terminal layout-recovery sequences are owned by
`_run_sinet_terminal_layout_recovery_sequence`. The helper preserves shuffle-
residual, pre-Add/PReLU, fan-out, Concat/dual-Resize affine, Softmax-mask, and
terminal constant-PReLU bridge ordering and contains only raw ModelIR
mutators. Its first caller remains immediately after terminal clamp/unary/ReLU
cleanup; its second remains after very-late indexed shape convergence.
Shape reconciliation and the distinct hard-swish and repeated pre-Add
successors remain outside and are asserted as boundaries. The extraction
removes 4 net lowerer lines without changing runtime invocation count, order,
or the registered-runner call-site count of 118.

Two AST-identical 7-call pre-Add/Mean attention recovery sequences are owned by
`_run_preadd_mean_attention_recovery_sequence`. The helper preserves three
pre-Add/PReLU/fan-out rewrites, constant Mul/Add and unary fan-out recovery,
Mean/Mul/Add recovery, and the existing Mean/attention registered-pass cluster.
The cluster keeps its own bounded state scope; no state is shared across the
preceding raw ModelIR mutators. One caller remains after the broad layout-
recovery prefix and before attention/gate/QDQ recovery, while the other remains
after channel-shuffle/Gather recovery and before its distinct SA/PA and limited-
gate suffix. The extraction removes 3 net lowerer lines without changing
runtime invocation count, order, or the registered-runner call-site count of
118.

Four AST-identical 6-call SiNet pre-Add/Resize recovery sequences are owned by
`_run_sinet_preadd_resize_recovery_sequence`, including the copy nested in the
broader 8-call SiNet terminal helper. The helper preserves pre-Add/PReLU,
fan-out, Concat/dual-Resize affine, and Softmax-mask recovery and contains only
raw ModelIR mutators. The four callers retain their distinct shuffle and
terminal constant-PReLU, QDQ and singleton-Reshape, repeated pre-Add, and shape-
reconciliation/CSP-attention boundaries. Recursively expanding the two SiNet
helpers produces an AST identical to the pre-extraction lowerer. The extraction
removes 12 net lowerer lines without changing runtime invocation count, order,
or the registered-runner call-site count of 118.

Three AST-identical 5-call safe binary-bridge recovery sequences are owned by
`_run_safe_binary_bridge_recovery_sequence`, including the copy nested in the
quantized-activation recovery helper. The symmetric legacy-only, single-post,
mixed-fanout, asymmetric-fanout, and full-post variants retain their exact
order. Separately, two AST-identical 5-call QLinear/Mean/Concat sequences are
owned by `_run_qlinear_mean_concat_recovery_sequence`, preserving Mean/
HardSigmoid, QLinear SiLU and Concat/Conv, pre-QDQ Concat, and Mean/MaxPool/
Concat/Conv recovery. Both helpers contain only raw ModelIR mutators. Their
conditional binary, post-QDQ, progress-description, layout-prefix, and Concat-
recovery boundaries remain outside. Recursive helper expansion produces an AST
identical to the pre-extraction lowerer. Together the helpers remove 6 net
lowerer lines without changing runtime invocation count, order, conditions, or
the registered-runner call-site count of 118.

The two repeated dead-prune/static-reconcile/dynamic-Reshape/static-reconcile
blocks execute through `_run_indexed_shape_convergence_cleanup`. The first
invocation builds its own `ModelIRGraphIndex`; the terminal convergence owner
supplies its already-built index to the second. Dead pruning removes operators
through differential compaction, both reconciliation calls reuse the indexed
producer map, and dynamic Reshape resolution enumerates only indexed
`RESHAPE` roots. Reconciliation and Reshape resolution change tensor shape,
signature, options, and constant data but do not change graph topology, so the
updated index remains valid through the complete block. Standalone callers
retain full-scan compatibility when no matching index is supplied. A focused
characterization compares every remaining operator and tensor with the former
four-call sequence and proves exact equality while observing exactly one index
build. Architecture checks preserve both late-pipeline call boundaries.

`_run_indexed_final_shape_activation_convergence` extends the terminal block
through HARD_SWISH shape sanitation, another static reconcile/dynamic Reshape/
static reconcile cycle, activation fusion, and final reconciliation without
constructing another index. HARD_SWISH sanitation enumerates only indexed
roots. Activation fusion uses case-normalized indexed producer dispatch and
indexed consumer counts, changes the producer output through the differential
setter, and removes the explicit activation through differential compaction;
it no longer rebuilds a complete consumer map after every match. Single-
operator removal also drops empty op-type buckets, making the maintained type
index identical to a fresh rebuild. The end-to-end characterization exercises
dead pruning, dynamic metadata, HARD_SWISH repair, and Conv/RELU fusion, proves
one index build, and compares the complete final ModelIR with the former
ten-call sequence.

Rank-four channelwise broadcast-constant repair now builds one
`ModelIRGraphIndex` instead of independently scanning the graph for producers,
consumers, and binary candidates. Exact ADD/SUB/MUL/DIV/MAXIMUM/MINIMUM/POW
candidates come from indexed type dispatch, producer layout evidence uses the
indexed producer, and cloned-constant input rewrites update consumers through
the differential setter. The consumer indices are intentionally snapshotted
at pass entry. This preserves the former artifact policy for a shared constant:
every candidate that was shared at entry receives its own deterministic clone,
even after earlier rewrites reduce the live fan-out. Focused tests block both
legacy map builders, observe no index refresh beyond the supplied initial
build, compare the maintained index with a fresh rebuild, and preserve the
existing rank-three, rank-four, inverse-rotation, ambiguous-layout, and no-op
characterizations.

Stale NCHW-to-NHWC channelwise-binary Transpose repair now uses one
`ModelIRGraphIndex` for exact binary candidate order, adapter/peer producer
lookup, and the single-consumer locality guard. A successful match rewrites the
binary input through the indexed setter, updates output shape metadata, and
removes the adapter through differential compaction before restarting from
current indexed candidates. No producer or consumer compatibility map is
rebuilt between matches. Fan-out adapters remain unchanged, and both data-
input positions plus channelwise-constant and Conv-peer match families retain
their former behavior.

The two terminal fixed three-round broadcast/Transpose/shape convergence loops
are owned by `_run_indexed_binary_layout_convergence`. One index is built per
complete loop and supplied to all three operations in all three rounds.
Broadcast repair changes only tensor data and indexed operator inputs, stale
Transpose repair performs differential input/removal mutations, and shape
reconciliation changes metadata only, so the index remains valid throughout.
Primary and fallback call the same owner. End-to-end characterization proves
one index build and complete ModelIR/stat equality with the former nine-call
sequence while a separate multi-match case compares the maintained index with
a fresh rebuild.

The adjacent singleton-Reshape and stale NCHW-to-NHWC Transpose repairs in
front of NHWC Conv inputs accept one shared `ModelIRGraphIndex`. Both enumerate
only indexed `CONV_2D` candidates, obtain the adapter producer and exact
consumer list from the index, rewrite the Conv data input through the indexed
setter, and remove an accepted adapter through differential compaction. The
primary and fallback pairs run through
`_run_indexed_conv_input_adapter_repairs`, which builds one index for both
repairs. The later standalone stale-Transpose cleanup remains outside that
ownership boundary and builds its own compatibility index. Exact singleton
shape, Transpose permutation, filter input-channel, single-consumer, and graph-
output guards remain unchanged. Characterization compares the complete
resulting ModelIR with the former explicit pair, proves one index build without
legacy producer/consumer maps, exercises multiple matches, and preserves
fan-out and graph-output adapters.

Wrong-way NCHW-to-NHWC Transpose-before-Conv sanitation is owned by the Torch/
TensorFlow-free `passes/conv_input_layout.py` module. A graph containing a
Transpose constructs or reuses one `ModelIRGraphIndex`; a Transpose-free graph
retains the former unused-tensor pruning but allocates no index. Candidate
order comes from the indexed `TRANSPOSE` type bucket and every adapter consumer
comes from the current consumer index. An adapter remains protected unless all
consumers are Conv data-input users whose filters expect the already-NHWC
source channel and reject the adapter's output channel. Accepted global input
replacement updates every affected consumer through the index before
differential operator removal. The lowerer's private API is a compatibility
wrapper, and the formerly duplicated safety valve inside the Swish-QDQ NHWC-
island optimizer delegates to the same owner at its unchanged execution point
and maps removals back to the existing Swish statistic. Public adapter outputs,
non-Conv fan-out, missing users, mismatched filters, ranks, and permutations
retain the former no-op behavior. Complete legacy-reference comparison covers
two removals and multi-Conv fan-out, observes one index build and no
compatibility consumer-map call, confirms the maintained producer, consumer,
duplicate-producer, identity, and type indexes equal a fresh rebuild, and
proves identical Swish-only safety-valve output and statistics.

Recurrent orphan-step alias repair is shared by direct lowering and PyTorch
normalization through the Torch-free
`passes/recurrent_alias.py::repair_orphan_recurrent_step_tensors` owner. It
preflights tensor names using the exact `*_h_step_N`/`*_c_step_N` grammar and
therefore builds no index when no candidate exists. Otherwise it reuses a
matching supplied `ModelIRGraphIndex` or builds one index, rejects already-
produced and public-input names, finds the first valid Reshape among indexed
consumers of the corresponding `*_step_shape_N` tensor, and rewrites indexed
alias consumers in place. Non-public orphan tensor metadata is removed; public
output metadata remains. The direct wrapper converts the repaired count to its
existing stats dictionary, while the PyTorch wrapper preserves its existing
`None` return. Exact legacy comparison and cross-wrapper comparison cover
multiple aliases, first-match order, public boundaries, missing and produced
aliases, no-consumer cleanup, zero-candidate allocation, and maintained-index
equivalence.

Unbound nonconstant-input discovery and layout repair are owned by the Torch/
TensorFlow-free `passes/unbound_input_layout.py` module. Standalone reporting
keeps a lightweight producer-name scan and returns the existing issue schema.
Repair snapshots consumer objects in graph/input order and builds one
`ModelIRGraphIndex` only when an issue exists. Current operator positions are
resolved by identity before every insertion, so one differential
`insert_operator()` call updates all producer, consumer, identity, and type
indexes without rescanning the graph. Later candidates observe newly inserted
producers and skip duplicate repairs for a shared orphan.

The owner keeps three explicit semantic families: DEQUANTIZE uses the exact
`_nhwc_bridge` preference and restricted ADD fallback; RESHAPE, SHAPE, and
SPLIT choose the nearest preceding compatible runtime tensor; ONNX-style
`input.*` MUL aliases require every consumer to be MUL data input zero and a
nearest compatible ADD `_input_nhwc` source. Shape, dtype, quantization,
signature, unique permutation tensor, and insertion-order policies are
unchanged. The lowerer wrapper preserves the stats schema and reconciles shape
metadata using the returned current index. Exact former-implementation
comparison covers five sequential insertions and two-consumer MUL fan-out;
separate checks prove nearest DEQUANTIZE fallback, strict exact-source
preference, no index allocation for clean graphs, and equality with a fresh
index rebuild.

Inverse layout Transposes around linear
DEQUANTIZE-(RELU/RELU6)-QUANTIZE chains are owned by the Torch/TensorFlow-free
`passes/quantized_activation.py` module. A no-Transpose preflight avoids index
allocation; otherwise one `ModelIRGraphIndex` supplies graph-order Transpose
candidates and every single-consumer edge. The DQ input and Q output change
through indexed setters, then both wrapper Transposes are removed in one
differential compaction. The ordered restart remains so a candidate made
linear by a later removal can still be reconsidered without rebuilding maps.

Intermediate and source public-boundary guards, exact inverse permutations,
per-tensor quantization eligibility, source shape/signature propagation,
destination dtype and quantization cloning, tensor pruning, lineage events,
and the existing stats key are unchanged. Characterization compares the full
two-chain RELU/RELU6 ModelIR with the former mutation sequence, blocks legacy
consumer-map construction, checks public, fan-out, per-channel, and permutation
no-ops, observes no index for a Transpose-free graph, and compares the
maintained index with a fresh rebuild.

Expanded HardSigmoid QDQ layout-bridge cleanup is owned by the same
Torch/TensorFlow-free `passes/quantized_activation.py` module. The owner
recognizes both `MUL->ADD->RELU_0_TO_1` and
`MUL->ADD->MAXIMUM->MINIMUM` forms, traverses every linear edge through one
`ModelIRGraphIndex`, updates the DQ/Q edges differentially, and removes both
inverse Transposes in one indexed compaction. Rank-matched side constants are
permuted only after every required constant has been validated: exclusive
constants update in place, while shared constants are cloned and rewired
through the index. Thus a missing late clamp constant is now a transactional
no-op rather than leaving an earlier constant partially remapped.

The owner preserves exact inverse-permutation and per-tensor quantization
guards, source shape/signature propagation, destination dtype/quantization
cloning, pruning, lineage, and the existing stats key. Every expanded
HardSigmoid intermediate, including the `MAXIMUM` output in the clamp form,
is treated as a public boundary when listed as a graph output. Focused
characterization covers both valid forms in one graph, private and shared
constant remapping, maintained-index equivalence, public intermediates and
source, fan-out, per-channel quantization, non-inverse permutations,
transactional rejection, and the no-Transpose/no-index preflight.

The adjacent expanded `MUL->ADD->PRELU` QDQ layout bridge is also owned by
`passes/quantized_activation.py`. It uses one `ModelIRGraphIndex` for
graph-order Transpose candidates, every linear edge, side-constant ownership,
DQ/Q rewiring, and batch removal of both wrappers. MUL and ADD retain either
data/constant input order; PRELU retains its strict data-input-zero and
constant-alpha contract. Public intermediates and source outputs, exact
inverse permutations, per-tensor quantization, source shape/signature,
destination metadata, pruning, lineage, and the existing stats key remain
protected.

HardSigmoid and expanded PReLU now share only the identical constant-remap
mechanism. `_plan_constant_layout_remaps` validates all inputs and snapshots
private/shared ownership without mutation; `_apply_constant_layout_remaps`
then updates private rank-matched tensors in place or clones and index-rewires
shared tensors. Eligibility remains explicit: HardSigmoid accepts any
non-`None` constant buffer, while the established expanded-PReLU rule requires
all three buffers to be NumPy arrays. Characterization covers two valid
PReLU chains in one graph, reversed MUL/ADD inputs, private MUL/alpha remaps,
shared ADD cloning with quantization metadata, maintained-index equivalence,
public boundaries, fan-out, per-channel quantization, non-inverse
permutations, non-array/missing alpha rejection, and no-Transpose/no-index
preflight. A simple public-output ONNX graph is intentionally not used as an
exact owner fixture because the ordered trailing-output cleanup removes its
post-Transpose before this later specialized owner runs.

Quantized logistic-gated MUL layout recovery is isolated in the Torch/
TensorFlow-free `passes/quantized_gate.py` module rather than being forced
through the linear activation owner. The pass recognizes the shared
quantized input feeding separate data and logistic DQ branches, the internal
LOGISTIC-Q-DQ gate, MUL-Q, and one or more inverse-Transpose output aliases.
One `ModelIRGraphIndex` owns graph-order post candidates, producer traversal,
strict branch-consumer guards, DQ rewiring, canonical Q-output selection,
alias-consumer replacement, and batch removal of the pre/post Transposes.
No producer or consumer compatibility map is rebuilt after a match.
The nested backward walk that distinguishes the MUL data input from the
LOGISTIC-Q-DQ input is isolated in `_match_logistic_gate_branch`; incomplete
gate-looking chains retain the former behavior of falling back to a data-
branch candidate, while duplicate data or gate branches remain ambiguous and
ineligible.

The first post alias in graph order remains canonical. Its tensor receives
the permuted MUL-Q shape/signature, dtype, and cloned quantization metadata;
later alias consumers are changed through indexed `_replace_tensor_inputs`.
All original topology, fixed rank-four permutations, per-tensor quantization,
metadata permutation, pruning, lineage, and stats contracts remain intact.
Public-boundary protection is strengthened to cover every data/gate
intermediate, not only the pre input, MUL output, and post aliases. Focused
characterization compares complete former results for simultaneous single-
and multi-post chains, reverses MUL inputs, blocks legacy graph-map builders,
checks maintained-index equivalence, and covers public intermediates/source/
aliases, branch fan-out, non-Transpose post users, per-channel quantization,
wrong permutations, and the no-Transpose/no-index preflight.

The primary branch-rewrite phase of the broader Swish-QDQ NHWC-island
optimizer is owned by the Torch/TensorFlow-free
`passes/quantized_swish_layout.py` module. Its explicit result contract returns
the rewritten branch count, removed pre-Transpose count, and immutable set of
NHWC-rewritten tensor names needed by later propagation phases. One optional
or locally constructed `ModelIRGraphIndex` supplies graph-order Transpose
candidates, every producer/consumer guard, both DQ source rewrites, and
differential removal of an unused pre-Transpose. No legacy producer/consumer
map is rebuilt, and a Transpose-free graph allocates no index.

Shared-input multi-branch ordering, quantized and float MUL tails, peer Swish
recognition, fixed spatial threshold, explicit concat-closure mode, public
intermediate/post-output guards, data fan-out, shape/signature permutation, and
the historical ordered restart are unchanged. Because only the two DQ source
edges and an unused pre-Transpose can change, downstream match edges are read
directly from the maintained index without copying the full consumer map.
Extraction-time differential comparison runs the prior committed phase AST
and the new owner over shared, closure, spatial-guard, public, and fan-out
fixtures and compares the complete ModelIR and result. The comprehensive
existing three-branch fixture retains digest
`529b9889fafe9982ebb37ca63687b9329fa11a837562c154480c1856bbc05760`,
with three rewritten branches, two removed pre-Transposes, and twenty rewritten
tensors. Metadata propagation, late Concat normalization, inverse-post cleanup,
and the independently owned Conv-input safety valve remain later ordered
phases of the compatibility orchestrator.

The immediately following Swish-QDQ metadata phase is owned by
`propagate_swish_qdq_nhwc_metadata` in the same module. It treats unary
quantization, binary broadcast, Pool/Resize channel preservation, and strict
Concat-Q-inverse-Transpose closure as one fixed-point contract because all four
families mutate the same rewritten-tensor state. Candidate order is the graph-
ordered union of relevant type buckets from one `ModelIRGraphIndex`; topology
does not change, so the stable indexed consumer relation is reused for every
iteration. An empty rewritten seed returns without constructing an index.
Unary shape/signature copying, binary static/dynamic broadcast fallback,
Pool/Resize channel guards, public outputs, negative Concat axis normalization,
strict tail fan-out, Concat axis/shape mutation, and quantized-output metadata
retain their former behavior. The shape/signature copier is a single module
owner also called by the later Dequantize-input repair in late Concat
normalization; the lowerer no longer relies on a deleted nested closure.

`run_swish_qdq_nhwc_primary_phases` constructs one index when a Transpose is
present and passes it to both the branch and metadata owners. The branch phase
maintains that index through its two source-edge changes and differential root
removal; the metadata phase changes no topology. Fixed-point, public-output,
channel-mismatch, and wrong-tail fixtures match the complete prior committed
phase AST and result. The comprehensive fixture retains post-metadata digest
`bab34e6351ec24bc564b9f95b4550bbfaca867f15906f9d77b92f7e8adf1d804`,
one rewritten Concat axis, and twenty-four rewritten tensors. Late Concat
normalization and inverse-post cleanup remain outside this shared-index
boundary because they perform additional rewires and removals.

Both inverse post-Transpose sweeps around late Concat normalization delegate to
`remove_inverse_post_transposes_for_swish_qdq` in the same module. The two
historical invocation points remain distinct, but match/guard/rewrite logic has
one semantic owner. Each invocation skips index construction for an empty
rewritten set or Transpose-free graph; otherwise one `ModelIRGraphIndex`
supplies graph-order candidates, global alias-consumer replacement, and
differential removal. A public post output, non-inverse permutation, or input
outside the rewritten-tensor state remains untouched. Full fan-out is safe
because every alias consumer is updated before removal, and a newly exposed
alias chain is reconsidered by the ordered restart without rescanning all
operators.

The two former lowerer loops have identical ASTs. Differential
characterization applies the prior committed loop and the indexed owner to a
fixture containing chained aliases, multi-consumer fan-out, a public alias,
wrong permutation, and untracked input; complete ModelIR and the three-removal
count match. A supplied index remains equal to a fresh rebuild.

Late mixed-input Concat normalization is now owned by
`normalize_late_swish_qdq_concat_inputs` in the same module. A complete match
plan is validated before mutation: every input must be rank four; at least one
direct or Dequantize-wrapped NHWC-to-NCHW adapter must be bypassable; normalized
batch and spatial dimensions must agree; the Concat output must be private;
and the tail must be exactly one Quantize whose users are all inverse
Transposes. Accepted Concat and Dequantize edges are updated through one
`ModelIRGraphIndex`, axis and tensor metadata are committed together, and only
input Transposes made unused by those rewires are removed. Processing restarts
after each transaction so a compaction cannot leave a stale candidate index.
Public source boundaries, mixed fan-out, missing tensors, mismatched shapes,
and invalid tail permutations remain no-ops.

`run_swish_qdq_late_concat_and_post_cleanup` shares that maintained index with
the immediately following inverse-post cleanup. The first historical post
cleanup remains before late normalization; the second is represented inside
the runner, preserving ordered behavior and cumulative statistics while
avoiding another full index build. An extraction-time check compiles the exact
prior committed late-loop AST and confirms complete ModelIR, rewritten-state,
axis-count, and two-input-adapter-removal equality on the mixed direct/DQ
fixture. The Swish compatibility orchestrator is now 69 lines with no raw
top-level mutation loop.

Concat-input exact-grid Q/DQ bypass is owned by
`passes/quantization_cleanup.py`. The owner recognizes only
`DEQUANTIZE(source_q)->float->QUANTIZE->q->DEQUANTIZE->concat_input` and
rewires the matching Concat slot to the first float tensor. The source and
destination quantized tensors must have exactly equal dtype, scale, zero point,
and quantized dimension; the two float shapes must match; `q` must be consumed
only by its Dequantize; and neither `q` nor the second Dequantize output may be
public. Arithmetic between the first Dequantize and Quantize therefore remains
ineligible, preserving observable rounding and clipping.

One optional or locally constructed `ModelIRGraphIndex` supplies graph-order
Concat candidates, all producer traversal, the exclusive quantized-consumer
guard, and differential Concat input replacement. Each accepted edge change
restarts the ordered scan against the maintained index; multiple matches no
longer rebuild complete producer and consumer maps after every rewrite. A graph
without Concat still performs the historical unused-tensor pruning but does not
allocate an index. The lowerer retains a thin compatibility wrapper and its two
ordered call sites remain unchanged. Extraction-time characterization compiles
the complete prior committed function AST and confirms exact ModelIR and stats
equality for both one and two simultaneous matches.

Terminal Transpose/Dequantize sanitation is owned by the same quantization-
cleanup module and keeps one index across both historical subphases. The first
subphase recognizes a private, exclusively consumed per-tensor-quantized
`Transpose->Dequantize->graph output` boundary, moves Dequantize before the
Transpose through indexed input/output replacement, creates the same uniquely
named intermediate tensor, updates output shape/signature, and reorders the two
operators through indexed remove/insert. The second subphase recognizes a
terminal `Dequantize->Transpose->graph output`, globally renames the private
pre-Transpose tensor to the public output through the index, and removes the
Transpose differentially. Their separate sanitation and removal counters and
ordered restart are unchanged.

The owner protects terminal-output consumers, public intermediate and
quantized inputs, shared Transpose outputs, missing tensors, invalid
permutations, and per-channel quantization. It performs historical pruning but
allocates no index unless both Dequantize and Transpose are present. The
lowerer is a thin compatibility wrapper at both established call sites.
Extraction-time characterization compiles the complete former function AST
and confirms exact ModelIR and both stats for each subphase with one and two
simultaneous matches.

The adjacent Transpose-Dequantize-keepdims-Mean-Quantize bridge is also owned
by `passes/quantization_cleanup.py`. One index supplies graph-order pre-
Transpose candidates and every exclusive linear-edge guard. A complete rewrite
plan normalizes negative reduction axes, maps them through the permutation,
computes Dequantize/Mean/bridge shapes and signatures, validates the
permutation, and reserves unique bridge/perm tensor names before mutation.
Commit then bypasses the pre-Transpose, rewrites axes and metadata, updates the
Quantize edge, inserts the preserving Transpose immediately before Quantize,
and removes the former pre-Transpose through the same maintained index.

Valid one- and two-match fixtures retain complete former ModelIR and stats.
Public/fan-out intermediates, shared axes, `keepDims=False`, invalid axes,
missing tensors, and missing required operator families remain no-ops. The
transactional preflight deliberately fixes one former rough edge: an invalid
permutation used to mutate Dequantize/Mean metadata and axes before discovering
that the output bridge could not be formed; it now leaves the complete ModelIR
unchanged. A graph without all four required operator types still performs
historical pruning without allocating an index.

Pseudo-op LeakyReLU fusion is owned by `passes/graph_cleanup.py`. The accepted
grammar remains deliberately exact: `RELU(x)` must be the first SUB input;
`MUL(alpha, RELU(NEG(x)))` must be the second; alpha may occupy either MUL
slot but must be a singleton constant; and all four branch intermediates must
be private and exclusively consumed by the next expected operator. When both
boundary tensors exist they must be floating point. Reversed SUB inputs,
source mismatch, integer boundaries, public intermediates, and any fan-out are
no-ops.

One optional or local `ModelIRGraphIndex` provides graph-order SUB candidates,
producer traversal, all consumer guards, in-place type/input replacement of
the retained SUB, and one batch compaction of NEG, both RELUs, and MUL. The
retained operator is normalized to the same fresh `LEAKY_RELU` fields as the
former object replacement, preserving graph position and output identity.
Tensor and optional LayoutState pruning occur after the fixed point; a graph
without the complete required operator family allocates no index. Exact former
AST comparison confirms complete ModelIR and stats equality for one and two
matches, including both alpha input orders.

The former YOLO-named MUL-square fold now has a model-neutral semantic owner,
`_optimize_mul_square_anchor_constant_chains`, in
`passes/constant_fold.py`. It matches only
`MUL(x,c)->MUL(a,a)->MUL(anchor)->MUL(scale)` with an exact self-square,
singleton finite pre-scale, floating anchor and scale buffers, finite fused
values, and exclusive consumption of all three intermediates. Constant inputs
retain either-side acceptance at each commutative MUL, while unrelated MUL
chains remain ineligible. The legacy lowerer function and stats key remain as
compatibility adapters only.

One optional or local graph index supplies graph-order MUL candidates,
producer traversal, duplicate-aware square consumption, differential input and
output rewrites, and batch removal of the first and final MUL. The fused buffer
retains anchor dtype, normalized shape/signature, cloned quantization metadata,
and optional LayoutState registration; the retained square and anchor MULs keep
their graph positions and the anchor assumes the public final output identity.
All topology and finite-value checks complete before mutation. Public `a`,
square, or anchor intermediates are now protected transactionally—a deliberate
fix for the former rule, which could remove producers of graph outputs. Valid
one- and two-match fixtures retain complete former ModelIR and statistics.

Leading-singleton Gather-to-Reshape canonicalization is owned by
`passes/gather_reshape_cleanup.py`. One optional or local
`ModelIRGraphIndex` enumerates graph-order Gather candidates, proves the sole
Reshape consumer, records the Reshape data-edge replacement through the common
lineage-aware setter, and removes the Gather differentially. Multiple matches
therefore use one index instead of rebuilding the complete consumer map after
each removal. The graph-order scan restarts only after a successful removal,
preserving the former fixed point when an inner Gather removal exposes an
outer Gather directly before Reshape. Graphs missing either required operator
family retain historical unused-tensor pruning without allocating an index,
and optional LayoutState pruning keeps the session contract current.

The value-preservation contract requires normalized axis zero, zero batch
dimensions, exactly one signed-integer zero in the constant index buffer, a
statically fixed leading-one input signature, exact rank-reduced tail
shape/signature, matching input/Gather-output dtype and quantization, a private
uniquely produced Gather output, and a topologically later Reshape that consumes
it only at data input zero. The singleton index may retain physical shape
`[1]`, matching direct TFLite scalar-index legalization. Every guard completes
before mutation. This deliberately prevents two unsafe former rewrites:
multiple zero indices, which repeat the selected slice, and a dynamic leading
signature, which cannot prove that bypassing Gather preserves element count.
Exact former-function AST comparison confirms complete ModelIR, lineage, and
statistics equality for valid independent one- and two-match fixtures and the
nested fixed point.

Marker-gated terminal Softmax/Transpose cleanup is owned by
`passes/terminal_softmax_layout.py`. The preceding canonicalizer imports the
same `_SOFTMAX_NHWC_PROPAGATED_MARKER`, so propagation and consumption no
longer duplicate a private string contract. One optional or local
`ModelIRGraphIndex` follows deterministic public-output order, rejects any
internally consumed terminal output, proves unique Transpose and Softmax
producers, and checks the private Softmax intermediate's exact consumer. The
lineage-aware indexed output setter moves public producer identity to Softmax;
the terminal Transpose is then removed differentially. Multiple public outputs
reuse the same index, and graphs missing either required family retain
historical tensor and optional LayoutState pruning without index construction.

All marker, permutation, arity, producer, consumer, public-input/output, and
operator-order guards complete before mutation. A terminal output cannot also
be a graph input. Rank-four source shape/signature and a destination tensor are
also required, and source
quantization is cloned before commit. The existing public tensor object and
its provenance remain stable while dtype, quantization, shape, and signature
take the retained Softmax output metadata; every other Softmax option,
including axis and beta, remains unchanged when the marker is removed. This
deliberately fixes two former invalid-IR paths: rewriting when the private
Softmax output is also public, and deleting the adapter when its source tensor
metadata is absent. Exact former-function AST comparison confirms complete
ModelIR, lineage, and statistics equality for valid one- and two-output
fixtures.

Pre-ArgMax channel-layout cleanup is owned by
`passes/terminal_argmax_layout.py`. One optional or local
`ModelIRGraphIndex` enumerates graph-order Transposes, proves each private
adapter's sole topologically later ArgMax consumer, tracks axis-constant
ownership, applies lineage-aware data/axis rewiring, and removes the adapter
differentially. Independent matches and changes from shared to private axis
ownership therefore reuse the same current index. Graphs missing either
required operator family retain historical tensor and optional LayoutState
pruning without index construction; successful calls synchronize LayoutState
after registering any cloned constant and pruning dead tensors.

The accepted adapter is exactly rank-four `[0,3,1,2]`. A singleton signed
INT32/INT64 axis must normalize to NCHW channel axis one and maps to NHWC axis
three. Source and adapter shape/signature must be the exact permutation, the
ArgMax output metadata must equal the rank-reduced NHWC prefix, and source and
adapter dtypes must agree. The adapter output cannot cross either public
boundary or fan out. Private axis constants retain in-place update semantics;
shared constants and constants at either public boundary receive a uniquely
named clone with preserved NumPy dtype and cloned quantization. All metadata,
clone, topology, and public-boundary decisions finish before mutation. This
deliberately prevents the former rule from changing a public axis output from
one to three or removing the producer of a public-input adapter tensor. Exact
former-function AST comparison confirms complete ModelIR, lineage, and
statistics equality for valid private, shared, and negative-axis fixtures.

Exact-grid quantized MaxPool cleanup is owned by
`passes/quantized_pool.py`. One optional or local `ModelIRGraphIndex`
enumerates graph-order Dequantize candidates, proves the exclusive and
topologically ordered MaxPool/Quantize consumers, applies lineage-aware Pool
input/output rewrites, and removes both wrappers differentially. Independent
INT8 and UINT8 chains therefore share one current index. Graphs missing any
required operator family retain historical unused-tensor and optional
LayoutState pruning without allocating an index; both production call sites
supply the Session-owned LayoutState.

The retained builtin requires exactly equal input/output quantization grids:
the dtype is the same INT8 or UINT8, scale is positive and finite, zero point
is identical and within the dtype range, and scale equality is exact rather
than tolerant. All four tensor records must exist. Both float bridges have the
same floating dtype, and quantized/float shape plus shape-signature metadata
must agree exactly at each rank-four boundary. The bridge tensors cannot cross
either public boundary, fan out, or have duplicate producers; the quantized
output cannot also be a graph input. Quantization cloning and every topology
and metadata guard finish before mutation. Pool options, version, ONNX
provenance, public output identity, and valid former statistics remain
unchanged. This deliberately prevents three former invalid rewrites: folding
near-equal but distinct grids, folding without float bridge metadata, and
removing a producer whose bridge is exposed as a public input. Exact former-
function differential execution confirms complete ModelIR and statistics
equality for valid one- and two-chain fixtures.

Canonical quantized Logistic cleanup is owned by
`passes/quantized_logistic.py`. One optional or local `ModelIRGraphIndex`
enumerates graph-order Dequantize candidates, proves the exclusive and
topologically ordered Logistic/Quantize consumers, applies lineage-aware
Logistic input/output rewrites, and removes both wrappers differentially.
Independent INT8 and UINT8 chains therefore share one current index. Graphs
missing any required operator family retain historical unused-tensor and
optional LayoutState pruning without allocating an index; both production
call sites supply the Session-owned LayoutState.

The quantized input and output use the same INT8 or UINT8 dtype. Input scale
must be positive and finite, with its zero point in the dtype range. Output
quantization is exactly the builtin's canonical scale `1/256` and zero point
`-128` for INT8 or `0` for UINT8; the former tolerance is intentionally not
used. All four tensor records must exist, the two bridge dtypes must be the
same floating type, and elementwise shape/signature metadata must agree across
the complete chain without imposing a rank-four restriction. Both bridges are
private, exclusively consumed, uniquely produced, and topologically ordered;
the quantized output cannot also be a graph input. All guards finish before
mutation. Logistic options and provenance remain on the retained object,
version becomes two for INT8 and one for UINT8, and the public output object
and canonical quantization remain stable. This deliberately prevents former
rewrites with a near-canonical output scale, absent or invalid input grid,
missing float metadata, or public-input bridge. Exact former-function
differential execution confirms complete ModelIR and statistics equality for
valid one- and two-chain fixtures.

Canonical quantized Softmax cleanup is owned by
`passes/quantized_softmax.py`. One optional or local `ModelIRGraphIndex`
enumerates graph-order Dequantize candidates, proves the exclusive and
topologically ordered Softmax/Quantize consumers, applies lineage-aware
Softmax input/output rewrites, and removes both wrappers differentially.
Independent INT8 and UINT8 chains therefore share one current index. Graphs
missing any required operator family retain historical unused-tensor and
optional LayoutState pruning without allocating an index; both production
call sites supply the Session-owned LayoutState.

Input and canonical output grid requirements match quantized Logistic. The
existing absolute beta tolerance of `1e-6` remains, with finite and parseable
beta now required. Tensor rank must be positive, and an explicit negative or
positive axis must normalize to the final dimension; when axis is omitted the
final dimension is the default. This makes the match consistent with the
serialized TFLite Softmax options, which retain beta but have no independent
axis field. All four tensor records must exist, float dtypes must agree, and
elementwise shape/signature metadata must match throughout. Both bridges are
private, exclusively consumed, uniquely produced, and topologically ordered;
the quantized output cannot also be a graph input. All guards finish before
mutation. Softmax options and provenance remain on the retained object,
version becomes two for INT8 and one for UINT8, and public output identity and
canonical quantization remain stable. This deliberately prevents former folds
with a near-canonical output grid, absent/invalid input grid, missing float
metadata, public-input bridge, malformed options, scalar rank, or non-last
axis. Exact former-function differential execution confirms complete ModelIR
and statistics equality for valid one- and two-chain fixtures, while a real
QLinearSoftmax wrap inference remains bit-exact to ONNX.

Expanded HardSigmoid QDQ cleanup is owned by
`passes/quantized_hardsigmoid.py`. One optional or local
`ModelIRGraphIndex` traverses the exact `DEQUANTIZE -> MUL -> ADD -> MAXIMUM ->
MINIMUM -> QUANTIZE` chain, supports either scalar-input position, proves every
exclusive producer/consumer and graph-order edge, applies lineage-aware data,
constant, and output rewrites, and removes both wrappers differentially.
Independent INT8/UINT8 matches share one current index. Graphs missing any
required operator family retain historical unused-tensor and optional
LayoutState pruning without allocating an index; both production call sites
supply the Session-owned LayoutState.

Input and output tensors require the same exact finite positive per-tensor
INT8/UINT8 grid and an in-range zero point. All seven data tensors must exist;
the five intermediate tensors have the same floating dtype, and every data
shape/signature is identical. The five float bridges are private, uniquely
produced, exclusively consumed, and topologically ordered; the quantized
output cannot also be a graph input. Each alpha, beta, low, and high side
tensor is a finite producer-free singleton whose quantized reconstruction is
within the preserved quarter-scale or `1e-3` tolerance.

Constant retargeting is now a four-item immutable pre-mutation plan. It owns
the quantized value, cloned grid, source metadata, private/shared/public
decision, and a deterministically reserved clone name. Private exclusive
constants retain in-place conversion; shared or public constants receive `_q`
clones. This fixes the former mutation of publicly observable scalar values.
Only after all four plans and all intermediate grid clones succeed are edges,
dtypes, grids, and output identity committed and wrappers removed. Injecting a
failure into the second grid clone proves complete transactional rejection;
the former helper had already changed the first constant's data and dtype
before raising. Exact former-function differential execution confirms complete
ModelIR/statistics equality for valid private one/two-match fixtures and a
shared-four-constant fixture, while near-equal grids, absent float metadata,
and public bridges are now no-ops.

Quantized TransposeConv QDQ cleanup is owned by
`passes/quantized_transpose_conv.py`. One optional or local
`ModelIRGraphIndex` enumerates graph-order Dequantize candidates, proves the
exclusive and topologically ordered TransposeConv/Quantize consumers, applies
lineage-aware input/output rewrites, and removes both wrappers differentially.
Independent chains therefore share one current index, and all three production
call sites supply the Session-owned LayoutState. Graphs missing any required
operator family retain historical unused-tensor and optional LayoutState
pruning without allocating an index. A graph containing the complete family
but no valid candidate is a complete no-op.

The retained operator keeps the exact TFLite input roles `[output_shape,
filter, data]`, its options, axis semantics, ONNX provenance, and at least
opcode version three. Input and output activations independently require valid
finite positive per-tensor INT8 grids with in-range zero points; they need not
share a grid. Quantized/float shape and signature metadata must agree at each
boundary, and both float bridges must share a floating dtype. Every bridge is
private, uniquely produced, exclusively consumed, and ordered; the quantized
output cannot also be a graph input.

The producer-free rank-four filter is planned before mutation. A valid INT8
filter remains unchanged. A finite FLOAT16/FLOAT32/FLOAT64 filter is quantized
in place only when it is private and exclusively consumed by this operator;
shared or public filters receive a deterministic `_q` clone. Filter metadata
must exactly describe its NumPy buffer. Output-grid cloning, filter data,
ownership, name reservation, and all topology/metadata guards finish before
the first mutation. This prevents the former partial conversion of a private
float filter when output-grid cloning raises, preserves public float filters,
and rejects missing bridge metadata, public-input bridges, invalid activation
grids, produced constants, and malformed buffers transactionally. Exact
former-function differential execution confirms complete ModelIR/statistics
equality for valid one/two-chain and shared-filter fixtures.

Decomposed InstanceNormalization layout repair is owned by
`passes/instance_normalization_layout.py`. It performs a marker-only preflight,
then uses one optional or local `ModelIRGraphIndex` for every rank-three,
rank-four, and rank-five candidate. The index proves the exact ordered
`MEAN -> SUB -> square -> MEAN -> epsilon ADD -> SQRT -> reciprocal DIV ->
normalize MUL -> scale MUL -> optional TRANSPOSE -> bias ADD` grammar, unique
producers, exclusive internal consumers, and private intermediate boundaries.
The final production call supplies the Session LayoutState. Graphs without a
marked first Mean allocate no index.

Logical layout selects channel axis one for NCW/NCHW/NCDHW and the final axis
for NWC/NHWC/NDHWC. Both Mean axes, reduced/full intermediate shapes, and
scale/bias broadcast buffers are immutable plans constructed before mutation.
An axes constant that must change may be shared only by the two Mean operators;
changing axes, scale, or bias constants requires producer-free, non-public
ownership and the exact expected consumers. Epsilon must be a finite singleton
and the reciprocal numerator must be the producer-free scalar one. Scale and
bias buffers must contain exactly the positive static channel count, and an
optional post-Transpose must contain a complete rank permutation before its
bias axis is derived.

Every tensor record, integer axis buffer, shape/signature, constant reshape,
operator type/arity/order, and ownership decision is validated before any
state is changed. This deliberately rejects the former acceptance of reversed
SUB, non-ADD epsilon nodes, public intermediate shape mutation, shared scale
mutation, floating axis buffers, incomplete channel constants, and malformed
post permutations. It also prevents a late malformed bias shape from raising
after axes, intermediate metadata, and scale data were already changed. Exact
former-function differential execution confirms complete ModelIR/statistics
equality for valid multi-layout and rank-five fixtures.

NCHW Concat/global-pool/Conv axis repair is owned by
`passes/concat_global_pool_layout.py`. A four-family preflight avoids index
construction for unrelated graphs. One optional or local `ModelIRGraphIndex`
then walks Conv candidates backward through the exact ordered and exclusive
`CONCATENATION -> global MEAN -> RESHAPE -> CONV_2D` chain. The sole production
call supplies the Session LayoutState.

The Mean must keep dimensions and reduce exactly rank-four spatial axes two
and three, accepting their equivalent negative representation. Every Concat
input has a fully positive NCHW rank-four shape with common batch and spatial
dimensions. The sum of axis-one channels must equal the constant OHWI Conv
filter input channel. Concat, Mean, and Reshape intermediates are uniquely
produced, exclusively consumed by the next operator, topologically ordered,
and private. The producer-free Reshape shape tensor is a private exclusive
integer constant with four elements.

Concat axis/options, Concat/Mean/Reshape shape and signature metadata, Reshape
options, and the shape buffer are immutable plans completed before mutation.
This prevents the former rule from changing non-global reductions, public or
fan-out intermediates, duplicate-producer chains, runtime/malformed filters,
shared/public/produced or non-integer shape tensors, and incomplete shape
buffers. It also prevents a late shape-buffer read exception from leaving the
Concat axis and three tensor records partially changed. Exact former-function
differential execution confirms complete ModelIR/statistics equality for valid
one/two-chain, negative-axis, and INT64-shape fixtures.

NCHW Concat/Transpose/(Transpose)Conv axis repair is owned by
`passes/concat_transpose_conv_layout.py`. A family preflight avoids index
construction for unrelated graphs. One optional or local `ModelIRGraphIndex`
walks Conv and TransposeConv candidates backward through optional post-
Transpose PAD/CAST/SUB and optional pre-Transpose
RELU/RELU6/QUANTIZE/DEQUANTIZE/CAST chains to one Concat. Every edge is
uniquely produced, exclusively consumed by its next ordered operator, and
private. The sole production call supplies the Session LayoutState.

The boundary Transpose is exactly `[0,2,3,1]` with a producer-free permutation
constant. Every Concat input has a fully positive NCHW rank-four shape with
common batch/spatial dimensions; its axis-one channel sum equals the constant
OHWI filter input channel and the filter buffer exactly matches its metadata.
The existing already-correct Transpose-output guard remains. Direct Conv
without a post-prefix retains its output-shape refresh, while prefixed Conv and
TransposeConv retain their former output metadata.

Concat options plus Concat, pre-passthrough, Transpose, and eligible direct-
Conv shape/signature records are one immutable plan. This prevents mutation of
public or fan-out adapters, duplicate producers, produced permutation/filter
constants, malformed/runtime filters, invalid input shapes, and nonexclusive
pre/post chains. Exact former-function differential execution confirms
complete ModelIR/statistics equality for direct Conv, pre/post-prefix Conv, and
TransposeConv fixtures.

Mixed singleton NCHW-input repair for an NHWC Concat is owned by
`passes/mixed_singleton_concat_layout.py`. A Concat-family preflight avoids
index construction for unrelated graphs. One optional or local
`ModelIRGraphIndex` enumerates every graph-order `CONCATENATION` exactly once,
plans all local adapters before mutation, and maintains producer/consumer and
operator-type state while inserting each Reshape. The sole production call
supplies the Session LayoutState.

A candidate must have at least two same-dtype inputs, one output, and exact
axis three. Every concrete rank-four dimension is positive; the output's last
dimension equals the input count, and every input is either the matching NHWC
singleton shape or its NCHW `[N,1,H,W]` projection. Shape signatures must
describe the same layout contract. One dynamic batch or spatial dimension is
preserved as the sole `-1` in the Reshape buffer, options, and adapter
signature. A contract needing two dynamic Reshape dimensions is left
unchanged because one TFLite Reshape cannot infer both safely.

Each runtime source is a graph input or has one earlier producer; producer-
free constants remain accepted. Duplicate, self, later, or unresolved
producers are rejected. Repeated uses of one NCHW source share one local
adapter instead of creating duplicate producers. Adapter and shape names are
reserved across all tensor, operator, and public-boundary names before any
operator is inserted. Source dtype and cloned quantization are retained, and
the integer shape tensor plus Reshape operator are immutable plan objects.

Indexed insertion and lineage-aware Concat rewiring occur only after every
fallible shape, name, and quantization decision for that candidate succeeds.
This prevents the former partial insertion when a later quantization clone
raises, and rejects output-channel mismatches, mixed dtypes, inconsistent
dynamic metadata, and ambiguous topology as complete no-ops. Exact
former-function differential execution confirms ModelIR/statistics equality
for valid static multi-candidate, multi-adapter, quantized-metadata, and name-
collision fixtures.

Swin-style window-partition canonicalization is owned by
`passes/window_partition_layout.py`. A two-Reshape/one-Transpose preflight
preserves historical unused-tensor pruning without constructing an index for
unrelated graphs. Otherwise one optional or local `ModelIRGraphIndex`
enumerates graph-order first-Reshape candidates and proves the exact ordered,
exclusive `RESHAPE -> TRANSPOSE -> RESHAPE` chain before replacing it with
`SPACE_TO_DEPTH -> RESHAPE`. Both production call sites supply the Session
LayoutState.

The input is rank-four NHWC and the first Reshape is exactly
`[N,OH,BS,OW,BS,C]`, where the square block is greater than one and reconstructs
the input height and width. The Transpose permutation is exactly
`[0,1,3,2,4,5]`, its output is `[N,OH,OW,BS,BS,C]`, and the retained final
Reshape is `[N*OH*OW,BS*BS,C]`. All data tensors have identical dtypes and
either no quantization or the same valid per-tensor grid. Shape and permutation
vectors are producer-free INT32/INT64 constants with exact vector metadata;
runtime graph inputs are not treated as constants.

Every produced boundary has one producer, internal outputs are private and
exclusively consumed by the next ordered operator, and the data source is a
graph input, producer-free constant, or uniquely produced earlier value. The
former Transpose object becomes SPACE_TO_DEPTH in place, preserving version,
axis semantics, and ONNX provenance while receiving the exact block option and
lineage-aware data edge. One differential index maintains its type, inputs,
and removal of the first Reshape.

Static valid graphs retain exact former ModelIR and statistics. Consistent
dynamic batch, spatial, or channel metadata is propagated through
SPACE_TO_DEPTH. When the retained final Reshape needs one inferred dimension,
its private shape vector and `newShape`/`onnxRawNewShape` options are planned as
one `-1` transaction and protected from later static cleanup. Two inferred
output dimensions cannot be represented by this Reshape and remain unchanged.
This also makes duplicate/later producers, public-input conflicts, floating or
produced constants, missing output metadata, mixed dtypes, invalid grids, and
shape/signature mismatches complete no-ops.

The paired Swin-style window-reverse canonicalization is owned by the same
`passes/window_partition_layout.py` module. Its two-Reshape/one-Transpose
preflight also preserves historical pruning without allocating an index for
unrelated graphs. Otherwise one optional or local `ModelIRGraphIndex` captures
the graph-order Reshape objects once and resolves their current positions as
earlier final Reshapes are removed. This avoids repeated consumer-map scans
while preserving the former sequential shared-constant behavior.

The reverse input is rank three `[N*OH*OW,BS*BS,C]`. Its first Reshape must be
exactly `[N,OH,OW,BS,BS,C]`, the square block must be greater than one, the
Transpose permutation must be `[0,1,3,2,4,5]`, and the final Reshape must be
`[N,OH*BS,OW*BS,C]`. The replacement retains the first Reshape with target
`[N,OH,OW,BS*BS*C]`, changes the same Transpose object in place to
DEPTH_TO_SPACE with the exact block option and final output, and removes the
last Reshape. Operator version, axis semantics, ONNX provenance, output
lineage, public output identity, and unused-tensor pruning are preserved.

All four data tensors have one dtype and either no quantization or the same
valid per-tensor grid. Every internal edge is private, uniquely produced, and
exclusively consumed in increasing graph order. The data source is a graph
input, producer-free constant, or uniquely produced earlier value. Shape and
permutation tensors are exact producer-free INT32/INT64 vectors; runtime graph
inputs are not accepted as constants, and the first shape vector cannot be a
public output because the rewrite changes its value.

The changed first-Reshape vector is prepared before graph mutation. A private
vector is updated in place. A shared vector is cloned with its original dtype
and quantization, using the former deterministic `*_d2s_shape` naming rule;
after that edge is updated in the differential index, a later sole consumer
can update the original vector in place exactly as before. Clone failure is a
complete no-op. One consistent dynamic batch, spatial, or channel dimension is
encoded as the sole `-1` in the changed shape vector, options, and tensor
signature. Contracts needing two inferred first-Reshape dimensions are not
rewritten.

Static public-input, produced-input, quantized, and shared-vector multi-chain
fixtures retain exact former ModelIR and statistics. Extra operator inputs,
floating shape vectors, a public first shape vector, mixed data dtypes, and
inconsistent signatures formerly matched but now remain transactionally
unchanged. Both production call sites supply the Session LayoutState, and a
real ONNX Reshape/Transpose/Reshape graph characterizes the production owner.

Conv1D-shim unary canonicalization is owned by
`passes/conv1d_unary_layout.py`. A two-Transpose, Squeeze, ExpandDims, and
supported-Unary preflight preserves historical unused-tensor pruning without
constructing an index for unrelated graphs. Otherwise one optional or local
`ModelIRGraphIndex` captures graph-order first-Transpose objects and resolves
each against the current graph after earlier chains are removed. The former
per-rewrite full consumer-map rebuild is eliminated.

The exact chain is NHWC rank-four input, Transpose `[0,3,1,2]`, one-axis
Squeeze to rank three, one shape-preserving unary, ExpandDims at the same axis,
and Transpose `[0,2,3,1]` back to the original NHWC shape. Concrete shapes are
positive and every shape signature is either its concrete dimension or `-1`.
The pre-Transpose, squeezed, unary, expanded, and final signatures must satisfy
the complete permutation/drop/insert/inverse-permutation equations. A squeezed
axis must be statically one in both shape and signature. When `squeezeDims` is
absent, exactly one axis must produce the recorded rank-three shape.

Both permutation vectors and the ExpandDims axis are producer-free INT32 or
INT64 constants with exact vector metadata; runtime graph inputs are not
treated as constants. The data source is a public input, producer-free
constant, or uniquely produced earlier value. Every intermediate has one
producer, is private, and is exclusively consumed by the next strictly later
operator. The final output has one producer, is not a graph input, and any
downstream consumers occur after the removed post-Transpose.

Transpose and Squeeze preserve one dtype and valid per-tensor quantization
grid; unary output and ExpandDims preserve another. Non-CAST unary operators
require those groups to be the same. CAST retains the former ability to change
dtype. The final output's dtype and cloned quantization continue to be repaired
from the unary output. That clone is prepared before mutation, so an
unclonable quantization object leaves the complete graph unchanged instead of
rewiring the unary and then raising.

The same unary object is retained with its options, version, axis semantics,
and ONNX provenance. Its input becomes the original NHWC source and its output
becomes the former post-Transpose output; four wrapper operators are removed
with one differential-index compaction. Static public-input, produced-input,
quantized, CAST, inferred-axis, and alternate-spatial-axis fixtures retain
exact former ModelIR and statistics. Floating/produced constants, inconsistent
internal shapes, mixed dtypes, and duplicate final producers formerly matched
but now remain complete no-ops. Consistent dynamic batch, spatial, and channel
signatures are preserved without introducing a Reshape. The sole production
call supplies the Session LayoutState.

The adjacent rank-four Conv1D-shim variant is owned by the same module. Its
exact chain is an NHWC rank-four source, Transpose `[0,3,1,2]`, a supported
unary, Transpose `[0,2,1,3]`, Reshape from `[1,H,C,W]` to `[H,C,W]`,
ExpandDims at axis two, and Transpose `[0,2,3,1]`. The indexed rewrite removes
the three Transposes, runs the unary directly on NHWC, changes the Reshape
target to `[H,W,C]`, and changes ExpandDims to axis one. The original unary,
Reshape, and ExpandDims objects retain their options, version, axis semantics,
and ONNX provenance.

All data tensors satisfy exact shape/signature permutation equations, strict
producer and consumer ordering, compatible dtypes, and valid per-tensor
quantization. CAST may transition dtype; other supported unary operators may
not. Permutation, shape, and axis tensors are typed producer-free integer
vectors and are not public inputs or outputs. At most one retained Reshape
dimension may be dynamic. Shared Reshape-shape or ExpandDims-axis constants
are cloned before their consumer edge changes; private constants update in
place. Quantization and constant clones are prepared before graph mutation, so
clone failure leaves ModelIR unchanged.

When the rank-three Reshape output has side consumers, the owner inserts one
`[0,2,1]` compatibility Transpose immediately before the earliest side
consumer and rewires only those consumers. The primary ExpandDims branch stays
in HWC order. This preserves the former side-branch values while fixing the
legacy helper's locally non-topological append order. Without fan-out, static
public-input, produced-input, quantized, and CAST fixtures retain exact former
ModelIR and statistics. Floating or produced constants, inconsistent metadata,
mixed dtypes, duplicate producers, backward consumers, and public
intermediates are complete no-ops. One supplied or locally constructed
`ModelIRGraphIndex` is maintained across all matches, and the sole production
call supplies the Session LayoutState.

Conv1D-shim unary fan-out bypass is the third semantic family in
`passes/conv1d_unary_layout.py`. It applies only when the rank-three unary
output has one exact ExpandDims/Transpose branch back to NHWC plus a retained
NCHW side use. That side use may be a strictly later operator consumer or the
unary output itself may be a public graph output. A chain without genuine
fan-out is left to the preceding full-chain owner and is not partially
rewritten by this pass.

The full-chain and fan-out families share one
`_resolve_unary_prefix_candidate` for the complete NHWC-to-NCHW Transpose,
Squeeze, unary, shape/signature, source, boundary, producer, consumer, dtype,
and quantization-independent prefix contract. The only prefix-policy
difference is whether a public unary output is permitted. Each family owns
only its distinct suffix validation and rewrite transaction.

The bypass keeps the original unary object but moves it before the retained
NHWC-to-NCHW Transpose. The unary now consumes the original NHWC source and
produces the former post-Transpose output. The retained Transpose consumes that
output, and the retained Squeeze produces the former unary output for every
NCHW side consumer. The selected ExpandDims and post-Transpose are removed in
one indexed compaction, after which the unary is inserted before the retained
Transpose. This produces a topological operator order instead of the legacy
helper's backward producer edge.

Shape and signature equations, typed producer-free permutations and axis,
strict producer/consumer order, private changed intermediates, dtypes, and
per-tensor quantization are validated before mutation. A public unary output
is allowed because its NCHW value and name remain unchanged; the pre-Transpose,
pre-unary Squeeze, and removed ExpandDims outputs are not public. CAST retains
its dtype transition. In that case the retained Transpose output dtype and
quantization are repaired from the unary output, correcting metadata that the
legacy helper left in the input dtype. Both required quantization clones are
prepared before graph mutation, so clone failure is a complete no-op.

Static public-input, produced-input, quantized, alternate-axis, and non-CAST
fixtures become byte-for-byte identical to the legacy result after applying a
topological sort to that legacy result. No-fan-out chains, floating or produced
constants, inconsistent shape/signature metadata, mixed dtypes or grids,
duplicate producers, backward consumers, and public removed intermediates are
complete no-ops. One supplied or locally constructed `ModelIRGraphIndex` is
maintained across all matches, and the sole production call supplies the
Session LayoutState.

Flattened InstanceNormalization Conv1D-shim canonicalization is isolated in
`passes/conv1d_instance_norm_layout.py`; it is not added to the unary owner.
The exact 17-operator path is NHWC-to-NCHW Transpose, axis-two Squeeze,
Reshape to `[N,1,C*W]`, two axis-two kept-dimension means and the complete
`SUB -> square -> variance+epsilon -> SQRT -> reciprocal -> normalize ->
scale -> bias` decomposition, Reshape to `[N,C,W]`, a supported unary,
axis-two ExpandDims, and NCHW-to-NHWC Transpose.

The indexed owner validates every producer, consumer multiplicity, operator
role and order, public boundary, concrete shape, dynamic signature, dtype,
quantization state, and constant before mutation. Mean axes are exact typed
producer-free `[2]` vectors. Epsilon, reciprocal numerator, scale, and bias
are finite producer-free floating scalars; epsilon is nonnegative and the
reciprocal numerator is exactly one. SUB and DIV retain their required input
order, while the mathematically commutative MUL/ADD edges may use either input
order. Extra fan-out from any layout-sensitive normalization intermediate is
rejected.

The rewrite removes only the two boundary Transposes. Squeeze consumes the
original `[N,1,W,C]` source at axis one, its output and the second Reshape and
unary outputs become `[N,W,C]`, the second Reshape target becomes `[N,W,C]`,
and ExpandDims moves to axis one with `[N,1,W,C]` output metadata. Downstream
consumers and graph outputs are redirected from the former post-Transpose name
to the retained ExpandDims output. The InstanceNormalization interior remains
unchanged because normalization across flattened `C*W` is permutation
equivariant.

Changed Reshape and ExpandDims constants are fully planned before graph
mutation. Private constants update in place; shared constants are cloned with
deterministic `*_nhwc_shape` and `*_nhwc_axis` names, preserving unrelated
consumers. Clone failure is a complete no-op. One consistent dynamic batch,
width, or channel dimension is preserved in signatures and the Reshape target;
targets requiring two inferred dimensions are rejected. CAST retains its
output dtype transition, while the floating normalization path requires one
unquantized FLOAT16 or FLOAT32 contract.

Static public-input, produced-input, CAST, and alternate-unary fixtures retain
exact legacy ModelIR and statistics. Floating or produced constants, negative
epsilon, reversed reciprocal DIV, mixed data dtypes, duplicate producers, and
inconsistent metadata formerly matched but are now complete no-ops. One
supplied or locally constructed `ModelIRGraphIndex` is maintained across all
matches, and the sole production call supplies the Session LayoutState.

The adjacent tencoder residual-gate family is isolated in
`passes/conv1d_tencoder_layout.py`. Its right branch reuses
`_resolve_flattened_instance_norm_prefix`, the side-effect-free prefix plan
shared with the complete InstanceNormalization/unary family. Consequently,
the residual pass cannot accept an arbitrary upstream Squeeze/Transpose path:
it must prove the exact two-Mean normalization topology, scalar contracts,
producer/consumer multiplicities, dtype, quantization, shape, and boundary
invariants before considering the gate suffix.

The suffix contract covers two complementary channel Slices from `[N,2C,W]`,
Logistic gating, elementwise multiplication, one `[C,1]` or `[1,C]` floating
scale, a simple rank-four Transpose/Squeeze or legacy rank-three Transpose left
branch, residual ADD, axis-two ExpandDims, NCHW-to-NHWC Transpose, and one or
more Conv consumers. Every changed intermediate is private and has an exact
FLOAT16 or FLOAT32 shape/signature contract. One dynamic batch or width
dimension is carried through the Reshape, Slice, residual, and bridge targets;
a dynamic split-channel dimension is rejected because the two Slice boundaries
are compile-time constants. Layout-sensitive fan-out inside either branch is
rejected instead of being silently reinterpreted.

The plan converts both residual inputs to NWC, updates the second Reshape and
Slice vectors, transposes the channel scale to `[1,C]`, changes ExpandDims to
axis one, and repairs Squeeze, Slice, Logistic, gate, residual, and rank-four
metadata. All integer and floating constant changes are preplanned. Shared
constants receive distinct deterministic clones, including the case where two
changed operator inputs originally share one tensor; private constants update
in place. The three boundary Transposes are removed through one maintained
`ModelIRGraphIndex` only after the complete plan succeeds.

When residual ADD has side consumers, one `[0,2,1]` NWC-to-NCW compatibility
Transpose is inserted immediately before the earliest side consumer and only
those edges are redirected. The Conv path consumes the retained ExpandDims
output directly. This preserves side values and emits topological operator
order, unlike the legacy helper's unconditional append. Exact NumPy
differential tests cover simple and legacy left branches with and without
fan-out, and the sole production call supplies the Session LayoutState.

Conv1D-shim Squeeze/unary/BatchMatMul canonicalization is owned by
`passes/conv1d_batchmatmul_layout.py`. The owner accepts one typed
NHWC-to-NCHW Transpose, an explicit singleton Squeeze, zero or more strict
supported unary operators, and a left-hand BatchMatMul tail. Every changed
intermediate is private and single-consumer; the source may be a public input,
a constant, or a uniquely produced earlier tensor. The right operand and
BatchMatMul output are also validated for unique producers, graph order,
dtype, per-tensor quantization, contracted dimensions, broadcast batch shape,
and concrete output shape.

The exact axis mapping determines the only permitted rewrite. Removing a
transposed channel singleton yields the same `[N,H,W]` rank-three order, so
`adjX` is preserved. Removing either supported spatial singleton changes
`[N,C,L]` to `[N,L,C]`; all Squeeze and unary shape/signature metadata swaps
its last two axes and `adjX` is toggled. In both cases the effective left
matrix, right operand, and BatchMatMul output shape are proved identical before
mutation. The batch dimension may retain a dynamic signature.

The Squeeze consumes the original NHWC tensor at the mapped source axis, the
original unary and BatchMatMul objects retain their unrelated options and ONNX
provenance, and the boundary Transpose is removed through the same
`ModelIRGraphIndex`. Floating or produced permutation constants, fan-out,
public changed intermediates, right-operand placement, shape/dtype/grid
mismatches, per-axis quantization, duplicate producers, and backward consumers
are complete no-ops. The sole production call supplies the Session
LayoutState.

Decoder BatchMatMul-to-TransposeConv input canonicalization is owned by
`passes/decoder_deconv_layout.py`. The exact path is a rank-three
BatchMatMul result, commutative constant-bias ADD, axis-two ExpandDims,
Transpose `[0,2,3,1]`, and input two of one TransposeConv. All changed
intermediates are private and single-consumer. Both matrix operands may be
public, constant, or uniquely produced earlier tensors; rank-two and rank-
three broadcast operands are supported.

The owner validates the original BatchMatMul output from both operand shapes,
`adjX`, `adjY`, and batch broadcasting. It separately proves that swapping the
operands and assigning `new adjX = not old adjY` and
`new adjY = not old adjX` produces the exact `[N,L,C]` transpose of the old
`[N,C,L]` result. ADD, ExpandDims, Transpose, and TransposeConv input metadata
must satisfy every corresponding shape/signature permutation equation. One
dynamic batch signature is retained.

The bias must be a producer-free floating constant with exactly `L` values and
an old broadcast shape of `[L]`, `[1,L]`, or `[1,1,L]`; it becomes
`[1,L,1]`. ExpandDims accepts equivalent axis two or negative axis `-2` and
moves to axis one. Private constants update in place. Shared bias and axis
constants receive deterministic clones, including quantization metadata,
before any operator or tensor changes.

After preflight, the existing BatchMatMul inputs are swapped, only its two
adjoint options change, rank-three result and ADD metadata swap their last two
axes, and the retained ExpandDims output adopts the former Transpose output
contract. TransposeConv input two is redirected and the sole Transpose is
removed through the maintained `ModelIRGraphIndex`. Fan-out, public changed
intermediates, produced/floating constants, incompatible bias, matrix or
layout shapes, per-axis quantization, duplicate producers, backward consumers,
and clone failures are complete no-ops. The sole production call supplies the
Session LayoutState.

Terminal decoder Squeeze/Mean canonicalization is owned by
`passes/terminal_squeeze_mean_layout.py`. The exact path is an NHWC-to-NCHW
Transpose `[0,3,1,2]`, rank-four Squeeze on axis two, rank-three Mean on axis
one with `keepDims=True`, and a second Squeeze on axis one. The source may be
a public input, a constant, or a uniquely produced earlier tensor. The three
changed intermediates are private and single-consumer, while the final rank-
two tensor retains its existing name, graph-output position, consumers, dtype,
shape, signature, and quantization contract.

One `ModelIRGraphIndex` proves the complete operator order and every producer,
consumer, public boundary, typed constant, shape/signature equation, singleton
axis, dtype, and per-tensor quantization contract before mutation. Equivalent
negative Squeeze and Mean axes are accepted. The Mean axis must be a typed,
producer-free one-element integer constant; a shared axis is cloned with a
deterministic `*_nhwc_axis` name so unrelated consumers retain the old value.
Clone allocation and every other changed value are preflighted before an edge
or metadata mutation.

The rewrite makes the first Squeeze consume the original `[N,1,W,C]` NHWC
source on axis one, changes its intermediate from `[N,C,W]` to `[N,W,C]`,
moves the kept-dimension Mean from axis one to axis two, changes its output
from `[N,1,W]` to `[N,W,1]`, and moves the final Squeeze to axis two. The final
`[N,W]` values and metadata therefore remain identical while the boundary
Transpose is removed through the maintained index. Dynamic batch, width, or
reduced-channel signatures are retained. Fan-out, public changed
intermediates, floating or produced
constants, dynamic singleton axes, inconsistent metadata, mixed dtypes or
quantization grids, per-axis quantization, duplicate producers, backward
consumers, and clone failures are complete no-ops. The sole production call
supplies the Session LayoutState.

The direct, side-Squeeze, Squeeze/unary/Reshape, and
Squeeze/residual-ADD/Reshape subsets of decomposed
InstanceNormalization pre/post canonicalization are owned by
`passes/instance_norm_prepost_layout.py`. Their common exact boundary is an
NHWC-to-NCHW Transpose, batch-axis Squeeze, Reshape back to rank four, the
two-Mean `SUB -> square -> variance+epsilon -> SQRT -> reciprocal -> normalize
-> scale -> bias` decomposition. The direct mode requires one private
NCHW-to-NHWC post-Transpose. The side mode requires that post-Transpose plus
exactly one additional axis-zero Squeeze consumer. The unary/Reshape mode
requires an axis-zero Squeeze, one of the thirteen retained unary operators, a
second rank-four Reshape, and the post-Transpose. The residual mode replaces
the unary with an ADD whose other input is normalized from either a rank-three
HWC-to-CHW Transpose or an NHWC-to-NCHW Transpose followed by Squeeze and an
optional supported unary. All four modes are fully owned; the compatibility
function is a 60-line graph-order/shared-32-cap dispatcher with no ModelIR
mutation of its own.

One `ModelIRGraphIndex` proves every producer, duplicate producer, consumer
multiplicity, operator role and order, public boundary, exact concrete shape,
dynamic signature, dtype, unquantized floating contract, and typed constant
before mutation. Both Means must keep dimensions and reduce the two NCHW
spatial axes; equivalent negative or reversed axis vectors are accepted. SUB
and reciprocal DIV retain their noncommutative input order. Epsilon is finite
and nonnegative, the reciprocal numerator is exactly one, and scale and bias
are finite `[1,C,1,1]` FLOAT16 or FLOAT32 constants.

All four rewrites move the first Squeeze and rank-four normalization computation
to NHWC, change the reduction axes to `[1,2]`, transpose every changed tensor
metadata contract, convert scale and bias to `[1,1,1,C]`, and remove only the
two boundary Transposes. Direct and side modes redirect the existing bias ADD
to the post-Transpose output name. In side mode, a fixed `[0,3,1,2]` adapter is
inserted immediately before the side Squeeze and reproduces its original
`inst_output` tensor name, NCHW shape/signature, dtype, and quantization
contract. The adapter constant is validated and reused when compatible or
allocated before mutation when absent. Unary/Reshape mode converts the second
Reshape's CHW input and NCHW output metadata to HWC/NHWC, rewrites its shape
constant and `newShape`, and moves the post-Transpose output name to that
Reshape. CAST alone may change dtype; all other supported unary operators must
preserve it. Dynamic height, width, or channel signatures are carried through
the full, kept-dimension, side-branch, unary, and second-Reshape tensors.
Residual mode also lifts its residual branch to HWC. If the ADD output has
non-Reshape consumers, one preplanned HWC-to-CHW adapter preserves all legacy
consumer slots; compatible INT32/INT64 permutation constants are reused and
incompatible, produced, public, or quantized constants reject the entire
transaction.

Changed Reshape, Mean-axis, scale, bias, and optional second-Reshape constants
are fully planned before mutation. A private constant updates in place; a constant shared with any
unrelated consumer receives one deterministic clone, including one shared
axis vector used by both Means. Clone failure is a complete no-op. Static
public-input, produced-input, FLOAT16, negative-axis, and commuted affine
fixtures produce exactly the same ModelIR as the committed legacy helper.
Separate but equivalent Mean-axis constants, which the legacy helper skipped
because it required one shared tensor, are now handled safely. Unsafe direct-,
side-, unary/Reshape-, or residual/Reshape-tail candidates are not passed back
to a legacy mutator. Invalid, public, produced, floating, or quantized
adapter/shape constants and unsafe tail tensors are complete no-ops. The production calls supply the Session
LayoutState. A supplied graph index is used by the indexed owner, refreshed
immediately after a retained legacy rewrite so subsequent dispatch remains
current, and reconciled once at the compatibility boundary. The compatibility
loop offers each pre-Transpose to all four indexed modes at its original graph
position, so mixed tail modes retain the legacy graph-order priority and one
shared 32-rewrite ceiling.

The adjacent decomposed-InstanceNormalization form whose bias ADD is already
after the NCHW-to-NHWC post-Transpose is owned separately by
`passes/instance_norm_post_bias_layout.py`. Its strict boundary is
`NHWC -> Transpose[0,3,1,2] -> two-Mean decomposition through scale ->
Transpose[0,2,3,1] -> bias ADD`. The common decomposition matcher, exact tensor
contracts, finite FLOAT16/FLOAT32 constant validation, deterministic constant
cloning, and constant-update application live in
`passes/decomposed_instance_norm.py`; both InstanceNorm owners consume this
same side-effect-free contract instead of maintaining a second rule chain.

One `ModelIRGraphIndex` validates the complete operator order, producer and
consumer multiplicity, duplicate producers, public boundaries, exact concrete
shape and dynamic signature permutations, dtype, quantization, typed
permutation/axis constants, nonnegative epsilon, unit reciprocal numerator,
and scale/bias broadcast forms before mutation. Equivalent positive, negative,
or reversed NCHW spatial axes and commuted affine operators are accepted. Scale
and bias may be scalar, `[1,C,1,1]`, or `[1,1,1,C]`; only coefficients that
actually change layout are updated, and unrelated consumers receive one
deterministic clone. A shared scale/bias tensor is planned once for both uses.

The transaction redirects both Mean/SUB source edges to the original NHWC
tensor, changes the Mean axes to `[1,2]`, permutes every retained full and
reduced core tensor contract, redirects the bias ADD from the post-Transpose
output to the scaled tensor, and removes only the two Transposes. Unused bridge
tensors are pruned only after a successful rewrite; a rejected candidate is an
exact no-op. The lowerer compatibility function is a 19-line dispatcher. The
two-pass normalization recovery loop shares one live graph index between the
four-tail owner and this post-bias owner, while all production calls supply the
Session `LayoutState`. The owner has a deterministic 32-rewrite ceiling and no
whole-graph consumer-map rebuild or fixed-point `while` loop.

The dual-branch decomposed-InstanceNormalization form whose normalized result
is added to a residual NCHW branch is owned by
`passes/instance_norm_residual_add_layout.py`. Its exact boundary is two
independent `NHWC -> Transpose[0,3,1,2]` inputs, the common two-Mean
InstanceNorm core through scale and bias, and a residual ADD with at least one
later NCHW consumer. The rewrite removes both input Transposes, performs the
normalization and residual ADD in NHWC, then inserts exactly one
`Transpose[0,3,1,2]` immediately before the earliest downstream consumer. The
adapter reproduces the original ADD output name and NCHW tensor contract, so
all downstream fan-out and repeated input slots remain unchanged.

One `ModelIRGraphIndex` proves the full producer, consumer, graph-order,
boundary, shape/signature, dtype, quantization, permutation, affine-constant,
and adapter contract before mutation. The common constant planner in
`passes/decomposed_instance_norm.py` prepares both Mean-axis updates and the
scale/bias changes as one transaction. Shared constants receive deterministic
clones when required; an existing adapter permutation is reused only when it
is a private unquantized INT32/INT64 `[0,3,1,2]` constant. Clone, output, and
adapter-name collisions, invalid public or produced constants, unsafe dynamic
contracts, and unsupported graph boundaries are complete no-ops.

The indexed owner scans Transpose candidates in graph order, applies at most
32 rewrites, updates the graph index differentially, and prunes unused tensors
only after at least one successful rewrite. Its lowerer compatibility function
is a 19-line dispatcher. The repeated normalization recovery loop supplies the
same live index used by the adjacent post-bias owner, and both production calls
supply the Session `LayoutState`. No full producer/consumer-map rebuild or
unbounded fixed-point loop remains in this path.

The following residual-MUL/CONCAT InstanceNormalization tail is owned by
`passes/instance_norm_residual_mul_concat_layout.py`. Its strict prefix is the
same direct NCHW decomposed core and bias ADD, followed by an
`NCHW -> NHWC` Transpose and a residual ADD. The tail has exactly two MUL users
of that ADD, two scalar or channelwise coefficients, one channel-axis CONCAT,
and a final `NCHW -> NHWC` Transpose. The compatibility name retains its
historical `...concat_conv...` spelling, but the owner deliberately preserves
the legacy contract by not requiring a Conv consumer: the final tensor may be
a public graph output or feed any later, graph-ordered consumers.

All three Transposes are removed in one transaction. The normalization core,
bias ADD, residual ADD, and both tail MULs execute in NHWC; the MUL outputs and
CONCAT metadata are permuted to NHWC; the CONCAT axis changes from 1 to 3; and
the CONCAT directly produces the former final-Transpose output name. Exact
shape/signature permutations, including dynamic height, width, or channel,
preserve the existing public and downstream tensor contract. CONCAT inputs are
compared with multiplicity rather than as a set, so duplicated or missing
branches cannot match accidentally.

`plan_nhwc_instance_norm_constant_updates` accepts additional coefficient uses
for this owner and plans both Mean axes, scale, bias, and the two tail-MUL
coefficients together before mutation. A constant shared by any subset of
those four affine sites is updated once; unrelated consumers receive one
deterministic clone. Invalid, non-finite, produced, public, quantized, mixed-
dtype, wrongly shaped, or colliding constants reject the whole transaction.
One `ModelIRGraphIndex` proves every producer, consumer multiplicity, graph
order, tensor, permutation, public boundary, and output-renaming contract. The
owner uses a bounded graph-order scan, a 32-rewrite ceiling, differential index
updates, success-only pruning, and Session `LayoutState` synchronization. Its
former 501-line lowerer mutator is now a 19-line dispatcher; all four production
calls supply LayoutState, and the repeated recovery loop reuses its live
`residual_graph_index`.

Dual-statistics normalization is intentionally not matched by the standard
decomposed-InstanceNormalization core. It is owned independently by
`passes/instance_norm_dual_stats_layout.py`. The input feeds two branches: one
reduces only NCHW spatial axes `[2,3]`, while the other reduces all non-batch
axes `[1,2,3]`. Each branch has the distinct
`Mean -> SUB -> square -> Mean -> variance-factor -> epsilon -> SQRT ->
DIV(centered,std) -> scale` contract. The standard core instead uses a unit
reciprocal followed by MUL and has no variance factor, so sharing its matcher
would silently conflate different mathematics. Only typed constant planning,
metadata, index, and tensor-contract utilities are shared.

The two scaled branches are added, followed by gamma MUL and beta ADD. Gamma
and beta may be scalar/NCHW/NHWC constants or exact rank-one/rank-two vectors
reshaped to `[1,C,1,1]`. Vector reshapes are validated through their producer,
typed shape constant, dtype, quantization, consumer, source, and graph-order
contracts; on success the NHWC operator consumes the vector directly and the
now-dead Reshape is removed. Direct constants and both branch scales use the
common grouped coefficient transaction. Spatial Mean axes are planned as one
`[1,2]` transaction, including deterministic clones for unrelated users;
global axes remain `[1,2,3]` and are validated but not changed. Finite,
nonnegative scalar variance factors and epsilon values are required.

Two exact output modes are supported. Direct mode removes the input and output
Transposes and makes beta ADD produce the former NHWC output name. Residual
mode additionally proves one independent NHWC-to-NCHW residual bridge, moves
the residual ADD to NHWC, removes all three Transposes, and makes that ADD
produce the preserved NHWC name. The residual ADD's old output contract is
validated and permuted before rename so dynamic axes cannot leak from NCHW
positions into NHWC metadata. The historical helper name contains `resize`,
but the legacy boundary does not require a Resize: any later graph-ordered
consumer is preserved, including fan-out and repeated input slots.

One `ModelIRGraphIndex` proves both complete paths, consumer multiplicity,
producer uniqueness, dependency order, public boundaries, concrete and dynamic
shape/signature permutations, dtype, quantization, typed permutation and axis
constants, coefficient ownership, optional Reshape removal, and final output
rename before mutation. The owner scans graph-order candidates with a
32-rewrite ceiling, updates the index differentially, prunes only after
success, and synchronizes Session `LayoutState`. The former 712-line lowerer
mutator is a 19-line dispatcher. All four production calls supply LayoutState,
and the repeated normalization loop reuses the live `residual_graph_index`.

The strict `MUL(c1) -> ADD(c2) -> MUL(c3)` affine fold is owned by
`passes/affine_chain_fold.py`. It replaces the three operators with
`MUL(c1*c3) -> ADD(c2*c3)` only for finite, non-variable FLOAT16, FLOAT32, or
FLOAT64 constants and unquantized tensors of the same dtype. All three binary
operators must have no fused activation. Static broadcast shapes and dynamic
shape signatures are validated for the original three operations and the
folded two-operation form before any constant or graph mutation.

Constant roles are planned together. A constant shared by the first MUL and
ADD is updated once, and sharing with the removed final MUL is treated as an
internal chain use. Any unrelated consumer receives a deterministic folded
clone while the original value remains unchanged. Produced, public, variable,
quantized, non-finite, incorrectly typed or shaped constants reject the whole
candidate. Clone names are reserved for the complete plan before mutation.

The two intermediate tensors must have exact single-consumer multiplicity,
unique producers, valid dependency order, and no public boundary role. The
source must resolve to a prior producer or public input; downstream fan-out and
repeated slots on the preserved final output remain untouched. The final
output tensor and its dtype, quantization, shape, signature, layout, and ONNX
provenance remain authoritative. This corrects the legacy behavior that copied
the removed intermediate ADD tensor's metadata onto the final output tensor.
Only the surviving first-MUL intermediate metadata is changed when folding a
final broadcast expansion into it.

Every candidate is resolved side-effect-free and resolved again immediately
before apply. The owner uses one differential `ModelIRGraphIndex`, a bounded
graph-order candidate snapshot, a configurable 32-rewrite ceiling,
success-only pruning, and Session `LayoutState` synchronization. The former
219-line raw lowerer loop is a 17-line compatibility dispatcher, and all three
production calls provide the Session LayoutState.

The strict
`NHWC -> Transpose(NCHW) -> MUL(const) -> ADD(const) -> Transpose(NHWC)`
layout island is owned by `passes/affine_prepost_layout.py`. Matching begins at
each MUL candidate and proves the unique pre-Transpose producer, sole ADD
consumer, and every inverse post-Transpose consumer. The pre adapter may remain
for unrelated branches. The ADD output may fan out only through valid inverse
post adapters; all post aliases are merged into the first graph-ordered output
while downstream fan-out and repeated input slots are preserved.

The owner supports finite FLOAT16, FLOAT32, and FLOAT64 scalar or rank-four
constants. Raw NCHW channel, spatial, and full tensors are transposed once to
NHWC. A constant already compatible only with the NHWC target is retained, so
recovery is idempotent after an earlier partial layout sweep. Known
logical/physical NCHW or NHWC annotations disambiguate orientation when
available. If both the direct and rotated non-invariant arrays are compatible
because channel and spatial dimensions coincide, the owner rejects the
candidate instead of guessing an axis meaning. Constants are grouped across
MUL and ADD; unrelated consumers receive deterministic `_nhwc` clones.

Pre/post permutation tensors must be private, typed INT32/INT64 vectors with
exact values. All data tensors must be rank four, unquantized, same-dtype, and
have exact NCHW/NHWC shape and dynamic-signature permutations. Fused binary
activations, public intermediates or post outputs, produced or variable
constants, legacy ADD-output consumers, duplicate producers, invalid order,
and partial alias contracts reject the entire plan before mutation. The
disabled legacy PRELU branch and unused `valid_posts` state are not part of the
owner contract.

The first post output tensor is the authoritative final contract and is not
overwritten. The surviving MUL intermediate adopts its shape, signature,
logical layout, and physical layout; the old ADD intermediate receives the
same metadata only so output renaming cannot contaminate dynamic axes before it
is pruned. This fixes the legacy path that first permuted the ADD intermediate,
copied it onto the canonical output, and then permuted the canonical tensor a
second time. A side-effect-free plan is resolved again immediately before
apply. One differential index, a graph-ordered candidate snapshot, a
configurable 32-rewrite ceiling, success-only pruning, and LayoutState sync
replace the unbounded full-map loop. The former 409-line helper is a 17-line
dispatcher, and all seven production calls provide the Session LayoutState.

The adjacent strict
`NHWC -> Transpose(NCHW) -> MUL(const) -> Transpose(NHWC) -> ADD(const)`
layout island is owned by `passes/affine_post_add_layout.py`. The owner starts
from graph-ordered MUL candidates, proves one typed NHWC-to-NCHW producer and
one typed NCHW-to-NHWC consumer, then validates every consumer of the private
post output as a plain ADD tail. Multiple ADD branches and downstream repeated
input slots are preserved. The pre adapter remains when another branch still
uses its NCHW output; otherwise both Transposes are removed.

The surviving MUL output adopts the removed post-Transpose tensor's exact
shape, dynamic signature, logical layout, and physical layout. This makes the
post tensor the sole authoritative layout contract and replaces the legacy
metadata permutation heuristic. ADD side inputs are finite same-dtype scalar
or exact `[1,1,1,C]` constants. MUL supports finite FLOAT16, FLOAT32, and
FLOAT64 scalar, raw NCHW channel/spatial/full, already-NHWC, and legacy direct
non-rank-four constants under the same shared orientation contract as the
affine pre/post owner. Ambiguous equal-axis non-invariant rank-four constants
are rejected. A changed MUL constant with an unrelated consumer receives one
deterministic `_nhwc` clone.

Producer uniqueness, exact consumer multiplicity, dependency order, public
boundaries, rank-four shape/signature permutations, dtype, quantization,
fused activation, constant provenance, and downstream order are resolved
before mutation with one `ModelIRGraphIndex`. The plan is resolved again at
apply time, including clone-name and removal preflight. Candidate traversal is
bounded by a configurable 32-rewrite ceiling; pruning is success-only and the
Session `LayoutState` is synchronized. The former 278-line lowerer helper is a
17-line dispatcher and all four production calls provide LayoutState. The
four-line Pad compatibility wrapper remains independently owned by
`passes/pad_layout.py` because its topology and constant contracts differ.

The strict two-stage SiNet Shuffle residual island is owned by
`passes/sinet_shuffle_residual_layout.py`. Its one candidate spans five layout
adapters and the complete graph
`ADD -> MUL -> ADD -> PRELU -> {post Transpose, channel Concat} -> MUL -> ADD
-> PRELU -> post Transpose`. The owner matches from the terminal post-
Transpose, proves all thirteen operators with one `ModelIRGraphIndex`, and
rewrites the two residual inputs, the independent Concat input, both affine/
PReLU stages, and the Concat axis as one transaction. The three input adapters
and both post adapters are removed only after the complete plan is re-resolved.

Every intermediate must have one unique producer, exact consumer
multiplicity, valid dependency order, and no public boundary role. The first
PReLU output must have exactly the intended Concat and inverse-Transpose uses;
the second PReLU output must feed only its inverse Transpose. Arbitrary later
fan-out and repeated input slots from the two canonical post outputs remain
unchanged. Both residual inputs must have identical concrete and dynamic
contracts. The independent branch, residual branch, and channel-axis Concat
must agree in batch and spatial dimensions, including conservative propagation
of unknown signature axes. All data tensors are rank-four, unquantized,
FLOAT16/FLOAT32/FLOAT64 tensors of one dtype; permutation tensors are typed,
private, exact constants; binary/Concat/PReLU fused activations must be NONE.

Six scalar or NCHW/NHWC channel constants cover both MUL, ADD, and PReLU
stages. Constant roles are grouped by tensor identity before mutation. A
constant reused by several roles is updated once, even across both stages;
unrelated consumers receive one deterministic `_nhwc` clone. Produced,
public, variable, quantized, non-finite, wrongly typed or shaped constants
reject the whole candidate. Clone names, all mutation indices, output tensors,
and removable operators are preflighted before the first write.

The existing post-Transpose tensors remain the authoritative public-facing
contracts. The first stage's ADD/MUL/ADD intermediates adopt the first post
tensor's exact shape, signature, and layouts; the Concat and second stage's
MUL/ADD intermediates adopt the second post tensor's contract. The PReLU
operators produce those canonical names directly, so neither canonical tensor
is permuted or overwritten. A graph-ordered candidate snapshot, configurable
32-rewrite ceiling, success-only pruning, and Session `LayoutState` sync
replace the legacy unbounded full-map loop. The former 482-line lowerer helper
is a 17-line dispatcher and its production call supplies LayoutState.

The paired partially restored SiNet variant shares the same
`_resolve_prefix` contract. Its variant-specific tail is
`Concat(NCHW) -> MUL -> Transpose(NHWC) -> ADD -> PRELU`; only the MUL and
Concat move from NCHW to NHWC, while the ADD/PReLU tail already has the target
layout. The terminal root is the post-MUL Transpose. It must be private, have
one exact MUL producer and one exact ADD consumer, and the ADD must have one
exact PReLU consumer. The final PReLU output may be a public output or retain
arbitrary later fan-out and repeated slots.

The MUL now produces the existing post-Transpose tensor name directly. That
canonical tensor, the ADD output, and the final PReLU output remain unchanged;
only the Concat intermediate adopts the canonical post tensor's shape,
signature, and layouts. The same grouped six-role constant transaction handles
the first-stage affine/PReLU constants, the pre-Transpose MUL constant, and the
already-NHWC ADD/PReLU constants. Raw channel constants accepted by the legacy
recovery path are normalized in the same all-or-nothing plan.

The owner re-resolves the shared prefix and variant tail immediately before
apply, preflights every constant clone, operator mutation, output rewrite,
metadata update, and five-adapter removal, and updates one differential index.
Its graph-order traversal is capped at 32 rewrites, with success-only pruning
and optional LayoutState synchronization. The former 470-line lowerer helper
is a 17-line dispatcher. The primary production call supplies the Session
LayoutState; the independently inferred fallback IR call passes the explicit
`None` boundary.

The adjacent late residual fan-out is a third indexed owner in
`passes/sinet_shuffle_residual_layout.py`. It recognizes the semantic island
`two NHWC inputs -> two NCHW adapters -> ADD -> MUL -> ADD -> PReLU`, where
exactly one input comes from a channel-last Concat. The PReLU feeds both a
conv-side NCHW-to-NHWC adapter and at least one legacy NCHW consumer. Matching
does not depend on the former fixed 40-by-40 spatial guard: exact rank-four
shape and dynamic-signature permutations, one floating dtype, unique
producers, consumer multiplicity, public boundaries, and dependency order are
the contract.

The ADD/MUL/ADD/PReLU island is lifted to NHWC and the two input adapters are
removed. PReLU produces the existing canonical NHWC post tensor directly. The
old post adapter remains as the one inverse NHWC-to-NCHW adapter and produces
the former PReLU tensor name, preserving every later legacy consumer and
repeated input slot. The canonical post tensor is not overwritten or
re-permuted; the three affine intermediates adopt its exact shape, signature,
logical layout, and physical layout, while the legacy tensor retains its NCHW
contract.

Affine and PReLU constants must be finite, same-dtype, private constants that
broadcast in the original NCHW operation. Non-scalars are explicitly rotated
and must also broadcast in the target NHWC operation. This replaces the old
size-specific assumption and safely supports raw channel constants as well as
already-oriented rank-four constants when their actual axes make both graphs
valid. Constants shared across roles are updated once; unrelated consumers
receive one deterministic clone. The retained permutation constant is part of
the same transaction and is cloned when another Transpose still needs its
original value.

The complete plan is resolved again before apply. Clone names, operator
indices, metadata targets, output renames, and both removals are preflighted
before the first mutation. Candidate traversal uses one differential
`ModelIRGraphIndex`, deterministic graph order, a configurable 32-rewrite
ceiling, success-only pruning, and optional Session `LayoutState`
synchronization. The former 331-line raw helper is a 17-line compatibility
dispatcher and its production call supplies the Session LayoutState.

The deep-skip Resize/affine residual island is isolated in
`passes/sinet_deep_skip_layout.py`. Its matcher is deliberately staged into
three bounded semantic resolvers: the terminal Concat/MUL/post-Transpose/
ADD/PReLU tail, the central ADD/MUL/ADD/PReLU residual stage, and the
Resize-plus-affine dual branch feeding the first Concat. A final resolver joins
those locally proven contracts and applies one all-or-nothing transaction; no
stage mutates a partially matched graph.

Four private NHWC/NCHW adapters are removed. The Resize branch's MUL/ADD,
both channel Concats, the central residual stage, and the terminal MUL move to
NHWC. The terminal ADD/PReLU remains on its existing canonical NHWC tensors.
The central residual input may be either an already annotated channel-last
tensor or an NCHW tensor with one explicit earlier NCHW-to-NHWC adapter. The
former 40-by-40 shape-name heuristic is not used: relational shape and dynamic
signature equality, typed permutations, and the tensor's logical/physical
layout select the form.

Each intermediate has one unique producer, exact consumer multiplicity, and
valid dependency order. Concat shapes and signatures are derived on axis 1 in
the original NCHW graph and axis 3 in the target NHWC graph. The Resize source,
independent branch, central residual, skip branch, and terminal canonical post
tensor must agree in batch and spatial axes, including conservative unknown
signature propagation. All activation tensors are unquantized rank-four
FLOAT16/FLOAT32/FLOAT64 tensors of one dtype; fused activations and public
intermediate boundaries reject the complete candidate.

Six NCHW-side affine/PReLU constants are validated against both the original
NCHW broadcast and the rotated NHWC broadcast. The terminal ADD/PReLU
constants use the canonical NHWC contract. Constant roles are grouped by
tensor identity across the entire island; unrelated consumers receive one
deterministic clone, and conflicting shared orientations reject the plan. The
SiNet owners share one constant-application and one metadata-application
implementation so clone rewiring and layout updates cannot diverge.

The complete graph is re-resolved before apply. Clone names, four removals,
five input changes, both Concat axes, and every metadata target are preflighted
before the first write. One differential `ModelIRGraphIndex`, graph-ordered
candidates, a configurable 32-rewrite ceiling, success-only pruning, and
Session `LayoutState` synchronization replace the legacy unbounded full-map
loop. The former 641-line lowerer helper is a 17-line dispatcher and its
production call supplies LayoutState.

The adjacent SiNet pre-ADD fan-out island is owned by
`passes/sinet_preadd_fanout_layout.py`. It shares the late residual owner's
strict terminal contract for
`ADD -> MUL -> ADD -> PReLU -> Transpose -> Conv`, including the later legacy
NCHW consumers. Its distinct prefix proves one channel-last Concat followed by
an NHWC-to-NCHW adapter and one direct NCHW source that already has exactly one
earlier NCHW-to-NHWC sibling adapter. The direct source may have no other
consumer, so the existing sibling tensor is the unique channel-last value used
by the lifted ADD.

The rewrite removes only the Concat-side adapter. The ADD consumes the
canonical Concat output and the retained sibling adapter output, and the
affine/PReLU tail moves to NHWC. PReLU produces the existing canonical post
tensor. The terminal post adapter is inverted in place and produces the former
PReLU NCHW tensor, preserving every later legacy consumer and repeated slot.
The retained sibling adapter remains available to its original branch.

Rank-four concrete shapes and dynamic signatures must prove the two exact
layout permutations and all stage tensors; no model name or fixed spatial
size is used. The Concat axis is explicitly channel-last. Floating dtype,
quantization, public boundaries, unique producers, exact consumer
multiplicity, dependency order, fused activation, downstream Conv ownership,
and the existence and ordering of legacy consumers are validated before any
mutation. The three affine/PReLU constants must broadcast in NCHW and after
their explicit NHWC rotation. The inverted terminal permutation participates
in the same grouped constant transaction, so unrelated uses receive a
deterministic clone.

Candidate resolution is repeated immediately before apply. Clone collisions,
all mutation indices, metadata targets, output swaps, and the adapter removal
are preflighted before the first write. One differential `ModelIRGraphIndex`,
graph-order candidates, a configurable 32-rewrite ceiling, success-only
pruning, and Session `LayoutState` synchronization replace the former full-map
loop. The former 359-line lowerer helper is a 17-line compatibility dispatcher
and its sole production call supplies LayoutState.

The two SiNet dual-Resize residual variants share one indexed owner in
`passes/sinet_dual_resize_layout.py`. Both variants prove the complete graph
of two
`Resize -> NHWC-to-NCHW Transpose -> MUL -> ADD` branches, a channel-axis
Concat, residual ADD, terminal MUL/ADD/PReLU, and one or more
NCHW-to-NHWC post adapters. Their only semantic difference is the residual
source boundary. The direct variant owns and removes a private
NHWC-to-NCHW residual adapter. The deep-skip variant reuses an earlier
NCHW-to-NHWC sibling adapter and leaves it available to its existing Conv
branch.

Both Resize outputs are the authoritative branch-local NHWC contracts. Their
MUL and ADD intermediates adopt the exact concrete shape, dynamic signature,
logical layout, and physical layout of the corresponding Resize output. The
two branch contracts independently prove both the original NCHW axis-1
Concat and target NHWC axis-3 Concat. The canonical first post-adapter output
is the authoritative merged contract for the Concat and terminal
ADD/MUL/ADD intermediates; it is not overwritten or permuted twice.

The owner accepts both bilinear and nearest-neighbor Resize operators and
finite FLOAT16/FLOAT32/FLOAT64 scalar or raw NCHW broadcast constants. Four
branch affine constants and three terminal affine/PReLU constants must
broadcast before and after explicit channel-axis rotation. Constants are
grouped across the entire island, so shared roles are updated once and
unrelated consumers receive one deterministic clone. Typed permutation,
producer uniqueness, exact fan-out, dependency order, public boundaries,
rank-four shape/signature relations, dtype, quantization, fused activation,
and Resize provenance are validated before mutation. No model name or fixed
spatial dimension participates in matching.

Multiple equivalent post adapters are merged into the first graph-ordered
canonical output while all downstream repeated input slots are preserved.
For the direct variant, later consumers of the old NCHW PReLU name receive one
topologically inserted inverse adapter, preserving local numerical semantics.
The deep-skip compatibility boundary retains the established pipeline
contract: later consumers are rewired to the canonical NHWC post tensor for
the following SiNet layout recovery passes. This behavior is explicit in the
mode-specific plan instead of being an incidental side effect of separate raw
loops.

The complete plan is resolved again immediately before apply. Constant clone
names, every mutation and removal index, repeated-slot rewrites, metadata
targets, and optional compatibility-adapter insertion are preflighted before
the first write. One differential `ModelIRGraphIndex`, graph-order candidates,
a configurable 32-rewrite ceiling, success-only pruning, and Session
`LayoutState` synchronization replace two independent full-map loops. The
former 505-line direct helper and 503-line deep-skip helper are both 17-line
compatibility dispatchers and both production calls supply LayoutState.

The adjacent shared-post SiNet fan-out island is owned by
`passes/sinet_shared_post_layout.py`. It proves two private
NHWC-to-NCHW adapters feeding
`ADD -> MUL -> ADD -> PReLU -> NCHW-to-NHWC Transpose`, with exactly one
adapter source produced by a channel-last Concat. The canonical post tensor
must fan out to at least one Conv or DepthwiseConv data input and at least one
plain ADD; any other consumer role rejects the complete candidate.

The canonical post tensor is the authoritative NHWC contract. Both source
tensors must have the same rank-four concrete shape and dynamic signature,
while every NCHW stage tensor must be its exact permutation. The rewrite
removes both input adapters and the terminal output adapter, makes the first
ADD consume the two canonical NHWC sources, makes PReLU produce the existing
post tensor directly, and updates only the three affine intermediates to the
canonical metadata. Post-consumer names and repeated ADD input slots remain
unchanged.

All activation tensors use one unquantized FLOAT16/FLOAT32/FLOAT64 contract.
Typed permutation constants, unique producers, exact private fan-out,
dependency order, public boundaries, Concat axis and fused activation, Conv
data-input ownership, and plain affine operators are validated before
mutation. MUL, ADD, and PReLU constants must be finite same-dtype broadcasts
in the original NCHW graph and after explicit NHWC rotation. Identity-grouped
roles are updated once; constants with unrelated consumers receive one
deterministic clone.

The plan is resolved again immediately before apply. Constant clone names,
all removal and mutation indices, post consumers, and metadata targets are
preflighted before the first write. One differential `ModelIRGraphIndex`,
graph-order candidates, a configurable 32-rewrite ceiling, success-only
pruning, and Session `LayoutState` synchronization replace the former full-map
fixed-point loop. The former 321-line lowerer helper is a 17-line compatibility
dispatcher and its sole production call supplies LayoutState. Matching uses no
model name or fixed spatial dimension.

The mid-stage SiNet Concat/Resize affine island is owned by
`passes/sinet_concat_resize_layout.py`. It proves three private
NHWC-to-NCHW adapters: one merged residual input, one independent Concat
branch, and one bilinear or nearest-neighbor Resize output followed by
MUL/ADD. The two branch results feed a plain NCHW axis-1 Concat, then the
merged residual ADD/MUL/ADD/PReLU tail and one or more NCHW-to-NHWC post
adapters.

The Resize output is the authoritative NHWC contract for its affine branch.
The first graph-ordered post output is the authoritative merged NHWC contract.
The independent branch and Resize branch must derive the original NCHW
axis-1 Concat and target NHWC axis-3 Concat exactly; the residual source and
all tail tensors must match the corresponding merged contracts. The rewrite
removes all three input adapters and every equivalent post adapter, changes
the Concat axis to 3, lifts the affine and residual tails to NHWC, and makes
PReLU produce the canonical post tensor directly.

Additional post aliases are merged into the canonical output while every
downstream repeated input slot is preserved. If the former PReLU NCHW tensor
has later consumers, one inverse adapter is inserted immediately before the
first such consumer. This preserves the existing downstream residual branch
without allowing a removed adapter or stale producer to survive.

All activation tensors use one unquantized FLOAT16/FLOAT32/FLOAT64 contract.
Typed permutations, exact Resize provenance, producer uniqueness, consumer
multiplicity, dependency order, public boundaries, fused activation, Concat
axis, rank-four concrete shape, dynamic signature, and logical/physical
layout are validated before mutation. Two Resize-branch constants and three
merged-tail constants must be finite same-dtype broadcasts before and after
explicit channel-axis rotation. Identity-grouped roles are updated once and
unrelated consumers receive deterministic clones.

The complete plan is resolved again immediately before apply. Constant clone
names, all mutation/removal indices, alias input slots, metadata targets, and
legacy-adapter insertion are preflighted before the first write. One
differential `ModelIRGraphIndex`, graph-order candidates, a configurable
32-rewrite ceiling, success-only pruning, and Session `LayoutState`
synchronization replace the former full-map fixed-point loop. The former
487-line lowerer helper is a 17-line compatibility dispatcher, and both
production calls supply LayoutState. Matching uses no model name or fixed
spatial dimension.

The two-Concat SiNet affine tail is owned by
`passes/sinet_tail_concat_layout.py`. It reuses the indexed adapter and
Resize-affine branch contracts from `sinet_concat_resize_layout.py`, then
proves two nested residual stages. The first stage merges an independent
branch and one Resize/affine branch, adds a same-width residual, and applies
MUL/ADD/PReLU. The second stage concatenates that result with an independent
skip adapter and applies a second MUL/ADD/PReLU plus one or more post adapters.

The first residual source is authoritative for the first stage's NHWC
contract. The first graph-ordered post output is authoritative for the second
stage's merged NHWC contract. Both original NCHW axis-1 Concats and target
NHWC axis-3 Concats are derived independently from their branch shapes and
dynamic signatures. Four input adapters and all equivalent post adapters are
removed only after both stages satisfy their exact relational contracts.

The rewrite reconnects the Resize-affine data input, both canonical branch
sources, and both residual/skip sources, changes both Concat axes to 3, and
makes the final PReLU produce the canonical post tensor. Additional post
aliases retain all repeated downstream slots. Later consumers of the former
final NCHW PReLU tensor receive one inverse adapter inserted before their
first use.

Eight finite FLOAT16/FLOAT32/FLOAT64 constants are validated as broadcasts in
their original NCHW stage and after explicit NHWC rotation: two Resize-branch
constants, three first-stage affine/PReLU constants, and three second-stage
constants. Identity-grouped roles update once and unrelated consumers receive
deterministic clones. Typed permutations, Resize provenance, unique producers,
exact fan-out, dependency order, public boundaries, fused activation,
quantization, rank-four shape/signature, and layout are complete guards.

The plan is resolved again immediately before apply. Clone names, mutation
and removal indices, alias slots, metadata targets, and the optional legacy
adapter are preflighted before mutation. One differential graph index,
graph-order candidates, a configurable 32-rewrite ceiling, success-only
pruning, and Session LayoutState synchronization replace the former full-map
fixed-point loop. The former 654-line lowerer helper is a 17-line compatibility
dispatcher and its sole production call supplies LayoutState. No model name or
fixed spatial dimension participates in matching.

The late SiNet Softmax-mask residual island is owned by
`passes/sinet_softmax_mask_layout.py`. It proves two private NHWC-to-NCHW
input adapters. The main branch applies MUL/ADD, wraps a channel Softmax in
the self-inverse `[0,3,2,1]` permutation, reduces its channel maximum, forms a
singleton-channel mask through SUB/Reshape/MUL, and combines that mask with a
PReLU side branch before the residual ADD and one or more post adapters.

The first graph-ordered post output is the authoritative NHWC contract. Every
main, side, mask, and residual NCHW tensor must be its exact rank-four
permutation; the Softmax wrapper has the derived NWHC contract, and the
ReduceMax/SUB/Reshape contracts are derived by removing and reinserting the
channel dimension. The rewrite removes both input adapters, both Softmax
wrapper adapters, and every equivalent terminal adapter. Softmax directly
consumes and produces NHWC tensors, ReduceMax axis 1 becomes axis 3, and the
Reshape target changes from `[N,1,H,W]` to `[N,H,W,1]`.

Three affine/PReLU constants and the full mask-expansion constant must be
finite same-dtype broadcasts in the original NCHW graph and after explicit
NHWC rotation. The two integer axis/shape constants retain their original
INT32 or INT64 dtype. All six transformed constants are grouped by identity;
unrelated consumers receive deterministic clones and incompatible shared
roles reject the candidate. The SUB singleton is validated but never mutated.

Post aliases retain every repeated downstream slot. Later consumers of the
former final NCHW residual receive one inverse adapter before their first use.
Typed permutations, unique producers, exact private fan-out, dependency order,
public boundaries, Softmax axis and finite beta, ReduceMax keep-dims behavior,
plain fused activations, FLOAT16/FLOAT32/FLOAT64 dtype, quantization, layout,
shape, and dynamic signature are complete preconditions.

The complete plan is resolved again before apply. Constant clone names,
mutation/removal indices, reshape options, alias slots, metadata targets, and
optional legacy-adapter insertion are preflighted before the first write. One
differential graph index, graph-order candidates, a configurable 32-rewrite
ceiling, success-only pruning, and Session LayoutState synchronization replace
the former full-map fixed-point loop. The former 612-line lowerer helper is a
17-line compatibility dispatcher and its sole production call supplies
LayoutState. Matching contains neither a model name nor a fixed spatial size.

The SiNet double-Logistic mix-attention compatibility island is owned by
`passes/sinet_mix_attention_layout.py`. A production audit established that
the legacy helper matches zero candidates across all six invocations in the
current `sinet_320_op.onnx` ordered pipeline. The semantic owner nevertheless
preserves both historical residual forms: one private NHWC-to-NCHW branch
adapter paired with either a direct residual adapter or an ADD of two private
residual adapters.

The resolver begins at the terminal NCHW-to-NHWC adapter and its sole post-
Conv consumer. It proves the double-Logistic gate and the
`gate * branch + (1 - gate) * residual` tail back to their shared source ADD.
The channel-attention branch is Mean, NCHW-to-NHWC, two Conv operators, and
NHWC-to-NCHW. The spatial-attention branch is Mean plus ReduceMax, channel
Concat, MirrorPad, NCHW-to-NHWC, Conv, and Reshape. The position-attention
merge uses two rank-five Reshapes, Concat, one rank-four Reshape, MirrorPad,
NCHW-to-NHWC, Conv, and NHWC-to-NCHW. Binary and Concat roles are resolved by
producer semantics and remain valid when their input order is reversed.

The terminal NHWC tensor is the authoritative rank-four contract. All NCHW
attention and tail tensors must be its exact typed permutation. Channel-
attention and spatial-attention shapes, both rank-five expansion shapes, their
Concat result, and the rank-four position-attention shape are derived
relationally. Reduce axes, both MirrorPad pair tensors, all Reshape targets,
and Concat axes retain their original INT32 or INT64 dtype while being
remapped. Transformed rank-five tensors receive unknown layout metadata rather
than an invalid rank-four label.

Every transformed constant is grouped by tensor identity. Unrelated consumers
receive deterministic clones and conflicting shared orientations reject the
candidate. Producer uniqueness, exact fan-out, dependency order, private
intermediates, public boundaries, plain fused activations, finite floating
data, dtype, quantization, layout, concrete shape, and dynamic signature are
complete guards across the island.

The plan is resolved again immediately before apply. Constant clone names,
operator indices, input slots, option changes, metadata targets, and the eight
direct-residual or nine ADD-residual adapters are preflighted before mutation.
The rewrite connects the branch and residual uses to their canonical NHWC
sources. This corrects the former compatibility helper's defective branch
rewrite, which removed the branch adapter but left consumers attached to its
now-unbound output name.

One differential `ModelIRGraphIndex`, graph-ordered candidates, a configurable
32-rewrite ceiling, success-only pruning, and Session `LayoutState`
synchronization replace the full-map fixed-point loop. The former 808-line
lowerer helper is a 17-line compatibility dispatcher and both production calls
supply LayoutState. Matching contains neither a model name nor a fixed spatial
size.

The preceding SiNet SA/PA MirrorPad compatibility island is owned by
`passes/sinet_sa_pa_mirrorpad_layout.py`. A production audit established that
both the legacy and indexed owners match zero candidates across all seven
invocations in the current `sinet_320_op.onnx` ordered pipeline. The path is
retained as a semantically guarded compatibility owner rather than broadened
against that artifact.

The resolver begins at one private NHWC-to-NCHW source adapter. Its exact
NCHW fan-out is Mean, ReduceMax, and one rank-five source Reshape; the legacy
form additionally feeds the terminal Mul. Mean and ReduceMax converge at a
channel Concat followed by MirrorPad, NCHW-to-NHWC, Conv, and one removable
Reshape. The second attention input is an external NHWC channel-attention
value behind a private NHWC-to-NCHW adapter. Their ADD result and the original
source are expanded to rank five, concatenated, reshaped to rank four, padded,
and passed through NCHW-to-NHWC and Conv before Logistic and Mul.

The terminal Mul has two explicit contracts. The direct form consumes the
original NHWC source and an NHWC gate. The legacy form consumes the NCHW
source and a gate behind one NHWC-to-NCHW adapter. The rewrite lifts both to
NHWC. For the legacy form, Mul receives a deterministic private NHWC output
and one inverse adapter is inserted immediately afterward, preserving the old
NCHW tensor name, public output, and all downstream consumers.

The source channel must be concretely one. This is the invariant that makes
the removed SA Reshape and channel-attention adapter value-preserving; the
former raw helper did not state it. Non-singleton candidates are now
transactional no-ops. Spatial dimensions remain arbitrary and dynamic
signatures are derived relationally.

Both reduce axes, both MirrorPad tensors, both rank-five expansion targets,
and the final rank-four target retain INT32 or INT64 dtype while being
remapped. Constant roles are grouped by tensor identity. Unrelated consumers
receive deterministic clones and incompatible shared roles reject the plan.
Rank-four metadata follows the proven NHWC contract, while rank-five tensors
are explicitly layout-unknown.

The complete plan is resolved again immediately before apply. Source
provenance, producer uniqueness, exact fan-out, dependency order, public
boundaries, constant clone and legacy-output names, input slots, option
changes, metadata targets, and all five direct or six legacy removals are
preflighted before mutation. One differential `ModelIRGraphIndex`, graph-
ordered candidates, a configurable 32-rewrite ceiling, success-only pruning,
and Session `LayoutState` synchronization replace the full-map fixed-point
loop. The former 683-line lowerer helper is a 17-line compatibility dispatcher
and all three production calls supply LayoutState. Matching contains neither a
model name nor a fixed spatial size.

The general symmetric and asymmetric Transpose/binary compatibility path is
owned by `passes/binary_bridge_layout.py`. The former 650-line lowerer helper
is a 17-line dispatcher, and its one non-QDQ production call supplies the
Session LayoutState. A pre-extraction audit found zero symmetric and zero
asymmetric matches in each of five short representative production models, so
the owner preserves the proven compatibility topologies without broadening
them from model-specific evidence.

Symmetric matching starts from a plain ADD, SUB, MUL, or DIV. Both inputs must
be private outputs of distinct Transposes with the same typed permutation. The
owner retains three output contracts. A sole inverse post is removed together
with both pre-adapters. With later legacy-layout users, the post is retained
and reversed to adapt the new raw-layout binary output back to the old tensor
name. With no inverse post, a uniquely named raw binary output and one adapter
are inserted immediately before the first existing legacy consumer. The
permanently disabled former Pattern C implementation is not part of the new
production owner.

Asymmetric matching requires exactly one private pre-Transpose and one sole
inverse post-Transpose. The pre operator is reused to transform the plain
operand with the inverse permutation, then the binary consumes the original
raw operand and the transformed plain operand in the order required by the
original expression. SUB and DIV therefore remain order-sensitive. The plain
operand must already exist before the reused Transpose; this explicit guard
prevents a producer-after-consumer graph that the raw helper could create.

Both resolvers require unique producers, resolved graph-input/constant/earlier
operator sources, exact consumer multiplicity, dependency order, protected
public boundaries, no fused activation, immutable INT32/INT64 permutation
constants, same dtype, per-tensor quantization, static broadcast consistency,
and compatible dynamic signatures. Mixed fan-out uses the already-proven pre
permutation tensor instead of mutating the possibly shared post constant. The
legacy-only raw tensor shape is derived from the broadcast contract, and its
adapter insertion index is preflighted before any mutation.

Every complete plan is re-resolved immediately before apply. One differential
`ModelIRGraphIndex`, graph-ordered binary candidates, symmetric-before-
asymmetric phase priority, a configurable 32-rewrite ceiling, success-only
pruning, and LayoutState synchronization replace the repeated full-map
producer/consumer rebuild and unbounded `while True` loop. The no-post path no
longer rewires inputs before later tensor and quantization guards, so a rejected
candidate is a transactional no-op.

The five late safe binary recovery modes share the same owner and one ordered
entry point, `run_safe_binary_bridge_recovery`. Their historical phase order
is fixed as symmetric legacy-only, symmetric single-post, symmetric mixed
fan-out, asymmetric fan-out, and symmetric full-post fan-out. The lowerer
retains 17-line compatibility dispatchers for the five former helpers, but all
three lowerer call sites now call the single ordered owner and supply Session
LayoutState. This replaces 938 lines of repeated full-map
fixed-point mutation.

Legacy-only and single-post modes reuse the strict symmetric resolver and
applier. The safe phases additionally retain the historical
`__preserve_layout_boundary__` marker on inserted or reversed adapters; the
earlier general bridge phase does not acquire this late cleanup boundary.
This distinction preserves later activation-fusion and adapter-cleanup order.

Mixed fan-out and full-post fan-out share one multi-post plan. Every output
consumer is classified as an inverse post or a legacy-layout user. Full-post
mode requires at least two inverse posts and no legacy user, selects the first
post output as the canonical raw tensor, rewires later aliases, and removes all
posts. Mixed mode requires both classes, converts the first post into the one
raw-to-legacy adapter, rewires later post aliases to the canonical raw tensor,
and removes only the remaining posts. The retained adapter references the
already-proven pre-permutation constant; the former mutation of a potentially
shared inverse-permutation buffer is gone. It must precede every legacy user.

Asymmetric fan-out retains its distinct requirement for an existing inverse
Transpose of the plain binary operand. That producer must already precede the
binary. The binary consumes the original raw pre-adapter source and this
existing raw-layout peer without changing SUB or DIV operand order. The first
inverse output post supplies the canonical result. When the original output
has other consumers or is public, one adapter is inserted immediately after
the binary to preserve the original tensor name and layout.

All five phases use the same differential `ModelIRGraphIndex`. Each phase
enumerates current binary candidates in graph order and has its own
configurable 32-rewrite ceiling. Typed permutations, source provenance,
producer uniqueness, exact fan-out, graph order, public boundaries, fused
activation, dtype, per-tensor quantization, broadcast shape, dynamic
signature, alias consumers, and every removal/insertion index are planned and
re-resolved before apply. Pruning and LayoutState synchronization occur once
after the selected ordered phases.

A production audit observed four ordered invocations per model across five
short representatives. Four modes matched zero candidates everywhere. Only
the first SiNet invocation was active: legacy-only rewrote `Add_52` and
`Add_109`, exactly two candidates. The indexed sequence preserves those two
matches and emits byte-identical SiNet float32, float16, correspondence, and
schema artifacts.

The binary/Split channelwise tail is owned by
`passes/split_channelwise_layout.py`. It recognizes one exact private
NHWC-to-NCHW Transpose followed by two plain ADD, SUB, MUL, DIV, MAXIMUM, or
MINIMUM operators and an equal channel Split. The former 218-line lowerer
helper is a compatibility dispatcher, and both unchanged ordered production
sequence positions supply Session `LayoutState`.

The resolver proves the typed INT32/INT64 `[0,3,1,2]` permutation, exact
rank-four source/adapter shape and dynamic-signature relationship, unique
producers, one-use linear prefix, producer-before-consumer order, same dtype,
per-tensor quantization, and NHWC broadcast validity for both binary
operations. Operand slots are retained, so SUB and DIV remain order-sensitive.
External rank-four values must carry NHWC physical evidence or satisfy the
source-relative NHWC broadcast contract; explicit NCHW inputs are rejected.

The root Split axis must be a private immutable axis-one INT32/INT64 constant.
Every Split output must have the exact channel-last metadata implied by its
former NCHW contract, with equal channels that reconstruct the input. A shared
axis constant receives a deterministic clone, preserving both the declared and
NumPy dtype. The same contract applies to downstream Split operators.

Closure discovery uses an edge-bounded consumer `deque`, not repeated graph
scans. Only the audited layout-preserving unary family, plain binary family,
channel Concat, and channel Split may enter the closure. Concat axis 1 or -3
becomes axis 3. Static shapes and dynamic signatures are recomputed for every
binary and Concat output. Every rewritten tensor must terminate at another
accepted closure operator or the sole public output; an unsupported consumer,
dead converted branch, multiple public outputs, or consumer-before-producer
ordering rejects the complete candidate.

One terminal NHWC-to-NCHW adapter preserves the public output name, original
NCHW shape, and layout. Its private input receives the converted NHWC metadata,
and it reuses the proven input permutation constant. This corrects the raw
helper behavior that permuted the public tensor metadata itself before adding
the terminal adapter. Removing the input adapter is also conditional on a
closed closure, so no unsupported consumer can retain an unbound or
wrong-layout tensor.

All tensor data, metadata, quantization, operator inputs/outputs/options,
provenance, axis-clone names, public private-name allocation, and graph order
are captured in an immutable plan. The complete plan is re-resolved before
apply, every slot and new name is preflighted, and only then are axis clones,
metadata, the terminal adapter, and input-adapter removal applied through one
`ModelIRGraphIndex`. Candidate-only execution and a configurable 32-rewrite
ceiling replace the former unbounded full-map fixed-point loops. LayoutState is
updated differentially and unused tensors are pruned only after success.

Pre-extraction production characterization observed zero matches in every
runtime invocation: four each for YuNet, FastestDet, HumanSeg, and OSNet, and
eight for SiNet. The semantic owner retains the active synthetic compatibility
path without broadening from those artifacts. Thirty-three focused tests cover
all six binary operations and both operand orientations, exact numerical
behavior, dynamic signatures, downstream Split/Concat closure, shared INT64
axis cloning, public output layout, differential index and LayoutState,
candidate limits, idempotence, deterministic names, and sixteen transactional
unsafe no-op contracts. A sequential YuNet comparison against the preceding
source checkpoint emitted byte-identical float32, float16, correspondence, and
schema artifacts.

The direct Split channelwise tail now shares that indexed owner while retaining
a separate root plan. It recognizes one exact private NHWC-to-NCHW Transpose
feeding a channel Split directly, then permits the same closed unary, binary,
Concat, and downstream Split closure plus the historically supported Slice
family. The former lowerer implementation is a compatibility dispatcher; its
two ordered production positions and stats key are unchanged, and both calls
supply Session `LayoutState`.

Slice propagation requires an exact three-input, one-output rank-four Slice.
Begin and size must be immutable typed INT32 or INT64 four-element constants,
the declared dtypes must agree, and the original NCHW and converted NHWC Slice
results must both agree with static output metadata and compatible dynamic
signatures. Begin and size are permuted from `[N,C,H,W]` to `[N,H,W,C]` only as
part of the complete tail transaction. Constants used exclusively by planned
Slice input slots may change in place; constants with any unrelated consumer
receive one deterministic shared clone for all planned uses. Conflicting roles,
producer-backed constants, variables, public constants, invalid bounds, dtype
mismatches, and per-axis quantization reject the candidate.

Closure discovery begins only from root Split outputs. The raw Transpose input
is deliberately not treated as converted closure state, so unrelated consumers
of that NHWC source remain untouched. Root and downstream Split axes use the
same copy-on-write INT32/INT64 contract as the binary-root family. Unsupported
closure consumers, dead converted branches, public intermediates, multiple
outputs, duplicate producers, stale order, and shared Transpose outputs reject
the entire plan before mutation.

The common immutable tail plan owns metadata, Split-axis updates, Concat-axis
updates, grouped Slice-constant updates, the accepted closure, and the sole
terminal output adapter. It is resolved again immediately before apply and
preflighted against every input/output slot and allocated name. Differential
graph-index updates, differential LayoutState synchronization, a configurable
32-rewrite ceiling, and success-only pruning preserve the same bounded
transactional contract as the binary-root family.

Pre-extraction characterization observed zero direct-root matches in all
runtime invocations on the same five short representatives. Twenty focused
tests cover INT32 and INT64 constants, static and dynamic signatures, numerical
equivalence, grouped shared-constant cloning, an unrelated source consumer,
candidate limits, idempotence, and thirteen unsafe transactional no-op cases.
The direct and binary-root suites pass together, and a sequential YuNet
comparison against the preceding checkpoint emits byte-identical float32,
float16, correspondence, and schema artifacts.

The unary/Split/Concat compatibility island is the third root family in
`passes/split_channelwise_layout.py`. It matches one exact private
NHWC-to-NCHW Transpose, one layout-preserving unary, a channel Split, every
Split output consumed either directly or through one allowed unary, exactly
one external Concat branch, and one channel-axis Concat. The lowerer retains a
thin compatibility dispatcher at both unchanged sequence positions and passes
Session `LayoutState`.

The root and every branch form a closed local island. The pre-Transpose output
must feed only the pre-Split unary, that unary must feed only the Split, every
Split output must appear exactly once, and an optional branch unary must feed
only the Concat. Duplicate or missing branches, a second external input,
external fan-out from a converted branch, public intermediate tensors,
duplicate producers, and consumer-before-producer order reject the plan. The
allowed unary set remains the exact historical RELU, RELU6, RELU_0_TO_1,
LOGISTIC, HARD_SWISH, LEAKY_RELU, and TANH family.

The external branch is either an already-proven NHWC tensor or the source of a
layout-only singleton Reshape. The bypass form now requires both rank-four
shapes and signatures to have the exact NHWC-to-NCHW relationship, channel
size one in both representations, matching dtype, per-tensor quantization,
resolved provenance, unique production, graph order, and non-NHWC physical
evidence on the Reshape output. This corrects the raw helper's misleading
"singleton" guard, which accepted arbitrary channel counts.

The immutable plan reuses the common typed Split-axis copy-on-write, Concat
shape/signature validation, metadata updates, private NHWC output allocation,
and local NHWC-to-NCHW adapter. Unlike the closed public-tail families, the
adapter may preserve either a graph output or an intermediate legacy NCHW
contract, so existing downstream consumers remain unchanged. The proven input
permutation is reused rather than creating another buffer.

The complete plan is resolved twice and preflighted before the pre-unary or
Concat input changes. This removes the raw partial-mutation path that changed
the Split axis, multiple tensor metadata records, and Concat inputs/options
before discovering a missing or invalid Concat output. One differential graph
index, graph-ordered Concat candidates, a configurable 32-rewrite ceiling,
success-only pruning, and LayoutState synchronization replace the full-map
unbounded loop.

Pre-extraction characterization observed zero matches in all 24 runtime
invocations on YuNet, FastestDet, HumanSeg, OSNet, and SiNet. Twenty-nine
focused tests cover INT32/INT64 axes, static and dynamic signatures, public and
local NCHW boundaries, exact numerical equivalence, a direct NHWC external
branch, shared axes and side consumers, candidate limits, idempotence, and
eighteen unsafe transactional no-op cases. A sequential YuNet comparison
against the preceding checkpoint emits five byte-identical artifacts.

The adjacent singleton gate/Conv/Concat compatibility island is owned by
`passes/singleton_gate_layout.py`. The former lowerer implementation mixed
matching, metadata writes, consumer rewiring, and operator deletion in one
unbounded full-map loop. Its two ordered production positions remain intact as
thin dispatches, and both supply Session `LayoutState` to the new owner.

The resolver accepts the exact historical gate topology: an NHWC clip tensor
viewed through a singleton NCHW Reshape feeds both the gate multiply and the
scalar-minus-clip branch; a second singleton adapter supplies either a direct
auxiliary signal or the input to a Logistic; the two multiplied branches join
through Add and one allowed unary before a singleton output adapter reaches a
channel-last Concat. An optional RGB multiply may consume the same clip through
either a proven `[0,3,1,2]` input Transpose or a private constant whose physical
NHWC evidence shows its NCHW metadata is stale.

Every removed Reshape must represent an exact `[N,H,W,1]` to `[N,1,H,W]` view,
including compatible dynamic signatures, equal dtype, and per-tensor
quantization. Typed shape inputs, when present, must be immutable INT32 or
INT64 constants consistent with the declared output. All core producers,
consumer multiplicities, public boundaries, graph order, unary/binary options,
static broadcasts, and dynamic broadcasts are resolved before a plan exists.
This replaces the raw predicate that accepted arbitrary channel counts and the
raw mutation path that could remove an adapter while leaving an unsupported
fan-out in the old layout.

The plan records every operator input rewrite, tensor contract, operator
contract, metadata update, and removal. It is fully re-resolved immediately
before apply. Split and terminal adapter aliases are rewired only for their
proven view-equivalent consumers; unrelated consumers of the original NHWC
sources remain unchanged. Shared side consumers are preserved. Constant data
with stale layout metadata is reshaped to the validated NHWC view at the same
time as its TensorIR metadata, so physical storage and declared shape cannot
diverge after the rewrite.

One differential `ModelIRGraphIndex`, graph-ordered Concat candidates, a
configurable 32-rewrite ceiling, success-only pruning, and differential
LayoutState updates replace the raw `while True` loop and repeated producer/
consumer map construction. Candidate-only operation and idempotence use the
same entry point as production.

Pre-extraction characterization observed zero matches in all 24 sequential
runtime invocations: four each on YuNet, FastestDet, HumanSeg, and OSNet, and
eight on SiNet. Twenty-nine focused tests cover both auxiliary variants,
static and dynamic signatures, the optional Transpose RGB bridge, the stale
constant RGB view, shared side consumers, exact numerical equivalence,
candidate and rewrite limits, differential-index freshness, LayoutState, and
nineteen unsafe transactional no-op contracts. A sequential YuNet comparison
against the preceding checkpoint emits byte-identical float32, float16,
correspondence, and schema artifacts.

Conv-family output passthrough chains are owned by
`passes/conv_output_passthrough_layout.py`. The owner recognizes a private
NHWC-to-NCHW Transpose produced by Conv2D, DepthwiseConv2D, or TransposeConv,
one or more strictly linear unary/binary operations, and one inverse
NCHW-to-NHWC Transpose whose output remains an intermediate tensor. The former
316-line lowerer helper is a thin dispatcher in its unchanged ordered position
and receives Session `LayoutState`.

Both permutation tensors must be immutable typed INT32/INT64 constants. The
resolver proves producer uniqueness and order, the exact rank-four shape and
dynamic-signature relation across each adapter, private intermediate tensors,
one-consumer linearity, per-tensor quantization, existing output metadata, and
consumer order after the inverse adapter. QUANTIZE, DEQUANTIZE, CAST, RELU,
RELU6, RELU_0_TO_1, and HARD_SWISH retain their unary semantics. ADD, SUB,
MUL, DIV, MAXIMUM, and MINIMUM retain their operand slots, so asymmetric SUB
and DIV are not reordered.

Non-scalar binary side inputs must be private immutable rank-four constants
whose data, TensorIR shape/signature, dtype, original NCHW broadcast, converted
NHWC broadcast, and output contract all agree. Exclusive constants are
transposed in place. A constant with any consumer outside the accepted chain
receives one deterministic NHWC clone shared by all planned slots; the legacy
constant and unrelated consumers remain unchanged. This grouping also avoids
the raw helper's repeated in-place transpose when one constant appears more
than once in a chain.

Every input/output rewrite, constant update, tensor/operator contract,
metadata update, and adapter removal belongs to one immutable plan that is
fully resolved again immediately before apply. The surviving inverse-adapter
output takes the converted final chain shape/signature and dtype/quantization
directly from the former chain output; it is not blindly permuted from possibly
stale existing metadata. One differential `ModelIRGraphIndex`, a candidate
list bounded by the current Transpose count, an optional explicit rewrite
limit, success-only pruning, and differential LayoutState updates replace the
repeated full-map rebuild and unbounded fixed-point loop.

Pre-extraction sequential characterization observed four invocations per
representative. Only the first was active: YuNet rewrote 10 chains, FastestDet
23, HumanSeg 27, OSNet 63, and SiNet zero. All 123 active chains were
Conv2D/DepthwiseConv2D followed by RELU, while the full historical unary,
binary, and TransposeConv capability remains covered synthetically. The
separate channel-one TransposeConv/Squeeze terminal helper matched zero in all
20 invocations and remains a distinct semantic family.

Fifty-six focused tests cover all three producer types, all seven unary types,
all six binary types in both operand positions, exact numerical equivalence,
dynamic signatures, grouped shared-constant cloning, candidate limits,
idempotence, GraphIndex/LayoutState integrity, and twenty unsafe transactional
no-op contracts. Sequential comparisons against the preceding checkpoint emit
byte-identical float32, float16, correspondence, and schema artifacts for all
four active representatives.

## Managed-corpus SWAP exclusion policy

Managed corpus validation remains strictly sequential. While each converter
subprocess is active, the bulk runner samples `VmSwap` for that subprocess and
all of its descendants from Linux `/proc`. Existing host-wide SWAP use and
SWAP use by unrelated processes are deliberately ignored. If any process in
the active model's process tree reports nonzero `VmSwap`, the runner terminates
that process tree, records the model as `swap_detected`, and includes the peak
tree total and per-process peak values in `bulk_status.json` and the generated
summary.

A model reported as `swap_detected` must be changed to
`baseline_classification: excluded` in the managed regression profile before
the next corpus run, with `baseline_reason` set to
`swap_detected_during_managed_validation`. The managed-profile count and exact
exclusion contract tests must be updated in the same checkpoint. Because that
changes profile identity, the next authoritative run starts with a clean
output directory rather than resuming results produced by the older profile.

## Remaining refactoring order

1. Improve Tier 0-4 layout, transpose, broadcast, shape reconciliation, and
   fusion failures using semantic passes and focused ONNX fixtures.
2. Continue moving validation, capability selection, and lowering into
   op-family modules while preserving the current public API and artifacts.
3. Complete quantization, split/crop, custom/pseudo op, report, and requested-
   artifact-only regression coverage on the validated ModelIR contract.
4. Complete the shared PyTorch/TorchScript/Dynamo ONNX/ExportedProgram
   canonicalization and artifact-emitter separation on top of the extracted
   native op and encoder-stage codegen boundaries.
5. Measure warm-run conversion time and peak RSS on the active Tier 0-4 set,
   then document improvements and remaining normalized failures.
