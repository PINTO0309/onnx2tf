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

The first Q/DQ transpose-bridge step in that prefix is owned by
`passes/transpose_qdq_bridge_layout.py`. It remains separate from terminal
exact-grid, Concat-input, Mean, activation, PReLU, Reshape, and TransposeConv
quantization cleanup owners. Its one lowerer compatibility wrapper is called
once by the ordered prefix, which expands at four unchanged runtime positions.
The owner keeps one fixed-point loop and the historical A→B→C→D priority:
complete Transpose/Q/DQ round trips, single Q-or-DQ bridges with fan-out,
two-branch QDQ/Add residual closure, and mixed float/QDQ Add residual closure.
All branch/public-output/per-tensor-grid/permutation/fan-out guards, metadata
and quantization cloning, direct mutation order, prune boundary, and three
stats keys remain unchanged. Direct tests cover A and B rewrites, guard no-ops,
idempotence, and wrapper equality; existing end-to-end fixtures retain positive
A/B/C/D and legacy-fan-out coverage. The extraction is mechanical and does not
claim differential-index performance; indexed mutation requires a later
family-by-family equivalence checkpoint.

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
HardSigmoid/Quantize bridge rewrite and ends before the independently indexed
Swish dispatcher. The Swish semantic owner maintains its own differential
index, so pass state is not shared across that phase boundary. The three
runners retain their exact order and diagnostics while constructing one graph
index instead of up to three.

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

The remaining generic NHWC pre-Concat compatibility matcher is now isolated in
`passes/nhwc_concat_legacy_layout.py`. Its complete 2,452-line implementation
moved as one semantic unit; this is not a source-line limit or a claim that the
fallback is modernized. Function-name-normalized AST comparison with the prior
lowerer owner is exact. The nested direct, unary, expanded-Swish,
pseudo-LeakyRelu, Pad, Dequantize, PReLU, Softmax, Slice, Split, and recursive
Add analysis/application helpers therefore retain their shared fixed-point
loop, action precedence, indexed-family exclusion contracts, constant
copy-on-write behavior, metadata propagation, mutation order, pruning, and
single historical statistic. The central lowerer keeps the private legacy
symbol as a one-call compatibility wrapper, while the indexed float,
quantized-indexed, and legacy dispatch order and all four composite production
positions remain unchanged.

The exact pseudo-LeakyRelu plus Pad-companion fixture establishes positive
legacy ownership, idempotence, and direct-owner/private-wrapper equality. The
complete float and quantized Concat family corpus continues to cover every
indexed/fallback ownership boundary. FastestDet and OSNet supplied fourteen
measured zero-owner runtime invocations before extraction; no production
legacy rewrite is claimed. FastestDet is the artifact control and retains its
accuracy, zero process-tree SWAP, and byte-identical float32, float16,
correspondence, schema, and generated-schema outputs across the move.

The adjacent strict rank-four Transpose→Slice→inverse-Transpose compatibility
rule is isolated in `passes/slice_prepost_layout.py`. Its complete matcher moved
with a function-name-normalized AST identical to the prior lowerer owner. It
still requires constant exclusive begin/size tensors, exclusive pre/Slice
links, non-public intermediates, exact inverse permutations, and rank-four
input/output metadata. The existing static-shape owner supplies both as-is and
NCHW→NHWC-remapped parameter validation; only the parameter set reproducing the
known public output shape is committed. Input/output rewrites, two-Transpose
removal, conditional pruning, fixed-point restart, and the historical statistic
are unchanged. The lowerer retains one private compatibility wrapper at its
single production position.

A compact synthetic corpus fixes both already-NHWC and remap-required
parameter forms, idempotence, public pre/Slice outputs, shared begin constants,
pre-Transpose fan-out, output-shape mismatch, wrong permutation, and direct-
owner/private-wrapper equality. Tier 0 UM Best Model and Tier 2 ALike supplied
two measured zero-owner production calls, so no non-zero real-model ownership
is claimed. UM Best Model remains accurate with zero process-tree SWAP and
byte-identical direct artifacts across the mechanical move.

The following NHWC-to-NCHW Shape-extraction compatibility rule is isolated in
`passes/shape_extract_layout.py`. Its complete 285-line implementation moved
with a function-name-normalized AST identical to the prior lowerer owner. It
remaps Gather indices from logical NCHW axes to the physical NHWC Shape vector,
remaps contiguous Slice selections, and converts non-contiguous Slice
selections to Gather. Shared constants use clone-on-write while retaining dtype
and quantization metadata. All Shape consumers must belong to the supported
families, public boundary and fan-out guards remain strict, and fixed-point
restart, conditional pruning, and the historical statistic are unchanged.
The lowerer retains one one-call private compatibility wrapper at all three
unchanged production positions.

Focused fixtures cover exclusive and shared Gather indices, contiguous Slice
remapping, non-contiguous Slice-to-Gather conversion, idempotence, every public,
fan-out, axis, constant, index, and empty-selection guard, and direct-owner/
private-wrapper equality. RetinaFace Dynamic establishes non-zero production
ownership with counts `1, 0, 0`; its accuracy, zero process-tree SWAP, and all
five core artifacts are unchanged across extraction. ALike supplies the
`0, 0, 0` zero-owner control and remains accurate with zero SWAP.

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

Unquantized pseudo-Swish transpose passthrough is independently owned by
`passes/activation_passthrough_layout.py`. The former 194-line raw helper is a thin
dispatcher at its two unchanged production positions: once in the ordered
activation recovery prefix and once in the no-layout-compatible late recovery.
Both calls receive Session `LayoutState`. This owner is separate from the
quantized Swish-QDQ phases because it recognizes the exact float or per-tensor
`Logistic(x) * x` residual topology rather than Dequantize/Quantize closure.

The resolver accepts a typed immutable INT32/INT64 permutation of rank two or
higher, an exact source-to-transposed tensor view, one Logistic consumer, one
residual Mul consumer in either operand order, and at least one typed inverse
post adapter. Static shapes, independently dynamic signatures, dtype,
per-tensor quantization, layout transition, provenance, unique production,
consumer slots, graph order, public aliases, and every post-adapter output are
proven before planning. Immutable operator-produced and constant sources are
both supported; the source data is never transposed or mutated because Swish
is elementwise and commutes with the boundary permutation.

All post aliases collapse to one source-layout Mul output. Exact consumer-slot
grouping preserves repeated inputs and selects a public post alias when one is
present. If the old transposed Mul result still has a legacy consumer or is a
graph output, one local adapter is inserted immediately after Mul. It reuses
the proven pre-permutation buffer. The old helper instead overwrote a selected
post-permutation buffer in place, which could corrupt an unrelated user when
that constant was shared.

The immutable plan records both head rewrites, every alias rewrite, metadata,
public lists, adapter removals, and complete tensor/operator contracts. A full
second resolution and preflight precede mutation. One differential graph index
updates slots, compacts the pre/post adapters, changes the Mul output, and
inserts the optional legacy adapter. LayoutState is updated only for surviving
source-layout tensors and pruning occurs only after success. Graph-ordered
Transpose candidates and an optional candidate-count limit replace the raw
full-map unbounded fixed-point loop.

Fifty-six focused cases cover rank-three/rank-four static and dynamic views,
INT32/INT64 permutations, both Mul operand orders, one/multiple posts, repeated
alias slots, a public post, immutable constant source, shared post permutation,
legacy and public transposed boundaries, exact numerical equivalence,
candidate limits, idempotence, GraphIndex, LayoutState, and twenty-seven
transactional rejection cases. The focused owner, adjacent input and
quantized-Swish owners, active Swish fixtures, and complete architecture suite
pass together with `304 passed in 44.59s`. TensorFlow-blocked
direct/default/`-cotof` checks pass sequentially, and YuNet reproduces all five
fixed artifact hashes.

The adjacent tanh-expanded GELU transpose passthrough is owned by the same
activation module under an independent immutable plan. Its production position
remains immediately after pseudo-Swish and before center/size-offset recovery.
The compatibility wrapper retains its legacy name and statistic, receives
Session `LayoutState`, and delegates only to the indexed owner.

The resolver proves the complete nine-operation approximation:
`x*x`, cubic multiplication, a cubic-scale Mul, residual Add, an outer-scale
Mul, Tanh, an offset Add, residual multiplication by `x`, and a final-scale
Mul. It validates all five direct source-consumer slots,
linear ownership of every other intermediate, unique producers, strict graph
order, and four immutable singleton constants before accepting the chain.
Constants retain their declared dtype, data, shape, provenance, and per-tensor
quantization. Both binary operand orders are supported where multiplication or
addition is commutative.

Typed rank-two-or-higher INT32/INT64 boundary permutations, exact static or
dynamic tensor views, dtype, quantization, and known layout transitions follow
the same contract as pseudo-Swish. All inverse post aliases collapse to one
source-layout final result. A public post alias can be selected as the
representative, while an existing consumer or public use of the old transposed
final tensor is preserved by one local adapter immediately after the last Mul.
The adapter reuses the proven pre-permutation, and no shared post-permutation
buffer is modified.

The complete plan is re-resolved before apply. One differential graph index
rewires exact input slots, changes the final output, inserts the optional local
adapter, compacts only proven private adapters, and updates LayoutState only
after acceptance. Candidate enumeration is graph ordered and bounded; the raw
full-map fixed-point loop and mutation-before-validation path are gone.

Fifty-six focused GELU cases cover static/dynamic rank-three and rank-four
views, INT32/INT64 permutations, binary operand order, one/multiple post
aliases, repeated slots, public and legacy boundaries, constant input, shared
permutation buffers, numerical equivalence, candidate limits, idempotence, and
thirty transactional rejection variants. The Swish and GELU owners, adjacent
layout suites, active fixtures, and complete architecture suite pass together
with `362 passed in 42.95s`. TensorFlow-blocked direct/default/`-cotof` checks
pass sequentially with `3 passed in 3.95s`, and YuNet again reproduces all five
fixed artifact hashes.

Center/size/offset terminal-head recovery is owned by
`passes/center_size_offset_layout.py`. Its former roughly 390-line raw helper
is a thin compatibility dispatcher at the same position between tanh-GELU and
LeakyReLU. The call receives Session `LayoutState`; its public symbol and
legacy statistic remain unchanged.

The owner classifies three independent typed NHWC-to-NCHW Transpose roots by
their semantic branch contracts rather than searching the following sixteen
operators. The center root must have singleton channel and close through
Logistic/Maximum/Minimum into a valid `[N,H*W]` Reshape. The size root uses the
same activation prefix and a valid `[N,C,H*W]` Reshape/GatherND tail. The offset
root closes directly through the equivalent Reshape/GatherND tail. Static
shape, independently dynamic signature, dtype, per-tensor quantization,
producer ownership, exact consumer slots, graph order, provenance, public
boundaries, and known logical/physical layouts are proven for every branch.
Ambiguous triples are rejected rather than selected by proximity.

Each GatherND coordinate Concat is independently classified as batch, dynamic
axis, and channel coordinates. The axis coordinate must be the exact output of
a typed Reshape; the two immutable integer grids must be in range for the
proven batch and channel dimensions. INT32 and INT64 coordinates, arbitrary
original Concat order, a shared coordinate Concat, and equal-valued batch and
channel grids are supported. The Concat output is closed over exactly the
planned GatherND input slots before its inputs change from
`[batch, channel, axis]` to `[batch, axis, channel]`.

Size and offset Reshape literals rotate from `[N,C,HW]` to `[N,HW,C]` while
preserving inferred `-1` dimensions and their option metadata. Shape constants
are grouped by identity and exact input slot. An exclusive constant changes in
place; a shared constant receives one deterministic typed copy-on-write clone,
leaving unrelated consumers on the original value. The six activation
intermediates move to the proven NHWC view, the two rank-three Reshape outputs
become NWC, and LayoutState changes only with the accepted transaction.

The immutable plan captures every input rewrite, constant use, option change,
metadata update, removal, tensor/operator contract, and public list. It is
fully re-resolved before apply. One differential graph index rewires slots and
compacts all three adapters; graph-ordered candidates and an optional rewrite
limit replace the raw fixed-point loop. Pruning occurs only after success.

Thirty-seven focused cases cover static/dynamic views, INT32/INT64 shape and
coordinate buffers, binary operand order, per-tensor quantization, inferred
dimensions, shared constants and coordinates, numerical equivalence,
GraphIndex, LayoutState, candidate limits, idempotence, and twenty-two unsafe
transactional no-op variants. Pre/post characterization on five short models
produces twenty zero-match invocations with unchanged operator/tensor counts.
The owner, adjacent activation/layout suites, active fixtures, and complete
architecture suite pass together with `399 passed in 42.94s`. TensorFlow-
blocked direct/default/`-cotof` checks pass sequentially with
`3 passed in 3.99s`, and YuNet reproduces all five fixed artifact hashes.

Pseudo-expanded LeakyReLU transpose passthrough is owned by
`passes/leakyrelu_passthrough_layout.py`. Its former roughly 240-line raw helper
is a thin dispatcher at the unchanged position between center/size/offset and
PReLU. The composite owner preserves the historical second phase: after all
accepted layout passthroughs, it invokes the existing indexed pseudo-LeakyReLU
fusion using the same differential graph index and Session LayoutState. Both
legacy statistic keys and their execution order remain unchanged.

The passthrough resolver proves the exact
`Neg(x) → Relu → Mul(alpha)` negative branch, the direct `Relu(x)` positive
branch, and ordered `Sub(positive, scaled-negative)` join. Both source slots,
every private intermediate, the immutable singleton alpha, unique production,
exact consumer slots, graph order, shape/dynamic signature, dtype, per-tensor
quantization, provenance, layout transition, and public boundary are resolved
before planning. Rank-three and rank-four typed INT32/INT64 permutations,
either Mul constant position, and immutable constant sources are supported.

All inverse post-Transpose aliases collapse onto one source-layout result. A
public post alias is preferred as the representative, and repeated downstream
alias slots are grouped exactly. When the old transposed Sub result remains a
graph output or has a legacy consumer, one local adapter is inserted
immediately after the retained join and reuses the immutable pre-permutation.
The raw helper instead overwrote the selected post-permutation buffer with an
INT32 copy of the opposite permutation, which could change dtype and corrupt an
unrelated consumer.

The immutable plan records every input/output slot, metadata update, removal,
optional adapter, tensor/operator contract, and public list, then performs a
full second resolution and preflight. One differential graph index rewires the
two heads, aliases, Sub output, adapter insertion, and pre/post compaction.
LayoutState updates only on acceptance and pruning is success-only. The
following fusion converts the retained Sub to native `LEAKY_RELU` and compacts
the four private producers through that same index.

Fifty-one focused cases cover static/dynamic rank-three and rank-four views,
INT32/INT64 permutations, both alpha positions, per-tensor quantization,
constant input, one/multiple/public post aliases, repeated slots, legacy/public
transposed boundaries, shared post permutations, numerical equivalence,
candidate limits, idempotence, GraphIndex, LayoutState, and twenty-seven unsafe
passthrough no-op variants. Pre/post characterization on five short models
produces twenty zero-rewrite/zero-fusion invocations with unchanged counts.
The owner, existing fusion owner, adjacent activation/layout suites, active
fixtures, and complete architecture suite pass together with
`467 passed in 42.33s`. TensorFlow-blocked direct/default/`-cotof` checks pass
sequentially with `3 passed in 4.01s`, and YuNet reproduces all five fixed
artifact hashes.

PReLU transpose passthrough is owned by
`passes/prelu_passthrough_layout.py`. Its former roughly 250-line lowerer
helper is a thin compatibility dispatcher at all three unchanged production
positions, each receiving Session LayoutState. The statistic key and the
historical prune-on-zero-match behavior are unchanged.

The resolver proves a typed rank-three-or-higher source Transpose, one PReLU
data edge, one or more typed inverse post adapters, exact producer/consumer
slots and graph order, static shape and independent dynamic signature, dtype,
per-tensor quantization, provenance, public boundaries, and known logical and
physical layout transitions. Unrelated users of the pre-Transpose result keep
that adapter. Post aliases are grouped by exact downstream input slot and one
public or consumed alias becomes the source-layout representative.

Alpha selection preserves the former priority: inverse-layout remap for an
equal-rank parameter, the original value, then the special rank-three channel
form. Every candidate must broadcast to the concrete source shape and its
dynamic signature. Exclusive parameters change in place; shared parameters
receive one deterministic `_nhwc` copy with the original declared/NumPy dtype,
quantization, layout, and ONNX tensor provenance. Scalar, rank-three,
rank-four, and ambiguous equal-shape forms are supported.

When the old transposed PReLU tensor remains observable, one local adapter is
retained. An exclusively owned post permutation and its existing adapter are
reused, preserving historical correspondence lineage. A shared post
permutation is never changed; the adapter uses the proven pre permutation
instead. INT32 and INT64 buffers retain their dtype. This closes the former
path that overwrote a shared post buffer with an INT32 opposite permutation.

The immutable plan captures the alpha update, every input/output rewrite,
metadata update, removal, tensor/operator contract, and public list, then is
fully re-resolved before apply. One differential graph index performs all
rewiring and compaction. LayoutState changes only after acceptance. Bounded
graph-order candidates and an optional rewrite limit replace the raw
fixed-point loop.

Twenty-eight focused cases cover typed static/dynamic rank-three and rank-four
views, alpha remap/reuse/copy-on-write, scalar and ambiguous parameters,
pre-adapter fan-out, multiple/public/legacy aliases, repeated slots, shared
post permutations, numerical equivalence, limits, idempotence, GraphIndex,
LayoutState, and fifteen transactional no-op variants. Five representative
models reach the helper six times each. YuNet, FastestDet, HumanSeg, and OSNet
remain zero-match; FastestDet retains its zero-match unused-tensor prune; and
SiNet retains two 23-rewrite invocations. Its five artifacts are byte-identical
to the preceding checkpoint. Adjacent owners, active fixtures, and the full
architecture suite pass with `454 passed in 50.22s`; TensorFlow-blocked
direct/default/`-cotof` checks pass sequentially with `3 passed in 4.38s`, and
YuNet reproduces all five fixed artifact hashes.

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

The fallback's `replaced_unsupported_split_with_slice` result is complete
mutation evidence for this owner. Tensor pruning and `LayoutState` syncing are
both nested under a positive rewrite count; a zero result changes neither the
ModelIR nor layout metadata. The direct lowerer therefore retains its existing
positive guard around the immediately following static-shape reconciliation.
The lowerer initializes a stable two-key zero result and, on a positive Split
rewrite, replaces it with the reconciliation's complete opt-in result. This
stages mutation evidence instead of discarding it without changing the guard,
Split result schema, or pass order.

The safety-fallback norm-only Pad cleanup has a different mutation contract.
Its raw rewrite counter is incomplete because the child owner prunes unused
tensors even after a zero rewrite. The fallback call therefore adds a clamped
before/after tensor-count delta while retaining the existing norm-rewrite guard
and recursive relowering order. The extra evidence remains observation-only;
cleanup alone does not trigger shape reconciliation.

The next fallback-only dynamic rank-one Unsqueeze/Reshape-shape owner exposes
one complete rewrite counter and performs no cleanup when it is zero.
Its existing result is assigned independently of the norm summary. The
immediately following topological sort and logical-layout inference remain
unconditional until the recursive return state and all prior fallback
mutations have a stronger equivalence proof.

The following fallback rank-four broadcast-constant repair has a complete
counter and no cleanup-only path. Its reconciliation is already guarded by a
positive rewrite and followed by topological/layout refreshes in the same
branch. A stable zero result and assigned opt-in complete reconciliation
evidence replace the discarded return value; the guard and both refreshes
remain unchanged.

The fallback SINet-shuffle plus SE/FC/Gather boundary already combines three
rewrite counters with a before/after tensor-count predicate, so cleanup-only
pruning remains visible. A stable zero static-shape result and an assigned
opt-in complete result replace the discarded return inside the unchanged
combined guard. No owner, predicate term, or pass order is altered.

The fallback placeholder-MatMul restore has one complete counter: its tensor
pruning executes only after a positive restore. The inline owner call is
separated from its guard, and both the owner result and complete reconciliation
result are staged while preserving the single invocation and following
topological sort.

The unbound-input compatibility wrapper owns static reconciliation after a
positive indexed repair and reuses the repair's live GraphIndex. The fallback
caller no longer repeats the same reconciliation immediately. The wrapper's
one-call contract remains intact, and only the duplicate caller-side full-graph
scan is removed.

The fallback Conv-input runner has two rewrite counters, but both indexed child
owners prune unused tensors even on zero. A clamped tensor-count delta plus a
stable complete reconciliation result now preserve that cleanup evidence. The
existing stale-Transpose-only reconciliation predicate remains unchanged;
singleton repair updates its output metadata directly, and pruning alone does
not require shape propagation.

The fallback mixed-NHWC-input repair for NCHW Concat has one complete rewrite
counter and no cleanup-only path. A stable complete zero result and assigned
opt-in reconciliation evidence replace the discarded result under the
unchanged positive guard, while the following Concat-axis repair boundary
remains fixed.

The fallback NCHW Concat/Transpose/Conv-axis repair also has one complete
counter and no cleanup-only path. The same stable zero and opt-in complete
reconciliation result replace the discarded return under its unchanged guard,
with the following stale binary-layout repair fixed as the outer boundary.

The stale channelwise binary-layout owner unconditionally prunes unused
tensors, including zero-rewrite calls. The fallback branch therefore records a
clamped tensor-count delta and initializes a stable complete reconciliation
result. The existing repair-only guard assigns the opt-in reconciliation
result, while the following topological sort remains unchanged; cleanup alone
does not require shape propagation.

The recursive safety fallback computes layout-validation metadata from its
terminal graph, after high-rank BatchMatMul compression, indexed binary
convergence, and the final topological sort. The validator is pure. A non-empty
result keeps the established error-list schema, while an empty terminal result
removes validation errors inherited from the recursive lower. This changes
diagnostics only; the mutation owners and their ordering remain fixed.

Gate-layout orchestration remains owned by
`passes/gate_layout_orchestration.py`. The lowerer re-exports the elementwise
gate runner from that module solely for the established Python/test import
contract; it does not call the runner directly or create another pass owner.

The static high-rank BatchMatMul owner prunes and synchronizes layout state
only after a successful rewrite, so its exact rewrite counter is complete
mutation evidence and a zero result is a true no-op. The fallback caller keeps
a stable complete reconciliation result and assigns the opt-in result under
the existing positive guard; its sort and following indexed binary-convergence
boundary remain fixed.

Indexed binary-layout convergence already aggregates the complete broadcast,
stale-Transpose, and static-shape mutation counts for up to three rounds and
stops after a stable round. Its fallback caller requires no additional
reconciliation and retains that complete result before the unchanged terminal
sort and layout validation.

The primary path validates layout annotations after indexed binary
convergence, high-rank binary coalescing, dynamic-boundary signature
realignment, and its final sort. The pure validator preserves the established
error-list schema and removes a stale validation-error key when the terminal
graph is valid. Progress reporting and all mutation owners remain outside this
diagnostic-only change.

All three terminal mutation owners already return complete evidence: indexed
binary convergence aggregates its bounded-round counters, high-rank binary
coalescing returns an exact rewrite count, and boundary-signature realignment
returns its update count. The primary caller retains those dictionaries without
adding reconciliation, scans, or changing the final sort/validation boundary.

The earlier final high-rank BatchMatMul owner is the same counter-complete
implementation used by the safety fallback. The primary guard retains the same
stable, opt-in complete static-shape result without changing its sort or the
following channel-first Pad repair boundary.

The final channel-first Pad repair mutates and synchronizes layout state only
when its exact adapter-insertion counter increments. Its zero path performs no
cleanup. The caller retains stable complete shape evidence under the existing
guard, with its sort and following Conv-input repair boundary fixed.

The standalone final stale Conv-input owner always prunes unused tensors, even
when its rewrite counter is zero. The caller therefore records a clamped
tensor-count delta plus a stable complete shape result. Cleanup-only evidence
does not broaden the existing rewrite-only guard, and the following mixed-
Concat boundary remains fixed.

The final mixed-NHWC-input repair for NCHW Concat is the same counter-complete
owner used by the safety fallback. The primary guard retains its stable
complete shape result without changing the guarded sort or following Concat-
axis owner boundary.

The final Concat-axis owner is counter-complete and has no cleanup-only path;
the adjacent stale-binary owner always prunes unused tensors. The caller keeps
a stable Concat-axis shape result followed by stale-binary tensor-delta
accounting and its stable shape result. Both rewrite-only guards, sorts, and
the progress boundary remain fixed.

The final SiNet Concat/Resize affine owner prunes and synchronizes layout state
only after a successful transactional rewrite. Its counter-complete primary
guard retains a stable complete shape result without changing the following
high-rank BatchMatMul boundary.

The five preceding final SiNet owners—late residual, pre-add fan-out, dual
Resize, shared-post fan-out, and deep-skip—share the same transactional
positive-only prune/layout-sync contract. Their stable complete shape results
preserve every guard and the exact owner order.

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

The raw 419-line `_optimize_nhwc_prefix_qlinear_silu_chains` compatibility
owner remains in the lowerer while its mutation contract is stabilized.
Synthetic characterization covers the direct LOGISTIC and decomposed
HardSigmoid paths, multiple fixed-point matches, post-Transpose removal,
legacy-consumer adapter insertion, and eight guard families. The production
QLinear recovery sequence and its two call boundaries are unchanged. Existing
sequential corpus instrumentation remains authoritative: all measured owners
returned zero rewrites with zero process-tree SWAP, so this checkpoint adds no
duplicate conversion.

Four strict xfails define its current unsafe boundary. The helper eagerly
creates the fixed-name NHWC-to-NCHW permutation tensor even when no legacy
adapter is committed; pruning then changes lineage metadata on rejected and
second no-rewrite calls. It also reuses a colliding tensor without validating
the payload and accepts a malformed Mul output signature before rewiring and
rank-four adapter insertion. The correction must prevalidate rank-four output
shape/signature and construct an immutable adapter plan before any edge or
metadata mutation. A permutation constant is allocated only when that plan
needs one, and an occupied name is reused only when its type, shape, and
payload are exact; otherwise a collision-safe name is required.

The corrected 509-line raw owner now resolves every rank-four metadata target
and effective signature before mutation, plans all adapter tensors/operators
and cumulative input updates, and commits only a valid plan. The internal
permutation is allocated lazily and an existing reserved name is reused only
for an exact immutable INT32 `[0,3,1,2]` constant. Pruning is conditional on a
non-zero rewrite, so rejected and repeated zero-rewrite calls preserve both
the graph and lineage metadata. The four characterization xfails are green.
Architecture checks keep metadata/signature/adapter planning before the first
tensor or edge mutation and keep the prune guard tied to the rewrite count.

One strict xfail now isolates a different legacy-consumer duplication. The
consumer map emits one operator index per matching input slot; iterating that
list and then enumerating all matching slots plans the same slots twice. A
two-slot ADD therefore receives four Transpose adapters. The next correction
must deduplicate consumer operator indices in first-observed order without
changing distinct-consumer order, adapter naming, or single-slot behavior.

The corrected owner now deduplicates final-Mul consumer operator indices with
an insertion-ordered set before classifying Transpose and legacy consumers.
Each distinct operator is visited once, then all matching slots in that
operator are planned once. Same-consumer two-slot and two-distinct-consumer
fixtures prove exactly two adapters, stable first-observed naming, and stable
consumer order. The former strict xfail is green; the corrected raw owner is
513 lines and has no remaining expected failure. Architecture coverage keeps
the ordered deduplication assignment before the legacy-consumer loop.

The corrected implementation is now owned by
`passes/qlinear_silu_prefix_layout.py`. Its 513-line function AST is identical
to the corrected lowerer predecessor at checkpoint `0cf699fd`; the lowerer
keeps a one-return private compatibility wrapper. The ordered
`_run_qlinear_mean_concat_recovery_sequence` continues to call that private
name in the same position, and its two production boundaries are unchanged.
Direct owner/wrapper characterization compares statistics, complete ModelIR
fingerprints, layout state, and metadata for LOGISTIC, decomposed HardSigmoid,
legacy-adapter, and reserved-name collision cases. The module cannot import the
lowerer, while architecture checks keep all transactional and deduplication
invariants on the module owner.

The adjacent raw 310-line
`_optimize_transpose_mean_maxpool_concat_conv_chains` owner remains in the
lowerer while its mutation contract is characterized. Positive fixtures cover
static and dynamic signatures, Mean-axis remapping, per-axis QDIM remapping,
multiple post adapters, multiple fixed-point matches, pruning, and idempotence;
ten rejection families preserve the established topology and boundary guards.
The ordered QLinear recovery sequence and both production boundaries remain
unchanged, and earlier sequential instrumentation remains authoritative with
zero rewrites and zero process-tree SWAP.

Nine strict xfails define the unsafe planning boundary. The owner rewires the
mean branch, writes the axes constant, changes Mean metadata, and rewires
Concat before validating every rank-four signature and every additional Concat
input. It also mutates axes owned by another consumer, graph input/output, or a
variable tensor. A correction must require a local immutable axes tensor and
resolve every source/target tensor, effective signature, planned Concat input,
output shape/signature, axis option, and QDIM update before the first ModelIR
mutation. Rejection must preserve graph and diagnostic metadata completely.

The corrected raw owner is 382 lines and now performs that complete planning.
It accepts only a local immutable INT32 axes tensor with exact INT32 backing
data, resolves rank-four effective metadata for every involved and planned
tensor, computes Mean/Concat/QDIM/alias/removal changes, and then enters a
mutation-only commit block with no later rejection. Pruning is guarded by the
non-zero rewrite count. All nine former xfails plus axes TensorIR dtype, buffer
dtype, and quantization guards are green. The unused producer-map construction
was removed, leaving one consumer map per fixed-point round and reducing the
central lowerer's existing Ruff findings to seven. Architecture tests keep
axes ownership and all plan objects before the first setter/constant/alias
mutation and keep conditional pruning after convergence.

The corrected owner is now in `passes/mean_maxpool_concat_layout.py`. Its
382-line function AST is identical to the corrected lowerer predecessor at
checkpoint `7b0f08a9`; the lowerer keeps a one-return private wrapper. The
ordered QLinear recovery sequence and its two production boundaries remain
unchanged. Direct owner/wrapper tests compare the complete ModelIR and
statistics for static, dynamic, multiple-post, multiple-chain, and rejection
contracts. The module cannot import the lowerer, and architecture checks apply
all ownership/planning/prune invariants to the module owner. Removing the
extracted owner's unused lowerer import leaves seven central Ruff findings.

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

The final static SQUEEZE-axis guard is owned by
`passes/squeeze_shape_sanitization.py`. The lowerer retains only the historical
private wrapper and its one terminal production call. The owner normalizes
negative, duplicate, and out-of-range axes, repairs a non-constant static input
dimension only when a constant payload does not disprove singleton extent,
and reconciles the SQUEEZE output shape and signature through the shared static
shape inference helpers. It intentionally performs one operator-list traversal
instead of constructing a fresh graph index for its single terminal call; it
does not query producers or consumers and does not change topology. Dedicated
tests preserve wrapper equivalence, constant-payload authority, metadata
repair, counters, idempotence, and the no-import-cycle boundary.

Final static runtime-shape/signature consistency is owned by
`passes/static_shape_signature_sanitization.py`. The lowerer retains the
historical private wrapper at both late production positions. The owner builds
one producer map, establishes dynamic-lineage roots from ONNX boundary
metadata and runtime-dependent WHERE, RANGE, RESHAPE, and TOPK_V2 outputs, and
uses a memoized cycle-safe ancestry walk to distinguish dynamic contracts from
stale internal signatures. Fully static internal tensors are completed or
repaired; boundary-map signatures, leading dynamic graph-output axes, and
dynamic descendants remain dynamic. Constant payloads terminate lineage. The
pass changes metadata only and deliberately does not mutate topology or
LayoutState. Focused tests cover each root family, boundary normalization,
recursive and cyclic lineage, constant termination, scalar/missing/rank/stale
repairs, idempotence, and compatibility-wrapper equality. Architecture tests
fix one module owner, one producer-map build, four stats keys, and two
production calls.

The companion dynamic-boundary map realigner is owned by the same module. It
keeps `_align_boundary_signature_to_current_shape` in `core/onnx_analysis.py`,
where the primitive is also used while constructing boundary metadata, and
moves only the ModelIR map traversal/update policy out of the lowerer. All
three late production positions call a thin compatibility wrapper. The owner
skips malformed maps, non-list entries, missing tensors, missing shapes, empty
signatures, and rank mismatches; unchanged aligned signatures are idempotent.
Repeated static extents retain the core helper's deterministic first-axis
assignment. Focused tests fix same-axis and layout-moved signatures, repeated
and insufficient static extents, malformed inputs, idempotence, and wrapper
equality. The rule changes metadata only and never visits operators.

The terminal LiteRT.js compatibility rewrite for ExpandDims and Squeeze is
owned by `passes/expand_squeeze_reshape.py`; the lowerer retains one private
compatibility wrapper at the unchanged production boundary. The owner keeps
static target construction, speculative inactive-If Squeeze handling, dynamic
SHAPE/GATHER target construction, semantic-axis metadata, pruning, and
LayoutState synchronization together. Dynamic pre-operators are collected by
original operator index and inserted in reverse index/reverse local order
through `ModelIRGraphIndex.insert_operator()`, preserving the intended runtime
SHAPE then GATHER then RESHAPE order without replacing the operator list. A
graph index is constructed only when dynamic pre-operators exist. Direct owner
tests fix wrapper equality, exact operator order, kept-axis data, LayoutState
validity, and idempotence; existing shape tests preserve all static, dynamic,
speculative-branch, and no-op guards.

Exact rank-four binary layout mismatch adaptation is owned by
`passes/binary_layout_adapter.py`. It remains distinct from indexed stale-
adapter removal and transpose-bridge reduction: this compatibility guard
inserts a missing input-1 Transpose only when the two full static shapes are
exact NHWC/NCHW permutations. Its four historical production positions call
one lowerer wrapper. The owner retains the bounded binary-op set, permutation
precedence, quantization cloning, fixed-point restart, and terminal prune.
Synthetic tests cover both directions for every supported binary operator,
dynamic/equal/non-permutation no-ops, idempotence, and wrapper equality. The
current corpus measurements are zero-owner evidence; differential indexing or
broader inference is not combined with the mechanical ownership move.

The same op-family module separately owns singleton-channel rank-four binary
adaptation. This rule is intentionally not folded into the exact full-rank
adapter: it uses the declared output shape to select one of four branches.
For an NCHW output it transposes the NHWC operand to NCHW; for an NHWC output
it reshapes the singleton NCHW operand to `[N,H,W,1]`. Either input position is
supported for ADD/MUL/SUB/DIV/MAXIMUM/MINIMUM. A NumPy broadcast check rejects
pairs that already produce the declared output without an adapter. The owner
preserves quantization cloning, fixed-point restart, and conditional pruning,
while the lowerer keeps only the historical wrapper at all four production
positions. Focused tests cover all four branches for every supported operator,
the broadcast no-op, idempotence, and wrapper equality. Architecture tests
keep the two binary policies as distinct public module owners and prevent
implementation from returning to the central lowerer.

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

Its focused extraction audit confirms that a short source shape in the Conv-
peer evidence branch is a complete no-op. The source `shape_signature` is now
materialized and required to be rank four before the indexed setter. A
malformed signature therefore retains a complete ModelIR fingerprint and zero
statistic instead of being assigned to a rank-four output after input rewiring.
Architecture tests keep signature materialization before the first mutation.

The channelwise-constant evidence branch now requires rank-four source and
adapter shapes before its `[1,1,1,C]` channel checks or either `[3]` read. Short
source or adapter metadata therefore returns a complete zero-statistic no-op.
Architecture tests keep the shared rank guard before both channelwise-constant
and Conv-peer evidence assignments. No strict xfail remains.

The corrected 132-line implementation is owned by
`passes/stale_binary_adapter_repair.py`; its AST is identical to the corrected
lowerer predecessor at checkpoint `c869c410`. The lowerer keeps only its
private compatibility wrapper and forwards an optional caller-owned index to
the module owner. Both standalone fallback/final production calls are
unchanged. A direct owner-versus-wrapper characterization compares return
statistics and the complete resulting ModelIR fingerprint on a multi-adapter
graph. The owner module cannot import the lowerer, and architecture checks
keep its rank/signature validation before mutation.

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
front of NHWC Conv inputs are now owned by
`passes/conv_input_adapter_repair.py`. Their shared runner builds one
`ModelIRGraphIndex`. Both enumerate only indexed `CONV_2D` candidates, obtain
the adapter producer and exact consumer list from the index, rewrite the Conv
data input through the indexed setter, and remove an accepted adapter through
differential compaction. The lowerer keeps private compatibility wrappers for
both repairs and the runner. Primary and fallback execute the runner; the later
standalone stale-Transpose cleanup remains outside that ownership boundary and
builds its own compatibility index. Exact singleton shape, Transpose
permutation, filter input-channel, single-consumer, and graph-output guards
remain unchanged. The 104-, 122-, and 23-line owner bodies are AST-identical to
the corrected lowerer predecessors. Characterization compares the complete
resulting ModelIR with the former explicit pair, proves one index build without
legacy producer/consumer maps, exercises multiple matches, preserves fan-out
and graph-output adapters, and proves direct owner/wrapper fingerprint and
statistic equality for all three APIs.

The extraction audit added two atomicity characterizations for malformed source
`shape_signature` metadata. Both raw repairs now materialize and require a
rank-four signature before the indexed setter changes the Conv input. A short
signature therefore returns a zero statistic with a complete unchanged ModelIR
instead of raising `IndexError` after a partial edge mutation. Architecture
tests keep the source-signature assignment before the first setter in both
repairs. No strict xfail remains.

The corrected 223-line `_repair_mixed_nhwc_inputs_for_nchw_concat` is now owned
by `passes/mixed_concat_input_repair.py`, with a two-line private lowerer
wrapper and two production calls on fallback and final ModelIR. Its body is
AST-identical to the corrected lowerer predecessor. The focused contract covers
canonical spatial selection from two agreeing inputs, the two-input output-
shape fallback, local NHWC-to-NCHW adapter insertion, output channel/shape
reconciliation, idempotence, wrong axis, missing input, invalid rank, and an
already-NCHW no-op. Architecture tests fix module ownership, quantization
cloning/remap, direct operator insertion, the compatibility input setter,
wrapper dispatch, and both production positions. Direct owner and wrapper
execution produce identical complete ModelIR fingerprints and statistics.

The owner now resolves the required Concat output tensor and builds all
prospective adapters into a complete plan before the first tensor or operator
insertion. Every source signature is rank-validated, tensor names are reserved
across the plan, final input/output shapes are computed up front, and cloned
per-axis quantization remaps NHWC dimension `3` to NCHW dimension `1`. A
malformed later signature or missing output is therefore a zero-statistic,
complete ModelIR no-op. The commit phase retains historical tensor and operator
insertion order. Architecture tests keep output/signature resolution and plan
construction before the first mutation. No strict xfail remains.

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
fixture.

The complete Swish-QDQ orchestration is now owned by
`optimize_transpose_swish_qdq_nhwc_islands` in that same module. It preserves
the ordered primary, first inverse-post, late Concat/post, independent
Conv-input safety-valve, and final pruning boundaries. The propagated-tensor,
Concat-axis, and removed-post statistics are aggregated only after each
phase's explicit result is returned. The spatial-agnostic residual-Concat
closure is a second module owner that fixes `min_spatial_stage=0` and
`require_concat_closure=True` before remapping the established statistics.
Both historical lowerer names are thin compatibility wrappers and the two
production positions remain one call each. After normalizing only the public
function and safety-owner names, both moved orchestration ASTs are identical
to their prior committed lowerer implementations. Focused tests additionally
fix phase order, option forwarding, statistics aggregation, closure remapping,
wrapper equality, direct safety-owner dispatch, final pruning, and the absence
of a lowerer import cycle.

HardSwish/SE/HardSigmoid gating-block layout recovery is isolated in
`passes/hardswish_se_layout.py`. The complete compatibility implementation
moves as one unit because its direct or decomposed activation root, keepdims
Mean, two Conv stages, expanded or fused HardSigmoid gate, residual Mul, and
four boundary Transposes are validated and committed as one ordered contract.
The lowerer retains the historical private wrapper at both production
positions. Function-name-normalized AST comparison is exact, so consumer-map
rebuilds, graph-order restart, constant-axis remapping, input/output rewrites,
metadata and quantization propagation, removal order, pruning, and the existing
statistic are unchanged.

Four positive synthetic combinations fix direct HARD_SWISH versus decomposed
ADD/MUL/MUL roots against expanded MUL/ADD/RELU_0_TO_1 versus fused
ADD(RELU6)/MUL gates. Public pre-Transpose output, invalid reduction axes, and
activation fan-out remain complete no-ops; a direct-owner/private-wrapper
comparison fixes the compatibility boundary. SSDLite MobileNetV3,
`inference_ops15`, and MobileNetV3 PyTorch provide six measured zero-owner
production invocations. SSDLite is the artifact control: it remains accurate
with zero process-tree SWAP and byte-identical direct artifacts across the
mechanical move.

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

The exact RELU/Split all-output compatibility island is owned by
`passes/split_all_outputs_layout.py`. It recognizes one private typed
NHWC-to-NCHW Transpose, RELU, an equal channel Split, and exactly one typed
inverse Transpose on every Split output. The former 182-line lowerer
implementation is a thin compatibility dispatcher at both unchanged ordered
positions, and both calls supply Session `LayoutState`.

The Split axis must be a private or shared immutable INT32/INT64 scalar
constant normalized to channel axis one. A declared `numSplits`, when present,
must equal the output count. The static input channel must divide evenly and
every Split output must have the exact equal NCHW shape; independently
compatible dynamic signatures are converted to NHWC. This deliberately
rejects the former synthetic fixture's invalid unequal 2/4 metadata for a
six-channel two-way TFLite `SPLIT`; the corrected contract is 3/3.

Both permutation buffers, source provenance, producer uniqueness, graph
order, private intermediate boundaries, exact fan-out, dtype, per-tensor
quantization, static/dynamic shape relations, and physical layout are resolved
before a plan exists. The pre-Transpose output feeds only RELU, the RELU output
feeds only Split, and each Split output feeds only its owned inverse adapter.
Every consumer of an inverse-adapter result must occur later in graph order.
Public intermediate, duplicate producer, missing tensor, variable/constant
intermediate, per-axis quantization, contradictory layout, or stale order
rejects the complete candidate without mutation.

All downstream input replacements are grouped by operator and exact slot. A
single Concat may therefore consume several former adapter outputs, and one
former output may have several later consumers, without one rewrite replacing
another. The existing Split tensor names survive and acquire NHWC metadata;
the adapter-output aliases and removed pre-Transpose tensor are pruned only
after success.

An exclusive Split axis changes from one to three in place. A shared axis
receives one deterministic clone with the same TensorIR/NumPy INT32 or INT64
dtype, quantization, layout metadata, and provenance, while unrelated users
retain the original value. The immutable plan captures tensor/operator
contracts, every input slot, metadata, clone name, and removal. It is resolved
again immediately before apply, then one differential `ModelIRGraphIndex`
updates inputs and compacts every adapter removal. Graph-ordered candidates,
an optional rewrite limit bounded by current candidate count, differential
LayoutState updates, and success-only pruning replace the raw repeated map
build and unbounded fixed-point loop.

Pre-extraction audit covered this owner together with the adjacent exact
Conv/Concat and direct Split/Conv/Concat bridge roots. All 41 sequential
invocations on YuNet, FastestDet, HumanSeg, OSNet, and SiNet were zero-match;
the all-output root accounted for 12 invocations. Thirty-two focused owner and
active-fixture tests cover two/three branches, INT32/INT64 and negative axes,
static/dynamic signatures, exact numerical equivalence, combined and multiple
consumers, shared-axis cloning, candidate limits, idempotence, GraphIndex,
LayoutState, and nineteen transactional rejection cases. The adjacent indexed
owners and complete architecture suite pass together, and sequential YuNet
conversion reproduces all five fixed artifact hashes.

The exact two-branch RELU/Split/Conv/Concat compatibility island is a separate
immutable plan in the same `passes/split_all_outputs_layout.py` owner. It
recognizes a private NHWC-to-NCHW Transpose and RELU, an equal two-way channel
Split, one branch returning to NHWC for Conv2D, that Conv output returning to
NCHW for RELU, the untouched Split branch joining through channel Concat, and
one final inverse Transpose. The former 340-line raw helper is a thin
dispatcher at both unchanged sequence positions, immediately after the
all-output plan, and both calls supply Session `LayoutState`.

The Conv branch may be either Split output and may occupy either Concat input
slot. Neither Split-output order nor Concat order is changed. The Split input
channel must divide exactly in half, both output shapes and signatures must
match that equal result, and `numSplits`, when present, must be two. The
historical synthetic fixture's invalid six-channel 2/4 split is therefore
corrected to an eight-channel 4/4 contract. Conv may change the branch channel
count; the old NCHW Concat and new NHWC Concat shapes are independently
derived from the Conv result and retained branch, then required to agree
through the final inverse adapter.

Every permutation, producer, consumer, input slot, runtime tensor, Conv side
input provenance, graph-order edge, public boundary, static/dynamic shape,
dtype, per-tensor activation quantization, and physical layout is proven before
planning. Adapter endpoints must have equal dtype and quantization parameters.
The owned pre-Transpose, Conv-input adapter, Conv-output adapter, and final
Concat adapter are removed only together. Fan-out from any owned NCHW
intermediate, an additional Split output, unequal Split metadata, unresolved
filter/bias, stale order, duplicate producer, or contradictory layout rejects
the whole candidate without mutation.

The plan rewires pre-RELU, Conv, post-Conv RELU, and every consumer of the final
adapter by exact input slot. Split-axis copy-on-write reuses the all-output
contract. RELU/Split/Concat tensor metadata and Session LayoutState change to
NHWC only after a second complete resolution; Concat axis changes from one or
negative-three to three at the same point. One differential graph index
performs every input update and one four-operator compaction. Graph-ordered
Concat candidates, an optional candidate-count rewrite limit, and success-only
pruning replace the raw repeated full-map fixed point.

Forty-nine dedicated tests cover both Split branch positions, both Concat
orders, INT32/INT64 and negative axes, static/dynamic signatures, Conv channel
changes, exact numerical equivalence, multiple final consumers, shared-axis
cloning, candidate limits, idempotence, GraphIndex, LayoutState, and twenty-nine
transactional rejection cases. With the active compatibility fixture, adjacent
indexed owners, and full architecture suite, `525` tests pass. TensorFlow-
blocked direct/default/`-cotof` checks pass sequentially, and YuNet reproduces
the five fixed artifact hashes.

The direct Split/Conv/Concat bridge is independently owned by
`passes/split_conv_concat_bridge_layout.py`. The former 287-line lowerer
implementation is a thin dispatcher at all three unchanged production
positions. Each invocation receives Session `LayoutState`, and its phase order
relative to the exact all-output, Conv/Concat, and late QKV helpers is
unchanged.

The resolver requires a private typed NHWC-to-NCHW adapter feeding an equal
channel Split exclusively. Exactly one Split output must own one typed inverse
adapter. Every other Split output must occur exactly once as a direct input of
one selected NCHW Concat; every remaining Concat input must be the exclusive
output of a typed NHWC-to-NCHW adapter. At least one of those post-adapter
inputs must be reachable from the converted Split branch through a bounded,
graph-ordered NHWC interior. The interior is deliberately operation-agnostic:
it preserves existing Conv and non-Conv computation instead of recognizing a
model name or one hard-coded chain.

Source and axis provenance, typed permutations, producer uniqueness,
consumer slots, graph order, public boundaries, equal Split metadata,
static/dynamic shapes, dtype, per-tensor quantization, physical layout, Concat
input classification, and the retained NCHW output contract are all resolved
before planning. Every consumer of the inverse branch adapter is rewired by
exact input slot. An unclassified Concat input, Split fan-out, post-adapter
fan-out, unreachable converted branch, duplicate producer, missing tensor,
stale order, per-axis quantization, or contradictory layout rejects the full
candidate without mutation.

The immutable plan records all input replacements, every Split metadata
update, Split-axis copy-on-write, the Concat axis/output change, a private NHWC
Concat tensor, the three-or-more adapter removal group, and all tensor/operator
contracts. The original local NCHW Concat tensor name and metadata remain the
output of one inserted post adapter, which reuses the proven pre-adapter
permutation. Shared INT32/INT64 axes receive deterministic typed clones; an
exclusive axis changes in place. A second complete resolution and preflight
precedes writes, one differential graph index performs compaction and adapter
insertion, LayoutState is updated differentially, and pruning occurs only
after success.

Forty-six dedicated tests cover two- and three-way Split, either branch,
multiple NHWC post paths, both Concat orders, static/dynamic signatures,
INT32/INT64 and negative axes, exact numerical equivalence, branch-side
consumers, shared-axis cloning, candidate limits, idempotence, GraphIndex,
LayoutState, and twenty-four transactional rejection cases. With the adjacent
indexed owners, two active fixtures, and complete architecture suite, `534`
tests pass. TensorFlow-blocked direct/default/`-cotof` checks pass
sequentially, and one sequential YuNet conversion reproduces all five fixed
artifact hashes.

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
20 invocations and is implemented as a distinct semantic plan in the same
op-family owner.

Fifty-six focused tests cover all three producer types, all seven unary types,
all six binary types in both operand positions, exact numerical equivalence,
dynamic signatures, grouped shared-constant cloning, candidate limits,
idempotence, GraphIndex/LayoutState integrity, and twenty unsafe transactional
no-op contracts. Sequential comparisons against the preceding checkpoint emit
byte-identical float32, float16, correspondence, and schema artifacts for all
four active representatives.

The channel-one terminal plan recognizes only a TransposeConv-produced NHWC
tensor whose static and dynamic channel dimension is exactly one, its private
NHWC-to-NCHW adapter, a strictly linear unary/binary chain containing exactly
one Squeeze, and a final graph output with no consumers. It retains its own
entry point and stats key; the lowerer dispatcher remains immediately after
the general passthrough dispatcher and also receives Session LayoutState.

Explicit, negative, and implicit Squeeze axes are normalized against the
original rank-four NCHW contract and remapped by semantic axis label to NHWC.
Every removed dimension must be statically and dynamically one. The surviving
semantic axis order after Squeeze must be identical in both paths. This admits
the intended channel removal and other order-preserving cases while rejecting
spatial-only rewrites that would silently expose a channel-last graph output
under the original channel-first rank-three contract.

The terminal plan preserves the larger historical unary family before or
after Squeeze. Non-scalar binary constants are accepted only before Squeeze
and use the shared rank-four constant planner; binary operations after Squeeze
must use a scalar constant. Static/dynamic broadcasts, dtype, per-tensor
quantization, operand order, producer/consumer order, public intermediates,
the exact graph output, Squeeze options, metadata, constants, and the one
adapter removal are fully planned and re-resolved before apply.

Fifty-seven focused tests cover explicit/negative/implicit axes, static and
dynamic signatures, all fourteen unary operations, all six binary operations
in both operand positions, exact numerical behavior, a post-Squeeze scalar
binary, shared constant cloning, candidate limits, idempotence, GraphIndex and
LayoutState integrity, and twenty-two transactional rejection cases. YuNet
retains four zero-match invocations and emits five byte-identical artifacts
against the preceding checkpoint.

Elementwise/Concat/Conv NHWC group recovery is owned by
`passes/elementwise_concat_layout.py`. The former lowerer implementation built
complete producer/consumer maps inside an unbounded fixed-point loop and
mutated a connected group while still discovering its boundaries. The lowerer
now has one thin dispatcher at each of the two unchanged ordered positions and
passes the Session `LayoutState` into a graph-indexed semantic owner.

The owner starts from graph-ordered channel-axis Concat candidates whose every
consumer is a private NCHW-to-NHWC adapter. Reverse traversal classifies one
connected closure containing the complete historical eleven-unary and six-
binary families. A boundary must be a direct rank-four graph input, a proven
NHWC-to-NCHW adapter, or a reusable private NCHW-to-NHWC alias. Additional
Concat roots that consume the same closure are admitted into the same plan, so
shared producers cannot be converted under only part of the connected group.
The traversal is bounded by the current graph input-edge count.

Every static and dynamic rank-four view, dtype, per-tensor quantization,
producer, consumer slot, graph-order relation, public boundary, typed
permutation, binary broadcast, constant buffer, and provenance field is fixed
before a plan exists. Binary operand positions are retained. A rank-four
constant is transposed only when it is immutable and every actual consumer
slot is contained in the accepted closure; unrelated use rejects the complete
candidate. Explicit pre-adapters with fan-out remain for their legacy users.
Supported external unary/binary users of a converted closure output share one
deterministic local NHWC-to-NCHW adapter, while unsupported or public fan-out
rejects the group transactionally.

The immutable plan records all input and output rewrites, Concat axis changes,
tensor metadata updates, constant transposes, post aliases, legacy adapters,
removals, and complete tensor/operator contracts. It is resolved again from
the same seed immediately before apply. One differential
`ModelIRGraphIndex` maintains rewritten slots, removals, and adapter
insertions. Candidate-only operation and an explicit rewrite limit use the
same production entry point. Pruning runs once at owner exit, including on a
zero-match call, preserving the historical permutation-buffer and
correspondence-event ordering.

Converted closure, constant, and Concat tensors are explicitly recorded as
NHWC in both TensorIR and Session `LayoutState`. Sequential comparison with
the preceding checkpoint shows identical operator topology, names, options,
shape/signature, dtype, quantization, constants, and provenance at every
FastestDet and HumanSeg invocation boundary. The only full ModelIR-digest
difference is this intentional replacement of stale `UNKNOWN` layout
provenance with the resolved NHWC contract. Float32, float16,
correspondence, and schema artifacts remain byte-identical.

Fifty-six focused tests cover all unary types, all binary types in both
operand positions, rank-four constants, INT32/INT64 and negative axes,
dynamic signatures, connected groups, direct/preexisting boundaries, multiple
post aliases, legacy adapters, pre-adapter fan-out, numerical equivalence,
candidate limits, idempotence, GraphIndex/LayoutState integrity, and twenty-
three transactional rejection cases. Sequential characterization observed
five calls on each short representative. FastestDet retained two groups and
HumanSeg one group in the first call; the other twenty-two calls were
zero-match. YuNet, FastestDet, and HumanSeg reproduce their fifteen fixed
artifacts. No Tier corpus was run for this checkpoint.

StridedSlice/Concat fan-in recovery is owned by
`passes/stridedslice_concat_layout.py`. The former lowerer helper rebuilt full
producer/consumer maps inside an unbounded fixed-point loop, changed Slice
constants and metadata before validating the full group, and rewrote the first
post-permutation buffer when a legacy NCHW boundary had to survive. Both
ordered production positions now call one thin dispatcher with Session
`LayoutState`.

The candidate root is one typed private NHWC-to-NCHW Transpose with at least
two consumers. Its entire output fan-out must consist of supported rank-four
`STRIDED_SLICE` operations. Each Slice has four distinct inputs, zero masks,
no offset, and one immutable nonzero stride vector. Begin, end, and stride
constants must be rank-four INT32 or INT64 values whose TensorIR dtype, NumPy
dtype, static/dynamic shape, producer state, public ownership, and complete
consumer-slot set agree. Each constant is exclusive to its one Slice and the
three roles are distinct; shared or conflicting roles reject the transaction
before mutation.

Every Slice output is private, rank four, and consumed exactly once by the
same channel-axis Concat. The resolver proves producer order, dtype,
per-tensor quantization, old NCHW and new NHWC views, Concat input uniqueness,
static/dynamic Concat result, output metadata, and all post consumers. At least
one private typed NCHW-to-NHWC post adapter is required. Additional private
posts are aliases of the first canonical NHWC output; repeated downstream
input slots are recorded explicitly. Arbitrary later NCHW consumers and a
public Concat output are preserved through one local inverse adapter.

All constant updates, Slice rewrites, metadata changes, Concat axis/output
changes, post aliases, compatibility-boundary changes, removals, and complete
tensor/operator contracts belong to one immutable plan. The same seed is
fully resolved a second time immediately before apply. One differential
`ModelIRGraphIndex`, a graph-ordered Transpose candidate list, candidate-only
operation, and an optional rewrite limit replace the full scans and unbounded
loop. Pruning still runs once at owner exit, including zero-match calls, to
preserve the historical side effect.

Ordinary and multi-post active forms retain byte-identical non-layout ModelIR
digests against the raw helper, including lineage metadata and event order.
Slice input rewrites retain the historical `replace_operator_input_at` event;
each alias retains one group-level replacement event even when the downstream
operator repeats that alias in multiple slots. TensorIR and Session
`LayoutState` now record converted Slice and Concat tensors as NHWC.

When a compatibility boundary is required, the owner reuses the already-
proven typed pre-permutation. The raw helper instead overwrote the first post
buffer with an INT32 opposite permutation, even when the buffer was INT64 or
shared. The indexed owner never changes that post buffer or unrelated users.
It also rejects a legacy consumer ordered before the adapter that would become
its producer, preventing an invalid topological result.

Fifty-eight focused tests cover INT32/INT64 parameters and permutations,
dynamic signatures, negative axes, multiple post aliases, repeated slots,
legacy/public boundaries, shared post buffers, exact numerical equivalence,
candidate limits, idempotence, GraphIndex/LayoutState integrity, lineage
compatibility, zero-match pruning, and forty-two transactional rejection
cases. Pre/post characterization observed five zero-match calls on each of
YuNet, FastestDet, HumanSeg, OSNet, and SiNet. The related owner and complete
architecture gate passes 542 tests, TensorFlow-blocked direct/default/`-cotof`
passes three tests sequentially, and YuNet reproduces its five fixed artifacts.
No Tier corpus was run for this checkpoint.

Split/mixed-Concat fan-in recovery is owned by
`passes/split_mixed_concat_layout.py`. The former lowerer helper rebuilt full
producer/consumer maps inside an unbounded fixed-point loop. It is now one thin
dispatcher at each of its three unchanged source positions, and every
production call receives the Session `LayoutState`.

The indexed root is a private rank-four channel-axis Concat with at least one
direct Split output. Every other input must be another output of an accepted
Split or the exclusive result of a typed NHWC-to-NCHW adapter. All Split
outputs are classified as a unit: they may feed the target Concat directly or
private typed NCHW-to-NHWC aliases, but public or unrelated consumers reject
the entire transaction. This keeps the following general input-chain helper a
separate ordered family instead of allowing either pass to consume a partial
match owned by the other.

The resolver proves rank-four static/dynamic views, dtype, per-tensor
quantization, explicit or unknown physical layout, graph order, producer
uniqueness, exact consumer slots, typed INT32/INT64 permutation constants, and
the derived NHWC Concat view. A Split axis must be an immutable scalar typed
constant normalized to channel axis one. It changes in place only when the
Split owns its only consumer slot; otherwise the plan creates one deterministic
dtype-preserving axis-three clone. Produced, variable, public, malformed, or
per-axis-quantized values are rejected.

The plan owns each pre-Split adapter, Split input and output metadata update,
alias rewire, direct-adapter removal, Concat input/axis/output change, terminal
NCHW compatibility adapter, tensor contract, operator contract, and public
graph boundary. The same Concat is resolved again immediately before apply.
One differential `ModelIRGraphIndex` maintains the rewrites, removals, and
insertions; graph-ordered Concat candidates, candidate-only operation, and an
explicit rewrite limit replace the raw scan and fixed-point loop. Converted
tensors are recorded in both TensorIR and Session `LayoutState`.

Seven sequential short Tier 0-4 characterization models containing Split and
Concat produced no runtime match for this specialized family, while the
following general input-chain family retained its existing matches. The active
contract is therefore fixed synthetically rather than broadened to absorb
production patterns owned elsewhere. Sixteen dedicated tests cover active and
all-Split forms, dynamic signatures, negative axes, shared axis cloning,
public/fan-out/quantization/producer guards, stale-plan rejection, candidate
limits, repeated sweeps, and index/layout integrity. The adjacent owner and
architecture gate passes 524 tests. Sequential `-cotof` validation preserves
the managed values for `yolov9` (`3.0517578125e-05`) and `sgscsh`
(`2.5331974029541016e-07`).

General Concat input-adapter recovery is owned by
`passes/concat_input_adapter_layout.py`. The former lowerer helper rebuilt full
producer and consumer maps inside an unbounded fixed-point loop. It is now a
thin compatibility dispatcher at its three direct production positions and
its conservative safe-transpose fallback position. Direct production calls
receive the Session `LayoutState`; the no-keyword fallback entry remains
compatible.

The indexed root is a rank-four channel-axis Concat. Each input branch must be
the private output of a typed NHWC-to-NCHW Transpose, or one of the historical
thirteen unary operations fed by that Transpose or by an exactly equivalent
singleton-channel NHWC-to-NCHW Reshape. Direct and unary forms may be mixed.
Repeated input slots reuse one branch plan so shared metadata is changed once.
The accepted Transpose, Reshape, and unary intermediates cannot be public or
have unrelated consumers.

Every static and dynamic view, dtype, per-tensor quantization, explicit or
unknown layout, producer and consumer relation, graph-order relation, typed
constant, unary view, and derived NHWC Concat output is proven before a plan
exists. The original NCHW Concat output name, metadata, graph-output role, and
existing consumers are preserved through one terminal compatibility
Transpose. If no proven permutation buffer is available, the plan creates one
deterministic INT32 constant. Per-axis quantization, malformed or produced
constants, duplicate producers, unresolved sources, unsafe fan-out, and stale
or backward edges reject the complete candidate before mutation.

The immutable plan contains branch classification, adapter removals, unary
rewires, metadata/layout updates, Concat changes, permutation ownership,
terminal insertion, tensor/operator contracts, and graph boundaries. It is
fully resolved again immediately before apply. A graph-ordered candidate list,
candidate-only operation, and an explicit rewrite limit replace the raw scan
and fixed-point loop. One differential `ModelIRGraphIndex` maintains all
rewrites, removals, and insertions. TensorIR and Session `LayoutState` are
updated together, and pruning retains the historical zero-match side effect.

Sequential characterization retained three first-invocation direct groups in
`yolov9` and two second-invocation direct groups in `sgscsh`; the other seven
invocations were zero-match. The preceding Split/mixed-Concat owner did not
claim those groups. Thirty-six dedicated cases and two existing active
fixtures cover direct, unary, singleton-Reshape, dynamic, public-boundary,
repeated-slot, stale-plan, bounded dispatch, safe-bundle, determinism, and
transactional rejection behavior. The related owner and architecture gate
passes 561 tests. Identical conversion-only invocations against checkpoint
`f5505052` reproduce all ten fixed artifacts across `yolov9` and `sgscsh`;
separate sequential `-cotof` checks remain below `1e-1`.

Slice/optional-Logistic/Concat/Reshape detection-tail recovery is owned by
`passes/slice_logistic_concat_reshape_tail_layout.py`. The former lowerer
helper rebuilt complete producer and consumer maps inside an unbounded
fixed-point loop and began changing Slice constants and metadata before the
complete multi-branch tail was known to be valid. Both ordered production
positions now call one thin dispatcher with Session `LayoutState`.

The indexed root is a rank-three spatial-axis Concat with at least two unique
inputs. Each input must be one private rank-three Reshape fed by one private
two-input rank-four channel-axis Concat. Both Concat inputs trace through an
optional Logistic to distinct private Slice operators. The two Slices must be
the exact fan-out of one typed NHWC-to-NCHW Transpose. Different branches own
different adapters and internal operators, although their already-NHWC source
tensor may be shared safely.

The resolver proves rank-three and rank-four static/dynamic views, dtype,
per-tensor quantization, explicit or unknown layout, unique producers, strict
graph order, exact consumer slots, typed INT32/INT64 permutation and Slice
constants, Slice bounds and derived results, branch and tail Concat results,
Reshape element counts, and every public or downstream boundary. Both
`newShape` and `onnxRawNewShape` are validated against the original and target
views before their channel/spatial dimensions are exchanged.

Slice begin and size constants must be immutable, non-public, non-variable,
non-produced, and exclusive to their exact operator slots. This makes their
axis remap safe and rejects the conflicting shared-role cases that the raw
helper could mutate twice. An exclusive Reshape shape changes in place. A
shape shared with an unrelated Reshape receives one deterministic clone that
retains TensorIR dtype, NumPy dtype, quantization, and original consumers.
Malformed values, zero dimensions, per-axis quantization, duplicate producers,
repeated branch inputs, unsafe fan-out, and backward consumers reject the
whole transaction.

The immutable plan contains all Slice rewrites/constants, optional Logistic
metadata, branch Concat axes and metadata, Reshape constant/input/options and
metadata, tail Concat axis/output, typed 3D post-permutation selection or
creation, terminal compatibility adapter, removals, tensor/operator contracts,
and graph boundaries. The same tail candidate is fully resolved immediately
before apply. A graph-ordered candidate list, candidate-only operation, and an
explicit rewrite limit replace the raw scans and loop. One differential
`ModelIRGraphIndex` maintains rewritten slots, insertion, and removal;
TensorIR and Session `LayoutState` are updated together. Exit pruning preserves
the legacy zero-match side effect.

Production characterization found one four-branch first-call match in
`nanodet-plus-m_416` followed by four zero-match calls. Each branch splits 37
channels into Logistic-wrapped 5-channel and direct 32-channel paths across
52x52, 26x26, 13x13, and 7x7 feature maps. The preceding general input owner
rewrites seven disjoint groups without consuming this owner’s tail. Yolov9
retains five zero-match calls. Forty dedicated cases and the existing active
fixture cover numerical equivalence, all optional-Logistic forms, dynamic
signatures, typed constants, copy-on-write, bounded dispatch, graph/layout
state, stale plans, and twenty-four transactional rejections. The related gate
passes 603 tests, all five nanodet conversion artifacts remain byte-identical
to checkpoint `762dcdef`, and sequential `-cotof` remains below `1e-1`.

Strict direct/unary residual-Add adapter recovery is owned by
`passes/pre_add_direct_unary_layout.py`. The existing
`_optimize_transpose_pre_add_nhwc_chains` entry point remains the compatibility
owner at all four production positions and in the conservative safe-transpose
bundle. It invokes the indexed sub-owner first, then retains the historical
implementation for Swish, Gather, constant affine, PReLU, broadcast, nested
Add, and direct-fallback patterns. Direct production calls receive the Session
`LayoutState`; the safe-bundle call remains source compatible without one.

The bounded input contract is an equal-shape rank-four residual Add. Each
operand must be either a typed NHWC-to-NCHW Transpose result or one of the
historical seven unary operations fed exclusively by such a Transpose. A
direct adapter may remain for unrelated NCHW consumers; a unary branch must
be closed so its input adapter can be removed and its output metadata can be
changed to NHWC. The output suffix may be the Add itself or one exclusive
unary of the same family. At least one private typed NCHW-to-NHWC post adapter
must expose the canonical NHWC result.

The resolver proves graph order, unique producers, exact consumer slots,
static and dynamic views, dtype, per-tensor quantization, explicit or unknown
physical layout, typed immutable permutations, post aliases, public
boundaries, and the old-NCHW/new-NHWC view relation. Multiple post adapters
become aliases of the first output. Arbitrary NCHW consumers are preserved by
retaining one local inverse adapter only when its INT32 permutation buffer is
owned exclusively by the accepted post set. Shared or produced post constants
reject the indexed candidate and leave it to compatibility behavior.

One immutable plan owns input classification, optional unary rewires,
metadata/LayoutState changes, Add and output-unary producer identity, post
aliases, retained-boundary rewrites, adapter removal, complete tensor and
operator contracts, and graph boundaries. The candidate is fully resolved a
second time immediately before apply. A graph-ordered Add candidate list,
candidate-only operation, rewrite bound, and one differential
`ModelIRGraphIndex` replace repeated producer/consumer map construction for
the accepted family. The indexed owner intentionally does not prune tensors:
the compatibility wrapper retains the single historical cleanup boundary so
lineage event grouping and correspondence reports remain byte-identical.

The initial direct-output-only version was evaluated before being broadened.
It produced zero indexed matches on SiNet, FastestDet, HumanSeg, and OSNet
while the fallback preserved every result. Characterizing that finding showed
that OSNet's strict residuals all use an optional ReLU suffix. After adding the
bounded output-unary contract, the indexed first call owns six OSNet residuals
and one HumanSeg residual. FastestDet remains on the direct-fallback path and
SiNet remains on affine/PReLU paths. A first cleanup attempt split one HumanSeg
lineage prune event; moving cleanup back to the wrapper restored the exact
correspondence report without changing the graph.

Twenty focused cases cover every input-unary type, direct and unary mixing,
optional output unary, multiple post aliases, retained NCHW consumers,
dynamic signatures, quantization and ownership guards, stale-plan rejection,
bounded dispatch, idempotence, GraphIndex/LayoutState consistency, and the
single cleanup boundary. The focused, QLinear, active compatibility, and full
architecture gate passes 235 tests. TensorFlow-blocked direct, default-direct,
and direct `-cotof` pass three tests. Sequential focused `-cotof` checks pass
for SiNet, OSNet, FastestDet, and HumanSeg with zero model-process SWAP and
their exact recorded maximum errors. All float32 and float16 TFLite files are
byte-identical to the prior checkpoint; final conversion-only checks also
restore the fixed correspondence-report hashes.

The complete indexed-first compatibility composite is now isolated in
`passes/pre_add_layout.py`. Its 1,593-line implementation moved as one semantic
unit with a function-name-normalized AST identical to the prior lowerer owner;
this is not a source-line limit or a semantic split. The indexed direct/unary
owner still runs first. Its fallback still owns Swish, unary, Mul-constant,
Mul/Sub-constant, Gather, constant-Add, nested-Add, PReLU, direct-NCHW bridge,
post aliases, and legacy-consumer handling in the same fixed-point order. All
producer/consumer rebuilds, copy-on-write decisions, metadata and quantization
updates, mutation order, marker behavior, pruning, and the single historical
statistic are unchanged. The lowerer retains a one-call private compatibility
wrapper at all four production positions and in the safe-transpose bundle.

The focused fallback-equivalence fixture disables the indexed sub-owner and
proves the module owner and lowerer wrapper produce identical ModelIR. The
existing Gather/shared-constant, shared-Concat/unary, LeakyReLU, nested affine,
QLinear, and indexed transactional fixtures remain active. FastestDet
establishes non-zero fallback ownership: its eight composite calls remain
`1,0,0,0,0,0,0,0`, while all eight indexed counts remain zero. Its sequential
`-cotof` accuracy, zero process-tree SWAP, and five core artifacts are
byte-identical across extraction.

The adjacent late dual-pre-Add/single-post adapter rule is isolated in
`passes/dual_pre_add_layout.py`. Its complete 166-line implementation moved
with a function-name-normalized AST identical to the prior lowerer owner. It
still accepts only two exclusive rank-four NHWC-to-NCHW Transpose results
feeding one non-public Add with no existing NCHW-to-NHWC post consumer, moves
the Add to NHWC, and inserts one NHWC-to-NCHW compatibility adapter after it.
Tensor metadata and quantization cloning, unique-name selection, operator
order, fixed-point restart, unconditional prune boundary, statistic, and the
single late production position remain unchanged. The lowerer retains a
one-call private wrapper.

Nine focused tests fix the positive rewrite, quantization cloning,
idempotence, public Add and adapter outputs, wrong permutation, rank mismatch,
input fan-out, existing inverse post adapter, and direct-owner/private-wrapper
equality. FastestDet, OSNet, and HumanSeg supplied three sequential zero-owner
controls with process-tree SWAP zero; FastestDet's five core artifacts are
byte-identical across extraction. The historical helper reuses an existing
`__nhwc_to_nchw_perm_rank4__` tensor without validating its dtype, payload,
producer, or visibility. This latent name-collision risk is recorded rather
than changed in the mechanical checkpoint; hardening requires an independent
compatibility fixture and artifact gate.

The following terminal Transpose/Mul/Add/Reshape/FullyConnected rule is
isolated in `passes/terminal_affine_fc_layout.py`. Its complete 293-line
implementation moved with a function-name-normalized AST identical to the
prior lowerer owner. It preserves exact chain exclusivity and public-boundary
guards, NCHW-to-NHWC channel-constant rotation, shared constant copy-on-write,
both FullyConnected weight orientations, flatten-order permutation, metadata
and quantization cloning, fixed-point restart, pruning, statistic, and the
single late production position. The lowerer retains a one-call private
wrapper.

Thirteen focused tests cover both weight orientations, shared affine and
weight constants, independent quantization clones, idempotence, every public
intermediate, wrong permutation, dynamic shape, weight-width mismatch, input
fan-out, and direct-owner/private-wrapper equality. OSNet supplies a measured
zero-owner artifact control with zero process-tree SWAP and five byte-identical
core artifacts. A read-only scan of root ONNX files up to 50 MiB found no raw
Transpose/Mul/Add/Reshape/Gemm-or-MatMul chain, so non-zero production ownership
is not claimed. The historical helper can rotate an exclusive Mul constant
before discovering that the Add constant is invalid, leaving a partial change
on a zero-stat result. This transactional defect is recorded without changing
compatibility in the mechanical ownership checkpoint.

The adjacent terminal Transpose/PReLU/Reshape/BatchMatMul rule is isolated in
`passes/terminal_prelu_bmm_layout.py`. Its complete 263-line implementation
moved with a function-name-normalized AST identical to the prior lowerer owner.
It preserves scalar, rank-three CHW, rank-four NCHW, and already-NHWC alpha
handling, shared alpha/RHS copy-on-write, NHWC flatten-order RHS permutation,
adjoint rejection, metadata and quantization cloning, fixed-point restart,
pruning, statistic, and its single conditional late production position. The
lowerer retains a one-call private wrapper.

Seventeen focused and existing positive cases cover all supported alpha forms,
shared constants, independent quantization clones, idempotence, public
intermediates, wrong permutation, dynamic shape, RHS width, adjX/adjY, input
fan-out, one-dimensional alpha rejection, and owner/wrapper equality.
`inference_ops15` supplies the zero-owner artifact control with zero
process-tree SWAP and five byte-identical core artifacts. A read-only scan of
root ONNX files up to 50 MiB found no complete raw source chain, so non-zero
production ownership is not claimed. Alpha and RHS tensors still lack complete
producer, variable-state, and graph-visibility ownership validation; that
semantic hardening remains separate from the exact ownership move.

The terminal Transpose/Mul/Add/PReLU/post-Transpose compatibility rule is now
isolated in `passes/terminal_affine_prelu_layout.py`. Its complete 295-line
implementation moved with a function-name-normalized AST identical to the
prior lowerer owner. It preserves commutative affine inputs, NCHW-to-NHWC
channel-constant rotation, shared-constant copy-on-write, multiple post-
Transpose aliases, retained legacy NCHW consumers through one reverse adapter,
metadata and quantization propagation, fixed-point restart, pruning,
statistics, and the single ordered production statement reached through four
runtime recovery invocations. The lowerer retains a one-call private wrapper.

The former giant direct-builder fixture is now a focused module and runs the
pass owner and private wrapper on deep copies, comparing the complete ModelIR.
It fixes the positive terminal rewrite together with the legacy-consumer
adapter. SiNet supplies four measured zero-owner invocations before and after
the move, records zero process-tree SWAP, and reproduces all five core artifacts
byte for byte. Positive production ownership is therefore not claimed. The raw
owner still builds complete maps in an unbounded fixed-point loop, and its
sequential constant rotation can leave a partial mutation when a later
constant rejects; transactional hardening remains separate from this exact
ownership checkpoint.

The Transpose/Mean/Mul/Add/post-Transpose compatibility rule is now isolated in
`passes/mean_affine_prepost_layout.py`. Its complete 359-line implementation
moved with a function-name-normalized AST identical to the prior lowerer owner.
It preserves NCHW-to-NHWC reduction-axis remapping, commutative affine inputs,
static broadcast validation, channel-constant rotation and copy-on-write,
post-Transpose alias collapse, tensor metadata and quantization propagation,
fixed-point restart, pruning, statistics, and all three ordered source call
positions reached through five runtime invocations. The lowerer retains a one-
call private wrapper.

The former giant direct-builder axis-remap fixture is now a focused module and
runs the pass owner and private wrapper on deep copies, comparing the complete
ModelIR. LINEA supplies five measured zero-owner invocations before and after
the move, records zero process-tree SWAP, and reproduces all five core artifacts
byte for byte. Positive production ownership is therefore not claimed. The raw
owner retains an unbounded complete-map scan and in-place axes/constant updates
without an immutable all-or-nothing plan; transactional hardening remains
separate from this exact ownership checkpoint.

The dual affine-input BatchMatMul compatibility rule is now isolated in
`passes/batchmatmul_affine_input_layout.py`. Its complete 317-line
implementation moved with a function-name-normalized AST identical to the
prior lowerer owner. It preserves commutative Mul/Add inputs, exact exclusive
branch matching, NCHW-to-NHWC channel-constant rotation, rank-three Reshape
shape reversal, left post-Transpose removal, `adjY=True` conversion, metadata
propagation, fixed-point restart, pruning, statistics, and both ordered
production positions. The lowerer retains a one-call private wrapper.

The former giant direct-builder dual-branch fixture is now a focused module and
runs the pass owner and private wrapper on deep copies, comparing the complete
ModelIR. LINEA supplies two measured zero-owner invocations before and after the
move, records zero process-tree SWAP, and reproduces all five core artifacts
byte for byte. Positive production ownership is therefore not claimed. The raw
owner still mutates both branches sequentially before all shape constants are
known valid, so a late rejection can leave partial input, metadata, and constant
changes; transactional hardening remains separate from this exact ownership
checkpoint.

The BatchMatMul-to-SE layout compatibility rule is now isolated in
`passes/batchmatmul_se_layout.py`. Its complete 363-line implementation moved
with a function-name-normalized AST identical to the prior lowerer owner. It
preserves the BatchMatMul/Reshape source, NCHW Mean and axis remap, NHWC Conv
gate branch, reverse gate adapter, Logistic and residual Mul merge, constant
updates, alias rewiring, metadata and quantization propagation, fixed-point
restart, pruning, statistics, and both ordered production positions. The
lowerer retains a one-call private wrapper.

The former giant direct-builder SE fixture is now a focused module and runs the
pass owner and private wrapper on deep copies, comparing the complete ModelIR.
LINEA supplies two measured zero-owner invocations before and after the move,
records zero process-tree SWAP, and reproduces all five core artifacts byte for
byte. Positive production ownership is therefore not claimed. The raw owner
still performs a long sequence of constant, option, edge, metadata, and alias
mutations without an immutable all-or-nothing plan; transactional hardening
remains separate from this exact ownership checkpoint.

The rank-three BatchMatMul input-adapter compatibility rule is now isolated in
`passes/batchmatmul_adjoint_layout.py`. Its complete 145-line implementation
moved with a function-name-normalized AST identical to the prior lowerer owner.
It preserves exclusive Transpose-output ownership, graph-output protection,
fully known positive shape checks, exact permutation/shape validation,
`[0,2,1]` Transpose removal with `adjX`/`adjY` toggling, singleton-preserving
Transpose-to-Reshape conversion with a new INT32 shape tensor, fixed-point
restart, conditional pruning, statistics, and both ordered production
positions. The lowerer retains a one-call private wrapper.

The focused owner fixture runs the module owner and private wrapper on deep
copies, compares the complete ModelIR, covers both input positions and both
rewrite forms, and fixes idempotence. Tier 0
`speech_command_classifier_trained.onnx` establishes positive production
ownership with runtime counts `1,0`; its sequential pre/post conversion-only
runs record zero process-tree SWAP and reproduce all five core artifacts byte
for byte. The one-sample pre-move accuracy checkpoint passes with
`max_abs=2.86102294921875e-06`. The mechanical owner still rebuilds complete
producer/consumer maps after every accepted adapter and directly mutates or
deletes operators without a transaction; indexed transactional migration
remains separate from this exact ownership checkpoint.

The probable-NHWC axis-sensitive sanitizer is now isolated in
`passes/probable_nhwc_axis_sanitizer.py`. Its complete 245-line implementation
moved with a function-name-normalized AST identical to the prior lowerer owner.
It preserves the historical probable-NHWC shape heuristic, SPLIT axis copy-on-
write, CONCATENATION and SLICE axis/constant repair, unary and binary metadata
propagation, public-layout and explicit-NCHW guards, conditional terminal
NHWC-to-NCHW output adapters, fixed-point restart, both statistics, and both
ordered production positions. The lowerer retains a one-call private wrapper.

The dedicated four-case fixture runs the module owner and private wrapper on
deep copies and compares the complete ModelIR for every positive and no-op
contract. FastestDet supplies four measured zero-owner invocations before and
after the move, records zero process-tree SWAP, and reproduces all five core
artifacts byte for byte. Positive production ownership is therefore not
claimed. The raw owner still rebuilds complete maps, mutates shared SLICE
constants without copy-on-write, and inserts terminal operators directly
without an invariant transaction; semantic hardening and indexed migration
remain separate from this exact ownership checkpoint.

The adjacent raw NCHW→NHWC elementwise roundtrip compatibility owner now has a
focused closed-subgraph and rejection contract. Characterization exposed that
the root output metadata was permuted once with the intermediate tensors and a
second time after copying it to the canonical post-Transpose output. The owner
now excludes the private root tensor from the intermediate metadata loop, so
the canonical tensor receives exactly one NHWC-to-NCHW permutation. Multi-
input rewiring, embedded constants, fan-out rejection, public-output rejection,
pruning, and idempotence are fixed by the focused tests. The implementation
remains in the lowerer pending positive production ownership evidence; this
semantic correction is not combined with an ownership extraction.

The opposite-direction NHWC→NCHW elementwise fan-out compatibility rule is
now isolated in `passes/elementwise_fanout_layout.py`. Its complete 555-line
implementation moved with a function-name-normalized AST identical to the
prior lowerer owner. It preserves forward elementwise-DAG discovery, external-
runtime-input rejection, local/shared per-channel constant rotation, inverse
boundary-Transpose collapse, legacy NCHW adapters, canonical aliases, metadata
and quantization propagation, candidate snapshots, unbound-input rollback,
fixed-point restart, pruning, statistics, and all three ordered production
positions. The lowerer retains a one-call private wrapper and the independent
unbound-input pass is imported through a compatibility alias without a reverse
lowerer dependency.

The former giant direct-builder fan-out fixture is now focused and runs the
module owner and private wrapper on deep copies, comparing the complete
ModelIR. Tier 0 `shadowformer_istd_160x240_split.onnx` supplies six measured
zero-owner invocations before and after the move, records zero process-tree
SWAP, and reproduces all five core artifacts byte for byte. Positive production
ownership is therefore not claimed. The raw owner still rebuilds complete maps
and deep-copies the whole ModelIR for each accepted candidate while retaining a
dormant external-input adapter branch behind a conservative rejection guard;
indexed transactional redesign remains separate from this exact ownership
checkpoint.

The broader residual Add/Mul/Add/PReLU compatibility rule is isolated in
`passes/residual_affine_prelu_layout.py`. Its complete 415-line implementation
moved with a function-name-normalized AST identical to the prior lowerer owner.
It preserves dual pre-Add input planning, affine and alpha constant
prevalidation and copy-on-write, broadcast-aware rotations, PReLU post aliases,
legacy NCHW consumer adapter retention, metadata and quantization propagation,
operator removal order, fixed-point restart, pruning, statistic, and all three
source call positions. The lowerer retains a one-call private wrapper; the
separate indexed SiNet late-residual owner and its ordering are unchanged.

The existing direct fixture now runs the module owner and private wrapper on
deep copies and compares every operator and tensor payload. The full indexed
SiNet residual suite remains green. SiNet establishes real production
ownership across fourteen runtime invocations with counts
`0,0,0,1,1,0,0,0,0,0,0,0,0,0`; its zero process-tree SWAP and five core
artifacts are unchanged across extraction. Constant producers, variable state,
and graph visibility remain less strict than the newer indexed planning
contracts and are recorded as future semantic-hardening boundaries.

The adjacent residual Add/Mul/Add/post-Transpose fan-out compatibility rule is
isolated in `passes/residual_affine_fanout_layout.py`. Its former 477-line
lowerer implementation moved with a function-name-normalized AST identical to
the module owner. The fixed-point matcher still accepts two NHWC-to-NCHW
Transpose inputs feeding one residual Add, one or more
Mul-constant→Add-constant→NCHW-to-NHWC branches, and optional legacy NCHW
consumers. It preserves the exact profitability guard, per-branch constant
prevalidation and rotation, shared-constant copy-on-write, one retained legacy
adapter, tensor metadata and quantization propagation, removal order, pruning,
statistic, and all three production positions. The lowerer retains a one-call
private wrapper.

The focused positive fixture fixes a two-branch graph with a legacy consumer
and a Mul constant shared outside the candidate. It runs the module owner and
private wrapper on deep copies, compares the complete ModelIR, proves that the
shared NCHW constant is retained while an NHWC clone is created, and verifies
idempotence. A public post-Transpose output supplies the no-op boundary case.
SiNet reaches the owner fourteen times with zero rewrites before and after the
move; its two strictly sequential conversions report zero process-tree SWAP
and produce byte-identical float32, float16, correspondence, schema, and
generated-schema artifacts. Positive production ownership is therefore not
claimed. Constant producer, variable-state, and graph-visibility validation
remain looser than newer immutable indexed contracts and require a separate
semantic-hardening checkpoint.

The following pre-unary Mul/Add/post-Transpose fan-out compatibility rule is
isolated in `passes/pre_unary_affine_fanout_layout.py`. Its former 401-line
lowerer implementation moved with a function-name-normalized AST identical to
the module owner. It preserves the strict private NHWC-to-NCHW Transpose,
single RELU/RELU6/LOGISTIC/TANH/HARD_SWISH/LEAKY_RELU/GELU producer, complete
Mul-constant→Add-constant→NCHW-to-NHWC fan-out, broadcast-aware constant
prevalidation and rotation, shared-constant copy-on-write, tensor metadata and
quantization propagation, exact removal order, fixed-point restart, pruning,
statistic, and all three production positions. The lowerer retains a one-call
private wrapper.

Ten focused cases cover every accepted unary, a two-branch graph, an externally
shared constant, full module-owner/private-wrapper ModelIR equality,
idempotence, unsupported unary rejection, and a public post-output boundary.
SiNet reaches the compatibility owner five times with zero rewrites before and
after extraction; its strictly sequential conversions report zero process-tree
SWAP and reproduce all five core artifacts byte for byte. This agrees with the
earlier fourteen-model, five-boundary characterization and 381-model active
Tier 0-4 ONNX topology scan, which also found no real owner. Positive production
ownership is not claimed. The raw compatibility contract still lacks a shared
GraphIndex/LayoutState transaction and complete producer, variable-state, and
graph-visibility validation for constants; hardening remains separate from the
mechanical ownership checkpoint.

The pre-Add rank-four to rank-three reshape suffix recovery now has an indexed
semantic owner in `pre_add_mulconst_reshape_suffix_layout.py`. The owner keeps
the historical position inside `_run_layout_reshape_attention_recovery_prefix`
and deliberately covers both direct/direct and direct/Mul-constant branches:
the compatibility helper historically claims both families despite its
Mul-const name. It dispatches only indexed Add operators and reuses one
`ModelIRGraphIndex` for each prefix invocation. Candidate resolution validates
typed `[0,3,1,2]` and `[0,2,1]` permutations, positive rank-four source and Add
views, exact `[N,C,H*W]` and `[N,H*W,C]` suffix views, shape/dtype/layout and
per-tensor quantization compatibility, graph boundaries, producer ordering,
exclusive mutable edges, and typed reshape constants before mutation.

Each accepted candidate becomes an immutable plan containing operator and
tensor contracts. Apply resolves the candidate again and rejects a stale plan
atomically. Channelwise Mul constants and shared reshape-shape constants use
copy-on-write; exclusive constants are updated in place to preserve existing
artifact names. Legacy NCHW consumers receive one dedicated indexed adapter,
while closed suffixes remove all redundant Transposes. `LayoutState` is updated
alongside tensor metadata. The indexed owner does not prune, so the existing
compatibility wrapper remains the sole cleanup and tensor-lineage report
boundary. Strict rejections continue through the original raw fallback.

The production characterization model is Tier 2
`iat_llie_180x320.onnx`: its three ordered invocations index 5, 4, and 4
rewrites, including seven direct/direct and six direct/Mul-constant chains.
Its float32, float16, and correspondence artifacts remain byte-identical to
the pre-extraction baseline, and the sequential accuracy gate passes with
maximum absolute error `4.470348358154297e-07` and zero process SWAP.

The indexed-first composite and its raw compatibility fallback are now isolated
in `pre_add_mulconst_reshape_suffix_compat_layout.py`. The former 509-line
lowerer implementation moved with a function-name-normalized AST identical to
the module owner. It still constructs one `ModelIRGraphIndex` for the indexed
dispatch, forwards the caller's `LayoutState`, starts the combined statistic
from the indexed result, runs the unchanged direct/direct and
direct/Mul-constant fallback to fixed point, and performs the sole historical
prune/report boundary. The lowerer retains one private wrapper at the unchanged
production position and forwards Session layout state without exposing the new
owner publicly.

The complete thirteen-case family suite now runs the compatibility module owner
and lowerer wrapper on deep copies, fixes their complete ModelIR equality and
single-prune behavior, and forces the indexed dispatch to zero to prove the raw
fallback still owns both accepted input forms. IAT-LLIE retains combined and
indexed counts `5,4,4` with fallback counts `0,0,0`; its two strictly sequential
conversions report zero process-tree SWAP and reproduce the five core artifacts
byte for byte. The indexed immutable plan is unchanged. The raw fallback still
has its historical producer/consumer rebuilds and looser constant ownership;
semantic hardening remains separate from this exact ownership move.

The formerly adjacent raw direct/direct-only helper has been removed. It was
not a second semantic owner: the compatibility helper above has always
accepted the same direct/direct producer and suffix contracts before it, in
addition to its Mul-constant family. Production-boundary instrumentation
confirmed zero residual rewrites in all three ordered invocations for
IAT-LLIE, five short zero-SWAP representatives, and LINEA. A topology scan of
the fixed 49-model Tier 0-4 measured-quick set found no other candidate model.
Removing the unreachable helper deletes 290 lines and avoids three repeated
producer/consumer-map builds plus full Add scans per conversion. Architecture
tests now forbid reintroducing either its private definition or call while the
single ordered indexed/compatibility owner retains all fixtures and artifact
behavior.

The rank-four Swish to rank-three reshape suffix now has a bounded indexed
owner in `pre_unary_reshape_suffix_layout.py`. It dispatches only indexed Mul
operators and accepts the generic
`Transpose -> Logistic/Mul -> Reshape -> Transpose` family when typed
permutations, exact NHWC/NCHW and NCW/NWC views, dtype and per-tensor
quantization, graph boundaries, exclusive mutable edges, operator order, and
an exclusive typed reshape constant all agree. An immutable plan records the
complete tensor/operator contract and is fully resolved again immediately
before mutation. Apply uses differential graph-index updates, explicitly
updates Session layout state, and has a deterministic rewrite bound. Plain
unary cases, shared constants, dynamic or relaxed views, and every strict
reject remain on the unchanged raw fallback. Pruning stays at the wrapper's
single historical cleanup boundary, which removes corresponding stale layout
entries after cleanup.

LINEA is the non-zero production model: the first prefix invocation indexes
one Swish suffix and the next two index zero. The first implementation exposed
one correspondence-only incompatibility because a whole-input mutation
reported `set_operator_inputs` instead of the historical
`replace_operator_input_at`. Recording and correcting that single lineage
source label restored all artifacts byte-for-byte. The sequential accuracy
gate passes with maximum absolute error `0.002297189086675644` and zero
process-tree SWAP.

The indexed-first Swish/plain-unary composite and its raw fallback are now
isolated in `pre_unary_reshape_suffix_compat_layout.py`. The former 302-line
lowerer implementation moved with a function-name-normalized AST identical to
the module owner. It still builds one `ModelIRGraphIndex` for the indexed Swish
dispatch, forwards caller `LayoutState`, accumulates the combined statistic,
then runs the unchanged thirteen-operation unary and relaxed Swish fallback to
fixed point. The sole prune/report boundary and LayoutState removal of pruned
tensor names remain in the compatibility owner. The lowerer retains one private
wrapper at the unchanged production position.

The focused family now includes complete compatibility-owner/lowerer-wrapper
equality for both the indexed Swish path and a plain LEAKY_RELU fallback, while
the existing direct fixture retains the raw unary graph contract. LINEA keeps
combined and indexed counts `1,0,0` with fallback counts `0,0,0`; its two
strictly sequential conversions report zero process-tree SWAP and reproduce all
five core artifacts byte for byte. The indexed immutable plan is unchanged.
Raw fallback whole-graph scans, relaxed constant mutation, and lack of a shared
differential index remain explicit future semantic work.

The factorized rank-four to rank-five detection-head reshape now has a strict
indexed Case B owner in `expanddims_reshape_layout.py`. The owner dispatches
each indexed Transpose candidate once and accepts only
`NHWC -> NCHW -> [N,A,B,H,W] -> [N,A,H,W,B]` when `A > 1`, `C=A*B`, both
typed permutations, both exclusively owned mutable constants, exact static
views and signatures, dtype/quantization, layouts, operator order, and graph
boundaries agree. The immutable plan is re-resolved before apply; graph edges
and removal use differential index updates and Session layout state is
reconciled explicitly. Singleton Case A and every strict reject remain on the
raw fallback. The wrapper retains one prune/report boundary and now also
rejects shared shape or permutation constants before its legacy in-place
mutation.

`yolov7-tiny.onnx` and `yolo_test.onnx` each record indexed counts 3, 0, 0,
and 0 at the four production invocations. A first focused implementation
mistakenly used a rank-four-only typed permutation helper for the rank-five
post edge; the problem was recorded before adding the bounded length-aware
reader. Both models then retained byte-identical float32, float16, and
correspondence artifacts. The sequential yolo_test accuracy gate passes with
maximum absolute error `2.4437904357910156e-06` and zero process-tree SWAP.

The indexed-first factorized/singleton compatibility composite is now isolated
in `expanddims_reshape_compat_layout.py`. The former 271-line lowerer
implementation moved with a function-name-normalized AST identical to the
module owner. It still builds one `ModelIRGraphIndex` per invocation, dispatches
the strict factorized Case B owner first, forwards caller `LayoutState`, then
runs the unchanged singleton Case A and relaxed compatibility fallback to fixed
point. Reshape and permutation constant updates, the combined statistic, the
sole prune/report boundary, and removal of pruned tensor names from LayoutState
remain in the compatibility owner. The lowerer retains one private adapter at
both unchanged production call positions.

The focused corpus compares the compatibility owner and lowerer adapter for
indexed Case B, singleton Case A, shared-constant rejection, and LayoutState
cleanup while retaining the two historical direct Case A fixtures. A strictly
sequential pre/post `yolo_test.onnx` conversion records zero process-tree SWAP
and reproduces all five core artifacts byte for byte. The indexed immutable
plan is unchanged. Whole-graph fallback scans and relaxed in-place constant
mutation remain explicit future semantic work.

The static rank-four to rank-three flatten-HW suffix now has a bounded indexed
owner in `flatten_hw_reshape_layout.py`. It accepts only the exact semantic
family `NHWC -> NCHW -> [N,C,H*W] -> [N,H*W,C]`. Candidate resolution checks
typed rank-four and rank-three permutation constants, exact positive shapes
and signatures, dtype and per-tensor quantization, graph ordering and
boundaries, exclusive mutable data edges, an exclusive typed reshape-shape
constant, and Session layout compatibility. Each candidate becomes an
immutable operator/tensor plan that is fully resolved again immediately
before differential mutation. The owner has a deterministic candidate bound,
uses one shared graph index, reconciles `LayoutState`, and never performs
internal pruning or a whole-graph producer/consumer rebuild.

The existing wrapper remains the compatibility boundary for dynamic or
otherwise relaxed variants and performs the single historical prune. Its raw
path now refuses to mutate a reshape-shape tensor visible through another
consumer, graph input/output, producer, or variable state. Thus strict
rejection cannot corrupt a shared constant. LINEA is the sole established
non-zero short zero-SWAP production model: indexed invocation counts are
`2, 0, 0, 0`, while thirteen other measured representatives remain zero at
all invocations. LINEA's float32, float16, and correspondence artifacts remain
byte-identical, and its sequential `-cotof` gate passes at maximum absolute
error `0.002297189086675644` with zero process-tree SWAP.

The indexed-first static/dynamic flatten-HW compatibility composite is now
isolated in `flatten_hw_reshape_compat_layout.py`. The former 175-line lowerer
implementation moved with a function-name-normalized AST identical to the
module owner. It still creates one `ModelIRGraphIndex` per invocation,
dispatches the strict static owner first, forwards caller `LayoutState`, then
runs the unchanged dynamic-signature and relaxed fallback to fixed point.
Reshape constant/option updates, the combined statistic, the sole prune/report
boundary, and removal of pruned tensor names from LayoutState remain in the
compatibility owner. The lowerer retains one private adapter at both unchanged
production call positions.

The focused corpus compares compatibility-owner and lowerer-adapter results for
the indexed static path, dynamic-signature fallback, shared/boundary/produced/
variable shape-constant rejection, and LayoutState cleanup. A strictly
sequential pre/post LINEA conversion records zero process-tree SWAP and
reproduces all five core artifacts byte for byte. The indexed immutable plan is
unchanged. Whole-graph fallback scans and relaxed in-place constant mutation
remain explicit future semantic work.

The static QKV rank adapter now has a bounded indexed owner in
`attention_qkv_reshape_layout.py`. It accepts only the production-proven
`[A,1,C] -> [A,H,D] -> [H,A,D] -> [1,H,A,D]` family with rank-three
permutation `[1,0,2]`, where `C=H*D`. Candidate resolution requires exact
positive shapes and signatures, matching dtype and per-tensor quantization,
typed and exclusively owned mutable shape/permutation constants, a typed tail
shape constant, resolved sources, exclusive data edges, graph order and
boundaries, and UNKNOWN Session layout throughout the semantic view chain.
Each candidate is captured in an immutable operator/tensor contract and fully
resolved immediately before apply. Mutation updates the shared graph index
differentially, records the historical output-lineage event, removes only the
tail Reshape, reconciles `LayoutState`, and obeys a deterministic rewrite
bound without internal pruning or repeated graph-wide map construction.

The wrapper retains one compatibility fallback and one historical prune.
Unproven permutation `[1,2,0]`, shared-constant copy-on-write, dynamic
signatures, and all other relaxed contracts therefore preserve their previous
behavior. Tier 3 `rf-detr-nano.onnx` establishes the real owner with indexed
invocation counts `5, 0, 0, 0`; the raw fallback has zero residual rewrites.
Its float32, float16, and correspondence outputs remain byte-identical to the
pre-extraction baseline. The sequential `-cotof` gate passes with maximum
absolute error `0.000102996826171875` and zero process-tree SWAP.

The indexed-first static/relaxed QKV compatibility composite is now isolated
in `attention_qkv_reshape_compat_layout.py`. The former 245-line lowerer
implementation moved with a function-name-normalized AST identical to the
module owner. It still creates one `ModelIRGraphIndex` per invocation,
dispatches the strict static `[1,0,2]` owner first, forwards caller
`LayoutState`, then runs the unchanged `[1,2,0]`, shared-constant copy-on-write,
dynamic-signature, and relaxed fallback to fixed point. Shape/permutation
constant cloning and updates, the combined statistic, the sole prune/report
boundary, and removal of pruned tensor names from LayoutState remain in the
compatibility owner. The lowerer retains one private adapter at both unchanged
production call positions.

The focused corpus compares compatibility-owner and lowerer-adapter results for
the indexed static path, HDA fallback, shared-constant copy-on-write, dynamic
fallback, and LayoutState cleanup. A strictly sequential pre/post RF-DETR Nano
conversion records zero process-tree SWAP and reproduces all five core
artifacts byte for byte. The indexed immutable plan is unchanged. Whole-graph
fallback scans and relaxed clone-on-write mutation remain explicit future
semantic work.

The static Swish-to-Squeeze rank adapter now has a bounded indexed owner in
`pre_unary_squeeze_suffix_layout.py`. It accepts only the production-proven
`NHWC -> NCHW -> Logistic/Mul -> Squeeze(axis=2) -> Transpose([0,2,1])`
family. Resolution requires typed rank-four and rank-three permutations,
positive and identical static shape signatures, exact NHWC/NCHW and NCW/NWC
views, matching dtype and per-tensor quantization, resolved source ownership,
exclusive mutable data edges, graph order and public-boundary safety, and
compatible Session layout state. Malformed Squeeze options are a strict
no-op. Each accepted candidate is captured as an immutable tensor/operator
contract and fully resolved again before mutation, so stale plans are rejected
atomically. Apply updates inputs and outputs through the shared graph index,
removes only the pre/post Transposes, explicitly reconciles layout state, and
has a deterministic candidate/rewrite bound without internal pruning or
whole-graph producer/consumer-map reconstruction.

The wrapper preserves the original raw implementation after the indexed
dispatch and remains the single historical prune/report boundary. Plain unary
operators, NCHW axis-3 Squeeze, dynamic signatures, relaxed or shared edges,
and every other strict rejection therefore retain compatibility behavior.
Tier 1 `inference_ops15.onnx` establishes the real owner with indexed counts
`1, 0, 0`. Its float32, float16, and correspondence outputs remain
byte-identical to the pre-extraction baseline. Sequential `-cotof` passes with
maximum absolute error `1.9073486328125e-06`, and both conversion checks record
zero process-tree SWAP.

The indexed-first static-Swish/plain-unary Squeeze composite and raw fallback
are now isolated in `pre_unary_squeeze_suffix_compat_layout.py`. The former
297-line lowerer implementation moved with a function-name-normalized AST
identical to the module owner. It still creates one `ModelIRGraphIndex` for the
indexed dispatch, forwards caller `LayoutState`, accumulates the combined
statistic, then runs the unchanged plain-unary, axis-3, dynamic-signature, and
relaxed Swish fallback to fixed point. Squeeze axis option remapping, tensor
metadata propagation, the sole prune/report boundary, and removal of pruned
names from LayoutState remain in the compatibility owner. The lowerer retains
one private wrapper at the unchanged production position.

All eight focused cases now compare compatibility-owner and lowerer-wrapper
results for indexed Swish, plain unary, axis-3 Swish, and dynamic-signature
fallbacks, in addition to indexed atomicity, determinism, and bounded dispatch.
`inference_ops15` keeps combined and indexed counts `1,0,0` with fallback counts
`0,0,0`; its two strictly sequential conversions report zero process-tree SWAP
and reproduce all five core artifacts byte for byte. The indexed immutable plan
is unchanged. The raw fallback's whole-graph scans and relaxed in-place axis
updates remain future semantic work.

The production-proven static Conv/Mul affine fold now has a bounded indexed
owner in `conv_mul_affine_fold.py`. It accepts only
`CONV_2D(fused=NONE) -> MUL(fused=NONE)` with an exclusive FLOAT32
`[1,1,1,O]` scale, exclusive FLOAT32 `[O,1,1,I]` filter and `[O]` bias,
static equal NHWC-compatible Conv/Mul output views, SAME padding, unit stride
and dilation, resolved graph ownership, safe public boundaries, and no
constant Add suffix. Each candidate is captured in an immutable
operator/tensor contract and resolved again immediately before apply. The
rewrite changes the Conv output and removes Mul through differential
`ModelIRGraphIndex` updates, reconciles the Session `LayoutState`, and obeys a
deterministic candidate/rewrite bound without internal pruning or complete
producer/consumer-map reconstruction.

`conv_mul_affine_fold_compat.py` now owns the single indexed-first
compatibility and cleanup boundary; the lowerer retains only a thin private
wrapper at all three production positions. The complete 381-line orchestration
moved with an AST identical after function-name normalization. Add-only,
Mul/Add, fused-ReLU, missing-bias, scalar or relaxed coefficients, dynamic
signatures, quantized/shared/public constants, and all other strict rejects
therefore continue through the historical raw fallback after indexed dispatch.
The indexed bias calculation deliberately retains the legacy float32
`bias * scale + positive_zero` operation order. That apparently redundant
addition canonicalizes an IEEE-754 negative-zero product to positive zero and
is required for byte-identical serialized buffers.

Tier 2 `iat_llie_180x320.onnx` establishes the real owner with indexed counts
`12, 0, 0`; all twelve are Mul-only folds and the fallback has no residual
rewrite. Its float32, float16, and correspondence artifacts remain
byte-identical across compatibility extraction. The sequential `-cotof` gate
passes with maximum absolute error `4.470348358154297e-07`; both extraction
control runs record zero process-tree SWAP.

Producer/activation fusion is isolated in `activation_fusion.py`. The public
compatibility surface remains the private lowerer wrapper
`_optimize_fuse_conv_activation_chains`, while the implementation module owns
indexed matching for Conv2D, depthwise Conv2D, Add, Sub, Mul, and Div followed
by their supported ReLU family. It preserves the historical single-consumer,
dtype, protected-boundary, graph-output bridge, existing fused-activation, and
operator-arity guards. Output replacement and activation removal update one
`ModelIRGraphIndex` differentially, and the exact Conv/Add/binary lineage event
types and per-family counters remain unchanged.

Cleanup remains at the same pass exit and now receives the Session
`LayoutState`, so removal of unused pre-fusion tensor names cannot leave stale
layout entries. All direct production calls and the shared final convergence
forward the same Session state; the final convergence also continues to reuse
its single graph index. The compatibility scan/restart loop is intentionally
unchanged in this mechanical checkpoint. Moving it out of the central lowerer
reduces unrelated lowering context without altering semantic ownership or
artifact order.

FastestDet, HumanSeg, OSNet, and IAT-LLIE establish real production ownership.
Their three invocation totals are respectively `35,0,0`, `60,0,0`,
`77,0,24`, and `1,0,0`. The families jointly cover Conv and Add fusion; the
focused synthetic contract additionally fixes depthwise Conv, Sub, Mul, and
Div behavior. Every float32, float16, and correspondence artifact for all four
models is byte-identical across extraction, and every monitored conversion
records zero process-tree SWAP.

Dynamic Reshape metadata resolution is isolated in
`dynamic_reshape_resolution.py`. The implementation owns both the static
element-count resolver and the ModelIR pass that reconciles `newShape`,
`onnxRawNewShape`, optional `allowZero`, constant shape inputs, and output
shape/signature metadata. It preserves runtime-driven empty templates,
dynamic `-1`, final high-rank runtime-inference preference, zero-copy axes,
stale static constants, layout-transpose-as-Reshape markers, and malformed or
unresolvable no-op behavior.

The lowerer retains thin compatibility wrappers for both historical private
names. Callers that already share a `ModelIRGraphIndex` still dispatch only
indexed Reshape operators; legacy standalone callers retain their original
single operator-list traversal. This metadata-only extraction does not alter
topology, pass order, counters, or public API, but removes the complete
shape-family decision tree from unrelated central lowering context.

Tier 3 `rf-detr-nano.onnx` establishes real production ownership: its first
four ordered resolver invocations report zero and the absolute-final
runtime-inference invocation resolves three Reshapes. Its float32, float16,
and correspondence artifacts remain byte-identical across extraction, and
both monitored conversions record zero process-tree SWAP. IAT-LLIE and OSNet
provide measured zero-owner controls at all five boundaries.

Static shape fixed-point reconciliation and its pure inference helpers are
isolated in `static_shape_reconciliation.py`. The module owns Slice and
StridedSlice extents, BatchMatMul broadcasting, reduction axes, Squeeze,
rank-four signature propagation, Conv/pool output dimensions, and the
cross-op fixed-point resolver. The historical lowerer private names remain
thin wrappers, preserving direct test and internal call compatibility without
making the new internal type part of the public API.

The reconciler retains its exact 32-sweep ceiling, operator order, per-op
guards, dynamic-signature preservation, constant updates, and update counter.
When a caller supplies `ModelIRGraphIndex`, the producer table is derived from
that shared index; standalone callers retain the historical producer-table
construction. The pass does not mutate topology and performs no consumer-map
scan. Extraction therefore removes the multi-op decision tree from central
lowering context without changing convergence semantics or pass scheduling.

Tier 3 `rf-detr-nano.onnx` establishes real ownership across 29 production
invocations. Six are non-zero with update counts `141,16,16,138,16,6`, for
333 total tensor updates. All 29 counts and every float32, float16, and
correspondence artifact are identical across extraction, with zero monitored
process-tree SWAP. IAT-LLIE provides a 29-invocation zero-owner control.

HARD_SWISH metadata sanitation is isolated in
`hardswish_shape_sanitization.py`. The lowerer retains the historical private
wrapper, while the module owns the invariant that HARD_SWISH output shape and
signature match the input and that a fully static input receives a fully
static signature. A valid shared `ModelIRGraphIndex` limits traversal to
HARD_SWISH operators; standalone callers retain the original single list
traversal. The pass is metadata-only and performs no topology, cleanup,
layout-state, lineage, or constant-buffer mutation.

All root models that directly contain ONNX `HardSwish` were measured before
the move. Their production invocations were zero-update controls and all
strictly sequential conversions recorded process-tree SWAP zero. The existing
synthetic stale-metadata fixture supplies the positive ownership contract.
Tier 1 `inference_ops15.onnx` is the fixed artifact representative: both
production invocations remain zero and its float32, float16, and correspondence
artifacts are byte-identical across extraction. This mechanical boundary must
not be broadened into semantic inference without first recording a real owner
and an independent regression contract.

Rank-four channelwise broadcast-constant repair is now owned by
`binary_layout_adapter.py` alongside the two related binary compatibility
policies. The complete former lowerer implementation moved with an AST that is
identical after normalizing only the function name. The lowerer retains the
historical private wrapper, three direct `lower_onnx_to_ir` calls, and the one
call inside three-round binary-layout convergence, so scheduling and statistics
aggregation are unchanged.

The owner reuses a caller-supplied `ModelIRGraphIndex` when it belongs to the
same ModelIR and otherwise builds exactly one index. Candidate traversal is
limited to indexed binary operators. Producer/layout hints still resolve
ambiguous rank-three and rank-four constants; exclusive constants mutate in
place, while shared constants use snapshot-based copy-on-write and update the
index through `_set_operator_inputs`. Standard NCHW-to-NHWC rotation, exact
inverse recovery for stale NHWC constants, dtype and quantization preservation,
unique clone naming, and the historical counter are unchanged.

The positive, no-op, shared-constant, GraphIndex-differential, and convergence
contracts pass, including module-owner versus lowerer-wrapper full ModelIR
fingerprint equality. FastestDet supplies the strictly sequential zero-owner
artifact control: all five invocations remain zero, process-tree SWAP is zero,
and float32, float16, tensor-correspondence, schema, and generated-schema
artifacts are byte-identical across extraction. Positive production ownership
is not claimed; the synthetic cases remain the semantic authority.

Conv/Pool output passthrough compatibility is now owned by
`convpool_output_passthrough_compat.py`. The complete corrected 556-line helper
moved with a function-name-normalized AST identical to its prior lowerer
implementation. The lowerer retains one private wrapper at the unchanged
single production position. Its former sole giant direct-builder fixture moved
to
`test_flatbuffer_direct_convpool_output_passthrough_layout.py`, where a compact
contract now covers the elementwise region, retained legacy NCHW adapter,
rank-four external-runtime adapter, keepdims Mean-axis absorption, and seven
unsafe-boundary no-ops. Every case compares the module owner and lowerer wrapper
on complete ModelIR fingerprints. The architecture gate explicitly records the
current whole-graph maps, direct append/delete mutation, cleanup boundary, one
wrapper dispatch, and one production call.

Characterization exposed and the following checkpoint corrected one unsafe
rejection path. Every external runtime tensor and its projected NHWC shape are
now validated into an immutable local plan before channel-last hints, rewiring,
adapter creation, metadata mutation, or topology mutation. A candidate with a
valid first external input and invalid later rank-three input is therefore a
complete ModelIR no-op, and the former strict xfail is an ordinary passing
atomicity contract. Successful rewrite order and artifacts are unchanged.
FastestDet, HumanSeg, OSNet, and inference_ops15 each produce one zero result in
strictly sequential, zero-SWAP traces; positive production ownership is not
claimed.

The adjacent quantized Mean/HardSigmoid/MulAdd recovery is now owned by
`mean_hardsigmoid_muladd_layout.py`. Its 496-line function body is AST-identical
to the corrected lowerer predecessor after function-name normalization. The
lowerer retains a two-line private wrapper and one syntactic call inside the
recovery sequence, executed at two ordered production boundaries. Its dedicated
ModelIR contract fixes the full two-branch graph, Mean axis remap, decomposed
HardSigmoid clamp, residual Mul/Add rewiring, three bridge removals, legacy-
output adapter, idempotence, and eight complete no-op guards. Architecture
tests record module ownership, wrapper dispatch, current full-map scans,
constant writes, direct insertion/deletion, prune, and the single production
call.

The Mean-axis rejection path is now atomic: rank normalization and the constant
update occur after all candidate guards but before the first graph rewiring or
dependent metadata mutation. Out-of-range axes and a rejected no-change
constant update both retain a complete ModelIR fingerprint and zero statistic.
A public residual `add0_out` is now rejected before the axes write, so its
declared NCHW output contract and full ModelIR fingerprint remain unchanged.
The focused contract has no remaining strict xfail. YuNet INT8, PPHumanSeg
INT8, and SSD MobileNet INT8 each produce two zero results in current strictly
sequential, zero-SWAP traces, matching the earlier broader zero-owner survey.
The completed ownership move retains those boundaries and adds a direct owner/
wrapper ModelIR fingerprint comparison.

The following QLinear Concat/Conv propagation is now owned by
`qlinear_concat_conv_compat.py`. Its corrected 612-line function body is
AST-identical to the lowerer predecessor after normalizing only the function
name. The lowerer retains a two-line private wrapper, one syntactic call in the
same recovery sequence, and two ordered runtime boundaries. Its dedicated
contract covers Pattern 1 quantized pre-Transposes, Pattern 2 float pre-
Transpose before Quantize, Pattern 3 singleton Reshape before Quantize, Pattern
4 singleton-spatial metadata reinterpretation, multiple output adapters, a
direct Concat adapter, dynamic batch signatures, per-axis qdim remap,
idempotence, and nine rejection guards. The former 119-line giant ModelIR
fixture is now in the focused qlinear module with an identical AST. Direct
owner and compatibility-wrapper calls produce identical complete ModelIR
fingerprints and statistics.

Required `concat_out` and `q_out` tensors are now prevalidated after all
prospective input shapes and before the first input or metadata mutation. Both
missing-tensor cases retain a complete ModelIR fingerprint and zero statistic.
A pending tensor-shape update whose tensor is public is now rejected before
axis validation or mutation, preserving public Dequantize outputs that would
change from NCHW to NHWC. An already-NHWC public Dequantize output with no
pending update remains eligible, retaining the safe feature. No strict xfail
remains. The established eight-model QLinear recovery survey recorded zero
rewrites for this helper, so positive production ownership is not claimed.

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

Expected or repeated quick-ceiling timeouts follow the same evidence-first
policy. Preserve the complete timed-out run before changing a profile; compare
completed artifacts and prior successful accuracy evidence to distinguish
runtime variance from a semantic regression. A model that is no longer
reliably short is retained as excluded history with a normalized reason such
as `repeated_quick_ceiling_timeout`. Do not increase the quick ceiling or
change converter source merely to keep that model in the short-runtime set.

## Raw Softmax/Transpose canonicalizer characterization

The remaining raw `_canonicalize_softmax_transpose_chains` owner is frozen at
its two existing ordered production boundaries before any implementation
change. Its positive contract matches the private chain
`NHWC-to-NCHW Transpose -> NCHW-to-NWHC Transpose -> last-axis Softmax ->
NCHW-to-NWHC Transpose`. The existing rewrite changes the two inner
permutations to NCHW-to-NHWC and NHWC-to-NCHW, adds the shared terminal-
cleanup marker, preserves all other Softmax options/provenance, accepts a
terminal public output only when it has no internal consumer, clones a shared
permutation buffer, processes independent branches in graph order, reaches a
fixed point, and retains historical unused-tensor pruning on a zero-match
graph.

The characterization records 24 concrete unsafe cases as strict xfails. One
case exposes incomplete metadata planning: the Softmax input is changed to
NHWC, but the Softmax output keeps its former NWHC metadata and the post-
Transpose is consequently assigned an H/W-swapped shape. Six cases prove that
the rewrite accepts non-last, out-of-range, and malformed Softmax axes even
though only normalized axis three preserves the chain's meaning. Seven cases
show that missing or non-rank-four source/intermediate/destination metadata is
accepted. Five cases show in-place mutation of a public-input, variable,
wrong-TensorIR-dtype, wrong-buffer-dtype, or quantized permutation tensor. A
public constant output is also mutated instead of being preserved through a
private clone. The final four cases cover duplicate Softmax/post producers,
reverse Softmax/post order, and an internally produced tensor also declared as
a public input. Every rejection contract requires a zero statistic and an
unchanged complete ModelIR state.

Production source is intentionally unchanged at this checkpoint. Correction
must resolve every required rank-four shape/signature, normalized last-axis
Softmax semantics, unique and topologically ordered producers, private
intermediates, and a complete immutable permutation update/clone plan before
the first tensor, operator, option, lineage, or metadata mutation. Valid
graph-order statistics, fixed-point behavior, marker sharing, pruning,
terminal-output behavior, and both ordered runtime boundaries must remain
unchanged.

The correction is now implemented in the raw owner. A single
`ModelIRGraphIndex` replaces consumer/producer-map reconstruction on every
fixed-point round. The matcher rejects duplicate producers, reverse producer/
consumer order, produced public inputs, non-last or malformed Softmax axes,
per-axis activation quantization, and any missing or non-rank-four required
shape/signature before planning mutations. Softmax input and output metadata
are both replanned as NHWC, and the post-Transpose metadata is derived from
that new output rather than from the stale NWHC shape.

Both permutation actions are planned together. A candidate may update only an
immutable local unquantized INT32 tensor with an INT32 backing buffer. Shared
or public-output constants receive deterministic private clones; public
inputs and variable constants reject. Candidate-local name reservations are
published only when the complete plan commits, so a rejected candidate cannot
change a later clone name. After both plans, all metadata, and marker options
exist, the commit phase has no rejection branch. All 24 former strict xfails
are green, including a post-permutation failure that proves the pre-
permutation remains unchanged. Normalized axis `-1`, existing axis `3`, shared
buffer cloning, public-output cloning, graph-order statistics, fixed-point
behavior, terminal outputs, pruning, and both production boundaries remain
covered.

Ownership now resides in
`passes/softmax_transpose_canonicalization.py`. The extracted function and the
corrected raw owner at checkpoint `9a9898e3` are each 343 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias
and retains the historical function name as a one-return compatibility
wrapper. Both nested recovery-sequence positions remain unchanged, and the
module imports the shared terminal marker from its existing owner without a
lowerer import cycle. Direct owner/wrapper comparison covers static shapes with
dynamic signatures, multiple branches, shared and public-output permutation
clones, negative last axis, terminal output, pruning, incomplete metadata,
unsafe axis, and a post-plan atomic rejection. Statistics and complete
normalized ModelIR state are identical in every case.

## Raw Concat/Mul/Transpose/Add bridge characterization

The next raw source-order owner is the 394-line
`_optimize_concat_mul_add_transpose_nhwc_bridge_chains`. It remains unchanged
at the third position of both terminal Concat recovery sequences. Its former
ordinary and legacy-consumer fixtures have been moved out of the giant direct
test module into a focused contract, reducing that central file by 211 lines.
The focused contract freezes ordinary static and dynamic-signature rewrites,
legacy-consumer and public-Concat-output adapters, graph-order multiple
matches, fixed-point behavior, scalar and rotated Mul constants, shared-
constant collision-safe cloning, zero-match no-prune behavior, operator
options/provenance, nine existing rejection guards, statistics, and both
ordered production boundaries.

Sixteen reproduced safety gaps are strict xfails. A legacy adapter is appended
after its consumer and therefore leaves the ModelIR non-topological. Five
missing, rank-three, or short-signature retained tensors are accepted. A
public-input or variable Mul constant is rotated in place, while a public
constant output is not preserved through a private clone. Ordinary and legacy
per-axis cases retain NCHW quantized dimension one after their tensors move to
NHWC instead of remapping it to three. The reserved adapter-permutation name
can overwrite a public input, and malformed legacy metadata raises after the
Mul constant has already changed. Duplicate post producers, reverse post/Add
order, and a produced pre-adapter tensor also declared as a public input are
also rewritten. Every rejection or atomicity contract compares the complete
normalized ModelIR state.

Production source is intentionally unchanged. Correction must build a unique,
topologically ordered chain plan; validate complete rank-four effective
metadata; plan every constant update or clone, QDIM remap, canonical tensor,
adapter constant/name, operator setter, removal, and adapter insertion before
the first mutation; and insert any compatibility adapter before its earliest
consumer. Public inputs and variables must reject, public outputs must remain
stable, and rejected candidates must not reserve names or emit lineage. Valid
candidate order, statistics, scalar handling, collision behavior, fixed point,
pruning, public Concat outputs, and both ordered runtime boundaries must remain
unchanged.

The correction is now implemented in the 652-line raw owner. One
`ModelIRGraphIndex` replaces full producer/consumer-map reconstruction in each
fixed-point round and is updated differentially through every setter, removal,
and adapter insertion. The complete candidate plan requires unique producers,
strict pre-Transpose/Concat/Mul/post-Transpose/Add order, private internal
edges, rank-four source/Concat/Mul metadata, valid broadcast constants, and a
complete immutable name set before mutation.

Mul constants are classified as unchanged scalar/broadcast values, private
in-place rotations, or collision-safe clones for shared and public-output
ownership. Public inputs and variables reject. NCHW-to-NHWC QDIM is cloned and
remapped for the Concat tensor, Mul output, and rotated Mul constant, including
the legacy canonical tensor. Adapter-permutation constants are reused only
when immutable and safe; otherwise a private collision-safe INT32 constant is
planned without overwriting the reserved name. Canonical Concat metadata,
options, all setters, the removal set, and adapter placement are also planned
before commit. Compatibility adapters are inserted directly after their
Concat producer and therefore precede every legacy consumer.

All 16 former strict xfails are green. Additional coverage proves one graph-
index construction for two matches and validates complete ModelIR invariants
for legacy/public-output adapters plus ordinary/legacy per-axis quantization.
The nine former rejections, scalar and shared constants, dynamic signatures,
multiple-match count, fixed point, no-match no-prune behavior, options/
provenance, and both ordered production boundaries remain green. Removing the
obsolete local `add_out_name` assignment reduces pre-existing lowerer Ruff
findings from seven to six.

Ownership now resides in
`passes/concat_mul_add_bridge_layout.py`. The extracted function and the
corrected raw owner at checkpoint `5193fc11` are each 652 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias
and retains the historical function name as a one-return compatibility
wrapper. Both terminal recovery-sequence positions and their immediate
predecessor/successor calls remain unchanged, and the focused module has no
lowerer import.

Direct owner/wrapper comparison covers ordinary static and dynamic metadata,
multiple matches, scalar constants, shared and public-output constant clones,
legacy and public-Concat adapters, ordinary and legacy per-axis quantization,
adapter-name collision, unmatched pruning behavior, missing metadata, reverse
topology, and a public internal boundary. Statistics and the complete
normalized ModelIR—including buffers, quantization, options, provenance,
topology, and diagnostics—are identical in every case. The extraction adds no
semantic change and does not alter public APIs, artifacts, dependencies,
corpus policy, or TensorFlow isolation.

## Raw Concat/Mul/Add/Transpose/Add bridge characterization

The next raw source-order owner is the 452-line
`_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains`. It remains
unchanged at the fourth position of both terminal Concat recovery sequences.
Its existing legacy-consumer fixture has moved from the giant direct test into
`test_flatbuffer_direct_concat_mul_add_transpose_add_bridge_layout.py`,
reducing the central test by 102 lines while preserving the same observable
contract.

The focused positive contract covers ordinary static and dynamic-batch
signatures, legacy compatibility output, two independent graph-order matches,
second-call fixed point, scalar Mul/Add constants, collision-safe cloning of
each shared affine constant, zero-match no-prune behavior, Concat options,
axis semantics, version and ONNX provenance, nine existing rejection guards,
statistics, and both ordered production boundaries.

Twenty-seven concrete safety gaps are strict xfails:

- the legacy adapter is appended after its existing consumer;
- seven missing, rank-three, or short-signature source/Concat/Mul/Add metadata
  cases still rewrite;
- public-input, variable, and public-output ownership is ignored for each of
  the Mul and pre-Transpose Add constants;
- ordinary and legacy per-axis tensors keep NCHW QDIM 1 after moving to NHWC;
- a public, variable, wrong-dtype, quantized, or wrong-value reserved adapter
  constant is reused or overwritten instead of preserving it and allocating a
  private INT32 permutation;
- invalid Add-constant or legacy-signature evidence discovered after Mul-
  constant rotation leaves partial mutation;
- a malformed Concat axis raises instead of producing a transactional no-op;
- duplicate post output producers, reverse post/tail-Add order, and a produced
  pre-Transpose tensor declared as a public input are accepted.

Production source is intentionally unchanged at this checkpoint. Correction
must build one indexed graph-order candidate plan that validates unique
producers, strict operator order, private internal edges, complete rank-four
shape/signature metadata, both affine-constant ownership and broadcasts,
per-axis QDIM remaps, adapter ownership/naming, and every setter, removal, and
insertion before the first mutation. Compatibility adapters must be inserted
before their earliest legacy consumer. Existing match order, statistics,
fixed point, scalar/shared constants, pruning, options/provenance, and both
runtime boundaries must remain unchanged.

The correction is now implemented in the 866-line raw owner. One
`ModelIRGraphIndex` replaces producer/consumer-map reconstruction on every
fixed-point round and is updated differentially through input/output setters,
batched removals, and adapter insertion. Two independent matches construct
the index once.

Each candidate now proves the strict pre-Transpose/Concat/Mul/Add/post-
Transpose/tail-Add order, unique retained producers, exact single-consumer
internal edges, private intermediate boundaries, complete rank-four source/
Concat/Mul/Add shapes and effective signatures, a valid Concat axis/options
mapping, and a broadcast-compatible NHWC tail constant before mutation.
Missing tensors, short signatures, malformed axes, duplicate producers,
reverse order, and public internal aliases return zero with complete ModelIR
equality.

Both affine constants are planned from the unchanged graph. Scalars and
already NHWC-broadcastable values remain untouched. Rotated constants require
immutable non-public-input ownership and valid target broadcasting; shared or
public-output values receive deterministic private clones, while variables and
public inputs reject. Per-axis quantization is cloned and remapped with the
same NCHW-to-NHWC permutation for both constants and the canonical Concat,
Mul-output, and Add-output tensors. Candidate-local names are published only
after every constant, metadata, quantization, adapter, setter, removal, and
insertion decision succeeds.

The reserved adapter tensor is reusable only as a private, immutable,
unquantized INT32 `[4]` constant with an INT32 buffer and the exact
permutation. Every unsafe collision is preserved and receives a private
collision-safe replacement. Legacy adapters are inserted immediately after
their Concat producer, before the main affine chain and every legacy consumer.
All 27 former strict xfails are green; public-output affine constants are
explicitly verified as successful private-clone rewrites. The original 18
characterization cases, plus clone and one-index coverage, now form 46 green
focused tests.

Ownership now resides in
`passes/concat_mul_add_transpose_add_bridge_layout.py`. The extracted function
and the corrected raw owner at checkpoint `4a5f0394` are each 866 lines and
have identical ASTs. The lowerer imports the module owner under a private pass
alias and retains the historical function name as a one-return compatibility
wrapper. Both terminal recovery-sequence positions and their immediate
neighbors remain unchanged, and the focused module has no lowerer import.

Nineteen direct owner/wrapper comparisons cover ordinary static and dynamic
metadata, multiple matches, scalar constants, separate shared Mul/Add
constant collisions, separate public-output clones, legacy adapters, ordinary
and legacy per-axis quantization, adapter collision, unmatched pruning,
missing and malformed metadata, late constant evidence, malformed axis,
reverse topology, and a public internal boundary. Statistics and complete
normalized ModelIR state are identical in every case. The mechanical move
does not alter public APIs, artifacts, dependencies, corpus policy, ordered
runtime behavior, or TensorFlow isolation.

## Raw Concat/Mul/Add/Add/Mean/Reshape characterization

The next raw source-order owner is the 461-line
`_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains`. It remains
unchanged at the fifth position of both terminal Concat recovery sequences.
Its existing positive fixture moved from the giant direct test into
`test_flatbuffer_direct_concat_mul_add_add_mean_reshape_layout.py`, reducing
the central test by 94 lines while preserving the public behavior.

The focused positive contract covers ordinary static and dynamic-batch
signatures, two independent graph-order matches, fixed point, scalar affine
constants, collision-safe shared cloning for all three affine constants,
shared Mean-axes cloning, exact old-Mean-shape rewriting, zero-match no-prune
behavior, Concat and Mean options/version/provenance, ten existing rejection
guards, statistics, and both ordered production boundaries.

Forty-two concrete safety gaps are strict xfails:

- eleven missing, rank-three, or short-signature source/Concat/Mul/Add/Mean
  metadata cases still rewrite;
- six public-input or variable affine constants rotate in place and three
  public affine outputs are not preserved through private clones;
- per-axis QDIM is not remapped for the three constants or five NHWC tensors;
- five unsafe Mean-axes ownership/dtype/buffer/quantization cases are accepted,
  and a public axes output is mutated rather than cloned;
- four unsafe Reshape-shape ownership/type cases are accepted, and shared or
  public-output shapes are mutated rather than cloned;
- invalid second/third affine constants, invalid axes, and malformed Mean
  metadata are discovered only after earlier constants have changed;
- a malformed Concat axis raises instead of producing a no-op;
- duplicate Mean producers, reverse Mean/Reshape order, and a produced public
  input alias are accepted;
- an identity Mean-axis mapping is treated as rewrite failure after affine
  constants have already changed.

Production source is intentionally unchanged. Correction must build one
indexed, strictly ordered candidate plan with complete rank-four effective
metadata and immutable plans for all three affine constants, Mean axes, and
the conditionally rewritten Reshape shape. Shared and public-output constants
must clone, public inputs and variables must reject when mutation is required,
INT32 shape/axes contracts must be explicit, QDIM must follow each permutation,
and an unchanged axes mapping must count as a valid plan. Every name, tensor,
setter, metadata update, removal, and pruning decision must be known before the
first mutation. Existing match order, statistics, fixed point, scalar/shared
handling, exact shape rewrite, provenance, and both runtime boundaries must
remain unchanged.

The correction is now implemented in the 869-line raw owner. One
`ModelIRGraphIndex` replaces the repeated complete producer/consumer scans and
is maintained through indexed setters and batched pre-Transpose removal. Two
independent matches construct it once.

Each candidate proves strict pre-Transpose/Concat/Mul/Add/Add/Mean/Reshape
order, unique retained producers, exact single-consumer internal edges,
private intermediates, complete rank-four source and retained tensor shapes/
effective signatures, valid Concat and Mean options, and Concat fan-out safety
before mutation. Missing or short metadata, malformed axes, duplicate
producers, reverse order, and public aliases return zero with complete ModelIR
equality.

All three affine constants use the same immutable plan: scalar and existing
NHWC broadcasts remain unchanged; NCHW rank-three/four values rotate only
after target-broadcast and ownership validation; shared and public-output
values clone; public inputs and variables reject. Constant QDIM follows the
rank-specific permutation, while Concat, Mul, both Add outputs, and Mean output
receive the rank-four QDIM remap.

Mean axes are normalized and remapped into a validated immutable unquantized
INT32 plan. Identity mappings are valid no-change actions. Changed shared or
public-output axes clone, while public inputs, variables, wrong TensorIR or
buffer dtype, and quantized tensors reject. Reshape shape is planned only when
its four values exactly equal the old Mean shape; that plan has the same INT32
and ownership contract and clones shared/public outputs. Every constant,
quantization, name, tensor, setter, metadata result, and removal is complete
before the first mutation. All 42 former strict xfails are green, and an
explicit Concat fan-out guard plus one-index test bring the focused contract to
65 green cases.

Ownership now resides in
`passes/concat_mul_add_add_mean_reshape_layout.py`. The extracted function and
the corrected raw owner at checkpoint `3c3579fd` are each 869 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias
and retains the historical function name as a one-return compatibility
wrapper. Both terminal recovery-sequence positions and their immediate
neighbors remain unchanged, and the focused module has no lowerer import.

Twenty-three direct owner/wrapper comparisons cover static and dynamic
metadata, multiple matches, scalar constants, separate shared and public-
output actions for all three affine constants, shared/public Mean axes, exact
and public Reshape shapes, QDIM, identity axes, unmatched pruning, missing and
late metadata, late affine evidence, malformed axis, reverse topology, and a
public internal boundary. Statistics and complete normalized ModelIR state are
identical in every case. The move adds no semantic change and does not alter
public APIs, artifacts, dependencies, corpus policy, ordered behavior, or
TensorFlow isolation.

## Raw nested-Concat/Mul/Transpose characterization

The next raw source-order owner is the 356-line
`_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains`. Production code
remains unchanged at its historical position immediately after the extracted
Concat/Mul/Add/Add/Mean/Reshape recovery and before singleton-gate recovery in
both terminal Concat sequences. The original public mixed-axis fixture moved
from the giant direct test into
`test_flatbuffer_direct_concat_tree_mul_add_bridge_layout.py`; the move removes
the associated private lowerer import from the giant test without changing
runtime behavior.

Twenty green characterization cases freeze static and dynamic-batch mixed-axis
Concat trees, graph-order multiple matches, fixed point, scalar and shared Mul
constants, normalized negative axes, zero-match no-prune behavior, twelve
existing rejection guards, the raw owner's current three-loop structure, and
both ordered production boundaries.

Nineteen concrete safety gaps are strict xfails:

- eight missing, rank-three, or short-signature source/Concat/Mul metadata
  cases are not rejected through a complete rank-four preflight;
- public-input and variable Mul constants rotate in place, and a public Mul
  constant output is changed rather than receiving a private clone;
- per-axis QDIM does not follow the NCHW-to-NHWC permutation;
- malformed inner or root Concat axes raise instead of producing an atomic
  no-op;
- late inner metadata failure occurs after the Mul constant has changed;
- duplicate post-Transpose producers, reverse post-Transpose/Add order,
  reverse inner/root Concat order, and a public pre-Transpose alias are
  accepted.

Correction must use one `ModelIRGraphIndex` to build the complete recursive
tree, topology, metadata, constant-ownership, quantization, setter, and removal
plan before the first mutation. Candidate enumeration and recursive tree order,
valid statistics, scalar/shared handling, fixed point, pruning behavior, and
both production boundaries must remain unchanged. The 356-line count records
the current owner size only; the 2,000 threshold in this project applies to
ONNX operation-count tiers, not source-file or function length.

That correction is now implemented in the 675-line raw owner. One
`ModelIRGraphIndex` replaces the repeated producer/consumer reconstruction and
is maintained by indexed input setters plus one batched adapter removal. Two
independent matches construct and refresh the index exactly once.

Candidate planning proves unique producers, strict pre-Transpose/nested-
Concat/Mul/post-Transpose/Add order, exact internal consumers, private bridge
tensors, valid normalized axes, and complete rank-four source, every Concat
output, and Mul-output shape/signature metadata. Every nested Concat input and
axis update, output metadata permutation, QDIM result, Add rewire, and removal
is immutable before commit. Missing tensors, rank-three sources, short or
malformed signatures, malformed axes, duplicate producers, reverse topology,
and public aliases now return zero with complete ModelIR equality.

The Mul constant has a separate immutable ownership plan. Scalars and values
already broadcastable in NHWC remain unchanged. A required NCHW-to-NHWC
rotation rejects public inputs and variables, updates a private single-use
constant, or creates a deterministic collision-safe clone for unrelated users
and public outputs. Per-axis QDIM follows the same permutation for the constant,
every nested Concat output, and the Mul output. The Add-side NHWC channel
broadcast is also checked against the planned Mul shape. All nineteen former
strict xfails are green. Explicit one-index, reverse source/adapter order, and
duplicate source-producer contracts bring the focused suite to forty-two cases
without changing valid statistics, fixed point, pruning, or either production
boundary.

Ownership now resides in
`passes/concat_tree_mul_add_bridge_layout.py`. The extracted owner and the
corrected raw owner at checkpoint `4111187c` are each 675 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias
and keeps the historical private name as a one-return compatibility wrapper.
Both terminal recovery-sequence positions and their immediate neighboring calls
are unchanged, and the focused module does not import the lowerer.

Seventeen direct owner/wrapper comparisons cover static and dynamic metadata,
multiple matches, scalar, shared, public-output, public-input, and variable Mul
constants, per-axis quantization, unmatched pruning, missing and late metadata,
malformed axes, reverse nested topology, public internal boundaries, and
reverse or duplicate source producers. Statistics and complete normalized
ModelIR state are identical in every case. The move adds no semantic change and
does not alter public APIs, artifacts, dependencies, corpus policy, ordered
behavior, or TensorFlow isolation.

## Raw StridedSlice/Pad/Concat bridge characterization

The next raw source-order owner is the 543-line
`_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains`.
Production code and all three historical call sites remain unchanged. Its
public fixture moved from the giant direct test into
`test_flatbuffer_direct_stridedslice_pad_concat_bridge_layout.py`, removing the
associated private lowerer import and 117 fixture lines from the central test.

Twenty-six green characterization cases freeze static and dynamic signatures,
two independent matches and fixed point, multiple Add users, ordinary Pad and
MirrorPad provenance/options, scalar Mul constants, collision-safe cloning of
shared Slice-end/Pad/Mul constants, zero-match no-prune behavior, seventeen
existing rejection guards, statistics, the raw owner's current two-loop shape,
and all three production calls.

Forty-two concrete safety gaps are strict xfails:

- ten missing, rank-three, or short-signature source/Slice/Pad/Concat/Mul
  metadata cases still rewrite;
- sixteen public-input, variable, wrong-dtype, or quantized Slice begin/end/
  stride and Pad constants lack an immutable typed ownership plan;
- changed public Slice-end and Pad constants mutate rather than clone;
- public-input and variable Mul constants rotate in place, and a public Mul
  output is not preserved through a private clone;
- per-axis QDIM is not remapped for Slice, Pad, Concat, Mul constant, or renamed
  Mul output tensors;
- public Slice and Pad intermediates change physical layout without an adapter
  or complete rejection;
- duplicate post producers, reverse post/Add or Slice/Pad order, a public pre-
  Transpose alias, and reverse or duplicate source producers are accepted;
- malformed Concat axes and Slice masks raise rather than producing an atomic
  no-op.

Correction must build one `ModelIRGraphIndex` and a complete immutable plan for
every branch before the first mutation. The plan must prove strict pre-
Transpose/StridedSlice/Pad/Concat/Mul/post-Transpose/Add order, unique
producers, exact private consumers, complete rank-four metadata, normalized
options, and the supported multi-Add tail. Slice vectors and Pad matrices must
have explicit unquantized INT32 metadata and be grouped by tensor identity and
target value; public inputs and variables must reject when treated as constants,
while unrelated users and changed public outputs receive deterministic clones.
The Mul constant needs the same ownership policy, and all retained per-axis
QDIM values must follow the data permutation. Every clone name, constant action,
metadata/QDIM result, input/output setter, mask/axis option, Mul-output rename,
and adapter removal must be known before commit. Valid statistics, Pad and
MirrorPad behavior, multiple Add users, fixed point, pruning, and all three call
boundaries must remain unchanged. The 543-line count is descriptive only; 2,000
remains the ONNX operation-count tier threshold.

That correction is now implemented in the 1,100-line raw owner. One
`ModelIRGraphIndex` replaces every repeated producer/consumer reconstruction
and remains current through indexed input/output setters plus one batched
adapter removal. Two independent matches construct and refresh it exactly once.

Each candidate proves strict source/pre-Transpose/StridedSlice/Pad/Concat/Mul/
post-Transpose/ordered-Add topology, unique producers, exact private internal
consumers, valid normalized options, complete rank-four source and retained-
output metadata, and the supported one-or-more Add tail. Slice, Pad, Concat,
and renamed Mul-output shape/signature/QDIM results and every Add broadcast are
complete before commit. Missing post-output tensors, malformed axes or masks,
duplicate producers, reverse order, and public aliases now leave the complete
ModelIR unchanged.

Slice begin/end/stride vectors and Pad matrices use one grouped immutable
constant transaction. Each tensor must be an unquantized, non-variable INT32
buffer with exact shape/signature and no runtime producer or public-input
ownership. Requirements are grouped by tensor identity and must agree on the
target value. An unchanged valid constant remains shared; a changed private
constant updates once; any unrelated consumer edge or public output receives
one deterministic collision-safe clone reused by all planned sites. The Mul
constant has the same update/clone/reject policy, including per-axis QDIM and
provenance-preserving clones.

All forty-two former strict xfails are green. Missing post-output rejection and
explicit one-index reuse bring the focused contract to seventy cases without
changing Pad/MirrorPad options, multiple Add users, statistics, fixed point,
pruning, or any of the three production calls.

Ownership now resides in
`passes/stridedslice_pad_concat_bridge_layout.py`. The extracted owner and the
corrected raw owner at checkpoint `95a5555b` are each 1,100 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias
and retains the historical private name as a one-return compatibility wrapper.
All three production calls are unchanged, and the focused module does not
import the lowerer.

Twenty direct owner/wrapper comparisons cover static and dynamic metadata,
multiple matches, multiple Add users, Pad and MirrorPad, scalar constants,
grouped shared constants, public-output/index-input/wrong-dtype index
ownership, public and variable Mul constants, per-axis quantization, unmatched
pruning, missing retained/post metadata, malformed options, reverse topology,
public intermediates, and duplicate source producers. Statistics and complete
normalized ModelIR state are identical in every case. The mechanical move does
not alter public APIs, artifacts, dependencies, corpus policy, ordered runtime
behavior, or TensorFlow isolation.

## Raw Reshape/Transpose collapse characterization

The next substantive raw source-order owner is the 218-line
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains`.
Production code and both historical calls remain unchanged. It previously had
only ordered architecture references; the focused
`test_flatbuffer_direct_reshape_transpose_collapse_layout.py` now owns its
synthetic ModelIR contract.

Fourteen green cases freeze the static rank-three to NHWC reshape collapse,
two independent matches and fixed point, collision-safe shared shape cloning,
ten existing permutation/public-boundary/fan-out/shape rejection guards,
statistics, the raw owner's current two-loop structure, and both production
calls.

Nineteen concrete safety gaps are strict xfails:

- dynamic batch signatures are collapsed into a concrete batch-one shape
  constant and options;
- a zero-match invocation still prunes unrelated tensors;
- public-input, variable, wrong-TensorIR-dtype, wrong-buffer-dtype, quantized,
  or data-less reshape-shape tensors are treated as mutable constants;
- a changed public shape output mutates rather than receiving a private clone;
- short input/intermediate/output signatures are ignored;
- duplicate output or source producers, reverse Transpose/Reshape order, a
  public internal input alias, and a reverse source producer are accepted.

Correction must use one `ModelIRGraphIndex` to prove unique producers, strict
Reshape/Transpose/Reshape/Transpose order, exact private internal consumers,
and complete rank-three/rank-four shape and signature metadata before mutation.
The target shape constant and `newShape`/`onnxRawNewShape` options must preserve
a compatible dynamic batch as `-1`. The shape tensor needs an explicit
unquantized INT32 ownership/type contract: public inputs, variables, missing
data, runtime producers, and invalid metadata reject; unrelated users and
changed public outputs clone through deterministic reserved-name allocation.
The output setter, option values, shape action, removals, and pruning decision
must all be known before commit. A no-match call must be a complete no-op.
Valid static behavior, statistics, fixed point, provenance/options, and both
production calls must remain unchanged. The 218-line count is descriptive
only; 2,000 remains the ONNX operation-count tier threshold.

That correction is now implemented in the 399-line raw owner. One
`ModelIRGraphIndex` replaces the repeated consumer scan and stays current
through the indexed Reshape input/output setters and one batched removal. Two
independent matches construct and refresh it exactly once.

Each candidate proves unique source, intermediate, and final producers; strict
Reshape/Transpose/Reshape/Transpose graph order; exact private internal
consumers; complete positive physical shapes; and compatible rank-three/rank-
four signatures. Non-batch signature dimensions must agree with the proven
physical shape. Compatible concrete/dynamic batch signatures produce one
target batch value, using `-1` whenever any boundary remains dynamic, and the
same planned value updates the shape buffer and both list-valued Reshape
options.

The shape input is now an explicit immutable unquantized INT32 contract with
exact TensorIR, buffer, shape, signature, ownership, and original-value checks.
Public inputs, variables, runtime producers, missing data, invalid dtypes, and
quantized values reject. A changed private shape updates once; an unrelated
consumer or public output receives a deterministic collision-safe clone that
preserves layout and ONNX provenance. Shape action, options, indexed output
rename, removal set, and prune decision are complete before mutation. All
nineteen former strict xfails are green, zero-match execution no longer prunes,
and the explicit one-index contract brings the focused suite to thirty-four
cases without changing valid static behavior, statistics, fixed point, or both
production calls.

Ownership now resides in
`passes/reshape_transpose_collapse_layout.py`. The extracted owner and the
corrected raw owner at checkpoint `48aae4b0` are each 399 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias,
retains the historical private name as a one-return compatibility wrapper, and
keeps both production calls unchanged. The focused module does not import the
lowerer.

Sixteen direct owner/wrapper comparisons cover static and dynamic metadata,
multiple matches, shared and public shape outputs, public-input, variable,
wrong-dtype, quantized, and missing-data shape rejection, zero-match no-prune,
short signatures, reverse topology, a public internal alias, duplicate source
producers, and missing output metadata. Statistics and complete normalized
ModelIR state are identical in every case. The mechanical move does not alter
public APIs, artifacts, dependencies, corpus policy, ordered runtime behavior,
or TensorFlow isolation.

## Raw attention Gather cleanup characterization

The next substantive raw source-order owner is the 293-line
`_optimize_attention_gather_transpose_reshape_cleanup_chains`. Production code
and both historical calls remain unchanged. It previously had only ordered
architecture references; the focused
`test_flatbuffer_direct_attention_gather_cleanup_layout.py` now owns its
synthetic ModelIR contract.

Thirty-three green cases freeze both supported rewrites: the rank-four
Gather/Transpose/Reshape attention tail, the double-Gather/Reshape identity,
negative-axis normalization, exact NumPy equality, mixed multiple matches and
fixed point, collision-safe shared-permutation naming, public Pattern-A
Reshape outputs, existing public-intermediate/fan-out/shape/axis/constant
rejections, statistics, the raw owner's current two-loop structure, and both
production calls.

Forty-six concrete safety gaps are strict xfails:

- a zero-match invocation prunes an unrelated tensor;
- thirty public-input, variable, wrong-TensorIR-dtype, wrong-buffer-dtype, or
  quantized index/permutation/shape cases are accepted as compile-time
  constants;
- a changed public permutation output mutates rather than receiving a private
  clone, while a shared permutation clone loses ONNX tensor provenance;
- multi-element all-zero index tensors are treated as scalar Gather indices;
- Pattern A collapses dynamic signatures and leaves a rank-shifted per-axis
  QDIM unchanged;
- Pattern B accepts inconsistent intermediate shapes and bypasses incompatible
  output quantization metadata;
- short or missing tensor metadata, duplicate producers, reverse graph order,
  a public internal input alias, and a reverse source producer are accepted.

Correction must build one `ModelIRGraphIndex` and complete immutable plans for
both patterns before mutation. Each plan must prove unique producers, strict
graph order, exact private intermediate consumers, scalar zero-index semantics,
complete compatible shape/signature/dtype/layout/quantization metadata, and
every public boundary. Index, permutation, and reshape-shape inputs need an
explicit unquantized INT32 TensorIR/buffer/ownership contract. Runtime/public
inputs, variables, producers, invalid metadata, and quantized constants must
reject. A changed permutation with unrelated users or a public output needs a
deterministic collision-safe clone preserving layout and ONNX provenance.
Pattern A must preserve dynamic axes, remap retained per-axis QDIM, and update
all permutation metadata consistently. Pattern B may bypass only metadata-
equivalent tensors with exact Gather-derived shapes. Constant action, metadata,
setters, removals, pruning, and statistics must be known before commit; a
zero-match call must remain a complete no-op. The 293-line count is descriptive
only; 2,000 remains the ONNX operation-count tier threshold.

That correction is now implemented in the 740-line raw owner. One
`ModelIRGraphIndex` replaces every repeated consumer reconstruction and stays
current through indexed input replacement and batched/single operator removal.
Two matches of each pattern construct and refresh it exactly once.

Both patterns now prove unique producers, source-before-consumer order, exact
private intermediate consumers, valid public boundaries, complete positive
physical shapes, full compatible signatures, data-preserving dtypes, and
scalar axis-zero Gather semantics before mutation. Pattern B additionally
proves the exact two singleton-axis removals, input/Reshape boundary layout and
quantization equivalence, and ordered downstream consumers before replacing
all uses. Pattern A proves the exact rank-four/rank-three shape algebra,
permutation options, target Reshape, and retained output before rank-lifting the
Transpose. Dynamic signature axes are permuted rather than concretized;
rank-three NCW/NWC annotations lift to NCHW/NHWC, and per-axis QDIM shifts by
one with the inserted leading singleton dimension.

Every index, permutation, and target-shape tensor now has an explicit immutable
unquantized INT32 TensorIR and NumPy-buffer contract. Scalar indices accept the
ONNX scalar `[]` and normalized singleton `[1]` representations but reject
multi-element zeros. Public inputs, variables, runtime producers, wrong
TensorIR or buffer dtypes, and quantized constants reject atomically. A changed
private permutation updates once; an unrelated consumer or public output
receives a deterministic collision-safe clone preserving layout and ONNX
provenance. Constant action, option values, metadata/QDIM, input replacement,
removals, and pruning are complete before the first mutation.

All forty-six former strict xfails are green. The scalar-index and single-index
construction contracts bring focused coverage to eighty-one cases. A no-match
call is a complete no-op; valid statistics, NumPy-exact outputs, negative axes,
multiple matches, fixed point, collision naming, public Pattern-A Reshape
outputs, and both production calls remain unchanged. The 740-line count is
descriptive only; 2,000 remains the ONNX operation-count tier threshold.

Ownership now resides in
`passes/attention_gather_cleanup_layout.py`. The extracted owner and the
corrected raw owner at checkpoint `a48ee607` are each 740 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias,
retains the historical private name as a one-return compatibility wrapper, and
keeps both production calls unchanged. The focused module does not import the
lowerer.

Sixteen direct owner/wrapper comparisons cover both ordinary patterns,
multiple and negative-axis matches, scalar and dynamic metadata, shared and
public permutation clones, unsafe variable indices, per-axis quantization,
zero-match no-prune, Pattern-B quantization mismatch, missing metadata, reverse
topology, a public internal alias, and duplicate source producers. Statistics
and complete normalized ModelIR state are identical in every case. The
mechanical move does not alter public APIs, artifacts, dependencies, corpus
policy, ordered runtime behavior, or TensorFlow isolation.

## Raw attention pre-projection rank-lift characterization

The next substantive raw source-order owner is the 190-line
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains`.
Production code and both historical calls remain unchanged. It previously had
only ordered architecture references; the focused
`test_flatbuffer_direct_attention_preproj_ranklift_layout.py` now owns its
synthetic ModelIR contract.

Twenty-one green cases freeze one and two Q/K-style BatchMatMul branches, all
four supported binary operators including reversed noncommutative inputs,
NumPy-exact outputs, multiple matches and fixed point, twelve existing public-
boundary/shape/fan-out/operator rejection guards, statistics, the raw owner's
current one-loop structure, and both production calls.

Twenty-seven concrete safety gaps are strict xfails:

- a zero-match invocation prunes an unrelated tensor;
- ten public-input, variable, wrong-TensorIR-dtype, wrong-buffer-dtype, or
  quantized leading/tail Reshape-shape tensors are accepted as compile-time
  constants;
- dynamic signatures are concretized and retained per-axis QDIM is not shifted
  for the inserted leading dimension;
- a rank-three bias that broadcasts before the rewrite but not after it is
  accepted, as are `adjX` and `adjY` BatchMatMul flags;
- tail shapes with two negative dimensions are accepted by product alone;
- short or missing tensor/bias metadata, dtype mismatch, duplicate producers,
  reverse graph order, a public internal input alias, and reverse or duplicate
  source producers are accepted.

Correction must construct one `ModelIRGraphIndex` and a complete plan for the
leading Reshape plus every BatchMatMul/binary/tail-Reshape branch before
mutation. The plan must prove unique producers, strict order, exact consumers,
complete compatible shape/signature/dtype/layout/quantization metadata,
untransposed BatchMatMul flags, positive tail dimensions, and broadcast
equivalence before and after rank lift. Leading and tail shape inputs need an
explicit immutable unquantized INT32 TensorIR/buffer/ownership contract.
Dynamic axes and per-axis QDIM must rank-lift by one. All branch input setters,
metadata updates, the leading removal, pruning, and statistics must be known
before commit, and a zero-match call must be a complete no-op. The 190-line
count is descriptive only; 2,000 remains the ONNX operation-count tier
threshold.

That correction is now implemented in the 563-line raw owner. One
`ModelIRGraphIndex` replaces the repeated consumer reconstruction and stays
current through indexed BatchMatMul input setters and the single leading-
Reshape removal. A two-branch rewrite constructs and refreshes it exactly once.

The leading source/Reshape and every branch now prove unique producers, strict
Reshape/BatchMatMul/binary/tail-Reshape order, exact private intermediate
consumers, complete positive physical shapes, full compatible signatures,
data-preserving dtypes, valid public boundaries, and concrete tail outputs.
`adjX`, `adjY`, `adj_x`, and `adj_y` must all be absent or false. Each binary's
other input must exist and broadcast to the exact old rank-three result and the
exact new rank-four result; this retains scalar and `[K]` biases while rejecting
rank-sensitive `[T,1,K]` values. All tail dimensions are positive and their
product is proven against the projection width.

Leading and tail shape inputs have immutable unquantized INT32 TensorIR and
NumPy-buffer contracts with exact values, shapes, signatures, ownership, and
runtime-producer absence. Dynamic sequence signatures rank-lift from
`[T,1,K]` to `[1,1,T,K]`; NCW/NWC metadata lifts to NCHW/NHWC, and retained
per-axis QDIM advances by one. Every branch setter and shape/signature/layout/
quantization update is planned before mutation. All twenty-seven former strict
xfails are green, zero-match execution is unchanged, and scalar-bias plus
single-index contracts bring focused coverage to fifty cases without changing
valid statistics, NumPy-exact outputs, fixed point, or both production calls.
The 563-line count is descriptive only; 2,000 remains the ONNX operation-count
tier threshold.

Ownership now resides in
`passes/attention_preproj_ranklift_layout.py`. The extracted owner and the
corrected raw owner at checkpoint `727c19c6` are each 563 lines and have
identical ASTs. The lowerer imports the module owner under a private pass alias,
retains the historical private name as a one-return compatibility wrapper, and
keeps both production calls unchanged. The focused module does not import the
lowerer.

Sixteen direct owner/wrapper comparisons cover ordinary and multiple branches,
reversed SUB, scalar and dynamic metadata, per-axis quantization, leading/tail
shape ownership, zero-match no-prune, rank-sensitive bias rejection,
BatchMatMul flags, missing bias/output metadata, reverse topology, a public
internal alias, and duplicate source producers. Statistics and complete
normalized ModelIR state are identical in every case. The mechanical move does
not alter public APIs, artifacts, dependencies, corpus policy, ordered runtime
behavior, or TensorFlow isolation.

## Raw NCHW-to-NHWC elementwise roundtrip characterization

The next substantive unextracted raw source-order owner is the 209-line
`_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains`. Production code
and its one ordered call remain unchanged. The focused fixture now expands its
former three cases into thirty-two green structural, numerical, all-op-family,
multiple-match, fixed-point, dynamic-signature, provenance, rejection,
statistics, owner-shape, and call-boundary contracts.

Twenty-eight concrete safety gaps are strict xfails: zero-match pruning;
public, variable, mistyped, quantized, or runtime-produced transpose
permutations; unremapped local and shared NHWC broadcast constants; stale
logical/physical layout and per-axis QDIM; and missing metadata, incompatible
shape/dtype/signature, public internal aliases, reverse topology, or malformed
multi-output candidates that are not rejected transactionally.

Correction must construct one `ModelIRGraphIndex` and the complete candidate
plan before mutation. The plan must prove unique ordered topology, exact
operator arity and boundaries, complete rank-four tensor compatibility,
immutable unquantized INT32 permutation ownership/type, valid old/new
broadcast semantics, and output availability. It must rotate local constants,
clone constants with nonlocal users, preserve dynamic axes, and remap layout
metadata and per-axis QDIM from NHWC to NCHW. All indexed rewires, output
rename, removals, pruning, and statistics must be known before commit; a
zero-match call must be a complete no-op. The 209-line count is descriptive
only; 2,000 remains the ONNX operation-count tier threshold.

That correction is now implemented in the 705-line raw owner. One differential
`ModelIRGraphIndex` replaces all repeated producer/consumer scans and remains
current through indexed input/output setters plus one batched removal. Exact
operator arity, unique ordered topology, private boundaries, complete positive
rank-four shape/signature/dtype/layout metadata, and both the original NHWC and
planned NCHW broadcasts are proven before mutation.

Pre/post permutations have immutable private unquantized INT32 TensorIR and
NumPy-buffer contracts. Non-scalar embedded constants are rank-expanded and
transposed to NCHW; shared/public constants are provenance-preserving clones,
while private constants are updated locally. Dynamic signatures,
logical/physical layout, and per-axis QDIM are remapped for constants,
intermediates, and the canonical output. All twenty-eight former strict xfails
are green, with additional constant-QDIM/ownership/provenance and duplicate-pre
coverage bringing the focused contract to sixty-four cases. Zero-match
execution no longer prunes, and valid numerical results, all allowed ops,
statistics, multiple matches, fixed point, and the production call are
unchanged. The 705-line count is descriptive only; 2,000 remains the ONNX
operation-count tier threshold.

Ownership now resides in
`passes/elementwise_roundtrip_nchw_nhwc_layout.py`. The extracted owner and the
corrected raw owner at checkpoint `79862309` are each 705 lines and have
identical ASTs. The lowerer retains a one-return compatibility wrapper and the
single ordered production call. Sixteen direct owner/wrapper families produce
identical statistics and complete normalized ModelIR state; a two-match runtime
case also proves one index construction with no refresh rebuild. The module
does not import the lowerer and the mechanical move does not alter public API,
artifacts, dependencies, corpus policy, ordered behavior, or TensorFlow
isolation.

All substantive top-level pass owners have now moved out of the lowerer. The
next structural boundary is its nested ordered orchestration. Before moving any
cluster, characterize the 66-line layout-recovery prefix and neighboring
51-line layout/reshape/attention prefix, including exact call order,
repetitions, layout-state and diagnostic propagation, and conditional gates.
Use that evidence to introduce an explicit phase-runner contract without
changing the current sequence.

That nested-orchestration characterization is now complete without production
changes. Existing tests already froze both complete call orders and repetition
boundaries. A separate focused fixture now freezes exact positional/keyword
contracts for all thirty-four calls, including model/layout/diagnostic routing
and the unary-passthrough flag. Both helpers are straight-line zero-argument
closures with no local state scope or control flow; their only captured data
names are `model_ir` and `session`.

Before extraction, resolve each call to its module owner. The two lowerer-local
pass clusters used by the layout prefix and the nested layout-prefix dependency
used by the attention prefix must become explicit callbacks. All other calls
should receive an immutable explicit phase context instead of capturing the
lowerer. The next implementation should define stable pass IDs and prove old/
new execution-order plus argument equality before switching production calls.

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

## Layout-recovery orchestration boundary

The repeated layout-recovery prefixes are now represented by the explicit
`passes/layout_recovery_orchestration.py` phase boundary. Its frozen
`LayoutRecoveryContext` carries the shared ModelIR, `LayoutState`, diagnostics,
and exactly three lowerer-local composite callbacks. Thirty pass owners are
imported directly from focused modules, so the phase module never imports or
captures the central lowerer.

`LAYOUT_RECOVERY_PASS_IDS` defines the nineteen-step layout prefix and
`ATTENTION_RECOVERY_PASS_IDS` defines the fifteen-step attention prefix. Each
step is an immutable `RecoveryInvocation` containing its stable ID, callable,
positional arguments, and keyword arguments. Builders verify that their
constructed order exactly matches the declared IDs before runners execute the
steps. The attention runner nests the complete layout runner as its first
explicit invocation, preserving the former closure structure and repetition.

The lowerer constructs one context after defining the three injected
composites. Its historical zero-argument helpers remain compatibility and
call-site boundaries, but now delegate to the phase runners and capture only
that context. Layout state and diagnostics are shared references by design;
the frozen context prevents rebinding while allowing each pass to mutate the
same validated conversion state as before.

Tests own both sides of this boundary. The focused orchestration fixture fixes
all stable IDs, argument routing, context capture, wrapper wiring, and
instrumented flattened execution order. Architecture and individual pass
fixtures count a stable orchestrated ID together with any remaining direct
lowerer call, retaining the former total execution-count guarantee without
depending on every call being visible in the lowerer AST.

The next characterized boundary consists of the seven-step
preadd/mean/attention recovery sequence and the ten-step attention/gate/QDQ
sequence. Both are currently lowerer-local straight-line closures over only
ModelIR and the conversion session. Three steps are lowerer-local composite
clusters; the other call slots are candidates for direct module ownership.
Focused contracts now fix all argument routing, the two and three outer
invocation counts, and the attention sequence's nested quantized-suffix
boundary before an explicit runner is introduced.

That boundary now uses `AttentionRecoveryContext` and stable seven-step and
ten-step invocation specifications in
`passes/attention_recovery_orchestration.py`. Fourteen pass owners are direct
module imports; the three composite clusters are explicit callbacks. The
historical lowerer helpers remain outer ordering boundaries but capture only
the context and delegate to the phase runners.

Immutable invocation execution and pre-execution ID-order validation are shared
through `passes/recovery_orchestration.py` by both attention and layout
orchestration. Architecture accounting retains an ordered ID sequence in
addition to its unique-ID set so a pass present in more than one phase
contributes its exact multiplicity to former direct-call totals.

The next characterized nested boundary is a one-step safe-binary recovery
runner reused as the final step of a six-step quantized activation/binary
runner. Both are parameterless straight-line lowerer closures over ModelIR and
layout state. Focused contracts fix the three total safe-binary invocations,
two quantized runner invocations, all ModelIR/layout arguments, the model-only
softmax canonicalization, and the nested final-step relationship before
production extraction.

That boundary now uses a frozen `QuantizedRecoveryContext` and the explicit
stable-ID specifications in
`passes/quantized_recovery_orchestration.py`. The one-step safe-binary phase
and six-step quantized activation/binary phase import all six existing module
owners directly; no lowerer-local callback or lowerer import is required. The
quantized phase nests the safe-binary runner as its final immutable invocation,
and the shared executor rejects ID-order drift before any callback runs.

The central lowerer constructs one context and retains both historical helper
names and all outer zero-argument call sites as compatibility/order boundaries.
Those helpers now delegate to the explicit runners. Ordered stable-ID
multiplicity accounts for the nested safe-binary invocation, so architecture
tests continue to prove three total safe-binary and two total quantized phase
executions while all characterized arguments and runtime order remain exact.

Focused, architecture, central lowerer, related quantized/binary family, and
TensorFlow-import-blocked suites pass. This mechanical extraction changes no
public API, CLI behavior, artifact, dependency, corpus/exclusion policy,
operation tier, pass semantics, or TensorFlow boundary. The next candidate is
the adjacent five-step qlinear mean/concat recovery sequence; characterize its
arguments and repetition boundaries before introducing another phase runner.

That qlinear sequence is now characterized without production changes. It is a
six-line parameterless straight-line closure over only ModelIR and calls five
already-extracted module owners with the same single positional argument. A
focused fixture freezes the complete order, argument contracts, two outer
zero-argument invocations, and both neighboring boundaries. No callback,
layout state, diagnostics, or conversion option needs to enter the eventual
phase context. The next change should introduce a frozen ModelIR-only context
and five stable IDs, then prove instrumented order before switching the
historical helper to a delegate.

That sequence now uses `QLinearRecoveryContext` and five stable IDs in
`passes/qlinear_recovery_orchestration.py`. The module imports all existing
owners directly and uses the shared immutable invocation executor, with no
lowerer import or callback dependency. The historical lowerer helper remains a
zero-argument order boundary but now captures only the context and delegates to
the runner; both outer calls and their neighbors are unchanged. Focused,
architecture, all five owner-family, core, and TensorFlow-import-blocked suites
pass. The next orchestration candidate is the mixed layout/attention/quantized
suffix, which must first be characterized because it carries layout,
diagnostics, nested phase runners, local composite callbacks, and an option
flag.

That mixed suffix is now characterized without production changes. It has
thirteen straight-line call slots, one required keyword-only duplicate-
transpose option, two outer invocations, and no local pass-state scope or
control flow. Focused contracts freeze all ModelIR/layout/diagnostics routing,
the option's single destination, and both outer neighbors. Ten slots have
direct module or extracted phase owners; the mean-attention cluster,
attention-gate/QDQ helper, and duplicate quantized-PReLU cluster must remain
explicit injected callbacks in a future frozen context. Extraction must prove
the same flattened order and per-invocation option value before changing the
historical helper.

That suffix now uses a frozen `LayoutAttentionQuantizedSuffixContext` and
thirteen stable IDs in
`passes/layout_attention_quantized_suffix_orchestration.py`. Ten owners are
direct imports and the three nested boundaries are explicit callbacks. The
duplicate-transpose value remains an invocation argument rather than context
state and is forwarded unchanged to its single destination. The historical
keyword-only lowerer helper and both outer calls remain as ordering boundaries
but now delegate through the context. Multiplicity-aware architecture checks
retain all former nested helper and registered-runner totals. Focused,
adjacent-owner, core, and TensorFlow-import-blocked suites pass. Characterize
the larger terminal slice/concat recovery sequence before attempting the next
phase extraction.

That terminal slice/concat sequence is now characterized without production
changes. It has fourteen straight-line call slots, two zero-argument outer
invocations, and ModelIR/layout/diagnostics routing with no local state scope or
control flow. Thirteen slots have existing module owners; the channel-slice/
pad/mul cluster is the only lowerer-local callback dependency. Focused tests
freeze all arguments and the two distinct outer neighbors. The eventual phase
should therefore use one frozen explicit context, one injected callback, and
fourteen stable IDs while preserving both top-level boundaries.

That sequence now uses a frozen `TerminalSliceConcatRecoveryContext` and
fourteen stable IDs in
`passes/terminal_slice_concat_recovery_orchestration.py`. Thirteen owners are
direct imports and the channel-slice/pad/mul boundary is one explicit callback.
The historical helper remains a zero-argument top-level boundary but now
captures only the context and delegates to the runner; both invocations and
their distinct neighbors are unchanged. Architecture and adjacent owner tests
combine stable phase multiplicity with remaining direct calls, preserving all
former execution totals. Focused, related-owner, core, and TensorFlow-import-
blocked suites pass. Characterize the neighboring terminal affine/concat/split
sequence before the next extraction.

The terminal affine/concat/split sequence is now characterized without
production changes. It has eleven straight-line calls, two zero-argument outer
invocations, six layout-aware argument contracts, and no callback,
diagnostics, option, control-flow, or local-state dependency. Every target has
an existing module owner. Focused tests freeze the complete order and both
distinct terminal neighbors. The eventual phase should use a frozen ModelIR/
layout context and eleven stable IDs with direct imports only.

That sequence now uses `TerminalAffineConcatSplitRecoveryContext` and eleven
stable IDs in
`passes/terminal_affine_concat_split_recovery_orchestration.py`. Every owner is
a direct import and the historical lowerer helper is a four-line delegate.
Both outer calls and their neighbors remain unchanged. Ordered stable-ID
multiplicity handles owners shared with the preceding terminal slice/concat
phase without losing execution-count guarantees. Focused, adjacent-owner,
core, and TensorFlow-import-blocked suites pass. Characterize the neighboring
SINet pre-add/resize sequence before the next extraction.

The neighboring SINet pre-add/resize sequence is now characterized without
production changes. It is a 20-line parameterless straight-line closure with
six ordered calls: two ModelIR-only calls followed by four ModelIR/layout
calls. All targets already have extracted owners, so the eventual context does
not need callbacks, diagnostics, or conversion options. Focused contracts
freeze every argument and all four zero-argument invocations, including the
one nested terminal-layout boundary and its three top-level repetitions. The
focused plus architecture suite passed 252 tests, and all 11 TensorFlow-import-
blocked tests passed. The next change should introduce a frozen ModelIR/layout
context and six stable IDs, then prove builder equality and instrumented order
before switching the historical helper to a delegate.

That sequence now uses `SINetPreaddResizeRecoveryContext` and six stable IDs in
`passes/sinet_preadd_resize_recovery_orchestration.py`. Every owner is a direct
import, the first two invocations carry ModelIR only, and the final four carry
ModelIR plus layout state. The historical helper is a four-line delegate while
all three top-level invocations, its nested terminal-layout invocation, and
their neighbors remain unchanged. Ordered architecture accounting combines
the new stable IDs with remaining direct calls so the existing owner totals
remain explicit. Focused, related-owner, core, and TensorFlow-import-blocked
suites pass apart from one stale direct fixture that fails identically at the
parent checkpoint. Characterize the adjacent three-step SINet terminal-layout
sequence before the next extraction.

That adjacent terminal-layout sequence is now characterized without production
changes. It is a seven-line parameterless straight-line closure with three
ordered slots: a ModelIR/layout shuffle-residual owner, the zero-argument SINet
pre-add/resize helper, and a ModelIR-only terminal affine/PReLU owner. Focused
contracts freeze all arguments, both top-level invocations, and both distinct
outer boundaries. The eventual phase needs a frozen ModelIR/layout context,
direct imports for the outer owners, and one explicit callback to retain the
historical nested helper boundary. The focused plus architecture suite passed
252 tests, and all 11 TensorFlow-import-blocked tests passed.

That sequence now uses `SINetTerminalLayoutRecoveryContext` and three stable
IDs in `passes/sinet_terminal_layout_recovery_orchestration.py`. The two outer
owners are direct imports, while the historical pre-add/resize helper is an
explicit zero-argument callback. The old helper is a four-line delegate and
both top-level boundaries remain unchanged. Stable-ID multiplicity retains the
nested pre-add/resize execution alongside its three direct calls. Focused,
outer-owner, architecture, core, and TensorFlow-import-blocked suites pass.
The next candidate is the terminal clamp/unary/ReLU cluster; characterize its
shared pass-state scope and every outer boundary before extraction.

That terminal clamp/unary/ReLU cluster is now characterized without production
changes. It contains three ordered cleanup runners and constructs exactly one
`ModelIRPassStateScope`, shared by all three along with the same ModelIR,
layout, and diagnostics values. It has one zero-argument outer invocation
between the layout-gated singleton-reshape and SINet terminal-layout
boundaries. Focused contracts freeze every argument and boundary, while the
existing pass-efficiency fixture proves one graph-index build at runtime. The
eventual phase needs a frozen ModelIR/layout/diagnostics context, three stable
IDs, and a fresh shared scope per phase invocation.

That cluster now uses `TerminalClampUnaryReLUContext` and three stable IDs in
`passes/terminal_clamp_unary_relu_orchestration.py`. Each invocation builder
creates exactly one fresh scope and shares it across all three cleanup calls.
The historical helper is a four-line delegate and its single outer boundary is
unchanged. The efficiency fixture now drives the explicit runner and still
observes one graph-index build. Focused, architecture, pass-efficiency, core,
and TensorFlow-import-blocked suites pass. Characterize the neighboring
terminal singleton-maxpool/reshape pair and its shared scope next.

That neighboring terminal singleton-maxpool/reshape pair is now characterized
without production changes. It contains two ordered cleanup runners and
constructs exactly one `ModelIRPassStateScope`, shared with the same ModelIR,
layout, and diagnostics values. Its sole zero-argument invocation remains
between two layout-gated blocks. Focused contracts freeze every argument and
boundary, while the existing efficiency fixture proves one graph-index build.
The eventual phase needs a frozen ModelIR/layout/diagnostics context, two
stable IDs, and one fresh shared scope per invocation.

That pair now uses `TerminalSingletonMaxPoolReshapeContext` and two stable IDs
in `passes/terminal_singleton_maxpool_reshape_orchestration.py`. Each builder
creates exactly one fresh scope shared by both cleanup invocations. The
historical helper is a four-line delegate and remains between the same layout-
gated blocks. The efficiency fixture now drives the explicit runner and still
observes one graph-index build. Focused, architecture, pass-efficiency, core,
and TensorFlow-import-blocked suites pass. Characterize the neighboring late
dequant/unary/fanout cluster and its shared scope next.

That late dequant/unary/fanout cluster is now characterized without production
changes. It contains three ordered cleanup runners and one shared
`ModelIRPassStateScope`, with identical ModelIR, layout, and diagnostics
routing. Its sole zero-argument call remains between the quantized HardSigmoid
bridge and swish passthrough. Focused contracts freeze every argument and
boundary, while the existing efficiency fixture proves one graph-index build.
The eventual phase needs a frozen ModelIR/layout/diagnostics context, three
stable IDs, and one fresh shared scope per invocation.

That cluster now uses `LateDequantUnaryFanoutContext` and three stable IDs in
`passes/late_dequant_unary_fanout_orchestration.py`. Each builder creates one
fresh scope shared by all three cleanup invocations. The historical helper is
a four-line delegate at the same boundary. Stable-ID multiplicity preserves
the moved occurrences alongside remaining direct runner calls, and the
efficiency fixture now drives the explicit phase while observing one graph-
index build. Focused, architecture, pass-efficiency, core, and TensorFlow-
import-blocked suites pass. Characterize the option-dependent transpose-unary
fanout cluster and its shared scope next.

That option-dependent cluster is now characterized without production
changes. It has four ordered runner slots and one fresh shared pass-state scope
per invocation. Layout-transpose and unary-passthrough cleanup are controlled
by separate keyword-only options; unary-fanout and unary-binary-fanout cleanup
always run. The attention-recovery callback exercises the default
`False`/`True` combination, while the direct post-QDQ boundary explicitly uses
`True`/`False`. Focused contracts freeze both active sequences, all ModelIR/
layout/diagnostics/scope arguments, both outer placements, and the callback
identity. The eventual phase should keep options outside its frozen context,
derive variant-specific expected IDs, and construct one fresh shared scope for
each build.

That cluster now uses `TransposeUnaryFanoutContext` and four canonical stable
IDs in `passes/transpose_unary_fanout_orchestration.py`. The frozen context
contains ModelIR, layout, and diagnostics only. Both Boolean choices remain
runner arguments, and the active expected-ID sequence is derived before the
shared executor runs either three-step variant. Every build creates one fresh
scope shared by all active invocations. The historical keyword-only helper is
a delegate with unchanged defaults, callback identity, direct option values,
and outer boundaries. Both efficiency fixtures now exercise the explicit
runner and retain one graph-index build. Focused, architecture, pass-
efficiency, core, and TensorFlow-import-blocked suites pass.

The late SPP/concat-unary-conv pair is now characterized without production
changes. It has two ordered cleanup owners, one fresh shared pass-state scope,
one zero-argument terminal invocation, and no option, callback, or control
flow. Focused contracts freeze all ModelIR/layout/diagnostics/scope arguments
and both outer boundaries. The eventual phase should use one frozen ModelIR/
layout/diagnostics context, two direct owner imports, two stable IDs, and a
fresh shared scope per build.

That pair now uses `LateSPPConcatUnaryConvContext` and two stable IDs in
`passes/late_spp_concat_unary_conv_orchestration.py`. Both owners are direct
imports, and every builder creates one fresh scope shared by the two immutable
invocations. The historical helper is a zero-argument delegate at the same
terminal boundary. The efficiency fixture now exercises the explicit runner
and retains one graph-index build. Focused, architecture, pass-efficiency,
core, and TensorFlow-import-blocked suites pass.

The boundary-batchmatmul/input-unary pair is now characterized without
production changes. It has two ordered cleanup owners, one fresh shared pass-
state scope, and no option or direct invocation. It is an explicit callback in
`LayoutRecoveryContext`; focused contracts freeze its complete ModelIR/layout/
diagnostics/scope routing, callback identity, and both stable-list neighbors.
The eventual phase should use a frozen ModelIR/layout/diagnostics context, two
direct owner imports, two stable IDs, and one fresh shared scope per build,
while retaining the historical callback boundary.

That pair now uses `BoundaryBatchMatMulUnaryContext` and two stable IDs in
`passes/boundary_batchmatmul_unary_orchestration.py`. Both owners are direct
imports, every builder creates one fresh scope shared by the immutable
invocations, and the lowerer no longer imports either runner. The historical
helper remains the same zero-argument `LayoutRecoveryContext` callback as a
four-line delegate, with unchanged stable neighbors. The efficiency fixture
now exercises the explicit runner and retains one graph-index build. Focused,
architecture, pass-efficiency, core, and TensorFlow-import-blocked suites pass.

The channel-slice/pad-mul pair is now characterized without production
changes. It has two ordered owners and one fresh shared pass-state scope. The
helper executes twice: first as the leading callback in the terminal-slice/
concat recovery phase and later as one direct zero-argument call between pre-
add and affine post-add recovery. Focused contracts freeze all ModelIR/layout/
diagnostics/scope arguments, callback identity, and both boundary forms. The
eventual phase should use a frozen ModelIR/layout/diagnostics context, two
direct owner imports, two stable IDs, and a fresh shared scope per execution.

That pair now uses `ChannelSlicePadMulContext` and two stable IDs in
`passes/channel_slice_pad_mul_orchestration.py`. Both owners are direct
imports, every builder creates one fresh scope shared by the two immutable
invocations, and the lowerer no longer imports either runner. The historical
helper remains both the leading terminal-slice/concat callback and the
additional late-terminal direct call as a four-line delegate. Stable helper
multiplicity preserves two executions, and the efficiency fixture now
exercises the explicit runner while retaining one graph-index build. Focused,
architecture, pass-efficiency, core, and TensorFlow-import-blocked suites pass.

The late hard-activation/layout pair is now characterized without production
changes. It always runs hard-activation passthrough with four fixed terminal
policy flags and conditionally runs layout-transpose cleanup under one required
keyword-only option. Both active runners share one fresh pass-state scope. The
sole caller forwards the global layout optimization choice, and focused
contracts freeze its outer terminal boundaries. The eventual phase should keep
the option outside a frozen ModelIR/layout/diagnostics context, derive active
expected IDs, and construct one fresh shared scope per build.

That pair now uses `LateHardActivationLayoutContext` and two canonical stable
IDs in `passes/late_hard_activation_layout_orchestration.py`. The required
Boolean remains a runner argument, active expected IDs distinguish the one-
step and two-step forms, and every build creates one fresh shared scope. All
four terminal hard-activation flags are retained. The historical helper and
caller keep their required keyword-only contract and outer boundaries. The
central lowerer no longer imports the hard-activation runner, and the
efficiency fixture exercises the enabled explicit phase with one graph-index
build. Focused, architecture, pass-efficiency, core, and TensorFlow-import-
blocked suites pass.

The absolute-final normalization/attention pair is now characterized without
production changes. It has two ordered owners, one fresh shared pass-state
scope, fixed `False`/`True` normalization flags, and one zero-argument call at
the terminal instance-normalization/dynamic-shape boundary. Focused contracts
freeze all ModelIR/layout/diagnostics/scope arguments and the mixed-attention
rationale. The eventual phase should use a frozen ModelIR/layout/diagnostics
context, two direct owner imports, two stable IDs, and one fresh shared scope
per build.

That pair now uses `AbsoluteFinalNormalizationAttentionContext` and two stable
IDs in `passes/absolute_final_normalization_attention_orchestration.py`. Both
owners are direct imports, every builder creates one fresh scope shared by both
immutable invocations, and the fixed normalization policy plus attention
rationale stay with their owners. The historical helper is a four-line zero-
argument delegate at the same sole terminal boundary. The efficiency fixture
now exercises the explicit runner while retaining one graph-index build.
Focused, architecture, pass-efficiency, core, and TensorFlow-import-blocked
suites pass.

The QKV attention cluster is now characterized without production changes. It
has three ordered owner slots and one fresh shared pass-state scope. Independent
keyword-only options control layout-transpose and prefix cleanup; bridge
cleanup always runs. Two calls use the default prefix-plus-bridge form, while a
late call disables prefix and forwards the global layout option. Focused
contracts freeze all ModelIR/layout/diagnostics/scope arguments and three
distinct outer boundaries. The eventual phase should keep both options outside
a frozen ModelIR/layout/diagnostics context and derive active expected IDs per
invocation.

That cluster now uses `QKVAttentionContext` and three canonical stable IDs in
`passes/qkv_attention_orchestration.py`. Both Boolean choices remain runner
arguments, active expected IDs cover the default and both late runtime forms,
and every build creates one fresh shared scope. The historical helper keeps
both defaults, three callers, caller values, and outer boundaries as a
delegate. The central lowerer no longer imports the QKV prefix or bridge
runner. The efficiency fixture exercises the explicit default phase with one
graph-index build. Focused, architecture, pass-efficiency, core, and
TensorFlow-import-blocked suites pass.

The duplicate-fanout/quantized-PReLU pair is now characterized without
production changes. Its required keyword-only transpose option is forwarded
only to duplicate-fanout cleanup, while quantized-PReLU cleanup always follows.
Both owners share one fresh `ModelIRPassStateScope` and receive the same
ModelIR, layout state, and diagnostics. The helper has no direct call and is
owned by one stable callback slot in the layout/attention/quantized suffix;
focused contracts freeze the callback identity, suffix option routing, and
both stable neighbors. The eventual phase should keep the option outside a
frozen ModelIR/layout/diagnostics context, expose two stable owner IDs, and
construct one fresh shared scope per build.

That pair now uses `DuplicateQuantizedPReLUContext` and two stable IDs in
`passes/duplicate_quantized_prelu_orchestration.py`. The required transpose
choice stays outside the frozen ModelIR/layout/diagnostics context, every
builder creates one fresh scope shared by both immutable invocations, and the
fixed owner order is identical for both option values. The historical helper
remains the same keyword-only callback in the layout/attention/quantized
suffix, with unchanged option routing and stable neighbors. Other independent
direct calls to both cleanup runners remain untouched. The efficiency fixture
now exercises the explicit phase and retains one graph-index build. Focused,
architecture, pass-efficiency, core, and TensorFlow-import-blocked suites pass.

The constant-fold/redundant-cast pair is now characterized without production
changes. Its optional keyword-only pass-state scope supports either fresh
allocation or reuse of a parent scope, and both ordered owners receive the
same ModelIR/layout/diagnostics/scope contract. Both current production calls
use the external-scope form: one between very-late gather and normalization,
and one after late layout/mean/SPP/gather cleanup. Focused contracts freeze
both caller parents and their internal boundaries. The eventual phase should
keep the optional scope outside a frozen ModelIR/layout/diagnostics context,
allocate only when absent, and expose two stable owner IDs.

That pair now uses `ConstantFoldCastContext` and two stable IDs in
`passes/constant_fold_cast_orchestration.py`. The optional scope stays outside
the frozen ModelIR/layout/diagnostics context: every scope-less build allocates
one fresh scope, while an external scope is preserved by identity. Both owners
share the resolved scope and fixed order. The historical helper remains a
keyword-only delegate, both parent phases retain their external-scope calls and
internal boundaries, and the central lowerer no longer imports either runner.
The efficiency fixture now exercises the explicit external-scope phase and
retains one graph-index build. Focused, architecture, pass-efficiency, core,
and TensorFlow-import-blocked suites pass.

The very-late gather/constant/normalization parent is now characterized without
production changes. One fresh pass-state scope spans transpose-gather cleanup,
the explicit constant-fold/cast pair, and fixed-policy normalization cleanup.
Its three phase calls expand to four effective owner steps, all sharing the
same ModelIR/layout/diagnostics/scope. Focused contracts freeze the flattened
order, `include_instance=False`, `include_flatten=True`, the sole terminal
call, and both outer boundaries. The eventual phase should compose the existing
constant-fold/cast builder with the parent scope and expose all four effective
owner IDs without duplicating the middle pair's argument construction.

That parent now uses `VeryLateGatherConstantNormalizationContext` and four
flattened stable IDs in
`passes/very_late_gather_constant_normalization_orchestration.py`. Each build
creates one fresh scope, builds the outer gather and fixed-policy normalization
steps locally, and splices in the existing constant-fold/cast builder with the
same external scope. The historical helper remains a zero-argument delegate at
the same terminal boundary. Constant-fold/cast now has one historical helper
caller plus one explicit builder composition; multiplicity-aware architecture
accounting records both executions and the efficiency fixture retains one
graph-index build. Focused, architecture, pass-efficiency, core, and
TensorFlow-import-blocked suites pass.

The SE-FC/gather-channel-fanout pair is now characterized without production
changes. It accepts positional target ModelIR/layout values, creates one fresh
target-specific pass-state scope, and forwards session diagnostics to two
ordered owners. One caller uses fallback ModelIR with no layout state; the
absolute-final caller uses the main ModelIR and session layout state. Focused
contracts freeze both forms and their identical SiNet-tail/reconciliation
boundaries. The eventual phase should use a frozen target context and two
stable IDs, with the historical helper constructing a context per call rather
than capturing one long-lived target.

That pair now uses `SEFCGatherChannelFanoutContext` and two stable IDs in
`passes/se_fc_gather_channel_fanout_orchestration.py`. The historical helper
constructs a frozen context per positional call, so fallback and main targets
cannot leak into each other. Every builder creates one fresh target-specific
scope shared by both immutable invocations. Both production caller forms and
their identical surrounding boundaries remain unchanged; unrelated direct
runner calls remain in the lowerer. The efficiency fixture now exercises the
explicit no-layout phase and retains one graph-index build. Focused,
architecture, pass-efficiency, core, and TensorFlow-import-blocked suites pass.

The terminal-boundary layout cluster is now characterized without production
changes. This fixed-target, zero-argument cluster shares one pass-state scope
across dual-MUL/CONCAT, boundary-input, PAD, layout-transpose, and transpose-
gather channel-fanout owners in that order. Every owner receives the main
ModelIR, session layout state, session diagnostics, and the shared scope. Its
sole call remains between the final InstanceNorm dual-stat residual-add/resize
rewrite and the terminal layout-optimization conditional. The eventual phase
should expose five stable IDs in a dedicated direct-owner module while
preserving the PAD-after-boundary and transpose-after-PAD recovery sequence.

That cluster now uses `TerminalBoundaryLayoutContext` and five stable IDs in
`passes/terminal_boundary_layout_orchestration.py`. Each build creates one
fresh scope shared by every immutable invocation, and the module imports all
owners directly without importing the lowerer. The historical helper is a
zero-argument delegate to a frozen main-model context at the same sole call and
outer boundaries. The explicit order retains boundary-input before PAD and PAD
before the final transpose sweep. One boundary-input direct import disappears;
unrelated direct uses of the other owners remain. Multiplicity-aware
architecture accounting preserves 120 effective ordered calls, and the
efficiency fixture retains one graph-index build. Focused, architecture,
pass-efficiency, core, and TensorFlow-import-blocked suites pass.

The late layout/mean/SPP/gather/constant-fold/cast parent is now characterized
without production changes. Its required keyword-only policy optionally adds
layout-transpose cleanup before three fixed direct owners and the existing
constant-fold/cast child. One fresh main-model scope spans all five required
effective owners or all six full-policy owners. The sole caller forwards the
global layout-optimization policy between shape-extract cleanup and expand/
squeeze canonicalization. The eventual phase should compose the existing
constant-fold/cast builder with its parent scope; after extraction, its dead
lowerer delegate/context/imports can be removed while preserving two effective
production executions of the constant-fold/cast pair.

That parent now uses `LateLayoutMeanSPPGatherConstantCastContext`, five
required IDs, and six full-policy IDs in
`passes/late_layout_mean_spp_gather_constant_cast_orchestration.py`. Every
build creates one fresh scope, conditionally prepends layout cleanup, builds
three fixed direct owners, and composes the existing constant-fold/cast builder
with the same scope. The historical helper is a required keyword-only delegate
at the same sole caller and boundaries. Its extraction retires the now-dead
lowerer constant-fold/cast helper, context, and imports. Production accounting
replaces those two child IDs plus four direct calls with the six full parent
IDs, retaining 120 effective ordered calls and two constant-fold/cast
compositions. The full-policy efficiency fixture retains one graph-index build.
Focused parent/child, architecture, pass-efficiency, core, and TensorFlow-
import-blocked suites pass.

The singleton/consecutive-reshape cluster is now characterized without
production changes. It accepts positional target ModelIR/layout values,
creates one fresh target-specific scope, and runs singleton-channel transpose,
reshape-only duplicate-fanout, and consecutive-reshape owners in fixed order.
Two callers use the main ModelIR/session layout; a normalization-rewrite-guarded
fallback caller uses fallback ModelIR with no layout state. Focused contracts
freeze all three owner contracts, `include_transpose=False`, caller forms,
multiplicity, the fallback guard, and all six surrounding boundaries. The
eventual phase should use a frozen target context constructed inside the
historical helper for each call and expose three stable IDs.

That cluster now uses `SingletonConsecutiveReshapeContext` and three stable IDs
in `passes/singleton_consecutive_reshape_orchestration.py`. Each build creates
one fresh target-specific scope and retains the fixed reshape-only duplicate-
fanout policy. The historical helper constructs its frozen context per
positional call, preventing main/fallback target leakage. Both main calls, the
guarded fallback call, the guard itself, and all boundaries remain unchanged.
Multiplicity-aware accounting moves one syntactic occurrence of each owner to
stable IDs, preserves 120 effective ordered calls, and retains the two
reshape-only duplicate-fanout and singleton-channel occurrences. The explicit
no-layout efficiency fixture retains one graph-index build. Focused,
architecture, pass-efficiency, core, and TensorFlow-import-blocked suites pass.

The gate-layout cluster is now characterized without production changes. Its
keyword-only `include_mixed_attention=True` policy selects eight full-policy
owners or seven required owners under one main-model scope. One direct caller
explicitly selects the reduced policy; the attention-recovery context stores
the same helper as an argument-free callback, thereby selecting the default
full policy. Focused contracts freeze the exact optional guard, complete owner
contracts and order, default, direct boundary, callback identity, and empty
callback arguments. The eventual phase should expose required/full stable-ID
sequences while retaining the historical helper as both delegate and callback.

That cluster now delegates to `GateLayoutContext` and the stable required/full
ID sequences in `passes/gate_layout_orchestration.py`. Each invocation build
creates one fresh main-model scope shared by every selected owner. The full
policy prepends mixed-attention cleanup to the seven required owners; the
reduced policy returns those same seven owners without constructing a second
execution path. The historical helper retains its keyword-only `True` default,
the direct caller still passes explicit `False`, and attention recovery still
holds the helper itself as an argument-free callback. Multiplicity-aware
accounting replaces eight direct owner calls with eight stable IDs while
preserving 120 effective ordered calls. The explicit full-policy efficiency
fixture retains one graph-index build.

The channel-shuffle/gather cluster is now characterized without production
changes. Its three keyword-only switches default to two-way shuffle enabled,
NHWC shuffle enabled, and post-gather cleanup disabled. Two owners form the
unconditional NCHW-shuffle/gather base; two optional shuffle owners precede it,
and the optional post group appends layout-transpose plus unary and binary-
fanout cleanup. The layout-recovery context stores the helper as an argument-
free callback and therefore selects the four-owner default. One direct caller
selects the seven-owner full-plus-post policy, while the late direct caller
disables both leading shuffles and selects only the two-owner base. Focused
contracts freeze the exact guards, all owner contracts/order, caller keywords,
callback identity, and both direct caller boundaries.

That cluster now delegates to `ChannelShuffleGatherContext` and stable leading,
base, default, post, and seven-owner union ID sequences in
`passes/channel_shuffle_gather_orchestration.py`. A single policy selector
builds every one of the eight boolean combinations in the same fixed relative
order, and every invocation build owns one fresh main-model scope shared by its
selected owners. The historical helper preserves all three keyword-only
defaults, both direct caller forms and boundaries remain unchanged, and layout
recovery still stores the helper itself as its argument-free callback.
Multiplicity-aware accounting replaces seven direct owner calls with seven
stable IDs while retaining 120 effective ordered calls. The explicit seven-
owner efficiency fixture retains one graph-index build.

The mean/attention cluster is now characterized without production changes.
Its keyword-only switches default to layer-normalization disabled and conv-
attention enabled. Five owners form the unconditional mean/SE base, with the
layer-normalization owner inserted after the first two and conv-attention
appended last under separate guards. Both attention recovery and the layout/
attention/quantized suffix store the same helper as an argument-free callback,
selecting the six-owner default. One direct caller enables layer normalization
while retaining conv attention; the terminal direct caller disables conv
attention while retaining the default-disabled layer normalization. Focused
contracts freeze all owner contracts/order, both exact guards, defaults,
callback identities, caller keywords, multiplicity, and all direct boundaries.

That cluster now delegates to `MeanAttentionContext` and stable prefix, base-
tail, base, layer-normalization, conv-attention, default, and seven-owner union
sequences in `passes/mean_attention_orchestration.py`. One selector covers all
four policy combinations in the fixed insertion order, and each invocation
build owns one fresh main-model scope shared by its selected owners. The helper
retains both keyword-only defaults, both direct caller forms and boundaries are
unchanged, and both parent compositions still store the helper itself as an
argument-free callback. Multiplicity-aware accounting replaces seven direct
owner calls with seven stable IDs while preserving 120 effective ordered calls.
The new explicit seven-owner efficiency fixture observes one graph-index build.

The singleton-reshape cluster is now characterized without production changes.
It has four keyword-only switches: three independently guard layout-transpose,
reshape-only duplicate-fanout, and multi-branch-gate owners, while the fourth is
forwarded to the always-active singleton-spatial owner's CONCAT-post-transpose
policy. Seven owners form the unconditional base under one main-model scope.
The layout-optimization caller selects layout plus multi-branch owners and the
default spatial policy for a nine-owner sequence. The terminal caller selects
duplicate fanout, disables the spatial post-transpose policy, and runs eight
owners. Focused contracts freeze all ten owner contracts/order, exact guards,
the two specialized owner keywords, defaults, both caller forms, and all four
boundaries.

That cluster now delegates to `SingletonReshapeContext` and stable layout,
prefix, duplicate, base-tail, multi-branch, active-policy, and ten-owner union
sequences in `passes/singleton_reshape_orchestration.py`. A single selector
covers all eight optional-owner combinations, while the independent spatial
CONCAT-post-transpose value is forwarded for both of its values; focused tests
therefore cover all sixteen boolean combinations. Every invocation build owns
one fresh main-model pass-state scope shared by exactly the selected owners.
The duplicate owner retains `include_transpose=False`, and the spatial owner
retains the forwarded `include_concat_post_transpose` value.

The historical helper is a one-call delegate with all four keyword-only
defaults intact. Both production caller forms, their exact keywords, and all
four surrounding boundaries are unchanged. Multiplicity-aware architecture
accounting replaces ten direct owner calls with ten stable IDs while preserving
120 effective ordered calls. The explicit ten-owner efficiency fixture observes
one graph-index build and thirteen diagnostic events. An AST audit now finds no
nested lowerer helper that directly executes more than one conversion pass;
every `_run_*` conversion cluster delegates to a dedicated orchestration
module.

The next context-consolidation boundary is characterized without production
changes. Twenty-one dedicated orchestration context classes expose the exact
same frozen `model_ir`, `layout_state`, and `diagnostics` identity contract.
Their twenty-two construction sites consist of eighteen main-session contexts,
two target-specific helper contexts, and two child constant-fold/cast contexts.
No site stores an option, callback, or other hidden policy in these types, and
none of the owner modules imports the lowerer. This makes a single core
ModelIR-pass context owned by `ConversionSession` a mechanical next step while
keeping callback-bearing parent contexts separate.

That boundary now uses the frozen `ModelIRPassContext` core contract.
`ConversionSession` constructs one identity-bound instance after its
`GraphIndex` and `LayoutState` and exposes it as `model_ir_pass_context`.
Twenty-one historical context names remain as internal aliases, so their
builders and runners keep the same signatures while importing one core type.
All eighteen main-model orchestration consumers reference the Session-owned
instance. The two primary/fallback helpers still create a fresh common context
per target call, preventing ModelIR or LayoutState leakage across targets.

Both composed constant-fold/cast parents now pass their existing common context
directly to the child builder instead of reconstructing an identical object.
Callback-bearing recovery contexts remain separate. Context reuse does not
reuse pass state: every invocation builder still creates one fresh
`ModelIRPassStateScope`, and only the owners selected within that build share
its lazy graph index. Pass IDs, policy arguments, runtime order, diagnostics
identity, and all caller boundaries remain unchanged.

The callback-bearing context composition boundary is now characterized without
production changes. `AttentionRecoveryContext`, `LayoutRecoveryContext`,
`LayoutAttentionQuantizedSuffixContext`, and
`TerminalSliceConcatRecoveryContext` each prepend the same ModelIR/LayoutState/
diagnostics identity triple to one or three callback fields. Their four lowerer
constructors use the main Session identities and ten exact callback objects.
Focused contracts preserve the three callback invocation forms: argument-free
cluster callbacks, the pre-Concat callback receiving ModelIR plus layout and
diagnostics, and duplicate/PReLU receiving its Boolean keyword policy. The
diagnostics-free SINet terminal context remains outside this boundary.

The callback-bearing context boundary is now composed around the Session-owned
`ModelIRPassContext`. The four recovery dataclasses each contain one explicit
`pass_context` followed by their existing callback fields, and every invocation
builder reads ModelIR, LayoutState, and diagnostics through that common object.
All four lowerer constructors receive the exact
`session.model_ir_pass_context` identity. The ten callbacks, their invocation
arguments, stable pass IDs, execution order, and fresh pass-state-scope behavior
are unchanged. `SINetTerminalLayoutRecoveryContext` remains independent because
its diagnostics-free identity contract is intentionally different.

The next diagnostics-free model/layout context boundary is characterized
without production changes. `SINetPreaddResizeRecoveryContext`,
`QuantizedRecoveryContext`, and `TerminalAffineConcatSplitRecoveryContext` are
frozen dataclasses with exactly the same ModelIR/LayoutState fields. Their four
invocation builders never read or forward diagnostics, and a full
`ModelIRPassContext` is already a behaviorally identical input: every ModelIR,
LayoutState, and nested-context identity remains exact. The lowerer constructs
all three from the main Session identities. The model-only QLinear context and
callback-bearing SINet terminal context remain explicitly outside this
boundary.

The diagnostics-free model/layout boundary is now consolidated into the common
Session context. The three historical type names are internal aliases of
`ModelIRPassContext`, increasing the shared alias inventory to twenty-four and
the main-model consumers to twenty-one. Their lowerer variables all reference
`shared_model_ir_pass_context`; no per-phase object is constructed. Diagnostics
are available on the common object but remain inert across all four builders.
ModelIR/LayoutState identity, nested safe-binary context identity, pass IDs,
layout policies, and execution order are unchanged. The model-only QLinear and
callback-bearing SINet terminal contexts remain independent.

The final two unconsolidated orchestration contexts are now characterized as
separate boundaries without production changes. The model-only
`QLinearRecoveryContext` feeds five invocations and already accepts a full
`ModelIRPassContext` without observing LayoutState or diagnostics. The
`SINetTerminalLayoutRecoveryContext` retains its ModelIR/LayoutState identity
and one argument-free pre-add/resize callback across three ordered invocations.
Both lowerer constructors, exact callback wiring, keyword policies, and module
independence are frozen before either type changes.

All orchestration context identity is now normalized. QLinear is the twenty-
fifth internal `ModelIRPassContext` alias and the twenty-second main-model
consumer. SINet terminal is the fifth callback-bearing dataclass and stores one
common `pass_context` plus its callback, bringing that callback inventory to
eleven. Every orchestration state holder is therefore either the common core
context or an explicit callback composition around it; no orchestration
dataclass independently repeats ModelIR, LayoutState, or diagnostics. QLinear's
five argument contracts and SINet's three invocation/callback contracts remain
unchanged.

The next non-context efficiency boundary is ModelIR pass diagnostic numbering.
`run_model_ir_pass_group()` currently scans the complete diagnostic history to
select the next group number, then scans the growing history again for every
pass result to calculate global `sequence` and per-pass `invocation` values.
This preserves correct numbering but makes bookkeeping quadratic in the number
of recorded pass events. The frozen contract ignores non-`model_ir_pass`
events, assigns the next group after the maximum existing `group_sequence`,
numbers new events after the existing ModelIR-pass event count, and advances
each pass ID independently. A strict expected-failure efficiency fixture proves
that the current implementation performs four history iterations for a
three-result group where one initialization scan is sufficient. The next
bounded implementation step may cache those three counters from one initial
scan, but must preserve the exact event schema, append order, caller-owned list,
failure diagnostics, and summary output.

The diagnostic-numbering boundary now performs exactly that one initialization
scan. It derives the existing ModelIR-pass event count, the maximum existing
group sequence, and per-pass invocation counts together, then advances the
local counters as new events are appended. The caller-owned diagnostic list,
non-pass records, event schema, append order, group semantics, failure path,
and summary remain unchanged. The strict scan-count fixture is green, reducing
one group with `P` results from `P + 1` full history iterations to one.

Cross-group numbering is the adjacent characterized boundary. The ordinary
list-compatible API must retain its one-scan fallback because callers may
mutate an arbitrary list between invocations. The production
`ConversionSession`, however, owns the same append-only diagnostics object for
the complete conversion. A strict expected-failure contract now requires that
Session-owned diagnostics retain numbering state across groups, lazily rebuild
that state once after an arbitrary list mutation, and keep subsequent group
numbering at constant bookkeeping cost. It also freezes list compatibility and
the same sequence, invocation, and group values across two repeated pass
groups. Production remains unchanged at this characterization checkpoint.

The Session-owned cross-group ledger is now implemented as a private
list-compatible core type. Append, extend, and insert update a valid numbering
snapshot incrementally; replacement, deletion, multiplication, pop, and remove
invalidate it; clear restores an empty valid snapshot. The next pass group
rebuilds invalid state once and subsequent groups reuse it. Ordinary external
lists continue through the prior one-scan fallback. The production append-only
Session path performs no rebuild across repeated groups, while the destructive-
mutation fixture performs exactly one. The internal type is not re-exported by
the public core package, and diagnostics remain the same mutable list contract
for existing consumers.

The next graph-scan boundary is the absolute-final SINet cleanup. Six bounded,
transactional SINet owners execute in a fixed terminal order, each followed by
an unconditional static-shape reconciliation. Every owner already returns one
complete mutation counter and has positive, no-op, rejection, and idempotence
fixtures. A strict expected-failure architecture contract now maps each owner
to its exact counter and requires its following reconciliation to be guarded by
that counter. This preserves all six pass calls and their order while allowing
the common zero-owner path to avoid six full operator/tensor reconciliation
scans. Production remains unchanged at this characterization checkpoint.

The six absolute-final SINet reconciliations are now guarded by their exact
owner counters. Every owner still runs once in the same terminal position with
the same LayoutState; only a zero result skips the immediately following
static-shape reconciliation. A synthetic lowerer contract proves that enabling
any one counter adds exactly one reconciliation relative to the all-zero path,
and the complete owner suites retain their positive/no-op/idempotence behavior.
The common non-SINet path therefore avoids six full reconciliation scans without
changing any rewrite or cross-owner ordering.

The next terminal reconciliation candidate is the indexed mixed-singleton
NCHW-input repair for an NHWC Concat. Its owner returns the single complete
`repaired_mixed_singleton_nchw_inputs_for_nhwc_concat` mutation counter, mutates
only when that counter is positive, and has explicit positive, no-op, index,
layout, fan-out, naming, and missing-Concat coverage. The production lowerer
still reconciles unconditionally at this characterization checkpoint. A strict
architecture expectation fixes the required immediate counter guard before
production changes. Nearby owners are not equivalent: the PReLU owner prunes
unused tensors even on a zero rewrite count, while the SE/FC/Gather cluster
currently discards multiple pass results. They must not reuse this guard without
separate counter-completeness work.

The absolute-final mixed-singleton Concat call now assigns that owner result and
guards only its immediately following static-shape reconciliation with the
exact counter. The owner still executes once in the same position with the same
Session LayoutState. A lowerer-level zero/one fixture proves that a positive
counter adds exactly one reconciliation over the zero-counter path, while the
complete owner suite preserves positive and no-op behavior. The unrelated
PReLU and SE/FC/Gather boundaries remain unconditional pending their separate
mutation-accounting contracts.

The absolute-final consecutive-Reshape runner is the next characterized scan
boundary. Its three returned counters cover no-op removal, consecutive-chain
bypass, and fan-out bypass. The fan-out path also increments the aggregate
consecutive counter, and tensor pruning/LayoutState synchronization occur only
after a positive mutation count. A no-candidate fixture freezes the exact
all-zero result, unchanged ModelIR, skipped diagnostic, and absence of pass-state
construction. Production still reconciles unconditionally at this checkpoint;
a strict architecture expectation requires one immediate guard over all three
counters before that call may change.

The terminal consecutive-Reshape result is now assigned, and the existing
immediate reconciliation is guarded by the sum of those three exact mutation
counters. The runner remains in its original position with the same Session
LayoutState and diagnostics. A lowerer-level fixture proves the zero path and
each individual positive counter; the complete runner suite retains both
rewrite behavior and its state-free no-candidate fast path.

The absolute-final PReLU boundary cannot use its rewrite counter alone. The
owner intentionally prunes unused tensors on every invocation, including when
the rewrite count is zero. A new owner fixture freezes that behavior and proves
the corresponding LayoutState remains current. The bounded guard contract
therefore records the tensor-table size immediately before the owner and
requires reconciliation after either a positive rewrite counter or a tensor
count reduction. Production remains unconditional at this characterization
checkpoint.

The absolute-final PReLU call now records the tensor count, assigns the owner
result, and reconciles only after a positive rewrite or a tensor-count decrease.
The owner still performs its unconditional prune and retains the same Session
LayoutState. A lowerer-level fixture covers unchanged, rewrite, and prune-only
outcomes, so the optimized guard preserves the previously unreported cleanup
mutation rather than treating a zero rewrite count as a no-op.

The shared recovery-invocation utility validates pass IDs and runs callbacks in
order but currently discards every callback result. This prevents the terminal
SE/FC/Gather wrapper from exposing its two already-complete runner dictionaries
to the lowerer. Strict expected-failure contracts now require the generic
utility and this specific wrapper to return ordered result tuples without
changing callback order, arguments, shared state scope, diagnostics, or
exception timing. Production remains unchanged at this characterization
checkpoint.

`run_recovery_invocations()` now returns callback values as an ordered tuple
after the same pre-execution ID validation. Existing orchestrators continue to
ignore that tuple. The SE/FC/Gather runner narrows and forwards its two result
dictionaries, and the lowerer's private wrapper returns them without changing
context construction or shared state scope. The complete orchestration corpus
confirms that result propagation is additive to existing behavior.

The remaining main and fallback SINet/SE-FC/Gather reconciliation boundaries
are now characterized together. Each boundary must record the tensor count,
retain the SINet shuffle result, unpack the two ordered cluster results, and
reconcile after any of the three exact rewrite counters or a tensor-count
decrease caused by zero-rewrite pruning. A strict structural expectation covers
both ModelIR targets and preserves the four-statement owner order. Production
remains unconditional at this checkpoint.

Both terminal boundaries now implement that four-statement contract. The main
and fallback paths record tensor count, retain the SINet result, unpack the two
cluster results, and reconcile only after a positive exact counter or pruning.
The owner order, main Session LayoutState, fallback `None` LayoutState, shared
diagnostics, and fallback/main separation remain unchanged. A lowerer fixture
covers all three counters plus prune-only behavior on the main path, while the
structural contract verifies identical fallback wiring.

The next safe late boundary contains static shape-signature sanitization,
exact rank-four binary adapter repair, singleton-broadcast adapter repair, and
one unconditional reconciliation. The sanitizer's `sanitized` counter and both
repair counters cover their graph/metadata mutations; the exact adapter owner
also prunes unused tensors on a zero rewrite. A new fixture freezes that prune
side effect. A strict structural expectation therefore requires all three
counters plus tensor-count reduction before this specific reconciliation may
be guarded. Production remains unchanged at this characterization checkpoint.

That late boundary now records tensor count and assigns all three results before
guarding its immediate reconciliation. A positive sanitizer or repair counter,
or a tensor-count decrease from exact-adapter pruning, preserves the scan; the
all-zero/no-prune path skips it. The surrounding first repair reconciliation
and later final signature sanitization are unchanged. Lowerer wiring covers all
three counters and prune-only behavior independently.

The earlier shared boundary also depends on the three-runner
singleton-channel/duplicate-fan-out/consecutive-Reshape cluster. Although the
generic recovery utility now preserves callback results, this specific runner
and the lowerer's private helper still discard them. A strict expected-failure
contract requires the ordered three-dictionary tuple without changing owner
order, shared state scope, arguments, diagnostics, or any call site. Production
remains unchanged at this characterization checkpoint.

`run_singleton_consecutive_reshape()` now forwards its three ordered result
dictionaries, and the lowerer's private helper preserves that tuple. All three
production call sites continue ignoring it, so this checkpoint changes neither
ModelIR nor reconciliation behavior. The complete orchestration corpus confirms
that tuple propagation is additive to the existing shared-state contract.

The earlier shared reconciliation can now be characterized over nine pure
mutation-result dictionaries: boundary-signature realignment, HardSwish shape,
Squeeze axes/shape, wrong-way Conv transpose, two binary repairs, and the three
singleton/consecutive cluster results. The empty-cluster contract freezes only
zero-valued mutation counters. A strict expected-failure boundary contract
requires all nine dictionaries to pass through one compact positive-count
predicate, plus tensor-count reduction for zero-rewrite pruning. Production is
unchanged at this characterization checkpoint.

The shared late boundary now records the tensor count, assigns all six direct
results, unpacks the three ordered cluster results, and guards only its
immediate reconciliation with `_stats_have_positive_count()` or a tensor-count
decrease. The helper accepts only pure mutation-count dictionaries. A synthetic
lowerer fixture independently covers all nine dictionaries and prune-only
behavior; every changed outcome adds exactly one reconciliation over the
all-zero/no-prune path. Structural tests that span extracted owners now resolve
their helper dependencies from the owning core module instead of the lowerer
compatibility surface.

The indexed binary-layout convergence coordinator is the next bounded scan
target. It currently executes all three broadcast-repair, stale-Transpose
repair, and static-shape reconciliation rounds even after a complete round
returns only zero mutation counters. All three owners return pure mutation
dictionaries; their zero paths do not prune tensors or mutate topology. Strict
expected-failure cases now require termination after the first stable round,
whether stability is immediate or follows one changing reconciliation round.
A separate passing contract preserves the existing three-round maximum while
shape reconciliation continues changing metadata. Production remains
unchanged at this characterization checkpoint.

The coordinator now adds all three current-round statistics before testing the
same pure mutation dictionaries with `_stats_have_positive_count()`. An
all-zero round terminates the loop; a changing repair or reconciliation keeps
the shared index and advances to the next round, still bounded by the existing
three-round cap. Immediate and second-round convergence fixtures are green,
the always-changing fixture still executes three rounds, and the original
multi-repair graph produces the same aggregate statistics and complete ModelIR
as the fixed three-round sequence.

The indexed dead-prune/shape/Reshape convergence helper is the next stable-scan
target. Its final reconciliation currently runs even when dead pruning, the
first reconciliation, and dynamic-Reshape resolution all return zero mutation
counters. A strict expected-failure fixture requires that complete zero path to
perform only the first reconciliation while retaining the one shared index.
Separate passing cases require the final reconciliation after a mutation from
each of the three owners, preserving possible second-order shape convergence.
Production remains unchanged at this characterization checkpoint.

The helper now initializes the final result with the exact zero counter and
uses `_stats_have_positive_count()` across prune, first-reconciliation, and
dynamic-Reshape results before invoking the second reconciliation. The stable
path therefore performs one shape scan, while any mutation retains the former
second scan and its aggregate statistic. Both production call boundaries,
LayoutState forwarding, and the one-index contract are unchanged. The original
dynamic-Reshape graph remains fully equivalent to the former four-call
sequence.

The first extra reconciliation in the indexed final shape/activation
coordinator is the next isolated scan. The preceding indexed shape-convergence
aggregate and HardSwish sanitizer both return pure mutation dictionaries. A
strict expected-failure fixture requires this reconciliation to be skipped when
both are completely zero, while passing fixtures retain it independently after
either predecessor changes. The later Reshape and activation-fusion
reconciliations remain explicitly present and ordered. Production is unchanged
at this characterization checkpoint.

`first_reconcile_stats` now starts with the exact zero counter and the first
extra scan runs only when the preceding convergence aggregate or HardSwish
sanitizer reports a positive mutation. Both changing paths preserve the scan;
the complete stable path proceeds directly to dynamic-Reshape resolution. The
later Reshape and fusion reconciliation boundaries, aggregate return schema,
shared index, and LayoutState forwarding remain unchanged. The full legacy
fixture retains identical ModelIR and statistics.

The second extra final-convergence reconciliation is now characterized over
its complete mutation interval. A strict expected-failure event-order fixture
requires it to be skipped when both the optional first reconciliation and
dynamic-Reshape resolution return zero. Separate passing paths retain it after
a first-reconciliation metadata change or a Reshape rewrite. The already
guarded first scan and the final post-fusion reconciliation remain unchanged in
production at this checkpoint.

`second_reconcile_stats` now starts with the exact zero counter and the second
scan runs only when the optional first reconciliation or dynamic-Reshape
resolver reports a positive mutation. Stable and predecessor-only paths proceed
directly to fusion; first-reconciliation metadata changes and Reshape rewrites
retain the scan. The final post-fusion reconciliation, aggregate return schema,
one-index ownership, and LayoutState forwarding remain unchanged. The complete
legacy fixture still produces identical ModelIR and aggregate statistics.

The final post-fusion reconciliation cannot use fusion counters alone.
Activation fusion returns complete per-family and aggregate rewrite counts but
unconditionally prunes unused tensors, including on a zero-rewrite call. A new
owner fixture freezes that prune-only path and LayoutState synchronization. A
strict expected-failure final-coordinator fixture requires the scan to be
skipped only when the second reconciliation and all fusion counters are zero
and the tensor table did not shrink. Passing paths retain it after a second
reconciliation change, fusion rewrite, or prune-only mutation. Production
remains unchanged at this characterization checkpoint.

The coordinator now records tensor count immediately before fusion,
initializes the final reconciliation result with the exact zero counter, and
guards the scan with `second_reconcile_stats`, every fusion counter, or a
tensor-count decrease. The stable path ends after fusion; second-scan metadata
changes, fusion rewrites, and zero-rewrite pruning each retain the final scan.
All three reconciliation guards remain ordered under one index, and the
complete legacy fixture preserves ModelIR and aggregate statistics.

The absolute-final placeholder-MatMul restoration block is the next bounded
reconciliation interval. A positive restoration still requires its first shape
scan. The immediately following exact and singleton binary repairs return pure
rewrite counters, while the exact owner may prune unused tensors on a zero
rewrite. A strict expected-failure lowerer fixture requires the second scan
only after a changing first reconciliation, either repair counter, or a
tensor-count decrease. Production remains unchanged at this characterization
checkpoint.

The restoration result is now assigned before the unchanged outer guard. A
positive restoration retains the first reconciliation, then records tensor
count and captures both binary-repair results. The second reconciliation runs
only after a changing first scan, either repair counter, or exact-repair
pruning. The following topology sort remains unconditional inside the outer
restoration block. Lowerer wiring covers unchanged, all three counters, and
prune-only behavior independently.

The conditional late binary-layout recovery sequence is the next extraction
boundary. It currently expands PReLU passthrough, dual pre-Add, terminal FC,
optional PReLU-BMM, affine pre/post, optional generic layout cleanup, and an
unconditional reconciliation inline. All owners expose rewrite counters, but
several prune on zero rewrites and generic layout cleanup also reports a
non-mutating iteration count. Strict expected-failure contracts require one
dedicated runner to preserve branch/order/context, return only mutation counts
plus net pruning, and let the lowerer reconcile only after a positive aggregate.
Production remains unchanged at this characterization checkpoint.

`run_late_binary_layout_recovery()` now owns that complete sequence in the
original order. The optional PReLU-BMM and generic layout cleanup owners remain
controlled by one `include_layout_transpose` flag, while the runner forwards
the shared LayoutState and diagnostics to the owners that accept them. Its
result contains the five owner rewrite counters, the four generic-layout
mutation counters, and `pruned_unused_tensors` from the net tensor-table
reduction. The non-mutating layout `iterations` field is deliberately omitted.

The lowerer branch is consequently reduced to one runner assignment and one
`_stats_have_positive_count()` guard. Static-shape reconciliation is skipped
when the complete recovery sequence is stable and retained after any rewrite
or zero-rewrite pruning. Dedicated runner tests fix owner order, optional-owner
behavior, context forwarding, normalized counters, pruning, and the empty-model
stable path. Architecture tests count calls at the new ownership boundary
rather than assuming every owner is invoked directly from the lowerer.

The next unconditional post-lowering reconciliation is not yet a safe local
optimization boundary: it follows a broad terminal phase whose owners do not
all return complete mutation or prune evidence. The next bounded extraction is
therefore the inline ONNX `Constant` lowering special case. Its current tensor
value, dtype, shape/signature, output, provenance, no-operator, and missing
`value` error behavior are fixed by characterization tests.

A strict expected-failure architecture contract requires one typed
`op_families.constant` owner called with only the ONNX node and
`LoweringContext`. The owner must preserve the existing collision and
placeholder replacement behavior, use an explicit `onnx.AttributeProto` cast
for protobuf attributes, and remain TensorFlow-independent. This removes the
protobuf stub's ambiguous `type[In]` inference from the central lowerer without
changing supported Constant attribute forms. Production remains unchanged at
this characterization checkpoint.

`op_families.constant.lower_constant_node()` now owns the tensor-valued
Constant path. The central node loop retains only the op-type guard, one typed
call with `node` and `ctx`, and `continue`. The owner uses an explicit
`onnx.AttributeProto` cast before reading `name` or `t`, removing the protobuf
stub ambiguity that Pylance reported in the central lowerer.

The extraction preserves new-tensor creation, in-place placeholder replacement,
legacy collision renaming, constants-map updates, dtype and shape/signature
normalization, provenance, the absence of a ModelIR operator, and the exact
missing tensor-`value` error. It deliberately does not add other ONNX Constant
attribute encodings or change registry dispatch. Focused owner and contract
coverage plus the core/architecture/TensorFlow-boundary gate are green.

The remaining special branch in the ONNX node loop performs demand-driven
shape reconciliation before Attention, Gather, GatherElements,
LayerNormalization, MatMul, and MultiHeadAttention. Characterization now fixes
its exact trigger: at least one wrapped input must still have rank zero or one,
must not carry constant data, and must have no raw shape hint. Non-target ops,
known rank-two-or-higher inputs, explicit rank-one hints, and constant tensors
remain no-ops.

A strict expected-failure architecture contract requires one
TensorFlow-independent `core.shape_readiness` owner to hold the target-op set,
unresolved-input predicate, and reconciliation call. The lowerer must call it
once with the wrapped node and `LoweringContext`, eliminating the per-node
nested closure while preserving the demand-driven scan boundary. Production is
unchanged at this characterization checkpoint.

`core.shape_readiness.reconcile_shape_sensitive_inputs_on_demand()` now owns
the complete policy. It performs one constant-time op-set check for every
non-Constant node, inspects inputs only for the six target ops, returns the
exact zero reconciliation dictionary on every stable path, and invokes the
existing static-shape owner only when an unresolved nonconstant input has no
raw shape hint.

The central loop now constructs `NodeView`, makes one typed `node`/`ctx` call,
and dispatches. The target set, raw-shape rules, constant exclusion, and nested
predicate are no longer duplicated there. Dedicated tests cover every target
op, all four no-op classes, the integration trigger before dispatch, and the
one-owner architecture contract.

The broad terminal phase still cannot drive its final reconciliation from a
complete aggregate, but its last composite cluster is now isolated as the next
result-propagation boundary. The late layout/mean/SPP/gather/constant-fold/cast
orchestrator builds five required invocations and one optional generic-layout
invocation. `run_recovery_invocations()` already returns their ordered values;
both the orchestrator and its private lowerer helper currently discard them.

Strict expected-failure cases require the exact five- or six-element tuple to
pass through both layers without reordering, filtering, or copying. This unit
does not capture the result at the production call site and does not change the
unconditional phase reconciliation. In particular, the optional layout result
still contains a non-mutating `iterations` field that must not later be treated
as mutation evidence. Production remains unchanged at this characterization
checkpoint.

`run_late_layout_mean_spp_gather_constant_cast()` now returns the raw ordered
tuple from `run_recovery_invocations()`, and its private lowerer helper returns
the same object. The required policy yields five dictionaries and the optional
layout policy yields six. Invocation order, one shared state scope, diagnostics,
and exception propagation remain unchanged.

The terminal production call is intentionally still an expression statement,
so this checkpoint changes no reconciliation or ModelIR behavior. Focused tests
verify both policies and both return boundaries. The next step must normalize
the optional layout dictionary by mutation keys before it can participate in a
phase aggregate; its `iterations` value is not a change counter.

The next cluster-local contract is a normalized mutation summary. Strict
expected-failure fixtures require a fixed schema for both policies: four
generic-layout mutation keys initialized to zero when layout cleanup is
disabled, every dictionary from the five required passes, and one
`pruned_unused_tensors` count. The generic layout `iterations` key must never
appear in the summary.

The terminal call site is also required to record tensor count immediately
before the cluster, capture the raw ordered results, and derive the summary from
their exact net tensor reduction. This remains observation-only: the following
Expand/Squeeze rewrite and unconditional phase reconciliation do not consume
the summary at this checkpoint. Production is unchanged.

`summarize_late_layout_mean_spp_gather_constant_cast_mutations()` now validates
the policy-specific tuple length, initializes the four layout mutation keys,
filters `iterations`, merges all five required result dictionaries, and clamps
the explicit prune count to a nonnegative integer. Both layout-enabled and
required-only policies therefore expose the same mutation-only schema.

The terminal call site records tensor count, captures the raw tuple, and stores
the derived summary in `_late_layout_cluster_stats` before the existing
Expand/Squeeze call. The leading underscore marks this as staged evidence until
the rest of the terminal phase has complete accounting. No phase guard or
reconciliation behavior changes in this checkpoint.

The adjacent terminal Expand/Squeeze-to-Reshape owner is the next evidence
boundary. It already returns `replaced_expand_dims_and_squeeze_with_reshape`
and `expand_dims_squeeze_rewrite_shape_tensors`, and it prunes only after a
positive rewrite. Its result is currently discarded immediately before the
same unconditional phase reconciliation.

A strict expected-failure structural contract requires that call to be assigned
to staged `_terminal_expand_squeeze_stats` while preserving its LayoutState and
the immediate reconciliation statement. This checkpoint does not use either
terminal evidence dictionary as a guard. Production remains unchanged.

The terminal call now assigns its unchanged two-counter dictionary to
`_terminal_expand_squeeze_stats`. The Session LayoutState argument, position
after `_late_layout_cluster_stats`, and immediately following unconditional
shape reconciliation are preserved. The underscored result is staged for a
future complete phase aggregate and is not yet consumed.

Moving backward, the late hard-activation/layout pair is the next complete
cluster boundary. Its required hard-activation owner and optional generic
layout owner both prune unused tensors even when rewrite counters are zero, and
the layout result also contains non-mutating `iterations`. Both runner layers
currently discard the ordered one- or two-result tuple.

Strict expected-failure contracts require raw tuple propagation, a fixed
mutation-only summary with four zero-default layout keys, and net tensor-prune
accounting. The lowerer must capture count/results/summary in three adjacent
statements between the existing Hardswish-SE rewrite and pre-Concat cleanup.
No phase reconciliation consumes this evidence yet.

The late hard-activation/layout runner now returns the recovery utility's raw
ordered result tuple. Its pure summary validates the policy-specific tuple
length, preserves the required hard-activation counters, emits four stable
layout mutation keys, excludes layout `iterations`, and records the cluster's
clamped net tensor reduction.

The lowerer stages the starting tensor count, raw results, and normalized
`_late_hard_activation_stats` at the original terminal call site. This remains
observation-only: pass selection and order, the adjacent Hardswish-SE and
pre-Concat rewrites, and every downstream reconciliation are unchanged.

Immediately before that cluster, the terminal Hardswish-SE owner returns one
rewrite counter but unconditionally prunes unused tensors. A zero counter is
therefore not sufficient evidence that the owner left ModelIR unchanged.
Strict characterization requires the production call to preserve its raw
counter while adding the exact net tensor reduction as
`pruned_unused_tensors`. The neighboring split/conv bridge owner and late
hard-activation cluster remain fixed boundaries; production is unchanged at
this checkpoint.

The production call now records the starting tensor count and builds
`_terminal_hardswish_se_stats` from the owner's unchanged rewrite dictionary
plus the clamped net tensor reduction. This captures zero-rewrite pruning
without changing owner semantics. The staged dictionary is not yet consumed by
a reconciliation guard, and both neighboring boundaries remain in their
original order.

The immediately preceding indexed split/conv/concat bridge owner prunes only
after a positive rewrite. Its single returned counter is therefore complete
mutation evidence, unlike the Hardswish-SE owner. Strict characterization
requires the terminal call to assign that unchanged result to staged
`_terminal_split_conv_concat_bridge_stats` between the late QKV cluster and the
Hardswish-SE tensor-count capture. Production remains unchanged in this
checkpoint.

The terminal invocation now assigns its unchanged owner result to
`_terminal_split_conv_concat_bridge_stats`. The other same-owner invocations
remain expressions, making the staging specific to the interval being
accounted. ModelIR and Session LayoutState forwarding and both neighboring
boundaries are unchanged; no reconciliation consumes the staged counter yet.

The late QKV cluster immediately before that owner has three production
policies. The terminal policy excludes the prefix and optionally runs generic
layout cleanup before the required bridge cleanup. Both active terminal owners
can prune with zero rewrite counters, while generic layout also reports the
non-mutating `iterations` metric. Strict characterization requires ordered raw
result propagation and one fixed mutation schema containing four layout, four
prefix, two bridge, and one net-prune counter. The terminal call must stage
count/results/summary without changing the other two production forms.

The QKV runner and lowerer delegate now return ordered raw results for every
policy. `summarize_qkv_attention_mutations()` validates the active tuple length,
fills inactive family keys with zeros, copies only ten declared mutation
counters, excludes `iterations`, and adds clamped net pruning. Only the
terminal form stages starting count, raw results, and `_late_qkv_stats`; the two
default forms continue to ignore their return values.

Immediately before late QKV, the shape-extract owner prunes only after a
positive rewrite, so its single returned counter fully describes mutation.
There are three production calls to this owner. Strict characterization selects
only the call between late SPP and the QKV starting-count capture and requires
it to assign `_late_pre_qkv_shape_extract_stats`. The other two calls must remain
untouched at this checkpoint.

That exact invocation now assigns its unchanged return dictionary to
`_late_pre_qkv_shape_extract_stats`. The earlier call and the later absolute-end
call remain discarded expressions. This preserves all three execution points
while giving the current terminal interval unambiguous mutation evidence.

The preceding late SPP pair contains two transactional owners. Each prunes only
after a positive rewrite, so their two returned counters completely describe
mutation and no separate tensor-count delta is needed. Strict characterization
requires ordered tuple propagation, exact two-result validation, and a fixed
two-counter summary staged between the preceding raw layout rewrite and the
pre-QKV shape-extract result.

The runner and lowerer delegate now return the ordered pair. The pure
`summarize_late_spp_concat_unary_conv_mutations()` helper validates exactly two
results and maps them to the two declared keys. Production stages
`late_spp_results` and `_late_spp_stats`; no unrelated fields or prune proxy are
added, and the pair remains observation-only.

The preceding strided-slice/pad/concat owner also prunes only after a positive
rewrite, so its one counter is complete evidence. Two direct production calls
surround the second terminal affine recovery. Strict characterization selects
only the second call and requires `_terminal_slice_pad_concat_stats` between
that recovery and `late_spp_results`; the first call remains an expression.

The second call now assigns its unchanged result to
`_terminal_slice_pad_concat_stats`. Boundary contracts distinguish the first
expression from the second assignment while resolving both to the same owner.
No tensor-count proxy is added, and the staged counter remains unused by a
reconciliation guard.

The preceding terminal affine recovery has eleven ordered child owners and
twelve declared mutation keys; the final sanitizer owns two keys. Child prune
conditions differ, so strict characterization requires raw tuple propagation,
exact eleven-result validation, fixed-key extraction, and one cluster-wide net
prune counter. Only the second of two production recovery calls may stage
count/results/summary; the first remains an expression.

The runner and lowerer delegate now return the raw eleven-result tuple. The
pure `summarize_terminal_affine_concat_split_mutations()` helper extracts only
the twelve declared keys and adds clamped net pruning. Production stages
starting count, `terminal_affine_results`, and `_terminal_affine_stats` only at
the second call; its forward-string annotation preserves the delegate's
context-only runtime load contract.

Immediately before the staged terminal affine cluster, the first direct
strided-slice/pad/concat call still discards its complete positive-only counter.
Strict characterization requires it to assign
`_pre_terminal_affine_slice_pad_concat_stats` between the preceding raw
transpose/Mul/Add owner and `terminal_affine_tensor_count`. The second direct
call retains its distinct staged target.

That first direct invocation now assigns its unchanged result to
`_pre_terminal_affine_slice_pad_concat_stats`. The later invocation continues
to assign `_terminal_slice_pad_concat_stats`, so the two observation points are
unambiguous while both still invoke the same owner with only ModelIR. Neither
counter is consumed by reconciliation, and execution order, rewrite guards,
pruning behavior, and generated artifacts remain unchanged.

Immediately before the first staged slice/pad/concat result, the indexed
affine post-ADD owner reports one fixed rewrite counter and prunes only after a
positive rewrite. Three direct production calls use this owner. Strict
characterization selects only the first call, between the channel-slice cluster
and `_pre_terminal_affine_slice_pad_concat_stats`, for a future
`_pre_terminal_affine_post_add_stats` assignment. The other two direct calls
and the orchestrated occurrence remain unchanged.

That selected call now assigns its unchanged owner result to
`_pre_terminal_affine_post_add_stats`. The channel-slice and first
slice/pad/concat boundary contracts identify the exact assignment, while the
two later direct calls remain expressions. No summary, tensor-count proxy, or
reconciliation consumer is added because the positive-only owner counter
already covers every mutation.

The preceding channel-slice/pad-Mul orchestration has two ordered child
results. The channel-slice child declares three rewrite counters and the
pad-Mul child declares one. All four underlying rewrite owners prune only after
a positive counter, so strict characterization requires ordered tuple
propagation and a fixed four-key summary without a tensor-count proxy. Only the
direct terminal invocation should stage results and summary; the callback used
inside terminal slice recovery remains behaviorally unchanged.

The orchestration runner and lowerer delegate now return the ordered pair, and
`summarize_channel_slice_pad_mul_mutations()` validates exactly two results
before extracting only the four declared counters. Production stages
`channel_slice_pad_mul_results` and
`_pre_terminal_channel_slice_pad_mul_stats` at the direct terminal call. The
terminal-slice callback still ignores the returned pair, and no reconciliation
consumer or tensor-count proxy is added.

The preceding composite pre-ADD owner has one direct production call, but it
unconditionally prunes unused tensors even when its rewrite counter is zero.
Its raw dictionary is therefore insufficient mutation evidence. Strict
characterization requires a starting tensor count followed by a dictionary
containing the unchanged owner counter and clamped net
`pruned_unused_tensors`, between terminal affine recovery and the staged
channel-slice results.

The direct call now records `pre_terminal_pre_add_tensor_count` and builds
`_pre_terminal_pre_add_stats` from the unchanged owner result plus clamped net
tensor reduction. This exposes zero-rewrite pruning without changing the
owner's unconditional cleanup. The three orchestration-owned occurrences and
all downstream reconciliation decisions remain unchanged.

Immediately before the pre-ADD count, the first of two terminal affine recovery
calls still discards its ordered eleven-result tuple. The existing fixed
twelve-key summary and cluster-wide net-prune accounting already provide the
complete contract. Strict characterization requires distinct
`pre_terminal_affine_tensor_count`, `pre_terminal_affine_results`, and
`_pre_terminal_affine_stats` assignments while preserving the independently
staged second occurrence.

The first recovery now stages those three assignments and reuses
`summarize_terminal_affine_concat_split_mutations()` with its own clamped
tensor-count delta. The second occurrence retains the separate
`terminal_affine_*` targets. Both summaries remain observation-only and no
reconciliation branch consumes either result.

Immediately before the first terminal affine count, the indexed dual-statistics
InstanceNorm residual/add/resize owner has the last of three direct production
calls. It returns one fixed counter and prunes only after a positive rewrite,
so its raw result is complete mutation evidence. Strict characterization
selects only that last direct call for a future
`_pre_terminal_affine_instancenorm_dualstats_stats` assignment; the earlier
direct and nested occurrences remain unchanged.

That selected call now assigns its unchanged raw result to
`_pre_terminal_affine_instancenorm_dualstats_stats`. The first terminal affine
count follows immediately, and the other three production occurrences retain
their existing forms. No tensor-count proxy or reconciliation consumer is
added because the counter already covers every owner mutation.

The immediately preceding indexed InstanceNorm residual/Mul/Concat/Conv owner
has the same four-occurrence shape: three direct calls and one nested call. Its
single counter is complete because pruning is positive-only. Strict
characterization selects only the last direct call, immediately before the
staged dual-statistics result, for
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`.

That last direct call now assigns its unchanged raw result to the staged target.
The two earlier direct calls and nested occurrence retain their previous forms,
and the dual-statistics assignment remains the immediate next owner. No
tensor-count proxy or reconciliation consumer is introduced.

The preceding indexed InstanceNorm post-transpose bias/add owner has four
direct production calls plus one nested call. Its single counter is complete
because pruning is positive-only. Strict characterization selects the third
direct call, immediately before the staged residual/Mul/Concat result, for
`_pre_terminal_affine_instancenorm_post_bias_stats`; the absolute-final fourth
direct call remains an expression.

That third direct call now assigns its unchanged result to the staged target.
The first direct call retains `_terminal_instancenorm_post_bias_stats`, the
second retains `_very_late_instancenorm_post_bias_stats`, the nested call
remains unchanged, and the absolute-final fourth call retains its distinct
target.
The staged counter is observation-only, and no tensor-count proxy is required
because pruning remains positive-only.

The preceding late-binary recovery is already a complete evidence boundary.
Its aggregate excludes non-mutating iteration metrics, exposes nine fixed
mutation counters plus clamped net tensor reduction, and its production guard
reconciles on any positive field. A structural contract connects that guarded
branch directly to the staged terminal post-bias result, so no duplicate
summary or counter is needed between them.

The same post-bias owner has a distinct absolute-final fourth direct call after
boundary-signature sanitization and affine post-ADD cleanup. Its positive-only
counter remains complete. Strict characterization requires that fourth call to
assign `_absolute_final_instancenorm_post_bias_stats` immediately before the
absolute-final normalization/attention pair, while preserving the separately
staged third call.

That absolute-final call now assigns its unchanged result to
`_absolute_final_instancenorm_post_bias_stats`. The first direct call retains
`_terminal_instancenorm_post_bias_stats`, the second retains
`_very_late_instancenorm_post_bias_stats`, the third retains
`_pre_terminal_affine_instancenorm_post_bias_stats`, and the fourth therefore
has an unambiguous observation point of its own. The result is not consumed by
reconciliation, and pass order, ModelIR/LayoutState forwarding, rewrite guards,
pruning, and generated artifacts remain unchanged.

Immediately before that result, the absolute-final indexed affine post-ADD
owner is the third of three direct production calls. Its single counter is
complete because pruning occurs only after a positive rewrite. Strict
characterization selects only that third call, after shape-signature
sanitization and before `_absolute_final_instancenorm_post_bias_stats`, for a
future `_absolute_final_affine_post_add_stats` assignment. The first
pre-terminal call retains its existing staged target and the intervening
very-late call remains an expression.

That absolute-final third call now assigns its unchanged raw result to
`_absolute_final_affine_post_add_stats`. It remains observation-only and is
followed directly by `_absolute_final_instancenorm_post_bias_stats`. The first
call retains `_pre_terminal_affine_post_add_stats`, and the second very-late
call remains an expression. No tensor-count proxy, reconciliation consumer,
pass invocation, or graph traversal is added.

The remaining second affine post-ADD call is a separate very-late occurrence
immediately after unbound-input layout-transpose repair and immediately before
the Gather/Constant normalization cluster. The same positive-only pruning
contract makes its raw counter complete. Strict characterization selects only
that second call for a future `_very_late_affine_post_add_stats` assignment,
while preserving the already staged first and third calls.

That second call now assigns its unchanged result to
`_very_late_affine_post_add_stats`. The three direct occurrences therefore have
distinct pre-terminal, very-late, and absolute-final targets. All remain
observation-only, and no reconciliation guard, tensor-count proxy, graph scan,
or pass invocation is added.

The following very-late Gather/Constant normalization runner has four ordered
child results: Gather-axis cleanup, three constant-input fold counters, two
redundant-Cast counters, and two normalization-Pad counters. The flatten
normalization owner prunes unused tensors even when its rewrite counter is
zero, so the raw tuple alone is incomplete. Strict characterization requires
tuple propagation, a fixed eight-key mutation summary plus clamped net tensor
reduction, and distinct lowerer tensor-count/result/summary assignments before
the final dynamic-Reshape resolution.

The runner now returns its existing four-result tuple. The pure
`summarize_very_late_gather_constant_normalization_mutations()` helper validates
that cardinality, extracts only the eight declared mutation counters, and adds
clamped `pruned_unused_tensors`. The lowerer stages
`very_late_normalization_tensor_count`,
`very_late_normalization_results`, and `_very_late_normalization_stats` around
the cluster. The summary remains observation-only and shares the existing pass
state; no owner is invoked twice.

The immediately following dynamic-Reshape resolver has one fixed
`resolved_dynamic_reshape_shapes` counter. It mutates only RESHAPE options,
shape-tensor metadata/data, and output metadata, increments the counter for
each changed operator, and performs no pruning or topology removal. Of two
direct lowerer calls, strict characterization selects only the very-late call
with `prefer_runtime_inferable_from_onnx_raw=True` for a future
`_very_late_dynamic_reshape_stats` assignment. The earlier core-cleanup call
remains unchanged, as do the two already consumed convergence-helper calls.

That selected call now assigns its unchanged raw result to
`_very_late_dynamic_reshape_stats`. It remains observation-only between
`_very_late_normalization_stats` and the indexed Conv-input adapter repair.
The earlier direct expression and both convergence-helper uses are unchanged;
no summary, tensor-count proxy, or reconciliation consumer is added.

The following indexed Conv-input adapter runner returns two fixed repair
counters, but both child owners invoke unused-tensor pruning even when their
counter is zero. Strict characterization therefore requires a starting tensor
count and a dictionary containing the unchanged two counters plus clamped net
`pruned_unused_tensors` at the direct very-late call. The independently consumed
`fallback_conv_input_stats` occurrence remains unchanged.

The direct call now records `very_late_conv_input_tensor_count` and builds
`_very_late_conv_input_stats` from the unchanged two counters plus clamped net
tensor reduction. The fallback call and its reconciliation guard remain
unchanged. The direct summary is observation-only and is followed by the same
stale channel-shuffle repair.

That following stale NCHW channel-shuffle owner has one production occurrence
and one fixed repair counter. It changes only Concat axis and associated tensor
metadata, counts each repair, and performs neither pruning nor topology
mutation. Strict characterization therefore selects the direct call for a
future `_very_late_stale_channel_shuffle_stats` assignment without a proxy or
summary adapter.

The call now assigns its unchanged raw result to
`_very_late_stale_channel_shuffle_stats`. The result remains observation-only
between `_very_late_conv_input_stats` and the NCHW
Concat/Transpose/Conv-axis repair. ModelIR/LayoutState/diagnostics forwarding
and the single invocation are unchanged.

The following NCHW Concat/Transpose/Conv-axis owner has three production
occurrences and one fixed repair counter. It changes only Concat options and
tensor shape metadata, counts each applied plan, and performs neither pruning
nor topology mutation. Strict characterization selects only the direct
very-late call for a future
`_very_late_concat_transpose_conv_axis_stats` assignment; the fallback and
final result assignments remain unchanged.

That selected call now assigns its unchanged raw result to
`_very_late_concat_transpose_conv_axis_stats`. It remains observation-only
between the staged channel-shuffle result and the NCHW
Concat/global-pool/Conv-axis owner. The fallback and final result assignments
and their reconciliation guards remain unchanged.

The following NCHW Concat/global-pool/Conv-axis owner has one production
occurrence and one fixed repair counter. It changes only Concat options and
tensor metadata, counts each applied plan, and performs neither pruning nor
topology mutation. Strict characterization therefore selects that call for a
future `_very_late_concat_global_pool_conv_axis_stats` assignment without a
proxy or summary.

The call now assigns its unchanged raw result to
`_very_late_concat_global_pool_conv_axis_stats`. It remains observation-only
between `_very_late_concat_transpose_conv_axis_stats` and the dynamic rank-one
Unsqueeze/Reshape-shape rewrite. ModelIR/LayoutState forwarding and the single
invocation are unchanged.

The following dynamic rank-one Unsqueeze/Reshape-shape owner has three
production occurrences and one fixed rewrite counter. It counts every
operator/tensor insertion and metadata-only rewrite and performs no pruning.
Strict characterization selects only the first very-late direct call for a
future `_very_late_dynamic_rank1_reshape_stats` assignment while preserving
the fallback and absolute-final expressions.

That selected call now assigns its unchanged raw result to
`_very_late_dynamic_rank1_reshape_stats`. It remains observation-only before
the existing static-shape reconciliation. The fallback and absolute-final calls
remain expressions, and no proxy, summary, or reconciliation consumer is
added.

The immediately following reconciliation owner exposes the legacy
`reconciled_static_tensor_shapes` count, but that key intentionally counts only
output tensor shape updates. A RESHAPE option or shape-parameter tensor can
change while the legacy count remains zero. Strict characterization therefore
preserves that key and requires an opt-in
`reconciled_static_shape_mutations` key that also covers parameter/option-only
changes. Only the very-late call requests it and stages
`_very_late_static_shape_stats`; existing callers retain their exact result
schema.

The reconciliation owner now counts all writes while performing its existing
fixed-point walk. The opt-in total includes output shape updates, operator
options, constant shape parameters, and direct tensor metadata. It adds no
fingerprint or graph traversal. Default callers still receive only
`reconciled_static_tensor_shapes`; the selected very-late call passes
`include_mutation_count=True` and stages `_very_late_static_shape_stats` with
both keys.

The absolute-final consecutive-Reshape owner exposes three counters covering
no-op removal, single-consumer chain removal, and fan-out bypass. Every graph
rewrite increments at least one of those counters, and unused-tensor pruning
plus optional layout-state synchronization occur only after a positive rewrite.
Strict characterization therefore preserves the existing three-counter guard
and requires a stable two-key `_final_consecutive_reshape_static_shape_stats`
value that receives the opt-in complete reconciliation result only inside that
guard. The following final SiNet owner remains adjacent and no sort or scan is
added.

The primary caller now initializes that stable result and replaces it with the
opt-in complete reconciliation dictionary only after the unchanged aggregate
guard succeeds. The runner result, duplicate positive fan-out accounting,
pruning/layout synchronization, and final SiNet boundary are unchanged.

The immediately preceding absolute-final PReLU owner deliberately prunes
unused tensors on every invocation, including zero-rewrite calls. Its caller
already samples the tensor-table size and reconciles after either a positive
rewrite counter or a net tensor reduction. Strict characterization preserves
that complete guard and requires a stable two-key
`_final_prelu_static_shape_stats` value that receives the opt-in complete
reconciliation result only inside the guard. The following consecutive-Reshape
owner remains adjacent.

The primary caller now initializes that stable result and replaces it with the
opt-in complete reconciliation dictionary only after the unchanged rewrite-or-
tensor-reduction guard succeeds. The unconditional owner prune, tensor-count
sample, raw result, and consecutive-Reshape boundary are unchanged.

The preceding final SiNet-shuffle plus SE/FC/Gather aggregate has three exact
rewrite counters. SiNet-shuffle prunes only after a positive rewrite, while the
SE/FC and Gather children can preserve legacy zero-rewrite pruning when their
candidate callbacks run. The existing aggregate-level tensor-count sample
captures those cleanup-only ModelIR changes. Strict characterization preserves
the counter-or-net-reduction guard and requires a stable two-key
`_final_se_fc_gather_static_shape_stats` value receiving the opt-in complete
reconciliation result only inside that guard. This mirrors the already-complete
recursive-fallback boundary and keeps final PReLU adjacent.

The primary caller now initializes that symmetric stable result and replaces it
with the opt-in complete reconciliation dictionary only after the unchanged
counter-or-tensor-reduction guard succeeds. No child invocation, pruning,
layout synchronization, scan, or fallback behavior changes.

The preceding final placeholder-MatMul block performs one reconciliation after
a positive restore and a conditional second reconciliation after exact or
singleton binary repair, cleanup-only tensor deletion, or a legacy output-shape
change from the first reconciliation. A complete first result cannot be passed
directly to the existing generic positive-count guard because its new
parameter-only key would broaden the second-scan condition. Strict
characterization therefore requires two stable complete results while
projecting only `reconciled_static_tensor_shapes` into the unchanged legacy
guard input. Both existing reconciliation calls opt into complete evidence; no
scan is added and the final SE/FC/Gather boundary stays adjacent.

The primary caller now initializes both stable results. After a positive
restore, it stores the complete first result and builds the one-key legacy guard
input from that in-memory dictionary. The unchanged inner guard stores the
complete second result when it already reconciles. No complete-only key can
trigger the second scan.

The preceding final mixed-singleton Concat owner counts every successfully
applied adapter/rewire plan and performs pruning plus optional layout-state sync
only after a positive count. All zero-result paths are true ModelIR no-ops.
Strict characterization preserves its single-counter guard and requires a
stable two-key `_final_mixed_singleton_concat_static_shape_stats` value that
receives the opt-in complete reconciliation result only inside that guard. The
following placeholder-MatMul block remains adjacent.

The primary caller now initializes that stable result and replaces it with the
opt-in complete reconciliation dictionary only after the unchanged exact
counter guard succeeds. No owner, scan, pruning, layout synchronization, or
placeholder boundary changes.

The preceding final rank-four channelwise broadcast owner counts every
in-place constant rotation and shared-constant clone/rewire, performs no prune,
and has no counter-zero ModelIR mutation. Strict characterization preserves its
single-counter guard and the existing reconciliation→topological-sort→layout-
inference order while requiring a stable two-key
`_final_broadcast_static_shape_stats` result. The following mixed-singleton
Concat owner remains adjacent.

The primary caller now initializes that stable result and replaces it with the
opt-in complete reconciliation dictionary as the unchanged first statement of
the positive guard. Topological sort and layout inference retain their exact
positions, and no scan is added.

The preceding final ConvInteger owner can propagate channel-last provenance as
a self-contained metadata/layout update without structurally repairing a
Transpose. The existing follow-up guard intentionally uses only
`repaired_channel_last_convinteger_input_transposes`, which counts every input
rewire, Transpose removal, associated tensor metadata update, prune, and layout
sync requiring reconciliation/sort/inference. Strict characterization keeps
the provenance counter outside that guard and requires stable two-key
`_final_convinteger_static_shape_stats` evidence only for the existing
repair-positive path. The following InstanceNorm owner remains adjacent.

The primary caller now initializes that stable result and replaces it with the
opt-in complete reconciliation dictionary as the unchanged first statement of
the structural-repair guard. Hint-only propagation still does not enter the
guard, and sort/layout inference retain their positions.

The immediately preceding dynamic rank-one Unsqueeze/Reshape-shape owner has
three production occurrences and one exact rewrite counter covering both
shape-parameter-only updates and inserted runtime SHAPE/CONCAT pipelines. It
does not prune. Very-late and recursive-fallback results are already retained;
strict characterization selects only the absolute-final direct expression for
an `_absolute_final_dynamic_rank1_stats` assignment while preserving the
following unconditional sort, layout inference, and ConvInteger boundary.

The absolute-final call now assigns its unchanged raw dictionary to that target.
The other two occurrences, all owner behavior, and the unconditional following
operations are unchanged; no guard, reconciliation, or traversal is added.

The immediately preceding absolute-final normalization/attention pair is a
two-pass recovery orchestration over one shared pass-state scope. Both owners
return stable mutation dictionaries and the common recovery runner returns
them as an ordered tuple. The pair-specific runner and lowerer helper now return
that tuple, and the primary caller retains it as
`_absolute_final_normalization_attention_results`. The adjacent post-bias and
dynamic-rank-one assignments remain fixed; no summarizer, guard,
reconciliation, or traversal is added.

Immediately before the absolute-final affine and normalization sequence, the
boundary-signature restore runs dynamic-map realignment followed by static
signature sanitization. The owners return, respectively, one exact mutation
counter and a four-counter repair/preservation dictionary. The absolute-final
caller now retains them as `_absolute_final_boundary_signature_stats` and
`_absolute_final_static_signature_stats`, matching the already retained other
production occurrences. Their adjacency and the following affine owner are
unchanged; no guard, reconciliation, scan, or traversal is added.

In the no-layout fallback only, the immediately preceding guarded final
cleanup reruns SE/FC layout propagation and strict constant affine pre/post
folding after an unconditional topological sort, then sorts again. Both owners
return stable one-counter dictionaries. This guarded occurrence now retains
them as `_no_layout_final_se_fc_stats` and
`_no_layout_final_affine_prepost_stats`. The guard, callback contracts,
surrounding sorts, and signature-restore boundary are unchanged; no guard-
external initialization or consumption is added.

Before the post-progress topological sort, the primary final precision sequence
rewrites safe constant divisors to reciprocal multiplication, folds consecutive
constant multiplications, then restores divisions on precision-sensitive
integer-cast lineages. Each owner returns a stable one-counter dictionary. The
primary caller now retains them as `_final_precision_div_rewrite_stats`,
`_final_precision_consecutive_mul_stats`, and
`_final_precision_div_restore_stats`. Earlier core-cleanup and recursive-
fallback calls and the following progress/sort boundary are unchanged.

The recursive safety fallback runs the same precision trio over `fallback_ir`
after its placeholder-MatMul reconciliation and topological sort, but without a
layout-state handoff; only the consecutive-Mul runner receives shared
diagnostics. The fallback retains the results as
`_fallback_precision_div_rewrite_stats`,
`_fallback_precision_consecutive_mul_stats`, and
`_fallback_precision_div_restore_stats`. Its exact callback contracts and the
following unbound-input repair boundary are unchanged.

The remaining earlier consecutive-Mul cleanup runs in the primary core-cleanup
phase after pseudo-LeakyReLU and YOLO-decode rewrites and before terminal
Transpose/Dequantize sanitization. It uses the same layout state, diagnostics,
and stable one-counter schema as the primary-final occurrence. Its result is
now retained as `_core_cleanup_consecutive_mul_stats`, so all three production
occurrences of the owner retain mutation evidence. Both adjacent cleanup
boundaries remain unchanged.

The two immediately preceding core-cleanup owners fuse guarded pseudo-LeakyReLU
subgraphs and fold YOLO decode `Mul(x,x)` plus anchor multiplication. Each has a
single direct lowerer occurrence and returns a stable one-counter dictionary.
The caller now retains them as `_core_cleanup_pseudo_leakyrelu_stats` and
`_core_cleanup_yolo_decode_stats`. The core-progress boundary, their exact
order, and the captured consecutive-Mul successor are unchanged.

Terminal quantization cleanup appears as two ordered pairs: once in core
cleanup and once after late recovery sweeps. Each pair normalizes terminal
Transpose/Dequantize boundaries, then transactionally removes exact-grid
terminal Quantize/Dequantize pairs before Conv-affine folding. The sanitizer
returns two counters and the runner one. The core and terminal callers retain
all four dictionaries under phase-specific targets. Both occurrence pairs,
their shared callback contracts, distinct preceding boundaries, and Conv-
affine successors are unchanged.

Each terminal quantization pair is followed by Conv MUL/ADD affine folding and
Conv/binary activation fusion. The affine owner returns four counters and has a
third later occurrence; activation fusion returns seven counters and occurs
only in these two pairs. The first two affine calls and both activation calls
retain their raw dictionaries under core-cleanup and terminal-cleanup targets.
Their phase-specific Q/DQ predecessors and distinct successors are fixed, while
the third affine occurrence remains unchanged for a separate audit.

The remaining affine occurrence uses the same four-counter owner and exact
Conv-ADD/layout-state contract. It retains its raw dictionary under a late
cost-volume-specific target immediately after the shared NDHWC-gate/cost-volume
state scope and immediately before construction of the late Concat layout-state
scope. Both boundaries remain fixed.

The late Concat state scope is consumed in order by axis-3 constant-Concat,
Dequantize/Concat/Quantize, LayerNorm-statistics, and generic Transpose-cleanup
runners. Their stable one-, one-, two-, and five-counter dictionaries are
retained under cluster-specific targets. The shared state scope, callback
contracts, following optimize-layout guard, and the two other lowerer
Transpose-cleanup occurrences remain unchanged.

The elementwise NHWC→NCHW fanout roundtrip owner returns one rewrite counter and
has two production occurrences under the same layout-optimization guard. The
first follows the captured late Concat cleanup dictionary; the second precedes
terminal singleton-MaxPool/Reshape orchestration. Both results are retained
under distinct phase targets inside their original guards. The model-only
callback contract and all four outer boundaries remain fixed.

The next adjacent ExpandDims and flatten-HW Transpose/Reshape compatibility
owners each return one rewrite counter, each has one production occurrence,
and both receive the live Session LayoutState. Their results are retained under
distinct late-layout targets. Their order between the captured late-Concat
fanout guard and the following NHWC-Reshape owner remains fixed.

The immediately following private rank-three layout-shim collapse owner returns
one rewrite counter, accepts only ModelIR, and has one direct production
occurrence. Its result is retained under a late NHWC-Reshape target between the
captured flatten-HW dictionary and the channel-shuffle/Gather cluster with both
optional shuffle families disabled. The model-only callback and boundaries are
unchanged.

The following channel-shuffle/Gather orchestrator selects two through seven
transactional child passes and internally receives their ordered result tuple.
The public runner and local helper propagate that typed tuple unchanged. The
guarded full-post and unguarded late-base production invocations retain it under
phase-specific targets without aggregation. Policy selection, ordering, shared
scope, and boundaries remain unchanged.

The following attention-QKV Reshape/Transpose compatibility owner returns one
rewrite counter, has one direct production occurrence, and receives the live
Session LayoutState. Its result is retained as
`_late_attention_qkv_reshape_stats` between the captured base-only
channel-shuffle/Gather tuple and attention Gather/Transpose/Reshape cleanup.
The value has no consumer, so result retention adds no graph work or policy.

The immediately following attention Gather/Transpose/Reshape cleanup owner
returns two pattern-specific rewrite counters, accepts only the ModelIR, and has
one direct late production call in addition to its recovery-runner selection.
The direct result is retained as `_late_attention_gather_cleanup_stats` between
the captured QKV dictionary and the live-LayoutState Gather-axis0 compatibility
owner. The retained value has no consumer, and the recovery-runner path remains
unchanged.

The following Gather-axis0 singleton-to-Reshape compatibility owner returns one
rewrite counter and receives the live Session LayoutState. It is selected by
the recovery runner and has one additional direct late production call whose
result is retained as `_late_gather_axis0_reshape_stats`. The target remains
between the captured attention-cleanup dictionary and the model-only attention
preprojection rank-lift owner. The retained value has no consumer, and the
recovery-runner path is unchanged.

The following attention-preprojection Reshape-to-BatchMatMul rank-lift owner
returns one rewrite counter, accepts only the ModelIR, and is selected by the
recovery runner in addition to one direct late production call. The direct
result is retained as `_late_attention_preproj_ranklift_stats` between the
captured Gather-axis0 dictionary and the live-LayoutState window-partition
owner. The retained value has no consumer, and the recovery-runner path remains
unchanged.

The following window-partition Reshape/Transpose-to-SpaceToDepth indexed owner
returns one rewrite counter and receives the live Session LayoutState. It is
selected by the recovery runner and has one additional direct late production
call whose result is retained as `_late_window_partition_stats`. The target
remains between the captured attention-preprojection dictionary and the window-
reverse owner. The retained value has no consumer, and the recovery-runner path
is unchanged.

The adjacent window-reverse Reshape/Transpose-to-DepthToSpace indexed owner
also returns one rewrite counter, receives the live Session LayoutState, and is
selected by the recovery runner in addition to one direct late production
call. The direct result is retained as `_late_window_reverse_stats` between the
captured window-partition dictionary and indexed final shape/activation
convergence. The retained value has no consumer, and the recovery-runner path
remains unchanged.

The following indexed final shape/activation convergence runner returns its
existing aggregate mutation dictionary and receives the live Session
LayoutState. Its sole production result is retained as
`_late_final_shape_activation_convergence_stats` between the captured window-
reverse dictionary and final boundary-input normalization with shared
LayoutState and diagnostics. The value has no consumer and adds no graph work.

`run_boundary_input_normalization_cleanup()` has two production occurrences
and returns one rewrite counter. The final occurrence follows indexed final
convergence and retains its result as
`_final_boundary_input_normalization_stats`; the earlier occurrence retains the
same schema as `_terminal_boundary_input_normalization_stats` between terminal
Softmax/Transpose cleanup and boundary-input Transpose/channel-slice rewriting.
Both values have no consumer, and both invocations preserve the shared live
LayoutState and diagnostics arguments.

The immediately following boundary-input Transpose/channel-slice owner returns
four mutation counters and has one production call with the live Session
LayoutState. Its result is retained as
`_terminal_boundary_input_channel_slice_stats` between captured terminal
boundary-input normalization and the first internal Transpose/channel-slice
propagation call. The retained value has no consumer.

The internal Transpose/channel-slice propagation owner also returns four
mutation counters and has two production calls. The first receives the live
Session LayoutState and retains its result as
`_terminal_internal_channel_slice_stats` between the captured boundary-input
channel-slice dictionary and the first Transpose/channel-slice MulAdd bridge.
The retained value has no consumer. The later model-only occurrence retains
the same schema as `_final_internal_channel_slice_stats` between captured final
boundary-input normalization and the later model-only MulAdd bridge. Neither
retained value has a consumer.

The following Transpose/channel-slice MulAdd-bridge owner returns one mutation
counter and likewise has two production calls. The first receives the live
Session LayoutState and retains its result as
`_terminal_channel_slice_muladd_bridge_stats` between captured terminal
internal channel-slice propagation and the first terminal slice/Concat
recovery sequence. The later model-only call retains its result as
`_final_channel_slice_muladd_bridge_stats` between captured final internal
channel-slice propagation and the later recovery sequence. Neither retained
value has a consumer.

The sole boundary-input Transpose/StridedSlice/QDQ/Concat production call
returns four mutation counters and receives the live Session LayoutState. Its
result is retained as `_terminal_boundary_stridedslice_qdq_concat_stats`
between the first terminal slice/Concat recovery sequence and the model-only
Swish-residual closure. The retained value has no consumer.

The immediately following model-only Swish-residual-closure owner returns four
mutation counters and has one production call. Its result is retained as
`_terminal_swish_residual_concat_closure_stats` between the
captured boundary StridedSlice/QDQ/Concat dictionary and the model-only
dequant-logistic-Mul-quantize bridge. The retained value has no consumer.

The immediately following dequant-logistic-Mul-quantize indexed owner returns
one mutation counter and has one model-only production call. Its result is
retained as `_terminal_dequant_logistic_mul_quantize_bridge_stats` between the
captured Swish-residual-closure dictionary and the model-only Swish-QDQ-island
owner. The retained value has no consumer.

The immediately following model-only Swish-QDQ-island owner returns five
mutation counters and has one production call using its default options. Its
result is retained as `_terminal_swish_qdq_island_stats` between the captured
dequant-logistic bridge dictionary and the live-LayoutState InstanceNorm post-
Transpose bias owner. The retained value has no consumer.

The following indexed InstanceNorm post-Transpose bias/add owner has four
direct production calls plus one nested convergence call. Its one-key result
is retained by all four direct calls. The first assigns
`_terminal_instancenorm_post_bias_stats` after
`_terminal_swish_qdq_island_stats`; the second assigns
`_very_late_instancenorm_post_bias_stats` in the very-late block between
diagnostics-aware pad-layout cleanup and the live-LayoutState
residual/Mul/Concat owner; and the two existing later targets remain distinct.
The nested convergence call and the following diagnostics-aware normalization-
pad cleanup boundary are unchanged.

The immediately following indexed InstanceNorm residual/Mul/Concat/Conv owner
has three direct production calls plus one nested convergence call. Its fixed
one-counter result is complete mutation evidence because pruning occurs only
after a positive rewrite. The first terminal direct call remains raw, the
second is retained as
`_very_late_instancenorm_residual_mul_concat_stats`, the third is retained as
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`, and the nested
call consumes its counter through the convergence guard.

The second direct call now assigns its unchanged result to that very-late
target. The captured `_very_late_instancenorm_post_bias_stats` predecessor and
following live-LayoutState dual-statistics InstanceNorm owner remain adjacent,
and the first, third, and nested occurrence contracts are unchanged.

The first direct occurrence now retains the terminal result as
`_terminal_instancenorm_residual_mul_concat_stats` between the live-LayoutState
residual/add-to-single-adapter and dual-statistics owners. The one-key positive-
only mutation contract, retained very-late and pre-terminal targets, and graph-
indexed nested convergence call remain fixed.

That following indexed dual-statistics InstanceNorm residual/add/resize owner
also has three direct production calls plus one nested convergence call. Its
fixed one-counter result is complete mutation evidence because pruning and
LayoutState synchronization occur only after a positive rewrite. The first
terminal direct call remains raw, the second is retained as
`_very_late_instancenorm_dualstats_stats`, the third is retained as
`_pre_terminal_affine_instancenorm_dualstats_stats`, and the nested call
consumes its counter through the shared convergence guard.

The second direct call now assigns its unchanged result to that very-late
target. The captured very-late residual/Mul/Concat predecessor and following
singleton consecutive-Reshape cluster remain adjacent, and the first, third,
and nested occurrence contracts are unchanged.

The first direct dual-statistics occurrence now retains the terminal result as
`_terminal_instancenorm_dualstats_stats` between
`_terminal_instancenorm_residual_mul_concat_stats` and the terminal boundary
cluster. The one-key positive-only mutation contract, retained very-late/pre-
terminal targets, and graph-indexed nested convergence call remain fixed.

That following singleton/consecutive-Reshape cluster returns three ordered,
pure mutation-count dictionaries for singleton-channel Transpose cleanup,
duplicate Reshape fan-out cleanup, and consecutive Reshape cleanup. All child
owners prune only after a positive counter. The private runner has three
production occurrences: the first model-level call retains its tuple as
`_very_late_singleton_consecutive_reshape_results`, the second model-level call
destructures all three dictionaries for the shared reconciliation guard, and
the conditional fallback call remains raw.

The captured very-late dual-statistics predecessor, following optional layout-
transpose cleanup branch, later destructuring assignment, and fallback
expression remain fixed. The retained tuple has no consumer and introduces no
graph work.

The QKV attention helper already returns an ordered tuple selected from layout-
transpose, prefix, and bridge child policies. Three direct production calls
exist. The late call is already retained as `late_qkv_results` and summarized
with an explicit net tensor-pruning delta; the two earlier default-policy calls
after the captured terminal and post-SiNet adj-flags dictionaries are raw.

A strict contract fixes future `_terminal_qkv_attention_results` and
`_post_sinet_qkv_attention_results` targets with empty arguments and keywords.
It preserves both adj-flags predecessors, their distinct successors, the total
three-call count, and the existing late policy/summary consumer. The two new
tuples must remain observation-only and must not replace the complete late
summary.

The immediately following guarded `run_layout_transpose_cleanup()` occurrence
returns `iterations` plus four rewrite counters. Those counters are useful
observability but are not complete mutation evidence: the low-level owner
prunes unused tensors unconditionally, including a zero-rewrite path, and the
result has no prune delta. Three direct lowerer occurrences exist; the earlier
layout block remains a guarded raw expression, while the late-Concat occurrence
retains its result with a shared state scope and the very-late guarded call now
retains `_very_late_layout_transpose_cleanup_stats`.

The `optimize_layout_transpose_chains` guard, captured singleton/consecutive
tuple predecessor, broadcast-constant repair successor, and other occurrence
forms remain fixed. The incomplete very-late result is observation-only and
has no reconciliation consumer.

The immediately following rank-four channelwise broadcast-constant repair
returns one complete rewrite counter. Each count corresponds to a constant
data/shape update or a shared-constant clone plus indexed input rewire, and the
owner has no cleanup-only mutation. Four production occurrences exist: indexed
binary convergence consumes one result at module scope; within
`lower_onnx_to_ir`, the very-late direct call now retains
`_very_late_broadcast_repair_stats`, and the fallback and final calls retain
their results for existing positive guards.

The guarded layout-Transpose predecessor, immediate static-shape
reconciliation successor, other three occurrence forms, and one-key schema
remain fixed. The new very-late target is observation-only and changes no
existing guard.

The immediate static-shape reconciliation remains unconditional because it
must cover every preceding owner, including the layout-Transpose cleanup whose
result omits prune-only mutation. Its default one-key result counts only tensor
shape updates and is not complete evidence. The reconciler's existing opt-in
`reconciled_static_shape_mutations` counter additionally covers parameter,
operator-option, and tensor-metadata writes without another graph traversal.

The unconditional call now requests `include_mutation_count=True` and retains
`_very_late_broadcast_static_shape_stats`. The captured broadcast-repair
predecessor and following tensor-count boundary remain fixed. No new guard or
consumer is introduced.

The later shared-late reconciliation is already guarded by nine pure mutation-
count dictionaries plus a tensor-count decrease that covers prune-only paths.
Runtime fixtures independently prove that every positive dictionary and the
prune delta add exactly one reconciliation over the all-zero/no-prune path.
Its execution predicate is therefore complete and must remain unchanged.

The call inside that existing guard now retains
`_shared_late_static_shape_stats` with `include_mutation_count=True`. All nine
evidence names, the tensor-count clause, following late-binary tensor-count
boundary, and the absence of a new consumer remain fixed.

The next late-binary reconciliation is also already guarded by complete
evidence: static-signature sanitization, rank-four binary adapter insertion,
singleton broadcast repair, and a tensor-count decrease that covers prune-only
cleanup. Its runtime fixture independently exercises every counter and the
prune path while preserving the all-zero skip.

That predicate remains unchanged, and its body now retains
`_late_binary_repair_static_shape_stats` with
`include_mutation_count=True`. The following optional late-binary layout-
recovery guard remains fixed, and the result has no consumer.

The following optional late-binary layout-recovery runner already returns a
complete aggregate that excludes iteration metrics and includes clamped net
tensor reduction. Its nested positive-count guard is covered by runtime rewrite,
prune, and stable outcomes and must remain unchanged.

The reconciliation inside that nested guard now retains
`_late_binary_layout_recovery_static_shape_stats` with
`include_mutation_count=True`. The surrounding option guard, recovery target,
positive predicate, and following terminal evidence boundary remain fixed.

The terminal Softmax/Transpose-after-NHWC-propagation indexed owner returns one
rewrite counter, receives the live Session LayoutState, and has one production
occurrence whose result is retained as `_terminal_softmax_transpose_stats`
between the diagnostics-aware Gather-channel-fanout runner and captured
terminal boundary-input normalization. The retained value has no consumer.

The immediately preceding diagnostics-aware Gather-channel-fanout runner
returns one rewrite counter. Its callback is also selected by two existing
orchestrators, while its sole direct production result is retained as
`_terminal_transpose_gather_channel_fanout_stats` between the live-LayoutState
ArgMax owner and captured terminal Softmax dictionary. The retained value has
no consumer, and orchestrated selections are unchanged.

The preceding terminal ArgMax owner returns the one-counter dictionary
`optimized_transpose_pre_argmax_nhwc_terminal_chains` from its sole production
call. That direct result is retained as `_terminal_pre_argmax_stats` between
captured terminal Conv-activation cleanup and captured Gather-channel-fanout
cleanup. The retained value has no consumer, and the live Session LayoutState
input is unchanged.

The preceding final decomposed-InstanceNorm owner prevalidates every constant
and tensor-shape plan, counts each candidate only after at least one planned
write is applied, performs no pruning or topology mutation, and synchronizes
layout only after a positive count. Strict characterization preserves its
single-counter guard and reconciliation→sort→layout-inference order while
requiring stable two-key `_final_instancenorm_static_shape_stats` evidence. The
following broadcast owner remains adjacent.

The primary caller now initializes that stable result and replaces it with the
opt-in complete reconciliation dictionary as the unchanged first statement of
the positive guard. The following sort and layout inference remain in place,
and no scan is added.

The terminal InstanceNorm residual-add-to-single-post-adapter owner returns the
fixed one-counter dictionary
`optimized_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains`.
It prunes unused tensors and synchronizes the live LayoutState only after a
positive rewrite, so that counter is complete mutation evidence for the owner.

Two production occurrences exist. Indexed binary-layout convergence consumes
the first result with `residual_graph_index`; the terminal direct call after
diagnostics-aware normalization/pad cleanup now retains its unchanged result as
`_terminal_instancenorm_residual_add_stats`. The contract fixes the live
LayoutState argument, retained terminal residual/Mul/Concat successor, and the
single graph-indexed nested occurrence. This is an assignment-only
orchestration change with no consumer or additional graph work.

The diagnostics-aware normalization/pad aggregate returns the fixed two-key
schema for decomposed InstanceNorm/Pad and flattened global-norm/Pad rewrites.
Both child owners prune unused tensors unconditionally after candidate
processing, while their counters report rewrites only. The aggregate is
therefore stable observation data but is not complete evidence for cleanup-only
mutation and must not be used by itself as a guard.

Within `lower_onnx_to_ir`, one loop-local result is consumed by convergence and
one terminal direct result after `_terminal_instancenorm_post_bias_stats`
is retained as `_terminal_normalization_pad_stats`. Two additional recovery
orchestrators select the same callback with flatten-only options and shared
pass-state scope. The contract preserves the loop-local consumer, orchestrated
selections, live LayoutState, diagnostics sink, and following captured
residual-add result. The retained dictionary remains observation-only and adds
no graph work.

The terminal boundary-layout orchestrator executes five ordered child runners
through one shared pass-state scope. `run_recovery_invocations()` already
returns their ordered result tuple. `run_terminal_boundary_layout()` and the
local `_run_terminal_boundary_layout_pass_cluster()` helper now transparently
return that tuple to the sole primary call.

The primary result is retained as `_terminal_boundary_layout_results` between
the captured terminal dual-statistics result and the optional terminal mean/
attention guard. Result propagation executes each child exactly once,
preserves the existing tuple order and shared scope, adds no consumer, and
leaves all mutation semantics unchanged.

The mean/attention orchestrator selects five to seven ordered child runners
from the independent LayerNorm and Conv-attention policy flags. The generic
recovery runner already creates the matching ordered result tuple, but
`run_mean_attention()` and the local
`_run_mean_attention_layout_pass_cluster()` helper now transparently return it.

The helper has two direct primary calls: the first enables LayerNorm and keeps
Conv-attention enabled, while the guarded terminal call disables
Conv-attention and keeps LayerNorm disabled. Two recovery contexts also retain
the helper as an argument-free callback and accept an arbitrary return value
without branching on it. The direct calls retain distinct
`_layout_pass_set_1_mean_attention_results` and
`_terminal_mean_attention_results` tuples. All four policy matrices, callback
references, shared scope, option guards, and adjacent calls remain fixed.

The BatchMatMul affine-transpose-input owner returns the fixed one-counter
`optimized_batchmatmul_affine_transpose_input_chains` dictionary. It rewrites
both affine inputs and `adjY` together, then prunes unused tensors
unconditionally. Its counter is therefore stable observation data but is not
complete evidence for cleanup-only pruning and must not guard later work.

Two direct production calls retain distinct results. The guarded terminal
occurrence follows `_terminal_mean_attention_results` and retains
`_terminal_batchmatmul_affine_input_stats`; the post-SiNet occurrence follows
SA/PA MirrorPad propagation and retains
`_post_sinet_batchmatmul_affine_input_stats`. Both precede the BatchMatMul
reshape/SE owner. Retention is assignment-only and observation-only and adds no
graph traversal or consumer.

The adjacent BatchMatMul reshape/SE owner returns the fixed one-counter
`optimized_batchmatmul_reshape_se_nhwc_chains` dictionary. It converts the
matched BATCH_MATMUL/RESHAPE/SE island atomically, then unconditionally prunes
unused tensors. Its rewrite counter is not complete evidence for cleanup-only
pruning and remains observation-only.

Two direct occurrences follow the captured terminal and post-SiNet affine-input
results and retain `_terminal_batchmatmul_reshape_se_stats` and
`_post_sinet_batchmatmul_reshape_se_stats`, respectively. Both precede the same
adj-flags owner. The retained values have no guard or consumer and add no graph
traversal.

The following BatchMatMul transpose-input-to-adj-flags owner returns the fixed
one-counter `optimized_batchmatmul_transpose_input_to_adj_flags` dictionary.
Each count follows the complete input rewrite or singleton-preserving RESHAPE
conversion plus the corresponding `adjX`/`adjY` toggle. Unused tensors are
pruned only after a positive rewrite, so the counter is complete owner mutation
evidence.

Two direct occurrences follow the captured terminal and post-SiNet reshape/SE
results and retain `_terminal_batchmatmul_adj_flags_stats` and
`_post_sinet_batchmatmul_adj_flags_stats`, respectively. Both precede QKV
attention clusters. The dictionaries have no consumer or new guard and add no
graph work.
