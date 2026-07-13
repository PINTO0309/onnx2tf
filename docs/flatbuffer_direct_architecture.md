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

A pass has a stable ID, phase, priority, maximum iteration count, and explicit
`changed` result. Repeating passes must use a graph fingerprint so a cycle
terminates deterministically. Risky rewrites use the transactional mode and
must leave the graph unchanged when invariant validation fails.

`GraphIndex` and `ModelIRGraphIndex` provide differential mutation contracts.
ONNX rewriters notify node input/output updates and node registration/removal;
ModelIR rewriters can replace inputs/outputs or insert/remove operators while
producer, consumer, duplicate-producer, and operator-position indices remain
consistent. A full `refresh()` is retained only for compatibility with
external mutations that bypass these APIs. Lineage-aware graph mutation
helpers accept an optional ModelIR index and update it atomically.

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

The adjacent Add/Concat/constant-suffix rule remains in the central lowerer
pending mechanical extraction. Its dedicated compact corpus fixes two branch
adapters, a shared base adapter, two Add fan-ins, channel Concat, rank-four
MUL/ADD suffix constants, inverse output adaptation, and downstream consumer
aliasing. Whole-ModelIR no-op cases cover branch and intermediate fan-out,
public pre/post outputs, invalid leading permutation, invalid Concat axis, and
a missing suffix constant. This frozen contract precedes any ownership or
mutation change.

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

Pad layout ownership is centralized in `passes/pad_layout.py`. In addition to
repairing a proven channel-last input/channel-first Pad mismatch, the module
owns direct inverse-transpose Pad folding and the guarded unary-to-Pad tail
rewrite that can retain one local NCHW adapter for legacy consumers. Padding
axis rotation, dynamic metadata, quantization, fan-out slots, and output names
are preserved; lowerer symbols remain compatibility wrappers.

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
compatibility helpers because they are not part of those production sequences.

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

CSP attention propagation is the final currently characterized large member
of `passes/attention_layout.py`. It validates both residual forms, expanded
HardSigmoid or sigmoid-self-Mul gates, singleton-spatial reshape adapters,
branch fan-out, and terminal layout before rewiring the region to NHWC.

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
proves the input is already NHWC. This eliminates the double transpose and
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

## Remaining refactoring order

1. Improve Tier 0-4 layout, transpose, broadcast, shape reconciliation, and
   fusion failures using semantic passes and focused ONNX fixtures.
2. Continue moving validation, capability selection, and lowering into
   op-family modules while preserving the current public API and artifacts.
3. Complete quantization, split/crop, custom/pseudo op, report, and requested-
   artifact-only regression coverage on the validated ModelIR contract.
4. Complete the shared PyTorch/TorchScript/Dynamo ONNX/ExportedProgram
   canonicalization and emitter separation.
5. Measure warm-run conversion time and peak RSS on the active Tier 0-4 set,
   then document improvements and remaining normalized failures.
