# flatbuffer_direct refactor handoff — 2026-07-12

## Current resumed checkpoint — `fb-refactor3`

Coverage and tensor-correspondence reporting have been extracted from
`lower_from_onnx2tf.py` into `tflite_builder/reporting.py`.

- Schema policy construction, node dispatch diagnostics, report assembly,
  lineage tracing, downstream correspondence inference, and both JSON writers
  now have one reporting owner.
- `_build_tensor_consumer_map` also moved to the reporting module and is
  imported back by the legacy lowerer for its existing layout passes.
- `lower_from_onnx2tf.py` retains the four public report functions as thin
  signature-preserving wrappers. Existing imports from both
  `lower_from_onnx2tf` and `tflite_builder` remain compatible.
- Imported ONNX-analysis and constant-fold helper symbols previously visible
  through `lower_from_onnx2tf.py` remain explicit compatibility re-exports.
- An architecture test verifies reporting implementation ownership, wrapper
  delegation, and the TensorFlow-free import boundary without imposing any
  source-line threshold.

Sequential verification in the core `uv` environment completed with:

- `33 passed, 749 deselected` for focused coverage/report integration;
- `35 passed` for reporting coverage plus architecture checks;
- `986 passed, 5 deselected, 2 warnings in 122.89s` for the full direct suite.

The five deselections and two FLOAT16 warnings are the same optional-environment
and expected-warning set documented below. No report schema, output path,
public signature, or direct conversion behavior changed.

Static high-rank BatchMatMul compression has also moved to
`passes/high_rank_matmul.py`. The legacy
`_compress_static_high_rank_batch_matmul` symbol remains a thin wrapper, so
existing tests and downstream imports continue to work. Its two dependencies,
`_is_fully_known_positive_shape` and `_prune_unused_tensors`, now have one
canonical implementation in `core/model_ir_utils.py`; the duplicate lowerer,
precision-pass, and constant-fold definitions were removed. A focused utility
test fixes deterministic pruning and lineage-event behavior, and an
architecture test fixes pass/helper ownership without a source-line gate.

Verification after this extraction completed with:

- `15 passed` for architecture and common ModelIR utility tests;
- `989 passed, 5 deselected, 2 warnings in 122.64s` for the full sequential
  direct suite.

The boundary input layout transpose pass now lives in
`passes/boundary_input_layout.py`, with the legacy lowerer symbol delegating to
it. `_build_tensor_consumer_map`, `_read_transpose_perm`, and
`_replace_tensor_inputs` are canonicalized in `core/model_ir_utils.py`; the
reporting and precision copies of the consumer map were removed. Focused tests
preserve the metadata-equality guard, the GatherND safety guard, deterministic
consumer indices, transpose permutation decoding, and lineage-aware input
replacement.

Verification completed with:

- `5 passed, 794 deselected` for focused boundary/report compatibility;
- `991 passed, 5 deselected, 2 warnings in 124.96s` for the full sequential
  direct suite.

The channel-slice and StridedSlice/QDQ/Concat boundary family now lives in
`passes/channel_slice_layout.py`. Five legacy lowerer entry points delegate to
that module: boundary channel-slice elision, internal channel-slice NHWC
propagation, Mul/Add bridge propagation, strict dual-Add bridge propagation,
and boundary StridedSlice/QDQ/Concat cleanup.

Their generic dependencies are canonicalized in `core/model_ir_utils.py`:
operator input/output mutation, indexed input replacement, constant-vector
read/write, static broadcasting, tensor metadata permutation, consumer maps,
transpose permutations, quantization cloning, pruning, and lineage events.
The family preserves all existing semantic guards and does not introduce a
model-name rule.

Verification completed with:

- `6 passed, 771 deselected` for focused channel-slice/StridedSlice and utility
  cases;
- `18 passed` for architecture and common ModelIR utility ownership;
- `992 passed, 5 deselected, 2 warnings in 125.00s` for the full sequential
  direct suite.

The boundary input Mul/Sum/Reshape and BatchMatMul rewrites now live in
`passes/boundary_input_chains.py`. The two legacy lowerer entry points are thin
delegating wrappers. The move retains the original fan-out, model-output,
permutation, constant-shape, axis, and metadata guards and reuses canonical
graph mutation and constant-vector helpers from `core/model_ir_utils.py`.
Focused ModelIR fixtures cover the positive rewrite paths, including NHWC
constant rotation, reduction-axis remapping, intermediate metadata updates,
and BatchMatMul input rewiring.

Verification completed with:

- `18 passed` for architecture and boundary-input-chain tests, including
  fan-out and shared-input no-op guards;
- `996 passed, 5 deselected, 2 warnings in 124.71s` for the full sequential
  direct suite.

The next extraction candidate should be selected from the remaining layout
rewrite families in `lower_from_onnx2tf.py`, favoring a cohesive group whose
generic graph helpers already have canonical core owners. Preserve the legacy
symbols as wrappers and add both semantic-guard fixtures and the existing full
direct-suite gate before committing.

The generic leading-input passthrough rewrite has subsequently moved to
`passes/input_passthrough_layout.py`. It folds a strictly linear sequence of
layout-agnostic unary and constant-side binary operators across a synthetic
input transpose and its inverse output transpose. The implementation is an
exact mechanical move apart from removing one unused local assignment. The
legacy lowerer symbol delegates to the pass module.

`_invert_perm` now has one canonical implementation in
`core/model_ir_utils.py`. Focused tests preserve the positive NHWC rewrite,
constant rotation and metadata behavior, the main-path fan-out no-op guard,
and invalid-permutation rejection. Architecture tests fix both pass and helper
ownership.

Verification completed with:

- `21 passed` for architecture, common ModelIR utilities, and leading-input
  passthrough behavior;
- `999 passed, 5 deselected, 2 warnings in 125.65s` for the full sequential
  direct suite.

The next cohesive extraction can extend `input_passthrough_layout.py` with the
adjacent ASIN/ERF/HardSwish/HardSigmoid semantic passthrough families. Move one
guarded family at a time, keep its legacy entry point, and gate each increment
with focused ModelIR fixtures before the sequential full suite.

The ASIN/ACOS decomposition passthrough is now the second implementation in
`passes/input_passthrough_layout.py`. It preserves the strict
`Mul(x,x) → Sub → Sqrt → Atan2` topology, singleton subtraction constant,
consumer, boundary, inverse-permutation, and output guards. Its legacy lowerer
entry point delegates to the pass module. `_is_singleton_constant_tensor` now
has one canonical implementation in `core/model_ir_utils.py` while remaining
available through the legacy lowerer import surface.

Verification completed with:

- `24 passed` for architecture, common ModelIR utilities, generic input
  passthrough, and ASIN positive/no-op behavior;
- `1002 passed, 5 deselected, 2 warnings in 125.00s` for the full sequential
  direct suite.

The standalone HardSigmoid passthrough has also moved to
`passes/input_passthrough_layout.py`. It retains the strict singleton-side
`Mul → Add → Relu0To1` or `Mul → Add → Maximum → Minimum` decomposition,
single-consumer, inverse-permutation, and metadata guards. The legacy lowerer
entry point remains a thin wrapper. Positive and noninverse-permutation no-op
fixtures cover the Relu0To1 path.

Verification completed with:

- `26 passed` for the focused architecture, utility, and input-passthrough
  suite;
- `1004 passed, 5 deselected, 2 warnings in 125.56s` for the full sequential
  direct suite.

The ERF polynomial decomposition passthrough now lives in
`passes/input_passthrough_layout.py`. Its legacy symbol delegates to the exact
moved implementation. The pass preserves the ABS/SIGN branch split,
reciprocal prelude, square/exponential branch, four-stage Horner polynomial,
final sign merge, singleton constants, exact consumer counts, output boundary,
and inverse-permutation guards. A generated ModelIR fixture exercises the full
21-operator topology, and a non-singleton coefficient fixture fixes the no-op
path.

Verification completed with:

- `28 passed` for the focused architecture, utility, and input-passthrough
  suite;
- `1006 passed, 5 deselected, 2 warnings in 125.19s` for the full sequential
  direct suite.

The pseudo-expanded HardSwish passthrough now also lives in
`passes/input_passthrough_layout.py`. It preserves the residual
`Add → optional Relu6 → Div-or-Mul → Mul(original, branch)` topology,
singleton constants, strict consumers, inverse terminal permutation, metadata,
and output-name guards. Focused fixtures cover the Relu6/Div positive path and
a non-singleton divisor no-op path.

Verification completed with:

- `30 passed` for the focused architecture, utility, and input-passthrough
  suite;
- `1008 passed, 5 deselected, 2 warnings in 125.48s` for the full sequential
  direct suite.

The larger HardSigmoid-plus-residual-Mul passthrough now completes the current
`passes/input_passthrough_layout.py` family. The implementation is moved
mechanically and the legacy symbol delegates to it. Existing characterization
coverage preserves the expanded clamp, residual Mul, legacy fan-out adapter,
optional Mean output, reduction-axis remapping, metadata, and output-name
behavior. Architecture coverage fixes module ownership.

Verification completed with:

- `31 passed, 758 deselected` for focused architecture, utility,
  input-passthrough, and legacy fan-out characterization;
- `1008 passed, 5 deselected, 2 warnings in 125.54s` for the full sequential
  direct suite.

Pad layout is now the next cohesive family. The direct inverse-transpose Pad
rewrite and unary-to-Pad tail rewrite moved from the lowerer into the existing
`passes/pad_layout.py`, alongside the channel-last-input repair. Legacy lowerer
entry points delegate to the module. Existing characterization fixtures retain
padding-axis rotation, inverse permutations, dynamic metadata, quantization,
legacy fan-out slots, the optional local NCHW adapter, and output naming.

Verification completed with:

- `18 passed, 759 deselected` for focused architecture, Pad layout repair, and
  Pad pre/post characterization;
- `1009 passed, 5 deselected, 2 warnings in 125.61s` for the full sequential
  direct suite.

The guarded `Transpose → Pad → Mul → Transpose → Add` rewrite has now joined
`passes/pad_layout.py`. The exact implementation moved behind a legacy wrapper.
Its existing characterization preserves broadcast proof, Pad-axis and Mul
constant rotation, shared-constant cloning, inverse permutations, metadata,
and output rewiring.

Verification completed with:

- `19 passed, 758 deselected` for focused architecture, Pad repair, and the
  Pad/Mul/Add characterization;
- `1009 passed, 5 deselected, 2 warnings in 125.75s` for the full sequential
  direct suite.

The normalization-subgraph Pad rewrite has now moved into
`passes/pad_layout.py`. Its helper routines were already nested and therefore
moved with the pass as one semantic unit. Four existing characterization
fixtures preserve reduction topology, channelwise constants, shared-constant
cloning, axes and padding remapping, fan-out adapters, quantization, metadata,
and output wiring. The lowerer retains only a compatibility wrapper.

Verification completed with:

- `22 passed, 755 deselected` for focused architecture, Pad repair, and
  normalization-subgraph Pad characterization;
- `1009 passed, 5 deselected, 2 warnings in 125.27s` for the full sequential
  direct suite.

The InstanceNorm decomposition followed by Pad now also belongs to
`passes/pad_layout.py`. The exact pass and its nested topology helpers moved as
one unit. Three existing characterizations preserve reduction axes, epsilon
and channel coefficients, Pad rotation, legacy consumer adapters, quantization,
metadata, and terminal output wiring. The legacy lowerer symbol delegates.

Verification completed with:

- `18 passed, 759 deselected` for focused architecture, Pad repair, and
  InstanceNorm/Pad characterization;
- `1009 passed, 5 deselected, 2 warnings in 125.74s` for the full sequential
  direct suite.

The flatten/global-normalization followed by Pad rewrite now lives in
`passes/pad_layout.py`. Its exact implementation and nested helpers moved as a
single unit behind the legacy wrapper. The existing full-topology fixture
preserves reshape targets, reduction axes, reciprocal and affine branches,
layout-sensitive constants, Pad rotation, metadata, and terminal output
wiring.

Verification completed with:

- `19 passed, 758 deselected` for focused architecture, Pad repair, and the
  flatten/global-normalization characterization;
- `1009 passed, 5 deselected, 2 warnings in 126.32s` for the full sequential
  direct suite.

The core Pad family is now substantially centralized. Before adding another
large pattern, audit remaining Pad-named passes for whether they truly share
this phase and helper contract or belong to attention/slice-specific families.

That audit classified the mixed Mean/ReduceMax/Concat/MirrorPad rewrite as an
attention layout pass rather than a core Pad pass. It now starts
`passes/attention_layout.py`, with the legacy lowerer symbol delegating to it.
The matcher is generic: it uses branch topology, reduction axes, inverse
permutations, MirrorPad pairs, and the Conv consumer, with no model-name guard.
The existing full-topology characterization preserves axis and padding
rotation, metadata, and terminal transpose removal.

Verification completed with:

- `17 passed, 758 deselected` for focused architecture and mixed-attention
  MirrorPad characterization;
- `1010 passed, 5 deselected, 2 warnings in 125.93s` for the full sequential
  direct suite.

The characterized QKV Slice canonicalization pair now also lives in
`passes/attention_layout.py`. The first pass replaces compatible Slice branches
with Gather/Reshape views, and the second replaces a compatible three-way QKV
Slice fan-out with Split. Both exact implementations moved with their nested
shape helpers; legacy lowerer names delegate. Existing tests cover each pass
individually and their ordered interaction on a shared graph.

Verification completed with:

- `18 passed, 757 deselected` for focused architecture and QKV Slice
  canonicalization;
- `1010 passed, 5 deselected, 2 warnings in 125.91s` for the full sequential
  direct suite.

The shared pre-Transpose QKV Slice rewrite now also belongs to
`passes/attention_layout.py`. Its implementation and nested begin/size helper
moved intact behind the legacy wrapper. The characterization fixes the shared
permutation, NCHW Slice-vector remapping, cloned shared constants, tensor
metadata, quantization, and removal of only the redundant transpose.

Verification completed with:

- `17 passed, 758 deselected` for focused architecture and shared
  pre-Transpose QKV slicing;
- `1010 passed, 5 deselected, 2 warnings in 125.56s` for the full sequential
  direct suite.

The QKV weighted-sum NHWC bridge now also belongs to
`passes/attention_layout.py`. The exact implementation and its nested constant
helper moved behind the legacy wrapper. Existing characterization preserves
the QKV producer topology, scalar weights, reduction/merge chain, cloned
shared constants, metadata, quantization, and terminal NHWC rewiring.

Verification completed with:

- `17 passed, 758 deselected` for focused architecture and the QKV weighted
  sum bridge;
- `1010 passed, 5 deselected, 2 warnings in 125.36s` for the full sequential
  direct suite.

The QKV Gather/Reshape/Transpose hoist now also lives in
`passes/attention_layout.py`. It preserves compatible two- and three-branch
projection forms, Gather axes, shape guards, fan-out, output contracts, and
ordered interaction with the Slice canonicalization passes. The move is exact
apart from deleting one pre-existing unused local variable.

Verification completed with:

- `17 passed, 758 deselected` for focused architecture and QKV hoisting;
- `1010 passed, 5 deselected, 2 warnings in 125.59s` for the full sequential
  direct suite.

The Conv-based attention NHWC propagation family now lives in
`passes/attention_layout.py`. Its implementation and nested matchers moved
intact behind the legacy wrapper. Four characterization topologies preserve
the basic reduction branch, expanded HardSigmoid gate, HardSwish activation,
self-HardSwish/Mean chain, fan-out adapters, constants, axes, metadata,
quantization, and output rewiring.

Verification completed with:

- `20 passed, 755 deselected` for focused architecture and all Conv-attention
  variants;
- `1010 passed, 5 deselected, 2 warnings in 125.58s` for the full sequential
  direct suite.

The CSP attention NHWC propagation family now lives in
`passes/attention_layout.py`. Its exact implementation and nested gate/reshape
matchers moved behind the legacy wrapper. Two characterized residual forms
preserve optional main Add behavior, HardSigmoid or sigmoid-self-Mul gates,
singleton-spatial reshape adapters, branch fan-out, constants, metadata,
quantization, and terminal output wiring.

Verification completed with:

- `18 passed, 757 deselected` for focused architecture and both CSP-attention
  variants;
- `1010 passed, 5 deselected, 2 warnings in 125.86s` for the full sequential
  direct suite.

All currently characterized attention-named families have an explicit module
owner. Uncharacterized attention mega-patterns remain in the legacy lowerer
until focused semantic fixtures are added. Re-audit the remaining generic
layout passes for the next cohesive family.

The first phase-3 GraphIndex enhancement is now implemented. `GraphIndex`
supports differential ONNX node update/register/unregister notifications.
`ModelIRGraphIndex` supports indexed input/output replacement and operator
insert/append/remove while maintaining producer, consumer,
duplicate-producer, and operator-identity maps. The canonical lineage-aware
mutation helpers accept an optional index and update graph state atomically.

The mixed Mean/ReduceMax/Concat/MirrorPad attention pass is the first migrated
consumer: it builds its ModelIR index once, performs input rewrites through the
indexed helpers, and removes the terminal transpose through the index instead
of rebuilding producer/consumer maps after a successful rewrite.

Verification completed with:

- `12 passed` for core GraphIndex and invariant tests;
- `23 passed, 771 deselected` for focused index, mutation-helper,
  architecture, and migrated-attention integration;
- `1014 passed, 5 deselected, 2 warnings in 125.18s` for the full sequential
  direct suite.

Next, migrate another small characterized pass to the same differential index
contract and measure map-build reductions before extending the API. Structural
mutations that still bypass the index must continue using `refresh()` until
they are converted explicitly.

Duplicate Transpose fan-out cleanup is now the second differential-index
consumer and the first member of `passes/graph_cleanup.py`. The canonical bulk
input replacement helper can update only indexed consumer operators, while
`ModelIRGraphIndex.remove_operator` maintains shifted producer/consumer and
operator-identity references. The legacy lowerer entry point delegates to the
new module. An instrumented fixture requires exactly one `refresh()`—the index
construction—through a successful deduplication.

Verification completed with:

- `25 passed, 770 deselected` for focused GraphIndex, mutation-helper,
  architecture, and duplicate-cleanup behavior;
- `1016 passed, 5 deselected, 2 warnings in 126.24s` for the full sequential
  direct suite.

Duplicate Reshape fan-out cleanup now also lives in
`passes/graph_cleanup.py`. It compares targets from `newShape` options or
constant shape tensors, preserves public outputs, merges compatible tensor
metadata, rewires only indexed consumers, and removes duplicates through the
structural index. Its instrumented fixture also requires one initial
`refresh()` and no post-rewrite full rebuild.

Verification completed with:

- `20 passed, 758 deselected` for focused architecture and duplicate Reshape
  cleanup;
- `1017 passed, 5 deselected, 2 warnings in 126.05s` for the full sequential
  direct suite.

The guarded Maximum/Minimum clamp fusion now also lives in
`passes/graph_cleanup.py`. It proves singleton finite constants equal to zero
and one, exclusive intermediate consumption, and output safety before replacing
the chain with `Relu0To1`. The surviving operator input and removed producer
are applied through the differential index, and its instrumented fixture
requires one initial refresh. `_read_singleton_constant_float` now has one
canonical implementation in `core/model_ir_utils.py`.

Verification completed with:

- `21 passed, 765 deselected` for focused singleton utility, architecture, and
  clamp cleanup;
- `1018 passed, 5 deselected, 2 warnings in 125.98s` for the full sequential
  direct suite.

The Squeeze/Reshape round-trip identity cleanup now also belongs to
`passes/graph_cleanup.py`. It validates explicit or inferred squeeze axes,
singleton dimensions, dynamic-compatible squeezed shape, and restored output
shape. Indexed consumer replacement and descending structural removals require
only one initial refresh. `_normalize_squeeze_axes_for_rank` now has one
canonical implementation in `core/model_ir_utils.py` for all remaining users.

Verification completed with:

- `22 passed, 765 deselected` for focused squeeze-axis utility, architecture,
  and indexed round-trip cleanup;
- `1019 passed, 5 deselected, 2 warnings in 126.92s` for the full sequential
  direct suite.

The three lowerer locations that consecutively invoked duplicate Transpose and
Reshape cleanup now call `run_duplicate_fanout_cleanup`. This is the first
post-lowering ordered pass group: it registers stable IDs in
`POST_LOWERING_CLEANUP`, shares one differential index, validates invariants
without rebuilding the index, and enables transactional deep-snapshot rollback.
The Transpose pass remains conditionally disabled for QDQ graphs while Reshape
cleanup always runs, preserving the previous call contract.

Verification completed with:

- `27 passed, 766 deselected` for focused ordered-pass, shared-index,
  architecture, and rollback behavior;
- `1021 passed, 5 deselected, 2 warnings in 129.92s` for the full sequential
  direct suite.

LayoutState is now connected to the ordered cleanup group. It supports full
phase-boundary synchronization from ModelIR, rename/remove operations, and
two-way consistency validation. `ConversionSession.refresh_indexes()` refreshes
both the ONNX index and layout state. Canonical tensor pruning and global rename
accept the optional state, and the duplicate cleanup group receives
`session.layout_state` from the lowerer. Success removes stale tensor entries;
transaction rollback restores and resynchronizes layout state before returning.

Verification completed with:

- `26 passed, 776 deselected` for focused session, LayoutState, rename/prune,
  ordered-cleanup, and architecture behavior;
- `1023 passed, 5 deselected, 2 warnings in 130.30s` for the full sequential
  direct suite.

`PassSpec` now has an explicit precondition callback. A false prerequisite
returns a zero-iteration `skipped_by_precondition` result without fingerprint,
deep snapshot, callback, or validation work. The ordered duplicate group uses
cheap Transpose-key and Reshape-input candidate scans, so transaction snapshots
are created only for graphs that can contain a duplicate. Stable zero-valued
statistics are still returned for skipped passes.

Verification completed with:

- `19 passed` for core pass-manager and graph-cleanup precondition behavior;
- `1024 passed, 5 deselected, 2 warnings in 130.57s` for the full sequential
  direct suite.

`ModelIRPassState` now centralizes one graph index, one LayoutState, combined
validation, deep snapshots, and rollback resynchronization for ordered ModelIR
pass groups. The duplicate cleanup runner uses this shared state instead of its
previous local transaction plumbing.

The mixed Mean/ReduceMax/Concat/MirrorPad rewrite is the first layout-sensitive
ordered group. All six lowerer call sites now use
`run_mixed_attention_layout_cleanup`, whose stable pass ID is
`layout.mixed_attention_mirrorpad` in `LAYOUT_PLAN`. A cheap topology
precondition avoids irrelevant snapshots. On success, changed logical layouts
are written to the session-owned LayoutState, pruned tensors are removed from
it, and graph/layout invariants are validated transactionally.

Verification completed with:

- `21 passed, 775 deselected` for focused shared-state, ordered attention,
  LayoutState, cleanup, and architecture behavior;
- `1025 passed, 5 deselected, 2 warnings in 128.05s` for the full sequential
  direct suite.

Boundary input adapter cleanup is now the second independent layout-sensitive
ordered group. The lowerer calls `run_boundary_input_layout_cleanup`, whose
stable ID is `layout.boundary_input_adapter` in `LAYOUT_PLAN`. The pass uses
indexed consumer replacement and operator removal, retains the Gather/Slice
safety guards and public input metadata, removes pruned internal tensors from
the session LayoutState, and validates through ModelIRPassState. Its focused
fixture requires one initial graph-index refresh.

Verification completed with:

- `20 passed, 757 deselected` for focused boundary runner, public-contract,
  GatherND no-op, LayoutState, and architecture behavior;
- `1026 passed, 5 deselected, 2 warnings in 128.23s` for the full sequential
  direct suite.

`run_model_ir_pass_group` now centralizes state creation, ordered spec
registration, execution, default-stat preservation, and removal of manager
control fields from semantic diagnostics. Duplicate Transpose/Reshape cleanup,
mixed attention layout, and boundary input layout all use this helper. Each
family retains its own candidate guard, rewrite callback, phase, stable pass
ID, transaction policy, and legacy integer result shape, so this removes only
repeated orchestration and does not merge semantic rules.

Verification completed with:

- `22 passed, 776 deselected` for the focused common runner, duplicate cleanup,
  mixed-attention, boundary-input, and architecture suite;
- `1027 passed, 5 deselected, 2 warnings in 128.97s` for the full sequential
  direct suite.

The next checkpoint should inventory the remaining direct calls to legacy
post-lowering/layout rewrite functions and select one small, already
characterized family for migration onto `PassSpec` and
`run_model_ir_pass_group`. Prefer a family that can reuse the session-owned
GraphIndex/LayoutState and add a cheap structural precondition. Preserve its
legacy wrapper and run the same focused and full sequential gates before
committing.

That inventory selected scalar clamp canonicalization because it has one
production call, one generic topology, an existing differential-index
implementation, and focused characterization. Production now calls
`run_clamp_cleanup` at the same terminal pipeline position. Its stable ID is
`canonicalize.scalar_clamp_relu0to1`; it shares ModelIRPassState's graph index
and the session LayoutState, runs transactionally, and skips the deep snapshot
when no Maximum-to-Minimum edge exists. The raw
`_optimize_maximum_minimum_relu0to1_chains` signature remains compatible while
accepting optional shared index/layout state internally.

Verification completed with:

- `4 passed, 21 deselected` for scalar-clamp positive/no-op behavior, one-index
  construction, LayoutState pruning, precondition snapshot avoidance, and
  architecture ownership;
- `1029 passed, 5 deselected, 2 warnings in 130.48s` for the full sequential
  direct suite.

The next small migration candidate is Squeeze/Reshape identity cleanup, but it
has eight production invocations. Before changing those calls, map their phase
positions and decide whether one runner per existing position preserves the
intended fixed-point behavior or whether adjacent calls can safely share a
group. Do not collapse those invocations without digest/runtime evidence.

The eight Squeeze/Reshape identity calls occur in distinct recovery sweeps,
including two calls separated by InstanceNorm rewriting. Because intervening
passes can expose new round trips, their positions and count were preserved
one-for-one instead of being collapsed. They now call
`run_squeeze_reshape_identity_cleanup` with the session LayoutState. Its stable
ID is `cleanup.squeeze_reshape_identity` in `POST_LOWERING_CLEANUP`; a
single-consumer Squeeze-to-Reshape precondition avoids snapshots on irrelevant
graphs, while the raw compatibility helper accepts an optional shared index
and layout state.

Verification completed with:

- `4 passed, 23 deselected` for ordered Squeeze/Reshape positive/no-op
  behavior, one-index construction, LayoutState pruning, snapshot avoidance,
  and architecture ownership;
- `1031 passed, 5 deselected, 2 warnings in 129.68s` for the full sequential
  direct suite.

The next step should avoid migrating another large layout rule immediately.
First inspect the common runner diagnostics: pass results are returned to
callers but the current production runners discard them. Determine how to
attach stable pass IDs, iteration counts, skip state, and invariant failures to
`ConversionSession.diagnostics` without changing the legacy report schemas or
public return dictionary. Add a focused contract test before wiring production
sessions.

Ordered pass diagnostics are now connected. All 19 production invocations of
the five current runners pass `session.diagnostics` into the common execution
boundary. Success events retain stable ID, phase, status, iteration count,
changed, cycle-stop, and precondition-skip fields. `PassInvariantError` remains
compatible with the previous `RuntimeError` behavior while exposing pass ID,
phase, iteration, and the complete invariant problem tuple; the common runner
records the same failure details after rollback and re-raises. Diagnostics
remain internal session state and do not alter ModelIR metadata, public return
dictionaries, or report JSON schemas.

Verification completed with:

- `16 passed, 29 deselected` for success/skip diagnostics, typed rollback
  failures, production diagnostic wiring, and existing ordered cleanup paths;
- `1033 passed, 5 deselected, 2 warnings in 130.80s` for the full sequential
  direct suite.

The next core contract gap is deterministic ModelIR fingerprinting. The generic
manager already detects cycles when supplied a fingerprint, but
`ModelIRPassState.create_ordered_manager()` does not yet provide one. Implement
a compact deterministic fingerprint that covers operator topology and mutable
tensor semantics without serializing large constant buffers repeatedly; hash
constant dtype/shape/content separately and cache immutable buffer digests.
Then add idempotence and two-state cycle tests before enabling iterations above
one for any production pass.

`ModelIRPassState` now supplies that deterministic fingerprint. It covers
topology, subgraphs, boundaries, tensor semantic metadata and provenance,
operator options/axis semantics/provenance, LayoutState, and constant content,
while excluding lineage metadata that should not affect graph fixed points.
Constant ndarrays become read-only lazily on the first fingerprint and content
digests are cached by object identity; replacing a buffer invalidates the
identity naturally. A two-state Add/Mul toggle proves deterministic cycle stop
after two iterations and emits `cycle_stopped` diagnostics.

To preserve large-model efficiency, the manager now computes fingerprints only
for specs whose `max_iterations` exceeds one. Current one-shot production
passes therefore do not serialize ModelIR or freeze constants at all. Their
explicit `changed` result remains the source of truth.

Verification completed with:

- `19 passed` for the complete core contracts, including fingerprint
  determinism, constant mutation sensitivity/cache behavior, cycle stop, and
  zero fingerprint work for one-shot passes;
- `1036 passed, 5 deselected, 2 warnings in 131.04s` for the full sequential
  direct suite.

The next pass-manager gap is phase-boundary observability. The current
production runner calls are separate groups, so `PassPhase` orders specs within
each group but does not yet produce a conversion-wide phase trace. Add a small
session-owned invocation counter/phase event contract, without reordering
calls, so diagnostics distinguish repeated invocations of the same stable pass
ID. This should remain internal and must not alter report schemas.

Pass diagnostics now include a conversion-wide `sequence` and a stable-ID-local
`invocation` number. Numbering spans separate runner groups, ignores existing
non-pass diagnostics, and therefore distinguishes all repeated recovery sweeps
without introducing a global manager or changing call order. The fields remain
internal to `ConversionSession.diagnostics`.

Verification completed with:

- `20 passed` for all core contracts, including repeated invocation numbering;
- `1037 passed, 5 deselected, 2 warnings in 130.65s` for the full sequential
  direct suite.

The next implementation unit should migrate one additional small, generic,
already-characterized cleanup into this runner contract. Prefer the adjacent
`Maximum(x, 0) → Relu` canonicalization: it has a single production call and
can share the scalar constant reader, differential index, precondition,
transaction, LayoutState, and diagnostics patterns established by the
zero-to-one clamp pass. Preserve its raw legacy helper and exact terminal order.

The float-only `Maximum(data, scalar-zero) → Relu` rewrite now lives in
`passes/graph_cleanup.py`, with the lowerer symbol retained as a compatibility
wrapper. The original input2-only, singleton-zero, output-arity, and
FLOAT16/FLOAT32 guards are unchanged. Its production call remains in the exact
terminal position but now uses `run_maximum_zero_relu_cleanup`, stable ID
`canonicalize.maximum_zero_relu`, the shared differential index and
LayoutState, transactional validation, precondition skip, and session
diagnostics. The production diagnostic-wiring architecture assertion now
covers all 20 ordered runner invocations.

Verification completed with:

- `5 passed, 784 deselected` for positive/no-op runner behavior, existing raw
  guard characterizations, implementation ownership, and production wiring;
- `1039 passed, 5 deselected, 2 warnings in 130.97s` for the full sequential
  direct suite.

The next candidate should be chosen from another single-call generic terminal
cleanup, but avoid model-named SiNet chains. Audit `fold consecutive Mul
constants`, redundant integer Cast cleanup, and terminal Q/DQ cleanup for
existing generic fixtures and differential-index readiness; select the
smallest family with complete positive and guard coverage.

The audit selected consecutive floating Mul constant folding. Its implementation
now lives in `passes/graph_cleanup.py`; the legacy lowerer symbol delegates to
it. The strict binary arity, exclusive intermediate consumer, model-output,
floating path/constant dtype, NumPy broadcast, finite-result, quantization, and
unique-name semantics are preserved. The rewrite now uses one differential
index for consumer/producer lookup, surviving-input mutation, and structural
operator removal. It registers the fused tensor in LayoutState before pruning.

All three existing call positions—including the fallback IR path—now invoke
`run_consecutive_mul_constants_cleanup` with stable ID
`canonicalize.fold_consecutive_mul_constants`. The fallback builds its own
LayoutState but contributes to the same session diagnostics; the other two use
the session-owned LayoutState. Call order was not collapsed.

Verification completed with:

- `6 passed, 28 deselected` for positive fusion, one-index behavior, fan-out,
  integer/non-finite constant guards, ownership, and all 23 production runner
  diagnostic call sites;
- `1043 passed, 5 deselected, 2 warnings in 132.21s` for the full sequential
  direct suite.

The next migration should address redundant Cast cleanup as one cohesive
family, but characterization must come first. Build compact ModelIR fixtures
for INT64→INT32 collapse, UINT64→UINT32 collapse, 32→64 alias passthrough,
fan-out, public-output, and mixed non-Cast consumer guards. Only after these
digests/structures are fixed should the two Cast implementations move from the
lowerer and share indexed mutation helpers.

Redundant integer Cast cleanup now lives in `passes/cast_cleanup.py`. Compact
fixtures characterize signed and unsigned variants of both transformations:
immediate 64→32 narrowing collapse and exclusive 32→64 alias removal. They also
fix fan-out, public intermediate/output, mixed non-Cast consumer, shape,
quantization, downstream `inDataType`, and no-op behavior.

`run_redundant_cast_cleanup` registers widening-alias cleanup first at priority
10 (`cleanup.cast_widening_alias`) and narrowing cleanup second at priority 20
(`cleanup.cast_narrowing_chain`), exactly matching the former call order. The
two specs share one ModelIRGraphIndex and LayoutState, use indexed input/output
mutation and structural removal, and validate transactionally. Both terminal
sweep pairs were replaced one-for-one by the group; the legacy lowerer symbols
remain wrappers.

Verification completed with:

- `12 passed, 17 deselected` for the Cast family fixtures, one-index behavior,
  implementation ownership, and all 25 ordered production call sites;
- `1 passed, 758 deselected` for the existing Div/Shape/Cast integration model;
- `1054 passed, 5 deselected, 2 warnings in 134.75s` for the full sequential
  direct suite.

The next safe unit is terminal Quantize/Dequantize cleanup, but it sits on an
accuracy-sensitive rounding boundary. Before migrating it, add ModelIR fixtures
for exact-grid positive collapse and no-op cases covering unequal scale/zero
point, public quantized input, shared quantized output, nonterminal float output,
and non-Dequantize producer. Retain the existing runtime rounding test as the
mandatory integration gate.

Terminal Q/DQ cleanup now lives in `passes/quantization_cleanup.py`. The
exact-grid predicate is colocated and compares quantized dtype, quantized
dimension, the full scale array, and the full zero-point array exactly. Compact
fixtures cover the positive terminal collapse and no-op cases for every grid
field, shared quantized/float tensors, nonterminal or consumed output, public
float input, and a non-Dequantize producer.

The output-name-preserving rename now accepts `ModelIRGraphIndex` and mutates
only indexed producers/consumers before updating graph boundaries, tensors,
LayoutState, and lineage. Terminal Q/DQ operators are removed structurally from
the same index. Both former production positions call
`run_terminal_quantize_dequantize_cleanup` with stable ID
`cleanup.terminal_quantize_dequantize`; raw helper symbols remain wrappers.

Verification completed with:

- `16 passed, 25 deselected` for exact-grid semantics, all terminal/boundary
  guards, indexed rename, ownership, and all 27 ordered production call sites;
- `1 passed, 758 deselected` for the existing terminal Q/DQ runtime rounding
  integration gate;
- `1068 passed, 5 deselected, 2 warnings in 134.71s` for the full sequential
  direct suite.

The next checkpoint should measure the orchestration overhead introduced by
the migrated pass groups on synthetic graph sizes before migrating another
family. Add a deterministic microbenchmark test/helper (not a timing-flaky CI
assertion) that records index refresh count, snapshot count, fingerprint count,
and visited operator count for no-candidate and one-candidate graphs. Use it to
identify remaining avoidable full scans and deep copies; retain the existing
host-level Tier timing gates for actual performance acceptance.

All nine current ModelIR runner families now provide a broad model-only
preflight. If false, `run_model_ir_pass_group` validates/registers and orders the
specs, returns their normal `skipped_by_precondition` results, and records the
same stable diagnostics without constructing ModelIRPassState, ModelIRGraphIndex,
or LayoutState. A true preflight still proceeds to the existing precise
state-level precondition and transactional validator.

Deterministic instrumentation uses a 256-operator identity graph rather than
wall-clock assertions. Across every production runner, the no-candidate path
requires zero index refreshes, zero snapshots, and zero fingerprints. A single
Maximum-zero candidate requires exactly one index build and one snapshot and no
fingerprint, while a custom preflight proves exactly one 256-operator visit.

Verification completed with:

- `45 passed, 18 deselected` for preflight instrumentation, core contracts,
  and every migrated cleanup family;
- `1071 passed, 5 deselected, 2 warnings in 134.34s` for the full sequential
  direct suite.

The next efficiency unit should reduce repeated broad scans across adjacent
runner calls. Inventory contiguous runner sequences and introduce a small
per-session operator-type summary/version only if mutations can invalidate it
differentially; do not cache stale topology. Alternatively, migrate an adjacent
pair into one semantically cohesive group when there is no intervening rewrite,
as done for Cast cleanup. Measure refresh/snapshot counts before and after.

The two repeated production sequences of constant Pad→Pool→Cast folding are now
each one `run_constant_input_fold_cleanup` call. Three specs retain the exact
dependency order with priorities 10, 20, and 30 and stable IDs
`canonicalize.constant_input_pad`, `canonicalize.constant_input_pool`, and
`canonicalize.constant_input_cast`. They share one ModelIRGraphIndex and
LayoutState, remove operators structurally, validate after each materialization,
and retain the quantized-accumulator runtime Cast guard. ScatterND and binary
constant folding remain independent compatibility helpers.

A compact full-chain fixture proves that Pad materialization enables Pool and
then Cast materialization in the same group, with one index build and correct
FLOAT16 output values. Both former lowerer triplets were replaced one-for-one;
the diagnostic-wiring assertion now covers 29 production runner calls.

Verification completed with:

- `9 passed, 776 deselected` for the full constant chain, runtime-Cast guard,
  existing Pad/Pool/Cast characterizations, preflight efficiency, ownership,
  and production wiring;
- `1074 passed, 5 deselected, 2 warnings in 137.66s` for the full sequential
  direct suite.

The next implementation unit should make preflight cost observable per
production conversion without exposing it publicly. Add lightweight integer
metrics to internal pass diagnostics—operators visited by preflight, whether
state was built, and whether a snapshot/fingerprint occurred—using counters
rather than timers. This will let Tier runs attribute orchestration work before
attempting a session-wide mutable topology cache.

Internal pass diagnostics now expose deterministic orchestration counters under
`metrics`: `preflight_operators_visited`, `state_built`, `snapshot_count`, and
`fingerprint_count`. OrderedPassManager counts actual snapshot and digest calls,
including the extra before-state digest that detects a cycle. Model-only
preflights return `ModelIRPreflightResult`; shared scanners stop at the first
matching operator or when all required op types have been seen.

The 256-operator no-candidate fixture now proves all 14 emitted spec events
report `operators_visited=256`, `state_built=false`, and zero snapshots and
fingerprints. A first-operator Maximum candidate reports one visited operator,
state construction, one snapshot, and zero fingerprints. The two-state cycle
reports five fingerprints, while invariant rollback reports one snapshot.

Verification completed with:

- `52 passed, 12 deselected` for metric accounting, preflight behavior, core
  manager contracts, and migrated cleanup families;
- `1074 passed, 5 deselected, 2 warnings in 136.55s` for the full sequential
  direct suite.

The next task should surface an aggregate of these internal counters in the
managed corpus result files used for local Tier analysis—not in public
conversion reports. Inspect the bulk runner's existing timing/RSS entry schema
and add an optional internal `pass_metrics` summary only where the conversion
session diagnostics are available without changing CLI/API artifacts.

Managed corpus metrics now cross the subprocess boundary through an internal,
opt-in path only. `lower_onnx_to_ir` has a private diagnostic sink finalized on
both normal and fallback returns. The flatbuffer builder activates it only when
`ONNX2TF_INTERNAL_PASS_METRICS_PATH` is present, writes an atomic aggregated
JSON, and otherwise allocates nothing and emits no file.

The sequential bulk runner sets that environment variable to each run's
`pass_metrics.json` immediately around `subprocess.run`, restores any previous
value in `finally`, and removes stale files before execution. Valid metrics are
stored in the managed entry and aggregated into `summary.pass_metrics` across
models. This does not add a public CLI flag, ModelIR metadata, accuracy-report
field, or conversion-result key.

Verification completed with:

- `50 passed` for core diagnostics, private lowerer sink, environment
  restoration, bulk entry/summary aggregation, and all existing bulk behavior;
- `1076 passed, 5 deselected, 2 warnings in 136.25s` for the full sequential
  direct suite.

The next practical step is to run a small real Tier 0 sample with the managed
bulk runner and inspect `pass_metrics` distributions, one model at a time. Use
the existing fixed order/profile and do not launch parallel workers. Based on
actual high-count pass IDs, select the next migration or grouping target rather
than guessing from source order.

A real sequential Tier 0 sample was run with only `Acos_11.onnx` in an isolated
root (`--root_only --tflite_only`, timeout 120 seconds). It passed in 2.26
seconds. The first metrics aggregation exposed that multi-spec group preflight
work was duplicated once per event. Metrics schema version 2 fixes this by
assigning `group_sequence`, deduplicating preflight/state construction per
runner group, and retaining snapshot/fingerprint accounting per pass event.

The corrected rerun passed and recorded:

- 37 pass events across 28 runner groups;
- 37 skips, zero changed events;
- 112 actual preflight operator visits;
- zero ModelIRPassState builds, snapshots, or fingerprints.

The latest schema-v2 core/bulk suite completed with `50 passed`; the latest full
sequential direct suite completed with
`1076 passed, 5 deselected, 2 warnings in 135.57s`. Both sample roots and output
trees were deleted after inspection, so no generated models or artifacts remain.

The highest repeated IDs in this sample were Squeeze/Reshape identity (eight
invocations) and mixed-attention layout (six), all skipped. The next cohesive
migration should therefore pair the legacy Squeeze/Unary/Reshape passthrough
that immediately precedes Squeeze/Reshape identity cleanup at its repeated
sites. Move it into the graph-cleanup family, make it differential-index and
LayoutState aware, then run both specs in their existing order through one
group. Do not merge calls separated by other rewrites.

The Squeeze/Unary/Reshape passthrough now lives in
`passes/graph_cleanup.py`. Its strict unary allowlist, axis-0 normalization,
shape compatibility, single-consumer, model-output, and fan-out guards are
preserved. Single-path rewrites retain only the unary operator. Fan-out rewrites
use indexed remove/reinsert to produce `unary(4D) → Squeeze(3D)` before the
remaining rank-3 consumers.

All six locations where the legacy pass immediately preceded Squeeze/Reshape
identity cleanup now invoke one runner with
`include_unary_passthrough=True`. Stable ID
`cleanup.squeeze_unary_reshape_passthrough` runs at priority 10, followed by
`cleanup.squeeze_reshape_identity` at priority 20, sharing one index,
LayoutState, preflight, and transaction boundary. The other two identity calls
remain identity-only because no unary pass preceded them.

Verification completed with:

- `8 passed, 32 deselected` for single-path folding, fan-out reorder, axis
  rejection, one-index behavior, implementation ownership, and production
  wiring;
- `1079 passed, 5 deselected, 2 warnings in 136.77s` for the full sequential
  direct suite.

The next data-driven target is mixed-attention layout, which had six skipped
invocations in the Tier 0 sample. Audit the legacy passes immediately adjacent
to those six runner calls and group only a generic attention-layout operation
that is present at every same-order site. If adjacency differs, retain the six
calls and select another high-count family instead.

That adjacency audit found two distinct neighborhoods. The first four
mixed-attention calls are preceded by Conv-attention, SA/PA MirrorPad, and a
SiNet-named tail rewrite; the fifth appears after a different QKV/split family;
the sixth is a late standalone recovery. No pass is adjacent in the same order
at all six sites, so the six calls were deliberately left separate and no
model-named rewrite was folded into the generic runner.

The independent generic Conv-attention rewrite was instead migrated as the
next bounded unit. All five existing production positions now call
`run_conv_attention_layout_cleanup` with the session LayoutState and diagnostic
sink. Stable pass ID `layout.conv_attention_nhwc` runs in `LAYOUT_PLAN` as one
transaction. Its broad preflight requires Transpose, Mean, and Conv/Depthwise
Conv capability before allocating indexed state. Candidate execution reuses
the state-owned producer/consumer index and refreshes it once per completed
structural iteration instead of rebuilding two edge maps at every scan. The
raw lowerer entry point remains a compatibility wrapper, and its four strict
motif fixtures continue to cover Logistic, HardSigmoid, HardSwish, and the
two-stage HardSwish/Mean chain.

Verification for this checkpoint completed sequentially in the core `uv`
environment:

- `4 passed, 779 deselected` for all Conv-attention motif fixtures;
- `24 passed` for architecture and deterministic pass-efficiency checks;
- `1079 passed, 5 deselected, 2 warnings in 136.45s` for the full direct suite;
- real Tier 2 `sinet_320_op.onnx` conversion with `-cotof` passed with no
  skipped output and maximum absolute error `2.572051016613841e-09`.

The real-model metrics recorded five safe preflight skips for the new pass and
zero snapshots/fingerprints; other recovery passes had already removed its
candidate motif. The temporary conversion directory and metrics file were
deleted after inspection. No package was added, TensorFlow was not imported,
and no inference process ran concurrently.

The next refactoring unit should inventory the repeated generic QKV attention
sequence (`gather/reshape hoist`, `slice replacement`, `slice-to-split`, split
collapse, shared pretranspose, and weighted-sum bridge). Group only exactly
contiguous same-order sequences, retain isolated calls, and first make each
selected rewrite differential-index aware. Do not include the neighboring
SiNet-, CSP-, or model-specific layout rules merely to reduce call count.

## Previous pause checkpoint — `fb-refactor2` after `19cb989`

### Completed work

- Completed the quantized op-family split. DynamicQuantizeLinear,
  QLinearMatMul/QGemm, QLinearAveragePool/GlobalAveragePool,
  QLinearAdd/QLinearMul, QLinearSigmoid/LeakyRelu/Softmax, QLinearConcat,
  QuantizeLinear/DequantizeLinear, QLinearConv, and ConvInteger now have
  dedicated family modules.
- Replaced the old combined `op_builders/quantized.py` with
  `op_builders/quantized_common.py`. The common module contains shared
  quantization, shape/signature, padding, and requantization primitives and no
  `build_*` entry point.
- Fixed pre-extraction normalized ModelIR fingerprints for each extracted
  family and consolidated duplicate fingerprint serialization in
  `tests/flatbuffer_direct_fingerprint.py`.
- Preserved the public builder imports, registry dispatch, TensorFlow-free
  boundary, and existing runtime behavior. The latest implementation commit is
  `19cb989` (`complete quantized op family split`) and is pushed to
  `origin/fb-refactor2`.
- Measured `lower_from_onnx2tf.py` with the Python AST. It contains 280
  top-level definitions, including 204 `_optimize_*` functions. The largest
  functions include `_optimize_transpose_pre_concat_nhwc_chains` (2,117
  lines), `lower_onnx_to_ir` (1,711 lines), and
  `_optimize_transpose_pre_add_nhwc_chains` (1,580 lines).
- Selected coverage/correspondence reporting as the next independent
  extraction boundary. The contiguous implementation is currently
  `_collect_schema_ops_for_range` through
  `write_tensor_correspondence_report`; `_build_tensor_consumer_map` at the top
  of the legacy module must move with it and be re-imported for layout passes.

### Incomplete work

- The reporting extraction has not been applied. An attempted generated patch
  failed context verification before changing any file; no partial
  `reporting.py` exists and all reporting functions remain in
  `lower_from_onnx2tf.py`.
- `lower_from_onnx2tf.py` remains approximately 78,000 lines and still owns the
  large layout-rule collection and the main lowering orchestration.
- The broader Goal remains incomplete: ordered pass ownership, transactional
  rewrite coverage, further layout-rule generalization, exporter cleanup, and
  final Tier 0–4/Tier 5 phase gates still require work.
- The managed Tier 0–4 baseline remains 368 passes, 6
  `missing_tflite_report`, 20 `tflite_fail`, and 26 excluded historical
  timeouts. The 26 active non-passes have explicit normalized causes; the two
  DEIM entries are accepted successes by user direction.

### Branch and working tree

- Branch: `fb-refactor2`, synchronized with `origin/fb-refactor2` at
  `19cb989` before this handoff-only checkpoint.
- There are no unfinished code changes or generated temporary files from the
  reporting attempt. This handoff document is the only intended checkpoint
  change before commit.

### Tests run

- Latest full sequential direct regression after the complete quantized split:
  `985 passed, 5 deselected, 2 warnings in 121.63s`.
- Quantized family fingerprint/architecture set: `26 passed`.
- Focused QLinearConv/ConvInteger extraction set: `12 passed, 760 deselected`.
- Reporting characterization command, run without the standard optional-test
  exclusions: `777 passed, 5 failed, 2 warnings in 117.09s`. All five failures
  are the already-known optional environment cases listed below; no reporting
  test failed.

### Failing tests and known issues

- Four TensorFlow converter tests fail because the core `uv` environment does
  not install the optional `tensorflow`/`tf_keras` extra:
  `test_tflite_backend_matrix_add`,
  `test_tflite_backend_matrix_hardswish_rewrite_on_off`,
  `test_tf_converter_resize_cubic_avoids_flex_resize_bicubic`, and
  `test_tf_converter_resize_cubic_honors_cubic_coeff_a`.
- `test_flatbuffer_direct_group_norm_alias_builtin_conversion` fails because a
  system Python 3.10 Torch binary is incompatible with the active Python 3.12
  `uv` environment. These five tests are the standard broad-suite exclusions.
- Two expected FLOAT16 cast overflow warnings remain in the ArgMax/ReduceMax
  and negative-infinity Where tests.
- The first reporting extraction patch failed only because its deletion hunk
  did not preserve the blank-line context before `_read_transpose_perm`; it
  made no filesystem change and is not a product defect.

### First action on resume

1. Reconfirm a clean `fb-refactor2` worktree and the current reporting function
   boundaries with `rg`/AST.
2. Create `onnx2tf/tflite_builder/reporting.py` with
   `_build_tensor_consumer_map`, coverage schema/policy/report writers, rewrite
   tracing, downstream correspondence inference, and correspondence writers.
3. Import/re-export `_build_tensor_consumer_map`, `build_op_coverage_report`,
   `write_op_coverage_report`, `build_tensor_correspondence_report`, and
   `write_tensor_correspondence_report` from `lower_from_onnx2tf.py` so existing
   Python imports remain compatible.
4. Run `tests/test_tflite_builder_op_coverage.py` plus the correspondence cases
   selected from `tests/test_tflite_builder_direct.py`, using the standard five
   optional-test exclusions, then run the full direct suite sequentially.
5. Only after identical reports and a green full suite, update this document,
   commit the reporting extraction, and batch the next push. Do not create a
   pull request.

## Current checkpoint — `fb-refactor2`

The opset-aware Resize lowering and numerically stable Inverse lowering recover
`onnx_dense_optimized.onnx` and its byte-identical `_org` counterpart without a
model-name rule or additional dependency.

- ONNX Resize in opset 10 now defaults to asymmetric coordinates, as required
  by that schema, instead of inheriting the opset 11+ `half_pixel` default.
- The generic 4×4 through 16×16 Inverse lowering now uses per-batch partial
  pivoting. At each Gauss–Jordan iteration it selects the largest absolute
  remaining pivot, swaps the selected row in both the matrix and identity
  state, and only substitutes a signed epsilon when the chosen pivot is
  genuinely near zero. Normal pivots are no longer shifted unconditionally.
- A synthetic opset-10 nearest-neighbor Resize test fixes the coordinate
  contract, and a batched 8×8 Inverse test requires an actual row swap and
  checks ONNX Runtime against the generated TFLite artifact.

Both dense models were evaluated sequentially with all seven outputs compared
and no skip. Their identical fixed-seed result is `evaluation_pass=true`,
`max_abs=0.00015753507614135742`, `mean_abs=2.2844531542128204e-06`,
`rmse=5.3903736593740145e-06`, and cosine similarity
`0.9999999998639824`. The recorded pre-fix baseline maximum was
`0.8238084316253662`; correcting Resize alone reduced it to
`0.16955818608403206`, and removing unconditional pivot perturbation reduced
it to `0.157419` before partial pivoting eliminated the amplified GridSample
error.

Legacy linear Upsample now follows the half-pixel coordinate semantics produced
by ONNX's v9-to-v11 version converter, while legacy nearest Upsample retains its
asymmetric behavior. This general rule recovers `modnet_old.onnx`, whose eight
linear downsample branches previously diverged before the first concatenation.
Its fixed-seed single output has no skip and reports `evaluation_pass=true`,
`max_abs=2.3931264877319336e-05`, `mean_abs=2.478012597297248e-07`,
`rmse=9.7739101243166e-07`, and cosine similarity `0.9999999999988499`.

`LINEA.onnx` also passes on the current static-input runtime path without an
additional lowering rule. Both outputs were compared with no skip:
`evaluation_pass=true`, `max_abs=0.002297189086675644`,
`mean_abs=3.4909796139056033e-06`, `rmse=7.162934073986925e-05`, and cosine
similarity `0.9994290428305682`.

The managed Tier 0–4 profile now records 368 passes, 6
`missing_tflite_report`, 20 `tflite_fail`, and 26 excluded historical timeouts.
There are 26 active non-passes, and every one now has an explicit normalized
cause or the expected invalid/custom-op runtime classification.

`rf-detr-nano.onnx` is promoted from its historical conversion failure to a
normal accuracy pass without a model-specific workaround. Its sequential
fixed-seed run compares both `pred_boxes` and `pred_logits` with no skip:
`evaluation_pass=true`, `max_abs=0.000102996826171875`,
`mean_abs=5.465599675581121e-06`, `rmse=1.015673578580791e-05`, and cosine
similarity `0.9999999999986942`. The source graph has 770 nodes and the lowered
graph has 722 nodes, keeping this recovery inside the Tier 3 active gate.

`LibreRFDETRn.onnx` is likewise promoted from its historical accuracy failure
to a normal pass without a model-specific workaround. Its sequential
fixed-seed run compares both `dets` and `labels` with no skip:
`evaluation_pass=true`, `max_abs=0.0001087188720703125`,
`mean_abs=5.6163524849373e-06`, `rmse=1.0355151438507286e-05`, and cosine
similarity `0.9999999999986449`. The model uses the existing explicit input
shape `input:1,3,384,384`; its source graph has 770 nodes and the lowered graph
has 722 nodes.

`bertsquad-12-int8.onnx` remains an active failure with normalized reason
`onnxruntime_u8s8_matmulinteger_cpu_saturation`. The direct implementation's
first U8×S8 MatMulInteger result matches an explicit INT32 NumPy product and
ONNX `ReferenceEvaluator` exactly. ONNX Runtime's CPUExecutionProvider differs
from that same product by `max_abs=11772`, mean absolute error
`326.5231577555339`, over 24,453 elements, identically at every graph
optimization level from disabled through all. This divergence starts at the
first encoder MatMulInteger even though the then-current preceding
DynamicQuantizeLinear differed at only two of 196,608 UINT8 elements by one.
That pre-correction final report had `max_abs=1.8257164359092712`. Emulating a
host-specific saturating CPU kernel would violate portable ONNX integer-matmul
semantics, so the exact lowering is retained and the previous failure-signature
hash remains fixed.

DynamicQuantizeLinear now uses nearest-even `ROUND` for both zero-point and
data quantization, and rounds `x / scale` before adding the integer zero point.
The previous `+0.5` then CAST path was half-up, and adding a large zero point
before rounding could erase a just-below-half fraction in FLOAT32. Synthetic
tests cover exact half values with both zero and odd nonzero zero points. Of
the active Tier 0–4 corpus, only `afhq_generator.v11.quant.onnx` and
`bertsquad-12-int8.onnx` contain this op; the only other occurrence is the
excluded Tier 4 timeout `vision_encoder_uint8.onnx`.

For `afhq_generator.v11.quant.onnx`, input DynamicQuantizeLinear now agrees at
every element and the first residual mismatch moves after an
InstanceNormalization difference of `4.76837158203125e-07`. Later quantization
boundaries still amplify sparse one-quantum differences through the decoder.
The final no-skip result improves from baseline `max_abs=0.22717905044555664`
to `0.21375656127929688`, with RMSE `0.03692099507463561` and cosine similarity
`0.999052579433886`; it remains a normal threshold failure with reason
`instance_normalization_drift_amplified_by_dynamic_quantization_decoder`.
The corrected BERT path remains dominated by ONNX Runtime's saturating CPU
MatMulInteger behavior; its no-skip fixed-seed maximum is now
`2.001576066017151`, with RMSE `1.2972128029177183` and cosine similarity
`0.9616353624777596`.

The now-fixed DynamicQuantizeLinear implementation was mechanically moved to
`op_builders/dynamic_quantize.py`, a dedicated 391-line op-family module. The
legacy combined `op_builders/quantized.py` shrinks from 3,235 to 2,850 lines.
The public builder import and registry dispatch remain unchanged. A normalized
ModelIR fingerprint covering all operators, tensor metadata, constants,
options, and quantization fields is identical at `d97cba6` and after the move:
`a83d642e4aa7903f9b34495fec2c1edb5ff8779ba6735bedde382578152657f5`
(22 operators, 27 tensors). The architecture regression verifies that the
implementation remains in its family module and is absent from the legacy
file. It does not impose a source-line limit.

QLinearMatMul and QGemm were then moved mechanically into the dedicated
238-line `op_builders/qlinear_fc.py` family module, reducing the remaining
legacy quantized builder from 2,850 to 2,634 lines. Pre-extraction ModelIR
fingerprints are now executable regression tests:
`633d083445fcf765023a948c038c0956c7a0b7646b73bdac0bb65cf4c14173c8`
for QLinearMatMul and
`bf71085f2cc3a5981b209b6d5b02cc65ea55a41251465229a5ef1636a319f70f`
for QGemm, each with 9 operators and 16 tensors. The architecture test keeps
both builders out of the legacy module and includes the new module in the
TensorFlow-import boundary. Sequential
one-sample CRNN verification through the new registry path is unchanged at
`max_abs=0.14842605590820312`, RMSE `0.0011565753987944503`, and cosine
similarity `0.999999996846642`.

QLinearAveragePool and QLinearGlobalAveragePool were subsequently moved
mechanically to `op_builders/qlinear_pool.py`. Public imports and registry
dispatch remain unchanged, and the legacy combined quantized module no longer
defines either builder. Focused ModelIR fingerprints are fixed at
`0bb8b9064ae208810addbcebb27846b05873d817e947a5af212f3fd8ee4a6b7c` and
`1b066e8245cb45f79df76dbc052ecf7485f07d7910fb789cff38b47c298b7f19`.
The full sequential direct regression completed with `970 passed, 5
deselected, 2 warnings in 121.97s`. The architecture checks enforce op-family
ownership and the TensorFlow-free import boundary only; they intentionally do
not enforce a source-line count.

The Goal's `2,000` threshold applies exclusively to ONNX graph operation/node
count: Tier 4 ends at 1,999 nodes and Tier 5 begins at 2,000 nodes. It is not a
limit on production or test source-file length.

QLinearAdd and QLinearMul were then moved mechanically to
`op_builders/qlinear_binary.py`. The dispatch and ModelIR contracts remain
unchanged. Their pre-extraction fingerprints are
`d2f0714a44b2dc376827b845269a217c1df894986f3957128994a2913d611c24`
(9 operators, 15 tensors) and
`b4d9d1a39202474faf52ab43fbde4938fe892a0a38c5739a87b6da2d9b882b34`
(4 operators, 6 tensors), respectively. The fingerprint implementation is now
shared by the FC, pooling, and binary family tests, removing duplicated
normalization and ModelIR serialization code. Existing QLinearAdd rounding,
QLinearConv chain, and QLinear FC chain runtime checks pass through the new
import path. The full sequential direct regression completed with `973 passed,
5 deselected, 2 warnings in 122.70s`.

QLinearSigmoid, QLinearLeakyRelu, and QLinearSoftmax were subsequently moved
mechanically to `op_builders/qlinear_activation.py`. Their pre-extraction
ModelIR fingerprints are
`67e5b3d23cf2cfe03ae8ef1a006ac5fecf221f328553d3c1904ceebad9a7d902`
(1 operator, 2 tensors),
`f1d0b1b74e6f0f056ca595912efcceb2827da416b059dc12992fd06ed137ab09`
(1 operator, 3 tensors), and
`56aef3cabbed33cabcaba95d36058a37b6a12428102f7e83b0aef334eadbb4ec`
(12 operators, 17 tensors). Focused Sigmoid/Softmax runtime and LeakyRelu rank
checks pass through the new import path. The full sequential direct regression
completed with `977 passed, 5 deselected, 2 warnings in 123.56s`.

QLinearConcat was then moved mechanically to
`op_builders/qlinear_concat.py`. Its pre-extraction fingerprint is
`924e1470c62f93ba44dde277144d84bf796f40c5123839b59b44e4cd89c5b927`
(6 operators, 7 tensors). The focused lowering and both concat-to-conv layout
propagation checks pass through the new import path.

QuantizeLinear and DequantizeLinear were moved mechanically to
`op_builders/quantize_linear.py`. Their shared two-node Q/DQ fixture retains
the pre-extraction fingerprint
`333343018c7bb32db3138cefdf4007353140b044472017ae6c3b4cce762e8f91`
(2 operators, 3 tensors). Focused Q/DQ rounding, per-axis quantization, layout,
and QLinearConcat tests completed with `12 passed, 761 deselected`. The full
sequential direct regression completed with `981 passed, 5 deselected, 2
warnings in 122.78s`.

QLinearConv was moved mechanically to `op_builders/qlinear_conv.py`. The
mixed UINT8 activation / INT8 filter fixture retains its pre-extraction
fingerprint
`c752a5b1e31744e65d483733f55a688f2189d6bf11436cabd498cfc6a2ef5019`
(17 operators, 29 tensors). Focused mixed-dtype runtime, filter layout,
explicit/symbolic padding, dynamic-batch, and unknown-rank checks pass through
the new import path.

ConvInteger was moved mechanically to `op_builders/conv_integer.py`. Its
pre-extraction fingerprint remains
`587f53091ce42815e43946d7b73324fe31ec7d5aeb1c3d2d749097351106dfb5`
(7 operators, 13 tensors), and its focused builtin lowering check remains
unchanged. With all builders extracted, `op_builders/quantized.py` was renamed
to `op_builders/quantized_common.py`; it now exposes only shared quantization,
shape, padding, and requantization primitives. All family imports and the
TensorFlow-free architecture boundary use the new common-module name. The
complete quantized family fingerprint/architecture set completed with `26
passed`, and the full sequential direct regression completed with `985 passed,
5 deselected, 2 warnings in 121.63s`.

`campp_vin.onnx` is promoted from an historical accuracy failure to a normal
pass. Its concretized dynamic-time artifact fails during XNNPACK reshape
preparation, so isolated evaluation now retries once, sequentially, with the
builtin interpreter after a default-delegate worker failure. The builtin run
compares the single `output` tensor with no skip and reports
`evaluation_pass=true`, `max_abs=3.3020973205566406e-05`,
`mean_abs=8.416682248935103e-06`, `rmse=1.0447499868661927e-05`, and cosine
similarity `0.9999999999694269`. Successful default-delegate evaluation is
unchanged, and builtin failures are not retried.

`best.onnx` and `best_org.onnx` remain failures rather than receiving a relaxed
tolerance. Both simplify to the same 516-node Q/DQ graph and produce identical
fixed-seed metrics with no output skip: `max_abs=58.7506103515625`,
`mean_abs=0.12212618568213444`, `rmse=0.9568672461041465`, and cosine similarity
`0.9998485974152098`. Their first material mismatch is a sparse QuantizeLinear
rounding outlier (`max_abs=0.13601922988891602` over a
`[1,16,128,160]` tensor), which is repeatedly amplified through the Q/DQ–Conv
backbone and detector decode. The managed reason is now
`qdq_rounding_outliers_amplified_by_detector_decode`; each model retains its
previous normalized failure-signature hash.

The two DEIM variants are treated as accepted successes by explicit user
direction despite the normal metric-threshold judgement remaining false.
Before the first decoder TopK, the small fp16 variant differs by normal fp16
backbone increments while the larger variant's score head is within
`max_abs=0.000110626220703125`. Near-tied score ordering then changes TopK
indices by as much as `1909` and `4205`, respectively. Query gathering and the
final postprocessor TopK amplify this discontinuity to final label maxima of
`27.0` and `20.0`. Both baseline entries record
`user_approved_topk_index_instability_from_near_tied_scores`; no model-name
lowering rule, global tolerance relaxation, or forced index ordering was added.
No cause-unclassified active Tier 0–4 model remains at this checkpoint.

`text_recognition_CRNN_CN_2021nov_int8.onnx` retains its failure but now has
the more precise reason
`lstm_float_drift_crosses_quantization_boundary_before_qlinear_matmul`. The
second fused LSTM is within `max_abs=4.181463737040758e-06`; exactly one value
at `[23,266]` crosses the next QuantizeLinear boundary by one quantum. Six
QLinearMatMul outputs consequently differ by one quantum. For both the ONNX
and direct tensors, an explicit INT32 NumPy matmul plus declared requantization
matches every QLinearMatMul element when fed that runtime's own quantized
input. The final `max_abs=0.14842605590820312` therefore does not justify a
matmul rewrite or a semantics-changing rounding bias.

Those GridSample siblings remain normal threshold failures. Their upstream
feature tensors agree to roughly `1e-4`, and the generated grid agrees except
for sparse coordinates (`max_abs=0.010283231735229492`). With
`align_corners=1` and zero padding, those coordinates cross the discontinuous
inside/outside boundary and the three GridSample outputs amplify the difference
before decoding. The final no-skip metrics are `max_abs=0.296916950494051`,
`rmse=0.007787096662049418`, cosine `0.9998945091127481` for `model_70`, and
`max_abs=0.2830471396446228`, `rmse=0.0074818409992842005`, cosine
`0.9999115783578202` for `model_grid_sample`. Both now use the normalized reason
`grid_coordinate_rounding_amplified_at_zero_padding_boundary` while retaining
their previous failure-signature hashes.

Validation completed in the core `uv` environment, with one pytest process and
no parallel workers:

- `967 passed, 5 deselected, 2 warnings` after the QLinear FC family
  extraction, its two pre-extraction fingerprint tests, and architecture test;
- `964 passed, 5 deselected, 2 warnings` after the DynamicQuantizeLinear
  op-family extraction and architecture boundary test;
- `963 passed, 5 deselected, 2 warnings` after the DynamicQuantizeLinear
  nearest-even and round-before-zero-point correction;
- `5 passed, 782 deselected` for the focused DynamicQuantizeLinear runtime and
  managed-profile checks;
- `961 passed, 5 deselected, 2 warnings` across the direct builder, op
  coverage, all `flatbuffer_direct` regression modules, and all accuracy
  evaluator modules after the delegate fallback change;
- `73 passed` for the focused accuracy evaluator and managed baseline set;
- `rf-detr-nano.onnx`, `LibreRFDETRn.onnx`, `bertsquad-12-int8.onnx`, and
  `campp_vin.onnx` were each run sequentially end to end with `-cotof`;
- `793 passed, 7 deselected, 2 warnings` across the direct builder, managed
  profile, architecture/import boundary, and the two new regression files;
- `28 passed, 772 deselected` for the focused Inverse, Resize, managed profile,
  and TensorFlow-free checks;
- `23 passed, 750 deselected` for the subsequent legacy Upsample, Resize,
  managed profile, and TensorFlow-free regression set;
- `1 passed, 756 deselected` for Compress after removing a pre-existing unused
  local from the touched Resize module;
- both dense corpus models passed sequential end-to-end `-cotof` runs.

The five current broad-suite deselections are four optional TensorFlow backend
tests and the optional Torch GroupNorm integration test. The core environment
does not install TensorFlow and exposes an incompatible system Python 3.10
Torch binary to Python 3.12. No optional dependency was installed for this
checkpoint.

## Previous checkpoint — static delegate family after `53bba37`

The static-input delegate capability introduced by `53bba37` also recovers a
four-model AnimeGAN/face-paint family that previously shared the same unresolved
accuracy signature:

- `anime-gan-v2.onnx`;
- `anime-gan-v2_org.onnx`;
- `face_paint_512_v2_0.onnx`;
- `model_paint_v2_test.onnx`.

All four fixed-seed sequential runs compared their only output with no skip and
produced identical metrics: `evaluation_pass=true`,
`max_abs=0.0017458945512771606`, `mean_abs=0.0003080219994663925`,
`rmse=0.0003674835052452457`, and cosine similarity
`0.9999997946255107`. Their previous delegate-free baseline maximum was
`0.037707426119595766`. No model-specific lowering or tolerance was added.

The managed Tier 0–4 profile now records 359 passes, 6
`missing_tflite_report`, 29 `tflite_fail`, and 26 excluded historical timeouts.
There are 35 active non-passes. The next unresolved generic accuracy group in
managed order is `onnx_dense_optimized.onnx` and its `_org` counterpart; the
earlier remaining failures already carry explicit normalized reasons.

## Previous checkpoint — `fb-refactor2` at `53bba37`

The current checkpoint recovers `vit_b_encoder.onnx` and removes a general
large-model evaluation bottleneck without changing conversion semantics.

- The direct branch releases its legacy GraphSurgeon graph before ModelIR
  lowering. That graph duplicated hundreds of megabytes of initializers and is
  never consumed by the direct pipeline.
- After export, unreachable ModelIR and serialization clones are collected.
  On glibc systems, unused allocator arenas are returned to the OS; the trim is
  optional and failure-tolerant on other libc/platform combinations.
- Isolated evaluation no longer pickles a complete ONNX protobuf through the
  multiprocessing pipe. It writes one managed evaluation model, passes only
  its path, and lets ONNX Runtime open it directly. The ONNX worker is still
  fully reaped before the TFLite worker starts.
- The evaluation graph and temporary worker model are released at their phase
  boundaries. Managed temporary files are removed after comparison.
- LiteRT's default delegate is enabled only when every requested input shape is
  statically positive. Dynamic-shape models retain the existing
  delegate-disabled safety path. This is a capability rule, not a model-name
  or model-size workaround.

Before the fix, each backend was healthy in isolation (ONNX Runtime inference
`2.68s`, LiteRT with XNNPACK `10.94s`), but conversion-plus-evaluation exceeded
300 seconds because the parent retained several graph/protobuf copies while a
delegate-free worker ran the ViT. The final sequential end-to-end run completed
in `40.55s`, with peak RSS `4,632,968 KiB`. Its only output was compared with no
skip:

- `evaluation_pass=true`;
- `max_abs=2.6226043701171875e-06`;
- `mean_abs=1.260113801429541e-07`;
- `rmse=1.9330447647107274e-07`;
- cosine similarity `0.9999999999992181`.

The expanded affected suite completed with `884 passed, 5 deselected,
2 warnings in 113.33s`. Focused evaluator, subprocess, import-boundary, managed
profile, and memory tests also passed. Every worker and model ran sequentially;
no process pool or parallel pytest worker was used.

The managed Tier 0–4 profile now records 355 passes, 6
`missing_tflite_report`, 33 `tflite_fail`, and 26 excluded historical timeouts.
There are 39 active non-passes. Every remaining missing-report entry has a
documented unsupported semantic or invalid-source reason. The next unresolved
accuracy group in managed order starts with `anime-gan-v2.onnx`.

During this checkpoint, obsolete Goal-generated temporary conversions,
diagnostic models, and historical bulk-run directories were removed from
`/tmp`. Available filesystem space increased from approximately `63 GiB` to
`157 GiB`; repository models and tracked files were not removed.

## Earlier checkpoint — `fb-refactor2` at `d278bcf`

The next checkpoint recovers `tiny_decoder_11.onnx` by making dynamic
ScatterND negative-index normalization safe for index tensors above rank 4.
The implementation remains TensorFlow-free and introduces no dependency.

- LiteRT's elementwise comparison broadcast path aborts in native code when
  the result rank exceeds four. The decoder exposed this with eight `LESS`
  operations over dynamic `[1,1,1,1,4]` ScatterND indices.
- The generic ScatterND helper now temporarily coalesces the leading index
  dimensions to `[-1,K]`, normalizes each negative coordinate against the
  indexed data-shape prefix, and reshapes the normalized coordinates back to
  their original runtime shape before both ScatterND operations.
- Only the comparison/normalization representation is flattened. The public
  output, updates, index-vector dimension, dynamic leading dimensions, and
  ScatterND semantics are unchanged.
- The implementation is isolated in `op_builders/scatter_utils.py`; the large
  central `index.py` loses duplicated normalization construction rather than
  gaining another rule.
- A dedicated synthetic regression varies the dynamic leading dimension,
  exercises negative coordinates in a rank-5 index tensor, verifies the safe
  rank-2 `LESS` contract in ModelIR, and requires exact ONNX Runtime/TFLite
  output equality.

Sequential `tiny_decoder_11.onnx` verification used all four managed shape
hints and `keep_shape_absolutely_input_names` values. All three outputs were
compared with no skip:

- `evaluation_pass=true`;
- `max_abs=5.048513412475586e-05`;
- `mean_abs=1.637148860798228e-05`;
- `rmse=2.167510270660002e-05`;
- cosine similarity `0.9999999999902514`.

The affected sequential suite completed with `864 passed, 5 deselected,
2 warnings in 89.20s`. The optional TensorFlow/import-boundary suite separately
completed with `13 passed in 5.79s`. The five deselections and two float16
overflow warnings are the existing environment-specific cases documented
below. No parallel pytest worker or concurrent inference process was used.

The managed Tier 0–4 profile now records 354 passes, 7
`missing_tflite_report`, 33 `tflite_fail`, and 26 excluded historical timeouts.
There are 40 active non-passes remaining. Six missing-report entries already
have explicit unsupported or invalid-source reasons. The next unresolved
generic missing-report model in managed order is `vit_b_encoder.onnx` (Tier 3).

## Earlier checkpoint — `fb-refactor2` at `5b0a098`

Commit `5b0a098` recovers `encoder.onnx` by replacing dynamic rank-4
`GridSample` custom fallback with a TensorFlow-free builtin lowering.

- Runtime image N/C/H/W are read with `SHAPE`; no static spatial dimensions
  are fabricated.
- The image is transposed and flattened once. Runtime batch/spatial offsets and
  global indices gather only the required samples.
- Bilinear/linear and nearest interpolation are supported for zeros and border
  padding with both align-corners modes. Zeros padding uses per-neighbor masks,
  avoiding a dynamic padded-image allocation.
- NaN coordinates retain the existing ONNX Runtime-compatible `-1`
  normalization.
- The implementation is isolated in
  `op_builders/grid_sample_utils.py`; no new package was introduced.

Sequential `encoder.onnx` verification compared its only output with no skip:

- `evaluation_pass=true`;
- `max_abs=1.9293278455734253e-05`;
- `rmse=2.1648828175950605e-07`;
- cosine similarity `0.9999999999964916`;
- all 24 GridSample nodes use builtin operators; no unresolved custom op
  remains.

The expanded affected suite completed with `879 passed, 5 deselected,
2 warnings in 90.32s`. The three new dynamic numerical cases cover
bilinear/zeros/align-corners, bilinear/border/half-pixel, and
nearest/zeros/half-pixel. Static GridSample, validation, managed profile, and
the prior Mask R-CNN tests are included in the same suite. The five deselected
optional-environment tests and two expected float16 warnings are unchanged.

The managed Tier 0–4 profile now records 353 passes, 8
`missing_tflite_report`, 33 `tflite_fail`, and 26 excluded historical timeouts.
There are 41 active non-passes remaining. The next actionable missing-report
model in managed order is `tiny_decoder_11.onnx`; its four recorded shape hints
and `keep_shape_absolutely_input_names` options must be retained during
reproduction.

## Previous checkpoint — `fb-refactor2` at `c3c5ff7`

This section supersedes the older `fb-refactor` checkpoint retained below for
historical context. The active implementation branch is `fb-refactor2`, and
commit `c3c5ff7` recovers the managed
`maskrcnn_resnet50_fpn.onnx` conversion without adding dependencies or using
TensorFlow in the direct path.

### Completed since the older checkpoint

- Repaired the malformed TorchVision paste-masks Loop capture using a guarded
  semantic pattern. The repair reconstructs `expand_boxes` from the padded and
  source mask dimensions and is shared by direct lowering and ONNX Runtime
  evaluation. Ambiguous patterns remain untouched.
- Added zero-batch-safe floating-point Pad lowering. It temporarily supplies a
  safe batch to LiteRT Pad and restores the true runtime output shape, including
  batch zero, without changing non-empty results.
- Replaced RoiAlign's per-ROI full feature-map duplication with a single NHWC
  flatten and four masked neighbor gathers. This removes the Mask R-CNN path
  that attempted to allocate roughly 25 GB for a 612-ROI feature level.
- Preserved dynamic axes through output retargeting, singleton layout
  transpose-to-reshape conversion, late reshape recovery, ConvTranspose
  intermediates, and consecutive dynamic-batch layout reshapes.
- Merged explicit control-flow-body metadata into Loop lowering and recovered
  missing Gather ranks from ONNX semantics. In particular, rank-1 data gathered
  by a scalar index remains a logical scalar before Unsqueeze.
- Split the new control, Pad, and RoiAlign helpers into focused modules by
  responsibility. There is no source-line acceptance limit.
- Promoted the managed Tier 0–4 profile from 351 to 352 expected passes. The
  active profile now contains 352 passes, 9 `missing_tflite_report` records,
  33 `tflite_fail` records, and 26 excluded historical timeouts.

### Verification at this checkpoint

- Final sequential Mask R-CNN run:
  - classification: `pass`;
  - duration: `17.98s`;
  - compared outputs: 4 of 4;
  - skipped outputs: 0;
  - `evaluation_pass=true`;
  - `max_abs=0.0`.
- Main affected suite: `868 passed, 5 deselected, 2 warnings in 88.50s`.
  The five deselections are the previously documented optional TensorFlow and
  incompatible external Python 3.10 Torch environment tests. The warnings are
  the existing expected float16 overflow warnings.
- Additional focused run after repairing the dynamic reshape interaction:
  `6 passed, 751 deselected`.
- `git diff --check`, undefined-name checks, import checks for the new modules,
  Python compilation, and managed baseline JSON parsing all passed.
- Every inference run used the `uv` environment and one active process at a
  time. No ProcessPool or parallel pytest worker was used.

### Remaining work after `c3c5ff7`

- Continue improving the remaining 42 active Tier 0–4 non-passes (9 missing
  reports and 33 accuracy failures), one model at a time in tier order.
- Run the complete sequential Tier 0–4 corpus before the final audit; the
  focused Mask R-CNN run and affected suites do not replace that corpus gate.
- Complete the original artifact-matrix, optional TensorFlow boundary,
  PyTorch-family exporter, performance/RSS, public-contract, and
  requirement-by-requirement audits.
- Tier 5 remains intentionally excluded until the Tier 0–4 core contract is
  stable.

### First action on the next resume

1. Confirm `fb-refactor2` is clean and synchronized with
   `origin/fb-refactor2`.
2. Select the next `missing_tflite_report` entry from
   `docs/baselines/flatbuffer_direct_active_tier0_4.json`, preserving tier and
   model order.
3. Reproduce it with a one-model temporary regression profile,
   `ONNX2TF_EVAL_IN_PROCESS=1`, fixed seed, and inference concurrency one.
4. Fix only a general semantic boundary with a synthetic unit test, then rerun
   the model and the affected suite before promoting its baseline.

No pull request should be created. Future checkpoints end at commit and push to
`fb-refactor2`.

This is the checkpoint for pausing work on `fb-refactor`. The worktree was
clean at the start of this handoff, and all implementation changes described
below were already pushed to `origin/fb-refactor` as commit `5944292`.

## Completed work

- Recovered `superpoint_lightglue_end2end_fused_cpu.onnx` by reconciling static
  tensor ranks on demand after attention/control-flow boundaries. This is in
  commit `0dbba12`; the recorded maximum absolute error is
  `1.946091651916504e-05`.
- Recovered the fixed-shape `silero_vad.onnx` with
  `-kat input state sr`:
  - runtime-state forward LSTM is lowered to ordinary TFLite primitives instead
    of an invalid mutable builtin-LSTM state connection;
  - flattened inactive `If` branches keep speculative Squeeze operations
    executable;
  - rank-1 singleton conditions use `SELECT_V2` when prefix-style `SELECT`
    broadcasting is invalid;
  - sample-rate control inputs use the deterministic value `16000` during
    accuracy evaluation;
  - ONNX Runtime's nested-LSTM rank-inference failure has a narrowly scoped
    ONNX ReferenceEvaluator fallback with complete `Y`, `Y_h`, and `Y_c`
    outputs.
- Verified fixed-shape Silero through the normal isolated `-cotof` path:
  `evaluation_pass=true`, `max_abs=1.375097781419754e-06`,
  `rmse=1.222495367934982e-07`.
- Promoted the managed Tier 0–4 profile to 343 expected passes and 51 expected
  non-passes. There are 394 active models and 26 recorded timeouts excluded
  from future validation. Tier 5 remains excluded; Tier 4 remains in scope.
- Confirmed that `silero_vad (1).onnx` is not an accuracy-comparable dynamic
  variant: the serialized source references 14 nonexistent lexical captures,
  including all four LSTM weight/bias captures. Both ONNX Runtime and ONNX
  ReferenceEvaluator reject it. It remains active as a documented non-pass
  input-model defect rather than being promoted with fabricated weights.
- Committed and pushed the Silero/control-flow work as:
  `5944292 recover silero recurrent control flow`.

## Incomplete work

- Continue improving every non-timeout Tier 0–4 model. The managed checkpoint
  currently has 19 `missing_tflite_report` and 32 `tflite_fail` entries.
- Remaining Tier 0 candidates are:
  - `inverse_11.onnx`;
  - `string_normalizer_11.onnx`;
  - `silero_vad (1).onnx`, whose source-model defect is described above.
- Tier 1–4 non-pass models remain improvement candidates after Tier 0.
- The original plan still needs a final requirement-by-requirement audit,
  including the complete artifact matrix, optional TensorFlow exporter
  boundary, PyTorch-family exporters, full sequential Tier 0–4 regression,
  and conversion-time/peak-RSS measurements. Existing partial tests are not
  evidence that this final audit is complete.

## Current branch and changed files

- Branch: `fb-refactor`
- Remote: `origin/fb-refactor`
- Implementation checkpoint: `5944292`
- The worktree was clean before adding this handoff document.
- Files changed by `5944292`:
  - `docs/baselines/flatbuffer_direct_active_tier0_4.json`
  - `docs/flatbuffer_direct_architecture.md`
  - `onnx2tf/tflite_builder/accuracy_evaluator.py`
  - `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
  - `onnx2tf/tflite_builder/op_builders/control.py`
  - `onnx2tf/tflite_builder/op_builders/elementwise.py`
  - `onnx2tf/tflite_builder/op_builders/recurrent.py`
  - `onnx2tf/tflite_builder/op_registry.py`
  - `onnx2tf/utils/onnx_reference_compat.py`
  - `tests/test_accuracy_evaluator_seeded_input.py`
  - `tests/test_flatbuffer_direct_bulk_runner.py`
  - `tests/test_onnx_reference_compat.py`
  - `tests/test_tflite_builder_direct.py`

## Tests already run

- Affected control-flow/LSTM/Squeeze/Where tests:
  `61 passed, 740 deselected, 1 warning`.
- ONNX reference compatibility and seeded evaluator tests:
  `30 passed`.
- Full relevant suite with the five unavailable optional-environment tests
  deselected:

  ```text
  uv run pytest -q \
    tests/test_tflite_builder_direct.py \
    tests/test_accuracy_evaluator_seeded_input.py \
    tests/test_onnx_reference_compat.py \
    tests/test_flatbuffer_direct_bulk_runner.py \
    -k 'not test_tflite_backend_matrix_add and not test_tflite_backend_matrix_hardswish_rewrite_on_off and not test_tf_converter_resize_cubic_avoids_flex_resize_bicubic and not test_tf_converter_resize_cubic_honors_cubic_coeff_a and not test_flatbuffer_direct_group_norm_alias_builtin_conversion'

  796 passed, 5 deselected, 2 warnings in 93.90s
  ```

- Fixed-shape Silero final verification:

  ```text
  uv run onnx2tf -i silero_vad.onnx \
    -o /tmp/silero_final_verify \
    -tb flatbuffer_direct -kat input state sr -cotof -v error
  ```

All inference checks were executed sequentially with one process active at a
time. No new package was introduced, and all commands used the `uv` environment.

## Failing tests and known issues

- Running the same full suite without deselection produces 796 passes and five
  environment-only failures:
  - four `tf_converter` tests require the optional TensorFlow/tf-keras extra,
    which is intentionally absent from the core environment;
  - one GroupNorm alias test imports an external Python 3.10 Torch binary from
    Python 3.12 and fails with `_PyCode_GetExtra`.
- Two existing float16 conversion tests emit an expected NumPy overflow warning
  while casting extreme values.
- `silero_vad (1).onnx` has the missing-capture defect described above.
- `inverse_11.onnx` is the next diagnosed non-pass. It contains `Resize` followed
  by a 224x224 `Inverse`. The direct builtin lowering intentionally supports
  matrices only up to 16x16, so conversion falls back to the unresolved custom
  op `ONNX_INVERSE` and LiteRT cannot allocate it. The evaluation compatibility
  layer maps the legacy empty-domain `Inverse` to `com.microsoft::Inverse`.
  ONNX Runtime produces finite but extremely large results (observed absolute
  values around `8.3e7`) because the resized matrices are nearly singular.
  A low-order approximate inverse is therefore unlikely to satisfy the `1e-1`
  accuracy requirement. No code was changed during this diagnosis.

## First work on resume

1. Confirm `git status --short --branch` is clean and still on `fb-refactor`.
2. Reproduce the `inverse_11.onnx` explicit evaluator failure:

   ```text
   ONNX2TF_EVAL_IN_PROCESS=1 uv run onnx2tf \
     -i inverse_11.onnx -o /tmp/inverse_explicit \
     -tb flatbuffer_direct --eval_with_onnx --eval_num_samples 1 -v error
   ```

3. Before implementing anything, determine whether an exact/stable 224x224
   inverse can be expressed with the existing TFLite primitive set and current
   dependencies while meeting `max_abs <= 1e-1` for the nearly singular
   reference. Do not introduce a silent approximation or a new dependency.
4. If no accuracy-preserving lowering is viable, record a precise normalized
   unsupported-capability reason, keep the model active as a non-pass, and move
   to `string_normalizer_11.onnx`. If a viable lowering exists, add a small
   well-conditioned numeric unit test first, then a singular/ill-conditioned
   guard, and finally rerun the root model sequentially.

The persistent project goal is paused at this checkpoint; it is not complete.
