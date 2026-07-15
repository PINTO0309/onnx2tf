# `flatbuffer_direct` refactor continuation checkpoint — 2026-07-14

## Status

The active branch is `fb-refactor5`, created from `main` after pull request
`#949` merged the complete `fb-refactor4` checkpoint. Pull request `#950` is
closed, and no open pull request tracks this branch. The Goal is active again;
subsequent work uses coherent commits and pushes without opening a pull
request.

The latest implementation unit moves the adjacent late SiNet residual fan-out
to `passes/sinet_shuffle_residual_layout.py` and reduces the former 331-line
raw mutator to a 17-line compatibility dispatcher. The owner replaces the
fixed 40-by-40 guard with exact NCHW/NHWC shape, broadcast, fan-out, and graph-
order contracts while preserving both the canonical conv branch and every
legacy NCHW branch.

The audited fast-precanonicalize orchestrator remains 294 lines, down from 482
lines at Goal resumption, 1,025 lines at the beginning of the previous
continuation, and 1,608 lines before the broader extraction.

## Completed work

The merged `fb-refactor4` checkpoints included:

- `062ddc4` — centralized DepthToSpace/Gather repair and the permuted-Conv
  statement decoder;
- `a522f1b` — centralized static NHWC Pool layout selection;
- `5eb86e1` — centralized CF Pool-neighbor repair with an explicit
  short-circuit contract;
- `d16265b` — centralized dynamic Pool layout repair and added the exact
  aligned-rank4 decoder;
- `e0bc280` — centralized simple-alias layout repair and moved all
  permuted-Conv decoder consumers into the policy owner;
- `afb5bb5` — centralizes aligned scalar-binary
  shape reconciliation and removes the now-unused aligned-rank4 and Softmax
  parser imports from the exporter.

The current `fb-refactor5` work contains 109 coherent continuations:

- `3ac19b40` centralizes the ordered fallback that repairs aligned binary
  shapes only when general binary repair made no change and the immediate next
  statement supplies matching BN, direct-return, or channel-first Resize
  evidence;
- `008e4ad0` centralizes the following Resize fallback, including
  exact direct and reshaped BN-constant parsing, preferred-channel selection,
  input-layout guards, and CF evidence propagation;
- `80d1d6a5` centralizes the aligned BatchNorm-constant rewrite itself while
  preserving the different direct and already-reshaped guards;
- `91a0a52d` centralizes LRN output evidence propagation without changing or
  broadening the generated source grammar;
- `b00774a7` centralizes literal static-shape recording while retaining each
  update at its original point in the ordered scan;
- `95fbd0cb` makes the NHWC AveragePool bridge own the CF/NHWC and static-shape
  state resulting from its rewrite;
- `907c91fa` routes all direct-export option reads through the normalized
  request and adds a structural boundary test;
- `5848cc28` adds request-aware optional exporter controls and removes eager
  parsing of unrequested PyTorch settings;
- `e3c03e3d` makes quant type and input/output quant dtype part of the guarded
  immutable quantization controls;
- `4f3d20b0` restores Session-owned consumer counts at the `LoweringContext`
  boundary;
- `6e8a8486` synchronizes lowering-time logical and physical layout mutations
  with the Session before the first post-lowering pass;
- `e7c1457d` centralizes lowering-time operator removal and protects synthetic
  inverse-Transpose fan-out with differential consumer counts;
- `2a2968cc` lazily shares `ModelIRPassState` across the repeated
  Mean/attention registered-pass cluster;
- `514fc683` applies the same bounded reuse contract to the repeated mixed-
  attention through dual-Mul/Concat registered-pass cluster;
- `9b32c680` shares state only across the separately audited late NDHWC-gate/
  cost-volume-scatter pair;
- `1aa636e3` shares state across the following four registered Concat,
  LayerNorm, and transpose-cleanup runners;
- `969d5e26` shares state across the repeated channel-shuffle/Gather-axis and
  unary fan-out runner clusters;
- `251edc58` shares state across four repeated boundary-input BatchMatMul/input-
  unary runner pairs;
- `543d7cc3` shares state across three repeated channel-slice-merge/Pad-Mul
  pairs;
- `93a0295a` shares state across the repeated long singleton/
  Reshape sequences and all three terminal singleton-channel/duplicate-fan-
  out/consecutive-Reshape triplets;
- `9a75c43d` shares state across two repeated QKV attention prefix/bridge
  pairs;
- `417ee06e` shares state across two repeated duplicate-fan-out/quantized-PReLU
  pairs;
- `0c76774e` shares state across two repeated constant-input-fold/redundant-
  Cast pairs;
- `f0dac050` shares state across the fallback and primary absolute-final SE-FC/
  Gather-channel-fan-out pairs;
- `e177face` shares state across the five-runner terminal boundary/layout
  sequence;
- `ce11c27f` shares state across the late Dequantize/Concat/Quantize and unary-
  fan-out sequence;
- `d8b2b58c` shares state across the terminal singleton-MaxPool/consecutive-
  Reshape pair;
- `fcae7233` shares state across the terminal scalar-clamp, unary-passthrough,
  and maximum-zero-to-ReLU sequence;
- `36f73b18` shares state across the late Mean/Mul/Add/Conv, generic SPP, and
  Gather-axis sequence;
- `887db85d` shares state across the late generic SPP and Concat/unary/Conv
  pair;
- `9680fa33` shares state across the absolute-final normalization-Pad and mixed-
  attention pair;
- `a8baec98` shares state across the post-QDQ layout-transpose and unary fan-out
  sequence;
- `916419d9` shares state across the late NCHW channel-shuffle and Gather-axis
  pair;
- `8eaab05b` shares state across the conditional late generic-transpose and
  QKV-bridge pair;
- `e6be5539` shares state across the very-late Gather-axis, constant-fold/Cast,
  and normalization-Pad sequence;
- `fa33fd67` shares state across the terminal hard-activation and optional
  generic-Transpose pair;
- `57d79e3b` shares state across the conditional generic-Transpose, late
  Mean/SPP/Gather, and constant-fold/Cast sequence;
- `a353580b` makes `ArtifactPlan` the only request input to artifact controls
  and progress planning;
- `5e4e14d5` centralizes TFLite evaluation-path selection from returned direct
  artifacts;
- `d7fb5969` centralizes direct report and quantized-artifact
  validation and completion logging without changing messages or skip
  behavior;
- `603d6557` makes the direct fast path the sole direct conversion
  owner and removes the unreachable TensorFlow-failure fallback and duplicate
  post-SavedModel direct serialization path;
- `69a4eccd` centralizes the four identical 19-call layout-recovery prefixes
  without sharing pass state across raw mutation boundaries;
- `c89e94b5` centralizes two identical 22-call attention and quantized-
  recovery suffixes while preserving the separate LayerNorm variant;
- `329306d3` centralizes three identical 16-call layout/reshape/attention
  recovery prefixes while preserving their distinct successors;
- `40dcd142` centralizes two identical 14-call terminal slice/Concat layout-
  recovery sequences while retaining their boundary variants;
- `daab0828` centralizes two identical 11-call terminal affine/Concat/split
  recovery sequences between their distinct raw boundaries;
- `04c7dc03` centralizes three identical 10-call attention/gate/QDQ recovery
  sequences while retaining their distinct successors;
- `1713d089` centralizes two identical 10-call quantized-activation/binary-
  bridge sequences while preserving their conditions;
- `501e616f` centralizes two identical 8-call SiNet terminal recovery sequences
  without crossing shape reconciliation;
- `9bd57ac2` centralizes two identical 7-call pre-Add/Mean attention-recovery
  sequences while retaining their distinct boundaries;
- `ef61c03c` centralizes four identical 6-call SiNet pre-Add/Resize recovery
  sequences while retaining all external boundaries;
- `48dd2324` centralizes the remaining safe-binary and QLinear/Mean/Concat
  5-call families while preserving their conditions;
- `4a9bde4c` shares one differential graph index through each repeated
  prune/reconcile/Reshape-resolution convergence block;
- `20290fce` extends the final indexed convergence boundary
  through HARD_SWISH sanitation and activation fusion without rebuilding
  consumers or the graph index;
- `864af4c9` indexes repeated rank-four channelwise broadcast-constant repair
  while preserving its start-of-pass shared-constant policy;
- `0027ccfa` indexes stale channelwise-binary Transpose repair and shares one
  index across both terminal three-round convergence loops;
- `902cab42` indexes the singleton-Reshape and stale-Transpose Conv-input
  repair pair and shares one index across its primary and fallback invocations;
- `79d30ae1` indexes the wrong-way NCHW-to-NHWC Transpose-before-Conv
  sanitizer and removes its per-match consumer-map rebuild;
- `1ad30cbc` centralizes direct and PyTorch recurrent orphan-step alias repair
  in one Torch-free differential-index owner;
- `2574ae1f` extracts all unbound-input layout repair families to one
  differential-index owner and removes their repeated graph rescans;
- `16bba4ea` extracts quantized RELU/RELU6 Transpose bridge cleanup to one
  differential-index activation owner;
- `bbc9d345` extracts both expanded HardSigmoid QDQ Transpose
  bridge forms to that owner, adds transactional constant preflight, and
  protects every clamp intermediate at the public boundary;
- `515bc99b` extracts expanded MUL/ADD/PRELU QDQ Transpose bridge cleanup and
  shares only the identical constant plan/apply mechanism;
- `49f53b1a` extracts quantized logistic-gated MUL bridge cleanup to a
  dedicated indexed owner with differential alias consolidation;
- `30d00239` gives the wrong-way Transpose-before-Conv sanitizer one dedicated
  owner and removes its duplicate Swish-local implementation;
- `bad1a806` extracts the primary Swish-QDQ NHWC branch rewrite into one
  differential-index owner with an explicit phase result contract;
- `406136b5` extracts its four-family metadata fixed point and shares one index
  across both ordered primary phases;
- `a91d7bad` gives the two identical inverse post-Transpose sweeps one
  differential-index owner while preserving their separate call sites;
- `02462462` extracts transactional late mixed-input Concat
  normalization and shares one maintained index with its following post-
  Transpose cleanup;
- `03742a6a` moves Concat pre-Q/DQ exact-grid bypass to its
  quantization owner and replaces repeated whole-graph maps with one
  differential index;
- `55ad5c88` moves both terminal Transpose/Dequantize sanitation
  subphases to the same owner and maintains one index through edge rewrites,
  operator movement, rename, and removal;
- `f6b62363` moves the Transpose-DQ-Mean-Q bridge to one indexed, fully planned
  quantization-cleanup transaction;
- `3042329e` moves pseudo-op LeakyReLU fusion to one indexed graph-cleanup
  owner with batch producer compaction;
- `9a513d4c` moves the former YOLO MUL-square fold to a generic indexed
  constant-fold owner and protects public intermediates;
- `616a6a6b` moves leading-singleton Gather-to-Reshape cleanup to one indexed
  shape/indexing owner and makes every unsafe metadata or topology case
  transactional;
- `f3da692f` moves marker-gated terminal Softmax/Transpose cleanup to one
  indexed terminal-layout owner and centralizes the propagation marker;
- `e1e8ab39` moves pre-ArgMax channel-layout cleanup to one indexed owner with
  transactional shape and constant-ownership guards;
- `0cfc1ef9` moves exact-grid quantized MaxPool cleanup to one indexed owner
  with transactional topology, grid, and metadata guards;
- `e3dde5a4` moves canonical quantized Logistic cleanup to one indexed owner
  with transactional topology, grid, and metadata guards;
- `2a6178f6` moves canonical last-axis quantized Softmax cleanup to one indexed
  owner with transactional option, grid, and metadata guards;
- `163f9875` moves expanded HardSigmoid QDQ cleanup to one indexed owner with
  a complete four-constant transaction;
- `cc699155` moves quantized TransposeConv QDQ cleanup to one indexed owner
  with a complete filter/output transaction;
- `2b181ed7` moves decomposed InstanceNormalization layout repair to one
  indexed owner with a complete tensor-metadata transaction;
- `558973fd` moves NCHW Concat/global-pool/Conv axis repair to one indexed
  owner with a complete options/metadata/buffer transaction;
- `78ba42ae` moves NCHW Concat/Transpose/(Transpose)Conv axis repair to one
  indexed owner with a complete metadata transaction;
- `bee33d8e` moves mixed singleton NCHW-input repair for NHWC Concat to one
  indexed owner with complete adapter transactions;
- `84ac0fae` moves Swin-style window-partition canonicalization to one indexed
  owner with complete topology/constant/metadata transactions;
- `b134767b` moves the paired Swin-style window-reverse
  canonicalization to that owner with deterministic shared-shape cloning and
  complete topology/constant/metadata transactions;
- `b86ce908` moves the Conv1D-shim Squeeze/Unary/ExpandDims
  canonicalization to one indexed owner with complete shape, topology,
  constant, dtype, and quantization transactions;
- `54d37f35` moves the adjacent rank-four
  Unary/Reshape/ExpandDims variant to that indexed owner, makes shared
  constant changes transactional, and keeps fan-out compatibility bridges
  topological;
- `0837ee1b` moves the unary fan-out bypass to the same indexed
  owner, shares the common Transpose/Squeeze/Unary prefix contract, preserves
  only genuine NCHW side branches, fixes operator ordering, and repairs CAST
  metadata;
- `54384df0` moves flattened InstanceNormalization Conv1D layout
  canonicalization to a dedicated indexed owner with complete decomposition
  and shared-constant transactions;
- `6be406ec` shares that exact normalization prefix with a
  dedicated indexed tencoder residual-gate owner, makes the complete dual-
  branch rewrite transactional, and keeps compatibility bridges topological;
- `2202bd0d` moves the adjacent Squeeze/unary/BatchMatMul layout rewrite to an
  indexed owner with explicit axis-to-adjoint semantics;
- `0ef2050c` moves the decoder deconvolution-input adapter to a complete
  indexed matrix/layout/constant transaction;
- `481098db` moves the terminal Squeeze/Mean decoder adapter to a complete
  indexed axis/layout/output transaction;
- `fd3d1d32` moves the direct decomposed-InstanceNormalization pre/post adapter
  to a complete indexed topology/layout/constant transaction;
- `c7496639` moves its dual-consumer post-Transpose plus side-Squeeze tail to
  the same indexed owner with a transactional local compatibility adapter;
- `50278afa` moves the Squeeze/unary/Reshape tail to the same indexed owner
  with a transactional second-Reshape constant and output-name rewrite;
- the current checkpoint extracts the common SiNet Shuffle residual prefix,
  moves the paired post-MUL Transpose variant to the same indexed module,
  preserves the already-NHWC ADD/PReLU tail, and removes its second legacy raw
  mutator from the compatibility dispatcher.
- the latest checkpoint moves the late residual affine/PReLU fan-out to that
  indexed module, preserves the conv and legacy branches with one inverse
  adapter, and replaces the spatial-size heuristic with semantic shape and
  broadcast validation.

The extraction preserves the ordered source-rewrite behavior. Layout evidence
continues to mutate only the per-run CF/NHWC sets; repair context maps remain
shared. Rules that formerly used `continue` return an explicit short-circuit
result to the exporter. Exact generated-statement grammars remain rule-local or
use the shared Torch-free parser owner.

No dependency was added and no TensorFlow path was introduced. The latest
checkpoint includes one sequential direct-backend integration smoke; no Tier
corpus run was performed.

## Current branch and changed files

Branch: `fb-refactor5`, tracking `origin/fb-refactor5`.

The current checkpoint changes:

- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`;
- `onnx2tf/tflite_builder/passes/sinet_shuffle_residual_layout.py`;
- `tests/test_flatbuffer_direct_indexed_sinet_shuffle_residual_layout.py`;
- `tests/test_flatbuffer_direct_architecture.py`;
- `docs/flatbuffer_direct_architecture.md`;
- this handoff document.

The expected handoff state after committing and pushing is an empty `git
status --short` with local `fb-refactor5` equal to `origin/fb-refactor5`.

## Important design decisions

- The exporter remains the ordered orchestration owner; match/guard/rewrite
  decisions move to `pytorch_fast_precanonicalize_policy.py` one coherent
  family at a time.
- Indexed helpers receive the current line index, shared source lines, mutable
  layout evidence, and the shared repair context. They do not rescan the full
  generated source unless the preserved rule already required a bounded scan.
- Former loop `continue` behavior is represented explicitly in helper results;
  extraction must not silently allow later rules to run.
- General binary repair remains first. The downstream-evidence fallback is
  called only from its unchanged no-rewrite branch, and its returned CF
  evidence is visible to the following Resize repair in the same scan.
- The fallback deliberately retains its narrower positional grammar and legacy
  `_in` naming evidence. It additionally requires an immediate matching BN,
  direct return, or channel-first Resize; mismatched channels and mixed-layout
  names remain no-ops.
- General Resize repair also remains first. The input/BN-evidence fallback runs
  only afterward, uses an immediate matching direct or reshaped BN constant as
  the preferred channel hint when available, and otherwise retains the legacy
  input/source channel fallback. Its returned CF evidence remains visible to
  Pool and later aligned-constant decisions in the same ordered scan.
- Explicit NHWC Resize inputs and already-channel-first target shapes remain
  no-ops. BN evidence refines the preferred channel count but is not a
  prerequisite for the legacy CF-input repair.
- Direct aligned BatchNorm constants require a registered channel count that
  matches the generated target channel before a reshape is introduced.
  Already-reshaped constants intentionally retain the older, narrower rule:
  their explicit reshape channel drives normalization without requiring a
  registered-buffer channel lookup. Both forms still require CF input and a
  BatchNorm-derived attribute name.
- LRN output propagation is state-only: exact CF input evidence adds the output
  to the CF set, removes stale NHWC evidence, and copies only a known rank-four
  static input shape. It does not mark the source file changed or rewrite the
  LRN statement.
- Rewritten-shape caching accepts only a literal `target_shape=[...]` or the
  exact trailing aligned shape. Dynamic and unparseable expressions do not
  replace existing cache entries, and binary/Resize/Pool callers still update
  the shared context immediately after their successful rewrite.
- The NHWC AveragePool bridge keeps its returned-name contract, but successful
  calls now update the layout sets and all four affected static-shape entries
  internally. To preserve behavior, the cached state shape is still recomputed
  from the pre-rewrite Pool shape after the layout sets change; it is not
  replaced by the rendered rewrite target.
- `ConversionRequest.from_kwargs` is the direct exporter's only raw-kwargs
  boundary. Quantization validation receives `request.options`, normal option
  reads use `request.get`, and typed artifact decisions remain on
  `request.artifacts`. Checkpoint `907c91fa` mechanically converted all 36
  former raw reads without changing keys, defaults, coercions, public return
  values, or downstream arguments.
- `resolve_requested_exporter_controls` now owns seven artifact-specific
  settings. It performs no option reads when SavedModel, PyTorch, and integer
  calibration are all unrequested. Requested output paths, persistence,
  timeout conversion, shape/test data, and custom-input values preserve their
  existing defaults and dependencies.
- Requested quantization controls now also own `quant_type`,
  `input_quant_dtype`, and `output_quant_dtype`. The builder reads these values
  only from the resolved immutable mapping; when quantization is unrequested it
  uses the legacy local defaults without touching the corresponding options.
- Artifact execution controls, exporter controls, and export-progress labels
  now accept `ArtifactPlan` directly. They do not receive independently
  reconstructed split, quantization, SavedModel, PyTorch, or calibration
  booleans. Derived PyTorch artifacts are normalized once by
  `ArtifactPlan.from_options`, and all downstream policy sees the same
  immutable dependency decision.
- TFLite evaluation consumes the direct builder's returned artifact mapping
  through one fixed seven-key selector. The three compatibility-layer exit
  paths no longer own parallel key-copy chains, cannot diverge in variant
  order, and do not infer a path for an artifact the builder did not return.
- `LoweringContext.tensor_consumer_count` is populated from
  `ConversionSession.tensor_consumer_count`, not an empty compatibility
  dictionary and not a new ONNX scan. This restores the original fan-out guard
  used by inverse-transpose elision and preserves duplicate input occurrences.
- `LoweringContext.set_tensor_layout()` is the lowering-time layout mutation
  boundary. It normalizes and writes `TensorIR` metadata and immediately
  records the same logical/physical values in the Session-owned `LayoutState`.
  Shape-family edge-Pad passthroughs, integer-linear Resize casts, and rank-three
  Resize adapters no longer assign layout fields directly. This fixes observed
  pre-pass staleness without adding an eager ModelIR-wide synchronization.
- `LoweringContext.add_operator()` increments a differential consumer count for
  every emitted input occurrence. `remove_operator()` decrements those counts
  and removes only producer entries owned by the removed object. The inverse
  Transpose helper uses the authoritative ONNX count when present; otherwise it
  adds the pending inverse use to the current synthetic IR count. An exclusive
  pair is still elided, while a synthetic side consumer keeps its producer.
  This replaces the only direct op-builder deletion and adds no partial-graph
  scan.
- `ModelIRPassStateScope` is lazy and identity-bound. A group with a successful
  model-only preflight acquires the state; subsequent adjacent groups reuse it
  and report `state_built: false`, so diagnostic `state_build_count` reflects
  actual construction rather than pass invocation count. The scope is never
  carried across legacy helpers that mutate ModelIR outside the differential
  index. All six production occurrences retain the exact order
  transpose-Mean, Mean/Mul/Add/Conv, optional LayerNorm, terminal Mean, SE conv,
  SE FC, and optional Conv attention.
- Each repeated shape-convergence block uses one `ModelIRGraphIndex`.
  Dead-operator pruning updates that index through `remove_operators`; the
  following reconciliation and dynamic-Reshape steps change only metadata,
  options, and constant data. The first block builds its own index. The final
  convergence owner supplies one index to the second block and retains it
  through HARD_SWISH sanitation, a second Reshape/reconcile cycle, activation
  fusion, and final reconciliation. Standalone callers retain compatibility
  fallbacks, and an index for another ModelIR is ignored safely.
- Indexed activation fusion preserves the former graph-order and case-
  normalized op matching. It queries producer/activation fan-out from the
  index, updates producer outputs through `_set_operator_outputs`, and removes
  fused activation operators through `remove_operator`. It no longer rebuilds
  the full consumer map for every successful match. Differential single-
  operator removal drops empty type buckets so its type dispatch exactly
  matches a fresh index.
- Rank-four channelwise broadcast-constant repair takes an optional matching
  graph index and otherwise builds exactly one. It enumerates only the exact
  binary-op family, queries producer layout evidence through the index, and
  routes cloned-constant input changes through the differential setter. Its
  consumer fan-out map is intentionally snapshotted once from that index:
  clone-versus-in-place decisions therefore retain the former start-of-pass
  behavior even after earlier candidates update live consumers.
- Stale channelwise-binary Transpose repair also accepts an optional matching
  graph index. It retains exact graph-order binary matching, resolves adapter
  and NHWC peer producers through the index, and requires the indexed adapter
  consumers to equal the current binary index. Successful rewrites use the
  differential input setter and operator removal; a fan-out adapter remains
  untouched. `_run_indexed_binary_layout_convergence` owns the existing three
  broadcast-repair, Transpose-repair, and reconciliation rounds and supplies
  the same index to all nine calls in both primary and fallback finalization.
- The same scope contract covers only the five repeated gate-layout sequences
  that were audited as contiguous registered runners. Four keep the exact
  mixed-attention, elementwise-gate, Pad, dual-postconv-gate, NDHWC-gate,
  cost-volume-scatter, Add/Concat-suffix, and dual-Mul/Concat order. The fifth
  starts at elementwise-gate exactly as before. All eight runners retain
  standalone behavior through an optional `state_scope` argument, and the
  later isolated mixed-attention, Pad, NDHWC, cost-volume, and dual-Mul calls
  intentionally do not share this scope.
- The late mixed-attention/NDHWC/cost-volume candidate is not one valid scope:
  the raw dequantize/HardSigmoid/quantize optimizer between mixed attention and
  NDHWC is a hard boundary. A new scope is therefore constructed only for the
  immediately adjacent NDHWC and cost-volume runners, and ends before the raw
  convolution-affine optimizer. Both runners preserve standalone behavior.
- After that convolution-affine boundary, axis-3 constant-Concat,
  Dequantize/Concat/Quantize, LayerNorm-statistics, and generic transpose
  cleanup form one independently audited four-runner scope. Each runner either
  already accepted a scope or now exposes the same optional
  standalone-compatible argument; the scope ends before the conditional raw
  elementwise-roundtrip optimizer.
- Two repeated cluster families now have explicit helper-owned scopes. The
  channel-shuffle helper preserves two-way, NHWC, NCHW, and Gather-axis order
  at all five call sites; only its final invocation enables the already-
  contiguous generic transpose and unary/binary fan-out suffix. The separate
  unary helper preserves passthrough, unary fan-out, and unary/binary fan-out
  order at four call sites. Every runner remains callable standalone through
  an optional scope argument.
- Four repeated boundary-input BatchMatMul/input-unary pairs now use a small
  helper-owned scope. The boundary BatchMatMul runner and the three-spec input-
  unary runner expose the same optional standalone-compatible scope argument.
  No scope crosses the legacy transformations surrounding any occurrence.
  Their two stale `_build_tensor_consumer_map` imports are removed; neither
  module constructs an ad hoc consumer map for these runners.
- Three repeated channel-slice-merge/Pad-Mul pairs now use a two-group helper-
  owned scope. Both runners retain optional standalone-compatible scope
  arguments, and the scope ends before the legacy optimizer following each
  pair.
- Two long singleton/Reshape sequences now use one flag-controlled helper-
  owned scope per occurrence. The first retains generic transpose cleanup and
  terminal multi-branch gate cleanup; the second retains reshape-only
  duplicate fan-out cleanup and disables only the former spatial post-Concat
  variant. Four singleton-Reshape-family runners, three graph-cleanup runners,
  singleton MaxPool, and multi-branch gate retain standalone behavior through
  optional scope arguments.
- The three later singleton-channel/reshape-only-duplicate/consecutive-Reshape
  triplets use a target-parameterized helper. Two invocations use the primary
  ModelIR and Session layout state; fallback relowering passes `fallback_ir`
  and no LayoutState, preventing state identity from crossing conversion
  instances. The terminal singleton-MaxPool/consecutive-Reshape pair remains
  outside this target-parameterized helper and owns a separate bounded scope.
- Two repeated QKV attention prefix/bridge pairs use a two-runner helper-owned
  scope. The four prefix specs and two bridge specs retain exact order and
  diagnostic grouping. Both runners expose optional standalone-compatible
  scope arguments; the separate later bridge-only call remains independent.
- Two repeated duplicate-fan-out/quantized-PReLU pairs use a helper-owned
  scope. The helper forwards
  `enable_duplicate_transpose_fanout_optimizations` unchanged, then runs all
  four PReLU specs. The quantized-PReLU runner retains standalone behavior
  through an optional scope argument, and no scope crosses the following raw
  quantized TransposeConv cleanup.
- Two repeated constant-input-fold/redundant-Cast pairs use a helper-owned
  scope. The constant Pad, Pool, and Cast specs retain order before the
  redundant widening-alias and narrowing-chain specs. Both runners expose
  optional standalone-compatible scopes, and neither production scope crosses
  the immediately following legacy mutator.
- The fallback and primary absolute-final SE-FC/Gather-channel-fan-out pairs
  use a target-parameterized helper. Fallback receives `fallback_ir` and no
  LayoutState; primary receives the main ModelIR and Session state. Gather
  channel fan-out now exposes an optional standalone-compatible scope, and
  neither target's scope crosses shape reconciliation.
- The five-runner terminal dual-Mul/Concat, boundary-input, Pad, generic
  transpose, and Gather-channel-fan-out sequence uses one helper-owned scope.
  Boundary-input cleanup now exposes an optional standalone-compatible scope.
  Architecture checks fix the raw InstanceNorm predecessor and conditional
  Mean/attention successor as hard boundaries.
- The late Dequantize/Concat/Quantize, unary-passthrough, and unary-fan-out
  sequence uses one helper-owned scope. All three runners already expose
  optional standalone-compatible scopes. Architecture checks fix the raw
  Dequantize/HardSigmoid/Quantize predecessor and raw swish successor as hard
  boundaries, preventing shared state from crossing a legacy mutator.
- The terminal singleton-MaxPool/consecutive-Reshape pair uses one helper-
  owned scope. The two singleton-MaxPool specs retain their order before the
  general Reshape cleanup spec. Architecture checks fix the conditional
  elementwise-roundtrip predecessor and conditional Conv/Pool-output successor
  as hard boundaries.
- The terminal scalar-clamp, unary-passthrough, and maximum-zero-to-ReLU
  sequence uses one helper-owned scope. Clamp and maximum-zero runners now
  expose optional standalone-compatible scopes. Their op-type rewrites use
  `ModelIRGraphIndex.replace_operator_type()` instead of direct assignment, so
  later passes see current type dispatch without a full refresh. Architecture
  checks fix the conditional terminal layout-recovery predecessor and raw
  SiNet successor as hard boundaries.
- The conditional generic-Transpose, late Mean/Mul/Add/Conv, generic SPP,
  Gather-axis, and constant-fold/Cast sequence uses one helper-owned scope.
  Disabling layout optimization skips only the first runner. Architecture
  checks fix the complete order, runtime flag, shared scope keywords, and raw
  shape-extract/ExpandDims boundaries.
- The late generic SPP and Concat/unary/Conv pair uses one helper-owned scope.
  Concat/unary/Conv cleanup now exposes an optional standalone-compatible
  scope and retains its differential index mutations. Architecture checks fix
  the raw StridedSlice/Pad/Concat predecessor and raw shape-extract successor
  as hard boundaries.
- The absolute-final flattened-normalization Pad and mixed-attention pair uses
  one helper-owned scope. Normalization-Pad cleanup now exposes an optional
  standalone-compatible scope. The helper preserves `include_instance=False`
  and `include_flatten=True`; architecture checks fix those flags and the raw
  InstanceNorm/dynamic-rank shape-rewrite boundaries.
- The post-QDQ layout-transpose, unary-fan-out, and unary/binary-fan-out
  sequence reuses the existing helper with compatible mode flags. Four prior
  invocations keep their default unary-passthrough path; one new invocation
  enables layout-transpose and disables unary-passthrough. Architecture checks
  fix the unique flag combination and the raw Softmax/transpose-binary
  boundaries.
- The late NCHW channel-shuffle/Gather-axis pair reuses the existing helper
  with two-way and NHWC shuffle disabled. Five prior invocations retain both
  modes by default. NHWC and NCHW shuffle op-type changes now use
  `replace_operator_type()`; real rewrite tests assert the reused type index.
  Architecture checks fix the unique flag combination and raw Reshape/QKV
  boundaries.
- The conditional late generic-transpose/QKV-bridge pair reuses the QKV helper
  with prefix disabled. Two prior invocations retain the default prefix-plus-
  bridge path. The new invocation forwards `optimize_layout_transpose_chains`
  to the layout mode while always running bridge cleanup. Architecture checks
  fix the runtime flag and raw shape-extract/split-Conv boundaries.
- The very-late Gather-axis, constant-fold/Cast, and normalization-Pad sequence
  uses one wrapper-owned scope. The constant-fold/Cast helper accepts an
  optional external scope; both production invocations now receive their
  enclosing wrapper scope. The normalization include flags remain false/true.
  Architecture checks fix both external scopes, runner order, flags, and raw
  repair/Reshape boundaries.
- The terminal hard-activation and conditional generic-Transpose pair uses one
  helper-owned scope. Hard activation now exposes an optional standalone-
  compatible scope. Its late false/true/true/reversed flags are unchanged,
  generic Transpose still depends on the runtime layout switch, and the scope
  cannot cross either neighboring raw rewrite.
- The singleton-Reshape and stale NCHW-to-NHWC Transpose Conv-input repairs
  accept an optional matching `ModelIRGraphIndex` while retaining standalone
  compatibility. Their primary and fallback pair owner constructs one index,
  and successful Conv input rewrites and adapter removals update that index
  differentially. Exact filter input-channel, one-consumer, graph-output,
  tensor-shape, and Transpose-permutation guards are unchanged. The later
  standalone stale-Transpose cleanup remains a separate compatibility call
  because intervening raw mutations form an ownership boundary.
- The wrong-way NCHW-to-NHWC Transpose-before-Conv sanitizer has one Torch/
  TensorFlow-free semantic owner in `passes/conv_input_layout.py`. It owns one
  `ModelIRGraphIndex` for an invocation with Transpose candidates, enumerates
  indexed roots, validates every indexed consumer before changing a shared
  adapter, rewrites all accepted Conv inputs through the indexed global-input
  replacement helper, and removes the adapter differentially. A graph without
  any Transpose skips index construction while preserving the former tensor-
  pruning side effect. The lowerer compatibility wrapper and the independent
  safety-valve phase inside the Swish-QDQ optimizer both delegate to this same
  owner; the latter preserves its historical execution point and removal
  statistics. Exact permutation, rank-four metadata, all-consumers-are-Conv,
  filter-channel, graph-output, and nonempty-consumer guards are unchanged.
- The primary Swish-QDQ branch phase has one Torch/TensorFlow-free owner in
  `passes/quantized_swish_layout.py`. Its result carries the exact branch and
  pre-Transpose counts plus an immutable rewritten-tensor set into the
  existing later phases. A matching supplied index is reused; otherwise one
  index is built only when a Transpose exists. Graph-order candidates, all
  source/gate/data/tail guards, both DQ rewrites, and unused pre-Transpose
  removal use the maintained index. Only source edges and an unused root can
  change, so downstream consumers are read directly without full-map copying.
  Shared-input ordering, quantized and float tails, peer-Swish acceptance,
  spatial and concat-closure modes, public boundaries, fan-out, metadata
  permutation, and ordered restart retain the former implementation exactly.
- The Swish-QDQ metadata phase keeps unary quantization, binary broadcast,
  Pool/Resize channel propagation, and strict Concat-tail normalization in one
  fixed-point owner because every family mutates the same rewritten-tensor
  state. It iterates only the graph-ordered relevant type buckets and reuses
  stable indexed consumers; it performs no topology mutation. Empty seeds
  allocate no index. A module-level runner gives the branch and metadata phases
  the same index, reducing the complete primary sequence to one construction.
  Public outputs, shape/signature copy, broadcast fallback, channel guards,
  normalized axis, tail fan-out, Concat/quantized metadata, and fixed-point
  restart semantics remain unchanged. The shape/signature copier is also the
  explicit owner used by the later Dequantize-input repair, preserving the
  previously hidden cross-phase dependency without a lowerer-local closure.
- The two Swish-QDQ inverse post-Transpose sweeps share one semantic owner.
  The first remains before late Concat normalization; the second is called by
  the shared late-phase runner after normalization. Empty rewritten state and
  Transpose-free graphs return without an index. Otherwise indexed graph-order
  candidates, global alias replacement, and differential removal preserve
  ordered restart without per-removal full scans. Public aliases, wrong
  permutations, and inputs outside the rewritten set remain protected; alias
  chains and arbitrary consumer fan-out are rewired before removal.
- Late Swish-QDQ Concat normalization validates one complete transaction before
  mutation. Direct and Dequantize-wrapped pre-Transposes, rank-four normalized
  shapes, a private Concat output, and the strict Quantize/all-inverse-
  Transpose tail must all agree. Accepted Concat/DQ edge rewrites update one
  maintained index; axis and shape/signature metadata commit together; and
  only newly unused input adapters are removed. The owner restarts after each
  accepted transaction to avoid stale indices after compaction. Its runner
  gives the immediately following inverse-post owner that same index while
  retaining the original statistics and phase order.
- Concat pre-Q/DQ bypass remains intentionally narrower than generic redundant-
  quantization cleanup. The Quantize input must be the direct output of a
  Dequantize, source and destination quantized tensors must share the complete
  exact grid, and no arithmetic intermediate is accepted. The owner rewires
  only the Concat edge; existing Q/DQ operators remain available to later
  cleanup, preserving the historical ordered pipeline. It restarts after each
  indexed edge change and performs the former pruning side effect even when no
  Concat exists, without allocating an index in that no-candidate case.
- Terminal Transpose/Dequantize sanitation retains two explicit counters and
  subphases even though a first-subphase match becomes eligible for the second
  after its indexed reorder. This preserves the established stats and phase
  semantics. Operator order changes use `remove_operator()` followed by
  `insert_operator()` on the same index; public output rename occurs before
  differential Transpose removal so the Dequantize remains the sole producer.
  Graphs missing either required operator type still receive historical tensor
  pruning without paying for an index.
- The Transpose-DQ-Mean-Q bridge commits only after the mapped axes, reduced
  metadata, bridge permutation, and both unique tensor names are valid. The new
  preserving Transpose is inserted immediately before the current Quantize,
  then the old pre-Transpose is resolved by object identity and removed from
  the same index. This preserves valid ordering while making invalid-
  permutation rejection a complete no-op instead of retaining the former
  partial DQ/Mean metadata mutation.
- Pseudo-LeakyReLU fusion remains an exact ordered grammar rather than a
  commutative algebraic matcher: positive RELU is SUB input zero and the scaled
  negative branch is input one. Only MUL's singleton alpha may swap sides. The
  retained SUB is converted through indexed type/input updates and its options,
  axis semantics, version, and ONNX provenance are reset to the same defaults
  as a fresh `OperatorIR`; all four private producers are then removed in one
  batch compaction.
- The MUL-square fold's semantic owner is model-neutral; only the compatibility
  wrapper and historical stats key retain `yolo` naming. The self-square must
  use the identical tensor name in both MUL slots, while constant side inputs
  at the pre/anchor/scale MULs remain commutative. Fused data is calculated in
  float32, checked finite, cast back to anchor dtype, and receives cloned anchor
  quantization. All three removable intermediates are now explicitly rejected
  when public before any tensor or edge mutation.
- Leading-singleton Gather-to-Reshape cleanup accepts one signed integer zero
  in either a scalar or TFLite-legalized singleton buffer, because both select
  the only leading slice without changing element order or count. It requires
  axis zero after negative-axis normalization, batch dimensions zero, a
  statically fixed leading-one signature, exact rank-reduced tail shape and
  signature, matching dtype and quantization, one topologically later Reshape
  consumer at data input zero, and no public or duplicate-produced Gather
  output. All guards finish before the indexed Reshape edge is changed and the
  Gather is removed. Missing Gather/Reshape families still receive historical
  unused-tensor pruning without allocating an index, and the active
  `LayoutState` receives the same pruning at both production call sites.
- Terminal Softmax/Transpose cleanup consumes only the shared
  `_SOFTMAX_NHWC_PROPAGATED_MARKER` produced by the preceding canonicalizer;
  the string literal no longer has two owners. Public outputs remain the
  deterministic candidate order. The maintained index requires the output to
  have no internal consumer and one Transpose producer, the private Softmax
  intermediate to have one Softmax producer and exactly that Transpose
  consumer, and Softmax to precede Transpose. Duplicate producers and a
  Softmax intermediate exposed at either public boundary are rejected; a
  terminal output cannot also be an input. Rank-four source
  shape/signature, destination existence, and cloned quantization are planned
  before mutation. Commit removes only the marker, uses the lineage-aware
  indexed output setter, copies the former source metadata onto the existing
  public tensor object, and removes the Transpose differentially. Missing
  Softmax/Transpose families retain historical pruning without index
  construction, including optional `LayoutState` pruning at the production
  call site.
- Pre-ArgMax terminal layout cleanup accepts only an exact rank-four
  `[0,3,1,2]` Transpose whose private output has one topologically later
  `ARG_MAX` consumer at data input zero. The signed INT32/INT64 singleton axis
  must normalize to NCHW channel axis one and is remapped to NHWC axis three.
  Source, transposed, and output shape/signature metadata must prove the same
  permutation and rank-reduced output, and source/adapter dtypes must agree.
  A private axis constant is updated in place; a shared or public-input/output
  axis constant receives a uniquely named clone with its NumPy dtype and
  cloned quantization. This preserves the public constant value that the
  former rule could silently change. All clone data and topology guards finish
  before either constant or edge mutation. Indexed ArgMax input replacement
  and Transpose removal preserve the fixed point, historical stats, and
  lineage; post-prune `LayoutState` synchronization registers any clone.
  Missing required families retain historical pruning without index
  construction.
- Quantized MaxPool cleanup accepts only an exact linear
  `DEQUANTIZE -> MAX_POOL_2D -> QUANTIZE` chain. Input and output grids must
  use the same INT8 or UINT8 dtype and exactly equal positive finite scale and
  in-range zero point; approximate equality is not sufficient because a
  quantized MaxPool builtin preserves integer samples and therefore requires
  identical grids. All four tensors must exist, float bridge dtypes must
  agree, and rank-four shapes and signatures must match across each Q/DQ
  boundary. Both bridge tensors are private, uniquely produced, exclusively
  consumed, and topologically ordered. The output cannot also be a graph
  input. All topology, metadata, and cloned-quantization planning completes
  before indexed Pool edge mutation and differential wrapper removal. This
  intentionally fixes former rewrites that accepted near-equal grids, absent
  float metadata, or a float bridge exposed as a public input. Missing
  required operator families retain historical pruning without allocating an
  index, and both production call sites pass the Session `LayoutState`.
- Quantized Logistic cleanup accepts only an exact linear
  `DEQUANTIZE -> LOGISTIC -> QUANTIZE` chain. Input and output tensors must use
  the same INT8 or UINT8 dtype. The input grid requires a positive finite
  scale and in-range zero point; the output grid is exactly scale `1/256` with
  zero point `-128` for INT8 or `0` for UINT8. Tolerant scale equality is not
  sufficient for the builtin's canonical output contract. All four tensors
  must exist, float dtypes must agree, and elementwise shape/signature metadata
  must be identical across the complete chain without imposing a fixed rank.
  Both float bridges are private, uniquely produced, exclusively consumed,
  and topologically ordered; the quantized output cannot also be an input.
  Every guard completes before indexed Logistic edge mutation, version
  selection, and differential wrapper removal. This intentionally fixes
  former rewrites that accepted a near-canonical output scale, missing or
  invalid input quantization, absent float metadata, or a public-input bridge.
  Missing required families retain historical pruning without allocating an
  index, and both production call sites pass the Session `LayoutState`.
- Quantized Softmax cleanup accepts only an exact linear
  `DEQUANTIZE -> SOFTMAX -> QUANTIZE` chain. Its grid requirements match the
  canonical quantized Logistic contract: the input uses a positive finite
  per-tensor INT8/UINT8 grid, while output scale is exactly `1/256` with zero
  point `-128` or `0`. The existing beta tolerance of `1e-6` is preserved, but
  beta must be finite and parseable. Rank must be positive, and an explicit or
  default axis must normalize to the final dimension because the serialized
  TFLite builtin has no independent axis field. All four tensors must exist,
  float dtypes must agree, and shape/signature metadata must be identical
  across the elementwise chain. Both float bridges are private, uniquely
  produced, exclusively consumed, and topologically ordered; the quantized
  output cannot also be an input. Every guard completes before indexed
  Softmax edge mutation, version selection, and differential wrapper removal.
  This intentionally fixes former rewrites that accepted near-canonical
  output scales, missing/invalid input grids, absent float metadata, public-
  input bridges, malformed options, or non-last axes. Missing required
  families retain historical pruning without index allocation, and both
  production call sites pass the Session `LayoutState`.
- Expanded HardSigmoid QDQ cleanup matches the exact linear
  `DEQUANTIZE -> MUL(alpha) -> ADD(beta) -> MAXIMUM(low) -> MINIMUM(high) ->
  QUANTIZE` grammar with either scalar-input position at each binary operator.
  Input and output use exactly the same finite positive per-tensor INT8/UINT8
  grid with an in-range zero point. All seven data tensor records must exist;
  the five float tensors have one dtype and every data tensor has identical
  elementwise shape/signature metadata. Every bridge is private, uniquely
  produced, exclusively consumed, and topologically ordered; the quantized
  output cannot also be an input. Each scalar must be finite, singleton,
  producer-free, and representable within the preserved quarter-scale/`1e-3`
  tolerance.
- The four constant retargets are immutable plans containing the quantized
  value, cloned quantization, ownership choice, metadata, and reserved clone
  name. Private exclusive constants retain in-place behavior. Shared or public
  constants receive deterministic `_q` clones, so a public float scalar no
  longer silently changes dtype/value. Only after all four plans and four
  intermediate quantization clones succeed are constant and data edges
  rewritten and DQ/Q wrappers removed. Fault injection proves that a clone
  failure on the second scalar leaves complete ModelIR unchanged, whereas the
  former helper changed the first scalar data/dtype before raising. Missing
  required families retain historical pruning without index allocation; both
  production call sites pass LayoutState, which is synchronized after clones
  and pruning.
- Quantized TransposeConv cleanup accepts only the exact linear
  `DEQUANTIZE -> TRANSPOSE_CONV -> QUANTIZE` grammar with TFLite input roles
  `[output_shape, filter, data]`. Input and output activations independently
  require finite positive per-tensor INT8 grids and in-range zero points.
  Their quantized/float shape and signature metadata must match at each
  boundary, the bridges must share a floating dtype, and both bridges are
  private, uniquely produced, exclusively consumed, and topologically ordered.
  The quantized output cannot also be a graph input.
- Filter conversion is a pre-mutation plan. A producer-free rank-four INT8
  filter with matching buffer metadata and a valid grid remains unchanged. A
  finite FLOAT16/FLOAT32/FLOAT64 filter is quantized in place only when it is
  private and exclusive; shared or public filters receive a deterministic
  `_q` clone. The filter data/grid/name and output quantization clone all
  complete before indexed input/output mutation. Fault injection proves that
  output-grid clone failure leaves the complete ModelIR unchanged, whereas the
  former helper had already converted a private float filter before raising.
  Missing families retain historical pruning without index allocation; all
  three production call sites pass the Session LayoutState.
- Decomposed InstanceNormalization repair requires the exact marked chain from
  the first Mean through bias Add, including correct Sub and reciprocal-Div
  operand roles, keep-dim reductions, graph order, unique producers, exclusive
  internal consumers, a finite epsilon, and a producer-free scalar one. The
  input logical layout chooses channel axis one or the final axis for ranks
  three through five; an optional post-Transpose must be a complete
  permutation before the bias broadcast axis is derived.
- Both Mean axes, nine intermediate shape/signature records, and scale/bias
  data plus metadata are planned before mutation. Changing a constant requires
  integer axes or a channel-count-sized buffer, no producer/public boundary,
  and exactly the expected Mean/Mul/Add consumers. This preserves the shared
  two-Mean axes tensor while rejecting external sharing. A malformed final
  bias shape is now a complete no-op instead of raising after earlier axes,
  shapes, and scale data were already changed. The final production call passes
  the Session LayoutState; graphs without the marked first Mean allocate no
  index.
- NCHW Concat/global-pool/Conv repair requires the exact ordered
  `CONCATENATION -> MEAN -> RESHAPE -> CONV_2D` chain. Each internal tensor is
  uniquely produced, exclusively consumed by the next operator, and private.
  The keep-dim Mean reduces rank-four axes two and three; negative axes are
  normalized. Fully positive NCHW Concat inputs share batch/spatial dimensions,
  and their axis-one channel sum must equal the producer-free OHWI Conv filter
  input channel.
- Concat and Reshape options, Concat/Mean/Reshape metadata, and the four-value
  integer Reshape buffer are a complete pre-mutation plan. The shape constant
  is producer-free, private, and exclusively consumed by that Reshape. This
  prevents a late buffer-read exception from leaving the Concat axis and three
  tensor records changed, and rejects the former non-global, fan-out/public,
  duplicate-producer, runtime-filter, and malformed/shared shape-buffer cases.
  The sole production call passes the Session LayoutState; incomplete operator
  families allocate no index.
- NCHW Concat/Transpose/(Transpose)Conv repair traces optional shape-preserving
  RELU/RELU6/QUANTIZE/DEQUANTIZE/CAST before the exact `[0,2,3,1]` Transpose
  and optional PAD/CAST/SUB afterward. Every internal output is uniquely
  produced, exclusively consumed by the next topologically ordered operator,
  and private. The permutation and OHWI filter are producer-free constants,
  and the filter buffer exactly matches its rank-four metadata.
- Fully positive NCHW Concat inputs share batch/spatial dimensions, and their
  axis-one channel sum equals the filter input channel. Concat options and all
  Concat/pre-passthrough/Transpose shape records are planned together. Direct
  Conv without a post-prefix also plans its output shape; prefixed Conv and
  TransposeConv intentionally preserve output metadata. This rejects the
  former public/fan-out/duplicate adapter and runtime-filter cases without
  broadening the four existing positive families. The production call passes
  the Session LayoutState; missing Concat or Conv families allocate no index.
- Mixed singleton Concat repair accepts an exact axis-three NHWC Concat only
  when its output channel equals the input count and every same-dtype input is
  either `[N,H,W,1]` or its singleton-channel NCHW projection. Input/output
  shape signatures must express the same contract. One dynamic dimension is
  retained as the sole Reshape `-1`; multiple dynamic dimensions cannot be
  represented by this local adapter and therefore remain untouched.
- A runtime input must be public or uniquely produced before the Concat; a
  producer-free constant is also valid. Duplicate, later, and unresolved
  producers are rejected. Names are reserved across all existing tensors,
  operator edges, and boundaries. Repeated source inputs share one adapter.
  Every shape tensor, adapter tensor, operator, and quantization clone is
  prepared before indexed insertion and lineage-aware rewiring. This prevents
  a late clone exception from leaving the first adapter behind. The production
  call passes the Session LayoutState; graphs without Concat allocate no index.
- Window-partition canonicalization requires exact NHWC input, six-dimensional
  partition Reshape, `[0,1,3,2,4,5]` Transpose, and three-dimensional window
  output Reshape equations. All internal edges are private, uniquely produced,
  exclusively consumed, and ordered. The input is public, constant, or
  uniquely produced earlier. Shape/permutation vectors are producer-free,
  non-input INT32/INT64 constants with exact vector metadata.
- All four data tensors share dtype and either no quantization or one exact
  per-tensor grid. The existing Transpose object becomes SPACE_TO_DEPTH so its
  version, axis semantics, and provenance survive. Static metadata remains
  exact. A consistent dynamic batch/spatial/channel signature is propagated;
  when the retained Reshape needs one `-1`, its private vector and both shape
  options are changed together and marked dynamic. Two inferred dimensions,
  shared/public shape mutation, or any incomplete contract is a complete
  no-op. Both production calls pass the Session LayoutState; incomplete
  operator families prune historically unused tensors without allocating an
  index.
- Recurrent orphan-step alias repair has one Torch-free semantic owner in
  `passes/recurrent_alias.py`. Candidate discovery occurs before index
  construction, so graphs without the exact step-name grammar allocate no
  index. A supplied matching index is reused; otherwise exactly one index is
  built. Producer rejection, shape-tensor consumer order, Reshape arity,
  public input/output, consumer rewrites, and non-public orphan tensor removal
  retain the direct implementation's behavior. The direct and PyTorch modules
  are compatibility wrappers only and do not carry parallel match/rewrite
  rules.
- Unbound nonconstant-input discovery and layout repair have one Torch/
  TensorFlow-free owner in `passes/unbound_input_layout.py`. Standalone issue
  reporting retains its lightweight producer-name scan. Repair first snapshots
  issue consumers by object identity; it constructs one `ModelIRGraphIndex`
  only when an issue exists, resolves current positions after each insertion,
  and skips later issues once an earlier bridge produces the same tensor.
  DEQUANTIZE exact/fallback source policy, nearest source ordering, SPLIT data
  slot, MUL all-consumers guard, dtype/shape checks, quantization metadata,
  unique perm naming, and insertion-before-consumer order remain unchanged.
  The lowerer wrapper reconciles shapes with the maintained index and preserves
  the existing stats key for both primary and fallback callers.
- Quantized RELU/RELU6 layout-bridge cleanup has one Torch/TensorFlow-free
  owner in `passes/quantized_activation.py`. It skips index allocation when no
  Transpose exists, otherwise uses one current index for exact chain traversal,
  DQ input and Q output rewrites, and batch removal of the inverse Transposes.
  The restart loop remains intentional: removing a later bridge may make an
  earlier graph-order candidate linear, while current indexed candidates avoid
  every compatibility-map rebuild. Public intermediate/source guards, exact
  inverse permutations, per-tensor-only quantization, source-shape propagation,
  destination dtype/quantization cloning, pruning, and stats are unchanged.
- Expanded HardSigmoid QDQ layout-bridge cleanup now shares that owner and
  recognizes both RELU_0_TO_1 and MAXIMUM/MINIMUM clamp forms. One current
  index supplies all linear-consumer and shared-constant decisions, indexed
  setters maintain cloned-constant and DQ/Q edges, and both Transposes are
  removed by one differential compaction. All required constants are
  validated before any mutation, making rejection transactional. Private
  rank-matched constants retain in-place remapping; shared constants retain
  clone-and-rewire behavior. Public guards now include the clamp form's
  MAXIMUM intermediate as well as every former boundary, while inverse-perm,
  per-tensor quantization, shape/signature, destination metadata, lineage,
  pruning, and stats contracts remain intact.
- Expanded MUL/ADD/PRELU QDQ bridge cleanup now has the same single owner.
  MUL/ADD input ordering and PRELU data-slot semantics are unchanged; one
  index supplies all topology and shared-constant decisions, both DQ/Q edge
  updates, and removal of the two wrapper Transposes. The common constant
  planner is mutation-free and the common applier owns private updates and
  shared clone rewires, but PReLU still requires all three buffers to be NumPy
  arrays while HardSigmoid accepts any non-`None` buffer. This prevents a
  convenience helper from broadening rule eligibility.
- Quantized logistic-gated MUL recovery has a separate
  `passes/quantized_gate.py` owner because its shared input, dual DQ/Q
  branches, and multi-post aliases are not a linear activation contract. One
  index supplies producer and consumer topology, both DQ rewires, canonical
  output selection, indexed alias replacement, and one compaction for the
  pre-Transpose plus every post-Transpose. The first graph-order post remains
  canonical and receives the permuted MUL-Q metadata. All internal data/gate
  tensors are now protected when publicly observable; fixed permutations,
  per-tensor quantization, pruning, lineage, and stats remain unchanged. A
  bounded `_match_logistic_gate_branch` helper isolates backward gate matching
  while preserving incomplete-chain fallback and duplicate-branch rejection.
- Shared parsers preserve the exact old generated syntax when broadening would
  change rule eligibility. Parser ownership tests prevent duplicate exporter
  implementations and unused compatibility imports.
- No real-model conversion gate is required for these mechanical checkpoints
  under the current instruction to prioritize implementation and minimize
  conversion tests. This does not prove broad corpus regression safety.

## Tests executed

The resumed downstream-binary, Resize-evidence, aligned-BatchNorm, LRN,
static-shape-cache, and NHWC-bridge checkpoints passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pytorch_fast_precanonicalize_policy.py \
  tests/test_flatbuffer_direct_architecture.py

124 passed
```

Seven pre/post-extraction characterization cases also preserve the exact
orchestrator output for matching BN, direct return, channel-first Resize,
channel mismatch, channel-last Resize, mixed operands, and an already-CF shape.
Five additional pre/post-extraction cases preserve direct BN, reshaped BN,
no-BN fallback, NHWC-input no-op, and already-CF Resize behavior.
Seven aligned-BatchNorm cases preserve direct rewrite, non-BN no-op, channel
mismatch no-op, NHWC-input no-op, reshaped rewrite, already-CF behavior, and
reshaped non-BN no-op.
The LRN checkpoint additionally passed a four-test selection covering CF/NHWC
and static-shape state, Pool/LRN interaction, architecture ownership, and the
existing generated-source integration case.
The cache checkpoint passed four focused cases covering aligned binary,
Resize/Pool, literal recording, parse-failure no-op, and architecture ownership.
The bridge checkpoint adds positive state-set/cache assertions and a no-op
state-preservation case to the existing whole-chain normalization test.
The request-boundary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_architecture.py

126 passed
```

Its AST gate proves zero `kwargs.get` calls and exactly one raw `kwargs` read,
as the argument to `ConversionRequest.from_kwargs`.
The requested-exporter checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_artifact_preparation.py \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_architecture.py

145 passed
```

Dedicated resolver tests use a mapping that raises on every `get` to prove
unrequested settings are untouched, then verify requested and calibration-only
values and timeout coercion.
The same 145-test selection passed after adding requested-only quant type/dtype
resolution. Focused assertions cover explicit values, the three legacy
defaults, immutable mapping behavior, and absence of direct `request.get`
calls for those keys.
The Session consumer-count checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_optimization \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_fanout_optimization \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_preserves_dynamic_batch_signature

32 passed
```

The core spy fixture uses one input in both Add and Identity and verifies the
context receives counts `{"x": 2, "y": 1}` from the Session index.
The lowering-time layout checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_pad_edge_lowering.py \
  tests/test_flatbuffer_direct_resize_integer_linear.py \
  tests/test_flatbuffer_direct_architecture.py::test_op_builders_mutate_layout_only_through_lowering_context

33 passed
```

The rank-three Resize regression hook inspects `LayoutState` before the first
post-lowering pass and verifies that the NWC/NHWC adapter tensors have no
ModelIR mismatch. The focused edge-Pad and integer-linear Resize tests also
serialize and execute their TFLite artifacts sequentially. The architecture
gate rejects future direct logical/physical layout assignments anywhere below
`op_builders`.
The synthetic-consumer checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_elides_inverse_transpose_chain_at_generation \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_no_dead_operator_outputs_after_prune \
  tests/test_flatbuffer_direct_architecture.py::test_op_builders_mutate_operator_list_only_through_lowering_context

35 passed
```

The two direct context cases prove both sides of the contract: exclusive
inverse Transposes remove their operator and differential indexes, while a
synthetic bridge already consumed by an Identity retains its producer. The
architecture gate rejects direct operator-list writes and mutating method calls
from op builders.
The adjacent pass-state reuse checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_mean_layout.py \
  tests/test_flatbuffer_direct_layernorm_layout.py \
  tests/test_flatbuffer_direct_terminal_mean_layout.py \
  tests/test_flatbuffer_direct_se_layout.py \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_pass_efficiency.py \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_mean_attention_cluster_reuses_one_pass_state_scope

65 passed
```

The focused reuse case observes one `ModelIRGraphIndex.refresh()` across two
candidate Mean runners and a diagnostic build sequence of `[true, false]`.
The all-preflight-miss case observes zero index refreshes and `[false, false]`.
The architecture gate fixes all seven runner calls, their order, their shared
scope keyword, and the six production cluster invocations.
The adjacent gate-pass reuse checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_elementwise_gate_layout.py \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_flatbuffer_direct_dual_postconv_gate_layout.py \
  tests/test_flatbuffer_direct_3d_gate_layout.py \
  tests/test_flatbuffer_direct_conv3d_gate_layout.py \
  tests/test_flatbuffer_direct_cost_volume_scatter_layout.py \
  tests/test_flatbuffer_direct_add_concat_suffix_layout.py \
  tests/test_flatbuffer_direct_dual_mul_concat_layout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_mixed_mean_reducemax_concat_mirrorpad_nhwc_chain \
  tests/test_flatbuffer_direct_pass_efficiency.py \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_gate_cluster_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

116 passed
```

The synthetic shared-scope fixture makes every runner's model-only preflight
match but contains no deep rewrite candidate. It records one
`ModelIRGraphIndex.refresh()` across all eight calls and 15 diagnostic events:
the first reports `state_built: true`, and every later event reports `false`.
The architecture checks fix the eight-runner order, five production helper
invocations, and the single invocation that omits mixed attention. They also
bring the global direct-runner count characterization up to date with both
bounded helper extractions.
A focused late-pair checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_ndhwc_cost_volume_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_ndhwc_cost_volume_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_3d_gate_layout.py \
  tests/test_flatbuffer_direct_conv3d_gate_layout.py \
  tests/test_flatbuffer_direct_cost_volume_scatter_layout.py

50 passed
```

The runtime characterization observes one graph-index refresh and diagnostic
build flags `[true, true, false]`: both events in the first two-spec runner
belong to the one state-building group, and the second runner reuses that
state. The architecture test fixes both raw boundaries and proves that mixed
attention receives no shared scope.
A focused late-Concat cluster checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_concat_layout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_concat_layout_cluster_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_axis3_const_concat_layout.py \
  tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py \
  tests/test_flatbuffer_direct_layernorm_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py

74 passed
```

The synthetic runtime fixture records one graph-index refresh and build flags
`[true, false, false, false, false]`. The architecture test fixes the four-
runner order and both raw boundaries. The core layout-handoff monkeypatch now
accepts and forwards the runner's optional scope without changing its original
pre-pass layout assertions.
A focused shuffle/unary cluster checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_shuffle_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_nhwc_channel_shuffle.py \
  tests/test_flatbuffer_direct_nchw_channel_shuffle.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_transpose_unary.py \
  tests/test_flatbuffer_direct_transpose_unary_fanout.py \
  tests/test_flatbuffer_direct_transpose_unary_binary_fanout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_shufflenet_transpose_shuffle_chain_optimized

29 passed
```

Both synthetic fixtures observe exactly one graph-index refresh. The seven-
runner fixture records one `state_built: true` followed by six `false` events;
the three-runner fixture records one `true` followed by two `false` events.
The architecture gate fixes the two helper orders, five and four invocations,
and the single extended channel-shuffle invocation.
A focused boundary-input pair checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_boundary_batchmatmul_unary_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_boundary_batchmatmul_unary_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_boundary_input_chains.py \
  tests/test_flatbuffer_direct_input_passthrough_layout.py

17 passed
```

The runtime fixture records one graph-index refresh and build flags
`[true, false, false, false]`; the latter three events belong to the reused
three-spec input-unary group. The architecture gate fixes the two-runner order
and all four helper invocations.
A focused channel-slice/Pad-Mul checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_channel_slice_pad_mul_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_channel_slice_pad_mul_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_tflite_builder_direct.py \
  -k 'channel_slice or transpose_pad_mul_posttranspose or channel_slice_pad_mul_pair or ordered_model_ir_runner or lowerer_channel_slice'

8 passed, 754 deselected
```

The runtime fixture records one graph-index refresh and build flags
`[true, true, true, false]`: the first three events belong to the one state-
building channel-slice group, and Pad-Mul reuses that state. The architecture
gate fixes the two-group order and all three helper invocations.
A focused singleton/Reshape checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_singleton_channel_transpose.py \
  tests/test_flatbuffer_direct_singleton_reshape.py \
  tests/test_flatbuffer_direct_singleton_maxpool.py \
  tests/test_flatbuffer_direct_flatten_concat_reshape.py \
  tests/test_flatbuffer_direct_consecutive_reshape.py \
  tests/test_flatbuffer_direct_singleton_spatial_reshape.py \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_osnet_gate_layout.py \
  <two new efficiency tests and two architecture checks>

41 passed
```

The long synthetic fixture makes all ten runner preflights match and records
13 diagnostic events with one graph-index refresh and build flags
`[true, false, ...]`. The short fixture records one refresh and flags
`[true, false, false]`. Architecture checks fix both long variants, all three
short-helper target/layout combinations, the shared scope keyword, and the
147-call global runner characterization.
A focused QKV attention checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  <seven focused QKV rewrite tests from test_tflite_builder_direct.py> \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_qkv_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_qkv_attention_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

10 passed
```

The preflight-only fixture records one graph-index refresh and build flags
`[true, true, true, true, false, false]`: the four prefix events belong to the
state-building group, and both bridge events reuse it. The seven existing
functional cases cover Gather/Reshape/Transpose hoisting, Gather-to-Slice,
Slice-to-Split, Split/Reshape collapse, shared pre-Transpose, weighted-sum
bridging, and the KV pipeline.
A focused duplicate/PReLU checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_quantized_prelu.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_duplicate_quantized_prelu_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_duplicate_quantized_prelu_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

28 passed
```

The QDQ preflight-only fixture disables duplicate-Transpose cleanup exactly as
production does, then records one graph-index refresh and build flags
`[true, false, false, false, false]` across the reshape-duplicate group and
four PReLU specs. Architecture checks fix the helper's flag forwarding, exact
runner order, two invocations, and the 143-call global characterization.
A focused constant-fold/Cast checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_constant_fold.py \
  tests/test_flatbuffer_direct_cast_cleanup.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_constant_fold_cast_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_constant_fold_cast_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

16 passed
```

The preflight-only widening-Cast fixture records one graph-index refresh and
build flags `[true, true, true, false, false]`: the three constant-fold events
belong to the state-building group, and both Cast-cleanup events reuse it.
Architecture checks fix runner order, both helper invocations, and the 141-call
global characterization.
A focused SE-FC/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_se_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_channel_fanout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_se_fc_gather_channel_fanout_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_se_fc_gather_fanout_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

14 passed
```

The preflight-only fixture records one graph-index refresh and build flags
`[true, false]`. Architecture checks fix both target/LayoutState combinations,
runner order, and shared scope keywords. The global characterization was 139
calls at that checkpoint, 137 after the later post-QDQ consolidation, and 135
after the subsequent NCHW channel-shuffle consolidation, then 134 after the
conditional QKV-bridge consolidation.
A focused terminal-boundary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_boundary_input_layout.py \
  tests/test_flatbuffer_direct_dual_mul_concat_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_channel_fanout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_boundary_layout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_boundary_layout_cluster_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

38 passed
```

The preflight-only fixture records one graph-index refresh across seven events;
only the first reports `state_built: true`. Architecture checks fix all five
runner calls, their shared scope, the preceding raw InstanceNorm rewrite, and
the following conditional stage.
A focused late-Dequantize/unary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_dequant_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_dequant_unary_fanout_cluster_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

41 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks fix all three
runner calls, their shared scope, the preceding raw QDQ bridge, and the
following raw swish rewrite.
A focused terminal singleton-MaxPool/Reshape checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_singleton_maxpool.py \
  tests/test_flatbuffer_direct_consecutive_reshape.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_singleton_maxpool_reshape_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_singleton_maxpool_reshape_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

8 passed
```

The preflight-only fixture records one graph-index refresh across three events;
the two singleton-MaxPool events report `state_built: true` because they share
the first registered group, and the Reshape event reports `false`.
Architecture checks fix both runner calls, their shared scope, and the exact
conditional legacy-rewrite boundaries.
A focused terminal clamp/unary/ReLU checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_clamp_unary_relu_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_clamp_unary_relu_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

27 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. The two real rewrite tests also
assert that the reused graph index removes `MAXIMUM`/`MINIMUM` type entries and
adds the correct `RELU_0_TO_1` or `RELU` entry without refreshing.
A focused late Mean/SPP/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_mean_layout.py \
  tests/test_flatbuffer_direct_spp_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_mean_spp_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_mean_spp_gather_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

51 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks fix the runner
order, shared scope keywords, conditional generic-transpose predecessor, and
constant-fold/Cast helper successor.
A focused late SPP/Concat-unary-Conv checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_spp_layout.py \
  tests/test_flatbuffer_direct_concat_unary_conv_layout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_spp_concat_unary_conv_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_spp_concat_unary_conv_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

73 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. Architecture
checks fix both runner calls, shared scope keywords, and both raw rewrite
boundaries.
A focused absolute-final normalization-Pad/attention checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_instancenorm_mirror_pad_prepost_nhwc_chain_optimized \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_flatten_globalnorm_pad_prepost_nhwc_chain_optimized \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_mixed_mean_reducemax_concat_mirrorpad_nhwc_chain \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_absolute_final_normalization_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_absolute_final_normalization_attention_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

6 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. The focused
selection also executes real InstanceNorm, flattened normalization-Pad, and
mixed-attention rewrites. Architecture checks preserve both include flags and
the exact raw boundaries.
A focused post-QDQ unary-fan-out checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_post_qdq_layout_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_post_qdq_unary_fanout_cluster_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

9 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks preserve four
default helper invocations, require exactly one alternate-mode invocation,
fix both raw boundaries, and characterize 137 registered runner calls.
A focused late NCHW channel-shuffle/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_nhwc_channel_shuffle.py \
  tests/test_flatbuffer_direct_nchw_channel_shuffle.py \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_shuffle_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_nchw_shuffle_gather_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_nchw_shuffle_gather_pair_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

15 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. Real NHWC and
NCHW rewrite tests assert that `RESHAPE` type entries disappear and the
surviving `GATHER` entry has its correct shifted index without refreshing.
Architecture checks preserve five default helper invocations, require one
NCHW-only invocation, fix both raw boundaries, and characterize 135 runner
calls.
A focused conditional layout/QKV-bridge checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_attention_qkv_shared_pretranspose_slice_nchw \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_attention_qkv_weighted_sum_bridge_to_nhwc \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_qkv_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_layout_qkv_bridge_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_qkv_attention_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_layout_qkv_bridge_pair_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

11 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the generic-transpose event reports `state_built: true`. Architecture
checks preserve two default helper invocations, require one bridge-only
invocation with the runtime layout flag, fix both raw boundaries, and
characterize 134 runner calls.
A focused very-late Gather/constant/normalization checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_constant_fold.py \
  tests/test_flatbuffer_direct_cast_cleanup.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_flatten_globalnorm_pad_prepost_nhwc_chain_optimized \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_constant_fold_cast_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_very_late_gather_constant_normalization_cluster_reuses_one_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_constant_fold_cast_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_very_late_gather_constant_normalization_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

21 passed
```

The preflight-only fixture records one graph-index refresh across seven events;
only the Gather-axis event reports `state_built: true`. Architecture checks
require both constant-fold/Cast invocations to receive an external scope,
preserve the normalization flags, and fix both raw boundaries.
A focused terminal hard-activation/layout scope checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_hard_activation_layout_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_hard_activation_layout_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

3 passed
```

The preflight-only fixture records one graph-index refresh across three events;
both HardSigmoid events report `state_built: true` for their shared group and
the following generic-Transpose event reports `false`. Architecture checks
preserve all four late hard-activation flags, the conditional layout switch,
and both raw rewrite boundaries. The related real hard-activation and generic-
Transpose selection passed separately with `7 passed`.
A focused late layout/Mean/SPP/Gather/constant/Cast scope checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_layout_mean_spp_gather_constant_cast_cluster_reuses_one_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_layout_mean_spp_gather_constant_cast_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

3 passed
```

The preflight-only fixture records one graph-index refresh across nine events;
only the conditional generic-Transpose group reports `state_built: true`.
Architecture checks preserve the full runner/helper order, the runtime layout
flag, both external constant-fold/Cast scopes, and the raw boundaries. The six
related real pass modules passed separately with `65 passed`.
A broader single-process selection of
`test_flatbuffer_direct_core.py`, `test_flatbuffer_direct_pass_efficiency.py`,
and the complete `test_flatbuffer_direct_architecture.py` passed with
`186 passed` after adding the late combined-scope checks.

The typed artifact-plan checkpoint passed its policy selection:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q tests/test_flatbuffer_direct_artifact_preparation.py

20 passed
```

Default, dynamic-range/integer-quantized, split, SavedModel, PyTorch-derived,
and report progress labels are characterized from `ArtifactPlan`. Rejecting
option mappings prove that an unrequested artifact reads no related option or
environment value. Sequential dynamic-range, strict-integer, and split-
manifest direct-backend smokes passed with `3 passed`. The combined artifact-
policy, core, pass-efficiency, and architecture selection passed with
`206 passed`.

The evaluation-artifact selection checkpoint passed its focused unit and
ownership selection with `4 passed`. Sequential dynamic-range, strict-integer,
and split-manifest direct-backend smokes again passed with `3 passed`. The
combined artifact-metadata, artifact-policy, core, pass-efficiency, and
architecture selection passed with `210 passed`.

The report/quantized-artifact finalization checkpoint passed its focused
single-owner structure test with `1 passed`. The sequential integration
quantization/evaluation/coverage, strict integer/int16, and split-manifest
direct-backend smokes passed with `3 passed`. The combined artifact-metadata,
artifact-policy, core, pass-efficiency, and architecture selection passed with
`211 passed`.

The terminal direct-boundary checkpoint passed the three focused evaluation-
selection, report/quantized-finalization, and fast-path control-flow contracts
with `3 passed`. TensorFlow-import-blocked direct and `-cotof`, followed by the
sequential integration quantization/evaluation/coverage, strict integer/int16,
and split-manifest smokes, passed with `5 passed`. The combined artifact-
metadata, artifact-policy, core, pass-efficiency, and architecture selection
passed with `212 passed`.

The ordered layout-recovery-prefix checkpoint passed its focused ordering,
runner-ownership, SPP, NDHWC Concat, and NHWC/NCHW channel-shuffle selection
with `68 passed`. A sequential quantization/evaluation/coverage integration
smoke passed with `1 passed`. The complete architecture selection passed with
`128 passed`; the combined artifact-metadata, artifact-policy, core, pass-
efficiency, and architecture selection passed with `213 passed`.

The attention/quantized-recovery-suffix checkpoint passed focused ordering,
scope-boundary, quantized PReLU, quantized Reshape, and trailing-output-
Transpose tests with `19 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`. The
combined artifact-metadata, artifact-policy, core, pass-efficiency, and
architecture selection passed with `214 passed`.

The layout/reshape/attention-recovery-prefix checkpoint passed its focused
owner, exact-order, successor-boundary, and runner-diagnostics selection with
`4 passed`. The complete architecture file passed with `130 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for the same combined selection total of `215 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The terminal slice/Concat layout-recovery checkpoint passed focused QKV,
channel-slice, exact-order, variant-boundary, runner-diagnostics, and layout-
Transpose ownership checks with `6 passed`. The complete architecture file
passed with `131 passed`; artifact-metadata, artifact-policy, core, and pass-
efficiency passed separately with `85 passed`, for a combined selection total
of `216 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The terminal affine/Concat/split recovery checkpoint passed focused exact-
order, raw-boundary, terminal slice/Concat, QKV-boundary, and runner-
diagnostics checks with `4 passed`. The complete architecture file passed with
`132 passed`; artifact-metadata, artifact-policy, core, and pass-efficiency
passed separately with `85 passed`, for a combined selection total of `217 passed`.
Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The attention/gate/QDQ recovery checkpoint passed focused exact-order, three-
boundary, outer-suffix, gate/unary-scope, and runner-diagnostics checks with
`5 passed`. The complete architecture file passed with `133 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for a combined selection total of `218 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The quantized-activation/binary-bridge recovery checkpoint passed its focused
exact-order and two-boundary selection with `4 passed`. The adapted post-QDQ
boundary selector and new owner passed together with `2 passed`. The complete
architecture file passed with `134 passed`; artifact-metadata, artifact-policy,
core, and pass-efficiency passed separately with `85 passed`, for a combined
selection total of `219 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The SiNet terminal layout-recovery checkpoint passed focused exact-order,
shape-boundary, terminal affine/slice, and runner-diagnostics checks with
`4 passed`. Its adapted terminal-clamp boundary and new owner passed together
with `2 passed`. The complete architecture file passed with `135 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for a combined selection total of `220 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The pre-Add/Mean attention-recovery checkpoint passed focused exact-order,
two-boundary, attention/QDQ composition, Mean-cluster scope, layout-prefix, and
runner-diagnostics checks with `5 passed`. The complete architecture file
passed with `136 passed`; artifact-metadata, artifact-policy, core, and pass-
efficiency passed separately with `85 passed`, for a combined selection total
of `221 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The SiNet pre-Add/Resize recovery checkpoint passed focused exact-order, four-
boundary, terminal-helper composition, terminal-clamp, and runner-diagnostics
checks with `4 passed`. Recursive helper expansion matches the preceding
lowerer AST exactly. The complete architecture file passed with `137 passed`;
artifact-metadata, artifact-policy, core, and pass-efficiency passed separately
with `85 passed`, for a combined selection total of `222 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The safe-binary and QLinear/Mean/Concat recovery checkpoint passed focused
exact-order, nested-helper composition, condition/progress/layout boundaries,
post-QDQ ownership, and runner-diagnostics checks with `6 passed`. Recursive
helper expansion matches the preceding lowerer AST exactly. The complete
architecture file passed with `139 passed`; artifact-metadata, artifact-policy,
core, and pass-efficiency passed separately with `85 passed`, for a combined
selection total of `224 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The indexed shape-convergence checkpoint passed its focused dynamic-Reshape,
shape-reconciliation, legacy-equivalence, single-index-build, and ownership
selection with `13 passed`. The complete architecture file passed with
`140 passed`; artifact-metadata, artifact-policy, core, and pass-efficiency
passed separately with `85 passed`, for a combined selection total of
`225 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The final indexed shape/activation convergence checkpoint passed its focused
legacy-equivalence, single-index-build, no-consumer-rescan, differential-index,
shape, and ownership selection with `18 passed`. The complete architecture
file passed with `141 passed`; artifact-metadata, artifact-policy, core, pass-
efficiency, and the two new convergence cases passed separately with
`87 passed`, for a combined selection total of `228 passed`. Existing Conv,
DepthwiseConv, Add, Sub, Mul, and Div activation-fusion coverage passed with
`20 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The indexed broadcast-constant repair checkpoint passed four focused
shared-constant, no-op, inverse-rotation, and ownership cases plus four existing
rank-three/rank-four repair characterizations (`8 passed`). The complete
architecture file passed with `142 passed`; artifact-metadata, artifact-policy,
core, pass-efficiency, indexed final-convergence, and binary-layout coverage
passed separately with `90 passed`, for a combined selection total of
`232 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The indexed binary-layout convergence checkpoint passed six focused multi-
match, reversed-input, peer-producer, channelwise-constant, fan-out, exact-
legacy, one-build, and ownership checks. The complete related Conv/layout and
indexed-convergence selection passed with `15 passed`. The complete
architecture file passed with `143 passed`; artifact-metadata, artifact-policy,
core, pass-efficiency, indexed final-convergence, binary-layout, Conv-layout,
and indexed binary-convergence coverage passed separately with `105 passed`,
for a combined selection total of `248 passed`. Its single sequential
quantization, evaluation, and coverage integration smoke passed with
`1 passed`.

The indexed Conv-input adapter checkpoint passed its exact former-pair
equivalence, two-match stale-Transpose, one-index-build, no-map-rebuild,
fan-out, graph-output, filter-channel, and ownership coverage. The complete
related Conv/layout and indexed-convergence selection passed with `18 passed`.
The complete architecture file passed with `144 passed`; artifact-metadata,
artifact-policy, core, pass-efficiency, indexed final-convergence, binary-
layout, Conv-layout, indexed binary-convergence, and indexed Conv-input repair
coverage passed with `108 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The indexed wrong-way Conv-Transpose sanitizer checkpoint passed its exact
former-implementation equivalence, two-match removal, multi-Conv consumer,
one-index-build, no-consumer-rescan, non-Conv fan-out, filter-channel, graph-
output, maintained-index, and ownership coverage with `3 passed`. The complete
related Conv/layout and indexed-convergence selection passed with `20 passed`.
The complete architecture file passed with `145 passed`; artifact-metadata,
artifact-policy, core, pass-efficiency, indexed final-convergence, binary-
layout, Conv-layout, indexed binary-convergence, indexed Conv-input repair, and
wrong-way sanitizer coverage passed with `110 passed`. Its single sequential
quantization, evaluation, and coverage integration smoke passed with
`1 passed`.

The shared indexed recurrent-alias checkpoint passed direct legacy-equivalence,
three-alias repair, first-Reshape ordering, public input/output, produced,
missing-shape, invalid-grammar, no-consumer, no-candidate/no-index, maintained-
index, direct/PyTorch wrapper equality, and ownership coverage with `8 passed`.
The complete recurrent, PyTorch normalization, recurrent-codegen-policy, and
new shared-owner selection passed with `18 passed`. The complete architecture
file passed with `146 passed`; the lightweight core/indexed selection passed
with `113 passed`. The real PyTorch exporter normalization regression passed
with `1 passed`, and the single sequential direct quantization, evaluation, and
coverage integration smoke passed with `1 passed`.

The indexed unbound-input layout checkpoint passed exact issue-report and
former-implementation equivalence across DEQUANTIZE, SHAPE, RESHAPE, SPLIT,
and two-consumer MUL-alias repair, plus nearest-source, quantization/signature,
mixed-fan-out guard, one-index-build, no-issue/no-index, maintained-index, and
ownership coverage plus nearest DEQUANTIZE fallback and strict exact-source
preference with `8 passed`. Its complete related QLinear/layout selection
passed with `9 passed`. The complete architecture file passed with
`147 passed`; the lightweight core/indexed selection passed with `117 passed`.
An actual GRU lowering/unbound-input check passed with `1 passed`, and the
single sequential direct quantization, evaluation, and coverage integration
smoke passed with `1 passed`.

The indexed quantized-activation checkpoint passed complete former-mutation
equivalence for two RELU/RELU6 chains, one-index-build, no-consumer-rescan,
maintained-index, public intermediate/source, fan-out, per-channel
quantization, non-inverse permutation, no-Transpose/no-index, ownership, and
real ONNX lowering coverage with `10 passed`. The complete architecture file
passed with `148 passed`; the lightweight core/indexed selection passed with
`125 passed`. Its single sequential direct quantization, evaluation, and
coverage integration smoke passed with `1 passed`.

The indexed expanded-HardSigmoid checkpoint passed exact valid-result
equivalence for RELU_0_TO_1 and MAXIMUM/MINIMUM forms in one graph, one-index
construction, no legacy consumer-map rebuild, maintained-index equivalence,
private and shared rank-four constant remapping, scalar constant preservation,
public add/clamp intermediates and source, fan-out, per-channel quantization,
non-inverse permutation, transactional missing-late-constant rejection, and
no-Transpose/no-index coverage. The complete architecture plus both indexed
quantized-activation files passed with `167 passed`; the lightweight core/
indexed selection passed with `112 passed`. Three real ONNX lowering checks,
TensorFlow-import-blocked direct and `-cotof`, and the sequential direct
quantization/evaluation/coverage integration smoke passed together with
`6 passed`.

The indexed expanded-PReLU checkpoint passed exact former-result equivalence
for two chains, one-index construction, no legacy consumer-map rebuild,
maintained-index equivalence, reversed MUL/ADD input order, private rank-four
MUL/alpha remapping, shared rank-four ADD cloning with quantization metadata,
scalar preservation, public intermediate/PRELU/source boundaries, fan-out,
per-channel quantization, non-inverse permutation, non-array and missing-alpha
rejection, and no-Transpose/no-index coverage. Complete architecture plus all
three indexed quantized-activation files passed with `178 passed`; the
lightweight core/indexed and related quantized-PReLU selection passed with
`128 passed`. Related real ONNX lowering, TensorFlow-import-blocked direct and
`-cotof`, and the sequential quantization/evaluation/coverage integration
smoke passed together with `7 passed`. An attempted exact public-output ONNX
fixture was not retained because the existing ordered trailing-output cleanup
correctly removes its post-Transpose before the specialized owner boundary.

The indexed quantized-logistic-gate checkpoint passed complete former-result
equivalence for simultaneous single- and multi-post chains, one-index
construction, no producer/consumer-map rebuild, maintained-index equivalence,
MUL input reversal, graph-order canonical output selection, alias consumer
consolidation, dtype/shape/signature/quantization propagation, public internal
data/gate/source/alias boundaries, pre/data/gate fan-out, non-Transpose post
users, per-channel quantization, wrong post permutation, and
no-Transpose/no-index coverage. Complete architecture plus all four indexed
quantized suites passed with `195 passed`; the lightweight core/indexed,
related quantized-PReLU, legacy logistic-gate characterization selection
passed with `145 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential direct integration selection passed with `7 passed`.

The single-owner wrong-way Conv-input safety-valve checkpoint preserves the
pre-extraction Swish-only ModelIR digest
`9b47e7f2e879895af600f66c6ac6929acc25580cfea8d5620fca9a6319ee4343` and
its two-removal Swish statistics. Focused exact legacy, multi-Conv, maintained-
index, public-output, mixed-fan-out, filter-channel, no-Transpose/no-index,
Swish-delegation, both existing Swish variants, and ownership coverage passed
with `7 passed`. The complete architecture, lightweight core/pass-efficiency,
focused sanitizer, and both Swish characterizations passed together with
`218 passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage integration smoke passed with `3 passed`.
No Tier corpus conversion was run.

The indexed primary Swish-QDQ branch checkpoint preserves complete former-
phase ModelIR and result equality across shared multi-branch, explicit concat-
closure, spatial guard, public intermediate, and data-fan-out fixtures. The
comprehensive existing fixture retains phase digest
`529b9889fafe9982ebb37ca63687b9329fa11a837562c154480c1856bbc05760`,
three rewritten branches, two removed pre-Transposes, and twenty rewritten
tensors. Focused shared quantized/float tails, one-index, maintained-index,
public/post-output/fan-out guards, small-spatial closure, no-Transpose/no-index,
both legacy Swish variants, and ownership coverage passed with `8 passed`.
Complete architecture, lightweight core/pass-efficiency, both indexed Swish/
Conv-safety suites, and the two legacy Swish characterizations passed together
with `224 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Swish-QDQ metadata checkpoint preserves complete prior-phase
ModelIR/result equality for reverse-ordered fixed-point, public-output,
Pool-channel-mismatch, and wrong-tail fixtures. The comprehensive fixture
retains post-metadata digest
`bab34e6351ec24bc564b9f95b4550bbfaca867f15906f9d77b92f7e8adf1d804`,
one rewritten Concat axis, and twenty-four rewritten tensors. Focused unary,
binary broadcast/signature, Pool/Resize, strict Concat tail, fixed-point,
family guards, empty-seed/no-index, shared primary-index, shared late-shape
owner, both legacy Swish variants, and ownership coverage passed with
`13 passed`. Complete architecture, lightweight core/pass-efficiency, both
indexed Swish/Conv-safety suites, and legacy Swish characterizations passed
together with `229 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage integration smoke passed with `3 passed`. No Tier corpus
conversion was run.

The indexed Swish-QDQ inverse-post checkpoint first proves the two former
lowerer loops have identical ASTs, then preserves complete former-loop ModelIR
and the three-removal result for chained aliases, multi-consumer fan-out,
public alias output, wrong permutation, and untracked input. Focused exact-
legacy, maintained-index, one-index, empty-seed/no-index, two-call ownership,
all existing indexed Swish cases, and both legacy Swish variants passed with
`16 passed`. Complete architecture, lightweight core/pass-efficiency, both
indexed Swish/Conv-safety suites, and legacy Swish characterizations passed
together with `232 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Swish-QDQ late-Concat checkpoint compiles the exact prior committed
late-loop AST and preserves its complete ModelIR, rewritten tensor set, one
axis rewrite, and two input-adapter removals on the mixed direct/DQ fixture.
Focused characterization covers maintained-index equivalence, mixed-input
rewiring, one shared late index, complete post-tail removal, retained direct
fan-out, public source and Concat outputs, missing tensors, mismatched shapes,
wrong tail permutation, transactional no-op behavior, and the missing-required-
type/no-index preflight. The complete indexed Swish and architecture selection
passed with `169 passed`. Architecture, core, pass-efficiency, indexed Swish,
wrong-way Conv safety, and both legacy Swish characterizations passed together
with `237 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Concat pre-Q/DQ checkpoint compiles the complete prior committed
function AST and preserves exact ModelIR and statistics for both one and two
simultaneous matches. Focused characterization covers two-match fixed-point
rewriting with one index construction, maintained-index equivalence, scale and
dtype mismatch, quantized fan-out, public quantized/dequantized boundaries,
shape mismatch, non-Dequantize provenance, rounding-preserving arithmetic
rejection, exact-grid acceptance, and no-Concat/no-index pruning. The focused
owner and legacy selection passed with `26 passed`. Architecture, core, pass-
efficiency, quantization cleanup, and the two established Concat-Q/DQ tests
passed together with `238 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage integration smoke
passed with `3 passed`. No Tier corpus conversion was run.

The indexed terminal Transpose/Dequantize checkpoint compiles the complete
prior committed function AST and preserves exact ModelIR and both statistics
for Transpose-to-Dequantize and Dequantize-to-Transpose forms with one and two
simultaneous matches. Focused characterization covers a single index build
across both subphases, maintained-index equivalence, output-name and metadata
preservation, terminal/public/consumer boundaries, shared Transpose output,
per-channel quantization, invalid permutation, missing tensor, required-type/
no-index pruning, ownership, and the established real ONNX lowering case. The
focused selection passed with `35 passed`. Architecture, core, pass-efficiency,
quantization cleanup, real terminal sanitation, and both Concat-Q/DQ
characterizations passed together with `249 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. No Tier corpus conversion was run.

The indexed Transpose-DQ-Mean-Q checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR and statistics for one
and two simultaneous matches. A separate differential check proves that an
invalid permutation, which formerly left partially rewritten DQ/Mean metadata
despite returning zero, is now a complete ModelIR no-op. Focused coverage
includes one-index multi-match execution, maintained-index equivalence,
negative-axis remapping, edge and operator order, shape/signature propagation,
public and fan-out boundaries at every intermediate, `keepDims`, shared axes,
invalid axes/permutation, missing tensors, missing-required-type/no-index
pruning, and ownership with `48 passed`. Architecture, core, pass-efficiency,
and the complete quantization-cleanup suite passed with `260 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The indexed pseudo-LeakyReLU checkpoint compiles the complete prior committed
function AST and preserves exact ModelIR and statistics for one and two
simultaneous matches, including alpha constants on both MUL sides and nondefault
legacy SUB fields. Focused coverage verifies one index build, maintained-index
and LayoutState equivalence, batch producer removal, alpha values, reversed SUB
rejection, missing constant, every public intermediate, fan-out at every edge,
negative-source mismatch, integer boundaries, missing-family/no-index pruning,
and ownership with `17 passed`. Architecture, core, pass-efficiency, complete
graph cleanup, and the indexed fusion suite passed with `249 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The generic indexed MUL-square constant-fold checkpoint compiles the complete
prior committed YOLO-named function AST and preserves exact valid ModelIR and
statistics for one and two simultaneous matches. A separate differential check
proves that a public square intermediate, formerly rewritten despite losing its
producer contract, is now a complete no-op. Focused coverage verifies one-index
multi-match execution, all pre/anchor/scale constant-side combinations,
maintained-index and LayoutState equivalence, float16 fused values, normalized
metadata, quantization cloning, batch compaction, public/fan-out guards for all
three intermediates, singleton and finite pre-scale, floating anchor/scale,
finite result, exact self-square, missing constants, no-MUL/no-index/no-prune,
and generic ownership with `18 passed`. Architecture, core, pass-efficiency,
constant-fold, and indexed fold coverage passed with `234 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The indexed leading-singleton Gather-to-Reshape checkpoint compiles the
complete prior committed function AST and preserves exact valid ModelIR,
lineage metadata, and statistics for one and two simultaneous matches. A
separate differential check proves that multiple zero indices and a dynamic
leading signature, both formerly rewritten, are now complete no-ops. Focused
coverage verifies one-index multi-match execution, negative-axis
normalization, nested fixed-point exposure, maintained-index and LayoutState
equivalence, matching quantization, all public/fan-out/duplicate/order/input-
position boundaries, axis and batch-dimension options, static and dynamic
metadata consistency, constant buffer dtype/value/cardinality,
dtype/quantization equality, missing tensors, transactional rejection,
missing-family/no-index pruning, and unique semantic ownership with `38 passed`.
Architecture, core, pass-efficiency, ModelIR utilities, dynamic Reshape, and
indexed Gather/Reshape coverage passed with `269 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`. No
Tier corpus conversion was run.

The indexed terminal Softmax/Transpose checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR, lineage metadata,
and statistics for one and two simultaneous terminal outputs. Separate
differential checks prove that a separately public Softmax intermediate and a
missing Softmax-output tensor, both formerly rewritten, are now complete
no-ops. Focused coverage verifies one-index multi-output execution,
maintained-index and LayoutState equivalence, marker removal with axis/options
preservation, public output identity, all public input/output boundaries,
dtype/shape/signature propagation, quantization cloning, terminal and Softmax
fan-out, duplicate producers, operator order, marker truth, exact permutation,
operator arity/type, missing permutation and runtime buffer, missing tensors,
rank/signature validation, non-output exclusion, missing-family/no-index
pruning, shared marker ownership, and unique semantic ownership with
`30 passed`. Architecture, core, pass-efficiency, terminal Mean layout, layout
Transpose, and indexed terminal
Softmax coverage passed with `255 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed pre-ArgMax terminal-layout checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR, lineage metadata,
and statistics for private singleton axes, two matches sharing one axis, and a
negative channel axis. Differential checks prove that a public axis remains
one while a private clone becomes three, instead of mutating the public value
as before, and that a Transpose intermediate exposed as a public input is now a
complete no-op. Focused coverage verifies one-index multi-match execution,
shared-axis ownership changes across differential removal, maintained-index
and LayoutState equivalence, private and public input/output axis constants,
negative-axis normalization, NumPy dtype and quantization cloning, exact
operator options/provenance, every public/fan-out/duplicate/order boundary,
permutation and operator arity/type, signed singleton axis validation, all
required tensors, rank-four permuted shape/signature and reduced-output
metadata, dtype agreement, transactional rejection, missing-family/no-index
pruning, and unique semantic ownership with `39 passed`. Architecture, core,
pass-efficiency, layout Transpose, indexed terminal ArgMax, and indexed terminal
Softmax coverage passed with `288 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed quantized-MaxPool checkpoint compiles the complete prior committed
function and preserves exact ModelIR and statistics for valid one- and two-
chain fixtures across INT8 and UINT8. Differential checks separately prove
that the former tolerant comparison folded a near-but-different scale, that
missing float bridge tensors were accepted, and that a public-input bridge
could lose its producer; all three are now transactional no-ops. Focused
coverage verifies one-index multi-match execution, maintained-index and
LayoutState equivalence, quantized input/output fan-out, dictionary grids,
Pool options/version/provenance, cloned quantization, public boundaries,
duplicate producers, operator order and arity, exact grid/dtype/range, float
bridge dtype, exact rank-four shape/signature metadata, missing tensors,
missing-family/no-index pruning, and unique semantic ownership with
`52 passed`. Architecture, core, pass-efficiency, established quantized-Pool and
quantization-cleanup coverage, and the new indexed suite passed together with
`318 passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the new
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed quantized-Logistic checkpoint compiles the complete prior
committed function and preserves exact ModelIR and statistics for valid one-
and two-chain fixtures across INT8 and UINT8. Differential checks separately
prove that the former tolerant comparison folded a near-canonical output
scale, missing input quantization was accepted, missing float bridge tensors
were accepted, and a public-input bridge could lose its producer; all four are
now transactional no-ops. Focused coverage verifies one-index multi-match
execution, maintained-index and LayoutState equivalence, quantized input/output
fan-out, dictionary grids, rank-independent elementwise metadata, Logistic
options/version/provenance, public boundaries, duplicate producers, operator
order and arity, input grid validity, exact canonical output grid, float dtype,
shape/signature equality, missing tensors, missing-family/no-index pruning,
the two established direct tests, and unique semantic ownership with
`55 passed`. Architecture, core, pass-efficiency, established quantized-Pool
and quantization-cleanup coverage, both indexed quantized suites, and the two
direct Logistic tests passed together with `373 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test and
architecture test, syntax compilation, and `git diff --check` passed. No Tier
corpus conversion was run.

The indexed quantized-Softmax checkpoint compiles the complete prior committed
function and preserves exact ModelIR and statistics for valid one- and two-
chain fixtures across INT8 and UINT8, including the former beta tolerance.
Differential checks separately prove that the former tolerant grid comparison
folded a near-canonical output scale, missing input quantization and float
bridge tensors were accepted, a public-input bridge could lose its producer,
and an explicit non-last axis was ignored; all five are now transactional no-
ops. Focused coverage verifies one-index multi-match execution, maintained-
index and LayoutState equivalence, quantized input/output fan-out, dictionary
grids, rank-two negative-axis handling, beta/default/axis option preservation,
Softmax version/provenance, public boundaries, duplicate producers, operator
order and arity, input grid validity, exact canonical output grid, malformed
options, positive rank, float dtype, shape/signature equality, missing tensors,
missing-family/no-index pruning, and unique semantic ownership with
`62 passed`. The real QLinearSoftmax wrap conversion and sequential inference
matched ONNX exactly with `1 passed`. Architecture, core, pass-efficiency,
established quantized-Pool and quantization-cleanup coverage, and all three
indexed quantized suites passed together with `433 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test and
architecture test, syntax compilation, and `git diff --check` passed. No Tier
corpus conversion was run.

The indexed expanded-HardSigmoid fold checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for private one- and
two-chain INT8/UINT8 fixtures and a shared-four-constant fixture. Differential
checks prove that a near-equal output grid and missing float tensors formerly
folded, a public-input bridge could lose its producer, and public scalar
outputs were mutated in place. A fault-injected second quantization clone also
proves the former helper changed the first scalar data/dtype before raising,
while the new four-plan transaction returns a complete no-op. Focused coverage
verifies one-index multi-match execution, maintained-index and LayoutState
equivalence, all scalar input positions, shared/public clone ownership and
names, operator options/version/provenance, every public/fan-out/duplicate/
order/arity boundary, exact grid validity, all data metadata, finite singleton
constants, producer rejection, representability, clone-failure transaction,
missing-family/no-index pruning, and unique semantic ownership with
`78 passed`. Architecture, core, pass-efficiency, the established quantized
Pool and quantization-cleanup suites, all prior indexed quantized folds,
quantized activation and
expanded-HardSigmoid bridge suites, and the new fold suite passed together
with `529 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test and architecture test, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed quantized-TransposeConv checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for valid private
one- and two-chain fixtures and a shared-filter fixture. Differential checks
prove that missing float input/output tensor records and a public-input bridge
formerly folded, a public float filter was mutated in place, an invalid
negative input scale was accepted, and output-grid clone failure changed a
private float filter before raising. The new owner rejects or clones each case
transactionally. Focused coverage verifies one-index multi-match execution,
maintained-index and LayoutState equivalence, private/shared/public and
already-INT8 filter ownership, exact output-shape/filter/data roles, operator
options/version/provenance, every public/fan-out/duplicate/order/arity
boundary, independent activation-grid validity, bridge dtype and metadata,
filter rank/buffer/dtype/grid/producer constraints, clone-failure transaction,
missing-family/no-index pruning, and unique semantic ownership with
`61 passed`. Architecture, core, pass-efficiency, the established quantized
Pool and quantization-cleanup suites, all prior indexed quantized folds,
quantized activation and expanded-HardSigmoid bridge/fold suites, and the new
TransposeConv suite passed together with `590 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test, scoped
architecture/lowerer checks, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed decomposed-InstanceNormalization checkpoint compiles the complete
prior committed function and preserves exact ModelIR/statistics for valid
NHWC, NCHW, post-Transpose, and rank-five fixtures. Differential checks prove
that reversed Sub operands, a non-Add epsilon node, a public Mean intermediate,
a shared scale, a wrong-sized scale, and floating axes formerly mutated graph
state. A malformed late bias shape additionally proves that the former helper
changed axes, intermediate shapes, and scale data before raising, while the
new owner returns a complete no-op. Focused coverage verifies one-index
multi-layout execution, maintained-index and LayoutState equivalence, ranks
three/four/five, separate/shared Mean axes, optional post-Transpose bias-axis
mapping, already-correct idempotence, plan-failure transaction, all operator
types/roles/arity/order, public/fan-out/duplicate boundaries, finite epsilon
and reciprocal-one constants, axes dtype/ownership, complete intermediate
metadata, scale/bias cardinality and ownership, missing-marker/no-index
behavior, and unique semantic ownership with `38 passed`. Four existing real
ONNX builder/serialization characterizations passed. Architecture, core,
pass-efficiency, the new indexed suite, and those real characterizations passed
together with `264 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed NCHW Concat/global-pool/Conv checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for valid one- and
two-chain fixtures, negative spatial axes, and INT64 Reshape shape buffers.
Differential checks prove that non-global Mean axes, a fan-out or public Concat
intermediate, a three-value or floating shape buffer, a duplicate Reshape
producer, and a missing runtime filter buffer formerly changed graph state. A
faulting late shape-buffer read additionally proves that the former helper
changed the Concat axis and three tensor records before raising, while the new
owner returns a complete no-op. Focused coverage verifies one-index multi-
match execution, maintained-index and LayoutState equivalence, exact operator
roles/order/arity, normalized global axes, public/fan-out/duplicate boundaries,
positive compatible NCHW inputs, the filter/channel equation and buffer,
private producer-free integer shape-buffer ownership, option/metadata/data
updates, incomplete-family/no-index behavior, the existing characterization,
and unique semantic ownership with `34 passed`. Architecture, core, pass-
efficiency, all existing Conv-layout tests, and the new indexed suite passed
together with `268 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed NCHW Concat/Transpose/(Transpose)Conv checkpoint compiles the
complete prior committed function and preserves exact ModelIR/statistics for a
combined direct-Conv, pre/post-prefix Conv, and TransposeConv fixture.
Differential checks prove that public or fan-out Transpose/pre-passthrough
outputs, a duplicate Transpose-output producer, a nonpositive input channel,
a produced permutation, and a missing runtime filter buffer formerly changed
graph state; the new owner rejects all eight cases transactionally. Focused
coverage verifies one-index multi-family execution, maintained-index and
LayoutState equivalence, the four established positive characterizations,
exact Transpose and data/filter roles, pre/post passthrough traversal,
public/fan-out/duplicate/order boundaries, positive compatible NCHW inputs,
the filter/channel/buffer equation, direct-Conv-only output refresh, already-
correct exclusion, incomplete-family/no-index behavior, and unique semantic
ownership with `30 passed`. Architecture, core, pass-efficiency, all existing
Conv-layout tests, and both new indexed Concat repair suites passed together
with `294 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed mixed-singleton NCHW-input/NHWC-Concat checkpoint compiles the
complete prior committed function and preserves exact ModelIR, lineage,
operator order, deterministic names, quantization metadata, and statistics for
valid multi-candidate, multi-adapter, and name-collision fixtures. Differential
checks prove that an output-channel mismatch, duplicate or later source
producer, inconsistent dynamic signature, and mixed dtype formerly changed
graph state; all five are now complete no-ops. A late fault in the second
quantization clone additionally proves that the former helper inserted the
first Reshape before raising, while the new owner leaves ModelIR unchanged.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, public/produced/constant sources, same-source adapter
reuse, global name reservation, deep quantization cloning, exact Concat axis,
arity, output-channel and dtype equations, static and one-dynamic-dimension
shape/signature contracts, final shape-reconciliation stability, duplicate/
later/unresolved producer rejection, clone-failure transaction, no-repair and
missing-family behavior, the established characterization, and unique
semantic ownership with `33 passed`. Architecture, core, pass-efficiency,
singleton-Reshape coverage, the two preceding indexed Concat owners, flatten-
Concat and NDHWC-Concat coverage, the new indexed suite, and the existing
characterization passed together with `336 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. Ruff on the new owner/test and architecture test,
scoped lowerer checks, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed window-partition checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, Transpose
options replacement, version/axis semantics/provenance, quantization metadata,
unused-tensor pruning, and statistics for valid public-input, produced-input,
and quantized multi-chain fixtures. Differential checks prove that duplicate
first-Reshape producers, floating shape vectors, missing final-output metadata,
mixed data dtypes, and a graph-input/producer conflict formerly changed graph
state; all five are now complete no-ops.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, exact block/spatial and three-stage shape equations,
static and one-dynamic-dimension contracts, final dynamic-Reshape convergence,
per-tensor quantization, in-place provenance, topology/order/arity, public and
fan-out boundaries, duplicate/later/unresolved producers, typed producer-free
shape/permutation vectors, dynamic shape-vector ownership, two-dynamic-output
rejection, historical prune/no-index behavior, and unique semantic ownership
with `52 passed`. Architecture, core, pass-efficiency, dynamic-Reshape, the new
indexed suite, and two real ONNX SpaceToDepth chain characterizations passed
together with `289 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test and architecture test, scoped lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed window-reverse checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, deterministic
shared-shape clone names, Reshape and DEPTH_TO_SPACE options, version/axis
semantics/provenance, quantization metadata, unused-tensor pruning, and
statistics for a five-chain public-input, produced-input, quantized, and
shared-vector fixture. Differential checks prove that an extra first-Reshape
input, a floating shape vector, a public first shape vector, mixed data dtypes,
and an inconsistent shape signature formerly changed graph state; all five are
now complete no-ops.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, legacy sequential shared-vector clone/update behavior,
exact reverse block/flatten/spatial and three-stage shape equations, static and
one-dynamic-dimension contracts, final shape-convergence stability, per-tensor
quantization, in-place provenance, topology/order/arity, public and fan-out
boundaries, duplicate/later/unresolved producers, typed producer-free vectors,
two-dynamic-target rejection, clone-failure transaction, historical
prune/no-index behavior, unique semantic ownership, and a production real-ONNX
characterization. The reverse and partition focused suites passed together
with `99 passed`. Architecture, core, pass-efficiency, dynamic-Reshape,
ModelIR-writer, strict-integer-quantization, both indexed window suites, and an
established DepthToSpace characterization passed together with `381 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed Conv1D-shim unary checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, unary identity,
options/version/axis semantics/provenance, output dtype and quantization repair,
unused-tensor pruning, and statistics for a five-chain public-input,
produced-input, quantized, CAST, inferred-axis, and alternate-axis fixture.
Differential checks prove that a floating permutation, produced ExpandDims
axis, inconsistent internal shape metadata, mixed input dtype, and duplicate
final producer formerly changed graph state; all five are now complete no-ops.
A faulting quantization clone additionally proves that the former helper
rewired the unary before raising, while the indexed owner leaves ModelIR
unchanged.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, all sixteen supported unary types, explicit/negative
and uniquely inferred axes, CAST dtype transition, output metadata repair,
consistent multi-dynamic signatures, exact NHWC/NCHW permutation and
drop/insert equations, every operator role/order/arity, public and fan-out
boundaries, duplicate/later/unresolved producers, typed producer-free
permutation and axis vectors, per-tensor quantization, per-axis rejection,
clone-failure transaction, historical prune/no-index behavior, and unique
semantic ownership with `75 passed`. Architecture, core, pass-efficiency,
three established transpose-unary suites, five adjacent Conv1D-shim
characterizations, and the new indexed suite passed together with `321
passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed rank-four Conv1D-shim unary checkpoint also compiles the complete
prior committed function. A static public-input, produced-input, quantized,
and CAST four-chain fixture preserves exact operators, tensors, inputs,
outputs, metadata, and statistics. Differential checks prove that floating
permutation constants, a produced ExpandDims axis, mixed data dtypes, and a
duplicate final producer formerly rewrote graph state; all four are now
complete no-ops. Focused coverage verifies one-index multi-match execution,
supplied-index and LayoutState equivalence, deterministic shared shape/axis
constant cloning, fan-out bridge topology, one dynamic height/width/channel
dimension, 53 unsafe contracts, quantization-clone failure, historical
prune/no-index behavior, and both established direct-builder
characterizations with `63 passed`. Both indexed Conv1D suites, architecture,
core, pass-efficiency, three established Transpose-Unary suites, and five
adjacent direct-builder characterizations passed together with `383 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed Conv1D unary fan-out checkpoint compiles the complete prior
committed helper. For four static public-input, produced-input, quantized, and
alternate-axis branches, the new ModelIR and statistics are exactly equal to
the legacy result after topologically sorting that result. The legacy result
required twelve operator moves; the new owner emits the final order directly.
CAST additionally repairs the retained Transpose output from `FLOAT32` to the
unary's `INT32` output contract. Differential checks prove that a chain without
fan-out, floating permutation, produced ExpandDims axis, duplicate final
producer, and mixed input dtype formerly rewrote graph state; all five are now
complete no-ops. A faulting quantization clone formerly raised after changing
the final dtype, while the indexed owner leaves ModelIR unchanged.

Focused coverage verifies one-index five-chain execution, supplied-index and
LayoutState equivalence, all sixteen unary types, public NCHW output, three
dynamic-signature forms, 27 unsafe contracts, clone-failure transaction,
equivalent negative axes, historical prune/no-index behavior, and the
established direct-builder characterization with `58 passed`. All three
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites,
and five adjacent direct-builder characterizations passed together with `437
passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff with only
the documented legacy unused-import exclusions, syntax compilation, and `git
diff --check` passed. No Tier corpus conversion was run.

The indexed flattened InstanceNormalization checkpoint compiles the complete
prior committed 491-line helper. Four static public-input, produced-input,
CAST, and alternate-unary branches retain exact operators, tensors, inputs,
outputs, metadata, and statistics. Differential checks prove that a floating
permutation, produced second-Reshape shape, negative epsilon, reversed
reciprocal DIV, mixed intermediate dtype, and duplicate final producer
formerly rewrote graph state; all six are now complete no-ops.

Focused coverage verifies one-index five-chain execution, supplied-index and
LayoutState equivalence, deterministic shared Reshape/axis cloning, all
sixteen unary types, one dynamic batch/width/channel dimension, 37 unsafe
contracts, clone-failure transaction, historical prune/no-index behavior, and
the established direct-builder characterization with `61 passed`. All four
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and five adjacent direct-builder characterizations
passed together with `498 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed tencoder checkpoint removes the former 723-line raw helper from
the lowerer. Its dedicated owner matches the exact flattened
InstanceNormalization prefix rather than searching an arbitrary upstream
producer path, validates the simple rank-four or legacy rank-three residual
branch, and proves the complete two-Slice/Logistic/Mul/scale gate, residual
ADD, ExpandDims, post-Transpose, and Conv consumer topology before mutation.
The InstanceNormalization owner now exposes that common prefix as a
side-effect-free plan while retaining its prior complete-chain behavior.

The rewrite converts both residual inputs from NCW to NWC, adjusts the second
Reshape, Slice begin/size, floating channel-scale, and ExpandDims constants,
repairs every changed tensor including Logistic and gate intermediates, and
removes the three boundary Transposes in one indexed compaction. Private
constants update in place; shared constants receive unique planned clones even
when two changed operator inputs originally share the same tensor. A side
consumer receives one `[0,2,1]` bridge immediately before its earliest use.

Focused coverage includes exact numeric equivalence for simple/legacy left
branches with and without fan-out, supplied-index and LayoutState equivalence,
topological bridge placement, deterministic shared integer/float cloning,
one dynamic batch or width dimension, eight unsafe transactional no-ops, the
three established characterizations, and both semantic-ownership checks with
`84 passed`. The four indexed Conv1D
suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and eight adjacent direct-builder characterizations
passed together with `520 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed Conv1D BatchMatMul checkpoint removes the former 223-line helper
and its repeated whole-graph consumer-map rebuild. Its dedicated owner accepts
only exact typed `[0,3,1,2]` Transpose roots, one explicit singleton Squeeze,
an optional strict supported-unary chain, and the left input of one
BatchMatMul. Source and intermediate shapes/signatures, producer order,
consumer multiplicity, public boundaries, dtype transitions, per-tensor
quantization, right operand, contracted dimensions, batch broadcasting, and
output shape are all validated before mutation.

Squeezing the transposed channel axis maps to the original NHWC channel axis
without changing rank-three order or `adjX`. Squeezing either supported
spatial axis maps back to the corresponding NHWC spatial axis, swaps the last
two dimensions of Squeeze and unary metadata, and toggles `adjX`; the effective
matrix and BatchMatMul output remain identical. The original unary and
BatchMatMul objects retain all unrelated options, provenance, version, and
axis semantics, and the single boundary Transpose is removed through the
maintained index.

Focused coverage includes twelve exact NumPy equivalence variants across all
three axes, both `adjX` values, and public/produced sources; zero-length unary
chains, rank-two right operands, `adjY`; all sixteen unary types; one dynamic
batch signature; fifteen unsafe transactional no-ops; the preflight/no-index
path; and semantic ownership with `49 passed`. The five
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and eight adjacent direct-builder characterizations
passed together with `569 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed decoder deconvolution-input checkpoint removes the former
193-line helper and its repeated producer/consumer-map rebuilds. Its dedicated
owner validates the complete BatchMatMul, commutative bias ADD, axis-two
ExpandDims, `[0,2,3,1]` Transpose, and input-two TransposeConv path. Both matrix
operands, every producer and consumer, public boundaries, concrete and dynamic
shape/signature equations, dtype, per-tensor quantization, contracted
dimension, batch broadcasting, bias broadcast, and operator order are proven
before any mutation.

The rewrite applies `(A·B)^T = B^T·A^T`: it swaps the original BatchMatMul
inputs, maps `adjX` to `not old_adjY` and `adjY` to `not old_adjX`, changes
`[N,C,L]` metadata to `[N,L,C]`, reshapes the length bias to `[1,L,1]`, moves
ExpandDims from axis two to axis one, and connects its retained
`[N,1,L,C]` output directly to TransposeConv. Unrelated BatchMatMul options and
provenance remain intact. Private constants update in place; shared bias and
axis constants are cloned before edges change. Clone failure and every unsafe
contract are complete no-ops.

Focused coverage includes sixteen exact NumPy equivalence variants across all
`adjX/adjY` values, both ADD input orders, and public/produced operands; rank-
two RHS, rank-two/three bias, negative axis, shared bias/axis cloning, one
dynamic batch signature, twenty-six unsafe transactional no-ops, clone
failure, preflight/no-index behavior, and semantic ownership with `51 passed`.
The six indexed Conv1D/decoder suites, architecture, core, pass-efficiency,
three established Transpose-Unary suites, and eight adjacent direct-builder
characterizations passed together with `620 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed terminal Squeeze/Mean checkpoint removes the former 152-line raw
helper and its repeated whole-graph consumer-map rebuild. Its dedicated owner
validates the exact `[0,3,1,2]` Transpose, axis-two Squeeze, axis-one kept-
dimension Mean, and axis-one terminal Squeeze topology. Every producer,
consumer, public boundary, typed constant, operator order, singleton,
rank-four/rank-three/rank-two shape and signature equation, dtype, and per-
tensor quantization contract is proven before mutation.

The rewrite removes the Transpose, moves the first Squeeze to the NHWC source
axis one, changes `[N,C,W]` metadata to `[N,W,C]`, moves the Mean to axis two,
changes `[N,1,W]` metadata to `[N,W,1]`, and moves the terminal Squeeze to axis
two. The final `[N,W]` tensor name, values, metadata, graph-output position,
and downstream edges remain unchanged. Private Mean axes update in place;
shared axes receive a deterministic clone. Negative equivalent axes and one
dynamic batch, width, or reduced-channel signature are preserved. Clone
failure and all unsafe contracts are complete no-ops.

Focused coverage includes sixteen exact NumPy equivalence variants across
public/produced sources and all positive/negative axis representations,
shared-axis cloning, dynamic batch/width/reduced-channel signatures, two-chain
execution, twenty-seven unsafe transactional no-ops, clone failure,
apply-preflight collision, and preflight/no-index behavior with `51 passed`;
the semantic-ownership test also passed. The seven indexed
Conv1D/decoder/terminal suites, architecture, core, pass-efficiency,
Mean/terminal-Mean, and three Transpose-Unary suites passed
together with `677 passed in 56.59s`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed in 6.81s`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed direct InstanceNormalization checkpoint makes the single private
post-Transpose tail the first bounded owner within the former four-mode raw
helper. It proves the pre-Transpose/Squeeze/Reshape boundary, both kept-
dimension Means, ordered SUB and reciprocal DIV, square/normalize/affine
branches, and the sole post-Transpose before changing ModelIR. Exact
shape/signature equations, all producers and consumers, graph order, public
boundaries, FLOAT16/FLOAT32 dtype, unquantized tensors, typed reshape/axis
constants, nonnegative finite epsilon, unit numerator, and finite scale/bias
are validated in one `ModelIRGraphIndex`.

The rewrite moves the normalization to NHWC axes `[1,2]`, converts rank-three
CHW and every full/reduced rank-four metadata contract, changes scale and bias
from `[1,C,1,1]` to `[1,1,1,C]`, reuses the post-Transpose output name on the
bias ADD, and removes the two boundary Transposes. Reshape, shared Mean-axis,
scale, and bias constants are planned together; unrelated consumers receive
deterministic clones. Five representative static legacy variants are exactly
ModelIR-identical to the committed helper, including lineage-event order.
Separate equivalent Mean axes are now handled transactionally instead of
being skipped by the legacy shared-axis guard.

The side-Squeeze, Squeeze/unary/Reshape, and
Squeeze/residual-ADD/Reshape modes remain in the compatibility path. Each is
an indexed-owner no-op followed by a numerically equivalent legacy rewrite;
supplied GraphIndex and LayoutState remain current. Unsafe direct candidates
such as reversed SUB, wrong axes, negative epsilon, or quantized intermediates
are explicitly short-circuited and cannot fall through to the legacy mutator.
Per-candidate dispatch preserves graph order across mixed direct and legacy
tails; a 33-chain characterization proves the original shared 32-rewrite
limit selects the same prefix rather than favoring all direct tails first.

Focused coverage includes thirty-two NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced source, shared/separate axes, positive/negative
axes, and affine input order; shared cloning for all four changed constant
roles; dynamic height/width/channel signatures; two-chain execution; forty-
four unsafe transactional no-ops; clone failure; apply-preflight collision;
four legacy-fallback blockers; all three retained legacy modes; compatibility
statistics; mixed-mode order/limit; and preflight/no-index behavior with
`93 passed`. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `495 passed in 54.58s`; twelve selected direct-builder
characterizations passed in `0.94s`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed in 6.70s`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed side-Squeeze InstanceNormalization checkpoint reuses that common
decomposition matcher for the exact dual-consumer tail: one NCHW-to-NHWC
post-Transpose and one axis-zero Squeeze. It moves the common normalization to
NHWC, removes the two boundary Transposes, and inserts one local
NHWC-to-NCHW adapter immediately before the side Squeeze. The side tensor name,
CHW shape/signature, dtype, quantization contract, public-output behavior, and
downstream edges remain unchanged. A compatible shared INT32 or INT64 adapter
permutation is reused; every absent, conflicting, produced, public, floating,
or quantized constant case is resolved before mutation. Unsafe side candidates
cannot fall through to the legacy mutator. Direct and side candidates retain
their original graph-order position and shared 32-rewrite ceiling; the two
remaining unary/Reshape and residual-ADD/Reshape modes are untouched.

Focused coverage includes sixteen NumPy equivalence variants across side
operator order, public/downstream-consumed side outputs, public/produced
sources, and FLOAT16/FLOAT32; dynamic height/width/channel signatures;
multi-chain constant reuse; five invalid existing-adapter cases; nine unsafe
side contracts; adapter-allocation collision; three compatibility-fallback
blockers; and the existing direct and mixed-mode cases with `131 passed in
0.89s`. Four representative side-tail variants were exactly ModelIR-identical
to the committed legacy helper. The related InstanceNormalization, Pad, Mean,
architecture, core, and pass-efficiency suites passed with `533 passed in
53.58s`; twelve selected direct-builder characterizations passed with `12
passed in 0.94s`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed in
6.67s`. Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed unary/Reshape InstanceNormalization checkpoint reuses the same
core matcher for the exact `Squeeze -> unary -> Reshape -> post-Transpose`
tail. The tail Squeeze and unary metadata move from CHW to HWC, the second
Reshape moves from NCHW to NHWC, its typed shape constant and `newShape` are
rewritten together, and that Reshape receives the former post-Transpose output
name. Shared shape constants receive deterministic clones. All thirteen
legacy unary operators remain accepted; CAST alone may change dtype, while
every other unary retains the core FLOAT16/FLOAT32 dtype. Quantized or unsafe
tail contracts are complete no-ops and cannot fall through to the legacy
mutator. The residual-ADD/Reshape tail is unchanged and remains the sole
legacy mode.

Focused coverage includes eight exact NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced sources, and shared/separate Mean axes; all
thirteen unary operators; dynamic height/width/channel signatures; shared
tail-shape cloning; multi-chain execution; thirteen unsafe transactional
no-ops; three compatibility-fallback blockers; mixed direct/side/unary graph
order under the shared 32-rewrite ceiling; and preflight/no-index behavior
with `175 passed in 0.73s`. Static, produced-source, FLOAT16, negative-axis,
commuted-affine, and CAST variants were exactly ModelIR-identical to the
committed legacy helper. Separate equivalent Mean axes are an intentional
improvement over the legacy shared-axis restriction. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `577 passed in 54.16s`; twelve selected direct-builder
characterizations passed with `12 passed in 0.94s`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed in 6.63s`. Ruff, syntax compilation, and `git diff
--check` passed. No Tier corpus conversion was run.

The indexed residual/Reshape InstanceNormalization checkpoint completes the
four-mode migration. It validates the main `Squeeze -> residual ADD -> Reshape
-> post-Transpose` tail together with either a rank-three HWC-to-CHW residual
bridge or an NHWC-to-NCHW bridge followed by Squeeze and an optional retained
unary. The residual bridge is removed only after every shape/signature, dtype,
quantization, public-boundary, fan-out, producer, consumer, and graph-order
contract succeeds. Residual unary CAST may change FLOAT16/FLOAT32 dtype; the
other twelve operators must preserve it.

ADD output fan-out is planned in the same transaction. Its Reshape path moves
to HWC/NHWC, while all other consumer slots share one deterministic
HWC-to-CHW adapter. A compatible INT32/INT64 fixed permutation is reused;
invalid, produced, public, quantized, or collision cases are complete no-ops.
After this migration the former 975-line compatibility helper contains no raw
ModelIR mutation: it is a 60-line dispatcher that tries the four indexed modes
at each pre-Transpose's original graph position and preserves their shared
32-rewrite ceiling.

Focused coverage includes twelve NumPy equivalence variants across all three
residual source forms, FLOAT16/FLOAT32, and public/produced main sources; all
thirteen residual unary operators; nine dynamic-signature combinations; four
fan-out position/repeated-slot variants; multi-chain execution; commuted ADD;
existing adapter reuse; fifteen unsafe transactional no-ops; four
compatibility-fallback blockers; adapter-allocation collision; all four mixed
tail modes under the shared cap; and preflight/no-index behavior with `238
passed in 0.92s`. Ten representative static, produced-source, FLOAT16,
negative-axis, commuted-affine, residual-source, and fan-out variants were
exactly ModelIR-identical to the committed legacy helper. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `640 passed in 54.91s`; twelve selected direct-builder
characterizations passed with `12 passed in 1.00s`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed in 6.92s`. Ruff, syntax compilation, and `git diff
--check` passed. No Tier corpus conversion was run.

Residual-specific fixtures and cases are isolated in
`test_flatbuffer_direct_indexed_instance_norm_residual_layout.py`; the common
ModelIR builder/evaluator remains in the direct-tail module. This keeps the
common characterization file at 1,825 lines and the residual module at 642
lines without duplicating fixture construction.

The focused indexed and architecture tests pass Ruff normally. The changed
legacy characterization file passes with its pre-existing `F401` findings
scoped out, and the lowerer passes with its pre-existing `F401` and `F841`
findings scoped out. Every changed Python file passes `python -m py_compile`,
and `git diff --check` passes. The
immediately preceding DepthToSpace, Pool, dynamic-Pool, simple-alias, and
aligned-scalar checkpoints passed their focused synthetic and ownership
selections.

The indexed post-Transpose-bias InstanceNormalization checkpoint extracts the
rank-four decomposition matcher and generic constant transaction into
`passes/decomposed_instance_norm.py`. Both the established pre/post-tail owner
and the new `passes/instance_norm_post_bias_layout.py` owner now validate the
same exact Mean/SUB/square/variance/epsilon/SQRT/reciprocal/normalize/scale
contract. Epsilon must be finite and nonnegative, the DIV numerator must be
exactly one, all retained tensors must share one unquantized FLOAT16/FLOAT32
contract, and every producer, consumer multiplicity, public boundary,
shape/signature, and graph-order relation is proven before mutation.

The new owner accepts shared or separate positive, negative, or reversed Mean
axes; commuted SUB and affine operands; scalar, NCHW, or already-NHWC scale and
bias constants; public or produced NHWC sources; and dynamic height, width, or
channel signatures. Axes and coefficients are planned by use. Private values
update in place, unrelated consumers receive deterministic clones, and one
constant shared by scale and bias is updated once for both slots. Rejected
candidates do not prune orphan tensors or partially rewrite constants. On
success the Means and SUB consume the original NHWC source, axes become
`[1,2]`, retained core metadata becomes NHWC, the bias ADD consumes the scaled
tensor directly, and only the two boundary Transposes are removed.

The lowerer helper is now a 19-line compatibility dispatcher. In the repeated
normalization recovery loop, the four-tail owner and post-bias owner share one
live `ModelIRGraphIndex`; every late production call supplies the Session
`LayoutState`. The new owner has one bounded graph-order candidate scan, a
32-rewrite ceiling, differential index updates, and no repeated full consumer
map or unbounded fixed-point loop. The existing four-tail characterization
remains unchanged after adopting the common matcher.

Focused coverage includes twenty-four NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced source, shared/separate and positive/negative/
reversed Mean axes, commuted SUB/affine operands, and valid scalar/NCHW-scale/
NHWC-bias forms; three dynamic-signature cases; a public bias-ADD output;
shared changed-constant cloning; one shared scale/bias tensor; legacy
coefficient-layout acceptance; two-chain capped execution; thirty-six unsafe
transactional no-ops; clone-allocation collision; and preflight/no-index
behavior. The post-bias, existing four-tail, compatibility direct-builder, and
ownership selections passed with `310 passed in 1.93s`. The full architecture
suite passed with `176 passed in 52.04s`; thirteen selected InstanceNorm direct-
builder tests passed with `13 passed in 1.42s`; TensorFlow-import-blocked import,
direct conversion, and `-cotof` passed sequentially with `3 passed in 3.97s`.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed InstanceNormalization residual-ADD checkpoint moves the adjacent
dual-branch layout rewrite to
`passes/instance_norm_residual_add_layout.py`. The owner accepts an exact
`NHWC -> NCHW` main branch, the common decomposed InstanceNorm core through
scale and bias, a second `NHWC -> NCHW` residual branch, and a tail ADD with
downstream NCHW consumers. It lifts both branches and the tail ADD to NHWC,
removes both pre-Transposes, and inserts one post-ADD `[0,3,1,2]` adapter that
preserves the original tensor name, NCHW metadata, downstream fan-out, and
repeated consumer slots.

The common helper now plans Mean axes, scale, and bias together before any
mutation. Private constants update in place, shared values receive deterministic
clones, scalar/NCHW/already-NHWC affine forms are handled explicitly, and
existing adapter constants are reused only after their dtype, value, ownership,
and quantization contracts are proven. The owner rejects public boundaries,
duplicate or backward producers, mixed dtypes, quantized normalization tensors,
invalid dynamic signatures, unsafe constant sharing, and all allocation
collisions without changing ModelIR. It uses one differential
`ModelIRGraphIndex`, a graph-order candidate scan, a deterministic 32-rewrite
ceiling, success-only pruning, and Session `LayoutState` synchronization. The
former 475-line lowerer mutator is now a 19-line dispatcher, and the repeated
normalization loop passes its live `residual_graph_index` into the owner.

Focused coverage includes thirty-two FLOAT16/FLOAT32 NumPy-equivalence
variants across public/produced main and residual sources, positive/negative/
reversed Mean axes, commuted affine operands, and alternate downstream
topologies; dynamic height, width, and channel signatures; fan-out and repeated
input slots; shared-constant cloning; valid adapter reuse; multi-chain rewrite
limits; thirty-nine transactional unsafe cases; three allocation-collision
cases; and preflight/no-index behavior. The new owner, adjacent post-bias and
four-tail owners, direct-builder characterizations, and ownership checks passed
with `395 passed in 2.45s`. The full architecture suite passed with
`177 passed in 56.80s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.47s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.21s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed residual-MUL/CONCAT InstanceNormalization checkpoint moves the
next three-Transpose tail to
`passes/instance_norm_residual_mul_concat_layout.py`. It validates the direct
decomposed core and bias, the first NCHW-to-NHWC bridge, one same-contract
residual ADD, exactly two tail MUL branches, one multiplicity-preserving
channel-axis CONCAT, and the final NCHW-to-NHWC bridge. The historical helper
name mentions Conv, but the indexed owner intentionally preserves the actual
legacy boundary: the former final-Transpose output may be a public output or
feed any later graph-ordered consumers, with no required Conv operator.

On success, all three Transposes are removed, the normalization and residual
tail run in NHWC, both tail-MUL outputs and CONCAT metadata are permuted, the
CONCAT axis changes from 1 to 3, and CONCAT directly produces the preserved
final output name. Dynamic height, width, and channel signatures, downstream
fan-out, repeated input slots, public outputs, and existing CONCAT option fields
are retained. The common constant planner now accepts additional coefficient
uses and plans both Mean axes, scale, bias, and both tail coefficients as one
transaction. Shared constants update once, unrelated users receive a
deterministic clone, and a late invalid tail coefficient can no longer leave
earlier axes or affine constants partially modified.

The owner requires exact producer, consumer multiplicity, graph-order,
shape/signature, dtype, quantization, typed-permutation, output-renaming, and
public-boundary contracts. It compares CONCAT inputs with `Counter`, uses one
differential `ModelIRGraphIndex`, scans candidates in graph order with a
32-rewrite ceiling, prunes only after success, and synchronizes the Session
`LayoutState`. Its former 501-line lowerer mutator is a 19-line dispatcher. All
four production calls pass LayoutState, and the repeated normalization loop
shares `residual_graph_index` with the preceding residual-ADD owner.

Focused coverage includes thirty-two FLOAT16/FLOAT32 logical-equivalence
variants across direct/produced main and residual sources, shared/separate and
positive/negative/reversed Mean axes, commuted operands, scalar/NCHW/NHWC
coefficients, and both CONCAT orders; three dynamic-signature cases; fan-out
and repeated input slots; public final output; shared-constant cloning; one
coefficient shared by all four affine uses; two-chain capped execution;
fifty-seven transactional unsafe cases; clone collision; and preflight/no-index
behavior. The new owner, adjacent five InstanceNorm owner groups,
direct-builder characterizations, and ownership checks passed with
`497 passed in 2.99s`. The full architecture suite passed with
`178 passed in 54.41s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.44s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.22s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed dual-statistics checkpoint moves the next normalization family to
`passes/instance_norm_dual_stats_layout.py`. Audit proved that neither branch
is the standard decomposed InstanceNorm core: one reduces spatial axes and one
reduces all non-batch axes, and both use
`SUB -> square -> Mean -> variance factor -> epsilon -> SQRT ->
DIV(centered,std) -> scale`. The owner therefore keeps a dedicated path matcher
and shares only typed constant, tensor-contract, metadata, and graph-index
utilities. This prevents a superficially similar topology from being assigned
the standard reciprocal/MUL normalization semantics.

The two branches feed blend ADD, gamma MUL, and beta ADD. Spatial axes are
planned together as `[1,2]`; global `[1,2,3]` axes are validated unchanged.
Branch scales and direct gamma/beta constants use one grouped coefficient
transaction, so shared constants update once and unrelated consumers receive a
deterministic clone. Exact `[1,C,1,1]` gamma/beta Reshapes from rank-one or
rank-two vectors are validated through their producer, shape constant, source,
consumer, dtype, quantization, and graph order, then bypassed and removed.
Variance factors and epsilon values must be finite, nonnegative scalar
FLOAT16/FLOAT32 constants.

Direct mode removes the input/output Transposes and gives beta ADD the former
NHWC output name. Residual mode additionally validates and removes an
independent NHWC-to-NCHW residual bridge, lifts the residual ADD to NHWC, and
uses that ADD as the preserved output producer. Its old NCHW output contract is
validated and permuted before rename, preventing dynamic-axis contamination.
The historical function name mentions Resize, but no Resize is required by the
legacy boundary; later ordered consumers, fan-out, and repeated input slots are
preserved.

The owner validates complete producer/consumer multiplicity, dependency order,
public boundaries, shape/signature, dtype, quantization, typed constants,
coefficient ownership, optional Reshape removal, and output rename before any
mutation. It uses a graph-order candidate scan, one differential
`ModelIRGraphIndex`, a 32-rewrite ceiling, success-only pruning, and Session
`LayoutState` synchronization. The former 712-line lowerer mutator is now a
19-line dispatcher; all four production calls pass LayoutState and the repeated
normalization loop shares `residual_graph_index` with the preceding owners.

Focused coverage includes forty-eight FLOAT16/FLOAT32 numerical-equivalence
variants across direct, residual-input, and produced-residual tails; direct and
produced main sources; shared/separate positive, negative, and arbitrarily
permuted axes; commuted operands; scalar/NCHW scales; direct and vector-Reshape
gamma/beta; six dynamic-signature modes; downstream fan-out and repeated slots
without
Resize; already-NHWC coefficients; shared-axis cloning; one coefficient shared
by all four affine sites; capped multi-chain execution; 143 transactional
unsafe contracts; clone collision; and preflight/no-index behavior. The new
owner and all indexed InstanceNorm owner groups passed with
`792 passed in 2.40s`. The full architecture suite passed with
`179 passed in 53.94s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.51s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.45s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed affine-chain checkpoint moves
`_optimize_fold_mul_add_mul_affine_chains` to
`passes/affine_chain_fold.py`. The legacy helper distinguished constants only
by the presence of tensor data, rebuilt full producer/consumer maps inside an
unbounded loop, compared intermediate users as sets, mutated shared constants
before all checks were known, and copied the removed intermediate ADD tensor's
metadata onto the preserved final output. It did not validate dtype,
quantization, fused activation, public boundaries, producer order, constant
provenance, or the original and folded broadcast contracts.

The new owner accepts finite, non-variable FLOAT16, FLOAT32, and FLOAT64
constants with matching array dtype and exact static tensor metadata. All data
and output tensors must share that dtype and be unquantized, and all three
binary operators must have `NONE` fused activation. Original static broadcast
shapes and dynamic signatures are checked across all three operators; the two
folded broadcasts must independently reproduce the final output contract.
When the removed final MUL introduced a broadcast expansion, the surviving
first-MUL intermediate receives the correctly expanded shape and signature.
The final output tensor itself remains untouched, retaining its dtype,
quantization, shape, signature, layouts, and ONNX provenance.

The first-MUL and ADD constant roles are grouped before apply. Constants shared
inside those roles are updated once; sharing with the removed final MUL is an
allowed internal use. Any unrelated consumer receives one deterministic
`_folded` clone while the original constant is preserved. Produced, public,
variable, quantized, non-finite, mismatched, colliding, or incompatible
constants reject the complete plan. Exact producer uniqueness, consumer
multiplicity, graph order, intermediate privacy, source resolution, downstream
fan-out, repeated final-output slots, and output rename are proven with one
`ModelIRGraphIndex`. The plan is resolved again immediately before apply, the
candidate scan is graph ordered and capped at 32 rewrites, pruning is
success-only, and LayoutState is synchronized. The former 219-line lowerer
helper is now a 17-line dispatcher; all three production calls pass
LayoutState.

Focused coverage contains forty-eight FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across all operand orders and scalar/channelwise
broadcasts; dynamic signatures; final broadcast expansion; constants shared
within the chain or with unrelated consumers; downstream fan-out and repeated
slots; candidate-only and capped execution; thirty-three transactional unsafe
contracts; clone collision; and no-index preflight. The focused owner plus the
pre-existing direct-builder characterization passed with `91 passed in 0.57s`.
The full architecture suite passed with `180 passed in 54.55s`; eighteen
selected affine direct-builder tests passed with
`18 passed, 737 deselected in 1.45s`; TensorFlow-import-blocked import, direct
conversion, and `-cotof` passed sequentially with `3 passed in 4.24s`. Scoped
Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed affine pre/post checkpoint moves
`_optimize_transpose_mul_add_const_prepost_nhwc_chains` to
`passes/affine_prepost_layout.py`. The legacy helper rebuilt full maps in an
unbounded loop, contained a permanently disabled PRELU branch and unused
`valid_posts`, and rotated rank-four constants heuristically up to three times.
It preflighted constants separately but then mutated them sequentially, so a
late failure or clone-name interaction could leave a partial rewrite. It also
permuted the old ADD output metadata, copied that metadata to the canonical
post output, and permuted the canonical tensor again, producing a double-
transpose metadata path.

The new owner matches from a MUL candidate and proves one exact private
NHWC-to-NCHW pre adapter, the MUL/ADD chain, and every private inverse post
adapter. The pre Transpose is removed only when that MUL is its last consumer;
other pre fan-out remains intact. The ADD output may have multiple post
adapters but no legacy consumers. Their downstream uses are redirected to the
first graph-ordered post output with exact multiplicity, preserving fan-out and
repeated slots before all post adapters are removed. The canonical post tensor
is not overwritten. Its shape, signature, logical layout, physical layout,
dtype, quantization, and provenance remain authoritative; the surviving MUL
intermediate adopts that contract.

Finite FLOAT16/FLOAT32/FLOAT64 scalar and rank-four constants are supported.
Raw NCHW channel, spatial, and full constants rotate once to NHWC, while
already-NHWC constants remain stable for idempotent recovery. Known layout
annotations disambiguate orientation. If direct and rotated non-invariant data
are both compatible because axes have equal lengths, the candidate is rejected
instead of guessing. MUL and ADD roles are grouped into one plan, so a shared
constant updates once and unrelated consumers receive deterministic `_nhwc`
clones. Produced, public, variable, quantized, non-finite, wrongly typed or
shaped constants reject the complete transaction.

Typed private permutation vectors, rank-four shape/signature permutations,
same floating dtype, no quantization, no fused activation, unique producers,
exact consumer multiplicity, dependency order, private intermediates, source
resolution, alias layouts, and final renaming are all checked with one
`ModelIRGraphIndex`. The plan is resolved again immediately before apply. The
candidate scan is graph ordered and capped at 32 rewrites; pruning and
LayoutState maintenance are explicit. The former 409-line helper is now a
17-line dispatcher, and all seven production calls pass LayoutState.

Focused coverage contains forty-eight FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across all operand orders and scalar, channel,
spatial, and full constants; dynamic signatures; already-NHWC idempotence;
canonical layout propagation; retained pre fan-out; multi-post alias merging
and repeated slots;
constants shared inside and outside the chain; candidate-only and capped
execution; forty-four transactional unsafe contracts; equal-axis ambiguity;
clone collision; and no-index preflight. The focused owner plus two existing
direct-builder characterizations passed with `104 passed in 0.53s`. The full
architecture suite passed with `181 passed in 50.32s`; three selected related
direct-builder tests passed with `3 passed, 752 deselected in 0.45s`;
TensorFlow-import-blocked import, direct conversion, and `-cotof` passed
sequentially with `3 passed in 4.01s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed affine post-ADD checkpoint moves
`_optimize_transpose_mul_posttranspose_add_nhwc_chains` to
`passes/affine_post_add_layout.py`. The legacy helper rebuilt full producer and
consumer maps inside an unbounded loop, collapsed sole-consumer multiplicity
through a set, mutated or cloned the MUL constant during matching, and
permuted the surviving MUL output metadata heuristically. It did not validate
dtype, quantization, fused activation, public boundaries, producer uniqueness,
constant provenance, exact graph order, or the complete post-ADD fan-out
before mutation.

The new owner resolves from a graph-ordered MUL candidate. It proves a typed
private NHWC-to-NCHW pre adapter, the MUL output's one exact typed inverse post
adapter, and every consumer of the private post output as a plain ADD with a
finite same-dtype scalar or exact `[1,1,1,C]` side constant. Multiple ADD tails
and their downstream repeated slots remain intact. The pre Transpose is kept
when another NCHW branch uses it. Otherwise both adapters are removed, all ADD
tails consume the surviving MUL output, and that output adopts the canonical
post tensor's shape, dynamic signature, logical layout, and physical layout.

MUL constants share the affine pre/post owner's finite FLOAT16/FLOAT32/FLOAT64
orientation contract. Scalar, raw NCHW channel/spatial/full, already-NHWC, and
legacy direct non-rank-four forms are retained. Ambiguous equal-axis
non-invariant rank-four tensors are rejected. A changed constant with an
unrelated consumer receives a deterministic `_nhwc` clone. The complete plan
is resolved again before apply, and clone names, input slots, and operator
removals are preflighted before any mutation. One differential index, a
32-rewrite ceiling, success-only pruning, and LayoutState synchronization
replace the legacy loop. The former 278-line helper is a 17-line dispatcher;
all four production calls pass LayoutState. The Pad compatibility wrapper
continues to dispatch only to its independent `passes/pad_layout.py` owner.

Focused coverage contains twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across both operand orders and scalar, channel,
spatial, and full MUL constants; dynamic signatures; two ADD tails; canonical
layout propagation; legacy already-NHWC/vector constant modes; retained pre
fan-out; unrelated constant cloning; candidate-only and capped execution;
twenty-three transactional unsafe contracts; equal-axis ambiguity; clone
collision; and no-index preflight. The focused owner plus two existing direct-
builder characterizations passed with `56 passed`; the full architecture suite
passed with `182 passed in 51.86s`; the selected direct-builder tests passed
with `2 passed, 753 deselected in 0.46s`; and TensorFlow-import-blocked direct,
default, and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.57s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed SiNet Shuffle residual checkpoint moves
`_optimize_sinet_shuffle_residual_transpose_chains` to
`passes/sinet_shuffle_residual_layout.py`. The legacy helper rebuilt full
producer and consumer maps in an unbounded loop and matched forward and
backward from the second post-Transpose with incomplete ownership checks. It
collapsed several consumer checks through sets, did not validate graph order,
dtype, quantization, fused activation, public intermediates, constant
provenance, concrete/dynamic Concat contracts, or downstream order, and
rotated/cloned six constants sequentially before the complete candidate was
known. Shared external constants could receive multiple redundant clones. It
also permuted intermediate and canonical post metadata heuristically.

The new owner roots at the terminal post-Transpose and resolves all thirteen
operators: three NHWC-to-NCHW input adapters, the first residual ADD/MUL/ADD/
PReLU, its side post adapter, the channel Concat, the second MUL/ADD/PReLU, and
the final post adapter. Every private edge has one unique producer, exact
consumer multiplicity, and valid dependency order. The first PReLU has exactly
the intended post and Concat branches; the second PReLU has exactly its final
post. Later consumers of the two post outputs, including fan-out and repeated
input slots, remain unchanged. The Concat input order and commuted ADD/MUL
operands are preserved while its axis moves from NCHW channel 1 to NHWC
channel 3.

Rank-four concrete shapes and dynamic signatures are proven through both
residual branches and the Concat. Unknown non-channel dimensions propagate
conservatively; channel signatures sum only when known. All data tensors share
one unquantized FLOAT16/FLOAT32/FLOAT64 dtype. Five typed private permutation
constants and NONE fused activations are required. Six finite scalar or exact
NCHW/NHWC channel constants cover both affine/PReLU stages. Roles sharing one
constant are grouped, so one update or one deterministic `_nhwc` clone serves
all uses, including a scalar shared across both stages. Produced, public,
variable, quantized, non-finite, wrongly typed, mismatched, or colliding
constants reject the complete plan.

Both post-Transpose tensors remain unchanged and authoritative. The first
stage intermediates adopt the first post tensor's exact shape, signature, and
layouts; the Concat and second-stage intermediates adopt the second post
contract. Both PReLUs produce the canonical names directly, after which all
five adapters are removed differentially. The plan is re-resolved before
apply, and clone names, mutation indices, output tensors, and removal indices
are preflighted before the first write. Candidate traversal is graph ordered
and capped at 32 rewrites; pruning and LayoutState synchronization occur only
after success. The former 482-line helper is a 17-line dispatcher and its one
production call supplies LayoutState.

Focused coverage contains twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across scalar/raw constants, commuted operands and
Concat inputs, and both legal post/Concat orders; dynamic signatures;
canonical layout propagation; two legacy already-NHWC modes; constants shared
within one stage or as one scalar across both stages; one-clone external
sharing; repeated downstream slots; candidate-only and capped execution;
fifty-eight transactional unsafe contracts; clone collision; no-index
preflight; differential index validation; and LayoutState validation. The
focused suite passed with `90 passed in 0.51s`; the full architecture suite
passed with `183 passed in 53.22s`; the selected adjacent SiNet direct-builder
characterization passed with `1 passed, 754 deselected in 0.48s`; and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed, 8 deselected in 3.65s`. Scoped Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The paired post-MUL SiNet checkpoint moves
`_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains` to the same
`passes/sinet_shuffle_residual_layout.py` owner. Its first nine operators are
identical to the full-tail island, so both variants now use one
`_resolve_prefix` implementation for the three input adapters, first
ADD/MUL/ADD/PReLU stage, side post adapter, channel Concat, shape/signature
proof, three constant roles, canonical first-post metadata, public boundaries,
fan-out, and graph order. This removes the risk that fixes to the formerly
duplicated residual prefix diverge between the two legacy helpers.

The variant-specific tail proves
`Concat(NCHW) -> MUL -> post-MUL Transpose(NHWC) -> ADD -> PReLU`. The MUL and
post output are private and have exact producer/consumer multiplicity; the ADD
has one exact PReLU consumer. The final PReLU output may remain a public output
or keep later graph-ordered fan-out. Concrete and dynamic contracts require
the MUL output to equal the NCHW Concat, and the post, ADD, and PReLU outputs to
share the exact NHWC permutation. All tail tensors share the prefix floating
dtype and are unquantized. Fused activations, duplicate producers, invalid
order, partial fan-out, public intermediates, and stale metadata reject the
complete plan.

The shared six-role constant transaction rotates or retains the first-stage
constants, the second MUL constant, and the already-NHWC ADD/PReLU constants
together. Sharing, external clones, constant provenance, dtype, finiteness,
shape, signature, quantization, collision, and variable state use the same
contract as the full-tail owner. The MUL produces the existing post-Transpose
name directly. The post tensor, ADD output, final PReLU output, and their
provenance remain untouched; only the Concat intermediate adopts the canonical
post contract. Both plans are re-resolved before apply, and all mutations and
five removals are preflighted before the first write.

Variant-focused coverage adds twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across scalar/mixed raw constants, commuted operands
and Concat inputs, and both legal post/Concat orders; two legacy raw tail
constant cases; external MUL-constant cloning; candidate-only and capped
execution; twenty-seven transactional unsafe tail contracts; clone collision;
and no-index preflight. The combined two-owner focused suite plus the existing
direct-builder characterization passed with `148 passed in 0.88s`; the full
architecture suite passed with `183 passed in 50.63s`; the selected SiNet
direct-builder characterization passed with
`1 passed, 754 deselected in 0.47s`; and TensorFlow-import-blocked direct,
default, and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.54s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed late-residual checkpoint moves
`_optimize_sinet_late_residual_pre_add_mul_add_prelu_chains` into the same
SiNet residual owner. The semantic candidate is two private
NHWC-to-NCHW adapters feeding `ADD -> MUL -> ADD -> PReLU`. Exactly one NHWC
source has a channel-last Concat producer. The PReLU output feeds one private
NCHW-to-NHWC adapter for a Conv2D or DepthwiseConv2D branch and one or more
later legacy NCHW consumers. The old fixed 40-by-40 predicate is gone; rank-
four concrete shapes and dynamic signatures must prove each permutation and
all affine intermediates must share one unquantized FLOAT16/FLOAT32/FLOAT64
contract.

The rewrite removes the two input adapters and lifts the affine/PReLU island
to NHWC. PReLU produces the existing canonical post tensor directly. The post
adapter is inverted in place and now produces the former PReLU tensor name, so
legacy consumers and repeated slots remain unchanged. Legacy consumers must
be graph ordered after that retained adapter; an earlier independent branch is
rejected instead of creating a producer-after-consumer edge. The canonical
post tensor remains authoritative and is never double-permuted. Only the
ADD/MUL/ADD intermediates adopt its exact shape, signature, and layouts; the
legacy output tensor retains its original NCHW metadata.

The three floating constants are grouped by tensor identity. Each must be a
finite, same-dtype, private constant that broadcasts in the original NCHW
graph; non-scalars are rotated and must also broadcast in the target NHWC
graph. This safely handles scalar and raw channel constants and retains the
legacy already-oriented rank-four case only when its actual axes make both
graphs valid. Unrelated consumers receive one deterministic clone. The
retained permutation constant participates in the same transaction and is
cloned when another Transpose still needs the original permutation.

The plan is re-resolved immediately before apply. Constant clones, mutation
indices, metadata targets, output-name swaps, and both adapter removals are
preflighted before the first write. One differential index, deterministic
candidate order, a 32-rewrite ceiling, success-only pruning, and LayoutState
synchronization replace the legacy full-map `while True` loop. The lowerer now
contains a 17-line dispatcher and the production call supplies the Session
LayoutState.

Focused coverage now passes with `207 passed in 0.69s`. It includes thirty-six
FLOAT16/FLOAT32/FLOAT64 numerical-equivalence combinations across both operand
orders, both downstream convolution families, scalar/raw constants, and the
formerly size-specific rank-four constant case on a non-40 spatial contract;
dynamic signatures; canonical, public, and repeated-slot legacy output
preservation; shared-role and external-use constant cloning; shared
permutation cloning; ambiguous oriented-constant rejection; candidate-only and
capped execution; fifteen transactional unsafe contracts; earlier legacy
consumer rejection; stale-plan revalidation; clone collision; no-index
preflight; differential-index validation; and LayoutState validation. The full
architecture suite passed with `183 passed in 50.11s`; the one sequential real
SiNet direct-builder characterization passed with `1 passed in 2.62s`; and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed with
`3 passed, 8 deselected in 3.58s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

## Failing tests and known issues

- No newly failing focused test is known at this checkpoint.
- A whole-file Ruff run on `pytorch_exporter.py` reports 282 pre-existing
  compatibility re-export, unused scaffold, and undefined-name findings. It is
  not used as the scoped checkpoint gate; changed owners/tests pass Ruff and
  the exporter passes syntax compilation.
- A whole-file Ruff run on `onnx2tf.py` reports pre-existing import-order,
  star-import, bare-except, undefined-name, and placeholder-f-string findings.
  This checkpoint uses the existing scoped exclusions for those categories;
  the changed helper owner/tests pass Ruff normally and all changed Python
  files pass syntax compilation.
- The optional PyTorch exporter suite runs when the host's Python 3.10
  `LD_LIBRARY_PATH` and `PYTHONPATH` are removed from the command environment.
  The focused results, restored native-codegen bindings, real-model artifact
  gate, and remaining inherited failures are recorded in
  `docs/flatbuffer_direct_pytorch_regression_2026-07-14.md`.
- The optional TensorFlow suite was not synchronized or run.
- Recent PyTorch source-policy checkpoints have not been followed by a Tier
  corpus conversion run. This is intentional under the current minimal-
  conversion instruction, but broad model-level regression remains unproven.

## Unfinished work

The full Goal is not complete. The fast-precanonicalize orchestrator still has
294 lines. Its remaining body is primarily the intended ordered helper
orchestration, source-line replacement, changed-flag handling, and the explicit
short-circuit boundaries required by the extracted policy decisions.

The broader fixed-pipeline, remaining artifact-plan coverage, artifact-matrix,
optional TensorFlow, PyTorch/TorchScript/Dynamo/ExportedProgram, and full Tier
regression work also remains subject to the original refactor plan and its
verification gates.

## Next work

1. Confirm `git status --short --branch` is clean and local `fb-refactor5`
   matches `origin/fb-refactor5`.
2. Treat `_optimize_transpose_swish_qdq_nhwc_islands` as a thin 69-line
   compatibility orchestrator unless a bounded phase-contract simplification
   is identified; all of its former raw top-level mutation loops now have
   indexed semantic owners.
3. Audit the adjacent 641-line
   `_optimize_sinet_deep_skip_concat_resize_affine_tail_chains` helper next.
   Characterize its deep-skip source ownership, Resize branch, two Concat
   boundaries, affine constants, terminal fan-out, and graph-order constraints
   before choosing the smallest complete semantic transaction. Reuse the SiNet
   constant, metadata, and residual-prefix contracts only where the topology
   proves they are identical.
4. Keep the terminal direct backend boundary explicit; do not reintroduce
   fallback into the legacy TensorFlow pipeline or broaden optional artifact
   execution.
5. Keep the audited 294-line PyTorch source orchestrator as explicit sequencing
   unless a new bounded decision is found.
6. Run only the focused synthetic/ownership/static checks unless the user asks
   for broader conversion validation. Use `uv`, run inference sequentially if
   any is explicitly requested, commit and push coherent units, and do not
   create a pull request.
