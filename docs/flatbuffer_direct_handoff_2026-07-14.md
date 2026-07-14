# `flatbuffer_direct` refactor continuation checkpoint — 2026-07-14

## Status

The active branch is `fb-refactor5`, created from `main` after pull request
`#949` merged the complete `fb-refactor4` checkpoint. Pull request `#950` is
closed, and no open pull request tracks this branch. The Goal is active again;
subsequent work uses coherent commits and pushes without opening a pull
request.

The latest implementation unit gives canonical
DEQUANTIZE-to-LOGISTIC-to-QUANTIZE cleanup one semantic owner in
`passes/quantized_logistic.py`. One maintained `ModelIRGraphIndex` supplies
graph-order Dequantize candidates, exact chain consumers/producers, indexed
Logistic edge rewrites, and differential wrapper removal. INT8/UINT8 input
grid validity, exact canonical output grid, shape/signature, Logistic
options/output identity, lineage, statistics, and pruning retain valid former
behavior. Near-canonical output scales, missing input quantization or float
bridge tensors, public bridges, duplicate producers, operator-order
violations, and inconsistent metadata are complete no-ops.
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

The current `fb-refactor5` work contains seventy-nine coherent continuations:

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
- the current checkpoint moves canonical quantized Logistic cleanup to one
  indexed owner with transactional topology, grid, and metadata guards.

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
- `onnx2tf/tflite_builder/passes/quantized_logistic.py`;
- `tests/test_flatbuffer_direct_indexed_quantized_logistic.py`;
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

The changed tests pass Ruff normally. The lowerer passes with its pre-existing
`F401` and `F841` findings scoped out. Every changed Python file passes
`python -m py_compile`, and `git diff --check` passes. The
immediately preceding DepthToSpace, Pool, dynamic-Pool, simple-alias, and
aligned-scalar checkpoints passed their focused synthetic and ownership
selections.

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
3. Audit `_optimize_dequant_softmax_quantize_chains` as the next bounded
   quantized-op cleanup. Preserve canonical output quantization, beta and
   Softmax options, public/fan-out, shape/signature, output identity,
   statistics, and pruning contracts while replacing its repeated producer/
   consumer maps with one maintained index and a complete pre-mutation plan.
4. Keep the terminal direct backend boundary explicit; do not reintroduce
   fallback into the legacy TensorFlow pipeline or broaden optional artifact
   execution.
5. Keep the audited 294-line PyTorch source orchestrator as explicit sequencing
   unless a new bounded decision is found.
6. Run only the focused synthetic/ownership/static checks unless the user asks
   for broader conversion validation. Use `uv`, run inference sequentially if
   any is explicitly requested, commit and push coherent units, and do not
   create a pull request.
