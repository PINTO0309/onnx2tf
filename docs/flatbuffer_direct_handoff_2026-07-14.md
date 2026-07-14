# `flatbuffer_direct` refactor continuation checkpoint — 2026-07-14

## Status

The active branch is `fb-refactor5`, created from `main` after pull request
`#949` merged the complete `fb-refactor4` checkpoint. Pull request `#950` is
closed, and no open pull request tracks this branch. The Goal is active again;
subsequent work uses coherent commits and pushes without opening a pull
request.

The latest implementation unit centralizes report and quantized-artifact
validation and completion logging across the two full direct-result
finalization paths. OP coverage, tensor correspondence, dynamic-range, INT8,
and both int16-activation variants now have one owner for required keys, skip
semantics, and failure messages. Tensor correspondence remains a default
compatibility artifact, and explicit call-site messages preserve the two
pre-existing dynamic-range log spellings. Split, SavedModel/PyTorch,
custom-op, and evaluation processing remain outside this helper because their
contexts are not identical.
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

The current `fb-refactor5` work contains forty coherent continuations:

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
- the current checkpoint centralizes direct report and quantized-artifact
  validation and completion logging without changing messages or skip
  behavior.

The extraction preserves the ordered source-rewrite behavior. Layout evidence
continues to mutate only the per-run CF/NHWC sets; repair context maps remain
shared. Rules that formerly used `continue` return an explicit short-circuit
result to the exporter. Exact generated-statement grammars remain rule-local or
use the shared Torch-free parser owner.

No dependency was added and no TensorFlow path was introduced. The latest
checkpoint includes three sequential direct-backend artifact smokes; no Tier
corpus run was performed.

## Current branch and changed files

Branch: `fb-refactor5`, tracking `origin/fb-refactor5`.

The current checkpoint changes:

- `onnx2tf/onnx2tf.py`;
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
2. Audit the third, reduced direct-result path separately. It intentionally
   performs only a subset of full finalization today; do not add report or
   quantization validation unless characterization proves that path should own
   those artifacts. Extract only behavior proven identical and preserve every
   legacy return key, message, and skip condition.
3. Add a focused production-boundary characterization before sharing another
   scope, and preserve exact diagnostics and rule order. Never carry a scope
   across a legacy helper or introduce a blanket refresh.
4. Keep the audited 294-line PyTorch source orchestrator as explicit sequencing
   unless a new bounded decision is found.
5. Run only the focused synthetic/ownership/static checks unless the user asks
   for broader conversion validation. Use `uv`, run inference sequentially if
   any is explicitly requested, commit and push coherent units, and do not
   create a pull request.
