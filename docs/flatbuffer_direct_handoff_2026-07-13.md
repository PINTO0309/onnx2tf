# flatbuffer_direct refactor handoff — 2026-07-13

## `fb-refactor4` rank-four bounded-family checkpoint

The first fourteen bounded families of the rank-four generic NHWC
pre-Concat matcher are now separated. `passes/nhwc_concat_layout.py` owns the
strict all-direct float path and the one-or-more-unary float path, with or
without direct inputs. The unary allowlist is RELU, RELU6, LOGISTIC, TANH, and
GELU. It also owns the one-or-more-Pad-plus-direct path and the one-or-more
Dequantize path and the one-or-more PReLU path, each with or without direct
inputs. It also owns exactly one Softmax plus at least one direct input, and
one or more expanded-Swish diamonds with direct or unary companion inputs.
The bounded Slice family additionally owns one or more exclusive direct-source
Slice inputs, optionally with direct inputs. The bounded Split family owns one
or more outputs from an exclusive direct-source Split, again optionally with
direct inputs. The bounded Add family owns the non-recursive direct-only Add
input form. The exact pseudo-LeakyRelu diamond is the eleventh family. All
eleven float-path families share one
`ModelIRGraphIndex`/`LayoutState` pass group and run transactionally under
stable IDs `layout.nhwc_pre_concat_direct` and
`layout.nhwc_pre_concat_unary`, `layout.nhwc_pre_concat_pad`, and
`layout.nhwc_pre_concat_dequantize`, `layout.nhwc_pre_concat_prelu`, and
`layout.nhwc_pre_concat_softmax`, `layout.nhwc_pre_concat_swish`, and
`layout.nhwc_pre_concat_slice`, plus `layout.nhwc_pre_concat_split` at all
seven production positions, followed by `layout.nhwc_pre_concat_add`.
The pseudo-LeakyRelu family runs last under
`layout.nhwc_pre_concat_leaky`.
The twelfth through fourteenth families are the separate direct, unary, and
Pad quantized-post passes `layout.nhwc_pre_concat_quantized_direct`,
`layout.nhwc_pre_concat_quantized_unary`, and
`layout.nhwc_pre_concat_quantized_pad` in
`passes/nhwc_concat_quantized_layout.py`.

The direct pass removes only exclusive, non-public leading adapters. Shared or
public direct adapters remain for their other consumers while the Concat is
rewired to the NHWC source. Public Concat/post tensors, invalid permutations
or ranks,
non-Transpose Concat fan-out, and wrong axes are rejected without mutation.
Stale spatial metadata remains accepted for this algebraically strict family,
matching the previous intentional behavior. Canonical per-axis quantization
now remaps NCHW dimension 1 to NHWC dimension 3. The unary family additionally
requires exclusive, non-public unary adapters and output, plus compatible
NHWC batch/spatial metadata, and remaps unary output quantization metadata.
The Pad family preserves optional Pad inputs, retains a shared leading
adapter, and remaps Pad output metadata. Exclusive pads constants are updated
in place; pads constants shared with any other operator or public boundary are
cloned and only the selected Pad input is rewired, preserving other consumers.
The Dequantize family does not rewrite scale or zero-point data: it bypasses
only the layout adapter, preserves source quantization provenance, and remaps
rank-four Dequantize output metadata and any per-axis dimension to NHWC.
PReLU preserves the legacy alpha candidate order for rank-4, unchanged, and
rank-3 constants. An exclusive transformed alpha is updated in place; a
shared or public alpha is cloned with dtype, variable flag, quantization,
layout, and ONNX provenance retained. Alpha and PReLU-output per-axis
dimensions move with their actual permutations.
Softmax retains its original NCHW last-axis semantics with two local,
self-inverse NHWC↔NHCW Transposes around the unchanged Softmax operator. The
old NCHW adapter and Concat post adapter are removed, so the eligible family
still reduces total Transpose count. New intermediate and final per-axis
quantization dimensions follow NHWC→NHCW, NCHW→NHCW, and NCHW→NHWC
permutations respectively.
The expanded-Swish family proves the complete `Logistic(x) * x` diamond and
accepts either Mul input order. Both Logistic and Mul are moved to the NHWC
source, with output shapes and per-axis quantization metadata remapped from
NCHW dimension 1 to NHWC dimension 3. The family rejects unsupported
operators, mismatched Mul data, invalid ranks or spatial metadata, raw
residual inputs, public adapter/Logistic/Mul outputs, and fan-out from any
internal edge before mutation. Rejecting a public Logistic output is an
intentional correctness improvement over the legacy matcher because rewriting
that public tensor would otherwise silently change its layout contract.
The bounded Slice family requires an exclusive rank-four NHWC→NCHW source
adapter and an exclusive rank-four Slice output. It remaps begin/size vectors,
Slice output shape, and per-axis quantization into NHWC. Exclusive parameter
tensors are updated once in place; shared or public parameters are cloned with
dtype, variable state, quantization, layout, and ONNX provenance preserved.
This closes two legacy correctness gaps: shared/public Slice parameters are no
longer silently mutated, and Slice-output quantization dimension 1 is remapped
to dimension 3. Shared source adapters and Slice outputs with valid inverse
post adapters deliberately remain in the legacy matcher so this bounded
ownership transfer cannot remove existing behavior.
The bounded Split family validates all outputs as rank four and requires each
to be unused or consumed only by the selected Concat. One Split may therefore
supply multiple Concat inputs while its source, axis, output metadata, and
quantization are rewritten exactly once. Negative channel axis `-3` and
positive axis `1` both canonicalize to NHWC axis `3`. Shared/public axis
tensors use the same provenance-preserving copy-on-write policy. This also
fixes the legacy omission of per-axis quantization remapping. Source-adapter
fan-out, output post adapters, and Add interactions remain available through
the legacy fallback.
The bounded Add family requires both operands to come through exclusive
rank-four NHWC→NCHW adapters and the Add output to feed only the selected
Concat. Both operands are rewired in their original order, all leading
adapters are removed, operator options are retained, and Add-output shape and
per-axis quantization are remapped once. Public/internal adapter boundaries
and invalid ranks now reject before mutation. Adapter sharing with the root
Concat, Add-output post adapters, unary operands, recursive Add, and other
mixed operand families deliberately remain in legacy.
The pseudo-LeakyRelu family proves the exact
`ReLU(x) - alpha * ReLU(-x)` topology. It accepts either Mul operand order,
requires scalar alpha, preserves Sub order, and supports direct or unary
Concat companions. Neg and positive Relu are rewired to the NHWC source; Neg,
both Relu outputs, Mul output, and Sub output shapes and quantization axes all
move to NHWC exactly once. This adds the alpha-first form that the legacy
matcher attempted but could not select. All public/fan-out internal edges,
rank errors, and partial diamonds reject before snapshot. Pad companions
deliberately remain in legacy.
The direct/unary/Pad quantized-post families validate
`adapters → optional bounded branch → Concat → Quantize → inverse Transpose(s)`
independently of the float group. They move Concat and supported branches to
NHWC, retain shared/public direct adapters, make the first post output
canonical, and rewire later aliases to it. Pad constants are reordered and
cloned with layout, quantization, variable-state, and ONNX provenance when
shared or public. The float and quantized paths use the same resolver and
materializer from `passes/nhwc_concat_pad.py`, preventing duplicate Pad rules
from drifting apart. Concat, branch, and quantized-output shapes and per-axis
metadata are remapped to NHWC; this fixes the legacy quantization-dimension
omission. Public boundaries, invalid ranks/spatial metadata, and non-Transpose
fan-out reject transactionally. All-Pad and other broader mixed quantized
inputs continue through legacy.

The lowerer compatibility helper still returns the original aggregate statistic
and runs the legacy matcher after the direct pass. The legacy matcher now
skips the fourteen indexed families, but continues to own broader
Split/Slice/Add/Leaky interactions and mixed quantized-post paths.

Changed files for this checkpoint:

- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/nhwc_concat_layout.py`
- `onnx2tf/tflite_builder/passes/nhwc_concat_pad.py`
- `onnx2tf/tflite_builder/passes/nhwc_concat_quantized_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_swish_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_slice_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_split_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_add_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_leaky_layout.py`
- `tests/test_flatbuffer_direct_nhwc_concat_quantized_layout.py`
- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`

Focused verification, all in the existing `uv` environment:

- Direct, unary, Pad, Dequantize, PReLU, Softmax, expanded-Swish,
  pseudo-LeakyRelu, and bounded Slice/Split/Add ModelIR characterization:
  `176 passed` across eight compact modules. Including the bounded direct and
  unary/Pad quantized-post suites, the compact inventory contains 218 tests
  across nine modules. The preceding combined run passed 208 tests; the
  expanded quantized module passes 42 tests, and the focused quantized/Pad
  selection after extracting the shared Pad plan passes 52 tests.
  The Softmax suite includes an exact NumPy equivalence check for the original
  and rewritten layouts. The Swish suite covers both Mul operand orders,
  all-Swish inputs, and fourteen whole-ModelIR unsafe/partial-match no-op
  boundaries. The Slice
  suite covers mixed and all-Slice success, shared/public parameter
  copy-on-write, fifteen complete no-op boundaries, and two broader cases that
  must continue through the legacy fallback. The Split suite covers both axis
  signs, multi-output single-application behavior, shared/public axis
  copy-on-write, fifteen no-op boundaries, and two preserved legacy cases. The
  Add suite covers mixed/all-Add success, fourteen complete no-op boundaries,
  and three broader cases retained in legacy. The pseudo-LeakyRelu suite
  covers both alpha operand orders, direct/unary/all-Leaky success, twenty
  complete no-op boundaries, and one Pad-mixed legacy fallback. The quantized
  suite covers canonical and multiple post outputs, shared/public adapter
  retention, all five supported unary operations, Pad-constant copy-on-write,
  and thirty no-op boundaries.
- Existing mixed-family NHWC matcher characterization: `5 passed`, `750`
  deselected.
- TensorFlow boundary and flatbuffer-direct architecture suite: `43 passed`.
- Ruff on the new pass and its compact test module: passed. A repository-wide
  Ruff gate is not configured; checking the pre-existing central lowerer also
  reports its known unused-import/local baseline.
- No ONNX corpus or large-model conversion was run for this checkpoint, per
  the instruction to minimize conversion testing and prioritize improvement.

Next work should characterize one remaining shared-adapter/post-adapter
Slice/Split/Add subfamily. Keep recursive
Add and mixed Swish/Add interactions in legacy until independently fixed. Do
not begin with a Tier 0–4 corpus run, and do not create a pull request.

The section below records the preceding rank-five checkpoint and remains as
historical context.

## `fb-refactor4` pause checkpoint — `1a343c5`

Work is paused at a clean implementation checkpoint. No new pull request must
be created on resume; use appropriately scoped commits and pushes to
`fb-refactor4` only. The local branch and `origin/fb-refactor4` are synchronized
at `1a343c5` (`index ndhwc pre concat layout pass`) before this handoff-only
commit.

### Completed work in `fb-refactor4`

- The managed Tier 0–4 profile runs in authoritative tier/model order and
  contains 420 recorded models: 382 active and 38 excluded from execution.
  The excluded set consists of 27 expected timeouts and 11 explicit user
  exclusions. The active baseline classifications are 356 pass, 20
  `tflite_fail`, and 6 `missing_tflite_report`.
- User-approved DEIM TopK index instability is accepted without discarding the
  raw maximum absolute errors or the unaccepted classification. The bulk
  result records both the managed acceptance and the underlying numeric
  result.
- `encoder.onnx` is classified as an expected 600-second timeout.
- The explicit future-validation exclusions are:
  `fast_acvnet_generalization_opset16_192x320.onnx`,
  `htdemucs_ft_onnx_1sec.onnx`, `maskrcnn_resnet50_fpn.onnx`, `model1.onnx`,
  `paddlepaddle_26_ocr.onnx`, `bread_180x320.onnx`,
  `bread_nonfm_180x320.onnx`, `double_gru.onnx`, `gtcrn_simple.onnx`,
  `conv_tasnet_dnn_ins.onnx`, and `spkrec-resnet-voxceleb.onnx`.
- The sequential bulk runner now samples Linux `VmSwap` for the active
  converter subprocess and all descendants. Any nonzero process-tree SWAP
  stops that model, records `swap_detected`, peak tree KiB, and per-process
  peaks, and leaves unrelated host-wide SWAP out of the decision. Future
  detected models must be added to the managed profile as `excluded` with
  reason `swap_detected_during_managed_validation` before the next managed
  run.
- The 258-line rank-five NDHWC pre-Concat matcher was mechanically moved to
  `passes/ndhwc_concat_layout.py` at `8908a90`. Its extracted function AST
  exactly matched the characterized central implementation with SHA-256
  `0b0c625290f2ed31351ca204b0bbc5f2a463fa09ffe1bf1eccb8ff15de6aee17`.
- Checkpoint `1a343c5` replaced its repeated producer/consumer maps and direct
  operator deletion with pure `ModelIRGraphIndex` candidate planning,
  differential mutation, `LayoutState` reconciliation, and transactional pass
  ID `layout.ndhwc_pre_concat`. All five raw production calls now use the
  stable runner. Per-axis quantization metadata is cloned and remapped from
  NCDHW dimension 1 to NDHWC dimension 4 for unary and canonical Concat
  tensors.
- The 2,000 threshold remains only the Tier 5 ONNX node-count boundary. It is
  not a source-file line limit.

### Unfinished work

- Continue staged characterization and indexed migration of the remaining
  central layout families. The adjacent rank-four generic
  `_optimize_transpose_pre_concat_nhwc_chains` matcher is much larger and must
  first be audited and divided into semantic characterization units; do not
  treat it as one blind monolithic extraction.
- Complete remaining lowerer/registry decomposition and consolidate op-family
  validation, capability selection, and lowering.
- Complete fixed-ModelIR coverage for quantization modes, split/crop,
  custom/pseudo ops, weights, reports, and requested-artifact-only execution.
- Complete shared PyTorch, TorchScript, Dynamo ONNX, and ExportedProgram
  canonicalization/emitter decomposition.
- Complete the public CLI/Python and artifact matrix audits, optional
  TensorFlow exporter compatibility, TensorFlow-free direct/`-cotof` boundary,
  remaining tier gates, normalized failure comparison, and three-run median
  timing/peak-RSS measurements.
- A full current Tier 0–4 run was intentionally not completed. The latest user
  direction is to keep conversion tests minimal and prioritize implementation
  improvements. Do not restart a whole-corpus run as the first resumed task.
- The previously noted DPT producer-rank investigation and any other
  corpus-only candidates remain unproven until a focused reproducer justifies
  work; do not infer broad non-regression from the partial run.

### Branch and changed files

- Branch: `fb-refactor4`
- Implementation checkpoint: `1a343c5`
- Local/remote divergence before this documentation commit: `0 0`
- Worktree before this documentation commit: clean
- Pull requests: do not create one on resume

Files changed by the `fb-refactor4` checkpoints covered here:

- `docs/baselines/flatbuffer_direct_active_tier0_4.json`
- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`
- `onnx2tf/utils/flatbuffer_direct_bulk_runner.py`
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/ndhwc_concat_layout.py`
- `tests/test_flatbuffer_direct_bulk_runner.py`
- `tests/test_flatbuffer_direct_architecture.py`
- `tests/test_flatbuffer_direct_ndhwc_concat_layout.py`

### Tests executed and results

All inference remained sequential with one model process at a time and all
commands used the existing `uv` environment.

- Managed-profile/bulk-runner focused suite after the two latest explicit
  exclusions: `34 passed`.
- SWAP monitoring, bulk-runner, and architecture focus: `78 passed`.
- Actual `Acos_11.onnx` SWAP-monitor smoke: pass,
  `swap_detected=false`, peak SWAP `0 KiB`, maximum absolute error
  `7.972121238708496e-07`.
- NDHWC mechanical extraction characterization plus architecture:
  `59 passed`.
- Indexed NDHWC characterization, transaction metrics, ownership, and
  architecture: `60 passed`.
- Differential comparison against checkpoint `8908a90`: all 16 cases (one
  success and fifteen unsafe boundaries) produced identical non-quantized
  ModelIR and statistics.
- Sequential `superpoint.onnx` direct/`-cotof` smoke:
  `evaluation_pass=true`, maximum absolute error
  `1.6666017472743988e-06`, RMSE `1.6207873294228388e-07`, cosine similarity
  `1.0`. The new rank-five pass skipped all five positions with zero snapshots
  and fingerprints on this unrelated rank-four graph.
- Ruff check of the touched NDHWC pass and tests: passed.
- The intentionally interrupted current-profile Tier 0 run completed 37 of
  382 active models: all 37 passed and none reported process-tree SWAP. This is
  diagnostic evidence only, not a complete tier gate.

### Failing tests and known issues

- No focused test is failing at this checkpoint.
- No process-tree SWAP was detected in the 37-model partial run. Existing
  host-wide SWAP was deliberately ignored and is not evidence against a model.
- The managed profile still intentionally records 20 numeric failures and 6
  missing reports among active models, plus 27 expected timeouts. These are
  known baseline classifications, not new failures from the NDHWC migration.
- Current broad corpus non-regression and performance targets are not proven by
  the deliberately minimal validation scope.
- Temporary copied ONNX, generated TFLite, and schema artifacts from the
  interrupted run were deleted. Its JSON/log evidence remains under
  `/tmp/onnx2tf_tier0_4_fb4_ccf5277` (about 7.5 MiB at pause time).

### First work on resume

1. Confirm `git status --short --branch` is clean and local/remote divergence
   is `0 0` after the handoff commit.
2. Do not start a whole Tier 0–4 conversion run. Re-run only the 60-test NDHWC
   focused gate if the environment or base commit changed.
3. Audit the rank-four `_optimize_transpose_pre_concat_nhwc_chains` matcher and
   its existing tests. Select one bounded semantic subfamily and add compact
   success/no-op characterization before moving implementation.
4. Preserve the staged sequence: characterization → exact mechanical
   extraction → indexed transactional runner. Use one representative model at
   most when the selected pass has a known exercising model.
5. If any future sequential model run reports `swap_detected`, immediately add
   that model to the managed profile exclusion set, update exact count tests,
   and start any later authoritative run with a clean output directory.
6. Commit and push each safe checkpoint to `fb-refactor4`; do not open a pull
   request.

## Pause checkpoint

- Branch: `fb-refactor3`
- Latest implementation checkpoint: `9a09553`
  (`characterize ndhwc pre concat layout`)
- Previous pause checkpoint: `3df2903`
  (`document flatbuffer direct pause checkpoint`)
- Remote: after this resumed documentation checkpoint is pushed, local and
  `origin/fb-refactor3` must again report `0 0` divergence.
- Pull request: none; do not create one on resume
- The axis-3 constant-Concat bridge matcher uses pure indexed planning,
  differential graph/layout mutation, and one stable transactional runner.
  Its lowerer wrapper remains; the raw production call is removed.
- The Dequantize/Concat/Quantize matcher uses pure indexed planning,
  differential graph/layout mutation, and one transactional runner at both
  production positions.
- The Concat/optional-unary/post-adapter/Conv matcher uses pure indexed
  planning, differential graph/layout mutation, and one transactional runner
  at both production positions.
- The seven-call SPP family has a generic characterization corpus and its
  complete matcher is owned by `passes/spp_layout.py`. Pure indexed planning,
  shared-constant copy-on-write, differential graph/layout mutation, and
  stable runner `layout.generic_spp_nhwc` replace all seven raw calls.
- The five-call NDHWC pre-Concat matcher now has a dedicated 16-case compact
  characterization corpus. Production remains central and unchanged.

## Completed work

This resumed interval completed sixteen adjacent semantic layout families
using the staged characterization → mechanical extraction → indexed runner
process and completed the same three-stage migration for the seventeenth
family. Characterization is complete for the eighteenth family.

1. Mean layout
   - Characterized the long Mean/Mul/Reshape/Add/Conv success path and Mean
     fan-out rejection in `c99418a`.
   - Moved both Mean layout matchers to `passes/mean_layout.py` in `efb15cd`.
   - Added differential graph-index/layout-state mutation and stable ordered
     runners in `06a9dbd`.
2. LayerNorm statistics layout
   - Characterized pre-Transpose removal, existing post-Transpose reuse, and
     centered-value fan-out rejection in `d7866d2`.
   - Moved both matchers to `passes/layernorm_layout.py` in `267126a`.
   - Replaced both adjacent raw call pairs with one shared-state two-spec
     runner in `bffde62`.
3. Terminal unary/Mean layout
   - Characterized shared-pre retention, unary fan-out rejection, and
     inverse-Transpose-tail deferral in `8bce913`.
   - Moved the complete matcher to `passes/terminal_mean_layout.py` with an
     identical AST in `92446d7`.
   - Added stable runner `layout.terminal_unary_mean_reshape`, differential
     indexing, indexed preconditions, layout synchronization, and all six
     production runner calls in `a872774`.
4. EfficientNet-style SE layout
   - Characterized shared-input, gate-fan-out, and public-boundary behavior in
     `5f5de07`, then moved both complete matchers to `passes/se_layout.py` in
     `ce519b6`.
   - Indexed SE-Conv and SE-FC independently in `a9c6971` and `817bfa0`, using
     shared graph/layout state and stable ordered runners at all production
     call sites.
5. Elementwise gate layout
   - Added missing SUM/Logistic/Sub/Mul/Add boundary characterization in
     `1623762` and mechanically extracted four rules to
     `passes/elementwise_gate_layout.py` in `2095a01`.
   - Indexed the four-rule group under stable ordered pass IDs, replacing five
     repeated raw call groups in `9832355`.
6. Generic multi-branch gate layout
   - Replaced model-specific coverage with a compact generic two-branch
     success/rejection corpus in `13ec048`.
   - Mechanically extracted the complete matcher to
     `passes/multi_branch_gate_layout.py` with AST equivalence in `42bb3e8`.
   - Migrated all reads and mutations to shared graph/layout state, added the
     ordered runner, and replaced the single production call in `b0d1248`.
7. Complementary dual-postconv gate layout
   - Added a generic two-output success fixture and gate-fan-out,
     data-adapter-fan-out, and public-intermediate no-op boundaries in
     `ed6d8c1`.
   - Mechanically moved the complete implementation to
     `passes/dual_postconv_gate_layout.py` with AST equivalence in `8d149cb`.
   - Migrated the matcher and all five production positions to shared indexed
     state and a stable ordered runner in `cc828c8`.
8. Complementary postadd gate layout
   - Added generic Add/Conv-tail success and gate-fan-out,
     data-adapter-fan-out, and public-intermediate no-op characterization in
     `ea78747`.
   - Mechanically moved the complete matcher beside its complementary-gate
     sibling with AST equivalence in `f961413`.
   - Integrated it as the second ordered spec over a shared complementary-gate
     prefix and removed all five raw calls in `78b0742`.
9. Rank-five Leaky/Logistic gate layout
   - Replaced the 177-line central inline fixture with a dedicated compact
     success graph and five unsafe-boundary cases in `ee3d2fd`.
   - Mechanically moved the complete matcher to
     `passes/ndhwc_gate_layout.py` with AST equivalence in `332612f`.
   - Migrated all mutation and six production calls to a stable indexed runner
     in `2871ade`.
10. Conv3D/Leaky/Unsqueeze gate layout
    - Replaced a 176-line central fixture with compact 4D/5D success variants
      and six unsafe-boundary cases in `49c72b9`.
    - Mechanically moved the complete matcher beside its NDHWC sibling with AST
      equivalence in `ae3c00b`.
    - Migrated producer/consumer reads, rewrites, structural removals, pruning,
      and layout synchronization to shared indexed state in `a470cce`.
    - Registered stable ordered pass ID
      `layout.ndhwc_conv3d_leaky_unsqueeze_gate` after the rank-five gate,
      removed all six raw production calls, and retained the compatibility
      wrapper.
11. Cost-volume/ScatterND layout
    - Replaced the 177-line central success fixture with a dedicated compact
      corpus in `4b6f297`.
    - Added six whole-ModelIR no-op boundaries covering leading-adapter
      fan-out, pre/post sides of the trailing adapter, a public intermediate,
      invalid leading permutation, and an invalid downstream operator.
    - Preserved production behavior at the characterization checkpoint.
    - Moved the complete matcher to
      `passes/cost_volume_scatter_layout.py` with AST equivalence in `d62e77d`;
      the compatibility wrapper and all six raw call positions remain.
    - Added pure indexed topology and constant planning, transactional runner
      `layout.cost_volume_scatter_ndhwc`, shared graph/layout mutation, and six
      production runner calls in `56516ef`.
    - Fixed late-validation partial mutation for invalid ScatterND shape,
      coordinate rank, and out-of-bounds coordinates; all now reject before a
      snapshot and preserve the complete ModelIR.
12. Add/Concat/constant-suffix layout
    - Added the first dedicated success corpus for the previously untested
      central matcher in `fcf24b2`.
    - Fixed nine complete no-op boundaries covering branch/Add/Concat/Mul
      fan-out, public intermediate/post output, invalid permutation/axis, and
      missing suffix constant.
    - Preserved production behavior and all five raw call positions;
      mechanical extraction remains the next checkpoint.
    - Moved the complete matcher to `passes/add_concat_suffix_layout.py` with
      AST equivalence in `73f96ca`; the compatibility wrapper and all five raw
      production positions remain.
    - Added shared indexed candidate/mutation state, suffix-constant
      copy-on-write, corrected post-tensor metadata, stable transactional runner
      `layout.add_concat_const_suffix_nhwc`, and five runner calls in `1b8c307`.
13. Dual-Mul/Concat layout
    - Moved the 131-line central success fixture to a dedicated compact corpus
      in `82d8777`, retaining shared-constant copy-on-write coverage.
    - Added ten whole-ModelIR no-op boundaries for adapter/Mul/Concat fan-out,
      public tensors, permutations, axis, missing constant, and non-shared data
      branches.
    - Preserved production behavior and all six raw call positions;
      mechanical extraction remains the next checkpoint.
    - Moved the complete matcher to `passes/dual_mul_concat_layout.py` with AST
      equivalence in `af26412`; the compatibility wrapper and all six raw
      positions remain.
    - Added pure indexed topology/broadcast planning, differential
      copy-on-write and graph/layout mutation, corrected post metadata, stable
      runner `layout.dual_mul_concat_nhwc`, and six runner calls in `64702b2`.
14. Axis-3 constant-Concat bridge characterization
    - Moved the sole 132-line central success fixture to
      `tests/test_flatbuffer_direct_axis3_const_concat_layout.py` in `019d3c6`.
    - Added compact success variants for multiple inverse post branches and a
      safely shared leading adapter, while retaining the legacy NCHW-consumer
      bridge case.
    - Added nine complete no-op boundaries for public Concat/post tensors,
      invalid pre/post permutations, invalid axis, constant rank/shape/data,
      and a constant shared outside the Concat.
    - Preserved the central production matcher and its single call exactly;
      extraction is intentionally deferred to the next checkpoint.
    - Moved the complete matcher to
      `passes/axis3_const_concat_layout.py` with exact AST equivalence in
      `5228444`; the compatibility wrapper and single raw production call
      remain.
    - Added pure indexed constant/topology/bridge planning, protected public
      adapter and constant boundaries, differential removal/insertion,
      `LayoutState` reconciliation, stable runner
      `layout.axis3_const_concat_bridge_nhwc`, and the single production runner
      call in `a261462`.
15. Dequantize/Concat/Quantize layout characterization
    - Added the first dedicated corpus for the previously untested central
      matcher in `ea74ffd`.
    - Fixed ordinary rewrite, multiple post-adapter canonicalization, shared
      pre-adapter retention, and quantization metadata preservation.
    - Added twelve complete no-op boundaries covering intermediate fan-out,
      public tensors, invalid permutations/axis, and a non-Dequantize branch.
    - Preserved the central production matcher and both raw calls exactly;
      extraction is the next checkpoint.
    - Moved the complete matcher to
      `passes/dequant_concat_quantize_layout.py` with exact AST equivalence in
      `35a4cb1`; the compatibility wrapper and both raw calls remain.
    - Added pure indexed topology and quantization-metadata planning,
      differential graph/layout mutation, stable runner
      `layout.dequant_concat_quantize_nhwc`, and both production runner calls
      in `3be0c3e`.
16. Concat/unary/Conv layout characterization
    - Added the first dedicated compact corpus for the central matcher in
      `f624388`.
    - Fixed unary-free and two-unary/two-post success variants, including
      Conv2D and DepthwiseConv2D consumers.
    - Added thirteen complete no-op boundaries for fan-out, graph outputs,
      permutations, axis, input/unary type, and non-Conv consumers.
    - Preserved production code and both raw calls exactly; extraction remains
      the next checkpoint.
    - Moved the complete matcher to `passes/concat_unary_conv_layout.py` with
      exact AST equivalence in `11e76bd`; the wrapper and both calls remain.
    - Added pure indexed adapter/unary/post/Conv planning, rank-four guards,
      differential graph/layout mutation, stable runner
      `layout.concat_unary_conv_nhwc`, and both production runner calls in
      `b86b31a`.
17. Generic two-island SPP layout characterization
    - Added the first dedicated corpus for the 371-line, seven-call matcher in
      `0804e37`.
    - Replaced implicit model-specific coverage with a compact four-branch,
      two-Concat, two-affine, two-Conv semantic graph.
    - Added sixteen complete no-op boundaries for fan-out, public tensors,
      permutation/axis, Resize producer, and missing Mul constants.
    - Preserved production code and all seven raw calls exactly at the
      characterization checkpoint.
    - Moved the complete matcher mechanically to `passes/spp_layout.py` in
      `c531b54`. Its function AST exactly matches characterization checkpoint
      `0804e37`; the lowerer retains a compatibility wrapper and all seven raw
      production calls.
    - Added full indexed topology/rank/constant planning, shared-constant
      copy-on-write, quantized-dimension remapping, differential graph/layout
      mutation, stable runner `layout.generic_spp_nhwc`, and all seven runner
      calls in `8edf5c2`.
18. NDHWC pre-Concat layout characterization
    - Moved the only 96-line central success fixture to
      `tests/test_flatbuffer_direct_ndhwc_concat_layout.py` in `9a09553`.
    - Extended success coverage to mixed direct/unary inputs with two inverse
      post adapters and canonical alias replacement.
    - Added fifteen complete no-op boundaries covering input/unary/Concat
      fan-out, public tensors, invalid permutations/axis, unsupported unary,
      invalid rank, and incompatible spatial shape.
    - Preserved the complete 258-line production matcher and all five raw
      production calls; mechanical extraction is the next checkpoint.

Compatibility wrappers remain in `lower_from_onnx2tf.py` for all extracted
families. Every implementation migrated through the indexed-runner stage
contains no whole-graph producer/consumer map construction and no direct
operator-list insertion/deletion. No dependency or TensorFlow import path was
added.

The 2,000 threshold is only the Tier 5 ONNX node-count boundary. It is not a
source-file line limit and no source-line gate should be introduced.

## Unfinished work

The overall Goal is not complete. In particular:

- Continue staged extraction/indexing of the remaining legacy layout rules.
  The immediate next unit is exact-AST mechanical extraction of the 258-line,
  five-call `_optimize_transpose_pre_concat_ndhwc_chains` matcher. Keep the
  much larger generic NHWC pre-Concat matcher as a separately planned family;
  source length is not a Goal gate.
- Complete the remaining central lowerer/registry decomposition and consolidate
  op-family validation, capability selection, and lowering.
- Reconnect and exhaustively validate quantization, split/crop, custom/pseudo
  ops, weights, and requested-artifact-only execution on the fixed ModelIR
  contract.
- Complete the planned PyTorch, TorchScript, Dynamo ONNX, and ExportedProgram
  exporter decomposition.
- Run the fixed corpus gates sequentially through Tier 0–Tier 5. Tier 5 means
  models with at least 2,000 ONNX nodes and remains a late-stage gate.
- Complete the artifact matrix, optional TensorFlow exporter compatibility,
  full public CLI/Python contract audit, and final baseline/failure-signature
  comparison.
- Measure three-run median conversion time and peak RSS by tier, enforce the
  +10% non-regression limit, and evaluate the Tier 4 15% speedup target.
- Produce the final requirement-by-requirement completion audit and developer
  documentation. Do not mark the Goal complete until every original plan item
  has direct evidence.

## Branch and changed files

Current branch is `fb-refactor3`. Before this resumed documentation update, the
working tree is clean at NDHWC characterization checkpoint `9a09553`. The
indexed SPP work and the dedicated NDHWC pre-Concat corpus are committed;
NDHWC mechanical extraction has not begun.
After the documentation commit is pushed, local/remote divergence must be
`0 0`. The implementation checkpoints since the previous pause changed:

- `docs/flatbuffer_direct_architecture.md`
- `docs/flatbuffer_direct_handoff_2026-07-12.md`
- `docs/flatbuffer_direct_handoff_2026-07-13.md`
- `onnx2tf/tflite_builder/core/model_ir_utils.py`
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`
- `onnx2tf/tflite_builder/passes/elementwise_gate_layout.py`
- `onnx2tf/tflite_builder/passes/axis3_const_concat_layout.py`
- `onnx2tf/tflite_builder/passes/dequant_concat_quantize_layout.py`
- `onnx2tf/tflite_builder/passes/concat_unary_conv_layout.py`
- `onnx2tf/tflite_builder/passes/spp_layout.py`
- `onnx2tf/tflite_builder/passes/dual_postconv_gate_layout.py`
- `onnx2tf/tflite_builder/passes/multi_branch_gate_layout.py`
- `onnx2tf/tflite_builder/passes/ndhwc_gate_layout.py`
- `onnx2tf/tflite_builder/passes/se_layout.py`
- `tests/test_flatbuffer_direct_architecture.py`
- `tests/test_flatbuffer_direct_elementwise_gate_layout.py`
- `tests/test_flatbuffer_direct_dual_postconv_gate_layout.py`
- `tests/test_flatbuffer_direct_3d_gate_layout.py`
- `tests/test_flatbuffer_direct_osnet_gate_layout.py`
- `tests/test_flatbuffer_direct_pass_efficiency.py`
- `tests/test_flatbuffer_direct_se_layout.py`
- `tests/test_flatbuffer_direct_axis3_const_concat_layout.py`
- `tests/test_flatbuffer_direct_concat_unary_conv_layout.py`
- `tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py`
- `tests/test_flatbuffer_direct_ndhwc_concat_layout.py`
- `tests/test_flatbuffer_direct_spp_layout.py`
- `tests/test_tflite_builder_direct.py`

The resumed handoff checkpoint updates documentation only. No implementation
or test file remains uncommitted after it.

## Tests executed

All commands ran in the existing `uv` environment. Inference was strictly
sequential with only one model/process active at a time.

- Terminal Mean characterization plus legacy tests: `5 passed`.
- Terminal Mean extraction/ownership focus: `6 passed`.
- Terminal Mean indexed runner, architecture, and efficiency focus:
  `40 passed`.
- Full direct selection after mechanical extraction:
  `1167 passed, 5 deselected, 2 warnings in 151.84s`.
- Tier 1 `superpoint.onnx`, sequential
  `-tb flatbuffer_direct -cotof` after indexed migration:
  `evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1170 passed, 5 deselected, 2 warnings in 162.39s`.
- SE characterization/extraction/indexed focused gates passed; the final SE
  full direct selection was
  `1178 passed, 5 deselected, 2 warnings in 158.98s`.
- Elementwise-gate characterization/extraction/indexed focused gates passed;
  the final elementwise full direct selection was
  `1183 passed, 5 deselected, 2 warnings in 152.05s`.
- Multi-branch gate characterization and extraction focused gate passed
  `3 tests`; its full direct selection was
  `1186 passed, 5 deselected, 2 warnings in 151.88s`.
- Pause verification of multi-branch characterization plus architecture:
  `35 passed in 17.23s`.
- Indexed multi-branch runner, architecture, and efficiency focus:
  `41 passed in 18.59s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed multi-branch migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed multi-branch migration:
  `1188 passed, 5 deselected, 2 warnings in 161.24s`.
- Dual-postconv gate compact characterization: `4 passed`.
- Dual-postconv extraction and ownership focus: `38 passed in 18.99s`.
- Full direct selection after mechanical extraction:
  `1193 passed, 5 deselected, 2 warnings in 169.93s`.
- Indexed dual-postconv runner, architecture, and efficiency focus:
  `46 passed in 19.29s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed dual-postconv migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed dual-postconv migration:
  `1197 passed, 5 deselected, 2 warnings in 161.48s`.
- Postadd complementary-gate compact characterization: `4 passed`.
- Postadd extraction, sibling, and ownership focus:
  `46 passed in 18.04s`.
- Full direct selection after mechanical extraction:
  `1201 passed, 5 deselected, 2 warnings in 157.71s`.
- Indexed postadd/complementary-gate runner, architecture, and efficiency
  focus: `54 passed in 17.73s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed postadd migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed postadd migration:
  `1205 passed, 5 deselected, 2 warnings in 154.34s`.
- Dedicated rank-five 3D gate characterization: `6 passed`.
- 3D extraction and ownership focus: `41 passed in 17.69s`.
- Full direct selection after 3D mechanical extraction:
  `1211 passed, 5 deselected, 2 warnings in 157.23s`.
- Indexed 3D runner, architecture, and efficiency focus:
  `51 passed in 18.61s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed 3D migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed 3D migration:
  `1217 passed, 5 deselected, 2 warnings in 156.83s`.
- Dedicated Conv3D gate characterization: `8 passed`.
- Conv3D extraction, sibling, and ownership focus:
  `55 passed in 17.93s`.
- Full direct selection after Conv3D mechanical extraction:
  `1224 passed, 5 deselected, 2 warnings in 157.08s`.
- Indexed Conv3D runner, rank-five sibling, architecture, and efficiency
  focus: `67 passed in 17.27s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed Conv3D migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed Conv3D migration:
  `1232 passed, 5 deselected, 2 warnings in 154.69s`.
- Dedicated cost-volume/ScatterND characterization: `7 passed in 0.29s`.
- Cost-volume/ScatterND focused selection with the central module present:
  `7 passed, 758 deselected in 2.72s`.
- Full direct selection after moving the fixture and adding six boundaries:
  `1238 passed, 5 deselected, 2 warnings in 153.83s`.
- Cost-volume/ScatterND extraction, characterization, and ownership focus:
  `43 passed in 18.64s`; the extracted function AST exactly matched
  `4b6f297`.
- Full direct selection after mechanical extraction:
  `1239 passed, 5 deselected, 2 warnings in 156.55s`.
- Indexed cost-volume/ScatterND runner, late-validation, architecture, and
  efficiency focus: `60 passed in 18.58s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed cost-volume/ScatterND migration:
  `1252 passed, 5 deselected, 2 warnings in 164.88s`.
- Dedicated Add/Concat/constant-suffix characterization: `10 passed in 0.32s`.
- Full direct selection after adding the previously missing success and nine
  unsafe-boundary cases:
  `1262 passed, 5 deselected, 2 warnings in 195.24s`.
- Add/Concat/constant-suffix extraction, characterization, and ownership focus:
  `47 passed in 18.70s`; the extracted function AST exactly matched
  `fcf24b2`.
- Full direct selection after mechanical extraction:
  `1263 passed, 5 deselected, 2 warnings in 158.07s`.
- Indexed Add/Concat suffix runner, shared-constant, architecture, and
  efficiency focus: `63 passed in 19.43s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed Add/Concat suffix migration:
  `1275 passed, 5 deselected, 2 warnings in 158.36s`.
- Dedicated dual-Mul/Concat characterization: `11 passed in 0.29s`.
- Focused dual-Mul selection including residual central-name coverage:
  `12 passed, 756 deselected in 2.71s`.
- Full direct selection after moving the fixture and adding ten boundaries:
  `1285 passed, 5 deselected, 2 warnings in 165.06s`.
- Dual-Mul/Concat extraction, characterization, and ownership focus:
  `49 passed in 20.24s`; the extracted function AST exactly matched `82d8777`.
- Full direct selection after mechanical extraction:
  `1286 passed, 5 deselected, 2 warnings in 163.50s`.
- Indexed dual-Mul/Concat runner, broadcast-plan, architecture, and efficiency
  focus: `64 passed in 20.85s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed dual-Mul/Concat migration:
  `1297 passed, 5 deselected, 2 warnings in 165.59s`.
- Dedicated axis-3 constant-Concat bridge characterization:
  `12 passed in 0.29s`.
- Residual central selection after moving the fixture:
  `756 deselected in 0.37s`; no duplicate central test remains.
- Full direct selection after characterization:
  `1308 passed, 5 deselected, 2 warnings in 166.51s`.
- Axis-3 constant-Concat extraction, characterization, and architecture focus:
  `51 passed in 20.60s`; the extracted function AST exactly matched `019d3c6`.
- Full direct selection after mechanical extraction:
  `1309 passed, 5 deselected, 2 warnings in 165.14s`.
- Indexed axis-3 constant-Concat runner, public-boundary, architecture, and
  irrelevant-graph efficiency focus: `69 passed in 21.95s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1323 passed, 5 deselected, 2 warnings in 166.52s`.
- Dedicated Dequantize/Concat/Quantize characterization:
  `15 passed in 0.34s`.
- Full direct selection after characterization:
  `1338 passed, 5 deselected, 2 warnings in 166.21s`.
- Dequantize/Concat/Quantize extraction, characterization, and architecture
  focus: `55 passed in 20.93s`; the extracted function AST exactly matched
  `ea74ffd`.
- Full direct selection after mechanical extraction:
  `1339 passed, 5 deselected, 2 warnings in 176.09s`.
- Indexed Dequantize/Concat/Quantize runner, metadata/rank boundary,
  architecture, and irrelevant-graph efficiency focus:
  `78 passed in 21.88s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1358 passed, 5 deselected, 2 warnings in 173.55s`.
- Dedicated Concat/unary/Conv characterization: `15 passed in 0.31s`.
- Full direct selection after characterization:
  `1373 passed, 5 deselected, 2 warnings in 172.58s`.
- Concat/unary/Conv extraction, characterization, and architecture focus:
  `56 passed in 23.44s`; the extracted function AST exactly matched `f624388`.
- Full direct selection after mechanical extraction:
  `1374 passed, 5 deselected, 2 warnings in 175.23s`.
- Indexed Concat/unary/Conv runner, rank boundary, architecture, and
  irrelevant-graph efficiency focus: `76 passed in 24.25s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed migration:
  `1390 passed, 5 deselected, 2 warnings in 173.99s`.
- Dedicated generic SPP characterization: `17 passed in 0.33s`.
- Full direct selection after characterization:
  `1407 passed, 5 deselected, 2 warnings in 176.30s`.
- SPP characterization plus architecture ownership after mechanical
  extraction: `59 passed in 24.76s`; the extracted function AST exactly
  matched `0804e37`.
- Full direct selection after SPP mechanical extraction:
  `1408 passed, 5 deselected, 2 warnings in 174.80s`.
- Indexed SPP success, shared-constant copy-on-write, quantized-dimension,
  no-op, runner, architecture, and irrelevant-graph efficiency focus:
  `85 passed in 23.15s`.
- Tier 1 `superpoint.onnx`, sequential `-tb flatbuffer_direct -cotof` after
  indexed SPP migration: `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.
- Full direct selection after indexed SPP migration:
  `1430 passed, 5 deselected, 2 warnings in 161.91s`.
- Dedicated NDHWC pre-ConCat characterization: `16 passed in 0.30s`.
- Full direct selection after moving the central fixture and adding fifteen
  no-op boundaries:
  `1445 passed, 5 deselected, 2 warnings in 170.91s`.
- Tier 1 `superpoint.onnx` was run sequentially after both indexed SE units and
  indexed elementwise gates. Every run retained `evaluation_pass=true`,
  `max_abs=1.6666017472743988e-06`,
  `rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The full direct command intentionally deselected the same five
optional/environment-sensitive cases used by the established gate:

- `test_tflite_backend_matrix_add`
- `test_tflite_backend_matrix_hardswish_rewrite_on_off`
- `test_tf_converter_resize_cubic_avoids_flex_resize_bicubic`
- `test_tf_converter_resize_cubic_honors_cubic_coeff_a`
- `test_flatbuffer_direct_group_norm_alias_builtin_conversion`

## Failing tests and known issues

- No newly failing test is known at this checkpoint.
- The two full-suite warnings are the established FLOAT16 overflow warnings in
  `ir.py` from ArgMax/ReduceMax and negative-infinity Where coverage.
- The five tests listed above remain explicitly outside the current core gate;
  they were not silently treated as passes.
- A full Tier 0–Tier 5 corpus run has not been performed after this checkpoint,
  so corpus-wide non-regression is not yet proven.
- Performance/RSS targets are not yet proven for the current architecture.
- Baseline-invalid `vit_h_encoder.onnx` remains classified as `invalid_onnx`.
- Per user direction, DEIM is considered a successful conversion family.

## First work on resume

1. Verify `git status --short --branch`, local/remote divergence, and the two
   latest commits; do not create a pull request.
2. Move the complete
   `_optimize_transpose_pre_concat_ndhwc_chains` implementation mechanically
   to a focused pass module while retaining its wrapper and all five raw calls.
3. Confirm exact function-AST equivalence against `9a09553`, add a single-owner
   architecture gate, and run focused plus full direct gates.
4. Commit and push extraction before beginning indexed candidate planning.

Resume constraints remain: commit and push at coherent checkpoints only; no
pull request; no new dependency; default direct TFLite and `-cotof` must remain
TensorFlow-free; use `uv`; and run inference validation sequentially with one
process.

## Resumed SE layout checkpoint

Checkpoint `5f5de07` added a dedicated compact SE corpus without duplicating
the large legacy fixtures. It fixes three important boundaries: an SE-Conv
gate with an additional consumer rejects unchanged, a public SE-FC gate
rejects unchanged, and an SE-FC target branch sharing its leading NCHW adapter
rewrites while retaining that adapter for the side branch. Together with the
six existing success variants, focused characterization passed 9 tests.

The complete `_optimize_transpose_se_conv_mul_prepost_nhwc_chains` and
`_optimize_transpose_se_fc_mul_prepost_nhwc_chains` implementations then moved
mechanically to `passes/se_layout.py`. Their ASTs, including docstrings, match
`5f5de07`. The lowerer retains signature-compatible wrappers; all six SE-Conv
and nine SE-FC production positions remain unchanged until the separate
indexed migration checkpoint. An architecture test fixes the single-owner
boundary.

Focused success, rejection, and ownership validation passed 10 tests. The
complete sequential direct selection passed:

```text
1174 passed, 5 deselected, 2 warnings in 149.62s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Next work is the GraphIndex/ordered-runner migration of this
extracted family.

### Indexed SE-Conv checkpoint

The extracted SE-Conv implementation now accepts the shared
`ModelIRGraphIndex` and `LayoutState`. Consumer/producer reads, Swish and gate
input rewrites, Mean/post adapter alias rewrites, canonical output rewrite,
structural removals, pruning, metadata updates, and layout reconciliation use
differential state. Its implementation contains no whole-graph map builder or
direct operator-list deletion.

All six production positions call `run_se_conv_layout_cleanup`, with stable
`LAYOUT_PLAN` ID `layout.se_conv_gate_nhwc`. A cheap model-only capability scan
precedes an indexed common-region guard covering the leading
Transpose/Logistic/Mul, Mean branch and accepted adapter class, exclusive
second gate, and terminal inverse-Transpose fan-out. The existing deep matcher
continues to validate Logistic, affine, and Squeeze/Reshape gate details.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 40 tests. The compact positive fixture uses one initial
index refresh and one snapshot; gate fan-out rejects before snapshotting. Tier
1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1176 passed, 5 deselected, 2 warnings in 150.22s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_se_conv_superpoint` artifacts were removed after metrics
inspection. SE-FC remains the next separate indexed migration unit.

### Indexed SE-FC checkpoint

The 977-line SE-FC implementation now accepts the shared
`ModelIRGraphIndex` and `LayoutState`. All normal and alternate-path
consumer/producer reads, cloned Mean-axis input replacement, pool/Mul/Conv/gate
rewrites, canonical output and aliases, structural removals, pruning, metadata,
and layout reconciliation use differential state. With SE-Conv already
indexed, `passes/se_layout.py` now contains no whole-graph map builder or
direct operator-list deletion.

`run_se_fc_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.se_fc_gate_nhwc`. Eight main-model positions receive the session layout
state and diagnostics. The ninth fallback-IR position receives diagnostics but
creates its own layout state because it operates on a distinct ModelIR.
Model-only Transpose/Mul/dense-or-Conv capability preflight skips irrelevant
graphs. The indexed guard proves public boundaries, a normal gate Reshape and
inverse output bridge, or the common ADD/MUL/inverse-bridge prefix of the
alternate float path; the existing matcher retains all deep topology checks.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 42 tests. A normal SE-FC rewrite uses one initial index
refresh and one snapshot; a public gate rejects before snapshotting. The
shared-pre runner fixture retains its leading Transpose for the side branch.
Tier 1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1178 passed, 5 deselected, 2 warnings in 158.98s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_se_fc_superpoint` artifacts were removed after metrics
inspection.

### Elementwise gate characterization and mechanical extraction

Checkpoint `1623762` added the missing compact SUM/Logistic/Sub/Mul/Add
characterization. The successful graph proves three leading adapter removals,
SUM axis remapping from NCHW axis 1 to NHWC axis 3, canonical ADD output, and
downstream rewiring. A reduction-input side consumer proves a complete no-op.
Together with existing Logistic/Mul/Add, weighted Swish, nested weighted
Swish, and legacy-user fixtures, focused characterization passed 6 tests.

The four complete implementations moved mechanically to
`passes/elementwise_gate_layout.py`. Their ASTs, including docstrings, match
`1623762`; the lowerer keeps signature-compatible wrappers and all five raw
production positions per rule until the separate indexed migration. The
previous lowerer-local `_is_scalar_like_tensor` helper moved unchanged to
`core/model_ir_utils.py`, while remaining a compatibility import from the
lowerer. Ownership tests fix both boundaries.

Focused characterization, legacy, and ownership validation passed 7 tests.
The complete sequential direct selection passed:

```text
1181 passed, 5 deselected, 2 warnings in 166.23s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Next work is the indexed migration of these four extracted
rules.

### Indexed elementwise gate checkpoint

All four extracted matchers now accept one shared `ModelIRGraphIndex` and
active `LayoutState`. Consumer/producer traversal, SUM and branch input
rewrites, canonical output and aliases, conditional legacy-adapter rewrites,
structural removals, pruning, metadata, and layout reconciliation use
differential state. The module contains no whole-graph map builder or direct
operator-list deletion.

The five repeated raw call groups became five calls to
`run_elementwise_gate_layout_cleanup`. Each invocation owns four ordered
`LAYOUT_PLAN` specs with stable IDs `layout.sum_logistic_muladd_nhwc`,
`layout.weighted_add_swish_nhwc`,
`layout.nested_weighted_add_swish_nhwc`, and
`layout.logistic_muladd_nhwc`. Model-only common capability preflight and
indexed per-pattern guards preserve the legacy order while sharing one state.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 40 tests. The SUM success fixture produces four diagnostics,
one initial index refresh, and exactly one snapshot/change. Reduction fan-out
produces four skips and zero snapshots. Existing Logistic/MulAdd and both
weighted-Swish success fixtures now execute through the grouped runner. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1183 passed, 5 deselected, 2 warnings in 152.05s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_elementwise_gate_superpoint` artifacts were removed after
metrics inspection.

### Multi-branch gate characterization and mechanical extraction

Checkpoint `13ec048` replaced the missing model-specific coverage with a
compact generic two-branch graph. It proves branch adapter removal, independent
Mean-axis constant cloning/remapping, Logistic/Mul leaf propagation, Add-root
output canonicalization, and a complete gate-fan-out rejection. Focused
characterization passed 2 tests.

The complete 518-line matcher moved mechanically to
`passes/multi_branch_gate_layout.py`, with an AST including docstrings that
matches `13ec048`. Despite its historical OSNet name, the test and matcher are
defined only by generic topology. The lowerer keeps a signature-compatible
wrapper and the one production call remains unchanged until indexed migration.
An architecture test fixes ownership.

Focused characterization and ownership validation passed 3 tests. The
complete sequential direct selection passed:

```text
1186 passed, 5 deselected, 2 warnings in 151.88s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed multi-branch gate checkpoint

Checkpoint `b0d1248` migrated the extracted matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal, cloned
Mean-axis input replacement, Relu and Logistic input rewrites, Add-root output
canonicalization, alias rewrites, structural removals, pruning, metadata, and
layout reconciliation now use differential state. The implementation contains
no whole-graph producer/consumer map builder and no direct operator-list
deletion.

`run_multi_branch_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.multi_branch_gate_add_tree_nhwc`. Its model-only required-op preflight
avoids state construction for irrelevant graphs. Its indexed guard proves an
inverse output bridge, a nested Add tree with at least two Mul leaves, guarded
Relu branches with keep-dims Mean users, exclusive Logistic gates, and accepted
gate adapters before the complete matcher performs deeper validation. The
single production position supplies session layout state and diagnostics; the
lowerer compatibility wrapper remains available.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 41 tests. The success fixture uses one initial index refresh
and one snapshot; gate fan-out rejects before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1188 passed, 5 deselected, 2 warnings in 161.24s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_multi_branch_gate_superpoint` artifacts were removed after
metrics inspection.

### Dual-postconv gate characterization and mechanical extraction

Checkpoint `ed6d8c1` added the missing compact generic corpus for the
complementary Logistic/Sub gate feeding two Mul/Add outputs and two downstream
Conv branches. The positive fixture proves removal of all three leading and
both trailing layout adapters, direct NHWC inputs to the elementwise graph,
canonical Add outputs, and unchanged Conv inputs. Parameterized no-op coverage
fixes Logistic gate fan-out, a data-adapter side consumer, and a public Add
intermediate. Focused characterization passed 4 tests.

The complete 323-line matcher moved mechanically to
`passes/dual_postconv_gate_layout.py`. Its AST, including docstrings, matches
`ed6d8c1`; the lowerer retains a signature-compatible wrapper and all five raw
production positions until the separate indexed migration. An architecture
test fixes single ownership.

Focused characterization and ownership validation passed 38 tests. The
complete sequential direct selection passed:

```text
1193 passed, 5 deselected, 2 warnings in 169.93s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed dual-postconv gate checkpoint

Checkpoint `cc828c8` migrated the extracted complementary-gate matcher to
shared `ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal,
Logistic/Mul/Add input rewrites, Add output canonicalization, post-output alias
rewrites, structural removals, pruning, metadata, and layout reconciliation
now use differential state. The implementation contains no whole-graph map
builder and no direct operator-list deletion.

`run_dual_postconv_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.dual_postconv_complementary_gate_nhwc`. Its model-only required-op
preflight skips irrelevant graphs without state construction. Its indexed
guard proves the exclusive Logistic/Sub complementary gate, distinct data
adapters, two Mul/Add branches, inverse output adapters, public boundaries, and
allowed data fan-out before the complete matcher retains deeper checks. All
five production positions now supply session layout state and diagnostics; the
lowerer compatibility wrapper remains available.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 46 tests. The successful two-Conv fixture uses one initial
index refresh and one snapshot. Gate fan-out, data-adapter fan-out, and public
intermediate variants all reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1197 passed, 5 deselected, 2 warnings in 161.48s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_dual_postconv_superpoint` artifacts were removed after metrics
inspection.

### Postadd complementary-gate characterization and extraction

Checkpoint `ea78747` added a compact generic graph for the complementary
Logistic/Sub gate whose two Mul outputs cross inverse adapters before an NHWC
Add and downstream Conv. The success fixture proves removal of all five layout
adapters, direct NHWC elementwise inputs, canonical Mul outputs, and unchanged
downstream Add inputs. Parameterized no-op coverage fixes Logistic gate
fan-out, a data-adapter side consumer, and a public Mul intermediate. Focused
characterization passed 4 tests.

The complete 272-line matcher moved mechanically beside the dual-postconv
matcher in `passes/dual_postconv_gate_layout.py`. Its AST, including
docstrings, matches `ea78747`; the lowerer retains a signature-compatible
wrapper and all five raw production positions until the separate indexed
migration. The family ownership test covers both matchers.

Focused postadd, indexed sibling, and ownership validation passed 46 tests.
The complete sequential direct selection passed:

```text
1201 passed, 5 deselected, 2 warnings in 157.71s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed postadd complementary-gate checkpoint

Checkpoint `78b0742` migrated the postadd matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal,
Logistic/Mul input rewrites, Mul output canonicalization, post-output alias
rewrites, structural removals, pruning, metadata, and layout reconciliation
now use differential state. Both complementary-gate matchers contain no
whole-graph map builder and no direct operator-list deletion.

The existing family runner now registers a second stable `LAYOUT_PLAN` ID,
`layout.postadd_complementary_gate_nhwc`, after
`layout.dual_postconv_complementary_gate_nhwc`. Both indexed guards reuse one
resolver for the three input adapters, Logistic/Sub gate, and two Mul branches;
only their Add-before-post versus post-before-Add output contracts remain
separate. Each of the five production groups invokes the runner once and shares
one graph/layout state while preserving the legacy rule order.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 54 tests. The postadd success fixture creates one initial
index and one snapshot across both ordered specs. Gate fan-out, data-adapter
fan-out, and public-intermediate variants all reject with zero snapshots. Tier
1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1205 passed, 5 deselected, 2 warnings in 154.34s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_postadd_superpoint` artifacts were removed after metrics
inspection.

### Rank-five 3D gate characterization and extraction

Checkpoint `ee3d2fd` replaced the 177-line success fixture embedded in the
central direct test module with a dedicated compact rank-five corpus. The
positive graph proves base Reshape remapping, skip LeakyRelu and gate Logistic
adapter removal, both Add output canonicalizations, and all five Transpose
removals. Parameterized no-op coverage fixes shared-base fan-out, gate fan-out,
a public Add intermediate, an invalid NDHWC-to-NCDHW permutation, and an
invalid Reshape-constant rank. Focused characterization passed 6 tests while
reducing the central test module by 177 lines.

The complete 378-line matcher moved mechanically to
`passes/ndhwc_gate_layout.py`. Its AST, including docstrings, matches
`ee3d2fd`; the lowerer retains a signature-compatible wrapper and all six raw
production positions until the separate indexed migration. An architecture
test fixes single ownership.

Focused characterization and ownership validation passed 41 tests. The
complete sequential direct selection passed:

```text
1211 passed, 5 deselected, 2 warnings in 157.23s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed rank-five 3D gate checkpoint

Checkpoint `2871ade` migrated the extracted rank-five matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal, base
Reshape, skip LeakyRelu, and gate Logistic input rewrites, both Add output
canonicalizations, constant-shape remapping, structural removals, pruning,
metadata, and layout reconciliation now use differential state. Two duplicate
legacy permutation assignments were also removed. The implementation contains
no whole-graph map builder and no direct operator-list deletion.

`run_ndhwc_gate_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.ndhwc_leaky_logistic_gate`. Its model-only required-op preflight skips
irrelevant graphs without state construction. Its indexed guard proves both
inverse Add outputs, the shared base and skip branches, exclusive gate and
Mul, exact rank-four/rank-five permutations, public boundaries, and the
rank-five Reshape constant before the complete matcher performs the rewrite.
All six production positions now supply session layout state and diagnostics;
the lowerer compatibility wrapper remains available.

Focused runner, ownership, architecture, and irrelevant-graph efficiency
validation passed 51 tests. The success fixture uses one initial index refresh
and one snapshot. Shared-base fan-out, gate fan-out, public intermediate,
invalid permutation, and invalid reshape-rank variants all reject before
snapshotting. Tier 1 `superpoint.onnx` passed sequential
`-tb flatbuffer_direct -cotof` with `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`.

The complete sequential direct selection passed:

```text
1217 passed, 5 deselected, 2 warnings in 156.83s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_ndhwc_gate_superpoint` artifacts were removed after metrics
inspection.

### Conv3D gate characterization and extraction

Checkpoint `49c72b9` replaced the 176-line central inline fixture with a
dedicated compact Conv3D/LeakyRelu/Reshape gate corpus. Separate positive cases
fix both accepted semantic adapter ranks: rank-four NHWC-to-NCHW and rank-five
NDHWC-to-NCDHW. They prove semantic Reshape remapping, Conv-side LeakyRelu
adapter removal, gated Mul output canonicalization, and unchanged downstream
Conv3D input. Six no-op boundaries cover Conv-adapter fan-out, LeakyRelu
fan-out, gate-Reshape fan-out, a public Mul intermediate, invalid permutation,
and invalid reshape rank. Focused characterization passed 8 tests.

The complete 226-line matcher moved mechanically beside the rank-five sibling
in `passes/ndhwc_gate_layout.py`. Its AST, including docstrings, matches
`49c72b9`; the lowerer retains a signature-compatible wrapper and all six raw
production positions until the separate indexed migration. Family ownership
coverage includes both matchers.

Focused Conv3D, indexed sibling, and ownership validation passed 55 tests. The
complete sequential direct selection passed:

```text
1224 passed, 5 deselected, 2 warnings in 157.08s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently.

### Indexed Conv3D gate checkpoint

Checkpoint `a470cce` migrated the extracted Conv3D gate matcher to shared
`ModelIRGraphIndex` and `LayoutState`. Producer/consumer traversal, semantic
Reshape and Conv-side LeakyRelu rewrites, gated Mul output canonicalization,
constant-shape remapping, structural removals, pruning, metadata, and layout
reconciliation now use differential state. Both matchers in
`passes/ndhwc_gate_layout.py` contain no whole-graph map builder and no direct
operator-list deletion.

`run_ndhwc_gate_layout_cleanup` now registers a second stable `LAYOUT_PLAN` ID,
`layout.ndhwc_conv3d_leaky_unsqueeze_gate`, after
`layout.ndhwc_leaky_logistic_gate`. Its indexed guard proves the inverse Mul
output adapter, exclusive LeakyRelu and Reshape branches, accepted rank-four or
rank-five semantic adapter, rank-five Conv adapter, public boundaries, and
rank-five remappable Reshape constant before snapshotting. All six production
groups invoke the shared runner once; the legacy raw calls were removed while
the compatibility wrapper remains available.

Focused runner, sibling, ownership, architecture, and irrelevant-graph
efficiency validation passed 67 tests. Both accepted semantic-rank fixtures
use one initial index refresh and one snapshot. Conv-adapter, LeakyRelu, and
Reshape fan-out, public intermediate, invalid permutation, and invalid
reshape-rank variants all reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1232 passed, 5 deselected, 2 warnings in 154.69s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_conv3d_gate_superpoint` artifacts were removed after metrics
inspection.

### Cost-volume/ScatterND characterization checkpoint

Checkpoint `4b6f297` moved the embedded cost-volume success fixture out of the
central direct test module into
`tests/test_flatbuffer_direct_cost_volume_scatter_layout.py`. The compact graph
retains both descriptor adapters, shared Slice constants, Mean-axis mapping,
Reshape, casted five-coordinate ScatterND indices, ScatterND shape mapping,
the inverse rank-five adapter, and the downstream Conv3D contract. It proves
the same constant values, tensor metadata, operator removal, and Conv3D input
as the former fixture while reducing the central module by 177 lines.

Six parameterized boundaries prove a complete ModelIR no-op for a leading
adapter side consumer, ScatterND-result side consumer, post-adapter side
consumer, public ScatterND intermediate, invalid leading permutation, and
non-Conv3D downstream consumer. The snapshots compare operator topology and
options plus every tensor dtype, shape, shape signature, and constant value.
Production code was intentionally unchanged at this checkpoint.

Focused characterization passed 7 tests. The complete sequential direct
selection passed:

```text
1238 passed, 5 deselected, 2 warnings in 153.83s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. The next unit is a strictly mechanical matcher extraction with
AST-equivalence and single-owner gates.

### Cost-volume/ScatterND mechanical extraction checkpoint

Checkpoint `d62e77d` moved the complete 536-line matcher mechanically to
`passes/cost_volume_scatter_layout.py`. Its function AST, including the
docstring and nested helpers, exactly matches checkpoint `4b6f297`. The central
lowerer now keeps only a signature-compatible wrapper. All six raw production
call positions remain unchanged so extraction does not alter rule ordering or
retry behavior.

The architecture gate fixes the focused module as the single implementation
owner, the lowerer alias and compatibility wrapper, and exactly six production
calls. Focused characterization and ownership validation passed 43 tests. The
complete sequential direct selection passed:

```text
1239 passed, 5 deselected, 2 warnings in 156.55s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed planning, transactional late-validation safety, shared
graph/layout state, and runner integration remain a separate next checkpoint.

### Indexed cost-volume/ScatterND checkpoint

Checkpoint `56516ef` replaced repeated producer/consumer map construction with
one shared `ModelIRGraphIndex` and introduced a pure candidate plan that proves
the complete upstream island and every mutable constant before rewriting.
Slice ranges, reduction axes, Concat axes, ScatterND output shape, casted index
coordinates, coordinate rank, and bounds are validated without modifying the
model. This closes a legacy failure mode where an invalid late ScatterND
constant could leave earlier Slice or Mean constants partially remapped even
though the matcher reported no rewrite.

Input and alias rewrites now update the differential index, structural removal
uses indexed operators, and pruning/metadata reconciliation synchronize the
shared `LayoutState`. `run_cost_volume_scatter_layout_cleanup` registers stable
`LAYOUT_PLAN` ID `layout.cost_volume_scatter_ndhwc`; all six production
positions call it with session state and diagnostics. The lowerer compatibility
wrapper remains available.

Focused success, nine complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 60
tests. The success graph uses one initial index refresh and one transactional
snapshot. All nine rejection cases, including invalid ScatterND shape,
coordinate rank, and out-of-bounds coordinates, reject before snapshotting.
Tier 1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1252 passed, 5 deselected, 2 warnings in 164.88s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_cost_volume_superpoint` artifacts were removed after metrics
inspection.

### Add/Concat/constant-suffix characterization checkpoint

Checkpoint `fcf24b2` added the first dedicated coverage for
`_optimize_transpose_add_concat_const_suffix_nhwc_chains`. The compact success
graph includes two independent branch adapters, one shared base adapter, both
Add fan-ins, channel Concat, strict MUL(const) then ADD(const) suffix, inverse
output adapter, and a downstream Conv consumer. It proves all four adapters
are removed, both Add inputs become NHWC, Concat moves to axis 3, both rank-four
constants are transposed to NHWC, metadata follows the rewrite, and the suffix
Add directly owns the canonical post-adapter tensor name.

Nine parameterized boundaries prove a complete ModelIR no-op for branch
adapter fan-out, Add output fan-out, Concat fan-out, Mul output fan-out, public
suffix intermediate, public post output, invalid leading permutation, invalid
Concat axis, and missing suffix constant. Snapshots compare operator topology
and options plus every tensor dtype, shape, shape signature, and constant
value. Production code and all five raw call positions were unchanged.

Focused characterization passed 10 tests. The complete sequential direct
selection passed:

```text
1262 passed, 5 deselected, 2 warnings in 195.24s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Mechanical extraction with AST-equivalence and ownership gates
is the next separate unit.

### Add/Concat/constant-suffix mechanical extraction checkpoint

Checkpoint `73f96ca` moved the complete 271-line matcher mechanically to
`passes/add_concat_suffix_layout.py`. Its function AST, including docstring,
exactly matches checkpoint `fcf24b2`. The central lowerer now retains only a
signature-compatible wrapper, while all five raw production positions remain
unchanged so rule order and retry behavior are identical.

The architecture gate fixes the focused module as the single implementation
owner, the lowerer import alias and wrapper, and exactly five production calls.
Focused characterization and ownership validation passed 47 tests. The
complete sequential direct selection passed:

```text
1263 passed, 5 deselected, 2 warnings in 158.07s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Shared-constant copy-on-write, indexed mutation, transactional
runner integration, and raw-call replacement remain the next checkpoint.

### Indexed Add/Concat/constant-suffix checkpoint

Checkpoint `1b8c307` replaced repeated producer/consumer map construction with
one indexed candidate plan and differential mutation state. The plan proves
all branch/base adapters, exclusive Add outputs, channel Concat, strict
MUL(const)→ADD(const) suffix, inverse output adapter, constants, fan-out, and
public boundaries before snapshotting. Add inputs, suffix output aliasing, and
operator removal now update `ModelIRGraphIndex`; pruning and metadata reconcile
the shared `LayoutState`.

Both suffix constants now use copy-on-write when another operator consumes the
same buffer. The optimized island receives an NHWC clone while unrelated
consumers retain the original NCHW data and metadata. The canonical post tensor
also retains the once-permuted NHWC shape instead of applying the legacy
metadata permutation twice. Dedicated tests cover both shared constants and
the corrected `[N,H,W,C]` output metadata.

`run_add_concat_suffix_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.add_concat_const_suffix_nhwc`; all five production positions call it
with session layout state and diagnostics. Focused success, shared-constant,
nine no-op boundaries, runner, ownership, architecture, and irrelevant-graph
efficiency validation passed 63 tests. The success graph uses one initial index
refresh and one snapshot; all unsafe boundaries reject before snapshotting.
Tier 1 `superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1275 passed, 5 deselected, 2 warnings in 158.36s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_add_concat_suffix_superpoint` artifacts were removed after
metrics inspection.

### Dual-Mul/Concat characterization checkpoint

Checkpoint `82d8777` moved the 131-line embedded success fixture to
`tests/test_flatbuffer_direct_dual_mul_concat_layout.py`. The compact graph
retains the shared data adapter, two Mul branches, channel Concat, inverse
output adapter, downstream Relu, and an external NCHW consumer of one Mul
constant. It proves direct NHWC data inputs, axis-3 Concat, canonical output
aliasing, in-place conversion of an exclusive constant, and an NHWC clone for
the externally shared constant while its original buffer remains NCHW.

Ten parameterized boundaries prove a complete ModelIR no-op for pre-adapter
fan-out, Mul-output fan-out, Concat fan-out, public Concat/post tensors, invalid
pre/post permutations, invalid Concat axis, missing constant data, and Mul
branches that do not share one adapted data input. Snapshots compare every
operator, option, tensor dtype, shape, signature, and constant value.
Production code and all six raw call positions remain unchanged.

Focused characterization passed 11 tests. The complete sequential direct
selection passed:

```text
1285 passed, 5 deselected, 2 warnings in 165.06s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Mechanical extraction with AST-equivalence and ownership gates
is the next separate checkpoint.

### Dual-Mul/Concat mechanical extraction checkpoint

Checkpoint `af26412` moved the complete 297-line matcher mechanically to
`passes/dual_mul_concat_layout.py`. Its function AST, including docstring and
nested copy-on-write helper, exactly matches checkpoint `82d8777`. The lowerer
keeps a signature-compatible wrapper, and all six raw production positions
remain unchanged to preserve ordering and retry behavior.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly six production calls. Focused
characterization and ownership validation passed 49 tests. The complete
sequential direct selection passed:

```text
1286 passed, 5 deselected, 2 warnings in 163.50s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential copy-on-write, layout
state reconciliation, runner integration, and raw-call replacement remain the
next checkpoint.

### Indexed dual-Mul/Concat checkpoint

Checkpoint `64702b2` introduced a pure indexed plan that proves the shared data
adapter, two exclusive Mul branches, Concat/post topology, public boundaries,
constant presence, rank, target broadcast compatibility, and whether each
constant requires cloning before any mutation. This prevents a later invalid
constant from leaving an earlier branch partially converted.

Constant copy-on-write and Mul input replacement now update one
`ModelIRGraphIndex`; Concat output aliasing, pre/post adapter removal, pruning,
and metadata reconciliation share the same `LayoutState`. The canonical post
tensor keeps its once-permuted NHWC shape instead of receiving the legacy
second metadata permutation. `run_dual_mul_concat_layout_cleanup` registers
stable `LAYOUT_PLAN` ID `layout.dual_mul_concat_nhwc`; all six production
positions now call it with session state and diagnostics.

Focused success, ten complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 64
tests. The success graph uses one initial index refresh and one snapshot; all
unsafe boundaries reject before snapshotting. Tier 1 `superpoint.onnx` passed
sequential `-tb flatbuffer_direct -cotof` with `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`.

The complete sequential direct selection passed:

```text
1297 passed, 5 deselected, 2 warnings in 165.59s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_dual_mul_concat_superpoint` artifacts were removed after metrics
inspection.

### Axis-3 constant-Concat bridge characterization checkpoint

Checkpoint `019d3c6` moved the only 132-line embedded success fixture to
`tests/test_flatbuffer_direct_axis3_const_concat_layout.py`. The compact base
graph retains one NHWC-to-NCHW input adapter, a rank-four NCHW constant,
axis-3 Concat, an inverse post adapter, and a legacy NCHW consumer. It proves
constant NCHW-to-NHWC conversion, axis remapping to 2, post-adapter bypass,
and insertion of exactly one NHWC-to-NCHW bridge for legacy consumers.

Two additional success variants prove that every inverse post branch is
bypassed and that a leading adapter shared with an unrelated consumer is
retained. Nine parameterized rejection cases prove a complete ModelIR no-op
for public Concat/post tensors, invalid pre/post permutations, invalid Concat
axis, invalid constant rank or incompatible shape, missing constant data, and
a constant shared outside the Concat. Snapshots compare every operator,
option, tensor dtype, shape, signature, and constant value.

Focused characterization passed 12 tests. The complete sequential direct
selection passed:

```text
1308 passed, 5 deselected, 2 warnings in 166.51s
```

Production code and the single raw call remain unchanged. No dependency or
TensorFlow path was added, and no inference process was run concurrently.
Mechanical extraction with exact AST-equivalence and single-owner gates is the
next separate checkpoint.

### Concat/unary/Conv mechanical extraction checkpoint

Checkpoint `11e76bd` moved the complete matcher mechanically to
`passes/concat_unary_conv_layout.py`. Its function AST, including the docstring
and optional-unary traversal, exactly matches characterization checkpoint
`f624388`. The lowerer keeps a signature-compatible wrapper and both raw
production calls.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly two production calls. Focused
characterization and architecture validation passed 56 tests. The complete
sequential direct selection passed:

```text
1374 passed, 5 deselected, 2 warnings in 175.23s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential graph/layout mutation,
transactional runner integration, and raw-call replacement remain the next
checkpoint.

### Indexed Concat/unary/Conv checkpoint

Checkpoint `b86b31a` introduced a pure indexed plan for every exclusive input
adapter, optional accepted-unary chain, complete inverse-post fan-out, and
Conv2D/DepthwiseConv2D consumer set. Rank-four source, Concat, unary, and post
metadata are validated before mutation; a new invalid-rank boundary rejects
before snapshot in addition to the thirteen characterized cases.

Concat input and axis mutation, Concat/unary metadata permutation, post alias
replacement, adapter/post removal, pruning, and layout reconciliation use one
shared `ModelIRGraphIndex` and `LayoutState`. The implementation contains no
whole-graph producer/consumer map construction and no direct operator-list
deletion. `run_concat_unary_conv_layout_cleanup` registers stable
`LAYOUT_PLAN` ID `layout.concat_unary_conv_nhwc`; both raw production calls are
replaced with the runner while the lowerer wrapper remains.

Focused success, fourteen complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 76
tests. The two-unary/two-post success graph uses one initial index refresh and
one snapshot; all unsafe boundaries reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1390 passed, 5 deselected, 2 warnings in 173.99s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_concat_unary_conv_superpoint` artifacts were removed after
metrics inspection.

### Generic two-island SPP characterization checkpoint

Checkpoint `0804e37` added `tests/test_flatbuffer_direct_spp_layout.py`, the
first dedicated coverage for the 371-line central matcher. The compact graph
uses four ResizeBilinear branches sharing one base adapter, four Add outputs,
the first channel Concat/Mul and inverse adapter, an NHWC affine/Conv, a return
adapter into a second base/Conv Concat/Mul, and a final inverse adapter,
affine, and Conv. It proves NHWC propagation through both islands, axis-3
Concat, channelwise constant conversion, and removal of eight adapters.

Sixteen parameterized boundaries prove a complete ModelIR no-op for branch,
Concat, Mul, inverse-post, and intermediate-Conv fan-out across both islands;
public base/first-Concat tensors; invalid leading permutation or either Concat
axis; a non-Resize branch producer; and missing first/second Mul constants.
Snapshots compare every operator, option, tensor shape/signature, and constant
value.

Focused characterization passed 17 tests. The complete sequential direct
selection passed:

```text
1407 passed, 5 deselected, 2 warnings in 176.30s
```

Checkpoint `c531b54` then moved the complete matcher mechanically to
`passes/spp_layout.py`. The function AST, including its docstring and legacy
selection/mutation order, exactly matches `0804e37`. The lowerer retains a
signature-compatible wrapper and all seven raw production calls. The
single-owner architecture gate fixes this boundary; focused SPP and
architecture validation passed 59 tests, and the complete sequential direct
selection passed 1,408 tests with the same five deselections and two known
warnings. No dependency or TensorFlow path was added, and no inference process
was run concurrently. Indexed planning and runner integration are the next
separate checkpoint.

### Indexed generic two-island SPP checkpoint

Checkpoint `8edf5c2` replaced the legacy map-rebuilding implementation with a
pure `_SppLayoutCandidate` that validates all four Resize/Add branches, both
Concat/Mul/adapter islands, the intervening and terminal Conv paths, every
fan-out/public boundary, rank-four metadata, and both constant payloads before
mutation. Shared constants are cloned only for the rewritten Mul inputs so
outside consumers retain the original NCHW payload. Per-axis quantization
metadata moves from NCHW dimension 1 to NHWC dimension 3 together with the
constant data.

All input rewrites, operator removals, pruning, and layout synchronization use
one shared `ModelIRGraphIndex` and `LayoutState`; the implementation contains
no whole-graph producer/consumer-map construction and no direct operator-list
deletion. `run_spp_layout_cleanup` registers stable `LAYOUT_PLAN` ID
`layout.generic_spp_nhwc`; all seven former raw production positions now call
it and the lowerer compatibility wrapper remains.

Focused validation passed 85 tests. A candidate uses one index refresh and one
transaction snapshot; every unsafe boundary rejects before snapshotting, and
an irrelevant 256-op graph builds no index, snapshot, or fingerprint. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with the
unchanged metrics `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`. The complete sequential direct selection passed:

```text
1430 passed, 5 deselected, 2 warnings in 161.91s
```

No dependency or TensorFlow import path was added. Temporary
`/tmp/onnx2tf_spp_superpoint` artifacts were removed after metrics inspection.
The next staged family is the adjacent 258-line, five-call NDHWC pre-Concat
matcher; the larger generic NHWC pre-Concat matcher remains a separate future
unit.

### NDHWC pre-Concat characterization checkpoint

Checkpoint `9a09553` moved the only 96-line central success fixture into
`tests/test_flatbuffer_direct_ndhwc_concat_layout.py` and expanded it into a
compact 16-case semantic corpus. The success graph combines one direct NDHWC
input adapter with one adapter/unary input, a channel-axis NCDHW Concat, and two
inverse post adapters. It fixes unary propagation in NDHWC, axis 4, canonical
post-output selection, alias replacement for the second post branch, and
removal of all three adapters.

Fifteen parameterized cases prove a complete ModelIR no-op for direct-adapter,
unary-adapter, unary-output, and Concat fan-out; public direct/unary/Concat/post
tensors; invalid pre/post permutations or Concat axis; an unsupported unary;
invalid direct-input rank; and incompatible projected spatial shapes. The
production matcher and all five raw call positions remain unchanged.

Focused characterization passed 16 tests. The complete sequential direct
selection passed:

```text
1445 passed, 5 deselected, 2 warnings in 170.91s
```

No dependency or TensorFlow path was added, and no inference process ran
concurrently. Exact-AST mechanical extraction against `9a09553` is the first
work on resume. The five-call count is authoritative; the earlier six-call
handoff text accidentally counted the function definition and is superseded.

### Dequantize/Concat/Quantize mechanical extraction checkpoint

Checkpoint `35a4cb1` moved the complete matcher mechanically to
`passes/dequant_concat_quantize_layout.py`. Its function AST, including the
docstring and all legacy selection/mutation order, exactly matches
characterization checkpoint `ea74ffd`. The lowerer keeps a
signature-compatible wrapper and both raw production calls.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly two production calls. Focused
characterization and architecture validation passed 55 tests. The complete
sequential direct selection passed:

```text
1339 passed, 5 deselected, 2 warnings in 176.09s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential graph/layout mutation,
transactional runner integration, and raw-call replacement remain the next
checkpoint.

### Indexed Dequantize/Concat/Quantize checkpoint

Checkpoint `3be0c3e` introduced pure indexed plans for every leading adapter,
exclusive Dequantize branch, rank-four Concat→Quantize edge, inverse post
branch, canonical quantized output, and adapter-removal decision. Quantize and
source quantization metadata and rank-four Concat metadata are validated before
mutation; three new malformed-metadata/rank cases reject before snapshot in
addition to the twelve characterized boundaries.

Dequantize input rewrites, metadata permutations, Concat axis mutation,
Quantize output canonicalization, post-alias replacement, adapter/post removal,
pruning, and layout reconciliation use one shared `ModelIRGraphIndex` and
`LayoutState`. The implementation contains no whole-graph producer/consumer
map construction and no direct operator-list deletion. The stable
`LAYOUT_PLAN` ID is `layout.dequant_concat_quantize_nhwc`; both raw production
calls are replaced with `run_dequant_concat_quantize_layout_cleanup`, while the
lowerer compatibility wrapper remains.

Focused success, fifteen complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 78
tests. The two-post success graph uses one initial index refresh and one
snapshot; all unsafe boundaries reject before snapshotting. Tier 1
`superpoint.onnx` passed sequential `-tb flatbuffer_direct -cotof` with
`evaluation_pass=true`, `max_abs=1.6666017472743988e-06`,
`rmse=1.6207873294228388e-07`, and cosine similarity `1.0`.

The complete sequential direct selection passed:

```text
1358 passed, 5 deselected, 2 warnings in 173.55s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_dequant_concat_quantize_superpoint` artifacts were removed after
metrics inspection.

### Concat/unary/Conv characterization checkpoint

Checkpoint `f624388` added
`tests/test_flatbuffer_direct_concat_unary_conv_layout.py`, the first dedicated
coverage for this central matcher. One compact success graph proves the
unary-free Concat path; a second proves a RELU/Tanh chain with two inverse post
adapters ending in Conv2D and DepthwiseConv2D. Both remove every adapter,
rewrite Concat inputs and axis to NHWC, permute Concat/unary metadata once, and
feed every Conv-family consumer directly from the NHWC tail.

Thirteen parameterized boundaries prove a complete ModelIR no-op for leading
adapter, Concat, or unary fan-out; public adapter, Concat, unary, or post
tensors; invalid pre/post permutations; invalid Concat axis; a non-Transpose
input; an unsupported unary; and a non-Conv post consumer. Snapshots compare
operator options and all tensor metadata and constant values.

Focused characterization passed 15 tests. The complete sequential direct
selection passed:

```text
1373 passed, 5 deselected, 2 warnings in 172.58s
```

Production code and both raw calls remain unchanged. No dependency or
TensorFlow path was added, and no inference process was run concurrently.
Mechanical extraction with exact AST-equivalence and single-owner gates is the
next separate checkpoint.

### Axis-3 constant-Concat bridge mechanical extraction checkpoint

Checkpoint `5228444` moved the complete matcher mechanically to
`passes/axis3_const_concat_layout.py`. Its function AST, including the
docstring and nested helpers, exactly matches characterization checkpoint
`019d3c6`. The lowerer keeps a signature-compatible wrapper and the single raw
production call, so pass order and retry behavior remain unchanged.

The architecture gate fixes the focused module as the single implementation
owner, its lowerer alias and wrapper, and exactly one production call. Focused
characterization and architecture validation passed 51 tests. The complete
sequential direct selection passed:

```text
1309 passed, 5 deselected, 2 warnings in 165.14s
```

No dependency or TensorFlow path was added, and no inference process was run
concurrently. Indexed candidate planning, differential graph/layout mutation,
transactional runner integration, and raw-call replacement remain the next
checkpoint.

### Indexed axis-3 constant-Concat bridge checkpoint

Checkpoint `a261462` introduced pure indexed planning for the unique leading
adapter, every exclusive rank-four constant conversion, every inverse post
branch, the retained-adapter decision, and the optional legacy NCHW bridge.
All bridge metadata and shape compatibility are validated before mutation.
Public adapter and constant tensors now reject before snapshot in addition to
the nine characterized boundaries, preventing graph-output producer loss or
silent constant-layout changes.

Constant buffers, Concat inputs/axis, post aliases, legacy inputs, adapter/post
removal, bridge insertion, pruning, and layout reconciliation use one shared
`ModelIRGraphIndex` and `LayoutState`. The implementation contains no
whole-graph producer/consumer map construction and no direct operator-list
insertion or deletion. `run_axis3_const_concat_layout_cleanup` registers stable
`LAYOUT_PLAN` ID `layout.axis3_const_concat_bridge_nhwc`; the single raw
production call is replaced with the runner while the lowerer compatibility
wrapper remains.

Focused success, eleven complete no-op boundaries, runner instrumentation,
ownership, architecture, and irrelevant-graph efficiency validation passed 69
tests. The success graph uses one initial index refresh and one snapshot; all
unsafe boundaries reject before snapshotting. Tier 1 `superpoint.onnx` passed
sequential `-tb flatbuffer_direct -cotof` with `evaluation_pass=true`,
`max_abs=1.6666017472743988e-06`, `rmse=1.6207873294228388e-07`, and cosine
similarity `1.0`.

The complete sequential direct selection passed:

```text
1323 passed, 5 deselected, 2 warnings in 166.52s
```

No dependency or TensorFlow path was added. Temporary
`/tmp/onnx2tf_axis3_const_concat_superpoint` artifacts were removed after
metrics inspection.

### Dequantize/Concat/Quantize characterization checkpoint

Checkpoint `ea74ffd` added
`tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py`, the first
dedicated coverage for this central matcher. The compact quantized graph uses
two NHWC INT8 inputs, two leading adapters and Dequantize branches, axis-1
Concat, Quantize, inverse post adapters, and downstream Dequantize consumers.
It proves direct NHWC Dequantize inputs, axis-3 Concat, removal of exclusive
adapters, and preservation of the quantized dtype, shape, and full
`QuantParamIR` on the canonical output.

Additional success variants prove that multiple post adapters merge into one
canonical quantized tensor and that a leading adapter shared outside the
island remains available. Twelve parameterized boundaries prove a complete
ModelIR no-op for Dequantize/Concat/quantized fan-out; public pre,
Dequantize, Concat, Quantize, and post tensors; invalid pre/post permutations;
invalid Concat axis; and a non-Dequantize branch. Snapshots include operator
options, tensor metadata, quantization, and constant values.

Focused characterization passed 15 tests. The complete sequential direct
selection passed:

```text
1338 passed, 5 deselected, 2 warnings in 166.21s
```

Production code and both raw calls remain unchanged. No dependency or
TensorFlow path was added, and no inference process was run concurrently.
Mechanical extraction with exact AST-equivalence and single-owner gates is the
next separate checkpoint.
