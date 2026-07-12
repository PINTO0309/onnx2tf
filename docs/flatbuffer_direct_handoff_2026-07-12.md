# flatbuffer_direct refactor handoff — 2026-07-12

## Current checkpoint — `fb-refactor2` after `286a0e7`

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

The managed Tier 0–4 profile now records 365 passes, 6
`missing_tflite_report`, 23 `tflite_fail`, and 26 excluded historical timeouts.
There are 29 active non-passes. The next accuracy failures without an explicit
normalized cause are the two GridSample models named below; earlier failures in
managed order now have documented quantization/runtime semantics.

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
The next cause-unclassified
models are `model_70_2023_0220_32_2_1_grid_sample_bilinear_sim.onnx` and
`model_grid_sample.onnx`.

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

- `793 passed, 7 deselected, 2 warnings` across the direct builder, managed
  profile, architecture/import boundary, and the two new regression files;
- `28 passed, 772 deselected` for the focused Inverse, Resize, managed profile,
  and TensorFlow-free checks;
- `23 passed, 750 deselected` for the subsequent legacy Upsample, Resize,
  managed profile, and TensorFlow-free regression set;
- `1 passed, 756 deselected` for Compress after removing a pre-existing unused
  local from the touched Resize module;
- both dense corpus models passed sequential end-to-end `-cotof` runs.

The seven broad-suite deselections are the optional TensorFlow backend matrix,
the optional Torch GroupNorm integration test (the core environment exposes an
incompatible system Python 3.10 Torch binary), and their explicitly named
companions. No optional dependency was installed for this checkpoint.

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
  `op_builders/grid_sample_utils.py` (1,237 lines), below the 2,000-line source
  limit; no new package was introduced.

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
- Split the new control, Pad, and RoiAlign helpers into focused modules well
  below the 2,000-line source limit.
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
