# flatbuffer_direct refactor handoff — 2026-07-12

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
