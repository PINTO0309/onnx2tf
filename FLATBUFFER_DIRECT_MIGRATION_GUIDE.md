# flatbuffer_direct Migration Guide

## Goal
Migrate from the default `tf_converter` backend to `flatbuffer_direct` in controlled stages while preserving production stability and diagnosability.

## Backend differences (quick view)
|Item|`tf_converter`|`flatbuffer_direct`|
|:-|:-|:-|
|Default|Yes|No (opt-in)|
|Final generation path|TensorFlow Lite Converter|Direct FlatBuffer builder|
|Optimization behavior|TF-path accumulated rewrites/heuristics|Direct preprocess + strict dispatch constraints|
|Failure model|Many patterns absorbed by TF conversion|Explicit failure with `reason_code`|
|Custom op path|Implicitly minimized by TF path|Explicit opt-in + allowlist|
|Fallback|N/A|`--flatbuffer_direct_fallback_to_tf_converter`|

## Recommended rollout
1. Keep baseline CI on `--tflite_backend tf_converter`.
2. Add one additional CI lane with `--tflite_backend flatbuffer_direct --report_op_coverage`.
3. Resolve failures by `reason_code` and adjust model/export options.
4. Only after stable float32/float16 conversion, enable quantization and split evaluation.

## Stage-by-stage commands
### Stage 0: Baseline direct export + diagnostics
```bash
python -m onnx2tf.onnx2tf \
  -i model.onnx \
  -o out \
  --tflite_backend flatbuffer_direct \
  --report_op_coverage
```

### Stage 1: Quantization + ONNX-based accuracy check
```bash
python -m onnx2tf.onnx2tf \
  -i model.onnx \
  -o out \
  --tflite_backend flatbuffer_direct \
  -odrqt -oiqt \
  --eval_with_onnx \
  --eval_target_tflite full_integer_quant \
  --eval_compare_mode dequant \
  --report_op_coverage
```

### Stage 2: Split generation + split accuracy check
```bash
python -m onnx2tf.onnx2tf \
  -i model.onnx \
  -o out \
  --tflite_backend flatbuffer_direct \
  --auto_split_tflite_by_size \
  --tflite_split_target_bytes 1060000000 \
  --tflite_split_max_bytes 1073741824 \
  --eval_split_models \
  --report_op_coverage
```

### Stage 3: Safety-net fallback for production jobs
```bash
python -m onnx2tf.onnx2tf \
  -i model.onnx \
  -o out \
  --tflite_backend flatbuffer_direct \
  --flatbuffer_direct_fallback_to_tf_converter
```

When direct export fails, conversion falls back to `tf_converter` and warning logs are emitted.

## Preprocess scope in direct path
`flatbuffer_direct` applies staged preprocess rules before lowering:
1. `pattern_fusion_wave2`
   - ReLU/Clip chain normalization
   - GELU chain fusion
   - SpaceToDepth chain fusion
2. `pseudo_ops_wave1`
   - HardSwish / LeakyRelu / PRelu / Gelu / limited Pow rewrites
3. `constant_fold_a5`
   - Limited constant folding for shape/axes and arithmetic helper chains
4. `normalize_attrs_a5`
   - `perm`/`axes` normalization and softmax-axis bridge insertion

Use `preprocess_report.applied_rules` in `*_op_coverage_report.json` to inspect actual rewrites.

## Custom OP policy
Use custom-op lowering only when builtin mapping is not feasible.

```bash
python -m onnx2tf.onnx2tf \
  -i model.onnx \
  -o out \
  --tflite_backend flatbuffer_direct \
  --flatbuffer_direct_allow_custom_ops \
  --flatbuffer_direct_custom_op_allowlist Einsum,TopK \
  --report_op_coverage
```

Behavior:
1. Without custom-op enablement, custom candidates fail with `reason_code=custom_op_candidate_disabled`.
2. If allowlist is specified and op is missing, conversion fails with `reason_code=custom_op_not_in_allowlist`.

## Known limitations and mitigation
|Symptom (`reason_code`)|Cause|Mitigation|
|:-|:-|:-|
|`unsupported_onnx_op`|No direct builtin/custom path|Use `tf_converter`, fallback, or model rewrite|
|`requires_constant_input`|Dynamic axes/perm/shape where constants are required|Pre-fold graph (`onnxsim`) or rewrite to constants|
|`unsupported_attribute_value`|Direct constraints unmet (axis/rank/mode)|Adjust exporter flags or rewrite subgraph|
|`custom_op_candidate_disabled`|Custom candidate encountered while custom mode disabled|Enable custom ops only if runtime supports them|
|`custom_op_not_in_allowlist`|Candidate op not in allowlist|Add to allowlist explicitly|

## Report files
1. Accuracy report: `*_accuracy_report.json`
2. Split plan: `*_split_plan.json`
3. Split manifest: `*_split_manifest.json`
4. Split accuracy: `*_split_accuracy_report.json`
5. OP coverage: `*_op_coverage_report.json`

## Operational checklist
1. Keep `tf_converter` lane green at all times.
2. Gate `flatbuffer_direct` rollout by model family (small -> medium -> large).
3. Require `--report_op_coverage` in CI for direct lane.
4. Review `unsupported_reason_counts` and `custom_op_policy` for every failure.
5. Avoid custom-op expansion unless runtime/serving side is ready.
