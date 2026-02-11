# flatbuffer_direct Migration Guide

## Goal
Migrate from the default `tf_converter` backend to `flatbuffer_direct` safely while keeping conversion reproducibility and diagnostics.

## Recommended rollout
1. Keep `--tflite_backend tf_converter` as baseline in CI.
2. Add one CI lane with `--tflite_backend flatbuffer_direct --report_op_coverage`.
3. Fix/allow operations based on `*_op_coverage_report.json`.
4. Enable quantization and split features in stages.

## Stage commands
1. Baseline direct export:
```bash
python -m onnx2tf.onnx2tf \
  -i model.onnx \
  -o out \
  --tflite_backend flatbuffer_direct \
  --report_op_coverage
```

2. Quantization + ONNX evaluation:
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

3. Auto split + split evaluation:
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

## Custom OP policy
Use Custom OP lowering only when builtin mapping is not feasible.

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

## Fallback policy
If direct export is required but you want a safety net in production jobs:

```bash
python -m onnx2tf.onnx2tf \
  -i model.onnx \
  -o out \
  --tflite_backend flatbuffer_direct \
  --flatbuffer_direct_fallback_to_tf_converter
```

When direct export fails, conversion falls back to `tf_converter` and prints warning logs.

## Report files
1. Accuracy report: `*_accuracy_report.json`
2. Split plan: `*_split_plan.json`
3. Split manifest: `*_split_manifest.json`
4. Split accuracy: `*_split_accuracy_report.json`
5. OP coverage: `*_op_coverage_report.json`
