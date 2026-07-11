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

Operator additions keep capability validation and lowering in the same
op-family module. Do not add model-name checks. A model-specific failure should
be reduced to a semantic graph pattern with explicit guards and a focused ONNX
fixture generated with `onnx.helper`.

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

Use the fixed order Tier 0 through Tier 5. Do not introduce a process pool or
parallel model runner. Every successful baseline model must remain successful;
accuracy reports must retain their existing stricter judgement and every
comparable float output must remain below a maximum absolute error of `1e-1`.

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

The root-only Tier 2 gate at commit `ad1d508` contains 113 models. The managed
result is `docs/baselines/flatbuffer_direct_tier2_root_ad1d508.json`: 80
passed, 4 conversion errors, 3 timeouts, 6 accuracy failures, and 20 missing
reports. Median and maximum durations were 7.124 and 120.360 seconds.

The root-only Tier 3 gate at commit `c838b42` contains 71 models. The managed
result is `docs/baselines/flatbuffer_direct_tier3_root_c838b42.json`: 22
passed, 15 conversion errors, 17 timeouts, 1 accuracy failure, and 16 missing
reports. Median and maximum durations were 17.248 and 120.662 seconds.
