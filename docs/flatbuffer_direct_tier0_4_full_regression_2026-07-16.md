# Tier 0-4 TFLite and Native PyTorch Full Regression Check

## Scope fixed before execution

- Branch under test: `fb-refactor5`.
- Starting checkpoint: `c5d52f80` (`Extract HARD_SWISH shape sanitization pass`).
- Managed profile:
  `docs/baselines/flatbuffer_direct_active_tier0_4.json`.
- Profile SHA-256:
  `d8150d5749eda3ed64fe15d721a178e87c8910b3d3ff8312cd7ff599291595bf`.
- Included ONNX tiers: Tier 0 through Tier 4, or 1-1,999 ONNX nodes.
- Per-model wall-clock ceiling: 600 seconds.
- Inference/conversion concurrency: one. Models are processed in the fixed
  managed-profile order and no inference ProcessPool or parallel worker is
  used.
- Every converter subprocess tree is monitored through Linux `/proc`. Any
  nonzero descendant `VmSwap` terminates that tree and is recorded as
  `swap_detected`; source correction is forbidden until the model and failure
  evidence have been recorded.
- TensorFlow remains blocked by the normal `flatbuffer_direct` boundary. The
  test requests only TFLite and the native PyTorch package; it does not request
  SavedModel or another optional TensorFlow artifact.

The profile contains 420 Tier 0-4 models. Thirty-nine are omitted before the
run because their managed history is either `timeout` (27) or explicit
`excluded` (12). This preserves all user-requested exclusions, the approved
`encoder.onnx` 600-second timeout classification, repeated quick-ceiling
timeouts, and the policy that a model which generated SWAP must remain out of
subsequent managed runs. All 381 remaining profile entries exist in the root
corpus.

The fixed active distribution is:

| Tier | Active models |
| ---: | ---: |
| 0 | 119 |
| 1 | 84 |
| 2 | 105 |
| 3 | 50 |
| 4 | 23 |
| **Total** | **381** |

Their recorded TFLite baseline classifications are 355 `pass`, 20
`tflite_fail`, and 6 `missing_tflite_report`. The latter two groups remain in
scope: they are needed to distinguish an unchanged known limitation from a
new branch regression by normalized failure signature and causal comparison.

## Pre-run infrastructure problem recorded before correction

The managed flatbuffer-direct runner already provided the required profile
filtering, per-model process isolation, fixed ordering, 600-second timeout,
accuracy reports, normalized error signatures, and process-tree SWAP monitor.
However, its PyTorch boolean expanded unconditionally to all four flags:

`-fdopt -fdots -fdodo -fdoep`

That behavior would test TorchScript, Dynamo ONNX, and ExportedProgram in
addition to the native PyTorch conversion requested here. It could introduce
unrelated failures, substantially extend runtime, and retain unnecessary
artifacts. Using `--tflite_only` would avoid the excess work but would fail to
test PyTorch. The separate legacy PyTorch bulk runner was not an acceptable
substitute because it has no managed-profile options or process-tree SWAP
monitor.

The problem was recorded before source correction. The safe correction adds a
new opt-in `--native_pytorch_only` mode. Default behavior is unchanged. In the
new mode the managed runner emits only `-fdopt`, still requires both TFLite and
PyTorch accuracy reports, and records the artifact mode in resumable state and
summary metadata. Resume rejects a mode mismatch. The mode is invalid when
PyTorch artifacts are disabled. Focused runner verification passes 38 tests;
the new command-contract test proves that `-fdopt` is present and `-fdots`,
`-fdodo`, and `-fdoep` are absent.

## Evidence and correction policy

The authoritative run directory is outside the repository. `bulk_status.json`
is updated after every model and is resumable. Command stdout/stderr, pass
metrics, both accuracy reports, classifications, durations, maximum absolute
errors, SWAP peaks, and normalized error signatures are retained. Large staged
models, TFLite files, generated schemas, and generated packages may be removed
only after the corresponding state entry has been persisted and both runtime
checks have finished; they can be regenerated from the unchanged root model.

No converter or exporter correction may be made when the first current-branch
run reports a problem. The report must first capture the model, tier, baseline
classification, current classification, duration, exit status, TFLite and
PyTorch metrics, SWAP status, normalized signature, and relevant log excerpt.
New failures are then compared against the pre-branch implementation with the
same command and fixed input before they can be called regressions. Only a
recorded, causally attributed, locally testable correction may proceed.

## Execution status

### Aborted environment-contaminated attempt

The first attempt was stopped after 81 entries because every model failed in
0.58-0.80 seconds before conversion. All 81 entries have the same normalized
signature SHA-256:

`5c700d994ec8cc08efa71cf0fc9b0ec1ef78363ff8d416f321f96fcf297e40bf`

The signature reports that Python 3.12 in `.venv` attempted to load
`libtorch_python.so` from the user's Python 3.10 installation under
`~/.local/lib/python3.10/site-packages/torch/lib`. The inherited environment
was:

- `PYTHONPATH=/usr/local/lib/python3.10/dist-packages:`;
- `LD_LIBRARY_PATH` beginning with
  `/home/b920405/.local/lib/python3.10/site-packages/torch/lib`.

An ordinary `uv run python` reproduced the undefined `_PyCode_GetExtra`
symbol error. Running the same interpreter with `PYTHONNOUSERSITE=1` and both
`PYTHONPATH` and `LD_LIBRARY_PATH` cleared imported the correct uv-managed
Torch 2.11.0+cpu from `.venv/lib/python3.12/site-packages`. This is a launch-
environment failure, not evidence of 81 model regressions. The complete
aborted state and per-model logs are retained outside the repository and will
not be resumed. A fresh authoritative output directory is required so the
invalid entries cannot be mixed with actual conversion results.

The corrective action is limited to the process environment: launch the
runner with `PYTHONNOUSERSITE=1`, empty `PYTHONPATH`, and empty
`LD_LIBRARY_PATH`. No converter, lowerer, exporter, model, baseline, or
acceptance rule is changed. A single-model monitored canary must succeed
before the 381-model run starts again.

### Authoritative run

The clean-environment canary used `1x128x64x64.onnx` with the exact production
command shape `-tb flatbuffer_direct -cotof -fdopt`. It completed in 4.475
seconds with classification `pass`, TFLite maximum absolute error
`9.5367431640625e-07`, native PyTorch maximum absolute error
`1.0728836059570312e-06`, and process-tree SWAP zero. The authoritative run
will therefore start from an empty directory with the same sanitized
environment and the full managed profile.
