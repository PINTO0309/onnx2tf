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

## Problems recorded before correction

### `GridSample_16.onnx`: native PyTorch runtime shape failure

The first authoritative-run problem occurred at index 12, Tier 0. Conversion
itself exited 0 in 3.664 seconds and generated float32/float16 TFLite plus the
native PyTorch package. TFLite evaluation passed with maximum absolute error
`8.13603401184082e-06`, exactly matching the managed `pass` baseline. The
process-tree SWAP peak was zero.

The native PyTorch evaluation did not produce its accuracy report. The
captured warning is:

`shape '[1, 3, 51076]' is invalid for input of size 253120`

The bulk classification is `missing_pytorch_report` and the normalized
signature SHA-256 is
`7886af494a718e66261194c22a8d65e7c1f43d61ea1ba58d8f3fa6d59e1d9d03`.
The TFLite and PyTorch package artifacts were both generated, so this is
currently classified as a native PyTorch runtime/code-generation problem, not
a TFLite conversion regression. There is no managed PyTorch baseline for this
model; whether the failure predates `fb-refactor5` remains unproven. No source
change will be made until the complete current-branch run is recorded and the
same command is causally compared with the pre-branch implementation.

## Completed authoritative run

The authoritative clean-environment run completed all 381 active entries at
checkpoint `a0f03fed`. The runner intentionally returned 1 because strict
failures were present; it did not crash. Total model wall time was
9,347.523 seconds, median model time was 9.528 seconds, and maximum model time
was 600.644 seconds. No completed model or descendant process used SWAP.

The combined TFLite and native-PyTorch classifications were:

| Classification | Models |
| --- | ---: |
| `pass` | 136 |
| `missing_pytorch_report` | 183 |
| `pytorch_fail` | 32 |
| `both_fail` | 12 |
| `missing_both_reports` | 10 |
| `missing_tflite_report` | 4 |
| `conversion_error` | 2 |
| `timeout` | 2 |
| **Total** | **381** |

These combined counts are observations, not branch-regression counts. In
particular, the native PyTorch path had no managed pre-branch baseline. The
current run contains 245 non-pass entries but 209 normalized signature
clusters, so rerunning every native-PyTorch failure on the old implementation
would be both wasteful and insufficiently targeted. Causal comparison is
performed by failure family and representative model instead.

Separating TFLite from native PyTorch gives 343 TFLite passes, 20 TFLite
accuracy failures, 14 missing TFLite reports, two conversion errors, and two
timeouts. Against the managed TFLite baseline:

- 341 of 355 baseline passes remained TFLite passes;
- 18 of 20 known TFLite failures remained failures;
- two known failures improved to passes: `yolox_nano.onnx` and
  `rtdetrv4_s.onnx`;
- all six known missing reports remained missing;
- 14 baseline passes were initially observed as non-passes and therefore
  required causal comparison.

The 14 observations include two user-approved DEIM TopK index mismatches and
two models that reached the 600-second exclusion boundary. The remaining ten
were compared with detached `main` checkpoint `a8640153` under the same uv
environment, sanitized import environment, `-tb flatbuffer_direct -cotof
-fdopt` options, one-process scheduling, 600-second limit, and process-tree
SWAP monitor.

## Timeout exclusions and timeout cleanup defect

`vit_b_encoder.onnx` reached 600.644 seconds and
`superpoint_lightglue_end2end_fused_cpu.onnx` reached 600.219 seconds. Both had
zero SWAP. They are outside the requested under-ten-minute corpus and must be
changed from active baseline entries to managed `timeout` exclusions before a
future full run. SuperPoint/LightGlue had already been treated as a timeout in
earlier analysis, so its active-profile presence was also a profile maintenance
defect.

After the `vit_b_encoder.onnx` timeout, its
`flatbuffer_direct_op_error_report` descendant was reparented and remained
alive while the next model started. At discovery it had PID 2952613, parent
PID 1533, elapsed time 10:49, RSS 67,098,632 KiB, and VmSwap zero. This
violated the one-inference-process requirement and reduced available disk
space from roughly 55 GiB to 20-30 GiB through still-open temporary state.
The evidence was captured before intervention. Only that orphan was sent TERM
and then KILL; it became a zero-RSS zombie and available disk returned to about
54 GiB. No converter output or classification was changed. The later
SuperPoint/LightGlue timeout left no orphan.

The bulk runner must be corrected so timeout and SWAP termination reap the
entire descendant process group before advancing. That infrastructure defect
is independent of model conversion accuracy and must receive a focused
process-tree regression test before another long corpus run.

## TFLite causal split recorded before source correction

The detached-main comparisons reduced the 14 baseline-pass observations to
four confirmed `fb-refactor5` TFLite regressions, six pre-existing or stale-
baseline limitations, two accepted DEIM results, and two new timeout
exclusions. No converter or exporter source was modified before this split was
recorded.

### Confirmed current-branch regressions

| Model | Tier | Current result | Current evidence | Detached `main` result |
| --- | ---: | --- | --- | --- |
| `nighttime_dehaze_realnight_1x3x180x320.onnx` | 2 | missing TFLite report in 19.485 s | invalid CONV preparation, `input_channel % filter_input_channel != 0 (90 != 0)`, signature `96308881e0a1433594a80e7a9c95802e2816af9248a36f0b752780119782594f` | TFLite pass in 40.828 s |
| `rtmpose_wholebody_l_1x3x256x192.onnx` | 2 | missing TFLite report in 14.032 s | invalid CONV preparation, `(512 != 0)`, signature `8e865d8383375fba4c8609fdfdaa902004c00bf910d0261b94b7a0dca809b828` | TFLite pass, max abs `4.67896e-06`, in 13.974 s |
| `rtmpose_wholebody_m_1x3x256x192.onnx` | 2 | missing TFLite report in 10.944 s | invalid CONV preparation, `(384 != 0)`, signature `6135fd17a10bc82106b4737b1ae84eb1e296b737bc8e05323a05b613e8b8055c` | TFLite pass in 11.341 s |
| `model_70_2023_0303_32_2_1_grid_sample_bilinear_no_pad_10_squeeze.onnx` | 3 | missing TFLite report in 58.384 s | `Input tensor 1048 lacks data`, signature `b231a3a54d114ccdfec6da4fd85dab9a63efe667376c7ca747f010fcd4ec1dbe` | TFLite pass in 75.225 s |

All four comparisons used zero SWAP. The RTMPose failures form one shape/
layout family; the nighttime dehaze and GridSample failures remain separate
families until a causal commit or pass is identified. Safe correction requires
bisecting these representatives across the post-`main` extraction commits,
then proving the fix on all four plus unaffected passing sentinels.

### Not attributable to `fb-refactor5`

- `silero_vad.onnx`: current and detached `main` both fail TFLite preparation
  because ADD inputs have shapes `[1,1,3,129]` and `[1,1,5,129]`. The managed
  pass baseline is stale for the exact current command/environment.
- `conv_tasnet_dnn.onnx` and
  `conv_tasnet_dnn_r_1_1_2_44100.onnx`: the representative fails on both
  current and detached `main` because a five-dimensional input is paired with
  a four-dimensional transpose permutation. The two current entries share
  signature `c11b375dfce50d047104d5aab700fb718bdc3495f13af251536a9a4259a89964`.
- `dpt_levit_224_224x224_.onnx` and
  `dpt_levit_224_224x224_org.onnx`: current and detached `main` both fail the
  direct lowerer with Softmax axis 3 applied to an inferred rank-3 tensor.
  The preceding `onnxsim` warning is not the fatal cause. The managed pass
  baseline is stale for the exact current command/environment.
- `mirnet_180x320.onnx`: current reaches an `Input tensor 226 lacks data`
  evaluation failure in 80.057 seconds, while detached `main` reaches the
  600-second timeout without producing a report. It is not a new branch
  regression and should not be used as a short validation sentinel.
- `deim_hgnetv2_n_wholebody28_1250query_fp16.onnx` and
  `deim_hgnetv2_s_wholebody28_ft_1250query_fixed.onnx`: retain the explicit
  user-approved success treatment for near-tied TopK index instability.

All representative detached-main comparisons used zero SWAP. The temporary
main worktree is pinned to `a8640153` and is not a source of repository
changes.
