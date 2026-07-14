# `flatbuffer_direct` Tier 0–4 quick regression check — 2026-07-14

## Outcome

No regression specific to `fb-refactor4` was confirmed in the selected
short-runtime Tier 0–4 corpus. The run exercised 54 models sequentially. It
produced 46 passes, four expected accuracy failures, two missing reports, and
two 60-second quick-ceiling timeouts. There were no conversion errors and no
SWAP detections.

The managed profile expected 49 passes, four accuracy failures, and one missing
report. Three expected passes did not pass the bulk run, but focused comparison
did not attribute any of them to `fb-refactor4`:

- `silero_vad.onnx` fails identically on `fb-refactor3` and emits a
  byte-identical TFLite artifact on both branches;
- `d3net_dnn_double_44.onnx` reaches the same 60-second ceiling on
  `fb-refactor3` and is not suitable for the quick profile;
- `nchw.onnx` passes direct execution on both branches with byte-identical
  artifacts and metrics. Only the instrumented bulk run reaches 60 seconds.

No converter or test-runner source was changed in response to these findings.

The immutable selection manifest is
[`docs/baselines/flatbuffer_direct_quick_tier0_4_2026-07-14.json`](baselines/flatbuffer_direct_quick_tier0_4_2026-07-14.json),
and the complete machine-readable result is
[`docs/baselines/flatbuffer_direct_quick_tier0_4_2026-07-14_result.json`](baselines/flatbuffer_direct_quick_tier0_4_2026-07-14_result.json).

## Scope and selection

The source was the managed Tier 0–4 profile. Recorded timeouts, explicit user
exclusions, known long-running non-passes, and Tier 5 were excluded before
selection. The 54 models were selected with smaller file size, characterized
input contracts, and operator-family diversity as structural runtime proxies.
Short known non-passes were retained so their classification and accuracy could
also be checked. The fixed tier quotas were 12, 12, 12, 10, and 8.

The run used commit `afb5bb55bedfc89119bc2064113c1720ed9def16` on
`fb-refactor4`. Focused comparisons used
`c52bc1699b4c7a11a03a535e0b7f10315e1292bd` from
`origin/fb-refactor3` in a detached temporary worktree.

The authoritative bulk command was:

```text
uv run --no-sync python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir . \
  -o <TEMP> \
  --regression_profile docs/baselines/flatbuffer_direct_quick_tier0_4_2026-07-14.json \
  --tflite_only \
  --timeout_sec 60
```

Execution was strictly sequential with one converter/inference subprocess at a
time. PyTorch and TensorFlow artifacts were not requested. A representative
`-cotof` import-blocker test passed after the run, and the core import scan found
no TensorFlow imports. The runner exited with status 1 because known non-passes
and timeout results are strict failures by design.

## Result summary

| Tier | Selected | Pass | Known TFLite fail | Missing report | 60 s timeout | Total time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 10 | 0 | 2 | 0 | 29.683 s |
| 1 | 12 | 11 | 1 | 0 | 0 | 47.854 s |
| 2 | 12 | 10 | 2 | 0 | 0 | 88.854 s |
| 3 | 10 | 9 | 1 | 0 | 0 | 161.417 s |
| 4 | 8 | 6 | 0 | 0 | 2 | 303.336 s |
| **Total** | **54** | **46** | **4** | **2** | **2** | **631.144 s** |

Median duration was 5.454 seconds. Apart from the user-approved DEIM result,
the largest maximum absolute error among passing models was `0.0443115234375`
for `pose_estimation_mediapipe_2023mar_int8bq.onnx`, below the required `1e-1`
ceiling. DEIM was counted as pass through its recorded profile acceptance;
its raw maximum absolute error was `27.0` because near-tied TopK indices are
intentionally accepted for that model.

The observed runtime distribution was:

| Runtime | Models |
| --- | ---: |
| under 5 seconds | 26 |
| 5 to under 15 seconds | 16 |
| 15 to under 30 seconds | 7 |
| 30 to under 60 seconds | 3 |
| 60 seconds or more | 2 |

## Detailed findings

### `silero_vad.onnx`: managed baseline mismatch, not branch-specific

The managed profile records a pass with maximum absolute error
`1.375097781419754e-06`. The current run instead generated the TFLite artifact
but could not produce the accuracy report because LiteRT rejected ADD node 27:

```text
Given shapes, [1,1,3,129] and [1,1,5,129], are not broadcastable.
Node number 27 (ADD) failed to prepare.
```

The exact profile options were present: `-kat input state sr`. Repeating the
same command with `fb-refactor3` produced the same error. Both branches emitted
the same float32 artifact SHA-256:

```text
6135e793229c8ce41fc10b26c3dddeb359531b5a117b35b30e52f99a9f0979dd
```

This disproves an `fb-refactor4`-specific converter regression. The remaining
possibilities are environment/runtime drift or drift in the untracked root
model corpus relative to the run that promoted the model. The root model's
current SHA-256 is
`7042fc01e3ae6191e025e1b614f1b2c126b6b937cb250388a52553a67a2dc8d2`.
The managed profile does not contain a per-model content hash, so that cause
cannot be distinguished from the recorded data alone.

### `d3net_dnn_double_44.onnx`: unsuitable for a 60-second quick run

The managed profile records a pass with maximum absolute error
`7.450580596923828e-08`. The instrumented `fb-refactor4` run reached 60.051
seconds. A direct `fb-refactor3` run with the same model options also reached
the 60-second ceiling. No SWAP was observed in the authoritative bulk run.

This is not evidence of an `fb-refactor4` regression. It should be omitted from
future 60-second quick profiles while remaining in the full managed profile.
No longer rerun was attempted because this check intentionally excludes
long-running models.

### `nchw.onnx`: bulk instrumentation consumes timeout headroom

The authoritative bulk run reached 60.149 seconds. Focused direct runs then
completed successfully:

| Branch | Duration | Max absolute error | Result |
| --- | ---: | ---: | --- |
| `fb-refactor3` | 40.31 s | `1.3709068298339844e-06` | pass |
| `fb-refactor4` | 40.88 s | `1.3709068298339844e-06` | pass |

Both branches emitted float32 TFLite SHA-256
`838b680071df322d0e10fbfced81b734595f73892400248b5dc816a133fe3155`.
The approximately 1.4% direct-run timing difference is not a meaningful
regression from a single non-warmed sample. The false timeout occurs only when
the bulk runner enables internal pass-metrics collection. The model should be
omitted from the 60-second quick profile or given instrumentation headroom; it
must remain a pass in the full regression profile.

### Known non-passes

All five selected known non-passes retained their expected classification and
behavior:

| Model | Managed reason | Baseline max abs | Current max abs |
| --- | --- | ---: | ---: |
| `string_normalizer_11.onnx` | unsupported stock TFLite string normalizer | n/a | n/a |
| `version-RFB-320-int8.onnx` | ONNX Runtime U8S8 saturating accumulation | `0.14972958900034428` | `0.14972957409918308` |
| `text_detection_en_ppocrv3_2023may_int8.onnx` | requantization outliers amplified by transpose convolution | `0.7411765307188034` | `0.7411765307188034` |
| `tmp_alike_debug3.onnx` | exact-equality instability from float accumulation | `1.0` | `1.0` |
| `best.onnx` | QDQ rounding outliers amplified by detector decode | `58.7506103515625` | `58.7506103515625` |

`text_detection_en_ppocrv3_2023may_int8.onnx` has a diagnostic signature drift:
the current raw signature selects a non-fatal `onnxsim` warning before the
accuracy report, while its classification and exact maximum error remain
unchanged. This is a normalized-error selection problem, not a conversion or
accuracy regression. It is recorded without changing the runner.

The profile hashes for `string_normalizer_11.onnx` and
`version-RFB-320-int8.onnx` are semantic hashes of
`classification | managed reason`, not raw runner signatures. Their different
raw hashes therefore do not represent signature regressions.

## Future quick-run selection

The executed manifest is intentionally left immutable so its content hash
continues to match the recorded result. A future quick manifest should exclude:

- `d3net_dnn_double_44.onnx`, which reaches 60 seconds on both branches;
- `nchw.onnx`, whose pass-metrics-instrumented run reaches 60 seconds even
  though direct conversion passes in about 40 seconds.

The following passed but are weak quick-run candidates and should be removed if
a stricter runtime budget is desired:

- `hitnet_middlebury_d400.onnx`: 54.830 seconds;
- `shadowformer_istd_160x240.onnx`: 54.111 seconds;
- `model_70_2023_0303_32_2_1_grid_sample_bilinear_no_pad_10_squeeze.onnx`:
  37.005 seconds.

No model triggered SWAP. Consequently no new managed SWAP exclusion is needed.

## Validation and retained evidence

Before the corpus run, the manifest loaded successfully and the bulk-runner
test module passed all 36 tests. After the run, both JSON documents were parsed
successfully. Focused comparison verified artifact hashes and accuracy metrics
for `silero_vad.onnx` and `nchw.onnx` across both branches.

The temporary 1.1 GiB bulk artifacts and detached comparison worktree are not
part of the retained evidence. The manifest, full per-model result JSON, and
this report contain the commands, classifications, durations, signatures,
accuracy values, artifact hashes needed for follow-up.
