# `flatbuffer_direct` Tier 0–4 measured-quick regression check — 2026-07-16

## Pre-fix outcome

This document freezes the regression evidence **before any source fix**. The
`fb-refactor5` run exercised 49 previously measured short-runtime models in a
fixed order with exactly one converter/inference subprocess at a time. It
produced 41 passes, three known accuracy failures, four missing TFLite accuracy
reports, and one 45-second quick-ceiling timeout. No conversion process used
SWAP, and there were no converter non-zero exits.

Two semantic regression candidates and one runtime-profile candidate require
focused branch attribution:

- `text_detection_en_ppocrv3_2023may_int8.onnx` changed from a known numeric
  accuracy failure to a TFLite model that LiteRT cannot execute;
- `imageclassifier.onnx` changed from pass to a TFLite model that LiteRT cannot
  execute;
- `hybridnets_384x640_sim.onnx` exceeded the 45-second quick ceiling after
  previously passing in 29.351 seconds.

These are candidates, not yet confirmed `fb-refactor5` regressions. No source
change, pass disablement, profile acceptance, or baseline promotion has been
made in response. The immutable machine-readable pre-fix evidence is
[`docs/baselines/flatbuffer_direct_quick_tier0_4_2026-07-16_result.json`](baselines/flatbuffer_direct_quick_tier0_4_2026-07-16_result.json).

Follow-up branch attribution and the narrowly guarded fix are now complete.
Both semantic regressions are resolved, the `hybridnets` timeout was classified
as one-off runtime variance, and no SWAP exclusion was added. See
[Follow-up resolution](#follow-up-resolution) and the machine-readable
[`docs/baselines/flatbuffer_direct_quick_tier0_4_2026-07-16_followup.json`](baselines/flatbuffer_direct_quick_tier0_4_2026-07-16_followup.json).

## Scope and selection

The selection manifest is
[`docs/baselines/flatbuffer_direct_quick_tier0_4_2026-07-16.json`](baselines/flatbuffer_direct_quick_tier0_4_2026-07-16.json).
It was derived from the 2026-07-14 sequential run by retaining only models that
completed in under 30 seconds and recorded zero model-process SWAP. Managed
timeouts, explicit user exclusions, Tier 5, and the following measured-long
models were not executed:

- `model_70_2023_0303_32_2_1_grid_sample_bilinear_no_pad_10_squeeze.onnx`
  (37.005 seconds);
- `hitnet_middlebury_d400.onnx` (54.830 seconds);
- `shadowformer_istd_160x240.onnx` (54.111 seconds);
- `d3net_dnn_double_44.onnx` (60-second timeout);
- `nchw.onnx` (60-second instrumented timeout).

The resulting Tier 0–4 model counts were 12, 12, 12, 9, and 4. The manifest
records all model options, and the result JSON records the SHA-256 of every
untracked root-corpus ONNX file. This prevents an unrecorded model-file change
from being confused with a branch change during follow-up.

The run used branch `fb-refactor5` at commit
`48f92183574c0d726696d1cf4b6eef8a90ce53e2`. Its command was:

```text
uv run --no-sync python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir . \
  -o <TEMP> \
  --regression_profile docs/baselines/flatbuffer_direct_quick_tier0_4_2026-07-16.json \
  --tflite_only \
  --timeout_sec 45
```

The bulk command exited with status 1 because known non-passes, missing
reports, and timeouts are strict failures by design. The runner monitored the
Linux `VmSwap` value of the converter subprocess tree and would have terminated
a model on any non-zero value.

## Result summary

| Tier | Selected | Pass | Known numeric fail | Missing report | 45 s timeout | Total time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 10 | 0 | 2 | 0 | 34.892 s |
| 1 | 12 | 11 | 1 | 0 | 0 | 50.278 s |
| 2 | 12 | 10 | 1 | 1 | 0 | 80.070 s |
| 3 | 9 | 7 | 1 | 1 | 0 | 114.471 s |
| 4 | 4 | 3 | 0 | 0 | 1 | 83.085 s |
| **Total** | **49** | **41** | **3** | **4** | **1** | **362.796 s** |

Median duration was 4.777 seconds. Apart from the user-approved DEIM TopK
index result, the largest maximum absolute error among passing models was
`0.0443115234375` for
`pose_estimation_mediapipe_2023mar_int8bq.onnx`, below the required `1e-1`
ceiling. DEIM remains pass through the recorded user-approved acceptance; its
raw maximum absolute error is `27.0` because near-tied TopK indices differ.

The observed duration distribution was:

| Runtime | Models |
| --- | ---: |
| under 5 seconds | 27 |
| 5 to under 15 seconds | 16 |
| 15 to under 30 seconds | 5 |
| 30 to under 45 seconds | 0 |
| 45 seconds or more | 1 |

## Regression candidates recorded before fixes

### `text_detection_en_ppocrv3_2023may_int8.onnx`

The 2026-07-14 run generated an executable TFLite model and retained the known
maximum absolute error `0.7411765307188034`. The current conversion still exits
zero and writes float32/float16 TFLite plus tensor correspondence metadata, but
LiteRT rejects float32 inference at CONV_2D node 1324:

```text
input_channel % filter_input_channel != 0 (24 != 0)
Node number 1324 (CONV_2D) failed to prepare.
```

The current float32 TFLite is 945,760 bytes with SHA-256
`995709a457c6fdddeda675af3b0ae1be91a3f70f0fb720eb7e7c40e2377b83b3`.
The instrumented run recorded changes from
`canonicalize.scalar_clamp_relu0to1` and
`layout.singleton_channel_transpose_as_reshape`. This does not prove either
pass is causal; both must be compared with the last passing branch before any
guard or rewrite is changed.

### `imageclassifier.onnx`

The 2026-07-14 run passed with maximum absolute error
`6.67572021484375e-06`. The current conversion exits zero and writes all normal
TFLite artifacts, but LiteRT rejects float32 inference at CONV_2D node 454:

```text
input_channel % filter_input_channel != 0 (192 != 0)
Node number 454 (CONV_2D) failed to prepare.
```

The current float32 TFLite is 19,035,924 bytes with SHA-256
`0ae62a5840325d49ae24410105bc0120e7328e15c87d1ef62a1deb7a65e61004`.
Eight passes reported a graph change: constant-input Cast cleanup,
consecutive-Reshape cleanup, duplicate-Transpose fan-out cleanup,
Concat→unary→Conv NHWC layout recovery, singleton-channel Transpose recovery,
Transpose-chain cleanup, unary fan-out bridge recovery, and unary Transpose
passthrough. Focused branch comparison must identify the first divergent IR
digest and invalid CONV input/filter metadata before a fix is proposed.

### `hybridnets_384x640_sim.onnx`

This model passed in 29.351 seconds on 2026-07-14 but reached the current
45-second ceiling at 45.174 seconds. The timeout occurred after float32 and
float16 TFLite files and the tensor correspondence report were written. The
float32 artifact is 55,136,628 bytes with SHA-256
`413855e793ba823ff7a4d24cc05c029435391959715ec6bce94c3652c58b9d7b`.
The timeout logs are empty because the subprocess output was still buffered
when it was terminated. No SWAP was observed.

This may be runtime variance, accuracy-evaluation cost, or a branch performance
regression. It is not a semantic failure yet. A single focused run with enough
headroom and timing/pass-metric comparison against the prior branch is needed.
If it remains slow without a safe performance fix, it should be removed from
the quick profile rather than reclassified as a conversion failure.

## Preserved known behavior

The following short known non-passes retained their prior behavior:

| Model | Current classification | Current maximum absolute error |
| --- | --- | ---: |
| `string_normalizer_11.onnx` | missing report; stock LiteRT custom op | n/a |
| `version-RFB-320-int8.onnx` | known numeric fail | `0.14972957409918308` |
| `tmp_alike_debug3.onnx` | known exact-equality instability | `1.0` |
| `best.onnx` | known detector decode amplification | `58.7506103515625` |

`silero_vad.onnx` remains the previously documented managed-baseline mismatch,
not a new `fb-refactor5` finding. It fails exactly at the same LiteRT ADD shape
preparation error recorded on 2026-07-14. Its current and prior quick-run
classifications are both `missing_tflite_report`.

## SWAP and execution safety

No selected model recorded model-process SWAP; every entry has
`peak_swap_kib = 0`. The host had approximately 4.3 GiB of pre-existing global
SWAP use before the run, and that value remained approximately unchanged. It is
not used for per-model classification: the authoritative measurement is the
converter subprocess-tree `VmSwap` captured by the runner.

All 49 conversions and inference checks were sequential. No ProcessPool,
parallel worker group, or overlapping converter subprocess was used.

## Validation completed before attribution

- `uv run --no-sync pytest -q tests/test_flatbuffer_direct_bulk_runner.py`:
  36 passed;
- regression manifest parser: 49 active models, Tier 0–4 only, concurrency 1;
- manifest and result JSON parse/shape checks: passed;
- fixed-order bulk run: completed all 49 entries;
- model-process SWAP detection: zero entries;
- conversion non-zero exits: zero entries.

## Pre-fix required next step

The detailed evidence above and the result JSON must be committed before
focused investigation begins. After that checkpoint, run only the two semantic
candidates on `origin/main` (the merged `fb-refactor4` baseline) with identical
model files, options, environment, and sequential execution. Compare the first
divergent pass/IR state and TFLite CONV tensor metadata. Run `hybridnets` once
per branch with a longer diagnostic ceiling to distinguish timing variance from
a repeatable regression. Only after branch attribution should a narrowly
guarded fix be implemented and verified against the affected models plus a
small set of already-passing structural neighbors.

## Follow-up resolution

### Branch attribution

Focused runs used the identical retained ONNX files, profile options, uv
environment, TFLite-only output, and sequential execution on `origin/main` at
commit `a86401539c57188a49b1ce0481c9e0d978a05aa6`. The baseline reproduced the
expected behavior:

| Model | `origin/main` | Pre-fix `fb-refactor5` |
| --- | --- | --- |
| `text_detection_en_ppocrv3_2023may_int8.onnx` | known `tflite_fail`, max abs `0.7411765307188034` | missing report; invalid CONV input channels |
| `imageclassifier.onnx` | pass, max abs `6.67572021484375e-06` | missing report; invalid CONV input channels |

A sequential git bisect identified
`78ba42aed51907f824124cb332814aaff507a1b7` (`Index NCHW concat transpose-conv
repair`) as the first bad commit for both models.

The indexed implementation had made two valid production patterns stricter
than the extracted legacy semantics:

- it rejected a corrected NCHW Concat/activation tensor whenever it fed more
  than one Transpose→Conv branch, even when every sibling used the same
  `[0,2,3,1]` layout adapter and required the same channel count;
- it accepted only an embedded ndarray filter and rejected the QLinearConv
  lowering form where a constant INT8 filter is shape-preservingly CAST to the
  runtime FLOAT32 Conv filter.

### Narrow fix

The indexed owner remains in
`onnx2tf/tflite_builder/passes/concat_transpose_conv_layout.py`; no legacy full
graph producer/consumer scan was restored. The repair now:

- permits fan-out only when every sibling is the same constant layout
  Transpose contract or a Conv-like consumer with the same required input
  channel count;
- accepts only a strict single-input CAST whose source is an embedded constant
  with an identical filter shape;
- still rejects public boundary tensors, unrelated side consumers, produced
  permutation tensors, duplicate producers, incompatible channels, and
  non-constant filter storage;
- collects all compatible plans before mutating the shared Concat, ensuring
  that every sibling branch receives consistent shape metadata.

### Post-fix results

| Model | Post-fix result | Max abs | Artifact comparison |
| --- | --- | ---: | --- |
| `text_detection_en_ppocrv3_2023may_int8.onnx` | known `tflite_fail` restored | `0.7411765307188034` | executable TFLite and report restored |
| `imageclassifier.onnx` | pass restored | `6.67572021484375e-06` | float32 TFLite byte-identical to `origin/main` |

Four already-passing structural neighbors were then run sequentially:
`face_detection_yunet_2023mar_int8.onnx`, `FastestDet.onnx`,
`yolox_nano_with_post.onnx`, and
`human_segmentation_pphumanseg_2021oct_org.onnx`. All four passed, and every
float32 TFLite SHA-256 remained identical to the pre-fix bulk artifact. No
model-process SWAP was detected.

### `hybridnets` timing attribution

With a 90-second diagnostic ceiling, `hybridnets_384x640_sim.onnx` passed in
25.559 seconds on `fb-refactor5` and 25.912 seconds on `origin/main`. Both runs
reported maximum absolute error `0.0002593994140625`, emitted the same float32
TFLite SHA-256
`413855e793ba823ff7a4d24cc05c029435391959715ec6bce94c3652c58b9d7b`,
and used zero model-process SWAP. The initial 45-second timeout is therefore a
one-off runtime variation, not a branch-specific regression. The model remains
eligible for the measured-quick profile.

### Post-fix validation

- indexed Concat→Transpose→Conv repair plus full architecture tests: 232
  passed;
- TensorFlow import blocker direct conversion and direct `-cotof`: 2 passed;
- affected-model sequential conversion: both expected classifications
  restored, SWAP 0;
- four already-passing neighboring models: 4 passed, byte-identical artifacts,
  SWAP 0;
- `hybridnets` branch timing comparison: both passed, equivalent accuracy and
  artifact, SWAP 0.

The full 49-model corpus was intentionally not rerun after this narrow fix.
The complete pre-fix run is retained, while post-fix conversion work was kept
to the affected models and small structural-neighbor set as requested.

## Full post-extraction checkpoint rerun at `f5a40947`

After the subsequent Split/mixed-Concat, general Concat input-adapter, and
Slice/Logistic/Concat/Reshape-tail extractions, the same 49-model manifest was
rerun in full at commit `f5a40947988cdb842c2f4015eb7237e905afdeb7`.
This was a regression check only: the worktree was clean before execution, and
no production source was changed in response before the evidence was
classified and recorded.

The fixed-order, single-process command completed all 49 models in 390.255
seconds. It produced 43 passes and the same six known strict non-passes. There
were no timeouts, conversion nonzero exits, missing models, or process-tree
SWAP detections. The runner exited 1 only because known numeric failures and
missing reports remain strict failures by design.

| Tier | Selected | Pass | Known numeric fail | Known missing report | Total time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 10 | 0 | 2 | 36.898 s |
| 1 | 12 | 11 | 1 | 0 | 55.517 s |
| 2 | 12 | 10 | 2 | 0 | 94.182 s |
| 3 | 9 | 8 | 1 | 0 | 136.367 s |
| 4 | 4 | 4 | 0 | 0 | 67.291 s |
| **Total** | **49** | **43** | **4** | **2** | **390.255 s** |

All 49 classifications match the corrected expected state. All 47 models that
emitted a numeric report reproduce the exact recorded expected maximum
absolute error. In particular:

- `text_detection_en_ppocrv3_2023may_int8.onnx` retains its executable TFLite
  and known `0.7411765307188034` mismatch;
- `imageclassifier.onnx` remains a pass at `6.67572021484375e-06`;
- `hybridnets_384x640_sim.onnx` remains a pass at
  `0.0002593994140625` and completed in 27.612 seconds;
- DEIM remains accepted under the user-approved near-tied TopK-index policy,
  with its unchanged raw maximum absolute error of `27.0`;
- the largest ordinary passing error remains
  `0.0443115234375` for the MediaPipe pose model, below `1e-1`.

The only aggregate pass-metric change among the 48 models with comparable
pre/post metrics is beneficial: `sinet_320_op.onnx` reduced preflight operator
visits from 33,303 to 31,906. Its event/status counts, snapshots, state builds,
classification, and maximum error are unchanged. This is consistent with the
new indexed lookup replacing repeated traversal without changing semantics.

Two single-sample runtime increases were recorded before deciding whether any
action was warranted. `text_detection_en_ppocrv3_2023may_int8.onnx` moved from
8.323 to 14.432 seconds, and DEIM moved from 24.043 to 37.630 seconds. Neither
changed pass metrics, classification, maximum error, or SWAP use. They are
therefore runtime observations rather than confirmed regressions; no source
fix or baseline relaxation was made. DEIM should be measured once more before
the next quick-profile refresh, and removed from that quick profile only if its
over-30-second runtime is reproducible.

The compact per-model machine-readable evidence is
[`docs/baselines/flatbuffer_direct_quick_tier0_4_f5a40947_result.json`](baselines/flatbuffer_direct_quick_tier0_4_f5a40947_result.json).
The temporary generated TFLite artifacts are not part of the retained
evidence and may be deleted after the JSON and documentation checkpoint is
validated.

The current checkpoint validation also includes:

- `uv run --no-sync pytest -q tests/test_flatbuffer_direct_bulk_runner.py`:
  36 passed;
- manifest parsing: 49 active entries, Tier 0–4 only, concurrency 1;
- retained JSON parsing and exact 49-entry comparison with runner state:
  passed;
- `git diff --check`: passed.
