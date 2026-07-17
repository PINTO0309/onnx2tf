# `fb-refactor6` Tier 0-4 TFLite and native PyTorch regression check

## Outcome

All 379 managed Tier 0-4 models that are not recorded as a timeout or an
explicit exclusion completed one strictly sequential current-branch run.
Every model finished below the 600-second ceiling and every monitored process
tree had zero `VmSwap`.

No `fb-refactor6`-specific TFLite regression is present in the completed
corpus. The current TFLite component has exactly the final `fb-refactor5`
problem set: 347 raw passes, 20 known numeric failures, ten missing reports,
and two known conversion errors. Two additional raw numeric failures are the
user-approved DEIM near-tied TopK index cases and remain accepted as TFLite
successes.

No `fb-refactor6`-specific native PyTorch regression is confirmed. Relative to
the documented post-correction `fb-refactor5` combined accounting, one
`pytorch_fail` became a combined pass and every other combined count is
unchanged. The branch does not modify the PyTorch exporter. Eight previously
compared PyTorch failure representatives retain the same messages and numeric
behavior. This result does not reclassify the many pre-existing native
PyTorch limitations as successes.

The run exposed one regression-runner infrastructure defect: its resumable
state JSON is overwritten in place. A concurrent read observed a partially
written 17 MiB file once. The writer recovered normally and no model result
was lost, but resumable state should be published through atomic replacement.
This problem and its evidence were recorded before any source correction.

The complete condensed result is
`docs/baselines/flatbuffer_direct_tier0_4_tflite_native_pytorch_fb6_13fcf7ce.json`.
It contains all 379 model results, durations, component metrics, SWAP values,
and normalized signatures without retaining generated models or artifacts.

## Scope fixed before execution

- Branch: `fb-refactor6`.
- Checkpoint: `13fcf7ce907a4bd935008afa3b7c50ada46d57b8`.
- Managed profile:
  `docs/baselines/flatbuffer_direct_active_tier0_4.json`.
- Profile file SHA-256:
  `b70885ce0b22f83014355739be4235efce7383b607a2eba75bdc1293b80cacbf`.
- Runner-normalized profile content SHA-256:
  `5f733297e09db07a5d4b9bd5767cece677d50cf247bae9313eec2c0dcab852e4`.
- Ordered active-model SHA-256:
  `51eb37986e9715b9486ae19b58d867c724a4e8989d0a09111d6406c2cbcc7545`.
- Included graph size: 1-1,999 ONNX nodes, or Tier 0 through Tier 4.
- Per-model wall-clock ceiling: 600 seconds.
- Inference/conversion concurrency: one.
- Requested artifacts: TFLite plus the native PyTorch package only. The run
  used `-fdopt` and did not request TorchScript, Dynamo ONNX, or
  ExportedProgram.
- Every converter ran in an isolated POSIX process group monitored through
  Linux `/proc`; nonzero descendant `VmSwap` would terminate and classify the
  complete group before the next model.
- TensorFlow was not requested or imported by the direct conversion command.
- The run used `uv run --no-sync` with Python 3.12.12 and Torch 2.11.0+cpu
  from the repository `.venv`. `PYTHONNOUSERSITE=1`, empty `PYTHONPATH` and
  `LD_LIBRARY_PATH`, and one OMP/MKL thread prevented contamination from the
  host's Python 3.10 Torch installation.

The profile contains 420 Tier 0-4 records. It excludes 29 recorded timeouts
and twelve explicit exclusions, leaving 379 active models. All 379 model files
were present. Their fixed tier distribution was 119/84/105/49/22.

## Temporary-storage policy

Before the run, only prior Goal-owned, reproducible `/tmp` entries matching
the scoped `onnx2tf*`, `fb5_*`, `fb6-*`, and `main_compare_*` names were
removed. This recovered approximately 4 GiB without touching unrelated
temporary data.

The authoritative run lived under one uniquely named `/tmp` directory. After
each entry was persisted, an independent non-inference cleanup watcher kept
the command logs, pass metrics, accuracy JSON, correspondence/error JSON/CSV,
and normalized state while deleting the staged ONNX, TFLite files, schema
copies, and generated native package. It never touched the active model.
Large ONNX Runtime memmaps were left in place while open and disappeared when
their owning converter exited. After the condensed JSON and this report were
verified, the uniquely named run directory and the atomic-write stress-test
directory were removed. No temporary directory owned by this run remains;
approximately 60 GiB was available on the backing filesystem after cleanup.

## Execution results

The runner intentionally returned nonzero because strict native-PyTorch
failures remain. It completed normally rather than crashing.

| Metric | Result |
| --- | ---: |
| Models completed | 379 |
| Sum of per-model wall time | 7,542.595 s |
| Median per-model wall time | 8.746 s |
| Maximum per-model wall time | 191.394 s |
| Models at or above 600 s | 0 |
| Models with nonzero SWAP | 0 |
| Converter nonzero exits | 2 |

Per-tier timing remained well below the ten-minute per-model boundary:

| Tier | Models | Total seconds | Median seconds | Maximum seconds | Slowest model |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 119 | 777.909 | 3.612 | 62.637 | `bertsquad-12_onehot_org.onnx` |
| 1 | 84 | 983.873 | 7.361 | 71.105 | `anime-gan-v2_org.onnx` |
| 2 | 105 | 2,123.154 | 11.322 | 191.394 | `yolov3-12-int8.onnx` |
| 3 | 49 | 2,129.678 | 30.011 | 185.445 | `stock_frcnn_11.onnx` |
| 4 | 22 | 1,527.982 | 58.461 | 181.402 | `wd-v1-4-moat-tagger-v2.onnx` |

The combined TFLite/native-PyTorch classifications are observations rather
than branch-regression counts:

| Combined classification | Models |
| --- | ---: |
| `pass` | 137 |
| `missing_pytorch_report` | 187 |
| `pytorch_fail` | 31 |
| `both_fail` | 12 |
| `missing_both_reports` | 6 |
| `missing_tflite_report` | 4 |
| `conversion_error` | 2 |
| **Total** | **379** |

Separating the two artifact components gives:

| Component result | TFLite | Native PyTorch |
| --- | ---: | ---: |
| Accuracy pass | 347 | 138 |
| Accuracy fail | 20 | 46 |
| Missing report | 10 | 195 |
| Conversion error before reports | 2 | 2 |

The native-PyTorch missing-report total in the table excludes the two common
conversion errors; native evaluation could not start for those models.

## TFLite baseline comparison

### Exact accounting

The managed baseline has 353 passes, 20 numeric failures, and six missing
reports among its active records.

- 345 of 353 baseline passes remained raw TFLite passes.
- Two more baseline passes are the explicitly accepted DEIM TopK index cases.
- Four baseline passes retain limitations already reproduced on detached
  `main`: `silero_vad.onnx`, two Conv-TasNet DNN variants, and MIRNet.
- Two baseline passes retain the DPT-LeViT Softmax rank error already
  reproduced on detached `main`.
- Eighteen of 20 known numeric failures remained numeric failures.
- Two known failures improved and remained improved:
  `yolox_nano.onnx` and `rtdetrv4_s.onnx`.
- All six known missing reports remained missing for the same model set.

This produces the same final component accounting as `fb-refactor5` after its
four causally attributed TFLite repairs: 347 raw passes, 20 raw failures, ten
missing reports, and two conversion errors.

### Baseline-pass raw non-passes

These eight observations were not modified during the run. None is an
`fb-refactor6` regression.

| Model | Current TFLite result | Seconds | Evidence and disposition |
| --- | --- | ---: | --- |
| `silero_vad.onnx` | missing report | 7.290 | ADD shapes `[1,1,3,129]` and `[1,1,5,129]`; reproduced on `fb-refactor5` and detached `main` |
| `conv_tasnet_dnn.onnx` | missing report | 37.442 | five-dimensional input with four-dimensional transpose; signature `c11b375d...99464`, identical on detached `main` |
| `conv_tasnet_dnn_r_1_1_2_44100.onnx` | missing report | 36.514 | same `c11b375d...99464` inherited failure |
| `deim_hgnetv2_n_wholebody28_1250query_fp16.onnx` | raw fail, max abs 27 | 29.292 | user-approved near-tied TopK index instability; accepted TFLite success |
| `deim_hgnetv2_s_wholebody28_ft_1250query_fixed.onnx` | raw fail, max abs 20 | 57.679 | user-approved near-tied TopK index instability; accepted TFLite success |
| `dpt_levit_224_224x224_.onnx` | conversion error | 6.069 | Softmax axis 3 on inferred rank 3; reproduced on detached `main`; signature `049c8637...f60c` |
| `dpt_levit_224_224x224_org.onnx` | conversion error | 6.194 | same inherited Softmax/rank error; signature `b3ce1954...05f0` |
| `mirnet_180x320.onnx` | missing report | 80.594 | `Input tensor 226 lacks data`; same as `fb-refactor5`, while detached `main` timed out at 600 s |

The six baseline missing-report models also remain the same:
`inverse_11.onnx`, `silero_vad (1).onnx`, `string_normalizer_11.onnx`,
`ssd_mobilenet_v1_12-int8.onnx`, `dynamics_rife_sim.onnx`, and
`conv_tasnet.onnx`.

### Known numeric failures and improvements

| Model | Current raw TFLite result | Maximum absolute error |
| --- | --- | ---: |
| `efficientnet-lite4-11-int8.onnx` | known fail | 0.013601 |
| `fcn-resnet50-12-int8.onnx` | known fail | 0.547120 |
| `text_recognition_CRNN_CN_2021nov_int8.onnx` | known fail | 0.148426 |
| `version-RFB-320-int8.onnx` | known fail | 0.149730 |
| `afhq_generator.v11.quant.onnx` | known fail | 0.190034 |
| `alike_l_opset11_192x320_post.onnx` | known fail | 290.000006 |
| `arcfaceresnet100-11-int8.onnx` | known fail | 0.368195 |
| `text_detection_en_ppocrv3_2023may_int8.onnx` | known fail | 0.741177 |
| `tmp_alike_debug3.onnx` | known fail | 1.0 |
| `tmp_alike_debug4.onnx` | known fail | 1.0 |
| `yolov3-12-int8.onnx` | known fail | 0.095639 |
| `yolov5s.onnx` | known fail | 0.329285 |
| `yolox_nano.onnx` | **improved pass** | 0.096283 |
| `best.onnx` | known fail | 58.750610 |
| `best_org.onnx` | known fail | 58.750610 |
| `dequantize_linear.onnx` | known fail | 58.750610 |
| `model_70_2023_0220_32_2_1_grid_sample_bilinear_sim.onnx` | known fail | 0.296917 |
| `model_grid_sample.onnx` | known fail | 0.245553 |
| `rtdetrv4_s.onnx` | **improved pass** | 0.00000495 |
| `bertsquad-12-int8.onnx` | known fail | 1.888175 |

Important repaired sentinels also retained their final values:

- nighttime dehaze: `0.000294536`;
- RTMPose wholebody L 256x192: `4.67896e-06`;
- RTMPose wholebody M 256x192: `2.72691e-06`;
- GridSample no-pad squeeze: `0.00802997`.

## Native PyTorch regression assessment

The authoritative `fb-refactor5` run was recorded before the four TFLite
repairs. Those four models had no PyTorch report both before and after repair,
so applying the documented TFLite-only transition changes four
`missing_both_reports` entries to `missing_pytorch_report`. The adjusted final
comparison is therefore:

| Combined classification | Adjusted final `fb-refactor5` | `fb-refactor6` |
| --- | ---: | ---: |
| `pass` | 136 | 137 |
| `missing_pytorch_report` | 187 | 187 |
| `pytorch_fail` | 32 | 31 |
| `both_fail` | 12 | 12 |
| `missing_both_reports` | 6 | 6 |
| `missing_tflite_report` | 4 | 4 |
| `conversion_error` | 2 | 2 |

Thus no combined category regressed and one native-PyTorch accuracy failure
improved to a combined pass. The raw historical per-model state was
intentionally cleaned after its earlier findings were documented, so this
count comparison is not represented as a byte-for-byte comparison of all 379
old entry objects.

The source comparison provides a second boundary: between the
`fb-refactor5` merge base `f2e5270b` and this checkpoint, no
`pytorch_exporter.py`, PyTorch emitter, generated-package runtime, or
PyTorch bulk-runner file changed. `fb-refactor6` extracts TFLite ModelIR
compatibility passes and adds owner/wrapper fingerprint tests.

The eight high-signal native failures previously compared with detached
`main` retain the same behavior:

| Representative | Current behavior, equal to the prior comparison |
| --- | --- |
| `GridSample_16.onnx` | invalid reshape to `[1,3,51076]`; signature `7886af49...d9d03` |
| `mspfn_320x480.onnx_cut3.onnx` | scalar passed as `other` to `torch.minimum`; signature `5944b474...12212` |
| `hair_segmenter.onnx` | tensor dimension 32 versus 128; signature `0edec0a0...aeeee` |
| `arcfaceresnet100-8.onnx` | PyTorch max abs `0.727120` |
| nighttime dehaze | Conv receives 320 rather than 64 channels; signature `1242d825...755c9` |
| RTMPose wholebody L 256x192 | dimension 60 versus 48; signature `d8b3ab71...bdca0` |
| RTMPose wholebody M 256x192 | Conv receives 1 rather than 192 channels; signature `2c1b4161...a066` |
| GridSample no-pad squeeze | batch matmul expects `[2,64]`, receives `[2,1]`; signature `5757cc6a...d1a1` |

This evidence supports zero confirmed `fb-refactor6`-specific native PyTorch
regressions. The current component totals, 138 passes, 46 accuracy failures,
195 missing reports, and two pre-report conversion errors, continue to expose
substantial inherited exporter work. They should not be hidden or treated as
successful conversions.

## Infrastructure problem recorded before correction

At 216 persisted entries, a monitoring read of `bulk_status.json` failed with:

```text
JSONDecodeError: Expecting value: line 540167 column 30 (char 17453524)
```

Two seconds later the same path contained a complete 216-entry state. The
runner continued through all 379 models, generated its final summary, and did
not lose or duplicate an entry. The cleanup watcher already treated JSON
decode failure as a transient read and never removed an unpersisted model.

The cause is local and deterministic: `_write_json()` opens the final path
with mode `w` and streams the increasingly large indented object directly.
Readers can therefore observe the path after truncation but before `json.dump`
finishes. The final full state was approximately 17 MiB because it embeds pass
metrics for every entry.

A safe correction is to write a same-directory temporary file, flush it, and
publish it with `os.replace()`. This uses only the standard library, preserves
the JSON schema and filenames, and gives readers either the old complete state
or the new complete state. It requires focused tests for replacement,
temporary-file cleanup on failure, and unchanged JSON formatting before it is
used in another long run.

No converter, lowerer, exporter, classification, profile, model, or accuracy
policy was changed before this report and condensed result were created.

## Correction applied after the evidence checkpoint

The complete pre-correction evidence above was committed and pushed as
`91a224c4` before the runner changed.

`_write_json()` now creates a unique same-directory file with exclusive
creation, preserves the existing destination mode when replacing a state,
writes the unchanged indented JSON representation, flushes and closes it, and
publishes it with `os.replace()`. Any serialization or replacement exception
closes the outstanding descriptor, removes the unpublished temporary file,
and re-raises while leaving the previous destination intact. A new file still
uses mode `0666` subject to the process umask, matching ordinary `open(...,
"w")` creation behavior.

The implementation deliberately does not add a disk `fsync` to every state
update. Complete close-before-replace is sufficient for concurrent readers,
while repeatedly synchronizing a state that reached roughly 17 MiB would add
avoidable I/O latency to a long corpus run. The correction addresses atomic
visibility, not power-loss durability, and does not change model scheduling,
inference concurrency, classification, resume schema, filenames, or JSON
formatting.

Focused verification after correction:

- bulk-runner suite: **42 passed**;
- runner, corpus-manifest, and architecture gate: **292 passed in 17.57s**;
- Ruff on the runner and its test: passed;
- `git diff --check`: passed;
- a same-process, non-inference reader/writer stress check published forty
  successive 8,000-entry states while a reader completed 1,742 JSON parses;
  no partial read occurred, the final generation was 39, and no temporary
  file remained.

No real-model rerun was required for this metadata-only writer correction.
The authoritative 379-model result remains tied to the pre-correction
converter checkpoint recorded above.

## Verification performed before source correction

- `uv run --no-sync pytest -q tests/test_flatbuffer_direct_bulk_runner.py`:
  **40 passed**.
- uv environment import check: Python 3.12.12, Torch 2.11.0+cpu from the
  repository `.venv`.
- Full managed current-branch conversion: **379/379 completed**, sequential,
  zero timeout, zero SWAP.
- TFLite component comparison: exact final `fb-refactor5` problem set and
  **zero confirmed `fb-refactor6` regression**.
- Native-PyTorch aggregate and representative comparison: no category
  degradation and **zero confirmed `fb-refactor6` regression**.

Atomic state publication is now implemented and verified. No model-conversion
correction is justified by this run.
