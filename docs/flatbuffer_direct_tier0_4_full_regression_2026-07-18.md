# `fb-refactor7` Tier 0-4 TFLite and native PyTorch regression check

## Outcome

All 379 managed Tier 0-4 models that are neither recorded timeouts nor explicit
user exclusions completed one strictly sequential `fb-refactor7` run. Every
model completed below the 600-second ceiling and every monitored converter
process tree recorded zero `VmSwap`.

No `fb-refactor7`-specific TFLite or native PyTorch regression was found. The
current run and the authoritative `fb-refactor6` result have identical model
order, component pass states, converter exit states, strict-pass states,
combined classifications, component counts, and normalized failure
signatures across all 379 models. No converter source correction was justified
or applied after this run.

The complete condensed evidence is
`docs/baselines/flatbuffer_direct_tier0_4_tflite_native_pytorch_fb7_0bf1bab4.json`.
It contains every model result, timing, component metric, SWAP value, pass
metrics, comparison result, profile identity, and source-state digest without
retaining generated models or packages.

## Fixed scope

- Branch: `fb-refactor7`.
- Converter checkpoint:
  `0bf1bab486da412797f27707a07e53f758235666`.
- Managed profile:
  `docs/baselines/flatbuffer_direct_active_tier0_4.json`.
- Profile file SHA-256:
  `b70885ce0b22f83014355739be4235efce7383b607a2eba75bdc1293b80cacbf`.
- Runner-normalized profile content SHA-256:
  `5f733297e09db07a5d4b9bd5767cece677d50cf247bae9313eec2c0dcab852e4`.
- Ordered active-model SHA-256:
  `51eb37986e9715b9486ae19b58d867c724a4e8989d0a09111d6406c2cbcc7545`.
- Included graph size: 1-1,999 ONNX nodes, Tier 0 through Tier 4.
- Per-model wall-clock ceiling: 600 seconds.
- Inference/conversion concurrency: one.
- Requested artifacts: TFLite plus the native PyTorch package only. The run
  used `-fdopt`; it did not request TorchScript, Dynamo ONNX, or
  ExportedProgram.
- Every converter ran in its own POSIX process group. Linux `/proc` monitoring
  would stop and classify the complete group on any descendant `VmSwap`.
- TensorFlow was not requested by the direct conversion command.
- The repository `uv` environment used Python 3.12.12 and Torch 2.11.0+cpu.
  `PYTHONNOUSERSITE=1`, empty `PYTHONPATH` and `LD_LIBRARY_PATH`, and one
  OMP/MKL thread prevented contamination from the host Python 3.10 Torch.

The profile contains 420 Tier 0-4 records. Twenty-nine recorded timeouts and
twelve explicit user exclusions remain as excluded history, leaving 379
active models. All 379 files were present. The active tier distribution was
119/84/105/49/22.

The run command was equivalent to:

```text
env PYTHONNOUSERSITE=1 PYTHONPATH= LD_LIBRARY_PATH= \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir /home/b920405/git/onnx2tf \
  --output_dir /tmp/fb7_tier0_4_full_0bf1bab4_20260718 \
  --timeout_sec 600 \
  --native_pytorch_only \
  --regression_profile \
  /home/b920405/git/onnx2tf/docs/baselines/flatbuffer_direct_active_tier0_4.json
```

## Execution results

The runner returned one by design because strict inherited failures remain. It
completed normally and produced all 379 entries and summaries.

| Metric | Current | `fb-refactor6` | Change |
| --- | ---: | ---: | ---: |
| Models completed | 379 | 379 | 0 |
| Sum of model wall time | 7,689.329 s | 7,542.595 s | +1.945% |
| Median model wall time | 8.755 s | 8.746 s | +0.109% |
| Maximum model wall time | 201.599 s | 191.394 s | +5.332% |
| Models at or above 600 s | 0 | 0 | 0 |
| Models with nonzero SWAP | 0 | 0 | 0 |
| Converter nonzero exits | 2 | 2 | 0 |

Every tier remained below ten minutes per model. Each tier's total and median
single-run time also remained within +10% of the comparison run. These are
single full-corpus observations, not the three-run warm-median performance
gate from the broader refactor plan.

| Tier | Models | Total seconds | Median seconds | Maximum seconds | Slowest model | Total vs baseline | Median vs baseline |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 0 | 119 | 790.962 | 3.738 | 74.034 | `zero_dce_640_dele.onnx` | +1.678% | +3.489% |
| 1 | 84 | 1,079.357 | 7.399 | 97.091 | `anime-gan-v2.onnx` | +9.705% | +0.509% |
| 2 | 105 | 2,130.687 | 11.577 | 201.599 | `yolov3-12-int8.onnx` | +0.355% | +2.249% |
| 3 | 49 | 2,195.113 | 32.070 | 186.618 | `stock_frcnn_11.onnx` | +3.073% | +6.862% |
| 4 | 22 | 1,493.209 | 58.449 | 179.430 | `wd-v1-4-moat-tagger-v2.onnx` | -2.276% | -0.020% |

The combined classifications exactly match `fb-refactor6`:

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

Component-level states also match exactly:

| Component result | TFLite | Native PyTorch |
| --- | ---: | ---: |
| Accuracy pass | 347 | 138 |
| Accuracy fail | 20 | 46 |
| Missing report, excluding the two conversion errors | 10 | 193 |
| Conversion error before reports | 2 | 2 |

The raw null-report counts are 12 for TFLite and 195 for native PyTorch; each
includes the same two common conversion errors. This distinction prevents the
conversion errors from being counted twice in the mutually exclusive table.

## Exact regression comparison

The following checks were performed before considering any source correction:

- ordered model list: identical;
- model-list digest: identical;
- combined classification differences: zero;
- strict-pass differences: zero;
- converter exit-code differences: zero;
- TFLite accuracy-state differences: zero;
- native PyTorch accuracy-state differences: zero;
- normalized failure-signature differences: zero;
- new timeout, SWAP, missing-model, or skipped-model classifications: zero.

Failure signatures were compared after replacing only the random suffix in
temporary `.onnx2tf_onnxsim_<random>.onnx` filenames. All diagnostic content
then matched. Raw signature hashes for onnxsim failures are expected to differ
when their random temporary suffix differs and are not semantic evidence.

There is one exact numeric-value difference among otherwise comparable
component reports:

| Model | Component | `fb-refactor6` max abs | Current max abs | Disposition |
| --- | --- | ---: | ---: | --- |
| `randnlike4.onnx` | native PyTorch | 1.828334451 | 1.811047643 | nondeterministic random output; failure classification unchanged and value decreased |

`randnlike4.onnx` contains random output behavior. Its TFLite result remains an
exact pass with max abs zero, while native PyTorch remains the same known
failure class. The numeric change is therefore neither a branch regression nor
a justification for a deterministic converter rewrite.

The two DEIM models remain accepted as TFLite successes under the recorded
user-approved near-tied TopK-index policy. Their native PyTorch reports remain
missing, exactly as in the comparison run.

The two DPT-LeViT models retain their inherited Softmax/rank conversion errors.
All other TFLite numeric failures, missing reports, native PyTorch failures, and
native PyTorch missing reports remain the same known problem set. This run does
not reclassify those inherited limitations as successes.

## Pass-efficiency evidence

Both runs recorded pass metrics for 377 of 379 models and exactly 196,325 pass
events. `state_build_count` remains 22,321, `snapshot_count` remains 5,419, and
`fingerprint_count` remains zero. `preflight_operators_visited` decreased by 16,
from 24,535,664 to 24,535,648. This is consistent with the branch's indexed
cleanup work and provides no evidence of an added scan or state-build
regression.

## Temporary-storage cleanup

Only the uniquely named run directory and its generated contents were in
scope. A watcher read only atomically persisted entries. For each completed
entry it retained command logs, pass metrics, and JSON/CSV/Markdown/text
diagnostics while deleting the staged ONNX copy, TFLite binaries, generated
native package, generated schema copies, and other reproducible artifacts. It
never touched the active model or a run not yet present in `bulk_status.json`.

The watcher cleaned all 379 persisted entries and deleted 51,656,627,955 bytes
(48.109 GiB) of reproducible data. The condensed evidence and this report were
created before removal of the remaining uniquely named `/tmp` run directory.
After evidence validation, that directory and both temporary helper scripts
were removed. No temporary path owned by this run remains, and no repository
model artifact is added to Git.

## Verification and disposition

- bulk runner and corpus-manifest tests: **44 passed**;
- full current-branch conversion: **379/379 completed**, sequential;
- timeout/SWAP/ten-minute gate: **0/0/0 violations**;
- component and combined comparison: **0 differences**;
- normalized failure signatures: **0 differences**;
- confirmed `fb-refactor7` regression: **0**;
- converter source changes made in response to this run: **none**.

Because no branch-specific regression was recorded, the requested
record-before-fix policy stops at the evidence checkpoint and no corrective
implementation or model rerun is required.
