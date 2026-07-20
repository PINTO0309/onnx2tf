# `fb-refactor8` Tier 0–4 TFLite and native PyTorch regression audit

## Result

No regression attributable to `fb-refactor8` was found in the complete eligible
Tier 0–4 corpus. All 379 models completed in the fixed order with one inference
process at a time. No model timed out, reached ten minutes, or caused SWAP.

The current run and the authoritative `fb-refactor7` result have identical:

- model order and corpus digest;
- combined classifications and strict-pass states;
- converter exit codes;
- TFLite and native PyTorch report states;
- TFLite and native PyTorch component counts;
- normalized failure signatures; and
- pass-event and graph-scan metrics.

No converter fix was made in response to this audit because there was no new
failure to fix. Existing failures remain visible and are not reclassified as
successes, except for the separately documented user-approved DEIM policy.

The complete condensed evidence is
`docs/baselines/flatbuffer_direct_tier0_4_tflite_native_pytorch_fb8_b35c36a0.json`.
It contains every model result, timing, component metric, SWAP value, pass
metrics, comparison result, profile identity, and source-state digest without
retaining generated models or packages.

## Fixed scope

- Branch: `fb-refactor8`
- Commit: `b35c36a0860b08b801a26bc3bd0cafd648355ac8`
- Root ONNX files: 456
- Root loadable Tier 0–4 models: 420
- Managed timeout exclusions: 29
- Managed explicit user exclusions: 12
- Eligible and executed models: 379
- Per-model timeout: 600 seconds
- Inference concurrency: 1
- Requested artifacts: TFLite plus the native PyTorch package only
- TensorFlow: not requested by the direct conversion path
- Environment: isolated `uv run --no-sync`, Python 3.12.12, PyTorch 2.11.0+cpu

The managed profile exactly matched all current root-level loadable models with
1–1,999 ONNX nodes. There were no profile-only or newly discovered Tier 0–4
model names. `vit_h_encoder.onnx` remains the one known unreadable ONNX file and
is outside the loadable Tier corpus.

The 379 active models were distributed as follows:

| Tier | ONNX nodes | Models |
| ---: | ---: | ---: |
| 0 | 1–49 | 119 |
| 1 | 50–199 | 84 |
| 2 | 200–499 | 105 |
| 3 | 500–999 | 49 |
| 4 | 1,000–1,999 | 22 |

All previously requested exclusions remained excluded. In particular, the run
did not reintroduce known timeout, explicit exclusion, or prior SWAP-risk
models. No new SWAP exclusion was necessary.

## Command

```bash
env PYTHONNOUSERSITE=1 PYTHONPATH= LD_LIBRARY_PATH= \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python -m onnx2tf.utils.flatbuffer_direct_bulk_runner \
  --root_dir /home/b920405/git/onnx2tf \
  --output_dir /tmp/fb8_tier0_4_full_b35c36a0_20260720 \
  --timeout_sec 600 \
  --native_pytorch_only \
  --regression_profile \
  /home/b920405/git/onnx2tf/docs/baselines/flatbuffer_direct_active_tier0_4.json
```

`--native_pytorch_only` generated and checked the direct TFLite artifact and
the native PyTorch package in the same isolated converter invocation. It did
not request TorchScript, Dynamo ONNX, or ExportedProgram artifacts. The runner
started only one converter subprocess group at a time and monitored the entire
`/proc` descendant tree for `VmSwap`.

## Execution results

| Measurement | `fb-refactor7` | `fb-refactor8` | Difference |
| --- | ---: | ---: | ---: |
| Eligible models | 379 | 379 | 0 |
| Completed models | 379 | 379 | 0 |
| Timeouts | 0 | 0 | 0 |
| SWAP detections | 0 | 0 | 0 |
| Models at or above 600 seconds | 0 | 0 | 0 |
| Converter nonzero exits | 2 | 2 | 0 |
| Total model time | 7,689.33 s | 7,376.09 s | -313.24 s (-4.07%) |
| Median model time | 8.76 s | 8.52 s | -2.71% |
| Maximum model time | 201.60 s | 208.95 s | +3.64% |

Every model remained below ten minutes. The slowest model was
`yolov3-12-int8.onnx` at 208.95 seconds. Its classification remained the same
known `both_fail`, and its time remained far below the 600-second eligibility
boundary.

Every tier's total and median single-run time improved or remained effectively
flat relative to `fb-refactor7`. These are single full-corpus observations, not
the three-run warm-median performance gate from the broader refactor plan.

| Tier | Models | Current total | Current median | Current maximum | Total ratio | Median ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 119 | 760.09 s | 3.74 s | 55.22 s | 0.9610 | 0.9992 |
| 1 | 84 | 999.71 s | 7.10 s | 82.31 s | 0.9262 | 0.9595 |
| 2 | 105 | 2,125.24 s | 10.98 s | 208.95 s | 0.9974 | 0.9484 |
| 3 | 49 | 2,053.34 s | 29.59 s | 190.23 s | 0.9354 | 0.9226 |
| 4 | 22 | 1,437.71 s | 58.14 s | 178.33 s | 0.9628 | 0.9948 |

One model, `rf-detr-seg-nano.onnx`, had a noisier individual single-run time
(70.43 seconds versus 52.97 seconds). It stayed below ten minutes, retained the
same result, and did not cause its tier total or median to regress. Therefore
this observation is recorded as single-run variance rather than a confirmed
performance regression; no speculative code change was applied.

## Classification and component comparison

The combined classification counts were exactly equal:

| Combined classification | `fb-refactor7` | `fb-refactor8` |
| --- | ---: | ---: |
| `pass` | 137 | 137 |
| `conversion_error` | 2 | 2 |
| `pytorch_fail` | 31 | 31 |
| `both_fail` | 12 | 12 |
| `missing_tflite_report` | 4 | 4 |
| `missing_pytorch_report` | 187 | 187 |
| `missing_both_reports` | 6 | 6 |
| `timeout` | 0 | 0 |

The raw component results were also exactly equal:

| Component result | TFLite | Native PyTorch |
| --- | ---: | ---: |
| Pass | 347 | 138 |
| Fail | 20 | 46 |
| Missing report | 10 | 193 |
| Converter error before report | 2 | 2 |

The strict runner reports 242 inherited non-pass entries because it deliberately
preserves missing reports, accuracy failures, and conversion errors. The audit
compared each of those entries with the previous branch rather than treating
the raw strict-fail count as 242 new regressions.

## Accuracy and known-result disposition

There were zero differences in component state and zero normalized failure
signature differences. Forty-nine raw signature hashes changed only because
`onnxsim` embeds a new random temporary filename in each invocation. After
normalizing that volatile suffix, their error text was identical; the same
hash-only behavior is present between earlier branch runs.

There was one numeric difference:

| Model | Component | `fb-refactor7` max abs | Current max abs | Disposition |
| --- | --- | ---: | ---: | --- |
| `randnlike4.onnx` | native PyTorch | 1.811047643 | 4.534230769 | nondeterministic random output; failure classification unchanged |

`randnlike4.onnx` contains random output behavior. Its TFLite result remained
an exact pass with max abs zero, while native PyTorch retained the same known
failure class. The numeric change is not evidence of a deterministic converter
regression and is not a basis for a converter rewrite.

The two DEIM models remain accepted as TFLite successes under the recorded
user-approved near-tied TopK-index policy. Their raw native PyTorch reports
remain missing exactly as in the comparison run; the raw evidence is retained
without rewriting its classification.

The two DPT-LeViT models retain their inherited Softmax/rank conversion errors.
No new converter error appeared and no existing successful model became a
converter error.

## Pass-efficiency evidence

Both runs recorded pass metrics for 377 of 379 models and exactly 196,325 pass
events. The aggregate values were identical:

| Metric | `fb-refactor7` | `fb-refactor8` |
| --- | ---: | ---: |
| `preflight_operators_visited` | 24,535,648 | 24,535,648 |
| `state_build_count` | 22,321 | 22,321 |
| `snapshot_count` | 5,419 | 5,419 |
| `fingerprint_count` | 0 | 0 |

This provides no evidence that the branch added graph scans, state rebuilds,
snapshots, or pass events.

## Temporary-storage cleanup

Only the uniquely named run directory and its generated contents were in
scope. Cleanup examined only entries already atomically persisted in
`bulk_status.json`. For each completed entry it retained command logs, pass
metrics, and JSON/CSV/Markdown/text diagnostics while deleting the staged ONNX
copy, TFLite binaries, generated native package, generated schema copies, and
other reproducible artifacts. It never touched an active model or another run.

Cleanup covered all 379 persisted entries and deleted 51,216,397,607 bytes
(47.70 GiB) of reproducible data. After the condensed evidence and report were
validated, the remaining uniquely named run directory and temporary helper
files were removed. No root ONNX model or generated model artifact is added to
Git.

## Verification and disposition

- bulk runner and corpus-manifest tests: **44 passed**;
- full current-branch conversion: **379/379 completed**, sequential;
- timeout/SWAP/ten-minute violations: **0/0/0**;
- component and combined comparison differences: **0**;
- normalized failure-signature differences: **0**;
- confirmed `fb-refactor8` regressions: **0**;
- converter source changes made in response to this run: **none**.

The branch passes this Tier 0–4 TFLite and native PyTorch regression gate.
