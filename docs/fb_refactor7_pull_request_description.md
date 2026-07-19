# Make `flatbuffer_direct` pass execution more explicit, indexed, and regression-safe

## Summary

This branch continues the staged refactoring of the TensorFlow-free
`flatbuffer_direct` backend. It makes pass ownership, mutation evidence, graph
index reuse, reconciliation decisions, and terminal validation substantially
more explicit while preserving the existing conversion policy and public
interfaces.

The central goal of this checkpoint is not to introduce a new converter or to
change the order of the existing compatibility rules. It is to make the
current behavior safer to evolve:

- pass results are retained instead of being silently discarded;
- graph-stability decisions use complete, typed mutation evidence;
- stable paths avoid redundant full-graph reconciliation scans;
- repeated adjacent cleanup passes share one live `ModelIRGraphIndex`;
- terminal layout validation observes the graph after the applicable repairs;
- focused owners and contracts isolate changes from unrelated model families;
- every production change follows a characterize-first regression workflow.

The release metadata is updated consistently from `2.6.5` to `2.6.6` in the
package, project metadata, lockfile, and documented container tags.

## Motivation

The direct FlatBuffer path has accumulated many interacting layout, shape,
quantization, recovery, and compatibility rules. Historically, a number of
those rules were invoked through raw calls in the central lowerer, returned
useful information that callers discarded, rebuilt producer/consumer state at
nearby boundaries, or triggered reconciliation even when the preceding owner
made no graph mutation.

Those patterns make apparently local changes difficult to reason about. A
counter can be incomplete because cleanup happened outside the counter, a
zero result can be mistaken for graph stability, or two adjacent passes can
independently rescan the same graph. This branch establishes explicit contracts
for these boundaries before using them for conservative scan elimination or
index sharing.

## Detailed changes

### Efficient diagnostic bookkeeping

`run_model_ir_pass_group()` no longer scans the complete and growing
diagnostic history for every emitted event. One scan initializes the global
event count, maximum group sequence, and per-pass invocation counts. Those
counters are then updated as events are appended.

`ConversionSession` retains this state across pass groups. Ordinary
caller-owned lists remain supported through a safe one-scan fallback. The
external diagnostic fields, numbering semantics, list identity, skip/cycle
behavior, and invariant-failure reporting are unchanged.

### Explicit pass results and mutation evidence

Late, very-late, terminal, recovery, attention, quantized, binary-layout,
Conv1D, and safety-fallback boundaries now return or stage fixed-schema result
dictionaries instead of dropping child results. Multi-pass orchestration
preserves child order and shared pass-state scopes.

Observation-only results remain intentionally unconsumed when their counters
are not yet safe control-flow inputs. Where a stability decision is made, the
owner contract accounts for every relevant mutation, including cleanup-only
tensor pruning. Non-mutating iteration counters are excluded from mutation
summaries.

This gives later work auditable evidence without adding ModelIR copies,
fingerprints, or extra graph walks.

### Complete static-shape reconciliation accounting

Static-shape reconciliation can now opt in to a complete mutation count. In
addition to ordinary tensor-shape updates, it records operator-option changes,
constant shape-parameter writes, and direct tensor metadata updates performed
during the existing fixed-point walk.

The default result schema remains compatible. Only characterized call sites
request the additional evidence, and their original guards, pruning behavior,
layout synchronization, and pass order are preserved.

This complete evidence is retained across primary, fallback, very-late, and
post-split reconciliation boundaries, including Conv-input, mixed-Concat,
Concat-axis, high-rank BatchMatMul, Pad, placeholder-MatMul, singleton Reshape,
SE/FC/Gather, PReLU, and SiNet paths.

### Conservative stable-path fast paths

Broad reconciliation is skipped only when all declared mutation evidence from
the immediately preceding owner is zero. This applies to selected singleton
Reshape, PReLU, SiNet, SE/Gather, late binary, shared late, final-shape,
post-fusion, placeholder-MatMul, and bounded convergence paths.

Mutation-positive behavior is unchanged. Cleanup-only owners use clamped net
tensor-count deltas where required, so a rewrite counter of zero is never
silently treated as stability when pruning may still have changed the graph.

Binary-layout convergence can also stop after a stable round instead of
running the remaining bounded rounds. The maximum round count and all
mutation-positive behavior remain unchanged.

### Shared graph indexes for adjacent cleanup work

`run_indexed_prune_reconcile_cleanup()` shares one `ModelIRGraphIndex` across
dead-operator pruning and static-shape reconciliation at three repeated phase
boundaries. It performs exactly those two existing operations and does not add
dynamic-reshape resolution, retries, or a new convergence branch.

`run_indexed_binary_layout_adapter_cleanup()` similarly shares one index
between the exact rank-four binary adapter and the singleton-broadcast adapter
at four repeated boundaries. The exact adapter still runs first. Both owners
now use indexed candidate lookup, operator insertion, and input replacement,
while preserving names, tensor metadata, quantization cloning, pruning, result
schemas, and existing guard inputs.

Fallback unbound-input repair also avoids one redundant reconciliation because
its indexed compatibility wrapper already performs the required positive-path
reconciliation with the live graph index.

### Terminal validation after repair

Primary and fallback terminal layout validation now observe the graph after
the applicable terminal mutations and final topological ordering. When the
terminal graph is valid, validation clears only stale validation errors
inherited from recursive lowering. The diagnostic schema and actual error
conditions are unchanged.

The branch retains the complete terminal mutation dictionaries used to make
this boundary reviewable, including bounded binary convergence, high-rank
binary coalescing, boundary-signature realignment, high-rank BatchMatMul, Pad,
Conv-input, mixed-Concat, Concat-axis, stale-binary, and SiNet repairs.

### Smaller, testable internal owners

Several responsibilities were moved behind focused internal interfaces:

- typed ONNX `Constant` lowering lives in its op-family module;
- demand-driven shape readiness lives in a dedicated core helper;
- late binary recovery has a dedicated runner and stable result schema;
- hard-activation, SPP, QKV, channel-slice/pad-Mul, affine, quantized, and
  attention orchestration exposes ordered results directly;
- `NodeView` value-list annotations and ONNX graph-input typing are clearer to
  static analysis without changing runtime behavior.

Private compatibility wrappers remain where existing structural callers rely
on them. The central lowerer keeps the same production call positions and
ordering unless a characterized shared runner replaces an exactly adjacent
pair.

### Characterize-first safety workflow

Each behavior-changing implementation unit was preceded by a focused contract
that froze the relevant result schema, no-op path, mutation path, order,
arguments, state-scope ownership, and surrounding phase boundary. The strict
expected failure was removed only after the corresponding implementation and
the broader sequential gate passed.

This branch adds extensive focused coverage for result propagation,
cleanup-only mutations, deterministic ordering, idempotence, shared-index
construction, layout recovery, attention and quantized paths, fallback
relowering, terminal validation, and architectural ownership.

## Compatibility and dependency boundaries

- The public CLI and Python API are unchanged.
- `flatbuffer_direct` remains the default backend.
- Existing artifact names, report formats, return behavior, and pass order are
  preserved.
- Normal direct TFLite conversion and `-cotof` do not import or execute
  TensorFlow.
- Optional SavedModel/H5/Keras/TFv1 behavior remains behind the existing
  TensorFlow-optional boundary.
- No dependency was added.
- No new exporter, quantization path, split path, inference worker, or
  multiprocessing behavior was introduced.
- Corpus inference and conversion validation remained strictly sequential.

## Validation

### Focused and structural regression gates

At the final implementation checkpoint before the corpus run:

- dedicated indexed binary-adapter contract: **3 passed**;
- focused affected-owner suite: **424 passed**;
- all 116 branch-changed test files: **1707 passed**;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  **passed**.

The broad gate includes the changed indexed owners, late and terminal
orchestration, static-shape reconciliation, shared ModelIR pass context, core
contracts, pass-efficiency checks, architecture constraints, and the optional
TensorFlow import boundary.

The corpus runner and manifest tests were also repeated after the full run:
**44 passed**.

### Tier 0-4 TFLite and native PyTorch corpus

All 379 active managed Tier 0-4 models were converted strictly one at a time at
converter checkpoint `0bf1bab4`. The scope excludes only previously recorded
timeouts and explicit user exclusions.

- completed models: **379 / 379**;
- per-model timeout ceiling: **600 seconds**;
- models at or above the ceiling: **0**;
- converter process trees with nonzero SWAP: **0**;
- TFLite/native-PyTorch component-state differences from `fb-refactor6`:
  **0**;
- combined-classification differences: **0**;
- exit-state and strict-pass differences: **0**;
- normalized failure-signature differences: **0**;
- confirmed branch-specific regressions: **0**.

The combined classifications exactly match the authoritative `fb-refactor6`
comparison:

| Classification | Models |
| --- | ---: |
| `pass` | 137 |
| `missing_pytorch_report` | 187 |
| `pytorch_fail` | 31 |
| `both_fail` | 12 |
| `missing_both_reports` | 6 |
| `missing_tflite_report` | 4 |
| `conversion_error` | 2 |
| **Total** | **379** |

The run requested TFLite and the native PyTorch package only. It did not
request TorchScript, Dynamo ONNX, ExportedProgram, or TensorFlow artifacts.

### Timing and pass-efficiency observations

This full-corpus run is a single regression observation, not the broader
three-run warm-median performance gate.

- total wall time: **7,689.329 s** (`+1.945%`);
- median model wall time: **8.755 s** (`+0.109%`);
- maximum model wall time: **201.599 s** (`+5.332%`);
- every tier total and median remained within `+10%` of the comparison run.

Both runs recorded exactly 196,325 pass events. `state_build_count`,
`snapshot_count`, and `fingerprint_count` were unchanged, while
`preflight_operators_visited` decreased by 16. This provides no evidence of an
added scan or state-build regression.

## Known inherited outcomes

This PR does not claim that every corpus model has numerical parity. It claims
that the branch introduces no new result or failure-signature regression
relative to the authoritative previous checkpoint.

- Two DPT-LeViT models retain their existing Softmax/rank conversion errors.
- Existing TFLite numerical failures, missing reports, native PyTorch failures,
  and native PyTorch missing reports keep the same classifications and
  normalized signatures.
- The two DEIM models remain accepted TFLite successes under the recorded
  near-tied TopK-index policy.
- `randnlike4.onnx` retains the same native-PyTorch known-failure class. Its
  nondeterministic max-absolute value decreased from `1.828334451` to
  `1.811047643`; this is not treated as a converter regression.

## Evidence and review guide

- Cumulative design and validation notes:
  `docs/fb_refactor7_improvements.md`
- Architecture and pass-boundary contracts:
  `docs/flatbuffer_direct_architecture.md`
- Full Tier 0-4 regression report:
  `docs/flatbuffer_direct_tier0_4_full_regression_2026-07-18.md`
- Condensed 379-model evidence:
  `docs/baselines/flatbuffer_direct_tier0_4_tflite_native_pytorch_fb7_0bf1bab4.json`
- Detailed continuation history:
  `docs/flatbuffer_direct_handoff_fb_refactor7.md`

The generated model artifacts were intentionally not committed. After the
evidence was persisted and validated, 48.109 GiB of reproducible temporary
artifacts and the remaining run-owned `/tmp` files were removed.

## Remaining scope

This is a coherent refactoring checkpoint, not completion of the entire
long-term `flatbuffer_direct` redesign. Later work may continue the fixed
`ConversionRequest`/`ConversionSession` contract, phase/pass manager,
centralized layout planning, unified lowering registry, exporter separation,
quantization/split refresh, and PyTorch-family restructuring. Those items are
not silently mixed into this PR.
