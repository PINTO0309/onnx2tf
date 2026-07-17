# `fb-refactor7` improvement summary

## Purpose and compatibility boundary

`fb-refactor7` improves the internal efficiency, ownership, and observability
of the TensorFlow-free `flatbuffer_direct` pipeline. It deliberately preserves
the public CLI and Python API, default backend, output names, ModelIR and
artifact semantics, pass order, optional TensorFlow boundary, dependency set,
and sequential-inference policy.

The branch follows a characterize-first workflow. Every production change is
preceded by a contract that freezes the existing behavior, including
zero-mutation and cleanup-only paths. The implementation is then admitted only
after the focused contract and a broader sequential regression gate pass.

## Diagnostic bookkeeping efficiency

- ModelIR pass diagnostic numbering no longer repeatedly scans the complete
  and growing diagnostic history for every emitted event.
- A single scan initializes event, group, and per-pass invocation counters for
  arbitrary caller-owned diagnostic lists.
- `ConversionSession`-owned diagnostic ledgers retain this numbering state
  across pass groups, avoiding one full history scan per group while preserving
  the exact external diagnostic schema and numbering rules.
- Ordinary lists remain supported through the safe one-scan fallback, so this
  optimization does not leak a new internal type through the public boundary.

## Stable-reconciliation and convergence fast paths

Several late and terminal layout paths previously reran broad shape/layout
reconciliation even when the immediately preceding owner reported no graph
mutation. The branch now uses explicit mutation results to skip those stable
rescans. Covered paths include:

- mixed singleton and final Reshape reconciliation;
- final PReLU and SINet reconciliation;
- terminal SE/Gather reconciliation;
- shared late reconciliation and late binary repair;
- binary-layout convergence rounds;
- final shape reconciliation and both final convergence scans;
- post-fusion and placeholder-MatMul repair follow-up scans.

The fast paths are conservative. Cleanup-only owners include clamped net
tensor-count deltas, and reconciliation is skipped only when all declared
mutation evidence is zero. Pass order and mutation-positive behavior remain
unchanged.

## Explicit mutation evidence and result propagation

Late and terminal orchestration now returns or stages ordered, fixed-schema
mutation evidence instead of silently discarding child results. This makes
graph-stability decisions auditable and prevents future optimizations from
using incomplete proxy counters.

The covered boundaries include:

- recovery orchestration and singleton consecutive-Reshape results;
- late layout, hard-activation, SPP, QKV, and binary-layout clusters;
- terminal Hardswish/SE, split/Conv/Concat bridges, shape extraction, and
  Expand/Squeeze/Reshape recovery;
- terminal and pre-terminal slice/pad/concat and affine recovery;
- channel-slice/pad-Mul and composite pre-ADD owners;
- indexed affine post-ADD and InstanceNorm post-bias,
  residual/Mul/Concat, and dual-statistics owners;
- the separate pre-terminal and absolute-final post-bias observation points.

Raw counters are used only when the owner prunes after a positive rewrite. An
owner that can prune with zero rewrites additionally reports the clamped net
tensor reduction. Non-mutating iteration counters are excluded from mutation
summaries. These rules make every stability guard conservative without adding
ModelIR copies or additional graph passes.

## Smaller internal owners

- Typed ONNX `Constant` lowering is isolated in its op-family module while
  preserving supported attribute forms, dtypes, provenance, and errors.
- Shape-readiness decisions are extracted into a demand-driven helper so the
  lowerer does not duplicate readiness logic.
- Late binary recovery has a dedicated runner with an explicit result schema.
- Late hard-activation, SPP, QKV, channel-slice/pad-Mul, affine recovery, and
  related orchestration modules expose their ordered results directly.
- `NodeView` value-list annotations were clarified for static type checking;
  runtime behavior is unchanged.

These extractions reduce implicit coupling in the main lowerer and make a pass
owner, its mutation contract, and its regression tests discoverable together.

## TensorFlow-free and artifact guarantees

No dependency was added. Normal `flatbuffer_direct` lowering and its focused
tests continue to run with `PYTHONNOUSERSITE=1 uv run --no-sync`. Optional
TensorFlow import-blocking coverage remains in every broad gate. The changes do
not add an exporter invocation, quantization path, split path, model-inference
worker, or TensorFlow import.

## Validation

The latest sequential gate covers the changed indexed owners, late/terminal
orchestrators, absolute-final normalization boundary, shared ModelIR pass
context, core contracts, pass efficiency, architecture constraints, and
optional TensorFlow import boundary. Result: `1200 passed`.

Focused validation for the final absolute-final post-bias unit produced `9
passed`. Focused Ruff, Python bytecode compilation, and `git diff --check` also
pass. These results are contract and orchestration tests; they do not claim a
new full model-corpus run for this final observation-only unit.

## Remaining work

The broader `flatbuffer_direct` refactor remains active. The next characterized
unit should inspect the absolute-final affine post-ADD occurrence immediately
before the newly staged post-bias result. Any new mutation evidence must remain
distinct from the existing pre-terminal observation point and must preserve
the current pass order, TensorFlow-free boundary, dependency set, and
sequential validation policy.
