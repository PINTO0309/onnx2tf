# `fb-refactor7` handoff

## Current state

- Branch: `fb-refactor7`.
- Starting checkpoint: `d3369d9a`, the merge of the completed
  `fb-refactor6` work into `main`.
- Pull requests are outside the continuation workflow. Continue with coherent
  commits and pushes to `origin/fb-refactor7` only.
- Inference remains strictly sequential. No model-inference ProcessPool or
  parallel worker is permitted.

## Diagnostic-numbering characterization checkpoint

The first `fb-refactor7` unit inventories a non-context core efficiency issue
without changing production code. `run_model_ir_pass_group()` calculates the
next diagnostic group with one full history scan, but `_record_event()` then
rescans the complete and growing history for every result to derive global
`sequence` and per-pass `invocation`. A group with `P` results therefore uses
`P + 1` diagnostic-list iterations and accumulates quadratic bookkeeping cost
across a conversion with many pass events.

The new semantic fixture freezes all externally visible numbering rules:

- non-`model_ir_pass` records are preserved and excluded from numbering;
- the next `group_sequence` is one greater than the maximum existing group;
- `sequence` counts all existing ModelIR-pass records, independent of stored
  legacy sequence values;
- `invocation` counts only records with the same pass ID;
- every event produced by one group receives the same group number.

A strict expected-failure efficiency fixture supplies a list subclass that
counts history iterations. The current implementation performs four scans for
one three-result group; the required implementation performs exactly one. The
focused characterization result is `2 passed, 1 xfailed`.

No production source, public API, artifact, pass order, graph index, layout,
diagnostic schema, dependency, TensorFlow boundary, corpus policy, or model
conversion changed in this checkpoint.

## Next action

In `run_model_ir_pass_group()`, scan existing diagnostics once to initialize:

1. the existing ModelIR-pass event count;
2. the maximum group sequence;
3. per-pass invocation counts.

Update those counters locally as events are appended. Preserve exact behavior
for ordinary, skipped, cycle-stopped, and invariant-failure events. Remove the
strict xfail only after the efficiency fixture passes, then run the complete
core, pass-efficiency, architecture, and TensorFlow-import-blocked gates. Keep
real-model conversion minimal because this unit cannot affect ModelIR or
artifacts. Commit and push only; do not create or update a pull request.

## Diagnostic-numbering implementation checkpoint

`run_model_ir_pass_group()` now scans an existing diagnostics list exactly
once. That scan initializes the total ModelIR-pass event count, maximum group
sequence, and a per-pass invocation-count map. `_record_event()` increments
those local counters while appending, rather than iterating over the complete
and growing history again for every pass result.

The implementation preserves the characterized behavior for unrelated
diagnostics, missing or negative legacy group values, repeated pass IDs,
multiple results in one group, skipped passes, cycles, and invariant failures.
It does not change pass execution, ModelIR, LayoutState, GraphIndex, result
details, diagnostic fields, summary schema, or the identity of the caller's
list.

Sequential validation completed as follows:

- focused numbering and pass-group contracts: `7 passed`;
- complete core, pass-efficiency, architecture, and TensorFlow-import-blocked
  gate: `324 passed in 27.74s`;
- the scan-count fixture changed from one strict xfail to pass and observes one
  history iteration for a three-result group;
- focused Ruff, Python compilation, and whitespace checks: passed.

No real-model conversion was run because this unit cannot change ModelIR or an
artifact. No public API, CLI behavior, dependency, pass order, corpus policy,
inference concurrency, or TensorFlow boundary changed.

At resume, return to the remaining direct lowerer-to-pass and core contract
inventory. Select another non-context unit only when it has an explicit
ownership or measurable scan/allocation cost, and characterize its behavior
before production changes. Continue with coherent commits and pushes only; do
not create or update a pull request.

## Cross-group diagnostic-ledger characterization checkpoint

The first implementation removes repeated history scans inside one pass group,
but an ordinary diagnostics list must still be scanned once at the start of
every group. That safe fallback is required for arbitrary caller-owned lists,
which may have changed between calls. Production is different:
`ConversionSession` owns one diagnostics object for the complete conversion and
only appends through the pass runner or `record_diagnostic()`.

A new strict expected-failure contract requires the Session-owned object to
retain its ModelIR-pass event count, maximum group, and per-pass invocation
counts across group calls. The fixture deliberately performs slice assignment
first, proving that arbitrary list mutation invalidates the ledger and causes
exactly one lazy rebuild. Two subsequent groups must reuse the rebuilt state
while preserving sequences `2, 3`, invocations `2, 3`, and groups `5, 6`.

No production source changed. At implementation, introduce a private
list-compatible diagnostic ledger owned only by `ConversionSession`. Normal
append/extend/insert operations may update valid counters incrementally;
replacement or removal operations must invalidate them. Arbitrary external
lists passed to `run_model_ir_pass_group()` must retain the existing one-scan
fallback. Do not expose the internal type through the public core API.

## Cross-group diagnostic-ledger implementation checkpoint

`ConversionSession.diagnostics` now defaults to the private list-compatible
`ModelIRPassDiagnostics` type. It maintains the ModelIR-pass event count,
maximum group sequence, and pass-ID invocation counts incrementally for append-
only production use. `run_model_ir_pass_group()` reads a copied numbering
snapshot from this ledger in constant time; arbitrary external lists continue
to use the characterized one-scan fallback.

List compatibility is explicit. Append, extend, and insert update valid state.
Slice/item replacement, deletion, multiplication, pop, and remove invalidate
state without changing the list operation; the next group lazily rebuilds it
once. Clear resets an empty valid ledger. Malformed externally appended entries
invalidate rather than raising early, preserving error timing until the next
numbering read. The type is deliberately absent from `core.__all__`.

Sequential validation completed as follows:

- focused diagnostic/pass-group contracts, including zero rebuilds for the
  ordinary Session append path and one rebuild after slice replacement: pass;
- complete core, pass-efficiency, architecture, and TensorFlow-import-blocked
  gate: `328 passed in 26.75s`;
- focused Ruff, Python compilation, and whitespace checks: passed.

No model conversion ran because pass execution and ModelIR are unaffected. No
public API, CLI behavior, artifact, dependency, pass ID/order, GraphIndex,
LayoutState, corpus policy, inference concurrency, or TensorFlow boundary
changed.

At resume, move beyond diagnostic bookkeeping and inventory the remaining
direct lowerer-to-pass/core boundaries again. Select a non-context unit with a
measurable graph scan, state allocation, or duplicated ownership contract;
characterize it before changing production. Commit and push coherent units
only, and do not create or update a pull request.
