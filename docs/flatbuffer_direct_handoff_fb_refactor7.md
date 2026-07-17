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
