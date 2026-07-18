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

## Absolute-final SINet reconciliation characterization checkpoint

The next non-context graph-scan cost is the terminal sequence of six SINet
owners:

- late residual Add/Mul/Add/PReLU;
- deep-skip pre-Add/Concat/PReLU fan-out;
- deep-skip dual-Resize affine recovery;
- shared-post PReLU fan-out;
- deep-skip Concat/Resize affine tail;
- final Concat/Resize affine bridge.

Each owner is already bounded and transactional, returns one exact rewrite
counter, and has positive/no-op/idempotence coverage. The lowerer nevertheless
runs `_reconcile_static_tensor_shapes()` unconditionally after every owner.
Thus the zero-owner path performs six unnecessary full reconciliation scans.

A strict expected-failure architecture fixture freezes the exact owner order,
counter key, immediate guard, and single reconciliation body for all six
owners. Production is unchanged in this checkpoint. At implementation, assign
each existing result and run its existing immediate reconciliation only when
the corresponding counter is greater than zero. Do not merge, reorder, skip,
or generalize any SINet pass, and do not alter LayoutState handling.

Characterization validation is `514 passed` across the six complete owner
suites, plus one strict architecture xfail for the currently unconditional
reconciliations. Focused Ruff and whitespace checks pass.

## Absolute-final SINet reconciliation implementation checkpoint

All six owners remain in their original terminal order and execute with the
same main ModelIR and Session LayoutState. Each return value is now assigned,
and only a positive value under its established mutation counter triggers the
existing immediately following `_reconcile_static_tensor_shapes()` call. No
owner was merged, skipped, generalized, or moved.

The architecture xfail is green. A synthetic Add lowerer fixture replaces the
six owners with controlled counters: the all-zero case establishes the base
reconciliation count, and each of the six one-changed cases adds exactly one
call. The fixture confirms both zero-owner scan elimination and positive-owner
maintenance wiring without requiring real-model inference.

Sequential validation completed as follows:

- six complete SINet owner suites, core, pass efficiency, architecture, and
  TensorFlow-import-blocked optional boundary: `844 passed in 29.04s`;
- focused runtime counter wiring: passed for the all-zero baseline and all six
  individual changed counters;
- focused Ruff, Python compilation, and whitespace checks: passed, excluding
  only the lowerer's two pre-existing F401 findings.

No public API, CLI behavior, artifact, dependency, pass call/order, LayoutState,
GraphIndex, corpus exclusion, operation-count tier, inference concurrency, or
TensorFlow boundary changed. No real-model conversion ran because the 514
positive/no-op owner fixtures plus lowerer wiring test cover this maintenance-
scan-only change.

At resume, inventory the other unconditional static-shape reconciliations near
this terminal block. Do not assume their runners expose complete mutation
counters: characterize each return contract and positive/no-op coverage before
guarding another scan. Commit and push only; do not create or update a pull
request.

## NodeView value-list typing checkpoint

`NodeView.inputs` and `NodeView.outputs` now explicitly carry
`list[_ValueView]` annotations. This resolves the editor-side protobuf type
misinference that reported `.name` as unavailable at the demand-driven input
rank recovery in `lower_from_onnx2tf.py`. The produced lists and their values
are unchanged at runtime.

Sequential validation covered the complete NodeView, flatbuffer-direct core,
and TensorFlow-import-blocked optional-boundary suites: `52 passed`. Focused
Ruff, Python compilation, and whitespace checks also pass. Continue with the
remaining unconditional static-shape reconciliation inventory; commit and push
only, with no pull request.

## Mixed-singleton Concat reconciliation characterization checkpoint

The first safe candidate in the remaining terminal inventory is
`_repair_mixed_singleton_nchw_inputs_for_nhwc_concat()`. Its indexed owner
returns one exact mutation counter and performs no mutation when that counter is
zero. Existing tests cover positive rewrites, complete no-op preservation,
maintained GraphIndex/LayoutState, fan-out, name allocation, and the fast path
without a Concat. The immediately following shape reconciliation is still
unconditional in production.

A strict expected-failure architecture contract now requires that exact call to
be assigned and its immediate reconciliation to be guarded by
`repaired_mixed_singleton_nchw_inputs_for_nhwc_concat > 0`. This checkpoint does
not change production. Do not apply the same assumption to the nearby PReLU
owner, which intentionally prunes unused tensors even when its rewrite count is
zero, or to the SE/FC/Gather cluster, whose wrapper currently returns no
aggregate mutation result.

At implementation, change only this one call site, add lowerer-level zero/one
counter wiring coverage, and preserve its order and Session LayoutState. Run the
complete mixed-singleton owner, core, architecture, pass-efficiency, and
TensorFlow-import-blocked gates sequentially. Commit and push only; do not
create or update a pull request.

## Mixed-singleton Concat reconciliation implementation checkpoint

The terminal mixed-singleton Concat owner still runs exactly once in its
original position with the Session LayoutState. Its result is now assigned, and
the immediately following `_reconcile_static_tensor_shapes()` runs only when
`repaired_mixed_singleton_nchw_inputs_for_nhwc_concat` is positive. No matcher,
rewrite, GraphIndex update, layout update, or neighboring owner changed.

The strict architecture expectation is green. A synthetic lowerer fixture
proves that the positive counter adds exactly one reconciliation relative to
the zero-counter path. Sequential validation across the complete owner, core,
pass-efficiency, architecture, and TensorFlow-import-blocked optional-boundary
suites is `363 passed in 25.91s`. Focused owner coverage is `31 passed`.

At resume, the absolute-final consecutive-Reshape runner is the next possible
complete-counter candidate: it exposes three mutation counters and mutates only
when their sum is positive. Characterize the exact terminal call before
changing it. Do not guard the PReLU or SE/FC/Gather boundaries without first
making their mutation accounting complete. Commit and push only; do not create
or update a pull request.

## Consecutive-Reshape reconciliation characterization checkpoint

The absolute-final `run_consecutive_reshape_cleanup()` is now characterized as
the next complete-counter boundary. Its three results cover no-op removal,
consecutive-chain rewrites, and fan-out bypass rewrites. The fan-out path also
increments the aggregate rewrite counter, and cleanup/synchronization only run
after a positive mutation. A new no-candidate fixture freezes the exact
all-zero result, byte-for-byte-equivalent ModelIR representation, skipped
diagnostic, and no pass-state construction.

Production remains unchanged. A strict expected-failure architecture contract
requires the terminal runner result to be assigned and its immediately
following shape reconciliation to be guarded by all three exact counters. At
implementation, keep the runner in place with the same LayoutState and
diagnostics. Add zero-counter and one-counter-per-key lowerer wiring coverage,
then run the consecutive-Reshape, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites sequentially. Commit and push only; do not
create or update a pull request.

## Consecutive-Reshape reconciliation implementation checkpoint

The absolute-final `run_consecutive_reshape_cleanup()` remains in place with
the same Session LayoutState and diagnostic ledger. Its result is now assigned,
and the immediately following static-shape reconciliation runs only when the
sum of its three exact mutation counters is positive. No reshape matching,
rewriting, pruning, diagnostics, ordering, or neighboring pass changed.

The strict architecture expectation is green. A synthetic lowerer fixture
establishes the all-zero baseline and proves that each individual positive
counter adds exactly one reconciliation. Sequential validation across the
complete consecutive-Reshape, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `337 passed in 25.69s`; the focused runner
suite is `3 passed`.

At resume, leave the absolute-final PReLU reconciliation unchanged until its
zero-rewrite tensor-pruning side effect has explicit mutation accounting. The
SE/FC/Gather cluster likewise needs an aggregate result contract before its
reconciliation can be guarded. Inventory another bounded scan or allocation
with a complete owner contract, characterize it first, then commit and push
only. Do not create or update a pull request.

## Absolute-final PReLU reconciliation characterization checkpoint

The absolute-final PReLU owner has one rewrite counter but also intentionally
calls `_prune_unused_tensors()` for zero-match invocations. A new owner fixture
uses `max_rewrites=0` with one unused tensor and freezes the complete behavior:
the rewrite counter remains zero, the tensor count decreases by one, and the
LayoutState stays synchronized.

A strict expected-failure architecture contract therefore does not permit a
rewrite-only guard. It requires the terminal lowerer to record the tensor-table
size immediately before the owner and reconcile when either the exact rewrite
counter is positive or the tensor count decreased. This preserves the existing
zero-rewrite cleanup contract without paying for reconciliation when neither a
rewrite nor pruning occurred. Production is unchanged at this checkpoint.

At implementation, change only the absolute-final call site, retain the owner
call and Session LayoutState, and add lowerer wiring coverage for zero-change,
rewrite, and prune-only outcomes. Run the complete PReLU owner, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Absolute-final PReLU reconciliation implementation checkpoint

The absolute-final PReLU owner remains in its original position with the same
Session LayoutState and its intentional unconditional tensor prune. The lowerer
now records the tensor-table size immediately before the owner, assigns its
result, and runs the existing following reconciliation only after a positive
rewrite counter or a tensor-count decrease.

The strict architecture expectation is green. A synthetic lowerer fixture
separately proves unchanged, rewrite, and prune-only outcomes; both changed
outcomes add exactly one reconciliation over the unchanged baseline. Sequential
validation across the complete PReLU owner, core, pass-efficiency, architecture,
and TensorFlow-import-blocked suites is `365 passed in 26.66s`; focused owner
coverage is `29 passed`.

At resume, the remaining absolute-final SE/FC/Gather reconciliation still has
no aggregate mutation result. Characterize the two-runner orchestration return
contracts and decide whether aggregation can be introduced without changing
their public runner results or diagnostics. If not, leave it unconditional and
inventory another bounded scan/allocation. Commit and push only; do not create
or update a pull request.

## Recovery-result propagation characterization checkpoint

`run_recovery_invocations()` currently validates ordered pass IDs and invokes
each callback but discards its return value. Consequently,
`run_se_fc_gather_channel_fanout()` cannot expose the two existing result
dictionaries needed for a complete terminal reconciliation guard.

Two strict expected-failure contracts now require ordered result tuples from
the generic recovery utility and the SE/FC/Gather wrapper. They preserve the
same callback order, arguments, shared `ModelIRPassStateScope`, diagnostics, ID
drift failure before execution, and exception timing. Production is unchanged.

At implementation, return a tuple from the generic runner and forward a typed
two-dictionary tuple from the SE/FC/Gather wrapper. Existing orchestrators may
continue ignoring results. Do not yet change either main/fallback terminal
reconciliation; first validate all recovery-orchestration and SE/FC/Gather
contracts sequentially. Commit and push only; do not create or update a pull
request.

## Recovery-result propagation implementation checkpoint

`run_recovery_invocations()` now returns the callback values as an ordered
tuple after the same pass-ID validation. Existing orchestrators continue to
ignore it. `run_se_fc_gather_channel_fanout()` returns its two typed result
dictionaries, and the lowerer's private helper forwards them while preserving
the same context, Session diagnostics, and shared `ModelIRPassStateScope`.

Sequential validation across every
`test_flatbuffer_direct_*orchestration.py` file is `292 passed in 4.37s`.
The core, pass-efficiency, architecture, and TensorFlow-import-blocked gate is
`336 passed in 25.72s`. Focused return-order, ID-drift, SE/FC/Gather, and helper
contracts pass.

At resume, characterize the main and fallback terminal reconciliation guards as
one combined ownership boundary: SINet shuffle result, both ordered cluster
results, and tensor-count reduction for zero-rewrite pruning. Do not change the
two call sites until both boundary forms have strict structural and lowerer
wiring coverage. Commit and push only; do not create or update a pull request.

## Terminal SE/FC/Gather reconciliation characterization checkpoint

The main ModelIR and fallback ModelIR have the same terminal boundary: the
SINet shuffle residual owner, the two-runner SE/FC/Gather cluster, then an
unconditional static-shape reconciliation. The SINet owner and both runner
results expose exact rewrite counters, while the two runner implementations may
also prune unused tensors on a zero rewrite count.

A strict expected-failure structural contract now covers both targets. It
requires each boundary to record the tensor count, assign the SINet result,
unpack both ordered cluster results, and guard the immediate reconciliation by
the three exact counters or a tensor-count decrease. Production is unchanged.

At implementation, preserve the same owner order, target LayoutState forms,
diagnostics, and fallback/main separation. Add lowerer-level unchanged,
per-counter, and prune-only wiring coverage before running the complete SINet
shuffle, SE layout, orchestration, core, architecture, pass-efficiency, and
TensorFlow-import-blocked gates sequentially. Commit and push only; do not
create or update a pull request.

## Terminal SE/FC/Gather reconciliation implementation checkpoint

Both main and fallback boundaries now record tensor count, assign the SINet
shuffle result, unpack the ordered SE-FC and Gather results, and guard the
existing immediate reconciliation by the three exact counters or a tensor-count
decrease. The owner order, main Session LayoutState, fallback `None` LayoutState,
diagnostics, and recursive fallback separation are unchanged.

The strict two-boundary architecture contract is green. A synthetic main-path
lowerer fixture covers unchanged, each of the three positive counters, and
prune-only outcomes; every changed outcome adds exactly one reconciliation.
Sequential validation across the complete related owners/orchestrator, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`566 passed in 27.36s`.

At resume, re-inventory the remaining unconditional static-shape
reconciliations outside this absolute-final block. Prefer a boundary with a
complete returned mutation contract; where a runner prunes on zero rewrite,
preserve that behavior with explicit accounting. Characterize before changing
production, then commit and push only. Do not create or update a pull request.

## Late binary-repair reconciliation characterization checkpoint

The late boundary after the first post-repair reconciliation consists of
static shape-signature sanitization, exact rank-four binary adapter repair,
singleton-broadcast adapter repair, and another unconditional reconciliation.
Their exact mutation counters are respectively
`sanitized_static_shape_signature_consistency`,
`inserted_rank4_binary_layout_fix_transpose`, and
`repaired_rank4_binary_singleton_broadcast_layout_mismatch`.

The exact adapter owner also prunes unused tensors even when its rewrite count
is zero. A new owner fixture freezes that behavior. A strict expected-failure
architecture contract therefore requires a pre-boundary tensor count, all
three assigned results, and an immediate guard covering every counter plus a
tensor-count decrease. Production is unchanged.

At implementation, change only this second late repair boundary. Add lowerer
wiring coverage for unchanged, all three positive counters, and prune-only
outcomes. Run the complete binary-adapter, static-signature, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Late binary-repair reconciliation implementation checkpoint

The second late repair boundary now records tensor count and assigns the static
signature, exact binary adapter, and singleton adapter results. Its immediate
reconciliation runs only after one of the three exact mutation counters is
positive or the exact adapter's zero-rewrite prune reduced the tensor table.
The first repair boundary and final boundary-signature sanitizer are unchanged.

The strict architecture expectation is green. A synthetic lowerer fixture
covers unchanged, all three individual positive counters, and prune-only
outcomes; every changed outcome adds exactly one reconciliation. Sequential
validation across the complete binary-adapter, static-signature, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`402 passed in 26.49s`; focused owner coverage is `63 passed`.

At resume, inspect the earlier shared reconciliation after static sanitizers,
two binary repairs, and the singleton-consecutive-Reshape cluster. It needs a
combined aggregate result from that cluster before it can be guarded safely.
Do not infer no-op status from only the two repair counters. Characterize first,
then commit and push only; do not create or update a pull request.

## Singleton/consecutive-Reshape result characterization checkpoint

The earlier shared reconciliation boundary includes the three-runner
singleton-channel transpose, duplicate-fan-out, and consecutive-Reshape
cluster. The generic recovery utility already returns ordered callback values,
but `run_singleton_consecutive_reshape()` and its private lowerer helper still
discard all three dictionaries.

A strict expected-failure contract now requires an ordered three-result tuple
while preserving owner order, the shared `ModelIRPassStateScope`, all three
target forms, arguments, diagnostics, and exception behavior. Production and
the shared reconciliation remain unchanged.

At implementation, forward a typed three-dictionary tuple from the runner and
private helper. Existing call sites may continue ignoring it. Validate every
orchestration suite and the core/architecture/TensorFlow boundary before
changing the shared reconciliation. Commit and push only; do not create or
update a pull request.

## Singleton/consecutive-Reshape result implementation checkpoint

`run_singleton_consecutive_reshape()` now returns the ordered result
dictionaries for singleton-channel transpose, duplicate fan-out, and
consecutive Reshape. The lowerer's private helper forwards the typed triple.
All three existing call sites continue ignoring it, so ModelIR and
reconciliation behavior remain unchanged in this checkpoint.

Sequential validation across every orchestration suite is
`294 passed in 3.93s`. The core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `339 passed in 25.88s`. Focused helper,
ordered-result, target-form, and shared-state-scope contracts pass.

At resume, characterize the earlier shared reconciliation using all preceding
sanitizer/repair results, the three cluster result dictionaries, and tensor-count
prune accounting. Do not use an incomplete subset of cluster counters. Commit
and push only; do not create or update a pull request.

## Shared late-reconciliation characterization checkpoint

The earlier shared reconciliation is owned by six direct results and the three
singleton/consecutive cluster results: dynamic boundary-signature realignment,
HardSwish shape sanitization, Squeeze sanitization, wrong-way Conv transpose
sanitization, exact and singleton binary repair, singleton-channel transpose,
duplicate Reshape fan-out, and consecutive Reshape cleanup.

All returned values in these nine dictionaries are mutation counts. The new
empty-cluster fixture freezes its exact zero-only dictionaries. A strict
expected-failure architecture contract requires all nine results to be assigned
and passed to one compact `_stats_have_positive_count()` predicate, with a
pre-boundary tensor count covering zero-rewrite pruning. Production remains
unchanged.

At implementation, add the private pure mutation-count helper, capture all nine
results in their existing order, and guard only the immediate shared
reconciliation. Add helper unit tests and lowerer wiring for every dictionary
plus prune-only behavior. Validate all owner/orchestration/core gates
sequentially, then commit and push only; do not create or update a pull request.

## Shared late-reconciliation implementation checkpoint

The earlier shared boundary now records its tensor count, assigns the six
direct sanitizer/repair results, and unpacks the three ordered
singleton/consecutive cluster results without changing their execution order.
Its immediate static-shape reconciliation runs only when one of those nine
pure mutation-count dictionaries contains a positive value or zero-rewrite
pruning reduces the tensor table.

`_stats_have_positive_count()` is deliberately narrow: it receives only pure
mutation dictionaries and treats zero or negative values as unchanged. A
synthetic lowerer fixture covers the all-zero path, every individual result,
and prune-only behavior; each changed outcome adds exactly one reconciliation.
The singleton/consecutive boundary contract was updated to recognize the
result-capturing assignment. Three older indexed-pass tests were also aligned
with the extracted helper ownership in `core.model_ir_utils`, removing stale
references to lowerer-private aliases without changing production behavior.

Sequential validation across the complete related owner/orchestration, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`432 passed in 27.29s`. The focused helper, lowerer wiring, extracted-owner,
and structural checks are `7 passed in 0.70s`.

At resume, re-inventory the remaining unconditional static-shape
reconciliations and select the next boundary only when every mutating owner has
a complete returned result or explicit prune accounting. Characterize it
before changing production, then commit and push only. Do not create or update
a pull request.

## Indexed binary-layout convergence characterization checkpoint

`_run_indexed_binary_layout_convergence()` still executes its complete
three-round sequence after reaching a stable round. Each round runs rank-four
broadcast-constant repair, stale NCHW-to-NHWC binary-Transpose repair, and
static-shape reconciliation with one shared `ModelIRGraphIndex`.

The three returned dictionaries contain only mutation counts. The two repair
owners prune only after a positive rewrite, and reconciliation changes metadata
only, so an all-zero round is a complete convergence signal. Strict
expected-failure cases require immediate zero-change convergence to stop after
one round and a reconciliation-only first change to stop after the following
zero round. A passing contract keeps the existing maximum at three rounds when
reconciliation reports a change every time and verifies the shared index.

At implementation, preserve owner order, one-index ownership, aggregate return
statistics, and the three-round cap. Break only after adding all three current
round results and confirming every mutation dictionary is zero. Run the full
indexed binary-layout owner suite plus core, pass-efficiency, architecture, and
TensorFlow-import-blocked gates sequentially. Commit and push only; do not
create or update a pull request.

## Indexed binary-layout convergence implementation checkpoint

`_run_indexed_binary_layout_convergence()` now accumulates broadcast repair,
stale binary-Transpose repair, and static-shape reconciliation results for the
current round, then stops when all three pure mutation dictionaries are zero.
Changing rounds continue with the same `ModelIRGraphIndex`, owner order, and
aggregate counters, and the existing three-round maximum remains unchanged.

The immediate-stability and second-round-stability characterizations are now
passing. The always-changing fixture still executes exactly three rounds, and
the original multi-repair fixture remains ModelIR/stat identical to the former
fixed three-round sequence. Two adjacent binary-owner tests were updated to
monkeypatch the extracted owner modules rather than aliases that no longer
exist in the lowerer; this changes test ownership only.

Sequential validation across the complete indexed convergence, binary layout,
binary adapter, shape reconciliation, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `402 passed in 26.66s`. The focused
convergence owner suite is `11 passed in 0.54s`.

At resume, continue the reconciliation inventory outside fixed-point helpers.
Choose a boundary only when all mutations since the preceding reconciliation
are represented by returned counters or explicit prune accounting. Preserve
phase order, characterize before production changes, and commit/push only. Do
not create or update a pull request.

## Indexed shape-convergence stable-scan characterization checkpoint

`_run_indexed_shape_convergence_cleanup()` owns dead-operator pruning, an
initial static-shape reconciliation, dynamic-Reshape resolution, and a final
static-shape reconciliation under one `ModelIRGraphIndex`. The final scan still
runs when all three preceding mutation dictionaries are zero.

A strict expected-failure fixture now requires that complete stable path to
execute only one reconciliation. Three passing fixtures preserve the final
reconciliation independently after a prune mutation, a first-reconciliation
metadata mutation, or a dynamic-Reshape mutation. This deliberately does not
assume that one changing reconciliation is already a fixed point.

At implementation, initialize the final result to the exact zero dictionary
and invoke the second reconciliation only when one of the three preceding pure
mutation dictionaries is positive. Preserve aggregate statistics, the shared
index, LayoutState forwarding, and both production call boundaries. Validate
dynamic Reshape, shape reconciliation, final convergence, core, architecture,
pass-efficiency, and TensorFlow-import-blocked suites sequentially. Commit and
push only; do not create or update a pull request.

## Indexed shape-convergence stable-scan implementation checkpoint

`_run_indexed_shape_convergence_cleanup()` now initializes the final
reconciliation result with the exact zero counter. It runs the second static
shape scan only when dead pruning, the first reconciliation, or dynamic-Reshape
resolution reports a positive mutation count.

The all-zero path therefore uses one reconciliation. Each independently
changing owner still triggers the second scan, preserving possible second-order
metadata convergence. Aggregate statistics, the shared `ModelIRGraphIndex`,
LayoutState forwarding, and both production call boundaries are unchanged. The
architecture contract directly verifies the three-result guard and the guarded
call's shared index.

Sequential validation across dynamic Reshape, shape reconciliation, indexed
final convergence, graph cleanup, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `380 passed in 25.82s`. Focused stable,
per-owner mutation, structure, and end-to-end equivalence coverage is
`6 passed in 2.07s`.

At resume, inspect the larger final shape/activation convergence coordinator.
Do not guard any of its three remaining reconciliation boundaries until the
preceding sanitizer, Reshape resolver, or fusion result completely accounts for
the relevant mutation interval. Characterize each candidate independently,
then commit and push only. Do not create or update a pull request.

## First final-convergence reconciliation characterization checkpoint

The indexed final shape/activation coordinator currently runs an additional
static-shape reconciliation immediately after
`_run_indexed_shape_convergence_cleanup()` and HardSwish shape sanitation, even
when both returned mutation dictionaries are entirely zero.

A strict expected-failure event-order fixture requires that one scan to be
absent on the complete stable path. Two passing cases preserve it after either
the convergence aggregate or HardSwish sanitizer reports a mutation. Every
stage receives the same `ModelIRGraphIndex`; the later Reshape reconciliation
and final post-fusion reconciliation remain in their original order.

At implementation, initialize only `first_reconcile_stats` with the exact zero
counter and guard that call with both predecessor dictionaries. Do not change
the second or final reconciliation in this checkpoint. Preserve aggregate
statistics, one-index ownership, LayoutState forwarding, and full legacy
ModelIR equality. Validate indexed final convergence, dynamic Reshape, shape
reconciliation, core, architecture, pass-efficiency, and
TensorFlow-import-blocked suites sequentially. Commit and push only; do not
create or update a pull request.

## First final-convergence reconciliation implementation checkpoint

The first extra reconciliation in
`_run_indexed_final_shape_activation_convergence()` now initializes its result
with the exact zero counter and runs only when the preceding indexed
shape-convergence aggregate or HardSwish sanitizer contains a positive
mutation count.

The complete stable path skips that scan. Either predecessor mutation retains
it in the same position with the same `ModelIRGraphIndex`. The later
dynamic-Reshape reconciliation and final post-fusion reconciliation are
unchanged. The architecture contract verifies the exact two-result guard,
guarded-call position, and shared index.

Sequential validation across indexed final convergence, dynamic Reshape,
shape reconciliation, graph cleanup, HardSwish/SE layout, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`391 passed in 25.95s`. Focused stable, per-predecessor mutation, structure,
and legacy-equivalence coverage is `6 passed in 2.08s`.

At resume, characterize the dynamic-Reshape-to-second-reconciliation boundary.
The guard must include both `first_reconcile_stats` and `reshape_stats` so a
first reconciliation that changed metadata can still receive a convergence
scan even when Reshape resolution is a no-op. Leave the final post-fusion scan
unchanged until its own interval is characterized. Commit and push only; do
not create or update a pull request.

## Second final-convergence reconciliation characterization checkpoint

The second additional scan in
`_run_indexed_final_shape_activation_convergence()` follows the optional first
reconciliation and dynamic-Reshape resolution. It still runs when both of
those mutation dictionaries are zero.

A strict expected-failure event-order fixture requires that stable path to
proceed directly to activation fusion. Two passing paths preserve the scan
after a first-reconciliation metadata mutation and after a dynamic-Reshape
rewrite. The shared `ModelIRGraphIndex`, aggregate statistics, already-guarded
first scan, and final post-fusion scan remain explicit in every fixture.

At implementation, initialize only `second_reconcile_stats` with the exact
zero counter and guard its call with `first_reconcile_stats` and
`reshape_stats`. Do not broaden the guard to earlier aggregate results and do
not change the final reconciliation. Update the structural contract to verify
both guards in source order, then validate indexed final convergence, dynamic
Reshape, shape reconciliation, core, architecture, pass-efficiency, and
TensorFlow-import-blocked suites sequentially. Commit and push only; do not
create or update a pull request.

## Second final-convergence reconciliation implementation checkpoint

`second_reconcile_stats` now initializes with the exact zero counter. The
second additional static-shape scan runs only when `first_reconcile_stats` or
`reshape_stats` reports a positive mutation count.

The complete stable path and predecessor-only changes whose first scan is
already stable now proceed directly to activation fusion. A changing first
reconciliation or dynamic-Reshape rewrite retains the second scan in its
original position. The structural contract verifies both reconciliation guards
in source order and confirms every direct and guarded owner receives the same
`ModelIRGraphIndex`. The final post-fusion scan is unchanged.

Sequential validation across indexed final convergence, dynamic Reshape,
shape reconciliation, graph cleanup, HardSwish/SE layout, core,
pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`394 passed in 26.57s`. Focused stable, mutation-source, ordered-guard, and
legacy-equivalence coverage is `9 passed in 0.56s`.

At resume, characterize the final post-fusion reconciliation. Its guard must
cover both `second_reconcile_stats` and the complete activation-fusion result,
so metadata changes from the second scan retain one convergence opportunity
even when fusion is a no-op. Characterize fusion counter completeness and any
zero-count prune behavior before changing production. Commit and push only; do
not create or update a pull request.

## Post-fusion reconciliation characterization checkpoint

The final scan in `_run_indexed_final_shape_activation_convergence()` follows
the optional second reconciliation and activation fusion. Fusion exposes
complete rewrite counters, but its owner unconditionally calls
`_prune_unused_tensors()` even when every counter is zero.

A new owner fixture freezes zero-fusion pruning, its lineage event, and
LayoutState synchronization. A strict expected-failure event-order fixture
requires the final scan to be skipped only for zero second-reconciliation and
fusion results with no tensor-count reduction. Passing fixtures preserve it
after a second-reconciliation metadata change, a fusion rewrite, and a
zero-rewrite prune-only mutation.

At implementation, record tensor count immediately before fusion, initialize
`final_reconcile_stats` with the exact zero counter, and guard the final scan
with `second_reconcile_stats`, every fusion counter through
`_stats_have_positive_count()`, or a tensor-count decrease. Extend the
structural contract to a third ordered guard. Validate activation fusion,
indexed final convergence, dynamic Reshape, shape reconciliation, core,
architecture, pass-efficiency, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Post-fusion reconciliation implementation checkpoint

The coordinator now records `fusion_tensor_count` immediately before
activation fusion and initializes `final_reconcile_stats` with the exact zero
counter. The final scan runs only when `second_reconcile_stats` or any fusion
counter is positive, or when zero-rewrite pruning reduces the tensor table.

The complete stable path now ends after fusion. A changing second
reconciliation, fusion rewrite, and prune-only mutation each preserve the scan
with the same `ModelIRGraphIndex`. The architecture contract verifies all three
guards in source order, the tensor-count boundary, and index forwarding. The
activation-fusion owner fixture confirms zero-count pruning and LayoutState
synchronization.

Sequential validation across activation fusion, indexed final convergence,
dynamic Reshape, shape reconciliation, graph cleanup, HardSwish/SE layout,
core, pass-efficiency, architecture, and TensorFlow-import-blocked suites is
`414 passed in 26.42s`. Focused final-coordinator, prune-only, structure, and
legacy-equivalence coverage is `13 passed in 2.27s`.

At resume, return to the larger lowerer reconciliation inventory. The three
indexed final-convergence scans now have explicit mutation contracts; do not
merge their guards unless tests can still distinguish each convergence
opportunity. Select the next boundary only when every mutation since the prior
scan has a returned counter or prune accounting. Commit and push only; do not
create or update a pull request.

## Placeholder-MatMul repair reconciliation characterization checkpoint

The absolute-final placeholder-MatMul restoration block currently performs a
first static-shape reconciliation, exact rank-four binary repair, singleton
broadcast repair, and a second unconditional reconciliation whenever the
restoration counter is positive.

The first scan must remain. Its returned mutation counter and both repair
counters completely describe their rewrites, while the exact repair may also
prune unused tensors on a zero rewrite. A strict expected-failure lowerer
fixture requires the second scan only after a changing first reconciliation,
an exact repair, a singleton repair, or prune-only tensor-count reduction.

At implementation, assign the restoration result, retain its existing outer
guard, capture the first reconciliation and both repair results, and record
tensor count before the repairs. Guard only the second reconciliation with the
three dictionaries or a tensor-count decrease. Preserve call order,
LayoutState forwarding, and the following topology sort. Validate placeholder
MatMul restoration, binary adapters, shape reconciliation, core,
architecture, pass-efficiency, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Placeholder-MatMul repair reconciliation implementation checkpoint

The restoration result is now assigned to
`final_placeholder_matmul_stats` before the unchanged positive-result guard.
Inside that block, the first reconciliation remains mandatory and its result is
captured. The lowerer then records tensor count and captures exact and
singleton binary-repair results in the original order.

The second reconciliation now runs only when the first scan or either repair
reports a positive mutation, or when zero-rewrite exact repair pruning reduces
the tensor table. The topology sort remains unconditional inside the
restoration block. A structural contract fixes all six statements and the
complete guard; lowerer wiring independently covers every mutation source and
prune-only behavior.

Sequential validation across placeholder-MatMul/dynamic Reshape, binary
adapters, shape reconciliation, core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites is `403 passed in 26.48s`. Focused wiring and
structure coverage is `2 passed in 2.27s`.

At resume, continue the direct lowerer reconciliation inventory. Avoid the
large phase-barrier scans unless every intervening owner result is preserved;
prefer another local conditional block with a complete result interval.
Characterize before production changes, then commit and push only. Do not
create or update a pull request.

## Late binary-layout recovery runner characterization checkpoint

The conditional block after late binary repair currently runs PReLU
passthrough, dual pre-Add recovery, terminal affine-FC recovery, optional
PReLU-BMM recovery, affine pre/post recovery, optional generic layout cleanup,
and then reconciles unconditionally.

Every owner returns rewrite counters, but several owners prune unused tensors
on zero rewrites. Generic layout cleanup also returns `iterations`, which is an
execution count rather than a mutation. Strict expected-failure architecture
and lowerer-wiring contracts now require one dedicated runner. It must preserve
the outer branch, owner order, optional branches, Session LayoutState and
diagnostics; return only mutation counts plus net pruning; and trigger
reconciliation only for a positive aggregate.

At implementation, add a TensorFlow-independent pass module that imports the
six existing owners directly. Normalize optional-owner counters to zero, filter
the layout `iterations` field, and add `pruned_unused_tensors` from the net
tensor-count reduction. Replace the inline lowerer sequence with one runner
assignment and one aggregate guard. Validate the runner, every affected owner,
core, architecture, pass-efficiency, and TensorFlow-import-blocked suites
sequentially. Commit and push only; do not create or update a pull request.

## Late binary-layout recovery runner implementation checkpoint

The new TensorFlow-independent `run_late_binary_layout_recovery()` pass module
now owns the full late recovery cluster. It preserves the existing PReLU,
dual-pre-Add, terminal affine-FC, optional PReLU-BMM, affine pre/post, and
optional generic-layout order. LayoutState and diagnostics cross the new
boundary explicitly, and optional owners return stable zero counters when they
are disabled.

The runner exposes only mutation evidence: five owner rewrite counters, four
generic-layout mutation counters, and `pruned_unused_tensors` calculated from
net tensor-table reduction. It filters generic layout cleanup's non-mutating
`iterations` count. The lowerer retains the original outer condition but now
contains only the runner call and an aggregate mutation guard, so reconciliation
is skipped on a fully stable sequence and preserved after a rewrite or
prune-only mutation.

Dedicated runner and lowerer-wiring coverage initially passed as `5 passed in
2.16s`. The seven legacy tests that encoded direct-lowerer call ownership were
updated to recognize the new runner boundary and passed as `7 passed in
2.43s`. The complete sequential related suite covering all six owners, layout
cleanup, core, pass efficiency, architecture, and TensorFlow import blocking is
`523 passed in 26.93s`.

At resume, inventory the next local lowerer phase boundary before changing
production. Prefer a bounded sequence whose complete mutation interval is
represented by returned counters or explicit prune accounting; characterize
that boundary first. Commit and push coherent units only. Do not create or
update a pull request.

## Typed Constant lowering characterization checkpoint

The next unconditional reconciliation after terminal layout cleanup spans a
large phase whose owners do not yet expose complete mutation and pruning
evidence, so it was deliberately rejected as the next optimization boundary.
The selected bounded unit is the inline ONNX `Constant` special case in the
node-lowering loop. This also addresses the outstanding Pylance diagnostic on
protobuf `attribute.name` access.

Characterization fixes tensor data, dtype, shape/signature, graph output,
provenance, no-operator behavior, and the exact missing tensor-`value` error. A
strict expected-failure architecture contract requires one
TensorFlow-independent `op_families.constant` owner, a two-argument typed call
from the lowerer, and an explicit `onnx.AttributeProto` cast inside the owner.
The existing tensor-`value` feature scope, collision handling, and placeholder
replacement behavior must remain unchanged.

At implementation, move only the current Constant branch into the typed owner;
do not route it through general registry dispatch or add support for additional
Constant attribute encodings in this mechanical extraction. Validate focused
Constant/core/architecture tests, then the sequential core, architecture,
pass-efficiency, and TensorFlow-import-blocked suites. Commit and push only;
do not create or update a pull request.

## Typed Constant lowering implementation checkpoint

`op_families.constant.lower_constant_node()` now owns the existing
tensor-valued ONNX Constant lowering behavior. The lowerer keeps the same
`Constant` guard and progress/error boundary but delegates through one typed
`node`/`ctx` call. The owner explicitly casts protobuf attributes to
`onnx.AttributeProto`, so Pylance no longer has to infer `name` and `t` from the
ambiguous generated container element type.

The new owner preserves direct tensor creation, in-place placeholder updates,
legacy collision renaming, constants-map synchronization, dtype and
shape/signature normalization, graph-output behavior, provenance, and the exact
missing tensor-`value` exception. No additional Constant encodings were added,
and the path remains TensorFlow-independent.

Focused owner, public behavior, and architecture coverage is `5 passed in
2.27s`. The sequential core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `351 passed in 26.06s`. Ruff, Python bytecode
compilation, and whitespace validation pass. Pyright is not installed in the
existing uv environment; the source contract directly verifies the explicit
`AttributeProto` cast without adding a dependency.

At resume, inventory the remaining special control flow inside the ONNX node
lowering loop, especially the demand-driven unresolved-rank reconciliation
before shape-sensitive ops. Characterize its trigger and no-op behavior before
extracting it. Do not fold Constant into registry dispatch as part of that unit.
Commit and push only; do not create or update a pull request.

## Demand-driven shape readiness characterization checkpoint

The remaining inline special control flow in the ONNX node loop reconciles the
partial ModelIR immediately before six shape-sensitive op types. Five
characterization cases freeze the trigger: an unresolved nonconstant MatMul
input reconciles once; an equally unresolved Add input, fully known MatMul
input, explicit rank-one hint, and constant tensor do not reconcile before
dispatch.

A strict expected-failure architecture contract requires one
TensorFlow-independent `core.shape_readiness` owner to contain the target-op
set, unresolved-input test, and static-shape reconciliation. The central loop
must make one typed `node`/`ctx` call and must no longer define the nested
`_has_unresolved_rank` closure.

At implementation, preserve the exact six-op set and return the existing shape
reconciliation result, or its exact zero dictionary when no scan is requested.
Do not broaden unresolved-rank semantics or combine this lowering-time scan with
post-lowering phase barriers. Validate focused trigger/owner/core/architecture
coverage, then the sequential core, pass-efficiency, architecture, and
TensorFlow-import-blocked suites. Commit and push only; do not create or update
a pull request.

## Demand-driven shape readiness implementation checkpoint

`core.shape_readiness.reconcile_shape_sensitive_inputs_on_demand()` now owns
the six-op target set, unresolved-rank predicate, and lowering-time static-shape
scan. It returns `{"reconciled_static_tensor_shapes": 0}` for non-target ops,
known rank-two-or-higher inputs, explicit rank-one shape hints, and constant
tensors. An unresolved nonconstant input with no raw shape hint invokes the
same static reconciliation owner as before.

The ONNX node loop now contains one typed `node`/`ctx` call between `NodeView`
construction and dispatch. Its inline op set, input-name collection, and nested
closure were removed. This preserves demand-driven behavior while making the
policy independently testable and avoiding graph scans for stable paths.

Focused owner, integration-trigger, and architecture coverage is `16 passed in
2.27s`. The sequential shape-readiness, Constant, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `367 passed in 26.36s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, return to the post-lowering phase inventory. The broad terminal
phase before its unconditional reconciliation is still unsafe to guard until
all participating owners return complete mutation and prune evidence. Select a
smaller orchestration cluster inside that interval and characterize its result
propagation before changing the phase barrier. Commit and push only; do not
create or update a pull request.

## Late layout cluster result propagation characterization checkpoint

The broad terminal phase before its unconditional reconciliation still lacks a
complete mutation aggregate. Its final composite cluster is a bounded first
step: late layout/mean/SPP/gather/constant-fold/cast builds five required
invocations and an optional generic-layout invocation. The shared recovery
utility returns all results in order, but the public orchestrator and private
lowerer helper currently discard them.

Strict expected-failure coverage requires the exact five- or six-result tuple
to pass unchanged through both layers. Result order, shared state scope,
optional policy, diagnostics, exceptions, and the production call site remain
unchanged. No phase reconciliation is guarded in this checkpoint. The optional
generic-layout result includes `iterations`; a later aggregate must filter that
execution count and use only mutation keys.

At implementation, annotate and return the raw ordered tuple from the
orchestrator, then return it from the lowerer helper. Keep the existing call
site as an ignored expression so ModelIR behavior is mechanical and identical.
Validate both optional policies, state reuse, orchestration structure, core,
architecture, pass efficiency, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Late layout cluster result propagation implementation checkpoint

`run_late_layout_mean_spp_gather_constant_cast()` now returns the exact raw
tuple produced by `run_recovery_invocations()`. The lowerer's private helper
returns that tuple unchanged. Required-only execution therefore exposes five
ordered dictionaries, while layout-enabled execution exposes six, with the
same invocation order, state scope, diagnostics, and exception semantics.

The terminal production call remains an ignored expression. No result is yet
captured, no mutation aggregate is formed, and the unconditional phase
reconciliation is unchanged. The optional layout dictionary still carries the
non-mutating `iterations` field and must be normalized before any future guard.

Focused orchestration and structural coverage is `15 passed in 2.20s`. The
sequential late-cluster, child constant-fold/cast, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `376 passed in 26.14s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, characterize a normalized mutation aggregate for this one cluster
without guarding the broad phase reconciliation yet. The aggregate must map
the optional layout result to its four mutation keys, retain each required
pass dictionary, and account for any zero-rewrite pruning performed by those
owners. Commit and push only; do not create or update a pull request.

## Late layout cluster mutation summary characterization checkpoint

The raw five- or six-result tuple is now available, but it is not safe to pass
directly to `_stats_have_positive_count()` because the optional layout result
contains the non-mutating `iterations` field. Strict expected-failure coverage
defines a fixed summary: the four layout mutation keys, all required-pass
mutation dictionaries, and `pruned_unused_tensors` from the cluster's net
tensor-count reduction. Layout keys are explicit zeros when the optional pass
is disabled, and `iterations` is absent.

The terminal call site must record tensor count, capture the ordered raw tuple,
and compute this summary in three adjacent statements. The subsequent
Expand/Squeeze owner and unconditional reconciliation remain unchanged and do
not consume the summary yet.

At implementation, add a pure validating summary function to the orchestration
module and capture its result at the current production call. Preserve the raw
runner/helper return contract and all owner order. Add no reconciliation guard.
Validate both policies, malformed result length, net pruning, structure, core,
architecture, pass efficiency, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Late layout cluster mutation summary implementation checkpoint

The pure
`summarize_late_layout_mean_spp_gather_constant_cast_mutations()` helper now
validates the expected five- or six-result tuple, emits four fixed layout
mutation keys, drops `iterations`, merges every required-pass dictionary, and
adds a clamped `pruned_unused_tensors` count.

The production call site records the cluster's starting tensor count, captures
the raw result tuple, and derives `_late_layout_cluster_stats` from net tensor
reduction. The underscore is intentional: this evidence is staged but not yet
used by the broad phase barrier. Expand/Squeeze and the unconditional static
reconciliation remain exactly where they were.

Focused summary, malformed-length, pruning, return, boundary, and architecture
coverage is `19 passed in 2.28s`. The sequential late-cluster, child
constant-fold/cast, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `379 passed in 26.27s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, propagate mutation/prune evidence from another small cluster in the
same terminal interval. Do not use `_late_layout_cluster_stats` to guard the
phase reconciliation until every preceding owner since the prior mandatory
shape barrier is accounted for. Commit and push only; do not create or update a
pull request.

## Terminal Expand/Squeeze result capture characterization checkpoint

Immediately after the staged late-layout summary, the terminal
Expand/Squeeze-to-Reshape owner returns two complete mutation counters but its
result is discarded. The owner prunes only after a positive rewrite, so those
counters completely describe whether it changed ModelIR.

A strict expected-failure structure test requires the production call to assign
`_terminal_expand_squeeze_stats`, preserve Session LayoutState forwarding, and
leave the following unconditional static reconciliation immediately in place.
No phase barrier is guarded in this checkpoint.

At implementation, change only the expression to a staged underscored
assignment and update the two surrounding boundary contracts. Validate the
complete Expand/Squeeze owner suite, late-layout orchestration, core,
architecture, pass efficiency, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Terminal Expand/Squeeze result capture implementation checkpoint

The terminal `_replace_expand_dims_and_squeeze_with_reshape()` call now assigns
its unchanged result dictionary to `_terminal_expand_squeeze_stats`. Its
Session LayoutState, ordering after the late-layout summary, and immediately
following unconditional static reconciliation are unchanged. The staged result
is not consumed by a guard.

Focused owner, boundary, summary, and orchestration coverage is `20 passed in
2.33s`. The sequential Expand/Squeeze, late-layout, child constant-fold/cast,
core, pass-efficiency, architecture, and TensorFlow-import-blocked gate is `382
passed in 27.09s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, continue moving backward through the same terminal interval and
propagate one small orchestration cluster's mutation/prune evidence. The broad
phase barrier must remain unconditional until every intervening owner has
complete accounting. Commit and push only; do not create or update a pull
request.

## Late hard-activation cluster evidence characterization checkpoint

Moving backward through the terminal interval, the late hard-activation/layout
pair is the next bounded cluster. The required hard-activation owner and
optional generic-layout owner both may prune on zero rewrites; the layout result
also contains non-mutating `iterations`. The orchestrator and private helper
currently discard the recovery utility's ordered one- or two-result tuple.

Strict expected-failure coverage requires raw tuple propagation, tuple-length
validation, four fixed layout mutation keys, `iterations` filtering, required
counter preservation, and net `pruned_unused_tensors`. The production call must
capture count/results/summary in order while preserving its Hardswish-SE and
pre-Concat boundaries. No reconciliation guard changes.

At implementation, return the raw tuple through both runner layers, add a pure
summary helper, and stage `_late_hard_activation_stats` at the existing call
site. Validate both policies, malformed lengths, pruning, owner order, shared
state, core, architecture, pass efficiency, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Late hard-activation cluster evidence implementation checkpoint

The orchestration runner and lowerer's private delegate now return the recovery
utility's ordered one- or two-result tuple. The pure
`summarize_late_hard_activation_layout_mutations()` helper validates the tuple
length for the active policy, preserves both required hard-activation counters,
emits four fixed zero-default layout mutation keys, excludes the non-mutating
layout `iterations` metric, and adds a clamped `pruned_unused_tensors` count.

At the original terminal call site, the lowerer records the starting tensor
count, captures the raw results, and derives `_late_hard_activation_stats` from
the exact net tensor reduction. The leading underscore marks staged evidence;
the broad phase reconciliation remains unconditional. The existing pass order,
optional-layout policy, shared LayoutState, Hardswish-SE/pre-Concat boundaries,
public behavior, artifacts, and TensorFlow-free direct path are unchanged.

Focused orchestration coverage is `12 passed in 0.64s`. The sequential late
hard-activation, Expand/Squeeze, late-layout, constant-fold/cast, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `394 passed
in 28.01s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, continue moving backward through the same terminal interval and
characterize another small owner or orchestration cluster before using any
staged evidence to guard broad reconciliation. Commit and push only; do not
create or update a pull request.

## Terminal Hardswish-SE evidence characterization checkpoint

The terminal Hardswish-SE owner immediately before the late hard-activation
cluster returns a complete rewrite counter, but it also invokes unused-tensor
pruning unconditionally. A dedicated zero-rewrite fixture confirms that the
owner can remove a tensor while returning zero, so its raw dictionary alone
cannot prove ModelIR stability.

A strict expected-failure structural contract requires two adjacent production
assignments: capture the starting tensor count, then merge the unchanged owner
result with an exact, clamped `pruned_unused_tensors` count. The preceding
split/conv bridge call and following late hard-activation count remain the
fixed boundaries. No pass call, ordering, policy, or reconciliation changes.

At implementation, replace only the discarded terminal call with this
prune-aware staged evidence and update the two affected boundary contracts.
Validate the complete Hardswish-SE owner suite, late hard-activation
orchestration, terminal recovery structure, core, architecture, pass
efficiency, and TensorFlow import blocking sequentially. Commit and push only;
do not create or update a pull request.

## Terminal Hardswish-SE evidence implementation checkpoint

The terminal call now records `terminal_hardswish_se_tensor_count` and stores
the owner's unchanged rewrite dictionary in `_terminal_hardswish_se_stats`
together with the exact, clamped net `pruned_unused_tensors` count. This makes
the owner's unconditional zero-rewrite pruning observable without changing the
owner, its compatibility wrapper, or its output counter.

The staged dictionary is observation-only. The preceding split/conv bridge,
following late hard-activation cluster, pass policy and order, shared
LayoutState, broad phase reconciliation, public behavior, artifacts, and
TensorFlow-free direct path are unchanged.

Focused Hardswish-SE and late hard-activation coverage is `22 passed in 0.74s`.
The sequential Hardswish-SE, late hard-activation, QKV, split/conv bridge,
SINet terminal, Expand/Squeeze, late-layout, constant-fold/cast, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `469 passed
in 28.37s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, continue backward with the adjacent split/conv bridge owner or the
late QKV cluster. Characterize raw results and zero-rewrite pruning before
extending the staged terminal aggregate. Do not guard broad reconciliation
until the full interval is accounted for. Commit and push only; do not create
or update a pull request.

## Terminal split/conv bridge evidence characterization checkpoint

The indexed split/conv/concat bridge owner immediately preceding the terminal
Hardswish-SE capture returns one rewrite counter and prunes unused tensors only
when that counter is positive. Its bounded candidate/idempotence coverage and
24 transactional unsafe-candidate fixtures establish that a zero result leaves
ModelIR unchanged. The raw return is therefore complete mutation evidence.

A strict expected-failure structural contract requires the terminal call to
assign its unchanged result to `_terminal_split_conv_concat_bridge_stats` while
preserving ModelIR and Session LayoutState arguments. The late QKV call and
Hardswish-SE tensor-count assignment remain the exact neighboring boundaries.
No reconciliation guard changes.

At implementation, replace only the discarded expression with the staged
assignment and update boundary-aware QKV and architecture tests. Validate the
complete indexed owner suite and the adjacent QKV, Hardswish-SE,
hard-activation, core, pass-efficiency, architecture, and TensorFlow import
blocker sequentially. Commit and push only; do not create or update a pull
request.

## Terminal split/conv bridge evidence implementation checkpoint

Only the split/conv/concat bridge invocation directly after the late QKV
cluster now assigns its unchanged result to
`_terminal_split_conv_concat_bridge_stats`. The earlier invocations of the same
owner remain untouched. This confines mutation staging to the terminal
interval currently being accounted and prevents an ambiguous first-match
rewrite.

The owner call still receives the same ModelIR and Session LayoutState, and the
late QKV and Hardswish-SE boundaries remain adjacent. Since pruning occurs only
after a positive rewrite, the existing counter remains complete evidence. The
staged result is observation-only and no reconciliation guard changes.

Focused indexed owner, QKV, and Hardswish-SE coverage is `69 passed in 0.80s`.
The sequential indexed owner, QKV, Hardswish-SE, late hard-activation, SINet
terminal, Expand/Squeeze, late-layout, constant-fold/cast, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `470 passed
in 28.41s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, characterize the late QKV cluster's ordered results and prune
behavior. Preserve `include_prefix=False`, the optional generic-layout policy,
and all three production forms. Commit and push only; do not create or update a
pull request.

## Late QKV mutation evidence characterization checkpoint

The QKV orchestration owner serves three production policies: required prefix
plus bridge, bridge only, and optional generic layout plus bridge. The terminal
form uses `include_prefix=False`. Its generic layout and required bridge owners
both may prune on zero rewrites, and generic layout also returns non-mutating
`iterations`, so raw counters alone are insufficient.

Strict expected-failure coverage requires the runner and private delegate to
return the ordered one- or two-result tuple for all three policies. A pure
summary must validate tuple length and emit a stable schema of four layout,
four prefix, two bridge, and one net-prune counter, with inactive-family zeros
and no `iterations`. The terminal production form must capture starting tensor
count, raw results, and normalized `_late_qkv_stats` between shape-extract and
the staged split/conv bridge. No reconciliation guard changes.

At implementation, preserve all invocation defaults and orders, return raw
results through both layers, add the fixed pure summary, and stage only the
terminal call. Validate all policies, malformed lengths, summary filtering,
net pruning, shared pass state, boundary structure, core, pass efficiency,
architecture, and TensorFlow import blocking sequentially. Commit and push
only; do not create or update a pull request.

## Late QKV mutation evidence implementation checkpoint

The QKV orchestration runner and lowerer's private delegate now return the
ordered raw tuple for all three production policies. The pure
`summarize_qkv_attention_mutations()` helper validates the policy-specific
tuple length and always emits four layout, four prefix, two bridge, and one
net-prune counter. Inactive families are explicit zeros, generic-layout
`iterations` is excluded, and pruning is clamped to the exact nonnegative
tensor-count reduction.

Only the terminal `include_prefix=False` invocation stages
`late_qkv_tensor_count`, `late_qkv_results`, and `_late_qkv_stats`. The two
default invocation forms still discard their now-available results. All
defaults, pass IDs and order, optional-layout policy, shared pass state,
shape-extract/split-bridge boundaries, and unconditional broad reconciliation
remain unchanged.

Focused QKV, indexed split/conv bridge, and Hardswish-SE coverage is `73 passed
in 0.91s`. The sequential QKV, split/conv bridge, Hardswish-SE, late
hard-activation, SINet terminal, Expand/Squeeze, late-layout,
constant-fold/cast, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `503 passed in 28.68s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, move backward to the shape-extract owner immediately before late
QKV. Confirm whether its return counter fully accounts for pruning before
staging it. Commit and push only; do not create or update a pull request.

## Pre-QKV shape-extract evidence characterization checkpoint

The shape-extract owner prunes unused tensors only after its rewrite counter is
positive. Its idempotence coverage and eight unsafe/unsupported snapshot
fixtures confirm that zero means no ModelIR mutation, so the raw result is
complete evidence.

Three production invocations share the same owner. A strict expected-failure
contract selects only the terminal call between the late SPP pair and
`late_qkv_tensor_count` and requires it to assign
`_late_pre_qkv_shape_extract_stats`. It also freezes the total call count at
three to prevent another ambiguous first-match edit. No arguments, order, or
reconciliation behavior change.

At implementation, replace only that exact terminal expression and update the
late-SPP, QKV, and architecture boundary contracts. Validate the complete
shape-extract suite and adjacent terminal orchestration, core, pass efficiency,
architecture, and TensorFlow import blocker sequentially. Commit and push only;
do not create or update a pull request.

## Pre-QKV shape-extract evidence implementation checkpoint

Only the shape-extract invocation between the late SPP pair and QKV starting
count now assigns its unchanged result to
`_late_pre_qkv_shape_extract_stats`. The first production call and the later
absolute-end shape-extract call remain expressions, and the structural contract
continues to require exactly three owner calls.

The owner still receives only ModelIR, its rewrite/prune behavior is unchanged,
and the staged result is not consumed by a reconciliation guard. Late SPP and
QKV remain its exact neighbors, with all other terminal evidence and the broad
reconciliation unchanged.

Focused shape-extract, late-SPP, and QKV coverage is `37 passed in 0.94s`. The
sequential shape-extract, late-SPP, QKV, split/conv bridge, Hardswish-SE, late
hard-activation, SINet terminal, Expand/Squeeze, late-layout,
constant-fold/cast, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `524 passed in 29.02s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, propagate ordered results and net-prune evidence from the late SPP
pair immediately before this owner. Commit and push only; do not create or
update a pull request.

## Late SPP pair mutation evidence characterization checkpoint

Inspection shows that both late SPP child owners prune only after a positive
rewrite. Their preflight/no-op paths leave ModelIR unchanged, so the SPP and
concat-unary-conv counters already provide complete evidence and no separate
net-prune count is required.

Strict expected-failure coverage requires the orchestration runner and lowerer
delegate to return the ordered two-result tuple. A pure summary validates the
exact length and emits the two declared mutation counters. The production call
must stage `late_spp_results` and `_late_spp_stats` between the preceding raw
layout rewrite and `_late_pre_qkv_shape_extract_stats`. No reconciliation guard
changes.

At implementation, return results through both layers, add the pure fixed-key
summary, and stage both assignments at the existing call. Validate malformed
length, order, shared state, complete SPP and concat-unary-conv owner suites,
adjacent terminal orchestration, core, pass efficiency, architecture, and the
TensorFlow import blocker sequentially. Commit and push only; do not create or
update a pull request.

## Late SPP pair mutation evidence implementation checkpoint

The late SPP orchestration runner and lowerer's private delegate now return the
ordered two-result tuple. The pure
`summarize_late_spp_concat_unary_conv_mutations()` helper validates the exact
length and exposes only the SPP and concat-unary-conv mutation counters. Since
both owners prune only after a positive rewrite, no tensor-count proxy is
needed.

Production captures `late_spp_results` and `_late_spp_stats` at the existing
call site. The preceding raw layout rewrite, following pre-QKV shape-extract
result, owner order, shared pass state, and broad reconciliation remain
unchanged. The staged summary is not consumed by a guard.

Focused late-SPP and shape-extract coverage is `23 passed in 0.71s`. The
sequential complete SPP, concat-unary-conv, late-SPP, shape-extract, QKV,
split/conv bridge, Hardswish-SE, late hard-activation, SINet terminal,
Expand/Squeeze, late-layout, constant-fold/cast, shared-context, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `596 passed
in 29.28s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately preceding terminal strided-slice/pad/concat
raw owner. Confirm its return and prune contract before staging evidence.
Commit and push only; do not create or update a pull request.

## Second terminal slice/pad/concat evidence characterization checkpoint

The strided-slice/pad/concat bridge owner prunes only after a positive rewrite,
and its transactional rejection and idempotence coverage make the single
counter complete mutation evidence. Two direct production calls use this owner,
with the second terminal affine recovery between them.

A strict expected-failure contract requires only the second call to assign
`_terminal_slice_pad_concat_stats`. It freezes the preceding terminal affine
recovery, following `late_spp_results`, direct-call count of two, and first call
as an unchanged expression. No arguments, owner behavior, or reconciliation
guard changes.

At implementation, replace the exact second expression and update late-SPP,
terminal-affine, and architecture boundary tests. Validate the complete bridge
owner suite and adjacent terminal orchestration, core, pass efficiency,
architecture, and TensorFlow import blocker sequentially. Commit and push only;
do not create or update a pull request.

## Second terminal slice/pad/concat evidence implementation checkpoint

Only the second direct strided-slice/pad/concat invocation now assigns its
unchanged counter to `_terminal_slice_pad_concat_stats`; the first direct call
remains an expression. The terminal affine boundary contracts distinguish the
two positions while confirming that both still invoke the same owner with only
ModelIR.

The first expanded gate found one stale architecture expectation that still
required an expression before late SPP (`693 passed, 1 failed`). Updating that
specific boundary produced `694 passed in 27.88s`; a focused three-test check
also confirmed that the unrelated very-late boundary stayed unchanged. The
staged result is observation-only and no reconciliation behavior changes.

Focused bridge, terminal-affine, and late-SPP coverage is `107 passed in
1.04s`. The final sequential bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `694 passed in 27.88s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the second terminal affine recovery's ordered child results.
Its current result is discarded and several child owners may prune, so
characterize complete mutation evidence before extending the terminal
aggregate. Commit and push only; do not create or update a pull request.

## Second terminal affine mutation evidence characterization checkpoint

Runtime schema probing on an empty ModelIR confirms eleven ordered results and
twelve declared mutation keys. The final probable-NHWC sanitizer contributes
two keys; each other owner contributes one. Because child prune conditions
differ, the cluster needs an explicit net tensor-reduction count in addition to
its raw counters.

Strict expected-failure coverage requires ordered tuple propagation through the
orchestrator and lowerer delegate, exact eleven-result validation, fixed-key
summary extraction, and clamped `pruned_unused_tensors`. Only the second of two
production recovery calls must stage starting tensor count, raw results, and
`_terminal_affine_stats`; the first stays an expression. The preceding first
slice/pad/concat call and following staged second call remain exact boundaries.

At implementation, add the pure summary, return raw results through both
layers, and replace only the second call with three adjacent assignments.
Validate malformed length, all twelve keys, net pruning, owner order, both
production calls, complete recovery coverage, core, pass efficiency,
architecture, and TensorFlow import blocking sequentially. Commit and push
only; do not create or update a pull request.

## Second terminal affine mutation evidence implementation checkpoint

The terminal affine orchestration runner and lowerer's delegate now return the
ordered eleven-result tuple. The pure
`summarize_terminal_affine_concat_split_mutations()` helper validates the exact
length, extracts only the twelve declared mutation keys, and appends clamped
net `pruned_unused_tensors` evidence.

The first production recovery remains an expression. The second records
`terminal_affine_tensor_count`, captures `terminal_affine_results`, and derives
`_terminal_affine_stats` in three adjacent assignments. A forward-string return
annotation keeps the straight-line delegate's runtime loaded-data contract
limited to its shared context. The first slice/pad/concat and staged second
slice/pad/concat boundaries are unchanged, and no reconciliation consumes the
summary yet.

Focused recovery, second slice/pad/concat, and architecture coverage is `101
passed in 0.87s`. The sequential bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `696 passed in 28.11s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, move backward to the first strided-slice/pad/concat call immediately
before this cluster. Its positive-only prune contract permits direct result
capture, but keep the earlier terminal affine recovery unchanged. Commit and
push only; do not create or update a pull request.

## First terminal slice/pad/concat evidence characterization checkpoint

The first direct strided-slice/pad/concat call shares the same positive-only
prune contract as the already staged second call, so its raw counter is complete
mutation evidence. A strict expected-failure contract requires the first call
to assign `_pre_terminal_affine_slice_pad_concat_stats` between the raw
transpose/Mul/Add owner and `terminal_affine_tensor_count`.

The contract also requires exactly two direct owner statements and preserves
the second target as `_terminal_slice_pad_concat_stats`. This prevents another
ambiguous occurrence edit. The earlier terminal affine recovery and all owner
arguments, ordering, and reconciliation behavior remain unchanged.

At implementation, replace only the first direct expression and update the
terminal-affine and bridge-owner boundary contracts. Validate both occurrence
targets, the complete bridge suite, terminal affine recovery, core, pass
efficiency, architecture, and TensorFlow import blocking sequentially. Commit
and push only; do not create or update a pull request.

## First terminal slice/pad/concat evidence implementation checkpoint

The first direct strided-slice/pad/concat invocation now assigns its unchanged
counter to `_pre_terminal_affine_slice_pad_concat_stats`. The second direct
invocation retains the distinct `_terminal_slice_pad_concat_stats` target, and
the architecture contracts verify both exact positions and the shared owner.

This is observation-only staging. The owner still receives only ModelIR, its
transactional guards and positive-only pruning contract are unchanged, the
earlier terminal affine recovery remains untouched, and no reconciliation
decision consumes either staged dictionary.

Focused bridge-owner, terminal-affine, and architecture coverage is `102
passed in 0.87s`. The sequential bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `697 passed in 28.25s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the raw transpose/Mul/Add owner immediately before this
first direct call, or the preceding channel-slice cluster if that owner cannot
provide complete mutation evidence. Characterize its return and prune contract
before changing production. Commit and push only; do not create or update a
pull request.

## Pre-terminal affine post-ADD evidence characterization checkpoint

The indexed affine post-ADD owner returns one fixed rewrite counter and calls
unused-tensor pruning only after at least one accepted rewrite. Its existing
transactional and idempotence contracts, plus an explicit prune-hook test, show
that the raw result is complete mutation evidence without a tensor-count proxy.

There are three direct lowerer calls and one orchestrated occurrence. A strict
expected-failure contract selects only the first direct call, immediately after
the channel-slice cluster and before
`_pre_terminal_affine_slice_pad_concat_stats`, and requires it to assign
`_pre_terminal_affine_post_add_stats`. The two later direct calls must remain
expressions, and owner arguments and LayoutState forwarding are frozen.

At implementation, replace only that exact direct expression, update the
channel-slice and first slice/pad/concat boundary contracts, and preserve the
other direct and orchestrated occurrences. Validate the complete indexed owner,
channel-slice, bridge-owner, terminal-affine, core, pass-efficiency,
architecture, and TensorFlow-import-blocked suites sequentially. Commit and
push only; do not create or update a pull request.

## Pre-terminal affine post-ADD evidence implementation checkpoint

Only the first direct indexed affine post-ADD invocation now assigns its
unchanged result to `_pre_terminal_affine_post_add_stats`. It remains between
the channel-slice cluster and
`_pre_terminal_affine_slice_pad_concat_stats`, and continues to forward the
same ModelIR and Session LayoutState.

The two later direct calls and the terminal-slice orchestration occurrence are
unchanged. No tensor-count proxy is needed because the owner prunes only after
a positive rewrite, and no reconciliation branch consumes the staged result.

Focused indexed-owner, channel-slice, bridge-owner, and terminal-affine
coverage is `165 passed in 1.17s`. The sequential indexed-owner,
channel-slice, bridge-owner, terminal-affine, complete SPP,
concat-unary-conv, late-SPP, shape-extract, QKV, split/conv bridge,
Hardswish-SE, late hard-activation, SINet terminal, Expand/Squeeze,
late-layout, constant-fold/cast, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `761 passed in 28.24s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the preceding channel-slice/pad/Mul orchestration cluster.
Determine whether its runner already propagates complete ordered results and
whether any child can prune without a positive counter before adding mutation
evidence. Commit and push only; do not create or update a pull request.

## Pre-terminal channel-slice/pad-Mul evidence characterization checkpoint

The channel-slice/pad-Mul orchestration has two ordered child results. Empty
ModelIR probing confirms three fixed channel-slice counters followed by one
fixed pad-Mul counter. Source and behavioral owner coverage confirm that all
four underlying rewrites prune only after a positive counter, so no net tensor
count is needed.

Strict expected-failure coverage requires the orchestration runner and lowerer
delegate to return the ordered pair, validates an exact four-key summary, and
requires only the direct terminal invocation to stage
`channel_slice_pad_mul_results` and
`_pre_terminal_channel_slice_pad_mul_stats`. The terminal-slice callback,
shared pass-state scope, Session LayoutState, diagnostics, and pass order remain
unchanged.

At implementation, add the pure fixed-schema summary, propagate the raw pair
through the runner and delegate, and replace only the direct expression with
two assignments. Update the pre-add and pre-terminal affine boundary contracts.
Validate malformed length, result order, the complete channel-slice and pad
owners, terminal-slice recovery, core, pass-efficiency, architecture, and
TensorFlow-import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Pre-terminal channel-slice/pad-Mul evidence implementation checkpoint

The channel-slice/pad-Mul runner and lowerer delegate now return the ordered
two-result tuple. The pure `summarize_channel_slice_pad_mul_mutations()` helper
validates the exact length and extracts only the three channel-slice counters
and one pad-Mul counter.

The direct terminal invocation stages `channel_slice_pad_mul_results` followed
by `_pre_terminal_channel_slice_pad_mul_stats`. The terminal-slice callback
continues to ignore the returned pair, shared pass-state and diagnostics remain
unchanged, and no reconciliation decision consumes the summary.

Focused runner, summary, owner, and boundary coverage is `17 passed in 0.66s`;
terminal-slice callback and architecture coverage is `17 passed in 0.71s`.
The first expanded gate exposed one stale boundary expectation (`782 passed, 1
failed`) after the two new assignments. Updating that exact contract produced
`783 passed in 28.91s` across channel-slice, pad, terminal-slice, callback,
affine, bridge, SPP, QKV, hard-activation, SINet, late-layout, shared-context,
core, pass-efficiency, architecture, and TensorFlow-import-blocked coverage.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the preceding `_optimize_transpose_pre_add_nhwc_chains`
owner. Establish its production occurrence count, return schema, and pruning
contract before choosing an unambiguous evidence target. Commit and push only;
do not create or update a pull request.

## Pre-terminal pre-ADD evidence characterization checkpoint

The composite pre-ADD owner has exactly one direct lowerer call. Its indexed
and compatibility rewrites report one aggregate
`optimized_transpose_pre_add_nhwc_chains` counter, but the compatibility owner
unconditionally prunes at exit. A dedicated zero-rewrite test confirms that an
unused tensor can be removed while the returned counter remains zero.

Strict expected-failure coverage therefore requires adjacent
`pre_terminal_pre_add_tensor_count` and `_pre_terminal_pre_add_stats`
assignments. The summary must preserve the raw counter and add clamped net
`pruned_unused_tensors`. Terminal affine recovery and
`channel_slice_pad_mul_results` are fixed outer boundaries; ModelIR and Session
LayoutState forwarding remain unchanged.

At implementation, replace the unique direct expression with count plus merged
stats assignments. Update the channel-slice and downstream bridge boundary
contracts, but leave the three orchestration-owned occurrences unchanged.
Validate zero-rewrite pruning, the complete indexed/compatibility owner,
channel-slice, terminal-affine, core, pass-efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Pre-terminal pre-ADD evidence implementation checkpoint

The unique direct pre-ADD invocation now records
`pre_terminal_pre_add_tensor_count` and assigns `_pre_terminal_pre_add_stats`
from the owner's unchanged aggregate counter plus clamped net
`pruned_unused_tensors`. This captures the compatibility owner's unconditional
cleanup when its rewrite counter is zero.

The three orchestration-owned occurrences remain unchanged. The first focused
run exposed two stale terminal-affine boundary expectations (`135 passed, 2
failed`) that resolved the immediately following `len` call instead of the
logical owner inside the merged dictionary. Updating only those contracts
produced `137 passed in 2.77s`.

The sequential indexed/compatibility pre-ADD, channel-slice, pad,
terminal-slice, callback, affine, bridge, SPP, QKV, hard-activation, SINet,
late-layout, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `806 passed in 29.07s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, inspect the first direct
`_run_terminal_affine_concat_split_recovery_sequence()` immediately before the
pre-ADD count. The second occurrence already has complete ordered-result and
net-prune evidence; characterize a distinct target for the first occurrence
without changing the shared runner or summary. Commit and push only; do not
create or update a pull request.

## First terminal affine mutation evidence characterization checkpoint

The first terminal affine recovery uses the same ordered eleven-result runner
and fixed twelve-key plus net-prune summary already validated for the second
occurrence. No new summary or child-owner contract is required.

Strict expected-failure coverage requires adjacent
`pre_terminal_affine_tensor_count`, `pre_terminal_affine_results`, and
`_pre_terminal_affine_stats` assignments between the final InstanceNorm
dual-statistics rewrite and `pre_terminal_pre_add_tensor_count`. The second
occurrence must retain its independent `terminal_affine_*` targets.

At implementation, replace only the first recovery expression with the three
assignments and reuse `summarize_terminal_affine_concat_split_mutations()` with
a clamped tensor-count delta. Update both terminal-affine boundary contracts.
Validate both occurrence targets, all eleven child results, pre-ADD,
channel-slice, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## First terminal affine mutation evidence implementation checkpoint

The first terminal affine recovery now stages
`pre_terminal_affine_tensor_count`, `pre_terminal_affine_results`, and
`_pre_terminal_affine_stats`. It reuses the existing exact eleven-result,
twelve-key summary and records its own clamped net tensor reduction.

The independently staged second recovery retains `terminal_affine_tensor_count`,
`terminal_affine_results`, and `_terminal_affine_stats`. Neither summary is
consumed by reconciliation, and child order, ModelIR/LayoutState forwarding,
and recovery semantics are unchanged.

Focused terminal-affine, pre-ADD, channel-slice, and architecture coverage is
`45 passed in 2.55s`. The sequential indexed/compatibility pre-ADD,
channel-slice, pad, terminal-slice, callback, affine, bridge, SPP, QKV,
hard-activation, SINet, late-layout, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `807 passed in 28.59s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately preceding
`_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains`
owner. Confirm its occurrence count, fixed result schema, and whether pruning
can occur with a zero counter before staging evidence. Commit and push only; do
not create or update a pull request.

## Pre-terminal affine InstanceNorm dual-statistics characterization checkpoint

The indexed dual-statistics InstanceNorm residual/add/resize owner has four
production occurrences, three of them direct top-level calls. Its one fixed
counter is complete mutation evidence because unused-tensor pruning occurs only
after a positive rewrite; an explicit prune-hook test freezes that behavior.

Strict expected-failure coverage selects only the last direct call, between the
terminal residual/Mul/Concat/Conv owner and
`pre_terminal_affine_tensor_count`, and requires
`_pre_terminal_affine_instancenorm_dualstats_stats`. The two earlier direct
calls and nested indexed-convergence occurrence remain unchanged.

At implementation, replace only that exact direct expression, update the first
terminal-affine boundary contracts, and preserve ModelIR and Session LayoutState
forwarding. Validate the complete indexed dual-statistics owner,
terminal-affine, pre-ADD, channel-slice, core, pass-efficiency, architecture,
and TensorFlow import blocking sequentially. Commit and push only; do not create
or update a pull request.

## Pre-terminal affine InstanceNorm dual-statistics implementation checkpoint

Only the last direct dual-statistics InstanceNorm residual/add/resize call now
assigns its unchanged result to
`_pre_terminal_affine_instancenorm_dualstats_stats`. The other two direct calls
and the nested indexed-convergence occurrence retain their previous forms.

The staged counter is complete because pruning remains positive-only. It is
not consumed by reconciliation, and ModelIR/LayoutState forwarding, owner
guards, and first terminal-affine ordering are unchanged.

Focused complete indexed-owner, terminal-affine, and architecture coverage is
`218 passed in 1.34s`. The sequential dual-statistics, indexed/compatibility
pre-ADD, channel-slice, pad, terminal-slice, callback, affine, bridge, SPP,
QKV, hard-activation, SINet, late-layout, shared-context, core,
pass-efficiency, architecture, and TensorFlow-import-blocked gate is `1014
passed in 29.74s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, inspect the immediately preceding
`_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains` owner.
Confirm all occurrence forms and its pruning contract before selecting only the
last direct call for evidence. Commit and push only; do not create or update a
pull request.

## Pre-terminal affine InstanceNorm residual/Mul/Concat characterization checkpoint

The indexed InstanceNorm residual/Mul/Concat/Conv owner has four production
occurrences, three as direct top-level calls. It returns one fixed rewrite
counter and prunes only after a positive rewrite; an explicit prune-hook test
freezes the completeness of that counter.

Strict expected-failure coverage selects only the last direct call between the
terminal InstanceNorm post-bias owner and the staged dual-statistics result. It
requires `_pre_terminal_affine_instancenorm_residual_mul_concat_stats`; the two
earlier direct calls and nested occurrence remain unchanged.

At implementation, replace only that exact expression and update the adjacent
dual-statistics and first terminal-affine boundary contracts. Validate the
complete indexed residual/Mul/Concat owner, dual-statistics, terminal-affine,
pre-ADD, core, pass-efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Pre-terminal affine InstanceNorm residual/Mul/Concat implementation checkpoint

Only the last direct InstanceNorm residual/Mul/Concat/Conv call now assigns its
unchanged result to
`_pre_terminal_affine_instancenorm_residual_mul_concat_stats`. The two earlier
direct calls and nested indexed-convergence occurrence remain unchanged.

The staged counter is complete because pruning remains positive-only. It is
observation-only, and the following staged dual-statistics owner, first
terminal-affine recovery, and all reconciliation decisions retain their order
and behavior.

Focused complete residual/Mul/Concat, dual-statistics, terminal-affine, and
architecture coverage is `321 passed in 1.59s`. The sequential
residual/Mul/Concat, dual-statistics, indexed/compatibility pre-ADD,
channel-slice, pad, terminal-slice, callback, affine, bridge, SPP, QKV,
hard-activation, SINet, late-layout, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `1117 passed in 29.74s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately preceding
`_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains` owner.
Confirm its occurrence forms, fixed result schema, and prune behavior before
selecting the last direct call for evidence. Commit and push only; do not create
or update a pull request.

## Pre-terminal affine InstanceNorm post-bias characterization checkpoint

The indexed InstanceNorm post-transpose bias/add owner has five production
occurrences: one nested call and four direct top-level calls. Its one fixed
rewrite counter is complete because pruning occurs only after a positive
rewrite; a prune-hook test freezes that behavior.

Strict expected-failure coverage selects the third direct call, between late
binary-layout recovery and the staged residual/Mul/Concat result, and requires
`_pre_terminal_affine_instancenorm_post_bias_stats`. The first two direct calls,
nested occurrence, and absolute-final fourth direct call remain unchanged.

At implementation, replace only the selected expression and update the
residual/Mul/Concat boundary contract. Validate the complete indexed post-bias,
residual/Mul/Concat, dual-statistics, terminal-affine, core, pass-efficiency,
architecture, absolute-final normalization, and TensorFlow import-blocking
suites sequentially. Commit and push only; do not create or update a pull
request.

## Pre-terminal affine InstanceNorm post-bias implementation checkpoint

Only the third direct InstanceNorm post-transpose bias/add call now assigns its
unchanged result to `_pre_terminal_affine_instancenorm_post_bias_stats`. The
nested occurrence, first two direct calls, and absolute-final fourth direct call
retain their existing forms.

The counter is complete because pruning remains positive-only. The result is
not consumed by reconciliation, and the following staged residual/Mul/Concat,
dual-statistics, and first terminal-affine owners retain their exact order.

Focused post-bias, residual/Mul/Concat, dual-statistics, terminal-affine,
absolute-final, and architecture coverage is `399 passed in 1.88s`. The
sequential InstanceNorm owner, absolute-final, pre-ADD, channel-slice, pad,
terminal-slice, callback, affine, bridge, SPP, QKV, hard-activation, SINet,
late-layout, shared-context, core, pass-efficiency, architecture, and
TensorFlow-import-blocked gate is `1195 passed in 29.59s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, audit the preceding `late_binary_layout_recovery_stats` interval.
Confirm that its existing mutation counter and conditional reconciliation form
a complete boundary for this newly staged terminal chain. If it does, move to
the absolute-final fourth post-bias call rather than duplicating already staged
evidence. Commit and push only; do not create or update a pull request.

## Late-binary to terminal evidence boundary audit checkpoint

The existing late-binary runner already returns nine fixed mutation counters
plus clamped `pruned_unused_tensors`, while excluding the layout owner's
non-mutating `iterations`. Existing zero-counter pruning coverage confirms the
net tensor delta is required and present.

Production stores that aggregate in `late_binary_layout_recovery_stats` and
reconciles whenever `_stats_have_positive_count()` sees any mutation. A new
architecture contract freezes the direct transition from this guarded branch
to `_pre_terminal_affine_instancenorm_post_bias_stats`. No additional summary
or staging is needed at this boundary.

Focused late-binary runner and boundary coverage passes. At resume, move to the
absolute-final fourth InstanceNorm post-bias call after final signature
sanitization. Characterize its distinct surrounding affine and normalization
owners before capturing its result. Commit and push only; do not create or
update a pull request.

## Absolute-final InstanceNorm post-bias characterization checkpoint

The fourth direct InstanceNorm post-transpose bias/add call is a distinct
absolute-final occurrence after boundary-signature sanitization. Its fixed raw
counter remains complete under the already frozen positive-only pruning
contract.

Strict expected-failure coverage requires
`_absolute_final_instancenorm_post_bias_stats` between the absolute-final affine
post-ADD owner and normalization/attention pair. It also freezes four direct
occurrences, with only the third pre-terminal call already staged and the first
two unchanged.

At implementation, replace only the fourth expression and update the
normalization/attention outer-boundary contract. Validate all post-bias owner
tests, occurrence targets, affine post-ADD, absolute-final normalization,
architecture, core, pass-efficiency, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Absolute-final InstanceNorm post-bias implementation checkpoint

Only the fourth direct InstanceNorm post-transpose bias/add call now assigns
its unchanged result to `_absolute_final_instancenorm_post_bias_stats`. The
first two direct calls remain expressions and the third retains its distinct
`_pre_terminal_affine_instancenorm_post_bias_stats` target.

The staged raw counter is complete because the owner prunes only after a
positive rewrite. It remains observation-only: no new reconciliation decision,
tensor-count proxy, pass invocation, graph scan, or artifact-producing action
was added. The absolute-final affine post-ADD owner still runs immediately
before it, and the normalization/attention pair still runs immediately after
it with the same ModelIR and Session LayoutState.

The first focused run produced `79 passed, 1 failed`; the failure was a stale
occurrence-target expectation that still required the fourth call to be an
expression. After updating that contract, focused coverage produced `80
passed`. The first expanded sequential gate produced `1199 passed, 1 failed`;
the remaining failure was a stale architecture boundary expectation. After
correcting that exact boundary, focused coverage produced `9 passed`, and the
expanded sequential gate completed with `1200 passed`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, characterize the immediately preceding absolute-final
`_optimize_transpose_mul_posttranspose_add_nhwc_chains` occurrence. Keep it
distinct from the already staged pre-terminal affine post-ADD occurrence, and
confirm its positive-only pruning contract before considering an
`_absolute_final_affine_post_add_stats` target. Commit and push only; do not
create or update a pull request.

## Absolute-final affine post-ADD characterization checkpoint

The indexed affine post-ADD owner has three direct top-level production calls.
The first already stages `_pre_terminal_affine_post_add_stats`, the second
remains a very-late expression after unbound-input repair, and the third is the
absolute-final call after boundary-signature realignment and sanitization.

The owner returns one fixed rewrite counter and invokes unused-tensor pruning
only when that counter is positive. Existing prune-hook coverage freezes this
contract, so the raw result is complete mutation evidence and needs neither a
tensor-count proxy nor a summary adapter.

Strict expected-failure coverage selects only the third call and requires an
`_absolute_final_affine_post_add_stats` target immediately before
`_absolute_final_instancenorm_post_bias_stats`. It freezes the first staged
target and requires the second call to remain an expression. Focused
absolute-final, indexed-owner, and pre-terminal boundary coverage is `64
passed, 92 deselected, 1 xfailed`.

At implementation, replace only the selected third expression and update the
post-bias and architecture boundary contracts. Preserve the second very-late
expression, ModelIR/LayoutState forwarding, and all pass order. Validate the
indexed affine owner, absolute-final normalization, pre-terminal affine
boundary, core, pass-efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Absolute-final affine post-ADD implementation checkpoint

Only the third direct affine post-ADD call now assigns its unchanged result to
`_absolute_final_affine_post_add_stats`. The first call retains the existing
`_pre_terminal_affine_post_add_stats` target, and the second very-late call
remains an expression.

The staged raw counter is complete under the owner's positive-only pruning
contract. It is observation-only and does not feed reconciliation. Boundary
signature realignment and sanitization still precede it, and
`_absolute_final_instancenorm_post_bias_stats` still follows immediately with
the same ModelIR and Session LayoutState.

Focused absolute-final, indexed-owner, pre-terminal, and architecture coverage
is `71 passed, 344 deselected`. The expanded sequential indexed-owner,
late/terminal orchestration, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `1201 passed in 29.71s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the remaining second direct affine post-ADD call after
unbound-input repair and immediately before the very-late Gather/Constant
normalization cluster. Characterize its exact boundaries independently before
considering a `_very_late_affine_post_add_stats` target. Commit and push only;
do not create or update a pull request.

## Very-late affine post-ADD characterization checkpoint

The remaining second direct affine post-ADD call is isolated between
`_repair_unbound_nonconstant_operator_inputs_with_layout_transpose` and the
very-late Gather/Constant normalization cluster. It handles strict affine
fragments recreated specifically by the unbound-input repair.

The same indexed owner returns one fixed counter and prunes only after a
positive rewrite, so its raw result is complete without a tensor-count proxy.
Strict expected-failure coverage requires
`_very_late_affine_post_add_stats`, preserves the first
`_pre_terminal_affine_post_add_stats` target and the third
`_absolute_final_affine_post_add_stats` target, and freezes both adjacent
owners. Focused coverage is `72 passed, 92 deselected, 1 xfailed`.

At implementation, replace only the second direct expression and update the
very-late orchestration and architecture boundary contracts. Preserve both
other staged targets, ModelIR/LayoutState forwarding, and pass order. Validate
the indexed owner, all three occurrence contracts, very-late and absolute-final
orchestration, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Very-late affine post-ADD implementation checkpoint

The second direct affine post-ADD call now assigns its unchanged result to
`_very_late_affine_post_add_stats`. Together with
`_pre_terminal_affine_post_add_stats` and
`_absolute_final_affine_post_add_stats`, every direct occurrence has a distinct
observation point.

The result remains unused by reconciliation. Unbound-input repair still runs
immediately before it, the very-late Gather/Constant normalization cluster
still runs immediately after it, and ModelIR/LayoutState forwarding and pass
order are unchanged.

Focused three-occurrence, very-late, absolute-final, indexed-owner, and
architecture coverage is `80 passed, 343 deselected`. The expanded sequential
gate, now explicitly including the complete very-late orchestration suite, is
`1209 passed in 30.03s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, audit result propagation from the immediately following very-late
Gather/Constant normalization cluster. Determine whether its child owners
already expose complete mutation evidence and whether any cleanup-only path
requires net tensor-count accounting before adding a new observation point.
Commit and push only; do not create or update a pull request.

## Very-late normalization mutation characterization checkpoint

The very-late Gather/Constant normalization runner builds four ordered
invocations but currently discards the tuple returned by
`run_recovery_invocations()`. The child schemas comprise one Gather-axis key,
three constant-input fold keys, two redundant-Cast keys, and two
normalization-Pad keys.

The flatten global-normalization Pad owner unconditionally prunes unused
tensors even when its rewrite count is zero. A prune-hook contract now freezes
that cleanup-only path. Therefore a complete cluster summary requires all eight
fixed mutation keys plus a clamped net tensor reduction measured around the
whole cluster.

Three strict expected-failure contracts require the runner's ordered tuple, a
fixed-schema
`summarize_very_late_gather_constant_normalization_mutations()` helper, and
lowerer assignments for `very_late_normalization_tensor_count`,
`very_late_normalization_results`, and `_very_late_normalization_stats` before
the final dynamic-Reshape resolution. Focused characterization is `9 passed, 3
xfailed`.

At implementation, return the existing ordered tuple without another pass,
validate exactly four results in the pure summary, and measure only the
cluster-wide tensor delta in the lowerer. Keep the summary observation-only;
do not add reconciliation or alter the shared pass-state scope. Validate
very-late orchestration, constant-fold/Cast, normalization-Pad, shared-context,
core, pass-efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Very-late normalization mutation implementation checkpoint

`run_very_late_gather_constant_normalization()` now returns the existing
ordered four-result tuple from `run_recovery_invocations()`. It does not rebuild
the invocation list or rerun an owner.

The pure
`summarize_very_late_gather_constant_normalization_mutations()` helper requires
exactly four results, extracts only the eight declared mutation keys, defaults
missing keys to zero, and adds clamped `pruned_unused_tensors`. Wrong result
counts raise `ValueError`.

The lowerer records `very_late_normalization_tensor_count`, stages
`very_late_normalization_results`, and builds `_very_late_normalization_stats`
from the cluster-wide tensor delta. The summary is observation-only. The same
shared `ModelIRPassStateScope`, child order, ModelIR/LayoutState/diagnostics
objects, and following dynamic-Reshape resolution are preserved.

Focused runner, summary, cleanup-only pruning, lowerer, architecture, and
pass-state coverage is `17 passed`. The expanded sequential indexed-owner,
late/terminal orchestration, shared-context, core, pass-efficiency,
architecture, and TensorFlow-import-blocked gate is `1216 passed in 30.03s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_resolve_dynamic_reshape_shapes(...,
prefer_runtime_inferable_from_onnx_raw=True)` owner. Confirm its fixed result
schema, occurrence count, and cleanup behavior before selecting this very-late
call for an independent observation point. Commit and push only; do not create
or update a pull request.

## Very-late dynamic-Reshape characterization checkpoint

The dynamic-Reshape resolver has four lowerer occurrences. Two are already
captured inside indexed convergence helpers. Two are direct top-level calls:
an earlier core-cleanup expression and the distinct very-late call with
`prefer_runtime_inferable_from_onnx_raw=True` immediately after
`_very_late_normalization_stats`.

The owner returns one fixed `resolved_dynamic_reshape_shapes` counter. Source
and behavior coverage confirm that it updates RESHAPE options, the optional
shape tensor, and output shape metadata, counts each changed operator, and does
not prune tensors or remove operators. Its raw result is therefore complete
mutation evidence.

Strict expected-failure coverage requires only the very-late direct call to
assign `_very_late_dynamic_reshape_stats`, preserves the earlier direct
expression, and freezes the following indexed Conv-input adapter boundary.
Focused very-late and dynamic-Reshape coverage is `28 passed, 1 xfailed`.

At implementation, replace only that selected expression and update the
very-late and architecture boundary contracts. Do not add a tensor-count
proxy, summary adapter, or reconciliation consumer. Validate dynamic-Reshape,
very-late normalization, indexed convergence, core, pass-efficiency,
architecture, and TensorFlow import blocking sequentially. Commit and push
only; do not create or update a pull request.

## Very-late dynamic-Reshape implementation checkpoint

Only the direct dynamic-Reshape call with
`prefer_runtime_inferable_from_onnx_raw=True` now assigns its unchanged result
to `_very_late_dynamic_reshape_stats`. The earlier direct core-cleanup call
remains an expression, and the two indexed convergence helpers continue to
consume their local `reshape_stats` results.

The staged single counter is complete because the owner performs no pruning or
topology removal. It remains observation-only immediately after
`_very_late_normalization_stats` and immediately before the indexed Conv-input
adapter repair. No summary, tensor-count proxy, or reconciliation branch was
added.

Focused very-late, dynamic-Reshape, indexed-convergence, and architecture
coverage is `44 passed, 256 deselected`. The expanded sequential gate,
explicitly including complete dynamic-Reshape and indexed-final-convergence
coverage, is `1243 passed in 30.55s`. Ruff, Python bytecode compilation, and
whitespace validation pass.

At resume, inspect the immediately following
`_run_indexed_conv_input_adapter_repairs(model_ir)` call. Compare it with the
fallback-path `fallback_conv_input_stats` occurrence, confirm its result schema
and pruning behavior, and characterize only the direct very-late call if its
raw result is complete. Commit and push only; do not create or update a pull
request.

## Very-late indexed Conv-input repair characterization checkpoint

The indexed Conv-input adapter runner returns two fixed counters for singleton
Reshape and stale Transpose repairs. Both child owners unconditionally invoke
unused-tensor pruning, including zero-rewrite calls. A prune-hook contract now
freezes the two cleanup opportunities, so the raw counters alone are not
complete mutation evidence.

Strict expected-failure coverage requires
`very_late_conv_input_tensor_count` followed by
`_very_late_conv_input_stats`, which spreads the unchanged runner result and
adds clamped `pruned_unused_tensors`. It selects only the direct call after
`_very_late_dynamic_reshape_stats`; the fallback-path
`fallback_conv_input_stats` assignment remains unchanged.

Focused indexed-owner and very-late coverage is `25 passed, 1 xfailed`. The
focused run also exposed a stale test assumption that legacy lowerer map-builder
attributes still existed. The sentinel monkeypatch now uses `raising=False`,
which preserves detection if either legacy builder is reintroduced and called.

At implementation, measure the tensor delta only around the direct call and
build the fixed dictionary in place. Do not alter the runner, fallback path,
following stale-channel-shuffle owner, or reconciliation decisions. Validate
indexed Conv-input repairs, very-late orchestration, dynamic-Reshape, core,
pass-efficiency, architecture, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Very-late indexed Conv-input repair implementation checkpoint

The direct call now records `very_late_conv_input_tensor_count` and builds
`_very_late_conv_input_stats` by spreading the runner's unchanged two counters
and adding clamped `pruned_unused_tensors`.

This captures cleanup-only pruning from both child owners without rerunning
either repair. The fallback-path `fallback_conv_input_stats` assignment and its
existing reconciliation decision are unchanged. The direct summary remains
observation-only between `_very_late_dynamic_reshape_stats` and the stale NCHW
channel-shuffle repair.

Focused indexed-owner, legacy-compatibility, dynamic-Reshape, very-late, and
architecture coverage is `42 passed, 255 deselected`. The expanded sequential
gate, explicitly including indexed Conv-input and legacy Conv-layout suites, is
`1265 passed in 30.75s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, inspect the immediately following
`run_stale_nchw_channel_shuffle_repair()` owner. Confirm its result schema,
pruning behavior, and production occurrence count before selecting the
very-late call for an independent observation point. Commit and push only; do
not create or update a pull request.

## Very-late stale channel-shuffle characterization checkpoint

`run_stale_nchw_channel_shuffle_repair()` has one production occurrence. Its
single `repaired_nchw_channel_shuffle_concat_gathers` counter covers all
changes to the Concat axis and related tensor metadata. The owner performs no
pruning and removes no operators, so the raw result is complete mutation
evidence.

Strict expected-failure coverage requires
`_very_late_stale_channel_shuffle_stats` immediately after
`_very_late_conv_input_stats`, preserves ModelIR/LayoutState/diagnostics
forwarding, and freezes the following NCHW Concat/Transpose/Conv axis repair.
Focused shuffle-owner and very-late coverage is `21 passed, 1 xfailed`.

At implementation, replace only the direct expression with the assignment. Do
not add a tensor-count proxy, summary adapter, reconciliation consumer, or
second pass invocation. Validate shuffle layout, very-late orchestration, core,
pass-efficiency, architecture, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Very-late stale channel-shuffle implementation checkpoint

The sole production call now assigns its unchanged result to
`_very_late_stale_channel_shuffle_stats`. Its single counter remains complete
because the owner neither prunes tensors nor changes topology.

The result is observation-only immediately after
`_very_late_conv_input_stats`. The following NCHW
Concat/Transpose/Conv-axis repair, argument forwarding, diagnostics, and pass
count are unchanged.

Focused shuffle-owner, very-late, and architecture coverage is `27 passed, 253
deselected`. The expanded sequential gate, explicitly including the complete
shuffle-layout suite, is `1270 passed in 30.20s`. Ruff, Python bytecode
compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_repair_nchw_concat_transpose_conv_axes(model_ir)` owner. Confirm its result
schema, occurrence count, and pruning behavior before selecting the very-late
call for an observation point. Commit and push only; do not create or update a
pull request.

## Very-late Concat/Transpose/Conv-axis characterization checkpoint

The NCHW Concat/Transpose/Conv-axis owner has three production occurrences: a
very-late direct expression, `fallback_concat_axis_stats`, and
`final_concat_axis_stats`. Its single
`repaired_nchw_concat_transpose_conv_axes` counter covers every applied repair
plan. The owner changes only Concat options and tensor metadata and performs no
pruning or topology mutation.

Strict expected-failure coverage selects only the direct occurrence for
`_very_late_concat_transpose_conv_axis_stats`, preserves Session LayoutState
forwarding, and freezes the following Concat/global-pool/Conv-axis owner. It
also requires the fallback and final targets to remain present. Focused indexed
owner, legacy Conv-layout, and very-late coverage is `58 passed, 1 xfailed`.

At implementation, replace only the direct expression with the assignment. Do
not add a proxy, summary, reconciliation consumer, or another owner invocation.
Validate indexed Concat/Transpose/Conv layout, legacy Conv layout, very-late
orchestration, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Very-late Concat/Transpose/Conv-axis implementation checkpoint

Only the direct very-late occurrence now assigns its unchanged result to
`_very_late_concat_transpose_conv_axis_stats`. The fallback and final
assignments retain their existing targets and reconciliation decisions.

The single raw counter is complete because the owner neither prunes tensors nor
changes topology. The result remains observation-only between
`_very_late_stale_channel_shuffle_stats` and the NCHW
Concat/global-pool/Conv-axis repair.

Focused indexed-owner, legacy Conv-layout, very-late, and architecture coverage
is `63 passed, 254 deselected`. The expanded sequential gate, explicitly
including the indexed Concat/Transpose/Conv suite, is `1299 passed in 30.62s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_repair_nchw_concat_global_pool_conv_axes(model_ir)` owner. Confirm its result
schema, production occurrence count, and pruning behavior before selecting the
very-late direct call. Commit and push only; do not create or update a pull
request.

## Very-late Concat/global-pool/Conv-axis characterization checkpoint

The NCHW Concat/global-pool/Conv-axis owner has one production occurrence. Its
single `repaired_nchw_concat_global_pool_conv_axes` counter covers every
applied plan. The owner changes only Concat options and tensor metadata and
performs no pruning or topology mutation.

Strict expected-failure coverage requires
`_very_late_concat_global_pool_conv_axis_stats` immediately after
`_very_late_concat_transpose_conv_axis_stats`, preserves Session LayoutState
forwarding, and freezes the following dynamic rank-one
Unsqueeze/Reshape-shape rewrite. Focused indexed owner, legacy Conv-layout, and
very-late coverage is `64 passed, 1 xfailed`.

At implementation, replace only the direct expression with the assignment. Do
not add a proxy, summary, reconciliation consumer, or another owner invocation.
Validate indexed Concat/global-pool layout, legacy Conv layout, very-late
orchestration, core, pass-efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Very-late Concat/global-pool/Conv-axis implementation checkpoint

The sole production call now assigns its unchanged result to
`_very_late_concat_global_pool_conv_axis_stats`. Its single raw counter remains
complete because the owner neither prunes tensors nor changes topology.

The result is observation-only immediately after
`_very_late_concat_transpose_conv_axis_stats` and before the dynamic rank-one
Unsqueeze/Reshape-shape rewrite. ModelIR/LayoutState forwarding and pass count
are unchanged.

Focused indexed-owner, legacy Conv-layout, very-late, and architecture coverage
is `69 passed, 254 deselected`. The expanded sequential gate, explicitly
including indexed Concat/global-pool coverage, is `1333 passed in 31.23s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following
`_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs()` owner. Confirm its
result schema, occurrence forms, and cleanup behavior before selecting this
very-late call for a distinct observation point. Commit and push only; do not
create or update a pull request.

## Very-late dynamic rank-one Reshape characterization checkpoint

The dynamic rank-one Unsqueeze/Reshape-shape owner has three production
occurrences: the current very-late direct call, a fallback expression, and an
absolute-final expression. Its single
`rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs` counter covers both
the indexed runtime SHAPE/Concat insertion path and the metadata-only
higher-rank fallback. The owner performs no pruning.

Strict expected-failure coverage selects only the first direct call for
`_very_late_dynamic_rank1_reshape_stats`, preserves Session LayoutState
forwarding, and freezes the following static-shape reconciliation. It also
requires the fallback and absolute-final calls to remain expressions with their
existing target ModelIRs. Focused dynamic-Reshape and very-late coverage is `33
passed, 1 xfailed`.

At implementation, replace only the selected direct expression with the
assignment. Do not add a proxy, summary, reconciliation consumer, or another
owner invocation. Validate dynamic-Reshape, very-late orchestration, fallback
and absolute-final occurrence contracts, core, pass-efficiency, architecture,
and TensorFlow import blocking sequentially. Commit and push only; do not
create or update a pull request.

## Very-late dynamic rank-one Reshape implementation checkpoint

Only the first very-late dynamic rank-one Unsqueeze/Reshape-shape call now
assigns its unchanged result to `_very_late_dynamic_rank1_reshape_stats`. The
fallback and absolute-final calls remain expressions targeting `fallback_ir`
and `model_ir`, respectively.

The raw counter is complete because it covers both operator/tensor insertion
and metadata-only rewrites and the owner performs no pruning. The result remains
observation-only before the unchanged static-shape reconciliation. The
occurrence contract compares the two remaining expression targets without
depending on unspecified `ast.walk()` traversal order.

Focused dynamic-Reshape, very-late, absolute-final, and architecture coverage
is `51 passed, 250 deselected`. The expanded sequential gate is `1334 passed in
30.38s`. Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, inspect the immediately following direct
`_reconcile_static_tensor_shapes(model_ir)` occurrence. Distinguish it from the
many conditional and convergence-owned reconciliations, confirm its result
schema and mutation completeness, and characterize only this very-late
boundary. Commit and push only; do not create or update a pull request.

## Very-late static reconciliation characterization checkpoint

The legacy `reconciled_static_tensor_shapes` key counts updates applied through
the output-tensor shape helper. It is not complete mutation evidence: a focused
RESHAPE fixture begins with correct output metadata but a stale `newShape`
option, and reconciliation corrects that option while returning the unchanged
legacy value `0`.

To preserve every existing caller and exact-dictionary contract, strict
expected-failure coverage requires an opt-in `include_mutation_count=True`
parameter. Only that mode adds `reconciled_static_shape_mutations`, covering
output shapes, constant shape parameters, operator options, and direct tensor
metadata updates. Default calls must continue returning only the legacy key.

A second strict expected-failure contract selects only the direct
reconciliation after `_very_late_dynamic_rank1_reshape_stats`, requires
`_very_late_static_shape_stats`, and freezes the following
`split_fallback_stats` assignment. Focused characterization is `24 passed, 2
xfailed`.

At implementation, instrument mutation sites during the existing fixed-point
walk; do not add a pre/post ModelIR fingerprint or another graph traversal.
Return the extra key only when requested, and enable it only at the selected
very-late call. Validate parameter-only mutation, all static-shape
reconciliation suites, dynamic-Reshape, very-late orchestration, core,
pass-efficiency, architecture, and TensorFlow import blocking sequentially.
Commit and push only; do not create or update a pull request.

## Very-late static reconciliation implementation checkpoint

`reconcile_static_tensor_shapes()` now accepts
`include_mutation_count=False`. The default path preserves the exact legacy
dictionary. Opt-in mode additionally returns
`reconciled_static_shape_mutations`.

Mutation accounting is embedded in the existing fixed-point walk. Small local
setters count output shape updates, operator-option changes, constant
shape-parameter writes, direct shape-signature updates, and constant-vector
metadata changes. No ModelIR fingerprint, copy, pre/post graph scan, or new
dependency is introduced.

Only the direct reconciliation after
`_very_late_dynamic_rank1_reshape_stats` enables the option and stages
`_very_late_static_shape_stats`. Other callers, including indexed convergence
and fallback/final paths, retain the legacy one-key schema. The focused
reconciliation/very-late gate is `47 passed, 250 deselected`; the complete
static reconciliation, convergence, core, architecture, and import-boundary
gate is `428 passed in 27.50s`; and the expanded sequential gate is `1339
passed in 30.54s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the immediately following `split_fallback_stats` owner and its
conditional reconciliation. Confirm whether its raw rewrite counter covers any
cleanup and whether the existing positive guard is complete before changing
that boundary. Commit and push only; do not create or update a pull request.

## Post-Split fallback reconciliation characterization checkpoint

`replace_unsupported_split_with_slice()` has one complete mutation predicate:
`replaced_unsupported_split_with_slice`. The owner creates tensors and replaces
operators only when that counter increments, and both tensor pruning and
`LayoutState` synchronization are nested under `rewritten > 0`. A new no-op
fixture proves that a zero result invokes neither maintenance path; a positive
fixture proves that both paths execute exactly once. The lowerer's existing
positive guard is therefore sound and must not be broadened or removed.

The remaining evidence gap is the result of the guarded
`_reconcile_static_tensor_shapes(model_ir)` call. A strict expected-failure
architecture contract requires a stable two-key zero default and, on a
positive Split rewrite, an assigned opt-in complete reconciliation result.
The assignment must use `_post_split_fallback_static_shape_stats`, preserve the
existing guard expression and order, and request `include_mutation_count=True`.

Focused owner/orchestration characterization is `6 passed, 278 deselected, 1
xfailed`. The broader sequential Split, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `432
passed, 1 xfailed in 26.37s`. Ruff and whitespace validation pass.

At implementation, change only that result plumbing. Do not change the Split
matcher, rewrite, raw result schema, pruning, GraphIndex updates, LayoutState
sync, or conditional execution. Run Split fallback, static reconciliation,
very-late orchestration, core, pass-efficiency, architecture, and TensorFlow
import-blocking tests sequentially. Commit and push only; do not create or
update a pull request.

## Post-Split fallback reconciliation implementation checkpoint

The Split owner, its exact one-key result, and the existing positive guard are
unchanged. The lowerer now initializes
`_post_split_fallback_static_shape_stats` with both complete reconciliation
keys set to zero. Only a positive
`replaced_unsupported_split_with_slice` value replaces that default by calling
`_reconcile_static_tensor_shapes(model_ir, include_mutation_count=True)`.

This is observation-only result plumbing. It adds no graph scan, fingerprint,
copy, dependency, rewrite, pruning, layout synchronization, or unconditional
maintenance work. The strict expected-failure contract is now green while the
zero/positive Split owner fixtures continue to prove that the raw rewrite
counter fully covers its maintenance paths.

Focused Split/orchestration/architecture validation is `7 passed, 278
deselected`. The broader sequential Split, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `433
passed in 26.91s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the safety-fallback boundary beginning with
`_find_unbound_nonconstant_operator_inputs(model_ir)`. Characterize the
fallback-only reconciliation results and recursive relowering boundaries
before changing them; do not alter fallback eligibility or broaden inference
validation. Commit and push coherent units only, and do not create or update a
pull request.

## Safety-fallback norm evidence characterization checkpoint

The first fallback-only owner after recursive relowering is the norm-only
`run_pad_layout_cleanup()` call. Its child
`_optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains()` unconditionally
prunes unused tensors after its fixed-point loop. A focused fixture proves that
an empty rewrite can remove an unused constant while returning
`optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains: 0`; the raw result
is therefore incomplete mutation evidence.

A strict expected-failure lowerer contract requires
`fallback_norm_tensor_count` immediately before the existing call and a
`pruned_unused_tensors` clamped net delta merged into `fallback_norm_stats`.
The existing norm-rewrite guard, its three child owners, reconciliation,
topological sort, recursive relowering arguments, and all later fallback work
remain unchanged.

Focused safety-fallback characterization is `1 passed, 1 xfailed`. The broader
sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `445
passed, 1 xfailed in 27.33s`. Ruff and whitespace validation pass.

At implementation, add only this observation point. Do not use the new cleanup
counter to broaden the current reconciliation guard: pruning unused tensors
alone does not require shape reconciliation. Validate the direct norm owner,
Pad orchestration, safety-fallback AST contract, pass efficiency, architecture,
core, and TensorFlow import blocking sequentially. Commit and push only; do not
create or update a pull request.

## Safety-fallback norm evidence implementation checkpoint

The norm-only fallback call now samples `fallback_norm_tensor_count` and
merges a clamped `pruned_unused_tensors` delta into the unchanged owner result.
This makes `fallback_norm_stats` complete for both rewrite and cleanup-only
ModelIR mutations without rerunning the owner or scanning the graph.

The following guard still reads only
`optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains`. Consequently a
prune-only result remains observation-only and does not start binary repairs,
singleton/Reshape cleanup, shape reconciliation, or topological sorting. Pass
order, recursive relowering, diagnostics, and all later fallback work are
unchanged.

Focused safety-fallback and singleton/Reshape orchestration coverage is `13
passed`. The broader sequential fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`446 passed in 27.35s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, inspect the immediately following fallback dynamic rank-one
Unsqueeze/Reshape-shape call. Characterize its result and the unconditional
topological/layout refresh separately; do not skip either refresh until the
recursive return-state and all preceding fallback mutation evidence are proven
complete. Commit and push only; do not create or update a pull request.

## Safety-fallback dynamic rank-one characterization checkpoint

`rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs()` returns one complete
counter. It increments for both the folded higher-rank metadata repair and the
indexed runtime SHAPE/CONCAT insertion path, synchronizes layout only after a
positive rewrite, and performs no tensor or operator cleanup on zero.

The fallback call still discards that result. A strict expected-failure AST
contract requires `_fallback_dynamic_rank1_stats` while preserving its sole
`fallback_ir` argument and the immediately following unconditional
`_topologically_sort_operators()` and `infer_model_ir_logical_layouts()` calls.
No refresh is skipped in this unit because recursive relowering and earlier
fallback repairs have not yet been proven to leave terminal layout metadata
equivalent on every zero-owner path.

Focused dynamic-rank-one characterization is `4 passed, 34 deselected, 1
xfailed`. The broader sequential fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`446 passed, 1 xfailed in 26.72s`. Ruff and whitespace validation pass.

At implementation, assign only this result and update the existing
three-occurrence contract so the absolute-final `model_ir` call remains the
sole expression. Validate dynamic Reshape, safety fallback, very-late
orchestration, core, pass efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

## Safety-fallback dynamic rank-one implementation checkpoint

The fallback-only call now assigns its unchanged one-key result to
`_fallback_dynamic_rank1_stats`. The immediately following topological sort and
logical-layout inference remain unconditional and in their original order.
The very-late main-path result remains `_very_late_dynamic_rank1_reshape_stats`,
and the absolute-final `model_ir` call remains the sole discarded expression.

This is observation-only plumbing: no matcher, metadata write, GraphIndex
operation, LayoutState behavior, recursive relowering, refresh, or return path
changed.

Focused dynamic-rank-one validation is `5 passed, 34 deselected`. The broader
sequential fallback-owner, reconciliation, convergence, core, pass-efficiency,
architecture, and TensorFlow import-blocking gate is `447 passed in 27.34s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, determine whether the recursive fallback return state plus complete
norm and dynamic-rank-one evidence is sufficient to guard either refresh. If
that equivalence cannot be proven locally, leave both unconditional and audit
the following `fallback_broadcast_repair_stats` reconciliation result instead.
Commit and push only; do not create or update a pull request.

## Safety-fallback broadcast reconciliation characterization checkpoint

`repair_rank4_channelwise_broadcast_constants_to_runtime_layout()` mutates
constant data/metadata or creates and rewires a clone only when
`repaired_rank4_channelwise_broadcast_constants` increments. It performs no
pruning or zero-rewrite cleanup, and existing positive, no-op, shared-constant,
and convergence fixtures cover its one-key result. The fallback guard is
therefore a complete mutation predicate.

The guarded static-shape reconciliation result is still discarded. A strict
expected-failure AST contract requires a stable two-key zero default named
`_fallback_broadcast_static_shape_stats` and an opt-in complete result on the
positive path. The existing guard, topological sort, logical-layout inference,
and following SE/FC/Gather recovery order remain unchanged.

Focused safety-fallback/broadcast/convergence characterization is `17 passed,
1 xfailed`. The broader sequential fallback-owner, reconciliation,
convergence, core, pass-efficiency, architecture, and TensorFlow
import-blocking gate is `450 passed, 1 xfailed in 27.02s`. Ruff and whitespace
validation pass.

At implementation, add only this reconciliation result plumbing. Do not alter
the broadcast matcher, clone behavior, raw result schema, convergence owner,
guard, or refresh calls. Validate binary layout, safety fallback, static
reconciliation, convergence, core, pass efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Safety-fallback broadcast reconciliation implementation checkpoint

The fallback broadcast owner, its exact one-key result, and its positive guard
are unchanged. The lowerer initializes
`_fallback_broadcast_static_shape_stats` with both complete reconciliation keys
set to zero. A positive broadcast rewrite replaces that default with
`_reconcile_static_tensor_shapes(fallback_ir,
include_mutation_count=True)` before the unchanged topological sort and
logical-layout inference.

This result plumbing adds no graph scan, fingerprint, copy, rewrite, cleanup,
dependency, or unconditional work. The broadcast matcher, shared-constant
clone path, convergence owner, and all later fallback processing retain their
existing behavior.

Focused safety-fallback/broadcast/convergence validation is `18 passed`. The
broader sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `451
passed in 27.03s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the following fallback SINet-shuffle plus SE/FC/Gather
aggregate. Confirm that its existing tensor-count delta covers cleanup-only
paths and characterize the guarded reconciliation result without changing the
combined predicate. Commit and push only; do not create or update a pull
request.

## Safety-fallback SE/FC/Gather reconciliation characterization checkpoint

The fallback boundary samples `fallback_se_fc_gather_tensor_count` before the
SINet shuffle and SE/FC/Gather owners. Its predicate combines all three exact
rewrite counters with `len(fallback_ir.tensors) <
fallback_se_fc_gather_tensor_count`, so zero-rewrite pruning is also covered.
This is complete mutation evidence for the existing reconciliation guard.

The guarded reconciliation return is discarded. A strict expected-failure AST
contract requires `_fallback_se_fc_gather_static_shape_stats`, initialized with
both zero keys and replaced by an opt-in complete result only when the existing
combined predicate is true. It freezes every predicate term and the following
placeholder-MatMul boundary.

Focused safety-fallback and SE/FC/Gather characterization is `18 passed, 1
xfailed`. The broader sequential fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`462 passed, 1 xfailed in 27.24s`. Ruff and whitespace validation pass.

At implementation, add only this result plumbing. Do not change the SINet,
SE/FC, Gather, pruning, tensor-delta, cluster, or guard contracts. Validate the
complete owner suites, safety fallback, static reconciliation, core, pass
efficiency, architecture, and TensorFlow import blocking sequentially. Commit
and push only; do not create or update a pull request.

## Safety-fallback SE/FC/Gather reconciliation implementation checkpoint

The fallback SINet-shuffle and SE/FC/Gather owners, their result schemas, the
before/after tensor count, and all four combined predicate terms are unchanged.
The lowerer initializes `_fallback_se_fc_gather_static_shape_stats` with both
complete reconciliation keys set to zero and replaces it with the opt-in
complete result only when the established predicate is true.

This adds no scan, fingerprint, copy, rewrite, cleanup, dependency, or
unconditional reconciliation. The following placeholder-MatMul restore
boundary remains immediately adjacent and unchanged.

Focused safety-fallback and SE/FC/Gather validation is `19 passed`. The broader
sequential fallback-owner, reconciliation, convergence, core, pass-efficiency,
architecture, and TensorFlow import-blocking gate is `463 passed in 27.00s`.
Ruff, Python bytecode compilation, and whitespace validation pass.

At resume, audit the placeholder-MatMul restore owner and its guarded
reconciliation. Confirm its positive-only pruning contract and characterize
the discarded reconciliation result without changing the guard. Commit and
push only; do not create or update a pull request.

## Safety-fallback placeholder-MatMul characterization checkpoint

`restore_placeholder_matmul_flattened_inputs()` rewires a proven
placeholder-Reshape consumer and removes that Reshape through the differential
GraphIndex. Its unused-tensor pruning runs only after `restored > 0`, so
`restored_placeholder_matmul_flattened_inputs` is complete mutation evidence.

The fallback lowerer currently invokes the owner inline in its guard and
discards both its result and the guarded reconciliation result. A strict
expected-failure AST contract requires one
`fallback_placeholder_matmul_stats` assignment, a stable two-key
`_fallback_placeholder_matmul_static_shape_stats` default, and an opt-in
complete reconciliation assignment under the unchanged positive predicate.
The immediately following topological sort remains fixed.

Focused fallback placeholder-MatMul characterization is `7 passed, 11
deselected, 1 xfailed`. The broader sequential fallback-owner, reconciliation,
convergence, core, pass-efficiency, architecture, and TensorFlow
import-blocking gate is `463 passed, 1 xfailed in 27.32s`. Ruff and whitespace
validation pass.

At implementation, preserve a single owner invocation and change no matcher,
GraphIndex, pruning, guard, or sort behavior. Validate dynamic Reshape,
safety fallback, static reconciliation, core, pass efficiency, architecture,
and TensorFlow import blocking sequentially. Commit and push only; do not
create or update a pull request.

## Safety-fallback placeholder-MatMul implementation checkpoint

The fallback owner now runs exactly once into
`fallback_placeholder_matmul_stats`. The lowerer initializes
`_fallback_placeholder_matmul_static_shape_stats` with both complete zero keys
and replaces it with the opt-in reconciliation result only when
`restored_placeholder_matmul_flattened_inputs` is positive.

No matcher, GraphIndex edit, positive-only pruning, guard, reconciliation
eligibility, or following topological sort changed. The former inline call is
only made explicit and its two results are retained.

Focused fallback placeholder-MatMul validation is `8 passed, 11 deselected`.
The broader sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `464
passed in 26.93s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit the following fallback unbound-input repair owner and its
guarded reconciliation. Confirm whether it performs cleanup or layout
maintenance outside its rewrite counter before selecting that boundary. Commit
and push only; do not create or update a pull request.

## Safety-fallback unbound reconciliation characterization checkpoint

`_repair_unbound_nonconstant_operator_inputs_with_layout_transpose()` already
owns static-shape reconciliation after a positive indexed repair and forwards
the repair's current GraphIndex. A new positive fixture proves exactly one
wrapper-owned reconciliation and confirms that the same ModelIR index is
reused.

The fallback caller nevertheless checks the returned counter and immediately
runs a second reconciliation, with no intervening graph or metadata mutation.
A strict expected-failure AST contract requires
`fallback_conv_input_stats` to follow `_fallback_unbound_repair_stats` directly,
removing only this duplicate full-graph scan. The owner invocation, wrapper
reconciliation, return schema, following Conv-input owner, and all guards after
it remain unchanged.

Focused indexed-unbound and safety-fallback characterization is `11 passed, 1
xfailed`. It also exposed a stale test assumption that removed legacy
producer/consumer map-builder attributes still existed; the sentinel
monkeypatches now use `raising=False` and still fail if either builder is
reintroduced and called. The broader sequential fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `469 passed, 1 xfailed in 27.76s`. Ruff and
whitespace validation pass.

At implementation, delete only the redundant caller-side guard and
reconciliation. Validate positive/no-op unbound repair, safety fallback,
static reconciliation, core, pass efficiency, architecture, and TensorFlow
import blocking sequentially. Commit and push only; do not create or update a
pull request.

## Safety-fallback unbound reconciliation implementation checkpoint

The redundant fallback caller-side guard and reconciliation are removed. A
positive unbound-input repair still performs exactly one static reconciliation
inside `_repair_unbound_nonconstant_operator_inputs_with_layout_transpose()`,
using the repair's live GraphIndex. The fallback then proceeds directly from
`_fallback_unbound_repair_stats` to `fallback_conv_input_stats`.

No indexed matcher, inserted transpose, wrapper result schema, GraphIndex,
no-op behavior, or following Conv-input repair changed. This is a strict
one-scan elimination with no intervening mutation to preserve.

Focused indexed-unbound and safety-fallback validation is `12 passed`. The
broader sequential fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `470
passed in 27.49s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit `fallback_conv_input_stats`. Its two child owners can prune
on zero rewrite, so do not treat the existing stale-transpose counter as
complete without tensor-delta evidence. Commit and push only; do not create or
update a pull request.

## Safety-fallback Conv-input evidence characterization checkpoint

`_run_indexed_conv_input_adapter_repairs()` returns exact counters for
singleton-Reshape and stale-Transpose repairs, but each child unconditionally
prunes unused tensors. Existing zero-rewrite coverage proves two cleanup
opportunities, so the raw dictionary is incomplete mutation evidence.

A strict expected-failure safety-fallback contract requires
`fallback_conv_input_tensor_count`, a clamped `pruned_unused_tensors` delta
merged into `fallback_conv_input_stats`, and a stable two-key
`_fallback_conv_input_static_shape_stats`. The existing reconciliation guard
continues to read only
`repaired_stale_nchw_to_nhwc_conv_input_transposes`; its opt-in complete result
is assigned instead of discarded. Singleton repair already writes the Conv
output shape/signature directly, and prune-only cleanup does not broaden this
guard. The following mixed-Concat owner remains fixed.

Focused Conv-input and safety-fallback characterization is `16 passed, 1
xfailed`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `479 passed, 1 xfailed in 27.90s`. Ruff and
whitespace validation pass.

At implementation, add only the tensor-delta and reconciliation result
plumbing. Do not change either indexed owner, shared GraphIndex, raw counters,
guard, or following fallback order. Validate indexed Conv-input repair, safety
fallback, static reconciliation, core, pass efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Safety-fallback Conv-input evidence implementation checkpoint

The fallback runner now samples `fallback_conv_input_tensor_count` and merges a
clamped `pruned_unused_tensors` delta with its unchanged singleton-Reshape and
stale-Transpose counters. This captures both child owners' zero-rewrite cleanup
without rerunning either owner or scanning the graph.

`_fallback_conv_input_static_shape_stats` supplies the stable complete zero
schema and receives the opt-in reconciliation result only under the unchanged
stale-Transpose predicate. Singleton repair continues to update Conv output
metadata directly, and cleanup-only pruning does not broaden the guard. The
following mixed-Concat owner remains adjacent.

Focused Conv-input, safety-fallback, and occurrence validation is `18 passed,
22 deselected`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `480 passed in 27.81s`. Ruff, Python
bytecode compilation, and whitespace validation pass.

At resume, audit `fallback_concat_layout_stats` and its guarded reconciliation.
Confirm its counter and absence or presence of cleanup-only mutation before
changing that boundary. Commit and push only; do not create or update a pull
request.

## Safety-fallback mixed-Concat reconciliation characterization checkpoint

`_repair_mixed_nhwc_inputs_for_nchw_concat()` inserts local transpose adapters,
rewires the Concat, and updates its output metadata only when
`repaired_mixed_nhwc_inputs_for_nchw_concat` increments. It performs no pruning
or zero-rewrite cleanup, and existing positive/no-op/idempotence/fan-out tests
make the one-key result a complete mutation predicate.

The fallback guard is already correct, but its reconciliation result is
discarded. A strict expected-failure AST contract requires a stable two-key
`_fallback_mixed_concat_static_shape_stats` default and an opt-in complete
result inside the unchanged positive guard. The following Concat-axis repair
remains fixed.

Focused mixed-Concat and safety-fallback characterization is `18 passed, 1
xfailed`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `490 passed, 1 xfailed in 27.67s`. Ruff and
whitespace validation pass.

At implementation, add only this result plumbing. Do not alter the matcher,
adapter insertion, Concat rewire, output metadata, raw result, guard, or next
owner. Validate mixed-Concat repair, safety fallback, static reconciliation,
core, pass efficiency, architecture, and TensorFlow import blocking
sequentially. Commit and push only; do not create or update a pull request.

## Safety-fallback mixed-Concat reconciliation implementation checkpoint

The fallback mixed-Concat owner, its exact one-key result, and positive guard
are unchanged. `_fallback_mixed_concat_static_shape_stats` supplies both zero
keys and receives the opt-in complete reconciliation result only after a
positive repair.

No matcher, adapter insertion, Concat rewire, output metadata, scan,
dependency, or following Concat-axis owner changed. This is observation-only
result plumbing under the existing conditional reconciliation.

Focused mixed-Concat and safety-fallback validation is `19 passed`. The broader
sequential indexed-owner, fallback-owner, reconciliation, convergence, core,
pass-efficiency, architecture, and TensorFlow import-blocking gate is `491
passed in 29.82s`. Ruff, Python bytecode compilation, and whitespace validation
pass.

At resume, audit `fallback_concat_axis_stats` and its guarded reconciliation.
Confirm its complete counter and cleanup behavior before changing that
boundary. Commit and push only; do not create or update a pull request.

## Safety-fallback Concat-axis reconciliation characterization checkpoint

`_repair_nchw_concat_transpose_conv_axes()` updates a proven Concat axis and
related tensor metadata only when
`repaired_nchw_concat_transpose_conv_axes` increments. The indexed owner has no
pruning or zero-rewrite cleanup, and existing positive/no-op/multi-rewrite
coverage makes the raw counter complete.

The fallback positive guard is unchanged, but its reconciliation result is
discarded. A strict expected-failure contract requires a stable
`_fallback_concat_axis_static_shape_stats` zero dictionary and an opt-in
complete result inside the same guard. The following stale binary-layout owner
remains fixed.

Focused indexed Concat-axis and safety-fallback characterization is `37 passed,
1 xfailed`. The broader sequential indexed-owner, fallback-owner,
reconciliation, convergence, core, pass-efficiency, architecture, and
TensorFlow import-blocking gate is `519 passed, 1 xfailed in 27.75s`. Ruff and
whitespace validation pass.

At implementation, add only this result plumbing. Do not change any of the
three production occurrences, indexed matching, axis/metadata writes, result
schema, guard, or following owner. Validate indexed Concat-axis repair, safety
fallback, static reconciliation, core, pass efficiency, architecture, and
TensorFlow import blocking sequentially. Commit and push only; do not create or
update a pull request.

## Safety-fallback Concat-axis reconciliation implementation checkpoint

Only the fallback occurrence now initializes
`_fallback_concat_axis_static_shape_stats` with both zero keys and assigns the
opt-in complete reconciliation result on a positive repair. Its exact owner
counter and guard are unchanged.

The very-late and final occurrences, indexed matching, axis/metadata writes,
and following stale binary-layout owner remain unchanged. No scan, dependency,
or unconditional work is added.

Focused indexed Concat-axis and safety-fallback validation is `38 passed`. The
broader sequential indexed-owner, fallback-owner, reconciliation, convergence,
core, pass-efficiency, architecture, and TensorFlow import-blocking gate is
`520 passed in 27.79s`. Ruff, Python bytecode compilation, and whitespace
validation pass.

At resume, audit `fallback_binary_layout_stats` and its guarded reconciliation.
Confirm the owner's cleanup behavior and counter completeness before changing
that boundary. Commit and push only; do not create or update a pull request.

## Safety-fallback binary-layout evidence characterization checkpoint

`_repair_stale_nchw_to_nhwc_channelwise_binary_transposes()` has one exact
rewrite counter but unconditionally prunes unused tensors. A focused zero-
rewrite fixture removes an unused constant while returning zero, proving that
the raw result is incomplete mutation evidence.

A strict expected-failure fallback contract requires
`fallback_binary_layout_tensor_count`, a clamped `pruned_unused_tensors` delta,
and a stable `_fallback_binary_layout_static_shape_stats` zero dictionary. The
existing stale-binary repair guard remains unchanged and assigns an opt-in
complete reconciliation result instead of discarding it. Cleanup-only evidence
does not broaden the guard, and the immediately following topological sort is
fixed.

At implementation, add only this result plumbing. Do not alter indexed
matching, rewiring, output metadata, pruning, raw counter, guard, or sort.
Validate stale binary repair, safety fallback, static reconciliation,
convergence, core, pass efficiency, architecture, and TensorFlow import
blocking sequentially. Commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- focused fallback/binary/convergence selection: `22 passed, 12 deselected, 1 xfailed`
- broad related regression gate: `533 passed, 1 xfailed`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet complete-evidence contract
defined by this checkpoint; there are no unexpected failures.

## Safety-fallback binary-layout evidence implementation checkpoint

Only the recursive safety-fallback occurrence now records the stale-binary
owner's complete local evidence. It captures the tensor count before the owner,
merges a clamped `pruned_unused_tensors` delta into the exact rewrite counter,
and initializes `_fallback_binary_layout_static_shape_stats` with both stable
zero keys. A positive rewrite still controls the same reconciliation guard,
which now stores the opt-in complete result instead of discarding it.

Cleanup-only calls do not trigger shape propagation. Indexed matching,
rewiring, output metadata, pruning behavior, the raw owner schema, and the
following unconditional topological sort are unchanged. The preceding
Concat-axis orchestration contract now identifies the binary-layout tensor
count as the next owner's first statement.

Implementation validation completed sequentially under `uv`:

- focused fallback/binary/convergence selection: `23 passed, 12 deselected`
- broad related regression gate: `534 passed in 27.64s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the fallback layout-validation metadata boundary before the
high-rank BatchMatMul compression owner. Commit and push coherent units only;
do not create or update a pull request.

## ONNX graph-input type-checking checkpoint

The ONNX protobuf repeated-field stub can cause Pylance to infer a graph input
iterator item as the generic class `type[In]`, which hides the concrete
`ValueInfoProto.name` attribute. The lowerer now narrows only that local item
with `typing.cast(onnx.ValueInfoProto, ...)`. This is a static-analysis-only
clarification: graph traversal, input filtering, names, shapes, and runtime
behavior are unchanged.

## Safety-fallback terminal layout-validation characterization checkpoint

`validate_model_ir_layout_annotations()` is a pure read-only query, but its
fallback call currently precedes high-rank BatchMatMul compression and indexed
binary convergence. It therefore describes a non-terminal graph. In addition,
an empty result leaves `logical_layout_validation_errors` inherited from the
recursive lower in metadata.

A passing purity fixture records an invalid annotation without modifying the
ModelIR. A strict expected-failure orchestration contract requires validation
after high-rank compression, its guarded reconciliation/sort, indexed binary
convergence, and the final sort. A non-empty result keeps the existing list
schema; an empty result removes only the stale validation-error key. The
`layout_optimize_fallback` metadata assignment remains before the high-rank
owner, and finalization remains the terminal operation.

At implementation, move only the pure validation/metadata block and add the
empty-result removal. Do not change either mutation owner, its guard, sorting,
fallback reason/samples, finalization, dependencies, or TensorFlow boundary.
Validate sequentially, then commit and push only; do not create or update a
pull request.

The first attempt to add the two existing rank-6 BatchMatMul tests to the broad
gate stopped during collection: `tests/test_tflite_builder_direct.py` imports
`run_elementwise_gate_layout_cleanup` from the lowerer, while commit
`5c4f72ae` removed that compatibility re-export during orchestration
extraction. An AST/runtime audit of all 115 names imported from the lowerer by
that test file found this as the sole missing name. The implementation remains
owned by `passes/elementwise_gate_layout.py`; restoring an explicitly marked
compatibility import is sufficient and does not reconnect it to lowerer
orchestration. Record and repair this independently before using that file as a
gate.

Characterization validation completed sequentially under `uv` after isolating
that collection issue:

- focused layout/binary fallback selection: `3 passed, 10 deselected, 1 xfailed`
- broad related regression gate: `535 passed, 1 xfailed in 27.37s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet terminal-validation contract;
the broad gate has no unexpected failure.

## Elementwise gate compatibility re-export checkpoint

The lowerer again exposes `run_elementwise_gate_layout_cleanup`, importing it
from the extracted gate-layout orchestration module and marking it as a
compatibility re-export. This restores the established Python/test import
contract removed by `5c4f72ae` without adding a direct lowerer call, a second
owner, a new scan, or a dependency.

Validate collection of `tests/test_tflite_builder_direct.py`, the two rank-6
BatchMatMul tests, gate orchestration/architecture, pass efficiency, and the
TensorFlow import boundary sequentially. Commit and push this repair
independently; do not create or update a pull request.

Compatibility validation completed sequentially under `uv`:

- previously blocked rank-6 structure/numeric tests: `2 passed`
- gate orchestration, architecture, pass efficiency, optional TensorFlow
  boundary, and rank-6 tests: `312 passed in 26.08s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

## Safety-fallback terminal layout-validation implementation checkpoint

The pure fallback validation block now runs after high-rank BatchMatMul
compression, its guarded reconciliation/sort, indexed binary convergence, and
the terminal topological sort. A non-empty result writes the same
`logical_layout_validation_errors` list. An empty result removes only that key
when it was inherited from the recursive lower.

The high-rank owner, convergence owner, guards, sorts,
`layout_optimize_fallback` reason/count/samples, finalizer, dependency set, and
TensorFlow-free boundary are unchanged. The strict characterization contract
is now a normal passing test.

Implementation validation completed sequentially under `uv`:

- focused layout/binary fallback selection: `4 passed, 10 deselected`
- broad related gate plus rank-6 structure/numeric parity: `538 passed in 27.90s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `fallback_high_rank_bmm_stats` for complete mutation evidence
and retain its guard and following indexed binary convergence boundary. Commit
and push only; do not create or update a pull request.

## Safety-fallback high-rank BatchMatMul characterization checkpoint

`_compress_static_high_rank_batch_matmul()` rewrites only fully-static,
same-batch-shape rank-greater-than-five `BATCH_MATMUL` operators. Its unused-
tensor pruning and optional layout-state sync occur only when the exact
`compressed_static_high_rank_batch_matmul` counter is positive. A focused
zero-rewrite fixture retains an unrelated unused constant, proving that the raw
counter is complete mutation evidence without a cleanup-only path. Existing
rank-6 structure and numeric-parity tests cover the positive path.

A strict expected-failure fallback contract requires a stable
`_fallback_high_rank_bmm_static_shape_stats` zero dictionary and assigns the
opt-in complete reconciliation result under the unchanged positive guard. The
guarded topological sort and immediately following indexed binary convergence
remain fixed.

At implementation, add only this result plumbing. Do not change eligibility,
reshape construction, graph-index mutation, pruning, raw result schema, guard,
sort, convergence, terminal validation, dependencies, or TensorFlow boundary.
Validate sequentially, then commit and push only; do not create or update a
pull request.

Characterization validation completed sequentially under `uv`:

- focused high-rank/fallback plus rank-6 parity selection:
  `4 passed, 13 deselected, 1 xfailed`
- broad related gate plus rank-6 structure/numeric parity:
  `539 passed, 1 xfailed in 28.13s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet result-staging contract; there
are no unexpected failures.

## Safety-fallback high-rank BatchMatMul implementation checkpoint

Only the recursive fallback occurrence now initializes
`_fallback_high_rank_bmm_static_shape_stats` with the two stable zero keys and
stores the opt-in complete reconciliation result under the existing positive
compression guard. The owner call, raw result schema, guard, and guarded
topological sort are unchanged.

Compression eligibility, reshape construction, graph-index mutation, pruning,
the final indexed binary convergence, terminal validation, dependency set, and
TensorFlow-free boundary are unchanged. The preceding terminal-validation
contract was updated only for the new result statement at the next owner's
boundary.

Implementation validation completed sequentially under `uv`:

- safety fallback plus rank-6 structure/numeric parity: `18 passed`
- broad related gate plus rank-6 structure/numeric parity: `540 passed in 27.98s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the result returned by
`_run_indexed_binary_layout_convergence(fallback_ir)` and retain its following
terminal sort/validation boundary. Commit and push only; do not create or
update a pull request.

## Safety-fallback indexed binary-convergence characterization checkpoint

`_run_indexed_binary_layout_convergence()` owns up to three rounds of
rank-four broadcast repair, stale binary-Transpose repair, and static-shape
reconciliation with one GraphIndex. It aggregates all three pure mutation
counters and stops after the first all-zero round. Existing immediate-stable,
second-round-stable, three-round-cap, multi-repair, index-reuse, and
legacy-equivalence tests prove its result is complete.

The fallback caller currently discards that complete dictionary. A strict
expected-failure orchestration contract requires only assigning it to
`_fallback_binary_layout_convergence_stats`. The owner call remains single and
the following terminal sort/validation boundary remains fixed; no additional
reconciliation is required.

At implementation, replace only the fallback expression with a result-capture
assignment. Do not change convergence rounds, owner order, GraphIndex reuse,
statistics, sorting, terminal validation, dependencies, or TensorFlow
boundary. Validate sequentially, then commit and push only; do not create or
update a pull request.

Characterization validation completed sequentially under `uv`:

- focused convergence/high-rank/terminal-layout selection:
  `14 passed, 13 deselected, 1 xfailed`
- broad related gate plus rank-6 structure/numeric parity:
  `540 passed, 1 xfailed in 28.63s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet result-capture contract; there
are no unexpected failures.

## Safety-fallback indexed binary-convergence implementation checkpoint

Only the recursive fallback occurrence now assigns the complete convergence
dictionary to `_fallback_binary_layout_convergence_stats`. The owner call
remains single and retains its three-round cap, stable-round exit, owner order,
single GraphIndex, and aggregate three-key schema.

No additional reconciliation, sort, scan, or dependency is added. The
following terminal sort and validation retain their order and now identify the
result-capture assignment as the preceding owner boundary.

Implementation validation completed sequentially under `uv`:

- focused convergence/high-rank/terminal-layout selection:
  `15 passed, 13 deselected`
- broad related gate plus rank-6 structure/numeric parity:
  `541 passed in 27.89s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the corresponding primary-path terminal layout-validation and
indexed binary-convergence boundary. Commit and push only; do not create or
update a pull request.

## Primary terminal layout-validation characterization checkpoint

The primary path currently computes `layout_problems` before indexed binary
convergence, high-rank binary coalescing, dynamic-boundary signature
realignment, and the final topological sort. The comment describes a terminal
graph, but the actual validation result can precede all four later mutations.
The same validator was already proven pure by the fallback checkpoint.

A dedicated strict expected-failure orchestration contract requires validation
immediately after the final sort. A non-empty result preserves the established
`logical_layout_validation_errors` list schema; an empty result removes only a
stale key. Finalization follows immediately. Progress advancement/spinner
closure remain before indexed convergence and are not moved.

At implementation, move only the pure validation/metadata block and add its
empty-result removal. Do not change progress reporting, convergence,
coalescing, signature realignment, sorting, finalization, dependencies, or the
TensorFlow boundary. Validate sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal validation, indexed convergence, and high-rank binary owners:
  `13 passed, 1 xfailed`
- broad related gate plus rank-6 structure/numeric parity:
  `543 passed, 1 xfailed in 27.93s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet primary terminal-validation
contract; there are no unexpected failures.

## Primary terminal layout-validation implementation checkpoint

The pure primary validation block now runs after indexed binary convergence,
high-rank binary coalescing, dynamic-boundary signature realignment, and the
final topological sort. A non-empty result writes the same
`logical_layout_validation_errors` list; an empty terminal result removes only
that stale key. Finalization remains immediately after validation.

Progress advancement/spinner closure, all four terminal mutation owners,
sorting, dependencies, and the TensorFlow-free boundary are unchanged. The
misleading pre-convergence validation comment now describes terminal layout
cleanup rather than claiming that the earlier graph was terminal.

Implementation validation completed sequentially under `uv`:

- terminal validation, indexed convergence, and high-rank binary owners:
  `14 passed`
- broad related gate plus rank-6 structure/numeric parity:
  `544 passed in 27.95s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit result capture for the primary terminal indexed convergence,
high-rank binary coalescing, and boundary-signature realignment calls. Commit
and push only; do not create or update a pull request.

## Primary terminal mutation-result characterization checkpoint

The primary terminal indexed binary-convergence owner returns its complete
three-counter aggregate. Static high-rank binary coalescing mutates only when
its exact rewrite counter increments and its zero path is covered as a no-op.
Dynamic-boundary signature realignment returns the exact updated-tensor count,
with changed, no-op, missing-metadata, wrapper-equivalence, and idempotency
coverage. None requires caller-side reconciliation.

The primary caller currently discards all three dictionaries. A strict
expected-failure orchestration contract requires assigning them respectively
to `_final_binary_layout_convergence_stats`,
`_final_high_rank_binary_stats`, and
`_final_dynamic_boundary_signature_stats`. Calls, arguments, layout-state
handoff, order, final sort, terminal validation, and finalization remain fixed.

At implementation, replace only the three expressions with assignments. Do
not change any owner, result schema, GraphIndex use, layout sync, sort,
validation, dependency, or TensorFlow boundary. Validate sequentially, then
commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused terminal/convergence/high-rank/signature selection:
  `22 passed, 11 deselected, 1 xfailed`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `563 passed, 1 xfailed in 28.27s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet three-result capture contract;
there are no unexpected failures.

## Primary terminal mutation-result implementation checkpoint

The primary path now assigns the complete terminal dictionaries to
`_final_binary_layout_convergence_stats`,
`_final_high_rank_binary_stats`, and
`_final_dynamic_boundary_signature_stats`. All three calls, arguments,
layout-state handoff, order, and result schemas are unchanged.

No reconciliation, scan, sort, or dependency is added. The terminal sort,
layout validation, stale-error removal, finalization, and TensorFlow-free
boundary remain unchanged. The strict characterization contract is now a
normal passing test.

Implementation validation completed sequentially under `uv`:

- focused terminal/convergence/high-rank/signature selection:
  `23 passed, 11 deselected`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `564 passed in 28.36s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, continue the primary final-pass reconciliation inventory with
`final_high_rank_bmm_stats`. Confirm counter/cleanup completeness before
retaining its guarded reconciliation result. Commit and push only; do not
create or update a pull request.

## Primary final high-rank BatchMatMul characterization checkpoint

`final_high_rank_bmm_stats` uses the same static high-rank BatchMatMul owner
already characterized in the recursive fallback. Its exact positive counter
contains every reshape/prune/layout-sync mutation, while zero is a true no-op.
Existing rank-6 structure and numeric-parity tests cover the positive path.

A strict expected-failure primary-path contract requires a stable
`_final_high_rank_bmm_static_shape_stats` zero dictionary and assigns the
opt-in complete reconciliation result under the unchanged positive guard. The
guarded topological sort and immediately following `final_pad_layout_stats`
owner remain fixed.

At implementation, add only result plumbing. Do not change compression
eligibility, graph-index mutation, pruning, layout sync, owner schema, guard,
sort, Pad boundary, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration plus rank-6 structure/numeric parity:
  `4 passed, 1 xfailed`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `564 passed, 1 xfailed in 28.33s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final high-rank reconciliation
contract; there are no unexpected failures.

## Primary final high-rank BatchMatMul implementation checkpoint

The primary path now initializes `_final_high_rank_bmm_static_shape_stats`
with both stable zero keys and stores the opt-in complete reconciliation result
under the unchanged positive compression guard. The guarded topological sort
and following Pad owner retain their positions.

Compression eligibility, reshape construction, graph-index mutation, pruning,
layout-state sync, raw result schema, dependencies, and the TensorFlow-free
boundary are unchanged. The strict characterization contract is now a normal
passing test.

Implementation validation completed sequentially under `uv`:

- terminal orchestration plus rank-6 structure/numeric parity: `5 passed`
- expanded broad related gate plus rank-6 structure/numeric parity:
  `565 passed in 28.72s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_pad_layout_stats` for counter/cleanup completeness and
retain its guarded reconciliation result without changing the following Conv-
input owner boundary. Commit and push only; do not create or update a pull
request.

## Primary final Pad reconciliation characterization checkpoint

`repair_channel_last_inputs_for_channel_first_pad()` inserts its adapter,
constant, and Transpose only when
`repaired_channel_last_inputs_for_channel_first_pad` increments. Optional
layout-state sync is under the same positive predicate. The zero path has no
pruning or cleanup, and existing positive/native-NCHW/unproven-mismatch tests
cover both paths plus GraphIndex/layout-state consistency.

A strict expected-failure primary-path contract requires a stable
`_final_pad_layout_static_shape_stats` zero dictionary and assigns the opt-in
complete reconciliation result under the unchanged positive guard. The
guarded topological sort and following `final_conv_input_stats` owner remain
fixed.

At implementation, add only result plumbing. Do not change Pad matching,
adapter construction, lineage, GraphIndex mutation, layout sync, result schema,
guard, sort, Conv boundary, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration and Pad owner: `7 passed, 1 xfailed`
- expanded broad related gate plus Pad/rank-6 coverage:
  `569 passed, 1 xfailed in 28.54s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final Pad reconciliation
contract; there are no unexpected failures.

## Primary final Pad reconciliation implementation checkpoint

The primary path now initializes `_final_pad_layout_static_shape_stats` with
both stable zero keys and stores the opt-in complete reconciliation result
under the unchanged positive Pad-repair guard. The guarded topological sort and
following Conv-input owner retain their positions.

Pad matching, adapter/constant/Transpose construction, lineage, GraphIndex
mutation, layout-state sync, raw result schema, dependencies, and the
TensorFlow-free boundary are unchanged. The strict characterization contract
is now a normal passing test.

Implementation validation completed sequentially under `uv`:

- terminal orchestration and Pad owner: `8 passed`
- expanded broad related gate plus Pad/rank-6 coverage:
  `570 passed in 28.52s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_conv_input_stats` for zero-rewrite cleanup/pruning
before retaining its guarded reconciliation result. Keep the following mixed-
Concat owner boundary fixed. Commit and push only; do not create or update a
pull request.

## Primary final Conv-input evidence characterization checkpoint

The standalone stale NCHW-to-NHWC Conv-input owner unconditionally prunes
unused tensors after its indexed rewrite loop. A focused real-tensor fixture
removes an unrelated unused constant while returning a zero rewrite counter,
proving that `final_conv_input_stats` is incomplete mutation evidence today.

A strict expected-failure primary-path contract requires
`final_conv_input_tensor_count`, a clamped `pruned_unused_tensors` delta, and a
stable `_final_conv_input_static_shape_stats` zero dictionary. The existing
rewrite-only guard assigns the opt-in complete reconciliation result but does
not run for cleanup-only evidence. Its guarded sort and the following mixed-
Concat owner remain fixed.

At implementation, add only this evidence plumbing. Do not change indexed
matching, rewiring, output metadata, pruning, raw owner schema, guard, sort,
mixed-Concat boundary, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- focused final/prune/convergence selection:
  `4 passed, 10 deselected, 1 xfailed`
- expanded broad related gate: `571 passed, 1 xfailed in 28.23s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet complete Conv-input evidence
contract; there are no unexpected failures.

## Primary final Conv-input evidence implementation checkpoint

The primary path now snapshots the tensor count, merges a clamped
`pruned_unused_tensors` delta into `final_conv_input_stats`, initializes
`_final_conv_input_static_shape_stats` with both stable zero keys, and stores
the opt-in complete reconciliation result under the unchanged positive rewrite
guard. Cleanup-only evidence does not trigger shape propagation.

Indexed matching, rewiring, output metadata, pruning behavior, the raw owner
schema, guarded sort, dependencies, and the following mixed-Concat boundary
are unchanged. The preceding Pad contract now identifies the Conv-input tensor
snapshot as the next owner start.

Implementation validation completed sequentially under `uv`:

- focused final/prune/convergence selection: `5 passed, 10 deselected`
- expanded broad related gate: `572 passed in 28.31s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_concat_layout_stats` for zero-rewrite cleanup before
retaining its guarded reconciliation result. Keep the following Concat-axis
owner boundary fixed. Commit and push only; do not create or update a pull
request.

## Primary final mixed-Concat reconciliation characterization checkpoint

`final_concat_layout_stats` uses the same mixed-NHWC-input repair for NCHW
Concat already characterized in the recursive fallback. Its exact counter
covers the input adapter/rewire and output metadata mutation; the zero path has
no pruning or cleanup.

A strict expected-failure primary-path contract requires a stable
`_final_mixed_concat_static_shape_stats` zero dictionary and assigns the opt-in
complete reconciliation result under the unchanged positive guard. The
guarded sort and following `final_concat_axis_stats` owner remain fixed.

At implementation, add only result plumbing. Do not change matching, adapter
insertion, rewiring, metadata, result schema, guard, sort, Concat-axis boundary,
dependencies, or TensorFlow behavior. Validate sequentially, then commit and
push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration and mixed-Concat owner: `15 passed, 1 xfailed`
- expanded broad related gate: `572 passed, 1 xfailed in 28.34s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final mixed-Concat
reconciliation contract; there are no unexpected failures.

## Primary final mixed-Concat reconciliation implementation checkpoint

The primary path now initializes `_final_mixed_concat_static_shape_stats` with
both stable zero keys and stores the opt-in complete reconciliation result
under the unchanged positive mixed-Concat guard. The guarded sort and following
Concat-axis owner retain their positions.

Matching, adapter insertion, rewiring, output metadata, raw result schema,
dependencies, and the TensorFlow-free boundary are unchanged. The strict
characterization contract is now a normal passing test.

Implementation validation completed sequentially under `uv`:

- terminal orchestration and mixed-Concat owner: `16 passed`
- expanded broad related gate: `573 passed in 28.97s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, stage complete reconciliation evidence for
`final_concat_axis_stats`, retaining its guard, sort, and following stale-
binary owner boundary. Commit and push only; do not create or update a pull
request.

## Primary final Concat-axis/binary evidence characterization checkpoint

The final Concat-axis owner is the same counter-complete, cleanup-free repair
already characterized in the fallback. The immediately following stale
channelwise-binary owner is also shared with the fallback and unconditionally
prunes unused tensors, including zero-rewrite calls.

A strict expected-failure primary-path contract requires a stable
`_final_concat_axis_static_shape_stats` result under the existing axis guard,
then `final_binary_layout_tensor_count`, a clamped `pruned_unused_tensors`
delta, and stable `_final_binary_layout_static_shape_stats` under the existing
binary guard. Both guarded sorts and the following progress boundary remain
fixed; cleanup-only evidence does not trigger reconciliation.

At implementation, add only result/prune plumbing. Do not change either owner,
matching, rewiring, metadata, pruning, raw schemas, guards, sorts, progress,
dependencies, or TensorFlow behavior. Validate sequentially, then commit and
push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal/Concat-axis/binary owner gate: `37 passed, 1 xfailed`
- expanded broad related gate: `573 passed, 1 xfailed in 29.12s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet complete final Concat-axis and
binary evidence contract; there are no unexpected failures.

## Primary final Concat-axis/binary evidence implementation checkpoint

The primary path now initializes `_final_concat_axis_static_shape_stats` with
both stable zero keys and stores the opt-in complete result under the unchanged
positive axis guard. It then snapshots the tensor count, merges a clamped
`pruned_unused_tensors` delta into `final_binary_layout_stats`, initializes
`_final_binary_layout_static_shape_stats`, and stores the opt-in complete result
under the unchanged positive binary guard.

Cleanup-only stale-binary evidence does not trigger reconciliation. Both
owners, matching, rewiring, metadata, pruning, raw schemas, guards, guarded
sorts, progress boundary, dependencies, and TensorFlow-free behavior are
unchanged.

Implementation validation completed sequentially under `uv`:

- terminal/Concat-axis/binary owner gate: `38 passed`
- expanded broad related gate: `574 passed in 28.45s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, continue the remaining primary final-pass reconciliation inventory
immediately before `final_high_rank_bmm_stats`, beginning with
`final_sinet_concat_resize_stats`. Confirm each owner's counter and cleanup
completeness before changing its caller. Commit and push only; do not create or
update a pull request.

## Primary final SiNet Concat/Resize characterization checkpoint

`_optimize_sinet_concat_resize_affine_transpose_chains()` mutates only after a
transactional plan revalidation succeeds. Its exact positive counter covers
rewiring, metadata updates, removals, optional legacy adapter insertion,
pruning, and layout-state sync. Preflight, unsafe/stale-plan, rewrite-cap, and
second-run zero results are true no-ops. Existing indexed and numeric-parity
coverage spans dtype, constant form, input order, Resize type, fan-out, and
legacy behavior.

A strict expected-failure primary-path contract requires a stable
`_final_sinet_concat_resize_static_shape_stats` zero dictionary and assigns the
opt-in complete reconciliation result under the unchanged positive guard. No
sort is added, and the following `final_high_rank_bmm_stats` boundary remains
fixed.

At implementation, add only result plumbing. Do not change matching,
transactional plan application, rewiring, metadata, pruning, layout sync, raw
schema, guard, following owner, dependencies, or TensorFlow behavior. Validate
sequentially, then commit and push only; do not create or update a pull
request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration and indexed SiNet Concat/Resize owner:
  `65 passed, 1 xfailed`
- expanded broad related gate: `632 passed, 1 xfailed in 28.51s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final SiNet Concat/Resize
reconciliation contract; there are no unexpected failures.

## Primary final SiNet Concat/Resize implementation checkpoint

The primary path now initializes
`_final_sinet_concat_resize_static_shape_stats` with both stable zero keys and
stores the opt-in complete result under the unchanged positive rewrite guard.
No sort is added, and the following high-rank BatchMatMul owner retains its
position.

Matching, transactional plan application, rewiring, metadata, pruning,
layout-state sync, raw result schema, dependencies, and TensorFlow-free
behavior are unchanged. The first broad run exposed one stale architecture
assumption that all six final SiNet guards immediately follow stats and discard
reconciliation results; that contract now distinguishes only this completed
owner while retaining the legacy shape for the other five.

Implementation validation completed sequentially under `uv`:

- terminal orchestration and indexed owner: `66 passed`
- targeted owner/architecture contract: `68 passed, 256 deselected`
- expanded broad related gate: `633 passed in 28.54s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_sinet_deep_skip_stats` for counter/cleanup completeness
before retaining its guarded reconciliation result. Keep the following SiNet
Concat/Resize owner boundary fixed. Commit and push only; do not create or
update a pull request.

## Remaining primary final SiNet reconciliation characterization checkpoint

The five preceding final SiNet owners—late residual, deep-skip pre-add fan-out,
deep-skip dual Resize, shared-post PReLU fan-out, and deep-skip Concat/Resize
tail—use indexed transactional plans. Each exact counter increments only after
preflight revalidation and mutation. Pruning and optional layout-state sync are
inside the same positive predicate; preflight, unsafe, stale-plan, capped, and
second-run zero results are true no-ops. Their dedicated suites cover numeric
parity, transactionality, idempotency, fan-out, constant cloning, GraphIndex,
and layout-state consistency.

A single strict expected-failure orchestration contract requires stable
two-key results for all five existing positive guards and assigns each opt-in
complete reconciliation result. No sort is added. The order from late residual
through final SiNet Concat/Resize remains exact.

At implementation, add only result plumbing for these five callers. Do not
change any owner, match/plan logic, rewiring, metadata, pruning, layout sync,
raw schema, guard, ordering, dependencies, or TensorFlow behavior. Validate all
five dedicated suites sequentially, then commit and push only; do not create or
update a pull request.

Characterization validation completed sequentially under `uv`:

- terminal orchestration plus all five dedicated SiNet owner suites:
  `464 passed, 1 xfailed in 1.67s`
- expanded broad related gate: `1089 passed, 1 xfailed in 29.28s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet five-owner result contract;
there are no unexpected failures.

## Remaining primary final SiNet reconciliation implementation checkpoint

The primary path now initializes stable two-key static-shape results for late
residual, pre-add fan-out, dual Resize, shared-post fan-out, and deep-skip, and
stores each opt-in complete result under its unchanged positive guard. No sort
or additional scan is added, and the exact owner order remains unchanged.

All five transactional owners, matching/planning, rewiring, metadata, pruning,
layout-state sync, raw schemas, dependencies, and TensorFlow-free behavior are
unchanged. The shared architecture contract now requires complete results for
all six final SiNet owners.

Implementation validation completed sequentially under `uv`:

- all five owner suites plus orchestration/architecture selection:
  `470 passed, 253 deselected in 3.57s`
- expanded broad related gate: `1090 passed in 29.25s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, continue the remaining primary final-pass reconciliation inventory
immediately before this SiNet cluster, beginning with
`final_consecutive_reshape_stats`. Confirm its multi-counter and cleanup
completeness before changing the caller. Commit and push only; do not create or
update a pull request.

## Primary final consecutive-Reshape reconciliation characterization checkpoint

The absolute-final consecutive-Reshape runner has three returned mutation
counters: no-op removal, ordinary consecutive-chain bypass/removal, and the
fan-out bypass subset. Every input/output rewire or operator removal increments
at least one of those counters. Unused-tensor pruning and optional layout-state
sync are inside the same positive `rewritten > 0 or removed_noop > 0` predicate;
preflight, no-candidate, unsafe, dynamic-shape, and preserved-semantic-rank
paths are true no-ops. The existing aggregate guard is therefore complete even
though the fan-out subset is intentionally represented by two positive keys.

A strict expected-failure orchestration contract now requires a stable two-key
`_final_consecutive_reshape_static_shape_stats` value and assigns the opt-in
complete reconciliation result under the unchanged aggregate guard. It also
fixes the immediately following `final_sinet_late_residual_stats` boundary.

At implementation, add only this result plumbing. Do not change the runner,
counter schema, matching, rewiring, operator removal, pruning, layout sync,
aggregate guard, pass order, dependencies, or TensorFlow behavior. Validate the
consecutive-Reshape owner and terminal orchestration sequentially, then commit
and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- consecutive-Reshape owner, terminal orchestration, and architecture:
  `281 passed, 1 xfailed in 17.27s`
- expanded broad related gate: `1093 passed, 1 xfailed in 29.50s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final consecutive-Reshape
reconciliation-result contract; there are no unexpected failures.

## Primary final consecutive-Reshape reconciliation implementation checkpoint

The primary path now initializes
`_final_consecutive_reshape_static_shape_stats` with the stable legacy-plus-
complete two-key schema and replaces it with the opt-in complete reconciliation
result only under the unchanged three-counter aggregate guard. This adds no
scan: reconciliation already ran on the same positive path.

The runner, its returned counter schema, matching, rewiring, operator removal,
positive-only pruning/layout synchronization, guard expression, pass order,
dependencies, and TensorFlow-free behavior are unchanged. The strict
characterization contract is now a normal passing contract, and the existing
architecture guard has been updated only for the retained result assignment.

Implementation validation completed sequentially under `uv`:

- consecutive-Reshape owner, terminal orchestration, and architecture:
  `282 passed in 18.99s`
- expanded broad related gate: `1094 passed in 29.69s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit `final_prelu_stats` and its tensor-count-assisted guard for
counter and cleanup completeness before retaining its guarded reconciliation
result. Keep the following consecutive-Reshape boundary fixed. Commit and push
only; do not create or update a pull request.

## Primary final PReLU reconciliation characterization checkpoint

The absolute-final PReLU owner has one exact rewrite counter, but intentionally
runs unused-tensor pruning on every invocation to preserve its legacy cleanup
contract. A zero-rewrite call can therefore mutate the tensor table. The caller
already samples `final_prelu_tensor_count` and combines the rewrite counter with
a clamped-by-comparison net tensor reduction condition. This guard covers both
rewrite and cleanup-only paths; stale, unsafe, capped, and idempotent calls with
no removable tensor remain true no-ops.

A strict expected-failure orchestration contract now requires a stable two-key
`_final_prelu_static_shape_stats` value and assigns the opt-in complete
reconciliation result under that unchanged counter-or-tensor-delta guard. It
also fixes the immediately following `final_consecutive_reshape_stats`
boundary.

At implementation, add only this result plumbing. Do not change the owner,
counter schema, matching/planning, rewiring, alpha handling, pruning, layout
sync, tensor-count sample, guard, ordering, dependencies, or TensorFlow
behavior. Validate the indexed PReLU owner, terminal orchestration, and
architecture sequentially, then commit and push only; do not create or update a
pull request.

Characterization validation completed sequentially under `uv`:

- indexed PReLU owner, terminal orchestration, and architecture:
  `297 passed, 1 xfailed in 17.16s`
- expanded broad related gate: `1123 passed, 1 xfailed in 29.66s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final PReLU reconciliation-
result contract; there are no unexpected failures.

## Primary final PReLU reconciliation implementation checkpoint

The primary path now initializes `_final_prelu_static_shape_stats` with the
stable two-key schema and replaces it with the opt-in complete reconciliation
dictionary only under the unchanged rewrite-or-tensor-reduction guard. This
adds no scan because reconciliation already ran on that positive path.

The indexed owner, raw counter, matching/planning, alpha copy-on-write and
metadata handling, unconditional prune/layout synchronization, tensor-count
sample, guard expression, pass order, dependencies, and TensorFlow-free
behavior are unchanged. The strict characterization contract is now a normal
passing contract, and the existing architecture guard checks the retained
result assignment.

Implementation validation completed sequentially under `uv`:

- indexed PReLU owner, terminal orchestration, and architecture:
  `298 passed in 19.18s`
- expanded broad related gate: `1124 passed in 30.38s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_sinet_shuffle_stats` plus
`final_se_fc_stats`/`final_gather_stats` aggregate and its tensor-count-assisted
guard for complete rewrite and cleanup evidence before retaining its guarded
reconciliation result. Keep the following final PReLU boundary fixed. Commit
and push only; do not create or update a pull request.

## Primary final SE/FC/Gather reconciliation characterization checkpoint

The absolute-final aggregate combines three exact rewrite counters. The
indexed SiNet shuffle owner prunes and syncs layout only after a positive
transactional rewrite. The SE/FC and Gather callbacks preserve legacy pruning
on every candidate execution and can therefore reduce the tensor table with a
zero rewrite counter. Their orchestration runner returns only the two ordered
child dictionaries and performs no independent cleanup. The existing
`final_se_fc_gather_tensor_count` sample plus the three-counter sum therefore
covers every ModelIR rewrite and cleanup-only deletion.

The recursive fallback already stages a stable complete reconciliation result
under the same contract. A strict expected-failure main-path contract now
requires the symmetric `_final_se_fc_gather_static_shape_stats` value and
assigns the opt-in complete result under the unchanged aggregate guard. It also
fixes the following `final_prelu_tensor_count` boundary.

At implementation, add only main-path result plumbing. Do not change any
owner, counter schema, orchestration order, matching/planning, rewiring,
pruning, layout sync, tensor-count sample, guard, dependencies, fallback path,
or TensorFlow behavior. Validate the three owners, orchestration, terminal
contract, and architecture sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- SiNet shuffle, SE/FC, Gather, orchestration, core guard, and architecture:
  `542 passed, 1 xfailed in 18.76s`
- expanded broad related gate: `1135 passed, 1 xfailed in 30.04s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet primary final aggregate
reconciliation-result contract; there are no unexpected failures.

## Primary final SE/FC/Gather reconciliation implementation checkpoint

The main path now initializes `_final_se_fc_gather_static_shape_stats` with the
same stable two-key schema as the recursive fallback and replaces it with the
opt-in complete reconciliation dictionary only under the unchanged three-
counter-or-tensor-reduction guard. No reconciliation, traversal, or invocation
is added.

The three owners, raw result schemas, orchestration order, matching/planning,
rewiring, positive and zero-rewrite pruning behavior, layout synchronization,
tensor-count sample, guard expression, fallback path, dependencies, and
TensorFlow-free behavior are unchanged. The strict main-path contract is now a
normal passing contract, and the shared boundary test requires symmetric main
and fallback result assignments.

Implementation validation completed sequentially under `uv`:

- SiNet shuffle, SE/FC, Gather, orchestration, core guard, and architecture:
  `543 passed in 17.93s`
- expanded broad related gate: `1136 passed in 30.02s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_placeholder_matmul_stats` conditional
block, including its first shape result and nested exact/singleton binary
repairs, before changing any retained evidence. Keep the following final
SE/FC/Gather boundary fixed. Commit and push only; do not create or update a
pull request.

## Primary final placeholder-MatMul reconciliation characterization checkpoint

The placeholder restore owner increments its exact counter for every MatMul
input rewire and Reshape removal, and prunes only after a positive restore. The
first static-shape reconciliation then runs unconditionally inside that
positive outer guard. Exact binary repair preserves legacy unconditional
pruning, while singleton-broadcast repair prunes only after a positive repair;
the existing inner tensor-count sample captures cleanup-only deletion.

The first reconciliation's legacy output-shape counter participates in the
inner second-reconciliation guard. Passing its opt-in complete dictionary
directly to `_stats_have_positive_count()` would also treat parameter-only
mutations as a new reason for the second scan and would therefore change the
current guard. A strict expected-failure contract instead requires stable
`_final_placeholder_matmul_static_shape_stats` and
`_final_placeholder_binary_static_shape_stats` results, while projecting only
the legacy output-shape key into `final_placeholder_reconcile_stats`. The exact
inner guard text and following final SE/FC/Gather boundary remain fixed.

At implementation, add only these two result assignments and the in-memory
legacy projection. Do not add a reconciliation, broaden the inner guard, or
change restore/binary owners, matching, rewiring, pruning, sorting, tensor-count
sampling, pass order, dependencies, fallback behavior, or TensorFlow behavior.
Validate dynamic-Reshape restore, both binary owners, terminal orchestration,
core runtime guards, and architecture sequentially, then commit and push only;
do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- restore, binary adapters, terminal orchestration, core guard, and architecture:
  `381 passed, 1 xfailed in 18.44s`
- expanded broad related gate: `1180 passed, 1 xfailed in 29.75s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet two-result placeholder
reconciliation contract; there are no unexpected failures.

## Primary final placeholder-MatMul reconciliation implementation checkpoint

The primary path now initializes stable complete results for the restore
reconciliation and nested binary-repair reconciliation. After a positive
restore, the first existing reconciliation opts into complete accounting. Its
legacy output-shape count is projected into the original one-key
`final_placeholder_reconcile_stats` dictionary, so the exact existing
`_stats_have_positive_count(...)` guard is preserved. The second existing
reconciliation stores its complete result under that unchanged guard.

No reconciliation, graph traversal, or binary-owner invocation is added. The
restore and binary owners, raw counter schemas, matching, rewiring, pruning,
tensor-count sample, inner condition, topological sort, pass order, fallback
path, dependencies, and TensorFlow-free behavior are unchanged. Runtime tests
confirm the second reconciliation invocation count remains identical for
unchanged, first-shape, exact-binary, singleton-binary, and cleanup-only cases.

Implementation validation completed sequentially under `uv`:

- restore, binary adapters, terminal orchestration, core guard, and architecture:
  `382 passed in 19.28s`
- expanded broad related gate: `1181 passed in 30.44s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_mixed_singleton_concat_stats` owner and
guard before retaining its reconciliation result. Keep the following
placeholder-MatMul block fixed. Commit and push only; do not create or update a
pull request.

## Primary final mixed-singleton Concat characterization checkpoint

The indexed mixed-singleton Concat owner builds all plans before applying them,
then increments its single counter for every successfully inserted Reshape
adapter and Concat rewire plan. Unused-tensor pruning and optional layout-state
sync run only after a positive count. No-Concat, already-NHWC, unsafe/dynamic,
clone-failure, unbound, and second-run zero results are true ModelIR no-ops.

A strict expected-failure orchestration contract now requires stable two-key
`_final_mixed_singleton_concat_static_shape_stats` evidence and assigns the
opt-in complete reconciliation result under the unchanged single-counter guard.
It also fixes the following `final_placeholder_matmul_stats` boundary.

At implementation, add only caller-side result plumbing. Do not change the
owner, counter schema, planning/application, adapter construction, rewiring,
pruning, layout sync, guard, ordering, dependencies, placeholder block, or
TensorFlow behavior. Validate the indexed owner, terminal orchestration, core
runtime guard, and architecture sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- indexed owner, terminal orchestration, core guard, and architecture:
  `356 passed, 1 xfailed in 17.83s`
- expanded broad related gate: `1212 passed, 1 xfailed in 29.97s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final mixed-singleton result
contract; there are no unexpected failures.

## Primary final mixed-singleton Concat implementation checkpoint

The primary path now initializes
`_final_mixed_singleton_concat_static_shape_stats` with the stable two-key
schema and replaces it with the opt-in complete reconciliation dictionary only
under the unchanged exact repair-counter guard. This adds no reconciliation or
graph traversal.

The indexed owner, raw counter, plan construction/application, adapter names and
metadata, Concat rewiring, positive-only pruning/layout synchronization, guard,
ordering, placeholder block, dependencies, and TensorFlow-free behavior are
unchanged. The strict characterization contract is now a normal passing
contract, and the existing architecture guard checks the retained assignment.

Implementation validation completed sequentially under `uv`:

- indexed owner, terminal orchestration, core guard, and architecture:
  `357 passed in 19.46s`
- expanded broad related gate: `1213 passed in 29.96s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_broadcast_repair_stats` owner and its
reconciliation/sort/layout-inference block before retaining complete evidence.
Keep the following mixed-singleton Concat boundary fixed. Commit and push only;
do not create or update a pull request.

## Primary final broadcast reconciliation characterization checkpoint

The indexed rank-four channelwise broadcast owner increments its exact counter
for every in-place constant rotation or shared-constant clone plus binary-input
rewire. It performs no pruning or topology removal, and unsupported,
already-broadcastable, unknown-layout, unsafe-shape, and second-run zero results
are ModelIR no-ops. The counter therefore completely guards shape follow-up.

A strict expected-failure orchestration contract now requires stable two-key
`_final_broadcast_static_shape_stats` evidence and assigns the opt-in complete
result at the existing first statement of the positive guard. The following
topological sort, layout inference, and `final_mixed_singleton_concat_stats`
boundary remain exact.

At implementation, add only caller-side result plumbing. Do not change the
owner, counter schema, constant clone/rotation, rewiring, guard, reconcile/sort/
infer order, dependencies, fallback behavior, following owner, or TensorFlow
behavior. Validate the binary-layout owner, terminal orchestration, convergence,
and architecture sequentially, then commit and push only; do not create or
update a pull request.

Characterization validation completed sequentially under `uv`:

- binary owner, convergence, terminal orchestration, and architecture:
  `285 passed, 1 xfailed in 17.22s`
- expanded broad related gate: `1213 passed, 1 xfailed in 30.33s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final broadcast reconciliation-
result contract; there are no unexpected failures.

## Primary final broadcast reconciliation implementation checkpoint

The primary path now initializes `_final_broadcast_static_shape_stats` with the
stable two-key schema and replaces it with the opt-in complete reconciliation
dictionary under the unchanged exact repair-counter guard. The existing
reconciliation call remains the first statement in that guard.

The indexed owner, raw counter, constant rotation and clone behavior, input
rewiring, guard, following topological sort and layout inference, mixed-
singleton boundary, dependencies, fallback path, and TensorFlow-free behavior
are unchanged. No reconciliation or graph traversal is added.

Implementation validation completed sequentially under `uv`:

- binary owner, convergence, terminal orchestration, and architecture:
  `286 passed in 17.42s`
- expanded broad related gate: `1214 passed in 30.53s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_instancenorm_repair_stats` owner and its
reconciliation/sort/layout-inference block before retaining complete evidence.
Keep the following broadcast repair boundary fixed. Commit and push only; do
not create or update a pull request.

## Primary final InstanceNorm reconciliation characterization checkpoint

The indexed decomposed-InstanceNorm owner constructs and validates all axes,
constant, and tensor-shape plans for a candidate before applying them. It
increments its exact counter only when at least one plan changes ModelIR,
performs no pruning or topology mutation, and synchronizes layout only after a
positive count. Missing markers, unsafe fan-out/boundaries, invalid constants,
plan failure, already-correct, and second-run zero results are transactional
ModelIR no-ops.

A strict expected-failure orchestration contract now requires stable two-key
`_final_instancenorm_static_shape_stats` evidence and assigns the opt-in
complete result as the existing first statement of the positive guard. The
following sort, layout inference, and `final_broadcast_repair_stats` boundary
remain exact.

At implementation, add only caller-side result plumbing. Do not change the
owner, plan validation/application, counter schema, constant/tensor metadata,
layout sync, guard, reconcile/sort/infer order, dependencies, following owner,
or TensorFlow behavior. Validate the indexed InstanceNorm owner, terminal
orchestration, and architecture sequentially, then commit and push only; do not
create or update a pull request.

Characterization validation completed sequentially under `uv`:

- indexed InstanceNorm owner, terminal orchestration, and architecture:
  `309 passed, 1 xfailed in 18.50s`
- expanded broad related gate: `1251 passed, 1 xfailed in 29.99s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final InstanceNorm
reconciliation-result contract; there are no unexpected failures.

## Primary final InstanceNorm reconciliation implementation checkpoint

The primary path now initializes `_final_instancenorm_static_shape_stats` with
the stable two-key schema and replaces it with the opt-in complete
reconciliation dictionary under the unchanged exact repair-counter guard. The
existing reconciliation remains the first guarded statement.

The indexed owner, plan validation/application, raw counter, constant and tensor
metadata writes, positive-only layout synchronization, guard, following sort
and layout inference, broadcast boundary, dependencies, and TensorFlow-free
behavior are unchanged. No reconciliation or graph traversal is added.

Implementation validation completed sequentially under `uv`:

- indexed InstanceNorm owner, terminal orchestration, and architecture:
  `310 passed in 18.35s`
- expanded broad related gate: `1252 passed in 30.10s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the preceding `final_convinteger_layout_stats` owner and its
reconciliation/sort/layout-inference block before retaining complete evidence.
Keep the following InstanceNorm boundary fixed. Commit and push only; do not
create or update a pull request.

## Primary final ConvInteger reconciliation characterization checkpoint

The ConvInteger owner returns two semantically distinct counters. Channel-last
hint propagation updates provenance metadata and tensor layout annotations but
is self-contained and does not require shape reconciliation or sorting. The
structural repair counter increments for every Conv input rewire and stale
Transpose removal; its positive path also updates chain shapes/layouts, prunes,
and synchronizes `LayoutState`. The existing repair-only guard is therefore the
correct boundary for reconciliation→sort→layout inference and must not be
broadened to hint-only changes.

A strict expected-failure orchestration contract now requires stable two-key
`_final_convinteger_static_shape_stats` evidence under that unchanged repair-
only guard. It explicitly rejects the propagation counter from the condition
and fixes the following `final_instancenorm_repair_stats` boundary.

At implementation, add only caller-side reconciliation result plumbing. Do not
change hint propagation, metadata/layout writes, structural repair, counter
schema, pruning, layout sync, guard, reconcile/sort/infer order, dependencies,
following owner, or TensorFlow behavior. Validate the quantized-layout owner,
terminal orchestration, and architecture sequentially, then commit and push
only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- quantized-layout owner, terminal orchestration, and architecture:
  `276 passed, 1 xfailed in 17.49s`
- expanded broad related gate: `1255 passed, 1 xfailed in 30.17s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final ConvInteger
reconciliation-result contract; there are no unexpected failures.

## Primary final ConvInteger reconciliation implementation checkpoint

The primary path now initializes `_final_convinteger_static_shape_stats` with
the stable two-key schema and replaces it with the opt-in complete
reconciliation dictionary only under the unchanged structural-repair counter
guard. Channel-last hint propagation remains outside that condition.

The owner, both raw counters, hint metadata/layout writes, structural chain
metadata/rewire/removal, positive-only pruning/layout synchronization, guard,
following sort and layout inference, InstanceNorm boundary, dependencies, and
TensorFlow-free behavior are unchanged. No reconciliation or scan is added.

Implementation validation completed sequentially under `uv`:

- quantized-layout owner, terminal orchestration, and architecture:
  `277 passed in 17.28s`
- expanded broad related gate: `1256 passed in 30.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the absolute-final
`_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs()` expression immediately
before the unconditional sort/layout inference and final ConvInteger owner.
Determine whether its exact raw result can be retained without adding a guard
or reconciliation. Commit and push only; do not create or update a pull request.

## Absolute-final dynamic rank-one result characterization checkpoint

The dynamic rank-one Unsqueeze/Reshape-shape owner increments its exact counter
for each shape-parameter metadata/data rewrite or runtime SHAPE/CONCAT pipeline
insertion plus RESHAPE rewire. It performs no pruning and synchronizes layout
only after a positive result. Of its three production occurrences, the very-
late and recursive-fallback calls already retain their raw dictionaries; only
the absolute-final direct call discards its result.

A strict expected-failure orchestration contract now requires that last call to
assign `_absolute_final_dynamic_rank1_stats`. It fixes all three occurrence
counts and preserves the immediately following unconditional topological sort,
layout inference, and `final_convinteger_layout_stats` boundary.

At implementation, change only the last expression to an assignment. Do not
change the owner, counter schema, option/tensor/operator writes, layout sync,
other two occurrences, add a guard or reconciliation, reorder sort/inference,
change dependencies, or affect TensorFlow behavior. Validate dynamic-Reshape,
terminal orchestration, architecture, and occurrence contracts sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- dynamic-Reshape owner, terminal orchestration, and architecture:
  `287 passed, 1 xfailed in 17.91s`
- expanded broad related gate: `1256 passed, 1 xfailed in 30.47s`
- Ruff and `git diff --check`: passed

The sole strict xfail is the deliberately unmet absolute-final raw-result
assignment; there are no unexpected failures.

## Absolute-final dynamic rank-one result implementation checkpoint

The primary path now assigns the unchanged raw result of the absolute-final
dynamic rank-one Unsqueeze/Reshape-shape rewrite to
`_absolute_final_dynamic_rank1_stats`. Together with the existing very-late and
recursive-fallback assignments, all three production occurrences now retain
their exact mutation evidence.

The owner, counter schema, option/tensor/operator mutations, positive-only
layout synchronization, input ModelIR selection, and the following
unconditional topological sort, layout inference, and final ConvInteger owner
are unchanged. No guard, reconciliation, graph traversal, or dependency is
added. The occurrence contract was updated to require the three assignments
and reject any remaining discarded-result expression.

Implementation validation completed sequentially under `uv`:

- dynamic-Reshape owner, terminal orchestration, architecture, and occurrence
  contracts: `311 passed in 18.45s`
- expanded broad related gate: `1257 passed in 30.51s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit result propagation through
`_run_absolute_final_normalization_attention_pass_pair()` and
`run_absolute_final_normalization_attention()`. The shared recovery runner
returns per-pass results, while this boundary currently declares and discards
`None`; characterize the exact caller and ordering contract before changing
it. Keep the following absolute-final dynamic rank-one rewrite fixed. Commit
and push only; do not create or update a pull request.

## Absolute-final normalization/attention result characterization checkpoint

The absolute-final pair invokes flattened-global-normalization Pad propagation
followed by mixed reduction/MirrorPad attention cleanup through one shared
`ModelIRPassStateScope`. Each owner returns a stable mutation dictionary, and
`run_recovery_invocations()` already returns the two callback results in the
declared pass-ID order. The pair-specific runner currently omits that return,
its lowerer helper declares `None`, and the primary path invokes the helper as
a discarded expression.

Strict expected-failure contracts now require ordered result propagation
through all three boundaries: the pair runner returns the two dictionaries,
the helper returns that tuple, and the primary path assigns it to
`_absolute_final_normalization_attention_results`. The adjacent
`_absolute_final_instancenorm_post_bias_stats` and
`_absolute_final_dynamic_rank1_stats` assignments, zero-argument helper call,
pass IDs, callback options, shared state scope, execution order, and
TensorFlow-free behavior remain fixed.

At implementation, add only return/assignment plumbing and accurate tuple type
annotations. Do not add a summarizer, guard, reconciliation, pass execution,
graph traversal, dependency, or metadata write, and do not reorder either pass
or either adjacent boundary. Validate the pair owner, pass-efficiency,
architecture, terminal orchestration, and broad related gates sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- pair owner, pass-efficiency, architecture, and terminal orchestration:
  `315 passed, 2 xfailed in 18.08s`
- expanded broad related gate: `1266 passed, 2 xfailed in 30.29s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The two strict xfails are the deliberately unmet runner-return and lowerer-
capture contracts; there are no unexpected failures.

## Absolute-final normalization/attention result implementation checkpoint

`run_absolute_final_normalization_attention()` now returns the ordered tuple
already produced by `run_recovery_invocations()`. The lowerer helper returns
that same tuple, and the primary path retains it as
`_absolute_final_normalization_attention_results`. The tuple contains the
flattened-global-normalization Pad result followed by the mixed-attention
MirrorPad result in the existing declared pass-ID order.

The two owners, callback arguments, shared `ModelIRPassStateScope`, diagnostics,
pass IDs, execution order, previous InstanceNorm post-bias assignment, and
following dynamic rank-one assignment are unchanged. No summarizer, guard,
reconciliation, pass invocation, graph traversal, metadata write, or dependency
is added.

Implementation validation completed sequentially under `uv`:

- pair owner, pass-efficiency, architecture, and terminal orchestration:
  `317 passed in 20.12s`
- expanded broad related gate: `1268 passed in 31.17s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the discarded dictionaries from
`_realign_dynamic_boundary_shape_signature_map()` and
`_sanitize_static_shape_signature_consistency()` immediately before the
absolute-final affine/normalization/attention sequence. Characterize their
exact mutation schemas and adjacent ordering before retaining any result. Keep
all existing final owners fixed. Commit and push only; do not create or update
a pull request.

## Absolute-final boundary-signature result characterization checkpoint

Dynamic-boundary signature realignment returns the exact count of metadata-map
entries changed. It has three direct primary-path occurrences: the shared-late
and terminal calls already retain their dictionaries, while the call directly
before the absolute-final sequence discards its result. Static-signature
sanitization returns one repair counter plus three dynamic-preservation
counters. Its earlier late call retains the complete dictionary, while its
absolute-final call discards it.

A strict expected-failure orchestration contract now requires assigning only
those two discarded results to `_absolute_final_boundary_signature_stats` and
`_absolute_final_static_signature_stats`. It fixes all occurrence counts and
targets, the realign→sanitize adjacency, and the immediately following
`_absolute_final_affine_post_add_stats` boundary.

At implementation, replace only the two expressions with assignments. Do not
change either owner or schema, metadata/tensor writes, other occurrences, add a
guard, reconciliation, scan, sort, dependency, or TensorFlow behavior, or
reorder the following absolute-final owners. Validate signature-owner,
terminal-orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- signature owner, terminal orchestration, and architecture:
  `294 passed, 1 xfailed in 17.57s`
- expanded broad related gate: `1268 passed, 1 xfailed in 30.76s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet two-result capture contract;
there are no unexpected failures.

## Absolute-final boundary-signature result implementation checkpoint

The absolute-final dynamic-boundary realignment call now retains its unchanged
one-counter dictionary as `_absolute_final_boundary_signature_stats`. The
immediately following static-signature sanitizer retains its complete
four-counter repair/preservation dictionary as
`_absolute_final_static_signature_stats`. All direct production occurrences of
both owners now retain their raw mutation evidence.

The owners, result schemas, metadata/tensor writes, other occurrence targets,
realign→sanitize order, and following absolute-final affine,
normalization/attention, dynamic-rank-one, and layout sequence are unchanged.
No guard, reconciliation, scan, sort, metadata write, dependency, or
TensorFlow behavior is added.

Implementation validation completed sequentially under `uv`:

- signature owner, terminal orchestration, absolute-final orchestration, and
  architecture: `306 passed in 17.88s`
- expanded broad related gate: `1269 passed in 30.78s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the two discarded results inside the guarded
`apply_safe_transpose_reduction_lite_on_no_layout_opt` block immediately before
the completed boundary-signature restore. Characterize their schemas, guarded
execution, and surrounding sort/signature boundaries before retaining any
evidence. Commit and push only; do not create or update a pull request.

## Guarded no-layout final cleanup result characterization checkpoint

When safe transpose reduction is enabled without the full layout optimizer,
the final cleanup reruns SE/FC layout propagation and the indexed constant
affine pre/post owner after final topological normalization, then sorts once
more. Each owner returns a stable one-counter dictionary and already handles
its own graph-index, pruning, and layout-state synchronization. This guarded
occurrence discards both results.

A strict expected-failure orchestration contract now requires assigning the
raw dictionaries to `_no_layout_final_se_fc_stats` and
`_no_layout_final_affine_prepost_stats`. It fixes the guard expression, both
calls and keyword contracts, the preceding and guarded topological sorts, and
the following `_absolute_final_boundary_signature_stats` boundary.

At implementation, replace only the two guarded expressions with assignments.
Do not initialize or consume the variables outside the guard, change either
owner/schema, add a pass, guard, reconciliation, scan, sort, dependency, or
metadata write, alter layout-state/diagnostics handoff, or affect the normal
layout-optimized path or TensorFlow boundary. Validate SE/FC, indexed affine,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- SE/FC, indexed affine, terminal orchestration, architecture, and pass
  efficiency: `416 passed, 1 xfailed in 18.02s`
- expanded broad related gate: `1371 passed, 1 xfailed in 30.72s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet guarded two-result capture
contract; there are no unexpected failures.

## Guarded no-layout final cleanup result implementation checkpoint

Inside the unchanged safe-transpose-reduction fallback guard, the final SE/FC
layout cleanup now retains its raw dictionary as
`_no_layout_final_se_fc_stats`, and the following indexed constant affine
pre/post cleanup retains its raw dictionary as
`_no_layout_final_affine_prepost_stats`. Neither variable is initialized or
consumed outside that conditional path.

The owners, schemas, guarded execution, callback arguments, diagnostics and
layout-state handoff, preceding and guarded topological sorts, following
boundary-signature restore, normal layout-optimized path, dependencies, and
TensorFlow-free behavior are unchanged. No pass, guard, reconciliation, scan,
sort, metadata write, or result summarizer is added.

Implementation validation completed sequentially under `uv`:

- SE/FC, indexed affine, terminal orchestration, architecture, and pass
  efficiency: `417 passed in 17.82s`
- expanded broad related gate: `1372 passed in 30.75s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit result propagation through the final precision cleanup
sequence (`_rewrite_constant_divisors_to_multiplicative_reciprocals()`,
`run_consecutive_mul_constants_cleanup()`, and
`_restore_precision_sensitive_reciprocal_divisions()`) immediately before the
post-progress topological sort. Characterize schemas, exact occurrence counts,
and surrounding order before retaining any evidence. Commit and push only; do
not create or update a pull request.

## Primary final precision-cleanup result characterization checkpoint

The final precision sequence first rewrites eligible floating-point constant
DIV operators to reciprocal MUL, then folds guarded consecutive constant MUL
chains, and finally restores DIV where reciprocal multiplication feeds an
integer cast through a precision-sensitive affine lineage. Each owner returns
a stable one-counter dictionary and performs its own indexed mutation,
pruning, diagnostics, and layout synchronization as applicable.

The lowerer contains two divisor-rewrite calls, three consecutive-Mul cleanup
calls, and two precision-restore calls. A strict expected-failure orchestration
contract selects only the adjacent primary final trio and requires raw result
targets `_final_precision_div_rewrite_stats`,
`_final_precision_consecutive_mul_stats`, and
`_final_precision_div_restore_stats`. Earlier core-cleanup and recursive-
fallback occurrences remain unchanged, as does the following post-progress
description and topological-sort boundary.

At implementation, replace only the three primary final expressions with
assignments. Do not change any owner/schema, other occurrence, callback
argument, diagnostics/layout-state handoff, add a guard, reconciliation, scan,
sort, dependency, metadata write, or TensorFlow behavior, or consume the new
variables. Validate precision, graph cleanup, terminal orchestration,
architecture, pass efficiency, and broad related gates sequentially, then
commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- precision, graph cleanup, terminal orchestration, architecture, and pass
  efficiency: `331 passed, 1 xfailed in 18.06s`
- expanded broad related gate: `1395 passed, 1 xfailed in 31.22s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet final three-result capture
contract; there are no unexpected failures.

## Primary final precision-cleanup result implementation checkpoint

The primary final precision sequence now retains the unchanged divisor-rewrite,
consecutive-Mul-fold, and precision-sensitive divisor-restore dictionaries as
`_final_precision_div_rewrite_stats`,
`_final_precision_consecutive_mul_stats`, and
`_final_precision_div_restore_stats`, respectively.

All three owners and schemas, indexed mutation, pruning, diagnostics and
layout-state synchronization, callback arguments, earlier core-cleanup and
recursive-fallback occurrences, following post-progress description and sort,
dependencies, and TensorFlow-free behavior are unchanged. No guard,
reconciliation, scan, sort, metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- precision, graph cleanup, terminal orchestration, architecture, and pass
  efficiency: `332 passed in 17.67s`
- expanded broad related gate: `1396 passed in 30.84s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the corresponding recursive-fallback precision trio. Its
callbacks operate on `fallback_ir`, omit layout-state handoff, and sit between
the fallback topological sort and unbound-input repair. Characterize those
exact differences before retaining results. Commit and push only; do not create
or update a pull request.

## Recursive-fallback precision-cleanup result characterization checkpoint

The safety fallback runs the same divisor rewrite→consecutive-Mul fold→
precision-sensitive divisor restore sequence over `fallback_ir` immediately
after placeholder-MatMul reconciliation and a topological sort. Unlike the
primary final trio, the direct precision owners receive no layout state; only
the transactional consecutive-Mul runner receives `session.diagnostics`. All
three returned one-counter dictionaries are currently discarded.

A strict expected-failure orchestration contract requires assigning the raw
results to `_fallback_precision_div_rewrite_stats`,
`_fallback_precision_consecutive_mul_stats`, and
`_fallback_precision_div_restore_stats`. The fallback input, exact keyword
contracts, preceding sort, and following `_fallback_unbound_repair_stats`
boundary remain fixed.

At implementation, replace only these three fallback expressions with
assignments. Do not change the owners/schemas, primary final or earlier core-
cleanup occurrences, add layout-state handoff, a guard, reconciliation, scan,
sort, dependency, metadata write, or result consumer, or affect TensorFlow
behavior. Validate safety fallback, precision, graph cleanup, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- safety fallback, precision, graph cleanup, terminal orchestration, and
  architecture: `318 passed, 1 xfailed in 17.85s`
- expanded broad related gate: `1396 passed, 1 xfailed in 31.03s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet fallback three-result capture
contract; there are no unexpected failures.

## Recursive-fallback precision-cleanup result implementation checkpoint

The safety fallback now retains its unchanged divisor-rewrite,
consecutive-Mul-fold, and precision-sensitive divisor-restore dictionaries as
`_fallback_precision_div_rewrite_stats`,
`_fallback_precision_consecutive_mul_stats`, and
`_fallback_precision_div_restore_stats`, respectively.

The owners/schemas, `fallback_ir` input, deliberate absence of layout-state
handoff, consecutive-Mul diagnostics, preceding sort, following unbound-input
repair, primary final and earlier core-cleanup occurrences, dependencies, and
TensorFlow-free behavior are unchanged. No guard, reconciliation, scan, sort,
metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- safety fallback, precision, graph cleanup, terminal orchestration, and
  architecture: `319 passed in 18.49s`
- expanded broad related gate: `1397 passed in 31.63s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the earlier core-cleanup
`run_consecutive_mul_constants_cleanup()` expression after the pseudo-LeakyReLU
and YOLO-decode cleanups. It is the only remaining discarded result among these
three precision owners. Fix its surrounding core-cleanup order before retaining
evidence. Commit and push only; do not create or update a pull request.

## Primary core-cleanup consecutive-Mul result characterization checkpoint

The remaining discarded precision-owner result is the earlier primary
`run_consecutive_mul_constants_cleanup()` call in the core-cleanup phase. It
runs after pseudo-LeakyReLU fusion and YOLO decode Mul-square/anchor cleanup,
and immediately before terminal Transpose/Dequantize sanitization. Its ModelIR,
layout-state, diagnostics, and one-counter schema match the already captured
primary-final occurrence.

A strict expected-failure orchestration contract requires assigning this call
to `_core_cleanup_consecutive_mul_stats`. It fixes both direct primary
occurrences and the existing `_final_precision_consecutive_mul_stats` target,
plus the exact pseudo-LeakyReLU→YOLO→consecutive-Mul→terminal-sanitizer order.
The recursive-fallback occurrence remains independently captured.

At implementation, replace only this one expression with an assignment. Do not
change the owner/schema, any other occurrence or target, callback arguments,
diagnostics/layout-state handoff, add a guard, reconciliation, scan, sort,
dependency, metadata write, or result consumer, or affect TensorFlow behavior.
Validate graph cleanup, terminal orchestration, architecture, pass efficiency,
and broad related gates sequentially, then commit and push only; do not create
or update a pull request.

Characterization validation completed sequentially under `uv`:

- graph cleanup, terminal orchestration, architecture, and pass efficiency:
  `329 passed, 1 xfailed in 17.58s`
- expanded broad related gate: `1397 passed, 1 xfailed in 31.02s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet core-cleanup result capture
contract; there are no unexpected failures.

## Primary core-cleanup consecutive-Mul result implementation checkpoint

The earlier primary core-cleanup call now retains the unchanged transactional
consecutive-Mul dictionary as `_core_cleanup_consecutive_mul_stats`. Together
with `_fallback_precision_consecutive_mul_stats` and
`_final_precision_consecutive_mul_stats`, every production occurrence of this
owner now retains its raw mutation evidence.

The owner/schema, ModelIR, layout-state and diagnostics handoff, preceding
pseudo-LeakyReLU/YOLO cleanups, following terminal Transpose/Dequantize
sanitizer, other occurrence targets, dependencies, and TensorFlow-free behavior
are unchanged. No guard, reconciliation, scan, sort, metadata write, or result
consumption is added.

Implementation validation completed sequentially under `uv`:

- graph cleanup, terminal orchestration, architecture, and pass efficiency:
  `330 passed in 18.00s`
- expanded broad related gate: `1398 passed in 31.86s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the immediately preceding
`_optimize_fuse_pseudo_leakyrelu_chains()` and
`_optimize_yolo_decode_mul_square_anchor_chains()` core-cleanup expressions.
Characterize their schemas, occurrence counts, and fixed progress/consecutive-
Mul boundaries before retaining evidence. Commit and push only; do not create
or update a pull request.

## Core-cleanup pseudo-LeakyReLU/YOLO result characterization checkpoint

The primary core-cleanup phase begins with guarded pseudo-LeakyReLU fusion,
followed by YOLO decode `Mul(x,x)` plus anchor-constant folding. Each owner has
one direct lowerer occurrence and returns a stable one-counter dictionary. The
pseudo-LeakyReLU owner uses an indexed topology/fan-out proof; the YOLO owner
prevalidates finite constants and shared-input/consumer boundaries. Both caller
results are currently discarded.

A strict expected-failure orchestration contract requires assigning the raw
results to `_core_cleanup_pseudo_leakyrelu_stats` and
`_core_cleanup_yolo_decode_stats`. It fixes the `core cleanup passes` progress
boundary, pseudo→YOLO order, exact arguments, and the following captured
`_core_cleanup_consecutive_mul_stats` boundary.

At implementation, replace only these two expressions with assignments. Do not
change either owner/schema, add GraphIndex/layout arguments, alter any matcher,
guard, pruning, add a reconciliation, scan, sort, dependency, metadata write,
or result consumer, reorder the core cleanup, or affect TensorFlow behavior.
Validate indexed pseudo-LeakyReLU, indexed YOLO constant folding, graph cleanup,
terminal orchestration, architecture, pass efficiency, and broad related gates
sequentially, then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- indexed pseudo-LeakyReLU, indexed YOLO folding, graph cleanup, terminal
  orchestration, architecture, and pass efficiency:
  `363 passed, 1 xfailed in 18.51s`
- expanded broad related gate: `1431 passed, 1 xfailed in 31.17s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet two-result capture contract;
there are no unexpected failures.

## Core-cleanup pseudo-LeakyReLU/YOLO result implementation checkpoint

The primary core-cleanup phase now retains the unchanged pseudo-LeakyReLU
fusion and YOLO decode constant-fold dictionaries as
`_core_cleanup_pseudo_leakyrelu_stats` and
`_core_cleanup_yolo_decode_stats`, respectively.

Both owners/schemas, indexed topology and constant safety guards, pruning,
single occurrence counts, `model_ir` arguments, core-progress boundary,
pseudo→YOLO→consecutive-Mul order, dependencies, and TensorFlow-free behavior
are unchanged. No GraphIndex/layout argument, guard, reconciliation, scan,
sort, metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- indexed pseudo-LeakyReLU, indexed YOLO folding, graph cleanup, terminal
  orchestration, architecture, and pass efficiency: `364 passed in 18.37s`
- expanded broad related gate: `1432 passed in 31.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the immediately following
`_sanitize_terminal_transpose_before_dequantize()` and
`run_terminal_quantize_dequantize_cleanup()` expressions. Characterize their
schemas, occurrence counts, exact diagnostics/layout handoff, and surrounding
consecutive-Mul/Conv-affine boundaries before retaining evidence. Commit and
push only; do not create or update a pull request.

## Terminal quantization-cleanup result characterization checkpoint

The primary path contains two identical ordered cleanup pairs: one immediately
after the captured core-cleanup consecutive-Mul result, and one after late
recovery sweeps under the `terminal cleanup passes` progress stage. Each pair
first normalizes or removes terminal Transpose/Dequantize boundaries, returning
two counters, then transactionally removes exact-grid terminal
Quantize/Dequantize pairs, returning one counter. Both pairs precede Conv-affine
folding, and all four caller results are currently discarded.

A strict expected-failure orchestration contract requires targets
`_core_cleanup_terminal_dequant_stats`,
`_core_cleanup_terminal_qdq_stats`,
`_terminal_cleanup_terminal_dequant_stats`, and
`_terminal_cleanup_terminal_qdq_stats`. It fixes both occurrence counts,
pairwise order, ModelIR/layout-state/diagnostics contracts, distinct preceding
boundaries, and both Conv-affine successors.

At implementation, replace only the four expressions with assignments. Do not
change either owner/schema, pass transaction, GraphIndex/pruning behavior,
callback arguments, progress stages, add a guard, reconciliation, scan, sort,
dependency, metadata write, or result consumer, reorder either pair, or affect
TensorFlow behavior. Validate quantization cleanup, terminal orchestration,
architecture, pass efficiency, and broad related gates sequentially, then
commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- quantization cleanup, terminal orchestration, architecture, and pass
  efficiency: `358 passed, 1 xfailed in 17.86s`
- expanded broad related gate: `1479 passed, 1 xfailed in 31.00s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict xfail is the deliberately unmet four-result capture contract;
there are no unexpected failures.

## Terminal quantization-cleanup result implementation checkpoint

The core-cleanup pair now retains its terminal Transpose/Dequantize sanitizer
and exact-grid Q/DQ dictionaries as `_core_cleanup_terminal_dequant_stats` and
`_core_cleanup_terminal_qdq_stats`. The later terminal-cleanup pair retains the
same owner results as `_terminal_cleanup_terminal_dequant_stats` and
`_terminal_cleanup_terminal_qdq_stats`.

Both owners/schemas, pair occurrence counts and order, GraphIndex/pruning and
transaction behavior, ModelIR/layout-state/diagnostics handoff, distinct
preceding progress/consecutive-Mul boundaries, following Conv-affine folds,
dependencies, and TensorFlow-free behavior are unchanged. No guard,
reconciliation, scan, sort, metadata write, or result consumption is added.

Implementation validation completed sequentially under `uv`:

- quantization cleanup, terminal orchestration, architecture, and pass
  efficiency: `359 passed in 17.79s`
- expanded broad related gate: `1480 passed in 31.26s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

At resume, audit the two following
`_optimize_fold_conv_mul_add_affine_chains()` and
`_optimize_fuse_conv_activation_chains()` pairs. Characterize both occurrence
schemas, the shared layout-state/Conv-ADD option contracts, and their fixed Q/DQ
predecessors before retaining evidence. Commit and push only; do not create or
update a pull request.

## Quantization-successor Conv result characterization checkpoint

Each captured terminal quantization-cleanup pair is followed by Conv MUL/ADD
affine folding and Conv/binary activation fusion. The affine owner returns a
four-counter total/category dictionary and has three direct primary-path
occurrences. Activation fusion returns seven Conv/binary/total counters and has
two direct occurrences, both in these pairs. The first two affine results and
both activation results are currently discarded.

A strict expected-failure orchestration contract requires phase-specific
targets `_core_cleanup_conv_affine_stats`,
`_core_cleanup_conv_activation_stats`,
`_terminal_cleanup_conv_affine_stats`, and
`_terminal_cleanup_conv_activation_stats`. It fixes the shared Conv-ADD option
and layout-state contracts, captured Q/DQ predecessors, and distinct dynamic-
Reshape/ArgMax successors. The third later affine expression remains explicitly
unchanged for a separate audit.

At implementation, replace only these four expressions with assignments. Do
not change either owner/schema, the third affine call, callback arguments,
indexed mutation/pruning/layout sync, add a guard, reconciliation, scan, sort,
dependency, metadata write, or result consumer, reorder either pair, or affect
TensorFlow behavior. Validate indexed Conv-affine folding, activation fusion,
terminal convergence/orchestration, architecture, and broad related gates
sequentially, then commit and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused Conv-affine/activation, terminal convergence/orchestration, and
  architecture gate: `321 passed, 1 xfailed in 17.90s`
- expanded broad related gate: `1507 passed, 1 xfailed in 31.15s`

The sole strict expected failure is the intentionally unimplemented four-result
retention contract above.

## Quantization-successor Conv result implementation checkpoint

The two Conv MUL/ADD affine-fold results and the two Conv/binary activation-
fusion results immediately following the captured terminal quantization pairs
are now retained under the four phase-specific targets fixed by
characterization. The pass calls, options, layout-state handoff, pair order,
mutation behavior, Q/DQ predecessors, and dynamic-Reshape/ArgMax successors are
unchanged. No guard, reconciliation, scan, sort, metadata write, or result
consumer was added. The third later Conv-affine expression remains unchanged.

Implementation validation completed sequentially under `uv`:

- focused Conv-affine/activation, terminal convergence/orchestration, and
  architecture gate: `322 passed in 19.48s`
- expanded broad related gate: `1508 passed in 30.41s`

Ruff, Python bytecode compilation, and `git diff --check` passed. At resume,
audit the third direct
`_optimize_fold_conv_mul_add_affine_chains()` occurrence between cost-volume
cleanup and `late_concat_layout_state_scope` before retaining its result.

## Late cost-volume Conv-affine result characterization checkpoint

The third and final direct Conv MUL/ADD affine-fold occurrence uses the same
stable four-counter owner, `enable_conv_add_only_fold=True`, and live
`session.layout_state` contract as the two captured terminal successors. Its
result is currently discarded. It is the raw boundary after the shared
`late_ndhwc_cost_volume_state_scope` pair and before construction of
`late_concat_layout_state_scope`.

A strict expected-failure orchestration contract requires the result under
`_late_cost_volume_conv_affine_stats`. It fixes the two adjacent state-scope
assignments, the NDHWC-gate and cost-volume runner predecessors, and the exact
owner arguments. The two earlier captured affine occurrences remain unchanged.

At implementation, replace only this final expression with an assignment. Do
not change the owner/schema, either state scope, callback arguments, pass order,
indexed/fallback mutation and pruning, layout synchronization, diagnostics,
add a guard, reconciliation, scan, sort, metadata write, result consumer,
dependency, or TensorFlow behavior. Validate the Conv-affine owner, both scope
clusters, pass efficiency, terminal orchestration, architecture, and broad
related gates sequentially, then commit and push only; do not create or update
a pull request.

Characterization validation completed sequentially under `uv`:

- focused Conv-affine, NDHWC/cost-volume, pass-efficiency, terminal-
  orchestration, and architecture gate: `372 passed, 1 xfailed in 18.44s`
- expanded broad related gate: `1556 passed, 1 xfailed in 31.03s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented late
cost-volume result-retention contract above.

## Late cost-volume Conv-affine result implementation checkpoint

The third and final direct Conv MUL/ADD affine-fold result is now retained as
`_late_cost_volume_conv_affine_stats`. This completes raw result capture for all
three production occurrences of the stable four-counter owner. Its exact
options, live layout-state handoff, shared NDHWC/cost-volume predecessor scope,
and following late-Concat scope construction are unchanged.

No owner/schema, callback argument, pass order, indexed/fallback mutation or
pruning, layout synchronization, diagnostic, guard, reconciliation, scan, sort,
metadata write, result consumer, dependency, or TensorFlow behavior changed.

Implementation validation completed sequentially under `uv`:

- focused Conv-affine, NDHWC/cost-volume, pass-efficiency, terminal-
  orchestration, and architecture gate: `373 passed in 18.53s`
- expanded broad related gate: `1557 passed in 32.00s`

At resume, audit the four result schemas and occurrence contracts for
`run_axis3_const_concat_layout_cleanup()`,
`run_dequant_concat_quantize_layout_cleanup()`,
`run_layernorm_statistics_layout_cleanup()`, and
`run_layout_transpose_cleanup()` inside `late_concat_layout_state_scope` before
retaining evidence. Commit and push only; do not create or update a pull
request.

## Late Concat shared-scope result characterization checkpoint

The four runners inside `late_concat_layout_state_scope` return stable mutation
dictionaries: axis-3 constant-Concat and Dequantize/Concat/Quantize each return
one counter, LayerNorm statistics returns two counters, and generic Transpose
cleanup returns five counters including its non-mutating iteration count. All
four results are currently discarded.

A strict expected-failure orchestration contract requires, in order,
`_late_concat_axis3_const_layout_stats`,
`_late_concat_dequant_quantize_layout_stats`,
`_late_concat_layernorm_layout_stats`, and
`_late_concat_transpose_layout_stats`. It fixes the captured late cost-volume
affine predecessor, exact shared-scope/diagnostic/layout arguments, and the
following optimize-layout guard. The other two lowerer Transpose-cleanup
occurrences remain expressions.

At implementation, replace only these four adjacent expressions with
assignments. Do not change any owner/schema, shared scope, callback argument,
pass order, transactional mutation or rollback, GraphIndex/pruning/layout
synchronization, diagnostic, guard, reconciliation, scan, sort, metadata write,
result consumer, dependency, or TensorFlow behavior. Validate all four owners,
the shared-scope efficiency and architecture contracts, terminal orchestration,
and broad related gates sequentially, then commit and push only; do not create
or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused four-owner, shared-scope efficiency, terminal-orchestration, and
  architecture gate: `386 passed, 1 xfailed in 18.03s`
- expanded broad related gate: `1629 passed, 1 xfailed in 31.76s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented four-result
retention contract above.

## Late Concat shared-scope result implementation checkpoint

The four adjacent late Concat runner results are now retained as
`_late_concat_axis3_const_layout_stats`,
`_late_concat_dequant_quantize_layout_stats`,
`_late_concat_layernorm_layout_stats`, and
`_late_concat_transpose_layout_stats`. Their stable one-, one-, two-, and
five-counter dictionaries are preserved without projection or aggregation.

The shared `late_concat_layout_state_scope`, exact callback arguments,
transactional mutation/rollback, GraphIndex/pruning/layout synchronization,
diagnostics, pass order, captured affine predecessor, following optimize-layout
guard, and two other lowerer Transpose-cleanup occurrences are unchanged. No
guard, reconciliation, scan, sort, metadata write, result consumer, dependency,
or TensorFlow behavior changed.

Implementation validation completed sequentially under `uv`:

- focused four-owner, shared-scope efficiency, terminal-orchestration, and
  architecture gate: `387 passed in 17.89s`
- expanded broad related gate: `1630 passed in 32.11s`

At resume, audit both guarded production occurrences of
`_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains()` and their
one-counter schema before retaining their results. Commit and push only; do not
create or update a pull request.

## Guarded elementwise-fanout result characterization checkpoint

The elementwise NHWC→NCHW fanout-roundtrip owner returns the stable one-counter
dictionary
`optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains`. Its two
production occurrences are both guarded by
`optimize_layout_transpose_chains`, and both results are currently discarded.

A strict expected-failure orchestration contract requires
`_late_concat_elementwise_fanout_stats` after the captured late-Concat
Transpose dictionary and `_terminal_elementwise_fanout_stats` before terminal
singleton-MaxPool/Reshape orchestration. It fixes both guards, the model-only
callback contract, and their distinct preceding/following boundaries.

At implementation, replace only the two guarded expressions with assignments.
Do not initialize values outside the guards, consume the results, or change the
owner/schema, guards, callback arguments, pass order, rollback/pruning,
metadata, dependencies, TensorFlow behavior, or any surrounding owner. Validate
the owner, terminal singleton boundary, terminal orchestration, architecture,
and broad related gates sequentially, then commit and push only; do not create
or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused elementwise-fanout owner, terminal-singleton boundary, terminal-
  orchestration, layout-recovery, and architecture gate:
  `298 passed, 1 xfailed in 19.61s`
- expanded broad related gate: `1644 passed, 1 xfailed in 33.76s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented guarded
two-result retention contract above.

## Guarded elementwise-fanout result implementation checkpoint

Both production results are now retained inside their existing guards as
`_late_concat_elementwise_fanout_stats` and
`_terminal_elementwise_fanout_stats`. Their unchanged one-counter dictionaries
remain unavailable when `optimize_layout_transpose_chains` is false; no
guard-external default or result consumer was added.

The owner/schema, model-only callback contract, guards, pass order,
rollback/pruning, metadata, dependencies, and four surrounding boundaries are
unchanged. TensorFlow behavior is unaffected.

Implementation validation completed sequentially under `uv`:

- focused elementwise-fanout owner, terminal-singleton boundary, terminal-
  orchestration, layout-recovery, and architecture gate:
  `299 passed in 19.15s`
- expanded broad related gate: `1645 passed in 32.11s`

At resume, audit the adjacent
`_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains()` and
`_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains()` result
schemas and occurrence boundaries before retaining evidence.

Workflow policy: pull request #955 was closed at the user's direction. Do not
create, reopen, or update any pull request. Future units must use commits and
pushes only.

## Late ExpandDims/flatten-HW result characterization checkpoint

The adjacent ExpandDims and flatten-HW Transpose/Reshape compatibility owners
each return a stable one-counter dictionary, each has exactly one production
occurrence, and both receive `model_ir` plus the live
`session.layout_state`. Their results are currently discarded.

A strict expected-failure orchestration contract requires
`_late_expanddims_reshape_layout_stats` followed by
`_late_flatten_hw_reshape_layout_stats`. It fixes the captured guarded
late-Concat fanout predecessor, exact adjacency and callback contracts, and the
following reshape/transpose-to-NHWC-Reshape owner.

At implementation, replace only these two expressions with assignments. Do not
change either compatibility/indexed owner or schema, fallback behavior,
GraphIndex/pruning/layout synchronization, callback arguments, pass order,
guard, reconciliation, scan, sort, metadata write, result consumer, dependency,
or TensorFlow behavior. Validate both indexed/compatibility owners, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed/compatibility owner, terminal-orchestration, and architecture
  gate: `300 passed, 1 xfailed in 18.25s`
- expanded broad related gate: `1660 passed, 1 xfailed in 32.09s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented adjacent
two-result retention contract above.

## Late ExpandDims/flatten-HW result implementation checkpoint

The two adjacent compatibility-owner dictionaries are now retained as
`_late_expanddims_reshape_layout_stats` and
`_late_flatten_hw_reshape_layout_stats`. Their stable one-counter schemas,
indexed-first/fallback behavior, GraphIndex/pruning/layout synchronization,
live Session LayoutState, exact order, and outer boundaries are unchanged.

No guard, reconciliation, scan, sort, metadata write, result consumer,
dependency, or TensorFlow behavior changed.

Implementation validation completed sequentially under `uv`:

- focused indexed/compatibility owner, terminal-orchestration, and architecture
  gate: `301 passed in 18.71s`
- expanded broad related gate: `1661 passed in 32.15s`

At resume, audit the immediately following
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains()` result,
its owner schema, and its captured predecessor/successor boundaries. Commit and
push only; do not create, reopen, or update a pull request.

## Late NHWC-Reshape collapse result characterization checkpoint

The private rank-three reshape/transpose layout-shim collapse owner returns the
stable one-counter dictionary
`optimized_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains`. It
accepts only `model_ir`, owns one direct production occurrence, and currently
discards that result.

A strict expected-failure orchestration contract requires
`_late_nhwc_reshape_collapse_stats`. It fixes the captured flatten-HW
predecessor, model-only callback contract, and following channel-shuffle/Gather
cluster invocation with both optional shuffle families disabled.

At implementation, replace only this expression with an assignment. Do not
change the owner/schema, GraphIndex mutation/pruning, callback arguments, pass
order, neighboring targets or cluster options, add a guard, reconciliation,
scan, sort, metadata write, result consumer, dependency, or TensorFlow behavior.
Validate the owner, channel-shuffle boundary, layout recovery, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused collapse-owner, channel-shuffle boundary, layout-recovery, terminal-
  orchestration, and architecture gate: `364 passed, 1 xfailed in 18.71s`
- expanded broad related gate: `1733 passed, 1 xfailed in 33.34s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented single-
result retention contract above.

## Late NHWC-Reshape collapse result implementation checkpoint

The single production result is now retained as
`_late_nhwc_reshape_collapse_stats`. Its unchanged one-counter dictionary is
captured without projection or consumption. The owner-internal GraphIndex,
positive-only pruning, model-only callback contract, captured flatten-HW
predecessor, and channel-shuffle/Gather successor options remain fixed.

No guard, reconciliation, scan, sort, metadata write, dependency, or TensorFlow
behavior changed. An initially misapplied AST-test edit was reverted before the
focused gate; no unrelated production or test contract remains changed.

Implementation validation completed sequentially under `uv`:

- focused collapse-owner, channel-shuffle boundary, layout-recovery, terminal-
  orchestration, and architecture gate: `365 passed in 20.07s`
- expanded broad related gate: `1734 passed in 32.13s`

At resume, audit the return schema of `run_channel_shuffle_gather()`, the local
`_run_channel_shuffle_gather_layout_pass_cluster()` helper's current `None`
contract, all helper invocations/policy combinations, and the late base-policy
boundary before propagating and retaining its result. Commit and push only; do
not create, reopen, or update a pull request.

## Channel-shuffle/Gather result propagation characterization checkpoint

`run_channel_shuffle_gather()` selects two through seven transactional child
passes from three independent policy flags. `run_recovery_invocations()`
already produces their ordered dictionaries, but the runner discards that
tuple. The local `_run_channel_shuffle_gather_layout_pass_cluster()` helper is
also annotated `None` and discards the runner result. Its two production calls
therefore expose no mutation evidence.

Two strict expected-failure contracts require the runner to return every
policy's exact ordered tuple and the helper to propagate it as
`Tuple[Dict[str, int], ...]`. The guarded full-post call must retain
`_layout_opt_channel_shuffle_gather_results`; the late base-only call must
retain `_late_channel_shuffle_gather_results`. Existing policy arguments,
shared-scope construction, the captured NHWC-Reshape predecessor, and following
QKV-attention boundary are fixed.

At implementation, return the already-created tuple through both layers and
replace only the two invocation expressions with assignments. Do not aggregate
or consume results, alter policy selection, callback order/arguments,
transactional behavior, shared scope, diagnostics, guards, boundaries,
dependencies, or TensorFlow behavior. Validate all eight policies, helper AST,
shared-state efficiency, boundaries, architecture, and broad related gates
sequentially, then commit and push only; do not create, reopen, or update a pull
request.

Characterization validation completed sequentially under `uv`:

- focused all-policy runner, helper AST, pass-efficiency, layout-recovery,
  terminal-orchestration, and architecture gate:
  `346 passed, 2 xfailed in 17.79s`
- expanded broad related gate: `1734 passed, 2 xfailed in 32.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The two strict expected failures are the deliberately unimplemented runner/
helper return propagation and the two production result assignments.

## Channel-shuffle/Gather result propagation implementation checkpoint

`run_channel_shuffle_gather()` now returns the existing ordered child tuple as
`Tuple[Dict[str, int], ...]`, and the local helper propagates the same type and
value. No child invocation, scan, copy, aggregation, or summary traversal was
added. The guarded full-post and unguarded late-base results are retained as
`_layout_opt_channel_shuffle_gather_results` and
`_late_channel_shuffle_gather_results`.

All eight policy combinations preserve exact selected-pass order. Shared state
scope, layout/diagnostic arguments, transaction behavior, guard placement,
captured NHWC-Reshape predecessor, pre-ADD/mean-attention and QKV-attention
successors, dependencies, and TensorFlow behavior are unchanged.

Implementation validation completed sequentially under `uv`:

- focused all-policy runner, helper AST, pass-efficiency, layout-recovery,
  terminal-orchestration, and architecture gate: `348 passed in 19.69s`
- expanded broad related gate: `1736 passed in 31.42s`

At resume, audit the immediately following
`_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains()`
result, owner schema, live LayoutState contract, and channel-shuffle/next-owner
boundaries before retaining evidence. Commit and push only; do not create,
reopen, or update a pull request.

## Late attention-QKV Reshape result characterization checkpoint

The attention-QKV Reshape/Transpose compatibility owner returns the stable
one-counter dictionary
`optimized_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains`.
It uses indexed-first/fallback handling, has one direct production occurrence,
and receives the live Session LayoutState. Its result is currently discarded.

A strict expected-failure orchestration contract requires
`_late_attention_qkv_reshape_stats`. It fixes the captured base-only
channel-shuffle/Gather tuple predecessor, exact ModelIR/LayoutState callback,
and following attention Gather/Transpose/Reshape cleanup owner.

At implementation, replace only this expression with an assignment. Do not
change the compatibility/indexed owner or schema, fallback behavior,
GraphIndex/pruning/layout synchronization, callback arguments, pass order,
neighboring targets, add a guard, reconciliation, scan, sort, metadata write,
result consumer, dependency, or TensorFlow behavior. Validate the indexed/
compatibility owner, channel-shuffle and layout-recovery boundaries, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed/compatibility owner, channel-shuffle/layout-recovery
  boundaries, terminal-orchestration, and architecture gate:
  `325 passed, 1 xfailed in 18.76s`
- expanded broad related gate: `1744 passed, 1 xfailed in 33.31s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented single-
result retention contract above.

## Late attention-QKV Reshape result retention implementation checkpoint

The sole production call now retains the existing one-counter dictionary as
`_late_attention_qkv_reshape_stats`. This is an assignment-only orchestration
change: the compatibility/indexed owner, return schema, fallback behavior,
GraphIndex and pruning behavior, live Session LayoutState argument, pass order,
guards, neighboring calls, dependencies, and TensorFlow behavior are unchanged.
The retained value has no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused indexed/compatibility owner, channel-shuffle/layout-recovery
  boundaries, terminal-orchestration, and architecture gate:
  `326 passed in 19.68s`
- branch-changed broad related suite plus indexed QKV, layout recovery, and
  pass-efficiency coverage: `1379 passed in 23.78s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_attention_gather_transpose_reshape_cleanup_chains()` result, owner
schema, production occurrences, and QKV/Gather-axis0 boundaries before adding
characterization. Commit and push only; do not create, reopen, or update a pull
request.

## Late attention Gather cleanup result characterization checkpoint

The attention Gather/Transpose/Reshape cleanup owner returns the stable
two-counter dictionary for pattern A and pattern B rewrites. It accepts only the
ModelIR. The owner is selected once by the existing attention-recovery runner
and has one additional direct late production call; only the latter currently
discards its result.

A strict expected-failure orchestration contract requires
`_late_attention_gather_cleanup_stats` for that direct call. It fixes the
captured QKV dictionary predecessor, exact model-only callback, and following
Gather-axis0 compatibility call with its live Session LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, two-key schema, recovery-runner selection or captured
results, GraphIndex/pruning behavior, pass order, callback arguments, neighbor
targets, guards, dependencies, or TensorFlow behavior. Validate the cleanup
owner, QKV/Gather-axis0 boundaries, terminal orchestration, architecture, and
broad related gates sequentially, then commit and push only; do not create,
reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused cleanup owner, QKV/Gather-axis0 boundaries, layout recovery,
  terminal-orchestration, and architecture gate:
  `436 passed, 1 xfailed in 18.94s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  layout recovery, and pass-efficiency coverage:
  `1513 passed, 1 xfailed in 23.86s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late attention Gather cleanup result retention implementation checkpoint

The sole direct late production call now retains the existing two-counter
dictionary as `_late_attention_gather_cleanup_stats`. This is an
assignment-only orchestration change. The owner and schema, separate recovery-
runner selection and captured results, GraphIndex/pruning behavior, callback
arguments, pass order, QKV predecessor, live-LayoutState Gather-axis0
successor, dependencies, and TensorFlow behavior are unchanged. The value has
no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused cleanup owner, QKV/Gather-axis0 boundaries, layout recovery,
  terminal-orchestration, and architecture gate: `437 passed in 19.04s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  layout recovery, and pass-efficiency coverage: `1514 passed in 24.34s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_gather_axis0_singleton_to_reshape_input_chains()` result, owner
schema, live LayoutState contract, production occurrences, and cleanup/
preprojection boundaries before adding characterization. Commit and push only;
do not create, reopen, or update a pull request.

## Late Gather-axis0 Reshape result characterization checkpoint

The Gather-axis0 singleton-to-Reshape compatibility owner returns the stable
one-counter dictionary
`optimized_gather_axis0_singleton_to_reshape_input_chains`. It uses the indexed
GraphIndex implementation, receives the live Session LayoutState, is selected
once by the existing attention-recovery runner, and has one additional direct
late production call. Only that direct call currently discards its result.

A strict expected-failure orchestration contract requires
`_late_gather_axis0_reshape_stats` for the direct call. It fixes the captured
attention-cleanup predecessor, exact ModelIR/LayoutState callback, and following
model-only attention-preprojection rank-lift owner.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/layout synchronization, pass order, callback arguments,
neighbor targets, guards, dependencies, or TensorFlow behavior. Validate the
Gather owner, cleanup/preprojection boundaries, layout recovery, terminal
orchestration, architecture, and broad related gates sequentially, then commit
and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused Gather owner, cleanup/preprojection boundaries, layout recovery,
  terminal-orchestration, and architecture gate:
  `495 passed, 1 xfailed in 18.96s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, layout recovery, and pass-efficiency coverage:
  `1580 passed, 1 xfailed in 24.26s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late Gather-axis0 Reshape result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_gather_axis0_reshape_stats`. This is an assignment-only
orchestration change. The indexed owner and schema, separate recovery-runner
selection and captured results, GraphIndex/layout synchronization, callback
arguments, pass order, attention-cleanup predecessor, preprojection successor,
dependencies, and TensorFlow behavior are unchanged. The value has no consumer
and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused Gather owner, cleanup/preprojection boundaries, layout recovery,
  terminal-orchestration, and architecture gate: `496 passed in 18.93s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, layout recovery, and pass-efficiency coverage:
  `1581 passed in 24.33s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains()` result,
owner schema, production occurrences, and Gather/window-partition boundaries
before adding characterization. Commit and push only; do not create, reopen, or
update a pull request.

## Late attention preprojection result characterization checkpoint

The attention-preprojection Reshape-to-BatchMatMul rank-lift owner returns the
stable one-counter dictionary
`optimized_attention_preproj_reshape_to_batchmatmul_ranklift_chains`. It accepts
only the ModelIR, is selected once by the existing attention-recovery runner,
and has one additional direct late production call. Only that direct call
currently discards its result.

A strict expected-failure orchestration contract requires
`_late_attention_preproj_ranklift_stats` for the direct call. It fixes the
captured Gather-axis0 predecessor, exact model-only callback, and following
window-partition callback with its live Session LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/pruning behavior, pass order, callback arguments, neighbor
targets, guards, dependencies, or TensorFlow behavior. Validate the
preprojection owner, Gather/window-partition boundaries, layout recovery,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused preprojection owner, Gather/window-partition boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `450 passed, 1 xfailed in 18.84s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window partition, layout recovery, and pass-efficiency
  coverage: `1632 passed, 1 xfailed in 24.26s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late attention preprojection result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_attention_preproj_ranklift_stats`. This is an assignment-
only orchestration change. The owner and schema, separate recovery-runner
selection and captured results, GraphIndex/pruning behavior, callback
arguments, pass order, Gather-axis0 predecessor, live-LayoutState window-
partition successor, dependencies, and TensorFlow behavior are unchanged. The
value has no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused preprojection owner, Gather/window-partition boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `451 passed in 18.88s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window partition, layout recovery, and pass-efficiency
  coverage: `1633 passed in 24.31s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_window_partition_reshape_transpose_to_space_to_depth_chains()`
result, owner schema, live LayoutState contract, production occurrences, and
preprojection/window-reverse boundaries before adding characterization. Commit
and push only; do not create, reopen, or update a pull request.

## Late window-partition result characterization checkpoint

The window-partition Reshape/Transpose-to-SpaceToDepth indexed owner returns the
stable one-counter dictionary
`optimized_window_partition_reshape_transpose_to_space_to_depth_chains`. It
receives the live Session LayoutState, is selected once by the existing
attention-recovery runner, and has one additional direct late production call.
Only that direct call currently discards its result.

A strict expected-failure orchestration contract requires
`_late_window_partition_stats` for the direct call. It fixes the captured
attention-preprojection predecessor, exact ModelIR/LayoutState callback, and
following window-reverse callback with the same live Session LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/layout synchronization, pass order, callback arguments,
neighbor targets, guards, dependencies, or TensorFlow behavior. Validate the
window owner, preprojection/window-reverse boundaries, layout recovery,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused window owner, preprojection/window-reverse boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `414 passed, 1 xfailed in 19.47s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window owners, layout recovery, and pass-efficiency coverage:
  `1633 passed, 1 xfailed in 24.87s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late window-partition result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_window_partition_stats`. This is an assignment-only
orchestration change. The indexed owner and schema, separate recovery-runner
selection and captured results, GraphIndex/layout synchronization, callback
arguments, pass order, attention-preprojection predecessor, window-reverse
successor, dependencies, and TensorFlow behavior are unchanged. The value has
no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused window owner, preprojection/window-reverse boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `415 passed in 18.79s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, window owners, layout recovery, and pass-efficiency coverage:
  `1634 passed in 25.07s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains()` result,
owner schema, live LayoutState contract, production occurrences, and window-
partition/final-convergence boundaries before adding characterization. Commit
and push only; do not create, reopen, or update a pull request.

## Late window-reverse result characterization checkpoint

The window-reverse Reshape/Transpose-to-DepthToSpace indexed owner returns the
stable one-counter dictionary
`optimized_window_reverse_reshape_transpose_to_depth_to_space_chains`. It
receives the live Session LayoutState, is selected once by the existing
attention-recovery runner, and has one additional direct late production call.
Only that direct call currently discards its result.

A strict expected-failure orchestration contract requires
`_late_window_reverse_stats` for the direct call. It fixes the captured window-
partition predecessor, exact ModelIR/LayoutState callback, and following
indexed final shape/activation convergence call with the same live Session
LayoutState.

At implementation, replace only the direct expression with an assignment. Do
not change the owner, one-key schema, recovery-runner selection or captured
results, GraphIndex/layout synchronization, pass order, callback arguments,
neighbor targets, guards, dependencies, or TensorFlow behavior. Validate the
window owners, partition/final-convergence boundaries, layout recovery,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused window owners, partition/final-convergence boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `410 passed, 1 xfailed in 19.38s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, both window owners, final convergence, layout recovery, and
  pass-efficiency coverage: `1682 passed, 1 xfailed in 24.62s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented direct
late-result retention contract above.

## Late window-reverse result retention implementation checkpoint

The sole direct late production call now retains the existing one-counter
dictionary as `_late_window_reverse_stats`. This is an assignment-only
orchestration change. The indexed owner and schema, separate recovery-runner
selection and captured results, GraphIndex/layout synchronization, callback
arguments, pass order, window-partition predecessor, indexed final-convergence
successor, dependencies, and TensorFlow behavior are unchanged. The value has
no consumer and triggers no additional graph work.

Implementation validation completed sequentially under `uv`:

- focused window owners, partition/final-convergence boundaries, layout
  recovery, terminal-orchestration, and architecture gate:
  `411 passed in 18.65s`
- branch-changed broad related suite plus cleanup, indexed QKV/Gather-axis0,
  preprojection, both window owners, final convergence, layout recovery, and
  pass-efficiency coverage: `1683 passed in 24.57s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following
`_run_indexed_final_shape_activation_convergence()` result, return schema, live
LayoutState contract, production occurrences, and window-reverse/boundary-
normalization boundaries before adding characterization. Commit and push only;
do not create, reopen, or update a pull request.

## Late indexed final convergence result characterization checkpoint

`_run_indexed_final_shape_activation_convergence()` returns the existing
aggregate mutation dictionary from indexed shape convergence, HardSwish shape
sanitation, dynamic-Reshape resolution, static reconciliation, and activation
fusion. It has one production call and receives the live Session LayoutState.
Its result is currently discarded.

A strict expected-failure orchestration contract requires
`_late_final_shape_activation_convergence_stats`. It fixes the captured window-
reverse predecessor, exact ModelIR/LayoutState callback, and following final
boundary-input normalization call with the same LayoutState and Session
diagnostics.

At implementation, replace only the expression with an assignment. Do not
change the aggregate schema, one-index convergence internals, conditional
reconciliation, fusion behavior, pass order, callback arguments, neighboring
targets, guards, dependencies, diagnostics, or TensorFlow behavior. Validate
indexed final convergence, window-reverse/boundary-normalization boundaries,
terminal orchestration, architecture, and broad related gates sequentially,
then commit and push only; do not create, reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed final convergence, window-reverse/boundary-normalization,
  terminal-orchestration, architecture, and pass-efficiency gate:
  `389 passed, 1 xfailed in 18.19s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage:
  `1687 passed, 1 xfailed in 24.58s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented aggregate-
result retention contract above.

## Late indexed final convergence result retention implementation checkpoint

The sole production call now retains the existing aggregate mutation
dictionary as `_late_final_shape_activation_convergence_stats`. This is an
assignment-only orchestration change. The aggregate schema, one-index
convergence internals, conditional reconciliation, fusion behavior, callback
arguments, pass order, window-reverse predecessor, final boundary-normalization
successor, diagnostics, dependencies, and TensorFlow behavior are unchanged.
The value has no consumer and triggers no additional graph work.

The architecture contract now locates the production assignment and verifies
its target while preserving its existing internal convergence assertions.

Implementation validation completed sequentially under `uv`:

- focused indexed final convergence, window-reverse/boundary-normalization,
  terminal-orchestration, architecture, and pass-efficiency gate:
  `390 passed in 20.13s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage: `1688 passed in 24.36s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately following final
`run_boundary_input_normalization_cleanup()` result, distinguish it from the
earlier production occurrence, and fix its convergence/internal-channel-slice
boundaries before adding characterization. Commit and push only; do not create,
reopen, or update a pull request.

## Final boundary-input normalization result characterization checkpoint

`run_boundary_input_normalization_cleanup()` returns the stable one-counter
dictionary
`rewritten_boundary_input_transpose_mul_sum_reshape_nhwc_chains` and has two
production occurrences. Both receive the live Session LayoutState and
diagnostics. The final occurrence follows indexed final convergence and
currently discards its result; the earlier occurrence is outside this unit and
remains a raw call.

A strict expected-failure orchestration contract requires
`_final_boundary_input_normalization_stats` only for the final occurrence. It
fixes the captured final-convergence predecessor, exact ModelIR/LayoutState/
diagnostics callback, and following model-only internal Transpose/channel-slice
propagation owner. It also fixes the earlier occurrence as a distinct raw call.

At implementation, replace only the final expression with an assignment. Do
not change the earlier occurrence, owner, one-key schema, transaction or
preflight behavior, GraphIndex/layout synchronization, pass order, callback
arguments, neighbor targets, diagnostics, guards, dependencies, or TensorFlow
behavior. Validate the normalization owner, both occurrence contracts,
convergence/channel-slice boundaries, terminal orchestration, architecture, and
broad related gates sequentially, then commit and push only; do not create,
reopen, or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused normalization owner, both occurrence contracts, convergence/channel-
  slice boundaries, terminal-orchestration, architecture, and pass-efficiency
  gate: `342 passed, 1 xfailed in 18.57s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage:
  `1688 passed, 1 xfailed in 24.53s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented final-
occurrence result retention contract above.

## Final boundary-input normalization result retention implementation checkpoint

Only the final of the two production calls now retains the existing one-counter
dictionary as `_final_boundary_input_normalization_stats`. The earlier
occurrence remains a raw call. This is an assignment-only orchestration change:
the owner and schema, transaction/preflight behavior, GraphIndex/layout
synchronization, callback arguments, diagnostics, pass order, final-convergence
predecessor, internal channel-slice successor, dependencies, and TensorFlow
behavior are unchanged. The retained value has no consumer or extra graph work.

Implementation validation completed sequentially under `uv`:

- focused normalization owner, both occurrence contracts, convergence/channel-
  slice boundaries, terminal-orchestration, architecture, and pass-efficiency
  gate: `343 passed in 18.16s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, both window owners, final convergence, boundary normalization,
  layout recovery, and pass-efficiency coverage: `1689 passed in 24.15s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the earlier raw
`run_boundary_input_normalization_cleanup()` result and its terminal-softmax/
boundary-channel-slice boundaries before adding characterization. Keep the
final target fixed. Commit and push only; do not create, reopen, or update a
pull request.

## Terminal boundary-input normalization result characterization checkpoint

The earlier `run_boundary_input_normalization_cleanup()` production occurrence
returns the same stable one-counter dictionary as the already captured final
occurrence and receives the same live Session LayoutState and diagnostics. Its
result is still discarded.

A strict expected-failure orchestration contract requires
`_terminal_boundary_input_normalization_stats` for only the earlier occurrence.
It fixes the terminal Softmax/Transpose predecessor, exact ModelIR/LayoutState/
diagnostics callback, following boundary-input Transpose/channel-slice owner,
and the existing `_final_boundary_input_normalization_stats` target.

At implementation, replace only the earlier expression with an assignment. Do
not change the final occurrence, owner, one-key schema, transaction or
preflight behavior, GraphIndex/layout synchronization, pass order, callback
arguments, neighbor targets, diagnostics, guards, dependencies, or TensorFlow
behavior. Validate the normalization and adjacent indexed owners, both
occurrence contracts, terminal orchestration, architecture, and broad related
gates sequentially, then commit and push only; do not create, reopen, or update
a pull request.

Characterization validation completed sequentially under `uv`:

- focused normalization owner, terminal Softmax and boundary channel-slice
  neighbors, both occurrence contracts, terminal-orchestration, architecture,
  and pass-efficiency gate: `359 passed, 1 xfailed in 18.23s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  Softmax, layout recovery, and pass-efficiency coverage:
  `1718 passed, 1 xfailed in 24.30s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented terminal-
occurrence result retention contract above.

## Terminal boundary-input normalization result retention implementation checkpoint

The earlier production call now retains the existing one-counter dictionary as
`_terminal_boundary_input_normalization_stats`; the already captured final call
keeps `_final_boundary_input_normalization_stats`. This is an assignment-only
orchestration change. The owner and schema, transaction/preflight behavior,
GraphIndex/layout synchronization, callback arguments, diagnostics, pass order,
terminal Softmax predecessor, boundary channel-slice successor, final target,
dependencies, and TensorFlow behavior are unchanged. Neither value has a
consumer or triggers additional graph work.

Implementation validation completed sequentially under `uv`:

- focused normalization owner, terminal Softmax and boundary channel-slice
  neighbors, both occurrence contracts, terminal-orchestration, architecture,
  and pass-efficiency gate: `360 passed in 18.72s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  Softmax, layout recovery, and pass-efficiency coverage:
  `1719 passed in 24.61s`

These are unit, contract, and orchestration checks; this accounting-only change
does not claim a new model-corpus run.

At resume, audit the immediately preceding
`_optimize_terminal_softmax_transpose_after_nhwc_propagation()` result, owner
schema, live LayoutState contract, production occurrences, and Gather-fanout/
boundary-normalization boundaries before adding characterization. Commit and
push only; do not create, reopen, or update a pull request.

## Terminal Softmax/Transpose result characterization checkpoint

The terminal Softmax/Transpose-after-NHWC-propagation indexed owner returns the
stable one-counter dictionary
`removed_terminal_softmax_transpose_after_nhwc_propagation`. It receives the
live Session LayoutState and has one production occurrence. Its result is
currently discarded.

A strict expected-failure orchestration contract requires
`_terminal_softmax_transpose_stats`. It fixes the exact ModelIR/LayoutState
callback between the diagnostics-aware Gather-channel-fanout runner and the
captured `_terminal_boundary_input_normalization_stats` successor.

At implementation, replace only the expression with an assignment. Do not
change the owner, one-key schema, GraphIndex/layout synchronization, pass order,
callback arguments, neighbor targets, diagnostics, guards, dependencies, or
TensorFlow behavior. Validate the indexed Softmax owner, Gather-fanout/
normalization boundaries, terminal orchestration, architecture, and broad
related gates sequentially, then commit and push only; do not create, reopen,
or update a pull request.

Characterization validation completed sequentially under `uv`:

- focused indexed Softmax, Gather-fanout, boundary normalization, terminal-
  orchestration, architecture, and pass-efficiency gate:
  `364 passed, 1 xfailed in 18.89s`
- branch-changed broad related suite plus cleanup, indexed attention/Gather,
  preprojection, window/final convergence, boundary normalization, terminal
  Softmax, Gather fanout, layout recovery, and pass-efficiency coverage:
  `1723 passed, 1 xfailed in 24.50s`
- Ruff, Python bytecode compilation, and `git diff --check`: passed

The sole strict expected failure is the intentionally unimplemented terminal
Softmax result retention contract above.
