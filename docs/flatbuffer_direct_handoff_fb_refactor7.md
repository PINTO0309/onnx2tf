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
