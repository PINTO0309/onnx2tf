# `flatbuffer_direct` refactor continuation checkpoint — 2026-07-14

## Status

The active branch is `fb-refactor5`, created from `main` after pull request
`#949` merged the complete `fb-refactor4` checkpoint. Pull request `#950` is
closed, and no open pull request tracks this branch. This checkpoint is ready
for the next Goal continuation. Work continues through coherent commits and
pushes without opening a pull request.

The latest implementation unit moves Slice/optional-Logistic/Concat/Reshape
detection-tail recovery into the TensorFlow-free indexed
`passes/slice_logistic_concat_reshape_tail_layout.py` owner. Its former
480-line full-map, unbounded fixed-point helper is now a thin compatibility
dispatcher at both unchanged source positions. The owner resolves an immutable
whole-group plan, re-resolves the complete contract before apply, and maintains
one differential `ModelIRGraphIndex` plus the Session `LayoutState` through
every rewrite.

The audited fast-precanonicalize orchestrator remains 294 lines, down from 482
lines at Goal resumption, 1,025 lines at the beginning of the previous
continuation, and 1,608 lines before the broader extraction.

### Split/mixed-Concat continuation — 2026-07-16

The extracted owner starts from graph-ordered channel-axis Concat candidates
instead of rebuilding complete producer and consumer maps. Every Concat input
must be classified as either a private typed NHWC-to-NCHW adapter or a direct
output of a channel-axis Split. At least one Split root is mandatory. The
resolver proves rank-four static and dynamic views, dtype, per-tensor
quantization, explicit or unknown layout, producer uniqueness, graph order,
exact consumer ownership, typed INT32/INT64 permutation and Split-axis
constants, the derived NHWC Concat result, and the private terminal boundary
before a plan exists.

All outputs of each accepted Split are classified together. An output may feed
the target Concat directly or feed one or more private NCHW-to-NHWC adapters;
any unrelated or public consumer rejects the complete group. Adapter aliases
are rewired to the converted Split output only after revalidation. Shared
Split-axis constants receive one deterministic dtype-preserving clone, while
an axis with exactly one owned consumer slot changes in place. Produced,
variable, public, malformed, or per-axis-quantized constants and tensors are
not rewritten.

The immutable plan records all axis ownership, new Split-input adapters,
converted output metadata, direct-input adapter removals, alias rewrites,
Concat axis/output changes, the terminal NCHW compatibility adapter, and full
tensor/operator and graph-boundary contracts. The same Concat is resolved
again immediately before mutation. Candidate-only operation and an explicit
rewrite limit use the production entry point; inserted and removed operators
update one differential graph index. Converted tensors are recorded in both
TensorIR and Session `LayoutState`, and success-only pruning preserves the
legacy zero-match side effect.

Before extraction, seven short Tier 0-4 models containing both ONNX Split and
Concat were converted sequentially under an instrumented helper:
`alike_t_opset11_192x320`, `yolov9_n_wholebody_with_wheelchair_post`,
`sgscsh`, `best`, `mobileformer`, `parseq-tiny`, and `LINEA`. The mixed helper
remained zero-match in every runtime invocation. The immediately following
input-chain helper remained active where expected, including three first-sweep
groups in `yolov9` and two later-sweep groups in `sgscsh`; extraction therefore
did not absorb or pre-empt that separate family.

Sixteen dedicated tests cover the active mixed and all-Split forms, dynamic
signatures, negative axes, shared-axis cloning, explicit layouts, public
boundaries, fan-out, per-axis quantization, produced and duplicate producers,
stale-plan rejection, candidate limits, idempotence, GraphIndex/LayoutState
integrity, the compatibility dispatcher, and repeated production sweeps. The
new owner, adjacent Split/Concat owners, boundary-input checks, complete
architecture suite, and TensorFlow import blocker pass together with `524
passed in 51.68s`. Scoped Ruff, syntax compilation, and `git diff --check`
pass.

Two post-extraction `-cotof` checks ran sequentially. `yolov9` remains a pass
with maximum absolute error `3.0517578125e-05`; `sgscsh` remains a pass with
maximum absolute error `2.5331974029541016e-07`. Both exactly reproduce the
managed quick-profile accuracy values and remain well below `1e-1`. No
TensorFlow import, new dependency, parallel inference, SWAP, timeout, or Tier
corpus run was introduced. Temporary characterization and conversion outputs
were removed.

### General Concat input-adapter continuation — 2026-07-16

The extracted owner starts from graph-ordered rank-four channel-axis Concat
candidates instead of rebuilding complete producer and consumer maps. Every
input must be either the private output of a typed NHWC-to-NCHW Transpose or
one of the historical thirteen unary operations immediately following such a
Transpose or a singleton-channel memory-equivalent Reshape. Direct and unary
forms may be mixed. Repeated Concat input slots share one resolved branch and
one metadata update rather than applying the old mutation twice.

The resolver proves static and dynamic rank-four views, dtype, per-tensor
quantization, logical and physical layout, producer uniqueness and graph
order, exact consumer ownership, typed permutation and Reshape-shape
constants, unary input/output equivalence, the derived NHWC Concat result, and
terminal consumer order. Public Concat outputs and existing NCHW consumers
remain on their original tensor name through one post-Concat compatibility
Transpose. Unsupported fan-out, malformed constants, per-axis quantization,
duplicate producers, stale plans, and unresolved sources reject the complete
candidate without mutation.

The immutable plan owns all adapter removals, unary rewires and metadata,
Concat inputs/axis/output, optional deterministic permutation creation, the
terminal adapter, complete tensor/operator contracts, and graph boundaries.
The same Concat is resolved again immediately before apply. Candidate-only
operation and an explicit rewrite limit use the production entry point;
removed and inserted operators update one differential graph index. TensorIR
and Session `LayoutState` both record resolved NHWC and retained NCHW
boundaries. Pruning runs once at owner exit even when no candidate matches,
preserving the legacy zero-match side effect and safe-bundle contract.

Pre-extraction characterization observed six invocations on each of `yolov9`
and `sgscsh`. `yolov9` retained three direct two-input groups in its first
invocation at spatial sizes 48x84, 24x42, and 12x21. `sgscsh` retained two
direct three-input groups in its second invocation; every other invocation was
zero-match. The preceding indexed Split/mixed-Concat owner remained zero-match
for these groups, confirming the ordered ownership boundary.

Thirty-six dedicated tests plus the two existing active fixtures cover direct,
dynamic, negative-axis, public-output, repeated-slot, existing-consumer,
existing/new permutation, all thirteen unary, singleton-Reshape, dispatcher,
candidate limit, stale-plan, zero-match pruning, safe-bundle, determinism, copy
isolation, and transactional rejection behavior. The new and adjacent indexed
owners, boundary-input checks, TensorFlow import blocker, and complete
architecture suite pass together with `561 passed in 51.11s`. Scoped Ruff,
format verification, syntax compilation, and `git diff --check` pass.

Baseline/current conversion-only comparisons used identical Python API calls
from detached source checkpoint `f5505052`. For both `yolov9` and `sgscsh`,
float32 TFLite, float16 TFLite, tensor correspondence report, `schema.fbs`,
and `schema_generated.py` are byte-identical. Separate sequential `-cotof`
checks pass at maximum absolute errors `3.0517578125e-05` and
`2.5331974029541016e-07`, respectively. An initial apparent `yolov9` report
difference was traced before any source change to unlike invocation modes:
the `-cotof` CLI path normalizes leading `/` names to `wa/`, whereas the
conversion-only Python comparison did not. Matching invocation modes removed
the difference. No implementation fix was necessary.

### Slice/Logistic/Concat/Reshape tail continuation — 2026-07-16

The extracted owner starts from graph-ordered rank-three channel/spatial tail
Concat candidates instead of rebuilding complete producer and consumer maps.
Every unique tail input must be a private rank-three Reshape fed by a private
two-input channel-axis Concat. Each branch input must be a private rank-four
Slice, optionally followed by Logistic, and both Slices must be the complete
fan-out of one typed NHWC-to-NCHW Transpose. Multiple branches may share the
same NHWC source, but every removable adapter and every downstream branch is
owned independently.

The resolver proves static and dynamic rank-three/rank-four views, dtype,
per-tensor quantization, explicit or unknown layout, unique producers, strict
graph order, exact consumer slots, typed INT32/INT64 permutation, Slice begin,
Slice size, and Reshape-shape constants, exact Slice bounds, both Concat
results, Reshape element counts, and the retained terminal NCHW boundary.
`newShape` and `onnxRawNewShape` options are validated and remapped together.
Exclusive constants change in place; a safe shared Reshape shape receives one
deterministic dtype-preserving clone. Slice parameters must remain exclusive,
so unrelated use rejects the complete group.

The immutable plan owns all Slice input rewrites and constants, optional
Logistic metadata, branch Concat axes, Reshape constants/options/metadata, tail
Concat axis/output, optional typed 3D permutation creation, the terminal
compatibility Transpose, adapter removals, complete tensor/operator contracts,
and graph boundaries. The same tail Concat is fully resolved again immediately
before apply. Candidate-only operation and an explicit rewrite limit use the
production entry point; inserted and removed operators update one differential
graph index. TensorIR and Session `LayoutState` are updated together. Pruning
runs once at owner exit, including zero-match calls.

Characterization selected the 4.5 MiB Tier-2
`nanodet-plus-m_416.onnx` without broad conversion. The helper is invoked five
times: the first call retains one four-branch group and the remaining four are
zero-match. Branch spatial sizes are 52x52, 26x26, 13x13, and 7x7; each splits
37 channels into a Logistic-wrapped 5-channel path and a direct 32-channel
path before Reshape. The preceding Split/mixed owner remains zero-match. The
general input-adapter owner rewrites seven separate groups on its first call
but deliberately leaves this tail group for its ordered owner. Five yolov9
calls remain zero-match because its superficially similar rank-three tail is
fed by Transposes rather than the required Slice/Concat branches.

Forty dedicated tests plus the existing active fixture cover all four optional
Logistic forms, numerical equivalence, dynamic full-extent Slice signatures,
INT32/INT64 constants, shared-shape copy-on-write, existing/new 3D
permutations, shared NHWC sources, compatibility dispatch, candidate limits,
stale-plan rejection, zero-match pruning, determinism, GraphIndex/LayoutState
integrity, and twenty-four unsafe transactional no-op cases. The new and
adjacent indexed owners, boundary-input checks, TensorFlow import blocker, and
complete architecture suite pass together with `603 passed in 49.68s`.
Scoped Ruff, format verification, syntax compilation, and `git diff --check`
pass.

Baseline/current conversion-only comparisons use the same Python API and raw
source checkpoint `762dcdef`. Nanodet float32 TFLite, float16 TFLite, tensor
correspondence report, `schema.fbs`, and `schema_generated.py` are all
byte-identical. The indexed helper retains the exact production sequence of
one rewrite followed by four zero-match calls. A separate sequential `-cotof`
check passes with maximum absolute error `6.19888e-06`, below both the managed
profile value `0.03541278839111328` and the required `1e-1` ceiling. The
conversion process reported `VmSwap: 0 kB`; no timeout or parallel inference
was introduced.

## Continuation snapshot — 2026-07-16

The indexed elementwise/Concat owner preserves the complete historical
capability set: eleven unary operations, all six binary operations without
reordering operands, connected Concat groups, rank-four broadcast constants,
direct graph-input boundaries, explicit NHWC-to-NCHW adapters, reusable
inverse-adapter aliases, multiple post aliases, safe legacy consumers, and
pre-adapter fan-out. Typed INT32/INT64 permutations, static and dynamic
rank-four views, dtype, per-tensor quantization, graph order, public
boundaries, producer uniqueness, exact consumer slots, constant ownership,
and broadcast results are proven before a plan is accepted.

The candidate set is fixed in graph order, reverse closure traversal is
bounded by the current input-edge count, and an optional rewrite limit bounds
accepted groups. Every input/output rewrite, axis and metadata update,
constant transpose, adapter insertion/removal, tensor contract, and operator
contract belongs to immutable plan state. The same seed is fully resolved a
second time immediately before mutation. Pruning runs once after all groups,
preserving the legacy zero-match side effect and correspondence-event order.

Sequential pre/post characterization reached the helper five times on each
of YuNet, FastestDet, HumanSeg, OSNet, and SiNet. FastestDet retained two
connected groups and HumanSeg retained one group in their first invocation;
the other 22 invocations were zero-match. The pre/post semantic ModelIR digest
matched the preceding checkpoint at all ten FastestDet/HumanSeg invocation
boundaries. The only full-digest difference is intentional: converted tensors
now carry explicit `NHWC` in both TensorIR and Session `LayoutState` instead of
leaving stale `UNKNOWN` layout provenance.

The dedicated 56-test suite, adjacent indexed layout owners, both existing
active fixtures, SPP owner, and complete architecture suite pass together:

```text
546 passed in 42.49s
```

TensorFlow-import-blocked explicit direct, default direct, and direct `-cotof`
conversion pass sequentially with `3 passed in 4.07s`. Sequential YuNet,
FastestDet, and HumanSeg conversions reproduce all fifteen fixed float32,
float16, correspondence, and schema artifact hashes. Scoped Ruff, syntax
compilation, and `git diff --check` pass. Temporary comparison worktrees and
outputs were removed. No Tier corpus was run.

### StridedSlice/Concat continuation

The indexed owner accepts only one private rank-four NHWC-to-NCHW adapter whose
entire fan-out consists of at least two supported `STRIDED_SLICE` operators.
Every slice must use four distinct, immutable typed INT32/INT64 begin/end/
stride constants, a nonzero stride vector, zero masks, and no offset. Every
slice output must feed exactly one common channel-axis Concat, and at least one
private NCHW-to-NHWC post adapter must establish the removable round trip.

Static and dynamic views, dtype, per-tensor quantization, layout, producer
uniqueness, graph order, exact consumer slots, Concat shape/signature, every
post alias, and every legacy boundary are fixed before a plan exists. All
constants, Slice input rewrites, output metadata, Concat axis/output changes,
alias rewrites, adapter reuse/removal, tensor contracts, and operator contracts
belong to immutable plan state and are fully resolved again immediately before
apply. Graph-ordered Transpose candidates and an optional rewrite limit replace
the raw fixed-point loop; pruning still runs once on every invocation to retain
the historical zero-match contract.

The ordinary and multi-post synthetic active forms produce the exact same
non-layout ModelIR digest as source checkpoint `44f3b2ca`, including tensor
lineage metadata and event ordering. Converted Slice and Concat tensors now
carry explicit NHWC in TensorIR and Session LayoutState. A retained NCHW
boundary reuses the already-proven typed pre-permutation rather than overwriting
the first post-permutation buffer as INT32; shared post-permutation users remain
unchanged. Conflicting constant roles and a legacy consumer ordered before the
retained adapter reject the complete transaction.

Pre- and post-extraction characterization reached the helper five times on
each of YuNet, FastestDet, HumanSeg, OSNet, and SiNet. All 25 invocations remain
zero-match with unchanged operator and tensor counts. Fifty-eight focused tests
cover typed constants and permutations, dynamic signatures, negative axes,
multiple post aliases, repeated input slots, public and legacy boundaries,
shared post permutations, numerical equivalence, candidate limits,
idempotence, GraphIndex/LayoutState integrity, lineage compatibility, zero-
match pruning, and forty-two unsafe transactional no-op cases.

The new owner, adjacent pre-Concat Slice/Split and Split/Conv/Concat owners,
elementwise/Concat owner, and complete architecture suite pass together with
`542 passed in 40.54s`. TensorFlow-import-blocked explicit direct, default
direct, and direct `-cotof` conversion pass sequentially with `3 passed in
4.01s`. One sequential YuNet conversion reproduces all five fixed artifacts.
Scoped Ruff, syntax compilation, and `git diff --check` pass. Temporary
comparison worktrees and outputs were removed. No Tier corpus was run.

## Continuation snapshot — 2026-07-15

The characterized
`_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw` helper
is now extracted. Its dedicated owner performs pure match/plan/revalidation
before mutation, uses one differential graph index, and replaces both the
outer fixed-point loop and closure-wide repeated scans with graph-ordered
candidates plus a bounded consumer `deque`. The existing two production
sequence positions and stats key are unchanged.

The plan proves typed permutation constants, exact rank-four shape and dynamic
signature relations, all six plain binary operations, operand order, resolved
sources, unique producers, graph order, per-tensor quantization, NHWC
broadcasts, equal channel Split contracts, channel Concat axes, and a closed
unary/binary/Concat/Split tail. Shared Split axes are cloned deterministically
with their original INT32 or INT64 dtype. Unsupported consumers, dead converted
branches, public intermediates, multiple outputs, fused activation, per-axis
quantization, explicit NCHW external inputs, stale order, missing tensors, and
duplicate producers reject the complete candidate without mutation.

The terminal adapter now preserves the original public NCHW tensor metadata
and moves only its private producer output to NHWC. It reuses the proven input
permutation. This fixes the raw helper's public-metadata mismatch and prevents
the input Transpose from being removed when any converted closure edge remains
unhandled.

Pre-extraction characterization remains authoritative: four zero-match runtime
invocations each on YuNet, FastestDet, HumanSeg, and OSNet, and eight zero-match
invocations on SiNet. The indexed active synthetic contract is covered by 33
focused tests. The two preceding binary owners, the new owner, and the complete
architecture suite passed together with `268 passed in 44.46s`. The existing
direct-builder active-path test and the new architecture ownership check passed
together with `2 passed in 1.67s`. TensorFlow-import-blocked explicit direct,
default direct, and `-cotof` conversion passed with
`3 passed, 8 deselected in 3.77s`.

One sequential YuNet conversion was compared with source checkpoint
`5a186acd`. Every emitted file was byte-identical. Float32 remains 236,564 bytes
with SHA-256
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`;
float16 remains 131,120 bytes with SHA-256
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`;
and the 105,578-byte correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
Both schema files also match. The detached worktree and all temporary outputs
were removed. No Tier corpus run was performed.

Scoped Ruff, syntax compilation, and `git diff --check` pass. No dependency or
TensorFlow import was added. There is no source-file 2,000-line Goal
requirement; the 2,000 threshold applies only to ONNX operation-count Tier
classification.

### Direct Split root continuation

The adjacent
`_optimize_transpose_split_channelwise_tail_to_single_post_nchw` helper is now
also extracted. Its two production positions and stats key are unchanged. A
separate immutable direct-root plan proves the private NHWC-to-NCHW Transpose,
the channel Split, and the shared closed-tail contract before the first write.
Only Split outputs seed closure traversal, so unrelated consumers of the raw
NHWC source are no longer accidentally classified as converted tensors.

The shared closure now optionally plans rank-four Slice operations. Typed
INT32/INT64 begin and size constants, bounds, original NCHW result metadata,
converted NHWC metadata, and dynamic signatures are all validated. A constant
changes in place only when every actual consumer slot is one of the planned
Slice uses with the same replacement. Otherwise every planned use shares one
deterministic clone, preserving both declared and NumPy dtype. Invalid or
conflicting constants reject the whole candidate without mutation.

The root Split axis uses the existing copy-on-write contract, the public output
keeps its original NCHW shape/layout, and one private NHWC producer output feeds
the terminal adapter. All metadata, axis, Concat, Slice-constant, graph-index,
and LayoutState changes are applied only after full re-resolution and
preflight. The configurable 32-rewrite ceiling and edge-bounded consumer queue
replace the old unbounded fixed-point scans.

The 20-test direct-root suite and 33-test binary-root suite pass together. The
two preceding binary owners, both Split-root suites, and the complete
architecture suite pass with `288 passed in 44.74s`. TensorFlow-import-blocked
explicit direct, default direct, and `-cotof` conversion pass with `3 passed in
3.85s`. A sequential YuNet comparison against `5f8e662a` emits five
byte-identical files with the same float32, float16, and correspondence hashes
recorded above. The detached worktree and temporary outputs were removed; no
Tier corpus run was performed.

### Unary/Split/Concat root continuation

The preceding `_optimize_transpose_unary_split_concat_single_post_nchw` helper
is now extracted into the same semantic owner under a separate immutable root
plan. Its two production positions, order relative to the direct and binary
Split roots, and stats key are unchanged. Both calls now supply Session
LayoutState.

The plan proves an exact typed private NHWC-to-NCHW Transpose, one allowed
pre-Split unary, one channel Split, every Split output exactly once through
either a direct edge or one allowed unary, exactly one external branch, and a
channel Concat. All converted branch fan-out is closed. Public intermediates,
duplicate or missing Split branches, multiple external branches, duplicate
producers, stale order, per-axis quantization, and malformed unary operators
reject the candidate without mutation.

An external singleton Reshape is bypassed only when its source/output shapes
and dynamic signatures have the exact NHWC/NCHW relationship and both channel
dimensions are one. Dtype, quantization, provenance, graph order, and physical
layout are also validated. Already-proven direct NHWC external inputs remain
supported. Shared external Reshape consumers are untouched.

The Concat output may be public or an intermediate legacy NCHW tensor. One
private NHWC producer output and the proven input permutation preserve that
local contract. Split-axis copy-on-write, metadata, Concat inputs/options,
adapter insertion, graph-index updates, and LayoutState updates occur only
after full re-resolution and preflight. This removes the raw path that could
change the Split axis, metadata, and Concat before discovering an invalid
output tensor.

Pre-extraction characterization recorded zero matches in all 24 invocations:
four each on YuNet, FastestDet, HumanSeg, and OSNet, and eight on SiNet. The
29-test owner suite, existing active-path test, adjacent Split owners, binary
bridge owners, and architecture suite pass together with `318 passed in
43.96s`. TensorFlow-import-blocked explicit direct, default direct, and
`-cotof` conversion pass with `3 passed in 3.83s`. A sequential YuNet
comparison against `0b4a0001` emits the same five byte-identical files. The
detached worktree and all temporary outputs were removed; no Tier corpus run
was performed.

### RELU/Split all-output continuation

The three adjacent raw Split/Conv/Concat helpers were audited together before
implementation. They are semantically exclusive: the all-output root requires
`Transpose -> RELU -> Split` with every Split result consumed by one inverse
Transpose; the Conv/Concat root adds an exact two-branch Conv/RELU/Concat tail;
and the bridge root begins with a direct Transpose-produced Split and retains a
local NCHW compatibility adapter after its Concat. The audit recorded 41
sequential runtime invocations on YuNet, FastestDet, HumanSeg, OSNet, and SiNet
and every invocation returned zero without changing operator or tensor counts.
The all-output root accounted for 12 of those invocations.

The new all-output owner resolves graph-ordered Split candidates against one
`ModelIRGraphIndex`. It accepts an immutable typed INT32/INT64 axis-one
constant, a private typed NHWC-to-NCHW adapter, RELU, an equal channel Split,
exactly one typed inverse adapter for every output, and all later consumers of
those adapters. Static shapes must be fully known and positive; dynamic shape
signatures are preserved and permuted independently. Source provenance,
unique producers, consumer multiplicity, graph order, dtype, per-tensor
quantization, physical layout, public boundaries, and `numSplits` are proven
before a plan exists.

Downstream rewrites are grouped by operator, so a Concat or other consumer that
uses several former post-Transpose outputs is updated once with every input
slot preserved. A shared Split axis receives one deterministic clone with the
original TensorIR and NumPy dtype; an exclusive axis changes in place. The
complete tensor/operator contract is resolved again immediately before apply,
then input rewrites, NHWC metadata, differential index compaction, LayoutState
updates, and success-only pruning occur as one bounded operation. Candidate
count and an optional explicit rewrite limit replace the raw unbounded
fixed-point loop.

Thirty-two focused tests cover two- and three-way Split, INT32 and INT64 axes,
negative axes, static and dynamic signatures, exact numerical equivalence,
one consumer using every branch, multiple consumers of one post output,
copy-on-write shared axes, candidate limits, idempotence, GraphIndex and
LayoutState integrity, and nineteen transactional unsafe no-op cases. The new
owner, adjacent indexed owners, active direct-builder fixture, and complete
architecture suite pass together with `475 passed in 46.51s`.
TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.01s`.

One sequential YuNet conversion emits the five artifacts already fixed by the
preceding checkpoint. Float32 remains 236,564 bytes with SHA-256
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`.
Float16 remains 131,120 bytes with SHA-256
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`.
The 105,578-byte correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`;
`schema.fbs` remains
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py` remains
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
No Tier corpus run was performed.

### RELU/Split/Conv/Concat continuation

The exact two-branch helper is now the second plan in the related Split-layout
owner. It requires a private NHWC-to-NCHW Transpose and RELU, a two-way equal
channel Split, one Split branch returning to NHWC for Conv2D, that result
returning to NCHW for RELU, the untouched branch joining it through channel
Concat, and one final inverse Transpose. Its two production positions and
order immediately after the all-output helper are unchanged.

The Conv branch may be either Split output and may appear in either Concat
slot. Neither order is normalized or commuted. Split input channels must divide
exactly by two; both output shapes/signatures and optional `numSplits` must
describe the equal result. The old synthetic fixture's invalid six-channel
2/4 split is corrected to a valid eight-channel 4/4 split. Conv output channels
may differ from the branch width, so old NCHW and new NHWC Concat results are
derived independently and required to agree through the final adapter.

Every typed permutation, source and Conv-side input provenance, unique
producer, exact consumer, graph-order edge, public boundary, tensor presence,
static/dynamic shape, dtype, per-tensor activation quantization, and physical
layout is resolved before a plan exists. Adapter endpoints must have identical
dtype and quantization. Fan-out from an owned intermediate, an additional or
unequal Split output, stale order, missing filter/bias, duplicate producer,
public intermediate, or contradictory layout rejects the candidate without
mutation.

The immutable plan records the pre-RELU, Split-axis, Conv, post-Conv RELU, and
all final-consumer input changes; five NHWC metadata updates; the Concat-axis
update; the four adapter removals; and every tensor/operator contract. Shared
Split axes use deterministic INT32/INT64 copy-on-write. A complete second
resolution and preflight precede all writes, after which one differential
graph index performs the input changes and one four-operator compaction.
Candidate count bounds execution, and pruning occurs only after success.

Forty-nine dedicated tests cover both Split branch positions, both Concat
orders, static and dynamic signatures, INT32/INT64 and negative axes, Conv
channel changes, exact numerical equivalence, multiple final consumers,
shared-axis cloning, candidate limits, idempotence, GraphIndex, LayoutState,
and twenty-nine transactional unsafe no-op cases. With the active fixture,
adjacent indexed owners, and complete architecture suite, `525 passed in
47.51s`. TensorFlow-import-blocked explicit direct, default direct, and
`-cotof` conversion pass sequentially with `3 passed in 4.33s`.

One sequential YuNet conversion reproduces the same five artifacts and hashes
recorded above: 236,564-byte float32, 131,120-byte float16, 105,578-byte
correspondence report, `schema.fbs`, and `schema_generated.py`. No Tier corpus
run was performed.

### Direct Split/Conv/Concat bridge continuation

The last raw helper in the audited three-member Split/Conv/Concat group is now
extracted to `passes/split_conv_concat_bridge_layout.py`. Its three production
positions and order relative to the preceding all-output and exact
Conv/Concat plans and the late QKV recovery remain unchanged. All three calls
now supply Session LayoutState.

The new resolver requires one private NHWC-to-NCHW pre adapter feeding an
equal channel Split exclusively. One Split result owns exactly one inverse
adapter; every other Split result must be a direct input of the same channel
Concat. Every remaining Concat input must be the exclusive output of an
NHWC-to-NCHW post adapter, and at least one post-adapter input must be reachable
from the converted branch before the Concat. The bounded reachability walk
allows an unchanged generic NHWC interior rather than relying on a model name,
a Conv opcode, or a hard-coded chain.

All typed permutations, INT32/INT64 axes, source provenance, producers,
consumers and exact input slots, public boundaries, graph order, equal Split
shapes, dynamic signatures, dtype, per-tensor quantization, physical layout,
Concat classification, and old/new Concat shapes are resolved before a plan
exists. Branch-side NHWC consumers are preserved through grouped exact-slot
rewrites. Unclassified inputs, fan-out on an owned NCHW edge, an unreachable
branch, stale order, duplicate producer, missing tensor, per-axis
quantization, or contradictory layout rejects the complete candidate without
mutation.

The immutable plan records input rewrites, Split metadata, axis copy-on-write,
the Concat axis and output change, the private NHWC Concat tensor, every
adapter removal, and all tensor/operator contracts. The original NCHW Concat
tensor name and metadata are retained behind one post adapter reusing the
proven input permutation. A full second resolution and preflight precedes all
writes; one differential graph index performs the transaction, LayoutState is
updated differentially, and pruning is success-only.

The focused suite passes with `46 passed`. It covers two/three Split
outputs, either branch position, one/two post paths, either Concat order,
static/dynamic signatures, INT32/INT64 and negative axes, exact numerical
equivalence, branch-side consumers, shared-axis cloning, candidate limits,
idempotence, GraphIndex, LayoutState, and twenty-four unsafe transactional
no-op cases. The new owner, adjacent indexed owners, two active fixtures, and
complete architecture suite pass together with `534 passed in 44.94s`.
TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.04s`.

One sequential YuNet conversion reproduced the fixed five hashes: float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed. No Tier corpus run was performed.

### Unquantized pseudo-Swish passthrough continuation

The raw `_optimize_swish_transpose_passthrough_chains` implementation is now
owned by `passes/activation_passthrough_layout.py`. Its position after registered
hard-activation cleanup in the ordered recovery prefix and its late
no-layout-compatible recovery position are unchanged. Both calls supply
Session LayoutState, and the legacy stats key remains unchanged.

The resolver requires a typed immutable rank-two-or-higher INT32/INT64
permutation, exact source/transposed shapes and independently dynamic
signatures, one Logistic consumer, and one residual Mul consumer in either
operand order. At least one typed inverse post adapter must close the
source-layout path. All producer, consumer-slot, graph-order, public-boundary,
dtype, per-tensor quantization, layout, and alias contracts are resolved before
a plan exists. Immutable operator-produced and constant sources are supported;
source buffers remain unchanged because elementwise Swish commutes with the
permutation.

All inverse-post aliases are grouped by exact downstream input slot and fold
to one representative source-layout Mul result. One public post alias may be
the representative. When the old transposed Mul tensor still has a legacy
consumer or is itself public, the plan inserts one adapter immediately after
Mul and reuses the proven pre-permutation. The old helper mutated the selected
post-permutation tensor to the opposite direction; the new structure leaves a
shared post constant and every unrelated consumer untouched.

Input rewrites, output changes, metadata, public lists, removals, and complete
tensor/operator contracts are immutable plan state and are fully re-resolved
before apply. One differential graph index performs all rewrites, one adapter
compaction, and the optional legacy-adapter insertion. LayoutState changes only
with the accepted transaction, candidate count bounds execution, and pruning
is success-only.

Pre-extraction and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. The helper was invoked five times on
each of the first four models and six times on SiNet. All 26 invocations were
zero-match and preserved operator/tensor counts before and after extraction.

The focused suite passes with `56 passed`. It covers rank-three/rank-four,
static/dynamic signatures, INT32/INT64 permutations, both Mul operand orders,
one/multiple posts, repeated alias slots, public post selection, immutable
constant source, shared post permutation, legacy/public transposed boundaries,
exact numerical equivalence, candidate limits, idempotence, GraphIndex,
LayoutState, and twenty-seven unsafe transactional no-op cases. The new owner,
adjacent input and quantized-Swish suites, four active Swish fixtures, and full
architecture suite pass together with `304 passed in 44.59s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.19s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### Tanh-expanded GELU passthrough continuation

The raw `_optimize_gelu_tanh_transpose_passthrough_chains` implementation is
now an indexed compatibility dispatcher in the common activation-passthrough
owner. Its sole production position remains immediately after pseudo-Swish and
before center/size-offset recovery, its stats key is unchanged, and its call
receives Session LayoutState.

The resolver proves the exact nine-operation topology from square and cubic
multiplication through cubic-scale, outer-scale, offset, and final-scale
singleton constant roles. All five uses of the original transposed source, every
linear intermediate, unique producer, graph order, constant dtype/data/shape/
provenance, and per-tensor quantization are validated before a plan exists.
Both valid operand orders are retained for commutative binary operations.

Typed rank-two-or-higher INT32/INT64 permutations, static or independently
dynamic views, dtype, quantization, layout transitions, source provenance, and
inverse post aliases are resolved before mutation. Immutable runtime and
constant sources are supported. Repeated downstream input slots remain exact,
one public post alias may be the representative, and a public or consumed old
transposed final tensor receives one local adapter immediately after the final
Mul. The adapter reuses the proven pre-permutation; shared post constants and
unrelated consumers remain untouched.

The immutable plan records every input/output slot, tensor and operator
contract, public list, metadata update, removal, and optional insertion. It is
fully re-resolved before apply. One differential graph index performs the
transaction, LayoutState changes only after acceptance, candidate enumeration
is graph ordered and bounded, and pruning is success-only. This removes the
raw full-map rebuilds, unbounded fixed-point loop, and partial mutation path.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. Each run reached the helper four times;
all 20 invocations were zero-match and preserved operator and tensor counts.

The dedicated suite passes with `56 passed`. It covers rank-three/rank-four
static and dynamic views, INT32/INT64 permutations, both operand orders,
one/multiple and repeated post aliases, public and legacy boundaries,
immutable constant input, shared permutations, exact numerical equivalence,
candidate limits, idempotence, GraphIndex, LayoutState, and thirty unsafe
transactional no-op cases. The Swish and GELU owners, adjacent input and
quantized-Swish suites, active fixtures, and complete architecture suite pass
together with `362 passed in 42.95s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 3.95s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### Center/size/offset terminal-head continuation

The raw `_optimize_center_size_offset_terminal_transpose_chains` implementation
is now a thin indexed compatibility dispatcher. Its sole production position,
public name, and stats key are unchanged, and its call receives Session
LayoutState. The dedicated `passes/center_size_offset_layout.py` owner has no
model-name or fixed operator-distance condition.

Three typed NHWC-to-NCHW roots are classified by complete semantic branches.
The singleton-channel center branch closes through
Logistic/Maximum/Minimum/Reshape to `[N,H*W]`; size uses the same activation
prefix plus `[N,C,H*W]` Reshape/GatherND; offset uses the matching direct
Reshape/GatherND tail. Static shape, dynamic signature, dtype, per-tensor
quantization, source provenance, layout, unique production, exact consumers,
graph order, and every private/public boundary are validated. Shape-compatible
multiple triples reject transactionally instead of being paired by proximity.

Gather coordinates are classified as immutable batch grid, dynamic axis
Reshape, and immutable channel grid using tensor shape, dtype, provenance, and
range evidence. The owner supports INT32/INT64 coordinates, arbitrary original
Concat order, a shared coordinate Concat, and the equal-grid case where either
constant name is numerically equivalent. Each coordinate output must be closed
over the planned GatherND slots before the order changes to
`[batch, axis, channel]`.

Size and offset reshape literals rotate to `[N,HW,C]`, including valid inferred
`-1` dimensions and both option fields. Shape constants are grouped by tensor
and consumer slot. Exclusive constants change in place; shared constants use
one deterministic typed clone, so unrelated Reshape consumers retain the old
`[N,C,HW]` value. Activation intermediates receive their proven NHWC metadata,
rank-three data views become NWC, and LayoutState is updated only after the
plan has been accepted.

Every input rewrite, option, constant value/use, metadata update, adapter
removal, tensor/operator contract, and public list is immutable plan state. A
full second resolution and preflight precede all writes. One differential
ModelIRGraphIndex performs the rewires and three-adapter compaction;
graph-ordered candidates plus a rewrite limit replace the raw full-map
fixed-point loop, and pruning is success-only.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. The helper was reached four times per
model; all 20 invocations were zero-match and preserved operator/tensor counts.

The dedicated suite passes with `37 passed`. It covers static/dynamic shape,
INT32/INT64 buffers, operand order, per-tensor quantization, inferred Reshape
dimensions, shared constant and coordinate closure, exact numerical
equivalence, candidate limits, idempotence, GraphIndex, LayoutState, and
twenty-two unsafe transactional no-op variants. The new owner, adjacent
activation/layout suites, active fixtures, and complete architecture suite pass
together with `399 passed in 42.94s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 3.99s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### Pseudo-expanded LeakyReLU passthrough continuation

The raw `_optimize_leakyrelu_transpose_passthrough_chains` implementation is
now a thin indexed compatibility dispatcher. Its production position, public
name, two returned statistics, and historical passthrough-then-fusion order are
unchanged. The call receives Session LayoutState, and the existing indexed
pseudo-LeakyReLU fusion reuses the passthrough owner's differential graph
index instead of rebuilding it.

The resolver accepts only the exact two-branch expansion: `Neg(x)` followed by
negative Relu and singleton-alpha Mul, direct positive Relu, then ordered
`Sub(positive, scaled-negative)`. Both source uses, every intermediate, alpha
dtype/data/shape/provenance, unique producer, graph order, exact consumer slot,
shape and independently dynamic signature, dtype, per-tensor quantization,
layout transition, and public boundary are proven before a plan exists. Typed
rank-three/rank-four INT32 or INT64 permutations, either alpha operand slot,
and immutable constant sources are supported.

Every inverse post alias folds to one source-layout representative, preferring
one public alias when present. Downstream rewrites are grouped per operator and
slot, including repeated aliases. A legacy consumer or public use of the old
transposed Sub result receives one local adapter immediately after the join.
It uses the pre-permutation tensor. The raw helper instead rewrote a selected
post-permutation buffer to an INT32 opposite permutation, potentially changing
its dtype and every unrelated shared use.

The plan captures all rewrites, metadata, public lists, removals, optional
insertion, and complete tensor/operator contracts. It is fully re-resolved
before apply. One differential ModelIRGraphIndex removes the outer adapters,
changes the join output, and inserts the local boundary; LayoutState changes
only after acceptance. The existing fusion then converts the retained Sub in
place to native `LEAKY_RELU` and removes its four private producers using the
same index. Pruning remains success-aware and occurs at the composite end.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. Each model reached the composite helper
four times; all 20 invocations reported zero passthrough rewrites, zero
fusions, and unchanged operator/tensor counts.

The dedicated suite passes with `51 passed`. It covers static/dynamic rank
three and four, INT32/INT64 permutations, both alpha positions, per-tensor
quantization, immutable constant source, one/multiple/public aliases, repeated
slots, legacy and public transposed boundaries, shared post constants, exact
numerical equivalence, candidate limits, idempotence, GraphIndex, LayoutState,
and twenty-seven unsafe transactional passthrough no-op variants. The new
owner, indexed fusion owner, adjacent activation/layout suites, active fixtures,
and complete architecture suite pass together with `467 passed in 42.33s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.01s`. One sequential YuNet
conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
`schema.fbs`
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`,
and `schema_generated.py`
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Temporary outputs were removed and no Tier corpus was run.

### PReLU passthrough continuation

The raw `_optimize_prelu_transpose_passthrough_chains` implementation is now a
thin compatibility dispatcher over
`passes/prelu_passthrough_layout.py`. Its three ordered production positions,
public name, statistic key, and zero-match tensor pruning remain unchanged.
Every call now receives Session LayoutState.

The resolver proves one typed rank-three-or-higher input Transpose, exactly one
PReLU data consumer, at least one typed inverse post-Transpose, unique
producers, graph order, exact consumer slots, static shape, independently
dynamic signature, dtype, per-tensor quantization, provenance, public
boundaries, and known logical/physical layout transitions. Other consumers of
the pre-Transpose result are permitted and retain that adapter. All post aliases
are grouped by exact downstream slot, including repeated inputs, and one public
alias is preferred as the source-layout representative.

Alpha selection retains the historical priority: rank-equal inverse-layout
transpose first, original data second, and the rank-three NCHW channel form
third. Every selected value must broadcast to both the concrete source view
and its dynamic signature. An exclusively owned alpha changes in place; a
shared alpha receives one deterministic `_nhwc` copy that preserves declared
and NumPy dtype, quantization, layout metadata, and ONNX provenance. Scalars,
rank-three/rank-four channel parameters, and ambiguous equal static shapes are
covered.

A legacy consumer or public use of the old transposed PReLU result retains one
local adapter. When its permutation buffer is exclusively owned, that existing
adapter and buffer are reused so the historical correspondence lineage is
unchanged. If the post permutation is shared, the adapter instead uses the
immutable pre-permutation and leaves every unrelated use untouched. INT32 and
INT64 buffers retain their dtype. The raw helper could overwrite a shared post
buffer with an INT32 opposite permutation.

The complete immutable plan is resolved a second time before apply. One
differential ModelIRGraphIndex rewires PReLU and alias slots, compacts only
proven adapters, and retains any required local boundary. LayoutState changes
only after acceptance. Graph-ordered Transpose candidates plus an optional
rewrite limit replace the raw unbounded fixed-point loop.

Pre- and post-extraction characterization ran sequentially on YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. Each model reached the helper six times.
The first four models remained zero-match; FastestDet also retained the former
zero-match 519-to-518 tensor prune. SiNet retained two active invocations of 23
rewrites each. Its float32, float16, correspondence, and both schema artifacts
are byte-identical to `5762128f`, including float32
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and correspondence
`24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`.

The dedicated suite passes with `28 passed`. It covers static/dynamic rank
three and four, INT32/INT64 permutations, in-place and copy-on-write alpha,
scalar and ambiguous alpha forms, pre fan-out, multiple/public/legacy post
boundaries, repeated alias slots, shared post permutations, numerical
equivalence, candidate limits, idempotence, GraphIndex, LayoutState, and
fifteen unsafe transactional no-op variants. The PReLU owner, adjacent
activation/layout owners, active fixtures, and complete architecture suite
pass together with `454 passed in 50.22s`.

TensorFlow-import-blocked explicit direct, default direct, and `-cotof`
conversion pass sequentially with `3 passed in 4.38s`. One sequential YuNet
conversion reproduced all five fixed hashes. Temporary worktrees and outputs
were removed and no Tier corpus was run.

## Completed work

The merged `fb-refactor4` checkpoints included:

- `062ddc4` — centralized DepthToSpace/Gather repair and the permuted-Conv
  statement decoder;
- `a522f1b` — centralized static NHWC Pool layout selection;
- `5eb86e1` — centralized CF Pool-neighbor repair with an explicit
  short-circuit contract;
- `d16265b` — centralized dynamic Pool layout repair and added the exact
  aligned-rank4 decoder;
- `e0bc280` — centralized simple-alias layout repair and moved all
  permuted-Conv decoder consumers into the policy owner;
- `afb5bb5` — centralizes aligned scalar-binary
  shape reconciliation and removes the now-unused aligned-rank4 and Softmax
  parser imports from the exporter.

The current `fb-refactor5` work contains 136 coherent implementation
continuations:

- `3ac19b40` centralizes the ordered fallback that repairs aligned binary
  shapes only when general binary repair made no change and the immediate next
  statement supplies matching BN, direct-return, or channel-first Resize
  evidence;
- `008e4ad0` centralizes the following Resize fallback, including
  exact direct and reshaped BN-constant parsing, preferred-channel selection,
  input-layout guards, and CF evidence propagation;
- `80d1d6a5` centralizes the aligned BatchNorm-constant rewrite itself while
  preserving the different direct and already-reshaped guards;
- `91a0a52d` centralizes LRN output evidence propagation without changing or
  broadening the generated source grammar;
- `b00774a7` centralizes literal static-shape recording while retaining each
  update at its original point in the ordered scan;
- `95fbd0cb` makes the NHWC AveragePool bridge own the CF/NHWC and static-shape
  state resulting from its rewrite;
- `907c91fa` routes all direct-export option reads through the normalized
  request and adds a structural boundary test;
- `5848cc28` adds request-aware optional exporter controls and removes eager
  parsing of unrequested PyTorch settings;
- `e3c03e3d` makes quant type and input/output quant dtype part of the guarded
  immutable quantization controls;
- `4f3d20b0` restores Session-owned consumer counts at the `LoweringContext`
  boundary;
- `6e8a8486` synchronizes lowering-time logical and physical layout mutations
  with the Session before the first post-lowering pass;
- `e7c1457d` centralizes lowering-time operator removal and protects synthetic
  inverse-Transpose fan-out with differential consumer counts;
- `2a2968cc` lazily shares `ModelIRPassState` across the repeated
  Mean/attention registered-pass cluster;
- `514fc683` applies the same bounded reuse contract to the repeated mixed-
  attention through dual-Mul/Concat registered-pass cluster;
- `9b32c680` shares state only across the separately audited late NDHWC-gate/
  cost-volume-scatter pair;
- `1aa636e3` shares state across the following four registered Concat,
  LayerNorm, and transpose-cleanup runners;
- `969d5e26` shares state across the repeated channel-shuffle/Gather-axis and
  unary fan-out runner clusters;
- `251edc58` shares state across four repeated boundary-input BatchMatMul/input-
  unary runner pairs;
- `543d7cc3` shares state across three repeated channel-slice-merge/Pad-Mul
  pairs;
- `93a0295a` shares state across the repeated long singleton/
  Reshape sequences and all three terminal singleton-channel/duplicate-fan-
  out/consecutive-Reshape triplets;
- `9a75c43d` shares state across two repeated QKV attention prefix/bridge
  pairs;
- `417ee06e` shares state across two repeated duplicate-fan-out/quantized-PReLU
  pairs;
- `0c76774e` shares state across two repeated constant-input-fold/redundant-
  Cast pairs;
- `f0dac050` shares state across the fallback and primary absolute-final SE-FC/
  Gather-channel-fan-out pairs;
- `e177face` shares state across the five-runner terminal boundary/layout
  sequence;
- `ce11c27f` shares state across the late Dequantize/Concat/Quantize and unary-
  fan-out sequence;
- `d8b2b58c` shares state across the terminal singleton-MaxPool/consecutive-
  Reshape pair;
- `fcae7233` shares state across the terminal scalar-clamp, unary-passthrough,
  and maximum-zero-to-ReLU sequence;
- `36f73b18` shares state across the late Mean/Mul/Add/Conv, generic SPP, and
  Gather-axis sequence;
- `887db85d` shares state across the late generic SPP and Concat/unary/Conv
  pair;
- `9680fa33` shares state across the absolute-final normalization-Pad and mixed-
  attention pair;
- `a8baec98` shares state across the post-QDQ layout-transpose and unary fan-out
  sequence;
- `916419d9` shares state across the late NCHW channel-shuffle and Gather-axis
  pair;
- `8eaab05b` shares state across the conditional late generic-transpose and
  QKV-bridge pair;
- `e6be5539` shares state across the very-late Gather-axis, constant-fold/Cast,
  and normalization-Pad sequence;
- `fa33fd67` shares state across the terminal hard-activation and optional
  generic-Transpose pair;
- `57d79e3b` shares state across the conditional generic-Transpose, late
  Mean/SPP/Gather, and constant-fold/Cast sequence;
- `a353580b` makes `ArtifactPlan` the only request input to artifact controls
  and progress planning;
- `5e4e14d5` centralizes TFLite evaluation-path selection from returned direct
  artifacts;
- `d7fb5969` centralizes direct report and quantized-artifact
  validation and completion logging without changing messages or skip
  behavior;
- `603d6557` makes the direct fast path the sole direct conversion
  owner and removes the unreachable TensorFlow-failure fallback and duplicate
  post-SavedModel direct serialization path;
- `69a4eccd` centralizes the four identical 19-call layout-recovery prefixes
  without sharing pass state across raw mutation boundaries;
- `c89e94b5` centralizes two identical 22-call attention and quantized-
  recovery suffixes while preserving the separate LayerNorm variant;
- `329306d3` centralizes three identical 16-call layout/reshape/attention
  recovery prefixes while preserving their distinct successors;
- `40dcd142` centralizes two identical 14-call terminal slice/Concat layout-
  recovery sequences while retaining their boundary variants;
- `daab0828` centralizes two identical 11-call terminal affine/Concat/split
  recovery sequences between their distinct raw boundaries;
- `04c7dc03` centralizes three identical 10-call attention/gate/QDQ recovery
  sequences while retaining their distinct successors;
- `1713d089` centralizes two identical 10-call quantized-activation/binary-
  bridge sequences while preserving their conditions;
- `501e616f` centralizes two identical 8-call SiNet terminal recovery sequences
  without crossing shape reconciliation;
- `9bd57ac2` centralizes two identical 7-call pre-Add/Mean attention-recovery
  sequences while retaining their distinct boundaries;
- `ef61c03c` centralizes four identical 6-call SiNet pre-Add/Resize recovery
  sequences while retaining all external boundaries;
- `48dd2324` centralizes the remaining safe-binary and QLinear/Mean/Concat
  5-call families while preserving their conditions;
- `4a9bde4c` shares one differential graph index through each repeated
  prune/reconcile/Reshape-resolution convergence block;
- `20290fce` extends the final indexed convergence boundary
  through HARD_SWISH sanitation and activation fusion without rebuilding
  consumers or the graph index;
- `864af4c9` indexes repeated rank-four channelwise broadcast-constant repair
  while preserving its start-of-pass shared-constant policy;
- `0027ccfa` indexes stale channelwise-binary Transpose repair and shares one
  index across both terminal three-round convergence loops;
- `902cab42` indexes the singleton-Reshape and stale-Transpose Conv-input
  repair pair and shares one index across its primary and fallback invocations;
- `79d30ae1` indexes the wrong-way NCHW-to-NHWC Transpose-before-Conv
  sanitizer and removes its per-match consumer-map rebuild;
- `1ad30cbc` centralizes direct and PyTorch recurrent orphan-step alias repair
  in one Torch-free differential-index owner;
- `2574ae1f` extracts all unbound-input layout repair families to one
  differential-index owner and removes their repeated graph rescans;
- `16bba4ea` extracts quantized RELU/RELU6 Transpose bridge cleanup to one
  differential-index activation owner;
- `bbc9d345` extracts both expanded HardSigmoid QDQ Transpose
  bridge forms to that owner, adds transactional constant preflight, and
  protects every clamp intermediate at the public boundary;
- `515bc99b` extracts expanded MUL/ADD/PRELU QDQ Transpose bridge cleanup and
  shares only the identical constant plan/apply mechanism;
- `49f53b1a` extracts quantized logistic-gated MUL bridge cleanup to a
  dedicated indexed owner with differential alias consolidation;
- `30d00239` gives the wrong-way Transpose-before-Conv sanitizer one dedicated
  owner and removes its duplicate Swish-local implementation;
- `bad1a806` extracts the primary Swish-QDQ NHWC branch rewrite into one
  differential-index owner with an explicit phase result contract;
- `406136b5` extracts its four-family metadata fixed point and shares one index
  across both ordered primary phases;
- `a91d7bad` gives the two identical inverse post-Transpose sweeps one
  differential-index owner while preserving their separate call sites;
- `02462462` extracts transactional late mixed-input Concat
  normalization and shares one maintained index with its following post-
  Transpose cleanup;
- `03742a6a` moves Concat pre-Q/DQ exact-grid bypass to its
  quantization owner and replaces repeated whole-graph maps with one
  differential index;
- `55ad5c88` moves both terminal Transpose/Dequantize sanitation
  subphases to the same owner and maintains one index through edge rewrites,
  operator movement, rename, and removal;
- `f6b62363` moves the Transpose-DQ-Mean-Q bridge to one indexed, fully planned
  quantization-cleanup transaction;
- `3042329e` moves pseudo-op LeakyReLU fusion to one indexed graph-cleanup
  owner with batch producer compaction;
- `9a513d4c` moves the former YOLO MUL-square fold to a generic indexed
  constant-fold owner and protects public intermediates;
- `616a6a6b` moves leading-singleton Gather-to-Reshape cleanup to one indexed
  shape/indexing owner and makes every unsafe metadata or topology case
  transactional;
- `f3da692f` moves marker-gated terminal Softmax/Transpose cleanup to one
  indexed terminal-layout owner and centralizes the propagation marker;
- `e1e8ab39` moves pre-ArgMax channel-layout cleanup to one indexed owner with
  transactional shape and constant-ownership guards;
- `0cfc1ef9` moves exact-grid quantized MaxPool cleanup to one indexed owner
  with transactional topology, grid, and metadata guards;
- `e3dde5a4` moves canonical quantized Logistic cleanup to one indexed owner
  with transactional topology, grid, and metadata guards;
- `2a6178f6` moves canonical last-axis quantized Softmax cleanup to one indexed
  owner with transactional option, grid, and metadata guards;
- `163f9875` moves expanded HardSigmoid QDQ cleanup to one indexed owner with
  a complete four-constant transaction;
- `cc699155` moves quantized TransposeConv QDQ cleanup to one indexed owner
  with a complete filter/output transaction;
- `2b181ed7` moves decomposed InstanceNormalization layout repair to one
  indexed owner with a complete tensor-metadata transaction;
- `558973fd` moves NCHW Concat/global-pool/Conv axis repair to one indexed
  owner with a complete options/metadata/buffer transaction;
- `78ba42ae` moves NCHW Concat/Transpose/(Transpose)Conv axis repair to one
  indexed owner with a complete metadata transaction;
- `bee33d8e` moves mixed singleton NCHW-input repair for NHWC Concat to one
  indexed owner with complete adapter transactions;
- `84ac0fae` moves Swin-style window-partition canonicalization to one indexed
  owner with complete topology/constant/metadata transactions;
- `b134767b` moves the paired Swin-style window-reverse
  canonicalization to that owner with deterministic shared-shape cloning and
  complete topology/constant/metadata transactions;
- `b86ce908` moves the Conv1D-shim Squeeze/Unary/ExpandDims
  canonicalization to one indexed owner with complete shape, topology,
  constant, dtype, and quantization transactions;
- `54d37f35` moves the adjacent rank-four
  Unary/Reshape/ExpandDims variant to that indexed owner, makes shared
  constant changes transactional, and keeps fan-out compatibility bridges
  topological;
- `0837ee1b` moves the unary fan-out bypass to the same indexed
  owner, shares the common Transpose/Squeeze/Unary prefix contract, preserves
  only genuine NCHW side branches, fixes operator ordering, and repairs CAST
  metadata;
- `54384df0` moves flattened InstanceNormalization Conv1D layout
  canonicalization to a dedicated indexed owner with complete decomposition
  and shared-constant transactions;
- `6be406ec` shares that exact normalization prefix with a
  dedicated indexed tencoder residual-gate owner, makes the complete dual-
  branch rewrite transactional, and keeps compatibility bridges topological;
- `2202bd0d` moves the adjacent Squeeze/unary/BatchMatMul layout rewrite to an
  indexed owner with explicit axis-to-adjoint semantics;
- `0ef2050c` moves the decoder deconvolution-input adapter to a complete
  indexed matrix/layout/constant transaction;
- `481098db` moves the terminal Squeeze/Mean decoder adapter to a complete
  indexed axis/layout/output transaction;
- `fd3d1d32` moves the direct decomposed-InstanceNormalization pre/post adapter
  to a complete indexed topology/layout/constant transaction;
- `c7496639` moves its dual-consumer post-Transpose plus side-Squeeze tail to
  the same indexed owner with a transactional local compatibility adapter;
- `50278afa` moves the Squeeze/unary/Reshape tail to the same indexed owner
  with a transactional second-Reshape constant and output-name rewrite;
- the current checkpoint extracts the common SiNet Shuffle residual prefix,
  moves the paired post-MUL Transpose variant to the same indexed module,
  preserves the already-NHWC ADD/PReLU tail, and removes its second legacy raw
  mutator from the compatibility dispatcher.
- the latest checkpoint moves the late residual affine/PReLU fan-out to that
  indexed module, preserves the conv and legacy branches with one inverse
  adapter, and replaces the spatial-size heuristic with semantic shape and
  broadcast validation.
- the current checkpoint moves the deep-skip Resize/dual-Concat/affine tail to
  a staged indexed owner, shares constant and metadata application across
  SiNet owners, and removes four adapters as one preflighted transaction.
- the latest checkpoint moves the adjacent pre-ADD Concat/PReLU fan-out to a
  dedicated indexed owner, shares the terminal affine/PReLU resolver with the
  late-residual owner, removes only the redundant Concat adapter, and retains
  both the direct sibling adapter and legacy NCHW branch.
- the current checkpoint unifies both dual-Resize residual variants in one
  indexed owner, makes direct versus sibling residual-adapter ownership
  explicit, and removes two more independent full-map raw mutators.
- the latest checkpoint moves the shared-post affine/PReLU fan-out to a
  dedicated indexed owner, validates exactly one channel-last Concat-backed
  input and the complete Conv/ADD consumer set, and removes three adapters as
  one preflighted transaction without a fixed spatial-size heuristic.
- the current checkpoint moves the mid-stage Concat/Resize affine residual
  island to a dedicated indexed owner, validates Resize provenance and both
  original/target Concat contracts, and preserves post aliases plus the legacy
  NCHW residual branch in one preflighted transaction.
- the latest checkpoint moves the two-Concat affine tail to a dedicated
  indexed owner, reuses the adapter/Resize branch contracts, validates both
  residual stages and eight constants, and preserves post aliases plus final
  legacy NCHW consumers transactionally.
- the current checkpoint moves the active Softmax-mask residual tail to a
  dedicated indexed owner, validates the channel Softmax and mask-shape axis
  transforms plus six grouped constants, and preserves post aliases and final
  legacy NCHW consumers in one preflighted transaction.
- the latest checkpoint moves the dormant double-Logistic mix-attention path
  to a dedicated indexed owner, validates the complete CA/SA/PA topology and
  both residual-source variants, replaces eight or nine removable adapters as
  one transaction, and corrects the legacy branch rewire that could leave an
  unbound NCHW tensor when this compatibility topology did match.
- the current checkpoint moves the preceding SA/PA MirrorPad compatibility
  path to a dedicated indexed owner, validates the singleton-channel condition
  required for numerical equivalence, preserves both direct-NHWC and legacy-
  NCHW Mul outputs, and removes five or six adapters as one transaction.
- the latest checkpoint moves the general transpose/binary bridge path to a
  dedicated indexed owner, preserves symmetric single-post, mixed-fan-out,
  synthesized legacy-adapter, and asymmetric forms, and applies every accepted
  rewrite only after permutation, source, shape, quantization, public-boundary,
  and graph-order contracts have been resolved and revalidated.
- the current checkpoint folds the five later safe binary recovery helpers
  into the same indexed owner, preserves their legacy-only, single-post,
  mixed multi-post, asymmetric fan-out, and full-post phase order, and shares
  one differential index per sequence invocation rather than rebuilding five
  producer/consumer maps after every rewrite.
- the latest checkpoint moves the binary/Split channelwise tail to a dedicated
  indexed owner, replaces repeated full-graph closure scans with a bounded
  consumer worklist, preserves all six binary operations and the public NCHW
  output, and applies Split axes, Concat options, metadata, and the terminal
  adapter as one revalidated transaction.
- the current checkpoint moves the direct Split channelwise tail to the same
  indexed owner under a separate root plan, adds fully transactional Slice
  begin/size remapping with copy-on-write for shared constants, stops unrelated
  source consumers from entering the closure, and preserves the original
  public NCHW contract through the common terminal adapter.
- the latest checkpoint moves the unary/Split/Concat compatibility island to
  the same indexed owner under a separate root plan, proves every Split branch
  and the one external branch, restricts the layout-only Reshape bypass to a
  true singleton channel, and preserves either a public or intermediate NCHW
  Concat contract as one revalidated transaction.
- the current checkpoint moves the singleton gate/Conv/Concat compatibility
  island to a dedicated indexed owner, proves both auxiliary variants and the
  optional RGB bridge, preserves view-equivalent side consumers, and applies
  all rewrites, metadata updates, constant reshapes, and adapter removals only
  after full plan revalidation.
- the latest checkpoint moves the active Conv-family output passthrough chain
  to a dedicated indexed owner, preserves every producer/unary/binary variant
  and operand slot, groups shared side-constant updates, and removes both
  layout adapters only after a complete revalidated transaction.
- the current checkpoint moves the separate channel-one TransposeConv/Squeeze
  terminal family into the same op-family module under an independent plan,
  validates semantic Squeeze-axis order and the exact graph output, and removes
  its leading adapter only after complete revalidation.
- the latest checkpoint moves the Transpose/RELU/Split all-output island to a
  dedicated indexed owner, proves every inverse-adapter branch and downstream
  input slot, and applies equal-Split metadata plus copy-on-write axis changes
  only after complete revalidation.
- the current checkpoint moves the adjacent exact two-branch
  Split/Conv/RELU/Concat island into a second plan in that owner, preserves
  both Split and Concat orders plus Conv channel changes, and removes four
  adapters only after complete contract revalidation.
- the latest checkpoint moves the direct Split/Conv/Concat bridge to its own
  indexed owner, classifies every direct and post-adapter Concat input, permits
  only a bounded proven NHWC branch interior, and preserves the original local
  NCHW Concat contract through one revalidated transaction.
- the current checkpoint moves unquantized pseudo-Swish passthrough to an
  indexed owner, proves all Logistic/Mul/post-alias slots, preserves an
  optional legacy transposed boundary with the pre-permutation, and eliminates
  mutation of shared post-permutation constants.
- the latest checkpoint moves tanh-expanded GELU passthrough to the same
  activation-family owner under an independent immutable plan, proves all nine
  operations, five source slots, four constants, and every post alias, and
  preserves public or consumed transposed boundaries with one local adapter.
- the current checkpoint moves center/size/offset terminal-head recovery to a
  dedicated indexed owner, replaces operator-distance pairing with complete
  branch/shape/coordinate contracts, and updates shared Reshape constants with
  deterministic copy-on-write before removing three layout adapters.
- the latest checkpoint moves pseudo-expanded LeakyReLU passthrough to a
  dedicated indexed owner, preserves public and legacy boundaries with the
  pre-permutation instead of mutating shared post constants, and hands the
  maintained graph index directly to the existing native-activation fusion.
- the current checkpoint moves PReLU passthrough to a dedicated indexed owner,
  preserves alpha remapping and zero-match pruning, uses copy-on-write for
  shared alpha, retains exact SiNet correspondence lineage, and never mutates
  a shared post-permutation buffer.
- the latest checkpoint moves connected elementwise/Concat NHWC recovery to a
  dedicated indexed owner, replaces the unbounded full-map loop with
  edge-bounded immutable group plans, preserves every unary/binary and legacy
  boundary form, and records converted layout consistently in TensorIR and
  Session LayoutState.
- the current checkpoint moves StridedSlice/Concat fan-in recovery to a
  dedicated indexed owner, proves every typed Slice parameter and whole-group
  boundary before mutation, preserves ordinary/alias lineage exactly, and
  replaces unsafe post-permutation mutation with typed pre-permutation reuse.

The extraction preserves the ordered source-rewrite behavior. Layout evidence
continues to mutate only the per-run CF/NHWC sets; repair context maps remain
shared. Rules that formerly used `continue` return an explicit short-circuit
result to the exporter. Exact generated-statement grammars remain rule-local or
use the shared Torch-free parser owner.

No dependency was added and no TensorFlow path was introduced. The latest
checkpoint includes one sequential direct-backend integration smoke; no Tier
corpus run was performed.

## Current branch and changed files

Branch: `fb-refactor5`, tracking `origin/fb-refactor5`.

The latest implementation checkpoint changes:

- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`;
- `onnx2tf/tflite_builder/passes/stridedslice_concat_layout.py`;
- `tests/test_flatbuffer_direct_indexed_stridedslice_concat_layout.py`;
- `tests/test_flatbuffer_direct_architecture.py`;
- `docs/flatbuffer_direct_architecture.md`;
- this handoff document.

The expected handoff state after committing and pushing is an empty `git
status --short` with local `fb-refactor5` equal to `origin/fb-refactor5`.

## Important design decisions

- The binary/Split tail has its own `split_channelwise_layout.py` owner rather
  than extending the binary bridge owner. Its semantic boundary is the closed
  axis-sensitive Split/Concat closure, not a source-line threshold.
- The consumer worklist is bounded by the current operator and input-edge
  count. A candidate is accepted only when every converted tensor terminates
  at an accepted closure operator or the sole public output; the pass does not
  silently expose NHWC data to an unsupported legacy consumer.
- The public tensor retains its original NCHW shape and receives the terminal
  adapter output. A deterministic private NHWC tensor receives the producer
  output. The proven input permutation constant is shared instead of creating
  another equal buffer.
- Split axis constants are immutable plan inputs. Private axes change in place;
  shared axes receive deterministic clones that preserve both TensorIR and
  NumPy INT32/INT64 dtype. Every plan is re-resolved before these writes.
- The direct Split root seeds closure discovery from Split outputs, not from
  the original NHWC Transpose input. Unrelated raw-source consumers remain
  untouched and do not affect the accepted tail.
- Slice begin/size tensors are grouped by identity across all planned uses.
  They change in place only if every real consumer slot participates with the
  same value; otherwise planned uses share one deterministic typed clone.
  Invalid bounds, conflicting roles, mutable constants, and per-axis
  quantization reject the transaction.
- The unary-root island requires the pre-Transpose and pre-unary edges plus
  every Split branch to have exact closed ownership through the Concat. Every
  Split output appears exactly once, with at most one allowed branch unary,
  and there is exactly one external Concat input.
- The external Reshape bypass is valid only for `[N,H,W,1]` to `[N,1,H,W]`
  with matching static/dynamic shape, dtype, per-tensor quantization,
  provenance, and physical-layout evidence. Other consumers of the Reshape
  output continue using the original NCHW view.
- The unary-root adapter is a local boundary rather than a sole-public-output
  boundary. It preserves the original Concat tensor name and NCHW metadata for
  either graph outputs or existing downstream consumers while moving only the
  producer output to a private NHWC tensor.
- The singleton-gate owner accepts only the exact gate topology and true
  `[N,H,W,1]`/`[N,1,H,W]` view adapters. Both the direct auxiliary and Logistic
  auxiliary variants are explicit plan forms; no model-name or chain-name
  special case is used.
- Optional RGB propagation removes an input Transpose only when its typed
  permutation, ownership, order, shape/signature, dtype, and quantization are
  proven. A direct stale-NCHW input is accepted only for a private constant
  with physical NHWC evidence, and its data buffer is reshaped together with
  its metadata.
- Every singleton-gate input rewrite, tensor/operator contract, metadata
  update, and removal is immutable plan state and is re-resolved before apply.
  Shared view-equivalent adapter consumers are rewired, unrelated source
  consumers are preserved, and an unsupported fan-out rejects the whole
  candidate without mutation.
- Conv-output passthrough and channel-one terminal Squeeze chains remain
  separate plans because the former preserves an intermediate NHWC boundary,
  while the latter terminates at a graph output and remaps Squeeze axes. They
  share only typed adapter, contract, constant-planning, and apply primitives.
- The general Conv-output owner preserves all three Conv producer types, all
  historical unary/binary operations, and binary operand slots. It validates
  static/dynamic broadcast results and never commutes SUB or DIV.
- Rank-four binary constants are grouped by tensor identity. All planned slots
  share one NHWC value; an unrelated consumer forces one deterministic clone,
  while an exclusive constant changes in place. Scalar constants remain
  unchanged.
- The inverse-adapter output metadata is derived from the converted final chain
  output rather than blindly permuting the existing output metadata. The plan
  captures and revalidates every tensor/operator, constant, input/output slot,
  metadata update, and removal before its first write.
- Terminal Squeeze axes are remapped by semantic label. Each squeezed static
  and dynamic dimension must equal one, and the surviving semantic axis order
  must match between the original NCHW and converted NHWC paths. A spatial-
  only squeeze that would reorder a public rank-three output is rejected.
- Non-scalar terminal-chain constants are allowed only before Squeeze and use
  the common grouped rank-four planner. Post-Squeeze binary operations must use
  scalar constants. Public intermediates, a nonterminal graph output, a second
  Squeeze, or any output consumer rejects the complete plan.
- The RELU/Split all-output island remains separate from the exact
  Conv/Concat root and the direct Split/Conv/Concat bridge. Its semantic
  boundary is that every Split result immediately returns to NHWC through one
  owned inverse Transpose; it does not absorb either adjacent Conv topology.
- TFLite `SPLIT` is an equal-partition operation. The owner verifies that the
  channel count divides the output count and that every static/dynamic output
  shape matches the equal result. The former synthetic fixture's unequal 2/4
  metadata was corrected to the valid 3/3 contract.
- A downstream operator may consume several former post-Transpose aliases or
  one alias more than once. Replacements are grouped per operator and input
  slot before apply, preventing one branch rewrite from overwriting another.
- Axis constants use copy-on-write based on exact consumer slots. Private
  INT32/INT64 buffers change from axis 1 to 3 in place; shared buffers retain
  their original value while the Split receives a deterministic typed clone.
- The exact Conv/Concat root is independent from the all-output root even
  though both share one op-family owner. It owns exactly four adapters and one
  Conv branch. The direct Split/Conv/Concat bridge is a separate owner because
  it accepts a generic bounded NHWC interior and retains a local legacy NCHW
  boundary rather than owning a fixed Conv/RELU chain.
- Split-output position and Concat-input position are semantic order. The plan
  evaluates both positions but never reorders them. Conv may change channel
  width, and both original NCHW and target NHWC Concat shapes must prove the
  same final adapter contract.
- Changing the Conv and post-RELU inputs is safe only when the removed adapter
  endpoints have equal dtype and quantization. Conv filter/bias inputs are not
  rewritten, but each must have resolved provenance before the Conv.
- The pre-Transpose, branch-to-Conv adapter, Conv-to-RELU adapter, and final
  Concat adapter are an all-or-nothing removal group. Any fan-out or public
  boundary inside that group rejects the plan transactionally.
- The direct bridge classifies every Split and Concat edge before mutation.
  All non-branch Split outputs feed one Concat directly, every remaining
  Concat input is an exclusive post-adapter output, and a bounded graph walk
  must connect the inverse-adapter branch to at least one post-adapter input.
- The bridge does not validate a Conv opcode or a model-specific chain. The
  existing NHWC interior is left unchanged; only proven boundary adapters,
  Split/Concat axes, and exact consumer slots are part of the transaction.
- The original Concat tensor remains the local NCHW contract. A deterministic
  private NHWC tensor becomes the Concat producer output and one new post
  adapter reuses the proven pre-adapter permutation. Split axes preserve their
  INT32/INT64 dtype and use copy-on-write when shared.
- Unquantized pseudo-Swish is independent from Swish-QDQ. Its semantic root is
  one `Logistic(x) * x` residual island with inverse post aliases, not a
  Dequantize/Quantize closure. Both Mul operand positions are preserved.
- Typed permutations may be rank two or higher and INT32 or INT64. Static
  shape and dynamic signature views must match in both directions. Known
  logical/physical layouts must describe the same transition; unknown layout
  remains valid when tensor views prove the permutation directly.
- Post aliases are grouped by exact consumer slot. A public alias becomes the
  representative when present. Multiple public aliases reject the candidate
  because removing them would otherwise change the output-name contract.
- A legacy consumer of the old transposed Mul tensor is preserved by inserting
  one adapter immediately after Mul. The adapter uses the immutable pre-
  permutation instead of reversing a post-permutation buffer in place, so
  shared post constants and unrelated consumers cannot be corrupted.
- Immutable constant sources are supported without data mutation. Moving
  Logistic and Mul to the source view is exact because Swish is elementwise;
  runtime intermediates and adapter endpoints remain fully typed and
  per-tensor quantized.
- Tanh-GELU uses its own immutable plan within the activation owner. It accepts
  only the complete nine-operation approximation and proves all five source
  slots, linear intermediate ownership, unique production, order, and four
  immutable singleton constants before any layout rewrite.
- GELU post aliases follow the same exact-slot and public-representative
  contract as Swish. If the old final transposed tensor remains observable, a
  local adapter reuses the pre-permutation; neither activation pass mutates a
  shared post-permutation constant.
- Center/size/offset association is semantic rather than positional. One
  singleton-channel center branch, one activated size branch, and one direct
  offset branch must agree on batch/spatial shape and the size/offset data view;
  more than one compatible triple rejects instead of choosing the nearest
  Transpose.
- GatherND coordinate roles are proven from the dynamic Reshape and bounded
  immutable integer grids. Every coordinate Concat consumer slot must belong
  to the planned size/offset gathers before its inputs are reordered.
- Reshape shape constants are grouped by identity and exact slot. Exclusive
  INT32/INT64 constants rotate in place; a constant with any unrelated use is
  cloned once for the planned size/offset Reshapes. Inferred `-1` values and
  option metadata stay symbolic.
- Pseudo-LeakyReLU layout recovery and native activation fusion remain one
  ordered composite contract. The passthrough owner exposes a focused
  transaction entry point for rejection tests, while production always runs
  the indexed fusion afterward on the same ModelIRGraphIndex.
- The Sub join is order-sensitive: only
  `positive_relu - alpha * relu(-x)` is accepted. Every inverse post alias is
  grouped by exact slot, one public alias may be representative, and observable
  legacy NCHW use gets a local adapter immediately after the join.
- The local adapter reuses the typed immutable pre-permutation. No post-
  permutation constant is reversed or recast, so an INT64 or shared constant
  retains both its bytes and every unrelated consumer contract.
- The exporter remains the ordered orchestration owner; match/guard/rewrite
  decisions move to `pytorch_fast_precanonicalize_policy.py` one coherent
  family at a time.
- Indexed helpers receive the current line index, shared source lines, mutable
  layout evidence, and the shared repair context. They do not rescan the full
  generated source unless the preserved rule already required a bounded scan.
- Former loop `continue` behavior is represented explicitly in helper results;
  extraction must not silently allow later rules to run.
- General binary repair remains first. The downstream-evidence fallback is
  called only from its unchanged no-rewrite branch, and its returned CF
  evidence is visible to the following Resize repair in the same scan.
- The fallback deliberately retains its narrower positional grammar and legacy
  `_in` naming evidence. It additionally requires an immediate matching BN,
  direct return, or channel-first Resize; mismatched channels and mixed-layout
  names remain no-ops.
- General Resize repair also remains first. The input/BN-evidence fallback runs
  only afterward, uses an immediate matching direct or reshaped BN constant as
  the preferred channel hint when available, and otherwise retains the legacy
  input/source channel fallback. Its returned CF evidence remains visible to
  Pool and later aligned-constant decisions in the same ordered scan.
- Explicit NHWC Resize inputs and already-channel-first target shapes remain
  no-ops. BN evidence refines the preferred channel count but is not a
  prerequisite for the legacy CF-input repair.
- Direct aligned BatchNorm constants require a registered channel count that
  matches the generated target channel before a reshape is introduced.
  Already-reshaped constants intentionally retain the older, narrower rule:
  their explicit reshape channel drives normalization without requiring a
  registered-buffer channel lookup. Both forms still require CF input and a
  BatchNorm-derived attribute name.
- LRN output propagation is state-only: exact CF input evidence adds the output
  to the CF set, removes stale NHWC evidence, and copies only a known rank-four
  static input shape. It does not mark the source file changed or rewrite the
  LRN statement.
- Rewritten-shape caching accepts only a literal `target_shape=[...]` or the
  exact trailing aligned shape. Dynamic and unparseable expressions do not
  replace existing cache entries, and binary/Resize/Pool callers still update
  the shared context immediately after their successful rewrite.
- The NHWC AveragePool bridge keeps its returned-name contract, but successful
  calls now update the layout sets and all four affected static-shape entries
  internally. To preserve behavior, the cached state shape is still recomputed
  from the pre-rewrite Pool shape after the layout sets change; it is not
  replaced by the rendered rewrite target.
- `ConversionRequest.from_kwargs` is the direct exporter's only raw-kwargs
  boundary. Quantization validation receives `request.options`, normal option
  reads use `request.get`, and typed artifact decisions remain on
  `request.artifacts`. Checkpoint `907c91fa` mechanically converted all 36
  former raw reads without changing keys, defaults, coercions, public return
  values, or downstream arguments.
- `resolve_requested_exporter_controls` now owns seven artifact-specific
  settings. It performs no option reads when SavedModel, PyTorch, and integer
  calibration are all unrequested. Requested output paths, persistence,
  timeout conversion, shape/test data, and custom-input values preserve their
  existing defaults and dependencies.
- Requested quantization controls now also own `quant_type`,
  `input_quant_dtype`, and `output_quant_dtype`. The builder reads these values
  only from the resolved immutable mapping; when quantization is unrequested it
  uses the legacy local defaults without touching the corresponding options.
- Artifact execution controls, exporter controls, and export-progress labels
  now accept `ArtifactPlan` directly. They do not receive independently
  reconstructed split, quantization, SavedModel, PyTorch, or calibration
  booleans. Derived PyTorch artifacts are normalized once by
  `ArtifactPlan.from_options`, and all downstream policy sees the same
  immutable dependency decision.
- TFLite evaluation consumes the direct builder's returned artifact mapping
  through one fixed seven-key selector. The three compatibility-layer exit
  paths no longer own parallel key-copy chains, cannot diverge in variant
  order, and do not infer a path for an artifact the builder did not return.
- `LoweringContext.tensor_consumer_count` is populated from
  `ConversionSession.tensor_consumer_count`, not an empty compatibility
  dictionary and not a new ONNX scan. This restores the original fan-out guard
  used by inverse-transpose elision and preserves duplicate input occurrences.
- `LoweringContext.set_tensor_layout()` is the lowering-time layout mutation
  boundary. It normalizes and writes `TensorIR` metadata and immediately
  records the same logical/physical values in the Session-owned `LayoutState`.
  Shape-family edge-Pad passthroughs, integer-linear Resize casts, and rank-three
  Resize adapters no longer assign layout fields directly. This fixes observed
  pre-pass staleness without adding an eager ModelIR-wide synchronization.
- `LoweringContext.add_operator()` increments a differential consumer count for
  every emitted input occurrence. `remove_operator()` decrements those counts
  and removes only producer entries owned by the removed object. The inverse
  Transpose helper uses the authoritative ONNX count when present; otherwise it
  adds the pending inverse use to the current synthetic IR count. An exclusive
  pair is still elided, while a synthetic side consumer keeps its producer.
  This replaces the only direct op-builder deletion and adds no partial-graph
  scan.
- `ModelIRPassStateScope` is lazy and identity-bound. A group with a successful
  model-only preflight acquires the state; subsequent adjacent groups reuse it
  and report `state_built: false`, so diagnostic `state_build_count` reflects
  actual construction rather than pass invocation count. The scope is never
  carried across legacy helpers that mutate ModelIR outside the differential
  index. All six production occurrences retain the exact order
  transpose-Mean, Mean/Mul/Add/Conv, optional LayerNorm, terminal Mean, SE conv,
  SE FC, and optional Conv attention.
- Each repeated shape-convergence block uses one `ModelIRGraphIndex`.
  Dead-operator pruning updates that index through `remove_operators`; the
  following reconciliation and dynamic-Reshape steps change only metadata,
  options, and constant data. The first block builds its own index. The final
  convergence owner supplies one index to the second block and retains it
  through HARD_SWISH sanitation, a second Reshape/reconcile cycle, activation
  fusion, and final reconciliation. Standalone callers retain compatibility
  fallbacks, and an index for another ModelIR is ignored safely.
- Indexed activation fusion preserves the former graph-order and case-
  normalized op matching. It queries producer/activation fan-out from the
  index, updates producer outputs through `_set_operator_outputs`, and removes
  fused activation operators through `remove_operator`. It no longer rebuilds
  the full consumer map for every successful match. Differential single-
  operator removal drops empty type buckets so its type dispatch exactly
  matches a fresh index.
- Rank-four channelwise broadcast-constant repair takes an optional matching
  graph index and otherwise builds exactly one. It enumerates only the exact
  binary-op family, queries producer layout evidence through the index, and
  routes cloned-constant input changes through the differential setter. Its
  consumer fan-out map is intentionally snapshotted once from that index:
  clone-versus-in-place decisions therefore retain the former start-of-pass
  behavior even after earlier candidates update live consumers.
- Stale channelwise-binary Transpose repair also accepts an optional matching
  graph index. It retains exact graph-order binary matching, resolves adapter
  and NHWC peer producers through the index, and requires the indexed adapter
  consumers to equal the current binary index. Successful rewrites use the
  differential input setter and operator removal; a fan-out adapter remains
  untouched. `_run_indexed_binary_layout_convergence` owns the existing three
  broadcast-repair, Transpose-repair, and reconciliation rounds and supplies
  the same index to all nine calls in both primary and fallback finalization.
- The same scope contract covers only the five repeated gate-layout sequences
  that were audited as contiguous registered runners. Four keep the exact
  mixed-attention, elementwise-gate, Pad, dual-postconv-gate, NDHWC-gate,
  cost-volume-scatter, Add/Concat-suffix, and dual-Mul/Concat order. The fifth
  starts at elementwise-gate exactly as before. All eight runners retain
  standalone behavior through an optional `state_scope` argument, and the
  later isolated mixed-attention, Pad, NDHWC, cost-volume, and dual-Mul calls
  intentionally do not share this scope.
- The late mixed-attention/NDHWC/cost-volume candidate is not one valid scope:
  the raw dequantize/HardSigmoid/quantize optimizer between mixed attention and
  NDHWC is a hard boundary. A new scope is therefore constructed only for the
  immediately adjacent NDHWC and cost-volume runners, and ends before the raw
  convolution-affine optimizer. Both runners preserve standalone behavior.
- After that convolution-affine boundary, axis-3 constant-Concat,
  Dequantize/Concat/Quantize, LayerNorm-statistics, and generic transpose
  cleanup form one independently audited four-runner scope. Each runner either
  already accepted a scope or now exposes the same optional
  standalone-compatible argument; the scope ends before the conditional raw
  elementwise-roundtrip optimizer.
- Two repeated cluster families now have explicit helper-owned scopes. The
  channel-shuffle helper preserves two-way, NHWC, NCHW, and Gather-axis order
  at all five call sites; only its final invocation enables the already-
  contiguous generic transpose and unary/binary fan-out suffix. The separate
  unary helper preserves passthrough, unary fan-out, and unary/binary fan-out
  order at four call sites. Every runner remains callable standalone through
  an optional scope argument.
- Four repeated boundary-input BatchMatMul/input-unary pairs now use a small
  helper-owned scope. The boundary BatchMatMul runner and the three-spec input-
  unary runner expose the same optional standalone-compatible scope argument.
  No scope crosses the legacy transformations surrounding any occurrence.
  Their two stale `_build_tensor_consumer_map` imports are removed; neither
  module constructs an ad hoc consumer map for these runners.
- Three repeated channel-slice-merge/Pad-Mul pairs now use a two-group helper-
  owned scope. Both runners retain optional standalone-compatible scope
  arguments, and the scope ends before the legacy optimizer following each
  pair.
- Two long singleton/Reshape sequences now use one flag-controlled helper-
  owned scope per occurrence. The first retains generic transpose cleanup and
  terminal multi-branch gate cleanup; the second retains reshape-only
  duplicate fan-out cleanup and disables only the former spatial post-Concat
  variant. Four singleton-Reshape-family runners, three graph-cleanup runners,
  singleton MaxPool, and multi-branch gate retain standalone behavior through
  optional scope arguments.
- The three later singleton-channel/reshape-only-duplicate/consecutive-Reshape
  triplets use a target-parameterized helper. Two invocations use the primary
  ModelIR and Session layout state; fallback relowering passes `fallback_ir`
  and no LayoutState, preventing state identity from crossing conversion
  instances. The terminal singleton-MaxPool/consecutive-Reshape pair remains
  outside this target-parameterized helper and owns a separate bounded scope.
- Two repeated QKV attention prefix/bridge pairs use a two-runner helper-owned
  scope. The four prefix specs and two bridge specs retain exact order and
  diagnostic grouping. Both runners expose optional standalone-compatible
  scope arguments; the separate later bridge-only call remains independent.
- Two repeated duplicate-fan-out/quantized-PReLU pairs use a helper-owned
  scope. The helper forwards
  `enable_duplicate_transpose_fanout_optimizations` unchanged, then runs all
  four PReLU specs. The quantized-PReLU runner retains standalone behavior
  through an optional scope argument, and no scope crosses the following raw
  quantized TransposeConv cleanup.
- Two repeated constant-input-fold/redundant-Cast pairs use a helper-owned
  scope. The constant Pad, Pool, and Cast specs retain order before the
  redundant widening-alias and narrowing-chain specs. Both runners expose
  optional standalone-compatible scopes, and neither production scope crosses
  the immediately following legacy mutator.
- The fallback and primary absolute-final SE-FC/Gather-channel-fan-out pairs
  use a target-parameterized helper. Fallback receives `fallback_ir` and no
  LayoutState; primary receives the main ModelIR and Session state. Gather
  channel fan-out now exposes an optional standalone-compatible scope, and
  neither target's scope crosses shape reconciliation.
- The five-runner terminal dual-Mul/Concat, boundary-input, Pad, generic
  transpose, and Gather-channel-fan-out sequence uses one helper-owned scope.
  Boundary-input cleanup now exposes an optional standalone-compatible scope.
  Architecture checks fix the raw InstanceNorm predecessor and conditional
  Mean/attention successor as hard boundaries.
- The late Dequantize/Concat/Quantize, unary-passthrough, and unary-fan-out
  sequence uses one helper-owned scope. All three runners already expose
  optional standalone-compatible scopes. Architecture checks fix the raw
  Dequantize/HardSigmoid/Quantize predecessor and independently indexed Swish
  successor as hard boundaries. The Swish owner maintains its own differential
  index rather than reusing pass state across that semantic phase boundary.
- The terminal singleton-MaxPool/consecutive-Reshape pair uses one helper-
  owned scope. The two singleton-MaxPool specs retain their order before the
  general Reshape cleanup spec. Architecture checks fix the conditional
  elementwise-roundtrip predecessor and conditional Conv/Pool-output successor
  as hard boundaries.
- The terminal scalar-clamp, unary-passthrough, and maximum-zero-to-ReLU
  sequence uses one helper-owned scope. Clamp and maximum-zero runners now
  expose optional standalone-compatible scopes. Their op-type rewrites use
  `ModelIRGraphIndex.replace_operator_type()` instead of direct assignment, so
  later passes see current type dispatch without a full refresh. Architecture
  checks fix the conditional terminal layout-recovery predecessor and raw
  SiNet successor as hard boundaries.
- The conditional generic-Transpose, late Mean/Mul/Add/Conv, generic SPP,
  Gather-axis, and constant-fold/Cast sequence uses one helper-owned scope.
  Disabling layout optimization skips only the first runner. Architecture
  checks fix the complete order, runtime flag, shared scope keywords, and raw
  shape-extract/ExpandDims boundaries.
- The late generic SPP and Concat/unary/Conv pair uses one helper-owned scope.
  Concat/unary/Conv cleanup now exposes an optional standalone-compatible
  scope and retains its differential index mutations. Architecture checks fix
  the raw StridedSlice/Pad/Concat predecessor and raw shape-extract successor
  as hard boundaries.
- The absolute-final flattened-normalization Pad and mixed-attention pair uses
  one helper-owned scope. Normalization-Pad cleanup now exposes an optional
  standalone-compatible scope. The helper preserves `include_instance=False`
  and `include_flatten=True`; architecture checks fix those flags and the raw
  InstanceNorm/dynamic-rank shape-rewrite boundaries.
- The post-QDQ layout-transpose, unary-fan-out, and unary/binary-fan-out
  sequence reuses the existing helper with compatible mode flags. Four prior
  invocations keep their default unary-passthrough path; one new invocation
  enables layout-transpose and disables unary-passthrough. Architecture checks
  fix the unique flag combination and the raw Softmax/transpose-binary
  boundaries.
- The late NCHW channel-shuffle/Gather-axis pair reuses the existing helper
  with two-way and NHWC shuffle disabled. Five prior invocations retain both
  modes by default. NHWC and NCHW shuffle op-type changes now use
  `replace_operator_type()`; real rewrite tests assert the reused type index.
  Architecture checks fix the unique flag combination and raw Reshape/QKV
  boundaries.
- The conditional late generic-transpose/QKV-bridge pair reuses the QKV helper
  with prefix disabled. Two prior invocations retain the default prefix-plus-
  bridge path. The new invocation forwards `optimize_layout_transpose_chains`
  to the layout mode while always running bridge cleanup. Architecture checks
  fix the runtime flag and raw shape-extract/split-Conv boundaries.
- The very-late Gather-axis, constant-fold/Cast, and normalization-Pad sequence
  uses one wrapper-owned scope. The constant-fold/Cast helper accepts an
  optional external scope; both production invocations now receive their
  enclosing wrapper scope. The normalization include flags remain false/true.
  Architecture checks fix both external scopes, runner order, flags, and raw
  repair/Reshape boundaries.
- The terminal hard-activation and conditional generic-Transpose pair uses one
  helper-owned scope. Hard activation now exposes an optional standalone-
  compatible scope. Its late false/true/true/reversed flags are unchanged,
  generic Transpose still depends on the runtime layout switch, and the scope
  cannot cross either neighboring raw rewrite.
- The singleton-Reshape and stale NCHW-to-NHWC Transpose Conv-input repairs
  accept an optional matching `ModelIRGraphIndex` while retaining standalone
  compatibility. Their primary and fallback pair owner constructs one index,
  and successful Conv input rewrites and adapter removals update that index
  differentially. Exact filter input-channel, one-consumer, graph-output,
  tensor-shape, and Transpose-permutation guards are unchanged. The later
  standalone stale-Transpose cleanup remains a separate compatibility call
  because intervening raw mutations form an ownership boundary.
- The wrong-way NCHW-to-NHWC Transpose-before-Conv sanitizer has one Torch/
  TensorFlow-free semantic owner in `passes/conv_input_layout.py`. It owns one
  `ModelIRGraphIndex` for an invocation with Transpose candidates, enumerates
  indexed roots, validates every indexed consumer before changing a shared
  adapter, rewrites all accepted Conv inputs through the indexed global-input
  replacement helper, and removes the adapter differentially. A graph without
  any Transpose skips index construction while preserving the former tensor-
  pruning side effect. The lowerer compatibility wrapper and the independent
  safety-valve phase inside the Swish-QDQ optimizer both delegate to this same
  owner; the latter preserves its historical execution point and removal
  statistics. Exact permutation, rank-four metadata, all-consumers-are-Conv,
  filter-channel, graph-output, and nonempty-consumer guards are unchanged.
- The primary Swish-QDQ branch phase has one Torch/TensorFlow-free owner in
  `passes/quantized_swish_layout.py`. Its result carries the exact branch and
  pre-Transpose counts plus an immutable rewritten-tensor set into the
  existing later phases. A matching supplied index is reused; otherwise one
  index is built only when a Transpose exists. Graph-order candidates, all
  source/gate/data/tail guards, both DQ rewrites, and unused pre-Transpose
  removal use the maintained index. Only source edges and an unused root can
  change, so downstream consumers are read directly without full-map copying.
  Shared-input ordering, quantized and float tails, peer-Swish acceptance,
  spatial and concat-closure modes, public boundaries, fan-out, metadata
  permutation, and ordered restart retain the former implementation exactly.
- The Swish-QDQ metadata phase keeps unary quantization, binary broadcast,
  Pool/Resize channel propagation, and strict Concat-tail normalization in one
  fixed-point owner because every family mutates the same rewritten-tensor
  state. It iterates only the graph-ordered relevant type buckets and reuses
  stable indexed consumers; it performs no topology mutation. Empty seeds
  allocate no index. A module-level runner gives the branch and metadata phases
  the same index, reducing the complete primary sequence to one construction.
  Public outputs, shape/signature copy, broadcast fallback, channel guards,
  normalized axis, tail fan-out, Concat/quantized metadata, and fixed-point
  restart semantics remain unchanged. The shape/signature copier is also the
  explicit owner used by the later Dequantize-input repair, preserving the
  previously hidden cross-phase dependency without a lowerer-local closure.
- The two Swish-QDQ inverse post-Transpose sweeps share one semantic owner.
  The first remains before late Concat normalization; the second is called by
  the shared late-phase runner after normalization. Empty rewritten state and
  Transpose-free graphs return without an index. Otherwise indexed graph-order
  candidates, global alias replacement, and differential removal preserve
  ordered restart without per-removal full scans. Public aliases, wrong
  permutations, and inputs outside the rewritten set remain protected; alias
  chains and arbitrary consumer fan-out are rewired before removal.
- Late Swish-QDQ Concat normalization validates one complete transaction before
  mutation. Direct and Dequantize-wrapped pre-Transposes, rank-four normalized
  shapes, a private Concat output, and the strict Quantize/all-inverse-
  Transpose tail must all agree. Accepted Concat/DQ edge rewrites update one
  maintained index; axis and shape/signature metadata commit together; and
  only newly unused input adapters are removed. The owner restarts after each
  accepted transaction to avoid stale indices after compaction. Its runner
  gives the immediately following inverse-post owner that same index while
  retaining the original statistics and phase order.
- Concat pre-Q/DQ bypass remains intentionally narrower than generic redundant-
  quantization cleanup. The Quantize input must be the direct output of a
  Dequantize, source and destination quantized tensors must share the complete
  exact grid, and no arithmetic intermediate is accepted. The owner rewires
  only the Concat edge; existing Q/DQ operators remain available to later
  cleanup, preserving the historical ordered pipeline. It restarts after each
  indexed edge change and performs the former pruning side effect even when no
  Concat exists, without allocating an index in that no-candidate case.
- Terminal Transpose/Dequantize sanitation retains two explicit counters and
  subphases even though a first-subphase match becomes eligible for the second
  after its indexed reorder. This preserves the established stats and phase
  semantics. Operator order changes use `remove_operator()` followed by
  `insert_operator()` on the same index; public output rename occurs before
  differential Transpose removal so the Dequantize remains the sole producer.
  Graphs missing either required operator type still receive historical tensor
  pruning without paying for an index.
- The Transpose-DQ-Mean-Q bridge commits only after the mapped axes, reduced
  metadata, bridge permutation, and both unique tensor names are valid. The new
  preserving Transpose is inserted immediately before the current Quantize,
  then the old pre-Transpose is resolved by object identity and removed from
  the same index. This preserves valid ordering while making invalid-
  permutation rejection a complete no-op instead of retaining the former
  partial DQ/Mean metadata mutation.
- Pseudo-LeakyReLU fusion remains an exact ordered grammar rather than a
  commutative algebraic matcher: positive RELU is SUB input zero and the scaled
  negative branch is input one. Only MUL's singleton alpha may swap sides. The
  retained SUB is converted through indexed type/input updates and its options,
  axis semantics, version, and ONNX provenance are reset to the same defaults
  as a fresh `OperatorIR`; all four private producers are then removed in one
  batch compaction.
- The MUL-square fold's semantic owner is model-neutral; only the compatibility
  wrapper and historical stats key retain `yolo` naming. The self-square must
  use the identical tensor name in both MUL slots, while constant side inputs
  at the pre/anchor/scale MULs remain commutative. Fused data is calculated in
  float32, checked finite, cast back to anchor dtype, and receives cloned anchor
  quantization. All three removable intermediates are now explicitly rejected
  when public before any tensor or edge mutation.
- Leading-singleton Gather-to-Reshape cleanup accepts one signed integer zero
  in either a scalar or TFLite-legalized singleton buffer, because both select
  the only leading slice without changing element order or count. It requires
  axis zero after negative-axis normalization, batch dimensions zero, a
  statically fixed leading-one signature, exact rank-reduced tail shape and
  signature, matching dtype and quantization, one topologically later Reshape
  consumer at data input zero, and no public or duplicate-produced Gather
  output. All guards finish before the indexed Reshape edge is changed and the
  Gather is removed. Missing Gather/Reshape families still receive historical
  unused-tensor pruning without allocating an index, and the active
  `LayoutState` receives the same pruning at both production call sites.
- Terminal Softmax/Transpose cleanup consumes only the shared
  `_SOFTMAX_NHWC_PROPAGATED_MARKER` produced by the preceding canonicalizer;
  the string literal no longer has two owners. Public outputs remain the
  deterministic candidate order. The maintained index requires the output to
  have no internal consumer and one Transpose producer, the private Softmax
  intermediate to have one Softmax producer and exactly that Transpose
  consumer, and Softmax to precede Transpose. Duplicate producers and a
  Softmax intermediate exposed at either public boundary are rejected; a
  terminal output cannot also be an input. Rank-four source
  shape/signature, destination existence, and cloned quantization are planned
  before mutation. Commit removes only the marker, uses the lineage-aware
  indexed output setter, copies the former source metadata onto the existing
  public tensor object, and removes the Transpose differentially. Missing
  Softmax/Transpose families retain historical pruning without index
  construction, including optional `LayoutState` pruning at the production
  call site.
- Pre-ArgMax terminal layout cleanup accepts only an exact rank-four
  `[0,3,1,2]` Transpose whose private output has one topologically later
  `ARG_MAX` consumer at data input zero. The signed INT32/INT64 singleton axis
  must normalize to NCHW channel axis one and is remapped to NHWC axis three.
  Source, transposed, and output shape/signature metadata must prove the same
  permutation and rank-reduced output, and source/adapter dtypes must agree.
  A private axis constant is updated in place; a shared or public-input/output
  axis constant receives a uniquely named clone with its NumPy dtype and
  cloned quantization. This preserves the public constant value that the
  former rule could silently change. All clone data and topology guards finish
  before either constant or edge mutation. Indexed ArgMax input replacement
  and Transpose removal preserve the fixed point, historical stats, and
  lineage; post-prune `LayoutState` synchronization registers any clone.
  Missing required families retain historical pruning without index
  construction.
- Quantized MaxPool cleanup accepts only an exact linear
  `DEQUANTIZE -> MAX_POOL_2D -> QUANTIZE` chain. Input and output grids must
  use the same INT8 or UINT8 dtype and exactly equal positive finite scale and
  in-range zero point; approximate equality is not sufficient because a
  quantized MaxPool builtin preserves integer samples and therefore requires
  identical grids. All four tensors must exist, float bridge dtypes must
  agree, and rank-four shapes and signatures must match across each Q/DQ
  boundary. Both bridge tensors are private, uniquely produced, exclusively
  consumed, and topologically ordered. The output cannot also be a graph
  input. All topology, metadata, and cloned-quantization planning completes
  before indexed Pool edge mutation and differential wrapper removal. This
  intentionally fixes former rewrites that accepted near-equal grids, absent
  float metadata, or a float bridge exposed as a public input. Missing
  required operator families retain historical pruning without allocating an
  index, and both production call sites pass the Session `LayoutState`.
- Quantized Logistic cleanup accepts only an exact linear
  `DEQUANTIZE -> LOGISTIC -> QUANTIZE` chain. Input and output tensors must use
  the same INT8 or UINT8 dtype. The input grid requires a positive finite
  scale and in-range zero point; the output grid is exactly scale `1/256` with
  zero point `-128` for INT8 or `0` for UINT8. Tolerant scale equality is not
  sufficient for the builtin's canonical output contract. All four tensors
  must exist, float dtypes must agree, and elementwise shape/signature metadata
  must be identical across the complete chain without imposing a fixed rank.
  Both float bridges are private, uniquely produced, exclusively consumed,
  and topologically ordered; the quantized output cannot also be an input.
  Every guard completes before indexed Logistic edge mutation, version
  selection, and differential wrapper removal. This intentionally fixes
  former rewrites that accepted a near-canonical output scale, missing or
  invalid input quantization, absent float metadata, or a public-input bridge.
  Missing required families retain historical pruning without allocating an
  index, and both production call sites pass the Session `LayoutState`.
- Quantized Softmax cleanup accepts only an exact linear
  `DEQUANTIZE -> SOFTMAX -> QUANTIZE` chain. Its grid requirements match the
  canonical quantized Logistic contract: the input uses a positive finite
  per-tensor INT8/UINT8 grid, while output scale is exactly `1/256` with zero
  point `-128` or `0`. The existing beta tolerance of `1e-6` is preserved, but
  beta must be finite and parseable. Rank must be positive, and an explicit or
  default axis must normalize to the final dimension because the serialized
  TFLite builtin has no independent axis field. All four tensors must exist,
  float dtypes must agree, and shape/signature metadata must be identical
  across the elementwise chain. Both float bridges are private, uniquely
  produced, exclusively consumed, and topologically ordered; the quantized
  output cannot also be an input. Every guard completes before indexed
  Softmax edge mutation, version selection, and differential wrapper removal.
  This intentionally fixes former rewrites that accepted near-canonical
  output scales, missing/invalid input grids, absent float metadata, public-
  input bridges, malformed options, or non-last axes. Missing required
  families retain historical pruning without index allocation, and both
  production call sites pass the Session `LayoutState`.
- Expanded HardSigmoid QDQ cleanup matches the exact linear
  `DEQUANTIZE -> MUL(alpha) -> ADD(beta) -> MAXIMUM(low) -> MINIMUM(high) ->
  QUANTIZE` grammar with either scalar-input position at each binary operator.
  Input and output use exactly the same finite positive per-tensor INT8/UINT8
  grid with an in-range zero point. All seven data tensor records must exist;
  the five float tensors have one dtype and every data tensor has identical
  elementwise shape/signature metadata. Every bridge is private, uniquely
  produced, exclusively consumed, and topologically ordered; the quantized
  output cannot also be an input. Each scalar must be finite, singleton,
  producer-free, and representable within the preserved quarter-scale/`1e-3`
  tolerance.
- The four constant retargets are immutable plans containing the quantized
  value, cloned quantization, ownership choice, metadata, and reserved clone
  name. Private exclusive constants retain in-place behavior. Shared or public
  constants receive deterministic `_q` clones, so a public float scalar no
  longer silently changes dtype/value. Only after all four plans and four
  intermediate quantization clones succeed are constant and data edges
  rewritten and DQ/Q wrappers removed. Fault injection proves that a clone
  failure on the second scalar leaves complete ModelIR unchanged, whereas the
  former helper changed the first scalar data/dtype before raising. Missing
  required families retain historical pruning without index allocation; both
  production call sites pass LayoutState, which is synchronized after clones
  and pruning.
- Quantized TransposeConv cleanup accepts only the exact linear
  `DEQUANTIZE -> TRANSPOSE_CONV -> QUANTIZE` grammar with TFLite input roles
  `[output_shape, filter, data]`. Input and output activations independently
  require finite positive per-tensor INT8 grids and in-range zero points.
  Their quantized/float shape and signature metadata must match at each
  boundary, the bridges must share a floating dtype, and both bridges are
  private, uniquely produced, exclusively consumed, and topologically ordered.
  The quantized output cannot also be a graph input.
- Filter conversion is a pre-mutation plan. A producer-free rank-four INT8
  filter with matching buffer metadata and a valid grid remains unchanged. A
  finite FLOAT16/FLOAT32/FLOAT64 filter is quantized in place only when it is
  private and exclusive; shared or public filters receive a deterministic
  `_q` clone. The filter data/grid/name and output quantization clone all
  complete before indexed input/output mutation. Fault injection proves that
  output-grid clone failure leaves the complete ModelIR unchanged, whereas the
  former helper had already converted a private float filter before raising.
  Missing families retain historical pruning without index allocation; all
  three production call sites pass the Session LayoutState.
- Decomposed InstanceNormalization repair requires the exact marked chain from
  the first Mean through bias Add, including correct Sub and reciprocal-Div
  operand roles, keep-dim reductions, graph order, unique producers, exclusive
  internal consumers, a finite epsilon, and a producer-free scalar one. The
  input logical layout chooses channel axis one or the final axis for ranks
  three through five; an optional post-Transpose must be a complete
  permutation before the bias broadcast axis is derived.
- Both Mean axes, nine intermediate shape/signature records, and scale/bias
  data plus metadata are planned before mutation. Changing a constant requires
  integer axes or a channel-count-sized buffer, no producer/public boundary,
  and exactly the expected Mean/Mul/Add consumers. This preserves the shared
  two-Mean axes tensor while rejecting external sharing. A malformed final
  bias shape is now a complete no-op instead of raising after earlier axes,
  shapes, and scale data were already changed. The final production call passes
  the Session LayoutState; graphs without the marked first Mean allocate no
  index.
- NCHW Concat/global-pool/Conv repair requires the exact ordered
  `CONCATENATION -> MEAN -> RESHAPE -> CONV_2D` chain. Each internal tensor is
  uniquely produced, exclusively consumed by the next operator, and private.
  The keep-dim Mean reduces rank-four axes two and three; negative axes are
  normalized. Fully positive NCHW Concat inputs share batch/spatial dimensions,
  and their axis-one channel sum must equal the producer-free OHWI Conv filter
  input channel.
- Concat and Reshape options, Concat/Mean/Reshape metadata, and the four-value
  integer Reshape buffer are a complete pre-mutation plan. The shape constant
  is producer-free, private, and exclusively consumed by that Reshape. This
  prevents a late buffer-read exception from leaving the Concat axis and three
  tensor records changed, and rejects the former non-global, fan-out/public,
  duplicate-producer, runtime-filter, and malformed/shared shape-buffer cases.
  The sole production call passes the Session LayoutState; incomplete operator
  families allocate no index.
- NCHW Concat/Transpose/(Transpose)Conv repair traces optional shape-preserving
  RELU/RELU6/QUANTIZE/DEQUANTIZE/CAST before the exact `[0,2,3,1]` Transpose
  and optional PAD/CAST/SUB afterward. Every internal output is uniquely
  produced, exclusively consumed by the next topologically ordered operator,
  and private. The permutation and OHWI filter are producer-free constants,
  and the filter buffer exactly matches its rank-four metadata.
- Fully positive NCHW Concat inputs share batch/spatial dimensions, and their
  axis-one channel sum equals the filter input channel. Concat options and all
  Concat/pre-passthrough/Transpose shape records are planned together. Direct
  Conv without a post-prefix also plans its output shape; prefixed Conv and
  TransposeConv intentionally preserve output metadata. This rejects the
  former public/fan-out/duplicate adapter and runtime-filter cases without
  broadening the four existing positive families. The production call passes
  the Session LayoutState; missing Concat or Conv families allocate no index.
- Mixed singleton Concat repair accepts an exact axis-three NHWC Concat only
  when its output channel equals the input count and every same-dtype input is
  either `[N,H,W,1]` or its singleton-channel NCHW projection. Input/output
  shape signatures must express the same contract. One dynamic dimension is
  retained as the sole Reshape `-1`; multiple dynamic dimensions cannot be
  represented by this local adapter and therefore remain untouched.
- A runtime input must be public or uniquely produced before the Concat; a
  producer-free constant is also valid. Duplicate, later, and unresolved
  producers are rejected. Names are reserved across all existing tensors,
  operator edges, and boundaries. Repeated source inputs share one adapter.
  Every shape tensor, adapter tensor, operator, and quantization clone is
  prepared before indexed insertion and lineage-aware rewiring. This prevents
  a late clone exception from leaving the first adapter behind. The production
  call passes the Session LayoutState; graphs without Concat allocate no index.
- Window-partition canonicalization requires exact NHWC input, six-dimensional
  partition Reshape, `[0,1,3,2,4,5]` Transpose, and three-dimensional window
  output Reshape equations. All internal edges are private, uniquely produced,
  exclusively consumed, and ordered. The input is public, constant, or
  uniquely produced earlier. Shape/permutation vectors are producer-free,
  non-input INT32/INT64 constants with exact vector metadata.
- All four data tensors share dtype and either no quantization or one exact
  per-tensor grid. The existing Transpose object becomes SPACE_TO_DEPTH so its
  version, axis semantics, and provenance survive. Static metadata remains
  exact. A consistent dynamic batch/spatial/channel signature is propagated;
  when the retained Reshape needs one `-1`, its private vector and both shape
  options are changed together and marked dynamic. Two inferred dimensions,
  shared/public shape mutation, or any incomplete contract is a complete
  no-op. Both production calls pass the Session LayoutState; incomplete
  operator families prune historically unused tensors without allocating an
  index.
- Recurrent orphan-step alias repair has one Torch-free semantic owner in
  `passes/recurrent_alias.py`. Candidate discovery occurs before index
  construction, so graphs without the exact step-name grammar allocate no
  index. A supplied matching index is reused; otherwise exactly one index is
  built. Producer rejection, shape-tensor consumer order, Reshape arity,
  public input/output, consumer rewrites, and non-public orphan tensor removal
  retain the direct implementation's behavior. The direct and PyTorch modules
  are compatibility wrappers only and do not carry parallel match/rewrite
  rules.
- Unbound nonconstant-input discovery and layout repair have one Torch/
  TensorFlow-free owner in `passes/unbound_input_layout.py`. Standalone issue
  reporting retains its lightweight producer-name scan. Repair first snapshots
  issue consumers by object identity; it constructs one `ModelIRGraphIndex`
  only when an issue exists, resolves current positions after each insertion,
  and skips later issues once an earlier bridge produces the same tensor.
  DEQUANTIZE exact/fallback source policy, nearest source ordering, SPLIT data
  slot, MUL all-consumers guard, dtype/shape checks, quantization metadata,
  unique perm naming, and insertion-before-consumer order remain unchanged.
  The lowerer wrapper reconciles shapes with the maintained index and preserves
  the existing stats key for both primary and fallback callers.
- Quantized RELU/RELU6 layout-bridge cleanup has one Torch/TensorFlow-free
  owner in `passes/quantized_activation.py`. It skips index allocation when no
  Transpose exists, otherwise uses one current index for exact chain traversal,
  DQ input and Q output rewrites, and batch removal of the inverse Transposes.
  The restart loop remains intentional: removing a later bridge may make an
  earlier graph-order candidate linear, while current indexed candidates avoid
  every compatibility-map rebuild. Public intermediate/source guards, exact
  inverse permutations, per-tensor-only quantization, source-shape propagation,
  destination dtype/quantization cloning, pruning, and stats are unchanged.
- Expanded HardSigmoid QDQ layout-bridge cleanup now shares that owner and
  recognizes both RELU_0_TO_1 and MAXIMUM/MINIMUM clamp forms. One current
  index supplies all linear-consumer and shared-constant decisions, indexed
  setters maintain cloned-constant and DQ/Q edges, and both Transposes are
  removed by one differential compaction. All required constants are
  validated before any mutation, making rejection transactional. Private
  rank-matched constants retain in-place remapping; shared constants retain
  clone-and-rewire behavior. Public guards now include the clamp form's
  MAXIMUM intermediate as well as every former boundary, while inverse-perm,
  per-tensor quantization, shape/signature, destination metadata, lineage,
  pruning, and stats contracts remain intact.
- Expanded MUL/ADD/PRELU QDQ bridge cleanup now has the same single owner.
  MUL/ADD input ordering and PRELU data-slot semantics are unchanged; one
  index supplies all topology and shared-constant decisions, both DQ/Q edge
  updates, and removal of the two wrapper Transposes. The common constant
  planner is mutation-free and the common applier owns private updates and
  shared clone rewires, but PReLU still requires all three buffers to be NumPy
  arrays while HardSigmoid accepts any non-`None` buffer. This prevents a
  convenience helper from broadening rule eligibility.
- Quantized logistic-gated MUL recovery has a separate
  `passes/quantized_gate.py` owner because its shared input, dual DQ/Q
  branches, and multi-post aliases are not a linear activation contract. One
  index supplies producer and consumer topology, both DQ rewires, canonical
  output selection, indexed alias replacement, and one compaction for the
  pre-Transpose plus every post-Transpose. The first graph-order post remains
  canonical and receives the permuted MUL-Q metadata. All internal data/gate
  tensors are now protected when publicly observable; fixed permutations,
  per-tensor quantization, pruning, lineage, and stats remain unchanged. A
  bounded `_match_logistic_gate_branch` helper isolates backward gate matching
  while preserving incomplete-chain fallback and duplicate-branch rejection.
- General transpose/binary bridge folding now has one TensorFlow-free owner in
  `passes/binary_bridge_layout.py`. Symmetric candidates are processed before
  asymmetric candidates to preserve the former phase priority. A supplied
  index is reused, candidates are fixed in graph order, and at most 32
  accepted rewrites are applied. Typed immutable permutations, unique source
  provenance, exact private pre-adapter use, producer-before-consumer order,
  per-tensor quantization, dtype, static broadcast, dynamic signature, public
  boundaries, output fan-out, and every insertion/removal index are resolved
  before mutation and re-resolved immediately before apply. The mixed-fan-out
  form reuses an existing pre-permutation constant instead of mutating the
  possibly shared post-permutation buffer. The legacy-only form creates and
  inserts its adapter only after the complete plan passes. The asymmetric form
  retains operand order for SUB and DIV and rejects a plain source produced
  after the Transpose it would reuse. The disabled Pattern C implementation is
  intentionally absent rather than carried as unreachable production code.
- The later safe recovery sequence is not five independent owners. One
  `run_safe_binary_bridge_recovery` call retains the exact legacy-only,
  single-post, mixed fan-out, asymmetric fan-out, and full-post order on one
  differential index. The 32-rewrite bound applies per phase. The first two
  phases reuse the strict symmetric plan and add the historical preserved-
  boundary marker only at this late recovery point. Mixed and full-post modes
  share one multi-post plan and canonical-first-post alias contract.
  Asymmetric fan-out requires an already-earlier inverse Transpose of the plain
  operand and preserves SUB/DIV order. Retained adapters reference the typed
  pre-permutation tensor rather than changing a shared inverse constant. All
  modes re-resolve their complete tensor/operator contracts before mutation,
  and pruning/LayoutState synchronization run once after the sequence.
- Elementwise/Concat layout recovery is grouped by connected Concat closure,
  not rewritten one node at a time. This prevents a shared elementwise
  producer from being converted under one Concat while another dependent
  Concat remains in the old layout.
- Existing inverse adapters are accepted as NHWC boundary aliases only after
  their typed permutation, shape/signature, dtype, quantization, graph order,
  and non-public output are proven. Explicit pre-adapters are removed only
  when every remaining consumer is handled; otherwise the adapter is retained
  for its untouched fan-out.
- Rank-four binary constants are transposed only when the entire consumer-slot
  set belongs to the accepted closure. A shared or mutable constant rejects
  the candidate, preventing a layout rewrite from changing unrelated binary
  behavior. SUB and DIV keep their original operand positions.
- Closure outputs used by supported legacy unary/binary consumers receive one
  deterministic local NHWC-to-NCHW adapter. Unsupported fan-out or a public
  closure intermediate rejects the complete group without mutation.
- The explicit NHWC layout annotation is an intentional internal contract
  improvement. Operator/tensor topology, metadata other than layout,
  constants, and emitted artifacts remain identical to the preceding
  checkpoint, while TensorIR and Session LayoutState no longer disagree about
  the converted physical view.
- StridedSlice fan-in is one whole-group transaction rooted at the shared
  pre-Transpose. The pass does not rewrite an individual Slice unless every
  pre-adapter consumer is a supported Slice, every Slice output joins the same
  Concat exactly once, and every Concat boundary has been classified.
- Begin, end, and stride vectors must be distinct immutable rank-four INT32 or
  INT64 constants whose TensorIR dtype, NumPy dtype, shape/signature, producer,
  public ownership, and exact consumer slots agree. Zero strides or conflicting
  parameter roles reject the whole candidate before any constant changes.
- Multiple private post adapters collapse to the first output while preserving
  the legacy `replace_operator_input_at`, output-rename, alias-replacement, and
  pruning lineage-event order. This keeps active-path correspondence behavior
  unchanged rather than treating report stability as incidental.
- A required NCHW compatibility boundary reuses the typed pre-permutation. The
  raw helper changed the first post-permutation buffer to an INT32 opposite
  permutation even when that tensor was INT64 or shared. The indexed owner
  leaves every post-permutation and unrelated consumer untouched.
- A retained inverse adapter must already precede every legacy consumer.
  Stale order is rejected transactionally instead of emitting a consumer before
  its new producer. Public Concat outputs and later public post paths remain
  supported through the same local boundary.
- Shared parsers preserve the exact old generated syntax when broadening would
  change rule eligibility. Parser ownership tests prevent duplicate exporter
  implementations and unused compatibility imports.
- No real-model conversion gate is required for these mechanical checkpoints
  under the current instruction to prioritize implementation and minimize
  conversion tests. This does not prove broad corpus regression safety.

## Tests executed

The resumed downstream-binary, Resize-evidence, aligned-BatchNorm, LRN,
static-shape-cache, and NHWC-bridge checkpoints passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pytorch_fast_precanonicalize_policy.py \
  tests/test_flatbuffer_direct_architecture.py

124 passed
```

Seven pre/post-extraction characterization cases also preserve the exact
orchestrator output for matching BN, direct return, channel-first Resize,
channel mismatch, channel-last Resize, mixed operands, and an already-CF shape.
Five additional pre/post-extraction cases preserve direct BN, reshaped BN,
no-BN fallback, NHWC-input no-op, and already-CF Resize behavior.
Seven aligned-BatchNorm cases preserve direct rewrite, non-BN no-op, channel
mismatch no-op, NHWC-input no-op, reshaped rewrite, already-CF behavior, and
reshaped non-BN no-op.
The LRN checkpoint additionally passed a four-test selection covering CF/NHWC
and static-shape state, Pool/LRN interaction, architecture ownership, and the
existing generated-source integration case.
The cache checkpoint passed four focused cases covering aligned binary,
Resize/Pool, literal recording, parse-failure no-op, and architecture ownership.
The bridge checkpoint adds positive state-set/cache assertions and a no-op
state-preservation case to the existing whole-chain normalization test.
The request-boundary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_architecture.py

126 passed
```

Its AST gate proves zero `kwargs.get` calls and exactly one raw `kwargs` read,
as the argument to `ConversionRequest.from_kwargs`.
The requested-exporter checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_artifact_preparation.py \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_architecture.py

145 passed
```

Dedicated resolver tests use a mapping that raises on every `get` to prove
unrequested settings are untouched, then verify requested and calibration-only
values and timeout coercion.
The same 145-test selection passed after adding requested-only quant type/dtype
resolution. Focused assertions cover explicit values, the three legacy
defaults, immutable mapping behavior, and absence of direct `request.get`
calls for those keys.
The Session consumer-count checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_optimization \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_fanout_optimization \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_quantize_transpose_preserves_dynamic_batch_signature

32 passed
```

The core spy fixture uses one input in both Add and Identity and verifies the
context receives counts `{"x": 2, "y": 1}` from the Session index.
The lowering-time layout checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_pad_edge_lowering.py \
  tests/test_flatbuffer_direct_resize_integer_linear.py \
  tests/test_flatbuffer_direct_architecture.py::test_op_builders_mutate_layout_only_through_lowering_context

33 passed
```

The rank-three Resize regression hook inspects `LayoutState` before the first
post-lowering pass and verifies that the NWC/NHWC adapter tensors have no
ModelIR mismatch. The focused edge-Pad and integer-linear Resize tests also
serialize and execute their TFLite artifacts sequentially. The architecture
gate rejects future direct logical/physical layout assignments anywhere below
`op_builders`.
The synthetic-consumer checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_core.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_elides_inverse_transpose_chain_at_generation \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_no_dead_operator_outputs_after_prune \
  tests/test_flatbuffer_direct_architecture.py::test_op_builders_mutate_operator_list_only_through_lowering_context

35 passed
```

The two direct context cases prove both sides of the contract: exclusive
inverse Transposes remove their operator and differential indexes, while a
synthetic bridge already consumed by an Identity retains its producer. The
architecture gate rejects direct operator-list writes and mutating method calls
from op builders.
The adjacent pass-state reuse checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_mean_layout.py \
  tests/test_flatbuffer_direct_layernorm_layout.py \
  tests/test_flatbuffer_direct_terminal_mean_layout.py \
  tests/test_flatbuffer_direct_se_layout.py \
  tests/test_flatbuffer_direct_core.py \
  tests/test_flatbuffer_direct_pass_efficiency.py \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_mean_attention_cluster_reuses_one_pass_state_scope

65 passed
```

The focused reuse case observes one `ModelIRGraphIndex.refresh()` across two
candidate Mean runners and a diagnostic build sequence of `[true, false]`.
The all-preflight-miss case observes zero index refreshes and `[false, false]`.
The architecture gate fixes all seven runner calls, their order, their shared
scope keyword, and the six production cluster invocations.
The adjacent gate-pass reuse checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_elementwise_gate_layout.py \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_flatbuffer_direct_dual_postconv_gate_layout.py \
  tests/test_flatbuffer_direct_3d_gate_layout.py \
  tests/test_flatbuffer_direct_conv3d_gate_layout.py \
  tests/test_flatbuffer_direct_cost_volume_scatter_layout.py \
  tests/test_flatbuffer_direct_add_concat_suffix_layout.py \
  tests/test_flatbuffer_direct_dual_mul_concat_layout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_mixed_mean_reducemax_concat_mirrorpad_nhwc_chain \
  tests/test_flatbuffer_direct_pass_efficiency.py \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_gate_cluster_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

116 passed
```

The synthetic shared-scope fixture makes every runner's model-only preflight
match but contains no deep rewrite candidate. It records one
`ModelIRGraphIndex.refresh()` across all eight calls and 15 diagnostic events:
the first reports `state_built: true`, and every later event reports `false`.
The architecture checks fix the eight-runner order, five production helper
invocations, and the single invocation that omits mixed attention. They also
bring the global direct-runner count characterization up to date with both
bounded helper extractions.
A focused late-pair checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_ndhwc_cost_volume_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_ndhwc_cost_volume_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_3d_gate_layout.py \
  tests/test_flatbuffer_direct_conv3d_gate_layout.py \
  tests/test_flatbuffer_direct_cost_volume_scatter_layout.py

50 passed
```

The runtime characterization observes one graph-index refresh and diagnostic
build flags `[true, true, false]`: both events in the first two-spec runner
belong to the one state-building group, and the second runner reuses that
state. The architecture test fixes both raw boundaries and proves that mixed
attention receives no shared scope.
A focused late-Concat cluster checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_concat_layout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_concat_layout_cluster_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_axis3_const_concat_layout.py \
  tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py \
  tests/test_flatbuffer_direct_layernorm_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py

74 passed
```

The synthetic runtime fixture records one graph-index refresh and build flags
`[true, false, false, false, false]`. The architecture test fixes the four-
runner order and both raw boundaries. The core layout-handoff monkeypatch now
accepts and forwards the runner's optional scope without changing its original
pre-pass layout assertions.
A focused shuffle/unary cluster checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_shuffle_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_nhwc_channel_shuffle.py \
  tests/test_flatbuffer_direct_nchw_channel_shuffle.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_transpose_unary.py \
  tests/test_flatbuffer_direct_transpose_unary_fanout.py \
  tests/test_flatbuffer_direct_transpose_unary_binary_fanout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_shufflenet_transpose_shuffle_chain_optimized

29 passed
```

Both synthetic fixtures observe exactly one graph-index refresh. The seven-
runner fixture records one `state_built: true` followed by six `false` events;
the three-runner fixture records one `true` followed by two `false` events.
The architecture gate fixes the two helper orders, five and four invocations,
and the single extended channel-shuffle invocation.
A focused boundary-input pair checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_boundary_batchmatmul_unary_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_boundary_batchmatmul_unary_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_boundary_input_chains.py \
  tests/test_flatbuffer_direct_input_passthrough_layout.py

17 passed
```

The runtime fixture records one graph-index refresh and build flags
`[true, false, false, false]`; the latter three events belong to the reused
three-spec input-unary group. The architecture gate fixes the two-runner order
and all four helper invocations.
A focused channel-slice/Pad-Mul checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_channel_slice_pad_mul_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_channel_slice_pad_mul_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_tflite_builder_direct.py \
  -k 'channel_slice or transpose_pad_mul_posttranspose or channel_slice_pad_mul_pair or ordered_model_ir_runner or lowerer_channel_slice'

8 passed, 754 deselected
```

The runtime fixture records one graph-index refresh and build flags
`[true, true, true, false]`: the first three events belong to the one state-
building channel-slice group, and Pad-Mul reuses that state. The architecture
gate fixes the two-group order and all three helper invocations.
A focused singleton/Reshape checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_singleton_channel_transpose.py \
  tests/test_flatbuffer_direct_singleton_reshape.py \
  tests/test_flatbuffer_direct_singleton_maxpool.py \
  tests/test_flatbuffer_direct_flatten_concat_reshape.py \
  tests/test_flatbuffer_direct_consecutive_reshape.py \
  tests/test_flatbuffer_direct_singleton_spatial_reshape.py \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_osnet_gate_layout.py \
  <two new efficiency tests and two architecture checks>

41 passed
```

The long synthetic fixture makes all ten runner preflights match and records
13 diagnostic events with one graph-index refresh and build flags
`[true, false, ...]`. The short fixture records one refresh and flags
`[true, false, false]`. Architecture checks fix both long variants, all three
short-helper target/layout combinations, the shared scope keyword, and the
147-call global runner characterization.
A focused QKV attention checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  <seven focused QKV rewrite tests from test_tflite_builder_direct.py> \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_qkv_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_qkv_attention_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

10 passed
```

The preflight-only fixture records one graph-index refresh and build flags
`[true, true, true, true, false, false]`: the four prefix events belong to the
state-building group, and both bridge events reuse it. The seven existing
functional cases cover Gather/Reshape/Transpose hoisting, Gather-to-Slice,
Slice-to-Split, Split/Reshape collapse, shared pre-Transpose, weighted-sum
bridging, and the KV pipeline.
A focused duplicate/PReLU checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_quantized_prelu.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_duplicate_quantized_prelu_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_duplicate_quantized_prelu_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

28 passed
```

The QDQ preflight-only fixture disables duplicate-Transpose cleanup exactly as
production does, then records one graph-index refresh and build flags
`[true, false, false, false, false]` across the reshape-duplicate group and
four PReLU specs. Architecture checks fix the helper's flag forwarding, exact
runner order, two invocations, and the 143-call global characterization.
A focused constant-fold/Cast checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_constant_fold.py \
  tests/test_flatbuffer_direct_cast_cleanup.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_constant_fold_cast_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_constant_fold_cast_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

16 passed
```

The preflight-only widening-Cast fixture records one graph-index refresh and
build flags `[true, true, true, false, false]`: the three constant-fold events
belong to the state-building group, and both Cast-cleanup events reuse it.
Architecture checks fix runner order, both helper invocations, and the 141-call
global characterization.
A focused SE-FC/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_se_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_channel_fanout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_se_fc_gather_channel_fanout_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_se_fc_gather_fanout_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

14 passed
```

The preflight-only fixture records one graph-index refresh and build flags
`[true, false]`. Architecture checks fix both target/LayoutState combinations,
runner order, and shared scope keywords. The global characterization was 139
calls at that checkpoint, 137 after the later post-QDQ consolidation, and 135
after the subsequent NCHW channel-shuffle consolidation, then 134 after the
conditional QKV-bridge consolidation.
A focused terminal-boundary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_boundary_input_layout.py \
  tests/test_flatbuffer_direct_dual_mul_concat_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pad_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_channel_fanout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_boundary_layout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_boundary_layout_cluster_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

38 passed
```

The preflight-only fixture records one graph-index refresh across seven events;
only the first reports `state_built: true`. Architecture checks fix all five
runner calls, their shared scope, the preceding raw InstanceNorm rewrite, and
the following conditional stage.
A focused late-Dequantize/unary checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_dequant_concat_quantize_layout.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_dequant_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_dequant_unary_fanout_cluster_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

41 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks fix all three
runner calls, their shared scope, the preceding raw QDQ bridge, and the
following independently indexed Swish dispatcher.
A focused terminal singleton-MaxPool/Reshape checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_singleton_maxpool.py \
  tests/test_flatbuffer_direct_consecutive_reshape.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_singleton_maxpool_reshape_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_singleton_maxpool_reshape_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

8 passed
```

The preflight-only fixture records one graph-index refresh across three events;
the two singleton-MaxPool events report `state_built: true` because they share
the first registered group, and the Reshape event reports `false`.
Architecture checks fix both runner calls, their shared scope, and the exact
conditional legacy-rewrite boundaries.
A focused terminal clamp/unary/ReLU checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_graph_cleanup.py \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_terminal_clamp_unary_relu_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_terminal_clamp_unary_relu_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

27 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. The two real rewrite tests also
assert that the reused graph index removes `MAXIMUM`/`MINIMUM` type entries and
adds the correct `RELU_0_TO_1` or `RELU` entry without refreshing.
A focused late Mean/SPP/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_mean_layout.py \
  tests/test_flatbuffer_direct_spp_layout.py \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_mean_spp_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_mean_spp_gather_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

51 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks fix the runner
order, shared scope keywords, conditional generic-transpose predecessor, and
constant-fold/Cast helper successor.
A focused late SPP/Concat-unary-Conv checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_spp_layout.py \
  tests/test_flatbuffer_direct_concat_unary_conv_layout.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_spp_concat_unary_conv_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_spp_concat_unary_conv_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

73 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. Architecture
checks fix both runner calls, shared scope keywords, and both raw rewrite
boundaries.
A focused absolute-final normalization-Pad/attention checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_instancenorm_mirror_pad_prepost_nhwc_chain_optimized \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_flatten_globalnorm_pad_prepost_nhwc_chain_optimized \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_mixed_mean_reducemax_concat_mirrorpad_nhwc_chain \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_absolute_final_normalization_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_absolute_final_normalization_attention_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

6 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. The focused
selection also executes real InstanceNorm, flattened normalization-Pad, and
mixed-attention rewrites. Architecture checks preserve both include flags and
the exact raw boundaries.
A focused post-QDQ unary-fan-out checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_post_qdq_layout_unary_fanout_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_post_qdq_unary_fanout_cluster_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

9 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the first reports `state_built: true`. Architecture checks preserve four
default helper invocations, require exactly one alternate-mode invocation,
fix both raw boundaries, and characterize 137 registered runner calls.
A focused late NCHW channel-shuffle/Gather checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_nhwc_channel_shuffle.py \
  tests/test_flatbuffer_direct_nchw_channel_shuffle.py \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_shuffle_gather_cluster_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_nchw_shuffle_gather_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_shuffle_and_unary_clusters_reuse_pass_state_scopes \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_nchw_shuffle_gather_pair_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

15 passed
```

The preflight-only fixture records one graph-index refresh across two events;
the first reports `state_built: true` and the second `false`. Real NHWC and
NCHW rewrite tests assert that `RESHAPE` type entries disappear and the
surviving `GATHER` entry has its correct shifted index without refreshing.
Architecture checks preserve five default helper invocations, require one
NCHW-only invocation, fix both raw boundaries, and characterize 135 runner
calls.
A focused conditional layout/QKV-bridge checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_layout_transpose.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_attention_qkv_shared_pretranspose_slice_nchw \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_attention_qkv_weighted_sum_bridge_to_nhwc \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_qkv_attention_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_layout_qkv_bridge_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_qkv_attention_pair_reuses_one_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_layout_qkv_bridge_pair_stays_between_raw_rewrites \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

11 passed
```

The preflight-only fixture records one graph-index refresh across three events;
only the generic-transpose event reports `state_built: true`. Architecture
checks preserve two default helper invocations, require one bridge-only
invocation with the runtime layout flag, fix both raw boundaries, and
characterize 134 runner calls.
A focused very-late Gather/constant/normalization checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_transpose_gather_axis.py \
  tests/test_flatbuffer_direct_constant_fold.py \
  tests/test_flatbuffer_direct_cast_cleanup.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_flatten_globalnorm_pad_prepost_nhwc_chain_optimized \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_constant_fold_cast_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_very_late_gather_constant_normalization_cluster_reuses_one_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_constant_fold_cast_pair_reuses_pass_state_scope \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_very_late_gather_constant_normalization_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

21 passed
```

The preflight-only fixture records one graph-index refresh across seven events;
only the Gather-axis event reports `state_built: true`. Architecture checks
require both constant-fold/Cast invocations to receive an external scope,
preserve the normalization flags, and fix both raw boundaries.
A focused terminal hard-activation/layout scope checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_hard_activation_layout_pair_reuses_one_pass_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_hard_activation_layout_pair_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

3 passed
```

The preflight-only fixture records one graph-index refresh across three events;
both HardSigmoid events report `state_built: true` for their shared group and
the following generic-Transpose event reports `false`. Architecture checks
preserve all four late hard-activation flags, the conditional layout switch,
and both raw rewrite boundaries. The related real hard-activation and generic-
Transpose selection passed separately with `7 passed`.
A focused late layout/Mean/SPP/Gather/constant/Cast scope checkpoint passed:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q \
  tests/test_flatbuffer_direct_pass_efficiency.py::test_late_layout_mean_spp_gather_constant_cast_cluster_reuses_one_state \
  tests/test_flatbuffer_direct_architecture.py::test_lowerer_late_layout_mean_spp_gather_constant_cast_cluster_reuses_scope \
  tests/test_flatbuffer_direct_architecture.py::test_ordered_model_ir_runner_calls_record_session_diagnostics

3 passed
```

The preflight-only fixture records one graph-index refresh across nine events;
only the conditional generic-Transpose group reports `state_built: true`.
Architecture checks preserve the full runner/helper order, the runtime layout
flag, both external constant-fold/Cast scopes, and the raw boundaries. The six
related real pass modules passed separately with `65 passed`.
A broader single-process selection of
`test_flatbuffer_direct_core.py`, `test_flatbuffer_direct_pass_efficiency.py`,
and the complete `test_flatbuffer_direct_architecture.py` passed with
`186 passed` after adding the late combined-scope checks.

The typed artifact-plan checkpoint passed its policy selection:

```text
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -q tests/test_flatbuffer_direct_artifact_preparation.py

20 passed
```

Default, dynamic-range/integer-quantized, split, SavedModel, PyTorch-derived,
and report progress labels are characterized from `ArtifactPlan`. Rejecting
option mappings prove that an unrequested artifact reads no related option or
environment value. Sequential dynamic-range, strict-integer, and split-
manifest direct-backend smokes passed with `3 passed`. The combined artifact-
policy, core, pass-efficiency, and architecture selection passed with
`206 passed`.

The evaluation-artifact selection checkpoint passed its focused unit and
ownership selection with `4 passed`. Sequential dynamic-range, strict-integer,
and split-manifest direct-backend smokes again passed with `3 passed`. The
combined artifact-metadata, artifact-policy, core, pass-efficiency, and
architecture selection passed with `210 passed`.

The report/quantized-artifact finalization checkpoint passed its focused
single-owner structure test with `1 passed`. The sequential integration
quantization/evaluation/coverage, strict integer/int16, and split-manifest
direct-backend smokes passed with `3 passed`. The combined artifact-metadata,
artifact-policy, core, pass-efficiency, and architecture selection passed with
`211 passed`.

The terminal direct-boundary checkpoint passed the three focused evaluation-
selection, report/quantized-finalization, and fast-path control-flow contracts
with `3 passed`. TensorFlow-import-blocked direct and `-cotof`, followed by the
sequential integration quantization/evaluation/coverage, strict integer/int16,
and split-manifest smokes, passed with `5 passed`. The combined artifact-
metadata, artifact-policy, core, pass-efficiency, and architecture selection
passed with `212 passed`.

The ordered layout-recovery-prefix checkpoint passed its focused ordering,
runner-ownership, SPP, NDHWC Concat, and NHWC/NCHW channel-shuffle selection
with `68 passed`. A sequential quantization/evaluation/coverage integration
smoke passed with `1 passed`. The complete architecture selection passed with
`128 passed`; the combined artifact-metadata, artifact-policy, core, pass-
efficiency, and architecture selection passed with `213 passed`.

The attention/quantized-recovery-suffix checkpoint passed focused ordering,
scope-boundary, quantized PReLU, quantized Reshape, and trailing-output-
Transpose tests with `19 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`. The
combined artifact-metadata, artifact-policy, core, pass-efficiency, and
architecture selection passed with `214 passed`.

The layout/reshape/attention-recovery-prefix checkpoint passed its focused
owner, exact-order, successor-boundary, and runner-diagnostics selection with
`4 passed`. The complete architecture file passed with `130 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for the same combined selection total of `215 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The terminal slice/Concat layout-recovery checkpoint passed focused QKV,
channel-slice, exact-order, variant-boundary, runner-diagnostics, and layout-
Transpose ownership checks with `6 passed`. The complete architecture file
passed with `131 passed`; artifact-metadata, artifact-policy, core, and pass-
efficiency passed separately with `85 passed`, for a combined selection total
of `216 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The terminal affine/Concat/split recovery checkpoint passed focused exact-
order, raw-boundary, terminal slice/Concat, QKV-boundary, and runner-
diagnostics checks with `4 passed`. The complete architecture file passed with
`132 passed`; artifact-metadata, artifact-policy, core, and pass-efficiency
passed separately with `85 passed`, for a combined selection total of `217 passed`.
Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The attention/gate/QDQ recovery checkpoint passed focused exact-order, three-
boundary, outer-suffix, gate/unary-scope, and runner-diagnostics checks with
`5 passed`. The complete architecture file passed with `133 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for a combined selection total of `218 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The quantized-activation/binary-bridge recovery checkpoint passed its focused
exact-order and two-boundary selection with `4 passed`. The adapted post-QDQ
boundary selector and new owner passed together with `2 passed`. The complete
architecture file passed with `134 passed`; artifact-metadata, artifact-policy,
core, and pass-efficiency passed separately with `85 passed`, for a combined
selection total of `219 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The SiNet terminal layout-recovery checkpoint passed focused exact-order,
shape-boundary, terminal affine/slice, and runner-diagnostics checks with
`4 passed`. Its adapted terminal-clamp boundary and new owner passed together
with `2 passed`. The complete architecture file passed with `135 passed`; artifact-
metadata, artifact-policy, core, and pass-efficiency passed separately with
`85 passed`, for a combined selection total of `220 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The pre-Add/Mean attention-recovery checkpoint passed focused exact-order,
two-boundary, attention/QDQ composition, Mean-cluster scope, layout-prefix, and
runner-diagnostics checks with `5 passed`. The complete architecture file
passed with `136 passed`; artifact-metadata, artifact-policy, core, and pass-
efficiency passed separately with `85 passed`, for a combined selection total
of `221 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The SiNet pre-Add/Resize recovery checkpoint passed focused exact-order, four-
boundary, terminal-helper composition, terminal-clamp, and runner-diagnostics
checks with `4 passed`. Recursive helper expansion matches the preceding
lowerer AST exactly. The complete architecture file passed with `137 passed`;
artifact-metadata, artifact-policy, core, and pass-efficiency passed separately
with `85 passed`, for a combined selection total of `222 passed`. Its single
sequential quantization, evaluation, and coverage integration smoke passed
with `1 passed`.

The safe-binary and QLinear/Mean/Concat recovery checkpoint passed focused
exact-order, nested-helper composition, condition/progress/layout boundaries,
post-QDQ ownership, and runner-diagnostics checks with `6 passed`. Recursive
helper expansion matches the preceding lowerer AST exactly. The complete
architecture file passed with `139 passed`; artifact-metadata, artifact-policy,
core, and pass-efficiency passed separately with `85 passed`, for a combined
selection total of `224 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The indexed shape-convergence checkpoint passed its focused dynamic-Reshape,
shape-reconciliation, legacy-equivalence, single-index-build, and ownership
selection with `13 passed`. The complete architecture file passed with
`140 passed`; artifact-metadata, artifact-policy, core, and pass-efficiency
passed separately with `85 passed`, for a combined selection total of
`225 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The final indexed shape/activation convergence checkpoint passed its focused
legacy-equivalence, single-index-build, no-consumer-rescan, differential-index,
shape, and ownership selection with `18 passed`. The complete architecture
file passed with `141 passed`; artifact-metadata, artifact-policy, core, pass-
efficiency, and the two new convergence cases passed separately with
`87 passed`, for a combined selection total of `228 passed`. Existing Conv,
DepthwiseConv, Add, Sub, Mul, and Div activation-fusion coverage passed with
`20 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The indexed broadcast-constant repair checkpoint passed four focused
shared-constant, no-op, inverse-rotation, and ownership cases plus four existing
rank-three/rank-four repair characterizations (`8 passed`). The complete
architecture file passed with `142 passed`; artifact-metadata, artifact-policy,
core, pass-efficiency, indexed final-convergence, and binary-layout coverage
passed separately with `90 passed`, for a combined selection total of
`232 passed`. Its single sequential quantization, evaluation, and coverage
integration smoke passed with `1 passed`.

The indexed binary-layout convergence checkpoint passed six focused multi-
match, reversed-input, peer-producer, channelwise-constant, fan-out, exact-
legacy, one-build, and ownership checks. The complete related Conv/layout and
indexed-convergence selection passed with `15 passed`. The complete
architecture file passed with `143 passed`; artifact-metadata, artifact-policy,
core, pass-efficiency, indexed final-convergence, binary-layout, Conv-layout,
and indexed binary-convergence coverage passed separately with `105 passed`,
for a combined selection total of `248 passed`. Its single sequential
quantization, evaluation, and coverage integration smoke passed with
`1 passed`.

The indexed Conv-input adapter checkpoint passed its exact former-pair
equivalence, two-match stale-Transpose, one-index-build, no-map-rebuild,
fan-out, graph-output, filter-channel, and ownership coverage. The complete
related Conv/layout and indexed-convergence selection passed with `18 passed`.
The complete architecture file passed with `144 passed`; artifact-metadata,
artifact-policy, core, pass-efficiency, indexed final-convergence, binary-
layout, Conv-layout, indexed binary-convergence, and indexed Conv-input repair
coverage passed with `108 passed`. Its single sequential quantization,
evaluation, and coverage integration smoke passed with `1 passed`.

The indexed wrong-way Conv-Transpose sanitizer checkpoint passed its exact
former-implementation equivalence, two-match removal, multi-Conv consumer,
one-index-build, no-consumer-rescan, non-Conv fan-out, filter-channel, graph-
output, maintained-index, and ownership coverage with `3 passed`. The complete
related Conv/layout and indexed-convergence selection passed with `20 passed`.
The complete architecture file passed with `145 passed`; artifact-metadata,
artifact-policy, core, pass-efficiency, indexed final-convergence, binary-
layout, Conv-layout, indexed binary-convergence, indexed Conv-input repair, and
wrong-way sanitizer coverage passed with `110 passed`. Its single sequential
quantization, evaluation, and coverage integration smoke passed with
`1 passed`.

The shared indexed recurrent-alias checkpoint passed direct legacy-equivalence,
three-alias repair, first-Reshape ordering, public input/output, produced,
missing-shape, invalid-grammar, no-consumer, no-candidate/no-index, maintained-
index, direct/PyTorch wrapper equality, and ownership coverage with `8 passed`.
The complete recurrent, PyTorch normalization, recurrent-codegen-policy, and
new shared-owner selection passed with `18 passed`. The complete architecture
file passed with `146 passed`; the lightweight core/indexed selection passed
with `113 passed`. The real PyTorch exporter normalization regression passed
with `1 passed`, and the single sequential direct quantization, evaluation, and
coverage integration smoke passed with `1 passed`.

The indexed unbound-input layout checkpoint passed exact issue-report and
former-implementation equivalence across DEQUANTIZE, SHAPE, RESHAPE, SPLIT,
and two-consumer MUL-alias repair, plus nearest-source, quantization/signature,
mixed-fan-out guard, one-index-build, no-issue/no-index, maintained-index, and
ownership coverage plus nearest DEQUANTIZE fallback and strict exact-source
preference with `8 passed`. Its complete related QLinear/layout selection
passed with `9 passed`. The complete architecture file passed with
`147 passed`; the lightweight core/indexed selection passed with `117 passed`.
An actual GRU lowering/unbound-input check passed with `1 passed`, and the
single sequential direct quantization, evaluation, and coverage integration
smoke passed with `1 passed`.

The indexed quantized-activation checkpoint passed complete former-mutation
equivalence for two RELU/RELU6 chains, one-index-build, no-consumer-rescan,
maintained-index, public intermediate/source, fan-out, per-channel
quantization, non-inverse permutation, no-Transpose/no-index, ownership, and
real ONNX lowering coverage with `10 passed`. The complete architecture file
passed with `148 passed`; the lightweight core/indexed selection passed with
`125 passed`. Its single sequential direct quantization, evaluation, and
coverage integration smoke passed with `1 passed`.

The indexed expanded-HardSigmoid checkpoint passed exact valid-result
equivalence for RELU_0_TO_1 and MAXIMUM/MINIMUM forms in one graph, one-index
construction, no legacy consumer-map rebuild, maintained-index equivalence,
private and shared rank-four constant remapping, scalar constant preservation,
public add/clamp intermediates and source, fan-out, per-channel quantization,
non-inverse permutation, transactional missing-late-constant rejection, and
no-Transpose/no-index coverage. The complete architecture plus both indexed
quantized-activation files passed with `167 passed`; the lightweight core/
indexed selection passed with `112 passed`. Three real ONNX lowering checks,
TensorFlow-import-blocked direct and `-cotof`, and the sequential direct
quantization/evaluation/coverage integration smoke passed together with
`6 passed`.

The indexed expanded-PReLU checkpoint passed exact former-result equivalence
for two chains, one-index construction, no legacy consumer-map rebuild,
maintained-index equivalence, reversed MUL/ADD input order, private rank-four
MUL/alpha remapping, shared rank-four ADD cloning with quantization metadata,
scalar preservation, public intermediate/PRELU/source boundaries, fan-out,
per-channel quantization, non-inverse permutation, non-array and missing-alpha
rejection, and no-Transpose/no-index coverage. Complete architecture plus all
three indexed quantized-activation files passed with `178 passed`; the
lightweight core/indexed and related quantized-PReLU selection passed with
`128 passed`. Related real ONNX lowering, TensorFlow-import-blocked direct and
`-cotof`, and the sequential quantization/evaluation/coverage integration
smoke passed together with `7 passed`. An attempted exact public-output ONNX
fixture was not retained because the existing ordered trailing-output cleanup
correctly removes its post-Transpose before the specialized owner boundary.

The indexed quantized-logistic-gate checkpoint passed complete former-result
equivalence for simultaneous single- and multi-post chains, one-index
construction, no producer/consumer-map rebuild, maintained-index equivalence,
MUL input reversal, graph-order canonical output selection, alias consumer
consolidation, dtype/shape/signature/quantization propagation, public internal
data/gate/source/alias boundaries, pre/data/gate fan-out, non-Transpose post
users, per-channel quantization, wrong post permutation, and
no-Transpose/no-index coverage. Complete architecture plus all four indexed
quantized suites passed with `195 passed`; the lightweight core/indexed,
related quantized-PReLU, legacy logistic-gate characterization selection
passed with `145 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential direct integration selection passed with `7 passed`.

The single-owner wrong-way Conv-input safety-valve checkpoint preserves the
pre-extraction Swish-only ModelIR digest
`9b47e7f2e879895af600f66c6ac6929acc25580cfea8d5620fca9a6319ee4343` and
its two-removal Swish statistics. Focused exact legacy, multi-Conv, maintained-
index, public-output, mixed-fan-out, filter-channel, no-Transpose/no-index,
Swish-delegation, both existing Swish variants, and ownership coverage passed
with `7 passed`. The complete architecture, lightweight core/pass-efficiency,
focused sanitizer, and both Swish characterizations passed together with
`218 passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage integration smoke passed with `3 passed`.
No Tier corpus conversion was run.

The indexed primary Swish-QDQ branch checkpoint preserves complete former-
phase ModelIR and result equality across shared multi-branch, explicit concat-
closure, spatial guard, public intermediate, and data-fan-out fixtures. The
comprehensive existing fixture retains phase digest
`529b9889fafe9982ebb37ca63687b9329fa11a837562c154480c1856bbc05760`,
three rewritten branches, two removed pre-Transposes, and twenty rewritten
tensors. Focused shared quantized/float tails, one-index, maintained-index,
public/post-output/fan-out guards, small-spatial closure, no-Transpose/no-index,
both legacy Swish variants, and ownership coverage passed with `8 passed`.
Complete architecture, lightweight core/pass-efficiency, both indexed Swish/
Conv-safety suites, and the two legacy Swish characterizations passed together
with `224 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Swish-QDQ metadata checkpoint preserves complete prior-phase
ModelIR/result equality for reverse-ordered fixed-point, public-output,
Pool-channel-mismatch, and wrong-tail fixtures. The comprehensive fixture
retains post-metadata digest
`bab34e6351ec24bc564b9f95b4550bbfaca867f15906f9d77b92f7e8adf1d804`,
one rewritten Concat axis, and twenty-four rewritten tensors. Focused unary,
binary broadcast/signature, Pool/Resize, strict Concat tail, fixed-point,
family guards, empty-seed/no-index, shared primary-index, shared late-shape
owner, both legacy Swish variants, and ownership coverage passed with
`13 passed`. Complete architecture, lightweight core/pass-efficiency, both
indexed Swish/Conv-safety suites, and legacy Swish characterizations passed
together with `229 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage integration smoke passed with `3 passed`. No Tier corpus
conversion was run.

The indexed Swish-QDQ inverse-post checkpoint first proves the two former
lowerer loops have identical ASTs, then preserves complete former-loop ModelIR
and the three-removal result for chained aliases, multi-consumer fan-out,
public alias output, wrong permutation, and untracked input. Focused exact-
legacy, maintained-index, one-index, empty-seed/no-index, two-call ownership,
all existing indexed Swish cases, and both legacy Swish variants passed with
`16 passed`. Complete architecture, lightweight core/pass-efficiency, both
indexed Swish/Conv-safety suites, and legacy Swish characterizations passed
together with `232 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Swish-QDQ late-Concat checkpoint compiles the exact prior committed
late-loop AST and preserves its complete ModelIR, rewritten tensor set, one
axis rewrite, and two input-adapter removals on the mixed direct/DQ fixture.
Focused characterization covers maintained-index equivalence, mixed-input
rewiring, one shared late index, complete post-tail removal, retained direct
fan-out, public source and Concat outputs, missing tensors, mismatched shapes,
wrong tail permutation, transactional no-op behavior, and the missing-required-
type/no-index preflight. The complete indexed Swish and architecture selection
passed with `169 passed`. Architecture, core, pass-efficiency, indexed Swish,
wrong-way Conv safety, and both legacy Swish characterizations passed together
with `237 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage integration smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed Concat pre-Q/DQ checkpoint compiles the complete prior committed
function AST and preserves exact ModelIR and statistics for both one and two
simultaneous matches. Focused characterization covers two-match fixed-point
rewriting with one index construction, maintained-index equivalence, scale and
dtype mismatch, quantized fan-out, public quantized/dequantized boundaries,
shape mismatch, non-Dequantize provenance, rounding-preserving arithmetic
rejection, exact-grid acceptance, and no-Concat/no-index pruning. The focused
owner and legacy selection passed with `26 passed`. Architecture, core, pass-
efficiency, quantization cleanup, and the two established Concat-Q/DQ tests
passed together with `238 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage integration smoke
passed with `3 passed`. No Tier corpus conversion was run.

The indexed terminal Transpose/Dequantize checkpoint compiles the complete
prior committed function AST and preserves exact ModelIR and both statistics
for Transpose-to-Dequantize and Dequantize-to-Transpose forms with one and two
simultaneous matches. Focused characterization covers a single index build
across both subphases, maintained-index equivalence, output-name and metadata
preservation, terminal/public/consumer boundaries, shared Transpose output,
per-channel quantization, invalid permutation, missing tensor, required-type/
no-index pruning, ownership, and the established real ONNX lowering case. The
focused selection passed with `35 passed`. Architecture, core, pass-efficiency,
quantization cleanup, real terminal sanitation, and both Concat-Q/DQ
characterizations passed together with `249 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. No Tier corpus conversion was run.

The indexed Transpose-DQ-Mean-Q checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR and statistics for one
and two simultaneous matches. A separate differential check proves that an
invalid permutation, which formerly left partially rewritten DQ/Mean metadata
despite returning zero, is now a complete ModelIR no-op. Focused coverage
includes one-index multi-match execution, maintained-index equivalence,
negative-axis remapping, edge and operator order, shape/signature propagation,
public and fan-out boundaries at every intermediate, `keepDims`, shared axes,
invalid axes/permutation, missing tensors, missing-required-type/no-index
pruning, and ownership with `48 passed`. Architecture, core, pass-efficiency,
and the complete quantization-cleanup suite passed with `260 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The indexed pseudo-LeakyReLU checkpoint compiles the complete prior committed
function AST and preserves exact ModelIR and statistics for one and two
simultaneous matches, including alpha constants on both MUL sides and nondefault
legacy SUB fields. Focused coverage verifies one index build, maintained-index
and LayoutState equivalence, batch producer removal, alpha values, reversed SUB
rejection, missing constant, every public intermediate, fan-out at every edge,
negative-source mismatch, integer boundaries, missing-family/no-index pruning,
and ownership with `17 passed`. Architecture, core, pass-efficiency, complete
graph cleanup, and the indexed fusion suite passed with `249 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The generic indexed MUL-square constant-fold checkpoint compiles the complete
prior committed YOLO-named function AST and preserves exact valid ModelIR and
statistics for one and two simultaneous matches. A separate differential check
proves that a public square intermediate, formerly rewritten despite losing its
producer contract, is now a complete no-op. Focused coverage verifies one-index
multi-match execution, all pre/anchor/scale constant-side combinations,
maintained-index and LayoutState equivalence, float16 fused values, normalized
metadata, quantization cloning, batch compaction, public/fan-out guards for all
three intermediates, singleton and finite pre-scale, floating anchor/scale,
finite result, exact self-square, missing constants, no-MUL/no-index/no-prune,
and generic ownership with `18 passed`. Architecture, core, pass-efficiency,
constant-fold, and indexed fold coverage passed with `234 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential quantization/
evaluation/coverage smoke passed with `3 passed`. No Tier corpus conversion was
run.

The indexed leading-singleton Gather-to-Reshape checkpoint compiles the
complete prior committed function AST and preserves exact valid ModelIR,
lineage metadata, and statistics for one and two simultaneous matches. A
separate differential check proves that multiple zero indices and a dynamic
leading signature, both formerly rewritten, are now complete no-ops. Focused
coverage verifies one-index multi-match execution, negative-axis
normalization, nested fixed-point exposure, maintained-index and LayoutState
equivalence, matching quantization, all public/fan-out/duplicate/order/input-
position boundaries, axis and batch-dimension options, static and dynamic
metadata consistency, constant buffer dtype/value/cardinality,
dtype/quantization equality, missing tensors, transactional rejection,
missing-family/no-index pruning, and unique semantic ownership with `38 passed`.
Architecture, core, pass-efficiency, ModelIR utilities, dynamic Reshape, and
indexed Gather/Reshape coverage passed with `269 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`. No
Tier corpus conversion was run.

The indexed terminal Softmax/Transpose checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR, lineage metadata,
and statistics for one and two simultaneous terminal outputs. Separate
differential checks prove that a separately public Softmax intermediate and a
missing Softmax-output tensor, both formerly rewritten, are now complete
no-ops. Focused coverage verifies one-index multi-output execution,
maintained-index and LayoutState equivalence, marker removal with axis/options
preservation, public output identity, all public input/output boundaries,
dtype/shape/signature propagation, quantization cloning, terminal and Softmax
fan-out, duplicate producers, operator order, marker truth, exact permutation,
operator arity/type, missing permutation and runtime buffer, missing tensors,
rank/signature validation, non-output exclusion, missing-family/no-index
pruning, shared marker ownership, and unique semantic ownership with
`30 passed`. Architecture, core, pass-efficiency, terminal Mean layout, layout
Transpose, and indexed terminal
Softmax coverage passed with `255 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed pre-ArgMax terminal-layout checkpoint compiles the complete prior
committed function AST and preserves exact valid ModelIR, lineage metadata,
and statistics for private singleton axes, two matches sharing one axis, and a
negative channel axis. Differential checks prove that a public axis remains
one while a private clone becomes three, instead of mutating the public value
as before, and that a Transpose intermediate exposed as a public input is now a
complete no-op. Focused coverage verifies one-index multi-match execution,
shared-axis ownership changes across differential removal, maintained-index
and LayoutState equivalence, private and public input/output axis constants,
negative-axis normalization, NumPy dtype and quantization cloning, exact
operator options/provenance, every public/fan-out/duplicate/order boundary,
permutation and operator arity/type, signed singleton axis validation, all
required tensors, rank-four permuted shape/signature and reduced-output
metadata, dtype agreement, transactional rejection, missing-family/no-index
pruning, and unique semantic ownership with `39 passed`. Architecture, core,
pass-efficiency, layout Transpose, indexed terminal ArgMax, and indexed terminal
Softmax coverage passed with `288 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed with
`3 passed`. No Tier corpus conversion was run.

The indexed quantized-MaxPool checkpoint compiles the complete prior committed
function and preserves exact ModelIR and statistics for valid one- and two-
chain fixtures across INT8 and UINT8. Differential checks separately prove
that the former tolerant comparison folded a near-but-different scale, that
missing float bridge tensors were accepted, and that a public-input bridge
could lose its producer; all three are now transactional no-ops. Focused
coverage verifies one-index multi-match execution, maintained-index and
LayoutState equivalence, quantized input/output fan-out, dictionary grids,
Pool options/version/provenance, cloned quantization, public boundaries,
duplicate producers, operator order and arity, exact grid/dtype/range, float
bridge dtype, exact rank-four shape/signature metadata, missing tensors,
missing-family/no-index pruning, and unique semantic ownership with
`52 passed`. Architecture, core, pass-efficiency, established quantized-Pool and
quantization-cleanup coverage, and the new indexed suite passed together with
`318 passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the new
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed quantized-Logistic checkpoint compiles the complete prior
committed function and preserves exact ModelIR and statistics for valid one-
and two-chain fixtures across INT8 and UINT8. Differential checks separately
prove that the former tolerant comparison folded a near-canonical output
scale, missing input quantization was accepted, missing float bridge tensors
were accepted, and a public-input bridge could lose its producer; all four are
now transactional no-ops. Focused coverage verifies one-index multi-match
execution, maintained-index and LayoutState equivalence, quantized input/output
fan-out, dictionary grids, rank-independent elementwise metadata, Logistic
options/version/provenance, public boundaries, duplicate producers, operator
order and arity, input grid validity, exact canonical output grid, float dtype,
shape/signature equality, missing tensors, missing-family/no-index pruning,
the two established direct tests, and unique semantic ownership with
`55 passed`. Architecture, core, pass-efficiency, established quantized-Pool
and quantization-cleanup coverage, both indexed quantized suites, and the two
direct Logistic tests passed together with `373 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test and
architecture test, syntax compilation, and `git diff --check` passed. No Tier
corpus conversion was run.

The indexed quantized-Softmax checkpoint compiles the complete prior committed
function and preserves exact ModelIR and statistics for valid one- and two-
chain fixtures across INT8 and UINT8, including the former beta tolerance.
Differential checks separately prove that the former tolerant grid comparison
folded a near-canonical output scale, missing input quantization and float
bridge tensors were accepted, a public-input bridge could lose its producer,
and an explicit non-last axis was ignored; all five are now transactional no-
ops. Focused coverage verifies one-index multi-match execution, maintained-
index and LayoutState equivalence, quantized input/output fan-out, dictionary
grids, rank-two negative-axis handling, beta/default/axis option preservation,
Softmax version/provenance, public boundaries, duplicate producers, operator
order and arity, input grid validity, exact canonical output grid, malformed
options, positive rank, float dtype, shape/signature equality, missing tensors,
missing-family/no-index pruning, and unique semantic ownership with
`62 passed`. The real QLinearSoftmax wrap conversion and sequential inference
matched ONNX exactly with `1 passed`. Architecture, core, pass-efficiency,
established quantized-Pool and quantization-cleanup coverage, and all three
indexed quantized suites passed together with `433 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test and
architecture test, syntax compilation, and `git diff --check` passed. No Tier
corpus conversion was run.

The indexed expanded-HardSigmoid fold checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for private one- and
two-chain INT8/UINT8 fixtures and a shared-four-constant fixture. Differential
checks prove that a near-equal output grid and missing float tensors formerly
folded, a public-input bridge could lose its producer, and public scalar
outputs were mutated in place. A fault-injected second quantization clone also
proves the former helper changed the first scalar data/dtype before raising,
while the new four-plan transaction returns a complete no-op. Focused coverage
verifies one-index multi-match execution, maintained-index and LayoutState
equivalence, all scalar input positions, shared/public clone ownership and
names, operator options/version/provenance, every public/fan-out/duplicate/
order/arity boundary, exact grid validity, all data metadata, finite singleton
constants, producer rejection, representability, clone-failure transaction,
missing-family/no-index pruning, and unique semantic ownership with
`78 passed`. Architecture, core, pass-efficiency, the established quantized
Pool and quantization-cleanup suites, all prior indexed quantized folds,
quantized activation and
expanded-HardSigmoid bridge suites, and the new fold suite passed together
with `529 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test and architecture test, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed quantized-TransposeConv checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for valid private
one- and two-chain fixtures and a shared-filter fixture. Differential checks
prove that missing float input/output tensor records and a public-input bridge
formerly folded, a public float filter was mutated in place, an invalid
negative input scale was accepted, and output-grid clone failure changed a
private float filter before raising. The new owner rejects or clones each case
transactionally. Focused coverage verifies one-index multi-match execution,
maintained-index and LayoutState equivalence, private/shared/public and
already-INT8 filter ownership, exact output-shape/filter/data roles, operator
options/version/provenance, every public/fan-out/duplicate/order/arity
boundary, independent activation-grid validity, bridge dtype and metadata,
filter rank/buffer/dtype/grid/producer constraints, clone-failure transaction,
missing-family/no-index pruning, and unique semantic ownership with
`61 passed`. Architecture, core, pass-efficiency, the established quantized
Pool and quantization-cleanup suites, all prior indexed quantized folds,
quantized activation and expanded-HardSigmoid bridge/fold suites, and the new
TransposeConv suite passed together with `590 passed`. TensorFlow-import-
blocked direct and `-cotof` plus the sequential quantization/evaluation/
coverage smoke passed with `3 passed`. Ruff on the new owner/test, scoped
architecture/lowerer checks, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed decomposed-InstanceNormalization checkpoint compiles the complete
prior committed function and preserves exact ModelIR/statistics for valid
NHWC, NCHW, post-Transpose, and rank-five fixtures. Differential checks prove
that reversed Sub operands, a non-Add epsilon node, a public Mean intermediate,
a shared scale, a wrong-sized scale, and floating axes formerly mutated graph
state. A malformed late bias shape additionally proves that the former helper
changed axes, intermediate shapes, and scale data before raising, while the
new owner returns a complete no-op. Focused coverage verifies one-index
multi-layout execution, maintained-index and LayoutState equivalence, ranks
three/four/five, separate/shared Mean axes, optional post-Transpose bias-axis
mapping, already-correct idempotence, plan-failure transaction, all operator
types/roles/arity/order, public/fan-out/duplicate boundaries, finite epsilon
and reciprocal-one constants, axes dtype/ownership, complete intermediate
metadata, scale/bias cardinality and ownership, missing-marker/no-index
behavior, and unique semantic ownership with `38 passed`. Four existing real
ONNX builder/serialization characterizations passed. Architecture, core,
pass-efficiency, the new indexed suite, and those real characterizations passed
together with `264 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed NCHW Concat/global-pool/Conv checkpoint compiles the complete prior
committed function and preserves exact ModelIR/statistics for valid one- and
two-chain fixtures, negative spatial axes, and INT64 Reshape shape buffers.
Differential checks prove that non-global Mean axes, a fan-out or public Concat
intermediate, a three-value or floating shape buffer, a duplicate Reshape
producer, and a missing runtime filter buffer formerly changed graph state. A
faulting late shape-buffer read additionally proves that the former helper
changed the Concat axis and three tensor records before raising, while the new
owner returns a complete no-op. Focused coverage verifies one-index multi-
match execution, maintained-index and LayoutState equivalence, exact operator
roles/order/arity, normalized global axes, public/fan-out/duplicate boundaries,
positive compatible NCHW inputs, the filter/channel equation and buffer,
private producer-free integer shape-buffer ownership, option/metadata/data
updates, incomplete-family/no-index behavior, the existing characterization,
and unique semantic ownership with `34 passed`. Architecture, core, pass-
efficiency, all existing Conv-layout tests, and the new indexed suite passed
together with `268 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed NCHW Concat/Transpose/(Transpose)Conv checkpoint compiles the
complete prior committed function and preserves exact ModelIR/statistics for a
combined direct-Conv, pre/post-prefix Conv, and TransposeConv fixture.
Differential checks prove that public or fan-out Transpose/pre-passthrough
outputs, a duplicate Transpose-output producer, a nonpositive input channel,
a produced permutation, and a missing runtime filter buffer formerly changed
graph state; the new owner rejects all eight cases transactionally. Focused
coverage verifies one-index multi-family execution, maintained-index and
LayoutState equivalence, the four established positive characterizations,
exact Transpose and data/filter roles, pre/post passthrough traversal,
public/fan-out/duplicate/order boundaries, positive compatible NCHW inputs,
the filter/channel/buffer equation, direct-Conv-only output refresh, already-
correct exclusion, incomplete-family/no-index behavior, and unique semantic
ownership with `30 passed`. Architecture, core, pass-efficiency, all existing
Conv-layout tests, and both new indexed Concat repair suites passed together
with `294 passed`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test, scoped architecture/lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed mixed-singleton NCHW-input/NHWC-Concat checkpoint compiles the
complete prior committed function and preserves exact ModelIR, lineage,
operator order, deterministic names, quantization metadata, and statistics for
valid multi-candidate, multi-adapter, and name-collision fixtures. Differential
checks prove that an output-channel mismatch, duplicate or later source
producer, inconsistent dynamic signature, and mixed dtype formerly changed
graph state; all five are now complete no-ops. A late fault in the second
quantization clone additionally proves that the former helper inserted the
first Reshape before raising, while the new owner leaves ModelIR unchanged.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, public/produced/constant sources, same-source adapter
reuse, global name reservation, deep quantization cloning, exact Concat axis,
arity, output-channel and dtype equations, static and one-dynamic-dimension
shape/signature contracts, final shape-reconciliation stability, duplicate/
later/unresolved producer rejection, clone-failure transaction, no-repair and
missing-family behavior, the established characterization, and unique
semantic ownership with `33 passed`. Architecture, core, pass-efficiency,
singleton-Reshape coverage, the two preceding indexed Concat owners, flatten-
Concat and NDHWC-Concat coverage, the new indexed suite, and the existing
characterization passed together with `336 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. Ruff on the new owner/test and architecture test,
scoped lowerer checks, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed window-partition checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, Transpose
options replacement, version/axis semantics/provenance, quantization metadata,
unused-tensor pruning, and statistics for valid public-input, produced-input,
and quantized multi-chain fixtures. Differential checks prove that duplicate
first-Reshape producers, floating shape vectors, missing final-output metadata,
mixed data dtypes, and a graph-input/producer conflict formerly changed graph
state; all five are now complete no-ops.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, exact block/spatial and three-stage shape equations,
static and one-dynamic-dimension contracts, final dynamic-Reshape convergence,
per-tensor quantization, in-place provenance, topology/order/arity, public and
fan-out boundaries, duplicate/later/unresolved producers, typed producer-free
shape/permutation vectors, dynamic shape-vector ownership, two-dynamic-output
rejection, historical prune/no-index behavior, and unique semantic ownership
with `52 passed`. Architecture, core, pass-efficiency, dynamic-Reshape, the new
indexed suite, and two real ONNX SpaceToDepth chain characterizations passed
together with `289 passed`. TensorFlow-import-blocked direct and `-cotof` plus
the sequential quantization/evaluation/coverage smoke passed with `3 passed`.
Ruff on the new owner/test and architecture test, scoped lowerer checks, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed window-reverse checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, deterministic
shared-shape clone names, Reshape and DEPTH_TO_SPACE options, version/axis
semantics/provenance, quantization metadata, unused-tensor pruning, and
statistics for a five-chain public-input, produced-input, quantized, and
shared-vector fixture. Differential checks prove that an extra first-Reshape
input, a floating shape vector, a public first shape vector, mixed data dtypes,
and an inconsistent shape signature formerly changed graph state; all five are
now complete no-ops.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, legacy sequential shared-vector clone/update behavior,
exact reverse block/flatten/spatial and three-stage shape equations, static and
one-dynamic-dimension contracts, final shape-convergence stability, per-tensor
quantization, in-place provenance, topology/order/arity, public and fan-out
boundaries, duplicate/later/unresolved producers, typed producer-free vectors,
two-dynamic-target rejection, clone-failure transaction, historical
prune/no-index behavior, unique semantic ownership, and a production real-ONNX
characterization. The reverse and partition focused suites passed together
with `99 passed`. Architecture, core, pass-efficiency, dynamic-Reshape,
ModelIR-writer, strict-integer-quantization, both indexed window suites, and an
established DepthToSpace characterization passed together with `381 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed Conv1D-shim unary checkpoint compiles the complete prior committed
function and preserves exact ModelIR, lineage, operator order, unary identity,
options/version/axis semantics/provenance, output dtype and quantization repair,
unused-tensor pruning, and statistics for a five-chain public-input,
produced-input, quantized, CAST, inferred-axis, and alternate-axis fixture.
Differential checks prove that a floating permutation, produced ExpandDims
axis, inconsistent internal shape metadata, mixed input dtype, and duplicate
final producer formerly changed graph state; all five are now complete no-ops.
A faulting quantization clone additionally proves that the former helper
rewired the unary before raising, while the indexed owner leaves ModelIR
unchanged.

Focused coverage verifies one-index multi-match execution, supplied-index and
LayoutState equivalence, all sixteen supported unary types, explicit/negative
and uniquely inferred axes, CAST dtype transition, output metadata repair,
consistent multi-dynamic signatures, exact NHWC/NCHW permutation and
drop/insert equations, every operator role/order/arity, public and fan-out
boundaries, duplicate/later/unresolved producers, typed producer-free
permutation and axis vectors, per-tensor quantization, per-axis rejection,
clone-failure transaction, historical prune/no-index behavior, and unique
semantic ownership with `75 passed`. Architecture, core, pass-efficiency,
three established transpose-unary suites, five adjacent Conv1D-shim
characterizations, and the new indexed suite passed together with `321
passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff on the
owner/test and architecture test, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed rank-four Conv1D-shim unary checkpoint also compiles the complete
prior committed function. A static public-input, produced-input, quantized,
and CAST four-chain fixture preserves exact operators, tensors, inputs,
outputs, metadata, and statistics. Differential checks prove that floating
permutation constants, a produced ExpandDims axis, mixed data dtypes, and a
duplicate final producer formerly rewrote graph state; all four are now
complete no-ops. Focused coverage verifies one-index multi-match execution,
supplied-index and LayoutState equivalence, deterministic shared shape/axis
constant cloning, fan-out bridge topology, one dynamic height/width/channel
dimension, 53 unsafe contracts, quantization-clone failure, historical
prune/no-index behavior, and both established direct-builder
characterizations with `63 passed`. Both indexed Conv1D suites, architecture,
core, pass-efficiency, three established Transpose-Unary suites, and five
adjacent direct-builder characterizations passed together with `383 passed`.
TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed Conv1D unary fan-out checkpoint compiles the complete prior
committed helper. For four static public-input, produced-input, quantized, and
alternate-axis branches, the new ModelIR and statistics are exactly equal to
the legacy result after topologically sorting that result. The legacy result
required twelve operator moves; the new owner emits the final order directly.
CAST additionally repairs the retained Transpose output from `FLOAT32` to the
unary's `INT32` output contract. Differential checks prove that a chain without
fan-out, floating permutation, produced ExpandDims axis, duplicate final
producer, and mixed input dtype formerly rewrote graph state; all five are now
complete no-ops. A faulting quantization clone formerly raised after changing
the final dtype, while the indexed owner leaves ModelIR unchanged.

Focused coverage verifies one-index five-chain execution, supplied-index and
LayoutState equivalence, all sixteen unary types, public NCHW output, three
dynamic-signature forms, 27 unsafe contracts, clone-failure transaction,
equivalent negative axes, historical prune/no-index behavior, and the
established direct-builder characterization with `58 passed`. All three
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites,
and five adjacent direct-builder characterizations passed together with `437
passed`. TensorFlow-import-blocked direct and `-cotof` plus the sequential
quantization/evaluation/coverage smoke passed with `3 passed`. Ruff with only
the documented legacy unused-import exclusions, syntax compilation, and `git
diff --check` passed. No Tier corpus conversion was run.

The indexed flattened InstanceNormalization checkpoint compiles the complete
prior committed 491-line helper. Four static public-input, produced-input,
CAST, and alternate-unary branches retain exact operators, tensors, inputs,
outputs, metadata, and statistics. Differential checks prove that a floating
permutation, produced second-Reshape shape, negative epsilon, reversed
reciprocal DIV, mixed intermediate dtype, and duplicate final producer
formerly rewrote graph state; all six are now complete no-ops.

Focused coverage verifies one-index five-chain execution, supplied-index and
LayoutState equivalence, deterministic shared Reshape/axis cloning, all
sixteen unary types, one dynamic batch/width/channel dimension, 37 unsafe
contracts, clone-failure transaction, historical prune/no-index behavior, and
the established direct-builder characterization with `61 passed`. All four
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and five adjacent direct-builder characterizations
passed together with `498 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed tencoder checkpoint removes the former 723-line raw helper from
the lowerer. Its dedicated owner matches the exact flattened
InstanceNormalization prefix rather than searching an arbitrary upstream
producer path, validates the simple rank-four or legacy rank-three residual
branch, and proves the complete two-Slice/Logistic/Mul/scale gate, residual
ADD, ExpandDims, post-Transpose, and Conv consumer topology before mutation.
The InstanceNormalization owner now exposes that common prefix as a
side-effect-free plan while retaining its prior complete-chain behavior.

The rewrite converts both residual inputs from NCW to NWC, adjusts the second
Reshape, Slice begin/size, floating channel-scale, and ExpandDims constants,
repairs every changed tensor including Logistic and gate intermediates, and
removes the three boundary Transposes in one indexed compaction. Private
constants update in place; shared constants receive unique planned clones even
when two changed operator inputs originally share the same tensor. A side
consumer receives one `[0,2,1]` bridge immediately before its earliest use.

Focused coverage includes exact numeric equivalence for simple/legacy left
branches with and without fan-out, supplied-index and LayoutState equivalence,
topological bridge placement, deterministic shared integer/float cloning,
one dynamic batch or width dimension, eight unsafe transactional no-ops, the
three established characterizations, and both semantic-ownership checks with
`84 passed`. The four indexed Conv1D
suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and eight adjacent direct-builder characterizations
passed together with `520 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed Conv1D BatchMatMul checkpoint removes the former 223-line helper
and its repeated whole-graph consumer-map rebuild. Its dedicated owner accepts
only exact typed `[0,3,1,2]` Transpose roots, one explicit singleton Squeeze,
an optional strict supported-unary chain, and the left input of one
BatchMatMul. Source and intermediate shapes/signatures, producer order,
consumer multiplicity, public boundaries, dtype transitions, per-tensor
quantization, right operand, contracted dimensions, batch broadcasting, and
output shape are all validated before mutation.

Squeezing the transposed channel axis maps to the original NHWC channel axis
without changing rank-three order or `adjX`. Squeezing either supported
spatial axis maps back to the corresponding NHWC spatial axis, swaps the last
two dimensions of Squeeze and unary metadata, and toggles `adjX`; the effective
matrix and BatchMatMul output remain identical. The original unary and
BatchMatMul objects retain all unrelated options, provenance, version, and
axis semantics, and the single boundary Transpose is removed through the
maintained index.

Focused coverage includes twelve exact NumPy equivalence variants across all
three axes, both `adjX` values, and public/produced sources; zero-length unary
chains, rank-two right operands, `adjY`; all sixteen unary types; one dynamic
batch signature; fifteen unsafe transactional no-ops; the preflight/no-index
path; and semantic ownership with `49 passed`. The five
indexed Conv1D suites, architecture, core, pass-efficiency, three established
Transpose-Unary suites, and eight adjacent direct-builder characterizations
passed together with `569 passed`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed`. Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed decoder deconvolution-input checkpoint removes the former
193-line helper and its repeated producer/consumer-map rebuilds. Its dedicated
owner validates the complete BatchMatMul, commutative bias ADD, axis-two
ExpandDims, `[0,2,3,1]` Transpose, and input-two TransposeConv path. Both matrix
operands, every producer and consumer, public boundaries, concrete and dynamic
shape/signature equations, dtype, per-tensor quantization, contracted
dimension, batch broadcasting, bias broadcast, and operator order are proven
before any mutation.

The rewrite applies `(A·B)^T = B^T·A^T`: it swaps the original BatchMatMul
inputs, maps `adjX` to `not old_adjY` and `adjY` to `not old_adjX`, changes
`[N,C,L]` metadata to `[N,L,C]`, reshapes the length bias to `[1,L,1]`, moves
ExpandDims from axis two to axis one, and connects its retained
`[N,1,L,C]` output directly to TransposeConv. Unrelated BatchMatMul options and
provenance remain intact. Private constants update in place; shared bias and
axis constants are cloned before edges change. Clone failure and every unsafe
contract are complete no-ops.

Focused coverage includes sixteen exact NumPy equivalence variants across all
`adjX/adjY` values, both ADD input orders, and public/produced operands; rank-
two RHS, rank-two/three bias, negative axis, shared bias/axis cloning, one
dynamic batch signature, twenty-six unsafe transactional no-ops, clone
failure, preflight/no-index behavior, and semantic ownership with `51 passed`.
The six indexed Conv1D/decoder suites, architecture, core, pass-efficiency,
three established Transpose-Unary suites, and eight adjacent direct-builder
characterizations passed together with `620 passed`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed terminal Squeeze/Mean checkpoint removes the former 152-line raw
helper and its repeated whole-graph consumer-map rebuild. Its dedicated owner
validates the exact `[0,3,1,2]` Transpose, axis-two Squeeze, axis-one kept-
dimension Mean, and axis-one terminal Squeeze topology. Every producer,
consumer, public boundary, typed constant, operator order, singleton,
rank-four/rank-three/rank-two shape and signature equation, dtype, and per-
tensor quantization contract is proven before mutation.

The rewrite removes the Transpose, moves the first Squeeze to the NHWC source
axis one, changes `[N,C,W]` metadata to `[N,W,C]`, moves the Mean to axis two,
changes `[N,1,W]` metadata to `[N,W,1]`, and moves the terminal Squeeze to axis
two. The final `[N,W]` tensor name, values, metadata, graph-output position,
and downstream edges remain unchanged. Private Mean axes update in place;
shared axes receive a deterministic clone. Negative equivalent axes and one
dynamic batch, width, or reduced-channel signature are preserved. Clone
failure and all unsafe contracts are complete no-ops.

Focused coverage includes sixteen exact NumPy equivalence variants across
public/produced sources and all positive/negative axis representations,
shared-axis cloning, dynamic batch/width/reduced-channel signatures, two-chain
execution, twenty-seven unsafe transactional no-ops, clone failure,
apply-preflight collision, and preflight/no-index behavior with `51 passed`;
the semantic-ownership test also passed. The seven indexed
Conv1D/decoder/terminal suites, architecture, core, pass-efficiency,
Mean/terminal-Mean, and three Transpose-Unary suites passed
together with `677 passed in 56.59s`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed in 6.81s`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed direct InstanceNormalization checkpoint makes the single private
post-Transpose tail the first bounded owner within the former four-mode raw
helper. It proves the pre-Transpose/Squeeze/Reshape boundary, both kept-
dimension Means, ordered SUB and reciprocal DIV, square/normalize/affine
branches, and the sole post-Transpose before changing ModelIR. Exact
shape/signature equations, all producers and consumers, graph order, public
boundaries, FLOAT16/FLOAT32 dtype, unquantized tensors, typed reshape/axis
constants, nonnegative finite epsilon, unit numerator, and finite scale/bias
are validated in one `ModelIRGraphIndex`.

The rewrite moves the normalization to NHWC axes `[1,2]`, converts rank-three
CHW and every full/reduced rank-four metadata contract, changes scale and bias
from `[1,C,1,1]` to `[1,1,1,C]`, reuses the post-Transpose output name on the
bias ADD, and removes the two boundary Transposes. Reshape, shared Mean-axis,
scale, and bias constants are planned together; unrelated consumers receive
deterministic clones. Five representative static legacy variants are exactly
ModelIR-identical to the committed helper, including lineage-event order.
Separate equivalent Mean axes are now handled transactionally instead of
being skipped by the legacy shared-axis guard.

The side-Squeeze, Squeeze/unary/Reshape, and
Squeeze/residual-ADD/Reshape modes remain in the compatibility path. Each is
an indexed-owner no-op followed by a numerically equivalent legacy rewrite;
supplied GraphIndex and LayoutState remain current. Unsafe direct candidates
such as reversed SUB, wrong axes, negative epsilon, or quantized intermediates
are explicitly short-circuited and cannot fall through to the legacy mutator.
Per-candidate dispatch preserves graph order across mixed direct and legacy
tails; a 33-chain characterization proves the original shared 32-rewrite
limit selects the same prefix rather than favoring all direct tails first.

Focused coverage includes thirty-two NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced source, shared/separate axes, positive/negative
axes, and affine input order; shared cloning for all four changed constant
roles; dynamic height/width/channel signatures; two-chain execution; forty-
four unsafe transactional no-ops; clone failure; apply-preflight collision;
four legacy-fallback blockers; all three retained legacy modes; compatibility
statistics; mixed-mode order/limit; and preflight/no-index behavior with
`93 passed`. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `495 passed in 54.58s`; twelve selected direct-builder
characterizations passed in `0.94s`. TensorFlow-import-blocked direct and
`-cotof` plus the sequential quantization/evaluation/coverage smoke passed
with `3 passed in 6.70s`. Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed side-Squeeze InstanceNormalization checkpoint reuses that common
decomposition matcher for the exact dual-consumer tail: one NCHW-to-NHWC
post-Transpose and one axis-zero Squeeze. It moves the common normalization to
NHWC, removes the two boundary Transposes, and inserts one local
NHWC-to-NCHW adapter immediately before the side Squeeze. The side tensor name,
CHW shape/signature, dtype, quantization contract, public-output behavior, and
downstream edges remain unchanged. A compatible shared INT32 or INT64 adapter
permutation is reused; every absent, conflicting, produced, public, floating,
or quantized constant case is resolved before mutation. Unsafe side candidates
cannot fall through to the legacy mutator. Direct and side candidates retain
their original graph-order position and shared 32-rewrite ceiling; the two
remaining unary/Reshape and residual-ADD/Reshape modes are untouched.

Focused coverage includes sixteen NumPy equivalence variants across side
operator order, public/downstream-consumed side outputs, public/produced
sources, and FLOAT16/FLOAT32; dynamic height/width/channel signatures;
multi-chain constant reuse; five invalid existing-adapter cases; nine unsafe
side contracts; adapter-allocation collision; three compatibility-fallback
blockers; and the existing direct and mixed-mode cases with `131 passed in
0.89s`. Four representative side-tail variants were exactly ModelIR-identical
to the committed legacy helper. The related InstanceNormalization, Pad, Mean,
architecture, core, and pass-efficiency suites passed with `533 passed in
53.58s`; twelve selected direct-builder characterizations passed with `12
passed in 0.94s`. TensorFlow-import-blocked direct and `-cotof` plus the
sequential quantization/evaluation/coverage smoke passed with `3 passed in
6.67s`. Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed unary/Reshape InstanceNormalization checkpoint reuses the same
core matcher for the exact `Squeeze -> unary -> Reshape -> post-Transpose`
tail. The tail Squeeze and unary metadata move from CHW to HWC, the second
Reshape moves from NCHW to NHWC, its typed shape constant and `newShape` are
rewritten together, and that Reshape receives the former post-Transpose output
name. Shared shape constants receive deterministic clones. All thirteen
legacy unary operators remain accepted; CAST alone may change dtype, while
every other unary retains the core FLOAT16/FLOAT32 dtype. Quantized or unsafe
tail contracts are complete no-ops and cannot fall through to the legacy
mutator. The residual-ADD/Reshape tail is unchanged and remains the sole
legacy mode.

Focused coverage includes eight exact NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced sources, and shared/separate Mean axes; all
thirteen unary operators; dynamic height/width/channel signatures; shared
tail-shape cloning; multi-chain execution; thirteen unsafe transactional
no-ops; three compatibility-fallback blockers; mixed direct/side/unary graph
order under the shared 32-rewrite ceiling; and preflight/no-index behavior
with `175 passed in 0.73s`. Static, produced-source, FLOAT16, negative-axis,
commuted-affine, and CAST variants were exactly ModelIR-identical to the
committed legacy helper. Separate equivalent Mean axes are an intentional
improvement over the legacy shared-axis restriction. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `577 passed in 54.16s`; twelve selected direct-builder
characterizations passed with `12 passed in 0.94s`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed in 6.63s`. Ruff, syntax compilation, and `git diff
--check` passed. No Tier corpus conversion was run.

The indexed residual/Reshape InstanceNormalization checkpoint completes the
four-mode migration. It validates the main `Squeeze -> residual ADD -> Reshape
-> post-Transpose` tail together with either a rank-three HWC-to-CHW residual
bridge or an NHWC-to-NCHW bridge followed by Squeeze and an optional retained
unary. The residual bridge is removed only after every shape/signature, dtype,
quantization, public-boundary, fan-out, producer, consumer, and graph-order
contract succeeds. Residual unary CAST may change FLOAT16/FLOAT32 dtype; the
other twelve operators must preserve it.

ADD output fan-out is planned in the same transaction. Its Reshape path moves
to HWC/NHWC, while all other consumer slots share one deterministic
HWC-to-CHW adapter. A compatible INT32/INT64 fixed permutation is reused;
invalid, produced, public, quantized, or collision cases are complete no-ops.
After this migration the former 975-line compatibility helper contains no raw
ModelIR mutation: it is a 60-line dispatcher that tries the four indexed modes
at each pre-Transpose's original graph position and preserves their shared
32-rewrite ceiling.

Focused coverage includes twelve NumPy equivalence variants across all three
residual source forms, FLOAT16/FLOAT32, and public/produced main sources; all
thirteen residual unary operators; nine dynamic-signature combinations; four
fan-out position/repeated-slot variants; multi-chain execution; commuted ADD;
existing adapter reuse; fifteen unsafe transactional no-ops; four
compatibility-fallback blockers; adapter-allocation collision; all four mixed
tail modes under the shared cap; and preflight/no-index behavior with `238
passed in 0.92s`. Ten representative static, produced-source, FLOAT16,
negative-axis, commuted-affine, residual-source, and fan-out variants were
exactly ModelIR-identical to the committed legacy helper. The related
InstanceNormalization, Pad, Mean, architecture, core, and pass-efficiency
suites passed with `640 passed in 54.91s`; twelve selected direct-builder
characterizations passed with `12 passed in 1.00s`. TensorFlow-import-blocked
direct and `-cotof` plus the sequential quantization/evaluation/coverage smoke
passed with `3 passed in 6.92s`. Ruff, syntax compilation, and `git diff
--check` passed. No Tier corpus conversion was run.

Residual-specific fixtures and cases are isolated in
`test_flatbuffer_direct_indexed_instance_norm_residual_layout.py`; the common
ModelIR builder/evaluator remains in the direct-tail module. This keeps the
common characterization file at 1,825 lines and the residual module at 642
lines without duplicating fixture construction.

The focused indexed and architecture tests pass Ruff normally. The changed
legacy characterization file passes with its pre-existing `F401` findings
scoped out, and the lowerer passes with its pre-existing `F401` and `F841`
findings scoped out. Every changed Python file passes `python -m py_compile`,
and `git diff --check` passes. The
immediately preceding DepthToSpace, Pool, dynamic-Pool, simple-alias, and
aligned-scalar checkpoints passed their focused synthetic and ownership
selections.

The indexed post-Transpose-bias InstanceNormalization checkpoint extracts the
rank-four decomposition matcher and generic constant transaction into
`passes/decomposed_instance_norm.py`. Both the established pre/post-tail owner
and the new `passes/instance_norm_post_bias_layout.py` owner now validate the
same exact Mean/SUB/square/variance/epsilon/SQRT/reciprocal/normalize/scale
contract. Epsilon must be finite and nonnegative, the DIV numerator must be
exactly one, all retained tensors must share one unquantized FLOAT16/FLOAT32
contract, and every producer, consumer multiplicity, public boundary,
shape/signature, and graph-order relation is proven before mutation.

The new owner accepts shared or separate positive, negative, or reversed Mean
axes; commuted SUB and affine operands; scalar, NCHW, or already-NHWC scale and
bias constants; public or produced NHWC sources; and dynamic height, width, or
channel signatures. Axes and coefficients are planned by use. Private values
update in place, unrelated consumers receive deterministic clones, and one
constant shared by scale and bias is updated once for both slots. Rejected
candidates do not prune orphan tensors or partially rewrite constants. On
success the Means and SUB consume the original NHWC source, axes become
`[1,2]`, retained core metadata becomes NHWC, the bias ADD consumes the scaled
tensor directly, and only the two boundary Transposes are removed.

The lowerer helper is now a 19-line compatibility dispatcher. In the repeated
normalization recovery loop, the four-tail owner and post-bias owner share one
live `ModelIRGraphIndex`; every late production call supplies the Session
`LayoutState`. The new owner has one bounded graph-order candidate scan, a
32-rewrite ceiling, differential index updates, and no repeated full consumer
map or unbounded fixed-point loop. The existing four-tail characterization
remains unchanged after adopting the common matcher.

Focused coverage includes twenty-four NumPy equivalence variants across
FLOAT16/FLOAT32, public/produced source, shared/separate and positive/negative/
reversed Mean axes, commuted SUB/affine operands, and valid scalar/NCHW-scale/
NHWC-bias forms; three dynamic-signature cases; a public bias-ADD output;
shared changed-constant cloning; one shared scale/bias tensor; legacy
coefficient-layout acceptance; two-chain capped execution; thirty-six unsafe
transactional no-ops; clone-allocation collision; and preflight/no-index
behavior. The post-bias, existing four-tail, compatibility direct-builder, and
ownership selections passed with `310 passed in 1.93s`. The full architecture
suite passed with `176 passed in 52.04s`; thirteen selected InstanceNorm direct-
builder tests passed with `13 passed in 1.42s`; TensorFlow-import-blocked import,
direct conversion, and `-cotof` passed sequentially with `3 passed in 3.97s`.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed InstanceNormalization residual-ADD checkpoint moves the adjacent
dual-branch layout rewrite to
`passes/instance_norm_residual_add_layout.py`. The owner accepts an exact
`NHWC -> NCHW` main branch, the common decomposed InstanceNorm core through
scale and bias, a second `NHWC -> NCHW` residual branch, and a tail ADD with
downstream NCHW consumers. It lifts both branches and the tail ADD to NHWC,
removes both pre-Transposes, and inserts one post-ADD `[0,3,1,2]` adapter that
preserves the original tensor name, NCHW metadata, downstream fan-out, and
repeated consumer slots.

The common helper now plans Mean axes, scale, and bias together before any
mutation. Private constants update in place, shared values receive deterministic
clones, scalar/NCHW/already-NHWC affine forms are handled explicitly, and
existing adapter constants are reused only after their dtype, value, ownership,
and quantization contracts are proven. The owner rejects public boundaries,
duplicate or backward producers, mixed dtypes, quantized normalization tensors,
invalid dynamic signatures, unsafe constant sharing, and all allocation
collisions without changing ModelIR. It uses one differential
`ModelIRGraphIndex`, a graph-order candidate scan, a deterministic 32-rewrite
ceiling, success-only pruning, and Session `LayoutState` synchronization. The
former 475-line lowerer mutator is now a 19-line dispatcher, and the repeated
normalization loop passes its live `residual_graph_index` into the owner.

Focused coverage includes thirty-two FLOAT16/FLOAT32 NumPy-equivalence
variants across public/produced main and residual sources, positive/negative/
reversed Mean axes, commuted affine operands, and alternate downstream
topologies; dynamic height, width, and channel signatures; fan-out and repeated
input slots; shared-constant cloning; valid adapter reuse; multi-chain rewrite
limits; thirty-nine transactional unsafe cases; three allocation-collision
cases; and preflight/no-index behavior. The new owner, adjacent post-bias and
four-tail owners, direct-builder characterizations, and ownership checks passed
with `395 passed in 2.45s`. The full architecture suite passed with
`177 passed in 56.80s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.47s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.21s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed residual-MUL/CONCAT InstanceNormalization checkpoint moves the
next three-Transpose tail to
`passes/instance_norm_residual_mul_concat_layout.py`. It validates the direct
decomposed core and bias, the first NCHW-to-NHWC bridge, one same-contract
residual ADD, exactly two tail MUL branches, one multiplicity-preserving
channel-axis CONCAT, and the final NCHW-to-NHWC bridge. The historical helper
name mentions Conv, but the indexed owner intentionally preserves the actual
legacy boundary: the former final-Transpose output may be a public output or
feed any later graph-ordered consumers, with no required Conv operator.

On success, all three Transposes are removed, the normalization and residual
tail run in NHWC, both tail-MUL outputs and CONCAT metadata are permuted, the
CONCAT axis changes from 1 to 3, and CONCAT directly produces the preserved
final output name. Dynamic height, width, and channel signatures, downstream
fan-out, repeated input slots, public outputs, and existing CONCAT option fields
are retained. The common constant planner now accepts additional coefficient
uses and plans both Mean axes, scale, bias, and both tail coefficients as one
transaction. Shared constants update once, unrelated users receive a
deterministic clone, and a late invalid tail coefficient can no longer leave
earlier axes or affine constants partially modified.

The owner requires exact producer, consumer multiplicity, graph-order,
shape/signature, dtype, quantization, typed-permutation, output-renaming, and
public-boundary contracts. It compares CONCAT inputs with `Counter`, uses one
differential `ModelIRGraphIndex`, scans candidates in graph order with a
32-rewrite ceiling, prunes only after success, and synchronizes the Session
`LayoutState`. Its former 501-line lowerer mutator is a 19-line dispatcher. All
four production calls pass LayoutState, and the repeated normalization loop
shares `residual_graph_index` with the preceding residual-ADD owner.

Focused coverage includes thirty-two FLOAT16/FLOAT32 logical-equivalence
variants across direct/produced main and residual sources, shared/separate and
positive/negative/reversed Mean axes, commuted operands, scalar/NCHW/NHWC
coefficients, and both CONCAT orders; three dynamic-signature cases; fan-out
and repeated input slots; public final output; shared-constant cloning; one
coefficient shared by all four affine uses; two-chain capped execution;
fifty-seven transactional unsafe cases; clone collision; and preflight/no-index
behavior. The new owner, adjacent five InstanceNorm owner groups,
direct-builder characterizations, and ownership checks passed with
`497 passed in 2.99s`. The full architecture suite passed with
`178 passed in 54.41s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.44s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.22s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed dual-statistics checkpoint moves the next normalization family to
`passes/instance_norm_dual_stats_layout.py`. Audit proved that neither branch
is the standard decomposed InstanceNorm core: one reduces spatial axes and one
reduces all non-batch axes, and both use
`SUB -> square -> Mean -> variance factor -> epsilon -> SQRT ->
DIV(centered,std) -> scale`. The owner therefore keeps a dedicated path matcher
and shares only typed constant, tensor-contract, metadata, and graph-index
utilities. This prevents a superficially similar topology from being assigned
the standard reciprocal/MUL normalization semantics.

The two branches feed blend ADD, gamma MUL, and beta ADD. Spatial axes are
planned together as `[1,2]`; global `[1,2,3]` axes are validated unchanged.
Branch scales and direct gamma/beta constants use one grouped coefficient
transaction, so shared constants update once and unrelated consumers receive a
deterministic clone. Exact `[1,C,1,1]` gamma/beta Reshapes from rank-one or
rank-two vectors are validated through their producer, shape constant, source,
consumer, dtype, quantization, and graph order, then bypassed and removed.
Variance factors and epsilon values must be finite, nonnegative scalar
FLOAT16/FLOAT32 constants.

Direct mode removes the input/output Transposes and gives beta ADD the former
NHWC output name. Residual mode additionally validates and removes an
independent NHWC-to-NCHW residual bridge, lifts the residual ADD to NHWC, and
uses that ADD as the preserved output producer. Its old NCHW output contract is
validated and permuted before rename, preventing dynamic-axis contamination.
The historical function name mentions Resize, but no Resize is required by the
legacy boundary; later ordered consumers, fan-out, and repeated input slots are
preserved.

The owner validates complete producer/consumer multiplicity, dependency order,
public boundaries, shape/signature, dtype, quantization, typed constants,
coefficient ownership, optional Reshape removal, and output rename before any
mutation. It uses a graph-order candidate scan, one differential
`ModelIRGraphIndex`, a 32-rewrite ceiling, success-only pruning, and Session
`LayoutState` synchronization. The former 712-line lowerer mutator is now a
19-line dispatcher; all four production calls pass LayoutState and the repeated
normalization loop shares `residual_graph_index` with the preceding owners.

Focused coverage includes forty-eight FLOAT16/FLOAT32 numerical-equivalence
variants across direct, residual-input, and produced-residual tails; direct and
produced main sources; shared/separate positive, negative, and arbitrarily
permuted axes; commuted operands; scalar/NCHW scales; direct and vector-Reshape
gamma/beta; six dynamic-signature modes; downstream fan-out and repeated slots
without
Resize; already-NHWC coefficients; shared-axis cloning; one coefficient shared
by all four affine sites; capped multi-chain execution; 143 transactional
unsafe contracts; clone collision; and preflight/no-index behavior. The new
owner and all indexed InstanceNorm owner groups passed with
`792 passed in 2.40s`. The full architecture suite passed with
`179 passed in 53.94s`; thirteen selected InstanceNorm direct-builder tests
passed with `13 passed, 742 deselected in 1.51s`; TensorFlow-import-blocked
import, direct conversion, and `-cotof` passed sequentially with
`3 passed in 4.45s`. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed affine-chain checkpoint moves
`_optimize_fold_mul_add_mul_affine_chains` to
`passes/affine_chain_fold.py`. The legacy helper distinguished constants only
by the presence of tensor data, rebuilt full producer/consumer maps inside an
unbounded loop, compared intermediate users as sets, mutated shared constants
before all checks were known, and copied the removed intermediate ADD tensor's
metadata onto the preserved final output. It did not validate dtype,
quantization, fused activation, public boundaries, producer order, constant
provenance, or the original and folded broadcast contracts.

The new owner accepts finite, non-variable FLOAT16, FLOAT32, and FLOAT64
constants with matching array dtype and exact static tensor metadata. All data
and output tensors must share that dtype and be unquantized, and all three
binary operators must have `NONE` fused activation. Original static broadcast
shapes and dynamic signatures are checked across all three operators; the two
folded broadcasts must independently reproduce the final output contract.
When the removed final MUL introduced a broadcast expansion, the surviving
first-MUL intermediate receives the correctly expanded shape and signature.
The final output tensor itself remains untouched, retaining its dtype,
quantization, shape, signature, layouts, and ONNX provenance.

The first-MUL and ADD constant roles are grouped before apply. Constants shared
inside those roles are updated once; sharing with the removed final MUL is an
allowed internal use. Any unrelated consumer receives one deterministic
`_folded` clone while the original constant is preserved. Produced, public,
variable, quantized, non-finite, mismatched, colliding, or incompatible
constants reject the complete plan. Exact producer uniqueness, consumer
multiplicity, graph order, intermediate privacy, source resolution, downstream
fan-out, repeated final-output slots, and output rename are proven with one
`ModelIRGraphIndex`. The plan is resolved again immediately before apply, the
candidate scan is graph ordered and capped at 32 rewrites, pruning is
success-only, and LayoutState is synchronized. The former 219-line lowerer
helper is now a 17-line dispatcher; all three production calls pass
LayoutState.

Focused coverage contains forty-eight FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across all operand orders and scalar/channelwise
broadcasts; dynamic signatures; final broadcast expansion; constants shared
within the chain or with unrelated consumers; downstream fan-out and repeated
slots; candidate-only and capped execution; thirty-three transactional unsafe
contracts; clone collision; and no-index preflight. The focused owner plus the
pre-existing direct-builder characterization passed with `91 passed in 0.57s`.
The full architecture suite passed with `180 passed in 54.55s`; eighteen
selected affine direct-builder tests passed with
`18 passed, 737 deselected in 1.45s`; TensorFlow-import-blocked import, direct
conversion, and `-cotof` passed sequentially with `3 passed in 4.24s`. Scoped
Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed affine pre/post checkpoint moves
`_optimize_transpose_mul_add_const_prepost_nhwc_chains` to
`passes/affine_prepost_layout.py`. The legacy helper rebuilt full maps in an
unbounded loop, contained a permanently disabled PRELU branch and unused
`valid_posts`, and rotated rank-four constants heuristically up to three times.
It preflighted constants separately but then mutated them sequentially, so a
late failure or clone-name interaction could leave a partial rewrite. It also
permuted the old ADD output metadata, copied that metadata to the canonical
post output, and permuted the canonical tensor again, producing a double-
transpose metadata path.

The new owner matches from a MUL candidate and proves one exact private
NHWC-to-NCHW pre adapter, the MUL/ADD chain, and every private inverse post
adapter. The pre Transpose is removed only when that MUL is its last consumer;
other pre fan-out remains intact. The ADD output may have multiple post
adapters but no legacy consumers. Their downstream uses are redirected to the
first graph-ordered post output with exact multiplicity, preserving fan-out and
repeated slots before all post adapters are removed. The canonical post tensor
is not overwritten. Its shape, signature, logical layout, physical layout,
dtype, quantization, and provenance remain authoritative; the surviving MUL
intermediate adopts that contract.

Finite FLOAT16/FLOAT32/FLOAT64 scalar and rank-four constants are supported.
Raw NCHW channel, spatial, and full constants rotate once to NHWC, while
already-NHWC constants remain stable for idempotent recovery. Known layout
annotations disambiguate orientation. If direct and rotated non-invariant data
are both compatible because axes have equal lengths, the candidate is rejected
instead of guessing. MUL and ADD roles are grouped into one plan, so a shared
constant updates once and unrelated consumers receive deterministic `_nhwc`
clones. Produced, public, variable, quantized, non-finite, wrongly typed or
shaped constants reject the complete transaction.

Typed private permutation vectors, rank-four shape/signature permutations,
same floating dtype, no quantization, no fused activation, unique producers,
exact consumer multiplicity, dependency order, private intermediates, source
resolution, alias layouts, and final renaming are all checked with one
`ModelIRGraphIndex`. The plan is resolved again immediately before apply. The
candidate scan is graph ordered and capped at 32 rewrites; pruning and
LayoutState maintenance are explicit. The former 409-line helper is now a
17-line dispatcher, and all seven production calls pass LayoutState.

Focused coverage contains forty-eight FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across all operand orders and scalar, channel,
spatial, and full constants; dynamic signatures; already-NHWC idempotence;
canonical layout propagation; retained pre fan-out; multi-post alias merging
and repeated slots;
constants shared inside and outside the chain; candidate-only and capped
execution; forty-four transactional unsafe contracts; equal-axis ambiguity;
clone collision; and no-index preflight. The focused owner plus two existing
direct-builder characterizations passed with `104 passed in 0.53s`. The full
architecture suite passed with `181 passed in 50.32s`; three selected related
direct-builder tests passed with `3 passed, 752 deselected in 0.45s`;
TensorFlow-import-blocked import, direct conversion, and `-cotof` passed
sequentially with `3 passed in 4.01s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed affine post-ADD checkpoint moves
`_optimize_transpose_mul_posttranspose_add_nhwc_chains` to
`passes/affine_post_add_layout.py`. The legacy helper rebuilt full producer and
consumer maps inside an unbounded loop, collapsed sole-consumer multiplicity
through a set, mutated or cloned the MUL constant during matching, and
permuted the surviving MUL output metadata heuristically. It did not validate
dtype, quantization, fused activation, public boundaries, producer uniqueness,
constant provenance, exact graph order, or the complete post-ADD fan-out
before mutation.

The new owner resolves from a graph-ordered MUL candidate. It proves a typed
private NHWC-to-NCHW pre adapter, the MUL output's one exact typed inverse post
adapter, and every consumer of the private post output as a plain ADD with a
finite same-dtype scalar or exact `[1,1,1,C]` side constant. Multiple ADD tails
and their downstream repeated slots remain intact. The pre Transpose is kept
when another NCHW branch uses it. Otherwise both adapters are removed, all ADD
tails consume the surviving MUL output, and that output adopts the canonical
post tensor's shape, dynamic signature, logical layout, and physical layout.

MUL constants share the affine pre/post owner's finite FLOAT16/FLOAT32/FLOAT64
orientation contract. Scalar, raw NCHW channel/spatial/full, already-NHWC, and
legacy direct non-rank-four forms are retained. Ambiguous equal-axis
non-invariant rank-four tensors are rejected. A changed constant with an
unrelated consumer receives a deterministic `_nhwc` clone. The complete plan
is resolved again before apply, and clone names, input slots, and operator
removals are preflighted before any mutation. One differential index, a
32-rewrite ceiling, success-only pruning, and LayoutState synchronization
replace the legacy loop. The former 278-line helper is a 17-line dispatcher;
all four production calls pass LayoutState. The Pad compatibility wrapper
continues to dispatch only to its independent `passes/pad_layout.py` owner.

Focused coverage contains twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across both operand orders and scalar, channel,
spatial, and full MUL constants; dynamic signatures; two ADD tails; canonical
layout propagation; legacy already-NHWC/vector constant modes; retained pre
fan-out; unrelated constant cloning; candidate-only and capped execution;
twenty-three transactional unsafe contracts; equal-axis ambiguity; clone
collision; and no-index preflight. The focused owner plus two existing direct-
builder characterizations passed with `56 passed`; the full architecture suite
passed with `182 passed in 51.86s`; the selected direct-builder tests passed
with `2 passed, 753 deselected in 0.46s`; and TensorFlow-import-blocked direct,
default, and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.57s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed SiNet Shuffle residual checkpoint moves
`_optimize_sinet_shuffle_residual_transpose_chains` to
`passes/sinet_shuffle_residual_layout.py`. The legacy helper rebuilt full
producer and consumer maps in an unbounded loop and matched forward and
backward from the second post-Transpose with incomplete ownership checks. It
collapsed several consumer checks through sets, did not validate graph order,
dtype, quantization, fused activation, public intermediates, constant
provenance, concrete/dynamic Concat contracts, or downstream order, and
rotated/cloned six constants sequentially before the complete candidate was
known. Shared external constants could receive multiple redundant clones. It
also permuted intermediate and canonical post metadata heuristically.

The new owner roots at the terminal post-Transpose and resolves all thirteen
operators: three NHWC-to-NCHW input adapters, the first residual ADD/MUL/ADD/
PReLU, its side post adapter, the channel Concat, the second MUL/ADD/PReLU, and
the final post adapter. Every private edge has one unique producer, exact
consumer multiplicity, and valid dependency order. The first PReLU has exactly
the intended post and Concat branches; the second PReLU has exactly its final
post. Later consumers of the two post outputs, including fan-out and repeated
input slots, remain unchanged. The Concat input order and commuted ADD/MUL
operands are preserved while its axis moves from NCHW channel 1 to NHWC
channel 3.

Rank-four concrete shapes and dynamic signatures are proven through both
residual branches and the Concat. Unknown non-channel dimensions propagate
conservatively; channel signatures sum only when known. All data tensors share
one unquantized FLOAT16/FLOAT32/FLOAT64 dtype. Five typed private permutation
constants and NONE fused activations are required. Six finite scalar or exact
NCHW/NHWC channel constants cover both affine/PReLU stages. Roles sharing one
constant are grouped, so one update or one deterministic `_nhwc` clone serves
all uses, including a scalar shared across both stages. Produced, public,
variable, quantized, non-finite, wrongly typed, mismatched, or colliding
constants reject the complete plan.

Both post-Transpose tensors remain unchanged and authoritative. The first
stage intermediates adopt the first post tensor's exact shape, signature, and
layouts; the Concat and second-stage intermediates adopt the second post
contract. Both PReLUs produce the canonical names directly, after which all
five adapters are removed differentially. The plan is re-resolved before
apply, and clone names, mutation indices, output tensors, and removal indices
are preflighted before the first write. Candidate traversal is graph ordered
and capped at 32 rewrites; pruning and LayoutState synchronization occur only
after success. The former 482-line helper is a 17-line dispatcher and its one
production call supplies LayoutState.

Focused coverage contains twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across scalar/raw constants, commuted operands and
Concat inputs, and both legal post/Concat orders; dynamic signatures;
canonical layout propagation; two legacy already-NHWC modes; constants shared
within one stage or as one scalar across both stages; one-clone external
sharing; repeated downstream slots; candidate-only and capped execution;
fifty-eight transactional unsafe contracts; clone collision; no-index
preflight; differential index validation; and LayoutState validation. The
focused suite passed with `90 passed in 0.51s`; the full architecture suite
passed with `183 passed in 53.22s`; the selected adjacent SiNet direct-builder
characterization passed with `1 passed, 754 deselected in 0.48s`; and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed, 8 deselected in 3.65s`. Scoped Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The paired post-MUL SiNet checkpoint moves
`_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains` to the same
`passes/sinet_shuffle_residual_layout.py` owner. Its first nine operators are
identical to the full-tail island, so both variants now use one
`_resolve_prefix` implementation for the three input adapters, first
ADD/MUL/ADD/PReLU stage, side post adapter, channel Concat, shape/signature
proof, three constant roles, canonical first-post metadata, public boundaries,
fan-out, and graph order. This removes the risk that fixes to the formerly
duplicated residual prefix diverge between the two legacy helpers.

The variant-specific tail proves
`Concat(NCHW) -> MUL -> post-MUL Transpose(NHWC) -> ADD -> PReLU`. The MUL and
post output are private and have exact producer/consumer multiplicity; the ADD
has one exact PReLU consumer. The final PReLU output may remain a public output
or keep later graph-ordered fan-out. Concrete and dynamic contracts require
the MUL output to equal the NCHW Concat, and the post, ADD, and PReLU outputs to
share the exact NHWC permutation. All tail tensors share the prefix floating
dtype and are unquantized. Fused activations, duplicate producers, invalid
order, partial fan-out, public intermediates, and stale metadata reject the
complete plan.

The shared six-role constant transaction rotates or retains the first-stage
constants, the second MUL constant, and the already-NHWC ADD/PReLU constants
together. Sharing, external clones, constant provenance, dtype, finiteness,
shape, signature, quantization, collision, and variable state use the same
contract as the full-tail owner. The MUL produces the existing post-Transpose
name directly. The post tensor, ADD output, final PReLU output, and their
provenance remain untouched; only the Concat intermediate adopts the canonical
post contract. Both plans are re-resolved before apply, and all mutations and
five removals are preflighted before the first write.

Variant-focused coverage adds twenty-four FLOAT16/FLOAT32/FLOAT64 numerical
equivalence combinations across scalar/mixed raw constants, commuted operands
and Concat inputs, and both legal post/Concat orders; two legacy raw tail
constant cases; external MUL-constant cloning; candidate-only and capped
execution; twenty-seven transactional unsafe tail contracts; clone collision;
and no-index preflight. The combined two-owner focused suite plus the existing
direct-builder characterization passed with `148 passed in 0.88s`; the full
architecture suite passed with `183 passed in 50.63s`; the selected SiNet
direct-builder characterization passed with
`1 passed, 754 deselected in 0.47s`; and TensorFlow-import-blocked direct,
default, and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.54s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed late-residual checkpoint moves
`_optimize_sinet_late_residual_pre_add_mul_add_prelu_chains` into the same
SiNet residual owner. The semantic candidate is two private
NHWC-to-NCHW adapters feeding `ADD -> MUL -> ADD -> PReLU`. Exactly one NHWC
source has a channel-last Concat producer. The PReLU output feeds one private
NCHW-to-NHWC adapter for a Conv2D or DepthwiseConv2D branch and one or more
later legacy NCHW consumers. The old fixed 40-by-40 predicate is gone; rank-
four concrete shapes and dynamic signatures must prove each permutation and
all affine intermediates must share one unquantized FLOAT16/FLOAT32/FLOAT64
contract.

The rewrite removes the two input adapters and lifts the affine/PReLU island
to NHWC. PReLU produces the existing canonical post tensor directly. The post
adapter is inverted in place and now produces the former PReLU tensor name, so
legacy consumers and repeated slots remain unchanged. Legacy consumers must
be graph ordered after that retained adapter; an earlier independent branch is
rejected instead of creating a producer-after-consumer edge. The canonical
post tensor remains authoritative and is never double-permuted. Only the
ADD/MUL/ADD intermediates adopt its exact shape, signature, and layouts; the
legacy output tensor retains its original NCHW metadata.

The three floating constants are grouped by tensor identity. Each must be a
finite, same-dtype, private constant that broadcasts in the original NCHW
graph; non-scalars are rotated and must also broadcast in the target NHWC
graph. This safely handles scalar and raw channel constants and retains the
legacy already-oriented rank-four case only when its actual axes make both
graphs valid. Unrelated consumers receive one deterministic clone. The
retained permutation constant participates in the same transaction and is
cloned when another Transpose still needs the original permutation.

The plan is re-resolved immediately before apply. Constant clones, mutation
indices, metadata targets, output-name swaps, and both adapter removals are
preflighted before the first write. One differential index, deterministic
candidate order, a 32-rewrite ceiling, success-only pruning, and LayoutState
synchronization replace the legacy full-map `while True` loop. The lowerer now
contains a 17-line dispatcher and the production call supplies the Session
LayoutState.

Focused coverage now passes with `207 passed in 0.69s`. It includes thirty-six
FLOAT16/FLOAT32/FLOAT64 numerical-equivalence combinations across both operand
orders, both downstream convolution families, scalar/raw constants, and the
formerly size-specific rank-four constant case on a non-40 spatial contract;
dynamic signatures; canonical, public, and repeated-slot legacy output
preservation; shared-role and external-use constant cloning; shared
permutation cloning; ambiguous oriented-constant rejection; candidate-only and
capped execution; fifteen transactional unsafe contracts; earlier legacy
consumer rejection; stale-plan revalidation; clone collision; no-index
preflight; differential-index validation; and LayoutState validation. The full
architecture suite passed with `183 passed in 50.11s`; the one sequential real
SiNet direct-builder characterization passed with `1 passed in 2.62s`; and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed with
`3 passed, 8 deselected in 3.58s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed deep-skip checkpoint moves
`_optimize_sinet_deep_skip_concat_resize_affine_tail_chains` to the dedicated
`passes/sinet_deep_skip_layout.py` owner. Matching is split into three bounded,
side-effect-free resolvers: the terminal Concat/MUL/post-Transpose/ADD/PReLU
tail, the central ADD/MUL/ADD/PReLU residual stage, and the Resize/affine dual
branch feeding the first Concat. The combined plan proves all sixteen semantic
operators and four removable adapters before any mutation; a partially
matched stage is never applied.

The Resize MUL/ADD, both channel Concats, the central residual stage, and the
terminal MUL are lifted to NHWC. The terminal ADD/PReLU retains its canonical
NHWC tensors. The central residual source may be an already annotated NHWC
tensor, as in the current SiNet graph, or an NCHW tensor with one explicit
earlier NCHW-to-NHWC adapter. Relational concrete shape, dynamic signature,
logical/physical layout, and typed permutation contracts replace the former
40-by-40 heuristic. The four removed adapters are the Resize input adapter,
the independent branch adapter, the deep-skip adapter, and the terminal post
adapter; the explicit central adapter, when present, remains available to its
existing branch and becomes the residual input.

Concat contracts are derived independently on original NCHW axis 1 and target
NHWC axis 3. Producer uniqueness, exact consumer multiplicity, dependency
order, private intermediates, floating dtype, quantization, fused activation,
Resize provenance, and canonical output metadata are validated across the
whole island. Six NCHW-side affine/PReLU constants must broadcast before and
after an explicit axis rotation. The already-NHWC terminal ADD/PReLU constants
use the canonical post contract. All eight roles are grouped by tensor
identity; unrelated consumers receive one deterministic clone and conflicting
shared orientations reject the plan.

Constant cloning/rewiring and metadata updates are now shared helpers used by
the full-tail, post-MUL, late-residual, and deep-skip SiNet owners. The deep-
skip plan is re-resolved immediately before apply, then clone names, four
removals, five data-input changes, both Concat axes, and every metadata target
are preflighted. One differential index, graph-ordered candidates, a 32-
rewrite ceiling, success-only pruning, and Session LayoutState synchronization
replace the full-map `while True` loop. The lowerer helper is now a 17-line
dispatcher and its sole production call supplies LayoutState.

Deep-skip focused coverage passed with `54 passed in 0.50s`; the combined
SiNet indexed suites passed with `261 passed in 0.79s`. Coverage includes
twenty-four FLOAT16/FLOAT32/FLOAT64 numerical-equivalence combinations across
both operand/Concat orders, both Resize families, scalar/raw constants, and
the explicit central adapter; four direct-NHWC correction/reference cases;
dynamic signatures; shared external constant cloning; candidate-only and
capped execution; twenty transactional unsafe contracts; explicit channel-
last selection; stale-plan revalidation; clone collision; no-index preflight;
differential-index validation; and LayoutState validation.

The sequential real SiNet integration passed with `1 passed in 2.51s`.
Legacy and indexed owners produced identical 449,824-byte float32 artifacts
with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and identical 253,452-byte float16 artifacts with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The sequential real-model `-cotof` run reported
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `184 passed in 49.18s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed with
`3 passed, 8 deselected in 3.66s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed pre-ADD fan-out checkpoint moves
`_optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains` to
`passes/sinet_preadd_fanout_layout.py`. Its distinct prefix is one
channel-last Concat followed by a private NHWC-to-NCHW adapter and one direct
NCHW input with exactly one earlier NCHW-to-NHWC sibling adapter. Those two
NCHW values feed the shared strict
`ADD -> MUL -> ADD -> PReLU -> Transpose -> Conv` tail. The PReLU also has one
or more later legacy NCHW consumers. The common terminal resolver now lives in
the existing SiNet residual owner, so downstream Conv ownership, affine
producer/consumer order, fused activation, and legacy fan-out semantics cannot
drift between the late-residual and pre-ADD variants.

The rewrite removes only the Concat-side adapter. The ADD consumes the
canonical NHWC Concat output and the already-existing sibling adapter output.
The affine/PReLU tail is lifted to NHWC and PReLU produces the canonical post
tensor directly. The terminal post adapter is inverted and produces the old
PReLU NCHW name, preserving all later legacy consumers and repeated input
slots. The direct sibling adapter is retained for its original branch. No
model name or fixed spatial dimension is used: relational rank-four concrete
shapes, dynamic signatures, logical/physical layout, typed permutations, and
the channel-last Concat axis prove the topology.

All activation tensors must share one unquantized FLOAT16/FLOAT32/FLOAT64
contract. Public intermediates, duplicate producers, unexpected direct-source
fan-out, a missing or late sibling adapter, invalid dependency order, wrong
Concat axis, non-Conv downstream ownership, or a missing/early legacy branch
reject the complete plan. The MUL, ADD, and PReLU constants must broadcast in
the original NCHW graph and after an explicit NHWC rotation. They are grouped
with the inverted terminal permutation; unrelated consumers receive one
deterministic clone. Candidate resolution is repeated before apply, followed
by collision, mutation-index, metadata-target, output-swap, and removal
preflight before the first write.

The focused pre-ADD suite passed with `47 passed in 0.47s`; combined with the
existing indexed SiNet residual and deep-skip suites it passed with
`308 passed in 0.92s`. Coverage includes twenty-four numerical-equivalence
combinations across FLOAT16/FLOAT32/FLOAT64, scalar/raw constants, both operand
and Concat orders, and both convolution families; idempotence; differential
index and LayoutState validation; shared permutation and external-constant
cloning; public repeated legacy slots; candidate-only and capped execution;
sixteen transactional unsafe contracts; earlier legacy rejection; stale-plan
revalidation; clone collision; and no-index preflight.

The sequential real SiNet integration passed with `1 passed in 2.55s`. The
current owner produced the same 449,824-byte float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`
as the preceding checkpoint. Therefore the already recorded sequential SiNet
`-cotof` evidence remains applicable: `max_abs=2.57205e-09`,
`rmse=9.15391e-11`, `cosine=1`, and `pass=True`. The full architecture suite
passed with `185 passed in 49.37s`; TensorFlow-import-blocked direct, default,
and `-cotof` conversion passed sequentially with
`3 passed, 8 deselected in 3.57s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed dual-Resize checkpoint moves both
`_optimize_sinet_dual_resize_affine_transpose_chains` and
`_optimize_sinet_deep_skip_dual_resize_affine_transpose_chains` to the shared
`passes/sinet_dual_resize_layout.py` owner. Both former raw mutators prove the
same two `Resize -> Transpose -> MUL -> ADD` branches, NCHW channel Concat,
residual ADD, terminal MUL/ADD/PReLU, and post-Transpose tail. Their only
semantic difference is now an explicit residual mode. The direct mode owns
and removes one private NHWC-to-NCHW adapter. The sibling mode requires one
earlier NCHW-to-NHWC adapter with exact residual fan-out and retains it for its
existing branch.

Each Resize output is the authoritative branch-local NHWC tensor. Its MUL and
ADD outputs adopt that exact concrete shape, dynamic signature, logical
layout, and physical layout. The two branches independently derive the
original axis-1 NCHW Concat and target axis-3 NHWC Concat. The first
graph-ordered post output is the authoritative merged contract and remains
unchanged; the Concat and terminal ADD/MUL/ADD intermediates adopt it. This
replaces the duplicated metadata permutation/copy sequences and removes the
former fixed 40-by-40 shape guard.

Four branch affine constants and three terminal affine/PReLU constants are
validated as finite same-dtype broadcasts in both the original NCHW graph and
the explicitly rotated NHWC graph. Identity-grouped roles are updated once,
and unrelated consumers receive deterministic clones. Both Resize families,
FLOAT16/FLOAT32/FLOAT64, scalar/raw constants, operand and Concat order,
typed permutations, producer uniqueness, exact consumer multiplicity,
dependency order, public boundaries, quantization, fused activation, and
dynamic signatures are part of the plan contract.

Equivalent post aliases are merged into the first canonical output while
downstream repeated slots are retained. Direct-mode legacy NCHW consumers
receive one topologically inserted inverse adapter and therefore preserve
local numerical semantics. Sibling mode retains the established ordered
pipeline boundary and rewires later compatibility consumers to the canonical
NHWC tensor for following SiNet recovery passes. The complete plan is
re-resolved before apply; clone names, mutation/removal indices, input slots,
metadata targets, alias rewrites, and optional adapter insertion are
preflighted before the first write. One differential index and a bounded
32-candidate rewrite ceiling replace both full-map `while True` loops. The
former 505-line and 503-line helpers are now 17-line dispatchers and both
production calls receive Session LayoutState.

The focused shared-owner suite passed with `102 passed in 0.60s`; combined
with the preceding indexed SiNet suites it passed with
`410 passed in 1.24s`. Coverage includes forty-eight numerical-equivalence
combinations across both residual modes, all three floating dtypes, both
Resize families, scalar/raw constants, and commuted operands/Concat inputs;
direct legacy inverse-adapter equivalence; sibling repeated-slot compatibility
rewiring; post-alias merging; external constant cloning; mode isolation;
candidate-only and capped execution; thirty-six shared transactional unsafe
contracts plus sibling fan-out/order guards; earlier legacy rejection;
stale-plan revalidation; clone collision; no-index preflight; differential
index validation; and LayoutState validation.

The existing direct dual-Resize characterization and the sequential real
SiNet integration passed together with `2 passed in 2.59s`. During the real
conversion, the direct owner retained the established two one-candidate
applications and the later sibling owner retained its two-candidate
application. The indexed result is the same 449,824-byte float32 artifact with
SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `186 passed in 50.01s`;
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed, 8 deselected in 3.61s`. Scoped Ruff, syntax
compilation, and `git diff --check` passed. No Tier corpus conversion was run.

The indexed shared-post checkpoint moves
`_optimize_sinet_shared_post_prelu_transpose_fanout_chains` to
`passes/sinet_shared_post_layout.py`. The complete candidate is two private
NHWC-to-NCHW adapters feeding an ADD/MUL/ADD/PReLU tail and one terminal
NCHW-to-NHWC adapter. Exactly one source must be produced by a plain
channel-last Concat. The canonical post tensor must be consumed as the data
input of at least one Conv or DepthwiseConv and by at least one plain ADD;
unsupported consumer roles reject the plan.

The canonical post tensor supplies the exact concrete NHWC shape, dynamic
signature, logical layout, and physical layout. Both input sources must match
that contract, and every internal NCHW tensor must match its exact
permutation. The rewrite removes the two input adapters and terminal adapter,
connects the first ADD directly to both NHWC sources, and makes PReLU produce
the existing post tensor. Consumer names and repeated ADD slots are preserved,
and only the three affine intermediates adopt canonical metadata. This removes
the former 40-by-40 heuristic without broadening the semantic topology.

Typed permutation constants, unique producers, exact fan-out and dependency
order, public boundaries, fused activation, dtype, quantization, layout, and
Concat/Conv input roles are proven before mutation. FLOAT16/FLOAT32/FLOAT64
MUL, ADD, and PReLU constants must broadcast in NCHW and after explicit NHWC
rotation. Shared roles are grouped by tensor identity, and unrelated consumers
receive deterministic clones. Candidate resolution is repeated immediately
before apply, followed by collision, removal-index, consumer-index, input,
output, and metadata-target preflight. One differential graph index, ordered
candidates, a 32-rewrite ceiling, success-only pruning, and LayoutState
synchronization replace the full-map fixed-point loop.

Focused shared-post coverage passed with `46 passed in 0.48s`; combined with
the preceding indexed SiNet suites it passed with `456 passed in 1.27s`.
Coverage includes twenty-four numerical-equivalence combinations across all
three floating dtypes, scalar/raw constants, both operand and Concat orders,
and both convolution families; non-40 spatial shapes and dynamic signatures;
idempotence; repeated post-ADD slots; external constant cloning; candidate-only
and capped execution; seventeen transactional unsafe contracts; stale-plan
revalidation; no-index preflight; differential-index validation; and
LayoutState validation.

The sequential real SiNet integration passed. The indexed result remains the
same 449,824-byte float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `187 passed in 48.16s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.64s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed Concat/Resize affine checkpoint moves
`_optimize_sinet_concat_resize_affine_transpose_chains` to
`passes/sinet_concat_resize_layout.py`. Its three input adapters carry the
merged residual source, an independent Concat branch, and a bilinear or
nearest-neighbor Resize result into NCHW. The Resize branch has a private
MUL/ADD affine tail. Both branch results feed a plain axis-1 Concat followed by
the residual ADD/MUL/ADD/PReLU tail and one or more NCHW-to-NHWC post adapters.

The Resize output is authoritative for the branch-local NHWC shape, dynamic
signature, and layout. The first graph-ordered post output is authoritative
for the merged NHWC contract. Independent and affine branch contracts derive
both the original NCHW axis-1 Concat and target NHWC axis-3 Concat; the
residual source and tail must match those merged contracts exactly. The
rewrite removes the three input adapters and every equivalent post adapter,
changes the Concat axis to 3, reconnects the two canonical branch sources,
and makes PReLU produce the canonical post tensor directly.

Additional post outputs are merged into the first canonical tensor while all
repeated downstream slots remain intact. Later consumers of the former PReLU
NCHW tensor receive one inverse adapter inserted before their first use.
FLOAT16/FLOAT32/FLOAT64 activation tensors must be unquantized and share the
same dtype. Typed permutations, Resize provenance, unique producers, exact
fan-out, dependency order, public boundaries, fused activation, shape,
signature, and layout are validated before mutation. Two branch constants and
three merged-tail constants must broadcast in the original NCHW graph and
after explicit NHWC rotation; unrelated consumers receive deterministic
clones.

The complete plan is resolved again immediately before apply, followed by
clone-name, mutation/removal-index, alias-slot, metadata-target, and legacy-
adapter preflight. One differential graph index, graph-ordered candidates, a
32-rewrite ceiling, success-only pruning, and LayoutState synchronization
replace the full-map fixed-point loop. Both existing production call sites
retain their exact order and now supply Session LayoutState.

Focused Concat/Resize coverage passed with `58 passed in 0.50s`; combined with
the preceding indexed SiNet suites it passed with `514 passed in 1.30s`.
Coverage includes twenty-four numerical-equivalence combinations across all
three floating dtypes, scalar/raw constants, both operand and Concat orders,
and both Resize families; non-40 spatial shapes and dynamic signatures;
idempotence; post-alias merging; repeated alias and legacy slots; no-legacy
operation; external constant cloning; candidate-only and capped execution;
twenty-seven transactional unsafe contracts; stale-plan revalidation; clone
collision; no-index preflight; differential-index validation; and LayoutState
validation.

The sequential real SiNet integration passed. The indexed result remains the
same 449,824-byte float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the same 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `188 passed in 48.25s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.68s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed two-Concat tail checkpoint moves
`_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains` to
`passes/sinet_tail_concat_layout.py`. It reuses the exact adapter and
Resize-affine branch resolvers from the preceding owner. The first stage
combines an independent branch and one Resize/MUL/ADD branch with a same-width
residual, then applies MUL/ADD/PReLU. A second channel Concat combines that
result with an independent skip adapter before the final MUL/ADD/PReLU and
post adapter.

The first residual source supplies the authoritative NHWC contract for the
first stage, while the first graph-ordered post output supplies the merged
second-stage contract. Both original NCHW axis-1 Concats and target NHWC
axis-3 Concats are derived from the concrete branch shapes and dynamic
signatures. The rewrite removes four input adapters and every equivalent post
adapter, reconnects the canonical sources, changes both axes to 3, and makes
the final PReLU produce the canonical post tensor directly.

Post aliases retain every repeated downstream slot. Later consumers of the
former final NCHW PReLU output receive one inverse adapter inserted before
their first use. Eight constants are grouped across three orientation domains:
two Resize-branch constants, three first-stage MUL/ADD/PReLU constants, and
three final-stage constants. Each must be a finite same-dtype broadcast before
and after explicit NHWC rotation; unrelated consumers receive deterministic
clones. Typed permutations, exact Resize provenance, producer uniqueness,
consumer multiplicity, dependency order, public boundaries, fused activation,
FLOAT16/FLOAT32/FLOAT64 dtype, quantization, layout, rank-four shape, and
signature are validated before mutation.

The complete plan is re-resolved before apply, followed by clone-name,
mutation/removal-index, alias-slot, metadata-target, and legacy-adapter
preflight. One differential graph index, graph-ordered candidates, a
32-rewrite ceiling, success-only pruning, and LayoutState synchronization
replace the full-map fixed-point loop. The sole production call retains its
ordered recovery position and now supplies Session LayoutState.

Focused two-Concat coverage passed with `52 passed in 0.50s`; combined with
the preceding indexed SiNet suites it passed with `566 passed in 1.42s`.
Coverage includes twenty-four numerical-equivalence combinations across all
three floating dtypes, scalar/raw constants, both operand/Concat orders, and
both Resize families; non-40 spatial shapes and dynamic signatures;
idempotence; post aliases; repeated alias and legacy slots; no-legacy
operation; external constant cloning; candidate-only and capped execution;
twenty-three transactional unsafe contracts; stale-plan revalidation;
no-index preflight; differential-index validation; and LayoutState validation.

The old and indexed owners both match zero candidates in the current real
SiNet ordered pipeline; this dormant-path fact was measured before extraction.
The sequential real integration nevertheless remained byte-identical: the
449,824-byte float32 artifact has SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`,
and the 253,452-byte float16 artifact has SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `189 passed in 48.07s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.67s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed Softmax-mask checkpoint moves
`_optimize_sinet_softmax_mask_residual_nhwc_tail_chains` to
`passes/sinet_softmax_mask_layout.py`. The active real-model island has two
private NHWC-to-NCHW input adapters. Its main MUL/ADD result enters a
`[0,3,2,1] -> Softmax(axis=3) -> [0,3,2,1]` wrapper, then
ReduceMax/SUB/Reshape/MUL forms the mask applied to the side PReLU branch
before the final residual ADD and post adapter.

The first graph-ordered post output supplies the authoritative NHWC contract.
All NCHW, NWHC, reduced-rank, and singleton-channel mask shapes and dynamic
signatures are derived from it. The rewrite removes both input adapters, both
Softmax wrapper adapters, and every equivalent terminal adapter. Softmax
directly consumes the main NHWC affine result and produces the former
soft-back tensor, ReduceMax axis `[1]` becomes `[3]`, and Reshape target
`[N,1,H,W]` becomes `[N,H,W,1]` without a fixed spatial size.

The main MUL/ADD constants, side PReLU alpha, and mask expansion constant must
be finite same-dtype broadcasts both before and after explicit rotation. The
ReduceMax axis and Reshape target constants preserve their original INT32 or
INT64 dtype. All six transformed roles are grouped by identity, so unrelated
consumers receive deterministic clones and conflicting shared roles reject
the plan. The SUB singleton is strictly validated but left unchanged.

Equivalent post aliases preserve every repeated downstream input slot. Later
consumers of the former NCHW residual receive one inverse adapter inserted
before their first use. Typed permutations, producer uniqueness, exact fan-
out, dependency order, public boundaries, fused activation, Softmax beta,
ReduceMax keep-dims, FLOAT16/FLOAT32/FLOAT64 dtype, quantization, layout,
shape, and signature are all checked before mutation. The complete plan is
re-resolved before apply, and clone names, mutation/removal indices, reshape
options, alias slots, metadata targets, and legacy-adapter insertion are
preflighted before the first write.

One differential graph index, graph-ordered candidates, a 32-rewrite ceiling,
success-only pruning, and LayoutState synchronization replace the full-map
fixed-point loop. The sole production call retains its ordered recovery
position and now supplies Session LayoutState. Focused coverage passed with
`50 passed in 0.53s`; combined with all preceding indexed SiNet suites it
passed with `616 passed in 1.53s`. Coverage includes eighteen numerical-
equivalence combinations over all three floating dtypes,
scalar/channel/full constants, both operand orders, dynamic signatures,
idempotence, post aliases, repeated
alias and legacy slots, no-legacy operation, external float and integer
constant cloning, candidate-only and capped execution, twenty-five unsafe
contracts, stale-plan and clone-collision revalidation, no-index preflight,
differential-index validation, and LayoutState validation.

The sequential production integration applied this owner once in each of the
float32 and float16 ModelIR builds. The output remains byte-identical: the
449,824-byte float32 artifact has SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`,
and the 253,452-byte float16 artifact has SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
This preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
The full architecture suite passed with `190 passed in 48.85s`, and
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.80s`. Scoped Ruff, syntax compilation, and
`git diff --check` passed. No Tier corpus conversion was run.

The indexed mix-attention checkpoint moves
`_optimize_sinet_mix_attention_double_logistic_nhwc_chains` to
`passes/sinet_mix_attention_layout.py`. A pre-extraction production audit ran
all six ordered invocations against `sinet_320_op.onnx`; the legacy owner
matched zero candidates in every invocation. The path is therefore retained
as a compatibility topology rather than used to tune the active model.

The new resolver starts at the terminal NCHW-to-NHWC adapter and proves its
sole post-Conv owner. It then resolves the two Logistic gates and the
`gate * branch + (1 - gate) * residual` tail back to the shared source ADD.
The branch is one private NHWC-to-NCHW adapter. The residual may be either a
private ADD of two such adapters or a single private adapter, preserving both
forms supported by the former helper. Binary and Concat operand order is
semantic, not positional.

The channel-attention branch must contain Mean, NCHW-to-NHWC, two Conv
operators, and NHWC-to-NCHW. The spatial-attention branch must contain Mean
and ReduceMax, a channel Concat, MirrorPad, NCHW-to-NHWC, Conv, and Reshape.
The position-attention merge must contain the two rank-five Reshapes, Concat,
rank-four Reshape, MirrorPad, NCHW-to-NHWC, Conv, and NHWC-to-NCHW. Exact
producer uniqueness, consumer multiplicity, graph order, public boundaries,
plain fused activations, floating dtype, quantization, layout, shape, and
dynamic signature contracts are checked for the whole island.

Reduce axes, both MirrorPad pair tensors, the two rank-five targets, the
rank-four target, and the rank-five Concat axis preserve INT32 or INT64 dtype
while being remapped to their NHWC contracts. Shared constants are grouped by
identity. An unrelated consumer receives a deterministic clone; incompatible
shared roles reject the plan. Metadata for rank-four tensors is derived from
the canonical terminal NHWC output, while transformed rank-five tensors are
explicitly layout-unknown rather than assigned a rank-four label.

The complete plan is resolved again immediately before apply. Constant clone
names, operator indices, input slots, option changes, metadata targets, and
all eight adapters in direct-residual mode or nine adapters in ADD-residual
mode are preflighted before the first write. The branch and residual tail are
rewired to their canonical NHWC sources. This deliberately fixes the former
helper's unreachable-topology defect that removed the branch adapter while
rewiring its consumers back to the removed adapter output, which could have
left an unbound tensor.

One differential `ModelIRGraphIndex`, graph-order candidates, a configurable
32-rewrite ceiling, success-only pruning, and Session `LayoutState`
synchronization replace the full-map `while True` loop. Both production calls
retain their ordered positions and now supply the Session LayoutState. No
model name or fixed spatial size participates in matching.

Focused coverage passed with `45 passed in 0.49s`; all indexed SiNet suites
passed with `661 passed in 1.67s`. Coverage spans FLOAT16/FLOAT32/FLOAT64,
both residual forms, reversed binary and Concat inputs, dynamic signatures,
shared and externally consumed constants, candidate-only and capped
execution, twenty-eight unsafe mutations, stale-plan revalidation, clone-name
collision, no-index preflight, differential-index validation, and LayoutState
validation. The full architecture suite, run together with the focused owner,
passed with `236 passed in 46.85s`, including `191` architecture tests.
TensorFlow-import-blocked direct, default, and `-cotof` conversion passed
sequentially with `3 passed in 3.66s`.

The sequential real SiNet integration smoke passed with `1 passed in 2.79s`.
A separate sequential CLI conversion produced the unchanged 449,824-byte
float32 artifact with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the unchanged 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed SA/PA MirrorPad checkpoint moves
`_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains` to
`passes/sinet_sa_pa_mirrorpad_layout.py`. A pre-extraction production audit
ran all seven ordered invocations against `sinet_320_op.onnx`; the legacy
owner matched zero candidates in every invocation. The replacement is
therefore a compatibility owner and does not use the current SiNet artifact to
broaden its guards.

The resolver starts at one private NHWC-to-NCHW source adapter. Its exact
fan-out is Mean, ReduceMax, and one source Reshape, plus the terminal Mul only
for the legacy NCHW-output form. Mean and ReduceMax converge at a channel
Concat followed by MirrorPad, NCHW-to-NHWC, Conv, and the removable singleton-
channel Reshape. The second ADD input must be one external NHWC channel-
attention value behind its own private adapter. The ADD result and original
source are expanded to rank five, concatenated, reshaped to rank four, padded,
and passed through NCHW-to-NHWC and Conv before Logistic and Mul.

The terminal Mul supports two historical forms. A direct NHWC source requires
an already-NHWC Logistic gate and retains its output name. A legacy NCHW source
requires the matching gate adapter; the owner rewires Mul to NHWC, gives it a
deterministic private NHWC output, and inserts one inverse adapter immediately
after Mul to preserve the original NCHW output name, public output, and all
downstream consumers.

The rewrite explicitly requires a concrete singleton source channel. Removing
the SA Reshape and channel-attention adapter is generally value-preserving only
under that condition; the former raw helper did not state this invariant. A
non-singleton candidate is now a transactional no-op instead of a potentially
incorrect layout rewrite. Concrete shape and dynamic signature relationships
are derived without a fixed spatial size.

Both reduction axes, both MirrorPad tensors, both rank-five expansion targets,
and the final rank-four target preserve INT32 or INT64 dtype while being
remapped to NHWC. Shared constants are grouped by identity. Unrelated users
receive deterministic clones; conflicting roles reject the plan. Rank-four
metadata follows the proven NHWC source contract, and rank-five intermediates
are explicitly layout-unknown.

The complete plan is resolved again immediately before apply. Constant clone
names, source provenance, unique producers, exact consumers, dependency order,
public boundaries, input slots, option changes, metadata targets, legacy
output names, and all five or six removals are preflighted before the first
write. One differential `ModelIRGraphIndex`, graph-ordered candidates, a
configurable 32-rewrite ceiling, success-only pruning, and Session
`LayoutState` synchronization replace the full-map `while True` loop. All
three production calls retain their ordered positions and now supply the
Session LayoutState.

Focused coverage passed with `42 passed in 0.47s`; all indexed SiNet suites
passed with `703 passed in 1.76s`. Coverage spans FLOAT16/FLOAT32/FLOAT64,
direct-NHWC and legacy-NCHW Mul forms, reversed binary and Concat inputs,
dynamic signatures, external constant cloning, candidate-only and capped
execution, twenty-five unsafe topology/provenance/layout/public-boundary
contracts, stale-plan revalidation, legacy-name collision, differential-index
validation, and LayoutState validation. The full architecture suite, run with
the focused owner, passed with `234 passed in 47.46s`, including `192`
architecture tests. TensorFlow-import-blocked direct, default, and `-cotof`
conversion passed sequentially with `3 passed in 3.70s`.

The sequential production conversion confirmed seven zero-match indexed
invocations and produced the unchanged 449,824-byte float32 artifact with
SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`
and the unchanged 253,452-byte float16 artifact with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`.
The byte identity preserves the recorded sequential `-cotof` evidence:
`max_abs=2.57205e-09`, `rmse=9.15391e-11`, `cosine=1`, and `pass=True`.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed transpose/binary bridge checkpoint moves
`_optimize_transpose_binary_bridges` to
`passes/binary_bridge_layout.py`. A pre-extraction sequential audit ran the
legacy helper once on each of `face_detection_yunet_2023mar.onnx`,
`FastestDet.onnx`, `human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`. Both the symmetric and
asymmetric counters were zero for all five models. QDQ graphs continue to skip
this production call exactly as before.

The new owner retains the active semantic families only. Symmetric mode
supports one inverse post with no extra fan-out, one inverse post plus later
legacy-layout consumers, and no inverse post plus a synthesized adapter before
the first legacy consumer. Asymmetric mode moves the inverse permutation to
the plain operand and preserves the original operand order. The permanently
disabled Pattern C body was removed.

Every candidate now requires a plain binary operation, typed INT32/INT64
permutation constants, exact inverses, unique producers, resolved sources,
private intermediate tensors, dependency order, public-output preservation,
same dtype, per-tensor quantization, static broadcast consistency, and dynamic
signature consistency. Mixed fan-out no longer mutates the post permutation
buffer; it references the already-proven pre permutation. A complete plan is
resolved again immediately before apply. This removes the former no-post
partial-mutation bug and rejects asymmetric rewrites whose replacement plain
source is not available before the reused Transpose.

Focused owner and existing direct-lowering coverage passed together with
`42 passed, 735 deselected in 1.12s`. The new 22-test owner suite covers all
four binary operations in all three symmetric modes, exact numerical
equivalence, asymmetric SUB/DIV on both operand sides, idempotence,
candidate-only and zero-limit operation, per-axis quantization, fused
activation, public boundary, duplicate producer, late-source rejection,
differential-index freshness, invariant validation, and LayoutState
synchronization. The complete architecture suite passed with
`193 passed in 44.80s`. TensorFlow-import-blocked direct, default, and `-cotof`
conversion passed sequentially with `3 passed in 3.84s`.

One sequential YuNet conversion was run both from preceding commit `c09b8b88`
and from the current implementation. Every emitted file was byte-identical.
The 236,564-byte float32 artifact has SHA-256
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
the 131,120-byte float16 artifact has SHA-256
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
and the 105,578-byte correspondence report has SHA-256
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
The temporary detached worktree and both output directories were removed.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed safe binary recovery checkpoint replaces the five helpers called
by `_run_safe_binary_bridge_recovery_sequence`. Before extraction, each helper
was observed four times per model in the ordered pipelines for
`face_detection_yunet_2023mar.onnx`, `FastestDet.onnx`,
`human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`. Asymmetric fan-out,
full-post fan-out, single-post, and mixed fan-out returned zero in all twenty
calls each. Legacy-only returned zero except for the first SiNet invocation,
where it rewrote exactly `Add_52` and `Add_109`.

The five modes now run through one indexed owner in their unchanged order.
Legacy-only and single-post reuse the strict symmetric plan while retaining
the late `__preserve_layout_boundary__` contract. Mixed and full-post fan-out
share a multi-post plan: the first post output is canonical, later aliases are
rewired through indexed consumers, and mixed mode keeps one verified adapter
before all legacy users. Its permutation input changes to the proven pre-
permutation tensor rather than overwriting the potentially shared inverse
constant. Asymmetric fan-out retains its separate existing-plain-Transpose
contract and rejects that Transpose when it would occur after the binary.

Typed permutations, source provenance, unique producers, exact consumer
multiplicity, graph order, public boundaries, fused activation, dtype,
per-tensor quantization, static broadcast, dynamic signatures, options,
metadata, aliases, and removal/insertion indices are captured in immutable
plans and re-resolved before apply. Each phase has a configurable 32-rewrite
ceiling. One `ModelIRGraphIndex` is reused across all selected phases, and
pruning plus Session LayoutState synchronization run once afterward. Five
lowerer compatibility helpers remain as 17-line dispatchers; production uses
the single ordered entry point.

The two focused binary owners and the complete architecture suite passed
together with `234 passed in 44.37s`, comprising 22 general bridge tests, 19
safe recovery tests, and 193 architecture tests. The safe recovery suite
covers all four binary operations for mixed/full multi-post fan-out,
order-sensitive asymmetric SUB/DIV on both sides, phase stats, candidate and
rewrite limits, retained-boundary markers, shared inverse constants,
differential-index freshness, LayoutState, public/order rejection, per-axis
quantization, idempotence, and exact numerical equivalence. The selected 23
adjacent direct-builder tests passed with `23 passed, 732 deselected in 1.21s`,
including the four no-post boundary regressions. TensorFlow-import-blocked
direct, default, and `-cotof` conversion
passed sequentially with `3 passed in 3.75s`.

The indexed production sequence still reports two legacy-only rewrites in the
first SiNet invocation and zero for every other phase/invocation. Sequential
conversion from preceding commit `53381952` and the current implementation
emitted byte-identical files: float32 remains 449,824 bytes with SHA-256
`40520abec7b36dae10dca3cd5271bf5169d096eea52f726f2023238694afa9bb`,
float16 remains 253,452 bytes with SHA-256
`180717a7e13963f4c1ab56dcb82288562ecf718e4a3a36738bbabc7fa9c0082c`,
and the 182,687-byte correspondence report remains
`24c423ea51b26b178d3764be027855e797bbf9b5ba1930810d2e1dbe281d8e25`.
The schema outputs also match. The detached worktree and temporary output
directories were removed. No Tier corpus conversion was run.

The indexed direct Split-root checkpoint replaces the former mixed
match/mutate/fixed-point helper with a separate direct-root plan and the common
closed-tail resolver in `passes/split_channelwise_layout.py`. Pre-extraction
characterization recorded zero matches in every invocation: four each on
YuNet, FastestDet, HumanSeg, and OSNet, and eight on SiNet. The helper remains
in both unchanged ordered production positions as a thin dispatcher and now
receives Session LayoutState.

The direct-root plan validates a typed private `[0,3,1,2]` Transpose, a sole
channel Split consumer, Split-axis copy-on-write, unique producers, graph
order, rank-four shape and dynamic-signature relationships, dtype,
per-tensor quantization, and the complete closed tail. Only root Split outputs
seed the closure. Unary, binary, Concat, downstream Split, and exact rank-four
Slice operations may propagate NHWC; unsupported consumers and dead branches
reject the candidate.

Slice begin/size constants are planned without mutation, validated against the
old and new layout, and grouped by tensor identity. Exclusive same-value uses
may update in place. Any unrelated consumer causes one deterministic shared
clone for all planned slots, retaining actual INT32 or INT64 NumPy dtype.
Conflicting roles, variables, producer-backed/public/missing constants, bounds
errors, dtype mismatches, public intermediates, duplicate producers, and stale
consumer order are transactional no-ops.

The new direct suite passed with `20 passed`; it and the binary-root suite
passed together with `53 passed in 0.51s`. The two indexed binary bridge
suites, both Split-root suites, and all architecture tests passed with
`288 passed in 44.74s`. Coverage includes exact semantics for INT32 and INT64
constants with static and dynamic signatures, shared Split and Slice constant
cloning, an unrelated source consumer, candidate and rewrite limits,
idempotence, differential graph-index freshness, LayoutState validation, and
thirteen unsafe no-op cases. TensorFlow-import-blocked explicit direct,
default direct, and `-cotof` conversion passed sequentially with `3 passed in
3.85s`.

One sequential YuNet conversion was run from source checkpoint `5f8e662a` and
from the current implementation. All five emitted files were byte-identical.
Float32 remains
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16 remains
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
and the correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
Both schema hashes also match. The detached worktree and temporary outputs were
removed. Scoped Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed unary/Split/Concat checkpoint replaces the former raw helper with
a separate unary-root plan in `passes/split_channelwise_layout.py`.
Pre-extraction characterization observed zero matches in all 24 sequential
runtime invocations: four each on YuNet, FastestDet, HumanSeg, and OSNet, and
eight on SiNet. The two existing production positions and their ordering before
the direct and binary Split roots are unchanged, and both now receive Session
LayoutState.

The owner validates the private typed Transpose, allowed pre-unary, channel
Split, exact one-to-one coverage of every Split output, optional branch
unaries, exactly one external Concat branch, channel Concat, unique producers,
resolved provenance, graph order, public boundaries, dtype, per-tensor
quantization, static shapes, and dynamic signatures. Every converted branch
has exact consumers through the Concat. The local terminal adapter preserves
the original NCHW Concat tensor for either a public output or later consumers.

The external layout-only Reshape now requires a real singleton channel and an
exact NHWC-to-NCHW shape/signature relation. A proven direct NHWC external
branch is also accepted. Shared Reshape or raw-source consumers remain
untouched. The full plan is re-resolved and preflighted before changing the
pre-unary input, Split axis, branch metadata, Concat inputs/options, or output
adapter. This specifically eliminates the raw partial-mutation failure when
the Concat output tensor is absent or invalid.

The focused suite passed with `29 passed in 0.44s`. It covers INT32/INT64
axes, static and dynamic signatures, public and intermediate NCHW boundaries,
exact numerical equivalence, a direct NHWC external branch, shared axis and
side-consumer preservation, candidate-only and capped execution, idempotence,
differential-index freshness, LayoutState validation, and eighteen unsafe
transactional no-op cases. The active legacy fixture was corrected to declare
its previously producerless external tensor as a graph input.

The new owner, existing active fixture, both preceding Split-root owners, both
binary bridge owners, and the complete architecture suite passed with
`318 passed in 43.96s`. TensorFlow-import-blocked explicit direct, default
direct, and `-cotof` conversion passed sequentially with `3 passed in 3.83s`.
One sequential YuNet conversion from source checkpoint `0b4a0001` and the
current implementation emitted five byte-identical files. Float32 remains
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16 remains
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
and correspondence remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`.
Both schema files also match. The detached worktree and temporary outputs were
removed. Scoped Ruff, syntax compilation, and `git diff --check` passed. No
Tier corpus conversion was run.

The indexed singleton-gate checkpoint replaces the former raw gate/Conv/
Concat helper with `passes/singleton_gate_layout.py`. Pre-extraction
characterization observed zero matches in all 24 sequential runtime
invocations: four each on YuNet, FastestDet, HumanSeg, and OSNet, and eight on
SiNet. The helper remains at both unchanged production sequence positions as a
thin dispatcher and now receives Session LayoutState.

The owner proves the exact clip adapter, gate multiply, scalar subtraction,
direct or Logistic auxiliary branch, signal multiply, Add, allowed terminal
unary, output adapter, and channel-last Concat before mutation. Every removed
adapter is a true singleton-channel NHWC/NCHW view with compatible static and
dynamic shapes, dtype, per-tensor quantization, typed immutable Reshape shape
input when present, unique producer, resolved provenance, and valid graph
order. Public intermediates, fused activations, duplicate producers, stale
order, unsupported fan-out, non-singleton channels, per-axis quantization, and
invalid broadcast contracts are transactional no-ops.

The optional RGB branch accepts either a fully owned typed `[0,3,1,2]`
Transpose or a private constant whose physical NHWC evidence proves that its
NCHW metadata is stale. The latter constant's data buffer is reshaped together
with its metadata. View-equivalent split and terminal adapter consumers are
rewired as a group, while unrelated source and side consumers remain intact.
All input rewrites, tensor/operator contracts, metadata updates, and removals
are captured in an immutable plan and fully re-resolved immediately before
apply through one differential graph index. Graph-ordered candidates, a
configurable 32-rewrite ceiling, success-only pruning, and LayoutState updates
replace the unbounded full-map loop.

The focused suite passes with `29 passed in 0.47s` and covers both auxiliary
forms across static/dynamic and RGB/no-RGB modes, direct stale-layout RGB,
shared side consumers, exact numerical equivalence, candidate-only and capped
execution, idempotence, differential-index freshness, LayoutState validation,
and nineteen unsafe no-op cases. The dedicated owner, two corrected active
fixtures, the unary/direct/binary Split roots, both binary bridge owners, and
the complete architecture suite pass together with `349 passed, 753
deselected in 43.88s`. TensorFlow-import-blocked explicit direct, default
direct, and `-cotof` conversion pass sequentially with `3 passed in 3.80s`.

One sequential YuNet conversion was run from source checkpoint `12680f75` and
from the current implementation. All five emitted files are byte-identical.
Float32 remains
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16 remains
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
the correspondence report remains
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
and both schema hashes also match. The detached worktree and all temporary
outputs were removed. Scoped Ruff, syntax compilation, and `git diff --check`
passed. No Tier corpus conversion was run.

The indexed Conv-output passthrough checkpoint replaces the former 316-line
raw `_optimize_transposeconv_output_nhwc_passthrough_chains` implementation
with `passes/conv_output_passthrough_layout.py`. Its one production position
and phase order before the channel-one terminal helper are unchanged, and the
dispatcher now receives Session LayoutState.

Pre-extraction characterization ran the two adjacent helpers sequentially on
the five short representatives. Each helper was invoked four times per model.
Only the first general-passthrough invocation was active: YuNet rewrote 10,
FastestDet 23, HumanSeg 27, OSNet 63, and SiNet zero chains. All 123 active
chains were Conv2D or DepthwiseConv2D followed by RELU. The distinct
channel-one TransposeConv/Squeeze terminal helper returned zero in all 20
invocations and remains a separate semantic family.

The new owner proves a typed private NHWC-to-NCHW Transpose produced by
Conv2D, DepthwiseConv2D, or TransposeConv, a strictly linear nonempty chain,
and one typed inverse Transpose with an intermediate NHWC output. It preserves
all seven historical unary operations and all six binary operations without
changing operand slots. Every tensor, producer, consumer, graph-order,
static/dynamic shape, dtype, quantization, public-boundary, and post-output
contract is resolved before mutation.

Rank-four binary constants require matching immutable data and TensorIR
metadata plus valid NCHW and NHWC broadcasts. Exclusive constants change in
place. Shared constants receive one deterministic NHWC clone reused by every
planned chain slot, while unrelated legacy consumers retain the original.
Scalar constants remain unchanged. The surviving post-adapter output metadata
is derived from the converted final chain output rather than blindly permuting
possibly stale existing metadata.

All input/output rewrites, constant changes, metadata, tensor/operator
contracts, and adapter removals are immutable plan state and are re-resolved
immediately before apply through one differential graph index. A current-
candidate bound and optional explicit rewrite limit replace the full-map
unbounded fixed-point loop. LayoutState is synchronized differentially and
pruning runs only after success.

The new focused suite passes with `56 passed in 0.51s`. It covers all three
producer types, all unary and binary operations, both operand positions, exact
SUB/DIV and other binary numerical behavior, dynamic signatures, grouped
shared constants, candidate/limit/idempotence behavior, differential index and
LayoutState, and twenty unsafe transactional no-op cases. The new owner,
singleton-gate and three Split-root owners, both binary bridge owners, existing
active fixtures, and complete architecture suite pass together with `408
passed, 751 deselected in 44.32s`. TensorFlow-import-blocked explicit direct,
default direct, and `-cotof` conversion pass sequentially with `3 passed in
3.91s`.

Sequential conversions from source checkpoint `410b86e6` and the current
implementation produced byte-identical float32, float16, correspondence, and
both schema files for every active representative: YuNet, FastestDet,
HumanSeg, and OSNet. The temporary detached worktree and outputs were removed.
Scoped Ruff, syntax compilation, and `git diff --check` passed. No Tier corpus
conversion was run.

The indexed channel-one terminal checkpoint replaces the remaining 390-line
raw `_optimize_transposeconv_output_channel1_terminal_transpose_chains`
implementation with a second immutable plan in
`passes/conv_output_passthrough_layout.py`. The helper remains immediately
after the general passthrough owner in the unchanged attention/gate recovery
sequence and now receives Session LayoutState.

The terminal resolver requires a TransposeConv output whose static and dynamic
NHWC channel is exactly one, one private typed NHWC-to-NCHW adapter, a strictly
linear unary/binary chain containing exactly one Squeeze, and a final graph
output with no consumer. It preserves all fourteen historical unary operations
and all six binary operations without reordering operands. Public
intermediates, a second Squeeze, stale order, fan-out, invalid quantization,
missing tensors, or a nonterminal output reject the complete candidate.

Explicit, negative, and implicit Squeeze axes are normalized in NCHW and
remapped to NHWC by semantic label. Every squeezed static and dynamic
dimension must equal one. The surviving labels must occur in the same order in
both paths, preventing a spatial-only Squeeze from exposing a reordered public
rank-three tensor. Metadata and LayoutState use NHWC/NWC only when the
surviving labels support that annotation; channel-free outputs remain
explicitly unknown rather than receiving a misleading layout.

Non-scalar binary constants are accepted only before Squeeze and reuse the
general owner's grouped rank-four constant planner. A post-Squeeze binary must
use a scalar constant. All chain inputs, Squeeze options, tensor/operator
contracts, constant updates, output metadata, and the one adapter removal are
re-resolved before apply through the differential graph index. Candidate count
and an optional limit bound execution; pruning occurs only after success.

The terminal suite passes with `57 passed in 0.69s`, covering explicit,
negative, and implicit axes, static/dynamic signatures, all unary and binary
operations, both operand positions, numerical equivalence, post-Squeeze scalar
binary, shared constants, candidate limits, idempotence, GraphIndex,
LayoutState, and twenty-two unsafe no-op cases. It and the preceding indexed
owners, active fixtures, and complete architecture suite pass together with
`465 passed, 751 deselected in 43.74s`. TensorFlow-import-blocked explicit
direct, default direct, and `-cotof` conversion pass sequentially with `3
passed in 3.95s`.

YuNet retained four zero-match terminal invocations. A sequential comparison
against source checkpoint `ac8c4cf7` emitted the same five byte-identical
files: float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
and both unchanged schema artifacts. The first comparison-harness attempt
omitted its output-directory environment variable and exited before current-
branch conversion; the corrected run passed. The temporary worktree and all
outputs were removed. No Tier corpus conversion was run.

The RELU/Split all-output checkpoint passed the dedicated owner and active
compatibility fixture first:

```text
uv run pytest -q \
  tests/test_flatbuffer_direct_indexed_split_all_outputs_layout.py \
  tests/test_tflite_builder_direct.py::test_flatbuffer_direct_transpose_relu_split_all_outputs_to_nhwc_chains_optimized

32 passed in 2.92s
```

The new owner, adjacent binary/Split, direct Split, unary/Split/Concat,
singleton-gate, Conv-output, terminal TransposeConv, and binary-bridge owners,
the complete architecture suite, and the active fixture then passed together:

```text
475 passed in 46.51s
```

TensorFlow import blocking was exercised separately and sequentially for the
explicit direct backend, default direct backend, and direct `-cotof` path:

```text
3 passed in 4.01s
```

Scoped Ruff passed for the new owner, dedicated tests, and architecture test.
All changed Python files passed `py_compile`; the lowerer and legacy direct
fixture passed Ruff with only their documented inherited F841/F401 categories
ignored. `git diff --check` passed. One sequential YuNet conversion reproduced
the five hashes recorded in the continuation snapshot. No Tier corpus was run.

The exact Split/Conv/Concat checkpoint passed its 49 dedicated cases and the
corrected active compatibility fixture with `50 passed in 0.61s`. The new
plan, all-output plan, adjacent binary/Split, direct Split,
unary/Split/Concat, singleton-gate, Conv-output, terminal TransposeConv, and
binary-bridge owners, both active fixtures, and the complete architecture
suite then passed together:

```text
525 passed in 47.51s
```

TensorFlow import blocking was again exercised sequentially for explicit
direct, default direct, and direct `-cotof` conversion:

```text
3 passed in 4.33s
```

One sequential YuNet conversion reproduced all five fixed hashes. Scoped Ruff,
`py_compile`, and `git diff --check` passed. No Tier corpus was run.

The tanh-expanded GELU checkpoint passed its 56 dedicated cases. The activation
owner suites, adjacent input/quantized-Swish suites, active Swish and GELU
fixtures, and the complete architecture suite passed together:

```text
362 passed in 42.95s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 3.95s
```

Scoped Ruff and syntax compilation passed before the checkpoint. The lowerer
uses only its documented inherited F841 exclusion. `git diff --check` passed.
Pre- and post-extraction characterization produced 20 zero-match invocations
with unchanged operator/tensor counts, and one sequential YuNet conversion
reproduced all five fixed hashes. Temporary outputs were removed. No Tier
corpus was run.

The center/size/offset checkpoint passed its 37 dedicated cases. The new
owner, adjacent activation/input/quantized-Swish owners, active center, Swish,
and GELU fixtures, and the complete architecture suite passed together:

```text
399 passed in 42.94s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 3.99s
```

Scoped Ruff and syntax compilation pass for the owner, dedicated test,
architecture test, and lowerer with only its inherited F841 exclusion.
`git diff --check` passes. Pre/post characterization produced the same 20
zero-match invocations and unchanged counts. One sequential YuNet conversion
reproduced all five fixed hashes; temporary outputs were removed. No Tier
corpus was run.

The pseudo-expanded LeakyReLU checkpoint passed its 51 dedicated cases. The
new passthrough owner, existing indexed fusion owner, adjacent activation and
layout owners, active fixtures, and the complete architecture suite passed
together:

```text
467 passed in 42.33s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.01s
```

Scoped Ruff and syntax compilation pass for the owner, dedicated test,
architecture test, and lowerer with its inherited F841 exclusion.
`git diff --check` passes. Pre/post characterization produced 20
zero-passthrough/zero-fusion invocations with unchanged counts. One sequential
YuNet conversion reproduced all five fixed hashes; temporary outputs were
removed. No Tier corpus was run.

The PReLU passthrough checkpoint passed its 28 dedicated cases. The new owner,
adjacent Swish/GELU/center/LeakyReLU owners, quantized-Swish owner, active
fixtures, and the complete architecture suite passed together:

```text
454 passed in 50.22s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.38s
```

Scoped Ruff, syntax compilation, and `git diff --check` pass. Sequential
pre/post characterization preserved six invocations on each of the five short
representatives: four models remained zero-match and SiNet retained 23+23
rewrites. FastestDet retained its zero-match 519-to-518 tensor prune. One
sequential YuNet conversion reproduced all five fixed hashes. SiNet also
reproduced its fixed float32, float16, correspondence, and schema hashes.
Temporary worktrees and outputs were removed. No Tier corpus was run.

The elementwise/Concat layout checkpoint passed all 56 dedicated cases. The
new owner, adjacent PReLU/LeakyReLU/center/GELU/Swish and quantized-Swish
owners, SPP owner, both active compatibility fixtures, and the complete
architecture suite passed together:

```text
546 passed in 42.49s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.07s
```

Pre/post characterization preserved five invocations on each of YuNet,
FastestDet, HumanSeg, OSNet, and SiNet. FastestDet retained two active groups,
HumanSeg retained one, and the other 22 invocations remained zero-match.
Non-layout ModelIR digests match the preceding checkpoint at every active-model
invocation; full digests differ only through the intentional explicit NHWC
layout provenance.

Sequential YuNet, FastestDet, and HumanSeg conversions reproduced all fifteen
fixed artifacts. The fixed float32/float16/correspondence hashes are:

```text
YuNet:      43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380
            13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433
            7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d
FastestDet: 3bdbec5d7ad81f98cf7890fbf1a98570ebeb1a4a5c19883aca23733b31e1573b
            a14bad05eba99dc211a09aa820eb38396329b98168a2d4b20e463eb64deab617
            2bd03e9e775b4dede0310813cf36e2efc3ad9d0635ce0c5797895fe18d7fb074
HumanSeg:   b69eb57a7628668d73fbf4e06ffa23403d02ebab33b5661f0c60a81395610bb9
            e79c4081989b5a69b65dc220b6e24ad0cff5633363698feebc23b59efe1139df
            87fd06dbd120aac7f0229b02484f7929d6e752c60c19555d6795221cb0a21e46
```

All three conversions also reproduced schema hashes
`0ea6e458755747b2d98c6b68323e65f0153ded77af908b2c6560db00f9dea28f`
and
`b3a49ac25835e627fe31b92eb5df2b6d88593a571f1175b366ef7aab8e264ce8`.
Scoped Ruff, syntax compilation, and `git diff --check` pass. Temporary
worktrees and outputs were removed. No Tier corpus was run.

The StridedSlice/Concat checkpoint passed all 58 dedicated cases. The new
owner, adjacent pre-Concat Slice/Split and Split/Conv/Concat owners,
elementwise/Concat owner, and the complete architecture suite passed together:

```text
542 passed in 40.54s
```

TensorFlow import blocking was exercised sequentially for explicit direct,
default direct, and direct `-cotof` conversion:

```text
3 passed in 4.01s
```

Pre/post characterization preserved five zero-match invocations on each of
YuNet, FastestDet, HumanSeg, OSNet, and SiNet, with unchanged operator and
tensor counts in all 25 calls. The ordinary and multi-post synthetic active
forms reproduce the exact source-checkpoint non-layout ModelIR digest including
lineage metadata:

```text
basic:   1b3187150bd2819fe9e2324568dad0582c7227e12d0509e1820929fe5e438ca6
aliases: 02a2db7154cc74250a8c8f2f7a14edf2293cd4c80bc14f48f7b86b41c26ad801
```

One sequential YuNet conversion reproduced float32
`43c65782ae622ea5aefc97632f2c69033fb8a314469e4c30703c88f9907cc380`,
float16
`13232a21173ef434c7b4986320931a17a28a211109fa894023c6da7672609433`,
correspondence
`7e2b57a9b2264ef08db5aaead11922109079274eb15befbfc90bf321de370b4d`,
and the two fixed schema hashes. Scoped Ruff, syntax compilation, and
`git diff --check` pass. Temporary comparison worktrees and outputs were
removed. No Tier corpus was run.

## Latest checkpoint file set

The Slice/Logistic/Concat/Reshape tail checkpoint changes only the following
tracked source, test, and documentation files relative to `762dcdef`:

- `onnx2tf/tflite_builder/passes/slice_logistic_concat_reshape_tail_layout.py`:
  new indexed semantic owner and immutable whole-tail transaction plan.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`: thin compatibility wrapper,
  owner import, and Session `LayoutState` at both direct call sites.
- `tests/test_flatbuffer_direct_indexed_slice_logistic_concat_reshape_tail_layout.py`:
  new focused capability, numerical, rejection, bounded-dispatch, and
  determinism coverage.
- `tests/test_flatbuffer_direct_architecture.py`: ownership, transaction, and
  unchanged direct call-boundary contract.
- `docs/flatbuffer_direct_architecture.md`: indexed owner responsibilities and
  validation boundary.
- `docs/flatbuffer_direct_handoff_2026-07-14.md`: checkpoint evidence,
  remaining work, known issues, and restart instruction.

No generated model artifact, root ONNX model, dependency file, bulk-regression
profile, or exclusion policy is changed by this checkpoint.

## Tier 0–4 measured-quick regression checkpoint after `f5a40947`

The complete 49-model measured-short, zero-SWAP Tier 0–4 manifest was rerun
sequentially after the Slice/Logistic/Concat/Reshape-tail checkpoint. The run
completed in 390.255 seconds with 43 passes and six preserved known
non-passes. There were zero timeouts, zero converter nonzero exits, and zero
model-process SWAP detections. All 49 classifications and all 47 available
numeric maximum-error results match the corrected recorded expectations, so
no new correctness regression was found and no production fix was applied.

The pass-metric comparison is also stable. Of 48 comparable models, only
`sinet_320_op.onnx` changed aggregate work: preflight operator visits fell from
33,303 to 31,906 while events, statuses, snapshots, state builds, accuracy, and
classification stayed identical. This is the intended indexed-traversal
benefit.

DEIM completed successfully with the already approved TopK-index acceptance,
but its one measured duration increased from 24.043 to 37.630 seconds. It used
zero SWAP and preserved the exact raw maximum error. Treat this as an
unconfirmed runtime observation, not a semantic failure: remeasure it once
before the next quick-profile refresh, and exclude it from that quick profile
only if the over-30-second duration repeats. Do not change source code in
response to this single sample.

Detailed analysis is in
`docs/flatbuffer_direct_quick_regression_2026-07-16.md`; compact per-model
evidence is in
`docs/baselines/flatbuffer_direct_quick_tier0_4_f5a40947_result.json`.

## Failing tests and known issues

- No newly failing focused test is known at this checkpoint.
- A whole-file Ruff run on `pytorch_exporter.py` reports 282 pre-existing
  compatibility re-export, unused scaffold, and undefined-name findings. It is
  not used as the scoped checkpoint gate; changed owners/tests pass Ruff and
  the exporter passes syntax compilation.
- A whole-file Ruff run on `onnx2tf.py` reports pre-existing import-order,
  star-import, bare-except, undefined-name, and placeholder-f-string findings.
  This checkpoint uses the existing scoped exclusions for those categories;
  the changed helper owner/tests pass Ruff normally and all changed Python
  files pass syntax compilation.
- A whole-file Ruff run on `lower_from_onnx2tf.py` reports 12 pre-existing
  unused-local (`F841`) findings outside this extracted helper. The changed
  lowerer passes Ruff with that inherited category ignored; the new owner and
  dedicated owner/architecture tests pass Ruff without exclusions.
- A whole-file Ruff run on `test_tflite_builder_direct.py` reports ten
  pre-existing unused compatibility imports (`F401`). The corrected
  equal-Split fixture passes with that inherited category ignored; the
  dedicated new owner test and architecture test pass Ruff without exclusions.
- The optional PyTorch exporter suite runs when the host's Python 3.10
  `LD_LIBRARY_PATH` and `PYTHONPATH` are removed from the command environment.
  The focused results, restored native-codegen bindings, real-model artifact
  gate, and remaining inherited failures are recorded in
  `docs/flatbuffer_direct_pytorch_regression_2026-07-14.md`.
- The optional TensorFlow suite was not synchronized or run.
- Recent PyTorch source-policy checkpoints have not been followed by a Tier
  corpus conversion run. This is intentional under the current minimal-
  conversion instruction, but broad model-level regression remains unproven.

## Unfinished work

The full Goal is not complete. The fast-precanonicalize orchestrator still has
294 lines. Its remaining body is primarily the intended ordered helper
orchestration, source-line replacement, changed-flag handling, and the explicit
short-circuit boundaries required by the extracted policy decisions.

The adjacent Split/Conv/Concat trio has now been fully extracted. The
all-output and exact two-branch Conv/Concat roots share the Split-layout owner;
the generic direct bridge has its own indexed owner and preserves all three
production positions. The five short representatives remain zero-match for
the bridge, and YuNet retains byte-identical artifacts.

The unquantized pseudo-Swish, tanh-GELU, center/size/offset, pseudo-LeakyReLU,
PReLU, connected elementwise/Concat, StridedSlice/Concat, Split/mixed-Concat,
general Concat input-adapter, Slice/Logistic/Concat/Reshape tail, and strict
direct/unary pre-Add recovery helpers are now indexed at their unchanged
production positions. The broader pre-Add compatibility helper remains in
place for its Swish, Gather, affine, PReLU, broadcast, nested-Add, and direct-
fallback families, including the conservative safe-transpose entry. The next
standalone raw recovery helper in the ordered prefix is
`_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains`.
The composite pre-Concat dispatcher also retains its previously documented
legacy-family fallback after the indexed families.

The broader fixed-pipeline, remaining artifact-plan coverage, artifact-matrix,
optional TensorFlow, PyTorch/TorchScript/Dynamo/ExportedProgram, and full Tier
regression work also remains subject to the original refactor plan and its
verification gates.

## Next work

1. Confirm `git status --short --branch` is clean and local `fb-refactor5`
   matches `origin/fb-refactor5`.
2. Characterize
   `_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains`
   at its ordered production boundary before modifying it. Fix its runtime
   match set, shape/layout contracts, constant ownership, public boundaries,
   and interaction with the new pre-Add indexed owner, then select only a
   bounded semantic family with real non-zero coverage.
3. Treat `_optimize_transpose_swish_qdq_nhwc_islands` as a thin 69-line
   compatibility orchestrator unless a bounded phase-contract simplification
   is identified; all of its former raw top-level mutation loops now have
   indexed semantic owners.
4. Keep the terminal direct backend boundary explicit; do not reintroduce
   fallback into the legacy TensorFlow pipeline or broaden optional artifact
   execution.
5. Keep the audited 294-line PyTorch source orchestrator as explicit sequencing
   unless a new bounded decision is found.
6. Run only the focused synthetic/ownership/static checks unless the user asks
   for broader conversion validation. Use `uv`, run inference sequentially if
   any is explicitly requested, commit and push coherent units, and do not
   create a pull request.

## Pre-Add direct/unary extraction: pre-fix observation

The first implementation attempt for the bounded
`pre_add_direct_unary_layout` owner passed 232 focused, interaction, QLinear,
and architecture tests. A sequential three-model `-cotof` check then found an
important coverage problem before any follow-up source adjustment: the new
owner reported zero rewrites at every one of its production invocations for
`sinet_320_op.onnx`, `FastestDet.onnx`, and
`human_segmentation_pphumanseg_2021oct_org.onnx`.

This is not a conversion regression. All three models passed, used zero
model-process SWAP, reproduced their exact recorded maximum errors, and
reproduced their fixed float32, float16, and correspondence-report hashes.
The observed values were:

- SiNet: max abs `2.572051016613841e-09`, owner events eleven zeroes;
- FastestDet: max abs `1.3113021850585938e-05`, owner events eight zeroes;
- HumanSeg: max abs `2.384185791015625e-07`, owner events eight zeroes.

The preserved legacy helper therefore still performed every applicable
rewrite. Characterization before the implementation had already shown why a
strict direct-only output contract is insufficient: FastestDet uses an
`ADD -> unary -> post-transpose` suffix and a shared input requiring the
historical direct fallback, HumanSeg contains both optional-unary and shared
branch forms, and SiNet is primarily affine/PReLU/fallback rather than the
bounded direct/unary family. No production fix was made in response to this
observation. The next investigation step is to check the previously
characterized OSNet strict residuals and then either add a transactionally
validated optional-output-unary contract or narrow the checkpoint claim to a
real, non-zero production family. The broad affine, PReLU, Gather, nested-Add,
and direct-fallback behavior must remain on the compatibility path.

After adding the bounded optional-output-unary contract, the first post-change
focused run exposed one public-report compatibility difference before cleanup
was adjusted. OSNet moved six residual Adds and HumanSeg moved one; SiNet and
FastestDet correctly remained on the compatibility path. All four models
passed with zero SWAP, exact recorded accuracy, and byte-identical float32 and
float16 TFLite files. OSNet, FastestDet, and SiNet also retained their exact
correspondence hashes. HumanSeg alone changed its correspondence hash from
`87fd06dbd120aac7f0229b02484f7929d6e752c60c19555d6795221cb0a21e46`
to
`3d0848515e026eb3f1b4c6f69b936da77584eaa2e5793f3276ae3cb447783f34`.

The JSON diff is fully classified: record content and all 370 historical
lineage events are unchanged, but the indexed owner pruned eight now-unused
adapter tensors immediately after its first rewrite. The compatibility helper
historically pruned the same eight names later as the first portion of a
larger event. This split one `prune_unused_tensors` event into two, increased
`lineage_event_count` from 370 to 371, and shifted later event indexes by one.
The TFLite graph and numeric behavior did not change. This report-only issue
was recorded before changing cleanup ownership; the safe correction is to
leave pruning at the existing compatibility-wrapper boundary instead of
performing an extra early prune in the indexed sub-owner.

## Pre-Add direct/unary extraction: final checkpoint

Cleanup ownership now remains at the historical compatibility-wrapper exit.
The indexed sub-owner no longer emits an early prune event. Final sequential
conversion-only checks reproduce HumanSeg correspondence
`87fd06dbd120aac7f0229b02484f7929d6e752c60c19555d6795221cb0a21e46`
and OSNet correspondence
`35a42832e43b2076b00399ba7b22a1ff5aff83795cd333474d6bf61bf7221677`.
Their float32/float16 hashes also remain fixed. The optional-output-unary
contract remains active: the first production call indexes one HumanSeg
residual and six OSNet residuals. FastestDet and SiNet remain zero-match in the
bounded owner and continue through the characterized compatibility families.

The final focused model gate ran SiNet, OSNet, FastestDet, and HumanSeg in
that fixed order, with one converter/inference subprocess at a time. All four
passed, recorded zero model-process SWAP, and retained exact maximum absolute
errors `2.572051016613841e-09`, `2.193450927734375e-05`,
`1.3113021850585938e-05`, and `2.384185791015625e-07`, respectively. Every
float32 and float16 TFLite file was byte-identical to its fixed checkpoint.

The final synthetic/architecture gate passes 235 tests in 47.81 seconds. The
TensorFlow import-blocked explicit direct, default-direct, and direct `-cotof`
gate passes three tests in 4.24 seconds. Scoped Ruff, syntax compilation, and
`git diff --check` pass; the lowerer uses only its existing F841 exclusion.

The checkpoint changes these tracked files:

- `onnx2tf/tflite_builder/passes/pre_add_direct_unary_layout.py`: immutable
  direct/unary residual plan, indexed resolution, differential apply, optional
  output unary, post aliases, and retained legacy boundary;
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py`: indexed-first compatibility
  wrapper and Session `LayoutState` at all four direct production calls;
- `tests/test_flatbuffer_direct_indexed_pre_add_direct_unary_layout.py`:
  focused capability, ownership, rejection, bounded-dispatch, stale-plan,
  cleanup-boundary, and determinism coverage;
- `tests/test_flatbuffer_direct_architecture.py`: indexed ownership and
  unchanged call-boundary contract;
- `docs/flatbuffer_direct_architecture.md` and this handoff: design,
  pre-fix finding, report-diff attribution, final evidence, remaining work,
  and restart instruction.

No ONNX corpus model, generated TFLite artifact, dependency file, managed
quick profile, exclusion policy, or public API is changed. No broad Tier run
was performed for this bounded checkpoint.

## Pre-Add Mul-const reshape suffix extraction: pre-fix observation

The next raw helper was characterized at its unchanged ordered production
boundary before source modification. Five short, zero-SWAP representatives
(`face_detection_yunet_2023mar_int8.onnx`, `FastestDet.onnx`,
`human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`) recorded zero rewrites
at all three invocations. A static corpus query then identified the bounded
Add/Reshape/Transpose topology in `iat_llie_180x320.onnx`, a Tier 2 model that
previously passed in 17.339 seconds with zero model-process SWAP.

Sequential conversion-only characterization of IAT-LLIE found 5, 4, and 4
rewrites at the three prefix invocations. The 13 rewrites comprise seven
direct/direct pre-Add branches and six direct/Mul-constant branches. Some
branches become eligible only after the preceding phase has rewritten their
input layout, so all three production positions are semantically relevant.
The complete temporary trace fixed every changed Add, Mul, and Reshape input
and output plus every removed Transpose before implementation work began.

The pre-fix artifacts are fixed as follows:

- float32 TFLite:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16 TFLite:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- tensor correspondence report:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

The characterization also exposes an existing ownership mismatch rather than
a new regression: despite its Mul-const name, the raw helper claims valid
direct/direct chains before the adjacent general direct helper can see them.
The extraction must preserve this ordering and artifact behavior. It must use
one graph index per invocation, immutable plans with stale-contract rejection,
copy-on-write constants, explicit layout updates, bounded dispatch, and the
existing compatibility wrapper's final prune/report boundary. No production
fix has been applied at this point.

## Pre-Add Mul-const reshape suffix extraction: final checkpoint

The bounded owner is now implemented and connected before the unchanged raw
compatibility fallback. It indexes only Add candidates, shares one
`ModelIRGraphIndex` for the complete invocation, resolves immutable plans, and
revalidates every operator/tensor contract before apply. Typed permutations,
rank-four and rank-three views, reshape semantics, layout, dtype,
quantization, graph boundaries, producer order, and constant ownership are
validated before mutation. Shared channel constants and reshape-shape
constants use copy-on-write; exclusive constants retain their historical names
and update in place. Legacy NCHW consumers retain one dedicated adapter.
`LayoutState` is updated with every physical layout change, while pruning and
lineage-report grouping remain at the historical wrapper exit.

IAT-LLIE now records all 13 production rewrites in the indexed owner itself:
5, 4, and 4 across the three ordered prefix invocations. The fallback adds no
rewrite for those accepted candidates. Conversion-only output retains the
exact pre-fix hashes:

- float32 TFLite:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16 TFLite:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- tensor correspondence report:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

The single sequential managed `-cotof` gate completed in 18.084 seconds with
`classification=pass`, maximum absolute error
`4.470348358154297e-07`, and `peak_swap_kib=0`. No timeout, converter failure,
accuracy regression, artifact difference, or SWAP exclusion was found.

### Changed files and design decisions

- `onnx2tf/tflite_builder/passes/pre_add_mulconst_reshape_suffix_layout.py`
  owns indexed candidate resolution, immutable transactional plans,
  differential graph updates, constant copy-on-write, layout reconciliation,
  bounded dispatch, and retained legacy adapters.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` invokes that owner first in
  the existing compatibility helper and supplies the Session `LayoutState` at
  the unchanged ordered prefix boundary.
- `tests/test_flatbuffer_direct_indexed_pre_add_mulconst_reshape_suffix_layout.py`
  covers direct/direct and direct/Mul-constant capability, closed and legacy
  graphs, shared constants, rejection atomicity, stale plans, bounded
  candidate dispatch, idempotence, index/layout consistency, and cleanup
  ownership.
- `tests/test_flatbuffer_direct_architecture.py` fixes the indexed-first
  ownership, differential-index policy, no-early-prune contract, and Session
  layout boundary.
- `docs/flatbuffer_direct_architecture.md` and this handoff record the
  semantic contract, pre-fix observation, evidence, and restart instructions.

The main compatibility decision is intentional: the new owner accepts
direct/direct as well as direct/Mul-constant inputs because the historical
helper already claims both families before the adjacent general direct helper.
Changing that ownership would alter pass timing and could alter artifacts.
Strict cases outside the new contract still use the raw fallback. No public
API, CLI default, artifact name, dependency, ONNX corpus model, managed
profile, exclusion rule, or optional TensorFlow boundary changed.

### Verification completed

- focused indexed owners, QLinear interactions, active suffix fixtures, and
  complete architecture suite: `257 passed in 44.99s`;
- TensorFlow import-blocked explicit direct, default direct, and direct
  `-cotof`: `3 passed in 4.53s`;
- dedicated owner during its focused run: `12 passed in 0.50s`;
- ordered owner/fixture integration during its focused run:
  `17 passed in 2.33s`;
- sequential IAT-LLIE managed `-cotof`: one pass, zero failure, zero SWAP;
- scoped Ruff, Python syntax compilation, and `git diff --check`: pass.

### Known issues and unfinished work

No new failing test or model regression is known at this checkpoint. The
whole-file inherited Ruff findings and optional exporter limitations recorded
earlier in this handoff remain unchanged. The optional TensorFlow suite was
not synchronized or executed; only its import-blocked direct boundary was
tested. No broad Tier run was performed because this checkpoint deliberately
used the only short real model with non-zero owner coverage. A later
post-commit 49-model measured-quick gate is recorded below and supersedes that
model-level limitation without changing source.

The original Goal remains incomplete: the fixed pipeline contract, remaining
raw layout/reshape helpers, op-family lowering consolidation, quantization and
split/crop refresh, requested-artifact matrix, optional TensorFlow exporters,
shared PyTorch/TorchScript/Dynamo/ExportedProgram package, and final Tier
0-through-5 performance/regression gates still require later checkpoints.

### Restart instruction

Confirm `fb-refactor5` is clean and synchronized with `origin/fb-refactor5`.
Then characterize the adjacent raw
`_optimize_transpose_pre_add_reshape_transpose_suffix_nhwc_chains` helper at
its unchanged production boundary. First establish its non-zero real-model
families and overlap with the newly indexed direct/direct owner; record any
problem before changing source. Extract only a bounded family with real
coverage, retain strict rejects on the compatibility path, and validate with
focused fixtures plus the smallest sequential zero-SWAP model set. Continue
to commit and push coherent units only; do not create a pull request.

## Post-checkpoint Tier 0-4 measured-quick regression gate

Commit `5b387bc6` was subsequently checked against the fixed 49-model Tier 0-4
measured-quick profile. The runner used `uv`, one converter/inference
subprocess at a time, a fixed model order, a 45-second per-model ceiling, and
subprocess-tree `VmSwap` monitoring. The broad run completed in 452.114
seconds. Every model recorded zero SWAP and no converter returned a nonzero
exit status.

The initial result contained 42 passes, the six preserved known non-passes,
and one new unconfirmed LINEA timeout. This issue was recorded in detail,
including all generated artifact hashes and the exact difference from the
prior 23.331-second pass, before any follow-up action. No source was changed.
One isolated 90-second-headroom run then restored LINEA to pass in 25.213
seconds with exact prior maximum absolute error `0.002297189086675644`, zero
SWAP, identical TFLite/report hashes, and identical pass metrics. The timeout
is therefore runtime variance rather than a semantic regression.

The final classification is zero newly confirmed regressions among the 49
selected models. IAT-LLIE, the only non-zero production model for the newly
indexed owner, passed in 17.404 seconds with maximum absolute error
`4.470348358154297e-07` and zero SWAP. DEIM retained its user-approved success
classification in 37.006 seconds. No source fix, exclusion, timeout-policy
change, or managed-profile update was necessary. Detailed and compact evidence
is stored in:

- `docs/flatbuffer_direct_quick_regression_2026-07-16.md`;
- `docs/baselines/flatbuffer_direct_quick_tier0_4_5b387bc6_result.json`;
- `docs/baselines/flatbuffer_direct_quick_tier0_4_5b387bc6_linea_followup.json`.

## Adjacent direct/direct reshape suffix helper: pre-change observation

The adjacent raw
`_optimize_transpose_pre_add_reshape_transpose_suffix_nhwc_chains` helper was
characterized at its unchanged production boundary before any source change.
The instrumentation counted both that helper and its immediately preceding
Mul-const compatibility owner at each of the three ordered prefix
invocations. All conversions used `uv`, ran sequentially, and generated only
the requested direct TFLite artifacts.

IAT-LLIE recorded 5, 4, and 4 rewrites in the preceding indexed/compatibility
owner, but 0, 0, and 0 in the adjacent direct/direct helper. Five additional
short zero-SWAP representatives
(`face_detection_yunet_2023mar_int8.onnx`, `FastestDet.onnx`,
`human_segmentation_pphumanseg_2021oct_org.onnx`,
`osnet025_Nx3x256x128.onnx`, and `sinet_320_op.onnx`) recorded zero rewrites in
both helpers at all invocations. A static topology query over the fixed
49-model Tier 0-4 measured-quick profile found only IAT-LLIE and LINEA with an
ONNX Add -> Reshape -> Transpose `[0,2,1]` suffix. LINEA then also recorded
0, 0, and 0 in the adjacent helper during an isolated successful conversion.

The observed issue is redundant work, not an accuracy regression. The
preceding compatibility helper deliberately accepts direct/direct as well as
direct/Mul-constant inputs. For a direct input its producer, permutation,
graph-output, and consumer guards are the same as the adjacent helper's
guards. Its suffix selection, legacy-user handling, shape-constant
copy-on-write, metadata permutation, adapter insertion, operator removal, and
final prune are also identical for that family. Because it runs first at the
same three boundaries, every candidate the adjacent helper could accept has
already been accepted and removed. The adjacent helper therefore performs
three repeated full producer/consumer-map builds and Add scans without owning
a reachable semantic family.

The safe change to evaluate is removal of only the unreachable private helper
and its three production invocations. The preceding indexed owner plus its raw
compatibility fallback must remain unchanged. The active direct/direct legacy
fixture, indexed owner tests, deterministic artifact hashes, TensorFlow import
blocker, and smallest real-model inference gate must pass before the removal
can be retained. No production fix has been applied at this point.

## Adjacent direct/direct reshape suffix helper: final checkpoint

The pre-change finding was confirmed and the unreachable private helper plus
its sole production call site were removed. This deletes 290 raw lowerer lines
and prevents the recovery prefix, which executes three times, from rebuilding
producer/consumer maps and scanning every Add for an already-owned family.
The preceding indexed owner and its compatibility fallback are unchanged, so
strict indexed rejects and all direct/direct or direct/Mul-constant behavior
remain at the historical ordered boundary. An architecture assertion now
prevents either the redundant definition or a call to it from returning.

The direct/direct legacy fixture and Mul-constant legacy fixture pass through
the remaining owner. The complete indexed owner plus architecture gate passes
223 tests in 45.73 seconds. TensorFlow-import-blocked explicit direct, default
direct, and direct `-cotof` pass 3 tests in 4.71 seconds. IAT-LLIE conversion
retains the exact fixed artifacts:

- float32 TFLite:
  `75e355ba8fc01f32b9e4cf2d3c630a4c5c18e6091615a07f30851be6c6eb2881`;
- float16 TFLite:
  `4e6f95610870597b74995f441c5cc755cdd2a555a5322504a919aef85f102c43`;
- tensor correspondence report:
  `a52ffab6c473547076538d993dbadd39305304521232b85dda74fae492f77322`.

The single sequential managed IAT-LLIE accuracy gate passes in 20.335 seconds
with maximum absolute error `4.470348358154297e-07`, converter exit code 0,
and `peak_swap_kib=0`. No model exclusion, timeout-policy change, dependency,
public API, CLI behavior, generated corpus artifact, optional TensorFlow
boundary, or managed profile changed. No failure was found and therefore no
additional corrective source change was required. Scoped architecture Ruff,
Python syntax compilation, and `git diff --check` pass. Whole-file Ruff on the
lowerer continues to report the same ten inherited F841 findings as the parent
commit; this deletion introduces no new lint finding.

### Restart instruction after this checkpoint

After confirming the branch is clean and synchronized, characterize the next
raw `_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains`
helper at its unchanged three production boundaries. Use the fixed short,
zero-SWAP corpus first and locate a real non-zero family before extracting
anything. Record any ownership conflict, artifact change, accuracy failure, or
SWAP before changing source. Keep conversion checks minimal, inference
strictly sequential, and continue with coherent commit/push units only; do not
create a pull request.

## Pre-unary reshape suffix extraction: pre-change observation

The next raw helper was instrumented at its unchanged production boundary
before source modification. Seven conversion-only checks ran sequentially
under `uv`. IAT-LLIE, Face Detection YuNet int8, FastestDet, HumanSeg, OSNet,
and SiNet recorded 0, 0, and 0 rewrites. LINEA recorded one rewrite at the
first invocation and zero at the next two. Its exact changed subgraph is:

```text
node_conv2d_98_output_nhwc
  -> TRANSPOSE [0,3,1,2] -> conv2d_98
  -> LOGISTIC -> val_855
  -> MUL(conv2d_98, val_855) -> silu_49
  -> RESHAPE(view_16_reshape_shape) -> view_16
  -> TRANSPOSE [0,2,1] -> permute_7
```

The helper rewrites this generic Swish suffix to operate directly on
`node_conv2d_98_output_nhwc`, changes the Reshape output to `permute_7`, and
removes both Transposes. The first invocation changes the operator count from
1,839 to 1,837. The current raw implementation rebuilds complete producer and
consumer maps for each fixed-point iteration, scans every Reshape, searches
operator identity to rediscover removal indices, mutates ModelIR directly,
and does not update the Session `LayoutState`. The issue is architectural and
performance-related; no LINEA accuracy regression is currently known.

The exact pre-change artifacts are fixed as follows:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence report:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`.

The fixed sequential LINEA follow-up baseline passes with maximum absolute
error `0.002297189086675644` and zero process-tree SWAP. The safe extraction
scope is only the non-zero generic Swish family. It must use one differential
`ModelIRGraphIndex`, immutable plans with full revalidation before apply,
typed constant and shape/layout guards, explicit `LayoutState` updates,
bounded dispatch, and no internal prune. Plain unary cases and all strict
rejects must remain on the raw compatibility fallback. No production fix has
been applied at this point.

### First indexed implementation report-contract finding

The initial strict owner accepted exactly the intended LINEA Swish candidate
at counts 1, 0, and 0. Both TFLite files remained byte-identical, but the
correspondence report changed from
`ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`
to `20f9736c4118fcfad42e35fe786783d2d9dec587012a4b55ddd62747cc7ed6e5`.
This problem was isolated and recorded before corrective source work.

The report summaries, all 920 records, all 579 lineage-event positions, and
578 lineage events are identical. Only event 269 differs. The historical raw
helper records the Mul data-edge rewrite with source
`replace_operator_input_at`; the first indexed implementation used
`set_operator_inputs`. The semantic edge is identical in both reports:
`conv2d_98 -> node_conv2d_98_output_nhwc`. This is a report-source-label
compatibility regression, not a graph or accuracy difference. The bounded
correction is to use the graph-index-aware single-slot mutation helper for
that one Mul data edge, matching the historical lineage contract without
altering any other mutation.

## Pre-unary reshape suffix extraction: final checkpoint

The bounded indexed owner is implemented in
`pre_unary_reshape_suffix_layout.py` and connected at the unchanged wrapper
boundary. It dispatches only indexed Mul candidates and reuses one
`ModelIRGraphIndex` per invocation. Candidate resolution requires the generic
Swish topology, typed `[0,3,1,2]` and `[0,2,1]` constants, exact positive
rank-four NHWC/NCHW views, exact rank-three NCW/NWC reshape semantics,
compatible dtype and per-tensor quantization, graph ordering and boundaries,
exclusive data edges, and an exclusive typed reshape constant. Accepted
candidates become immutable plans with complete tensor/operator contracts and
are fully revalidated before apply. Mutation is differential, bounded, and
updates the Session `LayoutState`; the raw wrapper remains the only prune and
lineage cleanup boundary. Plain unary paths and every strict reject remain on
the original fallback.

The recorded lineage-source incompatibility was corrected with the
graph-index-aware single-slot input helper. LINEA now records indexed owner
counts 1, 0, and 0, while the compatibility fallback adds no rewrite for the
accepted candidate. All fixed artifacts are restored exactly:

- float32 TFLite:
  `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16 TFLite:
  `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- tensor correspondence report:
  `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`.

The single sequential managed LINEA `-cotof` gate passes in 23.336 seconds
with maximum absolute error `0.002297189086675644`, converter exit code 0,
and `peak_swap_kib=0`. The indexed owner, preceding suffix owner, active
compatibility fixtures, and complete architecture suite pass 230 tests in
45.57 seconds. The dedicated owner and wrapper cleanup suite passes 6 tests;
TensorFlow-import-blocked explicit direct, default direct, and direct `-cotof`
pass 3 tests in 4.53 seconds. Scoped Ruff, Python syntax compilation, and
`git diff --check` pass. The same ten inherited lowerer F841 findings remain
unchanged.

### Changed files and restart instruction

- `onnx2tf/tflite_builder/passes/pre_unary_reshape_suffix_layout.py` owns the
  strict indexed Swish family, immutable plans, differential mutation, and
  rank-aware layout reconciliation.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` invokes the owner before the
  unchanged raw fallback, supplies Session layout state, and preserves one
  cleanup/report boundary.
- `tests/test_flatbuffer_direct_indexed_pre_unary_reshape_suffix_layout.py`
  covers capability, atomic rejection, bounded candidate dispatch, stale
  plans, determinism, graph-index/layout consistency, exact lineage source,
  and wrapper cleanup.
- `tests/test_flatbuffer_direct_architecture.py` fixes indexed-first ownership,
  bounded dispatch, no internal prune/full-map build, and the production
  layout-state boundary.
- `docs/flatbuffer_direct_architecture.md` and this handoff record the design,
  pre-change evidence, report-contract problem before correction, final
  evidence, and remaining work.

No public API, CLI behavior, dependency, managed profile/exclusion, timeout
policy, ONNX corpus model, or optional TensorFlow exporter changed. No broad
Tier run was performed because LINEA is the only established non-zero model
for this bounded family and the preceding fixed 49-model checkpoint already
confirmed zero new regressions.

After confirming the branch is clean and synchronized, characterize the next
raw `_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains` helper at
its unchanged production boundary. Establish a real non-zero family before
source work, record any problem before correction, and retain strict rejects
on the compatibility path. Continue with minimal sequential zero-SWAP model
checks and coherent commit/push units only; do not create a pull request.

## YOLO factorized expand-dims extraction: pre-change observation

The raw `_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains`
helper was instrumented at all four unchanged production invocations before
source modification. Fourteen conversion-only models ran sequentially under
`uv` with subprocess-tree `VmSwap` monitoring. Every conversion exited zero
and every model recorded `peak_swap_kib=0`. Twelve models recorded zero
rewrites throughout. `yolov7-tiny.onnx` and `yolo_test.onnx` each recorded
3, 0, 0, and 0 rewrites.

All six non-zero chains are the generic factorized YOLO Case B:

```text
NHWC -> TRANSPOSE [0,3,1,2] -> NCHW
     -> RESHAPE [N,A,B,H,W]
     -> TRANSPOSE [0,1,3,4,2] -> [N,A,H,W,B]

NHWC -> RESHAPE [N,H,W,A,B]
     -> TRANSPOSE [0,3,1,2,4] -> [N,A,H,W,B]
```

Both models use `A=3` at spatial sizes 80, 40, and 20. YOLOv7-tiny uses
`B=85`; yolo_test uses `B=8`. The first invocation removes three pre-layout
Transposes, rewrites three Reshape inputs, shapes, and options, and remaps the
three retained post-Transpose permutations. The next three invocations own no
additional chain.

The current raw Case B implementation lacks an explicit transactional
contract. It checks the post permutation value early enough that its normal
single-threaded path does not leave the Reshape half-written, but it mutates
both the Reshape shape and post-permutation constants in place without
checking exclusive ownership. A shared constant can therefore change an
unrelated consumer. The helper also rebuilds full consumer maps and scans all
Transposes seven times for a three-chain model: four fixed-point scans in the
first invocation plus one no-op scan at each later invocation. Constant type
and ownership, layouts, dtype, quantization, operator ordering, and graph
boundaries are not validated as one immutable contract.

The exact pre-change artifacts are:

- yolo_test float32:
  `439d9a8b893bf6bfbd92aa0155bd15a4185b5fcdb6e65ddb48f718a41b75bdfc`;
- yolo_test float16:
  `7b1ef8b13de65068b3fe8166d5481553e2e41194c0cfe9ee48f4be5ad3417eff`;
- yolo_test correspondence:
  `36d728e9294f1d4f1319c45306a088bced6b54ad393f71f4925f3178f0d9c1ca`;
- YOLOv7-tiny float32:
  `4738ec36f18f3ccfcf3d53e7b43a59091ea75b88a9620460e95c724bf363326e`;
- YOLOv7-tiny float16:
  `1675ba6a669e22f0e7cf24941c421c9d3241c837c681d7a97c465a5665a536bd`;
- YOLOv7-tiny correspondence:
  `c6667fcef416fafe7eed113ac43a6df4cab85c6c475006bbd86a0ad01de4ffb8`.

The pre-change sequential yolo_test `-cotof` gate passes in 10.489 seconds
with maximum absolute error `2.4437904357910156e-06`, converter exit code 0,
and zero SWAP. The safe extraction scope is only Case B. It must validate both
typed constants and all tensor/operator/layout contracts before any mutation,
use an immutable revalidated plan and one differential graph index, update
Session layout state, preserve the single wrapper prune/report boundary, and
leave singleton Case A plus every strict reject on the raw compatibility
fallback. No production fix has been applied at this point.

### First indexed Case B implementation guard finding

The first focused owner run rejected four valid synthetic Case B expectations;
the two historical Case A fallback fixtures and shared-constant atomicity still
passed. This was recorded before correction. The cause is bounded and does not
affect the raw production result: the initial owner reused a typed permutation
utility whose contract is deliberately rank four, so the valid rank-five post
permutation `[0,1,3,4,2]` could never pass its guard. The indexed owner
therefore returned zero and the unchanged fallback continued to own real
models. The correction is a local typed permutation reader parameterized by
the exact expected vector length, while retaining the stricter exclusive
mutable-vector guard for the post constant before apply.

## YOLO factorized expand-dims extraction: final checkpoint

The strict generic Case B owner is implemented in
`expanddims_reshape_layout.py` and connected at both historical call sites,
which execute at four production boundaries. It builds one
`ModelIRGraphIndex` per invocation and dispatches each indexed Transpose
candidate at most once. Resolution validates the exact factorized topology,
typed rank-four and rank-five permutations, positive static rank-four and
rank-five views/signatures, `A > 1`, `C=A*B`, dtype and per-tensor
quantization, graph boundaries and order, exclusive edges, and exclusive
typed mutable constants. The immutable plan records all tensor/operator
contracts and is fully re-resolved before differential apply. Session layout
state is updated, while pruning and report grouping remain at the single raw
wrapper boundary.

The raw compatibility path still owns singleton Case A and all relaxed
rejects. Its Case B mutation now validates both constant values and rejects a
shared Reshape-shape or post-permutation constant before any write, preventing
an unrelated consumer from observing a layout-specific in-place update. The
two historical Case A fixtures remain unchanged.

Both production models record all accepted rewrites in the indexed owner:
3, 0, 0, and 0 for `yolo_test.onnx`, and 3, 0, 0, and 0 for
`yolov7-tiny.onnx`. The raw fallback adds zero for those accepted candidates.
All six fixed artifacts remain exact:

- yolo_test float32:
  `439d9a8b893bf6bfbd92aa0155bd15a4185b5fcdb6e65ddb48f718a41b75bdfc`;
- yolo_test float16:
  `7b1ef8b13de65068b3fe8166d5481553e2e41194c0cfe9ee48f4be5ad3417eff`;
- yolo_test correspondence:
  `36d728e9294f1d4f1319c45306a088bced6b54ad393f71f4925f3178f0d9c1ca`;
- YOLOv7-tiny float32:
  `4738ec36f18f3ccfcf3d53e7b43a59091ea75b88a9620460e95c724bf363326e`;
- YOLOv7-tiny float16:
  `1675ba6a669e22f0e7cf24941c421c9d3241c837c681d7a97c465a5665a536bd`;
- YOLOv7-tiny correspondence:
  `c6667fcef416fafe7eed113ac43a6df4cab85c6c475006bbd86a0ad01de4ffb8`.

The final single sequential yolo_test `-cotof` gate passes in 11.264 seconds
with maximum absolute error `2.4437904357910156e-06`, converter exit code 0,
and `peak_swap_kib=0`. The three indexed suffix owners, active compatibility
fixtures, and complete architecture suite pass 241 tests in 44.41 seconds.
The final dedicated Case B plus Case A compatibility suite passes 9 tests in
0.74 seconds. TensorFlow-import-blocked explicit direct, default direct, and
direct `-cotof` pass 3 tests in 4.48 seconds. Scoped Ruff, Python syntax
compilation, and `git diff --check` pass; the same ten inherited lowerer F841
findings remain unchanged.

### Changed files and restart instruction

- `onnx2tf/tflite_builder/passes/expanddims_reshape_layout.py` owns strict
  factorized Case B resolution, immutable revalidation, differential mutation,
  constant ownership, bounded dispatch, and layout reconciliation.
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` invokes the owner at the two
  existing call sites, passes Session layout state, retains singleton Case A,
  and protects shared constants before the compatibility mutation.
- `tests/test_flatbuffer_direct_indexed_expanddims_reshape_layout.py` covers
  capability, Case A rejection/fallback, shared constants, atomic stale-plan
  rejection, candidate bounds, determinism, exact lineage, graph-index/layout
  consistency, and wrapper cleanup.
- `tests/test_flatbuffer_direct_architecture.py` fixes indexed-first ownership,
  single-pass bounded dispatch, no internal prune/full-map build, and both
  production layout-state boundaries.
- `docs/flatbuffer_direct_architecture.md` and this handoff record the
  characterization, pre-change ownership issue, first implementation guard
  finding before correction, final design, evidence, and restart order.

No public API, CLI behavior, dependency, managed profile/exclusion, timeout
policy, ONNX corpus model, or optional TensorFlow exporter changed. No broad
Tier run was repeated because the fixed 49-model checkpoint already reports
zero new regressions and this extraction used the only two established
non-zero, short, zero-SWAP models.

After confirming the branch is clean and synchronized, characterize the next
raw `_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains` helper at
all unchanged production boundaries. Record real non-zero ownership and any
problem before source work, keep strict rejects on the compatibility path,
and use only the smallest sequential zero-SWAP model gate. Continue with
coherent commit/push units only; do not create a pull request.

## Flatten-HW reshape suffix checkpoint: pre-change record

Before changing source, the raw
`_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains` helper was
instrumented at all four unchanged production invocations. Fourteen previously
measured short Tier 0-4 representatives were converted strictly sequentially:
IAT-LLIE, LINEA, YuNet INT8, FastestDet, PPHumanSeg, OSNet, SiNet,
Tiny-YOLOv2, YOLOv7-tiny, DAMO-YOLO, NanoDet, YOLOX INT8, YOLO-Free, and
yolo_test. All fourteen conversions exited 0, none timed out, and every
process-tree monitor recorded `peak_swap_kib=0`. Individual conversion times
were 1.793-8.535 seconds.

Only LINEA establishes real ownership. Its four invocation counts are
`2, 0, 0, 0`: the first prefix invocation collapses two exact
`NHWC -> NCHW -> [N,C,H*W] -> [N,H*W,C]` chains, with final shapes
`[1,6400,128]` and `[1,1600,128]`. The other thirteen models record zero at
every invocation. The unchanged pre-extraction LINEA artifacts are fixed as:

- float32: `fd91ea915b600b3581e8e0e68925fefd5302cd1bfb373ebca8b9b9410138c611`;
- float16: `c8e44a48221eeead187869d93dfef1f7775420335aae5c63873118738d39f9a8`;
- correspondence: `ac4bc30fd7076f40adb4b357f9556aef656dde9d6e27e0e8f9d95588a0d799dd`.

No observed conversion or accuracy regression prompted this checkpoint.
However, the two proven rewrites are currently owned by an unbounded
`while True` full-operator scan that rebuilds the whole consumer map after
each rewrite. The helper does not use the shared graph index or Session
`LayoutState`, accepts reshape/permutation constants without typed ownership
contracts, and mutates the reshape constant in place even when another
operator or a graph boundary can observe it. It also does not revalidate an
immutable plan immediately before mutation. These are recorded latent
correctness, interaction, and efficiency problems before implementation.
The safe scope is therefore only LINEA's exact static family; relaxed or
shared-constant variants must remain unchanged on a compatibility fallback.

### Implemented result and verification

`flatten_hw_reshape_layout.py` now owns that exact static family. It dispatches
the indexed Transpose set once, validates typed `[0,3,1,2]` and `[0,2,1]`
constants, exact positive shape/signature views, dtype and per-tensor
quantization, operator order, graph boundaries, exclusive data edges, an
exclusive typed reshape constant, and Session layout compatibility. Complete
operator/tensor contracts are stored in an immutable plan and resolved again
before apply. Apply uses graph-index-aware input/output mutation, removes both
Transposes differentially, and reconciles `LayoutState`; the owner neither
prunes nor builds whole-graph maps. The wrapper keeps the sole compatibility
prune and raw fallback. Shared, boundary-visible, produced, or variable shape
constants are rejected before the fallback can mutate them.

All four LINEA boundaries now report indexed counts `2, 0, 0, 0`; the wrapper
totals are the same, proving no residual raw rewrite for accepted candidates.
The post-extraction artifacts are byte-for-byte identical to the pre-change
files and retain the three hashes recorded above. The single sequential
managed LINEA `-cotof` gate exits 0 in 21.914 seconds, reports maximum absolute
error `0.002297189086675644` with `pass=True`, and records
`peak_swap_kib=0`. No new regression or exclusion was found.

A mechanical context patch made while hardening the fallback initially placed
the local `model_inputs` snapshot in an earlier legacy loop rather than this
wrapper. Immediate targeted diff inspection found the misplaced line and the
undefined target reference before conversion execution; both were corrected,
then the dedicated suite was rerun. This did not reach a committed or tested
checkpoint and produced no artifact difference.

Verification completed under `uv`:

- 14 sequential short Tier 0-4 characterization conversions: all exit 0,
  1.793-8.535 seconds each, no timeout, and SWAP 0;
- focused owner plus architecture selectors: 10 passed;
- the four adjacent reshape-suffix owners, architecture suite, active
  compatibility selectors, and TensorFlow-import-blocked explicit direct,
  default direct, and direct `-cotof`: 238 passed, 773 deselected in 47.59
  seconds;
- scoped Ruff, syntax compilation, and `git diff --check`: passed;
- whole-lowerer Ruff retains exactly the same ten inherited F841 findings and
  reports no new finding.

Changed files in this checkpoint are
`onnx2tf/tflite_builder/passes/flatten_hw_reshape_layout.py`,
`onnx2tf/tflite_builder/lower_from_onnx2tf.py`,
`tests/test_flatbuffer_direct_indexed_flatten_hw_reshape_layout.py`,
`tests/test_flatbuffer_direct_architecture.py`,
`docs/flatbuffer_direct_architecture.md`, and this handoff. No public API,
CLI behavior, dependency, corpus profile/exclusion, timeout policy, or
optional TensorFlow boundary changed.

After confirming this checkpoint is clean and synchronized, characterize the
next raw
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains`
helper at each unchanged production boundary. Record any non-zero model and
problem before source work, retain relaxed cases on fallback, and use only the
smallest sequential zero-SWAP accuracy gate. Continue with coherent
commit/push units only; do not create a pull request.

## Rank-3 to NHWC reshape shim: measured no-change decision

The raw
`_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains` helper
was instrumented at all four unchanged production invocations before any
source change. The same fourteen short Tier 0-4 representatives used for the
preceding checkpoint all converted successfully and sequentially in
1.788-8.917 seconds, with no timeout and `peak_swap_kib=0`. Every invocation
count was zero. A read-only ONNX topology scan then found zero exact
`Reshape -> Transpose[0,3,2,1] -> Reshape -> Transpose[0,2,3,1]` candidates in
both the fixed 49-model measured-quick set and all 420 active Tier 0-4 models.

The helper still contains an unbounded rewrite loop, repeated full consumer
map construction, raw constant copy-on-write, and no graph-index or Session
layout-state contract. These are architectural liabilities, but there is no
real current-corpus ownership evidence from which to freeze artifact and
accuracy behavior. Removing it would also discard an existing feature.
Accordingly, no source or test behavior was changed and no synthetic-only
replacement was introduced. Revisit this helper only when a real non-zero
model or a public compatibility fixture is available. The next evidence-first
candidate is
`_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains`.

## QKV reshape/transpose recovery checkpoint: pre-change record

Before changing source, the raw
`_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains`
helper was instrumented at all four production invocations. The fourteen short
zero-SWAP representatives and four additional Tier 0/2 static-topology
candidates all converted sequentially, exited 0, and recorded zero rewrites.
The additional candidates were ViT-B/16 partial, both CRNN English recognition
models, and monodepth; their conversions took 2.118-3.681 seconds with SWAP 0.

A shape-aware read-only scan of all 420 active Tier 0-4 ONNX graphs narrowed
the exact fully static family to htdemucs and three RF-DETR variants. htdemucs
was not converted because it is an explicit managed exclusion. The smallest
active RF-DETR representative, Tier 3 `rf-detr-nano.onnx` (770 ONNX nodes),
then converted in 10.890 seconds with exit code 0 and `peak_swap_kib=0`. Its
four raw invocation counts are `5, 0, 0, 0`; all five accepted chains use
`[300,1,256] -> [300,8,32] -> [8,300,32] -> [1,8,300,32]` with permutation
`[1,0,2]`. No `[1,2,0]` production owner has been established. The fixed
pre-extraction artifacts are:

- float32: `fda7d97eaad2b19ee2ac31411099067e78b747515952b7c65ba52a0f1454f1fb`;
- float16: `a80051b2d6bb871ee871f0d1528e1ea7c8d4e7f6ecbfc16daec4fa78d696fd1f`;
- correspondence: `262235cec5a8df73ff2afd7f1eb28678cc7312f4a19dd09d278fd8db77cbdec4`.

No observed regression prompted the change. The proven raw owner nevertheless
has a correctness hazard that is recorded before implementation: it updates
or clones the first Reshape shape constant before attempting the Transpose
permutation constant. If the second operation fails, the first mutation and
possibly its lineage event remain, so the rewrite is not transactional. The
helper also repeats a full consumer-map build after every rewrite, uses an
unbounded whole-RESHAPE scan, accepts untyped/produced/variable/boundary
constants, and does not update the shared graph index or Session
`LayoutState`. The safe implementation scope is the exact RF-DETR `[1,0,2]`
family with fully typed, exclusively mutable constants and immutable
revalidation. The unproven `[1,2,0]`, shared-constant copy-on-write, dynamic,
and otherwise relaxed cases must retain compatibility fallback behavior.

## QKV reshape/transpose recovery checkpoint: completed

The strict production-proven family is now owned by
`onnx2tf/tflite_builder/passes/attention_qkv_reshape_layout.py`. It dispatches
only indexed Reshape candidates, validates the exact positive static
`[A,1,C] -> [A,H,D] -> [H,A,D] -> [1,H,A,D]` contract with `C=H*D` and
permutation `[1,0,2]`, and requires typed exclusive mutable constants, matching
dtype/per-tensor quantization, resolved sources, exclusive edges, operator
order, graph boundaries, and UNKNOWN Session layout. Its immutable plan
captures tensor and operator contracts and is re-resolved before any mutation.
Apply changes the shared graph index differentially, preserves the historical
output lineage event, removes only the tail Reshape, and explicitly reconciles
`LayoutState`. Candidate and rewrite bounds are deterministic; the owner has
no internal prune, repeated producer/consumer-map construction, or unbounded
loop.

The lowerer wrapper invokes this indexed owner first and retains the unchanged
raw implementation for `[1,2,0]`, shared-constant copy-on-write, dynamic, and
otherwise relaxed compatibility cases. It remains the only prune boundary and
removes the corresponding stale layout entries. Both production boundaries
now supply the Session `LayoutState`. Focused fixtures prove indexed
differential mutation and lineage, wrapper cleanup, strict candidate/bound
handling, stale-plan atomic rejection, determinism, and the three named raw
fallback classes.

Post-change `rf-detr-nano.onnx` instrumentation records indexed invocation
counts `5, 0, 0, 0` and identical wrapper totals, so the raw residual is zero.
The conversion exited 0 in 12.121 seconds with `peak_swap_kib=0`. All three
outputs are byte-identical to the fixed pre-change baseline:

- float32: `fda7d97eaad2b19ee2ac31411099067e78b747515952b7c65ba52a0f1454f1fb`;
- float16: `a80051b2d6bb871ee871f0d1528e1ea7c8d4e7f6ecbfc16daec4fa78d696fd1f`;
- correspondence: `262235cec5a8df73ff2afd7f1eb28678cc7312f4a19dd09d278fd8db77cbdec4`.

Its sequential managed `-cotof` run exited 0 in 41.259 seconds, used no SWAP,
and passed all output checks with maximum absolute error
`0.000102996826171875`, RMSE approximately `1.01567e-05`, and cosine
similarity 1. The unchanged artifacts produced during that accuracy run have
the same three hashes.

Validation completed for this checkpoint:

- the three default/direct/`-cotof` TensorFlow import-blocker tests: `3 passed`;
- the selected adjacent indexed-owner, direct-builder, optional-TensorFlow,
  and architecture suite: `247 passed, 773 deselected`;
- focused QKV/architecture validation before the combined gate:
  `11 passed, 210 deselected`;
- scoped Ruff for the new owner and its tests: clean;
- `py_compile` for the new owner, lowerer, and related tests: clean;
- `git diff --check`: clean.

Whole-file Ruff for the legacy lowerer still reports the same ten inherited
`F841` unused-local findings at lines outside this change. No new finding was
introduced. No conversion failure, timeout, SWAP event, artifact drift,
TensorFlow import, or accuracy regression is known for this checkpoint.

Changed files in this checkpoint are:

- `onnx2tf/tflite_builder/passes/attention_qkv_reshape_layout.py` (new strict
  indexed owner);
- `onnx2tf/tflite_builder/lower_from_onnx2tf.py` (owner dispatch, Session
  layout propagation, and compatibility/prune boundary);
- `tests/test_flatbuffer_direct_indexed_attention_qkv_reshape_layout.py` (new
  unit and compatibility tests);
- `tests/test_flatbuffer_direct_architecture.py` (bounded owner/wiring
  constraints);
- `docs/flatbuffer_direct_architecture.md` and this handoff (design and measured
  evidence).

The preceding rank-3-to-NHWC reshape helper intentionally remains unchanged
because neither the 49-model measured-quick set nor all 420 active Tier 0-4
graphs established a real owner. The broader Goal also remains incomplete;
quantization/split/exporter extraction and the remaining raw layout helpers
have not been completed. At resume, first inspect the next adjacent raw helper,
`_optimize_attention_gather_transpose_reshape_cleanup_chains`, characterize
all unchanged production invocations with short zero-SWAP models, and record
any non-zero owner or problem before source changes. Continue with sequential
inference only, minimal conversion gates, coherent commit/push units, and no
pull request.

## Current-HEAD 49-model measured-quick gate: pre-exclusion record

Commit `0e666234` was subsequently exercised with the fixed 49-model Tier 0-4
measured-quick profile. This was a strictly sequential `uv` run with a
45-second per-model ceiling and subprocess-tree SWAP monitoring. It completed
all entries in 503.563 seconds with 42 passes, six preserved known non-passes,
and one timeout. Every model recorded `peak_swap_kib=0`; no converter returned
a nonzero exit status. All classifications and available numeric errors match
the stable `f5a40947` result except HybridNets. LINEA, IAT-LLIE, and DEIM all
passed with their established results.

Before any follow-up change, the HybridNets issue is fixed in detail in
`docs/baselines/flatbuffer_direct_quick_tier0_4_0e666234_result.json` and the
quick-regression report. It timed out at 45.115 seconds after writing all three
normal conversion artifacts. The float32 artifact is byte-identical to the
prior successful diagnostic, whose maximum absolute error was
`0.0002593994140625`, and SWAP remained zero. This is the model's second
45-second quick-ceiling timeout, so it is runtime variance rather than a
semantic regression but is no longer reliably short. The next action is a
profile-only exclusion with `repeated_quick_ceiling_timeout`, as required by
the user timeout-exclusion policy. Do not modify converter source or rerun the
model before applying that policy update.
