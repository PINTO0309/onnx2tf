# `fb-refactor8` improvement summary

## Purpose and compatibility boundary

`fb-refactor8` continues the characterize-first refactoring of the
TensorFlow-free `flatbuffer_direct` pipeline from the merged `2.6.6`
checkpoint. Public CLI/Python behavior, artifact names, pass order, dependency
set, optional TensorFlow boundary, and strictly sequential inference policy
remain fixed.

Development on this branch uses coherent commits and pushes only. No pull
request is to be created, reopened, or updated.

## Terminal Expand/Squeeze reconciliation characterization

The first unit inventories the raw static-shape reconciliation immediately
after `_terminal_expand_squeeze_stats` and before `_advance_post_progress()`.
The current boundary discards the default one-key result.

The new contracts establish that:

- the default result remains
  `{"reconciled_static_tensor_shapes": 0}` on a stable graph;
- the opt-in complete schema additionally reports
  `reconciled_static_shape_mutations`;
- reconciliation must remain unconditional because a stale shape produced by
  an earlier late-layout owner can require repair even when the terminal
  Expand/Squeeze owner reports zero rewrites;
- the fixture's static tensor shape is repaired while its existing explicit
  shape signature remains unchanged, matching the current reconciler
  contract;
- the call remains directly adjacent to the retained Expand/Squeeze result and
  `_advance_post_progress()`;
- there is no reusable live `ModelIRGraphIndex` at this boundary. The
  Expand/Squeeze owner creates a local index only for dynamic pre-operator
  insertion and does not expose it. Index sharing therefore requires a
  separate owner-level characterization and is not mixed into result
  retention.

The selected implementation is observation-only assignment to
`_terminal_expand_squeeze_static_shape_stats` with
`include_mutation_count=True`. It must not add a guard, consumer, graph scan,
pass, or index construction.

Characterization validation completed sequentially under `uv`:

- dedicated contract: `2 passed, 1 xfailed in 0.54s`;
- targeted Ruff, Python bytecode compilation, and whitespace validation:
  passed.

The sole expected failure is the intentionally unimplemented result
assignment. No production source, public API, dependency, pass order,
TensorFlow boundary, or model artifact changed in this checkpoint.
