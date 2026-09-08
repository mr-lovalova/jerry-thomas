# Testing data correctness

A finite test suite cannot prove that every future change is correct. The
practical standard is narrower: every declared data invariant must have an
exact or metamorphic test. Mutation testing then helps find assertions that are
missing from that contract.

## Test ownership

- Unit tests own one public contract or algorithmic invariant. Keep them beside
  the module's test area and avoid repeating the same unknown-field or error
  contract for every model.
- Integration tests use normal project loading, profile planning, artifact
  building, runtime execution, and persistence. They must not construct an
  alternative test-only sample assembly path or inject artifacts that
  production would build.
- Regression fixtures are small, readable, and behavior-dense. Prefer JSONL or
  CSV that can be reviewed in a diff over opaque serialized objects or large
  real-world dumps.
- Golden assertions cover complete persisted rows: sample order, complete keys,
  complete ordered feature and target maps, nulls, and values. Presence-only
  assertions are insufficient because unexpected columns can otherwise pass.

## Regression matrix

| Invariant | Coverage |
| --- | --- |
| Raw ordering | Reverse and seeded shuffle of every core regression source produce identical bytes |
| File layout | Single-file and deliberately misnamed multi-file glob inputs agree |
| Bounded sorting | A forced external spill produces the same persisted dataset |
| Long identity | `sample.keys` preserve entity identity in row keys |
| Wide identity | Remaining partition fields produce deterministic series IDs |
| Hybrid identity | Entity row keys and widened metric columns coexist |
| Alignment | Sparse aligned inputs only derive values at matching keys |
| Broadcast | Exact global records are reused across partitions and persisted with stable identity |
| Bivariate rolling | Strict slopes preserve partition state, missing resets, and numerical stability |
| Forward windows | Future-only sums exclude the current row and preserve strict terminal nulls |
| Logarithms | Natural-log domains, missing values, and near-zero `log1p` precision are exact |
| Non-finite values | Series and output boundaries normalize `NaN` to null and reject infinity |
| Sequencing | Fixed windows, stride, missing elements, and column order are golden |
| Collection | Exact bucket cardinality, ordering, placeholders, and wide identities are enforced |
| Scaling | Standard and folded feature/target statistics and values are golden |
| Leakage | Changing validation/test values cannot change fitted train statistics or introduce training columns |
| Walk-forward splits | Fold membership, purge exclusions, routing, and row order are golden |
| Artifact caching | Unchanged auto runs reuse artifacts; source/config changes rebuild stale artifacts |
| Persistence | Real serve profiles publish exact JSONL outputs and successful run metadata |

The integration fixtures live under `tests/fixtures/`. Their tests should call
`build_runtime_run_request()` and `run_profiles()` unless the test is explicitly
about a lower pipeline boundary.

## Mutation testing

Mutation testing is intentionally a reviewed diagnostic, separate from the fast
pull-request suite. The configured scope covers alignment, sorting, sample-key
generation, split labeling, typed vector conformance, scaling, and metadata
collection.

```bash
python -m pip install -e '.[test,mutation]'
mutmut run
mutmut results
```

Review every surviving mutant. A mutation that can change a key, value, null,
row, column, order, fold, resource lifetime, or persisted result requires a
test. Mutations confined to diagnostics, performance-only serialization
choices, or provably equivalent expressions may be excluded, but the exclusion
must describe that boundary rather than weakening a data assertion.

`mutmut` exits successfully even when mutants survive, and this scope includes
some diagnostics alongside the data algorithms. Therefore command success is
not a release gate: the reviewed survivor list is the result. Do not add a
numeric score gate that rewards tests for exact wording or incidental work.

Delete `mutants/` before every authoritative full audit. Mutmut caches results,
so a plain rerun can retain statuses from before a source or test change.
