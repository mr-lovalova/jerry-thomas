# Preprocess Transforms

Preprocess transforms run on mapped domain records before ordering. Configure
them under `preprocess:` on a source-backed stream.

## `where`

- Binary comparisons: `eq`, `ne`, `lt`, `le`, `gt`, `ge`.
- Membership: `in`, `not_in`.
- ISO or datetime literals are compared with timezone awareness.

```yaml
preprocess:
  - { operation: where, field: time, operator: ge, comparand: "${start_time}" }
  - { operation: where, field: station, operator: in, comparand: [a, b, c] }
```

## Built-In Transforms

- `round_time`: round timestamps on a fixed UTC cadence (`10m`, `1h`, `1d`).
  Required `direction: floor` rounds down; `direction: ceil` rounds up.
  Exact-grid timestamps stay unchanged in either direction.
  Every record and its fields are retained, including multiple records that
  land on the same timestamp. Use `floor` only when that earlier timestamp is intended.
  For completed period aggregation, use the ordered `resample` transform instead.
- `shift_time`: shift timestamps by a duration (`by: 1d`, `by: -1h`).
- `where`: filter records with the operator language above.

```yaml
preprocess:
  - { operation: round_time, cadence: 1h, direction: ceil }
```
