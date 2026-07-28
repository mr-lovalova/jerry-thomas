# Sample Postprocessing

Postprocessing runs after sample assembly. Configure its row filters
under `postprocess:` in `dataset.yaml`:

```yaml
postprocess:
  samples:
    features:
      threshold: 0.5
      ids: [price, volume]       # optional; defaults to every feature
    targets:
      threshold: 1.0
      ids: [return]
```

The pipeline has one fixed order:

1. assemble feature and target vectors;
2. conform vectors to the declared typed metadata contract;
3. filter sample rows by feature coverage, then target coverage.

Feature and target policies are separate:

- `FilterFeatureSamplesTransform` and `FilterTargetSamplesTransform` filter
  complete sample rows by the coverage of selected cells.

Sample filters accept an optional `ids` subset and default to every declared
ID. Empty, duplicate, or unknown IDs are errors.

Postprocessing does not mutate vector values. Repair missing record values with
an ordered stream `fill` or `forward_fill` operation before feature extraction.
This keeps history and partition semantics at the stage where they are known.

Postprocessing never chooses columns from observed coverage because doing so
before a split would let validation or test availability influence the training
schema. Use coverage reports to inspect sparse series and change the declarations
explicitly.

## Typed conformance

`ConformFeaturesTransform` and `ConformTargetsTransform` receive typed
`VectorMetadataEntry` objects directly. They fill wholly absent metadata
entries, order values by the metadata contract, canonicalize missing values,
and reject unexpected IDs and incorrect value shapes. They do not truncate
values, silently discard extra columns, or mutate the pipeline context.
