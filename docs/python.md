# Python Integrations

`jerrythomas.integrations.ml` exposes the final dataset pipeline without
persisting served output. Its prerequisites must already be current; run
`jerry build` or a normal `jerry serve` before opening it.

## Final samples

`iter_samples()` yields Jerry's canonical `Sample` values after fold selection,
scaling, and postprocess:

```python
from jerrythomas.integrations.ml import iter_samples


for sample in iter_samples(
    "path/to/project.yaml",
    dataset="default",
    output_id="holdout.train",
):
    print(sample.key, sample.features, sample.targets)
```

`dataset` is the named dataset ID from `datasets/<id>.yaml`.
Datasets with a configured split require an `output_id`; unsplit datasets
reject it. `limit` bounds the selected output without materializing it.

## Bounded model batches

Install the optional NumPy integration:

```bash
python -m pip install "jerry-thomas[ml]"
```

`iter_model_batches()` converts the same sample stream into bounded numerical
matrices:

```python
from jerrythomas.integrations.ml import iter_model_batches


for batch in iter_model_batches(
    "path/to/project.yaml",
    dataset="default",
    output_id="holdout.train",
    batch_size=4096,
    dtype="float32",
):
    print(batch.keys)
    print(batch.feature_columns, batch.features.shape)
    print(batch.target_columns, None if batch.targets is None else batch.targets.shape)
```

Feature and target columns follow postprocessed metadata order. Fixed-length
lists from `sequence`, `collect`, or intrinsically list-valued fields expand to
numbered columns such as `history[0]`. Missing, nonnumeric, non-finite, or
unrepresentable values fail with their sample key and column instead of being
silently coerced. Feature-only datasets expose `targets=None`.

The iterator preserves canonical sample order and yields a shorter final batch.
It does not shuffle or retain the full dataset. PyTorch can consume the arrays
without a Jerry-specific adapter:

```python
import torch


features = torch.from_numpy(batch.features)
targets = None if batch.targets is None else torch.from_numpy(batch.targets)
```

For repeated research and training passes, persist Parquet once and let the
analytical or training framework read it directly. Model batches are an
ephemeral, one-pass integration boundary; Parquet remains Jerry's durable
tabular format.

## Migrating from Jerry 6

Jerry 7 removes the stateful `VectorAdapter`, the row and DataFrame helpers,
`stream_vectors()`, and `torch_dataset()`. Use:

- `iter_samples()` for final keyed Jerry samples;
- `iter_model_batches()` for bounded numerical arrays;
- served Parquet with Pandas or Polars for tabular research; and
- `torch.from_numpy()` for framework conversion.

Import the new API from `jerrythomas.integrations.ml`; it is no longer
re-exported from `jerrythomas.integrations`. The `ml` extra installs NumPy
only. Install PyTorch separately when using the conversion example above.
