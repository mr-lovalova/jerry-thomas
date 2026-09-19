import pickle
from datetime import datetime, timezone

import pytest

from jerrythomas.domain.sample_key import SampleKeyContract
from jerrythomas.operations.artifacts import series_workers
from jerrythomas.operations.artifacts.series import _ProjectedRow, _ProjectedScalar


@pytest.mark.parametrize("batch_bytes", [1, 1024 * 1024])
def test_worker_transfer_snapshots_reused_values_before_advancing(
    monkeypatch, batch_bytes
):
    monkeypatch.setattr(series_workers, "_BATCH_BYTES", batch_bytes)
    sample_keys = SampleKeyContract(["ticker"])
    time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    values = []
    batches = []

    class Receiver:
        def send(self, batch):
            batches.append(batch)

    def rows():
        for value in range(3):
            values[:] = [value, None]
            sample_keys.validate(("A",))
            yield _ProjectedRow(
                key=(time, "A"),
                time=time,
                features=(_ProjectedScalar("value", values, True),),
                targets=(),
            )
        values.clear()

    assert series_workers._send_rows(rows(), sample_keys, Receiver()) == 3
    received = [pickle.loads(payload) for batch in batches for payload in batch.rows]
    assert [row.features[0].value for row in received] == [
        [0, None],
        [1, None],
        [2, None],
    ]
    assert [row.key for row in received] == [(time, "A")] * 3
    assert all(batch.key_types == ("string",) for batch in batches)
