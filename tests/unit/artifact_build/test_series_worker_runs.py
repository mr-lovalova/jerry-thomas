from datetime import datetime, timezone

import pytest

from jerrythomas.operations.artifacts.series import _ProjectedRow, _ProjectedScalar
from jerrythomas.pipelines.sort import merge_sort_runs, write_sort_runs


@pytest.mark.parametrize("buffer_bytes", [1, 1024 * 1024])
def test_worker_runs_snapshot_reused_values_before_advancing(tmp_path, buffer_bytes):
    time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    values = []

    def rows():
        for value in range(3):
            values[:] = [value, None]
            yield _ProjectedRow(
                key=(time, "A"),
                time=time,
                features=(_ProjectedScalar("value", values, True),),
                targets=(),
            )
        values.clear()

    def key(row):
        return row.key, row.time

    runs = write_sort_runs(rows(), buffer_bytes, key, tmp_path)
    assert runs.rows == 3
    assert len(runs.paths) == (3 if buffer_bytes == 1 else 1)
    received = list(merge_sort_runs(runs, key, tmp_path))
    assert [row.features[0].value for row in received] == [
        [0, None],
        [1, None],
        [2, None],
    ]
    assert [row.key for row in received] == [(time, "A")] * 3
