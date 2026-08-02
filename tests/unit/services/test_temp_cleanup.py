import os
from datetime import timedelta

import pytest

from jerrythomas.pipelines.sort import batch_sort
from jerrythomas.services.temp_cleanup import (
    clean_temp_dirs,
    find_temp_dirs,
    format_bytes,
    parse_age,
)


def test_find_temp_dirs_only_matches_sort_prefixes(tmp_path) -> None:
    cache_dir = tmp_path / "datapipeline-cache-project-a"
    sort_dir = tmp_path / "datapipeline-sort-run-a"
    other_dir = tmp_path / "other"
    cache_dir.mkdir()
    sort_dir.mkdir()
    other_dir.mkdir()
    (cache_dir / "data.bin").write_bytes(b"abc")
    (sort_dir / "data.bin").write_bytes(b"abcd")

    found = find_temp_dirs(root=tmp_path)

    assert [item.path.name for item in found] == [
        "datapipeline-sort-run-a",
    ]
    assert sum(item.size_bytes for item in found) == 4


def test_clean_temp_dirs_is_dry_run_by_default(tmp_path) -> None:
    sort_dir = tmp_path / "datapipeline-sort-run-a"
    sort_dir.mkdir()

    result = clean_temp_dirs(yes=False, root=tmp_path)

    assert result.dry_run is True
    assert result.removed == ()
    assert sort_dir.exists()


def test_clean_temp_dirs_removes_matches_with_yes(tmp_path) -> None:
    sort_dir = tmp_path / "datapipeline-sort-run-a"
    sort_dir.mkdir()

    result = clean_temp_dirs(yes=True, root=tmp_path)

    assert result.dry_run is False
    assert result.removed == (sort_dir,)
    assert not sort_dir.exists()


def test_clean_temp_dirs_does_not_remove_active_sort(tmp_path) -> None:
    rows = batch_sort(
        [3, 2, 1],
        buffer_bytes=1,
        key=lambda value: value,
        spill_dir=tmp_path,
    )
    assert next(rows) == 1
    spill_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(spill_dirs) == 1

    result = clean_temp_dirs(yes=True, root=tmp_path)

    assert result.candidates == ()
    assert result.removed == ()
    assert spill_dirs[0].is_dir()

    rows.close()
    assert list(tmp_path.iterdir()) == []


def test_find_temp_dirs_respects_age_filter(tmp_path) -> None:
    old_dir = tmp_path / "datapipeline-sort-old"
    new_dir = tmp_path / "datapipeline-sort-new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_time = 1_700_000_000
    os.utime(old_dir, (old_time, old_time))

    found = find_temp_dirs(root=tmp_path, older_than=timedelta(days=1))

    assert [item.path.name for item in found] == ["datapipeline-sort-old"]


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("30m", 1800),
        ("2h", 7200),
        ("1d", 86400),
        ("3", 10800),
    ],
)
def test_parse_age(value: str, seconds: int) -> None:
    assert parse_age(value).total_seconds() == seconds


@pytest.mark.parametrize("value", ["soon", "inf", "-inf", "nan", "1e308h"])
def test_parse_age_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="age must"):
        parse_age(value)


def test_format_bytes() -> None:
    assert format_bytes(0) == "0B"
    assert format_bytes(1024) == "1.0KB"
