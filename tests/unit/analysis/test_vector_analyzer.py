import json

import pytest

from jerrythomas.analysis.vector.matrix import MatrixBuilder, render_matrix_html
from jerrythomas.analysis.vector.coverage_stats import CoverageStatsAccumulator
from jerrythomas.artifacts.models import (
    ListCoverageColumnStats,
    ListVectorMetadataEntry,
    ScalarVectorMetadataEntry,
    CoverageStatsArtifact,
)
from jerrythomas.artifacts.registry import COVERAGE_STATS_SPEC
from jerrythomas.operations.runtime.coverage import _section_report


def _scalar(identifier: str, base_id: str):
    return ScalarVectorMetadataEntry(
        id=identifier,
        base_id=base_id,
        kind="scalar",
        present_count=1,
        null_count=0,
        value_types=("float",),
    )


def _sequence(identifier: str, base_id: str, length: int):
    return ListVectorMetadataEntry(
        id=identifier,
        base_id=base_id,
        kind="list",
        present_count=1,
        null_count=0,
        element_types=("float", "null"),
        length=length,
        observed_elements=1,
    )


def test_coverage_stats_accumulator_keeps_only_bounded_counters() -> None:
    accumulator = CoverageStatsAccumulator(
        (
            _scalar("speed__@station:A", "speed"),
            _scalar("speed__@station:B", "speed"),
            _sequence("history", "history", 2),
        )
    )

    accumulator.update(
        {
            "speed__@station:A": None,
            "speed__@station:B": 7.0,
            "history": [1.0, None],
        }
    )
    accumulator.update({})
    section = accumulator.finish()

    speed = next(entry for entry in section.bases if entry.id == "speed")
    assert speed.present_samples == 1
    assert speed.non_null_samples == 1
    station_a = next(
        entry for entry in section.columns if entry.id == "speed__@station:A"
    )
    assert station_a.present_samples == 1
    assert station_a.non_null_samples == 0
    history = next(entry for entry in section.columns if entry.id == "history")
    assert history.present_samples == 1
    assert history.non_null_samples == 1
    assert history.observed_elements == 1

    payload = section.model_dump(mode="json")
    assert "groups" not in json.dumps(payload)
    assert not hasattr(accumulator, "group_feature_status")


def test_coverage_stats_accumulator_rejects_metadata_drift() -> None:
    accumulator = CoverageStatsAccumulator((_scalar("speed", "speed"),))

    with pytest.raises(ValueError, match="missing from metadata"):
        accumulator.update({"temperature": 4.0})


def test_coverage_stats_accumulator_retains_declared_zero_count_columns() -> None:
    accumulator = CoverageStatsAccumulator(
        (_scalar("kept", "kept"), _scalar("removed", "removed"))
    )
    accumulator.update({"kept": None})

    section = accumulator.finish()

    assert [entry.id for entry in section.bases] == ["kept", "removed"]
    assert [entry.id for entry in section.columns] == ["kept", "removed"]
    removed = section.columns[1]
    assert removed.present_samples == 0
    assert removed.non_null_samples == 0


def test_coverage_stats_v3_rejects_old_snapshots_and_impossible_list_counts() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        CoverageStatsArtifact.model_validate({"schema_version": 2})

    with pytest.raises(ValueError, match="cannot exceed observed_elements"):
        ListCoverageColumnStats(
            id="history",
            base_id="history",
            kind="list",
            present_samples=2,
            non_null_samples=2,
            length=3,
            observed_elements=1,
        )

    with pytest.raises(ValueError, match=r"present_samples \* length"):
        ListCoverageColumnStats(
            id="history",
            base_id="history",
            kind="list",
            present_samples=1,
            non_null_samples=1,
            length=2,
            observed_elements=3,
        )


def test_coverage_stats_loader_rejects_v2_with_rebuild_message(tmp_path) -> None:
    path = tmp_path / "coverage_stats.json"
    path.write_text('{"schema_version": 2}', encoding="utf-8")

    with pytest.raises(ValueError, match="Rebuild coverage_stats in FORCE mode"):
        COVERAGE_STATS_SPEC.loader(path)


def test_coverage_counts_nulls_and_missing_elements_as_uncovered() -> None:
    accumulator = CoverageStatsAccumulator(
        (_scalar("speed", "speed"), _sequence("history", "history", 2))
    )
    accumulator.update({"speed": None, "history": [1.0, None]})
    accumulator.update({})

    report = _section_report(accumulator.finish(), total_samples=2, threshold=0.8)
    speed = next(metric for metric in report["columns"] if metric["id"] == "speed")
    history = next(metric for metric in report["columns"] if metric["id"] == "history")

    assert speed["coverage"] == 0.0
    assert speed["absent_samples"] == 1
    assert speed["null_samples"] == 1
    assert history["coverage"] == 0.25
    assert history["observed_elements"] == 1
    assert history["element_opportunities"] == 4
    assert report["below_threshold_columns"] == ["speed", "history"]


def test_matrix_is_bounded_by_rendered_cells_and_preserves_duplicate_labels() -> None:
    builder = MatrixBuilder(
        (_scalar("speed", "speed"), _sequence("history", "history", 2)),
        (),
        max_cells=6,
    )
    builder.add("same", {"speed": 1.0, "history": [1.0, None]}, {})
    builder.add("same", {"speed": None}, {})

    with pytest.raises(ValueError, match="exceeds max_cells=6 at sample 3"):
        builder.add("third", {}, {})

    matrix = builder.finish()
    assert [row.group for row in matrix.rows] == ["same", "same"]
    assert matrix.rows[0].features[1].status == "present"
    assert matrix.rows[0].features[1].elements == ("present", "null")
    assert matrix.rows[1].features[1].status == "absent"


def test_matrix_rows_separate_features_and_targets() -> None:
    builder = MatrixBuilder(
        (_scalar("speed", "speed"),),
        (_scalar("return", "return"),),
        max_cells=2,
    )
    builder.add("g0", {"speed": 1.0}, {"return": None})

    assert list(builder.finish().output_rows()) == [
        {
            "vector": "feature",
            "identifier": "speed",
            "group": "g0",
            "status": "present",
        },
        {
            "vector": "target",
            "identifier": "return",
            "group": "g0",
            "status": "null",
        },
    ]


def test_matrix_html_uses_bounded_matrix_data_and_escapes_script_text() -> None:
    builder = MatrixBuilder((_scalar("speed", "speed"),), (), max_cells=1)
    builder.add("</script>", {"speed": 1.0}, {})

    document = render_matrix_html(builder.finish())
    assert "<h1>Availability Matrix</h1>" in document
    assert "Feature Availability" in document
    assert "Target Availability" in document
    assert "setupMatrix('features'" in document
    assert '"</script>"' not in document
