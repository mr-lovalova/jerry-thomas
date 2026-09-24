import json
from pathlib import Path

import pytest
import yaml

from jerrythomas.execution.settings import CommandObservability
from jerrythomas.profiles.errors import ProfileCommandError
from jerrythomas.profiles.orchestration import run_profiles
from jerrythomas.profiles.request_builder import build_runtime_run_request
from jerrythomas.services.project_definition import load_project_definition

from tests.named_dataset_helpers import create_project, write_config


QUIET = CommandObservability(visuals=False, log_level="CRITICAL")


def _read_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _reference_sections(
    root: Path, dataset_id: str, sections: dict, directory: str = "shared"
) -> None:
    path = root / "datasets" / f"{dataset_id}.yaml"
    dataset = _read_config(path)
    for section, content in sections.items():
        relative = f"{directory}/{section}.yaml"
        write_config(root / relative, content)
        dataset[section] = {"file": relative}
    write_config(path, dataset)


def test_shared_sections_preserve_hashes_outputs_and_captured_recipe(tmp_path):
    project = create_project(tmp_path)
    dataset_path = tmp_path / "datasets" / "alpha.yaml"
    dataset = _read_config(dataset_path)
    dataset["features"].append({"id": "other", "stream": "rows", "field": "other"})
    dataset["postprocess"] = {"features": {"threshold": 1.0}}
    write_config(dataset_path, dataset)
    # The shared postprocess block must actually filter a row in this fixture.
    source = tmp_path / "rows.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    next(row for row in rows if row["value"] == 2.0)["other"] = None
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    inline = build_runtime_run_request(
        "serve",
        str(project),
        profile_name="alpha",
        artifact_mode="rebuild",
        command_observability=QUIET,
    )
    assert inline is not None
    (original,) = run_profiles(inline)
    original_rows = [path.read_bytes() for path in original.outputs]
    assert [output.row_count for output in original.metadata.outputs] == [1, 1]
    resolved_dataset = original.load_recipe().configuration["datasets"]["alpha"]
    sections = {name: dataset[name] for name in ("sample", "split", "postprocess")}
    _reference_sections(tmp_path, "alpha", sections)

    referenced = build_runtime_run_request(
        "serve",
        str(project),
        profile_name="alpha",
        artifact_mode="rebuild",
        command_observability=QUIET,
    )
    assert referenced is not None
    assert referenced.definition.datasets == inline.definition.datasets
    assert referenced.definition.artifact_hashes == inline.definition.artifact_hashes
    # Execution and recipes use the captured definition, not later edits to files.
    for name in sections:
        write_config(
            tmp_path / "shared" / f"{name}.yaml", {"changed_after_request": True}
        )
    (saved,) = run_profiles(referenced)
    assert [path.read_bytes() for path in saved.outputs] == original_rows
    assert saved.load_recipe().configuration["datasets"]["alpha"] == resolved_dataset

    _reference_sections(tmp_path, "alpha", sections, directory="moved/policies")
    relocated = load_project_definition(project)
    assert relocated.datasets == inline.definition.datasets
    assert relocated.artifact_hashes == inline.definition.artifact_hashes


@pytest.mark.parametrize(
    ("section", "changed_kinds"),
    [
        ("sample", {"series", "scaler", "metadata", "coverage_stats"}),
        ("split", {"scaler", "metadata", "coverage_stats"}),
        ("postprocess", {"coverage_stats"}),
    ],
)
def test_shared_section_changes_invalidate_only_affected_consumers(
    tmp_path, section, changed_kinds
):
    project = create_project(tmp_path)
    alpha = _read_config(tmp_path / "datasets" / "alpha.yaml")
    blocks = {
        "sample": alpha["sample"],
        "split": alpha["split"],
        "postprocess": {"features": {"threshold": 1.0}},
    }
    beta = _read_config(tmp_path / "datasets" / "beta.yaml")
    beta.update(blocks)
    write_config(tmp_path / "datasets" / "beta.yaml", beta)
    # Gamma has equal inline settings but does not consume the shared file.
    write_config(tmp_path / "datasets" / "gamma.yaml", beta)
    for dataset_id in ("alpha", "beta"):
        _reference_sections(tmp_path, dataset_id, {section: blocks[section]})
    before = load_project_definition(project).artifact_hashes.values

    changed = blocks[section]
    if section == "sample":
        changed["cadence"] = "12h"
    elif section == "split":
        changed["intervals"][0]["until"] = "2024-01-02T00:00:00Z"
    else:
        changed["features"]["threshold"] = 0.5
    write_config(tmp_path / "shared" / f"{section}.yaml", changed)
    after = load_project_definition(project).artifact_hashes.values

    assert after.keys() == before.keys()
    assert {key for key in before if before[key] != after[key]} == {
        f"dataset.{dataset_id}.{kind}"
        for dataset_id in ("alpha", "beta")
        for kind in changed_kinds
    }


@pytest.mark.parametrize("invalid", ["shared_sample", "inline_version", "duplicate_id"])
def test_shared_section_request_errors_preserve_context_without_exposing_values(
    tmp_path, monkeypatch, invalid
):
    secret = "sensitive-example/value"
    monkeypatch.setenv("JERRY_TEST_SECRET", secret)
    reference = "${env:JERRY_TEST_SECRET}"
    project = create_project(tmp_path)
    dataset_path = tmp_path / "datasets" / "alpha.yaml"
    dataset = _read_config(dataset_path)
    sample = dataset["sample"]
    if invalid == "shared_sample":
        sample["cadence"] = reference
    elif invalid == "inline_version":
        dataset["version"] = reference
    else:
        # The duplicate-ID validator embeds the resolved ID in its own message.
        feature = {**dataset["features"][0], "id": reference}
        dataset["features"] = [feature, feature.copy()]
    write_config(dataset_path, dataset)
    _reference_sections(tmp_path, "alpha", {"sample": sample})

    with pytest.raises(ProfileCommandError) as error:
        build_runtime_run_request(
            "serve", str(project), profile_name="alpha", command_observability=QUIET
        )

    message = str(error.value)
    assert str(dataset_path) in message
    assert "sample" in message
    assert str(tmp_path / "shared" / "sample.yaml") in message
    assert "validation error" in message
    assert secret not in message
    assert "input_value" not in message
    assert "input_type" not in message
