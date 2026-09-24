from collections import Counter
from pathlib import Path

import pytest
import yaml

from jerrythomas.config.dataset.dataset import DatasetConfig
from jerrythomas.io.yaml import read_yaml_document
from jerrythomas.services.dataset import dataset_from_document, load_datasets
from jerrythomas.services.definitions import ProjectManifest
from jerrythomas.services.project import load_project


_SAMPLE = {"cadence": "1d", "rounding": "exact", "keys": ["id_"]}
_SPLIT = {
    "mode": "hash",
    "ratios": {"train": 0.75, "test": 0.25},
    "folds": [{"id": "holdout", "train": ["train"], "test": ["test"]}],
}
_POSTPROCESS = {"features": {"threshold": 1.0, "ids": ["price"]}}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _project(
    root: Path,
    *,
    datasets: str = "datasets",
    globals_: dict | None = None,
) -> ProjectManifest:
    for directory in ("sources", "streams"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / datasets).mkdir(parents=True, exist_ok=True)
    project = root / "project.yaml"
    _write(
        project,
        {
            "schema_version": 7,
            "artifact_revision": 1,
            "paths": {
                "sources": "sources",
                "streams": "streams",
                "datasets": datasets,
                "artifacts": "artifacts",
            },
            "globals": globals_ or {},
        },
    )
    return load_project(project)


def _dataset(**sections: object) -> dict:
    return {
        "version": "v1",
        "sample": _SAMPLE,
        "features": [{"id": "price", "stream": "prices", "field": "value"}],
        **sections,
    }


def test_inline_dataset_sections_keep_the_existing_typed_configuration(tmp_path):
    project = _project(tmp_path)
    raw = _dataset(split=_SPLIT, postprocess=_POSTPROCESS)
    _write(tmp_path / "datasets" / "alpha.yaml", raw)

    assert load_datasets(project)["alpha"] == DatasetConfig.model_validate(raw)


def test_section_paths_are_project_relative_even_when_datasets_live_elsewhere(
    tmp_path, monkeypatch
):
    root = tmp_path / "project"
    project = _project(root, datasets="../definitions")
    _write(root / "blocks" / "sample.yaml", _SAMPLE)
    _write(tmp_path / "shared" / "split.yaml", _SPLIT)
    postprocess = tmp_path / "shared" / "postprocess.yaml"
    _write(postprocess, _POSTPROCESS)
    _write(
        tmp_path / "definitions" / "alpha.yaml",
        _dataset(
            sample={"file": "blocks/sample.yaml"},
            split={"file": "../shared/split.yaml"},
            postprocess={"file": str(postprocess)},
        ),
    )
    unrelated_cwd = tmp_path / "elsewhere"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    resolved = load_datasets(project)["alpha"]

    assert resolved == DatasetConfig.model_validate(
        _dataset(split=_SPLIT, postprocess=_POSTPROCESS)
    )
    assert "file" not in resolved.model_dump_json()


def test_standalone_document_loading_resolves_dataset_section_files(tmp_path):
    project = _project(tmp_path)
    _write(tmp_path / "blocks" / "sample.yaml", _SAMPLE)
    path = tmp_path / "elsewhere" / "standalone.yaml"
    _write(path, _dataset(sample={"file": "blocks/sample.yaml"}))

    result = dataset_from_document(project, read_yaml_document(path))

    assert result == DatasetConfig.model_validate(_dataset())


def test_shared_sections_are_read_and_resolved_once_per_load_without_shared_mutation(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    blocks = {"sample": _SAMPLE, "split": _SPLIT, "postprocess": _POSTPROCESS}
    paths = {name: tmp_path / "blocks" / f"{name}.yaml" for name in blocks}
    for name, block in blocks.items():
        _write(paths[name], block)
    for dataset_id, directory in (("alpha", "blocks"), ("beta", "blocks/../blocks")):
        _write(
            tmp_path / "datasets" / f"{dataset_id}.yaml",
            _dataset(**{name: {"file": f"{directory}/{name}.yaml"} for name in blocks}),
        )
    reads = Counter()
    resolutions = Counter()
    read_bytes = Path.read_bytes
    resolve_config = ProjectManifest.resolve_config

    def record_read(path):
        if path.resolve() in paths.values():
            reads[path.resolve()] += 1
        return read_bytes(path)

    def record_resolution(manifest, value, **kwargs):
        for name, block in blocks.items():
            if value == block:
                resolutions[name] += 1
        return resolve_config(manifest, value, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", record_read)
    monkeypatch.setattr(ProjectManifest, "resolve_config", record_resolution)
    datasets = load_datasets(project)

    assert reads == Counter({path: 1 for path in paths.values()})
    assert resolutions == Counter({name: 1 for name in blocks})
    alpha, beta = datasets["alpha"], datasets["beta"]
    assert alpha == beta
    alpha.sample.keys.append("mutated")
    alpha.split.folds[0].train.append("mutated")
    alpha.postprocess.features.ids.append("mutated")
    assert beta.sample.keys == ["id_"]
    assert beta.split.folds[0].train == ["train"]
    assert beta.postprocess.features.ids == ["price"]

    _write(paths["sample"], {**_SAMPLE, "cadence": "1h"})
    refreshed = load_datasets(project)
    assert reads == Counter({path: 2 for path in paths.values()})
    assert refreshed["alpha"].sample.cadence == "1h"
    assert refreshed["alpha"].sample.keys == ["id_"]
    assert refreshed["alpha"].split.folds[0].train == ["train"]
    assert refreshed["alpha"].postprocess.features.ids == ["price"]


def test_shared_file_uses_each_consuming_projects_captured_globals_and_environment(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("JERRY_SECTION_CADENCE", raising=False)
    shared = tmp_path / "shared" / "sample.yaml"
    _write(
        shared,
        {
            "cadence": "${env:JERRY_SECTION_CADENCE}",
            "rounding": "${rounding}",
            "keys": "${entity_keys}",
        },
    )
    (shared.parent / ".env").write_text("JERRY_SECTION_CADENCE=2w\n")
    manifests = []
    for name, cadence, key in (("first", "1d", "id_"), ("second", "1h", "asset")):
        root = tmp_path / name
        root.mkdir()
        (root / ".env").write_text(f"JERRY_SECTION_CADENCE={cadence}\n")
        project = _project(
            root,
            globals_={
                "rounding": "exact",
                "entity_keys": [key],
                "section_file": "../shared/sample.yaml",
            },
        )
        _write(
            root / "datasets" / "alpha.yaml",
            _dataset(sample={"file": "${section_file}"}),
        )
        manifests.append(project)
    monkeypatch.setenv("JERRY_SECTION_CADENCE", "7d")

    first, second = (load_datasets(project)["alpha"] for project in manifests)

    assert first.sample.cadence == "1d"
    assert first.sample.keys == ["id_"]
    assert second.sample.cadence == "1h"
    assert second.sample.keys == ["asset"]


@pytest.mark.parametrize(
    "reference",
    [
        {"file": ""},
        {"file": "   "},
        {"file": 7},
        {"file": "blocks/sample.yaml", "cadence": "1h"},
    ],
)
def test_section_reference_requires_one_nonblank_filename(tmp_path, reference):
    project = _project(tmp_path)
    _write(tmp_path / "blocks" / "sample.yaml", _SAMPLE)
    dataset_path = tmp_path / "datasets" / "alpha.yaml"
    _write(dataset_path, _dataset(sample=reference))

    with pytest.raises((ValueError, TypeError)) as exc:
        load_datasets(project)

    assert "alpha" in str(exc.value)
    assert "sample" in str(exc.value)
    assert "file" in str(exc.value)


@pytest.mark.parametrize(
    "content",
    [None, "cadence: [\n", "- not\n- a\n- mapping\n", "file: other.yaml\n"],
    ids=["missing", "malformed", "nonmapping", "reference-chain"],
)
def test_bad_section_files_report_dataset_section_and_path(tmp_path, content):
    project = _project(tmp_path)
    section = tmp_path / "blocks" / "sample.yaml"
    if content is not None:
        section.parent.mkdir()
        section.write_text(content)
    _write(
        tmp_path / "datasets" / "alpha.yaml",
        _dataset(sample={"file": "blocks/sample.yaml"}),
    )

    with pytest.raises((ValueError, TypeError, OSError)) as exc:
        load_datasets(project)

    message = str(exc.value)
    assert "alpha" in message
    assert "sample" in message
    assert str(section) in message


@pytest.mark.parametrize(
    ("section", "body", "invalid_field"),
    [
        ("sample", {**_SAMPLE, "cadence": "invalid"}, "cadence"),
        ("split", {**_SPLIT, "ratios": {"train": 0.1, "test": 0.2}}, "ratios"),
        ("postprocess", {"features": {"threshold": 2.0}}, "threshold"),
    ],
)
def test_referenced_sections_keep_typed_validation_and_file_context(
    tmp_path, section, body, invalid_field
):
    project = _project(tmp_path)
    path = tmp_path / "blocks" / f"{section}.yaml"
    _write(path, body)
    _write(
        tmp_path / "datasets" / "alpha.yaml",
        _dataset(**{section: {"file": str(path)}}),
    )

    with pytest.raises(ValueError) as exc:
        load_datasets(project)

    message = str(exc.value)
    assert "alpha" in message
    assert section in message
    assert str(path) in message
    assert invalid_field in message


@pytest.mark.parametrize("case", ["sequence", "future-target", "target-filter"])
def test_expanded_sections_preserve_cross_section_dataset_validation(tmp_path, case):
    project = _project(tmp_path)
    dataset = _dataset()
    if case == "target-filter":
        _write(
            tmp_path / "blocks" / "postprocess.yaml", {"targets": {"threshold": 1.0}}
        )
        dataset["postprocess"] = {"file": "blocks/postprocess.yaml"}
        expected_error = "postprocess.targets requires"
    else:
        _write(tmp_path / "blocks" / "split.yaml", _SPLIT)
        dataset["split"] = {"file": "blocks/split.yaml"}
        if case == "sequence":
            dataset["features"][0]["sequence"] = {"size": 2}
            expected_error = "hash splits cannot be used with sequenced"
        else:
            dataset["targets"] = [
                {"id": "future", "stream": "returns", "field": "value", "horizon": "1d"}
            ]
            expected_error = "hash splits cannot be used with positive target horizons"
    _write(tmp_path / "datasets" / "alpha.yaml", dataset)

    with pytest.raises(ValueError, match=expected_error):
        load_datasets(project)
