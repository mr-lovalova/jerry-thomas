"""Self-contained execution recipes, stored beside run receipts."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jerrythomas.io.json_file import write_json_object


class RunRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    command: Literal["serve", "export"]
    project: str
    dataset_id: str | None = None
    dataset_version: str | None = None
    configuration: dict[str, Any]
    implementation: dict[str, Any]
    inputs: dict[str, Any]
    artifacts: dict[str, Any] = Field(default_factory=dict)
    limitations: tuple[str, ...] = ()


class RecipeReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value or value in {".", ".."} or any(c in value for c in "/\\:\x00"):
            raise ValueError("recipe path must be an adjacent filename")
        return value


def recipe_path(receipt: Path) -> Path:
    name = (
        "recipe.json"
        if receipt.name == "run.json"
        else (receipt.name.removesuffix(".run.json") + ".recipe.json")
    )
    return receipt.with_name(name)


def recipe_reference(path: Path, recipe: RunRecipe) -> RecipeReference:
    content = json.dumps(
        recipe.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    return RecipeReference(path=path.name, sha256=hashlib.sha256(content).hexdigest())


def save_recipe(path: Path, recipe: RunRecipe, *, overwrite: bool) -> None:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Recipe must be a regular file: {path}")
    write_json_object(path, recipe.model_dump(mode="json"), overwrite=overwrite)


def load_recipe(directory: Path, reference: RecipeReference) -> RunRecipe:
    path = directory / reference.path
    if path.is_symlink():
        raise ValueError("Saved recipe must not be a symlink")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise ValueError(f"Saved recipe checksum mismatch: {path}")
    return RunRecipe.model_validate_json(content)
