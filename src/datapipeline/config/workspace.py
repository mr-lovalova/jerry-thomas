from pydantic import BaseModel, ConfigDict, Field

from datapipeline.config.constraints import NonEmptyString as WorkspacePath


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_root: WorkspacePath | None = None
    datasets: dict[str, WorkspacePath] = Field(
        default_factory=dict,
        description="Named dataset aliases mapping to project.yaml paths (relative to jerry.yaml).",
    )
    default_dataset: WorkspacePath | None = Field(
        default=None,
        description="Optional default dataset alias when --dataset/--project are omitted.",
    )
