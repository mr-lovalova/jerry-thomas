from pydantic import BaseModel, ConfigDict, Field

from jerrythomas.config.constraints import NonEmptyString as WorkspacePath


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin_root: WorkspacePath | None = None
    projects: dict[str, WorkspacePath] = Field(
        default_factory=dict,
        description="Named project aliases mapping to project.yaml paths (relative to jerry.yaml).",
    )
    default_project: WorkspacePath | None = Field(
        default=None,
        description="Optional default project alias when --project is omitted.",
    )
