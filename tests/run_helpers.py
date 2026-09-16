from jerrythomas.io.recipes import RunRecipe


def empty_recipe(command="serve") -> RunRecipe:
    """Minimal explicit recipe for low-level receipt tests."""
    return RunRecipe(
        command=command,
        project="project.yaml",
        configuration={"artifacts": {}},
        implementation={},
        inputs={},
    )
