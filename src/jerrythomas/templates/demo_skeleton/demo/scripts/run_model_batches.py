from pathlib import Path

from jerrythomas.integrations.ml import iter_model_batches


def main() -> None:
    project = Path(__file__).resolve().parents[1] / "project.yaml"
    batch = next(
        iter_model_batches(
            project,
            dataset="default",
            output_id="holdout.train",
            batch_size=32,
            dtype="float32",
        )
    )
    print("Keys:", batch.keys)
    print("Features:", batch.feature_columns, batch.features.shape)
    target_shape = None if batch.targets is None else batch.targets.shape
    print("Targets:", batch.target_columns, target_shape)


if __name__ == "__main__":
    main()
