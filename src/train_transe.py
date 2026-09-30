from pathlib import Path

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIPLES_FILE = PROJECT_ROOT / "data" / "processed" / "triples.tsv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "results" / "transe"
TRAINING_OUTPUT_DIR = MODEL_OUTPUT_DIR / "training_triples"
VALIDATION_OUTPUT_DIR = MODEL_OUTPUT_DIR / "validation_triples"
TESTING_OUTPUT_DIR = MODEL_OUTPUT_DIR / "testing_triples"


def main():
    triples_factory = TriplesFactory.from_path(
        TRIPLES_FILE,
        delimiter="\t",
    )

    training, validation, testing = triples_factory.split(
        ratios=[0.8, 0.1, 0.1],
        random_state=42,
    )

    training.to_path_binary(TRAINING_OUTPUT_DIR)
    validation.to_path_binary(VALIDATION_OUTPUT_DIR)
    testing.to_path_binary(TESTING_OUTPUT_DIR)

    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model="TransE",
        model_kwargs={
            "embedding_dim": 128,
        },
        training_kwargs={
            "num_epochs": 300,
            "use_tqdm": False,
        },
        optimizer="Adam",
        optimizer_kwargs={
            "lr": 0.001,
        },
        random_seed=42,
    )

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(MODEL_OUTPUT_DIR)

    print(f"Training triples: {training.mapped_triples.shape[0]}")
    print(f"Validation triples: {validation.mapped_triples.shape[0]}")
    print(f"Testing triples: {testing.mapped_triples.shape[0]}")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print(f"Training split saved to: {TRAINING_OUTPUT_DIR}")
    print(f"Validation split saved to: {VALIDATION_OUTPUT_DIR}")
    print(f"Testing split saved to: {TESTING_OUTPUT_DIR}")


if __name__ == "__main__":
    main()