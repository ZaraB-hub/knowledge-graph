from pathlib import Path

from pykeen.triples import TriplesFactory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIPLES_FILE = PROJECT_ROOT / "data" / "processed" / "triples.tsv"


def main():
    triples_factory = TriplesFactory.from_path(
        TRIPLES_FILE,
        delimiter="\t",
    )

    training, validation, testing = triples_factory.split(
        ratios=[0.8, 0.1, 0.1],
        random_state=42,
    )

    print(f"Total triples: {triples_factory.mapped_triples.shape[0]}")
    print(f"Training triples: {training.mapped_triples.shape[0]}")
    print(f"Validation triples: {validation.mapped_triples.shape[0]}")
    print(f"Testing triples: {testing.mapped_triples.shape[0]}")


if __name__ == "__main__":
    main()