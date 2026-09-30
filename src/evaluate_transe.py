import json
from pathlib import Path

import torch
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "results" / "transe"
MODEL_FILE = MODEL_DIR / "trained_model.pkl"
TRAINING_DIR = MODEL_DIR / "training_triples"
VALIDATION_DIR = MODEL_DIR / "validation_triples"
TESTING_DIR = MODEL_DIR / "testing_triples"
OUTPUT_FILE = MODEL_DIR / "evaluation_results.json"


def load_artifacts():
    required_paths = [MODEL_FILE, TRAINING_DIR, VALIDATION_DIR, TESTING_DIR]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing saved TransE artifacts:\n{missing}")

    model = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    training = TriplesFactory.from_path_binary(TRAINING_DIR)
    validation = TriplesFactory.from_path_binary(VALIDATION_DIR)
    testing = TriplesFactory.from_path_binary(TESTING_DIR)
    return model, training, validation, testing


def main():
    model, training, validation, testing = load_artifacts()
    evaluator = RankBasedEvaluator()

    # Include the known splits so filtered ranking does not penalize known facts.
    metric_results = evaluator.evaluate(
        model=model,
        mapped_triples=testing.mapped_triples,
        additional_filter_triples=[
            training.mapped_triples,
            validation.mapped_triples,
            testing.mapped_triples,
        ],
        use_tqdm=False,
    )

    metrics = {
        "test_triples": int(testing.mapped_triples.shape[0]),
        "mean_rank": metric_results.get_metric("mean_rank"),
        "mrr": metric_results.get_metric("inverse_harmonic_mean_rank"),
        "hits_at_1": metric_results.get_metric("hits_at_1"),
        "hits_at_3": metric_results.get_metric("hits_at_3"),
        "hits_at_10": metric_results.get_metric("hits_at_10"),
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2)

    print(f"Test triples: {metrics['test_triples']}")
    print(f"Mean Rank: {metrics['mean_rank']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Hits@1: {metrics['hits_at_1']:.4f}")
    print(f"Hits@3: {metrics['hits_at_3']:.4f}")
    print(f"Hits@10: {metrics['hits_at_10']:.4f}")
    print(f"Evaluation saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()