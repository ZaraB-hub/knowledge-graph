import argparse
import csv
import gzip
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "results" / "distmult"
MODEL_FILE = MODEL_DIR / "trained_model.pkl"
TRAINING_TRIPLES_DIR = MODEL_DIR / "training_triples"
RECIPES_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.json"
OUTPUT_FILE = MODEL_DIR / "recommendations.csv"


def load_recipe_embeddings():
    model = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)

    with gzip.open(
        TRAINING_TRIPLES_DIR / "entity_to_id.tsv.gz",
        "rt",
        encoding="utf-8",
    ) as mapping_file:
        entity_to_id = {
            row["label"]: int(row["id"])
            for row in csv.DictReader(mapping_file, delimiter="\t")
        }

    with RECIPES_FILE.open("r", encoding="utf-8") as recipes_file:
        recipes = json.load(recipes_file)

    recipe_name_by_id = {
        f"recipe:{recipe['id']}": recipe["name"] for recipe in recipes
    }
    recipe_id_by_name = {
        name: recipe_id for recipe_id, name in recipe_name_by_id.items()
    }
    recipe_entity_ids = sorted(
        entity_id
        for entity_id in entity_to_id
        if entity_id.startswith("recipe:") and entity_id in recipe_name_by_id
    )

    recipe_tensor_ids = torch.tensor(
        [entity_to_id[entity_id] for entity_id in recipe_entity_ids],
        dtype=torch.long,
    )
    embeddings = model.entity_representations[0](indices=recipe_tensor_ids).detach()

    return recipe_id_by_name, recipe_name_by_id, recipe_entity_ids, embeddings


def recommend(recipe_name, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    recipe_id_by_name, recipe_name_by_id, recipe_entity_ids, embeddings = (
        load_recipe_embeddings()
    )
    if recipe_name not in recipe_id_by_name:
        raise ValueError(f"Recipe not found: {recipe_name}")

    source_id = recipe_id_by_name[recipe_name]
    source_index = recipe_entity_ids.index(source_id)
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    similarities = normalized_embeddings @ normalized_embeddings[source_index]
    ranked_indices = torch.argsort(similarities, descending=True)

    recommendations = []
    for candidate_index in ranked_indices.tolist():
        candidate_id = recipe_entity_ids[candidate_index]
        if candidate_id == source_id:
            continue

        recommendations.append(
            {
                "recipe": recipe_name,
                "rank": len(recommendations) + 1,
                "recommended_recipe": recipe_name_by_id[candidate_id],
                "similarity": float(similarities[candidate_index]),
            }
        )
        if len(recommendations) == top_k:
            break

    return recommendations


def save_recommendations(recommendations):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["recipe", "rank", "recommended_recipe", "similarity"],
        )
        writer.writeheader()
        writer.writerows(recommendations)


def main():
    parser = argparse.ArgumentParser(
        description="Recommend recipes using saved DistMult embeddings."
    )
    parser.add_argument("recipe", help="Exact recipe name from recipes.json")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()

    recommendations = recommend(args.recipe, top_k=args.top_k)
    save_recommendations(recommendations)

    print(f"Recipe: {args.recipe}\n")
    for recommendation in recommendations:
        print(
            f"{recommendation['rank']}. "
            f"{recommendation['recommended_recipe']} - "
            f"similarity: {recommendation['similarity']:.4f}"
        )
    print(f"\nRecommendations saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()