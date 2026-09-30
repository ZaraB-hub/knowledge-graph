import csv
import gzip
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# chenge model dir
MODEL_DIR = PROJECT_ROOT / "results" / "distmult"
MODEL_FILE = MODEL_DIR / "trained_model.pkl"
TRAINING_TRIPLES_DIR = MODEL_DIR / "training_triples"
OUTPUT_FILE = MODEL_DIR / "recipe_similarity.csv"
RECIPES_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.json"


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
        f"recipe:{recipe['id']}": recipe["name"]
        for recipe in recipes
    }

    recipe_entity_ids = [
        entity_name
        for entity_name in entity_to_id
        if entity_name.startswith("recipe:")
    ]
    recipe_entity_ids.sort()

    recipe_tensor_ids = torch.tensor(
        [entity_to_id[entity_name] for entity_name in recipe_entity_ids],
        dtype=torch.long,
    )
    embeddings = model.entity_representations[0](indices=recipe_tensor_ids).detach()

    return recipe_entity_ids, recipe_name_by_id, embeddings


def calculate_similarities(recipe_ids, recipe_name_by_id, embeddings):
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    rows = []
    unique_pairs = {}

    for index, recipe_id in enumerate(recipe_ids):
        ranked_indices = torch.argsort(similarity_matrix[index], descending=True)
        similar_count = 0

        for candidate_index in ranked_indices.tolist():
            if candidate_index == index:
                continue

            similarity = float(similarity_matrix[index, candidate_index])
            candidate_id = recipe_ids[candidate_index]
            rows.append(
                (
                    recipe_name_by_id[recipe_id],
                    recipe_name_by_id[candidate_id],
                    similarity,
                )
            )

            pair = tuple(sorted((recipe_id, candidate_id)))
            unique_pairs[pair] = max(unique_pairs.get(pair, float("-inf")), similarity)
            similar_count += 1
            if similar_count == 5:
                break

    return rows, unique_pairs


def save_similarity_results(rows):
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["recipe1", "recipe2", "similarity"])
        writer.writerows(rows)


def main():
    recipe_ids, recipe_name_by_id, embeddings = load_recipe_embeddings()
    rows, unique_pairs = calculate_similarities(
        recipe_ids,
        recipe_name_by_id,
        embeddings,
    )
    save_similarity_results(rows)

    print(f"Recipes analyzed: {len(recipe_ids)}")
    print(f"Similarity rows saved: {len(rows)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("Top 10 most similar recipe pairs:")

    top_pairs = sorted(unique_pairs.items(), key=lambda item: item[1], reverse=True)
    for (recipe1_id, recipe2_id), similarity in top_pairs[:10]:
        print(
            f"{recipe_name_by_id[recipe1_id]} | "
            f"{recipe_name_by_id[recipe2_id]} | {similarity:.4f}"
        )


if __name__ == "__main__":
    main()