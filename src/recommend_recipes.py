import csv
import json
from pathlib import Path

import torch

from analyze_embeddings import load_recipe_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPES_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.json"
MODEL_DIR = PROJECT_ROOT / "results" / "transe"
RECOMMENDATIONS_FILE = MODEL_DIR / "recipe_recommendations.csv"
ANALYSIS_FILE = MODEL_DIR / "recommendation_analysis.csv"


def load_recipe_data():
    with RECIPES_FILE.open("r", encoding="utf-8") as recipes_file:
        recipes = json.load(recipes_file)

    recipe_data = {}
    for recipe in recipes:
        recipe_id = f"recipe:{recipe['id']}"
        recipe_data[recipe_id] = {
            "name": recipe["name"],
            "ingredients": {
                ingredient["name"] for ingredient in recipe.get("ingredients", [])
            },
        }

    return recipe_data


def calculate_recommendations(recipe_ids, embeddings):
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    similarity_matrix = normalized_embeddings @ normalized_embeddings.T
    recommendations = []

    for recipe_index, recipe_id in enumerate(recipe_ids):
        ranked_indices = torch.argsort(
            similarity_matrix[recipe_index],
            descending=True,
        )
        rank = 0

        for candidate_index in ranked_indices.tolist():
            if candidate_index == recipe_index:
                continue

            rank += 1
            recommendations.append(
                {
                    "recipe_id": recipe_id,
                    "recommended_id": recipe_ids[candidate_index],
                    "rank": rank,
                    "transE_similarity": float(
                        similarity_matrix[recipe_index, candidate_index]
                    ),
                }
            )
            if rank == 5:
                break

    return recommendations


def add_jaccard_analysis(recommendations, recipe_data):
    analysis = []

    for recommendation in recommendations:
        recipe_ingredients = recipe_data[recommendation["recipe_id"]]["ingredients"]
        recommended_ingredients = recipe_data[
            recommendation["recommended_id"]
        ]["ingredients"]
        shared_ingredients = recipe_ingredients & recommended_ingredients
        union = recipe_ingredients | recommended_ingredients
        jaccard = len(shared_ingredients) / len(union) if union else 0.0

        analysis.append(
            {
                "recipe": recipe_data[recommendation["recipe_id"]]["name"],
                "recommended_recipe": recipe_data[
                    recommendation["recommended_id"]
                ]["name"],
                "transE_similarity": recommendation["transE_similarity"],
                "jaccard_similarity": jaccard,
                "shared_ingredients": len(shared_ingredients),
            }
        )

    return analysis


def main():
    recipe_ids, recipe_name_by_id, embeddings = load_recipe_embeddings()
    recipe_data = load_recipe_data()
    recommendations = calculate_recommendations(recipe_ids, embeddings)
    analysis = add_jaccard_analysis(recommendations, recipe_data)

    with RECOMMENDATIONS_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["recipe", "rank", "recommended_recipe", "transE_similarity"])
        for recommendation in recommendations:
            writer.writerow(
                [
                    recipe_name_by_id[recommendation["recipe_id"]],
                    recommendation["rank"],
                    recipe_name_by_id[recommendation["recommended_id"]],
                    recommendation["transE_similarity"],
                ]
            )

    with ANALYSIS_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "recipe",
                "recommended_recipe",
                "TransE",
                "Jaccard",
                "shared_ingredients",
            ]
        )
        for row in analysis:
            writer.writerow(
                [
                    row["recipe"],
                    row["recommended_recipe"],
                    row["transE_similarity"],
                    row["jaccard_similarity"],
                    row["shared_ingredients"],
                ]
            )

    print(f"Recipes analyzed: {len(recipe_ids)}")
    print(f"Recommendation rows: {len(recommendations)}")
    print(f"Recommendations saved to: {RECOMMENDATIONS_FILE}")
    print(f"Recommendation analysis saved to: {ANALYSIS_FILE}")


if __name__ == "__main__":
    main()