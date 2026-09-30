import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPES_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.json"
OUTPUT_FILE = PROJECT_ROOT / "results" / "comparison" / "jaccard_recommendations.csv"


def load_recipes():
    with RECIPES_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        recipes = data.get("recipes", data)
    else:
        recipes = data

    return recipes


def get_ingredients(recipe):
    ingredients = recipe.get("ingredients", [])
    result = set()

    for ingredient in ingredients:
        if isinstance(ingredient, str):
            result.add(ingredient.strip().lower())
        elif isinstance(ingredient, dict):
            name = (
                ingredient.get("canonical")
                or ingredient.get("ingredient")
                or ingredient.get("name")
            )
            if name:
                result.add(name.strip().lower())

    return result


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def main():
    recipes = load_recipes()

    recipe_data = {}

    for recipe in recipes:
        name = recipe.get("name")
        if name:
            recipe_data[name] = get_ingredients(recipe)

    rows = []

    for recipe_name, ingredients in recipe_data.items():
        candidates = []

        for candidate_name, candidate_ingredients in recipe_data.items():
            if candidate_name == recipe_name:
                continue

            similarity = jaccard(ingredients, candidate_ingredients)

            candidates.append(
                (candidate_name, similarity)
            )

        candidates.sort(key=lambda x: x[1], reverse=True)

        for rank, (candidate_name, similarity) in enumerate(
            candidates[:5], start=1
        ):
            rows.append(
                (
                    recipe_name,
                    rank,
                    candidate_name,
                    similarity,
                )
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "recipe",
                "rank",
                "recommended_recipe",
                "jaccard_similarity",
            ]
        )

        writer.writerows(rows)

    print(f"Recipes analyzed: {len(recipe_data)}")
    print(f"Recommendation rows: {len(rows)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()