import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPES_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.json"
TRANSE_FILE = PROJECT_ROOT / "results" / "transe" / "recipe_similarity.csv"
DISTMULT_FILE = PROJECT_ROOT / "results" / "distmult" / "recipe_similarity.csv"
JACCARD_FILE = PROJECT_ROOT / "results" / "jaccard" / "recipe_jaccard.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"
DETAIL_FILE = OUTPUT_DIR / "recommendation_comparison.csv"
SUMMARY_FILE = OUTPUT_DIR / "recommendation_summary.csv"

REQUESTED_RECIPES = [
    "Apple Frangipan Tart",
    "Apple Pie",
    "Potato Salad (Olivier Salad)",
    "Madeira Cake",
    "Szechuan Beef",
    "Jamaican Rice and Peas",
    "Tom yum (hot & sour) soup with prawns",
]


def load_recipe_data():
    with RECIPES_FILE.open("r", encoding="utf-8") as input_file:
        recipes = json.load(input_file)

    recipe_names = {recipe["name"] for recipe in recipes}
    ingredient_sets = {
        recipe["name"]: {
            ingredient["name"]
            for ingredient in recipe.get("ingredients", [])
            if ingredient.get("name")
        }
        for recipe in recipes
    }
    return recipe_names, ingredient_sets


def pair_key(recipe1, recipe2):
    return tuple(sorted((recipe1.strip(), recipe2.strip())))


def load_embedding_recommendations(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding similarity file: {path}")

    recommendations = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"recipe1", "recipe2", "similarity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain: {', '.join(sorted(required))}")

        for row in reader:
            source = row["recipe1"].strip()
            candidate = row["recipe2"].strip()
            if source == candidate:
                continue
            recommendations.setdefault(source, []).append(
                (candidate, float(row["similarity"]))
            )

    for source in recommendations:
        recommendations[source].sort(key=lambda item: item[1], reverse=True)
    return recommendations


def load_jaccard_results(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing Jaccard similarity file: {path}")

    pair_scores = {}
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"recipe1", "recipe2", "jaccard"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain: {', '.join(sorted(required))}")

        for row in reader:
            recipe1 = row["recipe1"].strip()
            recipe2 = row["recipe2"].strip()
            if recipe1 != recipe2:
                pair_scores[pair_key(recipe1, recipe2)] = float(row["jaccard"])

    return pair_scores


def calculate_jaccard(recipe1, recipe2, ingredient_sets):
    ingredients1 = ingredient_sets.get(recipe1, set())
    ingredients2 = ingredient_sets.get(recipe2, set())
    shared = ingredients1 & ingredients2
    union = ingredients1 | ingredients2
    return (len(shared) / len(union) if union else 0.0), len(shared)


def select_recipes(recipe_names):
    extras = [
        "Garides Saganaki",
        "Aubergine couscous salad",
        "Algerian Kefta (Meatballs)",
    ]
    selected = []
    unavailable = []
    for recipe in REQUESTED_RECIPES + extras:
        if recipe in recipe_names:
            if recipe not in selected:
                selected.append(recipe)
        else:
            unavailable.append(recipe)
    return selected[:10], unavailable


def collect_method_rows(method, source, recommendations, jaccard_scores, ingredient_sets):
    rows = []
    for rank, (candidate, similarity) in enumerate(recommendations.get(source, [])[:5], 1):
        key = pair_key(source, candidate)
        if key in jaccard_scores:
            ingredient_jaccard = jaccard_scores[key]
            shared_ingredients = round(
                ingredient_jaccard
                * len(ingredient_sets.get(source, set()) | ingredient_sets.get(candidate, set()))
            )
        else:
            ingredient_jaccard, shared_ingredients = calculate_jaccard(
                source,
                candidate,
                ingredient_sets,
            )

        rows.append(
            {
                "recipe": source,
                "method": method,
                "rank": rank,
                "recommended_recipe": candidate,
                "similarity": similarity,
                "ingredient_jaccard": ingredient_jaccard,
                "shared_ingredients": shared_ingredients,
            }
        )
    return rows


def write_outputs(selected, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with DETAIL_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "recipe",
                "method",
                "rank",
                "recommended_recipe",
                "similarity",
                "ingredient_jaccard",
                "shared_ingredients",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    rows_by_method = {
        (row["recipe"], row["method"]): row
        for row in rows
        if row["rank"] == 1
    }
    with SUMMARY_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["recipe", "jaccard_top1", "transe_top1", "distmult_top1"])
        for recipe in selected:
            writer.writerow(
                [
                    recipe,
                    rows_by_method.get((recipe, "Jaccard"), {}).get(
                        "recommended_recipe", ""
                    ),
                    rows_by_method.get((recipe, "TransE"), {}).get(
                        "recommended_recipe", ""
                    ),
                    rows_by_method.get((recipe, "DistMult"), {}).get(
                        "recommended_recipe", ""
                    ),
                ]
            )


def main():
    recipe_names, ingredient_sets = load_recipe_data()
    selected, unavailable = select_recipes(recipe_names)
    transe = load_embedding_recommendations(TRANSE_FILE)
    distmult = load_embedding_recommendations(DISTMULT_FILE)
    jaccard_scores = load_jaccard_results(JACCARD_FILE)

    jaccard_recommendations = {}
    for source in selected:
        candidates = []
        for (recipe1, recipe2), similarity in jaccard_scores.items():
            if recipe1 == source:
                candidates.append((recipe2, similarity))
            elif recipe2 == source:
                candidates.append((recipe1, similarity))
        jaccard_recommendations[source] = sorted(
            candidates,
            key=lambda item: item[1],
            reverse=True,
        )

    rows = []
    for source in selected:
        rows.extend(
            collect_method_rows(
                "Jaccard",
                source,
                jaccard_recommendations,
                jaccard_scores,
                ingredient_sets,
            )
        )
        rows.extend(
            collect_method_rows(
                "TransE",
                source,
                transe,
                jaccard_scores,
                ingredient_sets,
            )
        )
        rows.extend(
            collect_method_rows(
                "DistMult",
                source,
                distmult,
                jaccard_scores,
                ingredient_sets,
            )
        )

    write_outputs(selected, rows)

    print(f"Selected recipes: {len(selected)}")
    if unavailable:
        print(f"Unavailable requested recipes: {', '.join(unavailable)}")
    print(f"Recommendation rows: {len(rows)}")
    print(f"Detailed output: {DETAIL_FILE}")
    print(f"Summary output: {SUMMARY_FILE}")
    for source in selected:
        print(f"\n{source}")
        for method in ("Jaccard", "TransE", "DistMult"):
            top = next(
                (
                    row["recommended_recipe"]
                    for row in rows
                    if row["recipe"] == source
                    and row["method"] == method
                    and row["rank"] == 1
                ),
                "unavailable",
            )
            print(f"  {method}: {top}")


if __name__ == "__main__":
    main()