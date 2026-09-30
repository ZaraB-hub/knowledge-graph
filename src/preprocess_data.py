import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "recipes.json"
NORMALIZATION_FILE = PROCESSED_DATA_DIR / "ingredient_normalization.json"


def clean_optional_value(value):
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def load_normalization_map():
    if not NORMALIZATION_FILE.exists():
        return {}

    with NORMALIZATION_FILE.open("r", encoding="utf-8") as mapping_file:
        mappings = json.load(mapping_file)

    return {
        entry["original"].strip().casefold(): entry["canonical"]
        for entry in mappings
    }


def extract_recipe(meal, normalization_map):
    recipe_id = clean_optional_value(meal.get("idMeal"))
    recipe_name = clean_optional_value(meal.get("strMeal"))
    if not recipe_id or not recipe_name:
        return None

    recipe = {
        "id": recipe_id,
        "name": recipe_name,
        "ingredients": [],
    }

    optional_fields = {
        "category": "strCategory",
        "area": "strArea",
        "instructions": "strInstructions",
    }
    for output_name, input_name in optional_fields.items():
        value = clean_optional_value(meal.get(input_name))
        if value is not None:
            recipe[output_name] = value

    for number in range(1, 21):
        ingredient = clean_optional_value(meal.get(f"strIngredient{number}"))
        if ingredient is None:
            continue

        measure = clean_optional_value(meal.get(f"strMeasure{number}")) or ""
        canonical_name = normalization_map.get(ingredient.casefold(), ingredient)
        recipe["ingredients"].append(
            {
                "name": ingredient,
                "canonical_name": canonical_name,
                "measure": measure,
            }
        )

    return recipe


def load_recipes():
    recipes = []
    normalization_map = load_normalization_map()
    raw_files = sorted(RAW_DATA_DIR.glob("meal_*.json"))
    failed_files = 0

    for raw_file in raw_files:
        try:
            with raw_file.open("r", encoding="utf-8") as input_file:
                payload = json.load(input_file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            failed_files += 1
            print(f"Skipping {raw_file.name}: {error}")
            continue

        meals = payload.get("meals") if isinstance(payload, dict) else None
        if not isinstance(meals, list):
            failed_files += 1
            print(f"Skipping {raw_file.name}: missing meals list")
            continue

        for meal in meals:
            if not isinstance(meal, dict):
                failed_files += 1
                print(f"Skipping invalid recipe in {raw_file.name}")
                continue

            recipe = extract_recipe(meal, normalization_map)
            if recipe is None:
                failed_files += 1
                print(f"Skipping recipe without idMeal or strMeal in {raw_file.name}")
                continue

                recipes.append(recipe)

            unique_recipes = {recipe["id"]: recipe for recipe in recipes}

            return raw_files, list(unique_recipes.values()), failed_files


def main():
    raw_files, recipes, failed_files = load_recipes()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(recipes, output_file, indent=2, ensure_ascii=False)

    print(f"Raw files processed: {len(raw_files)}")
    print(f"Recipes processed: {len(recipes)}")
    print(f"Recipes skipped/failed: {failed_files}")


if __name__ == "__main__":
    main()