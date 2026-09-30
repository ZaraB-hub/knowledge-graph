import json
import os
from pathlib import Path

from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.json"
SEMANTICS_FILE = PROJECT_ROOT / "data" / "processed" / "ingredient_semantics.json"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


RECIPE_QUERY = """
MERGE (recipe:Recipe {id: $id})
SET recipe.name = $name,
    recipe.instructions = $instructions
"""

INGREDIENTS_QUERY = """
MATCH (recipe:Recipe {id: $recipe_id})
UNWIND $ingredients AS ingredient
MERGE (item:Ingredient {name: ingredient.name})
SET item.canonical_name = ingredient.canonical_name
MERGE (recipe)-[rel:HAS_INGREDIENT]->(item)
SET rel.measure = ingredient.measure
"""

INGREDIENT_VARIANTS_QUERY = """
MATCH (recipe:Recipe {id: $recipe_id})
UNWIND $ingredients AS ingredient
WITH ingredient
WHERE ingredient.name <> ingredient.canonical_name
MATCH (variant:Ingredient {name: ingredient.name})
SET variant:IngredientVariant
MERGE (canonical:Ingredient {name: ingredient.canonical_name})
MERGE (variant)-[:IS_A]->(canonical)
"""

CATEGORY_QUERY = """
MATCH (recipe:Recipe {id: $recipe_id})
MERGE (category:Category {name: $category})
MERGE (recipe)-[:HAS_CATEGORY]->(category)
"""

AREA_QUERY = """
MATCH (recipe:Recipe {id: $recipe_id})
MERGE (area:Area {name: $area})
MERGE (recipe)-[:HAS_AREA]->(area)
"""

SEMANTICS_QUERY = """
MATCH (ingredient:Ingredient {name: $canonical})
MERGE (ingredient_type:IngredientType {name: $type})
SET ingredient.source = $ingredient_source,
    ingredient.source_id = $ingredient_source_id,
    ingredient_type.source = $type_source,
    ingredient_type.source_id = $type_source_id
MERGE (ingredient)-[relationship:IS_A]->(ingredient_type)
SET relationship.source = $type_source,
    relationship.source_id = $type_source_id
"""


def load_recipes():
    try:
        with INPUT_FILE.open("r", encoding="utf-8") as input_file:
            recipes = json.load(input_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {INPUT_FILE}: {error}") from error

    if not isinstance(recipes, list):
        raise RuntimeError(f"Expected a list of recipes in {INPUT_FILE}")

    return recipes


def load_semantics():
    if not SEMANTICS_FILE.exists():
        return []

    try:
        with SEMANTICS_FILE.open("r", encoding="utf-8") as semantics_file:
            semantics = json.load(semantics_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read {SEMANTICS_FILE}: {error}") from error

    if not isinstance(semantics, list):
        raise RuntimeError(f"Expected a list of semantic mappings in {SEMANTICS_FILE}")

    required_fields = {
        "canonical",
        "ingredient_source",
        "ingredient_source_id",
        "type",
        "type_source",
        "type_source_id",
    }
    for index, semantic in enumerate(semantics):
        if not isinstance(semantic, dict) or not required_fields.issubset(semantic):
            raise RuntimeError(
                f"Semantic mapping {index} must include: "
                f"{', '.join(sorted(required_fields))}"
            )
        if any(not str(semantic[field]).strip() for field in required_fields):
            raise RuntimeError(f"Semantic mapping {index} contains an empty provenance field")

    return semantics


def create_recipe_graph(transaction, recipe):
    # MERGE identifies each Recipe by its stable id and updates its values on reruns.
    transaction.run(
        RECIPE_QUERY,
        id=recipe["id"],
        name=recipe["name"],
        instructions=recipe.get("instructions"),
    ).consume()

    # Measurements belong to HAS_INGREDIENT relationships, not Ingredient nodes.
    transaction.run(
        INGREDIENTS_QUERY,
        recipe_id=recipe["id"],
        ingredients=recipe.get("ingredients", []),
    ).consume()
    transaction.run(
        INGREDIENT_VARIANTS_QUERY,
        recipe_id=recipe["id"],
        ingredients=recipe.get("ingredients", []),
    ).consume()

    if recipe.get("category"):
        transaction.run(
            CATEGORY_QUERY,
            recipe_id=recipe["id"],
            category=recipe["category"],
        ).consume()

    if recipe.get("area"):
        transaction.run(
            AREA_QUERY,
            recipe_id=recipe["id"],
            area=recipe["area"],
        ).consume()


STATISTICS_QUERIES = {
    "Recipe nodes": "MATCH (node:Recipe) RETURN count(node) AS count",
    "Canonical Ingredient nodes": (
        "MATCH (node:Ingredient) WHERE NOT node:IngredientVariant "
        "RETURN count(node) AS count"
    ),
    "Ingredient Variant nodes": (
        "MATCH (node:IngredientVariant) RETURN count(node) AS count"
    ),
    "IngredientType nodes": "MATCH (node:IngredientType) RETURN count(node) AS count",
    "Category nodes": "MATCH (node:Category) RETURN count(node) AS count",
    "Area nodes": "MATCH (node:Area) RETURN count(node) AS count",
    "HAS_INGREDIENT relationships": (
        "MATCH ()-[relationship:HAS_INGREDIENT]->() "
        "RETURN count(relationship) AS count"
    ),
    "HAS_CATEGORY relationships": (
        "MATCH ()-[relationship:HAS_CATEGORY]->() "
        "RETURN count(relationship) AS count"
    ),
    "HAS_AREA relationships": (
        "MATCH ()-[relationship:HAS_AREA]->() "
        "RETURN count(relationship) AS count"
    ),
    "IS_A relationships": (
        "MATCH ()-[relationship:IS_A]->() "
        "RETURN count(relationship) AS count"
    ),
}


def collect_statistics(session):
    statistics = {}
    for label, query in STATISTICS_QUERIES.items():
        statistics[label] = session.run(query).single()["count"]
    return statistics


def main():
    if not NEO4J_PASSWORD:
        raise RuntimeError("Set the NEO4J_PASSWORD environment variable before running.")

    recipes = load_recipes()
    semantics = load_semantics()

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()
        with driver.session(database=NEO4J_DATABASE) as session:
            for recipe in recipes:
                session.execute_write(create_recipe_graph, recipe)

            for semantic in semantics:
                session.run(SEMANTICS_QUERY, **semantic).consume()

            statistics = collect_statistics(session)

    print(f"Recipes loaded: {len(recipes)}")
    for label, count in statistics.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()