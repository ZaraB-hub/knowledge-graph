import argparse
import os

from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


SUBSTITUTION_QUERY = """
MATCH (original:Ingredient)
WHERE toLower(original.name) = toLower($ingredient_name)

MATCH (recipe:Recipe)-[:HAS_INGREDIENT]->(original)

MATCH (recipe)-[:HAS_INGREDIENT]->(candidate:Ingredient)
WHERE candidate <> original
  AND NOT candidate:IngredientVariant

WITH original,
     candidate,
     count(DISTINCT recipe) AS shared_recipe_count

OPTIONAL MATCH (original)-[:IS_A]->(original_type:IngredientType)

OPTIONAL MATCH (candidate)-[:IS_A]->(candidate_type:IngredientType)

WITH original,
     candidate,
     shared_recipe_count,
     original_type,
     candidate_type,
     CASE
         WHEN original_type IS NOT NULL
              AND candidate_type IS NOT NULL
              AND original_type.name = candidate_type.name
         THEN 1
         ELSE 0
     END AS semantic_bonus

RETURN
    original.name AS original_ingredient,
    candidate.name AS substitute_ingredient,
    shared_recipe_count,
    original_type.name AS original_ingredient_type,
    candidate_type.name AS ingredient_type,
    semantic_bonus,
    (
        shared_recipe_count * 1.0
        + semantic_bonus * 2.0
    ) AS score

ORDER BY score DESC, shared_recipe_count DESC, substitute_ingredient
LIMIT $limit
"""


def find_substitutes(ingredient_name, limit=5):
    """
    Find data-supported ingredient substitution candidates.

    Candidates are based primarily on recipe co-occurrence.
    A small additional score is given when the candidate shares
    the same IngredientType as the original ingredient.
    """

    if not ingredient_name or not ingredient_name.strip():
        raise ValueError("ingredient_name must not be empty")

    if limit < 1:
        raise ValueError("limit must be at least 1")

    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "Set the NEO4J_PASSWORD environment variable before running."
        )

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    ) as driver:

        driver.verify_connectivity()

        with driver.session(database=NEO4J_DATABASE) as session:
            records = session.run(
                SUBSTITUTION_QUERY,
                ingredient_name=ingredient_name.strip(),
                limit=limit,
            )

            results = []

            for record in records:
                shared_recipe_count = record["shared_recipe_count"]
                semantic_bonus = record["semantic_bonus"]
                ingredient_type = record["ingredient_type"]

                if semantic_bonus:
                    reason = (
                        f"Appears with the original ingredient in "
                        f"{shared_recipe_count} recipe(s) and shares "
                        f"the IngredientType '{ingredient_type}'."
                    )
                else:
                    reason = (
                        f"Appears with the original ingredient in "
                        f"{shared_recipe_count} recipe(s)."
                    )

                results.append(
                    {
                        "original_ingredient": record[
                            "original_ingredient"
                        ],
                        "substitute_ingredient": record[
                            "substitute_ingredient"
                        ],
                        "score": float(record["score"]),
                        "shared_recipe_count": shared_recipe_count,
                        "ingredient_type": ingredient_type,
                        "reason": reason,
                    }
                )

            return results


def main():
    parser = argparse.ArgumentParser(
        description="Find data-supported ingredient substitution candidates."
    )

    parser.add_argument(
        "ingredient",
        help="Ingredient name in the Neo4j graph",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results",
    )

    args = parser.parse_args()

    try:
        substitutions = find_substitutes(
            args.ingredient,
            limit=args.limit,
        )

    except (RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    print(f"Original ingredient: {args.ingredient}\n")

    if not substitutions:
        print("No supported substitution candidates found.")
        return 0

    print("Data-supported substitution candidates:\n")

    for index, substitution in enumerate(substitutions, start=1):

        print(
            f"{index}. "
            f"{substitution['substitute_ingredient']}"
        )

        print(
            f"   Score: "
            f"{substitution['score']:.2f}"
        )

        print(
            f"   Shared recipes: "
            f"{substitution['shared_recipe_count']}"
        )

        if substitution["ingredient_type"]:
            print(
                f"   Ingredient type: "
                f"{substitution['ingredient_type']}"
            )

        print(
            f"   Reason: "
            f"{substitution['reason']}"
        )

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())