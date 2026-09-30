import os

from neo4j import GraphDatabase


NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def run_query(session, query):
    record = session.run(query).single()
    return record[0] if record else 0


def main():
    if not NEO4J_PASSWORD:
        raise RuntimeError(
            "Set the NEO4J_PASSWORD environment variable before running."
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        driver.verify_connectivity()

        with driver.session(database=NEO4J_DATABASE) as session:

            print("\n=== KNOWLEDGE GRAPH STATISTICS ===\n")

            # -------------------------
            # Node counts
            # -------------------------
            print("NODES")

            node_queries = {
                "Recipes": """
                    MATCH (n:Recipe)
                    RETURN count(n)
                """,
                "Ingredients": """
                    MATCH (n:Ingredient)
                    RETURN count(n)
                """,
                "Ingredient Types": """
                    MATCH (n:IngredientType)
                    RETURN count(n)
                """,
                "Categories": """
                    MATCH (n:Category)
                    RETURN count(n)
                """,
                "Areas": """
                    MATCH (n:Area)
                    RETURN count(n)
                """,
                "Ingredient Variants": """
                    MATCH (n:IngredientVariant)
                    RETURN count(n)
                """,
            }

            for name, query in node_queries.items():
                print(f"{name}: {run_query(session, query)}")

            # -------------------------
            # Relationship counts
            # -------------------------
            print("\nRELATIONSHIPS")

            relationship_queries = {
                "HAS_INGREDIENT": """
                    MATCH ()-[r:HAS_INGREDIENT]->()
                    RETURN count(r)
                """,
                "HAS_CATEGORY": """
                    MATCH ()-[r:HAS_CATEGORY]->()
                    RETURN count(r)
                """,
                "HAS_AREA": """
                    MATCH ()-[r:HAS_AREA]->()
                    RETURN count(r)
                """,
                "IS_A": """
                    MATCH ()-[r:IS_A]->()
                    RETURN count(r)
                """,
            }

            for name, query in relationship_queries.items():
                print(f"{name}: {run_query(session, query)}")

            # -------------------------
            # Top ingredients
            # -------------------------
            print("\nTOP 10 MOST COMMON INGREDIENTS")

            result = session.run("""
                MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN i.name AS ingredient, count(r) AS recipe_count
                ORDER BY recipe_count DESC
                LIMIT 10
            """)

            for index, record in enumerate(result, start=1):
                print(
                    f"{index}. {record['ingredient']} "
                    f"({record['recipe_count']} recipes)"
                )

            # -------------------------
            # Recipes with most ingredients
            # -------------------------
            print("\nTOP 10 RECIPES BY NUMBER OF INGREDIENTS")

            result = session.run("""
                MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i:Ingredient)
                RETURN r.name AS recipe, count(i) AS ingredient_count
                ORDER BY ingredient_count DESC
                LIMIT 10
            """)

            for index, record in enumerate(result, start=1):
                print(
                    f"{index}. {record['recipe']} "
                    f"({record['ingredient_count']} ingredients)"
                )

            # -------------------------
            # Recipes per category
            # -------------------------
            print("\nRECIPES PER CATEGORY")

            result = session.run("""
                MATCH (r:Recipe)-[:HAS_CATEGORY]->(c:Category)
                RETURN c.name AS category, count(r) AS recipe_count
                ORDER BY recipe_count DESC
            """)

            for record in result:
                print(
                    f"{record['category']}: "
                    f"{record['recipe_count']}"
                )

            # -------------------------
            # Recipes per area
            # -------------------------
            print("\nTOP 10 AREAS BY NUMBER OF RECIPES")

            result = session.run("""
                MATCH (r:Recipe)-[:HAS_AREA]->(a:Area)
                RETURN a.name AS area, count(r) AS recipe_count
                ORDER BY recipe_count DESC
                LIMIT 10
            """)

            for index, record in enumerate(result, start=1):
                print(
                    f"{index}. {record['area']} "
                    f"({record['recipe_count']} recipes)"
                )

            # -------------------------
            # Most common ingredient pairs
            # -------------------------
            print("\nTOP 10 INGREDIENT PAIRS")

            result = session.run("""
                MATCH (r:Recipe)-[:HAS_INGREDIENT]->(i1:Ingredient)
                MATCH (r)-[:HAS_INGREDIENT]->(i2:Ingredient)
                WHERE i1.name < i2.name
                RETURN
                    i1.name AS ingredient1,
                    i2.name AS ingredient2,
                    count(r) AS shared_recipes
                ORDER BY shared_recipes DESC
                LIMIT 10
            """)

            for index, record in enumerate(result, start=1):
                print(
                    f"{index}. "
                    f"{record['ingredient1']} + "
                    f"{record['ingredient2']} "
                    f"({record['shared_recipes']} recipes)"
                )

            print("\n=== END ===\n")

    finally:
        driver.close()


if __name__ == "__main__":
    main()