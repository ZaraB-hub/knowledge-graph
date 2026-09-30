import csv
import os
from pathlib import Path

from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "triples.tsv"

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


TRIPLE_QUERY = """
MATCH (head)-[relationship]->(tail)
WHERE type(relationship) IN [
    "HAS_INGREDIENT",
    "HAS_CATEGORY",
    "HAS_AREA"
]
RETURN
    CASE
        WHEN head:Recipe THEN "recipe:" + head.id
        WHEN head:Ingredient THEN "ingredient:" + head.name
        WHEN head:Category THEN "category:" + head.name
        WHEN head:Area THEN "area:" + head.name
    END AS head,
    type(relationship) AS relation,
    CASE
        WHEN tail:Recipe THEN "recipe:" + tail.id
        WHEN tail:Ingredient THEN "ingredient:" + tail.name
        WHEN tail:Category THEN "category:" + tail.name
        WHEN tail:Area THEN "area:" + tail.name
    END AS tail
ORDER BY head, relation, tail
"""


def main():
    if not NEO4J_PASSWORD:
        raise RuntimeError("Set the NEO4J_PASSWORD environment variable before running.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    ) as driver:
        driver.verify_connectivity()

        with driver.session(database=NEO4J_DATABASE) as session:
            records = session.run(TRIPLE_QUERY)

            with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.writer(
                    output_file,
                    delimiter="\t",
                    lineterminator="\n",
                )

                triple_count = 0
                for record in records:
                    writer.writerow(
                        [
                            record["head"],
                            record["relation"],
                            record["tail"],
                        ]
                    )
                    triple_count += 1

    print(f"Triples exported: {triple_count}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()