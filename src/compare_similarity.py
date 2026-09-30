import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSE_FILE = PROJECT_ROOT / "results" / "transe" / "recipe_similarity.csv"
COMPARISON_DIR = PROJECT_ROOT / "results" / "comparison"
COMPARISON_FILE = COMPARISON_DIR / "similarity_comparison.csv"
TOP10_FILE = COMPARISON_DIR / "top10_comparison.csv"

JACCARD_COLUMNS = {
    "recipe1",
    "recipe2",
    "intersection",
    "union_size",
    "jaccard",
}


def pair_key(recipe1, recipe2):
    return tuple(sorted((recipe1.strip(), recipe2.strip())))


def find_jaccard_file():
    search_roots = [PROJECT_ROOT / "data", PROJECT_ROOT / "results"]
    candidates = []

    for root in search_roots:
        if root.exists():
            candidates.extend(root.rglob("*.csv"))

    for candidate in sorted(candidates):
        if candidate == TRANSE_FILE:
            continue

        with candidate.open("r", encoding="utf-8", newline="") as input_file:
            reader = csv.reader(input_file)
            header = set(next(reader, []))

        if JACCARD_COLUMNS.issubset(header):
            return candidate

    expected = ", ".join(sorted(JACCARD_COLUMNS))
    searched = ", ".join(str(root) for root in search_roots)
    raise FileNotFoundError(
        f"Could not find a Jaccard CSV with columns: {expected}. "
        f"Searched: {searched}"
    )


def load_similarity_file(path, similarity_column):
    if not path.exists():
        raise FileNotFoundError(f"Missing similarity file: {path}")

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"recipe1", "recipe2", similarity_column}
        if not required_columns.issubset(reader.fieldnames or []):
            expected = ", ".join(sorted(required_columns))
            raise ValueError(f"{path} must contain columns: {expected}")

        similarities = {}
        for row in reader:
            key = pair_key(row["recipe1"], row["recipe2"])
            try:
                similarities[key] = float(row[similarity_column])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {similarity_column} value in {path}: {row[similarity_column]}"
                ) from error

    if not similarities:
        raise ValueError(f"Similarity file is empty: {path}")

    return similarities


def pearson_correlation(values1, values2):
    if len(values1) < 2:
        return None

    mean1 = sum(values1) / len(values1)
    mean2 = sum(values2) / len(values2)
    centered1 = [value - mean1 for value in values1]
    centered2 = [value - mean2 for value in values2]
    denominator = math.sqrt(
        sum(value * value for value in centered1)
        * sum(value * value for value in centered2)
    )
    if denominator == 0:
        return None

    numerator = sum(value1 * value2 for value1, value2 in zip(centered1, centered2))
    return numerator / denominator


def write_outputs(joined_rows):
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    with COMPARISON_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["recipe1", "recipe2", "transe_similarity", "jaccard"])
        writer.writerows(joined_rows)

    top_transe = sorted(joined_rows, key=lambda row: row[2], reverse=True)[:10]
    top_jaccard = sorted(joined_rows, key=lambda row: row[3], reverse=True)[:10]

    with TOP10_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["method", "recipe1", "recipe2", "similarity"])
        writer.writerows(
            [["TransE", row[0], row[1], row[2]] for row in top_transe]
            + [["Jaccard", row[0], row[1], row[3]] for row in top_jaccard]
        )

    return top_transe, top_jaccard


def print_pairs(title, rows, column_index):
    print(title)
    for recipe1, recipe2, transe, jaccard in rows:
        similarity = transe if column_index == 2 else jaccard
        print(f"  {recipe1} | {recipe2} | {similarity:.4f}")


def main():
    jaccard_file = find_jaccard_file()
    transe = load_similarity_file(TRANSE_FILE, "similarity")
    jaccard = load_similarity_file(jaccard_file, "jaccard")

    common_pairs = sorted(set(transe) & set(jaccard))
    if not common_pairs:
        raise ValueError("The TransE and Jaccard files contain no common recipe pairs.")

    joined_rows = [
        [recipe1, recipe2, transe[(recipe1, recipe2)], jaccard[(recipe1, recipe2)]]
        for recipe1, recipe2 in common_pairs
    ]
    top_transe, top_jaccard = write_outputs(joined_rows)

    transe_values = [row[2] for row in joined_rows]
    jaccard_values = [row[3] for row in joined_rows]
    correlation = pearson_correlation(transe_values, jaccard_values)
    transe_pairs = {(row[0], row[1]) for row in top_transe}
    jaccard_pairs = {(row[0], row[1]) for row in top_jaccard}

    print(f"Jaccard file: {jaccard_file}")
    print(f"Pairs present in both methods: {len(joined_rows)}")
    print(f"Average TransE similarity: {sum(transe_values) / len(transe_values):.4f}")
    print(f"Average Jaccard similarity: {sum(jaccard_values) / len(jaccard_values):.4f}")
    print(
        "Pearson correlation: "
        + (f"{correlation:.4f}" if correlation is not None else "not available")
    )
    print(f"Top-10 ranking overlap: {len(transe_pairs & jaccard_pairs)} pairs")
    print_pairs("Top 10 pairs according to TransE:", top_transe, 2)
    print_pairs("Top 10 pairs according to Jaccard:", top_jaccard, 3)
    print(f"Comparison file: {COMPARISON_FILE}")
    print(f"Top-10 file: {TOP10_FILE}")


if __name__ == "__main__":
    main()