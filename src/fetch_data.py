import json
import string
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

BASE_URL = "https://www.themealdb.com/api/json/v1/1"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LETTERS = string.ascii_lowercase


def fetch_json(endpoint, params):
    url = f"{BASE_URL}/{endpoint}?{urlencode(params)}"

    try:
        with urlopen(url, timeout=30) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Request failed for {url}: {error}") from error


def save_json(filename, data):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (RAW_DATA_DIR / filename).open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)


def main():
    meal_ids = []
    seen_ids = set()
    failed_requests = 0

    for letter in LETTERS:
        try:
            search_data = fetch_json("search.php", {"f": letter})
            save_json(f"search_{letter}.json", search_data)
        except RuntimeError as error:
            failed_requests += 1
            print(error, file=sys.stderr)
            continue

        for meal in search_data.get("meals") or []:
            meal_id = meal.get("idMeal")
            if meal_id and meal_id not in seen_ids:
                seen_ids.add(meal_id)
                meal_ids.append(meal_id)

    successful_requests = 0
    for meal_id in meal_ids:
        try:
            detail_data = fetch_json("lookup.php", {"i": meal_id})
            save_json(f"meal_{meal_id}.json", detail_data)
            successful_requests += 1
        except RuntimeError as error:
            failed_requests += 1
            print(error, file=sys.stderr)

    print(f"Letters searched: {len(LETTERS)}")
    print(f"Unique meals found: {len(meal_ids)}")
    print(f"Successfully retrieved: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    return 1 if failed_requests else 0


if __name__ == "__main__":
    sys.exit(main())
