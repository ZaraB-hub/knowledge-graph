# Recipe Knowledge Graph

Recipe Knowledge Graph for ingredient substitution and recipe recommendation using TheMealDB, Neo4j, PyKEEN, TransE, DistMult, and Jaccard similarity.

## Features

- Ingests recipe data from TheMealDB.
- Preprocesses recipes while preserving ingredient names and measurements.
- Builds a Neo4j knowledge graph.
- Represents recipe, ingredient, category, area, ingredient variant, and ingredient type entities.
- Exports Neo4j relationships as PyKEEN triples.
- Trains and evaluates TransE and DistMult models.
- Generates recipe similarity and recommendation results.
- Provides a Streamlit interface for DistMult recommendations.
- Provides an explainable Neo4j-backed ingredient substitution function.

## Project Structure

```text
data/
  raw/                         TheMealDB JSON snapshots
  processed/
    recipes.json               Processed recipes
    ingredient_normalization.json
    ingredient_semantics.json  FoodOn-backed mappings
    triples.tsv                Exported KG triples

results/
  jaccard/                     Jaccard baseline results
  transe/                      TransE model and results
  distmult/                    DistMult model and results
  comparison/                  Method comparison results

src/
  fetch_data.py
  preprocess_data.py
  build_kg.py
  export_triples.py
  train_transe.py
  evaluate_transe.py
  analyze_embeddings.py
  train_distmult.py
  evaluate_distmult.py
  analyze_embeddings_distmult.py
  calculate_jaccard.py
  recommend_recipes.py
  recommend_distmult.py
  ingredient_substitution.py
  compare_recommendations.py
  app.py
```

## Current Dataset

- Recipes: 790
- Exported KG triples: 9,306
- Original ingredient names: 1,144
- Canonical ingredient names: 1,129
- Ingredient variants: 18
- Ingredient types: 14
- Categories: 14
- Areas: 37

The exported graph currently contains these relationship types:

- `HAS_INGREDIENT`
- `HAS_CATEGORY`
- `HAS_AREA`
- `IS_A`

## Installation

Use a Python virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```


## Neo4j Configuration

The Neo4j scripts read these environment variables:

```powershell
$env:NEO4J_URI = "neo4j://127.0.0.1:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_DATABASE = "neo4j"
$env:NEO4J_PASSWORD = Read-Host "Neo4j password"
```


## Pipeline

Run the stages in this order when rebuilding the project:

```text
TheMealDB API
  -> data/raw/
  -> preprocess_data.py
  -> data/processed/recipes.json
  -> build_kg.py
  -> Neo4j graph
  -> export_triples.py
  -> data/processed/triples.tsv
  -> train_transe.py / train_distmult.py
  -> saved PyKEEN models
  -> evaluation and recommendation scripts
```

### 1. Ingest recipe data

```powershell
python src/fetch_data.py
python src/preprocess_data.py
```

The preprocessing stage preserves original ingredient names and measurements. Normalization is additive through `canonical_name`.

### 2. Build the Neo4j graph

```powershell
python src/build_kg.py
```

The graph preserves:

```text
Recipe -[:HAS_INGREDIENT {measure}]-> Ingredient
Recipe -[:HAS_CATEGORY]-> Category
Recipe -[:HAS_AREA]-> Area
IngredientVariant -[:IS_A]-> Ingredient
Ingredient -[:IS_A]-> IngredientType
```

### 3. Export triples

```powershell
python src/export_triples.py
```

This writes `data/processed/triples.tsv` for PyKEEN.

### 4. Train and evaluate models

```powershell
python src/train_transe.py
python src/evaluate_transe.py

python src/train_distmult.py
python src/evaluate_distmult.py
```

The current training configuration uses:

- 80/10/10 train/validation/test split
- Random seed 42
- 128-dimensional embeddings
- 300 epochs
- Adam optimizer
- Learning rate 0.001

Training artifacts are saved under `results/transe/` and `results/distmult/`.

## Recommendation Usage

### DistMult command line recommender

```powershell
python src/recommend_distmult.py "Apple Frangipan Tart"
python src/recommend_distmult.py "Apple Pie" --top-k 10
```

The recommender:

1. Loads the saved DistMult model once for the run.
2. Uses the saved entity mapping to identify `recipe:*` entities.
3. Extracts the selected recipe embedding.
4. Compares it with all other recipe embeddings using cosine similarity.
5. Excludes the selected recipe.
6. Sorts candidates by similarity.
7. Returns the requested top-k recipes.

Output is written to `results/distmult/recommendations.csv`.

### Streamlit application

```powershell
streamlit run src/app.py
```

The application provides:

- Searchable recipe selection across the processed recipes.
- Configurable recommendation count.
- Recipe category, area, ingredients, and instructions.
- DistMult recommendations with similarity scores.
- Ingredient overlap and Jaccard explanation for each recommendation.
- Neo4j-backed ingredient substitution lookup.

The DistMult model is cached with Streamlit and is not retrained when the application starts.

## Ingredient Substitution

The backend substitution module can be tested independently:

```powershell
python src/ingredient_substitution.py "Butter"
python src/ingredient_substitution.py "Onion" --limit 5
python src/ingredient_substitution.py "Garlic" --limit 10
```

Substitutions are not generated by an LLM or external recommendation service. The query uses only existing Neo4j evidence:

- Recipe co-occurrence through `HAS_INGREDIENT`.
- Shared `IngredientType` through `IS_A`.
- Existing ingredient and variant nodes.

If the graph does not support a candidate, the module reports that no supported substitution was found.

## Similarity Baselines

Generate the Jaccard baseline with:

```powershell
python src/calculate_jaccard.py
```

Generate embedding similarities with:

```powershell
python src/analyze_embeddings.py
python src/analyze_embeddings_distmult.py
```

Compare selected recommendations with:

```powershell
python src/compare_recommendations.py
```

The comparison outputs are stored under `results/comparison/`.

## Important Artifacts

| Artifact | Purpose |
|---|---|
| `data/processed/recipes.json` | Authoritative processed recipe data |
| `data/processed/triples.tsv` | Current exported KG triples |
| `results/transe/trained_model.pkl` | Saved TransE model |
| `results/distmult/trained_model.pkl` | Saved DistMult model |
| `results/transe/evaluation_results.json` | TransE test metrics |
| `results/distmult/evaluation_results.json` | DistMult test metrics |
| `results/distmult/recommendations.csv` | DistMult CLI recommendations |
| `results/comparison/recommendation_comparison.csv` | Jaccard, TransE, and DistMult comparison |

## Scope

Implemented:

- Data ingestion
- Recipe preprocessing
- Knowledge graph construction
- Triple export
- TransE and DistMult embeddings
- Link-prediction evaluation
- Recipe similarity and recommendation
- Jaccard comparison
- Explainable ingredient substitution backend

Not implemented:

- LLM-generated substitutions
- Neural substitution generation
- Node2Vec
- GNNs
- Link prediction beyond the current evaluations
- A production deployment service