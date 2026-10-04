from pathlib import Path
import json
from collections import Counter

import streamlit as st
import torch

from recommend_distmult import load_recipe_embeddings
from ingredient_substitution import find_substitutes


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPES_FILE = PROJECT_ROOT / "data" / "processed" / "recipes.json"



st.set_page_config(
    page_title="Recipe Recommendation System",
    page_icon="🍽️",
    layout="wide",
)



@st.cache_data
def load_recipes():
    with RECIPES_FILE.open("r", encoding="utf-8") as input_file:
        recipes = json.load(input_file)

    return {
        recipe["name"]: recipe
        for recipe in recipes
    }


@st.cache_resource(show_spinner="Loading the trained DistMult model...")
def load_distmult_embeddings():
    return load_recipe_embeddings()


def recommend_from_embeddings(recipe_name, top_k, model_data):
    (
        recipe_id_by_name,
        recipe_name_by_id,
        recipe_entity_ids,
        embeddings,
    ) = model_data

    if recipe_name not in recipe_id_by_name:
        raise ValueError(f"Recipe not found: {recipe_name}")

    source_id = recipe_id_by_name[recipe_name]

    try:
        source_index = recipe_entity_ids.index(source_id)
    except ValueError:
        raise ValueError(
            f"Recipe '{recipe_name}' does not have an embedding."
        )

    normalized_embeddings = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1,
    )

    similarities = (
        normalized_embeddings
        @ normalized_embeddings[source_index]
    )

    ranked_indices = torch.argsort(
        similarities,
        descending=True,
    )

    recommendations = []

    for candidate_index in ranked_indices.tolist():

        candidate_id = recipe_entity_ids[candidate_index]

        if candidate_id == source_id:
            continue

        candidate_name = recipe_name_by_id[candidate_id]

        recommendations.append(
            {
                "Rank": len(recommendations) + 1,
                "Recipe": candidate_name,
                "Similarity": float(
                    similarities[candidate_index]
                ),
            }
        )

        if len(recommendations) == top_k:
            break

    return recommendations


def get_ingredient_names(recipe):
    ingredients = recipe.get("ingredients", [])

    names = []

    for ingredient in ingredients:
        name = ingredient.get("name")

        if name:
            names.append(name.strip())

    return names


def calculate_ingredient_overlap(recipe1, recipe2):
    ingredients1 = {
        ingredient.lower()
        for ingredient in get_ingredient_names(recipe1)
    }

    ingredients2 = {
        ingredient.lower()
        for ingredient in get_ingredient_names(recipe2)
    }

    shared = ingredients1 & ingredients2
    union = ingredients1 | ingredients2

    if not union:
        jaccard = 0.0
    else:
        jaccard = len(shared) / len(union)

    return shared, jaccard


GRAPH_STATS = {
    "Recipes": 790,
    "Ingredients": 1147,
    "Ingredient Types": 14,
    "Categories": 14,
    "Areas": 37,
    "Ingredient Variants": 18,
    "HAS_INGREDIENT": 7861,
    "HAS_CATEGORY": 790,
    "HAS_AREA": 600,
    "IS_A": 55,
}


TOP_INGREDIENTS = [
    ("Salt", 299),
    ("Garlic", 227),
    ("Onion", 217),
    ("Olive Oil", 180),
    ("Butter", 171),
    ("Sugar", 149),
    ("Water", 133),
    ("Egg", 117),
    ("Milk", 115),
    ("Vegetable Oil", 110),
]


TOP_RECIPES_BY_INGREDIENT_COUNT = [
    ("Lamb Biryani", 20),
    ("Panang chicken curry (kaeng panang gai)", 20),
    ("Szechuan Beef", 20),
    ("Bigos (Hunters Stew)", 20),
    ("Chivito sandwich", 20),
    ("Smoky Lentil Chili with Squash", 19),
    ("Vegetable Shepherds Pie", 19),
    ("Chicken Basquaise", 19),
    ("General Tsos Chicken", 19),
    ("Chinese Orange Chicken", 19),
]


RECIPES_PER_CATEGORY = [
    ("Dessert", 167),
    ("Vegetarian", 100),
    ("Beef", 95),
    ("Seafood", 84),
    ("Side", 84),
    ("Chicken", 80),
    ("Pork", 60),
    ("Lamb", 33),
    ("Miscellaneous", 33),
    ("Breakfast", 19),
    ("Starter", 14),
    ("Pasta", 12),
    ("Vegan", 7),
    ("Goat", 2),
]


TOP_AREAS = [
    ("British", 59),
    ("Spanish", 48),
    ("United States", 33),
    ("Turkish", 30),
    ("France", 28),
    ("Thai", 27),
    ("Vietnamese", 27),
    ("Jamaican", 27),
    ("Chinese", 27),
    ("Polish", 27),
]


TOP_INGREDIENT_PAIRS = [
    ("Garlic + Onion", 113),
    ("Garlic + Salt", 101),
    ("Onion + Salt", 99),
    ("Pepper + Salt", 91),
    ("Garlic + Olive Oil", 91),
    ("Salt + Water", 88),
    ("Olive Oil + Onion", 78),
    ("Butter + Salt", 75),
    ("Salt + Sugar", 75),
    ("Butter + Milk", 65),
]


def show_graph_statistics():

    st.subheader("📊 Knowledge Graph Statistics")

    st.caption(
        "Overview of the recipe knowledge graph used by the system."
    )

   
    st.markdown("### Nodes")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Recipes", GRAPH_STATS["Recipes"])
        st.metric("Ingredients", GRAPH_STATS["Ingredients"])

    with col2:
        st.metric("Ingredient Types", GRAPH_STATS["Ingredient Types"])
        st.metric("Categories", GRAPH_STATS["Categories"])

    with col3:
        st.metric("Areas", GRAPH_STATS["Areas"])
        st.metric(
            "Ingredient Variants",
            GRAPH_STATS["Ingredient Variants"],
        )

    
    st.markdown("### Relationships")

    relationship_data = [
        {
            "Relationship": "HAS_INGREDIENT",
            "Count": GRAPH_STATS["HAS_INGREDIENT"],
        },
        {
            "Relationship": "HAS_CATEGORY",
            "Count": GRAPH_STATS["HAS_CATEGORY"],
        },
        {
            "Relationship": "HAS_AREA",
            "Count": GRAPH_STATS["HAS_AREA"],
        },
        {
            "Relationship": "IS_A",
            "Count": GRAPH_STATS["IS_A"],
        },
    ]

    st.dataframe(
        relationship_data,
        hide_index=True,
        use_container_width=True,
    )

   
    st.markdown("### 🧂 Most Common Ingredients")

    ingredient_data = [
        {
            "Rank": index,
            "Ingredient": ingredient,
            "Recipes": count,
        }
        for index, (ingredient, count)
        in enumerate(TOP_INGREDIENTS, start=1)
    ]

    st.dataframe(
        ingredient_data,
        hide_index=True,
        use_container_width=True,
    )

  
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📂 Recipes per Category")

        category_data = [
            {
                "Category": category,
                "Recipes": count,
            }
            for category, count in RECIPES_PER_CATEGORY
        ]

        st.dataframe(
            category_data,
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.markdown("### 🌍 Top Recipe Areas")

        area_data = [
            {
                "Area": area,
                "Recipes": count,
            }
            for area, count in TOP_AREAS
        ]

        st.dataframe(
            area_data,
            hide_index=True,
            use_container_width=True,
        )


try:

    recipes = load_recipes()
    model_data = load_distmult_embeddings()

    recipe_id_by_name, _, _, _ = model_data

    recipe_names = sorted(recipe_id_by_name)

except (
    FileNotFoundError,
    OSError,
    RuntimeError,
) as error:

    st.error(
        f"Required project files could not be loaded: {error}"
    )

    st.stop()


st.title("🍽️ Recipe Recommendation System")

st.write(
    "Explore recipes and discover similar recipes using "
    "a knowledge graph and DistMult embeddings."
)

st.divider()



with st.sidebar:

    st.header("Recommendation Settings")

    default_index = (
        recipe_names.index("Apple Frangipan Tart")
        if "Apple Frangipan Tart" in recipe_names
        else 0
    )

    selected_recipe = st.selectbox(
        "Select a recipe",
        options=recipe_names,
        index=default_index,
    )

    top_k = st.selectbox(
        "Number of recommendations",
        options=[5, 10, 15, 20],
        index=0,
    )

    recommend_button = st.button(
        "Recommend",
        type="primary",
        use_container_width=True,
    )


recipe = recipes.get(selected_recipe)

if recipe is None:

    st.error(
        "Recipe information could not be found."
    )

    st.stop()


st.header(selected_recipe)

col1, col2, col3 = st.columns(3)

with col1:

    category = recipe.get(
        "category",
        "Unknown",
    )

    st.metric(
        "Category",
        category,
    )

with col2:

    area = recipe.get(
        "area",
        "Unknown",
    )

    st.metric(
        "Cuisine / Area",
        area,
    )

with col3:

    ingredient_count = len(
        get_ingredient_names(recipe)
    )

    st.metric(
        "Number of Ingredients",
        ingredient_count,
    )

st.subheader("Ingredients")

ingredients = recipe.get(
    "ingredients",
    [],
)

if ingredients:

    ingredient_rows = []

    for ingredient in ingredients:

        name = ingredient.get(
            "name",
            "",
        )

        measure = ingredient.get(
            "measure",
            "",
        )

        ingredient_rows.append(
            {
                "Ingredient": name,
                "Measure": measure,
            }
        )

    st.dataframe(
        ingredient_rows,
        hide_index=True,
        use_container_width=True,
    )

else:

    st.info(
        "No ingredient information available."
    )


st.subheader("Instructions")

instructions = recipe.get(
    "instructions"
)

if instructions:

    st.write(instructions)

else:

    st.info(
        "No instructions available."
    )


if recommend_button:

    st.divider()

    st.header("🔎 Recommended Recipes")

    try:

        recommendations = recommend_from_embeddings(
            selected_recipe,
            top_k,
            model_data,
        )

    except (
        ValueError,
        IndexError,
    ) as error:

        st.error(str(error))
        st.stop()


    for recommendation in recommendations:

        recommended_name = recommendation[
            "Recipe"
        ]

        similarity = recommendation[
            "Similarity"
        ]

        recommended_recipe = recipes.get(
            recommended_name
        )

        if recommended_recipe is None:
            continue

        shared, jaccard = calculate_ingredient_overlap(
            recipe,
            recommended_recipe,
        )

        with st.container(border=True):

            st.subheader(
                f"{recommendation['Rank']}. "
                f"{recommended_name}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "DistMult Similarity",
                    f"{similarity:.4f}",
                )

            with col2:

                st.metric(
                    "Shared Ingredients",
                    len(shared),
                )

            with col3:

                st.metric(
                    "Ingredient Overlap",
                    f"{jaccard:.2%}",
                )

            if shared:

                st.write(
                    "**Shared ingredients:** "
                    + ", ".join(
                        sorted(shared)
                    )
                )

            else:

                st.write(
                    "**Shared ingredients:** "
                    "None identified"
                )

            st.caption(
                "DistMult similarity is learned from "
                "the knowledge-graph representation. "
                "Ingredient overlap is calculated "
                "separately using Jaccard similarity."
            )



st.divider()

st.header("🥕 Ingredient Substitution")

st.write(
    "Find ingredients that frequently occur in the same "
    "recipes as the selected ingredient."
)


all_ingredients = set()

for recipe_data in recipes.values():

    for ingredient in recipe_data.get(
        "ingredients",
        [],
    ):

        name = ingredient.get("name")

        if name:
            all_ingredients.add(
                name.strip()
            )


ingredient_names = sorted(
    all_ingredients
)


selected_ingredient = st.selectbox(
    "Select an ingredient",
    options=ingredient_names,
)


substitution_limit = st.selectbox(
    "Number of suggestions",
    options=[5, 10],
    index=0,
)


if st.button(
    "Find Substitutes",
    use_container_width=True,
):

    try:

        substitutions = find_substitutes(
            selected_ingredient,
            limit=substitution_limit,
        )

        if not substitutions:

            st.info(
                "No data-supported substitution "
                "candidates were found."
            )

        else:

            st.subheader(
                f"Possible substitutes for "
                f"{selected_ingredient}"
            )

            st.dataframe(
                substitutions,
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Substitution candidates are based on "
                "ingredient co-occurrence in the recipe "
                "dataset and graph-supported relationships."
            )

    except (
        ValueError,
        RuntimeError,
        OSError,
    ) as error:

        st.error(str(error))


st.divider()

with st.expander(
    "📊 Explore Knowledge Graph Statistics"
):

    show_graph_statistics()



st.divider()

with st.expander(
    "ℹ️ About"
):

    st.write(
        """
        This system uses recipes from TheMealDB to construct
        a knowledge graph containing recipes, ingredients,
        categories, areas, and semantic ingredient information.

        Recipe relationships are exported as knowledge-graph
        triples and used to train a DistMult knowledge graph
        embedding model.

        The trained embeddings are then used to identify
        recipes that are similar in the learned embedding
        space.

        Ingredient overlap is calculated separately using
        Jaccard similarity and is displayed to help explain
        the recommendations.

        Ingredient substitution uses recipe co-occurrence
        information to identify ingredients that frequently
        appear together with the selected ingredient.
        """
    )