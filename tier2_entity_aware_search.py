"""
Tier 2 — Entity-Aware Search

Features:
1. Entity-type filtering
2. Entity-based score boosting
"""

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from semantic_pipeline import (
    load_and_preprocess,
    run_ner,
    compute_embeddings
)


def semantic_search_with_scores(
    query_embedding,
    corpus_embeddings,
    corpus_texts,
    top_k=10
):
    """
    Standard semantic search returning scores.
    """

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        corpus_embeddings
    ).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "text_index": idx,
            "text": corpus_texts[idx],
            "score": similarities[idx]
        })

    return results


def filter_by_entity_type(
    search_results,
    entity_df,
    entity_type
):
    """
    Keep only documents containing a specific entity type.

    Example:
        ORG
        PERSON
        GPE
    """

    filtered_results = []

    for result in search_results:

        text_index = result["text_index"]

        matching_entities = entity_df[
            (entity_df["text_index"] == text_index)
            &
            (entity_df["entity_label"] == entity_type)
        ]

        if not matching_entities.empty:

            entity_list = matching_entities[
                ["entity_text", "entity_label"]
            ].to_dict("records")

            result["matching_entities"] = entity_list

            filtered_results.append(result)

    return filtered_results


def boost_by_entity(
    search_results,
    entity_df,
    target_entity,
    boost=0.15
):
    """
    Boost documents mentioning a target entity.

    Example:
        IPCC
        NASA
        UN
    """

    boosted_results = []

    target_entity = target_entity.lower()

    for result in search_results:

        text_index = result["text_index"]

        doc_entities = entity_df[
            entity_df["text_index"] == text_index
        ]

        entity_texts = (
            doc_entities["entity_text"]
            .str.lower()
            .tolist()
        )

        boosted_score = result["score"]

        if target_entity in entity_texts:
            boosted_score += boost

        boosted_results.append({
            "text_index": text_index,
            "text": result["text"],
            "original_score": result["score"],
            "boosted_score": boosted_score,
            "contains_target_entity":
                target_entity in entity_texts
        })

    boosted_results = sorted(
        boosted_results,
        key=lambda x: x["boosted_score"],
        reverse=True
    )

    return boosted_results


def print_results(title, results, top_k=5):
    """
    Pretty-print results.
    """

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(results[:top_k], start=1):

        print(f"\nRank #{rank}")

        if "boosted_score" in result:

            print(
                f"Original: {result['original_score']:.4f}"
            )

            print(
                f"Boosted: {result['boosted_score']:.4f}"
            )

            print(
                f"Contains Target Entity: "
                f"{result['contains_target_entity']}"
            )

        else:
            print(f"Score: {result['score']:.4f}")

        print(f"Text: {result['text'][:180]}...")

        if "matching_entities" in result:
            print(
                f"Matching Entities: "
                f"{result['matching_entities'][:5]}"
            )


if __name__ == "__main__":

    # Load data
    df = load_and_preprocess(
        "data/climate_articles.csv"
    )

    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} texts")

    # Run NER
    print("\nRunning NER...")
    entity_df = run_ner(texts)

    print(f"Extracted {len(entity_df)} entities")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    model.eval()

    # Compute embeddings
    print("\nComputing embeddings...")

    corpus_embeddings = compute_embeddings(
        texts,
        tokenizer,
        model
    )

    print("Done.")

    # Example queries
    with open("data/example_queries.txt") as f:
        queries = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # Example entity filters
    entity_filter = "ORG"

    # Example boosted entity
    boosted_entity = "IPCC"

    for query in queries:

        print("\n\n")
        print("#" * 100)
        print(f"QUERY: {query}")
        print("#" * 100)

        # Query embedding
        query_embedding = compute_embeddings(
            [query],
            tokenizer,
            model
        )[0]

        # Base semantic search
        base_results = semantic_search_with_scores(
            query_embedding,
            corpus_embeddings,
            texts,
            top_k=10
        )

        print_results(
            "BASE SEMANTIC SEARCH",
            base_results
        )

        # Entity-type filtering
        filtered_results = filter_by_entity_type(
            base_results,
            entity_df,
            entity_filter
        )

        print_results(
            f"FILTERED RESULTS (Entity Type = {entity_filter})",
            filtered_results
        )

        # Entity boosting
        boosted_results = boost_by_entity(
            base_results,
            entity_df,
            boosted_entity,
            boost=0.15
        )

        print_results(
            f"BOOSTED RESULTS (Entity = {boosted_entity})",
            boosted_results
        )