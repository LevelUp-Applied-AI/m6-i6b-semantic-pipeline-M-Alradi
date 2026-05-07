"""
Tier 1 — Hybrid Search

Combine:
1. Semantic Search (DistilBERT embeddings)
2. TF-IDF Keyword Search

Hybrid score:
    alpha * semantic_score + (1 - alpha) * tfidf_score
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from semantic_pipeline import (
    load_and_preprocess,
    compute_embeddings
)


def compute_tfidf_matrix(texts):
    """
    Build TF-IDF vectors for the corpus.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


def tfidf_search(query, vectorizer, tfidf_matrix):
    """
    Compute TF-IDF cosine similarities between query and corpus.
    """
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    return similarities


def semantic_similarity(query_embedding, corpus_embeddings):
    """
    Compute semantic cosine similarities.
    """
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        corpus_embeddings
    ).flatten()

    return similarities


def hybrid_search(
    query,
    alpha,
    corpus_texts,
    corpus_embeddings,
    vectorizer,
    tfidf_matrix,
    tokenizer,
    model,
    top_k=5
):
    """
    Combine semantic and TF-IDF scores.
    """

    # Query embedding
    query_embedding = compute_embeddings(
        [query],
        tokenizer,
        model
    )[0]

    # Semantic scores
    semantic_scores = semantic_similarity(
        query_embedding,
        corpus_embeddings
    )

    # TF-IDF scores
    tfidf_scores = tfidf_search(
        query,
        vectorizer,
        tfidf_matrix
    )

    # Hybrid scores
    hybrid_scores = (
        alpha * semantic_scores
        + (1 - alpha) * tfidf_scores
    )

    # Sort descending
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "text": corpus_texts[idx],
            "semantic_score": semantic_scores[idx],
            "tfidf_score": tfidf_scores[idx],
            "hybrid_score": hybrid_scores[idx]
        })

    return results


def compare_rankings(results_by_alpha):
    """
    Compare ranking positions across alpha values.
    """

    print("\n================ RANKING CHANGES ================\n")

    for alpha, results in results_by_alpha.items():

        print(f"Alpha = {alpha}")

        for rank, result in enumerate(results, start=1):

            preview = result["text"][:80].replace("\n", " ")

            print(f"{rank}. {preview}...")

        print("-" * 60)


if __name__ == "__main__":

    # Load data
    df = load_and_preprocess("data/climate_articles.csv")

    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} texts")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    model.eval()

    # Semantic embeddings
    print("\nComputing embeddings...")
    corpus_embeddings = compute_embeddings(
        texts,
        tokenizer,
        model
    )

    print("Done.")

    # TF-IDF
    print("\nBuilding TF-IDF matrix...")
    vectorizer, tfidf_matrix = compute_tfidf_matrix(texts)

    print("Done.")

    # Example queries
    with open("data/example_queries.txt") as f:
        queries = [line.strip() for line in f if line.strip()]

    # Alpha values
    alpha_values = [0.3, 0.5, 0.7, 1.0]

    # Run experiments
    for query in queries:

        print("\n")
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results_by_alpha = {}

        for alpha in alpha_values:

            print(f"\nAlpha = {alpha}")
            print("-" * 60)

            results = hybrid_search(
                query=query,
                alpha=alpha,
                corpus_texts=texts,
                corpus_embeddings=corpus_embeddings,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                tokenizer=tokenizer,
                model=model,
                top_k=5
            )

            results_by_alpha[alpha] = results

            for rank, result in enumerate(results, start=1):

                print(f"\nRank #{rank}")

                print(
                    f"Hybrid: {result['hybrid_score']:.4f} | "
                    f"Semantic: {result['semantic_score']:.4f} | "
                    f"TF-IDF: {result['tfidf_score']:.4f}"
                )

                print(f"Text: {result['text'][:150]}...")

        # Compare rankings
        compare_rankings(results_by_alpha)