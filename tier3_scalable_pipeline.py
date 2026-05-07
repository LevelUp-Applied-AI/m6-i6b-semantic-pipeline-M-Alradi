"""
Tier 3 — Pipeline Architecture for Scale

Features:
1. Separate indexing from querying
2. Save/load embeddings and entities
3. Fast querying using precomputed embeddings
4. Incremental indexing
5. Benchmark query latency
"""

import os
import json
import time
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from semantic_pipeline import (
    load_and_preprocess,
    run_ner,
    compute_embeddings
)


INDEX_DIR = "indexes"

EMBEDDINGS_PATH = os.path.join(
    INDEX_DIR,
    "embeddings.npy"
)

TEXTS_PATH = os.path.join(
    INDEX_DIR,
    "texts.json"
)

ENTITIES_PATH = os.path.join(
    INDEX_DIR,
    "entities.json"
)


class Indexer:
    """
    Builds and stores the search index.
    """

    def __init__(self, tokenizer, model):

        self.tokenizer = tokenizer
        self.model = model

        self.texts = []
        self.embeddings = None
        self.entities = None

    def build_index(self, texts):

        print("\nComputing embeddings...")

        self.embeddings = compute_embeddings(
            texts,
            self.tokenizer,
            self.model
        )

        print("Running NER...")

        self.entities = run_ner(texts)

        self.texts = texts

        print("Index built.")

    def save_index(self):

        os.makedirs(INDEX_DIR, exist_ok=True)

        # Save embeddings
        np.save(
            EMBEDDINGS_PATH,
            self.embeddings
        )

        # Save texts
        with open(TEXTS_PATH, "w") as f:
            json.dump(self.texts, f)

        # Save entities
        self.entities.to_json(
            ENTITIES_PATH,
            orient="records"
        )

        print("\nIndex saved to disk.")

    def load_index(self):

        self.embeddings = np.load(
            EMBEDDINGS_PATH
        )

        with open(TEXTS_PATH) as f:
            self.texts = json.load(f)

        self.entities = pd.read_json(
            ENTITIES_PATH
        )

        print("\nIndex loaded from disk.")

    def add_documents(self, new_texts):
        """
        Add new documents without recomputing old embeddings.
        """

        print("\nAdding new documents...")

        new_embeddings = compute_embeddings(
            new_texts,
            self.tokenizer,
            self.model
        )

        new_entities = run_ner(new_texts)

        # Adjust text indexes
        offset = len(self.texts)

        new_entities["text_index"] += offset

        # Merge
        self.embeddings = np.vstack([
            self.embeddings,
            new_embeddings
        ])

        self.entities = pd.concat([
            self.entities,
            new_entities
        ])

        self.texts.extend(new_texts)

        print(f"Added {len(new_texts)} documents.")


class Searcher:
    """
    Query system using precomputed index.
    """

    def __init__(
        self,
        tokenizer,
        model
    ):

        self.tokenizer = tokenizer
        self.model = model

        self.embeddings = None
        self.entities = None
        self.texts = None

    def load_index(self):

        self.embeddings = np.load(
            EMBEDDINGS_PATH
        )

        with open(TEXTS_PATH) as f:
            self.texts = json.load(f)

        self.entities = pd.read_json(
            ENTITIES_PATH
        )

        print("\nSearcher loaded index.")

    def search(self, query, top_k=5):

        # Only compute query embedding
        query_embedding = compute_embeddings(
            [query],
            self.tokenizer,
            self.model
        )[0]

        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.embeddings
        ).flatten()

        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        results = []

        for idx in top_indices:

            entity_rows = self.entities[
                self.entities["text_index"] == idx
            ]

            entity_list = entity_rows[
                ["entity_text", "entity_label"]
            ].to_dict("records")

            results.append({
                "text": self.texts[idx],
                "score": similarities[idx],
                "entities": entity_list
            })

        return results


def benchmark_search(
    query,
    texts,
    embeddings,
    tokenizer,
    model,
    searcher
):
    """
    Compare:
    1. On-the-fly search
    2. Precomputed index search
    """

    print("\n")
    print("=" * 80)
    print("BENCHMARK")
    print("=" * 80)

    # On-the-fly
    start = time.time()

    query_embedding = compute_embeddings(
        [query],
        tokenizer,
        model
    )[0]

    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    ).flatten()

    np.argsort(similarities)[::-1][:5]

    end = time.time()

    on_the_fly_time = end - start

    # Precomputed
    start = time.time()

    searcher.search(query)

    end = time.time()

    indexed_time = end - start

    print(
        f"\nOn-the-fly query time: "
        f"{on_the_fly_time:.4f} seconds"
    )

    print(
        f"Indexed query time: "
        f"{indexed_time:.4f} seconds"
    )


if __name__ == "__main__":

    # Load dataset
    df = load_and_preprocess(
        "data/climate_articles.csv"
    )

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

    # =====================================================
    # INDEXING PHASE
    # =====================================================

    indexer = Indexer(
        tokenizer,
        model
    )

    indexer.build_index(texts)

    indexer.save_index()

    # =====================================================
    # QUERY PHASE
    # =====================================================

    searcher = Searcher(
        tokenizer,
        model
    )

    searcher.load_index()

    # Example queries
    with open("data/example_queries.txt") as f:
        queries = [
            line.strip()
            for line in f
            if line.strip()
        ]

    for query in queries:

        print("\n")
        print("#" * 100)
        print(f"QUERY: {query}")
        print("#" * 100)

        results = searcher.search(
            query,
            top_k=5
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nRank #{rank}")

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Text: "
                f"{result['text'][:180]}..."
            )

            print(
                f"Entities: "
                f"{result['entities'][:5]}"
            )

    # =====================================================
    # ADD NEW DOCUMENTS
    # =====================================================

    new_documents = [
        "The IPCC released a new climate report discussing global emissions.",
        "NASA scientists observed rising sea levels in coastal regions."
    ]

    indexer.load_index()

    indexer.add_documents(new_documents)

    indexer.save_index()

    # =====================================================
    # BENCHMARK
    # =====================================================

    benchmark_search(
        query="climate policy and emissions",
        texts=texts,
        embeddings=indexer.embeddings,
        tokenizer=tokenizer,
        model=model,
        searcher=searcher
    )