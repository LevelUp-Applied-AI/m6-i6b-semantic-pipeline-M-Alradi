# Tier 1 Report Analysis

> Hybrid search performed better for queries containing important technical keywords or named concepts because TF-IDF rewarded exact term matches. Pure semantic search worked better for broader conceptual queries and paraphrased language. Lower alpha values improved precision for keyword-heavy searches, while higher alpha values improved retrieval of semantically related articles. A balanced alpha such as 0.5 or 0.7 often produced the most useful results overall because it combined lexical accuracy with semantic understanding.


# Tier 2 Report Analysis

> Entity-aware search outperformed pure semantic retrieval when queries depended on structured information such as organizations, people, or locations. Entity filtering improved precision by removing semantically related documents that lacked the required entity types. Entity boosting helped prioritize documents mentioning important concepts like “IPCC” or “UN,” making rankings more useful for targeted information retrieval. However, the approach depends heavily on NER quality. Incorrect or missing entity extraction can remove relevant documents or fail to boost important results. Another limitation is that semantic relevance and entity presence do not always align, so aggressive filtering may reduce recall by excluding otherwise useful documents.


# Tier 3 Key Design Improvement

Before:

```
QUERY
  ↓
compute ALL embeddings
  ↓
search
```

After:

```
OFFLINE:
documents → embeddings → save

ONLINE:
query → query embedding → search stored vectors
```

# Tier 3 Report
> To scale this system to one million documents, several architectural changes would be required. Exact cosine similarity over all embeddings would become too slow, so approximate nearest neighbor search libraries such as FAISS or ScaNN would be needed. Embeddings and metadata would likely be stored in a vector database rather than local NumPy files. Query serving and indexing should be separated into independent services, allowing indexing to run asynchronously in the background while search remains responsive. Batch embedding generation on GPUs would become essential for throughput. Entity extraction and document preprocessing would also need distributed processing frameworks such as Spark or Ray. For reliability and scalability, the system would benefit from cloud object storage, caching layers, and sharded vector indexes distributed across multiple machines.
