# Milestone 5: Embeddings, pgvector and Semantic Retrieval

This learning guide walks through the mathematical foundations and systems engineering concepts behind vector embeddings, pgvector indexing, and semantic retrieval in RAGForge.

---

## 1. What is an Embedding?

In traditional lexical search (such as BM25 or keyword matching), documents and queries are treated as sparse collections of exact terms. If a user searches for *"automobile repair"*, an exact keyword search will miss a document discussing *"car maintenance"*, despite identical underlying semantic intent.

A **vector embedding** is a dense numerical representation of unstructured text in a continuous high-dimensional vector space:

$$\mathbf{e} = f_\theta(\text{text}) \in \mathbb{R}^d$$

Where:
- $d$ is the vector dimension (e.g. 384 for `all-MiniLM-L6-v2`, 1536 for OpenAI `text-embedding-3-small`).
- $f_\theta$ is a deep neural transformer model trained on massive contrastive learning objectives.

### Why Semantically Similar Text Has Nearby Vectors

During contrastive training, the embedding model adjusts weights so that pairs of sentences with similar meanings (such as questions and their corresponding answers, or synonyms) have representations with minimal angular distance between them. Conversely, unrelated sentences are pushed apart into distant regions of the hypersphere.

Because words with contextual proximity project onto correlated coordinates, semantic relationships emerge geometrically.

---

## 2. Vector Dimensions and Model Compatibility

### Dimensionality
Vector dimension represents the number of latent continuous features captured by the embedding model:
- `sentence-transformers/all-MiniLM-L6-v2`: 384 dimensions (~80MB model weight footprint; fast local CPU inference; ideal for development and edge workloads).
- `sentence-transformers/all-mpnet-base-v2`: 768 dimensions (higher quality; larger memory footprint).
- `Azure OpenAI text-embedding-3-small`: 1536 dimensions (supports dimension reduction via Matryoshka representation learning).

### Strict Model Compatibility
Vectors from different models—or even the same model with different dimensionalities—**cannot be compared**:
- Each model defines its own latent coordinate system. Point $[0.1, -0.4, \dots]$ in Model A means something completely different from $[0.1, -0.4, \dots]$ in Model B.
- Comparing vectors across different models or dimensions produces meaningless mathematical noise.
- **RAGForge Guardrail**: Every embedding row in `chunk_embeddings` records both `model_name` and `dimension`. Query embedding requests only search against vectors matching the active query model name.

---

## 3. Cosine Similarity vs. Cosine Distance

Given two vectors $\mathbf{u}$ and $\mathbf{v}$:

### Cosine Similarity
$$\text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

- When vectors are unit-normalized ($\|\mathbf{u}\| = \|\mathbf{v}\| = 1.0$), cosine similarity simplifies directly to the dot product: $\mathbf{u} \cdot \mathbf{v}$.
- Range: $[-1.0, 1.0]$. A value of $1.0$ indicates identical direction; $0.0$ indicates orthogonality (unrelated); $-1.0$ indicates opposite meaning.

### Cosine Distance
Nearest-neighbor search and distance metrics require non-negative distances where smaller numbers indicate closer neighbors:

$$\text{cosine\_distance}(\mathbf{u}, \mathbf{v}) = 1.0 - \text{cosine\_similarity}(\mathbf{u}, \mathbf{v})$$

- Range for normalized embeddings: $[0.0, 2.0]$.
- Exact match: $\text{distance} = 0.0 \implies \text{similarity} = 1.0$.

In `pgvector`, the `<=>` operator computes cosine distance directly:
```sql
SELECT chunk_id, embedding <=> :query_vector AS distance
FROM chunk_embeddings
ORDER BY distance ASC
LIMIT 5;
```

---

## 4. How pgvector Works in PostgreSQL

`pgvector` adds a native `vector` data type and distance operators directly into PostgreSQL:

### Distance Operators
- `<=>`: Cosine distance ($1 - \text{cosine similarity}$).
- `<->`: Euclidean / L2 distance ($\sqrt{\sum (u_i - v_i)^2}$).
- `<#>`: Negative inner product ($-\mathbf{u} \cdot \mathbf{v}$).

### Indexing Options
For small to medium datasets (<100,000 chunks), exact flat scan (`ORDER BY embedding <=> query LIMIT K`) executes in single-digit milliseconds with 100% recall.

For production datasets scaling to millions of chunks, `pgvector` provides two approximate nearest neighbor (ANN) index types:
1. **IVFFlat (Inverted File Flat)**:
   - Partitions vector space into Voronoi cells using k-means clustering.
   - Searches only the lists closest to the query vector.
   - Faster build times, lower memory usage, moderate query speed.
2. **HNSW (Hierarchical Navigable Small World)**:
   - Builds a multi-layer graph where lower layers contain detailed local neighbor connections and upper layers contain long-range skip links.
   - Superior query latency and recall trade-offs compared to IVFFlat, but higher memory consumption and slower index build times.

---

## 5. Architectural Separation: Worker vs. API Handlers

### Why Embedding Generation Stays Out of HTTP Handlers
1. **Latency & Event Loop Starvation**:
   - Neural transformer inference is CPU/GPU-bound.
   - Running model inference in an async request handler blocks Python's single-threaded event loop, delaying all concurrent HTTP traffic.
2. **Memory Footprint & Concurrency Control**:
   - Web workers scale with connection volume. Loading heavy PyTorch models into multiple web server worker processes can cause Out-Of-Memory (OOM) crashes.
   - Confining ingestion embedding to an asynchronous ARQ worker isolates heavy compute to dedicated worker containers with bounded batch sizes (`EMBEDDING_BATCH_SIZE=32`).
3. **Query Embedding Exception**:
   - In search (`POST /api/v1/search`), the single query string *must* be embedded synchronously to return results. To keep the event loop non-blocking, local model inference is delegated to worker threads via `asyncio.to_thread`.

---

## 6. Fake Embeddings vs. Real Semantic Quality

In testing, `FakeEmbeddingProvider` generates deterministic pseudo-random unit vectors based on string hashes:
- **What Fake Embeddings Prove**: System integration, database persistence, dimension validation, error handling, query parameter filtering, and API contracts.
- **What Fake Embeddings CANNOT Prove**: Semantic retrieval quality. Hashing tokens does not model linguistic synonymy or semantic overlap. Testing semantic relevance requires real neural weights (`all-MiniLM-L6-v2` or Azure OpenAI).

---

## 7. Retrieval Evaluation: Recall@K

To measure search quality objectively, RAGForge uses standard information retrieval (IR) benchmarks:

### Recall@K
$$\text{Recall@K} = \frac{\text{Relevant Chunks Retrieved in Top K}}{\text{Total Ground-Truth Relevant Chunks}}$$

- **Recall@1**: Did the exact ground-truth chunk rank as the #1 top match?
- **Recall@5**: Is the ground-truth chunk present anywhere in the top 5 returned results?

In RAG systems, **Recall@5 is a critical leading indicator**: if the relevant context chunk is not present in the top 5 retrieved items, downstream LLM generation will suffer from hallucination or lack of factual grounding.

---

## 8. Source Attribution and Citations

Every retrieved item returned by `POST /api/v1/search` includes:
- `document_id` and `filename`
- `chunk_index`
- `page_number` (preserved from original PDF pages)
- `similarity_score` and `cosine_distance`

In later milestones, this metadata enables the system to generate verifiable footnotes and citations for every statement generated by the Language Model.
