"""Retrieval Evaluation Script for RAGForge (Milestone 5).

Evaluates semantic retrieval performance (Recall@5, query latency, and result counts)
using a small, reproducible, labeled benchmark dataset and pgvector nearest-neighbor search.

Does not require Azure credentials. Runs locally using the configured sentence-transformers model.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from ragforge.core.config import Settings
from ragforge.db.models import ChunkEmbedding, Document, DocumentChunk, DocumentStatus
from ragforge.db.session import get_session_factory
from ragforge.services.embedding_provider import LocalSentenceTransformerProvider
from ragforge.services.retrieval_service import RetrievalService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_retrieval")


@dataclass
class EvalDoc:
    filename: str
    chunks: list[str]


@dataclass
class EvalQuery:
    query: str
    target_filename: str
    expected_chunk_substr: str


# 1. Benchmark Dataset (5 short multi-domain documents)
EVAL_DOCUMENTS: list[EvalDoc] = [
    EvalDoc(
        filename="db_connection_pooling.txt",
        chunks=[
            (
                "PostgreSQL connection pooling with PgBouncer manages backend server"
                " connections efficiently."
            ),
            (
                "Connection pools reuse active sockets to eliminate the high overhead of"
                " establishing new TCP sessions."
            ),
            (
                "Transaction pooling mode returns connections to the pool as soon as the"
                " current transaction finishes."
            ),
        ],
    ),
    EvalDoc(
        filename="distributed_queues.txt",
        chunks=[
            "ARQ is an asynchronous job queue for Python and asyncio built on top of Redis.",
            (
                "Workers pull serialized job payloads, execute coroutines concurrently, and"
                " persist results."
            ),
            (
                "Exponential backoff and bounded retries prevent transient worker crashes"
                " from overflowing queues."
            ),
        ],
    ),
    EvalDoc(
        filename="vector_embeddings.txt",
        chunks=[
            (
                "Dense vector embeddings capture semantic meaning by mapping text tokens to a"
                " continuous high-dimensional space."
            ),
            (
                "Cosine similarity measures the angle between two normalized vectors, ignoring"
                " their raw magnitudes."
            ),
            (
                "pgvector is an open-source vector similarity search extension for PostgreSQL"
                " supporting IVFFlat and HNSW indexing."
            ),
        ],
    ),
    EvalDoc(
        filename="sourdough_fermentation.txt",
        chunks=[
            (
                "Sourdough baking relies on wild yeast and lactic acid bacteria in a"
                " live starter culture."
            ),
            (
                "Autolyse is a technique where flour and water rest before salt is added,"
                " accelerating gluten development."
            ),
            (
                "Cold bulk fermentation develops complex acetic flavors and improves"
                " dough extensibility."
            ),
        ],
    ),
    EvalDoc(
        filename="audio_compression.txt",
        chunks=[
            (
                "Psychoacoustic masking models remove sound frequencies imperceptible to"
                " the human auditory system."
            ),
            (
                "Lossy audio encoders like AAC and MP3 discard psychoacoustically redundant"
                " frequency bands."
            ),
            (
                "Lossless formats like FLAC use linear predictive coding to compress"
                " waveform samples without fidelity loss."
            ),
        ],
    ),
]

# 2. Benchmark Queries with Ground-Truth Attribution
EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery(
        query="How does connection pooling reduce database overhead?",
        target_filename="db_connection_pooling.txt",
        expected_chunk_substr="overhead of establishing new TCP sessions",
    ),
    EvalQuery(
        query="What library runs asynchronous background jobs using Redis in Python?",
        target_filename="distributed_queues.txt",
        expected_chunk_substr="ARQ is an asynchronous job queue",
    ),
    EvalQuery(
        query="What PostgreSQL extension provides vector similarity search?",
        target_filename="vector_embeddings.txt",
        expected_chunk_substr="pgvector is an open-source vector similarity search extension",
    ),
    EvalQuery(
        query="Why is autolyse used in bread dough preparation?",
        target_filename="sourdough_fermentation.txt",
        expected_chunk_substr="accelerating gluten development",
    ),
    EvalQuery(
        query="How does perceptual audio compression eliminate inaudible frequencies?",
        target_filename="audio_compression.txt",
        expected_chunk_substr="Psychoacoustic masking models remove sound frequencies",
    ),
]


async def setup_eval_dataset(
    session: AsyncSession,
    provider: LocalSentenceTransformerProvider,
) -> dict[str, Any]:
    """Ingest evaluation documents, chunks, and embeddings into the database."""
    logger.info("Ingesting benchmark documents and generating dense embeddings...")
    created_doc_ids = []

    for doc_data in EVAL_DOCUMENTS:
        doc = Document(
            filename=f"eval_{doc_data.filename}",
            storage_key=f"eval/{doc_data.filename}",
            mime_type="text/plain",
            file_size_bytes=sum(len(c.encode("utf-8")) for c in doc_data.chunks),
            sha256_checksum="0" * 64,
            status=DocumentStatus.PROCESSED.value,
        )
        session.add(doc)
        await session.flush()
        created_doc_ids.append(doc.id)

        # Create chunks
        chunks = []
        for idx, text in enumerate(doc_data.chunks):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                content=text,
                page_number=1,
                character_count=len(text),
                token_count_estimate=len(text.split()),
            )
            chunks.append(chunk)
            session.add(chunk)
        await session.flush()

        # Generate dense embeddings
        texts = [c.content for c in chunks]
        vectors = await provider.embed_texts(texts)

        for chunk, vector in zip(chunks, vectors, strict=True):
            emb = ChunkEmbedding(
                chunk_id=chunk.id,
                provider="local",
                model_name=provider.model_name,
                dimension=provider.dimension,
                embedding=vector,
            )
            session.add(emb)

    await session.commit()
    return {"doc_ids": created_doc_ids}


async def cleanup_eval_dataset(session: AsyncSession, doc_ids: list[Any]) -> None:
    """Clean up benchmark documents from database."""
    logger.info("Cleaning up benchmark documents...")
    # Chunks and embeddings cascade via DB foreign keys
    await session.execute(delete(Document).where(Document.id.in_(doc_ids)))
    await session.commit()


async def run_evaluation() -> None:
    """Run retrieval evaluation suite and print benchmark report."""
    settings = Settings(
        EMBEDDING_PROVIDER="local",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        RETRIEVAL_DEFAULT_LIMIT=5,
        RETRIEVAL_MAX_LIMIT=10,
    )

    logger.info("Initializing LocalSentenceTransformerProvider (all-MiniLM-L6-v2)...")
    provider = LocalSentenceTransformerProvider(
        model_name=settings.EMBEDDING_MODEL,
        expected_dimension=settings.EMBEDDING_DIMENSION,
    )
    await provider.ensure_loaded()

    session_factory = get_session_factory()
    async with session_factory() as session:
        setup_meta = await setup_eval_dataset(session, provider)
        doc_ids = setup_meta["doc_ids"]

        retrieval_service = RetrievalService(
            db=session,
            embedding_provider=provider,
            settings=settings,
        )

        logger.info(f"Running evaluation on {len(EVAL_QUERIES)} benchmark queries (Recall@5)...")
        hits_at_1 = 0
        hits_at_5 = 0
        latencies_ms: list[float] = []

        print("\n" + "=" * 80)
        print("RAGFORGE SEMANTIC RETRIEVAL BENCHMARK REPORT (Milestone 5)")
        print("=" * 80)
        print(f"Model: {provider.model_name} | Dimension: {provider.dimension} | Top-K: 5\n")

        try:
            for i, q in enumerate(EVAL_QUERIES, 1):
                t0 = time.perf_counter()
                res = await retrieval_service.search(query=q.query, limit=5)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                # Check if target ground truth chunk is in results
                found_at_rank = None
                for rank, item in enumerate(res.items, 1):
                    if q.expected_chunk_substr.lower() in item.content.lower():
                        found_at_rank = rank
                        break

                if found_at_rank == 1:
                    hits_at_1 += 1
                    hits_at_5 += 1
                    status = f"HIT (Rank 1) - Similarity: {res.items[0].similarity_score:.4f}"
                elif found_at_rank is not None and found_at_rank <= 5:
                    hits_at_5 += 1
                    status = f"HIT (Rank {found_at_rank})"
                else:
                    status = "MISS"

                print(f"[{i}/{len(EVAL_QUERIES)}] Query: '{q.query}'")
                print(f"       Target: {q.target_filename}")
                print(f"       Result: {status} | Latency: {elapsed_ms:.2f} ms")
                if res.items:
                    print(f"       Top Match: '{res.items[0].content[:70]}...'")
                print("-" * 80)

            total_q = len(EVAL_QUERIES)
            recall_at_1 = (hits_at_1 / total_q) * 100.0
            recall_at_5 = (hits_at_5 / total_q) * 100.0
            avg_latency = sum(latencies_ms) / len(latencies_ms)
            p95_latency = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]

            print("\nSUMMARY METRICS:")
            print(f"  Total Queries Evaluated: {total_q}")
            print(f"  Recall@1:                {recall_at_1:.1f}% ({hits_at_1}/{total_q})")
            print(f"  Recall@5:                {recall_at_5:.1f}% ({hits_at_5}/{total_q})")
            print(f"  Average Query Latency:   {avg_latency:.2f} ms (Embedding + pgvector SQL)")
            print(f"  P95 Query Latency:       {p95_latency:.2f} ms")
            print("=" * 80)
            print(
                "Note: This measurement establishes an initial retrieval baseline on a controlled"
            )
            print(
                "dataset, not exhaustive proof of domain-specific production retrieval quality.\n"
            )

        finally:
            await cleanup_eval_dataset(session, doc_ids)


if __name__ == "__main__":
    try:
        asyncio.run(run_evaluation())
    except KeyboardInterrupt:
        sys.exit(130)
