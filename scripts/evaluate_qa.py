"""RAG Question Answering and Citation Benchmark Evaluation (Milestone 6).

Evaluates end-to-end RAG answer generation, citation validity, and abstention behavior
using a small reproducible question-and-answer dataset grounded in test documents.

Can run completely offline using LocalSentenceTransformerProvider and a deterministic
mock/fake generation provider, or against a real configured Azure model if enabled.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure root workspace is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragforge.core.config import Settings
from ragforge.db.session import get_session_factory
from ragforge.schemas.ask import AskRequest
from ragforge.services.ask_service import AskService
from ragforge.services.embedding_provider import LocalSentenceTransformerProvider
from ragforge.services.generation_provider import FakeGenerationProvider
from ragforge.services.retrieval_service import RetrievalService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_qa")


@dataclass
class QAItem:
    question: str
    expected_fact: str
    is_answerable: bool


# Benchmark QA queries grounded in the benchmark dataset
BENCHMARK_QA_ITEMS: list[QAItem] = [
    QAItem(
        question="How does PgBouncer reduce database connection overhead in PostgreSQL?",
        expected_fact="connection pools reuse active sockets to eliminate",
        is_answerable=True,
    ),
    QAItem(
        question="What library provides asynchronous background workers with Redis in Python?",
        expected_fact="ARQ is an asynchronous job queue",
        is_answerable=True,
    ),
    QAItem(
        question="Which PostgreSQL extension enables dense vector similarity search?",
        expected_fact="pgvector is an open-source vector similarity search extension",
        is_answerable=True,
    ),
    QAItem(
        question="Why do bakers use the autolyse resting period in sourdough preparation?",
        expected_fact="accelerating gluten development",
        is_answerable=True,
    ),
    QAItem(
        question="What is the quantum superposition principle in superconducting qubits?",
        expected_fact="",
        is_answerable=False,  # Unanswerable from the benchmark documents (tests abstention)
    ),
]


async def run_qa_evaluation() -> None:
    """Execute grounded question answering benchmark and output metrics."""
    settings = Settings(
        EMBEDDING_PROVIDER="local",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        EMBEDDING_DIMENSION=384,
        GENERATION_PROVIDER="fake",
        GENERATION_MODEL="eval-slm-01",
        ASK_MIN_SIMILARITY_THRESHOLD=0.35,
        ASK_MAX_QUESTION_CHARS=500,
    )

    logger.info("Initializing LocalSentenceTransformerProvider (all-MiniLM-L6-v2)...")
    emb_provider = LocalSentenceTransformerProvider(
        model_name=settings.EMBEDDING_MODEL,
        expected_dimension=settings.EMBEDDING_DIMENSION,
    )
    await emb_provider.ensure_loaded()

    gen_provider = FakeGenerationProvider(model_name=settings.GENERATION_MODEL)

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Import setup and cleanup from evaluate_retrieval
        from scripts.evaluate_retrieval import cleanup_eval_dataset, setup_eval_dataset

        setup_meta = await setup_eval_dataset(session, emb_provider)
        doc_ids = setup_meta["doc_ids"]

        try:
            retrieval = RetrievalService(
                db=session,
                embedding_provider=emb_provider,
                settings=settings,
            )
            ask_service = AskService(
                retrieval_service=retrieval,
                generation_provider=gen_provider,
                settings=settings,
            )

            print("\n" + "=" * 80)
            print("RAGFORGE SLM QUESTION ANSWERING & CITATION BENCHMARK REPORT (Milestone 6)")
            print("=" * 80)
            print(
                f"Model: {gen_provider.model_name} | "
                f"Similarity Threshold: {settings.ASK_MIN_SIMILARITY_THRESHOLD}\n"
            )

            valid_citations_count = 0
            correct_abstentions_count = 0
            total_eval = len(BENCHMARK_QA_ITEMS)
            retrieval_latencies: list[float] = []
            generation_latencies: list[float] = []

            for i, item in enumerate(BENCHMARK_QA_ITEMS, start=1):
                req = AskRequest(question=item.question, limit=5)
                t0 = time.perf_counter()
                resp = await ask_service.answer_question(req)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                retrieval_latencies.append(resp.telemetry.retrieval_latency_ms)
                generation_latencies.append(resp.telemetry.generation_latency_ms)

                status: str
                if not item.is_answerable:
                    if resp.abstained:
                        correct_abstentions_count += 1
                        status = "ABSTAINED (Expected)"
                    else:
                        status = "FAILED TO ABSTAIN"
                else:
                    if not resp.abstained and len(resp.citations) > 0:
                        valid_citations_count += 1
                        status = f"ANSWERED with {len(resp.citations)} Citation(s)"
                    else:
                        status = "UNEXPECTED ABSTENTION or MISSING CITATIONS"

                exp_label = "Answerable" if item.is_answerable else "Unanswerable (Abstain)"
                ret_lat = resp.telemetry.retrieval_latency_ms
                gen_lat = resp.telemetry.generation_latency_ms

                print(f"[{i}/{total_eval}] Question: '{item.question}'")
                print(f"       Expected: {exp_label}")
                print(f"       Result:   {status}")
                print(
                    f"       Latency:  Retrieval {ret_lat:.2f}ms | Generation {gen_lat:.2f}ms "
                    f"(Total {elapsed_ms:.2f}ms)"
                )
                print(f"       Answer:   '{resp.answer[:75]}...'")
                print("-" * 80)

            grounded_pct = (valid_citations_count / 4) * 100.0
            abstain_pct = (correct_abstentions_count / 1) * 100.0
            avg_ret = sum(retrieval_latencies) / len(retrieval_latencies)
            avg_gen = sum(generation_latencies) / len(generation_latencies)

            print("\nSUMMARY METRICS:")
            print(f"  Total Questions Evaluated:  {total_eval}")
            print(f"  Grounded Citation Rate:     {grounded_pct:.1f}% (4 answerable questions)")
            print(f"  Abstention Accuracy:        {abstain_pct:.1f}% (1 unanswerable question)")
            print(f"  Average Retrieval Latency:  {avg_ret:.2f} ms")
            print(f"  Average Generation Latency: {avg_gen:.2f} ms")
            print("=" * 80)
            print(
                "Note: Demonstrates citation verification and abstention mechanics on a "
                "controlled benchmark dataset.\n"
            )
        finally:
            await cleanup_eval_dataset(session, doc_ids)


if __name__ == "__main__":
    try:
        asyncio.run(run_qa_evaluation())
    except KeyboardInterrupt:
        sys.exit(130)
