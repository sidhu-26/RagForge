# Milestone 4 Learning Guide: Asynchronous Parsing, Chunking & Worker Pipeline

Welcome to the learning guide for **Milestone 4**. In this milestone, we evolved RAGForge from a document storage service into an asynchronous text processing pipeline. This guide explains key architectural concepts in clear, beginner-friendly terms.

---

## 1. FastAPI Request Handling vs. Background Workers

When a client uploads a document (such as a 50-page PDF), the server needs to:
1. Validate the file format.
2. Stream bytes to storage.
3. Save document metadata in the database.
4. Extract text from every page.
5. Split text into semantic chunks.
6. Save hundreds of chunk rows in the database.

If all six steps are performed inside the HTTP request:
- The user waits several seconds or minutes for a response.
- The web server's event loop and worker pool become congested, unable to process requests from other users.
- If the browser client disconnects or times out midway, the process may leave partially parsed data behind.

**The Solution:** Decouple ingestion from processing.
- The **FastAPI API server** handles steps 1–3 in milliseconds, enqueues a lightweight job payload (just the `document_id`), and returns HTTP 201 (`queued`).
- A dedicated **background worker process** picks up the job and executes steps 4–6 asynchronously in the background.

---

## 2. What Redis and ARQ Do

- **Redis**: An in-memory, ultra-fast data store. In RAGForge, Redis functions as a durable task message broker between the FastAPI API server and the background workers.
- **ARQ**: An asynchronous Python task queue library built specifically for `asyncio` and `Redis`. ARQ handles job scheduling, concurrency limits, retry backoff, and worker lifecycles with minimal overhead.

When a document is uploaded, the API calls:
```python
await pool.enqueue_job("process_document_job", str(document_id))
```
The ARQ worker constantly listens to Redis, pops incoming jobs, and dispatches them to worker functions.

---

## 3. Why Async SQLAlchemy Sessions Should Be Scoped to Individual Jobs

In web applications, database sessions are typically scoped to a single HTTP request lifecycle (`Depends(get_db)`). When the HTTP request finishes, the session commits or rolls back and closes.

Background workers work similarly:
- **Do not share sessions across jobs!** If Job A and Job B share a database session, uncommitted changes from Job A could be committed or rolled back unexpectedly by Job B.
- In `ragforge.worker`, each job creates a fresh session using an isolated async context manager:
  ```python
  async with session_factory() as session:
      await process_document(document_id_str, session, storage, settings)
  ```
- This guarantees clean connection checkout, isolated transaction boundaries, and automatic connection return to the pool when the job finishes.

---

## 4. Text Extraction vs. Chunking

These are two completely distinct responsibilities:

1. **Extraction (`DocumentExtractor`)**:
   - Converts raw binary files (PDF stream, DOCX zip, UTF-8 text) into structured strings.
   - Normalizes whitespace, strips non-printable control characters, and handles format-specific errors (corrupt zip, unreadable PDF stream).
   - Preserves page boundaries (e.g. PDF Page 1, Page 2).
   - Identifies empty or image-only documents that lack extractable text (which would require OCR).

2. **Chunking (`TextChunker`)**:
   - Operates strictly on extracted text—it has no knowledge of PDFs, Word files, or filesystem bytes.
   - Splits text into discrete semantic segments (~1,000 characters) following natural linguistic boundaries (paragraphs, sentences, words).
   - Enforces deterministic ordering and calculates character and estimated token counts.

---

## 5. Why Overlap Matters for RAG Retrieval

When text is split strictly at a hard character limit, critical semantic context can be sliced in half:

> *Example without overlap:*
> - **Chunk 1:** "...The financial penalty for non-compliance with Section 4 is"
> - **Chunk 2:** "$500,000 per violation or up to 2 years imprisonment..."

A vector search query asking *"What is the penalty for Section 4?"* might match Chunk 1 or Chunk 2, but neither chunk contains the full context.

**Overlap** (e.g., 150 characters) carries trailing text from Chunk 1 into the beginning of Chunk 2. This ensures that ideas bridging chunk boundaries are fully captured in at least one chunk embedding during semantic search.

---

## 6. What Idempotency Means

In distributed computing, an operation is **idempotent** if running it multiple times produces the exact same result as running it once.

In RAGForge's background worker:
- What happens if a job is retried by ARQ after a transient network blip?
- What happens if a user calls `POST /api/v1/documents/{id}/requeue` multiple times?
- **Idempotent behavior:**
  1. The worker removes any previous chunks for that document before persisting new chunks:
     ```python
     await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))
     ```
  2. If the document is already in the `PROCESSED` state, the worker exits cleanly without duplicating chunks or re-running extraction.
  3. Duplicate chunks are never created, preventing polluted search indices.

---

## 7. Why Database Transactions Matter

Database transactions follow **ACID** properties (Atomicity, Consistency, Isolation, Durability).

During document processing:
- We must insert 50 chunks AND update document status from `processing` to `processed`.
- If the database connection drops after chunk #25:
  - Without a transaction, 25 chunks remain in the database, and the document is left half-processed or corrupt.
  - With a single atomic transaction:
    ```python
    try:
        # insert all chunks
        # update document status
        await db.commit()
    except Exception:
        await db.rollback()  # undoes all chunk inserts cleanly
    ```
- Either the entire processing result commits together, or nothing commits and the document transitions safely to `failed`.

---

## 8. Worker State Transitions

Documents follow a strict state machine:

```
                  ┌──────────────┐
                  │   UPLOADED   │ (File stored on disk, DB record created)
                  └──────┬───────┘
                         │
             ┌───────────┴───────────┐
             │ Redis enqueue succeeds │
             ▼                       │ Redis down
      ┌──────────────┐               │ (recoverable via /requeue)
      │    QUEUED    │               │
      └──────┬───────┘               │
             │ Worker picks up job   │
             ▼                       ▼
      ┌──────────────┐        ┌──────────────┐
      │  PROCESSING  │◄───────┤   REQUEUED   │
      └──────┬───────┘        └──────────────┘
             │
     ┌───────┴───────┐
     │ Success       │ Extraction/Chunking error
     ▼               ▼
┌──────────────┐ ┌──────────────┐
│  PROCESSED   │ │    FAILED    │ (Stores error_code & safe error_message)
└──────────────┘ └──────────────┘
```

1. **`uploaded`**: Document payload saved in storage; metadata registered in PostgreSQL.
2. **`queued`**: Job successfully enqueued in Redis task queue.
3. **`processing`**: Background worker is actively parsing and chunking.
4. **`processed`**: All chunks and metadata atomically committed. Ready for future embedding and retrieval.
5. **`failed`**: Parsing or chunking failed (e.g. corrupt file, image-only PDF, text bounds exceeded). A machine-readable `error_code` and safe user-facing `error_message` are recorded for client troubleshooting.
