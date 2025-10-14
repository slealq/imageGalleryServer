# AI Image Generation & Training Service Architecture

## Overview

This service enables browsing, annotating, and training AI image generation datasets. It includes semantic image search, metadata filtering, tagging, and batch image retrieval, optimized for high performance (target: <200ms per batch fetch).

---

## 1. High-Level Architecture

```
                 ┌────────────────────────┐
                 │  Web / React Frontend  │
                 │ (Image browser, tags,  │
                 │  captions, filters)    │
                 └──────────┬─────────────┘
                            │
                            ▼
            ┌───────────────────────────────────┐
            │           API Gateway             │
            │  (FastAPI / gRPC / GraphQL)       │
            └──────────┬────────────────────────┘
                       │
     ┌─────────────────┴──────────────────────────────────────────┐
     │                            Backend                        │
     │  ┌───────────────────────────┬───────────────────────────┐  │
     │  │  Metadata Service         │  Embeddings Service       │  │
     │  │ (CRUD, tags, crops, etc.) │ (CLIP/LLaVA embeddings)   │  │
     │  └───────────┬───────────────┴──────────────┬────────────┘  │
     │              │                                │              │
     │  ┌───────────▼─────────────┐    ┌────────────▼──────────┐   │
     │  │  PostgreSQL (metadata)  │    │  Vector DB (search)   │   │
     │  │ + PostGIS (bbox, crops) │    │  (Qdrant / Milvus)    │   │
     │  └───────────┬─────────────┘    └────────────┬──────────┘   │
     │              │                                │              │
     │       ┌──────▼──────┐                 ┌────────▼────────┐    │
     │       │ Object Store│                 │ Cache (Redis)   │    │
     │       │ (S3/GCS)    │                 │ for hot batches │    │
     │       └─────────────┘                 └─────────────────┘    │
     └──────────────────────────────────────────────────────────────┘
```

---

## 2. Database Layer

### a. Metadata DB (PostgreSQL)

A robust metadata store with JSONB for flexible tagging and indexing.

```sql
CREATE TABLE images (
    id UUID PRIMARY KEY,
    url TEXT,
    file_path TEXT,
    caption TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    actors TEXT[],
    tags TEXT[],
    width INT,
    height INT,
    crop_boxes JSONB, -- [{"x":0.1,"y":0.1,"w":0.4,"h":0.4}]
    metadata JSONB,   -- {"source":"flickr", "downloaded_by":"script1"}
    embedding_id UUID  -- link to vector DB
);
CREATE INDEX idx_tags ON images USING GIN (tags);
CREATE INDEX idx_metadata ON images USING GIN (metadata);
```

**Optional:** Add **PostGIS** for bounding-box and region queries.

---

### b. Vector Database

Use **Qdrant**, **Milvus**, or **Weaviate** for semantic search.
Each image/crop stores an embedding and associated metadata.

**Example (Qdrant schema):**

```json
{
  "id": "uuid",
  "vector": [0.23, -0.15, ...],
  "payload": {
    "image_id": "uuid",
    "tags": ["car", "red"],
    "actors": ["person"],
    "url": "https://..."
  }
}
```

Supports hybrid search (vector + metadata):

```python
search(query_vector, filter={"tags": {"$in": ["car"]}})
```

---

### c. Image Storage

Use **AWS S3**, **Google Cloud Storage**, or **MinIO**.

* Predictable pathing: `/images/{uuid}.jpg`
* Separate bucket/folder for thumbnails for fast browsing.

---

## 3. Performance & Caching

### a. Fast Batch Fetching

For <200ms response times:

* Redis for metadata caching (`image_id → metadata blob`)
* CDN caching for thumbnails (CloudFront or Cloudflare)
* Async I/O with FastAPI or Go Fiber

**Flow:**

1. Request N image IDs.
2. Check Redis cache (80–90% hit rate).
3. Pull missing data from Postgres.
4. Serve combined results.

### b. Precomputation Pipelines

Background workers generate and enrich data:

* Embeddings (CLIP, OpenCLIP, SigLIP)
* Captions (BLIP, LLaVA)
* Tags and crops

Tools: **Celery + RabbitMQ / Kafka**

---

## 4. Semantic Search

Use text-to-image embedding models such as **OpenCLIP** or **SigLIP**.

Flow:

1. Text query → embedding vector
2. ANN (Approximate Nearest Neighbor) search in Qdrant
3. Filter via metadata (`tags`, `actors`, `source`)
4. Retrieve metadata from Redis/Postgres
5. Return ranked results

---

## 5. Stack Summary

| Layer           | Tech Choice                 | Notes                     |
| --------------- | --------------------------- | ------------------------- |
| **Frontend**    | React + Next.js + Tailwind  | Interactive image browser |
| **Backend API** | FastAPI (Python) / Go Fiber | Async, high-performance   |
| **Metadata DB** | PostgreSQL + SQLAlchemy     | JSONB & indexing          |
| **Vector DB**   | Qdrant                      | Semantic & tag filtering  |
| **Storage**     | MinIO (dev) / S3 (prod)     | Object storage            |
| **Cache**       | Redis                       | Hot metadata + thumbnails |
| **Workers**     | Celery + RabbitMQ/Kafka     | Asynchronous tasks        |
| **Auth**        | JWT / OAuth2                | Token-based access        |
| **Metrics**     | Prometheus + Grafana        | Performance monitoring    |

---

## 6. Query Flow Example

1. User query: “red car with two people”
2. Backend:

   * Converts text → CLIP embedding
   * Searches Qdrant with filter `{"tags": ["car"]}`
   * Retrieves top 50 matches
   * Metadata fetched from Redis/Postgres
3. UI displays results grid in <200ms (if cached)

---

## 7. Scaling Strategy

| Scale                       | Recommended Setup                                                   |
| --------------------------- | ------------------------------------------------------------------- |
| **Prototype (≤10k images)** | FastAPI + SQLite + FAISS + MinIO                                    |
| **Mid-Scale (10k–100k)**    | PostgreSQL + Qdrant + Redis                                         |
| **Large (≥1M)**             | Distributed Qdrant + Partitioned Postgres (Citus) + Kafka ingestion |

---

## 8. Future Enhancements

* Add image versioning (track crops and edits)
* Integrate labeling UI for dataset QA
* Include CLIP score filtering for automatic tag suggestions
* Provide REST & gRPC endpoints for AI training pipelines

---

**Goal:** Build a scalable, modular backend capable of powering an image dataset browser, annotation system, and training pipeline with near-real-time semantic search and high-throughput batch serving.
