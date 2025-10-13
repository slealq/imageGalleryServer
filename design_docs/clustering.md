# 🧠 Clustering & Similarity System — Integration Design Spec

## 🧾 Summary

This design defines the architecture and interfaces for integrating image clustering and similarity search into an AI-assisted image dataset curation app. Users can browse, caption, crop, and tag images to prepare training datasets. Clustering and similarity search will allow users to more efficiently find and group visually related images.

---

## 🏗️ High-Level Components

```
[User selects image] ──▶ [API: Find similar images]
                             │
                             ▼
                     [Embedding Index]
                             │
                  +----------+----------+
                  |                     |
         [Vector Search (k-NN)]    [Clustering Module]
                  |                     |
                  ▼                     ▼
      Return ranked similar       Build/update clusters
         image IDs                with metadata & tags
```

---

## 🧩 Component Breakdown

### 1. `ImageEmbeddingService`

**Responsibility**: Compute and store CLIP-based image embeddings (GPU-accelerated).

```python
class ImageEmbeddingService:
    def update_embeddings(image_ids: List[str]) -> None
    def get_embedding(image_id: str) -> np.ndarray
    def get_all_embeddings() -> Dict[str, np.ndarray]
```

* Use CLIP ViT-L/14 or ViT-B/32.
* Store embeddings as `.npy` or in a database.
* Use batch processing with GPU for efficiency.

---

### 2. `ClusteringService`

**Responsibility**: Periodically cluster all image embeddings (offline job).

```python
class ClusteringService:
    def run_clustering(embedding_dict: Dict[str, np.ndarray]) -> Dict[str, int]
    def get_cluster_for_image(image_id: str) -> int
```

* Uses HDBSCAN or KMeans + UMAP/PCA.
* Stores cluster ID as tag or DB field per image.
* Can be triggered manually or as a scheduled job.

---

### 3. `SimilaritySearchService`

**Responsibility**: Quickly find visually similar images to a selected reference.

```python
class SimilaritySearchService:
    def find_similar(image_id: str, top_k: int = 50) -> List[str]
```

* Uses FAISS or sklearn NearestNeighbors with embeddings.
* GPU-accelerated and cached for performance.
* Useful for real-time search based on selected image.

---

### 4. `TaggingService`

**Responsibility**: Apply a custom tag to a group of images.

```python
class TaggingService:
    def tag_images(image_ids: List[str], tag: str) -> None
```

* Tags are stored in the DB and can be used for filtering.
* Used after clustering or similarity search.

---

### 5. `ClusterManagerAPI`

**Responsibility**: Provide REST API for:

* Running clustering
* Finding similar images
* Tagging results

```http
POST /api/cluster/run
POST /api/similarity/find
POST /api/tagging/apply
```

---

## 💡 User Flow: "Find Similar Images"

1. User clicks “Find Similar” on an image in the gallery.
2. Frontend calls `/api/similarity/find?image_id=xyz`.
3. Backend returns ranked similar image IDs.
4. UI previews the images and lets user click “Tag All”.
5. Frontend sends `/api/tagging/apply` to tag all with e.g. `style_001`.
6. User now filters with this tag and captions/crops only these.

---

## 📁 Suggested Folder Structure (Backend)

```
backend/
├── services/
│   ├── embedding_service.py
│   ├── clustering_service.py
│   ├── similarity_service.py
│   └── tagging_service.py
├── api/
│   ├── cluster_manager.py
│   └── similarity_api.py
├── jobs/
│   └── nightly_cluster_job.py
└── config/
    └── clustering.yaml
```

---

## 🧠 Optional Enhancements

* **EmbeddingQueueService**: Use background worker (e.g. Redis + RQ) to update embeddings when new images are uploaded.
* **Smart Tag Suggestions**: Automatically name clusters based on caption or actor metadata.
* **Active Learning**: Let users rate search quality to refine future clustering/similarity logic.

---

## ✅ Next Steps to Build

1. Implement `ImageEmbeddingService` (CLIP + storage).
2. Add `SimilaritySearchService` using FAISS.
3. Build `ClusteringService` with HDBSCAN.
4. Wire API to UI for clustering and similarity search.
5. Add tagging and filtering to finalize dataset curation loop.

---
