# 📄 Design Specification Document: Modular Image Similarity System

## 🗉 Overview

This document defines the architecture and responsibilities of a modular system designed to find visually similar images using embeddings, clustering, and similarity search. The system is designed for experimentation and extensibility, with components separated into distinct services.

---

## 🎯 System Goals

* Allow experimentation with different embedding models, clustering methods, and search techniques
* Persist all outputs (embeddings, cluster assignments, FAISS indexes) in a reproducible, versioned manner
* Support evaluation using grounding sets of known-similar images
* Separate responsibilities into modular services with well-defined inputs/outputs
* Enable partial dataset usage and forced inclusion of specific images in runs

---

## 📦 Top-Level Project Structure

```
project_root/
├── services/
│   ├── dataset_service/
│   ├── embedding_service/
│   ├── clustering_service/
│   └── similarity_service/
├── data/                      # Raw input datasets
├── results/                   # All run outputs (grouped by pipeline type and run ID)
├── controller/                # Optional CLI / pipeline runner
└── configs/                   # Config presets
```

---

## 🔧 Component: Dataset Creation Service

### 🔹 Purpose

Generate a deterministic subset of the dataset based on a given percentage and seed. Guarantees reproducibility and forced inclusion of key images.

### 📅 Inputs

* `config.yaml` file containing:

  * `dataset_path`: path to source image dataset
  * `output_path`: path where output list is stored
  * `fraction`: float between 0.0 and 1.0 (e.g., 0.25 for 25%)
  * `force_include`: list of image paths to include regardless of sampling
  * `random_seed`: integer seed for deterministic behavior

### 📄 Outputs (in `results/dataset/<run_id>/`)

* `used_images.json`: list of selected image paths
* `config.yaml`

### 🧠 Tasks

* List all images in dataset path
* Sample a fraction using random seed
* Ensure forced-inclusion list is respected
* Save result to `used_images.json`

### 🔄 Interface (Python API)

```python
def create_dataset_subset(dataset_path: str, output_path: str, fraction: float, force_include: Optional[List[str]] = None, seed: int = 42):
    pass
```

---

## 🔧 Component: Embedding Service

### 🔹 Purpose

Generate vector representations (embeddings) of input images using different model backbones.

### 📅 Inputs

* `config.yaml` file containing:

  * `dataset_path`: path to images
  * `model_name`: e.g., `openclip-vit-b32`
  * `batch_size`, `device`, etc.

### 📄 Outputs (in `results/embeddings/<run_id>/`)

* `embeddings.npz`: image IDs + vector representations
* `config.yaml`: parameters used

### 🧠 Tasks

* Load and optionally subsample images
* Ensure inclusion of `force_include` images
* Generate embeddings using selected model
* Store output as `.npz` with filenames

### 🔄 Interface (Python API)

```python
def generate_embeddings(dataset_path: str, model_name: str, output_path: str, dataset_fraction: float = 1.0, force_include: Optional[List[str]] = None):
    pass
```

---

## 🔧 Component: Clustering Service

### 🔹 Purpose

Group similar images using clustering algorithms like HDBSCAN or KMeans.

### 📅 Inputs

* `config.yaml` file containing:

  * `embedding_input`: path to `.npz` file from embeddings service
  * `used_images`: path to `used_images.json` from embeddings output
  * `method`: e.g., `hdbscan`
  * `grounding_set`: (optional) path to known similar image groups
  * `params`: algorithm-specific parameters

### 📄 Outputs (in `results/clusters/<run_id>/`)

* `clusters.json`: image ID → cluster label map
* `metrics.json`: silhouette score, grounding coverage, etc.
* `used_images.json`: passed-through or filtered image list
* `config.yaml`

### 🧠 Tasks

* Load embeddings
* Restrict to provided image subset
* Run selected clustering algorithm
* Evaluate with grounding set (optional)
* Save cluster labels and evaluation metrics

### 🔄 Interface (Python API)

```python
def run_clustering(embedding_path: str, method: str, output_path: str, used_images_path: str, grounding_set: Optional[str] = None):
    pass
```

---

## 🔧 Component: Similarity Service

### 🔹 Purpose

Index embeddings for fast similarity search (e.g., FAISS) and support querying by image.

### 📅 Inputs

* `config.yaml` file containing:

  * `embedding_input`: path to `.npz` file
  * `used_images`: path to `used_images.json`
  * `index_type`: e.g., `faiss-flat`, `faiss-hnsw`
  * `query_image`: path to query image (optional)
  * `top_k`: number of similar items to retrieve

### 📄 Outputs (in `results/similarity/<run_id>/`)

* `faiss.index`: stored index
* `neighbors.json`: top-k results for query
* `used_images.json`: image list included in the index
* `config.yaml`

### 🧠 Tasks

* Build and store index on selected image subset
* Run search for provided query (if any)
* Return list of top-K similar images with distances

### 🔄 Interface (Python API)

```python
def build_index(embedding_path: str, output_path: str, index_type: str, used_images_path: str):
    pass

def search_similar(query_image: str, index_path: str, top_k: int) -> List[str]:
    pass
```

---

## ⚙️ Controller (Optional)

### 🔹 Purpose

Provide CLI or centralized runner to automate chained steps: embed → cluster → search.

### 🧠 Tasks

* Load pipeline config (YAML)
* Spawn subprocesses or service calls with correct I/O paths
* Automatically resolve run ID dependencies between services

---

## 📁 Run Output Organization

Each pipeline stores results using a numeric or timestamped run ID. Each run directory contains:

```
results/<pipeline_type>/<run_id>/
├── config.yaml
├── output_artifact (e.g., embeddings.npz, clusters.json)
├── used_images.json
├── metrics.json (if applicable)
```

---

## ✅ Deliverables Checklist (per service)

| Deliverable          | Status        | Notes                               |
| -------------------- | ------------- | ----------------------------------- |
| `run.py` script      | \[ ]          | CLI entry for each service          |
| `config.yaml` sample | \[ ]          | Documented config file              |
| Core logic module    | \[ ]          | `embedder.py`, etc.                 |
| Results saving logic | \[ ]          | Use `run_id`, save config/artifacts |
| CLI or controller    | \[ ] Optional | Optional but recommended            |

---

## 🚀 Next Steps

1. Scaffold each service directory
2. Implement `run.py` and logic modules
3. Define config schema per service
4. Add logging + evaluation
5. (Optional) Build controller script for chaining

---

This design enables experimentation, repeatability, and flexibility across all components of a modern visual similarity system. Developers can work independently on each service and validate their outputs using saved artifacts and config files.
