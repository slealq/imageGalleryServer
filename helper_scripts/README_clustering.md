# Image Clustering with CLIP and HDBSCAN

This script clusters images using CLIP embeddings and HDBSCAN clustering algorithm. It's designed to automatically group similar images together based on their visual content.

## Features

- **CLIP Embeddings**: Uses OpenAI's CLIP model to extract high-quality image embeddings
- **HDBSCAN Clustering**: Robust clustering algorithm that can handle noise and varying cluster densities
- **Visualization**: Creates t-SNE plots to visualize cluster relationships
- **Sample Export**: Saves representative images from each cluster
- **Flexible Parameters**: Adjustable clustering parameters for different datasets

## Installation

First, install the required dependencies:

```bash
cd image_server
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The main script can be run from the command line:

```bash
# Basic usage
python helper_scripts/categorization.py /path/to/images

# With custom parameters
python helper_scripts/categorization.py /path/to/images \
    --output-dir my_results \
    --min-cluster-size 5 \
    --min-samples 3 \
    --samples-per-cluster 5 \
    --model ViT-B-32
```

### Python API

You can also use the script programmatically:

```python
from helper_scripts.categorization import ImageClusterer

# Initialize clusterer
clusterer = ImageClusterer(model_name="ViT-B-32", pretrained="openai")

# Run clustering
clusterer.run_clustering(
    input_folder="/path/to/images",
    output_dir="clustering_results",
    min_cluster_size=5,
    min_samples=3,
    samples_per_cluster=5
)
```

### Example Script

Use the provided example script to cluster images from the project's images folder:

```bash
python helper_scripts/run_clustering_example.py
```

## Parameters

### Clustering Parameters

- `min_cluster_size` (default: 5): Minimum number of points required to form a cluster
- `min_samples` (default: 3): Minimum number of samples in a neighborhood for a point to be considered a core point

### Model Parameters

- `model` (default: "ViT-B-32"): CLIP model variant to use
  - `ViT-B-32`: Fast, good quality (default)
  - `ViT-L-14`: Larger, higher quality
  - `ViT-H-14`: Largest, highest quality
- `pretrained` (default: "openai"): Pretrained weights to use

### Output Parameters

- `samples_per_cluster` (default: 5): Number of sample images to save from each cluster
- `output_dir` (default: "clustering_results"): Directory to save results

## Output

The script creates the following outputs in the specified output directory:

```
clustering_results/
├── embeddings.npy              # Raw CLIP embeddings
├── cluster_labels.npy          # Cluster assignments
├── cluster_visualization.png   # t-SNE visualization
├── cluster_info.json           # Cluster metadata
└── samples/                    # Sample images from each cluster
    ├── cluster_0/
    │   ├── sample_1_image1.jpg
    │   ├── sample_2_image2.jpg
    │   └── ...
    ├── cluster_1/
    │   └── ...
    └── noise/                  # Images that didn't fit any cluster
        └── ...
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Tips for Best Results

1. **Dataset Size**: Works best with 50+ images for meaningful clustering
2. **Image Quality**: Higher quality images produce better embeddings
3. **Parameter Tuning**: 
   - For smaller datasets, reduce `min_cluster_size` and `min_samples`
   - For larger datasets, increase these values
4. **Model Selection**: 
   - Use `ViT-B-32` for speed
   - Use `ViT-L-14` or `ViT-H-14` for better quality (slower)

## Troubleshooting

### Common Issues

1. **No clusters found**: Reduce `min_cluster_size` and `min_samples`
2. **Too many noise points**: Increase `min_cluster_size` and `min_samples`
3. **Memory issues**: Use a smaller CLIP model or process images in smaller batches
4. **CUDA out of memory**: The script automatically falls back to CPU if GPU memory is insufficient

### Performance Notes

- First run will download the CLIP model (~500MB)
- GPU acceleration is used if available
- Processing time scales with number of images and model size

## Example Results

The script will output a summary like:

```
Clustering Summary:
Total images: 25
Clusters found: 4
Noise points: 3
Results saved to: clustering_results
```

This indicates that 25 images were processed, 4 distinct clusters were found, and 3 images didn't fit well into any cluster. 