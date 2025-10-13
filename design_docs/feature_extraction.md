# Feature Extraction Service

## Overview

The **Feature Extraction Service** extracts comprehensive visual features from images using multiple computer vision algorithms. This enables finding images with similar visual characteristics like edges, textures, colors, and shapes - perfect for content-based image retrieval and similarity analysis.

## Supported Features

### Edge Detection
- **Canny Edge Detection**: Detects object boundaries and structural elements
- **Sobel Gradients**: Directional edge information

### Corner & Keypoint Detection
- **Harris Corners**: Detects corner points and interest regions
- **ORB (Oriented FAST and Rotated BRIEF)**: Scale and rotation invariant keypoints

### Texture Analysis
- **LBP (Local Binary Patterns)**: Analyzes local texture patterns
- **GLCM (Gray-Level Co-occurrence Matrix)**: Statistical texture properties
- **Haralick Features**: Comprehensive texture characterization

### Shape & Structure
- **HOG (Histogram of Oriented Gradients)**: Shape and gradient information
- **Contour Analysis**: Object shape descriptors

### Color Features
- **Color Histograms**: Color distribution analysis
- **Color Moments**: Statistical color properties

## Installation

Install the required dependencies:

```bash
pip install scikit-image>=0.19.0 mahotas>=1.4.0
```

Or use the updated `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Configuration

### Basic Configuration

```yaml
# Input/Output
embeddings_path: "path/to/embeddings.npz"
output_path: "results/features"

# Feature Methods
feature_methods:
  - "canny"           # Edge detection
  - "harris"          # Corner detection
  - "lbp"             # Local texture patterns
  - "color_histogram" # Color features

# Processing
sample_percentage: 100
save_feature_images: true
```

### All Available Methods

```yaml
feature_methods:
  - "canny"           # Canny edge detection
  - "harris"          # Harris corner detection
  - "lbp"             # Local Binary Patterns
  - "color_histogram" # Color histogram features
  - "glcm"            # Gray-Level Co-occurrence Matrix
  - "orb"             # ORB keypoints and descriptors
  - "hog"             # Histogram of Oriented Gradients
  - "haralick"        # Haralick texture features
```

### Method-Specific Settings

#### Canny Edge Detection
```yaml
canny_low_threshold: 50    # Lower threshold
canny_high_threshold: 150  # Upper threshold
canny_aperture_size: 3     # Sobel kernel size
```

#### Harris Corner Detection
```yaml
harris_block_size: 2       # Neighborhood size
harris_ksize: 3           # Sobel kernel size
harris_k: 0.04            # Harris parameter
harris_threshold: 0.01    # Detection threshold
```

#### Local Binary Patterns (LBP)
```yaml
lbp_radius: 3             # Sample radius
lbp_n_points: 24          # Number of sample points
lbp_method: "uniform"     # Method: "uniform", "ror", "var"
```

#### Color Histograms
```yaml
hist_bins: 256            # Number of bins
hist_channels: [0, 1, 2]  # RGB channels
```

## Usage

### Command Line

```bash
python services/feature_extraction_service.py feature_config.yaml
```

### Python API

```python
from services.feature_extraction_service import FeatureExtractionService

service = FeatureExtractionService()
results_path = service.extract_features("feature_config.yaml")
print(f"Results saved to: {results_path}")
```

### Programmatic Configuration

```python
import yaml
from pathlib import Path

config = {
    'embeddings_path': 'embeddings.npz',
    'output_path': 'results',
    'feature_methods': ['canny', 'harris', 'lbp', 'color_histogram'],
    'sample_percentage': 100,
    'save_feature_images': True,
    'canny_low_threshold': 50,
    'canny_high_threshold': 150
}

# Save and run
config_path = Path('feature_config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(config, f)

service = FeatureExtractionService()
results = service.extract_features(config_path)
```

## Output Format

### Feature Data Structure

Each image produces comprehensive feature data:

```json
{
  "image_path": "path/to/image.jpg",
  "image_shape": [480, 640, 3],
  "has_features": true,
  "features": {
    "canny": {
      "method": "canny",
      "edge_density": 0.045,
      "edge_magnitude": 0.023,
      "edge_direction_histogram": [0.12, 0.15, 0.08, ...],
      "parameters": {"low_threshold": 50, "high_threshold": 150}
    },
    "harris": {
      "method": "harris",
      "corner_count": 127,
      "corner_density": 0.00041,
      "corner_strength": 0.045,
      "corner_distribution": [15, 23, 18, ...],
      "corner_locations": [[45, 123], [78, 234], ...]
    },
    "lbp": {
      "method": "lbp",
      "lbp_histogram": [0.023, 0.045, 0.012, ...],
      "lbp_variance": 45.67,
      "lbp_uniformity": 0.234,
      "lbp_entropy": 4.567
    },
    "color_histogram": {
      "method": "color_histogram",
      "histograms": {
        "channel_0": [0.001, 0.002, ...],
        "channel_1": [0.003, 0.001, ...],
        "channel_2": [0.002, 0.004, ...]
      },
      "combined_histogram": [0.001, 0.002, 0.003, ...],
      "color_moments": {
        "channel_0": {"mean": 127.5, "std": 45.2}
      }
    }
  },
  "feature_vectors": {
    "canny": [0.045, 0.023, 0.12, 0.15, ...],
    "harris": [0.00041, 0.045, 15, 23, 18, ...],
    "lbp": [0.023, 0.045, 0.012, ...],
    "color_histogram": [0.001, 0.002, 0.003, ...]
  }
}
```

### Feature Vectors for Similarity

Each method produces a normalized feature vector for similarity comparison:

- **Canny**: `[edge_density, edge_magnitude, direction_histogram...]`
- **Harris**: `[corner_density, corner_strength, spatial_distribution...]`
- **LBP**: `[lbp_histogram...]` (normalized texture pattern distribution)
- **Color**: `[combined_rgb_histogram...]` (normalized color distribution)
- **GLCM**: `[contrast, correlation, energy, homogeneity...]`
- **ORB**: `[keypoint_density, avg_size, avg_response, size_std, response_std]`
- **HOG**: `[hog_feature_vector...]` (shape and gradient descriptors)
- **Haralick**: `[13_texture_features...]` (comprehensive texture metrics)

### Statistics and Metadata

```json
{
  "total_images_processed": 1000,
  "successful_feature_extractions": 950,
  "success_rate": 0.95,
  "feature_methods": ["canny", "harris", "lbp", "color_histogram"],
  "method_success_rates": {
    "canny": 0.98,
    "harris": 0.96,
    "lbp": 0.94,
    "color_histogram": 0.99
  },
  "method_success_counts": {
    "canny": 980,
    "harris": 960,
    "lbp": 940,
    "color_histogram": 990
  }
}
```

## Feature Use Cases

### Content-Based Image Retrieval

#### Find Similar Textures
```yaml
feature_methods: ["lbp", "glcm", "haralick"]
lbp_radius: 2
lbp_n_points: 16
```

#### Find Similar Shapes
```yaml
feature_methods: ["canny", "harris", "hog"]
canny_low_threshold: 30
canny_high_threshold: 100
```

#### Find Similar Colors
```yaml
feature_methods: ["color_histogram"]
hist_bins: 64  # Reduce bins for broader color matching
hist_channels: [0, 1, 2]
```

#### Comprehensive Similarity
```yaml
feature_methods: ["canny", "harris", "lbp", "color_histogram", "hog"]
# Combines edges, corners, texture, color, and shape
```

### Specific Applications

#### Architecture/Building Analysis
```yaml
feature_methods: ["canny", "harris", "hog"]
# Emphasizes structural elements and geometric features
```

#### Nature/Landscape Analysis
```yaml
feature_methods: ["lbp", "color_histogram", "glcm"]
# Focuses on texture patterns and natural color distributions
```

#### Art/Painting Analysis
```yaml
feature_methods: ["color_histogram", "lbp", "hog"]
# Captures color composition, texture, and artistic style
```

#### Object/Product Detection
```yaml
feature_methods: ["orb", "harris", "hog"]
# Scale-invariant features for object recognition
```

## Integration with Other Services

### Similarity Search Pipeline

1. **Feature Extraction** → Extract visual features
2. **Similarity Service** → Compare feature vectors
3. **Clustering Service** → Group similar features

```python
# 1. Extract features
feature_service = FeatureExtractionService()
feature_results = feature_service.extract_features("feature_config.yaml")

# 2. Use feature vectors for similarity search
# Load feature_vectors.npz from results
feature_data = np.load(f"{feature_results}/feature_vectors.npz")
canny_vectors = feature_data['canny_vectors']
color_vectors = feature_data['color_histogram_vectors']

# 3. Combine with pose extraction for comprehensive analysis
pose_results = pose_service.extract_poses("pose_config.yaml")
```

### Custom Similarity Metrics

```python
def combined_similarity(features1, features2, weights=None):
    """Combine multiple feature types with weights."""
    if weights is None:
        weights = {'canny': 0.3, 'harris': 0.2, 'lbp': 0.3, 'color': 0.2}
    
    similarity = 0.0
    for method, weight in weights.items():
        if method in features1 and method in features2:
            # Cosine similarity for each feature type
            sim = cosine_similarity([features1[method]], [features2[method]])[0][0]
            similarity += weight * sim
    
    return similarity
```

## Performance Optimization

### Processing Speed

```yaml
# Fast processing for large datasets
feature_methods: ["canny", "color_histogram"]  # Use fewer, faster methods
image_resize: [256, 256]  # Resize for speed
sample_percentage: 10     # Process subset for testing
```

### Memory Optimization

```yaml
# Reduce memory usage
hist_bins: 64           # Fewer histogram bins
orb_max_features: 100   # Fewer ORB keypoints
hog_pixels_per_cell: [16, 16]  # Larger HOG cells
```

### Quality vs Speed Trade-offs

| Speed | Methods | Use Case |
|-------|---------|----------|
| **Fastest** | `["canny", "color_histogram"]` | Quick similarity check |
| **Balanced** | `["canny", "harris", "lbp", "color_histogram"]` | General-purpose |
| **Comprehensive** | `["canny", "harris", "lbp", "color_histogram", "glcm", "orb", "hog", "haralick"]` | Maximum accuracy |

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install `scikit-image` and `mahotas`
2. **Memory Errors**: Reduce `image_resize` or `sample_percentage`
3. **Slow Processing**: Use fewer feature methods or smaller images
4. **Empty Results**: Check image paths and file formats

### Error Handling

The service gracefully handles errors:
- Missing libraries: Skips methods requiring unavailable libraries
- Corrupted images: Continues processing other images
- Invalid parameters: Uses sensible defaults

### Debug Configuration

```yaml
# Minimal config for testing
feature_methods: ["canny"]
sample_percentage: 1  # Process just 1% of images
save_feature_images: true
image_resize: [128, 128]
```

## Advanced Features

### Custom Feature Combinations

Create application-specific feature combinations:

```python
# Define custom feature profiles
TEXTURE_PROFILE = ["lbp", "glcm", "haralick"]
SHAPE_PROFILE = ["canny", "harris", "hog", "orb"]
COLOR_PROFILE = ["color_histogram"]
COMPREHENSIVE_PROFILE = TEXTURE_PROFILE + SHAPE_PROFILE + COLOR_PROFILE
```

### Feature Vector Dimensionality

| Method | Vector Size | Description |
|--------|-------------|-------------|
| Canny | 10 | 2 statistics + 8 direction bins |
| Harris | 18 | 2 statistics + 16 spatial distribution |
| LBP | 26 | 26-bin uniform histogram |
| Color | 768 | 256 bins × 3 channels |
| GLCM | 5 | Mean values of 5 properties |
| ORB | 5 | Keypoint statistics |
| HOG | Variable | Depends on image size and cell size |
| Haralick | 13 | 13 texture properties |

### Multi-Scale Analysis

```yaml
# Extract features at multiple scales
image_resize: [256, 256]  # Process at 256x256
# Also process at original resolution for detail
# Combine results for multi-scale analysis
```

## Future Enhancements

Planned improvements:
- **Deep learning features** using pre-trained CNNs
- **Temporal features** for video analysis
- **3D features** for depth/stereo images
- **Custom feature training** for specific domains
- **Real-time feature extraction** for streaming
- **Feature selection** and dimensionality reduction
- **Ensemble similarity** combining multiple feature types 