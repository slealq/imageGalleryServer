# Ultralytics YOLO Pose Extraction

## Overview

We've added support for **Ultralytics YOLO v11** as a new pose extraction method alongside MediaPipe and Detectron2. This implementation provides state-of-the-art pose estimation with additional object detection capabilities, making it ideal for comprehensive scene analysis.

## Features

### Core Capabilities
- **Multi-person pose estimation** using YOLO v11 pose models
- **Additional object detection** beyond people (e.g., sports equipment, vehicles, animals)
- **GPU acceleration** with automatic device detection
- **Flexible model selection** from nano to extra-large models
- **High-performance inference** with optimized YOLO architecture

### Advanced Features
- **17-keypoint COCO format** pose detection
- **Confidence-based filtering** for reliable results
- **Bounding box information** for all detections
- **Class-specific detection** with configurable object classes
- **Batch processing** with progress tracking
- **Comprehensive statistics** including additional object counts

## Installation

Install the required dependency:

```bash
pip install ultralytics>=8.0.0
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
output_path: "results/ultralytics_pose"

# Method Selection
pose_method: "ultralytics"

# Core Settings
confidence_threshold: 0.5
person_detection_confidence: 0.6

# Ultralytics Settings
ultralytics_model: "yolo11n-pose.pt"
ultralytics_device: "auto"

# Features
enable_object_detection: true
object_classes: ["person"]
detect_multiple_people: true
save_pose_images: true
```

### Model Options

Choose from different YOLO v11 pose models based on your speed/accuracy requirements:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolo11n-pose.pt` | Nano | Fastest | Good | Real-time applications |
| `yolo11s-pose.pt` | Small | Fast | Better | Balanced performance |
| `yolo11m-pose.pt` | Medium | Moderate | High | Production use |
| `yolo11l-pose.pt` | Large | Slower | Higher | High accuracy needs |
| `yolo11x-pose.pt` | Extra Large | Slowest | Highest | Maximum accuracy |

### Device Configuration

```yaml
ultralytics_device: "auto"  # Automatic GPU/CPU selection
# ultralytics_device: "cpu"    # Force CPU
# ultralytics_device: "cuda"   # Force GPU
# ultralytics_device: 0        # Specific GPU device
```

### Object Detection Classes

Enable detection of specific object classes:

```yaml
object_classes: ["person"]  # People only
# object_classes: ["person", "bicycle", "car"]  # Multiple classes
# object_classes: ["all"]  # All COCO classes
```

Available COCO classes include:
- **People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
- **Furniture**: chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Usage

### Command Line

```bash
python services/pose_extraction_service.py ultralytics_config.yaml
```

### Python API

```python
from services.pose_extraction_service import PoseExtractionService

service = PoseExtractionService()
results_path = service.extract_poses("ultralytics_config.yaml")
print(f"Results saved to: {results_path}")
```

### Programmatic Configuration

```python
import yaml
from pathlib import Path

config = {
    'embeddings_path': 'embeddings.npz',
    'output_path': 'results',
    'pose_method': 'ultralytics',
    'ultralytics_model': 'yolo11s-pose.pt',
    'enable_object_detection': True,
    'object_classes': ['person', 'bicycle', 'car'],
    'confidence_threshold': 0.5,
    'person_detection_confidence': 0.6
}

# Save config
config_path = Path('ultralytics_config.yaml')
with open(config_path, 'w') as f:
    yaml.dump(config, f)

# Run extraction
service = PoseExtractionService()
results = service.extract_poses(config_path)
```

## Output Format

### Enhanced Pose Data

The Ultralytics method extends the standard pose data format:

```json
{
  "image_path": "path/to/image.jpg",
  "pose_method": "ultralytics",
  "has_pose": true,
  "person_count": 2,
  "people": [
    {
      "person_id": 0,
      "landmarks": [x1, y1, x2, y2, ...],  // 34 values (17 keypoints)
      "pose_confidence": 0.85,
      "detection_score": 0.92,
      "visibility_scores": [0.9, 0.8, ...],  // 17 values
      "bounding_box": {"x1": 100, "y1": 150, "x2": 300, "y2": 500},
      "keypoint_format": "coco_17",
      "class_name": "person"
    }
  ],
  "additional_detections": [
    {
      "class_name": "bicycle",
      "class_id": 1,
      "confidence": 0.75,
      "bounding_box": {"x1": 50, "y1": 200, "x2": 200, "y2": 400}
    }
  ],
  "total_detections": 1
}
```

### Statistics

Enhanced statistics include object detection metrics:

```json
{
  "pose_extraction_method": "ultralytics",
  "ultralytics_model": "yolo11s-pose.pt",
  "ultralytics_device": "cuda:0",
  "enable_object_detection": true,
  "object_classes": ["person", "bicycle"],
  "total_people_detected": 45,
  "total_additional_detections": 12,
  "images_with_additional_detections": 8,
  "average_additional_detections_per_image": 1.5
}
```

## Performance Comparison

| Method | Speed | Multi-Person | Additional Objects | 3D Landmarks | GPU Support |
|--------|-------|--------------|-------------------|--------------|-------------|
| MediaPipe | Fast | Limited | No | Yes | Yes |
| Detectron2 | Moderate | Excellent | No | No | Yes |
| **Ultralytics** | **Fast** | **Excellent** | **Yes** | **No** | **Yes** |

## Use Cases

### Sports Analysis
```yaml
pose_method: "ultralytics"
ultralytics_model: "yolo11m-pose.pt"  # Balance of speed and accuracy
enable_object_detection: true
object_classes: ["person", "sports ball", "tennis racket", "baseball bat"]
```

### Traffic Scene Analysis
```yaml
pose_method: "ultralytics"
enable_object_detection: true
object_classes: ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
```

### Comprehensive Scene Understanding
```yaml
pose_method: "ultralytics"
enable_object_detection: true
object_classes: ["all"]  # Detect all COCO classes
```

## Advanced Features

### Feet Detection
While YOLO doesn't have dedicated foot keypoints, you can detect shoes/feet through object detection:

```yaml
object_classes: ["person", "shoe"]  # If custom model includes shoes
# Or use ankle keypoints from pose data (keypoints 15-16 in COCO format)
```

### Custom Models
You can use custom-trained YOLO models:

```yaml
ultralytics_model: "path/to/custom_model.pt"
```

### Batch Processing
The service automatically handles batch processing with progress tracking:

```
🎯 Processing 1000 images with Ultralytics
Extracting poses: 100%|██████████| 1000/1000 [02:15<00:00, 7.4img/s]
Success: 950/1000, Rate: 95.0%, People: 1200, Avg: 1.3/img, Objects: 450
```

## Troubleshooting

### Common Issues

1. **Model Download**: First run downloads the model automatically
2. **GPU Memory**: Use smaller models (nano/small) for limited GPU memory
3. **CUDA Errors**: Ensure PyTorch CUDA version matches your CUDA installation
4. **Class Detection**: Verify class names match COCO dataset format

### Performance Optimization

1. **Use appropriate model size** for your hardware
2. **Enable GPU acceleration** with `use_gpu: true`
3. **Filter object classes** to reduce processing overhead
4. **Adjust confidence thresholds** to balance accuracy vs. speed

### Memory Management

For large datasets:
```yaml
sample_percentage: 10  # Process 10% of images for testing
confidence_threshold: 0.7  # Higher threshold = fewer detections
```

## Integration with Other Services

The Ultralytics pose extraction integrates seamlessly with other services:

- **Clustering Service**: Use pose landmarks for similarity clustering
- **Embedding Service**: Generate embeddings from detected objects
- **Similarity Service**: Compare poses and detected objects

## Future Enhancements

Planned improvements:
- **Custom object classes** training support
- **Temporal pose tracking** across video frames
- **3D pose estimation** using depth information
- **Real-time streaming** support
- **Advanced filtering** based on pose quality metrics 