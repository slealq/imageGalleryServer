#!/usr/bin/env python3
"""
Streamlined Image Clustering and Similarity Search Pipeline

This script combines embedding generation, clustering, and similarity search into
a single automated workflow with configurable parameters for experimentation.

Features:
- CLIP embedding extraction with configurable models (ViT-L/14, ViT-H/14 preferred)
- Multiple clustering algorithms with parameter grid search
- Automated similarity search using top clustering results
- Comprehensive metrics and logging
- Configurable for experimentation

Usage:
    python streamlined_image_clustering.py --target-image "your_image.jpg"
"""

import os
import sys
import json
import time
import shutil
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse
import gc

# Clustering libraries
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import hdbscan

# Optional UMAP for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Optional FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
    FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
    if FAISS_GPU_AVAILABLE:
        print("✅ FAISS GPU acceleration available")
    else:
        print("📊 FAISS CPU available (install faiss-gpu for GPU acceleration)")
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    print("❌ FAISS not available (pip install faiss-cpu)")
except:
    FAISS_AVAILABLE = True
    FAISS_GPU_AVAILABLE = False
    print("📊 FAISS CPU available")

# Visualization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Optional GPU support
GPU_AVAILABLE = False
try:
    import cupy as cp
    import cuml
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("❌ GPU acceleration not available")


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    # Paths
    input_images_dir: str = "/mnt/d/TEST/images"
    output_base_dir: str = "/mnt/c/Users/stuar/Downloads/image_clustering_results"
    target_image: str = "JillKassidy_JakeAdams__February_1_2019_4800_1074_lhf_0157.jpg"
    
    # Pipeline control
    skip_embeddings: bool = False  # Skip embedding extraction and load existing ones
    experiment_mode: bool = False  # Run experiments with multiple configurations
    force_regenerate: bool = False  # Force regeneration even if compatible embeddings exist
    
    # Known similar images for validation (NEW)
    known_similar_pairs: List[Tuple[str, str]] = None  # List of (image1, image2) tuples
    validation_weight: float = 0.3  # Weight for validation score in composite score
    
    # Enhanced embedding parameters
    clip_model: str = "ViT-L-14"  # Options: ViT-B-32, ViT-B-16, ViT-L-14, ViT-H-14
    clip_pretrained: str = "openai"  # Options: openai, laion2b_s34b_b88k, laion400m_e32
    input_resolution: int = 224  # 224 for most models, 336 for ViT-H-14
    batch_size: int = 512  # Optimized for GPU: 512 or 1024
    timeout_seconds: int = 30
    
    # Experiment configurations (used when experiment_mode=True)
    experiment_resolutions: List[int] = None  # e.g., [224, 336, 384]
    experiment_models: List[Tuple[str, str]] = None  # e.g., [("ViT-B-32", "openai"), ("ViT-L-14", "openai")]
    
    # Dataset sampling options
    dataset_sample_percentage: float = 100.0  # Percentage of dataset to use (1.0-100.0)
    random_seed: Optional[int] = None  # None = truly random each run, int = reproducible sampling
    
    # Preprocessing options - CHANGED DEFAULT TO LETTERBOX
    preprocessing_mode: str = "letterbox"  # Options: resize, center_crop, letterbox (PRESERVES FULL IMAGE)
    normalize_embeddings: bool = True  # Always normalize embeddings to unit L2 norm
    
    # Output format options
    save_format: str = "both"  # Options: numpy, parquet, both
    embedding_precision: str = "float32"  # Options: float32, float16
    
    # Clustering parameters
    execution_mode: str = "fast"  # debug, fast, balanced, comprehensive
    top_n_results: int = 5
    min_cluster_size: int = 10
    max_cluster_size: int = 300
    target_cluster_size: int = 200
    
    # Similarity search parameters
    max_similar_per_cluster: int = 50
    max_similar_per_experiment: int = 25
    similarity_k: int = 20  # Number of similar images to find with FAISS
    use_faiss: bool = True  # Use FAISS for similarity search
    
    # Performance
    use_gpu: bool = True
    parallel_processing: bool = False
    mixed_precision: bool = True  # Use AMP for faster inference
    
    def __post_init__(self):
        """Initialize known similar pairs and experiment configurations if not provided."""
        if self.known_similar_pairs is None:
            # Default: no hardcoded validation pairs, let user specify them
            self.known_similar_pairs = []
        
        # Set default experiment configurations if not provided
        if self.experiment_mode:
            if self.experiment_resolutions is None:
                self.experiment_resolutions = [224, 336, 384]
            
            if self.experiment_models is None:
                self.experiment_models = [
                    ("ViT-B-32", "openai"),
                    ("ViT-B-16", "openai"), 
                    ("ViT-L-14", "openai")
                ]


# Model resolution mappings
MODEL_RESOLUTIONS = {
    "ViT-B-32": 224,
    "ViT-B-16": 224,
    "ViT-L-14": 224,
    "ViT-H-14": 336,  # Higher resolution for better detail
}

# Available pretrained weights for each model
MODEL_PRETRAINS = {
    "ViT-B-32": ["openai", "laion400m_e32", "laion2b_s34b_b88k"],
    "ViT-B-16": ["openai", "laion400m_e32"],
    "ViT-L-14": ["openai", "laion400m_e32", "laion2b_s34b_b88k"],
    "ViT-H-14": ["laion2b_s34b_b79k"],  # Only LAION weights available
}


class TimeoutError(Exception):
    pass


def load_image_with_timeout(file_path, timeout_seconds=30):
    """Load image with timeout using threading."""
    result = [None, None]  # [image, exception]
    
    def load_image():
        try:
            image = Image.open(file_path).convert('RGB')
            result[0] = image
        except Exception as e:
            result[1] = e
    
    thread = threading.Thread(target=load_image)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutError(f"Loading image timed out after {timeout_seconds} seconds")
    
    if result[1] is not None:
        raise result[1]
    
    return result[0]


def create_enhanced_preprocessor(target_size: int, mode: str = "resize"):
    """Create enhanced preprocessing function with multiple options."""
    
    def preprocess_image(image: Image.Image) -> torch.Tensor:
        """Enhanced preprocessing with configurable options."""
        
        if mode == "center_crop":
            # Center crop to square, then resize
            min_dim = min(image.size)
            left = (image.width - min_dim) // 2
            top = (image.height - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
        elif mode == "letterbox":
            # Letterbox (pad to square maintaining aspect ratio)
            image = ImageOps.pad(image, (target_size, target_size), Image.Resampling.LANCZOS)
            
        else:  # mode == "resize"
            # Simple resize (may distort aspect ratio)
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor
    
    return preprocess_image


class StreamlinedImageClusteringPipeline:
    """Complete pipeline for image clustering and similarity search."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        # Validate model and pretrained combination
        self._validate_model_config()
        
        # Validate dataset sampling percentage
        if not (0.1 <= config.dataset_sample_percentage <= 100.0):
            raise ValueError(f"dataset_sample_percentage must be between 0.1 and 100.0, got {config.dataset_sample_percentage}")
        
        # Set appropriate resolution for model
        if config.input_resolution == 224:  # Use default if not specified
            self.config.input_resolution = MODEL_RESOLUTIONS.get(config.clip_model, 224)
        
        # Create versioned output directories
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.run_timestamp}"
        
        # Base output directories
        base_output_dir = Path(config.output_base_dir)
        self.output_dir = base_output_dir / self.run_id
        
        # Embeddings are stored in base directory for reuse across runs
        self.embeddings_dir = base_output_dir / "embeddings"
        
        # Run-specific directories
        self.clustering_dir = self.output_dir / "clustering_results"
        self.similarity_dir = self.output_dir / "similarity_results"
        self.visualizations_dir = self.output_dir / "visualizations"
        self.reports_dir = self.output_dir / "reports"
        
        # Create directories
        for dir_path in [self.output_dir, self.embeddings_dir, self.clustering_dir, 
                         self.similarity_dir, self.visualizations_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create symlink to latest run for convenience
        latest_link = base_output_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(self.run_id, target_is_directory=True)
        except:
            pass  # Windows may not allow symlinks without admin rights
        
        # Initialize model and preprocessing
        self.model = None
        self.preprocess = None
        self.custom_preprocess = create_enhanced_preprocessor(
            self.config.input_resolution, 
            self.config.preprocessing_mode
        )
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Results storage
        self.embeddings = None
        self.image_paths = None
        self.clustering_results = []
        self.metrics = {}
        
        # Validation setup
        self.validation_pairs_indices = []  # Will be populated after loading images
        
        print(f"🚀 Enhanced Pipeline initialized")
        print(f"🏷️ Run ID: {self.run_id}")
        print(f"📁 Input images: {config.input_images_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Embeddings directory: {self.embeddings_dir} (shared across runs)")
        print(f"🔗 Latest symlink: {Path(config.output_base_dir) / 'latest'}")
        print(f"🎯 Target image: {config.target_image}")
        print(f"🎨 CLIP model: {config.clip_model} ({config.clip_pretrained})")
        print(f"📐 Input resolution: {self.config.input_resolution}x{self.config.input_resolution}")
        print(f"🔄 Preprocessing: {config.preprocessing_mode}")
        print(f"📦 Batch size: {config.batch_size}")
        print(f"💻 Device: {self.device}")
        if config.dataset_sample_percentage < 100.0:
            seed_info = "random each run" if config.random_seed is None else f"seed: {config.random_seed}"
            print(f"🎲 Dataset sampling: {config.dataset_sample_percentage}% ({seed_info})")
        if FAISS_AVAILABLE and config.use_faiss:
            gpu_status = "GPU" if FAISS_GPU_AVAILABLE else "CPU"
            print(f"🔍 FAISS similarity search: ✅ Enabled ({gpu_status})")
            print(f"📊 Similarity search k: {config.similarity_k}")
        else:
            print(f"🔍 FAISS similarity search: ❌ Disabled")
        print(f"🎨 Visualizations: ✅ Enabled")
        print(f"📝 Reports: ✅ Enabled")
        
    def _validate_model_config(self):
        """Validate model and pretrained combination."""
        if self.config.clip_model not in MODEL_PRETRAINS:
            raise ValueError(f"Unsupported model: {self.config.clip_model}")
        
        available_pretrains = MODEL_PRETRAINS[self.config.clip_model]
        if self.config.clip_pretrained not in available_pretrains:
            print(f"⚠️ Warning: {self.config.clip_pretrained} may not be available for {self.config.clip_model}")
            print(f"Available options: {available_pretrains}")
    
    def get_embedding_config_id(self, model: str = None, pretrained: str = None, resolution: int = None, preprocessing: str = None) -> str:
        """Generate a unique identifier for embedding configuration."""
        model = model or self.config.clip_model
        pretrained = pretrained or self.config.clip_pretrained
        resolution = resolution or self.config.input_resolution
        preprocessing = preprocessing or self.config.preprocessing_mode
        
        # Create a short, readable identifier
        model_short = model.replace("ViT-", "").replace("-", "")
        pretrained_short = pretrained.replace("openai", "oa").replace("laion", "la")[:4]
        
        return f"{model_short}_{pretrained_short}_{resolution}_{preprocessing}"
    
    def get_embedding_dir_for_config(self, model: str = None, pretrained: str = None, resolution: int = None, preprocessing: str = None) -> Path:
        """Get the embedding directory for a specific configuration."""
        config_id = self.get_embedding_config_id(model, pretrained, resolution, preprocessing)
        return self.embeddings_dir / config_id
    
    def is_embedding_config_compatible(self, existing_info: dict) -> bool:
        """Check if existing embeddings are compatible with current configuration."""
        current_config = {
            'model': self.config.clip_model,
            'pretrained': self.config.clip_pretrained,
            'input_resolution': self.config.input_resolution,
            'preprocessing_mode': self.config.preprocessing_mode,
            'normalized': self.config.normalize_embeddings
        }
        
        # Check all critical parameters
        for key, current_value in current_config.items():
            if existing_info.get(key) != current_value:
                print(f"  ⚠️ Config mismatch - {key}: existing={existing_info.get(key)} vs current={current_value}")
                return False
        
        return True
    
    def find_compatible_embeddings(self) -> Tuple[Optional[Path], Optional[dict]]:
        """Find existing embeddings compatible with current configuration."""
        if self.config.force_regenerate:
            print("🔄 Force regenerate enabled - skipping compatibility check")
            return None, None
        
        current_dir = self.get_embedding_dir_for_config()
        
        # Check if embeddings exist for current configuration
        if current_dir.exists():
            info_path = current_dir / "embedding_info.json"
            if info_path.exists():
                with open(info_path, "r") as f:
                    embedding_info = json.load(f)
                
                if self.is_embedding_config_compatible(embedding_info):
                    print(f"✅ Found compatible embeddings: {current_dir.name}")
                    return current_dir, embedding_info
                else:
                    print(f"❌ Existing embeddings not compatible: {current_dir.name}")
        
        # Search other embedding directories for compatibility
        print("🔍 Searching for compatible embeddings in other configurations...")
        for config_dir in self.embeddings_dir.iterdir():
            if config_dir.is_dir() and config_dir != current_dir:
                info_path = config_dir / "embedding_info.json"
                if info_path.exists():
                    with open(info_path, "r") as f:
                        embedding_info = json.load(f)
                    
                    if self.is_embedding_config_compatible(embedding_info):
                        print(f"✅ Found compatible embeddings in: {config_dir.name}")
                        return config_dir, embedding_info
        
        print("❌ No compatible embeddings found")
        return None, None
    
    def run_experiment_mode(self) -> Dict[str, Dict]:
        """Run experiments with multiple configurations and return results."""
        print("🧪 Running experimental mode with multiple configurations...")
        
        experiment_results = {}
        original_config = {
            'clip_model': self.config.clip_model,
            'clip_pretrained': self.config.clip_pretrained,
            'input_resolution': self.config.input_resolution,
            'preprocessing_mode': self.config.preprocessing_mode
        }
        
        total_experiments = len(self.config.experiment_models) * len(self.config.experiment_resolutions)
        experiment_count = 0
        
        # Get image paths once and apply sampling
        all_image_paths = self.get_image_paths()
        sampled_image_paths = self.apply_dataset_sampling(all_image_paths)
        
        for model, pretrained in self.config.experiment_models:
            for resolution in self.config.experiment_resolutions:
                experiment_count += 1
                print(f"\n{'='*60}")
                print(f"🧪 Experiment {experiment_count}/{total_experiments}")
                print(f"   Model: {model} ({pretrained})")
                print(f"   Resolution: {resolution}x{resolution}")
                print(f"   Preprocessing: {self.config.preprocessing_mode}")
                print(f"{'='*60}")
                
                # Update configuration for this experiment
                self.config.clip_model = model
                self.config.clip_pretrained = pretrained
                self.config.input_resolution = resolution
                
                # Check for compatible embeddings
                compatible_dir, embedding_info = self.find_compatible_embeddings()
                
                if compatible_dir is not None and not self.config.force_regenerate:
                    print(f"✅ Using existing embeddings: {compatible_dir.name}")
                    # Load the existing embeddings for this config
                    embeddings_path = compatible_dir / "embeddings.npy"
                    if embeddings_path.exists():
                        embeddings = np.load(embeddings_path)
                        
                        # Store results
                        config_id = self.get_embedding_config_id(model, pretrained, resolution)
                        experiment_results[config_id] = {
                            'model': model,
                            'pretrained': pretrained,
                            'resolution': resolution,
                            'preprocessing': self.config.preprocessing_mode,
                            'embeddings_shape': embeddings.shape,
                            'embedding_info': embedding_info,
                            'reused_existing': True
                        }
                        continue
                
                try:
                    # Generate embeddings for this configuration
                    print(f"🔄 Generating embeddings for {model}@{resolution}...")
                    
                    # Initialize model for this configuration
                    self.initialize_clip_model()
                    
                    # Generate embeddings
                    processed_paths, embeddings = self.extract_embeddings(sampled_image_paths)
                    
                    # Save embeddings
                    self.save_embeddings(processed_paths, embeddings)
                    
                    # Store results
                    config_id = self.get_embedding_config_id(model, pretrained, resolution)
                    experiment_results[config_id] = {
                        'model': model,
                        'pretrained': pretrained,
                        'resolution': resolution,
                        'preprocessing': self.config.preprocessing_mode,
                        'embeddings_shape': embeddings.shape,
                        'total_images': len(processed_paths),
                        'reused_existing': False
                    }
                    
                    print(f"✅ Generated {embeddings.shape[0]} embeddings")
                    
                except Exception as e:
                    print(f"❌ Failed to generate embeddings for {model}@{resolution}: {str(e)}")
                    config_id = self.get_embedding_config_id(model, pretrained, resolution)
                    experiment_results[config_id] = {
                        'model': model,
                        'pretrained': pretrained,
                        'resolution': resolution,
                        'preprocessing': self.config.preprocessing_mode,
                        'error': str(e),
                        'reused_existing': False
                    }
        
        # Restore original configuration
        self.config.clip_model = original_config['clip_model']
        self.config.clip_pretrained = original_config['clip_pretrained']
        self.config.input_resolution = original_config['input_resolution']
        self.config.preprocessing_mode = original_config['preprocessing_mode']
        
        # Save experiment summary
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(experiment_results, f, indent=2, default=str)
        
        print(f"\n🎉 Completed {experiment_count} experiments!")
        print(f"📊 Results saved to: {summary_path}")
        print(f"💾 All embeddings saved in: {self.embeddings_dir}")
        
        # Print summary
        print("\n📈 Experiment Summary:")
        print("-" * 80)
        for config_id, result in experiment_results.items():
            status = "✅ SUCCESS" if 'error' not in result else "❌ FAILED"
            reused = " (reused)" if result.get('reused_existing', False) else " (new)"
            print(f"{config_id:25} | {status:10} | {result['model']:10} | {result['resolution']:3}px{reused}")
        
        return experiment_results
    
    def _setup_validation_pairs(self):
        """Find indices of known similar pairs for validation."""
        if not self.config.known_similar_pairs or not self.image_paths:
            return
        
        self.validation_pairs_indices = []
        
        print(f"🔍 Setting up validation for {len(self.config.known_similar_pairs)} known similar pairs...")
        
        for img1_name, img2_name in self.config.known_similar_pairs:
            idx1 = idx2 = None
            
            # Find indices of both images
            for i, img_path in enumerate(self.image_paths):
                img_name = Path(img_path).name
                if img_name == img1_name:
                    idx1 = i
                elif img_name == img2_name:
                    idx2 = i
            
            if idx1 is not None and idx2 is not None:
                self.validation_pairs_indices.append((idx1, idx2, img1_name, img2_name))
                print(f"  ✅ Found validation pair: {img1_name} (idx {idx1}) ↔ {img2_name} (idx {idx2})")
            else:
                missing = []
                if idx1 is None:
                    missing.append(img1_name)
                if idx2 is None:
                    missing.append(img2_name)
                print(f"  ❌ Missing validation images: {', '.join(missing)}")
        
        print(f"🎯 Setup complete: {len(self.validation_pairs_indices)} validation pairs ready")
    
    def _calculate_validation_score(self, labels: np.ndarray) -> float:
        """Calculate validation score based on whether known similar pairs are clustered together."""
        if not self.validation_pairs_indices:
            return 1.0  # No validation data, assume perfect
        
        correct_pairs = 0
        total_pairs = len(self.validation_pairs_indices)
        
        for idx1, idx2, img1_name, img2_name in self.validation_pairs_indices:
            if idx1 < len(labels) and idx2 < len(labels):
                cluster1 = labels[idx1]
                cluster2 = labels[idx2]
                
                # Both should be in the same non-noise cluster
                if cluster1 != -1 and cluster1 == cluster2:
                    correct_pairs += 1
        
        validation_score = correct_pairs / total_pairs if total_pairs > 0 else 1.0
        return validation_score
    
    def initialize_clip_model(self):
        """Initialize enhanced CLIP model with optimizations."""
        print(f"🔧 Loading CLIP model: {self.config.clip_model} ({self.config.clip_pretrained})")
        
        try:
            self.model, _, _ = open_clip.create_model_and_transforms(
                self.config.clip_model, 
                pretrained=self.config.clip_pretrained, 
                device=self.device
            )
            self.model.eval()
            
            # Enable mixed precision if requested and available
            if self.config.mixed_precision and self.device.type == 'cuda':
                self.model = self.model.half()
                print("✅ Mixed precision enabled (FP16)")
            
            print("✅ CLIP model loaded successfully")
            
            # Print model info
            if hasattr(self.model.visual, 'image_size'):
                print(f"📐 Model native resolution: {self.model.visual.image_size}")
            
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"🔢 Model parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # Fallback to smaller model if large model fails
            if self.config.clip_model in ["ViT-L-14", "ViT-H-14"]:
                print("🔄 Falling back to ViT-B-32...")
                self.config.clip_model = "ViT-B-32"
                self.config.clip_pretrained = "openai"
                self.config.input_resolution = 224
                self.initialize_clip_model()
            else:
                raise
    
    def get_image_paths(self) -> List[str]:
        """Get all valid image paths from input directory."""
        input_dir = Path(self.config.input_images_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        print(f"🔍 Scanning for images in {input_dir}")
        
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(input_dir.rglob(f"*{ext}"))
            image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        image_paths = [str(f) for f in image_files]
        print(f"📸 Found {len(image_paths)} total images")
        
        return image_paths
    
    def apply_dataset_sampling(self, image_paths: List[str]) -> List[str]:
        """Apply dataset sampling to image paths BEFORE embedding extraction."""
        if self.config.dataset_sample_percentage >= 100.0:
            print(f"📊 Using full dataset: {len(image_paths)} images")
            return image_paths
        
        print(f"\n🎲 DATASET SAMPLING: Using {self.config.dataset_sample_percentage}% of dataset")
        print("-" * 50)
        
        # Set random seed - use truly random if None, otherwise use specified seed
        if self.config.random_seed is None:
            # Generate truly random seed for each run
            actual_seed = np.random.randint(0, 2**32 - 1)
            print(f"🎲 Using truly random seed: {actual_seed} (different each run)")
        else:
            # Use specified seed for reproducible sampling
            actual_seed = self.config.random_seed
            print(f"🔄 Using reproducible seed: {actual_seed} (same subset each run)")
        
        np.random.seed(actual_seed)
        
        original_count = len(image_paths)
        sample_size = max(1, int(original_count * self.config.dataset_sample_percentage / 100.0))
        
        # Find target image index to ensure it's always included
        target_image_idx = None
        for i, img_path in enumerate(image_paths):
            if Path(img_path).name == self.config.target_image:
                target_image_idx = i
                print(f"🎯 Found target image '{self.config.target_image}' at index {i}")
                break
        
        if target_image_idx is None:
            print(f"⚠️ Target image '{self.config.target_image}' not found in dataset - will search all images")
            # Show first few image names for debugging
            sample_names = [Path(p).name for p in image_paths[:10]]
            print(f"📋 First 10 image names: {sample_names}")
        
        # Find validation pair image indices to ensure they're included
        validation_image_indices = set()
        if self.config.known_similar_pairs:
            print(f"🔍 Checking validation pair images for inclusion in sample...")
            for img1_name, img2_name in self.config.known_similar_pairs:
                for i, img_path in enumerate(image_paths):
                    img_name = Path(img_path).name
                    if img_name == img1_name or img_name == img2_name:
                        validation_image_indices.add(i)
                        print(f"  🎯 Found validation image '{img_name}' at index {i}")
        
        # Collect all indices that must be force-included
        force_include_indices = set()
        if target_image_idx is not None:
            force_include_indices.add(target_image_idx)
            print(f"🎯 Target image '{self.config.target_image}' force-included in sample")
        force_include_indices.update(validation_image_indices)
        
        if validation_image_indices:
            print(f"🎯 {len(validation_image_indices)} validation images force-included in sample")
        
        # Always include force-included images, even if it slightly exceeds percentage
        remaining_indices = list(range(original_count))
        for idx in force_include_indices:
            remaining_indices.remove(idx)  # Remove force-included from remaining pool
        
        # Sample the remaining images (sample_size - force_included_count)
        force_included_count = len(force_include_indices)
        remaining_sample_size = max(0, sample_size - force_included_count)
        
        if remaining_sample_size > 0 and len(remaining_indices) > 0:
            remaining_sample_size = min(remaining_sample_size, len(remaining_indices))
            sampled_remaining = np.random.choice(remaining_indices, size=remaining_sample_size, replace=False)
            sample_indices = np.concatenate([list(force_include_indices), sampled_remaining])
        else:
            sample_indices = list(force_include_indices)
        
        sample_indices = np.sort(sample_indices).astype(int)  # Keep original order and ensure integers
        
        # If no target image was found, warn user but continue
        if target_image_idx is None:
            print(f"⚠️ Target image '{self.config.target_image}' not found in original dataset - similarity search will fail")
        
        # Apply sampling to get final image paths
        sampled_paths = [image_paths[i] for i in sample_indices]
        
        print(f"✅ Sampled {len(sampled_paths)}/{original_count} images ({self.config.dataset_sample_percentage}%)")
        print(f"   📊 Reduction: {original_count - len(sampled_paths)} images skipped")
        print(f"   💡 Embeddings will be extracted ONLY for sampled images")
        
        return sampled_paths
    
    def extract_embeddings(self, image_paths: List[str]) -> Tuple[List[str], np.ndarray]:
        """Extract CLIP embeddings with enhanced preprocessing and batch processing."""
        print(f"🔄 Extracting embeddings from {len(image_paths)} images...")
        print(f"⚙️ Using {self.config.preprocessing_mode} preprocessing at {self.config.input_resolution}x{self.config.input_resolution}")
        
        if self.model is None:
            self.initialize_clip_model()
        
        successful_paths = []
        all_embeddings = []
        failed_count = 0
        
        # Enable automatic mixed precision if using CUDA
        use_amp = self.config.mixed_precision and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Process in batches
        total_batches = (len(image_paths) + self.config.batch_size - 1) // self.config.batch_size
        
        for i in tqdm(range(0, len(image_paths), self.config.batch_size), 
                     desc="Extracting embeddings", total=total_batches):
            batch_paths = image_paths[i:i + self.config.batch_size]
            
            # Load and preprocess images for this batch
            batch_images = []
            batch_successful_paths = []
            
            for img_path in batch_paths:
                try:
                    # Check file size (skip very large files)
                    file_size_mb = Path(img_path).stat().st_size / (1024 * 1024)
                    if file_size_mb > 200:  # Increased threshold for high-res models
                        failed_count += 1
                        continue
                    
                    # Load image with timeout
                    image = load_image_with_timeout(img_path, self.config.timeout_seconds)
                    
                    # Check image dimensions (skip extremely large images)
                    width, height = image.size
                    if width * height > 100_000_000:  # 100MP limit
                        failed_count += 1
                        continue
                    
                    # Apply enhanced preprocessing
                    processed_img = self.custom_preprocess(image)
                    batch_images.append(processed_img)
                    batch_successful_paths.append(img_path)
                    
                except Exception as e:
                    failed_count += 1
                    continue
            
            if not batch_images:
                continue
            
            # Extract embeddings for this batch
            try:
                # Stack and move to device
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Convert to half precision if using mixed precision
                if self.config.mixed_precision and self.device.type == 'cuda':
                    batch_tensor = batch_tensor.half()
                
                # Extract embeddings
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            batch_embeddings = self.model.encode_image(batch_tensor)
                    else:
                        batch_embeddings = self.model.encode_image(batch_tensor)
                    
                    # Always normalize embeddings to unit L2 norm (standard for CLIP)
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                    
                    # Convert to desired precision and move to CPU
                    if self.config.embedding_precision == "float16":
                        batch_embeddings = batch_embeddings.half()
                    else:
                        batch_embeddings = batch_embeddings.float()
                    
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    all_embeddings.extend(batch_embeddings)
                    successful_paths.extend(batch_successful_paths)
                
                # Clean up GPU memory
                del batch_tensor, batch_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Error processing batch {i//self.config.batch_size + 1}: {e}")
                failed_count += len(batch_images)
                continue
            
            # Clean up
            del batch_images
            gc.collect()
        
        final_embeddings = np.array(all_embeddings) if all_embeddings else np.array([])
        
        print(f"✅ Successfully extracted {len(successful_paths)} embeddings")
        print(f"📊 Embedding shape: {final_embeddings.shape}")
        if failed_count > 0:
            print(f"⚠️ Failed to process {failed_count} images")
        
        return successful_paths, final_embeddings
    
    def save_embeddings(self, image_paths: List[str], embeddings: np.ndarray):
        """Save embeddings in configuration-specific directory with enhanced metadata."""
        # Use configuration-specific directory
        config_dir = self.get_embedding_dir_for_config()
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_id = self.get_embedding_config_id()
        print(f"💾 Saving embeddings to config: {config_id}")
        
        # Prepare data
        relative_paths = [Path(path).name for path in image_paths]
        
        # Save in NumPy format
        if self.config.save_format in ["numpy", "both"]:
            np.save(config_dir / "embeddings.npy", embeddings)
            
            with open(config_dir / "image_paths.json", "w") as f:
                json.dump(relative_paths, f, indent=2)
            
            with open(config_dir / "full_image_paths.json", "w") as f:
                json.dump(image_paths, f, indent=2)
            
            print(f"  ✅ NumPy format saved")
        
        # Save in Parquet format
        if self.config.save_format in ["parquet", "both"]:
            # Create DataFrame with embeddings and metadata
            df_data = {
                "image_name": relative_paths,
                "full_path": image_paths,
                "embedding": [emb.tolist() for emb in embeddings]
            }
            
            df = pd.DataFrame(df_data)
            df.to_parquet(config_dir / "embeddings.parquet", compression="snappy")
            
            print(f"  ✅ Parquet format saved")
        
        # Save comprehensive embedding info
        embedding_info = {
            "config_id": config_id,
            "model": self.config.clip_model,
            "pretrained": self.config.clip_pretrained,
            "input_resolution": self.config.input_resolution,
            "preprocessing_mode": self.config.preprocessing_mode,
            "batch_size": self.config.batch_size,
            "embedding_precision": self.config.embedding_precision,
            "normalized": self.config.normalize_embeddings,
            "mixed_precision": self.config.mixed_precision,
            "total_images": len(image_paths),
            "embedding_dimensions": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "embedding_shape": list(embeddings.shape),
            "save_format": self.config.save_format,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.config)
        }
        
        with open(config_dir / "embedding_info.json", "w") as f:
            json.dump(embedding_info, f, indent=2, default=str)
        
        print(f"✅ Saved embeddings: {embeddings.shape} in {config_dir.name} ({self.config.save_format} format)")
    
    def load_existing_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Load compatible embeddings from disk with smart compatibility checking."""
        print("📂 Searching for compatible embeddings...")
        
        # Find compatible embeddings
        compatible_dir, embedding_info = self.find_compatible_embeddings()
        
        if compatible_dir is None:
            raise FileNotFoundError(
                f"No compatible embeddings found.\n"
                f"Current config: {self.get_embedding_config_id()}\n"
                f"Use --force-regenerate to create new embeddings."
            )
        
        # Load from the compatible directory
        embeddings_path = compatible_dir / "embeddings.npy"
        image_paths_path = compatible_dir / "image_paths.json"
        full_paths_path = compatible_dir / "full_image_paths.json"
        
        # Try to load from NumPy format first
        if embeddings_path.exists() and image_paths_path.exists():
            # Load embeddings and paths
            embeddings = np.load(embeddings_path)
            
            # Try to load full paths first, fallback to relative paths
            if full_paths_path.exists():
                with open(full_paths_path, "r") as f:
                    image_paths = json.load(f)
            else:
                with open(image_paths_path, "r") as f:
                    relative_paths = json.load(f)
                # Convert relative paths to full paths
                input_dir = Path(self.config.input_images_dir)
                image_paths = [str(input_dir / path) for path in relative_paths]
            
            # Display embedding info
            print(f"  📊 Model: {embedding_info.get('model', 'Unknown')} ({embedding_info.get('pretrained', 'Unknown')})")
            print(f"  📐 Resolution: {embedding_info.get('input_resolution', 'Unknown')}x{embedding_info.get('input_resolution', 'Unknown')}")
            print(f"  🔄 Preprocessing: {embedding_info.get('preprocessing_mode', 'Unknown')}")
            print(f"  📅 Created: {embedding_info.get('timestamp', 'Unknown')}")
            
            print(f"✅ Loaded {len(embeddings)} embeddings (shape: {embeddings.shape})")
            print(f"📁 Associated with {len(image_paths)} image paths")
            
            return image_paths, embeddings
            
        # Try to load from Parquet format as fallback
        parquet_path = compatible_dir / "embeddings.parquet"
        if parquet_path.exists():
            print("  📦 Loading from Parquet format...")
            df = pd.read_parquet(parquet_path)
            
            embeddings = np.array([emb for emb in df['embedding'].values])
            image_paths = df['full_path'].tolist()
            
            print(f"✅ Loaded {len(embeddings)} embeddings from Parquet (shape: {embeddings.shape})")
            return image_paths, embeddings
        
        raise FileNotFoundError(
            f"Compatible directory found but no embeddings files: {compatible_dir}\n"
            f"Expected files: embeddings.npy + image_paths.json OR embeddings.parquet"
        )
    
    def generate_clustering_parameters(self, data_size: int) -> Dict[str, List[Dict]]:
        """Generate parameter combinations for different algorithms."""
        parameters = {}
        
        # K-Means baseline with standard k values: 50, 100, 150, 200
        baseline_k_values = [50, 100, 150, 200]
        # Filter k values that make sense for dataset size (at least 2x the cluster size)
        valid_k_values = [k for k in baseline_k_values if data_size >= k * 2]
        if not valid_k_values:  # For very small datasets
            valid_k_values = [min(baseline_k_values[0], data_size // 5)]
        
        parameters["kmeans"] = [
            {"n_clusters": k, "init": "k-means++", "n_init": 10, "random_state": 42}
            for k in valid_k_values
        ]
        
        # Define cluster counts for Gaussian Mixture (same as K-means valid values)
        cluster_counts = valid_k_values
        base_clusters = valid_k_values[0] if valid_k_values else 50
        
        # HDBSCAN parameters - adapted for dataset size
        # Make min_cluster_size adaptive to dataset size
        max_cluster_size = min(100, data_size // 10)  # Max 10% of dataset as cluster size
        min_cluster_size_base = max(5, data_size // 100)  # At least 1% of dataset
        
        if self.config.execution_mode == "debug":
            hdbscan_params = [{"min_cluster_size": min_cluster_size_base, "min_samples": 3}]
        elif self.config.execution_mode == "fast":
            hdbscan_params = [
                {"min_cluster_size": min_cluster_size_base, "min_samples": 3},
                {"min_cluster_size": min(min_cluster_size_base * 2, max_cluster_size), "min_samples": 5},
                {"min_cluster_size": min(min_cluster_size_base * 3, max_cluster_size), "min_samples": 8}
            ]
        else:
            hdbscan_params = []
            cluster_sizes = [min_cluster_size_base, min_cluster_size_base * 2, min_cluster_size_base * 3, min_cluster_size_base * 4]
            cluster_sizes = [min(cs, max_cluster_size) for cs in cluster_sizes if cs <= max_cluster_size]
            
            for min_cluster in cluster_sizes:
                for min_samples in [3, 5, 8, 12]:
                    if min_samples < min_cluster:  # min_samples should be < min_cluster_size
                        hdbscan_params.append({
                            "min_cluster_size": min_cluster,
                            "min_samples": min_samples
                        })
        
        # Update HDBSCAN parameters with supported distance metrics and selection methods
        # HDBSCAN supported metrics: euclidean, manhattan, chebyshev, minkowski, etc.
        # Note: euclidean on normalized embeddings is equivalent to cosine similarity
        hdbscan_metrics = ["euclidean", "manhattan", "chebyshev"]
        cluster_selection_methods = ["eom", "leaf"]  # eom = excess of mass, leaf = leaf selection
        
        for i, params in enumerate(hdbscan_params):
            # Cycle through different metrics and selection methods for diversity
            metric = hdbscan_metrics[i % len(hdbscan_metrics)]
            selection_method = cluster_selection_methods[i % len(cluster_selection_methods)]
            params.update({"metric": metric, "cluster_selection_method": selection_method})
        
        parameters["hdbscan"] = hdbscan_params
        
        # Log the metrics and parameters being used
        if hdbscan_params:
            metrics_used = list(set(p["metric"] for p in hdbscan_params))
            selection_methods = list(set(p["cluster_selection_method"] for p in hdbscan_params))
            min_cluster_sizes = list(set(p["min_cluster_size"] for p in hdbscan_params))
            print(f"🎯 HDBSCAN adaptive parameters:")
            print(f"   Dataset size: {data_size:,}, Min cluster sizes: {sorted(min_cluster_sizes)}")
            print(f"   Distance metrics: {', '.join(metrics_used)}")
            print(f"   Selection methods: {', '.join(selection_methods)}")
        
        # DBSCAN parameters
        if self.config.execution_mode == "debug":
            dbscan_params = [{"eps": 0.25, "min_samples": 10}]
        else:
            eps_values = [0.2, 0.25, 0.3] if self.config.execution_mode == "fast" else [0.15, 0.2, 0.25, 0.3, 0.35]
            min_samples_values = [10, 15] if self.config.execution_mode == "fast" else [5, 10, 15, 20]
            
            dbscan_params = []
            for eps in eps_values:
                for min_samples in min_samples_values:
                    dbscan_params.append({
                        "eps": eps,
                        "min_samples": min_samples,
                        "metric": "cosine"
                    })
        
        parameters["dbscan"] = dbscan_params
        
        # Gaussian Mixture parameters
        if self.config.execution_mode == "debug":
            gmm_params = [{"n_components": base_clusters, "covariance_type": "diag"}]
        else:
            n_components_list = cluster_counts[:3] if self.config.execution_mode == "fast" else cluster_counts
            cov_types = ["diag"] if self.config.execution_mode == "fast" else ["diag", "full"]
            
            gmm_params = []
            for n_comp in n_components_list:
                for cov_type in cov_types:
                    gmm_params.append({
                        "n_components": n_comp,
                        "covariance_type": cov_type,
                        "max_iter": 100,
                        "random_state": 42
                    })
        
        parameters["gaussian_mixture"] = gmm_params
        
        # UMAP + HDBSCAN for complex clusters (optional)
        if UMAP_AVAILABLE and self.config.execution_mode in ["balanced", "comprehensive"]:
            umap_hdbscan_params = [
                {
                    "umap_n_neighbors": 30,
                    "umap_min_dist": 0.1,
                    "umap_metric": "cosine",
                    "hdbscan_min_cluster_size": 30,
                    "hdbscan_metric": "euclidean"
                }
            ]
            parameters["umap_hdbscan"] = umap_hdbscan_params
        
        return parameters
    
    def evaluate_clustering(self, embeddings: np.ndarray, labels: np.ndarray, 
                          algorithm: str, parameters: Dict, runtime: float) -> Dict:
        """Evaluate clustering quality with validation."""
        non_noise_mask = labels != -1
        clean_labels = labels[non_noise_mask]
        
        n_clusters = len(np.unique(clean_labels)) if len(clean_labels) > 0 else 0
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Calculate silhouette score
        if len(clean_labels) > 0 and n_clusters > 1:
            try:
                # Sample for efficiency
                if len(clean_labels) > 5000:
                    sample_idx = np.random.choice(len(clean_labels), 5000, replace=False)
                    sample_embeddings = embeddings[non_noise_mask][sample_idx]
                    sample_labels = clean_labels[sample_idx]
                else:
                    sample_embeddings = embeddings[non_noise_mask]
                    sample_labels = clean_labels
                
                silhouette = silhouette_score(sample_embeddings, sample_labels)
            except:
                silhouette = -1.0
        else:
            silhouette = -1.0
        
        # Calculate balance score
        if n_clusters > 1:
            unique_labels, counts = np.unique(clean_labels, return_counts=True)
            mean_size = np.mean(counts)
            std_size = np.std(counts)
            cv = std_size / mean_size if mean_size > 0 else float('inf')
            balance_score = max(0, 1 - cv/2)
        else:
            balance_score = 0.0
        
        # Calculate validation score (NEW)
        validation_score = self._calculate_validation_score(labels)
        
        # Enhanced composite score with validation (NEW)
        sil_norm = (silhouette + 1) / 2
        noise_penalty = max(0, 1 - noise_ratio)
        
        # Adjust weights based on validation availability
        if self.validation_pairs_indices:
            # Include validation in scoring
            val_weight = self.config.validation_weight
            other_weight = (1.0 - val_weight) / 3
            composite_score = (other_weight * sil_norm + 
                             other_weight * balance_score + 
                             other_weight * noise_penalty + 
                             val_weight * validation_score)
        else:
            # Original scoring when no validation data
            composite_score = 0.3 * sil_norm + 0.4 * balance_score + 0.3 * noise_penalty
        
        # Detailed validation information
        validation_details = {}
        if self.validation_pairs_indices:
            validation_details = self._get_validation_details(labels)
        
        return {
            "algorithm": algorithm,
            "parameters": parameters,
            "n_clusters": n_clusters,
            "n_points": len(labels),
            "noise_ratio": noise_ratio,
            "silhouette_score": silhouette,
            "balance_score": balance_score,
            "validation_score": validation_score,  # NEW
            "validation_details": validation_details,  # NEW
            "composite_score": composite_score,
            "runtime": runtime,
            "labels": labels
        }
    
    def _get_validation_details(self, labels: np.ndarray) -> Dict:
        """Get detailed validation information for reporting."""
        details = {
            "total_pairs": len(self.validation_pairs_indices),
            "correct_pairs": 0,
            "pair_results": []
        }
        
        for idx1, idx2, img1_name, img2_name in self.validation_pairs_indices:
            if idx1 < len(labels) and idx2 < len(labels):
                cluster1 = labels[idx1]
                cluster2 = labels[idx2]
                
                is_correct = cluster1 != -1 and cluster1 == cluster2
                if is_correct:
                    details["correct_pairs"] += 1
                
                details["pair_results"].append({
                    "image1": img1_name,
                    "image2": img2_name,
                    "cluster1": int(cluster1),
                    "cluster2": int(cluster2),
                    "same_cluster": is_correct,
                    "both_non_noise": cluster1 != -1 and cluster2 != -1
                })
        
        details["accuracy"] = details["correct_pairs"] / details["total_pairs"] if details["total_pairs"] > 0 else 0.0
        return details
    
    def run_clustering_experiment(self, embeddings: np.ndarray, algorithm: str, 
                                parameters: Dict) -> Optional[Dict]:
        """Run a single clustering experiment."""
        try:
            start_time = time.time()
            
            # Validate input embeddings
            if not isinstance(embeddings, np.ndarray):
                raise ValueError(f"Expected numpy array for embeddings, got {type(embeddings)}")
            
            if embeddings.dtype.kind not in ['f', 'i']:
                raise ValueError(f"Embeddings must be numerical, got dtype {embeddings.dtype}")
            
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
            
            # Embeddings are already normalized during extraction
            embeddings_normalized = embeddings
            
            # Run clustering based on algorithm
            if algorithm == "kmeans":
                if len(embeddings) > 20000:
                    # Use MiniBatch for large datasets
                    mb_params = {k: v for k, v in parameters.items() 
                               if k in ['n_clusters', 'init', 'max_iter', 'random_state']}
                    mb_params['batch_size'] = min(1000, len(embeddings) // 50)
                    clusterer = MiniBatchKMeans(**mb_params)
                else:
                    clusterer = KMeans(**parameters)
                labels = clusterer.fit_predict(embeddings_normalized)
                
            elif algorithm == "hdbscan":
                clusterer = hdbscan.HDBSCAN(**parameters)
                labels = clusterer.fit_predict(embeddings_normalized)
                
            elif algorithm == "dbscan":
                clusterer = DBSCAN(**parameters)
                labels = clusterer.fit_predict(embeddings_normalized)
                
            elif algorithm == "gaussian_mixture":
                clusterer = GaussianMixture(**parameters)
                clusterer.fit(embeddings_normalized)
                labels = clusterer.predict(embeddings_normalized)
                
            elif algorithm == "umap_hdbscan":
                # UMAP + HDBSCAN for complex clusters
                umap_model = umap.UMAP(
                    n_neighbors=parameters["umap_n_neighbors"],
                    min_dist=parameters["umap_min_dist"],
                    metric=parameters["umap_metric"],
                    random_state=42
                )
                umap_features = umap_model.fit_transform(embeddings_normalized)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=parameters["hdbscan_min_cluster_size"],
                    metric=parameters["hdbscan_metric"]
                )
                labels = clusterer.fit_predict(umap_features)
                
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            runtime = time.time() - start_time
            
            # Evaluate results
            result = self.evaluate_clustering(embeddings_normalized, labels, algorithm, parameters, runtime)
            
            # Cleanup
            del clusterer
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error in {algorithm}: {e}")
            return None
    
    def run_all_clustering_experiments(self, embeddings: np.ndarray) -> List[Dict]:
        """Run all clustering experiments."""
        print(f"🔬 Running clustering experiments...")
        
        parameter_sets = self.generate_clustering_parameters(len(embeddings))
        
        total_experiments = sum(len(params) for params in parameter_sets.values())
        print(f"📊 Total experiments: {total_experiments}")
        
        # Show KMeans baseline k values
        if "kmeans" in parameter_sets:
            k_values = [p["n_clusters"] for p in parameter_sets["kmeans"]]
            print(f"🎯 KMeans baseline k values: {k_values}")
            print(f"📏 Evaluation metric: Silhouette Score (higher is better)")
        
        results = []
        experiment_count = 0
        
        for algorithm, param_list in parameter_sets.items():
            print(f"\n🧪 Testing {algorithm} ({len(param_list)} configurations)")
            
            for params in param_list:
                experiment_count += 1
                print(f"  [{experiment_count}/{total_experiments}] {algorithm}: {params}")
                
                result = self.run_clustering_experiment(embeddings, algorithm, params)
                
                # Use more lenient threshold for HDBSCAN since it often produces more noise
                quality_threshold = 0.05 if algorithm == "hdbscan" else 0.1
                if result and result["composite_score"] > quality_threshold:
                    results.append(result)
                    val_info = ""
                    if "validation_score" in result and self.validation_pairs_indices:
                        val_score = result["validation_score"]
                        val_details = result.get("validation_details", {})
                        correct_pairs = val_details.get("correct_pairs", 0)
                        total_pairs = val_details.get("total_pairs", 0)
                        val_info = f", Validation: {correct_pairs}/{total_pairs} ({val_score:.1%})"
                    print(f"    ✅ Score: {result['composite_score']:.3f}, Clusters: {result['n_clusters']}, Noise: {result['noise_ratio']:.1%}{val_info}")
                else:
                    if result:
                        val_info = ""
                        if "validation_score" in result and self.validation_pairs_indices:
                            val_score = result["validation_score"]
                            val_details = result.get("validation_details", {})
                            correct_pairs = val_details.get("correct_pairs", 0)
                            total_pairs = val_details.get("total_pairs", 0)
                            val_info = f", Validation: {correct_pairs}/{total_pairs} ({val_score:.1%})"
                        print(f"    ❌ Poor quality result: Score={result['composite_score']:.3f}, Clusters={result['n_clusters']}, Noise={result['noise_ratio']:.1%}{val_info}")
                    else:
                        print(f"    ❌ Clustering failed completely")
        
        # Sort by composite score
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        
        print(f"\n🏆 Completed clustering: {len(results)} successful experiments")
        return results
    
    def save_clustering_results(self, results: List[Dict]):
        """Save clustering results."""
        print(f"💾 Saving clustering results...")
        
        # Save top results only
        top_results = results[:self.config.top_n_results]
        
        for i, result in enumerate(top_results, 1):
            result_dir = self.clustering_dir / f"rank_{i:02d}_{result['algorithm']}"
            result_dir.mkdir(exist_ok=True)
            
            # Save cluster labels
            np.save(result_dir / "cluster_labels.npy", result["labels"])
            
            # Create image-to-cluster mapping
            image_cluster_mapping = {}
            cluster_image_mapping = {}
            
            for img_path, label in zip(self.image_paths, result["labels"]):
                img_name = Path(img_path).name
                image_cluster_mapping[img_name] = int(label) if label != -1 else -1
                
                if label != -1:
                    label_int = int(label)
                    if label_int not in cluster_image_mapping:
                        cluster_image_mapping[label_int] = []
                    cluster_image_mapping[label_int].append(img_name)
            
            # Save mappings
            with open(result_dir / "image_to_cluster.json", "w") as f:
                json.dump(image_cluster_mapping, f, indent=2)
            
            with open(result_dir / "cluster_to_images.json", "w") as f:
                json.dump(cluster_image_mapping, f, indent=2)
            
            # Save result info
            result_info = {k: v for k, v in result.items() if k != "labels"}
            with open(result_dir / "clustering_info.json", "w") as f:
                json.dump(result_info, f, indent=2, default=str)
            
            print(f"  📁 Rank {i}: {result['algorithm']} (Score: {result['composite_score']:.3f})")
        
        # Save summary
        summary = {
            "total_experiments": len(results),
            "top_results_saved": len(top_results),
            "config": asdict(self.config),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "top_results": [
                {
                    "rank": i,
                    "algorithm": r["algorithm"],
                    "n_clusters": r["n_clusters"],
                    "noise_ratio": r["noise_ratio"],
                    "composite_score": r["composite_score"],
                    "runtime": r["runtime"],
                    "parameters": r["parameters"]
                }
                for i, r in enumerate(top_results, 1)
            ]
        }
        
        with open(self.clustering_dir / "clustering_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Saved top {len(top_results)} clustering results")
        
        # Verify target image presence in saved results
        print(f"🔍 Verifying target image '{self.config.target_image}' in clustering results...")
        target_image_name = Path(self.config.target_image).name
        
        for i, result in enumerate(top_results, 1):
            result_dir = self.clustering_dir / f"rank_{i:02d}_{result['algorithm']}"
            try:
                with open(result_dir / "image_to_cluster.json", "r") as f:
                    image_to_cluster = json.load(f)
                
                if target_image_name in image_to_cluster:
                    cluster_id = image_to_cluster[target_image_name]
                    if cluster_id == -1:
                        print(f"  ⚠️ Rank {i} ({result['algorithm']}): Target image in noise cluster")
                    else:
                        print(f"  ✅ Rank {i} ({result['algorithm']}): Target image in cluster {cluster_id}")
                else:
                    print(f"  ❌ Rank {i} ({result['algorithm']}): Target image NOT FOUND")
                    # Show a sample of what images are in the mapping
                    sample_images = list(image_to_cluster.keys())[:5]
                    print(f"      Sample images: {sample_images}")
                    
            except Exception as e:
                print(f"  ❌ Rank {i} ({result['algorithm']}): Error reading results - {e}")
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> 'faiss.Index':
        """Create FAISS index for similarity search."""
        d = embeddings.shape[1]
        
        if len(embeddings) > 50000:
            # Use IVF + PQ for large datasets (memory efficient)
            nlist = min(4096, int(np.sqrt(len(embeddings))))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, 64, 8)
            index.train(embeddings.astype(np.float32))
            index_type = f"IVF+PQ (nlist={nlist})"
        else:
            # Use flat index for smaller datasets (exact search)
            index = faiss.IndexFlatIP(d)
            index_type = "Flat"
        
        # Move to GPU if available
        if FAISS_GPU_AVAILABLE:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
            print(f"🚀 Created FAISS GPU {index_type} index for {len(embeddings)} embeddings")
        else:
            print(f"📦 Created FAISS CPU {index_type} index for {len(embeddings)} embeddings")
        
        index.add(embeddings.astype(np.float32))
        return index
    
    def find_similar_with_faiss(self, target_image: str, k: int = 10) -> Dict:
        """Find similar images using FAISS index with validation tracking."""
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available, falling back to cluster-based search")
            return self.find_similar_images(target_image)
        
        print(f"\n🎯 FAISS similarity search for: {target_image} (k={k})")
        
        # Find target image index
        target_idx = None
        for i, img_path in enumerate(self.image_paths):
            if Path(img_path).name == target_image:
                target_idx = i
                break
        
        if target_idx is None:
            print(f"❌ Target image '{target_image}' not found in dataset")
            return {}
        
        # Create FAISS index
        faiss_index = self._create_faiss_index(self.embeddings)
        
        # Query vector (already normalized)
        query_vector = self.embeddings[target_idx:target_idx+1].astype(np.float32)
        
        # Search for similar images
        scores, indices = faiss_index.search(query_vector, k + 1)  # +1 to exclude target itself
        
        # Filter out the target image and get results
        similar_images = []
        known_similar_found = []  # Track validation images found
        
        for score, idx in zip(scores[0], indices[0]):
            if idx != target_idx and idx < len(self.image_paths):
                img_name = Path(self.image_paths[idx]).name
                similar_images.append({
                    "image": img_name,
                    "similarity_score": float(score),
                    "index": int(idx)
                })
                
                # Check if this is a known similar image
                if self.validation_pairs_indices:
                    for val_idx1, val_idx2, val_img1, val_img2 in self.validation_pairs_indices:
                        if target_idx == val_idx1 and idx == val_idx2:
                            known_similar_found.append((val_img2, float(score), len(similar_images)))
                        elif target_idx == val_idx2 and idx == val_idx1:
                            known_similar_found.append((val_img1, float(score), len(similar_images)))
        
        # Validation analysis
        validation_analysis = {}
        if self.validation_pairs_indices and known_similar_found:
            for known_img, similarity_score, rank in known_similar_found:
                print(f"  🎯 Validation: Found known similar image '{known_img}' at rank {rank} (score: {similarity_score:.4f})")
                validation_analysis[known_img] = {
                    "rank": rank,
                    "similarity_score": similarity_score,
                    "found_in_top_k": rank <= k
                }
        elif self.validation_pairs_indices:
            # Check if we should have found a similar image
            for val_idx1, val_idx2, val_img1, val_img2 in self.validation_pairs_indices:
                if target_idx == val_idx1:
                    print(f"  ⚠️ Validation: Known similar image '{val_img2}' NOT found in top {k} results")
                elif target_idx == val_idx2:
                    print(f"  ⚠️ Validation: Known similar image '{val_img1}' NOT found in top {k} results")
        
        result = {
            "faiss_search": {
                "algorithm": "faiss_cosine_similarity",
                "query_image": target_image,
                "similar_images": similar_images[:k],
                "total_searched": len(self.embeddings),
                "search_method": "IVF+PQ" if len(self.embeddings) > 50000 else "Flat",
                "validation_analysis": validation_analysis  # NEW
            }
        }
        
        print(f"✅ Found {len(similar_images)} similar images using FAISS")
        if validation_analysis:
            known_count = len(validation_analysis)
            in_top_k = sum(1 for v in validation_analysis.values() if v["found_in_top_k"])
            print(f"🎯 Validation: {in_top_k}/{known_count} known similar images found in top {k}")
        
        return result
    
    def find_similar_in_cluster(self, target_image: str, cluster_labels: np.ndarray, k: int = 10) -> Dict:
        """Find similar images within the same cluster using FAISS."""
        if not FAISS_AVAILABLE:
            return {}
        
        # Find target image index and its cluster
        target_idx = None
        for i, img_path in enumerate(self.image_paths):
            if Path(img_path).name == target_image:
                target_idx = i
                break
        
        if target_idx is None or target_idx >= len(cluster_labels):
            return {}
        
        target_cluster = cluster_labels[target_idx]
        if target_cluster == -1:  # Noise cluster
            return {}
        
        # Get indices of images in the same cluster
        cluster_indices = np.where(cluster_labels == target_cluster)[0]
        if len(cluster_indices) <= 1:  # Only target image in cluster
            return {}
        
        # Create subset embeddings for this cluster
        cluster_embeddings = self.embeddings[cluster_indices]
        
        # Create FAISS index for cluster
        d = cluster_embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        
        # Use GPU if available
        if FAISS_GPU_AVAILABLE:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
        
        index.add(cluster_embeddings.astype(np.float32))
        
        # Find target within cluster embeddings
        target_in_cluster = np.where(cluster_indices == target_idx)[0][0]
        query_vector = cluster_embeddings[target_in_cluster:target_in_cluster+1].astype(np.float32)
        
        # Search within cluster
        scores, indices = index.search(query_vector, min(k + 1, len(cluster_embeddings)))
        
        # Convert cluster-local indices back to global indices
        similar_images = []
        for score, idx in zip(scores[0], indices[0]):
            global_idx = cluster_indices[idx]
            if global_idx != target_idx:
                img_name = Path(self.image_paths[global_idx]).name
                similar_images.append({
                    "image": img_name,
                    "similarity_score": float(score),
                    "cluster_id": int(target_cluster)
                })
        
        return {
            "cluster_filtered": {
                "algorithm": "faiss_cluster_filtered",
                "query_image": target_image,
                "cluster_id": int(target_cluster),
                "cluster_size": len(cluster_indices),
                "similar_images": similar_images[:k]
            }
        }
    
    def find_similar_images(self, target_image: str) -> Dict:
        """Find images similar to target image using clustering results."""
        print(f"\n🎯 Finding images similar to: {target_image}")
        
        # Check if target image exists in our dataset
        target_found = False
        target_index = None
        for i, img_path in enumerate(self.image_paths):
            if Path(img_path).name == target_image:
                target_found = True
                target_index = i
                break
        
        if not target_found:
            print(f"❌ Target image '{target_image}' not found in dataset")
            print(f"📋 Available images (first 10): {[Path(p).name for p in self.image_paths[:10]]}")
            return {}
        
        print(f"✅ Target image found at index {target_index}")
        similar_results = {}
        
        # Use in-memory clustering results if available (preferred method)
        if hasattr(self, 'clustering_results') and self.clustering_results:
            print(f"🧠 Using in-memory clustering results (ensures latest results)")
            
            top_results = self.clustering_results[:self.config.top_n_results]
            
            for i, result in enumerate(top_results, 1):
                algorithm = result['algorithm']
                labels = result['labels']
                
                # Get target image's cluster
                target_cluster = labels[target_index]
                
                if target_cluster == -1:
                    print(f"  ⚠️ Rank {i} ({algorithm}): Target image classified as noise (cluster -1)")
                    continue
                
                # Find all images in the same cluster
                cluster_indices = np.where(labels == target_cluster)[0]
                cluster_images = []
                
                for idx in cluster_indices:
                    if idx < len(self.image_paths):
                        img_name = Path(self.image_paths[idx]).name
                        if img_name != target_image:  # Exclude target image itself
                            cluster_images.append(img_name)
                
                if not cluster_images:
                    print(f"  ⚠️ Rank {i} ({algorithm}): No other images in cluster {target_cluster}")
                    continue
                
                # Limit number of similar images
                if len(cluster_images) > self.config.max_similar_per_experiment:
                    cluster_images = cluster_images[:self.config.max_similar_per_experiment]
                
                # Create meaningful result name
                meaningful_name = f"rank_{i:02d}_{algorithm}_score{result['composite_score']:.3f}"
                
                similar_results[meaningful_name] = {
                    "algorithm": algorithm,
                    "cluster_id": int(target_cluster),
                    "cluster_size": len(cluster_indices),
                    "similar_images": cluster_images,
                    "composite_score": result["composite_score"],
                    "parameters": result["parameters"],
                    "rank": i,
                    "original_exp_name": f"rank_{i:02d}_{algorithm}"
                }
                
                print(f"  ✅ {meaningful_name}: {len(cluster_images)} similar images (cluster {target_cluster}, size {len(cluster_indices)})")
        
        else:
            # Fallback to loading from disk (improved logic)
            print(f"💾 Loading clustering results from disk")
            
            # Process top clustering results
            for i in range(1, self.config.top_n_results + 1):
                result_dirs = list(self.clustering_dir.glob(f"rank_{i:02d}_*"))
                
                if not result_dirs:
                    print(f"  ⚠️ No results found for rank {i:02d}")
                    continue
                
                # If multiple directories match, pick the most recent one
                if len(result_dirs) > 1:
                    result_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    print(f"  🔄 Multiple results found for rank {i:02d}, using most recent: {result_dirs[0].name}")
                
                result_dir = result_dirs[0]
                exp_name = result_dir.name
                
                # Load mappings
                try:
                    with open(result_dir / "image_to_cluster.json", "r") as f:
                        image_to_cluster = json.load(f)
                    
                    with open(result_dir / "cluster_to_images.json", "r") as f:
                        cluster_to_images = json.load(f)
                    
                    with open(result_dir / "clustering_info.json", "r") as f:
                        info = json.load(f)
                    
                    # Convert cluster keys to integers
                    cluster_to_images = {int(k): v for k, v in cluster_to_images.items()}
                    
                    # Debug info
                    print(f"  🔍 {exp_name} ({info['algorithm']}): {len(image_to_cluster)} images, {len(cluster_to_images)} clusters")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {exp_name}: {e}")
                    continue
                
                # Check if target image is in this experiment
                if target_image not in image_to_cluster:
                    print(f"  ⚠️ {exp_name}: Target image not found in image_to_cluster mapping")
                    # Debug: check if any similar names exist
                    similar_names = [name for name in image_to_cluster.keys() if target_image[:20] in name]
                    if similar_names:
                        print(f"      💡 Similar names found: {similar_names[:3]}")
                    continue
                
                cluster_id = image_to_cluster[target_image]
                
                if cluster_id == -1:
                    print(f"  ⚠️ {exp_name}: Target image classified as noise (cluster -1)")
                    continue
                
                # Get similar images in the same cluster
                cluster_images = cluster_to_images.get(cluster_id, [])
                similar_images = [img for img in cluster_images if img != target_image]
                
                if not similar_images:
                    print(f"  ⚠️ {exp_name}: No other images in cluster {cluster_id}")
                    continue
                
                # Limit number of similar images
                if len(similar_images) > self.config.max_similar_per_experiment:
                    similar_images = similar_images[:self.config.max_similar_per_experiment]
                
                # Create meaningful result name
                meaningful_name = f"rank_{i:02d}_{info['algorithm']}_score{info['composite_score']:.3f}"
                
                similar_results[meaningful_name] = {
                    "algorithm": info["algorithm"],
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster_images),
                    "similar_images": similar_images,
                    "composite_score": info["composite_score"],
                    "parameters": info["parameters"],
                    "rank": i,
                    "original_exp_name": exp_name
                }
                
                print(f"  ✅ {meaningful_name}: {len(similar_images)} similar images (cluster {cluster_id}, size {len(cluster_images)})")
        
        return similar_results
    
    def save_similar_images(self, target_image: str, similar_results: Dict):
        """Save similar images to organized folders."""
        if not similar_results:
            print("❌ No similar images to save")
            return
        
        print(f"💾 Saving similar images...")
        
        # Create target-specific output directory
        target_name = Path(target_image).stem
        target_output_dir = self.similarity_dir / f"similar_to_{target_name}"
        target_output_dir.mkdir(exist_ok=True)
        
        # Copy target image
        target_source = None
        for img_path in self.image_paths:
            if Path(img_path).name == target_image:
                target_source = Path(img_path)
                break
        
        if target_source and target_source.exists():
            target_dest = target_output_dir / f"TARGET_{target_image}"
            shutil.copy2(target_source, target_dest)
            print(f"  📋 Copied target image")
        
        total_copied = 0
        
        # Process each experiment
        for exp_name, exp_data in similar_results.items():
            # Create meaningful experiment directory name
            if "faiss" in exp_name.lower():
                # FAISS-based search
                search_method = exp_data.get("search_method", "FAISS")
                meaningful_name = f"FAISS_{search_method}_top{len(exp_data['similar_images'])}"
            elif "cluster" in exp_name.lower():
                # Cluster-filtered FAISS search
                cluster_id = exp_data.get("cluster_id", "unknown")
                cluster_size = exp_data.get("cluster_size", 0)
                meaningful_name = f"ClusterFiltered_cluster{cluster_id}_size{cluster_size}"
            else:
                # Regular clustering-based search
                algorithm = exp_data.get("algorithm", "unknown")
                cluster_id = exp_data.get("cluster_id", "unknown")
                cluster_size = exp_data.get("cluster_size", 0)
                score = exp_data.get("composite_score", 0)
                meaningful_name = f"{algorithm.title()}_cluster{cluster_id}_size{cluster_size}_score{score:.3f}"
            
            exp_output_dir = target_output_dir / meaningful_name
            exp_output_dir.mkdir(exist_ok=True)
            
            copied_count = 0
            
            # Handle different similar_images formats
            similar_images_list = exp_data.get("similar_images", [])
            if isinstance(similar_images_list, list) and len(similar_images_list) > 0:
                # Check if it's a list of dictionaries (FAISS format) or strings (cluster format)
                if isinstance(similar_images_list[0], dict):
                    # FAISS format: list of {"image": name, "similarity_score": score}
                    image_names = [item["image"] for item in similar_images_list]
                else:
                    # Cluster format: list of image names
                    image_names = similar_images_list
            else:
                image_names = []
            
            for i, similar_image in enumerate(image_names):
                # Find source file
                source_file = None
                for img_path in self.image_paths:
                    if Path(img_path).name == similar_image:
                        source_file = Path(img_path)
                        break
                
                if source_file and source_file.exists():
                    # Include similarity score in filename if available
                    if (i < len(similar_images_list) and 
                        isinstance(similar_images_list[i], dict) and 
                        "similarity_score" in similar_images_list[i]):
                        score = similar_images_list[i]["similarity_score"]
                        dst_filename = f"{i+1:03d}_score{score:.4f}_{similar_image}"
                    else:
                        dst_filename = f"{i+1:03d}_{similar_image}"
                    
                    dst_path = exp_output_dir / dst_filename
                    
                    try:
                        shutil.copy2(source_file, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to copy {similar_image}: {e}")
            
            total_copied += copied_count
            
            # Save experiment info with safe field access
            exp_info = {
                "target_image": target_image,
                "experiment_name": meaningful_name,
                "original_exp_name": exp_name,
                "algorithm": exp_data.get("algorithm", "unknown"),
                "total_similar_found": len(image_names),
                "images_copied": copied_count,
                "similar_images": exp_data.get("similar_images", [])
            }
            
            # Add optional fields if they exist
            if "cluster_id" in exp_data:
                exp_info["cluster_id"] = exp_data["cluster_id"]
            if "cluster_size" in exp_data:
                exp_info["cluster_size"] = exp_data["cluster_size"]
            if "composite_score" in exp_data:
                exp_info["composite_score"] = exp_data["composite_score"]
            if "parameters" in exp_data:
                exp_info["parameters"] = exp_data["parameters"]
            if "search_method" in exp_data:
                exp_info["search_method"] = exp_data["search_method"]
            if "total_searched" in exp_data:
                exp_info["total_searched"] = exp_data["total_searched"]
            
            with open(exp_output_dir / "experiment_info.json", "w") as f:
                json.dump(exp_info, f, indent=2)
            
            print(f"  📁 {meaningful_name}: {copied_count} images")
        
        # Save overall summary
        summary = {
            "target_image": target_image,
            "total_experiments": len(similar_results),
            "total_images_copied": total_copied,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": asdict(self.config),
            "experiments": similar_results
        }
        
        with open(target_output_dir / "similarity_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved {total_copied} similar images to {target_output_dir}")
    
    def generate_comprehensive_metrics(self):
        """Generate comprehensive metrics and save to file."""
        print("📊 Generating comprehensive metrics...")
        
        metrics = {
            "pipeline_config": asdict(self.config),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_stats": {
                "total_images": len(self.image_paths) if self.image_paths else 0,
                "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
                "clip_model": self.config.clip_model,
                "clip_pretrained": self.config.clip_pretrained
            },
            "clustering_stats": {
                "total_experiments": len(self.clustering_results),
                "execution_mode": self.config.execution_mode,
                "top_results_analyzed": min(self.config.top_n_results, len(self.clustering_results))
            },
            "performance_summary": {}
        }
        
        # Add clustering performance details
        if self.clustering_results:
            top_results = self.clustering_results[:self.config.top_n_results]
            
            metrics["clustering_performance"] = {
                "best_result": {
                    "algorithm": top_results[0]["algorithm"],
                    "composite_score": top_results[0]["composite_score"],
                    "n_clusters": top_results[0]["n_clusters"],
                    "noise_ratio": top_results[0]["noise_ratio"],
                    "silhouette_score": top_results[0]["silhouette_score"],
                    "balance_score": top_results[0]["balance_score"],
                    "runtime": top_results[0]["runtime"],
                    "parameters": top_results[0]["parameters"]
                },
                "algorithm_performance": {}
            }
            
            # Algorithm performance summary
            for algorithm in ["kmeans", "hdbscan", "dbscan", "gaussian_mixture"]:
                algo_results = [r for r in self.clustering_results if r["algorithm"] == algorithm]
                if algo_results:
                    best_algo = max(algo_results, key=lambda x: x["composite_score"])
                    metrics["clustering_performance"]["algorithm_performance"][algorithm] = {
                        "experiments_run": len(algo_results),
                        "best_score": best_algo["composite_score"],
                        "best_clusters": best_algo["n_clusters"],
                        "best_noise_ratio": best_algo["noise_ratio"],
                        "best_parameters": best_algo["parameters"]
                    }
            
            # Score distribution
            scores = [r["composite_score"] for r in top_results]
            metrics["clustering_performance"]["score_distribution"] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "scores": scores
            }
        
        # Save metrics
        with open(self.output_dir / "comprehensive_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"✅ Comprehensive metrics saved")
        return metrics
    
    def create_clustering_visualizations(self):
        """Create comprehensive clustering visualizations."""
        if not hasattr(self, 'clustering_results') or not self.clustering_results:
            print("❌ No clustering results available for visualization")
            return
        
        print("🎨 Creating clustering visualizations...")
        
        # Prepare embeddings for dimensionality reduction
        embeddings_2d = self._reduce_dimensions_for_viz(self.embeddings)
        
        # Create visualizations for top results
        top_results = self.clustering_results[:min(5, len(self.clustering_results))]
        
        # 1. Individual clustering plots
        for i, result in enumerate(top_results, 1):
            self._create_individual_clustering_plot(
                embeddings_2d, result, i
            )
        
        # 2. Metrics comparison plots
        self._create_metrics_comparison_plots(self.clustering_results)
        
        # 3. Cluster size distribution plots
        self._create_cluster_size_plots(top_results)
        
        # 4. Silhouette analysis plots
        self._create_silhouette_plots(top_results)
        
        print(f"✅ Visualizations saved to {self.visualizations_dir}")
    
    def _reduce_dimensions_for_viz(self, embeddings: np.ndarray, method: str = "tsne") -> np.ndarray:
        """Reduce embeddings to 2D for visualization."""
        print(f"  🔄 Reducing dimensions using {method.upper()}...")
        
        # Sample if too many points for performance
        if len(embeddings) > 10000:
            sample_indices = np.random.choice(len(embeddings), 10000, replace=False)
            sample_embeddings = embeddings[sample_indices]
            print(f"  📊 Sampled {len(sample_embeddings)} points for visualization")
        else:
            sample_embeddings = embeddings
            sample_indices = np.arange(len(embeddings))
        
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_embeddings)//4))
            embeddings_2d = reducer.fit_transform(sample_embeddings)
        elif method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(sample_embeddings)//3))
            embeddings_2d = reducer.fit_transform(sample_embeddings)
        else:
            # Fallback to PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(sample_embeddings)
            method = "pca"
        
        return embeddings_2d, sample_indices, method
    
    def _create_individual_clustering_plot(self, embeddings_data: tuple, result: Dict, rank: int):
        """Create individual clustering visualization."""
        embeddings_2d, sample_indices, method = embeddings_data
        
        plt.figure(figsize=(12, 8))
        
        # Get labels for sampled points
        labels = result['labels'][sample_indices]
        unique_labels = np.unique(labels)
        
        # Color map
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c='black', s=20, alpha=0.6, label='Noise')
            else:
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[color], s=30, alpha=0.7, label=f'Cluster {label}')
        
        # Highlight target image if found
        target_image_name = self.config.target_image
        target_idx = None
        for i, img_path in enumerate(self.image_paths):
            if Path(img_path).name == target_image_name:
                target_idx = i
                break
        
        if target_idx is not None and target_idx in sample_indices:
            sample_target_idx = np.where(sample_indices == target_idx)[0][0]
            plt.scatter(embeddings_2d[sample_target_idx, 0], embeddings_2d[sample_target_idx, 1], 
                       c='red', s=200, marker='*', edgecolors='white', linewidth=2, 
                       label='Target Image', zorder=10)
        
        plt.title(f'Rank {rank}: {result["algorithm"].title()} Clustering\n'
                 f'Score: {result["composite_score"]:.3f}, Clusters: {result["n_clusters"]}, '
                 f'Noise: {result["noise_ratio"]:.1%}', fontsize=14)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        # Limit legend entries
        handles, labels_legend = plt.gca().get_legend_handles_labels()
        if len(handles) > 20:
            plt.legend(handles[:19] + [handles[-1]], labels_legend[:19] + [labels_legend[-1]], 
                      bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / f'rank_{rank:02d}_{result["algorithm"]}_clustering.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metrics_comparison_plots(self, results: List[Dict]):
        """Create metrics comparison plots."""
        # Prepare data
        algorithms = [r['algorithm'] for r in results]
        composite_scores = [r['composite_score'] for r in results]
        silhouette_scores = [r['silhouette_score'] for r in results]
        n_clusters = [r['n_clusters'] for r in results]
        noise_ratios = [r['noise_ratio'] for r in results]
        runtimes = [r['runtime'] for r in results]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clustering Algorithm Comparison', fontsize=16)
        
        # 1. Composite Score
        axes[0, 0].bar(range(len(results)), composite_scores, color='skyblue')
        axes[0, 0].set_title('Composite Score (Higher = Better)')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(range(len(results)))
        axes[0, 0].set_xticklabels([f'{alg}\n({i+1})' for i, alg in enumerate(algorithms)], rotation=45)
        
        # 2. Silhouette Score
        axes[0, 1].bar(range(len(results)), silhouette_scores, color='lightgreen')
        axes[0, 1].set_title('Silhouette Score (Higher = Better)')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(range(len(results)))
        axes[0, 1].set_xticklabels([f'{alg}\n({i+1})' for i, alg in enumerate(algorithms)], rotation=45)
        
        # 3. Number of Clusters
        axes[0, 2].bar(range(len(results)), n_clusters, color='orange')
        axes[0, 2].set_title('Number of Clusters')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_xticks(range(len(results)))
        axes[0, 2].set_xticklabels([f'{alg}\n({i+1})' for i, alg in enumerate(algorithms)], rotation=45)
        
        # 4. Noise Ratio
        axes[1, 0].bar(range(len(results)), [nr * 100 for nr in noise_ratios], color='salmon')
        axes[1, 0].set_title('Noise Ratio (Lower = Better)')
        axes[1, 0].set_ylabel('Percentage')
        axes[1, 0].set_xticks(range(len(results)))
        axes[1, 0].set_xticklabels([f'{alg}\n({i+1})' for i, alg in enumerate(algorithms)], rotation=45)
        
        # 5. Runtime
        axes[1, 1].bar(range(len(results)), runtimes, color='plum')
        axes[1, 1].set_title('Runtime (Lower = Better)')
        axes[1, 1].set_ylabel('Seconds')
        axes[1, 1].set_xticks(range(len(results)))
        axes[1, 1].set_xticklabels([f'{alg}\n({i+1})' for i, alg in enumerate(algorithms)], rotation=45)
        
        # 6. Algorithm Performance Summary
        algo_performance = {}
        for result in results:
            algo = result['algorithm']
            if algo not in algo_performance:
                algo_performance[algo] = []
            algo_performance[algo].append(result['composite_score'])
        
        algo_names = list(algo_performance.keys())
        avg_scores = [np.mean(scores) for scores in algo_performance.values()]
        max_scores = [np.max(scores) for scores in algo_performance.values()]
        
        x_pos = np.arange(len(algo_names))
        axes[1, 2].bar(x_pos - 0.2, avg_scores, 0.4, label='Average Score', color='lightblue')
        axes[1, 2].bar(x_pos + 0.2, max_scores, 0.4, label='Best Score', color='darkblue')
        axes[1, 2].set_title('Algorithm Performance Summary')
        axes[1, 2].set_ylabel('Composite Score')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(algo_names)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_cluster_size_plots(self, results: List[Dict]):
        """Create cluster size distribution plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Cluster Size Distributions', fontsize=16)
        
        for i, result in enumerate(results):
            row = i // 3
            col = i % 3
            
            if row >= 2:  # Only show top 5
                break
                
            labels = result['labels']
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            
            if len(unique_labels) > 0:
                axes[row, col].hist(counts, bins=min(20, len(counts)), alpha=0.7, edgecolor='black')
                axes[row, col].set_title(f'Rank {i+1}: {result["algorithm"].title()}\n'
                                       f'{len(unique_labels)} clusters')
                axes[row, col].set_xlabel('Cluster Size')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].axvline(np.mean(counts), color='red', linestyle='--', 
                                     label=f'Mean: {np.mean(counts):.1f}')
                axes[row, col].legend()
            
        # Hide empty subplots
        for i in range(len(results), 6):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.visualizations_dir / 'cluster_size_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_silhouette_plots(self, results: List[Dict]):
        """Create silhouette analysis plots."""
        print("  📊 Creating silhouette analysis plots...")
        
        for i, result in enumerate(results[:3], 1):  # Only top 3 for performance
            if result['silhouette_score'] <= 0:
                continue
                
            labels = result['labels']
            non_noise_mask = labels != -1
            
            if np.sum(non_noise_mask) < 100:
                continue
                
            # Sample for performance
            sample_size = min(5000, np.sum(non_noise_mask))
            sample_indices = np.random.choice(np.where(non_noise_mask)[0], sample_size, replace=False)
            
            sample_embeddings = self.embeddings[sample_indices]
            sample_labels = labels[sample_indices]
            
            try:
                from sklearn.metrics import silhouette_samples
                sample_silhouette_values = silhouette_samples(sample_embeddings, sample_labels)
                
                plt.figure(figsize=(10, 6))
                
                y_lower = 10
                unique_labels = np.unique(sample_labels)
                
                for label in unique_labels:
                    cluster_silhouette_values = sample_silhouette_values[sample_labels == label]
                    cluster_silhouette_values.sort()
                    
                    size_cluster = cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster
                    
                    color = plt.cm.Set3(label / len(unique_labels))
                    plt.fill_betweenx(np.arange(y_lower, y_upper),
                                    0, cluster_silhouette_values,
                                    facecolor=color, edgecolor=color, alpha=0.7)
                    
                    plt.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
                    y_lower = y_upper + 10
                
                plt.axvline(x=result['silhouette_score'], color="red", linestyle="--", 
                           label=f'Average Score: {result["silhouette_score"]:.3f}')
                
                plt.title(f'Rank {i}: {result["algorithm"].title()} - Silhouette Analysis')
                plt.xlabel('Silhouette Coefficient Values')
                plt.ylabel('Cluster Label')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(self.visualizations_dir / f'silhouette_rank_{i:02d}_{result["algorithm"]}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"  ⚠️ Silhouette plot failed for rank {i}: {e}")
    
    def create_human_readable_summary(self):
        """Create a comprehensive human-readable summary report."""
        print("📝 Creating human-readable summary report...")
        
        summary_path = self.reports_dir / "summary_report.md"
        
        with open(summary_path, "w") as f:
            f.write(f"# Image Clustering Pipeline Summary Report\n\n")
            f.write(f"**Run ID:** {self.run_id}  \n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
            f.write(f"**Target Image:** {self.config.target_image}\n\n")
            
            # Configuration Summary
            f.write("## 🔧 Configuration\n\n")
            f.write(f"- **CLIP Model:** {self.config.clip_model} ({self.config.clip_pretrained})\n")
            f.write(f"- **Input Resolution:** {self.config.input_resolution}x{self.config.input_resolution}\n")
            f.write(f"- **Preprocessing:** {self.config.preprocessing_mode}\n")
            f.write(f"- **Dataset Sample:** {self.config.dataset_sample_percentage}%\n")
            f.write(f"- **Execution Mode:** {self.config.execution_mode}\n")
            f.write(f"- **Batch Size:** {self.config.batch_size}\n")
            if self.config.use_faiss:
                f.write(f"- **FAISS Similarity Search:** Enabled (k={self.config.similarity_k})\n")
            f.write("\n")
            
            # Dataset Summary
            f.write("## 📊 Dataset Summary\n\n")
            f.write(f"- **Total Images Processed:** {len(self.image_paths):,}\n")
            f.write(f"- **Embedding Dimensions:** {self.embeddings.shape[1] if self.embeddings is not None else 'N/A'}\n")
            f.write(f"- **Input Directory:** `{self.config.input_images_dir}`\n\n")
            
            # Clustering Results
            if hasattr(self, 'clustering_results') and self.clustering_results:
                f.write("## 🏆 Clustering Results\n\n")
                f.write(f"**Total Experiments:** {len(self.clustering_results)}  \n")
                f.write(f"**Top Results Analyzed:** {min(self.config.top_n_results, len(self.clustering_results))}\n\n")
                
                # Top Results Table
                f.write("### Top Clustering Results\n\n")
                # Check if validation is available
                has_validation = self.validation_pairs_indices and len(self.clustering_results) > 0
                if has_validation:
                    f.write("| Rank | Algorithm | Score | Clusters | Noise | Silhouette | Validation | Runtime |\n")
                    f.write("|------|-----------|-------|----------|-------|------------|------------|----------|\n")
                else:
                    f.write("| Rank | Algorithm | Score | Clusters | Noise | Silhouette | Runtime |\n")
                    f.write("|------|-----------|-------|----------|-------|------------|----------|\n")
                
                top_results = self.clustering_results[:self.config.top_n_results]
                for i, result in enumerate(top_results, 1):
                    if has_validation and "validation_score" in result:
                        val_details = result.get("validation_details", {})
                        correct_pairs = val_details.get("correct_pairs", 0)
                        total_pairs = val_details.get("total_pairs", 0)
                        val_str = f"{correct_pairs}/{total_pairs}"
                        f.write(f"| {i} | {result['algorithm'].title()} | {result['composite_score']:.3f} | "
                               f"{result['n_clusters']} | {result['noise_ratio']:.1%} | "
                               f"{result['silhouette_score']:.3f} | {val_str} | {result['runtime']:.1f}s |\n")
                    else:
                        f.write(f"| {i} | {result['algorithm'].title()} | {result['composite_score']:.3f} | "
                               f"{result['n_clusters']} | {result['noise_ratio']:.1%} | "
                               f"{result['silhouette_score']:.3f} | {result['runtime']:.1f}s |\n")
                
                f.write("\n")
                
                # Best Result Details
                best_result = top_results[0]
                f.write("### 🥇 Best Result Details\n\n")
                f.write(f"**Algorithm:** {best_result['algorithm'].title()}  \n")
                f.write(f"**Composite Score:** {best_result['composite_score']:.3f}  \n")
                f.write(f"**Number of Clusters:** {best_result['n_clusters']}  \n")
                f.write(f"**Noise Ratio:** {best_result['noise_ratio']:.1%}  \n")
                f.write(f"**Silhouette Score:** {best_result['silhouette_score']:.3f}  \n")
                f.write(f"**Balance Score:** {best_result['balance_score']:.3f}  \n")
                f.write(f"**Runtime:** {best_result['runtime']:.2f} seconds\n\n")
                
                # Add validation information if available
                if self.validation_pairs_indices and "validation_score" in best_result:
                    val_details = best_result.get("validation_details", {})
                    f.write(f"**Validation Score:** {best_result['validation_score']:.1%}  \n")
                    f.write(f"**Known Similar Pairs Correctly Clustered:** {val_details.get('correct_pairs', 0)}/{val_details.get('total_pairs', 0)}\n\n")
                
                f.write("**Parameters:**\n")
                for key, value in best_result['parameters'].items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
                
                # Algorithm Performance Summary
                f.write("### 📈 Algorithm Performance Summary\n\n")
                algo_stats = {}
                for result in self.clustering_results:
                    algo = result['algorithm']
                    if algo not in algo_stats:
                        algo_stats[algo] = {
                            'count': 0,
                            'scores': [],
                            'best_score': 0,
                            'avg_clusters': []
                        }
                    algo_stats[algo]['count'] += 1
                    algo_stats[algo]['scores'].append(result['composite_score'])
                    algo_stats[algo]['best_score'] = max(algo_stats[algo]['best_score'], result['composite_score'])
                    algo_stats[algo]['avg_clusters'].append(result['n_clusters'])
                
                f.write("| Algorithm | Experiments | Best Score | Avg Score | Avg Clusters |\n")
                f.write("|-----------|-------------|------------|-----------|---------------|\n")
                
                for algo, stats in sorted(algo_stats.items(), key=lambda x: x[1]['best_score'], reverse=True):
                    avg_score = np.mean(stats['scores'])
                    avg_clusters = np.mean(stats['avg_clusters'])
                    f.write(f"| {algo.title()} | {stats['count']} | {stats['best_score']:.3f} | "
                           f"{avg_score:.3f} | {avg_clusters:.1f} |\n")
                f.write("\n")
            
            # Validation Analysis (NEW)
            if self.validation_pairs_indices:
                f.write("## 🎯 Validation Analysis\n\n")
                f.write(f"**Known Similar Pairs:** {len(self.validation_pairs_indices)}\n\n")
                
                for pair_idx, (idx1, idx2, img1, img2) in enumerate(self.validation_pairs_indices, 1):
                    f.write(f"### Validation Pair {pair_idx}\n")
                    f.write(f"- **Image 1:** {img1}\n")
                    f.write(f"- **Image 2:** {img2}\n\n")
                    
                    # Show how each algorithm handled this pair
                    if hasattr(self, 'clustering_results') and self.clustering_results:
                        f.write("**Clustering Results for This Pair:**\n\n")
                        f.write("| Rank | Algorithm | Image 1 Cluster | Image 2 Cluster | Same Cluster? |\n")
                        f.write("|------|-----------|-----------------|-----------------|---------------|\n")
                        
                        top_results = self.clustering_results[:self.config.top_n_results]
                        for i, result in enumerate(top_results, 1):
                            if idx1 < len(result['labels']) and idx2 < len(result['labels']):
                                cluster1 = result['labels'][idx1]
                                cluster2 = result['labels'][idx2]
                                same_cluster = "✅ Yes" if (cluster1 != -1 and cluster1 == cluster2) else "❌ No"
                                cluster1_str = str(cluster1) if cluster1 != -1 else "Noise"
                                cluster2_str = str(cluster2) if cluster2 != -1 else "Noise"
                                f.write(f"| {i} | {result['algorithm'].title()} | {cluster1_str} | {cluster2_str} | {same_cluster} |\n")
                        f.write("\n")
                f.write("\n")
            
            # Target Image Analysis
            target_cluster_info = self._get_target_image_cluster_info()
            if target_cluster_info:
                f.write("## 🎯 Target Image Analysis\n\n")
                f.write(f"**Target Image:** {self.config.target_image}\n\n")
                
                for rank, info in target_cluster_info.items():
                    f.write(f"### Rank {rank} - {info['algorithm'].title()}\n")
                    f.write(f"- **Cluster ID:** {info['cluster_id']}\n")
                    f.write(f"- **Cluster Size:** {info['cluster_size']} images\n")
                    f.write(f"- **Algorithm Score:** {info['score']:.3f}\n\n")
            
            # Similarity Search Results
            f.write("## 🔍 Similarity Search Summary\n\n")
            
            # Check if similarity results exist
            target_name = Path(self.config.target_image).stem
            similarity_results_dir = self.similarity_dir / f"similar_to_{target_name}"
            
            if similarity_results_dir.exists():
                summary_file = similarity_results_dir / "similarity_summary.json"
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as sf:
                            sim_data = json.load(sf)
                        
                        f.write(f"**Total Experiments:** {sim_data.get('total_experiments', 0)}  \n")
                        f.write(f"**Total Similar Images Found:** {sim_data.get('total_images_copied', 0)}\n\n")
                        
                        # Experiment breakdown
                        experiments = sim_data.get('experiments', {})
                        if experiments:
                            f.write("### Similarity Search Results by Method\n\n")
                            f.write("| Method | Similar Images | Details |\n")
                            f.write("|--------|----------------|----------|\n")
                            
                            for exp_name, exp_data in experiments.items():
                                similar_count = len(exp_data.get('similar_images', []))
                                if 'algorithm' in exp_data:
                                    method = exp_data['algorithm']
                                    if 'cluster_id' in exp_data:
                                        details = f"Cluster {exp_data['cluster_id']} (size: {exp_data.get('cluster_size', 'N/A')})"
                                    else:
                                        details = exp_data.get('search_method', 'N/A')
                                else:
                                    method = exp_name
                                    details = "N/A"
                                
                                f.write(f"| {method} | {similar_count} | {details} |\n")
                        
                    except Exception as e:
                        f.write(f"*Error loading similarity results: {e}*\n")
                else:
                    f.write("*Similarity search not yet completed*\n")
            else:
                f.write("*No similarity search results found*\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            
            if hasattr(self, 'clustering_results') and self.clustering_results:
                best_result = self.clustering_results[0]
                
                if best_result['composite_score'] > 0.8:
                    f.write("✅ **Excellent clustering quality** - The results are highly reliable.\n\n")
                elif best_result['composite_score'] > 0.6:
                    f.write("✅ **Good clustering quality** - The results are reliable for most use cases.\n\n")
                elif best_result['composite_score'] > 0.4:
                    f.write("⚠️ **Moderate clustering quality** - Consider adjusting parameters or trying different algorithms.\n\n")
                else:
                    f.write("❌ **Poor clustering quality** - Significant parameter tuning or different approach needed.\n\n")
                
                # Specific recommendations
                if best_result['noise_ratio'] > 0.3:
                    f.write("- **High noise ratio detected** - Consider using more restrictive clustering parameters.\n")
                
                if best_result['n_clusters'] < 10:
                    f.write("- **Few clusters found** - Dataset might benefit from different clustering approach or parameters.\n")
                elif best_result['n_clusters'] > 100:
                    f.write("- **Many small clusters** - Consider increasing minimum cluster size parameters.\n")
                
                # Algorithm recommendations
                best_algo = best_result['algorithm']
                f.write(f"- **Best performing algorithm:** {best_algo.title()}\n")
                
                if best_algo == 'hdbscan':
                    f.write("- HDBSCAN worked well - good for finding clusters of varying densities.\n")
                elif best_algo == 'kmeans':
                    f.write("- K-means worked well - dataset likely has spherical, well-separated clusters.\n")
                elif best_algo == 'gaussian_mixture':
                    f.write("- Gaussian Mixture worked well - dataset has overlapping or non-spherical clusters.\n")
            
            f.write("\n")
            
            # File Locations
            f.write("## 📁 Generated Files\n\n")
            f.write(f"**Run Directory:** `{self.output_dir}`\n")
            f.write(f"**Base Directory:** `{Path(self.config.output_base_dir)}`\n\n")
            f.write("### Key Directories:\n")
            f.write(f"- **Embeddings:** `{Path(self.config.output_base_dir)}/embeddings/` (shared across runs)\n")
            f.write(f"- **Clustering Results:** `{self.run_id}/clustering_results/`\n")
            f.write(f"- **Similar Images:** `{self.run_id}/similarity_results/`\n")
            f.write(f"- **Visualizations:** `{self.run_id}/visualizations/`\n")
            f.write(f"- **Reports:** `{self.run_id}/reports/`\n\n")
            
            f.write("### Key Files:\n")
            f.write(f"- **This Report:** `{self.run_id}/reports/summary_report.md`\n")
            f.write(f"- **Quick Summary:** `{self.run_id}/reports/quick_summary.txt`\n")
            f.write(f"- **Detailed Metrics:** `{self.run_id}/comprehensive_metrics.json`\n")
            f.write(f"- **Clustering Comparison:** `{self.run_id}/visualizations/metrics_comparison.png`\n")
            f.write(f"- **Clustering Plots:** `{self.run_id}/visualizations/rank_XX_algorithm_clustering.png`\n")
            f.write(f"- **Embeddings Data:** `embeddings/embeddings.npy` (reusable)\n")
            
            # Add timestamp
            f.write(f"\n---\n*Report generated on {time.strftime('%Y-%m-%d at %H:%M:%S')}*\n")
        
        print(f"✅ Human-readable summary saved to {summary_path}")
        
        # Also create a simple text version
        self._create_simple_text_summary()
    
    def _get_target_image_cluster_info(self) -> Dict:
        """Get target image cluster information across all results."""
        if not hasattr(self, 'clustering_results') or not self.clustering_results:
            return {}
        
        target_image_name = self.config.target_image
        target_idx = None
        
        for i, img_path in enumerate(self.image_paths):
            if Path(img_path).name == target_image_name:
                target_idx = i
                break
        
        if target_idx is None:
            return {}
        
        cluster_info = {}
        top_results = self.clustering_results[:self.config.top_n_results]
        
        for i, result in enumerate(top_results, 1):
            cluster_id = result['labels'][target_idx]
            if cluster_id != -1:
                cluster_size = np.sum(result['labels'] == cluster_id)
                cluster_info[i] = {
                    'algorithm': result['algorithm'],
                    'cluster_id': int(cluster_id),
                    'cluster_size': int(cluster_size),
                    'score': result['composite_score']
                }
        
        return cluster_info
    
    def _create_simple_text_summary(self):
        """Create a simple text summary for quick reading."""
        summary_path = self.reports_dir / "quick_summary.txt"
        
        with open(summary_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("IMAGE CLUSTERING PIPELINE - QUICK SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Target Image: {self.config.target_image}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if hasattr(self, 'clustering_results') and self.clustering_results:
                best = self.clustering_results[0]
                f.write("BEST CLUSTERING RESULT:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Algorithm: {best['algorithm'].title()}\n")
                f.write(f"Quality Score: {best['composite_score']:.3f}/1.0\n")
                f.write(f"Number of Clusters: {best['n_clusters']}\n")
                f.write(f"Noise Ratio: {best['noise_ratio']:.1%}\n")
                f.write(f"Runtime: {best['runtime']:.1f} seconds\n\n")
                
                # Quality assessment
                if best['composite_score'] > 0.7:
                    f.write("ASSESSMENT: Excellent clustering quality ✅\n")
                elif best['composite_score'] > 0.5:
                    f.write("ASSESSMENT: Good clustering quality ✅\n")
                elif best['composite_score'] > 0.3:
                    f.write("ASSESSMENT: Moderate clustering quality ⚠️\n")
                else:
                    f.write("ASSESSMENT: Poor clustering quality ❌\n")
                f.write("\n")
            
            f.write(f"Dataset: {len(self.image_paths):,} images processed\n")
            f.write(f"Results saved to: {self.output_dir}\n")
            
        print(f"✅ Quick summary saved to {summary_path}")
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        start_time = time.time()
        
        print("🚀 Starting Streamlined Image Clustering Pipeline")
        print("=" * 70)
        
        # Check if experiment mode is enabled
        if self.config.experiment_mode:
            print("🧪 EXPERIMENT MODE ENABLED")
            print("   This will test multiple CLIP models and resolutions")
            print("   Each configuration will generate/reuse embeddings as needed")
            print("=" * 70)
            
            # Run experiments and return early
            experiment_results = self.run_experiment_mode()
            
            total_time = time.time() - start_time
            print(f"\n🎉 Experiment mode completed in {total_time:.1f} seconds")
            print(f"📊 Generated/reused embeddings for {len(experiment_results)} configurations")
            
            return experiment_results
        
        try:
            # Phase 1: Extract embeddings
            print("\n📖 PHASE 1: EMBEDDING EXTRACTION")
            print("-" * 50)
            
            if self.config.skip_embeddings:
                print("⚠️ Skipping embedding extraction and loading existing embeddings.")
                try:
                    self.image_paths, self.embeddings = self.load_existing_embeddings()
                    print(f"  ✅ Loaded {len(self.embeddings)} embeddings from {self.embeddings_dir}")
                except FileNotFoundError:
                    raise ValueError("Embeddings not found and skip_embeddings is True. Please run embedding extraction first.")
            else:
                # Step 1: Get all image paths
                all_image_paths = self.get_image_paths()
                if not all_image_paths:
                    raise ValueError("No images found!")
                
                # Step 2: Apply dataset sampling BEFORE embedding extraction
                sampled_image_paths = self.apply_dataset_sampling(all_image_paths)
                
                # Step 3: Extract embeddings ONLY from sampled images
                self.image_paths, self.embeddings = self.extract_embeddings(sampled_image_paths)
                
                if len(self.embeddings) == 0:
                    raise ValueError("No embeddings extracted!")
                
                self.save_embeddings(self.image_paths, self.embeddings)
            
            # Validate that embeddings are numerical arrays, not strings
            if not isinstance(self.embeddings, np.ndarray):
                raise ValueError(f"Embeddings should be numpy array, got {type(self.embeddings)}")
            
            if self.embeddings.dtype.kind not in ['f', 'i']:  # float or integer
                raise ValueError(f"Embeddings should be numerical, got dtype {self.embeddings.dtype}")
            
            if self.embeddings.ndim != 2:
                raise ValueError(f"Embeddings should be 2D array, got shape {self.embeddings.shape}")
            
            print(f"✅ Validation passed: embeddings shape {self.embeddings.shape}, dtype {self.embeddings.dtype}")
            print(f"🔢 Image paths: {len(self.image_paths)} files")
            
            # Setup validation pairs now that we have image paths
            self._setup_validation_pairs()
            
            # Dataset sampling was applied before embedding extraction - much more efficient!
            print(f"✅ Final dataset ready: {len(self.image_paths)} images, embeddings shape: {self.embeddings.shape}")
            
            # Phase 2: Clustering
            print("\n🔬 PHASE 2: CLUSTERING")
            print("-" * 50)
            
            self.clustering_results = self.run_all_clustering_experiments(self.embeddings)
            
            if not self.clustering_results:
                raise ValueError("No successful clustering results!")
            
            self.save_clustering_results(self.clustering_results)
            
            # Phase 3: Similarity search
            print("\n🎯 PHASE 3: SIMILARITY SEARCH")
            print("-" * 50)
            
            # Use FAISS similarity search if available, otherwise fallback to cluster-based
            if FAISS_AVAILABLE and self.config.use_faiss:
                print("🚀 Using FAISS for fast similarity search")
                faiss_results = self.find_similar_with_faiss(self.config.target_image, k=self.config.similarity_k)
                cluster_results = self.find_similar_images(self.config.target_image)
                
                # Combine both results
                similar_results = {**faiss_results, **cluster_results}
            else:
                print("📊 Using cluster-based similarity search")
                similar_results = self.find_similar_images(self.config.target_image)
            
            if similar_results:
                self.save_similar_images(self.config.target_image, similar_results)
            else:
                print("❌ No similar images found")
            
            # Phase 4: Generate visualizations
            print("\n🎨 PHASE 4: VISUALIZATION GENERATION")
            print("-" * 50)
            
            self.create_clustering_visualizations()
            
            # Phase 5: Generate metrics and reports
            print("\n📊 PHASE 5: METRICS & REPORTING")
            print("-" * 50)
            
            metrics = self.generate_comprehensive_metrics()
            self.create_human_readable_summary()
            
            # Final summary
            total_time = time.time() - start_time
            base_output_dir = Path(self.config.output_base_dir)
            
            print("\n" + "=" * 70)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"**Run ID:** {self.run_id}")
            print(f"⏱️  Total time: {total_time/60:.1f} minutes")
            print(f"📸 Images processed: {len(self.image_paths):,}")
            print(f"🔬 Clustering experiments: {len(self.clustering_results)}")
            print(f"🏆 Best clustering score: {self.clustering_results[0]['composite_score']:.3f}")
            
            # Show validation results if available
            if self.validation_pairs_indices and len(self.clustering_results) > 0:
                best_result = self.clustering_results[0]
                if "validation_score" in best_result:
                    val_details = best_result.get("validation_details", {})
                    correct_pairs = val_details.get("correct_pairs", 0)
                    total_pairs = val_details.get("total_pairs", 0)
                    val_score = best_result["validation_score"]
                    print(f"🎯 Validation results: {correct_pairs}/{total_pairs} known similar pairs correctly clustered ({val_score:.1%})")
                    
                    # Show specific validation results
                    for pair_result in val_details.get("pair_results", []):
                        img1 = pair_result["image1"]
                        img2 = pair_result["image2"] 
                        if pair_result["same_cluster"]:
                            cluster_id = pair_result["cluster1"]
                            print(f"  ✅ {img1} ↔ {img2}: Both in cluster {cluster_id}")
                        else:
                            c1 = "Noise" if pair_result["cluster1"] == -1 else f"cluster {pair_result['cluster1']}"
                            c2 = "Noise" if pair_result["cluster2"] == -1 else f"cluster {pair_result['cluster2']}"
                            print(f"  ❌ {img1} ↔ {img2}: In different clusters ({c1} vs {c2})")
            
            print(f"📁 Results saved to: {self.output_dir}")
            
            if similar_results:
                total_similar = 0
                for exp_name, exp_data in similar_results.items():
                    similar_images = exp_data.get("similar_images", [])
                    if isinstance(similar_images, list) and len(similar_images) > 0:
                        if isinstance(similar_images[0], dict):
                            # FAISS format: count dictionary entries
                            total_similar += len(similar_images)
                        else:
                            # Cluster format: count string entries
                            total_similar += len(similar_images)
                print(f"🎯 Similar images found: {total_similar} across {len(similar_results)} experiments")
            
            print(f"\n📂 **Quick Access**: Latest results symlinked at: {Path(self.config.output_base_dir) / 'latest'}")
            print(f"📄 **Quick Summary**: {self.reports_dir / 'quick_summary.txt'}")
            print(f"📊 **Full Report**: {self.reports_dir / 'summary_report.md'}")
            
            print("\n🗂️ Key directories generated:")
            print(f"  • 💾 Embeddings: {base_output_dir}/embeddings/ (shared)")
            print(f"  • 🔬 Clustering results: {self.run_id}/clustering_results/")
            print(f"  • 🎯 Similar images: {self.run_id}/similarity_results/")
            print(f"  • 🎨 Visualizations: {self.run_id}/visualizations/")
            print(f"  • 📝 Reports: {self.run_id}/reports/")
            
            print("\n📋 Key files generated:")
            print(f"  • 📊 Metrics comparison: visualizations/metrics_comparison.png")
            print(f"  • 🎨 Clustering plots: visualizations/rank_XX_algorithm_clustering.png")
            print(f"  • 📈 Cluster distributions: visualizations/cluster_size_distributions.png")
            print(f"  • 📉 Silhouette analysis: visualizations/silhouette_rank_XX_algorithm.png")
            print(f"  • 📋 Comprehensive metrics: comprehensive_metrics.json")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            print(f"💾 Partial results may be available in: {self.output_dir}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Streamlined Image Clustering and Similarity Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with ViT-L/14 (recommended)
  python streamlined_image_clustering.py --target-image "my_image.jpg"
  
  # Fast testing with 1% of dataset (great for parameter tuning)
  python streamlined_image_clustering.py --target-image "my_image.jpg" --dataset-sample 1.0
  
  # Reproducible sampling - same 1% subset every run (useful for testing)
  python streamlined_image_clustering.py --target-image "my_image.jpg" --dataset-sample 1.0 --random-seed 42
  
  # Truly random sampling - different 5% subset every run (great for exploration)
  python streamlined_image_clustering.py --target-image "my_image.jpg" --dataset-sample 5.0
  
  # High-detail mode with ViT-H/14 at 336x336 resolution
  python streamlined_image_clustering.py --target-image "photo.png" --clip-model ViT-H-14 --input-resolution 336
  
  # GPU-optimized with large batch size and mixed precision
  python streamlined_image_clustering.py --target-image "img.jpg" --batch-size 1024 --mixed-precision
  
  # Center crop preprocessing for uniform aspect ratios
  python streamlined_image_clustering.py --target-image "image.jpg" --preprocessing-mode center_crop
  
  # Save in Parquet format only with float16 precision for smaller files
  python streamlined_image_clustering.py --target-image "pic.jpg" --save-format parquet --embedding-precision float16
  
  # Use FAISS for fast similarity search with 50 similar images
  python streamlined_image_clustering.py --target-image "image.jpg" --similarity-k 50
  
  # Disable FAISS and use only cluster-based search
  python streamlined_image_clustering.py --target-image "image.jpg" --no-faiss
  
  # Use custom validation pairs to guide clustering quality assessment
  python streamlined_image_clustering.py --target-image "image1.jpg" --validation-pairs "image1.jpg" "image2.jpg" "image3.jpg" "image4.jpg"
  
  # Adjust validation weight in composite scoring (higher = more emphasis on validation)
  python streamlined_image_clustering.py --target-image "image.jpg" --validation-weight 0.5
        """
    )
    
    # Required arguments
    parser.add_argument("--target-image", required=True,
                       help="Name of target image to find similar images for")
    
    # Validation arguments (NEW)
    parser.add_argument("--validation-pairs", nargs='*', 
                       help="Known similar image pairs for validation. Format: img1.jpg img2.jpg [img3.jpg img4.jpg ...]")
    parser.add_argument("--validation-weight", type=float, default=0.3,
                       help="Weight for validation score in composite score (0.0-1.0)")
    
    # Path arguments
    parser.add_argument("--input-dir", default="/mnt/d/TEST/images",
                       help="Directory containing input images")
    parser.add_argument("--output-dir", default="/mnt/c/Users/stuar/Downloads/image_clustering_results",
                       help="Base output directory")
    
    # Model arguments
    parser.add_argument("--clip-model", default="ViT-L-14",
                       choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14"],
                       help="CLIP model to use")
    parser.add_argument("--clip-pretrained", default="openai",
                       choices=["openai", "laion2b_s34b_b88k", "laion400m_e32", "laion2b_s34b_b79k"],
                       help="Pretrained weights to use")
    
    # Preprocessing arguments
    parser.add_argument("--input-resolution", type=int, default=224,
                       help="Input image resolution for CLIP model (e.g., 224, 336)")
    parser.add_argument("--preprocessing-mode", default="resize",
                       choices=["resize", "center_crop", "letterbox"],
                       help="Image preprocessing mode")
    parser.add_argument("--no-normalize-embeddings", action="store_true",
                       help="Disable embedding normalization (not recommended)")
    
    # Output format arguments
    parser.add_argument("--save-format", default="both",
                       choices=["numpy", "parquet", "both"],
                       help="Output format for embeddings (numpy, parquet, or both)")
    parser.add_argument("--embedding-precision", default="float32",
                       choices=["float32", "float16"],
                       help="Precision for embedding values (float32 or float16)")
    
    # Dataset sampling arguments
    parser.add_argument("--dataset-sample", type=float, default=100.0,
                       help="Percentage of dataset to use (1.0-100.0). Use 1.0 for 1%% of dataset for fast testing")
    parser.add_argument("--random-seed", type=int, default=None,
                       help="Random seed for dataset sampling. Use specific integer for reproducible sampling, or omit for truly random sampling each run (recommended for exploration)")
    
    # Clustering arguments
    parser.add_argument("--mode", default="fast",
                       choices=["debug", "fast", "balanced", "comprehensive"],
                       help="Execution mode")
    parser.add_argument("--top-results", type=int, default=5,
                       help="Number of top clustering results to analyze")
    
    # Similarity search arguments
    parser.add_argument("--similarity-k", type=int, default=20,
                       help="Number of similar images to find with FAISS")
    parser.add_argument("--no-faiss", action="store_true",
                       help="Disable FAISS similarity search (use cluster-based only)")
    
    # Experiment mode arguments
    parser.add_argument("--experiment-mode", action="store_true",
                       help="Run experiments with multiple CLIP models and resolutions")
    parser.add_argument("--experiment-models", nargs='*', 
                       help="Models to test in experiment mode. Format: ViT-B-32:openai ViT-L-14:openai")
    parser.add_argument("--experiment-resolutions", type=int, nargs='*',
                       help="Resolutions to test in experiment mode (e.g., 224 336 384)")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration of embeddings even if compatible ones exist")
    
    # Performance arguments
    parser.add_argument("--batch-size", type=int, default=512,
                       help="Batch size for embedding extraction (512 or 1024 recommended for GPU)")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout for loading each image")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable automatic mixed precision (FP16) for faster inference")
    parser.add_argument("--disable-mixed-precision", action="store_true",
                       help="Disable automatic mixed precision")
    parser.add_argument("--skip-embeddings", action="store_true",
                       help="Skip embedding extraction and load existing embeddings from output_dir/embeddings.npy")
    
    args = parser.parse_args()
    
    # Validate dataset sampling percentage
    if not (0.1 <= args.dataset_sample <= 100.0):
        parser.error(f"--dataset-sample must be between 0.1 and 100.0, got {args.dataset_sample}")
    
    # Determine mixed precision setting
    mixed_precision = True  # Default to enabled
    if args.disable_mixed_precision:
        mixed_precision = False
    elif args.mixed_precision:
        mixed_precision = True
    
    # Process validation pairs
    known_similar_pairs = None
    if args.validation_pairs:
        if len(args.validation_pairs) % 2 != 0:
            parser.error("--validation-pairs must contain an even number of image names (pairs)")
        
        known_similar_pairs = []
        for i in range(0, len(args.validation_pairs), 2):
            img1 = args.validation_pairs[i]
            img2 = args.validation_pairs[i + 1]
            known_similar_pairs.append((img1, img2))
        
        print(f"🎯 Custom validation pairs specified: {len(known_similar_pairs)} pairs")
        for i, (img1, img2) in enumerate(known_similar_pairs, 1):
            print(f"  {i}. {img1} ↔ {img2}")
    
    # Validate validation weight
    if not (0.0 <= args.validation_weight <= 1.0):
        parser.error(f"--validation-weight must be between 0.0 and 1.0, got {args.validation_weight}")
    
    # Process experiment mode arguments
    experiment_models = None
    experiment_resolutions = None
    
    if args.experiment_mode:
        print("🧪 Experiment mode enabled!")
        
        # Process experiment models
        if args.experiment_models:
            experiment_models = []
            for model_spec in args.experiment_models:
                if ':' in model_spec:
                    model, pretrained = model_spec.split(':', 1)
                    experiment_models.append((model, pretrained))
                else:
                    # Default to openai pretrained if not specified
                    experiment_models.append((model_spec, "openai"))
            print(f"  Models: {[f'{m}:{p}' for m, p in experiment_models]}")
        
        # Process experiment resolutions
        if args.experiment_resolutions:
            experiment_resolutions = args.experiment_resolutions
            print(f"  Resolutions: {experiment_resolutions}")
    
    # Create configuration
    config = PipelineConfig(
        input_images_dir=args.input_dir,
        output_base_dir=args.output_dir,
        target_image=args.target_image,
        skip_embeddings=args.skip_embeddings,
        experiment_mode=args.experiment_mode,  # NEW
        force_regenerate=args.force_regenerate,  # NEW
        known_similar_pairs=known_similar_pairs,  # NEW
        validation_weight=args.validation_weight,  # NEW
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        input_resolution=args.input_resolution,
        experiment_resolutions=experiment_resolutions,  # NEW
        experiment_models=experiment_models,  # NEW
        dataset_sample_percentage=args.dataset_sample,
        random_seed=args.random_seed,
        preprocessing_mode=args.preprocessing_mode,
        normalize_embeddings=not args.no_normalize_embeddings,
        save_format=args.save_format,
        embedding_precision=args.embedding_precision,
        execution_mode=args.mode,
        top_n_results=args.top_results,
        similarity_k=args.similarity_k,
        use_faiss=not args.no_faiss,
        use_gpu=not args.no_gpu,
        mixed_precision=mixed_precision,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout
    )
    
    # Run pipeline
    pipeline = StreamlinedImageClusteringPipeline(config)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 