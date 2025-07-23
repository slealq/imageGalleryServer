#!/usr/bin/env python3
"""
Unified Image Clustering System

A comprehensive, well-designed clustering solution that consolidates all clustering
functionality into a single, maintainable script using proper design patterns.

Features:
- Multiple clustering algorithms (K-Means, HDBSCAN, DBSCAN, Gaussian Mixture)
- Smart parameter selection (fast) vs comprehensive search
- GPU acceleration support (when available)
- Configurable execution modes
- Extensible architecture for adding new algorithms
- Quality evaluation and results management

Design Patterns:
- Strategy Pattern: Different clustering algorithms
- Factory Pattern: Algorithm creation
- Template Method: Common clustering workflow
- Configuration-driven execution
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import shutil
import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
import hdbscan

# Optional GPU support
GPU_AVAILABLE = False
try:
    import cupy as cp
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.preprocessing import normalize as cu_normalize
    # Test GPU availability
    cp.cuda.Device(0).compute_capability  # This will fail if no GPU
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available")
except (ImportError, RuntimeError, Exception) as e:
    # ImportError: cuML not installed
    # RuntimeError: CUDA runtime issues (like libcudart.so.12 not found)
    # Exception: Any other GPU-related issue
    GPU_AVAILABLE = False
    print(f"❌ GPU acceleration disabled: {str(e)[:100]}")

# Utilities
from tqdm import tqdm
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import psutil
import gc


# Configuration Classes
class ExecutionMode(Enum):
    """Execution modes with different speed/thoroughness tradeoffs."""
    DEBUG = "debug"         # 2-3 combinations for testing, ~5 minutes
    FAST = "fast"           # ~15-30 combinations, 30-60 minutes
    BALANCED = "balanced"   # ~50-80 combinations, 1-3 hours  
    COMPREHENSIVE = "comprehensive"  # ~200+ combinations, 4-12 hours
    GPU = "gpu"             # GPU-accelerated when available


@dataclass
class ClusteringConfig:
    """Configuration for clustering execution."""
    execution_mode: ExecutionMode = ExecutionMode.FAST
    algorithms: List[str] = None  # None = all available
    use_gpu: bool = True  # Use GPU if available
    parallel_cpu: bool = False  # Disabled by default for stability
    n_jobs: int = 1  # Single threaded by default
    max_combinations: Optional[int] = None
    target_cluster_size: int = 200  # Target images per cluster
    min_clusters: int = 10
    max_clusters: int = 300
    output_dir: str = "clustering_results"
    save_top_n: int = 10
    timeout_minutes: int = 10  # Reduced timeout for faster debugging
    checkpoint_interval: int = 1  # Save progress every experiment for debugging
    sample_images_per_cluster: int = 25  # Number of sample images to save per cluster
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["kmeans", "hdbscan", "dbscan", "gaussian_mixture"]
        if self.n_jobs == -1:
            self.n_jobs = 1  # Force single threading


@dataclass 
class ClusteringResult:
    """Results from a single clustering experiment."""
    algorithm: str
    parameters: Dict[str, Any]
    n_clusters: int
    n_points: int
    noise_ratio: float
    silhouette_score: float
    balance_score: float
    composite_score: float
    runtime: float
    labels: np.ndarray
    success: bool = True
    error: Optional[str] = None


# Data Management
class DataLoader:
    """Handles loading and preprocessing of embeddings and image paths."""
    
    def __init__(self, embeddings_file: str, image_paths_file: str):
        self.embeddings_file = Path(embeddings_file)
        self.image_paths_file = Path(image_paths_file)
        self._embeddings = None
        self._embeddings_normalized = None
        self._image_paths = None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load embeddings and image paths."""
        if self._embeddings is None:
            print(f"Loading embeddings from {self.embeddings_file}")
            self._embeddings = np.load(self.embeddings_file)
            
            with open(self.image_paths_file, 'r') as f:
                self._image_paths = json.load(f)
            
            print(f"Loaded {len(self._embeddings)} embeddings with {self._embeddings.shape[1]} dimensions")
            
            # Normalize for cosine similarity
            self._embeddings_normalized = normalize(self._embeddings, norm='l2')
            
        return self._embeddings, self._embeddings_normalized, self._image_paths
    
    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self.load_data()
        return self._embeddings
    
    @property
    def embeddings_normalized(self) -> np.ndarray:
        if self._embeddings_normalized is None:
            self.load_data()
        return self._embeddings_normalized
    
    @property
    def image_paths(self) -> List[str]:
        if self._image_paths is None:
            self.load_data()
        return self._image_paths


# Clustering Evaluation
class ClusteringEvaluator:
    """Evaluates clustering quality using multiple metrics."""
    
    @staticmethod
    def calculate_balance_score(labels: np.ndarray) -> float:
        """Calculate cluster balance score (0-1, higher is better)."""
        non_noise_labels = labels[labels != -1]
        if len(non_noise_labels) == 0:
            return 0.0
        
        unique_labels, counts = np.unique(non_noise_labels, return_counts=True)
        if len(counts) <= 1:
            return 0.0
        
        # Coefficient of variation (lower is better for balance)
        mean_size = np.mean(counts)
        std_size = np.std(counts)
        cv = std_size / mean_size if mean_size > 0 else float('inf')
        
        # Convert to score (0-1, higher is better)
        balance_score = max(0, 1 - cv/2)  # Divide by 2 for more lenient scoring
        return balance_score
    
    @staticmethod
    def calculate_silhouette_score(embeddings: np.ndarray, labels: np.ndarray, 
                                 sample_size: int = 5000) -> float:
        """Calculate silhouette score with sampling for efficiency."""
        non_noise_mask = labels != -1
        clean_embeddings = embeddings[non_noise_mask]
        clean_labels = labels[non_noise_mask]
        
        if len(clean_embeddings) == 0 or len(np.unique(clean_labels)) <= 1:
            return -1.0
        
        try:
            # Sample for efficiency on large datasets
            if len(clean_embeddings) > sample_size:
                sample_idx = np.random.choice(len(clean_embeddings), sample_size, replace=False)
                sample_embeddings = clean_embeddings[sample_idx]
                sample_labels = clean_labels[sample_idx]
            else:
                sample_embeddings = clean_embeddings
                sample_labels = clean_labels
            
            return silhouette_score(sample_embeddings, sample_labels)
        except Exception:
            return -1.0
    
    @classmethod
    def evaluate_clustering(cls, embeddings: np.ndarray, labels: np.ndarray, 
                          algorithm: str, parameters: Dict[str, Any], 
                          runtime: float) -> ClusteringResult:
        """Comprehensive clustering evaluation."""
        non_noise_mask = labels != -1
        clean_labels = labels[non_noise_mask]
        
        n_clusters = len(np.unique(clean_labels)) if len(clean_labels) > 0 else 0
        noise_ratio = np.sum(labels == -1) / len(labels)
        
        # Calculate metrics
        balance_score = cls.calculate_balance_score(labels)
        silhouette = cls.calculate_silhouette_score(embeddings, labels)
        
        # Composite score (0-1, higher is better)
        sil_norm = (silhouette + 1) / 2  # Convert -1,1 to 0,1
        noise_penalty = max(0, 1 - noise_ratio)
        
        composite_score = (
            0.3 * sil_norm +           # Silhouette quality
            0.4 * balance_score +      # Cluster balance
            0.3 * noise_penalty        # Noise penalty
        )
        
        return ClusteringResult(
            algorithm=algorithm,
            parameters=parameters,
            n_clusters=n_clusters,
            n_points=len(labels),
            noise_ratio=noise_ratio,
            silhouette_score=silhouette,
            balance_score=balance_score,
            composite_score=composite_score,
            runtime=runtime,
            labels=labels
        )


# Clustering Strategy Pattern
class ClusteringStrategy(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def get_parameter_combinations(self, data_size: int, config: ClusteringConfig) -> List[Dict[str, Any]]:
        """Generate parameter combinations for this algorithm."""
        pass
    
    @abstractmethod
    def cluster(self, embeddings: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Perform clustering with given parameters."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name."""
        pass
    
    @property
    def supports_gpu(self) -> bool:
        """Whether this algorithm supports GPU acceleration."""
        return False


class KMeansStrategy(ClusteringStrategy):
    """K-Means clustering strategy."""
    
    @property
    def name(self) -> str:
        return "kmeans"
    
    @property
    def supports_gpu(self) -> bool:
        return GPU_AVAILABLE
    
    def get_parameter_combinations(self, data_size: int, config: ClusteringConfig) -> List[Dict[str, Any]]:
        """Generate K-Means parameter combinations."""
        target_size = config.target_cluster_size
        min_clusters = max(config.min_clusters, data_size // 1000)
        max_clusters = min(config.max_clusters, data_size // 50)
        
        if config.execution_mode == ExecutionMode.DEBUG:
            # Minimal combinations for testing
            base_clusters = data_size // target_size
            cluster_counts = [max(min_clusters, base_clusters)]
            inits = ['k-means++']
        elif config.execution_mode == ExecutionMode.FAST:
            # Smart selection based on dataset size
            base_clusters = data_size // target_size
            cluster_counts = [
                max(min_clusters, int(base_clusters * 0.5)),
                max(min_clusters, int(base_clusters * 0.75)),
                max(min_clusters, base_clusters),
                min(max_clusters, int(base_clusters * 1.5)),
            ]
            cluster_counts = sorted(list(set(cluster_counts)))[:4]  # Limit to 4
            inits = ['k-means++']
        
        elif config.execution_mode == ExecutionMode.BALANCED:
            # More granular range for balanced mode
            cluster_counts = list(range(min_clusters, min(max_clusters, 200), 8))
            inits = ['k-means++', 'random']
        
        else:  # COMPREHENSIVE
            cluster_counts = list(range(min_clusters, max_clusters, 5))
            inits = ['k-means++', 'random']
        
        combinations = []
        for n_clusters in cluster_counts:
            for init in inits:
                combinations.append({
                    'n_clusters': n_clusters,
                    'init': init,
                    'n_init': 10,
                    'max_iter': 300,
                    'random_state': 42
                })
        
        return combinations
    
    def cluster(self, embeddings: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Perform K-Means clustering."""
        try:
            # Clear GPU memory if using GPU
            if GPU_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            
            # Use MiniBatch for large datasets
            if len(embeddings) > 20000:
                # Remove parameters not supported by MiniBatchKMeans
                mb_params = {k: v for k, v in parameters.items() if k in ['n_clusters', 'init', 'max_iter', 'random_state']}
                mb_params['batch_size'] = min(1000, len(embeddings) // 50)
                clusterer = MiniBatchKMeans(**mb_params)
            else:
                clusterer = KMeans(**parameters)
            
            return clusterer.fit_predict(embeddings)
        except Exception as e:
            # Fallback to CPU if GPU fails
            if "CUDA" in str(e) or "GPU" in str(e):
                print(f"GPU clustering failed, falling back to CPU: {e}")
                clusterer = KMeans(**parameters)
                return clusterer.fit_predict(embeddings)
            raise


class HDBSCANStrategy(ClusteringStrategy):
    """HDBSCAN clustering strategy."""
    
    @property
    def name(self) -> str:
        return "hdbscan"
    
    def get_parameter_combinations(self, data_size: int, config: ClusteringConfig) -> List[Dict[str, Any]]:
        """Generate HDBSCAN parameter combinations."""
        if config.execution_mode == ExecutionMode.DEBUG:
            # Single good combination for testing
            combinations = [
                {'min_cluster_size': 50, 'min_samples': 10, 'cluster_selection_epsilon': 0.1},
            ]
        elif config.execution_mode == ExecutionMode.FAST:
            # Key combinations that often work well
            combinations = [
                {'min_cluster_size': 50, 'min_samples': 10, 'cluster_selection_epsilon': 0.15},
                {'min_cluster_size': 75, 'min_samples': 15, 'cluster_selection_epsilon': 0.1},
                {'min_cluster_size': 100, 'min_samples': 20, 'cluster_selection_epsilon': 0.0},
                {'min_cluster_size': 30, 'min_samples': 5, 'cluster_selection_epsilon': 0.1},
            ]
        
        elif config.execution_mode == ExecutionMode.BALANCED:
            combinations = []
            for min_cluster_size in [30, 50, 75, 100, 150]:
                for min_samples in [5, 10, 15]:
                    for epsilon in [0.0, 0.1, 0.15]:
                        combinations.append({
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': epsilon
                        })
        
        else:  # COMPREHENSIVE
            combinations = []
            for min_cluster_size in [20, 30, 50, 75, 100, 150, 200]:
                for min_samples in [3, 5, 10, 15, 20]:
                    for epsilon in [0.0, 0.1, 0.15, 0.2, 0.25]:
                        combinations.append({
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': epsilon
                        })
        
        # Add common parameters
        for combo in combinations:
            combo.update({
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'
            })
        
        return combinations
    
    def cluster(self, embeddings: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Perform HDBSCAN clustering."""
        # Force aggressive garbage collection before heavy computation
        gc.collect()
        
        # For very large datasets, consider subsampling for parameter exploration
        if len(embeddings) > 50000:
            print(f"    📊 Large dataset ({len(embeddings)} points), using sample for parameter exploration")
            # Use a representative sample for parameter exploration
            sample_size = min(20000, len(embeddings))
            sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[sample_idx]
            
            # Cluster the sample
            clusterer = hdbscan.HDBSCAN(**parameters)
            sample_labels = clusterer.fit_predict(sample_embeddings)
            
            # Apply to full dataset if sample worked
            if len(np.unique(sample_labels[sample_labels != -1])) > 0:
                clusterer = hdbscan.HDBSCAN(**parameters)
                labels = clusterer.fit_predict(embeddings)
            else:
                # If sample failed, return all noise
                labels = np.array([-1] * len(embeddings))
        else:
            clusterer = hdbscan.HDBSCAN(**parameters)
            labels = clusterer.fit_predict(embeddings)
        
        # Cleanup
        del clusterer
        gc.collect()
        
        return labels


class DBSCANStrategy(ClusteringStrategy):
    """DBSCAN clustering strategy."""
    
    @property
    def name(self) -> str:
        return "dbscan"
    
    @property 
    def supports_gpu(self) -> bool:
        return GPU_AVAILABLE
    
    def get_parameter_combinations(self, data_size: int, config: ClusteringConfig) -> List[Dict[str, Any]]:
        """Generate DBSCAN parameter combinations."""
        if config.execution_mode == ExecutionMode.FAST:
            eps_values = [0.2, 0.25, 0.3]
            min_samples_values = [10, 15]
        elif config.execution_mode == ExecutionMode.BALANCED:
            eps_values = [0.15, 0.2, 0.25, 0.3, 0.35]
            min_samples_values = [5, 10, 15, 20]
        else:  # COMPREHENSIVE
            eps_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
            min_samples_values = [5, 10, 15, 20, 30]
        
        combinations = []
        for eps in eps_values:
            for min_samples in min_samples_values:
                combinations.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'metric': 'cosine'
                })
        
        return combinations
    
    def cluster(self, embeddings: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Perform DBSCAN clustering."""
        try:
            # Clear GPU memory if using GPU
            if GPU_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass
            
            clusterer = DBSCAN(**parameters)
            return clusterer.fit_predict(embeddings)
        except Exception as e:
            # Fallback handling
            if "CUDA" in str(e) or "GPU" in str(e):
                print(f"GPU clustering failed, falling back to CPU: {e}")
                clusterer = DBSCAN(**parameters)
                return clusterer.fit_predict(embeddings)
            raise


class GaussianMixtureStrategy(ClusteringStrategy):
    """Gaussian Mixture Model clustering strategy."""
    
    @property
    def name(self) -> str:
        return "gaussian_mixture"
    
    def get_parameter_combinations(self, data_size: int, config: ClusteringConfig) -> List[Dict[str, Any]]:
        """Generate Gaussian Mixture parameter combinations."""
        target_size = config.target_cluster_size
        base_components = max(10, data_size // target_size)
        
        if config.execution_mode == ExecutionMode.FAST:
            n_components_list = [
                max(10, int(base_components * 0.75)),
                base_components,
                min(100, int(base_components * 1.25))
            ]
            covariance_types = ['diag']
        elif config.execution_mode == ExecutionMode.BALANCED:
            n_components_list = list(range(10, min(120, base_components * 2), 15))
            covariance_types = ['diag', 'full']
        else:  # COMPREHENSIVE
            n_components_list = list(range(10, min(200, base_components * 3), 10))
            covariance_types = ['diag', 'full', 'tied']
        
        combinations = []
        for n_components in n_components_list:
            for cov_type in covariance_types:
                combinations.append({
                    'n_components': n_components,
                    'covariance_type': cov_type,
                    'max_iter': 100,
                    'random_state': 42
                })
        
        return combinations
    
    def cluster(self, embeddings: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Perform Gaussian Mixture clustering."""
        # Force garbage collection before heavy computation
        gc.collect()
        
        try:
            # Reduce iterations for large datasets to prevent memory issues
            if len(embeddings) > 30000:
                parameters = parameters.copy()
                parameters['max_iter'] = min(parameters.get('max_iter', 100), 50)
                print(f"    📊 Large dataset ({len(embeddings)} points), reducing max_iter to {parameters['max_iter']}")
            
            clusterer = GaussianMixture(**parameters)
            clusterer.fit(embeddings)
            labels = clusterer.predict(embeddings)
            
            # Check for convergence
            if not clusterer.converged_:
                print(f"    ⚠️  Gaussian Mixture did not converge")
            
            # Cleanup
            del clusterer
            gc.collect()
            
            return labels
            
        except Exception as e:
            print(f"    ❌ Gaussian Mixture failed: {e}")
            # Return random assignment as fallback
            n_components = parameters.get('n_components', 10)
            return np.random.randint(0, n_components, size=len(embeddings))


# Strategy Factory
class ClusteringStrategyFactory:
    """Factory for creating clustering strategies."""
    
    _strategies = {
        'kmeans': KMeansStrategy,
        'hdbscan': HDBSCANStrategy,
        'dbscan': DBSCANStrategy,
        'gaussian_mixture': GaussianMixtureStrategy,
    }
    
    @classmethod
    def create_strategy(cls, algorithm_name: str) -> ClusteringStrategy:
        """Create a clustering strategy instance."""
        if algorithm_name not in cls._strategies:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        return cls._strategies[algorithm_name]()
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithms."""
        return list(cls._strategies.keys())


# Main Clustering Manager
class UnifiedClusteringManager:
    """Main clustering manager that orchestrates the entire process."""
    
    def __init__(self, embeddings_file: str, image_paths_file: str, config: ClusteringConfig):
        self.data_loader = DataLoader(embeddings_file, image_paths_file)
        self.config = config
        self.evaluator = ClusteringEvaluator()
        self.strategy_factory = ClusteringStrategyFactory()
        self.results: List[ClusteringResult] = []
        self.completed_experiments = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Create output directory early for checkpoints
        Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def save_checkpoint(self) -> None:
        """Save current progress as checkpoint."""
        if not self.results:
            return
            
        checkpoint_file = Path(self.config.output_dir) / "checkpoint.pkl"
        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump({
                    'results': self.results,
                    'completed_experiments': self.completed_experiments,
                    'config': self.config
                }, f)
            print(f"📄 Checkpoint saved: {len(self.results)} results")
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> bool:
        """Load previous checkpoint if exists."""
        checkpoint_file = Path(self.config.output_dir) / "checkpoint.pkl"
        if not checkpoint_file.exists():
            return False
        
        try:
            with open(checkpoint_file, "rb") as f:
                data = pickle.load(f)
            
            self.results = data.get('results', [])
            self.completed_experiments = data.get('completed_experiments', 0)
            
            if self.results:
                print(f"📄 Resumed from checkpoint: {len(self.results)} previous results")
                return True
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
        
        return False
    
    def generate_all_parameter_combinations(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate all parameter combinations for selected algorithms."""
        embeddings, _, _ = self.data_loader.load_data()
        data_size = len(embeddings)
        
        all_combinations = []
        
        for algorithm_name in self.config.algorithms:
            if algorithm_name not in self.strategy_factory.get_available_algorithms():
                self.logger.warning(f"Skipping unknown algorithm: {algorithm_name}")
                continue
            
            strategy = self.strategy_factory.create_strategy(algorithm_name)
            combinations = strategy.get_parameter_combinations(data_size, self.config)
            
            for params in combinations:
                all_combinations.append((algorithm_name, params))
        
        if self.config.max_combinations:
            if len(all_combinations) > self.config.max_combinations:
                import random
                all_combinations = random.sample(all_combinations, self.config.max_combinations)
        
        self.logger.info(f"Generated {len(all_combinations)} parameter combinations")
        return all_combinations
    
    def run_single_clustering(self, algorithm_name: str, parameters: Dict[str, Any], 
                             embeddings_normalized: np.ndarray = None) -> Optional[ClusteringResult]:
        """Run a single clustering experiment with pre-loaded embeddings."""
        try:
            print(f"🔬 Starting {algorithm_name} with params: {parameters}")
            start_time = time.time()
            
            strategy = self.strategy_factory.create_strategy(algorithm_name)
            
            # Use pre-loaded embeddings if provided, otherwise load
            if embeddings_normalized is None:
                _, embeddings_normalized, _ = self.data_loader.load_data()
            
            # Force garbage collection and clear GPU memory before each experiment
            gc.collect()
            if GPU_AVAILABLE:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except:
                    pass
            
            print(f"  ⚡ Running clustering on {len(embeddings_normalized)} embeddings...")
            labels = strategy.cluster(embeddings_normalized, parameters)
            runtime = time.time() - start_time
            
            print(f"  ✅ Completed in {runtime:.1f}s, found {len(np.unique(labels[labels != -1]))} clusters")
            
            result = self.evaluator.evaluate_clustering(
                embeddings_normalized, labels, algorithm_name, parameters, runtime
            )
            
            print(f"  📊 Score: {result.composite_score:.3f}, Noise: {result.noise_ratio:.1%}")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Error in {algorithm_name}: {error_msg}")
            self.logger.error(f"Error in {algorithm_name} with {parameters}: {e}")
            return ClusteringResult(
                algorithm=algorithm_name,
                parameters=parameters,
                n_clusters=0,
                n_points=len(self.data_loader.embeddings) if self.data_loader._embeddings is not None else 0,
                noise_ratio=1.0,
                silhouette_score=-1.0,
                balance_score=0.0,
                composite_score=0.0,
                runtime=0.0,
                labels=np.array([-1] * (len(self.data_loader.embeddings) if self.data_loader._embeddings is not None else 1000)),
                success=False,
                error=error_msg
            )
    
    def run_clustering_experiments(self) -> List[ClusteringResult]:
        """Run all clustering experiments with improved resource management."""
        # Try to load checkpoint first
        checkpoint_loaded = self.load_checkpoint()
        
        combinations = self.generate_all_parameter_combinations()
        
        print(f"\n🚀 Starting clustering with {self.config.execution_mode.value} mode")
        print(f"📊 Testing {len(combinations)} parameter combinations")
        print(f"💻 GPU support: {'✅ Available' if GPU_AVAILABLE and self.config.use_gpu else '❌ CPU only'}")
        print(f"🔄 Parallelization: {'✅ Enabled' if self.config.parallel_cpu else '❌ Disabled (Sequential)'}")
        
        if checkpoint_loaded:
            print(f"📄 Resumed from checkpoint with {len(self.results)} previous results")
        
        # Pre-load embeddings once to avoid repeated loading
        print("📥 Pre-loading embeddings...")
        _, embeddings_normalized, _ = self.data_loader.load_data()
        embeddings_size_gb = embeddings_normalized.nbytes / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"🧠 Available memory: {available_memory_gb:.1f}GB, Embeddings: {embeddings_size_gb:.1f}GB")
        print(f"⏰ Timeout: {self.config.timeout_minutes} minutes per experiment")
        print("=" * 80)
        
        start_time = time.time()
        
        # Sequential execution with detailed progress tracking
        successful_count = 0
        failed_count = 0
        skipped_count = len(self.results) if checkpoint_loaded else 0
        
        # Skip already completed experiments if resuming from checkpoint
        start_index = len(self.results) if checkpoint_loaded else 0
        remaining_combinations = combinations[start_index:]
        
        for i, (algorithm, params) in enumerate(remaining_combinations):
            experiment_num = start_index + i + 1
            print(f"\n[{experiment_num}/{len(combinations)}] Running {algorithm}")
            print("-" * 60)
            
            try:
                # Set timeout for individual experiment
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Experiment timed out after {self.config.timeout_minutes} minutes")
                
                # Set signal handler for timeout (Unix only)
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.config.timeout_minutes * 60)
                
                result = self.run_single_clustering(algorithm, params, embeddings_normalized)
                
                # Clear timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                
                if result and result.success and result.composite_score > 0.1:
                    self.results.append(result)
                    successful_count += 1
                    print(f"  ✅ Added to results (Total successful: {successful_count})")
                else:
                    failed_count += 1
                    print(f"  ⚠️  Low quality result, skipped (Failed: {failed_count})")
                
                self.completed_experiments = experiment_num
                
                # Save checkpoint after every experiment for debugging
                if experiment_num % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()
                    print(f"  💾 Checkpoint saved")
                
                # Memory cleanup after each experiment
                gc.collect()
                
                # Print memory usage
                current_memory = psutil.virtual_memory()
                print(f"  🧠 Memory usage: {(current_memory.total - current_memory.available) / (1024**3):.1f}GB / {current_memory.total / (1024**3):.1f}GB")
                
            except TimeoutError as e:
                failed_count += 1
                print(f"  ⏰ Timeout: {e}")
                # Clear timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                continue
                
            except KeyboardInterrupt:
                print(f"\n⏹️  Interrupted by user at experiment {experiment_num}")
                break
                
            except Exception as e:
                failed_count += 1
                print(f"  ❌ Unexpected error: {e}")
                self.logger.error(f"Unexpected error in experiment {experiment_num}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        
        # Final checkpoint
        self.save_checkpoint()
        
        print("\n" + "=" * 80)
        print("🏁 CLUSTERING COMPLETED")
        print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
        print(f"✅ Successful experiments: {successful_count}")
        print(f"❌ Failed experiments: {failed_count}")
        print(f"📄 Resumed experiments: {skipped_count}")
        print(f"📊 Total valid results: {len(self.results)}")
        
        self.logger.info(f"Completed clustering: {len(self.results)} successful experiments in {elapsed_time:.1f} seconds")
        
        return self.results
    
    def get_top_results(self, n_top: int = None) -> List[ClusteringResult]:
        """Get top clustering results sorted by composite score."""
        if n_top is None:
            n_top = self.config.save_top_n
        
        # Filter and sort results
        valid_results = [
            r for r in self.results
            if (r.n_clusters >= self.config.min_clusters and 
                r.n_clusters <= self.config.max_clusters and
                r.composite_score > 0.2)
        ]
        
        sorted_results = sorted(valid_results, key=lambda x: x.composite_score, reverse=True)
        return sorted_results[:n_top]
    
    def save_cluster_sample_images(self, result: ClusteringResult, result_dir: Path, 
                                  max_samples: int = 25) -> None:
        """Save sample images for each cluster."""
        try:
            # Set random seed for reproducible sampling
            random.seed(42)
            
            # Get image paths
            image_paths = self.data_loader.image_paths
            
            if len(image_paths) != len(result.labels):
                self.logger.warning(f"Mismatch: {len(image_paths)} paths vs {len(result.labels)} labels")
                return
            
            # Create samples directory
            samples_dir = result_dir / "cluster_samples"
            samples_dir.mkdir(exist_ok=True)
            
            # Group images by cluster
            cluster_images = {}
            for img_path, label in zip(image_paths, result.labels):
                if label == -1:  # Skip noise points
                    continue
                if label not in cluster_images:
                    cluster_images[label] = []
                cluster_images[label].append(img_path)
            
            # Sample and copy images for each cluster
            total_copied = 0
            cluster_stats = []
            
            for cluster_id, cluster_img_paths in cluster_images.items():
                cluster_dir = samples_dir / f"cluster_{cluster_id:03d}"
                cluster_dir.mkdir(exist_ok=True)
                
                # Sample up to max_samples images
                sample_size = min(max_samples, len(cluster_img_paths))
                sampled_paths = random.sample(cluster_img_paths, sample_size)
                
                copied_count = 0
                for j, img_path in enumerate(sampled_paths):
                    src_path = Path(img_path)
                    if src_path.exists():
                        # Create a clean filename
                        file_extension = src_path.suffix
                        dst_filename = f"sample_{j:03d}{file_extension}"
                        dst_path = cluster_dir / dst_filename
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                            copied_count += 1
                        except Exception as e:
                            self.logger.warning(f"Failed to copy {src_path}: {e}")
                    else:
                        self.logger.warning(f"Source file not found: {img_path}")
                
                total_copied += copied_count
                cluster_stats.append({
                    "cluster_id": int(cluster_id),
                    "total_images": len(cluster_img_paths),
                    "sampled_images": copied_count
                })
                
                print(f"    📁 Cluster {cluster_id}: {copied_count}/{len(cluster_img_paths)} images")
            
            # Save cluster statistics
            with open(samples_dir / "cluster_statistics.json", "w") as f:
                json.dump({
                    "total_clusters": len(cluster_images),
                    "total_images_copied": total_copied,
                    "max_samples_per_cluster": max_samples,
                    "clusters": cluster_stats
                }, f, indent=2)
                
            print(f"    ✅ Saved {total_copied} sample images across {len(cluster_images)} clusters")
            
        except Exception as e:
            self.logger.error(f"Failed to save sample images: {e}")
            print(f"    ❌ Failed to save sample images: {e}")

    def save_results(self) -> None:
        """Save clustering results to disk."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(exist_ok=True)
        
        top_results = self.get_top_results()
        
        if not top_results:
            self.logger.warning("No valid clustering results to save!")
            return
        
        # Save individual results
        for i, result in enumerate(top_results, 1):
            result_dir = output_path / f"rank_{i:02d}_{result.algorithm}"
            result_dir.mkdir(exist_ok=True)
            
            print(f"\n💾 Saving result #{i}: {result.algorithm}")
            print(f"    📊 Score: {result.composite_score:.3f}, Clusters: {result.n_clusters}, Noise: {result.noise_ratio:.1%}")
            
            # Save cluster labels
            np.save(result_dir / "cluster_labels.npy", result.labels)
            
            # Save image-to-cluster mapping for similarity search
            image_paths = self.data_loader.image_paths
            image_cluster_mapping = {}
            cluster_image_mapping = {}
            
            for img_path, label in zip(image_paths, result.labels):
                # Convert absolute paths to just filenames for easier matching
                img_name = Path(img_path).name
                image_cluster_mapping[img_name] = int(label) if label != -1 else -1
                
                # Also create reverse mapping (cluster -> list of images)
                if label != -1:
                    label_int = int(label)
                    if label_int not in cluster_image_mapping:
                        cluster_image_mapping[label_int] = []
                    cluster_image_mapping[label_int].append(img_name)
            
            # Save mappings for similarity search
            with open(result_dir / "image_to_cluster.json", "w") as f:
                json.dump(image_cluster_mapping, f, indent=2)
            
            with open(result_dir / "cluster_to_images.json", "w") as f:
                json.dump(cluster_image_mapping, f, indent=2)
            
            # Save result info (without labels for JSON serialization)
            result_dict = asdict(result)
            result_dict.pop('labels')
            
            with open(result_dir / "clustering_info.json", "w") as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            # Save sample images for each cluster (DISABLED for speed)
            # print(f"    🖼️  Saving sample images...")
            # self.save_cluster_sample_images(result, result_dir, self.config.sample_images_per_cluster)
            print(f"    ⚡ Sample image saving disabled for faster processing")
        
        # Save summary
        summary = {
            "config": asdict(self.config),
            "gpu_available": GPU_AVAILABLE,
            "total_experiments": len(self.results),
            "successful_experiments": len(top_results),
            "embeddings_file": str(self.data_loader.embeddings_file),
            "image_paths_file": str(self.data_loader.image_paths_file),
            "sample_images_per_cluster": self.config.sample_images_per_cluster,
            "top_results": [
                {
                    "rank": i,
                    "algorithm": r.algorithm,
                    "n_clusters": r.n_clusters,
                    "noise_ratio": r.noise_ratio,
                    "composite_score": r.composite_score,
                    "runtime": r.runtime,
                    "parameters": r.parameters
                }
                for i, r in enumerate(top_results, 1)
            ]
        }
        
        with open(output_path / "clustering_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save all results for analysis
        with open(output_path / "all_results.pkl", "wb") as f:
            pickle.dump(self.results, f)
        
        print(f"\n🎉 Results saved to {output_path}")
        print("\nTop Results:")
        print("=" * 80)
        for i, result in enumerate(top_results[:5], 1):
            print(f"{i:2d}. {result.algorithm:<18} | "
                  f"Clusters: {result.n_clusters:3d} | "
                  f"Noise: {result.noise_ratio:5.1%} | "
                  f"Score: {result.composite_score:.3f} | "
                  f"Time: {result.runtime:5.1f}s")


# Auto-detection utilities
def find_existing_embeddings() -> Tuple[Optional[str], Optional[str]]:
    """Auto-detect existing embeddings files."""
    search_paths = [
        Path.cwd() / "advanced_clustering_results",
        Path.cwd().parent / "test_output_3", 
        Path("/mnt/c/Users/stuar/Downloads/test_output_3"),
        Path("/mnt/c/Users/stuar/Downloads/test_output_2"),
        Path.cwd(),
    ]
    
    for search_path in search_paths:
        embeddings_file = search_path / "embeddings.npy"
        paths_file = search_path / "image_paths.json"
        
        if embeddings_file.exists() and paths_file.exists():
            return str(embeddings_file), str(paths_file)
    
    return None, None


# Main CLI Interface
def main():
    parser = argparse.ArgumentParser(
        description="Unified Image Clustering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Modes:
  debug        - Minimal testing (2-3 combinations, ~5 minutes)
  fast         - Quick results (15-30 combinations, 30-60 minutes)
  balanced     - Good coverage (50-80 combinations, 1-3 hours)  
  comprehensive- Thorough search (200+ combinations, 4+ hours)
  gpu          - GPU-accelerated when available

Examples:
  python unified_clustering.py --mode debug
  python unified_clustering.py --mode fast
  python unified_clustering.py embeddings.npy image_paths.json --mode balanced
  python unified_clustering.py --mode gpu --algorithms kmeans dbscan
  python unified_clustering.py --mode fast --sample-images 50
        """
    )
    
    # Positional arguments (optional with auto-detection)
    parser.add_argument("embeddings_file", nargs='?', help="Path to embeddings.npy file")
    parser.add_argument("image_paths_file", nargs='?', help="Path to image_paths.json file")
    
    # Execution mode
    parser.add_argument("--mode", choices=['debug', 'fast', 'balanced', 'comprehensive', 'gpu'], 
                       default='debug', help="Execution mode (default: debug)")
    
    # Algorithm selection
    parser.add_argument("--algorithms", nargs='+', 
                       choices=['kmeans', 'hdbscan', 'dbscan', 'gaussian_mixture'],
                       help="Algorithms to use (default: all)")
    
    # Performance options
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing (disabled by default)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of CPU cores to use")
    parser.add_argument("--max-combinations", type=int, help="Maximum parameter combinations to test")
    
    # Clustering parameters
    parser.add_argument("--target-cluster-size", type=int, default=200,
                       help="Target number of images per cluster")
    parser.add_argument("--min-clusters", type=int, default=10,
                       help="Minimum number of clusters")
    parser.add_argument("--max-clusters", type=int, default=300,
                       help="Maximum number of clusters")
    
    # Output options
    parser.add_argument("--output-dir", default="clustering_results",
                       help="Output directory for results")
    parser.add_argument("--save-top-n", type=int, default=10,
                       help="Number of top results to save")
    parser.add_argument("--sample-images", type=int, default=25,
                       help="Number of sample images to save per cluster")
    
    # Resource management
    parser.add_argument("--timeout-minutes", type=int, default=15,
                       help="Timeout in minutes per clustering experiment")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                       help="Save checkpoint every N experiments")
    
    args = parser.parse_args()
    
    # Handle file arguments with auto-detection
    if args.embeddings_file and args.image_paths_file:
        embeddings_file = args.embeddings_file
        image_paths_file = args.image_paths_file
    else:
        print("🔍 Auto-detecting embeddings files...")
        embeddings_file, image_paths_file = find_existing_embeddings()
        
        if not embeddings_file or not image_paths_file:
            print("❌ Could not find embeddings files!")
            print("Please specify embeddings_file and image_paths_file arguments")
            print("Or ensure files exist in standard locations")
            sys.exit(1)
        
        print(f"✅ Found embeddings: {embeddings_file}")
        print(f"✅ Found image paths: {image_paths_file}")
    
    # Create configuration
    config = ClusteringConfig(
        execution_mode=ExecutionMode(args.mode),
        algorithms=args.algorithms,
        use_gpu=not args.no_gpu and GPU_AVAILABLE,
        parallel_cpu=args.parallel,
        n_jobs=args.n_jobs,
        max_combinations=args.max_combinations,
        target_cluster_size=args.target_cluster_size,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        output_dir=args.output_dir,
        save_top_n=args.save_top_n,
        timeout_minutes=args.timeout_minutes,
        checkpoint_interval=args.checkpoint_interval,
        sample_images_per_cluster=args.sample_images
    )
    
    print("🎯 Unified Image Clustering System")
    print("=" * 50)
    print(f"Mode: {config.execution_mode.value}")
    print(f"Algorithms: {', '.join(config.algorithms)}")
    print(f"GPU: {'✅ Enabled' if config.use_gpu else '❌ Disabled'}")
    print(f"Parallel: {'✅ Enabled' if config.parallel_cpu else '❌ Disabled'}")
    print(f"Timeout: {config.timeout_minutes} minutes per experiment")
    print(f"Sample images per cluster: {config.sample_images_per_cluster}")
    print("=" * 50)
    
    # Run clustering
    try:
        manager = UnifiedClusteringManager(embeddings_file, image_paths_file, config)
        
        start_time = time.time()
        results = manager.run_clustering_experiments()
        elapsed_time = time.time() - start_time
        
        if results:
            manager.save_results()
            
            print(f"\n✅ Clustering completed successfully!")
            print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
            print(f"🔬 Experiments: {len(results)} successful")
            print(f"📁 Results: {config.output_dir}")
        else:
            print("❌ No successful clustering results found")
            print("Try adjusting parameters or using a different mode")
    
    except KeyboardInterrupt:
        print("\n⏹️  Clustering interrupted by user")
        print("✅ Progress has been saved in checkpoint file")
    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 