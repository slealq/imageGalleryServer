#!/usr/bin/env python3
"""
Image Clustering Service

Clusters images based on similarity using pose data, embeddings, or combined features.
Supports multiple clustering algorithms including FAISS-based similarity clustering and provides 
detailed cluster analysis with examples that include pose estimation overlays.

This service reads configuration from a YAML file, loads data from pose extraction and/or
embedding services, performs similarity-based clustering using FAISS, and outputs
cluster examples and statistics for analysis.
"""

import json
import shutil
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
from PIL import Image
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Import common services
from common_fs_service import (
    ResultsManager, 
    discover_images, 
    validate_config, 
    create_service_runner,
    logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Try to import clustering libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available - FAISS-based clustering will be limited")
    FAISS_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn not available - some clustering methods will be unavailable")
    SKLEARN_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    logger.warning("HDBSCAN not available - HDBSCAN clustering will be unavailable")
    HDBSCAN_AVAILABLE = False


class ClusteringService:
    """
    Service for clustering images based on pose similarity, embedding similarity, or combined features.
    
    This service provides a clean interface for:
    - Loading and validating clustering configurations
    - Loading pose data and embeddings from previous pipeline steps
    - Creating feature vectors for clustering (pose, embedding, or combined)
    - Performing similarity-based clustering using FAISS or traditional algorithms
    - Generating cluster examples with pose estimation overlays
    - Saving results using standardized output structure
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        """
        Initialize the Clustering Service.
        
        Args:
            results_manager: Optional ResultsManager instance. If not provided,
                           will be created during run execution.
        """
        self.results_manager = results_manager
        self.config = None
        self.embeddings = None
        self.embedding_paths = None
        self.pose_data = None
        self.pose_paths = None
        # NEW: Canny edge feature data
        self.feature_data = None
        self.feature_paths = None
        self.canny_features = None
        self.feature_vectors = None
        self.image_paths = None
        self.clusters = None
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate configuration for clustering.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing validated configuration parameters
        """
        # Use common config validation service
        config = validate_config(config_path, 'clustering')
        
        # Set defaults for optional parameters
        config.setdefault('clustering_method', 'faiss_similarity')  # 'faiss_similarity', 'kmeans', 'hdbscan', 'dbscan'
        config.setdefault('feature_type', 'combined')  # 'pose', 'embedding', 'combined', 'canny', 'canny_embedding'
        config.setdefault('n_clusters', 'auto')  # number or 'auto'
        config.setdefault('pose_weight', 0.5)
        config.setdefault('embedding_weight', 0.5)
        # NEW: Canny edge feature weight
        config.setdefault('canny_weight', 0.4)
        config.setdefault('pose_confidence_threshold', 0.2)
        config.setdefault('max_examples_per_cluster', 50)
        config.setdefault('min_cluster_size', 5)
        config.setdefault('copy_examples', True)
        config.setdefault('generate_visualizations', True)
        config.setdefault('distance_metric', 'euclidean')
        config.setdefault('generate_pose_overlays', True)  # New: Generate pose overlays for cluster examples
        config.setdefault('generate_cluster_collages', True)  # New: Generate collages showing all cluster examples
        
        # FAISS similarity clustering defaults
        config.setdefault('faiss_index_type', 'IndexFlatIP')  # 'IndexFlatIP', 'IndexFlatL2', 'IndexIVFFlat', 'IndexHNSW'
        config.setdefault('similarity_threshold', 0.7)  # Minimum similarity for clustering
        config.setdefault('max_cluster_search_depth', 100)  # Maximum number of similar images to consider per seed
        config.setdefault('use_iterative_clustering', True)  # Use iterative approach to build clusters
        
        # Clustering method specific defaults
        if config['clustering_method'] == 'kmeans':
            config.setdefault('kmeans_max_iter', 300)
            config.setdefault('kmeans_random_state', 42)
        elif config['clustering_method'] == 'hdbscan':
            config.setdefault('hdbscan_min_cluster_size', 10)
            config.setdefault('hdbscan_min_samples', 5)
        elif config['clustering_method'] == 'dbscan':
            config.setdefault('dbscan_eps', 0.5)
            config.setdefault('dbscan_min_samples', 5)
        
        # Convert paths to Path objects
        config['output_path'] = Path(config['output_path'])
        
        if config.get('embeddings_path'):
            config['embeddings_path'] = Path(config['embeddings_path'])
        if config.get('pose_data_path'):
            config['pose_data_path'] = Path(config['pose_data_path'])
        # NEW: Feature extraction data path
        if config.get('feature_data_path'):
            config['feature_data_path'] = Path(config['feature_data_path'])
        
        # Validate feature type requirements
        if config['feature_type'] in ['embedding', 'combined', 'canny_embedding'] and not config.get('embeddings_path'):
            raise ValueError("embeddings_path required when feature_type is 'embedding', 'combined', or 'canny_embedding'")
        if config['feature_type'] in ['pose', 'combined'] and not config.get('pose_data_path'):
            raise ValueError("pose_data_path required when feature_type is 'pose' or 'combined'")
        # NEW: Validate canny feature requirements
        if config['feature_type'] in ['canny', 'canny_embedding'] and not config.get('feature_data_path'):
            raise ValueError("feature_data_path required when feature_type is 'canny' or 'canny_embedding'")
        
        # Validate weights for combined features
        if config['feature_type'] == 'combined':
            pose_weight = config['pose_weight']
            embedding_weight = config['embedding_weight']
            if abs((pose_weight + embedding_weight) - 1.0) > 0.01:
                logger.warning(f"pose_weight ({pose_weight}) + embedding_weight ({embedding_weight}) != 1.0")
        # NEW: Validate weights for canny+embedding features
        elif config['feature_type'] == 'canny_embedding':
            canny_weight = config['canny_weight']
            embedding_weight = config['embedding_weight']
            if abs((canny_weight + embedding_weight) - 1.0) > 0.01:
                logger.warning(f"canny_weight ({canny_weight}) + embedding_weight ({embedding_weight}) != 1.0")
        
        self.config = config
        logger.info(f"Configuration loaded and validated successfully from {config_path}")
        return config
    
    def load_embeddings(self, embeddings_path: Path) -> Tuple[np.ndarray, List[str]]:
        """
        Load pre-computed embeddings from numpy file.
        
        Args:
            embeddings_path: Path to the embeddings file (.npz)
            
        Returns:
            Tuple of (embeddings_array, image_paths)
        """
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
        # Load embeddings
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        image_paths = data['image_paths'].tolist()
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.embeddings = embeddings
        self.embedding_paths = image_paths
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_path}")
        return embeddings, image_paths
    
    def load_pose_data(self, pose_data_path: Path) -> Tuple[List[Dict], List[str]]:
        """
        Load pose data from pose extraction service results.
        
        Args:
            pose_data_path: Path to pose extraction results directory or pose_data.json file
            
        Returns:
            Tuple of (pose_data_list, pose_image_paths)
        """
        if not pose_data_path.exists():
            raise FileNotFoundError(f"Pose data path not found: {pose_data_path}")
        
        # Handle both directory and direct file paths
        if pose_data_path.is_dir():
            pose_json_path = pose_data_path / "pose_data.json"
        else:
            pose_json_path = pose_data_path
        
        if not pose_json_path.exists():
            raise FileNotFoundError(f"Pose data file not found: {pose_json_path}")
        
        # Load pose data
        with open(pose_json_path, 'r') as f:
            pose_data = json.load(f)
        
        # Filter poses by confidence threshold
        confidence_threshold = self.config.get('pose_confidence_threshold', 0.2)
        valid_poses = []
        valid_paths = []
        
        for pose_item in pose_data:
            if pose_item.get('pose_confidence', 0.0) >= confidence_threshold:
                valid_poses.append(pose_item)
                valid_paths.append(pose_item['image_path'])
        
        self.pose_data = valid_poses
        self.pose_paths = valid_paths
        
        logger.info(f"Loaded {len(valid_poses)} valid poses from {len(pose_data)} total poses")
        logger.info(f"Filtered poses with confidence >= {confidence_threshold}")
        
        return valid_poses, valid_paths
    
    def load_feature_data(self, feature_data_path: Path) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Load feature extraction data from feature extraction service results.
        
        Args:
            feature_data_path: Path to feature extraction results directory or feature_vectors.npz file
            
        Returns:
            Tuple of (feature_data_dict, feature_image_paths)
        """
        if not feature_data_path.exists():
            raise FileNotFoundError(f"Feature data path not found: {feature_data_path}")
        
        # Handle both directory and direct file paths
        if feature_data_path.is_dir():
            feature_npz_path = feature_data_path / "feature_vectors.npz"
        else:
            feature_npz_path = feature_data_path
        
        if not feature_npz_path.exists():
            raise FileNotFoundError(f"Feature vectors file not found: {feature_npz_path}")
        
        # Load feature data
        data = np.load(feature_npz_path)
        
        # Extract feature methods and image paths
        feature_methods = data.get('feature_methods', []).tolist()
        image_paths = data.get('image_paths', []).tolist()
        
        # Extract feature vectors for each method
        feature_data = {}
        for method in feature_methods:
            vector_key = f'{method}_vectors'
            if vector_key in data:
                feature_data[method] = data[vector_key]
        
        self.feature_data = feature_data
        self.feature_paths = image_paths
        
        logger.info(f"Loaded feature data for {len(image_paths)} images")
        logger.info(f"Available feature methods: {feature_methods}")
        if 'canny' in feature_data:
            logger.info(f"Canny features shape: {feature_data['canny'].shape}")
        
        return feature_data, image_paths
    
    def extract_canny_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract Canny edge feature vector for a specific image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Canny feature vector or None if not found
        """
        if self.feature_data is None or 'canny' not in self.feature_data:
            return None
        
        # Normalize paths for comparison
        target_path = Path(image_path).resolve()
        
        for i, feature_path in enumerate(self.feature_paths):
            if Path(feature_path).resolve() == target_path:
                return self.feature_data['canny'][i]
        
        return None
    
    def extract_pose_features(self, pose_item: Dict) -> Optional[np.ndarray]:
        """
        Extract feature vector from pose data.
        
        Args:
            pose_item: Pose data dictionary
            
        Returns:
            Feature vector or None if pose is invalid
        """
        # Check if pose data has the new format (with 'landmarks') or old format (with 'body_keypoints')
        if pose_item.get('landmarks'):
            # New format from pose extraction service
            landmarks = np.array(pose_item['landmarks'])
            
            # Reshape to (n_joints, 2) format (landmarks are already flattened x,y pairs)
            landmarks = landmarks.reshape(-1, 2)
            
            # Flatten to 1D feature vector (landmarks are already normalized)
            feature_vector = landmarks.flatten()
            
        elif pose_item.get('body_keypoints') and len(pose_item['body_keypoints']) > 0:
            # Legacy format compatibility
            keypoints = np.array(pose_item['body_keypoints'][0]['keypoints'])
            
            # Reshape to (n_joints, 3) format
            keypoints = keypoints.reshape(-1, 3)
            
            # Extract x, y coordinates (ignore confidence for feature vector)
            xy_coords = keypoints[:, :2]
            
            # Normalize by image size (assuming 368x368 input to OpenPose)
            normalized_coords = xy_coords / 368.0
            
            # Flatten to 1D feature vector
            feature_vector = normalized_coords.flatten()
        else:
            return None
        
        return feature_vector
    
    def create_combined_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create combined feature vectors from pose and embedding data.
        
        Returns:
            Tuple of (feature_vectors, common_image_paths)
        """
        # Find common images between pose and embedding data
        pose_paths_set = set(Path(p).resolve() for p in self.pose_paths)
        embedding_paths_set = set(Path(p).resolve() for p in self.embedding_paths)
        common_paths = pose_paths_set.intersection(embedding_paths_set)
        
        if len(common_paths) == 0:
            raise ValueError("No common images found between pose and embedding data")
        
        logger.info(f"Found {len(common_paths)} common images for feature combination")
        
        # Create mappings
        pose_path_to_idx = {Path(p).resolve(): i for i, p in enumerate(self.pose_paths)}
        embedding_path_to_idx = {Path(p).resolve(): i for i, p in enumerate(self.embedding_paths)}
        
        combined_features = []
        common_image_paths = []
        
        for common_path in common_paths:
            pose_idx = pose_path_to_idx[common_path]
            embedding_idx = embedding_path_to_idx[common_path]
            
            # Extract pose features
            pose_features = self.extract_pose_features(self.pose_data[pose_idx])
            if pose_features is None:
                continue
            
            # Get embedding features
            embedding_features = self.embeddings[embedding_idx]
            
            # Combine features with weights
            pose_weight = self.config.get('pose_weight', 0.5)
            embedding_weight = self.config.get('embedding_weight', 0.5)
            
            # Normalize feature dimensions
            pose_features_norm = pose_features / np.linalg.norm(pose_features)
            embedding_features_norm = embedding_features / np.linalg.norm(embedding_features)
            
            # Weighted combination
            combined_feature = np.concatenate([
                pose_weight * pose_features_norm,
                embedding_weight * embedding_features_norm
            ])
            
            combined_features.append(combined_feature)
            common_image_paths.append(str(common_path))
        
        if len(combined_features) == 0:
            raise ValueError("No valid combined features could be created")
        
        feature_matrix = np.array(combined_features)
        logger.info(f"Created {len(combined_features)} combined feature vectors of dimension {feature_matrix.shape[1]}")
        
        return feature_matrix, common_image_paths
    
    def create_pose_only_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature vectors from pose data only.
        
        Returns:
            Tuple of (feature_vectors, image_paths)
        """
        pose_features = []
        valid_paths = []
        
        for i, pose_item in enumerate(self.pose_data):
            features = self.extract_pose_features(pose_item)
            if features is not None:
                pose_features.append(features)
                valid_paths.append(self.pose_paths[i])
        
        if len(pose_features) == 0:
            raise ValueError("No valid pose features could be extracted")
        
        feature_matrix = np.array(pose_features)
        logger.info(f"Created {len(pose_features)} pose feature vectors of dimension {feature_matrix.shape[1]}")
        
        return feature_matrix, valid_paths
    
    def create_embedding_only_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature vectors from embedding data only.
        
        Returns:
            Tuple of (feature_vectors, image_paths)
        """
        # Embeddings are already normalized
        logger.info(f"Using {len(self.embeddings)} embedding feature vectors of dimension {self.embeddings.shape[1]}")
        return self.embeddings.copy(), self.embedding_paths.copy()
    
    def create_canny_only_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature vectors from Canny edge data only.
        
        Returns:
            Tuple of (feature_vectors, image_paths)
        """
        if self.feature_data is None or 'canny' not in self.feature_data:
            raise ValueError("No Canny feature data available")
        
        canny_features = self.feature_data['canny']
        feature_paths = self.feature_paths.copy()
        
        # Check if we have enhanced spatial features (86 dimensions) or legacy features (10 dimensions)
        feature_dim = canny_features.shape[1]
        logger.info(f"Canny features have {feature_dim} dimensions")
        
        if feature_dim >= 80:  # Enhanced spatial features
            logger.info("Using enhanced spatial Canny features for better shape clustering")
            
            # For spatial features, use more appropriate normalization
            # Option 1: Standard normalization (mean=0, std=1) per feature dimension
            canny_features_norm = (canny_features - np.mean(canny_features, axis=0)) / (np.std(canny_features, axis=0) + 1e-8)
            
            # Option 2: Min-max normalization (preserve relative magnitudes)
            # canny_features_norm = (canny_features - np.min(canny_features, axis=0)) / (np.max(canny_features, axis=0) - np.min(canny_features, axis=0) + 1e-8)
            
            # Option 3: Light L2 normalization (less aggressive than before)
            # canny_features_norm = canny_features / (np.linalg.norm(canny_features, axis=1, keepdims=True) + 1e-8)
            
        else:  # Legacy global features
            logger.info("Using legacy global Canny features (recommend upgrading to spatial features)")
            # Keep original L2 normalization for backward compatibility
            canny_features_norm = canny_features / np.linalg.norm(canny_features, axis=1, keepdims=True)
        
        logger.info(f"Using {len(canny_features_norm)} Canny edge feature vectors of dimension {canny_features_norm.shape[1]}")
        return canny_features_norm, feature_paths
    
    def create_canny_embedding_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create combined feature vectors from Canny edge and embedding data.
        
        Returns:
            Tuple of (feature_vectors, common_image_paths)
        """
        # Find common images between canny features and embedding data
        canny_paths_set = set(Path(p).resolve() for p in self.feature_paths)
        embedding_paths_set = set(Path(p).resolve() for p in self.embedding_paths)
        common_paths = canny_paths_set.intersection(embedding_paths_set)
        
        if len(common_paths) == 0:
            raise ValueError("No common images found between Canny features and embedding data")
        
        logger.info(f"Found {len(common_paths)} common images for Canny+embedding feature combination")
        
        # Create mappings
        canny_path_to_idx = {Path(p).resolve(): i for i, p in enumerate(self.feature_paths)}
        embedding_path_to_idx = {Path(p).resolve(): i for i, p in enumerate(self.embedding_paths)}
        
        combined_features = []
        common_image_paths = []
        
        for common_path in common_paths:
            canny_idx = canny_path_to_idx[common_path]
            embedding_idx = embedding_path_to_idx[common_path]
            
            # Get Canny features
            canny_features = self.feature_data['canny'][canny_idx]
            
            # Get embedding features
            embedding_features = self.embeddings[embedding_idx]
            
            # Combine features with weights
            canny_weight = self.config.get('canny_weight', 0.4)
            embedding_weight = self.config.get('embedding_weight', 0.6)
            
            # Normalize feature dimensions
            canny_features_norm = canny_features / np.linalg.norm(canny_features)
            embedding_features_norm = embedding_features / np.linalg.norm(embedding_features)
            
            # Weighted combination
            combined_feature = np.concatenate([
                canny_weight * canny_features_norm,
                embedding_weight * embedding_features_norm
            ])
            
            combined_features.append(combined_feature)
            common_image_paths.append(str(common_path))
        
        if len(combined_features) == 0:
            raise ValueError("No valid Canny+embedding combined features could be created")
        
        feature_matrix = np.array(combined_features)
        logger.info(f"Created {len(combined_features)} Canny+embedding feature vectors of dimension {feature_matrix.shape[1]}")
        
        return feature_matrix, common_image_paths
    
    def create_feature_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature vectors based on the configured feature type.
        
        Returns:
            Tuple of (feature_vectors, image_paths)
        """
        feature_type = self.config['feature_type']
        
        if feature_type == 'pose':
            return self.create_pose_only_features()
        elif feature_type == 'embedding':
            return self.create_embedding_only_features()
        elif feature_type == 'combined':
            return self.create_combined_features()
        elif feature_type == 'canny':
            return self.create_canny_only_features()
        elif feature_type == 'canny_embedding':
            return self.create_canny_embedding_features()
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
    
    def build_faiss_index(self, features: np.ndarray) -> 'faiss.Index':
        """
        Build FAISS index for similarity-based clustering.
        
        Args:
            features: Feature matrix
            
        Returns:
            Built FAISS index
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required for FAISS-based clustering")
        
        dimension = features.shape[1]
        index_type = self.config.get('faiss_index_type', 'IndexFlatIP')
        
        # Create FAISS index based on type
        if index_type == 'IndexFlatIP':
            # Inner Product (cosine similarity for normalized vectors)
            index = faiss.IndexFlatIP(dimension)
        elif index_type == 'IndexFlatL2':
            # L2 distance (Euclidean)
            index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IndexIVFFlat':
            # IVF with flat quantizer (faster for large datasets)
            nlist = min(100, max(1, features.shape[0] // 39))  # Rule of thumb
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(features.astype(np.float32))
        elif index_type == 'IndexHNSW':
            # Hierarchical NSW for fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 64
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")
        
        # Add features to index
        features_float32 = features.astype(np.float32)
        index.add(features_float32)
        
        logger.info(f"Built FAISS {index_type} index with {index.ntotal} vectors")
        return index
    
    def perform_faiss_similarity_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        Perform FAISS-based similarity clustering.
        
        This method uses FAISS to find similar images and groups them into clusters
        based on similarity thresholds rather than traditional clustering algorithms.
        
        Args:
            features: Feature matrix
            
        Returns:
            Cluster labels array
        """
        logger.info("Performing FAISS similarity-based clustering...")
        
        # Build FAISS index
        faiss_index = self.build_faiss_index(features)
        
        similarity_threshold = self.config.get('similarity_threshold', 0.7)
        max_search_depth = self.config.get('max_cluster_search_depth', 100)
        use_iterative = self.config.get('use_iterative_clustering', True)
        
        n_samples = len(features)
        cluster_labels = np.full(n_samples, -1, dtype=int)  # -1 = unclustered
        current_cluster_id = 0
        
        if use_iterative:
            # Iterative approach: start with highest confidence images
            unprocessed_indices = set(range(n_samples))
            
            while unprocessed_indices:
                # Pick the next seed (could be random or based on some criteria)
                seed_idx = next(iter(unprocessed_indices))
                
                # Find similar images to this seed
                query_vector = features[seed_idx:seed_idx+1].astype(np.float32)
                k = min(max_search_depth, len(unprocessed_indices))
                
                if self.config.get('faiss_index_type') == 'IndexFlatL2':
                    # For L2 distance, smaller is better
                    distances, indices = faiss_index.search(query_vector, k)
                    
                    # Better distance-to-similarity conversion for spatial features
                    # Use exponential decay: similarity = exp(-distance/scale)
                    # This gives more meaningful similarity values for spatial edge features
                    scale = np.std(distances[0]) + 1e-6  # Adaptive scale based on distance distribution
                    similarities = np.exp(-distances[0] / scale)
                else:
                    # For IndexFlatIP, higher is better (already similarity)
                    similarities, indices = faiss_index.search(query_vector, k)
                    similarities = similarities[0]
                
                # Find images above similarity threshold
                similar_mask = similarities >= similarity_threshold
                similar_indices = indices[0][similar_mask]
                
                # Only consider unprocessed images
                cluster_members = [idx for idx in similar_indices if idx in unprocessed_indices]
                
                if len(cluster_members) >= self.config.get('min_cluster_size', 5):
                    # Create cluster
                    for idx in cluster_members:
                        cluster_labels[idx] = current_cluster_id
                        unprocessed_indices.discard(idx)
                    
                    logger.debug(f"Created cluster {current_cluster_id} with {len(cluster_members)} members")
                    current_cluster_id += 1
                else:
                    # Mark as noise
                    cluster_labels[seed_idx] = -1
                    unprocessed_indices.discard(seed_idx)
        
        else:
            # Simple pairwise approach (slower but more thorough)
            for i in range(n_samples):
                if cluster_labels[i] != -1:  # Already assigned
                    continue
                
                # Find similar images
                query_vector = features[i:i+1].astype(np.float32)
                k = min(max_search_depth, n_samples)
                
                if self.config.get('faiss_index_type') == 'IndexFlatL2':
                    distances, indices = faiss_index.search(query_vector, k)
                    
                    # Better distance-to-similarity conversion for spatial features
                    scale = np.std(distances[0]) + 1e-6  # Adaptive scale based on distance distribution
                    similarities = np.exp(-distances[0] / scale)
                else:
                    similarities, indices = faiss_index.search(query_vector, k)
                    similarities = similarities[0]
                
                # Find unassigned similar images
                similar_mask = similarities >= similarity_threshold
                similar_indices = indices[0][similar_mask]
                unassigned_similar = [idx for idx in similar_indices if cluster_labels[idx] == -1]
                
                if len(unassigned_similar) >= self.config.get('min_cluster_size', 5):
                    # Create cluster
                    for idx in unassigned_similar:
                        cluster_labels[idx] = current_cluster_id
                    
                    current_cluster_id += 1
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"FAISS similarity clustering completed:")
        logger.info(f"  Found {n_clusters} clusters")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        logger.info(f"  Noise points: {n_noise}")
        
        return cluster_labels
    
    def determine_optimal_clusters(self, features: np.ndarray, max_k: int = 20) -> int:
        """
        Determine optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            features: Feature matrix
            max_k: Maximum number of clusters to test
            
        Returns:
            Optimal number of clusters
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, defaulting to 8 clusters")
            return 8
        
        n_samples = len(features)
        max_k = min(max_k, n_samples // 2)  # Ensure reasonable upper bound
        
        if max_k < 2:
            return 2
        
        logger.info(f"Determining optimal number of clusters (testing 2-{max_k})...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find elbow point (simplified)
        elbow_k = 2
        if len(inertias) > 2:
            # Calculate rate of change
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            if len(delta_deltas) > 0:
                elbow_idx = np.argmax(delta_deltas) + 2  # +2 because we start from k=2
                elbow_k = k_range[min(elbow_idx, len(k_range) - 1)]
        
        # Find best silhouette score
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_silhouette_k = k_range[best_silhouette_idx]
        
        # Choose based on both metrics (prefer silhouette if reasonable)
        if silhouette_scores[best_silhouette_idx] > 0.3:
            optimal_k = best_silhouette_k
            logger.info(f"Optimal clusters: {optimal_k} (best silhouette score: {silhouette_scores[best_silhouette_idx]:.3f})")
        else:
            optimal_k = elbow_k
            logger.info(f"Optimal clusters: {optimal_k} (elbow method)")
        
        return optimal_k
    
    def perform_clustering(self, features: np.ndarray) -> np.ndarray:
        """
        Perform clustering on feature vectors.
        
        Args:
            features: Feature matrix
            
        Returns:
            Cluster labels array
        """
        method = self.config['clustering_method']
        
        if method == 'faiss_similarity':
            return self.perform_faiss_similarity_clustering(features)
        elif method == 'kmeans':
            return self._cluster_kmeans(features)
        elif method == 'hdbscan':
            return self._cluster_hdbscan(features)
        elif method == 'dbscan':
            return self._cluster_dbscan(features)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _cluster_kmeans(self, features: np.ndarray) -> np.ndarray:
        """Perform K-means clustering."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for K-means clustering")
        
        n_clusters = self.config['n_clusters']
        if n_clusters == 'auto':
            n_clusters = self.determine_optimal_clusters(features)
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.get('kmeans_random_state', 42),
            max_iter=self.config.get('kmeans_max_iter', 300),
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(features)
        
        # Calculate clustering metrics
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(features, cluster_labels)
            calinski_score = calinski_harabasz_score(features, cluster_labels)
            logger.info(f"Silhouette score: {silhouette_avg:.3f}")
            logger.info(f"Calinski-Harabasz score: {calinski_score:.3f}")
        
        return cluster_labels
    
    def _cluster_hdbscan(self, features: np.ndarray) -> np.ndarray:
        """Perform HDBSCAN clustering."""
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN required for HDBSCAN clustering")
        
        logger.info("Performing HDBSCAN clustering...")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.get('hdbscan_min_cluster_size', 10),
            min_samples=self.config.get('hdbscan_min_samples', 5),
            metric='euclidean'
        )
        
        cluster_labels = clusterer.fit_predict(features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return cluster_labels
    
    def _cluster_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for DBSCAN clustering")
        
        logger.info("Performing DBSCAN clustering...")
        
        dbscan = DBSCAN(
            eps=self.config.get('dbscan_eps', 0.5),
            min_samples=self.config.get('dbscan_min_samples', 5),
            metric='euclidean'
        )
        
        cluster_labels = dbscan.fit_predict(features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return cluster_labels
    
    def analyze_clusters(self, cluster_labels: np.ndarray, 
                        image_paths: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Analyze clustering results and create cluster summaries.
        
        Args:
            cluster_labels: Array of cluster assignments
            image_paths: List of image paths corresponding to cluster labels
            
        Returns:
            Dictionary with cluster analysis results
        """
        logger.info("Analyzing cluster results...")
        
        cluster_analysis = {}
        unique_labels = np.unique(cluster_labels)
        
        for cluster_id in unique_labels:
            cluster_mask = cluster_labels == cluster_id
            cluster_images = [image_paths[i] for i in range(len(image_paths)) if cluster_mask[i]]
            cluster_size = len(cluster_images)
            
            # Skip noise points (cluster_id = -1) if too small
            if cluster_id == -1 and cluster_size < self.config.get('min_cluster_size', 5):
                continue
            
            # Sample examples for cluster
            max_examples = self.config.get('max_examples_per_cluster', 50)
            if cluster_size > max_examples:
                # Random sampling
                np.random.seed(42)
                example_indices = np.random.choice(len(cluster_images), max_examples, replace=False)
                example_images = [cluster_images[i] for i in example_indices]
            else:
                example_images = cluster_images
            
            # Create cluster summary
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': cluster_size,
                'percentage': cluster_size / len(image_paths) * 100,
                'example_images': example_images,
                'all_images': cluster_images,
                'is_noise': cluster_id == -1
            }
            
            cluster_analysis[cluster_id] = cluster_info
        
        # Sort clusters by size (largest first)
        sorted_clusters = dict(sorted(cluster_analysis.items(), 
                                    key=lambda x: x[1]['size'], reverse=True))
        
        logger.info(f"Created analysis for {len(sorted_clusters)} clusters")
        
        return sorted_clusters

    def find_pose_for_image(self, image_path: str) -> Optional[Dict]:
        """
        Find pose data for a specific image path.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Pose data dictionary or None if not found
        """
        if self.pose_data is None:
            return None
        
        # Normalize paths for comparison
        target_path = Path(image_path).resolve()
        
        for pose_item in self.pose_data:
            pose_path = Path(pose_item['image_path']).resolve()
            if pose_path == target_path:
                return pose_item
        
        return None

    def save_pose_visualization_for_cluster(self, image_path: str, pose_data: Dict[str, Any], 
                                          output_path: Path) -> Optional[str]:
        """
        Save pose visualization overlay for a cluster example image.
        
        Adapted from the pose extraction service visualization function.
        
        Args:
            image_path: Path to the source image
            pose_data: Pose data dictionary
            output_path: Output path for the visualization
            
        Returns:
            Path to saved visualization or None if failed
        """
        if not pose_data or not pose_data.get('has_pose', False):
            # Skip pose overlay if no pose data (don't create fallback copy)
            logger.debug(f"No pose data available for {Path(image_path).name}")
            return None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # Create visualization
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.imshow(image_rgb)
            
            # Define colors for different people
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # Draw poses for all detected people
            people = pose_data.get('people', [])
            
            # If no people in new format, try old format compatibility
            if not people and pose_data.get('landmarks'):
                people = [{
                    'landmarks': pose_data['landmarks'],
                    'visibility_scores': pose_data.get('visibility_scores', []),
                    'pose_confidence': pose_data.get('pose_confidence', 0.0),
                    'person_id': 0
                }]
            
            for person_idx, person in enumerate(people):
                if not person.get('landmarks'):
                    continue
                    
                # Get landmarks and visibility
                landmarks = np.array(person['landmarks']).reshape(-1, 2)
                visibility = person.get('visibility_scores', [])
                color = colors[person_idx % len(colors)]
                
                # Draw landmarks
                for i, (x, y) in enumerate(landmarks):
                    if i < len(visibility) and visibility[i] > 0.5:  # Only draw visible landmarks
                        ax.plot(x * w, y * h, 'o', color=color, markersize=6)
                
                # Draw connections (MediaPipe 33-point skeleton)
                # Body connections
                connections = [
                    # Face outline
                    (0, 1), (1, 2), (2, 3), (3, 7),
                    (0, 4), (4, 5), (5, 6), (6, 8),
                    (9, 10),
                    # Body
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
                    (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
                    (15, 17), (15, 19), (16, 18), (16, 20),  # Hands
                    (17, 19), (18, 20)
                ]
                
                for start_idx, end_idx in connections:
                    if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                        start_idx < len(visibility) and end_idx < len(visibility) and
                        visibility[start_idx] > 0.5 and visibility[end_idx] > 0.5):
                        x1, y1 = landmarks[start_idx] * [w, h]
                        x2, y2 = landmarks[end_idx] * [w, h]
                        ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=3, alpha=0.7)
                
                # Add person label
                if landmarks.size > 0:
                    # Use nose position for label if available, otherwise center
                    if len(landmarks) > 0:
                        label_x = landmarks[0, 0] * w  # Nose x
                        label_y = (landmarks[0, 1] - 0.1) * h  # Above nose
                    else:
                        label_x = np.mean(landmarks[:, 0]) * w
                        label_y = (np.min(landmarks[:, 1]) - 0.05) * h
                    
                    confidence = person.get('pose_confidence', 0.0)
                    if len(people) > 1:
                        label_text = f"Person {person_idx+1}\nConf: {confidence:.2f}"
                    else:
                        label_text = f"Confidence: {confidence:.2f}"
                    
                    ax.text(label_x, label_y, label_text, 
                           color=color, fontsize=12, ha='center', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Set title
            image_name = Path(image_path).name
            person_count = pose_data.get('person_count', len(people))
            avg_confidence = np.mean([p.get('pose_confidence', 0) for p in people]) if people else 0
            ax.set_title(f"{image_name}\nPeople: {person_count}, Avg Confidence: {avg_confidence:.3f}", 
                        fontsize=14, weight='bold')
            ax.axis('off')
            
            # Save with optimized settings for speed
            plt.savefig(output_path, bbox_inches='tight', dpi=100, facecolor='white', 
                       format='png')
            plt.close(fig)  # Close specific figure
            plt.clf()       # Clear current figure
            
            return str(output_path)
            
        except Exception as e:
            logger.debug(f"Failed to save pose visualization for {Path(image_path).name}: {e}")
            return None

    def create_cluster_collage(self, cluster_images: List[str], cluster_id: int, 
                             cluster_dir: Path, max_images: int = 30) -> Optional[str]:
        """
        Create a collage image showing all examples from a cluster.
        
        Args:
            cluster_images: List of image paths in the cluster
            cluster_id: Cluster ID for naming
            cluster_dir: Directory where collage will be saved
            max_images: Maximum number of images to include in collage
            
        Returns:
            Path to created collage or None if failed
        """
        if not self.config.get('generate_cluster_collages', True):
            logger.debug(f"Skipping collage generation for cluster {cluster_id}")
            return None

        try:
            # Limit number of images for collage
            images_to_use = cluster_images[:max_images]
            
            if len(images_to_use) == 0:
                return None
            
            # Calculate grid dimensions (roughly square)
            n_images = len(images_to_use)
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
            
            # Thumbnail size for each image
            thumb_size = (150, 150)
            
            # Create collage canvas
            collage_width = cols * thumb_size[0] + (cols + 1) * 10  # 10px spacing
            collage_height = rows * thumb_size[1] + (rows + 1) * 10 + 50  # Extra space for title
            
            # Create white background
            collage = Image.new('RGB', (collage_width, collage_height), 'white')
            
            # Load and place images
            loaded_images = 0
            for idx, image_path in enumerate(images_to_use):
                try:
                    # Calculate position
                    row = idx // cols
                    col = idx % cols
                    x = col * (thumb_size[0] + 10) + 10
                    y = row * (thumb_size[1] + 10) + 40  # 40px for title space
                    
                    # Load and resize image
                    img = Image.open(image_path)
                    
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize maintaining aspect ratio
                    img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    
                    # Center the thumbnail in the allocated space
                    thumb_x = x + (thumb_size[0] - img.width) // 2
                    thumb_y = y + (thumb_size[1] - img.height) // 2
                    
                    # Paste image onto collage
                    collage.paste(img, (thumb_x, thumb_y))
                    loaded_images += 1
                    
                except Exception as img_error:
                    logger.debug(f"Failed to load image {Path(image_path).name} for collage: {img_error}")
                    continue
            
            if loaded_images == 0:
                return None
            
            # Add title using PIL drawing
            try:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(collage)
                
                # Try to use a decent font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                
                if cluster_id == -1:
                    title = f"Noise Cluster ({loaded_images} images)"
                else:
                    title = f"Cluster {cluster_id} ({loaded_images} images)"
                
                # Calculate text position (centered)
                text_bbox = draw.textbbox((0, 0), title, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = (collage_width - text_width) // 2
                
                # Draw title
                draw.text((text_x, 10), title, fill='black', font=font)
                
            except Exception as text_error:
                logger.debug(f"Failed to add title to collage: {text_error}")
            
            # Save collage
            if cluster_id == -1:
                collage_filename = "noise_cluster_collage.jpg"
            else:
                collage_filename = f"cluster_{cluster_id:03d}_collage.jpg"
            
            collage_path = cluster_dir.parent / collage_filename
            collage.save(collage_path, 'JPEG', quality=85, optimize=True)
            
            logger.debug(f"Created collage for cluster {cluster_id} with {loaded_images} images")
            return str(collage_path)
            
        except Exception as e:
            logger.warning(f"Failed to create collage for cluster {cluster_id}: {e}")
            return None

    def copy_cluster_examples(self, cluster_analysis: Dict[int, Dict[str, Any]], 
                            output_dir: Path) -> Dict[int, str]:
        """
        Copy example images for each cluster to output directory.
        Optionally generates pose overlays for cluster examples.
        
        Args:
            cluster_analysis: Cluster analysis results
            output_dir: Output directory
            
        Returns:
            Dictionary mapping cluster_id to example directory path
        """
        if not self.config.get('copy_examples', True):
            return {}
        
        # Calculate total work for progress tracking
        total_images = sum(len(cluster_info['example_images']) for cluster_info in cluster_analysis.values())
        total_clusters = len(cluster_analysis)
        generate_pose_overlays = self.config.get('generate_pose_overlays', True)
        
        logger.info(f"Copying cluster example images...")
        logger.info(f"Processing {total_clusters} clusters with {total_images} example images")
        
        # Test file system permissions
        try:
            test_dir = output_dir / "test_permissions"
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / "test.txt"
            test_file.write_text("permission test")
            test_file.unlink()
            test_dir.rmdir()
            logger.info("✅ File system write permissions verified")
        except Exception as perm_test_error:
            logger.warning(f"⚠️  File system permission test failed: {perm_test_error}")
            logger.warning(f"   Output directory: {output_dir}")
            logger.warning(f"   This may cause copy failures...")
        
        if generate_pose_overlays:
            logger.info(f"Will generate {total_images} pose overlays (this may take a while)...")
        
        cluster_dirs = {}
        processed_images = 0
        processed_clusters = 0
        
        # Import progress bar
        try:
            from tqdm import tqdm
            use_progress = True
        except ImportError:
            logger.warning("tqdm not available, progress will be shown as log messages")
            use_progress = False
        
        cluster_items = list(cluster_analysis.items())
        if use_progress:
            cluster_iterator = tqdm(cluster_items, desc="Processing clusters", unit="cluster")
        else:
            cluster_iterator = cluster_items
        
        for cluster_id, cluster_info in cluster_iterator:
            processed_clusters += 1
            
            if cluster_info['is_noise']:
                cluster_dir = output_dir / "cluster_examples" / "noise"
                overlay_dir = output_dir / "cluster_examples_with_poses" / "noise"
                cluster_name = "noise"
            else:
                cluster_dir = output_dir / "cluster_examples" / f"cluster_{cluster_id:03d}"
                overlay_dir = output_dir / "cluster_examples_with_poses" / f"cluster_{cluster_id:03d}"
                cluster_name = f"cluster_{cluster_id:03d}"
            
            # Create directories with explicit permissions
            try:
                cluster_dir.mkdir(parents=True, exist_ok=True)
                # Test write permissions
                test_file = cluster_dir / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as dir_error:
                logger.warning(f"Failed to create or write to cluster directory {cluster_dir}: {dir_error}")
                continue
            
            cluster_generate_overlays = generate_pose_overlays
            if cluster_generate_overlays:
                try:
                    overlay_dir.mkdir(parents=True, exist_ok=True)
                except Exception as overlay_dir_error:
                    logger.warning(f"Failed to create overlay directory {overlay_dir}: {overlay_dir_error}")
                    cluster_generate_overlays = False  # Disable for this cluster only
            
            cluster_dirs[cluster_id] = str(cluster_dir)
            
            # Progress logging for clusters without tqdm
            if not use_progress and processed_clusters % 10 == 0:
                logger.info(f"Processing cluster {processed_clusters}/{total_clusters} ({cluster_name})")
            
            # Copy example images for this cluster
            cluster_images = cluster_info['example_images']
            for i, image_path in enumerate(cluster_images):
                processed_images += 1
                
                try:
                    source_path = Path(image_path)
                    if not source_path.exists():
                        continue
                    
                    # Create descriptive filename
                    extension = source_path.suffix
                    dest_filename = f"example_{i+1:03d}_{source_path.stem}{extension}"
                    dest_path = cluster_dir / dest_filename
                    
                    # Copy original image with robust error handling for mounted drives
                    try:
                        # Method 1: Simple byte copy (most compatible)
                        with open(source_path, 'rb') as src, open(dest_path, 'wb') as dst:
                            dst.write(src.read())
                    except PermissionError as perm_error:
                        logger.warning(f"Permission denied copying {source_path.name}: {perm_error}")
                        # Try to check directory permissions
                        if not dest_path.parent.exists():
                            logger.warning(f"Destination directory does not exist: {dest_path.parent}")
                        elif not os.access(dest_path.parent, os.W_OK):
                            logger.warning(f"No write permission to directory: {dest_path.parent}")
                        continue
                    except Exception as copy_error:
                        # Fallback: try shutil copy without metadata
                        try:
                            import shutil
                            shutil.copy(source_path, dest_path)
                        except Exception as fallback_error:
                            logger.warning(f"Failed to copy {source_path.name}: {copy_error}, fallback failed: {fallback_error}")
                            continue
                    
                    # Generate pose overlay if requested
                    if cluster_generate_overlays:
                        pose_data = self.find_pose_for_image(image_path)
                        overlay_filename = f"pose_{i+1:03d}_{source_path.stem}.png"
                        overlay_path = overlay_dir / overlay_filename
                        
                        # Generate pose overlay with better error handling
                        try:
                            result = self.save_pose_visualization_for_cluster(
                                image_path, pose_data, overlay_path
                            )
                            if not result:
                                logger.debug(f"No pose overlay generated for {source_path.name}")
                        except Exception as pose_error:
                            # Only log first few errors to avoid spam
                            if processed_images <= 10:
                                logger.warning(f"Failed to generate pose overlay for {source_path.name}: {pose_error}")
                            elif processed_images == 11:
                                logger.warning("Further pose overlay errors will be suppressed...")
                        
                        # Force matplotlib cleanup to prevent memory issues
                        try:
                            plt.close('all')
                        except:
                            pass
                    
                    # Progress logging for images without tqdm (reduced frequency)
                    if not use_progress and processed_images % 500 == 0:
                        progress_pct = (processed_images / total_images) * 100
                        logger.info(f"Processed {processed_images}/{total_images} images ({progress_pct:.1f}%)")
                    
                except Exception as e:
                    logger.warning(f"Failed to copy {image_path}: {e}")
            
            # Generate cluster collage after processing all images in this cluster
            try:
                collage_path = self.create_cluster_collage(
                    cluster_images, cluster_id, cluster_dir, 
                    max_images=self.config.get('max_examples_per_cluster', 30)
                )
                if collage_path:
                    logger.debug(f"Created collage for cluster {cluster_id}")
                else:
                    logger.debug(f"No collage created for cluster {cluster_id}")
            except Exception as collage_error:
                logger.warning(f"Failed to create collage for cluster {cluster_id}: {collage_error}")

        logger.info(f"✅ Copied examples for {len(cluster_dirs)} clusters")
        logger.info(f"✅ Processed {processed_images} images across {processed_clusters} clusters")
        if self.config.get('generate_cluster_collages', True):
            logger.info("✅ Generated cluster collages for quick review")
        if generate_pose_overlays:
            logger.info("✅ Generated pose overlays for cluster examples")
        
        return cluster_dirs
    
    def generate_visualizations(self, features: np.ndarray, cluster_labels: np.ndarray,
                              cluster_analysis: Dict[int, Dict[str, Any]], 
                              output_dir: Path) -> List[str]:
        """
        Generate clustering visualization plots.
        
        Args:
            features: Feature matrix used for clustering
            cluster_labels: Cluster assignments
            cluster_analysis: Cluster analysis results
            output_dir: Output directory for visualizations
            
        Returns:
            List of generated visualization file paths
        """
        if not self.config.get('generate_visualizations', True):
            return []
        
        try:
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            generated_files = []
            
            # 1. Cluster size distribution
            cluster_sizes = [info['size'] for info in cluster_analysis.values() if not info['is_noise']]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(cluster_sizes)), sorted(cluster_sizes, reverse=True))
            plt.xlabel('Cluster Rank')
            plt.ylabel('Number of Images')
            plt.title(f'Cluster Size Distribution ({self.config["clustering_method"]})')
            plt.xticks(range(0, len(cluster_sizes), max(1, len(cluster_sizes)//10)))
            
            size_dist_path = viz_dir / "cluster_size_distribution.png"
            plt.savefig(size_dist_path, dpi=150, bbox_inches='tight')
            plt.close()
            generated_files.append(str(size_dist_path))
            
            # 2. Cluster composition pie chart (top 10 clusters)
            top_clusters = sorted([(k, v['size']) for k, v in cluster_analysis.items() 
                                 if not v['is_noise']], key=lambda x: x[1], reverse=True)[:10]
            
            if len(top_clusters) > 1:
                plt.figure(figsize=(10, 8))
                sizes = [size for _, size in top_clusters]
                labels = [f"Cluster {cid}" for cid, _ in top_clusters]
                
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.title('Top 10 Clusters by Size')
                
                pie_chart_path = viz_dir / "cluster_composition.png"
                plt.savefig(pie_chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                generated_files.append(str(pie_chart_path))
            
            # 3. Feature dimensionality reduction (if possible)
            if features.shape[1] > 2:
                try:
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE
                    
                    # PCA visualization
                    pca = PCA(n_components=2, random_state=42)
                    features_2d = pca.fit_transform(features)
                    
                    plt.figure(figsize=(12, 8))
                    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                        c=cluster_labels, cmap='tab20', alpha=0.7)
                    plt.colorbar(scatter)
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                    plt.title(f'Clusters in PCA Space ({self.config["clustering_method"]})')
                    
                    pca_path = viz_dir / "clusters_pca.png"
                    plt.savefig(pca_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    generated_files.append(str(pca_path))
                    
                except ImportError:
                    logger.warning("Scikit-learn not available for dimensionality reduction")
            
            # 4. Clustering method specific visualizations
            if self.config['clustering_method'] == 'faiss_similarity':
                # Add similarity threshold visualization
                plt.figure(figsize=(10, 6))
                similarity_threshold = self.config.get('similarity_threshold', 0.7)
                plt.axhline(y=similarity_threshold, color='r', linestyle='--', 
                          label=f'Similarity Threshold ({similarity_threshold})')
                plt.xlabel('Image Pairs')
                plt.ylabel('Similarity Score')
                plt.title('FAISS Similarity Clustering Threshold')
                plt.legend()
                
                threshold_path = viz_dir / "similarity_threshold.png"
                plt.savefig(threshold_path, dpi=150, bbox_inches='tight')
                plt.close()
                generated_files.append(str(threshold_path))
            
            logger.info(f"Generated {len(generated_files)} visualizations")
            return generated_files
            
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")
            return []
    
    def save_results(self, cluster_analysis: Dict[int, Dict[str, Any]],
                    cluster_dirs: Dict[int, str], visualization_files: List[str],
                    config: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Save clustering results using standard output structure.
        
        Args:
            cluster_analysis: Cluster analysis results
            cluster_dirs: Mapping of cluster_id to example directory
            visualization_files: List of generated visualization files
            config: Configuration dictionary
            run_id: Optional run identifier
            
        Returns:
            Dictionary with result metadata
        """
        # Use existing results manager (created earlier)
        results_dir = self.results_manager.get_results_dir()
        
        # Save cluster analysis
        cluster_analysis_path = results_dir / "cluster_analysis.json"
        
        # Make analysis JSON-serializable
        serializable_analysis = {}
        for cluster_id, cluster_info in cluster_analysis.items():
            serializable_analysis[str(cluster_id)] = {
                'cluster_id': cluster_info['cluster_id'],
                'size': cluster_info['size'],
                'percentage': cluster_info['percentage'],
                'is_noise': cluster_info['is_noise'],
                'example_images': cluster_info['example_images'],
                'example_count': len(cluster_info['example_images'])
            }
        
        with open(cluster_analysis_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
        
        # Save detailed cluster information
        detailed_clusters_path = results_dir / "detailed_clusters.json"
        detailed_clusters = {}
        
        for cluster_id, cluster_info in cluster_analysis.items():
            detailed_clusters[str(cluster_id)] = {
                'cluster_id': cluster_info['cluster_id'],
                'size': cluster_info['size'],
                'percentage': cluster_info['percentage'],
                'is_noise': cluster_info['is_noise'],
                'all_images': cluster_info['all_images']  # Full image list
            }
        
        with open(detailed_clusters_path, 'w') as f:
            json.dump(detailed_clusters, f, indent=2, default=str)
        
        # Generate clustering statistics
        total_images = sum(info['size'] for info in cluster_analysis.values())
        valid_clusters = [info for info in cluster_analysis.values() if not info['is_noise']]
        noise_cluster = next((info for info in cluster_analysis.values() if info['is_noise']), None)
        
        stats = {
            'total_images_clustered': total_images,
            'number_of_clusters': len(valid_clusters),
            'largest_cluster_size': max(info['size'] for info in valid_clusters) if valid_clusters else 0,
            'smallest_cluster_size': min(info['size'] for info in valid_clusters) if valid_clusters else 0,
            'average_cluster_size': np.mean([info['size'] for info in valid_clusters]) if valid_clusters else 0,
            'noise_points': noise_cluster['size'] if noise_cluster else 0,
            'noise_percentage': noise_cluster['percentage'] if noise_cluster else 0,
            'clustering_method': config['clustering_method'],
            'feature_type': config['feature_type'],
            'pose_overlays_generated': config.get('generate_pose_overlays', False)
        }
        
        # Add method-specific stats
        if config['clustering_method'] == 'faiss_similarity':
            stats.update({
                'faiss_index_type': config.get('faiss_index_type', 'IndexFlatIP'),
                'similarity_threshold': config.get('similarity_threshold', 0.7),
                'max_cluster_search_depth': config.get('max_cluster_search_depth', 100)
            })
        
        stats_path = results_dir / "clustering_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Prepare additional metadata
        additional_metadata = {
            'total_images_clustered': total_images,
            'number_of_clusters': len(valid_clusters),
            'clustering_method': config['clustering_method'],
            'feature_type': config['feature_type'],
            'largest_cluster_size': stats['largest_cluster_size'],
            'noise_points': stats['noise_points'],
            'visualizations_created': len(visualization_files),
            'pose_overlays_generated': config.get('generate_pose_overlays', False),
            'method': f"{config['clustering_method']}_clustering_{config['feature_type']}_features"
        }
        
        # Save configuration with metadata
        config_path = self.results_manager.save_run_config(config, additional_metadata)
        
        # Create final result metadata
        result_metadata = self.results_manager.save_result_metadata({
            'total_images_clustered': total_images,
            'number_of_clusters': len(valid_clusters),
            'clustering_method': config['clustering_method'],
            'feature_type': config['feature_type'],
            'files_created': [
                str(cluster_analysis_path),
                str(detailed_clusters_path),
                str(stats_path),
                str(config_path)
            ] + visualization_files
        })
        
        logger.info(f"Results saved to: {self.results_manager.get_results_dir()}")
        logger.info(f"Created {len(valid_clusters)} clusters from {total_images} images")
        if noise_cluster:
            logger.info(f"Identified {noise_cluster['size']} noise points ({noise_cluster['percentage']:.1f}%)")
        if config.get('generate_pose_overlays', False):
            logger.info("Generated pose overlays for cluster examples")
        
        return result_metadata
    
    def cluster_images(self, config_path: Union[str, Path], 
                      run_id: Optional[str] = None) -> str:
        """
        Main method to cluster images based on configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            run_id: Optional run identifier
            
        Returns:
            Path to the generated results directory
        """
        # Load configuration
        config = self.load_config(config_path)
        
        # Create results manager early
        if self.results_manager is None:
            self.results_manager = create_service_runner('clustering', config['output_path'])
        
        # Create run directory
        actual_run_id = self.results_manager.create_run(run_id)
        
        # Load data based on feature type
        if config['feature_type'] in ['embedding', 'combined', 'canny_embedding']:
            logger.info("Loading embeddings...")
            self.load_embeddings(config['embeddings_path'])
        
        if config['feature_type'] in ['pose', 'combined']:
            logger.info("Loading pose data...")
            self.load_pose_data(config['pose_data_path'])
        
        # NEW: Load feature extraction data for Canny-based clustering
        if config['feature_type'] in ['canny', 'canny_embedding']:
            logger.info("Loading feature extraction data...")
            self.load_feature_data(config['feature_data_path'])
        
        # Create feature vectors
        logger.info(f"Creating {config['feature_type']} feature vectors...")
        feature_vectors, image_paths = self.create_feature_vectors()
        
        self.feature_vectors = feature_vectors
        self.image_paths = image_paths
        
        # Perform clustering
        logger.info(f"Performing {config['clustering_method']} clustering...")
        cluster_labels = self.perform_clustering(feature_vectors)
        
        # Analyze clusters
        cluster_analysis = self.analyze_clusters(cluster_labels, image_paths)
        
        # Copy example images (with pose overlays if configured)
        cluster_dirs = self.copy_cluster_examples(
            cluster_analysis, self.results_manager.get_results_dir()
        )
        
        # Generate visualizations
        visualization_files = self.generate_visualizations(
            feature_vectors, cluster_labels, cluster_analysis,
            self.results_manager.get_results_dir()
        )
        
        # Save results
        result_metadata = self.save_results(
            cluster_analysis, cluster_dirs, visualization_files, config, run_id
        )
        
        return str(self.results_manager.get_results_dir())


def cluster_images(config_path: Union[str, Path], run_id: Optional[str] = None) -> str:
    """
    Convenience function to cluster images using the ClusteringService.
    
    Args:
        config_path: Path to the YAML configuration file
        run_id: Optional run identifier
    
    Returns:
        Path to the generated results directory
    """
    service = ClusteringService()
    return service.cluster_images(config_path, run_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster images using pose and/or embedding features with FAISS similarity")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--run-id", help="Optional run identifier")
    
    args = parser.parse_args()
    
    try:
        service = ClusteringService()
        results_path = service.cluster_images(args.config, args.run_id)
        print(f"✅ Image clustering completed successfully!")
        print(f"   Results saved to: {results_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1) 