#!/usr/bin/env python3
"""
Image Similarity Service

Finds images similar to a target image or group of images using pre-computed embeddings and FAISS.
Supports multiple FAISS index types for efficient similarity search and outputs top-k results with similarity scores.

This service reads configuration from a YAML file, loads embeddings from previous pipeline steps,
builds a FAISS index for efficient search, and finds the most similar images to specified target images.
For multiple target images, their embeddings are averaged to create a single query vector.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
from PIL import Image
import logging
import faiss
from collections import defaultdict, Counter

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


class SimilarityService:
    """
    Service for finding images similar to target images using pre-computed embeddings and FAISS.
    
    This service provides a clean interface for:
    - Loading and validating similarity search configurations
    - Loading pre-computed embeddings from previous pipeline steps
    - Building FAISS indices for efficient similarity search
    - Finding similar images using FAISS with multiple index types
    - Multi-target search strategies for improved semantic search
    - Saving results using standardized output structure
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        """
        Initialize the Similarity Service.
        
        Args:
            results_manager: Optional ResultsManager instance. If not provided,
                           will be created during run execution.
        """
        self.results_manager = results_manager
        self.config = None
        self.embeddings = None
        self.image_paths = None
        self.target_embeddings = None
        self.pose_data = None
        self.pose_image_paths = None
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate configuration for similarity search.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing validated configuration parameters
        """
        # Use common config validation service
        config = validate_config(config_path, 'similarity')
        
        # Set defaults for optional parameters
        config.setdefault('faiss_index_type', 'IndexFlatIP')  # Inner Product for cosine similarity
        config.setdefault('top_k', 30)
        config.setdefault('include_target_in_results', False)
        config.setdefault('normalize_embeddings', True)
        config.setdefault('copy_images', True)  # Whether to copy images or just save metadata
        
        # Multi-target search strategy options
        config.setdefault('use_multi_target_search', True)  # Enable improved multi-target search
        config.setdefault('multi_target_strategy', 'weighted_union')  # Strategy for combining results
        config.setdefault('individual_search_k', 50)  # How many results to get per target
        config.setdefault('target_weight_decay', 0.9)  # Weight decay for multiple targets
        
        # Pose-aware similarity options
        config.setdefault('use_pose_similarity', False)  # Enable pose-aware similarity
        config.setdefault('pose_data_path', None)  # Path to pose extraction results
        config.setdefault('pose_weight', 0.3)  # Weight for pose similarity (0.0-1.0)
        config.setdefault('embedding_weight', 0.7)  # Weight for embedding similarity
        config.setdefault('pose_similarity_method', 'euclidean')  # Method for pose comparison
        config.setdefault('pose_confidence_threshold', 0.2)  # Min confidence for pose matching
        
        # Convert paths to Path objects
        config['embeddings_path'] = Path(config['embeddings_path'])
        config['output_path'] = Path(config['output_path'])
        
        # Handle pose data path if pose similarity is enabled
        if config['use_pose_similarity']:
            if config['pose_data_path'] is None:
                raise ValueError("pose_data_path must be specified when use_pose_similarity is True")
            config['pose_data_path'] = Path(config['pose_data_path'])
            
            # Validate pose weight values
            pose_weight = config['pose_weight']
            embedding_weight = config['embedding_weight']
            if not (0.0 <= pose_weight <= 1.0) or not (0.0 <= embedding_weight <= 1.0):
                raise ValueError("pose_weight and embedding_weight must be between 0.0 and 1.0")
            if abs((pose_weight + embedding_weight) - 1.0) > 0.01:
                logger.warning(f"pose_weight ({pose_weight}) + embedding_weight ({embedding_weight}) != 1.0, results may be unexpected")
        
        # Handle target_images - can be single string or list
        if isinstance(config['target_images'], str):
            config['target_images'] = [config['target_images']]
        config['target_images'] = [Path(path) for path in config['target_images']]
        
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
        
        # Ensure embeddings are normalized
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.embeddings = embeddings
        self.image_paths = image_paths
        
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
        
        # Extract image paths
        pose_image_paths = [item['image_path'] for item in pose_data]
        
        # Filter out poses with low confidence
        confidence_threshold = self.config.get('pose_confidence_threshold', 0.2)
        valid_poses = []
        valid_paths = []
        
        for pose_item in pose_data:
            if pose_item.get('pose_confidence', 0.0) >= confidence_threshold:
                valid_poses.append(pose_item)
                valid_paths.append(pose_item['image_path'])
        
        self.pose_data = valid_poses
        self.pose_image_paths = valid_paths
        
        logger.info(f"Loaded {len(valid_poses)} valid poses from {len(pose_data)} total poses")
        logger.info(f"Filtered poses with confidence >= {confidence_threshold}")
        
        return valid_poses, valid_paths
    
    def compute_pose_similarity(self, pose1: Dict, pose2: Dict, 
                              method: str = 'euclidean') -> float:
        """
        Compute similarity between two poses using the same algorithm as pose extraction service.
        
        Args:
            pose1: First pose data dictionary
            pose2: Second pose data dictionary
            method: Similarity computation method ('euclidean', 'cosine')
            
        Returns:
            Similarity score (higher = more similar)
        """
        # Check if both poses have valid keypoints
        if (not pose1.get('body_keypoints') or not pose2.get('body_keypoints') or
            len(pose1['body_keypoints']) == 0 or len(pose2['body_keypoints']) == 0):
            return 0.0
        
        # Use the first person from each pose (could be extended to handle multiple people)
        kp1 = np.array(pose1['body_keypoints'][0]['keypoints'])
        kp2 = np.array(pose2['body_keypoints'][0]['keypoints'])
        
        # Reshape keypoints to (n_joints, 3) format
        kp1 = kp1.reshape(-1, 3)
        kp2 = kp2.reshape(-1, 3)
        
        # Filter out zero-confidence keypoints
        valid_joints = (kp1[:, 2] > 0) & (kp2[:, 2] > 0)
        if not np.any(valid_joints):
            return 0.0
        
        # Extract only x, y coordinates for valid joints
        kp1_valid = kp1[valid_joints, :2]
        kp2_valid = kp2[valid_joints, :2]
        
        if method == 'euclidean':
            # Normalize keypoints by image size (assuming 368x368 input)
            kp1_norm = kp1_valid / 368.0
            kp2_norm = kp2_valid / 368.0
            
            # Compute euclidean distance
            distance = np.linalg.norm(kp1_norm - kp2_norm)
            # Convert to similarity (0-1, higher = more similar)
            similarity = 1.0 / (1.0 + distance)
            
        elif method == 'cosine':
            # Flatten and compute cosine similarity
            vec1 = kp1_valid.flatten()
            vec2 = kp2_valid.flatten()
            
            # Normalize vectors
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1_norm, vec2_norm)
            # Ensure positive similarity (0-1)
            similarity = (similarity + 1.0) / 2.0
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return float(similarity)
    
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
    
    def compute_combined_similarity(self, target_image_path: str, candidate_image_path: str,
                                  embedding_similarity: float) -> float:
        """
        Compute combined similarity using both embedding and pose similarity.
        
        Args:
            target_image_path: Path to target image
            candidate_image_path: Path to candidate image
            embedding_similarity: Pre-computed embedding similarity
            
        Returns:
            Combined similarity score
        """
        if not self.config.get('use_pose_similarity', False):
            return embedding_similarity
        
        # Find pose data for both images
        target_pose = self.find_pose_for_image(target_image_path)
        candidate_pose = self.find_pose_for_image(candidate_image_path)
        
        # If either image has no pose data, fall back to embedding similarity
        if target_pose is None or candidate_pose is None:
            return embedding_similarity
        
        # Compute pose similarity
        pose_similarity = self.compute_pose_similarity(
            target_pose, candidate_pose, 
            self.config.get('pose_similarity_method', 'euclidean')
        )
        
        # Combine similarities using weighted average
        pose_weight = self.config.get('pose_weight', 0.3)
        embedding_weight = self.config.get('embedding_weight', 0.7)
        
        combined_similarity = (
            embedding_weight * embedding_similarity + 
            pose_weight * pose_similarity
        )
        
        return combined_similarity
    
    def rerank_with_pose_similarity(self, target_image_paths: List[str], 
                                   similar_image_paths: List[str], 
                                   similarity_scores: List[float]) -> Tuple[List[str], List[float]]:
        """
        Rerank similarity results using combined embedding + pose similarity.
        
        Args:
            target_image_paths: Paths to target images
            similar_image_paths: Paths to similar images from FAISS search
            similarity_scores: Original embedding similarity scores
            
        Returns:
            Tuple of (reranked_image_paths, reranked_similarity_scores)
        """
        if not self.config.get('use_pose_similarity', False):
            return similar_image_paths, similarity_scores
        
        logger.info("Reranking results with pose similarity...")
        
        # Compute combined similarities for all results
        combined_results = []
        pose_matches = 0
        
        for i, (candidate_path, embedding_sim) in enumerate(zip(similar_image_paths, similarity_scores)):
            # For multiple targets, use the first target for pose comparison
            # This could be enhanced to handle multiple targets better
            target_path = target_image_paths[0] if target_image_paths else ""
            
            combined_sim = self.compute_combined_similarity(
                target_path, candidate_path, embedding_sim
            )
            
            # Track how many images had pose data
            if (self.find_pose_for_image(target_path) is not None and 
                self.find_pose_for_image(candidate_path) is not None):
                pose_matches += 1
            
            combined_results.append((candidate_path, combined_sim, embedding_sim))
        
        # Sort by combined similarity (descending)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Extract reranked paths and scores
        reranked_paths = [result[0] for result in combined_results]
        reranked_scores = [result[1] for result in combined_results]
        
        logger.info(f"Reranked {len(similar_image_paths)} results using pose similarity")
        logger.info(f"Found pose data for {pose_matches}/{len(similar_image_paths)} image pairs")
        
        return reranked_paths, reranked_scores
        
    def find_target_images(self, target_image_paths: List[Path], 
                          image_paths: List[str]) -> List[int]:
        """
        Find indices of target images in the embeddings dataset.
        
        Args:
            target_image_paths: List of target image paths
            image_paths: List of all image paths in embeddings
            
        Returns:
            List of indices for target images
        """
        target_indices = []
        image_paths_set = {Path(path).resolve() for path in image_paths}
        
        for target_path in target_image_paths:
            target_resolved = target_path.resolve()
            
            # Try to find exact match
            if target_resolved in image_paths_set:
                idx = next(i for i, path in enumerate(image_paths) 
                          if Path(path).resolve() == target_resolved)
                target_indices.append(idx)
                logger.info(f"Found target image: {target_path}")
            else:
                logger.warning(f"Target image not found in embeddings: {target_path}")
        
        if not target_indices:
            raise ValueError("No target images found in the embeddings dataset")
            
        return target_indices
        
    def compute_target_embedding(self, target_indices: List[int], 
                               embeddings: np.ndarray) -> np.ndarray:
        """
        Compute target embedding by averaging embeddings of target images.
        
        Args:
            target_indices: Indices of target images
            embeddings: All embeddings array
            
        Returns:
            Target embedding (average if multiple targets)
        """
        target_embeddings = embeddings[target_indices]
        
        if len(target_embeddings) == 1:
            target_embedding = target_embeddings[0]
            logger.info("Using single target image embedding")
        else:
            # Average the embeddings and normalize
            target_embedding = np.mean(target_embeddings, axis=0)
            target_embedding = target_embedding / np.linalg.norm(target_embedding)
            logger.info(f"Averaged {len(target_embeddings)} target embeddings")
            
        self.target_embeddings = target_embedding
        return target_embedding
        
    def build_faiss_index(self, embeddings: np.ndarray, 
                         index_type: str) -> faiss.Index:
        """
        Build FAISS index for efficient similarity search.
        
        Args:
            embeddings: Embeddings array to index
            index_type: Type of FAISS index to build
            
        Returns:
            Built FAISS index
        """
        dimension = embeddings.shape[1]
        
        # Create FAISS index based on type
        if index_type == 'IndexFlatIP':
            # Inner Product (cosine similarity for normalized vectors)
            index = faiss.IndexFlatIP(dimension)
        elif index_type == 'IndexFlatL2':
            # L2 distance (Euclidean)
            index = faiss.IndexFlatL2(dimension)
        elif index_type == 'IndexIVFFlat':
            # IVF with flat quantizer (faster for large datasets)
            nlist = min(100, max(1, embeddings.shape[0] // 39))  # Rule of thumb
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings.astype(np.float32))
        elif index_type == 'IndexHNSW':
            # Hierarchical NSW for fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 64
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")
        
        # Add embeddings to index
        embeddings_float32 = embeddings.astype(np.float32)
        index.add(embeddings_float32)
        
        logger.info(f"Built FAISS {index_type} index with {index.ntotal} vectors")
        return index
        
    def search_similar_images_multi_target(self, index: faiss.Index,
                                         target_indices: List[int],
                                         embeddings: np.ndarray,
                                         image_paths: List[str],
                                         top_k: int,
                                         include_target: bool,
                                         strategy: str = 'weighted_union',
                                         individual_k: int = 50,
                                         weight_decay: float = 0.9) -> Tuple[List[str], List[float], List[int]]:
        """
        Advanced multi-target search that finds similar images for each target individually
        and combines results using different strategies.
        
        Args:
            index: Built FAISS index
            target_indices: Indices of target images
            embeddings: All embeddings array
            image_paths: List of all image paths
            top_k: Number of final results to return
            include_target: Whether to include target images in results
            strategy: How to combine results ('weighted_union', 'intersection', 'rank_fusion')
            individual_k: How many results to get per individual target
            weight_decay: Weight decay factor for multiple targets
            
        Returns:
            Tuple of (similar_image_paths, similarity_scores, indices)
        """
        logger.info(f"Using multi-target search strategy: {strategy}")
        
        # Get individual results for each target
        all_target_results = []
        target_weights = []
        
        for i, target_idx in enumerate(target_indices):
            target_embedding = embeddings[target_idx:target_idx+1].astype(np.float32)
            
            # Search for more results than needed for filtering
            search_k = individual_k + len(target_indices) + 10 if not include_target else individual_k
            search_k = min(search_k, index.ntotal)
            
            # Search using FAISS
            scores, indices = index.search(target_embedding, search_k)
            scores = scores[0]
            indices = indices[0]
            
            # Filter out target images if requested
            if not include_target:
                target_indices_set = set(target_indices)
                filtered_results = [(score, idx) for score, idx in zip(scores, indices) 
                                  if idx not in target_indices_set]
                target_scores = [score for score, _ in filtered_results[:individual_k]]
                target_result_indices = [idx for _, idx in filtered_results[:individual_k]]
            else:
                target_scores = scores[:individual_k]
                target_result_indices = indices[:individual_k].tolist() if hasattr(indices, 'tolist') else list(indices[:individual_k])
            
            all_target_results.append(list(zip(target_result_indices, target_scores)))
            # Weight decay: first target gets weight 1.0, second gets weight_decay, etc.
            target_weights.append(weight_decay ** i)
            
            logger.info(f"Target {i+1}: Found {len(target_result_indices)} candidates with avg score {np.mean(target_scores):.4f}")
        
        # Combine results based on strategy
        if strategy == 'weighted_union':
            final_results = self._combine_weighted_union(all_target_results, target_weights, top_k)
        elif strategy == 'intersection':
            final_results = self._combine_intersection(all_target_results, target_weights, top_k)
        elif strategy == 'rank_fusion':
            final_results = self._combine_rank_fusion(all_target_results, top_k)
        else:
            raise ValueError(f"Unknown multi-target strategy: {strategy}")
        
        # Extract paths and scores
        final_indices = [idx for idx, _ in final_results]
        final_scores = [score for _, score in final_results]
        similar_paths = [image_paths[idx] for idx in final_indices]
        
        logger.info(f"Multi-target search combined {len(all_target_results)} target results into {len(final_results)} final results")
        return similar_paths, final_scores, final_indices
    
    def _combine_weighted_union(self, all_results: List[List[Tuple[int, float]]], 
                               weights: List[float], top_k: int) -> List[Tuple[int, float]]:
        """Combine results using weighted union - images get scores from all targets that found them."""
        combined_scores = defaultdict(float)
        
        for target_results, weight in zip(all_results, weights):
            for idx, score in target_results:
                combined_scores[idx] += score * weight
        
        # Sort by combined score and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_results[:top_k]]
    
    def _combine_intersection(self, all_results: List[List[Tuple[int, float]]], 
                            weights: List[float], top_k: int) -> List[Tuple[int, float]]:
        """Combine results using intersection - only images found by multiple targets."""
        # Count how many targets found each image
        image_counts = defaultdict(int)
        image_scores = defaultdict(list)
        
        for target_results, weight in zip(all_results, weights):
            for idx, score in target_results:
                image_counts[idx] += 1
                image_scores[idx].append(score * weight)
        
        # Filter to images found by at least 2 targets (if we have multiple targets)
        min_targets = min(2, len(all_results))
        intersection_results = []
        
        for idx, count in image_counts.items():
            if count >= min_targets:
                # Use average score weighted by how many targets found it
                avg_score = np.mean(image_scores[idx]) * (count / len(all_results))
                intersection_results.append((idx, avg_score))
        
        # If intersection is too small, fall back to weighted union
        if len(intersection_results) < top_k // 2:
            logger.warning(f"Intersection too small ({len(intersection_results)}), falling back to weighted union")
            return self._combine_weighted_union(all_results, weights, top_k)
        
        # Sort and return top-k
        sorted_results = sorted(intersection_results, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _combine_rank_fusion(self, all_results: List[List[Tuple[int, float]]], 
                           top_k: int) -> List[Tuple[int, float]]:
        """Combine results using reciprocal rank fusion."""
        rrf_scores = defaultdict(float)
        
        for target_results in all_results:
            for rank, (idx, score) in enumerate(target_results):
                # Reciprocal rank fusion: 1 / (rank + 60)
                rrf_scores[idx] += 1.0 / (rank + 60)
        
        # Sort by RRF score and return top-k
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        # Convert RRF scores back to similarity-like scores (0-1 range)
        max_rrf = sorted_results[0][1] if sorted_results else 1.0
        normalized_results = [(idx, score / max_rrf) for idx, score in sorted_results[:top_k]]
        return normalized_results

    def search_similar_images(self, index: faiss.Index, 
                            target_embedding: np.ndarray,
                            image_paths: List[str], 
                            target_indices: List[int],
                            top_k: int, 
                            include_target: bool) -> Tuple[List[str], List[float], List[int]]:
        """
        Search for similar images using FAISS index.
        
        Args:
            index: Built FAISS index
            target_embedding: Target embedding vector
            image_paths: List of all image paths
            target_indices: Indices of target images to potentially exclude
            top_k: Number of top results to return
            include_target: Whether to include target images in results
            
        Returns:
            Tuple of (similar_image_paths, similarity_scores, indices)
        """
        # Check if we should use multi-target search
        if (self.config.get('use_multi_target_search', True) and 
            len(target_indices) > 1):
            return self.search_similar_images_multi_target(
                index, target_indices, self.embeddings, image_paths,
                top_k, include_target,
                strategy=self.config.get('multi_target_strategy', 'weighted_union'),
                individual_k=self.config.get('individual_search_k', 50),
                weight_decay=self.config.get('target_weight_decay', 0.9)
            )
        
        # Original single-target or averaged-target search
        # Prepare target embedding for FAISS search
        target_embedding = target_embedding.astype(np.float32).reshape(1, -1)
        
        # Search for more results than needed in case we filter out targets
        search_k = top_k + len(target_indices) + 10 if not include_target else top_k
        search_k = min(search_k, index.ntotal)
        
        # Search using FAISS
        scores, indices = index.search(target_embedding, search_k)
        scores = scores[0]  # Remove batch dimension
        indices = indices[0]  # Remove batch dimension
        
        # Filter out target images if requested
        if not include_target:
            target_indices_set = set(target_indices)
            filtered_results = [(score, idx) for score, idx in zip(scores, indices) 
                              if idx not in target_indices_set]
            scores = [score for score, _ in filtered_results[:top_k]]
            indices = [idx for _, idx in filtered_results[:top_k]]
            final_indices = indices  # Already a list
        else:
            scores = scores[:top_k]
            indices = indices[:top_k]
            final_indices = indices.tolist() if hasattr(indices, 'tolist') else list(indices)
        
        # Get corresponding paths
        similar_paths = [image_paths[idx] for idx in final_indices]
        similarity_scores = [float(score) for score in scores]
        
        logger.info(f"Found top {len(similar_paths)} similar images using FAISS")
        return similar_paths, similarity_scores, final_indices
        
    def copy_similar_images(self, similar_image_paths: List[str], 
                          similarity_scores: List[float], 
                          output_dir: Path) -> List[Dict[str, Any]]:
        """
        Copy similar images to output directory and create metadata.
        
        Args:
            similar_image_paths: List of paths to similar images
            similarity_scores: List of similarity scores
            output_dir: Output directory for copied images
            
        Returns:
            List of metadata dictionaries for each copied image
        """
        # Create images subdirectory
        images_dir = output_dir / "similar_images"
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"Permission denied creating directory: {images_dir}")
            raise
        
        copied_images_metadata = []
        
        for i, (image_path, score) in enumerate(zip(similar_image_paths, similarity_scores)):
            try:
                source_path = Path(image_path)
                if not source_path.exists():
                    logger.warning(f"Source image not found: {source_path}")
                    continue
                
                # Create destination filename with rank and score
                rank = i + 1
                extension = source_path.suffix
                dest_filename = f"rank_{rank:03d}_score_{score:.4f}_{source_path.name}"
                dest_path = images_dir / dest_filename
                
                # Copy the image using manual byte copy to avoid permission issues
                try:
                    # Ensure parent directory exists
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    # Manual copy using read/write bytes to bypass shutil permission issues
                    dest_path.write_bytes(source_path.read_bytes())
                except PermissionError as pe:
                    logger.error(f"Permission denied copying {source_path} to {dest_path}: {pe}")
                    raise
                
                # Create metadata
                metadata = {
                    'rank': rank,
                    'similarity_score': score,
                    'original_path': str(source_path),
                    'copied_path': str(dest_path),
                    'filename': dest_filename
                }
                copied_images_metadata.append(metadata)
                
            except Exception as e:
                logger.warning(f"Failed to copy image {image_path}: {e}")
                
        logger.info(f"Copied {len(copied_images_metadata)} similar images to {images_dir}")
        return copied_images_metadata
        
    def save_results(self, similar_image_paths: List[str], 
                    similarity_scores: List[float],
                    copied_images_metadata: List[Dict[str, Any]],
                    config: Dict[str, Any], 
                    run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Save similarity search results using standard output structure.
        
        Args:
            similar_image_paths: List of paths to similar images
            similarity_scores: List of similarity scores
            copied_images_metadata: Metadata for copied images
            config: Configuration dictionary
            run_id: Optional run identifier
            
        Returns:
            Dictionary with result metadata
        """
        # Use existing results manager (created earlier)
        results_dir = self.results_manager.get_results_dir()
        
        # Save similar images list
        similar_images_path = results_dir / "similar_images.json"
        similar_images_data = {
            'target_images': [str(path) for path in config['target_images']],
            'faiss_index_type': config['faiss_index_type'],
            'top_k': config['top_k'],
            'multi_target_search_enabled': config.get('use_multi_target_search', False),
            'multi_target_strategy': config.get('multi_target_strategy', 'weighted_union'),
            'pose_aware_similarity_enabled': config.get('use_pose_similarity', False),
            'pose_weight': config.get('pose_weight', 0.0),
            'embedding_weight': config.get('embedding_weight', 1.0),
            'pose_similarity_method': config.get('pose_similarity_method', 'euclidean'),
            'similar_images': [
                {
                    'path': path,
                    'similarity_score': score,
                    'rank': i + 1
                }
                for i, (path, score) in enumerate(zip(similar_image_paths, similarity_scores))
            ]
        }
        
        with open(similar_images_path, 'w') as f:
            json.dump(similar_images_data, f, indent=2)
            
        # Save copied images metadata
        copied_images_metadata_path = results_dir / "copied_images_metadata.json"
        with open(copied_images_metadata_path, 'w') as f:
            json.dump(copied_images_metadata, f, indent=2)
            
        # Save similarity metrics
        metrics_path = results_dir / "similarity_metrics.json"
        metrics = {
            'total_images_searched': len(self.image_paths),
            'target_images_count': len(config['target_images']),
            'similar_images_found': len(similar_image_paths),
            'faiss_index_type': config['faiss_index_type'],
            'multi_target_search_enabled': config.get('use_multi_target_search', False),
            'multi_target_strategy': config.get('multi_target_strategy', 'weighted_union'),
            'pose_aware_similarity_enabled': config.get('use_pose_similarity', False),
            'pose_images_loaded': len(self.pose_data) if self.pose_data else 0,
            'pose_weight': config.get('pose_weight', 0.0),
            'embedding_weight': config.get('embedding_weight', 1.0),
            'average_similarity_score': float(np.mean(similarity_scores)) if similarity_scores else 0.0,
            'max_similarity_score': float(np.max(similarity_scores)) if similarity_scores else 0.0,
            'min_similarity_score': float(np.min(similarity_scores)) if similarity_scores else 0.0,
            'std_similarity_score': float(np.std(similarity_scores)) if similarity_scores else 0.0
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Prepare additional metadata
        additional_metadata = {
            'total_images_searched': len(self.image_paths),
            'target_images_count': len(config['target_images']),
            'similar_images_found': len(similar_image_paths),
            'images_copied': len(copied_images_metadata),
            'faiss_index_type': config['faiss_index_type'],
            'multi_target_search_enabled': config.get('use_multi_target_search', False),
            'pose_aware_similarity_enabled': config.get('use_pose_similarity', False),
            'pose_images_loaded': len(self.pose_data) if self.pose_data else 0,
            'average_similarity_score': metrics['average_similarity_score'],
            'method': 'faiss_similarity_search_with_pose' if config.get('use_pose_similarity', False) else 'faiss_similarity_search'
        }
        
        # Save configuration with metadata
        config_path = self.results_manager.save_run_config(config, additional_metadata)
        
        # Create final result metadata
        result_metadata = self.results_manager.save_result_metadata({
            'total_images_searched': len(self.image_paths),
            'target_images_count': len(config['target_images']),
            'similar_images_found': len(similar_image_paths),
            'images_copied': len(copied_images_metadata),
            'faiss_index_type': config['faiss_index_type'],
            'average_similarity_score': metrics['average_similarity_score'],
            'files_created': [
                str(similar_images_path),
                str(copied_images_metadata_path),
                str(metrics_path),
                str(config_path)
            ]
        })
        
        logger.info(f"Results saved to: {self.results_manager.get_results_dir()}")
        logger.info(f"Found {len(similar_image_paths)} similar images with average score: {metrics['average_similarity_score']:.4f}")
        
        return result_metadata
        
    def find_similar_images(self, config_path: Union[str, Path], 
                          run_id: Optional[str] = None) -> str:
        """
        Main method to find similar images based on configuration.
        
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
            self.results_manager = create_service_runner('similarity', config['output_path'])
        
        # Create run directory
        actual_run_id = self.results_manager.create_run(run_id)
        
        # Load embeddings
        embeddings, image_paths = self.load_embeddings(config['embeddings_path'])
        
        # Load pose data if pose-aware similarity is enabled
        if config.get('use_pose_similarity', False):
            logger.info("Loading pose data for pose-aware similarity...")
            pose_data, pose_image_paths = self.load_pose_data(config['pose_data_path'])
            logger.info(f"Pose-aware similarity enabled with weights: embedding={config['embedding_weight']:.1f}, pose={config['pose_weight']:.1f}")
        
        # Find target images in embeddings
        target_indices = self.find_target_images(config['target_images'], image_paths)
        
        # Compute target embedding (average if multiple targets)
        target_embedding = self.compute_target_embedding(target_indices, embeddings)
        
        # Build FAISS index
        faiss_index = self.build_faiss_index(embeddings, config['faiss_index_type'])
        
        # Search for similar images using FAISS
        similar_image_paths, similarity_scores, top_indices = self.search_similar_images(
            faiss_index, target_embedding, image_paths, target_indices, 
            config['top_k'], config['include_target_in_results']
        )
        
        # Apply pose-aware reranking if enabled
        if config.get('use_pose_similarity', False):
            target_image_paths = [str(path) for path in config['target_images']]
            similar_image_paths, similarity_scores = self.rerank_with_pose_similarity(
                target_image_paths, similar_image_paths, similarity_scores
            )
        
        # Copy similar images to output directory
        copied_images_metadata = self.copy_similar_images(
            similar_image_paths, similarity_scores, 
            self.results_manager.get_results_dir()
        )
        
        # Save results
        result_metadata = self.save_results(
            similar_image_paths, similarity_scores, copied_images_metadata, config, run_id
        )
        
        return str(self.results_manager.get_results_dir())


def find_similar_images(config_path: Union[str, Path], run_id: Optional[str] = None) -> str:
    """
    Convenience function to find similar images using the SimilarityService.
    
    Args:
        config_path: Path to the YAML configuration file
        run_id: Optional run identifier
    
    Returns:
        Path to the generated results directory
    """
    service = SimilarityService()
    return service.find_similar_images(config_path, run_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find similar images using pre-computed embeddings")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--run-id", help="Optional run identifier")
    
    args = parser.parse_args()
    
    try:
        service = SimilarityService()
        results_path = service.find_similar_images(args.config, args.run_id)
        print(f"✅ Similarity search completed successfully!")
        print(f"   Results saved to: {results_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
