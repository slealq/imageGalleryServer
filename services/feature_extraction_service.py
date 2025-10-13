#!/usr/bin/env python3
"""
Feature Extraction Service

Comprehensive visual feature extraction using various computer vision algorithms.
Extracts edges, corners, textures, color features, and other visual descriptors
for similarity analysis and image understanding.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import logging
import cv2
import time
import gc
from tqdm import tqdm

# Import common services
from common_fs_service import (
    ResultsManager, 
    validate_config, 
    create_service_runner,
    logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Try to import scikit-image for advanced features
try:
    from skimage import feature, measure, segmentation, filters
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
    logger.info("Scikit-image library imported successfully")
except ImportError as e:
    logger.warning(f"Scikit-image library not found: {e}")
    logger.warning("Please install scikit-image: pip install scikit-image")
    SKIMAGE_AVAILABLE = False

# Try to import mahotas for additional texture features
try:
    import mahotas
    MAHOTAS_AVAILABLE = True
    logger.info("Mahotas library imported successfully")
except ImportError as e:
    logger.warning(f"Mahotas library not found: {e}")
    logger.warning("Please install mahotas: pip install mahotas")
    MAHOTAS_AVAILABLE = False


class FeatureExtractionService:
    """
    Comprehensive feature extraction service using multiple computer vision algorithms.
    
    Supported features:
    - Edge detection (Canny, Sobel, Laplacian)
    - Corner detection (Harris, FAST, ORB)
    - Texture analysis (LBP, GLCM, Haralick)
    - Color features (histograms, moments)
    - Shape descriptors (contours, moments)
    - Frequency domain features (FFT)
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        """Initialize the Feature Extraction Service."""
        self.results_manager = results_manager
        self.config = None
        
    def load_config_from_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from a dictionary (for testing)."""
        config = config_dict.copy()
        
        # Set defaults
        config.setdefault('feature_methods', ['canny'])  # Default to Canny only for efficiency
        config.setdefault('sample_percentage', 100)  # Process all images by default
        config.setdefault('save_feature_images', False)  # Disable for efficiency
        config.setdefault('visualization_sample_size', 10)  # Number of visualizations to save when enabled
        config.setdefault('image_resize', None)  # Resize images for processing (width, height)
        
        # Validate visualization settings
        if config.get('save_feature_images', False):
            sample_size = config.get('visualization_sample_size', 10)
            if sample_size <= 0:
                logger.warning("visualization_sample_size must be > 0 when save_feature_images is enabled. Setting to 10.")
                config['visualization_sample_size'] = 10
            elif sample_size > 100:
                logger.warning(f"visualization_sample_size ({sample_size}) is very large. Consider reducing for memory efficiency.")
            
            # Note: Visualizations will be generated on-demand even with store_as_vectors_only=True
            logger.info(f"📊 Visualization settings: save_feature_images=True, sample_size={sample_size}")
        else:
            logger.info("📊 Visualization settings: save_feature_images=False (no visualizations will be saved)")
        
        # Memory optimization settings
        config.setdefault('batch_size', 100)  # Process images in batches of 100
        config.setdefault('max_visualizations', 0)  # Disable visualizations for memory
        config.setdefault('force_gc_interval', 50)  # Force garbage collection every N images
        config.setdefault('store_as_vectors_only', True)  # Store only vectors, not full feature data
        
        # Canny edge detection settings
        config.setdefault('canny_low_threshold', 50)
        config.setdefault('canny_high_threshold', 150)
        config.setdefault('canny_aperture_size', 3)
        
        # Harris corner detection settings
        config.setdefault('harris_block_size', 2)
        config.setdefault('harris_ksize', 3)
        config.setdefault('harris_k', 0.04)
        config.setdefault('harris_threshold', 0.01)
        
        # LBP (Local Binary Patterns) settings
        config.setdefault('lbp_radius', 3)
        config.setdefault('lbp_n_points', 24)
        config.setdefault('lbp_method', "uniform")
        
        # Color histogram settings
        config.setdefault('hist_bins', 256)
        config.setdefault('hist_channels', [0, 1, 2])
        
        # GLCM settings
        config.setdefault('glcm_distances', [1])
        config.setdefault('glcm_angles', [0, 45, 90, 135])
        
        # ORB settings
        config.setdefault('orb_max_features', 500)
        config.setdefault('orb_scale_factor', 1.2)
        config.setdefault('orb_n_levels', 8)
        
        # HOG settings
        config.setdefault('hog_orientations', 9)
        config.setdefault('hog_pixels_per_cell', [8, 8])
        config.setdefault('hog_cells_per_block', [2, 2])
        
        self.config = config
        return config

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and validate configuration for feature extraction."""
        config = validate_config(config_path, 'feature_extraction')
        
        # Set defaults
        config.setdefault('feature_methods', ['canny'])  # Default to Canny only for efficiency
        config.setdefault('sample_percentage', 100)  # Process all images by default
        config.setdefault('save_feature_images', False)  # Disable for efficiency
        config.setdefault('visualization_sample_size', 10)  # Number of visualizations to save when enabled
        config.setdefault('image_resize', None)  # Resize images for processing (width, height)
        
        # Validate visualization settings
        if config.get('save_feature_images', False):
            sample_size = config.get('visualization_sample_size', 10)
            if sample_size <= 0:
                logger.warning("visualization_sample_size must be > 0 when save_feature_images is enabled. Setting to 10.")
                config['visualization_sample_size'] = 10
            elif sample_size > 100:
                logger.warning(f"visualization_sample_size ({sample_size}) is very large. Consider reducing for memory efficiency.")
            
            # Note: Visualizations will be generated on-demand even with store_as_vectors_only=True
            logger.info(f"📊 Visualization settings: save_feature_images=True, sample_size={sample_size}")
        else:
            logger.info("📊 Visualization settings: save_feature_images=False (no visualizations will be saved)")
        
        # Memory optimization settings
        config.setdefault('batch_size', 100)  # Process images in batches of 100
        config.setdefault('max_visualizations', 0)  # Disable visualizations for memory
        config.setdefault('force_gc_interval', 50)  # Force garbage collection every N images
        config.setdefault('store_as_vectors_only', True)  # Store only vectors, not full data
        
        # Canny edge detection settings
        config.setdefault('canny_low_threshold', 50)
        config.setdefault('canny_high_threshold', 150)
        config.setdefault('canny_aperture_size', 3)
        
        # Harris corner detection settings
        config.setdefault('harris_block_size', 2)
        config.setdefault('harris_ksize', 3)
        config.setdefault('harris_k', 0.04)
        config.setdefault('harris_threshold', 0.01)
        
        # LBP (Local Binary Patterns) settings
        config.setdefault('lbp_radius', 3)
        config.setdefault('lbp_n_points', 24)
        config.setdefault('lbp_method', 'uniform')
        
        # Color histogram settings
        config.setdefault('hist_bins', 256)
        config.setdefault('hist_channels', [0, 1, 2])  # RGB channels
        
        # GLCM (Gray-Level Co-occurrence Matrix) settings
        config.setdefault('glcm_distances', [1])
        config.setdefault('glcm_angles', [0, 45, 90, 135])
        
        # ORB keypoint settings
        config.setdefault('orb_max_features', 500)
        config.setdefault('orb_scale_factor', 1.2)
        config.setdefault('orb_n_levels', 8)
        
        # HOG (Histogram of Oriented Gradients) settings
        config.setdefault('hog_orientations', 9)
        config.setdefault('hog_pixels_per_cell', [8, 8])
        config.setdefault('hog_cells_per_block', [2, 2])
        
        # Convert paths
        config['embeddings_path'] = Path(config['embeddings_path'])
        config['output_path'] = Path(config['output_path'])
        
        self.config = config
        logger.info(f"Configuration loaded from {config_path}")
        logger.info(f"Feature methods: {config['feature_methods']}")
        logger.info(f"Store as vectors only: {config['store_as_vectors_only']}")
        logger.info(f"Memory optimization - Batch size: {config['batch_size']}, Max visualizations: {config['max_visualizations']}")
        return config
    
    def load_embeddings(self, embeddings_path: Path) -> Tuple[np.ndarray, List[str]]:
        """Load pre-computed embeddings and optionally sample a subset."""
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        image_paths = data['image_paths'].tolist()
        
        total_images = len(image_paths)
        logger.info(f"Loaded {total_images} embeddings from {embeddings_path}")
        
        # Sample subset if specified
        sample_percentage = float(self.config.get('sample_percentage', 100))
        if sample_percentage < 100:
            import random
            
            # Calculate number of samples (handle float percentages like 0.1)
            num_samples = int(total_images * sample_percentage / 100)
            num_samples = max(1, num_samples)  # Ensure at least 1 sample
            
            # Create random indices
            random.seed(42)  # For reproducible sampling
            indices = random.sample(range(total_images), num_samples)
            indices.sort()  # Keep original order
            
            # Sample embeddings and paths
            embeddings = embeddings[indices]
            image_paths = [image_paths[i] for i in indices]
            
            logger.info(f"📊 Sampling {sample_percentage}% of images: {len(image_paths)}/{total_images} images")
        else:
            logger.info(f"📊 Processing all {total_images} images")
        
        return embeddings, image_paths
    
    def extract_canny_edges(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract Canny edge features with spatial information preserved."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        low_threshold = self.config.get('canny_low_threshold', 50)
        high_threshold = self.config.get('canny_high_threshold', 150)
        aperture_size = self.config.get('canny_aperture_size', 3)
        
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=aperture_size)
        
        # Original global statistics (for backward compatibility)
        edge_density = np.sum(edges > 0) / edges.size
        edge_magnitude = np.sum(edges) / 255.0 / edges.size
        
        # Edge direction histogram (global)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_angles = np.arctan2(sobel_y, sobel_x)
        edge_hist, _ = np.histogram(edge_angles[edges > 0], bins=8, range=(-np.pi, np.pi))
        edge_hist = edge_hist / np.sum(edge_hist) if np.sum(edge_hist) > 0 else edge_hist
        
        # NEW: Create spatial edge features that preserve shape information
        
        # 1. Resize edge map to standard size for consistent spatial features
        target_size = (64, 64)  # Reasonable size for spatial features
        edges_resized = cv2.resize(edges, target_size) / 255.0  # Normalize to [0,1]
        
        # 2. Grid-based edge features (divide image into blocks)
        grid_size = 8  # 8x8 grid = 64 spatial features
        h, w = edges_resized.shape
        block_h, block_w = h // grid_size, w // grid_size
        
        spatial_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * block_h, (i + 1) * block_h
                x1, x2 = j * block_w, (j + 1) * block_w
                block = edges_resized[y1:y2, x1:x2]
                
                # Edge density in this spatial block
                block_density = np.mean(block)
                spatial_features.append(block_density)
        
        # 3. Radial edge features (from center outward)
        center_y, center_w = target_size[0] // 2, target_size[1] // 2
        radial_features = []
        
        # Create radial zones
        max_radius = min(center_y, center_w)
        n_radial_zones = 8
        
        for r in range(n_radial_zones):
            r_inner = (r * max_radius) // n_radial_zones
            r_outer = ((r + 1) * max_radius) // n_radial_zones
            
            # Create mask for this radial zone
            y_coords, x_coords = np.ogrid[:target_size[0], :target_size[1]]
            distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_w)**2)
            mask = (distances >= r_inner) & (distances < r_outer)
            
            if np.sum(mask) > 0:
                zone_density = np.mean(edges_resized[mask])
            else:
                zone_density = 0.0
            
            radial_features.append(zone_density)
        
        # 4. Directional edge features (preserve edge orientations spatially)
        # Divide into angular sectors and measure edge content
        n_angular_sectors = 8
        angular_features = []
        
        for sector in range(n_angular_sectors):
            angle_start = (sector * 2 * np.pi) / n_angular_sectors - np.pi
            angle_end = ((sector + 1) * 2 * np.pi) / n_angular_sectors - np.pi
            
            # Create mask for pixels in this angular direction
            y_coords, x_coords = np.ogrid[:target_size[0], :target_size[1]]
            angles = np.arctan2(y_coords - center_y, x_coords - center_w)
            
            # Handle angle wrapping
            if angle_end > np.pi:
                mask = (angles >= angle_start) | (angles <= angle_end - 2*np.pi)
            else:
                mask = (angles >= angle_start) & (angles < angle_end)
            
            if np.sum(mask) > 0:
                sector_density = np.mean(edges_resized[mask])
            else:
                sector_density = 0.0
                
            angular_features.append(sector_density)
        
        # 5. Edge contour features (find and characterize major contours)
        contours, _ = cv2.findContours((edges > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_features = []
        if contours:
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Features from largest contours (up to 3)
            for i in range(min(3, len(contours))):
                contour = contours[i]
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    # Compactness (circularity measure)
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    compactness = 0.0
                
                # Relative area
                relative_area = area / (gray.shape[0] * gray.shape[1])
                
                contour_features.extend([relative_area, compactness])
            
            # Pad with zeros if fewer than 3 contours
            while len(contour_features) < 6:  # 3 contours * 2 features each
                contour_features.append(0.0)
        else:
            contour_features = [0.0] * 6
        
        return {
            'method': 'canny',
            'edge_map': edges,
            'edge_density': float(edge_density),
            'edge_magnitude': float(edge_magnitude),
            'edge_direction_histogram': edge_hist.tolist(),
            # NEW: Spatial edge features
            'spatial_features': spatial_features,  # 64 values (8x8 grid)
            'radial_features': radial_features,    # 8 values (radial zones)
            'angular_features': angular_features,  # 8 values (angular sectors)
            'contour_features': contour_features,  # 6 values (top 3 contours)
            'parameters': {
                'low_threshold': low_threshold,
                'high_threshold': high_threshold,
                'aperture_size': aperture_size,
                'spatial_grid_size': grid_size,
                'target_size': target_size
            }
        }
    
    def extract_harris_corners(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract Harris corner features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray = np.float32(gray)
        
        block_size = self.config.get('harris_block_size', 2)
        ksize = self.config.get('harris_ksize', 3)
        k = self.config.get('harris_k', 0.04)
        threshold = self.config.get('harris_threshold', 0.01)
        
        # Detect Harris corners
        harris_response = cv2.cornerHarris(gray, block_size, ksize, k)
        
        # Find corner locations
        corner_mask = harris_response > threshold * harris_response.max()
        corners = np.argwhere(corner_mask)
        
        # Calculate corner statistics
        corner_count = len(corners)
        corner_density = corner_count / (image.shape[0] * image.shape[1])
        corner_strength = np.mean(harris_response[corner_mask]) if corner_count > 0 else 0.0
        
        # Corner distribution (divide image into grid and count corners per region)
        h, w = image.shape[:2]
        grid_size = 4
        corner_distribution = np.zeros((grid_size, grid_size))
        
        if corner_count > 0:
            for corner in corners:
                grid_y = min(int(corner[0] / h * grid_size), grid_size - 1)
                grid_x = min(int(corner[1] / w * grid_size), grid_size - 1)
                corner_distribution[grid_y, grid_x] += 1
        
        return {
            'method': 'harris',
            'corner_map': harris_response,
            'corner_locations': corners.tolist(),
            'corner_count': corner_count,
            'corner_density': float(corner_density),
            'corner_strength': float(corner_strength),
            'corner_distribution': corner_distribution.flatten().tolist(),
            'parameters': {
                'block_size': block_size,
                'ksize': ksize,
                'k': k,
                'threshold': threshold
            }
        }
    
    def extract_lbp_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract Local Binary Pattern texture features."""
        if not SKIMAGE_AVAILABLE:
            logger.warning("Scikit-image not available, skipping LBP extraction")
            return {'method': 'lbp', 'error': 'scikit-image not available'}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        radius = self.config.get('lbp_radius', 3)
        n_points = self.config.get('lbp_n_points', 24)
        method = self.config.get('lbp_method', 'uniform')
        
        # Calculate LBP
        lbp = feature.local_binary_pattern(gray, n_points, radius, method=method)
        
        # Calculate LBP histogram
        if method == 'uniform':
            n_bins = n_points + 2
        else:
            n_bins = 2 ** n_points
        
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize
        
        # Calculate texture statistics
        lbp_variance = np.var(lbp)
        lbp_uniformity = np.sum(lbp_hist ** 2)  # Uniformity measure
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))  # Entropy
        
        return {
            'method': 'lbp',
            'lbp_map': lbp,
            'lbp_histogram': lbp_hist.tolist(),
            'lbp_variance': float(lbp_variance),
            'lbp_uniformity': float(lbp_uniformity),
            'lbp_entropy': float(lbp_entropy),
            'parameters': {
                'radius': radius,
                'n_points': n_points,
                'method': method
            }
        }
    
    def extract_color_histogram(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract color histogram features."""
        bins = self.config.get('hist_bins', 256)
        channels = self.config.get('hist_channels', [0, 1, 2])
        
        # Convert to RGB if BGR
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        color_histograms = {}
        combined_histogram = []
        
        for i, channel in enumerate(channels):
            if len(image_rgb.shape) == 3 and channel < image_rgb.shape[2]:
                hist, _ = np.histogram(image_rgb[:, :, channel], bins=bins, range=(0, 256))
                hist = hist / np.sum(hist)  # Normalize
                color_histograms[f'channel_{channel}'] = hist.tolist()
                combined_histogram.extend(hist)
            elif len(image_rgb.shape) == 2 and channel == 0:
                hist, _ = np.histogram(image_rgb, bins=bins, range=(0, 256))
                hist = hist / np.sum(hist)  # Normalize
                color_histograms['grayscale'] = hist.tolist()
                combined_histogram.extend(hist)
        
        # Color moments
        color_moments = {}
        if len(image_rgb.shape) == 3:
            for i, channel in enumerate(channels):
                if channel < image_rgb.shape[2]:
                    channel_data = image_rgb[:, :, channel].flatten()
                    color_moments[f'channel_{channel}'] = {
                        'mean': float(np.mean(channel_data)),
                        'std': float(np.std(channel_data)),
                        'skewness': float(np.mean(((channel_data - np.mean(channel_data)) / np.std(channel_data)) ** 3)),
                        'kurtosis': float(np.mean(((channel_data - np.mean(channel_data)) / np.std(channel_data)) ** 4))
                    }
        
        return {
            'method': 'color_histogram',
            'histograms': color_histograms,
            'combined_histogram': combined_histogram,
            'color_moments': color_moments,
            'parameters': {
                'bins': bins,
                'channels': channels
            }
        }
    
    def extract_glcm_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract GLCM (Gray-Level Co-occurrence Matrix) texture features."""
        if not SKIMAGE_AVAILABLE:
            logger.warning("Scikit-image not available, skipping GLCM extraction")
            return {'method': 'glcm', 'error': 'scikit-image not available'}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        distances = self.config.get('glcm_distances', [1])
        angles = [np.radians(angle) for angle in self.config.get('glcm_angles', [0, 45, 90, 135])]
        
        # Calculate GLCM
        glcm = graycomatrix(gray, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Calculate GLCM properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        glcm_features = {}
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            glcm_features[prop] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values.flatten().tolist()
            }
        
        return {
            'method': 'glcm',
            'glcm_features': glcm_features,
            'parameters': {
                'distances': distances,
                'angles': self.config.get('glcm_angles', [0, 45, 90, 135])
            }
        }
    
    def extract_orb_keypoints(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract ORB keypoint features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        max_features = self.config.get('orb_max_features', 500)
        scale_factor = self.config.get('orb_scale_factor', 1.2)
        n_levels = self.config.get('orb_n_levels', 8)
        
        # Create ORB detector
        orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=scale_factor, nlevels=n_levels)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        # Extract keypoint information
        keypoint_count = len(keypoints)
        keypoint_data = []
        
        for kp in keypoints:
            keypoint_data.append({
                'x': float(kp.pt[0]),
                'y': float(kp.pt[1]),
                'size': float(kp.size),
                'angle': float(kp.angle),
                'response': float(kp.response)
            })
        
        # Calculate keypoint statistics
        if keypoint_count > 0:
            sizes = [kp['size'] for kp in keypoint_data]
            responses = [kp['response'] for kp in keypoint_data]
            
            keypoint_stats = {
                'count': keypoint_count,
                'density': keypoint_count / (image.shape[0] * image.shape[1]),
                'avg_size': float(np.mean(sizes)),
                'avg_response': float(np.mean(responses)),
                'size_std': float(np.std(sizes)),
                'response_std': float(np.std(responses))
            }
        else:
            keypoint_stats = {
                'count': 0,
                'density': 0.0,
                'avg_size': 0.0,
                'avg_response': 0.0,
                'size_std': 0.0,
                'response_std': 0.0
            }
        
        return {
            'method': 'orb',
            'keypoints': keypoint_data,
            'descriptors': descriptors.tolist() if descriptors is not None else [],
            'keypoint_stats': keypoint_stats,
            'parameters': {
                'max_features': max_features,
                'scale_factor': scale_factor,
                'n_levels': n_levels
            }
        }
    
    def extract_hog_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract HOG (Histogram of Oriented Gradients) features."""
        if not SKIMAGE_AVAILABLE:
            logger.warning("Scikit-image not available, skipping HOG extraction")
            return {'method': 'hog', 'error': 'scikit-image not available'}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        orientations = self.config.get('hog_orientations', 9)
        pixels_per_cell = tuple(self.config.get('hog_pixels_per_cell', [8, 8]))
        cells_per_block = tuple(self.config.get('hog_cells_per_block', [2, 2]))
        
        # Extract HOG features
        hog_features, hog_image = feature.hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            feature_vector=True
        )
        
        # Calculate HOG statistics
        hog_mean = float(np.mean(hog_features))
        hog_std = float(np.std(hog_features))
        hog_energy = float(np.sum(hog_features ** 2))
        
        return {
            'method': 'hog',
            'hog_features': hog_features.tolist(),
            'hog_image': hog_image,
            'hog_statistics': {
                'mean': hog_mean,
                'std': hog_std,
                'energy': hog_energy,
                'feature_length': len(hog_features)
            },
            'parameters': {
                'orientations': orientations,
                'pixels_per_cell': pixels_per_cell,
                'cells_per_block': cells_per_block
            }
        }
    
    def extract_haralick_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract Haralick texture features."""
        if not MAHOTAS_AVAILABLE:
            logger.warning("Mahotas not available, skipping Haralick extraction")
            return {'method': 'haralick', 'error': 'mahotas not available'}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        try:
            # Extract Haralick features
            haralick_features = mahotas.features.haralick(gray)
            
            # Haralick returns features for 4 directions, we'll average them
            haralick_mean = np.mean(haralick_features, axis=0)
            haralick_std = np.std(haralick_features, axis=0)
            
            feature_names = [
                'angular_second_moment', 'contrast', 'correlation', 'variance',
                'inverse_diff_moment', 'sum_average', 'sum_variance', 'sum_entropy',
                'entropy', 'diff_variance', 'diff_entropy', 'info_measure_1',
                'info_measure_2'
            ]
            
            haralick_dict = {}
            for i, name in enumerate(feature_names):
                haralick_dict[name] = {
                    'mean': float(haralick_mean[i]),
                    'std': float(haralick_std[i])
                }
            
            return {
                'method': 'haralick',
                'haralick_features': haralick_dict,
                'haralick_vector': haralick_mean.tolist(),
                'parameters': {}
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract Haralick features: {e}")
            return {'method': 'haralick', 'error': str(e)}
    
    def extract_features_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract all configured features from a single image with memory optimization."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize image if specified
            resize_dims = self.config.get('image_resize')
            if resize_dims:
                width, height = resize_dims
                image = cv2.resize(image, (width, height))
            
            # Check if we only need vectors (for efficiency)
            store_vectors_only = self.config.get('store_as_vectors_only', False)
            
            if store_vectors_only:
                # Ultra-lightweight: only extract feature vectors
                feature_data = {
                    'image_path': image_path,
                    'feature_vectors': {},
                    'has_features': False
                }
            else:
                # Full feature extraction (legacy mode)
                feature_data = {
                    'image_path': image_path,
                    'image_shape': image.shape,
                    'features': {},
                    'feature_vectors': {},
                    'has_features': False
                }
            
            feature_methods = self.config.get('feature_methods', ['canny'])
            
            for method in feature_methods:
                try:
                    if method == 'canny':
                        result = self.extract_canny_edges(image)
                    elif method == 'harris':
                        result = self.extract_harris_corners(image)
                    elif method == 'lbp':
                        result = self.extract_lbp_texture(image)
                    elif method == 'color_histogram':
                        result = self.extract_color_histogram(image)
                    elif method == 'glcm':
                        result = self.extract_glcm_texture(image)
                    elif method == 'orb':
                        result = self.extract_orb_keypoints(image)
                    elif method == 'hog':
                        result = self.extract_hog_features(image)
                    elif method == 'haralick':
                        result = self.extract_haralick_texture(image)
                    else:
                        logger.warning(f"Unknown feature method: {method}")
                        continue
                    
                    if 'error' not in result:
                        if not store_vectors_only:
                            # Store full result but remove large arrays to save memory
                            result_clean = result.copy()
                            # Remove large numpy arrays immediately after processing
                            for key in ['edge_map', 'corner_map', 'lbp_map', 'hog_image']:
                                result_clean.pop(key, None)
                            feature_data['features'][method] = result_clean
                        
                        feature_data['has_features'] = True
                        
                        # Extract feature vector for similarity comparison
                        if method == 'canny':
                            # NEW: Use spatial features instead of just global statistics
                            if 'spatial_features' in result:
                                # Enhanced spatial Canny features (better for shape similarity)
                                feature_data['feature_vectors'][method] = (
                                    result['spatial_features'] +      # 64 spatial grid features
                                    result['radial_features'] +       # 8 radial zone features  
                                    result['angular_features'] +      # 8 angular sector features
                                    result['contour_features']        # 6 contour features
                                    # Total: 86 dimensions with spatial information
                                )
                            else:
                                # Fallback to original global features (backward compatibility)
                                feature_data['feature_vectors'][method] = [
                                    result['edge_density'], result['edge_magnitude']
                                ] + result['edge_direction_histogram']
                        elif method == 'harris':
                            feature_data['feature_vectors'][method] = [
                                result['corner_density'], result['corner_strength']
                            ] + result['corner_distribution']
                        elif method == 'lbp':
                            feature_data['feature_vectors'][method] = result['lbp_histogram']
                        elif method == 'color_histogram':
                            feature_data['feature_vectors'][method] = result['combined_histogram']
                        elif method == 'glcm':
                            # Combine mean values of all GLCM properties
                            vector = []
                            for prop_data in result['glcm_features'].values():
                                vector.append(prop_data['mean'])
                            feature_data['feature_vectors'][method] = vector
                        elif method == 'orb':
                            stats = result['keypoint_stats']
                            feature_data['feature_vectors'][method] = [
                                stats['density'], stats['avg_size'], stats['avg_response'],
                                stats['size_std'], stats['response_std']
                            ]
                        elif method == 'hog':
                            feature_data['feature_vectors'][method] = result['hog_features']
                        elif method == 'haralick':
                            feature_data['feature_vectors'][method] = result['haralick_vector']
                    
                except Exception as e:
                    logger.warning(f"Failed to extract {method} features from {image_path}: {e}")
                    if not store_vectors_only:
                        feature_data['features'][method] = {'method': method, 'error': str(e)}
            
            return feature_data
            
        except Exception as e:
            logger.warning(f"Failed to extract features from {image_path}: {e}")
            store_vectors_only = self.config.get('store_as_vectors_only', False)
            if store_vectors_only:
                return {
                    'image_path': image_path,
                    'feature_vectors': {},
                    'has_features': False,
                    'error': str(e)
                }
            else:
                return {
                    'image_path': image_path,
                    'features': {},
                    'feature_vectors': {},
                    'has_features': False,
                    'error': str(e)
                }
    
    def save_batch_results(self, batch_features: List[Dict[str, Any]], batch_idx: int) -> str:
        """Save a batch of features to disk to free up memory."""
        results_dir = self.results_manager.get_results_dir()
        batch_dir = results_dir / "batches"
        batch_dir.mkdir(exist_ok=True)
        
        store_vectors_only = self.config.get('store_as_vectors_only', False)
        
        if store_vectors_only:
            # Save as efficient .npz format (binary, compressed)
            batch_file = batch_dir / f"batch_{batch_idx:04d}.npz"
            
            # Extract vectors and paths
            image_paths = []
            feature_vectors = {}
            feature_methods = self.config.get('feature_methods', ['canny'])
            
            for feature_data in batch_features:
                if feature_data.get('has_features', False):
                    image_paths.append(feature_data['image_path'])
                    
                    for method in feature_methods:
                        if method in feature_data.get('feature_vectors', {}):
                            if method not in feature_vectors:
                                feature_vectors[method] = []
                            feature_vectors[method].append(feature_data['feature_vectors'][method])
            
            # Save as numpy arrays
            save_data = {'image_paths': image_paths}
            for method, vectors in feature_vectors.items():
                if vectors:  # Only save if we have vectors
                    # Ensure all vectors have same length
                    max_length = max(len(v) for v in vectors) if vectors else 0
                    if max_length > 0:
                        padded_vectors = []
                        for v in vectors:
                            if len(v) < max_length:
                                padded_v = v + [0.0] * (max_length - len(v))
                            else:
                                padded_v = v[:max_length]
                            padded_vectors.append(padded_v)
                        save_data[f'{method}_vectors'] = np.array(padded_vectors)
            
            np.savez_compressed(batch_file, **save_data)
            
        else:
            # Legacy mode: save as JSON
            batch_file = batch_dir / f"batch_{batch_idx:04d}.json"
            with open(batch_file, 'w') as f:
                json.dump(batch_features, f, indent=2, default=str)
        
        return str(batch_file)
    
    def extract_features_from_images_batched(self, image_paths: List[str]) -> Tuple[List[str], int]:
        """Extract features from all images using batched processing to manage memory."""
        batch_size = self.config.get('batch_size', 100)
        force_gc_interval = self.config.get('force_gc_interval', 50)
        
        batch_files = []
        successful_extractions = 0
        total_processed = 0
        
        feature_methods = self.config.get('feature_methods', ['canny'])
        logger.info(f"🎯 Processing {len(image_paths)} images in batches of {batch_size} with methods: {feature_methods}")
        
        # Process images in batches
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        with tqdm(total=len(image_paths), desc="Extracting features", unit="img") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(image_paths))
                batch_paths = image_paths[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_paths)} images)")
                
                batch_features = []
                batch_successful = 0
                
                for i, image_path in enumerate(batch_paths):
                    feature_data = self.extract_features_from_image(image_path)
                    
                    if feature_data['has_features']:
                        successful_extractions += 1
                        batch_successful += 1
                    
                    batch_features.append(feature_data)
                    total_processed += 1
                    
                    # Update progress
                    success_rate = (successful_extractions / total_processed) * 100
                    pbar.set_postfix({
                        'Success': f"{successful_extractions}/{total_processed}",
                        'Rate': f"{success_rate:.1f}%",
                        'Batch': f"{batch_idx + 1}/{num_batches}"
                    })
                    pbar.update(1)
                    
                    # Force garbage collection periodically
                    if (total_processed % force_gc_interval) == 0:
                        gc.collect()
                
                # Save batch to disk and free memory
                batch_file = self.save_batch_results(batch_features, batch_idx)
                batch_files.append(batch_file)
                
                logger.info(f"Batch {batch_idx + 1} completed: {batch_successful}/{len(batch_paths)} successful, saved to {batch_file}")
                
                # Clear batch from memory
                del batch_features
                gc.collect()
        
        logger.info(f"✅ Successfully extracted features from {successful_extractions}/{len(image_paths)} images")
        return batch_files, successful_extractions
    
    def consolidate_batch_results(self, batch_files: List[str]) -> List[Dict[str, Any]]:
        """Load and consolidate all batch results."""
        logger.info(f"Consolidating {len(batch_files)} batch files...")
        
        store_vectors_only = self.config.get('store_as_vectors_only', False)
        all_features = []
        
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            try:
                if batch_file.endswith('.npz'):
                    # Load .npz file (efficient binary format)
                    data = np.load(batch_file)
                    image_paths = data['image_paths'].tolist()
                    
                    # Load feature vectors for each method
                    feature_methods = self.config.get('feature_methods', ['canny'])
                    method_vectors = {}
                    for method in feature_methods:
                        if f'{method}_vectors' in data:
                            method_vectors[method] = data[f'{method}_vectors']  # Keep as numpy array
                    
                    # Reconstruct individual feature data for each image
                    for i, image_path in enumerate(image_paths):
                        feature_data = {
                            'image_path': image_path,
                            'feature_vectors': {},
                            'has_features': True
                        }
                        
                        # Extract vectors for this specific image
                        for method, vectors in method_vectors.items():
                            if i < len(vectors):
                                feature_data['feature_vectors'][method] = vectors[i].tolist()
                        
                        all_features.append(feature_data)
                else:
                    # Load JSON file (legacy mode)
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                        all_features.extend(batch_data)
            except Exception as e:
                logger.warning(f"Failed to load batch file {batch_file}: {e}")
        
        return all_features
    
    def save_feature_visualization(self, image_path: str, feature_data: Dict[str, Any], 
                                 output_dir: Path) -> Optional[str]:
        """Save feature visualization by re-extracting visual data on-demand."""
        if not feature_data['has_features']:
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            # Re-extract visual data for visualization (memory efficient)
            visual_features = self._extract_visual_features_for_visualization(image_path)
            if not visual_features:
                logger.warning(f"Could not re-extract visual features for {image_path}")
                return None
            
            features = visual_features
            n_features = len([f for f in features.values() if 'error' not in f])
            
            if n_features == 0:
                return None
            
            # Create subplot grid
            cols = min(4, n_features + 1)  # +1 for original image
            rows = ((n_features + 1) + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Load original image for reference
            original_image = cv2.imread(image_path)
            if original_image is None:
                logger.warning(f"Could not load image for visualization: {image_path}")
                return None
                
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            plot_idx = 0
            
            # Show original image first
            if plot_idx < len(axes):
                axes[plot_idx].imshow(original_image_rgb)
                axes[plot_idx].set_title("Original Image")
                axes[plot_idx].axis('off')
                plot_idx += 1
            
            for method, feature_result in features.items():
                if plot_idx >= len(axes):
                    break
                    
                if 'error' in feature_result:
                    continue
                    
                ax = axes[plot_idx]
                
                try:
                    if method == 'canny':
                        if 'edge_map' in feature_result:
                            ax.imshow(feature_result['edge_map'], cmap='gray')
                            density = feature_result.get('edge_density', 0)
                            ax.set_title(f"Canny Edges\nDensity: {density:.3f}")
                        else:
                            ax.text(0.5, 0.5, 'Canny edges\n(data missing)', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title("Canny Edges (Error)")
                        
                    elif method == 'harris':
                        if 'corner_map' in feature_result:
                            corner_map = feature_result['corner_map']
                            # Normalize for better visualization
                            corner_map_norm = cv2.normalize(corner_map, None, 0, 255, cv2.NORM_MINMAX)
                            ax.imshow(corner_map_norm, cmap='hot')
                            count = feature_result.get('corner_count', 0)
                            ax.set_title(f"Harris Corners\nCount: {count}")
                            
                            # Overlay corner points if available
                            if 'corner_locations' in feature_result:
                                corners = np.array(feature_result['corner_locations'])
                                if len(corners) > 0:
                                    ax.scatter(corners[:, 1], corners[:, 0], 
                                             c='cyan', s=10, marker='x', alpha=0.8)
                        else:
                            ax.text(0.5, 0.5, 'Harris corners\n(data missing)', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title("Harris Corners (Error)")
                        
                    elif method == 'lbp':
                        if 'lbp_map' in feature_result:
                            ax.imshow(feature_result['lbp_map'], cmap='gray')
                            uniformity = feature_result.get('lbp_uniformity', 0)
                            ax.set_title(f"LBP Texture\nUniformity: {uniformity:.3f}")
                        else:
                            ax.text(0.5, 0.5, 'LBP texture\n(data missing)', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title("LBP Texture (Error)")
                        
                    elif method == 'color_histogram':
                        if 'histograms' in feature_result:
                            # Plot color histograms
                            histograms = feature_result['histograms']
                            colors = ['red', 'green', 'blue', 'gray']
                            color_names = ['Red', 'Green', 'Blue', 'Grayscale']
                            
                            plot_count = 0
                            for i, (channel, hist) in enumerate(histograms.items()):
                                if plot_count < len(colors):
                                    ax.plot(hist, color=colors[plot_count], alpha=0.7, 
                                           label=color_names[plot_count] if plot_count < len(color_names) else channel)
                                    plot_count += 1
                            
                            ax.set_title("Color Histograms")
                            ax.set_xlabel("Intensity")
                            ax.set_ylabel("Frequency")
                            ax.legend()
                        else:
                            ax.text(0.5, 0.5, 'Color histogram\n(data missing)', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title("Color Histogram (Error)")
                    
                    elif method == 'glcm':
                        # Show GLCM features as bar chart
                        if 'glcm_features' in feature_result:
                            features_dict = feature_result['glcm_features']
                            feature_names = list(features_dict.keys())
                            feature_values = [features_dict[name]['mean'] for name in feature_names]
                            
                            bars = ax.bar(range(len(feature_names)), feature_values)
                            ax.set_title("GLCM Texture Features")
                            ax.set_xlabel("Feature")
                            ax.set_ylabel("Value")
                            ax.set_xticks(range(len(feature_names)))
                            ax.set_xticklabels([name[:8] for name in feature_names], rotation=45, ha='right')
                        else:
                            ax.text(0.5, 0.5, 'GLCM features\n(data missing)', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title("GLCM Features (Error)")
                    
                    elif method == 'orb':
                        # Show original image with ORB keypoints overlay
                        ax.imshow(original_image_rgb)
                        if 'keypoints' in feature_result:
                            keypoints = feature_result['keypoints']
                            if keypoints:
                                x_coords = [kp['x'] for kp in keypoints]
                                y_coords = [kp['y'] for kp in keypoints]
                                sizes = [kp['size'] for kp in keypoints]
                                
                                # Scale sizes for visualization
                                scaled_sizes = [(s * 2) for s in sizes]
                                ax.scatter(x_coords, y_coords, s=scaled_sizes, 
                                         c='yellow', alpha=0.7, marker='o', edgecolors='red', linewidth=1)
                            
                            count = len(keypoints)
                            ax.set_title(f"ORB Keypoints\nCount: {count}")
                        else:
                            ax.set_title("ORB Keypoints (Error)")
                    
                    elif method == 'hog':
                        if 'hog_image' in feature_result:
                            # Normalize HOG image for better visualization
                            hog_image = feature_result['hog_image']
                            hog_image_norm = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)
                            ax.imshow(hog_image_norm, cmap='gray')
                            energy = feature_result.get('hog_statistics', {}).get('energy', 0)
                            ax.set_title(f"HOG Features\nEnergy: {energy:.1f}")
                        else:
                            ax.text(0.5, 0.5, 'HOG features\n(data missing)', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title("HOG Features (Error)")
                    
                    elif method == 'haralick':
                        # Show Haralick features as bar chart
                        if 'haralick_features' in feature_result:
                            features_dict = feature_result['haralick_features']
                            feature_names = list(features_dict.keys())
                            feature_values = [features_dict[name]['mean'] for name in feature_names]
                            
                            bars = ax.bar(range(len(feature_names)), feature_values)
                            ax.set_title("Haralick Texture Features")
                            ax.set_xlabel("Feature")
                            ax.set_ylabel("Value")
                            ax.set_xticks(range(len(feature_names)))
                            ax.set_xticklabels([name[:8] for name in feature_names], rotation=45, ha='right')
                        else:
                            ax.text(0.5, 0.5, 'Haralick features\n(data missing)', 
                                   ha='center', va='center', transform=ax.transAxes)
                            ax.set_title("Haralick Features (Error)")
                    
                    else:
                        # Unknown method - show original image with method name
                        ax.imshow(original_image_rgb)
                        ax.set_title(f"{method.upper()}\n(Method processed)")
                
                except Exception as e:
                    logger.warning(f"Error visualizing {method}: {e}")
                    ax.text(0.5, 0.5, f'{method}\n(Visualization error)', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{method.upper()} (Error)")
                
                ax.axis('off')
                plot_idx += 1
            
            # Hide unused subplots
            for i in range(plot_idx, len(axes)):
                axes[i].axis('off')
            
            # Set main title
            image_name = Path(image_path).name
            methods_list = [m for m in features.keys() if 'error' not in features[m]]
            fig.suptitle(f"Feature Extraction: {image_name}\nMethods: {', '.join(methods_list)}", fontsize=14)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save
            vis_filename = f"features_{Path(image_path).stem}.png"
            vis_path = output_dir / vis_filename
            plt.savefig(vis_path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            
            return str(vis_path)
            
        except Exception as e:
            logger.warning(f"Failed to save feature visualization for {image_path}: {e}")
            return None
    
    def _extract_visual_features_for_visualization(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Re-extract feature data with visual components for visualization (memory efficient)."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image for visualization: {image_path}")
                return None
            
            # Resize image if specified
            resize_dims = self.config.get('image_resize')
            if resize_dims:
                width, height = resize_dims
                image = cv2.resize(image, (width, height))
            
            visual_features = {}
            feature_methods = self.config.get('feature_methods', ['canny'])
            
            for method in feature_methods:
                try:
                    if method == 'canny':
                        result = self.extract_canny_edges(image)
                    elif method == 'harris':
                        result = self.extract_harris_corners(image)
                    elif method == 'lbp':
                        result = self.extract_lbp_texture(image)
                    elif method == 'color_histogram':
                        result = self.extract_color_histogram(image)
                    elif method == 'glcm':
                        result = self.extract_glcm_texture(image)
                    elif method == 'orb':
                        result = self.extract_orb_keypoints(image)
                    elif method == 'hog':
                        result = self.extract_hog_features(image)
                    elif method == 'haralick':
                        result = self.extract_haralick_texture(image)
                    else:
                        logger.warning(f"Unknown feature method for visualization: {method}")
                        continue
                    
                    if 'error' not in result:
                        # Keep the full result including visual data for visualization
                        visual_features[method] = result
                
                except Exception as e:
                    logger.warning(f"Failed to extract {method} features for visualization: {e}")
                    visual_features[method] = {'method': method, 'error': str(e)}
            
            return visual_features
            
        except Exception as e:
            logger.warning(f"Failed to extract visual features for {image_path}: {e}")
            return None
    
    def save_results(self, all_features: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """Save feature extraction results with memory optimization."""
        results_dir = self.results_manager.get_results_dir()
        store_vectors_only = config.get('store_as_vectors_only', False)
        
        # Skip visualizations if disabled or in vectors-only mode
        visualization_paths = []
        max_visualizations = config.get('max_visualizations', 0)
        visualization_sample_size = config.get('visualization_sample_size', 10)
        
        # Determine how many visualizations to save
        if config.get('save_feature_images', False):
            # When save_feature_images is enabled, use visualization_sample_size (works even with store_vectors_only)
            num_visualizations = min(visualization_sample_size, max_visualizations) if max_visualizations > 0 else visualization_sample_size
        else:
            # When save_feature_images is disabled, use max_visualizations (for backward compatibility)
            num_visualizations = max_visualizations
        
        if num_visualizations > 0:
            vis_dir = results_dir / "feature_visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            features_with_vectors = [f for f in all_features if f.get('has_features', False)]
            features_to_visualize = features_with_vectors[:num_visualizations]
            logger.info(f"Saving {len(features_to_visualize)} feature visualizations (sample size: {num_visualizations})...")
            
            # Debug: Check what data we have for visualization
            if features_to_visualize:
                sample_feature = features_to_visualize[0]
                logger.info(f"Sample feature data keys: {list(sample_feature.keys())}")
                if 'features' in sample_feature:
                    logger.info(f"Sample features methods: {list(sample_feature['features'].keys())}")
                    for method, data in sample_feature['features'].items():
                        logger.info(f"  {method}: has_visualization_data={'edge_map' in data or 'corner_map' in data or 'lbp_map' in data or 'hog_image' in data}")
            
            for feature_data in tqdm(features_to_visualize, desc="Saving visualizations"):
                vis_path = self.save_feature_visualization(
                    feature_data['image_path'], feature_data, vis_dir
                )
                if vis_path:
                    visualization_paths.append(vis_path)
                gc.collect()
            
            logger.info(f"✅ Saved {len(visualization_paths)} feature visualizations to {vis_dir}")
        else:
            if config.get('save_feature_images', False):
                logger.info("Skipping visualizations (no successful feature extractions)")
            else:
                logger.info("Skipping visualizations (save_feature_images disabled for efficiency)")
        
        # Save feature data efficiently
        features_with_vectors = [f for f in all_features if f.get('has_features', False)]
        
        if store_vectors_only:
            # Skip JSON entirely - save only vectors as .npz
            logger.info(f"Saving feature vectors for {len(features_with_vectors)} images (vectors-only mode)...")
            
            if features_with_vectors:
                # Organize feature vectors by method
                feature_methods = config.get('feature_methods', ['canny'])
                feature_arrays = {}
                image_paths_with_features = []
                
                for feature_data in features_with_vectors:
                    image_paths_with_features.append(feature_data['image_path'])
                    
                    for method in feature_methods:
                        if method in feature_data.get('feature_vectors', {}):
                            if method not in feature_arrays:
                                feature_arrays[method] = []
                            feature_arrays[method].append(feature_data['feature_vectors'][method])
                
                # Save feature vectors
                vectors_path = results_dir / "feature_vectors.npz"
                save_data = {
                    'image_paths': image_paths_with_features,
                    'feature_methods': feature_methods
                }
                
                # Add feature arrays (process one method at a time to save memory)
                for method, vectors in feature_arrays.items():
                    try:
                        logger.info(f"Processing {method} vectors ({len(vectors)} vectors)...")
                        # Ensure all vectors have the same length
                        max_length = max(len(v) for v in vectors) if vectors else 0
                        if max_length > 0:
                            padded_vectors = []
                            for v in vectors:
                                if len(v) < max_length:
                                    padded_v = v + [0.0] * (max_length - len(v))
                                else:
                                    padded_v = v[:max_length]
                                padded_vectors.append(padded_v)
                            
                            save_data[f'{method}_vectors'] = np.array(padded_vectors)
                            del padded_vectors  # Free memory immediately
                            gc.collect()
                            
                    except Exception as e:
                        logger.warning(f"Failed to save {method} vectors: {e}")
                
                logger.info(f"Saving feature vectors to {vectors_path}...")
                np.savez_compressed(vectors_path, **save_data)
                
                # Clear save_data from memory
                del save_data
                gc.collect()
                
                # Skip JSON feature data entirely
                features_path = None
                
        else:
            # Legacy mode: save feature data as JSON
            features_path = results_dir / "feature_data.json"
            logger.info(f"Saving feature data for {len(all_features)} images...")
            
            with open(features_path, 'w') as f:
                json.dump(all_features, f, indent=2, default=str)
            
            # Save feature vectors as numpy array (process in chunks to manage memory)
            if features_with_vectors:
                logger.info(f"Saving feature vectors for {len(features_with_vectors)} images...")
                
                # Organize feature vectors by method
                feature_methods = config.get('feature_methods', [])
                feature_arrays = {}
                image_paths_with_features = []
                
                # Process in chunks to avoid memory issues
                chunk_size = 1000
                for i in range(0, len(features_with_vectors), chunk_size):
                    chunk = features_with_vectors[i:i + chunk_size]
                    
                    for feature_data in chunk:
                        image_paths_with_features.append(feature_data['image_path'])
                        
                        for method in feature_methods:
                            if method in feature_data.get('feature_vectors', {}):
                                if method not in feature_arrays:
                                    feature_arrays[method] = []
                                feature_arrays[method].append(feature_data['feature_vectors'][method])
                    
                    # Force garbage collection after each chunk
                    if i % (chunk_size * 5) == 0:
                        gc.collect()
                
                # Save feature vectors
                vectors_path = results_dir / "feature_vectors.npz"
                save_data = {
                    'image_paths': image_paths_with_features,
                    'feature_methods': feature_methods
                }
                
                # Add feature arrays (process one method at a time to save memory)
                for method, vectors in feature_arrays.items():
                    try:
                        logger.info(f"Processing {method} vectors ({len(vectors)} vectors)...")
                        # Ensure all vectors have the same length
                        max_length = max(len(v) for v in vectors) if vectors else 0
                        if max_length > 0:
                            padded_vectors = []
                            for v in vectors:
                                if len(v) < max_length:
                                    padded_v = v + [0.0] * (max_length - len(v))
                                else:
                                    padded_v = v[:max_length]
                                padded_vectors.append(padded_v)
                            
                            save_data[f'{method}_vectors'] = np.array(padded_vectors)
                            del padded_vectors  # Free memory immediately
                            gc.collect()
                            
                    except Exception as e:
                        logger.warning(f"Failed to save {method} vectors: {e}")
                
                logger.info(f"Saving feature vectors to {vectors_path}...")
                np.savez_compressed(vectors_path, **save_data)
                
                # Clear save_data from memory
                del save_data
                gc.collect()

        # Generate statistics
        total_images = len(all_features)
        successful_features = len(features_with_vectors)
        feature_methods = config.get('feature_methods', [])
        
        # Count successful extractions per method
        method_success = {}
        for method in feature_methods:
            method_success[method] = sum(
                1 for f in all_features 
                if f.get('has_features', False) and method in f.get('feature_vectors', {})
            )
        
        stats = {
            'total_images_processed': total_images,
            'successful_feature_extractions': successful_features,
            'success_rate': successful_features / total_images if total_images > 0 else 0.0,
            'feature_methods': feature_methods,
            'method_success_rates': {
                method: count / total_images for method, count in method_success.items()
            },
            'method_success_counts': method_success,
            'store_as_vectors_only': store_vectors_only
        }
        
        # Add method-specific settings
        for method in feature_methods:
            if method == 'canny':
                stats['canny_settings'] = {
                    'low_threshold': config.get('canny_low_threshold', 50),
                    'high_threshold': config.get('canny_high_threshold', 150)
                }
            elif method == 'harris':
                stats['harris_settings'] = {
                    'block_size': config.get('harris_block_size', 2),
                    'k': config.get('harris_k', 0.04)
                }
            elif method == 'lbp':
                stats['lbp_settings'] = {
                    'radius': config.get('lbp_radius', 3),
                    'n_points': config.get('lbp_n_points', 24)
                }
        
        stats_path = results_dir / "feature_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save configuration and metadata
        additional_metadata = {
            'total_images_processed': total_images,
            'successful_feature_extractions': successful_features,
            'success_rate': stats['success_rate'],
            'feature_methods': feature_methods,
            'visualizations_created': len(visualization_paths),
            'store_as_vectors_only': store_vectors_only,
            'method': 'feature_extraction'
        }
        
        config_path = self.results_manager.save_run_config(config, additional_metadata)
        
        files_created = [str(stats_path), str(config_path)] + visualization_paths
        if store_vectors_only:
            files_created.append(str(results_dir / "feature_vectors.npz"))
        else:
            if features_path:
                files_created.append(str(features_path))
            files_created.append(str(results_dir / "feature_vectors.npz"))
        
        result_metadata = self.results_manager.save_result_metadata({
            'total_images_processed': total_images,
            'successful_feature_extractions': successful_features,
            'success_rate': stats['success_rate'],
            'files_created': files_created
        })
        
        logger.info(f"Results saved to: {results_dir}")
        logger.info(f"Extracted features from {successful_features}/{total_images} images "
                   f"(success rate: {stats['success_rate']:.2%})")
        logger.info(f"Feature methods: {feature_methods}")
        if store_vectors_only:
            logger.info("📦 Stored as vectors only - maximum efficiency mode")
        
        return result_metadata
    
    def extract_features(self, config_path: Union[str, Path], run_id: Optional[str] = None) -> str:
        """Main method to extract features with memory optimization."""
        # Load configuration
        config = self.load_config(config_path)
        
        # Create results manager
        if self.results_manager is None:
            self.results_manager = create_service_runner('feature_extraction', config['output_path'])
        
        actual_run_id = self.results_manager.create_run(run_id)
        
        # Load embeddings and image paths
        embeddings, image_paths = self.load_embeddings(config['embeddings_path'])
        
        # Clear embeddings from memory immediately since we only need image paths
        del embeddings
        gc.collect()
        
        # Process images using batched approach
        start_time = time.time()
        batch_files, successful_extractions = self.extract_features_from_images_batched(image_paths)
        
        # Consolidate batch results
        logger.info("Consolidating batch results...")
        all_features = self.consolidate_batch_results(batch_files)
        
        processing_time = time.time() - start_time
        
        # Performance summary
        images_per_second = len(image_paths) / processing_time if processing_time > 0 else 0
        
        logger.info(f"📊 Performance Summary:")
        logger.info(f"   Total time: {processing_time:.1f}s")
        logger.info(f"   Speed: {images_per_second:.1f} images/second")
        logger.info(f"   Success rate: {successful_extractions}/{len(image_paths)} ({successful_extractions/len(image_paths)*100:.1f}%)")
        
        # Save results
        self.save_results(all_features, config)
        
        # Clean up batch files
        logger.info("Cleaning up batch files...")
        for batch_file in batch_files:
            try:
                Path(batch_file).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete batch file {batch_file}: {e}")
        
        # Remove batch directory if empty
        batch_dir = self.results_manager.get_results_dir() / "batches"
        try:
            if batch_dir.exists() and not any(batch_dir.iterdir()):
                batch_dir.rmdir()
        except Exception:
            pass
        
        return str(self.results_manager.get_results_dir())


def extract_features(config_path: Union[str, Path], run_id: Optional[str] = None) -> str:
    """Convenience function to extract features."""
    service = FeatureExtractionService()
    return service.extract_features(config_path, run_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract visual features using computer vision algorithms")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--run-id", help="Optional run identifier")
    
    args = parser.parse_args()
    
    try:
        service = FeatureExtractionService()
        results_path = service.extract_features(args.config, args.run_id)
        print(f"✅ Feature extraction completed successfully!")
        print(f"   Results saved to: {results_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1) 