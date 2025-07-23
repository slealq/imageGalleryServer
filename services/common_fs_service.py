#!/usr/bin/env python3
"""
Common File System Service

Provides shared file system operations, result management, and directory handling
for all services in the modular image processing system.

This service standardizes:
- Result directory creation and management
- Configuration loading/saving
- Image discovery and validation
- Run ID generation and tracking
- Common file operations
"""

import json
import os
import uuid
import yaml
from pathlib import Path
from typing import List, Dict, Set, Optional, Union, Any, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileSystemService:
    """
    Core file system operations service.
    Handles common file operations used across all services.
    """
    
    # Supported image extensions
    SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    @classmethod
    def discover_images(cls, dataset_path: Union[str, Path], 
                       recursive: bool = True) -> List[Path]:
        """
        Recursively discover all supported image files in a directory.
        
        Args:
            dataset_path: Path to search for images
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of Path objects for discovered images
            
        Raises:
            FileNotFoundError: If dataset path doesn't exist
            ValueError: If dataset path is not a directory
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        if not dataset_path.is_dir():
            raise ValueError(f"Dataset path is not a directory: {dataset_path}")
            
        images = []
        
        logger.info(f"Discovering images in: {dataset_path}")
        
        if recursive:
            search_pattern = dataset_path.rglob('*')
        else:
            search_pattern = dataset_path.glob('*')
            
        for file_path in search_pattern:
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_IMAGE_EXTENSIONS:
                images.append(file_path)
                
        logger.info(f"Discovered {len(images)} images")
        return images
    
    @classmethod
    def validate_image_paths(cls, image_paths: List[Union[str, Path]], 
                           base_path: Optional[Union[str, Path]] = None) -> Tuple[List[Path], List[Path]]:
        """
        Validate a list of image paths and return valid/invalid lists.
        
        Args:
            image_paths: List of image paths to validate
            base_path: Optional base path to resolve relative paths against
            
        Returns:
            Tuple of (valid_paths, invalid_paths)
        """
        valid_paths = []
        invalid_paths = []
        
        base_path = Path(base_path) if base_path else None
        
        for path_str in image_paths:
            path = Path(path_str)
            
            # Resolve relative paths against base_path if provided
            if base_path and not path.is_absolute():
                path = base_path / path
                
            try:
                resolved_path = path.resolve()
                if (resolved_path.exists() and 
                    resolved_path.is_file() and 
                    resolved_path.suffix.lower() in cls.SUPPORTED_IMAGE_EXTENSIONS):
                    valid_paths.append(resolved_path)
                else:
                    invalid_paths.append(path)
            except (OSError, ValueError):
                invalid_paths.append(path)
                
        return valid_paths, invalid_paths
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file with error handling.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
            
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Path objects to strings for YAML serialization
        serializable_config = cls._make_config_serializable(config)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(serializable_config, f, default_flow_style=False, indent=2)
            
        logger.info(f"Configuration saved to: {output_path}")
    
    @classmethod
    def _make_config_serializable(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a config dictionary to be JSON/YAML serializable.
        Recursively converts Path objects to strings.
        """
        serializable = {}
        
        for key, value in config.items():
            if isinstance(value, Path):
                serializable[key] = str(value)
            elif isinstance(value, list):
                serializable[key] = [str(item) if isinstance(item, Path) else item for item in value]
            elif isinstance(value, dict):
                serializable[key] = cls._make_config_serializable(value)
            else:
                serializable[key] = value
                
        return serializable
    
    @classmethod
    def save_json(cls, data: Any, output_path: Union[str, Path], 
                  indent: int = 2, ensure_ascii: bool = False) -> None:
        """
        Save data to JSON file with standard formatting.
        
        Args:
            data: Data to save (must be JSON serializable)
            output_path: Path where to save the JSON file
            indent: JSON indentation level
            ensure_ascii: Whether to ensure ASCII encoding
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
            
        logger.info(f"JSON data saved to: {output_path}")
    
    @classmethod
    def load_json(cls, input_path: Union[str, Path]) -> Any:
        """
        Load data from JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is malformed
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"JSON file not found: {input_path}")
            
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.info(f"JSON data loaded from: {input_path}")
        return data


class ResultsManager:
    """
    Manages result directories and standardizes output structure across services.
    
    Standard structure: {output_path}/results/{service_type}/{run_id}/
    """
    
    def __init__(self, service_type: str, output_base_path: Union[str, Path]):
        """
        Initialize results manager for a specific service.
        
        Args:
            service_type: Type of service (e.g., 'dataset', 'embeddings', 'clustering')
            output_base_path: Base path for all results
        """
        self.service_type = service_type
        self.output_base_path = Path(output_base_path)
        self.run_id = None
        self.results_dir = None
        
    def create_run(self, run_id: Optional[str] = None) -> str:
        """
        Create a new run with unique ID and directory structure.
        
        Args:
            run_id: Optional custom run ID (generates one if not provided)
            
        Returns:
            Generated or provided run ID
        """
        if run_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            run_id = f"{timestamp}_{unique_id}"
            
        self.run_id = run_id
        self.results_dir = self.output_base_path / "results" / self.service_type / run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created run directory: {self.results_dir}")
        return run_id
    
    def get_results_dir(self) -> Path:
        """
        Get the results directory for the current run.
        
        Returns:
            Path to results directory
            
        Raises:
            ValueError: If no run has been created
        """
        if self.results_dir is None:
            raise ValueError("No run created. Call create_run() first.")
        return self.results_dir
    
    def save_run_config(self, config: Dict[str, Any], 
                       additional_metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save configuration and metadata for the current run.
        
        Args:
            config: Service configuration
            additional_metadata: Additional metadata to include
            
        Returns:
            Path to saved config file
        """
        if self.results_dir is None:
            raise ValueError("No run created. Call create_run() first.")
            
        # Combine config with run metadata
        run_config = config.copy()
        run_config.update({
            'run_id': self.run_id,
            'service_type': self.service_type,
            'timestamp': datetime.now().isoformat(),
            'results_directory': str(self.results_dir)
        })
        
        if additional_metadata:
            run_config.update(additional_metadata)
            
        config_path = self.results_dir / "config.yaml"
        FileSystemService.save_config(run_config, config_path)
        
        return config_path
    
    def save_used_images(self, image_paths: List[Union[str, Path]]) -> Path:
        """
        Save the list of images used in this run.
        
        Args:
            image_paths: List of image paths that were processed
            
        Returns:
            Path to saved used_images.json file
        """
        if self.results_dir is None:
            raise ValueError("No run created. Call create_run() first.")
            
        # Convert to strings for JSON serialization
        image_paths_str = [str(path) for path in image_paths]
        
        used_images_path = self.results_dir / "used_images.json"
        FileSystemService.save_json(image_paths_str, used_images_path)
        
        return used_images_path
    
    def save_result_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save result metadata and return complete metadata dictionary.
        
        Args:
            metadata: Service-specific result metadata
            
        Returns:
            Complete metadata dictionary with run information
        """
        if self.results_dir is None:
            raise ValueError("No run created. Call create_run() first.")
            
        complete_metadata = {
            'run_id': self.run_id,
            'service_type': self.service_type,
            'timestamp': datetime.now().isoformat(),
            'results_directory': str(self.results_dir),
            **metadata
        }
        
        return complete_metadata
    
    @classmethod
    def find_latest_run(cls, service_type: str, 
                       output_base_path: Union[str, Path]) -> Optional[Path]:
        """
        Find the most recent run directory for a service type.
        
        Args:
            service_type: Type of service to search for
            output_base_path: Base path for results
            
        Returns:
            Path to most recent run directory, or None if not found
        """
        results_base = Path(output_base_path) / "results" / service_type
        
        if not results_base.exists():
            return None
            
        run_dirs = [d for d in results_base.iterdir() if d.is_dir()]
        
        if not run_dirs:
            return None
            
        # Sort by directory name (which includes timestamp) and return latest
        latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]
        return latest_run
    
    @classmethod
    def list_runs(cls, service_type: str, 
                  output_base_path: Union[str, Path]) -> List[Tuple[str, Path]]:
        """
        List all runs for a service type.
        
        Args:
            service_type: Type of service to search for
            output_base_path: Base path for results
            
        Returns:
            List of (run_id, run_path) tuples, sorted by run_id
        """
        results_base = Path(output_base_path) / "results" / service_type
        
        if not results_base.exists():
            return []
            
        runs = []
        for run_dir in results_base.iterdir():
            if run_dir.is_dir():
                runs.append((run_dir.name, run_dir))
                
        # Sort by run_id (directory name)
        runs.sort(key=lambda x: x[0])
        return runs


class ConfigValidator:
    """
    Validates configuration files for different service types.
    """
    
    # Common required fields across all services
    COMMON_REQUIRED_FIELDS = ['output_path']
    
    # Service-specific required fields
    SERVICE_REQUIRED_FIELDS = {
        'dataset': ['dataset_path', 'fraction', 'random_seed'],
        'embeddings': ['dataset_path', 'model_name'],
        'clustering': ['feature_type'],  # Updated for new clustering service
        'similarity': ['embeddings_path', 'target_images'],
        'pose_extraction': ['embeddings_path']
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], 
                       service_type: str) -> List[str]:
        """
        Validate configuration for a specific service type.
        
        Args:
            config: Configuration dictionary to validate
            service_type: Type of service ('dataset', 'embeddings', etc.)
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check common required fields
        for field in cls.COMMON_REQUIRED_FIELDS:
            if field not in config:
                errors.append(f"Missing required field: {field}")
                
        # Check service-specific required fields
        if service_type in cls.SERVICE_REQUIRED_FIELDS:
            for field in cls.SERVICE_REQUIRED_FIELDS[service_type]:
                if field not in config:
                    errors.append(f"Missing required field for {service_type}: {field}")
                    
        # Service-specific validations
        if service_type == 'dataset':
            if 'fraction' in config:
                fraction = config['fraction']
                if not (0.0 <= fraction <= 1.0):
                    errors.append(f"Fraction must be between 0.0 and 1.0, got: {fraction}")
                    
        elif service_type == 'similarity':
            # Validate FAISS index type
            if 'faiss_index_type' in config:
                index_type = config['faiss_index_type']
                valid_types = ['IndexFlatIP', 'IndexFlatL2', 'IndexIVFFlat', 'IndexHNSW']
                if index_type not in valid_types:
                    errors.append(f"Invalid faiss_index_type. Must be one of: {valid_types}, got: {index_type}")
            
            # Validate top_k
            if 'top_k' in config:
                top_k = config['top_k']
                if not isinstance(top_k, int) or top_k <= 0:
                    errors.append(f"top_k must be a positive integer, got: {top_k}")
                    
            # Validate target_images
            if 'target_images' in config:
                target_images = config['target_images']
                if isinstance(target_images, str):
                    target_images = [target_images]
                elif not isinstance(target_images, list):
                    errors.append(f"target_images must be a string or list of strings")
                elif len(target_images) == 0:
                    errors.append(f"target_images cannot be empty")
            
            # Validate pose-aware similarity options
            if config.get('use_pose_similarity', False):
                if 'pose_data_path' not in config or not config['pose_data_path']:
                    errors.append("pose_data_path is required when use_pose_similarity is True")
                
                # Validate weight values
                pose_weight = config.get('pose_weight', 0.3)
                embedding_weight = config.get('embedding_weight', 0.7)
                
                if not isinstance(pose_weight, (int, float)) or not (0.0 <= pose_weight <= 1.0):
                    errors.append("pose_weight must be a number between 0.0 and 1.0")
                
                if not isinstance(embedding_weight, (int, float)) or not (0.0 <= embedding_weight <= 1.0):
                    errors.append("embedding_weight must be a number between 0.0 and 1.0")
                
                # Validate pose similarity method
                pose_method = config.get('pose_similarity_method', 'euclidean')
                valid_pose_methods = ['euclidean', 'cosine']
                if pose_method not in valid_pose_methods:
                    errors.append(f"pose_similarity_method must be one of: {valid_pose_methods}")
                
                # Validate pose confidence threshold
                pose_threshold = config.get('pose_confidence_threshold', 0.2)
                if not isinstance(pose_threshold, (int, float)) or not (0.0 <= pose_threshold <= 1.0):
                    errors.append("pose_confidence_threshold must be a number between 0.0 and 1.0")
                    
        elif service_type == 'pose_extraction':
            # Validate pose confidence threshold
            if 'pose_confidence_threshold' in config:
                threshold = config['pose_confidence_threshold']
                if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                    errors.append(f"pose_confidence_threshold must be a float between 0.0 and 1.0, got: {threshold}")
            
            # Validate pose similarity method
            if 'pose_similarity_method' in config:
                method = config['pose_similarity_method']
                valid_methods = ['euclidean', 'cosine']
                if method not in valid_methods:
                    errors.append(f"Invalid pose_similarity_method. Must be one of: {valid_methods}, got: {method}")
            
            # Validate device configuration
            if 'device' in config:
                device = config['device']
                valid_devices = ['auto', 'cuda', 'cpu']
                if device not in valid_devices:
                    errors.append(f"Invalid device. Must be one of: {valid_devices}, got: {device}")
            
            # Validate boolean fields
            for bool_field in ['extract_body_pose', 'extract_hand_pose', 'save_pose_images']:
                if bool_field in config and not isinstance(config[bool_field], bool):
                    errors.append(f"{bool_field} must be a boolean, got: {type(config[bool_field])}")
            
            # Validate that at least body pose is enabled
            if config.get('extract_body_pose', True) is False and config.get('extract_hand_pose', False) is False:
                errors.append("At least extract_body_pose must be enabled")
            
            # Check if hand pose requires body pose
            if config.get('extract_hand_pose', False) and not config.get('extract_body_pose', True):
                errors.append("extract_hand_pose requires extract_body_pose to be enabled")
                
        elif service_type == 'clustering':
            # Validate feature type
            feature_type = config.get('feature_type', 'combined')
            valid_feature_types = ['pose', 'embedding', 'combined']
            if feature_type not in valid_feature_types:
                errors.append(f"feature_type must be one of: {valid_feature_types}, got: {feature_type}")
            
            # Validate data source requirements
            if feature_type in ['embedding', 'combined'] and not config.get('embeddings_path'):
                errors.append("embeddings_path is required when feature_type is 'embedding' or 'combined'")
            
            if feature_type in ['pose', 'combined'] and not config.get('pose_data_path'):
                errors.append("pose_data_path is required when feature_type is 'pose' or 'combined'")
            
            # Validate clustering method
            clustering_method = config.get('clustering_method', 'kmeans')
            valid_clustering_methods = ['kmeans', 'hdbscan', 'dbscan', 'faiss_similarity']
            if clustering_method not in valid_clustering_methods:
                errors.append(f"clustering_method must be one of: {valid_clustering_methods}, got: {clustering_method}")
            
            # Validate n_clusters
            n_clusters = config.get('n_clusters', 'auto')
            if n_clusters != 'auto' and (not isinstance(n_clusters, int) or n_clusters < 2):
                errors.append("n_clusters must be 'auto' or an integer >= 2")
            
            # Validate weights for combined features
            if feature_type == 'combined':
                pose_weight = config.get('pose_weight', 0.5)
                embedding_weight = config.get('embedding_weight', 0.5)
                
                if not isinstance(pose_weight, (int, float)) or not (0.0 <= pose_weight <= 1.0):
                    errors.append("pose_weight must be a number between 0.0 and 1.0")
                
                if not isinstance(embedding_weight, (int, float)) or not (0.0 <= embedding_weight <= 1.0):
                    errors.append("embedding_weight must be a number between 0.0 and 1.0")
            
            # Validate confidence threshold
            pose_threshold = config.get('pose_confidence_threshold', 0.2)
            if not isinstance(pose_threshold, (int, float)) or not (0.0 <= pose_threshold <= 1.0):
                errors.append("pose_confidence_threshold must be a number between 0.0 and 1.0")
            
            # Validate cluster size parameters
            min_cluster_size = config.get('min_cluster_size', 5)
            if not isinstance(min_cluster_size, int) or min_cluster_size < 1:
                errors.append("min_cluster_size must be a positive integer")
            
            max_examples = config.get('max_examples_per_cluster', 50)
            if not isinstance(max_examples, int) or max_examples < 1:
                errors.append("max_examples_per_cluster must be a positive integer")
            
            # Validate FAISS-specific parameters
            if clustering_method == 'faiss_similarity':
                # Validate FAISS index type
                faiss_index_type = config.get('faiss_index_type', 'IndexFlatIP')
                valid_faiss_types = ['IndexFlatIP', 'IndexFlatL2', 'IndexIVFFlat', 'IndexHNSW']
                if faiss_index_type not in valid_faiss_types:
                    errors.append(f"faiss_index_type must be one of: {valid_faiss_types}, got: {faiss_index_type}")
                
                # Validate similarity threshold
                similarity_threshold = config.get('similarity_threshold', 0.7)
                if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
                    errors.append("similarity_threshold must be a number between 0.0 and 1.0")
                
                # Validate max cluster search depth
                max_search_depth = config.get('max_cluster_search_depth', 100)
                if not isinstance(max_search_depth, int) or max_search_depth < 1:
                    errors.append("max_cluster_search_depth must be a positive integer")
                
                # Validate use_iterative_clustering
                use_iterative = config.get('use_iterative_clustering', True)
                if not isinstance(use_iterative, bool):
                    errors.append("use_iterative_clustering must be a boolean")
            
            # Validate boolean fields
            for bool_field in ['copy_examples', 'generate_visualizations', 'generate_pose_overlays']:
                if bool_field in config and not isinstance(config[bool_field], bool):
                    errors.append(f"{bool_field} must be a boolean, got: {type(config[bool_field])}")
                    
        return errors
    
    @classmethod
    def validate_and_load_config(cls, config_path: Union[str, Path], 
                                service_type: str) -> Dict[str, Any]:
        """
        Load and validate configuration file.
        
        Args:
            config_path: Path to configuration file
            service_type: Type of service
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        config = FileSystemService.load_config(config_path)
        errors = cls.validate_config(config, service_type)
        
        if errors:
            error_msg = f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
            
        return config


# Convenience functions for common operations

def create_service_runner(service_type: str, output_base_path: Union[str, Path]) -> ResultsManager:
    """
    Create a results manager for a service type.
    
    Args:
        service_type: Type of service
        output_base_path: Base path for results
        
    Returns:
        Configured ResultsManager instance
    """
    return ResultsManager(service_type, output_base_path)


def discover_images(dataset_path: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    Convenience function to discover images.
    
    Args:
        dataset_path: Path to search for images
        recursive: Whether to search recursively
        
    Returns:
        List of discovered image paths
    """
    return FileSystemService.discover_images(dataset_path, recursive)


def validate_config(config_path: Union[str, Path], service_type: str) -> Dict[str, Any]:
    """
    Convenience function to validate and load config.
    
    Args:
        config_path: Path to configuration file
        service_type: Type of service
        
    Returns:
        Validated configuration dictionary
    """
    return ConfigValidator.validate_and_load_config(config_path, service_type) 