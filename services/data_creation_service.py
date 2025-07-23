#!/usr/bin/env python3
"""
Dataset Creation Service

Generates a deterministic subset of the dataset based on a given percentage and seed.
Guarantees reproducibility and forced inclusion of key images.

This service reads configuration from a YAML file and outputs selected image paths
to a JSON file, maintaining deterministic behavior through random seeds.
"""

import random
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

# Import common services
from common_fs_service import (
    FileSystemService, ResultsManager, ConfigValidator,
    create_service_runner, discover_images, validate_config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetCreationService:
    """
    Service for creating deterministic dataset subsets with forced inclusion support.
    
    This service provides a clean interface for:
    - Loading and validating dataset creation configurations
    - Discovering images in dataset directories  
    - Sampling subsets deterministically with forced inclusions
    - Saving results using standardized output structure
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        """
        Initialize the Dataset Creation Service.
        
        Args:
            results_manager: Optional ResultsManager instance. If not provided,
                           will be created during run execution.
        """
        self.results_manager = results_manager
        self.config = None
        self.all_images = []
        self.selected_images = []
        
    def load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing validated configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
            ValueError: If required fields are missing or invalid
        """
        # Use common config validation service
        config = validate_config(config_path, 'dataset')
        
        # Set defaults for optional fields
        config.setdefault('force_include', [])
        
        # Convert paths to Path objects
        config['dataset_path'] = Path(config['dataset_path'])
        config['output_path'] = Path(config['output_path'])
        
        # Convert force_include paths to Path objects
        config['force_include'] = [Path(path) for path in config['force_include']]
        
        self.config = config
        logger.info(f"Configuration loaded and validated successfully from {config_path}")
        return config
        
    def discover_images(self, dataset_path: Path) -> List[Path]:
        """
        Recursively discover all supported image files in the dataset directory.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            List of Path objects for discovered images
            
        Raises:
            FileNotFoundError: If dataset path doesn't exist
        """
        # Use common image discovery service
        images = discover_images(dataset_path, recursive=True)
        self.all_images = images
        return images
        
    def validate_force_include_paths(self, force_include: List[Path], all_images: List[Path]) -> List[Path]:
        """
        Validate and filter force_include paths to ensure they exist in the dataset.
        
        Args:
            force_include: List of paths to force include
            all_images: List of all discovered images
            
        Returns:
            List of valid force_include paths
        """
        if not force_include:
            return []
            
        # Use common validation service to check paths
        valid_paths, invalid_paths = FileSystemService.validate_image_paths(force_include)
        
        # Further filter to ensure they exist in our discovered images
        all_images_set = {img.resolve() for img in all_images}
        valid_force_include = []
        
        for path in valid_paths:
            resolved_path = path.resolve()
            if resolved_path in all_images_set:
                valid_force_include.append(path)
            else:
                invalid_paths.append(path)
                
        if invalid_paths:
            logger.warning(f"The following force_include paths were not found in dataset: {invalid_paths}")
            
        logger.info(f"Validated {len(valid_force_include)} force_include paths")
        return valid_force_include
        
    def sample_images(self, all_images: List[Path], fraction: float, 
                     force_include: List[Path], random_seed: int) -> List[Path]:
        """
        Sample a deterministic subset of images with forced inclusion support.
        
        Args:
            all_images: List of all available images
            fraction: Fraction of images to select (0.0 to 1.0)
            force_include: List of images to force include
            random_seed: Random seed for deterministic sampling
            
        Returns:
            List of selected image paths
        """
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Validate force_include paths
        valid_force_include = self.validate_force_include_paths(force_include, all_images)
        
        # Calculate target number of images
        target_count = max(1, int(len(all_images) * fraction))
        
        # Start with force_include images
        selected_images = set(img.resolve() for img in valid_force_include)
        
        # If force_include already meets or exceeds target, return them
        if len(selected_images) >= target_count:
            logger.info(f"Force include list ({len(selected_images)}) meets target ({target_count})")
            return list(valid_force_include)
            
        # Get remaining images to sample from
        all_images_set = {img.resolve() for img in all_images}
        remaining_images = [img for img in all_images 
                          if img.resolve() not in selected_images]
        
        # Calculate how many more images we need
        additional_needed = target_count - len(selected_images)
        
        # Randomly sample additional images
        if additional_needed > 0 and remaining_images:
            additional_images = random.sample(
                remaining_images, 
                min(additional_needed, len(remaining_images))
            )
            
            # Combine force_include and additional sampled images
            final_selection = valid_force_include + additional_images
        else:
            final_selection = valid_force_include
            
        logger.info(f"Selected {len(final_selection)} images: "
                   f"{len(valid_force_include)} forced + "
                   f"{len(final_selection) - len(valid_force_include)} sampled")
        
        self.selected_images = final_selection
        return final_selection
        
    def save_results(self, selected_images: List[Path], 
                    config: Dict, run_id: Optional[str] = None) -> Dict:
        """
        Save the selected images list and configuration using standard output structure.
        
        Args:
            selected_images: List of selected image paths
            config: Configuration dictionary
            run_id: Optional run identifier (generates one if not provided)
            
        Returns:
            Dictionary with result metadata
        """
        # Create or use existing results manager
        if self.results_manager is None:
            self.results_manager = create_service_runner('dataset', config['output_path'])
            
        # Create run directory
        actual_run_id = self.results_manager.create_run(run_id)
        
        # Save used images
        used_images_path = self.results_manager.save_used_images(selected_images)
        
        # Prepare additional metadata
        additional_metadata = {
            'total_images_available': len(self.all_images),
            'images_selected': len(selected_images),
            'selection_fraction': len(selected_images) / len(self.all_images) if self.all_images else 0,
            'forced_inclusions': len(config.get('force_include', [])),
            'preprocessing': 'deterministic_sampling',
            'method': 'random_sampling_with_forced_inclusion'
        }
        
        # Save configuration with metadata
        config_path = self.results_manager.save_run_config(config, additional_metadata)
        
        # Create final result metadata
        result_metadata = self.results_manager.save_result_metadata({
            'total_images_available': len(self.all_images),
            'images_selected': len(selected_images),
            'selection_fraction': len(selected_images) / len(self.all_images) if self.all_images else 0,
            'forced_inclusions': len(config.get('force_include', [])),
            'files_created': [
                str(used_images_path),
                str(config_path)
            ]
        })
        
        logger.info(f"Results saved to: {self.results_manager.get_results_dir()}")
        logger.info(f"Selected {len(selected_images)} images out of {len(self.all_images)} total")
        
        return result_metadata
        
    def create_dataset_subset(self, config_path: Union[str, Path], 
                            run_id: Optional[str] = None) -> Dict:
        """
        Main method to create a dataset subset based on configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            run_id: Optional run identifier
            
        Returns:
            Dictionary with result metadata
        """
        # Load configuration
        config = self.load_config(config_path)
        
        # Discover all images
        all_images = self.discover_images(config['dataset_path'])
        
        if not all_images:
            raise ValueError(f"No images found in dataset path: {config['dataset_path']}")
            
        # Sample images based on configuration
        selected_images = self.sample_images(
            all_images=all_images,
            fraction=config['fraction'],
            force_include=config['force_include'],
            random_seed=config['random_seed']
        )
        
        # Save results
        result_metadata = self.save_results(
            selected_images=selected_images,
            config=config,
            run_id=run_id
        )
        
        return result_metadata


# Python API Interface
def create_dataset_subset(config_path: Union[str, Path], run_id: Optional[str] = None) -> Dict:
    """
    Create a deterministic dataset subset based on configuration.
    
    Args:
        config_path: Path to the YAML configuration file
        run_id: Optional run identifier
        
    Returns:
        Dictionary with result metadata
        
    Example:
        >>> result = create_dataset_subset("config.yaml")
        >>> print(f"Created dataset with {result['images_selected']} images")
    """
    service = DatasetCreationService()
    return service.create_dataset_subset(config_path, run_id)


# CLI Interface (if run as script)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create deterministic dataset subsets")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--run-id", help="Optional run identifier")
    
    args = parser.parse_args()
    
    try:
        result = create_dataset_subset(args.config, args.run_id)
        print(f"✅ Dataset creation completed successfully!")
        print(f"   Run ID: {result['run_id']}")
        print(f"   Images selected: {result['images_selected']}")
        print(f"   Output directory: {result['output_directory']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
