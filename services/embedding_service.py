import os
import json
import yaml
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch
import open_clip
from datetime import datetime

from common_fs_service import (
    ResultsManager, 
    discover_images, 
    validate_config, 
    create_service_runner,
    logger
)


def letterbox_resize(image: Image.Image, target_size: int) -> Image.Image:
    """
    Resize image to target size while preserving aspect ratio and full content.
    Adds padding as needed to ensure no cropping occurs.
    """
    # Get original dimensions
    width, height = image.size
    
    # Calculate scaling factor to fit within target size
    scale = min(target_size / width, target_size / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    image_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image centered
    new_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    
    # Calculate position to center the image
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    
    new_image.paste(image_resized, (x_offset, y_offset))
    
    return new_image


def load_image_safely(image_path: str) -> Optional[Image.Image]:
    """Load an image safely with error handling."""
    try:
        image = Image.open(image_path)
        # Convert to RGB to ensure consistency
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        return None


def get_image_paths(dataset_path: str) -> List[str]:
    """Get all image paths from the dataset directory using common service."""
    dataset_path_obj = Path(dataset_path)
    
    # Check if this is a results directory containing config.yaml and used_images.json
    config_yaml_path = dataset_path_obj / "config.yaml"
    used_images_json_path = dataset_path_obj / "used_images.json"
    
    if config_yaml_path.exists():
        logger.info(f"Found config.yaml in {dataset_path}, reading actual dataset path...")
        
        # Load the config to get the actual dataset path
        with open(config_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        actual_dataset_path = config.get('dataset_path')
        if not actual_dataset_path:
            raise ValueError(f"No 'dataset_path' found in {config_yaml_path}")
        
        logger.info(f"Actual dataset path from config: {actual_dataset_path}")
        
        # Check if there's a used_images.json file with specific images to use
        if used_images_json_path.exists():
            logger.info(f"Found used_images.json, loading specific image list...")
            with open(used_images_json_path, 'r') as f:
                used_images = json.load(f)
            
            # Validate that these images exist in the actual dataset path
            actual_dataset_path_obj = Path(actual_dataset_path)
            validated_paths = []
            
            for image_path in used_images:
                # Handle both absolute and relative paths
                if Path(image_path).is_absolute():
                    full_path = Path(image_path)
                else:
                    full_path = actual_dataset_path_obj / image_path
                
                if full_path.exists():
                    validated_paths.append(str(full_path))
                else:
                    logger.warning(f"Image not found: {full_path}")
            
            logger.info(f"Using {len(validated_paths)} images from used_images.json")
            return validated_paths
        else:
            # No used_images.json, discover all images in the actual dataset path
            image_paths = discover_images(actual_dataset_path, recursive=True)
            return [str(path) for path in image_paths]
    else:
        # Original behavior: discover images directly in the specified path
        image_paths = discover_images(dataset_path, recursive=True)
        return [str(path) for path in image_paths]


class EmbeddingService:
    """
    Service for generating image embeddings using various models.
    
    This service provides a clean interface for:
    - Loading and validating embedding generation configurations
    - Discovering and processing images from dataset directories
    - Generating embeddings using different CLIP models
    - Saving results using standardized output structure
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        """
        Initialize the Embedding Service.
        
        Args:
            results_manager: Optional ResultsManager instance. If not provided,
                           will be created during run execution.
        """
        self.results_manager = results_manager
        self.config = None
        self.model = None
        self.preprocess = None
        self.device = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and validate configuration for embedding generation.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing validated configuration parameters
        """
        # Use common config validation service
        config = validate_config(config_path, 'embeddings')
        
        # Set defaults for optional parameters
        config.setdefault('batch_size', 32)
        config.setdefault('device', 'auto')
        config.setdefault('target_size', 224)
        config.setdefault('pretrained', 'openai')  # Default pretrained weights
        
        # Convert paths to Path objects
        config['dataset_path'] = Path(config['dataset_path'])
        config['output_path'] = Path(config['output_path'])
        
        self.config = config
        logger.info(f"Configuration loaded and validated successfully from {config_path}")
        return config
        
    def setup_device(self, device_config: str) -> str:
        """
        Setup and validate device configuration.
        
        Args:
            device_config: Device configuration ('auto', 'cuda', or 'cpu')
            
        Returns:
            Actual device string to use
        """
        if device_config == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device_config == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        else:
            device = device_config
            
        self.device = device
        logger.info(f"Using device: {device}")
        return device
        
    def load_model(self, model_name: str, device: str, pretrained: str = 'openai') -> Tuple[str, Any, Any]:
        """
        Load CLIP model with fallback handling and QuickGELU support.
        
        Args:
            model_name: Name of the CLIP model to load
            device: Device to load model on
            pretrained: Pretrained weights to use (e.g., 'openai', 'laion2b_s34b_b79k')
            
        Returns:
            Tuple of (actual_model_name, model, preprocess_function)
        """
        def try_load_model(name: str, pretrained_weights: str = pretrained) -> Tuple[Any, Any, Any]:
            """Helper function to try loading a model with error handling."""
            return open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrained_weights,
                device=device
            )
        
        # List of model variants to try, in order of preference
        model_variants = []
        
        # For OpenAI pretrained weights, try quickgelu variant first to avoid warning
        if 'openai' in str(pretrained).lower():
            # Try with -quickgelu suffix first for OpenAI weights
            if not model_name.endswith('-quickgelu'):
                model_variants.append(f"{model_name}-quickgelu")
            model_variants.append(model_name)
        else:
            # For other pretrained weights, try original name first
            model_variants.append(model_name)
            if not model_name.endswith('-quickgelu'):
                model_variants.append(f"{model_name}-quickgelu")
        
        # Fallback model
        fallback_model = 'ViT-B-32'
        if fallback_model not in model_variants:
            model_variants.append(fallback_model)
        if f"{fallback_model}-quickgelu" not in model_variants:
            model_variants.append(f"{fallback_model}-quickgelu")
        
        # Try loading models in order
        last_error = None
        for variant in model_variants:
            try:
                logger.info(f"Attempting to load model: {variant} with pretrained weights: {pretrained}")
                model, _, preprocess = try_load_model(variant, pretrained)
                model.eval()
                
                # Log success and any QuickGELU information
                if variant.endswith('-quickgelu'):
                    logger.info(f"Successfully loaded model with QuickGELU: {variant}")
                    logger.info("Using QuickGELU activation to match pretrained weights")
                else:
                    logger.info(f"Successfully loaded model: {variant}")
                
                self.model = model
                self.preprocess = preprocess
                return variant, model, preprocess
                
            except Exception as e:
                logger.debug(f"Failed to load model {variant}: {e}")
                last_error = e
                continue
        
        # If all variants failed, raise the last error
        if last_error:
            logger.error(f"Failed to load any model variant for {model_name}")
            raise last_error
        else:
            raise ValueError(f"No model variants could be loaded for {model_name}")
        
    def process_images(self, image_paths: List[str], batch_size: int, 
                      target_size: int) -> Tuple[np.ndarray, List[str]]:
        """
        Process images and generate embeddings.
        
        Args:
            image_paths: List of image paths to process
            batch_size: Batch size for processing
            target_size: Target size for image preprocessing
            
        Returns:
            Tuple of (embeddings_array, valid_image_paths)
        """
        logger.info("Generating embeddings...")
        embeddings = []
        valid_image_paths = []
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                batch_valid_paths = []
                
                # Load and preprocess batch
                for img_path in batch_paths:
                    image = load_image_safely(img_path)
                    if image is not None:
                        # Use letterbox resize to preserve full image content
                        image_processed = letterbox_resize(image, target_size)
                        # Convert to tensor and normalize
                        image_tensor = self.preprocess(image_processed).unsqueeze(0)
                        batch_images.append(image_tensor)
                        batch_valid_paths.append(img_path)
                
                if batch_images:
                    # Stack batch and move to device
                    batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                    
                    # Generate embeddings
                    batch_embeddings = self.model.encode_image(batch_tensor)
                    batch_embeddings = batch_embeddings.cpu().numpy()
                    
                    embeddings.append(batch_embeddings)
                    valid_image_paths.extend(batch_valid_paths)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_paths)}/{len(image_paths)} images")
        
        if not embeddings:
            raise ValueError("No valid embeddings were generated")
        
        # Combine all embeddings
        embeddings_array = np.vstack(embeddings)
        
        # Normalize embeddings
        embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        logger.info(f"Generated {len(embeddings_array)} embeddings")
        return embeddings_array, valid_image_paths
        
    def save_results(self, embeddings_array: np.ndarray, valid_image_paths: List[str],
                    config: Dict[str, Any], actual_model_name: str, 
                    run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Save embeddings and configuration using standard output structure.
        
        Args:
            embeddings_array: Generated embeddings
            valid_image_paths: List of successfully processed image paths
            config: Configuration dictionary
            actual_model_name: Actual model name used (may differ from requested)
            run_id: Optional run identifier
            
        Returns:
            Dictionary with result metadata
        """
        # Create or use existing results manager
        if self.results_manager is None:
            self.results_manager = create_service_runner('embeddings', config['output_path'])
            
        # Create run directory
        actual_run_id = self.results_manager.create_run(run_id)
        
        # Save used images
        used_images_path = self.results_manager.save_used_images(valid_image_paths)
        
        # Save embeddings
        results_dir = self.results_manager.get_results_dir()
        embeddings_path = results_dir / "embeddings.npz"
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings_array,
            image_paths=valid_image_paths
        )
        
        # Prepare additional metadata
        additional_metadata = {
            'images_processed': len(valid_image_paths),
            'embedding_dimension': embeddings_array.shape[1],
            'device_used': self.device,
            'actual_model_name': actual_model_name,
            'preprocessing': 'letterbox',
            'target_size': config['target_size'],
            'batch_size': config['batch_size'],
            'method': 'clip_embedding_generation'
        }
        
        # Save configuration with metadata
        config_path = self.results_manager.save_run_config(config, additional_metadata)
        
        # Create final result metadata
        result_metadata = self.results_manager.save_result_metadata({
            'images_processed': len(valid_image_paths),
            'embedding_dimension': embeddings_array.shape[1],
            'device_used': self.device,
            'actual_model_name': actual_model_name,
            'files_created': [
                str(used_images_path),
                str(config_path),
                str(embeddings_path)
            ]
        })
        
        logger.info(f"Results saved to: {self.results_manager.get_results_dir()}")
        logger.info(f"Generated embeddings for {len(valid_image_paths)} images")
        
        return result_metadata
        
    def generate_embeddings(self, config_path: str, run_id: Optional[str] = None) -> str:
        """
        Main method to generate embeddings based on configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            run_id: Optional run identifier
            
        Returns:
            Path to the generated results directory
        """
        # Load configuration
        config = self.load_config(config_path)
        
        # Setup device
        device = self.setup_device(config['device'])
        
        # Discover images
        logger.info("Discovering images...")
        image_paths = get_image_paths(str(config['dataset_path']))
        logger.info(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {config['dataset_path']}")
        
        # Load model
        actual_model_name, model, preprocess = self.load_model(config['model_name'], device, config['pretrained'])
        
        # Process images
        embeddings_array, valid_image_paths = self.process_images(
            image_paths, config['batch_size'], config['target_size']
        )
        
        # Save results
        result_metadata = self.save_results(
            embeddings_array, valid_image_paths, config, actual_model_name, run_id
        )
        
        return str(self.results_manager.get_results_dir())


def generate_embeddings(config_path: str, run_id: Optional[str] = None) -> str:
    """
    Convenience function to generate embeddings using the EmbeddingService.
    
    Args:
        config_path: Path to the YAML configuration file
        run_id: Optional run identifier
    
    Returns:
        Path to the generated results directory
    """
    service = EmbeddingService()
    return service.generate_embeddings(config_path, run_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate image embeddings")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--run-id", help="Optional run identifier")
    
    args = parser.parse_args()
    
    try:
        service = EmbeddingService()
        results_path = service.generate_embeddings(args.config, args.run_id)
        print(f"✅ Embedding generation completed successfully!")
        print(f"   Results saved to: {results_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
