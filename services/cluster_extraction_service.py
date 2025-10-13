#!/usr/bin/env python3
"""
Cluster Extraction Service

Extracts all images from a specific cluster identified by cluster results and cluster number.
Creates a directory structure containing all images from the specified cluster.

This service reads clustering results from a previous clustering service run and
extracts images from a specific cluster into a new organized directory structure.
"""

import json
import shutil
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import logging

# Import common services
from common_fs_service import (
    ResultsManager, 
    validate_config, 
    create_service_runner,
    logger
)

# Configure logging
logging.basicConfig(level=logging.INFO)


class ClusterExtractionService:
    """
    Service for extracting images from a specific cluster.
    
    This service provides a clean interface for:
    - Loading and validating cluster extraction configurations
    - Reading clustering results from previous pipeline steps
    - Extracting all images from a specified cluster
    - Organizing extracted images in a structured directory
    - Saving results using standardized output structure
    """
    
    def __init__(self, results_manager: Optional[ResultsManager] = None):
        """
        Initialize the Cluster Extraction Service.
        
        Args:
            results_manager: Optional ResultsManager instance. If not provided,
                           will be created during run execution.
        """
        self.results_manager = results_manager
        self.config = None
        self.cluster_data = None
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate configuration for cluster extraction.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing validated configuration parameters
        """
        # Use common config validation service
        config = validate_config(config_path, 'cluster_extraction')
        
        # Set defaults for optional parameters
        config.setdefault('copy_method', 'copy')  # 'copy', 'symlink', 'hardlink'
        config.setdefault('preserve_structure', True)  # Preserve original directory structure
        config.setdefault('create_index', True)  # Create index file with image info
        config.setdefault('include_metadata', True)  # Include cluster metadata
        
        # Convert paths to Path objects
        config['cluster_results_path'] = Path(config['cluster_results_path'])
        config['output_path'] = Path(config['output_path'])
        
        # Normalize cluster_number to cluster_numbers (validation already done by common service)
        cluster_numbers = config['cluster_number']
        if isinstance(cluster_numbers, int):
            # Single cluster number - convert to list
            config['cluster_numbers'] = [cluster_numbers]
        elif isinstance(cluster_numbers, list):
            # Multiple cluster numbers - ensure all are integers
            config['cluster_numbers'] = [int(num) for num in cluster_numbers]
        else:
            # Fallback: convert to int and make list
            config['cluster_numbers'] = [int(cluster_numbers)]
        
        # Keep original for backward compatibility
        config['cluster_number'] = config['cluster_numbers'][0]
        
        # Validate cluster results path
        if not config['cluster_results_path'].exists():
            raise FileNotFoundError(f"Cluster results path not found: {config['cluster_results_path']}")
        
        self.config = config
        logger.info(f"Configuration loaded and validated successfully from {config_path}")
        logger.info(f"Will extract {len(config['cluster_numbers'])} cluster(s): {config['cluster_numbers']}")
        return config
    
    def load_cluster_data(self, cluster_results_path: Path) -> Dict[str, Any]:
        """
        Load cluster data from clustering service results.
        
        Args:
            cluster_results_path: Path to clustering service results directory
            
        Returns:
            Dictionary containing cluster data
        """
        # Try to find the detailed clusters file (contains all images)
        detailed_clusters_path = cluster_results_path / "detailed_clusters.json"
        if not detailed_clusters_path.exists():
            # Fallback to cluster_analysis.json (contains example images only)
            cluster_analysis_path = cluster_results_path / "cluster_analysis.json"
            if not cluster_analysis_path.exists():
                raise FileNotFoundError(f"No cluster data found in {cluster_results_path}")
            
            logger.warning("Using cluster_analysis.json (example images only). For all images, use detailed_clusters.json")
            with open(cluster_analysis_path, 'r') as f:
                cluster_data = json.load(f)
                # Convert to detailed format for consistency
                detailed_data = {}
                for cluster_id, cluster_info in cluster_data.items():
                    detailed_data[cluster_id] = {
                        'cluster_id': cluster_info['cluster_id'],
                        'size': cluster_info['size'],
                        'percentage': cluster_info['percentage'],
                        'is_noise': cluster_info['is_noise'],
                        'all_images': cluster_info.get('example_images', [])  # Use examples as fallback
                    }
                cluster_data = detailed_data
        else:
            # Load detailed clusters (preferred)
            with open(detailed_clusters_path, 'r') as f:
                cluster_data = json.load(f)
        
        self.cluster_data = cluster_data
        logger.info(f"Loaded cluster data for {len(cluster_data)} clusters from {cluster_results_path}")
        
        # Log cluster information
        for cluster_id, cluster_info in cluster_data.items():
            size = cluster_info.get('size', len(cluster_info.get('all_images', [])))
            is_noise = cluster_info.get('is_noise', False)
            logger.info(f"  Cluster {cluster_id}: {size} images{' (noise)' if is_noise else ''}")
        
        return cluster_data
    
    def validate_cluster_numbers(self, cluster_numbers: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Validate that the specified cluster numbers exist and return cluster info for each.
        
        Args:
            cluster_numbers: List of cluster numbers to validate
            
        Returns:
            Dictionary mapping cluster_number to cluster information
            
        Raises:
            ValueError: If any cluster number doesn't exist
        """
        validated_clusters = {}
        available_clusters = list(self.cluster_data.keys())
        
        for cluster_number in cluster_numbers:
            cluster_key = str(cluster_number)
            
            # Special case for noise cluster (-1)
            if cluster_number == -1:
                cluster_key = "-1"
            
            if cluster_key not in self.cluster_data:
                raise ValueError(f"Cluster {cluster_number} not found. Available clusters: {available_clusters}")
            
            cluster_info = self.cluster_data[cluster_key]
            image_count = len(cluster_info.get('all_images', []))
            is_noise = cluster_info.get('is_noise', False)
            
            logger.info(f"Found cluster {cluster_number}: {image_count} images{' (noise cluster)' if is_noise else ''}")
            validated_clusters[cluster_number] = cluster_info
        
        return validated_clusters
    
    def extract_cluster_images(self, cluster_info: Dict[str, Any], cluster_number: int) -> Tuple[List[str], int]:
        """
        Extract and copy images from the specified cluster.
        
        Args:
            cluster_info: Cluster information dictionary
            cluster_number: The cluster number being extracted
            
        Returns:
            Tuple of (copied_image_paths, success_count)
        """
        image_paths = cluster_info.get('all_images', [])
        
        if not image_paths:
            logger.warning(f"No images found in cluster {cluster_number}")
            return [], 0
        
        # Create cluster extraction directory
        results_dir = self.results_manager.get_results_dir()
        is_noise = cluster_info.get('is_noise', False)
        
        if is_noise:
            cluster_dir = results_dir / "extracted_images" / "noise_cluster"
        else:
            cluster_dir = results_dir / "extracted_images" / f"cluster_{cluster_number:03d}"
        
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        copy_method = self.config.get('copy_method', 'copy')
        preserve_structure = self.config.get('preserve_structure', True)
        
        copied_paths = []
        success_count = 0
        failed_count = 0
        
        logger.info(f"Extracting {len(image_paths)} images from cluster {cluster_number} using {copy_method}...")
        
        for i, image_path in enumerate(image_paths):
            try:
                source_path = Path(image_path)
                if not source_path.exists():
                    logger.warning(f"Source image not found: {source_path}")
                    failed_count += 1
                    continue
                
                # Determine destination path
                if preserve_structure:
                    # Try to preserve relative path structure
                    try:
                        # Get relative path from a common base
                        rel_path = source_path.name  # Default to just filename
                        dest_path = cluster_dir / rel_path
                        
                        # Handle filename conflicts
                        counter = 1
                        original_dest = dest_path
                        while dest_path.exists():
                            stem = original_dest.stem
                            suffix = original_dest.suffix
                            dest_path = cluster_dir / f"{stem}_{counter:03d}{suffix}"
                            counter += 1
                            
                    except Exception as e:
                        logger.debug(f"Failed to preserve structure for {source_path}: {e}")
                        dest_path = cluster_dir / f"image_{i+1:04d}{source_path.suffix}"
                else:
                    # Simple numbered naming
                    dest_path = cluster_dir / f"image_{i+1:04d}{source_path.suffix}"
                
                # Copy the image using the specified method
                if copy_method == 'copy':
                    shutil.copy2(source_path, dest_path)
                elif copy_method == 'symlink':
                    dest_path.symlink_to(source_path.resolve())
                elif copy_method == 'hardlink':
                    dest_path.hardlink_to(source_path)
                else:
                    raise ValueError(f"Unknown copy method: {copy_method}")
                
                copied_paths.append(str(dest_path))
                success_count += 1
                
                # Progress update
                if (i + 1) % 100 == 0 or (i + 1) == len(image_paths):
                    logger.info(f"  Progress: {i + 1}/{len(image_paths)} images processed")
                    
            except Exception as e:
                logger.warning(f"Failed to copy {image_path}: {e}")
                failed_count += 1
        
        logger.info(f"Extraction completed: {success_count} succeeded, {failed_count} failed")
        return copied_paths, success_count
    
    def create_cluster_index(self, cluster_info: Dict[str, Any], cluster_number: int, 
                           copied_paths: List[str]) -> str:
        """
        Create an index file with cluster and image information.
        
        Args:
            cluster_info: Cluster information dictionary
            cluster_number: The cluster number
            copied_paths: List of copied image paths
            
        Returns:
            Path to the created index file
        """
        results_dir = self.results_manager.get_results_dir()
        index_path = results_dir / "cluster_index.json"
        
        # Gather original image paths
        original_paths = cluster_info.get('all_images', [])
        
        # Create mapping of copied to original paths
        image_mapping = []
        for i, copied_path in enumerate(copied_paths):
            original_path = original_paths[i] if i < len(original_paths) else None
            image_mapping.append({
                'index': i + 1,
                'copied_path': copied_path,
                'original_path': original_path,
                'filename': Path(copied_path).name
            })
        
        # Create index data
        index_data = {
            'cluster_extraction_info': {
                'cluster_number': cluster_number,
                'cluster_id': cluster_info.get('cluster_id', cluster_number),
                'is_noise_cluster': cluster_info.get('is_noise', False),
                'total_images_in_cluster': cluster_info.get('size', len(original_paths)),
                'images_extracted': len(copied_paths),
                'extraction_date': self.results_manager.run_id if self.results_manager else None,
                'copy_method': self.config.get('copy_method', 'copy'),
                'preserve_structure': self.config.get('preserve_structure', True)
            },
            'cluster_metadata': {
                'size': cluster_info.get('size', len(original_paths)),
                'percentage': cluster_info.get('percentage', 0.0),
                'is_noise': cluster_info.get('is_noise', False)
            },
            'images': image_mapping,
            'statistics': {
                'total_images': len(image_mapping),
                'extraction_success_rate': len(copied_paths) / len(original_paths) if original_paths else 0.0,
                'source_cluster_results': str(self.config['cluster_results_path'])
            }
        }
        
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2, default=str)
        
        logger.info(f"Created cluster index: {index_path}")
        return str(index_path)
    
    def save_results(self, extraction_results: Dict[int, Dict[str, Any]], 
                    index_path: str, config: Dict[str, Any],
                    run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Save multi-cluster extraction results using standard output structure.
        
        Args:
            extraction_results: Dictionary mapping cluster_number to extraction results
            index_path: Path to the cluster index file
            config: Configuration dictionary
            run_id: Optional run identifier
            
        Returns:
            Dictionary with result metadata
        """
        # Use existing results manager (created earlier)
        results_dir = self.results_manager.get_results_dir()
        
        # Calculate overall statistics
        total_images_extracted = sum(len(result['copied_paths']) for result in extraction_results.values())
        total_source_size = sum(result['cluster_info'].get('size', 0) for result in extraction_results.values())
        cluster_numbers = list(extraction_results.keys())
        
        # Create extraction summary
        extraction_summary = {
            'clusters_extracted': cluster_numbers,
            'total_clusters': len(cluster_numbers),
            'total_images_extracted': total_images_extracted,
            'total_source_size': total_source_size,
            'overall_success_rate': total_images_extracted / total_source_size if total_source_size > 0 else 0.0,
            'extraction_method': config.get('copy_method', 'copy'),
            'preserve_structure': config.get('preserve_structure', True),
            'source_results_path': str(config['cluster_results_path']),
            'extracted_images_directory': str(results_dir / "extracted_images"),
            'cluster_details': {}
        }
        
        # Add per-cluster details
        for cluster_number, result in extraction_results.items():
            cluster_info = result['cluster_info']
            copied_paths = result['copied_paths']
            extraction_summary['cluster_details'][str(cluster_number)] = {
                'cluster_number': cluster_number,
                'cluster_id': cluster_info.get('cluster_id', cluster_number),
                'is_noise_cluster': cluster_info.get('is_noise', False),
                'source_cluster_size': cluster_info.get('size', 0),
                'images_extracted': len(copied_paths),
                'success_rate': len(copied_paths) / cluster_info.get('size', 1) if cluster_info.get('size', 0) > 0 else 0.0
            }
        
        summary_path = results_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(extraction_summary, f, indent=2, default=str)
        
        # Prepare additional metadata
        additional_metadata = {
            'clusters_extracted': cluster_numbers,
            'total_clusters': len(cluster_numbers),
            'total_images_extracted': total_images_extracted,
            'total_source_size': total_source_size,
            'overall_success_rate': extraction_summary['overall_success_rate'],
            'copy_method': config.get('copy_method', 'copy'),
            'method': 'multi_cluster_extraction'
        }
        
        # Save configuration with metadata
        config_path = self.results_manager.save_run_config(config, additional_metadata)
        
        # Collect sample image paths from all clusters
        all_copied_paths = []
        for result in extraction_results.values():
            all_copied_paths.extend(result['copied_paths'])
        
        # Create final result metadata
        result_metadata = self.results_manager.save_result_metadata({
            'clusters_extracted': cluster_numbers,
            'total_clusters': len(cluster_numbers),
            'total_images_extracted': total_images_extracted,
            'total_source_size': total_source_size,
            'files_created': [
                str(summary_path),
                str(config_path),
                index_path
            ] + all_copied_paths[:10]  # Include first 10 image paths as examples
        })
        
        logger.info(f"Results saved to: {self.results_manager.get_results_dir()}")
        logger.info(f"Extracted {total_images_extracted} images from {len(cluster_numbers)} clusters: {cluster_numbers}")
        
        return result_metadata
    
    def extract_cluster(self, config_path: Union[str, Path], 
                       run_id: Optional[str] = None) -> str:
        """
        Main method to extract cluster images based on configuration.
        
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
            self.results_manager = create_service_runner('cluster_extraction', config['output_path'])
        
        # Create run directory
        actual_run_id = self.results_manager.create_run(run_id)
        
        # Load cluster data
        logger.info("Loading cluster data...")
        self.load_cluster_data(config['cluster_results_path'])
        
        # Validate cluster numbers and get cluster info
        cluster_numbers = config['cluster_numbers']
        validated_clusters = self.validate_cluster_numbers(cluster_numbers)
        
        # Extract images from all specified clusters
        extraction_results = {}
        total_images_extracted = 0
        
        for cluster_number in cluster_numbers:
            cluster_info = validated_clusters[cluster_number]
            logger.info(f"Extracting cluster {cluster_number}...")
            
            copied_paths, success_count = self.extract_cluster_images(cluster_info, cluster_number)
            
            # Store results for this cluster
            extraction_results[cluster_number] = {
                'cluster_info': cluster_info,
                'copied_paths': copied_paths,
                'success_count': success_count
            }
            
            total_images_extracted += success_count
            logger.info(f"Cluster {cluster_number}: {success_count} images extracted")
        
        logger.info(f"All clusters extracted: {total_images_extracted} total images from {len(cluster_numbers)} clusters")
        
        # Create comprehensive index if requested
        index_path = None
        if config.get('create_index', True):
            index_path = self.create_multi_cluster_index(extraction_results)
        
        # Save results
        result_metadata = self.save_results(
            extraction_results, index_path, config, run_id
        )
        
        return str(self.results_manager.get_results_dir())


def extract_cluster(config_path: Union[str, Path], run_id: Optional[str] = None) -> str:
    """
    Convenience function to extract cluster images using the ClusterExtractionService.
    
    Args:
        config_path: Path to the YAML configuration file
        run_id: Optional run identifier
    
    Returns:
        Path to the generated results directory
    """
    service = ClusterExtractionService()
    return service.extract_cluster(config_path, run_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract images from a specific cluster")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--run-id", help="Optional run identifier")
    
    args = parser.parse_args()
    
    try:
        service = ClusterExtractionService()
        results_path = service.extract_cluster(args.config, args.run_id)
        print(f"✅ Cluster extraction completed successfully!")
        print(f"   Results saved to: {results_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1) 