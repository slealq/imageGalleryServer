#!/usr/bin/env python3
"""
Image Clustering Script using CLIP embeddings and HDBSCAN

This script:
1. Loads images from a specified folder in batches
2. Extracts CLIP embeddings using OpenCLIP
3. Clusters images using HDBSCAN
4. Saves sample images from each cluster

Memory-efficient version for large datasets.
"""

import os
import json
import shutil
import threading
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import open_clip
import hdbscan
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import gc


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
        # Thread is still running, timeout occurred
        raise TimeoutError(f"Loading image timed out after {timeout_seconds} seconds")
    
    if result[1] is not None:
        raise result[1]
    
    return result[0]


class ImageClusterer:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        Initialize the image clusterer with CLIP model.
        
        Args:
            model_name: CLIP model variant to use
            pretrained: Pretrained weights to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def get_image_paths(self, folder_path: str) -> List[str]:
        """
        Get all valid image paths from the specified folder.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            List of valid image file paths
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder {folder_path} does not exist")
            
        print(f"Scanning for images in {folder_path}...")
        
        # Get all potential image files first
        all_files = list(folder.glob("*"))
        image_files = [f for f in all_files if f.suffix.lower() in self.image_extensions]
        
        print(f"Found {len(image_files)} potential image files")
        return [str(f) for f in image_files]
    
    def process_image_batch(self, image_paths: List[str], timeout_seconds: int = 30, 
                           debug: bool = False) -> Tuple[List[str], np.ndarray]:
        """
        Process a batch of images and extract embeddings.
        
        Args:
            image_paths: List of image file paths to process
            timeout_seconds: Timeout for loading each image
            debug: Enable debug output
            
        Returns:
            Tuple of (successful_paths, embeddings)
        """
        successful_paths = []
        images = []
        failed_files = []
        
        # Load images in this batch
        for file_path in image_paths:
            try:
                if debug:
                    print(f"Loading: {Path(file_path).name}")
                
                # Check file size first (skip very large files)
                file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                if debug:
                    print(f"  File size: {file_size_mb:.1f} MB")
                
                if file_size_mb > 100:  # Skip files larger than 100MB
                    if debug:
                        print(f"Skipping large file ({file_size_mb:.1f}MB): {file_path}")
                    failed_files.append(file_path)
                    continue
                
                # Load image with timeout
                start_time = time.time()
                image = load_image_with_timeout(file_path, timeout_seconds)
                load_time = time.time() - start_time
                
                # Check image dimensions (skip extremely large images)
                width, height = image.size
                if debug:
                    print(f"  Dimensions: {width}x{height}, Load time: {load_time:.2f}s")
                
                if width * height > 50_000_000:  # Skip images with more than 50M pixels
                    if debug:
                        print(f"Skipping large image ({width}x{height}): {file_path}")
                    failed_files.append(file_path)
                    continue
                
                successful_paths.append(file_path)
                images.append(image)
                
                if debug:
                    print(f"  SUCCESS")
                
            except TimeoutError as e:
                if debug:
                    print(f"Timeout loading {file_path}: {e}")
                failed_files.append(file_path)
            except Exception as e:
                if debug:
                    print(f"Error loading {file_path}: {e}")
                failed_files.append(file_path)
        
        if not images:
            return successful_paths, np.array([])
        
        # Extract embeddings for this batch
        embeddings = []
        clip_batch_size = 32
        
        for i in range(0, len(images), clip_batch_size):
            batch = images[i:i + clip_batch_size]
            
            # Preprocess images
            processed_images = torch.stack([
                self.preprocess(img) for img in batch
            ]).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(processed_images)
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)
            
            # Clean up GPU memory
            del processed_images
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Clean up CPU memory
        del images
        gc.collect()
        
        return successful_paths, np.array(embeddings)
    
    def extract_embeddings_batched(self, image_paths: List[str], batch_size: int = 100,
                                  timeout_seconds: int = 30, debug: bool = False) -> Tuple[List[str], np.ndarray]:
        """
        Extract CLIP embeddings from images in batches to manage memory.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process in each batch
            timeout_seconds: Timeout for loading each image
            debug: Enable debug output
            
        Returns:
            Tuple of (successful_paths, all_embeddings)
        """
        print(f"Extracting CLIP embeddings from {len(image_paths)} images in batches of {batch_size}...")
        
        all_successful_paths = []
        all_embeddings = []
        failed_count = 0
        
        # Process images in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            
            if debug:
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
                print(f"Batch size: {len(batch_paths)}")
            
            try:
                successful_paths, embeddings = self.process_image_batch(
                    batch_paths, timeout_seconds, debug
                )
                
                if len(embeddings) > 0:
                    all_successful_paths.extend(successful_paths)
                    all_embeddings.append(embeddings)
                
                failed_in_batch = len(batch_paths) - len(successful_paths)
                failed_count += failed_in_batch
                
                if debug:
                    print(f"Batch results: {len(successful_paths)} successful, {failed_in_batch} failed")
                    
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                failed_count += len(batch_paths)
                continue
        
        # Combine all embeddings
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.array([])
        
        print(f"Successfully processed {len(all_successful_paths)} images")
        if failed_count > 0:
            print(f"Failed to process {failed_count} images")
        
        return all_successful_paths, final_embeddings
    
    def cluster_embeddings(self, embeddings: np.ndarray, 
                          min_cluster_size: int = 10,
                          min_samples: int = 5) -> Tuple[hdbscan.HDBSCAN, np.ndarray]:
        """
        Cluster embeddings using HDBSCAN.
        
        Args:
            embeddings: Array of embeddings
            min_cluster_size: Minimum size for a cluster
            min_samples: Minimum samples for core points
            
        Returns:
            Tuple of (clusterer, cluster_labels)
        """
        print("Clustering embeddings with HDBSCAN...")
        print(f"Embedding shape: {embeddings.shape}")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Define custom cosine distance function
        def cosine_distance(X, Y=None):
            """Custom cosine distance function for HDBSCAN."""
            if Y is None:
                Y = X
            return cosine_distances(X, Y)
        
        # Try different parameter combinations to reduce noise
        parameter_sets = [
            # More aggressive clustering - larger clusters, less noise
            {'min_cluster_size': max(15, min_cluster_size), 'min_samples': max(3, min_samples//2), 
             'cluster_selection_epsilon': 0.1, 'name': 'aggressive_v1'},
            {'min_cluster_size': max(20, min_cluster_size), 'min_samples': max(5, min_samples), 
             'cluster_selection_epsilon': 0.15, 'name': 'aggressive_v2'},
            {'min_cluster_size': max(25, min_cluster_size), 'min_samples': max(3, min_samples//3), 
             'cluster_selection_epsilon': 0.2, 'name': 'aggressive_v3'},
            # Original approach as fallback
            {'min_cluster_size': min_cluster_size, 'min_samples': min_samples, 
             'cluster_selection_epsilon': 0.0, 'name': 'original'}
        ]
        
        best_result = None
        best_noise_ratio = 1.0
        
        for params in parameter_sets:
            try:
                print(f"Trying HDBSCAN with {params['name']} (min_cluster_size={params['min_cluster_size']}, min_samples={params['min_samples']}, epsilon={params['cluster_selection_epsilon']})...")
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=params['min_cluster_size'],
                    min_samples=params['min_samples'],
                    metric='euclidean',  # Use euclidean on normalized embeddings
                    cluster_selection_method='eom',
                    cluster_selection_epsilon=params['cluster_selection_epsilon']
                )
                
                cluster_labels = clusterer.fit_predict(embeddings_normalized)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                n_noise = list(cluster_labels).count(-1)
                noise_ratio = n_noise / len(cluster_labels)
                
                print(f"  Found {n_clusters} clusters with {n_noise} noise points (noise ratio: {noise_ratio:.2%})")
                
                # Choose the result with the lowest noise ratio, but still reasonable number of clusters
                if noise_ratio < best_noise_ratio and n_clusters >= 5:
                    best_result = (clusterer, cluster_labels, params['name'])
                    best_noise_ratio = noise_ratio
                    print(f"  -> New best result!")
                
            except Exception as e:
                print(f"Failed with {params['name']}: {e}")
                continue
        
        if best_result is None:
            # Fallback to most aggressive settings if nothing else works
            try:
                print("Using fallback clustering with very aggressive settings...")
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=30,
                    min_samples=2,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    cluster_selection_epsilon=0.3
                )
                
                cluster_labels = clusterer.fit_predict(embeddings_normalized)
                best_result = (clusterer, cluster_labels, 'fallback')
                
            except Exception as e:
                raise RuntimeError(f"All clustering approaches failed: {e}")
        
        final_clusterer, final_labels, method_name = best_result
        n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
        n_noise = list(final_labels).count(-1)
        noise_ratio = n_noise / len(final_labels)
        
        print(f"Final clustering using {method_name}:")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise points: {n_noise} ({noise_ratio:.2%})")
        
        return final_clusterer, final_labels
    
    def visualize_clusters(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                          output_path: str = "cluster_visualization.png"):
        """
        Create a 2D visualization of clusters using t-SNE.
        
        Args:
            embeddings: Array of embeddings
            cluster_labels: Cluster labels
            output_path: Path to save visualization
        """
        print("Creating cluster visualization...")
        
        # For very large datasets, sample points for visualization
        max_points = 5000
        if len(embeddings) > max_points:
            print(f"Sampling {max_points} points for visualization from {len(embeddings)} total points")
            indices = np.random.choice(len(embeddings), max_points, replace=False)
            embeddings_sample = embeddings[indices]
            labels_sample = cluster_labels[indices]
        else:
            embeddings_sample = embeddings
            labels_sample = cluster_labels
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
        embeddings_2d = tsne.fit_transform(embeddings_sample)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = set(labels_sample)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels_sample == label
            if label == -1:
                # Noise points
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        plt.title('Image Clusters Visualization (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def save_cluster_samples(self, image_paths: List[str], cluster_labels: np.ndarray,
                           output_dir: str = "cluster_samples", 
                           samples_per_cluster: int = 5) -> Dict[str, Dict]:
        """
        Save sample images from each cluster and complete image lists.
        
        Args:
            image_paths: List of image file paths
            cluster_labels: Cluster labels
            output_dir: Directory to save samples
            samples_per_cluster: Number of samples to save per cluster
        Returns:
            A dictionary containing cluster information
        """
        print(f"Saving cluster samples and complete image lists to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Group images by cluster
        cluster_images = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_images:
                cluster_images[label] = []
            cluster_images[label].append(image_paths[i])
        
        # Save samples from each cluster and complete lists
        cluster_info = {}
        
        # Sort clusters by size (largest first), but keep noise last
        sorted_clusters = sorted(cluster_images.items(), 
                               key=lambda x: (x[0] == -1, -len(x[1])))
        
        for label, images in sorted_clusters:
            if label == -1:
                cluster_name = "noise"
            else:
                cluster_name = f"cluster_{label}"
            
            cluster_dir = output_path / cluster_name
            cluster_dir.mkdir(exist_ok=True)
            
            # Save complete image list for this cluster
            complete_list_path = cluster_dir / "all_images.json"
            with open(complete_list_path, "w") as f:
                json.dump({
                    "cluster_id": int(label) if label != -1 else -1,
                    "cluster_name": cluster_name,
                    "total_images": len(images),
                    "image_paths": images
                }, f, indent=2)
            
            # Select samples (randomly if more than needed)
            n_samples = min(samples_per_cluster, len(images))
            if len(images) > samples_per_cluster:
                # For larger clusters, try to get diverse samples
                selected_indices = np.linspace(0, len(images)-1, n_samples, dtype=int)
                selected_images = [images[i] for i in selected_indices]
            else:
                selected_images = images
            
            # Copy sample images
            sample_paths = []
            for i, img_path in enumerate(selected_images):
                img_name = Path(img_path).name
                dest_path = cluster_dir / f"sample_{i+1}_{img_name}"
                try:
                    shutil.copy2(img_path, dest_path)
                    sample_paths.append(str(dest_path))
                except Exception as e:
                    print(f"Warning: Could not copy {img_path}: {e}")
            
            cluster_info[cluster_name] = {
                "cluster_id": int(label) if label != -1 else -1,
                "size": len(images),
                "samples": sample_paths,
                "complete_list_file": str(complete_list_path),
                "noise_ratio": len(images) / len(image_paths) if label == -1 else None
            }
            
            print(f"{cluster_name}: {len(images)} images ({len(images)/len(image_paths)*100:.1f}%), saved {len(sample_paths)} samples")
        
        # Save cluster summary information
        with open(output_path / "cluster_info.json", "w") as f:
            json.dump(cluster_info, f, indent=2)
        
        # Save clustering statistics
        stats = {
            "total_images": len(image_paths),
            "total_clusters": len([k for k in cluster_info.keys() if k != "noise"]),
            "noise_images": cluster_info.get("noise", {}).get("size", 0),
            "noise_percentage": cluster_info.get("noise", {}).get("size", 0) / len(image_paths) * 100,
            "largest_clusters": sorted([
                {"name": k, "size": v["size"], "percentage": v["size"]/len(image_paths)*100}
                for k, v in cluster_info.items() if k != "noise"
            ], key=lambda x: x["size"], reverse=True)[:10]
        }
        
        with open(output_path / "clustering_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nClustering Statistics:")
        print(f"Total images: {stats['total_images']}")
        print(f"Total clusters: {stats['total_clusters']}")
        print(f"Noise images: {stats['noise_images']} ({stats['noise_percentage']:.1f}%)")
        print(f"Cluster information saved to {output_path / 'cluster_info.json'}")
        print(f"Detailed statistics saved to {output_path / 'clustering_stats.json'}")
        
        return cluster_info
    
    def run_clustering(self, input_folder: str, output_dir: str = "clustering_results",
                      min_cluster_size: int = 15, min_samples: int = 5,
                      samples_per_cluster: int = 8, debug: bool = False,
                      timeout_seconds: int = 30, batch_size: int = 100):
        """
        Run the complete clustering pipeline with batched processing.
        
        Args:
            input_folder: Folder containing images
            output_dir: Output directory for results
            min_cluster_size: Minimum cluster size for HDBSCAN (increased default for less noise)
            min_samples: Minimum samples for HDBSCAN (increased default for less noise)
            samples_per_cluster: Number of samples to save per cluster
            debug: Enable debug output
            timeout_seconds: Timeout for loading each image
            batch_size: Number of images to process in each batch
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get all image paths
        image_paths = self.get_image_paths(input_folder)
        
        if len(image_paths) == 0:
            print("No images found!")
            return
        
        print(f"Processing {len(image_paths)} images in batches of {batch_size}")
        print(f"Note: Batching is only for memory efficiency during embedding extraction.")
        print(f"All embeddings will be processed together for consistent clustering.")
        
        # Extract embeddings in batches
        successful_paths, embeddings = self.extract_embeddings_batched(
            image_paths, batch_size=batch_size, timeout_seconds=timeout_seconds, debug=debug
        )
        
        if len(embeddings) == 0:
            print("No embeddings extracted!")
            return
        
        print(f"\nSuccessfully extracted {len(embeddings)} embeddings")
        print(f"Embedding dimensions: {embeddings.shape}")
        
        # Save embeddings and paths
        np.save(output_path / "embeddings.npy", embeddings)
        with open(output_path / "image_paths.json", "w") as f:
            json.dump(successful_paths, f, indent=2)
        
        # Cluster embeddings (all at once for consistency)
        print(f"\nClustering {len(embeddings)} embeddings...")
        print(f"Using parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        
        clusterer, cluster_labels = self.cluster_embeddings(
            embeddings, min_cluster_size, min_samples
        )
        
        # Save cluster labels
        np.save(output_path / "cluster_labels.npy", cluster_labels)
        
        # Create visualization
        self.visualize_clusters(embeddings, cluster_labels, 
                               output_path / "cluster_visualization.png")
        
        # Save cluster samples and complete lists
        cluster_info = self.save_cluster_samples(successful_paths, cluster_labels,
                                                output_path / "samples", samples_per_cluster)
        
        # Print summary
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        noise_percentage = (n_noise / len(cluster_labels)) * 100
        
        print(f"\nFinal Clustering Summary:")
        print(f"=" * 50)
        print(f"Total images processed: {len(successful_paths)}")
        print(f"Clusters found: {n_clusters}")
        print(f"Noise points: {n_noise} ({noise_percentage:.1f}%)")
        print(f"Largest clusters:")
        
        # Show top clusters
        if cluster_info:
            top_clusters = sorted([
                (name, info["size"]) for name, info in cluster_info.items() 
                if name != "noise"
            ], key=lambda x: x[1], reverse=True)[:10]
            
            for i, (cluster_name, size) in enumerate(top_clusters, 1):
                percentage = (size / len(successful_paths)) * 100
                print(f"  {i:2d}. {cluster_name:<15}: {size:5d} images ({percentage:.1f}%)")
        
        print(f"Results saved to: {output_path}")
        
        return cluster_info


def main():
    parser = argparse.ArgumentParser(description="Cluster images using CLIP embeddings and HDBSCAN")
    parser.add_argument("input_folder", help="Folder containing images to cluster")
    parser.add_argument("--output-dir", default="clustering_results", 
                       help="Output directory for results")
    parser.add_argument("--min-cluster-size", type=int, default=15,
                       help="Minimum cluster size for HDBSCAN (default: 15, larger = less noise)")
    parser.add_argument("--min-samples", type=int, default=5,
                       help="Minimum samples for HDBSCAN (default: 5, larger = less noise)")
    parser.add_argument("--samples-per-cluster", type=int, default=8,
                       help="Number of samples to save per cluster (default: 8)")
    parser.add_argument("--model", default="ViT-B-32",
                       help="CLIP model to use")
    parser.add_argument("--pretrained", default="openai",
                       help="Pretrained weights to use")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output for troubleshooting")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Timeout in seconds for loading each image")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of images to process in each batch (lower = less memory usage)")
    
    args = parser.parse_args()
    
    print("Image Clustering with CLIP + HDBSCAN")
    print("=" * 40)
    print(f"Input folder: {args.input_folder}")
    print(f"Output directory: {args.output_dir}")
    print(f"Clustering parameters:")
    print(f"  - Min cluster size: {args.min_cluster_size}")
    print(f"  - Min samples: {args.min_samples}")
    print(f"  - Samples per cluster: {args.samples_per_cluster}")
    print(f"Processing parameters:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Timeout: {args.timeout}s")
    print(f"  - Debug mode: {args.debug}")
    print("=" * 40)
    
    # Initialize clusterer
    clusterer = ImageClusterer(model_name=args.model, pretrained=args.pretrained)
    
    # Run clustering
    cluster_info = clusterer.run_clustering(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        samples_per_cluster=args.samples_per_cluster,
        debug=args.debug,
        timeout_seconds=args.timeout,
        batch_size=args.batch_size
    )
    
    if cluster_info:
        print(f"\nSuccess! Check the output directory for:")
        print(f"  - Sample images from each cluster")
        print(f"  - Complete image lists (all_images.json in each cluster folder)")
        print(f"  - Clustering statistics (clustering_stats.json)")
        print(f"  - Cluster visualization")
    else:
        print("Clustering completed but no cluster info was returned.")


if __name__ == "__main__":
    main()
