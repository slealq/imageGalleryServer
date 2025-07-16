#!/usr/bin/env python3
"""
Image Clustering Script using CLIP embeddings and HDBSCAN

This script:
1. Loads images from a specified folder
2. Extracts CLIP embeddings using OpenCLIP
3. Clusters images using HDBSCAN
4. Saves sample images from each cluster
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torch
import open_clip
import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse


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
        
    def load_images(self, folder_path: str) -> Tuple[List[str], List[Image.Image]]:
        """
        Load all images from the specified folder.
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Tuple of (image_paths, images)
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder {folder_path} does not exist")
            
        image_paths = []
        images = []
        
        print(f"Loading images from {folder_path}...")
        
        for file_path in tqdm(list(folder.glob("*"))):
            if file_path.suffix.lower() in self.image_extensions:
                try:
                    image = Image.open(file_path).convert('RGB')
                    image_paths.append(str(file_path))
                    images.append(image)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
        print(f"Loaded {len(images)} images")
        return image_paths, images
    
    def extract_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract CLIP embeddings from images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Array of embeddings
        """
        print("Extracting CLIP embeddings...")
        
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(images), batch_size)):
            batch = images[i:i + batch_size]
            
            # Preprocess images
            processed_images = torch.stack([
                self.preprocess(img) for img in batch
            ]).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(processed_images)
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)
                
        return np.array(embeddings)
    
    def cluster_embeddings(self, embeddings: np.ndarray, 
                          min_cluster_size: int = 5,
                          min_samples: int = 3) -> Tuple[hdbscan.HDBSCAN, np.ndarray]:
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
        
        # Normalize embeddings
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Perform clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='cosine',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings_normalized)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Found {n_clusters} clusters with {n_noise} noise points")
        
        return clusterer, cluster_labels
    
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
        
        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = set(cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = cluster_labels == label
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
                           samples_per_cluster: int = 5):
        """
        Save sample images from each cluster.
        
        Args:
            image_paths: List of image file paths
            cluster_labels: Cluster labels
            output_dir: Directory to save samples
            samples_per_cluster: Number of samples to save per cluster
        """
        print(f"Saving cluster samples to {output_dir}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Group images by cluster
        cluster_images = {}
        for i, label in enumerate(cluster_labels):
            if label not in cluster_images:
                cluster_images[label] = []
            cluster_images[label].append(image_paths[i])
        
        # Save samples from each cluster
        cluster_info = {}
        
        for label, images in cluster_images.items():
            if label == -1:
                cluster_name = "noise"
            else:
                cluster_name = f"cluster_{label}"
            
            cluster_dir = output_path / cluster_name
            cluster_dir.mkdir(exist_ok=True)
            
            # Select samples (randomly if more than needed)
            n_samples = min(samples_per_cluster, len(images))
            selected_images = np.random.choice(images, n_samples, replace=False)
            
            # Copy sample images
            for i, img_path in enumerate(selected_images):
                img_name = Path(img_path).name
                dest_path = cluster_dir / f"sample_{i+1}_{img_name}"
                shutil.copy2(img_path, dest_path)
            
            cluster_info[cluster_name] = {
                "size": len(images),
                "samples": [str(dest_path) for dest_path in 
                           [cluster_dir / f"sample_{i+1}_{Path(img_path).name}" 
                            for i, img_path in enumerate(selected_images)]]
            }
            
            print(f"{cluster_name}: {len(images)} images, saved {n_samples} samples")
        
        # Save cluster information
        with open(output_path / "cluster_info.json", "w") as f:
            json.dump(cluster_info, f, indent=2)
        
        print(f"Cluster information saved to {output_path / 'cluster_info.json'}")
    
    def run_clustering(self, input_folder: str, output_dir: str = "clustering_results",
                      min_cluster_size: int = 5, min_samples: int = 3,
                      samples_per_cluster: int = 5):
        """
        Run the complete clustering pipeline.
        
        Args:
            input_folder: Folder containing images
            output_dir: Output directory for results
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            samples_per_cluster: Number of samples to save per cluster
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load images
        image_paths, images = self.load_images(input_folder)
        
        if len(images) == 0:
            print("No images found!")
            return
        
        # Extract embeddings
        embeddings = self.extract_embeddings(images)
        
        # Cluster embeddings
        clusterer, cluster_labels = self.cluster_embeddings(
            embeddings, min_cluster_size, min_samples
        )
        
        # Save embeddings and labels
        np.save(output_path / "embeddings.npy", embeddings)
        np.save(output_path / "cluster_labels.npy", cluster_labels)
        
        # Create visualization
        self.visualize_clusters(embeddings, cluster_labels, 
                               output_path / "cluster_visualization.png")
        
        # Save cluster samples
        self.save_cluster_samples(image_paths, cluster_labels,
                                output_path / "samples", samples_per_cluster)
        
        # Print summary
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"\nClustering Summary:")
        print(f"Total images: {len(images)}")
        print(f"Clusters found: {n_clusters}")
        print(f"Noise points: {n_noise}")
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster images using CLIP embeddings and HDBSCAN")
    parser.add_argument("input_folder", help="Folder containing images to cluster")
    parser.add_argument("--output-dir", default="clustering_results", 
                       help="Output directory for results")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                       help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min-samples", type=int, default=3,
                       help="Minimum samples for HDBSCAN")
    parser.add_argument("--samples-per-cluster", type=int, default=5,
                       help="Number of samples to save per cluster")
    parser.add_argument("--model", default="ViT-B-32",
                       help="CLIP model to use")
    parser.add_argument("--pretrained", default="openai",
                       help="Pretrained weights to use")
    
    args = parser.parse_args()
    
    # Initialize clusterer
    clusterer = ImageClusterer(model_name=args.model, pretrained=args.pretrained)
    
    # Run clustering
    clusterer.run_clustering(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        samples_per_cluster=args.samples_per_cluster
    )


if __name__ == "__main__":
    main()
