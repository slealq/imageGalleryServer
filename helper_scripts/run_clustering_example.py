#!/usr/bin/env python3
"""
Example script to run image clustering on the images folder.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the categorization module
sys.path.append(str(Path(__file__).parent.parent))

from helper_scripts.categorization import ImageClusterer

def main():
    # Path to the images folder (relative to the project root)
    images_folder = Path(__file__).parent.parent.parent / "images"
    
    if not images_folder.exists():
        print(f"Images folder not found at {images_folder}")
        print("Please make sure the images folder exists and contains image files.")
        return
    
    # Initialize the clusterer
    clusterer = ImageClusterer(
        model_name="ViT-B-32",  # You can try different models like "ViT-L-14", "ViT-H-14"
        pretrained="openai"
    )
    
    # Run clustering
    clusterer.run_clustering(
        input_folder=str(images_folder),
        output_dir="clustering_results",
        min_cluster_size=3,  # Adjust based on your dataset size
        min_samples=2,        # Adjust based on your dataset size
        samples_per_cluster=3 # Number of sample images to save per cluster
    )

if __name__ == "__main__":
    main() 