#!/usr/bin/env python3
"""
Find Similar Images Using Clustering Results

This script uses the clustering results from unified_clustering.py to find images
similar to a given input image based on cluster membership across different experiments.

Usage:
    python find_similar_images.py <image_name> [options]

Features:
- Searches across all clustering experiments
- Finds images in the same clusters as the input image
- Organizes results by experiment and cluster
- Provides similarity scoring based on multiple experiments
- Copies similar images to organized output folders

Examples:
    python find_similar_images.py "IMG_1234.jpg"
    python find_similar_images.py "sunset.png" --output-dir my_similar_images
    python find_similar_images.py "portrait.jpg" --top-experiments 5 --max-per-cluster 20
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
import time


class SimilarityFinder:
    """Find similar images using clustering results."""
    
    def __init__(self, results_dir: str, image_source_dir: str = None):
        self.results_dir = Path(results_dir)
        self.image_source_dir = Path(image_source_dir) if image_source_dir else None
        self.experiments = []
        self.image_paths_mapping = {}
        
        # Load available experiments
        self._load_experiments()
        
        # Try to auto-detect image source directory if not provided
        if not self.image_source_dir:
            self._auto_detect_image_source()
    
    def _load_experiments(self):
        """Load information about available clustering experiments."""
        print(f"🔍 Loading experiments from {self.results_dir}")
        
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")
        
        # Find all experiment directories (rank_XX_algorithm format)
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("rank_"):
                # Load experiment info
                info_file = exp_dir / "clustering_info.json"
                image_to_cluster_file = exp_dir / "image_to_cluster.json"
                cluster_to_images_file = exp_dir / "cluster_to_images.json"
                
                if info_file.exists() and image_to_cluster_file.exists() and cluster_to_images_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    with open(image_to_cluster_file, 'r') as f:
                        image_to_cluster = json.load(f)
                    
                    with open(cluster_to_images_file, 'r') as f:
                        cluster_to_images = json.load(f)
                    
                    # Convert cluster keys to integers
                    cluster_to_images = {int(k): v for k, v in cluster_to_images.items()}
                    
                    experiment = {
                        'name': exp_dir.name,
                        'algorithm': info['algorithm'],
                        'n_clusters': info['n_clusters'],
                        'composite_score': info['composite_score'],
                        'image_to_cluster': image_to_cluster,
                        'cluster_to_images': cluster_to_images,
                        'directory': exp_dir
                    }
                    
                    self.experiments.append(experiment)
        
        # Sort experiments by composite score (best first)
        self.experiments.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"✅ Loaded {len(self.experiments)} clustering experiments")
        if self.experiments:
            print("Top experiments:")
            for i, exp in enumerate(self.experiments[:5], 1):
                print(f"  {i}. {exp['name']} ({exp['algorithm']}) - Score: {exp['composite_score']:.3f}, Clusters: {exp['n_clusters']}")
    
    def _auto_detect_image_source(self):
        """Try to auto-detect the image source directory."""
        # Check common image directories
        potential_dirs = [
            Path.cwd() / "images",
            Path.cwd() / "test_images", 
            Path.cwd() / "source_images",
            Path.cwd(),
            Path.cwd().parent / "images",
        ]
        
        # Also check if there's a reference in the clustering summary
        summary_file = self.results_dir / "clustering_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Try to extract image directory from image_paths_file
            image_paths_file = summary.get('image_paths_file')
            if image_paths_file and Path(image_paths_file).exists():
                with open(image_paths_file, 'r') as f:
                    image_paths = json.load(f)
                
                if image_paths:
                    # Get directory from first image path
                    first_image_dir = Path(image_paths[0]).parent
                    if first_image_dir.exists():
                        potential_dirs.insert(0, first_image_dir)
        
        # Find directory with most image files
        best_dir = None
        max_images = 0
        
        for dir_path in potential_dirs:
            if dir_path.exists() and dir_path.is_dir():
                image_count = sum(1 for f in dir_path.rglob("*") 
                                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
                
                if image_count > max_images:
                    max_images = image_count
                    best_dir = dir_path
        
        if best_dir and max_images > 0:
            self.image_source_dir = best_dir
            print(f"📁 Auto-detected image source: {self.image_source_dir} ({max_images} images)")
        else:
            print("⚠️  Could not auto-detect image source directory")
    
    def find_image_file(self, image_name: str) -> Optional[Path]:
        """Find the full path of an image file by name."""
        if not self.image_source_dir:
            return None
        
        # Try exact match first
        exact_path = self.image_source_dir / image_name
        if exact_path.exists():
            return exact_path
        
        # Search recursively for the image
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            # Try with different extensions
            name_base = Path(image_name).stem
            for pattern in [f"{name_base}{ext}", f"{name_base}{ext.upper()}"]:
                matches = list(self.image_source_dir.rglob(pattern))
                if matches:
                    return matches[0]
        
        # Try partial match
        matches = list(self.image_source_dir.rglob(f"*{image_name}*"))
        if matches:
            return matches[0]
        
        return None
    
    def find_similar_images(self, target_image: str, max_experiments: int = None, 
                          max_per_cluster: int = 50) -> Dict[str, Dict]:
        """Find images similar to the target image across clustering experiments."""
        print(f"\n🎯 Finding images similar to: {target_image}")
        
        if max_experiments is None:
            max_experiments = len(self.experiments)
        
        similar_images = defaultdict(dict)
        experiments_found = 0
        total_similar_count = 0
        
        # Track which images appear similar across multiple experiments
        image_experiment_count = Counter()
        
        for i, experiment in enumerate(self.experiments[:max_experiments]):
            exp_name = experiment['name']
            image_to_cluster = experiment['image_to_cluster']
            cluster_to_images = experiment['cluster_to_images']
            
            # Check if target image exists in this experiment
            if target_image not in image_to_cluster:
                continue
            
            cluster_id = image_to_cluster[target_image]
            
            # Skip if image was classified as noise
            if cluster_id == -1:
                print(f"  ⚠️  {exp_name}: Image classified as noise, skipping")
                continue
            
            # Get all images in the same cluster
            cluster_images = cluster_to_images.get(cluster_id, [])
            
            # Remove the target image itself from results
            similar_in_cluster = [img for img in cluster_images if img != target_image]
            
            if not similar_in_cluster:
                continue
            
            # Limit number of images per cluster if requested
            if len(similar_in_cluster) > max_per_cluster:
                similar_in_cluster = similar_in_cluster[:max_per_cluster]
            
            similar_images[exp_name] = {
                'algorithm': experiment['algorithm'],
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_images),
                'similar_images': similar_in_cluster,
                'composite_score': experiment['composite_score']
            }
            
            # Count how many experiments each image appears in
            for img in similar_in_cluster:
                image_experiment_count[img] += 1
            
            experiments_found += 1
            total_similar_count += len(similar_in_cluster)
            
            print(f"  ✅ {exp_name}: Found {len(similar_in_cluster)} similar images in cluster {cluster_id} (size: {len(cluster_images)})")
        
        if experiments_found == 0:
            print(f"  ❌ Image '{target_image}' not found in any clustering experiments")
            return {}
        
        print(f"\n📊 Summary:")
        print(f"  • Found similar images in {experiments_found} experiments")
        print(f"  • Total similar images: {total_similar_count}")
        print(f"  • Unique similar images: {len(image_experiment_count)}")
        
        # Show most consistent similar images (appearing in multiple experiments)
        if len(image_experiment_count) > 0:
            top_consistent = image_experiment_count.most_common(10)
            print(f"  • Most consistent similar images:")
            for img, count in top_consistent:
                print(f"    - {img}: appears in {count} experiments")
        
        return dict(similar_images)
    
    def save_similar_images(self, target_image: str, similar_results: Dict[str, Dict], 
                          output_dir: str = "similar_images") -> None:
        """Save similar images to organized folders."""
        if not similar_results:
            print("❌ No similar images to save")
            return
        
        if not self.image_source_dir:
            print("❌ Image source directory not available for copying files")
            return
        
        # Create output directory structure
        target_name = Path(target_image).stem
        base_output_dir = Path(output_dir) / f"similar_to_{target_name}"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving similar images to: {base_output_dir}")
        
        # Copy target image to output directory
        target_file = self.find_image_file(target_image)
        if target_file:
            target_output = base_output_dir / f"TARGET_{target_image}"
            shutil.copy2(target_file, target_output)
            print(f"  📋 Copied target image: {target_output}")
        
        total_copied = 0
        
        # Process each experiment
        for exp_name, exp_data in similar_results.items():
            exp_output_dir = base_output_dir / exp_name
            exp_output_dir.mkdir(exist_ok=True)
            
            algorithm = exp_data['algorithm']
            cluster_id = exp_data['cluster_id']
            similar_images = exp_data['similar_images']
            
            copied_count = 0
            
            for i, similar_image in enumerate(similar_images):
                source_file = self.find_image_file(similar_image)
                
                if source_file and source_file.exists():
                    # Create descriptive filename
                    file_ext = source_file.suffix
                    dst_filename = f"{i+1:03d}_{similar_image}"
                    dst_path = exp_output_dir / dst_filename
                    
                    try:
                        shutil.copy2(source_file, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"    ⚠️  Failed to copy {similar_image}: {e}")
                else:
                    print(f"    ⚠️  Source file not found: {similar_image}")
            
            total_copied += copied_count
            
            # Save experiment info
            exp_info = {
                'target_image': target_image,
                'algorithm': algorithm,
                'cluster_id': cluster_id,
                'cluster_size': exp_data['cluster_size'],
                'composite_score': exp_data['composite_score'],
                'total_similar_found': len(similar_images),
                'images_copied': copied_count,
                'similar_images': similar_images
            }
            
            with open(exp_output_dir / "experiment_info.json", 'w') as f:
                json.dump(exp_info, f, indent=2)
            
            print(f"  📁 {exp_name}: {copied_count}/{len(similar_images)} images copied")
        
        # Create overall summary
        summary = {
            'target_image': target_image,
            'total_experiments': len(similar_results),
            'total_images_copied': total_copied,
            'output_directory': str(base_output_dir),
            'experiments': similar_results
        }
        
        with open(base_output_dir / "similarity_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 Similarity search completed!")
        print(f"  • Total images copied: {total_copied}")
        print(f"  • Results saved to: {base_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Find Similar Images Using Clustering Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_similar_images.py "IMG_1234.jpg"
  python find_similar_images.py "sunset.png" --output-dir my_results
  python find_similar_images.py "portrait.jpg" --top-experiments 5 --max-per-cluster 20
  python find_similar_images.py "landscape.jpg" --source-dir /path/to/images
        """
    )
    
    # Required arguments
    parser.add_argument("image_name", help="Name of the target image to find similar images for")
    
    # Optional arguments
    parser.add_argument("--results-dir", default="clustering_results",
                       help="Directory containing clustering results (default: clustering_results)")
    parser.add_argument("--source-dir", help="Directory containing source images (auto-detected if not provided)")
    parser.add_argument("--output-dir", default="similar_images",
                       help="Output directory for similar images (default: similar_images)")
    parser.add_argument("--top-experiments", type=int, 
                       help="Maximum number of top experiments to use (default: all)")
    parser.add_argument("--max-per-cluster", type=int, default=50,
                       help="Maximum similar images per cluster (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Find similar images but don't copy files")
    
    args = parser.parse_args()
    
    try:
        print("🔎 Similar Image Finder")
        print("=" * 50)
        
        # Initialize finder
        finder = SimilarityFinder(args.results_dir, args.source_dir)
        
        if not finder.experiments:
            print("❌ No clustering experiments found!")
            sys.exit(1)
        
        # Find similar images
        start_time = time.time()
        similar_results = finder.find_similar_images(
            args.image_name, 
            args.top_experiments, 
            args.max_per_cluster
        )
        
        if not similar_results:
            print(f"❌ No similar images found for '{args.image_name}'")
            print("\nPossible reasons:")
            print("  • Image name doesn't match any image in the clustering results")
            print("  • Image was classified as noise in all experiments")
            print("  • Clustering results don't contain the expected mapping files")
            sys.exit(1)
        
        # Save results unless dry run
        if not args.dry_run:
            finder.save_similar_images(args.image_name, similar_results, args.output_dir)
        else:
            print("\n🔍 Dry run completed - no files copied")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Completed in {elapsed_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 