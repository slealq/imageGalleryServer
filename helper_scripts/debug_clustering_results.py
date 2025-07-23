#!/usr/bin/env python3
"""
Debug clustering results to understand why target images might be missing.

Usage:
    python debug_clustering_results.py --target-image "your_image.jpg" --results-dir "/path/to/clustering_results"
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def debug_clustering_results(target_image: str, results_dir: Path):
    """Debug clustering results to understand target image presence."""
    
    print(f"🔍 Debugging clustering results for target: {target_image}")
    print(f"📁 Results directory: {results_dir}")
    print("=" * 70)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    clustering_dir = results_dir / "clustering_results"
    if not clustering_dir.exists():
        print(f"❌ Clustering results directory not found: {clustering_dir}")
        return
    
    # Load summary if available
    summary_path = clustering_dir / "clustering_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
        print(f"📊 Summary: {summary['total_experiments']} total experiments, {summary['top_results_saved']} saved")
        print(f"🔧 Config: {summary['config']['execution_mode']} mode")
        print()
    
    # Find all rank directories
    rank_dirs = list(clustering_dir.glob("rank_*"))
    rank_dirs.sort()
    
    if not rank_dirs:
        print("❌ No rank directories found")
        return
    
    print(f"📂 Found {len(rank_dirs)} rank directories")
    print()
    
    all_images_sets = []
    target_found_in = []
    target_noise_in = []
    
    for rank_dir in rank_dirs:
        rank_name = rank_dir.name
        print(f"🔍 Analyzing {rank_name}")
        
        try:
            # Load files
            with open(rank_dir / "clustering_info.json", "r") as f:
                info = json.load(f)
            
            with open(rank_dir / "image_to_cluster.json", "r") as f:
                image_to_cluster = json.load(f)
            
            with open(rank_dir / "cluster_to_images.json", "r") as f:
                cluster_to_images = json.load(f)
            
            # Basic stats
            total_images = len(image_to_cluster)
            total_clusters = len(cluster_to_images)
            noise_count = sum(1 for cluster_id in image_to_cluster.values() if cluster_id == -1)
            
            print(f"  📊 Algorithm: {info['algorithm']}")
            print(f"  📊 Score: {info['composite_score']:.3f}")
            print(f"  📊 Images: {total_images}, Clusters: {total_clusters}, Noise: {noise_count}")
            
            # Check target image
            if target_image in image_to_cluster:
                cluster_id = image_to_cluster[target_image]
                if cluster_id == -1:
                    print(f"  ⚠️ Target image found in NOISE cluster")
                    target_noise_in.append(rank_name)
                else:
                    cluster_size = len(cluster_to_images.get(str(cluster_id), []))
                    print(f"  ✅ Target image found in cluster {cluster_id} (size: {cluster_size})")
                    target_found_in.append((rank_name, cluster_id, cluster_size))
            else:
                print(f"  ❌ Target image NOT FOUND in this experiment")
                # Show some sample image names to help debug
                sample_names = list(image_to_cluster.keys())[:5]
                print(f"      Sample names: {sample_names}")
                
                # Check for similar names
                target_base = target_image[:20].lower()
                similar_names = [name for name in image_to_cluster.keys() if target_base in name.lower()]
                if similar_names:
                    print(f"      Similar names found: {similar_names[:3]}")
            
            all_images_sets.append(set(image_to_cluster.keys()))
            
        except Exception as e:
            print(f"  ❌ Error reading {rank_name}: {e}")
        
        print()
    
    # Cross-experiment analysis
    print("🔄 CROSS-EXPERIMENT ANALYSIS")
    print("-" * 50)
    
    if len(all_images_sets) > 1:
        # Find common images across all experiments
        common_images = set.intersection(*all_images_sets)
        print(f"📊 Images common to ALL experiments: {len(common_images)}")
        
        # Check if target is in common set
        if target_image in common_images:
            print(f"✅ Target image is in ALL experiments (but may be in noise clusters)")
        else:
            print(f"❌ Target image is NOT in all experiments")
            
            # Find which experiments have the target
            experiments_with_target = []
            for i, img_set in enumerate(all_images_sets):
                if target_image in img_set:
                    experiments_with_target.append(rank_dirs[i].name)
            
            print(f"   Target found in: {experiments_with_target}")
            print(f"   Target missing from: {[d.name for d in rank_dirs if d.name not in experiments_with_target]}")
        
        # Find image set sizes
        set_sizes = [len(s) for s in all_images_sets]
        print(f"📊 Image set sizes: min={min(set_sizes)}, max={max(set_sizes)}, avg={sum(set_sizes)/len(set_sizes):.1f}")
        
        if len(set(set_sizes)) > 1:
            print("⚠️ Different experiments have different numbers of images!")
            print("   This suggests sampling or filtering differences between algorithms")
    
    # Target image summary
    print("\n🎯 TARGET IMAGE SUMMARY")
    print("-" * 50)
    print(f"Target image: {target_image}")
    print(f"Found in {len(target_found_in)} experiments as clustered data")
    print(f"Found in {len(target_noise_in)} experiments as noise")
    print(f"Missing from {len(rank_dirs) - len(target_found_in) - len(target_noise_in)} experiments")
    
    if target_found_in:
        print("\nClustered occurrences:")
        for rank_name, cluster_id, cluster_size in target_found_in:
            print(f"  • {rank_name}: cluster {cluster_id} (size {cluster_size})")
    
    if target_noise_in:
        print(f"\nNoise occurrences: {', '.join(target_noise_in)}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 50)
    
    if len(target_found_in) == 0:
        print("❌ Target image not found in any clustered results")
        print("   • Check if target image exists in the dataset")
        print("   • Verify image name spelling and case")
        print("   • Consider if dataset sampling excluded the target")
        print("   • Try different clustering parameters")
    elif len(target_noise_in) > 0:
        print("⚠️ Target image classified as noise in some experiments")
        print("   • Consider adjusting clustering parameters (e.g., min_cluster_size)")
        print("   • Try different algorithms that are less sensitive to outliers")
    else:
        print("✅ Target image successfully clustered in multiple experiments")
        print("   • Use experiments where target was found for similarity search")


def main():
    parser = argparse.ArgumentParser(description="Debug clustering results")
    parser.add_argument("--target-image", required=True, help="Target image filename")
    parser.add_argument("--results-dir", required=True, help="Path to clustering results directory")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    debug_clustering_results(args.target_image, results_dir)


if __name__ == "__main__":
    main() 