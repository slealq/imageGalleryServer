#!/usr/bin/env python3
"""
Test script for Canny Edge Clustering functionality

This script demonstrates the new Canny edge-based clustering features
added to the clustering service.
"""

import sys
from pathlib import Path

# Add services directory to path
sys.path.append(str(Path(__file__).parent / "services"))

from clustering_service import ClusteringService


def test_canny_clustering():
    """Test Canny edge-only clustering."""
    print("🧪 Testing Canny Edge Clustering")
    print("=" * 50)
    
    # Create clustering service
    service = ClusteringService()
    
    # Use the canny test configuration
    config_path = "services/canny_clustering_test_config.yaml"
    
    try:
        print(f"📋 Loading configuration from {config_path}")
        
        # Run Canny edge clustering
        results_path = service.cluster_images(config_path)
        
        print(f"✅ Canny edge clustering completed successfully!")
        print(f"📁 Results saved to: {results_path}")
        print(f"🔍 Check the results for clusters based on edge patterns and textures")
        
    except Exception as e:
        print(f"❌ Error during Canny clustering: {e}")
        print(f"💡 Make sure you have:")
        print(f"   1. Feature extraction results with Canny features")
        print(f"   2. Correct paths in the configuration file")
        print(f"   3. Required dependencies (FAISS, numpy, etc.)")


def test_canny_embedding_clustering():
    """Test Canny + embedding hybrid clustering."""
    print("\n🧪 Testing Canny + Embedding Hybrid Clustering")
    print("=" * 50)
    
    # Create clustering service
    service = ClusteringService()
    
    # Create a temporary config for canny_embedding
    config_content = """
# Canny + Embedding Test Configuration
embeddings_path: "/mnt/d/TEST/similarity_engine_results/results/embeddings/20250721_135402_8a9ef90e/embeddings.npz"
feature_data_path: "/mnt/d/TEST/similarity_engine_results/results/feature_extraction/20250722_192749_17028669/feature_vectors.npz"
output_path: "/mnt/d/TEST/similarity_engine_results/"

clustering_method: "faiss_similarity"
feature_type: "canny_embedding"
canny_weight: 0.4
embedding_weight: 0.6
similarity_threshold: 0.75
min_cluster_size: 10
max_examples_per_cluster: 25
generate_cluster_collages: true
generate_visualizations: true
"""
    
    config_path = "services/temp_canny_embedding_config.yaml"
    
    try:
        # Write temporary config
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"📋 Running hybrid Canny + embedding clustering")
        
        # Run hybrid clustering
        results_path = service.cluster_images(config_path)
        
        print(f"✅ Canny + embedding clustering completed successfully!")
        print(f"📁 Results saved to: {results_path}")
        print(f"🔍 Check the results for clusters combining edge and semantic similarity")
        
        # Clean up
        Path(config_path).unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Error during Canny + embedding clustering: {e}")
        print(f"💡 Make sure you have both:")
        print(f"   1. Feature extraction results with Canny features")
        print(f"   2. Embedding results from the embedding service")
        # Clean up on error
        Path(config_path).unlink(missing_ok=True)


def print_usage_guide():
    """Print usage guide for the new Canny clustering features."""
    print("\n📖 Canny Edge Clustering Usage Guide")
    print("=" * 50)
    print()
    print("🎯 New Feature Types Added:")
    print("   • 'canny' - Clusters based on Canny edge features only")
    print("   • 'canny_embedding' - Combines Canny edges with semantic embeddings")
    print()
    print("📋 Required Data:")
    print("   • For 'canny': feature_data_path (from feature extraction service)")
    print("   • For 'canny_embedding': feature_data_path + embeddings_path")
    print()
    print("⚙️  Key Configuration Parameters:")
    print("   • canny_weight: Weight for edge features (0.0-1.0)")
    print("   • embedding_weight: Weight for semantic features (0.0-1.0)")
    print("   • similarity_threshold: Clustering threshold (0.6-0.9)")
    print()
    print("🎨 Best Use Cases:")
    print("   • Architectural images (building patterns, structures)")
    print("   • Texture analysis (fabric, materials, surfaces)")
    print("   • Pattern recognition (geometric shapes, designs)")
    print("   • Document clustering (text layout, borders)")
    print()
    print("📁 Configuration Files:")
    print("   • services/canny_clustering_test_config.yaml - Canny-only clustering")
    print("   • services/clustering_config_sample.yaml - Updated with new options")
    print()
    print("🚀 Quick Start:")
    print("   1. Run feature extraction service with 'canny' method")
    print("   2. Update config with feature_data_path")
    print("   3. Set feature_type to 'canny' or 'canny_embedding'")
    print("   4. Run clustering service with your config")


if __name__ == "__main__":
    print("🔬 Canny Edge Clustering Feature Test")
    print("=" * 50)
    
    # Print usage guide
    print_usage_guide()
    
    # Test both clustering modes
    test_canny_clustering()
    test_canny_embedding_clustering()
    
    print("\n🎉 Canny edge clustering tests completed!")
    print("   Check the output directories for clustering results.")
    print("   Look for clusters organized by edge patterns and textures.") 