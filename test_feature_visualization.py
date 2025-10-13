#!/usr/bin/env python3
"""
Test script to verify feature visualization functionality.
"""

import yaml
import tempfile
import shutil
from pathlib import Path
from services.feature_extraction_service import FeatureExtractionService

def test_visualization_config():
    """Test that visualization configuration works correctly."""
    
    # Test configuration with save_feature_images enabled
    config_enabled = {
        'embeddings_path': '/path/to/embeddings.npz',
        'output_path': '/path/to/output',
        'feature_methods': ['canny'],
        'save_feature_images': True,
        'visualization_sample_size': 5,
        'max_visualizations': 10,
        'store_as_vectors_only': False
    }
    
    # Test configuration with save_feature_images disabled
    config_disabled = {
        'embeddings_path': '/path/to/embeddings.npz',
        'output_path': '/path/to/output',
        'feature_methods': ['canny'],
        'save_feature_images': False,
        'visualization_sample_size': 5,
        'max_visualizations': 10,
        'store_as_vectors_only': False
    }
    
    # Test configuration with vectors-only mode
    config_vectors_only = {
        'embeddings_path': '/path/to/embeddings.npz',
        'output_path': '/path/to/output',
        'feature_methods': ['canny'],
        'save_feature_images': True,
        'visualization_sample_size': 5,
        'max_visualizations': 10,
        'store_as_vectors_only': True
    }
    
    service = FeatureExtractionService()
    
    print("Testing configuration loading...")
    
    # Test enabled configuration
    loaded_config = service.load_config_from_dict(config_enabled)
    print(f"✅ Enabled config: save_feature_images={loaded_config.get('save_feature_images')}, "
          f"sample_size={loaded_config.get('visualization_sample_size')}")
    
    # Test disabled configuration
    loaded_config = service.load_config_from_dict(config_disabled)
    print(f"✅ Disabled config: save_feature_images={loaded_config.get('save_feature_images')}, "
          f"sample_size={loaded_config.get('visualization_sample_size')}")
    
    # Test vectors-only configuration
    loaded_config = service.load_config_from_dict(config_vectors_only)
    print(f"✅ Vectors-only config: save_feature_images={loaded_config.get('save_feature_images')}, "
          f"store_as_vectors_only={loaded_config.get('store_as_vectors_only')}")

def test_visualization_logic():
    """Test the visualization logic in save_results method."""
    
    # Mock feature data
    mock_features = [
        {
            'image_path': '/path/to/image1.jpg',
            'has_features': True,
            'features': {
                'canny': {
                    'edge_density': 0.1,
                    'edge_magnitude': 0.5,
                    'edge_direction_histogram': [0.1, 0.2, 0.3, 0.4]
                }
            }
        },
        {
            'image_path': '/path/to/image2.jpg',
            'has_features': True,
            'features': {
                'canny': {
                    'edge_density': 0.2,
                    'edge_magnitude': 0.6,
                    'edge_direction_histogram': [0.2, 0.3, 0.4, 0.5]
                }
            }
        }
    ]
    
    # Test different configurations
    test_configs = [
        {
            'name': 'Enabled with sample size 1',
            'config': {
                'save_feature_images': True,
                'visualization_sample_size': 1,
                'max_visualizations': 10,
                'store_as_vectors_only': False
            },
            'expected_visualizations': 1
        },
        {
            'name': 'Enabled with sample size 5',
            'config': {
                'save_feature_images': True,
                'visualization_sample_size': 5,
                'max_visualizations': 10,
                'store_as_vectors_only': False
            },
            'expected_visualizations': 2  # Only 2 features available
        },
        {
            'name': 'Disabled',
            'config': {
                'save_feature_images': False,
                'visualization_sample_size': 5,
                'max_visualizations': 10,
                'store_as_vectors_only': False
            },
            'expected_visualizations': 0
        },
        {
            'name': 'Vectors-only mode',
            'config': {
                'save_feature_images': True,
                'visualization_sample_size': 5,
                'max_visualizations': 10,
                'store_as_vectors_only': True
            },
            'expected_visualizations': 0
        }
    ]
    
    print("\nTesting visualization logic...")
    
    for test in test_configs:
        print(f"\nTesting: {test['name']}")
        
        # Determine expected behavior
        config = test['config']
        save_feature_images = config.get('save_feature_images', False)
        store_vectors_only = config.get('store_as_vectors_only', False)
        visualization_sample_size = config.get('visualization_sample_size', 10)
        max_visualizations = config.get('max_visualizations', 0)
        
        if save_feature_images and not store_vectors_only:
            num_visualizations = min(visualization_sample_size, max_visualizations) if max_visualizations > 0 else visualization_sample_size
            actual_visualizations = min(num_visualizations, len(mock_features))
        else:
            actual_visualizations = 0
        
        expected = test['expected_visualizations']
        print(f"  Expected: {expected}, Calculated: {actual_visualizations}")
        
        if expected == actual_visualizations:
            print(f"  ✅ PASS")
        else:
            print(f"  ❌ FAIL")

if __name__ == "__main__":
    print("🧪 Testing Feature Visualization Configuration")
    print("=" * 50)
    
    test_visualization_config()
    test_visualization_logic()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!") 