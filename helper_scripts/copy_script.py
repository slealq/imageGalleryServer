#!/usr/bin/env python3
"""
Image Copy Script

Reads image paths from similarity_config_sample.yaml and copies them to a new directory.
Handles path conversion from Unix (/mnt/d/) to Windows (D:\) format.
"""

import os
import shutil
import yaml
from pathlib import Path
from typing import List
import sys

def convert_unix_to_windows_path(unix_path: str) -> str:
    """
    Convert Unix mount path to Windows path.
    /mnt/d/TEST/images/... -> D:\TEST\images\...
    """
    if unix_path.startswith('/mnt/d/'):
        # Remove /mnt/d/ and replace with D:\
        windows_path = 'D:' + unix_path[6:].replace('/', '\\')
        return windows_path
    elif unix_path.startswith('D:/') or unix_path.startswith('D:\\'):
        # Already Windows format, just normalize
        return str(Path(unix_path))
    else:
        # Assume it's already a relative/absolute Windows path
        return str(Path(unix_path))

def load_image_paths_from_yaml(yaml_file: str) -> List[str]:
    """
    Load target image paths from the YAML configuration file.
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        target_images = config.get('target_images', [])
        if not target_images:
            print("No target_images found in the YAML file!")
            return []
        
        print(f"Found {len(target_images)} image paths in {yaml_file}")
        return target_images
    
    except FileNotFoundError:
        print(f"Error: YAML file '{yaml_file}' not found!")
        return []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error reading YAML file: {e}")
        return []

def copy_images(source_paths: List[str], destination_dir: str):
    """
    Copy images from source paths to destination directory.
    """
    # Create destination directory if it doesn't exist
    dest_path = Path(destination_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Destination directory: {dest_path.absolute()}")
    
    successful_copies = 0
    failed_copies = 0
    skipped_files = 0
    
    print(f"\nStarting to copy {len(source_paths)} images...")
    print("-" * 60)
    
    for i, unix_path in enumerate(source_paths, 1):
        # Convert Unix path to Windows path
        source_path = convert_unix_to_windows_path(unix_path.strip())
        source_file = Path(source_path)
        
        # Skip if path looks incomplete (no extension or ends with '.')
        if not source_file.suffix or source_path.endswith('.'):
            print(f"[{i:3d}] SKIP: Incomplete path - {source_path}")
            skipped_files += 1
            continue
        
        # Check if source file exists
        if not source_file.exists():
            print(f"[{i:3d}] FAIL: File not found - {source_path}")
            failed_copies += 1
            continue
        
        # Create destination file path
        dest_file = dest_path / source_file.name
        
        # Check if destination file already exists
        if dest_file.exists():
            print(f"[{i:3d}] SKIP: Already exists - {source_file.name}")
            skipped_files += 1
            continue
        
        try:
            # Copy the file
            shutil.copy2(source_file, dest_file)
            print(f"[{i:3d}] OK:   Copied - {source_file.name}")
            successful_copies += 1
            
        except Exception as e:
            print(f"[{i:3d}] FAIL: Error copying {source_file.name} - {e}")
            failed_copies += 1
    
    # Print summary
    print("-" * 60)
    print(f"Copy operation completed!")
    print(f"  Successfully copied: {successful_copies} files")
    print(f"  Failed to copy:      {failed_copies} files")
    print(f"  Skipped:             {skipped_files} files")
    print(f"  Total processed:     {len(source_paths)} paths")
    
    if successful_copies > 0:
        print(f"\nImages copied to: {dest_path.absolute()}")

def main():
    """
    Main function to execute the copy operation.
    """
    # Configuration
    yaml_file = "services/similarity_config_sample.yaml"
    destination_directory = "D:/TEST/dataset_1"
    
    print("Image Copy Script")
    print("=" * 50)
    print(f"Source YAML: {yaml_file}")
    print(f"Destination: {destination_directory}")
    print()
    
    # Load image paths from YAML
    image_paths = load_image_paths_from_yaml(yaml_file)
    
    if not image_paths:
        print("No images to copy. Exiting.")
        return
    
    # Ask for confirmation
    print(f"\nAbout to copy {len(image_paths)} images to {destination_directory}")
    response = input("Continue? (y/N): ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Copy the images
    copy_images(image_paths, destination_directory)

if __name__ == "__main__":
    main()
