import os
import shutil
import argparse
from pathlib import Path
import logging
from typing import List

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def find_cropped_images(source_dir: Path) -> List[Path]:
    """
    Find all images that contain '_crop_' in their filename
    
    Args:
        source_dir: Source directory to search in
        
    Returns:
        List of Path objects for cropped images
    """
    cropped_images = []
    for file in source_dir.glob('*'):
        if file.is_file() and '_crop_' in file.name:
            cropped_images.append(file)
    return cropped_images

def move_cropped_images(source_dir: Path, target_dir: Path) -> tuple[int, int]:
    """
    Move all cropped images from source to target directory
    
    Args:
        source_dir: Source directory containing the images
        target_dir: Target directory to move images to
        
    Returns:
        Tuple of (successful_moves, failed_moves)
    """
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all cropped images
    cropped_images = find_cropped_images(source_dir)
    logging.info(f"Found {len(cropped_images)} cropped images")
    
    successful_moves = 0
    failed_moves = 0
    
    # Move each cropped image
    for image_path in cropped_images:
        try:
            target_path = target_dir / image_path.name
            shutil.move(str(image_path), str(target_path))
            logging.info(f"Moved {image_path.name} to {target_dir}")
            successful_moves += 1
        except Exception as e:
            logging.error(f"Failed to move {image_path.name}: {str(e)}")
            failed_moves += 1
    
    return successful_moves, failed_moves

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Move cropped images to a target directory')
    parser.add_argument('--source_dir', type=str, help='Source directory containing the images')
    parser.add_argument('--target_dir', type=str, help='Target directory to move cropped images to')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert to Path objects
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    
    # Validate source directory
    if not source_dir.exists():
        logging.error(f"Source directory does not exist: {source_dir}")
        return
    
    # Set up logging
    setup_logging()
    
    # Log start of operation
    logging.info(f"Starting to move cropped images from {source_dir} to {target_dir}")
    
    # Move the images
    successful_moves, failed_moves = move_cropped_images(source_dir, target_dir)
    
    # Log results
    logging.info(f"Operation completed:")
    logging.info(f"Successfully moved: {successful_moves} images")
    logging.info(f"Failed to move: {failed_moves} images")

if __name__ == "__main__":
    main()
